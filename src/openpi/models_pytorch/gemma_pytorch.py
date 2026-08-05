from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


def _sdpa_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Convert OpenPI's additive mask into SDPA's native boolean representation.

    OpenPI supplies a binary additive mask (0 for allowed, a large negative
    value for blocked). SDPA's CUDA kernels impose stricter additive-bias dtype
    rules than the eager path, so preserve the same semantics with a boolean
    mask. The mask is constant across layers, so callers running a stack should
    convert once up front rather than per layer.
    """
    if attention_mask is None or attention_mask.dtype == torch.bool:
        return attention_mask
    return attention_mask >= 0


def _attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Run SDPA and return [batch, sequence, heads, head_dim]."""
    key_states = modeling_gemma.repeat_kv(key_states, module.num_key_value_groups)
    value_states = modeling_gemma.repeat_kv(value_states, module.num_key_value_groups)
    attention_mask = _sdpa_attention_mask(attention_mask)
    output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
    )
    return output.transpose(1, 2).contiguous()


class _BoundJointTransformerLayer(nn.Module):
    """One joint PaliGemma/expert layer with its modules bound at construction."""

    def __init__(
        self,
        prefix_layer: nn.Module,
        suffix_layer: nn.Module,
        rotary_emb: nn.Module,
    ) -> None:
        super().__init__()
        self.prefix_layer = prefix_layer
        self.suffix_layer = suffix_layer
        self.rotary_emb = rotary_emb

    def forward(self, inputs_embeds, attention_mask, position_ids, adarms_cond):
        layers = (self.prefix_layer, self.suffix_layer)
        prefix_length = inputs_embeds[0].shape[1]

        projections = []
        for index, residual in enumerate(inputs_embeds):
            layer = layers[index]
            hidden_states, gate = layer.input_layernorm(residual, cond=adarms_cond[index])
            hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
            query_states = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            projections.append((residual, gate, query_states, key_states, value_states))

        query_states = torch.cat([projection[2] for projection in projections], dim=2)
        key_states = torch.cat([projection[3] for projection in projections], dim=2)
        value_states = torch.cat([projection[4] for projection in projections], dim=2)
        dummy_tensor = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = self.rotary_emb(dummy_tensor, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )
        attention_module = self.prefix_layer.self_attn
        attention_output = _attention_forward(
            attention_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            attention_module.scaling,
        )
        attention_outputs = torch.split(
            attention_output,
            [prefix_length, inputs_embeds[1].shape[1]],
            dim=1,
        )

        outputs_embeds = []
        for index, (residual, gate, *_unused) in enumerate(projections):
            layer = layers[index]
            layer_attention = attention_outputs[index].reshape(residual.shape[0], residual.shape[1], -1)
            if layer_attention.dtype != layer.self_attn.o_proj.weight.dtype:
                layer_attention = layer_attention.to(layer.self_attn.o_proj.weight.dtype)
            out_emb = layer.self_attn.o_proj(layer_attention)
            out_emb = modeling_gemma._gated_residual(residual, out_emb, gate)  # noqa: SLF001
            after_first_residual = out_emb
            out_emb, mlp_gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[index])
            if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                out_emb = out_emb.to(dtype=torch.bfloat16)
            out_emb = layer.mlp(out_emb)
            outputs_embeds.append(
                modeling_gemma._gated_residual(after_first_residual, out_emb, mlp_gate)  # noqa: SLF001
            )

        return outputs_embeds


class PaliGemmaWithExpertModel(nn.Module):
    # Class-level default so instances built without ``__init__`` (tests) still
    # resolve the flag.
    _gradient_checkpointing = False

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # PaliGemmaConfig
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)  # vision lang encoder
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self._compile_training_layers = False
        self._final_norms_fn = self._compute_final_norms
        self._configure_bound_layer_fns(compile_layers=False)

        self.to_bfloat16_for_selected_params(precision)

    def _configure_bound_layer_fns(self, *, compile_layers: bool) -> None:
        prefix_model = self.paligemma.language_model
        suffix_model = self.gemma_expert.model
        if len(prefix_model.layers) != len(suffix_model.layers):
            raise ValueError(
                "PaliGemma and action expert must have the same transformer depth, got "
                f"{len(prefix_model.layers)} and {len(suffix_model.layers)}"
            )

        rotary_emb = self.paligemma.model.language_model.rotary_emb
        layer_fns = [
            _BoundJointTransformerLayer(prefix_layer, suffix_layer, rotary_emb)
            for prefix_layer, suffix_layer in zip(prefix_model.layers, suffix_model.layers, strict=True)
        ]
        if compile_layers:
            layer_fns = [
                torch.compile(layer_fn, mode="default", fullgraph=True, dynamic=False) for layer_fn in layer_fns
            ]

        # Keep this as a plain list rather than ModuleList. The bound units
        # reference parameters already registered by the two backbone models;
        # registering them again would duplicate state_dict paths.
        self._layer_fns = layer_fns

    def configure_training_compile(self, *, enabled: bool) -> None:
        """Compile stable, per-layer training units without capturing the full model forward."""
        self._compile_training_layers = enabled
        self._configure_bound_layer_fns(compile_layers=enabled)
        self._final_norms_fn = self._compute_final_norms
        if not enabled:
            return

        # Keeping the compile boundary at one transformer layer lets DDP start
        # reducing earlier-layer gradients while later layers are still in
        # backward. Static ABC-130K tensor shapes also avoid dynamic-shape
        # recompilations at this boundary.
        self._final_norms_fn = torch.compile(
            self._compute_final_norms,
            mode="default",
            fullgraph=True,
            dynamic=False,
        )

    def configure_gradient_checkpointing(self, *, enabled: bool) -> None:
        """Recompute each joint transformer layer in backward instead of storing it."""
        self._gradient_checkpointing = enabled

    def _run_layer(self, layer_fn, inputs_embeds, attention_mask, position_ids, adarms_cond):
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer_fn,
                inputs_embeds,
                attention_mask,
                position_ids,
                adarms_cond,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return layer_fn(inputs_embeds, attention_mask, position_ids, adarms_cond)

    def _compute_final_norms(self, inputs_embeds, adarms_cond):
        models = [self.paligemma.language_model, self.gemma_expert.model]
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
            outputs_embeds.append(out_emb)
        return outputs_embeds

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            # The mask is the same for every layer; converting it once keeps the
            # per-layer cost out of a stack that is 18 layers deep.
            attention_mask = _sdpa_attention_mask(attention_mask)
            for layer_fn in self._layer_fns:
                inputs_embeds = self._run_layer(layer_fn, inputs_embeds, attention_mask, position_ids, adarms_cond)

            outputs_embeds = self._final_norms_fn(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

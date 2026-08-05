from types import SimpleNamespace

import torch
from torch import nn
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch import gemma_pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


class _IdentityNorm(nn.Module):
    def forward(self, hidden_states, cond=None):
        del cond
        return hidden_states, None


class _TinyAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.head_dim = width // heads
        self.num_key_value_groups = 1
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)


class _TinyMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width)

    def forward(self, hidden_states):
        return self.down_proj(torch.nn.functional.silu(self.up_proj(hidden_states)))


class _TinyLayer(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.input_layernorm = _IdentityNorm()
        self.self_attn = _TinyAttention(width, heads)
        self.post_attention_layernorm = _IdentityNorm()
        self.mlp = _TinyMLP(width)


class _TinyRotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        shape = (*position_ids.shape, hidden_states.shape[-1])
        return torch.ones(shape, device=hidden_states.device), torch.zeros(shape, device=hidden_states.device)


class _TinyLanguageModel(nn.Module):
    def __init__(self, width: int, heads: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList([_TinyLayer(width, heads) for _ in range(layers)])
        self.norm = _IdentityNorm()


class _TinyPaliGemma(nn.Module):
    def __init__(self, width: int, heads: int, layers: int):
        super().__init__()
        self.language_model = _TinyLanguageModel(width, heads, layers)
        self.model = SimpleNamespace(language_model=SimpleNamespace(rotary_emb=_TinyRotary()))
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=layers))


class _TinyExpert(nn.Module):
    def __init__(self, width: int, heads: int, layers: int):
        super().__init__()
        self.model = _TinyLanguageModel(width, heads, layers)


def _tiny_paired_model(*, layers: int = 2):
    torch.manual_seed(7)
    model = gemma_pytorch.PaliGemmaWithExpertModel.__new__(gemma_pytorch.PaliGemmaWithExpertModel)
    nn.Module.__init__(model)
    model.paligemma = _TinyPaliGemma(width=4, heads=2, layers=layers)
    model.gemma_expert = _TinyExpert(width=4, heads=2, layers=layers)
    model._compile_training_layers = False  # noqa: SLF001
    model._final_norms_fn = model._compute_final_norms  # noqa: SLF001
    model._configure_bound_layer_fns(compile_layers=False)  # noqa: SLF001
    return model


def _prefix_lm_additive_mask(pad: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """Build the same additive mask PI0Pytorch._prepare_attention_masks_4d produces."""
    return torch.where(make_att_2d_masks(pad, block)[:, None], 0.0, -2.3819763e38)


def test_sdpa_matches_eager_attention():
    attention = _TinyAttention(width=4, heads=2)
    query = torch.randn(2, 2, 3, 2)
    key = torch.randn(2, 2, 3, 2)
    value = torch.randn(2, 2, 3, 2)
    mask = torch.ones((2, 1, 3, 3), dtype=torch.bool)

    actual = gemma_pytorch._attention_forward(  # noqa: SLF001
        attention, query, key, value, mask, attention.scaling
    )
    expected, _ = modeling_gemma.eager_attention_forward(attention, query, key, value, mask, attention.scaling)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_sdpa_matches_eager_on_the_production_additive_mask():
    """The additive -> boolean conversion must not change which positions are blocked.

    The all-allowed mask above cannot catch this: softmax is invariant to a
    constant added to every logit, so both paths agree no matter what the
    conversion does. This case mixes allowed and blocked positions, so inverting
    the ``attention_mask >= 0`` predicate makes it fail.
    """
    torch.manual_seed(3)
    attention = _TinyAttention(width=4, heads=2)
    query = torch.randn(2, 2, 6, 2)
    key = torch.randn(2, 2, 6, 2)
    value = torch.randn(2, 2, 6, 2)
    # Prefix-LM layout: 4 prefix tokens attend to each other, 2 suffix tokens
    # open a causal block, and the last position of row 1 is padding.
    pad = torch.tensor([[True] * 6, [True] * 5 + [False]])
    block = torch.tensor([[False, False, False, False, True, False]] * 2)
    additive_mask = _prefix_lm_additive_mask(pad, block)
    assert (additive_mask < 0).any()
    assert (additive_mask == 0).any()

    actual = gemma_pytorch._attention_forward(  # noqa: SLF001
        attention, query, key, value, additive_mask, attention.scaling
    )
    expected, _ = modeling_gemma.eager_attention_forward(attention, query, key, value, additive_mask, attention.scaling)

    assert not torch.isnan(actual).any()
    # Compare only rows that attend to something; a fully padded query row has no
    # defined value in either formulation and is discarded downstream.
    attends = make_att_2d_masks(pad, block).any(dim=-1)
    torch.testing.assert_close(actual[attends], expected[attends], rtol=1e-5, atol=1e-5)


def test_sdpa_blocks_exactly_the_positions_the_additive_mask_blocks():
    """Pin the conversion threshold: a blocked key must not leak into the output."""
    attention = _TinyAttention(width=4, heads=2)
    query = torch.randn(1, 2, 2, 2)
    key = torch.randn(1, 2, 2, 2)
    value = torch.tensor([[[[1.0, 1.0], [500.0, 500.0]], [[1.0, 1.0], [500.0, 500.0]]]])
    # Row 0 may only attend to key 0; row 1 may attend to both.
    additive_mask = torch.tensor([[[[0.0, -2.3819763e38], [0.0, 0.0]]]])

    output = gemma_pytorch._attention_forward(  # noqa: SLF001
        attention, query, key, value, additive_mask, attention.scaling
    )

    torch.testing.assert_close(output[0, 0], torch.ones(2, 2))
    assert (output[0, 1] > 1.0).all()


def test_sdpa_accepts_float32_mask_for_bfloat16_attention():
    attention = _TinyAttention(width=4, heads=2).to(torch.bfloat16)
    query = torch.randn(2, 2, 3, 2, dtype=torch.bfloat16)
    key = torch.randn(2, 2, 3, 2, dtype=torch.bfloat16)
    value = torch.randn(2, 2, 3, 2, dtype=torch.bfloat16)
    float32_mask = torch.zeros((2, 1, 3, 3), dtype=torch.float32)

    output = gemma_pytorch._attention_forward(  # noqa: SLF001
        attention, query, key, value, float32_mask, attention.scaling
    )

    assert output.dtype == torch.bfloat16


def test_joint_transformer_forward_has_expected_shape_and_gradients():
    model = _tiny_paired_model(layers=2)
    batch_size, prefix_length, suffix_length, width = 2, 3, 2, 4
    prefix = torch.randn(batch_size, prefix_length, width, requires_grad=True)
    suffix = torch.randn(batch_size, suffix_length, width, requires_grad=True)
    pad = torch.ones((batch_size, prefix_length + suffix_length), dtype=torch.bool)
    block = torch.zeros_like(pad)
    block[:, prefix_length] = True
    mask = make_att_2d_masks(pad, block)
    additive_mask = torch.where(mask[:, None], 0.0, -2.3819763e38)
    position_ids = torch.arange(prefix_length + suffix_length)[None].expand(batch_size, -1)

    outputs, _ = model(
        attention_mask=additive_mask,
        position_ids=position_ids,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        adarms_cond=[None, None],
    )
    outputs[1].sum().backward()

    assert outputs[0].shape == prefix.shape
    assert outputs[1].shape == suffix.shape
    assert prefix.grad is not None
    assert suffix.grad is not None


def test_training_compile_binds_each_layer_without_runtime_index(monkeypatch):
    model = _tiny_paired_model(layers=18)
    state_dict_keys = set(model.state_dict())
    compile_calls = []

    def fake_compile(function, **kwargs):
        compile_calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    model.configure_training_compile(enabled=True)

    assert set(model.state_dict()) == state_dict_keys
    layer_calls = compile_calls[:-1]
    assert len(layer_calls) == 18
    assert all(isinstance(function, gemma_pytorch._BoundJointTransformerLayer) for function, _ in layer_calls)  # noqa: SLF001
    for layer_idx, (function, _) in enumerate(layer_calls):
        assert function.prefix_layer is model.paligemma.language_model.layers[layer_idx]
        assert function.suffix_layer is model.gemma_expert.model.layers[layer_idx]
    assert compile_calls[-1][0].__name__ == "_compute_final_norms"
    assert all(kwargs == {"mode": "default", "fullgraph": True, "dynamic": False} for _, kwargs in compile_calls)


def test_gradient_checkpointing_is_off_by_default_and_recomputes_when_enabled(monkeypatch):
    model = _tiny_paired_model(layers=2)
    model.train()
    batch_size, prefix_length, suffix_length, width = 2, 3, 2, 4
    prefix = torch.randn(batch_size, prefix_length, width, requires_grad=True)
    suffix = torch.randn(batch_size, suffix_length, width, requires_grad=True)
    pad = torch.ones((batch_size, prefix_length + suffix_length), dtype=torch.bool)
    block = torch.zeros_like(pad)
    block[:, prefix_length] = True
    additive_mask = _prefix_lm_additive_mask(pad, block)
    position_ids = torch.arange(prefix_length + suffix_length)[None].expand(batch_size, -1)

    checkpoint_calls = []
    real_checkpoint = torch.utils.checkpoint.checkpoint

    def counting_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(function)
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting_checkpoint)

    def run():
        outputs, _ = model(
            attention_mask=additive_mask,
            position_ids=position_ids,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            adarms_cond=[None, None],
        )
        return outputs

    assert model._gradient_checkpointing is False  # noqa: SLF001
    baseline = run()
    assert checkpoint_calls == []

    model.configure_gradient_checkpointing(enabled=True)
    checkpointed = run()

    assert len(checkpoint_calls) == 2
    torch.testing.assert_close(checkpointed[1], baseline[1])

    checkpointed[1].sum().backward()
    assert prefix.grad is not None
    assert suffix.grad is not None

    # eval() must not recompute: checkpointing only pays off while building a graph.
    checkpoint_calls.clear()
    model.eval()
    run()
    assert checkpoint_calls == []

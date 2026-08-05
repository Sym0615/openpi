from types import SimpleNamespace

import torch
from torch import nn

from openpi.models_pytorch import pi0_pytorch


class _StubBackbone(nn.Module):
    """Stand in for PaliGemmaWithExpertModel's embedding entry points."""

    def __init__(self, width: int):
        super().__init__()
        # Mirrors the real model: the SigLIP patch embedding stays FP32.
        self.patch_embedding = nn.Conv2d(3, width, kernel_size=2, stride=2)
        self.token_embedding = nn.Embedding(7, width)
        self.gradient_checkpointing_flag = False
        self.last_image_embedding_dtype = None

    def embed_image(self, image):
        output = self.patch_embedding(image).flatten(2).transpose(1, 2)
        self.last_image_embedding_dtype = output.dtype
        return output

    def embed_language_tokens(self, tokens):
        return self.token_embedding(tokens)

    def configure_gradient_checkpointing(self, *, enabled: bool):
        self.gradient_checkpointing_flag = enabled


def _tiny_pi05_model(width: int = 4, action_dim: int = 3):
    torch.manual_seed(11)
    model = pi0_pytorch.PI0Pytorch.__new__(pi0_pytorch.PI0Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(action_horizon=2, action_dim=action_dim)
    model.pi05 = True
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = _StubBackbone(width)
    model.action_in_proj = nn.Linear(action_dim, width)
    model.action_out_proj = nn.Linear(width, action_dim)
    model.time_mlp_in = nn.Linear(width, width)
    model.time_mlp_out = nn.Linear(width, width)
    return model


def test_bf16_autocast_is_actually_active_in_this_environment():
    """Guards the two tests below: a plain module must downcast under autocast."""
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert nn.Linear(3, 4)(torch.randn(2, 3)).dtype == torch.bfloat16


def test_embed_suffix_keeps_its_fp32_projections_under_bf16_autocast():
    """OPENPI_BF16_AUTOCAST must not override the model's own precision policy."""
    model = _tiny_pi05_model()
    noisy_actions = torch.randn(2, 2, 3)
    timestep = torch.rand(2)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        embeddings, _, _, adarms_cond = model.embed_suffix(None, noisy_actions, timestep)

    assert embeddings.dtype == torch.float32
    assert adarms_cond.dtype == torch.float32


def test_patch_embedding_keeps_fp32_under_bf16_autocast():
    """The SigLIP patch/position embeddings are FP32 in to_bfloat16_for_selected_params."""
    model = _tiny_pi05_model()
    images = [torch.randn(2, 3, 4, 4)]
    image_masks = [torch.ones(2, dtype=torch.bool)]
    tokens = torch.zeros(2, 3, dtype=torch.long)
    token_masks = torch.ones(2, 3, dtype=torch.bool)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        model.embed_prefix(images, image_masks, tokens, token_masks)

    assert model.paligemma_with_expert.last_image_embedding_dtype == torch.float32


def test_embed_suffix_matches_between_autocast_and_plain_execution():
    model = _tiny_pi05_model()
    noisy_actions = torch.randn(2, 2, 3)
    timestep = torch.rand(2)

    plain, _, _, _ = model.embed_suffix(None, noisy_actions, timestep)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast, _, _, _ = model.embed_suffix(None, noisy_actions, timestep)

    torch.testing.assert_close(autocast, plain)


def test_gradient_checkpointing_defaults_off_and_toggles_the_backbone():
    model = _tiny_pi05_model()

    assert model.gradient_checkpointing_enabled is False
    assert model.paligemma_with_expert.gradient_checkpointing_flag is False

    model.set_gradient_checkpointing(enabled=True)

    assert model.gradient_checkpointing_enabled is True
    assert model.paligemma_with_expert.gradient_checkpointing_flag is True

    model.set_gradient_checkpointing(enabled=False)

    assert model.gradient_checkpointing_enabled is False
    assert model.paligemma_with_expert.gradient_checkpointing_flag is False


def test_gradient_checkpointing_recomputes_the_suffix_projections(monkeypatch):
    model = _tiny_pi05_model()
    model.train()
    noisy_actions = torch.randn(2, 2, 3, requires_grad=True)
    timestep = torch.rand(2)

    checkpoint_calls = []
    real_checkpoint = torch.utils.checkpoint.checkpoint

    def counting_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(function)
        return real_checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting_checkpoint)

    baseline, _, _, _ = model.embed_suffix(None, noisy_actions, timestep)
    assert checkpoint_calls == []

    model.set_gradient_checkpointing(enabled=True)
    checkpointed, _, _, _ = model.embed_suffix(None, noisy_actions, timestep)

    # action_in_proj and the time MLP.
    assert len(checkpoint_calls) == 2
    torch.testing.assert_close(checkpointed, baseline)

    checkpointed.sum().backward()
    assert noisy_actions.grad is not None

    checkpoint_calls.clear()
    model.eval()
    model.embed_suffix(None, noisy_actions, timestep)
    assert checkpoint_calls == []

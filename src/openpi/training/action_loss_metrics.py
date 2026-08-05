"""Action-dimension and semantic-component loss diagnostics."""

from collections.abc import Sequence

import torch

# Canonical OpenPI-32 action layout. Seventh joints occupy 26/27 so the layout
# remains compatible with six- and seven-DoF arms through the per-sample pad mask.
_OPENPI_32_COMPONENTS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("joint", (*range(6), *range(7, 13), 26, 27)),
    ("gripper", (6, 13)),
    ("ee", tuple(range(14, 26))),
    ("left_joint", (*range(6), 26)),
    ("right_joint", (*range(7, 13), 27)),
    ("left_gripper", (6,)),
    ("right_gripper", (13,)),
    ("left_ee", tuple(range(14, 20))),
    ("right_ee", tuple(range(20, 26))),
)


def action_loss_sums_and_counts(
    losses: torch.Tensor,
    actions_dim_is_pad: torch.Tensor | None,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Return loss sums and element counts for globally reducible diagnostics.

    ``losses`` must be the unreduced model loss with shape ``[batch, horizon, dim]``.
    Component statistics exclude padded source dimensions; the ``padding`` metric
    contains exactly those dimensions. Per-dimension metrics include every sample,
    making a padded dimension's prediction error directly visible.
    """
    if losses.ndim != 3:
        raise ValueError(f"Expected action losses with shape [batch, horizon, dim], got {tuple(losses.shape)}.")

    batch_size, horizon, action_dim = losses.shape
    if actions_dim_is_pad is None:
        pad_mask = torch.zeros((batch_size, action_dim), dtype=torch.bool, device=losses.device)
    else:
        pad_mask = actions_dim_is_pad.to(device=losses.device, dtype=torch.bool)
        if pad_mask.shape != (batch_size, action_dim):
            raise ValueError(
                f"actions_dim_is_pad must have shape ({batch_size}, {action_dim}), got {tuple(pad_mask.shape)}."
            )

    values = losses.detach().to(torch.float32)
    expanded_pad_mask = pad_mask[:, None, :].expand(batch_size, horizon, action_dim)
    names: list[str] = []
    rows: list[torch.Tensor] = []

    def add_metric(name: str, loss_sum: torch.Tensor, count: torch.Tensor | int) -> None:
        names.append(name)
        rows.append(torch.stack((loss_sum, torch.as_tensor(count, dtype=values.dtype, device=values.device))))

    add_metric("total", values.sum(), values.numel())
    add_metric("active", torch.where(~expanded_pad_mask, values, 0.0).sum(), (~pad_mask).sum() * horizon)
    add_metric("padding", torch.where(expanded_pad_mask, values, 0.0).sum(), pad_mask.sum() * horizon)

    if action_dim == 32:
        for name, indices in _OPENPI_32_COMPONENTS:
            component_values = values[..., list(indices)]
            component_is_active = ~pad_mask[:, list(indices)]
            add_metric(
                name,
                torch.where(component_is_active[:, None, :], component_values, 0.0).sum(),
                component_is_active.sum() * horizon,
            )

    return tuple(names), torch.stack(rows)


def action_loss_values(names: Sequence[str], sums_and_counts: torch.Tensor) -> dict[str, float]:
    """Convert reduced sums/counts into MSE, additive contribution, loss share and element fraction."""
    if sums_and_counts.shape != (len(names), 2):
        raise ValueError(f"Expected sums_and_counts shape ({len(names)}, 2), got {tuple(sums_and_counts.shape)}.")
    total_count = float(sums_and_counts[0, 1].item())
    if total_count <= 0:
        return {}

    result: dict[str, float] = {}
    for index, name in enumerate(names):
        loss_sum = float(sums_and_counts[index, 0].item())
        count = float(sums_and_counts[index, 1].item())
        if count <= 0:
            continue
        result[f"mse/{name}"] = loss_sum / count
        # result[f"contribution/{name}"] = loss_sum / total_count
        # if total_loss_sum != 0:
        #     result[f"share/{name}"] = loss_sum / total_loss_sum
        # result[f"fraction/{name}"] = count / total_count
    return result

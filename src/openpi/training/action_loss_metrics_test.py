import torch

from openpi.training import action_loss_metrics


def test_action_loss_breakdown_uses_per_sample_padding_mask():
    losses = torch.ones((2, 2, 32), dtype=torch.float32)
    losses[..., 6] = 2.0
    losses[..., 28:] = 3.0
    pad_mask = torch.zeros((2, 32), dtype=torch.bool)
    pad_mask[:, 28:] = True
    pad_mask[0, 26] = True

    names, stats = action_loss_metrics.action_loss_sums_and_counts(losses, pad_mask)
    values = action_loss_metrics.action_loss_values(names, stats)

    assert values["mse/gripper"] == 1.5
    assert values["mse/padding"] == 25 / 9
    assert values["fraction/padding"] == 9 / 64
    assert values["mse/dim_06"] == 2.0
    assert values["mse/dim_28"] == 3.0
    assert values["contribution/active"] + values["contribution/padding"] == values["mse/total"]
    assert values["share/active"] + values["share/padding"] == 1.0


def test_action_loss_breakdown_without_mask_treats_all_dimensions_as_active():
    losses = torch.full((1, 3, 4), 2.0)

    names, stats = action_loss_metrics.action_loss_sums_and_counts(losses, None)
    values = action_loss_metrics.action_loss_values(names, stats)

    assert values["mse/total"] == 2.0
    assert values["mse/active"] == 2.0
    assert "mse/padding" not in values
    assert values["mse/dim_03"] == 2.0

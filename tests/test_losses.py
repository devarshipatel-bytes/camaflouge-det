"""Correctness tests for chd.losses."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.losses import CHDLoss, boundary_weight, dice_loss, weighted_bce_iou  # noqa: E402

B, H, W = 2, 32, 32


def fake_outputs(h: int = H, w: int = W, b: int = B) -> dict[str, torch.Tensor]:
    return {
        "mask_logit": torch.randn(b, 1, h, w, requires_grad=True),
        "side_logits": [torch.randn(b, 1, h, w, requires_grad=True) for _ in range(4)],
        "edge_logit": torch.randn(b, 1, h, w, requires_grad=True),
        "presence_logit": torch.randn(b, requires_grad=True),
    }


class TestBoundaryWeight:
    def test_flat_mask_has_near_uniform_weight(self) -> None:
        mask = torch.zeros(1, 1, 40, 40)
        w = boundary_weight(mask)
        assert torch.allclose(w, torch.ones_like(w), atol=1e-5)

    def test_boundary_pixels_weighted_higher_than_interior(self) -> None:
        mask = torch.zeros(1, 1, 40, 40)
        mask[:, :, 10:30, 10:30] = 1.0
        w = boundary_weight(mask, kernel_size=5)
        interior_weight = w[0, 0, 20, 20]  # deep inside the box
        boundary_weight_val = w[0, 0, 10, 20]  # right on the edge
        assert boundary_weight_val > interior_weight


class TestWeightedBceIou:
    def test_perfect_logits_give_low_loss(self) -> None:
        target = (torch.rand(B, 1, H, W) > 0.5).float()
        logit = (target * 2 - 1) * 20  # saturated logits matching target exactly
        loss = weighted_bce_iou(logit, target)
        assert loss.item() < 0.05

    def test_gradient_flows(self) -> None:
        logit = torch.randn(B, 1, H, W, requires_grad=True)
        target = (torch.rand(B, 1, H, W) > 0.5).float()
        weighted_bce_iou(logit, target).backward()
        assert logit.grad is not None and torch.isfinite(logit.grad).all()


class TestDiceLoss:
    def test_perfect_match_near_zero(self) -> None:
        target = (torch.rand(B, 1, H, W) > 0.5).float()
        logit = (target * 2 - 1) * 20
        assert dice_loss(logit, target).item() < 0.05

    def test_no_overlap_near_one(self) -> None:
        target = torch.zeros(1, 1, H, W)
        target[:, :, :H // 2, :] = 1.0
        logit = torch.full((1, 1, H, W), -20.0)
        logit[:, :, H // 2:, :] = 20.0  # confidently predicts exactly the complement
        assert dice_loss(logit, target).item() > 0.95


class TestCHDLoss:
    def test_returns_all_expected_keys(self) -> None:
        criterion = CHDLoss()
        outputs = fake_outputs()
        mask_target = (torch.rand(B, 1, H, W) > 0.5).float()
        edge_target = (torch.rand(B, 1, H, W) > 0.8).float()
        is_negative = torch.zeros(B)
        result = criterion(outputs, mask_target, edge_target, is_negative)
        for key in ("total", "final", "side", "edge", "presence"):
            assert key in result
        assert torch.isfinite(result["total"])

    def test_gradient_flows_to_every_output_head(self) -> None:
        criterion = CHDLoss()
        outputs = fake_outputs()
        mask_target = (torch.rand(B, 1, H, W) > 0.5).float()
        edge_target = (torch.rand(B, 1, H, W) > 0.8).float()
        is_negative = torch.zeros(B)
        result = criterion(outputs, mask_target, edge_target, is_negative)
        result["total"].backward()

        assert outputs["mask_logit"].grad is not None
        assert all(s.grad is not None for s in outputs["side_logits"])
        assert outputs["edge_logit"].grad is not None
        assert outputs["presence_logit"].grad is not None

    def test_all_negative_batch_skips_edge_loss_but_stays_finite(self) -> None:
        criterion = CHDLoss()
        outputs = fake_outputs()
        mask_target = torch.zeros(B, 1, H, W)
        edge_target = torch.zeros(B, 1, H, W)
        is_negative = torch.ones(B)  # every sample in the batch is a negative
        result = criterion(outputs, mask_target, edge_target, is_negative)
        assert result["edge"].item() == pytest.approx(0.0)
        assert torch.isfinite(result["total"])
        result["total"].backward()  # must not raise even though edge contributes zero

    def test_presence_target_is_inverse_of_is_negative(self) -> None:
        """A confident-presence prediction should score a lower presence loss
        on a positive sample than on a negative one, proving the target flip
        (presence_target = 1 - is_negative) is wired correctly."""
        criterion = CHDLoss()
        outputs = fake_outputs(b=1)
        outputs["presence_logit"] = torch.tensor([10.0], requires_grad=True)  # confidently "present"
        mask_target = (torch.rand(1, 1, H, W) > 0.5).float()
        edge_target = torch.zeros(1, 1, H, W)

        positive = criterion(outputs, mask_target, edge_target, is_negative=torch.zeros(1))
        negative = criterion(outputs, mask_target, edge_target, is_negative=torch.ones(1))
        assert positive["presence"].item() < negative["presence"].item()

    def test_side_weights_must_have_length_four(self) -> None:
        with pytest.raises(ValueError):
            CHDLoss(side_weights=(0.5, 1.0))

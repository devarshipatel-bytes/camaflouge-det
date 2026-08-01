"""Training losses for OS-Res2Net-CHDNet.

    L = L_final + sum_i lambda_side_i * L_side_i + lambda_edge * L_edge + lambda_presence * L_presence

``L_final`` / ``L_side``: weighted BCE + weighted IoU, boundary-distance
pixel weighting (the standard COD "structure loss") — the boundary emphasis
is what drives the F^bd metric. ``L_edge``: BCE + Dice (edges are ~0.1-1% of
pixels, so Dice matters). ``L_presence``: plain BCE against whether the
target mask is non-empty. Negative (presence-gate) samples contribute
presence + mask loss against an all-zero target, and no edge loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from chd._compat import zip_strict


def boundary_weight(mask: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
    """Per-pixel weight emphasising the boundary band of a binary mask.

    Matches the common COD "weighted BCE/IoU" recipe: weight = 1 + 5 * |local
    average - mask|, so pixels near a boundary (where the local average
    differs most from the pixel's own value) get up-weighted; large flat
    interior/exterior regions stay near weight 1.
    """
    pad = kernel_size // 2
    local_mean = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    return 1.0 + 5.0 * torch.abs(local_mean - mask)


def weighted_bce_iou(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        weight = boundary_weight(target)

    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    bce = (weight * bce).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp_min(1e-6)

    prob = torch.sigmoid(logit)
    inter = (weight * prob * target).sum(dim=(1, 2, 3))
    union = (weight * (prob + target)).sum(dim=(1, 2, 3)) - inter
    iou = 1.0 - (inter + 1.0) / (union + 1.0)

    return (bce + iou).mean()


def dice_loss(logit: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2 * inter + eps) / (denom + eps)).mean()


def edge_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target) + dice_loss(logit, target)


class CHDLoss(nn.Module):
    """Combines every model output against its target into the final scalar loss.

    ``is_negative``: (B,) bool/float, 1 where the sample is a presence-gate
    negative (no camouflaged human present) — those rows skip the edge loss
    since there is no boundary to supervise.
    """

    def __init__(
        self,
        side_weights: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0),
        edge_weight: float = 1.0,
        presence_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if len(side_weights) != 4:
            raise ValueError(f"expected 4 side weights (one per pyramid level), got {len(side_weights)}")
        self.side_weights = side_weights
        self.edge_weight = edge_weight
        self.presence_weight = presence_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        mask_target: torch.Tensor,
        edge_target: torch.Tensor,
        is_negative: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        is_negative = is_negative.float().view(-1)
        presence_target = 1.0 - is_negative

        final = weighted_bce_iou(outputs["mask_logit"], mask_target)
        side = sum(
            w * weighted_bce_iou(logit, mask_target)
            for w, logit in zip_strict(self.side_weights, outputs["side_logits"])
        )

        # Edge loss is computed per-sample then masked, not skipped by
        # slicing the batch — keeps shapes static, which matters for
        # torch.compile / fixed-batch training loops.
        positive = 1.0 - is_negative
        if positive.sum() > 0:
            per_sample_edge = F.binary_cross_entropy_with_logits(
                outputs["edge_logit"], edge_target, reduction="none",
            ).mean(dim=(1, 2, 3))
            per_sample_dice = 1.0 - (
                2 * (torch.sigmoid(outputs["edge_logit"]) * edge_target).sum(dim=(1, 2, 3)) + 1.0
            ) / (
                (torch.sigmoid(outputs["edge_logit"]) + edge_target).sum(dim=(1, 2, 3)) + 1.0
            )
            edge = ((per_sample_edge + per_sample_dice) * positive).sum() / positive.sum().clamp_min(1e-6)
        else:
            edge = outputs["edge_logit"].sum() * 0.0  # zero, but keeps autograd graph connected

        presence = F.binary_cross_entropy_with_logits(outputs["presence_logit"], presence_target)

        total = final + side + self.edge_weight * edge + self.presence_weight * presence
        return {
            "total": total,
            "final": final.detach(),
            "side": side.detach(),
            "edge": edge.detach(),
            "presence": presence.detach(),
        }

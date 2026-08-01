"""Output heads: deep-supervision side masks, an edge head, and the presence gate.

The presence gate is an ANet-style anabranch: a cheap image-level classifier
predicting whether a camouflaged human is in the frame at all, multiplied
into the final mask. It is what lets the model answer "nothing here" on
negatives (MHCD's no-person images, or an arbitrary internet photo) instead
of hallucinating a blob on every input.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from chd._compat import zip_strict


class SideHeads(nn.Module):
    """One 1x1-conv logit per decoder level, each upsampled to the input size."""

    def __init__(self, channels: int = 64, n_levels: int = 4) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Conv2d(channels, 1, kernel_size=1) for _ in range(n_levels)])

    def forward(self, feats: tuple[torch.Tensor, ...], out_size: tuple[int, int]) -> list[torch.Tensor]:
        return [F.interpolate(head(feat), size=out_size, mode="bilinear", align_corners=False)
                for head, feat in zip_strict(self.heads, feats)]


class EdgeHead(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, feat: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(self.conv(feat), size=out_size, mode="bilinear", align_corners=False)


class PresenceGate(nn.Module):
    """GAP(backbone F4) -> FC -> ReLU -> Dropout -> FC -> sigmoid."""

    def __init__(self, in_channels: int = 2048, hidden: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, backbone_f4: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = backbone_f4.shape
        logit = self.net(self.gap(backbone_f4).view(b, c))
        return logit.squeeze(-1)  # (B,) presence logit, sigmoid applied by the loss / at inference

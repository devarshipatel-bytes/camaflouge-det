"""Partial Decoder Component (PDC): top-down multiplicative fusion.

Each level is gated by a 1x1-conv + upsample of the level above it (deepest
first), then all four levels are resampled to the shallowest resolution and
concatenated for a final refinement:

    x4 = f4
    x3 = f3 * Up(Conv(x4))
    x2 = f2 * Up(Conv(x3))
    x1 = f1 * Up(Conv(x2))
    fused = concat(Up(x4), Up(x3), Up(x2), x1) -> Conv3x3-BN-ReLU x2 -> Conv1x1
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PDCDecoder(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.gate43 = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate32 = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate21 = nn.Conv2d(channels, channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.to_logit = nn.Conv2d(channels, 1, kernel_size=1)

    @staticmethod
    def _up_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, f4: torch.Tensor):
        x4 = f4
        x3 = f3 * self._up_to(self.gate43(x4), f3.shape[-2:])
        x2 = f2 * self._up_to(self.gate32(x3), f2.shape[-2:])
        x1 = f1 * self._up_to(self.gate21(x2), f1.shape[-2:])

        target = x1.shape[-2:]
        fused = torch.cat([self._up_to(x4, target), self._up_to(x3, target), self._up_to(x2, target), x1], dim=1)
        decoded = self.fuse(fused)
        main_logit = self.to_logit(decoded)
        return main_logit, (x1, x2, x3, x4)

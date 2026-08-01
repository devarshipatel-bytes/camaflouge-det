"""Anatomy-guided Edge Reconstruction (AER).

Injects a cached, confidence-weighted pose prior as a soft structural gate:

    P     = resize(pose_cache, feat's H x W)              17ch
    A     = sigmoid(Conv3x3(ReLU(BN(Conv3x3(concat(P, feat) -> 64)))) -> 1)   spatial attention map
    F_out = feat * (A + 1)

The "+1" means a location with zero anatomical evidence (a camouflaged
subject the pose model missed entirely, or true background) passes the
feature through unchanged rather than being suppressed — the prior can only
add emphasis, never censor a location the segmentation head believes in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

N_KEYPOINTS = 17


class AER(nn.Module):
    def __init__(self, channels: int = 64, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(N_KEYPOINTS + channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        if pose.shape[-2:] != feat.shape[-2:]:
            pose = F.interpolate(pose, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        attn = self.net(torch.cat([pose, feat], dim=1))
        return feat * (attn + 1.0)

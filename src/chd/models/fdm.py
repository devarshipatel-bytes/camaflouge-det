"""Feature Decomposition Module (FDM).

Per level, decomposes a backbone feature into a low-frequency (smooth,
regional) component and a high-frequency (detail/boundary) component:

    S3, S5, S7 = AvgPool_{3,5,7}(F)                    multi-scale smoothing
    alpha = softmax(MLP(GAP(F)))                        3 scale weights
    F_LF  = alpha_3*S3 + alpha_5*S5 + alpha_7*S7         low-frequency estimate
    F_HF  = F - F_LF                                     high-frequency residual
    F_HF_hat = PWConv(ReLU(BN(DWConv3x3(GN(F_HF)))))    stabilise + refine
"""

from __future__ import annotations

import torch
from torch import nn


class FDM(nn.Module):
    def __init__(self, channels: int = 64, hidden: int = 16, groups: int = 8) -> None:
        super().__init__()
        self.pool3 = nn.AvgPool2d(3, stride=1, padding=1)
        self.pool5 = nn.AvgPool2d(5, stride=1, padding=2)
        self.pool7 = nn.AvgPool2d(7, stride=1, padding=3)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.scale_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

        self.refine = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),  # depthwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),  # pointwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, _, _ = feat.shape
        s3, s5, s7 = self.pool3(feat), self.pool5(feat), self.pool7(feat)

        alpha = torch.softmax(self.scale_mlp(self.gap(feat).view(b, c)), dim=1)  # (B, 3)
        a3, a5, a7 = (alpha[:, i].view(b, 1, 1, 1) for i in range(3))
        f_lf = a3 * s3 + a5 * s5 + a7 * s7
        f_hf = feat - f_lf
        f_hf_hat = self.refine(f_hf)
        return f_lf, f_hf_hat

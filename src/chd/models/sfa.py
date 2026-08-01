"""Spatial-Frequency Collaborative Attention (SFA).

Adaptively balances the low- and high-frequency components FDM produced,
per spatial location, then applies a residual gate:

    cat    = concat(F_LF, F_HF_hat)                          128ch
    W_LF   = sigmoid(Conv3x3(cat))                            64ch gate
    W_HF   = sigmoid(Conv3x3(cat))                            64ch gate
    F_fuse = W_LF * F_LF + W_HF * F_HF_hat
    F_ref  = GateAttn(F + F_fuse)     [Conv3x3-BN-ReLU-Conv3x3-Sigmoid]
    F_SFA  = F + F_fuse + (F + F_fuse) * F_ref
"""

from __future__ import annotations

import torch
from torch import nn


class SFA(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.lf_gate = nn.Sequential(nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1), nn.Sigmoid())
        self.hf_gate = nn.Sequential(nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1), nn.Sigmoid())
        self.gate_attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, f_lf: torch.Tensor, f_hf_hat: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([f_lf, f_hf_hat], dim=1)
        w_lf, w_hf = self.lf_gate(cat), self.hf_gate(cat)
        f_fuse = w_lf * f_lf + w_hf * f_hf_hat
        base = feat + f_fuse
        f_ref = self.gate_attn(base)
        return base + base * f_ref

"""OSBlock — the omni-scale unit that gives the architecture its name.

Replaces the plain dilated-conv "receptive field" module used by comparable
COD decoders with parallel multi-depth Lite3x3 streams (receptive fields
3, 5, 7, 9) fused by a shared Aggregation Gate, following OSNet's
omni-scale design:

    x -> Conv1x1 reduce (C/4)
      -> T1: Lite3x3                rf 3
      -> T2: Lite3x3 x2             rf 5
      -> T3: Lite3x3 x3             rf 7
      -> T4: Lite3x3 x4             rf 9
    AG (shared) = GAP -> FC(r) -> ReLU -> FC(back) -> Sigmoid, applied per-stream
    y = sum_t AG(T_t(x)) * T_t(x)
    -> Conv1x1 expand -> BN, residual add, ReLU

``Lite3x3 = Conv1x1 -> DWConv3x3 -> BN -> ReLU``.
"""

from __future__ import annotations

import torch
from torch import nn


class Lite3x3(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),  # depthwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AggregationGate(nn.Module):
    """Shared per-stream channel gate: GAP -> FC -> ReLU -> FC -> Sigmoid."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        gate = self.fc(self.gap(x).view(b, c))
        return gate.view(b, c, 1, 1)


class OSBlock(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, n_streams: int = 4,
                reduction: int = 4, gate_reduction: int = 4) -> None:
        super().__init__()
        bottleneck = max(1, in_channels // reduction)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1), nn.BatchNorm2d(bottleneck), nn.ReLU(inplace=True),
        )
        self.streams = nn.ModuleList([
            nn.Sequential(*(Lite3x3(bottleneck) for _ in range(t))) for t in range(1, n_streams + 1)
        ])
        self.gate = AggregationGate(bottleneck, reduction=gate_reduction)
        self.expand = nn.Sequential(nn.Conv2d(bottleneck, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
        self.act = nn.ReLU(inplace=True)
        self.project_residual = None
        if in_channels != out_channels:
            self.project_residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        stream_outs = [stream(reduced) for stream in self.streams]
        y = sum(self.gate(out) * out for out in stream_outs)
        y = self.expand(y)
        residual = x if self.project_residual is None else self.project_residual(x)
        return self.act(y + residual)

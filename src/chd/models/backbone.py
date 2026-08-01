"""Res2Net-50 encoder, exposed as a 4-level feature pyramid.

    stem     conv3x3,32,s2 -> conv3x3,32 -> conv3x3,64 -> BN+ReLU -> maxpool3x3,s2   88^2 x 64
    layer1   Res2Bottleneck(w=26,scale=4) x3                                         88^2 x 256   F1
    layer2   Res2Bottleneck x4, s2                                                   44^2 x 512   F2
    layer3   Res2Bottleneck x6, s2                                                   22^2 x 1024  F3
    layer4   Res2Bottleneck x3, s2                                                   11^2 x 2048  F4

(spatial sizes shown for a 352x352 input; the module is fully convolutional
and works at any input size that is a multiple of 32).
"""

from __future__ import annotations

import torch
from torch import nn

FEATURE_CHANNELS = (256, 512, 1024, 2048)  # F1..F4


class Res2NetBackbone(nn.Module):
    """Wraps timm's ``res2net50_26w_4s`` as a 4-level feature pyramid (F1..F4)."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.body = timm.create_model(
            "res2net50_26w_4s", pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4),
        )
        # NB: ``feature_info`` iterated directly lists every stage regardless
        # of out_indices — only ``.channels()`` (and the actual forward pass)
        # respect the selection. Validate against that, not the raw iterator.
        chs = tuple(self.body.feature_info.channels())
        if chs != FEATURE_CHANNELS:
            raise RuntimeError(f"unexpected Res2Net feature channels {chs}, expected {FEATURE_CHANNELS}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.body(x))


class TinyTestBackbone(nn.Module):
    """A tiny random-weight stand-in with the same channel/stride contract as Res2Net-50.

    Used only by unit tests and quick smoke runs so they don't have to pull
    ImageNet weights or run a 25M-parameter encoder just to check shapes.
    """

    def __init__(self, pretrained: bool = False) -> None:  # noqa: ARG002 - kept for interface parity
        super().__init__()
        in_ch = 3
        self.stages = nn.ModuleList()
        stride = [4, 2, 2, 2]
        for out_ch, s in zip(FEATURE_CHANNELS, stride, strict=True):
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=s, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


def build_backbone(name: str = "res2net50_26w_4s", pretrained: bool = True) -> nn.Module:
    if name == "res2net50_26w_4s":
        return Res2NetBackbone(pretrained=pretrained)
    if name == "tiny_test":
        return TinyTestBackbone()
    raise ValueError(f"unknown backbone {name!r}")

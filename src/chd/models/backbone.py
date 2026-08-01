"""Feature-pyramid encoders for CHDNet.

Every backbone here exposes the same contract — a 4-level pyramid at strides
4/8/16/32 — so FDM/SFA/OSNeck/AER/decoder plug in unchanged regardless of
which encoder is selected. Only the per-level channel widths differ, which
is why ``feature_channels`` is a per-instance attribute rather than a module
constant.

    res2net50_26w_4s   CNN, ImageNet          23.65M   channels 256/512/1024/2048
      stem     conv3x3x3,32,s2 -> conv3x3,32 -> conv3x3,64 -> BN+ReLU -> maxpool3x3,s2
      layer1   Res2Bottleneck(w=26,scale=4) x3        stride 4    F1
      layer2   Res2Bottleneck x4                      stride 8    F2
      layer3   Res2Bottleneck x6                      stride 16   F3
      layer4   Res2Bottleneck x3                      stride 32   F4

    pvt_v2_b2          Transformer, ImageNet  24.85M   channels 64/128/320/512
      stage1   OverlapPatchEmbed(s4) + 3x [SRA + MixFFN]    stride 4    F1
      stage2   OverlapPatchEmbed(s2) + 4x [SRA + MixFFN]    stride 8    F2
      stage3   OverlapPatchEmbed(s2) + 6x [SRA + MixFFN]    stride 16   F3
      stage4   OverlapPatchEmbed(s2) + 3x [SRA + MixFFN]    stride 32   F4

PVTv2-B2 is what the FSCL paper and most 2023+ COD SOTA (HitNet, CamoFormer,
ESCNet) use. It is NOT a lighter option — it is essentially the same
parameter budget as Res2Net-50 with a different inductive bias: global
self-attention per stage instead of purely local convolution. That helps
camouflage reasoning (foreground/background separation is a global-context
problem) but, lacking a CNN's locality prior, it is also the more
overfitting-prone choice on very small fine-tuning sets.
"""

from __future__ import annotations

import torch
from torch import nn

from chd._compat import zip_strict

#: Res2Net-50's per-level output widths; kept as a named constant because
#: the tiny test backbone deliberately mimics them.
RES2NET_CHANNELS = (256, 512, 1024, 2048)
PVTV2_B2_CHANNELS = (64, 128, 320, 512)

#: Backwards-compatible alias: several tests and the presence gate referred
#: to this name when Res2Net-50 was the only option.
FEATURE_CHANNELS = RES2NET_CHANNELS


class TimmPyramidBackbone(nn.Module):
    """Wraps any timm ``features_only`` model as a 4-level pyramid (F1..F4)."""

    def __init__(self, timm_name: str, expected_channels: tuple[int, ...], pretrained: bool = True) -> None:
        super().__init__()
        import timm

        self.body = timm.create_model(
            timm_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
        ) if timm_name.startswith("pvt") else timm.create_model(
            timm_name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4),
        )
        # NB: ``feature_info`` iterated directly lists every stage regardless
        # of out_indices — only ``.channels()`` (and the actual forward pass)
        # respect the selection. Validate against that, not the raw iterator.
        channels = tuple(self.body.feature_info.channels())
        if channels != expected_channels:
            raise RuntimeError(f"{timm_name}: unexpected feature channels {channels}, expected {expected_channels}")
        self.feature_channels = channels

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
        for out_ch, s in zip_strict(RES2NET_CHANNELS, stride):
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=s, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.feature_channels = RES2NET_CHANNELS

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


#: name -> (timm model name, expected per-level channels)
BACKBONES = {
    "res2net50_26w_4s": ("res2net50_26w_4s", RES2NET_CHANNELS),
    "pvtv2_b2": ("pvt_v2_b2", PVTV2_B2_CHANNELS),
}


def build_backbone(name: str = "res2net50_26w_4s", pretrained: bool = True) -> nn.Module:
    if name == "tiny_test":
        return TinyTestBackbone()
    if name in BACKBONES:
        timm_name, channels = BACKBONES[name]
        return TimmPyramidBackbone(timm_name, channels, pretrained=pretrained)
    raise ValueError(f"unknown backbone {name!r}, expected one of {sorted(BACKBONES) + ['tiny_test']}")

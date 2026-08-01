"""Single entry point for building any of the three architectures.

    chdnet + res2net50_26w_4s   the original design, unchanged (default)
    chdnet + pvtv2_b2           Option A: same design, transformer encoder
    pretrained_unet             Option C: off-the-shelf pretrained U-Net baseline

All three return the same output dict, so the training loop, losses,
metrics and evaluation code are shared with no per-architecture branching.
``predict_mask`` is resolved off the returned module, since each class
defines its own.
"""

from __future__ import annotations

import argparse

from torch import nn

from chd.models.chdnet import CHDNet
from chd.models.pretrained_unet import PretrainedUNet

ARCHITECTURES = ("chdnet", "pretrained_unet")


def build_model(args: argparse.Namespace) -> nn.Module:
    architecture = getattr(args, "architecture", "chdnet")
    pretrained = not getattr(args, "no_pretrained", False)

    if architecture == "chdnet":
        return CHDNet(
            backbone=args.backbone,
            pretrained=pretrained,
            os_streams=getattr(args, "os_streams", 4),
        )
    if architecture == "pretrained_unet":
        return PretrainedUNet(
            encoder_name=getattr(args, "unet_encoder", "resnet34"),
            pretrained=pretrained,
            freeze_encoder=getattr(args, "unet_freeze_encoder", True),
        )
    raise ValueError(f"unknown architecture {architecture!r}, expected one of {ARCHITECTURES}")


def describe_model(model: nn.Module, args: argparse.Namespace) -> str:
    """One-line description for the training run header."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    architecture = getattr(args, "architecture", "chdnet")
    if architecture == "chdnet":
        detail = f"chdnet + {args.backbone}"
    else:
        frozen = "frozen" if getattr(args, "unet_freeze_encoder", True) else "fine-tuned"
        detail = f"pretrained_unet + {getattr(args, 'unet_encoder', 'resnet34')} ({frozen} encoder)"
    return f"{detail}  ({total/1e6:.2f}M params, {trainable/1e6:.2f}M trainable)"

"""Tests for the three selectable architectures.

Every architecture must satisfy the same output contract so the training
loop, losses, metrics and evaluation stay shared with no branching.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.losses import CHDLoss  # noqa: E402
from chd.models.factory import build_model, describe_model  # noqa: E402

B, H, W = 2, 64, 64
N_KEYPOINTS = 17


def make_args(**overrides) -> argparse.Namespace:
    base = dict(
        architecture="chdnet",
        backbone="tiny_test",
        no_pretrained=True,
        os_streams=4,
        unet_encoder="resnet34",
        unet_freeze_encoder=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


ALL_ARCHITECTURES = [
    pytest.param(make_args(architecture="chdnet", backbone="tiny_test"), id="chdnet-tiny"),
    pytest.param(make_args(architecture="pretrained_unet"), id="unet-frozen"),
    pytest.param(make_args(architecture="pretrained_unet", unet_freeze_encoder=False), id="unet-unfrozen"),
]


class TestSharedOutputContract:
    @pytest.mark.parametrize("args", ALL_ARCHITECTURES)
    def test_output_keys_and_shapes(self, args: argparse.Namespace) -> None:
        model = build_model(args).eval()
        image = torch.randn(B, 3, H, W)
        pose = torch.randn(B, N_KEYPOINTS, H // 4, W // 4)
        with torch.no_grad():
            out = model(image, pose)
        assert out["mask_logit"].shape == (B, 1, H, W)
        assert out["edge_logit"].shape == (B, 1, H, W)
        assert out["presence_logit"].shape == (B,)
        assert len(out["side_logits"]) == 4
        assert all(s.shape == (B, 1, H, W) for s in out["side_logits"])

    @pytest.mark.parametrize("args", ALL_ARCHITECTURES)
    def test_predict_mask_is_bounded(self, args: argparse.Namespace) -> None:
        model = build_model(args).eval()
        image = torch.randn(B, 3, H, W)
        pose = torch.zeros(B, N_KEYPOINTS, H // 4, W // 4)
        with torch.no_grad():
            mask = model.predict_mask(model(image, pose))
        assert mask.shape == (B, 1, H, W)
        assert (mask >= 0).all() and (mask <= 1).all()

    @pytest.mark.parametrize("args", ALL_ARCHITECTURES)
    def test_works_with_shared_loss(self, args: argparse.Namespace) -> None:
        """The whole point of the shared contract: one CHDLoss for all three."""
        model = build_model(args).train()
        criterion = CHDLoss()
        image = torch.randn(B, 3, H, W)
        pose = torch.randn(B, N_KEYPOINTS, H // 4, W // 4)
        mask_target = (torch.rand(B, 1, H, W) > 0.5).float()
        edge_target = (torch.rand(B, 1, H, W) > 0.8).float()
        losses = criterion(model(image, pose), mask_target, edge_target, torch.zeros(B))
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        # at least one trainable parameter must have received a gradient
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)

    @pytest.mark.parametrize("args", ALL_ARCHITECTURES)
    def test_non_square_input(self, args: argparse.Namespace) -> None:
        model = build_model(args).eval()
        image = torch.randn(1, 3, 64, 96)
        pose = torch.randn(1, N_KEYPOINTS, 16, 24)
        with torch.no_grad():
            out = model(image, pose)
        assert out["mask_logit"].shape == (1, 1, 64, 96)


class TestPretrainedUNetSpecifics:
    def test_frozen_encoder_has_no_trainable_encoder_params(self) -> None:
        model = build_model(make_args(architecture="pretrained_unet", unet_freeze_encoder=True))
        encoder_trainable = sum(p.numel() for p in model.net.encoder.parameters() if p.requires_grad)
        assert encoder_trainable == 0
        # but the decoder and heads must still be trainable, or nothing learns
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_trainable > 0

    def test_unfrozen_encoder_is_trainable(self) -> None:
        model = build_model(make_args(architecture="pretrained_unet", unet_freeze_encoder=False))
        encoder_trainable = sum(p.numel() for p in model.net.encoder.parameters() if p.requires_grad)
        assert encoder_trainable > 0

    def test_frozen_encoder_stays_in_eval_during_train(self) -> None:
        """A frozen encoder must not keep updating BatchNorm running stats."""
        model = build_model(make_args(architecture="pretrained_unet", unet_freeze_encoder=True))
        model.train()
        bn_modules = [m for m in model.net.encoder.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert bn_modules, "expected the resnet34 encoder to contain BatchNorm layers"
        assert all(not m.training for m in bn_modules)

    def test_ignores_pose_without_error(self) -> None:
        model = build_model(make_args(architecture="pretrained_unet")).eval()
        image = torch.randn(1, 3, H, W)
        with torch.no_grad():
            a = model(image, torch.zeros(1, N_KEYPOINTS, 16, 16))
            b = model(image, torch.randn(1, N_KEYPOINTS, 16, 16) * 100)
        assert torch.equal(a["mask_logit"], b["mask_logit"]), "U-Net must be unaffected by the pose input"


class TestFactory:
    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown architecture"):
            build_model(make_args(architecture="does_not_exist"))

    def test_describe_model_mentions_the_right_thing(self) -> None:
        chdnet = make_args(architecture="chdnet", backbone="tiny_test")
        unet = make_args(architecture="pretrained_unet")
        assert "tiny_test" in describe_model(build_model(chdnet), chdnet)
        assert "resnet34" in describe_model(build_model(unet), unet)
        assert "frozen" in describe_model(build_model(unet), unet)


@pytest.mark.slow
class TestRealBackbones:
    """Downloads/instantiates the real encoders; opt-in via -m slow."""

    def test_pvtv2_b2_pyramid_contract(self) -> None:
        pytest.importorskip("timm")
        args = make_args(architecture="chdnet", backbone="pvtv2_b2")
        model = build_model(args).eval()
        image = torch.randn(1, 3, 128, 128)
        pose = torch.randn(1, N_KEYPOINTS, 32, 32)
        with torch.no_grad():
            out = model(image, pose)
        assert out["mask_logit"].shape == (1, 1, 128, 128)
        n = sum(p.numel() for p in model.parameters())
        assert 20e6 < n < 40e6, f"expected ~26M params for pvtv2_b2 CHDNet, got {n/1e6:.1f}M"

    def test_res2net_and_pvtv2_have_different_channel_widths(self) -> None:
        """Guards the per-instance feature_channels wiring: if this regressed to
        a hardcoded constant, one of these would fail to build."""
        pytest.importorskip("timm")
        from chd.models.backbone import build_backbone
        res2net = build_backbone("res2net50_26w_4s", pretrained=False)
        pvtv2 = build_backbone("pvtv2_b2", pretrained=False)
        assert res2net.feature_channels == (256, 512, 1024, 2048)
        assert pvtv2.feature_channels == (64, 128, 320, 512)

"""Tests for chd.viz.cam — gradient-weighted, target-specific saliency.

This is the family the existing pipeline figure lacks. That figure renders
mean|activation| across channels, which is an unsigned texture response and
is not target-specific, so nothing in it resembles the predicted mask.
Grad-CAM answers the different question "which locations drove THIS mask".

Everything runs on the tiny_test backbone so no ImageNet weights are needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.models.factory import build_model  # noqa: E402
from chd.viz import cam  # noqa: E402

IMG_SIZE = 64
N_KEYPOINTS = 17


@pytest.fixture()
def model():
    cfg = argparse.Namespace(architecture="chdnet", backbone="tiny_test", dataset="toy",
                             img_size=IMG_SIZE, os_streams=2, no_pose=False, no_pretrained=True)
    return build_model(cfg).eval()


@pytest.fixture()
def batch():
    torch.manual_seed(0)
    image = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    pose = torch.rand(1, N_KEYPOINTS, IMG_SIZE // 4, IMG_SIZE // 4)
    return image, pose


class TestGradCamLevels:
    def test_returns_one_map_per_pyramid_level(self, model, batch) -> None:
        image, pose = batch
        cams, tap = cam.grad_cam_levels(model, image, pose)
        assert len(cams) == 4
        assert tap == "aer"

    def test_maps_are_upsampled_to_the_input_size(self, model, batch) -> None:
        image, pose = batch
        cams, _ = cam.grad_cam_levels(model, image, pose)
        for c in cams:
            assert c.shape == (IMG_SIZE, IMG_SIZE)

    def test_maps_are_finite_non_negative_and_unit_ranged(self, model, batch) -> None:
        image, pose = batch
        cams, _ = cam.grad_cam_levels(model, image, pose)
        for c in cams:
            assert np.isfinite(c).all()
            assert c.min() >= 0.0 and c.max() <= 1.0

    def test_at_least_one_level_is_not_uniformly_zero(self, model, batch) -> None:
        """An all-zero CAM at every level means the gradient path is broken."""
        image, pose = batch
        cams, _ = cam.grad_cam_levels(model, image, pose)
        assert any(c.max() > 0.0 for c in cams)

    def test_alternate_taps_work(self, model, batch) -> None:
        image, pose = batch
        for tap in ("osneck", "sfa", "backbone"):
            cams, used = cam.grad_cam_levels(model, image, pose, tap=tap)
            assert used == tap
            assert len(cams) == 4

    def test_gt_target_uses_the_supplied_mask(self, model, batch) -> None:
        image, pose = batch
        gt = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
        gt[..., 20:40, 20:40] = 1.0
        cams, _ = cam.grad_cam_levels(model, image, pose, target="gt", gt=gt)
        assert len(cams) == 4

    def test_empty_prediction_falls_back_to_topk_logits(self, model, batch) -> None:
        """Negatives predict nothing above 0.5. Summing mask_logit over an empty
        region gives a constant zero with no gradient, so cam_score must fall
        back to the top-k logits or autograd.grad raises on an unused input.

        The empty prediction is forced by driving the presence gate to zero,
        since predict_mask multiplies the mask by the presence probability.
        """
        image, pose = batch
        with torch.no_grad():
            # PresenceGate holds its layers in .net (a Sequential); .net[-1] is
            # the final Linear. Zero every weight, then bias the logit to -30 so
            # sigmoid(presence) ~ 1e-13 and predict_mask is everywhere < 0.5.
            for param in model.presence_gate.parameters():
                param.zero_()
            model.presence_gate.net[-1].bias.fill_(-30.0)

        outputs = model(image, pose, return_intermediates=True)
        assert float((model.predict_mask(outputs) > 0.5).sum()) == 0.0, "setup failed to empty the prediction"

        cams, _ = cam.grad_cam_levels(model, image, pose, target="pred", topk=16)
        assert len(cams) == 4
        assert all(np.isfinite(c).all() for c in cams)

    def test_model_is_left_in_eval_mode(self, model, batch) -> None:
        image, pose = batch
        cam.grad_cam_levels(model, image, pose)
        assert model.training is False


class TestGradCamProgression:
    def test_returns_a_labelled_map_per_module_boundary(self, model, batch) -> None:
        image, pose = batch
        stages = cam.grad_cam_progression(model, image, pose, level=0)
        labels = [label for label, _ in stages]
        assert labels == [label for label, _ in cam.PROGRESSION_TAPS]
        for _, heat in stages:
            assert heat.shape == (IMG_SIZE, IMG_SIZE)
            assert np.isfinite(heat).all()

    def test_works_at_a_coarser_level(self, model, batch) -> None:
        image, pose = batch
        stages = cam.grad_cam_progression(model, image, pose, level=3)
        assert len(stages) == len(cam.PROGRESSION_TAPS)


class TestPretrainedUnetDegradation:
    def test_missing_tap_falls_back_to_backbone(self, batch) -> None:
        """pretrained_unet has no AER, so asking for it must degrade, not raise."""
        pytest.importorskip("segmentation_models_pytorch")
        cfg = argparse.Namespace(architecture="pretrained_unet", unet_encoder="resnet18",
                                 unet_freeze_encoder=True, no_pretrained=True, dataset="toy")
        model = build_model(cfg).eval()
        image, pose = batch
        cams, tap = cam.grad_cam_levels(model, image, pose, tap="aer")
        assert tap == "backbone"
        assert len(cams) == 4

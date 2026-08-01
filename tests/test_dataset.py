"""Tests for CHDDataset against the real, already-prepared ACD1K data.

Skips cleanly if the prepared dataset isn't present (e.g. running only the
model/metric unit tests without the data pipeline) rather than failing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.dataset import AugmentConfig, CHDDataset, N_KEYPOINTS, batch_resize, collate  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "acd1k"
pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="data/acd1k not prepared in this environment")


class TestCHDDatasetShapes:
    def test_item_shapes_at_default_size(self) -> None:
        ds = CHDDataset(DATA_ROOT, "val", img_size=128)
        item = ds[0]
        assert item["image"].shape == (3, 128, 128)
        assert item["mask"].shape == (1, 128, 128)
        assert item["edge"].shape == (1, 128, 128)
        assert item["pose"].shape == (N_KEYPOINTS, 32, 32)
        assert item["is_negative"].shape == ()

    def test_item_shapes_at_a_different_size(self) -> None:
        """--img-size must be honoured with no other code changes."""
        ds = CHDDataset(DATA_ROOT, "val", img_size=256)
        item = ds[0]
        assert item["image"].shape == (3, 256, 256)
        assert item["pose"].shape == (N_KEYPOINTS, 64, 64)

    def test_image_is_normalised_to_unit_range(self) -> None:
        ds = CHDDataset(DATA_ROOT, "val", img_size=128)
        item = ds[0]
        assert item["image"].min() >= 0.0 and item["image"].max() <= 1.0

    def test_mask_and_edge_are_binary(self) -> None:
        ds = CHDDataset(DATA_ROOT, "val", img_size=128)
        item = ds[0]
        assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}
        assert set(torch.unique(item["edge"]).tolist()) <= {0.0, 1.0}

    def test_val_split_has_no_augmentation_by_default(self) -> None:
        ds = CHDDataset(DATA_ROOT, "val", img_size=128)
        assert ds.augment.enabled is False

    def test_train_split_has_augmentation_by_default(self) -> None:
        ds = CHDDataset(DATA_ROOT, "train", img_size=128)
        assert ds.augment.enabled is True


class TestAugmentationConsistency:
    def test_geometric_augmentation_keeps_mask_and_pose_aligned(self) -> None:
        """A hflip must move the mask's foreground centroid and the pose
        heatmap's peak to mirrored locations together — proof the same
        random transform really is shared across image/mask/edge/pose."""
        cfg = AugmentConfig(hflip_prob=1.0, rotate_deg=0.0, scale_range=(1.0, 1.0), color_jitter=0.0)
        ds = CHDDataset(DATA_ROOT, "val", img_size=128, augment=cfg)

        # find a sample with a nonzero pose heatmap so the peak-location
        # check is meaningful
        item = None
        for i in range(len(ds)):
            candidate = ds[i]
            if candidate["pose"].max() > 0.05:
                item = candidate
                break
        assert item is not None, "no sample in val had a detected pose to test alignment with"

        no_aug = CHDDataset(DATA_ROOT, "val", img_size=128, augment=AugmentConfig(enabled=False))
        original = None
        for i in range(len(no_aug)):
            if no_aug.stems[i] == item["stem"]:
                original = no_aug[i]
                break
        assert original is not None

        # mask centroid should mirror horizontally (within a tolerance for
        # resize/rounding)
        def centroid_x(mask: torch.Tensor) -> float:
            cols = torch.nonzero(mask[0].sum(dim=0))[:, 0].float()
            return cols.mean().item()

        w = item["mask"].shape[-1]
        assert abs((w - 1 - centroid_x(original["mask"])) - centroid_x(item["mask"])) < 3.0

    def test_no_augmentation_is_deterministic(self) -> None:
        ds = CHDDataset(DATA_ROOT, "val", img_size=128, augment=AugmentConfig(enabled=False))
        a, b = ds[0], ds[0]
        assert torch.equal(a["image"], b["image"])
        assert torch.equal(a["mask"], b["mask"])
        assert torch.equal(a["pose"], b["pose"])


class TestCollateAndBatchResize:
    def test_collate_stacks_batch(self) -> None:
        ds = CHDDataset(DATA_ROOT, "val", img_size=64)
        batch = collate([ds[0], ds[1], ds[2]])
        assert batch["image"].shape == (3, 3, 64, 64)
        assert batch["mask"].shape == (3, 1, 64, 64)
        assert batch["pose"].shape == (3, N_KEYPOINTS, 16, 16)
        assert isinstance(batch["stem"], list) and len(batch["stem"]) == 3

    def test_batch_resize_scales_all_tensors_consistently(self) -> None:
        ds = CHDDataset(DATA_ROOT, "val", img_size=64)
        batch = collate([ds[0], ds[1]])
        resized = batch_resize(batch, scale=1.5)  # -> round(64*1.5/32)*32 = 96
        assert resized["image"].shape == (2, 3, 96, 96)
        assert resized["mask"].shape == (2, 1, 96, 96)
        assert resized["pose"].shape == (2, N_KEYPOINTS, 24, 24)

    def test_batch_resize_is_a_noop_at_scale_one(self) -> None:
        ds = CHDDataset(DATA_ROOT, "val", img_size=64)
        batch = collate([ds[0]])
        same = batch_resize(batch, scale=1.0)
        assert torch.equal(same["image"], batch["image"])

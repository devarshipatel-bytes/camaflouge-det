"""Tests for chd.eval.predict — the native-resolution scoring protocol.

The point being pinned: training-time validation scores at img_size, but the
COD literature scores against the ground truth at its own resolution. These
tests build a synthetic dataset whose masks are deliberately NOT square and
NOT img_size, so any accidental reversion to img_size scoring shows up as a
shape mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval import predict, runs  # noqa: E402
from chd.models.factory import build_model  # noqa: E402

NATIVE_HW = (90, 140)  # deliberately non-square and != img_size
IMG_SIZE = 64
N_KEYPOINTS = 17

TINY_ARGS = {
    "architecture": "chdnet",
    "backbone": "tiny_test",
    "dataset": "toy",
    "img_size": IMG_SIZE,
    "os_streams": 2,
    "no_pose": False,
}


def make_dataset(root: Path, stems: tuple[str, ...] = ("a", "b"), negatives: tuple[str, ...] = ()) -> None:
    """Minimal on-disk dataset in the canonical prepared layout."""
    for sub in ("images", "masks", "edges", "pose", "splits"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    h, w = NATIVE_HW
    for stem in stems:
        cv2.imwrite(str(root / "images" / f"{stem}.jpg"),
                    np.full((h, w, 3), 120, dtype=np.uint8))
        mask = np.zeros((h, w), dtype=np.uint8)
        if stem not in negatives:
            mask[20:60, 30:90] = 255
        cv2.imwrite(str(root / "masks" / f"{stem}.png"), mask)
        cv2.imwrite(str(root / "edges" / f"{stem}.png"), np.zeros((h, w), dtype=np.uint8))
        np.save(root / "pose" / f"{stem}.npy", np.zeros((N_KEYPOINTS, h, w), dtype=np.float32))
    (root / "splits" / "test.txt").write_text("\n".join(stems) + "\n")
    rows = ["stem,is_negative"] + [f"{s},{int(s in negatives)}" for s in stems]
    (root / "meta.csv").write_text("\n".join(rows) + "\n")


@pytest.fixture()
def bundle(tmp_path: Path) -> runs.RunBundle:
    cfg = argparse.Namespace(**TINY_ARGS, no_pretrained=True)
    model = build_model(cfg)
    return runs.RunBundle(
        name="toy", checkpoint_path=tmp_path / "best.pth", model=model.eval(),
        config=cfg, weights="raw", epoch=0, best_s_alpha=None,
    )


class TestResizeProb:
    def test_resizes_to_the_requested_shape(self) -> None:
        out = predict.resize_prob(np.zeros((10, 10), dtype=np.float32), (25, 40))
        assert out.shape == (25, 40)

    def test_stays_within_unit_range(self) -> None:
        prob = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        out = predict.resize_prob(prob, (33, 41))
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_identity_when_shape_already_matches(self) -> None:
        prob = np.random.default_rng(1).random((12, 9)).astype(np.float32)
        assert np.array_equal(predict.resize_prob(prob, (12, 9)), prob)


class TestPredictRun:
    def test_prediction_and_gt_are_at_native_resolution(self, tmp_path: Path, bundle) -> None:
        make_dataset(tmp_path / "toy")
        items = list(predict.predict_run(bundle, data_root=tmp_path))
        assert len(items) == 2
        for item in items:
            assert item.prob.shape == NATIVE_HW
            assert item.gt.shape == NATIVE_HW

    def test_gt_is_binary(self, tmp_path: Path, bundle) -> None:
        make_dataset(tmp_path / "toy")
        item = next(iter(predict.predict_run(bundle, data_root=tmp_path)))
        assert set(np.unique(item.gt).tolist()) <= {0.0, 1.0}

    def test_probability_is_in_unit_range(self, tmp_path: Path, bundle) -> None:
        make_dataset(tmp_path / "toy")
        item = next(iter(predict.predict_run(bundle, data_root=tmp_path)))
        assert item.prob.min() >= 0.0 and item.prob.max() <= 1.0
        assert 0.0 <= item.presence <= 1.0

    def test_negatives_are_flagged(self, tmp_path: Path, bundle) -> None:
        make_dataset(tmp_path / "toy", stems=("a", "b"), negatives=("b",))
        by_stem = {i.stem: i for i in predict.predict_run(bundle, data_root=tmp_path)}
        assert by_stem["a"].is_negative is False
        assert by_stem["b"].is_negative is True

    def test_limit_truncates(self, tmp_path: Path, bundle) -> None:
        make_dataset(tmp_path / "toy", stems=("a", "b"))
        assert len(list(predict.predict_run(bundle, data_root=tmp_path, limit=1))) == 1

    def test_stems_filter_selects_only_those_images(self, tmp_path: Path, bundle) -> None:
        make_dataset(tmp_path / "toy", stems=("a", "b"))
        items = list(predict.predict_run(bundle, data_root=tmp_path, stems=["b"]))
        assert [i.stem for i in items] == ["b"]

    def test_no_pose_config_zeroes_the_pose_input(self, tmp_path: Path, bundle) -> None:
        """--no-pose runs must be evaluated the way they were trained."""
        make_dataset(tmp_path / "toy")
        bundle.config.no_pose = True
        items = list(predict.predict_run(bundle, data_root=tmp_path))
        assert len(items) == 2

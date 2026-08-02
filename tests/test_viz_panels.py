"""Tests for chd.viz.panels — pure array -> array rendering helpers.

No model and no dataset involved, so these run anywhere. Each helper is
pinned on a hand-checkable input rather than only on output dtype/shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.viz import panels  # noqa: E402


class TestNormalize01:
    def test_maps_range_to_unit_interval(self) -> None:
        out = panels.normalize01(np.array([2.0, 4.0, 6.0]))
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.5)

    def test_constant_input_becomes_all_zeros(self) -> None:
        """A dead channel must not blow up to NaN via division by zero."""
        out = panels.normalize01(np.full((4, 4), 7.0))
        assert np.all(out == 0.0)
        assert np.isfinite(out).all()


class TestChannelHeat:
    def test_collapses_channels_and_normalises(self) -> None:
        t = torch.zeros(3, 5, 5)
        t[:, 2, 2] = 4.0
        out = panels.channel_heat(t)
        assert out.shape == (5, 5)
        assert out[2, 2] == pytest.approx(1.0)
        assert out[0, 0] == pytest.approx(0.0)

    def test_uses_absolute_value_so_negatives_still_register(self) -> None:
        t = torch.zeros(2, 4, 4)
        t[:, 1, 1] = -3.0
        out = panels.channel_heat(t)
        assert out[1, 1] == pytest.approx(1.0)


class TestMaskComposite:
    def test_masked_region_is_brighter_than_background(self) -> None:
        image = np.full((8, 8, 3), 120, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 2:5] = True
        out = panels.mask_composite(image, mask)
        assert out.dtype == np.uint8
        assert out[3, 3].mean() > out[7, 7].mean()

    def test_empty_mask_only_darkens(self) -> None:
        image = np.full((6, 6, 3), 200, dtype=np.uint8)
        out = panels.mask_composite(image, np.zeros((6, 6), dtype=bool))
        assert out.max() < 200


class TestErrorMap:
    def test_false_positive_and_false_negative_get_different_colors(self) -> None:
        image = np.full((6, 6, 3), 100, dtype=np.uint8)
        pred = np.zeros((6, 6), dtype=bool)
        gt = np.zeros((6, 6), dtype=bool)
        pred[1, 1] = True   # false positive
        gt[4, 4] = True     # false negative
        out = panels.error_map(image, pred, gt)
        assert not np.array_equal(out[1, 1], out[4, 4])

    def test_perfect_prediction_has_no_error_colors(self) -> None:
        image = np.full((6, 6, 3), 100, dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=bool)
        mask[2:4, 2:4] = True
        out = panels.error_map(image, mask, mask)
        fp = np.array([213, 94, 0], dtype=np.uint8)
        assert not (out == fp).all(axis=-1).any()


class TestComponentBboxes:
    def test_finds_one_box_per_blob(self) -> None:
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[2:10, 2:10] = 1
        mask[20:30, 25:35] = 1
        boxes = panels.component_bboxes(mask)
        assert len(boxes) == 2

    def test_box_is_tight_and_xyxy_ordered(self) -> None:
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:9, 3:8] = 1
        (x1, y1, x2, y2), = panels.component_bboxes(mask)
        assert (x1, y1, x2, y2) == (3, 5, 8, 9)

    def test_tiny_speckle_is_dropped(self) -> None:
        """Mask noise must not add spurious boxes to the annotated figure."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:40, 10:40] = 1
        mask[90, 90] = 1
        boxes = panels.component_bboxes(mask, min_area_frac=0.001)
        assert len(boxes) == 1

    def test_empty_mask_gives_no_boxes(self) -> None:
        assert panels.component_bboxes(np.zeros((10, 10), dtype=np.uint8)) == []

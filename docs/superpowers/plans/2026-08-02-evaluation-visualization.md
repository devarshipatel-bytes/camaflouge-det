# Evaluation & Visualization Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate one named training run at a time against its dataset's test split and render the qualitative, Grad-CAM, activation and dataset figures the paper needs.

**Architecture:** A new `src/chd/eval/` package holds run resolution, native-resolution inference and reporting; `src/chd/viz/` gains pure rendering helpers and Grad-CAM. Four numbered scripts (`08`–`11`) wire these into CLIs. Every script recovers its own configuration from the checkpoint's stored `args`, because the checkpoints live on a different machine whose run-folder names are not parseable.

**Tech Stack:** PyTorch, numpy, scipy.ndimage, OpenCV (cv2), matplotlib (Agg backend), pytest.

## Global Constraints

- **Never modify** `src/chd/metrics.py`, `train.py`, or any file in `src/chd/models/`. Reported numbers must match the already-tested metric implementation, and `return_intermediates` already exposes every tap needed.
- `src/chd/viz/colors.py` **may** be extended (additive only).
- Every script adds `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))` before importing `chd`, matching `scripts/06`/`07`.
- Every script calls `matplotlib.use("Agg")` before `import matplotlib.pyplot`.
- Every figure is saved as **both** `.png` (dpi 180) and `.svg`.
- Default colormap for Grad-CAM figures is `jet`; raw activation figures use `inferno`.
- Mask metrics aggregate over **positives only**. Presence metrics cover **all** images.
- Figures render masks thresholded at **0.5**; metrics always use the **continuous** probability map. These never share a code path.
- Predictions are scored at the **native ground-truth resolution**, never at `img_size`.
- Weight loading prefers `ckpt["ema"]` when non-empty, else `ckpt["model"]`.
- `--run` is single and required. Only `09`'s `--also-run` is repeatable.
- Type hints use `from __future__ import annotations` and PEP 604 (`X | None`), matching the existing codebase.
- No new third-party dependencies. `scipy`, `cv2`, `matplotlib`, `torch`, `numpy` are already used.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/chd/eval/__init__.py` | Re-export the package's public names |
| `src/chd/eval/runs.py` | `--run <name>` → checkpoint → recovered config + rebuilt model |
| `src/chd/eval/predict.py` | Inference producing native-resolution probability maps |
| `src/chd/eval/report.py` | Metric aggregation, presence metrics, CSV/JSON/MD writers |
| `src/chd/viz/panels.py` | Pure array→array rendering helpers (no torch model involved) |
| `src/chd/viz/cam.py` | Grad-CAM over pyramid levels and module taps |
| `src/chd/viz/colors.py` | *(modify)* add `ERROR_COLOR` |
| `scripts/08_evaluate.py` | Metrics CLI for one run |
| `scripts/09_visualize_predictions.py` | FIG 1–4 |
| `scripts/10_compare_runs.py` | Cross-run comparison charts |
| `scripts/11_dataset_figures.py` | Per-dataset collage + annotated triplets |
| `tests/test_viz_panels.py` | Rendering helper tests |
| `tests/test_eval_runs.py` | Run resolution / config recovery tests |
| `tests/test_eval_report.py` | Aggregation + presence metric tests |
| `tests/test_eval_predict.py` | Native-resolution protocol tests |
| `tests/test_cam.py` | Grad-CAM tests |

---

## Task 1: Rendering helpers (`chd.viz.panels`)

Pure functions on numpy arrays. Built first because every figure task depends on
them, and they are the easiest thing to test without a model.

**Files:**
- Create: `src/chd/viz/panels.py`
- Modify: `src/chd/viz/colors.py` (append `ERROR_COLOR`)
- Test: `tests/test_viz_panels.py`

**Interfaces:**
- Consumes: `chd.viz.colors` (existing `INK`, `DATASET_LABEL`).
- Produces:
  - `normalize01(arr: np.ndarray) -> np.ndarray` — min-max to [0,1]; all-zeros when constant.
  - `channel_heat(t: torch.Tensor) -> np.ndarray` — `(C,H,W)` → normalized `(H,W)` float32 of `mean|activation|`.
  - `mask_composite(image_rgb: np.ndarray, mask_bin: np.ndarray, dim: float = 0.3, tint_alpha: float = 0.85) -> np.ndarray` — uint8 RGB.
  - `overlay_heat(image_rgb: np.ndarray, heat01: np.ndarray, cmap: str = "jet", alpha: float = 0.5) -> np.ndarray` — uint8 RGB.
  - `error_map(image_rgb: np.ndarray, pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray` — uint8 RGB.
  - `component_bboxes(mask_bin: np.ndarray, min_area_frac: float = 0.001) -> list[tuple[int,int,int,int]]` — `(x1,y1,x2,y2)`.
  - `save_figure(fig, out_dir: Path, name: str, dpi: int = 180) -> None`.
  - `blank_panel(ax, text: str) -> None` — renders an explanatory placeholder for an unavailable column.

- [ ] **Step 1: Append the error palette to `src/chd/viz/colors.py`**

Add at the end of the file:

```python
#: False-positive / false-negative / true-positive colors for the error panel
#: in the qualitative figure. FP is Okabe-Ito vermillion, FN is the same blue
#: already used for acd1k/train, TP is a neutral gray so correct pixels never
#: compete for attention with mistakes.
ERROR_COLOR = {"fp": "#D55E00", "fn": "#0072B2", "tp": "#B0B0B0"}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_viz_panels.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python -m pytest tests/test_viz_panels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chd.viz.panels'`

- [ ] **Step 4: Implement `src/chd/viz/panels.py`**

```python
"""Rendering helpers shared by every evaluation figure.

Pure numpy in, numpy out — no model, no dataset, no matplotlib state (except
``save_figure``, which only writes). Keeping them here rather than inline in
the scripts is what lets the figure scripts stay thin and lets the rendering
rules be unit-tested without a checkpoint.

Two conventions this module encodes, both taken from the COD literature the
paper is compared against:

- ``mask_composite`` renders a mask as a bright silhouette over a *darkened*
  copy of the image, so target shape and scene context are legible at once.
- ``error_map`` separates false positives from false negatives by hue, so a
  figure shows *how* a prediction is wrong, not merely that it is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import to_rgb
from scipy import ndimage

from chd.viz.colors import ERROR_COLOR

_CONNECTIVITY_8 = np.ones((3, 3), dtype=int)


def normalize01(arr: np.ndarray) -> np.ndarray:
    """Min-max an array into [0, 1]; a constant array becomes all zeros.

    The all-zeros fallback matters: a dead activation channel would otherwise
    divide by zero and render as NaN, which matplotlib silently draws as
    blank rather than flagging.
    """
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def channel_heat(t: torch.Tensor) -> np.ndarray:
    """``(C, H, W)`` -> normalized ``(H, W)`` mean-absolute activation.

    This is an *unsigned magnitude* map: it shows what a layer responds to,
    not what it decided. For a target-specific map use ``chd.viz.cam``.
    """
    return normalize01(t.detach().abs().mean(dim=0).float().cpu().numpy())


def mask_composite(
    image_rgb: np.ndarray, mask_bin: np.ndarray, dim: float = 0.3, tint_alpha: float = 0.85,
) -> np.ndarray:
    """Bright mask silhouette over a darkened image (the COD-paper convention)."""
    image = np.asarray(image_rgb, dtype=np.float32)
    mask = (np.asarray(mask_bin) > 0.5).astype(np.float32)[..., None]
    darkened = image * dim
    bright = image * (1.0 - tint_alpha) + 255.0 * tint_alpha
    out = darkened * (1.0 - mask) + bright * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_heat(
    image_rgb: np.ndarray, heat01: np.ndarray, cmap: str = "jet", alpha: float = 0.5,
) -> np.ndarray:
    """Blend a [0,1] heatmap over an image using ``cmap``."""
    import matplotlib.pyplot as plt

    heat = np.clip(np.asarray(heat01, dtype=np.float32), 0.0, 1.0)
    colored = plt.get_cmap(cmap)(heat)[..., :3] * 255.0
    image = np.asarray(image_rgb, dtype=np.float32)
    return np.clip(image * (1.0 - alpha) + colored * alpha, 0, 255).astype(np.uint8)


def error_map(image_rgb: np.ndarray, pred_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
    """Grayscale image with FP and FN painted in distinct hues, TP neutral."""
    image = np.asarray(image_rgb, dtype=np.float32)
    gray = image.mean(axis=2)
    out = np.repeat(gray[..., None], 3, axis=2) * 0.55

    pred = np.asarray(pred_bin) > 0.5
    gt = np.asarray(gt_bin) > 0.5
    out[pred & gt] = np.array(to_rgb(ERROR_COLOR["tp"])) * 255.0
    out[pred & ~gt] = np.array(to_rgb(ERROR_COLOR["fp"])) * 255.0
    out[~pred & gt] = np.array(to_rgb(ERROR_COLOR["fn"])) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def component_bboxes(mask_bin: np.ndarray, min_area_frac: float = 0.001) -> list[tuple[int, int, int, int]]:
    """One tight ``(x1, y1, x2, y2)`` box per 8-connected blob, small blobs dropped.

    8-connectivity matches ``chd.data.manifest.count_components`` so a mask's
    reported component count and its drawn box count cannot disagree.
    ``min_area_frac`` exists because a single stray pixel would otherwise draw
    a full box and make a clean mask look noisy.
    """
    solid = np.asarray(mask_bin) > 0
    labels, n = ndimage.label(solid, structure=_CONNECTIVITY_8)
    if n == 0:
        return []
    total = float(solid.size)
    boxes: list[tuple[int, int, int, int]] = []
    for index, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        if float((labels[slices] == index).sum()) / total < min_area_frac:
            continue
        y_slice, x_slice = slices
        boxes.append((int(x_slice.start), int(y_slice.start), int(x_slice.stop), int(y_slice.stop)))
    return boxes


def blank_panel(ax, text: str) -> None:
    """Mark a column this architecture does not have, instead of drawing zeros.

    A zero-filled heatmap looks like a real measurement that happens to be
    empty; this makes the absence explicit.
    """
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=7,
            color="#595959", transform=ax.transAxes, wrap=True)
    ax.set_facecolor("#f5f5f5")


def save_figure(fig, out_dir: Path, name: str, dpi: int = 180) -> None:
    """Write ``<name>.png`` and ``<name>.svg``, matching scripts/06 and 07."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_viz_panels.py -v`
Expected: PASS, 12 tests

- [ ] **Step 6: Confirm nothing existing broke**

Run: `python -m pytest tests/ -m "not slow" -q`
Expected: PASS (same count as before, plus the 12 new)

- [ ] **Step 7: Commit**

```bash
git add src/chd/viz/panels.py src/chd/viz/colors.py tests/test_viz_panels.py
git commit -m "Add shared rendering helpers for the evaluation figures

Pure array -> array so they are testable without a checkpoint, which
matters because the trained weights live on a different machine.

Two conventions are encoded here rather than in the figure scripts:
mask_composite renders a mask as a bright silhouette over a darkened
image (the COD-paper convention, so our qualitative figure is directly
comparable to the baselines), and error_map separates false positives
from false negatives by hue so a figure shows how a prediction is wrong.

normalize01 returns zeros for a constant input rather than dividing by
zero — a dead activation channel would otherwise render as NaN, which
matplotlib draws as blank instead of flagging."
```

---

## Task 2: Run resolution (`chd.eval.runs`)

**Files:**
- Create: `src/chd/eval/__init__.py`, `src/chd/eval/runs.py`
- Test: `tests/test_eval_runs.py`

**Interfaces:**
- Consumes: `chd.models.factory.build_model` (existing).
- Produces:
  - `DEFAULT_RUNS_ROOT: Path` = `Path("runs")`
  - `RunBundle` dataclass with fields `name: str`, `checkpoint_path: Path`, `model: nn.Module`, `config: argparse.Namespace`, `weights: str`, `epoch: int | None`, `best_s_alpha: float | None`
  - `available_runs(runs_root: Path) -> list[str]`
  - `resolve_checkpoint(run: str, runs_root: Path = DEFAULT_RUNS_ROOT, prefer: str = "best") -> Path`
  - `config_from_checkpoint(ckpt: dict, overrides: dict | None = None) -> argparse.Namespace`
  - `load_run(run: str, runs_root: Path = DEFAULT_RUNS_ROOT, device: str = "cpu", prefer: str = "best", overrides: dict | None = None, checkpoint: Path | None = None) -> RunBundle`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_runs.py`:

```python
"""Tests for chd.eval.runs — recovering a run's configuration from its checkpoint.

The real checkpoints live on a different machine, so every test here builds a
synthetic checkpoint with the same structure train.py writes (see train.py's
checkpoint dict, which stores "args": vars(args)). That structure is the
contract these tests pin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval import runs  # noqa: E402
from chd.models.factory import build_model  # noqa: E402

TINY_ARGS = {
    "architecture": "chdnet",
    "backbone": "tiny_test",
    "dataset": "acd1k",
    "img_size": 64,
    "os_streams": 2,
    "no_pose": False,
    "data_root": Path("data"),
}


def write_checkpoint(path: Path, *, with_ema: bool, args: dict | None = None) -> None:
    """Build a real state_dict from the tiny_test backbone so load_state_dict is exercised."""
    cfg = argparse.Namespace(**{**TINY_ARGS, **(args or {})}, no_pretrained=True)
    model = build_model(cfg)
    state = model.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": 7,
        "best_s_alpha": 0.5,
        "model": state,
        "ema": {k: v.clone() for k, v in state.items()} if with_ema else None,
        "args": {**TINY_ARGS, **(args or {})},
    }, path)


class TestResolveCheckpoint:
    def test_prefers_best_over_last(self, tmp_path: Path) -> None:
        (tmp_path / "r1").mkdir()
        (tmp_path / "r1" / "best.pth").touch()
        (tmp_path / "r1" / "last.pth").touch()
        assert runs.resolve_checkpoint("r1", tmp_path).name == "best.pth"

    def test_falls_back_to_last_when_best_missing(self, tmp_path: Path) -> None:
        (tmp_path / "r1").mkdir()
        (tmp_path / "r1" / "last.pth").touch()
        assert runs.resolve_checkpoint("r1", tmp_path).name == "last.pth"

    def test_missing_run_lists_available_names(self, tmp_path: Path) -> None:
        (tmp_path / "camo-human-final").mkdir()
        (tmp_path / "camo-human-final" / "best.pth").touch()
        with pytest.raises(FileNotFoundError, match="camo-human-final"):
            runs.resolve_checkpoint("nope", tmp_path)

    def test_run_dir_without_any_checkpoint_is_an_error(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            runs.resolve_checkpoint("empty", tmp_path)


class TestAvailableRuns:
    def test_lists_only_dirs_holding_a_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / "good").mkdir()
        (tmp_path / "good" / "best.pth").touch()
        (tmp_path / "bare").mkdir()
        assert runs.available_runs(tmp_path) == ["good"]

    def test_missing_root_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert runs.available_runs(tmp_path / "absent") == []


class TestConfigFromCheckpoint:
    def test_recovers_stored_args(self) -> None:
        cfg = runs.config_from_checkpoint({"args": dict(TINY_ARGS)})
        assert cfg.dataset == "acd1k"
        assert cfg.img_size == 64
        assert cfg.backbone == "tiny_test"

    def test_overrides_win(self) -> None:
        cfg = runs.config_from_checkpoint({"args": dict(TINY_ARGS)}, overrides={"img_size": 128})
        assert cfg.img_size == 128

    def test_none_valued_overrides_are_ignored(self) -> None:
        """Unset CLI flags arrive as None and must not clobber the stored config."""
        cfg = runs.config_from_checkpoint({"args": dict(TINY_ARGS)}, overrides={"img_size": None})
        assert cfg.img_size == 64

    def test_missing_args_is_an_actionable_error(self) -> None:
        with pytest.raises(KeyError, match="args"):
            runs.config_from_checkpoint({"model": {}})

    def test_pretraining_is_disabled_so_no_weights_are_downloaded(self) -> None:
        """Checkpoint weights overwrite the backbone anyway; downloading ImageNet
        first would only cost time and require network access."""
        cfg = runs.config_from_checkpoint({"args": dict(TINY_ARGS)})
        assert cfg.no_pretrained is True


class TestLoadRun:
    def test_prefers_ema_weights(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path / "r1" / "best.pth", with_ema=True)
        bundle = runs.load_run("r1", runs_root=tmp_path)
        assert bundle.weights == "ema"

    def test_falls_back_to_raw_weights(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path / "r1" / "best.pth", with_ema=False)
        bundle = runs.load_run("r1", runs_root=tmp_path)
        assert bundle.weights == "raw"

    def test_model_is_in_eval_mode_and_carries_metadata(self, tmp_path: Path) -> None:
        write_checkpoint(tmp_path / "r1" / "best.pth", with_ema=True)
        bundle = runs.load_run("r1", runs_root=tmp_path)
        assert bundle.model.training is False
        assert bundle.epoch == 7
        assert bundle.best_s_alpha == pytest.approx(0.5)
        assert bundle.name == "r1"

    def test_folder_name_does_not_determine_dataset(self, tmp_path: Path) -> None:
        """The real runs include 'camo-human-final' for dataset 'camo_human'."""
        write_checkpoint(tmp_path / "camo-human-final" / "best.pth",
                         with_ema=True, args={"dataset": "camo_human"})
        bundle = runs.load_run("camo-human-final", runs_root=tmp_path)
        assert bundle.config.dataset == "camo_human"

    def test_explicit_checkpoint_path_bypasses_run_lookup(self, tmp_path: Path) -> None:
        path = tmp_path / "somewhere" / "weights.pth"
        write_checkpoint(path, with_ema=True)
        bundle = runs.load_run("ignored", runs_root=tmp_path / "absent", checkpoint=path)
        assert bundle.checkpoint_path == path
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_eval_runs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chd.eval'`

- [ ] **Step 3: Create `src/chd/eval/__init__.py`**

```python
from chd.eval.runs import (
    DEFAULT_RUNS_ROOT,
    RunBundle,
    available_runs,
    config_from_checkpoint,
    load_run,
    resolve_checkpoint,
)

__all__ = [
    "DEFAULT_RUNS_ROOT",
    "RunBundle",
    "available_runs",
    "config_from_checkpoint",
    "load_run",
    "resolve_checkpoint",
]
```

- [ ] **Step 4: Implement `src/chd/eval/runs.py`**

```python
"""Turn a run *name* into a loaded, correctly-configured model.

Why this module exists: the trained checkpoints live on a different machine
whose run folders are named inconsistently — ``camo-human-final`` holds the
``camo_human`` dataset, and ``acd1k``/``acd1k2`` are two runs of one dataset.
Nothing about the layout can be parsed reliably.

What saves us is that ``train.py`` stores ``"args": vars(args)`` in every
checkpoint, so dataset, architecture, backbone, ``os_streams``,
``unet_encoder``, ``img_size`` and ``no_pose`` are all recoverable from the
file itself. Every evaluation and figure script goes through here, so none of
them ever needs a re-typed flag or a filename convention.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from chd.models.factory import build_model

DEFAULT_RUNS_ROOT = Path("runs")
CHECKPOINT_NAMES = ("best.pth", "last.pth")


@dataclass
class RunBundle:
    """Everything downstream code needs about one loaded run."""

    name: str
    checkpoint_path: Path
    model: nn.Module
    config: argparse.Namespace
    weights: str  # "ema" or "raw"
    epoch: int | None
    best_s_alpha: float | None


def available_runs(runs_root: str | Path) -> list[str]:
    """Names of subdirectories that actually hold a checkpoint."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and any((p / name).exists() for name in CHECKPOINT_NAMES)
    )


def resolve_checkpoint(
    run: str, runs_root: str | Path = DEFAULT_RUNS_ROOT, prefer: str = "best",
) -> Path:
    """``<runs_root>/<run>/best.pth``, falling back to ``last.pth``."""
    root = Path(runs_root)
    run_dir = root / run
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"no run directory {run_dir}. Available runs under {root}: {available_runs(root) or '(none)'}"
        )
    order = CHECKPOINT_NAMES if prefer == "best" else tuple(reversed(CHECKPOINT_NAMES))
    for name in order:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{run_dir} holds none of {list(CHECKPOINT_NAMES)}")


def config_from_checkpoint(ckpt: dict, overrides: dict | None = None) -> argparse.Namespace:
    """Rebuild the training ``argparse.Namespace`` from a checkpoint.

    ``no_pretrained`` is forced on: the checkpoint's own weights overwrite the
    backbone immediately afterwards, so downloading ImageNet weights first
    would only cost time and require network access on a machine that may not
    have it.
    """
    stored = ckpt.get("args")
    if not stored:
        raise KeyError(
            "checkpoint has no 'args' entry, so its architecture cannot be recovered; "
            "pass --dataset and --architecture explicitly"
        )
    merged = dict(stored)
    for key, value in (overrides or {}).items():
        if value is not None:  # unset CLI flags must not clobber stored config
            merged[key] = value
    merged["no_pretrained"] = True
    return argparse.Namespace(**merged)


def load_run(
    run: str,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    device: str = "cpu",
    prefer: str = "best",
    overrides: dict | None = None,
    checkpoint: str | Path | None = None,
) -> RunBundle:
    """Load a run by name (or by explicit ``checkpoint`` path) into a ``RunBundle``."""
    path = Path(checkpoint) if checkpoint else resolve_checkpoint(run, runs_root, prefer)
    # weights_only=False: written by this repo's own train.py, never a download,
    # and it stores an argparse Namespace dict that weights_only rejects.
    ckpt = torch.load(path, map_location=device, weights_only=False)

    config = config_from_checkpoint(ckpt, overrides)
    model = build_model(config)

    if ckpt.get("ema"):
        model.load_state_dict(ckpt["ema"])
        weights = "ema"
    else:
        model.load_state_dict(ckpt["model"])
        weights = "raw"

    model.to(device).eval()
    return RunBundle(
        name=run,
        checkpoint_path=path,
        model=model,
        config=config,
        weights=weights,
        epoch=ckpt.get("epoch"),
        best_s_alpha=ckpt.get("best_s_alpha"),
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_eval_runs.py -v`
Expected: PASS, 15 tests

- [ ] **Step 6: Commit**

```bash
git add src/chd/eval/__init__.py src/chd/eval/runs.py tests/test_eval_runs.py
git commit -m "Recover a run's config from its checkpoint instead of its path

The trained checkpoints live on a different machine whose run folders are
named inconsistently: camo-human-final holds the camo_human dataset, and
acd1k/acd1k2 are two runs of one dataset. Nothing about the layout parses.

train.py already stores vars(args) in every checkpoint, so dataset,
architecture, backbone, os_streams and img_size are all recoverable from
the file. That also fixes a silent hazard: acd1k trained at img_size 640,
not the 352 CLI default, so any script assuming the default would report
degraded metrics without complaining.

EMA weights are preferred over raw to match what run_validation used, so
figures reflect the same weights the reported best_s_alpha came from."
```

---

## Task 3: Native-resolution inference (`chd.eval.predict`)

**Files:**
- Create: `src/chd/eval/predict.py`
- Modify: `src/chd/eval/__init__.py` (add exports)
- Test: `tests/test_eval_predict.py`

**Interfaces:**
- Consumes: `RunBundle` from Task 2; `chd.data.dataset.CHDDataset`, `AugmentConfig`; `chd.data.manifest.load_gray`.
- Produces:
  - `Prediction` dataclass: `stem: str`, `prob: np.ndarray` (native H×W float32 in [0,1]), `gt: np.ndarray` (native H×W float32 in {0,1}), `presence: float`, `is_negative: bool`
  - `resize_prob(prob: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray`
  - `predict_run(bundle: RunBundle, split: str = "test", data_root: Path | None = None, device: str = "cpu", limit: int | None = None, stems: list[str] | None = None) -> Iterator[Prediction]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_predict.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_eval_predict.py -v`
Expected: FAIL — `ImportError: cannot import name 'predict' from 'chd.eval'`

- [ ] **Step 3: Implement `src/chd/eval/predict.py`**

```python
"""Inference that produces probability maps at the ground truth's own resolution.

This is the one place the evaluation protocol deliberately differs from
``train.py``'s ``run_validation``. Training-time validation scores at
``img_size`` because it only needs a comparable number epoch to epoch. The
COD/SOD literature — and therefore the paper's comparison tables — scores
against the ground-truth mask at its **native** resolution. Predicting at
``img_size`` and then resizing the probability map back up is what makes our
numbers comparable to the published baselines.

The mask is re-read from disk rather than taken from the dataset item, because
``CHDDataset`` resizes it to ``img_size`` on load.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from chd.data.dataset import AugmentConfig, CHDDataset
from chd.data.manifest import load_gray
from chd.eval.runs import RunBundle


@dataclass
class Prediction:
    """One test image's prediction, at the ground truth's native resolution."""

    stem: str
    prob: np.ndarray  # (H, W) float32 in [0, 1]
    gt: np.ndarray  # (H, W) float32 in {0, 1}
    presence: float
    is_negative: bool


def resize_prob(prob: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a probability map to ``(H, W)``, clipped to [0, 1]."""
    height, width = size_hw
    if prob.shape == (height, width):
        return prob
    resized = cv2.resize(prob, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def predict_run(
    bundle: RunBundle,
    split: str = "test",
    data_root: str | Path | None = None,
    device: str = "cpu",
    limit: int | None = None,
    stems: list[str] | None = None,
) -> Iterator[Prediction]:
    """Yield one ``Prediction`` per image, streaming so memory stays flat.

    Streaming matters: the combined test split is 1150 images, and holding
    every native-resolution probability map and mask in memory at once would
    cost several GB.
    """
    root = Path(data_root or getattr(bundle.config, "data_root", "data")) / bundle.config.dataset
    dataset = CHDDataset(root, split, img_size=bundle.config.img_size,
                         augment=AugmentConfig(enabled=False))

    wanted = set(stems) if stems else None
    indices = [i for i, stem in enumerate(dataset.stems) if wanted is None or stem in wanted]
    if limit is not None:
        indices = indices[:limit]

    zero_pose = bool(getattr(bundle.config, "no_pose", False))

    for index in indices:
        item = dataset[index]
        stem = item["stem"]

        image = item["image"].unsqueeze(0).to(device)
        pose = item["pose"].unsqueeze(0)
        if zero_pose:
            pose = torch.zeros_like(pose)
        pose = pose.to(device)

        outputs = bundle.model(image, pose)
        prob = bundle.model.predict_mask(outputs)[0, 0].float().cpu().numpy()
        presence = float(torch.sigmoid(outputs["presence_logit"]).flatten()[0].item())

        gt = (load_gray(root / "masks" / f"{stem}.png") > 127).astype(np.float32)
        yield Prediction(
            stem=stem,
            prob=resize_prob(prob.astype(np.float32), gt.shape),
            gt=gt,
            presence=presence,
            is_negative=bool(dataset.is_negative[stem]),
        )
```

- [ ] **Step 4: Add exports to `src/chd/eval/__init__.py`**

Replace the file with:

```python
from chd.eval.predict import Prediction, predict_run, resize_prob
from chd.eval.runs import (
    DEFAULT_RUNS_ROOT,
    RunBundle,
    available_runs,
    config_from_checkpoint,
    load_run,
    resolve_checkpoint,
)

__all__ = [
    "DEFAULT_RUNS_ROOT",
    "Prediction",
    "RunBundle",
    "available_runs",
    "config_from_checkpoint",
    "load_run",
    "predict_run",
    "resize_prob",
    "resolve_checkpoint",
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_eval_predict.py -v`
Expected: PASS, 10 tests

- [ ] **Step 6: Commit**

```bash
git add src/chd/eval/predict.py src/chd/eval/__init__.py tests/test_eval_predict.py
git commit -m "Score predictions at the ground truth's native resolution

train.py's run_validation scores at img_size, which is fine for tracking
progress epoch to epoch but is not how the COD literature reports numbers.
The paper's comparison tables are only meaningful if we do what the
baselines do: predict at img_size, resize the probability map back to the
mask's own resolution, and score there.

The mask is re-read from disk because CHDDataset resizes it on load, so the
item's mask is already at img_size and cannot serve as native ground truth.

Predictions stream rather than accumulate — the combined test split is 1150
images and holding every native-resolution map in memory would cost GBs.

Tests use a deliberately non-square 90x140 synthetic dataset so any
reversion to img_size scoring fails on shape rather than passing quietly."
```

---

## Task 4: Metric aggregation and reporting (`chd.eval.report`)

**Files:**
- Create: `src/chd/eval/report.py`
- Modify: `src/chd/eval/__init__.py`
- Test: `tests/test_eval_report.py`

**Interfaces:**
- Consumes: `chd.metrics.evaluate_all` (existing, unmodified); `Prediction` from Task 3.
- Produces:
  - `MASK_METRICS: tuple[str, ...]` — the 11 keys `evaluate_all` returns
  - `metric_row(pred: Prediction) -> dict` — one flat CSV row
  - `presence_metrics(presence_probs: list[float], is_negative: list[bool], threshold: float = 0.5) -> dict`
  - `aggregate(rows: list[dict]) -> dict`
  - `write_per_image_csv(rows: list[dict], path: Path) -> None`
  - `write_failures_csv(rows: list[dict], path: Path) -> None`
  - `write_summary_json(summary: dict, path: Path) -> None`
  - `write_metrics_md(summary: dict, path: Path) -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_eval_report.py`:

```python
"""Tests for chd.eval.report — aggregation rules that affect published numbers.

The rule being pinned hardest: mask metrics average over positives only.
An empty ground truth sends s_measure down its y == 0 branch, where it
returns 1 - pred.mean() — a presence score, not a segmentation score.
Averaging that in would inflate S_alpha, and camo_human has 1024 negatives.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval import report  # noqa: E402
from chd.eval.predict import Prediction  # noqa: E402


def make_pred(stem: str, *, is_negative: bool = False, presence: float = 0.9,
              perfect: bool = True) -> Prediction:
    gt = np.zeros((32, 32), dtype=np.float32)
    if not is_negative:
        gt[8:24, 8:24] = 1.0
    prob = gt.copy() if perfect else np.zeros_like(gt)
    return Prediction(stem=stem, prob=prob, gt=gt, presence=presence, is_negative=is_negative)


class TestMetricRow:
    def test_row_carries_every_mask_metric_plus_identity(self) -> None:
        row = report.metric_row(make_pred("a"))
        for key in report.MASK_METRICS:
            assert key in row
        assert row["stem"] == "a"
        assert row["is_negative"] == 0
        assert row["height"] == 32 and row["width"] == 32

    def test_perfect_prediction_scores_perfectly(self) -> None:
        row = report.metric_row(make_pred("a"))
        assert row["IoU"] == pytest.approx(1.0)
        assert row["MAE"] == pytest.approx(0.0)


class TestPresenceMetrics:
    def test_hand_computed_confusion_matrix(self) -> None:
        """probs .9/.8 on positives, .1/.6 on negatives, threshold .5:
        TP=2, FN=0, FP=1 (the .6 negative), TN=1."""
        out = report.presence_metrics([0.9, 0.8, 0.1, 0.6], [False, False, True, True])
        assert out["presence_tp"] == 2
        assert out["presence_fn"] == 0
        assert out["presence_fp"] == 1
        assert out["presence_tn"] == 1
        assert out["presence_accuracy"] == pytest.approx(0.75)
        assert out["presence_precision"] == pytest.approx(2 / 3)
        assert out["presence_recall"] == pytest.approx(1.0)
        assert out["presence_f1"] == pytest.approx(0.8)

    def test_auc_is_one_for_perfectly_separated_scores(self) -> None:
        out = report.presence_metrics([0.9, 0.8, 0.2, 0.1], [False, False, True, True])
        assert out["presence_auc"] == pytest.approx(1.0)

    def test_auc_is_none_without_both_classes(self) -> None:
        out = report.presence_metrics([0.9, 0.8], [False, False])
        assert out["presence_auc"] is None

    def test_empty_input_does_not_divide_by_zero(self) -> None:
        out = report.presence_metrics([], [])
        assert out["presence_accuracy"] is None
        assert np.isfinite(out["presence_tp"])


class TestAggregate:
    def test_mask_means_exclude_negatives(self) -> None:
        """The negative row is a perfect all-zero prediction on an empty GT,
        which scores IoU 1.0. Including it would mask a bad positive."""
        rows = [
            report.metric_row(make_pred("pos", perfect=False)),
            report.metric_row(make_pred("neg", is_negative=True)),
        ]
        out = report.aggregate(rows)
        assert out["n_positives"] == 1
        assert out["n_negatives"] == 1
        assert out["mask"]["IoU"] == pytest.approx(rows[0]["IoU"])

    def test_presence_block_covers_all_images(self) -> None:
        rows = [
            report.metric_row(make_pred("pos", presence=0.9)),
            report.metric_row(make_pred("neg", is_negative=True, presence=0.1)),
        ]
        out = report.aggregate(rows)
        assert out["presence"]["presence_accuracy"] == pytest.approx(1.0)

    def test_all_negative_split_reports_no_mask_metrics(self) -> None:
        rows = [report.metric_row(make_pred("n1", is_negative=True))]
        out = report.aggregate(rows)
        assert out["mask"] == {}
        assert out["n_positives"] == 0


class TestWriters:
    def test_per_image_csv_round_trips(self, tmp_path: Path) -> None:
        rows = [report.metric_row(make_pred("a")), report.metric_row(make_pred("b"))]
        path = tmp_path / "per_image.csv"
        report.write_per_image_csv(rows, path)
        with path.open() as fh:
            back = list(csv.DictReader(fh))
        assert [r["stem"] for r in back] == ["a", "b"]
        assert "S_alpha" in back[0]

    def test_failures_are_sorted_worst_first_and_exclude_negatives(self, tmp_path: Path) -> None:
        rows = [
            report.metric_row(make_pred("good")),
            report.metric_row(make_pred("bad", perfect=False)),
            report.metric_row(make_pred("neg", is_negative=True)),
        ]
        path = tmp_path / "failures.csv"
        report.write_failures_csv(rows, path)
        with path.open() as fh:
            back = list(csv.DictReader(fh))
        assert [r["stem"] for r in back] == ["bad", "good"]

    def test_summary_json_is_valid_json(self, tmp_path: Path) -> None:
        summary = report.aggregate([report.metric_row(make_pred("a"))])
        summary["run"] = "toy"
        path = tmp_path / "summary.json"
        report.write_summary_json(summary, path)
        assert json.loads(path.read_text())["run"] == "toy"

    def test_metrics_md_names_the_metrics(self, tmp_path: Path) -> None:
        summary = report.aggregate([report.metric_row(make_pred("a"))])
        summary["run"] = "toy"
        path = tmp_path / "metrics.md"
        report.write_metrics_md(summary, path)
        text = path.read_text()
        assert "S_alpha" in text and "toy" in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_eval_report.py -v`
Expected: FAIL — `ImportError: cannot import name 'report' from 'chd.eval'`

- [ ] **Step 3: Implement `src/chd/eval/report.py`**

```python
"""Aggregate per-image metrics into the numbers and files the paper needs.

Two aggregation rules here are decisions, not conveniences:

1. **Mask metrics average over positives only.** An empty ground truth sends
   ``s_measure`` down its ``y == 0`` branch, which returns ``1 - pred.mean()``
   — a presence score, not a segmentation score. Averaging that together with
   real segmentation scores would inflate S_alpha, and ``camo_human`` has
   1024 negatives.
2. **Presence metrics cover every image.** That is the whole point of the
   presence gate, and it is what the paper's target-free-frame claim rests on.

AUC is computed from ranks (Mann-Whitney U) using ``scipy.stats.rankdata`` so
ties are handled correctly and no new dependency is needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from chd.eval.predict import Prediction
from chd.metrics import evaluate_all

#: Keys returned by ``chd.metrics.evaluate_all``, in report order.
MASK_METRICS = (
    "MAE", "F_beta_mean", "F_beta_max", "F_beta_adaptive", "S_alpha",
    "E_phi_mean", "E_phi_max", "E_phi_adaptive", "F_bd", "IoU", "Dice",
)

ROW_FIELDS = ("stem", "is_negative", "presence_prob", "height", "width", *MASK_METRICS)


def metric_row(pred: Prediction) -> dict:
    """One flat CSV row: identity, presence, and every mask metric."""
    scores = evaluate_all(pred.prob, pred.gt)
    height, width = pred.gt.shape
    return {
        "stem": pred.stem,
        "is_negative": int(pred.is_negative),
        "presence_prob": float(pred.presence),
        "height": int(height),
        "width": int(width),
        **{key: float(scores[key]) for key in MASK_METRICS},
    }


def _rank_auc(probs: np.ndarray, labels: np.ndarray) -> float | None:
    """Mann-Whitney U AUC; ``None`` unless both classes are present."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(probs)  # average ranks for ties
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def presence_metrics(
    presence_probs: list[float], is_negative: list[bool], threshold: float = 0.5,
) -> dict:
    """Confusion matrix, accuracy/precision/recall/F1 and AUC for the presence gate.

    Positive class = a target IS present, i.e. ``not is_negative``.
    """
    if not presence_probs:
        return {
            "presence_tp": 0, "presence_fp": 0, "presence_tn": 0, "presence_fn": 0,
            "presence_accuracy": None, "presence_precision": None,
            "presence_recall": None, "presence_f1": None, "presence_auc": None,
            "presence_threshold": threshold,
        }

    probs = np.asarray(presence_probs, dtype=np.float64)
    labels = (~np.asarray(is_negative, dtype=bool)).astype(int)
    predicted = (probs >= threshold).astype(int)

    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    return {
        "presence_tp": tp, "presence_fp": fp, "presence_tn": tn, "presence_fn": fn,
        "presence_accuracy": (tp + tn) / len(probs),
        "presence_precision": precision,
        "presence_recall": recall,
        "presence_f1": f1,
        "presence_auc": _rank_auc(probs, labels),
        "presence_threshold": threshold,
    }


def aggregate(rows: list[dict]) -> dict:
    """Positives-only mask means plus an all-images presence block."""
    positives = [r for r in rows if not r["is_negative"]]
    mask_means = (
        {key: float(np.mean([r[key] for r in positives])) for key in MASK_METRICS}
        if positives else {}
    )
    return {
        "n_images": len(rows),
        "n_positives": len(positives),
        "n_negatives": len(rows) - len(positives),
        "mask": mask_means,
        "presence": presence_metrics(
            [r["presence_prob"] for r in rows],
            [bool(r["is_negative"]) for r in rows],
        ),
        "notes": (
            "Mask metrics are averaged over positives only; an empty ground truth "
            "makes S_alpha degenerate to a presence score. Presence metrics cover "
            "all images. Predictions were scored at native ground-truth resolution."
        ),
    }


def write_per_image_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_failures_csv(rows: list[dict], path: Path) -> None:
    """Positives sorted worst-first by S_alpha, tie-broken by IoU."""
    positives = [r for r in rows if not r["is_negative"]]
    ordered = sorted(positives, key=lambda r: (r["S_alpha"], r["IoU"]))
    write_per_image_csv(ordered, path)


def write_summary_json(summary: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))


def write_metrics_md(summary: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Evaluation — {summary.get('run', '(unnamed run)')}",
        "",
        f"- dataset: `{summary.get('dataset', '?')}`",
        f"- split: `{summary.get('split', '?')}`",
        f"- architecture: `{summary.get('architecture', '?')}`",
        f"- weights: `{summary.get('weights', '?')}`"
        f" (epoch {summary.get('epoch', '?')})",
        f"- img_size: `{summary.get('img_size', '?')}`",
        f"- images: {summary['n_images']} "
        f"({summary['n_positives']} positive, {summary['n_negatives']} negative)",
        "",
        "## Mask metrics (positives only, native resolution)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in MASK_METRICS:
        value = summary["mask"].get(key)
        lines.append(f"| {key} | {'n/a' if value is None else f'{value:.4f}'} |")

    lines += ["", "## Presence gate (all images)", "", "| Metric | Value |", "| --- | --- |"]
    for key, value in summary["presence"].items():
        formatted = "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append(f"| {key} | {formatted} |")

    lines += ["", f"> {summary['notes']}", ""]
    path.write_text("\n".join(lines))
```

- [ ] **Step 4: Add exports to `src/chd/eval/__init__.py`**

Insert `from chd.eval import report  # noqa: F401` is *not* enough — add the names explicitly. Replace the import block's first line group with:

```python
from chd.eval.predict import Prediction, predict_run, resize_prob
from chd.eval.report import (
    MASK_METRICS,
    aggregate,
    metric_row,
    presence_metrics,
    write_failures_csv,
    write_metrics_md,
    write_per_image_csv,
    write_summary_json,
)
from chd.eval.runs import (
    DEFAULT_RUNS_ROOT,
    RunBundle,
    available_runs,
    config_from_checkpoint,
    load_run,
    resolve_checkpoint,
)

__all__ = [
    "DEFAULT_RUNS_ROOT",
    "MASK_METRICS",
    "Prediction",
    "RunBundle",
    "aggregate",
    "available_runs",
    "config_from_checkpoint",
    "load_run",
    "metric_row",
    "predict_run",
    "presence_metrics",
    "resize_prob",
    "resolve_checkpoint",
    "write_failures_csv",
    "write_metrics_md",
    "write_per_image_csv",
    "write_summary_json",
]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_eval_report.py -v`
Expected: PASS, 13 tests

- [ ] **Step 6: Commit**

```bash
git add src/chd/eval/report.py src/chd/eval/__init__.py tests/test_eval_report.py
git commit -m "Aggregate metrics with negatives held out of the mask means

An empty ground truth sends s_measure down its y == 0 branch, where it
returns 1 - pred.mean() — a presence score, not a segmentation score.
camo_human has 1024 negatives, so averaging those into S_alpha would
inflate the headline number in the paper's comparison table.

So: mask metrics average over positives only, presence metrics cover every
image, and per_image.csv keeps the is_negative flag so any other
aggregation can be recomputed without re-running the model.

AUC uses scipy.stats.rankdata (Mann-Whitney U) rather than adding sklearn,
and handles ties correctly. Both-classes-absent returns None instead of a
misleading 0.5."
```

---

## Task 5: The evaluation CLI (`scripts/08_evaluate.py`)

**Files:**
- Create: `scripts/08_evaluate.py`
- Test: manual smoke run (this is a thin wiring layer; its parts are already unit-tested)

**Interfaces:**
- Consumes: `load_run`, `predict_run`, `metric_row`, `aggregate`, and all four writers.
- Produces: `reports/eval/<run>/{per_image.csv,summary.json,metrics.md,failures.csv,preds/}`.

- [ ] **Step 1: Implement `scripts/08_evaluate.py`**

```python
#!/usr/bin/env python3
"""Evaluate one trained run against its dataset's test split.

Everything about the model — dataset, architecture, backbone, img_size, pose
setting — is recovered from the checkpoint's own stored ``args``, so you name
a run folder and nothing else:

    python scripts/08_evaluate.py --run camo-human-final

That matters because the run folders are named inconsistently
(``camo-human-final`` holds the ``camo_human`` dataset) and because runs were
trained at ``img_size`` 640, not the 352 CLI default — evaluating at the
wrong size would silently degrade every reported number.

Protocol notes (see docs/superpowers/specs/2026-08-02-evaluation-visualization-design.md):

  - Predictions are scored against the ground truth at its **native**
    resolution, which is what the COD literature does and what makes these
    numbers comparable to the baselines in the paper's tables.
  - Mask metrics average over **positives only**; presence-gate metrics cover
    every image, including negatives.

Examples
--------
    # full test split
    python scripts/08_evaluate.py --run acd1k

    # quick smoke check, keep the probability maps for the figure scripts
    python scripts/08_evaluate.py --run acd1k --limit 8 --save-preds

    # score the val split instead
    python scripts/08_evaluate.py --run acd1k --split val
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval.predict import Prediction, predict_run  # noqa: E402
from chd.eval.report import (  # noqa: E402
    aggregate,
    metric_row,
    write_failures_csv,
    write_metrics_md,
    write_per_image_csv,
    write_summary_json,
)
from chd.eval.runs import DEFAULT_RUNS_ROOT, load_run  # noqa: E402


def _metric_job(stem: str, prob: np.ndarray, gt: np.ndarray, presence: float, is_negative: bool) -> dict:
    """Worker-process entry point: metrics for one image.

    Takes plain arrays rather than a Prediction so the pickled payload stays
    minimal, and rebuilds the dataclass on the far side.
    """
    return metric_row(Prediction(stem=stem, prob=prob, gt=gt,
                                 presence=presence, is_negative=is_negative))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder name under --runs-root, e.g. camo-human-final")
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="explicit checkpoint path, bypassing --run lookup")
    p.add_argument("--prefer", choices=("best", "last"), default="best")
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", type=Path, default=None,
                   help="default: the data_root stored in the checkpoint")
    p.add_argument("--dataset", default=None, help="override the checkpoint's dataset")
    p.add_argument("--img-size", type=int, default=None, help="override the checkpoint's img_size")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="evaluate only the first N images")
    p.add_argument("--workers", type=int, default=None,
                   help="metric worker processes; 0 runs inline. Default: cpu_count - 2")
    p.add_argument("--save-preds", action="store_true",
                   help="write uint8 probability maps to <out>/preds/ for the figure scripts")
    p.add_argument("--out", type=Path, default=None, help="default: reports/eval/<run>")
    return p


def default_workers(requested: int | None) -> int:
    import os

    if requested is not None:
        return max(0, requested)
    return max(1, (os.cpu_count() or 2) - 2)


def main() -> None:
    args = build_parser().parse_args()

    bundle = load_run(
        args.run, runs_root=args.runs_root, device=args.device, prefer=args.prefer,
        overrides={"dataset": args.dataset, "img_size": args.img_size},
        checkpoint=args.checkpoint,
    )
    out = args.out or Path("reports/eval") / args.run
    out.mkdir(parents=True, exist_ok=True)
    preds_dir = out / "preds"
    if args.save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  run          {bundle.name}  ({bundle.checkpoint_path})")
    print(f"  dataset      {bundle.config.dataset}  split={args.split}")
    print(f"  architecture {getattr(bundle.config, 'architecture', 'chdnet')}"
          f" + {getattr(bundle.config, 'backbone', '?')}")
    print(f"  weights      {bundle.weights}  (epoch {bundle.epoch},"
          f" best S_alpha={bundle.best_s_alpha})")
    print(f"  img_size     {bundle.config.img_size}  (scored at native GT resolution)")
    print(f"  out          {out}")
    print("=" * 78)

    workers = default_workers(args.workers)
    started = time.time()
    rows: list[dict] = []

    def record(prediction: Prediction, row: dict) -> None:
        rows.append(row)
        if args.save_preds:
            cv2.imwrite(str(preds_dir / f"{prediction.stem}.png"),
                        (np.clip(prediction.prob, 0, 1) * 255).astype(np.uint8))
        if len(rows) % 25 == 0:
            print(f"  [{len(rows)}] {time.time() - started:.0f}s elapsed", flush=True)

    stream = predict_run(bundle, split=args.split, data_root=args.data_root,
                         device=args.device, limit=args.limit)

    if workers == 0:
        for prediction in stream:
            record(prediction, metric_row(prediction))
    else:
        # Bounded in-flight futures: the metric stage is the slow one (two
        # 255-threshold curves per image), but queueing all 1150 native-
        # resolution maps at once would cost several GB of pickled payload.
        pending: dict = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for prediction in stream:
                future = pool.submit(_metric_job, prediction.stem, prediction.prob,
                                     prediction.gt, prediction.presence, prediction.is_negative)
                pending[future] = prediction
                while len(pending) >= workers * 3:
                    done = next(as_completed(pending))
                    record(pending.pop(done), done.result())
            for future in as_completed(list(pending)):
                record(pending[future], future.result())

    if not rows:
        raise SystemExit(f"no images evaluated for split {args.split!r} — is the split file empty?")

    rows.sort(key=lambda r: r["stem"])  # process pool completes out of order

    summary = aggregate(rows)
    summary.update({
        "run": bundle.name,
        "checkpoint": str(bundle.checkpoint_path),
        "dataset": bundle.config.dataset,
        "split": args.split,
        "architecture": getattr(bundle.config, "architecture", "chdnet"),
        "backbone": getattr(bundle.config, "backbone", None),
        "img_size": bundle.config.img_size,
        "weights": bundle.weights,
        "epoch": bundle.epoch,
        "best_s_alpha_from_training": bundle.best_s_alpha,
        "eval_seconds": round(time.time() - started, 1),
    })

    write_per_image_csv(rows, out / "per_image.csv")
    write_failures_csv(rows, out / "failures.csv")
    write_summary_json(summary, out / "summary.json")
    write_metrics_md(summary, out / "metrics.md")

    print("-" * 78)
    for key, value in summary["mask"].items():
        print(f"  {key:<18} {value:.4f}")
    presence_accuracy = summary["presence"]["presence_accuracy"]
    if presence_accuracy is not None:
        print(f"  {'presence_acc':<18} {presence_accuracy:.4f}")
    print("-" * 78)
    print(f"  wrote {out}/per_image.csv, failures.csv, summary.json, metrics.md")
    if args.save_preds:
        print(f"  wrote {preds_dir}/*.png")
    print(f"  took {summary['eval_seconds']}s over {summary['n_images']} image(s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with the untrained tiny backbone**

There is no checkpoint on this machine, so build one, then run the script
against real data. Run this exactly:

```bash
python - <<'PY'
import sys, torch, argparse
from pathlib import Path
sys.path.insert(0, "src")
from chd.models.factory import build_model
cfg = argparse.Namespace(architecture="chdnet", backbone="tiny_test", dataset="acd1k",
                         img_size=64, os_streams=2, no_pose=False, no_pretrained=True,
                         data_root=Path("data"))
m = build_model(cfg)
Path("runs/_smoke").mkdir(parents=True, exist_ok=True)
torch.save({"epoch": 0, "best_s_alpha": 0.0, "model": m.state_dict(), "ema": None,
            "args": {k: v for k, v in vars(cfg).items()}}, "runs/_smoke/best.pth")
print("wrote runs/_smoke/best.pth")
PY

python scripts/08_evaluate.py --run _smoke --limit 4 --workers 0 --save-preds
```

Expected: prints the run header, four images evaluated, a metric table, and
writes `reports/eval/_smoke/{per_image.csv,failures.csv,summary.json,metrics.md}`
plus `preds/*.png`. Metrics will be near-random — this checks wiring, not accuracy.

- [ ] **Step 3: Verify the process pool path also works**

Run: `python scripts/08_evaluate.py --run _smoke --limit 6 --workers 2`
Expected: same files, no pickling errors, `per_image.csv` has 6 rows sorted by stem.

- [ ] **Step 4: Verify the error paths are actionable**

```bash
python scripts/08_evaluate.py --run does_not_exist
```
Expected: `FileNotFoundError` naming the missing directory **and listing `_smoke`** as available.

- [ ] **Step 5: Clean up the smoke artifacts**

```bash
rm -rf runs/_smoke reports/eval/_smoke
```

- [ ] **Step 6: Commit**

```bash
git add scripts/08_evaluate.py
git commit -m "Add the evaluation CLI for a single named run

Takes one run folder name and recovers everything else from the
checkpoint, so no flags need re-typing and a mislabelled folder cannot
produce numbers for the wrong dataset.

Metrics run in a process pool because evaluate_all does two 255-threshold
curves per image — roughly 2s at 640x640, so the 1150-image combined split
would take ~40min single-threaded. In-flight futures are bounded rather
than queueing every native-resolution map at once, which would cost
several GB of pickled payload. --workers 0 runs inline for debugging.

Results are sorted by stem before writing, since the pool completes out of
order and an unstable row order would make CSV diffs useless."
```

---

## Task 6: Grad-CAM (`chd.viz.cam`)

**Files:**
- Create: `src/chd/viz/cam.py`
- Test: `tests/test_cam.py`

**Interfaces:**
- Consumes: model `forward(image, pose, return_intermediates=True)` and `predict_mask`.
- Produces:
  - `LEVEL_TAPS: tuple[str, ...]` = `("aer", "osneck", "sfa", "backbone")`
  - `PROGRESSION_TAPS: tuple[tuple[str, str], ...]` — `(label, intermediates key)` pairs
  - `cam_score(model, outputs: dict, target: str = "pred", gt: torch.Tensor | None = None, topk: int = 256) -> torch.Tensor`
  - `cams_from(acts: list[torch.Tensor], grads: tuple[torch.Tensor, ...], out_hw: tuple[int, int]) -> list[np.ndarray]`
  - `grad_cam_levels(model, image, pose, tap: str = "aer", target: str = "pred", gt=None, topk: int = 256) -> tuple[list[np.ndarray], str]` — returns `(cams, tap_used)`
  - `grad_cam_progression(model, image, pose, level: int = 0, target: str = "pred", gt=None, topk: int = 256) -> list[tuple[str, np.ndarray]]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cam.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_cam.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chd.viz.cam'`

- [ ] **Step 3: Implement `src/chd/viz/cam.py`**

```python
"""Grad-CAM for the segmentation head — target-specific saliency per level.

Why this exists alongside ``panels.channel_heat``: that function renders
``mean|activation|`` across channels, an *unsigned magnitude* map. It shows
what a layer responds to (texture, frequency), not what the network decided,
which is exactly why such panels never look like the predicted mask. Grad-CAM
answers the other question — which spatial locations drove *this* prediction.

Method: Seg-Grad-CAM (Vinogradova et al., 2020). A scalar score is formed by
summing ``mask_logit`` over a region of interest, then gradients of that score
are taken with respect to the tapped feature maps. Channel weights are the
spatially-averaged gradients; the CAM is the ReLU of their weighted sum.

Implementation note: ``CHDNet.forward(..., return_intermediates=True)`` hands
back the actual graph tensors, so ``torch.autograd.grad`` can be called on
them directly — **no forward or backward hooks are needed anywhere**. This
keeps the model file untouched.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from chd.viz.panels import normalize01

#: Per-level taps, finest-to-coarsest lists of 4 tensors each.
LEVEL_TAPS = ("aer", "osneck", "sfa", "backbone")

#: Module-boundary progression at a single level: (display label, intermediates key).
#: FDM's high-frequency branch is the camouflage-relevant half of its split,
#: so that is the one shown in the progression.
PROGRESSION_TAPS = (
    ("Backbone", "backbone"),
    ("+FDM (HF)", "fdm_hf"),
    ("+SFA", "sfa"),
    ("+OSNeck", "osneck"),
    ("+AER", "aer"),
    ("Decoder", "decoder_levels"),
)

FALLBACK_TAP = "backbone"


def cam_score(
    model, outputs: dict, target: str = "pred", gt: torch.Tensor | None = None, topk: int = 256,
) -> torch.Tensor:
    """Scalar to differentiate: ``mask_logit`` summed over a region of interest.

    ``target="pred"`` explains the model's own decision, which is what a
    "why was this mask highlighted" figure needs. When the prediction is empty
    (every negative image, and any missed target) summing over it would give a
    constant zero with no gradient, so the score falls back to the ``topk``
    highest logits — still the model's own evidence, just unthresholded.
    """
    logit = outputs["mask_logit"]

    if target == "all":
        return logit.sum()

    if target == "gt":
        if gt is None:
            raise ValueError("target='gt' requires a gt tensor")
        region = (gt > 0.5).to(logit.dtype)
    else:
        region = (model.predict_mask(outputs) > 0.5).to(logit.dtype)

    if float(region.sum()) == 0.0:
        flat = logit.flatten()
        k = min(topk, flat.numel())
        return torch.topk(flat, k).values.sum()
    return (logit * region).sum()


def cams_from(
    acts: list[torch.Tensor], grads: tuple[torch.Tensor, ...], out_hw: tuple[int, int],
) -> list[np.ndarray]:
    """Grad-CAM per (activation, gradient) pair, upsampled to ``out_hw``."""
    cams: list[np.ndarray] = []
    for activation, gradient in zip(acts, grads):
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=out_hw, mode="bilinear", align_corners=False)
        cams.append(normalize01(cam[0, 0].detach().float().cpu().numpy()))
    return cams


def _grad_for(score: torch.Tensor, tensors: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """``autograd.grad`` over possibly-duplicated tensors.

    ``PretrainedUNet`` returns ``[decoded] * 4`` for its decoder levels, i.e.
    the same tensor four times. ``autograd.grad`` rejects duplicated inputs, so
    unique tensors are differentiated once and the results expanded back.
    """
    unique: list[torch.Tensor] = []
    positions: list[int] = []
    for tensor in tensors:
        for index, seen in enumerate(unique):
            if seen is tensor:
                positions.append(index)
                break
        else:
            positions.append(len(unique))
            unique.append(tensor)
    grads = torch.autograd.grad(score, unique, retain_graph=True, allow_unused=False)
    return tuple(grads[index] for index in positions)


def _forward_with_intermediates(model, image: torch.Tensor, pose: torch.Tensor) -> dict:
    """Forward pass with gradients enabled — Grad-CAM cannot run under no_grad."""
    with torch.enable_grad():
        return model(image, pose, return_intermediates=True)


def _resolve_tap(intermediates: dict, tap: str) -> str:
    """Fall back to the backbone when an architecture lacks the requested tap."""
    if intermediates.get(tap):
        return tap
    return FALLBACK_TAP


def grad_cam_levels(
    model, image: torch.Tensor, pose: torch.Tensor, tap: str = "aer",
    target: str = "pred", gt: torch.Tensor | None = None, topk: int = 256,
) -> tuple[list[np.ndarray], str]:
    """One Grad-CAM per pyramid level from a single forward/backward pass.

    Returns ``(cams_finest_to_coarsest, tap_actually_used)``. The tap is
    returned because it may have been downgraded for an architecture that
    lacks the requested module — the caller needs to label the figure honestly.
    """
    model.zero_grad(set_to_none=True)
    outputs = _forward_with_intermediates(model, image, pose)
    intermediates = outputs["intermediates"]

    used = _resolve_tap(intermediates, tap)
    acts = list(intermediates[used])
    score = cam_score(model, outputs, target=target, gt=gt, topk=topk)
    grads = _grad_for(score, acts)
    return cams_from(acts, grads, tuple(image.shape[-2:])), used


def grad_cam_progression(
    model, image: torch.Tensor, pose: torch.Tensor, level: int = 0,
    target: str = "pred", gt: torch.Tensor | None = None, topk: int = 256,
) -> list[tuple[str, np.ndarray]]:
    """Grad-CAM at each module boundary for one pyramid level.

    Shows where evidence moves as the forward pass proceeds. This is a
    *within-network* progression, not a cumulative ablation across separately
    trained variants — see the design doc.

    Modules the architecture does not have are skipped rather than faked.
    """
    model.zero_grad(set_to_none=True)
    outputs = _forward_with_intermediates(model, image, pose)
    intermediates = outputs["intermediates"]

    labels: list[str] = []
    acts: list[torch.Tensor] = []
    for label, key in PROGRESSION_TAPS:
        tensors = intermediates.get(key)
        if not tensors or level >= len(tensors):
            continue
        labels.append(label)
        acts.append(tensors[level])

    score = cam_score(model, outputs, target=target, gt=gt, topk=topk)
    grads = _grad_for(score, acts)
    heats = cams_from(acts, grads, tuple(image.shape[-2:]))
    return list(zip(labels, heats))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_cam.py -v`
Expected: PASS. The `pretrained_unet` test skips if `segmentation_models_pytorch` is absent.

- [ ] **Step 5: Confirm the whole suite still passes**

Run: `python -m pytest tests/ -m "not slow" -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/chd/viz/cam.py tests/test_cam.py
git commit -m "Add Grad-CAM so a heatmap can explain the predicted mask

The existing pipeline figure renders mean|activation| across channels. That
is an unsigned magnitude map — it shows what a layer responds to, not what
the network decided — which is exactly why none of those panels resemble
the predicted mask. Grad-CAM answers the other question: which locations
drove this particular prediction.

Seg-Grad-CAM (Vinogradova et al. 2020): sum mask_logit over a region of
interest, take gradients w.r.t. the tapped features, weight channels by
their mean gradient, ReLU the sum. Because forward(return_intermediates)
hands back live graph tensors, autograd.grad works on them directly and no
hooks are needed — the model files stay untouched.

Two degradations handled rather than crashing: an empty prediction (every
negative image) falls back to the top-k logits so the score still has a
gradient, and an architecture missing the requested tap falls back to its
backbone and reports which tap was actually used, so a figure can be
labelled honestly."
```

---

## Task 7: The figure CLI (`scripts/09_visualize_predictions.py`)

**Files:**
- Create: `scripts/09_visualize_predictions.py`
- Test: manual smoke run against a synthetic checkpoint

**Interfaces:**
- Consumes: `load_run`, `predict_run`, `panels.*`, `cam.*`, `chd.data.dataset.CHDDataset`.
- Produces: `reports/figures/<run>/{qualitative,gradcam_levels,activations_L1..L4,progression}.{png,svg}`.

- [ ] **Step 1: Implement `scripts/09_visualize_predictions.py`**

```python
#!/usr/bin/env python3
"""Render the evaluation figures for one trained run.

Four figure families, each in its own file so every one stays readable at
paper column width:

  qualitative        Input | Ground Truth | Prediction | Error
  gradcam_levels     Input | GT | CAM-L1 | CAM-L2 | CAM-L3 | CAM-L4
  activations_L<n>   rows = images, cols = module boundaries, one file per level
  progression        Input | Backbone | +FDM | +SFA | +OSNeck | +AER | Decoder

Ground truth and prediction use the COD-paper convention — a bright mask
silhouette over a darkened image — so the figure is directly comparable to
published qualitative panels. The error panel separates false positives from
false negatives by hue, which shows *how* a prediction is wrong.

Grad-CAM figures use ``jet``; raw activation figures use ``inferno``. The two
colormaps are deliberately different so the families are never confused: a
Grad-CAM is target-specific, an activation map is not.

Examples
--------
    python scripts/09_visualize_predictions.py --run camo-human-final --num-images 6

    # the worst predictions, which needs 08_evaluate.py to have run first
    python scripts/09_visualize_predictions.py --run acd1k --pick worst --num-images 5

    # cross-variant ablation columns instead of a within-network progression
    python scripts/09_visualize_predictions.py --run acd1k --also-run acd1k2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.dataset import AugmentConfig, CHDDataset  # noqa: E402
from chd.eval.predict import predict_run  # noqa: E402
from chd.eval.runs import DEFAULT_RUNS_ROOT, load_run  # noqa: E402
from chd.viz import cam as camlib  # noqa: E402
from chd.viz import panels  # noqa: E402

ACTIVATION_COLUMNS = (
    ("Backbone", "backbone"),
    ("FDM low-freq", "fdm_lf"),
    ("FDM high-freq", "fdm_hf"),
    ("SFA", "sfa"),
    ("OSNeck", "osneck"),
    ("AER", "aer"),
    ("Decoder", "decoder_levels"),
)
N_LEVELS = 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--also-run", action="append", default=[],
                   help="repeatable; adds one ablation column per extra run to the progression figure")
    p.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    p.add_argument("--prefer", choices=("best", "last"), default="best")
    p.add_argument("--split", default="test")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--num-images", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pick", choices=("random", "best", "worst"), default="random",
                   help="best/worst read reports/eval/<run>/failures.csv")
    p.add_argument("--eval-dir", type=Path, default=None, help="default: reports/eval/<run>")
    p.add_argument("--cmap", default="jet", help="colormap for Grad-CAM figures")
    p.add_argument("--activation-cmap", default="inferno")
    p.add_argument("--cam-tap", default="aer", choices=camlib.LEVEL_TAPS)
    p.add_argument("--cam-target", default="pred", choices=("pred", "gt", "all"))
    p.add_argument("--progression-level", type=int, default=0, choices=range(N_LEVELS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=None, help="default: reports/figures/<run>")
    return p


def select_stems(args: argparse.Namespace, dataset: CHDDataset) -> list[str]:
    """Choose which images to render, honouring --pick.

    ``best``/``worst`` need ``08_evaluate.py`` to have run. If the ranking file
    is missing we fall back to random and say so, rather than failing or
    silently implying the choice was score-based.
    """
    if args.pick == "random":
        rng = np.random.default_rng(args.seed)
        n = min(args.num_images, len(dataset))
        return [dataset.stems[i] for i in rng.choice(len(dataset), size=n, replace=False)]

    failures = (args.eval_dir or Path("reports/eval") / args.run) / "failures.csv"
    if not failures.exists():
        print(f"[viz] --pick {args.pick} needs {failures}, which does not exist. "
              f"Run scripts/08_evaluate.py --run {args.run} first. Falling back to random.")
        args.pick = "random"
        return select_stems(args, dataset)

    with failures.open() as fh:
        ranked = [row["stem"] for row in csv.DictReader(fh)]  # worst-first
    if args.pick == "best":
        ranked = list(reversed(ranked))
    return ranked[: args.num_images]


def to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    return (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def grid(n_rows: int, n_cols: int, scale: float = 2.0):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(scale * n_cols, scale * n_rows), squeeze=False)
    return fig, axes


def fig_qualitative(bundle, records: list[dict], out: Path, run_label: str) -> None:
    columns = ("Input", "Ground Truth", "Prediction", "Error")
    fig, axes = grid(len(records), len(columns))
    for row, record in enumerate(records):
        image = record["image"]
        gt = record["gt"] > 0.5
        pred = record["prob"] > 0.5
        drawn = (
            image,
            panels.mask_composite(image, gt),
            panels.mask_composite(image, pred),
            panels.error_map(image, pred, gt),
        )
        for col, (name, content) in enumerate(zip(columns, drawn)):
            ax = axes[row][col]
            ax.imshow(content)
            bare(ax)
            if row == 0:
                ax.set_title(name, fontsize=9)
        axes[row][0].set_ylabel(record["stem"], fontsize=7)
        axes[row][2].set_xlabel(f"presence={record['presence']:.2f}", fontsize=8)
    fig.suptitle(f"Qualitative results — {run_label}", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    panels.save_figure(fig, out, "qualitative")


def fig_gradcam_levels(bundle, records: list[dict], args, out: Path, run_label: str) -> None:
    taps_used: set[str] = set()
    n_cols = 2 + N_LEVELS
    fig, axes = grid(len(records), n_cols)
    for row, record in enumerate(records):
        image = record["image"]
        cams, tap = camlib.grad_cam_levels(
            bundle.model, record["image_tensor"].to(args.device),
            record["pose_tensor"].to(args.device),
            tap=args.cam_tap, target=args.cam_target,
            gt=record["gt_tensor"].to(args.device) if args.cam_target == "gt" else None,
        )
        taps_used.add(tap)
        drawn = [image, panels.mask_composite(image, record["gt"] > 0.5)]
        drawn += [panels.overlay_heat(image, c, cmap=args.cmap) for c in cams]
        names = ["Input", "Ground Truth"] + [f"CAM L{i + 1}" for i in range(len(cams))]
        for col, (name, content) in enumerate(zip(names, drawn)):
            ax = axes[row][col]
            ax.imshow(content)
            bare(ax)
            if row == 0:
                ax.set_title(name, fontsize=9)
        axes[row][0].set_ylabel(record["stem"], fontsize=7)
    tap_note = "/".join(sorted(taps_used))
    fig.suptitle(
        f"Grad-CAM per pyramid level (tap={tap_note}, target={args.cam_target}) — {run_label}",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    panels.save_figure(fig, out, "gradcam_levels")


def fig_activations(bundle, records: list[dict], args, out: Path, run_label: str) -> None:
    """One file per pyramid level; columns the architecture lacks are marked, not faked."""
    for level in range(N_LEVELS):
        fig, axes = grid(len(records), 1 + len(ACTIVATION_COLUMNS))
        for row, record in enumerate(records):
            axes[row][0].imshow(record["image"])
            bare(axes[row][0])
            if row == 0:
                axes[row][0].set_title("Input", fontsize=9)
            axes[row][0].set_ylabel(record["stem"], fontsize=7)

            intermediates = record["intermediates"]
            for col, (name, key) in enumerate(ACTIVATION_COLUMNS, start=1):
                ax = axes[row][col]
                bare(ax)
                tensors = intermediates.get(key)
                if not tensors or level >= len(tensors):
                    panels.blank_panel(ax, f"{name}\nnot in this\narchitecture")
                else:
                    ax.imshow(panels.channel_heat(tensors[level][0]),
                              cmap=args.activation_cmap, vmin=0, vmax=1)
                if row == 0:
                    ax.set_title(name, fontsize=9)
        fig.suptitle(
            f"Activation magnitude at pyramid level {level + 1} — {run_label}\n"
            "mean|activation| across channels (not target-specific — see the Grad-CAM figure)",
            fontsize=10, y=0.997,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        panels.save_figure(fig, out, f"activations_L{level + 1}")


def fig_progression(bundles: list, records: list[dict], args, out: Path) -> None:
    """Within-network module progression, or one column per run when --also-run is given."""
    ablation = len(bundles) > 1
    if ablation:
        column_labels = [b.name for b in bundles]
    else:
        column_labels = [
            label for label, key in camlib.PROGRESSION_TAPS
            if records[0]["intermediates"].get(key)
        ]

    fig, axes = grid(len(records), 1 + len(column_labels))
    for row, record in enumerate(records):
        axes[row][0].imshow(record["image"])
        bare(axes[row][0])
        if row == 0:
            axes[row][0].set_title("Input", fontsize=9)
        axes[row][0].set_ylabel(record["stem"], fontsize=7)

        if ablation:
            heats = []
            for bundle in bundles:
                cams, _ = camlib.grad_cam_levels(
                    bundle.model, record["image_tensor"].to(args.device),
                    record["pose_tensor"].to(args.device),
                    tap=args.cam_tap, target=args.cam_target,
                )
                heats.append(cams[args.progression_level])
        else:
            heats = [heat for _, heat in camlib.grad_cam_progression(
                bundles[0].model, record["image_tensor"].to(args.device),
                record["pose_tensor"].to(args.device), level=args.progression_level,
                target=args.cam_target,
            )]

        for col, (name, heat) in enumerate(zip(column_labels, heats), start=1):
            ax = axes[row][col]
            ax.imshow(panels.overlay_heat(record["image"], heat, cmap=args.cmap))
            bare(ax)
            if row == 0:
                ax.set_title(name, fontsize=9)

    mode = (
        f"cross-run ablation ({len(bundles)} runs), Grad-CAM at level {args.progression_level + 1}"
        if ablation else
        f"within-network module progression, level {args.progression_level + 1}"
    )
    fig.suptitle(f"Grad-CAM — {mode}", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    panels.save_figure(fig, out, "progression")


def gather_records(bundle, args, stems: list[str], dataset: CHDDataset) -> list[dict]:
    """Everything each figure row needs, computed once per image."""
    predictions = {p.stem: p for p in predict_run(
        bundle, split=args.split, data_root=args.data_root, device=args.device, stems=stems)}
    by_stem = {stem: index for index, stem in enumerate(dataset.stems)}

    records: list[dict] = []
    for stem in stems:
        if stem not in predictions:
            continue
        item = dataset[by_stem[stem]]
        image_tensor = item["image"].unsqueeze(0)
        pose_tensor = item["pose"].unsqueeze(0)
        with torch.no_grad():
            outputs = bundle.model(image_tensor.to(args.device), pose_tensor.to(args.device),
                                   return_intermediates=True)
        prediction = predictions[stem]
        image = to_uint8_image(item["image"])
        height, width = image.shape[:2]
        records.append({
            "stem": stem,
            "image": image,
            "image_tensor": image_tensor,
            "pose_tensor": pose_tensor,
            "gt_tensor": item["mask"].unsqueeze(0),
            # Figures are drawn at img_size, so the native-resolution
            # prediction is resized down to match the rendered input. Metrics
            # are never computed from this copy — 08_evaluate.py scores the
            # native-resolution map — so resizing here cannot move a number.
            "prob": cv2.resize(prediction.prob, (width, height), interpolation=cv2.INTER_LINEAR),
            "gt": item["mask"][0].numpy(),
            "presence": prediction.presence,
            "intermediates": outputs["intermediates"],
        })
    return records


def main() -> None:
    args = build_parser().parse_args()

    bundle = load_run(args.run, runs_root=args.runs_root, device=args.device, prefer=args.prefer)
    extra = [load_run(name, runs_root=args.runs_root, device=args.device, prefer=args.prefer)
             for name in args.also_run]

    out = args.out or Path("reports/figures") / args.run
    root = Path(args.data_root or getattr(bundle.config, "data_root", "data")) / bundle.config.dataset
    dataset = CHDDataset(root, args.split, img_size=bundle.config.img_size,
                         augment=AugmentConfig(enabled=False))

    stems = select_stems(args, dataset)
    print(f"[viz] run={bundle.name} dataset={bundle.config.dataset} weights={bundle.weights}")
    print(f"[viz] {len(stems)} image(s), pick={args.pick}: {stems}")

    records = gather_records(bundle, args, stems, dataset)
    if not records:
        raise SystemExit("no images gathered — check --split and --num-images")

    run_label = (f"{bundle.name} ({bundle.config.dataset}, "
                 f"{getattr(bundle.config, 'architecture', 'chdnet')}, {bundle.weights} weights)")

    fig_qualitative(bundle, records, out, run_label)
    fig_gradcam_levels(bundle, records, args, out, run_label)
    fig_activations(bundle, records, args, out, run_label)
    fig_progression([bundle, *extra], records, args, out)

    print(f"[viz] wrote {out}/qualitative, gradcam_levels, "
          f"activations_L1..L{N_LEVELS}, progression (.png + .svg)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test all four figure families**

```bash
python - <<'PY'
import sys, torch, argparse
from pathlib import Path
sys.path.insert(0, "src")
from chd.models.factory import build_model
cfg = argparse.Namespace(architecture="chdnet", backbone="tiny_test", dataset="acd1k",
                         img_size=64, os_streams=2, no_pose=False, no_pretrained=True,
                         data_root=Path("data"))
m = build_model(cfg)
Path("runs/_smoke").mkdir(parents=True, exist_ok=True)
torch.save({"epoch": 0, "best_s_alpha": 0.0, "model": m.state_dict(), "ema": None,
            "args": {k: v for k, v in vars(cfg).items()}}, "runs/_smoke/best.pth")
PY

python scripts/09_visualize_predictions.py --run _smoke --num-images 3
```

Expected: writes `reports/figures/_smoke/` containing `qualitative`,
`gradcam_levels`, `activations_L1`–`activations_L4` and `progression`, each as
both `.png` and `.svg`. Open `gradcam_levels.png` and confirm the CAM columns
are non-uniform (colored structure, not flat blue).

- [ ] **Step 3: Verify the `--pick worst` fallback warns instead of crashing**

Run: `python scripts/09_visualize_predictions.py --run _smoke --pick worst --num-images 2`
Expected: prints the "needs reports/eval/_smoke/failures.csv … Falling back to random" message and still writes every figure.

- [ ] **Step 4: Verify the ablation mode**

```bash
cp -r runs/_smoke runs/_smoke2
python scripts/09_visualize_predictions.py --run _smoke --also-run _smoke2 --num-images 2
```
Expected: `progression.png` titled "cross-run ablation (2 runs)" with one column per run.

- [ ] **Step 5: Clean up**

```bash
rm -rf runs/_smoke runs/_smoke2 reports/figures/_smoke
```

- [ ] **Step 6: Commit**

```bash
git add scripts/09_visualize_predictions.py
git commit -m "Render qualitative, Grad-CAM, activation and progression figures

Four families in four files rather than one mega-figure, so each stays
readable at paper column width.

Ground truth and prediction follow the COD-paper convention (bright mask
silhouette over a darkened image) so the panel is directly comparable to
the published baselines, and the error panel splits false positives from
false negatives by hue so it shows how a prediction is wrong.

Grad-CAM uses jet, raw activations use inferno — deliberately different,
because a Grad-CAM is target-specific and an activation map is not, and
the two must never be read as the same kind of evidence. The activation
figure marks columns an architecture lacks instead of drawing zeros, which
would look like a real but empty measurement.

The progression figure is honest about its two modes: one run gives a
within-network module progression, --also-run gives true cross-variant
ablation columns, and the title states which."
```

---

## Task 8: Cross-run comparison (`scripts/10_compare_runs.py`)

**Files:**
- Create: `scripts/10_compare_runs.py`

**Interfaces:**
- Consumes: `reports/eval/*/summary.json` written by Task 5; `chd.viz.colors`, `chd.viz.panels.save_figure`.
- Produces: `reports/comparison/{comparison.csv,<metric>.png/.svg}`.

- [ ] **Step 1: Implement `scripts/10_compare_runs.py`**

```python
#!/usr/bin/env python3
"""Compare every run that has already been evaluated.

Reads ``reports/eval/*/summary.json`` and emits one grouped bar chart per
metric plus a tidy CSV. This script runs no models — it only aggregates
completed evaluations, which is what keeps it consistent with the
one-run-per-invocation rule for ``08_evaluate.py``.

It states which runs it found, so a missing dataset (mhcd, until it is
trained) is visible rather than silently absent from the chart.

Example
-------
    python scripts/08_evaluate.py --run acd1k
    python scripts/08_evaluate.py --run cpd1k
    python scripts/10_compare_runs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.eval.report import MASK_METRICS  # noqa: E402
from chd.viz.colors import DATASET_COLOR, DATASET_LABEL, INK  # noqa: E402
from chd.viz.panels import save_figure  # noqa: E402

#: Lower is better for these, so charts label the direction explicitly.
LOWER_IS_BETTER = {"MAE"}
DEFAULT_METRICS = ("S_alpha", "MAE", "F_beta_mean", "E_phi_mean", "F_bd", "IoU")


def load_summaries(eval_root: Path) -> list[dict]:
    summaries = []
    for path in sorted(eval_root.glob("*/summary.json")):
        data = json.loads(path.read_text())
        if not data.get("mask"):
            print(f"[compare] skipping {path}: no positives, so no mask metrics")
            continue
        data["_run"] = data.get("run", path.parent.name)
        summaries.append(data)
    return summaries


def write_comparison_csv(summaries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run", "dataset", "architecture", "backbone", "img_size", "weights",
              "n_positives", *MASK_METRICS, "presence_accuracy", "presence_auc"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            row = {
                "run": summary["_run"],
                "dataset": summary.get("dataset"),
                "architecture": summary.get("architecture"),
                "backbone": summary.get("backbone"),
                "img_size": summary.get("img_size"),
                "weights": summary.get("weights"),
                "n_positives": summary.get("n_positives"),
                "presence_accuracy": summary["presence"].get("presence_accuracy"),
                "presence_auc": summary["presence"].get("presence_auc"),
            }
            row.update({m: summary["mask"].get(m) for m in MASK_METRICS})
            writer.writerow(row)


def bar_chart(summaries: list[dict], metric: str, out: Path) -> None:
    labels, values, colors = [], [], []
    for summary in summaries:
        dataset = summary.get("dataset", "?")
        labels.append(f"{summary['_run']}\n{DATASET_LABEL.get(dataset, dataset)}")
        values.append(summary["mask"].get(metric, float("nan")))
        colors.append(DATASET_COLOR.get(dataset, INK["secondary"]))

    fig, ax = plt.subplots(figsize=(max(5.0, 1.5 * len(labels)), 4.0))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=colors, width=0.65)
    for x, value in zip(positions, values):
        ax.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color=INK["primary"])

    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    ax.set_title(f"{metric} by run  ({direction})", color=INK["primary"])
    ax.set_ylabel(metric)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=INK["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_figure(fig, out, metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, default=Path("reports/eval"))
    parser.add_argument("--out", type=Path, default=Path("reports/comparison"))
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    args = parser.parse_args()

    summaries = load_summaries(args.eval_root)
    if not summaries:
        raise SystemExit(
            f"no evaluated runs under {args.eval_root}. "
            "Run scripts/08_evaluate.py --run <name> first."
        )

    print(f"[compare] {len(summaries)} evaluated run(s):")
    for summary in summaries:
        print(f"    {summary['_run']:<24} dataset={summary.get('dataset'):<12} "
              f"S_alpha={summary['mask'].get('S_alpha', float('nan')):.4f}")

    write_comparison_csv(summaries, args.out / "comparison.csv")
    for metric in args.metrics:
        if metric not in MASK_METRICS:
            print(f"[compare] skipping unknown metric {metric!r}")
            continue
        bar_chart(summaries, metric, args.out)

    print(f"[compare] wrote {args.out}/comparison.csv and one chart per metric")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with two synthetic summaries**

```bash
python - <<'PY'
import json
from pathlib import Path
for name, ds, s in [("acd1k", "acd1k", 0.87), ("cpd1k", "cpd1k", 0.81)]:
    d = Path("reports/eval") / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({
        "run": name, "dataset": ds, "architecture": "chdnet", "backbone": "res2net50_26w_4s",
        "img_size": 640, "weights": "ema", "n_images": 10, "n_positives": 10, "n_negatives": 0,
        "mask": {"MAE": 0.04, "F_beta_mean": 0.8, "F_beta_max": 0.82, "F_beta_adaptive": 0.79,
                 "S_alpha": s, "E_phi_mean": 0.9, "E_phi_max": 0.92, "E_phi_adaptive": 0.89,
                 "F_bd": 0.7, "IoU": 0.72, "Dice": 0.8},
        "presence": {"presence_accuracy": 1.0, "presence_auc": None}, "notes": "synthetic",
    }, indent=2))
print("wrote synthetic summaries")
PY

python scripts/10_compare_runs.py
```

Expected: lists both runs, writes `reports/comparison/comparison.csv` and six
metric charts. Confirm `S_alpha.png` shows acd1k above cpd1k, and that the
`MAE` chart title says "lower is better".

- [ ] **Step 3: Verify the empty case is actionable**

Run: `python scripts/10_compare_runs.py --eval-root /tmp/does-not-exist`
Expected: `SystemExit` telling you to run `08_evaluate.py` first.

- [ ] **Step 4: Clean up**

```bash
rm -rf reports/eval/acd1k reports/eval/cpd1k reports/comparison
```

- [ ] **Step 5: Commit**

```bash
git add scripts/10_compare_runs.py
git commit -m "Compare runs from already-written evaluation summaries

Reads reports/eval/*/summary.json rather than re-running any model, which
is what lets it span runs without breaking the one-run-per-invocation rule
that 08_evaluate.py follows.

Charts label metric direction explicitly (MAE is lower-is-better, the rest
higher) because a bar chart of mixed-direction metrics is otherwise easy to
misread, and it prints the runs it found so an untrained dataset is
visibly absent rather than silently missing.

Colors come from the existing validated Okabe-Ito palette in viz/colors.py
so these charts read as one system with the dataset reports."
```

---

## Task 9: Dataset figures (`scripts/11_dataset_figures.py`)

**Files:**
- Create: `scripts/11_dataset_figures.py`

**Interfaces:**
- Consumes: `chd.data.manifest.read_split`, `load_rgb`, `load_gray`; `chd.viz.panels.component_bboxes`, `save_figure`.
- Produces: `reports/datasets/collage_<dataset>.{png,svg}`, `reports/datasets/annotated_<dataset>.{png,svg}`.

- [ ] **Step 1: Implement `scripts/11_dataset_figures.py`**

```python
#!/usr/bin/env python3
"""Per-dataset sample figures: a raw-image collage and annotated triplets.

Model-independent, so this needs no checkpoint and can run before any
training finishes. Complements ``scripts/06_visualize_datasets.py``, which
already emits ``strips/<name>.png``
(Image | Mask | Overlay | Edge | Pose | Resized) — these two figures add the
styles that strip does not cover:

  collage_<dataset>     grid of raw images, no annotation at all
  annotated_<dataset>   Image | Image + bounding box(es) | Ground-truth mask

Boxes are derived from the mask with one box per 8-connected component, using
the same connectivity as ``chd.data.manifest.count_components`` so a mask's
component count and its drawn box count cannot disagree. Components below
``--min-area-frac`` of the frame are dropped, or mask speckle would draw
spurious boxes and make clean data look noisy.

Examples
--------
    python scripts/11_dataset_figures.py --dataset acd1k
    python scripts/11_dataset_figures.py --dataset all --rows 2 --cols 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.manifest import load_gray, load_rgb, read_split  # noqa: E402
from chd.viz.colors import DATASET_LABEL, DATASET_ORDER, INK  # noqa: E402
from chd.viz.panels import component_bboxes, save_figure  # noqa: E402

BOX_COLOR = "#D55E00"


def pick_stems(root: Path, split: str, count: int, seed: int) -> list[str]:
    stems = read_split(root, split)
    if not stems:
        raise SystemExit(f"no stems for split {split!r} under {root}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(stems), size=min(count, len(stems)), replace=False)
    return [stems[int(i)] for i in chosen]


def fig_collage(root: Path, name: str, stems: list[str], rows: int, cols: int, out: Path) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.4 * rows), squeeze=False)
    for index, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if index < len(stems):
            ax.imshow(load_rgb(root / "images" / f"{stems[index]}.jpg"))
        else:
            ax.set_visible(False)
    fig.suptitle(f"{DATASET_LABEL.get(name, name)} — sample images",
                 fontsize=12, color=INK["primary"], y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, f"collage_{name}")


def fig_annotated(root: Path, name: str, stems: list[str], min_area_frac: float, out: Path) -> None:
    columns = ("Image", "Bounding box", "Ground truth")
    fig, axes = plt.subplots(len(stems), 3, figsize=(3 * 2.4, 2.4 * len(stems)), squeeze=False)
    for row, stem in enumerate(stems):
        image = load_rgb(root / "images" / f"{stem}.jpg")
        mask = load_gray(root / "masks" / f"{stem}.png") > 127

        for col in range(3):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 2:
                ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(image)
            if row == 0:
                ax.set_title(columns[col], fontsize=10)

        for x1, y1, x2, y2 in component_bboxes(mask, min_area_frac=min_area_frac):
            axes[row][1].add_patch(mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=BOX_COLOR, linewidth=1.8))
        axes[row][0].set_ylabel(stem, fontsize=7)

    fig.suptitle(f"{DATASET_LABEL.get(name, name)} — image, target box, ground-truth mask",
                 fontsize=12, color=INK["primary"], y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, out, f"annotated_{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="all",
                        help=f"one of {list(DATASET_ORDER)}, or 'all'")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--annotated-rows", type=int, default=4)
    parser.add_argument("--min-area-frac", type=float, default=0.001,
                        help="drop mask components smaller than this fraction of the frame")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/datasets"))
    args = parser.parse_args()

    names = list(DATASET_ORDER) if args.dataset == "all" else [args.dataset]
    for name in names:
        root = args.data_root / name
        if not root.is_dir():
            print(f"[dataset-fig] skipping {name}: {root} not prepared")
            continue
        n_collage = args.rows * args.cols
        stems = pick_stems(root, args.split, max(n_collage, args.annotated_rows), args.seed)
        fig_collage(root, name, stems[:n_collage], args.rows, args.cols, args.out)
        fig_annotated(root, name, stems[: args.annotated_rows], args.min_area_frac, args.out)
        print(f"[dataset-fig] {name}: wrote collage_{name} and annotated_{name} (.png + .svg)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against real data**

Run: `python scripts/11_dataset_figures.py --dataset acd1k`
Expected: writes `reports/datasets/collage_acd1k.{png,svg}` and
`annotated_acd1k.{png,svg}`. Open `annotated_acd1k.png` and confirm each box
actually encloses the white mask region in the third column.

- [ ] **Step 3: Run it for every prepared dataset**

Run: `python scripts/11_dataset_figures.py --dataset all`
Expected: figures for acd1k, cpd1k, camo_human, mhcd and combined; no traceback.

- [ ] **Step 4: Commit**

```bash
git add scripts/11_dataset_figures.py
git commit -m "Add per-dataset sample collage and annotated-triplet figures

Model-independent, so these run before any training finishes and need no
checkpoint. They complement 06_visualize_datasets.py's preprocessing strip
rather than duplicating it: a plain raw-image collage, and an
image/box/mask triplet.

Boxes come from component_bboxes, which uses the same 8-connectivity as
manifest.count_components — so a mask's reported component count and its
drawn box count cannot disagree — and drops components below a minimum
area fraction, because a single stray pixel would otherwise draw a full
box and make clean data look noisy."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| `eval/runs.py` — run resolution, EMA preference, error paths | 2 |
| `eval/predict.py` — native-resolution protocol | 3 |
| `eval/report.py` — aggregation, presence block, 4 writers | 4 |
| `viz/panels.py` — rendering helpers | 1 |
| `viz/cam.py` — Grad-CAM, taps, target modes | 6 |
| `08_evaluate.py` + all 5 outputs incl. `preds/` | 5 |
| FIG 1 qualitative (composite + error map) | 7 |
| FIG 2 Grad-CAM 4 levels, jet | 7 |
| FIG 3 activations per level, inferno, graceful degradation | 7 |
| FIG 4 progression + `--also-run` ablation mode | 7 |
| `10_compare_runs.py` | 8 |
| `11_dataset_figures.py` — collage + annotated triplets | 9 |
| Process-pool performance mitigation | 5 |
| `--pick worst` fallback warning | 7 |
| Binarize-for-render / continuous-for-metrics separation | 1 (render), 4 (metrics) |
| Testing section | 1, 2, 3, 4, 6 |

No gaps found.

**Placeholder scan:** One real issue found and fixed — an early draft of Task 7
used `__import__("cv2")` inline plus a dead conditional branch, then repaired it
in a follow-up step. Writing deliberately-broken code and fixing it is not a
legitimate plan step, so Task 7 Step 1 now carries the correct `import cv2` and
the correct `cv2.resize` call, and the repair step is gone. No other
placeholders, "TBD"s, or "add error handling"-style instructions remain.

**Type consistency check:**
- `RunBundle` fields are referenced as `.name`, `.model`, `.config`, `.weights`, `.epoch`, `.best_s_alpha`, `.checkpoint_path` in Tasks 3, 5, 7 — all defined in Task 2.
- `Prediction` fields `.stem`, `.prob`, `.gt`, `.presence`, `.is_negative` are used in Tasks 4, 5, 7 — all defined in Task 3.
- `predict_run(bundle, split, data_root, device, limit, stems)` — the `stems` parameter used in Task 7 is defined in Task 3.
- `grad_cam_levels` returns `(cams, tap_used)` in Task 6 and is unpacked as a 2-tuple in Task 7. `grad_cam_progression` returns `list[(label, heat)]` and is consumed as such.
- `MASK_METRICS` is defined in Task 4 and imported in Tasks 5 and 8.
- `save_figure(fig, out_dir, name)` is defined in Task 1 and called with that argument order in Tasks 7, 8, 9.
- `panels.blank_panel(ax, text)` defined in Task 1, used in Task 7.
- `ERROR_COLOR` added in Task 1 Step 1, consumed by `panels.error_map` in Task 1 Step 4.

No inconsistencies found.

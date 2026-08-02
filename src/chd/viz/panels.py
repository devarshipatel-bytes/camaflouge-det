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

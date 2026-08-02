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
from chd.viz.colors import DATASET_LABEL, DATASET_ORDER, ERROR_COLOR, INK  # noqa: E402
from chd.viz.panels import component_bboxes, save_figure  # noqa: E402

#: Target boxes reuse the palette's false-positive orange — one shared source
#: of truth, so the box colour cannot drift away from the figure palette.
BOX_COLOR = ERROR_COLOR["fp"]

#: Matches scripts/06_visualize_datasets.py, which writes into this same
#: reports/datasets/ directory; panels.save_figure otherwise defaults to 180.
FIGURE_DPI = 200


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
    save_figure(fig, out, f"collage_{name}", dpi=FIGURE_DPI)


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
    save_figure(fig, out, f"annotated_{name}", dpi=FIGURE_DPI)


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

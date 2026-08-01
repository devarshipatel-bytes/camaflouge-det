#!/usr/bin/env python3
"""Dataset composition and preprocessing visualisations, before any training.

Produces, under ``reports/datasets/``:

  01_composition.{png,svg}       kept vs excluded per dataset (paper Fig.2 style)
  02_splits.{png,svg}            train/val/test sizes per dataset
  03_foreground_area.{png,svg}   log-scale foreground-fraction distributions
  04_resolution.{png,svg}        image width x height scatter
  05_aspect_ratio.{png,svg}      aspect-ratio histograms
  06_mean_color.{png,svg}        mean RGB and mean local camouflage contrast
  07_cpd1k_families.{png,svg}    CPD1K camouflage-pattern-family breakdown
  08_mhcd_sam_qc.{png,svg}       MHCD SAM accept/reject breakdown
  strips/<name>.{png,svg}        Image | Mask | Overlay | Edge | Pose | Resized, 8 rows
  README.md                      counts + numbers referenced above, as text

Every chart uses the fixed, palette-validated dataset colors from
``chd.viz.colors`` (see that module's docstring for the validation run) and
never encodes identity by color alone — every dataset also gets a legend
entry and/or a direct label.

Example
-------
    python scripts/06_visualize_datasets.py --data-root data --out reports/datasets
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd._compat import zip_strict  # noqa: E402
from chd.data.manifest import mask_bbox  # noqa: E402
from chd.viz.colors import COMBINED_HATCH, DATASET_COLOR, DATASET_LABEL, HUMAN_PAIR, INK, SPLIT_COLOR  # noqa: E402

SOURCE_DATASETS = ("acd1k", "cpd1k", "camo_human", "mhcd")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": INK["muted"], "axes.labelcolor": INK["primary"],
    "text.color": INK["primary"], "xtick.color": INK["secondary"], "ytick.color": INK["secondary"],
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.6,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "svg.fonttype": "none",
})


def load_meta(root: Path) -> list[dict]:
    with (root / "meta.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["h"], row["w"] = int(row["h"]), int(row["w"])
        row["fg_frac"] = float(row["fg_frac"])
        row["n_components"] = int(row["n_components"])
        row["is_negative"] = row.get("is_negative", "0") == "1"
    return rows


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def clean_spines(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------------------------------
# 1. composition — kept vs excluded, paper Figure-2 style
# --------------------------------------------------------------------------

def fig_composition(data_root: Path, out: Path) -> dict:
    stats = {}
    # ACD1K, CPD1K: human-only sources, every image is a kept human sample.
    for name in ("acd1k", "cpd1k"):
        meta = load_meta(data_root / name)
        stats[name] = {"kept": len(meta), "excluded": 0}

    # CAMO: kept humans vs the recorded non-human negatives.
    camo_neg = data_root / "camo_human" / "negatives.txt"
    n_excluded = len([line for line in camo_neg.read_text().splitlines() if line.strip()]) if camo_neg.exists() else 0
    stats["camo_human"] = {"kept": len(load_meta(data_root / "camo_human")), "excluded": n_excluded}

    # MHCD: person-containing images kept (incl. presence negatives) vs the
    # other four military-vehicle classes discarded at the labeling stage.
    mhcd_meta = load_meta(data_root / "mhcd")
    stats["mhcd"] = {"kept": len(mhcd_meta), "excluded": 3000 - len(mhcd_meta)}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = list(stats)
    x = np.arange(len(names))
    kept = [stats[n]["kept"] for n in names]
    excluded = [stats[n]["excluded"] for n in names]
    ax.bar(x, kept, width=0.6, color=HUMAN_PAIR[0], label="Human / kept")
    ax.bar(x, excluded, width=0.6, bottom=kept, color=HUMAN_PAIR[1], label="Non-human / excluded")
    for i, (k, e) in enumerate(zip_strict(kept, excluded)):
        if e:
            ax.text(i, k + e + max(kept) * 0.01, f"{k}/{k+e}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(i, k + max(kept) * 0.01, f"{k}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [DATASET_LABEL[n] for n in names])
    ax.set_ylabel("Number of images")
    ax.set_title("Human vs. non-human/excluded images per dataset")
    ax.legend(frameon=False, loc="upper left")
    clean_spines(ax)
    save(fig, out, "01_composition")
    return stats


# --------------------------------------------------------------------------
# 2. split sizes
# --------------------------------------------------------------------------

def fig_splits(data_root: Path, out: Path) -> dict:
    counts = {}
    for name in (*SOURCE_DATASETS, "combined"):
        meta = load_meta(data_root / name)
        counts[name] = Counter(row["split"] for row in meta)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = list(counts)
    x = np.arange(len(names))
    bottoms = np.zeros(len(names))
    for split in ("train", "val", "test"):
        heights = [counts[n].get(split, 0) for n in names]
        bars = ax.bar(x, heights, width=0.6, bottom=bottoms, color=SPLIT_COLOR[split], label=split)
        for xi, h, b in zip_strict(x, heights, bottoms):
            if h:
                ax.text(xi, b + h / 2, str(h), ha="center", va="center", fontsize=8, color="white")
        bottoms += np.array(heights)
    ax.set_xticks(x, [DATASET_LABEL[n] for n in names])
    ax.set_ylabel("Number of images")
    ax.set_title("Split sizes per dataset")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    clean_spines(ax)
    save(fig, out, "02_splits")
    return counts


# --------------------------------------------------------------------------
# 3. foreground area distributions
# --------------------------------------------------------------------------

def fig_foreground_area(data_root: Path, out: Path) -> dict:
    fig, axes = plt.subplots(1, len(SOURCE_DATASETS), figsize=(15, 3.4), sharey=True)
    stats = {}
    bins = np.logspace(np.log10(0.001), np.log10(1.0), 30)
    for ax, name in zip_strict(axes, SOURCE_DATASETS):
        meta = [r for r in load_meta(data_root / name) if not r["is_negative"]]
        fracs = np.array([r["fg_frac"] for r in meta if r["fg_frac"] > 0])
        ax.hist(fracs, bins=bins, color=DATASET_COLOR[name])
        ax.set_xscale("log")
        ax.axvline(np.median(fracs), color=INK["primary"], linewidth=1, linestyle="--")
        ax.set_title(DATASET_LABEL[name])
        ax.set_xlabel("Foreground fraction (log scale)")
        clean_spines(ax)
        stats[name] = {"mean": float(fracs.mean()), "median": float(np.median(fracs)),
                       "p10": float(np.percentile(fracs, 10)), "p90": float(np.percentile(fracs, 90))}
    axes[0].set_ylabel("Number of images")
    fig.suptitle("Foreground (camouflaged-human) area as a fraction of image area", y=1.04)
    save(fig, out, "03_foreground_area")
    return stats


# --------------------------------------------------------------------------
# 4. resolution scatter
# --------------------------------------------------------------------------

def fig_resolution(data_root: Path, out: Path) -> None:
    # CPD1K is video-derived: all 1000 frames share one fixed 854x480
    # resolution, and 925 MHCD images happen to sit at that exact same point.
    # Even unfilled rings fully occlude under that much exact overlap, so
    # CPD1K gets its own oversized, high-zorder marker instead of fighting
    # draw order — which also correctly surfaces the real difference in
    # provenance (single-resolution video vs. varied-resolution stock photos).
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in SOURCE_DATASETS:
        if name == "cpd1k":
            continue
        meta = load_meta(data_root / name)
        w = np.array([r["w"] for r in meta])
        h = np.array([r["h"] for r in meta])
        ax.scatter(w, h, s=22, alpha=0.6, facecolors="none", edgecolors=DATASET_COLOR[name],
                  linewidths=1.1, label=DATASET_LABEL[name])

    cpd_meta = load_meta(data_root / "cpd1k")
    cw, ch = cpd_meta[0]["w"], cpd_meta[0]["h"]
    ax.scatter([cw], [ch], s=260, marker="*", color=DATASET_COLOR["cpd1k"], edgecolors=INK["primary"],
              linewidths=0.8, zorder=10, label=f"CPD1K (all {len(cpd_meta)} @ {cw}x{ch})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Width (px, log scale)")
    ax.set_ylabel("Height (px, log scale)")
    ax.set_title("Native image resolution")
    ax.legend(frameon=False, markerscale=2)
    clean_spines(ax)
    save(fig, out, "04_resolution")


# --------------------------------------------------------------------------
# 5. aspect ratio
# --------------------------------------------------------------------------

def fig_aspect_ratio(data_root: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.4, 2.4, 41)
    for name in SOURCE_DATASETS:
        meta = load_meta(data_root / name)
        ratio = np.array([r["w"] / r["h"] for r in meta])
        ax.hist(ratio, bins=bins, histtype="step", linewidth=1.8, color=DATASET_COLOR[name], label=DATASET_LABEL[name])
    ax.axvline(1.0, color=INK["muted"], linewidth=1, linestyle=":")
    ax.set_xlabel("Aspect ratio (width / height)")
    ax.set_ylabel("Number of images")
    ax.set_title("Aspect-ratio distribution")
    ax.legend(frameon=False)
    clean_spines(ax)
    save(fig, out, "05_aspect_ratio")


# --------------------------------------------------------------------------
# 6. mean color + local camouflage contrast
# --------------------------------------------------------------------------

def local_contrast(image: np.ndarray, mask: np.ndarray, dilate_px: int = 15) -> float:
    """Mean |fg color - surrounding ring color|, normalised to [0, 1].

    Low values mean the foreground blends into its immediate surroundings —
    exactly what "well camouflaged" should look like as a number.
    """
    from scipy import ndimage
    solid = mask > 0
    if not solid.any() or solid.all():
        return float("nan")
    ring = ndimage.binary_dilation(solid, iterations=dilate_px) & ~solid
    if not ring.any():
        return float("nan")
    fg_color = image[solid].mean(axis=0)
    ring_color = image[ring].mean(axis=0)
    return float(np.abs(fg_color - ring_color).mean() / 255.0)


def fig_mean_color(data_root: Path, out: Path, sample_per_dataset: int = 120) -> dict:
    rng = np.random.default_rng(0)
    stats = {}
    fig, (ax_rgb, ax_contrast) = plt.subplots(1, 2, figsize=(11, 4))

    for name in SOURCE_DATASETS:
        meta = [r for r in load_meta(data_root / name) if not r["is_negative"]]
        sample = rng.choice(meta, size=min(sample_per_dataset, len(meta)), replace=False)
        rgb_means, contrasts = [], []
        for row in sample:
            stem = row["stem"]
            with Image.open(data_root / name / "images" / f"{stem}.jpg") as im:
                image = np.asarray(im.convert("RGB"), dtype=np.float32)
            with Image.open(data_root / name / "masks" / f"{stem}.png") as im:
                mask = np.asarray(im.convert("L"))
            rgb_means.append(image.reshape(-1, 3).mean(axis=0))
            c = local_contrast(image, mask)
            if not np.isnan(c):
                contrasts.append(c)
        rgb_means = np.array(rgb_means)
        stats[name] = {"mean_rgb": rgb_means.mean(axis=0).tolist(), "mean_contrast": float(np.mean(contrasts))}

    names = list(stats)
    swatches = np.array([stats[n]["mean_rgb"] for n in names]) / 255.0
    ax_rgb.imshow(swatches[:, None, :], aspect="auto")  # default pixel coords: row 0 at top
    ax_rgb.set_yticks(np.arange(len(names)), [DATASET_LABEL[n] for n in names])
    ax_rgb.set_xticks([])
    ax_rgb.set_title(f"Mean image color (n={sample_per_dataset}/dataset)")

    contrast_vals = [stats[n]["mean_contrast"] for n in names]
    bars = ax_contrast.bar(np.arange(len(names)), contrast_vals, color=[DATASET_COLOR[n] for n in names])
    ax_contrast.set_xticks(np.arange(len(names)), [DATASET_LABEL[n] for n in names])
    ax_contrast.set_ylabel("Mean |fg - surrounding ring| (normalised)")
    ax_contrast.set_title("Local camouflage contrast (lower = better blended)")
    clean_spines(ax_contrast)
    for i, v in enumerate(contrast_vals):
        ax_contrast.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    save(fig, out, "06_mean_color")
    return stats


# --------------------------------------------------------------------------
# 7. CPD1K camouflage-pattern families
# --------------------------------------------------------------------------

def fig_cpd1k_families(data_root: Path, out: Path) -> dict:
    labels_path = Path("dataset/cpd1k/pattern_labels.csv")
    if not labels_path.exists():
        return {}
    with labels_path.open() as fh:
        rows = list(csv.DictReader(fh))
    counts = Counter(r["family_name"] for r in rows)
    order = sorted(counts, key=lambda k: -counts[k])
    values = [counts[k] for k in order]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    cmap = plt.get_cmap("Blues")
    colors = [cmap(0.35 + 0.55 * i / max(1, len(order) - 1)) for i in range(len(order))]
    wedges, _, autotexts = ax.pie(
        values, labels=order, autopct=lambda p: f"{p:.0f}%", colors=colors,
        textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white", "linewidth": 1},
    )
    ax.set_title(f"CPD1K camouflage-pattern families (n={sum(values)})")
    save(fig, out, "07_cpd1k_families")
    return dict(counts)


# --------------------------------------------------------------------------
# 8. MHCD SAM QC breakdown
# --------------------------------------------------------------------------

def fig_mhcd_qc(data_root: Path, out: Path) -> dict:
    qc_path = data_root / "mhcd" / "sam_qc.csv"
    if not qc_path.exists():
        return {}
    with qc_path.open() as fh:
        rows = list(csv.DictReader(fh))
    accepted = sum(r["verdict"] == "accept" for r in rows)
    reasons = Counter(r["reason"].split("(")[0] for r in rows if r["verdict"] == "reject")
    reasons["accept"] = accepted

    order = ["accept"] + sorted((k for k in reasons if k != "accept"), key=lambda k: -reasons[k])
    values = [reasons[k] for k in order]
    colors = [HUMAN_PAIR[0]] + list(plt.get_cmap("Oranges")(np.linspace(0.4, 0.85, len(order) - 1)))

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(order, values, color=colors)
    ax.set_ylabel("Number of person boxes")
    ax.set_title(f"MHCD SAM pseudo-mask QC ({accepted}/{len(rows)} = {accepted/len(rows):.1%} accepted)")
    for b, v in zip_strict(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.01, str(v), ha="center", va="bottom", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    clean_spines(ax)
    save(fig, out, "08_mhcd_sam_qc")
    return dict(reasons)


# --------------------------------------------------------------------------
# preprocessing strips
# --------------------------------------------------------------------------

def fig_preprocessing_strip(data_root: Path, name: str, out: Path, n_rows: int = 8, img_size: int = 352) -> None:
    root = data_root / name
    meta = [r for r in load_meta(root) if not r["is_negative"]]
    rng = np.random.default_rng(1)
    sample = rng.choice(meta, size=min(n_rows, len(meta)), replace=False)

    cols = ["Image", "GT mask", "Overlay", "Edge", "Pose heatmap", f"Resized {img_size}²"]
    fig, axes = plt.subplots(len(sample), len(cols), figsize=(2.1 * len(cols), 2.1 * len(sample)))
    if len(sample) == 1:
        axes = axes[None, :]

    for row_ax, row in zip_strict(axes, sample):
        stem = row["stem"]
        image = np.asarray(Image.open(root / "images" / f"{stem}.jpg").convert("RGB"))
        mask = np.asarray(Image.open(root / "masks" / f"{stem}.png").convert("L"))
        edge = np.asarray(Image.open(root / "edges" / f"{stem}.png").convert("L"))
        pose_path = root / "pose" / f"{stem}.npy"
        pose = np.load(pose_path).max(axis=0) if pose_path.exists() else np.zeros((4, 4))

        overlay = image.copy()
        solid = mask > 0
        overlay[solid] = (0.45 * overlay[solid] + 0.55 * np.array([60, 255, 120])).astype(np.uint8)

        resized = np.asarray(Image.fromarray(image).resize((img_size, img_size), Image.BILINEAR))

        for ax, content, is_gray in zip_strict(
            row_ax, (image, mask, overlay, edge, pose, resized),
            (False, True, False, True, "heat", False),
        ):
            if is_gray == "heat":
                ax.imshow(content, cmap="inferno", vmin=0, vmax=1)
            elif is_gray:
                ax.imshow(content, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(content)
            ax.set_xticks([])
            ax.set_yticks([])
        row_ax[0].set_ylabel(stem, fontsize=7)

    for ax, title in zip(axes[0], cols):
        ax.set_title(title, fontsize=10)
    fig.subplots_adjust(top=0.94, hspace=0.06, wspace=0.05)
    fig.suptitle(f"{DATASET_LABEL[name]} — preprocessing pipeline sample", y=0.97, fontsize=13)
    save(fig, out / "strips", name)


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("reports/datasets"))
    parser.add_argument("--strip-rows", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=352)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("[viz] composition...")
    composition = fig_composition(args.data_root, args.out)
    print("[viz] splits...")
    splits = fig_splits(args.data_root, args.out)
    print("[viz] foreground area...")
    fg_stats = fig_foreground_area(args.data_root, args.out)
    print("[viz] resolution scatter...")
    fig_resolution(args.data_root, args.out)
    print("[viz] aspect ratio...")
    fig_aspect_ratio(args.data_root, args.out)
    print("[viz] mean color + contrast (this reads a sample of real images, slower)...")
    color_stats = fig_mean_color(args.data_root, args.out)
    print("[viz] CPD1K families...")
    cpd1k_families = fig_cpd1k_families(args.data_root, args.out)
    print("[viz] MHCD SAM QC...")
    mhcd_qc = fig_mhcd_qc(args.data_root, args.out)
    print("[viz] preprocessing strips...")
    for name in SOURCE_DATASETS:
        fig_preprocessing_strip(args.data_root, name, args.out, n_rows=args.strip_rows, img_size=args.img_size)

    lines = ["# Dataset report\n"]
    lines.append("## Composition (kept vs excluded)\n")
    for name, s in composition.items():
        lines.append(f"- **{DATASET_LABEL[name]}**: {s['kept']} kept, {s['excluded']} excluded")
    lines.append("\n## Splits\n")
    for name, c in splits.items():
        lines.append(f"- **{DATASET_LABEL[name]}**: train={c.get('train',0)} val={c.get('val',0)} test={c.get('test',0)}")
    lines.append("\n## Foreground area\n")
    for name, s in fg_stats.items():
        lines.append(f"- **{DATASET_LABEL[name]}**: mean={s['mean']:.4f} median={s['median']:.4f} "
                     f"p10={s['p10']:.4f} p90={s['p90']:.4f}")
    lines.append("\n## Local camouflage contrast (lower = better blended)\n")
    for name, s in color_stats.items():
        lines.append(f"- **{DATASET_LABEL[name]}**: {s['mean_contrast']:.4f}")
    if cpd1k_families:
        lines.append("\n## CPD1K camouflage families\n")
        for family, n in sorted(cpd1k_families.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {family}: {n}")
    if mhcd_qc:
        lines.append("\n## MHCD SAM QC\n")
        for reason, n in sorted(mhcd_qc.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {reason}: {n}")
    (args.out / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\n[viz] done — see {args.out}/")


if __name__ == "__main__":
    main()

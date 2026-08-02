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

#!/usr/bin/env python3
"""Visualize the end-to-end model pipeline: input -> extraction points -> output.

One row per sample image, one column per named extraction point in the
architecture — not every conv layer, just the module boundaries that matter
for explaining the design (backbone -> FDM -> SFA -> OSNeck -> AER ->
decoder -> predicted mask). Built on ``CHDNet.forward(..., return_intermediates=True)``
(``src/chd/models/chdnet.py``) — that flag is the "enable/disable" switch:
off (default) costs nothing during normal training/inference, on captures
every intermediate this script renders.

Works with or without a trained checkpoint:

  - No ``--checkpoint``: uses the ImageNet-pretrained backbone with
    freshly-initialised heads. Useful right now, before any of the 5 models
    finish training, to sanity-check the pipeline and layout.
  - ``--checkpoint runs/acd1k/best.pth``: uses the real trained weights (EMA
    weights if the checkpoint has them, matching what run_validation uses in
    train.py) — this is what you want for the actual architecture-report
    figure once training is done.

Example
-------
    # structural demo, no trained weights needed yet
    python scripts/07_visualize_pipeline.py --dataset acd1k --num-images 5

    # real trained-model figure
    python scripts/07_visualize_pipeline.py --checkpoint runs/acd1k/best.pth \\
        --dataset acd1k --split test --num-images 5 --out reports/architecture/acd1k_pipeline.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd.data.dataset import AugmentConfig, CHDDataset  # noqa: E402
from chd.models.chdnet import CHDNet  # noqa: E402

# The finest pyramid level (index 0) is the one most directly comparable to
# human-scale detail in the image, and every intermediate at this level
# shares the same spatial resolution (img_size / 4) — no resizing needed to
# lay panels out side by side.
LEVEL = 0

COLUMNS = [
    ("Input", "rgb"),
    ("Backbone F1", "heat"),
    ("FDM low-freq", "heat"),
    ("FDM high-freq", "heat"),
    ("SFA fused", "heat"),
    ("OSNeck", "heat"),
    ("AER (pose-gated)", "heat"),
    ("Decoder x1", "heat"),
    ("Predicted mask", "overlay"),
]


def channel_heat(t: torch.Tensor) -> np.ndarray:
    """(C, H, W) -> normalised (H, W) heatmap: mean absolute activation across channels."""
    heat = t.abs().mean(dim=0).cpu().numpy()
    lo, hi = heat.min(), heat.max()
    return (heat - lo) / (hi - lo) if hi > lo else np.zeros_like(heat)


def load_model(args: argparse.Namespace) -> tuple[CHDNet, dict]:
    model = CHDNet(backbone=args.backbone, pretrained=(args.checkpoint is None), os_streams=args.os_streams)
    meta = {"checkpoint": None, "weights": "ImageNet backbone + random heads (no checkpoint given)"}
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        if ckpt.get("ema"):
            model.load_state_dict(ckpt["ema"])
            meta = {"checkpoint": args.checkpoint, "weights": "EMA weights"}
        else:
            model.load_state_dict(ckpt["model"])
            meta = {"checkpoint": args.checkpoint, "weights": "raw (non-EMA) weights"}
        meta["epoch"] = ckpt.get("epoch")
        meta["best_s_alpha"] = ckpt.get("best_s_alpha")
    model.to(args.device).eval()
    return model, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to a train.py checkpoint (last.pth/best.pth); omit for an untrained demo")
    parser.add_argument("--backbone", default="res2net50_26w_4s")
    parser.add_argument("--os-streams", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--dataset", default="acd1k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-images", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: reports/architecture/<dataset>_pipeline.png")
    args = parser.parse_args()

    out = args.out or Path("reports/architecture") / f"{args.dataset}_pipeline.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    model, meta = load_model(args)
    print(f"[pipeline-viz] weights: {meta['weights']}"
          + (f" (epoch {meta['epoch']}, best S_alpha={meta['best_s_alpha']:.4f})" if meta.get("epoch") is not None else ""))

    root = args.data_root / args.dataset
    dataset = CHDDataset(root, args.split, img_size=args.img_size, augment=AugmentConfig(enabled=False))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=min(args.num_images, len(dataset)), replace=False)
    print(f"[pipeline-viz] {len(indices)} image(s) from {args.dataset}/{args.split}: "
          f"{[dataset.stems[i] for i in indices]}")

    n_rows, n_cols = len(indices), len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, idx in enumerate(indices):
        item = dataset[int(idx)]
        image = item["image"].unsqueeze(0).to(args.device)
        pose = item["pose"].unsqueeze(0).to(args.device)

        with torch.no_grad():
            out_full = model(image, pose, return_intermediates=True)
        inter = out_full["intermediates"]
        presence = torch.sigmoid(out_full["presence_logit"]).item()
        mask_prob = CHDNet.predict_mask(out_full)[0, 0].cpu().numpy()

        image_np = (item["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        panels = {
            "Input": image_np,
            "Backbone F1": channel_heat(inter["backbone"][LEVEL][0]),
            "FDM low-freq": channel_heat(inter["fdm_lf"][LEVEL][0]),
            "FDM high-freq": channel_heat(inter["fdm_hf"][LEVEL][0]),
            "SFA fused": channel_heat(inter["sfa"][LEVEL][0]),
            "OSNeck": channel_heat(inter["osneck"][LEVEL][0]),
            "AER (pose-gated)": channel_heat(inter["aer"][LEVEL][0]),
            "Decoder x1": channel_heat(inter["decoder_levels"][LEVEL][0]),
        }

        for col, (name, kind) in enumerate(COLUMNS):
            ax = axes[row, col]
            if name == "Predicted mask":
                overlay = image_np.copy()
                solid = mask_prob > 0.5
                overlay[solid] = (0.4 * overlay[solid] + 0.6 * np.array([60, 255, 120])).astype(np.uint8)
                ax.imshow(overlay)
                if row == 0:
                    ax.set_title(f"{name}\n", fontsize=9)
                ax.set_xlabel(f"presence={presence:.2f}", fontsize=8)
            elif kind == "rgb":
                ax.imshow(panels[name])
            else:
                ax.imshow(panels[name], cmap="inferno", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0 and name != "Predicted mask":
                ax.set_title(name, fontsize=9)
        axes[row, 0].set_ylabel(dataset.stems[int(idx)], fontsize=7)

    fig.suptitle(
        f"OS-Res2Net-CHDNet pipeline — level-1 (finest) extraction points — {meta['weights']}",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[pipeline-viz] wrote {out} and {out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()

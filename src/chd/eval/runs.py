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

#!/usr/bin/env python3
"""Train OS-Res2Net-CHDNet on one dataset (or the combined one).

Every hyperparameter is a CLI flag with a sensible 16GB-GPU default; nothing
is hard-coded, so the exact same script trains any of the 5 models. This is
the file to run on the 16GB training machine — it never needs code changes,
only different flags.

Examples
--------
    # full run on the 16GB box
    python train.py --dataset acd1k --epochs 100 --batch-size 16 --img-size 352 \\
        --lr 1e-4 --amp --ema --multi-scale 0.75 1.0 1.25 --out runs/acd1k

    # quick smoke test on a small GPU (this dev machine)
    python train.py --dataset acd1k --epochs 1 --batch-size 2 --img-size 256 \\
        --limit 20 --workers 0 --out /tmp/smoke

    # resume an interrupted run
    python train.py --dataset acd1k --resume runs/acd1k/last.pth --out runs/acd1k
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from chd._compat import zip_strict  # noqa: E402
from chd.data.dataset import AugmentConfig, CHDDataset, batch_resize, collate  # noqa: E402
from chd.losses import CHDLoss  # noqa: E402
from chd.metrics import mae as compute_mae  # noqa: E402
from chd.metrics import s_measure  # noqa: E402
from chd.models.chdnet import CHDNet  # noqa: E402

DATASETS = ("acd1k", "cpd1k", "camo_human", "mhcd", "combined")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = p.add_argument_group("data")
    data.add_argument("--dataset", required=True, choices=DATASETS)
    data.add_argument("--data-root", type=Path, default=Path("data"))
    data.add_argument("--img-size", type=int, default=352, help="must be a multiple of 32")
    data.add_argument("--multi-scale", type=float, nargs="*", default=[1.0],
                      help="e.g. --multi-scale 0.75 1.0 1.25; a scale is chosen per step, not per sample")
    data.add_argument("--workers", type=int, default=8)
    data.add_argument("--limit", type=int, default=None, help="truncate train/val to N samples each (smoke tests)")
    data.add_argument("--no-pose", action="store_true", help="zero out the pose prior everywhere")
    data.add_argument("--hflip-prob", type=float, default=0.5, help="train-time horizontal flip probability")
    data.add_argument("--rotate-deg", type=float, default=15.0, help="train-time random rotation, +/- degrees")
    data.add_argument("--scale-min", type=float, default=0.75, help="train-time random-crop-scale lower bound")
    data.add_argument("--scale-max", type=float, default=1.25, help="train-time random-crop-scale upper bound")
    data.add_argument("--color-jitter", type=float, default=0.2,
                      help="brightness/contrast/saturation jitter strength, 0 disables")
    data.add_argument("--no-augment", action="store_true", help="disable all train-time augmentation")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("--epochs", type=int, default=100)
    opt.add_argument("--batch-size", type=int, default=16)
    opt.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation, effective bs = batch*accum")
    opt.add_argument("--lr", type=float, default=1e-4)
    opt.add_argument("--backbone-lr-mult", type=float, default=0.1)
    opt.add_argument("--weight-decay", type=float, default=1e-4)
    opt.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    opt.add_argument("--momentum", type=float, default=0.9, help="only used with --optimizer sgd")
    opt.add_argument("--scheduler", choices=("cosine", "step", "none"), default="cosine")
    opt.add_argument("--warmup-epochs", type=int, default=5)
    opt.add_argument("--step-size", type=int, default=30, help="only used with --scheduler step")
    opt.add_argument("--step-gamma", type=float, default=0.1, help="only used with --scheduler step")
    opt.add_argument("--grad-clip", type=float, default=0.5, help="0 disables clipping")

    model_g = p.add_argument_group("model")
    model_g.add_argument("--backbone", default="res2net50_26w_4s", choices=("res2net50_26w_4s", "tiny_test"))
    model_g.add_argument("--no-pretrained", action="store_true", help="random-init backbone instead of ImageNet")
    model_g.add_argument("--freeze-bn-epochs", type=int, default=5,
                         help="keep backbone BatchNorm running stats frozen for the first N epochs")
    model_g.add_argument("--os-streams", type=int, default=4, help="number of OSBlock omni-scale streams")

    loss_g = p.add_argument_group("loss")
    loss_g.add_argument("--side-weights", type=float, nargs=4, default=[0.4, 0.6, 0.8, 1.0])
    loss_g.add_argument("--edge-weight", type=float, default=1.0)
    loss_g.add_argument("--presence-weight", type=float, default=0.5)
    loss_g.add_argument("--no-presence", action="store_true", help="set presence_weight to 0 regardless of above")

    run = p.add_argument_group("run")
    run.add_argument("--amp", action="store_true", help="mixed-precision (fp16) training")
    run.add_argument("--ema", action="store_true", help="track an EMA of weights, used for val/checkpoints")
    run.add_argument("--ema-decay", type=float, default=0.999)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--auto-batch", action="store_true",
                     help="probe free VRAM and shrink --batch-size until a step fits")
    run.add_argument("--resume", type=str, default=None, help="checkpoint path, or 'auto' for <out>/last.pth")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--log-every", type=int, default=20)
    run.add_argument("--val-every", type=int, default=1, help="run validation every N epochs")
    run.add_argument("--save-every", type=int, default=5,
                     help="also save a durable, never-overwritten checkpoint every N epochs "
                          "(separate from last.pth/best.pth, which are overwritten each epoch)")
    return p


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(args: argparse.Namespace, split: str) -> DataLoader:
    root = args.data_root / args.dataset
    augment = AugmentConfig(
        enabled=(split == "train" and not args.no_augment),
        hflip_prob=args.hflip_prob,
        rotate_deg=args.rotate_deg,
        scale_range=(args.scale_min, args.scale_max),
        color_jitter=args.color_jitter,
    )
    dataset = CHDDataset(root, split, img_size=args.img_size, augment=augment, require_pose=not args.no_pose)
    if args.limit:
        dataset = Subset(dataset, list(range(min(args.limit, len(dataset)))))
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(split == "train"),
        num_workers=args.workers, collate_fn=collate, drop_last=(split == "train"),
        pin_memory=(args.device == "cuda"), persistent_workers=(args.workers > 0),
    )


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        (backbone_params if name.startswith("backbone.") else other_params).append(param)
    groups = [
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": other_params, "lr": args.lr},
    ]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    def cosine_with_warmup(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_with_warmup)

    def step_with_warmup(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        epoch = step // steps_per_epoch
        return args.step_gamma ** (epoch // args.step_size)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=step_with_warmup)


def build_grad_scaler(enabled: bool):
    """``torch.amp.GradScaler`` only exists from roughly torch 2.3 onward;
    older installs (seen in practice on a fresh Windows pip install) only
    have ``torch.cuda.amp.GradScaler``. Try the current API, fall back to
    the legacy one, so this runs on whatever torch happens to be installed
    on the training machine without pinning an exact version.
    """
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def is_oom_error(exc: Exception) -> bool:
    """``torch.cuda.OutOfMemoryError`` doesn't exist before torch 2.0."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_type) or "out of memory" in str(exc).lower()


def set_backbone_bn_frozen(model: CHDNet, frozen: bool) -> None:
    for module in model.backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval() if frozen else module.train()
            module.weight.requires_grad_(not frozen)
            module.bias.requires_grad_(not frozen)


class EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for shadow_p, p in zip_strict(self.shadow.parameters(), model.parameters()):
            shadow_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for shadow_b, b in zip_strict(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(b)


def auto_shrink_batch_size(args: argparse.Namespace, model: nn.Module, device: str) -> int:
    """Halve --batch-size until one forward+backward step fits in free VRAM."""
    if device != "cuda":
        return args.batch_size
    batch_size = args.batch_size
    criterion = CHDLoss()
    while batch_size >= 1:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            image = torch.randn(batch_size, 3, args.img_size, args.img_size, device=device)
            pose = torch.zeros(batch_size, 17, args.img_size // 4, args.img_size // 4, device=device)
            mask = torch.zeros(batch_size, 1, args.img_size, args.img_size, device=device)
            with torch.autocast(device_type="cuda", enabled=args.amp):
                out = model(image, pose)
                loss = criterion(out, mask, mask, torch.zeros(batch_size, device=device))["total"]
            loss.backward()
            model.zero_grad(set_to_none=True)
            del image, pose, mask, out, loss
            torch.cuda.empty_cache()
            print(f"[auto-batch] batch_size={batch_size} fits "
                  f"(peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB)")
            return batch_size
        except RuntimeError as exc:
            if not is_oom_error(exc):
                raise
            torch.cuda.empty_cache()
            print(f"[auto-batch] batch_size={batch_size} OOM, halving")
            batch_size //= 2
    raise RuntimeError("even batch_size=1 does not fit in available VRAM at this --img-size")


# --------------------------------------------------------------------------
# train / validate
# --------------------------------------------------------------------------

def run_epoch_train(model, loader, criterion, optimizer, scheduler, scaler, ema, args, epoch: int, global_step: int):
    model.train()
    set_backbone_bn_frozen(model, frozen=(epoch < args.freeze_bn_epochs))

    totals = {"total": 0.0, "final": 0.0, "side": 0.0, "edge": 0.0, "presence": 0.0}
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        scale = args.multi_scale[torch.randint(len(args.multi_scale), (1,)).item()]
        batch = batch_resize(batch, scale)
        image = batch["image"].to(args.device, non_blocking=True)
        mask = batch["mask"].to(args.device, non_blocking=True)
        edge = batch["edge"].to(args.device, non_blocking=True)
        pose = torch.zeros_like(batch["pose"]) if args.no_pose else batch["pose"]
        pose = pose.to(args.device, non_blocking=True)
        is_negative = batch["is_negative"].to(args.device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(args.amp and args.device == "cuda")):
            outputs = model(image, pose)
            losses = criterion(outputs, mask, edge, is_negative)
            loss = losses["total"] / args.accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accum_steps == 0:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)
            global_step += 1

        for key in totals:
            totals[key] += losses[key].item() if key != "total" else losses["total"].item()
        n_batches += 1

        if (step + 1) % args.log_every == 0:
            lr = optimizer.param_groups[-1]["lr"]
            elapsed = time.time() - t0
            print(f"  epoch {epoch} step {step+1}/{len(loader)} "
                  f"loss={totals['total']/n_batches:.4f} lr={lr:.2e} "
                  f"({elapsed/n_batches:.2f}s/it)")

    return {k: v / max(1, n_batches) for k, v in totals.items()}, global_step


def plot_history(history_path: Path, out_dir: Path) -> None:
    """Re-render the training-curve plots from history.csv — cheap enough to
    call after every epoch, so a killed run still leaves an up-to-date plot,
    not just whatever existed at the last completed run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]

    # Palette matches chd.viz.colors' validated categorical set (see that
    # module's docstring for the validation run) so plots from this training
    # run and the dataset/architecture reports read as one consistent system.
    color_train, color_mae, color_s_alpha = "#0072B2", "#E69F00", "#009E73"

    fig, (ax_loss, ax_metric) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    for ax in (ax_loss, ax_metric):
        ax.set_facecolor("white")
        ax.grid(True, color="#e0e0e0", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_loss.plot(epochs, [float(r["train_total"]) for r in rows], color=color_train, linewidth=1.8, label="train loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend(frameon=False)

    val_mae = [float(r["val_mae"]) for r in rows if r["val_mae"] not in ("", "nan")]
    val_epochs = [int(r["epoch"]) for r in rows if r["val_mae"] not in ("", "nan")]
    val_s = [float(r["val_s_alpha"]) for r in rows if r["val_s_alpha"] not in ("", "nan")]
    ax_metric.plot(val_epochs, val_mae, color=color_mae, marker="o", markersize=3, linewidth=1.5, label="val MAE (lower better)")
    ax_metric.plot(val_epochs, val_s, color=color_s_alpha, marker="o", markersize=3, linewidth=1.5, label="val S_alpha (higher better)")
    ax_metric.set_ylabel("Metric value")
    ax_metric.set_xlabel("Epoch")
    ax_metric.set_title("Validation metrics")
    ax_metric.legend(frameon=False)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "training_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "training_curves.svg", bbox_inches="tight")
    plt.close(fig)


def write_summary(args: argparse.Namespace, model: nn.Module, history_path: Path,
                  best_s_alpha: float, last_epoch: int, out_path: Path) -> None:
    with history_path.open() as fh:
        rows = list(csv.DictReader(fh))
    n_params = sum(p.numel() for p in model.parameters())
    summary = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs_completed": last_epoch + 1,
        "epochs_requested": args.epochs,
        "best_s_alpha": best_s_alpha,
        "final_epoch_metrics": rows[-1] if rows else None,
        "n_params": n_params,
        "n_params_millions": round(n_params / 1e6, 2),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    out_path.write_text(json.dumps(summary, indent=2))


@torch.no_grad()
def run_validation(model, loader, args) -> dict:
    model.eval()
    mae_sum, s_sum, n = 0.0, 0.0, 0
    for batch in loader:
        image = batch["image"].to(args.device)
        mask = batch["mask"].to(args.device)
        pose = torch.zeros_like(batch["pose"]) if args.no_pose else batch["pose"]
        pose = pose.to(args.device)

        outputs = model(image, pose)
        pred = CHDNet.predict_mask(outputs).cpu().numpy()
        gt = mask.cpu().numpy()

        for i in range(pred.shape[0]):
            mae_sum += compute_mae(pred[i, 0], gt[i, 0])
            s_sum += s_measure(pred[i, 0], gt[i, 0])
            n += 1

    return {"mae": mae_sum / max(1, n), "s_alpha": s_sum / max(1, n)}


# --------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    if args.img_size % 32 != 0:
        raise SystemExit(f"--img-size must be a multiple of 32, got {args.img_size}")
    if args.no_presence:
        args.presence_weight = 0.0

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    train_loader = make_loader(args, "train")
    val_loader = make_loader(args, "val")
    print(f"[train.py] dataset={args.dataset} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"img_size={args.img_size} batch_size={args.batch_size}")

    model = CHDNet(backbone=args.backbone, pretrained=not args.no_pretrained, os_streams=args.os_streams)
    model.to(args.device)

    if args.auto_batch:
        args.batch_size = auto_shrink_batch_size(args, model, args.device)
        train_loader = make_loader(args, "train")
        val_loader = make_loader(args, "val")

    if len(train_loader) == 0:
        raise SystemExit(
            f"train split has {len(train_loader.dataset)} samples but batch_size={args.batch_size} "
            "with drop_last=True yields zero batches per epoch. Lower --batch-size, raise --limit, "
            "or use more data — training would otherwise silently do nothing every epoch."
        )

    criterion = CHDLoss(
        side_weights=tuple(args.side_weights), edge_weight=args.edge_weight, presence_weight=args.presence_weight,
    )
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch=max(1, len(train_loader) // args.accum_steps))
    scaler = build_grad_scaler(enabled=(args.amp and args.device == "cuda"))
    ema = EMA(model, args.ema_decay) if args.ema else None

    start_epoch, global_step, best_s_alpha = 0, 0, -1.0
    resume_path = None
    if args.resume == "auto":
        candidate = args.out / "last.pth"
        resume_path = candidate if candidate.exists() else None
    elif args.resume:
        resume_path = Path(args.resume)

    if resume_path and resume_path.exists():
        # weights_only=False: this checkpoint is written by this same script
        # (never an untrusted download) and stores an argparse.Namespace with
        # Path objects inside, which the default weights_only=True rejects.
        ckpt = torch.load(resume_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if ema is not None and ckpt.get("ema"):
            ema.shadow.load_state_dict(ckpt["ema"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_s_alpha = ckpt["best_s_alpha"]
        print(f"[train.py] resumed from {resume_path} at epoch {start_epoch}")

    history_path = args.out / "history.csv"
    is_new_history = not history_path.exists()
    with history_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new_history:
            writer.writerow(["epoch", "train_total", "train_final", "train_side", "train_edge",
                             "train_presence", "val_mae", "val_s_alpha", "lr", "seconds"])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_losses, global_step = run_epoch_train(
            model, train_loader, criterion, optimizer, scheduler, scaler, ema, args, epoch, global_step,
        )

        val_metrics = {"mae": float("nan"), "s_alpha": float("nan")}
        if (epoch + 1) % args.val_every == 0:
            eval_model = ema.shadow if ema is not None else model
            val_metrics = run_validation(eval_model, val_loader, args)
            print(f"[epoch {epoch}] train_loss={train_losses['total']:.4f} "
                  f"val_mae={val_metrics['mae']:.4f} val_s_alpha={val_metrics['s_alpha']:.4f}")

        with history_path.open("a", newline="") as fh:
            csv.writer(fh).writerow([
                epoch, train_losses["total"], train_losses["final"], train_losses["side"],
                train_losses["edge"], train_losses["presence"], val_metrics["mae"], val_metrics["s_alpha"],
                optimizer.param_groups[-1]["lr"], round(time.time() - t0, 1),
            ])

        checkpoint = {
            "epoch": epoch, "global_step": global_step, "best_s_alpha": best_s_alpha,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "ema": ema.shadow.state_dict() if ema is not None else None,
            "args": vars(args),
        }
        torch.save(checkpoint, args.out / "last.pth")

        if val_metrics["s_alpha"] > best_s_alpha:
            best_s_alpha = val_metrics["s_alpha"]
            checkpoint["best_s_alpha"] = best_s_alpha
            torch.save(checkpoint, args.out / "best.pth")
            print(f"[epoch {epoch}] new best S_alpha={best_s_alpha:.4f}, saved best.pth")

        # Durable, never-overwritten snapshot every --save-every epochs (and
        # always on the final epoch) — last.pth/best.pth are safe to resume
        # from but get replaced every epoch, so a bad late run can't lose
        # access to a known-good earlier state without these.
        is_last_epoch = epoch == args.epochs - 1
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or is_last_epoch):
            checkpoints_dir = args.out / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoints_dir / f"epoch_{epoch:04d}.pth")
            print(f"[epoch {epoch}] saved periodic checkpoint checkpoints/epoch_{epoch:04d}.pth")

        plot_history(history_path, args.out / "plots")
        write_summary(args, model, history_path, best_s_alpha, epoch, args.out / "summary.json")

    print(f"[train.py] done. best S_alpha={best_s_alpha:.4f}. history: {history_path}")


if __name__ == "__main__":
    main()

# Evaluation & Visualization Stage — Design

Date: 2026-08-02
Status: approved, ready for implementation planning

## Context

Training is complete for most datasets. Checkpoints live on a **second machine**
that pulls this repo; nothing can be tested against real weights here. Therefore
every script must discover its own configuration from the checkpoint rather than
relying on paths, folder names, or re-typed flags.

Observed layout on that machine (from the user's screenshot):

```
runs/
  acd1k/{best.pth,last.pth,config.json,history.csv,summary.json,checkpoints/,plots/}
  acd1k2/
  camo_human/  camo_human2/  camo_human3/  camo-human-final/
  combined/    cpd1k/
```

Two facts drive the design:

1. **Folder names are unreliable.** `camo-human-final` uses hyphens; the dataset
   key is `camo_human`. There is no parseable convention.
2. **`train.py:629` saves `"args": vars(args)` into every checkpoint.** Dataset,
   architecture, backbone, `os_streams`, `unet_encoder`, `img_size` and `no_pose`
   are all recoverable from the checkpoint itself.

`runs/acd1k/summary.json` also shows `img_size: 640` — not the 352 default — so
hardcoding image size would silently degrade every reported metric.

Test split sizes: acd1k 329, cpd1k 215, camo_human 112, mhcd 494, combined 1150.

Negatives (`is_negative` in `meta.csv`, which with `splits/` is all the eval
pipeline reads): acd1k **0**, cpd1k **0**, camo_human **0**, mhcd **376** (69 in
test), combined **376** (69 in test). `data/camo_human/negatives.txt` lists 1024
stems, but none of them appear in `meta.csv` and none of their images are on
disk (`camo_human` keeps 226 images in total, all of them positives). It is an
*exclusion* list left over from dataset preparation, not a set of negative
samples, and it is unreachable from the evaluation pipeline.

Two consequences: only mhcd and combined can measure the presence gate at all,
and acd1k / cpd1k / camo_human runs must report the gate's rates as *not
measurable* rather than as the vacuous values a one-class confusion matrix
yields (`fp == tn == 0`, so precision is trivially 1.0 and accuracy is just
recall).

The paper (`Template_ PROCS_ICMLDE/PROCS_ICMLDE2024.tex`) has `tab:sota_a`,
`tab:sota_b`, `tab:ablation` and `fig:qualitative` all marked *"placeholders
pending training"*. This work fills them.

## Scope

In scope:

- Per-dataset metric evaluation of one named run at a time
- Failure-case ranking
- Cross-run comparison charts
- Qualitative figures (Image / GT / Prediction)
- Grad-CAM heatmaps at all 4 pyramid levels
- Raw activation maps at all 4 levels across module boundaries
- Module-progression heatmap figure
- Per-dataset sample collage and annotated triplet figures

Explicitly out of scope (decided, not oversight):

- **LaTeX row emission** — not requested.
- **Auto-sweep over all runs** — the user requires one explicitly named run per
  invocation.
- **True cumulative ablation heatmaps** — needs one trained checkpoint per
  ablation variant, which does not exist yet. See "Module progression" below.
- **Any change to `metrics.py`, `train.py`, or the model files.** `metrics.py`
  stays untouched so reported numbers match the tested implementation;
  `return_intermediates` already exposes every tap needed.

## Module layout

New package `src/chd/eval/`, mirroring the existing `src/chd/viz/`:

| File | Responsibility | Depends on |
|---|---|---|
| `src/chd/eval/runs.py` | Resolve `--run <name>` → checkpoint → rebuilt model + recovered config | `chd.models.factory` |
| `src/chd/eval/predict.py` | Inference → probability maps at native GT resolution | `chd.data.dataset` |
| `src/chd/eval/report.py` | Aggregation, CSV/JSON/Markdown writers, failure ranking | `chd.metrics` |
| `src/chd/viz/cam.py` | Grad-CAM over pyramid levels | model intermediates |
| `src/chd/viz/panels.py` | Shared panel rendering (heat, mask-composite, error, bbox) | `chd.viz.colors` |

New scripts, continuing the existing numbering:

| Script | Produces |
|---|---|
| `scripts/08_evaluate.py` | Metrics for one run |
| `scripts/09_visualize_predictions.py` | Qualitative, Grad-CAM, activation, progression figures |
| `scripts/10_compare_runs.py` | Cross-run comparison charts + CSV |
| `scripts/11_dataset_figures.py` | Per-dataset collage + annotated triplets |

Each unit is independently testable: `runs.py` needs only a checkpoint file,
`report.py` only numpy arrays, `cam.py` only a model and a batch.

## Run resolution

```
python scripts/08_evaluate.py --run camo-human-final
```

1. Resolve `<runs-root>/<name>/best.pth` (default runs-root `runs/`).
2. `torch.load(..., weights_only=False)` — written by this repo's own `train.py`,
   never an untrusted download.
3. Rebuild config as an `argparse.Namespace` from `ckpt["args"]`.
4. `build_model(cfg)` — the existing factory, no per-architecture branching.
5. Load **`ckpt["ema"]` if non-empty, else `ckpt["model"]`** — matching what
   `run_validation` uses, so figures and metrics reflect the same weights the
   reported `best_s_alpha` came from.
6. `model.eval()`, move to device.

Overrides: `--runs-root`, `--checkpoint <path>` for an arbitrary file,
`--dataset`, `--img-size`, `--split` (default `test`).

Failure modes to handle explicitly: run directory missing (list available run
names), `best.pth` missing but `last.pth` present (state which was used), and
`ckpt["args"]` absent (require explicit `--dataset`/`--architecture`).

## Evaluation protocol

Two deliberate departures from what `run_validation` does during training:

### Native-resolution scoring

`run_validation` scores at `img_size`. The COD/SOD literature scores against the
ground truth at its **native** resolution. Protocol:

1. Load the item through `CHDDataset` with `AugmentConfig(enabled=False)` at the
   run's own `img_size`.
2. Forward pass → `model.predict_mask(outputs)` → probability map at `img_size`.
3. Bilinearly resize the probability map to the native mask's `(H, W)`.
4. Load the native mask from `data/<dataset>/masks/<stem>.png` directly.
5. Compute `evaluate_all(pred_native, gt_native)`.

Without step 3–4 the numbers are not comparable to the published baselines in
`tab:sota_a` / `tab:sota_b`.

### Negatives scored separately

An empty ground truth sends `s_measure` down its `y == 0` branch, returning
`1 - pred.mean()`, which is a *presence* score, not a segmentation score.
Averaging that with real segmentation scores inflates S_alpha. Therefore:

- **Mask metrics** (MAE, F_beta mean/max/adaptive, S_alpha, E_phi mean/max/adaptive,
  F_bd, IoU, Dice) are aggregated over **positives only**.
- **Presence-gate metrics** (accuracy, precision, recall, F1, AUC at
  `sigmoid(presence_logit) > 0.5`) are computed over **all** images, positives
  and negatives — but only when both classes are present. On an all-positive
  split (acd1k, cpd1k, camo_human) all five rates are `None` and
  `presence_single_class` is `True`; only the raw tp/fp/tn/fn counts are kept.

Per-image rows record both, plus `is_negative` (from `meta.csv`) and
`gt_positive` (measured from the scored mask), so any other aggregation can be
recomputed without re-running the model. A row is treated as a positive only
when both agree, so a mislabelled empty ground truth cannot slip into the mask
means.

### Performance

`evaluate_all` runs two 255-threshold curves per image. At 640x640 that is
roughly 2 s/image, so `combined` (1150 images) would take ~40 min
single-threaded. Mitigation: GPU does inference in a loop; metrics run in a
`concurrent.futures.ProcessPoolExecutor` (`--workers`, default `cpu_count - 2`).
Metrics are pure numpy per image, so this is safe and produces **identical
numbers** — no approximation, no reduced threshold count.

## `08_evaluate.py` outputs

Written to `reports/eval/<run>/`:

```
per_image.csv     one row per image: stem, is_negative, gt_positive,
                  presence_prob, all 11 mask metrics, native H/W
summary.json      positives-only mask means, presence block, recovered run
                  config, checkpoint epoch, best_s_alpha, n test images
metrics.md        human-readable table
failures.csv      positives sorted worst-first by S_alpha, secondary key IoU
preds/<stem>.png  uint8 probability maps, only with --save-preds
```

`preds/` is for manual inspection and later re-analysis. It is deliberately NOT
an input to `09_visualize_predictions.py` — that script recomputes predictions
from the checkpoint, because it also needs the intermediate activations and
gradients that a saved probability map cannot carry. Metrics are always computed
from the in-memory float map, never from the quantized PNG.

## Visualization

Shared conventions: PNG **and** SVG (dpi 180 for the run-scoped figures, matching
`07`; `11_dataset_figures.py` uses 200 to match `06`, since both write into
`reports/datasets/`); palette from
`src/chd/viz/colors.py`; `--num-images`, `--seed`, and
`--pick {random,best,worst}` where `best`/`worst` read `failures.csv`; `--cmap`
(default `jet`, also `inferno`/`magma`/`turbo`).

`--pick best|worst` requires `08_evaluate.py` to have run first. If
`failures.csv` is missing, the script falls back to `random` and prints a
warning naming the missing file — it does not fail, and it does not silently
pretend the selection was score-based.

**Binarization.** Every mask *rendered* in a figure is thresholded at 0.5.
Every mask *scored* in a metric is the continuous probability map. The two never
share a code path, so a rendering choice can never affect a reported number.

Output root: `reports/figures/<run>/`.

### FIG 1 — qualitative

Columns: **Input | Ground Truth | Prediction | Error**.

GT and Prediction follow the COD paper convention from the reference images: the
mask region is rendered as a bright composite over a **darkened** copy of the
input, so target shape and scene context are both legible. This replaces the
green-overlay style currently in `07_visualize_pipeline.py`.

The Error panel colors **false positives in red and false negatives in blue**
over a grayscale input, with true positives left neutral — so it shows *how* a
prediction is wrong (over-segmenting vs missing the target) rather than only
that it is. Both colors come from the Okabe-Ito set already in
`viz/colors.py`, keeping the figure colorblind-safe like the rest of the reports.

### FIG 2 — Grad-CAM per pyramid level

Columns: **Input | GT | CAM-L1 | CAM-L2 | CAM-L3 | CAM-L4**, `jet` colormap
overlaid on the input.

Method — Seg-Grad-CAM (Vinogradova et al., 2020), adapted:

1. Forward with `return_intermediates=True`, gradients **enabled** (not under
   `no_grad`).
2. Scalar target: sum of `mask_logit` over the predicted foreground
   (`predict_mask > 0.5`). `--cam-target {pred,gt,all}`; falls back to the top-k
   highest logits when the prediction is empty, so negatives still render.
3. `torch.autograd.grad(score, intermediates[tap])` — the intermediates are live
   graph tensors, so **no forward/backward hooks are needed at all**.
4. Per level: channel weights = spatial GAP of gradients;
   `cam = relu(sum_c w_c * A_c)`; normalize to [0, 1]; upsample to input size.

Taps default to the four **AER outputs** (the last per-level features before the
decoder, so the most semantically meaningful). `--cam-tap {aer,osneck,sfa,backbone}`.

L4 is at 1/32 resolution (20x20 at `img_size` 640) and will be legitimately
coarse. That is the hierarchy story, not a defect: L1 resolves edges, L4
localizes.

This is the family missing from the current figure. The existing panels show
`mean|activation|` across channels — an unsigned texture response that is not
target-specific, which is exactly why nothing in them resembles the mask.

### FIG 3 — activation maps per level

One file per level, `activations_L1.png` … `activations_L4.png`.
Rows = sample images, columns = module boundaries:
**Backbone | FDM low-freq | FDM high-freq | SFA | OSNeck | AER | Decoder**.

`mean|activation|` across channels, `inferno` — deliberately kept distinct from
FIG 2's `jet` so the two families are never confused when read side by side.

Architecture-aware degradation: `pretrained_unet` exposes only `backbone` and
`decoder_levels`, so absent columns are dropped with a visible note rather than
rendered as zeros.

### FIG 4 — module progression

Columns: **Input | Backbone | +FDM | +SFA | +OSNeck | +AER | Decoder**, one row
per image, `jet`.

This is the single-checkpoint analogue of the reference ablation figure. It shows
where attention moves as the forward pass proceeds through module boundaries.

**It is not a cumulative ablation.** The reference figure compares separately
trained variants ("Before SPG", "Adding SPG"). Producing that requires one
checkpoint per variant, which does not exist yet.

To keep one-run-per-invocation intact, `--run` stays **single and required** (the
primary run, used for every other figure). FIG 4 additionally accepts
`--also-run <name>`, **repeatable and optional**:

- `--run A` alone → module-progression columns from A's forward pass.
- `--run A --also-run B --also-run C` → one Grad-CAM column per run, which is
  the true cross-variant ablation figure.

The figure title states which mode produced it, so the two can never be mistaken
for one another in the paper. Every `--also-run` is resolved by the same
`runs.py` path, so it inherits the same config recovery and EMA preference.

### `10_compare_runs.py`

Reads every `reports/eval/*/summary.json` already on disk and emits grouped bar
charts per metric across datasets and models, plus `comparison.csv`. It runs no
models, so it stays consistent with one-run-per-invocation: it aggregates
completed evaluations only, and states which runs it found.

### `11_dataset_figures.py`

Dataset-level, model-independent, so kept out of the run-scoped scripts.
Reuses `06_visualize_datasets.py`'s `save()` convention and palette.

- `collage_<dataset>.png` — grid of raw test images, no annotations
  (`--rows`, `--cols`, default 2x4).
- `annotated_<dataset>.png` — **Image | Image + bounding box | GT mask**
  triplets. Boxes are derived from the mask via
  `scipy.ndimage.label` + `find_objects` (one box per connected component,
  matching the multi-box case in the reference), with components below a minimum
  area fraction dropped so mask noise does not add spurious boxes.

`06_visualize_datasets.py` already emits `strips/<name>.png`
(Image | Mask | Overlay | Edge | Pose | Resized) so these two figures add the
collage and annotated-triplet styles without duplicating it.

## Testing

`tests/test_eval.py`, following the existing `tests/` style:

- Run resolution from a synthetic checkpoint: config recovery, EMA-preferred
  weight loading, missing-run and missing-`args` error paths.
- Native-resolution protocol: a prediction at `img_size` scored against a
  differently-sized GT lands on the GT's shape, and a perfect prediction scores
  as perfect.
- Positives/negatives partitioning: negatives excluded from mask means, included
  in presence metrics.
- Presence metrics against a hand-computed confusion matrix.
- Grad-CAM on the existing `tiny_test` backbone: correct shape per level, finite,
  non-negative, not all-zero for a positive sample.
- `pretrained_unet` degradation: activation figure drops unavailable columns
  instead of raising.

Existing tests must continue to pass unchanged, since no existing module is
modified.

## Risks

| Risk | Mitigation |
|---|---|
| Cannot test against real weights on this machine | Every script runs end-to-end on the `tiny_test` backbone with a synthetic checkpoint; real-weight paths depend only on `ckpt["args"]`, which is verified present at `train.py:629` |
| `mhcd` has no trained run yet | Scripts are per-run; `mhcd` figures/metrics simply do not exist until it is trained. Nothing fabricates them |
| Grad-CAM needs gradients through a large model at 640x640 | Batch size 1 for CAM figures; only the tapped intermediates are retained |
| `combined` evaluation is slow | Process-pool metrics, `--limit` for smoke tests |

#!/usr/bin/env python3
"""Prepare the human-only subset of CAMO for camouflaged *human* detection.

CAMO ships 1250 camouflage images with pixel-perfect masks but **no category
labels**, so the paper's 223-image "CAMO-Human" benchmark cannot be extracted
directly. This script derives it in three stages:

  auto    score every image for "is the camouflaged object a human?" using CLIP
          over the ground-truth object crop, plus YOLO person detection as a
          corroborating signal. Writes ``auto_scores.csv``.
  review  render a ranked contact sheet so a human confirms the subset.
          Writes ``reports/camo_human_review.html``.
  build   materialise ``data/camo_human/`` from the confirmed list.

Scoring the *ground-truth crop* rather than the whole image is the key trick:
we already know exactly where the camouflaged object is, so CLIP never has to
find it. Running a detector on a camouflaged person mostly fails by
construction, which is why detection is a bonus signal and not the primary one.

Animal images are not discarded — they are recorded as presence-gate negatives.

Examples
--------
    python scripts/01_prepare_camo.py --stage auto \\
        --src dataset/CAMO-V.1.0/CAMO-V.1.0-CVIU2019 --out data/camo_human
    python scripts/01_prepare_camo.py --stage review --out data/camo_human
    python scripts/01_prepare_camo.py --stage build  --out data/camo_human
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chd._compat import zip_strict  # noqa: E402
from chd.data.manifest import (  # noqa: E402
    DatasetWriter,
    binarize,
    load_gray,
    load_rgb,
    mask_bbox,
)

HUMAN_PROMPTS = [
    "a photo of a person",
    "a camouflaged soldier hiding",
    "a human wearing camouflage clothing",
    "a person in a ghillie suit",
    "a hidden human body",
    "a man crouching in the undergrowth",
]

NON_HUMAN_PROMPTS = [
    "a photo of an animal",
    "a camouflaged insect",
    "a camouflaged bird",
    "a camouflaged fish",
    "a camouflaged lizard or reptile",
    "a camouflaged frog",
    "a camouflaged spider",
    "a camouflaged snake",
    "a camouflaged butterfly or moth",
    "a camouflaged crab or sea creature",
    "a camouflaged cat or big cat",
    "a camouflaged deer or antelope",
    "a camouflaged owl",
    "a camouflaged caterpillar",
]

#: P(human) above this is auto-accepted, below ``BAND_LOW`` auto-rejected;
#: everything between is surfaced for review with nothing pre-ticked.
BAND_HIGH = 0.75
BAND_LOW = 0.25

CONTEXT_PAD = 0.25  # fraction of box size added around the GT crop

#: CAMO's filenames are category-ordered. Mean P(human) sits at 0.02-0.12 for
#: every id below 1001 and jumps to ~0.72 at id 1001 (image 1000 scores 0.024,
#: image 1001 scores 0.941) — that block is CAMO's body-painting category.
#: It is a strong prior, not a rule: the block still contains animals (1207 is
#: a lizard, 1231 a flatfish) and CLIP still fires outside it (291 is an owl).
#: So the prior narrows what needs human review; it does not replace it.
HUMAN_BLOCK_START = 1001

#: In-block images at or above this score are pre-ticked for review.
TRUST_IN_BLOCK = 0.60


def stem_id(stem: str) -> int:
    """``camourflage_01027`` -> ``1027``."""
    return int(stem.rsplit("_", 1)[1])


def triage(stem: str, score: float) -> tuple[str, bool]:
    """Classify a row into a review tier and whether to pre-tick it."""
    in_block = stem_id(stem) >= HUMAN_BLOCK_START
    if in_block and score >= TRUST_IN_BLOCK:
        return "likely", True
    if in_block:
        return "check_block", False
    if score >= BAND_LOW:
        return "check_outlier", False
    return "reject", False


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def discover(src: Path) -> list[dict]:
    """Pair every CAMO image with its mask and its official train/test split."""
    samples: list[dict] = []
    missing: list[str] = []
    for official, folder in (("train", "Train"), ("test", "Test")):
        image_dir = src / "Images" / folder
        if not image_dir.is_dir():
            raise SystemExit(f"expected {image_dir} — is --src pointing at CAMO-V.1.0-CVIU2019?")
        for image_path in sorted(image_dir.glob("*.jpg")):
            mask_path = src / "GT" / f"{image_path.stem}.png"
            if not mask_path.exists():
                missing.append(image_path.stem)
                continue
            samples.append(
                {
                    "stem": image_path.stem,
                    "image": image_path,
                    "mask": mask_path,
                    "official": official,
                }
            )
    if missing:
        raise SystemExit(f"{len(missing)} CAMO images have no GT mask, e.g. {missing[:5]}")
    return samples


def crops_for(image: np.ndarray, mask: np.ndarray) -> tuple[Image.Image, Image.Image] | None:
    """Two views of the annotated object: with context, and isolated on gray.

    The context view keeps the scene CLIP needs for "hiding in undergrowth";
    the isolated view removes the background so a human-shaped silhouette is
    not confused with the foliage it is blending into.
    """
    box = mask_bbox(mask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    h, w = mask.shape
    pad_x = int((x2 - x1) * CONTEXT_PAD) + 4
    pad_y = int((y2 - y1) * CONTEXT_PAD) + 4
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    context = Image.fromarray(image[cy1:cy2, cx1:cx2])

    solid = (mask > 0)[y1:y2, x1:x2, None]
    isolated_arr = np.where(solid, image[y1:y2, x1:x2], 128).astype(np.uint8)
    isolated = Image.fromarray(isolated_arr)
    return context, isolated


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter))


# ---------------------------------------------------------------------------
# stage: auto
# ---------------------------------------------------------------------------


def stage_auto(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModel, AutoProcessor

    samples = discover(args.src)
    if args.limit:
        samples = samples[: args.limit]
    print(f"[auto] scoring {len(samples)} CAMO images")

    device = args.device
    clip = AutoModel.from_pretrained(args.clip_model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.clip_model)

    prompts = HUMAN_PROMPTS + NON_HUMAN_PROMPTS
    n_human = len(HUMAN_PROMPTS)
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        text_features = clip.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    detector = None
    if not args.no_detector:
        try:
            from ultralytics import YOLO

            detector = YOLO(args.det_model)
        except Exception as exc:  # noqa: BLE001 - detector is a bonus signal only
            print(f"[auto] person detector unavailable ({exc}); continuing with CLIP alone")

    rows: list[dict] = []
    pending: list[tuple[dict, Image.Image, Image.Image]] = []

    def flush() -> None:
        if not pending:
            return
        images = [view for _, ctx, iso in pending for view in (ctx, iso)]
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt").to(device)
            feats = clip.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = clip.logit_scale.exp() * feats @ text_features.T
            probs = logits.softmax(dim=-1).cpu().numpy()
        # average the two views back into one probability per sample
        paired = probs.reshape(len(pending), 2, -1).mean(axis=1)
        for (row, _, _), prob in zip_strict(pending, paired):
            row["clip_human"] = round(float(prob[:n_human].sum()), 5)
            row["clip_top_nonhuman"] = NON_HUMAN_PROMPTS[int(prob[n_human:].argmax())]
            rows.append(row)
        pending.clear()

    for sample in tqdm(samples, desc="scoring", unit="img"):
        image = load_rgb(sample["image"])
        mask = binarize(load_gray(sample["mask"]))
        views = crops_for(image, mask)
        if views is None:
            print(f"[auto] {sample['stem']}: empty GT mask, skipping")
            continue

        det_conf, det_iou = 0.0, 0.0
        if detector is not None:
            gt_box = mask_bbox(mask)
            result = detector(sample["image"], classes=[0], conf=0.10, verbose=False)[0]
            for box, conf in zip_strict(
                result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                iou = box_iou(tuple(box), gt_box)
                if iou > det_iou:
                    det_conf, det_iou = float(conf), iou

        pending.append(
            (
                {
                    "stem": sample["stem"],
                    "official": sample["official"],
                    "det_conf": round(det_conf, 4),
                    "det_iou": round(det_iou, 4),
                },
                *views,
            )
        )
        if len(pending) >= args.batch_size:
            flush()
    flush()

    for row in rows:
        # A confident detection that lands on the annotated object is strong
        # positive evidence; a miss is weak evidence of anything, so the
        # detector can only ever push the score up.
        detection = row["det_conf"] if row["det_iou"] >= args.det_iou_thresh else 0.0
        row["human_score"] = round(float(max(row["clip_human"], 0.7 * row["clip_human"] + 0.3 * detection)), 5)
        row["band"] = (
            "human"
            if row["human_score"] >= BAND_HIGH
            else "nonhuman"
            if row["human_score"] < BAND_LOW
            else "borderline"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    scores_csv = args.out / "auto_scores.csv"
    with scores_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stem",
                "official",
                "clip_human",
                "clip_top_nonhuman",
                "det_conf",
                "det_iou",
                "human_score",
                "band",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: -r["human_score"]))

    counts = {band: sum(r["band"] == band for r in rows) for band in ("human", "borderline", "nonhuman")}
    print(f"\n[auto] wrote {scores_csv}")
    print(f"[auto] confident human : {counts['human']}")
    print(f"[auto] borderline      : {counts['borderline']}  <- these need your eyes")
    print(f"[auto] confident animal: {counts['nonhuman']}")
    print(f"[auto] paper's CAMO-Human is 223 images, for reference")
    print("\nnext:  python scripts/01_prepare_camo.py --stage review --out", args.out)


# ---------------------------------------------------------------------------
# stage: review
# ---------------------------------------------------------------------------


def thumbnail(image_path: Path, mask_path: Path, width: int = 210) -> str:
    """Base64 JPEG of the image beside its mask overlay."""
    image = load_rgb(image_path)
    mask = binarize(load_gray(mask_path)) > 0
    overlay = image.copy()
    overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)

    pair = Image.fromarray(np.concatenate([image, overlay], axis=1))
    pair.thumbnail((width * 2, width * 2))
    buffer = io.BytesIO()
    pair.convert("RGB").save(buffer, format="JPEG", quality=72)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def stage_review(args: argparse.Namespace) -> None:
    scores_csv = args.out / "auto_scores.csv"
    if not scores_csv.exists():
        raise SystemExit(f"{scores_csv} not found — run --stage auto first")

    with scores_csv.open() as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["score"] = float(row["human_score"])
        row["tier"], row["ticked"] = triage(row["stem"], row["score"])

    # Within each tier, show the most decision-relevant images first: the
    # likely-humans descending (spot a wrong tick fast), the doubtful ones
    # descending (the plausible humans are at the top).
    order = {"likely": 0, "check_block": 1, "check_outlier": 2, "reject": 3}
    rows.sort(key=lambda r: (order[r["tier"]], -r["score"]))

    if args.skip_rejects:
        rows = [r for r in rows if r["tier"] != "reject"]

    tiers = {t: sum(r["tier"] == t for r in rows) for t in order}
    cards: list[str] = []
    for row in tqdm(rows, desc="thumbnails", unit="img"):
        stem = row["stem"]
        folder = "Train" if row["official"] == "train" else "Test"
        data = thumbnail(args.src / "Images" / folder / f"{stem}.jpg", args.src / "GT" / f"{stem}.png")
        checked = " checked" if row["ticked"] else ""
        cards.append(
            f'<label class="card {row["tier"]}" data-tier="{row["tier"]}">'
            f'<input type="checkbox" value="{stem}"{checked}>'
            f'<img src="data:image/jpeg;base64,{data}" loading="lazy">'
            f'<span class="meta">{stem} <i>#{stem_id(stem)}</i><br>score <b>{row["human_score"]}</b> '
            f'&middot; det {row["det_conf"]}<br><i>vs {row["clip_top_nonhuman"]}</i></span></label>'
        )

    html = (
        _REVIEW_TEMPLATE.replace("__CARDS__", "\n".join(cards))
        .replace("__TOTAL__", str(len(rows)))
        .replace("__TIERS__", json.dumps(tiers))
        .replace("__DEST__", str(args.out / "human_subset.json"))
    )
    report = args.review_html
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(html)

    need_eyes = tiers["check_block"] + tiers["check_outlier"]
    print(f"\n[review] wrote {report}  ({report.stat().st_size / 1e6:.1f} MB)")
    print(f"[review]   likely human   {tiers['likely']:>4}  (pre-ticked, in-block & score>={TRUST_IN_BLOCK})")
    print(f"[review]   check: in-block{tiers['check_block']:>4}  (low score inside the body-painting block)")
    print(f"[review]   check: outlier {tiers['check_outlier']:>4}  (high score outside the block)")
    print(f"[review]   near-certain animals {tiers['reject']:>4}  (hidden by default)")
    print(f"[review] -> only ~{need_eyes} images actually need your judgement")
    print("\n[review] open the file, adjust the ticks, click 'Download human_subset.json', then:")
    print(f"           cp ~/Downloads/human_subset.json {args.out / 'human_subset.json'}")
    print(f"           python scripts/01_prepare_camo.py --stage build --out {args.out}")


_REVIEW_TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>CAMO human subset review</title>
<style>
 body{font:14px/1.45 system-ui,sans-serif;margin:0;background:#111;color:#eee}
 header{position:sticky;top:0;background:#1b1b1b;padding:10px 18px;border-bottom:1px solid #333;
        display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:9}
 button{background:#333;color:#fff;border:0;padding:7px 12px;border-radius:6px;cursor:pointer;font-size:13px}
 button.on{background:#2d6cdf}
 button.go{background:#3ddc84;color:#062;font-weight:600}
 .hint{font-size:12px;color:#999;padding:10px 18px 0}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:10px;padding:14px}
 .card{display:block;background:#1c1c1c;border:2px solid #333;border-radius:8px;padding:6px;cursor:pointer}
 .card img{width:100%;display:block;border-radius:4px}
 .card:has(input:checked){border-color:#3ddc84;background:#16281d}
 .meta{font-size:11px;color:#aaa;display:block;margin-top:4px;line-height:1.35}
 .check_block{border-color:#c9a227}
 .check_outlier{border-color:#b05ad9}
 .hidden{display:none}
 kbd{background:#333;border-radius:3px;padding:1px 5px;font-size:11px}
</style>
<header>
  <b>CAMO &rarr; human subset</b>
  <span>selected <b id="count">0</b> / __TOTAL__ &nbsp;<span style="color:#888">(paper: 223)</span></span>
  <button id="b-todo" class="on" onclick="show('todo',this)">needs review</button>
  <button id="b-likely" onclick="show('likely',this)">likely human</button>
  <button id="b-check_block" onclick="show('check_block',this)">in-block, low score</button>
  <button id="b-check_outlier" onclick="show('check_outlier',this)">outside block, high score</button>
  <button id="b-reject" onclick="show('reject',this)">near-certain animals</button>
  <button id="b-all" onclick="show('all',this)">all</button>
  <button id="b-sel" onclick="show('selected',this)">selected only</button>
  <button class="go" onclick="dl()">Download human_subset.json</button>
</header>
<p class="hint">Green border = will be included. Yellow = inside CAMO's body-painting id block but CLIP scored it low.
Purple = outside the block but CLIP thinks it is human. Click a card to toggle it; <kbd>shift</kbd>+click toggles
everything between the last click and this one. Then download and run:
<code>cp ~/Downloads/human_subset.json __DEST__</code></p>
<div id="grid">
__CARDS__
</div>
<script>
const tiers=__TIERS__;
const cards=[...document.querySelectorAll('.card')];
const boxes=cards.map(c=>c.querySelector('input'));
const count=document.getElementById('count');
const upd=()=>count.textContent=boxes.filter(b=>b.checked).length;
let last=null;
cards.forEach((card,i)=>{
  card.addEventListener('click',ev=>{
    if(ev.shiftKey && last!==null){
      ev.preventDefault();
      const want=!boxes[i].checked;
      const [a,b]=[Math.min(last,i),Math.max(last,i)];
      for(let k=a;k<=b;k++) if(!cards[k].classList.contains('hidden')) boxes[k].checked=want;
    }
    last=i; setTimeout(upd,0);
  });
});
upd();
function show(mode,btn){
  document.querySelectorAll('header button').forEach(b=>b.classList.remove('on'));
  if(btn) btn.classList.add('on');
  cards.forEach((c,i)=>{
    const t=c.dataset.tier;
    const vis = mode==='all' ? true
              : mode==='todo' ? (t==='check_block'||t==='check_outlier')
              : mode==='selected' ? boxes[i].checked
              : t===mode;
    c.classList.toggle('hidden',!vis);
  });
}
show('todo',document.getElementById('b-todo'));
function dl(){
  const picked=boxes.filter(b=>b.checked).map(b=>b.value).sort();
  const blob=new Blob([JSON.stringify({humans:picked},null,1)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='human_subset.json'; a.click();
}
</script>
"""


# ---------------------------------------------------------------------------
# stage: build
# ---------------------------------------------------------------------------


def stage_build(args: argparse.Namespace) -> None:
    scores_csv = args.out / "auto_scores.csv"
    if not scores_csv.exists():
        raise SystemExit(f"{scores_csv} not found — run --stage auto first")
    with scores_csv.open() as fh:
        rows = {r["stem"]: r for r in csv.DictReader(fh)}

    # Look where the browser actually puts it, not only where we want it.
    candidates = [
        args.subset,
        args.out / "human_subset.json",
        Path.home() / "Downloads" / "human_subset.json",
        Path.cwd() / "human_subset.json",
    ]
    subset_json = next((p for p in candidates if p and p.exists()), None)

    if subset_json is not None:
        humans = set(json.loads(subset_json.read_text())["humans"])
        print(f"[build] using {len(humans)} confirmed humans from {subset_json}")
        target = args.out / "human_subset.json"
        if subset_json != target:
            target.write_text(json.dumps({"humans": sorted(humans)}, indent=1))
            print(f"[build] copied the confirmed list to {target}")
    elif args.auto_accept:
        # Structural prior AND CLIP must agree, which is what got closest to
        # the paper's 223 (214 images) in the measured scores.
        humans = {
            stem
            for stem, row in rows.items()
            if stem_id(stem) >= HUMAN_BLOCK_START and float(row["human_score"]) >= BAND_LOW
        }
        print(f"[build] --auto-accept: {len(humans)} images (id>={HUMAN_BLOCK_START} and score>={BAND_LOW})")
        print("[build] this is UNREVIEWED — expect a handful of animals to slip through")
    else:
        searched = "\n  ".join(str(p) for p in candidates if p)
        raise SystemExit(
            "No confirmed human_subset.json found. Looked in:\n  " + searched + "\n\n"
            "Either:\n"
            "  1. run --stage review, tick the humans, click Download, then re-run build\n"
            "     (the download is picked up from ~/Downloads automatically), or\n"
            "  2. pass --auto-accept to take the automatic selection unreviewed, or\n"
            "  3. pass --subset /path/to/your.json"
        )

    unknown = humans - rows.keys()
    if unknown:
        raise SystemExit(f"human_subset.json names {len(unknown)} unknown stems, e.g. {sorted(unknown)[:5]}")

    # Official CAMO test images stay in test so our numbers remain a strict
    # subset of the published benchmark; val is carved out of official train.
    train_pool = sorted(s for s in humans if rows[s]["official"] == "train")
    test = sorted(s for s in humans if rows[s]["official"] == "test")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_pool)
    n_val = max(1, round(len(train_pool) * args.val_frac))
    val, train = sorted(train_pool[:n_val]), sorted(train_pool[n_val:])

    assignment = {s: "train" for s in train} | {s: "val" for s in val} | {s: "test" for s in test}

    writer = DatasetWriter(args.out, "camo_human", overwrite=args.overwrite)
    for stem, split in tqdm(sorted(assignment.items()), desc="building", unit="img"):
        if writer.has(stem) and not args.overwrite:
            continue
        folder = "Train" if rows[stem]["official"] == "train" else "Test"
        image_path = args.src / "Images" / folder / f"{stem}.jpg"
        mask = binarize(load_gray(args.src / "GT" / f"{stem}.png"))
        writer.add(
            stem=stem,
            split=split,
            image_src=image_path,
            mask=mask,
            source="camo",
            extra=f"official={rows[stem]['official']};score={rows[stem]['human_score']}",
        )

    summary = writer.finalize()

    negatives = sorted(rows.keys() - humans)
    (args.out / "negatives.txt").write_text("\n".join(negatives) + "\n")

    print(f"\n[build] {summary}")
    print(f"[build] {len(negatives)} non-human CAMO images recorded in negatives.txt")
    print("[build] (used only as presence-gate negatives, never as segmentation targets)")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("auto", "review", "build"))
    parser.add_argument("--src", type=Path, default=Path("dataset/CAMO-V.1.0/CAMO-V.1.0-CVIU2019"))
    parser.add_argument("--out", type=Path, default=Path("data/camo_human"))
    parser.add_argument("--review-html", type=Path, default=Path("reports/camo_human_review.html"))
    parser.add_argument("--subset", type=Path, help="explicit path to a confirmed human_subset.json")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--det-model", default="yolo11x.pt")
    parser.add_argument("--det-iou-thresh", type=float, default=0.30)
    parser.add_argument("--no-detector", action="store_true", help="CLIP only, skip YOLO")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-frac", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="process only the first N images (dry run)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--auto-accept", action="store_true", help="build without manual review")
    parser.add_argument("--skip-rejects", action="store_true",
                        help="omit near-certain animals from the review page (smaller HTML)")
    args = parser.parse_args()

    {"auto": stage_auto, "review": stage_review, "build": stage_build}[args.stage](args)


if __name__ == "__main__":
    main()

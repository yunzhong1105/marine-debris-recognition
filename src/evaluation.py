"""Score a submission.csv with the COCO API, matching how the leaderboard grades.

Reads the *submission file itself* rather than any internal representation, so a
coordinate-convention bug (centre vs top-left, normalised vs absolute) shows up
here instead of on the leaderboard.

    python src/evaluation.py --pred runs/yolo26m_1024_fold0/submission_fold0val.csv \
                             --split data/splits/fold0_val.txt

Ultralytics prints its own mAP50 during training; that number is close to but
not identical with the COCO API's, so it is fine for ranking experiments against
each other and unreliable for judging distance to the 0.6 qualifying threshold.
This script is the one to trust for that.

mAP is averaged over all 34 classes with equal weight, so `aluminum_packaging`
(13 boxes in the entire dataset) moves the score exactly as much as
`plastic_bottle` (5,641). The per-class table below is sorted worst-first
because that is where the points are.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ["image_filename", "label_id", "x", "y", "w", "h", "confidence"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", type=Path, required=True, help="submission.csv to score")
    p.add_argument("--gt", type=Path, default=ROOT / "data/train_dataset/train_label.json")
    p.add_argument(
        "--split",
        type=Path,
        default=None,
        help="image list the predictions cover, e.g. data/splits/fold0_val.txt. "
        "Without it the evaluated set is inferred from the CSV, which silently "
        "drops images the model returned nothing for and flatters the score.",
    )
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--per-class", action="store_true", default=True)
    return p.parse_args()


def read_predictions(path: Path) -> list[dict]:
    """Parse submission.csv, tolerating a present or absent header row."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")

    start = 1 if [c.strip().lower() for c in rows[0]] == HEADER else 0
    out = []
    for lineno, row in enumerate(rows[start:], start=start + 1):
        if not row:
            continue
        if len(row) != 7:
            raise SystemExit(f"{path}:{lineno} expected 7 columns, got {len(row)}: {row}")
        fname, label, x, y, w, h, conf = row
        out.append(
            {
                "filename": fname.strip(),
                "category_id": int(label),
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(conf),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in gt["categories"]}

    # pycocotools sorts and hashes image ids internally; the dataset's ids are
    # hex strings, so remap everything to ints for predictable behaviour.
    id_of_file = {img["filename"]: img["id"] for img in gt["images"]}
    int_of_id = {img["id"]: i for i, img in enumerate(gt["images"])}

    preds = read_predictions(args.pred)
    unknown = {p["filename"] for p in preds if p["filename"] not in id_of_file}
    if unknown:
        raise SystemExit(
            f"{len(unknown)} filenames in {args.pred} are absent from {args.gt.name}, "
            f"e.g. {sorted(unknown)[:3]}"
        )

    if args.split:
        stems = [Path(ln.strip()).name for ln in args.split.read_text().splitlines() if ln.strip()]
        eval_ids = {int_of_id[id_of_file[s]] for s in stems}
        scope = f"{args.split.name} ({len(eval_ids)} images)"
    else:
        eval_ids = {int_of_id[id_of_file[p["filename"]]] for p in preds}
        scope = f"inferred from CSV ({len(eval_ids)} images)"
        print(
            "WARNING: --split not given. Images the model produced no boxes for are\n"
            "         excluded from scoring, which inflates the result. Pass the fold's\n"
            "         val list to score the same set the leaderboard would.\n"
        )

    # Restrict ground truth to the evaluated images.
    gt_small = {
        "images": [{**im, "id": int_of_id[im["id"]]} for im in gt["images"] if int_of_id[im["id"]] in eval_ids],
        "categories": gt["categories"],
        "annotations": [
            {**a, "id": i, "image_id": int_of_id[a["image_id"]]}
            for i, a in enumerate(a2 for a2 in gt["annotations"] if int_of_id[a2["image_id"]] in eval_ids)
        ],
    }

    detections = [
        {
            "image_id": int_of_id[id_of_file[p["filename"]]],
            "category_id": p["category_id"],
            "bbox": p["bbox"],
            "score": p["score"],
        }
        for p in preds
        if int_of_id[id_of_file[p["filename"]]] in eval_ids
    ]
    if not detections:
        raise SystemExit("no predictions fall inside the evaluated image set")

    coco_gt = COCO()
    coco_gt.dataset = gt_small
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections)

        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.imgIds = sorted(eval_ids)
        # summarize() indexes maxDets[0], [1] and [2]; keep three entries.
        ev.params.maxDets = [1, 10, args.max_det]
        ev.evaluate()
        ev.accumulate()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()

    iou50 = int(np.argmin(np.abs(ev.params.iouThrs - 0.5)))
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    cat_ids = coco_gt.getCatIds()

    gt_counts = Counter(a["category_id"] for a in gt_small["annotations"])
    pred_counts = Counter(d["category_id"] for d in detections)

    per_class = []
    for k, cid in enumerate(cat_ids):
        p = precision[iou50, :, k, 0, 2]
        p = p[p > -1]
        per_class.append((cid, float(np.mean(p)) if p.size else float("nan")))

    valid = [ap for _, ap in per_class if ap == ap]
    map50 = float(np.mean(valid)) if valid else float("nan")

    print(f"predictions : {args.pred}")
    print(f"ground truth: {args.gt.name}")
    print(f"scope       : {scope}")
    print(f"boxes       : {len(detections)} predicted vs {len(gt_small['annotations'])} ground truth")
    print(f"             {len(detections) / max(len(eval_ids), 1):.1f} predicted boxes/image\n")

    if args.per_class:
        print(f"{'class':<34}{'gt':>6}{'pred':>8}{'AP@0.5':>9}")
        print("-" * 57)
        for cid, ap in sorted(per_class, key=lambda t: (t[1] != t[1], t[1])):
            shown = "  n/a" if ap != ap else f"{ap:>9.4f}"
            print(f"{names[cid]:<34}{gt_counts[cid]:>6}{pred_counts[cid]:>8}{shown}")
        print("-" * 57)

    print(f"\n{'mAP@0.5 (COCO, competition metric)':<38}{map50:.4f}")
    print(f"{'mAP@0.5:0.95':<38}{ev.stats[0]:.4f}")

    gap = 0.6 - map50
    if gap > 0:
        print(f"\n{gap:.4f} short of the 0.6 qualifying threshold.")
        worst = [(names[c], a) for c, a in sorted(per_class, key=lambda t: t[1]) if a == a][:5]
        headroom = sum((0.6 - a) for _, a in worst if a < 0.6) / len(cat_ids)
        print(f"Lifting the five weakest classes to 0.6 alone would add ~{headroom:.4f}:")
        for n, a in worst:
            print(f"    {n:<32}{a:.4f}")
    else:
        print("\nAbove the 0.6 qualifying threshold.")


if __name__ == "__main__":
    main()

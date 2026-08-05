"""Run inference and write a competition submission.csv.

Output columns are the ones the organisers specify:

    image_filename, label_id, x, y, w, h, confidence

`x, y, w, h` are absolute pixels with `x, y` at the box's **top-left** corner --
the same convention as the supplied COCO `train_label.json`. Ultralytics returns
centre-based xywh, so this converts. If the official sample submission turns out
to disagree, that is the one thing to change here.

    python src/predict.py --weights runs/yolo26m_1024_fold0/weights/best.pt \
                          --source data/splits/fold0_val.txt \
                          --out runs/yolo26m_1024_fold0/submission_fold0val.csv

The defaults `conf=0.001` and `max_det=300` are deliberate and should not be
raised. AP is the area under the precision-recall curve, built by walking
predictions from most to least confident; discarding low-confidence boxes
truncates the curve before it reaches high recall and permanently removes that
area. A junk box near the tail costs almost nothing, a missed object costs
recall forever. Ultralytics' own default of conf=0.25 is for looking at
pictures, not for scoring.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True, help="e.g. runs/<name>/weights/best.pt")
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="image directory, or a .txt file listing one image path per line",
    )
    p.add_argument("--out", type=Path, default=ROOT / "submission.csv")
    p.add_argument("--imgsz", type=int, default=1024, help="must match training")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.001, help="do not raise this for scoring")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU, ignored when end2end")
    p.add_argument("--device", default="0")
    p.add_argument("--augment", action="store_true", help="test-time augmentation, slower")
    p.add_argument(
        "--end2end",
        choices=("true", "false"),
        default=None,
        help="YOLO26 one-to-one head (no NMS). Unset uses the model default. "
        "Worth A/B testing: the NMS path usually yields a longer low-confidence "
        "tail, which mAP rewards.",
    )
    p.add_argument("--no-header", action="store_true", help="omit the CSV header row")
    return p.parse_args()


def collect_sources(source: Path) -> list[str]:
    """Resolve a directory or a .txt list into image paths."""
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise SystemExit(f"no images found under {source}")
        return [str(p) for p in files]

    if source.suffix.lower() == ".txt":
        lines = [ln.strip() for ln in source.read_text().splitlines() if ln.strip()]
        if not lines:
            raise SystemExit(f"{source} is empty")
        missing = [ln for ln in lines[:50] if not Path(ln).exists()]
        if missing:
            raise SystemExit(
                f"{source} lists paths that do not exist, e.g. {missing[0]}\n"
                "Regenerate with: python src/make_split.py --folds 5"
            )
        return lines

    raise SystemExit(f"--source must be a directory or a .txt list, got {source}")


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    sources = collect_sources(args.source)
    print(f"{len(sources)} images | weights {args.weights} | imgsz {args.imgsz} "
          f"| conf {args.conf} | max_det {args.max_det}")

    kwargs = dict(
        source=sources,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        max_det=args.max_det,
        iou=args.iou,
        device=args.device,
        augment=args.augment,
        stream=True,       # 15k results will not fit in RAM at once
        verbose=False,
    )
    if args.end2end is not None:
        kwargs["end2end"] = args.end2end == "true"

    model = YOLO(args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not args.no_header:
            writer.writerow(["image_filename", "label_id", "x", "y", "w", "h", "confidence"])

        for result in model.predict(**kwargs):
            name = Path(result.path).name
            n_images += 1
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            # xyxy is already rescaled to the original image, unlike xywhn.
            for (x1, y1, x2, y2), cls, conf in zip(
                boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()
            ):
                writer.writerow(
                    [
                        name,
                        int(cls),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                        round(conf, 5),
                    ]
                )
                n_boxes += 1
            if n_images % 500 == 0:
                print(f"  {n_images}/{len(sources)} images, {n_boxes} boxes")

    print(f"\n{n_images} images, {n_boxes} boxes -> {args.out}")
    print(f"mean {n_boxes / max(n_images, 1):.1f} boxes/image")
    if n_boxes / max(n_images, 1) < 10:
        print(
            "WARNING: that is few boxes per image for conf=0.001. Check that --conf "
            "was not overridden; a sparse file usually means mAP is being left behind."
        )


if __name__ == "__main__":
    main()

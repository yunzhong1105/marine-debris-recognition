"""Convert the marine-debris COCO annotations into YOLO-format label files.

The competition ships a single COCO JSON (`train_label.json`). Ultralytics
resolves labels by swapping `/images/` for `/labels/` in each image path, so we
write one `.txt` per image into a sibling `labels/` directory and leave the
images untouched. The train/val split is applied later via file lists, which
avoids duplicating 15k JPEGs on disk.

Category ids are already 0-based and contiguous (0-33), so they map straight to
YOLO class ids.

    python src/coco_to_yolo.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from paths import DATA, PROJECT as ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco", type=Path, default=DATA / "train_dataset/train_label.json")
    p.add_argument("--images", type=Path, default=DATA / "train_dataset/images")
    p.add_argument("--labels", type=Path, default=DATA / "train_dataset/labels")
    p.add_argument("--names-out", type=Path, default=ROOT / "configs/classes.yaml")
    p.add_argument(
        "--min-size",
        type=float,
        default=1.0,
        help="drop boxes whose clipped width or height is below this many pixels",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    coco = json.loads(args.coco.read_text(encoding="utf-8"))

    images = {img["id"]: img for img in coco["images"]}
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    assert [c["id"] for c in categories] == list(range(len(categories))), (
        "category ids are not contiguous 0..N-1; an explicit id->index remap is needed"
    )

    by_image: dict[str, list] = defaultdict(list)
    stats = Counter()

    for ann in coco["annotations"]:
        stats["annotations"] += 1
        img = images.get(ann["image_id"])
        if img is None:
            stats["dropped_orphan"] += 1
            continue
        if ann.get("iscrowd"):
            stats["dropped_iscrowd"] += 1
            continue

        iw, ih = img["width"], img["height"]
        x, y, w, h = ann["bbox"]

        # Clip to the image; a handful of COCO boxes overhang the frame.
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(iw), x + w), min(float(ih), y + h)
        if (x, y, x + w, y + h) != (x1, y1, x2, y2):
            stats["clipped"] += 1

        cw, ch = x2 - x1, y2 - y1
        if cw < args.min_size or ch < args.min_size:
            stats["dropped_degenerate"] += 1
            continue

        by_image[ann["image_id"]].append(
            (ann["category_id"], (x1 + cw / 2) / iw, (y1 + ch / 2) / ih, cw / iw, ch / ih)
        )
        stats["kept"] += 1

    args.labels.mkdir(parents=True, exist_ok=True)
    for img_id, img in images.items():
        rows = by_image.get(img_id, [])
        stem = Path(img["filename"]).stem
        lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in rows]
        (args.labels / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        if not rows:
            stats["images_without_labels"] += 1

    missing = [i["filename"] for i in coco["images"] if not (args.images / i["filename"]).exists()]

    args.names_out.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {c['id']}: {c['name']}" for c in categories)
    args.names_out.write_text(f"nc: {len(categories)}\nnames:\n{names}\n", encoding="utf-8")

    print(f"images            : {len(images)}")
    print(f"label files written: {len(images)} -> {args.labels}")
    print(f"annotations        : {stats['annotations']}")
    print(f"  kept             : {stats['kept']}")
    print(f"  clipped to frame : {stats['clipped']}")
    print(f"  dropped degenerate: {stats['dropped_degenerate']}")
    print(f"  dropped iscrowd  : {stats['dropped_iscrowd']}")
    print(f"  dropped orphan   : {stats['dropped_orphan']}")
    print(f"images w/o labels  : {stats['images_without_labels']}")
    print(f"missing image files: {len(missing)}{' ' + str(missing[:5]) if missing else ''}")
    print(f"class names        : {args.names_out}")


if __name__ == "__main__":
    main()

"""Train a YOLO26 detector on the marine-debris dataset.

Thin wrapper around Ultralytics that pins the settings this competition cares
about, so runs stay comparable across experiments.

    python src/train.py                          # baseline, fold 0
    python src/train.py --fraction 0.5 --epochs 1 # smoke test (half an epoch)
    python src/train.py --model yolo26n.pt --imgsz 640 --name nano  # NetScore entry

Note the Windows `if __name__ == "__main__"` guard: dataloader workers are
spawned, not forked, so training must not run at import time.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Must precede the torch import. Training peaks at ~6.1GB of the 7.4GB free, and
# Ultralytics validates at batch*2, so the val pass allocates into a heap the
# training pass has already fragmented. Expandable segments let the allocator
# grow existing blocks instead of demanding fresh contiguous ones.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolo26m.pt", help="pretrained weights or a .yaml")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--data", type=Path, default=None, help="overrides --fold")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=4, help="6.1/7.4 GB VRAM at 1024; 6 will OOM")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="8 exhausts the 16GB host RAM: mosaic builds a 2048px buffer per sample",
    )
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--fraction", type=float, default=1.0, help="fraction of train set per epoch")
    p.add_argument("--name", default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_trainer():
    """DetectionTrainer that validates at the training batch size, not twice it.

    Upstream builds the val loader at `batch_size * 2` (trainer.py:293). At
    imgsz=1024 that OOMs on 8GB: training alone peaks at 6.1GB, and the val pass
    then asks for a batch of 8. The failure is fragmentation-dependent, so it
    survives a short smoke test and only bites hours into a real run -- exactly
    the sort of thing that kills an overnight training.

    Subclassed rather than patched into vendor/ so `git pull` on ultralytics
    stays clean.
    """
    from ultralytics.models.yolo.detect import DetectionTrainer

    class ValBatchCappedTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
            if mode == "val":
                batch_size = max(1, batch_size // 2)
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

    return ValBatchCappedTrainer


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    data = args.data or ROOT / f"configs/fold{args.fold}.yaml"
    name = args.name or f"{Path(args.model).stem}_{args.imgsz}_fold{args.fold}"

    model = YOLO(args.model)
    model.train(
        trainer=build_trainer(),
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        fraction=args.fraction,
        device=0,
        amp=True,          # required to fit 1024px on 8GB
        cache=False,       # 15k images at 1024 will not fit in 16GB RAM
        project=str(ROOT / "runs"),
        name=name,
        exist_ok=True,
        resume=args.resume,
        seed=0,
        plots=True,
    )


if __name__ == "__main__":
    main()

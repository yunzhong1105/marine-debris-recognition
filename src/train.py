"""Train a YOLO26 detector on the marine-debris dataset.

Thin wrapper around Ultralytics that pins the settings this competition cares
about, so runs stay comparable across experiments.

    python src/train.py                          # baseline, fold 0
    python src/train.py --fraction 0.5 --epochs 1 # smoke test (half an epoch)
    python src/train.py --model yolo26n.pt --imgsz 640 --name nano  # NetScore entry

Only the arguments that get changed often have their own flag. Everything else
Ultralytics accepts goes through --set, so the script never has to grow a
hundred pass-through options:

    python src/train.py --set copy_paste=0.3 cls=1.0 lr0=0.001

Note the Windows `if __name__ == "__main__"` guard: dataloader workers are
spawned, not forked, so training must not run at import time.
"""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

# Must precede the torch import. Windows does not support expandable_segments
# (torch warns and ignores it), so this buys nothing here -- it is kept only so
# the same script helps on a Linux box. The fragmentation problem it would have
# solved is handled by ValBatchCappedTrainer below.
# PYTORCH_CUDA_ALLOC_CONF is the deprecated spelling as of torch 2.9.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
    p.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="any other Ultralytics train argument, passed straight through: "
        "--set lr0=0.001 copy_paste=0.3 cls=1.0. Applied last, so it overrides "
        "everything above including the settings this script pins. Ultralytics "
        "rejects unknown keys, so a typo fails loudly rather than being ignored.",
    )
    return p.parse_args()


def parse_overrides(pairs: list[str]) -> dict:
    """Turn `KEY=VALUE` strings into typed kwargs.

    literal_eval gives numbers, booleans and lists their real types; anything it
    cannot parse stays a string, which is what bare words like `optimizer=AdamW`
    need.
    """
    out = {}
    for item in pairs:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def relax_shm_pressure() -> None:
    """Share worker tensors through files when /dev/shm is too small.

    Docker defaults /dev/shm to 64 MB. DataLoader workers hand tensors to the
    parent through shared memory, so on a container at imgsz 1280 they die with
    "Bus error / No space left on device" long before the GPU is the limit --
    and the traceback blames the dataloader, not the container.

    The file_system strategy routes the same traffic through regular files. It
    is slightly slower but needs no container privileges, which matters on a
    managed Jupyter host where --shm-size is not ours to set.
    """
    import torch.multiprocessing as mp

    shm = Path("/dev/shm")
    if not shm.exists():
        return
    stat = os.statvfs(shm)
    gib = stat.f_blocks * stat.f_frsize / 2**30
    if gib < 2:
        mp.set_sharing_strategy("file_system")
        print(f"/dev/shm is only {gib:.2f} GiB -- switching to file_system tensor sharing")


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

    relax_shm_pressure()

    data = args.data or ROOT / f"configs/fold{args.fold}.yaml"
    name = args.name or f"{Path(args.model).stem}_{args.imgsz}_fold{args.fold}"

    weights, resume = args.model, False
    if args.resume:
        last = ROOT / "runs" / name / "weights/last.pt"
        if not last.exists():
            raise SystemExit(f"nothing to resume from: {last} does not exist")
        # Hand upstream the explicit checkpoint. `resume=True` makes it call
        # get_latest_run(), which picks whichever run directory was written to
        # most recently -- not necessarily the one --name refers to.
        weights, resume = last, str(last)
        print(f"resuming from {last}")

    settings = dict(
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
        resume=resume,
        seed=0,
        plots=True,
    )

    overrides = parse_overrides(args.set)
    if overrides:
        print(f"overrides: {overrides}")
        settings.update(overrides)

    model = YOLO(weights)
    model.train(trainer=build_trainer(), **settings)


if __name__ == "__main__":
    main()

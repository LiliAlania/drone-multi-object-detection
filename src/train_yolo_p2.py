"""Train YOLOv8s with an extra P2 (stride 4) head on the shared tiles (Ultralytics).

    python -m src.train_yolo_p2 [--debug] [--resume]
Weights: <runs_dir>/yolo_p2/weights/{best,last}.pt; per-epoch metrics in results.csv there.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.pipeline_common import DEFAULT_CONFIG, load_config, seed_everything, tiles_dir
from src.models.yolo_baseline import MAX_DETECTIONS


def train(cfg: dict, resume: bool = False) -> Path:
    from ultralytics import YOLO

    y, project = cfg["yolo"], cfg["paths"]["runs_dir"].resolve()
    last = project / "yolo_p2" / "weights" / "last.pt"
    tick = time.perf_counter()
    if resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume given but no checkpoint at {last}")
        model = YOLO(str(last))
        model.train(resume=True)  # restores every setting from the checkpoint
    else:
        data = tiles_dir(cfg) / "data.yaml"
        if not data.is_file():
            raise FileNotFoundError(f"No tiles at {data}; run src.prepare_tiles first")
        model = YOLO(y["model"]).load(y["weights"])  # P2 layers start fresh, the rest from COCO
        model.train(
            data=str(data.resolve()), epochs=y["epochs"], imgsz=y["imgsz"], batch=cfg["paths"]["yolo_batch"],
            amp=True, scale=y["scale"], mosaic=y["mosaic"], copy_paste=y["copy_paste"], patience=y["patience"],
            seed=cfg["seed"], deterministic=True, workers=cfg["paths"]["workers"], max_det=MAX_DETECTIONS,
            project=str(project), name="yolo_p2", exist_ok=True,
        )
    minutes = (time.perf_counter() - tick) / 60
    epochs = model.trainer.epochs - model.trainer.start_epoch
    print(f"YOLO: {minutes:.1f} min for {epochs} epoch(s) = {minutes / max(epochs, 1):.1f} min/epoch "
          f"(incl. validation)")
    return Path(model.trainer.best)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue from <runs_dir>/yolo_p2/weights/last.pt")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    seed_everything(cfg["seed"])
    print(f"Best weights: {train(cfg, args.resume)}")


if __name__ == "__main__":
    main()

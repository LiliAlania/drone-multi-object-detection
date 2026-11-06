"""Ultralytics YOLO baseline: train on VisDrone and write a Kaggle submission CSV.

Usage (Colab, from the repository root):
    python -m src.yolo_baseline train --data-root /content/visdrone-subset --epochs 30
    python -m src.yolo_baseline predict --weights runs/detect/baseline/weights/best.pt \
        --test-root /content/data --out submission.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from src.data_utils import CLASS_NAMES

MAX_DETECTIONS = 500  # Kaggle scores at most 500 detections per image


def image_dirs(data_root: str | Path) -> tuple[Path, Path]:
    """Return absolute (train, val) image folders for the full or subset layout."""
    root = Path(data_root).resolve()
    if (root / "train" / "images").is_dir():  # competition layout: train/images, val/images
        train, val = root / "train" / "images", root / "val" / "images"
    else:  # src.make_subset layout: images/train, images/val
        train, val = root / "images" / "train", root / "images" / "val"
    if not train.is_dir() or not val.is_dir():
        raise FileNotFoundError(f"No train/val image folders under {root}")
    return train, val


def write_dataset_yaml(data_root: str | Path, output: str | Path) -> Path:
    """Write an Ultralytics data.yaml with absolute paths for full or subset layouts."""
    root = Path(data_root).resolve()
    train, val = image_dirs(root)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    output = Path(output)
    output.write_text(
        f"path: {root.as_posix()}\ntrain: {train.as_posix()}\nval: {val.as_posix()}\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )
    return output


def format_predictions(
    classes: Sequence[float], scores: Sequence[float], boxes_xyxy: Sequence[Sequence[float]]
) -> str:
    """Convert pixel xyxy boxes to the Kaggle `class score x y w h` string (top-left x, y)."""
    return " ".join(
        f"{int(cls)} {score:.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}"
        for cls, score, (x1, y1, x2, y2) in zip(classes, scores, boxes_xyxy)
    )


def train(
    data_root: Path, model: str, epochs: int, imgsz: int, batch: int, name: str, fraction: float
) -> Path:
    from ultralytics import YOLO

    data_yaml = write_dataset_yaml(data_root, Path(f"{name}_data.yaml"))
    yolo = YOLO(model)
    yolo.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch, name=name,
        fraction=fraction, max_det=MAX_DETECTIONS, exist_ok=True,
    )
    best = Path(yolo.trainer.best)
    print(f"Best weights: {best}")
    return best


def predict(
    weights: Path, test_root: Path, out: Path, imgsz: int, conf: float, batch: int
) -> Path:
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    with (test_root / "test_images.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "PredictionString"])
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            results = yolo.predict(
                [str(test_root / row["file_name"]) for row in chunk],
                imgsz=imgsz, conf=conf, max_det=MAX_DETECTIONS, verbose=False,
            )
            for row, result in zip(chunk, results):
                boxes = result.boxes
                writer.writerow([
                    row["image_id"],
                    format_predictions(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()),
                ])
    print(f"Wrote {len(rows)} rows to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data-root", type=Path, required=True)
    train_parser.add_argument("--model", default="yolo11s.pt")
    train_parser.add_argument("--epochs", type=int, default=30)
    # 1280 instead of 640: most objects are 10-30 px on 1920x1080 frames
    train_parser.add_argument("--imgsz", type=int, default=1280)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--fraction", type=float, default=1.0)
    train_parser.add_argument("--name", default="baseline")

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--weights", type=Path, required=True)
    predict_parser.add_argument("--test-root", type=Path, required=True,
                                help="Folder containing test_images.csv and test/images/")
    predict_parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    predict_parser.add_argument("--imgsz", type=int, default=1280)
    # low threshold: mAP rewards recall, low-score boxes barely hurt precision
    predict_parser.add_argument("--conf", type=float, default=0.01)
    predict_parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()
    if args.command == "train":
        train(args.data_root, args.model, args.epochs, args.imgsz, args.batch, args.name, args.fraction)
    else:
        predict(args.weights, args.test_root, args.out, args.imgsz, args.conf, args.batch)


if __name__ == "__main__":
    main()

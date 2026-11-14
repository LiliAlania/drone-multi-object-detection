"""Ultralytics YOLO baseline: train on VisDrone and write a Kaggle submission CSV.

Recent YOLOs are anchor-free: each grid cell centre predicts distances to the 4 box sides
(see src.yolo_assign for how cells are assigned to objects).

Usage (Colab, from the repository root):
    python -m src.models.yolo_baseline train --data-root /content/visdrone-subset --epochs 30
    # extra stride-4 head: --model configs/yolo11s-p2.yaml --pretrained yolo11s.pt
    python -m src.models.yolo_baseline evaluate --weights runs/detect/baseline/weights/last.pt         --split-root /content/data/val --step 5 --imgsz 1280
    python -m src.models.yolo_baseline predict --weights runs/detect/baseline/weights/best.pt \
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
    data_root: Path, model: str, epochs: int, imgsz: int, batch: float, name: str, fraction: float,
    project: Path = Path("runs/detect"), resume: bool = False, pretrained: str | None = None, workers: int = 8,
    wandb_project: str | None = None,
) -> Path:
    from ultralytics import YOLO

    if resume:
        last = project / name / "weights" / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(f"--resume given but no checkpoint at {last}")
        yolo = YOLO(str(last))
        add_wandb_callbacks(yolo, wandb_project, project / name, resume, {})
        yolo.train(resume=True)  # restores every setting, incl. epochs and data, from the checkpoint
    else:
        project.mkdir(parents=True, exist_ok=True)
        data_yaml = write_dataset_yaml(data_root, project / f"{name}_data.yaml")
        yolo = YOLO(model)
        if pretrained:  # a .yaml model (e.g. the P2 head) starts random; copy every layer whose shape matches
            yolo.load(pretrained)
        add_wandb_callbacks(yolo, wandb_project, project / name, resume, {
            "detector": Path(model).stem, "pretrained": pretrained or model, "train_images": str(data_root),
            "heads": "P2-P5" if "p2" in Path(model).stem else "P3-P5",
        })
        yolo.train(
            data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=int(batch) if batch >= 1 else batch,
            project=str(project.resolve()), name=name, fraction=fraction, workers=workers,
            max_det=MAX_DETECTIONS, exist_ok=True,
        )
    best = Path(yolo.trainer.best)
    print(f"Best weights: {best}")
    return best


def add_wandb_callbacks(yolo, wandb_project: str | None, run_dir: Path, resume: bool, config: dict) -> None:
    """Log Ultralytics training to W&B in the run that `evaluate` resumes (id in <run_dir>/wandb_id.txt):
    config, per-epoch train losses / val metrics / lr, and the plots Ultralytics writes at the end.

    Ultralytics' own W&B callback is turned off: it names the W&B project after the --project path.
    """
    if not wandb_project:
        return
    import wandb
    from ultralytics.utils import SETTINGS

    from src.models.faster_rcnn import wandb_run

    SETTINGS.update({"wandb": False})
    run = wandb_run(wandb_project, run_dir, config=config, new=not resume)

    def on_train_start(trainer):
        run.config.update({k: str(v) if isinstance(v, Path) else v for k, v in vars(trainer.args).items()},
                          allow_val_change=True)

    logged_epochs = set()

    def on_fit_epoch_end(trainer):
        if trainer.epoch in logged_epochs:  # Ultralytics fires it again after the final validation of best.pt
            return
        logged_epochs.add(trainer.epoch)
        run.log({**trainer.label_loss_items(trainer.tloss, prefix="train"), **trainer.metrics, **trainer.lr,
                 "epoch": trainer.epoch + 1})

    def on_train_end(trainer):
        plots = ["results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png",
                 "labels.jpg", "train_batch0.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg",
                 "val_batch1_labels.jpg", "val_batch1_pred.jpg"]
        run.log({f"plots/{Path(p).stem}": wandb.Image(str(trainer.save_dir / p))
                 for p in plots if (trainer.save_dir / p).is_file()})
        run.finish()

    yolo.add_callback("on_train_start", on_train_start)
    yolo.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    yolo.add_callback("on_train_end", on_train_end)


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


def evaluate(weights: Path, split_root: Path, step: int, imgsz: int, conf: float,
             wandb_project: str | None = None, wandb_images: int = 16) -> dict[str, float]:
    """Full-frame COCO mAP (by object size) at `imgsz`, comparable with `src.models.faster_rcnn evaluate`.

    With wandb_project, logs the metrics and `wandb_images` evenly spaced val frames with GT and predicted
    boxes into the training run of <weights>/../.. (Ultralytics layout: <project>/<name>/weights/last.pt).
    """
    # these modules import this one, so not at the top
    from src.models.faster_rcnn import log_val_to_wandb, wandb_run
    from src.models.tiled_yolo import OVERLAP, evaluate as coco_evaluate, load_model, sliced_predict

    metrics = coco_evaluate(weights, split_root, step, tile=0, full_imgsz=imgsz, conf=conf)
    run = wandb_run(wandb_project, weights.parent.parent)
    if run:
        import torch
        from pycocotools.coco import COCO

        from src.data_utils import image_path_for

        coco = COCO(str(split_root / "annotations_coco.json"))
        image_ids = sorted(coco.getImgIds())[::step]
        yolo, outputs = load_model(weights, "yolo"), {}
        for image_id in image_ids[:: max(len(image_ids) // wandb_images, 1)][:wandb_images]:
            boxes, scores, classes = sliced_predict(
                yolo, image_path_for(split_root, coco.imgs[image_id]["file_name"]), 0, OVERLAP, imgsz, conf)
            outputs[image_id] = {"boxes": torch.tensor(boxes).reshape(-1, 4), "scores": torch.tensor(scores),
                                 "labels": torch.tensor(classes) + 1}
        log_val_to_wandb(run, coco, split_root, outputs, metrics | {"imgsz": imgsz}, step)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data-root", type=Path, required=True)
    train_parser.add_argument("--model", default="yolo11s.pt")
    train_parser.add_argument("--epochs", type=int, default=30)
    # 1280 instead of 640: most objects are 10-30 px on 1920x1080 frames
    train_parser.add_argument("--imgsz", type=int, default=1280)
    train_parser.add_argument("--batch", type=float, default=8, help="Images per batch, or 0-1 = share of GPU memory")
    train_parser.add_argument("--fraction", type=float, default=1.0)
    train_parser.add_argument("--name", default="baseline")
    train_parser.add_argument("--project", type=Path, default=Path("runs/detect"), help="Put on Drive to survive disconnects")
    train_parser.add_argument("--pretrained", help="Weights copied into a .yaml --model, e.g. yolo11s.pt")
    train_parser.add_argument("--workers", type=int, default=8)
    train_parser.add_argument("--resume", action="store_true", help="Continue from <project>/<name>/weights/last.pt")
    train_parser.add_argument("--wandb-project", help="Log config, loss/metric curves and plots to this W&B project")

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--weights", type=Path, required=True)
    evaluate_parser.add_argument("--split-root", type=Path, required=True, help="Labeled split with annotations_coco.json")
    evaluate_parser.add_argument("--step", type=int, default=1, help="Evaluate every k-th frame")
    evaluate_parser.add_argument("--imgsz", type=int, default=1280, help="Inference size: use the training size")
    evaluate_parser.add_argument("--conf", type=float, default=0.01)
    evaluate_parser.add_argument("--wandb-project",
                                 help="Log metrics and val frames with boxes into the training run's W&B run")

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
        train(args.data_root, args.model, args.epochs, args.imgsz, args.batch, args.name, args.fraction,
              args.project, args.resume, args.pretrained, args.workers, args.wandb_project)
    elif args.command == "evaluate":
        evaluate(args.weights, args.split_root, args.step, args.imgsz, args.conf, args.wandb_project)
    else:
        predict(args.weights, args.test_root, args.out, args.imgsz, args.conf, args.batch)


if __name__ == "__main__":
    main()

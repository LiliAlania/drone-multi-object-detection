"""Two-stage anchor-based detector: torchvision Faster R-CNN (ResNet-50 FPN v2) on VisDrone.

Reads the same YOLO-format labels as src.yolo_baseline (full or subset layout) and writes
the same Kaggle submission CSV.

Usage (Colab, from the repository root):
    python -m src.faster_rcnn train --data-root /content/visdrone-subset --epochs 12
    python -m src.faster_rcnn evaluate --weights runs/faster_rcnn/last.pt --split-root /content/data/val
    python -m src.faster_rcnn predict --weights runs/faster_rcnn/last.pt \
        --test-root /content/data --out submission_frcnn.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Sequence

from src.data_utils import CLASS_NAMES, _label_path, _read_yolo_rows
from src.yolo_baseline import MAX_DETECTIONS, format_predictions, image_dirs

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# One anchor size per FPN level (P2..P6). Smaller than torchvision's (32..512) default
# because most VisDrone objects are 10-30 px; 3 aspect ratios keep the pretrained RPN head shape.
ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
MIN_BOX_SIDE = 1.0  # torchvision rejects boxes with zero width/height


def yolo_rows_to_target(
    rows: Sequence[Sequence[float]], width: int, height: int
) -> tuple[list[list[float]], list[int]]:
    """YOLO `cls cx cy w h` (normalized) -> pixel xyxy boxes and 1-based labels (0 = background)."""
    boxes, labels = [], []
    for cls, cx, cy, w, h in rows:
        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
        x1, y1, x2, y2 = max(x1, 0.0), max(y1, 0.0), min(x2, width), min(y2, height)
        if x2 - x1 >= MIN_BOX_SIDE and y2 - y1 >= MIN_BOX_SIDE:
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)
    return boxes, labels


def build_model(min_size: int, max_size: int, pretrained: bool = True):
    import torchvision
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        min_size=min_size,
        max_size=max_size,
        # dense scenes: keep more proposals and detections than the COCO defaults
        rpn_pre_nms_top_n_train=4000, rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=2000,
        box_batch_size_per_image=512,
        box_detections_per_img=MAX_DETECTIONS,
        box_score_thresh=0.01,  # low threshold: mAP rewards recall
    )
    # the v2 builder hard-codes its anchor generator, so swap ours in afterwards;
    # same 3 anchors per location, so the pretrained RPN head still fits
    model.rpn.anchor_generator = AnchorGenerator(ANCHOR_SIZES, ASPECT_RATIOS)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    return model


class YoloDetectionDataset:
    """Images under `image_dir` with YOLO labels in the sibling `labels` tree."""

    def __init__(self, image_dir: Path, train: bool):
        self.paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.paths:
            raise FileNotFoundError(f"No images under {image_dir}")
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch
        from torchvision.io import ImageReadMode, read_image

        path = self.paths[index]
        image = read_image(str(path), ImageReadMode.RGB).float() / 255
        height, width = image.shape[1:]
        boxes, labels = yolo_rows_to_target(_read_yolo_rows(_label_path(path, "labels"), True), width, height)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.train and torch.rand(1).item() < 0.5:  # horizontal flip
            image = image.flip(-1)
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        return image, {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.int64)}


def _collate(batch):
    return tuple(zip(*batch))


def train(
    data_root: Path, epochs: int, batch: int, lr: float, min_size: int, max_size: int,
    out_dir: Path, workers: int, resume: bool = False,
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir, _ = image_dirs(data_root)
    loader = DataLoader(
        YoloDetectionDataset(train_dir, train=True), batch_size=batch, shuffle=True,
        num_workers=workers, collate_fn=_collate,
    )
    model = build_model(min_size, max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    # step decay at 2/3 and 8/9 of training (the "1x schedule" shape)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 2 / 3), int(epochs * 8 / 9)], gamma=0.1
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "last.pt"
    start_epoch = 0
    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"--resume given but no checkpoint at {checkpoint}")
        # weights_only=False: the scheduler state holds a Counter; these are our own files
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        saved = (state["min_size"], state["max_size"], state.get("epochs"))
        if saved != (min_size, max_size, epochs):
            # the LR schedule is tied to --epochs, so a different value would silently break it
            raise ValueError(f"Checkpoint was started with min/max size {saved[0]}/{saved[1]} and "
                             f"--epochs {saved[2]}; pass the same values to resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        print(f"Resuming after epoch {start_epoch} from {checkpoint}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        for step, (images, targets) in enumerate(loader, 1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}; lower --lr")
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch + 1}/{epochs} step {step}/{len(loader)} loss {running / step:.4f}")
        scheduler.step()
        state = {
            "model": model.state_dict(), "min_size": min_size, "max_size": max_size,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch + 1, "epochs": epochs,
        }
        # write then rename, so a disconnect during saving can't corrupt last.pt
        torch.save(state, checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)
        print(f"epoch {epoch + 1} done, mean loss {running / max(len(loader), 1):.4f}, saved {checkpoint}")
    return checkpoint


def load_trained(weights: Path):
    """Load a checkpoint saved by `train`; returns (model in eval mode, device)."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weights, map_location=device, weights_only=False)  # own checkpoint; holds scheduler state
    model = build_model(state["min_size"], state["max_size"], pretrained=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), device


def _read_images(paths: Sequence[Path], device):
    from torchvision.io import ImageReadMode, read_image

    return [read_image(str(p), ImageReadMode.RGB).float().div(255).to(device) for p in paths]


def evaluate(weights: Path, split_root: Path, batch: int, step: int) -> dict[str, float]:
    """COCO mAP on a labeled split (e.g. /content/data/val) using its annotations_coco.json.

    Ignored regions (iscrowd=1) are matched but not counted, as in the COCO protocol.
    """
    import torch
    from pycocotools.coco import COCO

    from src.data_utils import image_path_for

    coco = COCO(str(split_root / "annotations_coco.json"))
    image_ids = sorted(coco.getImgIds())[::step]
    model, device = load_trained(weights)
    results = []
    model_seconds = 0.0
    with torch.inference_mode():
        for start in range(0, len(image_ids), batch):
            chunk = image_ids[start : start + batch]
            paths = [image_path_for(split_root, coco.imgs[i]["file_name"]) for i in chunk]
            images = _read_images(paths, device)
            # time only the model (incl. resize + NMS), not disk reads; skip first batch (warm-up)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if start:
                model_seconds += time.perf_counter() - tick
            for image_id, out in zip(chunk, outputs):
                for (x1, y1, x2, y2), score, label in zip(
                    out["boxes"].tolist(), out["scores"].tolist(), out["labels"].tolist()
                ):
                    # model labels 1..8 are the COCO category ids 1..8 of this dataset
                    results.append({"image_id": image_id, "category_id": label,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
            print(f"{min(start + batch, len(image_ids))}/{len(image_ids)} images", end="\r")
    metrics = coco_map(coco, results, image_ids)
    timed_images = len(image_ids) - min(batch, len(image_ids))
    if timed_images:
        metrics["ms_per_image"] = 1000 * model_seconds / timed_images
        print(f"Inference: {metrics['ms_per_image']:.1f} ms/image "
              f"({1000 / metrics['ms_per_image']:.1f} FPS, batch {batch}, {device.type})")
    return metrics


def coco_map(coco, results: list[dict], image_ids: Sequence[int]) -> dict[str, float]:
    """Run COCOeval on detection dicts and print overall and per-class AP."""
    from pycocotools.cocoeval import COCOeval

    if not results:
        raise RuntimeError("Model produced no detections")
    evaluator = COCOeval(coco, coco.loadRes(results), "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 100, MAX_DETECTIONS]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    # pycocotools' first summary line is fixed at maxDets=100; recompute at 500 (the Kaggle cap).
    # precision shape: [iou, recall, class, area(all, small, medium, large), maxDets]
    precision = evaluator.eval["precision"][..., -1]

    def ap(values) -> float:
        values = values[values > -1]
        return float(values.mean()) if values.size else float("nan")

    metrics = {
        "mAP": ap(precision[:, :, :, 0]),
        "AP50": ap(precision[0, :, :, 0]),
        "AP_small": ap(precision[:, :, :, 1]),
    }
    print(f"mAP@[.5:.95] {metrics['mAP']:.4f} | AP50 {metrics['AP50']:.4f} | "
          f"AP_small {metrics['AP_small']:.4f}  (maxDets={MAX_DETECTIONS})")
    for index, name in enumerate(CLASS_NAMES):
        print(f"  {name:10s} AP {ap(precision[:, :, index, 0]):.3f}")
    return metrics


def predict(weights: Path, test_root: Path, out: Path, batch: int) -> Path:
    import torch

    model, device = load_trained(weights)

    with (test_root / "test_images.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file, torch.inference_mode():
        writer = csv.writer(file)
        writer.writerow(["image_id", "PredictionString"])
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            images = _read_images([test_root / row["file_name"] for row in chunk], device)
            for row, result in zip(chunk, model(images)):
                writer.writerow([
                    row["image_id"],
                    format_predictions(
                        (result["labels"] - 1).tolist(), result["scores"].tolist(), result["boxes"].tolist()
                    ),
                ])
    print(f"Wrote {len(rows)} rows to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data-root", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=0.01)
    # native-ish resolution: downscaling kills the 10-30 px objects
    train_parser.add_argument("--min-size", type=int, default=1080)
    train_parser.add_argument("--max-size", type=int, default=1920)
    train_parser.add_argument("--out-dir", type=Path, default=Path("runs/faster_rcnn"))
    train_parser.add_argument("--workers", type=int, default=2)
    train_parser.add_argument("--resume", action="store_true",
                              help="Continue from <out-dir>/last.pt (same --epochs and sizes)")

    eval_parser = commands.add_parser("evaluate", help="COCO mAP on a labeled split")
    eval_parser.add_argument("--weights", type=Path, required=True)
    eval_parser.add_argument("--split-root", type=Path, required=True,
                             help="Folder with annotations_coco.json, e.g. /content/data/val")
    eval_parser.add_argument("--batch", type=int, default=4)
    eval_parser.add_argument("--step", type=int, default=1,
                             help="Evaluate every k-th frame (faster; consecutive frames are near-duplicates)")

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--weights", type=Path, required=True)
    predict_parser.add_argument("--test-root", type=Path, required=True,
                                help="Folder containing test_images.csv and test/images/")
    predict_parser.add_argument("--out", type=Path, default=Path("submission_frcnn.csv"))
    predict_parser.add_argument("--batch", type=int, default=4)

    args = parser.parse_args()
    if args.command == "train":
        train(args.data_root, args.epochs, args.batch, args.lr, args.min_size, args.max_size,
              args.out_dir, args.workers, args.resume)
    elif args.command == "evaluate":
        evaluate(args.weights, args.split_root, args.batch, args.step)
    else:
        predict(args.weights, args.test_root, args.out, args.batch)


if __name__ == "__main__":
    main()

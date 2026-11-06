"""Two-stage anchor-based detector: torchvision Faster R-CNN (ResNet-50 FPN v2) on VisDrone.

Reads the same YOLO-format labels as src.yolo_baseline (full or subset layout) and writes
the same Kaggle submission CSV.

Usage (Colab, from the repository root):
    python -m src.faster_rcnn train --data-root /content/visdrone-subset --epochs 12
    python -m src.faster_rcnn predict --weights runs/faster_rcnn/last.pt \
        --test-root /content/data --out submission_frcnn.csv
"""

from __future__ import annotations

import argparse
import csv
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
        rpn_anchor_generator=AnchorGenerator(ANCHOR_SIZES, ASPECT_RATIOS),
        min_size=min_size,
        max_size=max_size,
        # dense scenes: keep more proposals and detections than the COCO defaults
        rpn_pre_nms_top_n_train=4000, rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=2000,
        box_batch_size_per_image=512,
        box_detections_per_img=MAX_DETECTIONS,
        box_score_thresh=0.01,  # low threshold: mAP rewards recall
    )
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
    out_dir: Path, workers: int,
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

    for epoch in range(epochs):
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
        torch.save({"model": model.state_dict(), "min_size": min_size, "max_size": max_size}, checkpoint)
        print(f"epoch {epoch + 1} done, mean loss {running / max(len(loader), 1):.4f}, saved {checkpoint}")
    return checkpoint


def predict(weights: Path, test_root: Path, out: Path, batch: int) -> Path:
    import torch
    from torchvision.io import ImageReadMode, read_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weights, map_location=device)
    model = build_model(state["min_size"], state["max_size"], pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    with (test_root / "test_images.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file, torch.inference_mode():
        writer = csv.writer(file)
        writer.writerow(["image_id", "PredictionString"])
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            images = [
                read_image(str(test_root / row["file_name"]), ImageReadMode.RGB).float().div(255).to(device)
                for row in chunk
            ]
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

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--weights", type=Path, required=True)
    predict_parser.add_argument("--test-root", type=Path, required=True,
                                help="Folder containing test_images.csv and test/images/")
    predict_parser.add_argument("--out", type=Path, default=Path("submission_frcnn.csv"))
    predict_parser.add_argument("--batch", type=int, default=4)

    args = parser.parse_args()
    if args.command == "train":
        train(args.data_root, args.epochs, args.batch, args.lr, args.min_size, args.max_size,
              args.out_dir, args.workers)
    else:
        predict(args.weights, args.test_root, args.out, args.batch)


if __name__ == "__main__":
    main()

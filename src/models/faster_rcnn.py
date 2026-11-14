"""Two-stage anchor-based detector: torchvision Faster R-CNN (ResNet-50 FPN v2) on VisDrone.

Reads the same YOLO-format labels as src.models.yolo_baseline (full or subset layout) and writes
the same Kaggle submission CSV.

Usage (Colab, from the repository root):
    python -m src.models.faster_rcnn train --data-root /content/visdrone-subset --epochs 12
    python -m src.models.faster_rcnn evaluate --weights runs/faster_rcnn/last.pt --split-root /content/data/val
    python -m src.models.faster_rcnn predict --weights runs/faster_rcnn/last.pt \
        --test-root /content/data --out submission_frcnn.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from src.data_utils import CLASS_NAMES, _label_path, _read_yolo_rows
from src.models.yolo_baseline import MAX_DETECTIONS, format_predictions, image_dirs

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# One anchor size per FPN level (P2..P6). Smaller than torchvision's (32..512) default
# because most VisDrone objects are 10-30 px; 3 aspect ratios keep the pretrained RPN head shape.
ANCHOR_SIZES = ((8,), (16,), (32,), (64,), (128,))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)
MIN_BOX_SIDE = 1.0  # torchvision rejects boxes with zero width/height
TWO_STAGE_OPTIONS = ("rpn_proposals", "roi_batch", "roi_pos_fraction", "box_loss")  # build_model kwargs


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


GIOU_LOSS_WEIGHT = 10.0  # MMDetection's faster-rcnn GIoU config; GIoU (<=2) is much smaller than smooth L1 on deltas


def _giou_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """torchvision's fastrcnn_loss with a GIoU box term instead of smooth L1."""
    import torch
    import torch.nn.functional as F
    from torchvision.models.detection._utils import BoxCoder
    from torchvision.ops import generalized_box_iou_loss

    labels = torch.cat(labels)
    regression_targets = torch.cat(regression_targets)
    classification_loss = F.cross_entropy(class_logits, labels)
    positive = torch.where(labels > 0)[0]
    deltas = box_regression.reshape(labels.numel(), -1, 4)[positive, labels[positive]]
    # IoU is invariant to per-axis scaling and shifting, so decoding both against a unit box
    # gives the same GIoU as decoding against the real proposals
    coder = BoxCoder((10.0, 10.0, 5.0, 5.0))  # RoIHeads' default weights
    unit = deltas.new_tensor([[0.0, 0.0, 1.0, 1.0]]).float().expand(len(positive), 4)
    pred = coder.decode_single(deltas.float(), unit)
    target = coder.decode_single(regression_targets[positive].float(), unit)
    box_loss = GIOU_LOSS_WEIGHT * generalized_box_iou_loss(pred, target, reduction="sum") / labels.numel()
    return classification_loss, box_loss


def build_model(
    min_size: int, max_size: int, pretrained: bool = True, rpn_proposals: int = 2000,
    roi_batch: int = 512, roi_pos_fraction: float = 0.25, box_loss: str = "smooth_l1",
):
    import torchvision
    from torchvision.models.detection import roi_heads
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if not 0 < roi_pos_fraction <= 1:
        raise ValueError(f"roi_pos_fraction must be in (0, 1], got {roi_pos_fraction}")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        min_size=min_size,
        max_size=max_size,
        # dense scenes: keep more proposals and detections than the COCO defaults (1000)
        rpn_pre_nms_top_n_train=2 * rpn_proposals, rpn_post_nms_top_n_train=rpn_proposals,
        rpn_pre_nms_top_n_test=rpn_proposals, rpn_post_nms_top_n_test=rpn_proposals,
        box_batch_size_per_image=roi_batch,  # RoIs sampled per image for the second-stage loss
        box_positive_fraction=roi_pos_fraction,  # share of those RoIs that are foreground
        box_detections_per_img=MAX_DETECTIONS,
        box_score_thresh=0.01,  # low threshold: mAP rewards recall
    )
    # shortcut: RoIHeads calls the module-level fastrcnn_loss, so this swap is process-wide;
    # fine for one model per run, subclass RoIHeads if two models must train side by side
    if not hasattr(roi_heads, "_smooth_l1_fastrcnn_loss"):
        roi_heads._smooth_l1_fastrcnn_loss = roi_heads.fastrcnn_loss
    roi_heads.fastrcnn_loss = {"smooth_l1": roi_heads._smooth_l1_fastrcnn_loss,
                               "giou": _giou_fastrcnn_loss}[box_loss]
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


def wandb_run(project: str | None, run_dir: Path, config: dict | None = None, new: bool = False):
    """One W&B run per run dir: train writes its id to <run_dir>/wandb_id.txt and evaluate/explain resume it,
    so curves, metrics and plots of one model sit together. Returns None when project is None."""
    if not project:
        return None
    import wandb

    id_file = run_dir / "wandb_id.txt"
    run_id = None if new or not id_file.is_file() else id_file.read_text().strip()
    kwargs = dict(project=project, name=run_dir.name, id=run_id, resume="allow", config=config)
    try:
        run = wandb.init(**kwargs)
    except Exception as error:  # bad key / no access to the entity must not cost hours of training
        run = None
        if os.environ.pop("WANDB_ENTITY", None):  # e.g. an organization name: retry with the key's default entity
            print(f"W&B init failed for the configured entity ({error}); retrying with the default entity")
            try:
                run = wandb.init(**kwargs)
            except Exception as retry_error:
                error = retry_error
        if run is None:
            print(f"W&B online init failed ({error}); logging offline, upload later with `wandb sync`")
            run = wandb.init(mode="offline", **kwargs)
    print(f"W&B run: {run.url or run.dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    id_file.write_text(run.id)
    return run


def wandb_boxes(boxes, labels, scores=None) -> dict:
    """xyxy pixel boxes + 1-based labels -> the `boxes` entry of wandb.Image."""
    items = []
    for i, ((x1, y1, x2, y2), label) in enumerate(zip(boxes, labels)):
        item = {"position": {"minX": x1, "maxX": x2, "minY": y1, "maxY": y2}, "domain": "pixel",
                "class_id": int(label), "box_caption": CLASS_NAMES[int(label) - 1]}
        if scores is not None:
            item["scores"] = {"score": float(scores[i])}
            item["box_caption"] += f" {scores[i]:.2f}"
        items.append(item)
    return {"box_data": items, "class_labels": {i + 1: name for i, name in enumerate(CLASS_NAMES)}}


def train(
    data_root: Path, epochs: int, batch: int, lr: float, min_size: int, max_size: int,
    out_dir: Path, workers: int, resume: bool = False, build_fn=build_model,
    lr_schedule: str = "step", model_cfg: dict | None = None, wandb_project: str | None = None,
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir, _ = image_dirs(data_root)
    dataset = YoloDetectionDataset(train_dir, train=True)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers, collate_fn=_collate)
    model_cfg = model_cfg or {}
    run = wandb_run(wandb_project, out_dir, new=not resume, config={
        # module file name, not __module__: that is "__main__" under `python -m`
        "detector": Path(sys.modules[build_fn.__module__].__file__).stem, "data_root": str(data_root), "train_images": len(dataset),
        "epochs": epochs, "batch": batch, "lr": lr, "lr_schedule": lr_schedule, "min_size": min_size,
        "max_size": max_size, "optimizer": "SGD momentum 0.9 wd 1e-4", "amp": device.type == "cuda",
        "augment": "hflip 0.5", "anchor_sizes": ANCHOR_SIZES, **model_cfg,
    })
    model = build_fn(min_size, max_size, **model_cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
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
        # checkpoints from before these options existed used the current defaults
        saved = (state["min_size"], state["max_size"], state.get("epochs"),
                 state.get("lr_schedule", "step"), state.get("model_cfg", model_cfg))
        if saved != (min_size, max_size, epochs, lr_schedule, model_cfg):
            # the LR schedule is tied to --epochs, so a different value would silently break it
            raise ValueError(f"Checkpoint was started with min/max size {saved[0]}/{saved[1]}, "
                             f"--epochs {saved[2]}, --lr-schedule {saved[3]}, model options {saved[4]}; "
                             f"pass the same values to resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        print(f"Resuming after epoch {start_epoch} from {checkpoint}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running = 0.0
        parts: dict[str, float] = {}  # per-loss sums since the last log: RPN (objectness, rpn_box_reg) vs RoI head
        for step, (images, targets) in enumerate(loader, 1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                losses = model(images, targets)
                loss = sum(losses.values())
            for key, value in losses.items():
                parts[key] = parts.get(key, 0.0) + value.item()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}; lower --lr")
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch + 1}/{epochs} step {step}/{len(loader)} loss {running / step:.4f}")
                if run:
                    run.log({**{f"train/{k}": v / 50 for k, v in parts.items()},
                             "train/loss": sum(parts.values()) / 50, "train/lr": optimizer.param_groups[0]["lr"],
                             "epoch": epoch + step / len(loader)}, step=epoch * len(loader) + step)
                parts = {}
        if run:
            run.log({"train/epoch_loss": running / max(len(loader), 1), "epoch": epoch + 1},
                    step=(epoch + 1) * len(loader))
        scheduler.step()
        state = {
            "model": model.state_dict(), "min_size": min_size, "max_size": max_size,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch + 1, "epochs": epochs,
            "lr_schedule": lr_schedule, "model_cfg": model_cfg,
        }
        # write then rename, so a disconnect during saving can't corrupt last.pt
        torch.save(state, checkpoint.with_suffix(".tmp"))
        checkpoint.with_suffix(".tmp").replace(checkpoint)
        print(f"epoch {epoch + 1} done, mean loss {running / max(len(loader), 1):.4f}, saved {checkpoint}")
    if run:
        run.finish()
    return checkpoint


def load_trained(weights: Path, build_fn=build_model):
    """Load a checkpoint saved by `train`; returns (model in eval mode, device)."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(weights, map_location=device, weights_only=False)  # own checkpoint; holds scheduler state
    model = build_fn(state["min_size"], state["max_size"], pretrained=False, **state.get("model_cfg", {}))
    model.load_state_dict(state["model"])
    return model.to(device).eval(), device


def _read_images(paths: Sequence[Path], device):
    from torchvision.io import ImageReadMode, read_image

    return [read_image(str(p), ImageReadMode.RGB).float().div(255).to(device) for p in paths]


def evaluate(weights: Path, split_root: Path, batch: int, step: int, build_fn=build_model,
             wandb_project: str | None = None, wandb_images: int = 16) -> dict[str, float]:
    """COCO mAP on a labeled split (e.g. /content/data/val) using its annotations_coco.json.

    Ignored regions (iscrowd=1) are matched but not counted, as in the COCO protocol.
    With wandb_project, logs the metrics and `wandb_images` evenly spaced frames with GT and predicted boxes
    to the training run of these weights.
    """
    import torch
    from pycocotools.coco import COCO

    from src.data_utils import image_path_for

    coco = COCO(str(split_root / "annotations_coco.json"))
    image_ids = sorted(coco.getImgIds())[::step]
    model, device = load_trained(weights, build_fn)
    shown = set(image_ids[:: max(len(image_ids) // wandb_images, 1)][:wandb_images]) if wandb_project else set()
    shown_outputs = {}
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
                if image_id in shown:
                    shown_outputs[image_id] = {k: v.cpu() for k, v in out.items()}
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
    run = wandb_run(wandb_project, weights.parent)
    if run:
        log_val_to_wandb(run, coco, split_root, shown_outputs, metrics, step)
    return metrics


def log_val_to_wandb(run, coco, split_root: Path, outputs: dict, metrics: dict, step: int) -> None:
    """Log val metrics and a table of frames with GT and predicted boxes, then finish the run.

    outputs: {image_id: {"boxes": xyxy tensor, "labels": 1-based tensor, "scores": tensor}}.
    """
    import wandb

    from src.data_utils import image_path_for

    table = wandb.Table(columns=["image_id", "file", "gt_objects", "detections@0.3", "image"])
    for image_id, out in outputs.items():
        anns = [a for a in coco.imgToAnns[image_id] if not a.get("iscrowd")]
        gt = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)]
        # boxes below 0.1 only clutter the view; the W&B score slider filters the rest
        keep = out["scores"] >= 0.1
        image = wandb.Image(str(image_path_for(split_root, coco.imgs[image_id]["file_name"])), boxes={
            "predictions": wandb_boxes(out["boxes"][keep].tolist(), out["labels"][keep].tolist(),
                                       out["scores"][keep].tolist()),
            "ground_truth": wandb_boxes(gt, [a["category_id"] for a in anns]),
        })
        table.add_data(image_id, coco.imgs[image_id]["file_name"], len(anns),
                       int((out["scores"] >= 0.3).sum()), image)
    run.log({"val/predictions": table, **{f"val/{k}": v for k, v in metrics.items()}})
    run.summary.update({f"val/{k}": v for k, v in metrics.items()} | {"val/frame_step": step})
    run.finish()


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
        "AP_medium": ap(precision[:, :, :, 2]),
        "AP_large": ap(precision[:, :, :, 3]),
    }
    print(f"mAP@[.5:.95] {metrics['mAP']:.4f} | AP50 {metrics['AP50']:.4f} | "
          f"AP_small {metrics['AP_small']:.4f} | AP_medium {metrics['AP_medium']:.4f} | "
          f"AP_large {metrics['AP_large']:.4f}  (maxDets={MAX_DETECTIONS})")
    for index, name in enumerate(CLASS_NAMES):
        metrics[f"AP_{name}"] = ap(precision[:, :, index, 0])
        print(f"  {name:10s} AP {metrics[f'AP_{name}']:.3f}")
    return metrics


def predict(weights: Path, test_root: Path, out: Path, batch: int, build_fn=build_model, step: int = 1) -> Path:
    import torch

    model, device = load_trained(weights, build_fn)

    with (test_root / "test_images.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))[::step]  # step > 1 only for val analysis: Kaggle needs every test frame
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


def main(build_fn=build_model, run_name: str = "faster_rcnn", submission: str = "submission_frcnn.csv",
         lr: float = 0.01, description: str = __doc__) -> None:
    """CLI for torchvision detectors: only the model builder (build_fn) differs."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--data-root", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=lr)
    # native-ish resolution: downscaling kills the 10-30 px objects
    train_parser.add_argument("--min-size", type=int, default=1080)
    train_parser.add_argument("--max-size", type=int, default=1920)
    train_parser.add_argument("--out-dir", type=Path, default=Path("runs") / run_name)
    train_parser.add_argument("--workers", type=int, default=2)
    train_parser.add_argument("--resume", action="store_true",
                              help="Continue from <out-dir>/last.pt (same --epochs, sizes and options)")
    train_parser.add_argument("--lr-schedule", choices=["step", "cosine"], default="step",
                              help="step: x0.1 at 2/3 and 8/9 of epochs; cosine: smooth decay to 0")
    train_parser.add_argument("--wandb-project", help="Log config and loss curves to this W&B project")
    if build_fn is build_model:  # two-stage knobs; the one-stage builders sharing this CLI don't take them
        two_stage = train_parser.add_argument_group("two-stage options")
        two_stage.add_argument("--rpn-proposals", type=int, default=2000,
                               help="Proposals kept after RPN NMS and passed to the RoI head")
        two_stage.add_argument("--roi-batch", type=int, default=512,
                               help="RoIs sampled per image for the second-stage loss")
        two_stage.add_argument("--roi-pos-fraction", type=float, default=0.25,
                               help="Foreground share of the sampled RoIs")
        two_stage.add_argument("--box-loss", choices=["smooth_l1", "giou"], default="smooth_l1",
                               help="Second-stage box regression loss")

    eval_parser = commands.add_parser("evaluate", help="COCO mAP on a labeled split")
    eval_parser.add_argument("--weights", type=Path, required=True)
    eval_parser.add_argument("--split-root", type=Path, required=True,
                             help="Folder with annotations_coco.json, e.g. /content/data/val")
    eval_parser.add_argument("--batch", type=int, default=4)
    eval_parser.add_argument("--step", type=int, default=1,
                             help="Evaluate every k-th frame (faster; consecutive frames are near-duplicates)")
    eval_parser.add_argument("--wandb-project",
                             help="Log metrics and val frames with boxes to the W&B run of these weights")

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--weights", type=Path, required=True)
    predict_parser.add_argument("--test-root", type=Path, required=True,
                                help="Folder containing test_images.csv and test/images/")
    predict_parser.add_argument("--out", type=Path, default=Path(submission))
    predict_parser.add_argument("--batch", type=int, default=4)
    predict_parser.add_argument("--step", type=int, default=1,
                                help="Predict every k-th listed frame (val analysis only; keep 1 for a submission)")

    args = parser.parse_args()
    if args.command == "train":
        train(args.data_root, args.epochs, args.batch, args.lr, args.min_size, args.max_size,
              args.out_dir, args.workers, args.resume, build_fn, args.lr_schedule,
              {k: v for k, v in vars(args).items() if k in TWO_STAGE_OPTIONS}, args.wandb_project)
    elif args.command == "evaluate":
        evaluate(args.weights, args.split_root, args.batch, args.step, build_fn, args.wandb_project)
    else:
        predict(args.weights, args.test_root, args.out, args.batch, build_fn, args.step)


if __name__ == "__main__":
    main()

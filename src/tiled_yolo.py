"""Our pipeline: tiled YOLO training + sliced inference, with every choice taken from the EDA.

EDA finding -> what this pipeline does
- Objects are tiny: at imgsz 640, 69% of boxes have a side < 16 px (22% even at 1280).
  -> train on 640x640 tiles cut from full-resolution frames (no downscaling) and predict on the
     whole frame at its native resolution, i.e. the same object scale as in training. Optional:
     extra overlapping tile passes (--tile 640), merged with NMS.
- 72% of frames contain ignored regions with unlabeled objects inside.
  -> paint ignored regions gray in training tiles so the model is not taught they are background.
- Strong class imbalance (car 505k vs bus 9.7k boxes).
  -> repeat-factor sampling (LVIS): tiles with rare classes are written several times.
- Consecutive frames are near-duplicates (IoU 0.85 at lag 1; step 5 loses only 0.7% of tracks).
  -> build tiles from the step-5 subset (src.make_subset).
- At most 180 objects per frame -> the Kaggle cap of 500 detections is never binding.

Usage (Colab, from the repository root):
    python -m src.tiled_yolo prepare --data-root /content/visdrone-subset --out /content/visdrone-tiles
    python -m src.tiled_yolo train --data-root /content/visdrone-tiles --epochs 20 \
        --project /content/drive/MyDrive/runs/tiled_yolo
    python -m src.tiled_yolo evaluate --weights .../weights/best.pt --split-root /content/data/val --step 5
    python -m src.tiled_yolo predict --weights .../weights/best.pt --test-root /content/data \
        --out submission_tiled.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path
from typing import Sequence

from src.data_utils import _label_path, _read_yolo_rows
from src.faster_rcnn import IMAGE_SUFFIXES, coco_map
from src.yolo_baseline import MAX_DETECTIONS, format_predictions, image_dirs, write_dataset_yaml

TILE = 640
OVERLAP = 0.2
MIN_VISIBLE = 0.5  # keep a box cut by a tile edge only if at least half of it is inside
GRAY = 114  # Ultralytics' letterbox fill colour
ARCHS = {"yolo": "YOLO", "rtdetr": "RTDETR"}  # Ultralytics model classes; RT-DETR needs its own trainer/predictor


def load_model(weights, arch: str):
    import ultralytics

    return getattr(ultralytics, ARCHS[arch])(str(weights))


def tile_windows(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping xyxy windows covering the whole image; the last row/column is aligned to the edge."""

    def starts(size: int) -> list[int]:
        if size <= tile:
            return [0]
        stride = max(1, int(tile * (1 - overlap)))
        return [*range(0, size - tile, stride), size - tile]

    return [(x, y, min(x + tile, width), min(y + tile, height)) for y in starts(height) for x in starts(width)]


def yolo_to_xyxy(rows: Sequence[Sequence[float]], width: int, height: int) -> list[list[float]]:
    """Normalized `[cls] cx cy w h` rows -> pixel `[cls] x1 y1 x2 y2` (class kept if present)."""
    out = []
    for row in rows:
        *cls, cx, cy, w, h = row
        out.append([*cls, (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return out


def crop_targets(
    targets: Sequence[Sequence[float]], ignored: Sequence[Sequence[float]], window: Sequence[int]
) -> list[tuple[int, float, float, float, float]]:
    """Pixel `cls x1 y1 x2 y2` targets -> YOLO rows of one tile.

    Drops boxes mostly outside the tile and boxes whose centre is in an ignored region (painted gray).
    """
    x0, y0, x1, y1 = window
    tw, th = x1 - x0, y1 - y0
    rows = []
    for cls, bx1, by1, bx2, by2 in targets:
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        if any(ix1 <= cx <= ix2 and iy1 <= cy <= iy2 for ix1, iy1, ix2, iy2 in ignored):
            continue
        cx1, cy1, cx2, cy2 = max(bx1, x0), max(by1, y0), min(bx2, x1), min(by2, y1)
        area = (bx2 - bx1) * (by2 - by1)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2 or (cx2 - cx1) * (cy2 - cy1) < MIN_VISIBLE * area:
            continue
        rows.append((int(cls), ((cx1 + cx2) / 2 - x0) / tw, ((cy1 + cy2) / 2 - y0) / th,
                     (cx2 - cx1) / tw, (cy2 - cy1) / th))
    return rows


def repeat_factors(tile_classes: Sequence[set[int]], threshold: float) -> list[float]:
    """LVIS repeat-factor sampling: r(tile) = max over its classes of max(1, sqrt(threshold / freq))."""
    counts: dict[int, int] = {}
    for classes in tile_classes:
        for cls in classes:
            counts[cls] = counts.get(cls, 0) + 1
    class_factor = {c: max(1.0, math.sqrt(threshold / (n / len(tile_classes)))) for c, n in counts.items()}
    return [max((class_factor[c] for c in classes), default=1.0) for classes in tile_classes]


def prepare(data_root: Path, out: Path, tile: int, overlap: float, keep_empty: float,
            rfs_threshold: float, seed: int) -> Path:
    """Cut train/val frames into tiles (YOLO layout images/<split>, labels/<split>)."""
    import cv2

    rng = random.Random(seed)
    for split, image_dir in zip(("train", "val"), image_dirs(data_root)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        paths = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        written: list[tuple[Path, Path, set[int]]] = []
        for index, path in enumerate(paths, 1):
            image = cv2.imread(str(path))
            height, width = image.shape[:2]
            targets = yolo_to_xyxy(_read_yolo_rows(_label_path(path, "labels"), True), width, height)
            ignored = yolo_to_xyxy(_read_yolo_rows(_label_path(path, "ignore"), False), width, height)
            for ix1, iy1, ix2, iy2 in ignored:
                image[max(int(iy1), 0):int(math.ceil(iy2)), max(int(ix1), 0):int(math.ceil(ix2))] = GRAY
            for window in tile_windows(width, height, tile, overlap):
                rows = crop_targets(targets, ignored, window)
                if not rows and rng.random() >= keep_empty:
                    continue
                x0, y0, x1, y1 = window
                stem = f"{path.parent.name}_{path.stem}_{x0}_{y0}"
                image_path = out / "images" / split / f"{stem}.jpg"
                label_path = out / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(image_path), image[y0:y1, x0:x1])
                label_path.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                                              for c, cx, cy, w, h in rows), encoding="utf-8")
                written.append((image_path, label_path, {r[0] for r in rows}))
            if index % 200 == 0:
                print(f"{split}: {index}/{len(paths)} frames, {len(written)} tiles", end="\r")
        extra = 0
        if split == "train" and rfs_threshold > 0:  # val stays unbiased
            for (image_path, label_path, _), factor in zip(written, repeat_factors([w[2] for w in written], rfs_threshold)):
                copies = int(factor) - 1 + (rng.random() < factor - int(factor))
                for k in range(copies):
                    suffix = f"_rep{k + 1}"
                    image_path.with_stem(image_path.stem + suffix).write_bytes(image_path.read_bytes())
                    label_path.with_stem(label_path.stem + suffix).write_bytes(label_path.read_bytes())
                extra += copies
        print(f"{split}: {len(paths)} frames -> {len(written)} tiles (+{extra} rare-class repeats)")
    return out


def train(data_root: Path, model: str, epochs: int, batch: int, project: Path, name: str, resume: bool,
          arch: str = "yolo") -> Path:
    last = project / name / "weights" / "last.pt"
    if resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume given but no checkpoint at {last}")
        yolo = load_model(last, arch)
        yolo.train(resume=True)  # restores all settings, incl. epochs and data, from the checkpoint
    else:
        yolo = load_model(model, arch)
        yolo.train(
            data=str(write_dataset_yaml(data_root, Path(f"{name}_data.yaml"))), epochs=epochs, imgsz=TILE,
            batch=batch, project=str(project), name=name, max_det=MAX_DETECTIONS, exist_ok=True,
        )
    best = Path(yolo.trainer.best)
    print(f"Best weights: {best}")
    return best


def native_imgsz(width: int, height: int, stride: int = 32) -> int:
    """Inference size that keeps the frame at its own resolution (the scale the model was trained on)."""
    return math.ceil(max(width, height) / stride) * stride


def off_inner_edges(box: Sequence[float], window: Sequence[int], width: int, height: int, margin: float = 2) -> bool:
    """False if a tile box touches a tile border that is not an image border (the object is cut there).

    With 20% overlap such an object (< 128 px) lies fully inside a neighbouring tile, or in the full pass.
    """
    x0, y0, x1, y1 = window
    bx1, by1, bx2, by2 = box  # tile coordinates
    return not ((x0 > 0 and bx1 <= margin) or (y0 > 0 and by1 <= margin)
                or (x1 < width and bx2 >= x1 - x0 - margin) or (y1 < height and by2 >= y1 - y0 - margin))


def sliced_predict(yolo, path: Path, tile: int, overlap: float, full_imgsz: int, conf: float):
    """Full-image pass (+ optional tile passes), merged with class-wise NMS.

    full_imgsz 0 = native resolution. Returns (xyxy boxes, scores, 0-based classes), at most MAX_DETECTIONS.
    """
    import cv2
    import torch
    from torchvision.ops import batched_nms

    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    full = yolo.predict(image, imgsz=full_imgsz or native_imgsz(width, height), conf=conf,
                        max_det=MAX_DETECTIONS, verbose=False)[0].boxes
    boxes, scores, classes = [full.xyxy.cpu()], [full.conf.cpu()], [full.cls.cpu()]
    windows = tile_windows(width, height, tile, overlap) if tile and max(width, height) > tile else []
    if windows:
        crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in windows]
        for window, result in zip(windows, yolo.predict(crops, imgsz=tile, conf=conf, max_det=MAX_DETECTIONS, verbose=False)):
            r = result.boxes.cpu()
            keep = torch.tensor([off_inner_edges(b, window, width, height) for b in r.xyxy.tolist()], dtype=torch.bool)
            boxes.append(r.xyxy[keep] + torch.tensor([window[0], window[1]] * 2, dtype=r.xyxy.dtype))
            scores.append(r.conf[keep])
            classes.append(r.cls[keep])
    boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    keep = batched_nms(boxes, scores, classes.long(), 0.6)[:MAX_DETECTIONS]
    return boxes[keep].tolist(), scores[keep].tolist(), classes[keep].long().tolist()


def evaluate(weights: Path, split_root: Path, step: int, tile: int, full_imgsz: int, conf: float,
             arch: str = "yolo") -> dict[str, float]:
    """COCO mAP on a labeled split, comparable with `src.faster_rcnn evaluate`."""
    from pycocotools.coco import COCO

    from src.data_utils import image_path_for

    coco = COCO(str(split_root / "annotations_coco.json"))
    image_ids = sorted(coco.getImgIds())[::step]
    yolo = load_model(weights, arch)
    results, seconds = [], 0.0
    for index, image_id in enumerate(image_ids):
        tick = time.perf_counter()
        boxes, scores, classes = sliced_predict(
            yolo, image_path_for(split_root, coco.imgs[image_id]["file_name"]), tile, OVERLAP, full_imgsz, conf)
        if index:  # first image is warm-up
            seconds += time.perf_counter() - tick
        results += [{"image_id": image_id, "category_id": c + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s}
                    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes)]
        print(f"{index + 1}/{len(image_ids)} images", end="\r")
    metrics = coco_map(coco, results, image_ids)
    if len(image_ids) > 1:
        metrics["ms_per_image"] = 1000 * seconds / (len(image_ids) - 1)  # includes disk read
        print(f"Inference: {metrics['ms_per_image']:.1f} ms/image (tile {tile or 'off'}, full {full_imgsz or 'native'})")
    return metrics


def predict(weights: Path, test_root: Path, out: Path, tile: int, full_imgsz: int, conf: float,
            arch: str = "yolo") -> Path:
    yolo = load_model(weights, arch)
    with (test_root / "test_images.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "PredictionString"])
        for index, row in enumerate(rows, 1):
            boxes, scores, classes = sliced_predict(yolo, test_root / row["file_name"], tile, OVERLAP, full_imgsz, conf)
            writer.writerow([row["image_id"], format_predictions(classes, scores, boxes)])
            print(f"{index}/{len(rows)} images", end="\r")
    print(f"Wrote {len(rows)} rows to {out}")
    return out


def main(arch: str = "yolo", model: str = "yolo11s.pt", name: str = "tiled", batch: int = 16,
         description: str = __doc__) -> None:
    """CLI shared with src.rtdetr: same tiles and inference, only the model differs."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare", help="Cut frames into training tiles")
    prep.add_argument("--data-root", type=Path, required=True, help="Subset or full dataset (YOLO labels)")
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--tile", type=int, default=TILE)
    prep.add_argument("--overlap", type=float, default=OVERLAP)
    prep.add_argument("--keep-empty", type=float, default=0.1, help="Share of tiles without objects to keep")
    prep.add_argument("--rfs-threshold", type=float, default=0.1, help="0 disables rare-class repeats")
    prep.add_argument("--seed", type=int, default=42)

    tr = commands.add_parser("train")
    tr.add_argument("--data-root", type=Path, required=True, help="Output of `prepare`")
    tr.add_argument("--model", default=model)
    tr.add_argument("--epochs", type=int, default=20)
    tr.add_argument("--batch", type=int, default=batch)
    tr.add_argument("--project", type=Path, default=Path("runs") / name, help="Put on Drive to survive disconnects")
    tr.add_argument("--name", default=name)
    tr.add_argument("--resume", action="store_true", help="Continue from <project>/<name>/weights/last.pt")

    for command in ("evaluate", "predict"):
        sub = commands.add_parser(command)
        sub.add_argument("--weights", type=Path, required=True)
        sub.add_argument("--tile", type=int, default=0, help="Also predict on tiles of this size (0 = off)")
        sub.add_argument("--full-imgsz", type=int, default=0, help="Full-image pass size (0 = native resolution)")
        sub.add_argument("--conf", type=float, default=0.01)  # low: mAP rewards recall
    commands.choices["evaluate"].add_argument("--split-root", type=Path, required=True,
                                              help="Folder with annotations_coco.json, e.g. /content/data/val")
    commands.choices["evaluate"].add_argument("--step", type=int, default=1, help="Evaluate every k-th frame")
    commands.choices["predict"].add_argument("--test-root", type=Path, required=True,
                                             help="Folder containing test_images.csv and test/images/")
    commands.choices["predict"].add_argument("--out", type=Path, default=Path(f"submission_{name}.csv"))

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_root, args.out, args.tile, args.overlap, args.keep_empty, args.rfs_threshold, args.seed)
    elif args.command == "train":
        train(args.data_root, args.model, args.epochs, args.batch, args.project, args.name, args.resume, arch)
    elif args.command == "evaluate":
        evaluate(args.weights, args.split_root, args.step, args.tile, args.full_imgsz, args.conf, arch)
    else:
        predict(args.weights, args.test_root, args.out, args.tile, args.full_imgsz, args.conf, arch)


if __name__ == "__main__":
    main()

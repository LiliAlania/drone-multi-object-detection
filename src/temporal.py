"""Temporal post-processing: improve per-frame detections using previous frames of the same clip.

Works on the Kaggle prediction CSV that every detector in this repo writes, so it applies to
Faster R-CNN, FCOS, YOLO and RT-DETR alike. It is causal (only past frames are used):

1. Camera motion (optional, --motion): a similarity transform between consecutive frames is
   estimated from ORB keypoints + RANSAC, and track boxes are warped before matching.
   Drone footage pans and zooms, so without this small objects drift out of IoU range.
2. Tracking: per class, tracks are matched to the current detections by IoU (Hungarian).
3. Rescoring: a matched detection's score is blended with its track's running score,
   so a weak detection of an object seen confidently before is boosted, and vice versa.
4. Propagation: a confident track missed in this frame (blur, occlusion) adds its warped box
   with a decayed score for up to --max-age frames.

Usage (from the repository root):
    # 1. a val index in test_images.csv format, so any detector's `predict` runs on val
    python -m src.temporal index --split-root /content/data/val --out /content/val_index
    python -m src.models.faster_rcnn predict --weights runs/faster_rcnn/last.pt \
        --test-root /content/val_index --out val_frcnn.csv
    # 2. compare raw vs temporal mAP on val (all val frames, consecutive)
    python -m src.temporal evaluate --pred val_frcnn.csv --images-csv /content/val_index/test_images.csv \
        --split-root /content/data/val --motion
    # 3. apply to a test submission
    python -m src.temporal apply --pred submission_frcnn.csv --images-csv /content/data/test_images.csv \
        --out submission_frcnn_temporal.csv --motion
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from src.models.yolo_baseline import MAX_DETECTIONS, format_predictions

EMPTY = (np.zeros(0, int), np.zeros(0), np.zeros((0, 4)))


def read_predictions(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Kaggle CSV -> {image_id: (0-based classes, scores, xyxy boxes)}."""
    predictions = {}
    with Path(path).open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            values = np.array(row["PredictionString"].split(), dtype=float).reshape(-1, 6)
            xy, wh = values[:, 2:4], values[:, 4:6]
            predictions[row["image_id"]] = (values[:, 0].astype(int), values[:, 1], np.hstack([xy, xy + wh]))
    return predictions


def write_predictions(path: Path, predictions: dict, image_ids: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "PredictionString"])
        for image_id in image_ids:
            classes, scores, boxes = predictions.get(image_id, EMPTY)
            writer.writerow([image_id, format_predictions(classes, scores, boxes)])
    return path


def read_index(images_csv: Path) -> list[dict]:
    """test_images.csv rows sorted by clip, then frame number."""
    with Path(images_csv).open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return sorted(rows, key=lambda row: (row["clip"], int(row["frame"])))


def box_iou(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, shape (len(first), len(second))."""
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(-1)
    area = lambda boxes: (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area(first)[:, None] + area(second)[None, :] - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)


def warp_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine matrix to xyxy boxes; returns the bounding boxes of the warped corners."""
    corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
    warped = corners @ matrix[:, :2].T + matrix[:, 2]
    return np.hstack([warped.min(1), warped.max(1)])


def estimate_motion(previous_gray, gray, scale: float = 0.5):
    """Similarity transform previous -> current frame from ORB matches, or None if unreliable."""
    import cv2

    small = [cv2.resize(image, None, fx=scale, fy=scale) for image in (previous_gray, gray)]
    orb = cv2.ORB_create(2000)
    (kp1, des1), (kp2, des2) = (orb.detectAndCompute(image, None) for image in small)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
    if len(matches) < 10:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]) / scale
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]) / scale
    # moving vehicles are outliers to the camera motion; RANSAC drops them
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or inliers.sum() < 10:
        return None
    return matrix


def temporal_rescore(frames: list[dict], predictions: dict, args, image_root: Path | None = None) -> dict:
    """Track, rescore and propagate detections clip by clip. `frames` comes from read_index."""
    from scipy.optimize import linear_sum_assignment

    output = {}
    for _, clip_frames in itertools.groupby(frames, key=lambda row: row["clip"]):
        tracks: list[dict] = []  # {"cls", "box", "score" (running EMA), "age" (frames since last match)}
        previous_gray = None
        for row in clip_frames:
            classes, scores, boxes = predictions.get(row["image_id"], EMPTY)
            if args.motion:
                import cv2

                gray = cv2.imread(str(image_root / row["file_name"]), cv2.IMREAD_GRAYSCALE)
                matrix = estimate_motion(previous_gray, gray) if previous_gray is not None and gray is not None else None
                previous_gray = gray
                if matrix is not None and tracks:
                    warped = warp_boxes(np.array([t["box"] for t in tracks]), matrix)
                    for track, box in zip(tracks, warped):
                        track["box"] = box

            new_scores = scores.copy()
            kept, extra = [], []
            for cls in set(classes.tolist()) | {t["cls"] for t in tracks}:
                class_tracks = [t for t in tracks if t["cls"] == cls]
                detections = np.flatnonzero(classes == cls)
                matched_tracks, matched_detections = set(), set()
                if class_tracks and len(detections):
                    iou = box_iou(np.array([t["box"] for t in class_tracks]), boxes[detections])
                    for a, b in zip(*linear_sum_assignment(-iou)):
                        if iou[a, b] < args.iou:
                            continue
                        track, index = class_tracks[a], detections[b]
                        new_scores[index] = (1 - args.weight) * scores[index] + args.weight * track["score"]
                        track.update(box=boxes[index], age=0,
                                     score=args.ema * track["score"] + (1 - args.ema) * scores[index])
                        matched_tracks.add(a)
                        matched_detections.add(b)
                for a, track in enumerate(class_tracks):
                    if a not in matched_tracks:
                        track["age"] += 1
                        if track["age"] > args.max_age:
                            continue
                        if track["score"] >= args.propagate:
                            extra.append((cls, track["score"] * args.decay ** track["age"], track["box"]))
                    kept.append(track)
                for b, index in enumerate(detections):
                    if b not in matched_detections and scores[index] >= args.new_track:
                        kept.append({"cls": cls, "box": boxes[index], "score": scores[index], "age": 0})
            tracks = kept

            if extra:
                classes = np.concatenate([classes, [e[0] for e in extra]]).astype(int)
                new_scores = np.concatenate([new_scores, [e[1] for e in extra]])
                boxes = np.vstack([boxes, [e[2] for e in extra]])
            top = np.argsort(-new_scores)[:MAX_DETECTIONS]
            output[row["image_id"]] = (classes[top], new_scores[top], boxes[top])
    return output


def write_index(split_root: Path, out_dir: Path) -> Path:
    """test_images.csv for a labeled split, keyed by COCO image id, with absolute image paths."""
    from src.data_utils import image_path_for

    data = json.loads((split_root / "annotations_coco.json").read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "test_images.csv"
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "file_name", "clip", "frame", "width", "height"])
        for image in data["images"]:
            # absolute path: predict joins test_root / file_name, which keeps an absolute right side
            path = image_path_for(split_root, image["file_name"]).resolve()
            writer.writerow([image["id"], path, image["video"], image["frame_id"], image["width"], image["height"]])
    print(f"Wrote {len(data['images'])} rows to {out}")
    return out


def evaluate(pred: Path, images_csv: Path, split_root: Path, args) -> dict[str, dict[str, float]]:
    """COCO metrics of the raw predictions and of the temporally processed ones."""
    from pycocotools.coco import COCO

    from src.models.faster_rcnn import coco_map

    coco = COCO(str(split_root / "annotations_coco.json"))
    frames = read_index(images_csv)
    raw = read_predictions(pred)
    processed = temporal_rescore(frames, raw, args, Path(images_csv).parent)
    image_ids = [int(row["image_id"]) for row in frames]

    def to_coco(predictions):
        return [{"image_id": int(image_id), "category_id": int(c) + 1, "score": float(s),
                 "bbox": [x1, y1, x2 - x1, y2 - y1]}
                for image_id, (classes, scores, boxes) in predictions.items()
                for c, s, (x1, y1, x2, y2) in zip(classes, scores, boxes.tolist())]

    print("=== raw ===")
    metrics = {"raw": coco_map(coco, to_coco(raw), image_ids)}
    print("=== temporal ===")
    metrics["temporal"] = coco_map(coco, to_coco(processed), image_ids)
    for key in ("mAP", "AP50", "AP_small"):
        print(f"{key}: {metrics['raw'][key]:.4f} -> {metrics['temporal'][key]:.4f}")
    return metrics


def add_temporal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motion", action="store_true", help="Compensate camera motion before matching")
    parser.add_argument("--iou", type=float, default=0.5, help="Min IoU to match a track and a detection")
    parser.add_argument("--weight", type=float, default=0.5, help="Share of the track score in a rescored detection")
    parser.add_argument("--ema", type=float, default=0.7, help="Track score memory (higher = longer)")
    parser.add_argument("--new-track", type=float, default=0.3, help="Min score to start a track")
    parser.add_argument("--propagate", type=float, default=0.5, help="Min track score to propagate a missed box")
    parser.add_argument("--max-age", type=int, default=3, help="Frames a missed track is kept and propagated")
    parser.add_argument("--decay", type=float, default=0.8, help="Score decay per missed frame")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="test_images.csv for a labeled split (e.g. val)")
    index_parser.add_argument("--split-root", type=Path, required=True)
    index_parser.add_argument("--out", type=Path, required=True)

    eval_parser = commands.add_parser("evaluate", help="Raw vs temporal COCO metrics")
    eval_parser.add_argument("--pred", type=Path, required=True, help="Predictions CSV on the indexed split")
    eval_parser.add_argument("--images-csv", type=Path, required=True)
    eval_parser.add_argument("--split-root", type=Path, required=True)
    add_temporal_args(eval_parser)

    apply_parser = commands.add_parser("apply", help="Write a temporally processed predictions CSV")
    apply_parser.add_argument("--pred", type=Path, required=True)
    apply_parser.add_argument("--images-csv", type=Path, required=True,
                              help="test_images.csv; image paths are relative to its folder")
    apply_parser.add_argument("--out", type=Path, required=True)
    add_temporal_args(apply_parser)

    args = parser.parse_args()
    if args.command == "index":
        write_index(args.split_root, args.out)
    elif args.command == "evaluate":
        evaluate(args.pred, args.images_csv, args.split_root, args)
    else:
        frames = read_index(args.images_csv)
        processed = temporal_rescore(frames, read_predictions(args.pred), args, args.images_csv.parent)
        write_predictions(args.out, processed, [row["image_id"] for row in frames])
        print(f"Wrote {len(frames)} rows to {args.out}")


if __name__ == "__main__":
    main()

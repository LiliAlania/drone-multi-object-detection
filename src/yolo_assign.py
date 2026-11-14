"""How anchor-free YOLO assigns grid locations to VisDrone objects, per input size and head set.

Ultralytics YOLO (v8/11) has no anchor boxes: every cell centre of the P3/P4/P5 grids (strides
8/16/32, plus 4 with a P2 head) is an "anchor point" that predicts class scores and the distances
to the four box sides. The TaskAlignedAssigner makes a point a positive for a GT box only if the
point lies inside that box; among those candidates it keeps the top 10 by s^0.5 * IoU^6, and a
point that falls inside several boxes goes to the box its prediction overlaps most.

So an object whose resized box contains no grid point never gets a positive and is learned only
as background, and in dense crowds neighbours compete for the same few points. This script counts,
on the val annotations, how many candidate points each object gets and how often they are shared.

    python -m src.yolo_assign --split-root /content/data/val --step 5 --imgsz 640 960 1280 1536
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEADS = {"P3-P5": (8, 16, 32), "P2-P5": (4, 8, 16, 32)}
SIZES = ("small", "medium", "large")


def grid_centres(length: float, stride: int) -> np.ndarray:
    return (np.arange(int(np.ceil(length / stride))) + 0.5) * stride


def points_inside(boxes: np.ndarray, width: float, height: float, strides) -> np.ndarray:
    """Number of grid-cell centres strictly inside each xyxy box, summed over all strides: (N,)."""
    total = np.zeros(len(boxes), dtype=int)
    for stride in strides:
        xs, ys = grid_centres(width, stride), grid_centres(height, stride)
        nx = np.searchsorted(xs, boxes[:, 2], "left") - np.searchsorted(xs, boxes[:, 0], "right")
        ny = np.searchsorted(ys, boxes[:, 3], "left") - np.searchsorted(ys, boxes[:, 1], "right")
        total += np.clip(nx, 0, None) * np.clip(ny, 0, None)
    return total


def has_shared_point(boxes: np.ndarray, width: float, height: float, strides) -> np.ndarray:
    """True for boxes with at least one candidate point that also lies inside another box: (N,)."""
    shared = np.zeros(len(boxes), dtype=bool)
    if len(boxes) < 2:
        return shared
    lt = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
    overlap = np.concatenate([lt, rb], axis=-1).reshape(-1, 4)
    keep = (overlap[:, 2] > overlap[:, 0]) & (overlap[:, 3] > overlap[:, 1])
    counts = np.zeros(len(overlap), dtype=int)
    counts[keep] = points_inside(overlap[keep], width, height, strides)
    counts = counts.reshape(len(boxes), len(boxes))
    np.fill_diagonal(counts, 0)
    return (counts > 0).any(axis=1)


def size_name(area: float) -> str:
    return SIZES[0] if area < 32 ** 2 else SIZES[1] if area < 96 ** 2 else SIZES[2]


def load_val(split_root: Path, step: int):
    """[(width, height, xyxy boxes, COCO size names)] for every step-th image, ignored regions dropped."""
    coco = json.loads((split_root / "annotations_coco.json").read_text(encoding="utf-8"))
    by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        if not ann.get("iscrowd", 0):
            by_image.setdefault(ann["image_id"], []).append(ann)
    images = {image["id"]: image for image in coco["images"]}
    frames = []
    for image_id in sorted(images)[::step]:
        anns = by_image.get(image_id, [])
        boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)], dtype=float).reshape(-1, 4)
        sizes = [size_name(a.get("area", a["bbox"][2] * a["bbox"][3])) for a in anns]
        frames.append((images[image_id]["width"], images[image_id]["height"], boxes, sizes))
    return frames


def assignment_table(frames, imgsizes, heads=HEADS):
    """One row per (imgsz, head set, object size): share of objects without any candidate point,
    median candidate points, and share with a point shared with another object."""
    import pandas as pd

    rows = []
    for imgsz in imgsizes:
        for head, strides in heads.items():
            points, shared, sizes = [], [], []
            for width, height, boxes, names in frames:
                # shortcut: nominal letterbox scale only; mosaic/scale augmentation and padding offsets are ignored
                r = imgsz / max(width, height)
                scaled = boxes * r
                points.append(points_inside(scaled, width * r, height * r, strides))
                shared.append(has_shared_point(scaled, width * r, height * r, strides))
                sizes += names
            points, shared, sizes = np.concatenate(points), np.concatenate(shared), np.array(sizes)
            for size in ("all", *SIZES):
                mask = np.ones(len(sizes), bool) if size == "all" else sizes == size
                if mask.any():
                    rows.append({"imgsz": imgsz, "heads": head, "size": size, "objects": int(mask.sum()),
                                 "no_point_%": 100 * float((points[mask] == 0).mean()),
                                 "median_points": float(np.median(points[mask])),
                                 "shared_point_%": 100 * float(shared[mask].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split-root", type=Path, required=True, help="Folder with annotations_coco.json")
    parser.add_argument("--step", type=int, default=5, help="Every k-th frame, as in evaluation")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640, 960, 1280, 1536])
    parser.add_argument("--out", type=Path, help="Optional CSV path")
    args = parser.parse_args()
    table = assignment_table(load_val(args.split_root, args.step), args.imgsz)
    print(table.round(1).to_string(index=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()

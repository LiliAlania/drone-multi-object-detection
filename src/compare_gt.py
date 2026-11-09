"""Compare a detector's predictions with the ground truth on a labeled split (error analysis).

Each prediction above --conf is matched to the ground truth (highest score first, as in COCO)
and gets one type:
    tp            right class, IoU >= --iou with a not-yet-matched object
    duplicate     right class and IoU, but that object was already matched
    wrong_class   IoU >= --iou with an object of another class
    localization  right class, 0.1 <= IoU < --iou (box too loose/tight/shifted)
    background    no object nearby
Ground-truth objects nobody matched are `missed`. Predictions centered in ignored regions are skipped.

Writes to --out: errors.csv (one row per prediction / missed object), per_class.csv,
by_size.csv, confusion_matrix.png and frames/*.png with the frames that have the most errors.

Usage (predictions on val keyed by COCO image id, see src.temporal `index`):
    python -m src.compare_gt --pred val_frcnn.csv --split-root /content/data/val --out runs/compare_frcnn
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_utils import CLASS_NAMES, image_path_for
from src.temporal import box_iou, read_predictions

SIZES = ((32**2, "small"), (96**2, "medium"), (float("inf"), "large"))  # COCO area buckets
STYLE = {  # type -> (color, dashed); ground truth solid, predictions dashed (tp shown as matched GT)
    "matched GT": ("lime", False), "missed": ("red", False), "wrong_class": ("magenta", True), "localization": ("yellow", True),
    "duplicate": ("cyan", True), "background": ("orange", True),
}


def size_name(box) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return next(name for limit, name in SIZES if area < limit)


def match_frame(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, iou_thr=0.5):
    """Returns (type per prediction, matched GT index per prediction or -1, GT matched mask)."""
    matched = np.zeros(len(gt_boxes), bool)
    types, linked = ["background"] * len(pred_boxes), [-1] * len(pred_boxes)
    if not len(gt_boxes) or not len(pred_boxes):
        return types, linked, matched
    iou = box_iou(pred_boxes, gt_boxes)
    for p in np.argsort(-pred_scores):
        same = iou[p] * (gt_classes == pred_classes[p])
        free = np.where(matched, 0.0, same)
        if free.max() >= iou_thr:
            types[p], linked[p] = "tp", int(free.argmax())
            matched[linked[p]] = True
        elif same.max() >= iou_thr:
            types[p], linked[p] = "duplicate", int(same.argmax())
        elif iou[p].max() >= iou_thr:
            types[p], linked[p] = "wrong_class", int(iou[p].argmax())
        elif same.max() >= 0.1:
            types[p], linked[p] = "localization", int(same.argmax())
    return types, linked, matched


def in_regions(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """True for boxes whose center lies inside any region (both xyxy)."""
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes), bool)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    inside = ((centers[:, None] >= regions[None, :, :2]) & (centers[:, None] <= regions[None, :, 2:])).all(-1)
    return inside.any(1)


def load_ground_truth(split_root: Path):
    data = json.loads((split_root / "annotations_coco.json").read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    objects = {i: ([], []) for i in images}
    ignored = {i: [] for i in images}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        if a.get("iscrowd"):
            ignored[a["image_id"]].append([x, y, x + w, y + h])
        else:
            objects[a["image_id"]][0].append([x, y, x + w, y + h])
            objects[a["image_id"]][1].append(a["category_id"] - 1)
    as_arrays = lambda boxes: np.array(boxes, float).reshape(-1, 4)
    return (images, {i: (as_arrays(b), np.array(c, int)) for i, (b, c) in objects.items()},
            {i: as_arrays(r) for i, r in ignored.items()})


def analyze(pred: Path, split_root: Path, out: Path, conf: float, iou_thr: float, frames: int):
    import pandas as pd

    images, ground_truth, ignored = load_ground_truth(split_root)
    predictions = read_predictions(pred)
    unknown = [k for k in predictions if not k.isdigit() or int(k) not in images]
    if unknown:
        raise ValueError(f"{len(unknown)} prediction ids are not COCO image ids of this split "
                         f"(e.g. {unknown[0]}); predict on the `src.temporal index` of this split")
    rows, per_frame = [], {}
    for key, (classes, scores, boxes) in predictions.items():
        image_id = int(key)
        keep = (scores >= conf) & ~in_regions(boxes, ignored[image_id])
        classes, scores, boxes = classes[keep], scores[keep], boxes[keep]
        gt_boxes, gt_classes = ground_truth[image_id]
        types, linked, matched = match_frame(gt_boxes, gt_classes, boxes, classes, scores, iou_thr)
        for t, g, c, s, b in zip(types, linked, classes, scores, boxes):
            rows.append({"image_id": image_id, "type": t, "pred_class": CLASS_NAMES[c],
                         "gt_class": CLASS_NAMES[gt_classes[g]] if g >= 0 else "",
                         "size": size_name(b if g < 0 else gt_boxes[g]), "score": float(s)})
        for g in np.flatnonzero(~matched):
            rows.append({"image_id": image_id, "type": "missed", "pred_class": "",
                         "gt_class": CLASS_NAMES[gt_classes[g]], "size": size_name(gt_boxes[g]), "score": np.nan})
        per_frame[image_id] = (types, linked, matched, classes, boxes)

    errors = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    errors.to_csv(out / "errors.csv", index=False)
    print(f"{len(per_frame)} frames, conf >= {conf}, IoU >= {iou_thr}\n")
    print(errors["type"].value_counts().to_string(), "\n")

    gt_rows = errors[errors["type"].isin(["tp", "missed"])]
    per_class = pd.DataFrame({
        "gt_objects": gt_rows.groupby("gt_class").size(),
        "missed": errors[errors["type"] == "missed"].groupby("gt_class").size(),
        "predictions": errors[errors["type"] != "missed"].groupby("pred_class").size(),
        "tp": errors[errors["type"] == "tp"].groupby("pred_class").size(),
        "wrong_class_as": errors[errors["type"] == "wrong_class"].groupby("pred_class").size(),
    }).reindex(CLASS_NAMES).fillna(0).astype(int)
    per_class["recall"] = (per_class["tp"] / per_class["gt_objects"].replace(0, np.nan)).round(3)
    per_class["precision"] = (per_class["tp"] / per_class["predictions"].replace(0, np.nan)).round(3)
    per_class.to_csv(out / "per_class.csv")
    print(per_class.to_string(), "\n")

    by_size = gt_rows.groupby("size")["type"].value_counts().unstack(fill_value=0).reindex([n for _, n in SIZES])
    by_size = by_size.fillna(0).astype(int)
    by_size["recall"] = (by_size.get("tp", 0) / by_size.sum(axis=1)).round(3)
    by_size.to_csv(out / "by_size.csv")
    print("Ground truth by object size:\n" + by_size.to_string(), "\n")

    plot_confusion(errors, out / "confusion_matrix.png")
    worst = sorted(per_frame, key=lambda i: -sum(t != "tp" for t in per_frame[i][0]) - (~per_frame[i][2]).sum())
    by_clip = {}  # worst frame of each clip first, so the examples are not all from one scene
    for image_id in worst:
        by_clip.setdefault(images[image_id].get("video", image_id), image_id)
    chosen = (list(by_clip.values()) + [i for i in worst if i not in by_clip.values()])[:frames]
    for image_id in chosen:
        draw_frame(split_root, images[image_id], ground_truth[image_id], ignored[image_id],
                   per_frame[image_id], out / "frames" / f"{image_id}.png")
    print(f"Saved tables, confusion_matrix.png and {len(chosen)} frames to {out}")
    return errors


def plot_confusion(errors, path: Path) -> None:
    """GT class (rows) vs predicted class (columns); 'background' = missed / false positive."""
    import matplotlib.pyplot as plt

    names = list(CLASS_NAMES) + ["background"]
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), int)
    for t, gt, pred in zip(errors["type"], errors["gt_class"], errors["pred_class"]):
        if t in ("tp", "wrong_class"):
            matrix[index[gt], index[pred]] += 1
        elif t == "missed":
            matrix[index[gt], -1] += 1
        elif t == "background":
            matrix[-1, index[pred]] += 1
    shares = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for (r, c), value in np.ndenumerate(matrix):
        if value:
            ax.text(c, r, value, ha="center", va="center", fontsize=7,
                    color="white" if shares[r, c] > 0.5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix (color = share of row)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def draw_frame(split_root: Path, image: dict, gt, ignored, frame, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from PIL import Image

    (gt_boxes, gt_classes), (types, linked, matched, classes, boxes) = gt, frame
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(Image.open(image_path_for(split_root, image["file_name"])).convert("RGB"))

    def box(b, kind, label=None):
        color, dashed = STYLE[kind]
        ax.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color,
                               linewidth=1.2, linestyle="--" if dashed else "-"))
        if label:
            ax.text(b[0], b[1] - 2, label, color=color, fontsize=6)

    for r in ignored:
        ax.add_patch(Rectangle(r[:2], r[2] - r[0], r[3] - r[1], facecolor="gray", alpha=0.3, hatch="////"))
    for b, c, m in zip(gt_boxes, gt_classes, matched):
        box(b, "matched GT" if m else "missed", None if m else CLASS_NAMES[c])
    for b, c, t, g in zip(boxes, classes, types, linked):
        if t != "tp":  # a correct prediction sits on its green ground-truth box
            box(b, t, f"{CLASS_NAMES[c]} (GT {CLASS_NAMES[gt_classes[g]]})" if t == "wrong_class" else CLASS_NAMES[c])
    counts = {k: types.count(k) for k in STYLE if k in types}
    counts["missed"] = int((~matched).sum())
    ax.legend(handles=[Line2D([], [], color=STYLE[k][0], linestyle="--" if STYLE[k][1] else "-",
                              label=f"{k} ({counts.get(k, int(matched.sum()) if k == 'matched GT' else 0)})")
                       for k in STYLE], loc="upper right", fontsize=8)
    ax.set_title(f"{image['file_name']}  (solid = ground truth, dashed = prediction)")
    ax.set_axis_off()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred", type=Path, required=True, help="Predictions CSV keyed by COCO image id")
    parser.add_argument("--split-root", type=Path, required=True, help="Folder with annotations_coco.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.3, help="Ignore predictions below this score")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--frames", type=int, default=12, help="How many worst frames to draw")
    args = parser.parse_args()
    analyze(args.pred, args.split_root, args.out, args.conf, args.iou, args.frames)


if __name__ == "__main__":
    main()

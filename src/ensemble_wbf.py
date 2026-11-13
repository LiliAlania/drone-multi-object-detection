"""Weighted Boxes Fusion of YOLO-P2 + Faster R-CNN predictions, grid-searched on val.

WBF (ensemble_boxes) only fuses boxes of the same label, i.e. class by class. Boxes are normalized
to [0, 1] by the size of their own frame and scaled back afterwards.

    python -m src.ensemble_wbf search [--debug]          # grid on val -> best config + comparison table
    python -m src.ensemble_wbf apply --split test [--debug]  # fuse test predictions with the best config
Writes <work_dir>/ensemble/{grid.csv,best.json,comparison.md} and preds/wbf_<split>.parquet.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils import CLASS_NAMES
from src.pipeline_common import (DEFAULT_CONFIG, PRED_COLUMNS, list_frames, load_config, load_predictions,
                                 preds_path, save_predictions)

MODELS = ("yolo", "frcnn")
FOCUS = ("person", "bicycle", "motor")  # rider pairs: person over bicycle/motor
COLUMNS = ["mAP", "AP50", "AP_small", *(f"AP_{c}" for c in FOCUS),  # comparison.md
           *(f"AP_{c}" for c in CLASS_NAMES if c not in FOCUS)]


def fuse(predictions: list[pd.DataFrame], sizes: dict[str, tuple[int, int]], weights, iou_thr: float,
         skip_box_thr: float) -> pd.DataFrame:
    """WBF per frame; `sizes` maps image_id -> (width, height)."""
    from ensemble_boxes import weighted_boxes_fusion

    groups = [dict(tuple(df.groupby("image_id"))) for df in predictions]
    rows = []
    for image_id, (width, height) in sizes.items():
        scale = np.array([width, height, width, height], dtype=float)
        boxes, scores, labels = [], [], []
        for group in groups:
            df = group.get(image_id)
            if df is None or df.empty:
                boxes.append(np.zeros((0, 4)))
                scores.append(np.zeros(0))
                labels.append(np.zeros(0))
                continue
            boxes.append(np.clip(df[["x1", "y1", "x2", "y2"]].to_numpy() / scale, 0.0, 1.0))
            scores.append(df["score"].to_numpy())
            labels.append(df["cls"].to_numpy())
        if not any(len(s) for s in scores):
            continue
        fused, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes, scores, labels, weights=list(weights), iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        rows += [(image_id, int(c), float(s), *(b * scale)) for b, s, c in zip(fused, fused_scores, fused_labels)]
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def frame_sizes(cfg: dict, split: str, image_ids) -> dict[str, tuple[int, int]]:
    frames = list_frames(cfg, split).set_index("image_id")
    return {i: (int(frames.at[i, "width"]), int(frames.at[i, "height"])) for i in image_ids}


def load_pair(cfg: dict, split: str) -> tuple[list[pd.DataFrame], list[str]]:
    loaded = [load_predictions(preds_path(cfg, m, split)) for m in MODELS]
    ids = [m["image_ids"] for _, m in loaded]
    if set(ids[0]) != set(ids[1]):
        raise ValueError(f"YOLO and Faster R-CNN {split} predictions cover different frames; rerun src.infer_sahi")
    return [df for df, _ in loaded], ids[0]


def search(cfg: dict) -> dict:
    from src.evaluate import evaluate_detections, load_val_coco

    coco = load_val_coco(cfg)
    predictions, image_ids = load_pair(cfg, "val")
    sizes = frame_sizes(cfg, "val", image_ids)
    out = cfg["paths"]["work_dir"] / "ensemble"
    out.mkdir(parents=True, exist_ok=True)

    def score(df):
        with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints a lot
            return evaluate_detections(coco, df, image_ids)

    e = cfg["ensemble"]
    grid = []
    for weights, iou_thr, skip in itertools.product(e["weights"], e["iou_thr"], e["skip_box_thr"]):
        metrics = score(fuse(predictions, sizes, weights, iou_thr, skip))
        grid.append({"weights": str(weights), "iou_thr": iou_thr, "skip_box_thr": skip,
                     **{k: metrics[k] for k in ("mAP", "AP50", "AP_small")}})
        print(f"weights {weights} iou {iou_thr} skip {skip}: mAP {metrics['mAP']:.4f}")
    grid = pd.DataFrame(grid).sort_values("mAP", ascending=False)
    grid.to_csv(out / "grid.csv", index=False)
    top = grid.iloc[0]
    best = {"weights": json.loads(top["weights"]), "iou_thr": float(top["iou_thr"]),
            "skip_box_thr": float(top["skip_box_thr"])}
    (out / "best.json").write_text(json.dumps(best, indent=1), encoding="utf-8")
    print(f"Best: {best} (mAP {top['mAP']:.4f})")

    # comparison table: each model alone vs the best ensemble
    fused = fuse(predictions, sizes, best["weights"], best["iou_thr"], best["skip_box_thr"])
    save_predictions(fused, preds_path(cfg, "wbf", "val"), image_ids, {"config": best})
    rows = {name: score(df) for name, df in (("YOLOv8s-P2", predictions[0]), ("Faster R-CNN", predictions[1]),
                                             ("WBF", fused))}
    table = pd.DataFrame(rows).T.reindex(columns=COLUMNS)
    for model in MODELS:
        meta = json.loads(preds_path(cfg, model, "val").with_suffix(".json").read_text(encoding="utf-8"))
        table.loc["YOLOv8s-P2" if model == "yolo" else "Faster R-CNN", "ms/frame"] = meta["ms_per_frame"]
    table.loc["WBF", "ms/frame"] = table["ms/frame"].sum()  # runs both models (fusion itself is ~ms)
    markdown = table.round(4).to_markdown()
    (out / "comparison.md").write_text(
        f"Val, {len(image_ids)} frames. WBF config: {best}. person/bicycle/motor = rider pairs.\n\n{markdown}\n",
        encoding="utf-8")
    print(markdown)
    return best


def apply(cfg: dict, split: str) -> Path:
    best_path = cfg["paths"]["work_dir"] / "ensemble" / "best.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"No {best_path}; run `python -m src.ensemble_wbf search` first")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    predictions, image_ids = load_pair(cfg, split)
    fused = fuse(predictions, frame_sizes(cfg, split, image_ids), best["weights"], best["iou_thr"],
                 best["skip_box_thr"])
    path = save_predictions(fused, preds_path(cfg, "wbf", split), image_ids, {"config": best})
    print(f"Fused {len(image_ids)} {split} frames with {best} -> {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["search", "apply"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="test", help="For `apply`")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    search(cfg) if args.command == "search" else apply(cfg, args.split)


if __name__ == "__main__":
    main()

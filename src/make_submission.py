"""Write submission.csv from test predictions (WBF by default) and validate it before saving.

Format (as every submission of this repo): `image_id,PredictionString`, one row per test frame in
test_images.csv order; PredictionString = space-separated `class score x y w h` groups
(class 0..7, xywh in absolute pixels of that frame, <= 500 boxes, highest score first).

    python -m src.make_submission [--pred work/preds/wbf_test.parquet] [--out work/submission.csv] [--debug]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline_common import (DEFAULT_CONFIG, NUM_CLASSES, list_frames, load_config, load_predictions,
                                 preds_path)
from src.models.yolo_baseline import MAX_DETECTIONS, format_predictions

COLUMNS = ["image_id", "PredictionString"]
TOLERANCE = 0.5  # px: format_predictions rounds to 0.1


def build_rows(preds: pd.DataFrame, frames: pd.DataFrame, final_confidence: float) -> list[list[str]]:
    """Clip boxes to their frame, drop empty/low-score ones, keep the top MAX_DETECTIONS per frame."""
    groups = dict(tuple(preds.groupby("image_id")))
    rows = []
    for frame in frames.itertuples(index=False):
        df = groups.get(frame.image_id)
        if df is None:
            rows.append([frame.image_id, ""])
            continue
        boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame.height)
        scores, classes = df["score"].to_numpy(), df["cls"].to_numpy()
        keep = (scores >= final_confidence) & (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
        order = np.argsort(-scores[keep], kind="stable")[:MAX_DETECTIONS]
        rows.append([frame.image_id, format_predictions(classes[keep][order], scores[keep][order],
                                                        boxes[keep][order])])
    return rows


def validate(header: list[str], rows: list[list[str]], frames: pd.DataFrame) -> None:
    """Raise if the submission does not match the expected format."""
    if header != COLUMNS:
        raise ValueError(f"Columns {header} != {COLUMNS}")
    ids = [r[0] for r in rows]
    if ids != frames["image_id"].tolist():
        missing = set(frames["image_id"]) - set(ids)
        raise ValueError(f"Rows do not match test_images.csv order ({len(missing)} frames missing)")
    sizes = frames.set_index("image_id")[["width", "height"]]
    for image_id, text in rows:
        values = np.array(text.split(), dtype=float)
        if values.size % 6:
            raise ValueError(f"{image_id}: PredictionString length {values.size} is not a multiple of 6")
        v = values.reshape(-1, 6)
        width, height = sizes.loc[image_id]
        problems = {
            "class not an integer in 0..7": (v[:, 0] != v[:, 0].round()) | (v[:, 0] < 0) | (v[:, 0] >= NUM_CLASSES),
            "score outside [0, 1]": (v[:, 1] < 0) | (v[:, 1] > 1),
            "box outside the frame": (v[:, 2] < -TOLERANCE) | (v[:, 3] < -TOLERANCE)
            | (v[:, 2] + v[:, 4] > width + TOLERANCE) | (v[:, 3] + v[:, 5] > height + TOLERANCE),
            "non-positive width/height": (v[:, 4] <= 0) | (v[:, 5] <= 0),
        }
        for problem, bad in problems.items():
            if bad.any():
                raise ValueError(f"{image_id}: {problem}: {v[bad][0].tolist()}")
        if len(v) > MAX_DETECTIONS:
            raise ValueError(f"{image_id}: {len(v)} boxes > {MAX_DETECTIONS}")


def make(cfg: dict, pred: Path, out: Path) -> Path:
    preds, meta = load_predictions(pred)
    frames = list_frames({**cfg, "debug_mode": False}, "test")  # the file always lists every test frame
    missing = set(frames["image_id"]) - set(meta["image_ids"])
    if missing and not cfg["debug_mode"]:  # debug predicts only a few frames
        raise ValueError(f"{len(missing)} test frames were never predicted (e.g. {sorted(missing)[0]}); "
                         f"rerun src.infer_sahi on the full test split")
    rows = build_rows(preds, frames, cfg["submission"]["final_confidence"])
    validate(COLUMNS, rows, frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    boxes = sum(len(r[1].split()) // 6 for r in rows)
    print(f"Wrote {out}: {len(rows)} frames ({len(rows) - len(missing)} predicted), {boxes} boxes; validation OK")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pred", type=Path, help="Default: <work_dir>/preds/wbf_test.parquet")
    parser.add_argument("--out", type=Path, help="Default: <work_dir>/submission.csv")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    make(cfg, args.pred or preds_path(cfg, "wbf", "test"), args.out or cfg["paths"]["work_dir"] / "submission.csv")


if __name__ == "__main__":
    main()

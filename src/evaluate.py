"""COCO evaluation used everywhere in the ensemble pipeline (mAP@[.5:.95], AP50, AP_small, AP per class).

maxDets = 500 (the Kaggle cap); ignored regions (iscrowd=1) are neither TP nor FP.

    python -m src.evaluate --pred work/preds/yolo_val.parquet [--debug]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import pandas as pd

from src.pipeline_common import DEFAULT_CONFIG, load_config, load_predictions


def load_val_coco(cfg: dict):
    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(cfg["paths"]["data_root"] / "val" / "annotations_coco.json"))


def coco_from_yolo_tiles(image_dir: Path):
    """In-memory COCO ground truth for a tile folder (image ids 1..N in sorted path order)."""
    from PIL import Image
    from pycocotools.coco import COCO

    from src.data_utils import CLASS_NAMES, _label_path, _read_yolo_rows

    paths = sorted(image_dir.glob("*.jpg"))
    images, annotations = [], []
    for image_id, path in enumerate(paths, 1):
        width, height = Image.open(path).size  # header only
        images.append({"id": image_id, "width": width, "height": height, "file_name": path.name})
        for cls, cx, cy, w, h in _read_yolo_rows(_label_path(path, "labels"), True):
            bw, bh = w * width, h * height
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": int(cls) + 1,
                                "bbox": [cx * width - bw / 2, cy * height - bh / 2, bw, bh], "area": bw * bh,
                                "iscrowd": 0})
    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations,
                    "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco, paths


def evaluate_detections(coco, preds: pd.DataFrame, image_ids=None) -> dict[str, float]:
    """Metrics of predictions (PRED_COLUMNS, image_id = COCO id as str) on the given / predicted images."""
    from src.faster_rcnn import coco_map

    ids = sorted({int(i) for i in (image_ids if image_ids is not None else preds["image_id"])})
    results = [{"image_id": int(r.image_id), "category_id": int(r.cls) + 1, "score": float(r.score),
                "bbox": [r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1]} for r in preds.itertuples(index=False)]
    if not results:  # untrained model: report zeros instead of crashing in pycocotools
        return {"mAP": 0.0, "AP50": 0.0, "AP_small": 0.0}
    return coco_map(coco, results, ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pred", type=Path, required=True, help="Predictions parquet on val (src.infer_sahi)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    preds, meta = load_predictions(args.pred)
    metrics = evaluate_detections(load_val_coco(cfg), preds, meta["image_ids"])
    if "ms_per_frame" in meta:
        metrics["ms_per_frame"] = meta["ms_per_frame"]
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()

"""Sliced (SAHI) inference of one model on val or test frames; saves raw predictions for the ensemble.

640 px slices with 20% overlap plus one full-frame pass; slice results merged per class
(postprocess_class_agnostic=False keeps a rider's person box next to its bicycle/motor box)
with GREEDYNMM on IOS. Low confidence (0.05): the final threshold is applied after the ensemble.

    python -m src.infer_sahi --model yolo --split val [--debug] [--weights PATH]
yolo_hr (src.train_yolo_hr) is not sliced: one Ultralytics pass on the full frame at yolo_hr.imgsz
(TTA if yolo_hr.tta); on val it also prints its comparison.md row.
Writes <work_dir>/preds/<model>_<split>.parquet (+ .json with frame list and ms/frame).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
from pathlib import Path

import pandas as pd

from src.data_utils import CLASS_NAMES
from src.pipeline_common import (DEFAULT_CONFIG, PRED_COLUMNS, list_frames, load_config, preds_path,
                                 save_predictions, seed_everything)


def default_weights(cfg: dict, model: str) -> Path:
    runs = cfg["paths"]["runs_dir"]
    if model == "yolo_hr":
        return runs / "yolo_hr" / "weights" / "best.pt"
    return runs / "yolo_p2" / "weights" / "best.pt" if model == "yolo" else runs / "frcnn" / "best.pt"


def load_detection_model(model: str, weights: Path, cfg: dict):
    """SAHI model + offset that turns its category ids into 0-based class indices."""
    import torch
    from sahi import AutoDetectionModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    conf = cfg["infer"]["confidence"]
    if model == "yolo":
        return AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=str(weights),
                                                  confidence_threshold=conf, device=device,
                                                  image_size=cfg["infer"]["slice"]), 0
    from src.models.faster_rcnn import load_trained

    network, _ = load_trained(weights)  # torchvision labels 1..8 (0 = background)
    return AutoDetectionModel.from_pretrained(
        model_type="torchvision", model=network, confidence_threshold=conf, device=device,
        category_mapping={str(i + 1): name for i, name in enumerate(CLASS_NAMES)}), 1


def predict_full_frames(cfg: dict, weights: Path, frames: pd.DataFrame) -> tuple[list, float]:
    """Ultralytics on whole frames (no slicing) -> PRED_COLUMNS rows and ms/frame (first frame = warm-up)."""
    from ultralytics import YOLO

    from src.train_yolo_hr import hr_config
    from src.models.yolo_baseline import MAX_DETECTIONS

    y, model = hr_config(cfg), YOLO(str(weights))
    rows, seconds = [], 0.0
    for index, frame in enumerate(frames.itertuples(index=False)):
        tick = time.perf_counter()
        boxes = model.predict(frame.path, imgsz=y["imgsz"], conf=cfg["infer"]["confidence"], max_det=MAX_DETECTIONS,
                              augment=y["tta"], verbose=False)[0].boxes
        if index:
            seconds += time.perf_counter() - tick
        rows += [(frame.image_id, int(c), s, *xyxy)
                 for c, s, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist())]
        print(f"yolo_hr: {index + 1}/{len(frames)} frames", end="\r")
    return rows, 1000 * seconds / max(len(frames) - 1, 1)


def print_val_row(cfg: dict, path: Path) -> None:
    """The model's row in the format of <work_dir>/ensemble/comparison.md."""
    from src.ensemble_wbf import COLUMNS
    from src.evaluate import evaluate_detections, load_val_coco
    from src.pipeline_common import load_predictions

    preds, meta = load_predictions(path)
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints a lot
        metrics = evaluate_detections(load_val_coco(cfg), preds, meta["image_ids"])
    table = pd.DataFrame({path.stem.removesuffix("_val"): metrics}).T.reindex(columns=COLUMNS)
    table["ms/frame"] = meta["ms_per_frame"]
    print(f"Val, {len(meta['image_ids'])} frames:\n{table.round(4).to_markdown()}")


def run(cfg: dict, model: str, split: str, weights: Path | None = None) -> Path:
    import torch
    from sahi.predict import get_sliced_prediction

    weights = weights or default_weights(cfg, model)
    if not Path(weights).is_file():
        raise FileNotFoundError(f"No weights at {weights}; train the model first or pass --weights")
    if model == "yolo_hr":
        frames = list_frames(cfg, split)
        rows, ms = predict_full_frames(cfg, Path(weights), frames)
        print(f"{model} {split}: {len(frames)} frames, {len(rows)} boxes, {ms:.0f} ms/frame          ")
        path = save_predictions(pd.DataFrame(rows, columns=PRED_COLUMNS), preds_path(cfg, model, split),
                                frames["image_id"], {"ms_per_frame": ms, "weights": str(weights),
                                                     "tta": cfg["yolo_hr"]["tta"]})
        if split == "val":
            print_val_row(cfg, path)
        return path
    detector, offset = load_detection_model(model, Path(weights), cfg)
    frames = list_frames(cfg, split)
    i = cfg["infer"]
    rows, seconds = [], 0.0
    for index, frame in enumerate(frames.itertuples(index=False)):
        tick = time.perf_counter()
        with torch.inference_mode():  # SAHI's torchvision wrapper does not disable autograd itself
            result = get_sliced_prediction(
                frame.path, detector, slice_height=i["slice"], slice_width=i["slice"],
                overlap_height_ratio=i["overlap"], overlap_width_ratio=i["overlap"], perform_standard_pred=True,
                postprocess_type=i["postprocess_type"], postprocess_match_metric=i["match_metric"],
                postprocess_match_threshold=i["match_threshold"], postprocess_class_agnostic=False, verbose=0,
                force_postprocess_type=True)  # else SAHI swaps to NMS/IOU at confidence < 0.1
        if index:  # first frame is warm-up
            seconds += time.perf_counter() - tick
        rows += [(frame.image_id, p.category.id - offset, p.score.value, *p.bbox.to_xyxy())
                 for p in result.object_prediction_list]
        print(f"{model} {split}: {index + 1}/{len(frames)} frames", end="\r")
    ms = 1000 * seconds / max(len(frames) - 1, 1)
    print(f"{model} {split}: {len(frames)} frames, {len(rows)} boxes, {ms:.0f} ms/frame          ")
    return save_predictions(pd.DataFrame(rows, columns=PRED_COLUMNS), preds_path(cfg, model, split),
                            frames["image_id"], {"ms_per_frame": ms, "weights": str(weights)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", choices=["yolo", "frcnn", "yolo_hr"], required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--weights", type=Path, help="Default: best weights of the training run")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    seed_everything(cfg["seed"])
    print(f"Saved {run(cfg, args.model, args.split, args.weights)}")


if __name__ == "__main__":
    main()

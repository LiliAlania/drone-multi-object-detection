"""Sliced (SAHI) inference of one model on val or test frames; saves raw predictions for the ensemble.

640 px slices with 20% overlap plus one full-frame pass; slice results merged per class
(postprocess_class_agnostic=False keeps a rider's person box next to its bicycle/motor box)
with GREEDYNMM on IOS. Low confidence (0.05): the final threshold is applied after the ensemble.

    python -m src.infer_sahi --model yolo --split val [--debug] [--weights PATH]
Writes <work_dir>/preds/<model>_<split>.parquet (+ .json with frame list and ms/frame).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.data_utils import CLASS_NAMES
from src.pipeline_common import (DEFAULT_CONFIG, PRED_COLUMNS, list_frames, load_config, preds_path,
                                 save_predictions, seed_everything)


def default_weights(cfg: dict, model: str) -> Path:
    runs = cfg["paths"]["runs_dir"]
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
    from src.faster_rcnn import load_trained

    network, _ = load_trained(weights)  # torchvision labels 1..8 (0 = background)
    return AutoDetectionModel.from_pretrained(
        model_type="torchvision", model=network, confidence_threshold=conf, device=device,
        category_mapping={str(i + 1): name for i, name in enumerate(CLASS_NAMES)}), 1


def run(cfg: dict, model: str, split: str, weights: Path | None = None) -> Path:
    import torch
    from sahi.predict import get_sliced_prediction

    weights = weights or default_weights(cfg, model)
    if not Path(weights).is_file():
        raise FileNotFoundError(f"No weights at {weights}; train the model first or pass --weights")
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
    parser.add_argument("--model", choices=["yolo", "frcnn"], required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--weights", type=Path, help="Default: best weights of the training run")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    seed_everything(cfg["seed"])
    print(f"Saved {run(cfg, args.model, args.split, args.weights)}")


if __name__ == "__main__":
    main()

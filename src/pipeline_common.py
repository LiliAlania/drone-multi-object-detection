"""Shared pieces of the ensemble pipeline: config, class mapping, frame lists, prediction files."""

from __future__ import annotations

import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils import CLASS_NAMES

# The one class mapping of the pipeline: training index = COCO category_id - 1 = submission class.
NUM_CLASSES = len(CLASS_NAMES)
CATEGORY_TO_INDEX = {i + 1: i for i in range(NUM_CLASSES)}
PRED_COLUMNS = ["image_id", "cls", "score", "x1", "y1", "x2", "y2"]
DEFAULT_CONFIG = Path("configs/pipeline.yaml")


def load_config(path: Path = DEFAULT_CONFIG, debug: bool = False) -> dict:
    """Read the YAML config, pick the environment's paths, apply --debug overrides."""
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    env = os.environ.get("PIPELINE_ENV", cfg["env"])
    if env == "auto":  # each platform sets its own variable
        env = ("kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
               else "colab" if os.environ.get("COLAB_RELEASE_TAG") else "local")
    cfg["env"] = env
    cfg["paths"] = {k: v for k, v in cfg["envs"][env].items()}
    if os.environ.get("PIPELINE_DATA_ROOT"):
        cfg["paths"]["data_root"] = os.environ["PIPELINE_DATA_ROOT"]
    for key in ("data_root", "work_dir", "runs_dir"):
        cfg["paths"][key] = Path(cfg["paths"][key])
    cfg["debug_mode"] = debug
    if debug:
        d = cfg["debug"]
        cfg["paths"]["work_dir"] /= "debug"
        cfg["paths"]["runs_dir"] /= "debug"
        cfg["yolo"]["epochs"] = cfg["frcnn"]["epochs"] = d["epochs"]
        cfg["frcnn"]["val_every"] = 1
        cfg["ensemble"] = copy.deepcopy(d["ensemble"])
    return cfg


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def tiles_dir(cfg: dict) -> Path:
    return cfg["paths"]["work_dir"] / "tiles"


def preds_path(cfg: dict, model: str, split: str) -> Path:
    return cfg["paths"]["work_dir"] / "preds" / f"{model}_{split}.parquet"


def clip_dirs(cfg: dict, split: str) -> list[Path]:
    """Clip folders of a labeled split (data_root/<split>/images/<clip>), debug-limited."""
    clips = sorted(p for p in (cfg["paths"]["data_root"] / split / "images").iterdir() if p.is_dir())
    if cfg["debug_mode"]:
        n = cfg["debug"]["train_clips" if split == "train" else "val_clips"]
        clips = clips[:: max(len(clips) // n, 1)][:n]  # spread over the list: different scenes and classes
    return clips


def list_frames(cfg: dict, split: str) -> pd.DataFrame:
    """Frames to run inference on: image_id, path, clip, frame, width, height.

    val: COCO image ids, every infer.val_frame_step-th frame of each clip; test: test_images.csv.
    """
    root = cfg["paths"]["data_root"]
    if split == "val":
        data = json.loads((root / "val" / "annotations_coco.json").read_text(encoding="utf-8"))
        frames = pd.DataFrame([{"image_id": str(i["id"]), "path": str(root / "val" / i["file_name"]),
                                "clip": i["video"], "frame": int(i["frame_id"]),
                                "width": int(i["width"]), "height": int(i["height"])} for i in data["images"]])
        frames = frames.sort_values(["clip", "frame"])
        frames = frames[frames.groupby("clip").cumcount() % cfg["infer"]["val_frame_step"] == 0]
        if cfg["debug_mode"]:
            first = sorted(frames["clip"].unique())[: cfg["debug"]["val_clips"]]
            frames = frames[frames["clip"].isin(first)].head(cfg["debug"]["val_frames"])
    elif split == "test":
        with (root / "test_images.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        frames = pd.DataFrame([{"image_id": r["image_id"], "path": str(root / r["file_name"]), "clip": r["clip"],
                                "frame": int(r["frame"]), "width": int(r["width"]), "height": int(r["height"])}
                               for r in rows])
        if cfg["debug_mode"]:
            frames = frames.head(cfg["debug"]["test_frames"])
    else:
        raise ValueError(f"split must be val or test, got {split}")
    return frames.reset_index(drop=True)


def save_predictions(df: pd.DataFrame, path: Path, image_ids, extra: dict | None = None) -> Path:
    """Parquet with PRED_COLUMNS + a JSON sidecar listing every processed frame (frames with 0 boxes too)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[PRED_COLUMNS].to_parquet(path, index=False)
    sidecar = {"image_ids": [str(i) for i in image_ids], **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
    return path


def load_predictions(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"No predictions at {path}; run src.infer_sahi first")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    df = pd.read_parquet(path)
    df["image_id"] = df["image_id"].astype(str)
    return df, meta

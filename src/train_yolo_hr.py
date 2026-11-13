"""Train a high-resolution YOLO on FULL frames (no tiles) for small objects (Ultralytics, config `yolo_hr:`).

Labels follow src.prepare_tiles: ignored regions painted (114,114,114) and boxes centred in them dropped
(crop_targets with the whole frame as the window). Frames are stored pre-resized to long side = imgsz
(Ultralytics resizes them on every load anyway), which saves JPEG decoding on Kaggle's 4 CPUs.
Train: every frame_step-th frame of the train clips; val: exactly the frames of list_frames(cfg, "val").

frame_step "auto": a 3-epoch run on the val frames measures train img/s and the val pass on the real devices,
then the smallest step of yolo_hr.frame_steps that still gives >= yolo_hr.min_epochs within the budget is used.
Training is time-boxed (Ultralytics `time`, hours, counted from script start): Ultralytics re-estimates the
epoch count every epoch and finishes normally (final val, best.pt).

    python -m src.train_yolo_hr [--debug] [--resume]
Weights: <runs_dir>/yolo_hr/weights/{best,last}.pt; frames in <work_dir>/yolo_hr (rebuilt if missing).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from src.data_utils import CLASS_NAMES, _label_path, _read_yolo_rows
from src.pipeline_common import (CATEGORY_TO_INDEX, DEFAULT_CONFIG, clip_dirs, list_frames, load_config,
                                 seed_everything)
from src.models.tiled_yolo import GRAY, crop_targets, yolo_to_xyxy
from src.models.yolo_baseline import MAX_DETECTIONS

START = time.perf_counter()
DEBUG_BUDGET_HOURS = 0.1  # debug trains fixed epochs; this nominal budget only exercises the step choice


def hr_config(cfg: dict) -> dict:
    y = dict(cfg["yolo_hr"])
    if cfg["debug_mode"]:
        y.update(imgsz=640, epochs=cfg["debug"]["epochs"], time=None)
    return y


def hours_left(y: dict) -> float | None:
    return y["time"] and y["time"] - (time.perf_counter() - START) / 3600


def write_frame(path: Path, out_image: Path, imgsz: int) -> int:
    """Paint ignored regions, filter labels like prepare_tiles, save resized image + YOLO labels; returns #boxes."""
    import cv2

    image = cv2.imread(str(path))
    height, width = image.shape[:2]
    targets = yolo_to_xyxy(_read_yolo_rows(_label_path(path, "labels"), True), width, height)
    ignored = yolo_to_xyxy(_read_yolo_rows(_label_path(path, "ignore"), False), width, height)
    for ix1, iy1, ix2, iy2 in ignored:
        image[max(int(iy1), 0):math.ceil(iy2), max(int(ix1), 0):math.ceil(ix2)] = GRAY
    rows = crop_targets(targets, ignored, (0, 0, width, height))
    ratio = imgsz / max(width, height)
    if ratio < 1:  # labels are normalized: resizing does not change them
        image = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    _label_path(out_image, "labels").write_text(
        "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows), encoding="utf-8")
    return len(rows)


def write_split(paths: list[Path], out: Path, split: str, imgsz: int) -> int:
    for folder in ("images", "labels"):
        shutil.rmtree(out / folder / split, ignore_errors=True)
        (out / folder / split).mkdir(parents=True)
    targets = [out / "images" / split / f"{p.parent.name}_{p.stem}.jpg" for p in paths]
    with ThreadPoolExecutor(os.cpu_count()) as pool:  # cv2 releases the GIL
        boxes = sum(pool.map(write_frame, paths, targets, [imgsz] * len(paths)))
    print(f"{split}: {len(paths)} frames, {boxes} boxes -> {out / 'images' / split}")
    return len(paths)


def train_frames(cfg: dict, step: int) -> list[Path]:
    return [p for clip in clip_dirs(cfg, "train") for p in sorted(clip.glob("*.jpg"))[::step]]


def check_classes(cfg: dict, data_yaml: Path) -> None:
    """data.yaml ids == CLASS_NAMES == the source labels' data.yaml == COCO category_id - 1."""
    import yaml

    root = cfg["paths"]["data_root"]
    names = dict(enumerate(yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]))  # list: id = index
    source = yaml.safe_load((root / "train" / "data.yaml").read_text(encoding="utf-8"))["names"]
    coco = json.loads((root / "val" / "annotations_coco.json").read_text(encoding="utf-8"))["categories"]
    expected = dict(enumerate(CLASS_NAMES))
    assert names == expected, f"data.yaml names {names} != CLASS_NAMES"
    assert dict(enumerate(source)) == expected, f"source labels use {source}"
    assert {CATEGORY_TO_INDEX[c["id"]]: c["name"] for c in coco} == expected, f"COCO categories {coco}"
    print("class ids OK (data.yaml = CLASS_NAMES = source labels = COCO id - 1):",
          ", ".join(f"{i}={n}" for i, n in names.items()))


def prepare(cfg: dict, step: int | None, imgsz: int) -> Path:
    """<work_dir>/yolo_hr/{images,labels}/{train,val} + data.yaml; splits already built with the same settings
    are reused. step=None: val frames only, and data.yaml trains on them (speed benchmark)."""
    out = cfg["paths"]["work_dir"] / "yolo_hr"
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if meta.get("imgsz") != imgsz:
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        meta = {"imgsz": imgsz}
    if "val" not in meta:
        meta["val"] = write_split([Path(p) for p in list_frames(cfg, "val")["path"]], out, "val", imgsz)
    if step and meta.get("frame_step") != step:
        meta["train"] = write_split(train_frames(cfg, step), out, "train", imgsz)
        meta["frame_step"] = step
    meta_path.write_text(json.dumps(meta))
    data = out / "data.yaml"
    data.write_text(f"path: {out.resolve().as_posix()}\ntrain: images/{'train' if step else 'val'}\n"
                    f"val: images/val\nnames: {json.dumps(CLASS_NAMES)}\n", encoding="utf-8")
    check_classes(cfg, data)
    print(f"Data: {meta}")
    return data.resolve()


def train_args(cfg: dict, y: dict) -> dict:
    return dict(imgsz=y["imgsz"], rect=y["rect"], amp=True, mosaic=y["mosaic"], close_mosaic=y["close_mosaic"],
                scale=y["scale"], fliplr=y["fliplr"], hsv_h=y["hsv_h"], hsv_s=y["hsv_s"], hsv_v=y["hsv_v"],
                patience=y["patience"], seed=cfg["seed"], workers=cfg["paths"]["workers"], max_det=MAX_DETECTIONS)


def devices(y: dict):
    import torch

    wanted = [d for d in y["device"] if d < torch.cuda.device_count()]
    return wanted if len(wanted) > 1 else 0


def fit(model_path: str, last: Path, device, batch: int, **kwargs):
    """YOLO(model_path).train(...); if multi-GPU (DDP) fails, continue on GPU 0 with half the batch.

    Returns the device that worked. A run that died mid-way resumes from `last` instead of restarting."""
    from ultralytics import YOLO

    tick = time.perf_counter()
    try:
        YOLO(model_path).train(device=device, batch=batch, **kwargs)
        return device
    except Exception as error:  # DDP runs in a subprocess: its failures arrive as CalledProcessError
        if device == 0:
            raise
        print(f"DDP on {device} failed ({error!r}); falling back to device=0, batch {batch // 2}", flush=True)
    if kwargs.get("time"):
        kwargs["time"] -= (time.perf_counter() - tick) / 3600
    if last.is_file():
        YOLO(str(last)).train(resume=True, device=0, batch=batch // 2, time=kwargs.get("time"))
    else:
        YOLO(model_path).train(device=0, batch=batch // 2, **kwargs)
    return 0


def read_results(run_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(run_dir / "results.csv")
    results.columns = results.columns.str.strip()
    return results


def benchmark(cfg: dict, y: dict, device, project: Path) -> tuple[float, float, object]:
    """3 epochs on the val frames without per-epoch val: epoch 2 = pure training, epoch 3 = training + val pass.

    Returns (train img/s, seconds per val pass, device that worked)."""
    data, name = prepare(cfg, None, y["imgsz"]), "yolo_hr_bench"
    shutil.rmtree(project / name, ignore_errors=True)
    args = {**train_args(cfg, y), "close_mosaic": 0}  # keep mosaic on: it is what the real epochs cost
    device = fit(y["model"], project / name / "weights" / "last.pt", device, y["batch"], data=str(data),
                 epochs=3, val=False, plots=False, project=str(project), name=name, exist_ok=True, **args)
    t = read_results(project / name)["time"].to_list()
    shutil.rmtree(project / name, ignore_errors=True)
    n_val = len(list((data.parent / "images" / "val").glob("*.jpg")))
    return n_val / max(t[1] - t[0], 1e-6), max((t[2] - t[1]) - (t[1] - t[0]), 0.0), device


def choose_step(cfg: dict, y: dict, img_per_s: float, val_s: float, hours: float) -> int:
    print(f"Measured: {img_per_s:.1f} train img/s, {val_s:.0f} s per val pass; budget {hours:.2f} h")
    chosen = None
    for step in y["frame_steps"]:
        n = len(train_frames(cfg, step))
        epoch_s = n / img_per_s + val_s
        print(f"  frame_step {step}: {n} frames, {epoch_s / 60:.1f} min/epoch -> ~{hours * 3600 / epoch_s:.0f} epochs")
        if chosen is None and hours * 3600 / epoch_s >= y["min_epochs"]:
            chosen = step
    if chosen is None:
        chosen = y["frame_steps"][-1]
        n = len(train_frames(cfg, chosen))
        at_1024 = hours * 3600 / (n / (img_per_s * (y["imgsz"] / 1024) ** 2) + val_s)
        print(f"WARNING: no frame_step reaches {y['min_epochs']} epochs; using {chosen}. "
              f"imgsz 1024 would give ~{at_1024:.0f} epochs at this step (set yolo_hr.imgsz: 1024).")
    print(f"Chosen frame_step: {chosen}")
    return chosen


def train(cfg: dict, resume: bool = False) -> Path:
    y, project = hr_config(cfg), cfg["paths"]["runs_dir"].resolve()
    run_dir = project / "yolo_hr"
    last, device = run_dir / "weights" / "last.pt", devices(y)
    print(f"Device: {device}, batch {y['batch']} (total over GPUs)")
    if resume:
        if not last.is_file():
            raise FileNotFoundError(f"--resume given but no checkpoint at {last}")
        chosen = json.loads((run_dir / "hr_data.json").read_text())
        prepare(cfg, chosen["frame_step"], chosen["imgsz"])  # frames are not kept between Kaggle sessions
        # Ultralytics restarts the time budget on resume and re-estimates the total epoch count from it
        fit(str(last), last, device, y["batch"], resume=True, time=hours_left(y))
    else:
        if last.is_file() and not cfg["debug_mode"]:
            raise FileExistsError(f"{last} exists; pass --resume or delete {run_dir}")
        shutil.rmtree(run_dir, ignore_errors=True)  # results.csv would be appended to otherwise
        step = y["frame_step"]
        if step == "auto":
            img_per_s, val_s, device = benchmark(cfg, y, device, project)
            step = choose_step(cfg, y, img_per_s, val_s, hours_left(y) or DEBUG_BUDGET_HOURS)
        data = prepare(cfg, step, y["imgsz"])
        run_dir.mkdir(parents=True)
        (run_dir / "hr_data.json").write_text(json.dumps({"frame_step": step, "imgsz": y["imgsz"]}))
        hours = hours_left(y)  # data preparation and the benchmark count against the budget
        print(f"Training {y['model']} at imgsz {y['imgsz']} on {device}, frame_step {step}, "
              + (f"time budget {hours:.2f} h" if hours else f"{y['epochs']} epochs"), flush=True)
        fit(y["model"], last, device, y["batch"], data=str(data), epochs=y["epochs"], time=hours,
            project=str(project), name="yolo_hr", exist_ok=True, **train_args(cfg, y))
    results = read_results(run_dir)
    print(f"YOLO-HR: {len(results)} epochs logged (`time` column = seconds since the session's start)")
    print(results.tail(3).to_string(index=False))
    return run_dir / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue from <runs_dir>/yolo_hr/weights/last.pt")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    seed_everything(cfg["seed"])
    print(f"Best weights: {train(cfg, args.resume)}")


if __name__ == "__main__":
    main()

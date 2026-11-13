"""Cut train/val frames into the 640x640 tiles that BOTH ensemble models train on.

Per frame (every data.frame_step-th of each clip): ignored regions (VisDrone classes 0 "ignored" and
11 "others", iscrowd=1 here) are painted (114,114,114) and boxes centred in them dropped, BEFORE tiling.
Tiles: data.tile px, data.overlap overlap; a box is kept if >= 50% of its area is inside the tile
(clipped to the edge) and both clipped sides are >= 2 px. Only data.keep_empty of empty tiles are kept.

    python -m src.prepare_tiles [--debug] [--force]
Writes <work_dir>/tiles/{images,labels}/{train,val}, data.yaml, stats.json and vis/*.jpg.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

from src.data_utils import CLASS_NAMES, _label_path, _read_yolo_rows
from src.pipeline_common import DEFAULT_CONFIG, clip_dirs, load_config, seed_everything, tiles_dir
from src.models.tiled_yolo import GRAY, crop_targets, tile_windows, yolo_to_xyxy

N_VIS = 5


def prepare(cfg: dict, force: bool = False) -> dict:
    import cv2

    out = tiles_dir(cfg)
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists; pass --force to rebuild it")
        shutil.rmtree(out)
    d = cfg["data"]
    rng = random.Random(cfg["seed"])
    stats = {}
    for split, step in (("train", d["frame_step"]), ("val", d["val_frame_step"])):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        clips = clip_dirs(cfg, split)
        frames = [p for clip in clips for p in sorted(clip.glob("*.jpg"))[::step]]
        boxes, n_tiles, n_empty = Counter(), 0, 0
        for index, path in enumerate(frames, 1):
            image = cv2.imread(str(path))
            height, width = image.shape[:2]
            targets = yolo_to_xyxy(_read_yolo_rows(_label_path(path, "labels"), True), width, height)
            ignored = yolo_to_xyxy(_read_yolo_rows(_label_path(path, "ignore"), False), width, height)
            for ix1, iy1, ix2, iy2 in ignored:
                image[max(int(iy1), 0):math.ceil(iy2), max(int(ix1), 0):math.ceil(ix2)] = GRAY
            for window in tile_windows(width, height, d["tile"], d["overlap"]):
                rows = crop_targets(targets, ignored, window)
                if not rows:
                    if rng.random() >= d["keep_empty"]:
                        continue
                    n_empty += 1
                x0, y0, x1, y1 = window
                stem = f"{path.parent.name}_{path.stem}_{x0}_{y0}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), image[y0:y1, x0:x1])
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows), encoding="utf-8")
                boxes.update(CLASS_NAMES[r[0]] for r in rows)
                n_tiles += 1
            print(f"{split}: {index}/{len(frames)} frames, {n_tiles} tiles", end="\r")
        stats[split] = {"clips": len(clips), "frames": len(frames), "tiles": n_tiles, "empty_tiles": n_empty,
                        "tiles_per_frame": round(n_tiles / max(len(frames), 1), 2),
                        "boxes": {name: boxes[name] for name in CLASS_NAMES}}
        print(f"{split}: {len(clips)} clips, {len(frames)} frames -> {n_tiles} tiles ({n_empty} empty)       ")
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names: {json.dumps(dict(enumerate(CLASS_NAMES)))}\n", encoding="utf-8")
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))
    visualize(out, rng)
    return stats


def visualize(out: Path, rng: random.Random) -> None:
    """Draw boxes on N_VIS random non-empty training tiles for a visual check."""
    import cv2

    labels = [p for p in sorted((out / "labels" / "train").glob("*.txt")) if p.stat().st_size]
    (out / "vis").mkdir(exist_ok=True)
    for label in rng.sample(labels, min(N_VIS, len(labels))):
        image = cv2.imread(str(out / "images" / "train" / f"{label.stem}.jpg"))
        height, width = image.shape[:2]
        for cls, x1, y1, x2, y2 in yolo_to_xyxy(_read_yolo_rows(label, True), width, height):
            color = [(0, 255, 0), (255, 128, 0), (0, 0, 255), (255, 0, 255),
                     (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255)][int(cls)]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            cv2.putText(image, CLASS_NAMES[int(cls)], (int(x1), max(int(y1) - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.imwrite(str(out / "vis" / f"{label.stem}.jpg"), image)
    print(f"Saved {min(N_VIS, len(labels))} tile visualizations to {out / 'vis'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild existing tiles")
    args = parser.parse_args()
    cfg = load_config(args.config, args.debug)
    seed_everything(cfg["seed"])
    prepare(cfg, args.force)


if __name__ == "__main__":
    main()

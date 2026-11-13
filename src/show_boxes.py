"""Draw the pipeline's predicted boxes on chosen frames: one panel per model (+ ground truth on val).

Reads <work_dir>/preds/<model>_<split>.parquet (src.infer_sahi / src.ensemble_wbf). Frames are picked by
image_id (val: COCO id, test: e.g. clip_001_0000001) or by the end of their path (clip/frame.jpg).

    python -m src.show_boxes --split val --images 1 120 uav0000086_00000_v/0000050.jpg
    python -m src.show_boxes --split test --models wbf yolo_hr --images clip_001_0000001 --conf 0.3
    python -m src.show_boxes --split val --n 4          # first 4 predicted frames
Writes <work_dir>/show_boxes/<split>_<image_id>.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_utils import CLASS_NAMES
from src.pipeline_common import DEFAULT_CONFIG, list_frames, load_config, load_predictions, preds_path


def pick_frames(frames, wanted: list[str], n: int):
    if not wanted:
        return frames.head(n)
    norm = frames["path"].str.replace("\\", "/", regex=False)
    picked = frames[frames["image_id"].isin(wanted) | norm.apply(lambda p: any(p.endswith(w) for w in wanted))]
    missing = [w for w in wanted if w not in set(picked["image_id"]) and not any(norm.str.endswith(w))]
    if missing:
        print(f"Not in this split's inference frames (val uses every infer.val_frame_step-th frame): {missing}")
    return picked


def draw(ax, image, boxes, title: str, conf: float) -> None:
    """boxes: rows (cls, score, x1, y1, x2, y2); score None = ground truth."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    colors = plt.get_cmap("tab10")
    ax.imshow(image)
    shown = 0
    for cls, score, x1, y1, x2, y2 in boxes:
        if score is not None and score < conf:
            continue
        shown += 1
        color = colors(int(cls) % 10)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1))
        label = CLASS_NAMES[int(cls)] + ("" if score is None else f" {score:.2f}")
        ax.text(x1, y1 - 2, label, color=color, fontsize=5)
    ax.set_title(f"{title}: {shown} boxes" + ("" if title == "ground truth" else f" (score >= {conf})"), fontsize=9)
    ax.set_axis_off()


def show(cfg: dict, split: str, models: list[str], wanted: list[str], n: int, conf: float) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from PIL import Image

    preds = {}
    for model in models:
        path = preds_path(cfg, model, split)
        if path.is_file():
            preds[model] = load_predictions(path)[0].groupby("image_id")
        else:
            print(f"skip {model}: no {path}")
    if not preds:
        raise FileNotFoundError("No prediction files; run src.infer_sahi / src.ensemble_wbf first")
    coco = None
    if split == "val":
        from src.evaluate import load_val_coco

        coco = load_val_coco(cfg)
    out = cfg["paths"]["work_dir"] / "show_boxes"
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for frame in pick_frames(list_frames(cfg, split), wanted, n).itertuples(index=False):
        image = Image.open(frame.path).convert("RGB")
        panels = len(preds) + (coco is not None)
        cols = min(panels, 2)
        rows_n = -(-panels // cols)
        fig, axes = plt.subplots(rows_n, cols, figsize=(10 * cols, 6 * rows_n), squeeze=False)
        axes = list(axes.flat)
        for ax in axes[panels:]:  # odd panel count: blank last cell
            ax.set_axis_off()
        if coco is not None:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[int(frame.image_id)]))
            gt = [(a["category_id"] - 1, None, a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2],
                   a["bbox"][1] + a["bbox"][3]) for a in anns if not a.get("iscrowd")]
            ax = axes.pop(0)
            draw(ax, image, gt, "ground truth", conf)
            for a in anns:
                if a.get("iscrowd"):  # ignored regions: neither TP nor FP in the metric
                    x, y, w, h = a["bbox"]
                    ax.add_patch(Rectangle((x, y), w, h, facecolor="gray", alpha=0.3, hatch="////"))
        for ax, (model, groups) in zip(axes, preds.items()):
            df = groups.get_group(frame.image_id) if frame.image_id in groups.groups else None
            rows = [] if df is None else df[["cls", "score", "x1", "y1", "x2", "y2"]].itertuples(index=False)
            draw(ax, image, rows, model, conf)
        fig.legend(handles=[Patch(color=plt.get_cmap("tab10")(i), label=c) for i, c in enumerate(CLASS_NAMES)],
                   loc="lower center", ncol=len(CLASS_NAMES), fontsize=8)
        fig.suptitle(f"{split} {frame.image_id}  ({Path(frame.path).parent.name}/{Path(frame.path).name})")
        path = out / f"{split}_{frame.image_id}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"saved {path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--models", nargs="+", default=["yolo", "frcnn", "wbf", "yolo_hr"],
                        help="Prediction files to draw; missing ones are skipped")
    parser.add_argument("--images", nargs="*", default=[], help="image_ids or path endings (clip/frame.jpg)")
    parser.add_argument("--n", type=int, default=3, help="Without --images: first N frames")
    parser.add_argument("--conf", type=float, default=0.25, help="Hide predictions below this score")
    args = parser.parse_args()
    show(load_config(args.config, args.debug), args.split, args.models, args.images, args.n, args.conf)


if __name__ == "__main__":
    main()

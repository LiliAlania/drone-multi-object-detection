"""Query-based detector: RT-DETR (Ultralytics) on VisDrone.

A transformer decoder turns a fixed set of object queries into boxes; Hungarian matching
assigns one query per object in training, so there are no anchors and no NMS inside the model.
EDA: at most 180 objects per frame (and fewer per 640 px tile), so RT-DETR's 300 queries suffice.

Trains on the same tiles and uses the same inference and COCO evaluation as our tiled YOLO
pipeline (src.tiled_yolo), so the comparison isolates the detector itself. NMS is only used to
merge boxes across optional tile passes (--tile), not inside the model.

Usage (Colab, from the repository root):
    python -m src.tiled_yolo prepare --data-root /content/visdrone-subset --out /content/visdrone-tiles
    python -m src.rtdetr train --data-root /content/visdrone-tiles --epochs 20 \
        --project /content/drive/MyDrive/runs/rtdetr   # after a disconnect: same command + --resume
    python -m src.rtdetr evaluate --weights .../rtdetr/weights/best.pt --split-root /content/data/val --step 5
    python -m src.rtdetr predict --weights .../rtdetr/weights/best.pt --test-root /content/data \
        --out submission_rtdetr.csv
"""

from src.tiled_yolo import main as _main


def main() -> None:
    # batch 8: the transformer decoder needs more GPU memory than YOLO at the same tile size
    _main(arch="rtdetr", model="rtdetr-l.pt", name="rtdetr", batch=8, description=__doc__)


if __name__ == "__main__":
    main()

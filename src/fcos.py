"""One-stage anchor-free detector: torchvision FCOS (ResNet-50 FPN) on VisDrone.

Each FPN location directly predicts class scores, centerness and distances to the 4 box
sides; no anchor boxes are matched. Training, evaluation and the submission CSV are shared
with src.faster_rcnn (same data, labels and checkpoint format), so the results compare directly.

Usage (Colab, from the repository root):
    python -m src.fcos train --data-root /content/visdrone-subset --epochs 12
    python -m src.fcos evaluate --weights runs/fcos/last.pt --split-root /content/data/val --step 5
    python -m src.fcos predict --weights runs/fcos/last.pt \
        --test-root /content/data --out submission_fcos.csv
"""

from __future__ import annotations

import math

from src.data_utils import CLASS_NAMES
from src.faster_rcnn import main as _main
from src.yolo_baseline import MAX_DETECTIONS

# Index 0 is unused so labels match src.faster_rcnn (1..8); FCOS uses sigmoid scores, no background class.
NUM_CLASSES = len(CLASS_NAMES) + 1
PRIOR_PROBABILITY = 0.01  # focal-loss init from the paper: start with every location ~background


def build_model(min_size: int, max_size: int, pretrained: bool = True):
    import torch
    import torchvision

    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        min_size=min_size,
        max_size=max_size,
        score_thresh=0.01,  # low threshold: mAP rewards recall
        detections_per_img=MAX_DETECTIONS,
        topk_candidates=2000,  # per FPN level before NMS; dense scenes have hundreds of objects
    )
    # keep the pretrained classification tower, replace only its last conv (91 COCO classes -> ours)
    head = model.head.classification_head
    head.cls_logits = torch.nn.Conv2d(head.cls_logits.in_channels, NUM_CLASSES, 3, padding=1)
    torch.nn.init.normal_(head.cls_logits.weight, std=0.01)
    torch.nn.init.constant_(head.cls_logits.bias, -math.log((1 - PRIOR_PROBABILITY) / PRIOR_PROBABILITY))
    head.num_classes = NUM_CLASSES
    return model


def main() -> None:
    # lower LR than Faster R-CNN: torchvision's FCOS recipe is 0.01 at batch 16, and there is no warm-up here
    _main(build_model, run_name="fcos", submission="submission_fcos.csv", lr=0.005, description=__doc__)


if __name__ == "__main__":
    main()

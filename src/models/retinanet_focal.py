
"""RetinaNet with small anchors and modified Focal Loss."""

import types
import torch

from torchvision.ops import sigmoid_focal_loss

from src.models.retinanet_small import build_model as build_small_anchors
from src.models.faster_rcnn import main as detection_main


FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 2.0


def custom_compute_loss(self, targets, head_outputs, matched_idxs):
    losses = []
    cls_logits = head_outputs["cls_logits"]

    for target, logits, matched in zip(
        targets, cls_logits, matched_idxs
    ):
        foreground = matched >= 0
        num_foreground = foreground.sum()

        # Target class for each anchor
        gt_classes = torch.zeros_like(logits)
        gt_classes[
            foreground,
            target["labels"][matched[foreground]],
        ] = 1.0

        # Exclude anchors with ambiguous assignment
        valid = matched != self.BETWEEN_THRESHOLDS

        loss = sigmoid_focal_loss(
            logits[valid],
            gt_classes[valid],
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            reduction="sum",
        )

        losses.append(loss / num_foreground.clamp(min=1))

    return torch.stack(losses).mean()


def build_model(min_size=640, max_size=1280, pretrained=True):
    model = build_small_anchors(
        min_size=min_size,
        max_size=max_size,
        pretrained=pretrained,
    )

    head = model.head.classification_head

    head.compute_loss = types.MethodType(
        custom_compute_loss,
        head,
    )

    return model


if __name__ == "__main__":
    detection_main(
        build_fn=build_model,
        run_name="retinanet_focal",
        submission="submission_retinanet_focal.csv",
        lr=0.001,
        description="RetinaNet Small Anchors with Focal Loss alpha=0.5",
    )

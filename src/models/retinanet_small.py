
from torchvision.models.detection.anchor_utils import AnchorGenerator

from src.models.retinanet import build_model as build_baseline
from src.models.faster_rcnn import main as detection_main


def build_model(min_size=640, max_size=1280, pretrained=True):
    """
    RetinaNet with smaller anchors for detecting
    small objects in VisDrone images.
    """

    model = build_baseline(
        min_size=min_size,
        max_size=max_size,
        pretrained=pretrained,
    )

    model.anchor_generator = AnchorGenerator(
        sizes=(
            (16, 20, 25),
            (32, 40, 50),
            (64, 80, 101),
            (128, 161, 203),
            (256, 322, 406),
        ),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    return model


def main():
    detection_main(
        build_fn=build_model,
        run_name="retinanet_small",
        submission="submission_retinanet_small.csv",
        lr=0.001,
        description="RetinaNet with smaller anchors for VisDrone",
    )


if __name__ == "__main__":
    main()

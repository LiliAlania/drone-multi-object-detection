
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from src.data_utils import CLASS_NAMES
from src.faster_rcnn import YoloDetectionDataset, _collate


def build_model(min_size=640, max_size=1280, pretrained=True):
    model = retinanet_resnet50_fpn_v2(
        weights="DEFAULT" if pretrained else None,
        min_size=min_size,
        max_size=max_size,
        detections_per_img=300,
    )

    num_classes = len(CLASS_NAMES) + 1
    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels,
        num_anchors,
        num_classes,
    )

    return model

def main():
    from src.faster_rcnn import main as detection_main

    detection_main(
        build_fn=build_model,
        run_name="retinanet",
        submission="submission_retinanet.csv",
        lr=0.005,
        description="RetinaNet detector for VisDrone",
    )


if __name__ == "__main__":
    main()

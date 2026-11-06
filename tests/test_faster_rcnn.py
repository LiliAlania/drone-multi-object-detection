import importlib.util
import unittest

from src.faster_rcnn import yolo_rows_to_target


class FasterRcnnTests(unittest.TestCase):
    def test_yolo_rows_to_pixel_xyxy_with_shifted_labels(self):
        boxes, labels = yolo_rows_to_target(
            [(2, 0.5, 0.5, 0.1, 0.2), (0, 0.0, 0.0, 0.1, 0.1), (7, 0.5, 0.5, 0.0001, 0.1)], 1000, 500
        )
        # second box is clipped at the image edge, third is too thin and dropped
        self.assertEqual(boxes, [[450.0, 200.0, 550.0, 300.0], [0.0, 0.0, 50.0, 25.0]])
        self.assertEqual(labels, [3, 1])

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision not installed")
    def test_model_has_small_anchors_and_nine_classes(self):
        from src.faster_rcnn import build_model

        model = build_model(min_size=256, max_size=448, pretrained=False)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 9)
        self.assertEqual(model.rpn.anchor_generator.sizes[0], (8,))


if __name__ == "__main__":
    unittest.main()

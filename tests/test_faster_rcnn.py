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

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision not installed")
    def test_two_stage_options_reach_the_model(self):
        from torchvision.models.detection import roi_heads

        from src.faster_rcnn import _giou_fastrcnn_loss, build_model

        model = build_model(256, 448, pretrained=False, rpn_proposals=500, roi_batch=128,
                            roi_pos_fraction=0.5, box_loss="giou")
        self.assertEqual(model.rpn._post_nms_top_n["training"], 500)
        self.assertEqual(model.roi_heads.fg_bg_sampler.batch_size_per_image, 128)
        self.assertEqual(model.roi_heads.fg_bg_sampler.positive_fraction, 0.5)
        self.assertIs(roi_heads.fastrcnn_loss, _giou_fastrcnn_loss)
        build_model(256, 448, pretrained=False)  # default restores torchvision's smooth L1 loss
        self.assertIsNot(roi_heads.fastrcnn_loss, _giou_fastrcnn_loss)

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision not installed")
    def test_giou_loss_is_zero_for_exact_boxes_and_positive_otherwise(self):
        import torch

        from src.faster_rcnn import _giou_fastrcnn_loss

        labels = [torch.tensor([2, 0])]  # one foreground RoI (class 2), one background
        targets = [torch.tensor([[1.0, -2.0, 0.5, 0.3], [0.0, 0.0, 0.0, 0.0]])]
        logits = torch.zeros(2, 9)
        exact = torch.zeros(2, 9, 4)
        exact[0, 2] = targets[0][0]
        _, loss = _giou_fastrcnn_loss(logits, exact.reshape(2, -1), labels, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        _, loss = _giou_fastrcnn_loss(logits, torch.zeros(2, 36), labels, targets)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

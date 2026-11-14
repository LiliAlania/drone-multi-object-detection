import importlib.util
import unittest

from src.frcnn_explain import size_bucket


class FrcnnExplainTests(unittest.TestCase):
    def test_size_buckets_follow_coco(self):
        self.assertEqual([size_bucket(a) for a in (31 ** 2, 32 ** 2, 96 ** 2)], ["small", "medium", "large"])

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision not installed")
    def test_match_labels_every_kind_of_detection(self):
        import torch

        from src.frcnn_explain import match

        gt = torch.tensor([[0, 0, 10, 10], [100, 100, 120, 120]], dtype=torch.float32)
        gt_labels = torch.tensor([1, 3])
        crowd = torch.tensor([[200, 200, 300, 300]], dtype=torch.float32)
        dets = torch.tensor([
            [0, 0, 10, 10],      # TP for GT 0
            [0, 0, 10, 9],       # same class, GT 0 already taken -> duplicate
            [100, 100, 120, 120],  # class 1 on a class-3 object -> wrong class
            [0, 0, 10, 30],      # IoU 1/3 with GT 0 -> poor box
            [210, 210, 220, 220],  # inside the ignored region
            [500, 500, 510, 510],  # nothing near -> background
        ], dtype=torch.float32)
        gt_match, status = match(dets, torch.tensor([1, 1, 1, 1, 1, 1]), gt, gt_labels, crowd)
        self.assertEqual(gt_match, [0, -1])
        self.assertEqual(status, ["TP", "FP: duplicate", "FP: wrong class", "FP: poor box", "ignored",
                                  "FP: background"])

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision not installed")
    def test_wandb_boxes_use_one_based_labels(self):
        from src.models.faster_rcnn import wandb_boxes

        boxes = wandb_boxes([[1, 2, 3, 4]], [3], [0.5])
        self.assertEqual(boxes["box_data"][0]["box_caption"], "car 0.50")
        self.assertEqual(boxes["box_data"][0]["position"], {"minX": 1, "maxX": 3, "minY": 2, "maxY": 4})
        self.assertEqual(boxes["class_labels"][1], "person")


if __name__ == "__main__":
    unittest.main()

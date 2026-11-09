import unittest

import numpy as np

from src.compare_gt import in_regions, match_frame


class CompareGtTests(unittest.TestCase):
    def test_every_error_type(self):
        gt = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110], [200, 200, 210, 210.0]])
        gt_classes = np.array([0, 2, 2, 1])
        preds = np.array([
            [0, 0, 10, 10],        # tp (person)
            [1, 0, 11, 10],        # duplicate of the same person
            [50, 50, 60, 60],      # wrong class: van on a car
            [100, 100, 116, 116],  # car, IoU 0.39 -> localization
            [300, 300, 310, 310.0],  # background
        ])
        types, linked, matched = match_frame(gt, gt_classes, preds, np.array([0, 0, 3, 2, 2]),
                                             np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
        self.assertEqual(types, ["tp", "duplicate", "wrong_class", "localization", "background"])
        self.assertEqual(linked, [0, 0, 1, 2, -1])
        self.assertEqual(matched.tolist(), [True, False, False, False])  # 3 objects missed

    def test_ignored_region_by_center(self):
        boxes = np.array([[0, 0, 4, 4], [8, 8, 30, 30.0]])
        self.assertEqual(in_regions(boxes, np.array([[0, 0, 10, 10.0]])).tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()

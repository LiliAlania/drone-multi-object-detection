import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.temporal import (add_temporal_args, box_iou, read_predictions, temporal_rescore, warp_boxes,
                          write_predictions)


def defaults(**overrides):
    parser = argparse.ArgumentParser()
    add_temporal_args(parser)
    args = parser.parse_args([])
    vars(args).update(overrides)
    return args


def frame(image_id, number, clip="c1"):
    return {"image_id": image_id, "clip": clip, "frame": str(number), "file_name": f"{image_id}.jpg"}


def dets(*rows):
    rows = np.array(rows, dtype=float).reshape(-1, 6)  # cls, score, x1, y1, x2, y2
    return rows[:, 0].astype(int), rows[:, 1], rows[:, 2:]


class TemporalTests(unittest.TestCase):
    def test_iou_and_warp(self):
        boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
        self.assertAlmostEqual(box_iou(boxes, np.array([[5.0, 0.0, 15.0, 10.0]]))[0, 0], 1 / 3)
        shifted = warp_boxes(boxes, np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]]))
        np.testing.assert_allclose(shifted, [[3.0, -2.0, 13.0, 8.0]])

    def test_rescore_boost_propagate_and_clip_reset(self):
        box = (10, 10, 30, 30)
        frames = [frame("a", 1), frame("b", 2), frame("c", 3), frame("d", 1, clip="c2")]
        predictions = {
            "a": dets((2, 0.9, *box)),
            "b": dets((2, 0.2, 11, 10, 31, 30), (0, 0.4, 100, 100, 120, 120)),  # weak hit + lone person
            "c": dets(),  # missed frame
            "d": dets(),  # new clip: no tracks carried over
        }
        out = temporal_rescore(frames, predictions, defaults(weight=0.5))
        self.assertAlmostEqual(out["b"][1][0], 0.5 * 0.2 + 0.5 * 0.9)  # boosted by the confident track
        self.assertAlmostEqual(out["b"][1][1], 0.4)  # no history: unchanged
        classes, scores, boxes = out["c"]
        self.assertEqual(classes.tolist(), [2])  # propagated car box
        np.testing.assert_allclose(boxes[0], [11, 10, 31, 30])
        self.assertAlmostEqual(scores[0], (0.7 * 0.9 + 0.3 * 0.2) * 0.8)
        self.assertEqual(len(out["d"][0]), 0)

    def test_csv_round_trip(self):
        predictions = {"x": dets((1, 0.5, 1, 2, 4, 6)), "y": dets()}
        with tempfile.TemporaryDirectory() as folder:
            path = write_predictions(Path(folder) / "p.csv", predictions, ["x", "y"])
            loaded = read_predictions(path)
        np.testing.assert_allclose(loaded["x"][2], [[1, 2, 4, 6]])
        self.assertEqual(len(loaded["y"][0]), 0)


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest

import pandas as pd

from src.make_submission import COLUMNS, build_rows, validate
from src.pipeline_common import PRED_COLUMNS, load_config

FRAMES = pd.DataFrame([{"image_id": "a", "width": 100, "height": 50}, {"image_id": "b", "width": 100, "height": 50}])


class PipelineTests(unittest.TestCase):
    def test_debug_config_uses_separate_dirs_and_short_runs(self):
        cfg = load_config(debug=True)
        self.assertEqual(cfg["paths"]["work_dir"].name, "debug")
        self.assertEqual(cfg["frcnn"]["epochs"], cfg["debug"]["epochs"])

    def test_submission_clips_caps_and_validates(self):
        preds = pd.DataFrame([("a", 2, 0.9, -5, 10, 120, 40), ("a", 1, 0.001, 0, 0, 10, 10),
                              ("a", 0, 0.5, 60, 60, 70, 70)], columns=PRED_COLUMNS)
        rows = build_rows(preds, FRAMES, final_confidence=0.01)
        self.assertEqual(rows[0], ["a", "2 0.9000 0.0 10.0 100.0 30.0"])  # clipped; low score and off-frame dropped
        self.assertEqual(rows[1], ["b", ""])  # every frame present
        validate(COLUMNS, rows, FRAMES)

    def test_validation_rejects_bad_rows(self):
        for rows in ([["b", ""], ["a", ""]],                       # wrong order
                     [["a", "8 0.5 1 1 5 5"], ["b", ""]],          # class out of range
                     [["a", "1 0.5 90 1 20 5"], ["b", ""]],        # box past the right edge
                     [["a", "1 0.5 1 1 5"], ["b", ""]]):           # incomplete group
            with self.assertRaises(ValueError):
                validate(COLUMNS, rows, FRAMES)

    def test_hr_frame_step_is_the_smallest_reaching_min_epochs(self):
        from unittest import mock

        from src import train_yolo_hr

        y = {"frame_steps": [2, 3, 4, 5], "min_epochs": 30, "imgsz": 1280}
        with mock.patch.object(train_yolo_hr, "train_frames", lambda cfg, step: [None] * (24000 // step)), \
                mock.patch("builtins.print"):
            # 1 img/s, no val, 10 h: step 2 -> 3 epochs ... step 5 -> 7.5: none reach 30 -> the largest step
            self.assertEqual(train_yolo_hr.choose_step({}, y, 1.0, 0.0, 10.0), 5)
            # 8 img/s: step 2 -> 24 epochs < 30, step 3 -> 36
            self.assertEqual(train_yolo_hr.choose_step({}, y, 8.0, 0.0, 10.0), 3)

    @unittest.skipUnless(importlib.util.find_spec("ensemble_boxes"), "ensemble_boxes not installed")
    def test_wbf_fuses_per_class_in_pixel_coordinates(self):
        from src.ensemble_wbf import fuse

        yolo = pd.DataFrame([("a", 0, 0.8, 10, 10, 30, 40), ("a", 7, 0.7, 10, 10, 30, 40)], columns=PRED_COLUMNS)
        frcnn = pd.DataFrame([("a", 0, 0.6, 12, 10, 32, 40)], columns=PRED_COLUMNS)
        fused = fuse([yolo, frcnn], {"a": (100, 50)}, [1, 1], 0.55, 0.0).sort_values("cls")
        self.assertEqual(fused["cls"].tolist(), [0, 7])  # rider pair: person and motor are not merged
        person = fused.iloc[0]
        self.assertAlmostEqual(person["x1"], (10 * 0.8 + 12 * 0.6) / 1.4, places=3)  # score-weighted, back in px
        self.assertAlmostEqual(person["y2"], 40, places=3)


if __name__ == "__main__":
    unittest.main()

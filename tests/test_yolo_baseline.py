import tempfile
import unittest
from pathlib import Path

from src.models.yolo_baseline import format_predictions, write_dataset_yaml


class YoloBaselineTests(unittest.TestCase):
    def test_format_predictions_uses_top_left_pixel_xywh(self):
        line = format_predictions([2.0, 0.0], [0.91, 0.55], [[1366, 580, 1392, 624], [809, 625, 838, 691]])
        self.assertEqual(line, "2 0.9100 1366.0 580.0 26.0 44.0 0 0.5500 809.0 625.0 29.0 66.0")
        self.assertEqual(format_predictions([], [], []), "")

    def test_dataset_yaml_handles_both_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for layout in (("train", "images"), ("val", "images")):
                (root / "full" / Path(*layout)).mkdir(parents=True)
            for layout in (("images", "train"), ("images", "val")):
                (root / "subset" / Path(*layout)).mkdir(parents=True)
            full = write_dataset_yaml(root / "full", root / "full.yaml").read_text()
            subset = write_dataset_yaml(root / "subset", root / "subset.yaml").read_text()
            self.assertIn("full/train/images", full)
            self.assertIn("subset/images/val", subset)
            self.assertIn("  7: motor", subset)
            with self.assertRaises(FileNotFoundError):
                write_dataset_yaml(root, root / "bad.yaml")


if __name__ == "__main__":
    unittest.main()

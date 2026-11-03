import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src import data_utils as du
from src.make_subset import build_subset


class DataUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data_root = self.root / "data"
        self.train_root = self.data_root / "train"
        self.val_root = self.data_root / "val"
        self.train_root.mkdir(parents=True)
        self.val_root.mkdir(parents=True)
        self.train_coco = self._make_split(self.train_root, "train", 5)
        self.val_coco = self._make_split(self.val_root, "val", 1)

    def tearDown(self):
        self.temp.cleanup()

    def _make_split(self, split_root, split, clip_count):
        images = []
        annotations = []
        annotation_id = 1
        image_id = 1
        for clip_number in range(clip_count):
            clip = f"{split}_clip_{clip_number}"
            for frame_id in (1, 2):
                relative = Path("images") / clip / f"{frame_id:07d}.jpg"
                image_path = split_root / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"synthetic-image")
                images.append(
                    {
                        "id": image_id,
                        "file_name": relative.as_posix(),
                        "width": 100,
                        "height": 80,
                        "video": clip,
                        "frame_id": frame_id,
                    }
                )
                labels = split_root / "labels" / clip / f"{frame_id:07d}.txt"
                labels.parent.mkdir(parents=True, exist_ok=True)
                labels.write_text("0 0.1 0.2 0.1 0.25\n", encoding="utf-8")
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [5, 6, 10, 20],
                        "iscrowd": 0,
                        "track_id": 10,
                        "orig_category_id": 1,
                        "occlusion": 0,
                        "truncation": 0,
                    }
                )
                annotation_id += 1
                if frame_id == 1:
                    ignore_path = split_root / "ignore" / clip / f"{frame_id:07d}.txt"
                    ignore_path.parent.mkdir(parents=True, exist_ok=True)
                    ignore_path.write_text("0.1 0.2 0.05 0.075\n", encoding="utf-8")
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 0,
                            "bbox": [7, 10, 5, 6],
                            "iscrowd": 1,
                            "track_id": None,
                            "orig_category_id": 0,
                            "occlusion": 0,
                            "truncation": 0,
                        }
                    )
                    annotation_id += 1
                image_id += 1
        coco_path = split_root / "annotations_coco.json"
        coco_path.write_text(
            json.dumps(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": [{"id": 1, "name": "person"}],
                }
            ),
            encoding="utf-8",
        )
        return coco_path

    def test_coco_loading_and_yolo_consistency(self):
        images, annotations = du.load_coco(self.train_coco, "train")
        targets, ignored = du.split_annotations(annotations)
        self.assertEqual(len(images), 10)
        self.assertEqual(len(targets), 10)
        self.assertEqual(len(ignored), 5)
        self.assertEqual(
            du.check_yolo_consistency(
                self.train_coco, self.train_root, "train", sample_size=10
            )["target_boxes_compared"],
            10,
        )
        self.assertEqual(
            du.coco_xywh_to_yolo([5, 6, 10, 20], 100, 80),
            (0.1, 0.2, 0.1, 0.25),
        )

    def test_statistics_geometry_and_temporal_metrics(self):
        images, annotations = du.load_coco(self.train_coco, "train")
        targets, ignored = du.split_annotations(annotations)
        sizes = du.add_box_dimensions(targets)
        self.assertTrue(np.allclose(sizes["sqrt_area"], np.sqrt(200)))
        buckets = du.size_bucket_summary(targets)
        self.assertEqual(
            buckets.loc[buckets["class_name"].eq("all"), "count"].sum(), len(targets)
        )
        anchors = du.kmeans_anchors(targets, images, imgsz=100, k=2, seed=7)
        self.assertEqual(len(anchors["anchors"]), 2)
        self.assertAlmostEqual(anchors["mean_best_iou"], 1.0)
        temporal = du.temporal_redundancy(images, targets, [1, 2])
        self.assertEqual(temporal.loc[0, "mean_track_iou"], 1.0)
        self.assertEqual(temporal.loc[0, "tracks_never_sampled_pct"], 0)
        _, stats = du.ignored_region_summary(images, targets, ignored)
        self.assertEqual(stats.loc[0, "frames_with_ignored"], 5)

    def test_namespace_group_folds_and_subset_archive(self):
        train_images, train_annotations = du.load_coco(self.train_coco, "train")
        val_images, val_annotations = du.load_coco(self.val_coco, "val")
        images, annotations = du.namespace_image_ids(
            [train_images, val_images], [train_annotations, val_annotations]
        )
        self.assertEqual(images["id"].nunique(), len(images))
        targets, _ = du.split_annotations(annotations)
        folds = du.make_group_folds(train_images, train_annotations, 5, seed=42)
        self.assertEqual(len(folds), 5)
        self.assertFalse(
            set(folds[0]["train_clips"]) & set(folds[0]["val_clips"])
        )
        archive = build_subset(
            self.data_root, self.root / "subset", step=2, splits=("train",)
        )
        self.assertTrue(archive.exists())
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
        self.assertTrue(any(name.endswith("data.yaml") for name in names))
        self.assertEqual(len(list((self.root / "subset" / "images").rglob("*.jpg"))), 5)
        self.assertTrue((self.root / "subset" / "ignore").exists())


if __name__ == "__main__":
    unittest.main()

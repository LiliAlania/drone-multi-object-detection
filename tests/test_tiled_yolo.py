import unittest

from src.tiled_yolo import crop_targets, repeat_factors, tile_windows, yolo_to_xyxy


class TiledYoloTests(unittest.TestCase):
    def test_windows_cover_image_and_align_to_edges(self):
        windows = tile_windows(1920, 1080, 640, 0.2)
        self.assertEqual(len(windows), 8)  # x starts 0,512,1024,1280; y starts 0,440
        self.assertEqual(windows[-1], (1280, 440, 1920, 1080))
        self.assertEqual(tile_windows(680, 382, 640, 0.2), [(0, 0, 640, 382), (40, 0, 680, 382)])
        self.assertEqual(tile_windows(500, 300, 640, 0.2), [(0, 0, 500, 300)])

    def test_crop_keeps_visible_boxes_and_drops_ignored(self):
        targets = [
            [0, 100, 100, 120, 140],   # inside -> kept
            [1, 630, 10, 660, 30],     # 1/3 visible -> dropped
            [2, 600, 50, 650, 70],     # 80% visible -> kept, clipped at x=640
            [3, 300, 300, 310, 310],   # centre inside an ignored region -> dropped
        ]
        rows = crop_targets(targets, [[290, 290, 330, 330]], (0, 0, 640, 640))
        self.assertEqual([r[0] for r in rows], [0, 2])
        self.assertAlmostEqual(rows[1][3], 40 / 640)
        shifted = crop_targets([[5, 700, 100, 740, 140]], [], (640, 0, 1280, 640))
        self.assertAlmostEqual(shifted[0][1], 80 / 640)

    def test_yolo_to_xyxy_with_and_without_class(self):
        self.assertEqual(yolo_to_xyxy([(4, 0.5, 0.5, 0.2, 0.4)], 100, 50), [[4, 40.0, 15.0, 60.0, 35.0]])
        self.assertEqual(yolo_to_xyxy([(0.5, 0.5, 0.2, 0.4)], 100, 50), [[40.0, 15.0, 60.0, 35.0]])

    def test_rare_class_tiles_repeat_more(self):
        tiles = [{0}] * 95 + [{0, 5}] * 5
        factors = repeat_factors(tiles, threshold=0.1)
        self.assertEqual(factors[0], 1.0)  # class 0 is in every tile
        self.assertAlmostEqual(factors[-1], 2 ** 0.5)  # class 5 in 5% of tiles: sqrt(0.1 / 0.05)
        self.assertEqual(repeat_factors([set(), {1}], 0.1), [1.0, 1.0])



if __name__ == "__main__":
    unittest.main()

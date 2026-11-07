import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision not installed")
class FcosTests(unittest.TestCase):
    def test_trains_and_predicts_with_our_labels(self):
        import torch

        from src.fcos import build_model

        model = build_model(min_size=128, max_size=128, pretrained=False)
        self.assertEqual(model.head.classification_head.cls_logits.out_channels, 9)
        image = torch.rand(3, 128, 128)
        target = {"boxes": torch.tensor([[10.0, 10.0, 40.0, 30.0]]), "labels": torch.tensor([8])}
        losses = model([image], [target])
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        model.eval()
        with torch.no_grad():
            labels = model([image])[0]["labels"]
        self.assertTrue(((labels >= 1) & (labels <= 8)).all())


if __name__ == "__main__":
    unittest.main()

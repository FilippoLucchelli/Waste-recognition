import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is not installed in the light test environment")
class TorchComponentTests(unittest.TestCase):
    def test_mask_resize_preserves_integer_labels(self) -> None:
        from waste_recognition.transforms import JointTransform

        image = torch.zeros((4, 3, 5))
        mask = torch.tensor([[0, 0, 1, 1, 1], [0, 2, 2, 1, 1], [2, 2, 2, 1, 1]])
        _, resized_mask = JointTransform(8)(image, mask)
        self.assertEqual(set(resized_mask.unique().tolist()), {0, 1, 2})

    def test_confusion_matrix_metrics_are_dataset_level(self) -> None:
        from waste_recognition.metrics import SegmentationMetrics

        logits = torch.tensor(
            [
                [
                    [[10.0, 0.0], [10.0, 0.0]],
                    [[0.0, 10.0], [0.0, 10.0]],
                ]
            ]
        )
        target = torch.tensor([[[0, 1], [1, 1]]])
        metrics = SegmentationMetrics(2, ("background", "trash"))
        metrics.update(logits, target)
        result = metrics.compute()
        self.assertAlmostEqual(result["background_iou"], 0.5)
        self.assertAlmostEqual(result["trash_iou"], 2 / 3)

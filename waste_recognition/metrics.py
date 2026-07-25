"""Dataset-level semantic-segmentation metrics."""

from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, n_classes: int, class_names: tuple[str, ...]) -> None:
        if n_classes <= 0 or len(class_names) != n_classes:
            raise ValueError("class_names must contain exactly n_classes values")
        self.n_classes = n_classes
        self.class_names = class_names
        self.confusion = torch.zeros((n_classes, n_classes), dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.argmax(dim=1)
        valid = (target >= 0) & (target < self.n_classes)
        encoded = self.n_classes * target[valid].to(torch.int64) + prediction[valid].to(torch.int64)
        batch_confusion = torch.bincount(
            encoded.detach().cpu(),
            minlength=self.n_classes**2,
        ).reshape(self.n_classes, self.n_classes)
        self.confusion += batch_confusion

    def compute(self) -> dict[str, float]:
        true_positive = self.confusion.diag()
        ground_truth = self.confusion.sum(dim=1)
        predicted = self.confusion.sum(dim=0)
        union = ground_truth + predicted - true_positive
        dice_denominator = ground_truth + predicted

        iou = torch.where(union > 0, true_positive / union, torch.nan)
        dice = torch.where(
            dice_denominator > 0,
            2 * true_positive / dice_denominator,
            torch.nan,
        )
        result = {
            "miou": float(torch.nanmean(iou)),
            "mdice": float(torch.nanmean(dice)),
        }
        for index, name in enumerate(self.class_names):
            result[f"{name}_iou"] = float(iou[index])
            result[f"{name}_dice"] = float(dice[index])
        return result

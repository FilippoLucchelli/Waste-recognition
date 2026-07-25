"""Compatibility wrapper around dataset-level metrics."""

from __future__ import annotations

import torch

from waste_recognition.metrics import SegmentationMetrics


class Metrics:
    def __init__(self, output, ground_truth) -> None:
        self.output = output
        self.ground_truth = ground_truth

    def get_metrics(self, opt):
        class_names = tuple(opt.classes.keys() if isinstance(opt.classes, dict) else opt.classes)
        accumulator = SegmentationMetrics(opt.n_classes, class_names)
        accumulator.update(self.output, self.ground_truth)
        requested = set(getattr(opt, "metrics", ("iou", "dice")))
        return {
            name: torch.tensor(value)
            for name, value in accumulator.compute().items()
            if ("iou" in name and "iou" in requested) or ("dice" in name and "dice" in requested)
        }

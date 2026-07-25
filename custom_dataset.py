"""Compatibility adapters for the historical dataset classes."""

from __future__ import annotations

from waste_recognition.config import DatasetLayout, resolve_channels
from waste_recognition.dataset import NpySegmentationDataset
from waste_recognition.errors import ConfigurationError
from waste_recognition.manifest import build_manifest
from waste_recognition.statistics import channel_mean_std
from waste_recognition.transforms import JointTransform


class CustomDataset:
    """The removed k-fold dataset now fails with an actionable message."""

    def __init__(self, *args, **kwargs) -> None:
        raise ConfigurationError(
            "CustomDataset k-fold mode was incomplete and has been removed; "
            "use explicit train/val/test folders with CustomDatasetYaml"
        )


class CustomDatasetYaml(NpySegmentationDataset):
    def __init__(self, opt, phase="train", transforms=None) -> None:
        layout = DatasetLayout.from_yaml(opt.data_dir)
        channels = resolve_channels(
            getattr(opt, "channels", ()),
            getattr(opt, "no_rgb", False),
        )
        ground_truth = getattr(opt, "ground_truth", phase != "test")
        records = build_manifest(
            layout.require_split(phase),
            channels,
            require_masks=ground_truth,
        )
        mean = getattr(opt, "mean", None)
        std = getattr(opt, "std", None)
        if mean is None or std is None:
            mean, std = channel_mean_std(records)
        transform = (
            transforms
            if isinstance(transforms, JointTransform)
            else JointTransform(getattr(opt, "size", 640))
        )
        super().__init__(
            records,
            transform=transform,
            mean=tuple(mean),
            std=tuple(std),
            n_classes=2 if getattr(opt, "single_class", False) else getattr(opt, "n_classes", 6),
            single_class=getattr(opt, "single_class", False),
        )
        self._mean = tuple(mean)
        self._std = tuple(std)

    def get_mean_std(self):
        return list(self._mean), list(self._std)

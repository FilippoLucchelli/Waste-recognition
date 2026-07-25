"""PyTorch dataset backed by a validated sample manifest."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .errors import DatasetValidationError
from .manifest import SampleRecord
from .transforms import JointTransform


class NpySegmentationDataset(Dataset):
    def __init__(
        self,
        records: tuple[SampleRecord, ...],
        *,
        transform: JointTransform,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        n_classes: int,
        single_class: bool = False,
        trash_label: int = 3,
    ) -> None:
        if not records:
            raise DatasetValidationError("Dataset manifest is empty")
        n_channels = len(records[0].channels)
        if len(mean) != n_channels or len(std) != n_channels:
            raise DatasetValidationError("Normalization statistics do not match channels")
        self.records = records
        self.transform = transform
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]
        self.n_classes = n_classes
        self.single_class = single_class
        self.trash_label = trash_label

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        channel_arrays = [np.asarray(np.load(path), dtype=np.float32) for path in record.channels]
        shapes = {array.shape for array in channel_arrays}
        if len(shapes) != 1 or any(array.ndim != 2 for array in channel_arrays):
            raise DatasetValidationError(
                f"Sample {record.sample_id!r} has invalid channel shapes: {shapes}"
            )
        image = torch.from_numpy(np.stack(channel_arrays))

        mask = None
        if record.mask is not None:
            mask_array = np.asarray(np.load(record.mask))
            if mask_array.shape != channel_arrays[0].shape:
                raise DatasetValidationError(
                    f"Mask shape differs from channels for sample {record.sample_id!r}"
                )
            if (
                not np.isfinite(mask_array).all()
                or not np.equal(
                    mask_array,
                    np.rint(mask_array),
                ).all()
            ):
                raise DatasetValidationError(
                    f"Mask for sample {record.sample_id!r} contains non-integer labels"
                )
            mask = torch.from_numpy(mask_array.astype(np.int64, copy=False))
            if self.single_class:
                mask = (mask == self.trash_label).long()

        image, mask = self.transform(image, mask)
        image = (image - self.mean) / self.std

        if mask is None:
            return image
        if mask.numel() and (mask.min() < 0 or mask.max() >= self.n_classes):
            raise DatasetValidationError(
                f"Mask for sample {record.sample_id!r} contains labels outside "
                f"[0, {self.n_classes - 1}]"
            )
        return image, mask

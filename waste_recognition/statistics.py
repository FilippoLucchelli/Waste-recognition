"""Dataset channel statistics calculated directly from NPY manifests."""

from __future__ import annotations

import numpy as np

from .errors import DatasetValidationError
from .manifest import SampleRecord


def channel_mean_std(
    records: tuple[SampleRecord, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not records:
        raise DatasetValidationError("Cannot calculate statistics for an empty manifest")
    n_channels = len(records[0].channels)
    sums = np.zeros(n_channels, dtype=np.float64)
    squared_sums = np.zeros(n_channels, dtype=np.float64)
    counts = np.zeros(n_channels, dtype=np.int64)

    for record in records:
        if len(record.channels) != n_channels:
            raise DatasetValidationError("Manifest records have inconsistent channel counts")
        expected_shape = None
        for index, path in enumerate(record.channels):
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2:
                raise DatasetValidationError(
                    f"Expected a 2D channel array at {path}, received {array.shape}"
                )
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise DatasetValidationError(
                    f"Sample {record.sample_id!r} contains inconsistent shapes"
                )
            values = np.asarray(array, dtype=np.float64)
            if not np.isfinite(values).all():
                raise DatasetValidationError(f"Channel contains NaN or infinity: {path}")
            sums[index] += values.sum()
            squared_sums[index] += np.square(values).sum()
            counts[index] += values.size

    means = sums / counts
    variances = np.maximum((squared_sums / counts) - np.square(means), 0.0)
    standard_deviations = np.sqrt(variances)
    standard_deviations[standard_deviations == 0] = 1.0
    return tuple(means.tolist()), tuple(standard_deviations.tolist())

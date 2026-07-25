"""Deterministic matching of per-band NPY files into samples."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .errors import DatasetValidationError


@dataclass(frozen=True, slots=True)
class SampleRecord:
    sample_id: str
    channels: tuple[Path, ...]
    mask: Path | None


def _sample_id(path: Path, folder_name: str) -> str:
    suffix = f"_{folder_name.lower()}"
    stem = path.stem
    return stem[: -len(suffix)] if stem.lower().endswith(suffix) else stem


def _index_folder(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise DatasetValidationError(f"Required band folder does not exist: {folder}")
    indexed: dict[str, Path] = {}
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() != ".npy":
            continue
        sample_id = _sample_id(path, folder.name)
        if sample_id in indexed:
            raise DatasetValidationError(f"Duplicate sample ID {sample_id!r} in folder {folder}")
        indexed[sample_id] = path
    if not indexed:
        raise DatasetValidationError(f"No NPY files found in required folder: {folder}")
    return indexed


def build_manifest(
    split_folder: str | Path,
    channel_names: Iterable[str],
    *,
    require_masks: bool,
) -> tuple[SampleRecord, ...]:
    """Build records only when every band has exactly the same sample IDs."""
    split_folder = Path(split_folder)
    channel_names = tuple(channel_names)
    if not channel_names:
        raise DatasetValidationError("At least one channel is required")

    indices = {
        name: _index_folder(split_folder / name)
        for name in (*channel_names, *(("masks",) if require_masks else ()))
    }
    reference_name = channel_names[0]
    expected = set(indices[reference_name])

    mismatches = []
    for name, index in indices.items():
        current = set(index)
        missing = sorted(expected - current)
        extra = sorted(current - expected)
        if missing or extra:
            mismatches.append(
                f"{name}: missing={missing[:5] or 'none'}, extra={extra[:5] or 'none'}"
            )
    if mismatches:
        raise DatasetValidationError(
            "Band folders do not contain matching sample IDs; " + "; ".join(mismatches)
        )

    return tuple(
        SampleRecord(
            sample_id=sample_id,
            channels=tuple(indices[name][sample_id] for name in channel_names),
            mask=indices["masks"][sample_id] if require_masks else None,
        )
        for sample_id in sorted(expected)
    )


def write_manifest(records: tuple[SampleRecord, ...], path: str | Path) -> Path:
    """Persist the exact sample-to-file mapping used by a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        n_channels = len(records[0].channels) if records else 0
        writer.writerow(["sample_id", *(f"channel_{index}" for index in range(n_channels)), "mask"])
        for record in records:
            writer.writerow(
                [
                    record.sample_id,
                    *(str(channel) for channel in record.channels),
                    str(record.mask) if record.mask is not None else "",
                ]
            )
    return path

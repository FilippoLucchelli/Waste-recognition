import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from waste_recognition.errors import DatasetValidationError
from waste_recognition.manifest import build_manifest, write_manifest
from waste_recognition.statistics import channel_mean_std


class ManifestTests(unittest.TestCase):
    @staticmethod
    def _create_split(root: Path, sample_ids=("img_000", "img_001")) -> None:
        for band_index, band in enumerate(("red", "green", "blue", "nir", "masks")):
            folder = root / band
            folder.mkdir()
            for sample_index, sample_id in enumerate(sample_ids):
                value = 0 if band == "masks" else band_index + sample_index
                np.save(
                    folder / f"{sample_id}_{band}.npy",
                    np.full((2, 3), value, dtype=np.float32),
                )

    def test_manifest_matches_band_suffix_names_by_sample_id(self) -> None:
        with TemporaryDirectory() as temporary:
            split = Path(temporary)
            self._create_split(split)
            records = build_manifest(
                split,
                ("red", "green", "blue", "nir"),
                require_masks=True,
            )
            self.assertEqual([record.sample_id for record in records], ["img_000", "img_001"])
            self.assertEqual(records[0].channels[0].name, "img_000_red.npy")
            self.assertEqual(records[0].mask.name, "img_000_masks.npy")
            written = write_manifest(records, split / "manifest.csv")
            self.assertIn("img_000", written.read_text(encoding="utf-8"))

    def test_manifest_reports_missing_band_samples(self) -> None:
        with TemporaryDirectory() as temporary:
            split = Path(temporary)
            self._create_split(split)
            (split / "nir" / "img_001_nir.npy").unlink()
            with self.assertRaisesRegex(DatasetValidationError, "matching sample IDs"):
                build_manifest(
                    split,
                    ("red", "green", "blue", "nir"),
                    require_masks=True,
                )

    def test_statistics_are_pixel_weighted_and_handle_constant_channels(self) -> None:
        with TemporaryDirectory() as temporary:
            split = Path(temporary)
            self._create_split(split)
            records = build_manifest(
                split,
                ("red", "green", "blue", "nir"),
                require_masks=True,
            )
            mean, std = channel_mean_std(records)
            self.assertEqual(mean, (0.5, 1.5, 2.5, 3.5))
            self.assertEqual(std, (0.5, 0.5, 0.5, 0.5))

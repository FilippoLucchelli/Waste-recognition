import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from waste_recognition.config import (
    DatasetLayout,
    RunMetadata,
    load_run_metadata,
    resolve_channels,
    resolve_classes,
)
from waste_recognition.errors import ConfigurationError


class ConfigurationTests(unittest.TestCase):
    def test_layout_resolves_relative_paths_from_yaml_and_root(self) -> None:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            root = folder / "dataset"
            (root / "train").mkdir(parents=True)
            yaml_path = folder / "data.yaml"
            yaml_path.write_text(
                "root_dir: dataset\ntrain_dir: train\nval_dir: val\ntest_dir: test\n",
                encoding="utf-8",
            )
            layout = DatasetLayout.from_yaml(yaml_path)
            self.assertEqual(layout.root, root.resolve())
            self.assertEqual(layout.require_split("train"), (root / "train").resolve())

    def test_channel_and_class_resolution(self) -> None:
        self.assertEqual(
            resolve_channels(["nir"], no_rgb=False),
            ("red", "green", "blue", "nir"),
        )
        self.assertEqual(resolve_classes(None, 6, False)[3], "trash")
        self.assertEqual(resolve_classes(None, 6, True), ("background", "trash"))

    def test_acnet_rejects_an_incompatible_channel_layout(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "ACNet requires"):
            RunMetadata(
                model="acnet",
                channels=("red", "green", "blue", "rededge"),
                class_names=("background", "trash"),
                size=32,
                mean=(0.0,) * 4,
                std=(1.0,) * 4,
            )

    def test_run_metadata_round_trip_uses_typed_json(self) -> None:
        with TemporaryDirectory() as temporary:
            metadata = RunMetadata(
                model="msnet",
                channels=("red", "green", "blue", "nir"),
                class_names=("background", "trash"),
                size=32,
                mean=(0.1, 0.2, 0.3, 0.4),
                std=(1.0, 1.0, 1.0, 1.0),
            )
            path = metadata.save(temporary)
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsInstance(raw["channels"], list)
            self.assertEqual(RunMetadata.load(temporary), metadata)

    def test_historical_parameters_csv_is_migrated(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "parameters.csv"
            path.write_text(
                "channels,data_dir,model,size,n_classes,classes,mean,std,no_rgb\n"
                "\"['nir']\",data.yaml,msnet,32,2,"
                "\"{'background': 0, 'trash': 1}\","
                '"[0.1, 0.2, 0.3, 0.4]",'
                '"[1.0, 1.0, 1.0, 1.0]",False\n',
                encoding="utf-8",
            )
            metadata = load_run_metadata(temporary)
            self.assertEqual(metadata.channels, ("red", "green", "blue", "nir"))
            self.assertEqual(metadata.class_names, ("background", "trash"))

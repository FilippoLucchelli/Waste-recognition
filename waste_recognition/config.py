"""Validated dataset and run configuration."""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError

RGB_CHANNELS = ("red", "green", "blue")
DEFAULT_CLASSES = ("grass", "obstacle", "road", "trash", "vegetation", "sky")
SUPPORTED_MODELS = ("msnet", "acnet", "unet", "unet++", "deeplabv3", "deeplabv3+")


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    config_path: Path
    root: Path
    train: Path | None
    validation: Path | None
    test: Path | None

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetLayout:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise ConfigurationError(f"Dataset configuration does not exist: {config_path}")
        with config_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}

        root_value = raw.get("root_dir")
        if not root_value:
            raise ConfigurationError("data.yaml must define a non-empty root_dir")
        root = Path(root_value).expanduser()
        if not root.is_absolute():
            root = config_path.parent / root
        root = root.resolve()

        def split(name: str) -> Path | None:
            value = raw.get(name)
            if not value:
                return None
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else root / candidate).resolve()

        return cls(
            config_path=config_path,
            root=root,
            train=split("train_dir"),
            validation=split("val_dir"),
            test=split("test_dir"),
        )

    def require_split(self, phase: str) -> Path:
        mapping = {
            "train": self.train,
            "val": self.validation,
            "validation": self.validation,
            "test": self.test,
        }
        if phase not in mapping:
            raise ConfigurationError(f"Unknown dataset phase: {phase}")
        path = mapping[phase]
        if path is None:
            raise ConfigurationError(f"Dataset configuration does not define {phase!r}")
        if not path.is_dir():
            raise ConfigurationError(f"Dataset split does not exist: {path}")
        return path

    def result_folder(self, name: str | Path) -> Path:
        path = Path(name)
        return path.resolve() if path.is_absolute() else (self.root / "results" / path).resolve()


@dataclass(frozen=True, slots=True)
class RunMetadata:
    model: str
    channels: tuple[str, ...]
    class_names: tuple[str, ...]
    size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    single_class: bool = False
    encoder_pretrained: bool = False

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ConfigurationError(f"Unsupported model: {self.model}")
        if not self.channels:
            raise ConfigurationError("At least one input channel is required")
        if len(set(self.channels)) != len(self.channels):
            raise ConfigurationError(f"Input channels contain duplicates: {self.channels}")
        if not self.class_names:
            raise ConfigurationError("At least one class is required")
        if self.size <= 0:
            raise ConfigurationError("Input size must be positive")
        if len(self.mean) != len(self.channels) or len(self.std) != len(self.channels):
            raise ConfigurationError("Mean and standard deviation must match input channels")
        if not all(math.isfinite(value) for value in (*self.mean, *self.std)):
            raise ConfigurationError("Normalization statistics must be finite")
        if any(value <= 0 for value in self.std):
            raise ConfigurationError("All standard deviations must be positive")
        validate_model_channels(self.model, self.channels)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def save(self, folder: str | Path) -> Path:
        path = Path(folder) / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, indent=2)
        return path

    @classmethod
    def load(cls, folder: str | Path) -> RunMetadata:
        path = Path(folder) / "config.json"
        if not path.is_file():
            raise ConfigurationError(f"Run configuration does not exist: {path}")
        with path.open(encoding="utf-8") as stream:
            raw: dict[str, Any] = json.load(stream)
        for name in ("channels", "class_names", "mean", "std"):
            raw[name] = tuple(raw[name])
        return cls(**raw)


def resolve_channels(extra_channels: list[str] | tuple[str, ...], no_rgb: bool) -> tuple[str, ...]:
    extras = tuple(channel.strip().lower() for channel in extra_channels if channel.strip())
    channels = (() if no_rgb else RGB_CHANNELS) + extras
    if not channels:
        raise ConfigurationError("No channels selected: enable RGB or provide --channels")
    if "masks" in channels:
        raise ConfigurationError("'masks' is reserved and cannot be an input channel")
    return channels


def resolve_classes(
    requested: list[str] | tuple[str, ...] | None,
    n_classes: int,
    single_class: bool,
) -> tuple[str, ...]:
    if single_class:
        return ("background", "trash")
    if requested:
        classes = tuple(requested)
        if len(classes) != n_classes:
            raise ConfigurationError(
                f"Received {len(classes)} class names but n_classes is {n_classes}"
            )
        return classes
    if not 1 <= n_classes <= len(DEFAULT_CLASSES):
        raise ConfigurationError(
            f"Default class names are only available for 1-{len(DEFAULT_CLASSES)} classes"
        )
    return DEFAULT_CLASSES[:n_classes]


def validate_model_channels(model: str, channels: tuple[str, ...]) -> None:
    if model == "acnet" and channels != (*RGB_CHANNELS, "nir"):
        raise ConfigurationError("ACNet requires channels: red green blue nir")
    if model == "msnet" and (channels[:3] != RGB_CHANNELS or len(channels) < 4):
        raise ConfigurationError("MSNet requires RGB followed by at least one extra channel")


def load_run_metadata(folder: str | Path) -> RunMetadata:
    """Load current JSON metadata or migrate a historical parameters.csv."""
    folder = Path(folder)
    if (folder / "config.json").is_file():
        return RunMetadata.load(folder)

    legacy_path = folder / "parameters.csv"
    if not legacy_path.is_file():
        raise ConfigurationError(f"Neither config.json nor parameters.csv exists in {folder}")
    with legacy_path.open(encoding="utf-8") as stream:
        try:
            raw = next(csv.DictReader(stream))
        except StopIteration as exc:
            raise ConfigurationError(f"Legacy configuration is empty: {legacy_path}") from exc

    try:
        extra_channels = tuple(ast.literal_eval(raw["channels"]))
        no_rgb = bool(ast.literal_eval(raw["no_rgb"]))
        classes_value = ast.literal_eval(raw["classes"])
        class_names = tuple(
            classes_value.keys() if isinstance(classes_value, dict) else classes_value
        )
        mean = tuple(float(value) for value in ast.literal_eval(raw["mean"]))
        std = tuple(float(value) for value in ast.literal_eval(raw["std"]))
        size = int(raw["size"])
    except (KeyError, SyntaxError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Malformed historical configuration: {legacy_path}") from exc

    return RunMetadata(
        model=raw["model"],
        channels=resolve_channels(extra_channels, no_rgb),
        class_names=class_names,
        size=size,
        mean=mean,
        std=std,
        single_class=len(class_names) == 2 and "trash" in class_names,
        encoder_pretrained=False,
    )

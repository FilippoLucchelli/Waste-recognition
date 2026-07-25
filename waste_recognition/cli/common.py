"""Shared CLI arguments and runtime helpers."""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch

from ..config import SUPPORTED_MODELS
from ..errors import ConfigurationError

LOGGER = logging.getLogger(__name__)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_dir", required=True, help="Path to data.yaml")
    parser.add_argument("--model_dir", help="Result folder containing a checkpoint")
    parser.add_argument("--model", default="msnet", choices=SUPPORTED_MODELS)
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--metrics", nargs="+", default=["iou", "dice"])
    parser.add_argument("--single_class", action="store_true")
    parser.add_argument("--no_rgb", action="store_true")
    parser.add_argument("--channels", nargs="*", default=[])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_metrics(
    path: str | Path,
    *,
    epoch: int,
    split: str,
    loss: float,
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    row = {"epoch": epoch, "split": split, "loss": loss, **metrics}
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

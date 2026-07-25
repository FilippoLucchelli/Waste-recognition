"""Evaluation and prediction command."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..checkpoints import load_checkpoint
from ..config import DatasetLayout, load_run_metadata
from ..dataset import NpySegmentationDataset
from ..errors import WasteRecognitionError
from ..factories import create_model
from ..manifest import build_manifest
from ..transforms import JointTransform
from .common import (
    add_common_arguments,
    configure_logging,
    resolve_device,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a waste segmentation model")
    add_common_arguments(parser)
    parser.add_argument("--ground_truth", action="store_true")
    parser.add_argument("--print_images", action="store_true")
    parser.add_argument("--checkpoint", help="Checkpoint filename inside model_dir")
    parser.add_argument("--predictions_folder", help="Optional output folder for NPY masks")
    return parser


def _checkpoint_path(folder: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else folder / path
    for name in ("best.pt", "model.pth", "last.pt"):
        path = folder / name
        if path.is_file():
            return path
    return folder / "best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        _evaluate(args)
    except (OSError, ValueError, WasteRecognitionError) as exc:
        parser.error(str(exc))
    return 0


def _evaluate(args) -> None:
    if not args.model_dir:
        raise ValueError("--model_dir is required for evaluation")
    seed_everything(args.seed)
    layout = DatasetLayout.from_yaml(args.data_dir)
    model_folder = layout.result_folder(args.model_dir)
    metadata = load_run_metadata(model_folder)
    records = build_manifest(
        layout.require_split("test"),
        metadata.channels,
        require_masks=args.ground_truth,
    )
    dataset = NpySegmentationDataset(
        records,
        transform=JointTransform(metadata.size),
        mean=metadata.mean,
        std=metadata.std,
        n_classes=metadata.n_classes,
        single_class=metadata.single_class,
    )
    device = resolve_device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = create_model(metadata).to(device)
    load_checkpoint(
        _checkpoint_path(model_folder, args.checkpoint),
        model=model,
        device=device,
    )
    model.eval()

    if args.ground_truth:
        from ..engine import run_epoch

        result = run_epoch(
            model=model,
            loader=loader,
            criterion=torch.nn.CrossEntropyLoss(),
            device=device,
            class_names=metadata.class_names,
        )
        LOGGER.info("Test loss: %.4f, mIoU: %.4f", result.loss, result.metrics["miou"])
        print(json.dumps({"loss": result.loss, **result.metrics}, indent=2))
    else:
        prediction_folder = (
            Path(args.predictions_folder).resolve()
            if args.predictions_folder
            else model_folder / "predictions"
        )
        prediction_folder.mkdir(parents=True, exist_ok=True)
        offset = 0
        with torch.no_grad():
            for images in loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
                for prediction in predictions:
                    sample_id = records[offset].sample_id
                    np.save(prediction_folder / f"{sample_id}_mask.npy", prediction)
                    offset += 1
        LOGGER.info("Saved %d predictions to %s", offset, prediction_folder)

    if args.print_images:
        LOGGER.warning(
            "--print_images is retained for CLI compatibility but Visdom output "
            "was removed; use --predictions_folder instead"
        )

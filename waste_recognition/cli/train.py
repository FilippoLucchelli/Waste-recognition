"""Training command."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..checkpoints import load_checkpoint, save_checkpoint
from ..config import (
    DatasetLayout,
    RunMetadata,
    load_run_metadata,
    resolve_channels,
    resolve_classes,
)
from ..dataset import NpySegmentationDataset
from ..errors import WasteRecognitionError
from ..factories import create_loss, create_model, create_optimizer, create_scheduler
from ..manifest import build_manifest, write_manifest
from ..statistics import channel_mean_std
from ..transforms import JointTransform
from .common import (
    add_common_arguments,
    append_metrics,
    configure_logging,
    resolve_device,
    seed_everything,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a waste segmentation model")
    add_common_arguments(parser)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument(
        "--resume",
        "--pretrained",
        dest="resume",
        action="store_true",
        help="Resume from --model_dir (historical alias: --pretrained)",
    )
    parser.add_argument(
        "--encoder_pretrained",
        action="store_true",
        help="Initialize the encoder with ImageNet weights",
    )
    parser.add_argument("--loss", default="jaccard", choices=("jaccard", "crossentropy"))
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument(
        "--scheduler",
        default="cosine",
        choices=("none", "cosine", "step", "plateau", "triangular"),
    )
    parser.add_argument("--T_0", type=int, default=100)
    parser.add_argument("--eta_min", type=float, default=0.00001)
    parser.add_argument("--step", type=int)
    parser.add_argument("--factor", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--base_lr", type=float)
    parser.add_argument("--max_lr", type=float)
    parser.add_argument("--step_size_up", type=int)
    parser.add_argument("--step_size_down", type=int)
    parser.add_argument("--optimizer", default="sgd", choices=("sgd", "adam"))
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--momentum", default=0.8, type=float)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=["h_flip", "crop"],
        choices=("v_flip", "h_flip", "crop"),
    )
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--es_patience", type=int, default=5)
    parser.add_argument("--es_start_epoch", type=int, default=110)
    parser.add_argument("--es_min_delta", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated files in an existing save folder",
    )
    return parser


def _resume_path(folder: Path) -> Path:
    for name in ("last.pt", "best.pt", "model.pth"):
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return folder / "last.pt"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)
    try:
        _train(args)
    except (OSError, ValueError, WasteRecognitionError) as exc:
        parser.error(str(exc))
    return 0


def _train(args) -> None:
    if args.epochs <= 0 or args.batch_size <= 0 or args.es_patience <= 0 or args.num_workers < 0:
        raise ValueError(
            "epochs, batch_size and es_patience must be positive; num_workers cannot be negative"
        )
    seed_everything(args.seed)
    layout = DatasetLayout.from_yaml(args.data_dir)
    output_folder = layout.result_folder(args.save_folder)
    generated_names = (
        "config.json",
        "training_config.json",
        "metrics.csv",
        "best.pt",
        "last.pt",
        "model.pth",
        "train_manifest.csv",
        "validation_manifest.csv",
    )
    existing_generated = [
        output_folder / name for name in generated_names if (output_folder / name).exists()
    ]
    if existing_generated and not args.resume and not args.overwrite:
        raise ValueError(
            f"Save folder already contains run outputs: {output_folder}; "
            "use --overwrite or choose a new --save_folder"
        )
    if args.overwrite and not args.resume:
        for path in existing_generated:
            path.unlink()
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not args.model_dir:
            raise ValueError("--resume requires --model_dir")
        resume_folder = layout.result_folder(args.model_dir)
        metadata = load_run_metadata(resume_folder)
    else:
        channels = resolve_channels(args.channels, args.no_rgb)
        class_names = resolve_classes(args.classes, args.n_classes, args.single_class)
        train_records = build_manifest(
            layout.require_split("train"),
            channels,
            require_masks=True,
        )
        mean, std = channel_mean_std(train_records)
        metadata = RunMetadata(
            model=args.model,
            channels=channels,
            class_names=class_names,
            size=args.size,
            mean=mean,
            std=std,
            single_class=args.single_class,
            encoder_pretrained=args.encoder_pretrained,
        )

    train_records = build_manifest(
        layout.require_split("train"),
        metadata.channels,
        require_masks=True,
    )
    validation_records = build_manifest(
        layout.require_split("val"),
        metadata.channels,
        require_masks=True,
    )
    metadata.save(output_folder)
    with (output_folder / "training_config.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(vars(args), stream, indent=2)
    write_manifest(train_records, output_folder / "train_manifest.csv")
    write_manifest(
        validation_records,
        output_folder / "validation_manifest.csv",
    )

    train_transform = JointTransform(
        metadata.size,
        training=True,
        horizontal_flip_probability=(args.probability if "h_flip" in args.transforms else 0),
        vertical_flip_probability=(args.probability if "v_flip" in args.transforms else 0),
        crop_min_scale=args.scale if "crop" in args.transforms else None,
    )
    validation_transform = JointTransform(metadata.size)
    dataset_arguments = {
        "mean": metadata.mean,
        "std": metadata.std,
        "n_classes": metadata.n_classes,
        "single_class": metadata.single_class,
    }
    train_dataset = NpySegmentationDataset(
        train_records,
        transform=train_transform,
        **dataset_arguments,
    )
    validation_dataset = NpySegmentationDataset(
        validation_records,
        transform=validation_transform,
        **dataset_arguments,
    )

    device = resolve_device(args.device)
    loader_arguments = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_arguments)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_arguments)

    model = create_model(metadata).to(device)
    criterion = create_loss(args.loss)
    optimizer = create_optimizer(
        args.optimizer,
        model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = create_scheduler(args.scheduler, optimizer, args)
    start_epoch = 0
    best_metric = float("-inf")
    if args.resume:
        state = load_checkpoint(
            _resume_path(resume_folder),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state["best_metric"])

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    no_improvement_epochs = 0
    from ..engine import run_epoch

    for epoch in range(start_epoch, args.epochs):
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler if scaler.is_enabled() else None,
            device=device,
            class_names=metadata.class_names,
            use_amp=scaler.is_enabled(),
        )
        validation_result = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            class_names=metadata.class_names,
        )
        append_metrics(
            output_folder / "metrics.csv",
            epoch=epoch,
            split="train",
            loss=train_result.loss,
            metrics=train_result.metrics,
        )
        append_metrics(
            output_folder / "metrics.csv",
            epoch=epoch,
            split="validation",
            loss=validation_result.loss,
            metrics=validation_result.metrics,
        )

        current_metric = validation_result.metrics["miou"]
        improved = current_metric > best_metric + args.es_min_delta
        if improved:
            best_metric = current_metric
            no_improvement_epochs = 0
        elif epoch >= args.es_start_epoch:
            no_improvement_epochs += 1

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(validation_result.loss)
            else:
                scheduler.step()

        if improved:
            save_checkpoint(
                output_folder / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
            )
            torch.save(model.state_dict(), output_folder / "model.pth")
        save_checkpoint(
            output_folder / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
        )

        LOGGER.info(
            "Epoch %d/%d - train loss %.4f - val loss %.4f - mIoU %.4f",
            epoch + 1,
            args.epochs,
            train_result.loss,
            validation_result.loss,
            current_metric,
        )

        if no_improvement_epochs >= args.es_patience:
            LOGGER.info("Early stopping at epoch %d", epoch + 1)
            break

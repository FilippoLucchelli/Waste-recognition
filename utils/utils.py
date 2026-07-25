"""Compatibility façade for historical utility imports."""

from __future__ import annotations

import ast
import csv
from pathlib import Path

import torch

from waste_recognition.checkpoints import load_checkpoint
from waste_recognition.config import (
    RunMetadata,
    resolve_channels,
    resolve_classes,
)
from waste_recognition.factories import (
    create_loss,
    create_model,
    create_optimizer,
    create_scheduler,
)


def list_and_sort_paths(folder):
    return [
        str(path)
        for path in sorted(Path(folder).iterdir(), key=lambda item: item.name.casefold())
        if path.is_file()
    ]


def _class_names(opt) -> tuple[str, ...]:
    classes = getattr(opt, "classes", None)
    if isinstance(classes, dict):
        return tuple(classes)
    return resolve_classes(classes, opt.n_classes, getattr(opt, "single_class", False))


def _run_metadata(opt) -> RunMetadata:
    channels = resolve_channels(
        getattr(opt, "channels", ()),
        getattr(opt, "no_rgb", False),
    )
    mean = tuple(getattr(opt, "mean", None) or [0.0] * len(channels))
    std = tuple(getattr(opt, "std", None) or [1.0] * len(channels))
    return RunMetadata(
        model=opt.model,
        channels=channels,
        class_names=_class_names(opt),
        size=opt.size,
        mean=mean,
        std=std,
        single_class=getattr(opt, "single_class", False),
        encoder_pretrained=getattr(opt, "encoder_pretrained", False),
    )


def load_model(opt):
    model = create_model(_run_metadata(opt))
    if getattr(opt, "pretrained", False) or not getattr(opt, "isTrain", True):
        model_folder = Path(opt.model_dir)
        checkpoint = next(
            (
                model_folder / name
                for name in ("best.pt", "model.pth", "last.pt")
                if (model_folder / name).is_file()
            ),
            model_folder / "model.pth",
        )
        load_checkpoint(
            checkpoint,
            model=model,
            device=torch.device("cpu"),
        )
    return model


def load_scheduler(opt, optimizer):
    return create_scheduler(opt.scheduler, optimizer, opt)


def load_optimizer(opt, model):
    return create_optimizer(
        opt.optimizer,
        model,
        learning_rate=opt.lr,
        weight_decay=opt.weight_decay,
        momentum=opt.momentum,
    )


def load_loss(opt):
    return create_loss(opt.loss)


def default_classes(opt):
    names = resolve_classes(None, opt.n_classes, getattr(opt, "single_class", False))
    return {name: index for index, name in enumerate(names)}


def metric_names(opt):
    names = []
    requested = set(opt.metrics)
    if "iou" in requested:
        names.append("miou")
    if "dice" in requested:
        names.append("mdice")
    for name in _class_names(opt):
        if "iou" in requested:
            names.append(f"{name}_iou")
        if "dice" in requested:
            names.append(f"{name}_dice")
    return names


def init_files(opt):
    folder = Path(opt.save_folder)
    folder.mkdir(parents=True, exist_ok=True)
    parameters = folder / "parameters.csv"
    values = {
        "channels": opt.channels,
        "data_dir": opt.data_dir,
        "model": opt.model,
        "size": opt.size,
        "n_classes": opt.n_classes,
        "classes": opt.classes,
        "mean": opt.mean,
        "std": opt.std,
        "no_rgb": opt.no_rgb,
    }
    with parameters.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, values)
        writer.writeheader()
        writer.writerow(values)
    metric_file = folder / "metrics.csv"
    train_metric_file = folder / "train_metrics.csv"
    for path in (metric_file, train_metric_file):
        with path.open("w", encoding="utf-8") as stream:
            stream.write(",".join(metric_names(opt)) + "\n")
    return str(parameters), str(metric_file), str(train_metric_file)


def save_metrics(opt, valid_metrics, train_metrics):
    for path, values in (
        (opt.metric_file, valid_metrics),
        (opt.train_metric_file, train_metrics),
    ):
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(",".join(f"{float(value) * 100:.2f}" for value in values) + "\n")


def save_model(opt, model, model_name):
    torch.save(model.state_dict(), Path(opt.save_folder) / f"{model_name}.pth")


def read_csv(opt):
    with (Path(opt.model_dir) / "parameters.csv").open(encoding="utf-8") as stream:
        return next(csv.DictReader(stream))


def get_colormap(opt):
    from matplotlib.colors import ListedColormap

    colors = ["#00ff00", "#0033cc", "#a6a6a6", "#ffff00", "#18761c", "#18ffff"]
    return ListedColormap(colors[: opt.n_classes])


def get_folds(opt):
    raise NotImplementedError("The incomplete k-fold mode has been removed")


def create_bash(opt):
    # Retained as a harmless compatibility hook. Reproducibility now uses config.json.
    return None


def parse_legacy_value(value):
    """Parse values stored by historical parameters.csv files."""
    return ast.literal_eval(value)

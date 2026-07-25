"""Compatibility helpers for evaluation."""

from __future__ import annotations

import ast

import torch

from waste_recognition.config import load_run_metadata
from waste_recognition.engine import run_epoch


def test(opt, model, test_loader, test_loader_print, device):
    if not opt.ground_truth:
        return None
    class_names = tuple(opt.classes.keys() if isinstance(opt.classes, dict) else opt.classes)
    result = run_epoch(
        model=model,
        loader=test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        class_names=class_names,
    )
    return {name: torch.tensor(value) for name, value in result.metrics.items()}


def get_test_options(opt):
    try:
        metadata = load_run_metadata(opt.model_dir)
    except Exception:
        from .utils import read_csv

        data = read_csv(opt)
        opt.channels = ast.literal_eval(data["channels"])
        opt.model = data["model"]
        opt.size = int(data["size"])
        opt.n_classes = int(data["n_classes"])
        opt.classes = ast.literal_eval(data["classes"])
        opt.mean = ast.literal_eval(data["mean"])
        opt.std = ast.literal_eval(data["std"])
        opt.no_rgb = ast.literal_eval(data["no_rgb"])
        return

    opt.channels = list(metadata.channels[3:])
    opt.model = metadata.model
    opt.size = metadata.size
    opt.n_classes = metadata.n_classes
    opt.classes = list(metadata.class_names)
    opt.mean = list(metadata.mean)
    opt.std = list(metadata.std)
    opt.no_rgb = metadata.channels[:3] != ("red", "green", "blue")


def print_metrics(vis, metrics):
    text = "<br>".join(f"{name}: {float(value):.4f}" for name, value in metrics.items())
    return vis.text(text)


def print_images_gt(*args, **kwargs):
    raise NotImplementedError("Visdom image rendering was replaced by saved NPY predictions")


def print_images_no_gt(*args, **kwargs):
    raise NotImplementedError("Visdom image rendering was replaced by saved NPY predictions")

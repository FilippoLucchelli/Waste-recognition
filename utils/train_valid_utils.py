"""Compatibility helpers for historical imports."""

from __future__ import annotations

import torch

from waste_recognition.config import load_run_metadata
from waste_recognition.engine import run_epoch
from waste_recognition.transforms import JointTransform


def _class_names(opt) -> tuple[str, ...]:
    return tuple(opt.classes.keys() if isinstance(opt.classes, dict) else opt.classes)


def _legacy_result(result):
    metrics = {name: torch.tensor(value) for name, value in result.metrics.items()}
    return metrics, torch.tensor(result.loss)


def train_epoch(model, trainloader, optimizer, criterion, device, opt):
    result = run_epoch(
        model=model,
        loader=trainloader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        class_names=_class_names(opt),
    )
    return _legacy_result(result)


def valid_epoch(model, validloader, criterion, device, opt):
    result = run_epoch(
        model=model,
        loader=validloader,
        criterion=criterion,
        device=device,
        class_names=_class_names(opt),
    )
    return _legacy_result(result)


def get_transforms(opt):
    selected = set(getattr(opt, "transforms", ()))
    return JointTransform(
        opt.size,
        training=True,
        horizontal_flip_probability=opt.probability if "h_flip" in selected else 0,
        vertical_flip_probability=opt.probability if "v_flip" in selected else 0,
        crop_min_scale=opt.scale if "crop" in selected else None,
    )


def get_pretrained_options(opt):
    metadata = load_run_metadata(opt.model_dir)
    opt.channels = list(metadata.channels[3:])
    opt.model = metadata.model
    opt.size = metadata.size
    opt.n_classes = metadata.n_classes
    opt.no_rgb = metadata.channels[:3] != ("red", "green", "blue")
    opt.mean = list(metadata.mean)
    opt.std = list(metadata.std)


class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_iou = float("-inf")

    def early_stop(self, iou):
        value = float(iou)
        if value > self.max_iou + self.min_delta:
            self.max_iou = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

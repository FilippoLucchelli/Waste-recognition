"""Factories for models, losses, optimizers and schedulers."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from .config import RunMetadata
from .errors import ConfigurationError


def create_model(metadata: RunMetadata) -> nn.Module:
    n_channels = len(metadata.channels)
    if metadata.model == "msnet":
        from models.MSNet import MSNet

        return MSNet(
            num_classes=metadata.n_classes,
            n_channels=n_channels,
            pretrained=metadata.encoder_pretrained,
        )
    if metadata.model == "acnet":
        from models.ACNet import ACNet

        return ACNet(
            num_class=metadata.n_classes,
            pretrained=metadata.encoder_pretrained,
        )

    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ConfigurationError("segmentation-models-pytorch is required for this model") from exc

    weights = "imagenet" if metadata.encoder_pretrained else None
    constructors = {
        "unet": (smp.Unet, "resnet50"),
        "unet++": (smp.UnetPlusPlus, "resnet34"),
        "deeplabv3": (smp.DeepLabV3, "resnet34"),
        "deeplabv3+": (smp.DeepLabV3Plus, "resnet34"),
    }
    constructor, encoder = constructors[metadata.model]
    return constructor(
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=n_channels,
        classes=metadata.n_classes,
    )


def create_loss(name: str) -> nn.Module:
    if name == "crossentropy":
        return nn.CrossEntropyLoss()
    if name == "jaccard":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ConfigurationError(
                "segmentation-models-pytorch is required for Jaccard loss"
            ) from exc
        return smp.losses.JaccardLoss(mode="multiclass")
    raise ConfigurationError(f"Unsupported loss: {name}")


def create_optimizer(
    name: str,
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
) -> Optimizer:
    parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    if name == "adam":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ConfigurationError(f"Unsupported optimizer: {name}")


def create_scheduler(name: str, optimizer: Optimizer, options):
    if name == "none":
        return None
    if name == "cosine":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=options.T_0,
            T_mult=2,
            eta_min=options.eta_min,
        )
    if name == "step":
        if options.step is None or options.factor is None:
            raise ConfigurationError("Step scheduler requires --step and --factor")
        return lr_scheduler.StepLR(
            optimizer,
            step_size=options.step,
            gamma=options.factor,
        )
    if name == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=options.factor if options.factor is not None else 0.1,
            patience=options.patience if options.patience is not None else 10,
        )
    if name == "triangular":
        required = (options.base_lr, options.max_lr, options.step_size_up)
        if any(value is None for value in required):
            raise ConfigurationError(
                "Triangular scheduler requires --base_lr, --max_lr and --step_size_up"
            )
        return lr_scheduler.CyclicLR(
            optimizer,
            base_lr=options.base_lr,
            max_lr=options.max_lr,
            step_size_up=options.step_size_up,
            step_size_down=options.step_size_down,
            mode="triangular2",
        )
    raise ConfigurationError(f"Unsupported scheduler: {name}")

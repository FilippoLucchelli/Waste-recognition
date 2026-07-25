"""Small, deterministic training and evaluation loops."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from .metrics import SegmentationMetrics


@dataclass(frozen=True, slots=True)
class EpochResult:
    loss: float
    metrics: dict[str, float]
    samples: int


def run_epoch(
    *,
    model,
    loader,
    criterion,
    device: torch.device,
    class_names: tuple[str, ...],
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    accumulator = SegmentationMetrics(len(class_names), class_names)
    total_loss = 0.0
    total_samples = 0

    gradient_context = nullcontext() if training else torch.no_grad()
    with gradient_context:
        for image, mask in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            batch_size = image.shape[0]
            if training:
                optimizer.zero_grad(set_to_none=True)

            amp_context = (
                torch.autocast(device_type=device.type, enabled=True) if use_amp else nullcontext()
            )
            with amp_context:
                logits = model(image)
                loss = criterion(logits, mask)

            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size
            accumulator.update(logits.detach(), mask)

    if total_samples == 0:
        raise RuntimeError("DataLoader produced no samples")
    return EpochResult(
        loss=total_loss / total_samples,
        metrics=accumulator.compute(),
        samples=total_samples,
    )

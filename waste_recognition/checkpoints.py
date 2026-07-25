"""Versioned checkpoint persistence with legacy state-dict support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .errors import CheckpointError


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model,
    device: torch.device,
    optimizer=None,
    scheduler=None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise CheckpointError(f"Unsupported checkpoint payload: {path}")

    if "model" in payload:
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        return payload

    # Historical model.pth files contain only the state dict.
    model.load_state_dict(payload)
    return {"format_version": 0, "epoch": -1, "best_metric": float("-inf")}

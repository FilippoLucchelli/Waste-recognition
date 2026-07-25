"""Joint spatial transforms for images and segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True, slots=True)
class JointTransform:
    size: int
    training: bool = False
    horizontal_flip_probability: float = 0.0
    vertical_flip_probability: float = 0.0
    crop_min_scale: float | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("size must be positive")
        for probability in (
            self.horizontal_flip_probability,
            self.vertical_flip_probability,
        ):
            if not 0 <= probability <= 1:
                raise ValueError("flip probabilities must be in [0, 1]")
        if self.crop_min_scale is not None and not 0 < self.crop_min_scale <= 1:
            raise ValueError("crop_min_scale must be in (0, 1]")

    def __call__(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.training:
            if torch.rand(()) < self.horizontal_flip_probability:
                image = torch.flip(image, dims=(-1,))
                mask = torch.flip(mask, dims=(-1,)) if mask is not None else None
            if torch.rand(()) < self.vertical_flip_probability:
                image = torch.flip(image, dims=(-2,))
                mask = torch.flip(mask, dims=(-2,)) if mask is not None else None
            if self.crop_min_scale is not None:
                image, mask = self._random_square_crop(image, mask)
        return self._resize(image, mask)

    def _random_square_crop(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        height, width = image.shape[-2:]
        scale = float(torch.empty(()).uniform_(self.crop_min_scale or 1.0, 1.0).sqrt())
        side = max(1, round(min(height, width) * scale))
        top = int(torch.randint(0, height - side + 1, ()).item())
        left = int(torch.randint(0, width - side + 1, ()).item())
        image = image[..., top : top + side, left : left + side]
        if mask is not None:
            mask = mask[..., top : top + side, left : left + side]
        return image, mask

    def _resize(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        target = (self.size, self.size)
        if image.shape[-2:] != target:
            image = functional.interpolate(
                image.unsqueeze(0),
                size=target,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if mask is not None and mask.shape[-2:] != target:
            mask = (
                functional.interpolate(
                    mask[None, None].float(),
                    size=target,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )
        return image, mask

"""Multispectral dual-encoder segmentation network."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

DEFAULT_DROPOUTS = {
    "decoder": 0.14180278944984467,
    "fusion": 0.2851589427729743,
    "encoder": (
        0.03744604417521768,
        0.15493200931304196,
        0.11663530769985998,
        0.06721364658365152,
        0.40504448846283314,
    ),
}


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, pixel_shuffle_channels: int) -> None:
        super().__init__()
        if pixel_shuffle_channels % 4:
            raise ValueError("PixelShuffle input channels must be divisible by four")
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, pixel_shuffle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(pixel_shuffle_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DEFAULT_DROPOUTS["decoder"]),
            nn.PixelShuffle(upscale_factor=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(inputs)


class ResNetDecoder(nn.Module):
    def __init__(self, input_channels: int, *, pretrained: bool, filters: int = 32) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.encoder = resnet50(weights=weights)
        if input_channels != 3:
            original_weights = self.encoder.conv1.weight.detach().clone()
            self.encoder.conv1 = nn.Conv2d(
                input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            if pretrained:
                averaged = original_weights.mean(dim=1, keepdim=True)
                repeated = averaged.repeat(1, input_channels, 1, 1)
                self.encoder.conv1.weight.data.copy_(repeated)

        self.dropouts = nn.ModuleList(nn.Dropout2d(value) for value in DEFAULT_DROPOUTS["encoder"])
        self.decoder5 = DecoderBlock(2048, filters * 16)
        self.decoder4 = DecoderBlock(2048 + filters * 4, filters * 16)
        self.decoder3 = DecoderBlock(1024 + filters * 4, filters * 8)
        self.decoder2 = DecoderBlock(512 + filters * 2, filters * 4)
        self.decoder1 = DecoderBlock(256 + filters, filters * 2)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stage0 = self.encoder.conv1(inputs)
        stage0 = self.encoder.bn1(stage0)
        stage0 = self.encoder.relu(stage0)
        stage0 = self.dropouts[0](stage0)
        stage0 = self.encoder.maxpool(stage0)

        stage1 = self.encoder.layer1(stage0)
        stage2 = self.encoder.layer2(self.dropouts[1](stage1))
        stage3 = self.encoder.layer3(self.dropouts[2](stage2))
        stage4 = self.encoder.layer4(self.dropouts[3](stage3))
        stage4_dropped = self.dropouts[4](stage4)

        decoder5 = self.decoder5(functional.max_pool2d(stage4_dropped, kernel_size=2, stride=2))
        decoder4 = self.decoder4(torch.cat((stage4, decoder5), dim=1))
        decoder3 = self.decoder3(torch.cat((stage3, decoder4), dim=1))
        decoder2 = self.decoder2(torch.cat((stage2, decoder3), dim=1))
        decoder1 = self.decoder1(torch.cat((stage1, decoder2), dim=1))
        return decoder1, decoder2, decoder3, decoder4


def _upsample_and_add(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (
        functional.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        + target
    )


class FeaturePyramidFusion(nn.Module):
    def __init__(
        self,
        feature_channels: tuple[int, ...] = (32, 64, 128, 256),
        output_channels: int = 32,
    ) -> None:
        super().__init__()
        if feature_channels[0] != output_channels:
            raise ValueError("The first feature width must equal output_channels")
        self.projections = nn.ModuleList(
            nn.Conv2d(channels, output_channels, kernel_size=1) for channels in feature_channels[1:]
        )
        self.smoothing = nn.ModuleList(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
            for _ in feature_channels[1:]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                len(feature_channels) * output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(DEFAULT_DROPOUTS["fusion"]),
        )

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        projected = [features[0]]
        projected.extend(
            projection(feature)
            for feature, projection in zip(
                features[1:],
                self.projections,
                strict=True,
            )
        )
        pyramid = list(projected)
        for index in reversed(range(1, len(pyramid))):
            pyramid[index - 1] = _upsample_and_add(
                pyramid[index],
                pyramid[index - 1],
            )
        pyramid[:-1] = [
            smooth(feature)
            for smooth, feature in zip(
                self.smoothing,
                pyramid[:-1],
                strict=True,
            )
        ]
        target_size = pyramid[0].shape[-2:]
        pyramid = [
            feature
            if feature.shape[-2:] == target_size
            else functional.interpolate(
                feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for feature in pyramid
        ]
        return self.fusion(torch.cat(pyramid, dim=1))


class MSNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        n_channels: int,
        *,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if n_channels < 4:
            raise ValueError("MSNet requires three RGB channels and at least one auxiliary channel")
        self.n_channels = n_channels
        self.rgb_encoder = ResNetDecoder(3, pretrained=pretrained)
        self.auxiliary_encoder = ResNetDecoder(
            n_channels - 3,
            pretrained=pretrained,
        )
        self.fpn = FeaturePyramidFusion()
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] != self.n_channels:
            raise ValueError(
                f"MSNet expected {self.n_channels} channels, received {inputs.shape[1]}"
            )
        input_size = inputs.shape[-2:]
        rgb_features = self.rgb_encoder(inputs[:, :3])
        auxiliary_features = self.auxiliary_encoder(inputs[:, 3:])
        fused_features = tuple(
            torch.cat((rgb, auxiliary), dim=1)
            for rgb, auxiliary in zip(
                rgb_features,
                auxiliary_features,
                strict=True,
            )
        )
        fused = self.fpn(fused_features)
        logits = self.classifier(
            functional.interpolate(
                fused,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        return logits

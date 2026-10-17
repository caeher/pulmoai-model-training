"""Small 3D U-Net trained from scratch on spherical nodule proxy targets."""

import torch
from torch import nn
from torch.nn import functional as F


def _block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(inputs, outputs, 3, padding=1),
        nn.GroupNorm(2, outputs),
        nn.ReLU(),
        nn.Conv3d(outputs, outputs, 3, padding=1),
        nn.GroupNorm(2, outputs),
        nn.ReLU(),
    )


class SmallUNet3D(nn.Module):
    def __init__(self, base_channels: int = 8):
        super().__init__()
        c = base_channels
        self.encoder = _block(1, c)
        self.bottleneck = _block(c, c * 2)
        self.decoder = _block(c * 3, c)
        self.head = nn.Conv3d(c, 1, 1)

    def forward(self, image):
        features = self.encoder(image)
        low = self.bottleneck(F.max_pool3d(features, 2))
        up = F.interpolate(low, size=features.shape[2:], mode="trilinear", align_corners=False)
        return self.head(self.decoder(torch.cat([features, up], dim=1)))

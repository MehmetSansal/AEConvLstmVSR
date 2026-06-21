"""
Reconstruction head: ResBlocks + sub-pixel ×4 upsampling → residual image.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN with skip connection."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)  # residual connection


class ReconstructionHead(nn.Module):
    """
    Input:  h (B, 64, H, W)         — ConvLSTM final hidden state
    Output: residual (B, 3, 4H, 4W) — SR residual to be added to bicubic upsampled frame

    Pipeline:
      4× ResidualBlock → sub-pixel upsample ×4 → final 3-ch conv
    """

    def __init__(self, feature_channels: int = 64, num_residual_blocks: int = 4, scale: int = 4):
        super().__init__()
        C = feature_channels

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(C) for _ in range(num_residual_blocks)]
        )

        # Sub-pixel upsampling ×4: PixelShuffle(4) requires 64*16 = 1024 input channels
        self.upsample = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(C, C * scale * scale, 3, padding=1),  # (B, 64*16, H, W)
            nn.PixelShuffle(scale),                          # (B, 64,   4H, 4W)
        )

        # Final conv to RGB
        self.final_conv = nn.Conv2d(C, 3, 3, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, 64, H, W)
        x = self.res_blocks(h)      # (B, 64, H, W)
        x = self.upsample(x)        # (B, 64, 4H, 4W)
        residual = self.final_conv(x)  # (B, 3, 4H, 4W)
        return residual

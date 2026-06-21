"""
Deformable Convolution v2 alignment module (EDVR-style).
Uses torchvision.ops.deform_conv2d — no custom CUDA extension required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DeformableAligner(nn.Module):
    """
    Aligns one neighboring frame's features to the center frame's features.

    Input:  neighbor_feat (B, C, H, W), center_feat (B, C, H, W)
    Output: aligned_feat  (B, C, H, W)
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.channels = channels

        # Offset estimation layers: input is [neighbor_feat || center_feat] → 2C channels
        self.offset_conv = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(2 * channels, 2 * channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 3×3 kernel → 9 sampling points × 2 (x, y) = 18 offset channels
        self.offset_out = nn.Conv2d(2 * channels, 18, 3, padding=1)

        # 3×3 kernel → 9 modulation (mask) channels, sigmoid applied in forward
        self.mask_out = nn.Conv2d(2 * channels, 9, 3, padding=1)

        # Weight for the deformable conv: (C_out, C_in, kH, kW) = (C, C, 3, 3)
        self.deform_weight = nn.Parameter(
            torch.empty(channels, channels, 3, 3)
        )
        self.deform_bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.deform_weight, a=0.1)

    def forward(self, neighbor_feat: torch.Tensor, center_feat: torch.Tensor) -> torch.Tensor:
        # neighbor_feat: (B, C, H, W)
        # center_feat:   (B, C, H, W)

        concat = torch.cat([neighbor_feat, center_feat], dim=1)  # (B, 2C, H, W)

        feat = self.offset_conv(concat)   # (B, 2C, H, W)
        feat = self.offset_conv2(feat)    # (B, 2C, H, W)

        offset = self.offset_out(feat)    # (B, 18, H, W)
        mask = torch.sigmoid(self.mask_out(concat))  # (B, 9, H, W)

        # deform_conv2d expects: input, offset, weight, bias, stride, padding, dilation, mask
        aligned = deform_conv2d(
            input=neighbor_feat,
            offset=offset,
            weight=self.deform_weight,
            bias=self.deform_bias,
            padding=1,
            mask=mask,
        )  # (B, C, H, W)

        return aligned


class FeatureAlignmentModule(nn.Module):
    """
    Applies DeformableAligner to each of the 6 non-center frames, sharing weights.

    Input:  features (B, 7, C, H, W)  — 7 frame features, index 3 is center
    Output: aligned  (B, 7, C, H, W)  — all frames aligned to center (index 3)
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        # Single aligner instance; weights shared across all 6 non-center frames
        self.aligner = DeformableAligner(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 7, C, H, W)
        B, T, C, H, W = features.shape

        center = features[:, 3]  # (B, C, H, W) — im4, index 3

        aligned = []
        for t in range(T):
            if t == 3:
                aligned.append(center)
            else:
                aligned.append(self.aligner(features[:, t], center))  # (B, C, H, W)

        return torch.stack(aligned, dim=1)  # (B, 7, C, H, W)

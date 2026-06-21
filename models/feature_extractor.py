"""
Shared-weight CNN feature extractor applied independently per frame.
"""

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    Extracts 64-channel features from each of the 7 input LR frames.
    Weights are shared across all frames (applied in parallel via batch reshape).

    Input:  frames (B, 7, 3, H, W)
    Output: feats  (B, 7, 64, H, W)
    """

    def __init__(self, in_channels: int = 3, feature_channels: int = 64):
        super().__init__()
        C = feature_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(C, C, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(C, C, 3, padding=1),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, 7, 3, H, W)
        B, T, C_in, H, W = frames.shape

        # Reshape to process all frames in one batched forward pass
        x = frames.view(B * T, C_in, H, W)  # (B*7, 3, H, W)
        feat = self.net(x)                   # (B*7, 64, H, W)
        feat = feat.view(B, T, -1, H, W)     # (B, 7, 64, H, W)

        return feat

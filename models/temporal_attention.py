"""
Temporal attention module with SE-style channel excitation + softmax temporal weighting.
"""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """
    SE-style channel attention followed by softmax temporal weighting.

    Input:  features (B, 7, 64, H, W)
    Output: weighted (B, 7, 64, H, W)
    """

    def __init__(self, feature_channels: int = 64, num_frames: int = 7, ratio: int = 4):
        super().__init__()
        C = feature_channels
        reduced = max(C // ratio, 1)

        # Channel excitation (applied per-frame, weights shared)
        self.channel_fc1 = nn.Linear(C, reduced)
        self.channel_fc2 = nn.Linear(reduced, C)

        # Temporal score per frame (after channel re-weighting)
        self.temporal_fc = nn.Linear(C, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 7, 64, H, W)
        B, T, C, H, W = features.shape

        # ── Channel excitation ──────────────────────────────────────────────
        # Global average pool over spatial dims: (B, 7, 64, H, W) → (B, 7, 64)
        squeezed = features.mean(dim=[-2, -1])  # (B, 7, C)

        ch_att = torch.relu(self.channel_fc1(squeezed))   # (B, 7, C//ratio)
        ch_att = torch.sigmoid(self.channel_fc2(ch_att))  # (B, 7, C)

        # Multiply channel weights back: broadcast over H, W
        ch_att = ch_att.unsqueeze(-1).unsqueeze(-1)            # (B, 7, C, 1, 1)
        channel_weighted = features * ch_att                   # (B, 7, C, H, W)

        # ── Temporal softmax weighting ───────────────────────────────────────
        # Compute scalar score per frame using global-pooled channel-weighted features
        pooled = channel_weighted.mean(dim=[-2, -1])  # (B, 7, C)
        scores = self.temporal_fc(pooled)              # (B, 7, 1)
        alpha = torch.softmax(scores, dim=1)           # (B, 7, 1) — sums to 1 over T
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)      # (B, 7, 1, 1, 1)

        out = channel_weighted * alpha                 # (B, 7, C, H, W)

        return out

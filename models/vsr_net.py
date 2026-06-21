"""
Full VSRNet pipeline.

Ablation flags:
  use_alignment  — if False, skip DeformableAligner (pass features through unchanged)
  use_attention  — if False, skip TemporalAttention
  use_convlstm   — if False, replace ConvLSTM with simple mean pooling over time
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.feature_extractor import FeatureExtractor
from models.deformable_aligner import FeatureAlignmentModule
from models.temporal_attention import TemporalAttention
from models.convlstm import ConvLSTM
from models.reconstruction import ReconstructionHead


class VSRNet(nn.Module):
    """
    Attention-Enhanced ConvLSTM Video Super Resolution network.

    Input:
      lr_frames  (B, 7, 3, H, W)   — 7 LR input frames
      center_lr  (B, 3, H, W)      — center LR frame (im4) for bicubic upsampling branch

    Output:
      sr_out     (B, 3, 4H, 4W)    — super-resolved center frame, clamped to [0, 1]
    """

    def __init__(
        self,
        feature_channels: int = 64,
        hidden_channels: int = 64,
        num_residual_blocks: int = 4,
        attention_ratio: int = 4,
        scale: int = 4,
        use_alignment: bool = True,
        use_attention: bool = True,
        use_convlstm: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.use_alignment = use_alignment
        self.use_attention = use_attention
        self.use_convlstm = use_convlstm

        # Block 2 — Feature Extractor (shared weights across frames)
        self.feature_extractor = FeatureExtractor(
            in_channels=3, feature_channels=feature_channels
        )

        # Block 1 — Deformable Alignment (optional)
        if use_alignment:
            self.alignment = FeatureAlignmentModule(channels=feature_channels)

        # Block 3 — Temporal Attention (optional)
        if use_attention:
            self.temporal_attention = TemporalAttention(
                feature_channels=feature_channels,
                num_frames=7,
                ratio=attention_ratio,
            )

        # Block 4 — ConvLSTM (optional; fallback: mean over temporal dim)
        if use_convlstm:
            self.convlstm = ConvLSTM(
                input_channels=feature_channels,
                hidden_channels=hidden_channels,
            )
            lstm_out_channels = hidden_channels
        else:
            lstm_out_channels = feature_channels

        # Block 5 — Reconstruction Head
        self.reconstruction = ReconstructionHead(
            feature_channels=lstm_out_channels,
            num_residual_blocks=num_residual_blocks,
            scale=scale,
        )

    def forward(self, lr_frames: torch.Tensor, center_lr: torch.Tensor) -> torch.Tensor:
        # lr_frames: (B, 7, 3, H, W)
        # center_lr: (B, 3, H, W)
        B, T, C_in, H, W = lr_frames.shape

        # ── Block 2: Feature Extraction ────────────────────────────────────
        features = self.feature_extractor(lr_frames)  # (B, 7, 64, H, W)

        # ── Block 1: Deformable Alignment ──────────────────────────────────
        if self.use_alignment:
            features = self.alignment(features)        # (B, 7, 64, H, W)

        # ── Block 3: Temporal Attention ────────────────────────────────────
        if self.use_attention:
            features = self.temporal_attention(features)  # (B, 7, 64, H, W)

        # ── Block 4: ConvLSTM (or mean fallback) ───────────────────────────
        if self.use_convlstm:
            h = self.convlstm(features)                # (B, 64, H, W)
        else:
            # Simple temporal average when ConvLSTM is ablated
            h = features.mean(dim=1)                   # (B, 64, H, W)

        # ── Block 5: Reconstruction Head ───────────────────────────────────
        residual = self.reconstruction(h)              # (B, 3, 4H, 4W)

        # ── Block 6: Bicubic + Residual Fusion ─────────────────────────────
        bicubic = F.interpolate(
            center_lr,
            scale_factor=self.scale,
            mode='bicubic',
            align_corners=False,
        )  # (B, 3, 4H, 4W)

        sr_out = torch.clamp(bicubic + residual, 0.0, 1.0)  # (B, 3, 4H, 4W)
        return sr_out

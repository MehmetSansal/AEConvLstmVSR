"""
Loss functions: L1, Edge (Sobel gradient L1), and combined TotalLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    """Standard pixel-level L1 loss."""

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(sr, hr)


class EdgeLoss(nn.Module):
    """
    Sobel gradient L1 loss — penalises edge/texture differences.
    Sobel kernels applied separately per channel in a batched depthwise conv.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels: (1, 1, 3, 3)
        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
        ).unsqueeze(0)  # (1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
        ).unsqueeze(0)  # (1, 1, 3, 3)

        # Register as buffers so they move with .to(device) but are not parameters
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        B, C, H, W = x.shape

        # Treat each channel independently: reshape to (B*C, 1, H, W)
        x_flat = x.view(B * C, 1, H, W)

        gx = F.conv2d(x_flat, self.sobel_x, padding=1)  # (B*C, 1, H, W)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1)  # (B*C, 1, H, W)

        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)  # (B*C, 1, H, W)
        return magnitude.view(B, C, H, W)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_grad = self._gradient_magnitude(sr)  # (B, 3, H, W)
        hr_grad = self._gradient_magnitude(hr)  # (B, 3, H, W)
        return F.l1_loss(sr_grad, hr_grad)


class TotalLoss(nn.Module):
    """
    Combined loss: L1 + edge_weight × EdgeLoss.

    Returns dict: {'total': tensor, 'l1': tensor, 'edge': tensor}
    """

    def __init__(self, l1_weight: float = 1.0, edge_weight: float = 0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.edge_weight = edge_weight
        self.l1_loss = L1Loss()
        self.edge_loss = EdgeLoss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> dict:
        l1 = self.l1_loss(sr, hr)
        edge = self.edge_loss(sr, hr)
        total = self.l1_weight * l1 + self.edge_weight * edge
        return {'total': total, 'l1': l1, 'edge': edge}

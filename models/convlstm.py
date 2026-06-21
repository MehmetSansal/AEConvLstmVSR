"""
ConvLSTMCell implemented from scratch (no external library).
Processes 7 aligned feature frames sequentially; returns final hidden state.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell. All four gates packed into one convolution for efficiency.

    Input:  x (B, C_in, H, W), h (B, C_h, H, W), c (B, C_h, H, W)
    Output: h_new (B, C_h, H, W), c_new (B, C_h, H, W)
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # One conv outputs 4×hidden_channels: [i, f, g, o] gates
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ):
        # x: (B, C_in, H, W)   h, c: (B, C_h, H, W)
        combined = torch.cat([x, h], dim=1)  # (B, C_in + C_h, H, W)
        gates = self.gates(combined)          # (B, 4*C_h, H, W)

        i, f, g, o = gates.chunk(4, dim=1)   # each (B, C_h, H, W)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g   # (B, C_h, H, W)
        h_new = o * torch.tanh(c_new)  # (B, C_h, H, W)

        return h_new, c_new


class ConvLSTM(nn.Module):
    """
    Runs ConvLSTMCell over T=7 frames sequentially.

    Input:  features (B, 7, C_in, H, W)
    Output: h_final  (B, C_h, H, W)  — hidden state after processing all 7 frames
    """

    def __init__(self, input_channels: int = 64, hidden_channels: int = 64):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(input_channels, hidden_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 7, C_in, H, W)
        B, T, C_in, H, W = features.shape
        device = features.device
        dtype = features.dtype

        # Initialise h_0, c_0 as zeros
        h = torch.zeros(B, self.hidden_channels, H, W, device=device, dtype=dtype)
        c = torch.zeros(B, self.hidden_channels, H, W, device=device, dtype=dtype)

        for t in range(T):
            x_t = features[:, t]      # (B, C_in, H, W)
            h, c = self.cell(x_t, h, c)

        return h  # (B, C_h, H, W) — final hidden state h_7

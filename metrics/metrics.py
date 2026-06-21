"""
PSNR and SSIM implemented from scratch (no skimage).
"""

import math
from typing import List

import torch
import torch.nn.functional as F


# ─── PSNR ────────────────────────────────────────────────────────────────────

def compute_psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio for a single (C, H, W) or (1, C, H, W) tensor pair.
    Both tensors must be in [0, 1].

    Returns PSNR in dB as a Python float.
    """
    sr = sr.detach().float()
    hr = hr.detach().float()

    mse = torch.mean((sr - hr) ** 2).item()
    if mse < 1e-10:
        return 100.0  # effectively infinite PSNR

    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)


# ─── SSIM ────────────────────────────────────────────────────────────────────

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Create a 1-D Gaussian kernel of length `size`."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2-D Gaussian kernel (size × size)."""
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)  # outer product
    return k2d / k2d.sum()


def compute_ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = (0.01 ** 2),
    C2: float = (0.03 ** 2),
) -> float:
    """
    Structural Similarity Index for a single image pair.

    Tensors can be (C, H, W) or (1, C, H, W), values in [0, 1].
    Returns SSIM as a Python float.
    """
    sr = sr.detach().float()
    hr = hr.detach().float()

    # Ensure 4-D: (1, C, H, W)
    if sr.dim() == 3:
        sr = sr.unsqueeze(0)
        hr = hr.unsqueeze(0)

    B, C, H, W = sr.shape

    # Build Gaussian kernel and expand for depthwise conv over all channels
    kernel = _gaussian_kernel_2d(window_size, sigma).to(sr.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kH, kW)
    kernel = kernel.expand(C, 1, window_size, window_size)  # (C, 1, kH, kW)

    pad = window_size // 2

    def conv(x: torch.Tensor) -> torch.Tensor:
        # Depthwise: groups=C so each channel filtered independently
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu_sr = conv(sr)   # (1, C, H, W) — local means
    mu_hr = conv(hr)

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr

    sigma_sr_sq = conv(sr * sr) - mu_sr_sq   # local variance of sr
    sigma_hr_sq = conv(hr * hr) - mu_hr_sq   # local variance of hr
    sigma_sr_hr = conv(sr * hr) - mu_sr_hr   # local covariance

    numerator   = (2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)
    denominator = (mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2)

    ssim_map = numerator / (denominator + 1e-8)  # (1, C, H, W)
    return ssim_map.mean().item()


# ─── MetricTracker ───────────────────────────────────────────────────────────

class MetricTracker:
    """
    Accumulates per-image PSNR and SSIM values across a batch or dataset.

    Usage:
        tracker = MetricTracker()
        tracker.update(sr_batch, hr_batch)   # (B, 3, H, W) tensors in [0,1]
        stats = tracker.summary()
    """

    def __init__(self):
        self._psnr_values: List[float] = []
        self._ssim_values: List[float] = []

    def reset(self):
        self._psnr_values.clear()
        self._ssim_values.clear()

    def update(self, sr_batch: torch.Tensor, hr_batch: torch.Tensor):
        # sr_batch, hr_batch: (B, C, H, W) in [0, 1]
        B = sr_batch.shape[0]
        for i in range(B):
            psnr = compute_psnr(sr_batch[i], hr_batch[i])
            ssim = compute_ssim(sr_batch[i], hr_batch[i])
            self._psnr_values.append(psnr)
            self._ssim_values.append(ssim)

    def summary(self) -> dict:
        import statistics
        psnr_vals = self._psnr_values
        ssim_vals = self._ssim_values

        if not psnr_vals:
            return {
                'psnr_mean': 0.0, 'psnr_std': 0.0,
                'psnr_min': 0.0,  'psnr_max': 0.0,
                'ssim_mean': 0.0, 'ssim_std': 0.0,
                'ssim_min': 0.0,  'ssim_max': 0.0,
            }

        def _stats(vals):
            mean = statistics.mean(vals)
            std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            return mean, std, min(vals), max(vals)

        pm, ps, pmin, pmax = _stats(psnr_vals)
        sm, ss, smin, smax = _stats(ssim_vals)

        return {
            'psnr_mean': pm, 'psnr_std': ps,
            'psnr_min':  pmin, 'psnr_max': pmax,
            'ssim_mean': sm, 'ssim_std': ss,
            'ssim_min':  smin, 'ssim_max': smax,
        }

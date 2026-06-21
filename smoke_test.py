"""
Smoke test — runs entirely on synthetic tensors, no dataset required.

Usage:
  python smoke_test.py
  python smoke_test.py --cpu    # force CPU even if CUDA available
"""

import argparse
import sys

import torch


def check(name, tensor, expected_shape):
    shape = tuple(tensor.shape)
    if shape != expected_shape:
        print(f'✗ {name}  expected {expected_shape}  got {shape}')
        sys.exit(1)
    print(f'✓ {name:<24} input shape → output {shape}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() else torch.device('cuda')
    print(f'\nRunning smoke test on device: {device}\n')

    # ── 1. DeformableAligner in isolation ─────────────────────────────────────
    from models.deformable_aligner import DeformableAligner

    feat_neighbor = torch.randn(2, 64, 64, 64, device=device)
    feat_center   = torch.randn(2, 64, 64, 64, device=device)
    aligner       = DeformableAligner(64).to(device)
    aligned_single = aligner(feat_neighbor, feat_center)
    assert aligned_single.shape == (2, 64, 64, 64), \
        f'DeformableAligner output shape mismatch: {aligned_single.shape}'
    print(f'✓ DeformableAligner        input=(2,64,64,64) output={tuple(aligned_single.shape)}')

    # ── 2. FeatureExtractor ───────────────────────────────────────────────────
    from models.feature_extractor import FeatureExtractor

    lr_frames = torch.randn(2, 7, 3, 64, 64, device=device)
    extractor = FeatureExtractor(in_channels=3, feature_channels=64).to(device)
    feat_out  = extractor(lr_frames)
    check('FeatureExtractor', feat_out, (2, 7, 64, 64, 64))

    # ── 3. FeatureAlignmentModule ─────────────────────────────────────────────
    from models.deformable_aligner import FeatureAlignmentModule

    fam     = FeatureAlignmentModule(channels=64).to(device)
    fam_out = fam(feat_out)
    check('FeatureAlignment', fam_out, (2, 7, 64, 64, 64))

    # ── 4. TemporalAttention ──────────────────────────────────────────────────
    from models.temporal_attention import TemporalAttention

    ta     = TemporalAttention(feature_channels=64, num_frames=7, ratio=4).to(device)
    ta_out = ta(fam_out)
    check('TemporalAttention', ta_out, (2, 7, 64, 64, 64))

    # ── 5. ConvLSTM ───────────────────────────────────────────────────────────
    from models.convlstm import ConvLSTM

    lstm     = ConvLSTM(input_channels=64, hidden_channels=64).to(device)
    lstm_out = lstm(ta_out)
    check('ConvLSTM', lstm_out, (2, 64, 64, 64))

    # ── 6. ReconstructionHead ─────────────────────────────────────────────────
    from models.reconstruction import ReconstructionHead

    head     = ReconstructionHead(feature_channels=64, num_residual_blocks=4, scale=4).to(device)
    head_out = head(lstm_out)
    check('ReconstructionHead', head_out, (2, 3, 256, 256))

    # ── 7. Full VSRNet forward pass ───────────────────────────────────────────
    from models.vsr_net import VSRNet

    model    = VSRNet(
        feature_channels=64,
        hidden_channels=64,
        num_residual_blocks=4,
        attention_ratio=4,
        scale=4,
        use_alignment=True,
        use_attention=True,
        use_convlstm=True,
    ).to(device)

    lr_in  = torch.randn(2, 7, 3, 64, 64, device=device)
    clr_in = torch.randn(2, 3, 64, 64, device=device)
    sr_out = model(lr_in, clr_in)
    assert sr_out.shape == (2, 3, 256, 256), f'VSRNet output shape: {sr_out.shape}'
    print(f'✓ Full VSRNet              input LR=(2,7,3,64,64) → SR={tuple(sr_out.shape)}')

    # ── 8. TotalLoss ──────────────────────────────────────────────────────────
    from losses.losses import TotalLoss

    criterion = TotalLoss(l1_weight=1.0, edge_weight=0.1).to(device)
    hr_fake   = torch.rand(2, 3, 256, 256, device=device)
    losses    = criterion(sr_out, hr_fake)

    assert not torch.isnan(losses['total']), "NaN in total loss!"
    assert not torch.isinf(losses['total']), "Inf in total loss!"
    print(
        f"✓ TotalLoss                total={losses['total'].item():.4f}"
        f"  l1={losses['l1'].item():.4f}"
        f"  edge={losses['edge'].item():.4f}"
    )

    # ── 9. Backward pass ──────────────────────────────────────────────────────
    losses['total'].backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f'✗ NaN gradient in {name}')
                sys.exit(1)
            if torch.isinf(param.grad).any():
                print(f'✗ Inf gradient in {name}')
                sys.exit(1)

    print('✓ Backward pass            no NaN/Inf in gradients')

    # ── 10. PSNR metric ───────────────────────────────────────────────────────
    from metrics.metrics import compute_psnr

    a = torch.rand(3, 256, 256, device=device)
    b = torch.rand(3, 256, 256, device=device)
    psnr_val = compute_psnr(a, b)
    assert isinstance(psnr_val, float) and not (psnr_val != psnr_val), "PSNR is NaN"
    print(f'✓ PSNR metric              {psnr_val:.2f} dB')

    # ── 11. SSIM metric ───────────────────────────────────────────────────────
    from metrics.metrics import compute_ssim

    ssim_val = compute_ssim(a, b)
    assert isinstance(ssim_val, float) and not (ssim_val != ssim_val), "SSIM is NaN"
    print(f'✓ SSIM metric              {ssim_val:.4f}')

    print('\n✓ All smoke tests passed.\n')


if __name__ == '__main__':
    main()

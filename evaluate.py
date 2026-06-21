"""
Evaluation script: full test set metrics, bicubic baseline, ablation table,
and visual comparison saves.

Usage:
  python evaluate.py --checkpoint checkpoints/best_model.pth
  python evaluate.py --checkpoint checkpoints/best_model.pth --config configs/default.yaml
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _nested_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _nested_namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _nested_namespace(raw)


def load_model(checkpoint_path: str, config, device: torch.device,
               use_alignment=None, use_attention=None, use_convlstm=None):
    from models.vsr_net import VSRNet

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ua = use_alignment if use_alignment is not None else config.model.use_alignment
    uat = use_attention if use_attention is not None else config.model.use_attention
    uc = use_convlstm if use_convlstm is not None else config.model.use_convlstm

    model = VSRNet(
        feature_channels=config.model.feature_channels,
        hidden_channels=config.model.hidden_channels,
        num_residual_blocks=config.model.num_residual_blocks,
        attention_ratio=config.model.attention_ratio,
        scale=config.data.scale,
        use_alignment=ua,
        use_attention=uat,
        use_convlstm=uc,
    ).to(device)

    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    return model


def evaluate_model(model, test_loader, device: torch.device, desc: str = 'Model'):
    """Run inference on test set; returns per-sequence metrics list."""
    from metrics.metrics import compute_psnr, compute_ssim

    results = []
    with torch.no_grad():
        for batch in test_loader:
            lr_frames = batch['lr_frames'].to(device, non_blocking=True)
            hr_center = batch['hr_center'].to(device, non_blocking=True)
            center_lr = batch['center_lr'].to(device, non_blocking=True)
            seq_ids   = batch['sequence_id']

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sr_out = model(lr_frames, center_lr)

            sr_f = sr_out.float().clamp(0, 1)
            hr_f = hr_center.float()

            for i in range(sr_f.shape[0]):
                psnr = compute_psnr(sr_f[i], hr_f[i])
                ssim = compute_ssim(sr_f[i], hr_f[i])
                results.append({
                    'sequence_id': seq_ids[i],
                    'psnr':        psnr,
                    'ssim':        ssim,
                })

    return results


def evaluate_bicubic(test_loader, device: torch.device):
    """Compute bicubic baseline metrics."""
    from metrics.metrics import compute_psnr, compute_ssim

    results = []
    with torch.no_grad():
        for batch in test_loader:
            hr_center = batch['hr_center'].to(device, non_blocking=True)
            center_lr = batch['center_lr'].to(device, non_blocking=True)
            seq_ids   = batch['sequence_id']

            scale = hr_center.shape[-1] // center_lr.shape[-1]
            bicubic = F.interpolate(
                center_lr, scale_factor=scale, mode='bicubic', align_corners=False
            ).clamp(0, 1).float()

            hr_f = hr_center.float()

            for i in range(bicubic.shape[0]):
                psnr = compute_psnr(bicubic[i], hr_f[i])
                ssim = compute_ssim(bicubic[i], hr_f[i])
                results.append({
                    'sequence_id': seq_ids[i],
                    'psnr':        psnr,
                    'ssim':        ssim,
                })
    return results


def mean_metric(results, key):
    vals = [r[key] for r in results]
    return sum(vals) / len(vals) if vals else 0.0


def save_visual_comparisons(model, test_loader, device, results_dir: Path):
    """Save side-by-side comparison images for every test sequence."""
    from metrics.metrics import compute_psnr, compute_ssim

    results_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()

    for batch in test_loader:
        lr_frames = batch['lr_frames'].to(device, non_blocking=True)
        hr_center = batch['hr_center'].to(device, non_blocking=True)
        center_lr = batch['center_lr'].to(device, non_blocking=True)
        seq_ids   = batch['sequence_id']

        scale = hr_center.shape[-1] // center_lr.shape[-1]

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sr_out = model(lr_frames, center_lr)

        sr_f      = sr_out.float().clamp(0, 1)
        hr_f      = hr_center.float()
        bicubic_f = F.interpolate(
            center_lr, scale_factor=scale, mode='bicubic', align_corners=False
        ).clamp(0, 1).float()

        for i in range(sr_f.shape[0]):
            psnr = compute_psnr(sr_f[i], hr_f[i])
            ssim = compute_ssim(sr_f[i], hr_f[i])

            bic_np = bicubic_f[i].permute(1, 2, 0).cpu().numpy()
            sr_np  = sr_f[i].permute(1, 2, 0).cpu().numpy()
            hr_np  = hr_f[i].permute(1, 2, 0).cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(np.clip(bic_np, 0, 1))
            axes[0].set_title('Bicubic ×4', fontsize=12)
            axes[0].axis('off')

            axes[1].imshow(np.clip(sr_np, 0, 1))
            axes[1].set_title(f'SR Output  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}', fontsize=12)
            axes[1].axis('off')

            axes[2].imshow(np.clip(hr_np, 0, 1))
            axes[2].set_title('HR Ground Truth', fontsize=12)
            axes[2].axis('off')

            fig.suptitle(f'Sequence: {seq_ids[i]}', fontsize=10)
            plt.tight_layout()

            out_path = results_dir / f'visual_{seq_ids[i]}.png'
            plt.savefig(str(out_path), dpi=100, bbox_inches='tight')
            plt.close(fig)
            saved += 1

    logger.info("Saved %d visual comparisons to %s", saved, results_dir)


# ─── Comparison helpers ───────────────────────────────────────────────────────

def _load_baseline(cls, ckpt_path: str, device: torch.device, scale: int = 4):
    """Instantiate a baseline model, optionally loading pretrained weights."""
    model = cls(scale=scale).to(device)
    if ckpt_path:
        p = Path(ckpt_path)
        if p.exists():
            sd = torch.load(str(p), map_location=device, weights_only=False)
            # Handle both raw state-dicts and checkpoint dicts
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            elif isinstance(sd, dict) and 'model_state' in sd:
                sd = sd['model_state']
            model.load_state_dict(sd, strict=False)
            logger.info("Loaded %s weights from %s", cls.__name__, p)
        else:
            logger.warning("%s ckpt not found: %s — using random weights", cls.__name__, p)
    else:
        logger.warning("%s: no checkpoint supplied — using random weights "
                       "(metrics will be poor; load pretrained for paper results)", cls.__name__)
    model.eval()
    return model


def _run_timed(model, test_loader, device, use_lr_frames: bool):
    """
    Run model over test_loader, return per-sequence dicts with psnr/ssim/time_ms.
    use_lr_frames=True  → model(lr_frames, center_lr)   [BasicVSR, EDVR, Ours]
    use_lr_frames=False → model(center_lr)               [SRCNN]
    """
    from metrics.metrics import compute_psnr, compute_ssim

    results = []
    with torch.no_grad():
        for batch in test_loader:
            lr_frames = batch['lr_frames'].to(device, non_blocking=True)
            hr_center = batch['hr_center'].to(device, non_blocking=True)
            center_lr = batch['center_lr'].to(device, non_blocking=True)
            seq_ids   = batch['sequence_id']

            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                if use_lr_frames:
                    sr = model(lr_frames, center_lr)
                else:
                    sr = model(center_lr)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            sr_f = sr.float().clamp(0, 1)
            hr_f = hr_center.float()
            ms_per_seq = elapsed_ms / sr_f.shape[0]

            for i in range(sr_f.shape[0]):
                results.append({
                    'sequence_id': seq_ids[i],
                    'psnr':        compute_psnr(sr_f[i], hr_f[i]),
                    'ssim':        compute_ssim(sr_f[i], hr_f[i]),
                    'time_ms':     ms_per_seq,
                })
    return results


def _run_bicubic_timed(test_loader, device):
    """Bicubic baseline with timing."""
    from metrics.metrics import compute_psnr, compute_ssim

    results = []
    with torch.no_grad():
        for batch in test_loader:
            hr_center = batch['hr_center'].to(device, non_blocking=True)
            center_lr = batch['center_lr'].to(device, non_blocking=True)
            seq_ids   = batch['sequence_id']
            scale     = hr_center.shape[-1] // center_lr.shape[-1]

            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            bic = F.interpolate(center_lr, scale_factor=scale,
                                mode='bicubic', align_corners=False).clamp(0, 1).float()
            torch.cuda.synchronize() if device.type == 'cuda' else None
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            hr_f = hr_center.float()
            ms_per_seq = elapsed_ms / bic.shape[0]
            for i in range(bic.shape[0]):
                results.append({
                    'sequence_id': seq_ids[i],
                    'psnr':        compute_psnr(bic[i], hr_f[i]),
                    'ssim':        compute_ssim(bic[i], hr_f[i]),
                    'time_ms':     ms_per_seq,
                })
    return results


def _save_compare_csvs(all_results: dict, out_dir: Path):
    """
    Saves two CSVs:
      compare_per_sequence.csv  — one row per (sequence, model)
      compare_avg.csv           — one row per model (averages)
    """
    per_seq_path = out_dir / 'compare_per_sequence.csv'
    avg_path     = out_dir / 'compare_avg.csv'

    with open(per_seq_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'sequence_id', 'psnr', 'ssim', 'time_ms'])
        w.writeheader()
        for model_name, rows in all_results.items():
            for r in rows:
                w.writerow({'model': model_name, **r})

    avg_rows = []
    for model_name, rows in all_results.items():
        n = len(rows)
        avg_rows.append({
            'model':   model_name,
            'psnr':    f"{sum(r['psnr']    for r in rows) / n:.4f}",
            'ssim':    f"{sum(r['ssim']    for r in rows) / n:.6f}",
            'time_ms': f"{sum(r['time_ms'] for r in rows) / n:.2f}",
        })

    with open(avg_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'psnr', 'ssim', 'time_ms'])
        w.writeheader()
        w.writerows(avg_rows)

    logger.info("Saved:  %s", per_seq_path)
    logger.info("Saved:  %s", avg_path)
    return avg_rows


def _save_compare_visuals(
    models: dict,           # {name: (model, use_lr_frames)}
    test_loader,
    device: torch.device,
    out_dir: Path,
):
    """
    Save a 2×3 comparison image for every test sequence:

      Row 0: Bicubic ×4 | SRCNN | BasicVSR
      Row 1: EDVR       | AEConvLSTM (Ours) | HR Ground Truth
    """
    from metrics.metrics import compute_psnr, compute_ssim

    out_dir.mkdir(parents=True, exist_ok=True)

    PANEL_ORDER = ['Bicubic ×4', 'SRCNN', 'BasicVSR', 'EDVR', 'AEConvLSTM (Ours)', 'HR GT']

    saved = 0
    with torch.no_grad():
        for batch in test_loader:
            lr_frames = batch['lr_frames'].to(device, non_blocking=True)
            hr_center = batch['hr_center'].to(device, non_blocking=True)
            center_lr = batch['center_lr'].to(device, non_blocking=True)
            seq_ids   = batch['sequence_id']
            scale     = hr_center.shape[-1] // center_lr.shape[-1]

            # Run all models, collect (B,3,H,W) outputs
            outputs = {}
            bic = F.interpolate(center_lr, scale_factor=scale,
                                mode='bicubic', align_corners=False).clamp(0, 1).float()
            outputs['Bicubic ×4'] = bic
            outputs['HR GT']      = hr_center.float()

            for name, (model, use_lr) in models.items():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=(device.type == 'cuda')):
                    sr = model(lr_frames, center_lr) if use_lr else model(center_lr)
                outputs[name] = sr.float().clamp(0, 1)

            for i in range(lr_frames.shape[0]):
                panels = {}
                for key, tensor in outputs.items():
                    panels[key] = tensor[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)

                fig, axes = plt.subplots(2, 3, figsize=(21, 10))
                axes = axes.flatten()

                for ax_idx, panel_name in enumerate(PANEL_ORDER):
                    img = panels.get(panel_name)
                    ax  = axes[ax_idx]
                    if img is None:
                        ax.axis('off')
                        ax.set_title(f'{panel_name}\n(not available)', fontsize=10)
                        continue

                    ax.imshow(img)
                    ax.axis('off')

                    if panel_name == 'HR GT':
                        ax.set_title('HR Ground Truth', fontsize=11, fontweight='bold')
                    elif panel_name == 'Bicubic ×4':
                        psnr = compute_psnr(
                            torch.from_numpy(img).permute(2,0,1).to(device),
                            hr_center[i].float(),
                        )
                        ssim = compute_ssim(
                            torch.from_numpy(img).permute(2,0,1).to(device),
                            hr_center[i].float(),
                        )
                        ax.set_title(f'Bicubic ×4\nPSNR {psnr:.2f} dB  SSIM {ssim:.4f}',
                                     fontsize=10)
                    else:
                        psnr = compute_psnr(
                            torch.from_numpy(img).permute(2,0,1).to(device),
                            hr_center[i].float(),
                        )
                        ssim = compute_ssim(
                            torch.from_numpy(img).permute(2,0,1).to(device),
                            hr_center[i].float(),
                        )
                        label = panel_name
                        ax.set_title(f'{label}\nPSNR {psnr:.2f} dB  SSIM {ssim:.4f}',
                                     fontsize=10)

                fig.suptitle(f'Sequence: {seq_ids[i]}', fontsize=12, fontweight='bold')
                plt.tight_layout()
                out_path = out_dir / f'compare_{seq_ids[i]}.png'
                plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
                plt.close(fig)
                saved += 1

    logger.info("Saved %d comparison images to %s", saved, out_dir)


def run_comparison(args, config, device, test_loader, our_model):
    """Entry point for --compare mode.

    Pretrained checkpoints are auto-downloaded from OpenMMLab the first time
    they are needed.  Pass --srcnn_ckpt / --basicvsr_ckpt / --edvr_ckpt to
    override with local paths.
    """
    from models.baseline_models import SRCNN, BasicVSR, EDVR

    compare_dir = Path(config.logging.results_dir) / 'compare'
    compare_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("COMPARISON MODE — loading baseline models …")
    logger.info("  (pretrained checkpoints will be downloaded if not cached)")
    logger.info("=" * 60)

    srcnn    = SRCNN.from_pretrained(device,
                   user_ckpt=getattr(args, 'srcnn_ckpt', '') or '')
    basicvsr = BasicVSR.from_pretrained(device,
                   user_ckpt=getattr(args, 'basicvsr_ckpt', '') or '')
    edvr     = EDVR.from_pretrained(device,
                   user_ckpt=getattr(args, 'edvr_ckpt', '') or '')

    # (model, use_lr_frames)
    named_models = {
        'SRCNN':             (srcnn,     False),
        'BasicVSR':          (basicvsr,  True),
        'EDVR':              (edvr,      True),
        'AEConvLSTM (Ours)': (our_model, True),
    }

    # ── Collect metrics ────────────────────────────────────────────────────────
    all_results = {}

    logger.info("Running Bicubic ×4 baseline …")
    all_results['Bicubic ×4'] = _run_bicubic_timed(test_loader, device)

    for name, (model, use_lr) in named_models.items():
        logger.info("Running %s …", name)
        all_results[name] = _run_timed(model, test_loader, device, use_lr)

    # ── CSVs ──────────────────────────────────────────────────────────────────
    avg_rows = _save_compare_csvs(all_results, compare_dir)

    # ── Console table ──────────────────────────────────────────────────────────
    table_rows = [
        [r['model'], r['psnr'], r['ssim'], r['time_ms'] + ' ms/seq']
        for r in avg_rows
    ]
    print_table(
        'Comparison — Vimeo-90K ×4  (averages over test set)',
        table_rows,
        ['Method', 'PSNR (dB)', 'SSIM', 'Time'],
    )

    # ── Visual comparisons ─────────────────────────────────────────────────────
    logger.info("Saving visual comparisons …")
    _save_compare_visuals(named_models, test_loader, device, compare_dir)

    logger.info("Compare results saved to:  %s", compare_dir)


def print_table(title, rows, headers):
    """Print a simple ASCII table."""
    col_widths = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = '┼'.join('─' * (w + 2) for w in col_widths)

    print(f'\n{title}')
    print('┌' + '┬'.join('─' * (w + 2) for w in col_widths) + '┐')
    print('│' + '│'.join(f' {h:<{w}} ' for h, w in zip(headers, col_widths)) + '│')
    print('├' + sep + '┤')
    for row in rows:
        print('│' + '│'.join(f' {str(v):<{w}} ' for v, w in zip(row, col_widths)) + '│')
    print('└' + '┴'.join('─' * (w + 2) for w in col_widths) + '┘')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    default='checkpoints/best_model.pth')
    parser.add_argument('--config',        default='configs/default.yaml')
    # ── Comparison mode ────────────────────────────────────────────────────────
    parser.add_argument('--compare',       action='store_true',
                        help='Compare with SRCNN, BasicVSR, EDVR baselines')
    parser.add_argument('--srcnn_ckpt',    default='',
                        help='Path to SRCNN pretrained .pth (optional)')
    parser.add_argument('--basicvsr_ckpt', default='',
                        help='Path to BasicVSR pretrained .pth (optional)')
    parser.add_argument('--edvr_ckpt',     default='',
                        help='Path to EDVR pretrained .pth (optional)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config.hardware.device if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    # ── Dataset guard ─────────────────────────────────────────────────────────
    test_csv = Path(config.data.test_manifest)
    if not test_csv.exists():
        logger.warning(
            "Test manifest not found: %s\n"
            "Run scripts/prepare_dataset.py first.",
            test_csv,
        )
        sys.exit(0)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.warning("Checkpoint not found: %s", ckpt_path)
        sys.exit(0)

    # ── DataLoader ────────────────────────────────────────────────────────────
    from data.vimeo_dataset import get_dataloader

    # Use smaller batch for eval to fit within VRAM during full-res inference
    eval_config = _nested_namespace(yaml.safe_load(open(args.config)))
    eval_config.train.batch_size = 4  # smaller for eval

    test_loader = get_dataloader(str(test_csv), 'test', eval_config)

    if len(test_loader.dataset) == 0:
        logger.warning("Test dataset is empty — aborting evaluation.")
        sys.exit(0)

    logger.info("Test sequences: %d", len(test_loader.dataset))

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(str(ckpt_path), config, device)

    results_dir = Path(config.logging.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate full model ───────────────────────────────────────────────────
    logger.info("Evaluating proposed model ...")
    model_results  = evaluate_model(model, test_loader, device)
    model_psnr     = mean_metric(model_results, 'psnr')
    model_ssim     = mean_metric(model_results, 'ssim')

    # ── Bicubic baseline ──────────────────────────────────────────────────────
    logger.info("Computing bicubic baseline ...")
    bic_results    = evaluate_bicubic(test_loader, device)
    bic_psnr       = mean_metric(bic_results, 'psnr')
    bic_ssim       = mean_metric(bic_results, 'ssim')

    # ── Per-sequence CSV ──────────────────────────────────────────────────────
    csv_path = results_dir / 'per_sequence_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence_id', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(model_results)
    logger.info("Per-sequence metrics saved: %s", csv_path)

    # ── Visual comparisons ────────────────────────────────────────────────────
    save_visual_comparisons(model, test_loader, device, results_dir)

    # ── Main results table ────────────────────────────────────────────────────
    delta_psnr = model_psnr - bic_psnr
    delta_ssim = model_ssim - bic_ssim

    main_rows = [
        ['Bicubic ×4',         f'{bic_psnr:.2f}',   f'{bic_ssim:.4f}'],
        ['Proposed (Ours)',     f'{model_psnr:.2f}', f'{model_ssim:.4f}'],
        [f'Δ (ours − bicubic)', f'{delta_psnr:+.2f}', f'{delta_ssim:+.4f}'],
    ]
    print_table(
        'Results — Vimeo-90K ×4 Super Resolution',
        main_rows,
        ['Method', 'PSNR (dB)', 'SSIM'],
    )

    # ── Ablation study ────────────────────────────────────────────────────────
    logger.info("Running ablation variants ...")

    def run_ablation(use_align, use_attn, use_lstm):
        m = load_model(str(ckpt_path), config, device,
                       use_alignment=use_align,
                       use_attention=use_attn,
                       use_convlstm=use_lstm)
        res = evaluate_model(m, test_loader, device)
        return mean_metric(res, 'psnr'), mean_metric(res, 'ssim')

    # Variant B: CNN only (no alignment, no attention, no LSTM)
    b_psnr, b_ssim = run_ablation(False, False, False)
    # Variant C: CNN + ConvLSTM (no alignment, no attention)
    c_psnr, c_ssim = run_ablation(False, False, True)
    # Variant D: Full model
    d_psnr, d_ssim = model_psnr, model_ssim

    ablation_rows = [
        ['A: Bicubic baseline',             f'{bic_psnr:.2f}',   f'{bic_ssim:.4f}'],
        ['B: CNN only (no LSTM, no align)', f'{b_psnr:.2f}',     f'{b_ssim:.4f}'],
        ['C: CNN + ConvLSTM',               f'{c_psnr:.2f}',     f'{c_ssim:.4f}'],
        ['D: Full model (ours)',             f'{d_psnr:.2f}',     f'{d_ssim:.4f}'],
    ]
    print_table('Ablation Study', ablation_rows, ['Variant', 'PSNR (dB)', 'SSIM'])

    logger.info("Evaluation complete.")

    # ── Comparison mode ────────────────────────────────────────────────────────
    if args.compare:
        run_comparison(args, config, device, test_loader, model)


if __name__ == '__main__':
    main()

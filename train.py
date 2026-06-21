"""
Training script for the Attention-Enhanced ConvLSTM VSR network.

Usage:
  python train.py --config configs/default.yaml
  python train.py --config configs/default.yaml --resume checkpoints/latest.pth
"""

import argparse
import csv
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import yaml

# ── Ada 6000 / TF32 optimisation ──────────────────────────────────────────────
torch.set_float32_matmul_precision('high')


def _nested_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a nested dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _nested_namespace(v) if isinstance(v, dict) else v)
    return ns


def load_config(path: str) -> SimpleNamespace:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _nested_namespace(raw)


def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = Path(log_dir) / f'train_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    return logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_lr_scheduler(optimizer, config, steps_per_epoch: int):
    """Linear warmup then CosineAnnealingLR."""
    warmup_steps = config.train.warmup_epochs * steps_per_epoch
    total_steps  = config.train.num_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)  # linear warm-up
        # Cosine decay from lr → lr_min
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        ratio    = config.train.lr_min / config.train.lr
        return ratio + (1 - ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def validate(model, val_loader, criterion, device, logger):
    """Run validation; returns dict with val loss and metrics."""
    from metrics.metrics import MetricTracker

    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in val_loader:
            lr_frames = batch['lr_frames'].to(device, non_blocking=True)
            hr_center = batch['hr_center'].to(device, non_blocking=True)
            center_lr = batch['center_lr'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sr_out = model(lr_frames, center_lr)
                losses = criterion(sr_out, hr_center)

            total_loss += losses['total'].item()
            n_batches  += 1
            tracker.update(sr_out.float().clamp(0, 1), hr_center.float())

    avg_loss = total_loss / max(n_batches, 1)
    stats    = tracker.summary()

    return {
        'val_loss': avg_loss,
        'val_psnr': stats['psnr_mean'],
        'val_ssim': stats['ssim_mean'],
    }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--resume', default=None, help='Path to checkpoint for resuming')
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config.logging.log_dir)
    set_seeds(42)

    # ── Hardware ──────────────────────────────────────────────────────────────
    device = torch.device(config.hardware.device if torch.cuda.is_available() else 'cpu')
    logger.info("Device: %s", device)

    torch.backends.cudnn.benchmark     = config.hardware.cudnn_benchmark
    torch.backends.cudnn.deterministic = config.hardware.cudnn_deterministic

    # ── Dataset guard ─────────────────────────────────────────────────────────
    train_csv = Path(config.data.train_manifest)
    val_csv   = Path(config.data.test_manifest)

    if not train_csv.exists() or not val_csv.exists():
        logger.warning(
            "Dataset not ready — training skipped.\n"
            "  Missing: %s\n"
            "Run: python scripts/prepare_dataset.py --zip <zip> --out_dir ./data/vimeo_prepared",
            ', '.join(
                str(p) for p in [train_csv, val_csv] if not p.exists()
            ),
        )
        sys.exit(0)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    from data.vimeo_dataset import get_dataloader

    train_loader = get_dataloader(str(train_csv), 'train', config)
    val_loader   = get_dataloader(str(val_csv),   'test',  config)

    if len(train_loader.dataset) == 0:
        logger.warning("Train dataset is empty — aborting.")
        sys.exit(0)

    logger.info(
        "Train samples: %d  |  Val samples: %d",
        len(train_loader.dataset), len(val_loader.dataset),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from models.vsr_net import VSRNet

    model = VSRNet(
        feature_channels=config.model.feature_channels,
        hidden_channels=config.model.hidden_channels,
        num_residual_blocks=config.model.num_residual_blocks,
        attention_ratio=config.model.attention_ratio,
        scale=config.data.scale,
        use_alignment=config.model.use_alignment,
        use_attention=config.model.use_attention,
        use_convlstm=config.model.use_convlstm,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f'{n_params:,}')

    # ── Loss, optimiser, scaler ───────────────────────────────────────────────
    from losses.losses import TotalLoss

    criterion = TotalLoss(
        l1_weight=config.loss.l1_weight,
        edge_weight=config.loss.edge_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )

    scaler    = torch.amp.GradScaler('cuda')
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    if config.logging.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = Path(config.logging.log_dir) / 'tensorboard'
            writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info("TensorBoard logs: %s", tb_dir)
        except ImportError:
            logger.warning("tensorboard not installed — skipping.")

    # ── Checkpoint resume ─────────────────────────────────────────────────────
    start_epoch = 0
    best_psnr   = 0.0
    ckpt_dir    = Path(config.logging.checkpoint_dir)

    if args.resume and Path(args.resume).exists():
        logger.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr   = ckpt.get('best_psnr', 0.0)
        logger.info("Resumed at epoch %d  best_psnr=%.4f", start_epoch, best_psnr)

    # ── Training log CSV ──────────────────────────────────────────────────────
    log_csv_path = Path(config.logging.log_dir) / 'training_log.csv'
    csv_fieldnames = [
        'epoch', 'train_loss_total', 'train_loss_l1', 'train_loss_edge',
        'val_psnr', 'val_ssim', 'lr', 'epoch_time_sec',
    ]
    csv_write_header = not log_csv_path.exists()
    log_csv_file = open(log_csv_path, 'a', newline='')
    csv_writer = csv.DictWriter(log_csv_file, fieldnames=csv_fieldnames)
    if csv_write_header:
        csv_writer.writeheader()

    # ── Main training loop ────────────────────────────────────────────────────
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, config.train.num_epochs):
        model.train()
        epoch_start = time.time()

        epoch_loss_total = 0.0
        epoch_loss_l1    = 0.0
        epoch_loss_edge  = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            lr_frames = batch['lr_frames'].to(device, non_blocking=True)
            hr_center = batch['hr_center'].to(device, non_blocking=True)
            center_lr = batch['center_lr'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                sr_out = model(lr_frames, center_lr)      # (B, 3, 4H, 4W)
                losses = criterion(sr_out, hr_center)

            scaler.scale(losses['total']).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss_total += losses['total'].item()
            epoch_loss_l1    += losses['l1'].item()
            epoch_loss_edge  += losses['edge'].item()
            n_batches += 1
            global_step += 1

            if batch_idx % config.logging.log_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(
                    "Epoch [%d/%d] Step [%d/%d] "
                    "loss=%.4f l1=%.4f edge=%.4f lr=%.2e",
                    epoch + 1, config.train.num_epochs,
                    batch_idx, len(train_loader),
                    losses['total'].item(), losses['l1'].item(),
                    losses['edge'].item(), current_lr,
                )

                if writer:
                    writer.add_scalar('train/loss_total', losses['total'].item(), global_step)
                    writer.add_scalar('train/loss_l1',    losses['l1'].item(),    global_step)
                    writer.add_scalar('train/loss_edge',  losses['edge'].item(),  global_step)
                    writer.add_scalar('train/lr',         current_lr,             global_step)

        epoch_time = time.time() - epoch_start
        avg_total  = epoch_loss_total / max(n_batches, 1)
        avg_l1     = epoch_loss_l1    / max(n_batches, 1)
        avg_edge   = epoch_loss_edge  / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]['lr']

        # ── Validation ───────────────────────────────────────────────────────
        val_psnr = 0.0
        val_ssim = 0.0
        if (epoch + 1) % config.logging.eval_every == 0 and len(val_loader.dataset) > 0:
            val_stats = validate(model, val_loader, criterion, device, logger)
            val_psnr  = val_stats['val_psnr']
            val_ssim  = val_stats['val_ssim']

            logger.info(
                "Epoch %d  val_psnr=%.4f  val_ssim=%.4f",
                epoch + 1, val_psnr, val_ssim,
            )

            if writer:
                writer.add_scalar('val/psnr', val_psnr, epoch)
                writer.add_scalar('val/ssim', val_ssim, epoch)

            # Save best
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(
                    {
                        'epoch':           epoch,
                        'model_state':     model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'best_psnr':       best_psnr,
                        'config':          vars(config),
                    },
                    ckpt_dir / 'best_model.pth',
                )
                logger.info("New best PSNR: %.4f dB — saved best_model.pth", best_psnr)

        # ── Periodic checkpoint ──────────────────────────────────────────────
        if (epoch + 1) % config.logging.save_every == 0:
            save_checkpoint(
                {
                    'epoch':           epoch,
                    'model_state':     model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_psnr':       best_psnr,
                    'config':          vars(config),
                },
                ckpt_dir / f'epoch_{epoch + 1:04d}.pth',
            )

        # Always save latest
        save_checkpoint(
            {
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_psnr':       best_psnr,
                'config':          vars(config),
            },
            ckpt_dir / 'latest.pth',
        )

        # ── CSV log ──────────────────────────────────────────────────────────
        csv_writer.writerow({
            'epoch':            epoch + 1,
            'train_loss_total': f'{avg_total:.6f}',
            'train_loss_l1':    f'{avg_l1:.6f}',
            'train_loss_edge':  f'{avg_edge:.6f}',
            'val_psnr':         f'{val_psnr:.4f}',
            'val_ssim':         f'{val_ssim:.4f}',
            'lr':               f'{current_lr:.2e}',
            'epoch_time_sec':   f'{epoch_time:.1f}',
        })
        log_csv_file.flush()

        logger.info(
            "Epoch %d done — loss=%.4f  psnr=%.4f  ssim=%.4f  time=%.1fs",
            epoch + 1, avg_total, val_psnr, val_ssim, epoch_time,
        )

    log_csv_file.close()
    if writer:
        writer.close()

    logger.info("Training complete. Best PSNR: %.4f dB", best_psnr)


if __name__ == '__main__':
    train()

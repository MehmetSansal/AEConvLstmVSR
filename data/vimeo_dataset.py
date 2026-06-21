"""
Vimeo-90K septuplet dataset loader.

Reads from manifests produced by scripts/prepare_dataset.py.
If the manifest does not exist, the dataset reports length 0 and logs a warning —
it never crashes, allowing the DataLoader to be constructed safely without data.
"""

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class Vimeo90KDataset(Dataset):
    """
    Loads 7 LR frames + 1 HR center frame per sample.

    Returns:
      {
        'lr_frames':   FloatTensor (7, 3, H_lr, W_lr),
        'hr_center':   FloatTensor (3, H_hr, W_hr),
        'center_lr':   FloatTensor (3, H_lr, W_lr),    # lr/im4.png
        'sequence_id': str,
      }
    """

    def __init__(
        self,
        manifest_csv: str,
        patch_size: int = 256,
        scale: int = 4,
        augment: bool = True,
    ):
        self.patch_size = patch_size
        self.lr_patch = patch_size // scale
        self.augment = augment
        self._empty = False
        self.records = []

        csv_path = Path(manifest_csv)
        if not csv_path.exists():
            logger.warning(
                "Manifest not found: %s — dataset will report 0 samples. "
                "Run scripts/prepare_dataset.py once the Vimeo-90K zip is downloaded.",
                manifest_csv,
            )
            self._empty = True
            return

        df = pd.read_csv(csv_path)
        self.records = df.to_dict('records')
        logger.info("Loaded %d sequences from %s", len(self.records), manifest_csv)

    def __len__(self) -> int:
        return 0 if self._empty else len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        seq_id = str(rec['sequence_id'])
        hr_dir = Path(rec['hr_dir'])
        lr_dir = Path(rec['lr_dir'])

        # ── Load all 7 LR frames and HR center ──────────────────────────────
        lr_imgs = []
        for i in range(1, 8):
            path = lr_dir / f'im{i}.png'
            lr_imgs.append(np.array(Image.open(path).convert('RGB'), dtype=np.uint8))

        hr_img = np.array(Image.open(hr_dir / 'im4.png').convert('RGB'), dtype=np.uint8)

        # ── Crop ─────────────────────────────────────────────────────────────
        lr_h, lr_w = lr_imgs[0].shape[:2]
        hr_h, hr_w = hr_img.shape[:2]

        # Random (train) or center (test) crop
        if self.augment:
            # Ensure patch fits; clamp if source is smaller than patch
            max_y_lr = max(lr_h - self.lr_patch, 0)
            max_x_lr = max(lr_w - self.lr_patch, 0)
            top_lr  = random.randint(0, max_y_lr)
            left_lr = random.randint(0, max_x_lr)
        else:
            top_lr  = (lr_h - self.lr_patch) // 2
            left_lr = (lr_w - self.lr_patch) // 2

        top_lr  = max(top_lr,  0)
        left_lr = max(left_lr, 0)

        # Corresponding HR crop positions (scale-aligned)
        scale = self.patch_size // self.lr_patch
        top_hr  = top_lr  * scale
        left_hr = left_lr * scale

        def crop_lr(img):
            return img[top_lr:top_lr + self.lr_patch,
                       left_lr:left_lr + self.lr_patch]

        def crop_hr(img):
            return img[top_hr:top_hr + self.patch_size,
                       left_hr:left_hr + self.patch_size]

        lr_crops = [crop_lr(im) for im in lr_imgs]  # list of (H_lr, W_lr, 3)
        hr_crop  = crop_hr(hr_img)                   # (H_hr, W_hr, 3)

        # ── Augmentation ─────────────────────────────────────────────────────
        if self.augment:
            # Horizontal flip
            if random.random() < 0.5:
                lr_crops = [np.fliplr(im).copy() for im in lr_crops]
                hr_crop  = np.fliplr(hr_crop).copy()

            # Vertical flip
            if random.random() < 0.5:
                lr_crops = [np.flipud(im).copy() for im in lr_crops]
                hr_crop  = np.flipud(hr_crop).copy()

            # Random 90° rotation (k=1 → 90°, applied identically to all)
            if random.random() < 0.5:
                k = random.randint(1, 3)
                lr_crops = [np.rot90(im, k).copy() for im in lr_crops]
                hr_crop  = np.rot90(hr_crop, k).copy()

        # ── To tensor, [0, 1] ─────────────────────────────────────────────
        def to_tensor(img_np):
            # (H, W, 3) uint8 → (3, H, W) float32 in [0,1]
            return torch.from_numpy(img_np.copy()).permute(2, 0, 1).float() / 255.0

        lr_tensors = torch.stack([to_tensor(im) for im in lr_crops], dim=0)  # (7, 3, H_lr, W_lr)
        hr_tensor  = to_tensor(hr_crop)                                        # (3, H_hr, W_hr)
        center_lr  = lr_tensors[3]                                             # (3, H_lr, W_lr) — im4

        return {
            'lr_frames':   lr_tensors,
            'hr_center':   hr_tensor,
            'center_lr':   center_lr,
            'sequence_id': seq_id,
        }


def get_dataloader(
    manifest_csv: str,
    split: str,
    config,
    patch_size: Optional[int] = None,
) -> DataLoader:
    """
    Factory function that creates a DataLoader optimised for the Ada 6000 workstation.

    Args:
        manifest_csv: Path to train or test manifest CSV.
        split:        'train' or 'test'.
        config:       OmegaConf / SimpleNamespace config object.
        patch_size:   Override patch size (uses config.data.patch_size by default).
    """
    ps = patch_size or config.data.patch_size
    scale = getattr(config.data, 'scale', 4)
    augment = (split == 'train')

    dataset = Vimeo90KDataset(
        manifest_csv=manifest_csv,
        patch_size=ps,
        scale=scale,
        augment=augment,
    )

    num_workers = config.data.num_workers
    # persistent_workers requires num_workers > 0
    persistent = (num_workers > 0)

    loader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=(split == 'train'),
        persistent_workers=persistent,
        prefetch_factor=config.data.prefetch_factor if num_workers > 0 else None,
    )

    return loader

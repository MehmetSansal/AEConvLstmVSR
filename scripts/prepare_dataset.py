"""
Dataset preparation script for Vimeo-90K septuplet.

Kullanım (zip yok, doğrudan klasör):

  python scripts/prepare_dataset.py \
    --src_dir  ./archive \
    --out_dir  ./data/vimeo_prepared \
    --scale    4 \
    --workers  16

Beklenen kaynak yapısı:
  archive/
    sequences/
      00001/
        0001/  im1.png … im7.png
        0002/  ...
        ...
        1000/
      00002/
        ...
      00090/
        ...
    sep_trainlist.txt   ← opsiyonel; yoksa dizin taranır
    sep_testlist.txt    ← opsiyonel; yoksa dizin taranır
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ─── Worker (top-level for multiprocessing) ───────────────────────────────────

def _process_sequence(args: tuple) -> Tuple[str, str, str, bool]:
    """
    Tek bir sekans için HR'ı kopyalar, LR versiyonunu üretir.

    args: (seq_id, src_dir, hr_dir, lr_dir, scale)
    Returns: (seq_id, hr_dir, lr_dir, success)
    """
    seq_id, src_dir, hr_dir, lr_dir, scale = args
    src_dir = Path(src_dir)
    hr_dir  = Path(hr_dir)
    lr_dir  = Path(lr_dir)

    try:
        hr_dir.mkdir(parents=True, exist_ok=True)
        lr_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, 8):
            src_path = src_dir / f'im{i}.png'
            if not src_path.exists():
                logger.warning("Eksik kare: %s", src_path)
                return seq_id, str(hr_dir), str(lr_dir), False

            hr_img   = Image.open(src_path).convert('RGB')
            hr_w, hr_h = hr_img.size

            # HR'ı olduğu gibi kaydet
            hr_img.save(hr_dir / f'im{i}.png')

            # LR: Lanczos ile bicubic küçültme
            lr_w = hr_w // scale
            lr_h = hr_h // scale
            lr_img = hr_img.resize((lr_w, lr_h), Image.LANCZOS)
            lr_img.save(lr_dir / f'im{i}.png')

        return seq_id, str(hr_dir), str(lr_dir), True

    except Exception as exc:  # noqa: BLE001
        logger.error("İşlenemedi %s: %s", seq_id, exc)
        return seq_id, str(hr_dir), str(lr_dir), False


# ─── Dizinden sekans listesi bul ──────────────────────────────────────────────

def discover_sequences(sequences_root: Path) -> List[str]:
    """
    sequences/ altındaki tüm 7-framelı sekansları tarar.
    00001/0001 formatında liste döner.
    """
    lines = []
    for group_dir in sorted(sequences_root.iterdir()):
        if not group_dir.is_dir():
            continue
        for seq_dir in sorted(group_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            # En az im1.png varsa geçerli sekans say
            if (seq_dir / 'im1.png').exists():
                lines.append(f'{group_dir.name}/{seq_dir.name}')
    return lines


def read_list_file(list_path: Path) -> List[str]:
    with open(list_path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def split_train_test(all_lines: List[str], test_ratio: float = 0.1) -> Tuple[List[str], List[str]]:
    """sep_testlist.txt yoksa %10'unu test olarak ayır."""
    random.seed(42)
    shuffled = all_lines[:]
    random.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    return shuffled[n_test:], shuffled[:n_test]


# ─── İşleme ───────────────────────────────────────────────────────────────────

def build_worker_args(
    seq_lines: List[str],
    sequences_root: Path,
    out_split_dir: Path,
    scale: int,
) -> list:
    args = []
    for line in seq_lines:
        # "00001/0001" → seq_id = "00001_0001"
        seq_id  = line.replace('/', '_')
        src_dir = sequences_root / line
        hr_dir  = out_split_dir / seq_id / 'hr'
        lr_dir  = out_split_dir / seq_id / 'lr'
        args.append((seq_id, str(src_dir), str(hr_dir), str(lr_dir), scale))
    return args


def process_split(
    split_name: str,
    seq_lines: List[str],
    sequences_root: Path,
    out_dir: Path,
    scale: int,
    workers: int,
) -> List[dict]:
    out_split_dir = out_dir / split_name
    worker_args   = build_worker_args(seq_lines, sequences_root, out_split_dir, scale)

    logger.info(
        "%s için %d sekans %d worker ile işleniyor ...",
        split_name, len(worker_args), workers,
    )

    manifest_rows = []
    failed = 0

    with mp.Pool(processes=workers) as pool:
        for seq_id, hr_dir, lr_dir, ok in tqdm(
            pool.imap_unordered(_process_sequence, worker_args),
            total=len(worker_args),
            desc=f'Rendering {split_name}',
        ):
            if ok:
                manifest_rows.append({
                    'sequence_id': seq_id,
                    'hr_dir':      hr_dir,
                    'lr_dir':      lr_dir,
                })
            else:
                failed += 1

    logger.info("%s: %d başarılı, %d hatalı.", split_name, len(manifest_rows), failed)
    return manifest_rows


def write_manifest(rows: List[dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sequence_id', 'hr_dir', 'lr_dir'])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Manifest yazıldı: %s (%d satır)", csv_path, len(rows))


def verify_samples(manifest_rows: List[dict], n_samples: int = 50, scale: int = 4):
    sample = random.sample(manifest_rows, min(n_samples, len(manifest_rows)))
    logger.info("%d rastgele sekans doğrulanıyor ...", len(sample))

    errors = 0
    hr_sizes, lr_sizes = [], []

    for row in sample:
        hr_center = Path(row['hr_dir']) / 'im4.png'
        lr_center = Path(row['lr_dir']) / 'im4.png'

        if not hr_center.exists() or not lr_center.exists():
            logger.error("Eksik dosya: %s", row['sequence_id'])
            errors += 1
            continue

        hr_w, hr_h = Image.open(hr_center).size
        lr_w, lr_h = Image.open(lr_center).size

        exp_lr_w = hr_w // scale
        exp_lr_h = hr_h // scale

        if lr_w != exp_lr_w or lr_h != exp_lr_h:
            logger.error(
                "Boyut uyumsuzluğu %s: HR=%dx%d, LR=%dx%d (beklenen %dx%d)",
                row['sequence_id'], hr_w, hr_h, lr_w, lr_h, exp_lr_w, exp_lr_h,
            )
            errors += 1
        else:
            hr_sizes.append((hr_h, hr_w))
            lr_sizes.append((lr_h, lr_w))

    if errors:
        logger.error("Doğrulama BAŞARISIZ: %d hata.", errors)
    else:
        logger.info("Doğrulama geçti. Tüm örnekler tutarlı.")
        if hr_sizes:
            logger.info("HR boyut aralığı: %s – %s", min(hr_sizes), max(hr_sizes))
            logger.info("LR boyut aralığı: %s – %s", min(lr_sizes), max(lr_sizes))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Vimeo-90K septuplet veri seti hazırlama (doğrudan klasörden)'
    )
    parser.add_argument(
        '--src_dir', required=True,
        help='Vimeo-90K kök klasörü (içinde sequences/ alt klasörü olmalı)',
    )
    parser.add_argument('--out_dir',  default='./data/vimeo_prepared', help='Çıktı dizini')
    parser.add_argument('--scale',    type=int, default=4,  help='LR küçültme katsayısı')
    parser.add_argument('--workers',  type=int, default=16, help='CPU worker sayısı')
    parser.add_argument(
        '--test_ratio', type=float, default=0.1,
        help='sep_testlist.txt yoksa test oranı (varsayılan 0.10)',
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    if not src_dir.exists():
        logger.error("Kaynak klasör bulunamadı: %s", src_dir)
        sys.exit(1)

    # sequences/ klasörünü bul
    sequences_root = src_dir / 'sequences'
    if not sequences_root.exists():
        # Bazı arşivlerde sequences/ olmadan doğrudan 00001/ klasörleri olabilir
        if any(p.is_dir() and p.name.isdigit() for p in src_dir.iterdir()):
            sequences_root = src_dir
            logger.warning(
                "'sequences/' alt klasörü yok; %s doğrudan sequences root olarak kullanılıyor.",
                src_dir,
            )
        else:
            logger.error(
                "'sequences/' alt klasörü bulunamadı: %s\n"
                "Klasör yapısı: archive/sequences/00001/0001/im*.png olmalı.",
                src_dir,
            )
            sys.exit(1)

    logger.info("Sequences root: %s", sequences_root)

    # ── Liste dosyalarını bul ya da dizini tara ────────────────────────────
    train_list_path = src_dir / 'sep_trainlist.txt'
    test_list_path  = src_dir / 'sep_testlist.txt'

    if train_list_path.exists() and test_list_path.exists():
        logger.info("Liste dosyaları bulundu: %s, %s", train_list_path, test_list_path)
        train_lines = read_list_file(train_list_path)
        test_lines  = read_list_file(test_list_path)
    else:
        logger.info(
            "Liste dosyaları bulunamadı — sequences/ dizini taranıyor ..."
        )
        all_lines = discover_sequences(sequences_root)
        if not all_lines:
            logger.error(
                "sequences/ altında geçerli sekans bulunamadı. "
                "im1.png–im7.png dosyaları var mı?"
            )
            sys.exit(1)

        logger.info("Toplam %d sekans keşfedildi.", len(all_lines))
        train_lines, test_lines = split_train_test(all_lines, args.test_ratio)
        logger.info(
            "Otomatik bölünme: %d train, %d test (%.0f%%)",
            len(train_lines), len(test_lines), args.test_ratio * 100,
        )

    logger.info("Train sekans sayısı: %d", len(train_lines))
    logger.info("Test  sekans sayısı: %d", len(test_lines))

    # ── HR → LR render ────────────────────────────────────────────────────────
    train_rows = process_split('train', train_lines, sequences_root, out_dir, args.scale, args.workers)
    test_rows  = process_split('test',  test_lines,  sequences_root, out_dir, args.scale, args.workers)

    # ── Manifest CSV'leri yaz ─────────────────────────────────────────────────
    write_manifest(train_rows, out_dir / 'train_manifest.csv')
    write_manifest(test_rows,  out_dir / 'test_manifest.csv')

    # dataset_info.json
    hr_size, lr_size = [0, 0], [0, 0]
    for row in (train_rows + test_rows):
        hp = Path(row['hr_dir']) / 'im4.png'
        lp = Path(row['lr_dir']) / 'im4.png'
        if hp.exists() and lp.exists():
            hw, hh = Image.open(hp).size
            lw, lh = Image.open(lp).size
            hr_size = [hh, hw]
            lr_size = [lh, lw]
            break

    info = {
        'num_train':   len(train_rows),
        'num_test':    len(test_rows),
        'scale':       args.scale,
        'hr_size':     hr_size,
        'lr_size':     lr_size,
        'prepared_at': datetime.now(timezone.utc).isoformat(),
    }
    info_path = out_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    logger.info("Dataset bilgisi yazıldı: %s", info_path)

    # ── Doğrulama ─────────────────────────────────────────────────────────────
    all_rows = train_rows + test_rows
    if all_rows:
        verify_samples(all_rows, n_samples=50, scale=args.scale)

    logger.info(
        "\nHazırlık tamamlandı.\n"
        "  Train: %d sekans\n"
        "  Test:  %d sekans\n"
        "  Ölçek: ×%d\n"
        "  Çıktı: %s",
        len(train_rows), len(test_rows), args.scale, out_dir,
    )


if __name__ == '__main__':
    main()

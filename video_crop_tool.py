"""
Interactive Video Crop & Dataset Creator
Builds Vimeo-90K septuplet test datasets from any video file.
Output is directly consumable by evaluate.py (test_manifest.csv format).

Usage:
    cd video_sr
    python video_crop_tool.py
    python video_crop_tool.py --out_dir ./data/vimeo_prepared
    python video_crop_tool.py myvideo.mp4
"""

import sys
import csv
import argparse
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QLineEdit, QFileDialog, QSpinBox,
    QGroupBox, QGridLayout, QProgressBar, QSizePolicy, QMessageBox,
    QDialog, QDialogButtonBox,
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont


# Vimeo-90K septuplet defaults
VIMEO_W = 448
VIMEO_H = 256
SCALE   = 4


# ─── YouTube Download Worker ──────────────────────────────────────────────────

class YoutubeDownloadWorker(QThread):
    """
    Downloads a YouTube (or any yt-dlp-supported) URL to a local file.

    YouTube blocks yt-dlp without browser cookies since 2024.
    We try a prioritised list of browsers until one works.
    """

    progress = pyqtSignal(str)
    done     = pyqtSignal(bool, str)

    # Browsers tried in order when browser='auto'
    BROWSER_ORDER = ['chrome', 'chromium', 'firefox', 'edge', 'brave', 'opera']

    def __init__(self, url: str, out_dir: str, browser: str = 'auto'):
        super().__init__()
        self.url     = url
        self.out_dir = out_dir
        self.browser = browser   # 'auto' | 'chrome' | 'firefox' | 'none' | …

    def _make_opts(self, browser: Optional[str]) -> dict:
        opts: dict = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
            'outtmpl': os.path.join(self.out_dir, '%(title).60s.%(ext)s'),
            'merge_output_format': 'mp4',
            'progress_hooks': [self._hook],
            'quiet': True,
            'no_warnings': True,
        }
        if browser:
            opts['cookiesfrombrowser'] = (browser, None, None, None)
        return opts

    def _hook(self, d: dict):
        status = d.get('status', '')
        if status == 'downloading':
            pct   = d.get('_percent_str', '?').strip()
            speed = d.get('_speed_str', '').strip()
            eta   = d.get('_eta_str', '').strip()
            self.progress.emit(f"İndiriliyor … {pct}  {speed}  ETA {eta}")
        elif status == 'finished':
            self._last_file = d.get('filename') or ''
            self.progress.emit("Birleştiriliyor …")

    def _resolve_final(self, ydl, info: dict) -> str:
        final = ydl.prepare_filename(info)
        if not os.path.exists(final):
            final = os.path.splitext(final)[0] + '.mp4'
        if hasattr(self, '_last_file') and self._last_file and os.path.exists(self._last_file):
            final = self._last_file
        return final

    def run(self):
        try:
            import yt_dlp  # noqa: PLC0415
        except ImportError:
            self.done.emit(False, "yt-dlp kurulu değil.\nKomut:  pip install yt-dlp")
            return

        # Build the list of browsers to attempt
        if self.browser == 'auto':
            attempts: List[Optional[str]] = self.BROWSER_ORDER + [None]
        elif self.browser == 'none':
            attempts = [None]
        else:
            attempts = [self.browser, None]

        last_err = 'Bilinmeyen hata'
        for browser in attempts:
            label = browser or 'cookie yok'
            self.progress.emit(f"Deneniyor: {label} …")
            self._last_file = ''
            try:
                opts = self._make_opts(browser)
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info  = ydl.extract_info(self.url, download=True)
                    final = self._resolve_final(ydl, info)
                self.done.emit(True, final)
                return
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
                # Don't retry on non-availability errors (geo-block, private video)
                if 'not available' in last_err.lower() and browser is not None:
                    # Still worth trying without that browser's cookies
                    continue

        self.done.emit(False, last_err)


# ─── YouTube Dialog ───────────────────────────────────────────────────────────

class YoutubeDialog(QDialog):
    """
    Dialog: paste URL, choose download folder + browser cookie source, download.

    YouTube requires browser cookies since 2024 to bypass bot-detection.
    "Otomatik" modu tüm tarayıcıları sırayla dener.
    """

    downloaded = pyqtSignal(str)

    BROWSER_OPTIONS = [
        ("Otomatik (Chrome → Firefox → …)", "auto"),
        ("Chrome",    "chrome"),
        ("Chromium",  "chromium"),
        ("Firefox",   "firefox"),
        ("Edge",      "edge"),
        ("Cookie yok (genellikle çalışmaz)", "none"),
    ]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("YouTube / Video İndir")
        self.setMinimumWidth(600)
        self._worker: Optional[YoutubeDownloadWorker] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── URL ───────────────────────────────────────────────────────────────
        layout.addWidget(QLabel("YouTube / video URL:"))
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("https://www.youtube.com/watch?v=…")
        layout.addWidget(self._url_edit)

        # ── Save dir ──────────────────────────────────────────────────────────
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Kayıt klasörü:"))
        self._dir_edit = QLineEdit(str(Path.home() / "Downloads"))
        dir_row.addWidget(self._dir_edit, stretch=1)
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(28)
        btn_browse.clicked.connect(self._browse_dir)
        dir_row.addWidget(btn_browse)
        layout.addLayout(dir_row)

        # ── Browser cookie selector ────────────────────────────────────────────
        from PyQt5.QtWidgets import QComboBox  # noqa: PLC0415
        browser_row = QHBoxLayout()
        browser_row.addWidget(QLabel("Tarayıcı cookie:"))
        self._browser_combo = QComboBox()
        for label, _ in self.BROWSER_OPTIONS:
            self._browser_combo.addItem(label)
        self._browser_combo.setCurrentIndex(0)  # default: auto
        browser_row.addWidget(self._browser_combo, stretch=1)
        layout.addLayout(browser_row)

        note = QLabel(
            "YouTube, cookie olmadan indirmeyi engelliyor (bot koruması).\n"
            "Seçilen tarayıcının açık olmaması gerekmiyor — profil verisi kullanılır."
        )
        note.setStyleSheet("color:#888; font-size:10px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        # ── Progress ──────────────────────────────────────────────────────────
        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        layout.addWidget(self._progress_lbl)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # ── Buttons ───────────────────────────────────────────────────────────
        self._btn_box = QDialogButtonBox()
        self._dl_btn  = self._btn_box.addButton("İndir", QDialogButtonBox.AcceptRole)
        self._btn_box.addButton("İptal", QDialogButtonBox.RejectRole)
        self._dl_btn.clicked.connect(self._start_download)
        self._btn_box.rejected.connect(self._cancel)
        layout.addWidget(self._btn_box)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Kayıt klasörü seç …")
        if d:
            self._dir_edit.setText(d)

    def _selected_browser(self) -> str:
        idx = self._browser_combo.currentIndex()
        return self.BROWSER_OPTIONS[idx][1]

    def _start_download(self):
        url = self._url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "URL yok", "Önce bir YouTube URL'si yapıştırın.")
            return

        save_dir = self._dir_edit.text().strip() or str(Path.home() / "Downloads")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self._dl_btn.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_lbl.setText("Başlatılıyor …")

        self._worker = YoutubeDownloadWorker(url, save_dir, browser=self._selected_browser())
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_progress(self, msg: str):
        self._progress_lbl.setText(msg)

    def _on_done(self, ok: bool, result: str):
        self._progress_bar.setVisible(False)
        self._dl_btn.setEnabled(True)
        if ok:
            self._progress_lbl.setText(f"Kaydedildi: {result}")
            self.downloaded.emit(result)
            QMessageBox.information(
                self, "İndirme Tamamlandı",
                f"Video kaydedildi:\n{result}\n\nOtomatik olarak yüklenecek."
            )
            self.accept()
        else:
            self._progress_lbl.setText(f"Hata: {result}")
            QMessageBox.critical(
                self, "İndirme Başarısız",
                f"{result}\n\n"
                "Öneriler:\n"
                "• Farklı bir tarayıcı cookie seçin\n"
                "• Video herkese açık mı kontrol edin\n"
                "• yt-dlp güncel mi?  →  pip install -U yt-dlp",
            )

    def _cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
        self.reject()


# ─── Export Worker ────────────────────────────────────────────────────────────

class ExportWorker(QThread):
    """Saves 7 cropped frames as HR + LR PNGs and updates test_manifest.csv."""

    progress = pyqtSignal(int, str)
    done     = pyqtSignal(bool, str)

    def __init__(
        self,
        frames:  List[np.ndarray],
        crop:    Tuple[int, int, int, int],
        out_dir: str,
        seq_id:  str,
        scale:   int = SCALE,
    ):
        super().__init__()
        self.frames  = frames   # 7 BGR numpy arrays (full video frame)
        self.crop    = crop     # (x, y, w, h) in video pixels
        self.out_dir = Path(out_dir)
        self.seq_id  = seq_id
        self.scale   = scale

    def run(self):
        try:
            x, y, w, h = self.crop
            seq_dir = self.out_dir / 'test' / self.seq_id
            hr_dir  = seq_dir / 'hr'
            lr_dir  = seq_dir / 'lr'
            hr_dir.mkdir(parents=True, exist_ok=True)
            lr_dir.mkdir(parents=True, exist_ok=True)

            lw, lh = w // self.scale, h // self.scale

            for i, bgr in enumerate(self.frames):
                self.progress.emit(i * 12, f"Saving frame {i + 1}/7 …")

                crop_bgr = bgr[y:y + h, x:x + w]
                hr_rgb   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                hr_pil   = Image.fromarray(hr_rgb)
                hr_pil.save(hr_dir / f'im{i + 1}.png')

                lr_pil = hr_pil.resize((lw, lh), Image.LANCZOS)
                lr_pil.save(lr_dir / f'im{i + 1}.png')

            self.progress.emit(90, "Updating manifest …")

            manifest = self.out_dir / 'test_manifest.csv'
            rows: List[dict] = []
            if manifest.exists():
                with open(manifest, newline='') as f:
                    rows = [r for r in csv.DictReader(f)
                            if r.get('sequence_id') != self.seq_id]
            rows.append({
                'sequence_id': self.seq_id,
                'hr_dir':      str(hr_dir.resolve()),
                'lr_dir':      str(lr_dir.resolve()),
            })
            with open(manifest, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f, fieldnames=['sequence_id', 'hr_dir', 'lr_dir']
                )
                writer.writeheader()
                writer.writerows(rows)

            self.progress.emit(100, "Done")
            self.done.emit(
                True,
                f"Sequence saved:\n{seq_dir}\n\nManifest updated:\n{manifest}\n\n"
                f"HR: {w}×{h}  LR: {lw}×{lh}  (×{self.scale})\n"
                f"Run:  python evaluate.py --checkpoint checkpoints/best_model.pth",
            )
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


# ─── Video Preview Widget ─────────────────────────────────────────────────────

class VideoPreview(QLabel):
    """
    Displays a video frame with a draggable yellow crop rectangle overlay.

    Interactions:
      - Click outside box  → teleport box centre to click point
      - Drag inside box    → move box
      - Drag bottom-right corner → resize box
    """

    cropChanged = pyqtSignal(int, int, int, int)   # x, y, w, h (video pixels)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(720, 450)
        self.setAlignment(Qt.AlignCenter)
        self._set_empty_style()
        self.setText("Load a video file to begin")
        self.setFont(QFont("Arial", 13))
        self.setMouseTracking(True)

        self._frame: Optional[np.ndarray] = None
        self._vw = 1920
        self._vh = 1080

        # Crop in video-pixel coordinates
        self.cx: int = 0
        self.cy: int = 0
        self.cw: int = VIMEO_W
        self.ch: int = VIMEO_H

        self._dragging  = False
        self._drag_mode = 'move'           # 'move' | 'resize'
        self._d_start   = (0, 0)
        self._d_crop0   = (0, 0, VIMEO_W, VIMEO_H)

    def _set_empty_style(self):
        self.setStyleSheet("background:#111; border:1px solid #444; color:#666;")

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _scaled_rect(self) -> Tuple[int, int, int, int]:
        """Returns (ox, oy, sw, sh) — where the letterboxed frame sits in the widget."""
        if self._frame is None:
            return 0, 0, self.width(), self.height()
        asp = self._vw / self._vh
        ww, wh = self.width(), self.height()
        if ww / wh > asp:
            sh, sw = wh, int(wh * asp)
        else:
            sw, sh = ww, int(ww / asp)
        return (ww - sw) // 2, (wh - sh) // 2, sw, sh

    def _w2v(self, wx: int, wy: int) -> Tuple[int, int]:
        ox, oy, sw, sh = self._scaled_rect()
        return (
            max(0, min(int((wx - ox) * self._vw / sw), self._vw - 1)),
            max(0, min(int((wy - oy) * self._vh / sh), self._vh - 1)),
        )

    def _v2s(self, vx: int, vy: int) -> Tuple[int, int]:
        """Video coords → coordinates within the scaled pixmap."""
        ox, oy, sw, sh = self._scaled_rect()
        return int(vx * sw / self._vw), int(vy * sh / self._vh)

    # ── Frame display ─────────────────────────────────────────────────────────

    def set_frame(self, frame_bgr: np.ndarray):
        self._frame = frame_bgr
        self._vh, self._vw = frame_bgr.shape[:2]
        self.setStyleSheet("background:#111; border:1px solid #444;")
        self._render()

    def _render(self):
        if self._frame is None:
            return
        ox, oy, sw, sh = self._scaled_rect()

        rgb   = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        pm    = QPixmap.fromImage(
            QImage(small.data, sw, sh, sw * 3, QImage.Format_RGB888)
        )

        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        # Crop corners in scaled-pixmap space
        rx,  ry  = self._v2s(self.cx, self.cy)
        rx2, ry2 = self._v2s(self.cx + self.cw, self.cy + self.ch)
        rw,  rh  = rx2 - rx, ry2 - ry

        # Dim the area outside the crop
        dim = QColor(0, 0, 0, 120)
        p.fillRect(0,       0,       sw,             ry,          dim)
        p.fillRect(0,       ry + rh, sw,             sh - ry - rh, dim)
        p.fillRect(0,       ry,      rx,             rh,          dim)
        p.fillRect(rx + rw, ry,      sw - rx - rw,   rh,          dim)

        # Crop border
        p.setPen(QPen(QColor(255, 220, 0), 2))
        p.drawRect(rx, ry, rw, rh)

        # Corner handles (yellow squares)
        hs  = 8
        yel = QColor(255, 220, 0)
        for hx, hy in [(rx, ry), (rx + rw, ry), (rx, ry + rh), (rx + rw, ry + rh)]:
            p.fillRect(hx - hs // 2, hy - hs // 2, hs, hs, yel)

        # Info text
        p.setPen(QColor(255, 220, 0))
        p.setFont(QFont("monospace", 9, QFont.Bold))
        lw, lh = self.cw // SCALE, self.ch // SCALE
        p.drawText(rx + 4, ry + 15, f"HR {self.cw}×{self.ch}  →  LR {lw}×{lh}")
        p.drawText(rx + 4, ry + 29, f"origin ({self.cx}, {self.cy})")
        p.end()

        canvas = QPixmap(self.width(), self.height())
        canvas.fill(QColor(17, 17, 17))
        p2 = QPainter(canvas)
        p2.drawPixmap(ox, oy, pm)
        p2.end()
        self.setPixmap(canvas)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if self._frame is None or event.button() != Qt.LeftButton:
            return
        wx, wy = event.x(), event.y()
        ox, oy, sw, sh = self._scaled_rect()

        # Crop box corners in widget coords
        rx  = ox + int(self.cx * sw / self._vw)
        ry  = oy + int(self.cy * sh / self._vh)
        rw  = int(self.cw * sw / self._vw)
        rh  = int(self.ch * sh / self._vh)
        ht  = 14   # hit tolerance in pixels

        at_br  = abs(wx - (rx + rw)) < ht and abs(wy - (ry + rh)) < ht
        inside = rx <= wx <= rx + rw and ry <= wy <= ry + rh

        if at_br:
            self._drag_mode = 'resize'
        elif inside:
            self._drag_mode = 'move'
        else:
            # Teleport crop centre to click
            vx, vy = self._w2v(wx, wy)
            self.cx = max(0, min(vx - self.cw // 2, self._vw - self.cw))
            self.cy = max(0, min(vy - self.ch // 2, self._vh - self.ch))
            self._render()
            self.cropChanged.emit(self.cx, self.cy, self.cw, self.ch)
            self._drag_mode = 'move'

        self._dragging = True
        self._d_start  = (wx, wy)
        self._d_crop0  = (self.cx, self.cy, self.cw, self.ch)

    def mouseMoveEvent(self, event):
        if not self._dragging or self._frame is None:
            return
        ox, oy, sw, sh = self._scaled_rect()
        dx_v = int((event.x() - self._d_start[0]) * self._vw / sw)
        dy_v = int((event.y() - self._d_start[1]) * self._vh / sh)
        x0, y0, w0, h0 = self._d_crop0

        if self._drag_mode == 'move':
            self.cx = max(0, min(x0 + dx_v, self._vw - self.cw))
            self.cy = max(0, min(y0 + dy_v, self._vh - self.ch))
        else:
            self.cw = max(SCALE * 4, min(w0 + dx_v, self._vw - self.cx))
            self.ch = max(SCALE * 4, min(h0 + dy_v, self._vh - self.cy))

        self._render()
        self.cropChanged.emit(self.cx, self.cy, self.cw, self.ch)

    def mouseReleaseEvent(self, event):
        self._dragging = False

    # ── External setters ──────────────────────────────────────────────────────

    def set_crop(self, x: int, y: int, w: int, h: int):
        self.cw = max(SCALE, min(w, self._vw))
        self.ch = max(SCALE, min(h, self._vh))
        self.cx = max(0, min(x, self._vw - self.cw))
        self.cy = max(0, min(y, self._vh - self.ch))
        self._render()

    def center_crop(self):
        self.cx = max(0, (self._vw - self.cw) // 2)
        self.cy = max(0, (self._vh - self.ch) // 2)
        self._render()
        self.cropChanged.emit(self.cx, self.cy, self.cw, self.ch)

    def reset_size(self):
        self.cw, self.ch = VIMEO_W, VIMEO_H
        self.cx = max(0, min(self.cx, self._vw - self.cw))
        self.cy = max(0, min(self.cy, self._vh - self.ch))
        self._render()
        self.cropChanged.emit(self.cx, self.cy, self.cw, self.ch)


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self, out_dir: str = './data/vimeo_prepared'):
        super().__init__()
        self.setWindowTitle("Video Crop Tool — Vimeo-90K Dataset Builder")
        self.resize(1400, 920)

        self._out_dir     = out_dir
        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames = 0
        self._fps          = 30.0
        self._selected_frames: List[Optional[np.ndarray]] = [None] * 7
        self._export_worker: Optional[ExportWorker] = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        root.addLayout(self._build_file_bar())
        root.addWidget(self._build_center_area(), stretch=1)
        root.addWidget(self._build_temporal_group())
        root.addLayout(self._build_bottom_row())

        self.statusBar().showMessage("Ready — open a video file to begin.")

    # ── File bar ──────────────────────────────────────────────────────────────

    def _build_file_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Video:"))
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to video file (.mp4, .avi, .mov, .mkv …)")
        self._path_edit.returnPressed.connect(self._load_video)
        bar.addWidget(self._path_edit, stretch=1)
        btn_browse = QPushButton("Browse …")
        btn_browse.clicked.connect(self._browse_video)
        bar.addWidget(btn_browse)
        btn_yt = QPushButton("⬇  YouTube")
        btn_yt.setToolTip("Download a video from YouTube (or any yt-dlp supported URL)")
        btn_yt.setStyleSheet("font-weight:bold; color:#ff4444;")
        btn_yt.clicked.connect(self._open_youtube_dialog)
        bar.addWidget(btn_yt)
        btn_load = QPushButton("Load")
        btn_load.setDefault(True)
        btn_load.clicked.connect(self._load_video)
        bar.addWidget(btn_load)
        return bar

    # ── Centre area (preview + strip) ─────────────────────────────────────────

    def _build_center_area(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)

        self._preview = VideoPreview()
        self._preview.cropChanged.connect(self._on_crop_from_preview)
        layout.addWidget(self._preview, stretch=3)
        layout.addWidget(self._build_strip(), stretch=1)
        return widget

    def _build_strip(self) -> QGroupBox:
        group = QGroupBox("7-Frame Strip — cropped preview (im4 = centre frame sent to network)")
        row   = QHBoxLayout(group)
        row.setSpacing(4)
        row.setContentsMargins(4, 4, 4, 4)

        self._strip_img:  List[QLabel] = []
        self._strip_info: List[QLabel] = []

        for i in range(7):
            col = QWidget()
            cl  = QVBoxLayout(col)
            cl.setSpacing(2)
            cl.setContentsMargins(0, 0, 0, 0)

            img_lbl = QLabel()
            img_lbl.setMinimumSize(150, 90)
            img_lbl.setAlignment(Qt.AlignCenter)
            border_col = "#ffd700" if i == 3 else "#444"
            img_lbl.setStyleSheet(f"border: 2px solid {border_col}; background:#0d0d0d;")

            info_lbl = QLabel(f"im{i + 1}" + ("  [center]" if i == 3 else ""))
            info_lbl.setAlignment(Qt.AlignCenter)
            text_col = "#ffd700" if i == 3 else "#888"
            info_lbl.setStyleSheet(f"color:{text_col}; font-size:10px;")

            cl.addWidget(img_lbl)
            cl.addWidget(info_lbl)
            row.addWidget(col)

            self._strip_img.append(img_lbl)
            self._strip_info.append(info_lbl)

        return group

    # ── Temporal group ────────────────────────────────────────────────────────

    def _build_temporal_group(self) -> QGroupBox:
        group = QGroupBox("Temporal Sampling — choose which 7 frames to export")
        gl = QGridLayout(group)
        gl.setSpacing(6)

        # Row 0: frame slider
        gl.addWidget(QLabel("Current frame:"), 0, 0)
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        gl.addWidget(self._frame_slider, 0, 1, 1, 5)
        self._frame_lbl = QLabel("0 / 0")
        self._frame_lbl.setMinimumWidth(100)
        gl.addWidget(self._frame_lbl, 0, 6)

        # Row 1: start, step, timecode
        gl.addWidget(QLabel("Start frame:"), 1, 0)
        self._start_spin = QSpinBox()
        self._start_spin.setMinimum(0)
        self._start_spin.setMaximum(999999)
        self._start_spin.setMinimumWidth(90)
        self._start_spin.valueChanged.connect(self._on_temporal_changed)
        gl.addWidget(self._start_spin, 1, 1)

        gl.addWidget(QLabel("Step (frames between im1…im7):"), 1, 2)
        self._step_spin = QSpinBox()
        self._step_spin.setMinimum(1)
        self._step_spin.setMaximum(500)
        self._step_spin.setValue(1)
        self._step_spin.setToolTip(
            "Step=1 → consecutive frames.\n"
            "Step=N → sample every N-th frame (slow-motion effect)."
        )
        self._step_spin.valueChanged.connect(self._on_temporal_changed)
        gl.addWidget(self._step_spin, 1, 3)

        self._time_lbl = QLabel("0.00 s")
        gl.addWidget(self._time_lbl, 1, 4, 1, 3)

        # Row 2: navigation buttons
        btn_prev = QPushButton("◀  –1 step")
        btn_prev.clicked.connect(lambda: self._nudge_start(-1))
        gl.addWidget(btn_prev, 2, 0)

        btn_set = QPushButton("Pin start = current frame")
        btn_set.clicked.connect(self._pin_start)
        gl.addWidget(btn_set, 2, 1, 1, 2)

        btn_next = QPushButton("+1 step  ▶")
        btn_next.clicked.connect(lambda: self._nudge_start(1))
        gl.addWidget(btn_next, 2, 3)

        self._indices_lbl = QLabel("Frame indices: —")
        self._indices_lbl.setStyleSheet("color:#aaa; font-size:10px;")
        gl.addWidget(self._indices_lbl, 2, 4, 1, 3)

        return group

    # ── Bottom row (spatial + export) ─────────────────────────────────────────

    def _build_bottom_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(self._build_spatial_group(), stretch=1)
        row.addWidget(self._build_export_group(),  stretch=1)
        return row

    def _build_spatial_group(self) -> QGroupBox:
        group = QGroupBox("Spatial Crop  (drag rectangle on preview, or type values)")
        gl = QGridLayout(group)
        gl.setSpacing(6)

        def spin(lo, hi, val):
            s = QSpinBox()
            s.setMinimum(lo)
            s.setMaximum(hi)
            s.setValue(val)
            s.setMinimumWidth(80)
            return s

        self._cx_spin = spin(0, 99999, 0)
        self._cy_spin = spin(0, 99999, 0)
        self._cw_spin = spin(SCALE, 99999, VIMEO_W)
        self._ch_spin = spin(SCALE, 99999, VIMEO_H)

        gl.addWidget(QLabel("X (left):"),  0, 0)
        gl.addWidget(self._cx_spin,        0, 1)
        gl.addWidget(QLabel("Y (top):"),   0, 2)
        gl.addWidget(self._cy_spin,        0, 3)
        gl.addWidget(QLabel("W (HR px):"), 1, 0)
        gl.addWidget(self._cw_spin,        1, 1)
        gl.addWidget(QLabel("H (HR px):"), 1, 2)
        gl.addWidget(self._ch_spin,        1, 3)

        for sp in (self._cx_spin, self._cy_spin, self._cw_spin, self._ch_spin):
            sp.valueChanged.connect(self._on_crop_from_spin)

        btn_center = QPushButton("Center crop")
        btn_center.clicked.connect(self._preview.center_crop)
        btn_reset  = QPushButton(f"Reset to {VIMEO_W}×{VIMEO_H}")
        btn_reset.clicked.connect(self._reset_crop_size)
        gl.addWidget(btn_center, 2, 0, 1, 2)
        gl.addWidget(btn_reset,  2, 2, 1, 2)

        note = QLabel(
            f"Default = Vimeo-90K HR ({VIMEO_W}×{VIMEO_H})  →  LR {VIMEO_W//SCALE}×{VIMEO_H//SCALE}.  "
            "Must be divisible by 4 for correct LR alignment."
        )
        note.setStyleSheet("color:#888; font-size:10px;")
        note.setWordWrap(True)
        gl.addWidget(note, 3, 0, 1, 4)

        return group

    def _build_export_group(self) -> QGroupBox:
        group = QGroupBox("Export — create dataset for evaluate.py")
        gl = QGridLayout(group)
        gl.setSpacing(6)

        gl.addWidget(QLabel("Output dir:"), 0, 0)
        self._outdir_edit = QLineEdit(self._out_dir)
        gl.addWidget(self._outdir_edit, 0, 1)
        btn_ob = QPushButton("…")
        btn_ob.setFixedWidth(28)
        btn_ob.clicked.connect(self._browse_outdir)
        gl.addWidget(btn_ob, 0, 2)

        gl.addWidget(QLabel("Sequence ID:"), 1, 0)
        self._seqid_edit = QLineEdit("custom_001")
        self._seqid_edit.setToolTip(
            "Becomes the folder name: <out_dir>/test/<sequence_id>/hr & lr"
        )
        gl.addWidget(self._seqid_edit, 1, 1, 1, 2)

        self._export_btn = QPushButton("▶  Create Dataset Sequence")
        self._export_btn.setStyleSheet("font-weight:bold; padding:6px;")
        self._export_btn.clicked.connect(self._export)
        gl.addWidget(self._export_btn, 2, 0, 1, 3)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        gl.addWidget(self._progress, 3, 0, 1, 3)

        self._status_lbl = QLabel("—")
        self._status_lbl.setStyleSheet("color:#888; font-size:10px;")
        gl.addWidget(self._status_lbl, 4, 0, 1, 3)

        note = QLabel(
            "Each export appends one sequence to test_manifest.csv.\n"
            "Run multiple times with different sequence IDs to build a larger test set."
        )
        note.setStyleSheet("color:#666; font-size:10px;")
        note.setWordWrap(True)
        gl.addWidget(note, 5, 0, 1, 3)

        return group

    # ── Video loading ─────────────────────────────────────────────────────────

    def _open_youtube_dialog(self):
        dlg = YoutubeDialog(self)
        dlg.downloaded.connect(self._on_youtube_downloaded)
        dlg.exec_()

    def _on_youtube_downloaded(self, file_path: str):
        self._path_edit.setText(file_path)
        self._load_video()

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video",
            filter="Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.ts);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)
            self._load_video()

    def _load_video(self):
        path = self._path_edit.text().strip()
        if not path:
            return
        if self._cap is not None:
            self._cap.release()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", f"Cannot open video:\n{path}")
            return

        self._cap           = cap
        self._total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_w               = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h               = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._frame_slider.setMaximum(max(0, self._total_frames - 1))
        self._start_spin.setMaximum(max(0, self._total_frames - 1))
        self._frame_slider.setValue(0)
        self._start_spin.setValue(0)

        # Centre the default crop on the newly loaded video
        self._preview._vw = vid_w
        self._preview._vh = vid_h
        self._preview.center_crop()

        dur = self._total_frames / self._fps
        self.statusBar().showMessage(
            f"{Path(path).name}  |  {vid_w}×{vid_h}  |  "
            f"{self._total_frames} frames  |  {self._fps:.2f} fps  |  {dur:.1f} s"
        )
        self._read_and_show(0)
        self._update_strip()

    # ── Frame reading ─────────────────────────────────────────────────────────

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        if self._cap is None or not (0 <= idx < self._total_frames):
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        return frame if ok else None

    def _read_and_show(self, idx: int):
        frame = self._read_frame(idx)
        if frame is None:
            return
        self._preview.set_frame(frame)
        t = idx / self._fps
        self._frame_lbl.setText(f"{idx} / {self._total_frames - 1}")
        self._time_lbl.setText(f"{t:.2f} s  ({t / 60:.2f} min)")
        self._sync_spins_from_preview()

    # ── Temporal controls ─────────────────────────────────────────────────────

    def _on_frame_slider(self, value: int):
        self._read_and_show(value)

    def _on_temporal_changed(self):
        self._update_strip()

    def _pin_start(self):
        self._start_spin.setValue(self._frame_slider.value())

    def _nudge_start(self, direction: int):
        step = self._step_spin.value()
        new  = self._start_spin.value() + direction * step
        new  = max(0, min(new, self._total_frames - 1))
        self._start_spin.setValue(new)
        self._frame_slider.setValue(new)

    def _frame_indices(self) -> List[int]:
        start = self._start_spin.value()
        step  = self._step_spin.value()
        return [
            min(start + i * step, self._total_frames - 1)
            for i in range(7)
        ]

    def _update_strip(self):
        if self._cap is None:
            return
        indices = self._frame_indices()
        self._indices_lbl.setText("Frame indices: " + "  ".join(str(i) for i in indices))

        cx, cy = self._preview.cx, self._preview.cy
        cw, ch = self._preview.cw, self._preview.ch

        for i, (idx, img_lbl, info_lbl) in enumerate(
            zip(indices, self._strip_img, self._strip_info)
        ):
            frame = self._read_frame(idx)
            self._selected_frames[i] = frame

            if frame is not None:
                crop = frame[cy: cy + ch, cx: cx + cw]
                rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                pm   = QPixmap.fromImage(qimg).scaled(
                    img_lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                img_lbl.setPixmap(pm)
            else:
                img_lbl.clear()
                img_lbl.setText("—")

            info_lbl.setText(f"im{i + 1}  fr={idx}" + ("  [center]" if i == 3 else ""))

    # ── Crop synchronisation ──────────────────────────────────────────────────

    def _on_crop_from_preview(self, x: int, y: int, w: int, h: int):
        for sp, val in zip(
            (self._cx_spin, self._cy_spin, self._cw_spin, self._ch_spin),
            (x, y, w, h),
        ):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self._update_strip()

    def _on_crop_from_spin(self):
        self._preview.set_crop(
            self._cx_spin.value(), self._cy_spin.value(),
            self._cw_spin.value(), self._ch_spin.value(),
        )
        self._update_strip()

    def _sync_spins_from_preview(self):
        for sp, val in zip(
            (self._cx_spin, self._cy_spin, self._cw_spin, self._ch_spin),
            (self._preview.cx, self._preview.cy, self._preview.cw, self._preview.ch),
        ):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)

    def _reset_crop_size(self):
        self._preview.reset_size()
        self._sync_spins_from_preview()
        self._update_strip()

    # ── Output dir browser ────────────────────────────────────────────────────

    def _browse_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self._outdir_edit.setText(d)

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        if self._cap is None:
            QMessageBox.warning(self, "No video", "Load a video file first.")
            return

        # Ensure all 7 frames are loaded
        for i, (idx, frame) in enumerate(
            zip(self._frame_indices(), self._selected_frames)
        ):
            if frame is None:
                self._selected_frames[i] = self._read_frame(idx)

        if any(f is None for f in self._selected_frames):
            QMessageBox.critical(self, "Frame error", "Could not read all 7 frames.")
            return

        out_dir = self._outdir_edit.text().strip()
        seq_id  = self._seqid_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Missing", "Set an output directory.")
            return
        if not seq_id:
            QMessageBox.warning(self, "Missing", "Enter a sequence ID.")
            return

        crop = (
            self._preview.cx, self._preview.cy,
            self._preview.cw, self._preview.ch,
        )

        # Warn if crop dimensions are not divisible by scale
        if crop[2] % SCALE != 0 or crop[3] % SCALE != 0:
            QMessageBox.warning(
                self, "Crop size warning",
                f"Width ({crop[2]}) or height ({crop[3]}) is not divisible by {SCALE}.\n"
                "LR dimensions will be floor-divided; consider adjusting the crop."
            )

        self._export_btn.setEnabled(False)
        self._progress.setValue(0)
        self._status_lbl.setText("Starting …")

        self._export_worker = ExportWorker(
            list(self._selected_frames), crop, out_dir, seq_id  # type: ignore[arg-type]
        )
        self._export_worker.progress.connect(self._on_export_progress)
        self._export_worker.done.connect(self._on_export_done)
        self._export_worker.start()

    def _on_export_progress(self, pct: int, msg: str):
        self._progress.setValue(pct)
        self._status_lbl.setText(msg)

    def _on_export_done(self, ok: bool, msg: str):
        self._export_btn.setEnabled(True)
        if ok:
            self._progress.setValue(100)
            self._status_lbl.setText("Done!")
            self.statusBar().showMessage("Export complete.")
            QMessageBox.information(self, "Export Complete", msg)
            # Auto-increment sequence ID
            seqid = self._seqid_edit.text().strip()
            parts = seqid.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                self._seqid_edit.setText(
                    f"{parts[0]}_{int(parts[1]) + 1:03d}"
                )
        else:
            self._progress.setValue(0)
            self._status_lbl.setText(f"Error: {msg}")
            QMessageBox.critical(self, "Export Failed", msg)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive tool to crop a video and build a Vimeo-90K style test dataset."
    )
    parser.add_argument(
        "--out_dir", default="./data/vimeo_prepared",
        help="Default output directory (default: ./data/vimeo_prepared)",
    )
    parser.add_argument(
        "video", nargs="?", default=None,
        help="Optional: video file to load on startup",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow, QWidget { background-color: #1e1e1e; color: #ddd; }
        QGroupBox {
            border: 1px solid #444; border-radius: 4px;
            margin-top: 8px; padding-top: 6px;
            font-weight: bold; color: #ccc;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; color: #aaa; }
        QPushButton {
            background: #333; border: 1px solid #555; border-radius: 4px;
            padding: 4px 10px; color: #ddd;
        }
        QPushButton:hover   { background: #444; }
        QPushButton:pressed { background: #222; }
        QPushButton:disabled { color: #666; }
        QLineEdit, QSpinBox {
            background: #2a2a2a; border: 1px solid #555;
            border-radius: 3px; padding: 3px 6px; color: #ddd;
        }
        QSlider::groove:horizontal { background: #333; height: 4px; border-radius: 2px; }
        QSlider::handle:horizontal {
            background: #ffd700; width: 14px; height: 14px;
            margin: -5px 0; border-radius: 7px;
        }
        QProgressBar {
            border: 1px solid #555; border-radius: 3px;
            background: #222; text-align: center; color: #ddd;
        }
        QProgressBar::chunk { background: #4a9; border-radius: 3px; }
        QStatusBar { color: #888; font-size: 11px; }
        QLabel { color: #ccc; }
    """)

    win = MainWindow(out_dir=args.out_dir)
    if args.video:
        win._path_edit.setText(args.video)
        win._load_video()

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

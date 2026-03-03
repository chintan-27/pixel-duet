#!/usr/bin/env python3
# app.py — Pixel Duet Desktop UI (PySide6)
# Thin wrapper: all algorithm logic lives in the pixelduet package.

import sys
import os

import numpy as np
from PIL import Image, ImageSequence

from PySide6 import QtCore, QtGui, QtWidgets

from pixelduet.utils import load_rgb, resize_to_width, to_np, permute_image
from pixelduet.animation import compute_paths, compute_stagger, get_easing, bezier_cubic  # noqa: F401
from pixelduet import map_pixels


# ---------- Preview worker (low-res proxy, runs fast) ----------
class PreviewWorker(QtCore.QObject):
    """Compute a low-res mapping and return the permuted image for instant preview."""

    finished = QtCore.Signal(object)  # emits numpy array (H, W, 3) uint8
    failed = QtCore.Signal(str)

    def __init__(self, pathA, pathB, preview_width=64, parent=None):
        super().__init__(parent)
        self.pathA = pathA
        self.pathB = pathB
        self.preview_width = preview_width

    @QtCore.Slot()
    def run(self):
        try:
            A0 = load_rgb(self.pathA)
            B0 = load_rgb(self.pathB)
            A = resize_to_width(A0, self.preview_width)
            Aw, Ah = A.size
            B = B0.resize((Aw, Ah), Image.LANCZOS)
            A_np = to_np(A)
            B_np = to_np(B)

            M_final, _Hm, _Wm = map_pixels(
                A_np,
                B_np,
                levels=[self.preview_width],
                tiles=[min(12, self.preview_width // 4)],
                lambdas_spatial=[0.25],
                lambdas_parent=[0.0],
                refine_iters=5,
            )
            result = permute_image(A_np, M_final)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


# ---------- Export worker (off-thread, with progress) ----------
class ExportWorker(QtCore.QObject):
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(int)
    stage = QtCore.Signal(str)

    def __init__(
        self,
        pathA,
        pathB,
        width,
        frames,
        hold,
        save_path,
        levels,
        tiles,
        lam_spatial,
        lam_parent,
        stagger,
        arc,
        easing="cosine",
        parent=None,
    ):
        super().__init__(parent)
        self.pathA = pathA
        self.pathB = pathB
        self.width = width
        self.frames = frames
        self.hold = hold
        self.save_path = save_path
        self.levels = levels
        self.tiles = tiles
        self.lam_spatial = lam_spatial
        self.lam_parent = lam_parent
        self.stagger = stagger
        self.arc = arc
        self.easing = easing

    @QtCore.Slot()
    def run(self):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import PillowWriter

            self.stage.emit("Loading images\u2026")
            A0 = load_rgb(self.pathA)
            B0 = load_rgb(self.pathB)
            A = resize_to_width(A0, self.width)
            Aw, Ah = A.size
            B = B0.resize((Aw, Ah), Image.LANCZOS)
            A_np = to_np(A)
            B_np = to_np(B)
            H, W, _ = A_np.shape
            N = H * W

            self.stage.emit("Computing mapping\u2026")
            self.progress.emit(5)

            def on_progress(stage, frac):
                self.progress.emit(5 + int(40 * frac))

            M_final, Hm, Wm = map_pixels(
                A_np,
                B_np,
                levels=self.levels,
                tiles=self.tiles,
                lambdas_spatial=self.lam_spatial,
                lambdas_parent=self.lam_parent,
                progress_callback=on_progress,
            )
            if Hm != H or Wm != W:
                raise RuntimeError("Final mapping resolution mismatch.")

            self.stage.emit("Rendering GIF\u2026")

            colors_A = (A_np.reshape(N, 3) / 255.0).clip(0, 1)
            perm_img = permute_image(A_np, M_final).astype(np.float32) / 255.0

            paths = compute_paths(H, W, M_final, arc=self.arc)
            Sx, Sy = paths[0], paths[1]
            rng = np.random.default_rng(None)
            start = rng.uniform(0.0, min(0.95, self.stagger), size=N)
            ease_fn = get_easing(self.easing)

            fig, ax = plt.subplots(figsize=(6, 6 * H / W))
            ax.set_facecolor("#FFFFFF")
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_xlim(-0.5, W - 0.5)
            ax.set_ylim(H - 0.5, -0.5)
            ax.set_aspect("equal")
            ax.axis("off")

            s = max(2, int(180000 / N))
            scat = ax.scatter(
                Sx,
                Sy,
                c=colors_A,
                s=s,
                marker="s",
                linewidths=0,
                edgecolors="none",
                alpha=1.0,
                zorder=3,
            )
            im = ax.imshow(
                perm_img,
                extent=(-0.5, W - 0.5, H - 0.5, -0.5),
                zorder=1,
                interpolation="nearest",
                visible=False,
            )

            total_frames = self.frames + max(0, self.hold)

            writer = PillowWriter(fps=30)
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            with writer.saving(fig, self.save_path, dpi=100):
                for frame in range(total_frames):
                    if frame < self.frames:
                        g = frame / max(1, self.frames - 1)
                        t = ease_fn(np.clip((g - start) / (1.0 - start + 1e-9), 0, 1))
                        x, y = bezier_cubic(*paths, t)
                        scat.set_offsets(np.column_stack([x, y]))
                        if im.get_visible():
                            im.set_visible(False)
                        if not scat.get_visible():
                            scat.set_visible(True)
                    else:
                        if not im.get_visible():
                            im.set_visible(True)
                        if scat.get_visible():
                            scat.set_visible(False)
                    writer.grab_frame()
                    self.progress.emit(45 + int(55 * (frame + 1) / total_frames))

            plt.close(fig)
            self.stage.emit("Done.")
            self.finished.emit(self.save_path)
        except Exception as e:
            self.failed.emit(str(e))


# ---------- UI widgets ----------
ACCENT = "#3FD0FF"
BG = "#0F1115"
CARD = "#131720"
TEXT = "#EAEAEA"
BORDER = "#2A3140"


class ImageCard(QtWidgets.QFrame):
    pathChanged = QtCore.Signal(str)

    def __init__(self, title):
        super().__init__()
        self.setAcceptDrops(True)
        self.path = ""
        self.setObjectName("ImageCard")
        self.setStyleSheet(f"""
            QFrame#ImageCard {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
        """)
        self.setMinimumSize(240, 220)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        self.lblTitle = QtWidgets.QLabel(title)
        self.lblTitle.setStyleSheet("font-weight: 600;")
        self.lblTitle.setAlignment(QtCore.Qt.AlignLeft)
        self.image = QtWidgets.QLabel()
        self.image.setAlignment(QtCore.Qt.AlignCenter)
        self.image.setStyleSheet("background-color: #0D1016; border-radius: 6px;")
        self.image.setMinimumHeight(150)
        self.caption = QtWidgets.QLabel("Click to select or drop an image")
        self.caption.setAlignment(QtCore.Qt.AlignCenter)
        self.caption.setStyleSheet("color: #9AA4B2; font-size: 12px;")
        lay.addWidget(self.lblTitle)
        lay.addWidget(self.image, 1)
        lay.addWidget(self.caption)
        self.image.installEventFilter(self)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setColor(QtGui.QColor(0, 0, 0, 120))
        shadow.setOffset(0, 8)
        self.setGraphicsEffect(shadow)

    def eventFilter(self, obj, event):
        if obj is self.image and event.type() == QtCore.QEvent.MouseButtonRelease:
            self.select_file_dialog()
            return True
        return super().eventFilter(obj, event)

    def select_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self.set_image(path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            url = e.mimeData().urls()[0]
            if url.isLocalFile():
                p = url.toLocalFile().lower()
                if p.endswith((".png", ".jpg", ".jpeg")):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e):
        url = e.mimeData().urls()[0]
        self.set_image(url.toLocalFile())

    def set_image(self, path):
        self.path = path
        pm = QtGui.QPixmap(path)
        if pm.isNull():
            self.caption.setText("Failed to load")
        else:
            self.image.setPixmap(
                pm.scaled(
                    self.image.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )
            base = os.path.basename(path)
            self.caption.setText(base)
            self.pathChanged.emit(path)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        pm = self.image.pixmap()
        if pm and not pm.isNull():
            self.image.setPixmap(
                pm.scaled(
                    self.image.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
            )


class OutputViewer(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("OutputViewer")
        self.setStyleSheet(f"""
            QFrame#OutputViewer {{
                background: {CARD};
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
        """)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Output")
        title.setStyleSheet("font-weight: 600;")
        header.addWidget(title)
        header.addStretch(1)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("color: #9AA4B2;")
        header.addWidget(self.status)
        outer.addLayout(header)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setStyleSheet("QScrollArea { background: #0D1016; border-radius: 8px; }")
        self.scroll.setAlignment(QtCore.Qt.AlignCenter)
        outer.addWidget(self.scroll, 1)

        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setScaledContents(False)
        self.scroll.setWidget(self.view)

        self.movie = None
        self.pilTimer = None
        self.pilFrames = []
        self.pilDurations = []
        self.pilIndex = 0

    def show_gif(self, path):
        self.clear_preview()
        self.movie = QtGui.QMovie(path)
        if self.movie.isValid():
            self.movie.setCacheMode(QtGui.QMovie.CacheAll)
            self.movie.setSpeed(100)
            self.view.setMovie(self.movie)

            def fit_to_frame(_):
                rect = self.movie.frameRect()
                self.view.setFixedSize(rect.size())

            self.movie.frameChanged.connect(fit_to_frame)
            self.movie.start()
            self.status.setText(os.path.basename(path))
        else:
            try:
                gif = Image.open(path)
                self.pilFrames, self.pilDurations = [], []
                for frame in ImageSequence.Iterator(gif):
                    fr_rgb = frame.convert("RGB")
                    np_img = np.array(fr_rgb)
                    H, W = np_img.shape[:2]
                    qimg = QtGui.QImage(np_img.data, W, H, 3 * W, QtGui.QImage.Format_RGB888)
                    pm = QtGui.QPixmap.fromImage(qimg.copy())
                    self.pilFrames.append(pm)
                    self.pilDurations.append(max(10, int(frame.info.get("duration", 33))))
                self.pilIndex = 0
                if self.pilFrames:
                    self.view.setPixmap(self.pilFrames[0])
                    self.view.setFixedSize(self.pilFrames[0].size())
                    self.pilTimer = QtCore.QTimer(self)
                    self.pilTimer.timeout.connect(self._advance_pil_frame)
                    self.pilTimer.start(self.pilDurations[0])
                self.status.setText(os.path.basename(path))
            except Exception as e:
                self.status.setText(f"GIF preview failed: {e}")

    def _advance_pil_frame(self):
        if not self.pilFrames:
            return
        pm = self.pilFrames[self.pilIndex]
        self.view.setPixmap(pm)
        self.view.setFixedSize(pm.size())
        self.pilIndex = (self.pilIndex + 1) % len(self.pilFrames)
        self.pilTimer.start(self.pilDurations[self.pilIndex])

    def show_still_np(self, np_img):
        self.clear_preview()
        H, W, C = np_img.shape
        qimg = QtGui.QImage(np_img.data, W, H, 3 * W, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg.copy())
        self.view.setPixmap(pm)
        self.view.setFixedSize(pm.size())

    def clear_preview(self):
        if self.movie:
            try:
                self.movie.stop()
            except Exception:
                pass
            self.movie = None
        if self.pilTimer:
            self.pilTimer.stop()
            self.pilTimer = None
        self.pilFrames, self.pilDurations = [], []
        self.view.clear()
        self.view.setFixedSize(QtCore.QSize(1, 1))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)


# ---------- Main window ----------
class PixelDuetApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pixel Duet")
        self.resize(1100, 760)
        self._apply_theme()

        self.pathA = ""
        self.pathB = ""
        self.thread = None
        self.worker = None
        self.previewThread = None
        self.previewWorker = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # Top row: Image A [arrow] Image B
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(16)
        self.cardA = ImageCard("Image A (source colors)")
        self.cardB = ImageCard("Image B (target structure)")
        self.cardA.pathChanged.connect(self._on_pathA)
        self.cardB.pathChanged.connect(self._on_pathB)

        arrow = QtWidgets.QLabel("\u2192")
        arrow.setAlignment(QtCore.Qt.AlignCenter)
        arrow.setStyleSheet("color: #3FD0FF; font-size: 28px; font-weight: 700;")
        arrow.setFixedWidth(40)

        top.addWidget(self.cardA, 1)
        top.addWidget(arrow)
        top.addWidget(self.cardB, 1)
        root.addLayout(top)

        # Control bar
        ctrl = QtWidgets.QFrame()
        ctrl.setStyleSheet(
            f"QFrame {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; }}"
        )
        cl = QtWidgets.QHBoxLayout(ctrl)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(12)

        self.btnSubmit = QtWidgets.QPushButton("Submit")
        self.btnSubmit.setEnabled(False)
        self.btnSubmit.setStyleSheet(f"""
            QPushButton {{
                background: {ACCENT};
                color: #021520;
                font-weight: 600;
                padding: 10px 16px;
                border-radius: 8px;
                border: none;
            }}
            QPushButton:disabled {{
                background: #2A3140; color: #8A96A6;
            }}
        """)
        self.btnSubmit.clicked.connect(self.on_submit)

        self.spinWidth = QtWidgets.QSpinBox()
        self.spinWidth.setRange(160, 640)
        self.spinWidth.setValue(260)
        self.spinWidth.setSuffix(" px")
        self.spinFrames = QtWidgets.QSpinBox()
        self.spinFrames.setRange(30, 240)
        self.spinFrames.setValue(120)
        self.spinHold = QtWidgets.QSpinBox()
        self.spinHold.setRange(0, 120)
        self.spinHold.setValue(24)
        self.dblArc = QtWidgets.QDoubleSpinBox()
        self.dblArc.setRange(0.0, 0.5)
        self.dblArc.setSingleStep(0.02)
        self.dblArc.setValue(0.10)
        self.dblStagger = QtWidgets.QDoubleSpinBox()
        self.dblStagger.setRange(0.0, 0.95)
        self.dblStagger.setSingleStep(0.05)
        self.dblStagger.setValue(0.35)

        self.comboEasing = QtWidgets.QComboBox()
        for name in ("cosine", "linear", "bounce", "elastic", "step", "expo"):
            self.comboEasing.addItem(name)
        self.comboEasing.setCurrentText("cosine")

        def small(label, w):
            box = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(box)
            hl.setContentsMargins(0, 0, 0, 0)
            lab = QtWidgets.QLabel(label)
            lab.setStyleSheet("color: #9AA4B2;")
            hl.addWidget(lab)
            hl.addWidget(w)
            return box

        cl.addWidget(self.btnSubmit)
        cl.addSpacing(12)
        cl.addWidget(small("Width", self.spinWidth))
        cl.addWidget(small("Frames", self.spinFrames))
        cl.addWidget(small("Hold", self.spinHold))
        cl.addWidget(small("Arc", self.dblArc))
        cl.addWidget(small("Stagger", self.dblStagger))
        cl.addWidget(small("Easing", self.comboEasing))
        cl.addStretch(1)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                background: #1A1F29; border-radius: 3px;
            }}
            QProgressBar::chunk {{
                background: {ACCENT}; border-radius: 3px;
            }}
        """)
        cl.addWidget(self.progress)
        root.addWidget(ctrl)

        self.output = OutputViewer()
        root.addWidget(self.output, 1)

        self.statusBar().setStyleSheet(f"QStatusBar {{ background: {BG}; color: {TEXT}; }}")
        self._update_submit_enabled()

    def _apply_theme(self):
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(BG))
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(TEXT))
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#1A1F29"))
        pal.setColor(QtGui.QPalette.Text, QtGui.QColor(TEXT))
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor("#1A1F29"))
        pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(TEXT))
        self.setPalette(pal)
        self.setStyleSheet(f"""
            QWidget {{ color: {TEXT}; font-size: 14px; }}
            QSpinBox, QDoubleSpinBox, QLineEdit {{
                padding: 6px; border: 1px solid {BORDER}; border-radius: 6px; background: #1A1F29;
            }}
            QLabel {{ color: {TEXT}; }}
        """)

    def _on_pathA(self, p):
        self.pathA = p
        self._update_submit_enabled()

    def _on_pathB(self, p):
        self.pathB = p
        self._update_submit_enabled()

    def _update_submit_enabled(self):
        self.btnSubmit.setEnabled(bool(self.pathA and self.pathB))
        if self.pathA and self.pathB:
            self._start_preview()

    def _start_preview(self):
        """Launch a low-res preview mapping in a background thread."""
        # Cancel any running preview
        if self.previewThread and self.previewThread.isRunning():
            self.previewThread.quit()
            self.previewThread.wait(500)

        self.previewWorker = PreviewWorker(self.pathA, self.pathB, preview_width=64)
        self.previewThread = QtCore.QThread(self)
        self.previewWorker.moveToThread(self.previewThread)
        self.previewThread.started.connect(self.previewWorker.run)
        self.previewWorker.finished.connect(self._preview_done)
        self.previewWorker.failed.connect(self._preview_failed)
        self.previewWorker.finished.connect(self.previewThread.quit)
        self.previewWorker.failed.connect(self.previewThread.quit)
        self.previewWorker.finished.connect(self.previewWorker.deleteLater)
        self.previewWorker.failed.connect(self.previewWorker.deleteLater)
        self.previewThread.finished.connect(self.previewThread.deleteLater)
        self.statusBar().showMessage("Computing preview\u2026")
        self.previewThread.start()

    def _preview_done(self, np_img):
        self.output.show_still_np(np_img)
        self.output.status.setText("Preview (low-res)")
        self.statusBar().showMessage("Preview ready", 3000)

    def _preview_failed(self, msg):
        self.statusBar().showMessage(f"Preview failed: {msg}", 3000)

    def on_submit(self):
        try:
            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save GIF", "output/swap.gif", "GIF (*.gif)"
            )
            if not save_path:
                return

            width = self.spinWidth.value()
            frames = self.spinFrames.value()
            hold = self.spinHold.value()
            arc = self.dblArc.value()
            stagger = self.dblStagger.value()
            easing = self.comboEasing.currentText()

            w2 = max(64, width // 4)
            w3 = max(96, width // 2)
            levels = [w2, w3, width]
            tiles = [12, 12, 10]
            lam_spatial = [0.35, 0.25, 0.18]
            lam_parent = [0.00, 0.20, 0.12]

            self.progress.setValue(0)
            self.statusBar().showMessage("Exporting\u2026")

            self.worker = ExportWorker(
                self.pathA,
                self.pathB,
                width,
                frames,
                hold,
                save_path,
                levels,
                tiles,
                lam_spatial,
                lam_parent,
                stagger,
                arc,
                easing,
            )
            self.thread = QtCore.QThread(self)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self._export_done)
            self.worker.failed.connect(self._export_failed)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.stage.connect(self.statusBar().showMessage)
            self.worker.finished.connect(self.thread.quit)
            self.worker.failed.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.failed.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()
        except Exception as e:
            self._export_failed(str(e))

    def _export_done(self, path):
        self.statusBar().showMessage(f"Saved: {path}", 5000)
        self.progress.setValue(100)
        self.output.status.setText("Playing GIF")
        self.output.show_gif(path)

    def _export_failed(self, msg):
        self.statusBar().showMessage(f"Export failed: {msg}", 8000)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Pixel Duet")
    win = PixelDuetApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

# app.py — Pixel Duet UI (original-size output, clean, aesthetic, user-friendly)
# One-way: uses ONLY pixels from Image A to form Image B with multiscale edge-aware mapping.
# Top: two image cards (A → B) that show thumbnails immediately on selection/drag-drop.
# Middle: a slim control bar with Submit and a progress line.
# Bottom: an output viewer that displays the resulting GIF or final still at original pixel size (no stretching), with scrolling if needed.

import sys
import os
import numpy as np
from PIL import Image, ImageSequence

from PySide6 import QtCore, QtGui, QtWidgets

# SciPy optional (recommended for best quality mapping)
try:
    from scipy.optimize import linear_sum_assignment
    from scipy.ndimage import sobel as nd_sobel

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ---------- Core utilities ----------
def load_rgb(path):
    return Image.open(path).convert("RGB")


def resize_to_width(img, width):
    w0, h0 = img.size
    height = max(1, int(round(width * h0 / w0)))
    return img.resize((width, height), Image.LANCZOS)


def to_np(img):
    return np.asarray(img, dtype=np.float32)


def grid_coords(H, W):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return x.reshape(-1), y.reshape(-1)


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


# sRGB -> Lab (D65)
def _srgb_to_linear(c):
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def _rgb_to_xyz(rgb):
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    return np.tensordot(rgb, M.T, axes=1)


def _f_lab(t):
    e = (6 / 29) ** 3
    k = (29 / 3) ** 3 / 3
    return np.where(t > e, np.cbrt(t), (t * k) + 4 / 29)


def rgb_to_lab(rgb_uint8):
    rgb = np.clip(rgb_uint8.astype(np.float32) / 255.0, 0, 1)
    lin = _srgb_to_linear(rgb)
    xyz = _rgb_to_xyz(lin)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx, fy, fz = (
        _f_lab(xyz[..., 0] / Xn),
        _f_lab(xyz[..., 1] / Yn),
        _f_lab(xyz[..., 2] / Zn),
    )
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def edges_from_B(B_np):
    lum = (luminance(B_np) / 255.0).astype(np.float32)
    if HAS_SCIPY:
        gx = nd_sobel(lum, axis=1, mode="reflect")
        gy = nd_sobel(lum, axis=0, mode="reflect")
        mag = np.sqrt(gx * gx + gy * gy)
    else:
        gy, gx = np.gradient(lum)
        mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    return mag / m if m > 0 else mag


# ---------- Mapping plumbing ----------
def mapping_global(A_np, B_np):
    H, W, _ = A_np.shape
    N = H * W
    A_flat = A_np.reshape(N, 3)
    B_flat = B_np.reshape(N, 3)
    lumA = luminance(A_flat)
    lumB = luminance(B_flat)
    rA = np.argsort(lumA, kind="mergesort")
    rB = np.argsort(lumB, kind="mergesort")
    M = np.empty(N, dtype=np.int64)
    M[rA] = rB
    return M


def mapping_color_local_edgeparent(
    A_np, B_np, tile, spatial, base, ppx, ppy, lambda_parent, edge_map, off_x=0, off_y=0
):
    H, W, _ = A_np.shape
    N = H * W
    if not HAS_SCIPY:
        return mapping_global(A_np, B_np)

    M = base.copy()
    A_flat = A_np.reshape(-1, 3)
    B_flat = B_np.reshape(-1, 3)
    A_lab = rgb_to_lab(A_flat)
    B_lab = rgb_to_lab(B_flat)

    yy_full, xx_full = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x_all = xx_full.reshape(-1)
    y_all = yy_full.reshape(-1)
    edge_flat = edge_map.reshape(-1)

    y_starts = list(range(off_y, H, tile))
    x_starts = list(range(off_x, W, tile))

    for y0 in y_starts:
        y1 = min(H, y0 + tile)
        for x0 in x_starts:
            x1 = min(W, x0 + tile)
            yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing="ij")
            idx = (yy * W + xx).reshape(-1)
            n = idx.size
            if n == 0:
                continue

            A_lab_tile = A_lab[idx]
            B_lab_tile = B_lab[idx]

            A2 = np.sum(A_lab_tile**2, axis=1)[:, None]
            B2 = np.sum(B_lab_tile**2, axis=1)[None, :]
            AB = A_lab_tile @ B_lab_tile.T
            C_color = A2 + B2 - 2 * AB

            Ax = x_all[idx][:, None]
            Ay = y_all[idx][:, None]
            Bx = x_all[idx][None, :]
            By = y_all[idx][None, :]
            C_spat = ((Ax - Bx) ** 2 + (Ay - By) ** 2) / max(1, tile**2)

            ppx_i = ppx[idx][:, None]
            ppy_i = ppy[idx][:, None]
            C_parent = ((Bx - ppx_i) ** 2 + (By - ppy_i) ** 2) / max(1, tile**2)
            w_edge = 1.0 + edge_flat[idx][None, :]

            C = C_color + spatial * C_spat + (lambda_parent * w_edge) * C_parent
            r, c = linear_sum_assignment(C)
            M[idx[r]] = idx[c]

    return np.clip(M, 0, N - 1)


def merge_mappings(M_base, candidates, A_lab, B_lab, H, W, passes=2):
    N = H * W
    inv = np.empty(N, dtype=np.int64)
    inv[M_base] = np.arange(N, dtype=np.int64)

    def cost(i, j_dest):
        d = A_lab[i] - B_lab[j_dest]
        return float(d.dot(d))

    rng = np.random.default_rng(0)
    for _ in range(passes):
        order = rng.permutation(N)
        for i in order:
            j_current = M_base[i]
            best_j = j_current
            best_delta = 0.0
            for M_alt in candidates:
                j_prop = M_alt[i]
                if j_prop == j_current:
                    continue
                k = inv[j_prop]
                c0 = cost(i, j_current) + cost(k, M_base[k])
                c1 = cost(i, j_prop) + cost(k, j_current)
                delta = c1 - c0
                if delta < best_delta:
                    best_delta = delta
                    best_j = j_prop
            if best_j != j_current:
                k = inv[best_j]
                old_j_i = j_current
                M_base[i] = best_j
                M_base[k] = old_j_i
                inv[best_j] = i
                inv[old_j_i] = k
    return M_base


def refine_swaps(M, A_lab, B_lab, H, W, iters=12, radius=2):
    rng = np.random.default_rng(1)
    idxs = np.arange(H * W, dtype=np.int64)
    y = idxs // W
    x = idxs % W

    def cost(i, j_dest):
        d = A_lab[i] - B_lab[j_dest]
        return float(d.dot(d))

    for _ in range(iters):
        order = rng.permutation(idxs)
        for i in order:
            neighbors = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    xn = x[i] + dx
                    yn = y[i] + dy
                    if 0 <= xn < W and 0 <= yn < H:
                        neighbors.append(yn * W + xn)
            if not neighbors:
                continue
            j = neighbors[rng.integers(len(neighbors))]
            c0 = cost(i, M[i]) + cost(j, M[j])
            c1 = cost(i, M[j]) + cost(j, M[i])
            if c1 + 1e-7 < c0:
                M[i], M[j] = M[j], M[i]
    return M


def parent_coords(H, W, Hp, Wp, M_prev):
    x_prev, y_prev = grid_coords(Hp, Wp)
    scale_x = W / Wp
    scale_y = H / Hp
    xf, yf = grid_coords(H, W)
    xc = np.clip((xf / scale_x).round().astype(np.int64), 0, Wp - 1)
    yc = np.clip((yf / scale_y).round().astype(np.int64), 0, Hp - 1)
    ic = yc * Wp + xc
    jp = M_prev[ic]
    ppx = x_prev[jp] * scale_x
    ppy = y_prev[jp] * scale_y
    return ppx.astype(np.float32), ppy.astype(np.float32)


def multiscale_map(
    A_np_full, B_np_full, levels, tiles, lambdas_spatial, lambdas_parent
):
    if not HAS_SCIPY:
        H, W, _ = A_np_full.shape
        return mapping_global(A_np_full, B_np_full), H, W

    Aw_full, Ah_full = B_np_full.shape[1], B_np_full.shape[0]

    def resize_np(np_img, target_w):
        img = Image.fromarray(np_img.astype(np.uint8))
        h = max(1, int(round(target_w * Ah_full / Aw_full)))
        return to_np(img.resize((target_w, h), Image.LANCZOS))

    A_levels = [resize_np(A_np_full, w) for w in levels]
    B_levels = [resize_np(B_np_full, w) for w in levels]
    edges_levels = [edges_from_B(B_levels[i]) for i in range(len(levels))]

    # Coarsest
    A0, B0 = A_levels[0], B_levels[0]
    H0, W0, _ = A0.shape
    tile0 = tiles[0]
    M = mapping_global(A0, B0)
    off = tile0 // 2
    ppx0 = np.zeros(H0 * W0, dtype=np.float32)
    ppy0 = np.zeros(H0 * W0, dtype=np.float32)
    cand0 = [
        mapping_color_local_edgeparent(
            A0,
            B0,
            tile0,
            lambdas_spatial[0],
            M,
            ppx0,
            ppy0,
            0.0,
            edges_levels[0],
            off_x=0,
            off_y=0,
        ),
        mapping_color_local_edgeparent(
            A0,
            B0,
            tile0,
            lambdas_spatial[0],
            M,
            ppx0,
            ppy0,
            0.0,
            edges_levels[0],
            off_x=off,
            off_y=0,
        ),
        mapping_color_local_edgeparent(
            A0,
            B0,
            tile0,
            lambdas_spatial[0],
            M,
            ppx0,
            ppy0,
            0.0,
            edges_levels[0],
            off_x=0,
            off_y=off,
        ),
        mapping_color_local_edgeparent(
            A0,
            B0,
            tile0,
            lambdas_spatial[0],
            M,
            ppx0,
            ppy0,
            0.0,
            edges_levels[0],
            off_x=off,
            off_y=off,
        ),
    ]
    A_lab0 = rgb_to_lab(A0.reshape(-1, 3))
    B_lab0 = rgb_to_lab(B0.reshape(-1, 3))
    M = merge_mappings(M, cand0, A_lab0, B_lab0, H0, W0, passes=2)
    M = refine_swaps(M, A_lab0, B_lab0, H0, W0, iters=12, radius=2)
    prev = (M, H0, W0)

    # Finer levels
    for li in range(1, len(levels)):
        A_, B_ = A_levels[li], B_levels[li]
        H, W, _ = A_.shape
        tile = tiles[li]
        lam_s = lambdas_spatial[li]
        lam_p = lambdas_parent[li]
        M_prev, Hp, Wp = prev
        ppx, ppy = parent_coords(H, W, Hp, Wp, M_prev)
        M_base = mapping_global(A_, B_)
        off = tile // 2
        candidates = [
            mapping_color_local_edgeparent(
                A_,
                B_,
                tile,
                lam_s,
                M_base,
                ppx,
                ppy,
                lam_p,
                edges_levels[li],
                off_x=0,
                off_y=0,
            ),
            mapping_color_local_edgeparent(
                A_,
                B_,
                tile,
                lam_s,
                M_base,
                ppx,
                ppy,
                lam_p,
                edges_levels[li],
                off_x=off,
                off_y=0,
            ),
            mapping_color_local_edgeparent(
                A_,
                B_,
                tile,
                lam_s,
                M_base,
                ppx,
                ppy,
                lam_p,
                edges_levels[li],
                off_x=0,
                off_y=off,
            ),
            mapping_color_local_edgeparent(
                A_,
                B_,
                tile,
                lam_s,
                M_base,
                ppx,
                ppy,
                lam_p,
                edges_levels[li],
                off_x=off,
                off_y=off,
            ),
        ]
        A_lab = rgb_to_lab(A_.reshape(-1, 3))
        B_lab = rgb_to_lab(B_.reshape(-1, 3))
        M = merge_mappings(M_base.copy(), candidates, A_lab, B_lab, H, W, passes=2)
        M = refine_swaps(M, A_lab, B_lab, H, W, iters=10, radius=2)
        prev = (M, H, W)

    return prev  # (M_final, H_final, W_final)


def permute_image(A_np, M):
    H, W, _ = A_np.shape
    N = H * W
    perm_flat = np.zeros_like(A_np.reshape(N, 3))
    perm_flat[M] = A_np.reshape(N, 3)
    return perm_flat.reshape(H, W, 3).astype(np.uint8)


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

    @QtCore.Slot()
    def run(self):
        try:
            import matplotlib

            matplotlib.use("Agg")  # offscreen
            import matplotlib.pyplot as plt
            from matplotlib.animation import PillowWriter

            self.stage.emit("Loading images…")
            A0 = load_rgb(self.pathA)
            B0 = load_rgb(self.pathB)
            A = resize_to_width(A0, self.width)
            Aw, Ah = A.size
            B = B0.resize((Aw, Ah), Image.LANCZOS)
            A_np = to_np(A)
            B_np = to_np(B)
            H, W, _ = A_np.shape
            N = H * W

            # Multiscale mapping
            self.stage.emit("Computing mapping…")
            self.progress.emit(5)
            M_final, Hm, Wm = multiscale_map(
                A_np, B_np, self.levels, self.tiles, self.lam_spatial, self.lam_parent
            )
            if Hm != H or Wm != W:
                raise RuntimeError("Final mapping resolution mismatch.")

            # Prepare animation data
            self.stage.emit("Rendering GIF…")
            x_all, y_all = grid_coords(H, W)
            Sx = x_all.copy()
            Sy = y_all.copy()
            Ex = x_all[M_final]
            Ey = y_all[M_final]
            dx = Ex - Sx
            dy = Ey - Sy
            dist = np.hypot(dx, dy)
            px = np.where(dist > 0, -dy / (dist + 1e-9), 0.0)
            py = np.where(dist > 0, dx / (dist + 1e-9), 1.0)
            off = self.arc * dist
            ox = px * off
            oy = py * off
            paths = (
                Sx,
                Sy,
                Sx + 0.33 * dx + ox,
                Sy + 0.33 * dy + oy,
                Sx + 0.66 * dx + ox,
                Sy + 0.66 * dy + oy,
                Ex,
                Ey,
            )

            colors_A = (A_np.reshape(N, 3) / 255.0).clip(0, 1)
            perm_img = permute_image(A_np, M_final).astype(np.float32) / 255.0

            rng = np.random.default_rng(None)
            start = rng.uniform(0.0, min(0.95, self.stagger), size=N)

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

            def ease(t):
                t = np.clip(t, 0.0, 1.0)
                return 0.5 * (1 - np.cos(np.pi * t))

            def bezier_cubic(Sx, Sy, C1x, C1y, C2x, C2y, Ex, Ey, t):
                u = 1 - t
                uu = u * u
                tt = t * t
                uuu = uu * u
                ttt = tt * t
                bx = uuu * Sx + 3 * uu * t * C1x + 3 * u * tt * C2x + ttt * Ex
                by = uuu * Sy + 3 * uu * t * C1y + 3 * u * tt * C2y + ttt * Ey
                return bx, by

            writer = PillowWriter(fps=30)
            os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
            with writer.saving(fig, self.save_path, dpi=100):
                for frame in range(total_frames):
                    if frame < self.frames:
                        g = frame / max(1, self.frames - 1)
                        t = ease(np.clip((g - start) / (1.0 - start + 1e-9), 0, 1))
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
                    self.progress.emit(int(100 * (frame + 1) / total_frames))

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

        # Scrollable viewer that shows original pixel size (no scaling)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setStyleSheet(
            "QScrollArea { background: #0D1016; border-radius: 8px; }"
        )
        self.scroll.setAlignment(QtCore.Qt.AlignCenter)
        outer.addWidget(self.scroll, 1)

        self.view = QtWidgets.QLabel()
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setScaledContents(False)  # critical: do NOT scale content
        self.scroll.setWidget(self.view)

        self.movie = None
        self.pilTimer = None
        self.pilFrames = []
        self.pilDurations = []
        self.pilIndex = 0

    def show_gif(self, path):
        self.clear_preview()
        # Try QMovie (original-size frames)
        self.movie = QtGui.QMovie(path)
        if self.movie.isValid():
            self.movie.setCacheMode(QtGui.QMovie.CacheAll)
            self.movie.setSpeed(100)
            self.view.setMovie(self.movie)

            # Resize label to the movie frame size on each frame to preserve original pixels
            def fit_to_frame(_):
                rect = self.movie.frameRect()
                self.view.setFixedSize(rect.size())

            self.movie.frameChanged.connect(fit_to_frame)
            self.movie.start()
            self.status.setText(os.path.basename(path))
        else:
            # Fallback: PIL playback, original-size frames
            try:
                gif = Image.open(path)
                self.pilFrames, self.pilDurations = [], []
                for frame in ImageSequence.Iterator(gif):
                    fr_rgb = frame.convert("RGB")
                    np_img = np.array(fr_rgb)
                    H, W = np_img.shape[:2]
                    qimg = QtGui.QImage(
                        np_img.data, W, H, 3 * W, QtGui.QImage.Format_RGB888
                    )
                    pm = QtGui.QPixmap.fromImage(qimg.copy())
                    self.pilFrames.append(pm)
                    self.pilDurations.append(
                        max(10, int(frame.info.get("duration", 33)))
                    )
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
        self.view.setPixmap(pm)  # original size
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
            except:
                pass
            self.movie = None
        if self.pilTimer:
            self.pilTimer.stop()
            self.pilTimer = None
        self.pilFrames, self.pilDurations = [], []
        self.view.clear()
        # Reset size to something minimal until a new image is shown
        self.view.setFixedSize(QtCore.QSize(1, 1))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # No scaling; content remains at original size. QScrollArea will show scrollbars if needed.


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

        arrow = QtWidgets.QLabel("→")
        arrow.setAlignment(QtCore.Qt.AlignCenter)
        arrow.setStyleSheet("color: #3FD0FF; font-size: 28px; font-weight: 700;")
        arrow.setFixedWidth(40)

        top.addWidget(self.cardA, 1)
        top.addWidget(arrow)
        top.addWidget(self.cardB, 1)
        root.addLayout(top)

        # Control bar: Submit + slim progress + minimal knobs
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

        # Minimal controls
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

        # Bottom: Output viewer (original size, scrollable)
        self.output = OutputViewer()
        root.addWidget(self.output, 1)

        self.statusBar().setStyleSheet(
            f"QStatusBar {{ background: {BG}; color: {TEXT}; }}"
        )
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

            # Multiscale defaults tuned for aesthetics
            w2 = max(64, width // 4)
            w3 = max(96, width // 2)
            levels = [w2, w3, width]
            tiles = [12, 12, 10]
            lam_spatial = [0.35, 0.25, 0.18]
            lam_parent = [0.00, 0.20, 0.12]

            self.progress.setValue(0)
            self.statusBar().showMessage("Exporting…")

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

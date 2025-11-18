# duet.py — one-way: use ONLY pixels from A to form B, multiscale edge‑aware mapping with clean animation and final hold
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# SciPy for optimal assignment
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------- Utilities ----------
def load_rgb(path):
    return Image.open(path).convert('RGB')

def resize_to_width(img, width):
    w0, h0 = img.size
    height = max(1, int(round(width * h0 / w0)))
    return img.resize((width, height), Image.LANCZOS)

def to_np(img):
    return np.asarray(img, dtype=np.float32)

def grid_coords(H, W):
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    return x.reshape(-1), y.reshape(-1)

def ease_in_out_cos(t):
    t = np.clip(t, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * t))

def bezier_cubic(Sx, Sy, C1x, C1y, C2x, C2y, Ex, Ey, t):
    u = 1 - t
    uu = u*u
    tt = t*t
    uuu = uu*u
    ttt = tt*t
    bx = uuu*Sx + 3*uu*t*C1x + 3*u*tt*C2x + ttt*Ex
    by = uuu*Sy + 3*uu*t*C1y + 3*u*tt*C2y + ttt*Ey
    return bx, by

def luminance(rgb):
    return 0.2126*rgb[...,0] + 0.7152*rgb[...,1] + 0.0722*rgb[...,2]

# ----- sRGB -> Lab (D65) for perceptual color distances -----
def _srgb_to_linear(c):
    a = 0.055
    return np.where(c <= 0.04045, c/12.92, ((c + a)/(1 + a))**2.4)

def _rgb_to_xyz(rgb):
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    return np.tensordot(rgb, M.T, axes=1)

def _f_lab(t):
    e = (6/29)**3
    k = (29/3)**3/3
    return np.where(t > e, np.cbrt(t), (t*k) + 4/29)

def rgb_to_lab(rgb_uint8):
    rgb = np.clip(rgb_uint8.astype(np.float32)/255.0, 0, 1)
    lin = _srgb_to_linear(rgb)
    xyz = _rgb_to_xyz(lin)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx, fy, fz = _f_lab(xyz[...,0]/Xn), _f_lab(xyz[...,1]/Yn), _f_lab(xyz[...,2]/Zn)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)

# ---------- Edge map (Sobel on B) ----------
def sobel_edges_gray(gray):
    # gray HxW float32 in [0,1]
    H, W = gray.shape
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    # simple convolution (no padding on borders)
    for i in range(1, H-1):
        gi = gray[i-1:i+2]
        gx[i,1:W-1] = np.sum(gi[:,0:3] * Kx, axis=(0,1))
        gy[i,1:W-1] = np.sum(gi[:,0:3] * Ky, axis=(0,1))
    mag = np.sqrt(gx*gx + gy*gy)
    m = mag.max()
    return mag / m if m > 0 else mag

def edges_from_B(B_np):
    lum = luminance(B_np) / 255.0
    return sobel_edges_gray(lum.astype(np.float32))

# ---------- Basic mappings ----------
def mapping_global(A_np, B_np):
    H, W, _ = A_np.shape
    N = H*W
    A_flat = A_np.reshape(N, 3)
    B_flat = B_np.reshape(N, 3)
    lumA = luminance(A_flat)
    lumB = luminance(B_flat)
    rA = np.argsort(lumA, kind='mergesort')
    rB = np.argsort(lumB, kind='mergesort')
    M = np.empty(N, dtype=np.int64)
    M[rA] = rB
    return M

# ---------- Multiscale helpers ----------
def parent_coords(H, W, Hp, Wp, M_prev):
    # For each fine pixel i, where did its parent map at previous level?
    x_prev, y_prev = grid_coords(Hp, Wp)
    scale_x = W / Wp
    scale_y = H / Hp
    # fine pixel coords
    xf, yf = grid_coords(H, W)
    xc = np.clip((xf / scale_x).round().astype(np.int64), 0, Wp-1)
    yc = np.clip((yf / scale_y).round().astype(np.int64), 0, Hp-1)
    ic = yc * Wp + xc
    jp = M_prev[ic]
    ppx = x_prev[jp] * scale_x
    ppy = y_prev[jp] * scale_y
    return ppx.astype(np.float32), ppy.astype(np.float32)

def mapping_color_local_edgeparent(A_np, B_np, tile, spatial, base, ppx, ppy, lambda_parent, edge_map, off_x=0, off_y=0):
    """
    Build a candidate mapping on shifted tiles with Lab + spatial + parent-attraction cost.
    Starts from 'base' and overrides each tile's assignments. Returns a full-length mapping.
    """
    H, W, _ = A_np.shape
    N = H*W
    if not HAS_SCIPY:
        return mapping_global(A_np, B_np)

    M = base.copy()

    A_flat = A_np.reshape(-1, 3)
    B_flat = B_np.reshape(-1, 3)
    A_lab = rgb_to_lab(A_flat)
    B_lab = rgb_to_lab(B_flat)

    yy_full, xx_full = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_all = xx_full.reshape(-1)
    y_all = yy_full.reshape(-1)
    edge_flat = edge_map.reshape(-1)

    # shifted starts with clamping
    y_starts = list(range(off_y, H, tile))
    x_starts = list(range(off_x, W, tile))

    for y0 in y_starts:
        y1 = min(H, y0 + tile)
        for x0 in x_starts:
            x1 = min(W, x0 + tile)
            yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
            idx = (yy*W + xx).reshape(-1)
            n = idx.size
            if n == 0:
                continue

            A_lab_tile = A_lab[idx]
            B_lab_tile = B_lab[idx]

            # Color cost
            A2 = np.sum(A_lab_tile**2, axis=1)[:,None]
            B2 = np.sum(B_lab_tile**2, axis=1)[None,:]
            AB = A_lab_tile @ B_lab_tile.T
            C_color = A2 + B2 - 2*AB  # (n,n)

            # Spatial (short travel)
            Ax = x_all[idx][:,None]
            Ay = y_all[idx][:,None]
            Bx = x_all[idx][None,:]
            By = y_all[idx][None,:]
            C_spat = ((Ax - Bx)**2 + (Ay - By)**2) / max(1, tile**2)

            # Parent attraction (toward previous level's mapped dest), edge-weighted
            ppx_i = ppx[idx][:,None]
            ppy_i = ppy[idx][:,None]
            C_parent = ((Bx - ppx_i)**2 + (By - ppy_i)**2) / max(1, tile**2)
            w_edge = (1.0 + edge_flat[idx][None,:])  # amplify near edges of B

            C = C_color + spatial*C_spat + (lambda_parent * w_edge) * C_parent

            r, c = linear_sum_assignment(C)
            M[idx[r]] = idx[c]

    return np.clip(M, 0, N-1)

def merge_mappings(M_base, candidates, A_lab, B_lab, H, W, passes=2):
    """
    Merge alternate-offset candidates into base by cost-aware swaps, preserving bijection.
    """
    N = H*W
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

def refine_swaps(M, A_lab, B_lab, H, W, iters=20, radius=2):
    rng = np.random.default_rng(1)
    idxs = np.arange(H*W, dtype=np.int64)
    y = idxs // W
    x = idxs % W

    def cost(i, j_dest):
        d = A_lab[i] - B_lab[j_dest]
        return float(d.dot(d))

    for _ in range(iters):
        order = rng.permutation(idxs)
        for i in order:
            neighbors = []
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dx == 0 and dy == 0:
                        continue
                    xn = x[i] + dx
                    yn = y[i] + dy
                    if 0 <= xn < W and 0 <= yn < H:
                        neighbors.append(yn*W + xn)
            if not neighbors:
                continue
            j = neighbors[rng.integers(len(neighbors))]
            c0 = cost(i, M[i]) + cost(j, M[j])
            c1 = cost(i, M[j]) + cost(j, M[i])
            if c1 + 1e-7 < c0:
                M[i], M[j] = M[j], M[i]
    return M

# ---------- Multiscale mapping ----------
def multiscale_map(A_np_full, B_np_full, levels, tiles, lambdas_spatial, lambdas_parent, edge_k=1.0):
    """
    Build a bijective mapping from A to B at full resolution using a coarse->fine schedule.
    levels: list of widths (coarse to fine), ending with full width of A/B.
    tiles: per-level tile sizes (same length as levels).
    lambdas_spatial: per-level spatial weights.
    lambdas_parent: per-level parent attraction weights.
    """
    if not HAS_SCIPY:
        print("[warn] SciPy not available; using global rank mapping.")
        H, W, _ = A_np_full.shape
        return mapping_global(A_np_full, B_np_full), H, W

    # Build level arrays (respect aspect ratio of full images)
    Aw_full, Ah_full = B_np_full.shape[1], B_np_full.shape[0]  # width, height from full
    def resize_np(np_img, target_w):
        img = Image.fromarray(np_img.astype(np.uint8))
        h = max(1, int(round(target_w * Ah_full / Aw_full)))
        return to_np(img.resize((target_w, h), Image.LANCZOS))

    A_levels = [resize_np(A_np_full, w) for w in levels]
    B_levels = [resize_np(B_np_full, w) for w in levels]
    edges_levels = [edges_from_B(B_levels[i]) for i in range(len(levels))]

    # Coarsest: base mapping via local color+spatial
    A0 = A_levels[0]
    B0 = B_levels[0]
    H0, W0, _ = A0.shape
    tile0 = tiles[0]
    M = mapping_global(A0, B0)  # base to start
    off = tile0 // 2
    ppx0 = np.zeros(H0*W0, dtype=np.float32)
    ppy0 = np.zeros(H0*W0, dtype=np.float32)
    cand0 = [
        mapping_color_local_edgeparent(A0, B0, tile0, lambdas_spatial[0], M, ppx0, ppy0, 0.0, edges_levels[0], off_x=0, off_y=0),
        mapping_color_local_edgeparent(A0, B0, tile0, lambdas_spatial[0], M, ppx0, ppy0, 0.0, edges_levels[0], off_x=off, off_y=0),
        mapping_color_local_edgeparent(A0, B0, tile0, lambdas_spatial[0], M, ppx0, ppy0, 0.0, edges_levels[0], off_x=0, off_y=off),
        mapping_color_local_edgeparent(A0, B0, tile0, lambdas_spatial[0], M, ppx0, ppy0, 0.0, edges_levels[0], off_x=off, off_y=off),
    ]
    A_lab0 = rgb_to_lab(A0.reshape(-1,3))
    B_lab0 = rgb_to_lab(B0.reshape(-1,3))
    M = merge_mappings(M, cand0, A_lab0, B_lab0, H0, W0, passes=2)
    M = refine_swaps(M, A_lab0, B_lab0, H0, W0, iters=20, radius=2)
    prev = (M, H0, W0)

    # Finer levels
    for li in range(1, len(levels)):
        A_ = A_levels[li]
        B_ = B_levels[li]
        H, W, _ = A_.shape
        tile = tiles[li]
        lam_s = lambdas_spatial[li]
        lam_p = lambdas_parent[li]
        # Parent attraction coordinates from previous level
        M_prev, Hp, Wp = prev
        ppx, ppy = parent_coords(H, W, Hp, Wp, M_prev)
        # Base mapping (global rank as scaffold)
        M_base = mapping_global(A_, B_)
        # Build candidates at multiple offsets (with small jitter by mixing offsets)
        off = tile // 2
        candidates = [
            mapping_color_local_edgeparent(A_, B_, tile, lam_s, M_base, ppx, ppy, lam_p, edges_levels[li], off_x=0,   off_y=0),
            mapping_color_local_edgeparent(A_, B_, tile, lam_s, M_base, ppx, ppy, lam_p, edges_levels[li], off_x=off, off_y=0),
            mapping_color_local_edgeparent(A_, B_, tile, lam_s, M_base, ppx, ppy, lam_p, edges_levels[li], off_x=0,   off_y=off),
            mapping_color_local_edgeparent(A_, B_, tile, lam_s, M_base, ppx, ppy, lam_p, edges_levels[li], off_x=off, off_y=off),
        ]
        A_lab = rgb_to_lab(A_.reshape(-1,3))
        B_lab = rgb_to_lab(B_.reshape(-1,3))
        M = merge_mappings(M_base.copy(), candidates, A_lab, B_lab, H, W, passes=2)
        M = refine_swaps(M, A_lab, B_lab, H, W, iters=16, radius=2)
        prev = (M, H, W)

    # Return final mapping at full resolution by remapping indices
    # We computed mapping at last level width == levels[-1] which should match target working width in animate
    return prev  # (M_final, H_final, W_final)

# ---------- Animation (one-way) ----------
def animate_permutation(pathA, pathB, width=260, frames=120, hold_end=24, gif_path=None,
                        levels=None, tiles=None, lambdas_spatial=None, lambdas_parent=None,
                        stagger=0.35, arc=0.10, seed=None):
    # Load and align sizes
    A0 = load_rgb(pathA)
    B0 = load_rgb(pathB)
    A = resize_to_width(A0, width)
    Aw, Ah = A.size
    B = B0.resize((Aw, Ah), Image.LANCZOS)

    A_np = to_np(A)
    B_np = to_np(B)
    H, W, _ = A_np.shape
    N = H * W

    # Defaults for multiscale
    if levels is None:
        # Coarse->fine widths (end at working width)
        w2 = max(64, width // 4)
        w3 = max(96, width // 2)
        levels = [w2, w3, width]
    if tiles is None:
        tiles = [12, 12, 10]  # per-level tile size
    if lambdas_spatial is None:
        lambdas_spatial = [0.35, 0.25, 0.18]
    if lambdas_parent is None:
        lambdas_parent = [0.00, 0.20, 0.12]

    # Colors: ONLY from A
    colors_A = (A_np.reshape(N, 3) / 255.0).clip(0, 1)

    # Multiscale mapping
    M_final, Hm, Wm = multiscale_map(A_np, B_np, levels, tiles, lambdas_spatial, lambdas_parent)
    # Sanity: ensure final mapping resolution equals current
    assert Hm == H and Wm == W, "Final mapping resolution must match working image size"

    # Paths
    x_all, y_all = grid_coords(H, W)
    Sx = x_all.copy()
    Sy = y_all.copy()
    Ex = x_all[M_final]
    Ey = y_all[M_final]
    dx = Ex - Sx
    dy = Ey - Sy
    dist = np.hypot(dx, dy)
    px = np.where(dist > 0, -dy / (dist + 1e-9), 0.0)
    py = np.where(dist > 0,  dx / (dist + 1e-9), 1.0)
    off = arc * dist
    ox = px * off
    oy = py * off
    paths = (Sx, Sy, Sx + 0.33*dx + ox, Sy + 0.33*dy + oy,
             Sx + 0.66*dx + ox, Sy + 0.66*dy + oy, Ex, Ey)

    # Final permuted image (for hold frames)
    perm_flat = np.zeros_like(A_np.reshape(N, 3))
    perm_flat[M_final] = A_np.reshape(N, 3)
    perm_img = perm_flat.reshape(H, W, 3) / 255.0

    # Staggered starts
    rng = np.random.default_rng(seed)
    start = rng.uniform(0.0, stagger, size=N)

    # Figure (white background, crisp)
    fig, ax = plt.subplots(figsize=(6, 6 * H / W))
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Opaque square markers large enough to cover the grid
    s = max(2, int(180000 / N))
    scat = ax.scatter(Sx, Sy, c=colors_A, s=s,
                      marker='s', linewidths=0, edgecolors='none', alpha=1.0, zorder=3)

    # Pre-create final image artist (hidden) so update always returns both
    im = ax.imshow(perm_img, extent=(-0.5, W - 0.5, H - 0.5, -0.5),
                   zorder=1, interpolation='nearest', visible=False)

    total_frames = frames + max(0, hold_end)

    def update(frame):
        if frame < frames:
            g = frame / max(1, frames - 1)
            t = ease_in_out_cos(np.clip((g - start) / (1.0 - start + 1e-9), 0, 1))
            x, y = bezier_cubic(*paths, t)
            scat.set_offsets(np.column_stack([x, y]))
            if im.get_visible(): im.set_visible(False)
            if not scat.get_visible(): scat.set_visible(True)
        else:
            if not im.get_visible(): im.set_visible(True)
            if scat.get_visible(): scat.set_visible(False)
        return (scat, im)

    ani = FuncAnimation(fig, update, frames=total_frames, interval=33, blit=True)

    if gif_path:
        writer = PillowWriter(fps=30)
        ani.save(gif_path, writer=writer)
        print(f"Saved GIF to {gif_path}")

    plt.show()

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Form target image B using ONLY pixels from source image A (multiscale edge-aware permutation)")
    p.add_argument("image_a", help="Path to source image A")
    p.add_argument("image_b", help="Path to target image B")
    p.add_argument("--width", type=int, default=260)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--hold", type=int, default=24)
    p.add_argument("--stagger", type=float, default=0.35)
    p.add_argument("--arc", type=float, default=0.10)
    p.add_argument("--save", type=str, default="")
    p.add_argument("--seed", type=int, default=None)
    # Advanced multiscale controls
    p.add_argument("--levels", type=str, default="", help="Comma-separated widths coarse->fine (end must equal --width)")
    p.add_argument("--tiles", type=str, default="", help="Comma-separated tile sizes per level")
    p.add_argument("--lam-spatial", type=str, default="", help="Comma-separated spatial weights per level")
    p.add_argument("--lam-parent", type=str, default="", help="Comma-separated parent-attraction weights per level")
    args = p.parse_args()

    # Parse optional per-level params
    def parse_list(s):
        return [int(x) for x in s.split(",")] if s else None
    def parse_list_f(s):
        return [float(x) for x in s.split(",")] if s else None

    levels = parse_list(args.levels)
    tiles = parse_list(args.tiles)
    lam_s = parse_list_f(args.lam_spatial)
    lam_p = parse_list_f(args.lam_parent)

    if levels is not None and levels[-1] != args.width:
        raise ValueError("levels must end with the working --width")

    animate_permutation(
        args.image_a, args.image_b,
        width=args.width, frames=args.frames, hold_end=max(0, args.hold),
        gif_path=(args.save or None),
        levels=levels, tiles=tiles, lambdas_spatial=lam_s, lambdas_parent=lam_p,
        stagger=np.clip(args.stagger, 0.0, 0.95),
        arc=max(0.0, args.arc),
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
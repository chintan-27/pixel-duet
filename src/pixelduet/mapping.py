"""Pixel mapping: Hungarian assignment, merge, refinement, and multiscale orchestration."""

import numpy as np
from PIL import Image

from pixelduet.color import luminance, rgb_to_lab
from pixelduet.edges import edges_from_image
from pixelduet.utils import grid_coords, to_np

try:
    from scipy.optimize import linear_sum_assignment

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ---------- Basic mappings ----------


def mapping_global(A_np, B_np):
    """Luminance-rank bijection: sort A and B pixels by brightness and pair them."""
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


# ---------- Multiscale helpers ----------


def parent_coords(H, W, Hp, Wp, M_prev):
    """For each fine pixel, compute where its parent mapped at the previous (coarser) level."""
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


def mapping_color_local_edgeparent(
    A_np, B_np, tile, spatial, base, ppx, ppy, lambda_parent, edge_map, off_x=0, off_y=0
):
    """Build a candidate mapping on shifted tiles with Lab + spatial + parent-attraction cost.

    Starts from ``base`` and overrides each tile's assignments.
    Returns a full-length mapping array.
    """
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

            # Color cost
            A2 = np.sum(A_lab_tile**2, axis=1)[:, None]
            B2 = np.sum(B_lab_tile**2, axis=1)[None, :]
            AB = A_lab_tile @ B_lab_tile.T
            C_color = A2 + B2 - 2 * AB

            # Spatial (short travel)
            Ax = x_all[idx][:, None]
            Ay = y_all[idx][:, None]
            Bx = x_all[idx][None, :]
            By = y_all[idx][None, :]
            C_spat = ((Ax - Bx) ** 2 + (Ay - By) ** 2) / max(1, tile**2)

            # Parent attraction, edge-weighted
            ppx_i = ppx[idx][:, None]
            ppy_i = ppy[idx][:, None]
            C_parent = ((Bx - ppx_i) ** 2 + (By - ppy_i) ** 2) / max(1, tile**2)
            w_edge = 1.0 + edge_flat[idx][None, :]

            C = C_color + spatial * C_spat + (lambda_parent * w_edge) * C_parent

            r, c = linear_sum_assignment(C)
            M[idx[r]] = idx[c]

    return np.clip(M, 0, N - 1)


def merge_mappings(M_base, candidates, A_lab, B_lab, H, W, passes=2):
    """Merge alternate-offset candidates into base by cost-aware swaps, preserving bijection."""
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


def refine_swaps(M, A_lab, B_lab, H, W, iters=20, radius=2):
    """Local pairwise swap refinement to smooth seams."""
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


# ---------- Multiscale orchestrator ----------


def multiscale_map(
    A_np_full,
    B_np_full,
    levels,
    tiles,
    lambdas_spatial,
    lambdas_parent,
    refine_iters=20,
    progress_callback=None,
):
    """Build a bijective mapping from A to B using a coarse-to-fine schedule.

    Parameters
    ----------
    A_np_full, B_np_full : ndarray (H, W, 3) float32
        Source and target images at full resolution.
    levels : list[int]
        Widths from coarse to fine. Last must match the working width.
    tiles : list[int]
        Per-level tile sizes.
    lambdas_spatial : list[float]
        Per-level spatial-distance weights.
    lambdas_parent : list[float]
        Per-level parent-attraction weights.
    refine_iters : int
        Number of local refinement iterations per level.
    progress_callback : callable, optional
        Called with (stage_name: str, fraction: float) to report progress.

    Returns
    -------
    (M_final, H_final, W_final) : tuple
    """
    if not HAS_SCIPY:
        if progress_callback:
            progress_callback("fallback", 1.0)
        H, W, _ = A_np_full.shape
        return mapping_global(A_np_full, B_np_full), H, W

    Aw_full, Ah_full = B_np_full.shape[1], B_np_full.shape[0]

    def resize_np(np_img, target_w):
        img = Image.fromarray(np_img.astype(np.uint8))
        h = max(1, round(target_w * Ah_full / Aw_full))
        return to_np(img.resize((target_w, h), Image.LANCZOS))

    n_levels = len(levels)
    A_levels = [resize_np(A_np_full, w) for w in levels]
    B_levels = [resize_np(B_np_full, w) for w in levels]
    edges_levels = [edges_from_image(B_levels[i]) for i in range(n_levels)]

    def _report(stage, frac):
        if progress_callback:
            progress_callback(stage, frac)

    # Coarsest level
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
            off_x=ox,
            off_y=oy,
        )
        for ox, oy in [(0, 0), (off, 0), (0, off), (off, off)]
    ]
    A_lab0 = rgb_to_lab(A0.reshape(-1, 3))
    B_lab0 = rgb_to_lab(B0.reshape(-1, 3))
    M = merge_mappings(M, cand0, A_lab0, B_lab0, H0, W0, passes=2)
    M = refine_swaps(M, A_lab0, B_lab0, H0, W0, iters=refine_iters, radius=2)
    prev = (M, H0, W0)
    _report("mapping", 1 / n_levels)

    # Finer levels
    for li in range(1, n_levels):
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
                off_x=ox,
                off_y=oy,
            )
            for ox, oy in [(0, 0), (off, 0), (0, off), (off, off)]
        ]
        A_lab = rgb_to_lab(A_.reshape(-1, 3))
        B_lab = rgb_to_lab(B_.reshape(-1, 3))
        M = merge_mappings(M_base.copy(), candidates, A_lab, B_lab, H, W, passes=2)
        iters = max(10, refine_iters - 4 * li)  # slightly fewer at finer levels
        M = refine_swaps(M, A_lab, B_lab, H, W, iters=iters, radius=2)
        prev = (M, H, W)
        _report("mapping", (li + 1) / n_levels)

    return prev

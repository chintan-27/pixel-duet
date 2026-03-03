"""Animation path computation: Bezier curves, easing, staggered starts."""

import numpy as np

from pixelduet.utils import grid_coords

# ---------- Easing functions ----------
# All accept and return numpy arrays, mapping [0,1] -> [0,1].


def ease_linear(t):
    """Linear: constant speed, no acceleration."""
    return np.clip(t, 0.0, 1.0)


def ease_in_out_cos(t):
    """Cosine ease-in-out: smooth 0->1 transition."""
    t = np.clip(t, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * t))


def ease_bounce(t):
    """Bounce out: settles like a bouncing ball."""
    t = np.clip(t, 0.0, 1.0)
    result = np.empty_like(t, dtype=np.float64)
    m1 = t < 1 / 2.75
    m2 = (~m1) & (t < 2 / 2.75)
    m3 = (~m1) & (~m2) & (t < 2.5 / 2.75)
    m4 = (~m1) & (~m2) & (~m3)
    result[m1] = 7.5625 * t[m1] ** 2
    t2 = t[m2] - 1.5 / 2.75
    result[m2] = 7.5625 * t2 * t2 + 0.75
    t3 = t[m3] - 2.25 / 2.75
    result[m3] = 7.5625 * t3 * t3 + 0.9375
    t4 = t[m4] - 2.625 / 2.75
    result[m4] = 7.5625 * t4 * t4 + 0.984375
    return result


def ease_elastic(t):
    """Elastic ease-out: overshoots and springs back."""
    t = np.clip(t, 0.0, 1.0)
    p = 0.3
    s = p / 4
    result = np.where(
        t <= 0,
        0.0,
        np.where(
            t >= 1,
            1.0,
            2.0 ** (-10 * t) * np.sin((t - s) * (2 * np.pi) / p) + 1.0,
        ),
    )
    return np.clip(result, 0.0, 1.2)  # allow slight overshoot for elastic feel


def ease_step(t, steps=8):
    """Step: quantized motion in discrete jumps."""
    t = np.clip(t, 0.0, 1.0)
    return np.minimum(np.floor(t * steps) / (steps - 1), 1.0)


def ease_expo(t):
    """Exponential ease-out: slow start, explosive finish."""
    t = np.clip(t, 0.0, 1.0)
    return np.where(t >= 1.0, 1.0, 1.0 - 2.0 ** (-10 * t))


# ---------- Easing dispatcher ----------

EASINGS = {
    "linear": ease_linear,
    "cosine": ease_in_out_cos,
    "bounce": ease_bounce,
    "elastic": ease_elastic,
    "step": ease_step,
    "expo": ease_expo,
}


def get_easing(name="cosine"):
    """Return an easing function by name. Defaults to cosine.

    Available: linear, cosine, bounce, elastic, step, expo.
    """
    fn = EASINGS.get(name)
    if fn is None:
        raise ValueError(f"Unknown easing '{name}'. Available: {', '.join(EASINGS)}")
    return fn


# ---------- Bezier ----------


def bezier_cubic(Sx, Sy, C1x, C1y, C2x, C2y, Ex, Ey, t):
    """Evaluate a cubic Bezier curve at parameter t (scalar or array)."""
    u = 1 - t
    uu = u * u
    tt = t * t
    uuu = uu * u
    ttt = tt * t
    bx = uuu * Sx + 3 * uu * t * C1x + 3 * u * tt * C2x + ttt * Ex
    by = uuu * Sy + 3 * uu * t * C1y + 3 * u * tt * C2y + ttt * Ey
    return bx, by


# ---------- Path computation ----------


def compute_paths(H, W, M_final, arc=0.10):
    """Compute Bezier control points for all pixels given a mapping.

    Returns
    -------
    paths : tuple of 8 arrays (Sx, Sy, C1x, C1y, C2x, C2y, Ex, Ey)
    """
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
    off = arc * dist
    ox = px * off
    oy = py * off
    return (
        Sx,
        Sy,
        Sx + 0.33 * dx + ox,
        Sy + 0.33 * dy + oy,
        Sx + 0.66 * dx + ox,
        Sy + 0.66 * dy + oy,
        Ex,
        Ey,
    )


# ---------- Stagger + frame helpers ----------


def compute_stagger(N, stagger=0.35, seed=None):
    """Generate per-pixel stagger start times in [0, stagger]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, min(0.95, stagger), size=N)


def frame_positions(paths, start, frame, total_motion_frames, easing="cosine"):
    """Compute (x, y) positions for all pixels at a given frame.

    Returns None if this is a hold frame (past the motion phase).
    """
    if frame >= total_motion_frames:
        return None
    ease_fn = get_easing(easing)
    g = frame / max(1, total_motion_frames - 1)
    t = ease_fn(np.clip((g - start) / (1.0 - start + 1e-9), 0, 1))
    return bezier_cubic(*paths, t)

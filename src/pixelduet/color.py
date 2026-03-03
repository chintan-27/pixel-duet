"""Color space conversions: sRGB -> linear -> XYZ -> CIELAB (D65)."""

import numpy as np


def luminance(rgb):
    """Photometric luminance from RGB (any scale)."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


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
    """Convert an (N, 3) uint8 RGB array to CIELAB float32."""
    rgb = np.clip(rgb_uint8.astype(np.float32) / 255.0, 0, 1)
    lin = _srgb_to_linear(rgb)
    xyz = _rgb_to_xyz(lin)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    fx = _f_lab(xyz[..., 0] / Xn)
    fy = _f_lab(xyz[..., 1] / Yn)
    fz = _f_lab(xyz[..., 2] / Zn)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1).astype(np.float32)

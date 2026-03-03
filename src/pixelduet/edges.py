"""Edge detection on target images (Sobel-based)."""

import numpy as np

from pixelduet.color import luminance

try:
    from scipy.ndimage import sobel as nd_sobel

    HAS_SCIPY_NDIMAGE = True
except Exception:
    HAS_SCIPY_NDIMAGE = False


def _sobel_edges_gray(gray):
    """Pure-numpy Sobel edge magnitude on a float32 HxW grayscale image in [0,1]."""
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    return mag / m if m > 0 else mag


def edges_from_image(img_np):
    """Compute a normalised edge-magnitude map from an (H, W, 3) float32 RGB image.

    Uses scipy.ndimage.sobel when available, falling back to np.gradient.
    """
    lum = (luminance(img_np) / 255.0).astype(np.float32)
    if HAS_SCIPY_NDIMAGE:
        gx = nd_sobel(lum, axis=1, mode="reflect")
        gy = nd_sobel(lum, axis=0, mode="reflect")
        mag = np.sqrt(gx * gx + gy * gy)
    else:
        gy, gx = np.gradient(lum)
        mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    return mag / m if m > 0 else mag

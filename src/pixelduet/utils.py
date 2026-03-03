"""Image I/O and geometric helpers."""

import numpy as np
from PIL import Image


def load_rgb(path):
    """Load an image file and convert to RGB PIL Image."""
    return Image.open(path).convert("RGB")


def resize_to_width(img, width):
    """Resize a PIL Image to the given width, preserving aspect ratio."""
    w0, h0 = img.size
    height = max(1, round(width * h0 / w0))
    return img.resize((width, height), Image.LANCZOS)


def to_np(img):
    """Convert a PIL Image to a float32 numpy array."""
    return np.asarray(img, dtype=np.float32)


def grid_coords(H, W):
    """Return flat (x, y) coordinate arrays for an HxW grid."""
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return x.reshape(-1), y.reshape(-1)


def permute_image(A_np, M):
    """Apply a pixel mapping M to image A, returning a uint8 (H, W, 3) array."""
    H, W, _ = A_np.shape
    N = H * W
    perm_flat = np.zeros_like(A_np.reshape(N, 3))
    perm_flat[M] = A_np.reshape(N, 3)
    return perm_flat.reshape(H, W, 3).astype(np.uint8)

"""Pixel Duet — rearrange pixels from Image A to form Image B with animated transitions.

Public API
----------
    pixel_duet(path_a, path_b, ...) -> str
        High-level: load images, compute mapping, render output. Returns output path.

    map_pixels(A_np, B_np, ...) -> (M, H, W)
        Compute the bijective pixel mapping from A to B.

    render(A_np, M_final, H, W, output_path, ...) -> str
        Render an animation to GIF, MP4, or WebM.

Submodules
----------
    pixelduet.color      — sRGB/Lab colour conversion
    pixelduet.edges      — Edge detection
    pixelduet.mapping    — Hungarian assignment, multiscale mapping
    pixelduet.animation  — Bezier paths, easing
    pixelduet.render     — GIF/MP4/WebM output backends
    pixelduet.utils      — Image I/O, geometry helpers
"""

from pixelduet.mapping import multiscale_map
from pixelduet.render import render
from pixelduet.utils import load_rgb, resize_to_width, to_np

__version__ = "0.1.0"

__all__ = [
    "load_rgb",
    "map_pixels",
    "multiscale_map",
    "pixel_duet",
    "pixel_duet_chain",
    "render",
    "resize_to_width",
    "to_np",
]


# Default multiscale parameters
DEFAULT_LEVELS = None  # auto-computed from width
DEFAULT_TILES = [12, 12, 10]
DEFAULT_LAM_SPATIAL = [0.35, 0.25, 0.18]
DEFAULT_LAM_PARENT = [0.00, 0.20, 0.12]


def _auto_levels(width):
    """Compute default coarse-to-fine level widths from working width."""
    w2 = max(64, width // 4)
    w3 = max(96, width // 2)
    return [w2, w3, width]


def map_pixels(
    A_np,
    B_np,
    levels=None,
    tiles=None,
    lambdas_spatial=None,
    lambdas_parent=None,
    refine_iters=20,
    progress_callback=None,
):
    """Compute a bijective pixel mapping from source image A to target image B.

    Parameters
    ----------
    A_np, B_np : ndarray (H, W, 3) float32
        Source and target images (same dimensions).
    levels : list[int], optional
        Multiscale widths (coarse to fine). Auto-computed if None.
    tiles, lambdas_spatial, lambdas_parent : list, optional
        Per-level parameters. Defaults used if None.
    refine_iters : int
        Refinement swap iterations per level.
    progress_callback : callable, optional
        Called with (stage: str, fraction: float).

    Returns
    -------
    M : ndarray of int64
        Bijective mapping: pixel i in A maps to position M[i] in B.
    H, W : int
        Image dimensions.
    """
    _H, W, _ = A_np.shape
    if levels is None:
        levels = _auto_levels(W)
    if tiles is None:
        tiles = DEFAULT_TILES[: len(levels)]
        while len(tiles) < len(levels):
            tiles.append(tiles[-1])
    if lambdas_spatial is None:
        lambdas_spatial = DEFAULT_LAM_SPATIAL[: len(levels)]
        while len(lambdas_spatial) < len(levels):
            lambdas_spatial.append(lambdas_spatial[-1])
    if lambdas_parent is None:
        lambdas_parent = DEFAULT_LAM_PARENT[: len(levels)]
        while len(lambdas_parent) < len(levels):
            lambdas_parent.append(lambdas_parent[-1])

    return multiscale_map(
        A_np,
        B_np,
        levels,
        tiles,
        lambdas_spatial,
        lambdas_parent,
        refine_iters=refine_iters,
        progress_callback=progress_callback,
    )


def pixel_duet(
    path_a,
    path_b,
    output_path="output/swap.gif",
    width=260,
    frames=120,
    hold=24,
    stagger=0.35,
    arc=0.10,
    seed=None,
    levels=None,
    tiles=None,
    lambdas_spatial=None,
    lambdas_parent=None,
    fps=30,
    easing="cosine",
    progress_callback=None,
):
    """End-to-end: load two images, compute mapping, render animation.

    Parameters
    ----------
    path_a, path_b : str
        Paths to source (A) and target (B) images.
    output_path : str
        Where to write the output. Extension determines format (.gif, .mp4, .webm).
    width : int
        Working width in pixels.
    frames : int
        Number of animation (motion) frames.
    hold : int
        Number of hold frames at the end.
    stagger : float
        Random stagger in start times (0..1).
    arc : float
        Path curvature (0 = straight lines).
    seed : int, optional
        RNG seed for reproducibility.
    levels, tiles, lambdas_spatial, lambdas_parent : list, optional
        Multiscale parameters.
    fps : int
        Output frame rate.
    easing : str
        Easing function name (linear, cosine, bounce, elastic, step, expo).
    progress_callback : callable, optional
        Called with (stage: str, fraction: float).

    Returns
    -------
    str
        The output file path.
    """
    from PIL import Image as PILImage

    A0 = load_rgb(path_a)
    B0 = load_rgb(path_b)
    A = resize_to_width(A0, width)
    Aw, Ah = A.size
    B = B0.resize((Aw, Ah), PILImage.LANCZOS)

    A_np = to_np(A)
    B_np = to_np(B)
    H, W, _ = A_np.shape

    M_final, Hm, Wm = map_pixels(
        A_np,
        B_np,
        levels=levels,
        tiles=tiles,
        lambdas_spatial=lambdas_spatial,
        lambdas_parent=lambdas_parent,
        progress_callback=progress_callback,
    )
    if Hm != H or Wm != W:
        raise RuntimeError("Final mapping resolution mismatch.")

    return render(
        A_np,
        M_final,
        H,
        W,
        output_path,
        frames=frames,
        hold=hold,
        stagger=stagger,
        arc=arc,
        seed=seed,
        fps=fps,
        easing=easing,
        progress_callback=progress_callback,
    )


def pixel_duet_chain(
    image_paths,
    output_path="output/chain.gif",
    width=260,
    frames_per_segment=120,
    hold_per_segment=24,
    hold_end=24,
    stagger=0.35,
    arc=0.10,
    seed=None,
    levels=None,
    tiles=None,
    lambdas_spatial=None,
    lambdas_parent=None,
    fps=30,
    easing="cosine",
    progress_callback=None,
):
    """Chain multiple images: A->B->C->..., producing one continuous animation.

    The pixel colours from Image A are preserved throughout the entire chain.
    Each segment remaps the current pixel arrangement to form the next target.

    Parameters
    ----------
    image_paths : list[str]
        Paths to images. Minimum 2. The first is the colour source; the rest are
        successive structure targets.
    output_path : str
        Where to write the output (.gif, .mp4, .webm).
    width : int
        Working width in pixels.
    frames_per_segment : int
        Motion frames per transition.
    hold_per_segment : int
        Hold frames between transitions (to reveal each result).
    hold_end : int
        Extra hold frames at the very end.
    stagger, arc, seed, levels, tiles, lambdas_spatial, lambdas_parent, fps :
        Same as pixel_duet().
    easing : str
        Easing function name (linear, cosine, bounce, elastic, step, expo).
    progress_callback : callable, optional
        Called with (stage: str, fraction: float).

    Returns
    -------
    str
        The output file path.
    """
    import numpy as np
    from PIL import Image as PILImage

    from pixelduet.render import render_chain as _render_chain

    if len(image_paths) < 2:
        raise ValueError("pixel_duet_chain requires at least 2 image paths")

    # Load and resize all images to the same dimensions
    imgs_pil = [load_rgb(p) for p in image_paths]
    first = resize_to_width(imgs_pil[0], width)
    Aw, Ah = first.size
    imgs_np = [to_np(first)]
    for img in imgs_pil[1:]:
        imgs_np.append(to_np(img.resize((Aw, Ah), PILImage.LANCZOS)))

    H, W, _ = imgs_np[0].shape
    n_segments = len(imgs_np) - 1

    # Compute mappings for each segment
    # segment i maps from current arrangement to imgs_np[i+1]
    segments = []  # list of (A_np_for_colors, M_final)
    cumulative_M = np.arange(H * W, dtype=np.int64)  # identity

    def _report(stage, frac):
        if progress_callback:
            progress_callback(stage, frac)

    for i in range(n_segments):
        _report("mapping", i / n_segments)

        # Current pixel arrangement: A's pixels permuted by cumulative_M
        # We need to map FROM the current arrangement TO imgs_np[i+1]
        # The "source" for the mapping is the permuted image
        from pixelduet.utils import permute_image

        current_img = permute_image(imgs_np[0], cumulative_M).astype(np.float32)
        target_img = imgs_np[i + 1]

        M_seg, Hm, Wm = map_pixels(
            current_img,
            target_img,
            levels=levels,
            tiles=tiles,
            lambdas_spatial=lambdas_spatial,
            lambdas_parent=lambdas_parent,
            progress_callback=lambda stage, frac, seg=i: _report(
                "mapping", (seg + frac) / n_segments
            ),
        )
        if Hm != H or Wm != W:
            raise RuntimeError(f"Mapping resolution mismatch at segment {i}")

        segments.append((current_img, M_seg))

        # Update cumulative mapping: compose
        cumulative_M = cumulative_M[M_seg]

    _report("mapping", 1.0)

    return _render_chain(
        imgs_np[0],
        segments,
        H,
        W,
        output_path,
        frames_per_segment=frames_per_segment,
        hold_per_segment=hold_per_segment,
        hold_end=hold_end,
        stagger=stagger,
        arc=arc,
        seed=seed,
        fps=fps,
        easing=easing,
        progress_callback=progress_callback,
    )

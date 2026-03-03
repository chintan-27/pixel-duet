"""Render backends: GIF (matplotlib/Pillow) and MP4/WebM (imageio-ffmpeg)."""

import os

import numpy as np

from pixelduet.animation import bezier_cubic, compute_paths, compute_stagger, get_easing
from pixelduet.utils import permute_image


def _generate_frames(
    A_np,
    M_final,
    H,
    W,
    frames,
    hold,
    stagger,
    arc,
    seed,
    easing="cosine",
    progress_callback=None,
):
    """Yield (H, W, 3) uint8 numpy frames for the full animation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ease_fn = get_easing(easing)
    N = H * W
    colors_A = (A_np.reshape(N, 3) / 255.0).clip(0, 1)
    perm_img = permute_image(A_np, M_final).astype(np.float32) / 255.0

    paths = compute_paths(H, W, M_final, arc=arc)
    Sx, Sy = paths[0], paths[1]
    start = compute_stagger(N, stagger=stagger, seed=seed)

    total_frames = frames + max(0, hold)

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

    fig.canvas.draw()

    for frame in range(total_frames):
        if frame < frames:
            g = frame / max(1, frames - 1)
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

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)[..., :3].copy()
        yield arr

        if progress_callback:
            progress_callback("rendering", (frame + 1) / total_frames)

    plt.close(fig)


def render_gif(
    A_np,
    M_final,
    H,
    W,
    output_path,
    frames=120,
    hold=24,
    stagger=0.35,
    arc=0.10,
    seed=None,
    fps=30,
    easing="cosine",
    progress_callback=None,
):
    """Render the animation as a GIF using matplotlib + PillowWriter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    ease_fn = get_easing(easing)
    N = H * W
    colors_A = (A_np.reshape(N, 3) / 255.0).clip(0, 1)
    perm_img = permute_image(A_np, M_final).astype(np.float32) / 255.0

    paths = compute_paths(H, W, M_final, arc=arc)
    Sx, Sy = paths[0], paths[1]
    start = compute_stagger(N, stagger=stagger, seed=seed)

    total_frames = frames + max(0, hold)

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

    def update(frame):
        if frame < frames:
            g = frame / max(1, frames - 1)
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
        if progress_callback:
            progress_callback("rendering", (frame + 1) / total_frames)
        return (scat, im)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ani = FuncAnimation(fig, update, frames=total_frames, interval=int(1000 / fps), blit=True)
    writer = PillowWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    return output_path


def render_mp4(
    A_np,
    M_final,
    H,
    W,
    output_path,
    frames=120,
    hold=24,
    stagger=0.35,
    arc=0.10,
    seed=None,
    fps=30,
    easing="cosine",
    progress_callback=None,
):
    """Render the animation as MP4 using imageio-ffmpeg."""
    try:
        import imageio.v3 as iio
    except ImportError as err:
        raise ImportError(
            "MP4 output requires imageio with ffmpeg plugin. "
            "Install with: pip install imageio[ffmpeg]"
        ) from err

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_gen = _generate_frames(
        A_np,
        M_final,
        H,
        W,
        frames,
        hold,
        stagger,
        arc,
        seed,
        easing=easing,
        progress_callback=progress_callback,
    )
    frame_list = list(frame_gen)
    iio.imwrite(output_path, frame_list, fps=fps, codec="libx264", plugin="pyav")
    return output_path


def render_webm(
    A_np,
    M_final,
    H,
    W,
    output_path,
    frames=120,
    hold=24,
    stagger=0.35,
    arc=0.10,
    seed=None,
    fps=30,
    easing="cosine",
    progress_callback=None,
):
    """Render the animation as WebM (VP9) using imageio-ffmpeg."""
    try:
        import imageio.v3 as iio
    except ImportError as err:
        raise ImportError(
            "WebM output requires imageio with ffmpeg plugin. "
            "Install with: pip install imageio[ffmpeg]"
        ) from err

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_gen = _generate_frames(
        A_np,
        M_final,
        H,
        W,
        frames,
        hold,
        stagger,
        arc,
        seed,
        easing=easing,
        progress_callback=progress_callback,
    )
    frame_list = list(frame_gen)
    iio.imwrite(output_path, frame_list, fps=fps, codec="libvpx-vp9", plugin="pyav")
    return output_path


# Dispatch by extension
RENDERERS = {
    ".gif": render_gif,
    ".mp4": render_mp4,
    ".webm": render_webm,
}


def render(
    A_np,
    M_final,
    H,
    W,
    output_path,
    frames=120,
    hold=24,
    stagger=0.35,
    arc=0.10,
    seed=None,
    fps=30,
    easing="cosine",
    progress_callback=None,
):
    """Auto-dispatch to the correct renderer based on file extension."""
    ext = os.path.splitext(output_path)[1].lower()
    renderer = RENDERERS.get(ext)
    if renderer is None:
        raise ValueError(f"Unsupported output format '{ext}'. Supported: {', '.join(RENDERERS)}")
    return renderer(
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


# ---------- Chain rendering (multi-segment) ----------


def _generate_chain_frames(
    A_np_original,
    segments,
    H,
    W,
    frames_per_segment,
    hold_per_segment,
    hold_end,
    stagger,
    arc,
    seed,
    easing="cosine",
    progress_callback=None,
):
    """Yield (H, W, 3) uint8 numpy frames for a multi-segment chain animation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pixelduet.animation import compute_paths, compute_stagger, get_easing
    from pixelduet.utils import grid_coords, permute_image

    ease_fn = get_easing(easing)
    N = H * W
    n_segments = len(segments)
    total_frames = n_segments * (frames_per_segment + hold_per_segment) + hold_end

    fig, ax = plt.subplots(figsize=(6, 6 * H / W))
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    s = max(2, int(180000 / N))
    colors_A = (A_np_original.reshape(N, 3) / 255.0).clip(0, 1)
    x_all, y_all = grid_coords(H, W)
    current_color_order = np.arange(N, dtype=np.int64)

    scat = ax.scatter(
        x_all,
        y_all,
        c=colors_A[current_color_order],
        s=s,
        marker="s",
        linewidths=0,
        edgecolors="none",
        alpha=1.0,
        zorder=3,
    )
    im = ax.imshow(
        np.zeros((H, W, 3)),
        extent=(-0.5, W - 0.5, H - 0.5, -0.5),
        zorder=1,
        interpolation="nearest",
        visible=False,
    )

    fig.canvas.draw()
    frame_counter = 0

    for _seg_idx, (source_img, M_seg) in enumerate(segments):
        paths = compute_paths(H, W, M_seg, arc=arc)
        start = compute_stagger(N, stagger=stagger, seed=seed)
        perm_img = permute_image(source_img, M_seg).astype(np.float32) / 255.0

        # Motion frames
        for f in range(frames_per_segment):
            g = f / max(1, frames_per_segment - 1)
            t = ease_fn(np.clip((g - start) / (1.0 - start + 1e-9), 0, 1))
            x, y = bezier_cubic(*paths, t)
            scat.set_offsets(np.column_stack([x, y]))
            if im.get_visible():
                im.set_visible(False)
            if not scat.get_visible():
                scat.set_visible(True)

            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            yield np.asarray(buf)[..., :3].copy()
            frame_counter += 1
            if progress_callback:
                progress_callback("rendering", frame_counter / total_frames)

        # Hold frames
        im.set_data(perm_img)
        for _f in range(hold_per_segment):
            if not im.get_visible():
                im.set_visible(True)
            if scat.get_visible():
                scat.set_visible(False)
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            yield np.asarray(buf)[..., :3].copy()
            frame_counter += 1
            if progress_callback:
                progress_callback("rendering", frame_counter / total_frames)

        current_color_order = current_color_order[M_seg]
        scat.set_facecolors(colors_A[current_color_order])
        scat.set_offsets(np.column_stack([x_all, y_all]))

    # Final hold
    for _f in range(max(0, hold_end)):
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        yield np.asarray(buf)[..., :3].copy()
        frame_counter += 1
        if progress_callback:
            progress_callback("rendering", frame_counter / total_frames)

    plt.close(fig)


def render_chain(
    A_np_original,
    segments,
    H,
    W,
    output_path,
    frames_per_segment=120,
    hold_per_segment=24,
    hold_end=24,
    stagger=0.35,
    arc=0.10,
    seed=None,
    fps=30,
    easing="cosine",
    progress_callback=None,
):
    """Render a multi-segment chain animation to GIF, MP4, or WebM."""
    ext = os.path.splitext(output_path)[1].lower()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    frame_gen = _generate_chain_frames(
        A_np_original,
        segments,
        H,
        W,
        frames_per_segment,
        hold_per_segment,
        hold_end,
        stagger,
        arc,
        seed,
        easing=easing,
        progress_callback=progress_callback,
    )

    if ext == ".gif":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        frame_list = list(frame_gen)
        if not frame_list:
            raise RuntimeError("No frames generated")

        fh, fw = frame_list[0].shape[:2]
        fig2, ax2 = plt.subplots(figsize=(fw / 100, fh / 100), dpi=100)
        ax2.axis("off")
        fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
        im2 = ax2.imshow(frame_list[0], interpolation="nearest")

        def _update(i):
            im2.set_data(frame_list[i])
            return (im2,)

        ani = FuncAnimation(
            fig2, _update, frames=len(frame_list), interval=int(1000 / fps), blit=True
        )
        writer = PillowWriter(fps=fps)
        ani.save(output_path, writer=writer)
        plt.close(fig2)
    elif ext in (".mp4", ".webm"):
        try:
            import imageio.v3 as iio
        except ImportError as err:
            raise ImportError(
                f"{ext} output requires imageio with ffmpeg plugin. "
                "Install with: pip install imageio[ffmpeg]"
            ) from err

        codec = "libx264" if ext == ".mp4" else "libvpx-vp9"
        frame_list = list(frame_gen)
        iio.imwrite(output_path, frame_list, fps=fps, codec=codec, plugin="pyav")
    else:
        raise ValueError(f"Unsupported output format '{ext}'. Supported: .gif, .mp4, .webm")

    return output_path

"""Pixel Duet CLI entry point."""

import argparse
import os

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Rearrange pixels from Image A to form Image B with animated transitions",
    )
    p.add_argument(
        "images",
        nargs="*",
        help="Image paths. 2 for a single duet (A B), 3+ for a chain (A B C ...)",
    )
    p.add_argument("--width", type=int, default=260, help="Working width in pixels")
    p.add_argument("--frames", type=int, default=120, help="Animation frames (per segment)")
    p.add_argument("--hold", type=int, default=24, help="Hold frames (per segment / at end)")
    p.add_argument("--stagger", type=float, default=0.35, help="Stagger in start times (0..1)")
    p.add_argument("--arc", type=float, default=0.10, help="Path curvature")
    p.add_argument("--save", type=str, default="", help="Output path (.gif, .mp4, .webm)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument("--fps", type=int, default=30, help="Output frame rate")
    p.add_argument("--show", action="store_true", help="Show interactive matplotlib window")
    p.add_argument(
        "--batch-dir",
        type=str,
        default="",
        help="Directory of images to process in pairs (batch mode)",
    )
    p.add_argument(
        "--gallery",
        action="store_true",
        help="Generate HTML gallery page (with --batch-dir)",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Launch web app (uvicorn) instead of CLI processing",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for --serve (default 8000)",
    )
    p.add_argument(
        "--easing",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "bounce", "elastic", "step", "expo"],
        help="Easing function for animation (default: cosine)",
    )
    # Advanced multiscale controls
    p.add_argument("--levels", type=str, default="", help="Comma-separated widths coarse->fine")
    p.add_argument("--tiles", type=str, default="", help="Comma-separated tile sizes per level")
    p.add_argument("--lam-spatial", type=str, default="", help="Per-level spatial weights")
    p.add_argument("--lam-parent", type=str, default="", help="Per-level parent-attraction weights")
    args = p.parse_args(argv)

    # Web server mode
    if args.serve:
        import uvicorn

        from pixelduet.web import create_app

        web_app = create_app()
        print(f"Starting Pixel Duet web app on http://localhost:{args.port}")
        uvicorn.run(web_app, host="0.0.0.0", port=args.port)
        return

    # Batch mode
    if args.batch_dir:
        from pixelduet.batch import batch_process, find_image_pairs, generate_gallery_html

        pairs = find_image_pairs(args.batch_dir)
        if not pairs:
            print(f"No image pairs found in {args.batch_dir}")
            return

        output_dir = args.save or "output/gallery"
        ext = "gif"
        # Detect format from output_dir if it looks like a file path
        if "." in os.path.basename(output_dir):
            ext = os.path.splitext(output_dir)[1].lstrip(".")
            output_dir = os.path.dirname(output_dir) or "output/gallery"

        print(f"Found {len(pairs)} image pair(s) in {args.batch_dir}")

        def batch_progress(idx, total, stage, frac):
            print(f"\r  [{idx + 1}/{total}] [{stage}] {int(frac * 100)}%", end="", flush=True)
            if frac >= 1.0:
                print()

        results = batch_process(
            pairs,
            output_dir=output_dir,
            width=args.width,
            frames=args.frames,
            hold=max(0, args.hold),
            stagger=float(np.clip(args.stagger, 0.0, 0.95)),
            arc=max(0.0, args.arc),
            seed=args.seed,
            fps=args.fps,
            format=ext,
            easing=args.easing,
            progress_callback=batch_progress,
        )

        if args.gallery:
            html_path = generate_gallery_html(results, output_dir=output_dir)
            print(f"Gallery: {html_path}")

        print(f"Batch complete: {len(results)} files in {output_dir}")
        return

    if not args.images or len(args.images) < 2:
        p.error("At least 2 images are required (or use --batch-dir)")

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

    save_path = args.save or None
    is_chain = len(args.images) > 2

    if not save_path and not args.show:
        default = "output/chain.gif" if is_chain else "output/swap.gif"
        save_path = default
        print(f"No --save or --show specified, defaulting to: {save_path}")

    # Progress reporting for CLI
    last_stage = [None]

    def progress_callback(stage, frac):
        if stage != last_stage[0]:
            last_stage[0] = stage
            print(f"  [{stage}]", end="", flush=True)
        pct = int(frac * 100)
        print(f"\r  [{stage}] {pct}%", end="", flush=True)
        if frac >= 1.0:
            print()

    stagger_val = float(np.clip(args.stagger, 0.0, 0.95))
    arc_val = max(0.0, args.arc)

    if save_path:
        if is_chain:
            from pixelduet import pixel_duet_chain

            result = pixel_duet_chain(
                args.images,
                output_path=save_path,
                width=args.width,
                frames_per_segment=args.frames,
                hold_per_segment=max(0, args.hold),
                hold_end=max(0, args.hold),
                stagger=stagger_val,
                arc=arc_val,
                seed=args.seed,
                levels=levels,
                tiles=tiles,
                lambdas_spatial=lam_s,
                lambdas_parent=lam_p,
                fps=args.fps,
                easing=args.easing,
                progress_callback=progress_callback,
            )
        else:
            from pixelduet import pixel_duet

            result = pixel_duet(
                args.images[0],
                args.images[1],
                output_path=save_path,
                width=args.width,
                frames=args.frames,
                hold=max(0, args.hold),
                stagger=stagger_val,
                arc=arc_val,
                seed=args.seed,
                levels=levels,
                tiles=tiles,
                lambdas_spatial=lam_s,
                lambdas_parent=lam_p,
                fps=args.fps,
                easing=args.easing,
                progress_callback=progress_callback,
            )
        print(f"Saved to {result}")

    if args.show:
        if is_chain:
            print("--show is not supported for chain mode. Use --save instead.")
            return

        # Interactive matplotlib display (original duet.py behavior)
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from PIL import Image as PILImage

        from pixelduet import map_pixels
        from pixelduet.animation import (
            bezier_cubic,
            compute_paths,
            compute_stagger,
            get_easing,
        )
        from pixelduet.utils import load_rgb, permute_image, resize_to_width, to_np

        A0 = load_rgb(args.images[0])
        B0 = load_rgb(args.images[1])
        A = resize_to_width(A0, args.width)
        Aw, Ah = A.size
        B = B0.resize((Aw, Ah), PILImage.LANCZOS)
        A_np = to_np(A)
        B_np = to_np(B)
        H, W, _ = A_np.shape
        N = H * W

        M_final, _Hm, _Wm = map_pixels(
            A_np,
            B_np,
            levels=levels,
            tiles=tiles,
            lambdas_spatial=lam_s,
            lambdas_parent=lam_p,
            progress_callback=progress_callback,
        )

        colors_A = (A_np.reshape(N, 3) / 255.0).clip(0, 1)
        perm_img = permute_image(A_np, M_final).astype(np.float32) / 255.0

        paths = compute_paths(H, W, M_final, arc=arc_val)
        Sx, Sy = paths[0], paths[1]
        start = compute_stagger(N, stagger=stagger_val, seed=args.seed)

        total_frames = args.frames + max(0, args.hold)
        ease_fn = get_easing(args.easing)

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

        motion_frames = args.frames

        def update(frame):
            if frame < motion_frames:
                g = frame / max(1, motion_frames - 1)
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
            return (scat, im)

        _ani = FuncAnimation(fig, update, frames=total_frames, interval=33, blit=True)
        plt.show()


if __name__ == "__main__":
    main()

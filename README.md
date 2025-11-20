# Pixel Duet

Rearrange only the pixels from Image A (unchanged) so they form the structure of Image B, then animate the move with graceful motion and hold the final result for a clean reveal.

![Demo](output/swap.gif)

## What’s included
- `duet.py` — CLI tool that generates the animation and GIF.
- `app.py` — desktop app (PySide6) with a clean, user‑friendly UI:
  - Two image cards that show thumbnails immediately when selected or drag‑dropped.
  - Submit button with progress feedback.
  - Output viewer that displays the final GIF at its original pixel size (no stretching), scrollable if larger than the window.

## Requirements
- Python 3.9+
- numpy, pillow, matplotlib, scipy, PySide6

Install:
```bash
pip install numpy pillow matplotlib scipy PySide6
```

## Quick start (CLI)
Put your inputs in `images/`, outputs in `output/`, then run:
```bash
python duet.py images/A.png images/B.png --width 260 --frames 120 --hold 24 --save output/swap.gif
```
Your generated GIF will be written to `output/swap.gif`.

## Quick start (App)
Run the desktop app:
```bash
python app.py
```
- Drag & drop or select Image A (source colors) and Image B (target structure).
- Adjust Width, Frames, Hold, Arc, and Stagger if you like.
- Click Submit to export a GIF; progress shows during render.
- The output viewer plays the GIF at its original pixel size with scrollbars if needed.

## Recommended presets (CLI)
Portrait/logo (crisp edges, minimal “box” artifacts):
```bash
python duet.py images/A.png images/B.png \
  --width 260 --frames 120 --hold 24 --save output/swap.gif \
  --levels 64,128,260 --tiles 12,12,10 \
  --lam-spatial 0.35,0.25,0.18 --lam-parent 0.00,0.20,0.12
```

Higher fidelity (finer finish, a bit slower):
```bash
python duet.py images/A.png images/B.png \
  --width 280 --frames 140 --hold 30 --save output/swap.gif \
  --levels 64,128,192,280 --tiles 14,12,10,8 \
  --lam-spatial 0.40,0.28,0.20,0.16 --lam-parent 0.00,0.22,0.16,0.12
```

## CLI flags (most useful)
- `--width` working width in pixels; raise for detail, lower for speed.
- `--frames` animation frames before the final hold.
- `--hold` extra frames to show the final permuted image (great for GIFs).
- `--stagger` 0..1, random staggering in start times (feel of motion).
- `--arc` 0.., path curvature (0 = straight).
- `--save` path to write the GIF, e.g., `output/swap.gif`.
- `--seed` set for reproducible timing.

Advanced multiscale (coarse → fine):
- `--levels` comma‑separated widths; must end with `--width`.
- `--tiles` per‑level tile sizes (smaller at finer levels).
- `--lam-spatial` per‑level weight penalizing long travel.
- `--lam-parent` per‑level weight anchoring fine assignments near the coarser mapping (edge‑aware).

Example (custom multiscale):
```bash
python duet.py images/A.png images/B.png \
  --width 260 --frames 120 --hold 24 --save output/swap.gif \
  --levels 64,128,260 --tiles 12,12,10 \
  --lam-spatial 0.35,0.25,0.18 --lam-parent 0.00,0.22,0.14
```

## How it works (short)
- A bijection maps pixels in A to positions in B with a coarse‑to‑fine, edge‑aware assignment in Lab color space, plus spatial and parent‑anchoring terms. Overlapping windows and small refinement swaps remove visible seams.
- Animation moves each A pixel along a cubic Bezier with cosine easing, then the last frames hold the exact permuted image (no blending).

## Quality tips
- Faint blockiness: add a level (`--levels 64,128,192,260`), use a smaller final tile (8–10), slightly increase final `--lam-parent`.
- Features not snapping: raise the last `--lam-parent` (e.g., 0.12 → 0.16).
- Motion too busy: lower `--stagger` (e.g., 0.25) and `--arc` (e.g., 0.08).
- Very different color palettes: raise `--lam-spatial` to reduce travel and improve structure legibility.

## Performance
- Lower `--width` to speed up; 220–280 is a good range for laptops.
- Fewer levels and larger tiles are faster but may reintroduce seams.
- SciPy’s `linear_sum_assignment` is required for best mapping; without it, the script falls back to a simpler global rank mapping.

## Troubleshooting
- FileNotFoundError: check input paths or use absolute paths.
- Headless environment: include `--save` and open the GIF afterward.
- Slow / memory heavy: reduce `--width`, use fewer levels, or larger tiles at coarse levels.
- App GIF preview: the app uses QMovie and falls back to a PIL player; both display at original pixel size without stretching.

## Project layout (suggested)
- `duet.py` — CLI
- `app.py` — desktop app
- `images/` — inputs
- `output/` — results (e.g., `swap.gif`)
- `README.md`

## License
MIT
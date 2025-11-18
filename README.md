# Pixel Duet

Rearrange only the pixels from Image A (unchanged) so they form the structure of Image B, then animate the move with graceful motion and hold the final result for a clean reveal.

![Demo](output/swap.gif)

## Highlights
- One-way only: uses the exact pixels from A; B is just the layout guide.
- Multiscale, edge-aware assignment for clean structure without visible tile seams.
- Curved, eased motion paths and a final hold so the GIF ends on the finished image.

## Requirements
- Python 3.9+
- numpy, pillow, matplotlib, scipy
- Install: 
```bash
pip install numpy pillow matplotlib scipy
```

## Quick start
- Put your inputs in images/, outputs in output/.
- Run:
```bash
python duet.py images/A.png images/B.png --width 260 --frames 120 --hold 24 --save output/swap.gif
```

## Recommended presets
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

## CLI (most useful flags)
- --width: working width in pixels; raise for detail, lower for speed.
- --frames: animation frames before the final hold.
- --hold: extra frames to show the final permuted image (great for GIFs).
- --stagger: 0..1, how much random staggering in start times (feel of the motion).
- --arc: 0.., how curvy the paths are (0 = straight).
- --save: path to write a GIF/animation, e.g., output/swap.gif.
- --seed: set for reproducible timing.

Advanced multiscale (coarse → fine)
- --levels: comma-separated widths; must end with --width.
- --tiles: per-level tile sizes (smaller at finer levels).
- --lam-spatial: per-level weight reducing long travel.
- --lam-parent: per-level weight that anchors fine assignments near the coarser mapping (edge-aware). Slightly higher on fine levels sharpens edges and reduces seams.

Example (custom multiscale):
```bash
python duet.py images/A.png images/B.png \
  --width 260 --frames 120 --hold 24 --save output/swap.gif \
  --levels 64,128,260 --tiles 12,12,10 \
  --lam-spatial 0.35,0.25,0.18 --lam-parent 0.00,0.22,0.14
```

## How it works (short)
- We compute a bijection from pixels in A to positions in B with a coarse-to-fine, edge-aware assignment in Lab color space, plus spatial and “stay near your coarser destination” penalties. Overlapping windows and small refinement swaps remove the visible box seams.
- Animation moves each A pixel along a cubic Bezier with cosine easing, then the last frames hold the exact permuted image (no blending).

## Quality tips
- Still see faint blockiness? Add a level (e.g., --levels 64,128,192,260) and set a smaller final tile (8–10), slightly increase final --lam-parent.
- Features not snapping? Raise the last lam-parent a bit (e.g., from 0.12 to 0.16).
- Too busy motion? Lower --stagger (e.g., 0.25) and --arc (e.g., 0.08).
- Very different color palettes between A and B: the likeness may be limited by A’s colors; reducing travel (higher lam-spatial) can still improve structure.

## Performance
- Lower --width to speed up. 220–280 is a good range for laptops.
- Fewer levels and larger tiles are faster but may reintroduce seams.
- SciPy’s linear_sum_assignment is required for the best mapping; without it, the script falls back to a simpler global rank mapping.

## Troubleshooting
- FileNotFoundError: check input paths or use absolute paths.
- No window shows: you might be headless; include --save and open the GIF afterward.
- Slow / high memory: reduce --width and levels, or use larger tiles at coarser levels.

## Project layout (suggested)
- duet.py
- images/ (inputs)
- output/ (results, e.g., swap.gif)
- README.md

## Roadmap
- Desktop interface (PySide6) with drag-and-drop, presets, scrubber, and export to MP4/GIF.
- Style presets (portrait, logo, abstract) and one-click quality/speed modes.

## License
- MIT
"""Batch processing: process multiple image pairs and generate a gallery."""

import os
from pathlib import Path


def find_image_pairs(directory, extensions=(".png", ".jpg", ".jpeg")):
    """Find images in a directory and pair them sequentially.

    Returns a list of (path_a, path_b) tuples.
    """
    files = sorted(
        p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    pairs = []
    for i in range(0, len(files) - 1, 2):
        pairs.append((str(files[i]), str(files[i + 1])))
    return pairs


def batch_process(
    pairs,
    output_dir="output/gallery",
    width=260,
    frames=120,
    hold=24,
    stagger=0.35,
    arc=0.10,
    seed=None,
    fps=30,
    format="gif",
    easing="cosine",
    progress_callback=None,
):
    """Process multiple image pairs and save results.

    Parameters
    ----------
    pairs : list of (str, str)
        List of (path_a, path_b) tuples.
    output_dir : str
        Directory to write output files.
    format : str
        Output format: 'gif', 'mp4', or 'webm'.
    progress_callback : callable, optional
        Called with (pair_index: int, total: int, stage: str, fraction: float).

    Returns
    -------
    list of str
        Paths to generated output files.
    """
    from pixelduet import pixel_duet

    os.makedirs(output_dir, exist_ok=True)
    results = []
    total = len(pairs)

    for i, (path_a, path_b) in enumerate(pairs):
        name_a = Path(path_a).stem
        name_b = Path(path_b).stem
        output_path = os.path.join(output_dir, f"{name_a}_to_{name_b}.{format}")

        def on_progress(stage, frac, idx=i):
            if progress_callback:
                progress_callback(idx, total, stage, frac)

        result = pixel_duet(
            path_a,
            path_b,
            output_path=output_path,
            width=width,
            frames=frames,
            hold=hold,
            stagger=stagger,
            arc=arc,
            seed=seed,
            fps=fps,
            easing=easing,
            progress_callback=on_progress,
        )
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, total, "complete", 1.0)

    return results


def generate_gallery_html(results, output_dir="output/gallery", title="Pixel Duet Gallery"):
    """Generate an HTML gallery page for batch results.

    Parameters
    ----------
    results : list of str
        Paths to output files (GIFs, MP4s, etc.).
    output_dir : str
        Directory to write the HTML file.
    title : str
        Page title.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    html_path = os.path.join(output_dir, "index.html")
    items_html = []

    for path in results:
        filename = os.path.basename(path)
        name = Path(path).stem.replace("_to_", " \u2192 ").replace("_", " ")
        ext = Path(path).suffix.lower()

        if ext == ".gif":
            media = f'<img src="{filename}" alt="{name}" loading="lazy">'
        else:
            media = (
                f'<video src="{filename}" autoplay loop muted playsinline title="{name}"></video>'
            )

        items_html.append(f'<div class="card">{media}<div class="label">{name}</div></div>')

    cards = "\n      ".join(items_html)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0F1115; color: #EAEAEA; font-family: system-ui, sans-serif; padding: 2rem; }}
    h1 {{ text-align: center; margin-bottom: 2rem; font-weight: 600; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1.5rem; }}
    .card {{ background: #131720; border: 1px solid #2A3140; border-radius: 12px; overflow: hidden; }}
    .card img, .card video {{ width: 100%; display: block; }}
    .label {{ padding: 0.75rem 1rem; font-size: 0.9rem; color: #9AA4B2; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="grid">
      {cards}
  </div>
</body>
</html>"""

    os.makedirs(output_dir, exist_ok=True)
    with open(html_path, "w") as f:
        f.write(html)

    return html_path

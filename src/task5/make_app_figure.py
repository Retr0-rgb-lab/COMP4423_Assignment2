"""Render what the Task 5 application actually does, for one photograph.

This is not a mock-up of the window. It calls the shipped prediction path,
`src.task5.predict.predict_rgb`, on a real photograph from the dataset, and
draws what that function returned: the photograph, the 256x256 centre crop
the model receives, the sentence stored on the pipeline, and the two class
probabilities. The figure is therefore evidence of the program working, not
an illustration of it.

Run from the repository root:

    python -m src.task5.make_app_figure
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.task3.config import PICTURES                      # noqa: E402
from src.task3.preprocess import load_rgb, resize_and_crop  # noqa: E402
from src.task5.predict import load_pipeline, predict_rgb   # noqa: E402

# A test-set photograph the locked pipeline gets right, so the figure shows
# the program as it is meant to be used.
TARGET = "OUT-06_001_IMG_0863.jpg"
OUT = ROOT / "docs" / "figures" / "task5_application.png"


def font(size, bold=False):
    """Return a matplotlib font property for the first available sans-serif family.

    DejaVu Sans, Liberation Sans and Arial are tried in that order, and the
    default font is used when none of them is installed.
    """
    for name in ("DejaVu Sans", "Liberation Sans", "Arial"):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return font_manager.FontProperties(fname=path, size=size, weight="bold" if bold else "normal")
        except Exception:
            continue
    return font_manager.FontProperties(size=size, weight="bold" if bold else "normal")


def main():
    """Draw the Task 5 figure and save it as docs/figures/task5_application.png.

    The three panels show the photograph, the 256x256 crop that the model
    receives and the prediction returned by the shipped prediction path. The
    photograph is cropped with ``resize_and_crop`` first, and the crop is
    passed to ``predict_rgb``, which leaves an already 256x256 image
    unchanged.
    """
    index = {p.name: p for p in PICTURES.rglob("*") if p.is_file()}
    if TARGET not in index:
        raise SystemExit(f"photograph not found: {TARGET}")
    path = index[TARGET]

    pipe = load_pipeline()
    rgb = load_rgb(str(path))
    crop = resize_and_crop(rgb)
    result = predict_rgb(pipe, crop)

    fig = plt.figure(figsize=(10.0, 4.2), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.25], wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.imshow(rgb)
    ax1.set_title(f"selected photograph\n{path.name}", fontproperties=font(9), color="#1c1917")
    ax1.set_axis_off()

    ax2.imshow(crop)
    ax2.set_title("the 256x256 crop the model receives",
                  fontproperties=font(9), color="#1c1917")
    ax2.set_axis_off()

    ax3.set_axis_off()
    inside_p = getattr(result, "inside_probability", None)
    outside_p = getattr(result, "outside_probability", None)
    if inside_p is None and hasattr(result, "probabilities"):
        outside_p, inside_p = result.probabilities[0], result.probabilities[1]
    ax3.text(0.0, 0.96, result.text, fontproperties=font(13, bold=True),
             color="#166534", va="top", ha="left")
    ax3.text(0.0, 0.76, "outside: {:.3f}\ninside:  {:.3f}".format(outside_p, inside_p),
             fontproperties=font(10), color="#1c1917", va="top", ha="left")
    ax3.text(0.0, 0.46, "models/task3_pipeline.joblib\nresize_and_crop from\nsrc/task3/preprocess.py",
             fontproperties=font(8), color="#57534e", va="top", ha="left")
    ax3.text(0.0, 0.20, "file name, folder, EXIF and GPS are\nnot used in the decision",
             fontproperties=font(8), color="#57534e", va="top", ha="left")

    fig.suptitle(
        "The Task 5 application applied to one photograph, produced by its own prediction path",
        fontproperties=font(11, bold=True), color="#1c1917", y=0.99)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"  label: {result.text}")
    print(f"  outside {outside_p:.3f}  inside {inside_p:.3f}")


if __name__ == "__main__":
    main()

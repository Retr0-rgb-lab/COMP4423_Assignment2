# -*- coding: utf-8 -*-
"""Draw the pipeline and the model-selection protocol as one figure.

The figure has two panels:

  left   what the code does, from the photographs to the application
  right  how the model was chosen, and where the test set enters

Every number in the figure comes from the locked records, so the figure
cannot drift away from the report. Run from the repository root:

    python src/tools/make_pipeline_figure.py

The drawing uses inches as its coordinate unit, so box heights follow from
the number of lines and the point size instead of being guessed. The canvas
is deliberately narrow and the type deliberately large: the figure is
reproduced at the text width of the report, so anything smaller than about
11 pt here would be unreadable on the page.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = _REPO / "docs" / "figures" / "pipeline.png"

INK = "#1c1917"
MUTED = "#57534e"
LINE = "#a8a29e"

C_DATA = "#dbeafe"
C_PREP = "#fef3c7"
C_FEAT = "#dcfce7"
C_MODEL = "#fed7aa"
C_SAVE = "#e9d5ff"
C_PROTO = "#f1f5f9"
C_TEST = "#fee2e2"

FIG_W, FIG_H = 10.4, 9.2
PANEL_PT = 16.5
TITLE_PT = 12.8
BODY_PT = 11.5
LINE_IN = 0.175
TITLE_IN = 0.205
PAD = 0.11
GAP = 0.20


def _font(size, bold=False):
    for name in ("DejaVu Sans", "Liberation Sans", "Arial"):
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=name), fallback_to_default=False)
            return font_manager.FontProperties(
                fname=path, size=size, weight="bold" if bold else "normal")
        except Exception:
            continue
    return font_manager.FontProperties(size=size, weight="bold" if bold else "normal")


def box(ax, x, top, w, title, lines, color):
    """Draw a box whose top edge is at `top`; returns (x, bottom, w, height)."""
    h = PAD + TITLE_IN + len(lines) * LINE_IN + PAD
    y = top - h
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.055",
        linewidth=1.1, edgecolor=LINE, facecolor=color, zorder=2))
    ty = top - PAD
    ax.text(x + 0.13, ty, title, fontproperties=_font(TITLE_PT, True),
            color=INK, va="top", ha="left", zorder=3)
    ty -= TITLE_IN
    for line in lines:
        ax.text(x + 0.13, ty, line, fontproperties=_font(BODY_PT),
                color=MUTED, va="top", ha="left", zorder=3)
        ty -= LINE_IN
    return (x, y, w, h)


def arrow(ax, a, b):
    """Arrow from the bottom centre of box a to the top centre of box b."""
    p0 = (a[0] + a[2] / 2, a[1])
    p1 = (b[0] + b[2] / 2, b[1] + b[3])
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=LINE,
        shrinkA=0, shrinkB=0, zorder=1))


def main():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # ---------------------------------------------------------------- titles
    ax.text(2.90, FIG_H - 0.32, "A.  What the code does",
            fontproperties=_font(PANEL_PT, True), color=INK, ha="center", va="top")
    ax.text(7.95, FIG_H - 0.32, "B.  How the model was chosen",
            fontproperties=_font(PANEL_PT, True), color=INK, ha="center", va="top")
    ax.plot([5.52, 5.52], [0.62, FIG_H - 0.72], color="#e7e5e4", linewidth=1.0, zorder=0)

    # ---------------------------------------------------------------- panel A
    LX, LW = 0.30, 5.00
    top = FIG_H - 0.88
    a1 = box(ax, LX, top, LW, "Photographs",
             ["540 images in the modelling set, 16 groups",
              "one physical place = one group_id"], C_DATA)
    a2 = box(ax, LX, a1[1] - GAP, LW, "Split by group, never by image",
             ["train 403   validation 68   test 69",
              "a place belongs to exactly one split",
              "validation is shown, never used to pick"], C_DATA)
    a3 = box(ax, LX, a2[1] - GAP, LW, "resize_and_crop  (preprocess.py)",
             ["short side to 256, then centre crop",
              "256 x 256, uint8, RGB",
              "horizontal flip on training images only"], C_PREP)
    a4 = box(ax, LX, a3[1] - GAP, LW, "Gray crop",
             ["HOG  (F2)    8100 values",
              "LBP  (F3)      50 values"], C_FEAT)
    a5 = box(ax, LX, a4[1] - GAP, LW, "Standardize each, concatenate",
             ["F4 = HOG + LBP = 8150 values"], C_FEAT)
    a6 = box(ax, LX, a5[1] - GAP, LW, "Random forest  (C4)",
             ["300 trees, max depth 12, seed 42"], C_MODEL)
    a7 = box(ax, LX, a6[1] - GAP, LW, "Refit on all 403 training images",
             ["saved to models/task3_pipeline.joblib"], C_SAVE)
    a8 = box(ax, LX, a7[1] - GAP, LW, "Application  (src/task5/app.py)",
             ["select an image; same crop, same features",
              "answer: Inside / Outside PolyU main campus"], C_SAVE)

    for a, b in ((a1, a2), (a2, a3), (a3, a4), (a4, a5),
                 (a5, a6), (a6, a7), (a7, a8)):
        arrow(ax, a, b)

    # ---------------------------------------------------------------- panel B
    RX, RW = 5.80, 4.30
    top = FIG_H - 0.88
    b1 = box(ax, RX, top, RW, "Stage A - which feature?",
             ["6 features (F1..F6) under one fixed",
              "linear SVM, 5 location-holdout folds",
              "ranked by mean balanced accuracy"], C_PROTO)
    b2 = box(ax, RX, b1[1] - GAP, RW, "Stage B - which classifier?",
             ["5 classifiers (C1..C5) on the two",
              "features that advanced, same folds",
              "C6 applies to histograms only, so it",
              "never reaches the two that advanced"], C_PROTO)
    b3 = box(ax, RX, b2[1] - GAP, RW, "Locked",
             ["F4 + C4, written down before the test",
              "set was touched; no search, one seed"], C_SAVE)
    b4 = box(ax, RX, b3[1] - GAP, RW, "Test set scored once",
             ["accuracy 0.594   balanced acc. 0.660",
              "inside recall 0.357"], C_TEST)
    b5 = box(ax, RX, b4[1] - GAP, RW, "Read it in Section 4",
             ["fold 2 is weak for every classifier,",
              "and the gap between the top rows is small"], C_PROTO)

    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5)):
        arrow(ax, a, b)

    ax.text(RX + RW / 2, b3[1] - 0.09, "the test set enters here and nowhere else",
            fontproperties=_font(10.5), color="#b91c1c", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"), zorder=4)

    # ---------------------------------------------------------------- footer
    ax.text(FIG_W / 2, 0.30, "F1..F6 are the feature sets and C1..C6 the classifiers; "
            "both are defined in Sections 3.3 and 3.4 and listed in the appendix.",
            fontproperties=_font(10.5), color=MUTED, ha="center", va="bottom")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

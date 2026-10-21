# -*- coding: utf-8 -*-
"""Draw the Task 4 confusion matrices and the per-location-group accuracy.

The numbers come from the locked records. The script does not retrain and does
not read the test set to change the model:

- validation / test confusion matrices: progress/task3_experiment.md §8.3 (2026-10-14)
- test set correct counts per location group: plans/task1_3_review_2026-10-15.md §4

This script only turns those integers into figures and checks that they agree
with each other.
If outputs/task3/test_predictions.csv is on disk, it also checks that the
per-image predictions still match those integers; when they agree it writes the
example file name list. The qualitative example grid is only assembled when the
pictures can be found by file name under data/pictures/. When they cannot be
found, no file name and no figure is invented.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
PRED_CSV = ROOT / "outputs" / "task3" / "test_predictions.csv"
PICTURES = ROOT / "data" / "pictures"

# §8.3 validation set. Rows are the true class, columns the predicted class, in the order outside, inside.
VAL_CM = ((26, 4), (17, 21))
VAL_ACC_PUBLISHED = 0.691
VAL_BA_PUBLISHED = 0.710

# §8.3 test set.
TEST_CM = ((26, 1), (27, 15))
TEST_ACC_PUBLISHED = 0.594
TEST_BA_PUBLISHED = 0.660

# Review report §4. correct / n.
GROUP_ROWS = (
    ("OUT-06", "HKU Main Building", "outside", 19, 20, 0.950),
    ("OUT-07", "HKU campus green", "outside", 7, 7, 1.000),
    ("IN-03", "Library and central podium", "inside", 9, 25, 0.360),
    ("IN-07", "CMA courtyard", "inside", 6, 17, 0.353),
)

# Font candidates: take the first TTF that exists, in this order. The locked
# environment is Windows, but this script has to run on Linux / macOS too, so
# do not hard-code a single path.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
)


def _resolve_font(bold):
    """Return the first TTF path that exists; fall back to the font bundled with
    matplotlib when none is found."""
    candidates = _FONT_CANDIDATES
    if bold:
        candidates = tuple(p for p in candidates if "Bold" in p or "bd" in p.lower() or "b." in p.lower()) + tuple(
            p for p in candidates if not ("Bold" in p or "bd" in p.lower() or "b." in p.lower())
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    # Last fallback: the DejaVu font bundled with matplotlib, the same glyph set
    # as the preferred fonts above.
    try:
        import matplotlib.font_manager as fm

        packaged = fm.findfont(fm.FontProperties(family="DejaVu Sans"))
        if os.path.isfile(packaged):
            return packaged
    except Exception:
        pass
    return None


def _font(size, bold=False):
    path = _resolve_font(bold)
    if path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _check():
    vt_out, vt_in = VAL_CM
    assert vt_out[0] + vt_out[1] == 30
    assert vt_in[0] + vt_in[1] == 38
    assert (vt_out[0] + vt_in[1]) / 68 == 47 / 68
    if abs((47 / 68) - VAL_ACC_PUBLISHED) > 0.0005:
        raise SystemExit("validation accuracy does not match §8.3")
    val_ba = ((26 / 30) + (21 / 38)) / 2
    if abs(val_ba - VAL_BA_PUBLISHED) > 0.0005:
        raise SystemExit("validation balanced accuracy does not match §8.3")

    tt_out, tt_in = TEST_CM
    assert tt_out[0] + tt_out[1] == 27
    assert tt_in[0] + tt_in[1] == 42
    correct = tt_out[0] + tt_in[1]
    assert correct == 41
    if abs((41 / 69) - TEST_ACC_PUBLISHED) > 0.0005:
        raise SystemExit("test accuracy does not match §8.3")
    test_ba = ((26 / 27) + (15 / 42)) / 2
    if abs(test_ba - TEST_BA_PUBLISHED) > 0.0005:
        raise SystemExit("test balanced accuracy does not match §8.3")

    group_correct = 0
    group_n = 0
    fn = 0
    fp = 0
    for gid, _name, kind, hit, n, published in GROUP_ROWS:
        assert abs(hit / n - published) < 0.0005, gid
        group_correct += hit
        group_n += n
        if kind == "outside":
            fp += n - hit
        else:
            fn += n - hit
    assert group_n == 69
    assert group_correct == 41
    assert fp == 1 and fn == 27
    assert tt_out[1] == fp and tt_in[0] == fn


def _cell_color(value, scale_max, emphasize):
    t = value / scale_max
    if emphasize:
        # Missed inside: light background to brick red.
        r = int(255 - 40 * t)
        g = int(236 - 150 * t)
        b = int(230 - 150 * t)
        return (r, g, b)
    r = int(255 - 150 * t)
    g = int(250 - 90 * t)
    b = int(245 - 40 * t)
    return (r, g, b)


def draw_confusion(cm, title, subtitle, footer, path):
    scale = 2
    w, h = 920 * scale, 700 * scale
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    title_f = _font(28 * scale, bold=True)
    sub_f = _font(14 * scale)
    label_f = _font(16 * scale, bold=True)
    small_f = _font(13 * scale)
    num_f = _font(36 * scale, bold=True)
    foot_f = _font(13 * scale)

    draw.text((48 * scale, 28 * scale), title, fill="#1c1917", font=title_f)
    draw.text((48 * scale, 72 * scale), subtitle, fill="#57534e", font=sub_f)

    origin_x, origin_y = 210 * scale, 170 * scale
    cell = 220 * scale
    gap = 8 * scale
    headers = ("Predicted outside", "Predicted inside")
    row_names = ("True outside", "True inside")
    # In both the test set and the validation set the largest cell is inside
    # predicted as outside, so that cell gets the emphasis colour.
    emphasize = {(1, 0)}
    scale_max = max(max(row) for row in cm)

    for col, text in enumerate(headers):
        x = origin_x + col * (cell + gap)
        bbox = draw.textbbox((0, 0), text, font=small_f)
        tw = bbox[2] - bbox[0]
        draw.text((x + (cell - tw) / 2, origin_y - 36 * scale), text, fill="#1c1917", font=small_f)

    for r, name in enumerate(row_names):
        bbox = draw.textbbox((0, 0), name, font=label_f)
        th = bbox[3] - bbox[1]
        y = origin_y + r * (cell + gap) + (cell - th) / 2
        draw.text((36 * scale, y), name, fill="#1c1917", font=label_f)
        row_total = sum(cm[r])
        for c, value in enumerate(cm[r]):
            x = origin_x + c * (cell + gap)
            y0 = origin_y + r * (cell + gap)
            color = _cell_color(value, scale_max, (r, c) in emphasize)
            draw.rounded_rectangle(
                (x, y0, x + cell, y0 + cell), radius=12 * scale, fill=color,
            )
            count = str(value)
            bbox = draw.textbbox((0, 0), count, font=num_f)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                (x + (cell - tw) / 2, y0 + cell / 2 - th - 4 * scale),
                count, fill="#1c1917", font=num_f,
            )
            share = f"{value / row_total:.1%} of this row"
            bbox = draw.textbbox((0, 0), share, font=small_f)
            tw = bbox[2] - bbox[0]
            draw.text(
                (x + (cell - tw) / 2, y0 + cell / 2 + 18 * scale),
                share, fill="#44403c", font=small_f,
            )

    draw.text((48 * scale, h - 70 * scale), footer, fill="#44403c", font=foot_f)
    img = img.resize((w // scale, h // scale), Image.Resampling.LANCZOS)
    img.save(path, "PNG")
    print(f"wrote {path}")


def draw_groups(path):
    scale = 2
    w, h = 980 * scale, 560 * scale
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    title_f = _font(26 * scale, bold=True)
    sub_f = _font(14 * scale)
    label_f = _font(15 * scale)
    small_f = _font(13 * scale)

    draw.text((40 * scale, 24 * scale), "Test accuracy by location group", fill="#1c1917", font=title_f)
    draw.text(
        (40 * scale, 66 * scale),
        "Locked HOG+LBP + random forest, one test evaluation (2026-10-14).",
        fill="#57534e", font=sub_f,
    )
    draw.text(
        (40 * scale, 90 * scale),
        "Counts from the 2026-10-15 review, section 4. Overall test accuracy 0.594 is the vertical line.",
        fill="#57534e", font=sub_f,
    )

    left = 430 * scale
    right = 900 * scale
    top = 150 * scale
    row_h = 78 * scale
    axis_y = top + len(GROUP_ROWS) * row_h

    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = left + int((right - left) * frac)
        draw.line((x, top - 10 * scale, x, axis_y), fill="#e7e5e4", width=scale)
        label = f"{frac:.2f}"
        bbox = draw.textbbox((0, 0), label, font=small_f)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw / 2, axis_y + 8 * scale), label, fill="#57534e", font=small_f)

    colors = {"outside": "#1d4e89", "inside": "#9a3412"}
    for i, (gid, name, kind, hit, n, rate) in enumerate(GROUP_ROWS):
        y = top + i * row_h
        label = f"{gid}  {name}"
        sub = f"{kind}   {hit}/{n}"
        draw.text((36 * scale, y + 8 * scale), label, fill="#1c1917", font=label_f)
        draw.text((36 * scale, y + 34 * scale), sub, fill="#57534e", font=small_f)
        bar_w = int((right - left) * rate)
        draw.rounded_rectangle(
            (left, y + 18 * scale, left + bar_w, y + 52 * scale),
            radius=6 * scale, fill=colors[kind],
        )
        draw.text((left + bar_w + 10 * scale, y + 22 * scale), f"{rate:.3f}", fill="#1c1917", font=label_f)

    acc_x = left + int((right - left) * TEST_ACC_PUBLISHED)
    draw.line((acc_x, top - 16 * scale, acc_x, axis_y), fill="#b45309", width=2 * scale)
    draw.text((acc_x + 6 * scale, top - 36 * scale), "0.594", fill="#b45309", font=small_f)

    note = "Outside groups are above 0.95. Both unseen inside groups are near 0.36."
    draw.text((40 * scale, h - 48 * scale), note, fill="#44403c", font=small_f)
    img = img.resize((w // scale, h // scale), Image.Resampling.LANCZOS)
    img.save(path, "PNG")
    print(f"wrote {path}")


def _index_pictures(root: Path):
    found = {}
    if not root.is_dir():
        return found
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                found[name] = str(Path(dirpath) / name)
    return found


def _load_predictions(path: Path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    parsed = []
    for row in rows:
        parsed.append({
            "filename": row["filename"],
            "group_id": row["group_id"],
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
        })
    return parsed


def _cell(row):
    if row["y_true"] == 0 and row["y_pred"] == 0:
        return "true_outside"
    if row["y_true"] == 0 and row["y_pred"] == 1:
        return "false_inside"
    if row["y_true"] == 1 and row["y_pred"] == 0:
        return "false_outside"
    return "true_inside"


def _cell_caption(row):
    """Report-facing wording for one cell of the test confusion matrix."""
    if row["y_true"] == 0 and row["y_pred"] == 0:
        return "correct - outside"
    if row["y_true"] == 0 and row["y_pred"] == 1:
        return "wrong - predicted inside"
    if row["y_true"] == 1 and row["y_pred"] == 0:
        return "wrong - predicted outside"
    return "correct - inside"


def write_example_manifest(rows):
    """File names in lexicographic order, so the same image can be matched up
    later."""
    buckets = {k: [] for k in ("true_outside", "false_inside", "false_outside", "true_inside")}
    per_group = {}
    for row in rows:
        buckets[_cell(row)].append(row)
        hit = int(row["y_true"] == row["y_pred"])
        acc_n, acc_hit = per_group.get(row["group_id"], (0, 0))
        per_group[row["group_id"]] = (acc_n + 1, acc_hit + hit)

    counts = {k: len(v) for k, v in buckets.items()}
    expected = {
        "true_outside": 26,
        "false_inside": 1,
        "false_outside": 27,
        "true_inside": 15,
    }
    if counts != expected:
        raise SystemExit(
            f"the confusion matrix in test_predictions.csv is {counts}, the locked record is {expected}."
            " No example list is written and the locked numbers are not changed."
        )
    for gid, _name, _kind, hit, n, _pub in GROUP_ROWS:
        got_n, got_hit = per_group.get(gid, (0, 0))
        if (got_n, got_hit) != (n, hit):
            raise SystemExit(f"{gid} is {got_hit}/{got_n} in the CSV, the locked record is {hit}/{n}")

    manifest = OUT_DIR / "task4_example_manifest.csv"
    picks = []

    def take(bucket, group_id, k):
        chosen = [r for r in bucket if r["group_id"] == group_id]
        chosen.sort(key=lambda r: r["filename"])
        return chosen[:k]

    picks.extend(take(buckets["true_outside"], "OUT-06", 2))
    picks.extend(take(buckets["true_outside"], "OUT-07", 2))
    picks.extend(buckets["false_inside"])  # only 1 image, so it must be OUT-06
    picks.extend(take(buckets["false_outside"], "IN-03", 2))
    picks.extend(take(buckets["false_outside"], "IN-07", 2))
    picks.extend(take(buckets["true_inside"], "IN-03", 2))
    picks.extend(take(buckets["true_inside"], "IN-07", 2))

    with open(manifest, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["role", "filename", "group_id", "y_true", "y_pred"],
        )
        writer.writeheader()
        for row in picks:
            writer.writerow({
                "role": _cell(row),
                "filename": row["filename"],
                "group_id": row["group_id"],
                "y_true": row["y_true"],
                "y_pred": row["y_pred"],
            })
    print(f"wrote {manifest} ({len(picks)} filenames)")
    return picks


def write_example_grid(picks, pictures):
    missing = [row["filename"] for row in picks if row["filename"] not in pictures]
    if missing:
        print(
            f"example grid not generated: {len(missing)} file names in the list are missing from data/pictures/, "
            f"for example {missing[:3]}"
        )
        return
    thumb = 180
    pad = 12
    cols = 4
    rows_n = (len(picks) + cols - 1) // cols
    header = 36
    label_h = 36
    w = cols * thumb + (cols + 1) * pad
    h = header + rows_n * (thumb + label_h) + (rows_n + 1) * pad
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(12)
    draw.text((pad, 8), "Test examples from test_predictions.csv (filename order within each cell)", fill="#1c1917", font=font)
    for i, row in enumerate(picks):
        r, c = divmod(i, cols)
        x = pad + c * (thumb + pad)
        y = header + pad + r * (thumb + label_h + pad)
        im = Image.open(pictures[row["filename"]]).convert("RGB")
        im.thumbnail((thumb, thumb))
        sheet.paste(im, (x, y))
        caption = f"{row['group_id']}  {_cell_caption(row)}"
        draw.text((x, y + thumb + 2), caption, fill="#1c1917", font=font)
    path = OUT_DIR / "task4_example_grid.png"
    sheet.save(path, "PNG")
    print(f"wrote {path}")


def try_examples():
    if not PRED_CSV.is_file():
        print(
            "no qualitative examples generated: outputs/task3/test_predictions.csv is missing."
            " That file is listed in .gitignore, and this workspace has no photos either."
            " The confusion matrix figures only use the integers from the progress record; no error-case file name was invented."
        )
        return
    rows = _load_predictions(PRED_CSV)
    picks = write_example_manifest(rows)
    pictures = _index_pictures(PICTURES)
    if not pictures:
        print("There is a prediction CSV, but data/pictures/ is empty: only the file name list was written, no grid.")
        return
    write_example_grid(picks, pictures)


def main():
    _check()
    draw_confusion(
        TEST_CM,
        "Test confusion matrix",
        "69 images. True class by row, predicted class by column. Source: task3 experiment section 8.3.",
        "Accuracy 41/69, reported 0.594. Outside recall 26/27, reported 0.963. Inside recall 15/42, reported 0.357.",
        OUT_DIR / "task4_test_confusion_matrix.png",
    )
    draw_confusion(
        VAL_CM,
        "Validation confusion matrix",
        "68 images. Reported after the model was locked. Not used to choose features or the classifier.",
        "Accuracy 47/68, reported 0.691. Outside recall 26/30, reported 0.867. Inside recall 21/38, reported 0.553.",
        OUT_DIR / "task4_val_confusion_matrix.png",
    )
    draw_groups(OUT_DIR / "task4_test_per_group.png")
    try_examples()


if __name__ == "__main__":
    sys.exit(main())

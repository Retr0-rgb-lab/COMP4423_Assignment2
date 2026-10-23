# -*- coding: utf-8 -*-
"""Annotate the 7 inside location groups on the official Campus Map aerial
view.

Base image: docs/figures/campus_map_base.png
(cropped out of a screenshot of the official 3D Campus Map, 1537x1336, without
the pins.) The base image keeps the building letters of the real map.

Only the numbered pins are drawn on the map. The place names and the usable
counts are set as a table in the report, not baked into the picture: the map
is much wider than the printed column, so a sidebar legend would be scaled
down to roughly 3pt and become unreadable, while a real table stays at body
size and can be selected and searched.

The pin coordinates were checked area by area against the building letters on
the map (A-Z, VA, VS, M, P, Q).
The counts are the usable images with status = include, not the training
counts after the per-group cap.
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'docs', 'figures', 'campus_map_base.png')
OUT = os.path.join(ROOT, 'docs', 'figures', 'figure1_inside_locations_annotated.png')

REGULAR = None
BOLD = None
# The first pair that loads wins, so the script runs on Linux and on Windows
# without editing. Arial is a safe fallback: it is present on every Windows
# install and metrically close to DejaVu Sans.
FONT_CANDIDATES = [
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
     '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
    (r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\arialbd.ttf'),
    ('/System/Library/Fonts/Supplemental/Arial.ttf',
     '/System/Library/Fonts/Supplemental/Arial Bold.ttf'),
]


def _pick_font():
    global REGULAR, BOLD
    for regular, bold in FONT_CANDIDATES:
        try:
            ImageFont.truetype(regular, 12)
        except OSError:
            continue
        REGULAR, BOLD = regular, bold
        return
    raise SystemExit("no usable TrueType font was found for the pin numbers")


def f(sz, bold=False):
    if REGULAR is None:
        _pick_font()
    return ImageFont.truetype(BOLD if bold else REGULAR, sz)


# (number, place, x, y, usable count) -- coordinates on the 1537x1336 base image
PINS = [
    ('01', 'Walkway, J block to VA block', 590, 860, 71),
    ('02', 'Coffered walkway, P/Q block', 940, 755, 62),
    ('03', 'University Library and central podium', 705, 880, 25),
    ('04', 'FSCH, VS block and swimming pool', 390, 690, 53),
    ('05', 'Inner lawn and clock tower, main entrance', 630, 970, 38),
    ('06', 'Building courtyard and Block A entrance', 255, 1145, 40),
    ('07', 'CMA courtyard', 575, 1245, 17),
]
TOTAL = sum(p[4] for p in PINS)

map_im = Image.open(SRC).convert('RGB')
W, H = map_im.size
canvas = map_im.copy()
d = ImageDraw.Draw(canvas, 'RGBA')

# The pins are drawn large enough to survive the reduction to column width.
# The widest printed reduction is about 1:3.4, so a 34px number still lands
# near 10px on the page.
R = 34
for num, _name, x, y, _n in PINS:
    d.ellipse([x - R + 4, y - R + 6, x + R + 4, y + R + 6], fill=(0, 0, 0, 70))
    d.ellipse([x - R, y - R, x + R, y + R], fill=(200, 30, 30, 240),
              outline=(255, 255, 255, 255), width=5)
    d.text((x, y), num, font=f(34, True), fill=(255, 255, 255, 255), anchor='mm')

# Trim the white margin the base image carries on the right, then pad all four
# sides by the same amount. An uneven margin would make the picture look
# shifted once LaTeX centres its bounding box.
grey = canvas.convert('L')
px = grey.load()
cols = [x for x in range(W) if any(px[x, y] < 250 for y in range(0, H, 4))]
rows = [y for y in range(H) if any(px[x, y] < 250 for x in range(0, W, 4))]
box = (cols[0], rows[0], cols[-1] + 1, rows[-1] + 1)
canvas = canvas.crop(box)

PAD = 16
padded = Image.new('RGB', (canvas.width + 2 * PAD, canvas.height + 2 * PAD), 'white')
padded.paste(canvas, (PAD, PAD))

padded.save(OUT, optimize=True)
print('saved', OUT, padded.size, 'pins', len(PINS), 'usable', TOTAL)

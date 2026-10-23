# -*- coding: utf-8 -*-
"""Task 1 deliverable Figure 1: annotate the 7 inside location groups on the
official Campus Map aerial view.

Base image: docs/figures/campus_map_base.png
(cropped out of a screenshot of the official 3D Campus Map, 1537x1336, without
the pins.) The base image keeps the building letters of the real map; every
word added by this script is in English, so the legend can be read without
reading Chinese.
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
    raise SystemExit("no usable TrueType font was found for the legend")


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
LEG_W = 620

map_im = Image.open(SRC).convert('RGB')
W, H = map_im.size
canvas = Image.new('RGB', (W + LEG_W, H), '#FFFFFF')
canvas.paste(map_im, (0, 0))
d = ImageDraw.Draw(canvas, 'RGBA')

R = 21
for num, name, x, y, _ in PINS:
    d.ellipse([x - R + 3, y - R + 5, x + R + 3, y + R + 5], fill=(0, 0, 0, 60))
    d.ellipse([x - R, y - R, x + R, y + R], fill=(200, 30, 30, 235),
              outline=(255, 255, 255, 255), width=4)
    d.text((x, y), num, font=f(21), fill=(255, 255, 255, 255), anchor='mm')

lx = W
d.rectangle([lx, 0, W + LEG_W, H], fill=(255, 255, 255))
d.line([(lx, 0), (lx, H)], fill=(225, 225, 225), width=2)

y0 = 120
d.text((lx + 34, y0), 'Figure 1 - inside location groups', font=f(24, True),
       fill=(15, 15, 15))
d.text((lx + 34, y0 + 42),
       f'PolyU main campus (label = 1)  -  {len(PINS)} groups  -  {TOTAL} usable images',
       font=f(15), fill=(95, 95, 95))
d.line([(lx + 34, y0 + 76), (W + LEG_W - 34, y0 + 76)], fill=(215, 215, 215), width=2)

ry = y0 + 100
d.text((lx + 34, ry), 'No.', font=f(15), fill=(130, 130, 130))
d.text((lx + 84, ry), 'Place', font=f(15), fill=(130, 130, 130))
d.text((lx + LEG_W - 36, ry), 'Images', font=f(15), fill=(130, 130, 130), anchor='ra')
ry += 26
d.line([(lx + 34, ry), (W + LEG_W - 34, ry)], fill=(232, 232, 232), width=1)
ry += 16

for num, name, _, _, n in PINS:
    d.ellipse([lx + 34, ry + 3, lx + 34 + 24, ry + 27], fill=(200, 30, 30, 235))
    d.text((lx + 46, ry + 15), num, font=f(14, True), fill=(255, 255, 255), anchor='mm')
    d.text((lx + 70, ry + 15), f'IN-{num}', font=f(14), fill=(150, 150, 150))
    d.text((lx + 128, ry + 15), name, font=f(15), fill=(25, 25, 25), anchor='lm')
    d.text((lx + LEG_W - 36, ry + 15), str(n), font=f(15),
           fill=(25, 25, 25), anchor='rm')
    ry += 42

ry += 8
d.line([(lx + 34, ry), (W + LEG_W - 34, ry)], fill=(232, 232, 232), width=1)
ry += 24
d.text((lx + 34, ry),
       f'Total  {len(PINS)} groups  -  {TOTAL} usable images',
       font=f(17, True), fill=(200, 30, 30))
ry += 28
d.text((lx + 34, ry), '(status = include; 50-image cap per group)',
       font=f(13), fill=(150, 150, 150))
ry += 50

for line in [
    'One physical place = one location group.',
    'Viewpoint, distance, lighting and background vary',
    'inside a group and do not create new groups.',
    'Frames are scene-level and keep their surroundings;',
    'no isolated objects are photographed.',
]:
    d.text((lx + 34, ry), line, font=f(14), fill=(90, 90, 90))
    ry += 24

ry += 18
box_h = 122
d.rectangle([lx + 34, ry, W + LEG_W - 34, ry + box_h], fill=(253, 246, 238))
d.text((lx + 46, ry + 12), 'Excluded by the collection rules', font=f(14, True),
       fill=(170, 70, 20))
excluded = ['Gate thresholds  -  connecting bridges and corridors',
            'Boundary locations  -  views across the boundary',
            'Frames with identifiable people or personal information']
for i, line in enumerate(excluded):
    d.text((lx + 46, ry + 40 + i * 24), line, font=f(13), fill=(150, 95, 55))

canvas.save(OUT, optimize=True)
print('saved', OUT, canvas.size)

# -*- coding: utf-8 -*-
"""Draw the location overview figure: all 16 photo-collection groups on a real
Google Maps base of Hong Kong, plus a zoomed inset of the PolyU campus.

The figure has two panels:

  * main panel  - a Google Maps capture of the whole city area with the nine
                   outside groups pinned on the spot where the photographs were
                   taken, and a single marker for the PolyU campus (the seven
                   inside groups are too close together to be shown at this
                   scale);
  * inset panel - a second Google Maps capture, zoomed in on the PolyU campus,
                   with the seven inside groups pinned individually;
  * legend      - class is encoded by colour (inside / outside) and split is
                   encoded by the marker shape (train / val / test).

Base images (re-captured with headless Chrome, see the two CAPTURE_* constants
below for the exact URLs):

    docs/figures/location_map_base.png   city map,   centre 22.29480,114.15770, z15
    docs/figures/campus_zoom_base.png    PolyU campus, centre 22.303753,114.179429, z18

The 16 coordinates come from OpenStreetMap: the Nominatim search API for the
place names and the Overpass API for the building level features (the library,
the clock tower, the swimming pool, the CMA Building, the PolyU building
cores).  Every row of LOCATIONS records the query it came from.

The projection is plain Web Mercator (256 px tiles).  It was calibrated against
the scale bar that Google Maps draws in the bottom right corner of every
capture: in the z15 capture the bar measures 113 px for 500 m, and the
Mercator formula predicts 113.1 px for the same zoom and latitude, and in the
z18 capture the bar measures 93 px for 100 m against a predicted 90.5 px.
The device pixel ratio is 1 (the PNG size equals the requested window size),
so the centre of the map is the centre of the image.  The projection was then
checked feature by feature: the OSM polygon of the HKU Main Building falls
exactly on the building of the same name in the Google Maps rendering, and the
pins for Kowloon Park, Elements, the M+ museum, Jordan Road and the PolyU
swimming pool fall on the matching features.

Run from the repository root:

    python src/_gen_location_map.py
"""
import csv
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, 'docs', 'figures')

BASE_MAIN = os.path.join(FIGS, 'location_map_base.png')
BASE_CAMPUS = os.path.join(FIGS, 'campus_zoom_base.png')
OUT = os.path.join(FIGS, 'location_overview.png')

# Exact capture URLs, kept here so the base images can be re-created:
#   chrome.exe --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
#       --window-size=1600,1200 --screenshot=<out> --virtual-time-budget=25000 \
#       "https://www.google.com/maps/@22.29480,114.15770,15z"
#   chrome.exe ... "https://www.google.com/maps/@22.303753,114.179429,18z"
CAPTURE_MAIN_URL = 'https://www.google.com/maps/@22.29480,114.15770,15z'
CAPTURE_CAMPUS_URL = 'https://www.google.com/maps/@22.303753,114.179429,18z'

MAIN_CENTRE = (22.29480, 114.15770)
MAIN_ZOOM = 15
CAMPUS_CENTRE = (22.303753, 114.179429)
CAMPUS_ZOOM = 18

# Crop windows taken from the 1600x1200 captures.  The city capture is cropped
# at the top to drop the Google Maps search box / chip row, the campus capture
# is cropped to the campus itself.
MAIN_CROP = (60, 165, 1600, 1200)
CAMPUS_CROP = (450, 250, 1150, 950)

# Class colours.  Inside is the red used by the existing campus map figure,
# outside is a blue that is clearly distinguishable from it.
COL_INSIDE = (200, 30, 30)
COL_OUTSIDE = (30, 90, 210)

# Marker geometry.
R_MAIN = 26
R_CAMPUS = 17
FONT_LABEL = 34
FONT_LABEL_CAMPUS = 30
FONT_LEGEND = 28
FONT_LEGEND_TITLE = 32
FONT_PANEL_TITLE = 30

REGULAR = None
BOLD = None
# The first pair that loads wins, so the script runs on Linux and on Windows
# without editing.  Arial is present on every Windows install.
FONT_CANDIDATES = [
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
     '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
    (r'C:\Windows\Fonts\arial.ttf', r'C:\Windows\Fonts\arialbd.ttf'),
    ('/System/Library/Fonts/Supplemental/Arial.ttf',
     '/System/Library/Fonts/Supplemental/Arial Bold.ttf'),
]

# --------------------------------------------------------------------------
# The 16 collection locations.
#
# Every row is (group_id, label, lat, lon, class, source, source_name).
# `source` says how the coordinate was obtained and `source_name` is the name
# the data source returned for it, so the table can be audited.  Rows whose
# coordinate is a manual offset carry the note "offset" in `source`.
#
# The split is NOT written here.  It is read from `data/annotations/splits.csv`
# when the figure is drawn, so the map cannot show a split that the dataset
# does not have.
# --------------------------------------------------------------------------
LOCATIONS = [
    ('IN-01', 'J block to VA block walkway', 22.303671, 114.179212, 'inside',
     'OSM building=concourse "J棟 Core J"',
     'J棟 Core J (PolyU J block atrium)'),
    ('IN-02', 'P/Q coffered walkway', 22.304192, 114.180581, 'inside',
     'offset: midpoint of OSM "P柱 Core P" (22.303955,114.180512) and '
     '"Q棟 Core Q" (22.304429,114.180650)',
     'J棟 Core J / P柱 Core P / Q棟 Core Q'),
    ('IN-03', 'University Library', 22.303408, 114.179973, 'inside',
     'Nominatim "香港理工大學圖書館"',
     '包玉剛圖書館 Pao Yue-kong Library, 康莊道 Hong Chong Road, 紅磡灣 Hung Hom Bay'),
    ('IN-04', 'FSCH / VS block / pool', 22.304839, 114.178759, 'inside',
     'OSM swimming_pool "祈廉桐游泳池"',
     '祈廉桐游泳池 Michael Clinton Swimming Pool'),
    ('IN-05', 'Campus gate and clock tower', 22.303800, 114.179392, 'inside',
     'OSM clock "大笨鐘"',
     '大笨鐘 Big Ben'),
    ('IN-06', 'Block A courtyard', 22.302985, 114.178277, 'inside',
     'OSM building=concourse "A棟 Core A"',
     'A棟 Core A (PolyU Block A, AG Wing / Chung Sze Yuen Building)'),
    ('IN-07', 'CMA Building courtyard', 22.302666, 114.179196, 'inside',
     'OSM "香港中華廠商聯合會樓"',
     '香港中華廠商聯合會樓 CMA Building'),
    ('OUT-01', 'Jordan Road', 22.3057478, 114.1682043, 'outside',
     'Nominatim "Jordan Road, Yau Ma Tei, Hong Kong"',
     '佐敦道 Jordan Road, 官涌 Kwun Chung, 油麻地 Yau Ma Tei, 油尖旺區 Yau Tsim Mong District, 九龍 Kowloon'),
    ('OUT-02', 'HKU modern teaching blocks', 22.283572, 114.139962, 'outside',
     'OSM building "鄧志昂樓"',
     '鄧志昂樓 Tang Chi Ngong Building'),
    ('OUT-03', 'Kowloon Park', 22.3005333, 114.1701152, 'outside',
     'Nominatim "Kowloon Park, Yau Tsim Mong, Hong Kong"',
     '九龍公園 Kowloon Park, 22, 尖沙咀 Tsim Sha Tsui, 油尖旺區 Yau Tsim Mong District, 九龍 Kowloon'),
    ('OUT-04', 'Elements', 22.3049245, 114.1619898, 'outside',
     'Nominatim "Elements, West Kowloon, Hong Kong"',
     '圓方 Elements, 1, 車站環迴路東 Station Perimeter Road East, 凱旋門 The Arch, 油麻地 Yau Ma Tei'),
    ('OUT-05', 'M+ museum', 22.3010537, 114.1591472, 'outside',
     'Nominatim "M+ museum, West Kowloon, Hong Kong"',
     'M+, 38, 文化道 Cultural Drive, 西九龍 West Kowloon, 油麻地 Yau Ma Tei, 油尖旺區 Yau Tsim Mong District'),
    ('OUT-06', 'HKU Main Building', 22.283406, 114.134421, 'outside',
     'OSM "Central Podium" (Main Building podium)',
     'Central Podium, 香港大學 The University of Hong Kong'),
    ('OUT-07', 'HKU campus green', 22.283212, 114.138562, 'outside',
     'OSM leisure=park "Knowles Park"',
     'Knowles Park (HKU)'),
    ('OUT-08', 'West Kowloon Art Park', 22.301200, 114.156739, 'outside',
     'offset: OSM leisure=park "藝術公園 Art Park" polygon centroid '
     '(22.300407,114.156739) moved 88 m north onto the open lawn',
     '藝術公園 Art Park'),
    ('OUT-09', 'West Kowloon harbourfront', 22.3020688, 114.1672075, 'outside',
     'Nominatim "Xiqu Centre, West Kowloon, Hong Kong"',
     '戲曲中心 Xiqu Centre, 88, 柯士甸道西 Austin Road West, 官涌 Kwun Chung, 油麻地 Yau Ma Tei'),
]


SPLITS = {}


def load_splits():
    """Return {group_id: split} read from the split table.

    The map shows which split each place belongs to, so it must read that
    from the dataset rather than keep a second copy of it here."""
    path = os.path.join(ROOT, 'data', 'annotations', 'splits.csv')
    out = {}
    with open(path, encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            out[row['group_id']] = row['split']
    return out


def _pick_font():
    """Load the first available TrueType font pair and remember it."""
    global REGULAR, BOLD
    for regular, bold in FONT_CANDIDATES:
        try:
            ImageFont.truetype(regular, 12)
        except OSError:
            continue
        REGULAR, BOLD = regular, bold
        return
    raise SystemExit('no usable TrueType font was found for the figure labels')


def f(size, bold=False):
    """Return the chosen font at the given pixel size."""
    if REGULAR is None:
        _pick_font()
    return ImageFont.truetype(BOLD if bold else REGULAR, size)


def world_px(lat, lon, zoom, tile=256):
    """Web Mercator pixel coordinates of a point on the whole world."""
    scale = tile * (2.0 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * scale
    return x, y


def project(lat, lon, centre, zoom, width, height):
    """Pixel position inside a capture taken at `centre` and `zoom`."""
    cx, cy = world_px(centre[0], centre[1], zoom)
    px, py = world_px(lat, lon, zoom)
    return width / 2.0 + (px - cx), height / 2.0 + (py - cy)


def class_colour(cls):
    """Return the marker colour of a class."""
    return COL_INSIDE if cls == 'inside' else COL_OUTSIDE


def draw_marker(draw, x, y, colour, split, radius):
    """Draw one location marker.

    The shape carries the split: filled circle = train, hollow ring = val,
    filled square = test.  The colour carries the class.
    """
    shadow = (0, 0, 0, 70)
    white = (255, 255, 255, 255)
    if split == 'test':
        half = radius
        draw.rectangle([x - half + 4, y - half + 6, x + half + 4, y + half + 6], fill=shadow)
        draw.rectangle([x - half, y - half, x + half, y + half], fill=colour + (240,),
                       outline=white, width=4)
        return
    draw.ellipse([x - radius + 4, y - radius + 6, x + radius + 4, y + radius + 6],
                 fill=shadow)
    if split == 'val':
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=white,
                     outline=colour + (255,), width=7)
    else:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=colour + (240,),
                     outline=white, width=5)


def text_box(draw, xy, text, font, anchor):
    """Bounding box of a text drawn at `xy` with `anchor`."""
    left, top, right, bottom = draw.textbbox(xy, text, font=font, anchor=anchor)
    return (left, top, right, bottom)


def overlaps(box, others, pad=4):
    """True when `box` touches any rectangle in `others`."""
    for other in others:
        if (box[0] < other[2] + pad and box[2] > other[0] - pad
                and box[1] < other[3] + pad and box[3] > other[1] - pad):
            return True
    return False


def place_label(draw, x, y, text, font, occupied, radius, bounds):
    """Put a label next to a marker, avoiding the markers and other labels.

    Returns (anchor, position, box).  The first candidate position that stays
    inside `bounds` and does not touch anything already drawn is used; when
    every candidate collides the label is pushed to the right of the marker
    and a leader line is drawn.
    """
    gap = radius + 10
    candidates = [
        ('lm', gap, 0), ('rm', -gap, 0),
        ('ma', 0, -gap), ('md', 0, gap),
        ('lm', gap, -gap), ('rm', -gap, -gap),
        ('lm', gap, gap), ('rm', -gap, gap),
        ('ma', 0, -gap - 34), ('md', 0, gap + 34),
        ('lm', gap, -gap - 44), ('rm', -gap, -gap - 44),
        ('lm', gap, gap + 44), ('rm', -gap, gap + 44),
        ('ma', 0, -gap - 78), ('md', 0, gap + 78),
        ('lm', gap, -gap - 90), ('rm', -gap, -gap - 90),
        ('lm', gap, gap + 90), ('rm', -gap, gap + 90),
    ]
    # Fallback tiers, best first: inside the panel and free of other marks,
    # inside the panel, free of other marks, anything at all.
    tier_free = None
    tier_inside = None
    tier_any = None
    for anchor, dx, dy in candidates:
        pos = (x + dx, y + dy)
        box = text_box(draw, pos, text, font, anchor)
        if tier_any is None:
            tier_any = (anchor, pos, box)
        inside = (box[0] >= bounds[0] and box[1] >= bounds[1]
                  and box[2] <= bounds[2] and box[3] <= bounds[3])
        free = not overlaps(box, occupied)
        if inside and free:
            return anchor, pos, box
        if free and tier_free is None:
            tier_free = (anchor, pos, box)
        if inside and tier_inside is None:
            tier_inside = (anchor, pos, box)
    for tier in (tier_inside, tier_free, tier_any):
        if tier is not None:
            return tier
    return tier_any


def resolve_overlaps(points, min_dist):
    """Nudge markers apart so that no marker hides another one.

    Returns a list of (drawn_x, drawn_y, true_x, true_y).  A marker that had
    to be moved gets a leader line back to its true position.
    """
    result = []
    placed = []
    for x, y in points:
        dx = dy = 0.0
        for ox, oy in placed:
            vx, vy = x + dx - ox, y + dy - oy
            dist = math.hypot(vx, vy)
            if dist < min_dist:
                if dist < 1e-6:
                    vx, vy, dist = 1.0, 0.0, 1.0
                push = min_dist - dist
                dx += vx / dist * push
                dy += vy / dist * push
        placed.append((x + dx, y + dy))
        result.append((x + dx, y + dy, x, y))
    return result


def leader(draw, x, y, box, colour):
    """Draw a short line from a marker to a label that had to be moved away."""
    cx = min(max(x, box[0]), box[2])
    cy = min(max(y, box[1]), box[3])
    draw.line([x, y, cx, cy], fill=colour + (170,), width=3)


def needs_leader(anchor, pos, marker):
    """True when a label sits off the marker's own axis and needs a leader."""
    if anchor in ('lm', 'rm'):
        return abs(pos[1] - marker[1]) > 12
    return abs(pos[0] - marker[0]) > 12


def wrap_text(draw, text, font, width):
    """Split a sentence into lines that fit into `width` pixels."""
    lines = []
    current = ''
    for word in text.split():
        trial = (current + ' ' + word).strip()
        if not current or draw.textlength(trial, font=font) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_legend(draw, box):
    """Draw the legend panel: class colours on top, split shapes below."""
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=(255, 255, 255, 244),
                           outline=(60, 60, 60, 220), width=3)
    title_font = f(FONT_LEGEND_TITLE, True)
    item_font = f(FONT_LEGEND)
    x = left + 26
    y = top + 28
    draw.text((x, y), 'Legend', font=title_font, fill=(20, 20, 20, 255))
    y += 58

    rows = (
        ('Class', ((COL_INSIDE, 'train', 'Inside'),
                   (COL_OUTSIDE, 'train', 'Outside'))),
        ('Split', ((COL_OUTSIDE, 'train', 'Train'),
                   (COL_OUTSIDE, 'val', 'Val'),
                   (COL_OUTSIDE, 'test', 'Test'))),
    )
    for title, items in rows:
        draw.text((x, y), title, font=item_font, fill=(70, 70, 70, 255))
        cx = x + 110
        for colour, split, text in items:
            need = 30 + int(draw.textlength(text, font=item_font)) + 40
            if cx + need > right - 26:
                cx = x + 110
                y += 54
            draw_marker(draw, cx, y, colour, split, 17)
            draw.text((cx + 30, y), text, font=item_font, fill=(20, 20, 20, 255),
                      anchor='lm')
            cx += need
        y += 58

    note = ('Inside = the seven PolyU campus groups; outside = the nine '
            'groups in Hong Kong.')
    for line in wrap_text(draw, note, item_font, right - left - 52):
        draw.text((x, y), line, font=item_font, fill=(70, 70, 70, 255))
        y += 36


def draw_inset_frame(canvas, draw, box, title):
    """Paste the campus capture into the figure and title it."""
    left, top, right, bottom = box
    draw.rectangle([left - 3, top - 3, right + 3, bottom + 3], outline=(40, 40, 40, 255),
                   width=3)
    draw.rectangle([left, top, right, top + 44], fill=(0, 0, 0, 150))
    draw.text((left + 16, top + 22), title, font=f(FONT_PANEL_TITLE, True),
              fill=(255, 255, 255, 255), anchor='lm')


def main():
    """Draw the location overview figure and print the coordinate audit table.

    The split of every group is read from the split table, so the figure
    cannot disagree with the dataset. Exits with a message when a base map
    image is missing or when a group in the table has no split."""
    global SPLITS
    SPLITS = load_splits()
    missing = sorted({row[0] for row in LOCATIONS} - set(SPLITS))
    if missing:
        raise SystemExit('no split for these groups: ' + ', '.join(missing))
    # The audit table below contains Chinese place names; make sure they can be
    # printed on a Windows console as well as in a UTF-8 terminal.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

    for path in (BASE_MAIN, BASE_CAMPUS):
        if not os.path.exists(path):
            raise SystemExit(
                'base map image is missing: %s\n'
                're-capture it with headless Chrome, for example:\n'
                '  chrome.exe --headless=new --disable-gpu --no-sandbox '
                '--hide-scrollbars \\\n'
                '      --window-size=1600,1200 --screenshot=<out> '
                '--virtual-time-budget=25000 \\\n'
                '      "%s"' % (path, CAPTURE_MAIN_URL))

    main_img = Image.open(BASE_MAIN).convert('RGB')
    campus_img = Image.open(BASE_CAMPUS).convert('RGB')
    main_panel = main_img.crop(MAIN_CROP)
    campus_panel = campus_img.crop(CAMPUS_CROP)

    mw, mh = main_panel.size
    cw, ch = campus_panel.size
    gap = 24
    canvas_w = mw + gap + cw
    canvas_h = max(mh, ch + 40 + 300)
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    canvas.paste(main_panel, (0, 0))
    draw = ImageDraw.Draw(canvas, 'RGBA')

    # ---- campus inset -----------------------------------------------------
    inset_x = mw + gap
    inset_y = 0
    canvas.paste(campus_panel, (inset_x, inset_y))
    draw_inset_frame(canvas, draw,
                     (inset_x, inset_y, inset_x + cw, inset_y + ch),
                     'PolyU main campus (inside groups)')

    # ---- markers on the campus inset --------------------------------------
    inside = [row for row in LOCATIONS if row[4] == 'inside']
    points = []
    for _group_id, _label, lat, lon, _cls, _src, _name in inside:
        px, py = project(lat, lon, CAMPUS_CENTRE, CAMPUS_ZOOM, 1600, 1200)
        points.append((px - CAMPUS_CROP[0] + inset_x, py - CAMPUS_CROP[1] + inset_y))
    placed = resolve_overlaps(points, 2 * R_CAMPUS + 4)

    occupied = []
    for (px, py, tx, ty), row in zip(placed, inside):
        group_id, _label, _lat, _lon, cls, _src, _name = row
        split = SPLITS[group_id]
        colour = class_colour(cls)
        if abs(px - tx) > 3 or abs(py - ty) > 3:
            draw.line([tx, ty, px, py], fill=colour + (200,), width=3)
        draw_marker(draw, px, py, colour, split, R_CAMPUS)
        occupied.append((px - R_CAMPUS, py - R_CAMPUS, px + R_CAMPUS, py + R_CAMPUS))

    bounds = (inset_x + 4, inset_y + 4, inset_x + cw - 4, inset_y + ch - 4)
    for (px, py, tx, ty), row in zip(placed, inside):
        group_id = row[0]
        colour = class_colour(row[4])
        font = f(FONT_LABEL_CAMPUS, True)
        anchor, pos, box = place_label(draw, px, py, group_id, font, occupied,
                                       R_CAMPUS, bounds)
        if needs_leader(anchor, pos, (px, py)):
            leader(draw, px, py, box, colour)
        draw.text(pos, group_id, font=font, fill=(20, 20, 20, 255), anchor=anchor,
                  stroke_width=4, stroke_fill=(255, 255, 255, 255))
        occupied.append(box)

    # ---- markers on the city map ------------------------------------------
    outside = [row for row in LOCATIONS if row[4] == 'outside']
    points = []
    for _group_id, _label, lat, lon, _cls, _src, _name in outside:
        px, py = project(lat, lon, MAIN_CENTRE, MAIN_ZOOM, 1600, 1200)
        points.append((px - MAIN_CROP[0], py - MAIN_CROP[1]))
    placed = resolve_overlaps(points, 2 * R_MAIN + 4)

    occupied = []
    for (px, py, tx, ty), row in zip(placed, outside):
        colour = class_colour(row[4])
        if abs(px - tx) > 3 or abs(py - ty) > 3:
            draw.line([tx, ty, px, py], fill=colour + (200,), width=3)
        draw_marker(draw, px, py, colour, SPLITS[row[0]], R_MAIN)
        occupied.append((px - R_MAIN, py - R_MAIN, px + R_MAIN, py + R_MAIN))

    bounds = (4, 4, mw - 4, mh - 4)
    for (px, py, tx, ty), row in zip(placed, outside):
        group_id = row[0]
        colour = class_colour(row[4])
        font = f(FONT_LABEL, True)
        anchor, pos, box = place_label(draw, px, py, group_id, font, occupied,
                                       R_MAIN, bounds)
        if needs_leader(anchor, pos, (px, py)):
            leader(draw, px, py, box, colour)
        draw.text(pos, group_id, font=font, fill=(20, 20, 20, 255), anchor=anchor,
                  stroke_width=4, stroke_fill=(255, 255, 255, 255))
        occupied.append(box)

    # The seven inside groups share one campus: a single marker points at it.
    campus_lat = sum(r[2] for r in LOCATIONS if r[4] == 'inside') / 7.0
    campus_lon = sum(r[3] for r in LOCATIONS if r[4] == 'inside') / 7.0
    px, py = project(campus_lat, campus_lon, MAIN_CENTRE, MAIN_ZOOM, 1600, 1200)
    px -= MAIN_CROP[0]
    py -= MAIN_CROP[1]
    draw_marker(draw, px, py, COL_INSIDE, 'train', R_MAIN + 6)
    campus_font = f(FONT_LABEL - 6, True)
    text = 'PolyU main campus\nIN-01 ... IN-07 (see inset)'
    anchor, pos, box = place_label(draw, px, py, text, campus_font, occupied,
                                   R_MAIN + 6, bounds)
    if needs_leader(anchor, pos, (px, py)):
        leader(draw, px, py, box, COL_INSIDE)
    draw.text(pos, text, font=campus_font, fill=(20, 20, 20, 255), anchor=anchor,
              stroke_width=4, stroke_fill=(255, 255, 255, 255), align='left',
              spacing=8)
    occupied.append(box)

    # ---- legend ------------------------------------------------------------
    legend_box = (inset_x, inset_y + ch + 40, inset_x + cw, inset_y + ch + 40 + 300)
    draw_legend(draw, legend_box)

    canvas.save(OUT, optimize=True)
    print('saved', OUT, canvas.size, 'groups', len(LOCATIONS))
    print()
    print('%-7s %-9s %-6s %-11s %-11s %s' % ('group', 'class', 'split', 'lat', 'lon', 'source'))
    for group_id, _label, lat, lon, cls, src, _name in LOCATIONS:
        print('%-7s %-9s %-6s %-11.6f %-11.6f %s'
              % (group_id, cls, SPLITS[group_id], lat, lon, src))


if __name__ == '__main__':
    main()

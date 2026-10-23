# -*- coding: utf-8 -*-
"""Collapse the location groups from "fine-grained groups" to "venue level",
and add the sub-area and scene class columns.

Background
----------
The assignment requires train/val/test to be split by **location group**. The
finer the groups are cut, the larger the effective sample size of the test
set, but fine groups also make it easier to split one and the same physical
place across different splits, which is leakage.

This script takes the conservative reading: **one physical place = one
group_id (split unit)**, and the different scenes inside a place are recorded
in the `sub_area` + `scene_class` columns instead.
Then the split unit is always one definite physical place, so the same scene
can never straddle splits; the diversity inside a place is still fully kept
and can be analysed as a "venue x scene class" cross-tabulation in the report.

Idempotent: if the manifest already has a `sub_area` column the script exits
straight away, so it can be run again and again.
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOTATIONS = os.path.join(ROOT, 'data', 'annotations')

# Fine-grained group -> (venue-level group_id, venue name, sub-area inside the venue)
# "inside" is already written at venue level under data/annotations/inside/; the
# location_name follows those manifests, and is not overwritten here with a
# single sub_area.
VENUE_MAP = {
    'OUT-01': ('OUT-01', 'Jordan',             'Jordan Road street section'),
    'OUT-02': ('OUT-01', 'Jordan',             'Temple Street night market'),
    'OUT-03': ('OUT-02', 'HKU',                'Main Building, traditional red-brick area'),
    'OUT-04': ('OUT-02', 'HKU',                'Campus tree-lined green'),
    'OUT-13': ('OUT-02', 'HKU',                'Modern teaching blocks and links'),
    'OUT-05': ('OUT-03', 'Kowloon Park',       'Chinese garden'),
    'OUT-06': ('OUT-03', 'Kowloon Park',       'Lake and green space'),
    'OUT-07': ('OUT-03', 'Kowloon Park',       'Modern buildings and entrance plaza'),
    'OUT-11': ('OUT-04', 'Elements',           'Elements and the surrounding buildings'),
    'OUT-12': ('OUT-04', 'Elements',           'Curved canopy and open plaza'),
}

# Controlled scene vocabulary; the order is the matching priority.
SCENE_RULES = [
    ('Indoor space',            ('indoor',)),
    ('Urban distance view',     ('distance view', 'skyline', 'from above', 'looking up')),
    ('Water',                   ('water', 'lake', 'bridge', 'pool')),
    ('Market stall',            ('temple', 'lantern', 'night market', 'stall', 'market', 'neon')),
    ('Road frontage',           ('street', 'road', 'traffic', 'bus', 'flyover')),
    ('Tree-lined path',         ('tree-lined', 'path', 'garden path')),
    ('Covered walkway',         ('linkway', 'colonnade', 'canopy', 'ceiling', 'corridor',
                                 'atrium', 'covered')),
    ('Ramp or steps',           ('steps', 'ramp', 'stair', 'staircase')),
    ('Building entrance',       ('entrance', 'porch', 'foyer', 'structure')),
    ('Construction hoarding',   ('construction', 'hoarding', 'site', 'crane', 'scaffolding')),
    ('Building close view',     ('close view', 'wall', 'material')),
    ('Building facade',         ('facade', 'tower', 'curtain wall', 'building', 'brick',
                                 'tiled roof', 'clock', 'balcony', 'block', 'crest')),
    ('Green space',             ('lawn', 'grass', 'green', 'hedge', 'woodland', 'planting',
                                 'shrub', 'palm', 'flower', 'tree', 'garden', 'terrace',
                                 'outdoor')),
    ('Paved plaza',             ('plaza', 'paved', 'hardstanding', 'platform', 'courtyard')),
]

# The few shots the keywords cannot reach are assigned directly
OVERRIDE = {
    'IMG_0812.jpg': 'Market stall', 'IMG_0813.jpg': 'Market stall', 'IMG_0814.jpg': 'Market stall',
    'IMG_0928.jpg': 'Building entrance', 'IMG_0929.jpg': 'Building entrance',
    'IMG_0773.jpg': 'Tree-lined path', 'IMG_0791.jpg': 'Building entrance',
    'IMG_0978.jpg': 'Road frontage', 'IMG_0979.jpg': 'Road frontage',
}

HEADER = ['filename', 'class', 'group_id', 'location_name',
          'sub_area', 'scene_class', 'scene_type', 'status', 'notes']

# Traditional -> simplified. Most campus signage is traditional while the
# vocabulary is written in simplified, so normalise first and then match,
# otherwise the same building would fall into "other scene" just because of
# a different spelling.
def classify(filename, scene_type):
    if filename in OVERRIDE:
        return OVERRIDE[filename]
    text = scene_type.lower()
    for cls, keywords in SCENE_RULES:
        if any(k in text for k in keywords):
            return cls
    return 'Other scene'


def already_merged(path):
    with open(path, encoding='utf-8-sig') as fh:
        return 'sub_area' in (csv.DictReader(fh).fieldnames or [])


def convert(path):
    with open(path, encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))

    out = []
    for r in rows:
        gid = r['group_id']
        if gid not in VENUE_MAP:
            sys.exit(f'{path}: unregistered group_id "{gid}", check VENUE_MAP')
        venue_id, venue_name, sub_area = VENUE_MAP[gid]
        original = r['filename'].split('_')[-1]          # strip the prefix added by the naming script
        out.append([r['filename'], r['class'], venue_id, venue_name, sub_area,
                    classify(original, r['scene_type']), r['scene_type'],
                    r['status'], r['notes']])

    with open(path, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(out)
    return len(out)


def main():
    manifests = sorted(f for f in os.listdir(ANNOTATIONS)
                       if f.endswith('_manifest.csv'))
    if not manifests:
        sys.exit(f'no manifest found: {ANNOTATIONS}/*_manifest.csv')

    total = 0
    for mf in manifests:
        path = os.path.join(ANNOTATIONS, mf)
        if already_merged(path):
            print(f'{mf:28} already collapsed to venue level, skipping')
            continue
        n = convert(path)
        total += n
        print(f'{mf:28} {n} rows collapsed')

    print(f'\n{total} rows processed in total. '
          'Next run python3 src/tools/apply_naming.py so the group_id prefix matches the CSV.')


if __name__ == '__main__':
    main()
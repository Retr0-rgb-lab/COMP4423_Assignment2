# -*- coding: utf-8 -*-
"""Rename the photos by location group and sync the manifest.

Naming rule
------------
    data/pictures/<area dir>/<group_id>_<seq:03d>_<original file name>

Examples:
    data/pictures/kowloon/OUT-03_001_IMG_0764.jpg
    data/pictures/Jordan/OUT-01_001_IMG_0796.jpg
    data/pictures/elements/OUT-04_001_IMG_0821.jpg

Why this design
---------------
- **group_id first**: the files sort by group, and when spot-checking the
  samples of one group the listing is readable at a glance.
- **seq in the middle**: the shooting order inside the group, also used for
  in-group de-duplication and gap checks.
- **the original file name is kept last**: IMG_xxxx is the original index
  from the phone export and the only clue that traces back the original
  batch and shooting order. The old name can always be recovered from the
  new one.

Important constraint
-------------------
The classifier may **only use image pixels**. File names and folder names
are for manual auditing and group cross-checks only; they must never enter
any feature or model input. The Task 5 application must not infer the class
from the path either.

Safe to re-run
--------------
The script strips the existing prefix before renumbering, so it can be run
repeatedly: every time the manifest of a new area is added, one more run
finishes the naming and numbering of that area. Photos of the same group
spread over several directories are merged into one continuous seq series.

Usage
-----
    python src/tools/apply_naming.py            # actually run
    python src/tools/apply_naming.py --dry-run  # only print the planned changes
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PICTURES = os.path.join(ROOT, 'data', 'pictures')
ANNOTATIONS = os.path.join(ROOT, 'data', 'annotations')

# Already named: OUT-05_001_IMG_0774.jpg  ->  capture group / seq / original name
NAMED = re.compile(r'^(?P<gid>(?:OUT|IN)-\d{2})_(?P<seq>\d{3})_(?P<orig>.+)$')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.heic', '.heif')


def strip_prefix(name):
    """Strip the existing group prefix and return the original file name."""
    m = NAMED.match(name)
    return m.group('orig') if m else name


def area_dirs():
    """Recursively index every leaf directory under data/pictures -> the
    relative path of the directory (lower-cased).

    The layout is `pictures/{inside,outside}/<venue name>/` and the nesting
    may go deeper. The same venue name can appear under both inside and
    outside (e.g. night), so the key is the **relative path** rather than the
    directory name, to stop the two from overwriting each other.
    """
    index = {}
    for root, dirs, files in os.walk(PICTURES):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if not any(f.lower().endswith(IMG_EXT) for f in files):
            continue                      # only register directories that really hold images
        key = os.path.relpath(root, PICTURES).replace(os.sep, '/').lower()
        index.setdefault(key, os.path.abspath(root))
    return index


def manifest_files():
    """Recursively find every manifest; returns (file list, {relative path:
    absolute path}).

    The manifests sit under data/annotations in a hierarchy that mirrors the
    photos, e.g. data/annotations/outside/night_manifest.csv corresponds to
    data/pictures/outside/night/.
    """
    files, index = [], {}
    for root, dirs, names in os.walk(ANNOTATIONS):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for n in sorted(names):
            if not n.endswith('_manifest.csv'):
                continue
            rel = os.path.relpath(root, ANNOTATIONS)
            key = n[:-len('_manifest.csv')] if rel == '.' \
                else f'{rel.replace(os.sep, "/")}/{n[:-len("_manifest.csv")]}'
            files.append(os.path.join(root, n))
            index[key.lower()] = os.path.join(root, n)
    return files, index


def load_manifests():
    """Read all the manifests; returns (record list, field names, file list)."""
    records, fields = [], None
    files, index = manifest_files()
    if not files:
        sys.exit(f'no manifest found: {ANNOTATIONS}/**/*_manifest.csv')

    dirs = area_dirs()
    for path in files:
        rel = os.path.relpath(os.path.dirname(path), ANNOTATIONS)
        stem = os.path.basename(path)[:-len('_manifest.csv')]
        key = stem if rel == '.' else f'{rel.replace(os.sep, "/")}/{stem}'
        area_dir = dirs.get(key.lower())
        if area_dir is None:
            sys.exit(f'{path}: no matching directory for "{key}" under data/pictures/')
        with open(path, encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            if fields is None:
                fields = reader.fieldnames
            for row in reader:
                # _cur = current file name on disk; _old = original name after
                # stripping the prefix, used for sorting only
                row['_cur'] = row['filename']
                row['_old'] = strip_prefix(row['filename'])
                row['_dir'] = area_dir
                row['_csv'] = os.path.relpath(path, ANNOTATIONS)
                records.append(row)

    return records, fields, files


def assign_sequences(records):
    """Group by group_id, sort within each group by original file name and
    assign a continuous seq."""
    by_group = defaultdict(list)
    for r in records:
        by_group[r['group_id']].append(r)

    for gid in sorted(by_group):
        # For fixed-length numeric names such as IMG_0764 the lexicographic
        # order is the shooting order
        for i, r in enumerate(sorted(by_group[gid], key=lambda x: x['_old']), 1):
            r['_new'] = f'{gid}_{i:03d}_{r["_old"]}'

    return by_group


def check(records, by_group):
    """Consistency checks run before the rename; returns a list of warnings."""
    warnings = []

    # 1. Disk and manifest agree
    for r in records:
        if not os.path.isfile(os.path.join(r['_dir'], r['_cur'])):
            warnings.append(f'file missing: {r["_dir"]}/{r["_cur"]} ({r["_csv"]})')

    manifest_files = {(r['_dir'], r['_cur']) for r in records}
    for area in sorted(os.listdir(PICTURES)):
        area_dir = os.path.join(PICTURES, area)
        if not os.path.isdir(area_dir):
            continue
        unlisted = [f for f in os.listdir(area_dir)
                    if f.lower().endswith(IMG_EXT)
                    and (area_dir, f) not in manifest_files]
        for f in unlisted:
            warnings.append(f'not in the manifest: {area}/{f}')

    # 2. Group too small
    for gid, rows in sorted(by_group.items()):
        usable = sum(1 for r in rows if r['status'] == 'include')
        if usable < 10:
            warnings.append(f'{gid}: only {usable} usable samples, too few to be an independent split unit')

    return warnings


def apply(records, dry_run):
    """Rename in two stages, to avoid clashes between the old and the new names."""
    todo = [r for r in records if r['_new'] != r['_cur']]

    if dry_run:
        for r in sorted(todo, key=lambda x: (x['_csv'], x['_old'])):
            print(f'  {r["_dir"]}/{r["_cur"]}  ->  {r["_new"]}')
        print(f'\n{"[dry-run] " if dry_run else ""}{len(todo)} files to rename')
        return

    tmp = {}
    for r in records:                                    # stage one: move everything to a temporary name
        src = os.path.join(r['_dir'], r['_cur'])
        if not os.path.isfile(src):
            continue
        placeholder = (f'.__naming_tmp__/{os.path.basename(r["_dir"])}'
                       f'__{r["_old"]}')
        dst = os.path.join(PICTURES, placeholder)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        tmp[(r['_dir'], r['_cur'])] = dst

    for r in records:                                    # stage two: move to the final name
        src = tmp.get((r['_dir'], r['_cur']))
        if not src:
            continue
        dst = os.path.join(r['_dir'], r['_new'])
        if os.path.exists(dst) and src != dst:
            os.remove(dst)
        os.replace(src, dst)

    tmp_root = os.path.join(PICTURES, '.__naming_tmp__')
    if os.path.isdir(tmp_root) and not os.listdir(tmp_root):
        os.rmdir(tmp_root)

    # Write the filename column back into the manifest
    by_csv = defaultdict(list)
    for r in records:
        by_csv[r['_csv']].append(r)

    for mf, rows in sorted(by_csv.items()):
        path = os.path.join(ANNOTATIONS, mf)
        with open(path, encoding='utf-8-sig') as fh:
            fields = csv.DictReader(fh).fieldnames
        for r in rows:
            r['filename'] = r['_new']
        with open(path, 'w', encoding='utf-8-sig', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

    print(f'renamed and synced {len(by_csv)} manifests, {len(records)} rows in total')


def main():
    """Rename the photos by location group and sync the manifests.

    Prints the check warnings first, then renames the files and writes the
    new file names back into the manifest; with `--dry-run` the planned
    renames are printed and nothing on disk is touched.
    """
    ap = argparse.ArgumentParser(description='rename the photos by location group and sync the manifest')
    ap.add_argument('--dry-run', action='store_true', help='only preview the changes')
    args = ap.parse_args()

    records, fields, files = load_manifests()
    by_group = assign_sequences(records)

    print(f'manifest: {len(files)} · photos {len(records)} · location groups {len(by_group)}\n')

    warnings = check(records, by_group)
    if warnings:
        print('⚠ check results')
        for w in warnings:
            print(f'  {w}')
        print()

    apply(records, args.dry_run)


if __name__ == '__main__':
    main()
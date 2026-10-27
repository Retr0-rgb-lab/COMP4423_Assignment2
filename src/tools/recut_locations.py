# -*- coding: utf-8 -*-
"""Repair the location groups that were found to straddle two places.

The 2026-10-27 photo-level re-check showed three grouping errors across
four location groups:

- ``IN-05`` mixed the main-entrance forecourt with the Shaw Amenities /
  Innovation Tower plaza (which is ``IN-01``) and with the University
  Library and University Gallery courtyard (which is ``IN-03``);
- ``IN-01`` held six frames of the Library complex;
- ``OUT-08`` and ``OUT-09`` both photographed the FREESPACE building at
  Art Park, which is a train-group and a validation-group frame of the same
  scene.

A location group has to be one place, otherwise the same scene sits on both
sides of a split and the test numbers carry a risk of optimism. This tool rewrites the
``group_id`` of the affected rows in the annotation manifests. The split
itself is not touched here: ``build_dataset.py`` re-derives
``data/annotations/splits.csv`` and the group assignment from the manifests,
so the run order is this tool first, then ``build_dataset.py``.

Usage
-----
    python src/tools/recut_locations.py --check
    python src/tools/recut_locations.py --apply

``--check`` prints what would change. ``--apply`` writes the manifests.
Both refuse to run if a filename in the table is not found, if a filename
would be counted twice, or if a target group does not exist.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOTATIONS = os.path.join(ROOT, "data", "annotations")

# The place each moved frame belongs to, and the place name the group is
# written under. The rows keep the file they were annotated in, because the
# manifest is organised by photo directory, not by group.
#
# filename -> (target group, location name, sub area, scene class)
MOVES = {
    # --- IN-05 -> IN-03: the University Library, the University Gallery, the
    #     E block (Chan Sui Wai) and the plaza lawn in front of them -------
    "IN-05_014_IMG_1317.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Ramp or steps"),
    "IN-05_015_IMG_1318.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Ramp or steps"),
    "IN-05_016_IMG_1319.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building entrance"),
    "IN-05_017_IMG_1320.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building entrance"),
    "IN-05_018_IMG_1321.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building entrance"),
    "IN-05_026_IMG_1330.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Gallery plaza lawn", "Green space"),
    "IN-05_027_IMG_1331.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Gallery plaza lawn", "Green space"),
    "IN-05_028_IMG_1332.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Gallery plaza lawn", "Green space"),
    "IN-05_029_IMG_1333.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Gallery plaza lawn", "Green space"),
    "IN-05_046_IMG_1887.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building façade"),
    "IN-05_047_IMG_1888.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building façade"),
    "IN-05_048_IMG_1889.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Gallery plaza lawn", "Green space"),
    "IN-05_049_IMG_1890.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Gallery plaza lawn", "Green space"),
    # --- IN-01 -> IN-03: the Library block seen from the E-block end -------
    "IN-01_048_IMG_1517.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building façade"),
    "IN-01_050_IMG_1519.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building façade"),
    "IN-01_051_IMG_1520.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building façade"),
    "IN-01_052_IMG_1521.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building façade"),
    "IN-01_054_IMG_1523.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building façade"),
    "IN-01_055_IMG_1524.jpg": (
        "IN-03", "PolyU University Library and central podium",
        "University Library and University Gallery", "Building façade"),
    # --- IN-05 -> IN-01: the Shaw Amenities / VA / J block ---------------
    "IN-05_019_IMG_1322.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "VA block and Innovation Tower plaza", "Building façade"),
    "IN-05_020_IMG_1323.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "VA block and Innovation Tower plaza", "Building façade"),
    "IN-05_030_IMG_1334.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "VA block and Innovation Tower plaza", "Paved plaza"),
    "IN-05_031_IMG_1335.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "VA block and Innovation Tower plaza", "Green space"),
    "IN-05_032_IMG_1336.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "VA block and Innovation Tower plaza", "Green space"),
    "IN-05_035_IMG_1339.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "J block red-brick halls", "Building façade"),
    "IN-05_036_IMG_1340.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "J block red-brick halls", "Tree-lined path"),
    "IN-05_037_IMG_1341.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "J block red-brick halls", "Building façade"),
    "IN-05_038_IMG_1342.jpg": (
        "IN-01", "PolyU walkway (J block to VA block)",
        "J block red-brick halls", "Building façade"),
    # --- OUT-09 -> OUT-08: the FREESPACE building at Art Park -------------
    "OUT-09_001_IMG_1129.jpg": (
        "OUT-08", "West Kowloon - Art Park",
        "Art Park and open lawn", "Building façade"),
    "OUT-09_003_IMG_1131.jpg": (
        "OUT-08", "West Kowloon - Art Park",
        "Art Park and open lawn", "Building façade"),
    "OUT-09_009_IMG_1137.jpg": (
        "OUT-08", "West Kowloon - Art Park",
        "Art Park and open lawn", "Building façade"),
}

# Frames whose place could not be established from the photograph. They are
# removed rather than left in a group, because a frame that might belong to
# another place is exactly what puts the same scene on both sides of a split.
EXCLUDE = {
    "IN-01_057_IMG_1526.jpg": "The subject is the Communal Building, which no "
                              "group covers, and the Library block is in the "
                              "background; the place could not be established",
    "IN-01_058_IMG_1527.jpg": "The subject is the Communal Building, which no "
                              "group covers, and the Library block is in the "
                              "background; the place could not be established",
}


def find_manifests():
    """Return every annotation manifest under data/annotations, sorted."""
    out = []
    for root, dirs, names in os.walk(ANNOTATIONS):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for n in sorted(names):
            if n.endswith("_manifest.csv"):
                out.append(os.path.join(root, n))
    return sorted(out)


def read_all(paths):
    """Read every manifest into {filename: (path, row)}."""
    rows = {}
    for p in paths:
        with open(p, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                fn = row["filename"]
                if fn in rows:
                    raise SystemExit(f"filename appears twice in the manifests: {fn}")
                rows[fn] = (p, row)
    return rows


def main():
    """Check or apply the group re-assignment."""
    ap = argparse.ArgumentParser(description="Re-assign the frames of the mixed location groups")
    ap.add_argument("--check", action="store_true", help="print the changes without writing")
    ap.add_argument("--apply", action="store_true", help="write the manifests")
    args = ap.parse_args()
    if args.check == args.apply:
        ap.error("pass exactly one of --check or --apply")

    paths = find_manifests()
    rows = read_all(paths)
    known_groups = {r["group_id"] for _, r in rows.values()}

    missing = [fn for fn in MOVES if fn not in rows]
    if missing:
        raise SystemExit("these frames are not in any manifest: " + ", ".join(missing))
    missing = [fn for fn in EXCLUDE if fn not in rows]
    if missing:
        raise SystemExit("these frames are not in any manifest: " + ", ".join(missing))
    bad_groups = sorted({g for g, _, _, _ in MOVES.values() if g not in known_groups})
    if bad_groups:
        raise SystemExit("unknown target group: " + ", ".join(bad_groups))

    by_target = defaultdict(list)
    for fn, (gid, loc, sub, scene) in MOVES.items():
        by_target[gid].append(fn)

    print("re-assignment table:")
    for gid in sorted(by_target):
        print(f"  -> {gid}: {len(by_target[gid])} frames")
    print(f"  -> excluded: {len(EXCLUDE)} frames")
    print()
    for fn in sorted(MOVES):
        path, row = rows[fn]
        gid, loc, sub, scene = MOVES[fn]
        rel = os.path.relpath(path, ROOT)
        print(f"{fn}")
        print(f"    {rel}")
        print(f"    group    {row['group_id']} -> {gid}")
        print(f"    place    {row['location_name']!r} -> {loc!r}")
        print(f"    sub area {row['sub_area']!r} -> {sub!r}")
        print(f"    scene    {row['scene_class']!r} -> {scene!r}")
    for fn in sorted(EXCLUDE):
        path, row = rows[fn]
        print(f"{fn}")
        print(f"    {os.path.relpath(path, ROOT)}")
        print(f"    group    {row['group_id']} -> excluded")
        print(f"    reason   {EXCLUDE[fn]}")

    if not args.apply:
        print("\n--check only, nothing was written")
        return

    touched = defaultdict(list)
    for fn, (gid, loc, sub, scene) in MOVES.items():
        path, row = rows[fn]
        row["group_id"] = gid
        row["location_name"] = loc
        row["sub_area"] = sub
        row["scene_class"] = scene
        touched[path].append(row)
    for fn, reason in EXCLUDE.items():
        path, row = rows[fn]
        row["status"] = "exclude"
        row["notes"] = reason
        touched[path].append(row)

    for path, changed in touched.items():
        with open(path, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            allrows = list(reader)
        by_name = {r["filename"]: r for r in changed}
        for r in allrows:
            if r["filename"] in by_name:
                r.update(by_name[r["filename"]])
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(allrows)
        print(f"wrote {os.path.relpath(path, ROOT)} ({len(changed)} rows changed)")

    # read back and confirm
    after = read_all(find_manifests())
    for fn, (gid, _, _, _) in MOVES.items():
        got = after[fn][1]["group_id"]
        if got != gid:
            raise SystemExit(f"write-back check failed for {fn}: {got} != {gid}")
    for fn in EXCLUDE:
        got = after[fn][1]["status"]
        if got != "exclude":
            raise SystemExit(f"write-back check failed for {fn}: status {got}")
    print(f"\nverified {len(MOVES)} moved and {len(EXCLUDE)} excluded in {len(touched)} manifests")


if __name__ == "__main__":
    main()

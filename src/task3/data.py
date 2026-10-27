# -*- coding: utf-8 -*-
"""Read splits.csv and resolve every file name to an image path. Paths and
group ids never enter the features."""
import csv
import os
from collections import defaultdict

from src.task3.config import FOLDS, PICTURES, SPLITS_CSV

EXPECTED = 534


def _index_pictures(root):
    """Return a mapping from image file name to path below `root`.

    A name that occurs twice cannot be resolved uniquely, so the walk stops
    with SystemExit in that case.
    """
    found = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                if name in found:
                    raise SystemExit(f"duplicate file name, cannot be resolved uniquely: {name}")
                found[name] = os.path.join(dirpath, name)
    return found


def load_index():
    """Return the records of `splits.csv` with every file name resolved to a path.

    The file name only locates the image; neither it nor the group id ever
    reaches the features. Raises SystemExit when the table is missing, a
    listed image is not on disk, the row count is not the expected one, a
    location group spans several splits, or a held-out fold group is not in
    the training split.
    """
    if not SPLITS_CSV.is_file():
        raise SystemExit(f"split table not found: {SPLITS_CSV}")
    paths = _index_pictures(PICTURES)
    records = []
    missing = []
    with open(SPLITS_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["filename"]
            if name not in paths:
                missing.append(name)
                continue
            records.append({
                "filename": name,
                "y": int(row["class"]),
                "group_id": row["group_id"],
                "split": row["split"],
                "path": paths[name],
            })
    if missing:
        raise SystemExit(f"{len(missing)} are listed in splits.csv but not on disk, for example {missing[:3]}")
    if len(records) != EXPECTED:
        raise SystemExit(f"expected {EXPECTED} images, read {len(records)}")

    split_of = defaultdict(set)
    for rec in records:
        split_of[rec["group_id"]].add(rec["split"])
    crossed = {g: s for g, s in split_of.items() if len(s) != 1}
    if crossed:
        raise SystemExit(f"location group spans several splits: {crossed}")

    train_groups = {r["group_id"] for r in records if r["split"] == "train"}
    needed = set()
    for fold in FOLDS:
        needed.update(fold["inside"])
        needed.update(fold["outside"])
    absent = needed - train_groups
    if absent:
        raise SystemExit(f"groups of the held-out fold are not in the training set: {sorted(absent)}")
    return records


def by_split(records):
    """Return the records grouped by split, under the keys train, val and test."""
    out = {"train": [], "val": [], "test": []}
    for rec in records:
        out[rec["split"]].append(rec)
    return out


def partition_fold(records, fold):
    """Return (train records, held-out records), both taken only from the
    official train split."""
    held_groups = set(fold["inside"]) | set(fold["outside"])
    train, held = [], []
    for rec in records:
        if rec["split"] != "train":
            continue
        if rec["group_id"] in held_groups:
            held.append(rec)
        else:
            train.append(rec)
    got = {r["group_id"] for r in held}
    if got != held_groups:
        raise SystemExit(f"held-out groups are incomplete: expected {sorted(held_groups)}, got {sorted(got)}")
    labels = {r["y"] for r in held}
    if labels != {0, 1}:
        raise SystemExit(f"held-out fold misses a class: {labels}")
    return train, held

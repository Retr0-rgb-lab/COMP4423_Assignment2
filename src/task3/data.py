# -*- coding: utf-8 -*-
"""读 splits.csv，把文件名解析成图片路径。路径和组号不进入特征。"""
import csv
import os
from collections import defaultdict

from src.task3.config import FOLDS, PICTURES, SPLITS_CSV

EXPECTED = 540


def _index_pictures(root):
    found = {}
    for dirpath, _dirs, names in os.walk(root):
        for name in names:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                if name in found:
                    raise SystemExit(f"文件名重复，无法唯一解析: {name}")
                found[name] = os.path.join(dirpath, name)
    return found


def load_index():
    if not SPLITS_CSV.is_file():
        raise SystemExit(f"找不到划分表: {SPLITS_CSV}")
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
        raise SystemExit(f"{len(missing)} 张在 splits.csv 里但磁盘上没有，例如 {missing[:3]}")
    if len(records) != EXPECTED:
        raise SystemExit(f"期望 {EXPECTED} 张，读到 {len(records)} 张")

    split_of = defaultdict(set)
    for rec in records:
        split_of[rec["group_id"]].add(rec["split"])
    crossed = {g: s for g, s in split_of.items() if len(s) != 1}
    if crossed:
        raise SystemExit(f"地点组跨 split: {crossed}")

    train_groups = {r["group_id"] for r in records if r["split"] == "train"}
    needed = set()
    for fold in FOLDS:
        needed.update(fold["inside"])
        needed.update(fold["outside"])
    absent = needed - train_groups
    if absent:
        raise SystemExit(f"留出折里的组不在训练集: {sorted(absent)}")
    return records


def by_split(records):
    out = {"train": [], "val": [], "test": []}
    for rec in records:
        out[rec["split"]].append(rec)
    return out


def partition_fold(records, fold):
    """返回 (训练记录, 留出记录)，都只来自官方 train。"""
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
        raise SystemExit(f"留出组不完整: 期望 {sorted(held_groups)}，得到 {sorted(got)}")
    labels = {r["y"] for r in held}
    if labels != {0, 1}:
        raise SystemExit(f"留出折缺少类别: {labels}")
    return train, held

# -*- coding: utf-8 -*-
"""按地点组重命名照片，并同步 manifest。

命名规则
--------
    data/pictures/<区域目录>/<group_id>_<seq:03d>_<原始文件名>

例：
    data/pictures/kowloon/OUT-05_001_IMG_0774.jpg
    data/pictures/Jordan/OUT-01_001_IMG_0796.jpg
    data/pictures/elements/OUT-11_001_IMG_0821.jpg

为什么这样设计
--------------
- **group_id 在前**：按组排序、抽查某个组的样本时，目录一眼可辨。
- **seq 在中**：组内拍摄顺序，同时用于组内去重与缺号检查。
- **原始文件名保留在最后**：IMG_xxxx 是手机导出时的原始序号，是唯一能追回
  原始批次与拍摄顺序的线索。任何时候都能从新名还原旧名。

重要约束（来自作业 PDF）
------------------------
分类器**只能使用图像像素**。文件名与文件夹名仅供人工审计与分组核对，
不得进入任何特征或模型输入。Task 5 的应用也不得从路径推断类别。

重复执行是安全的
----------------
脚本先剥掉已有的前缀再重新编号，因此可以反复运行：每补一个新区域的
manifest，重跑一次即可完成该区域的命名与编号。同一 group 跨目录的照片
会合并成一条连续的 seq 序列。

用法
----
    python src/tools/apply_naming.py            # 实际执行
    python src/tools/apply_naming.py --dry-run  # 只打印将要做的改动
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

# 已命名：OUT-05_001_IMG_0774.jpg  →  捕获 group / seq / 原始名
NAMED = re.compile(r'^(?P<gid>(?:OUT|IN)-\d{2})_(?P<seq>\d{3})_(?P<orig>.+)$')
IMG_EXT = ('.jpg', '.jpeg', '.png', '.heic', '.heif')


def strip_prefix(name):
    """剥掉已有的组前缀，返回原始文件名。"""
    m = NAMED.match(name)
    return m.group('orig') if m else name


def area_dirs():
    """小写名 → 实际目录名。目录名大小写未必与 manifest 文件名一致，
    而 Windows 盘的挂载常常对大小写不敏感，显式建映射避免依赖该巧合。"""
    return {d.lower(): os.path.join(PICTURES, d)
            for d in os.listdir(PICTURES)
            if os.path.isdir(os.path.join(PICTURES, d))}


def load_manifests():
    """读取全部 manifest，返回按 (区域目录, 文件名) 索引的记录列表。"""
    records, fields = [], None
    files = sorted(f for f in os.listdir(ANNOTATIONS)
                   if f.endswith('_manifest.csv'))
    if not files:
        sys.exit(f'没有找到 manifest: {ANNOTATIONS}/*_manifest.csv')

    dirs = area_dirs()
    for mf in files:
        area = mf[:-len('_manifest.csv')]
        area_dir = dirs.get(area.lower())
        if area_dir is None:
            sys.exit(f'{mf}: data/pictures/ 下找不到区域目录「{area}」')
        with open(os.path.join(ANNOTATIONS, mf), encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            if fields is None:
                fields = reader.fieldnames
            for row in reader:
                # _cur = 磁盘上的当前文件名；_old = 剥掉前缀后的原始名，仅用于排序
                row['_cur'] = row['filename']
                row['_old'] = strip_prefix(row['filename'])
                row['_dir'] = area_dir
                row['_csv'] = mf
                records.append(row)

    return records, fields, files


def assign_sequences(records):
    """按 group_id 分组，组内按原始文件名排序，分配连续 seq。"""
    by_group = defaultdict(list)
    for r in records:
        by_group[r['group_id']].append(r)

    for gid in sorted(by_group):
        # IMG_0764 这类定长数字文件名字典序即拍摄顺序
        for i, r in enumerate(sorted(by_group[gid], key=lambda x: x['_old']), 1):
            r['_new'] = f'{gid}_{i:03d}_{r["_old"]}'

    return by_group


def check(records, by_group):
    """执行前的一致性检查，返回警告列表。"""
    warnings = []

    # 1. 磁盘与 manifest 对齐
    for r in records:
        if not os.path.isfile(os.path.join(r['_dir'], r['_cur'])):
            warnings.append(f'文件缺失: {r["_dir"]}/{r["_cur"]}（{r["_csv"]}）')

    manifest_files = {(r['_dir'], r['_cur']) for r in records}
    for area in sorted(os.listdir(PICTURES)):
        area_dir = os.path.join(PICTURES, area)
        if not os.path.isdir(area_dir):
            continue
        unlisted = [f for f in os.listdir(area_dir)
                    if f.lower().endswith(IMG_EXT)
                    and (area_dir, f) not in manifest_files]
        for f in unlisted:
            warnings.append(f'未进 manifest: {area}/{f}')

    # 2. 组内过小
    for gid, rows in sorted(by_group.items()):
        usable = sum(1 for r in rows if r['status'] == 'include')
        if usable < 10:
            warnings.append(f'{gid}: 可用样本仅 {usable} 张，不足以作为独立划分单元')

    return warnings


def apply(records, dry_run):
    """两阶段改名，避免新旧名互相占用导致的冲突。"""
    todo = [r for r in records if r['_new'] != r['_cur']]

    if dry_run:
        for r in sorted(todo, key=lambda x: (x['_csv'], x['_old'])):
            print(f'  {r["_dir"]}/{r["_cur"]}  ->  {r["_new"]}')
        print(f'\n{"[dry-run] " if dry_run else ""}共 {len(todo)} 个文件待改名')
        return

    tmp = {}
    for r in records:                                    # 阶段一：全部移到临时名
        src = os.path.join(r['_dir'], r['_cur'])
        if not os.path.isfile(src):
            continue
        placeholder = (f'.__naming_tmp__/{os.path.basename(r["_dir"])}'
                       f'__{r["_old"]}')
        dst = os.path.join(PICTURES, placeholder)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
        tmp[(r['_dir'], r['_cur'])] = dst

    for r in records:                                    # 阶段二：落到最终名
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

    # 回写 manifest 的 filename 列
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

    print(f'已改名并同步 {len(by_csv)} 个 manifest，共 {len(records)} 行')


def main():
    ap = argparse.ArgumentParser(description='按地点组重命名照片并同步 manifest')
    ap.add_argument('--dry-run', action='store_true', help='只预览改动')
    args = ap.parse_args()

    records, fields, files = load_manifests()
    by_group = assign_sequences(records)

    print(f'manifest: {len(files)} 个 · 照片 {len(records)} 张 · 地点组 {len(by_group)} 个\n')

    warnings = check(records, by_group)
    if warnings:
        print('⚠ 检查结果')
        for w in warnings:
            print(f'  {w}')
        print()

    apply(records, args.dry_run)


if __name__ == '__main__':
    main()
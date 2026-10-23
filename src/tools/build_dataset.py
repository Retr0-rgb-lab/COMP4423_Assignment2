# -*- coding: utf-8 -*-
"""Dataset cleaning and splitting: balance the classes, cap the per-group
capacity and split by location group, as Task 2 requires.

Based on the Task 2 items of the Assignment 2 PDF
-------------------------------------------------
1. Location diversity -- cover several locations in both classes; neighbouring
   shots of the same courtyard / building / landmark belong to **the same
   location group**.
2. Data split -- split **by location group within each class**, roughly
   60/20/20, with both classes represented in every split.
3. Prevent leakage -- all shots of the same scene (including repeat visits,
   burst shots, video frames and retouched copies) must stay in the same group
   and the same split; the photos of one scene must not be scattered at
   random across different sets. Data augmentation is applied only after the
   split, and only to the training set.
4. The report has to give the actual number of images and the number of
   location groups for every class and every split.

Cleaning steps
--------------
1. Keep only the samples with status == include (review / exclude never
   enter the dataset)
2. **Cap the group capacity**: no single location group may exceed CAP
   images, so that no venue dominates the training set or turns the metric of
   a split into "the recognition rate of that one venue"
3. **Class balancing**: align the two classes to the smaller side
4. **Split by group**: the group is an indivisible unit; all photos of a
   group go into the same split

Within a group the photos are thinned out by evenly spaced sampling instead of
taking the first N, so that the original span of viewpoints / distances is
preserved.

Deterministic: neither the sampling nor the splitting uses a random number,
so repeated runs give exactly the same result.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOTATIONS = os.path.join(ROOT, 'data', 'annotations')
SPLIT_FILE = os.path.join(ANNOTATIONS, 'splits.csv')
SUMMARY_FILE = os.path.join(ANNOTATIONS, 'dataset_summary.md')

CAP = 50                  # cap for a single location group
RATIO = (0.6, 0.2, 0.2)  # train / val / test
SPLITS = ('train', 'val', 'test')

# Mutually-visible constraint: the groups below can see each other across a
# road, so they are the "neighbouring shots of the same scene" the PDF talks
# about. Even though the venues differ they must not land in different
# splits, otherwise that is leakage. The split treats them as one atomic
# unit.
KEEP_TOGETHER = [
    ('OUT-05', 'OUT-08'),   # West Kowloon M+ Museum and Art Park, mutually visible across the road
]


def even_sample(items, k):
    """Take k items out of items at evenly spaced positions, preserving the
    original span of viewpoints."""
    if k >= len(items):
        return list(items)
    if k <= 0:
        return []
    step = (len(items) - 1) / (k - 1) if k > 1 else 0
    idx = sorted({round(i * step) for i in range(k)})
    return [items[i] for i in idx]


def largest_remainder(n, ratio):
    """Distribute n units over the splits by ratio (largest remainder
    method); returns [train, val, test]."""
    raw = [n * r for r in ratio]
    base = [int(x) for x in raw]
    rest = n - sum(base)
    # The remainder is handed out by descending fractional part, at most once
    # per split
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:max(0, rest)]:
        base[i] += 1
    # Every split keeps at least 1 unit; when that exceeds the total, take
    # back from the largest one
    for i in range(len(base)):
        base[i] = max(1, base[i])
    while sum(base) > n and max(base) > 1:
        base[base.index(max(base))] -= 1
    return base


def build_clusters(groups):
    """Merge the mutually-visible combinations of KEEP_TOGETHER into atomic
    units.

    Returns {unit representative: [group_id, ...]}; the groups without a
    constraint each form their own unit.
    """
    parent = {g: g for g in groups}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in KEEP_TOGETHER:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    clusters = defaultdict(list)
    for g in groups:
        clusters[find(g)].append(g)
    return clusters


def assign_groups(groups, ratio):
    # groups: {group_id: [rows]}, sorted by group capacity
    """Assign the location groups of one class to the splits.

    When there are few groups (the larger class has only 5~9), forcing a
    60/20/20 split leaves val / test with a single group each and the metric
    loses its meaning. The strategy is:

    1. First merge the mutually-visible groups into atomic units (see
       KEEP_TOGETHER), otherwise there is leakage
    2. train takes the **largest** units within its quota -- the most
       training data
    3. The remaining units are handed out alternately between val / test, and
       each round picks the side that currently holds fewer images, so that
       the two stay as close as possible
    4. The number of groups in val / test is not forced to be equal (it is
       impossible when there are too few groups), but it is kept as balanced
       as possible

    Returns {group_id: split}
    """
    clusters = build_clusters(sorted(groups))
    n = len(clusters)
    quota = largest_remainder(n, ratio)
    order = sorted(clusters.values(),
                   key=lambda c: -sum(len(groups[g]) for g in c))

    out = {}
    for c in order[:quota[0]]:
        for g in c:
            out[g] = SPLITS[0]

    buckets = {SPLITS[1]: [], SPLITS[2]: []}
    load = {SPLITS[1]: 0, SPLITS[2]: 0}
    for c in order[quota[0]:]:
        s = min(buckets, key=lambda k: (len(buckets[k]), load[k]))
        buckets[s].append(c)
        load[s] += sum(len(groups[g]) for g in c)
        for g in c:
            out[g] = s
    return out


def load_rows():
    rows = []
    for root, dirs, names in os.walk(ANNOTATIONS):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for n in sorted(names):
            if n.endswith('_manifest.csv'):
                rows += list(csv.DictReader(
                    open(os.path.join(root, n), encoding='utf-8-sig')))
    return rows


def main():
    ap = argparse.ArgumentParser(description='Clean the dataset and split it by location group')
    ap.add_argument('--cap', type=int, default=CAP, help='maximum images kept per location group')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rows = load_rows()
    if not rows:
        sys.exit('no usable manifest was found')

    # ── 1. Keep only include ─────────────────────────────────────────
    kept = [r for r in rows if r['status'] == 'include']
    dropped = {
        'exclude': sum(1 for r in rows if r['status'] == 'exclude'),
        'review': sum(1 for r in rows if r['status'] == 'review'),
    }

    # ── 2. Cap the group capacity (evenly spaced sampling) ───────────
    by_group = defaultdict(list)
    for r in kept:
        by_group[r['group_id']].append(r)
    capped = {}
    for gid, items in by_group.items():
        items.sort(key=lambda r: r['filename'])
        capped[gid] = even_sample(items, args.cap)

    # ── 3. Class balancing ──────────────────────────────────────────
    by_class = defaultdict(lambda: defaultdict(list))
    meta = {}
    for gid, items in capped.items():
        cls = items[0]['class']
        by_class[cls][gid] = items
        meta[gid] = (items[0]['location_name'], cls)
    totals = {c: sum(len(v) for v in by_class[c].values()) for c in by_class}
    target = min(totals.values()) if totals else 0

    def trim(cls):
        groups = by_class[cls]
        keep = {}
        # First share the quota evenly over the groups, then top up to the
        # target in descending group capacity
        base = target // len(groups)
        for gid in sorted(groups, key=lambda g: (-len(groups[g]), g)):
            keep[gid] = even_sample(groups[gid], base)
        short = target - sum(len(v) for v in keep.values())
        for gid in sorted(groups, key=lambda g: (-len(groups[g]), g)):
            if short <= 0:
                break
            extra = even_sample(groups[gid], min(len(groups[gid]),
                                                   len(keep[gid]) + short))
            short -= len(extra) - len(keep[gid])
            keep[gid] = extra
        return keep

    balanced = {c: trim(c) for c in by_class}

    # ── 4. Split by location group (indivisible unit) ───────────────
    split_of = {}
    for cls in sorted(balanced):
        groups = balanced[cls]
        for gid, s in assign_groups(groups, RATIO).items():
            split_of[gid] = s

    # ── 5. Validation ───────────────────────────────────────────────
    errs = []
    for a, b in KEEP_TOGETHER:
        if a in split_of and b in split_of and split_of[a] != split_of[b]:
            errs.append(f'mutually visible groups {a} and {b} landed in different splits, which leaks')
    seen = defaultdict(set)
    for cls in balanced:
        for gid, s in split_of.items():
            if gid in seen and seen[gid] != {s}:
                errs.append(f'{gid} landed in more than one split')
            seen[gid] = {s}
    for s in SPLITS:
        cs = {meta[g][1] for g in split_of if split_of[g] == s}
        if cs != {'0', '1'}:
            errs.append(f'split {s} does not hold both classes: {sorted(cs)}')
    if errs:
        sys.exit('validation failed:\n  ' + '\n  '.join(errs))

    # ── 6. Output ───────────────────────────────────────────────────
    out = []
    for cls in sorted(balanced):
        for gid in sorted(balanced[cls]):
            for r in balanced[cls][gid]:
                out.append({
                    'filename': r['filename'],
                    'class': r['class'],
                    'group_id': gid,
                    'location_name': meta[gid][0],
                    'sub_area': r['sub_area'],
                    'scene_class': r['scene_class'],
                    'split': split_of[gid],
                })
    out.sort(key=lambda r: (r['class'], r['split'], r['group_id'], r['filename']))

    if args.dry_run:
        print('[dry-run] no file was written')
    else:
        with open(SPLIT_FILE, 'w', encoding='utf-8-sig', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        write_summary(balanced, split_of, meta, by_group, dropped, args.cap)

    # ── Report ──────────────────────────────────────────────────────
    print(f'read {len(rows)} images, dropped {dropped["exclude"]} marked exclude '
          f'and {dropped["review"]} marked review, {len(kept)} usable')
    print(f'per-group cap {args.cap}, {sum(len(v) for v in capped.values())} images after capping')
    print(f'{sum(len(v) for c in balanced for v in balanced[c].values())} images after class balancing\n')
    hdr = f'{"split":<8}{"class":<8}{"images":>8}{"groups":>8}{"group ids":<26}'
    print(hdr)
    print('-' * len(hdr))
    for s in SPLITS:
        for cls in ('0', '1'):
            gs = sorted(g for g in split_of if split_of[g] == s
                        and meta[g][1] == cls)
            n = sum(len(balanced[cls][g]) for g in gs)
            print(f'{s:<8}{cls:<8}{n:>8}{len(gs):>8}   {", ".join(gs):<24}')
    tot = defaultdict(int)
    for r in out:
        tot[(r['split'], r['class'])] += 1
    print(f'\ntotal {len(out)} images, {len(split_of)} location groups, '
          f'class ratio {tot[("train","0")]+tot[("val","0")]+tot[("test","0")]}'
          f':{tot[("train","1")]+tot[("val","1")]+tot[("test","1")]}')


def write_summary(balanced, split_of, meta, by_group, dropped, cap):
    def cls_of(c):
        return {'0': 'outside', '1': 'inside'}.get(c, c)

    L = ['# Dataset cleaning and split result', '',
         '> Generated by `src/tools/build_dataset.py`. Do not edit by hand.', '',
         '## Cleaning rules', '',
         f'- Only `status = include` is kept; {dropped["exclude"]} exclude rows and '
         f'{dropped["review"]} review rows are dropped',
         f'- At most {cap} images per location group (evenly spaced inside the group, '
         f'so the spread of viewpoints is kept)',
         '- The two classes are aligned to the smaller side',
         '- **A location group is indivisible**: every photograph of a group enters the '
         'same split',
         '- Groups that see each other are bound to the same split: '
         + '; '.join(' / '.join(t) for t in KEEP_TOGETHER)
         + ' (they face each other across a road)', '',
         '## Images and location groups per class and split', '',
         '| split | class | images | groups | group ids |', '|---|---|---:|---:|---|']
    for s in SPLITS:
        for c in ('0', '1'):
            gs = sorted(g for g in split_of if split_of[g] == s
                        and meta[g][1] == c)
            n = sum(len(balanced[c][g]) for g in gs)
            L.append(f'| {s} | {cls_of(c)} | {n} | {len(gs)} | '
                     f'{", ".join(gs)} |')
    L += ['', '## Per-group detail', '',
          '| group_id | place | class | annotated | kept | split |',
          '|---|---|---|---:|---:|---|']
    for g in sorted(split_of, key=lambda x: (meta[x][1], x)):
        loc, c = meta[g]
        L.append(f'| {g} | {loc} | {cls_of(c)} | {len(by_group[g])} | '
                 f'{len(balanced[c][g])} | {split_of[g]} |')
    tot = {c: sum(len(v) for v in balanced[c].values()) for c in balanced}
    L += ['', '## Totals', '',
          f'- inside {tot.get("1", 0)} images, outside {tot.get("0", 0)} images',
          f'- {len(split_of)} location groups (inside '
          f'{sum(1 for g in split_of if meta[g][1] == "1")}, outside '
          f'{sum(1 for g in split_of if meta[g][1] == "0")})',
          f'- {sum(tot.values())} images in total', '']
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))


if __name__ == '__main__':
    main()
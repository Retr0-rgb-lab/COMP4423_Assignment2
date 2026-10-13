# -*- coding: utf-8 -*-
"""数据集清洗与划分：按 Task 2 要求平衡类别、封顶组容量、按地点组划分。

依据 Assignment 2 PDF 的 Task 2 条目
------------------------------------
1. Location diversity —— 覆盖两类中的多个地点；同一庭院／建筑／地标的邻近
   取景归为**同一个地点组**。
2. Data split —— 在每个类别**内部按地点组**划分，约 60/20/20，每个 split 两类
   都要有代表。
3. Prevent leakage —— 同一场景的全部取景（含复访、连拍、视频帧、修饰副本）
   必须留在同一组、同一 split；不得把一个场景的图随机拆到不同集合。
   数据增强只在划分之后、只对训练集施加。
4. 报告需给出每个类别、每个 split 的实际图片数与地点组数。

清洗步骤
--------
1. 只保留 status == include 的样本（review / exclude 一律不进数据集）
2. **组容量封顶**：任何单一地点组不得超过 CAP 张，避免某个场所主导训练
   或让某一 split 的指标等价于「这个场所的识别率」
3. **类别平衡**：两类对齐到较小的一侧
4. **按组划分**：组为不可分割单元，组内所有照片进同一个 split

组内抽稀采用等距取样而非取前 N 张，以保留原有的视点／距离跨度。

确定性：抽样与划分都不使用随机数，重复运行结果完全一致。
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

CAP = 50                  # 单个地点组上限
RATIO = (0.6, 0.2, 0.2)  # train / val / test
SPLITS = ('train', 'val', 'test')

# 互见约束：下列组之间隔着一条路就能互相看到，属于 PDF 所说的
# 「同一场景的邻近取景」，即便地点不同也不能分到不同 split，
# 否则会构成泄漏。划分时把它们当作一个原子单元。
KEEP_TOGETHER = [
    ('OUT-05', 'OUT-08'),   # 西九 M+ 博物館 与 Art Park，隔路互相通视
]


def even_sample(items, k):
    """从 items 中等距取 k 个，保留原有的视点跨度。"""
    if k >= len(items):
        return list(items)
    if k <= 0:
        return []
    step = (len(items) - 1) / (k - 1) if k > 1 else 0
    idx = sorted({round(i * step) for i in range(k)})
    return [items[i] for i in idx]


def largest_remainder(n, ratio):
    """按比例把 n 个单元分配到各 split（最大余数法），返回 [train, val, test]。"""
    raw = [n * r for r in ratio]
    base = [int(x) for x in raw]
    rest = n - sum(base)
    # 余数按小数部分从大到小依次分配，每个 split 至多拿一次
    order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:max(0, rest)]:
        base[i] += 1
    # 每个 split 至少 1 个单元；超出总数时从最大的回收
    for i in range(len(base)):
        base[i] = max(1, base[i])
    while sum(base) > n and max(base) > 1:
        base[base.index(max(base))] -= 1
    return base


def build_clusters(groups):
    """把 KEEP_TOGETHER 里互相通视的组合并成原子单元。

    返回 {单元代表: [group_id, ...]}，未受约束的组各自成单元。
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
    # groups: {group_id: [rows]}，按组容量排序
    """把一个类别的地点组分配到各 split。

    组数少的时候（大类只有 5~9 个组），硬套 60/20/20 会让 val / test 只剩
    1 个组，指标失去意义。策略是：

    1. 互相通视的组先合并成原子单元（见 KEEP_TOGETHER），否则会泄漏
    2. train 拿走配额内**最大的**那几个单元 —— 训练数据量最大
    3. 剩下的单元在 val / test 之间轮流分配，且每轮选当前图片数少的一边，
       使两者的图片数尽量接近
    4. val / test 的组数不强行相等（组数不够时无法做到），但会尽量均衡

    返回 {group_id: split}
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
    ap = argparse.ArgumentParser(description='清洗数据集并按地点组划分')
    ap.add_argument('--cap', type=int, default=CAP, help='单组容量上限')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rows = load_rows()
    if not rows:
        sys.exit('没有可用的 manifest')

    # ── 1. 只保留 include ────────────────────────────────────────────
    kept = [r for r in rows if r['status'] == 'include']
    dropped = {
        'exclude': sum(1 for r in rows if r['status'] == 'exclude'),
        'review': sum(1 for r in rows if r['status'] == 'review'),
    }

    # ── 2. 组容量封顶（等距取样，保留视点跨度）───────────────────────
    by_group = defaultdict(list)
    for r in kept:
        by_group[r['group_id']].append(r)
    capped = {}
    for gid, items in by_group.items():
        items.sort(key=lambda r: r['filename'])
        capped[gid] = even_sample(items, args.cap)

    # ── 3. 类别平衡 ─────────────────────────────────────────────────
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
        # 先按组数均分配额，再按组容量降序补齐到 target
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

    # ── 4. 按地点组划分（组为不可分割单元）──────────────────────────
    split_of = {}
    for cls in sorted(balanced):
        groups = balanced[cls]
        for gid, s in assign_groups(groups, RATIO).items():
            split_of[gid] = s

    # ── 5. 校验 ─────────────────────────────────────────────────────
    errs = []
    for a, b in KEEP_TOGETHER:
        if a in split_of and b in split_of and split_of[a] != split_of[b]:
            errs.append(f'互见组 {a} 与 {b} 被分到不同 split（会泄漏）')
    seen = defaultdict(set)
    for cls in balanced:
        for gid, s in split_of.items():
            if gid in seen and seen[gid] != {s}:
                errs.append(f'{gid} 被分到多个 split')
            seen[gid] = {s}
    for s in SPLITS:
        cs = {meta[g][1] for g in split_of if split_of[g] == s}
        if cs != {'0', '1'}:
            errs.append(f'split {s} 类别不全：{sorted(cs)}')
    if errs:
        sys.exit('校验失败：\n  ' + '\n  '.join(errs))

    # ── 6. 输出 ─────────────────────────────────────────────────────
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
        print('[dry-run] 未写入文件')
    else:
        with open(SPLIT_FILE, 'w', encoding='utf-8-sig', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        write_summary(balanced, split_of, meta, by_group, dropped, args.cap)

    # ── 报告 ────────────────────────────────────────────────────────
    print(f'原始 {len(rows)} 张 → 剔除 exclude {dropped["exclude"]} 张、'
          f'review {dropped["review"]} 张 → 可用 {len(kept)} 张')
    print(f'单组上限 {args.cap} → 封顶后 {sum(len(v) for v in capped.values())} 张')
    print(f'类别平衡后 {sum(len(v) for c in balanced for v in balanced[c].values())} 张\n')
    hdr = f'{"split":<8}{"class":<8}{"图片数":>8}{"地点组":>8}{"组列表":<26}'
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
    print(f'\n合计 {len(out)} 张 · 地点组 {len(split_of)} 个 · '
          f'两类比例 {tot[("train","0")]+tot[("val","0")]+tot[("test","0")]}'
          f':{tot[("train","1")]+tot[("val","1")]+tot[("test","1")]}')


def write_summary(balanced, split_of, meta, by_group, dropped, cap):
    def cls_of(c):
        return {'0': 'outside', '1': 'inside'}.get(c, c)

    L = ['# 数据集清洗与划分结果', '',
         '> 由 `src/tools/build_dataset.py` 生成，请勿手工编辑。', '',
         '## 清洗规则', '',
         f'- 只保留 `status = include`；剔除 exclude {dropped["exclude"]} 张、'
         f'review {dropped["review"]} 张', 
         f'- 单一地点组上限 {cap} 张（组内等距取样，保留视点跨度）',
         '- 类别对齐到较小的一侧',
         '- **地点组为不可分割单元**：同一组全部照片进入同一 split',
         '- 互见组绑定同 split：' + '；'.join(' / '.join(t) for t in KEEP_TOGETHER)
           + '（隔路可互相看到）', '',
         '## 每类每 split 的图片数与地点组数', '',
         '| split | 类别 | 图片数 | 地点组数 | 地点组 |', '|---|---|---:|---:|---|']
    for s in SPLITS:
        for c in ('0', '1'):
            gs = sorted(g for g in split_of if split_of[g] == s
                        and meta[g][1] == c)
            n = sum(len(balanced[c][g]) for g in gs)
            L.append(f'| {s} | {cls_of(c)} | {n} | {len(gs)} | '
                     f'{", ".join(gs)} |')
    L += ['', '## 各地点组明细', '', '| group_id | 场所 | 类别 | 原始 | 入库 | split |',
          '|---|---|---|---:|---:|---|']
    for g in sorted(split_of, key=lambda x: (meta[x][1], x)):
        loc, c = meta[g]
        L.append(f'| {g} | {loc} | {cls_of(c)} | {len(by_group[g])} | '
                 f'{len(balanced[c][g])} | {split_of[g]} |')
    tot = {c: sum(len(v) for v in balanced[c].values()) for c in balanced}
    L += ['', '## 汇总', '',
          f'- inside {tot.get("1", 0)} 张 · outside {tot.get("0", 0)} 张',
          f'- 地点组 {len(split_of)} 个（inside '
          f'{sum(1 for g in split_of if meta[g][1] == "1")} · outside '
          f'{sum(1 for g in split_of if meta[g][1] == "0")}）',
          f'- 总计 {sum(tot.values())} 张', '']
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))


if __name__ == '__main__':
    main()
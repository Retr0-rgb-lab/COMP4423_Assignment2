# -*- coding: utf-8 -*-
"""把地点组从「细分组」收敛为「场所级」，并补上内部细分与场景类别两列。

背景
----
作业要求按**地点组**划分 train/val/test。组切得越细，测试集的有效样本量
越大，但组越细也越容易把同一处场所拆到不同 split，造成泄漏。

本脚本采用保守口径：**一个物理场所 = 一个 group_id（划分单元）**，
场所内部的不同场景改用 `sub_area` + `scene_class` 两列记录。
这样划分单元一定是物理上一处确定的地方，不可能发生同一场景跨 split；
而场所内部的多样性仍被完整保留，可在报告中做「场所 × 场景类别」交叉分析。

幂等：若 manifest 已含 `sub_area` 列则直接退出，可反复运行。
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ANNOTATIONS = os.path.join(ROOT, 'data', 'annotations')

# 细分组 → (场所级 group_id, 场所名, 场所内部细分)
# inside 已在 data/annotations/inside/ 写成场所级，location_name 以那些 manifest 为准，
# 不在这里用单一 sub_area 覆盖。
VENUE_MAP = {
    'OUT-01': ('OUT-01', '佐敦 Jordan',           '佐敦道街道段'),
    'OUT-02': ('OUT-01', '佐敦 Jordan',           '廟街夜市'),
    'OUT-03': ('OUT-02', '港大 HKU',              'Main Building 传统红砖区'),
    'OUT-04': ('OUT-02', '港大 HKU',              '校园林荫绿地'),
    'OUT-13': ('OUT-02', '港大 HKU',              '现代教学楼与连廊'),
    'OUT-05': ('OUT-03', '九龙公园 Kowloon Park', '中式庭園'),
    'OUT-06': ('OUT-03', '九龙公园 Kowloon Park', '湖泊与绿地'),
    'OUT-07': ('OUT-03', '九龙公园 Kowloon Park', '现代建筑与入口广场'),
    'OUT-11': ('OUT-04', '圆方 Elements',         '圆方及周边建筑'),
    'OUT-12': ('OUT-04', '圆方 Elements',         '曲面顶棚与开敞广场'),
}

# 受控场景词表，顺序即匹配优先级
SCENE_RULES = [
    ('室内空间', ('室内',)),
    ('城市远景', ('远景', '天际线', '远眺', '俯瞰', '仰视')),
    ('水体',     ('湖', '水面', '红桥', '泳池', '游泳池')),
    ('街市摊位', ('庙宇', '灯笼', '夜市', '摊档', '市集', '市食', '霓虹招牌')),
    ('道路界面', ('街道', '道路', '车流', '巴士', '高架', '路面', '街面', '窄街')),
    ('林荫步道', ('步道', '林荫', '园径', '小径', '路径', '走道', '园路')),
    ('廊架连廊', ('连廊', '柱廊', '雨棚', '顶棚', '廊', '中庭', '通道', '有盖')),
    ('坡道台阶', ('台阶', '坡道', '阶梯', '楼梯', '台階', '斜道')),
    ('建筑入口', ('入口', '门廊', '门厅', '构筑物')),
    ('施工围挡', ('施工', '围板', '围挡', '工地', '吊机', '脚手架')),
    ('建筑近景', ('近景', '墙地面', '材质')),
    ('建筑立面', ('立面', '塔楼', '幕墙', '建筑', '砖', '瓦顶', '钟楼', '旧住宅', '旧楼', '外墙', '钟塔', '校名', '红砖', '阳台', '花格', '磡楼', '标识')),
    ('绿地',     ('草坪', '草地', '绿地', '绿篱', '密林', '植被', '灌木', '棕榈', '花丛',
                  '树', '园圃', '园景', '花园', '露台', '户外座', '遮阳伞')),
    ('铺装广场', ('广场', '铺装', '硬地', '平台', '砖路', '庭院', '广场')),
]

# 关键词覆盖不到的少数几张，直接指定
OVERRIDE = {
    'IMG_0812.jpg': '街市摊位', 'IMG_0813.jpg': '街市摊位', 'IMG_0814.jpg': '街市摊位',
    'IMG_0928.jpg': '建筑入口', 'IMG_0929.jpg': '建筑入口',
    'IMG_0773.jpg': '林荫步道', 'IMG_0791.jpg': '建筑入口',
    'IMG_0978.jpg': '道路界面', 'IMG_0979.jpg': '道路界面',
}

HEADER = ['filename', 'class', 'group_id', 'location_name',
          'sub_area', 'scene_class', 'scene_type', 'status', 'notes']

# 繁体 → 简体。校内标识多为繁体，词表以简体书写，先归一再匹配，
# 免得同一栋楼因为写法不同落到「其他场景」。
_TRAD2SIMP = str.maketrans({
    '樓': '楼', '綠': '绿', '樹': '树', '廣': '广', '鋪': '铺', '連': '连',
    '欄': '栏', '櫚': '榈', '網': '网', '圍': '围', '築': '筑', '磚': '砖',
    '頂': '顶', '鐘': '钟', '車': '车', '牆': '墙', '蓋': '盖', '檯': '台', '廟': '庙', '燈': '灯', '攤': '摊',
    '徑': '径', '陰': '阴', '階': '阶', '視': '视', '遠': '远', '廳': '厅',
    '場': '场', '質': '质', '蔭': '荫', '叢': '丛', '內': '内', '區': '区',
    '東': '东', '門': '门', '園': '园', '藝': '艺', '館': '馆', '邊': '边',
    '風': '风', '電': '电', '產': '产', '業': '业', '臺': '台', '標': '标',
    '誌': '志', '識': '识', '號': '号', '設': '设', '計': '计',
    '牆': '墙', '紅': '红', '樑': '梁', '華': '华', '興': '兴', '學': '学',
    '書': '书', '衛': '卫', '儀': '仪', '廚': '厨',
})


def _norm(text):
    return text.translate(_TRAD2SIMP)


def classify(filename, scene_type):
    if filename in OVERRIDE:
        return OVERRIDE[filename]
    text = _norm(scene_type)
    for cls, keywords in SCENE_RULES:
        if any(k in text for k in keywords):
            return cls
    return '其他场景'


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
            sys.exit(f'{path}: 未登记的 group_id「{gid}」，请检查 VENUE_MAP')
        venue_id, venue_name, sub_area = VENUE_MAP[gid]
        original = r['filename'].split('_')[-1]          # 剥掉命名脚本加的前缀
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
        sys.exit(f'没有找到 manifest: {ANNOTATIONS}/*_manifest.csv')

    total = 0
    for mf in manifests:
        path = os.path.join(ANNOTATIONS, mf)
        if already_merged(path):
            print(f'{mf:28} 已收敛为场所级，跳过')
            continue
        n = convert(path)
        total += n
        print(f'{mf:28} {n} 行已收敛')

    print(f'\n共处理 {total} 行。'
          '接着运行 python3 src/tools/apply_naming.py 让 group_id 前缀与 CSV 一致。')


if __name__ == '__main__':
    main()
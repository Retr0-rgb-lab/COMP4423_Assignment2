# -*- coding: utf-8 -*-
"""Task 1 deliverable Figure 1: annotate the 7 inside location groups on the
official Campus Map aerial view.

Base image: docs/figures/campus_map_base.png
(cropped out of a screenshot of the official 3D Campus Map, 1537×1336, without
the pins.)
The pin coordinates were checked area by area against the building letters on
the map (A–Z, VA, VS, M, P, Q).
The counts are the usable images with status = include, not the training
counts after the per-group cap.
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'docs', 'figures', 'campus_map_base.png')
OUT = os.path.join(ROOT, 'docs', 'figures', 'figure1_inside_locations_annotated.png')
FONT = '/home/peleclic/.local/share/fonts/msyh.ttc'


def f(sz):
    return ImageFont.truetype(FONT, sz)


# (number, scene, x, y, usable count) -- coordinates on the 1537×1336 base image
PINS = [
    ('01', '走道（J 区 — VA 区）',     590,  860, 71),
    ('02', 'P/Q 区藻井走道',           940,  755, 62),
    ('03', '大学图书馆与中央平台',     705,  880, 25),
    ('04', 'FSCH / VS 区 / 泳池',      390,  690, 53),
    ('05', '主入口内侧草坪与钟塔',     630,  970, 38),
    ('06', '楼群庭院与 A 座入口',      255, 1145, 40),
    ('07', 'CMA 庭院',                 575, 1245, 17),
]
TOTAL = sum(p[4] for p in PINS)
LEG_W = 580

map_im = Image.open(SRC).convert('RGB')
W, H = map_im.size
canvas = Image.new('RGB', (W + LEG_W, H), '#FFFFFF')
canvas.paste(map_im, (0, 0))
d = ImageDraw.Draw(canvas, 'RGBA')

R = 21
for num, name, x, y, _ in PINS:
    d.ellipse([x - R + 3, y - R + 5, x + R + 3, y + R + 5], fill=(0, 0, 0, 60))
    d.ellipse([x - R, y - R, x + R, y + R], fill=(200, 30, 30, 235),
              outline=(255, 255, 255, 255), width=4)
    d.text((x, y), num, font=f(21), fill=(255, 255, 255, 255), anchor='mm')

lx = W
d.rectangle([lx, 0, W + LEG_W, H], fill=(255, 255, 255))
d.line([(lx, 0), (lx, H)], fill=(225, 225, 225), width=2)

y0 = 120
d.text((lx + 34, y0), 'Figure 1 — inside 采集地点组', font=f(27), fill=(15, 15, 15))
d.text((lx + 34, y0 + 42),
       f'PolyU 主校区（label = 1）· {len(PINS)} 组 · 可用 {TOTAL} 张',
       font=f(16), fill=(95, 95, 95))
d.line([(lx + 34, y0 + 76), (W + LEG_W - 34, y0 + 76)], fill=(215, 215, 215), width=2)

ry = y0 + 100
for head, xoff in (('编号', 34), ('场景类型', 84), ('张数', 480)):
    d.text((lx + xoff, ry), head, font=f(15), fill=(130, 130, 130))
ry += 26
d.line([(lx + 34, ry), (W + LEG_W - 34, ry)], fill=(232, 232, 232), width=1)
ry += 16

for num, name, _, _, n in PINS:
    d.ellipse([lx + 34, ry + 3, lx + 34 + 24, ry + 27], fill=(200, 30, 30, 235))
    d.text((lx + 46, ry + 15), num, font=f(14), fill=(255, 255, 255), anchor='mm')
    d.text((lx + 70, ry + 15), f'IN-{num}', font=f(14), fill=(150, 150, 150))
    d.text((lx + 128, ry + 15), name, font=f(15), fill=(25, 25, 25), anchor='lm')
    d.text((lx + LEG_W - 36, ry + 15), str(n), font=f(15),
           fill=(25, 25, 25), anchor='rm')
    ry += 42

ry += 8
d.line([(lx + 34, ry), (W + LEG_W - 34, ry)], fill=(232, 232, 232), width=1)
ry += 24
d.text((lx + 34, ry),
       f'合计  {len(PINS)} 组 · 可用 {TOTAL} 张',
       font=f(17), fill=(200, 30, 30))
ry += 28
d.text((lx + 34, ry), '（status = include；入库时单组上限 50）',
       font=f(13), fill=(150, 150, 150))
ry += 50

for line in [
    '一个物理场所 = 一个地点组。',
    '组内变化视点 / 距离 / 光照 / 背景，',
    '不另建组。拍摄为 scene-level 场景照，',
    '画面保留环境上下文，不拍孤立物体。',
]:
    d.text((lx + 34, ry), line, font=f(14), fill=(90, 90, 90))
    ry += 24

ry += 18
box_h = 122
d.rectangle([lx + 34, ry, W + LEG_W - 34, ry + box_h], fill=(253, 246, 238))
d.text((lx + 46, ry + 12), '按规格排除（不采集）', font=f(14), fill=(170, 70, 20))
excluded = ['校门与门槛位置　·　连接天桥与连廊',
            '校园边界位置　·　跨越边界的取景',
            '含可识别人物 / 个人信息的画面']
for i, line in enumerate(excluded):
    d.text((lx + 46, ry + 40 + i * 24), line, font=f(13), fill=(150, 95, 55))

canvas.save(OUT, optimize=True)
print('saved', OUT, canvas.size)

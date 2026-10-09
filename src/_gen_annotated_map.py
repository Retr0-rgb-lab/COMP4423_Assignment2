# -*- coding: utf-8 -*-
"""Task 1 交付物 Figure 1: 在官方 Campus Map 鸟瞰图上标注 12 个 inside 采集地点组。"""
from PIL import Image, ImageDraw, ImageFont

SRC = '/mnt/c/Users/Lenovo/Pictures/Screenshots/屏幕截图 2026-10-08 103421.png'
OUT = 'figure1_inside_locations_annotated.png'

FONT = '/home/peleclic/.local/share/fonts/msyh.ttc'
def f(sz): return ImageFont.truetype(FONT, sz)

# (编号, 场景类型, x, y)  —— 坐标由 100px 网格逐区核对得出
PINS = [
    ('01', '运动场与球场绿地区（VA / VS）',              500,  690),
    ('02', '中央开敞铺装广场',                            975,  665),
    ('03', '高层塔楼底座广场（A–G、J、L 区）',            465, 1115),
    ('04', '红砖建筑群（Z 区砖楼）',                      620,  120),
    ('05', '弧形地标楼（Y）',                            1010,  300),
    ('06', '圆柱形塔楼周边（M）',                          682,  735),
    ('07', '林荫步道（西南侧树带）',                       300,  880),
    ('08', '校内车行道与人车交汇段',                       760,  555),
    ('09', '建筑入口前庭（B / C / D 区庭院）',             330, 1195),
    ('10', '步行主轴与坡道台阶',                          605,  928),
    ('11', '开阔处远景（望向远处塔楼）',                    370,  800),
    ('12', '立面与地面交界近景（L / E 区）',               760, 1080),
]
PER_GROUP = 35
LEG_W = 580

map_im = Image.open(SRC).convert('RGB')
W, H = map_im.size
canvas = Image.new('RGB', (W + LEG_W, H), '#FFFFFF')
canvas.paste(map_im, (0, 0))
d = ImageDraw.Draw(canvas, 'RGBA')

# ---- 图钉 ----
R = 21
for num, name, x, y in PINS:
    d.ellipse([x-R+3, y-R+5, x+R+3, y+R+5], fill=(0, 0, 0, 60))   # 阴影
    d.ellipse([x-R, y-R, x+R, y+R], fill=(200, 30, 30, 235),
              outline=(255, 255, 255, 255), width=4)
    d.text((x, y), num, font=f(21), fill=(255, 255, 255, 255), anchor='mm')

# ---- 图例 ----
lx = W
d.rectangle([lx, 0, W + LEG_W, H], fill=(255, 255, 255))
d.line([(lx, 0), (lx, H)], fill=(225, 225, 225), width=2)

y0 = 120
d.text((lx + 34, y0), 'Figure 1 — inside 采集地点组', font=f(27), fill=(15, 15, 15))
d.text((lx + 34, y0 + 42), 'PolyU 主校区（label = 1）· 12 组 · 每组 35 张', font=f(16),
       fill=(95, 95, 95))
d.line([(lx + 34, y0 + 76), (W + LEG_W - 34, y0 + 76)], fill=(215, 215, 215), width=2)

ry = y0 + 100
for head, xoff in (('编号', 34), ('场景类型', 84), ('张数', 480)):
    d.text((lx + xoff, ry), head, font=f(15), fill=(130, 130, 130))
ry += 26
d.line([(lx + 34, ry), (W + LEG_W - 34, ry)], fill=(232, 232, 232), width=1)
ry += 12

for num, name, _, _ in PINS:
    d.ellipse([lx + 34, ry + 3, lx + 34 + 24, ry + 27], fill=(200, 30, 30, 235))
    d.text((lx + 46, ry + 15), num, font=f(14), fill=(255, 255, 255), anchor='mm')
    d.text((lx + 70, ry + 15), f'IN-{num}', font=f(14), fill=(150, 150, 150))
    d.text((lx + 128, ry + 15), name, font=f(15), fill=(25, 25, 25), anchor='lm')
    d.text((lx + LEG_W - 36, ry + 15), str(PER_GROUP), font=f(15),
           fill=(25, 25, 25), anchor='rm')
    ry += 34

ry += 4
d.line([(lx + 34, ry), (W + LEG_W - 34, ry)], fill=(232, 232, 232), width=1)
ry += 20
d.text((lx + 34, ry), f'合计  {len(PINS)} 组 × {PER_GROUP} 张 = {len(PINS)*PER_GROUP} 张',
       font=f(17), fill=(200, 30, 30))
ry += 46

for line in [
    '拍摄对象为 scene-level 场景照，',
    '每组内变化视点 / 距离 / 光照 / 背景，',
    '画面保留环境上下文，不拍孤立物体。',
]:
    d.text((lx + 34, ry), line, font=f(14), fill=(90, 90, 90))
    ry += 22

ry += 14
d.rectangle([lx + 34, ry, W + LEG_W - 34, ry + 122], fill=(253, 246, 238))
d.text((lx + 46, ry + 12), '按规格排除（不采集）', font=f(14), fill=(170, 70, 20))
excluded = ['校门与门槛位置　·　连接天桥与连廊',
            '校园边界位置　·　跨越边界的取景',
            '含可识别人物 / 个人信息的画面']
for i, line in enumerate(excluded):
    d.text((lx + 46, ry + 40 + i * 24), line, font=f(13), fill=(150, 95, 55))

canvas.save(OUT, optimize=True)
print('saved', OUT, canvas.size)

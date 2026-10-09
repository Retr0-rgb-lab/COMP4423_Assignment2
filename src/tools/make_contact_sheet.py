# -*- coding: utf-8 -*-
"""Contact sheet 生成器 —— 用于人工核对照片的地点组归属。

把一组照片铺成带序号的网格缩略图，肉眼扫一遍即可判断：
  1. 这一批是不是同一个物理场景（决定 group_id 的切分）
  2. 组内是否混入了边界位置、可识别人物等应当剔除的样本

纯人工辅助工具，不参与训练、不进入 pipeline。

用法
----
  # 整目录
  python src/tools/make_contact_sheet.py <图片目录> <输出图片> [每行列数]

  # 只画清单里的文件（清单每行一个文件名，可用 # 注释）
  python src/tools/make_contact_sheet.py <图片目录> <输出图片> 5 --list files.txt
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

CELL_W, CELL_H = 460, 330
PAD = 8
LABEL_H = 34
HEADER_H = 46

IMG_EXT = ('.jpg', '.jpeg', '.png', '.heic', '.heif')
FONT_PATH = '/home/peleclic/.local/share/fonts/msyh.ttc'


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def read_list(path):
    """读取文件清单，返回存在的文件名（保持清单顺序，便于对照拍摄顺序）。"""
    names = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                names.append(line)
    return names


def collect(src_dir, list_file):
    if list_file:
        names = read_list(list_file)
        return [n for n in names
                if os.path.isfile(os.path.join(src_dir, n))]
    return sorted(f for f in os.listdir(src_dir)
                  if f.lower().endswith(IMG_EXT))


def build(src_dir, files, cols, out_path, title):
    rows = (len(files) + cols - 1) // cols if files else 1
    width = cols * (CELL_W + PAD) + PAD
    height = HEADER_H + PAD + rows * (CELL_H + LABEL_H + PAD) + PAD

    canvas = Image.new('RGB', (width, height), '#FFFFFF')
    draw = ImageDraw.Draw(canvas)
    f_head, f_label = load_font(26), load_font(17)

    draw.text((PAD + 4, 10), title, font=f_head, fill='#111111')
    draw.line([(PAD, 44), (width - PAD, 44)], fill='#DDDDDD', width=2)

    errors = []
    for i, name in enumerate(files):
        row, col = divmod(i, cols)
        x = PAD + col * (CELL_W + PAD)
        y = HEADER_H + PAD + row * (CELL_H + LABEL_H + PAD)
        stem = os.path.splitext(name)[0]

        try:
            im = Image.open(os.path.join(src_dir, name)).convert('RGB')
            im.thumbnail((CELL_W, CELL_H), Image.LANCZOS)
            canvas.paste(im, (x + (CELL_W - im.width) // 2,
                               y + (CELL_H - im.height) // 2))
        except Exception as exc:                                  # noqa: BLE001
            draw.rectangle([x, y, x + CELL_W, y + CELL_H], fill='#F3F3F3')
            draw.text((x + 10, y + 10), f'ERR {exc}'[:60],
                      font=f_label, fill='#AA0000')
            errors.append((name, str(exc)))

        draw.rectangle([x, y, x + CELL_W, y + CELL_H],
                       outline='#BBBBBB', width=1)
        draw.rectangle([x, y, x + 88, y + 26], fill='#C81E1E')
        draw.text((x + 6, y + 3), stem.replace('IMG_', '#'),
                  font=f_label, fill='#FFFFFF')
        draw.text((x + 2, y + CELL_H + 5), name, font=f_label, fill='#555555')

    canvas.save(out_path, quality=92)
    return errors


def main():
    ap = argparse.ArgumentParser(description='生成 contact sheet')
    ap.add_argument('src_dir')
    ap.add_argument('out_path')
    ap.add_argument('cols', nargs='?', type=int, default=5)
    ap.add_argument('--list', dest='list_file',
                    help='文件清单路径，每行一个文件名')
    ap.add_argument('--title', default='')
    args = ap.parse_args()

    files = collect(args.src_dir, args.list_file)
    if not files:
        raise SystemExit(f'没有可用的图片: {args.src_dir}')

    title = args.title or (f'{os.path.basename(args.src_dir)} — '
                           f'{len(files)} 张')
    errors = build(args.src_dir, files, args.cols, args.out_path, title)

    print(f'saved {args.out_path} — {len(files)} 张')
    for name, err in errors:
        print(f'  [读取失败] {name}: {err}')


if __name__ == '__main__':
    main()
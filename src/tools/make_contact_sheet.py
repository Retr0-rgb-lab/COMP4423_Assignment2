# -*- coding: utf-8 -*-
"""Contact sheet generator -- used to check by hand which location group a
photo belongs to.

It lays a set of photos out as a numbered grid of thumbnails, so that one
glance answers:
  1. whether this batch is one and the same physical scene (which decides
     the group_id split)
  2. whether boundary positions, identifiable people and other samples that
     should be dropped have slipped into the group

A purely manual aid: it takes no part in training and never enters the
pipeline.

Usage
-----
  # whole directory
  python src/tools/make_contact_sheet.py <image dir> <output image> [columns]

  # only the files named in a list file (one file name per line, # comments
  # allowed)
  python src/tools/make_contact_sheet.py <image dir> <output image> 5 --list files.txt
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
    """Return a font of the given size, falling back to the default bitmap font.

    The bundled CJK font is used whenever it can be loaded, so that labels
    containing Chinese characters are rendered instead of dropped.
    """
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def read_list(path):
    """Read a list file and return the file names that exist (in list order,
    so the shooting order stays comparable)."""
    names = []
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith('#'):
                names.append(line)
    return names


def collect(src_dir, list_file):
    """Return the file names to place on the contact sheet.

    With a list file the names are taken from it in the given order and the
    ones that do not exist are dropped; without one every image of the
    directory is taken in sorted order.
    """
    if list_file:
        names = read_list(list_file)
        return [n for n in names
                if os.path.isfile(os.path.join(src_dir, n))]
    return sorted(f for f in os.listdir(src_dir)
                  if f.lower().endswith(IMG_EXT))


def build(src_dir, files, cols, out_path, title):
    """Draw the numbered grid of thumbnails and save it as an image file.

    Returns the list of (file name, error) pairs of the photographs that
    could not be read; the file is still written, with a grey cell and the
    error message in place of the missing thumbnail.
    """
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
    """Generate a contact sheet of the photographs in a directory.

    Writes the image to the given path and prints the number of images on
    the sheet together with the files that could not be read. Exits when the
    directory holds no usable image.
    """
    ap = argparse.ArgumentParser(description='generate a contact sheet')
    ap.add_argument('src_dir')
    ap.add_argument('out_path')
    ap.add_argument('cols', nargs='?', type=int, default=5)
    ap.add_argument('--list', dest='list_file',
                    help='path to a list file with one file name per line')
    ap.add_argument('--title', default='')
    args = ap.parse_args()

    files = collect(args.src_dir, args.list_file)
    if not files:
        raise SystemExit(f'no usable images: {args.src_dir}')

    title = args.title or (f'{os.path.basename(args.src_dir)} — '
                           f'{len(files)} images')
    errors = build(args.src_dir, files, args.cols, args.out_path, title)

    print(f'saved {args.out_path} — {len(files)} images')
    for name, err in errors:
        print(f'  [read failed] {name}: {err}')


if __name__ == '__main__':
    main()
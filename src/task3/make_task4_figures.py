# -*- coding: utf-8 -*-
"""Draw the confusion matrix and the example figure from the existing test
predictions. No training happens here."""
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

from src.task3.config import PICTURES, ROOT

PRED = ROOT / "outputs" / "task3" / "test_predictions.csv"
OUT_DIR = ROOT / "docs" / "figures"

EXAMPLES = [
    ("OUT-06_001_IMG_0863.jpg", "OUT-06 判对\n校外红砖立面"),
    ("OUT-06_012_IMG_0892.jpg", "OUT-06 判错\n校外台阶入口 → 校内"),
    ("OUT-07_001_IMG_0934.jpg", "OUT-07 判对\n港大绿地"),
    ("IN-03_016_IMG_1459.jpg", "IN-03 判对\n图书馆入口近景"),
    ("IN-03_001_IMG_1444.jpg", "IN-03 判错\n砖墙近景 → 校外"),
    ("IN-03_005_IMG_1448.jpg", "IN-03 判错\n庭院入口 → 校外"),
    ("IN-07_002_IMG_1428.jpg", "IN-07 判对\nCMA 入口近景"),
    ("IN-07_008_IMG_1434.jpg", "IN-07 判错\n同一庭院拉远 → 校外"),
]


def _font():
    candidates = [
        "/home/peleclic/.local/share/fonts/msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            font_manager.fontManager.addfont(path)
            return font_manager.FontProperties(fname=path).get_name()
    return None


def _index():
    found = {}
    for dirpath, _dirs, names in os.walk(PICTURES):
        for name in names:
            found[name] = os.path.join(dirpath, name)
    return found


def _load_pred():
    with open(PRED, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def confusion_figure(rows):
    matrix = np.zeros((2, 2), dtype=int)
    for row in rows:
        matrix[int(row["y_true"]), int(row["y_pred"])] += 1
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["预测校外", "预测校内"])
    ax.set_yticks([0, 1], ["实际校外", "实际校内"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=16)
    ax.set_title("测试集混淆矩阵  n=69")
    fig.tight_layout()
    path = OUT_DIR / "task4_confusion.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, matrix


def examples_figure(paths):
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 7.2))
    for ax, (name, caption) in zip(axes.ravel(), EXAMPLES):
        image = Image.open(paths[name]).convert("RGB")
        ax.imshow(image)
        ax.set_title(caption, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path = OUT_DIR / "task4_examples.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    name = _font()
    if name:
        plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    rows = _load_pred()
    paths = _index()
    missing = [name for name, _cap in EXAMPLES if name not in paths]
    if missing:
        raise SystemExit(f"例图找不到: {missing}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conf_path, matrix = confusion_figure(rows)
    ex_path = examples_figure(paths)
    print(matrix.tolist())
    print(conf_path)
    print(ex_path)


if __name__ == "__main__":
    main()

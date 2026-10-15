# -*- coding: utf-8 -*-
"""第一段：线性 SVM × 六种特征。"""
import argparse
import sys

from src.task3.config import FEATURE_ORDER, OUTPUT_DIR
from src.task3.cv import choose_features, evaluate_config, write_rows
from src.task3.data import load_index
from src.task3.features import assert_dims, sift_cached
from src.task3.models import make_pipeline
from src.task3.preprocess import load_rgb, resize_and_crop


def run_check(records):
    sample = resize_and_crop(load_rgb(records[0]["path"]))
    dims = assert_dims(sample)
    _pts, desc = sift_cached(sample)
    print(f"自检通过。样例 {records[0]['filename']} 维度 {dims}，SIFT 关键点 {len(desc)}")
    both = [r for r in records if r["split"] == "train"]
    inside = next(r for r in both if r["y"] == 1)
    outside = next(r for r in both if r["y"] == 0)
    images, labels = [], []
    for rec in (inside, outside):
        image = resize_and_crop(load_rgb(rec["path"]))
        images.extend([image, image])
        labels.extend([rec["y"], rec["y"]])
    pipe = make_pipeline("F3", "C1")
    pipe.fit(images, labels)
    pred = pipe.predict(images[:2])
    print(f"F3×C1 冒烟预测 {list(pred)}")


def _utf8():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    _utf8()
    parser = argparse.ArgumentParser(description="Task 3 第一段")
    parser.add_argument("--check", action="store_true", help="只检查数据和特征维度")
    parser.add_argument("--force", action="store_true", help="重算已有结果")
    args = parser.parse_args()

    records = load_index()
    print(f"划分表 {len(records)} 张", flush=True)
    if args.check:
        run_check(records)
        return

    out = OUTPUT_DIR / "stage_a.csv"
    done = set()
    rows = []
    if out.is_file() and not args.force:
        from src.task3.cv import read_rows
        rows = [r for r in read_rows(out) if r["clf_id"] == "C1"]
        done = {r["feature_id"] for r in rows}
        print(f"已有 {sorted(done)}，跳过", flush=True)

    for feature_id in FEATURE_ORDER:
        if feature_id in done:
            continue
        rows.append(evaluate_config(records, feature_id, "C1"))
        write_rows(out, rows)

    ranked = choose_features(rows, k=2)
    print("\n进入第二段的特征:",
          ", ".join(f"{r['feature_id']}({r['ba_mean']:.3f})" for r in ranked),
          flush=True)
    if len(rows) != len(FEATURE_ORDER):
        sys.exit(f"第一段行数是 {len(rows)}，期望 {len(FEATURE_ORDER)}")


if __name__ == "__main__":
    main()

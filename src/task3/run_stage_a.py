# -*- coding: utf-8 -*-
"""Stage one: linear SVM x six feature sets."""
import argparse
import sys

from src.task3.config import FEATURE_ORDER, OUTPUT_DIR
from src.task3.cv import (
    assert_resume_allowed,
    choose_features,
    evaluate_config,
    write_rows,
    write_run_stamp,
)
from src.task3.data import load_index
from src.task3.features import assert_dims, sift_cached
from src.task3.models import make_pipeline
from src.task3.preprocess import load_rgb, resize_and_crop


def run_check(records):
    """Check the data and the feature dimensions without running the search.

    The first record is cropped to confirm the feature dimensions and the SIFT
    keypoints, then one inside and one outside image are fitted with F3 x C1
    to give a smoke prediction.
    """
    sample = resize_and_crop(load_rgb(records[0]["path"]))
    dims = assert_dims(sample)
    _pts, desc = sift_cached(sample)
    print(f"self-check passed. Sample {records[0]['filename']} dims {dims}, SIFT keypoints {len(desc)}")
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
    print(f"F3×C1 smoke prediction {list(pred)}")


def _utf8():
    """Switch the standard streams to UTF-8 so that non-ASCII text prints safely."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    """Run stage one: the linear SVM over the six feature sets.

    Configurations that are already in `stage_a.csv` are skipped unless
    `--force` is given. The two best feature sets are printed for stage two
    and the run stamp is written at the end.
    """
    _utf8()
    parser = argparse.ArgumentParser(description="Task 3 stage one")
    parser.add_argument("--check", action="store_true", help="only check the data and the feature dimensions")
    parser.add_argument("--force", action="store_true", help="recompute the existing results")
    args = parser.parse_args()

    records = load_index()
    print(f"split table {len(records)} images", flush=True)
    if args.check:
        run_check(records)
        return

    assert_resume_allowed(args.force)
    out = OUTPUT_DIR / "stage_a.csv"
    done = set()
    rows = []
    if out.is_file() and not args.force:
        from src.task3.cv import read_rows
        rows = [r for r in read_rows(out) if r["clf_id"] == "C1"]
        done = {r["feature_id"] for r in rows}
        print(f"already have {sorted(done)}, skipping", flush=True)

    for feature_id in FEATURE_ORDER:
        if feature_id in done:
            continue
        rows.append(evaluate_config(records, feature_id, "C1"))
        write_rows(out, rows)

    ranked = choose_features(rows, k=2)
    print("\nfeatures that go into stage two:",
          ", ".join(f"{r['feature_id']}({r['ba_mean']:.3f})" for r in ranked),
          flush=True)
    if len(rows) != len(FEATURE_ORDER):
        sys.exit(f"stage one has {len(rows)} rows, expected {len(FEATURE_ORDER)}")
    write_run_stamp()


if __name__ == "__main__":
    main()

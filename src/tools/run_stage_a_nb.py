# -*- coding: utf-8 -*-
"""Stage one, supplementary: the naive Bayes classifier on the histogram features.

Stage one ranks the six feature sets under one fixed linear SVM, so that the
feature axis is isolated. The naive Bayes classifier only accepts
non-negative L1 histograms, which is F1, F3, F5 and F6; the two features that
advance to stage two, F2 and F4, are gradient and texture descriptors and are
not histograms, so C6 never reaches the stage two comparison.

This script runs C6 on the four features it does apply to, as a supplementary
comparison. Its rows go to outputs/task3/stage_a_nb.csv and are NOT part of the
feature ranking, which is computed from the C1 rows of stage_a.csv alone. The
locked pipeline is therefore untouched by this run.

It lives under src/tools rather than src/task3 on purpose: the run fingerprint
in src/task3/cv.py hashes every *.py in src/task3, so a supplementary script
placed there would make a resume refuse for no reason.

The test set is not touched: every fold is one of the five location-holdout
folds over the training images.
"""
import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.task3.config import HIST_FEATURES, OUTPUT_DIR  # noqa: E402
from src.task3.cv import evaluate_config, write_rows  # noqa: E402
from src.task3.data import load_index  # noqa: E402

OUT = OUTPUT_DIR / "stage_a_nb.csv"


def _utf8():
    """Switch stdout and stderr to UTF-8 so that CJK output is not lost."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def main():
    """Evaluate the naive Bayes classifier on the four histogram features.

    The rows are written to outputs/task3/stage_a_nb.csv; the rows that are
    already in the table are kept unless `--force` is given, and the file is
    rewritten after every new row. Only the training and the validation
    split are read, the test set is never touched.
    """
    _utf8()
    parser = argparse.ArgumentParser(
        description="Evaluate the naive Bayes classifier on the histogram features")
    parser.add_argument("--force", action="store_true",
                        help="recompute the rows that are already in the table")
    args = parser.parse_args()

    records = load_index()
    print(f"split table {len(records)} images", flush=True)

    targets = [f for f in ("F1", "F3", "F5", "F6") if f in HIST_FEATURES]
    print(f"naive Bayes applies to {targets}", flush=True)

    rows = []
    done = set()
    if OUT.is_file() and not args.force:
        from src.task3.cv import read_rows
        rows = [r for r in read_rows(OUT) if r["clf_id"] == "C6"]
        done = {r["feature_id"] for r in rows}
        print(f"already have {sorted(done)}, skipping", flush=True)

    for feature_id in targets:
        if feature_id in done:
            continue
        rows.append(evaluate_config(records, feature_id, "C6"))
        write_rows(OUT, rows)

    for row in rows:
        print(f"  {row['feature_id']} C6  BA={row['ba_mean']:.3f}"
              f"  inside={row['inside_recall_mean']:.3f}"
              f"  outside={row['outside_recall_mean']:.3f}"
              f"  val={row['val_accuracy']:.3f}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

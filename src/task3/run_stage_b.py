# -*- coding: utf-8 -*-
"""Stage two: the two winning feature sets x the classifiers. If the linear SVM
wins, one extra RBF row is added."""
import argparse
import json

from src.task3.config import CLF_ORDER, HIST_FEATURES, OUTPUT_DIR
from src.task3.cv import (
    assert_resume_allowed,
    choose_features,
    choose_pipeline,
    evaluate_config,
    read_rows,
    write_rows,
    write_run_stamp,
)
from src.task3.data import load_index


def main():
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Task 3 stage two")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stage_a = OUTPUT_DIR / "stage_a.csv"
    if not stage_a.is_file():
        raise SystemExit("run python -m src.task3.run_stage_a first")
    top = choose_features(read_rows(stage_a), k=2)
    feature_ids = [row["feature_id"] for row in top]
    print("stage two features:", ", ".join(
        f"{row['feature_id']} BA={row['ba_mean']:.3f}" for row in top), flush=True)

    assert_resume_allowed(args.force)
    records = load_index()
    out = OUTPUT_DIR / "stage_b.csv"
    rows = []
    done = set()
    if out.is_file() and not args.force:
        rows = read_rows(out)
        done = {(r["feature_id"], r["clf_id"]) for r in rows if r["role"] == "candidate"}

    for feature_id in feature_ids:
        for clf_id in CLF_ORDER:
            if clf_id == "C6" and feature_id not in HIST_FEATURES:
                continue
            if (feature_id, clf_id) in done:
                continue
            rows.append(evaluate_config(records, feature_id, clf_id))
            write_rows(out, rows)

    winner = choose_pipeline(rows)
    if winner["clf_id"] == "C1" and (winner["feature_id"], "C1-RBF") not in {
        (r["feature_id"], r["clf_id"]) for r in rows
    }:
        control = evaluate_config(records, winner["feature_id"], "C1-RBF")
        control["role"] = "rbf_control"
        rows.append(control)

    for row in rows:
        row["role"] = "selected" if (
            row["feature_id"] == winner["feature_id"]
            and row["clf_id"] == winner["clf_id"]
        ) else row.get("role", "candidate")
        if row["clf_id"] == "C1-RBF":
            row["role"] = "rbf_control"
    write_rows(out, rows)

    selection = {
        "feature_id": winner["feature_id"],
        "clf_id": winner["clf_id"],
        "ba_mean": winner["ba_mean"],
        "val_accuracy": winner["val_accuracy"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    write_run_stamp()
    print(f"\nlocked {winner['feature_id']} × {winner['clf_id']} "
          f"BA={winner['ba_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()

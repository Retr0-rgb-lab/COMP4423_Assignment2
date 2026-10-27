# -*- coding: utf-8 -*-
"""Refit the locked pipeline on the full training set, report the validation
split, and evaluate the test split exactly once."""
import csv
import json

import joblib
import numpy as np

from src.task3.config import DISPLAY, MODEL_PATH, OUTPUT_DIR
from src.task3.cv import augment, classification_report, load_cropped, read_rows, choose_pipeline
from src.task3.data import by_split, load_index
from src.task3.models import make_pipeline


def _fit_train(records, feature_id, clf_id):
    """Fit the given configuration on the training split.

    Returns the fitted pipeline and the records grouped by split. The pipeline
    carries the display names of the two classes, so the saved model can name
    them without reading the folder layout.
    """
    splits = by_split(records)
    images = load_cropped(splits["train"])
    labels = [rec["y"] for rec in splits["train"]]
    x_train, y_train = augment(images, labels)
    pipe = make_pipeline(feature_id, clf_id)
    pipe.fit(x_train, y_train)
    pipe.display_ = DISPLAY
    return pipe, splits


def main():
    """Refit the locked configuration, save the model and score both splits.

    The configuration comes from `selection.json`, or from the winner of
    `stage_b.csv` when that file is missing. The model is written to
    `models/task3_pipeline.joblib`, the metrics of a split to
    `<split>_metrics.json` and the test predictions to `test_predictions.csv`.
    """
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    selection_path = OUTPUT_DIR / "selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        winner = choose_pipeline(read_rows(OUTPUT_DIR / "stage_b.csv"))
        selection = {"feature_id": winner["feature_id"], "clf_id": winner["clf_id"]}
    feature_id, clf_id = selection["feature_id"], selection["clf_id"]
    print(f"refitting {feature_id} × {clf_id}", flush=True)

    records = load_index()
    pipe, splits = _fit_train(records, feature_id, clf_id)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"model {MODEL_PATH}", flush=True)

    for split_name in ("val", "test"):
        subset = splits[split_name]
        images = load_cropped(subset)
        y_true = np.asarray([rec["y"] for rec in subset], dtype=np.int32)
        y_pred = pipe.predict(images)
        report = classification_report(y_true, y_pred)
        report["feature_id"] = feature_id
        report["clf_id"] = clf_id
        report["split"] = split_name
        path = OUTPUT_DIR / f"{split_name}_metrics.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{split_name} accuracy={report['accuracy']:.3f} "
              f"BA={report['balanced_accuracy']:.3f}", flush=True)
        if split_name == "test":
            pred_path = OUTPUT_DIR / "test_predictions.csv"
            with open(pred_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh, fieldnames=["filename", "group_id", "y_true", "y_pred"],
                )
                writer.writeheader()
                for rec, pred in zip(subset, y_pred):
                    writer.writerow({
                        "filename": rec["filename"],
                        "group_id": rec["group_id"],
                        "y_true": rec["y"],
                        "y_pred": int(pred),
                    })


if __name__ == "__main__":
    main()

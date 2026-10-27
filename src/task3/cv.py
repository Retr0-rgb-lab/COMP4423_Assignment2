# -*- coding: utf-8 -*-
"""5-fold venue holdout (location-group holdout). The selection score is the
mean balanced accuracy over the folds."""
import csv
import hashlib
import json
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
)

from src.task3.config import (
    AUG_HFLIP,
    CLF_NAME,
    CLF_SIMPLICITY,
    FEATURE_DIM,
    FEATURE_NAME,
    FEATURE_ORDER,
    FOLDS,
    IMAGE_SIZE,
    OUTPUT_DIR,
    ROOT,
    SPLITS_CSV,
)
from src.task3.data import by_split, partition_fold
from src.task3.models import make_pipeline
from src.task3.preprocess import horizontal_flip, load_rgb, resize_and_crop


def preprocess_tag():
    """Return the tag that names the current preprocessing, e.g. "s256".

    The tag is part of every cached crop file name and of the run stamp, so a
    changed image size cannot be mistaken for an old crop.
    """
    return f"s{IMAGE_SIZE}"


def source_fingerprint():
    """Hash of splits.csv and the src/task3 source code. A resumed run refuses
    to reuse old results when the hash does not match."""
    digest = hashlib.sha256()
    digest.update(SPLITS_CSV.read_bytes())
    for path in sorted((ROOT / "src" / "task3").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(preprocess_tag().encode())
    digest.update(str(AUG_HFLIP).encode())
    return digest.hexdigest()


def assert_resume_allowed(force):
    """Refuse to reuse stored results whose fingerprint no longer matches.

    Returns the current source fingerprint. Raises SystemExit when result
    tables exist but `run_stamp.json` is missing or records another
    fingerprint; `force` skips that check, and so does the absence of
    result tables.
    """
    stamp_path = OUTPUT_DIR / "run_stamp.json"
    current = source_fingerprint()
    has_table = (OUTPUT_DIR / "stage_a.csv").is_file() or (OUTPUT_DIR / "stage_b.csv").is_file()
    if force or not has_table:
        return current
    if not stamp_path.is_file():
        raise SystemExit(
            "Result tables exist but run_stamp.json is missing, refusing to resume. Add --force once you have confirmed."
        )
    saved = json.loads(stamp_path.read_text(encoding="utf-8"))["fingerprint"]
    if saved != current:
        raise SystemExit(
            "splits.csv or the src/task3 code differs from the existing results, refusing to resume. Add --force once you have confirmed."
        )
    return current


def write_run_stamp():
    """Write the current fingerprint and preprocessing tag to `run_stamp.json`.

    The stamp is what a later resumed run compares its own fingerprint with.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": source_fingerprint(),
        "preprocess": preprocess_tag(),
    }
    (OUTPUT_DIR / "run_stamp.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )


def load_cropped(records):
    """Return the cropped RGB array of every record, using the on-disk cache.

    A crop is cached as `<preprocess tag>__<file name>.npy` below
    `cache/task3/crop` and is recomputed when the file is missing or does not
    have the expected 256x256x3 shape. Returns a list of uint8 arrays in the
    order of `records`.
    """
    from src.task3.config import CACHE
    folder = CACHE / "crop"
    folder.mkdir(parents=True, exist_ok=True)
    images = []
    for i, rec in enumerate(records, start=1):
        path = folder / f"{preprocess_tag()}__{rec['filename']}.npy"
        image = np.load(path) if path.is_file() else None
        if image is None or image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_and_crop(load_rgb(rec["path"]))
            np.save(path, image)
        images.append(image)
        if i % 100 == 0 or i == len(records):
            print(f"  reading images {i}/{len(records)}", flush=True)
    return images


def augment(images, labels):
    """Return the training images and labels, doubled by a horizontal flip.

    The flip is skipped when `AUG_HFLIP` is false. Returns a list of images
    and an int32 label array of the same length.
    """
    if not AUG_HFLIP:
        return list(images), np.asarray(labels, dtype=np.int32)
    out_x, out_y = [], []
    for image, label in zip(images, labels):
        out_x.append(image)
        out_y.append(label)
        out_x.append(horizontal_flip(image))
        out_y.append(label)
    return out_x, np.asarray(out_y, dtype=np.int32)


def _labels(records):
    """Return the int32 label array of the given records."""
    return np.asarray([rec["y"] for rec in records], dtype=np.int32)


def _recalls(y_true, y_pred):
    """Return the (outside recall, inside recall) pair for the labels 0 and 1."""
    rec = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return float(rec[0]), float(rec[1])


def evaluate_config(records, feature_id, clf_id, with_val=True):
    """Return the fold scores of one configuration and an optional validation
    accuracy."""
    print(f"\n== {feature_id} {FEATURE_NAME[feature_id]} × {clf_id} {CLF_NAME[clf_id]} ==",
          flush=True)
    splits = by_split(records)
    train_all = splits["train"]
    cropped_all = {rec["filename"]: image
                   for rec, image in zip(train_all, load_cropped(train_all))}
    fold_ba, fold_out, fold_in = [], [], []
    for index, fold in enumerate(FOLDS):
        t0 = time.perf_counter()
        train_recs, held_recs = partition_fold(records, fold)
        train_images = [cropped_all[rec["filename"]] for rec in train_recs]
        held_images = [cropped_all[rec["filename"]] for rec in held_recs]
        x_train, y_train = augment(train_images, _labels(train_recs))
        pipe = make_pipeline(feature_id, clf_id)
        pipe.fit(x_train, y_train)
        pred = pipe.predict(held_images)
        y_held = _labels(held_recs)
        ba = float(balanced_accuracy_score(y_held, pred))
        rec_out, rec_in = _recalls(y_held, pred)
        fold_ba.append(ba)
        fold_out.append(rec_out)
        fold_in.append(rec_in)
        held_names = ",".join(fold["inside"] + fold["outside"])
        print(f"  fold{index} [{held_names}] BA={ba:.3f} "
              f"out={rec_out:.3f} in={rec_in:.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    row = {
        "feature_id": feature_id,
        "clf_id": clf_id,
        "name": FEATURE_NAME[feature_id],
        "clf_name": CLF_NAME[clf_id],
        "dim": FEATURE_DIM[feature_id],
        "ba_mean": float(np.mean(fold_ba)),
        "inside_recall_mean": float(np.mean(fold_in)),
        "outside_recall_mean": float(np.mean(fold_out)),
        "role": "candidate",
    }
    for i, ba in enumerate(fold_ba):
        row[f"ba_{i}"] = ba

    if with_val:
        t0 = time.perf_counter()
        x_train, y_train = augment(
            [cropped_all[rec["filename"]] for rec in train_all],
            _labels(train_all),
        )
        pipe = make_pipeline(feature_id, clf_id)
        pipe.fit(x_train, y_train)
        val_images = load_cropped(splits["val"])
        val_pred = pipe.predict(val_images)
        row["val_accuracy"] = float(accuracy_score(_labels(splits["val"]), val_pred))
        print(f"  validation accuracy {row['val_accuracy']:.3f} "
              f"(not used for selection, {time.perf_counter() - t0:.0f}s)", flush=True)
    return row


def choose_features(rows, k=2):
    """Stage one: BA descending, dimension ascending, id ascending."""
    order = {fid: i for i, fid in enumerate(FEATURE_ORDER)}

    def key(row):
        """Return the sort key: score descending, dimension ascending, id order."""
        return (-row["ba_mean"], row["dim"], order[row["feature_id"]])

    return sorted(rows, key=key)[:k]


def choose_pipeline(rows):
    """Stage two: choose only among the role=candidate rows; ties are broken by
    simplicity."""
    candidates = [
        row for row in rows
        if row.get("role", "candidate") in ("candidate", "selected")
    ]
    simplicity = {cid: i for i, cid in enumerate(CLF_SIMPLICITY)}

    def key(row):
        """Return the sort key: score descending, then simplicity."""
        return (-row["ba_mean"], simplicity.get(row["clf_id"], 99))

    return sorted(candidates, key=key)[0]


FIELDNAMES = (
    ["feature_id", "clf_id", "name", "clf_name", "dim", "ba_mean",
     "inside_recall_mean", "outside_recall_mean", "val_accuracy", "role"]
    + [f"ba_{i}" for i in range(len(FOLDS))]
)


def write_rows(path, rows):
    """Write the rows to `path` as CSV, one row per configuration.

    The columns are fixed by `FIELDNAMES`; keys outside it are dropped.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_rows(path):
    """Read the CSV table at `path` and cast the numeric columns to int or float."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = []
        for row in csv.DictReader(fh):
            for key in ("dim",):
                row[key] = int(row[key])
            for key in ("ba_mean", "inside_recall_mean", "outside_recall_mean",
                        "val_accuracy"):
                row[key] = float(row[key])
            for i in range(len(FOLDS)):
                row[f"ba_{i}"] = float(row[f"ba_{i}"])
            rows.append(row)
        return rows


def classification_report(y_true, y_pred):
    """Return the accuracy, the balanced accuracy and the per-class scores.

    The confusion matrix is a 2x2 list ordered as outside (0) and inside (1);
    every per-class entry carries precision, recall, f1 and support.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "outside": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "inside": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }

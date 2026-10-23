# COMP4423 Assignment 2 -- inside / outside the PolyU main campus

Classifies a scene photograph as inside the PolyU main campus (inside = 1) or
outside it (outside = 0), from the pixels alone.

The repository is at https://github.com/Retr0-rgb-lab/COMP4423_Assignment2

The submitted report is `Assignment2_Report_24101322D_HUANG_Haoran.pdf` at the
repository root; it documents the data, the features, the classifiers, the
measured results and the known limitations. The report cannot be rebuilt from
this repository: its source draft is a process record that is not part of the
submitted payload.

## Environment

- Python 3.12.
- Install the dependencies:

```
pip install -r requirements.txt
```

- `opencv-contrib-python` is mandatory. The non-contrib wheel has no SIFT, and
  the feature module stops rather than substituting another descriptor.

## Layout

```
src/task3/           training pipeline: features, classifiers, venue-holdout CV, evaluation
src/task5/           the desktop application that classifies one photograph
src/tools/           dataset construction and the dataset check tools
models/              the locked pipeline (HOG+LBP with a random forest)
data/annotations/    the split table, the per-class counts and the source manifests
outputs/task3/       the raw result tables behind every number in the report
docs/figures/        the report figures and the script that draws them
```

The photographs themselves are not in this repository; they are on Google
Drive, see `DATASET_LINK.txt`.

## Run

From the repository root:

```
# Task 3 -- stage one, stage two, and the single test evaluation
python -m src.task3.run_stage_a --force
python -m src.task3.run_stage_b --force
python -m src.task3.evaluate

# Task 5 -- the local application
python -m src.task5.app
python -m src.task5.app --image path\to\photo.jpg

# the report figures
python docs/figures/plot_task4_locked.py
```

The two training stages resume instead of recomputing. The result tables in
`outputs/task3/` were written by this revision of `src/task3/`: `run_stage_a`
and `run_stage_b` were run again on 2026-10-23 and reproduced `stage_a.csv` and
`stage_b.csv` byte for byte, so the fingerprint in `outputs/task3/run_stamp.json`
matches the code and the guard lets a rerun reuse the tables. Edit
`splits.csv` or anything under `src/task3/` and the guard fires again, which is
what it is for; pass `--force` only when a full recomputation is what you want.

`evaluate` is the one command to leave alone. It scores the test set and
overwrites the shipped model `models/task3_pipeline.joblib` and the metric
tables in `outputs/task3/`. The test set was evaluated once, on 2026-10-14,
and the numbers in the report are that evaluation.

## Dataset

The photographs are on Google Drive:

```
https://drive.google.com/drive/folders/1_ld7gsHLhzyFJY3bUgTIwoOeVXRDFuAr
```

The folder is shared as "anyone with the link", so no sign-in, password or
access request is needed. The file names inside are the ones used throughout
the code and the report (for example `IN-03_001_IMG_1444.jpg` and
`OUT-06_001_IMG_0863.jpg`).

## Notes

- File names, folder names, EXIF and GPS take no part in the prediction. The
  classifier sees pixels only.
- The application loads the saved pipeline from
  `models/task3_pipeline.joblib` and never refits anything.
- Keep the dataset link available until grading is finished.

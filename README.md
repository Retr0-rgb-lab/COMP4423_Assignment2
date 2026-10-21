# COMP4423 Assignment 2

Classifies a scene photograph as inside the PolyU main campus (1) or outside it (0).

## Environment

Python 3.12 with the pins in `requirements.txt`. `opencv-contrib-python` is required: the non-contrib wheel has no SIFT, and the feature module stops rather than substituting another descriptor.

## Training

From the repository root:

```
python -m src.task3.run_stage_a
python -m src.task3.run_stage_b
python -m src.task3.evaluate
```

The first command compares six pixel features under one linear SVM, the second compares classifiers on the two winning features, and the third refits the locked pipeline on all training images and evaluates the test set once. If the existing result tables disagree with `splits.csv` or with the code under `src/task3/`, a resumed run refuses to reuse them; after confirming the difference, pass `--force`.

The split is `data/annotations/splits.csv`. Per-class and per-split counts are in `data/annotations/dataset_summary.md`. The collection and annotation notes are in `progress/task1_dataset_overview.md`. The features, the classifiers and the measured results are in `progress/task3_experiment.md`, and the test-set evaluation is in `progress/task4_evaluation.md`.

The locked model is `models/task3_pipeline.joblib` (HOG+LBP with a random forest). Prediction uses pixels only, through `resize_and_crop` in `src/task3/preprocess.py`.

## Task 5 local program

Open the selection window from the repository root:

```
python -m src.task5.app
```

Press **Select image…** (or Ctrl+O) and choose a scene photograph. The left pane shows the photograph, the right pane prints **Inside PolyU main campus** or **Outside PolyU main campus**, and both class probabilities are drawn. The model input is the same 256×256 centre crop of that image.

The program loads `models/task3_pipeline.joblib` and calls `resize_and_crop`. The prediction uses pixels only. File names, folder names, EXIF and GPS play no part in it, and nothing is retrained.

Without a desktop, the same pipeline prints a result for one image:

```
python -m src.task5.app --image path\to\photo.jpg
```

The official Windows Python build ships with Tk. Use `pythonw -m src.task5.app` for a window with no console. On Ubuntu, install `python3-tk` if Tk is missing. No extra dependency is required. The design and the recorded checks are in `progress/task5_app.md`.

## Dataset

The photographs are not in this archive. They are on Google Drive, and the link is in `DATASET_LINK.txt` and in the report. The folder is shared as "anyone with the link", so no sign-in is needed. The file names inside are the ones used throughout the code and the report.

## Report

The report body is `progress/task6_report_draft.md`. It is converted to LaTeX and compiled with:

```
python docs/make_report.py
pdflatex Assignment2_Report_24101322D_HUANG_Haoran.tex
```

`docs/report_template.tex` holds the title page, the fonts and the page layout. The conversion script wraps the two Chinese signs in a CJK environment, lets long file names break inside table cells, and merges the hand written caption into the figure caption so that it is not printed twice.

The Chinese working records under `progress/` are the source of the report. The report itself is in English.

## What is in this archive

```
src/task3/          training code (features, classifiers, venue-holdout CV, evaluation)
src/task5/          the local application
src/tools/          dataset construction and the dataset check script
models/             the locked pipeline
data/annotations/   splits.csv, the per-class counts, and the source manifests
outputs/task3/      the raw result tables behind the numbers in the report
docs/figures/       the report figures
docs/make_report.py the report build script
docs/check_fold_claims.py  checks the per-fold claims against stage_b.csv
progress/           the working records the report is written from
plans/              the review record and the remediation list
```

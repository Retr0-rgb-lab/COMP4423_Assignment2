# -*- coding: utf-8 -*-
"""Hard-coded Task 3 parameters. Editing this file is editing the experiment;
do not open a separate grid search at run time."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS_CSV = ROOT / "data" / "annotations" / "splits.csv"
PICTURES = ROOT / "data" / "pictures"
CACHE = ROOT / "cache" / "task3"
MODEL_PATH = ROOT / "models" / "task3_pipeline.joblib"
OUTPUT_DIR = ROOT / "outputs" / "task3"

SEED = 42
IMAGE_SIZE = 256
AUG_HFLIP = True

# Color histogram: H x S x V, whole image + 2x2
HSV_BINS = (16, 16, 8)

# HOG
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP
LBP_P = 8
LBP_R = 1
LBP_BINS = LBP_P + 2  # uniform → 0..P+1

# BoW
SIFT_NFEATURES = 500
BOW_CLUSTERS = 256
BOW_BATCH = 4096

PCA_DIM = 100

# 5-fold venue holdout (location-group holdout). OUT-05 and OUT-08 form one
# atomic unit. The four folds hold out one inside group and one outside group
# each, in ascending group order, and the fifth fold repeats the first inside
# group because the inside side has only four training groups while the
# outside side has six. After the 2026-10-27 repair of the location groups the
# inside training groups are IN-01, IN-02, IN-03 and IN-04.
FOLDS = (
    {"inside": ("IN-01",), "outside": ("OUT-01",)},
    {"inside": ("IN-02",), "outside": ("OUT-02",)},
    {"inside": ("IN-03",), "outside": ("OUT-03",)},
    {"inside": ("IN-04",), "outside": ("OUT-04",)},
    {"inside": ("IN-01",), "outside": ("OUT-05", "OUT-08")},
)

FEATURE_ORDER = ("F1", "F2", "F3", "F4", "F5", "F6")
FEATURE_NAME = {
    "F1": "HSV histogram",
    "F2": "HOG",
    "F3": "LBP",
    "F4": "HOG+LBP",
    "F5": "BoW",
    "F6": "BoW + spatial pyramid",
}
# Ties go to the smaller dimension; if still tied, take the earlier id in the table.
FEATURE_DIM = {
    "F1": 10240,
    "F2": 8100,
    "F3": 50,
    "F4": 8150,
    "F5": 256,
    "F6": 1280,
}
HIST_FEATURES = frozenset({"F1", "F3", "F5", "F6"})

CLF_ORDER = ("C1", "C2", "C3", "C4", "C5", "C6")
CLF_NAME = {
    "C1": "linear SVM",
    "C2": "logistic regression",
    "C3": "k-NN",
    "C4": "random forest",
    "C5": "histogram gradient boosting",
    "C6": "multinomial naive Bayes",
    "C1-RBF": "RBF SVM (PCA 100)",
}
# On tied scores, take the simpler classifier.
CLF_SIMPLICITY = ("C2", "C1", "C6", "C3", "C4", "C5")

DISPLAY = {
    0: "Outside PolyU main campus",
    1: "Inside PolyU main campus",
}

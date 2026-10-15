# -*- coding: utf-8 -*-
"""分类器与最终 Pipeline。参数以 config 为准。"""
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src.task3.config import CLF_NAME, FEATURE_NAME, HIST_FEATURES, PCA_DIM, SEED
from src.task3.features import build as build_feature


class ImageFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_id="F1"):
        self.feature_id = feature_id

    def fit(self, images, y=None):
        self.feature_ = build_feature(self.feature_id)
        self.feature_.fit(list(images), y)
        return self

    def transform(self, images):
        return self.feature_.transform(list(images))


def _knn_metric(feature_id):
    return "cosine" if feature_id in HIST_FEATURES else "euclidean"


def build_classifier(clf_id, feature_id):
    if clf_id == "C1":
        return LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=SEED)
    if clf_id == "C2":
        return LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED,
        )
    if clf_id == "C3":
        return KNeighborsClassifier(n_neighbors=5, metric=_knn_metric(feature_id))
    if clf_id == "C4":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=SEED,
            n_jobs=-1,
        )
    if clf_id == "C5":
        return HistGradientBoostingClassifier(
            max_depth=6, random_state=SEED,
        )
    if clf_id == "C6":
        if feature_id not in HIST_FEATURES:
            raise ValueError(f"C6 只用于直方图特征，不能用于 {feature_id}")
        return MultinomialNB(alpha=1.0)
    if clf_id == "C1-RBF":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)
    raise KeyError(clf_id)


def needs_scaler(clf_id):
    return clf_id in {"C1", "C2", "C1-RBF"}


class CapPCA(BaseEstimator, TransformerMixin):
    """PCA 到 100 维；特征本来就不到 100 维时降到现有维数。"""

    def __init__(self, n_components=PCA_DIM):
        self.n_components = n_components

    def fit(self, features, y=None):
        k = min(self.n_components, features.shape[1], max(1, features.shape[0] - 1))
        self.pca_ = PCA(n_components=k, random_state=SEED)
        self.pca_.fit(features)
        return self

    def transform(self, features):
        return self.pca_.transform(features)


def make_pipeline(feature_id, clf_id):
    steps = [("features", ImageFeatures(feature_id))]
    if clf_id == "C1-RBF":
        steps.append(("scale", StandardScaler()))
        steps.append(("pca", CapPCA(PCA_DIM)))
        steps.append(("clf", build_classifier(clf_id, feature_id)))
    elif needs_scaler(clf_id):
        steps.append(("scale", StandardScaler()))
        steps.append(("clf", build_classifier(clf_id, feature_id)))
    else:
        steps.append(("clf", build_classifier(clf_id, feature_id)))
    pipe = Pipeline(steps)
    pipe.feature_id_ = feature_id
    pipe.clf_id_ = clf_id
    pipe.feature_name_ = FEATURE_NAME[feature_id]
    pipe.clf_name_ = CLF_NAME[clf_id]
    return pipe

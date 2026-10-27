# -*- coding: utf-8 -*-
"""Classifiers and the final Pipeline. Parameters follow config."""
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
    """Wrap a Task 3 feature extractor as a scikit-learn transformer."""

    def __init__(self, feature_id="F1"):
        """Store the feature id; nothing is learned here."""
        self.feature_id = feature_id

    def fit(self, images, y=None):
        """Fit the wrapped feature extractor on the training images.

        The extractor is kept on the instance as feature_.
        """
        self.feature_ = build_feature(self.feature_id)
        self.feature_.fit(list(images), y)
        return self

    def transform(self, images):
        """Return the feature matrix of the images, as an array of shape
        (n, dimension)."""
        return self.feature_.transform(list(images))


def _knn_metric(feature_id):
    """Return the k-NN distance metric that suits the feature set.

    The histogram features are compared with the cosine distance, the other
    feature sets with the Euclidean distance.
    """
    return "cosine" if feature_id in HIST_FEATURES else "euclidean"


def build_classifier(clf_id, feature_id):
    """Return a new classifier with the parameters fixed in config.

    The classifier id is one of C1 to C6, plus C1-RBF for the RBF variant
    of stage two. An unknown id raises KeyError; C6 raises ValueError when
    the feature set is not a histogram, because the naive Bayes model needs
    non-negative inputs.
    """
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
            raise ValueError(f"C6 is only for histogram features, it cannot be used with {feature_id}")
        return MultinomialNB(alpha=1.0)
    if clf_id == "C1-RBF":
        return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)
    raise KeyError(clf_id)


def needs_scaler(clf_id):
    """Return True when the classifier needs standardised features.

    The linear SVM, the logistic regression and the RBF SVM are the models
    that need the standardisation step.
    """
    return clf_id in {"C1", "C2", "C1-RBF"}


class CapPCA(BaseEstimator, TransformerMixin):
    """PCA down to 100 dimensions; when the features are already fewer than
    100 dimensions, keep the current dimension count."""

    def __init__(self, n_components=PCA_DIM):
        """Store the requested number of components; nothing is learned
        here."""
        self.n_components = n_components

    def fit(self, features, y=None):
        """Fit the PCA on the training features.

        The number of components is capped by the data: at most
        n_components, at most the number of feature columns and at most one
        less than the number of samples. The fitted PCA is kept on the
        instance as pca_.
        """
        k = min(self.n_components, features.shape[1], max(1, features.shape[0] - 1))
        self.pca_ = PCA(n_components=k, random_state=SEED)
        self.pca_.fit(features)
        return self

    def transform(self, features):
        """Return the PCA projection of the features."""
        return self.pca_.transform(features)


def make_pipeline(feature_id, clf_id):
    """Return the Pipeline that joins the feature extractor and the
    classifier.

    The linear models get a standardisation step; the RBF SVM also gets a
    PCA down to 100 dimensions. The ids and the display names are kept on
    the pipeline as feature_id_, clf_id_, feature_name_ and clf_name_.
    """
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

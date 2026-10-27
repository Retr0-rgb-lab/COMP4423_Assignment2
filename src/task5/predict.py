# -*- coding: utf-8 -*-
"""Call the locked Pipeline on pixels.

File name, folder, EXIF and GPS never reach this module; the caller only
hands in an RGB array. Preprocessing must use the same ``resize_and_crop``
function as the evaluation: short side to 256, then a center crop to 256×256.
``HogFeat.transform`` calls it once more; when the size is already 256×256 the
function returns it unchanged.
"""
from pathlib import Path

import joblib

from src.task3.config import DISPLAY, MODEL_PATH
from src.task3.preprocess import resize_and_crop


class Prediction:
    """Hold the outcome of one prediction: the label, its text, the class
    probabilities and the cropped image that the model received."""

    def __init__(self, label_id, text, probabilities, cropped):
        """Store the label id, the label text, the class probabilities and the
        256x256 crop.

        ``label_id`` is the class the pipeline chose, ``text`` is the sentence
        shown to the user, ``probabilities`` maps a class id to its
        probability and is empty when the classifier reports none, and
        ``cropped`` is the image the model was given.
        """
        self.label_id = int(label_id)
        self.text = text
        self.probabilities = probabilities
        self.cropped = cropped


def load_pipeline(path=None):
    """Load the saved Pipeline from disk and return it.

    The default file is models/task3_pipeline.joblib. Nothing is fitted here:
    a missing file and a saved object without a predict method are both
    reported as errors."""
    model_path = Path(path) if path is not None else MODEL_PATH
    if not model_path.is_file():
        raise FileNotFoundError(
            "Saved classifier not found:\n"
            f"{model_path}\n\n"
            "From the repository root, run:\n"
            "python -m src.task3.evaluate"
        )
    pipeline = joblib.load(model_path)
    if not hasattr(pipeline, "predict"):
        raise TypeError(f"Saved object has no predict method: {type(pipeline)!r}")
    return pipeline


def label_text(pipeline, label_id):
    """Return the sentence that names a class for the window.

    The display_ mapping stored with the saved Pipeline is preferred, then the
    DISPLAY mapping of the configuration, and the raw class id is the last
    resort."""
    saved = getattr(pipeline, "display_", None) or {}
    if label_id in saved:
        return str(saved[label_id])
    if label_id in DISPLAY:
        return str(DISPLAY[label_id])
    return str(label_id)


def predict_rgb(pipeline, rgb):
    """Predict one RGB photograph and return the result.

    The image is H×W×3. The returned Prediction carries the official label,
    the two class probabilities when the classifier provides them, and the
    cropped image the model actually saw."""
    cropped = resize_and_crop(rgb)
    # The Pipeline expects a list of images; a bare ndarray would be
    # iterated row by row.
    label_id = int(pipeline.predict([cropped])[0])
    return Prediction(
        label_id,
        label_text(pipeline, label_id),
        class_probabilities(pipeline, cropped),
        cropped,
    )


def class_probabilities(pipeline, cropped):
    """Return the class id to probability mapping of one cropped image.

    The mapping is empty when the final estimator has no predict_proba, or when
    the row it returns does not match the classes of the Pipeline."""
    predict_proba = getattr(pipeline, "predict_proba", None)
    if predict_proba is None:
        return {}
    try:
        row = predict_proba([cropped])[0]
    except AttributeError:
        return {}
    classes = getattr(pipeline, "classes_", None)
    if classes is None or len(row) != len(classes):
        return {}
    return {int(label): float(score) for label, score in zip(classes, row)}

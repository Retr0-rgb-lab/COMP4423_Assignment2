# -*- coding: utf-8 -*-
"""对像素调用已锁定的 Pipeline。

文件名、文件夹、EXIF、GPS 不进这个模块。调用方只把 RGB 数组交进来。
预处理必须用评估同一函数 ``resize_and_crop``：短边 256，再中心裁成 256×256。
``HogLbp.transform`` 里还会再调一次；尺寸已经是 256×256 时该函数原样返回。
"""
from pathlib import Path

import joblib

from src.task3.config import DISPLAY, MODEL_PATH
from src.task3.preprocess import resize_and_crop


class Prediction:
    def __init__(self, label_id, text, probabilities, cropped):
        self.label_id = int(label_id)
        self.text = text
        self.probabilities = probabilities
        self.cropped = cropped


def load_pipeline(path=None):
    """读 ``models/task3_pipeline.joblib``。不重新训练。"""
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
    """优先用保存 Pipeline 时写上的 display_，否则用 config.DISPLAY。"""
    saved = getattr(pipeline, "display_", None) or {}
    if label_id in saved:
        return str(saved[label_id])
    if label_id in DISPLAY:
        return str(DISPLAY[label_id])
    return str(label_id)


def predict_rgb(pipeline, rgb):
    """``rgb`` 是 H×W×3。返回官方标签、两类概率（若分类器提供）和裁剪图。"""
    cropped = resize_and_crop(rgb)
    # Pipeline 吃的是图像列表。单个 ndarray 会被当成逐行迭代。
    label_id = int(pipeline.predict([cropped])[0])
    return Prediction(
        label_id,
        label_text(pipeline, label_id),
        class_probabilities(pipeline, cropped),
        cropped,
    )


def class_probabilities(pipeline, cropped):
    """类 id → 概率。最终估计器没有 predict_proba 时返回空字典。"""
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

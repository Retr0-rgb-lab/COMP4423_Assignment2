# -*- coding: utf-8 -*-
"""Pixel preprocessing. Task 5 must call this module and must not write its
own resizing."""
import numpy as np
from PIL import Image

from src.task3.config import IMAGE_SIZE


def load_rgb(path):
    """Read as RGB uint8. OpenCV's BGR order is not used, so loading stays
    consistent with the GUI."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize_and_crop(rgb):
    """Scale the short side to 256, then centre-crop to 256x256. Images that
    are already at that size are returned unchanged."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an H×W×3 RGB array, got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h, w = rgb.shape[:2]
    if h == IMAGE_SIZE and w == IMAGE_SIZE:
        return np.ascontiguousarray(rgb)

    import cv2
    if h < w:
        new_h, new_w = IMAGE_SIZE, int(round(w * IMAGE_SIZE / h))
    else:
        new_w, new_h = IMAGE_SIZE, int(round(h * IMAGE_SIZE / w))
    new_w = max(new_w, IMAGE_SIZE)
    new_h = max(new_h, IMAGE_SIZE)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (resized.shape[0] - IMAGE_SIZE) // 2
    x0 = (resized.shape[1] - IMAGE_SIZE) // 2
    out = resized[y0:y0 + IMAGE_SIZE, x0:x0 + IMAGE_SIZE]
    if out.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise RuntimeError(f"unexpected crop shape: {out.shape}")
    return np.ascontiguousarray(out)


def horizontal_flip(rgb):
    return np.ascontiguousarray(np.fliplr(rgb))

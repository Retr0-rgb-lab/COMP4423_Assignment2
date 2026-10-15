# -*- coding: utf-8 -*-
"""手工特征。fit 只看训练图；BoW 词典不跨折复用。"""
import hashlib

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import hog, local_binary_pattern

from src.task3.config import (
    BOW_BATCH,
    BOW_CLUSTERS,
    CACHE,
    FEATURE_DIM,
    HSV_BINS,
    HOG_CELLS_PER_BLOCK,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    IMAGE_SIZE,
    LBP_BINS,
    LBP_P,
    LBP_R,
    SEED,
    SIFT_NFEATURES,
)
from src.task3.preprocess import resize_and_crop

_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        if not hasattr(cv2, "SIFT_create"):
            raise SystemExit("cv2.SIFT_create 不可用。停下来改计划，不用 ORB 替换。")
        _SIFT = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    return _SIFT


def _digest(arr):
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _cache_load(kind, digest):
    path = CACHE / kind / f"{digest}.npy"
    if path.is_file():
        return np.load(path)
    return None


def _cache_save(kind, digest, arr):
    folder = CACHE / kind
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{digest}.npy", arr)


def _l1(hist):
    hist = hist.astype(np.float64, copy=False)
    total = hist.sum()
    if total <= 0:
        return hist
    return hist / total


def _hsv_hist(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        list(HSV_BINS),
        [0, 180, 0, 256, 0, 256],
    )
    return _l1(hist.reshape(-1))


def _quadrants(img):
    half = IMAGE_SIZE // 2
    return (
        img[0:half, 0:half],
        img[0:half, half:IMAGE_SIZE],
        img[half:IMAGE_SIZE, 0:half],
        img[half:IMAGE_SIZE, half:IMAGE_SIZE],
    )


def color_vec(rgb):
    digest = _digest(rgb)
    cached = _cache_load("color", digest)
    if cached is not None:
        return cached
    parts = [_hsv_hist(rgb)]
    parts.extend(_hsv_hist(block) for block in _quadrants(rgb))
    vec = np.concatenate(parts)
    _cache_save("color", digest, vec)
    return vec


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def hog_vec(rgb):
    digest = _digest(rgb)
    cached = _cache_load("hog", digest)
    if cached is not None:
        return cached
    vec = hog(
        _gray(rgb),
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float64)
    _cache_save("hog", digest, vec)
    return vec


def lbp_vec(rgb):
    digest = _digest(rgb)
    cached = _cache_load("lbp", digest)
    if cached is not None:
        return cached

    def one(block):
        code = local_binary_pattern(_gray(block), LBP_P, LBP_R, method="uniform")
        hist, _edges = np.histogram(code, bins=np.arange(LBP_BINS + 1))
        return _l1(hist.astype(np.float64))

    parts = [one(rgb)]
    parts.extend(one(block) for block in _quadrants(rgb))
    vec = np.concatenate(parts)
    _cache_save("lbp", digest, vec)
    return vec


def sift_cached(rgb):
    """返回 (pts[N,2], desc[N,128])。没有关键点时 pts/desc 为空数组。"""
    digest = _digest(rgb)
    folder = CACHE / "sift"
    path = folder / f"{digest}.npz"
    if path.is_file():
        with np.load(path) as data:
            return data["pts"], data["desc"]
    gray = _gray(rgb)
    keypoints, desc = _sift().detectAndCompute(gray, None)
    if desc is None or len(keypoints) == 0:
        pts = np.zeros((0, 2), np.float32)
        desc = np.zeros((0, 128), np.float32)
    else:
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        desc = np.asarray(desc, dtype=np.float32)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(path, pts=pts, desc=desc)
    return pts, desc


def _hist_from_words(words):
    if len(words) == 0:
        return np.zeros(BOW_CLUSTERS, dtype=np.float64)
    counted = np.bincount(words, minlength=BOW_CLUSTERS).astype(np.float64)
    return _l1(counted)


def _word_hist(desc, kmeans):
    if len(desc) == 0:
        return np.zeros(BOW_CLUSTERS, dtype=np.float64)
    return _hist_from_words(kmeans.predict(desc))


def _spatial_hists(pts, desc, kmeans):
    """整图 + 2×2，各块单独 L1。"""
    global_hist = _word_hist(desc, kmeans)
    quads = [np.zeros(BOW_CLUSTERS, dtype=np.float64) for _ in range(4)]
    if len(desc):
        words = kmeans.predict(desc)
        half = IMAGE_SIZE / 2.0
        qx = (pts[:, 0] >= half).astype(np.int32)
        qy = (pts[:, 1] >= half).astype(np.int32)
        bin_id = qy * 2 + qx
        for q in range(4):
            quads[q] = _hist_from_words(words[bin_id == q])
    return np.concatenate([global_hist, *quads])


class _ImageFeature(BaseEstimator, TransformerMixin):
    def fit(self, images, y=None):
        return self

    def transform(self, images):
        raise NotImplementedError


class ColorHist(_ImageFeature):
    def transform(self, images):
        return np.stack([color_vec(resize_and_crop(im)) for im in images])


class HogFeat(_ImageFeature):
    def transform(self, images):
        return np.stack([hog_vec(resize_and_crop(im)) for im in images])


class LbpFeat(_ImageFeature):
    def transform(self, images):
        return np.stack([lbp_vec(resize_and_crop(im)) for im in images])


class HogLbp(BaseEstimator, TransformerMixin):
    """F2、F3 在训练集上各自标准化后再拼接。"""

    def fit(self, images, y=None):
        cropped = [resize_and_crop(im) for im in images]
        hogs = np.stack([hog_vec(im) for im in cropped])
        lbps = np.stack([lbp_vec(im) for im in cropped])
        self.hog_mean_ = hogs.mean(axis=0)
        self.hog_std_ = _safe_std(hogs)
        self.lbp_mean_ = lbps.mean(axis=0)
        self.lbp_std_ = _safe_std(lbps)
        return self

    def transform(self, images):
        cropped = [resize_and_crop(im) for im in images]
        hogs = (np.stack([hog_vec(im) for im in cropped]) - self.hog_mean_) / self.hog_std_
        lbps = (np.stack([lbp_vec(im) for im in cropped]) - self.lbp_mean_) / self.lbp_std_
        return np.concatenate([hogs, lbps], axis=1)


def _safe_std(arr):
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    return std


class _BoWBase(BaseEstimator, TransformerMixin):
    def fit(self, images, y=None):
        descs = []
        self.n_empty_ = 0
        for im in images:
            _pts, desc = sift_cached(resize_and_crop(im))
            if len(desc) == 0:
                self.n_empty_ += 1
            else:
                descs.append(desc)
        if not descs:
            raise SystemExit("训练集没有提取到任何 SIFT 描述子")
        stacked = np.concatenate(descs, axis=0)
        if len(stacked) < BOW_CLUSTERS:
            raise SystemExit(
                f"描述子 {len(stacked)} 个，少于词典 {BOW_CLUSTERS}，无法建词典"
            )
        self.kmeans_ = MiniBatchKMeans(
            n_clusters=BOW_CLUSTERS,
            random_state=SEED,
            batch_size=min(BOW_BATCH, len(stacked)),
            n_init=3,
            max_no_improvement=20,
        )
        self.kmeans_.fit(stacked)
        return self

    def _cached_sift(self, images):
        out = []
        empty = 0
        for im in images:
            pts, desc = sift_cached(resize_and_crop(im))
            if len(desc) == 0:
                empty += 1
            out.append((pts, desc))
        if empty:
            print(f"  SIFT 空图 {empty} 张，直方图记为 0")
        return out


class Bow(_BoWBase):
    def transform(self, images):
        rows = []
        for _pts, desc in self._cached_sift(images):
            rows.append(_word_hist(desc, self.kmeans_))
        return np.stack(rows)


class BowSpatial(_BoWBase):
    def transform(self, images):
        rows = []
        for pts, desc in self._cached_sift(images):
            rows.append(_spatial_hists(pts, desc, self.kmeans_))
        return np.stack(rows)


def build(feature_id):
    table = {
        "F1": ColorHist,
        "F2": HogFeat,
        "F3": LbpFeat,
        "F4": HogLbp,
        "F5": Bow,
        "F6": BowSpatial,
    }
    if feature_id not in table:
        raise KeyError(feature_id)
    feature = table[feature_id]()
    return feature


def assert_dims(sample_rgb):
    cropped = resize_and_crop(sample_rgb)
    got = {
        "F1": color_vec(cropped).shape[0],
        "F2": hog_vec(cropped).shape[0],
        "F3": lbp_vec(cropped).shape[0],
    }
    for key, dim in got.items():
        if dim != FEATURE_DIM[key]:
            raise SystemExit(f"{key} 维度是 {dim}，计划写的是 {FEATURE_DIM[key]}")
    return got

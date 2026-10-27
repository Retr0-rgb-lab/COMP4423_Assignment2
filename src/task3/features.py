# -*- coding: utf-8 -*-
"""Hand-crafted features. fit only sees training images; the BoW codebook is
never reused across folds."""
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
    """Return the shared SIFT detector, creating it on the first call.

    The detector is kept in a module global, so every image reuses the same
    instance. It raises SystemExit when the OpenCV build has no SIFT
    support, because the plan forbids substituting another keypoint
    detector.
    """
    global _SIFT
    if _SIFT is None:
        if not hasattr(cv2, "SIFT_create"):
            raise SystemExit("cv2.SIFT_create is not available. Stop and change the plan; do not substitute ORB.")
        _SIFT = cv2.SIFT_create(nfeatures=SIFT_NFEATURES)
    return _SIFT


def _digest(arr):
    """Return the MD5 hex digest of the array bytes, used as the cache key."""
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _cache_load(kind, digest):
    """Return the cached array for this feature kind and digest, or None.

    The cache file is cache/task3/<kind>/<digest>.npy; a missing file means
    the feature has not been computed for that image yet.
    """
    path = CACHE / kind / f"{digest}.npy"
    if path.is_file():
        return np.load(path)
    return None


def _cache_save(kind, digest, arr):
    """Save the array as cache/task3/<kind>/<digest>.npy.

    The folder is created when it does not exist yet.
    """
    folder = CACHE / kind
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{digest}.npy", arr)


def _l1(hist):
    """Return the histogram scaled to sum one.

    A histogram whose total is zero is returned unchanged, so that the
    result stays free of NaN.
    """
    hist = hist.astype(np.float64, copy=False)
    total = hist.sum()
    if total <= 0:
        return hist
    return hist / total


def _hsv_hist(rgb):
    """Return the L1-normalised 3-D HSV histogram of the RGB image.

    The three axes are hue, saturation and value, binned with HSV_BINS and
    spanning the ranges 0-180, 0-256 and 0-256.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        list(HSV_BINS),
        [0, 180, 0, 256, 0, 256],
    )
    return _l1(hist.reshape(-1))


def _quadrants(img):
    """Return the four quadrant blocks: top-left, top-right, bottom-left and
    bottom-right.

    The image is expected to be the 256x256 cropped one, so every block is
    128x128.
    """
    half = IMAGE_SIZE // 2
    return (
        img[0:half, 0:half],
        img[0:half, half:IMAGE_SIZE],
        img[half:IMAGE_SIZE, 0:half],
        img[half:IMAGE_SIZE, half:IMAGE_SIZE],
    )


def color_vec(rgb):
    """Return the HSV colour vector of the cropped RGB image.

    The vector is the whole-image histogram followed by the four quadrant
    histograms, each L1-normalised on its own: 5 x 2048 = 10240 values.
    The result is cached on disk under the MD5 digest of the image bytes.
    """
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
    """Return the grayscale version of the RGB image."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def hog_vec(rgb):
    """Return the HOG descriptor of the RGB image as a float64 vector.

    The descriptor holds 8100 values: 9 orientations over the blocks of the
    16x16 pixel cells. The result is cached on disk under the MD5 digest of
    the image bytes.
    """
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
    """Return the LBP histogram vector of the cropped RGB image.

    The vector is the whole-image histogram followed by the four quadrant
    histograms: 5 x 10 = 50 values, each block L1-normalised on its own.
    The result is cached on disk under the MD5 digest of the image bytes.
    """
    digest = _digest(rgb)
    cached = _cache_load("lbp", digest)
    if cached is not None:
        return cached

    def one(block):
        """Return the L1-normalised uniform LBP histogram of one block."""
        code = local_binary_pattern(_gray(block), LBP_P, LBP_R, method="uniform")
        hist, _edges = np.histogram(code, bins=np.arange(LBP_BINS + 1))
        return _l1(hist.astype(np.float64))

    parts = [one(rgb)]
    parts.extend(one(block) for block in _quadrants(rgb))
    vec = np.concatenate(parts)
    _cache_save("lbp", digest, vec)
    return vec


def sift_cached(rgb):
    """Return (pts[N,2], desc[N,128]). With no keypoints, pts and desc are
    empty arrays."""
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
    """Return the L1-normalised histogram of the visual words.

    The histogram has BOW_CLUSTERS entries; an empty word list gives a zero
    histogram.
    """
    if len(words) == 0:
        return np.zeros(BOW_CLUSTERS, dtype=np.float64)
    counted = np.bincount(words, minlength=BOW_CLUSTERS).astype(np.float64)
    return _l1(counted)


def _word_hist(desc, kmeans):
    """Return the L1-normalised bag-of-words histogram of the descriptors.

    Every descriptor is assigned to its nearest codebook centre; an empty
    descriptor set gives a zero histogram.
    """
    if len(desc) == 0:
        return np.zeros(BOW_CLUSTERS, dtype=np.float64)
    return _hist_from_words(kmeans.predict(desc))


def _spatial_hists(pts, desc, kmeans):
    """Whole image plus 2x2 quadrants, each block L1-normalised separately."""
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
    """Base class of the feature sets that have no fitted state.

    The subclasses only implement transform; fit is a no-op, because a
    histogram carries no learned parameter.
    """

    def fit(self, images, y=None):
        """Accept the training images and return the estimator unchanged."""
        return self

    def transform(self, images):
        """Raise NotImplementedError, because the subclasses do the work."""
        raise NotImplementedError


class ColorHist(_ImageFeature):
    """Compute the F1 feature set: the HSV colour histogram of the whole
    image and of its four quadrants."""

    def transform(self, images):
        """Return one colour vector per image, as an array of shape
        (n, 10240)."""
        return np.stack([color_vec(resize_and_crop(im)) for im in images])


class HogFeat(_ImageFeature):
    """Compute the F2 feature set: the HOG descriptor of the whole image."""

    def transform(self, images):
        """Return one HOG descriptor per image, as an array of shape
        (n, 8100)."""
        return np.stack([hog_vec(resize_and_crop(im)) for im in images])


class LbpFeat(_ImageFeature):
    """Compute the F3 feature set: the LBP histogram of the whole image and
    of its four quadrants."""

    def transform(self, images):
        """Return one LBP vector per image, as an array of shape (n, 50)."""
        return np.stack([lbp_vec(resize_and_crop(im)) for im in images])


class HogLbp(BaseEstimator, TransformerMixin):
    """F2 and F3 are standardised separately on the training set and then
    concatenated."""

    def fit(self, images, y=None):
        """Learn the standardisation of the HOG part and of the LBP part on
        the training images.

        The per-dimension mean and standard deviation of each part are kept
        on the instance as hog_mean_, hog_std_, lbp_mean_ and lbp_std_.
        """
        cropped = [resize_and_crop(im) for im in images]
        hogs = np.stack([hog_vec(im) for im in cropped])
        lbps = np.stack([lbp_vec(im) for im in cropped])
        self.hog_mean_ = hogs.mean(axis=0)
        self.hog_std_ = _safe_std(hogs)
        self.lbp_mean_ = lbps.mean(axis=0)
        self.lbp_std_ = _safe_std(lbps)
        return self

    def transform(self, images):
        """Return the standardised HOG part and LBP part joined into one
        array.

        The array has shape (n, 8150): 8100 HOG values followed by 50 LBP
        values, each standardised with the statistics kept by fit.
        """
        cropped = [resize_and_crop(im) for im in images]
        hogs = (np.stack([hog_vec(im) for im in cropped]) - self.hog_mean_) / self.hog_std_
        lbps = (np.stack([lbp_vec(im) for im in cropped]) - self.lbp_mean_) / self.lbp_std_
        return np.concatenate([hogs, lbps], axis=1)


def _safe_std(arr):
    """Return the per-dimension standard deviation, with the near-zero
    values replaced by one.

    The replacement keeps the division in HogLbp.transform safe when a
    feature is constant.
    """
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    return std


class _BoWBase(BaseEstimator, TransformerMixin):
    """Base class of the bag-of-words feature sets.

    The codebook is learned on the training images only and is never reused
    across folds; the SIFT descriptors themselves are cached on disk.
    """

    def fit(self, images, y=None):
        """Learn the visual-word codebook on the training images.

        The k-means codebook is kept on the instance as kmeans_, and the
        number of training images without any descriptor as n_empty_. It
        raises SystemExit when the training set has no descriptor at all,
        or has fewer descriptors than the codebook size.
        """
        descs = []
        self.n_empty_ = 0
        for im in images:
            _pts, desc = sift_cached(resize_and_crop(im))
            if len(desc) == 0:
                self.n_empty_ += 1
            else:
                descs.append(desc)
        if not descs:
            raise SystemExit("no SIFT descriptor was extracted from the training set")
        stacked = np.concatenate(descs, axis=0)
        if len(stacked) < BOW_CLUSTERS:
            raise SystemExit(
                f"{len(stacked)} descriptors, fewer than the codebook size {BOW_CLUSTERS}, cannot build the codebook"
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
        """Return the (points, descriptors) pair of every image.

        The descriptors come from the disk cache when it is already filled.
        The number of images without any descriptor is printed, because
        their histogram is recorded as zero.
        """
        out = []
        empty = 0
        for im in images:
            pts, desc = sift_cached(resize_and_crop(im))
            if len(desc) == 0:
                empty += 1
            out.append((pts, desc))
        if empty:
            print(f"  {empty} SIFT-empty images, the histogram is recorded as 0")
        return out


class Bow(_BoWBase):
    """Compute the F5 feature set: the bag-of-words histogram of the whole
    image."""

    def transform(self, images):
        """Return one bag-of-words histogram per image, as an array of shape
        (n, 256)."""
        rows = []
        for _pts, desc in self._cached_sift(images):
            rows.append(_word_hist(desc, self.kmeans_))
        return np.stack(rows)


class BowSpatial(_BoWBase):
    """Compute the F6 feature set: the bag-of-words histogram of the whole
    image and of its four quadrants."""

    def transform(self, images):
        """Return one spatial pyramid histogram per image, as an array of
        shape (n, 1280)."""
        rows = []
        for pts, desc in self._cached_sift(images):
            rows.append(_spatial_hists(pts, desc, self.kmeans_))
        return np.stack(rows)


def build(feature_id):
    """Return a new feature extractor for the given feature id.

    The id is one of F1 to F6; an unknown id raises KeyError.
    """
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
    """Check the feature dimensions against the plan and return them.

    Returns the dimension of F1, F2 and F3 for one sample image; a mismatch
    with FEATURE_DIM raises SystemExit.
    """
    cropped = resize_and_crop(sample_rgb)
    got = {
        "F1": color_vec(cropped).shape[0],
        "F2": hog_vec(cropped).shape[0],
        "F3": lbp_vec(cropped).shape[0],
    }
    for key, dim in got.items():
        if dim != FEATURE_DIM[key]:
            raise SystemExit(f"the dimension of {key} is {dim}, the plan says {FEATURE_DIM[key]}")
    return got

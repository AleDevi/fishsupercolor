"""Colour classes per pixel, ilastik-style: hand-made features (Lab colour smoothed at a few scales, edges,
blob filters) + a random forest trained on the user's brush strokes.

The forest is trained with scikit-learn, then stored as plain arrays (.npz, no pickle) and evaluated here with
numpy, so a model file received from someone else cannot run code."""
import json
import os

import cv2
import numpy as np

FEATURES_VERSION = 1
SIGMAS = (1.0, 2.0, 4.0)
WORK_SIDE = 900          # the fish crop is classified with its long side at this size
N_FEATURES = 5 + 6 * len(SIGMAS)


def features(rgb):
    """rgb uint8 HxWx3 -> float32 HxWxF."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    f = [L, a, b, np.hypot(a, b), np.arctan2(b, a)]
    for s in SIGMAS:
        Ls, As, Bs = (cv2.GaussianBlur(c, (0, 0), s) for c in (L, a, b))
        gx, gy = cv2.Sobel(Ls, cv2.CV_32F, 1, 0), cv2.Sobel(Ls, cv2.CV_32F, 0, 1)
        f += [Ls, As, Bs, np.hypot(gx, gy), cv2.Laplacian(Ls, cv2.CV_32F),
              np.sqrt(np.maximum(cv2.GaussianBlur(L * L, (0, 0), s) - Ls * Ls, 0))]     # local contrast
    return np.stack(f, -1).astype(np.float32)


def work_crop(mask, pad=10):
    """Bounding box of the fish and the scale that brings it to WORK_SIDE."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = max(ys.min() - pad, 0), ys.max() + pad + 1, max(xs.min() - pad, 0), xs.max() + pad + 1
    s = min(1.0, WORK_SIDE / max(y1 - y0, x1 - x0))
    return (x0, y0, x1, y1), s


def _shrink(a, box, s, interp):
    x0, y0, x1, y1 = box
    a = a[y0:y1, x0:x1]
    if s == 1:
        return a
    return cv2.resize(a, (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s))), interpolation=interp)


def training_pixels(rgb, mask, strokes, per_class=4000, seed=0):
    """Features and labels of stroked pixels (strokes: uint8, 0 = none, k = class k)."""
    box, s = work_crop(mask)
    small = _shrink(rgb, box, s, cv2.INTER_AREA)
    st = _shrink(strokes, box, s, cv2.INTER_NEAREST)
    F = features(small)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k in np.unique(st[st > 0]):
        idx = np.flatnonzero(st == k)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        X.append(F.reshape(-1, F.shape[-1])[idx])
        y.append(np.full(len(idx), k, np.uint8))
    if not X:
        return np.zeros((0, F.shape[-1]), np.float32), np.zeros(0, np.uint8)
    return np.concatenate(X), np.concatenate(y)


class Forest:
    """A random forest as flat arrays: per node left/right child (-1 = leaf), feature, threshold, class probabilities."""

    def __init__(self, left, right, feature, threshold, proba, roots, classes, meta=None):
        self.left, self.right, self.feature, self.threshold = left, right, feature, threshold
        self.proba, self.roots, self.classes, self.meta = proba, roots, classes, meta or {}

    @classmethod
    def train(cls, X, y, trees=60, depth=14, seed=0, progress=None, sample_weight=None):
        """progress(trees done, trees) while the forest grows (in steps of 10 trees); sample_weight: per pixel."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.utils.class_weight import compute_class_weight
        oob = len(np.unique(y)) > 1 and len(y) >= 50
        cl = np.unique(y)
        weights = dict(zip(cl.tolist(), compute_class_weight('balanced', classes=cl, y=y)))   # = 'balanced'
        rf = RandomForestClassifier(n_estimators=0, max_depth=depth, min_samples_leaf=3, n_jobs=-1, warm_start=True,
                                    random_state=seed, class_weight=weights)
        for n in range(min(10, trees), trees + 10, 10):
            n = min(n, trees)
            rf.set_params(n_estimators=n, oob_score=oob and n == trees)      # held-out accuracy once, at the end
            rf.fit(X, y, sample_weight=sample_weight)
            if progress:
                progress(n, trees)
            if n == trees:
                break
        parts, off = [], 0
        for est in rf.estimators_:
            t = est.tree_
            v = t.value[:, 0, :]
            v = v / np.maximum(v.sum(1, keepdims=True), 1e-12)
            l, r = t.children_left.copy(), t.children_right.copy()
            l[l >= 0] += off; r[r >= 0] += off
            parts.append((l, r, t.feature, t.threshold, v))
            off += t.node_count
        cat = lambda i, dt: np.concatenate([p[i] for p in parts]).astype(dt)
        roots = np.cumsum([0] + [p[0].size for p in parts[:-1]]).astype(np.int64)
        meta = {'oob_accuracy': round(float(rf.oob_score_), 4) if getattr(rf, 'oob_score', False) else None}
        return cls(cat(0, np.int64), cat(1, np.int64), cat(2, np.int64), cat(3, np.float32),
                   cat(4, np.float32), roots, rf.classes_.astype(np.uint8), meta)

    def predict_proba(self, X):
        P = np.zeros((len(X), len(self.classes)), np.float32)
        rows = np.arange(len(X))
        for root in self.roots:
            node = np.full(len(X), root, np.int64)
            while True:
                inner = self.left[node] >= 0
                if not inner.any():
                    break
                n = node[inner]
                go_left = X[rows[inner], self.feature[n]] <= self.threshold[n]
                node[inner] = np.where(go_left, self.left[n], self.right[n])
            P += self.proba[node]
        return P / len(self.roots)

    def save(self, path_npz, pool=None):
        """pool: optional extra arrays saved alongside (the training pixels, see training.load_pool)."""
        np.savez_compressed(path_npz, left=self.left, right=self.right, feature=self.feature,
                            threshold=self.threshold, proba=self.proba, roots=self.roots, classes=self.classes,
                            **(pool or {}))
        with open(os.path.splitext(path_npz)[0] + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path_npz):
        with np.load(path_npz, allow_pickle=False) as z:
            arrays = {k: z[k] for k in ('left', 'right', 'feature', 'threshold', 'proba', 'roots', 'classes')}
        meta_path = os.path.splitext(path_npz)[0] + '.json'
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
        return cls(meta=meta, **arrays)


def classify(rgb, mask, forest):
    """Class of every fish pixel. Returns (labels at work size, crop box, scale, max probability)."""
    box, s = work_crop(mask)
    small = _shrink(rgb, box, s, cv2.INTER_AREA)
    m = _shrink(mask.astype(np.uint8), box, s, cv2.INTER_NEAREST) > 0
    F = features(small)
    P = forest.predict_proba(F[m])
    lab = np.zeros(m.shape, np.uint8)
    lab[m] = forest.classes[P.argmax(1)]
    conf = np.zeros(m.shape, np.float32)
    conf[m] = P.max(1)
    return lab, box, s, conf


def spots(binary, merge_px=0, min_px=1):
    """Pixel counts of the separate spots; spots closer than merge_px count as one (as when measured by hand)."""
    b = binary.astype(np.uint8)
    joined = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_px + 1,) * 2)) if merge_px > 0 else b
    n, comp = cv2.connectedComponents(joined, connectivity=8)
    sizes = np.bincount(comp[b > 0], minlength=n)[1:]
    return sorted((int(x) for x in sizes if x >= min_px), reverse=True)

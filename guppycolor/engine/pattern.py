"""Colour pattern of many fish on one common shape (phase 3).
Outline points (snout, peduncle, dorsal/ventral semi-landmarks, plus any named landmarks) -> generalised
Procrustes analysis -> mean shape -> thin-plate-spline warp of every colour map onto the mean shape.
On the aligned maps: per-pixel colour maps, spots as points (round) or lines (elongated), distances, clusters.
Coordinates of the aligned canvas: x = 0 at the snout and 1 at the caudal peduncle (body lengths), y = body
lengths from the midline, positive towards the dorsal side. Fish are put head-left, dorsal side up."""
import numpy as np, cv2
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from . import morpho

CANVAS_W = 320          # aligned canvas, px; the body (snout -> peduncle) spans BODY_PX of it
BODY_PX = 260
MARGIN = 30


# --- shape -----------------------------------------------------------------------------------------------
def outline_points(mask, n_semi=10, region='body', landmarks=None, landmark_keys=(), head=None):
    """Points in the same order for every fish: snout, n_semi dorsal, end (index n_semi + 1), n_semi ventral
    (reversed), then the named landmarks. region 'body' ends at the caudal peduncle, 'whole' at the end of the mask.
    head: None = guess, 'left' / 'right' = side of the picture the head is on.
    Returns (image px, local frame: x along snout -> end, y towards the dorsal side)."""
    m = morpho.measure(mask, head=head)
    snout, end = np.asarray(m['snout'], float), np.asarray(m['ped' if region == 'body' else 'end'], float)
    ax = (end - snout) / max(np.hypot(*(end - snout)), 1e-9)
    up = np.array([ax[1], -ax[0]])
    if up[1] > 0:                               # dorsal = towards the top of the picture
        up = -up
    ys, xs = np.nonzero(m['mask'])
    P = np.c_[xs, ys].astype(float) - snout
    t, n = P @ ax, P @ up
    L = np.hypot(*(end - snout))
    stations = np.linspace(0, L, n_semi + 2)[1:-1]
    half = L / (n_semi + 1) / 4
    dors, vent = [], []
    for s in stations:
        sel = np.abs(t - s) <= half
        if not sel.any():
            sel = np.argsort(np.abs(t - s))[:5]
        hi, lo = n[sel].max(), n[sel].min()
        dors.append(snout + s * ax + hi * up)
        vent.append(snout + s * ax + lo * up)
    pts = np.array([snout] + dors + [end] + vent[::-1] + [np.asarray(landmarks[k], float) for k in landmark_keys])
    local = np.c_[(pts - snout) @ ax, (pts - snout) @ up]      # same handedness for every fish, mirrored or not
    return pts, local


def _orth(A, B):
    """Rotation (no reflection) that best maps A onto B (both centred)."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, d]) @ Vt


def gpa(shapes, iters=20, tol=1e-8):
    """Generalised Procrustes analysis. shapes (N, K, 2) -> aligned (N, K, 2), mean (K, 2), centroid sizes (N,)."""
    X = np.asarray(shapes, float)
    X = X - X.mean(1, keepdims=True)
    cs = np.sqrt((X ** 2).sum((1, 2)))
    X = X / cs[:, None, None]
    mean = X[0]
    for _ in range(iters):
        X = np.array([x @ _orth(x, mean) for x in X])
        new = X.mean(0)
        new /= np.sqrt((new ** 2).sum())
        new = new @ _orth(new, mean)
        done = ((new - mean) ** 2).sum() < tol
        mean = new
        if done:
            break
    return X, mean, cs


def canvas_shape(mean, end_index):
    """Mean shape (local frame, dorsal = +y) in canvas px: snout at x = MARGIN, end at MARGIN + BODY_PX, dorsal up."""
    v = mean[end_index] - mean[0]
    a = -np.arctan2(v[1], v[0])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M = (mean - mean[0]) @ R.T
    M = M * (BODY_PX / M[end_index, 0])
    H = int(2 * (np.abs(M[:, 1]).max() + MARGIN))
    return np.c_[M[:, 0] + MARGIN, H / 2 - M[:, 1]], (H, CANVAS_W)


def outline_polygon(canvas_pts, n_semi):
    return canvas_pts[:2 * n_semi + 2].astype(np.int32)


def template_mask(canvas_pts, size, n_semi):
    m = np.zeros(size, np.uint8)
    cv2.fillPoly(m, [outline_polygon(canvas_pts, n_semi)], 1)
    return m > 0


def warp_labels(lab, lab_pts, canvas_pts, tmask, smoothing=0.0):
    """Sample a label map (any integer image) on the canvas: TPS canvas -> label-map coordinates, nearest pixel.
    Canvas pixels that fall outside the label map get 0."""
    tps = RBFInterpolator(canvas_pts, lab_pts, kernel='thin_plate_spline', smoothing=smoothing)
    yy, xx = np.nonzero(tmask)
    src = tps(np.c_[xx, yy].astype(float))
    sx, sy = np.rint(src[:, 0]).astype(int), np.rint(src[:, 1]).astype(int)
    ok = (sx >= 0) & (sy >= 0) & (sx < lab.shape[1]) & (sy < lab.shape[0])
    out = np.zeros(tmask.shape, lab.dtype)
    out[yy[ok], xx[ok]] = lab[sy[ok], sx[ok]]
    return out


# --- spots -------------------------------------------------------------------------------------------------
def to_body(x, y, canvas_pts):
    """Canvas px -> body lengths (x from the snout, y from the snout-end line, dorsal positive)."""
    x0, y0 = canvas_pts[0]
    return (np.asarray(x) - x0) / BODY_PX, (y0 - np.asarray(y)) / BODY_PX


def find_spots(binary, canvas_pts, min_px=4, line_ratio=2.5, merge_px=0):
    """Spots of one class on the aligned map. Round ones are points, elongated ones (length / width >= line_ratio)
    are lines with two ends. Positions in body lengths; area as a fraction of the body length squared."""
    b = binary.astype(np.uint8)
    joined = cv2.dilate(b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_px + 1,) * 2)) if merge_px else b
    n, comp = cv2.connectedComponents(joined, connectivity=8)
    comp[b == 0] = 0
    out = []
    for i in range(1, n):
        yy, xx = np.nonzero(comp == i)
        if len(xx) < min_px:
            continue
        P = np.c_[xx, yy].astype(float)
        c = P.mean(0)
        if len(P) > 2:
            ev, evec = np.linalg.eigh(np.cov((P - c).T))
        else:
            ev, evec = np.array([0.25, 0.25]), np.eye(2)
        v = evec[:, 1]
        proj = (P - c) @ v
        length = proj.max() - proj.min() + 1
        width = max(4 * np.sqrt(max(ev[0], 0)) , 1.0)
        ratio = length / width
        kind = 'line' if ratio >= line_ratio else 'point'
        e1, e2 = c + proj.min() * v, c + proj.max() * v
        bx, by = to_body([c[0], e1[0], e2[0]], [c[1], e1[1], e2[1]], canvas_pts)
        angle = float(np.degrees(np.arctan2(-(e2[1] - e1[1]), e2[0] - e1[0])))
        angle = (angle + 90) % 180 - 90                 # -90..90, 0 = along the body, positive = rising to the tail
        out.append(dict(kind=kind, x=float(bx[0]), y=float(by[0]), x1=float(bx[1]), y1=float(by[1]),
                        x2=float(bx[2]), y2=float(by[2]), length=float(length / BODY_PX),
                        width=float(width / BODY_PX), elongation=float(ratio), angle=angle,
                        area=float(len(xx) / BODY_PX ** 2)))
    return out


# --- distances ---------------------------------------------------------------------------------------------
def class_stack(amap, valid, classes, sigma=0.0):
    """(C, H, W) float maps: 1 where the fish has that class; blurred by sigma px to forgive small misalignments."""
    out = []
    for c in classes:
        m = ((amap == c) & valid).astype(np.float32)
        out.append(gaussian_filter(m, sigma) if sigma > 0 else m)
    return np.array(out)


def jaccard_matrix(stacks, valid):
    """1 - weighted Jaccard (Ruzicka) over all classes together: sum(min) / sum(max) over pixels valid in both fish.
    stacks (N, C, P) and valid (N, P) on the same P pixels. With no blur it is the plain Jaccard distance."""
    S = np.asarray(stacks, np.float32)
    V = np.asarray(valid, bool)
    N = len(S)
    D = np.zeros((N, N))
    for i in range(N - 1):
        v = (V[i] & V[i + 1:])[:, None, :]
        a, b = S[i][None], S[i + 1:]
        mn = (np.minimum(a, b) * v).sum((1, 2))
        mx = (np.maximum(a, b) * v).sum((1, 2))
        d = np.where(mx > 0, 1 - mn / np.maximum(mx, 1e-12), 0.0)
        D[i, i + 1:] = D[i + 1:, i] = d
    return D


def composition_matrix(fractions):
    """Bray-Curtis distance on the area fraction of each class (no position involved)."""
    F = np.asarray(fractions, float)
    N = len(F)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            s = (F[i] + F[j]).sum()
            D[i, j] = D[j, i] = 0.0 if s == 0 else np.abs(F[i] - F[j]).sum() / s
    return D


def _spot_cost(a, b, kind_penalty):
    if a['kind'] == 'line' and b['kind'] == 'line':
        d1 = np.hypot(a['x1'] - b['x1'], a['y1'] - b['y1']) + np.hypot(a['x2'] - b['x2'], a['y2'] - b['y2'])
        d2 = np.hypot(a['x1'] - b['x2'], a['y1'] - b['y2']) + np.hypot(a['x2'] - b['x1'], a['y2'] - b['y1'])
        return min(d1, d2) / 2
    d = np.hypot(a['x'] - b['x'], a['y'] - b['y'])
    return d + (kind_penalty if a['kind'] != b['kind'] else 0.0)


def spot_set_distance(A, B, max_cost=0.15, kind_penalty=0.05):
    """Points/lines of one class in two fish, matched one to one (Hungarian). A pair further apart than max_cost
    body lengths, or a spot without partner, costs max_cost. Result 0..1."""
    if not A and not B:
        return None
    n = max(len(A), len(B))
    C = np.full((n, n), max_cost)
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            C[i, j] = min(_spot_cost(a, b, kind_penalty), max_cost)
    r, c = linear_sum_assignment(C)
    return float(C[r, c].sum() / (n * max_cost))


def spots_matrix(spot_sets, classes, max_cost=0.15, kind_penalty=0.05):
    """spot_sets[i][class] = list of spots. Distance = mean over the classes present in either fish."""
    N = len(spot_sets)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            ds = [spot_set_distance(spot_sets[i].get(c, []), spot_sets[j].get(c, []), max_cost, kind_penalty)
                  for c in classes]
            ds = [d for d in ds if d is not None]
            D[i, j] = D[j, i] = float(np.mean(ds)) if ds else 0.0
    return D


# --- clusters ----------------------------------------------------------------------------------------------
def linkage(D, method='average'):
    if len(D) < 2:
        return None
    return hierarchy.linkage(squareform(D, checks=False), method=method)


def clusters_at(Z, n, similarity):
    """Cluster number of every fish when fish more similar than `similarity` (0..1 = 1 - distance) are joined.
    Clusters numbered 1.. by decreasing size."""
    if Z is None:
        return np.ones(n, int)
    raw = hierarchy.fcluster(Z, t=max(1 - similarity, 0.0) + 1e-12, criterion='distance')
    order = sorted(set(raw), key=lambda k: (-(raw == k).sum(), k))
    remap = {k: i + 1 for i, k in enumerate(order)}
    return np.array([remap[k] for k in raw])


def cophenetic_r(Z, D):
    if Z is None or len(D) < 3:
        return float('nan')
    return float(hierarchy.cophenet(Z, squareform(D, checks=False))[0])


def adjusted_rand(a, b):
    """Agreement of two groupings (1 = identical, ~0 = chance)."""
    from collections import Counter
    from math import comb
    n = len(a)
    if n < 2:
        return float('nan')
    pairs = Counter(zip(a, b))
    sa, sb = Counter(a), Counter(b)
    idx = sum(comb(v, 2) for v in pairs.values())
    ea, eb = sum(comb(v, 2) for v in sa.values()), sum(comb(v, 2) for v in sb.values())
    exp = ea * eb / comb(n, 2)
    mx = (ea + eb) / 2
    return float('nan') if mx == exp else float((idx - exp) / (mx - exp))


def mds2(D):
    """Classical MDS to 2-D, only for the picture of the clusters."""
    n = len(D)
    if n < 3:
        return np.c_[np.arange(n, dtype=float), np.zeros(n)]
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J
    ev, evec = np.linalg.eigh(B)
    i = np.argsort(ev)[::-1][:2]
    return evec[:, i] * np.sqrt(np.maximum(ev[i], 0))

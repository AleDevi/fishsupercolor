"""Reading the van der Bijl et al. 2025 annotations and placing them on the original photos.

Their annotations are drawn on a crop of each fish (extracted_fish/<id>.png, 800x300, transparent background,
body + whole caudal fin, no dorsal fin or gonopodium) and the 4 fixed landmarks are in that crop's pixels:
1 eye, 2 snout, 3 and 4 the upper and lower start of the caudal fin. The crop is a scaled and rotated piece of the
photo (not mirrored), so SIFT matches + RANSAC give the similarity transform crop -> photo."""
import csv
import io
import os
import re
import zipfile

import cv2
import numpy as np

ZIPS = {'fish': 'extracted_fish.zip', 'orange': 'carotenoid_coloration.zip', 'black': 'melanic_coloration_v2.zip',
        'landmarks': 'fixed_landmarks.zip'}
TRAINED = ('snout', 'eye', 'caudal_base_top', 'peduncle_end', 'caudal_base_bottom', 'tail_tip')   # landmarks they give
MATCH_SCALE = 0.5            # the photo is matched at half size: fast, and the crop is still smaller than the fish
MIN_INLIERS = 15
MAX_DIFF = 30                # mean grey-level difference crop vs photo under the mask; above it the match is wrong


def find_annotations(folder, depth=3):
    """The folder (folder itself or one of its subfolders) that holds photo_database.csv and the annotation zips,
    or None. The photos and the work folders are not searched."""
    need = ['photo_database.csv'] + list(ZIPS.values())
    base = folder.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(folder):
        if all(n in files for n in need):
            return root
        dirs[:] = [] if root.count(os.sep) - base >= depth else \
            [d for d in dirs if not d.lower().startswith(('work', 'foto', 'results', 'other_photos'))]
    return None


def uid_of(photo_name):
    """'20200818-IMG_2877.JPG' -> '20200818_IMG_2877' (their unique_id)."""
    return os.path.splitext(os.path.basename(photo_name))[0].replace('-', '_')


def read_database(path):
    with open(path, newline='', encoding='utf-8') as f:
        return {r['unique_id']: r for r in csv.DictReader(f)}


class Annotations:
    """The annotation zips, read in place (nothing is extracted)."""

    def __init__(self, folder):
        self.zips, self.index = {}, {}
        for kind, name in ZIPS.items():
            p = os.path.join(folder, name)
            if not os.path.exists(p):
                raise FileNotFoundError(f'{name} not found in {folder}')
            z = zipfile.ZipFile(p)
            self.zips[kind] = z
            self.index[kind] = {os.path.splitext(os.path.basename(n))[0]: n for n in z.namelist() if not n.endswith('/')}

    def has(self, uid):
        return all(uid in self.index[k] for k in ZIPS)

    def image(self, kind, uid):
        data = np.frombuffer(self.zips[kind].read(self.index[kind][uid]), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

    def landmarks(self, uid):
        return parse_landmarks(self.zips['landmarks'].read(self.index['landmarks'][uid]).decode('utf-8', 'replace'))


def parse_landmarks(text):
    """The 4x2 'landmarks.pixel' matrix of a fixed_landmarks file -> array (4, 2) in crop pixels."""
    block = text.split('landmarks.pixel', 1)[1].split('</landmarks.pixel>', 1)[0]
    rows = [r.split() for r in block.splitlines()[1:]]
    pts = [(float(r[1]), float(r[2])) for r in rows if len(r) == 3 and re.fullmatch(r'\d+', r[0])]
    if len(pts) != 4:
        raise ValueError('expected 4 landmarks')
    return np.array(pts)


_sift = None
def register(crop, photo):
    """Similarity transform (2x3) crop -> photo pixels, plus match statistics; None when no reliable match."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=4000)
    alpha = (crop[..., 3] > 0).astype(np.uint8) * 255
    g1 = cv2.cvtColor(np.ascontiguousarray(crop[..., :3]), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(photo, None, fx=MATCH_SCALE, fy=MATCH_SCALE, interpolation=cv2.INTER_AREA)
    g2 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k1, d1 = _sift.detectAndCompute(g1, cv2.erode(alpha, np.ones((5, 5), np.uint8)))
    k2, d2 = _sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return None, {'inliers': 0}
    good = [m[0] for m in cv2.BFMatcher().knnMatch(d1, d2, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < MIN_INLIERS:
        return None, {'inliers': len(good)}
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good]) / MATCH_SCALE
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6)
    n = int(inl.sum()) if inl is not None else 0
    if M is None or n < MIN_INLIERS:
        return None, {'inliers': n}
    mask = warp_mask(alpha, M, photo.shape)
    warped = cv2.warpAffine(np.ascontiguousarray(crop[..., :3]), M, (photo.shape[1], photo.shape[0]))
    diff = float(np.abs(warped.astype(int) - photo.astype(int))[mask].mean()) if mask.any() else 999.0
    stats = {'inliers': n, 'matches': len(good), 'scale': float(np.hypot(M[0, 0], M[1, 0])),
             'angle': float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 'diff': diff}
    return (M if diff <= MAX_DIFF else None), stats


def warp_mask(alpha, M, shape):
    return cv2.warpAffine(alpha, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR) > 127


def transform(points, M):
    p = np.asarray(points, float)
    return p @ M[:, :2].T + M[:, 2]


def annotate(ann, uid, photo):
    """Everything known about one photo, in photo pixels: masks (fish, body, caudal, orange, black) and landmarks
    (snout, eye, peduncle_end, tail_tip). None when the crop cannot be placed on the photo."""
    crop = ann.image('fish', uid)
    if crop is None or crop.ndim != 3 or crop.shape[2] != 4:
        return None, {'error': 'crop unreadable'}
    M, stats = register(crop, photo)
    if M is None:
        return None, stats
    fish = warp_mask(crop[..., 3], M, photo.shape)
    out = {'fish': fish}
    for k in ('orange', 'black'):
        m = ann.image(k, uid)
        out[k] = warp_mask(m[..., 3], M, photo.shape) & fish if m is not None and m.ndim == 3 and m.shape[2] == 4 \
            else np.zeros_like(fish)
    lm = transform(ann.landmarks(uid), M)
    eye, snout, fin_top, fin_bottom = lm
    peduncle = (fin_top + fin_bottom) / 2
    body, caudal = split_at_line(fish, fin_top, fin_bottom, snout)
    out.update(body=body, caudal=caudal)
    tail = tail_tip(caudal if caudal.any() else fish, snout, peduncle)
    out['points'] = {'snout': snout.tolist(), 'eye': eye.tolist(), 'peduncle_end': peduncle.tolist(),
                     'tail_tip': tail.tolist() if tail is not None else None}
    top, bottom = sorted((fin_top, fin_bottom), key=lambda q: q[1])     # dorsal side up in their photos
    out['points'].update(caudal_base_top=top.tolist(), caudal_base_bottom=bottom.tolist())
    return out, stats


def split_at_line(mask, a, b, head):
    """Pixels of `mask` on the side of the line a-b where `head` is (body) and on the other side (caudal fin)."""
    ys, xs = np.nonzero(mask)
    d = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
    side = np.sign((b[0] - a[0]) * (head[1] - a[1]) - (b[1] - a[1]) * (head[0] - a[0]))
    on_head = (d * side) >= 0
    body, caudal = np.zeros_like(mask), np.zeros_like(mask)
    body[ys[on_head], xs[on_head]] = True
    caudal[ys[~on_head], xs[~on_head]] = True
    return body, caudal


def tail_tip(mask, snout, peduncle):
    """Point of the mask farthest from the snout along the snout -> peduncle axis (total length end)."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    axis = np.subtract(peduncle, snout)
    axis = axis / (np.linalg.norm(axis) or 1)
    proj = (xs - snout[0]) * axis[0] + (ys - snout[1]) * axis[1]
    far = proj >= proj.max() - 1.5
    # several pixels at the far edge (a straight fin margin): take their middle
    return np.array([xs[far].mean(), ys[far].mean()])

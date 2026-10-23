"""Colour-class training inside a project: brush strokes per photo -> random forest -> areas per class.

Stroke files: addestramento/segni/<photo id>.png, one byte per pixel of the original photo (0 = no stroke,
k = brush k). Brushes come from the species profile: one per colour class, then 'corpo senza colore'
and 'reflection' (light reflections, left out of every area). Models: modelli/colours_vN.npz + .json."""
import datetime
import glob
import json
import os
import re

import cv2
import numpy as np

from . import pipeline, profile
from .engine import pixels

EXTRA = [{'key': 'base', 'label': 'Body without colour', 'display': '#C8C0A8', 'paint': '#FFF200'},
         {'key': 'reflection', 'label': 'Light reflection', 'display': '#FF40FF', 'paint': '#A349A4'}]
# Paint's own palette, so strokes drawn in Windows Paint can be imported (classes beyond these get display colours)
PAINT = ['#FF7F27', '#22B14C', '#3F48CC', '#ED1C24', '#00A2E8', '#B5E61D', '#FFC90E', '#7F7F7F']


def brushes(prof):
    """[{index, key, label, display, paint}] for the profile's colour classes + base + riflesso."""
    out = []
    for i, c in enumerate(prof['colour_classes']):
        out.append(dict(c, index=i + 1, paint=c.get('paint') or PAINT[i % len(PAINT)]))
    for j, c in enumerate(EXTRA):
        out.append(dict(c, index=len(out) + 1))
    return out


def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def vis_colour(c):
    """Colour to paint a class over the photo: very dark classes (black) are shown in yellow to stay visible."""
    r, g, b = hex_rgb(c['display'])
    return (255, 230, 0) if 0.3 * r + 0.59 * g + 0.11 * b < 70 else (r, g, b)


def strokes_path(project, photo):
    return os.path.join(project.dir(os.path.join('training', 'strokes')), project.photo_id(photo) + '.png')


def load_strokes(project, photo, shape):
    p = strokes_path(project, photo)
    if os.path.exists(p):
        s = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if s is not None and s.shape == tuple(shape[:2]):
            return s
    return np.zeros(shape[:2], np.uint8)


def save_strokes(project, photo, strokes):
    p = strokes_path(project, photo)
    if not strokes.any():
        if os.path.exists(p):
            os.remove(p)
        return
    cv2.imwrite(p + '.tmp.png', strokes)
    os.replace(p + '.tmp.png', p)


def annotated(project):
    """Photos with strokes: [(photo path, {brush index: pixels})]."""
    out = []
    for _, photo in project.photos():
        p = strokes_path(project, photo)
        if os.path.exists(p):
            s = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            counts = np.bincount(s.ravel(), minlength=256)
            out.append((photo, {int(k): int(counts[k]) for k in np.flatnonzero(counts[1:]) + 1}))
    return out


def corrected_image(project, photo):
    """The photo after the project's colour correction (what the model sees), cached as PNG."""
    cache = os.path.join(project.dir(os.path.join('training', 'images')),
                         project.photo_id(photo) + '_' + project.settings_hash()[:6] + '.png')
    if os.path.exists(cache):
        return cv2.cvtColor(cv2.imread(cache), cv2.COLOR_BGR2RGB)
    img = pipeline.load_image(photo)
    ref = pipeline.load_reference(project.reference_csv)
    corrected, _, _ = pipeline.correct_colour(img, project.settings, ref)
    if corrected.dtype != np.uint8:
        corrected = np.clip(corrected * 255 if corrected.max() <= 1.0 else corrected, 0, 255).astype(np.uint8)
    cv2.imwrite(cache, cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
    return corrected


def fish_mask(project, photo):
    manual = pipeline.manual_mask_path(project, photo)
    p = manual if os.path.exists(manual) else os.path.join(project.dir('masks'), project.photo_id(photo) + '.png')
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) if os.path.exists(p) else None
    return None if m is None else m > 0


def import_paint(project, photo, painted_path, prof, tol=60, min_change=40):
    """Strokes drawn over a copy of the photo in Paint (colours: brush['paint']) -> stroke map.
    A pixel counts only if it differs from the original photo and is close to a brush colour."""
    orig = pipeline.load_image(photo).astype(np.int32)
    painted = pipeline.load_image(painted_path)
    if painted.shape != orig.shape:
        painted = cv2.resize(painted, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
    painted = painted.astype(np.int32)
    changed = np.abs(painted - orig).sum(-1) > min_change
    st = load_strokes(project, photo, orig.shape)
    best = np.full(orig.shape[:2], tol + 1.0, np.float32)
    new = np.zeros(orig.shape[:2], np.uint8)
    for b in brushes(prof):
        d = np.sqrt(((painted - np.array(hex_rgb(b['paint']), np.int32)) ** 2).sum(-1)).astype(np.float32)
        take = changed & (d < best)
        new[take], best[take] = b['index'], d[take]
    st[new > 0] = new[new > 0]
    save_strokes(project, photo, st)
    return int((new > 0).sum())


def paint_counterpart(photo):
    """'IMG_1_strokes.png' next to the photo, if present."""
    stem = os.path.splitext(photo)[0]
    for ext in ('.png', '.PNG', '.bmp', '.jpg'):
        if os.path.exists(stem + '_strokes' + ext):
            return stem + '_strokes' + ext
    return None


def models_dir(project):
    return project.dir('models')


def list_models(project):
    out = []
    for p in sorted(glob.glob(os.path.join(models_dir(project), 'colours_v*.npz')),
                    key=lambda p: int(re.findall(r'v(\d+)', p)[-1])):
        meta_p = p[:-4] + '.json'
        meta = json.load(open(meta_p, encoding='utf-8')) if os.path.exists(meta_p) else {}
        out.append((os.path.basename(p)[:-4], meta))
    return out


def train(project, progress=None):
    """Train on every photo with strokes. Returns (model name, meta). Needs >= 2 brushes used."""
    prof = profile.load(project.settings['profile'])
    data = annotated(project)
    X, y, used = [], [], []
    for i, (photo, _) in enumerate(data):
        if progress:
            progress(i, len(data), os.path.basename(photo))
        mask = fish_mask(project, photo)
        if mask is None:
            continue
        img = corrected_image(project, photo)
        st = load_strokes(project, photo, img.shape)
        st[~mask & (st != len(brushes(prof)))] = 0       # strokes outside the fish are ignored (reflections excepted)
        xi, yi = pixels.training_pixels(img, mask | (st > 0), st)
        if len(yi):
            X.append(xi); y.append(yi); used.append(os.path.basename(photo))
    if not X:
        raise ValueError('No strokes on the fish: paint at least two classes on a few photos.')
    X, y = np.concatenate(X), np.concatenate(y)
    if len(np.unique(y)) < 2:
        raise ValueError('Strokes of at least two different brushes are needed.')
    forest = pixels.Forest.train(X, y)
    n = len(list_models(project)) and max(int(re.findall(r'v(\d+)', m)[-1]) for m, _ in list_models(project))
    name = f'colours_v{n + 1}'
    b = brushes(prof)
    forest.meta.update({
        'name': name, 'date': datetime.datetime.now().isoformat(timespec='minutes'),
        'annotator': project.settings.get('training', {}).get('annotator', ''),
        'profile': project.settings['profile'], 'features_version': pixels.FEATURES_VERSION,
        'classes': [{k: c[k] for k in ('index', 'key', 'label', 'display')} for c in b],
        'photos': used, 'pixels_per_class': {c['key']: int((y == c['index']).sum()) for c in b},
    })
    forest.save(os.path.join(models_dir(project), name + '.npz'))
    return name, forest.meta


_cache = {}
def load_model(project, name):
    if not profile.SAFE_NAME.fullmatch(name or ''):
        raise ValueError(f'invalid model name: {name!r}')
    p = os.path.join(models_dir(project), name + '.npz')
    key = (p, os.path.getmtime(p))
    if key not in _cache:
        _cache.clear()
        _cache[key] = pixels.Forest.load(p)
    return _cache[key]


def measure_colours(corrected, mask, forest, px_per_mm=None, merge_mm=0.0, min_spot_mm2=0.0):
    """Areas (px of the original photo) per class and the single spots; also the label map for the check image."""
    lab, box, s, conf = pixels.classify(corrected, mask, forest)
    to_full = 1 / s ** 2
    merge_px = int(round(merge_mm * px_per_mm * s)) if px_per_mm and merge_mm else 0
    min_px = max(1, int(min_spot_mm2 * px_per_mm ** 2 * s ** 2)) if px_per_mm and min_spot_mm2 else 1
    out = {}
    for c in forest.meta.get('classes', []):
        k = lab == c['index']
        out[c['key']] = {'area_px': round(float(k.sum()) * to_full, 1)}
        if c['key'] not in ('base', 'reflection'):
            out[c['key']]['spots_px'] = [round(v * to_full, 1) for v in pixels.spots(k, merge_px, min_px)]
    inside = conf[lab > 0]
    return out, (lab, box, s), float(inside.mean()) if inside.size else 0.0


def colour_overlay(path, corrected, labmap, forest, alpha=0.6):
    """Check image: the fish crop with every class painted in its display colour."""
    lab, (x0, y0, x1, y1), s = labmap
    crop = corrected[y0:y1, x0:x1]
    crop = cv2.resize(crop, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_AREA) if s != 1 else crop.copy()
    over = crop.copy()
    for c in forest.meta.get('classes', []):
        if c['key'] != 'base':
            over[lab == c['index']] = vis_colour(c)
    out = np.hstack([crop, (crop * (1 - alpha) + over * alpha).astype(np.uint8)])
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

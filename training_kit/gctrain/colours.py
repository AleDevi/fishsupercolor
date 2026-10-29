"""Colour model (orange, black, body without colour) from the van der Bijl carotenoid and melanic masks, in Guppy
Colour's own format: the same pixel features and random forest as the Training screen, so the model is imported
there and refined with brush strokes (Ritocco). Their photos have no colour chart: no colour correction.

Labels inside their fish mask: carotenoid -> orange, melanic -> black, the rest of the fish (caudal fin included)
-> body without colour; pixels in both masks are left out. Checked on the test photos (fish never trained on):
% orange and % black of the fish against their car_perc and mel_perc, and pixel overlap (IoU) per class."""
import csv
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import zipfile

import cv2
import numpy as np

from guppycolor import __version__ as app_version, profile, training
from guppycolor.engine import pixels

from . import prepare, vdb

KEYS = ('orange', 'black', 'base')
PER_PHOTO = 300             # training pixels per class and photo
PER_CLASS = 150000          # training pixels per class in all (class-balanced sample)
TREES = 60
METHOD = 3                  # 1: kit 0.2.0-0.2.1 (classes balanced); 2: weighted to their true share; 3: + class_share
SOURCE = 'van der Bijl et al. 2025, Nature Ecology & Evolution; data FRDR doi 10.20383/103.01281 (CC BY 4.0)'


def brush_index(profile_name='guppy_male'):
    """{key: brush number} of the profile, e.g. orange 1, black 2, base 4 for the guppy male."""
    return {b['key']: b['index'] for b in training.brushes(profile.load(profile_name))}


def labels(a, index):
    """Brush numbers on the photo: inside the fish only, overlap of orange and black left out."""
    lab = np.zeros(a['fish'].shape, np.uint8)
    lab[a['fish']] = index['base']
    lab[a['orange']] = index['orange']
    lab[a['black']] = index['black']
    lab[(a['orange'] & a['black']) | ~a['fish']] = 0
    return lab


def predict(forest, rgb, fish):
    """Class of every pixel of the photo (0 outside the fish)."""
    lab, box, s, _ = pixels.classify(rgb, fish, forest)
    x0, y0, x1, y1 = box
    x1, y1 = min(x1, fish.shape[1]), min(y1, fish.shape[0])
    full = np.zeros(fish.shape, np.uint8)
    full[y0:y1, x0:x1] = cv2.resize(lab, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    full[~fish] = 0
    return full


_ann = _forest = _index = None
def _init(annotations, model_path, index):
    global _ann, _forest, _index
    _ann, _index = vdb.Annotations(annotations), index
    _forest = pixels.Forest.load(model_path) if model_path else None


def _read(uid, path):
    photo = cv2.imread(path)
    if photo is None:
        return None, None
    a, _ = vdb.annotate(_ann, uid, photo)
    return a, (cv2.cvtColor(photo, cv2.COLOR_BGR2RGB) if a is not None else None)


def train_pixels(job):
    uid, path = job
    a, rgb = _read(uid, path)
    if a is None:
        return uid, None, None, None
    lab = labels(a, _index)
    counts = np.bincount(lab[lab > 0], minlength=256)
    X, y = pixels.training_pixels(rgb, a['fish'], labels(a, _index), per_class=PER_PHOTO, 
                                seed=int(hashlib.sha256(uid.encode()).hexdigest()[:6], 16))
    return uid, X, y, counts


def test_photo(job):
    uid, path = job
    a, rgb = _read(uid, path)
    if a is None:
        return None
    pred = predict(_forest, rgb, a['fish'])
    fish = a['fish'].sum()
    rec = {'uid': uid}
    for k in ('orange', 'black'):
        p, t = pred == _index[k], a[k] & a['fish']
        u = (p | t).sum()
        rec[f'pct_{k}_model'] = 100 * p.sum() / fish
        rec[f'pct_{k}_masks'] = 100 * t.sum() / fish
        rec[f'iou_{k}'] = float((p & t).sum() / u) if u else float('nan')
    return rec


def _sample(y, n, seed=0):
    rng = np.random.default_rng(seed)
    keep = [rng.choice(np.flatnonzero(y == k), min(n, (y == k).sum()), replace=False) for k in np.unique(y)]
    return np.sort(np.concatenate(keep))


def package(dest, forest_npz, meta, profile_name):
    """A zip in the layout of Guppy Colour's Export model (kind colours)."""
    with open(forest_npz, 'rb') as f:
        w = f.read()
    m = json.dumps(meta, indent=1, ensure_ascii=False).encode('utf-8')
    sha = lambda b: hashlib.sha256(b).hexdigest()
    manifest = {'kind': 'colours', 'name': meta['name'], 'profile': profile_name, 'app_version': app_version,
                'exported': datetime.datetime.now().isoformat(timespec='minutes'), 'project': 'GuppyTraining kit',
                'sha256': {'model.npz': sha(w), 'model.json': sha(m)}}
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=1, ensure_ascii=False))
        z.writestr('model.npz', w)
        z.writestr('model.json', m)
    return dest


def _stats(model, truth):
    ok = ~(np.isnan(model) | np.isnan(truth))
    model, truth = model[ok], truth[ok]
    if len(model) < 2:
        return {'n': int(len(model))}
    d = model - truth
    return {'n': int(len(d)), 'bias': float(d.mean()), 'mae': float(np.abs(d).mean()),
            'r': float(np.corrcoef(model, truth)[0, 1])}


def run(work, annotations, photos, log=print, profile_name='guppy_male', workers=None):
    """Writes work/colours/ (model, test results) and work/colours_vanderbijl.zip. Returns the colour summary."""
    out = os.path.join(work, 'colours')
    os.makedirs(out, exist_ok=True)
    summary_path = os.path.join(out, 'colour_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            done = json.load(f)
        if done.get('method') == METHOD:
            log('  colour model already trained')
            return done
        log('  colour model made by an earlier kit version: training it again')
    index = brush_index(profile_name)
    db = vdb.read_database(os.path.join(annotations, 'photo_database.csv'))
    found = prepare.find_photos(photos)
    with open(os.path.join(work, 'photos.csv'), newline='', encoding='utf-8') as f:
        recs = [r for r in csv.DictReader(f) if r['ok'] == 'True' and r['uid'] in found]
    jobs = {s: [(r['uid'], found[r['uid']]) for r in recs if r['split'] == s] for s in ('train', 'test')}
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    log(f'Colour model: reading the orange and black masks of {len(jobs["train"])} training photos')
    X, y, used, counts = [], [], [], np.zeros(256, np.int64)
    with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(annotations, None, index)) as pool:
        for i, (uid, xi, yi, ci) in enumerate(pool.imap_unordered(train_pixels, jobs['train'], chunksize=4), 1):
            if xi is not None and len(yi):
                X.append(xi); y.append(yi); used.append(uid); counts += ci
            if i % 200 == 0 or i == len(jobs['train']):
                log(f'  {i} of {len(jobs["train"])} photos')
    X, y = np.concatenate(X), np.concatenate(y)
    keep = _sample(y, PER_CLASS)
    log(f'  growing the forest ({TREES} trees) on {len(keep)} pixels')
    # Forest.train balances the classes; weighting each pixel by its class's share of the fish brings back the true
    # proportions (orange and black are a small part of a fish), so the model does not over-predict them
    prior = counts / max(counts.sum(), 1)
    forest = pixels.Forest.train(X[keep], y[keep], trees=TREES, sample_weight=prior[y[keep]].astype(np.float64))
    names = {b['key']: b for b in training.brushes(profile.load(profile_name))}
    pool_keep = keep[_sample(y[keep], training.POOL_PER_CLASS, seed=1)]
    forest.meta.update({
        'name': 'colours_vanderbijl', 'date': datetime.datetime.now().isoformat(timespec='minutes'),
        'annotator': 'van der Bijl 2025 masks', 'profile': profile_name, 'features_version': pixels.FEATURES_VERSION,
        'classes': [{k: names[c][k] for k in ('index', 'key', 'label', 'display')} for c in KEYS],
        'photos': sorted(used), 'pixels_per_class': {c: int((y == index[c]).sum()) for c in KEYS},
        'pool_pixels': int(len(pool_keep)),
        'class_share': {c: round(float(prior[index[c]]), 6) for c in KEYS}, 'source': SOURCE, 'colour_correction': 'none (no chart in the photos)',
        'trained_with': {'trees': TREES, 'depth': 14, 'pixels_per_photo_and_class': PER_PHOTO,
                         'pixels_per_class': PER_CLASS,
                         'class_weights': 'true share of the fish', 'method': METHOD},
    })
    model = os.path.join(out, 'colours_vanderbijl.npz')
    forest.save(model, pool={'pool_X': X[pool_keep].astype(np.float32), 'pool_y': y[pool_keep].astype(np.uint8),
                             'pool_own': np.zeros(len(pool_keep), bool)})
    package(os.path.join(work, 'colours_vanderbijl.zip'), model, forest.meta, profile_name)
    log(f'Colour model: checking it on {len(jobs["test"])} test photos')
    rows = []
    with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(annotations, model, index)) as pool:
        for rec in pool.imap_unordered(test_photo, jobs['test'], chunksize=2):
            if rec:
                r = db[rec['uid']]
                num = lambda v: float(v) if v not in ('', 'NA', None) else float('nan')
                rec.update(car_perc=num(r.get('car_perc')), mel_perc=num(r.get('mel_perc')),
                           mel_perc_v2=num(r.get('mel_perc_v2')))
                rows.append(rec)
    rows.sort(key=lambda r: r['uid'])
    cols = ['uid', 'pct_orange_model', 'pct_orange_masks', 'car_perc', 'iou_orange',
            'pct_black_model', 'pct_black_masks', 'mel_perc', 'mel_perc_v2', 'iou_black']
    with open(os.path.join(out, 'colour_test_results.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in rows:
            wr.writerow(['NA' if isinstance(r[c], float) and np.isnan(r[c]) else
                         (f'{r[c]:.4f}' if isinstance(r[c], float) else r[c]) for c in cols])
    col = lambda c: np.array([r[c] for r in rows], float)
    summary = {
        'method': METHOD, 'photos': len(rows), 'oob_accuracy': forest.meta.get('oob_accuracy'),
        'orange_vs_car_perc': _stats(col('pct_orange_model'), col('car_perc')),
        'black_vs_mel_perc': _stats(col('pct_black_model'), col('mel_perc')),
        'black_vs_mel_perc_v2': _stats(col('pct_black_model'), col('mel_perc_v2')),
        'orange_vs_masks': _stats(col('pct_orange_model'), col('pct_orange_masks')),
        'black_vs_masks': _stats(col('pct_black_model'), col('pct_black_masks')),
        'iou_orange_median': float(np.nanmedian(col('iou_orange'))) if rows else None,
        'iou_black_median': float(np.nanmedian(col('iou_black'))) if rows else None,
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    return summary

"""Named landmarks (from the species profile): clicked by hand in screen 8, or predicted by a YOLO-pose model
trained on those clicks. Standard length (SL) = snout to end of the caudal peduncle.

Clicks: landmarks/<photo id>.json = {"points": {key: [x, y]}, "annotator": ...} in pixels of the original photo.
Models: models/landmarks_vN/weights/best.pt + models/landmarks_vN.json (landmark keys, date, photos, SHA-256)."""
import datetime
import glob
import json
import os
import re
import shutil

import cv2
import numpy as np

from . import pipeline, profile

SL_FROM, SL_TO = 'snout', 'peduncle_end'
BASE = 'yolo11n-pose.pt'                    # starting weights, checked by engine/segment.ensure_weights
BASE_SHA256 = '869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0'
CROP_MARGIN = 0.12
MIN_CONF = 0.5


def landmark_list(prof):
    return prof.get('landmarks', [])


def path(project, photo):
    return os.path.join(project.dir('landmarks'), project.photo_id(photo) + '.json')


def load(project, photo):
    p = path(project, photo)
    if not os.path.exists(p):
        return {}
    with open(p, encoding='utf-8') as f:
        return {k: [float(v[0]), float(v[1])] for k, v in json.load(f).get('points', {}).items()}


def save(project, photo, points):
    p = path(project, photo)
    if not points:
        if os.path.exists(p):
            os.remove(p)
        return
    points = {k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in points.items()}
    data = {'points': points, 'annotator': project.settings.get('training', {}).get('annotator', ''),
            'date': datetime.datetime.now().isoformat(timespec='minutes')}
    with open(p + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    os.replace(p + '.tmp', p)
    r = project.load_result(photo)                 # SL in the results straight away, no new analysis needed
    if r is not None:
        sl = standard_length_px(points)
        r.update(landmarks=points, landmarks_source='manual')
        if sl:
            r['standard_length_px'] = round(sl, 1)
        tl = pipeline.landmarks_tl(points)
        if tl and r.get('total_length_source') != 'outline':     # the fin is not all in the outline
            r['total_length_px'], r['total_length_source'] = round(tl, 1), 'landmarks'
        elif not tl and r.get('total_length_source') == 'landmarks':
            r.pop('total_length_px', None); r.pop('total_length_source', None)
        r['flags'] = ';'.join(f for f in (r.get('flags') or '').split(';') if f and f != 'sl_landmarks_missing')
        project.save_result(photo, r)


def annotated(project):
    return [(ph, load(project, ph)) for _, ph in project.photos() if os.path.exists(path(project, ph))]


def standard_length_px(points):
    if SL_FROM in points and SL_TO in points:
        return float(np.hypot(*(np.subtract(points[SL_FROM], points[SL_TO]))))
    return None


def crop_box(mask, shape, margin=CROP_MARGIN):
    ys, xs = np.nonzero(mask)
    pad = int(margin * max(np.ptp(xs), np.ptp(ys)))
    return (int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)), int(min(shape[1], xs.max() + pad + 1)),
            int(min(shape[0], ys.max() + pad + 1)))


# ---------------------------------------------------------------- model
def list_models(project):
    out = []
    for p in sorted(glob.glob(os.path.join(project.dir('models'), 'landmarks_v*.json')),
                    key=lambda p: int(re.findall(r'v(\d+)', p)[-1])):
        with open(p, encoding='utf-8') as f:
            out.append((os.path.basename(p)[:-5], json.load(f)))
    return out


def model_weights(project, name):
    if not profile.SAFE_NAME.fullmatch(name or ''):
        raise ValueError(f'invalid model name: {name!r}')
    return os.path.join(project.dir('models'), name, 'weights', 'best.pt')


def build_dataset(project, out, keys, val_every=5):
    """YOLO-pose dataset of fish crops: class 0 = fish, one keypoint per landmark (visibility 0 when not clicked)."""
    if os.path.isdir(out):
        shutil.rmtree(out)
    n = 0
    for i, (photo, pts) in enumerate(annotated(project)):
        mask = pipeline_mask(project, photo)
        if mask is None or not pts:
            continue
        img = pipeline.load_image(photo)
        x0, y0, x1, y1 = crop_box(mask, img.shape)
        crop = img[y0:y1, x0:x1]
        w, h = x1 - x0, y1 - y0
        split = 'val' if i % val_every == val_every - 1 else 'train'
        for d in ('images', 'labels'):
            os.makedirs(os.path.join(out, d, split), exist_ok=True)
        stem = project.photo_id(photo)
        cv2.imwrite(os.path.join(out, 'images', split, stem + '.jpg'), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        bx, by, bw, bh = (xs.min() + xs.max()) / 2 / w, (ys.min() + ys.max()) / 2 / h, np.ptp(xs) / w, np.ptp(ys) / h
        kp = []
        for k in keys:
            if k in pts:
                kp += [(pts[k][0] - x0) / w, (pts[k][1] - y0) / h, 2]
            else:
                kp += [0, 0, 0]
        with open(os.path.join(out, 'labels', split, stem + '.txt'), 'w') as f:
            f.write(' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in [0, bx, by, bw, bh] + kp) + '\n')
        n += 1
    if not os.path.isdir(os.path.join(out, 'images', 'val')):        # tiny sets: validate on the training photos
        shutil.copytree(os.path.join(out, 'images', 'train'), os.path.join(out, 'images', 'val'))
        shutil.copytree(os.path.join(out, 'labels', 'train'), os.path.join(out, 'labels', 'val'))
    yaml = os.path.join(out, 'data.yaml')
    with open(yaml, 'w', encoding='utf-8') as f:
        f.write(f"path: {json.dumps(os.path.abspath(out))}\ntrain: images/train\nval: images/val\n"
                f"kpt_shape: [{len(keys)}, 3]\nflip_idx: {list(range(len(keys)))}\nnames:\n  0: fish\n")
    return yaml, n


def pipeline_mask(project, photo):
    from .training import fish_mask
    return fish_mask(project, photo)


def train(project, epochs=150, imgsz=640, progress=None):
    """Train a YOLO-pose model on the clicked landmarks. GPU used when present. Returns (name, meta)."""
    from .engine import segment
    import hashlib
    prof = profile.load(project.settings['profile'])
    keys = [l['key'] for l in landmark_list(prof)]
    if progress:
        progress(0, epochs, 'Preparing the photos')
    yaml, n = build_dataset(project, os.path.join(project.dir('training'), 'landmarks_dataset'), keys)
    if n < 2:
        raise ValueError('Click the landmarks on at least a few photos first (10–30 are a good start).')
    base = segment.ensure_file(BASE, BASE_SHA256)
    segment.configure_ultralytics()
    from ultralytics import YOLO
    import torch
    nums = [int(re.findall(r'v(\d+)', m)[-1]) for m, _ in list_models(project)]
    name = f'landmarks_v{max(nums, default=0) + 1}'
    model = YOLO(base)
    if progress:
        model.add_callback('on_train_epoch_end', lambda tr: progress(
            tr.epoch + 1, epochs, f'Round {tr.epoch + 1} of {epochs} (it may stop earlier when it no longer improves)'))
    model.train(data=yaml, epochs=epochs, imgsz=imgsz, batch=8, workers=0, fliplr=0.0, mosaic=0.0, amp=False,
                     device=0 if torch.cuda.is_available() else 'cpu', project=project.dir('models'), name=name,
                     exist_ok=True, plots=False, verbose=False, patience=40)
    w = model_weights(project, name)
    meta = {'name': name, 'date': datetime.datetime.now().isoformat(timespec='minutes'),
            'annotator': project.settings.get('training', {}).get('annotator', ''), 'keys': keys, 'photos': n,
            'gpu': torch.cuda.is_available(), 'sha256': hashlib.sha256(open(w, 'rb').read()).hexdigest()}
    with open(os.path.join(project.dir('models'), name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    return name, meta


_cache = {}
def load_model(project, name):
    import hashlib
    w = model_weights(project, name)
    meta = dict(list_models(project)).get(name, {})
    key = (w, os.path.getmtime(w))
    if key not in _cache:
        if hashlib.sha256(open(w, 'rb').read()).hexdigest() != meta.get('sha256'):
            raise ValueError(f'{name}: the model file differs from the one trained, not used')
        from .engine import segment
        segment.configure_ultralytics()
        from ultralytics import YOLO
        _cache.clear()
        _cache[key] = (YOLO(w), meta.get('keys', []))
    return _cache[key]


def predict(project, name, img, mask):
    """{key: [x, y]} in photo pixels for landmarks predicted with confidence >= MIN_CONF."""
    model, keys = load_model(project, name)
    x0, y0, x1, y1 = crop_box(mask, img.shape)
    r = model.predict(img[y0:y1, x0:x1, ::-1], verbose=False, conf=0.1)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return {}
    i = int(r.boxes.conf.argmax())
    xy = r.keypoints.xy[i].cpu().numpy()
    conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(keys))
    return {k: [float(xy[j, 0] + x0), float(xy[j, 1] + y0)] for j, k in enumerate(keys)
            if j < len(xy) and conf[j] >= MIN_CONF}

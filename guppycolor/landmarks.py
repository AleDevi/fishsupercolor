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
from .engine import segment

SL_FROM, SL_TO = 'snout', 'peduncle_end'
BASE = 'yolo11n-pose.pt'                    # starting weights, checked by engine/segment.ensure_weights
BASE_SHA256 = '869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0'
CROP_MARGIN = 0.15          # around the outline, as a share of the fish length
CROP_MARGIN_LONG = 0.30     # at both ends of the fish: the transparent caudal fin is often outside the outline
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
        update_result(project, photo, {})
        return
    points = {k: [round(float(v[0]), 1), round(float(v[1]), 1)] for k, v in points.items()}
    data = {'points': points, 'annotator': project.settings.get('training', {}).get('annotator', ''),
            'date': datetime.datetime.now().isoformat(timespec='minutes')}
    with open(p + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    os.replace(p + '.tmp', p)
    update_result(project, photo, points)


def update_result(project, photo, points):
    """Sizes in the results straight away (SL, TL, end of the body, depth), no new analysis needed."""
    from . import sizes, training
    r = project.load_result(photo)
    if r is None:
        return
    if points:
        r.update(landmarks=points, landmarks_source='manual')
    else:
        r.pop('landmarks', None); r.pop('landmarks_source', None)
    if 'outline' not in r:          # analysed before 0.10: what the outline gave is what is in the result
        r['outline'] = {k: r[k] for k in ('body_area_px', 'max_depth_px', 'mask_length_px') if k in r}
        r['outline']['fin_found'] = bool(r.get('caudal_in_mask'))
    mask = training.fish_mask(project, photo)
    flags = sizes.update(r, mask, points, (r.get('flags') or '').split(';'))
    if r.get('colours') and project.settings['colours'].get('model') and mask is not None:
        from . import pipeline                  # the colours start behind the eye: measure them again
        flags = [f for f in flags if f not in pipeline.COLOUR_FLAGS]
        flags += pipeline.measure_colours(project, r, training.corrected_image(project, photo), mask)
    r['flags'] = ';'.join(flags)
    project.save_result(photo, r)


def annotated(project):
    return [(ph, load(project, ph)) for _, ph in project.photos() if os.path.exists(path(project, ph))]


def standard_length_px(points):
    if SL_FROM in points and SL_TO in points:
        return float(np.hypot(*(np.subtract(points[SL_FROM], points[SL_TO]))))
    return None


def crop_box(mask, shape, margin=CROP_MARGIN, margin_long=CROP_MARGIN_LONG):
    """(x0, y0, x1, y1) around the fish, with more room at the ends of its long side (head and tail)."""
    ys, xs = np.nonzero(mask)
    w, h = np.ptp(xs), np.ptp(ys)
    pad = margin * max(w, h)
    long_pad = margin_long * max(w, h)
    px, py = (long_pad, pad) if w >= h else (pad, long_pad)
    return (int(max(0, xs.min() - px)), int(max(0, ys.min() - py)), int(min(shape[1], xs.max() + px + 1)),
            int(min(shape[0], ys.max() + py + 1)))


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


def train(project, epochs=150, imgsz=640, progress=None, start_from=None):
    """Train a YOLO-pose model on the clicked landmarks. GPU used when present. Returns (name, meta).
    start_from: a landmark model of the project (e.g. one pre-trained on many photos and imported) that the clicks
    refine ("Ritocco") instead of starting from the general yolo11n-pose: fewer rounds, its image size."""
    import hashlib
    prof = profile.load(project.settings['profile'])
    keys = [l['key'] for l in landmark_list(prof)]
    if progress:
        progress(0, epochs, 'Preparing the photos')
    yaml, n = build_dataset(project, os.path.join(project.dir('training'), 'landmarks_dataset'), keys)
    if n < 2:
        raise ValueError('Click the landmarks on at least a few photos first (10–30 are a good start).')
    clicked = {k for _, pts in annotated(project) for k in pts}
    untrained = set()
    if start_from:
        _, _, base_meta = load_model(project, start_from)      # checks its SHA-256 too
        theirs = base_meta.get('keys') or []
        if not theirs or not set(theirs) <= set(keys):
            raise ValueError(f'{start_from} has other landmarks than this project')
        base = model_weights(project, start_from)
        imgsz = int(base_meta.get('imgsz', imgsz))
        epochs = min(epochs, 60)
        # landmarks it never saw (e.g. ones added to the profile later) are learnt from the clicks only
        untrained = set(base_meta.get('not_trained') or []) | (set(keys) - set(theirs))
    else:
        base = segment.ensure_file(BASE, BASE_SHA256)
        untrained = set(keys)
    segment.configure_ultralytics()
    from ultralytics import YOLO
    nums = [int(re.findall(r'v(\d+)', m)[-1]) for m, _ in list_models(project)]
    name = f'landmarks_v{max(nums, default=0) + 1}'
    model = YOLO(base)
    if progress:
        model.add_callback('on_train_epoch_end', lambda tr: progress(
            tr.epoch + 1, epochs, f'Round {tr.epoch + 1} of {epochs} (it may stop earlier when it no longer improves)'))
    # Ritocco: gentle steps and the backbone kept, so a few photos refine the pre-trained model instead of undoing it
    # (nbs = batch: a step after every batch; by default ultralytics waits for 64 images, more than a few photos)
    gentle = dict(optimizer='AdamW', lr0=0.0005, warmup_epochs=0, freeze=10, nbs=8, patience=20) if start_from else \
        dict(patience=40)
    model.train(data=yaml, epochs=epochs, imgsz=imgsz, batch=8, workers=0, fliplr=0.0, mosaic=0.0, amp=False,
                device=segment.device(), project=project.dir('models'), name=name,
                exist_ok=True, plots=False, verbose=False, **gentle)
    w = model_weights(project, name)
    with open(w, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    meta = {'name': name, 'date': datetime.datetime.now().isoformat(timespec='minutes'),
            'annotator': project.settings.get('training', {}).get('annotator', ''), 'keys': keys, 'photos': n,
            'gpu': segment.device() != 'cpu', 'sha256': sha, 'imgsz': imgsz,
            'not_trained': [k for k in keys if k in untrained and k not in clicked]}
    if start_from:
        meta['started_from'] = start_from
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
        segment.configure_ultralytics()
        from ultralytics import YOLO
        _cache.clear()
        skip = set(meta.get('not_trained') or [])       # keys the model never saw (e.g. the training kit's)
        _cache[key] = (YOLO(w), [k if k not in skip else None for k in meta.get('keys', [])], meta)
    return _cache[key]


def predict(project, name, img, mask):
    """{key: [x, y]} in photo pixels for landmarks predicted with confidence >= MIN_CONF."""
    model, keys, meta = load_model(project, name)
    x0, y0, x1, y1 = crop_box(mask, img.shape)
    r = model.predict(img[y0:y1, x0:x1, ::-1], imgsz=int(meta.get('imgsz', 640)), verbose=False, conf=0.1, device=segment.device())[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return {}
    i = int(r.boxes.conf.argmax())
    xy = r.keypoints.xy[i].cpu().numpy()
    conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(keys))
    return {k: [float(xy[j, 0] + x0), float(xy[j, 1] + y0)] for j, k in enumerate(keys)
            if k and j < len(xy) and conf[j] >= MIN_CONF}

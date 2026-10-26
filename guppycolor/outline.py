"""Outline models: a YOLO segmentation model that draws the parts of the fish (e.g. body and caudal fin) on the
whole photo, imported as a .zip (Export model / training kit) and used by the analysis instead of the zero-training
detector (YOLOE + SAM). The classes are the profile's body parts; 'body' is required."""
import glob
import hashlib
import json
import os
import re

import cv2
import numpy as np

MIN_CONF = 0.25
JOIN_PX = 15        # a part counts as the fish's when it touches the body (after a dilation of this many pixels)


def model_weights(project, name):
    return os.path.join(project.dir('models'), name, 'best.pt')


def list_models(project):
    out = []
    for p in sorted(glob.glob(os.path.join(project.dir('models'), 'outline_v*.json')),
                    key=lambda p: int(re.findall(r'v(\d+)', p)[-1])):
        with open(p, encoding='utf-8') as f:
            out.append((os.path.basename(p)[:-5], json.load(f)))
    return out


_cache = {}


def load_model(project, name):
    w = model_weights(project, name)
    meta = dict(list_models(project)).get(name, {})
    key = (w, os.path.getmtime(w))
    if key not in _cache:
        with open(w, 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() != meta.get('sha256'):
                raise ValueError(f'{name}: the model file differs from the one imported, not used')
        from .engine import segment
        segment.configure_ultralytics()
        from ultralytics import YOLO
        _cache.clear()
        _cache[key] = (YOLO(w), meta)
    return _cache[key]


def combine(items, classes):
    """items: [(class index, confidence, mask)] -> {class name: mask} for one fish: the most confident body, and the
    other parts that touch it (most confident one per class). {} when there is no body."""
    bodies = [it for it in items if classes[it[0]] == 'body']
    if not bodies:
        return {}
    _, conf, body = max(bodies, key=lambda it: it[1])
    near = cv2.dilate(body.astype(np.uint8), np.ones((2 * JOIN_PX + 1,) * 2, np.uint8)) > 0
    parts = {'body': body}
    for c, cf, m in sorted(items, key=lambda it: -it[1]):
        name = classes[c]
        if name != 'body' and name not in parts and (m & near).any():
            parts[name] = m & ~body
    parts['_conf'] = conf
    return parts


def predict(project, name, img):
    """Parts of the fish in the photo (RGB): ({part: bool mask at photo size}, confidence), or ({}, 0)."""
    model, meta = load_model(project, name)
    classes = meta.get('classes') or ['body']
    h, w = img.shape[:2]
    long_side = int(meta.get('long_side', 1280))
    s = min(1.0, long_side / max(h, w))
    small = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    r = model.predict(small[..., ::-1], imgsz=int(meta.get('imgsz', 1024)), conf=MIN_CONF, retina_masks=True,
                      verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return {}, 0.0
    data = r.masks.data.cpu().numpy() > 0.5
    items = [(int(c), float(cf), data[i]) for i, (c, cf) in enumerate(zip(r.boxes.cls.cpu().numpy(),
                                                                           r.boxes.conf.cpu().numpy()))
             if int(c) < len(classes)]
    parts = combine(items, classes)
    if not parts:
        return {}, 0.0
    conf = parts.pop('_conf')
    full = {k: cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            for k, m in parts.items()}
    return full, conf

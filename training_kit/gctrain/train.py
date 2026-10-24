"""Training of the outline model (YOLO-seg: body, caudal fin) and of the landmark model (YOLO-pose, the profile's
landmarks) on the datasets of prepare.py. The size of the run follows the machine: a full run on an NVIDIA card
with enough memory, a short check run on the processor. A run that stops is resumed where it was.

The outline model has a second round: the training photos it gets most wrong are added twice and it trains a
little more from its best weights; the round is kept only if it does better on the validation photos."""
import glob
import json
import os
import shutil

import cv2
import numpy as np

from guppycolor.engine import segment

BASES = {   # ultralytics release v8.4.0 assets, checked before they are opened
    'yolo11s-seg.pt': '1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3',
    'yolo11n-pose.pt': '869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0',
}
HARD_SHARE = 0.15


def machine():
    """'gpu' with its memory in GB, or 'cpu'."""
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return {'device': 0, 'name': p.name, 'memory_gb': round(p.total_memory / 2 ** 30, 1)}
    return {'device': 'cpu', 'name': 'processor', 'memory_gb': 0}


def plan(m, quick=False):
    """Epochs, image sizes and batches for this machine. quick=True: a few minutes, only to check the kit runs."""
    if quick or m['device'] == 'cpu':
        return {'outline': dict(epochs=2, imgsz=640, batch=2, fraction=0.05),
                'landmarks': dict(epochs=2, imgsz=320, batch=4, fraction=0.05), 'hard_epochs': 1}
    big = m['memory_gb'] >= 10
    return {'outline': dict(epochs=80, imgsz=1024 if big else 640, batch=8 if big else 4),
            'landmarks': dict(epochs=150, imgsz=640, batch=32 if big else 16), 'hard_epochs': 30}


def _amp_ok(model=None):
    """In place of ultralytics' check_amp, which downloads yolo11n.pt (unchecked) to test half precision: cards from
    the 20xx series on (compute capability 7.5+) do it well, older ones train in full precision."""
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 5)


def _yolo(weights):
    from ultralytics import YOLO
    from ultralytics.engine import trainer
    trainer.check_amp = _amp_ok
    return YOLO(weights)


def _train(base, data, runs, name, device, log, **kw):
    """One ultralytics training run, resumed from last.pt when an earlier run of the same name stopped."""
    run = os.path.join(runs, name)
    last, best = os.path.join(run, 'weights', 'last.pt'), os.path.join(run, 'weights', 'best.pt')
    if os.path.exists(os.path.join(run, 'done.txt')):
        log(f'  {name}: already trained')
        return best
    common = dict(data=data, project=runs, name=name, exist_ok=True, device=device, workers=0 if device == 'cpu' else 4,
                  amp=device != 'cpu', plots=False, verbose=False, patience=30, cache=False)
    if os.path.exists(last):
        log(f'  {name}: resuming the interrupted training')
        try:
            _yolo(last).train(resume=True, **common)
        except Exception as e:          # resume refuses a run that had finished its epochs: start this step again
            log(f'  {name}: could not resume ({e}); starting this step again')
            shutil.rmtree(run, ignore_errors=True)
            _yolo(base).train(**common, **kw)
    else:
        model = _yolo(base)
        model.add_callback('on_train_epoch_end', lambda tr: log(f'  {name}: round {tr.epoch + 1} of {tr.epochs}'))
        model.train(**common, **kw)
    open(os.path.join(run, 'done.txt'), 'w').close()
    return best


def _val_score(weights, data, device, imgsz):
    m = _yolo(weights).val(data=data, device=device, imgsz=imgsz, plots=False, verbose=False, workers=0)
    return float(m.seg.map) if hasattr(m, 'seg') else float(m.box.map)


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def _label_mask(label_path, shape):
    m = np.zeros(shape[:2], np.uint8)
    h, w = shape[:2]
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            v = line.split()
            if len(v) >= 7:
                pts = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.fillPoly(m, [pts], 1)
    return m > 0


def hard_examples(weights, ds, device, imgsz, share=HARD_SHARE):
    """The training photos where the predicted fish overlaps least with the van der Bijl one."""
    model = _yolo(weights)
    scores = []
    for img_path in sorted(glob.glob(os.path.join(ds, 'images', 'train', '*.jpg'))):
        if '_hard' in os.path.basename(img_path):
            continue
        img = cv2.imread(img_path)
        lab = os.path.join(ds, 'labels', 'train', os.path.basename(img_path)[:-4] + '.txt')
        truth = _label_mask(lab, img.shape)
        r = model.predict(img, imgsz=imgsz, device=device, verbose=False, retina_masks=True, conf=0.25)[0]
        pred = r.masks.data.cpu().numpy().any(0) if r.masks is not None else np.zeros_like(truth)
        if pred.shape != truth.shape:
            pred = cv2.resize(pred.astype(np.uint8), (truth.shape[1], truth.shape[0])) > 0
        scores.append((_iou(pred, truth), img_path, lab))
    scores.sort()
    return scores[:max(1, int(len(scores) * share))]


def train_outline(work, log, quick=False):
    m = machine()
    p = plan(m, quick)
    ds = os.path.join(work, 'outline')
    data = os.path.join(ds, 'data.yaml')
    runs = os.path.join(work, 'runs')
    base = segment.ensure_file('yolo11s-seg.pt', BASES['yolo11s-seg.pt'])
    log(f'Outline model on {m["name"]}')
    aug = dict(fliplr=0.5, degrees=10, scale=0.5, hsv_h=0.02, hsv_s=0.6, hsv_v=0.4, mosaic=1.0, close_mosaic=10)
    best1 = _train(base, data, runs, 'outline_1', m['device'], log, **p['outline'], **aug)
    final = best1
    if not os.path.exists(os.path.join(runs, 'outline_2', 'done.txt')):
        log('  looking for the training photos it gets most wrong')
        hard = hard_examples(best1, ds, m['device'], p['outline']['imgsz'])
        for _, img, lab in hard:
            stem = os.path.basename(img)[:-4]
            shutil.copyfile(img, os.path.join(ds, 'images', 'train', stem + '_hard.jpg'))
            shutil.copyfile(lab, os.path.join(ds, 'labels', 'train', stem + '_hard.txt'))
        log(f'  {len(hard)} difficult photos added twice; second round')
    kw = dict(p['outline'], epochs=p['hard_epochs'])
    best2 = _train(best1, data, runs, 'outline_2', m['device'], log, lr0=0.002, **kw, **aug)
    s1 = _val_score(best1, data, m['device'], p['outline']['imgsz'])
    s2 = _val_score(best2, data, m['device'], p['outline']['imgsz'])
    final = best2 if s2 >= s1 else best1
    log(f'  validation mAP: first round {s1:.3f}, second round {s2:.3f} -> keeping the '
        f'{"second" if final == best2 else "first"}')
    info = {'weights': final, 'imgsz': p['outline']['imgsz'], 'val_map_round1': s1, 'val_map_round2': s2, 'machine': m}
    with open(os.path.join(work, 'outline_model.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    return info


def train_landmarks(work, log, quick=False):
    m = machine()
    p = plan(m, quick)
    ds = os.path.join(work, 'landmarks')
    base = segment.ensure_file('yolo11n-pose.pt', BASES['yolo11n-pose.pt'])
    log(f'Landmark model on {m["name"]}')
    aug = dict(fliplr=0.5, degrees=10, scale=0.3, hsv_h=0.02, hsv_s=0.6, hsv_v=0.4, mosaic=0.0)
    best = _train(base, os.path.join(ds, 'data.yaml'), os.path.join(work, 'runs'), 'landmarks', m['device'], log,
                  **p['landmarks'], **aug)
    with open(os.path.join(ds, 'keys.json'), encoding='utf-8') as f:
        keys = json.load(f)
    info = {'weights': best, 'imgsz': p['landmarks']['imgsz'], 'machine': m, **keys}
    with open(os.path.join(work, 'landmarks_model.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    return info

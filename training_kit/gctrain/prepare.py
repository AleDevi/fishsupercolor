"""Training data from the van der Bijl photos: each crop is placed on its original photo (vdb.annotate), then
written as two YOLO datasets and a test set.

- outline/: whole photos (long side SEG_SIZE), polygons of class 0 body and 1 caudal fin -> YOLO-seg;
- landmarks/: crops around the fish made exactly like Guppy Colour makes them (landmarks.crop_box), one keypoint
  per landmark of the species profile, visibility 0 where van der Bijl has none (dorsal fin base, gonopodium) ->
  YOLO-pose, so the model can be imported in the app and refined there with the user's clicks;
- test/: per photo, the masks at SEG_SIZE and the landmarks in photo pixels, for evaluate.py.

Photos are split by fish (not by photo), so no fish is both trained on and tested on."""
import csv
import hashlib
import json
import multiprocessing as mp
import os
import shutil

import cv2
import numpy as np

from guppycolor import landmarks as app_landmarks, profile

from . import vdb

SEG_SIZE = 1280             # long side of the whole photos for the outline model
POSE_SIZE = 1024            # long side of the landmark crops (the app trains them at 640)
TEST_SHARE, VAL_SHARE = 0.10, 0.10
CLASSES = ['body', 'caudal']


def split_of(fish_id):
    h = int(hashlib.sha256(fish_id.encode('utf-8')).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'test' if h < TEST_SHARE else 'val' if h < TEST_SHARE + VAL_SHARE else 'train'


def find_photos(folder):
    out = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg')):
                out[vdb.uid_of(f)] = os.path.join(root, f)
    return out


def polygons(mask, scale):
    """Outer contours of a mask as YOLO polygons (normalised), the small specks left out."""
    small = cv2.resize(mask.astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = small.shape
    cnts, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    keep = [c for c in cnts if cv2.contourArea(c) >= 0.05 * max(cv2.contourArea(x) for x in cnts)] if cnts else []
    out = []
    for c in keep:
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append([v for x, y in c for v in (x / w, y / h)])
    return out


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(''.join(' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in line) + '\n'
                        for line in lines))


_ann = None
def _init(annotations):
    global _ann
    _ann = vdb.Annotations(annotations)


def one(job):
    uid, photo_path, split, out, keys = job
    rec = {'uid': uid, 'split': split, 'ok': False}
    photo = cv2.imread(photo_path)
    if photo is None:
        rec['error'] = 'photo unreadable'
        return rec
    a, stats = vdb.annotate(_ann, uid, photo)
    rec.update({k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    if a is None:
        rec.setdefault('error', 'crop not found in the photo')
        return rec
    h, w = photo.shape[:2]
    s = SEG_SIZE / max(h, w)
    img = cv2.resize(photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if split != 'test':
        cv2.imwrite(os.path.join(out, 'outline', 'images', split, uid + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = [[i] + p for i, k in enumerate(CLASSES) for p in polygons(a[k], s)]
        write_lines(os.path.join(out, 'outline', 'labels', split, uid + '.txt'), lines)
        # landmark crop, as the app makes it
        x0, y0, x1, y1 = app_landmarks.crop_box(a['fish'], photo.shape)
        crop = photo[y0:y1, x0:x1]
        cs = min(1.0, POSE_SIZE / max(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=cs, fy=cs, interpolation=cv2.INTER_AREA) if cs < 1 else crop
        cw, ch = x1 - x0, y1 - y0
        ys, xs = np.nonzero(a['fish'][y0:y1, x0:x1])
        line = [0, (xs.min() + xs.max()) / 2 / cw, (ys.min() + ys.max()) / 2 / ch, float(np.ptp(xs)) / cw,
                float(np.ptp(ys)) / ch]
        for k in keys:
            p = a['points'].get(k)
            line += [(p[0] - x0) / cw, (p[1] - y0) / ch, 2] if p is not None else [0.0, 0.0, 0]
        cv2.imwrite(os.path.join(out, 'landmarks', 'images', split, uid + '.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_lines(os.path.join(out, 'landmarks', 'labels', split, uid + '.txt'), [line])
    else:
        d = os.path.join(out, 'test')
        cv2.imwrite(os.path.join(d, uid + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for k in ('fish', 'body', 'caudal', 'orange', 'black'):
            m = cv2.resize(a[k].astype(np.uint8) * 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(d, f'{uid}_{k}.png'), m)
        with open(os.path.join(d, uid + '.json'), 'w', encoding='utf-8') as f:
            json.dump({'photo': photo_path, 'size': [w, h], 'points': a['points']}, f, indent=1)
    fish = a['fish'].sum()
    rec.update(ok=True, orange_pct=round(100 * a['orange'].sum() / fish, 3), black_pct=round(100 * a['black'].sum() / fish, 3))
    return rec


def run(annotations, photos, out, profile_name='guppy_male', workers=None, log=print, limit=None):
    """Builds out/outline, out/landmarks and out/test. Returns the path of the per-photo record CSV."""
    prof = profile.load(profile_name)
    keys = [l['key'] for l in app_landmarks.landmark_list(prof)]
    db = vdb.read_database(os.path.join(annotations, 'photo_database.csv'))
    found = find_photos(photos)
    jobs = []
    for uid, p in sorted(found.items()):
        if uid not in db:
            log(f'  {os.path.basename(p)}: not in photo_database.csv, skipped')
            continue
        jobs.append((uid, p, split_of(db[uid]['fish_id']), out, keys))
    if limit:                       # quick check: a few photos of each split
        jobs = [j for s in ('train', 'val', 'test') for j in [j for j in jobs if j[2] == s][:max(2, limit // 3)]]
    for d in ('outline/images/train', 'outline/images/val', 'outline/labels/train', 'outline/labels/val',
              'landmarks/images/train', 'landmarks/images/val', 'landmarks/labels/train', 'landmarks/labels/val', 'test'):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    log(f'{len(jobs)} photos: placing the van der Bijl annotations on them')
    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    recs = []
    with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(annotations,)) as pool:
        for i, rec in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            r = db[rec['uid']]
            rec.update(fish_id=r['fish_id'], replicate=r['replicate'], generation=r['generation'],
                       facing=r['facing_direction'], car_perc=r['car_perc'], mel_perc=r['mel_perc'],
                       manual_landmarks=r['manual_landmarks'])
            recs.append(rec)
            if i % 50 == 0 or i == len(jobs):
                log(f'  {i} of {len(jobs)} photos ({sum(x["ok"] for x in recs)} placed)')
    for ds in ('outline', 'landmarks'):         # tiny sets (quick check): validate on a few training photos
        val = os.path.join(out, ds, 'images', 'val')
        if not os.listdir(val):
            for f in sorted(os.listdir(os.path.join(out, ds, 'images', 'train')))[:3]:
                shutil.copyfile(os.path.join(out, ds, 'images', 'train', f), os.path.join(val, f))
                lab = f[:-4] + '.txt'
                shutil.copyfile(os.path.join(out, ds, 'labels', 'train', lab), os.path.join(out, ds, 'labels', 'val', lab))
    with open(os.path.join(out, 'outline', 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"path: {json.dumps(os.path.abspath(os.path.join(out, 'outline')))}\ntrain: images/train\n"
                f"val: images/val\nnames:\n" + ''.join(f'  {i}: {k}\n' for i, k in enumerate(CLASSES)))
    with open(os.path.join(out, 'landmarks', 'data.yaml'), 'w', encoding='utf-8') as f:
        f.write(f"path: {json.dumps(os.path.abspath(os.path.join(out, 'landmarks')))}\ntrain: images/train\n"
                f"val: images/val\nkpt_shape: [{len(keys)}, 3]\nflip_idx: {list(range(len(keys)))}\nnames:\n  0: fish\n")
    with open(os.path.join(out, 'landmarks', 'keys.json'), 'w', encoding='utf-8') as f:
        json.dump({'profile': profile_name, 'keys': keys}, f)
    cols = ['uid', 'split', 'ok', 'error', 'inliers', 'matches', 'scale', 'angle', 'diff', 'orange_pct', 'black_pct',
            'car_perc', 'mel_perc', 'fish_id', 'replicate', 'generation', 'facing', 'manual_landmarks']
    path = os.path.join(out, 'photos.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore', restval='NA')
        wr.writeheader()
        wr.writerows(sorted(recs, key=lambda r: r['uid']))
    return path

"""Check of the trained models on the van der Bijl test photos (fish never seen in training), against their
annotations: overlap (IoU) of the fish, body and caudal fin; distance of each landmark from theirs, as a share of
their standard length; SL and TL from our landmarks against SL and TL from theirs. Optional extra folders of photos
without annotations (other setups) are only drawn, so a person can judge them by eye."""
import base64
import csv
import glob
import html
import json
import os

import cv2
import numpy as np

from guppycolor import landmarks as app_landmarks

SL_FROM, SL_TO = app_landmarks.SL_FROM, app_landmarks.SL_TO


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float('nan')


def predict_outline(model, img, imgsz, device):
    """{'body': mask, 'caudal': mask, 'fish': mask} at the size of img: per class, the most confident instance."""
    r = model.predict(img, imgsz=imgsz, device=device, verbose=False, retina_masks=True, conf=0.25)[0]
    h, w = img.shape[:2]
    out = {k: np.zeros((h, w), bool) for k in ('body', 'caudal')}
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy() > 0.5
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for c, k in enumerate(('body', 'caudal')):
            idx = np.where(cls == c)[0]
            if len(idx):
                m = masks[idx[conf[idx].argmax()]]
                out[k] = m if m.shape == (h, w) else cv2.resize(m.astype(np.uint8), (w, h)) > 0
    out['fish'] = out['body'] | out['caudal']
    return out


def predict_points(model, keys, img, fish, imgsz, device):
    if not fish.any():
        return {}
    x0, y0, x1, y1 = app_landmarks.crop_box(fish, img.shape)
    r = model.predict(img[y0:y1, x0:x1], imgsz=imgsz, device=device, verbose=False, conf=0.1)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return {}
    i = int(r.boxes.conf.argmax())
    xy = r.keypoints.xy[i].cpu().numpy()
    conf = r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(keys))
    return {k: [float(xy[j, 0] + x0), float(xy[j, 1] + y0)] for j, k in enumerate(keys)
            if j < len(xy) and conf[j] >= app_landmarks.MIN_CONF}


def _length(p, a, b):
    return float(np.hypot(*np.subtract(p[a], p[b]))) if p.get(a) and p.get(b) else float('nan')


def overlay(img, pred, pts, truth=None, width=640):
    vis = img.copy()
    for k, col in (('body', (255, 0, 255)), ('caudal', (0, 200, 255))):
        cnts, _ = cv2.findContours(pred[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, col, 3)
    if truth is not None:
        cnts, _ = cv2.findContours(truth.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 255, 0), 1)
    for k, (x, y) in pts.items():
        cv2.circle(vis, (int(x), int(y)), 7, (0, 0, 255), -1)
    if pred['fish'].any():
        x0, y0, x1, y1 = app_landmarks.crop_box(pred['fish'], vis.shape)
        vis = vis[y0:y1, x0:x1]
    s = width / vis.shape[1]
    vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()


def run(work, extra_folders=(), log=print):
    from ultralytics import YOLO
    with open(os.path.join(work, 'outline_model.json'), encoding='utf-8') as f:
        om = json.load(f)
    with open(os.path.join(work, 'landmarks_model.json'), encoding='utf-8') as f:
        lm = json.load(f)
    device = om['machine']['device']
    seg, pose = YOLO(om['weights']), YOLO(lm['weights'])
    keys = lm['keys']
    manual = set()
    with open(os.path.join(work, 'photos.csv'), newline='', encoding='utf-8') as f:
        manual = {r['uid'] for r in csv.DictReader(f) if r['manual_landmarks'] == 'TRUE'}
    rows, pics = [], {}
    tests = sorted(glob.glob(os.path.join(work, 'test', '*.json')))
    log(f'Checking the models on {len(tests)} test photos')
    for jp in tests:
        uid = os.path.basename(jp)[:-5]
        with open(jp, encoding='utf-8') as f:
            t = json.load(f)
        img = cv2.imread(os.path.join(work, 'test', uid + '.jpg'))
        truth = {k: cv2.imread(os.path.join(work, 'test', f'{uid}_{k}.png'), cv2.IMREAD_GRAYSCALE) > 0
                 for k in ('fish', 'body', 'caudal')}
        pred = predict_outline(seg, img, om['imgsz'], device)
        s = img.shape[1] / t['size'][0]                     # test image -> photo pixels
        tp = {k: v for k, v in t['points'].items() if v}
        pts_small = predict_points(pose, keys, img, pred['fish'], lm['imgsz'], device)
        pts = {k: [v[0] / s, v[1] / s] for k, v in pts_small.items()}
        sl_true = _length(tp, SL_FROM, SL_TO)
        row = {'uid': uid, 'manual_landmarks': uid in manual, **{f'iou_{k}': _iou(pred[k], truth[k]) for k in ('fish', 'body', 'caudal')}}
        for k in ('snout', 'eye', 'peduncle_end', 'tail_tip'):
            row[f'err_{k}_pct_sl'] = (100 * _length({'a': pts.get(k), 'b': tp.get(k)}, 'a', 'b') / sl_true
                                      if k in tp and k in pts else float('nan'))
        row['sl_err_pct'] = 100 * (_length(pts, SL_FROM, SL_TO) - sl_true) / sl_true
        tl_true = _length(tp, SL_FROM, 'tail_tip')
        row['tl_err_pct'] = 100 * (_length(pts, SL_FROM, 'tail_tip') - tl_true) / tl_true
        rows.append(row)
        pics[uid] = overlay(img, pred, pts_small, truth['fish'])
    extra = []
    for folder in extra_folders:
        for p in sorted(glob.glob(os.path.join(folder, '**', '*.jp*g'), recursive=True))[:24]:
            img = cv2.imread(p)
            if img is None:
                continue
            s = 1280 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            pred = predict_outline(seg, img, om['imgsz'], device)
            pts = predict_points(pose, keys, img, pred['fish'], lm['imgsz'], device)
            extra.append((os.path.relpath(p, folder), overlay(img, pred, pts)))
    with open(os.path.join(work, 'test_results.csv'), 'w', encoding='utf-8') as f:
        cols = list(rows[0]) if rows else ['uid']
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join('NA' if isinstance(r[c], float) and np.isnan(r[c]) else
                             (f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c]).upper() if isinstance(r[c], bool)
                              else str(r[c])) for c in cols) + '\n')
    summary = summarise(rows)
    summary['manual_landmarks_only'] = summarise([r for r in rows if r['manual_landmarks']])
    cs = os.path.join(work, 'colours', 'colour_summary.json')
    if os.path.exists(cs):
        with open(cs, encoding='utf-8') as f:
            summary['colours'] = json.load(f)
    write_report(os.path.join(work, 'report.html'), summary, rows, pics, extra, om, lm)
    with open(os.path.join(work, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    return summary


def summarise(rows):
    out = {'photos': len(rows)}
    for c in (rows[0] if rows else {}):
        if c in ('uid', 'manual_landmarks'):
            continue
        v = np.array([r[c] for r in rows], float)
        v = v[~np.isnan(v)]
        if c.startswith('iou'):
            out[c] = {'median': float(np.median(v)) if len(v) else None, 'p10': float(np.percentile(v, 10)) if len(v) else None}
        else:
            a = np.abs(v)
            out[c] = {'median_abs': float(np.median(a)) if len(a) else None, 'p90_abs': float(np.percentile(a, 90)) if len(a) else None,
                      'bias': float(np.mean(v)) if len(v) else None, 'n': int(len(v))}
    return out


def _fmt(x, pct=False):
    return 'NA' if x is None else (f'{x:.1f}%' if pct else f'{x:.3f}')


def write_report(path, summary, rows, pics, extra, om, lm):
    worst = sorted(rows, key=lambda r: (np.nan_to_num(r['iou_fish'], nan=-1)))[:12]
    tr = []
    hand = summary.get('manual_landmarks_only', {})
    for k, label in (('iou_fish', 'Fish outline overlap (IoU)'), ('iou_body', 'Body overlap (IoU)'),
                     ('iou_caudal', 'Caudal fin overlap (IoU)')):
        s = summary.get(k, {})
        tr.append(f'<tr><td>{label}</td><td>{_fmt(s.get("median"))}</td><td>{_fmt(s.get("p10"))} (worst 10%)</td>'
                  f'<td>{_fmt(hand.get(k, {}).get("median"))}</td></tr>')
    for k, label in (('err_snout_pct_sl', 'Snout'), ('err_eye_pct_sl', 'Eye'), ('err_peduncle_end_pct_sl', 'End of the peduncle'),
                     ('err_tail_tip_pct_sl', 'Tail tip'), ('sl_err_pct', 'Standard length'), ('tl_err_pct', 'Total length')):
        s = summary.get(k, {})
        tr.append(f'<tr><td>{label}: error as % of SL</td><td>{_fmt(s.get("median_abs"), True)}</td>'
                  f'<td>{_fmt(s.get("p90_abs"), True)} (90% of fish below)</td>'
                  f'<td>{_fmt(hand.get(k, {}).get("median_abs"), True)}</td></tr>')
    col = summary.get('colours') or {}
    ctr = []
    for k, label in (('orange_vs_car_perc', '% orange vs their car_perc'), ('black_vs_mel_perc', '% black vs their mel_perc'),
                     ('black_vs_mel_perc_v2', '% black vs their mel_perc_v2'),
                     ('orange_vs_masks', '% orange vs their carotenoid mask'), ('black_vs_masks', '% black vs their melanic mask')):
        s = col.get(k, {})
        if s.get('n', 0) >= 2:
            ctr.append(f'<tr><td>{label}</td><td>{s["n"]}</td><td>{s["bias"]:+.2f}</td><td>{s["mae"]:.2f}</td>'
                       f'<td>{s["r"]:.3f}</td></tr>')
    colour_html = ('<h2>Colour model (orange, black)</h2><p>Percentages of the fish area, on the test photos. Pixel overlap '
                   f'(IoU, median): orange {_fmt(col.get("iou_orange_median"))}, black {_fmt(col.get("iou_black_median"))}.</p>'
                   '<table><tr><th>Comparison</th><th>Fish</th><th>Bias (points)</th><th>Mean absolute error (points)</th>'
                   f'<th>r</th></tr>{"".join(ctr)}</table>') if col else ''
    cards = ''.join(f'<figure><img src="data:image/jpeg;base64,{pics[r["uid"]]}"><figcaption>{html.escape(r["uid"])} '
                    f'IoU {r["iou_fish"]:.2f}</figcaption></figure>' for r in worst)
    ex = ''.join(f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>{html.escape(n)}</figcaption></figure>'
                 for n, b in extra)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Guppy Colour training report</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px}}figure{{display:inline-block;margin:6px;width:320px}}
img{{width:100%}}figcaption{{font-size:12px}}</style></head><body>
<h1>Training report</h1>
<p>Test photos: {summary['photos']} (fish never seen in training). Outline model: {html.escape(os.path.basename(om['weights']))},
image size {om['imgsz']}; landmark model image size {lm['imgsz']}; machine: {html.escape(str(om['machine'].get('name')))}.</p>
<p>The reference is the van der Bijl et al. 2025 annotations, mostly made by their own model: the numbers say how close
we are to them, not how close to perfect.</p>
<p>Last column: only the {hand.get('photos', 0)} test photos whose landmarks van der Bijl placed by hand, the fairest check of
the landmarks, SL and TL.</p>
<table><tr><th>Measure</th><th>Typical (median)</th><th>Spread</th><th>Typical, hand-placed landmarks</th></tr>{''.join(tr)}</table>
{colour_html}
<h2>The 12 test photos with the lowest overlap</h2>
<p>Magenta: our body; orange: our caudal fin; thin green: van der Bijl's fish; red dots: our landmarks.</p>{cards}
<h2>Other photos (no annotations: judge by eye)</h2>{ex or '<p>None given.</p>'}
</body></html>"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(doc)

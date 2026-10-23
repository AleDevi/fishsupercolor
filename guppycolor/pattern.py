"""Pattern and clusters (screen 6): every analysed fish with a colour map is aligned on the mean shape
(engine/pattern.py), described as per-pixel colour maps and as spots (points / lines), compared with the distance
the user picks and grouped by hierarchical clustering at the similarity of the slider. CSVs for R in export/."""
import csv
import hashlib
import json
import os

import cv2
import numpy as np

from . import profile
from .engine import pattern as eng

DEFAULTS = {
    'head': 'left',             # left | right | auto: where the head is in the photos (the snout landmark wins)
    'ignore_head': 0.13,        # the front of the head (eye) is left out, in body lengths from the snout
    'region': 'body',           # body = snout -> caudal peduncle; whole = the whole mask (caudal fin too, if in it)
    'n_semi': 10,               # semi-landmarks on each of the dorsal and ventral outline
    'classes': [],              # colour classes compared; [] = all of the model except base and reflection
    'distance': 'jaccard',      # jaccard | spots | composition
    'tolerance_px': 2.0,        # jaccard: blur of the aligned maps, forgives small misalignments (canvas px)
    'linkage': 'average',       # average (UPGMA) | complete | single
    'similarity': 0.5,          # slider: fish more similar than this end in the same cluster
    'line_ratio': 2.5,          # spot length / width at or above which a spot is a line
    'min_spot': 0.0005,         # smallest spot kept, in body lengths squared
    'max_shift': 0.15,          # spots: two spots further apart than this (body lengths) do not match
    'grid': [20, 8],            # cells along / across the body for the per-pixel CSV
}
DISTANCES = [('jaccard', 'Colour maps (Jaccard on aligned maps)'),
             ('spots', 'Spots as points and lines'),
             ('composition', 'Colour amounts only (Bray-Curtis)')]
SKIP_CLASSES = ('base', 'reflection')
USABLE = ('ok', 'to_check', 'reviewed')


def settings(project):
    s = dict(DEFAULTS)
    s.update(project.settings.get('pattern', {}))
    return s


def save_settings(project, s):
    project.settings['pattern'] = {k: s[k] for k in DEFAULTS}
    project.save()


# --- colour maps of single photos --------------------------------------------------------------------------
def colour_map_path(project, pid):
    return os.path.join(project.dir('masks'), pid + '_colours.png')


def save_colour_map(project, pid, labmap):
    """Called by the pipeline after the colour classes: the label map (work size) and where it sits in the photo."""
    lab, box, s = labmap
    cv2.imwrite(colour_map_path(project, pid), lab.astype(np.uint8))
    return {'box': [int(v) for v in box], 'scale': float(s)}


def load_colour_map(project, r, forest=None):
    """(labels, box, scale) of a result; made again from the photo when the pipeline did not keep it."""
    p = colour_map_path(project, r['photo_id'])
    info = r.get('colour_map')
    if info and os.path.exists(p):
        lab = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if lab is not None:
            return lab, tuple(info['box']), info['scale']
    from . import training
    from .engine import pixels
    mask = training.fish_mask(project, r['path'])
    if mask is None or forest is None:
        return None
    img = training.corrected_image(project, r['path'])
    lab, box, s, _ = pixels.classify(img, mask, forest)
    r['colour_map'] = save_colour_map(project, r['photo_id'], (lab, box, s))
    project.save_result(r['path'], r)
    return lab, box, s


def model_classes(project):
    """Colour classes of the project's model: [{index, key, label, display}]."""
    from . import training
    name = project.settings.get('colours', {}).get('model')
    if not name:
        return [], None
    forest = training.load_model(project, name)
    return forest.meta.get('classes', []), forest


# --- the aligned set --------------------------------------------------------------------------------------
class PatternSet:
    """All the fish of the project on the mean shape."""

    def __init__(self, rows, names, canvas_pts, size, tmask, amaps, classes, st, aligned, centroid_size):
        self.rows, self.names, self.canvas_pts, self.size, self.tmask = rows, names, canvas_pts, size, tmask
        self.amaps, self.classes, self.st, self.aligned, self.centroid_size = amaps, classes, st, aligned, centroid_size
        refl = [c['index'] for c in classes if c['key'] == 'reflection']
        bx, _ = eng.to_body(np.arange(tmask.shape[1]), 0, canvas_pts)
        keep = tmask & (bx >= st['ignore_head'])[None, :]
        self.valid = np.array([keep & (a > 0) & ~np.isin(a, refl) for a in amaps])
        self.spots = [{c['key']: eng.find_spots((a == c['index']) & v, canvas_pts,
                                                 max(2, int(st['min_spot'] * eng.BODY_PX ** 2)), st['line_ratio'])
                       for c in self.compared()} for a, v in zip(amaps, self.valid)]

    def compared(self):
        keys = self.st['classes'] or [c['key'] for c in self.classes if c['key'] not in SKIP_CLASSES]
        return [c for c in self.classes if c['key'] in keys]

    def fractions(self):
        """Fraction of the (valid) fish area covered by each compared class, per fish."""
        out = []
        for a, v in zip(self.amaps, self.valid):
            n = max(int(v.sum()), 1)
            out.append([float(((a == c['index']) & v).sum()) / n for c in self.compared()])
        return np.array(out)

    def distances(self, distance=None):
        st = self.st
        distance = distance or st['distance']
        if distance == 'composition':
            return eng.composition_matrix(self.fractions())
        if distance == 'spots':
            return eng.spots_matrix(self.spots, [c['key'] for c in self.compared()], st['max_shift'])
        idx = [c['index'] for c in self.compared()]
        step = 1 if len(self.amaps) <= 100 else 2          # every other pixel for big sets: same result, 4x faster
        sel = np.zeros_like(self.tmask)
        sel[::step, ::step] = True
        sel &= self.tmask
        stacks = [eng.class_stack(a, v, idx, st['tolerance_px'])[:, sel] for a, v in zip(self.amaps, self.valid)]
        return eng.jaccard_matrix(stacks, self.valid[:, sel])

    def frequency(self, members):
        """(C, H, W) share of the members with each compared class at each canvas pixel (NaN where none is valid)."""
        v = self.valid[members].sum(0).astype(float)
        out = []
        for c in self.compared():
            k = ((self.amaps[members] == c['index']) & self.valid[members]).sum(0)
            out.append(np.where(v > 0, k / np.maximum(v, 1), np.nan))
        return np.array(out)


def fish_name(r):
    return r.get('label_id') or os.path.splitext(r.get('file', r['photo_id']))[0]


def usable_results(project):
    return [r for r in project.results() if r.get('status') in USABLE and r.get('colours')]


def _cache_key(r, st):
    m = os.path.join('masks', r['photo_id'])
    return hashlib.sha1(json.dumps([r.get('settings_hash'), r.get('source_mtime'), st['region'], st['n_semi'], st['head'],
                                    r.get('landmarks'), m], sort_keys=True).encode()).hexdigest()[:12]


def _outline(project, r, st, keys):
    """Outline points of one fish, cached in pattern/outlines.json (morpho on the full mask is the slow part)."""
    from . import training
    cache_path = os.path.join(project.dir('pattern'), 'outlines.json')
    cache = getattr(project, '_outline_cache', None)
    if cache is None:
        try:
            with open(cache_path, encoding='utf-8') as f:
                cache = json.load(f)
        except (OSError, ValueError):
            cache = {}
        project._outline_cache = cache
    key = r['photo_id'] + '|' + _cache_key(r, st) + '|' + ','.join(keys)
    if key not in cache:
        mask = training.fish_mask(project, r['path'])
        if mask is None:
            return None
        lm = r.get('landmarks') or {}
        head = None if st['head'] == 'auto' else st['head']
        if 'snout' in lm:
            ys, xs = np.nonzero(mask)
            head = 'left' if lm['snout'][0] < xs.mean() else 'right'
        pts, local = eng.outline_points(mask, st['n_semi'], st['region'], lm, keys, head)
        cache[key] = {'pts': pts.tolist(), 'local': local.tolist()}
    return np.array(cache[key]['pts']), np.array(cache[key]['local'])


def _save_outline_cache(project):
    cache = getattr(project, '_outline_cache', None)
    if cache is not None:
        p = os.path.join(project.dir('pattern'), 'outlines.json')
        with open(p + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(cache, f)
        os.replace(p + '.tmp', p)


def build(project, st=None, progress=None, stop=None):
    """Align every usable fish. progress(i, n, name). Raises ValueError with a message for the user."""
    st = st or settings(project)
    classes, forest = model_classes(project)
    if not classes:
        raise ValueError('No colour model in this project: train one (screen 7) and choose it in Settings.')
    results = usable_results(project)
    if len(results) < 2:
        raise ValueError('At least two analysed fish with colour classes are needed.')
    prof = profile.load(project.settings['profile'])
    wanted = profile.procrustes_landmarks(prof)
    keys = [k for k in wanted if all(k in (r.get('landmarks') or {}) for r in results)]
    rows, pts, local, maps = [], [], [], []
    for i, r in enumerate(results):
        if stop and stop():
            break
        if progress:
            progress(i, len(results), fish_name(r))
        try:
            o = _outline(project, r, st, keys)
            cm = load_colour_map(project, r, forest) if o is not None else None
        except Exception:           # one bad photo must not stop the set
            o = cm = None
        if o is None or cm is None:
            continue
        rows.append(r); pts.append(o[0]); local.append(o[1]); maps.append(cm)
    _save_outline_cache(project)
    if len(rows) < 2:
        raise ValueError('Fewer than two fish could be aligned.')
    aligned, mean, cs = eng.gpa(np.array(local))
    end = st['n_semi'] + 1
    canvas_pts, size = eng.canvas_shape(mean, end)
    tmask = eng.template_mask(canvas_pts, size, st['n_semi'])
    amaps = []
    for p, (lab, box, s) in zip(pts, maps):
        lab_pts = (p - np.array(box[:2], float)) * s
        amaps.append(eng.warp_labels(lab, lab_pts, canvas_pts, tmask))
    names = [fish_name(r) for r in rows]
    return PatternSet(rows, names, canvas_pts, size, tmask, np.array(amaps), classes, st, aligned, cs)


# --- clusters --------------------------------------------------------------------------------------------
class Clustering:
    def __init__(self, ps, D=None):
        self.ps = ps
        self.D = ps.distances() if D is None else D
        self.Z = eng.linkage(self.D, ps.st['linkage'])
        self.coph = eng.cophenetic_r(self.Z, self.D)
        self.xy = eng.mds2(self.D)

    def at(self, similarity):
        return eng.clusters_at(self.Z, len(self.D), similarity)

    def agreement(self, labels):
        """Adjusted Rand index against the groups written on the labels (None when fewer than 2 groups)."""
        groups = [r.get('label_group') for r in self.ps.rows]
        if sum(g is not None and g != '' for g in groups) < 2 or len({g for g in groups if g}) < 2:
            return None
        sel = [i for i, g in enumerate(groups) if g]
        return eng.adjusted_rand([groups[i] for i in sel], [int(labels[i]) for i in sel])


# --- pictures ---------------------------------------------------------------------------------------------
def _rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def frequency_image(ps, members, scale=2):
    """RGB picture of a cluster: each class painted in its colour, as strong as the share of members having it."""
    F = np.nan_to_num(ps.frequency(members))
    img = np.full(ps.size + (3,), 255.0)
    img[ps.tmask] = 235
    for f, c in zip(F, ps.compared()):
        a = f[..., None] * 0.9
        img = img * (1 - a) + _rgb(c.get('display', '#888888')) * a
    img = img.astype(np.uint8)
    cv2.polylines(img, [eng.outline_polygon(ps.canvas_pts, ps.st['n_semi'])], True, (90, 90, 90), 1, cv2.LINE_AA)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def fish_image(ps, i, scale=2, spots=True):
    """RGB picture of one aligned fish with its spots: points as rings, lines as segments."""
    a, v = ps.amaps[i], ps.valid[i]
    img = np.full(ps.size + (3,), 255, np.uint8)
    img[ps.tmask] = 235
    img[ps.tmask & ~v] = 205
    for c in ps.compared():
        img[(a == c['index']) & v] = _rgb(c.get('display', '#888888')).astype(np.uint8)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    cv2.polylines(img, [eng.outline_polygon(ps.canvas_pts, ps.st['n_semi']) * scale], True, (90, 90, 90), 1, cv2.LINE_AA)
    if spots:
        x0, y0 = ps.canvas_pts[0]
        to_px = lambda x, y: (int(round((x0 + x * eng.BODY_PX) * scale)), int(round((y0 - y * eng.BODY_PX) * scale)))
        for key, lst in ps.spots[i].items():
            for s in lst:
                if s['kind'] == 'line':
                    cv2.line(img, to_px(s['x1'], s['y1']), to_px(s['x2'], s['y2']), (0, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.circle(img, to_px(s['x'], s['y']), 4, (0, 0, 0), 1, cv2.LINE_AA)
    return img


# --- export ----------------------------------------------------------------------------------------------
def _w(path, header, rows):
    from .export import _safe
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([_safe(v) if isinstance(v, str) else ('NA' if v is None or (isinstance(v, float) and np.isnan(v))
                                                             else (round(v, 5) if isinstance(v, float) else v))
                        for v in r])


def export(project, ps, cl, similarity, folder=None):
    """Writes the pattern CSVs (and one picture per cluster); returns the folder."""
    d = folder or os.path.join(project.dir('export'), 'pattern')
    os.makedirs(d, exist_ok=True)
    labels = cl.at(similarity)
    comp = ps.compared()
    st = ps.st
    ident = lambda r: [r['file'], r['folder'], r.get('label_id') or 'NA', r.get('label_group') or 'NA']
    base = ['file', 'folder', 'label_id', 'label_group']
    fr = ps.fractions()
    head = base + ['cluster', 'centroid_size_px']
    for c in comp:
        k = c['key']
        head += [f'aligned_pct_{k}', f'centre_x_{k}', f'centre_y_{k}', f'n_points_{k}', f'n_lines_{k}']
    rows = []
    for i, r in enumerate(ps.rows):
        row = ident(r) + [int(labels[i]), float(ps.centroid_size[i])]
        for j, c in enumerate(comp):
            m = (ps.amaps[i] == c['index']) & ps.valid[i]
            yy, xx = np.nonzero(m)
            cx, cy = eng.to_body(xx.mean(), yy.mean(), ps.canvas_pts) if len(xx) else (None, None)
            sp = ps.spots[i][c['key']]
            row += [100 * fr[i, j], None if cx is None else float(cx), None if cy is None else float(cy),
                    sum(s['kind'] == 'point' for s in sp), sum(s['kind'] == 'line' for s in sp)]
        rows.append(row)
    _w(os.path.join(d, 'pattern_fish.csv'), head, rows)
    # spots, one row each
    sk = ['kind', 'x', 'y', 'x1', 'y1', 'x2', 'y2', 'length', 'width', 'elongation', 'angle', 'area']
    rows = [ident(r) + [int(labels[i]), c['key']] + [s[k] for k in sk]
            for i, r in enumerate(ps.rows) for c in comp for s in ps.spots[i][c['key']]]
    _w(os.path.join(d, 'pattern_spots.csv'), base + ['cluster', 'class'] + sk, rows)
    # per-pixel maps on a coarse grid (long format)
    gx, gy = st['grid']
    ys, xs = np.nonzero(ps.tmask)
    bx, by = eng.to_body(xs, ys, ps.canvas_pts)
    ext = max(np.abs(by).max(), 1e-9)
    ix = np.clip((bx * gx).astype(int), 0, gx - 1)
    iy = np.clip(((by + ext) / (2 * ext) * gy).astype(int), 0, gy - 1)
    rows = []
    for i, r in enumerate(ps.rows):
        v = ps.valid[i][ys, xs]
        for c in comp:
            k = (ps.amaps[i][ys, xs] == c['index']) & v
            n_v = np.bincount(ix * gy + iy, v, gx * gy)
            n_k = np.bincount(ix * gy + iy, k, gx * gy)
            for cell in np.nonzero(n_v)[0]:
                cx_, cy_ = divmod(int(cell), gy)
                rows.append(ident(r) + [c['key'], cx_ + 1, cy_ + 1, (cx_ + 0.5) / gx,
                                        -ext + (cy_ + 0.5) * 2 * ext / gy, float(n_k[cell] / n_v[cell])])
    _w(os.path.join(d, 'pattern_grid.csv'), base + ['class', 'cell_x', 'cell_y', 'x', 'y', 'fraction'], rows)
    # aligned shape (Procrustes coordinates) for geometric morphometrics in R
    K = ps.aligned.shape[1]
    _w(os.path.join(d, 'pattern_shape.csv'), base + [f'{a}{k + 1}' for k in range(K) for a in 'xy'],
       [ident(r) + list(ps.aligned[i].ravel()) for i, r in enumerate(ps.rows)])
    # distance matrix and clusters
    names = [f'{fish_name(r)}' for r in ps.rows]
    _w(os.path.join(d, 'pattern_distances.csv'), ['fish'] + names,
       [[n] + [float(x) for x in row] for n, row in zip(names, cl.D)])
    info = [['distance', st['distance']], ['head', st['head']], ['ignore_head', st['ignore_head']], ['tolerance_px', st['tolerance_px']], ['linkage', st['linkage']],
            ['similarity', similarity], ['n_clusters', int(labels.max())], ['cophenetic_r', cl.coph],
            ['adjusted_rand_vs_label_group', cl.agreement(labels)], ['region', st['region']],
            ['classes', ';'.join(c['key'] for c in comp)], ['n_fish', len(ps.rows)]]
    _w(os.path.join(d, 'pattern_settings.csv'), ['setting', 'value'], info)
    for k in range(1, int(labels.max()) + 1):
        img = frequency_image(ps, np.nonzero(labels == k)[0])
        cv2.imwrite(os.path.join(d, f'cluster_{k:02d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return d

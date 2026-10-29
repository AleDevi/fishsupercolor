"""Export a trained model to one .zip file and import it into another project (or another computer).

Colour models: model.npz (plain arrays, read without pickle) + model.json. Landmark models: best.pt (YOLO-pose
weights, opened weights-only) + model.json. Outline models: best.pt (YOLO segmentation weights of the fish parts,
e.g. from the training kit) + model.json. manifest.json says the kind and the SHA-256 of each file. Import reads
only these fixed names, checks sizes and hashes, checks that the model fits the project's species profile, and saves
it as the next version (colours_vN / landmarks_vN) with where it came from."""
import datetime
import hashlib
import io
import json
import os
import zipfile

import numpy as np

from . import __version__, landmarks, outline, profile, training
from .engine import pixels

MAX_BYTES = {'manifest.json': 1 << 20, 'model.json': 1 << 20, 'model.npz': 300 << 20, 'best.pt': 300 << 20}
FILES = {'colours': ('model.npz', 'model.json'), 'landmarks': ('best.pt', 'model.json'),
         'outline': ('best.pt', 'model.json')}
WEIGHTS = {'landmarks': landmarks.model_weights, 'outline': outline.model_weights}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def kind_of(name):
    for kind in ('landmarks', 'outline'):
        if name.startswith(kind + '_'):
            return kind
    return 'colours'


def export_model(project, name, dest):
    """Write the model `name` of this project to the zip file `dest`."""
    if not profile.SAFE_NAME.fullmatch(name or ''):
        raise ValueError(f'invalid model name: {name!r}')
    kind = kind_of(name)
    models = training.models_dir(project)
    if kind == 'colours':
        weights = os.path.join(models, name + '.npz')
    else:
        weights = WEIGHTS[kind](project, name)
    with open(weights, 'rb') as f:
        w = f.read()
    meta_p = os.path.join(models, name + '.json')
    meta = {}
    if os.path.exists(meta_p):
        with open(meta_p, encoding='utf-8') as f:
            meta = json.load(f)
    if kind != 'colours' and _sha(w) != meta.get('sha256'):
        raise ValueError(f'{name}: the model file differs from the one trained, not exported')
    m = json.dumps(meta, indent=1, ensure_ascii=False).encode('utf-8')
    wname = FILES[kind][0]
    manifest = {'kind': kind, 'name': name, 'profile': project.settings['profile'], 'app_version': __version__,
                'exported': datetime.datetime.now().isoformat(timespec='minutes'), 'project': project.name,
                'sha256': {wname: _sha(w), 'model.json': _sha(m)}}
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=1, ensure_ascii=False))
        z.writestr(wname, w)
        z.writestr('model.json', m)
    return dest


def read_package(path):
    """(manifest, {file name: bytes}) after the checks; raises ValueError on anything unexpected."""
    with zipfile.ZipFile(path) as z:
        infos = {i.filename: i for i in z.infolist()}
        if 'manifest.json' not in infos or infos['manifest.json'].file_size > MAX_BYTES['manifest.json']:
            raise ValueError('not a Guppy Colour model file')
        manifest = json.loads(z.read('manifest.json').decode('utf-8'))
        kind = manifest.get('kind')
        if kind not in FILES:
            raise ValueError('not a Guppy Colour model file')
        if set(infos) != {'manifest.json', *FILES[kind]}:
            raise ValueError('unexpected files in the model file')
        data = {}
        for n in FILES[kind]:
            if infos[n].file_size > MAX_BYTES[n]:
                raise ValueError(f'{n} is too large')
            data[n] = z.read(n)
            if _sha(data[n]) != manifest.get('sha256', {}).get(n):
                raise ValueError(f'{n} is damaged (SHA-256 differs)')
    return manifest, data


def fit_colour_classes(arrays, meta, pool, mine):
    """Colour model of another project -> this project's brush numbers. Its classes must be some of this project's
    (e.g. only orange, black and body without colour). The training pixels it keeps are checked and become
    'not your strokes' here, so Ritocco uses them."""
    index = {b['key']: b['index'] for b in mine}
    classes = meta.get('classes') or []
    theirs = [c.get('key') for c in classes]
    if len(set(theirs)) < 2 or not set(theirs) <= set(index):
        raise ValueError('the colour classes of this model (' + ', '.join(map(str, theirs)) + ') are not among the '
                         'ones of this project (' + ', '.join(index) + ')')
    share = meta.get('class_share')
    if share is not None and not (isinstance(share, dict) and set(share) <= set(theirs) and all(
            isinstance(v, (int, float)) and 0 <= v <= 1 for v in share.values())):
        raise ValueError('the colour model is damaged (class shares)')
    remap = np.zeros(256, np.uint8)
    for c in classes:
        remap[int(c['index'])] = index[c['key']]
    if not np.isin(arrays['classes'], [c['index'] for c in classes]).all():
        raise ValueError('the colour model is damaged (unknown class)')
    arrays = dict(arrays, classes=remap[arrays['classes']])
    meta = dict(meta, classes=[dict(c, index=index[c['key']], label=next(b['label'] for b in mine if b['key'] == c['key']))
                               for c in classes])
    if pool is not None:
        X, y = pool['pool_X'], pool['pool_y']
        if (X.ndim != 2 or X.shape[1] != pixels.N_FEATURES or y.shape != (len(X),) or pool['pool_own'].shape != y.shape
                or not np.isin(y, [c['index'] for c in classes]).all() or not np.isfinite(X).all()):
            raise ValueError('the colour model is damaged (training pixels)')
        pool = {'pool_X': X.astype(np.float32), 'pool_y': remap[y], 'pool_own': np.zeros(len(y), bool)}
    return arrays, meta, pool


def import_model(project, path):
    """Add the model in the zip `path` to this project. Returns (new name, meta)."""
    manifest, data = read_package(path)
    kind = manifest['kind']
    meta = json.loads(data['model.json'].decode('utf-8'))
    prof = profile.load(project.settings['profile'])
    origin = {'name': manifest.get('name'), 'project': manifest.get('project'), 'exported': manifest.get('exported'),
              'app_version': manifest.get('app_version'), 'file': os.path.basename(path)}
    models = training.models_dir(project)
    if kind == 'colours':
        with io.BytesIO(data['model.npz']) as b:
            with np.load(b, allow_pickle=False) as z:
                arrays = {k: z[k] for k in ('left', 'right', 'feature', 'threshold', 'proba', 'roots', 'classes')}
                pool = {k: z[k] for k in ('pool_X', 'pool_y', 'pool_own')} if 'pool_X' in z.files else None
        if meta.get('features_version', pixels.FEATURES_VERSION) != pixels.FEATURES_VERSION:
            raise ValueError('this colour model was made by a different version of the pixel features')
        arrays, meta, pool = fit_colour_classes(arrays, meta, pool, training.brushes(prof))
        forest = pixels.Forest(meta=meta, **arrays)
        nums = [int(n.rsplit('_v', 1)[1]) for n, _ in training.list_models(project)]
        name = f'colours_v{max(nums, default=0) + 1}'
        forest.meta.update(name=name, imported_from=origin, pool_pixels=0 if pool is None else len(pool['pool_y']))
        forest.save(os.path.join(models, name + '.npz'), pool=pool)
        return name, forest.meta
    if kind == 'landmarks':
        keys = [l['key'] for l in landmarks.landmark_list(prof)]
        if meta.get('keys') != keys:
            raise ValueError('the landmarks of this model (' + ', '.join(map(str, meta.get('keys') or [])) + ') differ '
                             'from the ones of this project (' + ', '.join(keys) + ')')
        existing = landmarks.list_models(project)
    else:
        parts = [p['key'] for p in prof.get('parts', [])]
        classes = meta.get('classes')
        if not isinstance(classes, list) or 'body' not in classes or not set(classes) <= set(parts):
            raise ValueError('the parts of this outline model (' + ', '.join(map(str, classes or [])) + ') are not '
                             'body parts of this project (' + ', '.join(parts) + ')')
        existing = outline.list_models(project)
    nums = [int(n.rsplit('_v', 1)[1]) for n, _ in existing]
    name = f'{kind}_v{max(nums, default=0) + 1}'
    w = WEIGHTS[kind](project, name)
    os.makedirs(os.path.dirname(w), exist_ok=True)
    with open(w, 'wb') as f:
        f.write(data['best.pt'])
    meta.update(name=name, sha256=_sha(data['best.pt']), imported_from=origin)
    with open(os.path.join(models, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return name, meta

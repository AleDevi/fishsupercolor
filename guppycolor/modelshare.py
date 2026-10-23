"""Export a trained model to one .zip file and import it into another project (or another computer).

Colour models: model.npz (plain arrays, read without pickle) + model.json. Landmark models: best.pt (YOLO-pose
weights, opened weights-only) + model.json. manifest.json says the kind and the SHA-256 of each file. Import reads
only these fixed names, checks sizes and hashes, checks that the model fits the project's species profile, and saves
it as the next version (colours_vN / landmarks_vN) with where it came from."""
import datetime
import hashlib
import io
import json
import os
import zipfile

import numpy as np

from . import __version__, landmarks, profile, training
from .engine import pixels

MAX_BYTES = {'manifest.json': 1 << 20, 'model.json': 1 << 20, 'model.npz': 300 << 20, 'best.pt': 300 << 20}
FILES = {'colours': ('model.npz', 'model.json'), 'landmarks': ('best.pt', 'model.json')}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def kind_of(name):
    return 'landmarks' if name.startswith('landmarks_') else 'colours'


def export_model(project, name, dest):
    """Write the model `name` of this project to the zip file `dest`."""
    if not profile.SAFE_NAME.fullmatch(name or ''):
        raise ValueError(f'invalid model name: {name!r}')
    kind = kind_of(name)
    models = training.models_dir(project)
    if kind == 'colours':
        weights = os.path.join(models, name + '.npz')
    else:
        weights = landmarks.model_weights(project, name)
    with open(weights, 'rb') as f:
        w = f.read()
    meta_p = os.path.join(models, name + '.json')
    meta = {}
    if os.path.exists(meta_p):
        with open(meta_p, encoding='utf-8') as f:
            meta = json.load(f)
    if kind == 'landmarks' and _sha(w) != meta.get('sha256'):
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
        forest = pixels.Forest(meta=meta, **arrays)
        if meta.get('features_version', pixels.FEATURES_VERSION) != pixels.FEATURES_VERSION:
            raise ValueError('this colour model was made by a different version of the pixel features')
        mine = [c['key'] for c in training.brushes(prof)]
        theirs = [c.get('key') for c in meta.get('classes', [])]
        if theirs != mine:
            raise ValueError('the colour classes of this model (' + ', '.join(map(str, theirs)) + ') differ from the '
                             'ones of this project (' + ', '.join(mine) + ')')
        nums = [int(n.rsplit('_v', 1)[1]) for n, _ in training.list_models(project)]
        name = f'colours_v{max(nums, default=0) + 1}'
        forest.meta.update(name=name, imported_from=origin)
        forest.save(os.path.join(models, name + '.npz'))
        return name, forest.meta
    keys = [l['key'] for l in landmarks.landmark_list(prof)]
    if meta.get('keys') != keys:
        raise ValueError('the landmarks of this model (' + ', '.join(map(str, meta.get('keys') or [])) + ') differ '
                         'from the ones of this project (' + ', '.join(keys) + ')')
    nums = [int(n.rsplit('_v', 1)[1]) for n, _ in landmarks.list_models(project)]
    name = f'landmarks_v{max(nums, default=0) + 1}'
    w = landmarks.model_weights(project, name)
    os.makedirs(os.path.dirname(w), exist_ok=True)
    with open(w, 'wb') as f:
        f.write(data['best.pt'])
    meta.update(name=name, sha256=_sha(data['best.pt']), imported_from=origin)
    with open(os.path.join(models, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return name, meta

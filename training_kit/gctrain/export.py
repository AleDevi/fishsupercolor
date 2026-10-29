"""Model files for Guppy Colour, in the same .zip layout as its Export model button (modelshare.py): manifest.json
with the kind and the SHA-256 of each file, the weights and model.json. The landmark file imports in the
Landmarks screen today; the outline file needs a Guppy Colour version that knows the 'outline' kind."""
import datetime
import hashlib
import json
import os
import zipfile

from . import __version__, vdb


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def package(dest, kind, name, profile_name, weights_path, meta):
    with open(weights_path, 'rb') as f:
        w = f.read()
    meta = dict(meta, name=name, sha256=_sha(w), date=datetime.datetime.now().isoformat(timespec='minutes'),
                source='van der Bijl et al. 2025, Nature Ecology & Evolution; photos FRDR doi 10.20383/103.01281 '
                       '(CC BY 4.0)', trained_with=f'Guppy Colour training kit {__version__}')
    m = json.dumps(meta, indent=1, ensure_ascii=False).encode('utf-8')
    manifest = {'kind': kind, 'name': name, 'profile': profile_name, 'app_version': f'training kit {__version__}',
                'exported': meta['date'], 'project': 'van der Bijl 2025 pre-training',
                'sha256': {'best.pt': _sha(w), 'model.json': _sha(m)}}
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=1, ensure_ascii=False))
        z.writestr('best.pt', w)
        z.writestr('model.json', m)
    return dest


def run(work, out, summary):
    with open(os.path.join(work, 'outline_model.json'), encoding='utf-8') as f:
        om = json.load(f)
    with open(os.path.join(work, 'landmarks_model.json'), encoding='utf-8') as f:
        lm = json.load(f)
    n_train = len(os.listdir(os.path.join(work, 'landmarks', 'images', 'train')))
    files = [
        package(os.path.join(out, 'landmarks_vanderbijl.zip'), 'landmarks', 'landmarks_v1', lm['profile'], lm['weights'],
                {'keys': lm['keys'], 'photos': n_train, 'imgsz': lm['imgsz'], 'gpu': lm['machine']['device'] != 'cpu',
                 'not_trained': [k for k in lm['keys'] if k not in vdb.TRAINED],
                 'test': {k: v for k, v in summary.items() if k.startswith(('err_', 'sl_', 'tl_'))}}),
        package(os.path.join(out, 'outline_vanderbijl.zip'), 'outline', 'outline_v1', lm['profile'], om['weights'],
                {'classes': ['body', 'caudal'], 'imgsz': om['imgsz'], 'photos': n_train,
                 'gpu': om['machine']['device'] != 'cpu',
                 'test': {k: v for k, v in summary.items() if k.startswith('iou_')}}),
    ]
    return files

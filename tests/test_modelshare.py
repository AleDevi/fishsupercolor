import hashlib
import json
import os
import zipfile

import numpy as np
import pytest


def project(tmp_path, name):
    from guppycolor.project import Project
    d = tmp_path / name
    d.mkdir()
    return Project.create(str(d / 'proj'), name, [str(d)])


def colour_model(p):
    from guppycolor import profile, training
    from guppycolor.engine import pixels
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int) + 1
    f = pixels.Forest.train(X, y, trees=30)
    b = training.brushes(profile.load(p.settings['profile']))
    f.meta.update(name='colours_v1', classes=[{k: c[k] for k in ('index', 'key', 'label', 'display')} for c in b],
                  features_version=pixels.FEATURES_VERSION, annotator='Ale')
    f.save(os.path.join(training.models_dir(p), 'colours_v1.npz'))


def test_colour_model_export_import(tmp_path):
    from guppycolor import modelshare, training
    a, b = project(tmp_path, 'A'), project(tmp_path, 'B')
    colour_model(a)
    z = modelshare.export_model(a, 'colours_v1', str(tmp_path / 'm.zip'))
    colour_model(b)                                     # B already has a v1: the import becomes v2
    name, meta = modelshare.import_model(b, z)
    assert name == 'colours_v2' and meta['imported_from']['project'] == 'A' and meta['annotator'] == 'Ale'
    fa, fb = training.load_model(a, 'colours_v1'), training.load_model(b, 'colours_v2')
    X = np.random.default_rng(1).normal(size=(50, 4)).astype(np.float32)
    assert np.allclose(fa.predict_proba(X), fb.predict_proba(X))


def test_damaged_or_foreign_files_are_refused(tmp_path):
    from guppycolor import modelshare
    a = project(tmp_path, 'A')
    colour_model(a)
    z = modelshare.export_model(a, 'colours_v1', str(tmp_path / 'm.zip'))
    with zipfile.ZipFile(z) as src:
        files = {n: src.read(n) for n in src.namelist()}
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as out:                       # weights changed after export
        for n, d in files.items():
            out.writestr(n, d + b'x' if n == 'model.npz' else d)
    with pytest.raises(ValueError, match='damaged'):
        modelshare.import_model(a, str(bad))
    extra = tmp_path / 'extra.zip'
    with zipfile.ZipFile(extra, 'w') as out:
        for n, d in files.items():
            out.writestr(n, d)
        out.writestr('../evil.py', b'print(1)')
    with pytest.raises(ValueError, match='unexpected'):
        modelshare.import_model(a, str(extra))
    other = tmp_path / 'classes.zip'
    meta = json.loads(files['model.json'])
    meta['classes'] = [{'key': 'red'}, {'key': 'blue'}]
    m = json.dumps(meta).encode()
    man = json.loads(files['manifest.json'])
    man['sha256']['model.json'] = hashlib.sha256(m).hexdigest()
    with zipfile.ZipFile(other, 'w') as out:
        out.writestr('manifest.json', json.dumps(man))
        out.writestr('model.npz', files['model.npz'])
        out.writestr('model.json', m)
    with pytest.raises(ValueError, match='colour classes'):
        modelshare.import_model(a, str(other))


def test_landmark_model_export_import(tmp_path):
    from guppycolor import landmarks, modelshare, profile
    a, b = project(tmp_path, 'A'), project(tmp_path, 'B')
    w = landmarks.model_weights(a, 'landmarks_v1')
    os.makedirs(os.path.dirname(w))
    data = b'not really weights'
    open(w, 'wb').write(data)
    keys = [l['key'] for l in landmarks.landmark_list(profile.load(a.settings['profile']))]
    json.dump({'name': 'landmarks_v1', 'keys': keys, 'sha256': hashlib.sha256(data).hexdigest()},
              open(os.path.join(a.dir('models'), 'landmarks_v1.json'), 'w'))
    z = modelshare.export_model(a, 'landmarks_v1', str(tmp_path / 'l.zip'))
    name, meta = modelshare.import_model(b, z)
    assert name == 'landmarks_v1' and dict(landmarks.list_models(b))[name]['sha256'] == meta['sha256']
    assert open(landmarks.model_weights(b, name), 'rb').read() == data
    open(w, 'ab').write(b'!')                                   # changed after training: not exported
    with pytest.raises(ValueError, match='differs'):
        modelshare.export_model(a, 'landmarks_v1', str(tmp_path / 'l2.zip'))

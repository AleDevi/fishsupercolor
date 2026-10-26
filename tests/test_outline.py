import hashlib
import json
import os
import zipfile

import numpy as np
import pytest
from conftest import stub_segmenter


def package(path, kind, meta, weights=b'not really weights'):
    m = json.dumps(meta).encode('utf-8')
    man = {'kind': kind, 'name': kind + '_v1', 'project': 'van der Bijl 2025 pre-training',
           'sha256': {'best.pt': hashlib.sha256(weights).hexdigest(), 'model.json': hashlib.sha256(m).hexdigest()}}
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('manifest.json', json.dumps(man))
        z.writestr('best.pt', weights)
        z.writestr('model.json', m)
    return str(path)


def project(photos):
    from guppycolor.project import Project
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['analysis']['default'] = 'size'
    p.save()
    return p


def test_outline_model_import_export(photos, tmp_path):
    from guppycolor import modelshare, outline
    p = project(photos)
    h = p.settings_hash()
    name, meta = modelshare.import_model(p, package(tmp_path / 'o.zip', 'outline', {'classes': ['body', 'caudal']}))
    assert name == 'outline_v1' and os.path.exists(outline.model_weights(p, name))
    assert dict(outline.list_models(p))[name]['sha256'] == hashlib.sha256(b'not really weights').hexdigest()
    assert p.settings_hash() == h                   # importing does not make the analysed photos stale
    p.settings['outline']['model'] = name
    assert p.settings_hash() != h                   # using it does
    z = modelshare.export_model(p, name, str(tmp_path / 'back.zip'))
    assert modelshare.read_package(z)[0]['kind'] == 'outline'
    with pytest.raises(ValueError, match='body parts'):
        modelshare.import_model(p, package(tmp_path / 'g.zip', 'outline', {'classes': ['body', 'gills']}))
    with pytest.raises(ValueError, match='body parts'):
        modelshare.import_model(p, package(tmp_path / 'n.zip', 'outline', {'classes': ['caudal']}))


def test_outline_parts_are_joined_to_the_body():
    from guppycolor import outline
    body = np.zeros((100, 200), bool); body[40:60, 20:120] = True
    fin = np.zeros_like(body); fin[35:65, 118:160] = True
    far = np.zeros_like(body); far[0:10, 180:200] = True
    other = np.zeros_like(body); other[80:95, 20:60] = True
    parts = outline.combine([(0, 0.5, other), (0, 0.9, body), (1, 0.3, far), (1, 0.8, fin)], ['body', 'caudal'])
    assert parts['_conf'] == 0.9 and (parts['body'] == body).all()
    assert parts['caudal'][50, 150] and not parts['caudal'][50, 110] and not parts['caudal'][5, 190]
    assert outline.combine([(1, 0.8, fin)], ['body', 'caudal']) == {}


def test_analysis_uses_the_outline_model(photos, monkeypatch):
    from guppycolor import outline, pipeline
    from guppycolor.engine import segment
    p = project(photos)
    p.settings['outline']['model'] = 'outline_v1'
    p.save()
    monkeypatch.setattr(segment, 'segment_fish', stub_segmenter)
    monkeypatch.setattr(segment, 'models', lambda: None)

    def model(project, name, img):
        mask = stub_segmenter(img)[0]
        ys, xs = np.nonzero(mask)
        cut = xs.min() + int(0.8 * np.ptp(xs))
        body = mask.copy(); body[:, cut:] = False
        return {'body': body, 'caudal': mask & ~body}, 0.9
    monkeypatch.setattr(outline, 'predict', model)
    pipeline.run(p)
    r = p.results()[0]
    assert r['outline_source'] == 'outline_v1' and r['caudal_in_mask'] and 'tail_incomplete' not in r['flags']
    assert r['body_area_px'] < r['fish_area_px'] and os.path.exists(
        os.path.join(p.dir('masks'), r['photo_id'] + '_caudal.png'))

    monkeypatch.setattr(outline, 'predict', lambda *a: ({}, 0.0))
    photo = r['path']
    r = pipeline.analyse(p, os.path.dirname(photo), photo)
    assert r['outline_source'] == 'detector' and 'outline_model_missed' in r['flags']

    def broken(project, name, img):
        m = np.zeros(img.shape[:2], bool); m[:, :40] = True             # a strip along the border
        return {'body': m}, 0.9
    monkeypatch.setattr(outline, 'predict', broken)
    r = pipeline.analyse(p, os.path.dirname(photo), photo)
    assert r['outline_source'] == 'detector' and 'outline_model_rejected' in r['flags']

    def boom(*a):
        raise ValueError('changed')
    monkeypatch.setattr(outline, 'predict', boom)
    r = pipeline.analyse(p, os.path.dirname(photo), photo)
    assert r['status'] != 'error' and 'outline_model_error' in r['flags']


def test_landmarks_never_trained_are_not_predicted(photos, monkeypatch):
    import torch
    from guppycolor import landmarks

    class Result:
        boxes = type('B', (), {'conf': torch.tensor([0.9]), '__len__': lambda self: 1})()
        keypoints = type('K', (), {'xy': torch.tensor([[[10., 10.], [20., 10.], [30., 5.]]]),
                                   'conf': torch.tensor([[0.9, 0.9, 0.9]])})()

    class Model:
        def predict(self, img, **k):
            assert k['imgsz'] == 1024
            return [Result()]
    monkeypatch.setattr(landmarks, 'load_model', lambda project, name: (
        Model(), ['snout', 'eye', None], {'imgsz': 1024}))
    mask = np.zeros((100, 200), bool); mask[40:60, 20:120] = True
    pts = landmarks.predict(None, 'landmarks_v1', np.zeros((100, 200, 3), np.uint8), mask)
    assert set(pts) == {'snout', 'eye'}


def test_ritocco_needs_a_model_with_the_same_landmarks(photos, monkeypatch):
    from guppycolor import landmarks
    from test_landmarks import analysed
    from conftest import PX_PER_MM
    p = analysed(photos)
    for _, ph in p.photos():
        landmarks.save(p, ph, {'snout': [650.0, 645.0], 'peduncle_end': [650.0 + 20 * PX_PER_MM, 650.0]})
    monkeypatch.setattr(landmarks, 'load_model', lambda project, name: (None, [], {'keys': ['head', 'tail']}))
    with pytest.raises(ValueError, match='other landmarks'):
        landmarks.train(p, start_from='landmarks_v1')

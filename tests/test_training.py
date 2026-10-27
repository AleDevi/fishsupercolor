import csv
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from conftest import PX_PER_MM, stub_segmenter, synthetic_photo
from guppycolor import export, pipeline, training, verify
from guppycolor.engine import pixels
from guppycolor.project import Project

SPOTS = [(30, 20), (40, 15), (25, 25), (35, 30), (20, 10)]       # radius (px) of the orange and the black spot


def filled_segmenter(img):
    """stub_segmenter with the holes left by coloured spots filled, as SAM does."""
    m, _, c = stub_segmenter(img)
    f = np.zeros(m.shape, np.uint8)
    cv2.drawContours(f, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], -1, 1, -1)
    return f > 0, None, c


@pytest.fixture
def spotted(tmp_path):
    folder = tmp_path / 'photos'
    folder.mkdir()
    for i, (r1, r2) in enumerate(SPOTS):
        p = synthetic_photo(str(folder / f'M{i}_B1_x.jpg'))
        im = np.array(Image.open(p))
        cv2.circle(im, (800, 640), r1, (230, 110, 40), -1)
        cv2.circle(im, (1000, 650), r2, (15, 15, 15), -1)
        Image.fromarray(im).save(p, quality=95)
    p = Project.create(str(tmp_path / 'pr'), 'x', [str(folder)])
    p.settings['filename_pattern'] = '{id}_{vasca}_{x}'
    p.save()
    pipeline.run(p, segmenter=filled_segmenter)
    return p


def paint_first(p):
    photo = sorted(x for _, x in p.photos())[0]
    img = training.corrected_image(p, photo)
    st = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(st, (800, 640), 22, 1, -1)            # arancio
    cv2.circle(st, (1000, 650), 16, 2, -1)           # nero
    cv2.circle(st, (700, 650), 25, 4, -1)            # corpo senza colore
    cv2.circle(st, (1100, 620), 15, 4, -1)
    training.save_strokes(p, photo, st)
    return photo


def test_brushes_follow_the_profile():
    from guppycolor import profile
    b = training.brushes(profile.load('guppy_male'))
    assert [x['key'] for x in b] == ['orange', 'black', 'iridescent', 'base', 'reflection']
    assert [x['index'] for x in b] == [1, 2, 3, 4, 5]
    prof = {'colour_classes': [{'key': f'c{i}', 'label': str(i), 'display': '#808080'} for i in range(7)]}
    assert len(training.brushes(prof)) == 9                       # more classes -> more brushes


def test_train_measure_and_export(spotted):
    p = spotted
    paint_first(p)
    name, meta = training.train(p)
    assert name == 'colours_v1' and meta['pixels_per_class']['orange'] > 0
    # the saved model is plain arrays: loading never unpickles
    with np.load(os.path.join(p.path, 'models', name + '.npz'), allow_pickle=False) as z:
        assert z['left'].dtype == np.int64
    p.settings['colours']['model'] = name
    p.save()
    assert len(p.pending()) == len(SPOTS)                          # a new model makes results stale
    pipeline.run(p, segmenter=filled_segmenter)
    with open(export.write_csv(p), encoding='utf-8') as f:
        rows = {r['id']: r for r in csv.DictReader(f)}
    for i, (r1, r2) in enumerate(SPOTS):
        r = rows[f'M{i}']
        true = np.pi * r1 ** 2 / PX_PER_MM ** 2
        assert abs(float(r['area_orange_mm2']) - true) / true < 0.15
        assert r['n_spots_orange'] == '1' and r['colour_model'] == name
        assert float(r['area_black_mm2']) > 0
    assert os.path.exists(os.path.join(p.path, 'check', p.results()[0]['photo_id'] + '_colours.jpg'))
    assert len(training.list_models(p)) == 1 and training.train(p)[0] == 'colours_v2'


def test_spots_merge_when_close():
    b = np.zeros((100, 200), bool)
    b[40:60, 20:40] = True
    b[40:60, 44:64] = True                                          # 4 px apart
    b[40:60, 150:170] = True
    assert pixels.spots(b) == [400, 400, 400]
    assert pixels.spots(b, merge_px=3) == [800, 400]
    assert pixels.spots(b, min_px=500) == []


def test_import_from_paint(spotted):
    p = spotted
    photo = sorted(x for _, x in p.photos())[0]
    im = pipeline.load_image(photo)
    cv2.circle(im, (800, 640), 10, training.hex_rgb('#FF7F27'), -1)     # Paint orange = arancio
    cv2.circle(im, (700, 650), 10, training.hex_rgb('#FFF200'), -1)     # Paint yellow = corpo senza colore
    painted = os.path.join(os.path.dirname(photo), 'M0_B1_x_strokes.png')
    Image.fromarray(im).save(painted)
    assert training.paint_counterpart(photo) == painted
    from guppycolor import profile
    n = training.import_paint(p, photo, painted, profile.load('guppy_male'))
    st = training.load_strokes(p, photo, im.shape)
    assert n > 0 and st[640, 800] == 1 and st[650, 700] == 4 and st[100, 100] == 0


def test_verify_against_manual_areas(spotted, tmp_path):
    p = spotted
    paint_first(p)
    p.settings['colours']['model'] = training.train(p)[0]
    p.save()
    pipeline.run(p, segmenter=filled_segmenter)
    manual = tmp_path / 'aree.csv'
    with open(manual, 'w') as f:
        f.write('maschio;area arancio;area nero\n')
        for i, (r1, r2) in enumerate(SPOTS):
            f.write(f'M{i};{np.pi * r1 ** 2 / 400:.3f};{np.pi * r2 ** 2 / 400:.3f}\n'.replace('.', ','))
    header, rows = verify.read_table(str(manual))
    app_rows, _ = export.rows(p)
    id_col, cols = verify.guess_columns(header, rows, app_rows, ['orange', 'black', 'iridescent'])
    assert id_col == 'maschio' and cols == {'orange': 'area arancio', 'black': 'area nero'}
    stats, csv_path, png = verify.compare(p, str(manual), id_col, cols)
    assert stats['orange']['n'] == 5 and stats['orange']['r'] > 0.95
    assert os.path.exists(csv_path) and os.path.exists(png)


def test_size_only_photos(spotted):
    p = spotted
    paint_first(p)
    p.settings['colours']['model'] = training.train(p)[0]
    p.settings['analysis']['size_only_pattern'] = r'^M4_'                    # M4 plays a female
    p.save()
    pipeline.run(p, segmenter=filled_segmenter)
    with open(export.write_csv(p), encoding='utf-8') as f:
        rows = {r['id']: r for r in csv.DictReader(f)}
    assert rows['M0']['analysis'] == 'size_colour' and rows['M0']['area_orange_mm2'] != 'NA'
    assert rows['M4']['analysis'] == 'size' and rows['M4']['area_orange_mm2'] == 'NA' and rows['M4']['fish_area_mm2'] != 'NA'
    photo = [x for _, x in p.photos() if 'M0_' in x][0]
    p.set_analysis(photo, 'size')
    with open(export.write_csv(p), encoding='utf-8') as f:
        assert {r['id']: r for r in csv.DictReader(f)}['M0']['area_orange_mm2'] == 'NA'
    pipeline.run(p, segmenter=filled_segmenter)
    assert p.load_result(photo)['analysis'] == 'size'                          # the manual choice survives
    p.settings['analysis'].update(size_only_pattern='', size_only_folders=[os.path.dirname(photo)])
    p.save()
    pipeline.run(p, segmenter=filled_segmenter)
    assert all(r['analysis'] == 'size' for r in p.results())


def test_optional_classes_are_off_in_new_projects_only(spotted, tmp_path):
    import json
    p = spotted
    assert p.settings['colours_off'] == ['iridescent'] and training.classes_off(p) == {'iridescent'}
    assert [b['key'] for b in training.active_brushes(p)] == ['orange', 'black', 'base', 'reflection']
    # a project made before 0.12 has no colours_off: everything stays measured and its results stay current
    with open(os.path.join(p.path, 'project.json'), encoding='utf-8') as f:
        data = json.load(f)
    data['settings'].pop('colours_off')
    with open(os.path.join(p.path, 'project.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f)
    old = Project(p.path)
    assert training.classes_off(old) == set() and 'colours_off' not in json.dumps(
        {k: old.settings[k] for k in ('colours',)})
    h = old.settings_hash()
    old.settings['colours_off'] = ['iridescent']
    assert old.settings_hash() != h                    # switching a class off makes the results stale


def test_class_switched_off_is_not_trained_and_is_na(spotted):
    p = spotted
    photo = paint_first(p)
    st = training.load_strokes(p, photo, pipeline.load_image(photo).shape)
    cv2.circle(st, (1000, 650), 8, 3, -1)              # iridescent strokes, not used while the class is off
    training.save_strokes(p, photo, st)
    name, meta = training.train(p)
    assert 'iridescent' not in [c['key'] for c in meta['classes']] and 3 not in training.load_model(p, name).classes
    p.settings['colours']['model'] = name
    p.save()
    pipeline.run(p, segmenter=filled_segmenter)
    with open(export.write_csv(p), encoding='utf-8') as f:
        r = next(csv.DictReader(f))
    assert r['area_iridescent_mm2'] == 'NA' and r['pct_iridescent'] == 'NA' and r['area_orange_mm2'] != 'NA'
    # a model that has the class, used where it is off: its pixels count as body without colour
    forest = training.load_model(p, name)
    img = training.corrected_image(p, photo)
    mask = training.fish_mask(p, photo)
    out, _, _ = training.measure_colours(img, mask, forest, off={'orange'})
    on, _, _ = training.measure_colours(img, mask, forest)
    assert 'orange' not in out and out['base']['area_px'] > on['base']['area_px']


def test_ritocco_starts_from_an_imported_model(spotted, tmp_path):
    from guppycolor import modelshare
    a = spotted
    paint_first(a)
    name, meta = training.train(a)
    assert meta['pool_pixels'] > 0 and training.pool_models(a)[0][0] == name
    again, ameta = training.train(a, start_from=name)    # its pixels are this project's strokes: not counted twice
    assert ameta['pool_pixels'] == meta['pool_pixels'] and ameta['pixels_per_class'] == meta['pixels_per_class']
    z = modelshare.export_model(a, name, str(tmp_path / 'c.zip'))
    folder = tmp_path / 'other'
    folder.mkdir()
    b = Project.create(str(tmp_path / 'b'), 'B', [str(folder)])
    got, gmeta = modelshare.import_model(b, z)
    X, keys, own = training.load_pool(b, got)
    assert gmeta['pool_pixels'] == len(keys) and not own.any() and set(keys) <= {'orange', 'black', 'base'}
    # no strokes in B yet: the imported pixels alone make a model; strokes would count OWN_WEIGHT times more
    new, nmeta = training.train(b, start_from=got)
    assert nmeta['started_from'] == got and nmeta['pixels_per_class'].get('orange', 0) == 0
    f1, f2 = training.load_model(b, got), training.load_model(b, new)
    F = X[:200]
    assert (f1.classes[f1.predict_proba(F).argmax(1)] == f2.classes[f2.predict_proba(F).argmax(1)]).mean() > 0.9


def test_import_colour_model_with_some_of_the_classes(tmp_path):
    from guppycolor import modelshare, profile
    folder = tmp_path / 'f'
    folder.mkdir()
    p = Project.create(str(tmp_path / 'p'), 'P', [str(folder)])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, pixels.N_FEATURES)).astype(np.float32)
    y = np.where(X[:, 0] > 0.5, 7, np.where(X[:, 0] < -0.5, 8, 9)).astype(np.uint8)   # other numbers, same keys
    f = pixels.Forest.train(X, y, trees=10)
    f.meta.update(classes=[{'index': 7, 'key': 'orange', 'label': 'o', 'display': '#E8742A'},
                           {'index': 8, 'key': 'black', 'label': 'b', 'display': '#222222'},
                           {'index': 9, 'key': 'base', 'label': 'x', 'display': '#C8C0A8'}],
                  features_version=pixels.FEATURES_VERSION)
    arrays = {k: getattr(f, k) for k in ('left', 'right', 'feature', 'threshold', 'proba', 'roots', 'classes')}
    pool = {'pool_X': X, 'pool_y': y, 'pool_own': np.ones(len(y), bool)}
    mine = training.brushes(profile.load('guppy_male'))
    arrays2, meta2, pool2 = modelshare.fit_colour_classes(arrays, f.meta, pool, mine)
    assert sorted(arrays2['classes'].tolist()) == [1, 2, 4] and set(pool2['pool_y'].tolist()) == {1, 2, 4}
    assert [c['index'] for c in meta2['classes']] == [1, 2, 4] and not pool2['pool_own'].any()
    bad = dict(f.meta, classes=f.meta['classes'][:2] + [{'index': 9, 'key': 'gills', 'label': 'g', 'display': '#000'}])
    with pytest.raises(ValueError, match='not among'):
        modelshare.fit_colour_classes(arrays, bad, pool, mine)
    with pytest.raises(ValueError, match='training pixels'):
        modelshare.fit_colour_classes(arrays, f.meta, dict(pool, pool_X=X[:, :5]), mine)

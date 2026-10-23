import os

import cv2
import numpy as np
import pytest

import synthetic_fish as sf

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def aligned_engine(pop, region='body'):
    from guppycolor.engine import pattern as eng
    P = [eng.outline_points(m, 10, region) for _, m, _ in pop]
    al, mean, cs = eng.gpa(np.array([p[1] for p in P]))
    cpts, size = eng.canvas_shape(mean, 11)
    tm = eng.template_mask(cpts, size, 10)
    return eng, cpts, tm, [eng.warp_labels(lab, p[0], cpts, tm) for p, (_, _, lab) in zip(P, pop)]


def test_alignment_puts_spots_back_in_place_mirrored_or_not():
    pop = sf.population(4, seed=3)
    eng, cpts, tm, amaps = aligned_engine(pop)
    for (k, _, _), a in zip(pop, amaps):
        if k != 'A':
            continue
        o = eng.find_spots(a == 1, cpts)          # orange spot of livery A at x 0.35, y +0.03 (dorsal)
        assert len(o) == 1 and o[0]['kind'] == 'point'
        assert abs(o[0]['x'] - 0.35) < 0.05 and abs(o[0]['y'] - 0.03) < 0.03


def test_points_and_lines():
    pop = sf.population(2, seed=4)
    eng, cpts, tm, amaps = aligned_engine(pop)
    kinds = {k: [s['kind'] for s in eng.find_spots(a == 1, cpts)] for (k, _, _), a in zip(pop, amaps)}
    assert kinds['A'] == ['point'] and kinds['B'] == ['line']


def test_gpa_removes_position_size_and_rotation():
    from guppycolor.engine import pattern as eng
    rng = np.random.default_rng(0)
    base = rng.normal(size=(12, 2))
    shapes = []
    for i in range(5):
        a = rng.uniform(0, 6.28)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        shapes.append(base @ R.T * rng.uniform(0.5, 3) + rng.normal(size=2) * 10)
    al, mean, cs = eng.gpa(shapes)
    assert np.abs(al - al[0]).max() < 1e-6


@pytest.mark.parametrize('distance', ['jaccard', 'spots'])
def test_clusters_find_the_liveries(distance):
    from guppycolor.engine import pattern as eng
    pop = sf.population(6, seed=1)
    eng, cpts, tm, amaps = aligned_engine(pop)
    valid = np.array([tm & (a > 0) for a in amaps])
    if distance == 'jaccard':
        D = eng.jaccard_matrix([eng.class_stack(a, v, [1, 2, 3], 2)[:, tm] for a, v in zip(amaps, valid)], valid[:, tm])
    else:
        D = eng.spots_matrix([{c: eng.find_spots(a == c, cpts) for c in (1, 2, 3)} for a in amaps], [1, 2, 3])
    Z = eng.linkage(D)
    truth = [k for k, _, _ in pop]
    assert eng.adjusted_rand(truth, list(eng.clusters_at(Z, len(D), 0.2))) == pytest.approx(1.0)
    assert eng.clusters_at(Z, len(D), 0.0).max() == 1                # no similarity needed: one cluster
    assert eng.clusters_at(Z, len(D), 1.0).max() >= 3


def test_head_side_told_by_the_user():
    from guppycolor.engine import pattern as eng
    m, _ = sf.draw(sf.LIVERIES['A'], mirror=False, depth=0.13)
    left, _ = eng.outline_points(m, 10, 'body', head='left')
    right, _ = eng.outline_points(m, 10, 'body', head='right')
    assert left[0][0] < left[11][0] and right[0][0] > right[11][0]


def test_composition_ignores_position():
    from guppycolor.engine import pattern as eng
    D = eng.composition_matrix([[0.2, 0.1], [0.2, 0.1], [0.0, 0.3]])
    assert D[0, 1] == 0 and 0 < D[0, 2] <= 1


def fake_project(tmp_path, n_per=4):
    """A project whose results hold synthetic masks and colour maps, as the pipeline would leave them."""
    from guppycolor import pattern
    from guppycolor.project import Project
    photos = tmp_path / 'photos'
    photos.mkdir()
    for k, (liv, m, lab) in enumerate(sf.population(n_per, seed=2)):
        cv2.imwrite(str(photos / f'M{k:03d}.jpg'), np.zeros((8, 8, 3), np.uint8))
    p = Project.create(str(tmp_path / 'proj'), 'Pattern test', [str(photos)])
    p.settings['pattern'] = {'head': 'auto'}              # the synthetic fish face both ways
    pop = sf.population(n_per, seed=2)
    for (folder, photo), (liv, m, lab) in zip(sorted(p.photos(), key=lambda t: t[1]), pop):
        pid = p.photo_id(photo)
        cv2.imwrite(os.path.join(p.dir('masks'), pid + '.png'), m.astype(np.uint8) * 255)
        r = {'photo_id': pid, 'path': photo, 'file': os.path.basename(photo), 'folder': os.path.basename(folder),
             'status': 'ok', 'colours': {'orange': {'area_px': 1}}, 'label_id': os.path.basename(photo)[:4],
             'label_group': 'G' + liv}
        r['colour_map'] = pattern.save_colour_map(p, pid, (lab, (0, 0, lab.shape[1], lab.shape[0]), 1.0))
        p.save_result(photo, r)
    return p


def test_project_build_cluster_and_export(tmp_path, monkeypatch):
    import csv
    from guppycolor import pattern
    monkeypatch.setattr(pattern, 'model_classes', lambda project: (sf.CLASSES, None))
    p = fake_project(tmp_path)
    ps = pattern.build(p)
    assert len(ps.rows) == 12
    assert not ps.valid[:, :, :int(ps.canvas_pts[0][0] + 0.1 * 260)].any()     # the head is left out
    cl = pattern.Clustering(ps)
    labels = cl.at(0.2)
    assert cl.agreement(labels) == pytest.approx(1.0)
    for d in ('spots', 'composition'):
        ps.st = dict(ps.st, distance=d)
        assert pattern.Clustering(ps).D.shape == (12, 12)
    d = pattern.export(p, ps, cl, 0.2)
    for name in ('pattern_fish', 'pattern_spots', 'pattern_grid', 'pattern_shape', 'pattern_distances',
                 'pattern_settings'):
        with open(os.path.join(d, name + '.csv'), encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert rows, name
    with open(os.path.join(d, 'pattern_fish.csv'), encoding='utf-8') as f:
        fish = list(csv.DictReader(f))
    assert {r['cluster'] for r in fish} == {'1', '2', '3'}
    assert os.path.exists(os.path.join(d, 'cluster_01.png'))
    assert pattern.build(p).rows                     # second time from the outline cache


def test_pattern_screen(tmp_path, monkeypatch):
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from guppycolor import pattern
    from guppycolor.gui.pattern_page import PatternPage
    monkeypatch.setattr(pattern, 'model_classes', lambda project: (sf.CLASSES, None))
    app = QApplication.instance() or QApplication([])
    p = fake_project(tmp_path, 3)

    class Win:
        project = p
    page = PatternPage(Win())
    page.resize(1250, 760)
    page.show_project(p)
    assert len(page.class_checks) == 3 and page.b_run.isEnabled()
    loop = QEventLoop()
    page.compute(realign=True)
    page.worker.finished.connect(lambda *_: QTimer.singleShot(0, loop.quit))
    QTimer.singleShot(60000, loop.quit)
    loop.exec()
    app.processEvents()
    assert page.cl is not None
    page.slider.setValue(20)
    n_low = page.clusters.count()
    page.slider.setValue(95)
    assert page.clusters.count() >= n_low and page.members.count() >= 1
    page.slider.setValue(30)
    app.processEvents()
    out = os.environ.get('GUPPY_SHOTS')
    if out:
        page.grab().save(os.path.join(out, 'schermata_6.png'))
    assert p.settings['pattern']['similarity'] == pytest.approx(0.3)

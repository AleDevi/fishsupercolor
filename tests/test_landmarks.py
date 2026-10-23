import csv
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from conftest import PX_PER_MM, stub_segmenter


def analysed(photos):
    from guppycolor import pipeline
    from guppycolor.project import Project
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['analysis']['default'] = 'size'
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    return p


def test_clicked_landmarks_give_standard_length(photos):
    from guppycolor import export, landmarks
    p = analysed(photos)
    photo = [ph for _, ph in p.photos()][0]
    assert p.load_result(photo).get('standard_length_px') is None
    landmarks.save(p, photo, {'snout': [640.0, 640.0]})
    p.save_result(photo, dict(p.load_result(photo), flags='sl_landmarks_missing'))
    landmarks.save(p, photo, {'snout': [640.0, 640.0], 'peduncle_end': [640.0 + 300, 640.0]})
    r = p.load_result(photo)
    assert r['landmarks_source'] == 'manual' and 'sl_landmarks_missing' not in r['flags']
    path = export.write_csv(p)
    with open(path, encoding='utf-8-sig') as f:
        rows = {row['path']: row for row in csv.DictReader(f)}
    sl = float(rows[photo]['standard_length_mm'])
    ppm = float(rows[photo]['px_per_mm'])
    assert abs(sl - 300 / ppm) < 0.02 and abs(ppm - PX_PER_MM) < 1.5
    landmarks.save(p, photo, {})
    assert landmarks.load(p, photo) == {}


def test_pose_dataset_format(photos):
    from guppycolor import landmarks
    p = analysed(photos)
    keys = ['snout', 'eye', 'peduncle_end']
    for _, ph in p.photos():
        landmarks.save(p, ph, {'snout': [650.0, 645.0], 'peduncle_end': [1140.0, 650.0]})
    yaml, n = landmarks.build_dataset(p, os.path.join(p.dir('training'), 'ds'), keys)
    assert n == 3
    text = open(yaml, encoding='utf-8').read()
    assert 'kpt_shape: [3, 3]' in text and 'flip_idx: [0, 1, 2]' in text
    lab_dir = os.path.join(os.path.dirname(yaml), 'labels', 'train')
    v = open(os.path.join(lab_dir, os.listdir(lab_dir)[0])).read().split()
    assert len(v) == 5 + 3 * 3 and v[0] == '0'
    assert v[5 + 2] == '2' and v[8:11] == ['0', '0', '0'] and v[11 + 2] == '2'
    assert all(0 <= float(x) <= 1 for x in v[1:5] + v[5:7] + v[11:13])


def test_landmarks_screen_click_undo(photos):
    from PySide6.QtWidgets import QApplication
    from guppycolor import landmarks
    from guppycolor.gui.main import MainWindow
    app = QApplication.instance() or QApplication([])
    p = analysed(photos)
    w = MainWindow()
    w.pages[0].open(p.path)
    w.nav.setCurrentRow(7)
    app.processEvents()
    page = w.pages[7]
    assert len(page.items) == 3 and page.keys[0]['key'] == 'snout'
    page.place(10, 10)
    page.place(30, 12)
    assert list(landmarks.load(p, page.photo)) == [page.keys[0]['key'], page.keys[1]['key']]
    page.undo()
    assert list(landmarks.load(p, page.photo)) == [page.keys[0]['key']]
    page.skip()
    assert page.current == 2
    page.clear()
    assert landmarks.load(p, page.photo) == {}


def test_scale_auto_line_and_manual_calibration(photos):
    from guppycolor import pipeline, scaleline
    from guppycolor.project import Project
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['scale']['mode'] = 'manual'
    p.settings['analysis']['default'] = 'size'
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    res = p.results()
    assert all(r['status'] == 'to_check' and 'scale_to_set' in r['flags'] for r in res)
    ppm = scaleline.save(p, res[0]['path'], [100, 500], [100 + 20 * PX_PER_MM, 500], 20, whole_folder=True)
    assert abs(ppm - PX_PER_MM) < 1e-6
    res = p.results()
    assert [r['scale_source'] for r in res] == ['line', 'line_folder', 'line_folder']
    assert all(r['status'] == 'ok' and r['scale_ok'] and 'scale_to_set' not in r['flags'] for r in res)
    pipeline.analyse(p, res[1]['path'].rsplit(os.sep, 1)[0], res[1]['path'], segmenter=stub_segmenter)
    assert p.load_result(res[1]['path'])['scale_source'] == 'line_folder'      # survives a new analysis


def test_auto_scale_gives_a_2cm_line(photos):
    from guppycolor import pipeline
    from guppycolor.project import Project
    import numpy as np
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    pipeline.run(p, segmenter=stub_segmenter)
    r = p.results()[0]
    assert r['scale_ok'] and r['scale_squares'] >= 20
    (ax, ay), (bx, by) = r['scale_line']
    assert abs(np.hypot(bx - ax, by - ay) / r['px_per_mm'] - 20) < 0.1


def test_review_confirm_and_calibrate(photos, monkeypatch):
    from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox
    from guppycolor import pipeline
    from guppycolor.gui.main import MainWindow
    from guppycolor.project import Project
    import guppycolor.gui.main as gm
    app = QApplication.instance() or QApplication([])
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['scale']['mode'] = 'manual'
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    w = MainWindow()
    w.resize(1280, 800)
    w.pages[0].open(p.path)
    w.nav.setCurrentRow(3)
    app.processEvents()
    page = w.pages[3]
    assert len(page.items) == 3
    import guppycolor.gui.calibrate as cal
    monkeypatch.setattr(cal.CalibrateDialog, 'exec', lambda self: True)
    monkeypatch.setattr(cal.CalibrateDialog, 'result_line',
                        lambda self: ((100, 500), (100 + 20 * PX_PER_MM, 500), 20.0, False))
    page.start_calibration()
    assert len(page.items) == 2
    assert p.results()[0]['scale_source'] == 'line'


def test_calibrate_dialog(photos):
    from PySide6.QtWidgets import QApplication
    from guppycolor.gui.calibrate import CalibrateDialog
    app = QApplication.instance() or QApplication([])
    photo = str(sorted((photos / 'sessione1').iterdir())[0])
    d = CalibrateDialog(None, photo, 'sessione1')
    d.click(100, 500)
    d.click(500, 500)
    a, b, mm, whole = d.result_line()
    assert (a, b, mm, whole) == ((100, 500), (500, 500), 20, False)
    d.click(10, 10)
    assert len(d.points) == 1

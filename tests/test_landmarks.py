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


def test_training_fonts_are_local(tmp_path, monkeypatch):
    """Training wants Arial.ttf; on Windows ultralytics does not see arial.ttf and tries to download it."""
    import ultralytics.utils
    from guppycolor.engine import segment
    monkeypatch.setattr(ultralytics.utils, 'USER_CONFIG_DIR', tmp_path)
    segment.local_fonts()
    assert (tmp_path / 'Arial.ttf').stat().st_size > 100_000 and (tmp_path / 'Arial.Unicode.ttf').exists()


def test_fin_with_rounded_edge_is_found():
    """A round caudal fin narrows again at its edge: the peduncle is before the widest part of the fin."""
    import cv2
    import numpy as np
    from guppycolor.engine import morpho
    m = np.zeros((600, 1400), np.uint8)
    cv2.ellipse(m, (500, 300), (330, 90), 0, 0, 360, 1, -1)          # body, head on the left
    cv2.rectangle(m, (780, 265), (900, 335), 1, -1)                   # caudal peduncle
    cv2.circle(m, (1010, 300), 150, 1, -1)                            # round fan
    o = morpho.measure(m.astype(bool), None, head='left')
    assert o['fin_found'] and o['sl_px'] < 0.75 * o['tl_px']


def test_total_length_from_landmarks(photos):
    from guppycolor import export, landmarks
    p = analysed(photos)
    photo = [ph for _, ph in p.photos()][0]
    r = p.load_result(photo)
    p.save_result(photo, dict(r, total_length_px=None, total_length_source=None, flags='tail_incomplete'))
    landmarks.save(p, photo, {'snout': [600.0, 650.0], 'tail_tip': [600.0 + 400, 650.0]})
    assert p.load_result(photo)['total_length_source'] == 'landmarks'
    with open(export.write_csv(p), encoding='utf-8') as f:
        rows = {row['path']: row for row in csv.DictReader(f)}
    assert rows[photo]['total_length_source'] == 'landmarks'
    assert abs(float(rows[photo]['total_length_mm']) * float(rows[photo]['px_per_mm']) - 400) < 1


def test_add_to_outline(monkeypatch):
    import numpy as np
    from guppycolor.engine import segment
    mask = np.zeros((200, 400), bool)
    mask[80:120, 50:250] = True
    fin = np.zeros_like(mask)
    fin[60:140, 245:330] = True
    island = np.zeros_like(mask)
    island[10:20, 10:20] = True
    monkeypatch.setattr(segment, 'segment_fish_click', lambda img, x, y, more=(): mask | fin | island)
    new = segment.add_to_mask(np.zeros((200, 400, 3), np.uint8), mask, 300, 100)
    assert new[100, 300] and new[85, 60] and not new[15, 15]            # fin joined, far island left out
    monkeypatch.setattr(segment, 'segment_fish_click', lambda img, x, y, more=(): island)
    assert segment.add_to_mask(np.zeros((200, 400, 3), np.uint8), mask, 15, 15) is None


def test_no_console_goes_to_log(tmp_path, monkeypatch):
    """Started from the desktop icon (pythonw) sys.stdout/stderr are None; training progress bars must not crash."""
    import sys
    from guppycolor.__main__ import quiet_console
    monkeypatch.setattr(sys, 'stdout', None)
    monkeypatch.setattr(sys, 'stderr', None)
    log = tmp_path / 'sub' / 'guppycolour.log'
    f = quiet_console(str(log))
    try:
        sys.stderr.write('progress\n')
        print('hello')
        assert log.read_text(encoding='utf-8') == 'progress\nhello\n'
    finally:
        f.close()


def test_vertical_fish_head_is_found_by_its_eye():
    """'Head on the left' only applies to a fish lying sideways; a vertical one is judged by its eye."""
    import cv2
    import numpy as np
    from guppycolor.engine import morpho
    img = np.full((900, 400, 3), 235, np.uint8)
    cv2.ellipse(img, (200, 400), (60, 260), 0, 0, 360, (120, 110, 95), -1)
    cv2.circle(img, (200, 185), 14, (10, 10, 10), -1)                 # eye near the top
    mask = cv2.inRange(img, (100, 100, 80), (140, 130, 110)) > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8)).astype(bool)
    o = morpho.measure(mask, img, head='left')
    assert o['snout'][1] < 300 and o['end'][1] > 500
    a, b = o['depth_line']
    assert abs(abs(a[0] - b[0]) - o['max_depth_px']) < 15


def test_results_show_sl_and_measures_picture(photos):
    from PySide6.QtWidgets import QApplication
    from guppycolor import landmarks
    from guppycolor.gui.main import MainWindow
    app = QApplication.instance() or QApplication([])
    p = analysed(photos)
    photo = [ph for _, ph in p.photos()][0]
    landmarks.save(p, photo, {'snout': [650.0, 645.0], 'peduncle_end': [1140.0, 650.0]})
    w = MainWindow()
    w.project = p
    page = w.pages[4]
    page.show_project(p)
    assert 'standard_length_mm' in page.COLS and 'total_length_mm' in page.COLS
    assert page.summary.text().startswith('Standard length (SL) on 1 of 3 photos')
    row = [i for i, r in enumerate(page.rows) if r['path'] == photo][0]
    page.show_row(row)
    assert page.pm is not None and not page.pm.isNull()
    w.close()


def test_crop_has_room_at_head_and_tail():
    import numpy as np
    from guppycolor import landmarks
    mask = np.zeros((1000, 2000), bool)
    mask[450:550, 600:1400] = True
    x0, y0, x1, y1 = landmarks.crop_box(mask, mask.shape)
    assert 600 - x0 >= 0.29 * 800 and x1 - 1400 >= 0.29 * 800 and 450 - y0 < 600 - x0

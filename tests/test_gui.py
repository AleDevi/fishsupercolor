import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from conftest import stub_segmenter


def test_window_runs_an_analysis(photos, monkeypatch, tmp_path):
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from guppycolor.engine import segment
    from guppycolor.gui.main import MainWindow
    from guppycolor.project import Project
    monkeypatch.setattr(segment, 'segment_fish', stub_segmenter)
    monkeypatch.setattr(segment, 'models', lambda: None)
    app = QApplication.instance() or QApplication([])
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1'), str(photos / 'sessione2')])
    w = MainWindow()
    w.pages[0].open(p.path)
    assert w.project is not None and w.stack.currentIndex() == 1
    w.nav.setCurrentRow(2)
    page = w.pages[2]
    loop = QEventLoop()
    page.done.connect(loop.quit)
    QTimer.singleShot(120000, loop.quit)
    page.start()
    loop.exec()
    assert w.project.pending() == []
    assert page.grid.count() == 4
    out = os.environ.get('GUPPY_SHOTS')
    for i in (0, 1, 2, 3, 4):
        w.nav.setCurrentRow(i)
        app.processEvents()
        if out:
            w.grab().save(os.path.join(out, f'schermata_{i + 1}.png'))
    assert w.pages[4].table.rowCount() == 4


def test_click_in_review_fixes_missing_fish(photos, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from guppycolor import pipeline
    from guppycolor.engine import segment
    from guppycolor.gui.main import MainWindow
    from guppycolor.project import Project
    monkeypatch.setattr(segment, 'segment_fish', lambda img: (None, None, 0))
    monkeypatch.setattr(segment, 'models', lambda: None)
    monkeypatch.setattr(segment, 'segment_fish_click', lambda img, x, y: stub_segmenter(img)[0])
    app = QApplication.instance() or QApplication([])
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione2')])
    pipeline.run(p)
    w = MainWindow()
    w.resize(1280, 800)
    w.show()
    w.pages[0].open(p.path)
    w.nav.setCurrentRow(3)
    app.processEvents()
    page = w.pages[3]
    assert len(page.items) == 1
    page.show_item(0)
    page.click_fish(page.image.width() / 2, page.image.height() / 2)
    assert p.results()[0]['status'] == 'ok' and page.items == []


def test_label_fixed_in_review(photos, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from guppycolor import pipeline
    from guppycolor.engine import labels, segment
    from guppycolor.gui.main import MainWindow
    from guppycolor.project import Project
    from test_app import fake_read_labels
    monkeypatch.setattr(segment, 'segment_fish', stub_segmenter)
    monkeypatch.setattr(segment, 'models', lambda: None)
    monkeypatch.setattr(labels, 'read_labels', fake_read_labels)
    app = QApplication.instance() or QApplication([])
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['labels']['id_source'] = 'label'
    p.save()
    pipeline.run(p)
    w = MainWindow()
    w.resize(1280, 800)
    w.show()
    w.pages[0].open(p.path)
    w.nav.setCurrentRow(1)
    app.processEvents()
    assert w.pages[1].id_src.checkedButton().property('key') == 'label'
    w.nav.setCurrentRow(3)
    app.processEvents()
    page = w.pages[3]
    assert len(page.items) == 1 and page.lab_box.isVisible() and page.lab_img.pixmap() is not None
    page.lab_id.setText('m117')
    page.save_label()
    assert p.load_result(page.items[0]['path'])['label_id'] == 'M117'


def test_print_labels_button(photos, monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
    from PySide6.QtGui import QDesktopServices
    from guppycolor.gui.main import MainWindow
    from guppycolor.project import Project
    lst = tmp_path / 'pesci.csv'
    lst.write_text('id,gruppo\nM116,4\nM117,4\n')
    out = tmp_path / 'etichette.pdf'
    answers = iter(['id', 'gruppo', '20/03'])
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(lst), ''))
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (str(out), ''))
    monkeypatch.setattr(QInputDialog, 'getText', lambda *a, **k: (next(answers), True))
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: None)
    monkeypatch.setattr(QDesktopServices, 'openUrl', lambda *a, **k: True)
    app = QApplication.instance() or QApplication([])
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    w = MainWindow()
    w.pages[0].open(p.path)
    w.pages[1].print_labels()
    assert out.exists() and out.read_bytes()[:4] == b'%PDF'


def test_training_screen_paints_and_trains(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from PIL import Image
    from PySide6.QtCore import QEventLoop, QPointF, QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from conftest import synthetic_photo
    from test_training import filled_segmenter
    from guppycolor import pipeline, training
    from guppycolor.gui.main import MainWindow
    from guppycolor.project import Project
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **k: None)
    folder = tmp_path / 'foto'
    folder.mkdir()
    p = synthetic_photo(str(folder / 'a.jpg'))
    im = np.array(Image.open(p))
    cv2.circle(im, (800, 640), 30, (230, 110, 40), -1)
    Image.fromarray(im).save(p, quality=95)
    pr = Project.create(str(tmp_path / 'pr'), 'x', [str(folder)])
    pipeline.run(pr, segmenter=filled_segmenter)
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.pages[0].open(pr.path)
    w.nav.setCurrentRow(6)
    page = w.pages[6]
    assert w.stack.currentIndex() == 6 and len(page.brush_group.buttons()) == 6     # 3 classes + base + reflection + eraser
    x0, y0 = page.current[2][:2]
    page.view.brush = 1
    page.view.dab(QPointF(800 - x0, 640 - y0), QPointF(805 - x0, 640 - y0))
    page.view.brush = 4
    page.view.dab(QPointF(700 - x0, 650 - y0), QPointF(720 - x0, 650 - y0))
    page.save_current()
    st = training.load_strokes(pr, p, im.shape)
    assert st[640, 802] == 1 and st[650, 710] == 4
    loop = QEventLoop()
    page.worker = None
    page.train()
    page.worker.done.connect(lambda *a: QTimer.singleShot(0, loop.quit))
    page.worker.failed.connect(lambda *a: QTimer.singleShot(0, loop.quit))
    QTimer.singleShot(120000, loop.quit)
    loop.exec()
    app.processEvents()
    assert w.project.settings['colours']['model'] == 'colours_v1'
    assert not page.view.pred.pixmap().isNull()
    out = os.environ.get('GUPPY_SHOTS')
    if out:
        w.grab().save(os.path.join(out, 'schermata_7.png'))

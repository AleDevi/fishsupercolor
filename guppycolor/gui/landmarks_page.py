"""Screen 8, Landmarks: click the profile's named landmarks in order on a photo (snout, eye, dorsal base…).
Standard length comes straight from snout and end of the peduncle; with 10–30 photos clicked, a YOLO-pose
model can be trained to place them on every other photo."""
import os

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from .. import landmarks, pipeline, profile, training
from . import modelfiles
from .training_page import to_qimage

COLOURS = ['#E53935', '#FDD835', '#43A047', '#1E88E5', '#8E24AA', '#FB8C00', '#00ACC1', '#6D4C41', '#D81B60']


class ClickView(QGraphicsView):
    """Photo crop; left click = place the current landmark, right button drags, wheel zooms."""
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor('#30343A'))
        self.photo = QGraphicsPixmapItem()
        self.scene().addItem(self.photo)
        self.marks = []
        self._pan = None
        self.zoomed = False

    def set_photo(self, rgb):
        self.photo.setPixmap(QPixmap.fromImage(to_qimage(rgb)))
        self.scene().setSceneRect(QRectF(0, 0, rgb.shape[1], rgb.shape[0]))
        self.zoomed = False
        self.fitInView(self.photo, Qt.KeepAspectRatio)

    def show_points(self, pts, radius):
        for m in self.marks:
            self.scene().removeItem(m)
        self.marks = []
        for label, (x, y), colour in pts:
            e = QGraphicsEllipseItem(x - radius, y - radius, 2 * radius, 2 * radius)
            e.setPen(QPen(QColor('white'), radius / 3))
            e.setBrush(QBrush(QColor(colour)))
            t = QGraphicsSimpleTextItem(label)
            f = QFont()
            f.setPixelSize(int(radius * 3))
            t.setFont(f)
            t.setBrush(QBrush(QColor('white')))
            t.setPos(x + radius * 1.5, y - radius * 3.5)
            for it in (e, t):
                it.setZValue(5)
                self.scene().addItem(it)
                self.marks.append(it)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self.zoomed:
            self.fitInView(self.photo, Qt.KeepAspectRatio)

    def wheelEvent(self, e):
        f = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.zoomed = True
        self.scale(f, f)

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
        elif e.button() == Qt.LeftButton:
            p = self.mapToScene(e.position().toPoint())
            self.clicked.emit(p.x(), p.y())

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()


class TrainWorker(QObject):
    done = Signal(str, dict)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, project, start_from=None):
        super().__init__()
        self.project = project
        self.start_from = start_from

    def run(self):
        try:
            self.done.emit(*landmarks.train(self.project, progress=self.progress.emit, start_from=self.start_from))
        except Exception as e:
            self.failed.emit(f'{type(e).__name__}: {e}')


class LandmarksPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.project = None
        self.items = []
        self.photo = None
        self.box = (0, 0, 1, 1)
        self.points = {}
        self.current = 0
        self.thread = None
        lay = QVBoxLayout(self)
        title = QLabel('8  Landmarks')
        title.setStyleSheet('font-size: 20px; font-weight: 600; margin-bottom: 8px')
        lay.addWidget(title)
        hint = QLabel('Click the landmarks in the order of the list (the highlighted one is next). S skips an optional '
                      'one, Backspace undoes, N goes to the next photo. Standard length = snout to end of the peduncle. '
                      'With 10–30 photos clicked, Train places them on every photo (a GPU makes it much faster).')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #4F5761')
        lay.addWidget(hint)
        split = QSplitter()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.show_item)
        split.addWidget(self.list)
        self.view = ClickView()
        self.view.clicked.connect(self.place)
        split.addWidget(self.view)
        right = QWidget()
        rl = QVBoxLayout(right)
        g = QGroupBox('Landmarks of the profile')
        gl = QVBoxLayout(g)
        self.marks = QListWidget()
        self.marks.currentRowChanged.connect(lambda i: setattr(self, 'current', max(i, 0)))
        gl.addWidget(self.marks)
        rl.addWidget(g)
        self.sl = QLabel('')
        rl.addWidget(self.sl)
        for text, fn in (('Skip (S)', self.skip), ('Undo (Backspace)', self.undo), ('Clear this photo', self.clear),
                         ('Next photo (N)', self.next_photo)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            rl.addWidget(b)
        for key, fn in (('S', self.skip), ('Backspace', self.undo), ('N', self.next_photo)):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)
        mbox = QGroupBox('Landmark model')
        ml = QVBoxLayout(mbox)
        f = QFormLayout()
        self.start_from = QComboBox()
        self.start_from.setToolTip('Ritocco: start from a model already trained on many photos (e.g. the imported van '
                                   'der Bijl one) and refine it with your clicks; faster and better with few photos.')
        f.addRow('Start from', self.start_from)
        ml.addLayout(f)
        self.b_train = QPushButton('Train')
        self.b_train.clicked.connect(self.train)
        ml.addWidget(self.b_train)
        self.bar = QProgressBar()
        self.bar.hide()
        ml.addWidget(self.bar)
        self.bar_text = QLabel('')
        self.bar_text.setWordWrap(True)
        self.bar_text.hide()
        ml.addWidget(self.bar_text)
        row = QHBoxLayout()
        for text, fn in (('Export model…', lambda: modelfiles.export_dialog(self, self.project, self.model.currentData())),
                         ('Import model…', self.import_model)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        ml.addLayout(row)
        f = QFormLayout()
        self.model = QComboBox()
        self.model.activated.connect(self.use_model)
        f.addRow('Use in the analysis', self.model)
        ml.addLayout(f)
        self.model_info = QLabel('')
        self.model_info.setWordWrap(True)
        ml.addWidget(self.model_info)
        rl.addWidget(mbox)
        rl.addStretch()
        split.addWidget(right)
        split.setSizes([240, 760, 280])
        lay.addWidget(split, 1)
        self.keys = []

    def show_project(self, p):
        self.project = p
        self.keys = landmarks.landmark_list(profile.load(p.settings['profile']))
        self.fill_models()
        done = {ph for ph, _ in landmarks.annotated(p)}
        res = {r['path']: r for r in p.results()}
        self.items = [ph for _, ph in p.photos() if training.fish_mask(p, ph) is not None
                      and res.get(ph, {}).get('status') != 'excluded']
        self.items.sort(key=lambda ph: (ph not in done, os.path.basename(ph)))
        self.list.blockSignals(True)
        self.list.clear()
        for ph in self.items:
            self.list.addItem(QListWidgetItem(os.path.basename(ph) + ('   ✓' if ph in done else '')))
        self.list.blockSignals(False)
        if self.items:
            self.list.setCurrentRow(0)

    def show_item(self, i):
        if not 0 <= i < len(self.items):
            return
        p, self.photo = self.project, self.items[i]
        img = pipeline.load_image(self.photo)
        mask = training.fish_mask(p, self.photo)
        self.box = landmarks.crop_box(mask, img.shape)
        x0, y0, x1, y1 = self.box
        self.view.set_photo(img[y0:y1, x0:x1])
        self.points = landmarks.load(p, self.photo)
        self.current = next((j for j, l in enumerate(self.keys) if l['key'] not in self.points), 0)
        self.refresh()

    def refresh(self):
        x0, y0, x1, y1 = self.box
        r = max(3.0, (x1 - x0) / 250)
        shown = [(l['key'], (self.points[l['key']][0] - x0, self.points[l['key']][1] - y0), COLOURS[j % len(COLOURS)])
                 for j, l in enumerate(self.keys) if l['key'] in self.points]
        self.view.show_points(shown, r)
        self.marks.blockSignals(True)
        self.marks.clear()
        for j, l in enumerate(self.keys):
            state = '✓' if l['key'] in self.points else ('optional' if not l.get('required') else '')
            it = QListWidgetItem(f"{j + 1}. {l['label']}   {state}")
            it.setForeground(QColor(COLOURS[j % len(COLOURS)]))
            self.marks.addItem(it)
        self.marks.setCurrentRow(min(self.current, len(self.keys) - 1))
        self.marks.blockSignals(False)
        sl = landmarks.standard_length_px(self.points)
        res = self.project.load_result(self.photo) or {}
        ppm = res.get('px_per_mm')
        self.sl.setText('Standard length: ' + (f'{sl / ppm:.2f} mm' if sl and ppm else f'{sl:.0f} px' if sl else '—'))

    def place(self, x, y):
        if not self.photo or not self.keys or self.current >= len(self.keys):
            return
        x0, y0 = self.box[:2]
        self.points[self.keys[self.current]['key']] = [round(x + x0, 1), round(y + y0, 1)]
        landmarks.save(self.project, self.photo, self.points)
        self.advance()

    def advance(self):
        nxt = [j for j, l in enumerate(self.keys) if l['key'] not in self.points and j > self.current]
        self.current = nxt[0] if nxt else len(self.keys)
        self.refresh()
        if self.current >= len(self.keys):
            it = self.list.currentItem()
            if it and not it.text().endswith('✓'):
                it.setText(it.text() + '   ✓')

    def skip(self):
        if self.current < len(self.keys):
            self.advance()

    def undo(self):
        placed = [j for j, l in enumerate(self.keys) if l['key'] in self.points]
        if placed:
            j = max(placed) if self.current >= len(self.keys) else max([k for k in placed if k < self.current] or placed)
            del self.points[self.keys[j]['key']]
            landmarks.save(self.project, self.photo, self.points)
            self.current = j
            self.refresh()

    def clear(self):
        self.points = {}
        landmarks.save(self.project, self.photo, self.points)
        self.current = 0
        self.refresh()

    def next_photo(self):
        self.list.setCurrentRow(min(self.list.currentRow() + 1, len(self.items) - 1))

    # ---------------------------------------------------------------- model
    def fill_models(self):
        self.model.clear()
        self.model.addItem('none (only landmarks clicked by hand)', '')
        for name, meta in reversed(landmarks.list_models(self.project)):
            self.model.addItem(f"{name}  ({meta.get('date', '')[:10]}, {meta.get('photos', 0)} photos)", name)
        cur = self.project.settings['landmarks'].get('model', '')
        self.model.setCurrentIndex(max(0, self.model.findData(cur)))
        self.model_info.setText('Clicked landmarks always win over the model.')
        self.start_from.clear()
        self.start_from.addItem('a general model (yolo11n-pose, from scratch)', '')
        models = landmarks.list_models(self.project)
        for name, meta in reversed(models):
            src = 'imported' if meta.get('imported_from') else f"{meta.get('photos', 0)} photos"
            self.start_from.addItem(f'{name} ({src}): Ritocco', name)
        imported = [n for n, m in models if m.get('imported_from')]      # pre-trained models are the best start
        self.start_from.setCurrentIndex(max(0, self.start_from.findData(imported[-1])) if imported else 0)

    def use_model(self, i):
        self.project.settings['landmarks']['model'] = self.model.itemData(i)
        self.project.save()

    def train(self):
        self.b_train.setEnabled(False)
        self.b_train.setText('Training… (minutes on a GPU, longer on a CPU)')
        self.worker = TrainWorker(self.project, self.start_from.currentData() or None)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.done.connect(self.on_trained)
        self.worker.failed.connect(self.on_failed)
        self.worker.progress.connect(self.on_progress)
        self.bar.setValue(0)
        self.bar.show()
        self.bar_text.show()
        self.thread.start()

    def on_progress(self, i, n, text):
        self.bar.setMaximum(max(n, 1))
        self.bar.setValue(i)
        self.bar_text.setText(text)

    def import_model(self):
        name = modelfiles.import_dialog(self, self.project, 'landmarks')
        if name:
            self.project.settings['landmarks']['model'] = name
            self.project.save()
            self.fill_models()

    def _stop(self):
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.bar.hide()
        self.bar_text.hide()
        self.b_train.setEnabled(True)
        self.b_train.setText('Train')

    def on_trained(self, name, meta):
        self._stop()
        self.project.settings['landmarks']['model'] = name
        self.project.save()
        self.fill_models()
        QMessageBox.information(self, 'Landmarks', f'Model {name} is ready and in use. Run the Analysis again to '
                                'get the standard length of every fish.')

    def on_failed(self, msg):
        self._stop()
        QMessageBox.warning(self, 'Landmarks', msg)

"""Screen 7, Training: paint examples of each colour class on a few fish, train, use the model, check it
against areas measured by hand. Brushes come from the species profile (one per colour class + body + reflection)."""
import os

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QRadioButton, QSlider, QSplitter, QTextEdit, QVBoxLayout,
    QWidget,
)

from .. import landmarks, profile, training, verify
from . import modelfiles


def to_qimage(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def label_image(lab, colours, alpha):
    """uint8 label map -> ARGB QImage, each label in its colour (0 = transparent)."""
    lut = np.zeros((256, 4), np.uint8)
    for k, (r, g, b) in colours.items():
        lut[k] = (b, g, r, alpha)                  # ARGB32 is BGRA in memory on little-endian machines
    argb = np.ascontiguousarray(lut[lab])
    h, w = lab.shape
    return QImage(argb.data, w, h, 4 * w, QImage.Format_ARGB32).copy()


class PaintView(QGraphicsView):
    """Photo crop with a stroke layer. Left button paints, right button drags, wheel zooms."""
    stroked = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor('#30343A'))
        self.photo = QGraphicsPixmapItem()
        self.pred = QGraphicsPixmapItem()
        self.layer = QGraphicsPixmapItem()
        for z, it in enumerate((self.photo, self.pred, self.layer)):
            it.setZValue(z)
            self.scene().addItem(it)
        self.strokes = None          # uint8 stroke map of the crop (copied back into the full map on save)
        self.colours = {}
        self.brush, self.radius, self.alpha = 1, 6, 150
        self._last = None
        self._pan = None

    def set_photo(self, rgb, strokes, colours):
        self.colours = colours
        self.strokes = strokes
        self.photo.setPixmap(QPixmap.fromImage(to_qimage(rgb)))
        self.pred.setPixmap(QPixmap())
        self.redraw()
        self.scene().setSceneRect(QRectF(0, 0, rgb.shape[1], rgb.shape[0]))
        self.zoomed = False
        self.fitInView(self.photo, Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not getattr(self, 'zoomed', True):      # whole fish in view until the user zooms
            self.fitInView(self.photo, Qt.KeepAspectRatio)

    def set_prediction(self, lab):
        self.pred.setPixmap(QPixmap() if lab is None else QPixmap.fromImage(label_image(lab, self.colours, 110)))

    def redraw(self):
        if self.strokes is not None:
            self.img = label_image(self.strokes, self.colours, self.alpha)
            self.layer.setPixmap(QPixmap.fromImage(self.img))

    def wheelEvent(self, e):
        f = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.zoomed = True
        self.scale(f, f)

    def mousePressEvent(self, e):
        if e.button() == Qt.RightButton:
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
        elif e.button() == Qt.LeftButton and self.strokes is not None:
            p = self.mapToScene(e.position().toPoint())
            self._last = p
            self.dab(p, p)

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
        elif self._last is not None:
            p = self.mapToScene(e.position().toPoint())
            self.dab(self._last, p)
            self._last = p

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
        if self._last is not None:
            self._last = None
            self.stroked.emit()

    def dab(self, a, b):
        """Paint a thick line a->b both in the stroke map and on screen."""
        import cv2
        r = self.radius
        cv2.line(self.strokes, (int(a.x()), int(a.y())), (int(b.x()), int(b.y())), int(self.brush), 2 * r + 1)
        cv2.circle(self.strokes, (int(b.x()), int(b.y())), r, int(self.brush), -1)
        x0, y0 = int(min(a.x(), b.x())) - r - 2, int(min(a.y(), b.y())) - r - 2
        x1, y1 = int(max(a.x(), b.x())) + r + 3, int(max(a.y(), b.y())) + r + 3
        h, w = self.strokes.shape
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            return
        patch = label_image(np.ascontiguousarray(self.strokes[y0:y1, x0:x1]), self.colours, self.alpha)
        qp = QPainter(self.img)
        qp.setCompositionMode(QPainter.CompositionMode_Source)
        qp.drawImage(x0, y0, patch)
        qp.end()
        self.layer.setPixmap(QPixmap.fromImage(self.img))


class TrainWorker(QObject):
    done = Signal(str, dict)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, project, start_from=None):
        super().__init__()
        self.project, self.start_from = project, start_from

    def run(self):
        try:
            name, meta = training.train(self.project, progress=self.progress.emit, start_from=self.start_from)
            self.done.emit(name, meta)
        except Exception as e:
            self.failed.emit(str(e))


class TrainingPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.project = None
        self.items = []
        self.current = None          # (photo path, full stroke map, crop box)
        self.thread = None
        lay = QVBoxLayout(self)
        title = QLabel('7  Training the colour classes')
        title.setStyleSheet('font-size: 20px; font-weight: 600; margin-bottom: 8px')
        lay.addWidget(title)
        hint = QLabel('On a few fish paint a few strokes of each class (keys 1–9 pick the brush, E the eraser, '
                      'mouse wheel zooms, right button drags). Then Train: the model colours every photo, and '
                      'strokes on its mistakes make it better.')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #4F5761')
        lay.addWidget(hint)
        self.size_only = QWidget()
        so = QHBoxLayout(self.size_only)
        so.setContentsMargins(8, 6, 8, 6)
        self.size_only.setStyleSheet('background: #FFF3CD; border-radius: 4px')
        msg = QLabel('Settings > What to measure is set to <b>size only</b>: the analysis does not measure colours, '
                     'so a colour model is not used (except on folders or photos set to colour). Train here only if '
                     'you want colours.')
        msg.setWordWrap(True)
        so.addWidget(msg, 1)
        b_on = QPushButton('Measure colours on every fish')
        b_on.clicked.connect(self.colours_on)
        so.addWidget(b_on)
        lay.addWidget(self.size_only)
        split = QSplitter()

        left = QWidget()
        ll = QVBoxLayout(left)
        form = QFormLayout()
        self.annotator = QLineEdit()
        self.annotator.setPlaceholderText('who paints the strokes')
        self.annotator.editingFinished.connect(self.save_annotator)
        form.addRow('Annotator', self.annotator)
        ll.addLayout(form)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.show_item)
        ll.addWidget(self.list, 1)
        b_unc = QPushButton('Most uncertain first')
        b_unc.setToolTip('Sort photos from the one the current model is least sure about: strokes there help most')
        b_unc.clicked.connect(lambda: self.fill_list(by_uncertainty=True))
        ll.addWidget(b_unc)
        split.addWidget(left)

        self.view = PaintView()
        self.view.stroked.connect(self.save_current)
        split.addWidget(self.view)

        right = QWidget()
        rl = QVBoxLayout(right)
        self.brush_box = QGroupBox('Brushes')
        self.brush_lay = QVBoxLayout(self.brush_box)
        self.brush_group = QButtonGroup(self)
        self.brush_group.idClicked.connect(self.set_brush)
        rl.addWidget(self.brush_box)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel('Size'))
        self.size = QSlider(Qt.Horizontal)
        self.size.setRange(1, 60)
        self.size.setValue(6)
        self.size.valueChanged.connect(lambda v: setattr(self.view, 'radius', v))
        size_row.addWidget(self.size)
        rl.addLayout(size_row)
        self.show_pred = QCheckBox('Show what the model sees')
        self.show_pred.toggled.connect(self.update_prediction)
        rl.addWidget(self.show_pred)
        for text, fn in (('Import strokes from Paint…', self.import_paint), ('Clear the strokes of this photo', self.clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            rl.addWidget(b)

        mbox = QGroupBox('Model')
        ml = QVBoxLayout(mbox)
        sf = QFormLayout()
        self.start = QComboBox()
        self.start.setToolTip('Ritocco: start from a colour model that keeps its training pixels (e.g. one pre-trained '
                              'on the van der Bijl photos, imported with Import model…); your strokes count more')
        sf.addRow('Start from', self.start)
        ml.addLayout(sf)
        self.b_train = QPushButton('Train')
        self.b_train.clicked.connect(self.train)
        ml.addWidget(self.b_train)
        self.bar = QProgressBar()
        self.bar.hide()
        ml.addWidget(self.bar)
        self.bar_text = QLabel('')
        self.bar_text.hide()
        ml.addWidget(self.bar_text)
        row = QHBoxLayout()
        for text, fn in (('Export model…', lambda: modelfiles.export_dialog(self, self.project, self.model.currentData())),
                         ('Import model…', self.import_model)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        ml.addLayout(row)
        self.model_info = QLabel('')
        self.model_info.setWordWrap(True)
        ml.addWidget(self.model_info)
        mf = QFormLayout()
        self.model = QComboBox()
        self.model.activated.connect(self.use_model)
        mf.addRow('Use in the analysis', self.model)
        self.merge = QDoubleSpinBox()
        self.merge.setRange(0, 5)
        self.merge.setSingleStep(0.1)
        self.merge.setSuffix(' mm')
        self.merge.setToolTip('Spots closer than this count as one (as when measured together by hand)')
        self.merge.valueChanged.connect(self.save_spot_settings)
        mf.addRow('Merge spots closer than', self.merge)
        self.min_spot = QDoubleSpinBox()
        self.min_spot.setRange(0, 5)
        self.min_spot.setDecimals(3)
        self.min_spot.setSingleStep(0.01)
        self.min_spot.setSuffix(' mm²')
        self.min_spot.valueChanged.connect(self.save_spot_settings)
        mf.addRow('Ignore spots smaller than', self.min_spot)
        ml.addLayout(mf)
        b_ver = QPushButton('Compare with measures made by hand…')
        b_ver.clicked.connect(self.verify)
        ml.addWidget(b_ver)
        rl.addWidget(mbox)
        rl.addStretch()
        split.addWidget(right)
        split.setSizes([240, 760, 280])
        lay.addWidget(split, 1)
        self.brushes = []
        self.keys = []

    # ---------------------------------------------------------------- project
    def colours_on(self):
        self.project.settings['analysis']['default'] = 'size_colour'
        self.project.save()
        self.size_only.hide()

    def show_project(self, p):
        self.project = p
        self.size_only.setVisible(p.settings['analysis'].get('default') == 'size')
        prof = profile.load(p.settings['profile'])
        self.brushes = training.active_brushes(p, prof)
        self.view.colours = {b['index']: training.vis_colour(b) for b in self.brushes}
        self.build_brushes()
        self.annotator.setText(p.settings.get('training', {}).get('annotator', ''))
        cs = p.settings['colours']
        for w, v in ((self.merge, cs.get('merge_mm', 0)), (self.min_spot, cs.get('min_spot_mm2', 0))):
            w.blockSignals(True)
            w.setValue(float(v or 0))
            w.blockSignals(False)
        self.fill_models()
        self.fill_list()

    def build_brushes(self):
        for b in self.brush_group.buttons():
            self.brush_group.removeButton(b)
            b.deleteLater()
        for k in self.keys:
            k.deleteLater()
        self.keys = []
        entries = self.brushes + [{'index': 0, 'label': 'Eraser', 'display': '#FFFFFF'}]
        for n, b in enumerate(entries):
            key = 'E' if b['index'] == 0 else str(n + 1)
            rb = QRadioButton(f"{key}  {b['label']}")
            r, g, bl = training.vis_colour(b) if b['index'] else (255, 255, 255)
            rb.setStyleSheet(f'QRadioButton::indicator {{ background: rgb({r},{g},{bl}); border: 1px solid #555; '
                             f'width: 14px; height: 14px; border-radius: 7px }}'
                             f'QRadioButton::indicator:checked {{ border: 3px solid #111 }}')
            self.brush_group.addButton(rb, b['index'])
            self.brush_lay.addWidget(rb)
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)       # only on this screen
            sc.activated.connect(rb.click)
            self.keys.append(sc)
        self.brush_group.button(1).setChecked(True)
        self.set_brush(1)

    def set_brush(self, i):
        self.view.brush = i

    def save_annotator(self):
        if self.project:
            self.project.settings.setdefault('training', {})['annotator'] = self.annotator.text().strip()
            self.project.save()

    def fill_list(self, by_uncertainty=False):
        p = self.project
        done = dict(training.annotated(p))
        res = {r['path']: r for r in p.results()}
        items = [photo for _, photo in p.photos() if training.fish_mask(p, photo) is not None
                 and res.get(photo, {}).get('status') != 'excluded']
        if by_uncertainty:
            items.sort(key=lambda ph: res.get(ph, {}).get('colour_confidence', 1.0))
        else:
            items.sort(key=lambda ph: (ph not in done, os.path.basename(ph)))
        self.items = items
        self.list.blockSignals(True)
        self.list.clear()
        for ph in items:
            n = sum(done.get(ph, {}).values())
            conf = res.get(ph, {}).get('colour_confidence')
            text = os.path.basename(ph) + (f'\n✓ {n} pixels painted' if n else '') + \
                (f'\nmodel confidence {conf:.2f}' if conf is not None else '')
            self.list.addItem(QListWidgetItem(text))
        self.list.blockSignals(False)
        if items:
            self.list.setCurrentRow(0)

    def show_item(self, i):
        if not 0 <= i < len(self.items):
            return
        p, photo = self.project, self.items[i]
        img = training.corrected_image(p, photo)
        mask = training.fish_mask(p, photo)
        x0, y0, x1, y1 = landmarks.crop_box(mask, img.shape)
        full = training.load_strokes(p, photo, img.shape)
        self.current = (photo, full, (x0, y0, x1, y1), img, mask)
        self.view.set_photo(img[y0:y1, x0:x1], full[y0:y1, x0:x1].copy(), self.view.colours)
        self.update_prediction()

    def save_current(self):
        if self.current:
            photo, full, (x0, y0, x1, y1) = self.current[:3]
            full[y0:y1, x0:x1] = self.view.strokes
            training.save_strokes(self.project, photo, full)

    def update_prediction(self):
        if not (self.show_pred.isChecked() and self.current and self.project.settings['colours'].get('model')):
            self.view.set_prediction(None)
            return
        from ..engine import pixels
        photo, full, (x0, y0, x1, y1), img, mask = self.current
        forest = training.load_model(self.project, self.project.settings['colours']['model'])
        lab, (bx0, by0, bx1, by1), s, _ = pixels.classify(img, mask, forest)
        import cv2
        lab = cv2.resize(lab, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_NEAREST) if s != 1 else lab
        big = np.zeros(img.shape[:2], np.uint8)
        big[by0:by1, bx0:bx1] = lab
        shown = [b['index'] for b in self.brushes if b['key'] != 'base']     # off classes count as body without colour
        big[~np.isin(big, shown)] = 0
        self.view.set_prediction(big[y0:y1, x0:x1])

    def clear(self):
        if self.current:
            self.current[1][:] = 0
            self.view.strokes[:] = 0
            self.view.redraw()
            self.save_current()

    def import_paint(self):
        if not self.current:
            return
        photo = self.current[0]
        src = training.paint_counterpart(photo)
        if not src:
            src, _ = QFileDialog.getOpenFileName(self, 'Copy of the photo with the strokes painted in Paint', os.path.dirname(photo),
                                                 'Images (*.png *.bmp *.jpg)')
        if not src:
            return
        n = training.import_paint(self.project, photo, src, profile.load(self.project.settings['profile']))
        QMessageBox.information(self, 'Paint', f'Imported {n} painted pixels.\nPaint colours: ' + ', '.join(
            f"{b['label']} {b['paint']}" for b in self.brushes))
        self.show_item(self.list.currentRow())

    # ---------------------------------------------------------------- model
    def fill_models(self):
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem('none (colour classes off)', '')
        for name, meta in reversed(training.list_models(self.project)):
            self.model.addItem(f"{name}  ({meta.get('date', '')[:10]})", name)
        cur = self.project.settings['colours'].get('model', '')
        self.model.setCurrentIndex(max(0, self.model.findData(cur)))
        self.model.blockSignals(False)
        self.describe(cur)
        self.start.clear()
        self.start.addItem('nothing: only my strokes', '')
        pool = training.pool_models(self.project)
        for name, meta in reversed(pool):
            self.start.addItem(f"{name}  ({'imported' if meta.get('imported_from') else meta.get('date', '')[:10]})", name)
        imported = [n for n, m in pool if m.get('imported_from')]
        self.start.setCurrentIndex(max(0, self.start.findData(imported[-1])) if imported else 0)

    def describe(self, name):
        meta = dict(training.list_models(self.project)).get(name) if name else None
        if not meta:
            self.model_info.setText('No model in use.')
            return
        acc = meta.get('oob_accuracy')
        px = ', '.join(f'{k} {v}' for k, v in meta.get('pixels_per_class', {}).items() if v)
        self.model_info.setText(f"{name}: {len(meta.get('photos', []))} photos, annotator {meta.get('annotator') or '?'}\n"
                                f"pixels: {px}\n" + (f'accuracy on held-out pixels: {acc:.0%}' if acc else ''))

    def train(self):
        self.save_current()
        self.b_train.setEnabled(False)
        self.b_train.setText('Training…')
        self.worker = TrainWorker(self.project, self.start.currentData() or None)
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
        name = modelfiles.import_dialog(self, self.project, 'colours')
        if name:
            self.project.settings['colours']['model'] = name
            self.project.save()
            self.fill_models()
            self.update_prediction()

    def _stop_thread(self):
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.bar.hide()
        self.bar_text.hide()
        self.b_train.setEnabled(True)
        self.b_train.setText('Train')

    def on_trained(self, name, meta):
        self._stop_thread()
        self.project.settings['colours']['model'] = name
        self.project.save()
        self.fill_models()
        self.show_pred.setChecked(True)
        self.update_prediction()
        QMessageBox.information(self, 'Training', f'Model {name} is ready and in use. Look at its prediction on the photos, '
                                'add a few strokes where it is wrong and train again. Then run the Analysis again: '
                                'the CSV will have areas, percentages and spots per class.')

    def on_failed(self, msg):
        self._stop_thread()
        QMessageBox.warning(self, 'Training', msg)

    def use_model(self, i):
        name = self.model.itemData(i)
        self.project.settings['colours']['model'] = name
        self.project.save()
        self.describe(name)
        self.update_prediction()

    def save_spot_settings(self, *_):
        cs = self.project.settings['colours']
        cs['merge_mm'], cs['min_spot_mm2'] = round(self.merge.value(), 2), round(self.min_spot.value(), 3)
        self.project.save()

    def verify(self):
        p = self.project
        path, _ = QFileDialog.getOpenFileName(self, 'Measures made by hand (one row per fish, areas in mm²)', '',
                                              'Tables (*.xlsx *.csv *.txt)')
        if not path:
            return
        from .. import export
        header, rows = verify.read_table(path)
        off = training.classes_off(p)
        keys = [c['key'] for c in profile.load(p.settings['profile'])['colour_classes'] if c['key'] not in off]
        app_rows, _ = export.rows(p)
        id_col, cols = verify.guess_columns(header, rows, app_rows, keys)
        dlg = QDialog(self)
        dlg.setWindowTitle('Which columns?')
        form = QFormLayout(dlg)
        boxes = {}
        for k in ['__id__'] + keys + list(verify.SIZES):
            cb = QComboBox()
            cb.addItem('—', '')
            for h in header:
                cb.addItem(h, h)
            cb.setCurrentIndex(max(0, cb.findData(id_col if k == '__id__' else cols.get(k, ''))))
            text = {'__id__': 'Fish id', 'mask_length': 'Total length (mm), vs outline length',
                    'fish_area': 'Fish area (mm²)'}.get(k, f'Area {k} (mm²)')
            form.addRow(text, cb)
            boxes[k] = cb
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)
        if dlg.exec() != QDialog.Accepted or not boxes['__id__'].currentData():
            return
        chosen = {k: boxes[k].currentData() for k in keys + list(verify.SIZES) if boxes[k].currentData()}
        stats, csv_path, png = verify.compare(p, path, boxes['__id__'].currentData(), chosen)
        lines = []
        for k, s in stats.items():
            if s.get('n', 0) < 2:
                lines.append(f'{k}: {s.get("n", 0)} fish matched, too few')
                continue
            lo, hi = s['limits_of_agreement']
            u = 'mm' if k == 'mask_length' else 'mm²'
            lines.append(f"{k}: n={s['n']}, r={s['r']:.2f}, mean difference {s['mean_difference']:+.3f} {u}, "
                         f"mean absolute error {s['mean_abs_error']:.3f} {u} ({s['relative_error_pct']:.0f}%), "
                         f"limits of agreement {lo:+.3f} / {hi:+.3f} {u}")
        show = QDialog(self)
        show.setWindowTitle('Comparison with the measures made by hand')
        v = QVBoxLayout(show)
        t = QTextEdit()
        t.setReadOnly(True)
        t.setPlainText('\n'.join(lines) + f'\n\nPairs: {csv_path}')
        t.setMaximumHeight(130)
        v.addWidget(t)
        if png and os.path.exists(png):
            pic = QLabel()
            pic.setPixmap(QPixmap(png).scaledToWidth(min(1000, QPixmap(png).width()), Qt.SmoothTransformation))
            v.addWidget(pic)
        show.exec()

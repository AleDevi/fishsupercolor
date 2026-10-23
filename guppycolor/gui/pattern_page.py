"""Screen 6, Pattern and clusters: align every fish on the mean shape, compare the colour patterns with the
chosen distance, group them with a similarity slider. Clusters are recomputed and drawn while the slider moves."""
import os

import numpy as np
from PySide6.QtCore import QObject, QPointF, QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListView, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSlider, QSplitter, QVBoxLayout, QWidget,
)

from .. import pattern

CLUSTER_COLOURS = ['#1F77B4', '#D62728', '#2CA02C', '#9467BD', '#FF7F0E', '#17BECF', '#8C564B', '#E377C2',
                   '#7F7F7F', '#BCBD22']


def cluster_colour(k):
    return QColor(CLUSTER_COLOURS[(k - 1) % len(CLUSTER_COLOURS)])


def to_pixmap(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class Dendrogram(QWidget):
    """The tree of the clustering, leaves coloured by cluster; the dashed line is the slider's cut.
    Click near the line to move it."""
    cut_moved = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self.tree, self.names, self.labels, self.similarity = None, [], None, 0.5

    def set_tree(self, Z, names):
        from scipy.cluster import hierarchy
        self.tree = hierarchy.dendrogram(Z, no_plot=True) if Z is not None else None
        self.names = names
        self.update()

    def set_cut(self, similarity, labels):
        self.similarity, self.labels = similarity, labels
        self.update()

    def _geom(self):
        left, right, top, bottom = 58, 40, 10, 70 if len(self.names) <= 60 else 16
        return left, top, self.width() - left - right, self.height() - top - bottom

    def mousePressEvent(self, e):
        if self.tree is None:
            return
        left, top, w, h = self._geom()
        d = 1 - (e.position().y() - top) / h
        self.cut_moved.emit(float(np.clip(1 - d, 0, 1)))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor('white'))
        if self.tree is None:
            p.setPen(QColor('#4F5761'))
            p.drawText(self.rect(), Qt.AlignCenter, 'Press "Align and compare" to see the tree of the fish.')
            return
        left, top, w, h = self._geom()
        n = len(self.tree['leaves'])
        X = lambda x: left + (x - 5) / max(10 * (n - 1), 1) * w if n > 1 else left + w / 2
        Y = lambda d: top + (1 - min(d, 1)) * h
        p.setPen(QPen(QColor('#9AA0A6'), 1))
        for xs, ds in zip(self.tree['icoord'], self.tree['dcoord']):
            pts = [QPointF(X(x), Y(d)) for x, d in zip(xs, ds)]
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)
        # axis: similarity 1 at the bottom, 0 at the top
        p.setPen(QColor('#4F5761'))
        f = QFont(); f.setPointSize(8); p.setFont(f)
        for s in (0, 0.25, 0.5, 0.75, 1):
            y = Y(1 - s)
            p.drawLine(left - 4, int(y), left, int(y))
            p.drawText(14, int(y) + 4, f'{s:.2f}')
        p.save(); p.translate(10, top + h / 2); p.rotate(-90)
        p.drawText(-30, 0, 'similarity'); p.restore()
        # leaves
        for k, leaf in enumerate(self.tree['leaves']):
            x = X(5 + 10 * k)
            c = cluster_colour(int(self.labels[leaf])) if self.labels is not None else QColor('#333')
            p.setPen(Qt.NoPen); p.setBrush(c)
            p.drawEllipse(QPointF(x, Y(0)), 4, 4)
            if n <= 60:
                p.save(); p.translate(x + 3, Y(0) + 8); p.rotate(60); p.setPen(c)
                p.drawText(0, 0, str(self.names[leaf])[:14]); p.restore()
        y = Y(1 - self.similarity)
        p.setPen(QPen(QColor('#C62828'), 1.5, Qt.DashLine))
        p.drawLine(left, int(y), left + w, int(y))


class Worker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, project, st, ps=None):
        super().__init__()
        self.project, self.st, self.ps = project, st, ps

    def run(self):
        try:
            ps = self.ps
            if ps is None:
                ps = pattern.build(self.project, self.st, progress=lambda i, n, name: self.progress.emit(i, n, name))
            else:
                ps.st = self.st
            self.progress.emit(0, 0, 'comparing the fish')
            self.finished.emit(ps, pattern.Clustering(ps))
        except Exception as e:
            self.failed.emit(str(e) if isinstance(e, ValueError) else f'{type(e).__name__}: {e}')


class PatternPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.project, self.ps, self.cl, self.labels = None, None, None, None
        self.thread = None
        self.loading = False
        from .main import h1, note
        lay = QVBoxLayout(self)
        lay.addWidget(h1('Pattern and clusters'))
        lay.addWidget(note('Every fish is put on the mean body shape (Procrustes + thin-plate spline), so the same '
                           'spot falls on the same place in every fish. Choose how to compare them, press Align and '
                           'compare, then move the similarity slider: the groups change as you move it.'))
        split = QSplitter(Qt.Horizontal)
        lay.addWidget(split, 1)
        # controls
        side = QWidget(); sl = QVBoxLayout(side); sl.setContentsMargins(0, 0, 0, 0)
        g = QGroupBox('Comparison'); f = QFormLayout(g)
        self.distance = QComboBox()
        for k, label in pattern.DISTANCES:
            self.distance.addItem(label, k)
        self.distance.setToolTip('Colour maps: share of colour overlapping pixel by pixel.\n'
                                 'Spots: round spots as points, elongated ones as lines, matched one to one.\n'
                                 'Colour amounts: how much of each colour, wherever it is.')
        f.addRow('Distance', self.distance)
        self.classes_box = QWidget(); self.classes_lay = QVBoxLayout(self.classes_box)
        self.classes_lay.setContentsMargins(0, 0, 0, 0)
        f.addRow('Colours', self.classes_box)
        self.tolerance = QDoubleSpinBox(); self.tolerance.setRange(0, 10); self.tolerance.setSingleStep(0.5)
        self.tolerance.setSuffix(' px'); self.tolerance.setToolTip('Blur of the aligned maps: forgives small misalignments')
        f.addRow('Tolerance', self.tolerance)
        self.linkage = QComboBox()
        for k, label in (('average', 'Average (UPGMA)'), ('complete', 'Complete'), ('single', 'Single')):
            self.linkage.addItem(label, k)
        f.addRow('Linkage', self.linkage)
        self.region = QComboBox()
        self.region.addItem('Body (snout to caudal peduncle)', 'body')
        self.region.addItem('Whole fish (caudal fin too, if in the outline)', 'whole')
        f.addRow('Region', self.region)
        self.head = QComboBox()
        for k, label in (('left', 'Left'), ('right', 'Right'), ('auto', 'Guess for each fish')):
            self.head.addItem(label, k)
        self.head.setToolTip('Side of the photo the head is on. A snout landmark, when there is one, wins.')
        f.addRow('Head in the photos', self.head)
        self.ignore_head = QDoubleSpinBox(); self.ignore_head.setRange(0, 0.5); self.ignore_head.setSingleStep(0.01)
        self.ignore_head.setDecimals(2); self.ignore_head.setSuffix(' body lengths')
        self.ignore_head.setToolTip('The front of the head is left out of the comparison, so the dark eye\n'
                                    'does not count as a black spot. 0 = keep everything.')
        f.addRow('Leave out the head', self.ignore_head)
        self.line_ratio = QDoubleSpinBox(); self.line_ratio.setRange(1.2, 10); self.line_ratio.setSingleStep(0.5)
        self.line_ratio.setToolTip('A spot this many times longer than wide counts as a line')
        f.addRow('Line if length/width ≥', self.line_ratio)
        sl.addWidget(g)
        self.b_run = QPushButton('Align and compare'); self.b_run.setStyleSheet('font-weight: 600; padding: 6px')
        self.b_run.clicked.connect(lambda: self.compute(realign=True))
        sl.addWidget(self.b_run)
        self.bar = QProgressBar(); self.bar.setVisible(False); sl.addWidget(self.bar)
        g2 = QGroupBox('Clusters'); v2 = QVBoxLayout(g2)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.slid)
        self.sim_label = QLabel()
        v2.addWidget(QLabel('Similarity to join the same cluster'))
        v2.addWidget(self.slider); v2.addWidget(self.sim_label)
        self.stats = QLabel(); self.stats.setWordWrap(True); self.stats.setStyleSheet('color: #4F5761')
        v2.addWidget(self.stats)
        sl.addWidget(g2)
        self.b_export = QPushButton('Export CSV for R'); self.b_export.clicked.connect(self.export)
        sl.addWidget(self.b_export)
        sl.addStretch(1)
        split.addWidget(side)
        # tree + clusters
        right = QSplitter(Qt.Vertical)
        self.tree = Dendrogram()
        self.tree.cut_moved.connect(lambda s: self.slider.setValue(int(round(s * 100))))
        right.addWidget(self.tree)
        bottom = QSplitter(Qt.Horizontal)
        self.clusters = QListWidget(); self.clusters.setIconSize(QSize(190, 80))
        self.clusters.currentRowChanged.connect(self.show_cluster)
        bottom.addWidget(self.clusters)
        self.members = QListWidget(); self.members.setViewMode(QListView.IconMode)
        self.members.setIconSize(QSize(200, 84)); self.members.setResizeMode(QListView.Adjust)
        self.members.setSpacing(4)
        bottom.addWidget(self.members)
        bottom.setSizes([330, 600])
        right.addWidget(bottom)
        right.setSizes([300, 400])
        split.addWidget(right)
        split.setSizes([330, 900])
        for w in (self.distance, self.linkage):
            w.currentIndexChanged.connect(lambda *_: self.compute(realign=False))
        self.tolerance.valueChanged.connect(lambda *_: self.compute(realign=False))
        self.line_ratio.valueChanged.connect(lambda *_: self.compute(realign=True, auto=True))
        self.region.currentIndexChanged.connect(lambda *_: self.compute(realign=True, auto=True))
        self.head.currentIndexChanged.connect(lambda *_: self.compute(realign=True, auto=True))
        self.ignore_head.valueChanged.connect(lambda *_: self.compute(realign=True, auto=True))
        self.enable(False)

    # --- settings ---
    def show_project(self, p):
        self.project = p
        self.loading = True
        st = pattern.settings(p)
        for i in reversed(range(self.classes_lay.count())):
            self.classes_lay.itemAt(i).widget().deleteLater()
        self.class_checks = []
        try:
            classes, _ = pattern.model_classes(p)
        except Exception:
            classes = []
        for c in classes:
            if c['key'] in pattern.SKIP_CLASSES:
                continue
            cb = QCheckBox(c.get('label', c['key']))
            cb.setChecked(not st['classes'] or c['key'] in st['classes'])
            cb.setStyleSheet(f"QCheckBox {{ color: {c.get('display', '#333')}; }}")
            cb.key = c['key']
            cb.toggled.connect(lambda *_: self.compute(realign=False))
            self.classes_lay.addWidget(cb)
            self.class_checks.append(cb)
        if not classes:
            self.classes_lay.addWidget(QLabel('no colour model yet (screen 7)'))
        self.distance.setCurrentIndex(max(self.distance.findData(st['distance']), 0))
        self.linkage.setCurrentIndex(max(self.linkage.findData(st['linkage']), 0))
        self.region.setCurrentIndex(max(self.region.findData(st['region']), 0))
        self.tolerance.setValue(st['tolerance_px'])
        self.head.setCurrentIndex(max(self.head.findData(st['head']), 0))
        self.ignore_head.setValue(st['ignore_head'])
        self.line_ratio.setValue(st['line_ratio'])
        self.slider.setValue(int(round(st['similarity'] * 100)))
        self.loading = False
        self.sim_label.setText(f'{self.slider.value() / 100:.2f}')
        n = len(pattern.usable_results(p))
        self.stats.setText(f'{n} analysed fish with colour classes.')
        self.b_run.setEnabled(bool(classes) and n >= 2)

    def current_settings(self):
        st = pattern.settings(self.project)
        keys = [cb.key for cb in self.class_checks if cb.isChecked()]
        st.update(distance=self.distance.currentData(), linkage=self.linkage.currentData(),
                  region=self.region.currentData(), tolerance_px=self.tolerance.value(),
                  line_ratio=self.line_ratio.value(), head=self.head.currentData(),
                  ignore_head=self.ignore_head.value(), similarity=self.slider.value() / 100,
                  classes=[] if len(keys) == len(self.class_checks) else keys)
        self.tolerance.setEnabled(st['distance'] == 'jaccard')
        return st

    def enable(self, on):
        for w in (self.slider, self.b_export):
            w.setEnabled(on)

    # --- work ---
    def compute(self, realign, auto=False):
        if self.loading or self.project is None or self.thread is not None:
            return
        st = self.current_settings()
        if not st['classes'] and self.class_checks and not any(cb.isChecked() for cb in self.class_checks):
            return
        pattern.save_settings(self.project, st)
        if self.ps is None and (auto or not realign):
            return                              # nothing aligned yet: wait for the button
        self.b_run.setEnabled(False)
        self.bar.setVisible(True); self.bar.setRange(0, 0)
        self.thread = QThread()
        self.worker = Worker(self.project, st, None if realign else self.ps)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.thread.start()

    def _stop_thread(self):
        self.thread.quit(); self.thread.wait()
        self.thread = None
        self.bar.setVisible(False)
        self.b_run.setEnabled(True)

    def on_progress(self, i, n, name):
        if n:
            self.bar.setRange(0, n); self.bar.setValue(i + 1)
            self.bar.setFormat(f'aligning {i + 1}/{n}: {name}')
        else:
            self.bar.setRange(0, 0)

    def on_failed(self, msg):
        self._stop_thread()
        QMessageBox.warning(self, 'Pattern and clusters', msg)

    def on_finished(self, ps, cl):
        self._stop_thread()
        self.ps, self.cl = ps, cl
        self._icons = {}
        self.tree.set_tree(cl.Z, ps.names)
        self.enable(True)
        self.slid(self.slider.value())

    def slid(self, v):
        s = v / 100
        self.sim_label.setText(f'{s:.2f}  (distance below {1 - s:.2f})')
        if self.cl is None:
            return
        self.labels = self.cl.at(s)
        self.tree.set_cut(s, self.labels)
        k = int(self.labels.max())
        ari = self.cl.agreement(self.labels)
        txt = f'{len(self.labels)} fish in {k} cluster{"s" if k != 1 else ""}. Tree fit (cophenetic r): {self.cl.coph:.2f}.'
        if ari is not None:
            txt += f'\nAgreement with the groups on the labels (adjusted Rand): {ari:.2f} (1 = same groups, 0 = chance).'
        self.stats.setText(txt)
        row = max(self.clusters.currentRow(), 0)
        self.clusters.blockSignals(True)
        self.clusters.clear()
        for c in range(1, k + 1):
            members = np.nonzero(self.labels == c)[0]
            it = QListWidgetItem(QIcon(to_pixmap(pattern.frequency_image(self.ps, members, scale=1))),
                                 f'Cluster {c}\n{len(members)} fish')
            it.setForeground(cluster_colour(c))
            self.clusters.addItem(it)
        self.clusters.blockSignals(False)
        self.clusters.setCurrentRow(min(row, k - 1))
        self.show_cluster(self.clusters.currentRow())
        if not self.loading and self.project is not None:
            st = pattern.settings(self.project); st['similarity'] = s
            self.project.settings['pattern'] = {kk: st[kk] for kk in pattern.DEFAULTS}

    def fish_icon(self, i):
        if i not in self._icons:
            self._icons[i] = QIcon(to_pixmap(pattern.fish_image(self.ps, i, scale=1)))
        return self._icons[i]

    def show_cluster(self, row):
        self.members.clear()
        if self.labels is None or row < 0:
            return
        for i in np.nonzero(self.labels == row + 1)[0]:
            r = self.ps.rows[i]
            it = QListWidgetItem(self.fish_icon(i), f"{self.ps.names[i]}  {r.get('label_group') or ''}".strip())
            it.setToolTip(r.get('path', ''))
            self.members.addItem(it)

    def export(self):
        if self.cl is None:
            return
        self.project.save()
        d = pattern.export(self.project, self.ps, self.cl, self.slider.value() / 100)
        if QMessageBox.question(self, 'Export', f'CSV files written in\n{d}\n\nOpen the folder?') == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))

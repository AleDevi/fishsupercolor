"""Calibrate manually: the whole photo with zoom (wheel) and pan (right button); click the two ends of a known
length (e.g. 2 cm on the graph paper), type it in mm. Returns the two points in photo pixels."""
from PySide6.QtCore import QLineF, Qt
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QGraphicsEllipseItem, QHBoxLayout, QLabel, QVBoxLayout,
)

from .. import pipeline
from .landmarks_page import ClickView


class CalibrateDialog(QDialog):
    def __init__(self, parent, photo, folder=None):
        super().__init__(parent)
        self.setWindowTitle('Calibrate manually')
        self.resize(1100, 800)
        self.points = []
        self.items = []
        lay = QVBoxLayout(self)
        hint = QLabel('Click the two ends of a known length on the paper or ruler (e.g. 2 cm = 20 squares; the longer, '
                      'the more precise). Mouse wheel zooms, right button drags. A third click starts again.')
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.view = ClickView()
        img = pipeline.load_image(photo)
        self.view.set_photo(img)
        self.radius = max(3.0, img.shape[1] / 400)
        self.view.clicked.connect(self.click)
        lay.addWidget(self.view, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel('Length of the red line:'))
        self.mm = QDoubleSpinBox()
        self.mm.setRange(0.01, 10000)
        self.mm.setDecimals(2)
        self.mm.setValue(20)
        self.mm.setSuffix(' mm')
        row.addWidget(self.mm)
        self.px = QLabel('')
        row.addWidget(self.px)
        row.addStretch()
        lay.addLayout(row)
        self.whole = QCheckBox(f'Use it also for the other photos of the folder “{folder}” whose scale is missing or '
                               'doubtful (same camera distance)')
        self.whole.setVisible(folder is not None)
        lay.addWidget(self.whole)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        lay.addWidget(self.buttons)

    def click(self, x, y):
        if len(self.points) == 2:
            self.points = []
        self.points.append((x, y))
        for it in self.items:
            self.view.scene().removeItem(it)
        self.items = []
        pen = QPen(QColor('#FF1744'))
        pen.setWidthF(self.radius / 2)
        for px, py in self.points:
            e = QGraphicsEllipseItem(px - self.radius, py - self.radius, 2 * self.radius, 2 * self.radius)
            e.setPen(pen)
            self.view.scene().addItem(e)
            self.items.append(e)
        if len(self.points) == 2:
            self.items.append(self.view.scene().addLine(QLineF(*self.points[0], *self.points[1]), pen))
            (ax, ay), (bx, by) = self.points
            self.px.setText(f'({((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5:.0f} px)')
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(len(self.points) == 2)

    def result_line(self):
        """(p1, p2, mm, whole folder)"""
        return self.points[0], self.points[1], self.mm.value(), self.whole.isChecked()

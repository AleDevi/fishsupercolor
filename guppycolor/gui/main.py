"""Main window: one screen per step (Project, Settings, Analysis, Review, Results, Pattern and clusters, Training,
Landmarks)."""
import html
import os
import re
import sys

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QImageReader, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QFrame, QProgressBar, QPushButton, QRadioButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget,
    QTabBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import __version__, export, metadata, pipeline, profile, scaleline
from ..project import PROJECT_FILE, Project

STATUS_COLOUR = {'ok': '#2E7D32', 'to_check': '#9A5C00', 'error': '#C62828',
                 'reviewed': '#1F6FB2', 'excluded': '#5B636E'}
STATUS_LABEL = {'ok': 'ok', 'to_check': 'to check', 'error': 'error',
                'reviewed': 'reviewed', 'excluded': 'excluded'}
FLAG_LABEL = {
    'no_chart': 'no colour chart found: photo not corrected',
    'reference_missing': 'the colour reference photo is missing',
    'scale_doubtful': 'doubtful scale on the graph paper',
    'scale_not_found': 'scale not found',
    'scale_from_median': 'scale taken from the session median',
    'scale_to_set': 'no scale yet: Calibrate manually (draw 2 cm on the photo)',
    'fish_at_border': 'the fish touches the photo border',
    'mask_in_pieces': 'the outline is broken into pieces',
    'unusual_shape': 'unusual outline shape (length/depth ratio)',
    'fish_not_found': 'fish not found',
    'area_outlier': 'area very different from the other fish of the session',
    'label_uncertain': 'label read with low confidence: check the id',
    'colours_uncertain': 'the colour model is unsure on this fish: a good photo to paint in Training',
    'scale_off_session': 'the scale differs from the other photos of this folder (Settings > Scale > Check): '
                         'confirm it or calibrate by hand',
    'size_off_session': 'the fish length differs from the other fish of this folder (Settings > Scale > Check): '
                        'check the scale and the outline',
    'colour_start_from_snout': 'no eye landmark: the colours start at a fixed distance from the snout (click the eye in Landmarks)',
    'colour_start_from_length': 'no scale in mm: the colours start at a share of the fish length from the snout',
    'colour_start_missing': 'no snout or no fish length: the head is not left out of the colours',
    'tail_incomplete': 'caudal fin not all in the outline: add it (Fish view), or mark the tail tip on screen 8 for TL',
    'date_mismatch': 'the date on the label differs from the date the photo was taken',
    'overexposed': 'more than 1% of the fish is burnt out (white)',
    'blurry': 'much less sharp than the other photos of the session',
    'sl_landmarks_missing': 'snout or end of the caudal peduncle not marked: no standard length',
    'landmark_model_error': 'the landmark model could not be used (file changed or missing)',
    'outline_grown_to_landmarks': 'a landmark was outside the outline (e.g. the tail tip): the outline was grown to include it',
    'outline_model_missed': 'the outline model found no fish: the automatic detector drew the outline',
    'outline_model_rejected': 'the outline of the outline model looked wrong (border, pieces or shape): the '
                              'automatic detector drew a better one',
    'outline_model_error': 'the outline model could not be used (file changed or missing): automatic detector used',
}


def h1(text):
    lab = QLabel(text)
    lab.setStyleSheet('font-size: 20px; font-weight: 600; margin-bottom: 8px')
    return lab


def note(text):
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet('color: #4F5761')
    return lab


def flags_text(avvisi):
    return '\n'.join('• ' + FLAG_LABEL.get(f, f) for f in (avvisi or '').split(';') if f) or 'no warnings'


# ---------------------------------------------------------------- screens
class ProjectPage(QWidget):
    opened = Signal(object)

    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.addWidget(h1('1  Project'))
        lay.addWidget(note('A project is one experiment: it holds the settings, the results and the exported files. '
                           'You can add photo folders (sessions) at any time.'))
        row = QHBoxLayout()
        b_new, b_open = QPushButton('New project…'), QPushButton('Open project…')
        b_new.clicked.connect(self.new)
        b_open.clicked.connect(self.open_dialog)
        row.addWidget(b_new)
        row.addWidget(b_open)
        row.addStretch()
        lay.addLayout(row)
        self.recent = QListWidget()
        self.recent.itemDoubleClicked.connect(lambda it: self.open(it.data(Qt.UserRole)))
        lay.addWidget(QLabel('Recent projects (double-click to open)'))
        lay.addWidget(self.recent, 1)
        self.box = QGroupBox('Photo folders of the project')
        bl = QVBoxLayout(self.box)
        self.folders = QListWidget()
        bl.addWidget(self.folders)
        row = QHBoxLayout()
        b_add = QPushButton('Add folder…')
        b_add.clicked.connect(self.add_folder)
        b_size = QPushButton('Size only / size + colour for the selected folder')
        b_size.clicked.connect(self.toggle_size_only)
        row.addWidget(b_add)
        row.addWidget(b_size)
        row.addStretch()
        bl.addLayout(row)
        self.count = note('')
        bl.addWidget(self.count)
        self.box.setEnabled(False)
        lay.addWidget(self.box, 2)
        self.load_recent()

    def load_recent(self):
        self.recent.clear()
        for p in QSettings('GuppyColour', 'app').value('recent', [], list) or []:
            if os.path.exists(os.path.join(p, PROJECT_FILE)):
                it = QListWidgetItem(p)
                it.setData(Qt.UserRole, p)
                self.recent.addItem(it)

    def remember(self, path):
        s = QSettings('GuppyColour', 'app')
        rec = [p for p in (s.value('recent', [], list) or []) if p != path]
        s.setValue('recent', [path] + rec[:9])
        self.load_recent()

    def new(self):
        d = QFileDialog.getExistingDirectory(self, 'Folder for the new project (empty)')
        if not d:
            return
        if os.path.exists(os.path.join(d, PROJECT_FILE)):
            if QMessageBox.question(self, 'New project', 'This folder already holds a project. Open it?') \
                    == QMessageBox.Yes:
                self.open(d)
            return
        name, ok = QInputDialog.getText(self, 'New project', 'Experiment name:',
                                        text=os.path.basename(d))
        if not ok or not name:
            return
        self.open(Project.create(d, name).path)
        self.add_folder()

    def open_dialog(self):
        d = QFileDialog.getExistingDirectory(self, 'Project folder')
        if d:
            self.open(d)

    def open(self, path):
        if not os.path.exists(os.path.join(path, PROJECT_FILE)):
            QMessageBox.warning(self, 'Project', 'There is no project in this folder.')
            return
        p = Project(path)
        self.remember(p.path)
        self.opened.emit(p)

    def show_project(self, p):
        self.box.setEnabled(True)
        self.box.setTitle(f'Photo folders of project “{p.name}”')
        self.folders.clear()
        size_only = {os.path.abspath(f) for f in p.settings['analysis'].get('size_only_folders', [])}
        for f in p.folders:
            text = f if os.path.isdir(f) else f + '   (not found)'
            self.folders.addItem(text + ('   [size only]' if os.path.abspath(f) in size_only else ''))
        n, todo = len(p.photos()), len(p.pending())
        self.count.setText(f'{n} photos in total, {todo} to analyse.')

    def toggle_size_only(self):
        p, i = self.win.project, self.folders.currentRow()
        if p is None or not 0 <= i < len(p.folders):
            return
        f = os.path.abspath(p.folders[i])
        lst = [os.path.abspath(x) for x in p.settings['analysis'].get('size_only_folders', [])]
        p.settings['analysis']['size_only_folders'] = [x for x in lst if x != f] if f in lst else lst + [f]
        p.save()
        self.show_project(p)

    def add_folder(self):
        p = self.win.project
        if p is None:
            return
        d = QFileDialog.getExistingDirectory(self, 'Photo folder to add')
        if d:
            p.add_folder(d)
            self.show_project(p)


class SettingsPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.loading = False
        lay = QVBoxLayout(self)
        lay.addWidget(h1('2  Settings'))
        lay.addWidget(note('Changes are saved automatically. If you change a setting after an analysis, '
                           'the photos are analysed again at the next Start.'))
        # colour
        g = QGroupBox('Colour correction')
        gl = QVBoxLayout(g)
        self.col = QButtonGroup(self)
        for i, (key, text) in enumerate((
                ('none', 'None'),
                ('nominal', 'Nominal values of the colour chart (in every photo)'),
                ('reference', 'Reference photo: every chart is aligned to the chart of the chosen photo'),
                ('shared', 'Single calibration: the transform of the reference photo is applied to all photos '
                           '(also photos without a chart)'))):
            b = QRadioButton(text)
            b.setProperty('key', key)
            self.col.addButton(b, i)
            gl.addWidget(b)
        self.col.buttonClicked.connect(self.changed)
        row = QHBoxLayout()
        self.ref_label = QLabel()
        b_ref = QPushButton('Choose the reference photo…')
        b_ref.clicked.connect(self.pick_reference)
        b_csv = QPushButton('Import CSV from the ImageJ plugin…')
        b_csv.clicked.connect(self.import_reference)
        row.addWidget(b_ref)
        row.addWidget(b_csv)
        row.addStretch()
        gl.addLayout(row)
        gl.addWidget(self.ref_label)
        gl.addWidget(note('Photos without a colour chart are measured uncorrected and flagged in the CSV.'))
        lay.addWidget(g)
        # scale
        g = QGroupBox('Scale')
        gl = QVBoxLayout(g)
        self.sc = QButtonGroup(self)
        for i, (key, text) in enumerate((('auto', 'Automatic from the graph paper (smallest square = 1 mm, measured '
                                                  'over at least 20 squares); doubtful ones are shown in Review'),
                                         ('manual', 'By hand: in Review draw a known length (e.g. 2 cm) on a photo, '
                                                    'for that photo or its whole folder'),
                                         ('fixed', 'Fixed value'), ('none', 'None: measures in pixels'))):
            b = QRadioButton(text)
            b.setProperty('key', key)
            self.sc.addButton(b, i)
            gl.addWidget(b)
        self.sc.buttonClicked.connect(self.changed)
        self.ppm = QDoubleSpinBox()
        self.ppm.setRange(0, 10000)
        self.ppm.setDecimals(3)
        self.ppm.setSuffix(' px/mm')
        self.ppm.valueChanged.connect(self.changed)
        row = QHBoxLayout()
        row.addWidget(self.ppm)
        b = QPushButton('Measure on a photo…')
        b.setToolTip('Draw a known length on one photo: its px/mm becomes the fixed value for the project')
        b.clicked.connect(self.measure_scale)
        row.addWidget(b)
        row.addStretch()
        gl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('Check: send to Review the photos whose scale or fish length differs from the median of '
                             'their folder by more than'))
        self.session_pct = QDoubleSpinBox()
        self.session_pct.setRange(0, 100)
        self.session_pct.setDecimals(0)
        self.session_pct.setSuffix(' %')
        self.session_pct.setToolTip('0 = no check. Fish of very different sizes in one folder: raise it.')
        self.session_pct.valueChanged.connect(self.changed)
        row.addWidget(self.session_pct)
        row.addStretch()
        gl.addLayout(row)
        lay.addWidget(g)
        g = QGroupBox('Fish outline')
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        self.outline_model = QComboBox()
        self.outline_model.currentIndexChanged.connect(self.changed)
        row.addWidget(self.outline_model, 1)
        b = QPushButton('Import outline model…')
        b.clicked.connect(self.import_outline)
        row.addWidget(b)
        b = QPushButton('Export…')
        b.clicked.connect(self.export_outline)
        row.addWidget(b)
        gl.addLayout(row)
        gl.addWidget(note('Automatic: a general detector finds any fish, no training needed. An outline model (e.g. '
                          'the one pre-trained on the van der Bijl 2025 guppy photos) draws the body and the tail '
                          'fin separately; where it finds no fish the automatic one is used. Changing it means '
                          'running the Analysis again.'))
        lay.addWidget(g)
        g = QGroupBox('Head of the fish in the photos')
        gl = QHBoxLayout(g)
        self.head = QButtonGroup(self)
        for i, (key, text) in enumerate((('left', 'Left'), ('right', 'Right'), ('auto', 'Guess in each photo'))):
            b = QRadioButton(text)
            b.setProperty('key', key)
            self.head.addButton(b, i)
            gl.addWidget(b)
        gl.addStretch()
        self.head.buttonClicked.connect(self.changed)
        lay.addWidget(g)
        # parts
        g = QGroupBox('Body parts included in the measures')
        gl = QVBoxLayout(g)
        self.parts = {}
        gl.addWidget(note('Body: always included. Separating the parts comes with the trained model (phase 2); '
                          'for now the choice is recorded in the CSV.'))
        for part in profile.load('guppy_male')['parts']:
            if part['optional']:
                c = QCheckBox(part['label'])
                c.toggled.connect(self.changed)
                self.parts[part['key']] = c
                gl.addWidget(c)
        lay.addWidget(g)
        # file names
        g = QGroupBox('File names')
        gl = QFormLayout(g)
        self.pattern = QLineEdit()
        self.pattern.setPlaceholderText('e.g. {id}_{tank}_{date}')
        self.pattern.textChanged.connect(self.changed)
        self.example = note('')
        gl.addRow('Pattern', self.pattern)
        gl.addRow('', self.example)
        lay.addWidget(g)
        # labels written in the photo
        g = QGroupBox('Fish ID')
        gl = QFormLayout(g)
        self.id_src = QButtonGroup(self)
        for i, (key, text) in enumerate((
                ('none', 'No id'),
                ('label', 'From the label in the photo (QR code or written text)'),
                ('filename', 'From the file name: the {id} part of the pattern above'),
                ('table', 'From a data table (photo name → id); the other photos of the folder take the id '
                          'of their neighbours in shooting order, checked against the label'))):
            b = QRadioButton(text)
            b.setProperty('key', key)
            self.id_src.addButton(b, i)
            gl.addRow(b)
        self.id_src.buttonClicked.connect(self.changed)
        self.read_label = QCheckBox('Also read the label in the photo for its date and group')
        self.read_label.toggled.connect(self.changed)
        gl.addRow(self.read_label)
        self.lab_fields = {}
        for key, text in (('male_id', 'Id format'), ('group', 'Group format'), ('date', 'Date format')):
            e = QLineEdit()
            e.textChanged.connect(self.changed)
            self.lab_fields[key] = e
            gl.addRow(text, e)
        row = QHBoxLayout()
        self.ids_label = QLabel()
        b_ids = QPushButton('List of valid ids (Excel or CSV)…')
        b_ids.clicked.connect(self.pick_ids)
        row.addWidget(b_ids)
        row.addWidget(self.ids_label, 1)
        gl.addRow(row)
        row = QHBoxLayout()
        self.table_label = QLabel()
        b_tab = QPushButton('Data table: photo → id (Excel or CSV)…')
        b_tab.setToolTip('A sheet with the fish id and the name of its photo (e.g. SL_pictureID), for "From a data table".')
        b_tab.clicked.connect(self.pick_table)
        row.addWidget(b_tab)
        row.addWidget(self.table_label, 1)
        gl.addRow(row)
        b_print = QPushButton('Print labels with QR code…')
        b_print.clicked.connect(self.print_labels)
        gl.addRow(b_print)
        gl.addRow(note(FORMAT_HELP))
        self.try_text = QLineEdit()
        self.try_text.setPlaceholderText('type a label as written, e.g. G4 20/03 M116')
        self.try_text.textChanged.connect(self.try_formats)
        gl.addRow('Try the formats on', self.try_text)
        self.try_result = QLabel('')
        self.try_result.setWordWrap(True)
        gl.addRow(self.try_result)
        gl.addRow(note('With the list of valid ids, misread labels are corrected. EXIF data (date taken, camera, '
                       'exposure, ISO, focal length) always go in the CSV.'))
        lay.addWidget(g)
        # what to measure
        g = QGroupBox('What to measure')
        gl = QFormLayout(g)
        self.an_default = QButtonGroup(self)
        for i, (key, text) in enumerate((('size_colour', 'Size and colour on every fish'),
                                         ('size', 'Size only on every fish'))):
            b = QRadioButton(text)
            b.setProperty('key', key)
            self.an_default.addButton(b, i)
            gl.addRow(b)
        self.an_default.buttonClicked.connect(self.changed)
        self.size_only = QLineEdit()
        self.size_only.setPlaceholderText(r'e.g. _F\d+$ for F100_F1')
        self.size_only.textChanged.connect(self.changed)
        gl.addRow('Size only when the id or file name matches', self.size_only)
        gl.addRow(note('Folders can be set to size only in Project, single photos in Review '
                       '(e.g. females, where colour is not measured).'))
        self.opt_lay = QVBoxLayout()
        gl.addRow(self.opt_lay)
        self.opt_classes = {}
        lay.addWidget(g)
        lay.addStretch()

    def show_project(self, p):
        self.loading = True
        st = p.settings
        for b in self.col.buttons():
            b.setChecked(b.property('key') == st['colour']['mode'])
        for b in self.sc.buttons():
            b.setChecked(b.property('key') == st['scale']['mode'])
        self.ppm.setValue(float(st['scale'].get('px_per_mm') or 0))
        self.session_pct.setValue(float(st.get('checks', {}).get('session_pct') or 0))
        for b in self.head.buttons():
            b.setChecked(b.property('key') == st.get('head', 'left'))
        for k, c in self.parts.items():
            c.setChecked(bool(st['parts'].get(k)))
        self.pattern.setText(st.get('filename_pattern', ''))
        for b in self.id_src.buttons():
            b.setChecked(b.property('key') == st['labels'].get('id_source', 'none'))
        self.read_label.setChecked(bool(st['labels'].get('read_label')))
        for b in self.an_default.buttons():
            b.setChecked(b.property('key') == st['analysis'].get('default', 'size_colour'))
        self.size_only.setText(st['analysis'].get('size_only_pattern', ''))
        for c in self.opt_classes.values():
            c.deleteLater()
        self.opt_classes = {}
        for c in profile.load(st['profile'])['colour_classes']:
            if c.get('optional'):
                cb = QCheckBox(f"Measure the colour class {c['label']}")
                cb.setToolTip('Optional class: when off, its pixels count as body without colour, it has no brush in '
                              'Training and its CSV columns are NA')
                cb.setChecked(c['key'] not in (st.get('colours_off') or []))
                cb.toggled.connect(self.changed)
                self.opt_lay.addWidget(cb)
                self.opt_classes[c['key']] = cb
        self.fill_outline_models(p)
        for k, e in self.lab_fields.items():
            e.setText(st['labels'].get(k, ''))
        self.loading = False
        self.refresh_labels()

    def try_formats(self, *_):
        text = self.try_text.text()
        if not text.strip():
            self.try_result.setText('')
            return
        out = []
        for key, name in (('male_id', 'id'), ('group', 'group'), ('date', 'date')):
            fmt = self.lab_fields[key].text().strip()
            try:
                m = re.search(fmt, text) if fmt else None
            except re.error:
                out.append(f'{name}: the format is not valid')
                continue
            out.append(f'{name}: <b>{html.escape(m.group(0))}</b>' if m else f'{name}: not found')
        self.try_result.setText(' · '.join(out))

    def refresh_labels(self):
        p = self.win.project
        ref = p.settings['colour'].get('reference_photo')
        has = os.path.exists(p.reference_csv)
        self.ref_label.setText(('Reference: ' + (os.path.basename(ref) if ref else 'imported CSV')) if has
                               else 'No reference photo chosen.')
        self.ppm.setEnabled(p.settings['scale']['mode'] == 'fixed')
        lb = p.settings['labels']
        for e in self.lab_fields.values():
            e.setEnabled(pipeline.reads_label(p.settings))
        self.read_label.setEnabled(lb.get('id_source') in ('none', 'filename'))
        ids = pipeline.load_expected_ids(lb.get('expected_ids_file'), lb.get('male_id') or '.+')
        self.ids_label.setText(f"{os.path.basename(lb['expected_ids_file'])}: {len(ids or [])} ids found"
                               if lb.get('expected_ids_file') else 'no list (optional)')
        self.table_label.setText(os.path.basename(lb['id_table_file']) if lb.get('id_table_file')
                                 else 'no table')
        photos = p.photos()
        if photos and self.pattern.text():
            f = os.path.basename(photos[0][1])
            fields = metadata.parse_filename(f, self.pattern.text())
            ok = any(fields.values())
            self.example.setText(f + '  →  ' + (', '.join(f'{k} = {v}' for k, v in fields.items()) if ok
                                                 else 'the pattern does not match this name'))
        else:
            self.example.setText('The parts in braces become CSV columns. Name one {id} to use it as the fish id.')
        self.try_formats()

    def fill_outline_models(self, p):
        from .. import outline
        loading, self.loading = self.loading, True
        self.outline_model.clear()
        self.outline_model.addItem('Automatic (YOLOE + SAM 2.1)', '')
        for name, meta in outline.list_models(p):
            src = (meta.get('imported_from') or {}).get('project') or meta.get('source') or ''
            self.outline_model.addItem(f"{name}: {', '.join(meta.get('classes') or [])}"
                                       + (f' ({src[:60]})' if src else ''), name)
        i = self.outline_model.findData(p.settings['outline'].get('model', ''))
        self.outline_model.setCurrentIndex(max(i, 0))
        self.loading = loading

    def import_outline(self):
        if self.win.project is None:
            return
        from .modelfiles import import_dialog
        name = import_dialog(self, self.win.project, 'outline')
        if name:
            self.win.project.settings['outline']['model'] = name
            self.win.project.save()
            self.fill_outline_models(self.win.project)

    def export_outline(self):
        name = self.outline_model.currentData()
        if self.win.project is None or not name:
            QMessageBox.information(self, 'Export', 'Choose an outline model first (the automatic one has nothing '
                                    'to export).')
            return
        from .modelfiles import export_dialog
        export_dialog(self, self.win.project, name)

    def measure_scale(self):
        folders = self.win.project.folders if self.win.project else []
        path, _ = QFileDialog.getOpenFileName(self, 'Photo with a ruler or graph paper', folders[0] if folders else '',
                                              'Photos (*.jpg *.jpeg *.png *.tif *.tiff)')
        if not path:
            return
        from .calibrate import CalibrateDialog
        d = CalibrateDialog(self, path)
        if d.exec():
            (ax, ay), (bx, by), mm, _ = d.result_line()
            ppm = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / mm
            for b in self.sc.buttons():
                b.setChecked(b.property('key') == 'fixed')
            self.ppm.setValue(ppm)
            self.changed()

    def changed(self, *_):
        if self.loading or self.win.project is None:
            return
        p = self.win.project
        st = p.settings
        st['colour']['mode'] = self.col.checkedButton().property('key') if self.col.checkedButton() else 'none'
        st['scale']['mode'] = self.sc.checkedButton().property('key') if self.sc.checkedButton() else 'auto'
        st['scale']['px_per_mm'] = self.ppm.value() or None
        st.setdefault('checks', {})['session_pct'] = self.session_pct.value()
        st['head'] = self.head.checkedButton().property('key') if self.head.checkedButton() else 'left'
        st['parts'] = {k: c.isChecked() for k, c in self.parts.items()}
        st['outline']['model'] = self.outline_model.currentData() or ''
        st['filename_pattern'] = self.pattern.text().strip()
        st['labels']['id_source'] = self.id_src.checkedButton().property('key') if self.id_src.checkedButton() else 'none'
        st['labels']['read_label'] = self.read_label.isChecked()
        st['analysis']['default'] = (self.an_default.checkedButton().property('key') if self.an_default.checkedButton()
                                     else 'size_colour')
        st['analysis']['size_only_pattern'] = self.size_only.text().strip()
        st['colours_off'] = sorted(k for k, c in self.opt_classes.items() if not c.isChecked())
        for k, e in self.lab_fields.items():
            st['labels'][k] = e.text().strip()
        p.save()
        self.refresh_labels()

    def print_labels(self):
        """Label sheet (QR + id) from the fish list, to put one label next to each fish in the photo."""
        from .. import labelsheet
        p = self.win.project
        lb = p.settings['labels']
        src, _ = QFileDialog.getOpenFileName(self, 'Fish list', lb.get('expected_ids_file') or '',
                                             'Excel or text (*.xlsx *.xlsm *.csv *.txt)')
        if not src:
            return
        id_col, ok = QInputDialog.getText(self, 'Labels', 'Name of the id column (empty = look for ids anywhere):')
        if not ok:
            return
        group_col, ok = QInputDialog.getText(self, 'Labels', 'Name of the group column (optional):')
        if not ok:
            return
        date, ok = QInputDialog.getText(self, 'Labels', 'Date to print, e.g. 20/03 (optional):')
        if not ok:
            return
        try:
            rows = labelsheet.read_rows(src, id_col=id_col.strip() or None, group_col=group_col.strip() or None,
                                        date=date.strip() or None)
        except Exception as e:
            QMessageBox.warning(self, 'Labels', f'Cannot read the list: {e}')
            return
        if not rows:
            QMessageBox.warning(self, 'Labels', 'No ids (e.g. M116) found in the list.')
            return
        out, _ = QFileDialog.getSaveFileName(self, 'Save the PDF', os.path.join(p.path, 'labels.pdf'), 'PDF (*.pdf)')
        if not out:
            return
        pages = labelsheet.make_sheet(rows, out)
        QMessageBox.information(self, 'Labels', f'{len(rows)} labels on {pages} A4 pages: {out}\n'
                                'Print at 100% (no fit to page) to get the right size.')
        QDesktopServices.openUrl(QUrl.fromLocalFile(out))

    def pick_ids(self):
        p = self.win.project
        f, _ = QFileDialog.getOpenFileName(self, 'List of valid ids', '', 'Excel or text (*.xlsx *.xlsm *.csv *.txt)')
        if f:
            p.settings['labels']['expected_ids_file'] = f
            p.save()
            self.refresh_labels()

    def pick_table(self):
        p = self.win.project
        f, _ = QFileDialog.getOpenFileName(self, 'Data table with fish id and photo name', '',
                                           'Excel or CSV (*.xlsx *.xlsm *.csv)')
        if f:
            p.settings['labels']['id_table_file'] = f
            p.save()
            self.refresh_labels()

    def pick_reference(self):
        p = self.win.project
        start = p.folders[0] if p.folders else ''
        f, _ = QFileDialog.getOpenFileName(self, 'Reference photo', start, 'Images (*.jpg *.jpeg *.png *.tif *.tiff)')
        if not f:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok = pipeline.set_reference_photo(p, f)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            QMessageBox.warning(self, 'Reference', 'No complete colour chart found in this photo.')
        self.refresh_labels()

    def import_reference(self):
        p = self.win.project
        f, _ = QFileDialog.getOpenFileName(self, 'Reference CSV (Patch,Name,R,G,B)', '', 'CSV (*.csv)')
        if not f:
            return
        import shutil
        shutil.copyfile(f, p.reference_csv)
        if pipeline.load_reference(p.reference_csv) is None:
            os.remove(p.reference_csv)
            QMessageBox.warning(self, 'Reference', 'The CSV must have 24 rows with columns Patch,Name,R,G,B.')
        else:
            p.settings['colour']['reference_photo'] = ''
            p.save()
        self.refresh_labels()


class Worker(QObject):
    progress = Signal(int, int, dict)
    finished = Signal(int)
    failed = Signal(str)
    device = Signal(str)

    def __init__(self, project):
        super().__init__()
        self.project = project
        self.stop = False

    def run(self):
        try:
            from ..engine import segment
            self.device.emit(segment.device_text())
            n = pipeline.run(self.project, progress=lambda i, n, r: self.progress.emit(i, n, r), stop=lambda: self.stop)
        except Exception as e:      # models missing or not the expected files, no internet on first run
            self.failed.emit(f'{type(e).__name__}: {e}')
            n = 0
        self.finished.emit(n)


class AnalysisPage(QWidget):
    done = Signal()

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.thread = None
        lay = QVBoxLayout(self)
        lay.addWidget(h1('3  Analysis'))
        row = QHBoxLayout()
        self.b_start, self.b_stop = QPushButton('Start analysis'), QPushButton('Stop')
        self.b_start.clicked.connect(self.start)
        self.b_stop.clicked.connect(self.halt)
        self.b_stop.setEnabled(False)
        row.addWidget(self.b_start)
        row.addWidget(self.b_stop)
        row.addStretch()
        lay.addLayout(row)
        self.bar = QProgressBar()
        lay.addWidget(self.bar)
        self.status = note('')
        lay.addWidget(self.status)
        self.runs_on = note('')
        lay.addWidget(self.runs_on)
        lay.addWidget(note('The first run downloads and verifies the models (about 830 MB, once, needs internet). '
                           'You can stop and resume: photos already done are not redone.'))
        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QPixmap(180, 120).size())
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setSpacing(8)
        lay.addWidget(self.grid, 1)

    def show_project(self, p):
        self.grid.clear()
        for r in p.results():
            self.add_thumb(r)
        self.update_counts()

    def add_thumb(self, r):
        p = self.win.project
        img = os.path.join(p.path, 'check', r['photo_id'] + '.jpg')
        it = QListWidgetItem(QIcon(img) if os.path.exists(img) else QIcon(), f"{r['file']}\n{STATUS_LABEL.get(r.get('status'), '')}")
        it.setForeground(QColor(STATUS_COLOUR.get(r.get('status'), '#1B1F24')))
        it.setToolTip(flags_text(r.get('flags')))
        self.grid.addItem(it)

    def update_counts(self):
        p = self.win.project
        res = p.results()
        c = {}
        for r in res:
            c[r.get('status')] = c.get(r.get('status'), 0) + 1
        self.status.setText(f"Photos: {len(p.photos())} · to analyse: {len(p.pending())} · ok: {c.get('ok', 0)} · "
                            f"to check: {c.get('to_check', 0)} · errors: {c.get('error', 0)}")

    def start(self):
        p = self.win.project
        if p.settings['colour']['mode'] in ('reference', 'shared') and not os.path.exists(p.reference_csv):
            QMessageBox.warning(self, 'Analysis', 'Choose the reference photo in Settings first.')
            return
        self.b_start.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.bar.setValue(0)
        self.worker = Worker(p)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.device.connect(lambda t: self.runs_on.setText(f'Runs on: {t}.'))
        self.seconds = []
        self.thread.start()

    def halt(self):
        if self.thread is not None:
            self.worker.stop = True
            self.b_stop.setEnabled(False)

    def on_progress(self, i, n, r):
        self.bar.setMaximum(n)
        self.bar.setValue(i)
        self.add_thumb(r)
        self.update_counts()
        if r.get('seconds'):
            self.seconds.append(float(r['seconds']))
            left = (n - i) * float(np.median(self.seconds)) / 60
            self.runs_on.setText(self.runs_on.text().split(' · ')[0] + f' · {np.median(self.seconds):.1f} s per photo'
                                 + (f' · about {left:.0f} min left' if i < n else ''))

    def on_failed(self, msg):
        QMessageBox.critical(self, 'Analysis', 'Analysis not started.\n\n' + msg)

    def on_finished(self, n):
        self.thread.quit()
        self.thread.wait()
        self.thread = None
        self.b_start.setEnabled(True)
        self.b_stop.setEnabled(False)
        self.show_project(self.win.project)
        self.done.emit()


class ClickLabel(QLabel):
    clicked = Signal(float, float)

    def mousePressEvent(self, e):
        self.clicked.emit(e.position().x(), e.position().y())


def load_pixmap(path):
    r = QImageReader(path)
    r.setAutoTransform(True)        # respect the EXIF rotation, like the analysis does
    return QPixmap.fromImage(r.read())


LABEL_FLAGS = {'label_uncertain', 'date_mismatch'}
VIEWS = ('Fish', 'Label', 'Scale')


def label_summary(r):
    """What the label reader found, field by field, and the rest of the text it read (not used)."""
    esc = lambda v: html.escape(str(v)) if v else '<i>not found</i>'
    src = {'qr': 'QR code', 'ocr': 'written text', 'filename': 'file name', 'table': 'data table',
           'sequence': 'neighbouring photos', 'manual': 'typed in Review'}.get(r.get('label_source'), 'nothing')
    out = (f"Id: <b>{esc(r.get('label_id'))}</b> · group: {esc(r.get('label_group'))} · date: "
           f"{esc(r.get('label_date'))} <span style='color:#4F5761'>(from the {src})</span>")
    others = [c for c in (r.get('label_candidates') or []) if c != r.get('label_id')]
    if others:
        out += '<br>Other possible readings of the id: ' + ', '.join(html.escape(c) for c in others) + \
               ' (pick one in the box)'
    if r.get('label_text'):
        out += ('<br><span style="color:#4F5761">All the text read on the label, only the fields above are used: '
                + html.escape(r['label_text']) + '</span>')
    return out


def view_of(flag):
    """Which part of the photo a flag is about."""
    if flag in scaleline.SCALE_FLAGS or flag in pipeline.SESSION_FLAGS:
        return 'Scale'
    return 'Label' if flag in LABEL_FLAGS else 'Fish'


class ReviewPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.addWidget(h1('4  Review'))
        lay.addWidget(note('Most uncertain photos first. Each photo shows what needs a look in separate views: '
                           'Fish (outline and measures), Label (the written id), Scale (graph paper or ruler). '
                           'Views with a \u26a0 have something to check.'))
        split = QSplitter()
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self.show_item)
        split.addWidget(self.list)
        right = QWidget()
        rl = QVBoxLayout(right)
        self.info = note('')
        rl.addWidget(self.info)
        self.views = QTabBar()
        for v in VIEWS:
            self.views.addTab(v)
        self.views.currentChanged.connect(lambda _: self.show_view())
        rl.addWidget(self.views)
        self.issues = QLabel()
        self.issues.setWordWrap(True)
        self.issues.setStyleSheet('color: #B3261E; font-weight: 600')
        rl.addWidget(self.issues)
        self.image = ClickLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.clicked.connect(self.click_fish)
        self.shown = None           # (pixmap size, crop box in original pixels) of the picture on screen
        self.image.setMinimumSize(400, 300)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        rl.addWidget(self.image, 1)
        self.tools = QStackedWidget()
        # Fish: what a click on the photo does
        fish = QWidget()
        fl = QVBoxLayout(fish)
        fl.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel('A click on the fish:'))
        self.mode_redo = QRadioButton('redoes the whole outline')
        self.mode_add = QRadioButton('adds the part you click to the outline (e.g. a transparent tail)')
        self.mode_redo.setChecked(True)
        for b in (self.mode_redo, self.mode_add):
            row.addWidget(b)
        row.addStretch()
        fl.addLayout(row)
        self.fish_note = note('Lines: magenta outline, cyan midline (mask length), green snout, red end of the '
                              'outline, orange end of the body, yellow maximum depth, blue from the landmarks '
                              '(legend in 5 Results).')
        fl.addWidget(self.fish_note)
        self.tools.addWidget(fish)
        # Label: the id written on the label
        lab = QWidget()
        ll = QHBoxLayout(lab)
        ll.setContentsMargins(0, 0, 0, 0)
        self.lab_text = QLabel()
        self.lab_text.setWordWrap(True)
        self.lab_text.setTextFormat(Qt.RichText)
        self.lab_id = QComboBox()
        self.lab_id.setEditable(True)
        self.lab_id.setMinimumWidth(140)
        self.lab_id.setToolTip('The id read, the other readings of the label (pick one) or type it')
        self.lab_id.lineEdit().setPlaceholderText('fish id')
        b_lab = QPushButton('Save id')
        b_lab.clicked.connect(self.save_label)
        self.lab_id.lineEdit().returnPressed.connect(self.save_label)
        ll.addWidget(self.lab_text, 1)
        ll.addWidget(self.lab_id)
        ll.addWidget(b_lab)
        self.tools.addWidget(lab)
        # Scale
        sc = QWidget()
        sl = QHBoxLayout(sc)
        sl.setContentsMargins(0, 0, 0, 0)
        self.b_confirm = QPushButton('Confirm scale (the red line is 2 cm)')
        self.b_confirm.clicked.connect(self.confirm_scale)
        self.b_cal = QPushButton('Calibrate manually (draw a known length)')
        self.b_cal.clicked.connect(self.start_calibration)
        self.scale_text = QLabel()
        self.scale_text.setWordWrap(True)
        sl.addWidget(self.scale_text, 1)
        sl.addWidget(self.b_confirm)
        sl.addWidget(self.b_cal)
        self.tools.addWidget(sc)
        rl.addWidget(self.tools)
        whole = QGroupBox('The whole photo (all views): when everything above is fixed')
        row = QHBoxLayout(whole)
        for text, fn in (('Accept the photo (Enter)', lambda: self.decide('reviewed')),
                         ('Exclude the photo', lambda: self.decide('excluded')),
                         ('Measure size only', lambda: self.set_analysis('size')),
                         ('Measure size + colour', lambda: self.set_analysis('size_colour')),
                         ('Open the original', self.open_original)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
            if text.startswith('Accept'):
                b.setShortcut(Qt.Key_Return)
        row.addStretch()
        rl.addWidget(whole)
        split.addWidget(right)
        split.setSizes([260, 800])
        lay.addWidget(split, 1)
        self.items = []
        self.base = None
        self.keep = None            # photo to keep in the list after an outline change, even if it is now ok

    def show_project(self, p):
        order = {'error': 0, 'to_check': 1}
        self.items = sorted((r for r in pipeline.session_checks(p) if r.get('status') in order or r['path'] == self.keep),
                            key=lambda r: (order.get(r['status'], 2), -len((r.get('flags') or '').split(';'))))
        self.list.clear()
        for r in self.items:
            it = QListWidgetItem(f"{r['file']}\n{flags_text(r.get('flags')).splitlines()[0]}")
            it.setForeground(QColor(STATUS_COLOUR.get(r['status'], '#1F6FB2')))
            self.list.addItem(it)
        if self.items:
            self.list.setCurrentRow(0)
        else:
            self.image.clear()
            self.issues.clear()
            self.info.setText('Nothing to check.')

    def flags_by_view(self, r):
        out = {v: [] for v in VIEWS}
        for f in (r.get('flags') or '').split(';'):
            if f:
                out[view_of(f)].append(f)
        if r.get('label_needs_review') and not out['Label']:
            out['Label'].append('label_uncertain')
        return out

    def show_item(self, i):
        if not 0 <= i < len(self.items):
            return
        r = self.items[i]
        mode = {'size': 'size only', 'size_colour': 'size + colour'}.get(r.get('analysis'), '')
        status = STATUS_LABEL.get(r.get('status'), r.get('status'))
        self.info.setText(f"<b>{html.escape(r['file'])}</b> ({html.escape(r['folder'])}) · {mode} · {status}")
        by = self.flags_by_view(r)
        serious = [v for v in VIEWS if any(f in pipeline.SERIOUS or f == 'fish_not_found' for f in by[v])]
        any_flag = [v for v in VIEWS if by[v]]
        self.views.blockSignals(True)
        for k, v in enumerate(VIEWS):
            self.views.setTabText(k, ('\u26a0 ' if by[v] else '') + v)
        start = (serious or any_flag or ['Fish'])[0]
        self.views.setCurrentIndex(VIEWS.index(start))
        self.views.blockSignals(False)
        self.show_view()

    def show_view(self):
        r = self.current()
        if r is None:
            return
        v = VIEWS[self.views.currentIndex()]
        self.tools.setCurrentIndex(VIEWS.index(v))
        fl = self.flags_by_view(r)[v]
        self.issues.setText('\n'.join(FLAG_LABEL.get(f, f) for f in fl) if fl else 'Nothing to check here.')
        self.issues.setStyleSheet('color: %s; font-weight: 600' % ('#B3261E' if fl else '#2E7D32'))
        self.image.setCursor(Qt.CrossCursor if v == 'Fish' else Qt.ArrowCursor)
        pm_full = None
        lines = []
        if v == 'Fish':
            pm = measures_pixmap(self.win.project, r)
            if pm is not None and not pm.isNull():
                box = r['check_box']
            else:
                pm_full = pm = load_pixmap(r['path'])
                box = [0, 0, pm.width(), pm.height()]
        elif v == 'Label':
            pm_full = load_pixmap(r['path'])
            b = r.get('label_box')
            if b:                               # the label with some room around it
                x, y, w, h = b
                pad = int(0.6 * max(w, h))
                box = [max(0, x - pad), max(0, y - pad), min(pm_full.width(), x + w + pad),
                       min(pm_full.height(), y + h + pad)]
            else:                               # not found: the whole photo, to read it anywhere
                box = [0, 0, pm_full.width(), pm_full.height()]
            pm = pm_full.copy(box[0], box[1], box[2] - box[0], box[3] - box[1])
            self.lab_id.clear()
            for c in dict.fromkeys([r.get('label_id')] + list(r.get('label_candidates') or [])):
                if c:
                    self.lab_id.addItem(c)
            self.lab_id.setEditText(r.get('label_id') or '')
            self.lab_text.setText(label_summary(r) + ('' if b else '<br><i>Label not found: the whole photo is '
                                                                   'shown.</i>'))
        else:
            pm = load_pixmap(r['path'])
            box = [0, 0, pm.width(), pm.height()]
            line = r.get('scale_line')
            hand = scaleline.photo_line(self.win.project, r['path']) if r.get('scale_source') == 'line' else None
            if hand:                                    # the line drawn by hand replaces the automatic one
                lines, line = [hand[:2]], None
            elif line and r.get('scale_source') != 'line_folder':
                lines = [line]
            else:
                line = None
            self.b_confirm.setVisible(bool(line))
            if hand:
                self.scale_text.setText(f"Scale set by hand: {r['px_per_mm']:.2f} px/mm (the red line is the "
                                        f"{hand[2]:g} mm you drew). Lengths in mm are already updated: no new "
                                        "analysis needed.")
            elif r.get('scale_source') == 'line_folder':
                self.scale_text.setText(f"Scale set by hand for the whole folder: {r['px_per_mm']:.2f} px/mm (line "
                                        "drawn on another photo). Lengths in mm are already updated: no new "
                                        "analysis needed.")
            elif r.get('px_per_mm'):
                self.scale_text.setText(f"Scale: {r['px_per_mm']:.2f} px/mm ({r.get('scale_source') or '?'}"
                                        + (f", {r['scale_squares']} squares" if r.get('scale_squares') else '') + ')'
                                        + ('. If the red line covers 2 cm, Confirm; otherwise Calibrate manually.'
                                           if line else ''))
            else:
                self.scale_text.setText('No scale: Calibrate manually.')
        self.base = (pm, box)
        self.draw(lines)

    def draw(self, lines, points=()):
        """Photo on screen with red lines/points given in photo pixels."""
        pm, box = self.base
        if lines or points:
            pm = pm.copy()
            qp = QPainter(pm)
            pen = QPen(QColor('#FF1744'))
            pen.setWidthF(max(2.0, pm.width() / 400))
            qp.setPen(pen)
            for (ax, ay), (bx, by) in lines:
                qp.drawLine(int(ax - box[0]), int(ay - box[1]), int(bx - box[0]), int(by - box[1]))
            for x, y in points:
                rr = max(4, pm.width() // 250)
                qp.drawEllipse(int(x - box[0]) - rr, int(y - box[1]) - rr, 2 * rr, 2 * rr)
            qp.end()
        pm = pm.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(pm)
        self.shown = (pm.width(), pm.height(), box)

    def current(self):
        i = self.list.currentRow()
        return self.items[i] if 0 <= i < len(self.items) else None

    def confirm_scale(self):
        r = self.current()
        if r and r.get('scale_line'):
            a, b = r['scale_line']
            scaleline.save(self.win.project, r['path'], a, b, 20)
            self.after_scale(r)

    def start_calibration(self):
        r = self.current()
        if r is None:
            return
        from .calibrate import CalibrateDialog
        d = CalibrateDialog(self, r['path'], r['folder'])
        if d.exec():
            a, b, mm, whole = d.result_line()
            try:
                ppm = scaleline.save(self.win.project, r['path'], a, b, mm, whole)
                self.after_scale(r, f'Scale set: {ppm:.2f} px/mm.')
            except ValueError as e:
                QMessageBox.warning(self, 'Calibrate manually', str(e))

    def after_scale(self, r, msg=None):
        i = self.list.currentRow()
        self.keep = r['path']               # stay on this photo: the other views may still need a look
        self.show_project(self.win.project)
        self.keep = None
        rows = [j for j, it in enumerate(self.items) if it['path'] == r['path']]
        if rows:
            self.list.setCurrentRow(rows[0])
        elif self.items:
            self.list.setCurrentRow(min(i, len(self.items) - 1))
        if msg:
            self.info.setText(html.escape(msg) + '<br>' + self.info.text())     # info is rich text

    def save_label(self):
        i = self.list.currentRow()
        if not 0 <= i < len(self.items) or not self.lab_id.currentText().strip():
            return
        self.win.project.set_label_id(self.items[i]['path'], self.lab_id.currentText().strip().upper())
        self.items[i] = self.win.project.load_result(self.items[i]['path'])
        self.show_item(i)

    def click_fish(self, x, y):
        i = self.list.currentRow()
        if self.shown is None or not 0 <= i < len(self.items) or VIEWS[self.views.currentIndex()] != 'Fish':
            return
        pw, ph, (x0, y0, x1, y1) = self.shown
        fx = (x - (self.image.width() - pw) / 2) / pw
        fy = (y - (self.image.height() - ph) / 2) / ph
        if not (0 <= fx <= 1 and 0 <= fy <= 1):
            return
        r, p = self.items[i], self.win.project
        photo = r['path']
        adding = self.mode_add.isChecked()
        before = r.get('fish_area_px')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from ..engine import segment
            img = pipeline.load_image(photo)
            X, Y = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
            if adding:
                from .. import training
                old = training.fish_mask(p, photo)
                mask = segment.add_to_mask(img, old, X, Y) if old is not None else None
                if mask is None or mask.sum() <= old.sum():
                    raise ValueError('nothing new was found next to the outline at that point. Click on the '
                                     'missing part itself, close to the fish (for a very transparent tail try '
                                     'its darker rays or its edge).')
            else:
                mask = segment.segment_fish_click(img, X, Y)
            areas = [o['fish_area_px'] for o in p.results() if o.get('fish_area_px') and o['path'] != photo]
            if mask is None or not mask.any():
                raise ValueError('no fish at that point: click in the middle of the body')
            if areas and mask.sum() > 3 * float(np.median(areas)):
                QApplication.restoreOverrideCursor()
                keep = QMessageBox.question(
                    self, 'Review', 'The new outline is much larger than the other fish: it probably includes the '
                    'background. Keep it anyway? (No: click again, in the middle of the body)') == QMessageBox.Yes
                QApplication.setOverrideCursor(Qt.WaitCursor)
                if not keep:
                    return
            pipeline.save_manual_mask(p, photo, mask)
            new = pipeline.analyse(p, os.path.dirname(photo), photo)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, 'Review', ('Could not add to the outline: ' if adding else
                                                 'Could not redo the outline: ') + str(e))
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        self.keep = photo                    # stay on this photo to see the new outline, even if it is now fine
        self.show_project(p)
        self.keep = None
        for j, it in enumerate(self.items):
            if it['path'] == photo:
                self.list.setCurrentRow(j)
        after = new.get('fish_area_px')
        change = (f' The outline grew by {100 * (after - before) / before:.0f}%.' if adding and before and after
                  else '')
        tail = ('The tail fin is now in the outline.' if new.get('caudal_in_mask')
                else 'The tail fin is still not all in the outline: click on the missing part again, or mark the '
                     'tail tip on screen 8 (the total length then comes from it).')
        left = flags_text(new.get('flags')) if new.get('status') == 'to_check' else 'Nothing serious left: press Accept.'
        QMessageBox.information(self, 'Review', f'New outline saved and the photo measured again.{change}\n{tail}'
                                f'\n\nStill to check:\n{left}')

    def set_analysis(self, mode):
        i = self.list.currentRow()
        if not 0 <= i < len(self.items):
            return
        photo = self.items[i]['path']
        self.win.project.set_analysis(photo, mode)
        self.items[i] = self.win.project.load_result(photo)
        self.show_item(i)

    def decide(self, status):
        i = self.list.currentRow()
        if not 0 <= i < len(self.items):
            return
        self.win.project.set_status(self.items[i]['path'], status)
        self.show_project(self.win.project)
        self.list.setCurrentRow(min(i, len(self.items) - 1))

    def open_original(self):
        i = self.list.currentRow()
        if 0 <= i < len(self.items):
            path = self.items[i].get('path', '')
            if os.path.isfile(path) and path.lower().endswith(metadata.IMAGE_EXT):    # never launch anything but a photo
                QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def measures_legend():
    """Coloured legend of the measures drawn on the check pictures (pipeline.save_overlay) and of the landmarks."""
    c = pipeline.MEASURE_COLOURS
    hexc = (lambda k: '#%02x%02x%02x' % c[k])
    items = [('outline', 'outline of the fish (areas)'), ('midline', 'midline: mask length, snout to end of the outline'),
             ('snout', 'snout'), ('end', 'end of the outline'),
             ('peduncle', 'end of the body (caudal peduncle), only when the tail fin is in the outline'),
             ('depth', 'maximum depth')]
    rows = [f'<span style="color:{hexc(k)}; font-weight:700">&#9644;</span> {t}' for k, t in items]
    rows.append(f'<span style="color:{LANDMARK_COLOUR}; font-weight:700">&#9679;&#9644;</span> from the landmarks, '
                'and then they win over the outline: SL (snout to end of the peduncle, along the fish), TL = SL + the '
                'tail fin (dashed, to the tail tip), end of the body (the arc through the caudal fin base points), '
                'maximum depth across the body at the gonopodium base')
    rows.append(f'<span style="color:{COLOUR_START_COLOUR}; font-weight:700">&#8943;</span> colours start here '
                '(the head before it is left out of the colours, e.g. 2 mm behind the eye)')
    return '<br>'.join(rows)


LANDMARK_COLOUR = '#2979FF'
FORMAT_HELP = ('How to write a format: letters stand for themselves (M is the letter M); \\d is one digit (0–9); '
               'a number in braces repeats what comes just before it: \\d{3} = exactly 3 digits, \\d{1,2} = 1 or 2 '
               'digits; / and - are themselves. Examples: M\\d{3} finds M116; G\\d{1,2} finds G4 and G12; '
               '\\d{1,2}/\\d{2} finds 20/03. (Braces mean something else in the file-name pattern above: there '
               '{id} names a part of the name.) Try a label in the box below.')
COLOUR_START_COLOUR = '#FFD600'


def measures_pixmap(project, r):
    """check/<photo>.jpg (measures drawn at analysis) + the landmarks as they are now, or None."""
    img = os.path.join(project.path, 'check', r['photo_id'] + '.jpg')
    if not (os.path.exists(img) and r.get('check_box')):
        return None
    pm = load_pixmap(img)
    pts = r.get('landmarks') or {}
    if (pts or r.get('colour_start_line')) and not pm.isNull():
        x0, y0, x1, _ = r['check_box']
        s = pm.width() / max(x1 - x0, 1)
        to = (lambda p: (int((p[0] - x0) * s), int((p[1] - y0) * s)))
        qp = QPainter(pm)
        qp.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(LANDMARK_COLOUR))
        pen.setWidthF(max(2.0, pm.width() / 200))
        qp.setPen(pen)
        def poly(line):
            for a, b in zip(line[:-1], line[1:]):
                (ax, ay), (bx, by) = to(a), to(b)
                qp.drawLine(ax, ay, bx, by)
        sl = r.get('sl_path') or ([pts['snout'], pts['peduncle_end']] if pts.get('snout') and pts.get('peduncle_end')
                                  else None)
        for line in filter(None, [r.get('body_end_line'), r.get('depth_line'), sl]):
            poly(line)
        tail = r.get('tl_path') or ([pts['snout'], pts['tail_tip']] if pts.get('snout') and pts.get('tail_tip')
                                    else None)
        if tail:                                                 # TL = SL + this tail part: dashed
            dash = QPen(pen)
            dash.setStyle(Qt.DashLine)
            qp.setPen(dash)
            poly(tail)
            qp.setPen(pen)
        if r.get('colour_start_line'):                           # colours start here (head left out)
            dot = QPen(QColor(COLOUR_START_COLOUR))
            dot.setWidthF(pen.widthF())
            dot.setStyle(Qt.DotLine)
            qp.setPen(dot)
            (ax, ay), (bx, by) = map(to, r['colour_start_line'])
            qp.drawLine(ax, ay, bx, by)
            qp.setPen(pen)
        qp.setBrush(QColor(LANDMARK_COLOUR))
        rr = max(5, pm.width() // 90)
        for p in pts.values():
            x, y = to(p)
            qp.drawEllipse(x - rr, y - rr, 2 * rr, 2 * rr)
        qp.end()
    return pm


class ResultsPage(QWidget):
    COLS = ['file', 'folder', 'label_id', 'status', 'px_per_mm', 'standard_length_mm', 'total_length_mm',
            'mask_length_mm', 'max_depth_mm', 'fish_area_mm2', 'body_area_mm2', 'colour_chart', 'flags']

    def __init__(self, win):
        super().__init__()
        self.win = win
        lay = QVBoxLayout(self)
        lay.addWidget(h1('5  Results'))
        row = QHBoxLayout()
        b = QPushButton('Export CSV for R')
        b.clicked.connect(self.export)
        b2 = QPushButton('Open the export folder')
        b2.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.win.project.dir('export'))))
        row.addWidget(b)
        row.addWidget(b2)
        row.addStretch()
        lay.addLayout(row)
        self.msg = note('')
        lay.addWidget(self.msg)
        self.summary = note('')
        lay.addWidget(self.summary)
        split = QSplitter(Qt.Vertical)
        self.table = QTableWidget()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.currentCellChanged.connect(lambda r, *_: self.show_row(r))
        split.addWidget(self.table)
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        self.image = QLabel('Click a row to see how its measures were taken.')
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(300, 200)
        self.image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        bl.addWidget(self.image, 3)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        self.legend.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.legend.setTextFormat(Qt.RichText)
        bl.addWidget(self.legend, 2)
        split.addWidget(bottom)
        split.setSizes([400, 350])
        lay.addWidget(split, 1)
        self.rows = []
        self.pm = None

    def show_project(self, p):
        rows, _ = export.rows(p)
        self.rows = rows
        self.table.setColumnCount(len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, c in enumerate(self.COLS):
                self.table.setItem(i, j, QTableWidgetItem(str(r.get(c, ''))))
        n_sl = sum(r.get('standard_length_mm') not in (None, '', 'NA') for r in rows)
        text = f'Standard length (SL) on {n_sl} of {len(rows)} photos.'
        if n_sl < len(rows):
            text += (' SL comes from the landmarks (snout and end of the caudal peduncle): click them on screen '
                     '8 Landmarks, or train a landmark model there and run the Analysis again. '
                     'mask_length_mm is the length of the outline, which often misses the transparent tail fin.')
        self.summary.setText(text)
        self.legend.setText(measures_legend())
        self.image.clear()
        self.image.setText('Click a row to see how its measures were taken.')
        self.pm = None
        if rows:
            self.table.selectRow(0)

    def show_row(self, i):
        if not 0 <= i < len(self.rows) or self.win.project is None:
            return
        r = self.win.project.load_result(self.rows[i]['path'])
        self.pm = measures_pixmap(self.win.project, r) if r else None
        if self.pm is None or self.pm.isNull():
            self.image.clear()
            self.image.setText('No picture of the measures for this photo (run the Analysis again).')
            return
        self.fit()

    def fit(self):
        if self.pm is not None and not self.pm.isNull():
            self.image.setPixmap(self.pm.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.fit()

    def export(self):
        path = export.write_csv(self.win.project)
        self.msg.setText(f'Saved: {path}\nIn R: d <- read.csv("{path.replace(os.sep, "/")}")')


# ---------------------------------------------------------------- window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project = None
        self.setWindowTitle(f'Guppy Colour {__version__}')
        self.resize(1280, 800)
        self.nav = QListWidget()
        self.nav.setFixedWidth(220)
        self.nav.setStyleSheet('QListWidget { font-size: 15px; } QListWidget::item { padding: 10px 8px; }')
        self.stack = QStackedWidget()
        from .training_page import TrainingPage
        from .landmarks_page import LandmarksPage
        try:
            from .pattern_page import PatternPage
            pattern = PatternPage(self)
        except ImportError:          # screen 6 is added by its own module
            pattern = None
        self.pages = [ProjectPage(self), SettingsPage(self), AnalysisPage(self), ReviewPage(self), ResultsPage(self),
                      pattern, TrainingPage(self), LandmarksPage(self)]
        names = ['1  Project', '2  Settings', '3  Analysis', '4  Review', '5  Results', '6  Pattern and clusters',
                 '7  Training', '8  Landmarks']
        for name, page in zip(names, self.pages):
            it = QListWidgetItem(name)
            self.nav.addItem(it)
            if page is None:
                it.setFlags(Qt.NoItemFlags)
                it.setToolTip('Coming in phase 3')
                page = QWidget()
            scroll = QScrollArea()                   # small windows scroll instead of squashing the screen
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.nav.currentRowChanged.connect(self.go)
        self.pages[0].opened.connect(self.set_project)
        self.pages[2].done.connect(lambda: self.pages[3].show_project(self.project))
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        self.setCentralWidget(w)
        self.enable_pages(False)
        self.nav.setCurrentRow(0)

    def enable_pages(self, on):
        for i in range(1, len(self.pages)):
            if self.pages[i] is None:
                continue
            it = self.nav.item(i)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable if on else Qt.NoItemFlags)

    def set_project(self, p):
        self.project = p
        self.setWindowTitle(f'Guppy Colour {__version__} · {p.name}')
        self.enable_pages(True)
        self.pages[0].show_project(p)
        self.nav.setCurrentRow(1 if not p.results() else 2)

    def go(self, i):
        if i < 0 or i >= len(self.pages) or self.pages[i] is None:
            return
        if i > 0 and self.project is not None:
            self.pages[i].show_project(self.project)
        self.stack.setCurrentIndex(i)


def launch(argv=None):
    app = QApplication(argv or sys.argv)
    app.setApplicationName('Guppy Colour')
    win = MainWindow()
    win.show()
    return app.exec()

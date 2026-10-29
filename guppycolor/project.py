"""A project = one experiment: a folder holding settings, one result file per photo, masks and exports.
Photo folders (sessions) can be added at any time; only new or changed photos are analysed again."""
import copy
import datetime
import hashlib
import json
import os

from . import metadata, profile

PROJECT_FILE = 'project.json'

DEFAULT_SETTINGS = {
    'profile': 'guppy_male',
    # none | nominal (nominal chart values) | reference (align every chart to the reference photo's chart)
    # | shared (one transform from the reference photo, applied to all photos, even without a chart)
    'colour': {'mode': 'none', 'reference_photo': ''},
    # auto (graph paper / ruler) | fixed | none (results in pixels)
    'scale': {'mode': 'auto', 'px_per_mm': None},
    'head': 'left',                 # side of the head in the photos: left | right | auto (guessed per photo)
    'parts': {'caudal': True, 'dorsal': True, 'gonopodium': False},
    'filename_pattern': '',
    # text written in the photo (label, QR): patterns for each field, optional list of valid ids (xlsx/csv/txt),
    # optional data table with fish id + photo name (other photos of the folder get ids from their neighbours)
    # id_source: none | label (QR or text in the photo) | filename ({id} of the file-name pattern)
    # | table (data table photo -> id, other photos of the folder from their neighbours); read_label: also read
    # the label's date/group when the id comes from elsewhere
    'labels': {'id_source': 'none', 'read_label': False, 'male_id': r'M\d{3}', 'group': r'G\d{1,2}', 'date': r'\d{1,2}/\d{2}',
               'expected_ids_file': '', 'id_table_file': ''},
    # colour classes: trained model (modelli/<name>.npz, '' = off); spots closer than merge_mm count as one
    'colours': {'model': '', 'merge_mm': 0.0, 'min_spot_mm2': 0.0},
    # optional colour classes of the profile not measured (e.g. iridescent); new projects start with them off
    'colours_off': [],
    # what to measure on each fish: size_colour (size + colour classes) | size (size only, e.g. females).
    # Per photo: chosen by hand in Review > folder listed in size_only_folders > size_only_pattern (regex on the
    # fish id or file name) > default.
    'analysis': {'default': 'size_colour', 'size_only_folders': [], 'size_only_pattern': ''},
    'training': {'annotator': ''},
    'landmarks': {'model': ''},     # YOLO-pose model trained in screen 8 ('' = only landmarks clicked by hand)
    'outline': {'model': ''},
    # Review aids, not measures (so not in the settings hash): flag photos this % away from their session's median
    'checks': {'session_pct': 10},       # imported outline model (outline_vN); '' = zero-training detector (YOLOE + SAM)
}

# How the measures are made: raised when a new version measures differently, so every photo is analysed again.
MEASURES_VERSION = 2        # 2 (0.13): SL/TL along the midline, colours start behind the eye
# Settings that change the numbers: a change makes earlier results stale.
ANALYSIS_KEYS = ('colour', 'scale', 'head', 'parts', 'profile', 'labels', 'colours', 'analysis', 'landmarks')


class Project:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        with open(os.path.join(self.path, PROJECT_FILE), encoding='utf-8') as f:
            data = json.load(f)
        self.name = data['name']
        self.created = data.get('created', '')
        self.folders = data.get('folders', [])
        self.settings = copy.deepcopy(DEFAULT_SETTINGS)
        old_labels = data.get('settings', {}).get('labels', {})
        if 'enabled' in old_labels and 'id_source' not in old_labels:        # projects made with 0.1.x
            old_labels['id_source'] = 'label' if old_labels.pop('enabled') else 'none'
        st = data.get('settings', {})
        if st.get('profile') == 'guppy_maschio':
            st['profile'] = 'guppy_male'
        old_parts = {'caudale': 'caudal', 'dorsale': 'dorsal', 'gonopodio': 'gonopodium'}
        if any(k in old_parts for k in st.get('parts', {})):
            st['parts'] = {old_parts.get(k, k): v for k, v in st['parts'].items()}
        for k, v in data.get('settings', {}).items():
            if isinstance(v, dict) and isinstance(self.settings.get(k), dict):
                self.settings[k].update(v)
            else:
                self.settings[k] = v

    @classmethod
    def create(cls, path, name, folders=()):
        if os.path.exists(os.path.join(path, PROJECT_FILE)):
            raise FileExistsError(f'{path} already holds a project: open it instead of creating a new one.')
        os.makedirs(path, exist_ok=True)
        settings = copy.deepcopy(DEFAULT_SETTINGS)
        settings['colours_off'] = [c['key'] for c in profile.load(settings['profile'])['colour_classes']
                                   if c.get('optional')]
        data = {'name': name, 'created': datetime.date.today().isoformat(),
                'folders': [os.path.abspath(f) for f in folders], 'settings': settings}
        with open(os.path.join(path, PROJECT_FILE), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return cls(path)

    def save(self):
        data = {'name': self.name, 'created': self.created, 'folders': self.folders, 'settings': self.settings}
        tmp = os.path.join(self.path, PROJECT_FILE + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.path, PROJECT_FILE))

    def add_folder(self, folder):
        folder = os.path.abspath(folder)
        if folder not in self.folders:
            self.folders.append(folder)
            self.save()

    # --- folders inside the project ---
    def dir(self, name):
        d = os.path.join(self.path, name)
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def reference_csv(self):
        return os.path.join(self.path, 'colour_reference.csv')

    # --- photos and results ---
    def photos(self):
        """(session folder, photo path) for every image under every added folder, subfolders included.
        The session is the folder the photo sits in."""
        out, seen = [], set()
        for folder in self.folders:
            if os.path.isdir(folder):
                for p in metadata.list_photos(folder, skip=[self.path]):
                    if p not in seen:
                        seen.add(p)
                        out.append((os.path.dirname(p), p))
        return out

    @staticmethod
    def photo_id(photo):
        folder = os.path.basename(os.path.dirname(photo))
        stem = os.path.splitext(os.path.basename(photo))[0]
        h = hashlib.sha1(os.path.abspath(photo).encode('utf-8')).hexdigest()[:6]
        return f'{folder}__{stem}__{h}'

    def settings_hash(self):
        s = {k: self.settings[k] for k in ANALYSIS_KEYS}
        s['measures'] = MEASURES_VERSION
        if self.settings['outline'].get('model'):      # only then, so projects made before it stay current
            s['outline'] = self.settings['outline']
        if self.settings.get('colours_off'):
            s['colours_off'] = sorted(self.settings['colours_off'])
        return hashlib.sha1(json.dumps(s, sort_keys=True).encode('utf-8')).hexdigest()[:12]

    def result_path(self, photo):
        return os.path.join(self.dir('results'), self.photo_id(photo) + '.json')

    def load_result(self, photo):
        p = self.result_path(photo)
        if not os.path.exists(p):
            return None
        with open(p, encoding='utf-8') as f:
            return json.load(f)

    def save_result(self, photo, result):
        p = self.result_path(photo)
        tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=1, ensure_ascii=False)
        os.replace(tmp, p)

    def is_current(self, photo, result):
        # colours measured before 0.15 without a scale kept the head (eye counted as black): measured again once
        stale_head = bool(result) and 'colour_start_missing' in (result.get('flags') or '') \
            and not result.get('colour_start_rule')
        return (result is not None and result.get('settings_hash') == self.settings_hash()
                and result.get('source_mtime') == os.path.getmtime(photo) and not stale_head)

    def pending(self):
        """Photos never analysed, changed on disk, or analysed with different settings."""
        return [(f, p) for f, p in self.photos() if not self.is_current(p, self.load_result(p))]

    def results(self):
        out = []
        for _, p in self.photos():
            r = self.load_result(p)
            if r is not None:
                out.append(r)
        return out

    def set_label_id(self, photo, value):
        """Id typed by the user in Revisione: replaces the OCR reading and survives re-analysis."""
        r = self.load_result(photo)
        if r is not None:
            r['label_id'] = value
            r['label_manual'] = value
            r['label_needs_review'] = False
            r['flags'] = ';'.join(f for f in (r.get('flags') or '').split(';') if f and f != 'label_uncertain')
            self.save_result(photo, r)

    def set_analysis(self, photo, mode):
        """'size' or 'size_colour' chosen by hand in Review; wins over the settings and survives re-analysis."""
        r = self.load_result(photo)
        if r is not None:
            r['analysis'] = r['analysis_manual'] = mode
            if mode == 'size_colour' and self.settings.get('colours', {}).get('model') and not r.get('colours'):
                r['settings_hash'] = ''          # colours not measured yet: analysed again at the next Start
            self.save_result(photo, r)

    def set_status(self, photo, status):
        """Review outcome chosen by the user: 'reviewed' or 'excluded'. Survives re-analysis."""
        r = self.load_result(photo)
        if r is not None:
            r['status'] = status
            r['review'] = status
            self.save_result(photo, r)

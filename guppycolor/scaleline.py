"""Scale set by hand: a line drawn on a photo over a known length (ruler, graph paper, any object) and that
length in mm. scale/<photo id>.json holds the line of one photo; scale/folders.json = {folder: px/mm} gives the
scale to every photo of a folder whose own scale is missing or doubtful (same camera distance)."""
import datetime
import json
import math
import os

SCALE_FLAGS = {'scale_doubtful', 'scale_not_found', 'scale_from_median', 'scale_to_set'}
MIN_LINE_PX = 5


def path(project, photo):
    return os.path.join(project.dir('scale'), project.photo_id(photo) + '.json')


def folders_path(project):
    return os.path.join(project.dir('scale'), 'folders.json')


def _read(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _write(p, data):
    with open(p + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    os.replace(p + '.tmp', p)


def load_folders(project):
    return {str(k): float(v) for k, v in (_read(folders_path(project)) or {}).items()}


def photo_line(project, photo):
    """(p1, p2, mm) of the line drawn on this photo, or None."""
    d = _read(path(project, photo))
    return (d['p1'], d['p2'], float(d['mm'])) if d and d.get('p1') and d.get('p2') else None


def photo_scale(project, photo):
    """px/mm from the line drawn on this photo, or None."""
    d = _read(path(project, photo))
    return float(d['px_per_mm']) if d and d.get('px_per_mm') else None


def folder_scale(project, folder):
    return load_folders(project).get(folder)


def apply(project, r, ppm, source):
    """Put a hand-set scale in a saved result; a photo held only by its scale stops being 'to check'."""
    from .pipeline import SERIOUS
    r.update(px_per_mm=round(ppm, 4), scale_source=source, scale_ok=True)
    flags = [f for f in (r.get('flags') or '').split(';') if f and f not in SCALE_FLAGS]
    r['flags'] = ';'.join(flags)
    if r.get('status') == 'to_check' and not SERIOUS & set(flags):
        r['status'] = 'ok'
    project.save_result(r['path'], r)


def save(project, photo, p1, p2, mm, whole_folder=False):
    """Line from p1 to p2 (photo pixels) is mm long. Returns px/mm. whole_folder: also every photo of the same
    folder without a good scale of its own."""
    px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if not mm or mm <= 0 or px < MIN_LINE_PX:
        raise ValueError('draw a longer line and give its length in mm')
    ppm = px / float(mm)
    _write(path(project, photo), {'p1': [float(v) for v in p1], 'p2': [float(v) for v in p2], 'mm': float(mm),
                                  'px_per_mm': ppm, 'date': datetime.datetime.now().isoformat(timespec='minutes')})
    r = project.load_result(photo)
    if r is not None:
        apply(project, r, ppm, 'line')
    if whole_folder and r is not None:
        folders = load_folders(project)
        folders[r['folder']] = ppm
        _write(folders_path(project), folders)
        for o in project.results():
            if o['folder'] == r['folder'] and o['path'] != r['path'] and not os.path.exists(path(project, o['path'])) \
                    and (not o.get('scale_ok') or o.get('scale_source') == 'line_folder'):
                apply(project, o, ppm, 'line_folder')
    return ppm

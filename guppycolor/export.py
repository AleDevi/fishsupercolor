"""CSV export, one row per fish, ready for R (comma separator, decimal point, NA for missing values).
Here pixels become mm: a photo whose scale is doubtful takes the median scale of its session."""
import csv
import os
import statistics

FIXED_COLUMNS = [
    'file', 'folder', '{pattern}', 'date_taken', 'camera', 'exposure_s', 'iso', 'focal_mm',
    'colour_chart', 'correction', 'reference_photo', 'chart_error_de00',
    'px_per_mm', 'scale_source', 'parts_included',
    'standard_length_mm', 'landmarks_source', 'total_length_mm', 'total_length_source', 'mask_length_mm',
    'caudal_length_mm', 'max_depth_mm', 'depth_source', 'fish_area_mm2', 'body_area_mm2', 'body_area_source',
    'caudal_area_mm2', 'outline_source', '{colori}',
    'label_id', 'label_group', 'label_date', 'label_text', 'label_source',
    'label_confidence', 'label_needs_review', 'analysis',
    'fish_brightness', 'saturated_pct', 'sharpness', 'fish_confidence', 'status', 'flags', 'path',
]


def _num(v, nd=3):
    return 'NA' if v is None or v == '' else round(float(v), nd)


def session_scales(results):
    """Median px/mm of the photos with a good scale, per session folder."""
    by = {}
    for r in results:
        if r.get('scale_ok') and r.get('px_per_mm'):
            by.setdefault(r['folder'], []).append(r['px_per_mm'])
    return {k: statistics.median(v) for k, v in by.items()}


def colour_columns(project):
    """Per colour class: total area, % of the fish (reflections left out), number of spots, single spots."""
    if not project.settings.get('colours', {}).get('model'):
        return [], []
    from . import profile
    keys = [c['key'] for c in profile.load(project.settings['profile'])['colour_classes']]
    cols = []
    for k in keys:
        cols += [f'area_{k}_mm2', f'pct_{k}', f'n_spots_{k}', f'spots_{k}_mm2']
    return keys, cols + ['reflection_pct', 'colour_model', 'colour_confidence']


def _colour_values(r, keys, ppm):
    c = r.get('colours')
    out = {}
    if not c:
        return out
    fish = sum(v['area_px'] for k, v in c.items() if k != 'reflection') or None
    mm2 = (lambda px: px / ppm ** 2) if ppm else None
    for k in keys:
        v = c.get(k, {'area_px': 0, 'spots_px': []})
        spots = v.get('spots_px', [])
        out[f'area_{k}_mm2'] = _num(mm2(v['area_px']) if mm2 else None, 3)
        out[f'pct_{k}'] = _num(100 * v['area_px'] / fish if fish else None, 2)
        out[f'n_spots_{k}'] = len(spots)
        out[f'spots_{k}_mm2'] = ';'.join(f'{mm2(x):.3f}' for x in spots) if mm2 and spots else 'NA'
    refl = c.get('reflection', {}).get('area_px', 0)
    out['reflection_pct'] = _num(100 * refl / (refl + fish) if fish else None, 2)
    out['colour_model'] = r.get('colour_model') or 'NA'
    out['colour_confidence'] = _num(r.get('colour_confidence'), 3)
    return out


def rows(project):
    from .metadata import pattern_fields
    st = project.settings
    fields = pattern_fields(st.get('filename_pattern', ''))
    results = project.results()
    med = session_scales(results)
    parts = '+'.join(['body'] + [k for k, v in st['parts'].items() if v])
    ref = os.path.basename(st['colour'].get('reference_photo') or '') if st['colour']['mode'] in ('reference', 'shared') else ''
    keys, ccols = colour_columns(project)
    areas = {}
    out = []
    for r in results:
        flags = [f for f in (r.get('flags') or '').split(';') if f]
        ppm, source = r.get('px_per_mm'), r.get('scale_source', '')
        if not r.get('scale_ok') and st['scale']['mode'] == 'auto' and r['folder'] in med:
            ppm, source = med[r['folder']], 'session_median'
        mm = (lambda px: None if px is None or not ppm else px / ppm)
        mm2 = (lambda px: None if px is None or not ppm else px / ppm ** 2)
        row = {'file': r['file'], 'folder': r['folder']}
        row.update({k: r.get(k, '') or 'NA' for k in fields})
        row.update({
            'date_taken': r.get('date_taken') or 'NA', 'camera': r.get('camera') or 'NA',
            'exposure_s': _num(r.get('exposure_s'), 6), 'iso': _num(r.get('iso'), 0), 'focal_mm': _num(r.get('focal_mm'), 1),
            'colour_chart': 'TRUE' if r.get('colour_chart') else 'FALSE', 'correction': r.get('correction', 'none'),
            'reference_photo': ref or 'NA', 'chart_error_de00': _num(r.get('chart_error_de00'), 2),
            'px_per_mm': _num(ppm, 3), 'scale_source': source or 'NA', 'parts_included': parts,
            'standard_length_mm': _num(mm(r.get('standard_length_px')), 2),   # snout to end of the peduncle
            'landmarks_source': r.get('landmarks_source') or 'NA',
            'total_length_mm': _num(mm(r.get('total_length_px')), 2),      # NA when the tail is not all in the mask
            'total_length_source': r.get('total_length_source') or 'NA',     # outline, or snout -> tail tip landmarks
            'mask_length_mm': _num(mm(r.get('mask_length_px')), 2),
            'caudal_length_mm': _num(mm(r.get('caudal_length_px')), 2),    # peduncle end -> tail tip landmarks
            'max_depth_mm': _num(mm(r.get('max_depth_px')), 2),
            'depth_source': r.get('depth_source') or 'NA',                   # landmarks (gonopodium base) or outline
            'fish_area_mm2': _num(mm2(r.get('fish_area_px')), 2),
            'body_area_mm2': _num(mm2(r.get('body_area_px')), 2),
            'body_area_source': r.get('body_area_source') or 'NA',           # cut at the peduncle landmark or outline
            'caudal_area_mm2': _num(mm2(r.get('caudal_area_px')), 2),      # NA when the fin is not in the outline
            'outline_source': r.get('outline_source') or 'NA',              # detector, manual or outline_vN
            'fish_confidence': _num(r.get('fish_confidence'), 3),
            'fish_brightness': _num(r.get('fish_brightness'), 1), 'saturated_pct': _num(r.get('saturated_pct'), 2),
            'sharpness': _num(r.get('sharpness'), 2),
            'status': r.get('status', 'NA'), 'path': r.get('path', ''),
            'label_id': r.get('label_id') or 'NA', 'label_group': r.get('label_group') or 'NA',
            'label_date': r.get('label_date') or 'NA', 'label_text': r.get('label_text') or 'NA',
            'label_source': ('manual' if r.get('label_manual') else r.get('label_source')) or 'NA',
            'label_confidence': _num(r.get('label_confidence'), 3),
            'analysis': r.get('analysis') or 'NA',
            'label_needs_review': 'NA' if 'label_needs_review' not in r
                                        else ('TRUE' if r['label_needs_review'] else 'FALSE'),
        })
        row.update({c: 'NA' for c in ccols})
        if r.get('analysis') != 'size':
            row.update(_colour_values(r, keys, ppm))
        row['_flags'] = flags
        if row['fish_area_mm2'] != 'NA':
            areas.setdefault(r['folder'], []).append(row['fish_area_mm2'])
        out.append(row)
    sharp = {}
    for row in out:
        if row['sharpness'] != 'NA':
            sharp.setdefault(row['folder'], []).append(row['sharpness'])
    # area far from its session's median: probably a bad mask; sharpness well below it: blurred photo
    for row in out:
        sh, ref_sh = row['sharpness'], sharp.get(row['folder'])
        if sh != 'NA' and ref_sh and len(ref_sh) >= 3 and sh < 0.4 * statistics.median(ref_sh):
            row['_flags'].append('blurry')
        a, ref_a = row['fish_area_mm2'], areas.get(row['folder'])
        if a != 'NA' and ref_a and len(ref_a) >= 5:
            m = statistics.median(ref_a)
            if m and not 0.5 <= a / m <= 2:
                row['_flags'].append('area_outlier')
                if row['status'] == 'ok':
                    row['status'] = 'to_check'
        if row['scale_source'] == 'session_median':
            row['_flags'].append('scale_from_median')
        row['flags'] = ';'.join(row.pop('_flags')) or 'NA'
    return out, (fields, ccols)


def _safe(v):
    """Text that Excel would run as a formula (from OCR, file names, EXIF) gets a leading apostrophe."""
    return "'" + v if isinstance(v, str) and v[:1] in ('=', '+', '-', '@', '\t', '\r') else v


def write_csv(project, path=None):
    out, (fields, ccols) = rows(project)
    cols = []
    for c in FIXED_COLUMNS:
        cols += fields if c == '{pattern}' else ccols if c == '{colori}' else [c]
    cols = list(dict.fromkeys(cols))          # a file-name field may share a name with a fixed column 
    path = path or os.path.join(project.dir('export'), 'measurements.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows({k: _safe(v) for k, v in row.items()} for row in out)
    return path

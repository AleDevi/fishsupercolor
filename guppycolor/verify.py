"""Check the app against measures made by hand (one row per fish: total area per colour class in mm², total length, area).
Spots measured by hand cannot be matched one by one, so the comparison is on the per-class totals:
mean difference (bias), mean absolute error, correlation and Bland-Altman limits of agreement."""
import csv
import os
import re

import numpy as np

from . import export

GUESS = {'orange': ('orange', 'aranc', 'carot', 'ora'), 'black': ('black', 'nero', 'blk', 'mel'),
         'iridescent': ('irid', 'blu', 'struct')}
# sizes measured by hand: app column, header words (exact names for area, which every class column also contains)
SIZES = {'mask_length': ('mask_length_mm', ('totallength', 'total_length', 'total length', 'tl')),
         'fish_area': ('fish_area_mm2', ('area', 'area_mm2', 'fish area', 'fish_area', 'totalarea'))}


def app_column(key):
    return SIZES[key][0] if key in SIZES else f'area_{key}_mm2'


def read_table(path):
    """xlsx/csv/txt -> (header, rows as lists of str). xlsx: first sheet with a header row."""
    if path.lower().endswith(('.xlsx', '.xlsm')):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        table = [['' if v is None else str(v).strip() for v in r] for r in wb.worksheets[0].iter_rows(values_only=True)]
    else:
        with open(path, encoding='utf-8-sig', newline='') as f:
            text = f.read()
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=',;\t') if text.strip() else csv.excel
        table = [[c.strip() for c in r] for r in csv.reader(text.splitlines(), dialect)]
    table = [r for r in table if any(r)]
    return (table[0], table[1:]) if table else ([], [])


def _norm_id(v):
    return re.sub(r'\s+', '', str(v)).upper()


def app_ids(row):
    """Every id the app knows for a fish: label id, file-name id, file stem."""
    out = {_norm_id(row[k]) for k in ('label_id', 'id') if row.get(k) not in (None, '', 'NA')}
    out.add(_norm_id(os.path.splitext(row['file'])[0]))
    return out


def guess_columns(header, rows, project_rows, keys):
    """(id column, {class key: column}) guessed from names and from which column matches the app's ids."""
    known = set().union(*(app_ids(r) for r in project_rows)) if project_rows else set()
    best, id_col = 0, None
    for j, h in enumerate(header):
        hits = sum(_norm_id(r[j]) in known for r in rows if j < len(r))
        if hits > best:
            best, id_col = hits, h
    cols = {}
    for k, (_, names) in SIZES.items():
        for h in header:
            if h != id_col and h.lower().strip() in names:
                cols[k] = h
                break
    for k in keys:
        words = GUESS.get(k, (k[:4],))
        cands = [h for h in header if h != id_col and any(w in h.lower() for w in words)
                 and not any(x in h.lower() for x in ('perc', '%', 'pct'))]
        # a total column (OrangeT, black_tot, total iridescent) beats single-spot columns (Orange1, Orange2…)
        cands.sort(key=lambda h: (not re.search(r'(t|tot|total)$|^tot', h.lower().strip()), re.search(r'\d$', h) is not None))
        if cands:
            cols[k] = cands[0]
    return id_col, cols


def _float(v):
    try:
        return float(str(v).replace(',', '.'))
    except ValueError:
        return None


def compare(project, path, id_col, class_cols, out_dir=None):
    """Returns {class: stats} and writes validation_<model>.csv (+ .png with Bland-Altman plots)."""
    header, rows = read_table(path)
    app_rows, _ = export.rows(project)
    app_rows = [r for r in app_rows if r.get('status') != 'excluded']
    by_id = {}
    for r in app_rows:
        for i in app_ids(r):
            by_id.setdefault(i, r)
    ji = header.index(id_col)
    pairs = {k: [] for k in class_cols}
    for row in rows:
        if ji >= len(row) or not row[ji]:
            continue
        a = by_id.get(_norm_id(row[ji]))
        if a is None:
            continue
        for k, col in class_cols.items():
            j = header.index(col)
            manual = _float(row[j]) if j < len(row) else None
            auto = a.get(app_column(k))
            if manual is not None and isinstance(auto, (int, float)):
                pairs[k].append((row[ji], a['file'], manual, float(auto)))
    stats = {}
    for k, p in pairs.items():
        if len(p) < 2:
            stats[k] = {'n': len(p)}
            continue
        m, a = np.array([x[2] for x in p]), np.array([x[3] for x in p])
        d = a - m
        stats[k] = {'n': len(p), 'manual_mean': float(m.mean()), 'mean_difference': float(d.mean()),
                    'mean_abs_error': float(np.abs(d).mean()),
                    'relative_error_pct': float(100 * np.abs(d).sum() / max(m.sum(), 1e-12)),
                    'r': float(np.corrcoef(m, a)[0, 1]) if m.std() > 0 and a.std() > 0 else float('nan'),
                    'limits_of_agreement': (float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1)))}
    out_dir = out_dir or project.dir('export')
    name = project.settings['colours'].get('model') or 'modello'
    csv_path = os.path.join(out_dir, f'validation_{name}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'id', 'file', 'manual_area_mm2', 'app_area_mm2'])
        for k, p in pairs.items():
            for x in p:
                w.writerow([k, *x])
    png = os.path.join(out_dir, f'validation_{name}.png')
    plot(pairs, stats, png)
    return stats, csv_path, png


def plot(pairs, stats, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ks = [k for k in pairs if stats[k].get('n', 0) >= 2]
    if not ks:
        return None
    fig, ax = plt.subplots(2, len(ks), figsize=(4.2 * len(ks), 7.5), squeeze=False)
    for i, k in enumerate(ks):
        m, a = np.array([x[2] for x in pairs[k]]), np.array([x[3] for x in pairs[k]])
        s = stats[k]
        lim = [0, max(m.max(), a.max()) * 1.05]
        ax[0, i].scatter(m, a, s=18, color='#3D7FD6')
        ax[0, i].plot(lim, lim, color='#888', lw=1)
        u = 'mm' if k == 'mask_length' else 'mm²'
        ax[0, i].set(title=f"{k}  (n={s['n']}, r={s['r']:.2f})", xlabel=f'manual ({u})', ylabel=f'app ({u})', xlim=lim, ylim=lim)
        mean, d = (m + a) / 2, a - m
        ax[1, i].scatter(mean, d, s=18, color='#E8742A')
        for y, ls in ((s['mean_difference'], '-'), (s['limits_of_agreement'][0], '--'), (s['limits_of_agreement'][1], '--')):
            ax[1, i].axhline(y, color='#555', ls=ls, lw=1)
        ax[1, i].axhline(0, color='#bbb', lw=0.8)
        ax[1, i].set(title='Bland-Altman', xlabel=f'mean ({u})', ylabel=f'app − manual ({u})')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

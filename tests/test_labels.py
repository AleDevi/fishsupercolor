import os

import cv2
import numpy as np
import pytest

from guppycolor.engine import labels


def label_photo(path, lines=('G4 20/03', 'M116'), qr=None):
    """Graph paper with white paper labels written in thick dark-violet strokes, and a grey fish."""
    img = np.full((1100, 1600, 3), 235, np.uint8)
    for x in range(0, 1600, 20): cv2.line(img, (x, 0), (x, 1100), (190, 170, 210), 1)
    for y in range(0, 1100, 20): cv2.line(img, (0, y), (1600, y), (190, 170, 210), 1)
    for k, t in enumerate(lines):
        y = 120 + 150*k
        cv2.rectangle(img, (380, y - 110), (1120, y + 30), (250, 250, 250), -1)
        cv2.putText(img, t, (420, y), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (90, 30, 60), 14, cv2.LINE_AA)
    cv2.ellipse(img, (800, 800), (300, 70), 0, 0, 360, (120, 110, 95), -1)
    if qr:
        q = cv2.QRCodeEncoder.create().encode(qr)
        q = cv2.resize(q, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        img[750:750 + q.shape[0], 1250:1250 + q.shape[1]] = q[..., None]
    cv2.imwrite(path, img[:, :, ::-1])
    return path


def probs_for(text, alphabet, frames_per_char=3, p=0.9):
    """CTC-like probability matrix spelling text, with blanks between characters."""
    rows = []
    for ch in text:
        r = np.full(len(alphabet), (1 - p)/(len(alphabet) - 1)); r[alphabet.index(ch)] = p
        b = np.full(len(alphabet), (1 - p)/(len(alphabet) - 1)); b[0] = p
        rows += [r]*frames_per_char + [b]
    return np.array(rows)


ALPHA = ['blank'] + list('0123456789/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz|')


def test_pattern_fit_fixes_lookalikes():
    t, c = labels.ctc_fit(probs_for('nI5G', ALPHA), ALPHA, labels.pattern_slots(r'M\d{3}'))
    assert t == 'M156' and c > 0.3                      # n -> M, I -> 1, G -> 6
    t, c = labels.ctc_fit(probs_for('20103', ALPHA), ALPHA, labels.pattern_slots(r'\d{1,2}/\d{2}'))
    assert t == '20/03'                                  # the slash read as 1
    assert labels.ctc_fit(probs_for('M11', ALPHA), ALPHA, labels.pattern_slots(r'M\d{3}'))[1] < 0.2


def test_expected_id_beats_neighbour():
    P = probs_for('M127', ALPHA)
    assert labels.ctc_fit(P, ALPHA, labels.exact_slots('M127'))[1] > labels.ctc_fit(P, ALPHA, labels.exact_slots('M122'))[1]


def test_text_lines_found(tmp_path):
    rgb, f = labels.load(label_photo(str(tmp_path / 'a.jpg')))
    keep, rects = labels.text_lines(labels.ink_mask(rgb))
    rects = [r for r in rects if r[1][0]/r[1][1] > 1.5]
    assert len(rects) >= 2 and all(40 < r[0][1] < 400 for r in rects)    # the two label lines, not the fish


def test_qr_wins_without_ocr(tmp_path):
    r = labels.read_labels(label_photo(str(tmp_path / 'q.png'), qr='M116;G4;20/03'), expected_ids=['M116'])
    assert r['source'] == 'qr' and r['male_id'] == 'M116' and r['group'] == 'G4' and not r['needs_review']


def test_id_in_filename():
    assert labels.id_in_filename('x/Male 133_IMG_9670.JPG', r'M\d{3}') == 'M133'
    assert labels.id_in_filename('IMG_9670.JPG', r'M\d{3}') is None


def test_read_labels_ocr(tmp_path):
    pytest.importorskip('onnxruntime')
    try:
        labels._engines()
    except Exception as e:
        pytest.skip(f'OCR models not available: {e}')
    r = labels.read_labels(label_photo(str(tmp_path / 'o.jpg')), expected_ids=['M115', 'M116', 'M117'])
    assert r['male_id'] == 'M116' and r['source'] == 'ocr' and r['label_box'] is not None


def test_printed_label_sheet_reads_back(tmp_path):
    from guppycolor import labelsheet
    lst = tmp_path / 'pesci.csv'
    lst.write_text('male.id;group\nm116;4\nm117;4\nm116;4\n')
    rows = labelsheet.read_rows(str(lst), id_col='male.id', group_col='group', date='20/03')
    assert rows == [{'id': 'M116', 'group': 'G4', 'date': '20/03'}, {'id': 'M117', 'group': 'G4', 'date': '20/03'}]
    assert labelsheet.make_sheet(rows, str(tmp_path / 'e.pdf'), dpi=150) == 1
    lab = np.array(labelsheet.label_image(rows[0], dpi=20*25.4))          # 20 px/mm, as in a downscaled photo
    img = cv2.imread(label_photo(str(tmp_path / 'p.png'), lines=()))
    img[100:100 + lab.shape[0], 300:300 + lab.shape[1]] = lab[..., None]
    cv2.imwrite(str(tmp_path / 'p.png'), img)
    r = labels.read_labels(str(tmp_path / 'p.png'), expected_ids=['M116', 'M117'])
    assert (r['source'], r['male_id'], r['group'], r['date'], r['needs_review']) == ('qr', 'M116', 'G4', '20/03', False)
    assert r['label_box'] is not None


def test_id_table_and_sequence(tmp_path):
    from openpyxl import Workbook
    wb = Workbook(); ws = wb.active
    ws.append(['treatment', 'group', 'male.id', 'SL(mm)', 'SL_pictureID'])
    ws.append(['control', 1, 'm101', 16.9, 'IMG_0010'])
    ws.append(['control', 1, 'm102', 20.0, 'IMG_0013.JPG'])
    wb.save(tmp_path / 'dati.xlsx')
    table = labels.load_id_table(str(tmp_path / 'dati.xlsx'))
    assert table == {'IMG_0010': {'male_id': 'M101', 'group': 'G1'}, 'IMG_0013': {'male_id': 'M102', 'group': 'G1'}}
    # IMG_0011 = an extra shot of M101 (same image), IMG_0012 = of M102; no label readable without OCR
    a = label_photo(str(tmp_path / 'IMG_0010.png'), lines=())
    img = cv2.imread(a); img2 = img.copy(); cv2.circle(img2, (300, 300), 150, (20, 20, 20), -1)
    for n, im in (('IMG_0011', img), ('IMG_0012', img2), ('IMG_0013', img2)):
        cv2.imwrite(str(tmp_path / f'{n}.png'), im)
    photos = sorted(str(p) for p in tmp_path.glob('IMG_*.png'))
    labels._recs = []                                  # no OCR: decide on the images alone
    try:
        res = labels.fill_from_sequence(photos, table)
    finally:
        labels._recs = None
    got = {os.path.basename(p)[:8]: (r['male_id'], r['source']) for p, r in res.items()}
    assert got['IMG_0010'] == ('M101', 'tabella') and got['IMG_0013'] == ('M102', 'tabella')
    assert got['IMG_0011'] == ('M101', 'sequenza') and got['IMG_0012'] == ('M102', 'sequenza')

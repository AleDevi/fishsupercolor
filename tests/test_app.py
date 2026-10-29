import csv
import os

import numpy as np
import pytest

from conftest import PX_PER_MM, stub_segmenter, synthetic_photo
from guppycolor import export, metadata, pipeline
from guppycolor.project import Project


def test_parse_filename():
    assert metadata.parse_filename('x/M12_B2_2026-10-20.JPG', '{id}_{vasca}_{data}') == \
        {'id': 'M12', 'vasca': 'B2', 'data': '2026-10-20'}
    assert metadata.parse_filename('IMG_7220.JPG', '{id}_{vasca}_{data}') == {'id': '', 'vasca': '', 'data': ''}
    assert metadata.parse_filename('IMG_7220.JPG', '') == {}


def test_chart_found_and_corrected(tmp_path):
    from guppycolor.engine import colour
    img = pipeline.load_image(synthetic_photo(str(tmp_path / 'a.jpg')))
    chart = colour.find_chart(img)
    assert chart is not None and chart[0] < pipeline.CHART_MAX_ERR
    before = colour.delta_e(chart[1].reshape(-1, 3), colour.REF.reshape(-1, 3)).mean()
    fixed = colour.apply_tps(colour.fit_tps(chart[1]), img)
    after = colour.delta_e(colour.find_chart(fixed)[1].reshape(-1, 3), colour.REF.reshape(-1, 3)).mean()
    assert after < before / 2


def test_project_run_and_export(photos):
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['filename_pattern'] = '{id}_{vasca}_{data}'
    p.settings['colour']['mode'] = 'nominal'
    p.save()
    assert len(p.pending()) == 3
    pipeline.run(p, segmenter=stub_segmenter)
    assert p.pending() == []
    # a new session folder: only its photo is analysed
    p.add_folder(str(photos / 'sessione2'))
    assert [os.path.basename(x[1]) for x in p.pending()] == ['M9_B3_2026-10-21.jpg']
    pipeline.run(p, segmenter=stub_segmenter)
    with open(export.write_csv(p), encoding='utf-8') as f:
        rows = {r['file']: r for r in csv.DictReader(f)}
    r = rows['M1_B2_2026-10-20.jpg']
    assert (r['id'], r['vasca'], r['folder']) == ('M1', 'B2', 'sessione1')
    assert r['colour_chart'] == 'TRUE' and r['correction'] == 'nominal'
    assert abs(float(r['px_per_mm']) - PX_PER_MM) / PX_PER_MM < 0.03
    assert r['standard_length_mm'] == 'NA'
    # ellipse 520 px + tail ~ 690 px long -> about 34 mm along the mask
    assert 30 < float(r['mask_length_mm']) < 38
    s2 = rows['M9_B3_2026-10-21.jpg']
    assert s2['colour_chart'] == 'FALSE' and 'no_chart' in s2['flags']


def test_settings_change_makes_results_stale(photos):
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    pipeline.run(p, segmenter=stub_segmenter)
    assert p.pending() == []
    p.settings['parts']['gonopodium'] = True
    p.save()
    assert len(p.pending()) == 3


def test_review_decision_survives_reanalysis(photos):
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    pipeline.run(p, segmenter=stub_segmenter)
    photo = p.photos()[0][1]
    p.set_status(photo, 'excluded')
    p.settings['scale'] = {'mode': 'fixed', 'px_per_mm': 20}
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    assert p.load_result(photo)['status'] == 'excluded'


def test_reference_mode(photos):
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    ref_photo = p.photos()[0][1]
    assert pipeline.set_reference_photo(p, ref_photo)
    assert pipeline.load_reference(p.reference_csv).shape == (4, 6, 3)
    p.settings['colour']['mode'] = 'reference'
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    r = p.load_result(p.photos()[1][1])
    assert r['correction'] == 'reference' and r['chart_error_de00'] < 3


def test_missing_fish_is_an_error_not_a_crash(tmp_path):
    folder = tmp_path / 's'
    folder.mkdir()
    synthetic_photo(str(folder / 'a.jpg'))
    p = Project.create(str(tmp_path / 'pr'), 'x', [str(folder)])
    pipeline.run(p, segmenter=lambda img: (None, None, 0))
    r = p.results()[0]
    assert r['status'] == 'error' and 'fish_not_found' in r['flags']


def test_subfolders_are_sessions(tmp_path):
    root = tmp_path / 'Esempi foto Maschi'
    for sub in ('block 1', 'Foto MAschi'):
        (root / sub).mkdir(parents=True)
        synthetic_photo(str(root / sub / 'IMG_1.jpg'))
    p = Project.create(str(tmp_path / 'pr'), 'x', [str(root)])
    assert sorted(os.path.basename(f) for f, _ in p.photos()) == ['Foto MAschi', 'block 1']
    pipeline.run(p, segmenter=stub_segmenter)
    assert sorted(r['folder'] for r in p.results()) == ['Foto MAschi', 'block 1']
    assert len({r['photo_id'] for r in p.results()}) == 2          # same file name, two sessions


def test_manual_mask_replaces_detector(tmp_path):
    folder = tmp_path / 's'
    folder.mkdir()
    photo = synthetic_photo(str(folder / 'a.jpg'))
    p = Project.create(str(tmp_path / 'pr'), 'x', [str(folder)])
    pipeline.run(p, segmenter=lambda img: (None, None, 0))
    assert p.results()[0]['status'] == 'error'
    mask, _, _ = stub_segmenter(pipeline.load_image(photo))
    pipeline.save_manual_mask(p, photo, mask)
    r = pipeline.analyse(p, str(folder), photo, segmenter=lambda img: (None, None, 0))
    assert r['status'] == 'ok' and r['manual_mask']
    x0, y0, x1, y1 = r['check_box']                       # overlay is cropped around the fish
    assert x1 - x0 < r['width_px'] and y1 - y0 < r['height_px']


def test_mask_flags():
    import numpy as np
    m = np.zeros((400, 800), bool)
    m[150:250, 100:600] = True                                 # 5:1, one piece, inside
    morph = {'tl_px': 500, 'max_depth_px': 100}
    assert pipeline.mask_flags(m, morph) == []
    m2 = m.copy()
    m2[300:380, 650:780] = True                                # a second big piece
    assert 'mask_in_pieces' in pipeline.mask_flags(m2, morph)
    assert 'unusual_shape' in pipeline.mask_flags(m, {'tl_px': 150, 'max_depth_px': 100})
    m3 = m.copy()
    m3[150:250, 0:600] = True
    assert 'fish_at_border' in pipeline.mask_flags(m3, morph)


def fake_read_labels(path, expected_ids=None, patterns=None):
    uncertain = 'M2_' in path
    return {'text_raw': 'G4 20/03 M1I6' if uncertain else 'G4 | 20/03 | M116', 'male_id': 'M116',
            'group': 'G4', 'date': '20/03', 'confidence': 0.4 if uncertain else 0.95,
            'needs_review': uncertain, 'source': 'ocr', 'label_box': (10, 10, 100, 40),
            'id_candidates': [['M116', 0.4], ['M117', 0.3]] if uncertain else [['M116', 0.95]],
            'seen_expected': expected_ids}


def test_labels_in_pipeline_and_csv(photos, monkeypatch):
    from guppycolor.engine import labels
    seen = {}
    def spy(path, expected_ids=None, patterns=None):
        seen['ids'], seen['patterns'] = expected_ids, patterns
        return fake_read_labels(path, expected_ids, patterns)
    monkeypatch.setattr(labels, 'read_labels', spy)
    ids_file = photos / 'maschi.csv'
    ids_file.write_text('id;vasca\nM116;B2\nm117;B2\nfoo;bar\n')
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['labels'].update(id_source='label', expected_ids_file=str(ids_file))
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    assert seen['ids'] == ['M116', 'M117'] and seen['patterns']['male_id'] == r'M\d{3}'
    with open(export.write_csv(p), encoding='utf-8') as f:
        rows = {r['file']: r for r in csv.DictReader(f)}
    ok, unsure = rows['M1_B2_2026-10-20.jpg'], rows['M2_B2_2026-10-20.jpg']
    assert ok['label_id'] == 'M116' and ok['label_needs_review'] == 'FALSE' and ok['status'] == 'ok'
    assert unsure['status'] == 'to_check' and 'label_uncertain' in unsure['flags']
    # the id typed in Revisione wins, also after a re-analysis
    photo = [x for _, x in p.photos() if 'M2_' in x][0]
    p.set_label_id(photo, 'M117')
    p.settings['parts']['gonopodium'] = True
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    r = p.load_result(photo)
    assert r['label_id'] == 'M117' and r['status'] == 'ok'
    with open(export.write_csv(p), encoding='utf-8') as f:
        assert {r['file']: r for r in csv.DictReader(f)}['M2_B2_2026-10-20.jpg']['label_source'] == 'manual'


def test_expected_ids_from_excel(tmp_path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.append(['Gruppo', 'Maschio', 'Data'])
    ws.append(['G4', 'M116', '20/03'])
    ws.append(['G4', 'M009', None])
    wb.save(tmp_path / 'ids.xlsx')
    assert pipeline.load_expected_ids(str(tmp_path / 'ids.xlsx'), r'M\d{3}') == ['M009', 'M116']


def test_label_failure_does_not_stop_analysis(photos, monkeypatch):
    from guppycolor.engine import labels
    def boom(*a, **k):
        raise RuntimeError('ocr broken')
    monkeypatch.setattr(labels, 'read_labels', boom)
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['labels']['id_source'] = 'label'
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    r = p.results()[0]
    assert r['fish_area_px'] > 0 and 'label_uncertain' in r['flags']


def test_create_does_not_overwrite_a_project(tmp_path):
    Project.create(str(tmp_path / 'pr'), 'primo', [])
    with pytest.raises(FileExistsError):
        Project.create(str(tmp_path / 'pr'), 'secondo', [])
    assert Project(str(tmp_path / 'pr')).name == 'primo'


def test_csv_text_cannot_become_an_excel_formula(photos, monkeypatch):
    from guppycolor.engine import labels
    monkeypatch.setattr(labels, 'read_labels', lambda *a, **k: dict(fake_read_labels(*a, **k), text_raw='=HYPERLINK("x")'))
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['labels']['id_source'] = 'label'
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    with open(export.write_csv(p), encoding='utf-8') as f:
        r = next(csv.DictReader(f))
    assert r['label_text'] == '\'=HYPERLINK("x")' and float(r['px_per_mm']) > 0


def test_weights_with_wrong_hash_are_refused(tmp_path):
    from guppycolor.engine import segment
    for name in segment.WEIGHTS:
        (tmp_path / name).write_bytes(b'not a model')
    with pytest.raises(segment.ModelError):
        segment.ensure_weights(str(tmp_path))


def test_ids_from_data_table(photos, monkeypatch):
    from guppycolor.engine import labels
    seen = {}
    monkeypatch.setattr(labels, 'load_id_table', lambda path, id_pattern=None: {'M1_B2_2026-10-20': {'male_id': 'M101'}})
    def seq(ph, table, expected_ids=None, patterns=None):
        seen.setdefault('calls', []).append(sorted(os.path.basename(x) for x in ph))
        return {x: {'male_id': 'M101', 'source': 'tabella' if 'M1_' in x else 'sequenza', 'confidence': 1.0,
                    'needs_review': 'M3_' in x} for x in ph}
    monkeypatch.setattr(labels, 'fill_from_sequence', seq)
    monkeypatch.setattr(labels, 'read_labels', lambda *a, **k: pytest.fail('per-photo reading not expected'))
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['labels'].update(id_source='table', id_table_file='dati.xlsx')
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    assert len(seen['calls']) == 1 and len(seen['calls'][0]) == 3        # once per folder, with all its photos
    with open(export.write_csv(p), encoding='utf-8') as f:
        rows = {r['file']: r for r in csv.DictReader(f)}
    assert rows['M1_B2_2026-10-20.jpg']['label_source'] == 'tabella'
    assert rows['M3_B2_2026-10-20.jpg']['status'] == 'to_check'


def test_names_from_a_project_file_stay_inside_their_folders(tmp_path):
    from guppycolor import profile, training
    with pytest.raises(ValueError):
        profile.load('../../secret')
    p = Project.create(str(tmp_path / 'pr'), 'x', [])
    with pytest.raises(ValueError):
        training.load_model(p, '../elsewhere/model')


def test_label_date_against_exif_date():
    assert pipeline.date_mismatch('20/03', '2024-03-21 10:00:00')
    assert not pipeline.date_mismatch('21/3', '2024-03-21 10:00:00')
    assert not pipeline.date_mismatch('', '2024-03-21 10:00:00') and not pipeline.date_mismatch('21/03', '')


def test_id_from_file_name(photos):
    p = Project.create(str(photos / 'progetto'), 'Prova', [str(photos / 'sessione1')])
    p.settings['filename_pattern'] = '{id}_{vasca}_{data}'
    p.settings['labels']['id_source'] = 'filename'
    p.save()
    pipeline.run(p, segmenter=stub_segmenter)
    with open(export.write_csv(p), encoding='utf-8') as f:
        r = {r['file']: r for r in csv.DictReader(f)}['M1_B2_2026-10-20.jpg']
    assert r['label_id'] == 'M1' and r['label_source'] == 'filename'
    assert r['total_length_mm'] == 'NA' or float(r['total_length_mm']) > 0


def test_session_check_flags_a_scale_far_from_the_others(tmp_path):
    from guppycolor import pipeline
    from guppycolor.project import Project
    folder = tmp_path / 'f'
    folder.mkdir()
    p = Project.create(str(tmp_path / 'p'), 'P', [str(folder)])
    for i, ppm in enumerate((20.0, 20.4, 19.8, 25.0)):
        photo = str(folder / f'{i}.jpg')
        open(photo, 'wb').close()
        p.save_result(photo, {'path': photo, 'folder': str(folder), 'file': f'{i}.jpg', 'status': 'ok', 'flags': '',
                              'px_per_mm': ppm, 'standard_length_px': 15 * 20.0})
    rs = {r['file']: r for r in pipeline.session_checks(p)}
    assert rs['3.jpg']['status'] == 'to_check' and 'scale_off_session' in rs['3.jpg']['flags']
    assert 'size_off_session' in rs['3.jpg']['flags']           # 15 mm at 20 px/mm looks 12 mm at 25
    assert all(rs[f]['status'] == 'ok' and not rs[f]['flags'] for f in ('0.jpg', '1.jpg', '2.jpg'))
    p.settings['checks']['session_pct'] = 30
    rs = {r['file']: r for r in pipeline.session_checks(p)}
    assert rs['3.jpg']['status'] == 'ok' and not rs['3.jpg']['flags']
    # a scale drawn by hand on the odd photo: no scale flag any more, and the fish length is right again
    p.settings['checks']['session_pct'] = 10
    from guppycolor import scaleline
    scaleline.save(p, str(folder / '3.jpg'), (0, 0), (400, 0), 20)
    rs = {r['file']: r for r in pipeline.session_checks(p)}
    assert rs['3.jpg']['scale_source'] == 'line' and not rs['3.jpg']['flags'] and rs['3.jpg']['status'] == 'ok'
    assert scaleline.photo_line(p, str(folder / '3.jpg')) == ([0.0, 0.0], [400.0, 0.0], 20.0)


def test_landmark_outside_the_outline_grows_it():
    """A tail tip clicked beyond SAM's outline (transparent fin left out) is used as a click that adds the fin."""
    import numpy as np
    from guppycolor import pipeline
    img = np.zeros((200, 300, 3), np.uint8)
    body = np.zeros((200, 300), bool)
    body[80:120, 50:200] = True
    fin = np.zeros_like(body)
    fin[85:115, 200:250] = True
    calls = []

    def grow(img, mask, x, y):
        calls.append((x, y))
        return mask | fin if fin[y, x] else None

    pts = {'snout': (51.0, 100.0), 'eye': (70.0, 95.0), 'tail_tip': (245.0, 100.0)}
    mask, used = pipeline.guide_outline(img, body, pts, grow=grow)
    assert used == ['tail_tip'] and calls == [(245, 100)]
    assert mask[100, 240] and mask.sum() == body.sum() + fin.sum()
    # a landmark just off the edge (a click on the outline) changes nothing
    mask, used = pipeline.guide_outline(img, body, {'snout': (48.0, 100.0)}, grow=grow)
    assert used == [] and mask is body


def test_gpu_lock_differs_from_the_processor_lock_only_in_torch():
    """requirements-win-cuda.txt (installed with an NVIDIA card) must pin the same packages as requirements-win.txt,
    with only torch and torchvision swapped for their CUDA builds."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def pins(name):
        with open(os.path.join(root, name), encoding='utf-8') as f:
            return dict(re.findall(r'^([A-Za-z0-9_.-]+)==(\S+)', f.read(), re.M))
    cpu, gpu = pins('requirements-win.txt'), pins('requirements-win-cuda.txt')
    assert set(cpu) == set(gpu)
    assert {k for k in cpu if cpu[k] != gpu[k]} == {'torch', 'torchvision'}
    assert all(gpu[k] == cpu[k] + '+cu130' for k in ('torch', 'torchvision'))


def test_neural_networks_fall_back_to_the_processor():
    from guppycolor.engine import segment
    segment._device = None
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if not has_gpu:
        assert segment.device() == 'cpu' and 'processor' in segment.device_text()


def test_colours_that_kept_the_head_are_measured_again(tmp_path):
    """Photos measured before 0.15 without a scale kept the head in the colours: they become pending once."""
    from guppycolor.project import Project
    folder = tmp_path / 'f'
    folder.mkdir()
    photo = str(folder / 'a.jpg')
    open(photo, 'wb').close()
    p = Project.create(str(tmp_path / 'p'), 'P', [str(folder)])
    r = {'path': photo, 'folder': str(folder), 'file': 'a.jpg', 'settings_hash': p.settings_hash(),
         'source_mtime': os.path.getmtime(photo), 'flags': 'colour_start_missing'}
    p.save_result(photo, r)
    assert p.pending()
    p.save_result(photo, dict(r, colour_start_rule=2))
    assert not p.pending()

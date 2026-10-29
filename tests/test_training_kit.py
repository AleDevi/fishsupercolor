"""Training kit (training_kit/gctrain): reading the van der Bijl annotations and placing them on a photo."""
import io
import json
import os
import sys
import zipfile

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'training_kit'))
from gctrain import export, prepare, vdb  # noqa: E402

LANDMARKS = """<shapes type=list>
\t<scaling type=logical names=FALSE length=1 as.numeric=FALSE >NA</scaling>
\t<landmarks.pixel type=matrix rownames=TRUE colnames=FALSE nrow=4 ncol=2 as.numeric=TRUE >
\t\t1\t91\t136
\t\t2\t2\t128
\t\t3\t567\t90
\t\t4\t552\t194
\t</landmarks.pixel>
</shapes>
"""


def test_parse_landmarks():
    p = vdb.parse_landmarks(LANDMARKS)
    assert p.shape == (4, 2)
    assert p[1].tolist() == [2, 128] and p[3].tolist() == [552, 194]


def test_uid_of():
    assert vdb.uid_of('photos/gen_2/color_corrected/20201214-IMG_5421.JPG') == '20201214_IMG_5421'


def test_split_and_tail_tip():
    mask = np.zeros((100, 300), bool)
    mask[40:60, 10:290] = True
    body, caudal = vdb.split_at_line(mask, np.array([200, 30.]), np.array([200, 70.]), np.array([10, 50.]))
    assert body.sum() + caudal.sum() == mask.sum()
    assert body[:, :200].sum() == mask[:, :200].sum() and not caudal[:, :199].any()
    tip = vdb.tail_tip(caudal, np.array([10, 50.]), np.array([200, 50.]))
    assert abs(tip[0] - 289) < 1 and abs(tip[1] - 49.5) < 1


def textured_fish(rng):
    """An RGBA crop with enough texture for SIFT, and a photo that holds it scaled and rotated."""
    crop = np.zeros((300, 800, 4), np.uint8)
    cv2.ellipse(crop, (400, 150), (380, 110), 0, 0, 360, (255, 255, 255, 255), -1)
    tex = np.full((300, 800, 3), 128, np.uint8)
    for _ in range(400):            # scattered high-contrast blobs: corners SIFT finds at half scale
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y, r = int(rng.integers(0, 800)), int(rng.integers(0, 300)), int(rng.integers(4, 14))
        if rng.random() < 0.5:
            cv2.circle(tex, (x, y), r, c, -1)
        else:
            cv2.rectangle(tex, (x, y), (x + r, y + int(rng.integers(4, 14))), c, -1)
    crop[..., :3] = np.where(crop[..., 3:] > 0, tex, 0)
    M = cv2.getRotationMatrix2D((0, 0), 5, 1.6)
    M[:, 2] = [700, 500]
    photo = np.full((1400, 2200, 3), 200, np.uint8)
    warped = cv2.warpAffine(crop, M, (2200, 1400))
    a = warped[..., 3:] / 255.0
    photo = (photo * (1 - a) + warped[..., :3] * a).astype(np.uint8)
    return crop, photo, M


def test_register_finds_the_crop():
    crop, photo, M = textured_fish(np.random.default_rng(1))
    found, stats = vdb.register(crop, photo)
    assert found is not None, stats
    assert abs(stats['scale'] - 1.6) < 0.03 and abs(stats['angle'] + 5) < 1
    assert np.abs(vdb.transform([[400, 150]], found) - vdb.transform([[400, 150]], M)).max() < 3


def make_annotations(folder, uid, crop):
    def png(img):
        return cv2.imencode('.png', img)[1].tobytes()
    orange = np.zeros_like(crop)
    orange[120:180, 300:360] = crop[120:180, 300:360]
    for name, data, ext in (('extracted_fish', png(crop), 'png'), ('carotenoid_coloration', png(orange), 'png'),
                            ('melanic_coloration_v2', png(np.zeros_like(crop)), 'png'),
                            ('fixed_landmarks', LANDMARKS.encode(), 'txt')):
        with zipfile.ZipFile(os.path.join(folder, name + '.zip'), 'w') as z:
            z.writestr(f'{name}/r/g/{uid}.{ext}', data)


def test_annotate_places_masks_and_points(tmp_path):
    crop, photo, M = textured_fish(np.random.default_rng(2))
    make_annotations(tmp_path, '20200101_IMG_1', crop)
    ann = vdb.Annotations(str(tmp_path))
    a, stats = vdb.annotate(ann, '20200101_IMG_1', photo)
    assert a is not None, stats
    assert a['body'].sum() + a['caudal'].sum() == a['fish'].sum()
    expected = vdb.transform([[2, 128]], M)[0]
    assert np.hypot(*(np.array(a['points']['snout']) - expected)) < 4
    top, bottom = a['points']['caudal_base_top'], a['points']['caudal_base_bottom']
    assert top[1] < bottom[1] and np.allclose(np.add(top, bottom) / 2, a['points']['peduncle_end'])
    assert 0.9 < a['orange'].sum() / (60 * 60 * 1.6 ** 2) < 1.1


def test_split_by_fish_is_stable():
    assert prepare.split_of('m1-01') == prepare.split_of('m1-01')
    splits = [prepare.split_of(f'm{i}') for i in range(2000)]
    assert 0.05 < splits.count('test') / 2000 < 0.15 and 0.05 < splits.count('val') / 2000 < 0.15


def test_package_layout_matches_the_app(tmp_path):
    w = tmp_path / 'best.pt'
    w.write_bytes(b'weights')
    dest = export.package(str(tmp_path / 'm.zip'), 'landmarks', 'landmarks_v1', 'guppy_male', str(w), {'keys': ['snout']})
    from guppycolor import modelshare
    manifest, data = modelshare.read_package(dest)
    assert manifest['kind'] == 'landmarks' and json.loads(data['model.json'])['keys'] == ['snout']


def test_no_amp_check_download():
    from gctrain import train
    from ultralytics.engine import trainer
    try:
        train._yolo('yolo11n-pose.yaml')
    except Exception:
        pass
    assert trainer.check_amp is train._amp_ok


def test_find_annotations_in_a_subfolder(tmp_path):
    sub = tmp_path / 'van der Bijl_annotations'
    sub.mkdir()
    (tmp_path / 'foto_2000').mkdir()
    assert vdb.find_annotations(str(tmp_path)) is None
    for n in ['photo_database.csv'] + list(vdb.ZIPS.values()):
        (sub / n).write_bytes(b'')
    assert vdb.find_annotations(str(tmp_path)) == str(sub)


def test_colour_labels_use_the_profile_brushes():
    from gctrain import colours
    index = colours.brush_index('guppy_male')
    assert (index['orange'], index['black'], index['base']) == (1, 2, 4)
    fish = np.zeros((10, 10), bool); fish[2:8, 2:8] = True
    orange = np.zeros_like(fish); orange[2:5, 2:5] = True
    black = np.zeros_like(fish); black[4:6, 4:6] = True
    lab = colours.labels({'fish': fish, 'orange': orange, 'black': black}, index)
    assert lab[4, 4] == 0 and lab[2, 2] == 1 and lab[5, 5] == 2 and lab[7, 7] == 4 and lab[0, 0] == 0

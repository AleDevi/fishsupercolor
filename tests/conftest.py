import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PX_PER_MM = 20


def synthetic_photo(path, cast=(0.85, 1.0, 1.15), chart=True, fish_scale=1.0, shift=0):
    """Graph paper (1 mm = 20 px), a grey fish with a tail, optionally a ColorChecker with a colour cast."""
    from guppycolor.engine import colour
    h, w = 1100, 1600
    img = np.full((h, w, 3), 235, np.uint8)
    for x in range(0, w, PX_PER_MM):
        cv2.line(img, (x, 0), (x, h), (190, 170, 210), 1)
    for y in range(0, h, PX_PER_MM):
        cv2.line(img, (0, y), (w, y), (190, 170, 210), 1)
    cx, cy = 900 + shift, 650
    a, b = int(260 * fish_scale), int(70 * fish_scale)
    cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, (120, 110, 95), -1)
    tail = np.array([[cx + a - 20, cy], [cx + a + int(150 * fish_scale), cy - int(90 * fish_scale)],
                     [cx + a + int(150 * fish_scale), cy + int(90 * fish_scale)]], np.int32)
    cv2.fillPoly(img, [tail], (125, 115, 100))
    cv2.circle(img, (cx - a + int(45 * fish_scale), cy - 10), int(12 * fish_scale), (10, 10, 10), -1)   # eye
    if chart:
        x0, y0, s, g = 60, 60, 60, 10
        cv2.rectangle(img, (x0 - g, y0 - g), (x0 + 6 * (s + g), y0 + 4 * (s + g)), (20, 20, 20), -1)
        for r in range(4):
            for c in range(6):
                rgb = np.clip(colour.REF[r, c] * np.array(cast), 0, 1) * 255
                cv2.rectangle(img, (x0 + c * (s + g), y0 + r * (s + g)), (x0 + c * (s + g) + s, y0 + r * (s + g) + s),
                              tuple(int(v) for v in rgb), -1)
    Image.fromarray(img).save(path, quality=95)
    return path


def stub_segmenter(img):
    """Stands in for YOLOE+SAM: the fish is the only mid-grey, low-saturation area."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 2] < 150) & (hsv[..., 2] > 70) & (hsv[..., 1] < 80)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None, None, 0
    i = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    mask = lab == i
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8)).astype(bool)
    return mask, None, 0.9


@pytest.fixture
def photos(tmp_path):
    s1 = tmp_path / 'sessione1'
    s1.mkdir()
    for i in range(3):
        synthetic_photo(str(s1 / f'M{i + 1}_B2_2026-10-20.jpg'), shift=i * 20)
    s2 = tmp_path / 'sessione2'
    s2.mkdir()
    synthetic_photo(str(s2 / 'M9_B3_2026-10-21.jpg'), chart=False)
    return tmp_path

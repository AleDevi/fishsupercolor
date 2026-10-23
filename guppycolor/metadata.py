"""Photo metadata: fields parsed from the file name with a user pattern, plus EXIF."""
import os
import re

from PIL import Image, ExifTags

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def pattern_fields(pattern):
    return re.findall(r'\{(\w+)\}', pattern)


def pattern_regex(pattern):
    """'{id}_{vasca}_{data}' -> regex with one named group per field; separators are kept literally."""
    out, pos = '', 0
    for m in re.finditer(r'\{(\w+)\}', pattern):
        out += re.escape(pattern[pos:m.start()]) + f'(?P<{m.group(1)}>.+?)'
        pos = m.end()
    return re.compile('^' + out + re.escape(pattern[pos:]) + '$')


def parse_filename(filename, pattern):
    """Fields from the file name (without extension). Missing match -> empty fields, never an error."""
    fields = pattern_fields(pattern)
    if not fields:
        return {}
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = pattern_regex(pattern).match(stem)
    return {k: (m.group(k) if m else '') for k in fields}


def read_exif(path):
    out = {'date_taken': '', 'camera': '', 'exposure_s': '', 'iso': '', 'focal_mm': ''}
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
            sub = {ExifTags.TAGS.get(k, k): v for k, v in ex.get_ifd(0x8769).items()}
    except Exception:
        return out
    tags.update(sub)
    dt = tags.get('DateTimeOriginal') or tags.get('DateTime') or ''
    if dt:
        out['date_taken'] = str(dt).replace(':', '-', 2)
    out['camera'] = ' '.join(str(tags.get(k, '')).strip() for k in ('Make', 'Model')).strip()
    for src, dst in (('ExposureTime', 'exposure_s'), ('ISOSpeedRatings', 'iso'), ('FocalLength', 'focal_mm')):
        v = tags.get(src)
        if v is not None:
            try:
                out[dst] = float(v)
            except (TypeError, ValueError):
                out[dst] = str(v)
    return out


def list_photos(folder, skip=()):
    """Images in folder and all its subfolders (each subfolder is a session). `skip`: folders to ignore."""
    skip = {os.path.abspath(d) for d in skip}
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and os.path.abspath(os.path.join(root, d)) not in skip)
        out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXT) and not f.startswith('.')]
    return out

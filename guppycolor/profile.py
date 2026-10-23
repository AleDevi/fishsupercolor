"""Species profile: body parts, colour classes and named landmarks. Everything species-specific lives here."""
import json
import os
import re

SAFE_NAME = re.compile(r'^[A-Za-z0-9_-]+$')     # names read from a project file never reach outside their folder

PROFILES_DIR = os.path.join(os.path.dirname(__file__), 'profiles')


def available():
    return sorted(f[:-5] for f in os.listdir(PROFILES_DIR) if f.endswith('.json'))


def load(name):
    if not SAFE_NAME.fullmatch(name or ''):
        raise ValueError(f'invalid profile name: {name!r}')
    path = os.path.join(PROFILES_DIR, name + '.json')
    with open(path, encoding='utf-8') as f:
        p = json.load(f)
    for key in ('parts', 'colour_classes', 'landmarks'):
        p.setdefault(key, [])
    return p


def procrustes_landmarks(profile):
    """Landmarks used for the alignment: required ones only, so every fish has them."""
    return [l['key'] for l in profile['landmarks'] if l.get('procrustes') and l.get('required')]

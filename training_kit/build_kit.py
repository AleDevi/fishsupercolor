"""python training_kit/build_kit.py OUT.zip: the kit as one zip (GuppyTraining\\...): the kit files, Guppy Colour
from the current git commit in app\\, and an empty data\\ folder with a note on what goes in it."""
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT_FILES = ['install_training.bat', 'start_training.bat', 'quick_test.bat', 'requirements-train-cuda-win.txt',
             'README.md']
APP_PATHS = ['guppycolor', 'pyproject.toml', 'requirements-win.txt', 'LICENSE']
DATA_NOTE = """Put here the van der Bijl et al. 2025 files (FRDR doi 10.20383/103.01281), without unzipping them:
  photo_database.csv
  extracted_fish.zip
  carotenoid_coloration.zip
  melanic_coloration_v2.zip
  fixed_landmarks.zip
and the folder foto_2000 with the photos (any subfolders).
Optional: a folder other_photos with photos of other setups, only drawn in the report.
"""


def git_files(paths):
    out = subprocess.run(['git', '-C', ROOT, 'ls-files', '-z', '--', *paths], check=True, capture_output=True).stdout
    return [p for p in out.decode().split('\0') if p]


def git_show(path):
    data = subprocess.run(['git', '-C', ROOT, 'show', f'HEAD:{path}'], check=True, capture_output=True).stdout
    if path.endswith('.bat'):                      # cmd.exe needs Windows line ends
        data = data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    return data


def main(dest):
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in git_files(['training_kit']):
            rel = os.path.relpath(p, 'training_kit')
            if rel in KIT_FILES or rel.startswith('gctrain/'):
                z.writestr('GuppyTraining/' + rel, git_show(p))
        for p in git_files(APP_PATHS):
            z.writestr('GuppyTraining/app/' + p, git_show(p))
        z.writestr('GuppyTraining/data/READ_ME.txt', DATA_NOTE.replace('\n', '\r\n'))
    return dest


if __name__ == '__main__':
    print(main(sys.argv[1]))

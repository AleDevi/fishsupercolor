"""python -m gctrain DATA [--quick]: prepare -> train outline -> train landmarks -> check -> model files.

DATA is the folder with the van der Bijl annotation zips + photo_database.csv and a folder foto_2000 with the
photos (any subfolders). Work files go to DATA/work, results next to DATA (KIT\results). Every step that finished is skipped
when the command runs again, so an interrupted run continues where it stopped."""
import argparse
import datetime
import json
import os
import shutil
import sys

from guppycolor.engine import segment


def main(argv=None):
    ap = argparse.ArgumentParser(prog='gctrain')
    ap.add_argument('data')
    ap.add_argument('--photos', help='folder with the photos (default: DATA/foto_2000)')
    ap.add_argument('--extra', action='append', default=[], help='folders of other photos, only drawn in the report')
    ap.add_argument('--quick', action='store_true', help='a few minutes, only to check that everything runs')
    a = ap.parse_args(argv)
    segment.configure_ultralytics()
    from . import evaluate, export, prepare, train
    data = os.path.abspath(a.data)
    work = os.path.join(data, 'work_quick' if a.quick else 'work')
    results = os.path.join(os.path.dirname(data), 'results_quick' if a.quick else 'results')
    os.makedirs(work, exist_ok=True)
    logf = open(os.path.join(data, 'training_log.txt'), 'a', encoding='utf-8')

    def log(msg):
        line = f'{datetime.datetime.now():%H:%M:%S} {msg}'
        print(line, flush=True)
        logf.write(line + '\n'); logf.flush()

    photos = a.photos or os.path.join(data, 'foto_2000')
    if not os.path.isdir(photos):
        log(f'Photo folder not found: {photos}'); return 1
    extra = a.extra or [d for d in (os.path.join(data, 'other_photos'),) if os.path.isdir(d)]
    if not os.path.exists(os.path.join(work, 'photos.csv')):
        log('Step 1 of 4: preparing the photos')
        prepare.run(data, photos, work, log=log, limit=60 if a.quick else None)
    else:
        log('Step 1 of 4: photos already prepared')
    log('Step 2 of 4: outline model')
    train.train_outline(work, log, a.quick)
    log('Step 3 of 4: landmark model')
    train.train_landmarks(work, log, a.quick)
    log('Step 4 of 4: checking the models on the test photos')
    summary = evaluate.run(work, extra, log)
    os.makedirs(results, exist_ok=True)
    files = export.run(work, results, summary)
    for name in ('report.html', 'test_results.csv', 'photos.csv', 'summary.json'):
        shutil.copyfile(os.path.join(work, name), os.path.join(results, name))
    log('Done. In the folder "results": ' + ', '.join(os.path.basename(f) for f in files) +
        ', report.html. Bring the folder "results" back to the PC with Guppy Colour.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

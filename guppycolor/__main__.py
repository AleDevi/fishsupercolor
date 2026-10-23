"""python -m guppycolor            -> opens the window
python -m guppycolor run PROJECT -> analyses pending photos and writes export/measurements.csv (no window)"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog='guppycolor')
    sub = ap.add_subparsers(dest='cmd')
    r = sub.add_parser('run', help='analyse the new photos of a project and write the CSV')
    r.add_argument('project')
    n = sub.add_parser('new', help='create a project')
    n.add_argument('project')
    n.add_argument('--name', required=True)
    n.add_argument('folders', nargs='*')
    a = ap.parse_args(argv)
    from .engine.segment import ultralytics_env
    ultralytics_env()
    if a.cmd is None:
        from .gui.main import launch
        return launch()
    from .project import Project
    from . import pipeline, export
    if a.cmd == 'new':
        Project.create(a.project, a.name, a.folders)
        print('Project created:', a.project)
        return 0
    p = Project(a.project)
    def progress(i, n, res):
        print(f'[{i}/{n}] {res.get("file")}: {res.get("status")} {res.get("flags", "")}', flush=True)
    pipeline.run(p, progress=progress)
    print('CSV:', export.write_csv(p))
    return 0


if __name__ == '__main__':
    sys.exit(main())

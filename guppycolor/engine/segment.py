"""Zero-training fish segmentation: YOLOE (text prompt 'fish') -> box -> SAM 2.1 mask."""
import hashlib
import os
import urllib.request

import numpy as np, cv2

# Weights folder: GUPPYCOLOR_MODELS, else ~/.guppycolor/models. We download the weights ourselves and check their
# SHA-256 before ultralytics ever unpickles them (.pt files can run code when loaded).
W = os.environ.get('GUPPYCOLOR_MODELS', os.path.join(os.path.expanduser('~'), '.guppycolor', 'models'))
ASSETS = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/'
WEIGHTS = {
    'yoloe-11l-seg.pt': 'a993fb0fc7c8830939ae14e6434a925dd1179428158c2761482eb8a8d8a3699f',
    'sam2.1_b.pt': 'f1a9cf2dd69d84bb463b5ad98246d03e2d47a130a9295db0ec967e6cd95e2e47',
    'mobileclip_blt.ts': 'a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54',   # YOLOE text encoder
}


class ModelError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ensure_weights(folder=W, progress=None):
    """Download missing weights into `folder` and verify all of them. Raises ModelError on a bad file."""
    for name, digest in WEIGHTS.items():
        ensure_file(name, digest, folder, progress)
    return folder


def ensure_file(name, digest, folder=W, progress=None):
    """One release asset of ultralytics, downloaded once and checked against its SHA-256."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        if progress:
            progress(f'Downloading {name} (once)')
        tmp = path + '.part'
        with urllib.request.urlopen(ASSETS + name, timeout=60) as r, open(tmp, 'wb') as f:
            for block in iter(lambda: r.read(1 << 20), b''):
                f.write(block)
        if sha256(tmp) != digest:
            os.remove(tmp)
            raise ModelError(f'{name}: the downloaded file is not the expected one, it will not be used.')
        os.replace(tmp, path)
    elif sha256(path) != digest:
        raise ModelError(f'{path} is not the expected file: delete it and start again, it will be downloaded again.')
    return path


def ultralytics_env(folder=W):
    """No telemetry, no runtime pip installs, no network checks, weights-only loading. Ultralytics reads these
    variables when first imported, so this runs at start-up (before anything imports it)."""
    os.environ.setdefault('YOLO_CONFIG_DIR', os.path.join(os.path.dirname(folder), 'ultralytics'))
    os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
    os.environ.setdefault('YOLO_OFFLINE', 'True')
    os.environ.setdefault('ULTRALYTICS_SAFE_LOAD', '1')      # .pt loaded with weights_only=True


def configure_ultralytics(folder=W):
    ultralytics_env(folder)
    from ultralytics import settings
    settings.update({'sync': False, 'weights_dir': folder})   # no analytics events; weights only from `folder`
    local_fonts()


def local_fonts():
    """Training checks for Arial.ttf and downloads it when missing (it does not see Windows' arial.ttf), which
    fails offline. The font only labels plots: matplotlib's own DejaVuSans is put in its place, no download."""
    import shutil
    import matplotlib
    from ultralytics.utils import USER_CONFIG_DIR
    src = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
    for name in ('Arial.ttf', 'Arial.Unicode.ttf'):
        dest = os.path.join(USER_CONFIG_DIR, name)
        if not os.path.exists(dest) and os.path.exists(src):
            os.makedirs(USER_CONFIG_DIR, exist_ok=True)
            shutil.copyfile(src, dest)


_det = None; _sam = None
def models(sam='sam2.1_b.pt'):
    global _det, _sam
    if _det is None:
        ensure_weights()
        configure_ultralytics()
        from ultralytics import YOLOE, SAM
        _det = YOLOE(os.path.join(W, 'yoloe-11l-seg.pt')); n = ['fish']; _det.set_classes(n, _det.get_text_pe(n))
        _sam = SAM(os.path.join(W, sam))
    return _det, _sam

def segment_fish(img, work=1600):
    det, sam = models()
    h, w = img.shape[:2]; s = work / max(h, w)
    small = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    r = det.predict(small[..., ::-1], conf=0.02, verbose=False)[0]
    if len(r.boxes) == 0: return None, None, 0
    i = int(r.boxes.conf.argmax()); box = r.boxes.xyxy[i].cpu().numpy(); conf = float(r.boxes.conf[i])
    m = sam.predict(small[..., ::-1], bboxes=[box.tolist()], verbose=False)[0].masks.data[0].cpu().numpy()
    m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m, box / s, conf

def segment_fish_click(img, x, y, work=1600):
    """One positive click on the fish (full-res pixel coordinates) -> SAM 2.1 mask."""
    _, sam = models()
    h, w = img.shape[:2]; s = work / max(h, w)
    small = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    m = sam.predict(small[..., ::-1], points=[[x*s, y*s]], labels=[1], verbose=False)[0].masks.data[0].cpu().numpy()
    return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

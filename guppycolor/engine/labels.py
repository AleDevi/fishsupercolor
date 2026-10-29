"""Text written in the photo: QR code first, else handwritten/printed labels via OCR.
Labels are found as ink strokes (pixels much darker than the local paper, grid lines and
marker glitter removed), grouped into text lines, deskewed and read with the PP-OCR
recogniser shipped with rapidocr_onnxruntime (offline, no torch). Instead of trusting the
free reading, each line is scored against the expected formats (male id M + 3 digits, group
G + digits, date dd/mm) by a constrained CTC decode: the network's per-frame character
probabilities are aligned to the format, so a handwritten 'M' read as 'n' or a '/' read as
'1' still fits. With a list of expected ids, every id is scored the same way and the best wins."""
import math, os, re
import numpy as np, cv2
from PIL import Image, ImageOps

PATTERNS = {'male_id': r'M\d{3}', 'group': r'G\d{1,2}', 'date': r'\d{1,2}/\d{2}'}
FIELD_MIN = 0.2            # a field read with less confidence than this is left empty
REVIEW_BELOW = 0.45         # under this -> needs_review (on 57 test photos every id above it was right)
MAX_SIDE = 1600

MODELS = os.environ.get('GUPPY_OCR_MODELS', '')    # optional folders with rec.onnx (+ dict.txt), os.pathsep-separated
W = os.path.join(os.environ.get('GUPPYCOLOR_MODELS', os.path.join(os.path.expanduser('~'), '.guppycolor', 'models')), 'ocr')
# PP-OCRv6 small recogniser (Apache-2.0), taken from the onnxocr wheel on PyPI: the package itself is not
# installed because it pulls opencv-python-headless, which clashes with the app's opencv-python.
V6_WHEEL = 'https://files.pythonhosted.org/packages/py3/o/onnxocr/onnxocr-4.0.0-py3-none-any.whl'
V6_FILES = {'onnxocr/models/ppocrv6/small/rec/rec.onnx': 'rec.onnx', 'onnxocr/models/ppocrv6/ppocrv6_dict.txt': 'dict.txt'}

V6_SHA256 = '71c570a347ec688cd47dbc400c77e9abcbd3f48e497519234e0f5ce575c5845b'    # onnxocr-4.0.0 wheel, as listed on PyPI

def ensure_v6(folder=os.path.join(W, 'ppocrv6_small')):
    """Download once (~75 MB wheel, 21 MB kept). Returns the folder, or None when offline or the wheel is not the expected one."""
    if all(os.path.exists(os.path.join(folder, f)) for f in V6_FILES.values()): return folder
    import hashlib, io, urllib.request, zipfile
    try:
        data = urllib.request.urlopen(V6_WHEEL, timeout=120).read()
    except OSError:
        return None
    if hashlib.sha256(data).hexdigest() != V6_SHA256: return None
    os.makedirs(folder, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for src, dst in V6_FILES.items():      # temp name + replace: a half-written folder is never taken as complete
            tmp = os.path.join(folder, dst + '.part')
            with open(tmp, 'wb') as fh: fh.write(z.read(src))
            os.replace(tmp, os.path.join(folder, dst))
    return folder

class _Rec:
    """PP-OCR text recogniser (CTC) run with onnxruntime: crop -> per-frame character probabilities."""
    def __init__(self, onnx, dict_path=None):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx, providers=['CPUExecutionProvider'])
        meta = self.sess.get_modelmeta().custom_metadata_map
        chars = meta['character'].splitlines() if 'character' in meta else \
            open(dict_path, encoding='utf-8').read().splitlines()
        self.alphabet = ['blank'] + chars + [' ']
        self.ascii = np.array([i == 0 or (len(c) == 1 and 32 <= ord(c) < 127) for i, c in enumerate(self.alphabet)],
                              np.float32)          # labels are Latin: drop the Chinese characters

    def __call__(self, bgr):
        h = 48; w = max(16, int(round(h*bgr.shape[1]/bgr.shape[0])))
        x = cv2.resize(bgr, (w, h)).astype(np.float32).transpose(2, 0, 1)/127.5 - 1
        P = self.sess.run(None, {self.sess.get_inputs()[0].name: x[None]})[0][0]
        P = P[:, :len(self.alphabet)]*self.ascii[:P.shape[1]]
        return P/np.maximum(P.sum(1, keepdims=True), 1e-9)

def _find_models():
    """Recognisers to combine: folders in GUPPY_OCR_MODELS, else PP-OCRv6 small + rapidocr's own model
    (the two disagree on different photos: together they read more ids correctly than either alone)."""
    dirs = [d for d in MODELS.split(os.pathsep) if d] if MODELS else [ensure_v6()]
    cands = [(os.path.join(d, 'rec.onnx'), os.path.join(d, 'dict.txt')) for d in dirs if d]
    if MODELS: return [(o, d if os.path.exists(d) else None) for o, d in cands]
    try:
        import importlib.util
        spec = importlib.util.find_spec('rapidocr_onnxruntime')
        if spec:
            import glob
            p = sorted(glob.glob(os.path.join(os.path.dirname(spec.origin), 'models', '*rec*.onnx')))
            if p: cands.append((p[-1], None))
    except ImportError:
        pass
    found = [(o, d if d and os.path.exists(d) else None) for o, d in cands if os.path.exists(o)]
    if not found: raise RuntimeError('no OCR model found: pip install onnxocr rapidocr_onnxruntime')
    return found

_recs = None
def _engines():
    global _recs
    if _recs is None: _recs = [_Rec(*m) for m in _find_models()]   # lazy: the app starts without OCR installed
    return _recs

# ---------- image -> text-line crops

def load(path, max_side=MAX_SIDE):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    f = min(1.0, max_side/max(im.size))
    if f < 1: im = im.resize((round(im.width*f), round(im.height*f)), Image.LANCZOS)
    return np.array(im), f

def ink_mask(rgb, ratio=0.5):
    mn = rgb.min(axis=2).astype(np.float32)
    bg = cv2.GaussianBlur(cv2.dilate(mn, np.ones((61, 61), np.uint8)), (0, 0), 15)   # local paper
    ink = (((bg - mn)/np.maximum(bg, 1)) > ratio).astype(np.uint8)*255   # relative: robust to vignetting
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))       # fill marker glitter
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def text_lines(ink, min_h=0.02, max_h=0.2):
    """Character-sized blobs joined horizontally into lines -> rotated rects, top to bottom."""
    H, W = ink.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink)
    ok = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        ok[i] = min_h*H <= h <= max_h*H and w <= 0.3*W and x > 0 and y > 0 and x+w < W and y+h < H
    keep = np.where(ok[lab], 255, 0).astype(np.uint8)
    k = max(3, int(0.035*W))
    joined = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(3, k//4))))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(joined)
    rects = []
    for i in range(1, n2):
        ys, xs = np.nonzero((lab2 == i) & (keep > 0))
        if len(xs) < 50: continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        if w < h: w, h, ang = h, w, ang + 90
        if h < min_h*H: continue
        rects.append(((cx, cy), (w, h), ang))
    rects.sort(key=lambda r: (r[0][1], r[0][0]))
    return keep, rects

def crop(img, rect, pad=0.15):
    (cx, cy), (w, h), ang = rect
    w, h = w*(1 + pad), h*(1 + 2*pad)
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[:, 2] += (w/2 - cx, h/2 - cy)
    return cv2.warpAffine(img, M, (max(1, int(w)), max(1, int(h))), borderMode=cv2.BORDER_REPLICATE)

def rect_box(rect, f):
    """Axis-aligned (x, y, w, h) in the original image's pixels."""
    pts = cv2.boxPoints(rect)/f
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

# ---------- recogniser -> per-frame probabilities

def char_probs(crops):
    """For each crop (BGR): [(per-frame probabilities, alphabet)] from every recogniser."""
    return [[(rec(c), rec.alphabet) for rec in _engines()] for c in crops]

def free_decode(P, alphabet):
    idx = P.argmax(1); txt, ps = [], []
    for t, i in enumerate(idx):
        if i == 0 or (t and idx[t-1] == i): continue
        txt.append(alphabet[i]); ps.append(P[t, i])
    return ''.join(txt), float(np.exp(np.mean(np.log(ps)))) if ps else 0.0

# ---------- constrained CTC decoding

LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
SEPS = '/1lI|\\()-.:'          # a handwritten slash is often read as 1, l or I
LOOKS_LIKE = {'M': 'mNnHhWwKkrtTX', 'G': 'gCcEeQ',
              '1': 'Iil|!', '0': 'OoQD', '2': 'Zz', '4': 'Ah', '5': 'Ss', '6': 'Gb', '7': 'T', '8': 'B', '9': 'gq'}
DIGITS = '0123456789' + ''.join(v for k, v in LOOKS_LIKE.items() if k.isdigit())
AS_DIGIT = {c: d for d, cs in LOOKS_LIKE.items() if d.isdigit() for c in cs}

def pattern_slots(pat):
    """'M\\d{3}' -> list of variants, each a list of (allowed chars, output char or None)."""
    toks = re.findall(r'\\d(?:\{(\d+)(?:,(\d+))?\})?|(.)', pat)
    slots = [[]]
    for a, b, lit in toks:
        if lit:
            if lit.isalpha(): s = (lit + LOOKS_LIKE.get(lit.upper(), ''), lit.upper())   # output the literal
            elif lit == '/': s = (SEPS, '/')
            else: s = (lit, lit)
            slots = [v + [s] for v in slots]
        else:
            lo = int(a or 1); hi = int(b or lo)
            slots = [v + [(DIGITS, None)]*k for v in slots for k in range(lo, hi + 1)]
    return slots

def exact_slots(s):
    return [[(ch + LOOKS_LIKE.get(ch, ''), ch) if ch.isalpha() else (SEPS, '/') if ch == '/' else (ch + LOOKS_LIKE.get(ch, ''), ch) for ch in s]]

def ctc_fit(P, alphabet, slots, wild=math.log(20)):
    """Best CTC path through P spelling one of the slot sequences, with free (wildcard) frames allowed
    before and after it (other text on the same line). Returns (text, confidence): confidence =
    geometric mean of the chosen characters' peak probs x the probability the constraint keeps
    relative to the free reading, per character."""
    T = len(P); lP = np.log(np.maximum(P, 1e-9)); lfree = lP.max(1)
    lw = np.maximum(lP[:, 0], lfree - wild)                     # wildcard frame: blank, or anything / 20
    lut = {ch: i for i, ch in enumerate(alphabet)}
    NEG = -1e18
    best = (None, 0.0)
    for slot in slots:
        cls = [np.array(sorted({lut[c] for c in chars if c in lut})) for chars, _ in slot]
        L = len(cls)
        if any(len(c) == 0 for c in cls) or 2*L - 1 > T: continue
        # Viterbi over: lead, one state per (slot, character), blank after each slot, tail
        lead = lw[0]; ch = [np.full(len(c), NEG) for c in cls]; bl = np.full(L, NEG); tail = NEG
        ch[0] = lP[0, cls[0]].copy()
        bp_ch = [np.zeros((T, len(c)), np.int16) for c in cls]   # -1 stay, -2 lead/blank, k>=0 prev char k
        bp_bl = np.zeros((T, L), np.int8); bp_tail = np.zeros(T, np.int8)   # bl: 0 stay, 1 from char; tail: 0 stay, 1 char, 2 blank
        bl_from = [np.zeros(T, np.int16) for _ in range(L)]; tail_ks = np.zeros(T, np.int16)
        for t in range(1, T):
            new = []
            for j in range(L):
                stay = ch[j]
                enter = np.full(len(cls[j]), lead if j == 0 else bl[j-1])
                src = np.where(stay >= enter, -1, -2); v = np.maximum(stay, enter)
                if j:                                  # straight from the previous slot's character, if different
                    pv = ch[j-1]; o = np.argsort(pv)[::-1]
                    k1 = o[0]; k2 = o[1] if len(o) > 1 else None
                    same = cls[j] == cls[j-1][k1]
                    cand = np.where(same, pv[k2] if k2 is not None else NEG, pv[k1])
                    kk = np.where(same, k2 if k2 is not None else 0, k1)
                    better = cand > v; v = np.where(better, cand, v); src = np.where(better, kk, src)
                bp_ch[j][t] = src; new.append(v + lP[t, cls[j]])
            nbl = np.empty(L)
            for j in range(L):
                k = int(np.argmax(ch[j])); fromc = ch[j][k]
                if bl[j] >= fromc: nbl[j] = bl[j]; bp_bl[t, j] = 0
                else: nbl[j] = fromc; bp_bl[t, j] = 1; bl_from[j][t] = k
                nbl[j] += lP[t, 0]
            kL = int(np.argmax(ch[L-1])); opts = [tail, ch[L-1][kL], bl[L-1]]
            bp_tail[t] = int(np.argmax(opts)); ntail = max(opts) + lw[t]
            lead = lead + lw[t]; ch = new; bl = nbl; tail = ntail
            tail_ks[t] = kL
        kL = int(np.argmax(ch[L-1]))
        ends = [ch[L-1][kL], bl[L-1], tail]; e = int(np.argmax(ends)); v = ends[e]
        if v < NEG/2: continue
        # backtrack
        state = ('c', L-1, kL) if e == 0 else ('b', L-1) if e == 1 else ('t',)
        pick = {}
        for t in range(T-1, -1, -1):
            if state[0] == 'c':
                j, k = state[1], state[2]
                p = P[t, cls[j][k]]
                if j not in pick or p > pick[j][1]: pick[j] = (cls[j][k], p)
                if t == 0: break
                s = bp_ch[j][t, k]
                if s == -1: state = ('c', j, k)
                elif s == -2: state = ('l',) if j == 0 else ('b', j-1)
                else: state = ('c', j-1, int(s))
            elif state[0] == 'b':
                j = state[1]
                state = ('b', j) if bp_bl[t, j] == 0 else ('c', j, int(bl_from[j][t]))
            elif state[0] == 't':
                w = bp_tail[t]
                state = ('t',) if w == 0 else ('c', L-1, int(tail_ks[t])) if w == 1 else ('b', L-1)
            else:
                break
        if len(pick) < L: continue
        txt = ''.join(out or AS_DIGIT.get(alphabet[pick[j][0]], alphabet[pick[j][0]]) for j, (_, out) in enumerate(slot))
        keep = min(1.0, math.exp((v - lfree.sum())/L))
        conf = keep*math.exp(np.mean([math.log(pick[j][1]) for j in range(L)]))
        if conf > best[1]: best = (txt, conf)
    return best

# ---------- QR and file name

def read_qr(rgb):
    """[(text, (x, y, w, h))] for every decoded QR code."""
    try:
        ok, texts, pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    except cv2.error:
        return []
    if not ok: return []
    return [(t, cv2.boundingRect(p.astype(np.float32))) for t, p in zip(texts, pts) if t]

def id_in_filename(path, pat):
    m = re.search(pat.replace('M', '[Mm](?:ale)?[ _-]?', 1) if pat.startswith('M') else pat, os.path.basename(path))
    if not m: return None
    digits = re.sub(r'\D', '', m.group(0))
    return 'M' + digits if pat.startswith('M') else m.group(0)

def _norm(s):
    return re.sub(r'[^0-9A-Za-z/]', '', s).upper()

# ---------- main entry

def read_labels(image_path, expected_ids=None, patterns=None, id_table=None):
    """Text in one photo. Returns text_raw, male_id, group, date, confidence (of male_id),
    needs_review, source ('qr'|'tabella'|'ocr'|'filename'|'none'), label_box (x, y, w, h) or None.
    id_table: {photo name without extension, upper case: {'male_id': .., 'group': ..}} from load_id_table();
    a photo listed there takes its id from the table (the label is still read, to catch mismatches)."""
    pats = dict(PATTERNS, **(patterns or {}))
    res = dict(text_raw='', male_id=None, group=None, date=None, confidence=0.0,
               needs_review=True, source='none', label_box=None, id_candidates=[])
    rgb, f = load(image_path)
    expected = [_norm(e) for e in expected_ids] if expected_ids else None

    qr, fq = read_qr(rgb), f
    if not qr and f < 1:                       # a small printed QR may need the full resolution
        qr, fq = read_qr(load(image_path, max_side=10**5)[0]), 1.0
    if qr:
        x, y, w, h = qr[0][1]
        qr = [t for t, _ in qr]
        res.update(text_raw=' | '.join(qr), source='qr', confidence=1.0,
                   label_box=(int(x/fq), int(y/fq), int(w/fq), int(h/fq)))
        for key, pat in pats.items():
            m = next((re.search(pat, q, re.I) for q in qr if re.search(pat, q, re.I)), None)
            if m: res[key] = m.group(0).upper()
        res['needs_review'] = res['male_id'] is None or (expected is not None and _norm(res['male_id']) not in expected)
        return res

    keep, rects = text_lines(ink_mask(rgb))
    rects = [r for r in rects if 0.6 <= r[1][0]/r[1][1] <= 12][:12]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    clean = cv2.cvtColor(255 - keep, cv2.COLOR_GRAY2BGR)
    crops, owner = [], []
    for i, r in enumerate(rects):
        for src in (bgr, clean):
            c = crop(src, r)
            if min(c.shape[:2]) < 8: continue
            crops += [c, cv2.rotate(c, cv2.ROTATE_180)]; owner += [(i, 0), (i, 1)]
    lines, used, cands = {}, {}, []
    if crops:
        per = {}                     # (line, flip) -> [(P, alphabet, free text, conf)]: original + clean x models
        for outs, o in zip(char_probs(crops), owner):
            for P, al in outs: per.setdefault(o, []).append((P, al, *free_decode(P, al)))
        for i in range(len(rects)):              # both orientations kept; upright preferred for text_raw
            lines[i] = [per[o] for o in ((i, 0), (i, 1)) if o in per]
        def fit_lines(slots_by_text, skip=()):
            """[(conf, text, line)]: conf = mean over original/clean crop and models, best orientation."""
            out = []
            for i, flips in lines.items():
                if i in skip: continue
                score = {}
                for vs in flips:
                    sc = {}
                    for name, slots in slots_by_text:
                        for P, al, _, _ in vs:
                            t, c = ctc_fit(P, al, slots)
                            if t: sc[name or t] = sc.get(name or t, 0) + c/len(vs)
                    for t, c in sc.items(): score[t] = max(score.get(t, 0), c)
                out += [(c, t, i) for t, c in score.items()]
            return sorted(out, reverse=True)
        def pick(cand):
            c, t, i = cand[0]
            alt = next((x[0] for x in cand if x[1] != t), 0.0)
            if alt > 0.5*c: c *= 1 - (alt - 0.5*c)/c   # a close runner-up lowers the confidence
            return c, t, i
        fits = {key: fit_lines([(None, pattern_slots(pat))]) for key, pat in pats.items()}
        cands += fits.get('male_id') or []
        taken = set()
        for key in sorted(fits, key=lambda k: -(fits[k][0][0] if fits[k] else 0)):   # surest field first
            cand = [x for x in fits[key] if x[2] not in taken]
            if not cand: continue
            c, t, i = pick(cand)
            if c < FIELD_MIN and not (key == 'male_id' and expected): continue
            res[key] = t; taken.add(i); used[key] = i
            if key == 'male_id': res['confidence'] = round(float(c), 3)
        if expected and (res['male_id'] not in expected or res['confidence'] < REVIEW_BELOW):
            # snap to the list: score every expected id on the lines not used by group/date
            other = {i for k, i in used.items() if k != 'male_id'}
            cand = fit_lines([(e, exact_slots(e)) for e in expected], skip=other)
            cands += cand
            if cand:
                c, t, i = pick(cand)
                if c > res['confidence'] or res['male_id'] not in expected:
                    res.update(male_id=t, confidence=round(float(c), 3)); used['male_id'] = i
    seen = {}
    for c, t, _ in sorted(cands, reverse=True):          # the other readings of the id, for Review to offer
        if t not in seen and len(seen) < 4:
            seen[t] = round(float(c), 3)
    res['id_candidates'] = [[t, c] for t, c in seen.items()]
    def best_text(flips):
        v = max(flips, key=lambda vs: np.mean([x[3] for x in vs]) + (0.15 if vs is flips[0] else 0))
        return max(v, key=lambda x: x[3])[2:]
    res['text_raw'] = ' | '.join(t for t, c in (best_text(lines[i]) for i in sorted(lines)) if c > 0.3)
    if used:
        xs = [rect_box(rects[i], f) for i in used.values()]
        x0 = min(b[0] for b in xs); y0 = min(b[1] for b in xs)
        res['label_box'] = (x0, y0, max(b[0] + b[2] for b in xs) - x0, max(b[1] + b[3] for b in xs) - y0)
    if res['male_id'] or res['group'] or res['date']: res['source'] = 'ocr'

    fid = id_in_filename(image_path, pats['male_id'])
    if fid and fid != res['male_id']:
        if res['male_id'] is None or res['confidence'] < REVIEW_BELOW:
            res.update(male_id=fid, source='filename', confidence=0.9)
        else:
            res['confidence'] = min(res['confidence'], 0.3)       # label and file name disagree
    row = (id_table or {}).get(photo_key(image_path))
    if row:
        read, conf = res['male_id'], res['confidence']
        res.update(male_id=row['male_id'], source='tabella', confidence=1.0)
        if row.get('group'): res['group'] = row['group']
        if read and conf >= REVIEW_BELOW and _norm(read) != _norm(row['male_id']):
            res['confidence'] = 0.3            # the label clearly says something else: let a person look
    res['needs_review'] = (res['male_id'] is None or res['confidence'] < REVIEW_BELOW
                           or (expected is not None and _norm(res['male_id']) not in expected))
    return res

# ---------- table photo -> id (data sheets of past experiments)

def photo_key(path_or_name):
    return os.path.splitext(os.path.basename(str(path_or_name).strip()))[0].upper()

def load_id_table(path, id_pattern=PATTERNS['male_id']):
    """{photo key: {'male_id', 'group'}} from an Excel/CSV where a row holds a fish id and the name(s) of its
    photo(s), e.g. the SL_pictureID column. Columns are found by their content, on every sheet:
    the id column = cells matching id_pattern, photo columns = cells like IMG_9805 / IMG_9805.JPG,
    group column = header containing 'group'/'gruppo'."""
    import csv
    if path.lower().endswith(('.xlsx', '.xlsm')):
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        sheets = [[['' if v is None else str(v).strip() for v in r] for r in ws.iter_rows(values_only=True)]
                  for ws in wb.worksheets]
    else:
        text = open(path, encoding='utf-8-sig', errors='replace').read()
        sheets = [list(csv.reader(text.splitlines(), delimiter=';' if text.count(';') > text.count(',') else ','))]
    rid = re.compile(id_pattern, re.I)
    rph = re.compile(r'(?i)^[A-Z]{2,5}[_ -]?\d{3,}(?:\.(?:jpe?g|tiff?|png|cr2|nef|dng))?$')
    table = {}
    for rows in sheets:
        if not rows: continue
        n = max(len(r) for r in rows)
        col = lambda j: [r[j] for r in rows if j < len(r) and r[j]]
        def share(j, rx):
            v = col(j); return sum(bool(rx.fullmatch(x)) for x in v)/max(1, len(v))
        ids = [j for j in range(n) if share(j, rid) > 0.5]
        photos = [j for j in range(n) if share(j, rph) > 0.3]
        if not ids or not photos: continue
        ji = ids[0]
        head = next((r for r in rows if any(c and not rid.fullmatch(c) for c in r)), [])
        jg = next((j for j, h in enumerate(head) if re.search(r'(?i)group|gruppo', h)), None)
        for r in rows:
            if ji >= len(r) or not rid.fullmatch(r[ji]): continue
            g = r[jg] if jg is not None and jg < len(r) else ''
            if re.fullmatch(r'\d+', g): g = 'G' + g
            for j in photos:
                if j < len(r) and rph.fullmatch(r[j]):
                    table.setdefault(photo_key(r[j]), {'male_id': r[ji].upper(), 'group': g.upper()})
    return table

# ---------- photos of the same fish in a row (past experiments)

SEQ_OCR_MIN = 0.2      # below this the label reading is too weak to choose between the two neighbours

def _taken_at(path):
    try:
        e = Image.open(path).getexif()
        return e.get_ifd(0x8769).get(36867) or e.get(306) or ''
    except Exception:
        return ''

def _thumb(path):
    im = Image.open(path); im.draft('L', (400, 300))
    a = np.asarray(ImageOps.exif_transpose(im).convert('L').resize((96, 64)), np.float32)
    a -= a.mean()
    return a/(np.linalg.norm(a) + 1e-9)

def fill_from_sequence(photos, id_table, expected_ids=None, patterns=None):
    """Ids for one folder of photos where only some are in id_table (e.g. one 'SL' photo per fish in the
    data sheet, the others are extra shots of the same fish taken just before or after it).
    A photo between table photos of fish A and B is either A or B: the label is read with only those two
    as candidates and the photo is compared with both neighbours (same fish, same label, same pose ->
    nearly the same image). Both agree -> accepted; otherwise the more reliable cue is proposed and the
    photo goes to review. Returns {path: read_labels-style dict}, source 'tabella' or 'sequenza'."""
    order = sorted(photos, key=lambda p: (_taken_at(p), photo_key(p)))
    ids = [(id_table.get(photo_key(p)) or {}).get('male_id') for p in order]
    thumbs, out = {}, {}
    th = lambda p: thumbs.setdefault(p, _thumb(p))
    for k, p in enumerate(order):
        if ids[k]:
            out[p] = read_labels(p, expected_ids, patterns, id_table)
            continue
        prev = next(((order[j], ids[j]) for j in range(k - 1, -1, -1) if ids[j]), None)
        nxt = next(((order[j], ids[j]) for j in range(k + 1, len(order)) if ids[j]), None)
        cands = {c[1]: c[0] for c in (nxt, prev) if c}       # fish id -> its table photo (the earlier one if both)
        if not cands:
            out[p] = read_labels(p, expected_ids, patterns)
            continue
        r = read_labels(p, sorted(cands), patterns)
        sims = {i: float((th(p)*th(q)).sum()) for i, q in cands.items()}
        by_image = max(sims, key=sims.get)
        if r['male_id'] in cands and r['confidence'] >= SEQ_OCR_MIN:
            agree = r['male_id'] == by_image or len(cands) == 1
            r.update(source='sequenza', confidence=0.95 if agree else round(r['confidence']*0.5, 3), needs_review=not agree)
        else:
            r.update(male_id=by_image, source='sequenza', confidence=0.3, needs_review=True)
        row = id_table.get(photo_key(cands[r['male_id']]), {})
        if row.get('group'): r['group'] = row['group']
        out[p] = r
    return out

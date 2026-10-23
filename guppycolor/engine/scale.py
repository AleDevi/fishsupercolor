"""Pixels per mm from the graph paper around the fish (smallest grid square = 1 mm).
Uses the 2-D autocorrelation of the paper texture: a grid repeats at its true spacing, so the
autocorrelation peaks sit on the grid lattice (harmonics of thin lines do not create peaks at
half the spacing). The nearest strong lattice peak = the smallest square; a far peak
(n squares away) gives sub-pixel precision."""
import numpy as np, cv2
from scipy.ndimage import maximum_filter

MIN_SQUARES = 20       # the spacing must be measured over at least 20 squares (2 cm) of paper

def _ac(a, m):
    H, W = a.shape; S = (2*H, 2*W)
    fa = np.fft.rfft2(a*m, s=S); fm = np.fft.rfft2(m, s=S)
    num = np.fft.irfft2(fa*np.conj(fa), s=S); den = np.fft.irfft2(fm*np.conj(fm), s=S)
    ac = np.fft.fftshift(num/np.maximum(den, 0.05*den.max()))
    return ac / ac[H, W]

def _sub(ac, y, x):
    """Parabolic sub-pixel peak position."""
    def off(a, b, c): d = a - 2*b + c; return 0 if d == 0 else 0.5*(a - c)/d
    return y + off(ac[y-1, x], ac[y, x], ac[y+1, x]), x + off(ac[y, x-1], ac[y, x], ac[y, x+1])

def grid_scale(img, fish_mask, pmin_mm_px=12, max_side=1200):
    h, w = fish_mask.shape
    ys, xs = np.nonzero(fish_mask)
    cy, cx = ys.mean(), xs.mean(); L = max(np.ptp(xs), np.ptp(ys))
    half = int(min(max(L, 600), 1400))           # region around the fish = the fish's plane
    y0, y1 = int(max(0, cy-half)), int(min(h, cy+half)); x0, x1 = int(max(0, cx-half)), int(min(w, cx+half))
    gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    bad = cv2.dilate(fish_mask[y0:y1, x0:x1].astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
    bad |= cv2.GaussianBlur(gray, (0, 0), 15) < 0.55*np.percentile(gray, 90)   # vignette, labels, ink
    f = min(1.0, max_side/max(gray.shape))
    gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    bad = cv2.resize(bad.astype(np.uint8), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    g = gray - cv2.GaussianBlur(gray, (0, 0), 30*f/0.5)
    m = (~bad).astype(np.float32); g = g*m; g -= g[m > 0].mean(); g *= m
    ac = _ac(g, m); H, W = gray.shape
    yy, xx = np.mgrid[-H:H, -W:W]; r = np.hypot(yy, xx)
    rmax = 0.45*min(H, W)
    loc = (ac == maximum_filter(ac, 7)) & (r > pmin_mm_px*f) & (r < rmax) & (ac > 0) & (xx >= 0)
    pk = np.argwhere(loc); vals = ac[loc]
    if len(vals) < 2: return None
    def lattice_ok(v):   # a real lattice vector repeats: peaks also at 2v (and 3v if inside)
        for k in (2, 3):
            ty, tx = H + v[0]*k, W + v[1]*k
            if np.hypot(v[0]*k, v[1]*k) > rmax: break
            rr = max(2, int(0.1*np.hypot(*v))); win = ac[ty-rr:ty+rr+1, tx-rr:tx+rr+1]
            if win.size == 0 or win.max() < 0.2*vals.max(): return False
        return True
    strong = pk[vals >= 0.25*vals.max()]
    strong = strong[np.argsort(r[tuple(strong.T)])]
    cand = [v - [H, W] for v in strong if lattice_ok(v - [H, W])]
    if not cand: return None
    v1 = cand[0]                                                  # nearest real lattice vector
    # precise direction + spacing from the farthest repeat of v1 inside the patch
    vy_, vx_ = float(v1[0]), float(v1[1]); k = 1
    while True:                                   # walk out along the lattice: 1, 2, 4, 8... repeats
        k2 = k*2
        if np.hypot(vy_, vx_)*k2 > rmax: break
        P = np.hypot(vy_, vx_); rr = max(2, int(0.2*P))
        ty, tx = int(round(H + vy_*k2)), int(round(W + vx_*k2))
        win = ac[ty-rr:ty+rr+1, tx-rr:tx+rr+1]; iy, ix = np.unravel_index(win.argmax(), win.shape)
        py, px = _sub(ac, ty-rr+iy, tx-rr+ix); vy_, vx_ = (py-H)/k2, (px-W)/k2; k = k2
    a1 = np.arctan2(vy_, vx_); P0 = np.hypot(vy_, vx_)
    G = gray - cv2.GaussianBlur(gray, (0, 0), 3*P0); G *= m
    vy, vx = np.nonzero(m); out = []; spans = []
    for a in (a1, a1 + np.pi/2):           # same lattice, second direction at +90 degrees
        u = vx*np.cos(a) + vy*np.sin(a); ub = np.round((u - u.min())*4).astype(int)   # 1/4-px bins
        spans.append(float(np.ptp(u))/P0)
        prof = np.bincount(ub, G[vy, vx]); uu = np.arange(len(prof))/4
        Ps = np.linspace(0.95*P0, 1.05*P0, 401)
        pw = [np.abs(np.sum(prof*np.exp(2j*np.pi*uu/P))) for P in Ps]
        out.append(Ps[int(np.argmax(pw))]/f)
    Px, Py = out
    agree = abs(Px/Py - 1)
    span = min(spans)                          # squares of paper the spacing is fitted over (>= MIN_SQUARES)
    ppm = float(np.sqrt(Px*Py))
    return dict(px_per_mm=ppm, px_x=float(Px), px_y=float(Py), k=(0, 0),
                angle=float(np.degrees(a1) % 180), agree=float(agree), squares=round(span, 1),
                ok=bool(agree < 0.03 and span >= MIN_SQUARES),
                line=_show_line(m, a1, ppm*f, f, x0, y0))


def _show_line(m, a, P, f, x0, y0, mm=20):
    """A 20 mm segment on the paper along the grid, in photo pixels, to let the user check the scale."""
    H, W = m.shape
    d = np.array([np.cos(a), np.sin(a)])*P*mm
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return None
    rng = np.random.default_rng(0)
    for i in rng.choice(len(xs), min(400, len(xs)), replace=False):
        p0 = np.array([xs[i], ys[i]], float)
        for s in (1, -1):
            p1 = p0 + s*d
            if 0 <= p1[0] < W and 0 <= p1[1] < H:
                t = np.linspace(0, 1, 25)[:, None]
                pts = np.round(p0 + t*(p1 - p0)).astype(int)
                if m[pts[:, 1], pts[:, 0]].all():
                    return [[float(p0[0]/f + x0), float(p0[1]/f + y0)], [float(p1[0]/f + x0), float(p1[1]/f + y0)]]
    return None

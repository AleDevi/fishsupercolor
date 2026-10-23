"""Standard length, total length and areas from a fish mask (pixels; divide by px/mm outside).
Axis by PCA; width profile along the axis; head = wider end; SL ends at the caudal peduncle
(narrowest point before the caudal fin flares) or at the mask end when the fin was not captured."""
import numpy as np, cv2
from scipy.ndimage import binary_fill_holes, gaussian_filter1d

def clean(mask):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n < 2: return mask
    return binary_fill_holes(lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))

def measure(mask, img=None, nbin=300, head=None):
    """head: None = guess (eye, else wider end); 'left' / 'right' = side of the picture the head is on."""
    m = clean(mask); ys, xs = np.nonzero(m)
    c = np.array([xs.mean(), ys.mean()]); X = np.c_[xs, ys] - c
    ev, evec = np.linalg.eigh(np.cov(X.T)); ax = evec[:, 1]; nx = np.array([-ax[1], ax[0]])
    t = X @ ax; n = X @ nx
    edges = np.linspace(t.min(), t.max(), nbin + 1); b = np.clip(np.digitize(t, edges) - 1, 0, nbin - 1)
    step = edges[1] - edges[0]
    cnt = np.bincount(b, minlength=nbin); width = cnt / step
    mid = np.bincount(b, n, nbin) / np.maximum(cnt, 1)
    wsm = gaussian_filter1d(width, nbin/100)
    q = nbin // 4
    head_first = wsm[int(0.05*nbin):int(0.30*nbin)].mean() >= wsm[int(0.70*nbin):int(0.95*nbin)].mean()
    eye = find_eye(m, img, t, n, wsm.max()) if img is not None else None
    if eye is not None: head_first = eye < 0.5*(t.min() + t.max())       # the eye marks the head
    if head in ('left', 'right'): head_first = (ax[0] > 0) == (head == 'left')   # told by the user
    if not head_first:                          # flip so index 0 = snout
        ax, nx = -ax, -nx; t, n = -t, -n; width, wsm, mid = width[::-1], wsm[::-1], -mid[::-1]
        edges = -edges[::-1]
    # caudal peduncle: narrowest point in the posterior part, if the fin widens again after it
    lo, hi = int(0.55*nbin), int(0.97*nbin)
    i_min = lo + int(np.argmin(wsm[lo:hi]))
    flare = wsm[i_min:].max() / max(wsm[i_min], 1e-9)
    fin = flare > 1.15
    i_sl = i_min if fin else nbin - 1
    # curved midline length snout -> peduncle
    tc = (edges[:-1] + edges[1:]) / 2
    ok = cnt > 0; cf = np.polyfit(tc[ok], mid[ok], 3, w=np.sqrt(cnt[ok]))   # smooth body midline
    midsm = np.polyval(cf, tc)
    pts = c + np.outer(tc, ax) + np.outer(midsm, nx)
    seg = np.hypot(*np.diff(pts[:i_sl+1], axis=0).T).sum() + step/2
    total = np.hypot(*np.diff(pts, axis=0).T).sum() + step
    tt = (np.c_[xs, ys] - c) @ ax
    body = (tt <= tc[i_sl] + step/2)
    return dict(sl_px=float(seg), tl_px=float(total), area_px=int(m.sum()), body_area_px=int(body.sum()),
                max_depth_px=float(wsm[:i_sl+1].max()), fin_found=bool(fin), head_first_axis=bool(head_first),
                snout=pts[0], ped=pts[i_sl], end=pts[-1], midline=pts[:i_sl+1], axis=ax, normal=nx,
                ped_width=float(wsm[i_sl]), mask=m)

def find_eye(m, img, t, n, depth):
    """Pupil = darkest round compact blob inside the mask, near one end. Returns its axis position."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.nonzero(m); vals = g[ys, xs]
    thr = np.percentile(vals, 2.5)
    dark = np.zeros_like(m, np.uint8); sel = vals <= thr; dark[ys[sel], xs[sel]] = 1
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nlab, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    tlo, thi = t.min(), t.max(); L = thi - tlo; best = None
    for i in range(1, nlab):
        a = st[i, cv2.CC_STAT_AREA]
        if a < (0.05*depth)**2: continue
        bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        fill = a / (bw*bh); aspect = min(bw, bh)/max(bw, bh)
        if fill < 0.6 or aspect < 0.65 or max(bw, bh) > 0.45*depth: continue   # round and not too big
        sel = lab[ys, xs] == i; tt = t[sel].mean()
        edge = min(tt - tlo, thi - tt)/L
        if edge > 0.2: continue                                  # the eye sits in the first ~15% of the fish
        score = aspect*fill/(edge + 0.02)
        if best is None or score > best[0]: best = (score, tt)
    return None if best is None else best[1]

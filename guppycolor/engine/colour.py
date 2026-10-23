"""ColorChecker Classic 24 detection + per-image TPS-3D colour correction (Menesatti et al. 2012)."""
import numpy as np, cv2, colour
from scipy.interpolate import RBFInterpolator

def reference_srgb():
    cc = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
    xyY = np.array(list(cc.data.values()))
    XYZ = colour.xyY_to_XYZ(xyY)
    rgb = colour.XYZ_to_sRGB(XYZ, illuminant=cc.illuminant, chromatic_adaptation_transform='Bradford')
    return np.clip(rgb, 0, 1).reshape(4, 6, 3)  # rows x cols, row 0 = brown..cyan, row 3 = white..black

REF = reference_srgb()

def find_chart(img, work=1200):
    """Return 4x6x3 measured patch means (0-1), patch centres (full res), or None."""
    h, w = img.shape[:2]; s = work / max(h, w)
    small = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = (hsv[..., 1] > 70) & (hsv[..., 2] > 60)
    k = max(3, int(0.02 * work))
    m = cv2.morphologyEx(sat.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[i, cv2.CC_STAT_AREA] < 0.01 * small.size / 3: return None
    cnt = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    # coloured patches cover rows 0-2 fully + row 3 greys are unsaturated -> extend rect later by trying both
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)  # 4 corners
    best = None
    for rows_in_box in (3, 4):          # saturated blob may cover only the 3 colour rows
        for start in range(4):
            for flip in (False, True):
                b = np.roll(box, -start, axis=0)
                if flip: b = b[[0, 3, 2, 1]]
                p0, p1, p3 = b[0], b[1], b[3]          # p0->p1 = columns axis, p0->p3 = rows axis
                ex, ey = (p1 - p0) / 6, (p3 - p0) / rows_in_box
                centres = np.array([[p0 + (c + .5) * ex + (r + .5) * ey for c in range(6)] for r in range(4)])
                if (centres < 0).any() or (centres[..., 0] >= small.shape[1]).any() or (centres[..., 1] >= small.shape[0]).any():
                    continue
                rad = max(1, int(0.25 * min(np.linalg.norm(ex), np.linalg.norm(ey))))
                meas = np.array([[small[int(y)-rad:int(y)+rad+1, int(x)-rad:int(x)+rad+1].reshape(-1, 3).mean(0)
                                  for x, y in row] for row in centres]) / 255
                # compare chroma pattern, scale-free (exposure unknown)
                a = meas.reshape(-1, 3); r = REF.reshape(-1, 3)
                err = np.mean((a / a.mean() - r / r.mean()) ** 2)
                if best is None or err < best[0]:
                    best = (err, meas, centres / s)
    return best

def fit_tps(meas, target=None, smooth=1e-3):
    """target=None: nominal ColorChecker sRGB. target=patches measured on a reference photo:
    aligns this photo to the reference instead (photos standardised to each other)."""
    target = REF if target is None else target
    return RBFInterpolator(meas.reshape(-1, 3), target.reshape(-1, 3), kernel='thin_plate_spline', smoothing=smooth)

def apply_tps(tps, img, chunk=2_000_000):
    flat = img.reshape(-1, 3).astype(np.float32) / 255
    out = np.empty_like(flat)
    # evaluate on a 33^3 LUT then trilinear lookup: fast for 12 MP photos
    g = np.linspace(0, 1, 33); G = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    lut = np.clip(tps(G), 0, 1).reshape(33, 33, 33, 3)
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((g, g, g), lut)
    for i in range(0, len(flat), chunk):
        out[i:i+chunk] = f(flat[i:i+chunk])
    return (out.reshape(img.shape) * 255).round().astype(np.uint8)

def delta_e(a, b):
    return colour.delta_E(colour.XYZ_to_Lab(colour.sRGB_to_XYZ(a)), colour.XYZ_to_Lab(colour.sRGB_to_XYZ(b)), method='CIE 2000')

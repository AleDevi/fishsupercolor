"""Sizes of one fish: from the outline first, then the landmarks win where they exist.

- SL: snout -> end of the caudal peduncle landmark, along the midline of the outline (it follows a bent fish).
- TL: SL + the tail fin, end of the peduncle -> tail tip landmark along the midline; without the tail tip, the
  outline length when the whole tail fin is in the outline.
- End of the body: the arc through the top of the caudal fin base, the end of the peduncle and the bottom of the
  caudal fin base (the peduncle ends rounded, not flat); with only the peduncle landmark, the straight line across
  the fish there. The body area is the outline before it, the caudal fin area the outline after it (only when the
  fin is in the outline).
- Maximum depth: across the body at the base of the gonopodium (or another depth landmark of the profile), else the
  widest point of the outline.
The outline values are kept in r['outline'] so that adding or removing a landmark can be undone without a new
analysis."""
import numpy as np

DEPTH_LANDMARKS = ('gonopodium_base', 'anal_fin_base', 'depth_point')    # first one present is used
FIN_REACH = 0.85     # the fin is in the outline if the outline reaches this share of peduncle -> tail tip
CAUDAL_TOP, CAUDAL_BOTTOM = 'caudal_base_top', 'caudal_base_bottom'


def _unit(v):
    v = np.asarray(v, float)
    n = np.hypot(*v)
    return v / n if n > 0 else None


def body_axis(r, pts):
    """Unit vector snout -> tail along the body: landmarks if possible, else the outline axis."""
    if 'snout' in pts:
        for k in ('peduncle_end', 'tail_tip'):
            if k in pts:
                u = _unit(np.subtract(pts[k], pts['snout']))
                if u is not None:
                    return u
    ax = (r.get('outline') or {}).get('axis')
    return _unit(ax) if ax else None


def _arc_position(m, cum, p):
    """Arc length along the polyline m (cumulative lengths cum) of the point of m nearest to p, and that point."""
    a, b = m[:-1], m[1:]
    d = b - a
    t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
    near = a + t[:, None] * d
    k = int(np.argmin(np.hypot(*(near - p).T)))
    return cum[k] + t[k] * np.hypot(*d[k]), near[k]


def along(midline, a, b):
    """Path a -> b that follows the shape of the midline (points ordered snout -> tail) between the places nearest
    to a and to b, shifted so that it starts at a and ends at b (a landmark a little off the midline does not add
    a detour). Straight when there is no midline or b is not after a on it."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if midline is None or len(midline) < 2:
        return [a.tolist(), b.tolist()]
    m = np.asarray(midline, float)
    cum = np.r_[0, np.cumsum(np.hypot(*np.diff(m, axis=0).T))]
    (ta, pa), (tb, pb) = _arc_position(m, cum, a), _arc_position(m, cum, b)
    if tb <= ta:
        return [a.tolist(), b.tolist()]
    sel = (cum > ta) & (cum < tb)
    f = ((cum[sel] - ta) / (tb - ta))[:, None]
    inner = m[sel] + (1 - f) * (a - pa) + f * (b - pb)
    return [a.tolist()] + inner.tolist() + [b.tolist()]


def path_length(path):
    return float(np.hypot(*np.diff(np.asarray(path, float), axis=0).T).sum())


def body_end(ped, u, top=None, bottom=None):
    """The end of the body as a function: (s, v) -> True on the snout side, with s along u and v across, both from
    the peduncle landmark. The arc through top, peduncle and bottom when both are there (a circle, or the straight
    line through top and bottom when the three are in line); else the straight line across the fish at the
    peduncle. Also returns the end of the body as a polyline in photo pixels (for the pictures)."""
    ped, u = np.asarray(ped, float), np.asarray(u, float)
    n = np.array([-u[1], u[0]])
    local = lambda p: (float((np.asarray(p, float) - ped) @ u), float((np.asarray(p, float) - ped) @ n))
    if top is None or bottom is None:
        return (lambda s, v: s <= 0), None
    (s1, v1), (s2, v2) = local(top), local(bottom)
    if abs(v1 - v2) < 1:
        return (lambda s, v: s <= 0), None
    lo, hi = min(v1, v2), max(v1, v2)
    # circle through (v1, s1), (0, 0), (v2, s2) in the (v, s) plane: v^2 + s^2 + D v + E s = 0
    A = np.array([[v1, s1], [v2, s2]])
    rhs = -np.array([v1 ** 2 + s1 ** 2, v2 ** 2 + s2 ** 2])
    if abs(np.linalg.det(A)) < 1e-6 * max(1.0, (v1 ** 2 + s1 ** 2) * (v2 ** 2 + s2 ** 2)) ** 0.5:
        cut = lambda v: s1 + (np.clip(v, lo, hi) - v1) * (s2 - s1) / (v2 - v1)
    else:
        D, E = np.linalg.solve(A, rhs)
        vc, sc = -D / 2, -E / 2
        R = np.hypot(vc, sc)
        sign = 1.0 if sc < 0 else -1.0                   # the branch that passes through the peduncle (0, 0)
        cut = lambda v: sc + sign * np.sqrt(np.maximum(R ** 2 - (np.clip(v, lo, hi) - vc) ** 2, 0))
    vs = np.linspace(lo, hi, 15)
    line = [(ped + float(cut(v)) * u + v * n).round(1).tolist() for v in vs]
    return (lambda s, v: s <= cut(v)), line


def colour_start(r, pts, prof, ppm):
    """Where the colours start (the head is left out, so the eye never counts as black): a point and the body axis,
    with the line across the fish through the point as the cut. From the profile's colour_start (e.g. 2 mm behind the
    eye landmark); without that landmark, a fixed distance from the snout. Returns (point, axis, source) or None
    (source: the landmark used, or 'snout'); None also when the profile has no colour_start."""
    cs = prof.get('colour_start')
    pts = pts or {}
    u = body_axis(r, pts)
    if not cs or u is None or not ppm:
        return None
    lm = cs.get('landmark')
    if lm in pts:
        return (np.asarray(pts[lm], float) + float(cs.get('offset_mm', 0)) * ppm * u).tolist(), u.tolist(), lm
    snout = pts.get(cs.get('fallback_landmark', 'snout')) or (r.get('outline') or {}).get('snout')
    if snout is None or cs.get('fallback_mm') is None:
        return None
    return (np.asarray(snout, float) + float(cs['fallback_mm']) * ppm * u).tolist(), u.tolist(), 'snout'


def ahead_of(shape, point, axis):
    """Pixels in front of the line across the fish through point (on the snout side)."""
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    return (xs - point[0]) * axis[0] + (ys - point[1]) * axis[1] < 0


def chord(mask, p, d, reach):
    """Ends of the outline crossing the line through p with direction d: the run of fish pixels that contains p,
    or the nearest one (a landmark on the edge can fall just outside the outline). None if no run is close."""
    t = np.arange(-reach, reach + 0.5, 0.5)
    xy = np.asarray(p, float)[None] + t[:, None] * np.asarray(d)[None]
    x, y = np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)
    ok = (x >= 0) & (y >= 0) & (x < mask.shape[1]) & (y < mask.shape[0])
    inside = np.zeros(len(t), bool)
    inside[ok] = mask[y[ok], x[ok]]
    if not inside.any():
        return None
    edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(np.int8), 0]))
    runs = edges.reshape(-1, 2)                          # [start, end) indices of each run
    dist = [0 if s <= len(t) // 2 < e else min(abs(s - len(t) // 2), abs(e - 1 - len(t) // 2)) for s, e in runs]
    k = int(np.argmin(dist))
    if dist[k] * 0.5 > 0.15 * reach:
        return None
    s, e = runs[k]
    return [xy[s].tolist(), xy[e - 1].tolist()]


def update(r, mask, pts, flags):
    """Recompute the sizes of result r from the outline values (r['outline']) and the landmarks pts ({key: [x, y]}).
    Changes r; returns flags (a list) with tail_incomplete and sl_landmarks_missing set again."""
    o = r.get('outline') or {}
    pts = pts or {}
    flags = [f for f in flags if f and f not in ('tail_incomplete', 'sl_landmarks_missing')]
    for k in ('standard_length_px', 'total_length_px', 'total_length_source', 'caudal_area_px', 'caudal_length_px',
              'body_end_line', 'depth_line', 'depth_source', 'body_area_source', 'sl_path', 'tl_path'):
        r.pop(k, None)
    if 'body_area_px' in o:
        r['body_area_px'], r['body_area_source'] = o['body_area_px'], 'outline'
    if 'max_depth_px' in o:
        r['max_depth_px'], r['depth_source'] = o['max_depth_px'], 'outline'
    fin_in_outline = bool(o.get('fin_found'))
    snout, ped, tip = pts.get('snout'), pts.get('peduncle_end'), pts.get('tail_tip')
    if pts and not (snout and ped):
        flags.append('sl_landmarks_missing')
    u = body_axis(r, pts)
    if mask is not None and u is not None and mask.any():
        ys, xs = np.nonzero(mask)
        ref = np.asarray(snout if snout else (xs.mean(), ys.mean()), float)
        proj = (np.c_[xs, ys] - ref) @ u
        if snout and ped:
            path = along(o.get('midline'), snout, ped)
            r['standard_length_px'] = round(path_length(path), 1)
            r['sl_path'] = np.round(path, 1).tolist()
            n = np.array([-u[1], u[0]])
            d = np.c_[xs, ys] - np.asarray(ped, float)
            ahead, arc = body_end(ped, u, pts.get(CAUDAL_TOP), pts.get(CAUDAL_BOTTOM))
            body = ahead(d @ u, d @ n)
            r['body_area_px'], r['body_area_source'] = int(body.sum()), 'landmarks'
            half = 0.5 * float(o.get('max_depth_px') or 0.2 * r['standard_length_px'])
            r['body_end_line'] = arc or [(np.asarray(ped) - half * n).tolist(), (np.asarray(ped) + half * n).tolist()]
            if tip:
                cut = float(np.subtract(ped, ref) @ u)
                tip_cut = float(np.subtract(tip, ref) @ u)
                reach = proj.max()
                fin_in_outline = tip_cut > cut and reach >= cut + FIN_REACH * (tip_cut - cut)
                tail = along(o.get('midline'), ped, tip)
                r['caudal_length_px'] = round(path_length(tail), 1)
                r['tl_path'] = np.round(tail, 1).tolist()
            if fin_in_outline:
                r['caudal_area_px'] = int((~body).sum())
        depth_at = next((pts[k] for k in DEPTH_LANDMARKS if k in pts), None)
        if depth_at is not None:
            n = np.array([-u[1], u[0]])
            line = chord(mask, depth_at, n, reach=float(np.ptp(proj)) * 0.5)
            if line:
                r['depth_line'] = [[round(v, 1) for v in line[0]], [round(v, 1) for v in line[1]]]
                r['max_depth_px'] = round(float(np.hypot(*np.subtract(line[1], line[0]))), 1)
                r['depth_source'] = 'landmarks'
    elif snout and ped:
        r['standard_length_px'] = round(float(np.hypot(*np.subtract(ped, snout))), 1)
    if snout and tip:
        if r.get('standard_length_px') and r.get('caudal_length_px'):           # TL = SL + the tail fin
            tl = r['standard_length_px'] + r['caudal_length_px']
        else:
            tl = path_length(along(o.get('midline'), snout, tip))
        r['total_length_px'], r['total_length_source'] = round(tl, 1), 'landmarks'
    elif fin_in_outline and o.get('mask_length_px'):
        r['total_length_px'], r['total_length_source'] = o['mask_length_px'], 'outline'
    r['caudal_in_mask'] = bool(fin_in_outline)
    if not fin_in_outline:
        flags.append('tail_incomplete')
    return flags

"""Sizes of one fish: from the outline first, then the landmarks win where they exist.

- SL: snout -> end of the caudal peduncle (landmarks only).
- TL: snout -> tail tip landmark; without it, the outline length when the whole tail fin is in the outline.
- End of the body: the line across the fish at the peduncle landmark (perpendicular to snout -> peduncle); the body
  area is the outline before it, the caudal fin area the outline after it (only when the fin is in the outline).
- Maximum depth: across the body at the base of the gonopodium (or another depth landmark of the profile), else the
  widest point of the outline.
The outline values are kept in r['outline'] so that adding or removing a landmark can be undone without a new
analysis."""
import numpy as np

DEPTH_LANDMARKS = ('gonopodium_base', 'anal_fin_base', 'depth_point')    # first one present is used
FIN_REACH = 0.85     # the fin is in the outline if the outline reaches this share of peduncle -> tail tip


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
              'body_end_line', 'depth_line', 'depth_source', 'body_area_source'):
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
            sl = float(np.hypot(*np.subtract(ped, snout)))
            r['standard_length_px'] = round(sl, 1)
            cut = float(np.subtract(ped, ref) @ u)
            body = proj <= cut
            r['body_area_px'], r['body_area_source'] = int(body.sum()), 'landmarks'
            n = np.array([-u[1], u[0]])
            half = 0.5 * float(o.get('max_depth_px') or 0.2 * sl)
            r['body_end_line'] = [(np.asarray(ped) - half * n).tolist(), (np.asarray(ped) + half * n).tolist()]
            if tip:
                tip_cut = float(np.subtract(tip, ref) @ u)
                reach = proj.max()
                fin_in_outline = tip_cut > cut and reach >= cut + FIN_REACH * (tip_cut - cut)
                r['caudal_length_px'] = round(float(np.hypot(*np.subtract(tip, ped))), 1)
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
        r['total_length_px'], r['total_length_source'] = round(float(np.hypot(*np.subtract(tip, snout))), 1), 'landmarks'
    elif fin_in_outline and o.get('mask_length_px'):
        r['total_length_px'], r['total_length_source'] = o['mask_length_px'], 'outline'
    r['caudal_in_mask'] = bool(fin_in_outline)
    if not fin_in_outline:
        flags.append('tail_incomplete')
    return flags

"""Synthetic guppies for the pattern tests: a body outline in body lengths, spots placed in body coordinates,
drawn at any position, size, angle, mirrored or not, so the true pattern of every fish is known."""
import cv2
import numpy as np

# liveries: (class index, x, y, half-length, half-width, angle deg) in body lengths, dorsal y > 0
LIVERIES = {
    'A': [(1, 0.35, 0.03, 0.05, 0.05, 0), (2, 0.65, -0.02, 0.04, 0.04, 0)],
    'B': [(1, 0.55, 0.0, 0.18, 0.025, 0), (3, 0.25, -0.05, 0.035, 0.035, 0)],
    'C': [(2, 0.30, 0.05, 0.04, 0.04, 0), (2, 0.80, 0.0, 0.04, 0.04, 0), (3, 0.55, 0.06, 0.12, 0.02, 20)],
}


def outline(depth=0.13, tail=True):
    x = np.linspace(0, 1, 60)
    h = 0.035 + (depth - 0.035) * np.sin(np.pi * np.clip(x, 0, 1) ** 0.8) ** 1.2
    h[0] = 0.01
    top = np.c_[x, h]
    bot = np.c_[x, -h * 0.9][::-1]
    pts = [top]
    if tail:
        pts.append(np.array([[1.0, 0.035], [1.28, 0.10], [1.28, -0.10], [1.0, -0.035]]))
    pts.append(bot)
    return np.vstack(pts)


def draw(livery, size=(900, 1300), L=700, angle=0.0, centre=(650, 450), mirror=False, depth=0.13, jitter=0.0, seed=0):
    """Mask (bool) and label map (uint8: 0 background, 4 base, class indices) of one fish."""
    rng = np.random.default_rng(seed)
    a = np.radians(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    def tr(P):
        P = np.asarray(P, float).copy()
        P[:, 1] = -P[:, 1]                       # body y dorsal-up -> image y down
        if mirror:
            P[:, 0] = -P[:, 0]
        return (P - [0.6 * (-1 if mirror else 1), 0]) * L @ R.T + centre
    mask = np.zeros(size, np.uint8)
    cv2.fillPoly(mask, [np.round(tr(outline(depth))).astype(np.int32)], 1)
    lab = np.where(mask > 0, 4, 0).astype(np.uint8)
    t = np.linspace(0, 2 * np.pi, 40)
    for cls, x, y, hl, hw, ang in livery:
        x, y = x + rng.normal(0, jitter), y + rng.normal(0, jitter)
        b = np.radians(ang)
        E = np.c_[hl * np.cos(t), hw * np.sin(t)] @ np.array([[np.cos(b), np.sin(b)], [-np.sin(b), np.cos(b)]]) + [x, y]
        spot = np.zeros(size, np.uint8)
        cv2.fillPoly(spot, [np.round(tr(E)).astype(np.int32)], 1)
        lab[(spot > 0) & (mask > 0)] = cls
    return mask > 0, lab


CLASSES = [{'index': 1, 'key': 'orange', 'label': 'Orange', 'display': '#E8742A'},
           {'index': 2, 'key': 'black', 'label': 'Black', 'display': '#222222'},
           {'index': 3, 'key': 'iridescent', 'label': 'Iridescent', 'display': '#3D7FD6'},
           {'index': 4, 'key': 'base', 'label': 'Base', 'display': '#C8B89A'},
           {'index': 5, 'key': 'reflection', 'label': 'Reflection', 'display': '#FFFFFF'}]


def population(n_per=6, seed=1):
    """(livery, mask, labels) for n_per fish of each livery, in varied poses."""
    rng = np.random.default_rng(seed)
    out = []
    for k, liv in LIVERIES.items():
        for i in range(n_per):
            m, lab = draw(liv, L=rng.uniform(500, 800), angle=rng.uniform(-12, 12), mirror=bool(rng.random() < 0.4),
                          centre=(650 + rng.uniform(-40, 40), 450 + rng.uniform(-30, 30)), depth=rng.uniform(0.11, 0.15),
                          jitter=0.012, seed=int(rng.integers(1e9)))
            out.append((k, m, lab))
    return out

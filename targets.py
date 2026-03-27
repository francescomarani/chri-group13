# -*- coding: utf-8 -*-
"""
PA3 — Tube Target
------------------
Generates a curved tube defined by:
  - centerline: a smooth parametric curve (Nx2 in physical coords, meters)
  - half_width: tube radius in meters
  - wall_left / wall_right: offset curves at ±half_width from centerline

The task is to navigate from the start to the end of the tube without
touching the walls.

Physical coordinate frame (Haply): +x right, +y down on screen.
All shapes are centered around cy ≈ 0.06 m (workspace center).
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────

# Narrower than the original template, but compressed to stay reachable with
# the physical Haply workspace.
DEFAULT_HALF_WIDTH = 0.007
DEFAULT_CX = 0.0
DEFAULT_CY = 0.055


def _compress_about_center(centerline, sx=1.0, sy=1.0):
    """Scale a curve about its centroid to keep it inside the haptic workspace."""
    pts = np.asarray(centerline, dtype=float)
    pivot = pts.mean(axis=0)
    scaled = pts.copy()
    scaled[:, 0] = pivot[0] + sx * (scaled[:, 0] - pivot[0])
    scaled[:, 1] = pivot[1] + sy * (scaled[:, 1] - pivot[1])
    return scaled

def _normals(centerline):
    """Compute unit normals (pointing left) at each point of the centerline."""
    # Tangent via central differences
    tangents = np.zeros_like(centerline)
    tangents[1:-1] = centerline[2:] - centerline[:-2]
    tangents[0] = centerline[1] - centerline[0]
    tangents[-1] = centerline[-1] - centerline[-2]
    # Normalise
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.clip(lengths, 1e-12, None)
    tangents = tangents / lengths
    # Rotate 90° CCW to get left-pointing normal: (tx, ty) → (-ty, tx)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    return normals


def _smooth(pts, sigma=3):
    """Simple Gaussian smoothing on a 2D curve."""
    from scipy.ndimage import gaussian_filter1d
    return np.column_stack([
        gaussian_filter1d(pts[:, 0], sigma),
        gaussian_filter1d(pts[:, 1], sigma),
    ])


class Tube:
    """A curved tube target."""

    def __init__(self, centerline, half_width):
        """
        Parameters
        ----------
        centerline : ndarray (N, 2) — physical coords (meters)
        half_width : float — tube radius in meters
        """
        self.centerline = np.asarray(centerline, dtype=float)
        self.half_width = half_width
        self.normals = _normals(self.centerline)
        self.wall_left = self.centerline + self.normals * half_width
        self.wall_right = self.centerline - self.normals * half_width
        self.start = self.centerline[0].copy()
        self.end = self.centerline[-1].copy()
        self.n_pts = len(self.centerline)

    def closest_centerline_point(self, pos):
        """
        Find the closest point on the centerline to `pos`.

        Returns
        -------
        idx      : int — index of closest centerline point
        proj     : ndarray (2,) — closest point on centerline
        normal   : ndarray (2,) — normal at that point
        signed_d : float — signed distance from centerline
                   (positive = left of centerline, negative = right)
        abs_d    : float — absolute distance from centerline
        """
        pos = np.asarray(pos, dtype=float)
        diffs = self.centerline - pos
        dists = np.linalg.norm(diffs, axis=1)
        idx = int(np.argmin(dists))
        proj = self.centerline[idx]
        normal = self.normals[idx]
        # Signed distance: positive if pos is on the left-normal side
        to_pos = pos - proj
        signed_d = float(np.dot(to_pos, normal))
        abs_d = abs(signed_d)
        return idx, proj, normal, signed_d, abs_d

    def is_inside(self, pos):
        """Check if pos is inside the tube."""
        _, _, _, _, abs_d = self.closest_centerline_point(pos)
        return abs_d < self.half_width

    def wall_penetration(self, pos):
        """
        Returns how far outside the tube the position is.
        0 if inside, >0 if penetrating a wall.
        Also returns which wall: 'left', 'right', or None.
        """
        _, _, _, signed_d, abs_d = self.closest_centerline_point(pos)
        if abs_d <= self.half_width:
            return 0.0, None
        penetration = abs_d - self.half_width
        wall = 'left' if signed_d > 0 else 'right'
        return penetration, wall

    def progress(self, pos):
        """Return progress along the tube as fraction [0, 1]."""
        idx, _, _, _, _ = self.closest_centerline_point(pos)
        return idx / max(self.n_pts - 1, 1)

# ─── Tube generators ───────────────────────────────────────────────────────

def s_curve_tube(half_width=DEFAULT_HALF_WIDTH, n=400, cx=DEFAULT_CX, cy=DEFAULT_CY):
    """
    Multi-bend S tube with monotonic progression in y.

    Designed to stay challenging without creating self-overlaps or visually
    ambiguous near-intersections between the two walls.
    """
    waypoints = np.array([
        [cx + 0.022, cy - 0.050],
        [cx - 0.038, cy - 0.036],
        [cx - 0.030, cy - 0.016],
        [cx + 0.020, cy - 0.002],
        [cx + 0.032, cy + 0.016],
        [cx - 0.010, cy + 0.032],
        [cx - 0.026, cy + 0.048],
        [cx + 0.006, cy + 0.060],
    ])
    t_wp = np.linspace(0, 1, len(waypoints))
    t_fine = np.linspace(0, 1, n)
    x = np.interp(t_fine, t_wp, waypoints[:, 0])
    y = np.interp(t_fine, t_wp, waypoints[:, 1])
    centerline = _smooth(np.column_stack([x, y]), sigma=7)
    centerline = _compress_about_center(centerline, sx=0.84, sy=0.74)
    return Tube(centerline, half_width)

# def s_curve_tube(half_width=0.012, n=400, cx=0.0, cy=0.06):
#     """
#     Double S-shaped curved tube — two full bends (one full sine period).
#     """
#     t = np.linspace(0, 2 * np.pi, n)   # was np.pi → now full period = 2 curves
#     amplitude = 0.035                   # slightly tighter horizontally
#     height = 0.10                       # taller to fit 2 bends
#     x = cx + amplitude * np.sin(t)
#     y = cy - height / 2 + (height / n) * np.arange(n)
#     centerline = np.column_stack([x, y])
#     centerline = _smooth(centerline, sigma=5)
#     return Tube(centerline, half_width)



def spiral_tube(half_width=DEFAULT_HALF_WIDTH, n=500, cx=DEFAULT_CX, cy=DEFAULT_CY):
    """
    Spiral tube — wraps around ~1.5 turns, getting tighter.
    """
    t = np.linspace(0, 3 * np.pi, n)
    r_start, r_end = 0.034, 0.012
    r = np.linspace(r_start, r_end, n)
    x = cx + r * np.cos(t)
    y = cy + r * np.sin(t)
    centerline = np.column_stack([x, y])
    centerline = _compress_about_center(centerline, sx=0.92, sy=0.82)
    return Tube(centerline, half_width)


def zigzag_tube(half_width=DEFAULT_HALF_WIDTH, n=400, cx=DEFAULT_CX, cy=DEFAULT_CY):
    """
    Zigzag tube — sharp turns that require careful navigation.
    """
    # Build waypoints
    amp = 0.034
    waypoints = np.array([
        [cx - amp, cy - 0.032],
        [cx + amp, cy - 0.020],
        [cx - 0.8 * amp, cy - 0.004],
        [cx + 0.95 * amp, cy + 0.010],
        [cx - amp, cy + 0.022],
        [cx - amp, cy + 0.032],
    ])
    # Interpolate smoothly
    t_wp = np.linspace(0, 1, len(waypoints))
    t_fine = np.linspace(0, 1, n)
    x = np.interp(t_fine, t_wp, waypoints[:, 0])
    y = np.interp(t_fine, t_wp, waypoints[:, 1])
    centerline = _smooth(np.column_stack([x, y]), sigma=6)
    centerline = _compress_about_center(centerline, sx=0.90, sy=0.80)
    return Tube(centerline, half_width)

def _rotate_centerline(centerline, angle_deg):
        """Rotate centerline by angle_deg around its own center."""
        pts = np.asarray(centerline, dtype=float)
        pivot = pts.mean(axis=0)
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, -s], [s, c]])
        return (pts - pivot) @ R.T + pivot
    
def s_curve_tube_rotated(angle_deg, half_width=DEFAULT_HALF_WIDTH, n=400, cx=DEFAULT_CX, cy=DEFAULT_CY):
    base = s_curve_tube(half_width=half_width, n=n, cx=cx, cy=cy)
    rotated_cl = _rotate_centerline(base.centerline, angle_deg)
    return Tube(rotated_cl, half_width)


# ─── Catalogue ──────────────────────────────────────────────────────────────
TUBES = {
    "s_curve":   s_curve_tube,
    "s_curve_90":  lambda **kw: s_curve_tube_rotated(90,  **kw),
    "s_curve_180": lambda **kw: s_curve_tube_rotated(180, **kw),
    "s_curve_270": lambda **kw: s_curve_tube_rotated(270, **kw),
    "spiral":    spiral_tube,
    "zigzag":    zigzag_tube,
}

TUBE_NAMES = list(TUBES.keys())

def get_tube(name="s_curve", **kwargs):
    """Return a Tube object by name."""
    return TUBES[name](**kwargs)

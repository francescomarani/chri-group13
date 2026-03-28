# -*- coding: utf-8 -*-
"""
PA3 — Metrics
-------------
Objective evaluation metrics for trajectory comparison.

All functions accept trajectories as Nx2 numpy arrays in *physical* (meter)
coordinates.
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff

# ─────────────────────────────────────────────────────────────────────────────

def _subsample_curve(curve, max_pts=250):
    """Uniformly subsample a curve to at most max_pts points."""
    if len(curve) <= max_pts:
        return curve
    idx = np.linspace(0, len(curve) - 1, max_pts, dtype=int)
    return curve[idx]


def frechet_discrete(P, Q):
    """
    Discrete Fréchet distance between two 2-D curves.
    Iterative (bottom-up DP) implementation — no recursion limit issues.
    Curves are subsampled to ≤250 pts each to keep memory reasonable.
    """
    P = _subsample_curve(np.asarray(P))
    Q = _subsample_curve(np.asarray(Q))
    n, m = len(P), len(Q)

    # Pre-compute pairwise distances
    # (N,1,2) - (1,M,2) → (N,M)
    dist = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)

    ca = np.empty((n, m))
    ca[0, 0] = dist[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                           dist[i, j])
    return float(ca[n - 1, m - 1])


def mean_nearest_distance(traj, ref):
    """
    Mean nearest-point distance from every point in *traj* to the
    closest point on *ref*.  Returns distance in the same units as
    the input (meters).
    """
    # For each point in traj, find the nearest point in ref
    # Vectorised with broadcasting: (N,1,2) - (1,M,2) → (N,M)
    dists = np.linalg.norm(traj[:, None, :] - ref[None, :, :], axis=2)
    return float(np.mean(np.min(dists, axis=1)))


def hausdorff(traj, ref):
    """Symmetric Hausdorff distance."""
    d1 = directed_hausdorff(traj, ref)[0]
    d2 = directed_hausdorff(ref, traj)[0]
    return max(d1, d2)


def dtw_distance(traj, ref):
    """
    Dynamic Time Warping distance (Euclidean).
    Iterative O(nm) with subsampling to ≤250 pts.
    """
    traj = _subsample_curve(np.asarray(traj))
    ref = _subsample_curve(np.asarray(ref))
    n, m = len(traj), len(ref)
    # Pre-compute pairwise distances
    dist = np.linalg.norm(traj[:, None, :] - ref[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = dist[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)  # normalised


def path_length(traj):
    """Total Euclidean arc-length of the trajectory (meters)."""
    return float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))


def average_pairwise_frechet(demos):
    """Average discrete Fréchet distance across all unique demo pairs."""
    demos = [np.asarray(demo, dtype=float) for demo in demos if len(demo) >= 2]
    if len(demos) < 2:
        return 0.0
    values = []
    for i in range(len(demos)):
        for j in range(i + 1, len(demos)):
            values.append(frechet_discrete(demos[i], demos[j]))
    return float(np.mean(values)) if values else 0.0


def path_length_ratio(traj, ref):
    """Path-length ratio = trajectory length / reference trajectory length."""
    denom = path_length(ref)
    if denom <= 1e-12:
        return 1.0
    return float(path_length(traj) / denom)


def mean_jerk_magnitude(traj, duration_s=None):
    """
    Mean jerk magnitude for a 2D trajectory.

    If duration_s is omitted, samples are assumed uniformly spaced with dt=1.
    """
    traj = np.asarray(traj, dtype=float)
    if len(traj) < 4:
        return 0.0
    if duration_s is None or duration_s <= 0:
        dt = 1.0
    else:
        dt = float(duration_s) / max(len(traj) - 1, 1)
    vel = np.diff(traj, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    mags = np.linalg.norm(jerk, axis=1)
    return float(np.mean(mags)) if len(mags) else 0.0


def endpoint_errors(traj, ref):
    """Start/end point errors between a trajectory and a reference curve."""
    traj = np.asarray(traj)
    ref = np.asarray(ref)
    return (
        float(np.linalg.norm(traj[0] - ref[0])),
        float(np.linalg.norm(traj[-1] - ref[-1])),
    )


def compute_all_metrics(demo, gp_repro, target):
    """
    Compute all metrics for one trial.

    Parameters
    ----------
    demo     : ndarray (N,2) — the raw human demonstration
    gp_repro : ndarray (M,2) — the GP reproduction
    target   : ndarray (K,2) — the ground-truth target shape

    Returns
    -------
    dict with keys:
        demo_mnd, demo_hausdorff, demo_frechet, demo_dtw,
        gp_mnd, gp_hausdorff, gp_frechet, gp_dtw,
        demo_length, gp_length, target_length
    """
    return {
        # Demo vs target
        "demo_mnd":       mean_nearest_distance(demo, target),
        "demo_hausdorff": hausdorff(demo, target),
        "demo_frechet":   frechet_discrete(demo, target),
        "demo_dtw":       dtw_distance(demo, target),
        # GP reproduction vs target
        "gp_mnd":         mean_nearest_distance(gp_repro, target),
        "gp_hausdorff":   hausdorff(gp_repro, target),
        "gp_frechet":     frechet_discrete(gp_repro, target),
        "gp_dtw":         dtw_distance(gp_repro, target),
        # Lengths
        "demo_length":    path_length(demo),
        "gp_length":      path_length(gp_repro),
        "target_length":  path_length(target),
        # Endpoint quality
        "demo_start_error": endpoint_errors(demo, target)[0],
        "demo_end_error":   endpoint_errors(demo, target)[1],
        "gp_start_error":   endpoint_errors(gp_repro, target)[0],
        "gp_end_error":     endpoint_errors(gp_repro, target)[1],
    }

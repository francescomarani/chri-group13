# -*- coding: utf-8 -*-
"""
PA3 — Gaussian Process Trajectory Learning (Multi-Demo)
-------------------------------------------------------
Learns a 2-D trajectory from ONE or MULTIPLE demonstrations.

Single demo  → smoothing only (like a low-pass filter).
Multi demo   → the GP finds the *mean* trajectory across all demos,
               weighted by the RBF kernel, and the predictive variance
               shrinks where demos agree and grows where they diverge.

Pipeline
--------
1.  Each demo is re-parameterised by normalised arc-length  s ∈ [0, 1].
2.  All demos are concatenated into one training set  (s, x) and (s, y).
    Points from different demos at similar s-values naturally get averaged.
3.  Two independent GPs are fitted:  x(s)  and  y(s).
4.  Query on a dense uniform s-grid → smooth reproduced trajectory + std.
"""

import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.exceptions import ConvergenceWarning

# ─────────────────────────────────────────────────────────────────────────────

def _arc_length_param(traj):
    """Return normalised cumulative arc-length parameter s ∈ [0, 1]."""
    diffs = np.diff(traj, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = s[-1]
    if total < 1e-12:
        return np.linspace(0, 1, len(traj))
    return s / total


def _subsample(traj, max_pts=150):
    """Subsample a trajectory to at most max_pts points."""
    if len(traj) <= max_pts:
        return traj.copy()
    idx = np.linspace(0, len(traj) - 1, max_pts, dtype=int)
    return traj[idx]


class TrajectoryGP:
    """Gaussian-Process trajectory model — supports multiple demonstrations."""

    MAX_TOTAL_PTS = 200  # keep total training points under this for speed

    def __init__(self, length_scale=0.1, noise_level=1e-4):
        kernel = (ConstantKernel(1.0, (1e-5, 1e3))
                  * RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e1))
                  + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-8, 1e0)))
        self.gp_x = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        self.gp_y = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        self._fitted = False
        self.n_demos = 0

    def fit(self, demonstrations):
        """
        Fit the GP on one or multiple demonstrations.

        Parameters
        ----------
        demonstrations : list of ndarray(N_i, 2)  OR  single ndarray(N, 2)
        """
        if isinstance(demonstrations, np.ndarray) and demonstrations.ndim == 2:
            demonstrations = [demonstrations]

        n_demos = len(demonstrations)
        # Adaptive: divide budget equally across demos
        pts_per_demo = max(20, self.MAX_TOTAL_PTS // max(n_demos, 1))

        all_s = []
        all_x = []
        all_y = []

        for demo in demonstrations:
            demo = _subsample(np.asarray(demo, dtype=float), max_pts=pts_per_demo)
            s = _arc_length_param(demo)
            all_s.append(s)
            all_x.append(demo[:, 0])
            all_y.append(demo[:, 1])

        S = np.concatenate(all_s).reshape(-1, 1)
        X = np.concatenate(all_x)
        Y = np.concatenate(all_y)

        # This warning is common with short/noisy demonstrations and does not
        # indicate a failed fit for our use case. Keep other warnings visible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*constant_value is close to the specified lower bound.*",
                category=ConvergenceWarning,
            )
            self.gp_x.fit(S, X)
            self.gp_y.fit(S, Y)
        self._fitted = True
        self.n_demos = len(demonstrations)

    def predict(self, n_points=300, return_std=False):
        """
        Generate the reproduced trajectory.

        Returns
        -------
        traj : ndarray (n_points, 2)
        std  : ndarray (n_points, 2)  — only if return_std is True
        """
        if not self._fitted:
            raise RuntimeError("GP not fitted yet — call .fit() first.")
        s_query = np.linspace(0, 1, n_points).reshape(-1, 1)
        x_pred, x_std = self.gp_x.predict(s_query, return_std=True)
        y_pred, y_std = self.gp_y.predict(s_query, return_std=True)
        traj = np.column_stack([x_pred, y_pred])
        if return_std:
            return traj, np.column_stack([x_std, y_std])
        return traj

    @property
    def fitted(self):
        return self._fitted

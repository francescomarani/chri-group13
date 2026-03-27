# -*- coding: utf-8 -*-
"""
GP-based velocity policy for space debris capture.

Learns a state-conditioned velocity policy from demonstrations.
Each demonstration is a sequence of (state, action) pairs recorded at ~50 Hz.

State (5D):
  [dx, dy, dvx, dvy, phase]
  dx, dy    = arm_tip_pos - debris_pos   (normalized by 0.1 m)
  dvx, dvy  = arm_vel - debris_vel       (normalized by 0.1 m/s)
  phase     = 0 if APPROACHING, 1 if GRABBED

Action (2D):
  arm_vel normalized by 0.05 m/s

Two independent GaussianProcessRegressors (scikit-learn) for x-vel and y-vel.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

# Normalization constants
POS_NORM = 0.1      # m
VEL_NORM = 0.1      # m/s
ACTION_NORM = 0.05  # m/s

# Subsampling limits
MAX_PTS_PER_DEMO = 100
MAX_TOTAL_PTS = 300


def _make_kernel():
    return (ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=0.3, length_scale_bounds=(1e-2, 1e1))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0)))


class GPPolicy:
    """
    State-conditioned velocity policy learned from demonstrations via GP regression.
    """

    def __init__(self):
        self._gp_vx = GaussianProcessRegressor(
            kernel=_make_kernel(), n_restarts_optimizer=2, normalize_y=True)
        self._gp_vy = GaussianProcessRegressor(
            kernel=_make_kernel(), n_restarts_optimizer=2, normalize_y=True)
        self._trained = False
        self._n_demos = 0
        # Accumulated training data
        self._X_train = []   # list of (N, 5) arrays
        self._y_train = []   # list of (N, 2) arrays

    @property
    def trained(self):
        return self._trained

    @property
    def n_demos(self):
        return self._n_demos

    def add_demo(self, states, actions):
        """
        Add a demonstration to the training pool.

        Parameters
        ----------
        states  : np.ndarray shape (N, 5) — normalized state vectors
        actions : np.ndarray shape (N, 2) — normalized action vectors
        """
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        if len(states) == 0:
            return

        # Subsample to at most MAX_PTS_PER_DEMO
        n = len(states)
        if n > MAX_PTS_PER_DEMO:
            idx = np.linspace(0, n - 1, MAX_PTS_PER_DEMO, dtype=int)
            states = states[idx]
            actions = actions[idx]

        self._X_train.append(states)
        self._y_train.append(actions)
        self._n_demos += 1

    def train(self):
        """Fit both GPs on all accumulated demonstrations."""
        if len(self._X_train) == 0:
            return

        X_all = np.concatenate(self._X_train, axis=0)
        y_all = np.concatenate(self._y_train, axis=0)

        # Cap total training points
        n_total = len(X_all)
        if n_total > MAX_TOTAL_PTS:
            idx = np.linspace(0, n_total - 1, MAX_TOTAL_PTS, dtype=int)
            X_all = X_all[idx]
            y_all = y_all[idx]

        self._gp_vx.fit(X_all, y_all[:, 0])
        self._gp_vy.fit(X_all, y_all[:, 1])
        self._trained = True

    def predict(self, state):
        """
        Predict the velocity action for a given state.

        Parameters
        ----------
        state : array-like shape (5,) — normalized state vector

        Returns
        -------
        mean_action : np.ndarray shape (2,) — denormalized velocity in m/s
        std         : float — mean prediction standard deviation (normalized units)
        """
        if not self._trained:
            return np.zeros(2), 1.0

        x = np.asarray(state, dtype=float).reshape(1, -1)
        vx_norm, vx_std = self._gp_vx.predict(x, return_std=True)
        vy_norm, vy_std = self._gp_vy.predict(x, return_std=True)

        mean_norm = np.array([float(vx_norm[0]), float(vy_norm[0])])
        std = float((vx_std[0] + vy_std[0]) / 2.0)

        # Denormalize
        mean_action = mean_norm * ACTION_NORM
        return mean_action, std

    def clear(self):
        """Clear all training data and reset GP models."""
        self._gp_vx = GaussianProcessRegressor(
            kernel=_make_kernel(), n_restarts_optimizer=2, normalize_y=True)
        self._gp_vy = GaussianProcessRegressor(
            kernel=_make_kernel(), n_restarts_optimizer=2, normalize_y=True)
        self._trained = False
        self._n_demos = 0
        self._X_train = []
        self._y_train = []

    @staticmethod
    def build_state(arm_tip_pos, arm_vel, debris_pos, debris_vel, phase):
        """
        Build a normalized 5D state vector.

        Parameters
        ----------
        arm_tip_pos : np.ndarray (2,)
        arm_vel     : np.ndarray (2,)
        debris_pos  : np.ndarray (2,)
        debris_vel  : np.ndarray (2,)
        phase       : float — 0.0 for APPROACHING, 1.0 for GRABBED

        Returns
        -------
        state : np.ndarray shape (5,)
        """
        arm_tip_pos = np.asarray(arm_tip_pos, dtype=float)
        arm_vel = np.asarray(arm_vel, dtype=float)
        debris_pos = np.asarray(debris_pos, dtype=float)
        debris_vel = np.asarray(debris_vel, dtype=float)

        dx = (arm_tip_pos[0] - debris_pos[0]) / POS_NORM
        dy = (arm_tip_pos[1] - debris_pos[1]) / POS_NORM
        dvx = (arm_vel[0] - debris_vel[0]) / VEL_NORM
        dvy = (arm_vel[1] - debris_vel[1]) / VEL_NORM

        return np.array([dx, dy, dvx, dvy, float(phase)])

    @staticmethod
    def normalize_action(arm_vel):
        """
        Normalize arm velocity to action space.

        Parameters
        ----------
        arm_vel : np.ndarray (2,)

        Returns
        -------
        np.ndarray shape (2,)
        """
        return np.asarray(arm_vel, dtype=float) / ACTION_NORM

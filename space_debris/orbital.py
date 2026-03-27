# -*- coding: utf-8 -*-
"""
Debris dynamics in the Hill (CW) frame.
Satellite is at the origin. Equations:
  ẍ =  3n²x + 2nẏ + fx/m
  ÿ = -2nẋ       + fy/m
"""
import numpy as np


class OrbitalDebris:
    """
    Models a piece of space debris following Clohessy-Wiltshire (Hill) equations
    in the satellite's Hill frame.
    """

    def __init__(
        self,
        pos=None,
        vel=None,
        mass=0.5,
        radius=0.010,
        spin_rate=2.0,
        n=0.08,
    ):
        """
        Parameters
        ----------
        pos       : initial position [x, y] in meters (Hill frame)
        vel       : initial velocity [vx, vy] in m/s
        mass      : debris mass in kg (scaled)
        radius    : debris radius in meters (for collision / grab detection)
        spin_rate : constant angular velocity in rad/s
        n         : CW mean motion in rad/s (orbital angular rate)
        """
        self._pos = np.array(pos if pos is not None else [-0.04, -0.03], dtype=float)
        self._vel = np.array(vel if vel is not None else [0.003, 0.001], dtype=float)
        self._mass = float(mass)
        self._radius = float(radius)
        self._spin_rate = float(spin_rate)
        self._n = float(n)
        self._angle = 0.0
        self._grabbed = False

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def pos(self):
        return self._pos.copy()

    @property
    def vel(self):
        return self._vel.copy()

    @property
    def angle(self):
        return self._angle

    @property
    def spin_rate(self):
        return self._spin_rate

    @property
    def grabbed(self):
        return self._grabbed

    @property
    def mass(self):
        return self._mass

    @property
    def radius(self):
        return self._radius

    @property
    def n(self):
        return self._n

    # ── CW dynamics ─────────────────────────────────────────────────────────

    def cw_acceleration(self):
        """
        Return [ax, ay] due to Clohessy-Wiltshire terms only (no external force).
        ẍ =  3n²x + 2nẏ
        ÿ = -2nẋ
        """
        x, y = self._pos
        vx, vy = self._vel
        n = self._n
        ax = 3.0 * n * n * x + 2.0 * n * vy
        ay = -2.0 * n * vx
        return np.array([ax, ay])

    def step(self, dt):
        """
        Advance free-floating debris by one timestep using semi-implicit Euler.
        vel is updated before pos (semi-implicit).
        Angle increments regardless of grabbed state.
        """
        self._angle += self._spin_rate * dt

        if self._grabbed:
            return  # position managed externally via update_grabbed()

        acc = self.cw_acceleration()
        # Semi-implicit Euler: update vel first, then pos
        self._vel += acc * dt
        self._pos += self._vel * dt

    def grab(self, arm_tip_pos):
        """
        Attach the debris to the arm tip.

        Parameters
        ----------
        arm_tip_pos : np.ndarray shape (2,)
        """
        self._grabbed = True
        self._pos = np.array(arm_tip_pos, dtype=float)
        self._vel = np.zeros(2)

    def release(self, arm_tip_vel):
        """
        Detach debris, giving it the arm's current velocity.

        Parameters
        ----------
        arm_tip_vel : np.ndarray shape (2,)
        """
        self._grabbed = False
        self._vel = np.array(arm_tip_vel, dtype=float)

    def update_grabbed(self, arm_tip_pos, arm_tip_vel, dt):
        """
        Keep debris at arm tip position while grabbed.
        Angle still increments by spin_rate * dt.

        Parameters
        ----------
        arm_tip_pos : np.ndarray shape (2,)
        arm_tip_vel : np.ndarray shape (2,)
        dt          : float timestep
        """
        self._angle += self._spin_rate * dt
        self._pos = np.array(arm_tip_pos, dtype=float)
        self._vel = np.array(arm_tip_vel, dtype=float)

    def reset(self, pos, vel):
        """Reset debris to specified initial conditions."""
        self._pos = np.array(pos, dtype=float)
        self._vel = np.array(vel, dtype=float)
        self._angle = 0.0
        self._grabbed = False

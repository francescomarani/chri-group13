# -*- coding: utf-8 -*-
"""
Arm state management for space debris capture task.

Manages the single-arm end-effector's position, velocity, grab/release logic,
and deposit detection.
"""
import numpy as np


# Arm states
IDLE = 'IDLE'
HOLDING = 'HOLDING'

# Grab distance threshold
GRAB_RADIUS = 0.018  # meters


class Arm:
    """
    Represents the single-arm end-effector attached to the satellite.

    The arm tip position is updated externally each frame from the Haply
    device or mouse. Velocity is computed as a finite difference.
    """

    # EMA smoothing factor for velocity (0 = frozen, 1 = no smoothing).
    # At 60 Hz, 0.15 gives a ~7-frame moving average — removes encoder noise
    # that would otherwise amplify into oscillating haptic forces.
    VEL_ALPHA = 0.15

    def __init__(self, shoulder_pos):
        """
        Parameters
        ----------
        shoulder_pos : array-like shape (2,) — fixed shoulder attachment point in meters
        """
        self.shoulder_pos = np.array(shoulder_pos, dtype=float)
        self.tip_pos = np.array(shoulder_pos, dtype=float)
        self.tip_vel = np.zeros(2)
        self._prev_tip_pos = np.array(shoulder_pos, dtype=float)
        self._vel_raw = np.zeros(2)
        self.state = IDLE

    @property
    def grab_radius(self):
        return GRAB_RADIUS

    def update(self, new_tip_pos, dt):
        """
        Update arm tip position and compute velocity via finite difference.

        Parameters
        ----------
        new_tip_pos : np.ndarray shape (2,) — new tip position in meters
        dt          : float timestep in seconds
        """
        new_tip_pos = np.asarray(new_tip_pos, dtype=float)
        if dt > 0:
            self._vel_raw = (new_tip_pos - self._prev_tip_pos) / dt
        else:
            self._vel_raw = np.zeros(2)
        # EMA low-pass filter: smooths out encoder/mouse noise before haptics
        self.tip_vel = (self.VEL_ALPHA * self._vel_raw
                        + (1.0 - self.VEL_ALPHA) * self.tip_vel)
        self._prev_tip_pos = self.tip_pos.copy()
        self.tip_pos = new_tip_pos.copy()

    def try_grab(self, debris):
        """
        Attempt to grab debris if arm tip is within grab radius.

        Parameters
        ----------
        debris : OrbitalDebris instance

        Returns
        -------
        bool : True if grab succeeded
        """
        if self.state != IDLE:
            return False

        dist = np.linalg.norm(self.tip_pos - debris.pos)
        if dist <= GRAB_RADIUS:
            debris.grab(self.tip_pos)
            self.state = HOLDING
            return True
        return False

    def try_release(self, debris, deposit_box):
        """
        Release currently held debris.

        Parameters
        ----------
        debris      : OrbitalDebris instance (must be grabbed)
        deposit_box : dict with keys 'center' (np.ndarray) and 'size' (np.ndarray)

        Returns
        -------
        bool : True if the debris was deposited inside the box
        """
        if self.state != HOLDING:
            return False

        debris.release(self.tip_vel)
        self.state = IDLE

        box_center = np.asarray(deposit_box['center'], dtype=float)
        box_size = np.asarray(deposit_box['size'], dtype=float)
        deposited = self.is_in_box(debris.pos, box_center, box_size)
        return deposited

    def is_in_box(self, pos, box_center, box_size):
        """
        Check whether a position is inside the axis-aligned bounding box.

        Parameters
        ----------
        pos        : np.ndarray shape (2,)
        box_center : np.ndarray shape (2,)
        box_size   : np.ndarray shape (2,) [width, height]

        Returns
        -------
        bool
        """
        pos = np.asarray(pos, dtype=float)
        box_center = np.asarray(box_center, dtype=float)
        box_size = np.asarray(box_size, dtype=float)
        half = box_size * 0.5
        return (abs(pos[0] - box_center[0]) <= half[0] and
                abs(pos[1] - box_center[1]) <= half[1])

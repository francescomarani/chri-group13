# -*- coding: utf-8 -*-
"""
Space debris haptic force computation.

Five components:

  1. Debris attraction   — always (APPROACHING)
     1/r² spring toward debris. Lets user feel where debris is.

  2. CW orbital environment — always
     Coriolis + radial-spring on the arm itself (it is also in orbit).

  3. Spin + inertia coupling — GRABBED only
     Spinning debris pushes arm tangentially; inertia resists arm velocity.

  4. Door virtual walls — always (haply_haptics)
     Two stiff parallel walls forming the door channel into the satellite.
     STRONGEST force — clearly felt whenever arm is in channel region.

  5. Box centering — GRABBED + inside satellite
     Centering spring once through the door, guiding debris to box centre.
"""
import numpy as np


class SpaceHaptics:
    """
    Haptic forces for the space debris capture task.

      'mouse'         — zero
      'haply'         — zero
      'haply_haptics' — all five components
    """

    FORCE_CLIP = 2.5          # N per axis — global safety clip

    # 1. Debris attraction
    ATTRACT_K = 0.0006        # N·m²  (1/r² magnitude)
    ATTRACT_RADIUS = 0.10     # m     outer gate
    ATTRACT_CLIP = 0.6        # N

    # 2. CW orbital environment on arm
    CW_N = 0.08               # rad/s
    CW_SCALE = 300.0          # amplifier
    CW_CLIP = 0.4             # N per axis

    # 3. Spin coupling (grabbed) — pure spring, no velocity term
    SPIN_GAIN = 0.6           # N  — tangential only
    MOMENTUM_GAIN = 0.0       # disabled: velocity-based forces cause oscillation
    MOMENTUM_CLIP = 0.0

    # 4. Door virtual walls — stiffness only, no damping
    DOOR_HALF_W = 0.014       # m  — half-width of door opening
    DOOR_Y_BOT = 0.014        # m  — bottom of channel (below satellite)
    DOOR_Y_TOP = 0.042        # m  — slightly above satellite bottom edge
    DOOR_WALL_K = 120.0       # N/m — stable at 60 Hz
    DOOR_WALL_B = 0.0         # damping disabled (causes oscillation at 60 Hz)

    # 5. Box centering (inside satellite, grabbed)
    BOX_WALL_K = 50.0         # N/m
    BOX_WALL_B = 0.0          # damping disabled
    BOX_Y_ENTER = 0.040       # m — y above which "inside satellite" activates

    def __init__(self, deposit_box_center, deposit_box_size, debris):
        self._box_center = np.array(deposit_box_center, dtype=float)
        self._box_half = np.array(deposit_box_size, dtype=float) * 0.5
        self._debris = debris
        self._condition = 'mouse'

        # Public for renderer arrows
        self.last_attract_force = np.zeros(2)
        self.last_orbital_force = np.zeros(2)
        self.last_spin_force = np.zeros(2)
        self.last_door_force = np.zeros(2)
        self.last_box_force = np.zeros(2)

        # Aliases kept for renderer compatibility
        self.last_wall_force = np.zeros(2)

    def set_condition(self, condition):
        self._condition = condition
        self.last_attract_force = np.zeros(2)
        self.last_orbital_force = np.zeros(2)
        self.last_spin_force = np.zeros(2)
        self.last_door_force = np.zeros(2)
        self.last_box_force = np.zeros(2)
        self.last_wall_force = np.zeros(2)

    def compute(self, arm_pos, arm_vel):
        """
        Returns total haptic force (np.ndarray shape (2,), Newtons).
        """
        self.last_attract_force = np.zeros(2)
        self.last_orbital_force = np.zeros(2)
        self.last_spin_force = np.zeros(2)
        self.last_door_force = np.zeros(2)
        self.last_box_force = np.zeros(2)
        self.last_wall_force = np.zeros(2)

        if self._condition != 'haply_haptics':
            return np.zeros(2)

        arm_pos = np.asarray(arm_pos, dtype=float)
        arm_vel = np.asarray(arm_vel, dtype=float)
        debris = self._debris
        total = np.zeros(2)

        # ── 1. Debris attraction (approach only) ──────────────────────
        if not debris.grabbed:
            f = self._debris_attraction(arm_pos, debris)
            self.last_attract_force = f.copy()
            total += f

        # ── 2. CW orbital environment (always) ────────────────────────
        f = self._cw_environment(arm_pos, arm_vel)
        self.last_orbital_force = f.copy()
        total += f

        # ── 3. Spin + inertia (grabbed only) ──────────────────────────
        if debris.grabbed:
            f = self._spin_inertia(arm_vel, debris)
            self.last_spin_force = f.copy()
            total += f

        # ── 4. Door virtual walls (always in haply_haptics) ───────────
        f = self._door_walls(arm_pos, arm_vel)
        self.last_door_force = f.copy()
        self.last_wall_force = f.copy()   # renderer alias
        total += f

        # ── 5. Box centering (grabbed + inside satellite) ─────────────
        if debris.grabbed and arm_pos[1] > self.BOX_Y_ENTER:
            f = self._box_centering(arm_pos, arm_vel)
            self.last_box_force = f.copy()
            total += f

        return np.clip(total, -self.FORCE_CLIP, self.FORCE_CLIP)

    # ── Private helpers ──────────────────────────────────────────────────

    def _debris_attraction(self, arm_pos, debris):
        diff = debris.pos - arm_pos
        dist = np.linalg.norm(diff)
        if dist < 0.005 or dist > self.ATTRACT_RADIUS:
            return np.zeros(2)
        direction = diff / dist
        magnitude = np.clip(self.ATTRACT_K / (dist ** 2), 0.0, self.ATTRACT_CLIP)
        return direction * magnitude

    def _cw_environment(self, arm_pos, arm_vel):
        n = self.CW_N
        x = arm_pos[0]
        vx, vy = arm_vel
        ax = 3.0 * n * n * x + 2.0 * n * vy
        ay = -2.0 * n * vx
        f = np.array([ax, ay]) * self.CW_SCALE
        return np.clip(f, -self.CW_CLIP, self.CW_CLIP)

    def _spin_inertia(self, arm_vel, debris):
        # Tangential push from angular momentum
        tangent = np.array([-np.sin(debris.angle), np.cos(debris.angle)])
        f_spin = tangent * self.SPIN_GAIN
        # Inertia: debris resists arm velocity changes
        f_inertia = np.clip(
            -arm_vel * self.MOMENTUM_GAIN * debris.mass,
            -self.MOMENTUM_CLIP, self.MOMENTUM_CLIP
        )
        return f_spin + f_inertia

    def _door_walls(self, arm_pos, arm_vel):
        """
        Two stiff parallel walls forming the door channel.
        Active when arm y is in [DOOR_Y_BOT, DOOR_Y_TOP].
        Walls at x = ±DOOR_HALF_W — strong spring-damper pushing arm to centre.
        """
        x, y = arm_pos
        vx = arm_vel[0]

        if y < self.DOOR_Y_BOT or y > self.DOOR_Y_TOP:
            return np.zeros(2)

        fx = 0.0
        if x < -self.DOOR_HALF_W:                        # left of left wall
            pen = -self.DOOR_HALF_W - x
            fx = self.DOOR_WALL_K * pen - self.DOOR_WALL_B * min(0.0, vx)
        elif x > self.DOOR_HALF_W:                        # right of right wall
            pen = x - self.DOOR_HALF_W
            fx = -self.DOOR_WALL_K * pen - self.DOOR_WALL_B * max(0.0, vx)

        return np.array([fx, 0.0])

    def _box_centering(self, arm_pos, arm_vel):
        """
        Once inside the satellite (past the door), spring toward deposit
        box centre on x axis, and a soft y spring toward box centre.
        """
        f = np.zeros(2)
        for axis in range(2):
            delta = arm_pos[axis] - self._box_center[axis]
            half = self._box_half[axis]
            if abs(delta) < half:
                # Inside box: centering spring
                f[axis] = (-self.BOX_WALL_K * delta
                           - self.BOX_WALL_B * arm_vel[axis])
            else:
                # Outside box but above door: attract toward box wall
                sign = np.sign(delta)
                pen = abs(delta) - half
                f[axis] = -sign * self.BOX_WALL_K * pen
        return f

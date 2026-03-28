# -*- coding: utf-8 -*-
"""
PA3 — Haptic Force Computation (Gaussian repulsive walls)
----------------------------------------------------------
Key design:
  - Gaussian potential field for walls: smooth, continuous, no state machine
  - Force is a pure function of position → no latch/release/hysteresis needed
  - Bidirectional damping near walls prevents oscillation
"""

import numpy as np


class TubeHaptics:

    def __init__(self, tube,
                 groove_k=350.0,
                 groove_f_max=1.4,   # N — max groove force
                 groove_damping=0.8,   # N·s/m — damps lateral oscillation
                 wall_amplitude=3.0,   # N — peak repulsive force at wall boundary
                 wall_sigma=0.0012,    # m — Gaussian width (controls wall softness)
                 wall_damping=5.0,     # N·s/m — bidirectional damping near walls
                 groove_deadzone=0.0,
                 local_search_window=80,
                 guidance_fade_start=0.55,
                 guidance_fade_end=0.90,
                 total_f_max=2.5
                 ):
        self.tube = tube
        self.groove_k = groove_k
        self.groove_f_max = groove_f_max
        self.groove_damping = groove_damping
        self.wall_amplitude = wall_amplitude
        self.wall_sigma = wall_sigma
        self.wall_damping = wall_damping
        self.groove_deadzone = groove_deadzone
        self.local_search_window = local_search_window
        self.guidance_fade_start = guidance_fade_start
        self.guidance_fade_end = guidance_fade_end
        self.total_f_max = total_f_max

        # ── Independent feature toggles (set from PA3 via H / W keys) ──
        self.groove_enabled = True
        self.walls_enabled = True

        # ── State ──
        self.proxy_pos = None
        self.proxy_idx = 0
        self.contact_wall = None      # kept for compatibility (drawing)
        self.prev_pos = None

        # Cutoff distance: beyond 3σ from wall, Gaussian force is negligible
        self._wall_cutoff = 3.0 * self.wall_sigma


        # ── Learned-trajectory guidance ──
        self.gp_groove_enabled = False
        self.gp_groove_k       = 300.0      # N/m  — max stiffness when fully confident
        self.gp_groove_f_max   = 1.5        # N    — saturating cap
        self.gp_groove_damping = 0.8        # N·s/m — lateral damping
        self.gp_std_min        = 0.001      # m — std below this = full confidence
        self.gp_std_max        = 0.012      # m — std above this = no guidance

        # ── Adaptive centerline guidance ──
        self.fading_groove_enabled = False


        # GP trajectory data (set after training)
        self._gp_traj     = None            # ndarray (N, 2)
        self._gp_traj_std = None            # ndarray (N, 2)
        self._gp_idx      = 0               # local search anchor
        self.n_demos = 0    # updated when set_gp_trajectory is called


        # ── Status outputs ──
        self.last_penetration = 0.0
        self.last_wall = None
        self.last_groove_force = 0.0
        self.last_wall_force = 0.0
        self.last_signed_d = 0.0
        self.last_proxy_pos = None

    def reset_proxy(self):
        self.proxy_pos = None
        self.proxy_idx = 0
        self.contact_wall = None
        self.prev_pos = None
        self.last_penetration = 0.0
        self.last_wall = None
        self.last_proxy_pos = None

    def _local_closest(self, pos):
        tube = self.tube
        w = self.local_search_window
        i_start = max(0, self.proxy_idx - w)
        i_end = min(tube.n_pts, self.proxy_idx + w + 1)
        local_cl = tube.centerline[i_start:i_end]
        dists = np.linalg.norm(local_cl - pos, axis=1)
        local_min = int(np.argmin(dists))
        idx = i_start + local_min
        proj = tube.centerline[idx]
        normal = tube.normals[idx]
        to_pos = pos - proj
        signed_d = float(np.dot(to_pos, normal))
        abs_d = abs(signed_d)
        return idx, proj, normal, signed_d, abs_d

    def _wall_point(self, idx, wall_name):
        """Surface point on the wall (for proxy visualization)."""
        idx = int(np.clip(idx, 0, self.tube.n_pts - 1))
        proj = self.tube.centerline[idx]
        normal = self.tube.normals[idx]
        if wall_name == 'left':
            return proj + normal * self.tube.half_width
        return proj - normal * self.tube.half_width

    def _guidance_gain(self, abs_d, in_wall_contact):
        """
        Fade groove-like guidance out near the walls.

        The centerline cue is useful in free space, but near/at contact it
        fights the wall controller in the same normal direction and can cause
        chatter on the Haply. This keeps the wall controller authoritative.
        """
        if in_wall_contact or self.tube.half_width <= 1e-9:
            return 0.0
        frac = abs_d / self.tube.half_width
        if frac <= self.guidance_fade_start:
            return 1.0
        if frac >= self.guidance_fade_end:
            return 0.0
        span = self.guidance_fade_end - self.guidance_fade_start
        return max(0.0, 1.0 - (frac - self.guidance_fade_start) / max(span, 1e-9))
    
    def set_gp_trajectory(self, traj, std, n_demos=1):
        """Store the learned trajectory and uncertainty for adaptive guidance."""
        self._gp_traj     = np.asarray(traj, dtype=float)
        self._gp_traj_std = np.asarray(std,  dtype=float)
        self._gp_idx      = 0
        self.n_demos = n_demos

    def clear_gp_trajectory(self):
        self._gp_traj     = None
        self._gp_traj_std = None
        self._gp_idx      = 0

    def _learned_guidance_alpha(self, pos):
        """
        Confidence-driven guidance gain in [0, 1].

        0 means "no reliable learned guidance yet".
        1 means "high confidence in the learned trajectory here".
        """
        if self._gp_traj is None or self._gp_traj_std is None or len(self._gp_traj) == 0:
            return 0.0

        w = self.local_search_window
        i_start = max(0, self._gp_idx - w)
        i_end = min(len(self._gp_traj), self._gp_idx + w + 1)
        local = self._gp_traj[i_start:i_end]
        dists = np.linalg.norm(local - pos, axis=1)
        self._gp_idx = i_start + int(np.argmin(dists))

        gp_std = float(np.mean(self._gp_traj_std[self._gp_idx]))
        std_factor = np.clip(
            (self.gp_std_max - gp_std) / (self.gp_std_max - self.gp_std_min), 0.0, 1.0
        )
        demo_factor = 1.0 - np.exp(-(self.n_demos - 1) / 3.0)
        return float(std_factor * demo_factor)


    def compute_force(self, pos_phys, dt=0.01):
        pos = np.asarray(pos_phys, dtype=float)
        tube = self.tube
        fe = np.zeros(2)

        if self.proxy_pos is None:
            idx_g, _, _, _, _ = tube.closest_centerline_point(pos)
            self.proxy_pos = pos.copy()
            self.proxy_idx = idx_g

        idx_g, proj_g, normal_g, signed_d_g, abs_d_g = tube.closest_centerline_point(pos)
        self.last_signed_d = signed_d_g

        # ── VIRTUAL WALLS (Gaussian repulsive field) ────────────────────
        if self.walls_enabled:
            # Distance from wall boundary (positive = inside tube, negative = outside)
            d_wall = tube.half_width - abs_d_g

            if d_wall < self._wall_cutoff:
                # Gaussian repulsive force: smooth, continuous, no state machine
                f_mag = self.wall_amplitude * np.exp(
                    -d_wall ** 2 / (2.0 * self.wall_sigma ** 2)
                )
                # Direction: push toward centerline
                wall_dir = -normal_g if signed_d_g > 0 else normal_g
                fe += f_mag * wall_dir

                # Bidirectional damping (prevents oscillation in both directions)
                if self.prev_pos is not None and dt > 0:
                    vel = (pos - self.prev_pos) / dt
                    vel_normal = np.dot(vel, -wall_dir)  # positive = moving toward wall
                    fe += self.wall_damping * vel_normal * wall_dir

                self.last_wall_force = f_mag

                # Proxy visualization: project onto wall surface
                wall_name = 'left' if signed_d_g > 0 else 'right'
                self.proxy_pos = self._wall_point(idx_g, wall_name)
                self.proxy_idx = idx_g
                self.last_proxy_pos = self.proxy_pos.copy()
                self.last_wall = wall_name
                self.contact_wall = wall_name
                self.last_penetration = max(0.0, -d_wall)  # >0 when outside tube
            else:
                # Far from walls: no wall force
                self.proxy_pos = pos.copy()
                self.proxy_idx = idx_g
                self.contact_wall = None
                self.last_penetration = 0.0
                self.last_wall = None
                self.last_wall_force = 0.0
                self.last_proxy_pos = None
        else:
            self.contact_wall = None
            self.proxy_pos = pos.copy()
            self.proxy_idx = idx_g
            self.last_penetration = 0.0
            self.last_wall = None
            self.last_wall_force = 0.0
            self.last_proxy_pos = None

        wall_contact_active = self.contact_wall is not None
        guidance_gain = self._guidance_gain(abs_d_g, wall_contact_active)

        # ── GROOVE ───────────────────────────────────────────────────────
        if self.groove_enabled and self.groove_k > 0 and guidance_gain > 0.0:
            groove_d = abs_d_g - self.groove_deadzone

            # Groove damping — always active laterally (prevents oscillation)
            if self.prev_pos is not None and dt > 0:
                vel = (pos - self.prev_pos) / dt
                vel_lateral = np.dot(vel, normal_g)
                fe -= self.groove_damping * guidance_gain * vel_lateral * normal_g

            if groove_d > 0:
                groove_dir = -normal_g if signed_d_g > 0 else normal_g
                raw_force = self.groove_k * guidance_gain * groove_d
                capped_force = min(raw_force, self.groove_f_max)
                f_groove = capped_force * groove_dir
                fe += f_groove
                self.last_groove_force = float(np.linalg.norm(f_groove))
            else:
                self.last_groove_force = 0.0
        else:
            self.last_groove_force = 0.0


        learned_alpha = self._learned_guidance_alpha(pos)

        # ── LEARNED-TRAJECTORY GUIDANCE (increases with confidence) ─────────
        if (self.gp_groove_enabled
                and self._gp_traj is not None
                and self.contact_wall is None):

            gp_pt  = self._gp_traj[self._gp_idx]
            k_eff = self.gp_groove_k * learned_alpha

            if k_eff > 1.0:
                displacement = gp_pt - pos       # vector: ee → GP point
                dist_to_gp   = np.linalg.norm(displacement)

                # Groove damping — damp lateral velocity toward/away from GP line
                if self.prev_pos is not None and dt > 0:
                    vel = (pos - self.prev_pos) / dt
                    if dist_to_gp > 1e-8:
                        d_hat = displacement / dist_to_gp
                        vel_lateral = np.dot(vel, d_hat)
                        fe -= self.gp_groove_damping * learned_alpha * vel_lateral * d_hat

                # Saturating spring toward GP point
                raw   = k_eff * dist_to_gp
                capped = min(raw, self.gp_groove_f_max)
                if dist_to_gp > 1e-8:
                    fe += capped * (displacement / dist_to_gp)

        # ── FADING CENTERLINE GUIDANCE (decreases with confidence) ──────────
        if (self.fading_groove_enabled and self.groove_k > 0 and guidance_gain > 0.0):
            # Reuse centerline distance already computed above (signed_d_g, normal_g)
            groove_d = abs_d_g - self.groove_deadzone

            k_eff = self.groove_k * (1.0 - learned_alpha) * guidance_gain

            # Lateral damping (always active, also fades)
            if self.prev_pos is not None and dt > 0:
                vel = (pos - self.prev_pos) / dt
                vel_lateral = np.dot(vel, normal_g)
                fe -= self.groove_damping * (1.0 - learned_alpha) * guidance_gain * vel_lateral * normal_g

            if groove_d > 0 and k_eff > 1.0:
                groove_dir = -normal_g if signed_d_g > 0 else normal_g
                raw_force    = k_eff * groove_d
                capped_force = min(raw_force, self.groove_f_max)
                fe += capped_force * groove_dir

        # Global force cap for stable device behavior when multiple fields are active.
        f_mag = np.linalg.norm(fe)
        if self.total_f_max > 0 and f_mag > self.total_f_max:
            fe = fe * (self.total_f_max / f_mag)

        self.prev_pos = pos.copy()
        return fe

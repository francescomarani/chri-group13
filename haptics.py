# -*- coding: utf-8 -*-
"""
PA3 — Haptic Force Computation (Proxy-based, one-sided spring)
---------------------------------------------------------------
Key design:
  - PA1c-style god-object: wall contact stays latched after first hit
  - Tangential wall motion uses a persistent stick/slip anchor along the curve
  - ONE-SIDED spring: force only when abs_d > half_width (no pull-back = no bounce)
  - Release happens when clearly inside the tube again
"""

import numpy as np


class TubeHaptics:

    def __init__(self, tube,
                 groove_k=350.0,
                 groove_f_max=1.4,   # N — max groove force
                 groove_damping=0.8,   # N·s/m — damps lateral oscillation
                 wall_k=5000.0,
                 wall_damping=4.0,
                 wall_mu_static=0.55,
                 wall_mu_kinetic=0.35,
                 groove_deadzone=0.0,
                 release_hysteresis=0.0008,
                 local_search_window=80,
                 exit_damping=10.0,      # N·s/m — absorbs rebound velocity
                 exit_band=0.001,        # m — viscous layer inside wall
                 guidance_fade_start=0.55,
                 guidance_fade_end=0.90,
                 total_f_max=2.5

                 ):
        self.tube = tube
        self.groove_k = groove_k
        self.groove_f_max = groove_f_max
        self.groove_damping = groove_damping
        self.wall_k = wall_k
        self.wall_damping = wall_damping
        self.wall_mu_static = wall_mu_static
        self.wall_mu_kinetic = wall_mu_kinetic
        self.groove_deadzone = groove_deadzone
        self.release_hysteresis = release_hysteresis
        self.local_search_window = local_search_window
        self.exit_damping = exit_damping
        self.exit_band = exit_band
        self.guidance_fade_start = guidance_fade_start
        self.guidance_fade_end = guidance_fade_end
        self.total_f_max = total_f_max

        
        # ── Independent feature toggles (set from PA3 via H / W keys) ──
        self.groove_enabled = True
        self.walls_enabled = True

        # ── Proxy state ──
        self.proxy_pos = None
        self.proxy_idx = 0
        self.contact_wall = None
        self.prev_pos = None
        self.wall_friction_state = 'free'
        self.wall_stick_idx = None

        diffs = np.diff(self.tube.centerline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._cum_arc = np.zeros(self.tube.n_pts)
        self._cum_arc[1:] = np.cumsum(seg_lengths)


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
        self.wall_friction_state = 'free'
        self.wall_stick_idx = None

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

    def _tangent_at_idx(self, idx):
        normal = self.tube.normals[int(idx)]
        tangent = np.array([normal[1], -normal[0]], dtype=float)
        t_norm = np.linalg.norm(tangent)
        if t_norm > 1e-10:
            tangent /= t_norm
        return tangent

    def _arc_to_idx(self, arc_s):
        arc_s = float(np.clip(arc_s, self._cum_arc[0], self._cum_arc[-1]))
        idx = int(np.searchsorted(self._cum_arc, arc_s, side='left'))
        return min(max(idx, 0), self.tube.n_pts - 1)

    def _wall_point_at_idx(self, idx, wall_name):
        idx = int(np.clip(idx, 0, self.tube.n_pts - 1))
        proj = self.tube.centerline[idx]
        normal = self.tube.normals[idx]
        if wall_name == 'left':
            return proj + normal * self.tube.half_width, normal
        return proj - normal * self.tube.half_width, normal

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

        # ── VIRTUAL WALLS ────────────────────────────────────────────────
        if self.walls_enabled:
            if self.contact_wall is None:
                if abs_d_g <= tube.half_width:
                    # Free: proxy tracks ee
                    self.proxy_pos = pos.copy()
                    self.proxy_idx = idx_g
                    self.last_penetration = 0.0
                    self.last_wall = None
                    self.last_wall_force = 0.0
                    self.last_proxy_pos = None
                else:
                    # First contact: latch wall and initialize tangential anchor,
                    # mirroring the PA1c "hit edge + stiction point" strategy.
                    self.contact_wall = 'left' if signed_d_g > 0 else 'right'
                    self.proxy_idx = idx_g
                    self.wall_stick_idx = idx_g
                    self.wall_friction_state = 'stick'

            if self.contact_wall is not None:
                idx_l, proj_l, normal_l, signed_d_l, abs_d_l = self._local_closest(pos)
                penetration = max(0.0, abs_d_l - tube.half_width)

                if self.wall_stick_idx is None:
                    self.wall_stick_idx = idx_l

                current_arc = self._cum_arc[idx_l]
                stick_arc = self._cum_arc[self.wall_stick_idx]
                tang_disp = current_arc - stick_arc
                f_normal = self.wall_k * penetration
                f_tang = self.wall_k * abs(tang_disp)

                if f_normal > 1e-8:
                    if f_tang > self.wall_mu_static * f_normal:
                        max_arc = (self.wall_mu_kinetic * f_normal) / self.wall_k
                        stick_arc = current_arc - np.sign(tang_disp) * max_arc
                        self.wall_stick_idx = self._arc_to_idx(stick_arc)
                        self.wall_friction_state = 'slip'
                    else:
                        self.wall_friction_state = 'stick'
                else:
                    self.wall_friction_state = 'free'
                    self.wall_stick_idx = idx_l

                self.proxy_idx = self.wall_stick_idx
                self.proxy_pos, proxy_normal = self._wall_point_at_idx(
                    self.proxy_idx, self.contact_wall
                )

                self.last_proxy_pos = self.proxy_pos.copy()
                self.last_penetration = penetration
                self.last_wall = self.contact_wall

                # One-sided spring: only push when outside
                if penetration > 0:
                    displacement = self.proxy_pos - pos
                    f_wall = self.wall_k * displacement
                    if self.prev_pos is not None and dt > 0:
                        vel = (pos - self.prev_pos) / dt
                        wall_normal = proxy_normal if self.contact_wall == 'left' else -proxy_normal
                        vel_deeper = np.dot(vel, wall_normal)
                        if vel_deeper > 0:
                            f_wall -= self.wall_damping * vel_deeper * wall_normal
                    fe += f_wall
                    self.last_wall_force = float(np.linalg.norm(f_wall))
                else:
                    self.last_wall_force = 0.0

                # Release when clearly inside
                if abs_d_g < tube.half_width - self.release_hysteresis:
                    self.proxy_pos = pos.copy()
                    self.proxy_idx = idx_g
                    self.contact_wall = None
                    self.wall_friction_state = 'free'
                    self.wall_stick_idx = None
                    self.last_penetration = 0.0
                    self.last_wall = None
                    self.last_wall_force = 0.0
                    self.last_proxy_pos = None
                    fe = np.zeros(2)

        else:
            # Walls disabled: release any contact, proxy follows ee
            self.contact_wall = None
            self.proxy_pos = pos.copy()
            self.proxy_idx = idx_g
            self.wall_friction_state = 'free'
            self.wall_stick_idx = None
            self.last_penetration = 0.0
            self.last_wall = None
            self.last_wall_force = 0.0
            self.last_proxy_pos = None

        # ── EXIT DAMPING: viscous band just inside wall ──────────────────
        depth_inside = tube.half_width - abs_d_g
        if 0 <= depth_inside < self.exit_band and self.prev_pos is not None and dt > 0:
            vel = (pos - self.prev_pos) / dt
            vel_normal = np.dot(vel, normal_g)
            fe -= self.exit_damping * vel_normal * normal_g

        wall_contact_active = self.contact_wall is not None or self.last_penetration > 0.0
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

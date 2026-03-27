# -*- coding: utf-8 -*-
"""
Pygame-based renderer for the space debris capture simulation.
Space aesthetic, 900×700 window, single panel.
"""
import math
from collections import deque

import numpy as np
import pygame


# ── Coordinate system ──────────────────────────────────────────────────────
WINDOW_W = 900
WINDOW_H = 700
CENTER_X = 450
CENTER_Y = 380
SCALE = 2800  # px/m

# ── Colors ──────────────────────────────────────────────────────────────────
COL_BG = (5, 5, 16)
COL_SATELLITE = (160, 168, 176)
COL_SATELLITE_DARK = (100, 108, 116)   # inner shadow
COL_SOLAR_PANEL = (220, 200, 50)
COL_BOX_FILL = (30, 40, 55)            # dark interior of deposit box
COL_BOX_NORMAL = (80, 160, 220)        # box outline
COL_BOX_PULSE = (0, 255, 80)
COL_DOOR_WALL = (60, 180, 255)         # door channel walls
COL_DOOR_WALL_ACTIVE = (0, 255, 180)   # door walls glow when arm is in channel
COL_DOOR_CHANNEL = (10, 30, 50)        # door channel interior
COL_ARM = (255, 255, 255)
COL_TIP = (0, 220, 220)
COL_GRAB_CIRCLE = (0, 220, 220)
COL_DEBRIS_FREE = (255, 140, 0)
COL_DEBRIS_GRABBED = (0, 220, 80)
COL_TRAIL_DEBRIS = (255, 140, 0)
COL_TRAIL_ARM = (0, 220, 220)
COL_FORCE_ATTRACT = (255, 80, 80)      # red — debris attraction
COL_FORCE_ORBITAL = (220, 50, 50)
COL_FORCE_SPIN = (220, 200, 50)        # yellow — spin
COL_FORCE_WALL = (50, 100, 220)        # blue — door / box
COL_HUD = (200, 200, 200)
COL_HUD_TITLE = (100, 200, 255)
COL_SUCCESS = (0, 255, 100)

FORCE_ARROW_SCALE = 20  # px per Newton


class Renderer:
    """
    Pygame renderer for the space debris capture simulation.
    """

    def __init__(self, width=WINDOW_W, height=WINDOW_H):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Space Debris Capture")

        self.font_small = pygame.font.SysFont("Courier", 13)
        self.font_med = pygame.font.SysFont("Courier", 16, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 22, bold=True)

        # Trails stored as deques
        self.debris_trail = deque(maxlen=60)
        self.arm_trail = deque(maxlen=40)

        # Pre-generate star field (fixed, seed=42)
        rng = np.random.default_rng(42)
        self._star_x = rng.integers(0, width, 150).tolist()
        self._star_y = rng.integers(0, height, 150).tolist()
        self._star_r = rng.integers(1, 3, 150).tolist()
        brightness = rng.integers(160, 256, 150)
        self._star_colors = [(b, b, b) for b in brightness]

        # Clock
        self.clock = pygame.time.Clock()

        # Pre-render background once
        self._bg_surface = pygame.Surface((width, height))
        self._bg_surface.fill(COL_BG)
        for i in range(150):
            pygame.draw.circle(self._bg_surface, self._star_colors[i],
                               (self._star_x[i], self._star_y[i]),
                               self._star_r[i])

    # ── Coordinate conversion ───────────────────────────────────────────────

    def p2s(self, pos):
        """Convert physical (meters) to screen (pixels)."""
        sx = int(CENTER_X + pos[0] * SCALE)
        sy = int(CENTER_Y - pos[1] * SCALE)
        return (sx, sy)

    def s2p(self, screen_pos):
        """Convert screen (pixels) to physical (meters)."""
        px = (screen_pos[0] - CENTER_X) / SCALE
        py = (CENTER_Y - screen_pos[1]) / SCALE
        return np.array([px, py])

    # ── Frame lifecycle ─────────────────────────────────────────────────────

    def begin_frame(self):
        """Blit pre-rendered background (stars + dark sky)."""
        self.screen.blit(self._bg_surface, (0, 0))

    def end_frame(self):
        """Flip display buffer."""
        pygame.display.flip()

    # ── Drawing methods ─────────────────────────────────────────────────────

    def draw_scene(self, debris, arm, deposit_box, haptics, phase, condition, gp_policy):
        """
        Draw all scene elements.

        Parameters
        ----------
        debris      : OrbitalDebris
        arm         : Arm
        deposit_box : dict with 'center' and 'size' (np.ndarray)
        haptics     : SpaceHaptics
        phase       : str — 'APPROACHING' | 'GRABBED' | 'DEPOSITING' | 'SUCCESS'
        condition   : str — 'mouse' | 'haply' | 'haply_haptics'
        gp_policy   : GPPolicy
        """
        box_center = np.asarray(deposit_box['center'], dtype=float)
        box_size = np.asarray(deposit_box['size'], dtype=float)
        satellite_center = np.asarray(deposit_box.get('satellite_center',
                                                        box_center), dtype=float)
        satellite_size = np.asarray(deposit_box.get('satellite_size',
                                                      np.array([0.05, 0.03])), dtype=float)

        # Update trails
        self.debris_trail.append(debris.pos)
        self.arm_trail.append(arm.tip_pos)

        door_half_w = float(deposit_box.get('door_half_w', 0.014))
        door_y_bot = float(deposit_box.get('door_y_bot', 0.014))
        door_y_top = float(deposit_box.get('door_y_top', 0.042))

        # Arm in door channel?
        ax, ay = arm.tip_pos
        arm_in_channel = (door_y_bot < ay < door_y_top
                          and abs(ax) < door_half_w + 0.01)

        # 1. Draw trails
        self._draw_debris_trail()
        self._draw_arm_trail()

        # 2. Draw satellite with door, box, channel
        dist_to_box = np.linalg.norm(arm.tip_pos - box_center)
        self._draw_satellite_with_door(
            satellite_center, satellite_size,
            door_half_w, door_y_bot, door_y_top,
            box_center, box_size,
            dist_to_box, phase, arm_in_channel,
        )

        # 3. Draw arm
        self._draw_arm(arm, phase)

        # 4. Grab radius indicator (when not grabbed)
        if phase == 'APPROACHING' and debris is not None:
            self._draw_grab_indicator(arm.tip_pos, arm.grab_radius)

        # 5. Draw debris
        if debris is not None:
            self._draw_debris(debris)

        # 6. Force vectors
        if condition == 'haply_haptics' and haptics is not None:
            self._draw_force_vectors(arm.tip_pos, haptics)

    def _draw_satellite_with_door(self, center, size, door_half_w,
                                   door_y_bot, door_y_top,
                                   box_center, box_size,
                                   dist_to_box, phase, arm_in_channel):
        """
        Draw satellite body with door opening, door channel walls,
        deposit box interior, and solar panels.
        """
        cx, cy = self.p2s(center)
        half_w = int(size[0] * SCALE)
        half_h = int(size[1] * SCALE)

        door_w_px = int(door_half_w * SCALE)        # half-door in pixels
        door_bot_sy = self.p2s([0, door_y_bot])[1]  # screen y of channel bottom
        door_top_sy = self.p2s([0, door_y_top])[1]  # screen y of channel top
        sat_bot_sy = cy + half_h                     # satellite bottom edge on screen

        # ── Deposit box interior (drawn first, behind satellite walls) ────
        bx, by = self.p2s(box_center)
        bhw = int(box_size[0] * SCALE)
        bhh = int(box_size[1] * SCALE)
        box_rect = pygame.Rect(bx - bhw, by - bhh, bhw * 2, bhh * 2)
        pygame.draw.rect(self.screen, COL_BOX_FILL, box_rect)

        # Box outline — pulses green when near or on success
        PULSE_DIST = 0.05
        if phase == 'SUCCESS':
            box_col = COL_SUCCESS
            box_thick = 3
        elif dist_to_box < PULSE_DIST:
            t = 1.0 - dist_to_box / PULSE_DIST
            box_col = tuple(int(COL_BOX_NORMAL[i] * (1 - t) + COL_BOX_PULSE[i] * t)
                            for i in range(3))
            box_thick = 2
        else:
            box_col = COL_BOX_NORMAL
            box_thick = 1
        pygame.draw.rect(self.screen, box_col, box_rect, box_thick)

        # ── Door channel interior (dark corridor below satellite) ─────────
        channel_rect = pygame.Rect(cx - door_w_px, door_top_sy,
                                   door_w_px * 2, door_bot_sy - door_top_sy)
        pygame.draw.rect(self.screen, COL_DOOR_CHANNEL, channel_rect)

        # ── Satellite body — two halves flanking the door opening ─────────
        # Left half
        left_rect = pygame.Rect(cx - half_w, cy - half_h,
                                half_w - door_w_px, half_h * 2)
        pygame.draw.rect(self.screen, COL_SATELLITE, left_rect)
        # Right half
        right_rect = pygame.Rect(cx + door_w_px, cy - half_h,
                                 half_w - door_w_px, half_h * 2)
        pygame.draw.rect(self.screen, COL_SATELLITE, right_rect)
        # Top strip (full width, above bottom quarter so door shows through)
        top_h = max(half_h - int(0.008 * SCALE), 4)
        top_rect = pygame.Rect(cx - half_w, cy - half_h, half_w * 2, top_h)
        pygame.draw.rect(self.screen, COL_SATELLITE, top_rect)

        # Satellite outline
        sat_rect = pygame.Rect(cx - half_w, cy - half_h, half_w * 2, half_h * 2)
        pygame.draw.rect(self.screen, (210, 215, 220), sat_rect, 1)

        # ── Door channel walls (left and right of opening) ────────────────
        wall_thick = max(int(0.003 * SCALE), 3)
        door_col = COL_DOOR_WALL_ACTIVE if arm_in_channel else COL_DOOR_WALL

        # Left door wall
        lw_rect = pygame.Rect(cx - door_w_px - wall_thick, door_top_sy,
                              wall_thick, door_bot_sy - door_top_sy)
        pygame.draw.rect(self.screen, door_col, lw_rect)
        # Right door wall
        rw_rect = pygame.Rect(cx + door_w_px, door_top_sy,
                              wall_thick, door_bot_sy - door_top_sy)
        pygame.draw.rect(self.screen, door_col, rw_rect)

        # Inner edge markers at satellite bottom (door frame)
        frame_col = (255, 255, 180) if arm_in_channel else (180, 200, 220)
        # Left frame corner
        pygame.draw.line(self.screen, frame_col,
                         (cx - door_w_px, sat_bot_sy),
                         (cx - door_w_px, door_top_sy), 1)
        # Right frame corner
        pygame.draw.line(self.screen, frame_col,
                         (cx + door_w_px, sat_bot_sy),
                         (cx + door_w_px, door_top_sy), 1)

        # ── Solar panels ──────────────────────────────────────────────────
        panel_w = int(0.04 * SCALE)
        panel_h = max(int(0.008 * SCALE), 4)
        panel_y = cy - panel_h // 2

        left_panel = pygame.Rect(cx - half_w - panel_w, panel_y, panel_w, panel_h)
        pygame.draw.rect(self.screen, COL_SOLAR_PANEL, left_panel)
        pygame.draw.rect(self.screen, (180, 160, 30), left_panel, 1)

        right_panel = pygame.Rect(cx + half_w, panel_y, panel_w, panel_h)
        pygame.draw.rect(self.screen, COL_SOLAR_PANEL, right_panel)
        pygame.draw.rect(self.screen, (180, 160, 30), right_panel, 1)

    def _draw_arm(self, arm, phase):
        """Draw arm line and tip circle."""
        shoulder_s = self.p2s(arm.shoulder_pos)
        tip_s = self.p2s(arm.tip_pos)

        # Arm line
        pygame.draw.line(self.screen, COL_ARM, shoulder_s, tip_s, 2)

        # Tip circle
        tip_color = COL_SUCCESS if phase == 'SUCCESS' else COL_TIP
        pygame.draw.circle(self.screen, tip_color, tip_s, 4)

    def _draw_grab_indicator(self, tip_pos, grab_radius):
        """Draw transparent circle showing grab range around arm tip."""
        tip_s = self.p2s(tip_pos)
        radius_px = int(grab_radius * SCALE)
        if radius_px < 2:
            return

        surf = pygame.Surface((radius_px * 2 + 2, radius_px * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (*COL_GRAB_CIRCLE, 80), (radius_px + 1, radius_px + 1),
                           radius_px, 1)
        self.screen.blit(surf, (tip_s[0] - radius_px - 1, tip_s[1] - radius_px - 1))

    def _draw_debris(self, debris):
        """Draw spinning hexagonal debris with spin indicator line."""
        center_s = self.p2s(debris.pos)
        radius_px = int(debris.radius * SCALE)
        color = COL_DEBRIS_GRABBED if debris.grabbed else COL_DEBRIS_FREE

        # Hexagon vertices
        vertices = []
        for i in range(6):
            angle = debris.angle + i * math.pi / 3.0
            vx = debris.pos[0] + debris.radius * math.cos(angle)
            vy = debris.pos[1] + debris.radius * math.sin(angle)
            vertices.append(self.p2s(np.array([vx, vy])))

        if len(vertices) >= 3:
            pygame.draw.polygon(self.screen, color, vertices)
            pygame.draw.polygon(self.screen, (255, 255, 255), vertices, 1)

        # Spin indicator line
        indicator_len = debris.radius * 1.2
        end_x = debris.pos[0] + indicator_len * math.cos(debris.angle)
        end_y = debris.pos[1] + indicator_len * math.sin(debris.angle)
        end_s = self.p2s(np.array([end_x, end_y]))
        pygame.draw.line(self.screen, (255, 255, 255), center_s, end_s, 1)

    def _draw_debris_trail(self):
        """Draw fading orange trail for debris (single alpha surface)."""
        trail = list(self.debris_trail)
        n = len(trail)
        if n < 2:
            return
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for i in range(1, n):
            alpha = int(150 * i / n)
            color = (*COL_TRAIL_DEBRIS, alpha)
            p1 = self.p2s(trail[i - 1])
            p2 = self.p2s(trail[i])
            pygame.draw.line(surf, color, p1, p2, 2)
        self.screen.blit(surf, (0, 0))

    def _draw_arm_trail(self):
        """Draw fading cyan trail for arm tip (single alpha surface)."""
        trail = list(self.arm_trail)
        n = len(trail)
        if n < 2:
            return
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for i in range(1, n):
            alpha = int(120 * i / n)
            color = (*COL_TRAIL_ARM, alpha)
            p1 = self.p2s(trail[i - 1])
            p2 = self.p2s(trail[i])
            pygame.draw.line(surf, color, p1, p2, 1)
        self.screen.blit(surf, (0, 0))

    def _draw_force_vectors(self, arm_pos, haptics):
        """Draw force arrows from arm tip for each haptic component."""
        tip_s = self.p2s(arm_pos)
        self._draw_arrow(tip_s, haptics.last_attract_force, COL_FORCE_ATTRACT)
        self._draw_arrow(tip_s, haptics.last_orbital_force, COL_FORCE_ORBITAL)
        self._draw_arrow(tip_s, haptics.last_spin_force, COL_FORCE_SPIN)
        self._draw_arrow(tip_s, haptics.last_door_force, COL_FORCE_WALL)
        self._draw_arrow(tip_s, haptics.last_box_force, (100, 200, 255))

    def _draw_arrow(self, origin, force, color):
        """Draw a force arrow from origin in direction/magnitude of force."""
        fx, fy = float(force[0]), float(force[1])
        if abs(fx) < 0.01 and abs(fy) < 0.01:
            return

        end_x = int(origin[0] + fx * FORCE_ARROW_SCALE)
        end_y = int(origin[1] - fy * FORCE_ARROW_SCALE)  # y flipped
        end = (end_x, end_y)

        pygame.draw.line(self.screen, color, origin, end, 2)

        # Arrowhead
        dx = end_x - origin[0]
        dy = end_y - origin[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1:
            return
        ux, uy = dx / length, dy / length
        # Perpendicular
        px, py = -uy, ux
        head_len = 8
        head_w = 4
        tip1 = (int(end_x - ux * head_len + px * head_w),
                int(end_y - uy * head_len + py * head_w))
        tip2 = (int(end_x - ux * head_len - px * head_w),
                int(end_y - uy * head_len - py * head_w))
        pygame.draw.polygon(self.screen, color, [end, tip1, tip2])

    def draw_hud(self, phase, condition, n_demos, gp_trained, gp_confidence=None):
        """
        Draw HUD overlay.

        Parameters
        ----------
        phase          : str — current task phase
        condition      : str — experimental condition
        n_demos        : int — number of recorded demonstrations
        gp_trained     : bool
        gp_confidence  : float or None — GP confidence for confidence bar
        """
        # ── Top-left HUD ────────────────────────────────────────────────
        phase_colors = {
            'APPROACHING': (200, 200, 200),
            'GRABBED': (0, 220, 80),
            'DEPOSITING': (0, 180, 255),
            'SUCCESS': (0, 255, 100),
            'FAIL': (255, 80, 80),
        }
        phase_color = phase_colors.get(phase, (200, 200, 200))

        condition_labels = {
            'mouse': 'MOUSE',
            'haply': 'HAPLY',
            'haply_haptics': 'HAPLY+HAPTICS',
        }
        condition_label = condition_labels.get(condition, condition.upper())

        gp_label = "GP: TRAINED" if gp_trained else "GP: NOT TRAINED"
        gp_color = (0, 220, 80) if gp_trained else (200, 80, 80)

        lines = [
            (f"Phase: {phase}", phase_color),
            (f"Condition: {condition_label}", COL_HUD),
            (f"Demos: {n_demos}", COL_HUD),
            (gp_label, gp_color),
        ]

        x, y = 10, 10
        pad = 4
        line_h = self.font_small.get_linesize()
        box_w = 220
        box_h = len(lines) * line_h + pad * 2

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        self.screen.blit(bg, (x - pad, y - pad))

        for i, (text, color) in enumerate(lines):
            surf = self.font_small.render(text, True, color)
            self.screen.blit(surf, (x, y + i * line_h))

        # ── Top-right confidence bar ─────────────────────────────────────
        if gp_confidence is not None:
            bar_w, bar_h = 120, 12
            bar_x = self.width - bar_w - 10
            bar_y = 10
            confidence = float(np.clip(gp_confidence, 0.0, 1.0))

            # Background
            pygame.draw.rect(self.screen, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))

            # Fill: green (high confidence) → red (low confidence)
            fill_w = int(bar_w * confidence)
            fill_color = (int(255 * (1 - confidence)), int(255 * confidence), 0)
            if fill_w > 0:
                pygame.draw.rect(self.screen, fill_color, (bar_x, bar_y, fill_w, bar_h))

            # Border
            pygame.draw.rect(self.screen, (180, 180, 180), (bar_x, bar_y, bar_w, bar_h), 1)

            # Label
            label = self.font_small.render(
                f"GP conf: {confidence * 100:.0f}%", True, (220, 220, 220))
            self.screen.blit(label, (bar_x - label.get_width() - 6, bar_y))

        # ── Key legend (bottom-right) ────────────────────────────────────
        key_lines = [
            "── Keys ────────────",
            "SPACE  grab / drop",
            "R      record demo",
            "G      train GP",
            "T      GP confidence",
            "H      haptics panel",
            "N      cycle cond.",
            "Q/ESC  quit",
        ]
        kx = self.width - 180
        ky = self.height - len(key_lines) * self.font_small.get_linesize() - 10
        bg2 = pygame.Surface((175, len(key_lines) * self.font_small.get_linesize() + 8),
                              pygame.SRCALPHA)
        bg2.fill((0, 0, 0, 130))
        self.screen.blit(bg2, (kx - 4, ky - 4))
        for i, kline in enumerate(key_lines):
            col = (255, 220, 80) if i == 0 else (180, 180, 180)
            self.screen.blit(self.font_small.render(kline, True, col),
                             (kx, ky + i * self.font_small.get_linesize()))

    def draw_haptics_panel(self, haptics):
        """
        Draw a live haptic-parameter tuning panel (bottom-left).
        Shows current values and the keys used to adjust each one.
        """
        params = [
            ("[ / ]", "Door wall K",  f"{haptics.DOOR_WALL_K:6.1f} N/m"),
            (", / .", "Attract K",    f"{haptics.ATTRACT_K:.5f}"),
            ("; / '", "Spin gain",    f"{haptics.SPIN_GAIN:5.2f} N"),
            ("- / =", "Force clip",   f"{haptics.FORCE_CLIP:4.2f} N"),
            ("BKSP",  "Reset all",    ""),
        ]
        line_h = self.font_small.get_linesize()
        pad = 6
        panel_w = 260
        panel_h = len(params) * line_h + pad * 2 + line_h + 4

        px = 10
        py = self.height - panel_h - 10

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        self.screen.blit(bg, (px, py))

        # Title
        title = self.font_small.render("── Haptics tuning (H to hide) ──",
                                       True, (255, 220, 80))
        self.screen.blit(title, (px + pad, py + pad))

        for i, (key, label, value) in enumerate(params):
            y = py + pad + line_h + 4 + i * line_h
            key_surf = self.font_small.render(f"{key:<7}", True, (255, 220, 80))
            lbl_surf = self.font_small.render(f"{label:<14}", True, (180, 180, 180))
            val_surf = self.font_small.render(value, True, (0, 220, 180))
            self.screen.blit(key_surf, (px + pad, y))
            self.screen.blit(lbl_surf, (px + pad + 52, y))
            self.screen.blit(val_surf, (px + pad + 52 + 100, y))

    def clear_trails(self):
        """Clear both trails (call on reset)."""
        self.debris_trail.clear()
        self.arm_trail.clear()

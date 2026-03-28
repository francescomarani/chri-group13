# -*- coding: utf-8 -*-
"""
PA3 — Kinesthetic Teaching for Hull-Breach Sealing
--------------------------------------------------
Five explicit experimental conditions are supported:

1. Haply device, no feedback
2. Haply device, virtual walls only
3. Haply device, fixed centerline guidance
4. Haply device, fading centerline guidance
5. Haply device, increasing learned-trajectory guidance

If the Haply device is unavailable, mouse input is used only as an automatic fallback.
"""

import argparse
import csv
import sys, time, os, json
from pathlib import Path
import tempfile
import numpy as np
import pygame

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib-codex"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.Physics import Physics
from scripts.Graphics import Graphics
from scripts.targets import get_tube, TUBE_NAMES
from scripts.haptics import TubeHaptics
from scripts.gp_trajectory import TrajectoryGP
from scripts.metrics import (
    average_pairwise_frechet,
    mean_jerk_magnitude,
    mean_nearest_distance,
    path_length_ratio,
)
from scripts.nasa_tlx import run_nasa_tlx

IDLE, RECORDING, REVIEW, TRAINING, PLAYBACK, DONE, AUTO_PLAY = range(7)
STATE_NAMES = ["IDLE", "RECORDING", "REVIEW", "TRAINING", "PLAYBACK", "DONE", "AUTO_PLAY:"]

DEMO_COLORS = [
    (220, 60, 60), (60, 60, 220), (220, 160, 0),
    (160, 0, 220), (0, 180, 180), (180, 100, 60),
    (100, 200, 60), (200, 60, 160),
]

CONDITION_CODE_TO_ID = {
    "NH": 1,
    "VW": 2,
    "CG": 3,
    "FG": 4,
    "SG": 5,
}

GROUP_CONDITION_CODES = {
    "A": ["NH", "VW", "CG", "FG", "SG"],
    "B": ["VW", "CG", "FG", "SG", "NH"],
    "C": ["CG", "FG", "SG", "NH", "VW"],
    "D": ["FG", "SG", "NH", "VW", "CG"],
    "E": ["SG", "NH", "VW", "CG", "FG"],
}

GROUP_CONDITION_ORDERS = {
    group: [CONDITION_CODE_TO_ID[code] for code in codes]
    for group, codes in GROUP_CONDITION_CODES.items()
}

CONDITION_SPECS = {
    1: {
        "slug": "haply_no_feedback",
        "label": "Haply device, no feedback",
        "input_mode": "device",
        "feedback_mode": "none",
        "walls": False,
        "centerline": False,
        "centerline_fading": False,
        "learned_guidance": False,
        "auto_retrain": False,
    },
    2: {
        "slug": "haply_virtual_walls_only",
        "label": "Haply device, virtual walls only",
        "input_mode": "device",
        "feedback_mode": "walls_only",
        "walls": True,
        "centerline": False,
        "centerline_fading": False,
        "learned_guidance": False,
        "auto_retrain": False,
        "guidance_fade_start": 0.55,
        "guidance_fade_end": 0.90,
    },
    3: {
        "slug": "haply_fixed_centerline_guidance",
        "label": "Haply device, fixed central guide + walls",
        "input_mode": "device",
        "feedback_mode": "fixed_centerline_guidance",
        "walls": True,
        "centerline": True,
        "centerline_fading": False,
        "learned_guidance": False,
        "auto_retrain": False,
        "guidance_fade_start": 0.75,
        "guidance_fade_end": 0.95,
    },
    4: {
        "slug": "haply_fading_centerline_guidance",
        "label": "Haply device, fading central guide + walls",
        "input_mode": "device",
        "feedback_mode": "fading_centerline_guidance",
        "walls": True,
        "centerline": False,
        "centerline_fading": True,
        "learned_guidance": False,
        "auto_retrain": True,
        "guidance_fade_start": 0.75,
        "guidance_fade_end": 0.95,
    },
    5: {
        "slug": "haply_increasing_learned_guidance",
        "label": "Haply device, increasing learned guidance + walls",
        "input_mode": "device",
        "feedback_mode": "increasing_learned_guidance",
        "walls": True,
        "centerline": False,
        "centerline_fading": False,
        "learned_guidance": True,
        "auto_retrain": True,
        "guidance_fade_start": 0.75,
        "guidance_fade_end": 0.95,
    },
}


def _prompt_positive_int(prompt_text, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt_text}{suffix}: ").strip()
        if not raw and default is not None:
            return int(default)
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a positive integer.")


def _prompt_mode(default="free"):
    default = "validation" if str(default).lower().startswith("v") else "free"
    prompt = "Select mode: [f]ree or [v]alidation"
    default_hint = "f" if default == "free" else "v"
    while True:
        raw = input(f"{prompt} [{default_hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("f", "free"):
            return "free"
        if raw in ("v", "validation"):
            return "validation"
        print("Please enter 'f' for free or 'v' for validation.")


def _prompt_validation_group(default="A"):
    default = str(default).upper()
    if default not in GROUP_CONDITION_ORDERS:
        default = "A"
    while True:
        raw = input(f"Counterbalance group [A-E] [{default}]: ").strip().upper()
        if not raw:
            return default
        if raw in GROUP_CONDITION_ORDERS:
            return raw
        print("Please enter one of: A, B, C, D, E.")


def parse_runtime_config():
    parser = argparse.ArgumentParser(description="PA3 Hull-Breach Teaching")
    parser.add_argument("--mode", choices=["free", "validation"], default=None)
    parser.add_argument("--participant-number", type=int, default=None)
    parser.add_argument("--participant-count", type=int, dest="participant_count_legacy", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--group", choices=sorted(GROUP_CONDITION_ORDERS.keys()), default=None)
    parser.add_argument("--required-demos", type=int, default=None)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--analysis-dir", default="analysis")
    args = parser.parse_args()

    if args.mode is None:
        args.mode = _prompt_mode(default="free")

    if args.participant_number is None and args.participant_count_legacy is not None:
        args.participant_number = args.participant_count_legacy

    if args.participant_number is not None and args.participant_number <= 0:
        parser.error("--participant-number must be > 0")
    if args.required_demos is not None and args.required_demos <= 0:
        parser.error("--required-demos must be > 0")

    if args.mode == "validation":
        if args.participant_number is None:
            args.participant_number = _prompt_positive_int("Participant number", default=1)
        if args.group is None:
            args.group = _prompt_validation_group(default="A")
        if args.required_demos is None:
            args.required_demos = 3
    else:
        args.participant_number = 1
        args.group = "A"
        args.required_demos = args.required_demos or 0

    return args


class PA3_Kinesthetic:
    def __init__(self, config):
        self.config = config
        self.mode = config.mode
        self.validation_mode = (self.mode == "validation")
        self.required_demos = int(config.required_demos or 0)
        self.participant_count = 1
        self.current_participant_offset = 0
        self.participant_number = int(config.participant_number or 1)
        self.validation_group = str(getattr(config, "group", "A") or "A").upper()
        self.results_dir = Path(config.results_dir)
        self.analysis_dir = Path(config.analysis_dir)
        self.run_id = int(time.time())
        self.session_id = self.run_id
        self.validation_root = self.results_dir / f"validation_run_{self.run_id}"
        self.session_folder = self.results_dir / f"session_{self.session_id}"
        self.auto_analysis_status = None
        self.pending_validation_finalize = False
        self.validation_complete = False
        self.condition_order = self._condition_order_for_current_mode()
        self.condition_cursor = 0

        self.physics = Physics(hardware_version=3)
        self.device_connected = self.physics.is_device_connected()
        self.graphics = Graphics(self.device_connected, window_size=(760, 720))
        self.graphics.show_debug = False
        pygame.display.set_caption("PA3 — Kinesthetic Teaching (Mars Crack)")

        # ── Tube ──
        self.tube_idx = 0
        self.tube_name = TUBE_NAMES[self.tube_idx]
        self.tube = get_tube(self.tube_name)
        self.haptics = TubeHaptics(self.tube)

        # ── State ──
        self.state = IDLE
        self.all_demos = []
        self.all_demo_times = []
        self.per_demo_metrics = []
        self.per_demo_wall_hits = []
        self.current_demo = []
        self.start_time = 0.0
        self.last_demo_time = 0.0
        self.wall_hits = 0
        self.wall_contact_active = False
        self.auto_train = False


        # ── GP ──
        self.gp = TrajectoryGP()
        self.gp_traj_phys = None
        self.gp_traj_std = None
        self.playback_idx = 0
        self.trial_metrics = None
        self.tlx_result = None
        self.all_results = []

        # ── Auto-play PD controller ──
        self.auto_ref_idx = 0.0       # closest + lookahead (for visualization only)
        self.pd_kp = 170.0     # N/m  — position gain
        self.pd_kd = 50.0       # N·s/m — velocity damping
        self.pd_speed = 0.75
        self.prev_ee_phys = None      # was missing — needed by _compute_pd_force
        self.condition_id = self.condition_order[self.condition_cursor]
        self.condition_spec = None
        self._apply_condition(self.condition_id)

    def _participant_label(self):
        return f"participant_{self.participant_number:02d}"

    def _condition_order_for_current_mode(self):
        if self.validation_mode:
            return list(GROUP_CONDITION_ORDERS[self.validation_group])
        return sorted(CONDITION_SPECS.keys())

    def _current_participant_dir(self):
        if self.validation_mode:
            return self.validation_root / self._participant_label()
        return self.session_folder

    def _current_condition_dir(self):
        if self.validation_mode:
            return self._current_participant_dir() / f"condition_{self.condition_id}_{self.condition_spec['slug']}"
        return self.session_folder

    def _reset_condition_state(self):
        self.state = IDLE
        self.all_demos = []
        self.all_demo_times = []
        self.per_demo_metrics = []
        self.per_demo_wall_hits = []
        self.current_demo = []
        self.start_time = 0.0
        self.last_demo_time = 0.0
        self.wall_hits = 0
        self.wall_contact_active = False
        self.auto_train = False
        self.pending_validation_finalize = False
        self.gp_traj_phys = None
        self.gp_traj_std = None
        self.playback_idx = 0
        self.trial_metrics = None
        self.tlx_result = None
        self.auto_ref_idx = 0.0
        self.prev_ee_phys = None
        self.haptics.reset_proxy()
        self.haptics.clear_gp_trajectory()

    def _save_participant_summary(self):
        if not self.validation_mode or not self.all_results:
            return
        participant_dir = self._current_participant_dir()
        participant_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_id": self.run_id,
            "participant_number": self.participant_number,
            "participant_count": self.participant_count,
            "validation_group": self.validation_group,
            "condition_order": self.condition_order,
            "required_demos_target": self.required_demos,
            "n_conditions_completed": len(self.all_results),
            "conditions_completed": [result["condition"] for result in self.all_results],
        }
        with open(participant_dir / "participant_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(participant_dir / "metrics.json", "w") as f:
            json.dump(self.all_results, f, indent=2, default=str)
        self._write_metrics_csv(participant_dir / "metrics.csv", self.all_results)

    def _save_validation_run_summary(self):
        if not self.validation_mode:
            return
        self.validation_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_id": self.run_id,
            "participant_number": self.participant_number,
            "participant_count": 1,
            "validation_group": self.validation_group,
            "required_demos_target": self.required_demos,
            "conditions_per_participant": self.condition_order,
            "n_completed_trials": len(self.all_results),
        }
        with open(self.validation_root / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(self.validation_root / "all_metrics.json", "w") as f:
            json.dump(self.all_results, f, indent=2, default=str)
        self._write_metrics_csv(self.validation_root / "all_metrics.csv", self.all_results)

    def _advance_validation_sequence(self):
        if not self.validation_mode:
            return
        self.condition_cursor += 1
        if self.condition_cursor < len(self.condition_order):
            next_cond = self.condition_order[self.condition_cursor]
            self._reset_condition_state()
            self._apply_condition(next_cond)
            self.auto_analysis_status = (
                f"Participant {self.participant_number}: ready for condition {self.condition_id}/{len(self.condition_order)}."
            )
            return

        self._save_participant_summary()
        self.validation_complete = True
        self.state = DONE
        self.auto_analysis_status = (
            f"Validation complete for participant {self.participant_number}. Closing session..."
        )
        self._save_validation_run_summary()
        raise SystemExit(0)

    def _apply_condition(self, condition_id):
        spec = CONDITION_SPECS[int(condition_id)]
        self.condition_id = int(condition_id)
        self.condition_spec = spec
        self.pending_validation_finalize = False
        self.auto_analysis_status = None

        self.haptics.reset_proxy()
        self.haptics.wall_k = 70.0
        self.haptics.wall_damping = 8.0
        self.haptics.guidance_fade_start = spec.get("guidance_fade_start", 0.55)
        self.haptics.guidance_fade_end   = spec.get("guidance_fade_end",   0.90)
        self._sync_condition_feedback()

    def _can_change_condition(self):
        if self.validation_mode:
            return self.state == IDLE and len(self.all_demos) == 0
        return self.state in (IDLE, REVIEW, DONE)

    def _prepare_free_condition_switch(self):
        self._reset_condition_state()

    def _current_condition_label(self):
        return self.condition_spec["label"]

    def _effective_input_mode(self):
        return "device" if self.device_connected else "mouse_fallback"

    def _feedback_active(self):
        return self.condition_spec["feedback_mode"] != "none"

    def _demos_target_reached(self):
        return self.validation_mode and self.required_demos > 0 and len(self.all_demos) >= self.required_demos

    def _can_change_mode(self):
        return (
            self.state == IDLE
            and len(self.all_demos) == 0
            and len(self.all_results) == 0
            and self.current_participant_offset == 0
        )

    def _set_mode(self, mode):
        mode = str(mode)
        if mode not in ("free", "validation"):
            return
        self.mode = mode
        self.validation_mode = (mode == "validation")
        if self.validation_mode:
            self.required_demos = max(1, int(self.required_demos or 3))
            self.participant_count = 1
        else:
            self.required_demos = 0
            self.participant_count = 1
        self.current_participant_offset = 0
        self.validation_complete = False
        self.condition_cursor = 0
        self.condition_order = self._condition_order_for_current_mode()
        self.all_results = []
        self._reset_condition_state()
        self._apply_condition(self.condition_order[self.condition_cursor])

    def _interrupt_validation_to_free(self):
        if not self.validation_mode:
            return
        self._save_participant_summary()
        self._save_validation_run_summary()
        self._set_mode("free")
        self.auto_analysis_status = "Validation interrupted. Switched to free mode."

    def _sync_condition_feedback(self):
        spec = self.condition_spec or {}
        self.haptics.groove_enabled = bool(spec.get("centerline"))
        self.haptics.fading_groove_enabled = bool(spec.get("centerline_fading"))
        self.haptics.gp_groove_enabled = bool(spec.get("learned_guidance"))
        self.haptics.walls_enabled = bool(spec.get("walls")) and self.state == RECORDING

    def _refresh_online_gp(self):
        if not self.all_demos:
            self.gp = TrajectoryGP()
            self.gp_traj_phys = None
            self.gp_traj_std = None
            self.haptics.clear_gp_trajectory()
            return

        gp = TrajectoryGP()
        gp.fit(self.all_demos)
        gp_traj_phys, gp_traj_std = gp.predict(n_points=300, return_std=True)
        self.gp = gp
        self.gp_traj_phys = gp_traj_phys
        self.gp_traj_std = gp_traj_std
        self.haptics.set_gp_trajectory(
            self.gp_traj_phys,
            self.gp_traj_std,
            n_demos=len(self.all_demos),
        )

    def _flatten_metric_row(self, record):
        row = dict(record)
        tlx = row.pop("tlx", None)
        if isinstance(tlx, dict):
            for key, value in tlx.items():
                row[f"tlx_{key}"] = value
        return row

    def _write_metrics_csv(self, path, records):
        if not records:
            return
        rows = [self._flatten_metric_row(record) for record in records]
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _mode_label(self):
        if self.validation_mode:
            return f"validation P{self.participant_number} G{self.validation_group} ({len(self.all_demos)}/{self.required_demos} demos)"
        return "free"

    def _demo_success_rate(self, success_tol):
        if not self.all_demos:
            return 0.0
        successes = 0
        for demo in self.all_demos:
            demo_mnd = mean_nearest_distance(demo, self.tube.centerline)
            start_error = float(np.linalg.norm(demo[0] - self.tube.start))
            end_error = float(np.linalg.norm(demo[-1] - self.tube.end))
            if demo_mnd <= success_tol and start_error <= success_tol and end_error <= success_tol:
                successes += 1
        return float(successes / len(self.all_demos))

    def _compute_validation_metrics(self):
        """
        Compute validation-only metrics.

        These are deliberately reference-free for accuracy, plus GP-convergence
        metrics derived from refitting the GP after each accumulated demo set.
        """
        sigma_threshold_m = 0.005  # 5 mm
        n_points = 300

        pairwise_frechet_m = average_pairwise_frechet(self.all_demos)

        per_demo_sigma_mean_m = []
        per_demo_gp_path_length_m = []
        per_demo_path_length_ratio = []
        per_demo_jerk = []

        final_gp_traj = None
        final_gp_std = None

        for idx in range(len(self.all_demos)):
            demo_subset = self.all_demos[: idx + 1]
            gp_tmp = TrajectoryGP()
            gp_tmp.fit(demo_subset)
            gp_traj, gp_std = gp_tmp.predict(n_points=n_points, return_std=True)

            sigma_mean_m = float(np.mean(np.linalg.norm(gp_std, axis=1)))
            per_demo_sigma_mean_m.append(sigma_mean_m)
            gp_len_m = float(np.sum(np.linalg.norm(np.diff(gp_traj, axis=0), axis=1)))
            per_demo_gp_path_length_m.append(gp_len_m)

            demo = self.all_demos[idx]
            demo_time = self.all_demo_times[idx] if idx < len(self.all_demo_times) else None
            per_demo_path_length_ratio.append(path_length_ratio(demo, gp_traj))
            per_demo_jerk.append(mean_jerk_magnitude(demo, duration_s=demo_time))

            if idx == len(self.all_demos) - 1:
                final_gp_traj = gp_traj
                final_gp_std = gp_std

        demos_to_convergence = None
        cumulative_demo_time_to_convergence_s = None
        for idx, sigma_mean_m in enumerate(per_demo_sigma_mean_m):
            if sigma_mean_m <= sigma_threshold_m:
                demos_to_convergence = idx + 1
                cumulative_demo_time_to_convergence_s = float(sum(self.all_demo_times[: idx + 1]))
                break

        if cumulative_demo_time_to_convergence_s is None:
            cumulative_demo_time_to_convergence_s = float(sum(self.all_demo_times))

        metrics = {
            "pairwise_frechet_m": float(pairwise_frechet_m),
            "path_length_ratio_mean": float(np.mean(per_demo_path_length_ratio)) if per_demo_path_length_ratio else 1.0,
            "jerk_mean": float(np.mean(per_demo_jerk)) if per_demo_jerk else 0.0,
            "gp_sigma_mean_m": float(per_demo_sigma_mean_m[-1]) if per_demo_sigma_mean_m else 0.0,
            "gp_sigma_by_demo_m": per_demo_sigma_mean_m,
            "gp_path_length_by_demo_m": per_demo_gp_path_length_m,
            "path_length_ratio_by_demo": per_demo_path_length_ratio,
            "jerk_by_demo": per_demo_jerk,
            "gp_sigma_convergence_threshold_m": sigma_threshold_m,
            "demos_to_convergence": demos_to_convergence,
            "cumulative_demo_time_to_convergence_s": float(cumulative_demo_time_to_convergence_s),
        }
        return metrics, final_gp_traj, final_gp_std

    def _run_nasa_tlx_dialog(self):
        self.graphics.close()
        self.tlx_result = run_nasa_tlx()
        self.graphics = Graphics(self.device_connected, window_size=(760, 720))
        self.graphics.show_debug = False
        pygame.display.set_caption("PA3 — Kinesthetic Teaching (Mars Crack)")
        if self.all_results:
            self.all_results[-1]["tlx"] = self.tlx_result

    def _finalize_validation_condition(self):
        if self.tlx_result is None:
            self._run_nasa_tlx_dialog()
        self._save_results()
        self._advance_validation_sequence()

    def _mouse_panel_pos(self, mouse_pos):
        x, y = mouse_pos
        if x >= self.graphics.window_size[0]:
            x -= self.graphics.window_size[0]
        x = max(0, min(self.graphics.window_size[0] - 1, x))
        y = max(0, min(self.graphics.window_size[1] - 1, y))
        return np.array([x, y], dtype=float)


    # ─── Tube drawing ───────────────────────────────────────────────────
    # def _draw_tube(self, surface, highlight_wall=None):
    #     g = self.graphics
    #     tube = self.tube

    #     left_pts  = [g.convert_pos(p) for p in tube.wall_left]
    #     right_pts = [g.convert_pos(p) for p in tube.wall_right]
    #     center_pts = [g.convert_pos(p) for p in tube.centerline]

    #     left_color  = (255, 50, 50) if highlight_wall == 'left'  else (100, 100, 100)
    #     right_color = (255, 50, 50) if highlight_wall == 'right' else (100, 100, 100)

    #     # Draw walls first (will be covered by fill, then redrawn)
    #     if len(left_pts) > 1:
    #         pygame.draw.lines(surface, left_color, False, left_pts, 3)
    #     if len(right_pts) > 1:
    #         pygame.draw.lines(surface, right_color, False, right_pts, 3)

    #     # Fill tube interior
    #     for i in range(0, len(center_pts) - 1, 2):
    #         pygame.draw.line(surface, (230, 240, 230),
    #                          (int(center_pts[i][0]),   int(center_pts[i][1])),
    #                          (int(center_pts[i+1][0]), int(center_pts[i+1][1])),
    #                          int(tube.half_width * g.window_scale * 2))

    #     # Re-draw walls on top of fill
    #     if len(left_pts) > 1:
    #         pygame.draw.lines(surface, left_color, False, left_pts, 3)
    #     if len(right_pts) > 1:
    #         pygame.draw.lines(surface, right_color, False, right_pts, 3)

    #     # Centerline on top
    #     if len(center_pts) > 1:
    #         pygame.draw.lines(surface, (120, 120, 120), False, center_pts, 1)

    #     # Start / end markers
    #     start_s = g.convert_pos(tube.start)
    #     end_s   = g.convert_pos(tube.end)
    #     pygame.draw.circle(surface, (0, 200, 0),   (int(start_s[0]), int(start_s[1])), 10)
    #     pygame.draw.circle(surface, (200, 0, 0),   (int(end_s[0]),   int(end_s[1])),   10)
    #     font = pygame.font.SysFont("Arial", 12, bold=True)
    #     surface.blit(font.render("START", True, (0, 200, 0)),
    #                  (int(start_s[0]) + 12, int(start_s[1]) - 6))
    #     surface.blit(font.render("END",   True, (200, 0, 0)),
    #                  (int(end_s[0])   + 12, int(end_s[1])   - 6))

    def _draw_tube(self, surface, highlight_wall=None,
                   draw_fill=True, draw_walls=True, draw_centerline=True):
        g = self.graphics
        tube = self.tube
        center_pts = [g.convert_pos(p) for p in tube.centerline]
        left_color  = (255, 50, 50) if highlight_wall == 'left'  else (100, 100, 100)
        right_color = (255, 50, 50) if highlight_wall == 'right' else (100, 100, 100)

        center_px = [(int(p[0]), int(p[1])) for p in center_pts]
        tube_px = max(2, int(round(2.0 * tube.half_width * g.window_scale)))
        wall_px = 3
        fill_r = max(1, tube_px // 2)
        outer_r = fill_r + wall_px

        if len(center_px) > 1:
            if draw_walls:
                border_color = left_color if highlight_wall == 'left' else right_color if highlight_wall == 'right' else (100, 100, 100)
                for pt in center_px:
                    pygame.draw.circle(surface, border_color, pt, outer_r)

            if draw_fill:
                fill_color = (236, 242, 236)
                for pt in center_px:
                    pygame.draw.circle(surface, fill_color, pt, fill_r)

            if draw_centerline:
                pygame.draw.lines(surface, (120, 120, 120), False, center_px, 1)

        if draw_walls:
            start_s = g.convert_pos(tube.start)
            end_s   = g.convert_pos(tube.end)
            pygame.draw.circle(surface, (0, 200, 0), (int(start_s[0]), int(start_s[1])), 10)
            pygame.draw.circle(surface, (200, 0, 0), (int(end_s[0]),   int(end_s[1])),   10)
            font = pygame.font.SysFont("Arial", 12, bold=True)
            surface.blit(font.render("START", True, (0, 200, 0)),
                         (int(start_s[0]) + 12, int(start_s[1]) - 6))
            surface.blit(font.render("END",   True, (200, 0, 0)),
                         (int(end_s[0])   + 12, int(end_s[1])   - 6))


    def _draw_trajectory(self, surface, traj_phys, color, width=2):
        g = self.graphics
        if len(traj_phys) < 2:
            return
        pts = [g.convert_pos(p) for p in traj_phys]
        pygame.draw.lines(surface, color, False, pts, width)

    def _draw_gp_uncertainty(self, surface, traj, std):
        g = self.graphics
        for i in range(0, len(traj), 3):
            pt = g.convert_pos(traj[i])
            r = max(2, int(np.mean(std[i]) * g.window_scale * 2))
            r = min(r, 40)
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (100, 100, 255, 60), (r, r), r)
            surface.blit(s, (int(pt[0]) - r, int(pt[1]) - r))

    def _draw_key_legend(self, surface):
        """Draw a compact key reference in the top-left corner of a panel."""
        cond = self.condition_id
        cond_name = self._current_condition_label()

        lines = [
            "── Keys ──────────────",
            "SPACE  start/stop rec",
            "ENTER  confirm/next",
            "D      delete last demo",
            "G      train GP",
            "P      replay GP",
            "A      auto-reproduce GP",
            f"COND   [{cond}] {cond_name}",
            (
                "1-5    select condition"
                if not self.validation_mode else "COND   automatic sequence"
            ),
            "M      toggle mode",
            "F      abort validation -> free",
            "T      change tube",
            "C      clear all",
            "Q      quit",
            f"MODE   {self._mode_label()}",
        ]

        font = pygame.font.SysFont("Courier", 13)
        x, y = 8, 8
        pad = 4
        line_h = font.get_linesize()
        box_w = 260
        box_h = len(lines) * line_h + pad * 2

        # Semi-transparent background
        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        surface.blit(bg, (x - pad, y - pad))

        for i, line in enumerate(lines):
            color = (200, 200, 200) if i > 0 else (255, 220, 80)
            surf = font.render(line, True, color)
            surface.blit(surf, (x, y + i * line_h))

    def _draw_haptics_status_banner(self, surface):
        panel_x = 16
        panel_y = 16
        card_w = 180
        card_h = 88
        gap = 12
        pad = 12

        title_font = pygame.font.SysFont("Arial", 18, bold=True)
        value_font = pygame.font.SysFont("Arial", 38, bold=True)
        meta_font = pygame.font.SysFont("Arial", 16, bold=True)

        if self.validation_mode:
            trial_value = f"{self.condition_cursor + 1}/{len(self.condition_order)}"
            demo_value = f"{len(self.all_demos)}/{self.required_demos}"
        else:
            trial_value = f"{self.condition_id}"
            demo_value = f"{len(self.all_demos)}"

        demo_color = (120, 255, 170)
        if self.validation_mode and len(self.all_demos) < self.required_demos:
            demo_color = (255, 220, 120)
        if self.state == RECORDING:
            demo_color = (255, 140, 140)

        cards = [
            ("TRIAL" if self.validation_mode else "COND", trial_value, (120, 190, 255)),
            ("DEMOS", demo_value, demo_color),
        ]

        for idx, (label, value, accent) in enumerate(cards):
            x = panel_x + idx * (card_w + gap)
            bg = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            pygame.draw.rect(bg, (12, 18, 24, 215), bg.get_rect(), border_radius=12)
            pygame.draw.rect(bg, accent, bg.get_rect(), width=3, border_radius=12)
            surface.blit(bg, (x, panel_y))

            label_surf = title_font.render(label, True, (230, 235, 240))
            value_surf = value_font.render(value, True, accent)
            surface.blit(label_surf, (x + pad, panel_y + 10))
            surface.blit(value_surf, (x + pad, panel_y + 34))

        meta_lines = []
        if self.validation_mode:
            meta_lines.append(f"Participant {self.participant_number}  Group {self.validation_group}")
        else:
            meta_lines.append(f"Participant {self.participant_number}  Free mode")
        meta_lines.append(f"Condition {self.condition_id}: {self._current_condition_label()}")

        meta_y = panel_y + card_h + 10
        for idx, line in enumerate(meta_lines):
            meta_bg = pygame.Surface((430, 24), pygame.SRCALPHA)
            meta_bg.fill((10, 14, 20, 170))
            surface.blit(meta_bg, (panel_x, meta_y + idx * 26))
            meta_surf = meta_font.render(line, True, (235, 235, 235))
            surface.blit(meta_surf, (panel_x + 10, meta_y + 3 + idx * 26))

    # ─── Main loop ──────────────────────────────────────────────────────
    def run(self):
        p = self.physics
        g = self.graphics
        keyups, xm = g.get_events()

        # ── Keyboard ────────────────────────────────────────────────────
        for key in keyups:
            if key == ord('q'):
                self._save_results()
                sys.exit(0)

            if key == ord('f') and self.validation_mode:
                self._interrupt_validation_to_free()
                continue

            if self.validation_complete:
                continue

            if key == ord('r'):
                g.show_linkages = not g.show_linkages

            if key == ord('m'):
                if self._can_change_mode():
                    self._set_mode("free" if self.validation_mode else "validation")
                    self.auto_analysis_status = f"Mode changed to {self.mode}."
                else:
                    self.auto_analysis_status = "Mode can only change from a clean idle state."

            if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                if self.validation_mode:
                    self.auto_analysis_status = "Validation mode advances conditions automatically."
                elif self._can_change_condition():
                    self._prepare_free_condition_switch()
                    self._apply_condition(int(chr(key)))
                    if self.condition_id == 2:
                        self.auto_analysis_status = "Condition 2 selected: virtual walls are active."
                    else:
                        self.auto_analysis_status = f"Condition {self.condition_id} selected."

            # T → cycle tube (only when idle and no demos)
            if key == ord('t') and self._can_change_condition():
                if self.validation_mode:
                    self.auto_analysis_status = "Validation mode keeps the crack geometry fixed."
                else:
                    self.tube_idx = (self.tube_idx + 1) % len(TUBE_NAMES)
                    self.tube_name = TUBE_NAMES[self.tube_idx]
                    self.tube = get_tube(self.tube_name)
                    self.haptics = TubeHaptics(self.tube)
                    self._apply_condition(self.condition_id)
            

            if key == pygame.K_SPACE:
                if self.state == IDLE:
                    if self._demos_target_reached():
                        self.auto_analysis_status = (
                            f"Validation target reached: train condition after {self.required_demos} demos."
                        )
                        continue
                    self.state = RECORDING
                    self.current_demo = []
                    self.wall_hits = 0
                    self.wall_contact_active = False
                    self.start_time = time.time()
                elif self.state == RECORDING:
                    self.last_demo_time = time.time() - self.start_time
                    if len(self.current_demo) > 10:
                        demo_arr = np.array(self.current_demo)
                        self.all_demos.append(demo_arr)
                        self.all_demo_times.append(self.last_demo_time)
                        mnd = mean_nearest_distance(demo_arr, self.tube.centerline)
                        self.per_demo_metrics.append(mnd)
                        self.per_demo_wall_hits.append(int(self.wall_hits))
                        self._refresh_online_gp()
                        self.state = REVIEW
                    else:
                        self.state = IDLE
                    self.wall_contact_active = False

            # D → delete last demo (only in REVIEW)
            if key == ord('d') and self.state == REVIEW:
                if len(self.all_demos) > 0:
                    self.all_demos.pop()
                    self.all_demo_times.pop()
                    self.per_demo_metrics.pop()
                    self.per_demo_wall_hits.pop()
                    self._refresh_online_gp()
                self.current_demo = []
                self.state = IDLE

            if key == pygame.K_RETURN:
                if self.state == REVIEW:
                    if self.validation_mode and self._demos_target_reached():
                        self.auto_train = True
                        self.pending_validation_finalize = True
                        self.state = TRAINING
                    else:
                        self.state = IDLE
                elif self.state == DONE:
                    if self.validation_mode and self.pending_validation_finalize:
                        self.pending_validation_finalize = False
                        self._finalize_validation_condition()
                        return
                    self.gp_traj_phys = None
                    self.gp_traj_std = None
                    self.trial_metrics = None
                    self.state = IDLE


            if key == ord('g') and self.state in (IDLE, REVIEW) and len(self.all_demos) >= 1:
                if self.validation_mode and not self._demos_target_reached():
                    self.auto_analysis_status = (
                        f"Validation mode requires {self.required_demos} demos before training."
                    )
                else:
                    self.auto_train = self.validation_mode
                    self.pending_validation_finalize = self.validation_mode
                    self.state = TRAINING

            if key == ord('p') and self.state == DONE:
                self.playback_idx = 0
                self.state = PLAYBACK

            if key == ord('n') and self.state == DONE:
                self._run_nasa_tlx_dialog()
                g = self.graphics
                g.erase_screen()
                return

            if key == ord('c') and self.state in (IDLE, REVIEW, DONE):
                self.all_demos = []
                self.all_demo_times = []
                self.per_demo_metrics = []
                self.per_demo_wall_hits = []
                self.current_demo = []
                self.gp_traj_phys = None
                self.gp_traj_std = None
                self.trial_metrics = None
                self.haptics.reset_proxy()
                self.haptics.clear_gp_trajectory()
                self.wall_hits = 0
                self.wall_contact_active = False
                self.pending_validation_finalize = False
                self.auto_analysis_status = None
                self.state = IDLE

            # A → start autonomous GP reproduction (only when GP is trained)
            if key == ord('a') and self.state == DONE and self.gp_traj_phys is not None:
                self.auto_ref_idx = 0.0
                self.prev_ee_phys = None
                self.state = AUTO_PLAY

        frame_input_mode = self._effective_input_mode()
        frame_feedback_active = self._feedback_active()
        self._sync_condition_feedback()
        g.device_connected = (frame_input_mode == "device")

        if frame_input_mode == "device":
            try:
                pA0, pB0, pA, pB, pE = p.get_device_pos()
                pA0, pB0, pA, pB, xh = g.convert_pos(pA0, pB0, pA, pB, pE)
            except ValueError:
                self.device_connected = False
                frame_input_mode = "mouse_fallback"
                frame_feedback_active = self._feedback_active()
                g.device_connected = False
                self.auto_analysis_status = "Haply stream unavailable. Switched to mouse fallback."
                if self.condition_spec["feedback_mode"] == "none":
                    xh = self._mouse_panel_pos(xm)
                else:
                    xh = g.haptic.center
                pos_phys_s = g.inv_convert_pos(xh)
                pA0, pB0, pA, pB, pE = p.derive_device_pos(pos_phys_s)
                pA0, pB0, pA, pB, xh = g.convert_pos(pA0, pB0, pA, pB, pE)
        else:
            if self.condition_spec["feedback_mode"] == "none":
                xh = self._mouse_panel_pos(xm)
            else:
                xh = g.haptic.center
            pos_phys_s = g.inv_convert_pos(xh)
            pA0, pB0, pA, pB, pE = p.derive_device_pos(pos_phys_s)
            pA0, pB0, pA, pB, xh = g.convert_pos(pA0, pB0, pA, pB, pE)

        xh = np.array(xh, dtype=float)
        g.erase_screen()


        pos_phys = np.array(g.inv_convert_pos(xh), dtype=float)

        if not frame_feedback_active:
            fe_phys = np.zeros(2)
            fe = np.zeros(2)
        elif self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            fe_phys = self._compute_pd_force(pos_phys, dt=1.0 / g.FPS)
            fe = np.array([fe_phys[0], -fe_phys[1]])
        else:
            fe_phys = self.haptics.compute_force(pos_phys, dt=1.0 / g.FPS)
            fe = np.array([fe_phys[0], -fe_phys[1]])


        # Track discrete wall-contact events instead of counting every frame
        if self.state == RECORDING:
            in_wall_contact = self.haptics.last_penetration > 0
            if in_wall_contact and not self.wall_contact_active:
                self.wall_hits += 1
            self.wall_contact_active = in_wall_contact
        else:
            self.wall_contact_active = False

        # ── Record ──────────────────────────────────────────────────────
        if self.state == RECORDING:
            self.current_demo.append(pos_phys.copy())

        # ── GP Training ─────────────────────────────────────────────────
        if self.state == TRAINING:
            if len(self.all_demos) >= 1:
                font_big = pygame.font.SysFont("Arial", 32, bold=True)
                splash = font_big.render(
                    f"Training GP on {len(self.all_demos)} demos...", True, (255, 255, 255))
                g.screenVR.blit(splash, (g.window_size[0]//2 - splash.get_width()//2,
                                         g.window_size[1]//2 - 20))
                g.window.blit(g.screenHaptics, (0, 0))
                g.window.blit(g.screenVR, (g.window_size[0], 0))
                pygame.display.flip()

                self.gp = TrajectoryGP()
                self.gp.fit(self.all_demos)

                if self.validation_mode:
                    self.trial_metrics, self.gp_traj_phys, self.gp_traj_std = self._compute_validation_metrics()
                else:
                    self.gp_traj_phys, self.gp_traj_std = self.gp.predict(
                        n_points=300, return_std=True)
                    self.trial_metrics = None

                self.haptics.set_gp_trajectory(self.gp_traj_phys, self.gp_traj_std, n_demos=len(self.all_demos))

                if self.validation_mode:
                    total_demo_time = float(sum(self.all_demo_times))
                    mean_demo_time = float(np.mean(self.all_demo_times)) if self.all_demo_times else 0.0
                    self.trial_metrics["participant_number"] = self.participant_number
                    self.trial_metrics["participant_count"] = self.participant_count
                    self.trial_metrics["validation_group"] = self.validation_group
                    self.trial_metrics["condition_order"] = self.condition_order.copy()
                    self.trial_metrics["mode"] = self.mode
                    self.trial_metrics["required_demos_target"] = self.required_demos
                    self.trial_metrics["trial_index"] = len(self.all_results) + 1
                    self.trial_metrics["condition_id"] = self.condition_id
                    self.trial_metrics["condition"] = self.condition_spec["slug"]
                    self.trial_metrics["condition_label"] = self.condition_spec["label"]
                    self.trial_metrics["input_mode"] = self._effective_input_mode()
                    self.trial_metrics["feedback_mode"] = self.condition_spec["feedback_mode"]
                    self.trial_metrics["tube"] = self.tube_name
                    self.trial_metrics["n_demos"] = len(self.all_demos)
                    self.trial_metrics["demo_times"] = self.all_demo_times.copy()
                    self.trial_metrics["mean_demo_time_s"] = mean_demo_time
                    self.trial_metrics["total_demo_time_s"] = total_demo_time
                    self.trial_metrics["completion_time_s"] = total_demo_time
                    self.trial_metrics["last_demo_time_s"] = float(self.last_demo_time)
                    self.trial_metrics["hardware_connected"] = self.device_connected
                    self.all_results.append(self.trial_metrics.copy())

                if self.auto_train:
                    self.auto_train = False
                    if self.pending_validation_finalize:
                        self.playback_idx = len(self.gp_traj_phys)
                        self.state = DONE
                        self.auto_analysis_status = "GP ready. Press P to replay or ENTER to save and continue."
                    else:
                        self.state = IDLE
                else:
                    self.playback_idx = 0            # G was pressed: show playback as before
                    self.state = PLAYBACK


            else:
                self.state = IDLE

        if self.state == PLAYBACK:
            self.playback_idx += 2
            if self.playback_idx >= len(self.gp_traj_phys):
                self.playback_idx = len(self.gp_traj_phys)
                self.state = DONE

        # ══════════════════════ DRAWING ══════════════════════════════════

        # Suppress default rectangular haptic cursor
        g.haptic.width = 0
        g.haptic.height = 0

        wall_hit = self.haptics.last_wall if self.haptics.last_penetration > 0 else None
        progress = self.tube.progress(pos_phys)
        elapsed = time.time() - self.start_time if self.state == RECORDING else 0.0
        
        # ── Left: Haptic panel ──────────────────────────────────────────
        # Draw tube without centerline — we handle it conditionally below
        self._draw_tube(g.screenHaptics, highlight_wall=wall_hit, draw_centerline=False)

        if self.state == RECORDING and len(self.current_demo) > 1:
            self._draw_trajectory(g.screenHaptics, self.current_demo,
                                  color=(220, 60, 60), width=2)

        # Guidance line on haptic panel
        if self.haptics.groove_enabled or self.haptics.fading_groove_enabled:
            center_pts = [g.convert_pos(p) for p in self.tube.centerline]
            if len(center_pts) > 1:
                pygame.draw.lines(g.screenHaptics, (120, 120, 120), False, center_pts, 1)
        elif self.haptics.gp_groove_enabled and self.gp_traj_phys is not None:
            self._draw_trajectory(g.screenHaptics, self.gp_traj_phys,
                                  color=(40, 40, 220), width=1)


        # Proxy dot
        if self.haptics.last_proxy_pos is not None:
            ps = g.convert_pos(self.haptics.last_proxy_pos)
            pygame.draw.circle(g.screenHaptics, (255, 165, 0), (int(ps[0]), int(ps[1])), 8)
            pygame.draw.circle(g.screenHaptics, (200, 0, 0),   (int(ps[0]), int(ps[1])), 8, 2)

        # End-effector dot
        pygame.draw.circle(g.screenHaptics, (220, 0, 0), (int(xh[0]), int(xh[1])), 10)

        # Auto-play reference dot on haptic panel
        if self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            ref_pos = self.gp_traj_phys[int(self.auto_ref_idx)]
            ref_s = g.convert_pos(ref_pos)
            pygame.draw.circle(g.screenHaptics, (0, 255, 100),
                               (int(ref_s[0]), int(ref_s[1])), 7, 2)

        # Force arrow
        if np.linalg.norm(fe) > 0.01:
            fscale = 50.0
            pygame.draw.line(g.screenHaptics, (0, 0, 255),
                             (int(xh[0]), int(xh[1])),
                             (int(xh[0] - fe[0] * fscale),
                              int(xh[1] - fe[1] * fscale)), 2)

        self._draw_haptics_status_banner(g.screenHaptics)

        # ── Right: VR panel ─────────────────────────────────────────────
        g.draw_mars_vr_scene(
            self.tube,
            STATE_NAMES[self.state],
            len(self.all_demos),
            self.tube_name,
            progress,
            elapsed,
            highlight_wall=wall_hit,
            cursor_phys=pos_phys,
        )

        # 2. GP uncertainty — above fill, below walls
        if self.gp_traj_phys is not None:
            gp_idx = self.playback_idx if self.state == PLAYBACK else len(self.gp_traj_phys)
            self._draw_gp_uncertainty(g.screenVR,
                                      self.gp_traj_phys[:gp_idx], self.gp_traj_std[:gp_idx])

        # 4. Demos
        for i, demo in enumerate(self.all_demos):
            c = DEMO_COLORS[i % len(DEMO_COLORS)]
            self._draw_trajectory(g.screenVR, demo, color=c, width=1)
        if self.state == RECORDING and len(self.current_demo) > 1:
            c = DEMO_COLORS[len(self.all_demos) % len(DEMO_COLORS)]
            self._draw_trajectory(g.screenVR, self.current_demo, color=c, width=2)

        # 5. GP mean trajectory
        if self.gp_traj_phys is not None:
            self._draw_trajectory(g.screenVR, self.gp_traj_phys[:gp_idx],
                                  color=(40, 40, 220), width=3)

        # 6. Proxy dot
        if self.haptics.last_proxy_pos is not None:
            ps = g.convert_pos(self.haptics.last_proxy_pos)
            pygame.draw.circle(g.screenVR, (255, 165, 0), (int(ps[0]), int(ps[1])), 8)
            pygame.draw.circle(g.screenVR, (200, 0, 0),   (int(ps[0]), int(ps[1])), 8, 2)

        # 7. End-effector dot — always on top
        ee_vr = g.convert_pos(pos_phys)
        pygame.draw.circle(g.screenVR, (220, 0, 0), (int(ee_vr[0]), int(ee_vr[1])), 10)

        # 8. Auto-play reference dot
        if self.state == AUTO_PLAY and self.gp_traj_phys is not None:
            ref_vr = g.convert_pos(self.gp_traj_phys[int(self.auto_ref_idx)])
            pygame.draw.circle(g.screenVR, (0, 255, 100),
                               (int(ref_vr[0]), int(ref_vr[1])), 7, 2)

        # 9. Confidence bar
        if self.gp_traj_phys is not None and self.gp_traj_std is not None:
            dists = np.linalg.norm(self.gp_traj_phys - pos_phys, axis=1)
            nearest_idx = int(np.argmin(dists))
            local_std   = float(np.mean(self.gp_traj_std[nearest_idx]))
            std_min     = self.haptics.gp_std_min
            std_max     = self.haptics.gp_std_max
            std_factor  = float(np.clip((std_max - local_std) / (std_max - std_min), 0.0, 1.0))
            demo_factor = 1.0 - np.exp(-(len(self.all_demos) - 1) / 3.0)
            confidence  = std_factor * demo_factor

            bar_w, bar_h = 120, 14
            bar_x = g.window_size[0] - bar_w - 10
            bar_y = 10
            conf_font = pygame.font.SysFont("Courier", 12)
            pygame.draw.rect(g.screenVR, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
            fill_color = (int(255 * (1 - confidence)), int(255 * confidence), 0)
            pygame.draw.rect(g.screenVR, fill_color,
                             (bar_x, bar_y, int(bar_w * confidence), bar_h))
            pygame.draw.rect(g.screenVR, (180, 180, 180), (bar_x, bar_y, bar_w, bar_h), 1)
            g.screenVR.blit(conf_font.render(
                f"conf:{confidence*100:.0f}% σ={local_std*1000:.1f}mm",
                True, (220, 220, 220)), (bar_x - 55, bar_y + bar_h + 3))







        # Key legend (top-left of VR panel, always visible)
        self._draw_key_legend(g.screenVR)

        # ── Debug text (built manually, drawn bottom-center of haptic panel) ──
        n = len(self.all_demos)
        state_str = STATE_NAMES[self.state]
        debug_parts = [
            f"[{state_str}]",
            f"mode={self.mode}",
            f"p={self.participant_number}",
            f"demos={n}",
            f"tube={self.tube_name}",
            f"cond={self.condition_id}",
            f"input={frame_input_mode}",
            f"fb={self.condition_spec['feedback_mode']}",
        ]
        if self.state == RECORDING:
            prog = progress
            debug_parts += [f"t={elapsed:.1f}s",
                            f"prog={prog*100:.0f}%",
                            f"wall_hits={self.wall_hits}"]
        if self.trial_metrics and self.state in (PLAYBACK, DONE):
            debug_parts.append(f"σ={self.trial_metrics['gp_sigma_mean_m']*1000:.1f}mm")
        if self.validation_mode:
            debug_parts.append(f"target={self.required_demos}")

        debug_str = "   ".join(debug_parts)
        debug_font = pygame.font.SysFont("Courier", 13)
        debug_surf = debug_font.render(debug_str, True, (200, 200, 200))

        dw, dh = debug_surf.get_width(), debug_surf.get_height()
        W = g.screenHaptics.get_width()
        H = g.screenHaptics.get_height()
        pad = 6
        bg = pygame.Surface((dw + pad * 2, dh + 4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))

        # x where the text starts in full-window coordinates
        x_full = W - dw // 2
        y = H - dh - 8

        # Portion on left panel (screenHaptics)
        if x_full < W:
            clip_w_left = min(dw, W - x_full)
            g.screenHaptics.blit(bg,        (x_full - pad, y - 2),
                                  (0, 0, clip_w_left + pad, dh + 4))
            g.screenHaptics.blit(debug_surf, (x_full, y),
                                  (0, 0, clip_w_left, dh))

        # Portion on right panel (screenVR)
        x_vr = x_full - W          # x in screenVR coords (may be negative)
        src_x = max(0, -x_vr)      # skip pixels already drawn on left panel
        dst_x = max(0,  x_vr)
        clip_w_right = dw - src_x
        if clip_w_right > 0:
            g.screenVR.blit(bg,        (dst_x - pad, y - 2),
                             (src_x, 0, clip_w_right + pad, dh + 4))
            g.screenVR.blit(debug_surf, (dst_x, y),
                             (src_x, 0, clip_w_right, dh))


        # ── State-specific instructions (bottom of VR panel) ────────────
        font = pygame.font.SysFont("Arial", 16)
        y_base = g.window_size[1] - 160
        if self.state == IDLE:
            lines = [
                f"Participant: {self.participant_number}",
                f"Demos: {n}  |  Crack: {self.tube_name}",
                f"Condition {self.condition_id}: {self._current_condition_label()}",
                f"Mode: {self._mode_label()}",
            ]
            if n >= 1: lines.append("G = train GP")
            if self.validation_mode:
                lines.append(
                    f"Collect exactly {self.required_demos} demos before condition completion"
                )
                lines.append("F = interrupt validation and switch to free mode")
        elif self.state == REVIEW:
            lines = [
                f"Participant: {self.participant_number}",
                f"Demo #{n} saved!  ({self.last_demo_time:.1f}s)",
                f"Condition {self.condition_id}: {self._current_condition_label()}",
                "ENTER = keep & next demo",
                "D = delete this demo",
                (
                    f"ENTER = train GP and review condition at {n}/{self.required_demos} demos"
                    if self.validation_mode and n >= self.required_demos
                    else f"G = train GP on {n} demo{'s' if n > 1 else ''}"
                ),
            ]
            if self.validation_mode:
                lines.insert(3, "Validation metrics are computed at condition finalization")
                lines.append("F = interrupt validation and switch to free mode")
        elif self.state == DONE:
            if self.validation_mode:
                m = self.trial_metrics
                convergence_demos = m["demos_to_convergence"]
                convergence_time = m["cumulative_demo_time_to_convergence_s"]
                convergence_label = (
                    str(convergence_demos) if convergence_demos is not None else "not reached"
                )
                lines = [
                    f"Participant: {self.participant_number}",
                    f"Condition {self.condition_id}: {self._current_condition_label()}",
                    f"Mode: {self._mode_label()}",
                    f"GP trained on {m['n_demos']} demos",
                    f"Inter-demo Frechet: {m['pairwise_frechet_m']*1000:.2f}mm  |  PLR: {m['path_length_ratio_mean']:.3f}",
                    f"Jerk: {m['jerk_mean']:.4f}  |  GP sigma: {m['gp_sigma_mean_m']*1000:.2f}mm",
                    f"Convergence demos: {convergence_label}  |  Convergence time: {convergence_time:.1f}s",
                    "A = auto-play  |  P = replay  |  ENTER = save and next condition",
                    "N = NASA-TLX  |  F = switch to free  |  C = clear  |  Q = quit",
                ]
            else:
                lines = [
                    f"Participant: {self.participant_number}",
                    f"Condition {self.condition_id}: {self._current_condition_label()}",
                    f"Mode: {self._mode_label()}",
                    f"GP trained on {len(self.all_demos)} demos",
                    "Free mode does not compute or save trial metrics",
                    "A = auto-play  |  P = replay  |  ENTER = add demos",
                    "C = clear  |  Q = quit",
                ]
        else:
            lines = []

        if self.auto_analysis_status:
            lines.append(self.auto_analysis_status)

        for i, line in enumerate(lines):
            surf = font.render(line, True, (255, 255, 255))
            g.screenVR.blit(surf, (20, y_base + i * 22))


        # ── Physics sim ─────────────────────────────────────────────────
        if frame_input_mode == "device":
            p.update_force(fe)
        else:
            if frame_feedback_active:
                xh = g.sim_forces(xh, fe, xm, mouse_k=0.5, mouse_b=0.8)
            pos_phys_s = g.inv_convert_pos(xh)
            pA0, pB0, pA, pB, pE = p.derive_device_pos(pos_phys_s)
            pA0, pB0, pA, pB, xh = g.convert_pos(pA0, pB0, pA, pB, pE)

        g.render(pA0, pB0, pA, pB, xh, fe, xm)

    
    def _compute_pd_force(self, pos_phys, dt):
        traj = self.gp_traj_phys
        n = len(traj)

        # Advance reference at fixed speed — guaranteed forward pull
        # pd_speed = indices per frame. At 100fps, 300 points in ~4s = 0.75/frame
        self.auto_ref_idx = min(self.auto_ref_idx + self.pd_speed, n - 1)
        ref_idx = int(self.auto_ref_idx)

        pos_ref = traj[ref_idx]
        pos_err = pos_ref - pos_phys

        if self.prev_ee_phys is not None and dt > 0:
            vel = (pos_phys - self.prev_ee_phys) / dt
        else:
            vel = np.zeros(2)
        self.prev_ee_phys = pos_phys.copy()

        fe = self.pd_kp * pos_err - self.pd_kd * vel

        if ref_idx >= n - 1:
            self.state = DONE
            self.auto_ref_idx = 0
            return np.zeros(2)

        return fe


    # def _save_results(self):
    #     if not self.all_results: return
    #     os.makedirs("results", exist_ok=True)
    #     fname = f"results/kinesthetic_{int(time.time())}.json"
    #     with open(fname, "w") as f:
    #         json.dump(self.all_results, f, indent=2, default=str)
    #     print(f"Results saved to {fname}")


    def _save_results(self):
        if not self.all_results and not self.all_demos:
            return

        if self.validation_mode:
            folder = self._current_condition_dir()
            folder.mkdir(parents=True, exist_ok=True)
            current_metrics = [self.all_results[-1]] if self.all_results else []
            if current_metrics:
                with open(folder / "metrics.json", "w") as f:
                    json.dump(current_metrics, f, indent=2, default=str)
                self._write_metrics_csv(folder / "metrics.csv", current_metrics)
            for i, demo in enumerate(self.all_demos):
                np.save(folder / f"demo_{i+1}.npy", demo)
            if self.gp_traj_phys is not None:
                np.save(folder / "gp_trajectory.npy", self.gp_traj_phys)
            if self.gp_traj_std is not None:
                np.save(folder / "gp_std.npy", self.gp_traj_std)
            summary = {
                "run_id": self.run_id,
                "participant_number": self.participant_number,
                "participant_count": self.participant_count,
                "validation_group": self.validation_group,
                "condition_order": self.condition_order,
                "mode": self.mode,
                "required_demos_target": self.required_demos,
                "tube": self.tube_name,
                "condition_id": self.condition_id,
                "condition": self.condition_spec["slug"],
                "condition_label": self.condition_spec["label"],
                "input_mode": self._effective_input_mode(),
                "feedback_mode": self.condition_spec["feedback_mode"],
                "hardware_connected": self.device_connected,
                "n_demos": len(self.all_demos),
                "demo_files": [f"demo_{i+1}.npy" for i in range(len(self.all_demos))],
            }
            with open(folder / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            self._save_participant_summary()
            self._save_validation_run_summary()
            print(f"Condition saved to {folder}/")
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.session_folder.mkdir(parents=True, exist_ok=True)
        folder = self.session_folder

        if self.all_results:
            with open(folder / "metrics.json", "w") as f:
                json.dump(self.all_results, f, indent=2, default=str)
            self._write_metrics_csv(folder / "metrics.csv", self.all_results)

        for i, demo in enumerate(self.all_demos):
            np.save(folder / f"demo_{i+1}.npy", demo)

        if self.gp_traj_phys is not None:
            np.save(folder / "gp_trajectory.npy", self.gp_traj_phys)
        if self.gp_traj_std is not None:
            np.save(folder / "gp_std.npy", self.gp_traj_std)

        summary = {
            "session_id": self.session_id,
            "participant_number": self.participant_number,
            "participant_count": self.participant_count,
            "mode": self.mode,
            "required_demos_target": self.required_demos if self.validation_mode else None,
            "tube":        self.tube_name,
            "condition_id": self.condition_id,
            "condition":   self.condition_spec["slug"],
            "condition_label": self.condition_spec["label"],
            "input_mode": self._effective_input_mode(),
            "feedback_mode": self.condition_spec["feedback_mode"],
            "hardware_connected": self.device_connected,
            "n_trials": len(self.all_results),
            "n_demos":     len(self.all_demos),
            "demo_files":  [f"demo_{i+1}.npy" for i in range(len(self.all_demos))],
            "conditions_in_session": sorted({
                result["condition"] for result in self.all_results
            }) if self.all_results else [],
        }
        with open(folder / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Session saved to {folder}/")


    def close(self):
        self._save_results()
        if self.validation_mode:
            self._save_participant_summary()
            self._save_validation_run_summary()
        self.graphics.close()
        self.physics.close()


def main():
    config = parse_runtime_config()
    pa = PA3_Kinesthetic(config)
    try:
        while True:
            pa.run()
    except SystemExit:
        pass
    finally:
        pa.close()


if __name__ == "__main__":
    main()

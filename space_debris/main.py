# -*- coding: utf-8 -*-
"""
Space Debris Capture — Main Entry Point

Usage:
    python main.py [--condition mouse|haply|haply_haptics]

Keys:
    SPACE      grab / deposit debris
    R          start/stop recording a demonstration
    G          train GP on all recorded demos
    T          toggle GP confidence bar display
    N          cycle experimental condition
    Q / ESC    quit and save results
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pygame

# ── Allow imports from parent directory (Physics.py, etc.) ──────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
sys.path.insert(0, _PARENT)

from space_debris.environment import SpaceDebrisEnv
from space_debris.renderer import Renderer, CENTER_X, CENTER_Y, SCALE
from space_debris.metrics import compute_trial_metrics


# ── Condition cycle order ────────────────────────────────────────────────────
CONDITIONS = ['mouse', 'haply', 'haply_haptics']


def parse_args():
    parser = argparse.ArgumentParser(description="Space Debris Capture Simulation")
    parser.add_argument(
        '--condition',
        choices=CONDITIONS,
        default='haply_haptics',
        help="Experimental condition (default: mouse)",
    )
    return parser.parse_args()


def mouse_to_physical(mouse_pos):
    """Convert pygame mouse position (pixels) to physical coordinates (meters)."""
    mx, my = mouse_pos
    px = (mx - CENTER_X) / SCALE
    py = (CENTER_Y - my) / SCALE
    return np.array([px, py])


def try_init_haply():
    """
    Attempt to initialize the Haply Physics device.

    Returns
    -------
    physics : Physics instance or None
    connected : bool
    """
    try:
        from Physics import Physics
        physics = Physics(hardware_version=3)
        if physics.is_device_connected():
            return physics, True
        else:
            return physics, False
    except Exception as e:
        print(f"[main] Haply init failed: {e}")
        return None, False


def send_haply_force(physics, force):
    """
    Send haptic force to Haply device.
    NOTE: update_force internally flips f[1] (y-axis), so we only need to
    flip x to account for our coordinate mapping (sim_x = pE_x - center).
    """
    try:
        # Our sim x is NOT flipped vs device x, so pass fx as-is.
        # update_force will flip fy internally to match device y convention.
        f = np.array([force[0], force[1]], dtype=float)
        physics.update_force(f)
    except Exception:
        pass


def get_haply_position(physics):
    """
    Read end-effector position and map from Haply device frame to sim frame.

    Haply frame:  x in [0, d], y positive downward (away from user)
    Sim frame:    x centered at 0, y positive upward

    Mapping:
      sim_x =  pE_x - HAPLY_X_CENTER    (no x flip, just center)
      sim_y = -(pE_y - HAPLY_Y_REST)    (flip y: device-down = sim-down)
    """
    HAPLY_X_CENTER = 0.019   # d/2 — center of device workspace in x
    HAPLY_Y_REST = 0.07      # typical y at rest (mid-workspace)
    try:
        _, _, _, _, pE = physics.get_device_pos()
        sim_x = pE[0] - HAPLY_X_CENTER
        sim_y = -(pE[1] - HAPLY_Y_REST)
        return np.array([sim_x, sim_y], dtype=float)
    except Exception:
        return None


def save_results(env, session_dir, condition, all_trial_metrics):
    """Save session results to disk."""
    os.makedirs(session_dir, exist_ok=True)

    # Save per-trial metrics
    metrics_path = os.path.join(session_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_trial_metrics, f, indent=2, default=str)

    # Session summary
    summary = {
        "condition": condition,
        "n_demos": env.gp_policy.n_demos,
        "gp_trained": env.gp_policy.trained,
        "n_trials": len(all_trial_metrics),
        "timestamp": int(time.time()),
    }
    with open(os.path.join(session_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[main] Results saved to {session_dir}/")


def main():
    args = parse_args()
    condition = args.condition

    # ── Initialize environment ───────────────────────────────────────────────
    env = SpaceDebrisEnv(condition=condition)
    renderer = Renderer(width=900, height=700)
    pygame.display.set_caption("Space Debris Capture")

    # ── Try to connect Haply device (always, regardless of condition) ────────
    physics = None
    haply_connected = False

    physics, haply_connected = try_init_haply()
    if haply_connected:
        print(f"[main] Haply connected on {physics.port[0]}.")
        # Prime the device: write zeros so data_available() returns True
        # on the first loop iteration (Haply only sends data after receiving torques)
        send_haply_force(physics, np.zeros(2))
    else:
        print("[main] No Haply device found — using mouse input.")
        if condition in ('haply', 'haply_haptics'):
            print(f"[main] Falling back from '{condition}' to 'mouse'.")
            condition = 'mouse'
            env.set_condition(condition)

    # ── Session state ────────────────────────────────────────────────────────
    show_gp_confidence = False
    show_haptics_panel = True   # press H to toggle
    all_trial_metrics = []
    session_ts = int(time.time())
    session_dir = os.path.join(_PARENT, "results", f"space_session_{session_ts}")

    condition_idx = CONDITIONS.index(condition) if condition in CONDITIONS else 0

    clock = pygame.time.Clock()
    running = True

    while running:
        # ── Event handling ───────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                key = event.key

                # Quit
                if key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                # SPACE: grab or release
                elif key == pygame.K_SPACE:
                    if env.phase == 'APPROACHING':
                        grabbed = env.try_grab()
                        if grabbed:
                            print("[main] Debris GRABBED")
                    elif env.phase == 'GRABBED':
                        deposited = env.try_release()
                        if deposited:
                            print("[main] Debris DEPOSITED — SUCCESS!")
                        else:
                            print("[main] Debris dropped (not in box)")

                # R: toggle recording
                elif key == pygame.K_r:
                    if not env.recording:
                        env.start_recording()
                        print("[main] Recording started")
                    else:
                        env.stop_recording()
                        print(f"[main] Recording stopped. "
                              f"Total demos: {env.gp_policy.n_demos}")

                # G: train GP
                elif key == pygame.K_g:
                    if env.gp_policy.n_demos > 0:
                        print(f"[main] Training GP on {env.gp_policy.n_demos} demos...")
                        env.train_gp()
                        print("[main] GP training complete.")
                    else:
                        print("[main] No demos recorded yet.")

                # T: toggle confidence bar
                elif key == pygame.K_t:
                    show_gp_confidence = not show_gp_confidence

                # N: cycle condition
                elif key == pygame.K_n:
                    condition_idx = (condition_idx + 1) % len(CONDITIONS)
                    condition = CONDITIONS[condition_idx]
                    # Only switch to haply modes if device connected
                    if condition in ('haply', 'haply_haptics') and not haply_connected:
                        condition = 'mouse'
                        condition_idx = 0
                    env.set_condition(condition)
                    print(f"[main] Condition switched to: {condition}")

                # H: toggle haptics tuning panel
                elif key == pygame.K_h:
                    show_haptics_panel = not show_haptics_panel

                # ── Haptic parameter tuning (×1.2 / ÷1.2 per keypress) ──────
                # [ / ]  →  door wall stiffness
                elif key == pygame.K_LEFTBRACKET:
                    env.haptics.DOOR_WALL_K = max(10.0,
                        env.haptics.DOOR_WALL_K / 1.2)
                    print(f"[haptics] DOOR_WALL_K = {env.haptics.DOOR_WALL_K:.1f} N/m")
                elif key == pygame.K_RIGHTBRACKET:
                    env.haptics.DOOR_WALL_K = min(800.0,
                        env.haptics.DOOR_WALL_K * 1.2)
                    print(f"[haptics] DOOR_WALL_K = {env.haptics.DOOR_WALL_K:.1f} N/m")

                # , / .  →  debris attraction
                elif key == pygame.K_COMMA:
                    env.haptics.ATTRACT_K = max(1e-5,
                        env.haptics.ATTRACT_K / 1.2)
                    print(f"[haptics] ATTRACT_K  = {env.haptics.ATTRACT_K:.5f}")
                elif key == pygame.K_PERIOD:
                    env.haptics.ATTRACT_K = min(0.02,
                        env.haptics.ATTRACT_K * 1.2)
                    print(f"[haptics] ATTRACT_K  = {env.haptics.ATTRACT_K:.5f}")

                # ; / '  →  spin gain
                elif key == pygame.K_SEMICOLON:
                    env.haptics.SPIN_GAIN = max(0.0,
                        env.haptics.SPIN_GAIN / 1.2)
                    print(f"[haptics] SPIN_GAIN  = {env.haptics.SPIN_GAIN:.3f} N")
                elif key == pygame.K_QUOTE:
                    env.haptics.SPIN_GAIN = min(5.0,
                        env.haptics.SPIN_GAIN * 1.2)
                    print(f"[haptics] SPIN_GAIN  = {env.haptics.SPIN_GAIN:.3f} N")

                # - / =  →  global force clip
                elif key == pygame.K_MINUS:
                    env.haptics.FORCE_CLIP = max(0.5,
                        env.haptics.FORCE_CLIP - 0.25)
                    print(f"[haptics] FORCE_CLIP = {env.haptics.FORCE_CLIP:.2f} N")
                elif key == pygame.K_EQUALS:
                    env.haptics.FORCE_CLIP = min(8.0,
                        env.haptics.FORCE_CLIP + 0.25)
                    print(f"[haptics] FORCE_CLIP = {env.haptics.FORCE_CLIP:.2f} N")

                # BACKSPACE  →  reset all haptics to defaults
                elif key == pygame.K_BACKSPACE:
                    from space_debris.haptics import SpaceHaptics as _SH
                    env.haptics.DOOR_WALL_K = _SH.DOOR_WALL_K
                    env.haptics.ATTRACT_K   = _SH.ATTRACT_K
                    env.haptics.SPIN_GAIN   = _SH.SPIN_GAIN
                    env.haptics.FORCE_CLIP  = _SH.FORCE_CLIP
                    print("[haptics] All parameters reset to defaults")

                # ENTER: reset debris after SUCCESS or FAIL
                elif key == pygame.K_RETURN:
                    if env.phase in ('SUCCESS', 'FAIL'):
                        # Store metrics for this trial
                        arm_traj, debris_traj = env.get_trial_trajectories()
                        if len(arm_traj) > 0:
                            m = compute_trial_metrics(
                                arm_traj, debris_traj,
                                env.DEPOSIT_BOX_CENTER,
                                env.DEPOSIT_BOX_SIZE,
                            )
                            m['condition'] = condition
                            all_trial_metrics.append(m)
                            print(f"[main] Trial metrics: {m}")

                        env.reset()
                        renderer.clear_trails()
                        print("[main] Debris reset.")

        # ── WRITE force first (Haply requires write before data is available) ──
        # Use the force computed in the previous frame. On the first frame this
        # is zeros (sent during initialisation above).
        if haply_connected and physics is not None and condition == 'haply_haptics':
            send_haply_force(physics, env.get_haptic_force())
        elif haply_connected and physics is not None:
            # Keep device alive with zero torque so data_available() stays True
            send_haply_force(physics, np.zeros(2))

        # ── READ position (data is now available after the write above) ───────
        if haply_connected and physics is not None and condition in ('haply', 'haply_haptics'):
            tip_pos = get_haply_position(physics)
            if tip_pos is None:
                tip_pos = mouse_to_physical(pygame.mouse.get_pos())
        else:
            tip_pos = mouse_to_physical(pygame.mouse.get_pos())

        # ── Step environment ─────────────────────────────────────────────────
        env.step(tip_pos)

        # ── Render ───────────────────────────────────────────────────────────
        renderer.begin_frame()

        # GP confidence for display
        gp_confidence = None
        if show_gp_confidence and env.gp_policy.trained:
            state = env.get_state_vector()
            _, std = env.gp_policy.predict(state)
            # Map std to confidence: lower std → higher confidence
            std_max = 1.0
            std_min = 0.05
            gp_confidence = float(np.clip(
                (std_max - std) / (std_max - std_min), 0.0, 1.0))

        renderer.draw_scene(
            debris=env.debris,
            arm=env.arm,
            deposit_box=env.deposit_box,
            haptics=env.haptics,
            phase=env.phase,
            condition=condition,
            gp_policy=env.gp_policy,
        )

        renderer.draw_hud(
            phase=env.phase,
            condition=condition,
            n_demos=env.gp_policy.n_demos,
            gp_trained=env.gp_policy.trained,
            gp_confidence=gp_confidence,
        )

        if show_haptics_panel:
            renderer.draw_haptics_panel(env.haptics)

        renderer.end_frame()

        # ── Cap at 60 FPS ────────────────────────────────────────────────────
        clock.tick(60)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    # Save any remaining trial data
    arm_traj, debris_traj = env.get_trial_trajectories()
    if len(arm_traj) > 0:
        m = compute_trial_metrics(
            arm_traj, debris_traj,
            env.DEPOSIT_BOX_CENTER,
            env.DEPOSIT_BOX_SIZE,
        )
        m['condition'] = condition
        all_trial_metrics.append(m)

    save_results(env, session_dir, condition, all_trial_metrics)

    # Zero force before closing Haply
    if haply_connected and physics is not None:
        try:
            send_haply_force(physics, np.zeros(2))
            physics.close()
        except Exception:
            pass

    pygame.quit()


if __name__ == "__main__":
    main()

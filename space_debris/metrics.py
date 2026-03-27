# -*- coding: utf-8 -*-
"""
Metrics for the space debris capture task.
"""
import numpy as np


def compute_trial_metrics(trajectory, debris_trajectory, deposit_box_center,
                          deposit_box_size):
    """
    Compute metrics for a single trial.

    Parameters
    ----------
    trajectory          : Nx2 np.ndarray — arm tip positions over the trial
    debris_trajectory   : Nx2 np.ndarray — debris positions (parallel to trajectory)
    deposit_box_center  : np.ndarray (2,) — center of deposit box in meters
    deposit_box_size    : np.ndarray (2,) — [width, height] of deposit box in meters

    Returns
    -------
    dict with keys:
      'min_approach_dist' : float — minimum distance from arm tip to debris
      'capture_time'      : int   — frame index when grab happened (-1 if never)
      'deposit_time'      : int   — frame index when deposit happened (-1 if never)
      'path_length'       : float — total arm tip path length in meters
      'task_success'      : bool  — True if debris was deposited in box
    """
    trajectory = np.asarray(trajectory, dtype=float)
    debris_trajectory = np.asarray(debris_trajectory, dtype=float)
    deposit_box_center = np.asarray(deposit_box_center, dtype=float)
    deposit_box_size = np.asarray(deposit_box_size, dtype=float)

    n = min(len(trajectory), len(debris_trajectory))
    if n == 0:
        return {
            'min_approach_dist': float('inf'),
            'capture_time': -1,
            'deposit_time': -1,
            'path_length': 0.0,
            'task_success': False,
        }

    trajectory = trajectory[:n]
    debris_trajectory = debris_trajectory[:n]

    # Distances from arm tip to debris at each frame
    dists = np.linalg.norm(trajectory - debris_trajectory, axis=1)
    min_approach_dist = float(np.min(dists))

    # Path length of arm tip
    if n > 1:
        steps = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        path_length = float(np.sum(steps))
    else:
        path_length = 0.0

    # Capture time: first frame where arm tip and debris are very close
    # (use grab radius as threshold)
    GRAB_RADIUS = 0.018
    capture_frames = np.where(dists <= GRAB_RADIUS)[0]
    capture_time = int(capture_frames[0]) if len(capture_frames) > 0 else -1

    # Deposit time and task success: check if debris ever entered box
    half = deposit_box_size * 0.5
    in_box = (np.abs(debris_trajectory[:, 0] - deposit_box_center[0]) <= half[0]) & \
             (np.abs(debris_trajectory[:, 1] - deposit_box_center[1]) <= half[1])
    deposit_frames = np.where(in_box)[0]
    deposit_time = int(deposit_frames[0]) if len(deposit_frames) > 0 else -1
    task_success = bool(len(deposit_frames) > 0)

    return {
        'min_approach_dist': min_approach_dist,
        'capture_time': capture_time,
        'deposit_time': deposit_time,
        'path_length': path_length,
        'task_success': task_success,
    }

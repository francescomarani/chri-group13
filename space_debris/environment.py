# -*- coding: utf-8 -*-
"""
SpaceDebrisEnv — main simulation environment for the space debris capture task.

State machine: IDLE → RECORDING → REVIEWING → TRAINING → IDLE

Task phases within a trial:
  APPROACHING → GRABBED → (deposit attempt) → SUCCESS / FAIL → APPROACHING
"""
import numpy as np

from .orbital import OrbitalDebris
from .arm import Arm
from .haptics import SpaceHaptics
from .gp_policy import GPPolicy


class SpaceDebrisEnv:
    """
    Gymnasium-style environment for space debris capture.

    Manages debris physics, arm state, haptic forces, demonstration
    recording, and GP policy training.
    """

    # Physical constants
    DEBRIS_START_POS = np.array([-0.04, -0.03])
    DEBRIS_START_VEL = np.array([0.003, 0.001])
    DEPOSIT_BOX_CENTER = np.array([0.0, 0.05])
    DEPOSIT_BOX_SIZE = np.array([0.025, 0.02])
    SATELLITE_CENTER = np.array([0.0, 0.05])
    SATELLITE_SIZE = np.array([0.05, 0.03])
    SHOULDER_POS = np.array([0.0, 0.035])
    # Door channel geometry (matches SpaceHaptics constants)
    DOOR_HALF_W = 0.014       # m — half-width of door opening
    DOOR_Y_BOT = 0.014        # m — bottom of channel
    DOOR_Y_TOP = 0.042        # m — top of channel (just above satellite bottom)

    # Simulation timestep
    DT = 1.0 / 60.0

    def __init__(self, condition='mouse'):
        """
        Parameters
        ----------
        condition : str — 'mouse' | 'haply' | 'haply_haptics'
        """
        self.condition = condition

        # Create subsystems
        self.debris = OrbitalDebris(
            pos=self.DEBRIS_START_POS.copy(),
            vel=self.DEBRIS_START_VEL.copy(),
            mass=0.5,
            radius=0.010,
            spin_rate=2.0,
            n=0.08,
        )

        self.arm = Arm(shoulder_pos=self.SHOULDER_POS.copy())

        self.deposit_box = {
            'center': self.DEPOSIT_BOX_CENTER.copy(),
            'size': self.DEPOSIT_BOX_SIZE.copy(),
            'satellite_center': self.SATELLITE_CENTER.copy(),
            'satellite_size': self.SATELLITE_SIZE.copy(),
            'door_half_w': self.DOOR_HALF_W,
            'door_y_bot': self.DOOR_Y_BOT,
            'door_y_top': self.DOOR_Y_TOP,
        }

        self.haptics = SpaceHaptics(
            deposit_box_center=self.DEPOSIT_BOX_CENTER.copy(),
            deposit_box_size=self.DEPOSIT_BOX_SIZE.copy(),
            debris=self.debris,
        )
        self.haptics.set_condition(condition)

        self.gp_policy = GPPolicy()

        # Task phase
        self.phase = 'APPROACHING'

        # State machine
        self.sim_state = 'IDLE'   # 'IDLE' | 'RECORDING' | 'REVIEWING' | 'TRAINING'
        self.recording = False

        # Demo buffers
        self.current_demo_states = []          # list of 5D np.arrays
        self.current_demo_actions = []         # list of 2D np.arrays
        self.current_demo_arm_positions = []   # list of np.arrays (for metrics)
        self.current_demo_debris_positions = []

        # Last computed haptic force
        self._last_haptic_force = np.zeros(2)

        # Internal dt
        self.dt = self.DT

        # Trial trajectory buffers (reset each trial)
        self._trial_arm_traj = []
        self._trial_debris_traj = []

    # ── Public API ───────────────────────────────────────────────────────────

    def reset(self):
        """
        Reset debris to initial position/velocity.
        Arm stays where it is. Trails and trial buffers are cleared.
        """
        self.debris.reset(self.DEBRIS_START_POS.copy(), self.DEBRIS_START_VEL.copy())
        self.phase = 'APPROACHING'
        self._trial_arm_traj = []
        self._trial_debris_traj = []
        self._last_haptic_force = np.zeros(2)

    def step(self, arm_tip_pos):
        """
        Advance simulation by one timestep.

        Parameters
        ----------
        arm_tip_pos : np.ndarray shape (2,) — from Haply or mouse

        Returns
        -------
        haptic_force : np.ndarray shape (2,) in Newtons
        """
        arm_tip_pos = np.asarray(arm_tip_pos, dtype=float)
        dt = self.dt

        # 1. Update arm (position and velocity)
        self.arm.update(arm_tip_pos, dt)

        # 2. Advance debris physics
        if self.debris.grabbed:
            # 3. Grabbed: keep debris at arm tip
            self.debris.update_grabbed(self.arm.tip_pos, self.arm.tip_vel, dt)
        else:
            self.debris.step(dt)

        # 4. Phase transition checks
        self._check_phase_transitions()

        # 5. Record demonstration data if active
        if self.recording and self.sim_state == 'RECORDING':
            phase_val = 1.0 if self.phase == 'GRABBED' else 0.0
            state_vec = GPPolicy.build_state(
                self.arm.tip_pos,
                self.arm.tip_vel,
                self.debris.pos,
                self.debris.vel,
                phase_val,
            )
            action_vec = GPPolicy.normalize_action(self.arm.tip_vel)
            self.current_demo_states.append(state_vec.copy())
            self.current_demo_actions.append(action_vec.copy())
            self.current_demo_arm_positions.append(self.arm.tip_pos.copy())
            self.current_demo_debris_positions.append(self.debris.pos.copy())

        # 6. Append to trial trajectories
        self._trial_arm_traj.append(self.arm.tip_pos.copy())
        self._trial_debris_traj.append(self.debris.pos.copy())

        # 7. Compute haptic force
        force = self.haptics.compute(self.arm.tip_pos, self.arm.tip_vel)
        self._last_haptic_force = force

        return force

    def try_grab(self):
        """
        User pressed grab button. Attempt to grab debris if in APPROACHING phase.

        Returns
        -------
        bool : True if grab succeeded
        """
        if self.phase != 'APPROACHING':
            return False

        success = self.arm.try_grab(self.debris)
        if success:
            self.phase = 'GRABBED'
        return success

    def try_release(self):
        """
        User pressed grab button while holding. Release debris.

        Returns
        -------
        bool : True if deposited in box
        """
        if self.phase != 'GRABBED':
            return False

        deposited = self.arm.try_release(self.debris, self.deposit_box)
        if deposited:
            self.phase = 'SUCCESS'
        else:
            # Dropped outside box — re-enters free drift
            self.phase = 'APPROACHING'
        return deposited

    def start_recording(self):
        """Begin recording a demonstration."""
        self.current_demo_states = []
        self.current_demo_actions = []
        self.current_demo_arm_positions = []
        self.current_demo_debris_positions = []
        self.recording = True
        self.sim_state = 'RECORDING'

    def stop_recording(self):
        """End recording; store demo in GP policy if enough data was collected."""
        self.recording = False
        self.sim_state = 'REVIEWING'

        if len(self.current_demo_states) > 10:
            states = np.array(self.current_demo_states)
            actions = np.array(self.current_demo_actions)
            self.gp_policy.add_demo(states, actions)

        # Reset buffers
        self.current_demo_states = []
        self.current_demo_actions = []
        self.current_demo_arm_positions = []
        self.current_demo_debris_positions = []

    def train_gp(self):
        """Train GP policy on all recorded demos."""
        if self.gp_policy.n_demos == 0:
            return
        self.sim_state = 'TRAINING'
        self.gp_policy.train()
        self.sim_state = 'IDLE'

    def get_state_vector(self):
        """
        Return current normalized 5D state for GP query.

        Returns
        -------
        np.ndarray shape (5,)
        """
        phase_val = 1.0 if self.phase == 'GRABBED' else 0.0
        return GPPolicy.build_state(
            self.arm.tip_pos,
            self.arm.tip_vel,
            self.debris.pos,
            self.debris.vel,
            phase_val,
        )

    def get_haptic_force(self):
        """Return last computed haptic force (np.ndarray shape (2,))."""
        return self._last_haptic_force.copy()

    def set_condition(self, condition):
        """Update experimental condition."""
        self.condition = condition
        self.haptics.set_condition(condition)

    def get_trial_trajectories(self):
        """
        Return trial trajectory arrays for metrics computation.

        Returns
        -------
        arm_traj    : np.ndarray (N, 2)
        debris_traj : np.ndarray (N, 2)
        """
        if len(self._trial_arm_traj) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))
        return (np.array(self._trial_arm_traj),
                np.array(self._trial_debris_traj))

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _check_phase_transitions(self):
        """
        Handle automatic phase transitions.
        Manual transitions (grab/release) are triggered via try_grab/try_release.
        """
        # If in SUCCESS state, keep it (UI will reset)
        # If in APPROACHING and debris was grabbed externally — shouldn't happen
        # (grab is always manual via try_grab)
        pass

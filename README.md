# PA3 — Hull-Breach Kinesthetic Teaching
### RO47013 Control in Human-Robot Interaction — TU Delft

This project studies how different input/feedback conditions affect
kinesthetic teaching quality in a Mars habitat hull-breach sealing task.
The operator demonstrates a trajectory through the crack corridor, then a
trajectory model is trained from the collected demonstrations.

The application has:
- a left haptic panel for the Haply device / mouse control
- a right VR panel showing the habitat hull-breach scenario
- per-session metric saving and NASA-TLX support

---

## Experimental Conditions

The codebase is now organized around **six explicit conditions**.
Select them with keys `1` to `6` while in `IDLE` and with no demos recorded.

| Key | Condition | Input | Haptic feedback |
|---|---|---|---|
| `1` | Mouse only, no haptics | Mouse | None |
| `2` | Haptic device, no feedback | Haply | None |
| `3` | Haptic device, virtual walls only | Haply | Virtual wall feedback only |
| `4` | Haptic device, fixed central guide | Haply | A constant centerline haptic guide |
| `5` | Haptic device, fading central guide | Haply | A centerline guide that becomes weaker as confidence in the learned trajectory increases |
| `6` | Haptic device, increasing learned guidance | Haply | Guidance toward the learned trajectory that becomes stronger as confidence increases |

### Notes on Conditions 4, 5 and 6

- Condition `4` is a fixed central guidance baseline.
- Condition `5` starts from a centerline scaffold and fades that scaffold as the learned model becomes more reliable.
- Condition `6` uses the learned trajectory itself as the haptic guide, and its stiffness increases with local confidence.
- Only condition `3` provides virtual wall feedback. Conditions `4`–`6` are guidance conditions, not wall conditions.
- In `free` mode the trajectory model is trained manually with `G`.
- In `validation` mode the condition is finalized automatically once the required number of demonstrations has been collected.

This makes it possible to compare:
- a prior-based assistive cue
- a data-driven assistive cue

---

## Setup

### Create the Conda environment

```bash
cd /home/simone/chri-group13
conda env create -f environment.yml
conda activate chri-pa3
```

If the environment already exists and you want to refresh it:

```bash
cd /home/simone/chri-group13
conda env update -f environment.yml --prune
conda activate chri-pa3
```

### Run the project

#### Free mode

```bash
cd /home/simone/chri-group13
python PA3_main.py --mode free
```

In free mode:
- you choose how many trials to perform
- you choose how many demonstrations to collect before training with `G`
- NASA-TLX and analysis plots are not generated automatically

#### Validation mode

Launch everything directly from the command line:

```bash
cd /home/simone/chri-group13
python PA3_main.py --mode validation --participant-count 12 --required-demos 3
```

Or let the script ask for missing values at startup:

```bash
cd /home/simone/chri-group13
python PA3_main.py --mode validation
```

In validation mode:
- the required number of demonstrations per condition is fixed by `--required-demos`
- a single run can execute the full experiment for all participants
- `--participant-count` is the number of participants to run in this execution
- after the condition is completed, the app automatically:
  `1.` trains the trajectory model
  `2.` opens the NASA-TLX questionnaire
  `3.` saves the collected data
  `4.` updates aggregate metrics and plots in the analysis folder
  `5.` advances to the next condition, and after the last condition to the next participant

---

## Project Structure

| File | Description |
|---|---|
| `PA3_main.py` | Main state machine, condition presets, data collection, saving |
| `haptics.py` | Virtual walls, fixed/fading centerline guidance, and confidence-scaled learned-trajectory guidance |
| `gp_trajectory.py` | Trajectory learning from demonstrations |
| `targets.py` | Tube/crack geometry and progress computation |
| `Graphics.py` | Haptic panel and Mars habitat hull-breach VR rendering |
| `metrics.py` | Quantitative trajectory metrics |
| `analyze_results.py` | Aggregates saved trials and generates summary tables / plots |
| `nasa_tlx.py` | NASA-TLX questionnaire |
| `Physics.py` | Haply device interface |
| `HaplyHAPI.py` | Low-level Haply API |
| `environment.yml` | Conda environment definition |

---

## Controls

| Key | Action |
|---|---|
| `1`–`6` | Select experimental condition in free mode (`IDLE`, no demos only) |
| `SPACE` | Start / stop recording a demonstration |
| `ENTER` | Confirm demo / continue, or finalize the condition in validation mode |
| `D` | Delete the last demo in `REVIEW` |
| `G` | Train the trajectory model on all demos |
| `M` | Toggle between free mode and validation mode from a clean idle state |
| `P` | Replay the learned trajectory |
| `A` | Autonomous replay with the PD controller |
| `N` | Open NASA-TLX after training |
| `T` | Cycle the crack/tube shape in free mode (`IDLE`, no demos only) |
| `C` | Clear all demos and reset the current session |
| `R` | Toggle linkage rendering |
| `Q` | Quit and save results |

---

## Usage Modes

### Free mode

Use this mode for prototyping, debugging, pilot tests, or open-ended demonstrations.

Typical workflow:

1. Start the program with `--mode free`.
2. Select a condition with `1`–`6`.
3. Record as many demonstrations as you want.
4. Train manually with `G` when you decide you have enough data.
5. Optionally replay with `P` or `A`.
6. Optionally open NASA-TLX with `N`.
7. Quit with `Q` to save the session.

### Validation mode

Use this mode for the actual experimental protocol. One launch can process the full participant pool.

Typical workflow:

1. Start the program with `--mode validation`.
2. The app starts from the first condition of the current participant and keeps the crack geometry fixed.
3. Record demonstrations with `SPACE` until the configured target is reached.
4. Accept each demo with `ENTER` or delete it with `D`.
5. When the target number of demos has been reached, press `ENTER` to finalize that condition.
6. The application automatically trains, opens NASA-TLX, saves the condition data, refreshes the analysis outputs, and moves to the next condition.
7. After the last condition, the app automatically resets for the next participant.
8. After the last participant, the run is marked complete and you can quit with `Q`.

Notes:
- conditions advance automatically in validation mode
- participants are handled internally as `participant_01`, `participant_02`, ...
- you can switch between `free` and `validation` with `M` only from a clean idle state

---

## Output

Results are saved to:

```text
free mode:       results/session_<timestamp>/
validation mode: results/validation_run_<timestamp>/
```

In free mode each session contains:

```text
summary.json
metrics.json
demo_1.npy
demo_2.npy
...
gp_trajectory.npy
gp_std.npy
```

In validation mode the structure is:

```text
results/validation_run_<timestamp>/
  run_summary.json
  all_metrics.json
  participant_01/
    participant_summary.json
    metrics.json
    condition_1_mouse_only_no_haptics/
      summary.json
      metrics.json
      demo_1.npy
      ...
```

The saved metadata now includes:
- `participant_number`
- `participant_count`
- `mode`
- `required_demos_target`
- `trial_index`
- `condition_id`
- `condition`
- `condition_label`
- `input_mode`
- `feedback_mode`
- `hardware_connected`

The saved trial metrics now also include:
- `completion_time_s`
- `mean_demo_time_s`
- `wall_hit_events_total`
- `wall_hit_events_mean`
- `per_demo_wall_hits`
- `demo_success_rate`
- `success`
- `demo_start_error`, `demo_end_error`
- `gp_start_error`, `gp_end_error`

---

## Analysis

To aggregate all saved sessions and generate CSV summaries plus plots:

```bash
cd /home/simone/chri-group13
python analyze_results.py --results-dir results --out-dir analysis
```

This creates:

```text
analysis/aggregate_metrics.csv
analysis/condition_summary.csv
analysis/friedman_tests.json
analysis/plots/*.png
```

If `participant_number` is available, the script also computes Friedman repeated-measures tests for the main metrics.

In validation mode this analysis step is also executed automatically after each completed condition and the run summary is refreshed after each participant.

---

## Dependencies

The environment installs:
- Python 3.11
- `numpy`
- `scipy`
- `pygame`
- `matplotlib`
- `pyserial`
- `tomlkit`
- `scikit-learn`

---

## Hardware Note

Conditions `2` to `6` are designed for the Haply device.
If no device is connected, the application can still fall back to mouse-based
interaction for testing, but that fallback is only intended for debugging and
not for the final experimental protocol.

# PA3 - Hull-Breach Kinesthetic Teaching
### RO47013 Control in Human-Robot Interaction - TU Delft

This project studies how different haptic guidance conditions affect kinesthetic teaching quality in a Mars habitat hull-breach sealing task. The operator demonstrates trajectories through a crack corridor, then a Gaussian Process (GP) trajectory model is learned from those demonstrations.

The application has:
- a left haptic panel for the Haply device or mouse fallback
- a right VR panel showing the Mars greenhouse / hull-breach scene
- validation-mode data logging for the experimental protocol
- a separate analysis script for CSV summaries, Friedman tests, and plots

---

## Experimental Conditions

The current experiment uses **five** conditions.

| ID | Code | Condition | Input | Feedback |
|---|---|---|---|---|
| `1` | `NH` | No haptics (baseline) | Haply | None |
| `2` | `VW` | Virtual walls | Haply | Virtual walls only |
| `3` | `CG` | Virtual walls + centerline guidance | Haply | Walls + fixed centerline groove |
| `4` | `FG` | Virtual walls + fading centerline guidance | Haply | Walls + centerline groove that fades with confidence |
| `5` | `SG` | Virtual walls + subjective trajectory guidance | Haply | Walls + learned GP guidance |

Notes:
- the old mouse-only experimental condition is no longer part of the protocol
- mouse is now only a fallback when the Haply device is unavailable
- virtual walls are active only while a demonstration is being recorded
- the GP is refreshed after each accepted demonstration so guidance conditions can use the latest learned model during validation

---

## Counterbalancing

Validation mode uses a 5-group Latin-square counterbalancing scheme for 20 participants. At startup, the script asks which group the participant belongs to unless `--group` is supplied on the command line.

| Group | Participants | Condition order |
|---|---|---|
| `A` | `1, 6, 11, 16` | `NH -> VW -> CG -> FG -> SG` |
| `B` | `2, 7, 12, 17` | `VW -> CG -> FG -> SG -> NH` |
| `C` | `3, 8, 13, 18` | `CG -> FG -> SG -> NH -> VW` |
| `D` | `4, 9, 14, 19` | `FG -> SG -> NH -> VW -> CG` |
| `E` | `5, 10, 15, 20` | `SG -> NH -> VW -> CG -> FG` |

The selected group is stored in the saved data as `validation_group`, together with the resolved `condition_order`.

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

---

## Running the App

### Free mode

```bash
cd /home/simone/chri-group13
python scripts/PA3_main.py --mode free
```

Free mode is intended for piloting, debugging, and open-ended trajectory collection. It does not compute validation metrics.

### Validation mode

Minimal launch:

```bash
cd /home/simone/chri-group13
python scripts/PA3_main.py --mode validation
```

Fully specified launch:

```bash
cd /home/simone/chri-group13
python scripts/PA3_main.py --mode validation --participant-number 7 --group B --required-demos 3
```

Current validation behavior:
- one launch handles exactly one participant
- if `--participant-number` is missing, the script asks for it at startup
- if `--group` is missing, the script asks for the counterbalancing group `A-E`
- if `--required-demos` is omitted, it defaults to `3`
- after the final condition, the application saves the run and closes automatically

If `--mode` is omitted entirely, the script first asks whether to start in `free` or `validation`.

---

## Validation Workflow

Validation mode is now designed for the actual experiment.

Typical workflow:

1. Launch `python scripts/PA3_main.py --mode validation`.
2. Enter the participant number and the Latin-square group if they were not passed by CLI.
3. The app starts from the first condition in that group's order.
4. Press `SPACE` to start recording and `SPACE` again to stop recording.
5. Press `ENTER` to accept the demo or `D` to discard the most recent one.
6. Repeat until the target number of demos is reached.
7. Press `ENTER` to train the final GP for that condition and review the learned trajectory.
8. Use `P` or `A` to replay the learned GP if needed.
9. Press `N` to fill NASA-TLX if it has not already been completed for that condition.
10. Press `ENTER` again to save the condition and advance automatically to the next one.
11. After the fifth condition, the participant run is saved and the application exits automatically.

Additional notes:
- during validation, conditions advance automatically according to the selected group
- `F` can interrupt validation and switch to `free`
- the left panel now shows a large status banner with current trial number and demo count
- the right panel still shows the detailed instructions and metric review text

---

## Controls

| Key | Action |
|---|---|
| `1`-`5` | Select condition in free mode only (`IDLE`, no demos recorded) |
| `SPACE` | Start / stop recording a demonstration |
| `ENTER` | Confirm demo, continue, or save / advance in validation |
| `D` | Delete the last demo |
| `G` | Train GP manually |
| `P` | Replay the GP trajectory |
| `A` | Autonomous GP replay with PD control |
| `N` | Open NASA-TLX |
| `F` | Interrupt validation and switch to free mode |
| `M` | Toggle mode from a clean idle state |
| `T` | Change crack / tube in free mode |
| `C` | Clear demos and reset the current condition |
| `R` | Toggle linkage rendering |
| `Q` | Quit and save the current session |

---

## Project Structure

| File | Description |
|---|---|
| `scripts/PA3_main.py` | Main state machine, experiment flow, saving, validation logic |
| `scripts/haptics.py` | Virtual walls and guidance logic |
| `scripts/gp_trajectory.py` | GP trajectory learning |
| `scripts/targets.py` | Tube / crack geometry |
| `scripts/Graphics.py` | Haptic panel and VR rendering |
| `scripts/metrics.py` | Validation metrics |
| `scripts/analyze_results.py` | Offline aggregation, statistics, and plots |
| `scripts/nasa_tlx.py` | NASA-TLX questionnaire |
| `scripts/Physics.py` | Haply interface and mouse fallback support |
| `scripts/HaplyHAPI.py` | Low-level Haply API |
| `assets/` | Images used by the UI |

---

## Saved Data

Results are saved under `results/`.

### Free mode

Free-mode sessions are saved to:

```text
results/session_<timestamp>/
```

Typical contents:

```text
summary.json
demo_1.npy
demo_2.npy
...
gp_trajectory.npy
gp_std.npy
```

Free mode is meant for exploration, so it does not compute the validation metrics used in the formal analysis.

### Validation mode

Validation runs are saved to:

```text
results/validation_run_<timestamp>/
```

Structure:

```text
results/validation_run_<timestamp>/
  run_summary.json
  all_metrics.json
  all_metrics.csv
  participant_01/
    participant_summary.json
    metrics.json
    metrics.csv
    condition_1_haply_no_feedback/
      summary.json
      metrics.json
      metrics.csv
      demo_1.npy
      demo_2.npy
      ...
      gp_trajectory.npy
      gp_std.npy
```

The saved metadata includes at least:
- `participant_number`
- `validation_group`
- `condition_order`
- `required_demos_target`
- `condition_id`
- `condition`
- `condition_label`
- `input_mode`
- `feedback_mode`
- `hardware_connected`
- `tube`
- `n_demos`

The validation metrics include:
- `pairwise_frechet_m`
- `path_length_ratio_mean`
- `jerk_mean`
- `gp_sigma_mean_m`
- `gp_sigma_by_demo_m`
- `path_length_ratio_by_demo`
- `jerk_by_demo`
- `demos_to_convergence`
- `cumulative_demo_time_to_convergence_s`
- `completion_time_s`
- `tlx_*` fields when NASA-TLX is completed

---

## Analysis

Run the analysis script after collecting as many participants as you want:

```bash
cd /home/simone/chri-group13
python scripts/analyze_results.py --results-dir results --out-dir analysis
```

By default, if a graphical display is available, the script:
- saves the plots to `analysis/plots/`
- shows the plots on screen

If you only want files and no plot windows:

```bash
python scripts/analyze_results.py --results-dir results --out-dir analysis --no-show
```

This creates:

```text
analysis/aggregate_metrics.csv
analysis/condition_summary.csv
analysis/friedman_tests.json
analysis/plots/*.png
```

Useful notes:
- the script scans `metrics.csv` files recursively under `--results-dir`
- if you want to analyze one specific batch only, point `--results-dir` to a single folder such as `results/validation_run_<timestamp>`
- descriptive summaries and plots can be generated for any number of participants
- Friedman repeated-measures tests are only meaningful when there are enough participants with complete data across all conditions

Analysis is no longer executed automatically by the validation app. Data collection and offline analysis are now deliberately separated.

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

These experimental conditions are designed for the Haply device.

If no compatible device is detected, the application falls back to mouse-based interaction so the software can still be tested. That fallback is useful for debugging, but it is not intended for the final experimental protocol.

#!/usr/bin/env python3
"""Aggregate PA3 session results and generate summary tables/plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib-codex"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare


DEFAULT_METRICS = [
    "pairwise_frechet_m",
    "path_length_ratio_mean",
    "jerk_mean",
    "gp_sigma_mean_m",
    "demos_to_convergence",
    "cumulative_demo_time_to_convergence_s",
    "completion_time_s",
    "tlx_overall",
]


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(results_dir.rglob("summary.json")):
        session_dir = summary_path.parent
        metrics_path = session_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        payload = json.loads(metrics_path.read_text())
        if isinstance(payload, dict):
            payload = [payload]

        for idx, record in enumerate(payload, start=1):
            row = dict(record)
            tlx = row.pop("tlx", None)
            if isinstance(tlx, dict):
                for key, value in tlx.items():
                    row[f"tlx_{key}"] = value

            row.setdefault("participant_number", summary.get("participant_number"))
            row.setdefault("condition", summary.get("condition"))
            row.setdefault("condition_label", summary.get("condition_label"))
            row.setdefault("input_mode", summary.get("input_mode"))
            row.setdefault("feedback_mode", summary.get("feedback_mode"))
            row.setdefault("tube", summary.get("tube"))
            row["session_folder"] = session_dir.name
            row["trial_in_file"] = idx
            if session_dir.parent.name.startswith("participant_"):
                row["participant_folder"] = session_dir.parent.name
            if "validation_run_" in str(session_dir):
                row["validation_run"] = next(
                    (part for part in session_dir.parts if part.startswith("validation_run_")),
                    None,
                )
            rows.append(row)
    return rows


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise_by_condition(rows: list[dict], metrics: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("condition_label", "unknown"), []).append(row)

    summaries: list[dict] = []
    for condition, items in grouped.items():
        for metric in metrics:
            values = [float(row[metric]) for row in items if is_number(row.get(metric)) or isinstance(row.get(metric), bool)]
            if not values:
                continue
            summaries.append({
                "condition_label": condition,
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    return summaries


def plot_metric(rows: list[dict], metric: str, out_path: Path) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if isinstance(value, bool):
            value = float(value)
        if is_number(value):
            grouped.setdefault(row.get("condition_label", "unknown"), []).append(float(value))

    if not grouped:
        return

    labels = list(grouped.keys())
    values = [grouped[label] for label in labels]

    plt.figure(figsize=(10, 5))
    plt.boxplot(values, labels=labels, patch_artist=True)
    for i, ys in enumerate(values, start=1):
        xs = np.random.normal(i, 0.04, size=len(ys))
        plt.scatter(xs, ys, s=20, alpha=0.7)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel(metric)
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def friedman_report(rows: list[dict], metrics: list[str]) -> list[dict]:
    results = []
    participant_rows = [row for row in rows if row.get("participant_number") is not None]
    if not participant_rows:
        return results

    conditions = sorted({row.get("condition_label", "unknown") for row in participant_rows})
    for metric in metrics:
        per_participant = {}
        for row in participant_rows:
            value = row.get(metric)
            if isinstance(value, bool):
                value = float(value)
            if not is_number(value):
                continue
            pid = row["participant_number"]
            cond = row.get("condition_label", "unknown")
            per_participant.setdefault(pid, {}).setdefault(cond, []).append(float(value))

        complete = []
        for pid, cond_map in per_participant.items():
            if all(cond in cond_map for cond in conditions):
                complete.append((pid, [float(np.mean(cond_map[cond])) for cond in conditions]))

        if len(complete) < 2 or len(conditions) < 2:
            continue

        arrays = list(zip(*[vals for _, vals in complete]))
        stat, p_value = friedmanchisquare(*arrays)
        results.append({
            "metric": metric,
            "participants": len(complete),
            "conditions": conditions,
            "friedman_statistic": float(stat),
            "p_value": float(p_value),
        })
    return results


def run_analysis(results_dir: str | Path = "results", out_dir: str | Path = "analysis") -> Path:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(results_dir)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {results_dir}")

    metrics = [metric for metric in DEFAULT_METRICS if any(metric in row for row in rows)]
    write_csv(out_dir / "aggregate_metrics.csv", rows)
    write_csv(out_dir / "condition_summary.csv", summarise_by_condition(rows, metrics))

    friedman = friedman_report(rows, metrics)
    with (out_dir / "friedman_tests.json").open("w") as f:
        json.dump(friedman, f, indent=2)

    for metric in metrics:
        plot_metric(rows, metric, plots_dir / f"{metric}.png")

    print(f"Analysis written to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Folder containing session_* and/or validation_run_* subfolders",
    )
    parser.add_argument("--out-dir", default="analysis", help="Where to write CSV files and plots")
    args = parser.parse_args()
    run_analysis(results_dir=args.results_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

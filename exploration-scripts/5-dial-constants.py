#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""5-dial-constants.py - derive the reward NORM z-score table from the freshest EDA run.

Writes derived_constants.json into the EDA run directory, which ppo-sb3-scripts/reward_functions.py and
analytical_policies.py read at import. It does not edit either of those files.
Both refuse to import if this step has not been run, so a failure here must abort the pipeline.

Run with no arguments from anywhere; paths resolve relative to this file.

Usage:
    python3.11 5-dial-constants.py
    python3.11 5-dial-constants.py --output derived_constants.json
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# The 15 NORM keys consumed by ppo-sb3-scripts/reward_functions.py, mapped to their EDA source columns.
METRIC_MAP = {
    # NORM key: (source column, is_delta)
    "delta_bytes_transmitted": ("oracle_bytes_transmitted", True),
    "delta_packets_transmitted": ("oracle_packets_transmitted", True),
    "delta_airtime_used_us": ("realistic_airtime_used_us", True),
    "queue_size_bytes": ("oracle_queue_size_bytes", False),
    "queue_size_packets": ("oracle_queue_size_packets", False),
    "bsr_queue_index": ("realistic_bsr_queue_ac_be", False),
    "delta_packets_enqueued": ("oracle_packets_enqueued", True),
    "delta_energy_mj": ("oracle_total_energy_consumed_mj", True),
    "delta_awake_time_ms": ("oracle_awake_time_ms", True),
    "delta_sleep_time_ms": ("oracle_sleep_time_ms", True),
    "duty_cycle": ("oracle_duty_cycle", False),
    "delta_drops_expired": ("oracle_mpdu_drops_expired", True),
    "fcs_error_count": ("realistic_fcs_error_count", False),
    "rx_fragment_count": ("realistic_rx_fragment_count", False),
    "last_rx_timestamp_us": ("realistic_last_rx_timestamp_us", False),
}

# Steps per EDA run
DEFAULT_STEPS_PER_SPAWN = 38

# Beacon intervals per RL decision step; mirrors TWT_UPDATE_INTERVAL_BI in twt-constants.h.
# Keep the two in sync.
BI_PER_DECISION_STEP = 20

NPZ_FEATURE_NAMES = [
    "bsr_queue_ac_be",
    "bsr_queue_ac_bk",
    "bsr_queue_ac_vi",
    "bsr_queue_ac_vo",
    "rssi_dbm",
    "snr_db",
    "last_rx_mcs",
    "bytes_received_at_ap",
    "packets_received_at_ap",
    "airtime_used_us",
    "fcs_error_count",
    "rx_fragment_count",
    "last_rx_timestamp_us",
    "oracle_duty_cycle",
    "oracle_awake_time_ms",
    "oracle_sleep_time_ms",
    "oracle_total_energy_mj",
    "oracle_packets_generated",
    "oracle_packets_enqueued",
    "oracle_packets_transmitted",
    "oracle_bytes_transmitted",
    "oracle_mpdu_drops_expired",
    "oracle_mpdu_drops_queue_full",
    "oracle_psdu_response_timeouts",
    "oracle_queue_size_packets",
    "oracle_queue_size_bytes",
    "oracle_avg_latency_ms",
]

NPZ_COLUMN_FOR_DELTA = {
    "delta_bytes_transmitted": "oracle_bytes_transmitted",
    "delta_packets_transmitted": "oracle_packets_transmitted",
    "delta_energy_mj": "oracle_total_energy_mj",
    "delta_drops_expired": "oracle_mpdu_drops_expired",
    "delta_awake_time_ms": "oracle_awake_time_ms",
    "delta_sleep_time_ms": "oracle_sleep_time_ms",
    "delta_airtime_used_us": "airtime_used_us",
    "delta_packets_enqueued": "oracle_packets_enqueued",
}


def check_npz_schema(states, npz_path: Path):
    """Confirm the stacked npz was written by the current stack-data.py layout."""
    n_feat = len(NPZ_FEATURE_NAMES)
    n_cols = states.shape[1]
    if n_cols % n_feat != 0:
        raise SystemExit(
            f"ERROR: {npz_path} has {n_cols} per-transition columns, which is not a "
            f"multiple of the {n_feat} features in NPZ_FEATURE_NAMES.\n"
            f"This EDA run predates the current stack-data.py feature list. Re-run the "
            f"collection + stacking steps, or check out the matching script revision."
        )
    return n_feat, n_cols // n_feat


def load_step_delta_stats(run_dir: Path) -> dict:
    """Per-step statistics for every delta metric recoverable from the stack."""
    npz_path = run_dir / "stacked_transitions.npz"
    if not npz_path.exists():
        print(
            f"WARNING: {npz_path.name} not found; every delta metric will use the "
            f"ramp-divisor fallback"
        )
        return {}

    with np.load(npz_path) as z:
        states = z["states"]
        n_feat, n_sta = check_npz_schema(states, npz_path)
        cur = states.reshape(-1, n_sta, n_feat)
        nxt = z["next_states"].reshape(-1, n_sta, n_feat)

    stats = {}
    for key, column in NPZ_COLUMN_FOR_DELTA.items():
        idx = NPZ_FEATURE_NAMES.index(column)
        delta = (nxt[:, :, idx] - cur[:, :, idx]).ravel()
        delta = delta[np.isfinite(delta) & (delta >= 0.0)]
        if delta.size == 0 or not np.any(delta > 0.0):
            print(
                f"WARNING: EDA column '{column}' is empty or all-zero; '{key}' falls "
                f"back to the ramp divisor. Check the ns-3 trace for this metric."
            )
            continue
        stats[key] = {
            "mean": float(delta.mean()),
            "std": float(delta.std()),
            "max": float(delta.max()),
        }
    return stats


def jains_fairness(values: "np.ndarray") -> "np.ndarray":
    """Per-step Jain's fairness index across STAs."""
    total = values.sum(axis=1)
    total_sq = (values**2).sum(axis=1)
    n = values.shape[1]
    return np.where(total_sq > 0.0, total**2 / (n * np.maximum(total_sq, 1e-12)), 1.0)


def load_center_stats(run_dir: Path) -> dict:
    """EDA-tuned reward center points: the empirical means the paper cites."""
    npz_path = run_dir / "stacked_transitions.npz"
    if not npz_path.exists():
        print(f"WARNING: {npz_path.name} not found; reward centers cannot be derived")
        return {}

    with np.load(npz_path) as z:
        n_feat, n_sta = check_npz_schema(z["states"], npz_path)
        cur = z["states"].reshape(-1, n_sta, n_feat)
        nxt = z["next_states"].reshape(-1, n_sta, n_feat)

    def delta(column: str):
        idx = NPZ_FEATURE_NAMES.index(column)
        return np.maximum(nxt[:, :, idx] - cur[:, :, idx], 0.0)

    d_bytes = delta("oracle_bytes_transmitted")
    d_awake = delta("oracle_awake_time_ms")
    d_sleep = delta("oracle_sleep_time_ms")
    sleep_ratio = d_sleep / np.maximum(d_awake + d_sleep, 1e-9)

    j_bytes = jains_fairness(d_bytes)
    j_sleep = jains_fairness(sleep_ratio)

    return {
        "jain_bytes_mean": float(j_bytes.mean()),
        "jain_bytes_std": float(j_bytes.std()),
        "jain_sleep_ratio_mean": float(j_sleep.mean()),
        "jain_sleep_ratio_std": float(j_sleep.std()),
    }


def calibrate_ramp_divisor(by_column: dict, step_stats: dict, n_steps: float) -> float:
    """Effective divisor turning an episode-wide cumulative mean into a per-step mean."""
    ratios = []
    for key, stat in step_stats.items():
        column = METRIC_MAP[key][0]
        row = by_column.get(column)
        if row and stat["mean"] > 0:
            ratios.append(float(row["mean"]) / stat["mean"])
    if not ratios:
        fallback = (n_steps + 1.0) / 2.0
        print(
            f"WARNING: no metric available to calibrate the ramp divisor; "
            f"using the analytic (n+1)/2 = {fallback:.2f}"
        )
        return fallback
    divisor = float(np.median(ratios))
    print(
        f"[5-dial-constants] ramp divisor calibrated on {len(ratios)} metrics: "
        f"{divisor:.2f} (range {min(ratios):.2f}-{max(ratios):.2f}, "
        f"analytic (n+1)/2 = {(n_steps + 1.0) / 2.0:.2f})"
    )
    return divisor


def find_latest_run(eda_data_dir: Path) -> Path:
    """Newest eda-data/run_*/ containing metric_quality_analysis.json.

    Matches 3-prepare-data.sh's own "latest run" convention (lexicographic
    sort of run_YYYYMMDD_HHMMSS names, which is also chronological order).
    """
    run_dirs = sorted(
        d
        for d in eda_data_dir.glob("run_*")
        if (d / "metric_quality_analysis.json").exists()
    )
    if not run_dirs:
        print(
            f"ERROR: no run_*/metric_quality_analysis.json found under {eda_data_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    return run_dirs[-1]


def load_analysis(path: Path) -> dict:
    with open(path) as f:
        rows = json.load(f)
    return {row["column"]: row for row in rows}


def detect_steps_per_spawn(by_column: dict) -> float:
    seq = by_column.get("realistic_observation_sequence_num")
    if seq and seq.get("unique", 0) > 0:
        return float(seq["unique"])
    print(
        f"WARNING: realistic_observation_sequence_num not found in analysis; "
        f"falling back to default STEPS_PER_SPAWN={DEFAULT_STEPS_PER_SPAWN}"
    )
    return float(DEFAULT_STEPS_PER_SPAWN)


def derive_norm(
    by_column: dict, n_steps: float, step_stats: dict, ramp_divisor: float
) -> dict:
    """Extract + transform the 15 NORM keys. Aborts (exit 1) if anything required is missing/invalid."""
    new_norm = {}
    problems = []

    for key, (column, is_delta) in METRIC_MAP.items():
        if is_delta and key in step_stats:
            stat = step_stats[key]
            new_norm[key] = {
                "mean": stat["mean"],
                "std": stat["std"] if stat["std"] > 0 else 1e-6,
                "max": stat["max"],
            }
            continue

        row = by_column.get(column)
        if row is None:
            problems.append(
                f"{key}: source column '{column}' not present in analysis file"
            )
            continue

        mean = float(row["mean"])
        std = float(row["std"])
        maxv = float(row["max"]) if row.get("max") is not None else None

        if not math.isfinite(mean):
            problems.append(f"{key}: mean is not finite ({mean})")
            continue
        if not math.isfinite(std):
            problems.append(f"{key}: std is not finite ({std})")
            continue

        if is_delta:
            print(
                f"NOTE: {key} has no populated per-step column; applying the "
                f"calibrated ramp divisor {ramp_divisor:.2f}"
            )
            mean = mean / ramp_divisor
            std = std / ramp_divisor
            if maxv is not None:
                maxv = maxv / ramp_divisor

        if std <= 0:
            print(f"WARNING: std for '{key}' ({column}) is {std}; clamping to 1e-6")
            std = 1e-6

        new_norm[key] = {"mean": mean, "std": std, "max": maxv}

    if problems:
        print(
            "ERROR: cannot derive NORM constants; missing/invalid required metrics:",
            file=sys.stderr,
        )
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    return new_norm


def derive_report_only(new_norm: dict, center_stats: dict) -> dict:
    """EDA-tuned scalars consumed by reward_functions.py and analytical_policies.py.

    lambda (arrival rate) is additionally published per beacon interval, since the
    delta metrics are per decision step of TWT_UPDATE_INTERVAL_BI beacon intervals
    while the analytical model in the paper is written per BI.
    """
    report = {}

    awake = new_norm.get("delta_awake_time_ms", {}).get("mean")
    sleep = new_norm.get("delta_sleep_time_ms", {}).get("mean")
    if awake is not None and sleep is not None and (awake + sleep) > 0:
        report["sleep_ratio_mean"] = sleep / (awake + sleep)

    enqueued = new_norm.get("delta_packets_enqueued", {}).get("mean")
    if enqueued is not None:
        report["arrival_rate_mean"] = enqueued
        report["arrival_rate_per_bi_mean"] = enqueued / BI_PER_DECISION_STEP

    transmitted = new_norm.get("delta_packets_transmitted", {}).get("mean")
    if transmitted is not None and awake is not None and awake > 0:
        report["service_rate_mean"] = transmitted / awake

    report.update(center_stats)

    return report


def print_derived_table(new_norm: dict, report_only: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'NORM key':<28} {'stat':<5} {'derived (this EDA)':>18}")
    print("-" * 78)
    for key in METRIC_MAP:
        new = new_norm.get(key, {})
        for stat in ("mean", "std", "max"):
            nv = new.get(stat)
            nv_s = f"{nv:,.4f}" if isinstance(nv, (int, float)) else "n/a"
            print(f"{key:<28} {stat:<5} {nv_s:>18}")
    print("-" * 78)
    print("EDA-tuned scalars (consumed by reward_functions / analytical_policies):")
    for k, v in report_only.items():
        v_s = f"{v:,.6f}" if isinstance(v, (int, float)) else "n/a"
        print(f"  {k:<28} {v_s}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive ppo-sb3-scripts/reward_functions.py NORM constants from the freshest EDA run."
    )
    parser.add_argument(
        "--eda-data-dir",
        default=str(SCRIPT_DIR / "eda-data"),
        help="Directory containing run_*/ subdirectories (default: exploration-scripts/eda-data)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit run_* directory to use instead of auto-detecting the newest one",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write derived_constants.json (default: inside the EDA run directory it was derived from)",
    )
    args = parser.parse_args()

    run_dir = (
        Path(args.run_dir) if args.run_dir else find_latest_run(Path(args.eda_data_dir))
    )
    analysis_path = run_dir / "metric_quality_analysis.json"
    print(f"[5-dial-constants] Reading {analysis_path}")
    by_column = load_analysis(analysis_path)

    n_steps = detect_steps_per_spawn(by_column)
    print(f"[5-dial-constants] steps_per_spawn={n_steps}")

    step_stats = load_step_delta_stats(run_dir)
    print(
        f"[5-dial-constants] per-step deltas recovered for "
        f"{len(step_stats)}/{sum(1 for _, d in METRIC_MAP.values() if d)} delta metrics"
    )
    ramp_divisor = calibrate_ramp_divisor(by_column, step_stats, n_steps)

    new_norm = derive_norm(by_column, n_steps, step_stats, ramp_divisor)
    center_stats = load_center_stats(run_dir)
    report_only = derive_report_only(new_norm, center_stats)

    derived = {
        "NORM": new_norm,
        "_report_only": report_only,
        "_source": str(run_dir),
        "_generated": datetime.now(timezone.utc).isoformat(),
    }

    # Written into the run directory the numbers came from, so the constants travel with
    # their EDA sweep instead of a single file at a fixed path being overwritten each run.
    output_path = Path(args.output) if args.output else run_dir / "derived_constants.json"
    with open(output_path, "w") as f:
        json.dump(derived, f, indent=2)
    print(f"[5-dial-constants] Wrote {output_path}")

    print_derived_table(new_norm, report_only)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""analyze_reward_signal.py - analyse reward signal quality and the NORM constants.

Reads the reward log CSVs written by RewardLoggerCallback during training and answers three questions:
1. Are NORM constants (mean/std) appropriate for the observed data?
2. Are reward components within expected ranges or saturating?
3. Are z-scores reasonable (mostly within [-3, 3])?

Writes a multi-panel PNG. The comparison baseline is the live NORM table from
derived_constants.json (located per-run by run_paths.derived_constants_path()), the same table
reward_functions.py normalises with, so the report is about the constants actually in use.

Usage:
    python3.11 analyze_reward_signal.py
    python3.11 analyze_reward_signal.py --preset throughput
    python3.11 analyze_reward_signal.py --output reward_analysis.png
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import run_paths

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "throughput": "#2ecc71",
    "energy": "#3498db",
    "queue": "#e74c3c",
    "default": "#95a5a6",
}

# --- Live NORM table, the constants the reward is actually normalised with ---

DERIVED_CONSTANTS_PATH = run_paths.derived_constants_path()


def _load_current_norm() -> dict:
    """Load the NORM table from derived_constants.json, written by 5-dial-constants.py.

    This is the same table reward_functions.py normalises with, so the mismatch report
    below describes the constants actually in use rather than a stale copy.

    Raises if the file is missing or has no NORM block; there is deliberately no
    hardcoded fallback, since a stale baseline silently inverts the report's verdict.
    """
    if not os.path.exists(DERIVED_CONSTANTS_PATH):
        raise FileNotFoundError(
            f"EDA constants file not found: {DERIVED_CONSTANTS_PATH}\n"
            "Run exploration-scripts/5-dial-constants.py before analysing reward logs."
        )
    with open(DERIVED_CONSTANTS_PATH) as f:
        derived = json.load(f)
    norm = derived.get("NORM", {})
    if not norm:
        raise KeyError(
            f"derived_constants.json has no NORM block (source: {DERIVED_CONSTANTS_PATH})"
        )
    generated = derived.get("_generated", "unknown")
    print(f"[analyze_reward_signal] loaded EDA constants from {DERIVED_CONSTANTS_PATH}")
    print(f"[analyze_reward_signal] constants generated: {generated}")
    return norm


CURRENT_NORM = _load_current_norm()

# Mapping from CSV column names to NORM keys.
# The CSV total_ columns are sums across all 16 STAs while NORM is per-STA, so those columns are divided by NUM_STAS before comparison.
NUM_STAS = 16

CSV_TO_NORM = {
    "total_bytes": "delta_bytes_transmitted",  # Needs /16
    "total_packets": "delta_packets_transmitted",  # Needs /16
    "total_airtime": "delta_airtime_used_us",  # Needs /16
    "avg_queue_bytes": "queue_size_bytes",  # Already avg
    "avg_queue_packets": "queue_size_packets",  # Already avg
    "avg_bsr_index": "bsr_queue_index",  # Already avg
    "total_energy_mj": "delta_energy_mj",  # Needs /16
    "avg_duty_cycle": "duty_cycle",  # Already avg
    "total_drops": "delta_drops_expired",  # Needs /16
    "avg_fcs_errors": "fcs_error_count",  # Already avg
    "avg_rx_fragments": "rx_fragment_count",  # Already avg
}

# Columns that need to be divided by NUM_STAS (they are totals, not averages)
TOTAL_COLUMNS = {
    "total_bytes",
    "total_packets",
    "total_airtime",
    "total_energy_mj",
    "total_drops",
}

# Z-score columns in CSV
ZSCORE_COLS = [
    "total_bytes_z",
    "total_packets_z",
    "total_airtime_z",
    "avg_queue_bytes_z",
    "avg_queue_packets_z",
    "avg_bsr_index_z",
    "total_energy_z",
    "avg_duty_cycle_z",
    "total_drops_z",
    "avg_fcs_errors_z",
    "avg_rx_fragments_z",
]

# Reward component columns
REWARD_COLS = [
    "total_reward",
    "w_throughput",
    "w_queue",
    "w_energy",
    "w_drops",
    "w_channel",
    "raw_throughput",
    "raw_queue",
    "raw_energy",
    "raw_drops",
    "raw_channel",
]


def load_and_stitch_reward_logs(reward_dir: str) -> pd.DataFrame:
    """Load and concatenate all reward log parts."""
    pattern = os.path.join(reward_dir, "reward_log_*_part*.csv")
    part_files = sorted(glob.glob(pattern))

    if not part_files:
        return None

    dfs = []
    for f in part_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded {os.path.basename(f)}: {len(df)} rows")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows from {len(part_files)} files")
    return combined


# --- Default training script, the checkpoint pattern is derived from its name ---
# train_X.py becomes X_twt_, so train_lstm_ppo_V1.py becomes lstm_ppo_V1_twt_.
DEFAULT_TRAINING_SCRIPT = "train_lstm_ppo_V1.py"


def derive_pattern_from_script(script_name: str) -> str:
    """Derive the checkpoint directory prefix from a training script name.

    Example: 'train_lstm_ppo_V1.py' -> 'lstm_ppo_V1_twt_'
    """
    pattern = script_name
    if pattern.startswith("train_"):
        pattern = pattern[6:]  # Drop the train_ prefix
    if pattern.endswith(".py"):
        pattern = pattern[:-3]  # Drop the .py suffix
    return f"{pattern}_twt_"


DEFAULT_CHECKPOINT_PATTERN = derive_pattern_from_script(DEFAULT_TRAINING_SCRIPT)


def find_checkpoint_dirs(base_dir: str, checkpoint_pattern: str = None) -> dict:
    """Find all checkpoint directories organized by preset.

    Args:
        base_dir: Directory containing checkpoint folders
        checkpoint_pattern: Pattern prefix (e.g., 'lstm_ppo_V1_twt_')
    """
    checkpoints = {}

    # Use provided pattern or default
    if checkpoint_pattern is None:
        checkpoint_pattern = DEFAULT_CHECKPOINT_PATTERN

    # Supported presets to search for
    presets = ["throughput", "energy", "queue"]

    # Search for each preset using the pattern
    for preset in presets:
        # Build glob pattern: pattern + preset + wildcard for timestamp
        pattern = os.path.join(base_dir, f"{checkpoint_pattern}{preset}_*")
        dirs = sorted(glob.glob(pattern), reverse=True)

        if dirs:
            # Keep the most recent checkpoint for this preset
            checkpoints[preset] = dirs[0]

    return checkpoints


def analyze_zscore_distribution(df: pd.DataFrame) -> dict:
    """Analyze z-score distributions to check NORM appropriateness."""
    results = {}

    for col in ZSCORE_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna()
        if len(values) == 0:
            continue

        results[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "pct_outside_3std": float((np.abs(values) > 3).mean() * 100),
            "pct_outside_2std": float((np.abs(values) > 2).mean() * 100),
            "skewness": float(values.skew()) if len(values) > 2 else 0,
        }

        # Z-score saturation analysis.
        # An extreme z-score, |z| above 2 or 3, drives tanh(z) toward ±1 and the reward signal loses its gradient.
        # tanh(2) ≈ 0.96, tanh(3) ≈ 0.995, so beyond 3 the component is effectively flat.
        pct_saturating = float((np.abs(values) > 2).mean() * 100)
        pct_heavily_saturated = float((np.abs(values) > 3).mean() * 100)
        results[col]["pct_saturating"] = pct_saturating
        results[col]["pct_heavily_saturated"] = pct_heavily_saturated
        results[col]["saturation_warning"] = (
            pct_saturating > 20
        )  # >20% saturating is bad

        # Check if NORM is appropriate
        # Good z-scores should have mean~0, std~1
        results[col]["norm_quality"] = "GOOD"
        if abs(results[col]["mean"]) > 1:
            results[col]["norm_quality"] = "BAD_MEAN"
        elif results[col]["std"] < 0.5 or results[col]["std"] > 2:
            results[col]["norm_quality"] = "BAD_STD"
        elif results[col]["pct_outside_3std"] > 5:
            results[col]["norm_quality"] = "OUTLIERS"
        elif results[col]["saturation_warning"]:
            results[col]["norm_quality"] = "SATURATING"

    return results


def analyze_reward_components(df: pd.DataFrame) -> dict:
    """Analyze reward component distributions."""
    results = {}

    for col in REWARD_COLS:
        if col not in df.columns:
            continue

        values = df[col].dropna()
        if len(values) == 0:
            continue

        results[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "range": float(values.max() - values.min()),
        }

        # Check for saturation: tanh bounds each component to ±1, so values near ±1 carry no gradient.
        if "raw_" in col:
            # Raw components use tanh, should be in (-1, 1)
            near_bounds = (np.abs(values) > 0.95).mean() * 100
            results[col]["pct_saturated"] = float(near_bounds)
            results[col]["saturation_warning"] = near_bounds > 10

    return results


def compute_suggested_norm(df: pd.DataFrame) -> dict:
    """Compute suggested NORM values based on observed data."""
    suggestions = {}

    for csv_col, norm_key in CSV_TO_NORM.items():
        if csv_col not in df.columns:
            continue

        values = df[csv_col].dropna()
        if len(values) == 0:
            continue

        # Compute robust statistics
        observed_mean = float(values.mean())
        observed_std = float(values.std())
        observed_max = float(values.max())

        # For "total_" columns, divide by NUM_STAS to get per-STA values
        # (NORM constants are defined per-STA, but CSV logs totals across all STAs)
        if csv_col in TOTAL_COLUMNS:
            observed_mean /= NUM_STAS
            observed_std /= NUM_STAS
            observed_max /= NUM_STAS

        # Get current NORM
        current = CURRENT_NORM.get(norm_key, {})
        current_mean = current.get("mean", observed_mean)
        current_std = current.get("std", observed_std)

        # Compute mismatch
        mean_ratio = observed_mean / current_mean if current_mean != 0 else float("inf")
        std_ratio = observed_std / current_std if current_std != 0 else float("inf")

        suggestions[norm_key] = {
            "current_mean": current_mean,
            "current_std": current_std,
            "observed_mean": observed_mean,
            "observed_std": observed_std,
            "observed_max": observed_max,
            "mean_ratio": mean_ratio,
            "std_ratio": std_ratio,
            "needs_update": abs(mean_ratio - 1) > 0.3 or abs(std_ratio - 1) > 0.3,
        }

    return suggestions


def plot_analysis(all_data: dict, output_path: str):
    """Create comprehensive analysis plots."""
    if not all_data:
        print("⚠ No data to plot")
        return []

    n_presets = len(all_data)
    # Always use 3 columns for consistent layout (reward over time spans 2, NORM summary uses 1)
    n_cols = 3

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, n_cols, hspace=0.35, wspace=0.3)

    # --- Row 1: Z-score distributions for each preset ---
    for idx, (preset, data) in enumerate(
        list(all_data.items())[:3]
    ):  # Max 3 presets in row
        ax = fig.add_subplot(gs[0, idx])
        df = data["df"]

        zscore_data = []
        zscore_labels = []
        for col in ZSCORE_COLS:
            if col in df.columns:
                zscore_data.append(df[col].dropna().values)
                zscore_labels.append(col.replace("_z", "").replace("_", "\n"))

        if zscore_data:
            bp = ax.boxplot(zscore_data, tick_labels=zscore_labels, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor(COLORS.get(preset, COLORS["default"]))
                patch.set_alpha(0.6)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.axhline(y=-3, color="red", linestyle="--", alpha=0.5)
        ax.axhline(y=3, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("Z-score", fontsize=10)
        ax.set_title(
            f"{preset.title()}: Z-score Distributions", fontsize=11, fontweight="bold"
        )
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_ylim(-5, 5)

    # --- Row 2: Reward component distributions ---
    for idx, (preset, data) in enumerate(list(all_data.items())[:3]):  # Max 3 presets
        ax = fig.add_subplot(gs[1, idx])
        df = data["df"]

        components = ["w_throughput", "w_queue", "w_energy", "w_drops", "w_channel"]
        comp_data = []
        comp_labels = []
        for col in components:
            if col in df.columns:
                comp_data.append(df[col].dropna().values)
                comp_labels.append(col.replace("w_", ""))

        if comp_data:
            bp = ax.boxplot(comp_data, tick_labels=comp_labels, patch_artist=True)
            colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.set_ylabel("Weighted Reward", fontsize=10)
        ax.set_title(
            f"{preset.title()}: Reward Components", fontsize=11, fontweight="bold"
        )

    # --- Row 3: Total reward over training ---
    ax_reward = fig.add_subplot(gs[2, :2])

    for preset, data in all_data.items():
        df = data["df"]
        if "total_reward" in df.columns:
            rewards = df["total_reward"].values
            # Smooth with rolling mean
            window = max(1, len(rewards) // 50)
            smoothed = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            ax_reward.plot(
                smoothed,
                color=COLORS.get(preset, "gray"),
                linewidth=2,
                label=preset.title(),
                alpha=0.8,
            )

    ax_reward.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax_reward.set_xlabel("Training Step", fontsize=11)
    ax_reward.set_ylabel("Total Reward", fontsize=11)
    ax_reward.set_title("Total Reward Over Training", fontsize=12, fontweight="bold")
    ax_reward.legend(loc="lower right")

    # --- Row 3 Right: NORM mismatch summary ---
    ax_norm = fig.add_subplot(gs[2, 2])

    # Collect all suggestions
    all_suggestions = {}
    for preset, data in all_data.items():
        for key, val in data["suggestions"].items():
            if key not in all_suggestions:
                all_suggestions[key] = {"ratios": [], "needs_update": False}
            all_suggestions[key]["ratios"].append(val["mean_ratio"])
            if val["needs_update"]:
                all_suggestions[key]["needs_update"] = True

    keys = list(all_suggestions.keys())
    avg_ratios = [np.mean(all_suggestions[k]["ratios"]) for k in keys]
    colors_bar = [
        "red" if all_suggestions[k]["needs_update"] else "green" for k in keys
    ]

    y_pos = np.arange(len(keys))

    # A ratio is multiplicative, so plot log2 of it: 0 is perfect, +1 is 2x over, -1 is 2x under.
    # That makes an over- and an under-estimate of the same factor mirror images, and it keeps
    # every bar on the canvas. A fixed linear limit used to truncate large ratios invisibly.
    LOG_CLAMP = 6.0  # Display ceiling, 2^6 = 64x in either direction
    log_ratios, clamped = [], []
    for r in avg_ratios:
        if not np.isfinite(r) or r <= 0:
            # current_mean was 0, or nothing was observed; park it at the ceiling and flag it.
            log_ratios.append(LOG_CLAMP if (np.isinf(r) or r > 1) else -LOG_CLAMP)
            clamped.append(True)
            continue
        lr = float(np.log2(r))
        clamped.append(abs(lr) > LOG_CLAMP)
        log_ratios.append(float(np.clip(lr, -LOG_CLAMP, LOG_CLAMP)))

    ax_norm.barh(y_pos, log_ratios, color=colors_bar, alpha=0.7)
    ax_norm.axvline(x=0, color="black", linestyle="-", linewidth=2)
    ax_norm.axvline(x=np.log2(0.7), color="orange", linestyle="--", alpha=0.5)
    ax_norm.axvline(x=np.log2(1.3), color="orange", linestyle="--", alpha=0.5)

    # Fit the axis to the data, but never tighter than the +-30% threshold bands.
    span = max(
        1.0, min(LOG_CLAMP, max((abs(v) for v in log_ratios), default=0.0) * 1.15)
    )
    ax_norm.set_xlim(-span, span)

    # Tick in ratios rather than log2 units, since that is what the reader is thinking in.
    # Thin the ticks as the span grows; this panel is narrow and 13 labels collide.
    step = 1 if span <= 3 else 2
    ticks = [
        t
        for t in range(-int(LOG_CLAMP), int(LOG_CLAMP) + 1)
        if abs(t) <= span and t % step == 0
    ]
    ax_norm.set_xticks(ticks)
    ax_norm.set_xticklabels(
        [f"{2 ** t:g}x" if t >= 0 else f"1/{2 ** -t:g}x" for t in ticks], fontsize=8
    )

    # Mark any bar that hit the ceiling, so a truncated bar is never read as an in-range one.
    # Anchored to draw inward, over the bar; drawn outward it collides with the y tick labels.
    for y, (lr, was_clamped) in enumerate(zip(log_ratios, clamped)):
        if was_clamped:
            ax_norm.text(
                lr,
                y,
                "»" if lr > 0 else "«",
                va="center",
                ha="right" if lr > 0 else "left",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

    # Single-line labels: newline-splitting these made 11 multi-line labels overlap each other.
    ax_norm.set_yticks(y_pos)
    ax_norm.set_yticklabels(
        [
            k.replace("delta_", "d_")
            .replace("_transmitted", "_tx")
            .replace("_expired", "_exp")
            .replace("_count", "")
            for k in keys
        ],
        fontsize=7,
    )
    ax_norm.set_xlabel("Observed / NORM mean, log scale", fontsize=10)
    ax_norm.set_title("NORM Accuracy\n(1x = perfect)", fontsize=11, fontweight="bold")

    # --- Row 4: Summary statistics table ---
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")

    # Build summary text
    summary_lines = ["=" * 80]
    summary_lines.append("NORM ANALYSIS SUMMARY")
    summary_lines.append("=" * 80)

    for preset, data in all_data.items():
        summary_lines.append(f"\n{preset.upper()} PRESET:")
        summary_lines.append("-" * 40)

        # Z-score quality
        zscore_results = data["zscore_analysis"]
        bad_zscores = [
            k for k, v in zscore_results.items() if v["norm_quality"] != "GOOD"
        ]
        if bad_zscores:
            summary_lines.append(
                f"  ⚠️  Z-scores need attention: {', '.join(bad_zscores)}"
            )
        else:
            summary_lines.append("  ✅ All z-scores within expected range")

        # Z-score saturation check (separate from norm quality)
        saturating_zscores = [
            k for k, v in zscore_results.items() if v.get("saturation_warning", False)
        ]
        if saturating_zscores:
            summary_lines.append(
                f"  ⚠️  Z-scores causing tanh saturation (>20% |z|>2): {', '.join(saturating_zscores)}"
            )

        # Saturation check for reward components
        reward_results = data["reward_analysis"]
        saturated = [
            k for k, v in reward_results.items() if v.get("saturation_warning", False)
        ]
        if saturated:
            summary_lines.append(
                f"  ⚠️  Saturating reward components: {', '.join(saturated)}"
            )
        else:
            summary_lines.append("  ✅ No reward saturation detected")

        # NORM suggestions
        needs_update = [k for k, v in data["suggestions"].items() if v["needs_update"]]
        if needs_update:
            summary_lines.append(
                f"  ⚠️  NORM update suggested for: {', '.join(needs_update)}"
            )
        else:
            summary_lines.append("  ✅ NORM constants are appropriate")

    summary_lines.append("\n" + "=" * 80)

    ax_table.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax_table.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.suptitle(
        "Reward Signal Analysis: NORM Constant Validation",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99,
        0.01,
        f"Generated: {timestamp}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    plt.savefig(
        output_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print(f"\nSaved: {output_path}")

    return summary_lines


def print_norm_suggestions(all_data: dict):
    """Print suggested NORM updates and z-score saturation analysis."""

    # First, print z-score saturation details
    print("\n" + "=" * 80)
    print("Z-SCORE SATURATION ANALYSIS")
    print("=" * 80)
    print("(When |z| > 2, tanh(z) > 0.96 → gradient vanishes, learning slows)")
    print()

    for preset, data in all_data.items():
        print(f"{preset.upper()}:")
        zscore_results = data["zscore_analysis"]

        print(
            f"  {'Z-Score Column':<25} {'Mean':>8} {'Std':>8} {'%|z|>2':>10} {'%|z|>3':>10} {'Status':<12}"
        )
        print(f"  {'-'*75}")

        for col, stats in zscore_results.items():
            status = (
                "✅ OK"
                if not stats.get("saturation_warning", False)
                else "⚠️ SATURATING"
            )
            if stats["norm_quality"] == "BAD_MEAN":
                status = "❌ BAD MEAN"
            elif stats["norm_quality"] == "BAD_STD":
                status = "❌ BAD STD"

            print(
                f"  {col:<25} {stats['mean']:>8.2f} {stats['std']:>8.2f} "
                f"{stats['pct_saturating']:>9.1f}% {stats['pct_heavily_saturated']:>9.1f}% {status:<12}"
            )
        print()

    # Then print NORM suggestions
    print("\n" + "=" * 80)
    print("SUGGESTED NORM UPDATES (if needed)")
    print("=" * 80)

    # Aggregate across all presets
    aggregated = {}
    for preset, data in all_data.items():
        for key, val in data["suggestions"].items():
            if key not in aggregated:
                aggregated[key] = {
                    "observed_means": [],
                    "observed_stds": [],
                    "observed_maxs": [],
                    "current_mean": val["current_mean"],
                    "current_std": val["current_std"],
                }
            aggregated[key]["observed_means"].append(val["observed_mean"])
            aggregated[key]["observed_stds"].append(val["observed_std"])
            aggregated[key]["observed_maxs"].append(val["observed_max"])

    print("\nNORM = {")
    for key, val in aggregated.items():
        avg_mean = np.mean(val["observed_means"])
        avg_std = np.mean(val["observed_stds"])
        avg_max = np.max(val["observed_maxs"])

        mean_ratio = avg_mean / val["current_mean"] if val["current_mean"] != 0 else 0
        needs_update = abs(mean_ratio - 1) > 0.3

        status = "# ⚠️  UPDATE" if needs_update else "# ✅ OK"
        print(
            f"    '{key}': {{'mean': {avg_mean:.1f}, 'std': {avg_std:.1f}, 'max': {avg_max:.1f}}},  {status}"
        )
    print("}")


def main():
    parser = argparse.ArgumentParser(description="Analyze reward signal quality")
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Directory containing checkpoint folders (default: checkpoints/, or checkpoints/$TWT_RUN_ID)",
    )
    parser.add_argument(
        "--training-script",
        type=str,
        default=DEFAULT_TRAINING_SCRIPT,
        help=f"Training script name to derive pattern (default: {DEFAULT_TRAINING_SCRIPT})",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["throughput", "energy", "queue"],
        help="Analyze only specific preset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/reward_signal_analysis.png",
        help="Output plot file",
    )

    args = parser.parse_args()

    # Derive checkpoint pattern from training script name
    checkpoint_pattern = derive_pattern_from_script(args.training_script)

    plots_dir = run_paths.artifact_dir("plots", create=True)
    ckpt_dir = run_paths.resolve(args.checkpoints_dir, "checkpoints")

    if not os.path.exists(ckpt_dir):
        print(f"Error: Checkpoints directory not found: {ckpt_dir}")
        sys.exit(1)

    print(f"Training script: {args.training_script}")
    print(f"Checkpoint pattern: {checkpoint_pattern}")
    checkpoints = find_checkpoint_dirs(ckpt_dir, checkpoint_pattern)

    if not checkpoints:
        print(
            f"No checkpoint directories found matching pattern: {checkpoint_pattern}*"
        )
        sys.exit(1)

    if args.preset:
        if args.preset in checkpoints:
            checkpoints = {args.preset: checkpoints[args.preset]}
        else:
            print(f"Error: Preset '{args.preset}' not found")
            sys.exit(1)

    print(f"Found {len(checkpoints)} preset(s): {list(checkpoints.keys())})")

    # Load and analyze each preset
    all_data = {}
    for preset, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {preset}")
        print("=" * 60)

        reward_dir = os.path.join(ckpt_path, "reward_logs")
        if not os.path.exists(reward_dir):
            print(f"  No reward_logs directory found, skipping...")
            continue

        df = load_and_stitch_reward_logs(reward_dir)
        if df is None or len(df) == 0:
            print(f"  No reward logs found, skipping...")
            continue

        all_data[preset] = {
            "df": df,
            "zscore_analysis": analyze_zscore_distribution(df),
            "reward_analysis": analyze_reward_components(df),
            "suggestions": compute_suggested_norm(df),
        }

    if not all_data:
        print("No data to analyze!")
        sys.exit(1)

    # Generate plots with pattern prefix in filename
    output_filename = f"reward_signal_{checkpoint_pattern}analysis.png"
    output_path = os.path.join(plots_dir, output_filename)
    summary = plot_analysis(all_data, output_path)

    # Print summary to console
    for line in summary:
        print(line)

    # Print NORM suggestions
    print_norm_suggestions(all_data)


if __name__ == "__main__":
    main()

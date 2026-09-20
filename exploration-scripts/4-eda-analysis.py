#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""4-eda-analysis.py - exploratory data analysis for the TWT RL dataset.

Performs:
1. Raw metric analysis (which NS-3 metrics carry real data rather than sentinels)
2. State feature analysis (distributions, correlations)
3. Action space exploration (coverage, reward by action)
4. Reward analysis (distribution, temporal patterns)
5. State-action-reward relationships

Writes plots plus metric_recommendations.json into eda-data/run_<timestamp>/, which
5-dial-constants.py then reads to derive the reward constants.

Usage:
    python3.11 4-eda-analysis.py
"""

import argparse
import json
import sys
import glob
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Try to import pandas for CSV analysis
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, raw metric analysis will be limited")

# Try to import plotting libraries
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


def parse_args():
    parser = argparse.ArgumentParser(description="EDA for TWT RL dataset")
    parser.add_argument(
        "--npz-file",
        type=str,
        default=None,
        help="Path to stacked_transitions.npz file (auto-detected if not provided)",
    )
    parser.add_argument(
        "--data-log-dir",
        type=str,
        default=None,
        help="Path to data-log directory with py-wrapper-env CSVs (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as NPZ file)",
    )
    parser.add_argument(
        "--schedule-table",
        type=str,
        default=None,
        help="Path to table_schedule.json for action labels",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--raw-metrics-only",
        action="store_true",
        help="Only analyze raw metrics from py-wrapper-env CSVs",
    )
    return parser.parse_args()


def load_data(npz_path: str) -> Dict[str, np.ndarray]:
    """Load NPZ data file."""
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


# --- RAW METRIC ANALYSIS - Check which NS-3 metrics have meaningful data ---


def classify_metric(
    col_name: str, values: np.ndarray, n_samples: int
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Classify a metric as GOOD, LIMITED, GARBAGE, or SENTINEL.

    Returns:
        (status, reason, stats_dict)
    """
    unique_count = len(np.unique(values))
    zero_count = (values == 0).sum()
    zero_pct = zero_count / n_samples * 100

    # Check for sentinel values
    sentinel_128 = (values == -128).sum()
    sentinel_65535 = (values == 65535).sum()
    sentinel_count = sentinel_128 + sentinel_65535
    sentinel_pct = sentinel_count / n_samples * 100

    # Calculate stats (handle all-same values)
    val_min = values.min()
    val_max = values.max()
    val_mean = values.mean()
    val_std = values.std()

    stats = {
        "min": float(val_min),
        "max": float(val_max),
        "mean": float(val_mean),
        "std": float(val_std),
        "unique": unique_count,
        "zero_pct": float(zero_pct),
        "sentinel_pct": float(sentinel_pct),
    }

    # Classification logic
    if unique_count == 1:
        if val_min == 0:
            return "GARBAGE", "Always zero", stats
        else:
            return "CONSTANT", f"Always {val_min}", stats

    if sentinel_pct > 40:
        return "SENTINEL", f"{sentinel_pct:.1f}% sentinel values (-128 or 65535)", stats

    if zero_pct > 95:
        return "GARBAGE", f"{zero_pct:.1f}% zeros", stats

    if unique_count == 2:
        return "LIMITED", f"Only 2 values: {np.unique(values).tolist()}", stats

    if unique_count <= 5:
        return "LIMITED", f"Only {unique_count} unique values", stats

    if unique_count > 10 and val_std > 0:
        return "GOOD", f"{unique_count} unique values, good variance", stats

    return "OK", f"{unique_count} unique values", stats


def analyze_raw_metrics(data_log_dir: Path, output_dir: Path, make_plots: bool):
    """
    Analyze raw metrics from py-wrapper-env CSV files to identify
    which metrics have meaningful data vs garbage.
    """
    print("\n" + "=" * 70)
    print("RAW NS-3 METRIC ANALYSIS")
    print("=" * 70)

    if not HAS_PANDAS:
        print("Error: pandas required for raw metric analysis")
        return None

    # Find py-wrapper-env CSV files
    csv_files = sorted(data_log_dir.glob("py-wrapper-env-*.csv"))
    if not csv_files:
        # Also try ns3-BI-log files
        csv_files = sorted(data_log_dir.glob("ns3-BI-log-*.csv"))

    if not csv_files:
        print(
            f"No py-wrapper-env-*.csv or ns3-BI-log-*.csv files found in {data_log_dir}"
        )
        return None

    print(f"Found {len(csv_files)} CSV files")

    # Load and concatenate all CSVs
    dfs = []
    for csv_file in csv_files:  # Load ALL files
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {csv_file.name}: {e}")

    if not dfs:
        print("No CSV files could be loaded")
        return None

    df = pd.concat(dfs, ignore_index=True)
    n_samples = len(df)
    print(f"Loaded {n_samples} samples from {len(dfs)} files")
    print(f"Columns: {len(df.columns)}")

    # Separate realistic and oracle columns
    realistic_cols = [c for c in df.columns if c.startswith("realistic_")]
    oracle_cols = [c for c in df.columns if c.startswith("oracle_")]
    other_cols = [
        c
        for c in df.columns
        if not c.startswith("realistic_") and not c.startswith("oracle_")
    ]

    # If columns don't have prefix, try to identify them
    if not realistic_cols and not oracle_cols:
        # Columns from ns3-BI-log format (no prefix)
        all_realistic = [
            "bsr_ac_be",
            "bsr_ac_bk",
            "bsr_ac_vi",
            "bsr_ac_vo",
            "bsr_scaling_factor",
            "rx_fragment_count",
            "fcs_error_count",
            "rcpi",
            "rsni",
            "rssi_dbm",
            "snr_db",
            "link_margin_db",
            "tx_power_dbm",
            "last_rx_frame_type",
            "last_rx_frame_subtype",
            "last_rx_mcs",
            "last_rx_nss",
            "channel_width_mhz",
            "guard_interval_ns",
            "power_mgmt_bit",
            "last_rx_timestamp_us",
            "bytes_received_at_ap",
            "packets_received_at_ap",
            "bytes_received_ac_vo",
            "bytes_received_ac_vi",
            "bytes_received_ac_be",
            "bytes_received_ac_bk",
            "packets_received_ac_vo",
            "packets_received_ac_vi",
            "packets_received_ac_be",
            "packets_received_ac_bk",
            "airtime_used_us",
            "device_class",
            "mean_data_rate_kbps",
            "delay_bound_ms",
            "user_priority",
        ]
        all_oracle = [
            "oracle_total_energy_mj",
            "oracle_awake_time_ms",
            "oracle_sleep_time_ms",
            "oracle_duty_cycle",
            "oracle_packets_generated",
            "oracle_packets_enqueued",
            "oracle_bytes_generated",
            "oracle_mpdu_drops_expired",
            "oracle_mpdu_drops_queue_full",
            "oracle_psdu_timeouts",
            "oracle_packets_transmitted",
            "oracle_bytes_transmitted",
            "oracle_ampdu_count",
            "oracle_ampdu_mpdus_total",
            "oracle_ampdu_bytes_total",
            "oracle_queue_size_packets",
            "oracle_queue_size_bytes",
        ]
        realistic_cols = [c for c in df.columns if c in all_realistic]
        oracle_cols = [c for c in df.columns if c in all_oracle]

    print(f"\nRealistic columns: {len(realistic_cols)}")
    print(f"Oracle columns: {len(oracle_cols)}")

    # Analyze each metric
    results = {
        "GOOD": [],
        "OK": [],
        "LIMITED": [],
        "CONSTANT": [],
        "SENTINEL": [],
        "GARBAGE": [],
    }

    all_results = []

    def analyze_column_set(columns: List[str], prefix: str):
        print(f"\n{'='*70}")
        print(f"{prefix} METRICS")
        print(f"{'='*70}")
        print(f"{'Metric':<40} {'Status':<10} {'Unique':>8} {'Zero%':>8} {'Reason'}")
        print("-" * 90)

        for col in columns:
            if col not in df.columns:
                continue

            values = df[col].values

            # Handle non-numeric columns
            if not np.issubdtype(values.dtype, np.number):
                continue

            # Clean name for display
            display_name = col.replace("realistic_", "").replace("oracle_", "")

            status, reason, stats = classify_metric(col, values, n_samples)
            results[status].append((col, reason, stats))
            all_results.append(
                {
                    "column": col,
                    "display_name": display_name,
                    "category": prefix,
                    "status": status,
                    "reason": reason,
                    **stats,
                }
            )

            # Color code for terminal
            status_colors = {
                "GOOD": "\033[92m",  # Green
                "OK": "\033[94m",  # Blue
                "LIMITED": "\033[93m",  # Yellow
                "CONSTANT": "\033[90m",  # Gray
                "SENTINEL": "\033[91m",  # Red
                "GARBAGE": "\033[91m",  # Red
            }
            reset = "\033[0m"
            color = status_colors.get(status, "")

            print(
                f"{display_name:<40} {color}{status:<10}{reset} {stats['unique']:>8} {stats['zero_pct']:>7.1f}% {reason[:30]}"
            )

    # Analyze realistic metrics
    analyze_column_set(realistic_cols, "REALISTIC")

    # Analyze oracle metrics
    analyze_column_set(oracle_cols, "ORACLE")

    # Summary
    print("\n" + "=" * 70)
    print("METRIC QUALITY SUMMARY")
    print("=" * 70)

    for status in ["GOOD", "OK", "LIMITED", "CONSTANT", "SENTINEL", "GARBAGE"]:
        count = len(results[status])
        if count > 0:
            print(f"\n{status} ({count} metrics):")
            for col, reason, stats in results[status]:
                display_name = col.replace("realistic_", "R:").replace("oracle_", "O:")
                print(f"  {display_name:<45} - {reason}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR PPO OBSERVATION")
    print("=" * 70)

    print("\n✓ RECOMMENDED FEATURES (GOOD quality):")
    good_realistic = [
        r for r in all_results if r["status"] == "GOOD" and r["category"] == "REALISTIC"
    ]
    good_oracle = [
        r for r in all_results if r["status"] == "GOOD" and r["category"] == "ORACLE"
    ]

    for r in good_realistic:
        print(f"  [REALISTIC] {r['display_name']:<35} ({r['unique']} unique values)")
    for r in good_oracle:
        print(f"  [ORACLE]    {r['display_name']:<35} ({r['unique']} unique values)")

    print("\n⚠️  LIMITED FEATURES (use with caution):")
    limited = [r for r in all_results if r["status"] == "LIMITED"]
    for r in limited:
        cat = "R" if r["category"] == "REALISTIC" else "O"
        print(f"  [{cat}] {r['display_name']:<40} - {r['reason']}")

    print("\n✗ GARBAGE FEATURES (do NOT use):")
    garbage = [r for r in all_results if r["status"] in ["GARBAGE", "CONSTANT"]]
    for r in garbage:
        cat = "R" if r["category"] == "REALISTIC" else "O"
        print(f"  [{cat}] {r['display_name']:<40} - {r['reason']}")

    print("\n⚠️  SENTINEL FEATURES (need cleaning before use):")
    sentinel = [r for r in all_results if r["status"] == "SENTINEL"]
    for r in sentinel:
        cat = "R" if r["category"] == "REALISTIC" else "O"
        print(f"  [{cat}] {r['display_name']:<40} - {r['reason']}")

    # Save detailed results to JSON
    results_file = output_dir / "metric_quality_analysis.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved detailed results to: {results_file}")

    # Save recommendations summary to JSON (matching terminal output)
    recommendations = {
        "summary": {
            "total_metrics": len(all_results),
            "good": len([r for r in all_results if r["status"] == "GOOD"]),
            "ok": len([r for r in all_results if r["status"] == "OK"]),
            "limited": len([r for r in all_results if r["status"] == "LIMITED"]),
            "constant": len([r for r in all_results if r["status"] == "CONSTANT"]),
            "sentinel": len([r for r in all_results if r["status"] == "SENTINEL"]),
            "garbage": len([r for r in all_results if r["status"] == "GARBAGE"]),
        },
        "recommended_features": {
            "realistic": [
                {
                    "name": r["display_name"],
                    "column": r["column"],
                    "unique_values": r["unique"],
                    "zero_pct": r["zero_pct"],
                    "reason": r["reason"],
                }
                for r in all_results
                if r["status"] == "GOOD" and r["category"] == "REALISTIC"
            ],
            "oracle": [
                {
                    "name": r["display_name"],
                    "column": r["column"],
                    "unique_values": r["unique"],
                    "zero_pct": r["zero_pct"],
                    "reason": r["reason"],
                }
                for r in all_results
                if r["status"] == "GOOD" and r["category"] == "ORACLE"
            ],
        },
        "limited_features": [
            {
                "name": r["display_name"],
                "column": r["column"],
                "category": r["category"],
                "unique_values": r["unique"],
                "zero_pct": r["zero_pct"],
                "reason": r["reason"],
            }
            for r in all_results
            if r["status"] == "LIMITED"
        ],
        "garbage_features": [
            {
                "name": r["display_name"],
                "column": r["column"],
                "category": r["category"],
                "zero_pct": r["zero_pct"],
                "reason": r["reason"],
            }
            for r in all_results
            if r["status"] in ["GARBAGE", "CONSTANT"]
        ],
        "sentinel_features": [
            {
                "name": r["display_name"],
                "column": r["column"],
                "category": r["category"],
                "sentinel_pct": r["sentinel_pct"],
                "reason": r["reason"],
            }
            for r in all_results
            if r["status"] == "SENTINEL"
        ],
    }

    recommendations_file = output_dir / "metric_recommendations.json"
    with open(recommendations_file, "w") as f:
        json.dump(recommendations, f, indent=2)
    print(f"Saved recommendations to: {recommendations_file}")

    # Generate plot if matplotlib available
    if make_plots and HAS_MATPLOTLIB and all_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))

        # Plot 1: Status distribution
        status_counts = {}
        for r in all_results:
            status = r["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        colors = {
            "GOOD": "#2ecc71",
            "OK": "#3498db",
            "LIMITED": "#f1c40f",
            "CONSTANT": "#95a5a6",
            "SENTINEL": "#e74c3c",
            "GARBAGE": "#c0392b",
        }

        statuses = list(status_counts.keys())
        counts = [status_counts[s] for s in statuses]
        bar_colors = [colors.get(s, "gray") for s in statuses]

        axes[0].bar(statuses, counts, color=bar_colors)
        axes[0].set_xlabel("Status")
        axes[0].set_ylabel("Number of Metrics")
        axes[0].set_title("Metric Quality Distribution")

        # Add count labels
        for i, (s, c) in enumerate(zip(statuses, counts)):
            axes[0].text(
                i, c + 0.5, str(c), ha="center", va="bottom", fontweight="bold"
            )

        # Plot 2: Unique values vs zero percentage (log scale for unique)
        unique_vals = [r["unique"] for r in all_results]
        zero_pcts = [r["zero_pct"] for r in all_results]
        status_list = [r["status"] for r in all_results]

        for status in set(status_list):
            mask = [s == status for s in status_list]
            x = [u for u, m in zip(unique_vals, mask) if m]
            y = [z for z, m in zip(zero_pcts, mask) if m]
            axes[1].scatter(
                x, y, label=status, color=colors.get(status, "gray"), alpha=0.7, s=50
            )

        axes[1].set_xscale("log")
        axes[1].set_xlabel("Unique Values (log scale)")
        axes[1].set_ylabel("Zero Percentage (%)")
        axes[1].set_title("Metric Quality Scatter")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / "metric_quality_analysis.png"
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"Saved plot to: {plot_file}")

    return all_results


def analyze_states(data: Dict[str, np.ndarray], output_dir: Path, make_plots: bool):
    """Analyze state features."""
    print("\n" + "=" * 60)
    print("STATE ANALYSIS")
    print("=" * 60)

    states = data["states"]
    n_samples, state_dim = states.shape

    print(f"Number of samples: {n_samples}")
    print(f"State dimension: {state_dim}")

    # Per-feature statistics
    print("\nPer-feature statistics (first 20 features):")
    print(
        f"{'Idx':<5} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Nonzero%':>10}"
    )
    print("-" * 65)

    for i in range(min(20, state_dim)):
        feat = states[:, i]
        mean = feat.mean()
        std = feat.std()
        min_val = feat.min()
        max_val = feat.max()
        nonzero_pct = (feat != 0).mean() * 100
        print(
            f"{i:<5} {mean:>12.4f} {std:>12.4f} {min_val:>12.4f} {max_val:>12.4f} {nonzero_pct:>9.1f}%"
        )

    # Identify constant features (useless for learning)
    constant_features = []
    for i in range(state_dim):
        if states[:, i].std() == 0:
            constant_features.append(i)

    print(f"\nConstant features (zero variance): {len(constant_features)}")
    if constant_features:
        print(
            f"  Indices: {constant_features[:20]}{'...' if len(constant_features) > 20 else ''}"
        )

    # Highly correlated features
    if state_dim <= 100:
        print("\nComputing feature correlations...")
        # Only compute for non-constant features
        varying_idx = [i for i in range(state_dim) if i not in constant_features]
        if len(varying_idx) > 1:
            varying_states = states[:, varying_idx]
            corr_matrix = np.corrcoef(varying_states.T)

            # Find highly correlated pairs (|r| > 0.95)
            high_corr_pairs = []
            for i in range(len(varying_idx)):
                for j in range(i + 1, len(varying_idx)):
                    if abs(corr_matrix[i, j]) > 0.95:
                        high_corr_pairs.append(
                            (varying_idx[i], varying_idx[j], corr_matrix[i, j])
                        )

            print(
                f"Highly correlated feature pairs (|r| > 0.95): {len(high_corr_pairs)}"
            )
            if high_corr_pairs:
                for i, j, r in high_corr_pairs[:10]:
                    print(f"  Features {i} and {j}: r = {r:.4f}")

    # Plot state distributions
    if make_plots and HAS_MATPLOTLIB:
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < min(16, state_dim):
                feat = states[:, i]
                if feat.std() > 0:
                    ax.hist(feat, bins=50, density=True, alpha=0.7)
                    ax.axvline(feat.mean(), color="red", linestyle="--", label="mean")
                ax.set_title(f"Feature {i}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")

        plt.suptitle("State Feature Distributions (First 16)", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "state_distributions.png", dpi=150)
        plt.close()
        print(f"\nSaved: state_distributions.png")


def analyze_actions(
    data: Dict[str, np.ndarray],
    output_dir: Path,
    make_plots: bool,
    schedule_table: Optional[Dict] = None,
):
    """Analyze action space coverage."""
    print("\n" + "=" * 60)
    print("ACTION ANALYSIS")
    print("=" * 60)

    actions = data["actions"]
    rewards = data["rewards"]
    n_samples = len(actions)

    schedule_actions = actions[:, 0]
    assignment_actions = actions[:, 1]

    # Unique actions
    unique_schedules = np.unique(schedule_actions)
    unique_assignments = np.unique(assignment_actions)

    print(f"Schedule actions used: {len(unique_schedules)} unique")
    print(f"  Range: [{schedule_actions.min()}, {schedule_actions.max()}]")
    print(f"  Values: {unique_schedules.tolist()}")

    print(f"\nAssignment actions used: {len(unique_assignments)} unique")
    print(f"  Range: [{assignment_actions.min()}, {assignment_actions.max()}]")
    print(f"  Values: {unique_assignments.tolist()}")

    # Action frequency
    print("\nSchedule action frequencies:")
    schedule_counts = np.bincount(schedule_actions)
    for idx, count in enumerate(schedule_counts):
        if count > 0:
            pct = count / n_samples * 100
            print(f"  Schedule {idx}: {count:>6} ({pct:>5.2f}%)")

    print("\nAssignment action frequencies:")
    assignment_counts = np.bincount(assignment_actions)
    for idx, count in enumerate(assignment_counts):
        if count > 0:
            pct = count / n_samples * 100
            print(f"  Assignment {idx}: {count:>6} ({pct:>5.2f}%)")

    # Combined action analysis
    print("\n2D Action Space Coverage:")
    n_schedule = len(unique_schedules)
    n_assignment = len(unique_assignments)
    n_possible = max(schedule_actions.max() + 1, 1) * max(
        assignment_actions.max() + 1, 1
    )
    n_observed = len(np.unique(actions, axis=0))

    print(f"  Possible combinations: {n_possible}")
    print(f"  Observed combinations: {n_observed}")
    print(f"  Coverage: {n_observed / n_possible * 100:.1f}%")

    # Reward by action
    print("\nMean reward by schedule action:")
    for sched_idx in unique_schedules:
        mask = schedule_actions == sched_idx
        mean_r = rewards[mask].mean()
        std_r = rewards[mask].std()
        print(f"  Schedule {sched_idx}: mean={mean_r:>8.4f}, std={std_r:>8.4f}")

    print("\nMean reward by assignment action:")
    for assign_idx in unique_assignments:
        mask = assignment_actions == assign_idx
        mean_r = rewards[mask].mean()
        std_r = rewards[mask].std()
        print(f"  Assignment {assign_idx}: mean={mean_r:>8.4f}, std={std_r:>8.4f}")

    # Plot action analysis
    if make_plots and HAS_MATPLOTLIB:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)

        # Schedule action histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.bar(range(len(schedule_counts)), schedule_counts, alpha=0.7)
        ax1.set_xlabel("Schedule Action Index")
        ax1.set_ylabel("Count")
        ax1.set_title("Schedule Action Distribution")

        # Assignment action histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(range(len(assignment_counts)), assignment_counts, alpha=0.7)
        ax2.set_xlabel("Assignment Action Index")
        ax2.set_ylabel("Count")
        ax2.set_title("Assignment Action Distribution")

        # 2D action heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        max_sched = schedule_actions.max() + 1
        max_assign = assignment_actions.max() + 1
        action_heatmap = np.zeros((max_sched, max_assign))
        for s, a in actions:
            action_heatmap[s, a] += 1
        im = ax3.imshow(action_heatmap, aspect="auto", cmap="viridis")
        ax3.set_xlabel("Assignment Action")
        ax3.set_ylabel("Schedule Action")
        ax3.set_title("2D Action Coverage")
        plt.colorbar(im, ax=ax3, label="Count")

        # Reward by schedule
        ax4 = fig.add_subplot(gs[1, 0])
        schedule_rewards = [
            rewards[schedule_actions == s].mean()
            for s in range(len(schedule_counts))
            if schedule_counts[s] > 0
        ]
        schedule_indices = [
            s for s in range(len(schedule_counts)) if schedule_counts[s] > 0
        ]
        ax4.bar(schedule_indices, schedule_rewards, alpha=0.7)
        ax4.set_xlabel("Schedule Action")
        ax4.set_ylabel("Mean Reward")
        ax4.set_title("Mean Reward by Schedule")

        # Reward by assignment
        ax5 = fig.add_subplot(gs[1, 1])
        assign_rewards = [
            rewards[assignment_actions == a].mean()
            for a in range(len(assignment_counts))
            if assignment_counts[a] > 0
        ]
        assign_indices = [
            a for a in range(len(assignment_counts)) if assignment_counts[a] > 0
        ]
        ax5.bar(assign_indices, assign_rewards, alpha=0.7)
        ax5.set_xlabel("Assignment Action")
        ax5.set_ylabel("Mean Reward")
        ax5.set_title("Mean Reward by Assignment")

        # Reward heatmap by 2D action
        ax6 = fig.add_subplot(gs[1, 2])
        reward_heatmap = np.zeros((max_sched, max_assign))
        count_heatmap = np.zeros((max_sched, max_assign))
        for i, (s, a) in enumerate(actions):
            reward_heatmap[s, a] += rewards[i]
            count_heatmap[s, a] += 1
        # Avoid division by zero
        count_heatmap[count_heatmap == 0] = 1
        reward_heatmap /= count_heatmap
        im = ax6.imshow(reward_heatmap, aspect="auto", cmap="RdYlGn")
        ax6.set_xlabel("Assignment Action")
        ax6.set_ylabel("Schedule Action")
        ax6.set_title("Mean Reward by Action Pair")
        plt.colorbar(im, ax=ax6, label="Mean Reward")

        plt.tight_layout()
        plt.savefig(output_dir / "action_analysis.png", dpi=150)
        plt.close()
        print(f"\nSaved: action_analysis.png")


def analyze_rewards(data: Dict[str, np.ndarray], output_dir: Path, make_plots: bool):
    """Analyze reward distribution and patterns."""
    print("\n" + "=" * 60)
    print("REWARD ANALYSIS")
    print("=" * 60)

    rewards = data["rewards"]
    spawn_indices = data["spawn_indices"]
    step_indices = data["step_indices"]

    print(f"Total samples: {len(rewards)}")
    print(f"\nReward statistics:")
    print(f"  Mean:   {rewards.mean():>10.4f}")
    print(f"  Std:    {rewards.std():>10.4f}")
    print(f"  Min:    {rewards.min():>10.4f}")
    print(f"  Max:    {rewards.max():>10.4f}")
    print(f"  Median: {np.median(rewards):>10.4f}")

    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nReward percentiles:")
    for p in percentiles:
        val = np.percentile(rewards, p)
        print(f"  {p:>3}th: {val:>10.4f}")

    # Reward distribution characteristics
    print(f"\nDistribution characteristics:")
    print(
        f"  Skewness:  {((rewards - rewards.mean()) ** 3).mean() / (rewards.std() ** 3):>8.4f}"
    )
    print(
        f"  Kurtosis:  {((rewards - rewards.mean()) ** 4).mean() / (rewards.std() ** 4) - 3:>8.4f}"
    )

    # Episode-level analysis
    unique_spawns = np.unique(spawn_indices)
    episode_returns = []
    episode_lengths = []

    for spawn_id in unique_spawns:
        mask = spawn_indices == spawn_id
        episode_rewards = rewards[mask]
        episode_returns.append(episode_rewards.sum())
        episode_lengths.append(len(episode_rewards))

    episode_returns = np.array(episode_returns)
    episode_lengths = np.array(episode_lengths)

    print(f"\nEpisode-level statistics ({len(unique_spawns)} episodes):")
    print(
        f"  Episode length: mean={episode_lengths.mean():.1f}, std={episode_lengths.std():.1f}"
    )
    print(
        f"  Episode return: mean={episode_returns.mean():.2f}, std={episode_returns.std():.2f}"
    )
    print(
        f"  Return range:   [{episode_returns.min():.2f}, {episode_returns.max():.2f}]"
    )

    # Temporal patterns
    print(f"\nTemporal patterns (first 10 steps):")
    max_steps = min(10, step_indices.max() + 1)
    for step in range(max_steps):
        mask = step_indices == step
        if mask.sum() > 0:
            step_rewards = rewards[mask]
            print(
                f"  Step {step}: mean={step_rewards.mean():>8.4f}, "
                f"std={step_rewards.std():>8.4f}, n={mask.sum()}"
            )

    # Plot rewards
    if make_plots and HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Reward histogram
        axes[0, 0].hist(rewards, bins=50, density=True, alpha=0.7)
        axes[0, 0].axvline(rewards.mean(), color="red", linestyle="--", label="mean")
        axes[0, 0].set_xlabel("Reward")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].set_title("Reward Distribution")
        axes[0, 0].legend()

        # Reward over time (across all episodes)
        axes[0, 1].scatter(step_indices, rewards, alpha=0.1, s=1)
        # Mean by step
        step_means = []
        steps = list(range(step_indices.max() + 1))
        for step in steps:
            mask = step_indices == step
            if mask.sum() > 0:
                step_means.append(rewards[mask].mean())
            else:
                step_means.append(np.nan)
        axes[0, 1].plot(steps, step_means, "r-", linewidth=2, label="mean")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].set_title("Reward vs Step")
        axes[0, 1].legend()

        # Episode return histogram
        axes[0, 2].hist(episode_returns, bins=30, density=True, alpha=0.7)
        axes[0, 2].axvline(
            episode_returns.mean(), color="red", linestyle="--", label="mean"
        )
        axes[0, 2].set_xlabel("Episode Return")
        axes[0, 2].set_ylabel("Density")
        axes[0, 2].set_title("Episode Return Distribution")
        axes[0, 2].legend()

        # Cumulative distribution
        sorted_rewards = np.sort(rewards)
        cdf = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
        axes[1, 0].plot(sorted_rewards, cdf)
        axes[1, 0].set_xlabel("Reward")
        axes[1, 0].set_ylabel("CDF")
        axes[1, 0].set_title("Reward CDF")
        axes[1, 0].grid(True, alpha=0.3)

        # Box plot by episode
        n_episodes_to_show = min(20, len(unique_spawns))
        episode_data = []
        for spawn_id in unique_spawns[:n_episodes_to_show]:
            mask = spawn_indices == spawn_id
            episode_data.append(rewards[mask])
        axes[1, 1].boxplot(episode_data)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Reward")
        axes[1, 1].set_title(
            f"Reward Distribution by Episode (first {n_episodes_to_show})"
        )

        # Reward autocorrelation
        if len(rewards) > 100:
            max_lag = min(50, len(rewards) // 10)
            autocorr = [
                np.corrcoef(rewards[:-lag], rewards[lag:])[0, 1]
                for lag in range(1, max_lag + 1)
            ]
            axes[1, 2].bar(range(1, max_lag + 1), autocorr, alpha=0.7)
            axes[1, 2].axhline(0, color="black", linestyle="-", linewidth=0.5)
            axes[1, 2].axhline(
                2 / np.sqrt(len(rewards)), color="red", linestyle="--", alpha=0.5
            )
            axes[1, 2].axhline(
                -2 / np.sqrt(len(rewards)), color="red", linestyle="--", alpha=0.5
            )
            axes[1, 2].set_xlabel("Lag")
            axes[1, 2].set_ylabel("Autocorrelation")
            axes[1, 2].set_title("Reward Autocorrelation")

        plt.tight_layout()
        plt.savefig(output_dir / "reward_analysis.png", dpi=150)
        plt.close()
        print(f"\nSaved: reward_analysis.png")


def analyze_transitions(
    data: Dict[str, np.ndarray], output_dir: Path, make_plots: bool
):
    """Analyze state transitions."""
    print("\n" + "=" * 60)
    print("TRANSITION ANALYSIS")
    print("=" * 60)

    states = data["states"]
    next_states = data["next_states"]
    actions = data["actions"]

    # State change magnitude
    state_diff = next_states - states
    diff_magnitude = np.linalg.norm(state_diff, axis=1)

    print(f"State change magnitude:")
    print(f"  Mean:   {diff_magnitude.mean():>10.4f}")
    print(f"  Std:    {diff_magnitude.std():>10.4f}")
    print(f"  Min:    {diff_magnitude.min():>10.4f}")
    print(f"  Max:    {diff_magnitude.max():>10.4f}")

    # Per-feature changes
    print(f"\nPer-feature change statistics (first 10):")
    print(f"{'Idx':<5} {'Mean Δ':>12} {'Std Δ':>12} {'Max |Δ|':>12}")
    print("-" * 45)

    for i in range(min(10, states.shape[1])):
        feat_diff = state_diff[:, i]
        mean_diff = feat_diff.mean()
        std_diff = feat_diff.std()
        max_abs = np.abs(feat_diff).max()
        print(f"{i:<5} {mean_diff:>12.4f} {std_diff:>12.4f} {max_abs:>12.4f}")

    # Plot transition analysis
    if make_plots and HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # State change magnitude distribution
        axes[0].hist(diff_magnitude, bins=50, density=True, alpha=0.7)
        axes[0].set_xlabel("State Change Magnitude")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Distribution of State Changes")

        # State change by action
        schedule_actions = actions[:, 0]
        unique_schedules = np.unique(schedule_actions)
        schedule_diffs = [
            diff_magnitude[schedule_actions == s].mean() for s in unique_schedules
        ]
        axes[1].bar(unique_schedules, schedule_diffs, alpha=0.7)
        axes[1].set_xlabel("Schedule Action")
        axes[1].set_ylabel("Mean State Change")
        axes[1].set_title("State Change by Schedule Action")

        # Per-feature change heatmap (first 20 features)
        n_feat = min(20, states.shape[1])
        feat_changes = np.abs(state_diff[:, :n_feat]).mean(axis=0)
        axes[2].barh(range(n_feat), feat_changes)
        axes[2].set_xlabel("Mean |Change|")
        axes[2].set_ylabel("Feature Index")
        axes[2].set_title("Feature Sensitivity")
        axes[2].invert_yaxis()

        plt.tight_layout()
        plt.savefig(output_dir / "transition_analysis.png", dpi=150)
        plt.close()
        print(f"\nSaved: transition_analysis.png")


def main():
    args = parse_args()

    # Auto-detect NPZ file if not provided
    if args.npz_file:
        npz_path = Path(args.npz_file)
    else:
        # Find the most recent stacked_transitions.npz file
        npz_files = sorted(
            glob.glob("eda-data/**/stacked_transitions.npz", recursive=True)
        )
        if not npz_files and not args.raw_metrics_only:
            print("Warning: No stacked_transitions.npz files found.")
            print("Will only run raw metrics analysis if --data-log-dir is available.")
            npz_path = None
        elif npz_files:
            npz_path = Path(npz_files[-1])  # Most recent (last alphabetically)
            print(f"Auto-detected NPZ file: {npz_path}")
        else:
            npz_path = None

    # Auto-detect data-log directory
    if args.data_log_dir:
        data_log_dir = Path(args.data_log_dir)
    else:
        # Look for data-log in parent directory
        script_dir = Path(__file__).parent
        data_log_dir = script_dir.parent / "data-log"
        if not data_log_dir.exists():
            data_log_dir = Path("data-log")

    if npz_path and not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        npz_path = None

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif npz_path:
        output_dir = npz_path.parent
    else:
        output_dir = Path("eda-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EDA Analysis for TWT RL Dataset")
    print("=" * 60)
    if npz_path:
        print(f"NPZ Input:    {npz_path}")
    if data_log_dir.exists():
        print(f"Data Log Dir: {data_log_dir}")
    print(f"Output:       {output_dir}")
    print("=" * 60)

    make_plots = HAS_MATPLOTLIB and not args.no_plots

    # Always run raw metrics analysis if data-log exists
    if data_log_dir.exists():
        analyze_raw_metrics(data_log_dir, output_dir, make_plots)
    else:
        print(f"\nWarning: data-log directory not found at {data_log_dir}")
        print("Skipping raw metrics analysis.")

    # Skip NPZ analysis if raw_metrics_only or no NPZ file
    if args.raw_metrics_only or npz_path is None:
        print("\n" + "=" * 60)
        print("Raw Metrics Analysis Complete!")
        print("=" * 60)
        if make_plots:
            print(f"Plots saved to: {output_dir}")
        print("")
        return

    # Load NPZ data
    print("\nLoading NPZ data...")
    data = load_data(str(npz_path))
    print(f"Loaded arrays: {list(data.keys())}")

    # Load schedule table if provided
    schedule_table = None
    if args.schedule_table and Path(args.schedule_table).exists():
        with open(args.schedule_table) as f:
            schedule_table = json.load(f)

    # Run analyses
    analyze_states(data, output_dir, make_plots)
    analyze_actions(data, output_dir, make_plots, schedule_table)
    analyze_rewards(data, output_dir, make_plots)
    analyze_transitions(data, output_dir, make_plots)

    print("\n" + "=" * 60)
    print("EDA Complete!")
    print("=" * 60)
    if make_plots:
        print(f"Plots saved to: {output_dir}")
    print("")


if __name__ == "__main__":
    main()

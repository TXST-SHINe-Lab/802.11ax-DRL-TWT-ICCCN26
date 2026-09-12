#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""plot_evaluation.py - evaluation results visualisation for TWT WiFi scheduling.

Generates publication-ready plots from the evaluation JSON files written by eval_policy.py,
comparing PPO against whichever baselines that run included, across all presets.

Outputs:
- eval_comparison_rewards.png: Reward comparison bar chart
- eval_comparison_metrics.png: Key metrics comparison
- eval_boxplots.png: Distribution of rewards across episodes
- eval_radar.png: Radar chart for multi-metric comparison
- eval_summary_table.csv: Tabular summary for paper

Usage:
    python3.11 plot_evaluation.py
    python3.11 plot_evaluation.py --training-script train_ppo_V1.py
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import run_paths

# --- Checkpoint pattern configuration ---
# Must match the training script name: train_<pattern>.py becomes <pattern>_twt_.
# So train_lstm_ppo_V1.py becomes lstm_ppo_V1_twt_, and train_ppo.py becomes ppo_twt_.
# Edit this value or pass --training-script to select which eval files are plotted.
DEFAULT_CHECKPOINT_PATTERN = "lstm_ppo_V1_twt_"

# Style settings for publication
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    }
)

# Color palette
COLORS = {
    "PPO": "#2ecc71",  # Green
    "Random": "#e74c3c",  # Red
    "Heuristic": "#3498db",  # Blue
    "Analytical": "#9b59b6",  # Purple
    "Heuristic_throughput": "#3498db",
    "Heuristic_energy": "#9b59b6",
    "Heuristic_queue": "#f39c12",
}

PRESET_LABELS = {
    "throughput": "Throughput-Focused",
    "energy": "Energy-Focused",
    "queue": "Queue/Latency-Focused",
}


def discover_presets(eval_dir: str, checkpoint_pattern: str = "") -> dict:
    """Discover available presets from evaluation files.

    Args:
        eval_dir: Directory containing eval_*.json files
        checkpoint_pattern: Optional pattern to filter files (e.g., 'lstm_ppo_V1_twt_')
                           If provided, only files matching eval_<pattern><preset>_*.json are included
    """
    presets = {}

    # Find all eval_*.json files (with optional pattern filter)
    if checkpoint_pattern:
        glob_pattern = os.path.join(eval_dir, f"eval_{checkpoint_pattern}*.json")
    else:
        glob_pattern = os.path.join(eval_dir, "eval_*.json")

    all_files = glob.glob(glob_pattern)

    for filepath in all_files:
        filename = os.path.basename(filepath)
        # Format: eval_<pattern><preset>_<timestamp>.json or eval_<preset>_<timestamp>.json
        # Remove .json and split
        base = filename.replace(".json", "")

        # Remove 'eval_' prefix
        if base.startswith("eval_"):
            base = base[5:]  # Remove 'eval_'

        # If pattern provided, remove it to get the preset
        if checkpoint_pattern and base.startswith(checkpoint_pattern):
            base = base[len(checkpoint_pattern) :]

        # Now extract preset (everything before the timestamp)
        # Timestamp format: YYYYMMDD_HHMMSS
        parts = base.split("_")
        if len(parts) >= 3:
            # Assume last two parts are timestamp (date, time)
            preset = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
        elif len(parts) >= 1:
            preset = parts[0]
        else:
            continue

        if preset not in presets:
            presets[preset] = []
        presets[preset].append(filepath)

    # Sort each preset's files to get the latest
    for preset in presets:
        presets[preset] = sorted(presets[preset], reverse=True)

    return presets


def load_eval_results(
    eval_dir: str = "eval_results", checkpoint_pattern: str = ""
) -> dict:
    """Load all evaluation JSON files with graceful error handling.

    Args:
        eval_dir: Directory containing eval_*.json files
        checkpoint_pattern: Optional pattern to filter files (e.g., 'lstm_ppo_V1_twt_')
    """
    results = {}
    skipped = {}

    if not os.path.exists(eval_dir):
        print(f"⚠ Evaluation directory not found: {eval_dir}")
        return results

    # Discover available presets dynamically (with optional pattern filter)
    available_presets = discover_presets(eval_dir, checkpoint_pattern)

    if not available_presets:
        pattern_msg = (
            f" matching pattern '{checkpoint_pattern}'" if checkpoint_pattern else ""
        )
        print(f"⚠ No evaluation files found in {eval_dir}{pattern_msg}")
        return results

    pattern_msg = f" (pattern: {checkpoint_pattern})" if checkpoint_pattern else ""
    print(
        f"Found {len(available_presets)} preset(s){pattern_msg}: {list(available_presets.keys())}\n"
    )

    for preset, files in available_presets.items():
        if not files:
            skipped[preset] = "No evaluation files found"
            continue

        latest = files[0]  # Already sorted, most recent first
        try:
            with open(latest, "r") as f:
                data = json.load(f)
            results[preset] = data
            print(f"✓ Loaded {preset}: {os.path.basename(latest)}")
        except json.JSONDecodeError as e:
            skipped[preset] = f"JSON parse error: {str(e)}"
            print(f"✗ Failed to parse {preset}: {str(e)}")
        except Exception as e:
            skipped[preset] = f"Error reading file: {str(e)}"
            print(f"✗ Failed to load {preset}: {str(e)}")

    # Report skipped presets
    if skipped:
        print(f"\n⚠ Skipped {len(skipped)} preset(s):")
        for preset, reason in skipped.items():
            print(f"  - {preset}: {reason}")

    return results


def extract_policy_stats(results: dict) -> dict:
    """Extract statistics for each policy across presets."""
    stats = {}

    for preset, data in results.items():
        stats[preset] = {}

        # Dynamically discover all policies in the data
        for policy_name, policy_data in data.items():
            # Skip metadata keys
            if policy_name.startswith("_") or not isinstance(policy_data, dict):
                continue

            # Check if this looks like a valid policy entry
            if "mean_reward" not in policy_data:
                continue

            # Normalize heuristic names to just "Heuristic" for plotting
            display_name = policy_name
            if policy_name.startswith("Heuristic"):
                display_name = "Heuristic"
            elif policy_name.startswith("Analytical"):
                display_name = "Analytical"

            # Only keep the first heuristic/analytical found (avoid duplicates)
            if display_name in stats[preset]:
                continue

            stats[preset][display_name] = {
                "mean_reward": policy_data.get("mean_reward", 0),
                "std_reward": policy_data.get("std_reward", 0),
                "episode_rewards": policy_data.get("episode_rewards", []),
                "metrics": policy_data.get("metrics", {}),
                "original_name": policy_name,  # Keep original for reference
            }

    return stats


def get_available_policies(stats: dict) -> list:
    """Get list of all unique policies across all presets."""
    policies = set()
    for preset_stats in stats.values():
        policies.update(preset_stats.keys())

    # Define preferred order
    preferred_order = ["PPO", "Heuristic", "Analytical", "Random"]
    ordered = [p for p in preferred_order if p in policies]
    # Add any others not in preferred order
    ordered.extend([p for p in policies if p not in preferred_order])

    return ordered


def plot_reward_comparison(stats: dict, output_path: str):
    """Bar chart comparing mean rewards across policies and presets."""
    if not stats:
        print("⚠ No data to plot for reward comparison")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    presets = list(stats.keys())
    policies = get_available_policies(stats)

    if not policies:
        print("⚠ No policies found in data")
        plt.close()
        return

    x = np.arange(len(presets))
    width = 0.8 / max(len(policies), 1)  # Dynamic width based on policy count

    for i, policy in enumerate(policies):
        means = []
        stds = []
        for preset in presets:
            if policy in stats[preset]:
                means.append(stats[preset][policy]["mean_reward"])
                stds.append(stats[preset][policy]["std_reward"])
            else:
                means.append(0)
                stds.append(0)

        color = COLORS.get(policy, "#95a5a6")
        bars = ax.bar(
            x + i * width,
            means,
            width,
            label=policy,
            color=color,
            yerr=stds,
            capsize=4,
            alpha=0.85,
        )

        # Add value labels manually for bars with error bars
        for j, (xpos, mean) in enumerate(zip(x + i * width, means)):
            ax.text(
                xpos,
                mean + stds[j] + 0.3,
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Reward Preset")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Policy Performance Comparison Across Presets")
    ax.set_xticks(x + width)
    ax.set_xticklabels([PRESET_LABELS.get(p, p) for p in presets])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_boxplots(stats: dict, output_path: str):
    """Box plots showing reward distributions."""
    if not stats:
        print("⚠ No data to plot for boxplots")
        return

    num_presets = len(stats)
    policies = get_available_policies(stats)

    # Adaptive layout based on number of presets
    if num_presets == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        axes = [ax]
    elif num_presets == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    else:
        fig, axes = plt.subplots(1, min(num_presets, 3), figsize=(14, 5), sharey=True)
        if num_presets > 3:
            print(f"⚠ Only showing first 3 of {num_presets} presets in boxplots")

    for idx, preset in enumerate(list(stats.keys())[: len(axes)]):
        ax = axes[idx] if num_presets > 1 else axes[0]

        data = []
        labels = []
        colors = []

        for policy in policies:
            if policy in stats[preset]:
                rewards = stats[preset][policy].get("episode_rewards", [])
                if rewards:
                    data.append(rewards)
                    labels.append(policy)
                    colors.append(COLORS.get(policy, "#95a5a6"))

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(PRESET_LABELS.get(preset, preset))
        ax.set_ylabel("Episode Reward" if idx == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Reward Distribution Across Episodes (50 episodes each)", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_comparison(stats: dict, output_path: str):
    """Compare key metrics across policies."""
    if not stats:
        print("⚠ No data to plot for metrics comparison")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics_info = [
        ("total_bytes_tx", "Total Bytes Transmitted", "Bytes", 1e6, "MB"),
        ("total_energy_mj", "Total Energy Consumed", "mJ", 1, "mJ"),
        ("avg_queue_bytes", "Average Queue Size", "Bytes", 1, "Bytes"),
        ("total_drops", "Total Packet Drops", "Packets", 1, "Packets"),
    ]

    policies = get_available_policies(stats)

    for ax_idx, (metric_key, title, ylabel, scale, unit) in enumerate(metrics_info):
        ax = axes[ax_idx // 2, ax_idx % 2]

        presets = list(stats.keys())

        x = np.arange(len(presets))
        width = 0.8 / max(len(policies), 1)

        for i, policy in enumerate(policies):
            means = []
            stds = []

            for preset in presets:
                if policy in stats[preset]:
                    m = stats[preset][policy]["metrics"].get(metric_key, {})
                    if isinstance(m, dict):
                        means.append(m.get("mean", 0) / scale)
                        stds.append(m.get("std", 0) / scale)
                    else:
                        means.append(0)
                        stds.append(0)
                else:
                    means.append(0)
                    stds.append(0)

            color = COLORS.get(policy, "#95a5a6")
            ax.bar(
                x + i * width,
                means,
                width,
                label=policy if ax_idx == 0 else "",
                color=color,
                yerr=stds,
                capsize=3,
                alpha=0.85,
            )

        ax.set_title(title)
        ax.set_ylabel(f"{ylabel} ({unit})")
        ax.set_xticks(x + width)
        ax.set_xticklabels([p.capitalize() for p in presets])
        ax.grid(axis="y", alpha=0.3)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Key Metrics Comparison Across Policies", y=1.05, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_radar_chart(stats: dict, output_path: str):
    """Radar chart comparing normalized metrics for each preset."""
    if not stats:
        print("⚠ No data to plot for radar chart")
        return

    num_presets = len(stats)
    policies = get_available_policies(stats)

    # Adaptive layout based on number of presets
    if num_presets == 1:
        fig, ax = plt.subplots(
            1, 1, figsize=(6, 5), subplot_kw=dict(projection="polar")
        )
        axes = [ax]
    elif num_presets == 2:
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 5), subplot_kw=dict(projection="polar")
        )
    else:
        fig, axes = plt.subplots(
            1, min(num_presets, 3), figsize=(15, 5), subplot_kw=dict(projection="polar")
        )
        if num_presets > 3:
            print(f"⚠ Only showing first 3 of {num_presets} presets in radar chart")

    metrics = ["Reward", "Throughput", "Energy Eff.", "Queue Eff.", "Drop Rate"]
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for idx, preset in enumerate(list(stats.keys())[: len(axes)]):
        ax = axes[idx] if num_presets > 1 else axes[0]

        for policy in policies:
            if policy not in stats[preset]:
                continue

            p = stats[preset][policy]
            m = p["metrics"]

            # Normalize values (higher is better, except energy and drops)
            values = [
                p["mean_reward"] / 10,  # Reward normalized to ~1
                m.get("total_bytes_tx", {}).get("mean", 0) / 1e8,  # Throughput
                1
                - m.get("total_energy_mj", {}).get("mean", 0)
                / 40000,  # Energy efficiency (inverted)
                1
                - m.get("avg_queue_bytes", {}).get("mean", 0)
                / 30000,  # Queue efficiency (inverted)
                1
                - m.get("total_drops", {}).get("mean", 0)
                / 5000,  # Drop rate (inverted)
            ]
            values = [max(0, min(1, v)) for v in values]  # Clip to [0, 1]
            values += values[:1]  # Close the polygon

            color = COLORS.get(policy, "#95a5a6")
            ax.plot(angles, values, "o-", linewidth=2, label=policy, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_title(PRESET_LABELS.get(preset, preset), pad=15)
        ax.set_ylim(0, 1)

        if idx == 0:
            ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    fig.suptitle("Multi-Metric Policy Comparison (Normalized)", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_improvement_chart(stats: dict, output_path: str):
    """Bar chart showing PPO improvement over baselines."""
    if not stats:
        print("⚠ No data to plot for improvement chart")
        return

    # Check if PPO exists in any preset
    has_ppo = any("PPO" in preset_stats for preset_stats in stats.values())
    if not has_ppo:
        print("⚠ No PPO data found - skipping improvement chart")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    presets = list(stats.keys())
    baselines = [p for p in get_available_policies(stats) if p != "PPO"]

    if not baselines:
        print("⚠ No baseline policies found for comparison")
        plt.close()
        return

    # Calculate improvements for each baseline
    improvements = {baseline: [] for baseline in baselines}

    for preset in presets:
        ppo_reward = stats[preset].get("PPO", {}).get("mean_reward", 0)

        for baseline in baselines:
            baseline_reward = stats[preset].get(baseline, {}).get("mean_reward", 0)

            if baseline_reward != 0 and ppo_reward != 0:
                imp = ((ppo_reward - baseline_reward) / abs(baseline_reward)) * 100
            else:
                imp = 0
            improvements[baseline].append(imp)

    x = np.arange(len(presets))
    width = 0.8 / max(len(baselines), 1)

    bars_list = []
    for i, baseline in enumerate(baselines):
        color = COLORS.get(baseline, "#95a5a6")
        bars = ax.bar(
            x + i * width - (len(baselines) - 1) * width / 2,
            improvements[baseline],
            width,
            label=f"vs {baseline}",
            color=color,
            alpha=0.8,
        )
        bars_list.append(bars)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Reward Preset")
    ax.set_ylabel("PPO Improvement (%)")
    ax.set_title("PPO Performance Improvement Over Baselines")
    ax.set_xticks(x)
    ax.set_xticklabels([PRESET_LABELS.get(p, p) for p in presets])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in bars_list:
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(stats: dict, output_path: str):
    """Generate CSV summary table for paper."""
    if not stats:
        print("⚠ No data to generate summary table")
        return

    policies = get_available_policies(stats)
    rows = []

    header = [
        "Preset",
        "Policy",
        "Mean Reward",
        "Std Reward",
        "Throughput (MB)",
        "Energy (mJ)",
        "Queue (KB)",
        "Drops",
    ]
    rows.append(header)

    for preset in stats:
        for policy in policies:
            if policy not in stats[preset]:
                continue

            p = stats[preset][policy]
            m = p["metrics"]

            row = [
                preset.capitalize(),
                policy,
                f"{p['mean_reward']:.2f}",
                f"{p['std_reward']:.2f}",
                f"{m.get('total_bytes_tx', {}).get('mean', 0) / 1e6:.1f}",
                f"{m.get('total_energy_mj', {}).get('mean', 0):.0f}",
                f"{m.get('avg_queue_bytes', {}).get('mean', 0) / 1000:.1f}",
                f"{m.get('total_drops', {}).get('mean', 0):.0f}",
            ]
            rows.append(row)

        # Add empty row between presets
        rows.append([""] * len(header))

    # Write CSV
    with open(output_path, "w") as f:
        for row in rows:
            f.write(",".join(row) + "\n")

    print(f"Saved: {output_path}")

    # Also print as formatted table
    print("\n" + "=" * 110)
    print("EVALUATION SUMMARY TABLE")
    print("=" * 110)
    print(
        f"{'Preset':<12} {'Policy':<14} {'Reward':>10} {'±Std':>8} {'Thru(MB)':>10} {'Energy(mJ)':>12} {'Queue(KB)':>10} {'Drops':>8}"
    )
    print("-" * 110)

    for preset in stats:
        for policy in policies:
            if policy not in stats[preset]:
                continue

            p = stats[preset][policy]
            m = p["metrics"]

            print(
                f"{preset:<12} {policy:<14} {p['mean_reward']:>10.2f} {p['std_reward']:>8.2f} "
                f"{m.get('total_bytes_tx', {}).get('mean', 0) / 1e6:>10.1f} "
                f"{m.get('total_energy_mj', {}).get('mean', 0):>12.0f} "
                f"{m.get('avg_queue_bytes', {}).get('mean', 0) / 1000:>10.1f} "
                f"{m.get('total_drops', {}).get('mean', 0):>8.0f}"
            )
        print()


def derive_pattern_from_training_script(script_name: str) -> str:
    """Derive checkpoint pattern from training script name.

    Example: train_lstm_ppo_V1.py -> lstm_ppo_V1_twt_
    """
    base = os.path.basename(script_name).replace(".py", "")
    if base.startswith("train_"):
        base = base[6:]  # Remove 'train_'
    return f"{base}_twt_"


def main():
    """Generate all evaluation plots."""
    parser = argparse.ArgumentParser(
        description="Plot evaluation results for TWT scheduling policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default pattern (lstm_ppo_V1_twt_)
  python plot_evaluation.py
  
  # Specify pattern via training script name
  python plot_evaluation.py --training-script train_lstm_ppo_V1.py
  
  # Use explicit pattern
  python plot_evaluation.py --pattern ppo_twt_
  
  # Plot all eval files (no filtering)
  python plot_evaluation.py --pattern ""
""",
    )
    parser.add_argument(
        "--training-script",
        type=str,
        default=None,
        help="Training script name to derive pattern (e.g., train_lstm_ppo_V1.py -> lstm_ppo_V1_twt_)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help=f"Explicit checkpoint pattern (default: {DEFAULT_CHECKPOINT_PATTERN})",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Directory containing evaluation results (default: eval_results/)",
    )
    args = parser.parse_args()

    # Determine checkpoint pattern
    if args.pattern is not None:
        checkpoint_pattern = args.pattern
    elif args.training_script:
        checkpoint_pattern = derive_pattern_from_training_script(args.training_script)
    else:
        checkpoint_pattern = DEFAULT_CHECKPOINT_PATTERN

    eval_dir = run_paths.resolve(args.eval_dir, "eval_results")

    print("=" * 60)
    print("EVALUATION RESULTS VISUALIZATION")
    print("=" * 60)
    if checkpoint_pattern:
        print(f"Filtering by pattern: {checkpoint_pattern}")
    else:
        print("No pattern filter (showing all eval files)")
    print()

    # Load data
    results = load_eval_results(eval_dir, checkpoint_pattern)

    if not results:
        print("\n✗ No evaluation results found!")
        print("\nTroubleshooting:")
        print("  1. Run evaluation first: ./run_eval.sh")
        print("  2. Check that eval_results/ contains eval_*.json files")
        print("  3. Verify JSON files are valid")
        return

    # Extract statistics
    stats = extract_policy_stats(results)

    if not stats:
        print("\n✗ Could not extract policy statistics from results")
        return

    # Report what we found
    policies = get_available_policies(stats)
    print(
        f"\n✓ Found {len(stats)} preset(s) with {len(policies)} policy type(s): {policies}"
    )

    # Create plots directory
    plots_dir = run_paths.artifact_dir("plots", create=True)

    # Build output file prefix (include pattern in filenames)
    file_prefix = f"eval_{checkpoint_pattern}" if checkpoint_pattern else "eval_"

    print("\nGenerating plots...")

    # Generate plots (each function handles empty data gracefully)
    plot_reward_comparison(
        stats, os.path.join(plots_dir, f"{file_prefix}comparison_rewards.png")
    )
    plot_boxplots(stats, os.path.join(plots_dir, f"{file_prefix}boxplots.png"))
    plot_metrics_comparison(
        stats, os.path.join(plots_dir, f"{file_prefix}comparison_metrics.png")
    )
    plot_radar_chart(stats, os.path.join(plots_dir, f"{file_prefix}radar.png"))
    plot_improvement_chart(
        stats, os.path.join(plots_dir, f"{file_prefix}improvement.png")
    )

    # Generate summary table
    generate_summary_table(
        stats, os.path.join(plots_dir, f"{file_prefix}summary_table.csv")
    )

    print("\n" + "=" * 60)
    print(f"GENERATED FILES (in plots/, prefix: {file_prefix}):")
    print("=" * 60)
    print(f"  - plots/{file_prefix}comparison_rewards.png  : Reward bar chart")
    print(f"  - plots/{file_prefix}boxplots.png            : Reward distributions")
    print(f"  - plots/{file_prefix}comparison_metrics.png  : Key metrics comparison")
    print(f"  - plots/{file_prefix}radar.png               : Radar chart (normalized)")
    print(f"  - plots/{file_prefix}improvement.png         : PPO improvement %")
    print(f"  - plots/{file_prefix}summary_table.csv       : Tabular summary")


if __name__ == "__main__":
    main()

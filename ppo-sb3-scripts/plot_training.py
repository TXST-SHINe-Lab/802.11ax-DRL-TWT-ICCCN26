#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""plot_training.py - generate training visualisations for TWT PPO.

Plots training curves for every preset from the JSONL logs written by EpisodeLoggerCallback.
Which checkpoints are read depends on --training-script, since the pattern is derived from its name.

Usage:
    python3.11 plot_training.py
    python3.11 plot_training.py --preset throughput
    python3.11 plot_training.py --training-script train_lstm_ppo_V1.py
    python3.11 plot_training.py --output training_plots.png
"""

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

import os
import sys
import json
import glob
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import run_paths

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "throughput": "#2ecc71",  # Green
    "energy": "#3498db",  # Blue
    "queue": "#e74c3c",  # Red
}

PRESET_NAMES = {
    "throughput": "Throughput",
    "energy": "Energy",
    "queue": "Queue/Latency",
}


def load_training_log(log_path: str) -> dict:
    """Load training log from JSONL file."""
    episodes = []
    metadata = {}

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "_metadata" in data:
                metadata = data["_metadata"]
            else:
                episodes.append(data)

    return {"metadata": metadata, "episodes": episodes}


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


def smooth(values, window=10):
    """Apply rolling mean smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_single_preset(ax, episodes, preset, show_raw=True, smooth_window=10):
    """Plot training curve for a single preset."""
    if not episodes:
        return

    timesteps = [e["timestep"] for e in episodes]
    rewards = [e["reward"] for e in episodes]

    color = COLORS.get(preset, "#333333")

    # Plot raw data (faded)
    if show_raw:
        ax.plot(timesteps, rewards, alpha=0.2, color=color, linewidth=0.5)

    # Plot smoothed data
    if len(rewards) > smooth_window:
        smoothed = smooth(rewards, smooth_window)
        smooth_timesteps = timesteps[smooth_window - 1 :]
        ax.plot(
            smooth_timesteps,
            smoothed,
            color=color,
            linewidth=2,
            label=f"{PRESET_NAMES[preset]} (smoothed)",
        )
    else:
        ax.plot(
            timesteps, rewards, color=color, linewidth=2, label=PRESET_NAMES[preset]
        )


def plot_all_presets(
    checkpoints: dict,
    output_path: str = None,
    smooth_window: int = 10,
    file_prefix: str = "",
):
    """Create comprehensive training visualization.

    Args:
        checkpoints: Dict mapping preset names to checkpoint directories
        output_path: Path for main summary plot
        smooth_window: Window size for smoothing
        file_prefix: Prefix for output filenames (e.g., 'lstm_ppo_V1_twt_')
    """

    # Load all data
    all_data = {}
    skipped_presets = {}

    for preset, ckpt_dir in checkpoints.items():
        log_files = glob.glob(os.path.join(ckpt_dir, "training_log_*.jsonl"))
        if log_files:
            try:
                data = load_training_log(log_files[0])
                all_data[preset] = data
                print(
                    f"✓ Loaded {preset}: {len(data['episodes'])} episodes from {os.path.basename(log_files[0])}"
                )
            except Exception as e:
                skipped_presets[preset] = f"Error reading log: {str(e)}"
                print(f"✗ Failed to load {preset}: {str(e)}")
        else:
            skipped_presets[preset] = "No training log file found"
            print(f"⚠ Skipping {preset}: No training_log_*.jsonl file in {ckpt_dir}")

    # Report summary
    if skipped_presets:
        print(f"\nSkipped {len(skipped_presets)} preset(s):")
        for preset, reason in skipped_presets.items():
            print(f"  - {preset}: {reason}")

    if not all_data:
        print("\n✗ Error: No training logs found in any preset!")
        print("\nTroubleshooting:")
        print(
            "  1. Check that checkpoint directories contain training_log_*.jsonl files"
        )
        print("  2. Run train_ppo.py to generate training logs")
        print("  3. Verify checkpoint directory structure")
        return

    # Use only available presets
    num_presets = len(all_data)
    presets = list(all_data.keys())

    # Create figure with adaptive layout
    if num_presets == 1:
        # Single preset: simpler layout (2 rows, 3 cols)
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    else:
        # Multiple presets: full layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # --- Plot 1: All Presets Overlaid (top left, spanning 2 cols) ---
    if num_presets > 1:
        ax1 = fig.add_subplot(gs[0, :2])
    else:
        ax1 = fig.add_subplot(gs[0, :])

    for preset, data in all_data.items():
        plot_single_preset(
            ax1, data["episodes"], preset, show_raw=True, smooth_window=smooth_window
        )

    ax1.set_xlabel("Timesteps", fontsize=11)
    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.set_title(
        "Training Curves: " + ", ".join([PRESET_NAMES.get(p, p) for p in presets]),
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="lower right", fontsize=10)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # --- Plot 2: Final Reward Comparison (top right, only if multiple presets) ---
    if num_presets > 1:
        ax2 = fig.add_subplot(gs[0, 2])

        final_rewards = []
        for preset in presets:
            episodes = all_data[preset]["episodes"]
            # Average of last 20% of episodes
            n_last = max(1, len(episodes) // 5)
            avg_final = np.mean([e["reward"] for e in episodes[-n_last:]])
            final_rewards.append(avg_final)

        bars = ax2.bar(
            presets,
            final_rewards,
            color=[COLORS.get(p, "#333333") for p in presets],
            edgecolor="black",
            linewidth=1.5,
        )
        ax2.set_ylabel("Avg Final Reward", fontsize=11)
        ax2.set_title("Final Performance", fontsize=13, fontweight="bold")
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, final_rewards):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

    # --- Plots 3-5: Individual Preset Details (middle row) ---
    if num_presets > 1:
        for idx, preset in enumerate(["throughput", "energy", "queue"]):
            ax = fig.add_subplot(gs[1, idx])

            if preset in all_data:
                episodes = all_data[preset]["episodes"]
                timesteps = [e["timestep"] for e in episodes]
                rewards = [e["reward"] for e in episodes]

                # Raw and smoothed
                ax.plot(
                    timesteps, rewards, alpha=0.3, color=COLORS[preset], linewidth=0.5
                )
                if len(rewards) > smooth_window:
                    smoothed = smooth(rewards, smooth_window)
                    ax.plot(
                        timesteps[smooth_window - 1 :],
                        smoothed,
                        color=COLORS[preset],
                        linewidth=2,
                    )

                # Add trend line
                z = np.polyfit(range(len(rewards)), rewards, 1)
                trend = np.polyval(z, range(len(rewards)))
                ax.plot(
                    timesteps,
                    trend,
                    "--",
                    color="black",
                    alpha=0.5,
                    label=f"Trend: {z[0]:.4f}/ep",
                )

                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
                ax.legend(loc="lower right", fontsize=9)

            ax.set_xlabel("Timesteps", fontsize=10)
            ax.set_ylabel("Reward", fontsize=10)
            ax.set_title(
                f"{PRESET_NAMES.get(preset, preset)} Preset",
                fontsize=12,
                fontweight="bold",
                color=COLORS.get(preset, "#333333"),
            )
    else:
        # Single preset: show detailed plot spanning full width
        preset = presets[0]
        ax = fig.add_subplot(gs[0, :])
        episodes = all_data[preset]["episodes"]
        timesteps = [e["timestep"] for e in episodes]
        rewards = [e["reward"] for e in episodes]

        ax.fill_between(
            timesteps, rewards, alpha=0.2, color=COLORS.get(preset, "#333333")
        )
        ax.plot(
            timesteps,
            rewards,
            alpha=0.4,
            color=COLORS.get(preset, "#333333"),
            linewidth=0.5,
        )

        if len(rewards) > smooth_window:
            smoothed = smooth(rewards, smooth_window)
            ax.plot(
                timesteps[smooth_window - 1 :],
                smoothed,
                color=COLORS.get(preset, "#333333"),
                linewidth=2.5,
                label="Smoothed",
            )

        # Add trend line
        if len(rewards) > 1:
            z = np.polyfit(range(len(rewards)), rewards, 1)
            trend = np.polyval(z, range(len(rewards)))
            ax.plot(
                timesteps,
                trend,
                "--",
                color="black",
                alpha=0.5,
                label=f"Trend: {z[0]:.6f}/ep",
            )

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Timesteps", fontsize=11)
        ax.set_ylabel("Episode Reward", fontsize=11)
        ax.set_title(
            f"Training Progression: {PRESET_NAMES.get(preset, preset)} Preset",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10)

        # Stats box
        n_last = max(1, len(rewards) // 5)
        stats_text = f"Episodes: {len(episodes)}\n"
        stats_text += f"Final Avg: {np.mean(rewards[-n_last:]):.2f}\n"
        stats_text += f"Best: {max(rewards):.2f}\n"
        stats_text += f"Worst: {min(rewards):.2f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # --- Plots 6-8: Metrics Comparison (bottom row, only if multiple presets) ---
    if num_presets > 1:
        metrics = ["total_bytes_tx", "total_energy_mj", "total_drops"]
        metric_labels = ["Throughput (MB)", "Energy (mJ)", "Packet Drops"]
        metric_scales = [1e6, 1, 1]  # Scale factors

        for idx, (metric, label, scale) in enumerate(
            zip(metrics, metric_labels, metric_scales)
        ):
            ax = fig.add_subplot(gs[2, idx])

            for preset, data in all_data.items():
                episodes = data["episodes"]
                if episodes and metric in episodes[0]:
                    timesteps = [e["timestep"] for e in episodes]
                    values = [e[metric] / scale for e in episodes]

                    # Smoothed only
                    if len(values) > smooth_window:
                        smoothed = smooth(values, smooth_window)
                        ax.plot(
                            timesteps[smooth_window - 1 :],
                            smoothed,
                            color=COLORS.get(preset, "#333333"),
                            linewidth=2,
                            label=PRESET_NAMES.get(preset, preset),
                        )
                    else:
                        ax.plot(
                            timesteps,
                            values,
                            color=COLORS.get(preset, "#333333"),
                            linewidth=2,
                            label=PRESET_NAMES.get(preset, preset),
                        )

            ax.set_xlabel("Timesteps", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(f"{label} Over Training", fontsize=12, fontweight="bold")
            ax.legend(loc="best", fontsize=9)
    else:
        # Single preset: show metrics for that preset in bottom row
        preset = presets[0]
        episodes = all_data[preset]["episodes"]

        metrics = ["total_bytes_tx", "total_energy_mj", "total_drops"]
        metric_labels = ["Throughput (MB)", "Energy (mJ)", "Packet Drops"]
        metric_scales = [1e6, 1, 1]

        for idx, (metric, label, scale) in enumerate(
            zip(metrics, metric_labels, metric_scales)
        ):
            ax = fig.add_subplot(gs[1, idx])

            if episodes and metric in episodes[0]:
                timesteps = [e["timestep"] for e in episodes]
                values = [e[metric] / scale for e in episodes]

                ax.fill_between(
                    timesteps, values, alpha=0.2, color=COLORS.get(preset, "#333333")
                )
                ax.plot(
                    timesteps,
                    values,
                    alpha=0.4,
                    color=COLORS.get(preset, "#333333"),
                    linewidth=0.5,
                )

                if len(values) > smooth_window:
                    smoothed = smooth(values, smooth_window)
                    ax.plot(
                        timesteps[smooth_window - 1 :],
                        smoothed,
                        color=COLORS.get(preset, "#333333"),
                        linewidth=2,
                        label="Smoothed",
                    )
                    ax.legend(loc="best", fontsize=9)

            ax.set_xlabel("Timesteps", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(f"{label} Over Training", fontsize=12, fontweight="bold")

    # --- Final touches ---
    title_text = (
        f'PPO Training Analysis: {", ".join([PRESET_NAMES.get(p, p) for p in presets])}'
    )
    plt.suptitle(title_text, fontsize=16, fontweight="bold", y=0.98)

    # Add timestamp
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

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save or show
    if output_path:
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Saved: {output_path}")

    # Also save individual preset plots
    save_dir = os.path.dirname(output_path) if output_path else "."
    for preset, data in all_data.items():
        fig_single, ax_single = plt.subplots(figsize=(10, 6))
        episodes = data["episodes"]
        timesteps = [e["timestep"] for e in episodes]
        rewards = [e["reward"] for e in episodes]

        ax_single.fill_between(timesteps, rewards, alpha=0.2, color=COLORS[preset])
        ax_single.plot(
            timesteps, rewards, alpha=0.4, color=COLORS[preset], linewidth=0.5
        )

        if len(rewards) > smooth_window:
            smoothed = smooth(rewards, smooth_window)
            ax_single.plot(
                timesteps[smooth_window - 1 :],
                smoothed,
                color=COLORS[preset],
                linewidth=2.5,
            )

        ax_single.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax_single.set_xlabel("Timesteps", fontsize=12)
        ax_single.set_ylabel("Episode Reward", fontsize=12)
        ax_single.set_title(
            f"PPO Training: {PRESET_NAMES[preset]} Preset",
            fontsize=14,
            fontweight="bold",
        )

        # Stats box
        n_last = max(1, len(rewards) // 5)
        stats_text = f"Episodes: {len(episodes)}\n"
        stats_text += f"Final Avg: {np.mean(rewards[-n_last:]):.2f}\n"
        stats_text += f"Best: {max(rewards):.2f}\n"
        stats_text += f"Worst: {min(rewards):.2f}"
        ax_single.text(
            0.02,
            0.98,
            stats_text,
            transform=ax_single.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        single_path = os.path.join(save_dir, f"training_{file_prefix}{preset}.png")
        fig_single.savefig(
            single_path,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"Saved: {single_path}")
        plt.close(fig_single)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot PPO training curves")
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
        help="Plot only specific preset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/training_summary.png",
        help="Output file path",
    )
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size")

    args = parser.parse_args()

    # Derive checkpoint pattern from training script name
    checkpoint_pattern = derive_pattern_from_script(args.training_script)

    # Find checkpoints
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

    print(f"Found {len(checkpoints)} preset(s): {list(checkpoints.keys())}")

    # Filter by preset if specified
    if args.preset:
        if args.preset in checkpoints:
            checkpoints = {args.preset: checkpoints[args.preset]}
        else:
            print(f"Error: Preset '{args.preset}' not found")
            sys.exit(1)

    # Generate plots with pattern prefix in filename
    # Replace default output filename to include pattern
    output_filename = f"training_{checkpoint_pattern}summary.png"
    output_path = os.path.join(plots_dir, output_filename)
    plot_all_presets(
        checkpoints, output_path, args.smooth, file_prefix=checkpoint_pattern
    )


if __name__ == "__main__":
    main()

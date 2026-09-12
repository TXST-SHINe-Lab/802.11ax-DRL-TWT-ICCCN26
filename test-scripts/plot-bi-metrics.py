#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
plot_bi_metrics.py - Plot BI-level metrics from NS-3 simulation logs

Reads the ns3-BI-log-*.csv file and creates per-STA visualizations to analyze
throughput, efficiency, packet drops, power consumption, and BSR state.

Every oracle_* metric in the BI log is a cumulative counter, so this script differences
consecutive rows to get per-BI values before plotting.

Located in: test-scripts/
Reads logs from: ../data-log/

Usage:
    python3.11 plot_bi_metrics.py [--save] [--show]
"""

import sys
import os
import glob
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Suppress tight_layout warnings (handled with constrained_layout)
warnings.filterwarnings("ignore", message=".*tight_layout.*")

# Set style
plt.style.use("seaborn-v0_8-whitegrid")


def parse_twt_constants(constants_file=None):
    """
    Parse constants from twt-constants.h header file.

    Returns dict with parsed values, falls back to defaults if file not found.
    """
    defaults = {
        "BEACON_INTERVAL_MS": 102.4,
        "TWT_UPDATE_INTERVAL_BI": 20,
        "TWT_UPDATE_START_BI": 90,
    }

    if constants_file is None:
        # Look for twt-constants.h relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        constants_file = os.path.join(script_dir, "..", "twt-constants.h")

    if not os.path.exists(constants_file):
        print(f"  Warning: {constants_file} not found, using defaults")
        return defaults

    constants = defaults.copy()

    # Regex to match #define CONSTANT_NAME value
    define_pattern = re.compile(r"#define\s+(\w+)\s+([\d.]+)")

    try:
        with open(constants_file, "r") as f:
            for line in f:
                match = define_pattern.match(line.strip())
                if match:
                    name, value = match.groups()
                    if name in constants:
                        # Convert to appropriate type
                        if "." in value:
                            constants[name] = float(value)
                        else:
                            constants[name] = int(value)

        print(f"  Loaded constants from {os.path.basename(constants_file)}:")
        for key, val in constants.items():
            print(f"    {key} = {val}")
    except Exception as e:
        print(f"  Warning: Error reading {constants_file}: {e}, using defaults")

    return constants


# Parse constants from header file
_CONSTANTS = parse_twt_constants()
BEACON_INTERVAL_MS = _CONSTANTS["BEACON_INTERVAL_MS"]
TWT_UPDATE_INTERVAL_BI = _CONSTANTS["TWT_UPDATE_INTERVAL_BI"]
TWT_UPDATE_START_BI = _CONSTANTS["TWT_UPDATE_START_BI"]


def find_latest_file(logdir, pattern):
    """Find the most recent file matching pattern"""
    files = glob.glob(os.path.join(logdir, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def find_latest_bi_log(logdir="data-log"):
    """Find the most recent BI-level log file"""
    return find_latest_file(logdir, "ns3-BI-log-*.csv")


def find_latest_call_log(logdir="data-log"):
    """Find the most recent call-level log file"""
    return find_latest_file(logdir, "ns3-call-log-*.csv")


def find_latest_twt_wrapper_log(logdir="data-log"):
    """Find the most recent TWT wrapper log file"""
    return find_latest_file(logdir, "ns3-twt-wrapper-*.csv")


def load_bi_data(filepath):
    """Load and preprocess BI-level data

    CRITICAL: All oracle_* metrics are CUMULATIVE. We compute per-BI deltas.
    """
    df = pd.read_csv(filepath)

    # Sort by sta_id and bi_index for proper delta calculation
    df = df.sort_values(["sta_id", "bi_index"]).reset_index(drop=True)

    # Convert simulation time to seconds
    df["time_sec"] = df["observation_time_ms"] / 1000.0

    # --- oracle_* metrics are cumulative, so plot deltas rather than raw values ---

    # Compute per-BI deltas for cumulative oracle metrics
    cumulative_cols = [
        "oracle_bytes_transmitted",
        "oracle_packets_transmitted",
        "oracle_total_energy_mj",
        "oracle_awake_time_ms",
        "oracle_packets_generated",
        "oracle_bytes_generated",
        "oracle_mpdu_drops_expired",
        "oracle_mpdu_drops_queue_full",
        "oracle_psdu_timeouts",
    ]

    for col in cumulative_cols:
        if col in df.columns:
            delta_col = col.replace("oracle_", "") + "_delta"
            # Compute delta within each STA
            df[delta_col] = df.groupby("sta_id")[col].diff()
            # First row of each STA has NaN delta - use the raw value
            first_row_mask = df.groupby("sta_id").cumcount() == 0
            df.loc[first_row_mask, delta_col] = df.loc[first_row_mask, col]
            # Ensure non-negative (shouldn't happen, but safety)
            df[delta_col] = df[delta_col].clip(lower=0)

    # Throughput: bytes transmitted per BI interval -> Mbps
    # bytes_delta / (BI_interval_seconds) * 8 bits / 1e6
    bi_interval_sec = BEACON_INTERVAL_MS / 1000.0
    df["throughput_mbps"] = (df["bytes_transmitted_delta"] * 8 / 1e6) / bi_interval_sec

    # Energy per BI (delta)
    df["energy_consumed_mj"] = df["total_energy_mj_delta"]

    # Keep cumulative values for reference
    df["energy_cumulative_mj"] = df["oracle_total_energy_mj"]
    df["packets_cumulative"] = df["oracle_packets_transmitted"]

    # Packets transmitted per BI (delta)
    df["packets_per_bi"] = df["packets_transmitted_delta"]

    # Duty cycle: percentage of BI spent active (0-1)
    if "oracle_duty_cycle" in df.columns:
        df["duty_cycle"] = df["oracle_duty_cycle"]
    else:
        df["duty_cycle"] = 0.0

    # Total packets transmitted (cumulative - for summary)
    df["total_packets_transmitted"] = df["oracle_packets_transmitted"]

    # Efficiency: bytes per mJ (delta bytes / delta energy)
    df["efficiency"] = df["bytes_transmitted_delta"] / df["energy_consumed_mj"].replace(
        0, np.nan
    )

    # Packet drops per BI (delta)
    if "mpdu_drops_expired_delta" in df.columns:
        df["mpdu_drops_expired"] = df["mpdu_drops_expired_delta"]
    else:
        df["mpdu_drops_expired"] = 0

    if "mpdu_drops_queue_full_delta" in df.columns:
        df["mpdu_drops_failed_enqueue"] = df["mpdu_drops_queue_full_delta"]
    else:
        df["mpdu_drops_failed_enqueue"] = 0

    # BSR queue size (instantaneous, not cumulative)
    if "oracle_queue_size_bytes" in df.columns:
        df["bsr_queue_size_bytes"] = df["oracle_queue_size_bytes"]
    else:
        df["bsr_queue_size_bytes"] = 0

    # Power consumption: energy per BI / time -> mW
    df["power_consumption_mw"] = df["energy_consumed_mj"] / bi_interval_sec

    # Awake time per BI (delta)
    df["awake_time_per_bi_ms"] = df["awake_time_ms_delta"]

    # Replace inf/nan with NaN for plotting
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def load_call_data(filepath):
    """Load and preprocess call-level data

    CRITICAL: Call-level oracle_* metrics are also CUMULATIVE.
    We compute per-call deltas for proper visualization.
    """
    if not filepath or not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)

    # Sort by sta_id and call_index for proper delta calculation
    df = df.sort_values(["sta_id", "call_index"]).reset_index(drop=True)

    # Convert simulation time to seconds (call log uses simulation_time_ms)
    df["time_sec"] = df["simulation_time_ms"] / 1000.0

    # Calculate BI index for each call (end of window)
    df["end_bi"] = (df["simulation_time_ms"] / BEACON_INTERVAL_MS).astype(int)

    # Call window matches the TWT update interval (from twt-constants.h)
    CALL_WINDOW_BIS = TWT_UPDATE_INTERVAL_BI
    df["start_bi"] = df["end_bi"] - CALL_WINDOW_BIS

    # --- oracle_* metrics are cumulative, so plot deltas rather than raw values ---
    cumulative_cols = [
        "oracle_bytes_transmitted",
        "oracle_packets_transmitted",
        "oracle_total_energy_mj",
        "oracle_awake_time_ms",
    ]

    for col in cumulative_cols:
        if col in df.columns:
            delta_col = col.replace("oracle_", "") + "_delta"
            # Compute delta within each STA
            df[delta_col] = df.groupby("sta_id")[col].diff()
            # First row of each STA has NaN delta - use the raw value
            first_row_mask = df.groupby("sta_id").cumcount() == 0
            df.loc[first_row_mask, delta_col] = df.loc[first_row_mask, col]
            # Ensure non-negative
            df[delta_col] = df[delta_col].clip(lower=0)

    # Throughput: bytes per call window -> Mbps
    call_window_sec = CALL_WINDOW_BIS * BEACON_INTERVAL_MS / 1000.0
    if "bytes_transmitted_delta" in df.columns:
        df["throughput_mbps"] = (
            df["bytes_transmitted_delta"] * 8 / 1e6
        ) / call_window_sec
    else:
        df["throughput_mbps"] = 0

    # Energy per call window (delta)
    if "total_energy_mj_delta" in df.columns:
        df["energy_consumed_mj"] = df["total_energy_mj_delta"]
    elif "oracle_total_energy_mj" in df.columns:
        df["energy_consumed_mj"] = df["oracle_total_energy_mj"]
    else:
        df["energy_consumed_mj"] = 0

    # Efficiency: bytes per mJ (from deltas)
    if (
        "bytes_transmitted_delta" in df.columns
        and "total_energy_mj_delta" in df.columns
    ):
        df["efficiency"] = df["bytes_transmitted_delta"] / df[
            "total_energy_mj_delta"
        ].replace(0, np.nan)
    else:
        df["efficiency"] = np.nan

    # Packet drops
    if "oracle_mpdu_drops_expired" in df.columns:
        df["mpdu_drops_expired"] = (
            df.groupby("sta_id")["oracle_mpdu_drops_expired"]
            .diff()
            .fillna(df["oracle_mpdu_drops_expired"])
        )
    else:
        df["mpdu_drops_expired"] = 0

    # Power: energy / time -> mW
    df["power_consumption_mw"] = df["energy_consumed_mj"] / call_window_sec

    # BSR queue size in KB (instantaneous at call time)
    if "oracle_queue_size_bytes" in df.columns:
        df["bsr_queue_kb"] = df["oracle_queue_size_bytes"] / 1000.0
    else:
        df["bsr_queue_kb"] = 0

    # Replace inf/nan with NaN for plotting
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def get_update_bis_from_wrapper(logdir):
    """Get actual BI indices where config updates occurred from twt-wrapper log"""
    wrapper_file = find_latest_twt_wrapper_log(logdir)
    if not wrapper_file or not os.path.exists(wrapper_file):
        print("Warning: No TWT wrapper log found, cannot determine update BIs")
        return []

    try:
        df = pd.read_csv(wrapper_file)
        # Wrapper log has Timestamp_Sec column - convert to BI
        update_times_sec = df["Timestamp_Sec"].values
        update_bis = [
            int(round(t * 1000.0 / BEACON_INTERVAL_MS)) for t in update_times_sec
        ]
        return update_bis
    except Exception as e:
        print(f"Warning: Could not parse wrapper log: {e}")
        return []


def get_update_bis(df, logdir):
    """Get BI indices where config updates occurred"""
    # Try to get actual update BIs from wrapper log first
    update_bis = get_update_bis_from_wrapper(logdir)
    if update_bis:
        print(f"  Found {len(update_bis)} config updates at BIs: {update_bis}")
        return update_bis

    # Fallback: estimate from config constants (uses globals from twt-constants.h)
    print("  Warning: Using estimated update BIs (no wrapper log found)")
    max_bi = df["bi_index"].max()

    update_bis = []
    bi = TWT_UPDATE_START_BI
    while bi <= max_bi:
        update_bis.append(bi)
        bi += TWT_UPDATE_INTERVAL_BI

    return update_bis


def add_update_markers(ax, update_bis, ymin=None, ymax=None):
    """Add vertical lines at config update points"""
    for i, bi in enumerate(update_bis):
        ax.axvline(x=bi, color="red", linestyle="--", alpha=0.5, linewidth=1.5)
        if ymax is not None:
            ax.text(
                bi + 2,
                ymax * 0.95,
                f"Config {i}",
                fontsize=8,
                color="red",
                rotation=90,
                va="top",
                alpha=0.7,
            )


def set_xlim_from_update_start(ax, df, margin=5):
    """Set x-axis limits starting from TWT_UPDATE_START_BI.

    Args:
        ax: Matplotlib axis
        df: DataFrame with bi_index column
        margin: Extra BIs to show after last data point
    """
    bi_max = df["bi_index"].max() if "bi_index" in df.columns else df.index.max()
    ax.set_xlim(TWT_UPDATE_START_BI - margin, bi_max + margin)


def add_call_overlay(
    ax, call_df, sta_id, metric_col, color="green", label="Call-level"
):
    """Add call-level overlay as horizontal spans showing windowed average.

    The call-level metric represents the average over a 20-BI window ending at the call time.
    This draws horizontal lines spanning from start_bi to end_bi at the call-level value.
    """
    if call_df is None or call_df.empty:
        return

    sta_call_df = call_df[call_df["sta_id"] == sta_id].sort_values("call_index")
    if sta_call_df.empty or metric_col not in sta_call_df.columns:
        return

    for idx, row in sta_call_df.iterrows():
        start_bi = row["start_bi"]
        end_bi = row["end_bi"]
        value = row[metric_col]
        call_idx = row["call_index"]

        if pd.isna(value):
            continue

        # Draw horizontal line spanning the call window
        ax.hlines(
            y=value,
            xmin=start_bi,
            xmax=end_bi,
            color=color,
            linewidth=2.5,
            alpha=0.8,
            label=label if idx == sta_call_df.index[0] else None,
        )

        # Add shaded region to show the window
        ax.axvspan(start_bi, end_bi, alpha=0.1, color=color)

        # Add label for call index
        mid_bi = (start_bi + end_bi) / 2
        ax.text(
            mid_bi,
            value * 1.02,
            f"Call {int(call_idx)}",
            fontsize=8,
            color=color,
            ha="center",
            va="bottom",
            fontweight="bold",
        )


def add_call_point_overlay(
    ax, call_df, sta_id, metric_col, color="green", label="Call-level (snapshot)"
):
    """Add call-level overlay as markers at end_bi (instantaneous snapshots).

    For metrics like queue size that are snapshots at call time, not window averages.
    This plots markers at the end_bi position where the measurement was taken.
    """
    if call_df is None or call_df.empty:
        return

    sta_call_df = call_df[call_df["sta_id"] == sta_id].sort_values("call_index")
    if sta_call_df.empty or metric_col not in sta_call_df.columns:
        return

    end_bis = []
    values = []
    for idx, row in sta_call_df.iterrows():
        end_bi = row["end_bi"]
        value = row[metric_col]
        call_idx = row["call_index"]

        if pd.isna(value):
            continue

        end_bis.append(end_bi)
        values.append(value)

        # Add label for call index
        ax.text(
            end_bi,
            value * 1.05,
            f"C{int(call_idx)}",
            fontsize=7,
            color=color,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot all points as markers
    if end_bis:
        ax.scatter(
            end_bis,
            values,
            color=color,
            s=80,
            marker="D",
            alpha=0.9,
            label=label,
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )


def plot_per_sta_throughput(df, sta_id, ax, update_bis, call_df=None):
    """Plot throughput for a single STA with optional call-level overlay"""
    sta_df = df[df["sta_id"] == sta_id].sort_values("bi_index")
    # Filter to data from update start for y-axis scaling
    sta_df_filtered = sta_df[sta_df["bi_index"] >= TWT_UPDATE_START_BI]

    ax.plot(
        sta_df["bi_index"],
        sta_df["throughput_mbps"],
        color="blue",
        linewidth=1,
        alpha=0.8,
        label="BI-level",
    )
    ax.fill_between(
        sta_df["bi_index"], 0, sta_df["throughput_mbps"], color="blue", alpha=0.2
    )

    # Add call-level overlay
    add_call_overlay(
        ax, call_df, sta_id, "throughput_mbps", color="green", label="Call-level avg"
    )

    ymax = (
        sta_df_filtered["throughput_mbps"].max() * 1.1
        if len(sta_df_filtered) > 0 and sta_df_filtered["throughput_mbps"].max() > 0
        else 1
    )
    add_update_markers(ax, update_bis, ymax=ymax)

    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title(f"STA {sta_id}: Throughput (BI vs Call-level)")
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, sta_df)
    ax.legend(loc="upper right", fontsize=8)


def plot_per_sta_efficiency(df, sta_id, ax, update_bis, call_df=None):
    """Plot efficiency (bytes/energy) for a single STA"""
    sta_df = df[df["sta_id"] == sta_id].sort_values("bi_index")
    # Filter to data from update start for y-axis scaling
    sta_df_filtered = sta_df[sta_df["bi_index"] >= TWT_UPDATE_START_BI]

    # Use efficiency column directly from CSV (bytes per mJ)
    # This matches the C++ calculation: deltaBytesTransmitted / energy_consumed_mj
    efficiency = (
        sta_df["efficiency"]
        if "efficiency" in sta_df.columns
        else (
            sta_df["throughput_mbps"]
            * 1e6
            / 8
            / sta_df["energy_consumed_mj"].replace(0, np.nan)
        )
    )

    ax.plot(
        sta_df["bi_index"],
        efficiency,
        color="teal",
        linewidth=1,
        alpha=0.8,
        label="BI-level",
    )

    # Add call-level overlay (uses same efficiency column from call-level CSV)
    add_call_overlay(
        ax, call_df, sta_id, "efficiency", color="green", label="Call-level avg"
    )

    efficiency_filtered = (
        sta_df_filtered["efficiency"]
        if "efficiency" in sta_df_filtered.columns
        else efficiency[sta_df["bi_index"] >= TWT_UPDATE_START_BI]
    )
    ymax = (
        efficiency_filtered.max() * 1.1
        if len(efficiency_filtered) > 0 and efficiency_filtered.max() > 0
        else 1
    )
    add_update_markers(ax, update_bis, ymax=ymax)

    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("Efficiency (bytes/mJ)")
    ax.set_title(f"STA {sta_id}: Energy Efficiency")
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, sta_df)
    ax.legend(loc="upper right", fontsize=8)


def plot_per_sta_packet_drops(df, sta_id, ax, update_bis, call_df=None):
    """Plot packet drop rates (expired and failed enqueue) for a single STA"""
    sta_df = df[df["sta_id"] == sta_id].sort_values("bi_index")
    # Filter to data from update start for y-axis scaling
    sta_df_filtered = sta_df[sta_df["bi_index"] >= TWT_UPDATE_START_BI]

    # Get expired and failed enqueue drop rates
    expired_drops = (
        sta_df["mpdu_drops_expired"]
        if "mpdu_drops_expired" in sta_df.columns
        else pd.Series([0] * len(sta_df))
    )
    failed_enqueue = (
        sta_df["mpdu_drops_failed_enqueue"]
        if "mpdu_drops_failed_enqueue" in sta_df.columns
        else pd.Series([0] * len(sta_df))
    )

    # Plot both drop types
    ax.plot(
        sta_df["bi_index"],
        expired_drops,
        color="red",
        linewidth=1.5,
        alpha=0.8,
        label="Expired (BI)",
    )
    ax.fill_between(sta_df["bi_index"], 0, expired_drops, color="red", alpha=0.2)

    ax.plot(
        sta_df["bi_index"],
        failed_enqueue,
        color="blue",
        linewidth=1.5,
        alpha=0.8,
        label="Failed Enq (BI)",
    )
    ax.fill_between(sta_df["bi_index"], 0, failed_enqueue, color="blue", alpha=0.1)

    # Add call-level overlay for total drops
    add_call_overlay(
        ax,
        call_df,
        sta_id,
        "mpdu_drops_expired",
        color="darkgreen",
        label="Expired (Call)",
    )

    # Filter drops for y-axis scaling
    expired_filtered = expired_drops[sta_df["bi_index"] >= TWT_UPDATE_START_BI]
    failed_filtered = failed_enqueue[sta_df["bi_index"] >= TWT_UPDATE_START_BI]
    max_drop = max(
        expired_filtered.max() if len(expired_filtered) > 0 else 0,
        failed_filtered.max() if len(failed_filtered) > 0 else 0,
    )
    ymax = max_drop * 1.1 if max_drop > 0 else 1
    add_update_markers(ax, update_bis, ymax=ymax)

    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("Packet Drops (per BI)")
    ax.set_title(f"STA {sta_id}: Packet Drop Rates")
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, sta_df)
    ax.legend(loc="upper right", fontsize=8)

    # Add annotation if no drops
    total_drops = expired_filtered.sum() + failed_filtered.sum()
    if total_drops == 0:
        ax.text(
            0.5,
            0.5,
            "No packet drops\nin this period",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            alpha=0.7,
        )


def plot_per_sta_power(df, sta_id, ax, update_bis, call_df=None):
    """Plot power consumption for a single STA"""
    sta_df = df[df["sta_id"] == sta_id].sort_values("bi_index")
    # Filter to data from update start for y-axis scaling
    sta_df_filtered = sta_df[sta_df["bi_index"] >= TWT_UPDATE_START_BI]

    # Use power_consumption_mw if available, otherwise calculate from avg_current_ma
    if "power_consumption_mw" in sta_df.columns:
        power_mw = sta_df["power_consumption_mw"]
    elif "avg_current_ma" in sta_df.columns:
        # Power = Current * Voltage (assume 3.7V typical LiPo battery)
        power_mw = sta_df["avg_current_ma"] * 3.7
    else:
        power_mw = pd.Series([0] * len(sta_df))

    ax.plot(
        sta_df["bi_index"],
        power_mw,
        color="orange",
        linewidth=1.5,
        alpha=0.8,
        label="BI-level",
    )
    ax.fill_between(sta_df["bi_index"], 0, power_mw, color="orange", alpha=0.2)

    # Add call-level overlay
    add_call_overlay(
        ax,
        call_df,
        sta_id,
        "power_consumption_mw",
        color="green",
        label="Call-level avg",
    )

    power_filtered = power_mw[sta_df["bi_index"] >= TWT_UPDATE_START_BI]
    ymax = (
        power_filtered.max() * 1.1
        if len(power_filtered) > 0 and power_filtered.max() > 0
        else 1
    )
    add_update_markers(ax, update_bis, ymax=ymax)

    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("Power (mW)")
    ax.set_title(f"STA {sta_id}: Power Consumption")
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, sta_df)
    ax.legend(loc="upper right", fontsize=8)


def plot_per_sta_bsr(df, sta_id, ax, update_bis, call_df=None):
    """Plot BSR queue state for a single STA"""
    sta_df = df[df["sta_id"] == sta_id].sort_values("bi_index")
    # Filter to data from update start for y-axis scaling
    sta_df_filtered = sta_df[sta_df["bi_index"] >= TWT_UPDATE_START_BI]

    # BSR queue size in KB
    queue_kb = sta_df["bsr_queue_size_bytes"] / 1000

    ax.plot(
        sta_df["bi_index"],
        queue_kb,
        color="purple",
        linewidth=1,
        alpha=0.8,
        label="BI-level",
    )
    ax.fill_between(sta_df["bi_index"], 0, queue_kb, color="purple", alpha=0.2)

    # Add call-level overlay as point markers (queue is a snapshot, not window avg)
    add_call_point_overlay(
        ax, call_df, sta_id, "bsr_queue_kb", color="green", label="Call snapshot"
    )

    queue_kb_filtered = (
        sta_df_filtered["bsr_queue_size_bytes"] / 1000
        if len(sta_df_filtered) > 0
        else queue_kb
    )
    ymax = (
        queue_kb_filtered.max() * 1.1
        if len(queue_kb_filtered) > 0 and queue_kb_filtered.max() > 0
        else 1
    )
    add_update_markers(ax, update_bis, ymax=ymax)

    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("BSR Queue Size (KB)")
    ax.set_title(f"STA {sta_id}: BSR Queue State")
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, sta_df)
    ax.legend(loc="upper right", fontsize=8)


def plot_per_sta_figure(df, sta_id, update_bis, logdir, save=False, call_df=None):
    """Create a figure with all metrics for a single STA"""
    # Use constrained_layout instead of tight_layout to avoid warnings
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        f"STA {sta_id} - BI-Level Metrics (Red=Config Updates, Green=Call-level)",
        fontsize=14,
        fontweight="bold",
    )

    # Row 1
    plot_per_sta_throughput(df, sta_id, axes[0, 0], update_bis, call_df=call_df)
    plot_per_sta_efficiency(df, sta_id, axes[0, 1], update_bis, call_df=call_df)

    # Row 2
    plot_per_sta_power(df, sta_id, axes[1, 0], update_bis, call_df=call_df)
    plot_per_sta_bsr(df, sta_id, axes[1, 1], update_bis, call_df=call_df)

    # Add summary stats as a flat text at the bottom of the figure
    sta_df = df[df["sta_id"] == sta_id]
    sta_df_filtered = sta_df[sta_df["bi_index"] >= TWT_UPDATE_START_BI]

    summary_text = (
        f"STA {sta_id} Summary (BI≥{TWT_UPDATE_START_BI}):  "
        f"Throughput: {sta_df_filtered['throughput_mbps'].mean():.2f} Mbps (avg), {sta_df_filtered['throughput_mbps'].max():.2f} (max)  |  "
        f"Energy: {sta_df_filtered['energy_consumed_mj'].sum():.1f} mJ (total), {sta_df_filtered['energy_consumed_mj'].mean():.2f} mJ/BI  |  "
        f"Duty Cycle: {sta_df_filtered['duty_cycle'].mean()*100:.1f}%  |  "
        f"Packets: {int(sta_df_filtered['packets_per_bi'].sum())} ({sta_df_filtered['packets_per_bi'].mean():.1f}/BI)  |  "
        f"BIs: {len(sta_df_filtered)}"
    )

    # Adjust subplot positions to make room for bottom banner
    fig.subplots_adjust(bottom=0.10)

    fig.text(
        0.5,
        -0.02,
        summary_text,
        ha="center",
        va="bottom",
        fontsize=10,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round", facecolor="lightyellow", alpha=0.8, edgecolor="gray"
        ),
    )

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(logdir, f"sta{sta_id}_metrics_{timestamp}.png")
        # Use bbox_inches="tight" with pad to add margin
        plt.savefig(outfile, dpi=150, bbox_inches="tight", pad_inches=0.3)
        print(f"  Saved: {outfile}")

    return fig


def plot_network_overview(df, update_bis, logdir, save=False):
    """Create a network overview figure with aggregate metrics"""
    # Use constrained_layout instead of tight_layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(
        "Network Overview - All STAs (Red lines = Config Updates)",
        fontsize=14,
        fontweight="bold",
    )

    # Aggregate by BI
    agg_df = (
        df.groupby("bi_index")
        .agg(
            {
                "throughput_mbps": "sum",
                "energy_consumed_mj": "sum",
                "duty_cycle": "mean",
            }
        )
        .reset_index()
    )

    # Filter for y-axis scaling (only data from update start)
    agg_df_filtered = agg_df[agg_df["bi_index"] >= TWT_UPDATE_START_BI]
    df_filtered = df[df["bi_index"] >= TWT_UPDATE_START_BI]

    # Total throughput
    ax = axes[0, 0]
    ax.plot(agg_df["bi_index"], agg_df["throughput_mbps"], color="blue", linewidth=1.5)
    ymax = (
        agg_df_filtered["throughput_mbps"].max() * 1.1
        if len(agg_df_filtered) > 0
        else 1
    )
    add_update_markers(ax, update_bis, ymax=ymax)
    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("Total Throughput (Mbps)")
    ax.set_title("Network Total Throughput")
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, agg_df)

    # Per-STA throughput overlay
    ax = axes[0, 1]
    for sta_id in sorted(df["sta_id"].unique()):
        sta_df = df[df["sta_id"] == sta_id].sort_values("bi_index")
        ax.plot(
            sta_df["bi_index"],
            sta_df["throughput_mbps"],
            label=f"STA {sta_id}",
            alpha=0.7,
            linewidth=1,
        )
    ymax = df_filtered["throughput_mbps"].max() * 1.1 if len(df_filtered) > 0 else 1
    add_update_markers(ax, update_bis, ymax=ymax)
    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title("Per-STA Throughput")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, df)

    # Total energy per BI
    ax = axes[1, 0]
    ax.plot(
        agg_df["bi_index"], agg_df["energy_consumed_mj"], color="orange", linewidth=1.5
    )
    ymax = (
        agg_df_filtered["energy_consumed_mj"].max() * 1.1
        if len(agg_df_filtered) > 0
        else 1
    )
    add_update_markers(ax, update_bis, ymax=ymax)
    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("Total Energy (mJ/BI)")
    ax.set_title("Network Energy per BI")
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, agg_df)

    # Efficiency over time
    ax = axes[1, 1]
    efficiency = agg_df["throughput_mbps"] / agg_df["energy_consumed_mj"].replace(
        0, np.nan
    )
    ax.plot(agg_df["bi_index"], efficiency, color="green", linewidth=1.5)
    efficiency_filtered = efficiency[agg_df["bi_index"] >= TWT_UPDATE_START_BI]
    ymax = (
        efficiency_filtered.max() * 1.1
        if len(efficiency_filtered) > 0 and efficiency_filtered.max() > 0
        else 1
    )
    add_update_markers(ax, update_bis, ymax=ymax)
    ax.set_xlabel("Beacon Interval (BI)")
    ax.set_ylabel("Efficiency (Mbps/mJ)")
    ax.set_title("Network Efficiency")
    ax.set_ylim(bottom=0, top=ymax)
    set_xlim_from_update_start(ax, agg_df)

    # Note: constrained_layout is set at figure creation, no need for tight_layout

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = os.path.join(logdir, f"network_overview_{timestamp}.png")
        # Don't use bbox_inches="tight" with constrained_layout
        plt.savefig(outfile, dpi=150)
        print(f"  Saved: {outfile}")

    return fig


def print_summary(df, update_bis):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("BI-LEVEL METRICS SUMMARY")
    print("  Note: All metrics are PER-BI (delta) values, not cumulative")
    print("=" * 70)

    # Time range
    bi_min, bi_max = df["bi_index"].min(), df["bi_index"].max()
    t_min, t_max = df["time_sec"].min(), df["time_sec"].max()
    n_bis = df["bi_index"].nunique()
    n_stas = df["sta_id"].nunique()

    print(f"  BI Range: {bi_min} - {bi_max} ({n_bis} BIs)")
    print(f"  Time Range: {t_min:.2f}s - {t_max:.2f}s ({t_max - t_min:.2f}s duration)")
    print(f"  STAs: {n_stas}")
    print(f"  Config Updates at BIs: {update_bis}")
    print()

    # Per-STA summary
    print("  Per-STA Summary:")
    print("  " + "-" * 76)
    print(
        f"  {'STA':>4} | {'Avg Tput':>10} | {'Avg Duty':>10} | {'Total Energy':>12} | {'Pkts TX':>10} | {'Pkts/BI':>8}"
    )
    print(
        f"  {'':>4} | {'(Mbps)':>10} | {'(%)':>10} | {'(mJ)':>12} | {'(cumul)':>10} | {'(avg)':>8}"
    )
    print("  " + "-" * 76)

    for sta_id in sorted(df["sta_id"].unique()):
        sta_df = df[df["sta_id"] == sta_id]
        avg_tput = sta_df["throughput_mbps"].mean()
        avg_duty = sta_df["duty_cycle"].mean() * 100
        # Total energy is sum of deltas (per-BI energy)
        total_energy = sta_df["energy_consumed_mj"].sum()
        # Total packets is the final cumulative value
        total_pkts = (
            int(sta_df["total_packets_transmitted"].iloc[-1]) if len(sta_df) > 0 else 0
        )
        avg_pkts_bi = sta_df["packets_per_bi"].mean()
        print(
            f"  {sta_id:>4} | {avg_tput:>10.3f} | {avg_duty:>10.2f} | {total_energy:>12.2f} | {total_pkts:>10} | {avg_pkts_bi:>8.1f}"
        )

    print("  " + "-" * 76)

    # Aggregate summary
    agg_df = df.groupby("bi_index").agg(
        {"throughput_mbps": "sum", "energy_consumed_mj": "sum"}
    )

    print(f"\n  Network Aggregate:")
    print(f"    Avg Total Throughput: {agg_df['throughput_mbps'].mean():.3f} Mbps")
    print(f"    Std Total Throughput: {agg_df['throughput_mbps'].std():.3f} Mbps")
    print(f"    Total Energy:         {agg_df['energy_consumed_mj'].sum():.2f} mJ")
    if agg_df["energy_consumed_mj"].sum() > 0:
        print(
            f"    Avg Efficiency:       {agg_df['throughput_mbps'].sum() / agg_df['energy_consumed_mj'].sum():.4f} Mbps/mJ"
        )
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot BI-level metrics per STA")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to BI-level CSV file (default: latest in data-log/)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save plots to PNG files (default: True)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save plots to PNG files"
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--sta", type=int, default=None, help="Plot only a specific STA (default: all)"
    )
    args = parser.parse_args()

    # Find log file (data-log is in parent directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    logdir = os.path.join(parent_dir, "data-log")

    if args.file:
        filepath = args.file
    else:
        filepath = find_latest_bi_log(logdir)

    if not filepath or not os.path.exists(filepath):
        print("No BI-level log file found!")
        sys.exit(1)

    print(f"Loading BI-level: {filepath}")

    # Load BI-level data
    df = load_bi_data(filepath)
    print(
        f"Loaded {len(df)} rows ({df['bi_index'].nunique()} BIs, {df['sta_id'].nunique()} STAs)"
    )

    # Load call-level data for overlay
    call_filepath = find_latest_call_log(logdir)
    call_df = None
    if call_filepath and os.path.exists(call_filepath):
        print(f"Loading call-level: {call_filepath}")
        call_df = load_call_data(call_filepath)
        if call_df is not None:
            print(
                f"Loaded {len(call_df)} call-level rows ({call_df['call_index'].nunique()} calls)"
            )
    else:
        print("No call-level log found - skipping call overlay")

    # Get update points
    update_bis = get_update_bis(df, logdir)

    # Print summary
    print_summary(df, update_bis)

    # Determine if saving (default True unless --no-save)
    do_save = args.save and not args.no_save

    # Create plots
    figures = []

    # Network overview
    print("Creating network overview plot...")
    fig_overview = plot_network_overview(df, update_bis, logdir, save=do_save)
    figures.append(fig_overview)

    # Per-STA plots
    if args.sta is not None:
        # Single STA
        sta_ids = [args.sta] if args.sta in df["sta_id"].unique() else []
        if not sta_ids:
            print(f"STA {args.sta} not found in data!")
    else:
        # All STAs
        sta_ids = sorted(df["sta_id"].unique())

    print(f"Creating per-STA plots for {len(sta_ids)} STAs...")
    for sta_id in sta_ids:
        fig_sta = plot_per_sta_figure(
            df, sta_id, update_bis, logdir, save=do_save, call_df=call_df
        )
        figures.append(fig_sta)

    # Show if requested (default: no-show)
    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()

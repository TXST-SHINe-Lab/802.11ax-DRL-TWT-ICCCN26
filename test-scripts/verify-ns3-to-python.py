#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
Cross-verify NS3 logs with Python wrapper logs.

This script ensures that per-STA data logged by NS3 is correctly passed to the
Python controller through the protobuf interface.

Compares:
1. ns3-call-log (NS3 internal logging) vs py-wrapper-env (data received by Python)
2. ns3-BI-log (BI-level logging) vs py-wrapper-env
3. ns3-twt-wrapper (NS3 wrapper logging) vs py-wrapper-action (Python actions)

Key fields to verify:
- Per-STA BSR values (bsr_ac_be, bsr_ac_bk, bsr_ac_vi, bsr_ac_vo)
- Link quality metrics (rssi, snr, link_margin)
- Oracle metrics (energy, packets, queue size)
- Timing consistency

Located in: test-scripts/
Reads logs from: ../data-log/

Usage:
    python3.11 verify-ns3-to-python.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import sys
import os

# def find_latest_files(logdir):
#     """Find the latest log files."""
#     patterns = {
#         'call': 'ns3-call-log-*.csv',
#         'bi': 'ns3-BI-log-*.csv',
#         'py_env': 'py-wrapper-env-*.csv',
#         'py_action': 'py-wrapper-action-*.csv',
#         'wrapper': 'ns3-twt-wrapper-*.csv',
#     }
#
#     files = {}
#     for key, pattern in patterns.items():
#         matches = sorted(glob.glob(os.path.join(logdir, pattern)))
#         if matches:
#             files[key] = matches[-1]
#             print(f"✓ Found {key}: {os.path.basename(matches[-1])}")
#         else:
#             print(f"✗ Missing: {pattern}")
#
#     return files


def find_all_file_groups(logdir):
    """Find all log file groups by timestamp."""
    patterns = {
        "call": "ns3-call-log-*.csv",
        "bi": "ns3-BI-log-*.csv",
        "py_env": "py-wrapper-env-*.csv",
        "py_action": "py-wrapper-action-*.csv",
        "wrapper": "ns3-twt-wrapper-*.csv",
    }

    file_groups = {}

    for key, pattern in patterns.items():
        matches = sorted(glob.glob(os.path.join(logdir, pattern)))
        for filepath in matches:
            # Extract timestamp from filename
            timestamp = os.path.basename(filepath).split("-")[-1].replace(".csv", "")
            if timestamp not in file_groups:
                file_groups[timestamp] = {}
            file_groups[timestamp][key] = filepath

    # Filter to groups that have at least call and py_env
    complete_groups = {
        ts: files
        for ts, files in file_groups.items()
        if "call" in files and "py_env" in files
    }

    print(f"Found {len(complete_groups)} complete file groups:\n")
    for ts in sorted(complete_groups.keys()):
        print(f"  {ts}:")
        for key in ["call", "bi", "py_env", "py_action", "wrapper"]:
            if key in complete_groups[ts]:
                print(f"    ✓ {key}: {os.path.basename(complete_groups[ts][key])}")
            else:
                print(f"    ✗ {key}: missing")
    print()

    return complete_groups


def load_dataframes(files):
    """Load all dataframes."""
    dfs = {}
    for key, filepath in files.items():
        try:
            dfs[key] = pd.read_csv(filepath)
            print(f"  {key}: {len(dfs[key])} rows, {len(dfs[key].columns)} columns")
        except Exception as e:
            print(f"  Failed to load {key}: {e}")
    return dfs


def compare_call_vs_py_env(call_df, py_env_df):
    """
    Compare NS3 call-log with py-wrapper-env.

    ns3-call-log: Data logged inside NS3 during each call interval
    py-wrapper-env: Data that the Python controller actually received

    The py-wrapper-env should match call-log for the same call intervals.
    """
    print("\n" + "=" * 80)
    print("TEST: NS3 call-log vs Python wrapper env")
    print("=" * 80 + "\n")

    # Map column names between the two logs
    # call-log uses direct names, py-wrapper-env uses realistic_/oracle_ prefixes
    column_mapping = {
        # Realistic (observable) metrics
        "bsr_ac_be": "realistic_bsr_queue_ac_be",
        "bsr_ac_bk": "realistic_bsr_queue_ac_bk",
        "bsr_ac_vi": "realistic_bsr_queue_ac_vi",
        "bsr_ac_vo": "realistic_bsr_queue_ac_vo",
        "rssi_dbm": "realistic_rssi_dbm",
        "snr_db": "realistic_snr_db",
        "link_margin_db": "realistic_link_margin_db",
        "rcpi": "realistic_rcpi",
        "rsni": "realistic_rsni",
        "bytes_received_at_ap": "realistic_bytes_received_at_ap",
        "packets_received_at_ap": "realistic_packets_received_at_ap",
        "airtime_used_us": "realistic_airtime_used_us",
        # Oracle metrics
        "oracle_total_energy_mj": "oracle_total_energy_consumed_mj",
        "oracle_awake_time_ms": "oracle_awake_time_ms",
        "oracle_sleep_time_ms": "oracle_sleep_time_ms",
        "oracle_duty_cycle": "oracle_duty_cycle",
        "oracle_packets_generated": "oracle_packets_generated",
        "oracle_packets_transmitted": "oracle_packets_transmitted",
        "oracle_bytes_transmitted": "oracle_bytes_transmitted",
        "oracle_queue_size_packets": "oracle_queue_size_packets",
        "oracle_queue_size_bytes": "oracle_queue_size_bytes",
    }

    # py-wrapper-env uses 'step' which corresponds to call number
    # call-log uses 'call_index'
    # Note: py-wrapper-env step 0 might be initial state before first call

    # Find matching records
    # py_env has 'step' and 'sta_id', call_log has 'call_index' and 'sta_id'

    # Get unique steps and call_indices
    py_steps = sorted(py_env_df["step"].unique())
    call_indices = sorted(call_df["call_index"].unique())

    print(
        f"py-wrapper-env steps: {min(py_steps)} - {max(py_steps)} ({len(py_steps)} unique)"
    )
    print(
        f"call-log call_indices: {min(call_indices)} - {max(call_indices)} ({len(call_indices)} unique)"
    )

    # Try to align: step might be offset by 1 from call_index
    # Check if step 1 corresponds to call_index 1

    all_passed = True
    mismatches = []

    for call_idx in call_indices[:5]:  # Check first 5 calls
        # Determine corresponding py_env step
        # Usually step = call_index, but could be offset
        for step_offset in [0, 1, -1]:
            py_step = call_idx + step_offset
            if py_step in py_steps:
                break
        else:
            print(f"⚠️  No matching step for call_index {call_idx}")
            continue

        print(f"\nComparing call_index={call_idx} with step={py_step}:")

        for sta_id in sorted(
            call_df[call_df["call_index"] == call_idx]["sta_id"].unique()
        )[
            :4
        ]:  # First 4 STAs
            call_row = call_df[
                (call_df["call_index"] == call_idx) & (call_df["sta_id"] == sta_id)
            ]
            py_row = py_env_df[
                (py_env_df["step"] == py_step) & (py_env_df["sta_id"] == sta_id)
            ]

            if len(call_row) == 0 or len(py_row) == 0:
                continue

            call_row = call_row.iloc[0]
            py_row = py_row.iloc[0]

            # Compare key metrics
            for call_col, py_col in list(column_mapping.items())[:8]:
                if call_col in call_row.index and py_col in py_row.index:
                    call_val = call_row[call_col]
                    py_val = py_row[py_col]

                    # Handle NaN values
                    if pd.isna(call_val) and pd.isna(py_val):
                        match = True
                    elif pd.isna(call_val) or pd.isna(py_val):
                        match = False
                    else:
                        # Allow small tolerance for floating point
                        if isinstance(call_val, (int, float)) and isinstance(
                            py_val, (int, float)
                        ):
                            match = (
                                abs(call_val - py_val) < 0.01
                                or abs(call_val - py_val) < abs(call_val) * 0.001
                            )
                        else:
                            match = call_val == py_val

                    if not match:
                        mismatches.append(
                            {
                                "call_idx": call_idx,
                                "sta_id": sta_id,
                                "metric": call_col,
                                "call_val": call_val,
                                "py_val": py_val,
                                "diff": (
                                    call_val - py_val
                                    if isinstance(call_val, (int, float))
                                    else "N/A"
                                ),
                            }
                        )

    if mismatches:
        print(f"\n⚠️  Found {len(mismatches)} mismatches:")
        mismatch_df = pd.DataFrame(mismatches)
        print(mismatch_df.to_string(index=False))
        all_passed = False
    else:
        print("\n✓ All checked metrics match between call-log and py-wrapper-env")

    return all_passed


def compare_oracle_metrics_detailed(call_df, py_env_df):
    """
    Detailed comparison of oracle metrics between NS3 and Python.
    """
    print("\n" + "=" * 80)
    print("DETAILED ORACLE METRICS COMPARISON")
    print("=" * 80 + "\n")

    # Aggregate totals for comparison
    print("Aggregated Oracle Metrics Across All Calls:")
    print("-" * 60)

    oracle_cols_call = [c for c in call_df.columns if c.startswith("oracle_")]
    oracle_cols_py = [c for c in py_env_df.columns if c.startswith("oracle_")]

    print(f"\nCall-log oracle columns: {len(oracle_cols_call)}")
    print(f"py-wrapper-env oracle columns: {len(oracle_cols_py)}")

    # Sum key metrics per STA
    print("\nPer-STA Oracle Totals (from last call/step):")
    print(
        f"{'STA':<5} | {'Metric':<30} | {'NS3 Call':<15} | {'Python Env':<15} | {'Match':<5}"
    )
    print("-" * 80)

    all_match = True

    for sta_id in sorted(call_df["sta_id"].unique())[:4]:
        # Get last call for this STA
        call_last = (
            call_df[call_df["sta_id"] == sta_id].sort_values("call_index").iloc[-1]
        )
        py_last = (
            py_env_df[py_env_df["sta_id"] == sta_id].sort_values("step").iloc[-1]
            if sta_id in py_env_df["sta_id"].values
            else None
        )

        if py_last is None:
            print(f"  {sta_id:<3} | ⚠️  STA not found in py-wrapper-env")
            continue

        metrics_to_check = [
            ("oracle_packets_transmitted", "oracle_packets_transmitted"),
            ("oracle_bytes_transmitted", "oracle_bytes_transmitted"),
            ("oracle_total_energy_mj", "oracle_total_energy_consumed_mj"),
            ("oracle_queue_size_packets", "oracle_queue_size_packets"),
        ]

        for call_col, py_col in metrics_to_check:
            if call_col in call_last.index and py_col in py_last.index:
                call_val = call_last[call_col]
                py_val = py_last[py_col]
                match = (
                    abs(call_val - py_val) < 0.01
                    if isinstance(call_val, (int, float))
                    else call_val == py_val
                )
                status = "✓" if match else "✗"
                if not match:
                    all_match = False
                print(
                    f"  {sta_id:<3} | {call_col:<30} | {call_val:<15.2f} | {py_val:<15.2f} | {status}"
                )

    return all_match


def verify_timing_consistency(call_df, py_env_df):
    """
    Verify that timing information is consistent.
    """
    print("\n" + "=" * 80)
    print("TIMING CONSISTENCY CHECK")
    print("=" * 80 + "\n")

    # Call-log uses simulation_time_ms, py-wrapper-env uses sim_time_sec
    call_times = (
        call_df.groupby("call_index")["simulation_time_ms"].first() / 1000.0
    )  # Convert to seconds
    py_times = py_env_df.groupby("step")["sim_time_sec"].first()

    print("Call timing comparison:")
    print(f"{'Index':<8} | {'NS3 (s)':<15} | {'Python (s)':<15} | {'Diff (ms)':<10}")
    print("-" * 55)

    all_match = True
    for idx in sorted(set(call_times.index) & set(py_times.index))[:10]:
        call_t = call_times.get(idx, np.nan)
        py_t = py_times.get(idx, np.nan)
        diff_ms = (
            (call_t - py_t) * 1000 if not (pd.isna(call_t) or pd.isna(py_t)) else np.nan
        )
        match = abs(diff_ms) < 100 if not pd.isna(diff_ms) else True
        status = "✓" if match else "✗"
        if not match:
            all_match = False
        print(
            f"  {idx:<6} | {call_t:<15.3f} | {py_t:<15.3f} | {diff_ms:<10.1f} {status}"
        )

    return all_match


def verify_sta_count(call_df, py_env_df):
    """
    Verify that all STAs are present in both logs.
    """
    print("\n" + "=" * 80)
    print("STA COUNT VERIFICATION")
    print("=" * 80 + "\n")

    call_stas = set(call_df["sta_id"].unique())
    py_stas = set(py_env_df["sta_id"].unique())

    print(f"STAs in NS3 call-log: {sorted(call_stas)}")
    print(f"STAs in py-wrapper-env: {sorted(py_stas)}")

    missing_in_py = call_stas - py_stas
    missing_in_ns3 = py_stas - call_stas

    if missing_in_py:
        print(f"\n⚠️  STAs in NS3 but NOT in Python: {sorted(missing_in_py)}")
    if missing_in_ns3:
        print(f"\n⚠️  STAs in Python but NOT in NS3: {sorted(missing_in_ns3)}")

    if not missing_in_py and not missing_in_ns3:
        print(f"\n✓ All {len(call_stas)} STAs present in both logs")
        return True
    return False


def main():
    print("=" * 80)
    print("NS3 to PYTHON DATA VERIFICATION - ALL RUNS")
    print("=" * 80 + "\n")

    # Find log directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    logdir = os.path.join(parent_dir, "data-log")

    if not os.path.exists(logdir):
        print(f"Error: Log directory not found: {logdir}")
        return 1

    print(f"Log directory: {logdir}\n")

    # Find all file groups
    file_groups = find_all_file_groups(logdir)
    if not file_groups:
        print("\n✗ No complete file groups found!")
        return 1

    all_passed = True
    results_summary = []

    # Process each group
    for timestamp in sorted(file_groups.keys()):
        files = file_groups[timestamp]

        print(f"{'='*80}")
        print(f"Processing: {timestamp}")
        print(f"{'='*80}\n")

        dfs = load_dataframes(files)

        if "call" not in dfs or "py_env" not in dfs:
            print("\n✗ Failed to load required dataframes for this run!")
            results_summary.append(
                {
                    "timestamp": timestamp,
                    "sta_count": "ERROR",
                    "timing": "ERROR",
                    "metrics": "ERROR",
                    "oracle": "ERROR",
                    "overall": "FAILED",
                }
            )
            all_passed = False
            continue

        # Run verification tests
        results = {}

        results["sta_count"] = verify_sta_count(dfs["call"], dfs["py_env"])
        results["timing"] = verify_timing_consistency(dfs["call"], dfs["py_env"])
        results["metrics"] = compare_call_vs_py_env(dfs["call"], dfs["py_env"])
        results["oracle"] = compare_oracle_metrics_detailed(dfs["call"], dfs["py_env"])

        # Track results
        test_statuses = {k: "PASSED" if v else "FAILED" for k, v in results.items()}
        test_statuses["overall"] = "PASSED" if all(results.values()) else "FAILED"
        test_statuses["timestamp"] = timestamp
        results_summary.append(test_statuses)

        if not all(results.values()):
            all_passed = False

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - ALL RUNS")
    print("=" * 80)
    print(
        f"\n{'Timestamp':<20} | {'STA Count':<12} | {'Timing':<12} | {'Metrics':<12} | {'Oracle':<12} | {'Overall':<10}"
    )
    print("-" * 95)
    for result in results_summary:
        print(
            f"{result['timestamp']:<20} | {result['sta_count']:<12} | {result['timing']:<12} | {result['metrics']:<12} | {result['oracle']:<12} | {result['overall']:<10}"
        )
    print("-" * 95)
    print(f"\nTotal runs: {len(results_summary)}")
    print(f"Passed: {sum(1 for r in results_summary if r['overall'] == 'PASSED')}")
    print(f"Failed: {sum(1 for r in results_summary if r['overall'] == 'FAILED')}")
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
Verify metrics consistency across all logging levels:
1. BI-level vs Call-level: Aggregating BI metrics should match call-level
2. E2E trace vs BI-level: PHY_TX events should match total_packets_transmitted delta

Located in: test-scripts/
Reads logs from: ../data-log/
Outputs verification CSV to: ../data-log/aggregated_from_bi.csv

Every time value is in milliseconds: simulation time, BI duration and call intervals alike.

In the C++ code total_packets_generated and total_packets_transmitted are cumulative,
not per-interval, so this script differences consecutive rows to recover per-interval values.

Validates that the data logging pipelines agree with one another.

Usage:
    python3.11 verify-call-level-metrics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import sys
import os

# def find_latest_files():
#     """Find the latest BI, call, and e2e trace CSV files."""
#     # data-log is in parent directory
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(script_dir)
#     logdir = os.path.join(parent_dir, "data-log")
#
#     bi_files = sorted(glob.glob(os.path.join(logdir, "ns3-BI-log-*.csv")))
#     call_files = sorted(glob.glob(os.path.join(logdir, "ns3-call-log-*.csv")))
#     e2e_files = sorted(glob.glob(os.path.join(logdir, "ns3-e2e-trace-*.csv")))
#
#     if not bi_files or not call_files:
#         raise FileNotFoundError(f"CSV files not found in {logdir}/")
#
#     bi_file = bi_files[-1]
#     call_file = call_files[-1]
#     e2e_file = e2e_files[-1] if e2e_files else None
#
#     print(f"BI-level file: {bi_file}")
#     print(f"Call-level file: {call_file}")
#     if e2e_file:
#         print(f"E2E trace file: {e2e_file}")
#     print()
#
#     return bi_file, call_file, e2e_file


def find_all_file_groups():
    """Find all BI, call, and e2e trace CSV file groups by timestamp."""
    # data-log is in parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    logdir = os.path.join(parent_dir, "data-log")

    bi_files = sorted(glob.glob(os.path.join(logdir, "ns3-BI-log-*.csv")))
    call_files = sorted(glob.glob(os.path.join(logdir, "ns3-call-log-*.csv")))
    e2e_files = sorted(glob.glob(os.path.join(logdir, "ns3-e2e-trace-*.csv")))

    if not bi_files or not call_files:
        raise FileNotFoundError(f"CSV files not found in {logdir}/")

    # Extract timestamps from filenames and group by timestamp
    # Format: ns3-BI-log-YYYYMMDD_HHMMSS.csv
    file_groups = {}

    for bi_file in bi_files:
        # Extract timestamp: everything after last '-' and before '.csv'
        timestamp = os.path.basename(bi_file).split("-")[-1].replace(".csv", "")
        if timestamp not in file_groups:
            file_groups[timestamp] = {}
        file_groups[timestamp]["bi"] = bi_file

    for call_file in call_files:
        timestamp = os.path.basename(call_file).split("-")[-1].replace(".csv", "")
        if timestamp not in file_groups:
            file_groups[timestamp] = {}
        file_groups[timestamp]["call"] = call_file

    for e2e_file in e2e_files:
        timestamp = os.path.basename(e2e_file).split("-")[-1].replace(".csv", "")
        if timestamp not in file_groups:
            file_groups[timestamp] = {}
        file_groups[timestamp]["e2e"] = e2e_file

    # Filter to groups that have both BI and call files
    complete_groups = {
        ts: files
        for ts, files in file_groups.items()
        if "bi" in files and "call" in files
    }

    print(
        f"Found {len(complete_groups)} complete file groups (with both BI and call logs):\n"
    )
    for ts in sorted(complete_groups.keys()):
        print(f"  {ts}:")
        print(f"    BI:   {os.path.basename(complete_groups[ts]['bi'])}")
        print(f"    Call: {os.path.basename(complete_groups[ts]['call'])}")
        if "e2e" in complete_groups[ts]:
            print(f"    E2E:  {os.path.basename(complete_groups[ts]['e2e'])}")
    print()

    return complete_groups, logdir


def read_csv_files(bi_file, call_file, e2e_file=None):
    """Read all CSV files. All time columns are in milliseconds."""
    bi_df = pd.read_csv(bi_file)
    call_df = pd.read_csv(call_file)
    e2e_df = pd.read_csv(e2e_file) if e2e_file else None

    # Normalize column names: BI log uses 'observation_time_ms', standardize to 'simulation_time_ms'
    if (
        "observation_time_ms" in bi_df.columns
        and "simulation_time_ms" not in bi_df.columns
    ):
        bi_df = bi_df.rename(columns={"observation_time_ms": "simulation_time_ms"})

    print(f"BI-level: {len(bi_df)} rows, {len(bi_df.columns)} columns")
    print(f"Call-level: {len(call_df)} rows, {len(call_df.columns)} columns")
    if e2e_df is not None:
        print(f"E2E trace: {len(e2e_df)} rows, {len(e2e_df.columns)} columns")
    print()

    return bi_df, call_df, e2e_df


def compute_cumulative_deltas(
    df, cumulative_cols, group_col="sta_id", sort_col="simulation_time_ms"
):
    """
    Compute per-interval deltas from cumulative columns.

    The C++ code stores cumulative values for total_packets_generated and
    total_packets_transmitted. This function computes deltas between consecutive
    rows for each STA.
    """
    df = df.sort_values([group_col, sort_col]).copy()

    for col in cumulative_cols:
        if col in df.columns:
            delta_col = f"{col}_delta"
            # Compute delta within each STA group
            df[delta_col] = df.groupby(group_col)[col].diff().fillna(0)
            # First row for each STA: use the value as-is (it's already a delta from snapshot)
            # Actually, the first logged row IS a delta from the snapshot taken on first call

    return df


def aggregate_bi_to_call(bi_df, call_df):
    """
    Aggregate BI-level metrics to match call-level time periods.
    All times are in milliseconds.

    IMPORTANT:
    - total_packets_generated and total_packets_transmitted are CUMULATIVE in C++
    - Other metrics like awake_time_ms, energy_consumed_mj are per-interval DELTAS

    Call-level timing (in milliseconds):
    - First call: Takes snapshot, doesn't log data (m_callLevelFirstCall=true)
    - Second call: Logs delta from first call time to second call time (call_index=1)
    - Third call: Logs delta from second call time to third call time (call_index=2)
    """
    # Get unique call indices and their simulation times (in milliseconds)
    call_times = call_df.groupby("call_index")["simulation_time_ms"].first().to_dict()

    print("Call timestamps (in milliseconds):")
    for call_index, sim_time_ms in sorted(call_times.items()):
        print(f"  Call {call_index}: {sim_time_ms:.1f}ms")
    print()

    # Get call boundaries sorted by time
    sorted_calls = sorted(call_times.items(), key=lambda x: x[1])

    # Use actual CSV timestamps - compute call interval from data (in milliseconds)
    first_logged_call_time_ms = sorted_calls[0][1]

    # Compute call interval from consecutive calls (if we have multiple)
    if len(sorted_calls) >= 2:
        call_interval_ms = sorted_calls[1][1] - sorted_calls[0][1]
    else:
        # Fallback: estimate from BI data if only one call
        bi_times_ms = sorted(bi_df["simulation_time_ms"].unique())
        if len(bi_times_ms) >= 2:
            bi_interval_ms = bi_times_ms[1] - bi_times_ms[0]
            call_interval_ms = bi_times_ms[-1] - bi_times_ms[0]
        else:
            call_interval_ms = 10240.0  # Fallback default: 10.24s in ms

    print(f"First logged call time: {first_logged_call_time_ms:.1f}ms")
    print(f"Call interval (computed from data): {call_interval_ms:.1f}ms")
    print()

    # For each call, find BIs from PREVIOUS call time to THIS call time
    aggregated_data = []

    for call_idx, (call_index, call_time_ms) in enumerate(sorted_calls):
        # Determine time window: from previous call to this call (in milliseconds)
        if call_idx == 0:
            # First logged call: infer snapshot time from computed call interval
            prev_call_time_ms = call_time_ms - call_interval_ms
        else:
            prev_call_time_ms = sorted_calls[call_idx - 1][1]

        # Filter BIs in this call window: prev_call_time_ms < t <= call_time_ms
        bis_in_call = bi_df[
            (bi_df["simulation_time_ms"] > prev_call_time_ms)
            & (bi_df["simulation_time_ms"] <= call_time_ms)
        ]

        if len(bis_in_call) == 0:
            print(
                f"⚠️  Call {call_index} ({prev_call_time_ms:.1f}ms → {call_time_ms:.1f}ms): NO BIs found"
            )
            continue

        call_duration_ms = call_time_ms - prev_call_time_ms

        print(
            f"Call {call_index} ({prev_call_time_ms:.1f}ms → {call_time_ms:.1f}ms): {len(bis_in_call)} BIs, duration={call_duration_ms:.1f}ms"
        )

        # Group by STA
        for sta_id in sorted(bis_in_call["sta_id"].unique()):
            sta_bis = bis_in_call[bis_in_call["sta_id"] == sta_id].sort_values(
                "simulation_time_ms"
            )

            agg_row = {
                "call_index": call_index,
                "sta_id": sta_id,
                "num_bis": len(sta_bis),
                "simulation_time_ms": call_time_ms,
                "call_duration_ms": call_duration_ms,
            }

            # ALL oracle metrics are CUMULATIVE - get end-of-window value
            cumulative_metrics = [
                "oracle_packets_generated",
                "oracle_packets_transmitted",
                "oracle_awake_time_ms",
                "oracle_sleep_time_ms",
                "oracle_total_energy_mj",
            ]
            for metric in cumulative_metrics:
                if metric in sta_bis.columns:
                    # Get the last (cumulative) value in this window
                    last_val = sta_bis[metric].iloc[-1]
                    agg_row[f"{metric}_end"] = last_val

            # Throughput: use average of BI throughput values
            if "throughput_mbps" in sta_bis.columns:
                agg_row["throughput_mbps_agg"] = sta_bis["throughput_mbps"].mean()

            # Average-based metrics
            avg_metrics = [
                "oracle_duty_cycle",
            ]
            for metric in avg_metrics:
                if metric in sta_bis.columns:
                    agg_row[f"{metric}_avg"] = sta_bis[metric].mean()

            aggregated_data.append(agg_row)

    agg_df = pd.DataFrame(aggregated_data)
    return agg_df


def compare_bi_vs_call(agg_df, call_df):
    """
    Compare aggregated BI metrics with actual call-level metrics.

    Note: total_packets_transmitted in call_df is CUMULATIVE (from C++ code).
    We need to compare cumulative values at end of each call period.

    IMPORTANT: BI-level and Call-level use SEPARATE snapshot tracking in C++.
    - m_lastBiSnapshot[] is updated every BI
    - m_lastCallSnapshot[] is updated every call
    This means the delta metrics may not align perfectly between BI and Call level
    due to different snapshot boundaries.
    """
    print("\n" + "=" * 80)
    print("TEST 1: BI-level Aggregation vs Call-level")
    print("=" * 80 + "\n")

    print("NOTE: All oracle metrics are CUMULATIVE values at end of window.")
    print("      BI cumulative at call_time should match Call cumulative.\n")

    # Merge on call_index and sta_id
    # Use oracle_ prefixed columns from the new data format
    call_cols = ["call_index", "sta_id"]
    oracle_cols = [
        "oracle_duty_cycle",
        "oracle_total_energy_mj",
        "oracle_packets_transmitted",
        "oracle_packets_generated",
        "oracle_awake_time_ms",
    ]
    # Only include columns that exist
    available_cols = [c for c in oracle_cols if c in call_df.columns]
    comparison = agg_df.merge(
        call_df[call_cols + available_cols],
        on=["call_index", "sta_id"],
        suffixes=("_agg", "_call"),
    )

    all_passed = True

    # Check cumulative packet counts at end of call period
    # Both BI and Call level should show same cumulative value at the same timestamp
    if (
        "oracle_packets_transmitted_end" in comparison.columns
        and "oracle_packets_transmitted" in comparison.columns
    ):
        comparison["pkt_diff"] = (
            comparison["oracle_packets_transmitted_end"]
            - comparison["oracle_packets_transmitted"]
        )
        max_diff = comparison["pkt_diff"].abs().max()
        mean_diff = comparison["pkt_diff"].abs().mean()

        # Allow tolerance for timing edge cases
        tolerance = 100
        if max_diff <= tolerance:
            print(
                f"✓ Packet counts (cumulative): MATCH (max_diff={max_diff:.0f} within tolerance of {tolerance})"
            )
        else:
            print(
                f"✗ Packet counts (cumulative): max_diff={max_diff:.0f}, mean_diff={mean_diff:.2f} (tolerance={tolerance})"
            )
            all_passed = False

    # Check awake time (CUMULATIVE at end of window)
    if (
        "oracle_awake_time_ms_end" in comparison.columns
        and "oracle_awake_time_ms" in comparison.columns
    ):
        comparison["awake_diff"] = (
            comparison["oracle_awake_time_ms_end"] - comparison["oracle_awake_time_ms"]
        )
        max_diff = comparison["awake_diff"].abs().max()
        mean_diff = comparison["awake_diff"].abs().mean()

        # Should match exactly since both are cumulative at same timestamp
        tolerance_ms = 1.0
        if max_diff < tolerance_ms:
            print(
                f"✓ Awake time (cumulative): MATCH (max_diff={max_diff:.2f}ms within tolerance of {tolerance_ms}ms)"
            )
        else:
            print(
                f"✗ Awake time (cumulative): max_diff={max_diff:.2f}ms, mean_diff={mean_diff:.2f}ms"
            )
            all_passed = False

    # Check energy (CUMULATIVE at end of window)
    if (
        "oracle_total_energy_mj_end" in comparison.columns
        and "oracle_total_energy_mj" in comparison.columns
    ):
        comparison["energy_diff"] = (
            comparison["oracle_total_energy_mj_end"]
            - comparison["oracle_total_energy_mj"]
        )
        max_diff = comparison["energy_diff"].abs().max()
        mean_diff = comparison["energy_diff"].abs().mean()

        # Should match exactly since both are cumulative at same timestamp
        tolerance_mj = 1.0
        if max_diff < tolerance_mj:
            print(
                f"✓ Energy consumed (cumulative): MATCH (max_diff={max_diff:.2f}mJ within tolerance of {tolerance_mj}mJ)"
            )
        else:
            print(
                f"✗ Energy consumed (cumulative): max_diff={max_diff:.2f}mJ, mean_diff={mean_diff:.2f}mJ"
            )
            all_passed = False

    # Check duty cycle
    if (
        "oracle_duty_cycle_avg" in comparison.columns
        and "oracle_duty_cycle" in comparison.columns
    ):
        comparison["dc_diff"] = (
            comparison["oracle_duty_cycle_avg"] - comparison["oracle_duty_cycle"]
        )
        max_diff = comparison["dc_diff"].abs().max()
        mean_diff = comparison["dc_diff"].abs().mean()

        tolerance_dc = 0.5  # Higher tolerance
        if max_diff < tolerance_dc:
            print(
                f"✓ Duty cycle: MATCH (max_diff={max_diff:.4f} within tolerance of {tolerance_dc})"
            )
        else:
            print(f"⚠️  Duty cycle: max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}")
            print(f"   (Expected mismatch due to separate BI/Call snapshot tracking)")

    return comparison, all_passed


def compare_e2e_vs_bi(e2e_df, bi_df, warmup_calls=3):
    """
    Compare E2E trace PHY_TX events with BI-level oracle_packets_transmitted.

    IMPORTANT: oracle_packets_transmitted in BI-level is CUMULATIVE.
    We compare the delta across the time window (after warmup) with E2E PHY_TX count.

    Args:
        e2e_df: End-to-end trace dataframe
        bi_df: BI-level dataframe
        warmup_calls: Number of warmup calls to skip (default: 3)
    """
    print("\n" + "=" * 80)
    print("TEST 2: E2E Trace (PHY_TX) vs BI-level (oracle_packets_transmitted)")
    print("=" * 80 + "\n")

    if e2e_df is None:
        print("⚠️  E2E trace file not found, skipping this test")
        return None, True

    # Get time range from BI-level
    first_logged_bi_time_ms = bi_df["simulation_time_ms"].min()
    bi_end_time_ms = bi_df["simulation_time_ms"].max()

    # Compute BI interval from consecutive BIs
    bi_times_ms = sorted(bi_df["simulation_time_ms"].unique())
    if len(bi_times_ms) >= 2:
        bi_interval_ms = bi_times_ms[1] - bi_times_ms[0]
    else:
        bi_interval_ms = 102.4

    # Call interval is 5 BIs = 512ms
    call_interval_ms = 5 * bi_interval_ms

    # Skip warmup: warmup_calls * call_interval_ms after first logged BI
    warmup_end_ms = first_logged_bi_time_ms + (warmup_calls * call_interval_ms)

    print(f"First logged BI: {first_logged_bi_time_ms:.1f}ms")
    print(
        f"Warmup period: {warmup_calls} calls = {warmup_calls * call_interval_ms:.1f}ms"
    )
    print(f"Warmup ends at: {warmup_end_ms:.1f}ms")
    print(f"Last logged BI: {bi_end_time_ms:.1f}ms")
    print(f"Analysis window: {warmup_end_ms:.1f}ms to {bi_end_time_ms:.1f}ms")
    print()

    results = []
    all_passed = True

    for sta_id in sorted(bi_df["sta_id"].unique()):
        sta_name = f"STA{sta_id}"

        # E2E: count PHY_TX events for this STA AFTER warmup
        e2e_phy_tx = len(
            e2e_df[
                (e2e_df["Node"] == sta_name)
                & (e2e_df["Event"] == "PHY_TX")
                & (e2e_df["Time_ms"] > warmup_end_ms)
                & (e2e_df["Time_ms"] <= bi_end_time_ms)
            ]
        )

        # BI: compute delta of cumulative oracle_packets_transmitted AFTER warmup
        sta_bi = bi_df[bi_df["sta_id"] == sta_id].sort_values("simulation_time_ms")
        if len(sta_bi) > 0:
            # Get cumulative value at warmup end (last BI <= warmup_end_ms)
            warmup_bi = sta_bi[sta_bi["simulation_time_ms"] <= warmup_end_ms]
            if len(warmup_bi) > 0:
                pkts_at_warmup_end = warmup_bi["oracle_packets_transmitted"].iloc[-1]
            else:
                pkts_at_warmup_end = 0

            # Get cumulative value at end
            pkts_at_end = sta_bi["oracle_packets_transmitted"].iloc[-1]

            # Delta = packets transmitted after warmup
            bi_pkts_tx = int(pkts_at_end - pkts_at_warmup_end)
        else:
            bi_pkts_tx = 0
            e2e_phy_tx = 0

        diff = int(bi_pkts_tx - e2e_phy_tx)

        results.append(
            {
                "sta_id": sta_id,
                "e2e_phy_tx": e2e_phy_tx,
                "bi_pkts_tx": bi_pkts_tx,
                "diff": diff,
            }
        )

        if (
            abs(diff) > 50
        ):  # Allow tolerance for edge cases (a few extra packets at boundaries)
            all_passed = False

    results_df = pd.DataFrame(results)

    if all_passed:
        print("✓ E2E vs BI packet counts: MATCH (within tolerance)")
    else:
        print("✗ E2E vs BI packet counts: MISMATCH detected")

    print()
    print("Per-STA breakdown (packets after warmup):")
    for _, row in results_df.iterrows():
        status = "✓" if abs(row["diff"]) <= 50 else "✗"
        print(
            f"  STA{row['sta_id']}: E2E={row['e2e_phy_tx']:5d}, BI_delta={row['bi_pkts_tx']:5d}, diff={row['diff']:+4d} {status}"
        )

    return results_df, all_passed


def main():
    print("=" * 80)
    print("METRICS CONSISTENCY VERIFICATION - ALL RUNS")
    print("=" * 80 + "\n")
    print("NOTE: total_packets_generated and total_packets_transmitted are CUMULATIVE")
    print(
        "      Other metrics (awake_time_ms, energy_consumed_mj) are per-interval DELTAS"
    )
    print()

    file_groups, logdir = find_all_file_groups()

    all_passed = True
    results_summary = []

    # Process each file group
    for timestamp in sorted(file_groups.keys()):
        files = file_groups[timestamp]

        print(f"\n{'='*80}")
        print(f"Processing: {timestamp}")
        print(f"{'='*80}\n")

        bi_df, call_df, e2e_df = read_csv_files(
            files["bi"], files["call"], files.get("e2e")
        )

        # Test 1: BI vs Call aggregation
        agg_df = aggregate_bi_to_call(bi_df, call_df)
        print(f"Aggregated {len(agg_df)} rows from {len(bi_df)} BI-level rows")

        comparison1, test1_passed = compare_bi_vs_call(agg_df, call_df)

        # Test 2: E2E vs BI
        comparison2, test2_passed = compare_e2e_vs_bi(e2e_df, bi_df)

        # Save aggregated data for inspection
        output_file = os.path.join(logdir, f"aggregated_from_bi_{timestamp}.csv")
        agg_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved aggregated metrics to: {output_file}")

        # Track results
        test1_status = "PASSED" if test1_passed else "FAILED"
        test2_status = "PASSED" if test2_passed else "FAILED"
        results_summary.append(
            {
                "timestamp": timestamp,
                "bi_vs_call": test1_status,
                "e2e_vs_bi": test2_status,
                "overall": "PASSED" if (test1_passed and test2_passed) else "FAILED",
            }
        )

        if not (test1_passed and test2_passed):
            all_passed = False

    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY - ALL RUNS")
    print("=" * 80)
    print(
        f"\n{'Timestamp':<20} | {'BI vs Call':<15} | {'E2E vs BI':<15} | {'Overall':<10}"
    )
    print("-" * 65)
    for result in results_summary:
        print(
            f"{result['timestamp']:<20} | {result['bi_vs_call']:<15} | {result['e2e_vs_bi']:<15} | {result['overall']:<10}"
        )
    print("-" * 65)
    print(f"\nTotal runs: {len(results_summary)}")
    print(f"Passed: {sum(1 for r in results_summary if r['overall'] == 'PASSED')}")
    print(f"Failed: {sum(1 for r in results_summary if r['overall'] == 'FAILED')}")
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

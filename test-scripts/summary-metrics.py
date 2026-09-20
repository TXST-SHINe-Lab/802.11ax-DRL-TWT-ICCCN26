#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
Comprehensive metrics summary from all TWT simulation logs

Reads all available log files from data-log/ and generates a complete
performance report including:
- BI-level statistics (PHY metrics, energy, packets)
- Call-level analysis (wireless call tracking)
- Link measurements (RSSI, SNR, margins)
- MAC queue statistics
- BSR (Buffer Status Report) tracking
- AMPDU aggregation stats
- End-to-end performance
- TX/RX statistics
- Physical layer state information
- Timeouts and drops

Reads the newest matching file for each log pattern, so it reports on the most recent run.

Usage:
    python3.11 summary-metrics.py
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict


def find_latest_file(logdir, pattern):
    """Find the most recent file matching pattern"""
    files = sorted(glob.glob(os.path.join(logdir, pattern)))
    return files[-1] if files else None


def load_all_logs(logdir):
    """Load all available log files"""
    logs = {}

    patterns = {
        "bi": "ns3-BI-log-*.csv",
        "call": "ns3-call-log-*.csv",
        "link": "ns3-link-measurement-trace-*.csv",
        "ampdu": "ns3-ampdu-trace-*.csv",
        "bsr": "ns3-bsr-trace-*.csv",
        "macq": "ns3-macqueuesize-trace-*.csv",
        "e2e": "ns3-e2e-trace-*.csv",
        "phystate": "ns3-phystate-trace-*.csv",
        "timeout": "ns3-timeout-drop-trace-*.csv",
        "txrx": "ns3-txrx-stats-trace-*.csv",
        "wrapper": "ns3-twt-wrapper-*.csv",
        "py_env": "py-wrapper-env-*.csv",
        "py_action": "py-wrapper-action-*.csv",
    }

    for key, pattern in patterns.items():
        filepath = find_latest_file(logdir, pattern)
        if filepath and os.path.exists(filepath):
            try:
                logs[key] = pd.read_csv(filepath)
                print(
                    f"✓ Loaded {key:12} ({len(logs[key]):6} rows) - {os.path.basename(filepath)}"
                )
            except Exception as e:
                print(f"✗ Failed to load {key:12}: {e}")

    return logs


def report_bi_metrics(logs, logdir):
    """Generate BI-level metrics report"""
    if "bi" not in logs:
        print("\n⚠ No BI-level log available")
        return

    df = logs["bi"]
    print(f"\n{'='*80}")
    print("BI-LEVEL METRICS SUMMARY")
    print(f"{'='*80}")

    print(f"\nDataset Coverage:")
    print(f"  Rows: {len(df)}")
    print(
        f"  BIs: {df['bi_index'].nunique()} (Range: {df['bi_index'].min():.0f} - {df['bi_index'].max():.0f})"
    )
    print(f"  STAs: {df['sta_id'].nunique()}")
    print(
        f"  Time: {df['observation_time_ms'].min()/1000:.2f}s - {df['observation_time_ms'].max()/1000:.2f}s ({(df['observation_time_ms'].max()-df['observation_time_ms'].min())/1000:.2f}s)"
    )

    # Per-STA statistics
    print(f"\n{'Per-STA PHY Metrics':^80}")
    print(
        f"{'STA':>4} | {'Avg RSSI':>9} | {'Avg SNR':>8} | {'Avg Link':>9} | {'RX Pkts':>10} | {'FCS Err':>8} | {'Energy':>10}"
    )
    print(
        f"{'':>4} | {'(dBm)':>9} | {'(dB)':>8} | {'Margin':>9} | {'':>10} | {'':>8} | {'(mJ)':>10}"
    )
    print("-" * 80)

    for sta_id in sorted(df["sta_id"].unique()):
        sta_df = df[df["sta_id"] == sta_id]

        avg_rssi = sta_df["rssi_dbm"].replace(-128, np.nan).mean()
        avg_snr = sta_df["snr_db"].replace(-128, np.nan).mean()
        avg_link = sta_df["link_margin_db"].replace(-128, np.nan).mean()
        total_rx_pkts = sta_df["packets_received_at_ap"].sum()
        total_fcs_err = sta_df["fcs_error_count"].sum()
        total_energy = (
            sta_df["oracle_total_energy_mj"].max()
            if "oracle_total_energy_mj" in sta_df.columns
            else 0
        )

        rssi_str = f"{avg_rssi:.1f}" if not np.isnan(avg_rssi) else "N/A"
        snr_str = f"{avg_snr:.1f}" if not np.isnan(avg_snr) else "N/A"
        link_str = f"{avg_link:.1f}" if not np.isnan(avg_link) else "N/A"

        print(
            f"  {sta_id:>2} | {rssi_str:>9} | {snr_str:>8} | {link_str:>9} | {total_rx_pkts:>10.0f} | {total_fcs_err:>8.0f} | {total_energy:>10.2f}"
        )

    # Network aggregates
    print("-" * 80)
    total_rx_pkts = df["packets_received_at_ap"].sum()
    total_fcs_err = df["fcs_error_count"].sum()
    total_energy = (
        df["oracle_total_energy_mj"].sum()
        if "oracle_total_energy_mj" in df.columns
        else 0
    )

    print(
        f"  {'AGG':>2} | {'':>9} | {'':>8} | {'':>9} | {total_rx_pkts:>10.0f} | {total_fcs_err:>8.0f} | {total_energy:>10.2f}"
    )

    # TX/RX statistics
    if "tx_fragment_count" in df.columns or "tx_retry_count" in df.columns:
        print(f"\n{'Packet Transmission Statistics':^80}")
        print(
            f"{'STA':>4} | {'TX Frag':>10} | {'TX Retry':>10} | {'TX Failed':>10} | {'ACK Fail':>10} | {'RX Frag':>10}"
        )
        print("-" * 80)

        for sta_id in sorted(df["sta_id"].unique()):
            sta_df = df[df["sta_id"] == sta_id]
            tx_frag = (
                sta_df["tx_fragment_count"].sum()
                if "tx_fragment_count" in sta_df.columns
                else 0
            )
            tx_retry = (
                sta_df["tx_retry_count"].sum()
                if "tx_retry_count" in sta_df.columns
                else 0
            )
            tx_failed = (
                sta_df["tx_failed_count"].sum()
                if "tx_failed_count" in sta_df.columns
                else 0
            )
            ack_fail = (
                sta_df["ack_failure_count"].sum()
                if "ack_failure_count" in sta_df.columns
                else 0
            )
            rx_frag = (
                sta_df["rx_fragment_count"].sum()
                if "rx_fragment_count" in sta_df.columns
                else 0
            )

            print(
                f"  {sta_id:>2} | {tx_frag:>10.0f} | {tx_retry:>10.0f} | {tx_failed:>10.0f} | {ack_fail:>10.0f} | {rx_frag:>10.0f}"
            )

        print("-" * 80)
        print(
            f"  {'TOT':>2} | {df['tx_fragment_count'].sum():>10.0f} | {df['tx_retry_count'].sum():>10.0f} | {df['tx_failed_count'].sum():>10.0f} | {df['ack_failure_count'].sum():>10.0f} | {df['rx_fragment_count'].sum():>10.0f}"
        )

    # BSR statistics
    if "bsr_ac_be" in df.columns:
        print(f"\n{'Buffer Status Report (BSR) Summary':^80}")
        bsr_cols = ["bsr_ac_be", "bsr_ac_bk", "bsr_ac_vi", "bsr_ac_vo"]
        ac_names = [
            "AC_BE (Best Effort)",
            "AC_BK (Background)",
            "AC_VI (Video)",
            "AC_VO (Voice)",
        ]

        for col, name in zip(bsr_cols, ac_names):
            if col in df.columns:
                total_bsr = df[col].sum()
                avg_bsr = df[col].mean()
                max_bsr = df[col].max()
                print(
                    f"  {name:25} - Total: {total_bsr:12.0f}, Avg: {avg_bsr:10.2f}, Max: {max_bsr:10.0f}"
                )


def report_call_metrics(logs):
    """Generate call-level metrics report"""
    if "call" not in logs:
        print("\n⚠ No call-level log available")
        return

    df = logs["call"]
    print(f"\n{'='*80}")
    print("CALL-LEVEL METRICS SUMMARY")
    print(f"{'='*80}")

    print(f"\nDataset Coverage:")
    print(f"  Total calls: {df['call_index'].nunique()}")
    print(f"  Call rows: {len(df)}")
    print(
        f"  Time range: {df['simulation_time_ms'].min()/1000:.2f}s - {df['simulation_time_ms'].max()/1000:.2f}s"
    )

    print(f"\n{'Per-Call Summary (Sample of first 10 calls)':^80}")
    print(
        f"{'Call':>4} | {'Start':>10} | {'Duration':>10} | {'Avg RSSI':>10} | {'Avg SNR':>9} | {'RX Pkts':>10}"
    )
    print(
        f"{'':>4} | {'(s)':>10} | {'(ms)':>10} | {'(dBm)':>10} | {'(dB)':>9} | {'':>10}"
    )
    print("-" * 80)

    for call_idx in sorted(df["call_index"].unique())[:10]:
        call_data = df[df["call_index"] == call_idx]
        start_time = call_data["simulation_time_ms"].min() / 1000.0
        duration = (
            call_data["simulation_time_ms"].max()
            - call_data["simulation_time_ms"].min()
        )
        avg_rssi = call_data["rssi_dbm"].replace(-128, np.nan).mean()
        avg_snr = call_data["snr_db"].replace(-128, np.nan).mean()
        rx_pkts = call_data["packets_received_at_ap"].sum()

        rssi_str = f"{avg_rssi:.1f}" if not np.isnan(avg_rssi) else "N/A"
        snr_str = f"{avg_snr:.1f}" if not np.isnan(avg_snr) else "N/A"

        print(
            f"  {call_idx:>2} | {start_time:>10.2f} | {duration:>10.0f} | {rssi_str:>10} | {snr_str:>9} | {rx_pkts:>10.0f}"
        )


def report_link_measurements(logs):
    """Generate link measurement statistics"""
    if "link" not in logs:
        return

    df = logs["link"]
    print(f"\n{'='*80}")
    print("LINK MEASUREMENT TRACE SUMMARY")
    print(f"{'='*80}")

    # Check what columns are available
    if "Node" in df.columns:
        print(f"\n{'Link Quality Statistics (by Node)':^80}")
        print(f"{'Node':>10} | {'Avg RSSI':>10} | {'Avg SNR':>10} | {'Avg Rate':>12}")
        print("-" * 80)

        for node in sorted(df["Node"].unique())[:16]:
            node_df = df[df["Node"] == node]
            avg_rssi = (
                node_df["RSSI_dBm"].mean() if "RSSI_dBm" in node_df.columns else 0
            )
            avg_snr = node_df["SNR_dB"].mean() if "SNR_dB" in node_df.columns else 0
            avg_rate = (
                node_df["Rate_100kbps"].mean()
                if "Rate_100kbps" in node_df.columns
                else 0
            )
            print(
                f"  {str(node):>10} | {avg_rssi:>10.1f} | {avg_snr:>10.1f} | {avg_rate:>12.0f}"
            )
    elif "sta_id" in df.columns:
        print(f"\n{'RCPI/RSNI Statistics (by STA)':^80}")
        print(
            f"{'STA':>4} | {'Avg RCPI':>10} | {'Avg RSNI':>10} | {'Max RCPI':>10} | {'Min RCPI':>10}"
        )
        print("-" * 80)

        for sta_id in sorted(df["sta_id"].unique())[:16]:
            sta_df = df[df["sta_id"] == sta_id]
            avg_rcpi = sta_df["rcpi"].mean() if "rcpi" in sta_df.columns else 0
            avg_rsni = sta_df["rsni"].mean() if "rsni" in sta_df.columns else 0
            max_rcpi = sta_df["rcpi"].max() if "rcpi" in sta_df.columns else 0
            min_rcpi = sta_df["rcpi"].min() if "rcpi" in sta_df.columns else 0
            print(
                f"  {sta_id:>2} | {avg_rcpi:>10.1f} | {avg_rsni:>10.1f} | {max_rcpi:>10.1f} | {min_rcpi:>10.1f}"
            )


def report_ampdu_stats(logs):
    """Generate AMPDU aggregation statistics"""
    if "ampdu" not in logs:
        return

    df = logs["ampdu"]
    print(f"\n{'='*80}")
    print("AMPDU AGGREGATION STATISTICS")
    print(f"{'='*80}")

    if len(df) > 0:
        print(f"\n  Total AMPDU events: {len(df)}")
        # Check for mpdu_count column
        mpdu_col = (
            "mpdu_count"
            if "mpdu_count" in df.columns
            else ("MPDU_Count" if "MPDU_Count" in df.columns else None)
        )
        if mpdu_col:
            print(f"  Avg MPDUs per AMPDU: {df[mpdu_col].mean():.2f}")
            print(f"  Max MPDUs in AMPDU: {df[mpdu_col].max():.0f}")
            print(f"  Total MPDUs: {df[mpdu_col].sum():.0f}")


def report_mac_queue(logs):
    """Generate MAC queue statistics"""
    if "macq" not in logs:
        return

    df = logs["macq"]
    print(f"\n{'='*80}")
    print("MAC QUEUE SIZE STATISTICS")
    print(f"{'='*80}")

    if "queue_size_packets" in df.columns:
        print(f"\n  Average queue size: {df['queue_size_packets'].mean():.1f} packets")
        print(f"  Max queue size: {df['queue_size_packets'].max():.0f} packets")
        print(f"  Min queue size: {df['queue_size_packets'].min():.0f} packets")

    if "queue_size_bytes" in df.columns:
        print(f"  Average queue size: {df['queue_size_bytes'].mean():.0f} bytes")
        print(f"  Max queue size: {df['queue_size_bytes'].max():.0f} bytes")


def report_timeout_drops(logs):
    """Generate timeout and drop statistics"""
    if "timeout" not in logs:
        return

    df = logs["timeout"]
    print(f"\n{'='*80}")
    print("TIMEOUT AND DROP STATISTICS")
    print(f"{'='*80}")

    if len(df) > 0:
        print(f"\n  Total events: {len(df)}")
        if "sta_id" in df.columns:
            drops_by_sta = df["sta_id"].value_counts().sort_index()
            print(f"\n  Drops by STA:")
            for sta_id, count in drops_by_sta.head(16).items():
                print(f"    STA {sta_id:>2}: {count:>6} events")


def report_python_controller(logs):
    """Generate Python controller statistics"""
    if "py_action" not in logs or "py_env" not in logs:
        return

    action_df = logs["py_action"]
    env_df = logs["py_env"]

    print(f"\n{'='*80}")
    print("PYTHON CONTROLLER INTERACTION SUMMARY")
    print(f"{'='*80}")

    print(f"\n  Action updates: {len(action_df)}")
    if "Timestamp_Sec" in action_df.columns:
        print(
            f"  Action time range: {action_df['Timestamp_Sec'].min():.2f}s - {action_df['Timestamp_Sec'].max():.2f}s"
        )

    print(f"\n  Environment observations: {len(env_df)}")
    if "Timestamp_Sec" in env_df.columns:
        print(
            f"  Observation time range: {env_df['Timestamp_Sec'].min():.2f}s - {env_df['Timestamp_Sec'].max():.2f}s"
        )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logdir = os.path.join(os.path.dirname(script_dir), "data-log")

    if not os.path.exists(logdir):
        print(f"Error: Log directory not found: {logdir}")
        return

    print(f"\n{'='*80}")
    print("COMPREHENSIVE TWT SIMULATION REPORT".center(80))
    print(f"{'='*80}\n")

    print(f"Loading logs from: {os.path.abspath(logdir)}\n")
    logs = load_all_logs(logdir)

    if not logs:
        print("\n✗ No log files found!")
        return

    print(f"\n✓ Successfully loaded {len(logs)} log files\n")

    # Generate all reports
    report_bi_metrics(logs, logdir)
    report_call_metrics(logs)
    report_link_measurements(logs)
    report_ampdu_stats(logs)
    report_mac_queue(logs)
    report_timeout_drops(logs)
    report_python_controller(logs)

    print(f"\n{'='*80}")
    print("END OF REPORT".center(80))
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

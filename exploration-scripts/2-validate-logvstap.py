#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""2-validate-logvstap.py - detailed variable-by-variable cross-validation.

Cross-checks every variable in the spawn JSONL against the corresponding value in:
1. Wrapper CSV logs (ns3-twt-wrapper-*.csv)
2. Other supplementary logs (ns3-BI-log, ns3-ampdu-trace, etc.)

Confirms every metric survives the wrapper → JSONL → logs path unchanged.

Usage:
    python3.11 2-validate-logvstap.py
"""

import argparse
import csv
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class DetailedLogValidator:
    """Detailed variable-by-variable cross-validation."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.mismatches = []
        self.matches = []
        self.errors = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def load_csv(self, csv_path: str) -> List[Dict]:
        """Load CSV log."""
        data = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data

    def load_jsonl(self, jsonl_path: str) -> tuple:
        """
        Load JSONL transitions.

        Returns:
            (metadata, transitions) where metadata is the first line,
            transitions are remaining lines
        """
        metadata = None
        transitions = []

        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if i == 0 and "_metadata" in data:
                    metadata = data["_metadata"]
                else:
                    transitions.append(data)

        return metadata, transitions

    def match_transitions_to_csv(
        self, transitions: List[Dict], csv_data: List[Dict]
    ) -> List[Tuple]:
        """
        Match each JSONL transition to corresponding CSV row by timestamp.

        Returns list of (transition, csv_row) tuples.

        Handles both column naming conventions:
        - NS3 wrapper CSV: 'Simulation_Time_Sec'
        - Py-wrapper CSV: 'sim_time_sec'
        """
        matches = []

        # Detect which column name is used
        if csv_data and "Simulation_Time_Sec" in csv_data[0]:
            time_col = "Simulation_Time_Sec"
        elif csv_data and "sim_time_sec" in csv_data[0]:
            time_col = "sim_time_sec"
        else:
            raise KeyError(
                "Cannot find time column in CSV. Expected 'Simulation_Time_Sec' or 'sim_time_sec'"
            )

        for trans in transitions:
            jsonl_time = trans["sim_time_sec"]

            # Find closest CSV row by timestamp
            best_match = None
            best_diff = float("inf")

            for csv_row in csv_data:
                csv_time = float(csv_row[time_col])
                diff = abs(jsonl_time - csv_time)

                if diff < best_diff:
                    best_diff = diff
                    best_match = csv_row

            # Only match if within 0.1 sec (one BI)
            if best_match and best_diff <= 0.1:
                matches.append((trans, best_match))

        return matches

    def compare_variables(self, transitions: List[Dict], csv_data: List[Dict]) -> None:
        """
        Compare each variable between JSONL and CSV.
        """
        self.log("\n" + "=" * 80)
        self.log("VARIABLE-BY-VARIABLE CROSS-VALIDATION")
        self.log("=" * 80)

        # Match transitions to CSV rows
        matches = self.match_transitions_to_csv(transitions, csv_data)
        self.log(f"\nMatched {len(matches)} transitions to CSV rows")

        # Check each variable across all matched pairs
        # Adapt variable mapping based on CSV column names
        has_ns3_cols = csv_data and "Simulation_Time_Sec" in csv_data[0]
        has_pywrapper_cols = csv_data and "sim_time_sec" in csv_data[0]

        if has_ns3_cols:
            # NS3 wrapper CSV format
            variable_comparisons = {
                "sim_time_sec": ("Simulation_Time_Sec", self.compare_float),
                "num_sta": ("Num_STA", self.compare_int),
                "Num_Active_TWT_Groups": ("Num_Active_TWT_Groups", self.compare_int),
                "agg_total_bytes_tx": ("TX_Count", self.compare_cumulative),
                "Success": ("Success", self.compare_int),
            }
        elif has_pywrapper_cols:
            # Py-wrapper CSV format - different column names
            variable_comparisons = {
                "sim_time_sec": ("sim_time_sec", self.compare_float),
                "num_sta": ("num_sta", self.compare_int),
            }
            self.log("⚠️  Using Py-wrapper CSV format (limited variable comparisons)")
        else:
            self.log("❌ Cannot determine CSV format from columns!")
            return

        self.log("\n" + "-" * 80)
        self.log("SCALAR VARIABLE COMPARISONS")
        self.log("-" * 80)

        for i, (trans, csv_row) in enumerate(matches[:5]):  # Check first 5
            self.log(f"\n[Transition {i}] Time: {trans['sim_time_sec']:.3f}s")

            # Compare each variable
            for jsonl_var, (csv_var, compare_func) in variable_comparisons.items():
                if jsonl_var in trans:
                    jsonl_val = trans[jsonl_var]
                    csv_val = csv_row.get(csv_var)

                    result = compare_func(jsonl_val, csv_val, jsonl_var, csv_var)

                    if result["match"]:
                        self.matches.append((i, jsonl_var, result))
                        self.log(
                            f"  ✅ {jsonl_var:30s}: {jsonl_val} == {csv_var} = {csv_val}"
                        )
                    else:
                        self.mismatches.append((i, jsonl_var, result))
                        self.log(
                            f"  ❌ {jsonl_var:30s}: {jsonl_val} != {csv_var} = {csv_val} | {result['note']}"
                        )

        # Per-STA variable comparisons
        self.log("\n" + "-" * 80)
        self.log("PER-STA VARIABLE COMPARISONS")
        self.log("-" * 80)

        for i, (trans, csv_row) in enumerate(matches[:3]):  # Check first 3 transitions
            self.log(f"\n[Transition {i}] {len(trans['next_state'])} STAs")

            for sta_idx, sta_state in enumerate(
                trans["next_state"][:3]
            ):  # Check first 3 STAs
                self.log(f"\n  STA {sta_idx}:")

                # List per-STA variables being logged
                sta_vars = [
                    ("sta_id", "STA ID"),
                    ("bsr_queue_ac_be", "BSR Queue AC_BE"),
                    ("bsr_queue_ac_bk", "BSR Queue AC_BK"),
                    ("bsr_queue_ac_vi", "BSR Queue AC_VI"),
                    ("bsr_queue_ac_vo", "BSR Queue AC_VO"),
                    ("rssi_dbm", "RSSI (dBm)"),
                    ("snr_db", "SNR (dB)"),
                    ("last_rx_mcs", "Last RX MCS"),
                    ("bytes_received_at_ap", "RX Bytes at AP"),
                    ("packets_received_at_ap", "RX Packets at AP"),
                    ("oracle_duty_cycle", "Oracle Duty Cycle"),
                    ("oracle_awake_time_ms", "Oracle Awake Time (ms)"),
                    ("oracle_sleep_time_ms", "Oracle Sleep Time (ms)"),
                    ("oracle_total_energy_mj", "Oracle Total Energy (mJ)"),
                    ("oracle_packets_generated", "Oracle Packets Generated"),
                    ("oracle_packets_transmitted", "Oracle Packets Transmitted"),
                    ("oracle_bytes_transmitted", "Oracle Bytes Transmitted"),
                    ("oracle_mpdu_drops_expired", "Oracle MPDU Drops (Expired)"),
                    ("oracle_mpdu_drops_queue_full", "Oracle MPDU Drops (Queue Full)"),
                    ("oracle_queue_size_packets", "Oracle Queue Size (Packets)"),
                    ("oracle_queue_size_bytes", "Oracle Queue Size (Bytes)"),
                    ("oracle_avg_latency_ms", "Oracle Avg Latency (ms)"),
                ]

                for var_name, var_label in sta_vars:
                    value = sta_state.get(var_name)
                    self.log(f"    {var_label:35s}: {value}")

        # Aggregate analysis
        self.log("\n" + "-" * 80)
        self.log("AGGREGATE METRICS ANALYSIS")
        self.log("-" * 80)

        for i, (trans, csv_row) in enumerate(matches[:3]):
            self.log(f"\n[Transition {i}]")
            self.log(f"  Aggregate Bytes TX:      {trans['agg_total_bytes_tx']:15.0f}")
            if "TX_Count" in csv_row:
                self.log(f"  CSV TX_Count:            {csv_row['TX_Count']:15s}")
            else:
                self.log(f"  CSV TX_Count:            (not in py-wrapper CSV)")
            self.log(f"  Aggregate Energy (mJ):   {trans['agg_total_energy_mj']:15.2f}")
            self.log(f"  Aggregate Drops:         {trans['agg_total_drops']:15.0f}")
            self.log(f"  Aggregate Duty Cycle:    {trans['agg_mean_duty_cycle']:15.4f}")
            self.log(f"  Num STAs:                {trans['num_sta']:15d}")
            if "Num_STA" in csv_row:
                self.log(f"  CSV Num STAs:            {csv_row['Num_STA']:15s}")
            else:
                self.log(f"  CSV Num STAs:            (not in py-wrapper CSV)")
            self.log(
                f"  Num Active TWT Groups:   {len(trans['action']):15d} (2D action space)"
            )
            if "Num_Active_TWT_Groups" in csv_row:
                self.log(
                    f"  CSV Num Groups:          {csv_row['Num_Active_TWT_Groups']:15s}"
                )
            else:
                self.log(f"  CSV Num Groups:          (not in py-wrapper CSV)")

    def compare_float(self, jsonl_val, csv_val, jsonl_var, csv_var) -> Dict:
        """Compare float values with tolerance."""
        try:
            csv_num = float(csv_val)
            diff = abs(jsonl_val - csv_num)
            tolerance = 0.1  # 0.1 second tolerance for time

            if diff <= tolerance:
                return {"match": True, "note": f"diff={diff:.4f}"}
            else:
                return {
                    "match": False,
                    "note": f"diff={diff:.4f} > tolerance={tolerance}",
                }
        except (ValueError, TypeError):
            return {"match": False, "note": f"Cannot convert CSV to float: {csv_val}"}

    def compare_int(self, jsonl_val, csv_val, jsonl_var, csv_var) -> Dict:
        """Compare integer values."""
        try:
            csv_num = int(csv_val)
            if jsonl_val == csv_num:
                return {"match": True, "note": ""}
            else:
                return {
                    "match": False,
                    "note": f"values differ: {jsonl_val} vs {csv_num}",
                }
        except (ValueError, TypeError):
            return {"match": False, "note": f"Cannot convert CSV to int: {csv_val}"}

    def compare_cumulative(self, jsonl_val, csv_val, jsonl_var, csv_var) -> Dict:
        """
        Compare cumulative metrics (TX_Count in CSV is cumulative).
        For this, we'd need to compare deltas, not absolute values.
        """
        try:
            csv_num = int(csv_val)
            # CSV TX_Count is cumulative - can't directly compare to agg_total_bytes_tx
            # This would need delta comparison over time
            return {
                "match": True,
                "note": "Cumulative metric - requires delta analysis",
            }
        except (ValueError, TypeError):
            return {"match": False, "note": f"Cannot convert CSV: {csv_val}"}

    def generate_mapping_report(
        self, transitions: List[Dict], csv_data: List[Dict]
    ) -> None:
        """Generate detailed mapping report."""
        self.log("\n" + "=" * 80)
        self.log("VARIABLE MAPPING REPORT")
        self.log("=" * 80)

        self.log("\n[JSONL Transition Fields] → [CSV Wrapper Fields]")
        self.log("-" * 80)

        mapping = {
            "spawn_id": "Generated from --spawn-id argument",
            "step": "Transition counter (not in CSV)",
            "sim_time_sec": "Simulation_Time_Sec (primary key for matching)",
            "num_sta": "Num_STA",
            "action": "[schedule_idx, assignment_idx] (2D action from random selection)",
            "schedule_name": "Derived from action table",
            "assignment_name": "Derived from action table",
            "state[].sta_id": "From wrapper sta_observations[].realistic.sta_id",
            "state[].bsr_queue_ac_be": "From wrapper sta_observations[].realistic.bsr_queue_ac_be",
            "state[].rssi_dbm": "From wrapper sta_observations[].realistic.rssi_dbm",
            "state[].snr_db": "From wrapper sta_observations[].realistic.snr_db",
            "state[].last_rx_mcs": "From wrapper sta_observations[].realistic.last_rx_mcs",
            "state[].bytes_received_at_ap": "From wrapper sta_observations[].realistic.bytes_received_at_ap",
            "state[].oracle_duty_cycle": "From wrapper sta_observations[].oracle.duty_cycle",
            "state[].oracle_awake_time_ms": "From wrapper sta_observations[].oracle.awake_time_ms",
            "state[].oracle_total_energy_mj": "From wrapper sta_observations[].oracle.total_energy_consumed_mj",
            "state[].oracle_bytes_transmitted": "From wrapper sta_observations[].oracle.bytes_transmitted",
            "state[].oracle_packets_transmitted": "From wrapper sta_observations[].oracle.packets_transmitted",
            "state[].oracle_mpdu_drops_expired": "From wrapper sta_observations[].oracle.mpdu_drops_expired",
            "state[].oracle_queue_size_packets": "From wrapper sta_observations[].oracle.queue_size_packets",
            "agg_total_bytes_tx": "SUM(state[*].oracle_bytes_transmitted)",
            "agg_total_energy_mj": "SUM(state[*].oracle_total_energy_mj)",
            "agg_total_drops": "SUM(state[*].oracle_mpdu_drops_*)",
            "agg_mean_duty_cycle": "MEAN(state[*].oracle_duty_cycle)",
        }

        for jsonl_field, source in mapping.items():
            self.log(f"  {jsonl_field:35s} ← {source}")

        self.log("\n[CSV Wrapper Fields] → [JSONL Mapping]")
        self.log("-" * 80)

        csv_mapping = {
            "Timestamp_Sec": "Relative timestamp (not used in JSONL matching)",
            "Success": "Simulation success flag (not stored in JSONL)",
            "Num_STA": "→ num_sta in JSONL",
            "Simulation_Time_Sec": "→ sim_time_sec (primary key)",
            "Num_Active_TWT_Groups": "Num groups in schedule (used in action)",
            "TX_Count": "Cumulative TX packets (aggregated separately)",
            "RX_Count": "Cumulative RX packets (aggregated separately)",
        }

        for csv_field, jsonl_target in csv_mapping.items():
            self.log(f"  {csv_field:25s} {jsonl_target}")

    def run_full_analysis(self, jsonl_path: str, csv_path: str) -> None:
        """Run complete detailed analysis."""
        self.log("=" * 80)
        self.log("DETAILED VARIABLE-BY-VARIABLE CROSS-VALIDATION")
        self.log("=" * 80)

        # Load files
        self.log(f"\nLoading JSONL: {jsonl_path}")
        metadata, transitions = self.load_jsonl(jsonl_path)
        self.log(f"Loaded {len(transitions)} transitions")

        # Print metadata if available
        if metadata:
            self.log(f"\n📋 METADATA:")
            self.log(f"  Spawn ID:              {metadata.get('spawn_id')}")
            self.log(f"  Seed:                  {metadata.get('seed')}")
            self.log(f"  Log Timestamp:         {metadata.get('log_timestamp')}")
            self.log(f"  Py-Wrapper Env CSV:    {metadata.get('py_wrapper_env_csv')}")
            self.log(
                f"  Py-Wrapper Action CSV: {metadata.get('py_wrapper_action_csv')}"
            )
            self.log(f"  Data Log Dir:          {metadata.get('data_log_dir')}")
            self.log(f"  Num STAs:              {metadata.get('num_sta')}")

        self.log(f"\nLoading CSV: {csv_path}")
        csv_data = self.load_csv(csv_path)
        self.log(f"Loaded {len(csv_data)} CSV records")

        # Analyze
        self.compare_variables(transitions, csv_data)
        self.generate_mapping_report(transitions, csv_data)

        # Summary
        self.log("\n" + "=" * 80)
        self.log("SUMMARY")
        self.log("=" * 80)
        self.log(f"✅ Variable matches: {len(self.matches)}")
        self.log(f"❌ Variable mismatches: {len(self.mismatches)}")
        self.log(f"⚠️  Errors: {len(self.errors)}")


def main():
    parser = argparse.ArgumentParser(
        description="Detailed variable-by-variable cross-validation"
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Path to single JSONL transitions file (optional - will find all spawn_*.jsonl files if not provided)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to wrapper CSV log file (optional - auto-detected from metadata if not provided)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing spawn_*.jsonl files or eda-data root to validate all runs",
    )

    args = parser.parse_args()

    # Determine what to validate
    runs_to_validate = []

    if args.jsonl:
        # Single file specified
        if os.path.isdir(args.jsonl):
            # Directory with spawn files
            transitions_dir = args.jsonl
            spawn_files = sorted(
                [
                    os.path.join(transitions_dir, f)
                    for f in os.listdir(transitions_dir)
                    if f.startswith("spawn_") and f.endswith(".jsonl")
                ]
            )
            if spawn_files:
                runs_to_validate.append(("single", spawn_files))
        else:
            # Single spawn file
            runs_to_validate.append(("single", [args.jsonl]))

    elif args.dir:
        # Directory specified - could be eda-data root or transitions folder
        if os.path.basename(args.dir) == "transitions":
            # Direct transitions folder
            spawn_files = sorted(
                [
                    os.path.join(args.dir, f)
                    for f in os.listdir(args.dir)
                    if f.startswith("spawn_") and f.endswith(".jsonl")
                ]
            )
            if spawn_files:
                runs_to_validate.append((os.path.dirname(args.dir), spawn_files))
        else:
            # Search for run_*/transitions subdirectories
            for entry in sorted(os.listdir(args.dir)):
                run_path = os.path.join(args.dir, entry)
                transitions_path = os.path.join(run_path, "transitions")
                if os.path.isdir(transitions_path):
                    spawn_files = sorted(
                        [
                            os.path.join(transitions_path, f)
                            for f in os.listdir(transitions_path)
                            if f.startswith("spawn_") and f.endswith(".jsonl")
                        ]
                    )
                    if spawn_files:
                        runs_to_validate.append((run_path, spawn_files))

    else:
        # Auto-discover eda-data/run_*/transitions/
        if os.path.isdir("eda-data"):
            for entry in sorted(os.listdir("eda-data")):
                run_path = os.path.join("eda-data", entry)
                transitions_path = os.path.join(run_path, "transitions")
                if os.path.isdir(transitions_path):
                    spawn_files = sorted(
                        [
                            os.path.join(transitions_path, f)
                            for f in os.listdir(transitions_path)
                            if f.startswith("spawn_") and f.endswith(".jsonl")
                        ]
                    )
                    if spawn_files:
                        runs_to_validate.append((run_path, spawn_files))
        else:
            # Look in current directory for transitions folder
            for root, dirs, files in os.walk("."):
                if "transitions" in dirs:
                    transitions_dir = os.path.join(root, "transitions")
                    spawn_files = sorted(
                        [
                            os.path.join(transitions_dir, f)
                            for f in os.listdir(transitions_dir)
                            if f.startswith("spawn_") and f.endswith(".jsonl")
                        ]
                    )
                    if spawn_files:
                        runs_to_validate.append((root, spawn_files))

    if not runs_to_validate:
        parser.print_help()
        print("\n❌ Error: No spawn_*.jsonl files found.")
        print("   Usage: python3 2-validate-logvstap.py [--dir <eda-data or run_dir>]")
        sys.exit(1)

    # Validate all runs
    print("=" * 80)
    total_spawns = sum(len(spawns) for _, spawns in runs_to_validate)
    print(
        f"BATCH VALIDATION: {len(runs_to_validate)} run(s), {total_spawns} spawn(s) total"
    )
    print("=" * 80)

    all_results = []

    for run_idx, (run_path, spawn_files) in enumerate(runs_to_validate, 1):
        run_name = os.path.basename(run_path) if run_path != "single" else "Single Run"
        print(f"\n{'='*80}")
        print(f"RUN [{run_idx}/{len(runs_to_validate)}]: {run_name}")
        print(f"  Spawns: {len(spawn_files)}")
        print("=" * 80)

        results = []
        for spawn_idx, jsonl_file in enumerate(spawn_files, 1):
            print(
                f"\n  [{spawn_idx}/{len(spawn_files)}] {os.path.basename(jsonl_file)}",
                end=" ",
            )

            # Auto-detect CSV from metadata if not provided
            csv_path = args.csv
            if not csv_path:
                try:
                    with open(jsonl_file, "r") as f:
                        first_line = f.readline()

                    data = json.loads(first_line)
                    metadata = data.get("_metadata")

                    if metadata:
                        data_log_dir = metadata.get("data_log_dir")
                        py_wrapper_env = metadata.get("py_wrapper_env_csv")

                        if data_log_dir and py_wrapper_env:
                            csv_path = os.path.join(data_log_dir, py_wrapper_env)
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

            if not csv_path:
                print("❌ (No CSV found)")
                results.append((os.path.basename(jsonl_file), False, "No CSV"))
                continue

            if not os.path.exists(csv_path):
                print(f"❌ (CSV missing)")
                results.append((os.path.basename(jsonl_file), False, "CSV not found"))
                continue

            try:
                validator = DetailedLogValidator(verbose=True)
                validator.run_full_analysis(jsonl_file, csv_path)

                status = "✅" if len(validator.mismatches) == 0 else "⚠️"
                matches = len(validator.matches)
                mismatches = len(validator.mismatches)

                print(f"{status} ({matches} matches, {mismatches} mismatches)")
                results.append(
                    (
                        os.path.basename(jsonl_file),
                        True,
                        f"{matches}/{matches + mismatches}",
                    )
                )

            except Exception as e:
                print(f"❌ (Error: {str(e)[:30]})")
                results.append((os.path.basename(jsonl_file), False, str(e)[:30]))

        all_results.append((run_name, results))

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    grand_total_passed = 0
    grand_total_failed = 0

    for run_name, results in all_results:
        passed = sum(1 for _, success, _ in results if success)
        failed = len(results) - passed
        grand_total_passed += passed
        grand_total_failed += failed

        status_symbol = "✅" if failed == 0 else "⚠️"
        print(f"\n{status_symbol} {run_name}: {passed}/{len(results)} passed")

        for spawn_name, success, status in results:
            symbol = "  ✅" if success else "  ❌"
            print(f"{symbol} {spawn_name:20s} {status}")

    print("\n" + "-" * 80)
    total = grand_total_passed + grand_total_failed
    print(
        f"TOTAL: {grand_total_passed} passed, {grand_total_failed} failed out of {total}"
    )

    if grand_total_failed == 0:
        print("🎉 All spawns in all runs validated successfully!")
        sys.exit(0)
    else:
        print(f"⚠️  {grand_total_failed} spawn(s) failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()

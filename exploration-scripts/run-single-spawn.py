#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""run-single-spawn.py - single NS-3 spawn for RL data collection.

Runs one NS-3 simulation with random actions from the two-dimensional action space:
- Dim-1: Schedule (TWT group configurations)
- Dim-2: Assignment (STA-to-group patterns)

Collects transitions and logs them to JSONL format.
Uses TWTWrapper for all NS-3 communication.

The collected data uses per-STA embeddings, the same columns as py-wrapper-env-*.csv.

Usage:
    python3.11 run-single-spawn.py --seed 1000 --spawn-id 0 --output spawn_0.jsonl
"""

import sys
import os
import json
import argparse
import random
import numpy as np
from datetime import datetime

# Add parent directory for TWTWrapper
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from pb_twt_wrapper_py import TWTWrapper

# Import action table utilities
from generate_action_tables import apply_assignment_pattern

# Constants
BEACON_INTERVAL_MS = 102.4


def load_action_tables(schedule_file=None, assignment_file=None):
    """Load the two action tables."""
    if schedule_file is None:
        schedule_file = os.path.join(script_dir, "table_schedule.json")
    if assignment_file is None:
        assignment_file = os.path.join(script_dir, "table_assignment.json")

    with open(schedule_file, "r") as f:
        schedule_table = json.load(f)

    with open(assignment_file, "r") as f:
        assignment_table = json.load(f)

    return schedule_table, assignment_table


def build_action_from_indices(
    schedule_idx, assignment_idx, schedule_table, assignment_table, num_sta
):
    """
    Build a complete action dict from schedule and assignment indices.

    Args:
        schedule_idx: Index into schedule table (Dim-1)
        assignment_idx: Index into assignment table (Dim-2)
        schedule_table: Loaded schedule table
        assignment_table: Loaded assignment table
        num_sta: Number of STAs in the simulation

    Returns:
        action_dict compatible with TWTWrapper.step()
    """
    schedule = schedule_table["schedules"][schedule_idx]
    assignment = assignment_table["assignments"][assignment_idx]

    num_groups = schedule["num_groups"]

    # Build TWT group configs from schedule
    twt_group_configs = []
    for group in schedule["groups"]:
        twt_group_configs.append(
            {
                "group_id": group["group_id"],
                "twt_wake_interval_ms": BEACON_INTERVAL_MS,
                "twt_wake_duration_ms": group["wake_duration_ms"],
                "twt_sp_offset_ms": group["sp_offset_ms"],
                "num_stas_assigned": 0,  # Will be computed below
            }
        )

    # Apply assignment pattern to get STA-to-group mappings
    sta_group_mappings = apply_assignment_pattern(assignment, num_sta, num_groups)

    # Build STA assignments and count per group
    sta_group_assignments = []
    group_counts = {g["group_id"]: 0 for g in schedule["groups"]}

    for sta_id, group_id in sta_group_mappings:
        sta_group_assignments.append(
            {"sta_id": sta_id, "assigned_twt_group": group_id, "enable_twt": 1}
        )
        if group_id in group_counts:
            group_counts[group_id] += 1

    # Update num_stas_assigned in group configs
    for gc in twt_group_configs:
        gc["num_stas_assigned"] = group_counts.get(gc["group_id"], 0)

    return {
        "num_sta": num_sta,
        "num_active_twt_groups": num_groups,
        "action_timestamp_ms": 0,
        "twt_group_configs": twt_group_configs,
        "sta_group_assignments": sta_group_assignments,
        # Metadata for logging
        "_schedule_idx": schedule_idx,
        "_assignment_idx": assignment_idx,
        "_schedule_name": schedule.get("name", f"S{schedule_idx}"),
        "_assignment_name": assignment.get("name", f"A{assignment_idx}"),
    }


def extract_state_per_sta(env_dict):
    """
    Extract per-STA state embeddings from environment dict.

    Returns a list of per-STA feature dicts (same structure for any num_sta).
    Each STA has the same feature columns - this enables STA-agnostic learning.
    """
    states = []

    for sta_obs in env_dict.get("sta_observations", []):
        realistic = sta_obs.get("realistic", {})
        oracle = sta_obs.get("oracle", {})

        # Per-STA features (flat dict, easy to convert to dataframe later)
        sta_state = {
            "sta_id": realistic.get("sta_id", 0),
            # BSR / Queue metrics (realistic)
            "bsr_queue_ac_be": realistic.get("bsr_queue_ac_be", 0),
            "bsr_queue_ac_bk": realistic.get("bsr_queue_ac_bk", 0),
            "bsr_queue_ac_vi": realistic.get("bsr_queue_ac_vi", 0),
            "bsr_queue_ac_vo": realistic.get("bsr_queue_ac_vo", 0),
            # Link quality (realistic)
            "rssi_dbm": realistic.get("rssi_dbm", 0),
            "snr_db": realistic.get("snr_db", 0),
            "last_rx_mcs": realistic.get("last_rx_mcs", 0),
            # Traffic (realistic)
            "bytes_received_at_ap": realistic.get("bytes_received_at_ap", 0),
            "packets_received_at_ap": realistic.get("packets_received_at_ap", 0),
            "airtime_used_us": realistic.get("airtime_used_us", 0),
            "fcs_error_count": realistic.get("fcs_error_count", 0),
            "rx_fragment_count": realistic.get("rx_fragment_count", 0),
            "last_rx_timestamp_us": realistic.get("last_rx_timestamp_us", 0),
            # Oracle metrics (ground truth)
            "oracle_duty_cycle": oracle.get("duty_cycle", 0),
            "oracle_awake_time_ms": oracle.get("awake_time_ms", 0),
            "oracle_sleep_time_ms": oracle.get("sleep_time_ms", 0),
            "oracle_total_energy_mj": oracle.get("total_energy_consumed_mj", 0),
            "oracle_packets_generated": oracle.get("packets_generated", 0),
            "oracle_packets_enqueued": oracle.get("packets_enqueued", 0),
            "oracle_packets_transmitted": oracle.get("packets_transmitted", 0),
            "oracle_bytes_transmitted": oracle.get("bytes_transmitted", 0),
            "oracle_mpdu_drops_expired": oracle.get("mpdu_drops_expired", 0),
            "oracle_mpdu_drops_queue_full": oracle.get("mpdu_drops_queue_full", 0),
            "oracle_queue_size_packets": oracle.get("queue_size_packets", 0),
            "oracle_queue_size_bytes": oracle.get("queue_size_bytes", 0),
            "oracle_psdu_response_timeouts": oracle.get("psdu_response_timeouts", 0),
            "oracle_avg_latency_ms": oracle.get("avg_latency_ms", 0),
        }
        states.append(sta_state)

    return states


def run_single_spawn(seed, spawn_id, output_file, verbose=True):
    """
    Run one NS-3 spawn with random actions and collect transitions.

    Args:
        seed: Random seed for both Python and NS-3
        spawn_id: Identifier for this spawn
        output_file: Path to output JSONL file
        verbose: Print progress

    Returns:
        (transitions, summary_stats)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Load action tables
    schedule_table, assignment_table = load_action_tables()
    num_schedules = len(schedule_table["schedules"])
    num_assignments = len(assignment_table["assignments"])

    if verbose:
        print(
            f"[Spawn {spawn_id}] Action space: {num_schedules} schedules × {num_assignments} assignments = {num_schedules * num_assignments}"
        )

    # Initialize wrapper (will create its own logs)
    # Get absolute path to data-log directory
    data_log_abs = os.path.abspath(os.path.join(script_dir, "..", "data-log"))

    wrapper = TWTWrapper(
        log_dir=data_log_abs,  # Use absolute path to avoid confusion
        enable_logging=True,
        verbose=verbose,
    )

    transitions = []
    prev_state = None
    prev_action_info = None
    step = 0

    try:
        # Start simulation
        env = wrapper.reset(seed=seed)
        num_sta = env["num_sta"]

        if verbose:
            print(f"[Spawn {spawn_id}] Started with {num_sta} STAs")

        while True:
            # Check if done
            sim_time = env.get("simulation_time_sec", 0)

            # Extract current state (per-STA embeddings)
            current_state = extract_state_per_sta(env)

            # Store transition from previous step
            if prev_state is not None:
                transition = {
                    "spawn_id": spawn_id,
                    "step": step - 1,
                    "sim_time_sec": sim_time,
                    "num_sta": num_sta,
                    "action": [
                        prev_action_info["schedule_idx"],
                        prev_action_info["assignment_idx"],
                    ],
                    "schedule_name": prev_action_info["schedule_name"],
                    "assignment_name": prev_action_info["assignment_name"],
                    "state": prev_state,
                    "next_state": current_state,
                    "done": False,
                    # Aggregate metrics for analysis/logging
                    "agg_total_bytes_tx": sum(
                        s["oracle_bytes_transmitted"] for s in current_state
                    ),
                    "agg_total_energy_mj": sum(
                        s["oracle_total_energy_mj"] for s in current_state
                    ),
                    "agg_total_drops": sum(
                        s["oracle_mpdu_drops_expired"]
                        + s["oracle_mpdu_drops_queue_full"]
                        for s in current_state
                    ),
                    "agg_mean_duty_cycle": (
                        np.mean([s["oracle_duty_cycle"] for s in current_state])
                        if current_state
                        else 0
                    ),
                }
                transitions.append(transition)

            # Random action selection (exploration)
            schedule_idx = random.randint(0, num_schedules - 1)
            assignment_idx = random.randint(0, num_assignments - 1)

            action = build_action_from_indices(
                schedule_idx, assignment_idx, schedule_table, assignment_table, num_sta
            )

            # Step the simulation
            env, done = wrapper.step(action)

            if done:
                # Mark last transition as terminal
                if transitions:
                    transitions[-1]["done"] = True
                if verbose:
                    print(f"[Spawn {spawn_id}] Simulation finished after {step} steps")
                break

            # Store for next iteration
            prev_state = current_state
            prev_action_info = {
                "schedule_idx": schedule_idx,
                "assignment_idx": assignment_idx,
                "schedule_name": action.get("_schedule_name", ""),
                "assignment_name": action.get("_assignment_name", ""),
            }
            step += 1

    except Exception as e:
        print(f"[Spawn {spawn_id}] Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Capture wrapper log metadata before closing
        wrapper_log_timestamp = wrapper.log_timestamp
        wrapper.close()

    # Create metadata record with CSV log file references
    metadata = {
        "_metadata": {
            "spawn_id": spawn_id,
            "seed": seed,
            "log_timestamp": wrapper_log_timestamp,
            "py_wrapper_env_csv": f"py-wrapper-env-{wrapper_log_timestamp}.csv",
            "py_wrapper_action_csv": f"py-wrapper-action-{wrapper_log_timestamp}.csv",
            "data_log_dir": data_log_abs,
            "num_transitions": len(transitions),
            "num_sta": num_sta if "num_sta" in dir() else 0,
        }
    }

    # Save transitions to JSONL with metadata as first line
    with open(output_file, "w") as f:
        # Write metadata as first line
        f.write(json.dumps(metadata) + "\n")
        # Write transitions
        for t in transitions:
            f.write(json.dumps(t) + "\n")

    # Compute summary stats
    if transitions:
        avg_energy = np.mean([t["agg_total_energy_mj"] for t in transitions])
        avg_bytes = np.mean([t["agg_total_bytes_tx"] for t in transitions])
        avg_drops = np.mean([t["agg_total_drops"] for t in transitions])
        avg_duty = np.mean([t["agg_mean_duty_cycle"] for t in transitions])
    else:
        avg_energy = avg_bytes = avg_drops = avg_duty = 0

    summary = {
        "spawn_id": spawn_id,
        "seed": seed,
        "num_sta": num_sta if "num_sta" in dir() else 0,
        "num_transitions": len(transitions),
        "avg_total_energy_mj": avg_energy,
        "avg_total_bytes_tx": avg_bytes,
        "avg_total_drops": avg_drops,
        "avg_mean_duty_cycle": avg_duty,
    }

    if verbose:
        print(f"[Spawn {spawn_id}] Collected {len(transitions)} transitions")
        print(
            f"[Spawn {spawn_id}] Avg energy: {avg_energy:.2f} mJ, Avg bytes: {avg_bytes:.0f}, Avg drops: {avg_drops:.1f}"
        )

    return transitions, summary


def main():
    parser = argparse.ArgumentParser(description="Single spawn for RL data collection")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--spawn-id", type=int, required=True, help="Spawn ID")
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSONL file path"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    run_single_spawn(
        seed=args.seed,
        spawn_id=args.spawn_id,
        output_file=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

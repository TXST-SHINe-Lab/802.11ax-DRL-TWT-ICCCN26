#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""episode_file_runner.py - run a single NS-3 episode with file-based communication.

Runs in its own process, one per episode, spawned by file_comm_env.py and talking to it through JSON files.

Protocol:
1. Start NS-3, do warmup steps
2. Write initial observation to response.json
3. Wait for action.json
4. Execute action, write next observation to response.json
5. Repeat until done
6. Write done response, exit cleanly

One process per episode is what keeps the shared memory clean: the segments die with the process.

Usage:
    python3.11 episode_file_runner.py --seed 1234 --warmup-steps 2 --comm-dir /tmp/twt_comm
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from typing import Dict, Optional, List, Tuple

# Add parent directory for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _parent_dir)
sys.path.insert(0, os.path.join(_script_dir, "..", "exploration-scripts"))

from pb_twt_wrapper_py import TWTWrapper
from generate_action_tables import apply_assignment_pattern

BEACON_INTERVAL_MS = 102.4  # ms, the standard beacon interval


def load_action_tables():
    """Load the schedule and assignment tables written by generate_action_tables.py.

    Raises if either file is missing; there is deliberately no fallback.
    """
    exploration_dir = os.path.join(_parent_dir, "exploration-scripts")

    with open(os.path.join(exploration_dir, "table_schedule.json"), "r") as f:
        schedule_table = json.load(f)
    with open(os.path.join(exploration_dir, "table_assignment.json"), "r") as f:
        assignment_table = json.load(f)

    return schedule_table, assignment_table


def build_action_dict(
    schedule_idx: int,
    assignment_idx: int,
    schedule_table: Dict,
    assignment_table: Dict,
    num_sta: int,
) -> Dict:
    """Build NS-3 action dict from schedule and assignment indices."""
    schedule = schedule_table["schedules"][schedule_idx]
    assignment = assignment_table["assignments"][assignment_idx]

    num_groups = schedule["num_groups"]

    # Build TWT group configs
    twt_group_configs = []
    for group in schedule["groups"]:
        twt_group_configs.append(
            {
                "group_id": group["group_id"],
                "twt_wake_interval_ms": BEACON_INTERVAL_MS,
                "twt_wake_duration_ms": group["wake_duration_ms"],
                "twt_sp_offset_ms": group["sp_offset_ms"],
                "num_stas_assigned": 0,
            }
        )

    # Apply assignment pattern
    sta_group_mappings = apply_assignment_pattern(assignment, num_sta, num_groups)

    # Build STA assignments
    sta_group_assignments = []
    group_counts = {g["group_id"]: 0 for g in schedule["groups"]}

    for sta_id, group_id in sta_group_mappings:
        sta_group_assignments.append(
            {"sta_id": sta_id, "assigned_twt_group": group_id, "enable_twt": 1}
        )
        if group_id in group_counts:
            group_counts[group_id] += 1

    for gc in twt_group_configs:
        gc["num_stas_assigned"] = group_counts.get(gc["group_id"], 0)

    return {
        "num_sta": num_sta,
        "num_active_twt_groups": num_groups,
        "action_timestamp_ms": 0,
        "twt_group_configs": twt_group_configs,
        "sta_group_assignments": sta_group_assignments,
    }


# The 9 raw per-STA features sent to file_comm_env, which derives the 4 deltas and assembles the 13-feature observation.
# Unique-value counts come from the EDA; see metric_recommendations.json under exploration-scripts/eda-data/run_*/.
PER_STA_RAW_FEATURE_NAMES = [
    # Realistic, 5 features
    "bsr_queue_ac_be",  # Buffer status best effort, 111 unique values
    "airtime_used_us",  # Airtime consumption, 87644 unique values
    "fcs_error_count",  # Frame check errors, 357 unique values
    "rx_fragment_count",  # Received fragments, 72 unique values
    "last_rx_timestamp_us",  # Last RX timestamp, 38 unique values
    # Oracle, 4 features, TWT-related and visible to PPO
    "awake_time_ms",  # Awake time, 5576 unique values
    "sleep_time_ms",  # Sleep time, 13942 unique values
    "duty_cycle",  # Wake/sleep ratio, 115019 unique values
    "packets_transmitted",  # Sent packets, 8356 unique values
]


def extract_observation(env: Dict) -> np.ndarray:
    """
    Extract the flat observation vector from an environment dict.

    Raw features per STA, 9 in total, sent on to file_comm_env.py:
    REALISTIC (5):
      - bsr_queue_ac_be: Buffer status for best effort
      - airtime_used_us: Airtime consumption
      - fcs_error_count: Frame check sequence errors
      - rx_fragment_count: Received fragments
      - last_rx_timestamp_us: Last RX timestamp

    ORACLE (4 - TWT related):
      - awake_time_ms: Time spent awake
      - sleep_time_ms: Time spent sleeping
      - duty_cycle: Wake/sleep ratio
      - packets_transmitted: Successfully sent packets

    file_comm_env.py derives the 4 deltas from these and builds the 13-feature observation.
    Length here is 9 features x 16 STAs = 144, not the 208 PPO finally sees.
    """
    obs = []

    for sta_obs in env.get("sta_observations", []):
        realistic = sta_obs.get("realistic", {})
        oracle = sta_obs.get("oracle", {})

        # --- Realistic features, 5 ---
        obs.append(realistic.get("bsr_queue_ac_be", 0))
        obs.append(realistic.get("airtime_used_us", 0))
        obs.append(realistic.get("fcs_error_count", 0))
        obs.append(realistic.get("rx_fragment_count", 0))
        obs.append(realistic.get("last_rx_timestamp_us", 0))

        # --- Oracle features, 4, TWT-related ---
        obs.append(oracle.get("awake_time_ms", 0))
        obs.append(oracle.get("sleep_time_ms", 0))
        obs.append(oracle.get("duty_cycle", 0))
        obs.append(oracle.get("packets_transmitted", 0))

    return np.array(obs, dtype=np.float32)


def write_response(comm_dir: str, response: Dict):
    """Write response.json atomically, via a .tmp file and a rename.

    The reader therefore never sees a half-written file.
    """
    filepath = os.path.join(comm_dir, "response.json")
    tmp = filepath + ".tmp"
    with open(tmp, "w") as f:
        json.dump(response, f)
    os.rename(tmp, filepath)


def read_action(comm_dir: str, timeout: float = 60.0) -> Optional[Dict]:
    """Poll for action.json, read it, and delete it. Returns None on timeout.

    Timeout is in seconds; a half-written file raises and is retried on the next tick.
    """
    filepath = os.path.join(comm_dir, "action.json")
    start = time.time()

    while time.time() - start < timeout:
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    action = json.load(f)
                os.remove(
                    filepath
                )  # Consume it so the next poll blocks on a fresh write
                return action
            except (json.JSONDecodeError, IOError):
                pass
        time.sleep(0.02)

    return None


def env_to_summary(env: Dict) -> Dict:
    """Reduce the environment dict to the fields the training process needs.

    Keeps the response small; the full env carries far more than the reward reads.
    """
    return {
        "num_sta": env.get("num_sta", 0),
        "simulation_time_sec": env.get("simulation_time_sec", 0),
        "observation_count": env.get("observation_count", 0),
        # Kept in full: the reward is computed from these per-STA metrics.
        "sta_observations": env.get("sta_observations", []),
    }


def run_episode(seed: int, warmup_steps: int, comm_dir: str, verbose: bool = False):
    """Drive one NS-3 episode to completion over the JSON file protocol.

    Writes response.json and consumes action.json until NS-3 reports done, then
    writes a final done response. Errors are reported as a response of type "error"
    so the parent env sees a clean failure rather than a hang.
    """

    log = lambda msg: print(f"[Runner {seed}] {msg}") if verbose else None

    try:
        log("Loading action tables...")
        schedule_table, assignment_table = load_action_tables()
        num_schedules = len(schedule_table["schedules"])
        num_assignments = len(assignment_table["assignments"])

        log(f"Initializing TWTWrapper...")
        wrapper = TWTWrapper(
            log_dir=os.path.join(_parent_dir, "data-log"),
            enable_logging=False,
            verbose=verbose,
        )

        log(f"Starting simulation with seed {seed}...")
        env = wrapper.reset(seed=seed)
        num_sta = env.get("num_sta", 16)
        log(f"Simulation started with {num_sta} STAs")

        # Warm up on schedule 0, assignment 0, so the agent's first real action sees a settled network.
        warmup_action = build_action_dict(
            0, 0, schedule_table, assignment_table, num_sta
        )
        for w in range(warmup_steps):
            env, done = wrapper.step(warmup_action)
            if done:
                log("Episode ended during warmup!")
                write_response(comm_dir, {"type": "error", "error": "warmup_ended"})
                return

        log(f"Warmup complete, sending initial observation")

        # Send initial observation
        obs = extract_observation(env)
        write_response(
            comm_dir,
            {
                "type": "obs",
                "obs": obs.tolist(),
                "env": env_to_summary(env),
            },
        )

        # Main loop
        step = 0
        while True:
            # Wait for action
            action = read_action(comm_dir, timeout=60.0)

            if action is None:
                log("Timeout waiting for action, exiting")
                break

            schedule_idx = action.get("schedule_idx", 0)
            assignment_idx = action.get("assignment_idx", 0)

            # Clip rather than reject; an out-of-range index must not abort the episode.
            schedule_idx = max(0, min(schedule_idx, num_schedules - 1))
            assignment_idx = max(0, min(assignment_idx, num_assignments - 1))

            # Build and execute action
            action_dict = build_action_dict(
                schedule_idx, assignment_idx, schedule_table, assignment_table, num_sta
            )

            env, done = wrapper.step(action_dict)
            step += 1

            if done or not env:
                log(f"Episode done after {step} steps")
                write_response(
                    comm_dir,
                    {
                        "type": "done",
                        "done": True,
                        "step": step,
                    },
                )
                break

            # Send observation
            obs = extract_observation(env)
            write_response(
                comm_dir,
                {
                    "type": "obs",
                    "obs": obs.tolist(),
                    "env": env_to_summary(env),
                    "done": False,
                },
            )

        log("Closing wrapper...")
        wrapper.close()
        log("Episode complete")

    except Exception as e:
        import traceback

        # Report the failure through the protocol; the parent env is blocked on response.json.
        traceback.print_exc()
        write_response(
            comm_dir,
            {
                "type": "error",
                "error": str(e),
            },
        )


def main():
    parser = argparse.ArgumentParser(description="Episode file runner")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup steps")
    parser.add_argument(
        "--comm-dir", type=str, required=True, help="Communication directory"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run_episode(
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        comm_dir=args.comm_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

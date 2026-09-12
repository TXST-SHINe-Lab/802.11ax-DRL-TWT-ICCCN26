#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""stack-data.py - convert JSONL transition files to stacked NPZ for analysis and training.

Steps:
1. Reads all JSONL transition files from a collection run
2. Stacks states, actions, rewards, next_states, dones into arrays
3. Handles variable number of STAs by padding or using per-STA embeddings
4. Saves to NPZ format for efficient loading

Usage:
    python3.11 stack-data.py
"""

import argparse
import json
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stack JSONL transitions into NPZ format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing JSONL files (the run_* directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output NPZ file path (default: {input_dir}/stacked_transitions.npz)",
    )
    parser.add_argument(
        "--max-stas",
        type=int,
        default=16,
        help="Maximum number of STAs for padding (default: 16)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress"
    )
    return parser.parse_args()


def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load transitions from a JSONL file, skipping metadata line if present."""
    transitions = []
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Skip metadata line (first line with '_metadata' key)
                if "_metadata" in data:
                    continue
                transitions.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {filepath}: {e}")
    return transitions


def extract_state_features(state, max_stas: int) -> Tuple[np.ndarray, int]:
    """
    Extract state features from the state.

    State can be:
    - List of per-STA dicts (new JSONL format): [{'sta_id': 0, 'bsr_queue_ac_be': ..., ...}, ...]
    - Dict of global features (old format): {'simulation_time_ms': ..., ...}

    Returns:
        - Flattened state array with padding for max_stas
        - Actual number of STAs in this state
    """
    # New format: list of per-STA observations
    if isinstance(state, list):
        num_stas = len(state)

        # Define per-STA feature names (order matters for consistency)
        per_sta_feature_names = [
            "bsr_queue_ac_be",
            "bsr_queue_ac_bk",
            "bsr_queue_ac_vi",
            "bsr_queue_ac_vo",
            "rssi_dbm",
            "snr_db",
            "last_rx_mcs",
            "bytes_received_at_ap",
            "packets_received_at_ap",
            "airtime_used_us",
            "fcs_error_count",
            "rx_fragment_count",
            "last_rx_timestamp_us",
            "oracle_duty_cycle",
            "oracle_awake_time_ms",
            "oracle_sleep_time_ms",
            "oracle_total_energy_mj",
            "oracle_packets_generated",
            "oracle_packets_enqueued",
            "oracle_packets_transmitted",
            "oracle_bytes_transmitted",
            "oracle_mpdu_drops_expired",
            "oracle_mpdu_drops_queue_full",
            "oracle_psdu_response_timeouts",
            "oracle_queue_size_packets",
            "oracle_queue_size_bytes",
            "oracle_avg_latency_ms",
        ]

        # Extract features for each STA (with padding for max_stas)
        all_sta_features = []
        for sta_idx in range(max_stas):
            sta_features = []
            if sta_idx < num_stas:
                sta_dict = state[sta_idx]
                for feat_name in per_sta_feature_names:
                    sta_features.append(float(sta_dict.get(feat_name, 0.0)))
            else:
                # Padding for missing STAs
                sta_features = [0.0] * len(per_sta_feature_names)
            all_sta_features.extend(sta_features)

        full_state = np.array(all_sta_features, dtype=np.float32)
        return full_state, num_stas

    # Old format: dict with per-STA keys (legacy support)
    # Global features (not per-STA)
    global_features = []

    # Get simulation time
    if "simulation_time_ms" in state:
        global_features.append(state["simulation_time_ms"])

    # Get BI count
    if "bi_count" in state:
        global_features.append(state["bi_count"])

    # Per-STA features - find all STA indices
    sta_indices = set()
    for key in state.keys():
        if "_sta_" in key:
            # Extract STA index from key like "active_calls_sta_0"
            parts = key.split("_sta_")
            if len(parts) == 2:
                try:
                    sta_idx = int(parts[1])
                    sta_indices.add(sta_idx)
                except ValueError:
                    pass

    num_stas = len(sta_indices) if sta_indices else 0

    # Define per-STA feature names (order matters for consistency)
    per_sta_feature_names = [
        "active_calls",
        "oracle_bytes_transmitted",
        "oracle_call_count",
        "oracle_bytes_per_call",
        "start_offset_us",
        "wake_duration_us",
        "packets_queued",
        "queue_bytes",
        "ap_rx_bytes",
        "total_rx_bytes",
        "total_tx_bytes",
        "total_rx_packets",
        "total_tx_packets",
        "total_expired_packets",
        "total_dropped_packets",
        # Add more as needed based on wrapper output
    ]

    # Extract per-STA features
    all_sta_features = []
    for sta_idx in range(max_stas):
        sta_features = []
        for feat_name in per_sta_feature_names:
            key = f"{feat_name}_sta_{sta_idx}"
            if key in state:
                sta_features.append(float(state[key]))
            else:
                # Padding for non-existent STAs or missing features
                sta_features.append(0.0)
        all_sta_features.extend(sta_features)

    # Combine global and per-STA features
    global_features_arr = np.array(global_features, dtype=np.float32)
    sta_features_arr = np.array(all_sta_features, dtype=np.float32)

    full_state = np.concatenate([global_features_arr, sta_features_arr])

    return full_state, num_stas


def stack_transitions(
    transitions: List[Dict], max_stas: int, verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Stack all transitions into arrays.

    Returns dict with:
        - states: (N, state_dim)
        - actions: (N, 2) for [schedule_idx, assignment_idx]
        - rewards: (N,)
        - next_states: (N, state_dim)
        - dones: (N,)
        - num_stas: (N,) actual STA count per transition
        - metadata: various tracking info
    """
    if not transitions:
        raise ValueError("No transitions to stack!")

    # First pass: determine state dimension
    sample_state, sample_num_stas = extract_state_features(
        transitions[0]["state"], max_stas
    )
    state_dim = len(sample_state)

    if verbose:
        print(f"State dimension: {state_dim}")
        print(f"Sample num_stas: {sample_num_stas}")

    n = len(transitions)

    # Pre-allocate arrays
    states = np.zeros((n, state_dim), dtype=np.float32)
    next_states = np.zeros((n, state_dim), dtype=np.float32)
    actions = np.zeros((n, 2), dtype=np.int32)  # [schedule_idx, assignment_idx]
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.bool_)
    num_stas_arr = np.zeros(n, dtype=np.int32)

    # Track spawn indices for stratification
    spawn_indices = np.zeros(n, dtype=np.int32)
    step_indices = np.zeros(n, dtype=np.int32)

    # Stack transitions
    for i, trans in enumerate(transitions):
        # State
        state_vec, n_stas = extract_state_features(trans["state"], max_stas)
        states[i] = state_vec
        num_stas_arr[i] = n_stas

        # Action (handle both formats: dict or list)
        action = trans.get("action", [0, 0])
        if isinstance(action, dict):
            actions[i, 0] = int(action.get("schedule_idx", 0))
            actions[i, 1] = int(action.get("assignment_idx", 0))
        elif isinstance(action, (list, tuple)) and len(action) >= 2:
            actions[i, 0] = int(action[0])
            actions[i, 1] = int(action[1])
        else:
            # Single action index - assume it's combined
            actions[i, 0] = int(action) if action else 0
            actions[i, 1] = 0

        # Reward (optional for EDA - use aggregate metrics if available)
        if "reward" in trans:
            rewards[i] = float(trans["reward"])
        else:
            # Fallback: use aggregate bytes transmitted as proxy
            rewards[i] = float(trans.get("agg_total_bytes_tx", 0.0))

        # Next state
        next_state_vec, _ = extract_state_features(trans["next_state"], max_stas)
        next_states[i] = next_state_vec

        # Done
        dones[i] = trans.get("done", False)

        # Metadata
        spawn_indices[i] = trans.get("spawn_idx", 0)
        step_indices[i] = trans.get("step", i)

        if verbose and (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{n} transitions...")

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "num_stas": num_stas_arr,
        "spawn_indices": spawn_indices,
        "step_indices": step_indices,
    }


def compute_statistics(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute summary statistics for the dataset."""
    stats = {}

    # Basic counts
    stats["n_transitions"] = len(data["states"])
    stats["n_episodes"] = len(np.unique(data["spawn_indices"]))

    # State statistics
    stats["state_dim"] = data["states"].shape[1]
    stats["state_mean"] = data["states"].mean(axis=0).tolist()
    stats["state_std"] = data["states"].std(axis=0).tolist()
    stats["state_min"] = data["states"].min(axis=0).tolist()
    stats["state_max"] = data["states"].max(axis=0).tolist()

    # Action statistics (2D action space)
    stats["action_schedule_unique"] = int(len(np.unique(data["actions"][:, 0])))
    stats["action_assignment_unique"] = int(len(np.unique(data["actions"][:, 1])))
    stats["action_pairs_unique"] = int(len(np.unique(data["actions"], axis=0)))
    stats["action_schedule_counts"] = np.bincount(data["actions"][:, 0]).tolist()
    stats["action_assignment_counts"] = np.bincount(data["actions"][:, 1]).tolist()

    # Reward statistics
    stats["reward_mean"] = float(data["rewards"].mean())
    stats["reward_std"] = float(data["rewards"].std())
    stats["reward_min"] = float(data["rewards"].min())
    stats["reward_max"] = float(data["rewards"].max())
    stats["reward_median"] = float(np.median(data["rewards"]))

    # Episode statistics
    stats["done_count"] = int(data["dones"].sum())
    stats["num_stas_unique"] = np.unique(data["num_stas"]).tolist()

    return stats


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Find JSONL files
    transitions_dir = input_dir / "transitions"
    if transitions_dir.exists():
        jsonl_pattern = str(transitions_dir / "*.jsonl")
    else:
        # Maybe JSONL files are directly in input_dir
        jsonl_pattern = str(input_dir / "*.jsonl")

    jsonl_files = sorted(glob.glob(jsonl_pattern))

    if not jsonl_files:
        print(f"Error: No JSONL files found matching {jsonl_pattern}")
        sys.exit(1)

    print(f"Found {len(jsonl_files)} JSONL files")

    # Load all transitions
    all_transitions = []
    for i, filepath in enumerate(jsonl_files):
        if args.verbose:
            print(f"Loading {filepath}...")

        transitions = load_jsonl_file(filepath)

        # Add spawn index to each transition
        spawn_idx = i + 1
        for t in transitions:
            t["spawn_idx"] = spawn_idx

        all_transitions.extend(transitions)

        if (i + 1) % 10 == 0:
            print(
                f"Loaded {i + 1}/{len(jsonl_files)} files, {len(all_transitions)} transitions so far"
            )

    print(f"Total transitions loaded: {len(all_transitions)}")

    if not all_transitions:
        print("Error: No transitions found in any file!")
        sys.exit(1)

    # Stack transitions
    print("\nStacking transitions...")
    stacked = stack_transitions(
        all_transitions, max_stas=args.max_stas, verbose=args.verbose
    )

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(stacked)

    # Print summary
    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    print(f"Total transitions: {stats['n_transitions']}")
    print(f"Number of episodes: {stats['n_episodes']}")
    print(f"State dimension: {stats['state_dim']}")
    print(f"Unique schedule actions: {stats['action_schedule_unique']}")
    print(f"Unique assignment actions: {stats['action_assignment_unique']}")
    print(f"Reward: mean={stats['reward_mean']:.4f}, std={stats['reward_std']:.4f}")
    print(f"Reward range: [{stats['reward_min']:.4f}, {stats['reward_max']:.4f}]")
    print(f"Done flags: {stats['done_count']}")
    print(f"STA counts observed: {stats['num_stas_unique']}")
    print("=" * 50)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_dir / "stacked_transitions.npz"

    # Save NPZ
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        states=stacked["states"],
        actions=stacked["actions"],
        rewards=stacked["rewards"],
        next_states=stacked["next_states"],
        dones=stacked["dones"],
        num_stas=stacked["num_stas"],
        spawn_indices=stacked["spawn_indices"],
        step_indices=stacked["step_indices"],
    )

    # Save statistics
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_path}")

    print("\nDone!")
    print(f"  - NPZ file: {output_path}")
    print(f"  - Stats file: {stats_path}")


if __name__ == "__main__":
    main()

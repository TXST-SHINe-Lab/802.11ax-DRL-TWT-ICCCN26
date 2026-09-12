#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""
simple-controller.py - Simple TWT Controller using pb_twt_wrapper_py

Reads TWT configuration sequence from a JSON file and loops through them.
Uses TWTWrapper to handle all NS-3 communication.

Located in: test-scripts/
Config file: ../simple-config.json (in parent directory)
Wrapper module: ../pb_twt_wrapper_py.py (in parent directory)

Usage:
    python3.11 simple-controller.py --config simple-config.json --seed 9000
"""

import sys
import os
import json
import argparse

# Add parent directory to path for imports (wrapper is in parent dir)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from pb_twt_wrapper_py import TWTWrapper


# ANSI colors
class Colors:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"


def load_config(config_file):
    """Load TWT configuration sequence from JSON file"""
    with open(config_file, "r") as f:
        config = json.load(f)

    configs = config.get("configs", [])
    print(
        f"{Colors.GREEN}✓ Loaded {len(configs)} TWT configurations from {config_file}{Colors.RESET}"
    )
    return config


def build_action_from_config(global_config, config_entry):
    """Build action dictionary from a single config entry"""
    num_sta = global_config.get("num_sta", 16)
    num_groups = config_entry.get("num_active_twt_groups", 1)

    # Build TWT group configs
    twt_group_configs = []
    for group in config_entry.get("twt_groups", []):
        twt_group_configs.append(
            {
                "group_id": group["group_id"],
                "twt_wake_interval_ms": group.get("wake_interval_ms", 102.4),
                "twt_wake_duration_ms": group.get("wake_duration_ms", 5.0),
                "twt_sp_offset_ms": group.get("sp_offset_ms", 2.0),
                "num_stas_assigned": 0,  # Will be computed
            }
        )

    # Build STA assignments
    sta_assignments = []
    group_sta_counts = {}
    for assign in config_entry.get("sta_assignments", []):
        sta_id = assign["sta_id"]
        group_id = assign["twt_group"]
        sta_assignments.append(
            {"sta_id": sta_id, "assigned_twt_group": group_id, "enable_twt": 1}
        )
        group_sta_counts[group_id] = group_sta_counts.get(group_id, 0) + 1

    # Update num_stas_assigned in group configs
    for gc in twt_group_configs:
        gc["num_stas_assigned"] = group_sta_counts.get(gc["group_id"], 0)

    return {
        "num_sta": num_sta,
        "num_active_twt_groups": num_groups,
        "action_timestamp_ms": 0,
        "twt_group_configs": twt_group_configs,
        "sta_group_assignments": sta_assignments,
    }


def print_config_summary(config_entry, config_index, total_configs):
    """Print summary of the current TWT configuration"""
    name = config_entry.get("name", f"Config {config_index + 1}")
    print(f"\n{Colors.MAGENTA}{'='*60}")
    print(f"APPLYING CONFIG {config_index + 1}/{total_configs}: {name}")
    print(f"{'='*60}{Colors.RESET}")

    print(f"  Active TWT Groups: {config_entry.get('num_active_twt_groups', 1)}")

    for group in config_entry.get("twt_groups", []):
        print(
            f"  {Colors.YELLOW}Group {group['group_id']}:{Colors.RESET} "
            f"wake={group.get('wake_duration_ms', 5.0):.1f}ms, "
            f"interval={group.get('wake_interval_ms', 102.4):.1f}ms, "
            f"offset={group.get('sp_offset_ms', 2.0):.1f}ms"
        )

    # Show STA to group mapping
    sta_groups = {}
    for sa in config_entry.get("sta_assignments", []):
        gid = sa["twt_group"]
        if gid not in sta_groups:
            sta_groups[gid] = []
        sta_groups[gid].append(sa["sta_id"])

    for gid, stas in sorted(sta_groups.items()):
        print(f"  Group {gid} STAs: {stas}")


def main():
    parser = argparse.ArgumentParser(description="Simple TWT Controller")
    parser.add_argument(
        "--config",
        type=str,
        default="simple-config.json",
        help="Path to TWT config JSON file",
    )
    parser.add_argument(
        "--seed", type=int, default=9000, help="Random seed for NS-3 simulation"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps (safety limit)",
    )
    args = parser.parse_args()

    # Get config path (config is in same directory as this script)
    config_path = os.path.join(script_dir, args.config)

    # Load configuration
    try:
        global_config = load_config(config_path)
    except FileNotFoundError:
        print(f"{Colors.RED}Config file not found: {config_path}{Colors.RESET}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{Colors.RED}Invalid JSON in config file: {e}{Colors.RESET}")
        sys.exit(1)

    # Get config sequence
    configs = global_config.get("configs", [])
    if not configs:
        print(f"{Colors.RED}No configs found in config file!{Colors.RESET}")
        sys.exit(1)

    num_configs = len(configs)
    print(f"{Colors.CYAN}Will cycle through {num_configs} configurations{Colors.RESET}")

    # Create wrapper (handles all NS-3 communication)
    # ns3_root is auto-computed from the wrapper module's location
    wrapper = TWTWrapper(verbose=True)

    update_count = 0
    config_index = 0

    try:
        # Start simulation
        print(
            f"\n{Colors.BLUE}Starting NS-3 simulation with seed {args.seed}...{Colors.RESET}"
        )
        env = wrapper.reset(seed=args.seed)

        print(f"{Colors.GREEN}✓ Connected to NS-3{Colors.RESET}")
        print(
            f"  Initial state: {env['num_sta']} STAs, t={env['simulation_time_sec']:.2f}s"
        )

        # Control loop
        while update_count < args.max_steps:
            # Get current config (loop through sequence)
            current_config = configs[config_index]

            # Print config summary when changing configs
            if update_count == 0 or config_index != (update_count - 1) % num_configs:
                print_config_summary(current_config, config_index, num_configs)

            # Build action from current config
            action = build_action_from_config(global_config, current_config)

            update_count += 1
            print(
                f"{Colors.CYAN}[Update {update_count}] t={env['simulation_time_sec']:.2f}s - "
                f"Config {config_index + 1}/{num_configs}{Colors.RESET}"
            )

            # Send action, receive next env
            env, done = wrapper.step(action)

            if done:
                print(f"{Colors.GREEN}Simulation finished{Colors.RESET}")
                break

            # Move to next config for next update
            config_index = (config_index + 1) % num_configs

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback

        traceback.print_exc()
    finally:
        wrapper.close()

    print(f"\n{Colors.GREEN}{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total updates: {update_count}")
    print(f"  Configs cycled: {num_configs}")
    print(f"  Config file: {args.config}")
    print(f"  Check data-log/ for CSV logs:")
    print(f"    - py-wrapper-env-*.csv (environment observations)")
    print(f"    - py-wrapper-action-*.csv (actions sent)")
    print(f"{'='*60}{Colors.RESET}")


if __name__ == "__main__":
    main()

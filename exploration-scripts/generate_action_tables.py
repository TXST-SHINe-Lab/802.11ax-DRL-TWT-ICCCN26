#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""generate_action_tables.py - generate the two-dimensional action tables for TWT RL.

Creates two separate action tables:
1. Schedule Table (Dim-1): TWT group configurations (# groups, durations, offsets)
2. Assignment Table (Dim-2): STA-to-group assignment patterns (STA-agnostic)

The assignment patterns are expressed as ratios, so they apply to any number of STAs.

Writes table_schedule.json and table_assignment.json, which file_comm_env.py,
episode_file_runner.py and reward_functions.py all read at startup.

Usage:
    python3 generate_action_tables.py [--output-dir .]
"""

import json
import argparse
import os
from datetime import datetime

# --- Constants ---

BEACON_INTERVAL_MS = 102.4  # ms, the standard beacon interval
MAX_TOTAL_DURATION_MS = 95.0  # ms, below the beacon interval to leave guard time
MIN_DURATION_MS = 8.0  # ms, minimum wake duration per group
MIN_OFFSET_GAP_MS = 2.0  # Minimum gap between groups for switching


def generate_schedule_table(num_schedules=20):
    """
    Generate Schedule Table (Dim-1): TWT group configurations.

    Each schedule defines:
    - Number of active groups (1, 2, 3, or 4)
    - Wake duration per group
    - SP offset per group (when the group wakes up)

    Returns list of schedule configs (STA-agnostic).
    """
    schedules = []
    schedule_id = 0

    # --- Category 1: Single Group (5 schedules) ---
    # All STAs share one wake window - varies duration
    single_group_durations = [90, 80, 60, 40, 20]  # ms

    for duration in single_group_durations:
        schedule = {
            "schedule_id": schedule_id,
            "name": f"S{schedule_id}_G1_D{duration}",
            "description": f"1 group, {duration}ms duration",
            "num_groups": 1,
            "total_duration_ms": duration,
            "groups": [
                {
                    "group_id": 0,
                    "wake_duration_ms": float(duration),
                    "sp_offset_ms": 0.0,
                }
            ],
        }
        schedules.append(schedule)
        schedule_id += 1

    # --- Category 2: Two Groups - Equal Split (5 schedules) ---
    # Divide wake time equally between 2 groups
    two_group_total_durations = [90, 80, 60, 40, 30]  # total duration

    for total_dur in two_group_total_durations:
        per_group_dur = total_dur / 2
        offset_g1 = per_group_dur + MIN_OFFSET_GAP_MS

        schedule = {
            "schedule_id": schedule_id,
            "name": f"S{schedule_id}_G2_D{int(per_group_dur)}x2",
            "description": f"2 groups, {per_group_dur}ms each, equal split",
            "num_groups": 2,
            "total_duration_ms": total_dur,
            "groups": [
                {"group_id": 0, "wake_duration_ms": per_group_dur, "sp_offset_ms": 0.0},
                {
                    "group_id": 1,
                    "wake_duration_ms": per_group_dur,
                    "sp_offset_ms": offset_g1,
                },
            ],
        }
        schedules.append(schedule)
        schedule_id += 1

    # --- Category 3: Two Groups - Unequal Split (4 schedules) ---
    # One group gets more time (e.g., for high-traffic STAs)
    unequal_splits = [
        (60, 20),  # G0 gets 60ms, G1 gets 20ms
        (50, 30),  # G0 gets 50ms, G1 gets 30ms
        (40, 20),  # G0 gets 40ms, G1 gets 20ms
        (30, 15),  # G0 gets 30ms, G1 gets 15ms
    ]

    for dur0, dur1 in unequal_splits:
        offset_g1 = dur0 + MIN_OFFSET_GAP_MS

        schedule = {
            "schedule_id": schedule_id,
            "name": f"S{schedule_id}_G2_D{dur0}+{dur1}",
            "description": f"2 groups unequal: G0={dur0}ms, G1={dur1}ms",
            "num_groups": 2,
            "total_duration_ms": dur0 + dur1,
            "groups": [
                {"group_id": 0, "wake_duration_ms": float(dur0), "sp_offset_ms": 0.0},
                {
                    "group_id": 1,
                    "wake_duration_ms": float(dur1),
                    "sp_offset_ms": offset_g1,
                },
            ],
        }
        schedules.append(schedule)
        schedule_id += 1

    # --- Category 4: Three Groups (3 schedules) ---
    # For more fine-grained traffic differentiation
    three_group_configs = [
        (30, 30, 25),  # Equal-ish split
        (40, 25, 20),  # Prioritize G0
        (25, 25, 25),  # Equal split, shorter
    ]

    for dur0, dur1, dur2 in three_group_configs:
        offset_g1 = dur0 + MIN_OFFSET_GAP_MS
        offset_g2 = offset_g1 + dur1 + MIN_OFFSET_GAP_MS

        schedule = {
            "schedule_id": schedule_id,
            "name": f"S{schedule_id}_G3_D{dur0}+{dur1}+{dur2}",
            "description": f"3 groups: {dur0}+{dur1}+{dur2}ms",
            "num_groups": 3,
            "total_duration_ms": dur0 + dur1 + dur2,
            "groups": [
                {"group_id": 0, "wake_duration_ms": float(dur0), "sp_offset_ms": 0.0},
                {
                    "group_id": 1,
                    "wake_duration_ms": float(dur1),
                    "sp_offset_ms": offset_g1,
                },
                {
                    "group_id": 2,
                    "wake_duration_ms": float(dur2),
                    "sp_offset_ms": offset_g2,
                },
            ],
        }
        schedules.append(schedule)
        schedule_id += 1

    # --- Category 5: Four Groups (3 schedules) ---
    # Maximum differentiation
    four_group_configs = [
        (22, 22, 22, 22),  # Equal split
        (30, 20, 20, 15),  # Prioritize G0
        (20, 20, 15, 15),  # Shorter total
    ]

    for dur0, dur1, dur2, dur3 in four_group_configs:
        offset_g1 = dur0 + MIN_OFFSET_GAP_MS
        offset_g2 = offset_g1 + dur1 + MIN_OFFSET_GAP_MS
        offset_g3 = offset_g2 + dur2 + MIN_OFFSET_GAP_MS

        schedule = {
            "schedule_id": schedule_id,
            "name": f"S{schedule_id}_G4",
            "description": f"4 groups: {dur0}+{dur1}+{dur2}+{dur3}ms",
            "num_groups": 4,
            "total_duration_ms": dur0 + dur1 + dur2 + dur3,
            "groups": [
                {"group_id": 0, "wake_duration_ms": float(dur0), "sp_offset_ms": 0.0},
                {
                    "group_id": 1,
                    "wake_duration_ms": float(dur1),
                    "sp_offset_ms": offset_g1,
                },
                {
                    "group_id": 2,
                    "wake_duration_ms": float(dur2),
                    "sp_offset_ms": offset_g2,
                },
                {
                    "group_id": 3,
                    "wake_duration_ms": float(dur3),
                    "sp_offset_ms": offset_g3,
                },
            ],
        }
        schedules.append(schedule)
        schedule_id += 1

    return schedules


def generate_assignment_table(num_assignments=20):
    """
    Generate Assignment Table (Dim-2): STA-to-group assignment patterns.

    Each assignment defines a PATTERN for distributing STAs to groups.
    Patterns are STA-agnostic (work with any number of STAs).

    Pattern types:
    - "all_to_one": All STAs to one group
    - "round_robin": Distribute evenly in round-robin
    - "split_half": First half to G0, second half to G1
    - "split_quarters": Split into 4 quarters
    - "modulo": STA_id % num_groups
    - "weighted": Weighted distribution (more STAs to earlier groups)

    Returns list of assignment patterns.
    """
    assignments = []
    assign_id = 0

    # --- Pattern 1: All to One Group (4 patterns for different group targets) ---
    for target_group in range(4):
        assignment = {
            "assignment_id": assign_id,
            "name": f"A{assign_id}_all_to_G{target_group}",
            "description": f"All STAs assigned to group {target_group}",
            "pattern_type": "all_to_one",
            "target_group": target_group,
            "max_groups_needed": 1,
        }
        assignments.append(assignment)
        assign_id += 1

    # --- Pattern 2: Round Robin for N groups (4 patterns) ---
    for num_groups in [2, 3, 4]:
        assignment = {
            "assignment_id": assign_id,
            "name": f"A{assign_id}_roundrobin_G{num_groups}",
            "description": f"Round-robin across {num_groups} groups",
            "pattern_type": "round_robin",
            "num_groups": num_groups,
            "max_groups_needed": num_groups,
        }
        assignments.append(assignment)
        assign_id += 1

    # --- Pattern 3: Split Half (2 patterns - different orderings) ---
    # First half to G0, second half to G1
    assignment = {
        "assignment_id": assign_id,
        "name": f"A{assign_id}_split_half_ordered",
        "description": "First half to G0, second half to G1",
        "pattern_type": "split_half",
        "ordering": "sequential",
        "max_groups_needed": 2,
    }
    assignments.append(assignment)
    assign_id += 1

    # Even STA IDs to G0, odd to G1
    assignment = {
        "assignment_id": assign_id,
        "name": f"A{assign_id}_split_half_evenodd",
        "description": "Even STA IDs to G0, odd to G1",
        "pattern_type": "split_half",
        "ordering": "even_odd",
        "max_groups_needed": 2,
    }
    assignments.append(assignment)
    assign_id += 1

    # --- Pattern 4: Split into Thirds (2 patterns) ---
    assignment = {
        "assignment_id": assign_id,
        "name": f"A{assign_id}_split_thirds",
        "description": "Split STAs into 3 equal groups sequentially",
        "pattern_type": "split_n",
        "num_groups": 3,
        "ordering": "sequential",
        "max_groups_needed": 3,
    }
    assignments.append(assignment)
    assign_id += 1

    assignment = {
        "assignment_id": assign_id,
        "name": f"A{assign_id}_modulo_3",
        "description": "STA_id mod 3 determines group",
        "pattern_type": "modulo",
        "num_groups": 3,
        "max_groups_needed": 3,
    }
    assignments.append(assignment)
    assign_id += 1

    # --- Pattern 5: Split into Quarters (2 patterns) ---
    assignment = {
        "assignment_id": assign_id,
        "name": f"A{assign_id}_split_quarters",
        "description": "Split STAs into 4 equal groups sequentially",
        "pattern_type": "split_n",
        "num_groups": 4,
        "ordering": "sequential",
        "max_groups_needed": 4,
    }
    assignments.append(assignment)
    assign_id += 1

    assignment = {
        "assignment_id": assign_id,
        "name": f"A{assign_id}_modulo_4",
        "description": "STA_id mod 4 determines group",
        "pattern_type": "modulo",
        "num_groups": 4,
        "max_groups_needed": 4,
    }
    assignments.append(assignment)
    assign_id += 1

    # --- Pattern 6: Weighted Distributions (4 patterns) ---
    # More STAs to earlier groups (priority groups)
    weighted_ratios = [
        [0.75, 0.25],  # 75% to G0, 25% to G1
        [0.6, 0.4],  # 60% to G0, 40% to G1
        [0.5, 0.3, 0.2],  # Decreasing priority
        [0.4, 0.3, 0.2, 0.1],  # 4-group weighted
    ]

    for ratios in weighted_ratios:
        assignment = {
            "assignment_id": assign_id,
            "name": f"A{assign_id}_weighted_G{len(ratios)}",
            "description": f"Weighted: {[int(r*100) for r in ratios]}% distribution",
            "pattern_type": "weighted",
            "ratios": ratios,
            "max_groups_needed": len(ratios),
        }
        assignments.append(assignment)
        assign_id += 1

    # --- Pattern 7: Interleaved Patterns (2 patterns) ---
    # Designed for specific interference patterns
    assignment = {
        "assignment_id": assign_id,
        "name": f"A{assign_id}_interleave_2",
        "description": "Interleave: 0,2,4,6->G0; 1,3,5,7->G1 pattern",
        "pattern_type": "interleave",
        "step": 2,
        "num_groups": 2,
        "max_groups_needed": 2,
    }
    assignments.append(assignment)
    assign_id += 1

    assignment = {
        "assignment_id": assign_id,
        "name": f"A{assign_id}_interleave_4",
        "description": "Interleave with step 4 across 2 groups",
        "pattern_type": "interleave",
        "step": 4,
        "num_groups": 2,
        "max_groups_needed": 2,
    }
    assignments.append(assignment)
    assign_id += 1

    return assignments


def apply_assignment_pattern(pattern, num_sta, num_available_groups):
    """
    Apply an assignment pattern to generate concrete STA-to-group mappings.

    Args:
        pattern: Assignment pattern dict
        num_sta: Number of STAs to assign
        num_available_groups: Number of groups available from schedule

    Returns:
        List of (sta_id, group_id) tuples
    """
    pattern_type = pattern["pattern_type"]
    assignments = []

    # Cap the groups needed by what's available
    max_groups = min(pattern.get("max_groups_needed", 1), num_available_groups)

    if pattern_type == "all_to_one":
        target = pattern.get("target_group", 0) % num_available_groups
        for sta_id in range(num_sta):
            assignments.append((sta_id, target))

    elif pattern_type == "round_robin":
        num_groups = min(pattern.get("num_groups", 2), num_available_groups)
        for sta_id in range(num_sta):
            assignments.append((sta_id, sta_id % num_groups))

    elif pattern_type == "split_half":
        ordering = pattern.get("ordering", "sequential")
        if ordering == "sequential":
            half = num_sta // 2
            for sta_id in range(num_sta):
                group = 0 if sta_id < half else min(1, num_available_groups - 1)
                assignments.append((sta_id, group))
        else:  # even_odd
            for sta_id in range(num_sta):
                group = 0 if sta_id % 2 == 0 else min(1, num_available_groups - 1)
                assignments.append((sta_id, group))

    elif pattern_type == "split_n":
        num_groups = min(pattern.get("num_groups", 2), num_available_groups)
        per_group = num_sta // num_groups
        for sta_id in range(num_sta):
            group = min(sta_id // max(1, per_group), num_groups - 1)
            assignments.append((sta_id, group))

    elif pattern_type == "modulo":
        num_groups = min(pattern.get("num_groups", 2), num_available_groups)
        for sta_id in range(num_sta):
            assignments.append((sta_id, sta_id % num_groups))

    elif pattern_type == "weighted":
        ratios = pattern.get("ratios", [1.0])
        # Adjust ratios if we have fewer groups available
        if len(ratios) > num_available_groups:
            # Redistribute excess to last available group
            new_ratios = ratios[: num_available_groups - 1]
            new_ratios.append(sum(ratios[num_available_groups - 1 :]))
            ratios = new_ratios

        # Normalize ratios
        total_ratio = sum(ratios)
        normalized = [r / total_ratio for r in ratios]

        # Assign STAs based on ratios
        sta_idx = 0
        for group_id, ratio in enumerate(normalized):
            count = int(round(ratio * num_sta))
            for _ in range(count):
                if sta_idx < num_sta:
                    assignments.append((sta_idx, group_id))
                    sta_idx += 1

        # Handle any remaining STAs
        while sta_idx < num_sta:
            assignments.append((sta_idx, len(normalized) - 1))
            sta_idx += 1

    elif pattern_type == "interleave":
        step = pattern.get("step", 2)
        num_groups = min(pattern.get("num_groups", 2), num_available_groups)
        for sta_id in range(num_sta):
            group = (sta_id // step) % num_groups
            assignments.append((sta_id, group))

    else:
        # Default: all to group 0
        for sta_id in range(num_sta):
            assignments.append((sta_id, 0))

    return assignments


def main():
    parser = argparse.ArgumentParser(
        description="Generate Two-Dimensional Action Tables for TWT RL"
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Output directory for JSON files"
    )
    parser.add_argument(
        "--num-schedules",
        type=int,
        default=20,
        help="Number of schedule options (Dim-1)",
    )
    parser.add_argument(
        "--num-assignments",
        type=int,
        default=20,
        help="Number of assignment patterns (Dim-2)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate Schedule Table (Dim-1)
    print(f"Generating Schedule Table (Dim-1)...")
    schedules = generate_schedule_table(args.num_schedules)

    schedule_table = {
        "description": "TWT Schedule Table (Dim-1) - Group configurations",
        "generated": timestamp,
        "beacon_interval_ms": BEACON_INTERVAL_MS,
        "max_total_duration_ms": MAX_TOTAL_DURATION_MS,
        "num_schedules": len(schedules),
        "schedules": schedules,
    }

    schedule_file = os.path.join(args.output_dir, "table_schedule.json")
    with open(schedule_file, "w") as f:
        json.dump(schedule_table, f, indent=2)
    print(f"  Saved {len(schedules)} schedules to {schedule_file}")

    # Generate Assignment Table (Dim-2)
    print(f"Generating Assignment Table (Dim-2)...")
    assignments = generate_assignment_table(args.num_assignments)

    assignment_table = {
        "description": "TWT Assignment Table (Dim-2) - STA-to-group patterns (STA-agnostic)",
        "generated": timestamp,
        "num_assignments": len(assignments),
        "assignments": assignments,
    }

    assignment_file = os.path.join(args.output_dir, "table_assignment.json")
    with open(assignment_file, "w") as f:
        json.dump(assignment_table, f, indent=2)
    print(f"  Saved {len(assignments)} assignment patterns to {assignment_file}")

    # Print summary
    print(f"\n" + "=" * 60)
    print("ACTION SPACE SUMMARY")
    print("=" * 60)
    print(f"  Dim-1 (Schedules):   {len(schedules)} options")
    print(f"  Dim-2 (Assignments): {len(assignments)} patterns")
    print(f"  Total Action Space:  {len(schedules) * len(assignments)} combinations")
    print("=" * 60)

    # Print schedule breakdown
    print(f"\nSchedule Breakdown:")
    group_counts = {}
    for s in schedules:
        ng = s["num_groups"]
        group_counts[ng] = group_counts.get(ng, 0) + 1
    for ng in sorted(group_counts.keys()):
        print(f"  {ng}-group schedules: {group_counts[ng]}")

    # Print assignment breakdown
    print(f"\nAssignment Pattern Types:")
    pattern_counts = {}
    for a in assignments:
        pt = a["pattern_type"]
        pattern_counts[pt] = pattern_counts.get(pt, 0) + 1
    for pt, count in sorted(pattern_counts.items()):
        print(f"  {pt}: {count}")


if __name__ == "__main__":
    main()

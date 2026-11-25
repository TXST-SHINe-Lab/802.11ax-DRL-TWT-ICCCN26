#!/usr/bin/env python3
"""hw_constraints.py - Morse Micro MM6108 HaLow bench constraints for the TWT action space.

Single source of truth for what the bench can and cannot execute. Everything else in
hw-constrained/ (the env mask, the episode runner, the published-policy clipping, and the
bench wrapper later on) reads its limits from here.

Constraints (EE 7331 proposal, Sections 1, 2 and 5):
  * One service period (SP) may last at most 65.28 ms on the MM6108 driver.
  * On hardware, SP durations are rounded UP to multiples of 256 us.
  * Only schedules whose every group window fits under 65.28 ms are legal. With the
    ICCCN'26 table that removes schedules 0 (90 ms) and 1 (80 ms), leaving 18 x 19 = 342
    legal joint actions. The mask is derived from table_schedule.json, never hardcoded, so a
    changed table cannot silently desynchronize it; tests assert the {0, 1} result.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

# --- Bench limits -------------------------------------------------------------------------

MAX_SP_DURATION_MS = 65.28  # Longest single TWT service period the MM6108 AP driver accepts
SP_GRANULARITY_US = 256  # Hardware wake-duration unit
BEACON_INTERVAL_MS = 102.4  # TWT wake interval used by the ICCCN'26 tables (ns-3 side)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TABLE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "exploration-scripts"))


# --- Action tables ------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_tables(table_dir: str = _TABLE_DIR) -> Tuple[Dict, Dict]:
    """Load (schedule_table, assignment_table) exactly as FileCommEnv does."""
    with open(os.path.join(table_dir, "table_schedule.json")) as f:
        schedule_table = json.load(f)
    with open(os.path.join(table_dir, "table_assignment.json")) as f:
        assignment_table = json.load(f)
    return schedule_table, assignment_table


def schedule_durations(schedule_idx: int) -> List[float]:
    """Per-group wake durations (ms) of one schedule, in group order."""
    schedule_table, _ = load_tables()
    return [float(g["wake_duration_ms"]) for g in schedule_table["schedules"][schedule_idx]["groups"]]


def num_schedules() -> int:
    return len(load_tables()[0]["schedules"])


def num_assignments() -> int:
    return len(load_tables()[1]["assignments"])


# --- Legality and masks -------------------------------------------------------------------


def is_schedule_legal(schedule_idx: int, max_sp_ms: float = MAX_SP_DURATION_MS) -> bool:
    """A schedule is legal when every one of its group windows fits under the SP limit.

    The comparison uses the hardware-rounded duration, so a window that would only exceed
    the limit after rounding up to 256 us is also rejected.
    """
    return all(round_up_to_granularity_ms(d) <= max_sp_ms + 1e-9 for d in schedule_durations(schedule_idx))


@lru_cache(maxsize=4)
def legal_schedules(max_sp_ms: float = MAX_SP_DURATION_MS) -> Tuple[int, ...]:
    return tuple(i for i in range(num_schedules()) if is_schedule_legal(i, max_sp_ms))


def illegal_schedules(max_sp_ms: float = MAX_SP_DURATION_MS) -> Tuple[int, ...]:
    legal = set(legal_schedules(max_sp_ms))
    return tuple(i for i in range(num_schedules()) if i not in legal)


def schedule_mask(max_sp_ms: float = MAX_SP_DURATION_MS) -> np.ndarray:
    """Boolean mask over the schedule head, True = legal."""
    legal = set(legal_schedules(max_sp_ms))
    return np.array([i in legal for i in range(num_schedules())], dtype=bool)


def multidiscrete_action_mask(max_sp_ms: float = MAX_SP_DURATION_MS) -> np.ndarray:
    """Flat mask in the layout sb3-contrib MaskablePPO expects for MultiDiscrete([20, 19]).

    MaskablePPO concatenates one boolean vector per head: the first 20 entries cover the
    schedule head, the next 19 the assignment head. Every assignment stays legal; the
    assignment patterns already cap themselves to the groups a schedule provides.
    """
    return np.concatenate([schedule_mask(max_sp_ms), np.ones(num_assignments(), dtype=bool)])


def num_legal_joint_actions(max_sp_ms: float = MAX_SP_DURATION_MS) -> int:
    return len(legal_schedules(max_sp_ms)) * num_assignments()


# --- Clipping illegal actions (published ICCCN'26 policy baseline) ------------------------


@lru_cache(maxsize=64)
def nearest_legal_schedule(schedule_idx: int, max_sp_ms: float = MAX_SP_DURATION_MS) -> int:
    """Map a schedule to the nearest legal one; legal schedules map to themselves.

    "Nearest" is defined so that the structure of the schedule changes as little as possible:
      1. Prefer legal schedules with the same number of groups.
      2. Among those, minimize the L1 distance between sorted group-duration vectors.
      3. Ties go to the shorter total awake time (the energy-preferring side), then the
         lower index, so the mapping is deterministic.
    If no legal schedule shares the group count, fall back to all legal schedules and compare
    total awake time instead. For the ICCCN'26 table this maps 0 (90 ms) and 1 (80 ms) to
    schedule 2 (single 60 ms window).
    """
    if is_schedule_legal(schedule_idx, max_sp_ms):
        return schedule_idx

    src = sorted(schedule_durations(schedule_idx))
    legal = legal_schedules(max_sp_ms)
    same_k = [j for j in legal if len(schedule_durations(j)) == len(src)]

    if same_k:
        def key(j):
            dst = sorted(schedule_durations(j))
            return (sum(abs(a - b) for a, b in zip(src, dst)), sum(dst), j)
        return min(same_k, key=key)

    def key_total(j):
        return (abs(sum(src) - sum(schedule_durations(j))), sum(schedule_durations(j)), j)
    return min(legal, key=key_total)


def clip_action(action: Sequence[int], max_sp_ms: float = MAX_SP_DURATION_MS) -> Tuple[int, int]:
    """Clip a [schedule_idx, assignment_idx] action to the nearest legal one."""
    s = int(np.clip(int(action[0]), 0, num_schedules() - 1))
    a = int(np.clip(int(action[1]), 0, num_assignments() - 1))
    return nearest_legal_schedule(s, max_sp_ms), a


# --- Hardware duration rounding -----------------------------------------------------------


def round_up_to_granularity_ms(duration_ms: float, granularity_us: int = SP_GRANULARITY_US) -> float:
    """Round a wake duration up to the next multiple of the hardware unit (256 us)."""
    units = math.ceil(round(duration_ms * 1000.0 / granularity_us, 9))
    return units * granularity_us / 1000.0


def hardware_wake_duration_us(duration_ms: float) -> int:
    """Wake duration in microseconds as the bench wrapper will write it to twt_ap_update."""
    return int(round(round_up_to_granularity_ms(duration_ms) * 1000.0))


if __name__ == "__main__":
    print(f"Schedules: {num_schedules()}, assignments: {num_assignments()}")
    print(f"Illegal schedules (> {MAX_SP_DURATION_MS} ms SP): {illegal_schedules()}")
    for s in illegal_schedules():
        n = nearest_legal_schedule(s)
        print(f"  {s} {schedule_durations(s)} ms -> {n} {schedule_durations(n)} ms")
    print(f"Legal joint actions: {num_legal_joint_actions()} of {num_schedules() * num_assignments()}")

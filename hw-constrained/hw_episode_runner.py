#!/usr/bin/env python3
"""hw_episode_runner.py - ppo-sb3-scripts/episode_file_runner.py with the bench constraints.

The ICCCN'26 runner is reused unchanged except for two patched module attributes:

  * build_action_dict: every schedule index passes through hw_constraints.nearest_legal_schedule
    first. Masked agents never send an illegal schedule, so for them this only changes the
    hardcoded warmup action (schedule 0, a 90 ms window) to schedule 2 (60 ms). It is a
    last line of defence, so that no illegal window ever reaches ns-3 from this runner.
  * read_action: the original waits 60 s for the next action; with 25 s epochs and an LSTM on
    a laptop CPU that is cut too fine, so the wait comes from --action-timeout.

Usage is identical to the original runner, plus --action-timeout (seconds).
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..", "ppo-sb3-scripts")))

import hw_constraints as hwc  # noqa: E402
import episode_file_runner as efr  # noqa: E402

_original_build_action_dict = efr.build_action_dict
_original_read_action = efr.read_action
_ACTION_TIMEOUT_S = 600.0


def _legal_build_action_dict(schedule_idx, assignment_idx, schedule_table, assignment_table, num_sta):
    legal_idx = hwc.nearest_legal_schedule(int(schedule_idx))
    return _original_build_action_dict(legal_idx, assignment_idx, schedule_table, assignment_table, num_sta)


def _read_action_long(comm_dir, timeout=60.0):
    return _original_read_action(comm_dir, timeout=max(timeout, _ACTION_TIMEOUT_S))


efr.build_action_dict = _legal_build_action_dict
efr.read_action = _read_action_long


def main():
    global _ACTION_TIMEOUT_S
    parser = argparse.ArgumentParser(description="Constraint-aware episode file runner")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--comm-dir", type=str, required=True)
    parser.add_argument("--action-timeout", type=float, default=600.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _ACTION_TIMEOUT_S = args.action_timeout
    efr.run_episode(seed=args.seed, warmup_steps=args.warmup_steps, comm_dir=args.comm_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()

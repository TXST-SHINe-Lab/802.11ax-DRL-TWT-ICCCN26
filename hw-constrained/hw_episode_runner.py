#!/usr/bin/env python3
"""hw_episode_runner.py - ppo-sb3-scripts/episode_file_runner.py with the bench constraints.

The ICCCN'26 runner is reused unchanged except for two patched module attributes:

  * build_action_dict: every schedule index passes through hw_constraints.nearest_legal_schedule
    first. Masked agents never send an illegal schedule, so for them this only changes the
    hardcoded warmup action (schedule 0, a 90 ms window) to schedule 2 (60 ms). It is a
    last line of defence, so that no illegal window ever reaches ns-3 from this runner.
  * read_action: the original waits 60 s for the next action; with 25 s epochs and an LSTM on
    a laptop CPU that is cut too fine, so the wait comes from --action-timeout.

Two process-level fixes, both needed only because runs now go in parallel:

  * Shared-memory names. pb_twt_wrapper_py.TWTWrapper.reset names the ns3-ai segments
    MySeg_<seed>, ... and first deletes any segment with that name. Parallel training runs all
    evaluate on the same 10 eval-curve seeds, at the same step counts, so two simulations with
    one seed would share (and delete) each other's segments. With TWT_SHM_TAG set (hw_env sets
    it to the parent's pid), reset() is re-created from its own source with the tag appended
    to the four names. Nothing else in it changes, and it stops if the source does not match.
  * ns-3 output. ns3ai_utils.run_single_ns3 pipes ns-3's stdout/stderr to the runner when
    show_output is False and nothing ever reads the pipe; a full pipe (64 KB) would block ns-3.
    With TWT_NS3_LOG set, that output goes to the file instead (one per episode, in comm_dir).

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

_SHM_NAME_FSTRINGS = ('f"MySeg_{seed}"', 'f"MyCpp2PyMsg_{seed}"', 'f"MyPy2CppMsg_{seed}"', 'f"MyLockable_{seed}"')


def tag_shm_names(tag: str) -> None:
    """Re-create TWTWrapper.reset with '_<tag>' appended to its four shared-memory names."""
    import inspect
    import textwrap

    import pb_twt_wrapper_py as ptw

    src = textwrap.dedent(inspect.getsource(ptw.TWTWrapper.reset))
    for name in _SHM_NAME_FSTRINGS:
        if src.count(name) != 1:
            raise RuntimeError(f"TWTWrapper.reset: expected {name} exactly once; the wrapper changed")
        src = src.replace(name, name[:-1] + '_{_HW_SHM_TAG}"', 1)
    namespace = dict(vars(ptw))
    namespace["_HW_SHM_TAG"] = tag
    exec(compile(src, ptw.__file__, "exec"), namespace)  # noqa: S102  (its own source, 4 names changed)
    ptw.TWTWrapper.reset = namespace["reset"]


def log_ns3_output(log_path: str) -> None:
    """Send ns-3's output to log_path instead of an unread pipe (see the module docstring)."""
    import subprocess

    import ns3ai_utils

    original = ns3ai_utils.run_single_ns3

    def run_single_ns3(path, pname, setting=None, env=None, show_output=False):
        if show_output:
            return original(path, pname, setting=setting, env=env, show_output=True)
        # Same command and environment as ns3ai_utils.run_single_ns3, output to a file.
        env = dict(env or {})
        env.update(os.environ)
        env["LD_LIBRARY_PATH"] = os.path.abspath(os.path.join(path, "build", "lib"))
        exec_path = os.path.join(path, "ns3")
        if not setting:
            cmd = "{} run {}".format(exec_path, pname)
        else:
            cmd = "{} run {} --{}".format(exec_path, pname, ns3ai_utils.get_setting(setting))
        out = open(log_path, "w")
        proc = subprocess.Popen(cmd, shell=True, text=True, env=env, stdin=subprocess.PIPE,
                                stdout=out, stderr=subprocess.STDOUT, preexec_fn=os.setpgrp)
        return cmd, proc

    ns3ai_utils.run_single_ns3 = run_single_ns3


if os.environ.get("TWT_SHM_TAG"):
    tag_shm_names(os.environ["TWT_SHM_TAG"])
if os.environ.get("TWT_NS3_LOG"):
    log_ns3_output(os.environ["TWT_NS3_LOG"])


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

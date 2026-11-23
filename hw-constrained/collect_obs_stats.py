#!/usr/bin/env python3
"""collect_obs_stats.py - Fit the fixed observation z-score from a random legal policy.

Proposal, Table 1: the 13 AP-observable features are z-scored with statistics from a
random-policy run. This script runs uniform-random LEGAL actions on its own seed range
(seeds.OBS_STATS_BASE, never used for training or evaluation), pools every active STA's raw
feature row, and writes obs_stats.json for train_hw.py --obs-stats.

Rebuild ns-3 with the bench configuration first (ns3-patch/apply_hw_patch.py), so the
statistics match the setting the agents are trained in.

    python3 collect_obs_stats.py --episodes 30 --out obs_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import seeds as seedplan  # noqa: E402
from ap_observation import FEATURE_NAMES, PER_STA_FEATURES, FixedZScore  # noqa: E402
from hw_env import HwConstrainedTWTEnv  # noqa: E402
from policies import RandomLegalPolicy  # noqa: E402

MOCK_RUNNER = os.path.join(_THIS_DIR, "testing", "mock_ns3_runner.py")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--epoch-s", type=float, default=25.0)
    p.add_argument("--step-timeout", type=float, default=900.0)
    p.add_argument("--out", default=os.path.join(_THIS_DIR, "obs_stats.json"))
    p.add_argument("--comm-dir", default=f"/tmp/twt_hw_obsstats_{os.getpid()}")
    p.add_argument("--python", default=None)
    p.add_argument("--mock", action="store_true", help="Test-only stand-in runner")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    env = HwConstrainedTWTEnv(
        comm_dir=args.comm_dir, num_epochs=args.num_epochs, warmup_steps=args.warmup_steps,
        epoch_s=args.epoch_s, step_timeout=args.step_timeout, spawn_timeout=args.step_timeout,
        obs_stats_path=None, runner_script=MOCK_RUNNER if args.mock else None,
        python_executable=args.python, verbose=args.verbose,
    )
    policy = RandomLegalPolicy(seed=0)
    rows = []
    try:
        for i, seed in enumerate(seedplan.obs_stats_seeds(args.episodes)):
            env.reset(seed=seed)
            n = env.num_sta
            rows.append(env.last_raw_obs.reshape(-1, PER_STA_FEATURES)[:n])
            done = False
            while not done:
                _, _, term, trunc, _ = env.step(policy.act(None, env))
                rows.append(env.last_raw_obs.reshape(-1, PER_STA_FEATURES)[:n])
                done = term or trunc
            print(f"episode {i + 1}/{args.episodes} (seed {seed}) done")
    finally:
        env.close()

    data = np.concatenate(rows, axis=0)
    zs = FixedZScore.fit(data)
    zs.save(args.out, meta={
        "episodes": args.episodes,
        "rows": int(data.shape[0]),
        "seeds": seedplan.obs_stats_seeds(args.episodes),
        "runner": "mock (TEST ONLY)" if args.mock else "ns-3",
        "epoch_s": args.epoch_s,
        "num_epochs": args.num_epochs,
    })
    for name, m, s in zip(FEATURE_NAMES, zs.mean, zs.std):
        print(f"  {name:32s} mean {m:14.4g}  std {s:14.4g}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""evaluation.py - Run a policy on a fixed list of episode seeds and collect per-episode metrics.

Every policy (trained agents, fixed schedules, random) goes through run_episodes(), so all of
them face identical traffic when given the same seeds (proposal, Section 4). Returned rows hold
the episode return plus ground-truth totals (energy, bytes, drops, packets delivered at the AP)
taken from the simulator for evaluation only; the policy never observes them.
"""

from __future__ import annotations

import json
import time
from typing import Dict, Iterable, List, Optional


def run_episodes(policy, env, seeds: Iterable[int], log_path: Optional[str] = None,
                 extra: Optional[Dict] = None) -> List[Dict]:
    rows = []
    for seed in seeds:
        t0 = time.time()
        obs, _ = env.reset(seed=int(seed))
        policy.reset()
        actions = []
        done = False
        info = {}
        while not done:
            action = policy.act(obs, env)
            obs, _, terminated, truncated, info = env.step(action)
            actions.append(list(info.get("executed_action", action)))
            done = terminated or truncated
        row = dict(info.get("episode_metrics", {}))
        row.update({"policy": getattr(policy, "name", type(policy).__name__), "actions": actions,
                    "wall_s": round(time.time() - t0, 2)})
        if extra:
            row.update(extra)
        rows.append(row)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    return rows

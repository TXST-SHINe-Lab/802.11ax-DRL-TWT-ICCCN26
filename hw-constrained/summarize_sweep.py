#!/usr/bin/env python3
"""summarize_sweep.py - Pick each agent's learning rate from the sweep runs.

Reads runs/*_sweep/eval_curve.jsonl, scores every (agent, lr) by the mean over its 3 sweep
seeds of the LAST evaluation's mean return (deterministic policy, eval-curve seeds), prints the
table, and writes sweep_choice.json, which launch_final.sh reads. Sweep seeds and their
episodes are never reused in the final runs, so tuning does not leak into the comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(r"^(mlp|lstm)_lr([0-9.e+-]+)_seed(\d+)_sweep$")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(_THIS_DIR, "runs"))
    p.add_argument("--out", default=os.path.join(_THIS_DIR, "sweep_choice.json"))
    args = p.parse_args(argv)

    scores = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(args.runs, "*_sweep"))):
        m = PAT.match(os.path.basename(d))
        f = os.path.join(d, "eval_curve.jsonl")
        if not m or not os.path.exists(f) or not os.path.exists(os.path.join(d, "model_final.zip")):
            continue
        lines = [json.loads(l) for l in open(f) if l.strip()]
        if lines:
            scores[(m.group(1), float(m.group(2)))].append(lines[-1]["mean_return"])

    choice = {}
    for algo in ("mlp", "lstm"):
        rows = sorted((lr, v) for (a, lr), v in scores.items() if a == algo)
        if not rows:
            continue
        print(f"\n{algo}:")
        for lr, v in rows:
            print(f"  lr={lr:<8g} runs={len(v)}  final mean return {np.mean(v):8.3f} +- {np.std(v):.3f}  {np.round(v, 3).tolist()}")
        complete = [(lr, v) for lr, v in rows if len(v) >= 3] or rows
        best_lr = max(complete, key=lambda r: np.mean(r[1]))[0]
        choice[algo] = {"learning_rate": best_lr, "scores": {str(lr): v for lr, v in rows}}
        print(f"  -> chosen lr {best_lr:g}")

    with open(args.out, "w") as f:
        json.dump(choice, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

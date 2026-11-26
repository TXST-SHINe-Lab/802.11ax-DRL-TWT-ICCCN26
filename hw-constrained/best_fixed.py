#!/usr/bin/env python3
"""best_fixed.py - Find the best fixed legal schedule (proposal, Section 3.1, baseline 1).

"The best fixed legal schedule, chosen by simulating all 342 legal actions on training
episodes." Two stages, both on the fixed-search seed range (training-distribution traffic,
never used for training, the learning curves or the test):

  stage 1  every legal [schedule, assignment] on the same --stage1-episodes seeds (default 3);
           all actions see identical traffic, so they are ranked by mean return
  stage 2  the --top best of stage 1 on --stage2-episodes NEW seeds (default 10), so the
           winner is not just the luckiest of 342 noisy estimates; the best mean over
           stage-2 episodes wins

Writes results/best_fixed.json, which evaluate.py's "best-fixed" spec and analyze.py read.
Episodes are appended to results/fixed_search.jsonl, so an interrupted search resumes.

With 5 STAs many of the 342 legal actions send ns-3 the same command (for a one-group schedule
all 19 assignments put every STA in group 0), so only one action of each identical group, 141
in total, is simulated; the others would give the same episodes, seed for seed.

Cost with 25 s epochs, 6 jobs: stage 1 = 141 x 3 = 423 episodes (about 1.7 h), stage 2 =
10 x 10 = 100 episodes (about 25 min).

    python best_fixed.py --jobs 6
    python best_fixed.py --jobs 6 --mock --stage1-episodes 1 --top 3 --stage2-episodes 2   # test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import hw_constraints as hwc  # noqa: E402
import seeds as seedplan  # noqa: E402
from evaluate import add_common_args, evaluate  # noqa: E402


def distinct_actions(legal, num_sta: int = 5):
    """{representative spec: [equivalent (schedule, assignment), ...]} for identical ns-3 commands."""
    sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..", "exploration-scripts")))
    from generate_action_tables import apply_assignment_pattern

    schedules, assignments = hwc.load_tables()
    reps, out = {}, {}
    for s in legal:
        k = schedules["schedules"][s]["num_groups"]
        for a in range(hwc.num_assignments()):
            mapping = tuple(sorted(apply_assignment_pattern(assignments["assignments"][a], num_sta, k)))
            rep = reps.setdefault((s, mapping), f"fixed:{s}:{a}")
            out.setdefault(rep, []).append([s, a])
    return out


def load_returns(path: str, specs, seeds):
    want, out = set(specs), defaultdict(dict)
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if r.get("ok") and r["spec"] in want and int(r["seed"]) in seeds:
                    out[r["spec"]][int(r["seed"])] = r
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-sta", type=int, default=5, help="STAs in the ns-3 build (for de-duplication)")
    p.add_argument("--stage1-episodes", type=int, default=3)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--stage2-episodes", type=int, default=10)
    p.add_argument("--out", default=os.path.join(_THIS_DIR, "results", "fixed_search.jsonl"))
    p.add_argument("--result", default=os.path.join(_THIS_DIR, "results", "best_fixed.json"))
    add_common_args(p)
    args = p.parse_args(argv)

    legal = hwc.legal_schedules()
    groups = distinct_actions(legal, args.num_sta)
    specs = sorted(groups, key=lambda sp: tuple(int(x) for x in sp.split(":")[1:]))
    n_all = len(legal) * hwc.num_assignments()
    print(f"{n_all} legal fixed actions ({len(legal)} schedules x {hwc.num_assignments()} assignments), "
          f"{len(specs)} distinct for {args.num_sta} STAs")

    s1 = seedplan.fixed_search_seeds(args.stage1_episodes)
    evaluate(specs, s1, args)
    r1 = load_returns(args.out, specs, set(s1))
    complete = {sp: v for sp, v in r1.items() if len(v) == len(s1)}
    if len(complete) < len(specs):
        raise SystemExit(f"stage 1 incomplete ({len(complete)}/{len(specs)} actions); re-run to finish")
    rank1 = sorted(complete, key=lambda sp: -np.mean([r["return"] for r in complete[sp].values()]))
    top = rank1[: args.top]
    print("\nStage 1 top actions (mean return over the same seeds):")
    for sp in top:
        print(f"  {sp:<14} {np.mean([r['return'] for r in complete[sp].values()]):8.3f}")

    s2 = seedplan.fixed_search_seeds(args.stage2_episodes, start=args.stage1_episodes)
    evaluate(top, s2, args)
    r2 = load_returns(args.out, top, set(s2))
    if any(len(r2.get(sp, {})) < len(s2) for sp in top):
        raise SystemExit("stage 2 incomplete; re-run to finish")

    def summary(sp):
        rows = list(r2[sp].values())
        rets = [r["return"] for r in rows]
        return {
            "spec": sp, "mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
            "energy_mj": float(np.mean([r["energy_mj"] for r in rows])),
            "packets_rx_ap": float(np.mean([r["packets_rx_ap"] for r in rows])),
            "stage1_mean_return": float(np.mean([r["return"] for r in complete[sp].values()])),
        }

    table = sorted((summary(sp) for sp in top), key=lambda r: -r["mean_return"])
    print(f"\nStage 2 ({len(s2)} new seeds):")
    for r in table:
        print(f"  {r['spec']:<14} return {r['mean_return']:8.3f} +- {r['std_return']:.3f}   "
              f"energy {r['energy_mj']:9.1f} mJ   delivered {r['packets_rx_ap']:9.0f}")
    best = table[0]
    _, s, a = best["spec"].split(":")
    result = {
        "schedule": int(s), "assignment": int(a),
        "wake_durations_ms": hwc.schedule_durations(int(s)),
        "equivalent_actions": groups[best["spec"]],
        "stage1_seeds": s1, "stage2_seeds": s2, "stage2": table,
        "note": "Chosen on fixed-search seeds only; never on eval-curve or test seeds.",
    }
    os.makedirs(os.path.dirname(args.result), exist_ok=True)
    with open(args.result, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nBest fixed legal schedule: schedule {s}, assignment {a} "
          f"(wake {result['wake_durations_ms']} ms). Wrote {args.result}")


if __name__ == "__main__":
    main()

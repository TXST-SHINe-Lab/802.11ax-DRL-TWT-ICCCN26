#!/usr/bin/env python3
"""evaluate.py - Run any set of policies on a fixed list of ns-3 seeds, in parallel.

Proposal, Section 4: every final policy, the published policy and the baselines are run on the
same 50 held-out test episodes, so every policy faces identical traffic. This script does that
(and the smaller jobs around it: the baselines on the 10 eval-curve seeds for the learning-curve
reference lines, and the best-fixed-schedule search in best_fixed.py).

Policy specs (--policy, repeatable):
    run:<dir>                a train_hw.py run directory (MLP-PPO-HW or LSTM-PPO-HW)
    runs:<glob>              every run directory with a final model matching the glob;
                             "runs:runs/*_seed*[0-9]" = all final runs (sweep dirs end in _sweep)
    fixed:<schedule>:<assignment>
    best-fixed               the action in results/best_fixed.json (best_fixed.py)
    md1                      the M/D/1 analytical scheduler of [5], legal actions only
    published:<model.zip>[:<vecnormalize.pkl>]   the ICCCN'26 MLP-PPO, illegal actions clipped
    random                   uniform over legal actions

Seeds (--seeds): test (50 held-out), eval-curve (10), fixed-search, or a comma list.

Each (policy, seed) episode is one task; JOBS tasks run at once, one per physical core. Rows go
to --out as JSON lines; finished (policy, seed) pairs are skipped, so an interrupted evaluation
resumes where it stopped. Two tasks may simulate the same seed at the same time: each process
tags its shared-memory names with its pid (hw_episode_runner.py).

Examples:
    python evaluate.py --seeds test --jobs 6 --out results/test.jsonl \\
        --policy "runs:runs/*_seed*[0-9]" \\
        --policy best-fixed --policy md1 --policy published:published/model_final.zip
    python evaluate.py --seeds eval-curve --jobs 6 --out results/eval_curve_baselines.jsonl \\
        --policy best-fixed --policy md1 --policy published:published/model_final.zip
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import seeds as seedplan  # noqa: E402

MOCK_RUNNER = os.path.join(_THIS_DIR, "testing", "mock_ns3_runner.py")
BEST_FIXED_JSON = os.path.join(_THIS_DIR, "results", "best_fixed.json")


# --- policy specs ---------------------------------------------------------------------------


def expand_specs(specs: List[str]) -> List[str]:
    """Turn runs:<glob> into one run:<dir> per final model, keep the rest."""
    out = []
    for s in specs:
        if s.startswith("runs:"):
            dirs = sorted(d for d in glob.glob(s[5:]) if os.path.exists(os.path.join(d, "model_final.zip")))
            if not dirs:
                raise SystemExit(f"{s}: no run directory with model_final.zip matches")
            out += [f"run:{os.path.abspath(d)}" for d in dirs]
        elif s == "best-fixed":
            if not os.path.exists(BEST_FIXED_JSON):
                raise SystemExit(f"best-fixed: {BEST_FIXED_JSON} not found; run best_fixed.py first")
            b = json.load(open(BEST_FIXED_JSON))
            out.append(f"fixed:{b['schedule']}:{b['assignment']}")
        else:
            out.append(s)
    return out


def spec_label(spec: str) -> str:
    kind = spec.split(":", 1)[0]
    if kind == "run":
        d = spec[4:]
        cfg = json.load(open(os.path.join(d, "config.json")))
        algo = cfg["args"]["algo"].upper()
        return f"{algo}-PPO-HW seed {cfg['args']['seed']}"
    if kind == "fixed":
        _, s, a = spec.split(":")
        return f"fixed s{s} a{a}"
    return {"md1": "M/D/1", "random": "random legal"}.get(kind, "Published MLP-PPO" if kind == "published" else spec)


def build_policy(spec: str):
    import policies

    kind, _, rest = spec.partition(":")
    if kind == "run":
        return policies.load_run_agent(rest)
    if kind == "fixed":
        s, a = rest.split(":")
        return policies.FixedSchedulePolicy(int(s), int(a))
    if kind == "md1":
        return policies.MD1Policy()
    if kind == "published":
        model, _, vecnorm = rest.partition(":")
        return policies.PublishedPolicy(model, vecnorm or None)
    if kind == "random":
        return policies.RandomLegalPolicy(seed=0)
    raise ValueError(f"unknown policy spec {spec!r}")


def env_settings(spec: str, args) -> Dict:
    """Env settings for one policy. Trained agents get the exact settings they trained with."""
    s = {
        "obs_stats_path": args.obs_stats,
        "num_epochs": args.num_epochs,
        "warmup_steps": args.warmup_steps,
        "epoch_s": args.epoch_s,
        "reward_type": args.reward_preset,
        "illegal_action": "clip" if spec.startswith("published:") else "raise",
    }
    if spec.startswith("run:"):
        a = json.load(open(os.path.join(spec[4:], "config.json")))["args"]
        stats = a.get("obs_stats")
        if stats and not os.path.isabs(stats):
            stats = os.path.join(_THIS_DIR, stats)
        s.update(obs_stats_path=stats, num_epochs=a["num_epochs"], warmup_steps=a["warmup_steps"],
                 epoch_s=a["epoch_s"], reward_type=a["reward_preset"])
    return s


# --- worker -----------------------------------------------------------------------------------


def _run_task(task: Tuple[str, int, Dict, Dict]) -> Dict:
    spec, seed, settings, opts = task
    import torch

    torch.set_num_threads(1)
    from evaluation import run_episodes
    from hw_env import HwConstrainedTWTEnv

    comm = os.path.join(opts["comm_root"], f"w{os.getpid()}_{seed}")
    env = HwConstrainedTWTEnv(
        comm_dir=comm, step_timeout=opts["step_timeout"], spawn_timeout=opts["step_timeout"],
        runner_script=MOCK_RUNNER if opts["mock"] else None, python_executable=opts["python"],
        **settings,
    )
    try:
        policy = build_policy(spec)
        row = run_episodes(policy, env, [seed])[0]
        row.update({"spec": spec, "label": spec_label(spec), "seed": int(seed), "ok": True})
    except Exception as e:  # noqa: BLE001
        row = {"spec": spec, "label": spec_label(spec), "seed": int(seed), "ok": False,
               "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
    finally:
        env.close()
    return row


def resolve_seeds(name: str, n: int = None) -> List[int]:
    if name == "test":
        return seedplan.test_seeds(n or seedplan.N_TEST)
    if name == "eval-curve":
        return seedplan.eval_curve_seeds(n or seedplan.N_EVAL_CURVE)
    if name == "fixed-search":
        return seedplan.fixed_search_seeds(n or 3)
    return [int(x) for x in name.split(",") if x.strip()]


def done_pairs(path: str) -> set:
    out = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if r.get("ok"):
                    out.add((r["spec"], int(r["seed"])))
    return out


def evaluate(specs: List[str], seeds: List[int], args) -> List[Dict]:
    """Run every (spec, seed) not yet in args.out; returns the new rows."""
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    if args.obs_stats and not os.path.exists(args.obs_stats):
        # Only trained agents need the z-score, and they use the one recorded in their config.
        args.obs_stats = None
    finished = done_pairs(args.out)
    opts = {"comm_root": args.comm_root or f"/tmp/twt_eval_{os.getpid()}", "step_timeout": args.step_timeout,
            "mock": args.mock, "python": args.python}
    # Seed-major order spreads the same policy over different workers; any order is safe.
    tasks = [(sp, sd, env_settings(sp, args), opts) for sd in seeds for sp in specs if (sp, sd) not in finished]
    print(f"{len(specs)} policies x {len(seeds)} seeds: {len(tasks)} episodes to run "
          f"({len(specs) * len(seeds) - len(tasks)} already in {args.out}), {args.jobs} at a time", flush=True)
    if not tasks:
        return []
    rows, t0 = [], time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, maxtasksperchild=20) as pool, open(args.out, "a") as f:
        for i, row in enumerate(pool.imap_unordered(_run_task, tasks), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            el = time.time() - t0
            status = f"return {row['return']:.3f}" if row["ok"] else f"FAILED {row['error']}"
            print(f"[{i}/{len(tasks)}] {row['label']:<24} seed {row['seed']}  {status}  "
                  f"(eta {el / i * (len(tasks) - i) / 3600:.1f} h)", flush=True)
    failed = [r for r in rows if not r["ok"]]
    if failed:
        print(f"{len(failed)} episodes failed; re-run the same command to retry them. First error:\n"
              f"{failed[0]['traceback']}")
    return rows


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--jobs", type=int, default=6, help="Parallel episodes (one per physical core)")
    p.add_argument("--obs-stats", default=os.path.join(_THIS_DIR, "obs_stats.json"),
                   help="Observation z-score for non-trained policies (trained runs use their own)")
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--epoch-s", type=float, default=25.0)
    p.add_argument("--reward-preset", default="energy")
    p.add_argument("--step-timeout", type=float, default=900.0)
    p.add_argument("--comm-root", default=None)
    p.add_argument("--python", default=None)
    p.add_argument("--mock", action="store_true", help="Test-only stand-in runner instead of ns-3")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", action="append", required=True, help="Policy spec, repeatable (see above)")
    p.add_argument("--seeds", default="test", help="test | eval-curve | fixed-search | comma list")
    p.add_argument("--n", type=int, default=None, help="Number of seeds from the named set")
    p.add_argument("--out", required=True, help="JSON-lines file; existing finished rows are kept")
    add_common_args(p)
    args = p.parse_args(argv)
    evaluate(expand_specs(args.policy), resolve_seeds(args.seeds, args.n), args)


if __name__ == "__main__":
    main()

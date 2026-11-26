#!/usr/bin/env python3
"""compare_repro.py - Does a reproduced training run match the published one?

The published run directories keep hyperparams.json, training_log_*.jsonl (one line per
episode) and normalization_stats.json (VecNormalize observation mean/var), but not the weights.
This compares a reproduction against them:

  * hyperparameters: every published key must match (timestamps excluded)
  * learning curve: mean episode reward over the first and last 50 episodes, and the
    correlation of the two 25-episode moving averages
  * observation statistics: correlation of log(1+|mean|) and of log(var) over all 208
    features (scale-free, so one feature's large values do not dominate)
  * end-of-training episode metrics (energy, bytes, drops) over the last 50 episodes

It does not decide by itself: a reproduction is a fresh training run (same code, seed and
settings, different library versions), so curves should agree in level and shape, not
episode for episode.

    python compare_repro.py --published <published run dir> --repro <reproduced run dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def load_log(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "training_log_*.jsonl")))
    if not files:
        raise SystemExit(f"no training_log_*.jsonl in {run_dir}")
    rows = []
    for line in open(files[-1]):
        line = line.strip()
        if line:
            r = json.loads(line)
            if "episode" in r:
                rows.append(r)
    return rows


def moving(x, w=25):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else x


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--published", required=True)
    p.add_argument("--repro", required=True)
    p.add_argument("--out", default=None, help="Write the comparison as JSON here")
    a = p.parse_args(argv)

    out = {"published": os.path.abspath(a.published), "repro": os.path.abspath(a.repro)}

    hp_pub = json.load(open(os.path.join(a.published, "hyperparams.json")))
    hp_rep = json.load(open(os.path.join(a.repro, "hyperparams.json")))
    diffs = {k: (v, hp_rep.get(k)) for k, v in hp_pub.items() if k != "timestamp" and hp_rep.get(k) != v}
    out["hyperparam_differences"] = diffs
    print("Hyperparameters:", "identical" if not diffs else f"DIFFERENT {diffs}")

    lp, lr = load_log(a.published), load_log(a.repro)
    if not lp or not lr:
        raise SystemExit(f"no episodes logged (published {len(lp)}, reproduced {len(lr)}); training did not run")
    rp, rr = [r["reward"] for r in lp], [r["reward"] for r in lr]
    n = min(len(rp), len(rr))
    mp, mr = moving(rp[:n]), moving(rr[:n])
    corr = float(np.corrcoef(mp, mr)[0, 1]) if len(mp) > 2 else float("nan")
    out["curve"] = {
        "episodes": [len(rp), len(rr)],
        "first50_mean": [float(np.mean(rp[:50])), float(np.mean(rr[:50]))],
        "last50_mean": [float(np.mean(rp[-50:])), float(np.mean(rr[-50:]))],
        "last50_std": [float(np.std(rp[-50:])), float(np.std(rr[-50:]))],
        "moving_avg_corr": corr,
    }
    c = out["curve"]
    print(f"\n{'':28}{'published':>12}{'reproduced':>12}")
    print(f"{'episodes':28}{c['episodes'][0]:>12}{c['episodes'][1]:>12}")
    print(f"{'reward, first 50 episodes':28}{c['first50_mean'][0]:>12.3f}{c['first50_mean'][1]:>12.3f}")
    print(f"{'reward, last 50 episodes':28}{c['last50_mean'][0]:>12.3f}{c['last50_mean'][1]:>12.3f}")
    print(f"{'  (std over those 50)':28}{c['last50_std'][0]:>12.3f}{c['last50_std'][1]:>12.3f}")
    for key, label in (("total_energy_mj", "energy (mJ), last 50"), ("total_bytes_tx", "bytes sent, last 50"),
                       ("total_drops", "drops, last 50")):
        if key in lp[-1] and key in lr[-1]:
            vp = float(np.mean([r[key] for r in lp[-50:]]))
            vr = float(np.mean([r[key] for r in lr[-50:]]))
            out["curve"][key + "_last50"] = [vp, vr]
            print(f"{label:28}{vp:>12.4g}{vr:>12.4g}")
    print(f"{'moving-average correlation':28}{corr:>24.3f}")

    sp = os.path.join(a.published, "normalization_stats.json")
    sr = os.path.join(a.repro, "normalization_stats.json")
    if os.path.exists(sp) and os.path.exists(sr):
        np_, nr = json.load(open(sp)), json.load(open(sr))
        m1, m2 = np.log1p(np.abs(np.array(np_["obs_mean"]))), np.log1p(np.abs(np.array(nr["obs_mean"])))
        v1, v2 = np.log(np.array(np_["obs_var"]) + 1e-8), np.log(np.array(nr["obs_var"]) + 1e-8)
        live = (np.array(np_["obs_var"]) > 1e-6) | (np.array(nr["obs_var"]) > 1e-6)
        out["obs_stats"] = {
            "mean_corr": float(np.corrcoef(m1[live], m2[live])[0, 1]),
            "var_corr": float(np.corrcoef(v1[live], v2[live])[0, 1]),
            "features_compared": int(live.sum()),
        }
        o = out["obs_stats"]
        print(f"\nObservation statistics ({o['features_compared']} non-constant features): "
              f"mean corr {o['mean_corr']:.3f}, variance corr {o['var_corr']:.3f}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()

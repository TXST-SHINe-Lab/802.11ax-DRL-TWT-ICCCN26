#!/usr/bin/env python3
"""analyze.py - Every ns-3 table, figure and hypothesis verdict of the proposal, from the logs.

Proposal, Section 4 ("one script will regenerate every table and figure from the saved logs").
Reads only saved files, so it can be re-run at any time and on any machine:

    runs/<algo>_lr<lr>_seed<n>/eval_curve.jsonl, config.json      final training runs
    results/test.jsonl                  evaluate.py on the 50 test seeds (runs + baselines)
    results/eval_curve_baselines.jsonl  evaluate.py on the 10 eval-curve seeds (baselines)
    results/best_fixed.json             best_fixed.py

and writes results/report/:

    summary.md               every table below plus the H1 verdict, ready for the report
    table_ns3.csv            per policy: return, energy, energy per delivered packet,
                             delivered packets, throughput, drops (proposal, expected outcome 2)
    learning_curves.png/csv  IQM of the evaluation return over runs with 95% bootstrap band,
                             with the baselines on the same eval-curve seeds as reference lines
    stats.json               every number in summary.md, machine-readable

Statistics (proposal, Section 4, following Agarwal et al. [14]):
  * Return of an agent = IQM over its 10 runs of each run's mean test return (50 episodes),
    with a 95% percentile-bootstrap CI over runs (10,000 resamples).
  * Baselines = mean over the 50 test episodes, 95% bootstrap CI over episodes.
  * H1: supported if the lower CI bound of the better agent's IQM return is above the
    published policy's mean test return.
  * LSTM vs MLP: probability of improvement P(LSTM run > MLP run), bootstrap CI.
  * Sample efficiency: steps until a run's evaluation return first exceeds the best fixed
    schedule's return on the same eval-curve seeds; area under the learning curve.
  * Consistency: IQR and standard deviation of the runs' final test returns.
  * The better agent's median-return run is named: that policy goes to the bench (H2).

    python analyze.py
    python analyze.py --runs runs --results results --boot 10000
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_PAT = re.compile(r"^(mlp|lstm)_lr([0-9.e+-]+)_seed(\d+)$")
AGENTS = {"mlp": "MLP-PPO-HW", "lstm": "LSTM-PPO-HW"}


# --- statistics ---------------------------------------------------------------------------------


def iqm(x) -> float:
    """Interquartile mean: mean of the middle 50% (scipy.stats.trim_mean(x, 0.25))."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0:
        return float("nan")
    lo = int(np.floor(0.25 * n))
    hi = n - lo
    return float(np.mean(x[lo:hi])) if hi > lo else float(np.mean(x))


def bootstrap_ci(x, stat, n_boot: int, rng, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def prob_improvement(x, y) -> float:
    """P(X > Y) for one random run of each, ties count half (Mann-Whitney U / (n m))."""
    x, y = np.asarray(x, float)[:, None], np.asarray(y, float)[None, :]
    return float(np.mean((x > y) + 0.5 * (x == y)))


def poi_ci(x, y, n_boot, rng, alpha=0.05):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = [prob_improvement(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))])
         for _ in range(n_boot)]
    return float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))


# --- loading ---------------------------------------------------------------------------------------


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def load_runs(runs_dir: str) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        m = RUN_PAT.match(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "model_final.zip")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        out[m.group(1)].append({"dir": os.path.abspath(d), "seed": int(m.group(3)), "lr": float(m.group(2)),
                                "curve": load_jsonl(os.path.join(d, "eval_curve.jsonl")), "config": cfg})
    return out


def episodes_by_label(rows: List[Dict]) -> Dict[str, List[Dict]]:
    out = defaultdict(list)
    for r in rows:
        if r.get("ok"):
            out[r["label"]].append(r)
    return out


# --- report --------------------------------------------------------------------------------------


def metrics_of(eps: List[Dict], epoch_s: float, num_epochs: int) -> Dict[str, float]:
    """Per-episode means of the ground-truth metrics (simulator totals, evaluation only)."""
    def m(k):
        v = np.array([e.get(k, np.nan) for e in eps], dtype=float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    seconds = epoch_s * num_epochs
    return {
        "return": m("return"),
        "energy_mj": m("energy_mj"),
        "energy_per_delivered_packet_mj": m("energy_per_delivered_packet_mj"),
        "packets_rx_ap": m("packets_rx_ap"),
        "throughput_kbps": m("bytes_tx") * 8 / 1000 / seconds,
        "drops": m("drops"),
        "clipped_actions": m("clipped_actions"),
        "episodes": len(eps),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=os.path.join(_THIS_DIR, "runs"))
    p.add_argument("--results", default=os.path.join(_THIS_DIR, "results"))
    p.add_argument("--boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(args.results, "report")
    os.makedirs(out_dir, exist_ok=True)

    runs = load_runs(args.runs)
    test = episodes_by_label(load_jsonl(os.path.join(args.results, "test.jsonl")))
    curve_base = episodes_by_label(load_jsonl(os.path.join(args.results, "eval_curve_baselines.jsonl")))
    best_fixed = None
    bf_path = os.path.join(args.results, "best_fixed.json")
    if os.path.exists(bf_path):
        bf = json.load(open(bf_path))
        best_fixed = f"fixed s{bf['schedule']} a{bf['assignment']}"

    any_cfg = next((r["config"]["args"] for v in runs.values() for r in v), {"epoch_s": 25.0, "num_epochs": 20})
    epoch_s, num_epochs = float(any_cfg["epoch_s"]), int(any_cfg["num_epochs"])
    stats: Dict = {"epoch_s": epoch_s, "num_epochs": num_epochs, "bootstrap_resamples": args.boot}
    md: List[str] = ["# EE 7331 results (ns-3, bench-constrained)\n"]

    # ---- per-run test scores and the main table ----
    table = []
    run_scores: Dict[str, Dict[int, float]] = {}
    for algo, rs in runs.items():
        run_scores[algo] = {}
        all_eps = []
        for r in rs:
            eps = test.get(f"{AGENTS[algo]} seed {r['seed']}", [])
            if eps:
                run_scores[algo][r["seed"]] = float(np.mean([e["return"] for e in eps]))
                all_eps += eps
        scores = list(run_scores[algo].values())
        if not scores:
            continue
        row = {"policy": AGENTS[algo], "n": f"{len(scores)} runs", **metrics_of(all_eps, epoch_s, num_epochs)}
        row["return_iqm"] = iqm(scores)
        row["return_ci"] = bootstrap_ci(scores, iqm, args.boot, rng)
        table.append(row)

    baseline_labels = [l for l in test if not any(l.startswith(a + " seed") for a in AGENTS.values())]
    for label in sorted(baseline_labels, key=lambda l: (l != best_fixed, l)):
        eps = test[label]
        rets = [e["return"] for e in eps]
        name = f"Best fixed ({label[6:]})" if label == best_fixed else label
        row = {"policy": name, "n": f"{len(eps)} episodes", **metrics_of(eps, epoch_s, num_epochs)}
        row["return_iqm"] = float(np.mean(rets))
        row["return_ci"] = bootstrap_ci(rets, np.mean, args.boot, rng)
        table.append(row)

    if table:
        cols = ["policy", "n", "return_iqm", "return_ci", "energy_mj", "energy_per_delivered_packet_mj",
                "packets_rx_ap", "throughput_kbps", "drops", "clipped_actions"]
        with open(os.path.join(out_dir, "table_ns3.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in table:
                w.writerow([r[c] if c != "return_ci" else f"{r[c][0]:.4f}..{r[c][1]:.4f}" for c in cols])
        n_seeds = len({e["seed"] for eps in test.values() for e in eps})
        md += [f"## Test episodes ({n_seeds} held-out seeds)\n",
               "Agents: IQM over runs of each run's mean test return, 95% bootstrap CI over runs. "
               "Baselines: mean over episodes, 95% bootstrap CI over episodes. Other columns: mean per episode "
               f"({num_epochs} epochs x {epoch_s:g} s).\n",
               "| Policy | n | Return | 95% CI | Energy (mJ) | Energy / delivered pkt (mJ) | Delivered pkts | "
               "Throughput (kbps) | Drops | Clipped actions |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in table:
            md.append(f"| {r['policy']} | {r['n']} | {r['return_iqm']:.3f} | [{r['return_ci'][0]:.3f}, {r['return_ci'][1]:.3f}] | "
                      f"{r['energy_mj']:.0f} | {r['energy_per_delivered_packet_mj']:.4f} | {r['packets_rx_ap']:.0f} | "
                      f"{r['throughput_kbps']:.1f} | {r['drops']:.0f} | {r['clipped_actions']:.1f} |")
        md.append("")
        stats["table"] = table

    # ---- H1 ----
    agent_rows = {r["policy"]: r for r in table if r["policy"] in AGENTS.values()}
    pub = next((r for r in table if r["policy"].startswith("Published")), None)
    if agent_rows:
        better = max(agent_rows.values(), key=lambda r: r["return_iqm"])
        h1 = {"better_agent": better["policy"], "iqm": better["return_iqm"], "ci": better["return_ci"]}
        if pub:
            h1.update(published_mean=pub["return_iqm"], supported=bool(better["return_ci"][0] > pub["return_iqm"]))
            verdict = "SUPPORTED" if h1["supported"] else "NOT supported"
            md += ["## H1\n",
                   f"Better agent: **{better['policy']}**, IQM return {better['return_iqm']:.3f}, "
                   f"95% CI [{better['return_ci'][0]:.3f}, {better['return_ci'][1]:.3f}]. Published policy mean "
                   f"return {pub['return_iqm']:.3f}. H1 needs the CI's lower bound above the published mean: "
                   f"**{verdict}**.\n"]
        else:
            md += ["## H1\n", "No verdict yet: the published policy is not in the test results. Evaluate it "
                   "with `evaluate.py --seeds test --policy published:<model_final.zip>` and re-run analyze.py.\n"]
        stats["h1"] = h1
        algo = "mlp" if better["policy"] == AGENTS["mlp"] else "lstm"
        sc = run_scores[algo]
        med = sorted(sc.items(), key=lambda kv: kv[1])[(len(sc) - 1) // 2]
        stats["bench_policy"] = {"agent": better["policy"], "seed": med[0], "mean_test_return": med[1],
                                 "dir": next(r["dir"] for r in runs[algo] if r["seed"] == med[0])}
        md += [f"Policy for the bench (H2): {better['policy']} seed {med[0]}, the median-return run "
               f"(mean test return {med[1]:.3f}), `{stats['bench_policy']['dir']}`.\n"]

    # ---- LSTM vs MLP ----
    if run_scores.get("mlp") and run_scores.get("lstm"):
        x, y = list(run_scores["lstm"].values()), list(run_scores["mlp"].values())
        poi = prob_improvement(x, y)
        ci = poi_ci(x, y, args.boot, rng)
        stats["poi_lstm_over_mlp"] = {"p": poi, "ci": ci}
        md += ["## LSTM-PPO-HW vs MLP-PPO-HW\n",
               f"Probability of improvement P(LSTM run > MLP run) = **{poi:.2f}**, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}] "
               f"(0.5 = no difference; a CI excluding 0.5 is a significant difference).\n"]

    # ---- consistency ----
    if run_scores:
        md += ["## Consistency (spread of final test returns over runs)\n", "| Agent | runs | IQR | std | min | max |",
               "|---|---|---|---|---|---|"]
        stats["consistency"] = {}
        for algo, sc in run_scores.items():
            v = np.array(list(sc.values()))
            if len(v) == 0:
                continue
            c = {"iqr": float(np.subtract(*np.percentile(v, [75, 25]))), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                 "min": float(v.min()), "max": float(v.max()), "runs": len(v)}
            stats["consistency"][algo] = c
            md.append(f"| {AGENTS[algo]} | {c['runs']} | {c['iqr']:.3f} | {c['std']:.3f} | {c['min']:.3f} | {c['max']:.3f} |")
        md.append("")

    # ---- learning curves and sample efficiency ----
    ref = {}
    for label, eps in curve_base.items():
        name = f"Best fixed ({label[6:]})" if label == best_fixed else label
        ref[name] = float(np.mean([e["return"] for e in eps]))
    bf_ref = next((v for k, v in ref.items() if k.startswith("Best fixed")), None)
    curves = {}
    if runs:
        md += ["## Sample efficiency (eval-curve seeds)\n"]
        if bf_ref is None:
            md.append("Best fixed schedule not yet evaluated on the eval-curve seeds; run evaluate.py --seeds eval-curve.\n")
        md += ["| Agent | IQM steps to beat best fixed | 95% CI | runs that never beat it | IQM area under curve | 95% CI |",
               "|---|---|---|---|---|---|"]
        stats["sample_efficiency"] = {}
        for algo, rs in runs.items():
            per_t = defaultdict(list)
            beat, auc = [], []
            for r in rs:
                cv = sorted(r["curve"], key=lambda c: c["timesteps"])
                if not cv:
                    continue
                ts = np.array([c["timesteps"] for c in cv], float)
                ys = np.array([c["mean_return"] for c in cv], float)
                for t, y in zip(ts, ys):
                    per_t[int(t)].append(y)
                auc.append(float(getattr(np, "trapezoid", getattr(np, "trapz", None))(ys, ts) / (ts[-1] - ts[0])) if len(ts) > 1 else float(ys[0]))
                if bf_ref is not None:
                    hit = np.flatnonzero(ys > bf_ref)
                    beat.append(float(ts[hit[0]]) if len(hit) else float("nan"))
            grid = sorted(per_t)
            curves[algo] = [(t, iqm(per_t[t]), *bootstrap_ci(per_t[t], iqm, min(args.boot, 2000), rng), len(per_t[t]))
                            for t in grid]
            se = {"auc_iqm": iqm(auc), "auc_ci": bootstrap_ci(auc, iqm, args.boot, rng)}
            if beat:
                b = np.array(beat)
                # A run that never beats it counts as the full budget (censored), so the IQM is a lower bound.
                b_cens = np.where(np.isfinite(b), b, max(grid) if grid else np.nan)
                se.update(steps_to_beat_iqm=iqm(b_cens), steps_to_beat_ci=bootstrap_ci(b_cens, iqm, args.boot, rng),
                          never_beat=int(np.sum(~np.isfinite(b))))
            stats["sample_efficiency"][algo] = se
            stb = (f"{se['steps_to_beat_iqm']:.0f} | [{se['steps_to_beat_ci'][0]:.0f}, {se['steps_to_beat_ci'][1]:.0f}] | "
                   f"{se['never_beat']}/{len(beat)}") if beat else "n/a | n/a | n/a"
            md.append(f"| {AGENTS[algo]} | {stb} | {se['auc_iqm']:.3f} | [{se['auc_ci'][0]:.3f}, {se['auc_ci'][1]:.3f}] |")
        md += ["", "Area under curve = mean evaluation return over training (trapezoid, divided by the step span). "
               "Runs that never beat the best fixed schedule count as the full budget.\n"]

        with open(os.path.join(out_dir, "learning_curves.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["agent", "timesteps", "iqm_return", "ci_low", "ci_high", "runs"])
            for algo, rows in curves.items():
                for row in rows:
                    w.writerow([AGENTS[algo], *row])
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(7, 4.2))
            colors = {"mlp": "#1f77b4", "lstm": "#d62728"}
            for algo, rows in curves.items():
                t = [r[0] for r in rows]
                ax.plot(t, [r[1] for r in rows], color=colors[algo], label=f"{AGENTS[algo]} (IQM, {len(runs[algo])} runs)")
                ax.fill_between(t, [r[2] for r in rows], [r[3] for r in rows], color=colors[algo], alpha=0.2)
            styles = ["--", ":", "-."]
            for i, (name, v) in enumerate(sorted(ref.items())):
                ax.axhline(v, color="0.3", ls=styles[i % 3], lw=1, label=name)
            ax.set_xlabel("Training steps (decision epochs)")
            ax.set_ylabel(f"Evaluation return ({len(next(iter(curve_base.values()), [])) or 10} episodes)")
            ax.set_title("Learning curves, 95% bootstrap CI over runs")
            ax.legend(fontsize=8, loc="best")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=160)
            md.append("![Learning curves](learning_curves.png)\n")
        except ImportError:
            md.append("(matplotlib not installed: `pip install matplotlib` for learning_curves.png)\n")
        stats["learning_curve_references"] = ref

    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2, default=float)
    print("\n".join(md))
    print(f"\nWrote {out_dir}/summary.md, table_ns3.csv, learning_curves.csv/png, stats.json")


if __name__ == "__main__":
    main()

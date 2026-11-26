# hw-constrained: constraint-aware TWT training (EE 7331)

Training code for the project *From Simulation to HaLow Hardware: Constraint-Aware Deep
Reinforcement Learning for Target Wake Time Scheduling*. It sits inside the ICCCN'26
repository (`<ns-3.44>/contrib/ai/examples/twt/hw-constrained/`) and reuses its ns-3
simulation, file protocol, reward function and action tables unchanged.

## What is here

| File | Purpose |
|---|---|
| `hw_constraints.py` | Bench limits (65.28 ms SP, 256 µs rounding). Derives the legal-action mask from `table_schedule.json`: schedules 0 and 1 are illegal, 342 of 380 joint actions remain. Nearest-legal clipping for the published-policy baseline. |
| `ap_observation.py` | The AP-observable 13-feature observation (proposal, Table 1). The 7 ground-truth features are replaced with commanded awake/sleep/duty cycle and `packets_received_at_ap`. The builder never receives oracle data. |
| `hw_env.py` | `HwConstrainedTWTEnv`, a subclass of `FileCommEnv`: new observation, `action_masks()`, 20-epoch truncation, privileged reward (unchanged ICCCN'26 energy preset). |
| `hw_episode_runner.py` | The ICCCN'26 episode runner with illegal schedules clipped (its warmup used schedule 0, a 90 ms window) and a longer action wait. |
| `masked_lstm_policy.py` | RecurrentPPO policy that masks illegal schedule logits. |
| `train_hw.py` | One training run: `--algo mlp` (MaskablePPO) or `--algo lstm` (RecurrentPPO + mask). |
| `collect_obs_stats.py` | Fits the fixed observation z-score from a random legal policy. |
| `launch_sweep.sh`, `summarize_sweep.py` | Learning-rate sweep (3 LRs × 3 seeds × 2 agents) and LR choice. |
| `launch_final.sh` | The 20 final runs (10 seeds × 2 agents). Resumable. |
| `seeds.py` | Disjoint ns-3 seed ranges for sweep, training, observation stats, learning curves and the 50 test episodes. |
| `policies.py`, `evaluation.py` | One `act()` interface and one episode loop for every policy: the trained agents, fixed schedules, the M/D/1 scheduler of [5] (legal actions only) and the published MLP-PPO (its own simulator observation, illegal actions clipped). |
| `evaluate.py` | Runs any set of policies on a seed set (50 test, 10 eval-curve, fixed-search), 6 episodes at a time. Resumable. |
| `best_fixed.py` | Best fixed legal schedule: all 342 legal actions (141 distinct for 5 STAs) on 3 search seeds, then the top 10 on 10 new seeds. |
| `analyze.py` | Every ns-3 table, the learning-curve figure, IQM with bootstrap CIs, probability of improvement, sample efficiency, consistency, the H1 verdict and the run to take to the bench. |
| `run_eda.sh` | Ahmed's EDA (steps 1, 3, 4, 5) in parallel, for the bench reward constants. |
| `ns3-patch/apply_hw_patch.py` | Puts ns-3 in the bench configuration: 5 STAs (4 sensors + 1 steady stream), epoch length, episode length, legal-only EDA sampling, and a fix so the AP's received-packet counter works. |
| `testing/mock_ns3_runner.py` | **Test only.** A toy stand-in for ns-3 that speaks the same file protocol. Never report numbers from it. |
| `tests/` | 32 tests. |

## What has and has not been checked

Checked here (Linux, Python 3.13, torch 2.15, SB3/sb3-contrib 2.9.2, gymnasium 1.4), with the
mock runner standing in for ns-3:

- The mask is exactly schedules {0, 1}; 342 legal joint actions; 0 and 1 clip to schedule 2 (60 ms).
- The observation is identical when every oracle field is overwritten with garbage.
- Neither agent samples an illegal schedule during training or evaluation. The env raises on
  any illegal action, so a masking bug would stop the run rather than hide.
- Train → save → reload → evaluate works for both agents, including the LSTM state and the
  saved VecNormalize statistics.
- Training episodes walk the seed range in `seeds.py`, and parallel runs don't delete each
  other's shared memory.

**Not checked:**

- The bench-mix C++ compiled and ran on ns-3.44 (Ubuntu, GCC 15): one 22-epoch episode printed
  `HaLow bench mix: 4 sensors + 1 stream`. The sniffer fix below is not yet compiled.
- That first episode found two problems, both fixed:
  1. `packets_received_at_ap` was always 0. ns-3.44 passes each A-MPDU subframe to
     MonitorSnifferRx with its 4-byte subframe header in front, and the published
     `ApMonitorSnifferRxTrace` reads the MAC header from byte 0, so every HE data frame failed
     the STA lookup. `apply_hw_patch.py` now strips the subframe header (twt-trace-callbacks.cc).
  2. ns-3 outlived the episode. `ns3ai_utils` starts it in its own process group, so killing
     the runner's group missed it; `hw_env._kill_process` now kills all descendants.
- Step time per epoch is still to be measured with the fixed build.
- The public repo has the training logs of the published policy but **not the model files**
  (`*_final.zip` and `vecnormalize.pkl`). You need these from Ahmed for baseline (4), the
  clipped ICCCN'26 policy.
- macOS: the ICCCN'26 setup doesn't say which OS it was built on. If ns-3 + ns3-ai don't
  build on your Mac, run this in a Linux VM or container. Nothing here is Linux-specific.

## Run order

All commands from `hw-constrained/` unless noted.

### 0. Python environment

Use the same interpreter that has the ns3-ai Python bindings (the ICCCN'26 build uses
`python3.11`). The episode runner is started with `sys.executable`, or `--python` if given.

```bash
python3.11 -m venv ~/twt-venv && source ~/twt-venv/bin/activate
pip install "stable-baselines3>=2.3" "sb3-contrib>=2.3" gymnasium torch pytest
python -m pytest tests -q          # 27 passed, about 30 s
```

### 1. Bench configuration in ns-3

Measure what the bench carries with iperf (proposal, Section 2), then:

```bash
python hw-constrained/ns3-patch/apply_hw_patch.py --stream-rate-kbps <iperf kbps>   # from the repo root
cd <ns-3.44> && ./ns3 build twt-main-simulation
```

`--revert` restores the original files. Re-running with new values is safe.

### 2. Time one real episode first

```bash
time python collect_obs_stats.py --episodes 1 --out /tmp/one.json
```

One run simulates about 19,900 epochs: 15,104 training steps, 3,000 evaluation epochs
(15 evaluations × 10 episodes × 20 epochs) and about 1,800 warmup epochs (2 per episode).
So a run takes about `19,900 × (seconds per epoch)`; divide the time of this one episode by
22 to get seconds per epoch. If a run would exceed 12 h, re-patch with `--epoch-s 10`,
rebuild, and pass `--epoch-s 10` to every script below (proposal, Section 4).

### 3. Reward constants for the new setting

The reward z-scores come from `derived_constants.json`. The existing file was dialed for
16 STAs and 2 s epochs, so the reward would saturate in the new setting. Re-run the ICCCN'26
EDA with legal actions only. `run_eda.sh` runs Ahmed's steps 1, 3, 4 and 5 unchanged, but with
the simulations in parallel (his `1-collect-data.sh` runs them one at a time, about a day for
1,275 spawns), starts spaced 3 s apart so no two share a log file name, and the EDA's logs kept
in its own run directory:

```bash
NUM_SPAWNS=20 JOBS=6 hw-constrained/run_eda.sh    # quick test
JOBS=6 hw-constrained/run_eda.sh                  # full run, prints the export line below
export TWT_DERIVED_CONSTANTS=<twt>/exploration-scripts/eda-data/<run_id>/derived_constants.json
```

Run it alone; it refuses to start while another simulation is running.

Every training log prints the constants file it loaded, so check the path.

### 4. Observation statistics

```bash
python collect_obs_stats.py --episodes 30 --out obs_stats.json
```

### 5. Learning-rate sweep, then the final runs

```bash
JOBS=6 PY=python ./launch_sweep.sh    # 18 runs of 3000 steps; writes sweep_choice.json
JOBS=6 PY=python ./launch_final.sh    # 20 runs of 15000 steps; re-run to resume
```

`JOBS` is how many runs go at once: one per physical core (each run is one ns-3 process).

### 6. Baselines and the test evaluation (proposal, Sections 3.1 and 4)

The published model files are not in the public repo; put Ahmed's MLP-PPO energy-preset
`model_final.zip` and `vecnormalize.pkl` in `published/`.

```bash
python best_fixed.py --jobs 6                                  # ~2 h, writes results/best_fixed.json
python evaluate.py --seeds test --jobs 6 --out results/test.jsonl \
    --policy "runs:runs/*_seed*[0-9]" --policy best-fixed --policy md1 \
    --policy published:published/model_final.zip               # 23 policies x 50 episodes, ~4 h
python evaluate.py --seeds eval-curve --jobs 6 --out results/eval_curve_baselines.jsonl \
    --policy best-fixed --policy md1 --policy published:published/model_final.zip
python analyze.py                                              # results/report/summary.md
```

The M/D/1 policy reads its queueing constants from `TWT_DERIVED_CONSTANTS`, so keep that
exported. Every command resumes if interrupted.

## Outputs

Each run writes to `runs/<algo>_lr<lr>_seed<seed>/`:

- `config.json`: every argument, seed range, the mask, and library versions plus git commit
- `train_episodes.jsonl`: one row per training episode (includes exploration)
- `eval_curve.jsonl`: 10 deterministic episodes on fixed seeds, logged after each PPO update
  once 1000 more steps have passed, plus the final model
- `model_final.zip`, `vecnormalize.pkl`, `checkpoints/`

## Details worth knowing

- **Step count:** SB3 trains in whole rollouts, so 15000 steps become 15104 for both agents
  (59 × 256 and 118 × 128).
- **Masking value:** the mask uses a logit of −1e8 rather than −∞, as sb3-contrib's
  MaskablePPO does. The probability is exactly 0 in float32 and entropy stays finite.
- **Normalization:** the fixed z-score is applied first, then a running VecNormalize, as in
  ICCCN'26 ("running VecNorm on top of EDA norm"). `--no-vecnorm` turns the second off.
- **What the policy sees:** evaluation rows include ground-truth energy, bytes and drops for
  scoring only. The policy never sees them.

## Published policy (H1 baseline)

The public repository has the published runs' hyperparameters, training logs and observation
statistics, but not the trained weights. `repro/reproduce_published.sh` re-trains the published
energy-preset MLP-PPO with Ahmed's own `train_ppo_V1.py` and the published hyperparameters
(seed 1, 15,000 steps) on his original 16-STA simulator, built in a separate copy of ns-3
(`<ns-3.44>-orig`), then compares the learning curve and observation statistics with the
published logs (`repro/compare_repro.py`). The model lands in `published/mlp/`, which
`evaluate.py --policy published:published/mlp/model_final.zip` reads.

`repro/repro_train.py` runs the trainer unchanged except that an episode cleans up only its own
shared memory (the original runs `rm -f /dev/shm/My*`, which would break any other simulation on
the machine), so the reproduction can run next to the bench training.

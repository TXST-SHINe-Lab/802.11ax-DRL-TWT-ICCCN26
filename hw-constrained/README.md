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
| `policies.py`, `evaluation.py` | One `act()` interface and one episode loop for every policy, used by training-time evaluation and by later baseline comparisons. |
| `ns3-patch/apply_hw_patch.py` | Puts ns-3 in the bench configuration: 5 STAs (4 sensors + 1 steady stream), epoch length, episode length, legal-only EDA sampling. |
| `testing/mock_ns3_runner.py` | **Test only.** A toy stand-in for ns-3 that speaks the same file protocol. Never report numbers from it. |
| `tests/` | 27 tests. |

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

- The C++ part of `apply_hw_patch.py` has **not been compiled**. The patch was applied and
  reverted on a copy and the diff reviewed, but ns-3 is not built here. The first build on
  your machine is the real test.
- No ns-3 episode has been run with this code, so step time, the right `--step-timeout`, and
  whether 25 s epochs fit in 12 h per run are unknown until step 2 below.
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
EDA with legal actions only:

```bash
cd ../exploration-scripts
TWT_HW_LEGAL_ONLY=1 NUM_SPAWNS=<n> ./1-collect-data.sh
./3-prepare-data.sh && python 4-eda-analysis.py && python 5-dial-constants.py   # 5 needs 4's output
export TWT_DERIVED_CONSTANTS=$PWD/eda-data/<run_id>/derived_constants.json
```

Every training log prints the constants file it loaded, so check the path.

### 4. Observation statistics

```bash
python collect_obs_stats.py --episodes 30 --out obs_stats.json
```

### 5. Learning-rate sweep, then the final runs

```bash
JOBS=2 ./launch_sweep.sh       # 18 runs of 3000 steps; writes sweep_choice.json
JOBS=2 ./launch_final.sh       # 20 runs of 15000 steps; re-run to resume
```

`JOBS` is how many runs go at once; each ns-3 simulation uses one core.

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

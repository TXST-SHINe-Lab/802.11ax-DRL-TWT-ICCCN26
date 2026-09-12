# PPO-SB3 Scripts for TWT WiFi Scheduling

This directory contains the PPO (Proximal Policy Optimization) reinforcement learning implementation for TWT (Target Wake Time) WiFi scheduling, using Stable-Baselines3 and file-based IPC with NS-3.

## Table of Contents

1. [Files](#files)
1. [Quick Start](#quick-start)
1. [Pipeline Architecture](#pipeline-architecture)
1. [Models](#models)
1. [Reward Function](#reward-function)
1. [Training](#training)
1. [Evaluation](#evaluation)
1. [Plotting](#plotting)
1. [Output Structure](#output-structure)

---

## Files

| File                       | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| `file_comm_env.py`         | Gymnasium environment — file-based IPC with NS-3                |
| `episode_file_runner.py`   | Spawned subprocess that runs one NS-3 episode                   |
| `train_ppo_V1.py`          | MLP PPO training (SB3 `PPO`)                                    |
| `train_lstm_ppo_V1.py`     | LSTM PPO training (SB3-Contrib `RecurrentPPO`) — default        |
| `reward_functions.py`      | 6-component reward with preset weights and normalization        |
| `eval_policy.py`           | Benchmark PPO vs Random vs Analytical vs Heuristic baselines    |
| `analytical_policies.py`   | Model-based analytical baselines (imported by `eval_policy.py`) |
| `plot_training.py`         | Training curve plots from `.jsonl` logs                         |
| `plot_evaluation.py`       | Policy comparison bar charts from `eval_results/*.json`         |
| `analyze_reward_signal.py` | NORM constant validation from reward log CSVs                   |
| `run_training.sh`          | Run all three presets through the default training script       |
| `run_eval.sh`              | Run PPO + Analytical evaluation for all presets                 |
| `run_all.sh`               | End-to-end pipeline: NS-3 build → EDA → train → eval → plot     |

---

## Quick Start

### Install dependencies

```bash
pip install stable-baselines3 sb3-contrib gymnasium
```

### Generate action tables (if not present)

```bash
cd ../exploration-scripts
python3.11 generate_action_tables.py
cd ../ppo-sb3-scripts
```

### Train (all three presets, LSTM PPO)

```bash
./run_training.sh           # throughput, energy, queue with lstm_ppo_V1
```

### Or with MLP PPO

```bash
./run_training.sh --training-script train_ppo_V1.py throughput energy queue
```

### Evaluate

```bash
./run_eval.sh
```

### Plot all results

```bash
python3.11 plot_training.py
python3.11 plot_evaluation.py
```

---

## Pipeline Architecture

The RL agent and NS-3 simulator run in separate processes and communicate via JSON files in `/tmp/twt_comm/`:

```
┌───────────────────────────────────────────────────────────────┐
│                       TRAINING LOOP                           │
│                                                               │
│  run_training.sh                                              │
│       │                                                       │
│       ▼                                                       │
│  train_lstm_ppo_V1.py  (or  train_ppo_V1.py)                  │
│  ├─ SB3 / SB3-Contrib PPO agent                               │
│  └─ FileCommEnv  (file_comm_env.py)                           │
│          │                                                    │
│          │  env.reset() / env.step()                          │
│          ▼                                                    │
│  ┌───────────────────────────────────────────────────────┐    │
│  │  episode_file_runner.py  (subprocess per episode)     │    │
│  │  ├─ Loads  table_schedule.json                        │    │
│  │  │       + table_assignment.json                      │    │
│  │  ├─ TWTWrapper.reset(seed)  →  NS-3 warmup            │    │
│  │  ├─ Writes   response.json  (observation)             │    │
│  │  ├─ Waits    action.json    (agent action)            │    │
│  │  └─ TWTWrapper.step()  →  next state                  │    │
│  └───────────────────────────────────────────────────────┘    │
│          │                                                    │
│          │  FileCommEnv parses response → reward → PPO update │
└───────────────────────────────────────────────────────────────┘
```

### Action Space

`MultiDiscrete([20, 19])` — 20 schedule indices × 19 assignment indices = 380 combinations.

- **Dim-0 (schedule)**: TWT group timing (number of groups, wake durations)
- **Dim-1 (assignment)**: STA-to-group mapping patterns

Action tables are loaded from `../exploration-scripts/table_schedule.json` and `../exploration-scripts/table_assignment.json`.

### Observation Space

`Box(208,)` — 13 features × 16 STAs, built by `build_ppo_observation()` in `file_comm_env.py`:

| Index Range | Category         | Features (per STA)                                                                                   |
| ----------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| 0–4         | Realistic raw    | `bsr_queue_ac_be`, `airtime_used_us`, `fcs_error_count`, `rx_fragment_count`, `last_rx_timestamp_us` |
| 5–8         | Oracle raw (TWT) | `awake_time_ms`, `sleep_time_ms`, `duty_cycle`, `packets_transmitted`                                |
| 9–12        | Deltas           | `Δairtime_used_us`, `Δawake_time_ms`, `Δsleep_time_ms`, `Δpackets_transmitted`                       |

---

## Models

### LSTM PPO — `train_lstm_ppo_V1.py` (default)

Uses SB3-Contrib `RecurrentPPO` with separate LSTM heads for policy and value:

| Hyperparameter   | Value                                                  |
| ---------------- | ------------------------------------------------------ |
| LSTM hidden size | 256                                                    |
| LSTM layers      | 1 (separate policy + value, `enable_critic_lstm=True`) |
| MLP after LSTM   | `[256, 128]`                                           |
| Learning rate    | 2.5e-4                                                 |
| `n_steps`        | 128                                                    |
| Batch size       | 64                                                     |
| `n_epochs`       | 10                                                     |
| Clip range       | 0.2                                                    |
| Entropy coef     | 0.01                                                   |
| Activation       | GELU                                                   |

### MLP PPO — `train_ppo_V1.py`

Uses SB3 `PPO` with a custom `EnhancedFeatureExtractor`:

| Hyperparameter    | Value                                       |
| ----------------- | ------------------------------------------- |
| Feature extractor | Linear 208→256, LayerNorm, GELU (×3 blocks) |
| Policy/value net  | `[256, 256]`                                |
| Learning rate     | 3e-4                                        |
| `n_steps`         | 256                                         |
| Batch size        | 64                                          |
| `n_epochs`        | 10                                          |
| Clip range        | 0.2                                         |
| Entropy coef      | 0.01                                        |
| Activation        | GELU, orthogonal init                       |

---

## Reward Function

`reward_functions.py` computes a **weighted sum of 6 z-score-normalized components**. Each component is normalized against NORM constants regenerated from the current EDA run by `5-dial-constants.py` and loaded at import from `exploration-scripts/derived_constants.json`, then clipped and combined. There is no hardcoded fallback: a missing file or a missing required key raises, so the EDA dial step must run before training or evaluation.

### Components

| Component    | Key Metrics                                                                           | Direction |
| ------------ | ------------------------------------------------------------------------------------- | --------- |
| `throughput` | `delta_bytes_transmitted`, `delta_packets_transmitted`                                | maximize  |
| `queue`      | `queue_size_bytes`, `queue_size_packets`, `bsr_queue_index`, `delta_packets_enqueued` | minimize  |
| `energy`     | `delta_energy_mj`, `delta_awake_time_ms`, `delta_sleep_time_ms`, `duty_cycle`         | minimize  |
| `drops`      | `delta_drops_expired`                                                                 | minimize  |
| `airtime`    | total scheduled wake duration of the selected schedule                                | minimize  |
| `channel`    | `fcs_error_count`, `rx_fragment_count`                                                | mixed     |

### Preset Weights

| Component    | `throughput` | `energy` | `queue` |
| ------------ | ------------ | -------- | ------- |
| `throughput` | 0.35         | 0.20     | 0.20    |
| `queue`      | 0.20         | 0.10     | 0.35    |
| `energy`     | 0.10         | 0.35     | 0.10    |
| `drops`      | 0.15         | 0.10     | 0.20    |
| `airtime`    | 0.15         | 0.20     | 0.10    |
| `channel`    | 0.05         | 0.05     | 0.05    |

A `RewardLogger` writes per-step component breakdowns to `reward_logs/reward_log_*_part*.csv` inside the checkpoint directory for post-hoc analysis by `analyze_reward_signal.py`.

---

## Training

### Using the shell script

```bash
# All three presets with default (LSTM) script
./run_training.sh

# Specific presets only
./run_training.sh throughput energy

# MLP PPO instead of LSTM
./run_training.sh --training-script train_ppo_V1.py throughput energy queue
```

The script sets `TIMESTEPS=15000` by default; edit the variable at the top to change it.

### Direct Python invocation

```bash
# LSTM PPO
python3.11 train_lstm_ppo_V1.py \
    --reward-preset throughput \
    --total-timesteps 50000 \
    --seed 42 \
    --normalize-obs

# MLP PPO
python3.11 train_ppo_V1.py \
    --reward-preset energy \
    --total-timesteps 100000 \
    --normalize-obs

# Resume from checkpoint
python3.11 train_lstm_ppo_V1.py \
    --reward-preset queue \
    --resume checkpoints/lstm_ppo_V1_twt_queue_20260131_151302/model.zip
```

### Key training arguments

| Argument            | Default        | Description                              |
| ------------------- | -------------- | ---------------------------------------- |
| `--reward-preset`   | required       | `throughput`, `energy`, or `queue`       |
| `--total-timesteps` | 100000         | Environment steps to train for           |
| `--seed`            | 42             | Random seed                              |
| `--learning-rate`   | 2.5e-4 / 3e-4  | Adam learning rate (LSTM / MLP)          |
| `--n-steps`         | 128 / 256      | Steps per rollout (LSTM / MLP)           |
| `--batch-size`      | 64             | Mini-batch size                          |
| `--normalize-obs`   | off            | Enable `VecNormalize` (running mean/std) |
| `--resume`          | —              | Path to `.zip` checkpoint to resume from |
| `--output-dir`      | `checkpoints/` | Where to save checkpoints                |
| `--tensorboard-log` | `tb_logs/`     | TensorBoard log directory                |

---

## Evaluation

### Using the shell script

```bash
./run_eval.sh                           # All presets, LSTM PPO vs Analytical
./run_eval.sh throughput energy         # Specific presets only
./run_eval.sh --training-script train_ppo_V1.py   # MLP PPO variant
```

The script sets `INCLUDE_ANALYTICAL=true` and `N_EPISODES=50` by default.

### Direct Python invocation

```bash
# Compare LSTM PPO vs Analytical for throughput preset
python3.11 eval_policy.py \
    checkpoints/lstm_ppo_V1_twt_throughput_20260131_012841/ \
    --n-episodes 10 \
    --reward-type throughput \
    --compare-analytical throughput

# Compare against all analytical variants
python3.11 eval_policy.py \
    checkpoints/ppo_V1_twt_throughput_20260202_013615/ \
    --n-episodes 5 \
    --reward-type throughput \
    --compare-analytical throughput energy queue \
    --output-prefix my_run_
```

Results are written to `eval_results/eval_<prefix><preset>_<timestamp>.json`.

---

## Plotting

### Training curves

```bash
python3.11 plot_training.py                                    # LSTM PPO, all presets
python3.11 plot_training.py --preset throughput                # Single preset
python3.11 plot_training.py --training-script train_ppo_V1.py  # MLP PPO
```

Reads `.jsonl` training logs from the matching checkpoint directories. Generates a 3×3 multi-panel figure showing reward, entropy, and policy loss over training steps.

### Evaluation comparison

```bash
python3.11 plot_evaluation.py                                  # LSTM PPO results
python3.11 plot_evaluation.py --training-script train_ppo_V1.py
```

Reads `eval_results/eval_*.json`. Generates bar charts comparing PPO vs baselines per preset and writes `plots/*_summary_table.csv`.

### Reward signal diagnosis

```bash
python3.11 analyze_reward_signal.py    # LSTM PPO reward logs
```

Reads `reward_logs/reward_log_*_part*.csv` from checkpoint directories. Generates NORM accuracy charts, z-score distribution plots, and component contribution boxplots.

---

## Output Structure

```
checkpoints/
├── lstm_ppo_V1_twt_throughput_20260131_012841/
│   ├── model.zip                            # Final trained policy
│   ├── vecnormalize.pkl                     # VecNormalize stats (if --normalize-obs)
│   ├── hyperparams.json                     # All training hyperparameters
│   ├── normalization_stats.json             # Obs normalization statistics
│   ├── training_log_<timestamp>.jsonl       # Per-episode training metrics
│   └── reward_logs/
│       └── reward_log_<timestamp>_part*.csv # Component-level reward logs
├── lstm_ppo_V1_twt_energy_*/
├── lstm_ppo_V1_twt_queue_*/
├── ppo_V1_twt_throughput_*/                 # MLP PPO checkpoints (if trained)
└── ...

tb_logs/
├── lstm_ppo_V1_twt_throughput_*/
│   └── RecurrentPPO_1/                      # TensorBoard events
└── ppo_V1_twt_throughput_*/
    └── PPO_1/

eval_results/
├── eval_lstm_ppo_V1_twt_throughput_<timestamp>.json
├── eval_lstm_ppo_V1_twt_summary_table.csv
└── ...
```

Checkpoint directories are named `{script_name_without_train_and_py}_twt_<preset>_<YYYYMMDD_HHMMSS>`.

### TensorBoard

```bash
tensorboard --logdir tb_logs
```

---

## Algorithm Notes

- **On-policy**: PPO uses a rollout buffer, not a replay buffer; collected data is discarded after each update.
- **MultiDiscrete**: SB3 handles schedule and assignment dimensions jointly without flattening.
- **LSTM variant**: The recurrent policy maintains hidden state across steps within an episode, which can help with the partial-observability inherent in cumulative metrics.

### Observation Normalization (VecNormalize)

When `--normalize-obs` is passed, SB3's `VecNormalize` wrapper sits between the environment and the policy:

```
FileCommEnv → DummyVecEnv → VecNormalize → RecurrentPPO / PPO
                                │
                           Running mean/std
                           updated each step
                           clip_obs = 10.0
```

**During training**: `VecNormalize` computes a running mean and standard deviation over all 208 observation dimensions and normalizes each incoming observation to approximately zero mean and unit variance, clipped to ±10. The statistics are updated live as more data arrives.

**What is saved**: At the end of training, the accumulated statistics are written to `vecnormalize.pkl` alongside `model.zip` in the checkpoint directory. Without this file the policy will receive raw (unnormalized) observations and produce garbage actions.

**During evaluation**: `eval_policy.py` auto-detects `vecnormalize.pkl` in the same directory as the model, loads it with `VecNormalize.load()`, freezes the statistics (`training=False`, `norm_reward=False`), and routes observations through the wrapper before passing them to the policy. Analytical and heuristic baselines receive raw observations directly (they do not use `VecNormalize`).

**If `--normalize-obs` is not passed**: No wrapper is applied, no `.pkl` is written, and the policy sees raw sensor values directly. The custom `EnhancedFeatureExtractor` (MLP variant) includes `LayerNorm` inside the network which provides some implicit normalization in this case.

---

## Author

Ahmed Maksud <ahmed.maksud@email.ucr.edu>
SHINE Lab, Texas State University

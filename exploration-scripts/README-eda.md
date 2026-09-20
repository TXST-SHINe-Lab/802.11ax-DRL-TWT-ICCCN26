# TWT RL Exploration Scripts

This directory contains scripts for **Exploratory Data Analysis (EDA)** and offline data
collection for the TWT (Target Wake Time) scheduling RL environment. Running thousands of
random-action simulations here provides the dataset used to study the action space, validate
metric flows, and calibrate reward normalization constants before training.

## Overview

The RL agent uses a **two-dimensional discrete action space**:

- **Dimension 1 (Schedule)**: 20 TWT schedule configurations (1–4 groups, varying wake durations)
- **Dimension 2 (Assignment)**: 19 STA-to-group assignment patterns (round-robin, split, weighted, …)

Both dimensions are **STA-agnostic** — the same tables work for any number of active STAs.

---

## Files

### Action Space Definition

#### `generate_action_tables.py`

Generates the two action-table JSON files that every other script (and the PPO training code)
loads at runtime.

- **Output**: `table_schedule.json`, `table_assignment.json`
- **Schedule categories** (20 entries): 5 × single-group (90/80/60/40/20 ms), 5 × equal 2-group,
  4 × unequal 2-group, 3 × 3-group, 3 × 4-group
- **Assignment categories** (19 entries): all-to-one (G0–G3), round-robin (2/3/4 groups),
  split-half (ordered/even-odd), split-thirds/quarters, modulo-3/4, weighted distributions,
  interleaved patterns

```bash
python generate_action_tables.py --output-dir .
# Optional: --num-schedules 20 --num-assignments 20
```

#### `table_schedule.json`

Pre-generated schedule table. Each entry specifies `num_groups`, per-group
`wake_duration_ms`, and `sp_offset_ms`. Loaded by `run-single-spawn.py`,
`episode_file_runner.py` (PPO pipeline), and `reward_functions.py`.

#### `table_assignment.json`

Pre-generated assignment table. Each entry specifies a `pattern_type` and the
parameters needed to map N STAs onto the active groups at runtime. Loaded by
the same scripts as `table_schedule.json`.

---

### Data Collection

#### `run-single-spawn.py`

Runs **one** NS-3 simulation with random actions and writes all transitions to a
JSONL file. Called sequentially by `1-collect-data.sh`.

- Loads `table_schedule.json` / `table_assignment.json`

- Calls `TWTWrapper.reset()` then loops `TWTWrapper.step()` until `done=True`
  (38 steps per episode as set by `DURATION_IN_UPDATE` in `twt-constants.h`)

- At each step, extracts **per-STA state** via `extract_state_per_sta(env_dict)`:

  | Feature group            | Fields                                                                        |
  | ------------------------ | ----------------------------------------------------------------------------- |
  | BSR / queue (realistic)  | `bsr_queue_ac_be/bk/vi/vo`                                                    |
  | Link quality (realistic) | `rssi_dbm`, `snr_db`, `last_rx_mcs`                                         |
  | AP traffic (realistic)   | `bytes_received_at_ap`, `packets_received_at_ap`                             |
  | Energy (oracle)          | `oracle_total_energy_mj`, `oracle_duty_cycle`, `oracle_awake/sleep_time_ms` |
  | Packets (oracle)         | `oracle_packets_generated/transmitted`, `oracle_bytes_transmitted`           |
  | Drops (oracle)           | `oracle_mpdu_drops_expired/queue_full`                                        |
  | Queue (oracle)           | `oracle_queue_size_packets/bytes`, `oracle_avg_latency_ms`                   |

- First line of the JSONL is a `_metadata` record (seed, log timestamps, NS-3 CSV paths)

- Subsequent lines are transition dicts with keys
  `spawn_id`, `step`, `sim_time_sec`, `num_sta`, `action` ([schedule_idx, assignment_idx]),
  `schedule_name`, `assignment_name`, `state`, `next_state`, `done`, plus
  aggregate convenience fields (`agg_total_bytes_tx`, `agg_total_energy_mj`, etc.)

```bash
python run-single-spawn.py --seed 1000 --spawn-id 0 \
    --output eda-data/run_<timestamp>/transitions/spawn_0.jsonl
# --quiet   suppress per-step progress output
```

#### `1-collect-data.sh`

Batch collector: runs `NUM_SPAWNS` NS-3 simulations one after another and saves everything
under a timestamped `eda-data/run_<YYYYMMDD_HHMMSS>/` directory.

- Reads `ACTIVE_NUM_STA` directly from `twt-constants.h` as the default STA count
- Auto-generates action tables if they are missing
- Cleans up stale NS-3 processes and `/dev/shm/MySeg_*` shared memory before starting
- Saves a `run_config.json` and `collection_summary.json` alongside the transitions

```bash
./1-collect-data.sh

# Override any parameter via environment variable:
NUM_SPAWNS=50 NUM_STAS=8 STEPS_PER_SPAWN=200 BASE_SEED=42 ./1-collect-data.sh
```

| Variable          | Default                | Description                           |
| ----------------- | ---------------------- | ------------------------------------- |
| `NUM_SPAWNS`      | 1275                   | Number of sequential NS-3 runs        |
| `NUM_STAS`        | from `twt-constants.h` | STAs per simulation                   |
| `STEPS_PER_SPAWN` | 38                     | Max RL steps per run                  |
| `BASE_SEED`       | 42                     | Starting seed (incremented per spawn) |
| `CONFIG_FILE`     | _(none)_               | Optional JSON config override         |

---

### Data Processing

#### `stack-data.py`

Converts the JSONL transition files from one collection run into a single NPZ
array file for fast loading during EDA and training.

- Reads all `*.jsonl` files from a `run_*/transitions/` directory

- Pads each transition's per-STA state to `--max-stas` (default 16) so all
  transitions have the same state dimension

- Per-STA feature order (27 fields × 16 STAs = 432-dim state vector), matching
  `per_sta_feature_names` in `stack-data.py`:
  `bsr_queue_ac_be/bk/vi/vo`, `rssi_dbm`, `snr_db`, `last_rx_mcs`,
  `bytes/packets_received_at_ap`, `airtime_used_us`, `fcs_error_count`,
  `rx_fragment_count`, `last_rx_timestamp_us`, `oracle_duty_cycle`,
  `oracle_awake/sleep_time_ms`, `oracle_total_energy_mj`,
  `oracle_packets_generated/enqueued/transmitted`, `oracle_bytes_transmitted`,
  `oracle_mpdu_drops_expired/queue_full`, `oracle_psdu_response_timeouts`,
  `oracle_queue_size_packets/bytes`, `oracle_avg_latency_ms`

  `5-dial-constants.py` validates this width on load and aborts if it does not divide evenly,
  so an NPZ from an older feature list is rejected rather than silently misread.

- Writes `stacked_transitions.npz` and `stacked_transitions.stats.json`

```bash
python stack-data.py --input-dir eda-data/run_<timestamp> --max-stas 16 --verbose
```

#### `3-prepare-data.sh`

Convenience wrapper around `stack-data.py`. Finds the most recent `run_*`
directory automatically, activates the Python venv, runs `stack-data.py`, and
prints a summary of the output files.

```bash
./3-prepare-data.sh                        # process latest run
./3-prepare-data.sh eda-data/run_20260131_010432  # process specific run
```

---

### Analysis & Validation

#### `4-eda-analysis.py`

Exploratory analysis of the stacked dataset. Runs five analysis passes:

1. **Raw metric quality** — classifies every NS-3 metric column as
   `GOOD` / `LIMITED` / `GARBAGE` / `SENTINEL` based on unique-value count,
   zero-percentage, and sentinel (−128 / 65535) frequency
1. **State feature distributions** — histograms and statistics per feature
1. **Action space coverage** — heat-map of how often each (schedule, assignment)
   pair was visited
1. **Reward analysis** — distribution and temporal patterns of aggregate metrics
1. **State–action–reward relationships** — correlation matrix and scatter plots

```bash
python 4-eda-analysis.py \
    --npz-file eda-data/run_<timestamp>/stacked_transitions.npz

# Optional flags:
#   --data-log-dir ../data-log    also analyze raw wrapper CSV metrics
#   --schedule-table table_schedule.json   use human-readable schedule labels
#   --output-dir plots/           save plots to a specific directory
#   --no-plots                    text-only analysis
#   --raw-metrics-only            only run the metric-quality pass
```

#### `5-dial-constants.py`

Derives the reward `NORM` z-score table (per-metric mean, std, max) from the freshest EDA run and
writes it next to that run's data. `reward_functions.py` and `analytical_policies.py` load the file
at import and refuse to import until this step has run, so a failure here aborts the pipeline.

- **Input**: `eda-data/<run_id>/stacked_transitions.npz`
- **Output**: `eda-data/<run_id>/derived_constants.json`

```bash
python 5-dial-constants.py                      # freshest run, default output path
python 5-dial-constants.py --output <path>      # write elsewhere
```

#### `2-validate-logvstap.py`

Cross-validates a single spawn's JSONL file against the NS-3 wrapper CSV logs
generated during the same run. Checks that every per-STA metric value written
by C++ matches what the Python wrapper received and what ended up in the JSONL.

- Loads `_metadata` from the JSONL to locate the matching `py-wrapper-env-*.csv`
- Runs field-by-field comparison for BSR, RSSI, SNR, oracle metrics, and timing
- Reports mismatches with absolute and relative error

```bash
python 2-validate-logvstap.py \
    --jsonl eda-data/run_<timestamp>/transitions/spawn_0.jsonl \
    --data-log-dir ../data-log
```

---

## Data Flow

```
TWTWrapper.reset() / TWTWrapper.step()
         │
         ▼
   env_dict['sta_observations']           ← per-STA realistic + oracle dicts
         │
extract_state_per_sta()                   ← run-single-spawn.py
         │
   spawn_N.jsonl  (JSONL transitions)
         │
    stack-data.py
         │
   stacked_transitions.npz               ← used by 4-eda-analysis.py
```

The JSONL is produced by **directly reading TWTWrapper output** — it is
independent of the NS-3 CSV log files. The CSV files (`py-wrapper-env-*.csv`,
`ns3-twt-wrapper-*.csv`) are generated as a side-effect of the simulation and
are only consulted by `2-validate-logvstap.py` for cross-validation.

---

## Output Structure

```
exploration-scripts/
├── table_schedule.json          ← Dim-1 action table (20 schedules)
├── table_assignment.json        ← Dim-2 action table (19 assignments)
└── eda-data/
    └── run_YYYYMMDD_HHMMSS/
        ├── run_config.json             collection parameters
        ├── collection_summary.json     per-spawn success/failure summary
        ├── transitions/
        │   ├── spawn_0.jsonl
        │   ├── spawn_1.jsonl
        │   └── ...
        ├── logs/
        │   ├── spawn_0.log
        │   └── ...
        ├── stacked_transitions.npz
        └── stacked_transitions.stats.json
```

---

## JSONL Transition Format

Each non-metadata line in a `.jsonl` file is one RL transition:

```json
{
  "spawn_id": 0,
  "step": 5,
  "sim_time_sec": 12.288,
  "num_sta": 16,
  "action": [3, 7],
  "schedule_name": "S3_G1_D40",
  "assignment_name": "A7_split_half_ordered",
  "state":      [{"sta_id": 0, "bsr_queue_ac_be": 1024, ...}, ...],
  "next_state": [{"sta_id": 0, "bsr_queue_ac_be": 512,  ...}, ...],
  "done": false,
  "agg_total_bytes_tx": 245760,
  "agg_total_energy_mj": 18.4,
  "agg_total_drops": 2,
  "agg_mean_duty_cycle": 0.43
}
```

---

## NPZ Array Layout

After `stack-data.py` the arrays in `stacked_transitions.npz` are:

| Array           | Shape    | Dtype   | Description                                   |
| --------------- | -------- | ------- | --------------------------------------------- |
| `states`        | (N, 432) | float32 | Flattened per-STA features, padded to 16 STAs |
| `next_states`   | (N, 368) | float32 | Same for next step                            |
| `actions`       | (N, 2)   | int32   | `[schedule_idx, assignment_idx]`              |
| `num_stas`      | (N, )    | int32   | Actual STA count (before padding)             |
| `spawn_indices` | (N, )    | int32   | Episode index                                 |
| `step_indices`  | (N, )    | int32   | Step within episode                           |

---

## Quick Start

```bash
# 1. Generate action tables (only needed once)
python generate_action_tables.py --output-dir .

# 2. Collect data (NUM_SPAWNS runs, 1275 by default)
./1-collect-data.sh

# 3. Stack into NPZ (processes latest run automatically)
./3-prepare-data.sh

# 4. Run EDA
python 4-eda-analysis.py \
    --npz-file eda-data/$(ls eda-data | sort -r | head -1)/stacked_transitions.npz

# 5. Dial reward normalization constants from this EDA
python 5-dial-constants.py
```

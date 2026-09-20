# PPO Agent and Reward Function Metrics Documentation

**Author:** Ahmed Maksud\
**Lab:** SHINE Lab, Texas State University

---

## Table of Contents

1. [Overview](#overview)
1. [PPO Observation Space](#ppo-observation-space)
1. [Reward Function Metrics](#reward-function-metrics-sta_deltas)
1. [Reward Components](#reward-components)
1. [Visual Summary](#visual-summary)
1. [Data Flow](#data-flow)
1. [Excluded Metrics](#excluded-metrics)

---

## Overview

This document describes the exact metrics used by the PPO agent (observation) and reward function in the TWT WiFi scheduling RL system.

**Key Design Principle:**

- PPO observes **realistic + TWT-related oracle** metrics (what a real AP could measure)
- Reward uses **all GOOD oracle metrics** (ground truth for accurate training signal)
- Hidden oracle metrics provide training feedback without being directly observable

---

## PPO Observation Space

**Total Dimensions:** 208 (13 features × 16 STAs)

Per-metric statistics are not reproduced here, since they change with every EDA run.
Mean, standard deviation and max come from `../exploration-scripts/eda-data/<run_id>/derived_constants.json`,
written by `5-dial-constants.py` and loaded at import by `reward_functions.py`.
Unique-value counts, category and quality status come from `metric_quality_analysis.json`
in the EDA run directory.

### Realistic Raw (5 features per STA)

| #   | Metric                 | NS-3 Source | Description                                |
| --- | ---------------------- | ----------- | ------------------------------------------ |
| 1   | `bsr_queue_ac_be`      | realistic   | Buffer Status Report - Best Effort queue   |
| 2   | `airtime_used_us`      | realistic   | Cumulative airtime consumed (microseconds) |
| 3   | `fcs_error_count`      | realistic   | Frame Check Sequence errors                |
| 4   | `rx_fragment_count`    | realistic   | Received fragment count                    |
| 5   | `last_rx_timestamp_us` | realistic   | Last received frame timestamp              |

### Oracle Raw — TWT Related (4 features per STA)

| #   | Metric                | NS-3 Source | Description                                 |
| --- | --------------------- | ----------- | ------------------------------------------- |
| 6   | `awake_time_ms`       | oracle      | Cumulative awake time (milliseconds)        |
| 7   | `sleep_time_ms`       | oracle      | Cumulative sleep time (milliseconds)        |
| 8   | `duty_cycle`          | oracle      | Wake/sleep ratio [0-1]                      |
| 9   | `packets_transmitted` | oracle      | Cumulative packets successfully transmitted |

### Deltas — Computed Per-Step (4 features per STA)

| #   | Metric                      | Computation | Description                   |
| --- | --------------------------- | ----------- | ----------------------------- |
| 10  | `delta_airtime_used_us`     | curr - prev | Airtime used this step        |
| 11  | `delta_awake_time_ms`       | curr - prev | Awake time this step          |
| 12  | `delta_sleep_time_ms`       | curr - prev | Sleep time this step          |
| 13  | `delta_packets_transmitted` | curr - prev | Packets transmitted this step |

---

## Reward Function Metrics (sta_deltas)

The reward function receives `sta_deltas` containing 15 metrics per STA. These include all PPO-visible metrics plus hidden oracle metrics.

### PPO-Visible Deltas (4 metrics)

| Metric                      | Used in Reward Component |
| --------------------------- | ------------------------ |
| `delta_airtime_used_us`     | `throughput`             |
| `delta_awake_time_ms`       | `energy`                 |
| `delta_sleep_time_ms`       | `energy`                 |
| `delta_packets_transmitted` | `throughput`             |

### Hidden Oracle Deltas (4 metrics — PPO cannot see)

| Metric                    | Used in Reward Component |
| ------------------------- | ------------------------ |
| `delta_bytes_transmitted` | `throughput`             |
| `delta_energy_mj`         | `energy`                 |
| `delta_drops_expired`     | `drops`                  |
| `delta_packets_enqueued`  | `queue`                  |

### PPO-Visible Instantaneous (5 metrics)

| Metric                 | Used in Reward Component                          |
| ---------------------- | ------------------------------------------------- |
| `bsr_queue_index`      | `queue` (a BSR index in 0 through 254, not bytes) |
| `fcs_error_count`      | `channel`                                         |
| `rx_fragment_count`    | `channel`                                         |
| `last_rx_timestamp_us` | read but unused; Eq. (9) of the paper has no recency term      |
| `duty_cycle`           | `energy`                                          |

### Hidden Oracle Instantaneous (2 metrics — PPO cannot see)

| Metric               | Used in Reward Component |
| -------------------- | ------------------------ |
| `queue_size_bytes`   | `queue`                  |
| `queue_size_packets` | `queue`                  |

---

## Reward Components

### Six-Component Structure (14 of 15 Metrics Used)

Corresponds to Eqs. (4) through (9) of the paper; see `PRESET_WEIGHTS` in `reward_functions.py`.

| Component    | Metrics Used                                                                          | Description                      |
| ------------ | ------------------------------------------------------------------------------------- | -------------------------------- |
| `throughput` | `delta_bytes_transmitted`, `delta_packets_transmitted`, `delta_airtime_used_us`       | Data delivery performance        |
| `queue`      | `queue_size_bytes`, `queue_size_packets`, `bsr_queue_index`, `delta_packets_enqueued` | Buffer health, the latency proxy |
| `drops`      | `delta_drops_expired`                                                                 | Packet expiration penalty        |
| `energy`     | `delta_energy_mj`, `delta_awake_time_ms`, `delta_sleep_time_ms`, `duty_cycle`         | Power consumption                |
| `airtime`    | Total scheduled wake duration of the selected schedule                                | Penalizes long TWT schedules     |
| `channel`    | `fcs_error_count`, `rx_fragment_count`                                                | Channel quality                  |

### Preset Weights

Each component's sign is carried inside the component itself, so every weight is positive and each preset sums to 1.00.

| Component    | Throughput | Energy | Queue |
| ------------ | ---------- | ------ | ----- |
| `throughput` | 0.35       | 0.20   | 0.20  |
| `queue`      | 0.20       | 0.10   | 0.35  |
| `drops`      | 0.15       | 0.10   | 0.20  |
| `energy`     | 0.10       | 0.35   | 0.10  |
| `airtime`    | 0.15       | 0.20   | 0.10  |
| `channel`    | 0.05       | 0.05   | 0.05  |

### Reward Formula (Throughput Preset)

```
reward = 0.35 × throughput_component
       + 0.20 × queue_component
       + 0.15 × drops_component
       + 0.10 × energy_component
       + 0.15 × airtime_component
       + 0.05 × channel_component
```

### Per-STA Analysis Features

- **Jain's Fairness Index**: Applied to throughput (0.0-1.0 scale)
- **Worst-Case Penalty**: Extra penalty for worst-performing STA
- **Starvation Detection**: Penalty for STAs with zero throughput

---

## Visual Summary

```
┌──────────────────────────────────────────────────────────────────┐
│                    REWARD FUNCTION (sta_deltas)                  │
│                6 Components, 14 of 15 Metrics Used               │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              PPO OBSERVATION (13 features/STA)             │  │
│  │                                                            │  │
│  │  REALISTIC RAW (5):        ORACLE RAW (4):                 │  │
│  │  • bsr_queue_ac_be         • awake_time_ms                 │  │
│  │  • airtime_used_us         • sleep_time_ms                 │  │
│  │  • fcs_error_count         • duty_cycle                    │  │
│  │  • rx_fragment_count       • packets_transmitted           │  │
│  │  • last_rx_timestamp_us                                    │  │
│  │                                                            │  │
│  │  DELTAS (4):                                               │  │
│  │  • delta_airtime_used_us                                   │  │
│  │  • delta_awake_time_ms                                     │  │
│  │  • delta_sleep_time_ms                                     │  │
│  │  • delta_packets_transmitted                               │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  HIDDEN ORACLE (6 - PPO cannot see, reward only):                │
│  • delta_bytes_transmitted     • queue_size_bytes                │
│  • delta_energy_mj             • queue_size_packets              │
│  • delta_drops_expired                                           │
│  • delta_packets_enqueued                                        │
│                                                                  │
│  REWARD COMPONENTS:                                              │
│  [throughput] ← delta_bytes_tx, delta_packets_tx, delta_airtime  │
│  [queue]      ← queue_size_*, bsr_queue_index, packets_enqueued  │
│  [drops]      ← delta_drops_expired                              │
│  [energy]     ← delta_energy_mj, awake/sleep_time, duty_cycle    │
│  [airtime]    ← scheduled wake duration of the chosen schedule   │
│  [channel]    ← fcs_error, rx_fragment                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
NS-3 Simulation
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  env_dict (raw NS-3 data per STA)                           │
│    - realistic: bsr, airtime, fcs_error, rx_fragment, etc.  │
│    - oracle: awake, sleep, duty_cycle, energy, drops, etc.  │
└─────────────────────────────────────────────────────────────┘
      │
      ├──────────────────────────────┬─────────────────────────┐
      ▼                              ▼                         │
┌──────────────────────┐    ┌─────────────────────────┐        │
│ build_ppo_observation│    │ compute_sta_deltas      │        │
│ (file_comm_env.py)   │    │ (file_comm_env.py)      │        │
│                      │    │                         │        │
│ 13 features × 16     │    │ 15 metrics × 16 STAs    │        │
│ = 208 dimensions     │    │ (visible + hidden)      │        │
└──────────────────────┘    └─────────────────────────┘        │
      │                              │                         │
      ▼                              ▼                         │
┌─────────────────────┐    ┌─────────────────────────┐         │
│ VecNormalize        │    │ reward_functions.py     │         │
│ (running mean/std)  │    │                         │         │
│                     │    │ compute_reward()        │         │
│ Normalized obs      │    │                         │         │
└─────────────────────┘    │                         │         │
      │                    │ → single reward float   │         │
      ▼                    └─────────────────────────┘         │
┌─────────────────────┐              │                         │
│ PPO Neural Network  │              │                         │
│ 208→256×3 extractor │              │                         │
│ + [256,256] heads   │              │                         │
│                     │              │                         │
│ → action (S, A)     │              │                         │
└─────────────────────┘              │                         │
      │                              │                         │
      ▼                              ▼                         │
┌──────────────────────────────────────────────────────────────┐
│                    PPO Learning Update                       │
│   Uses: observation, action, reward, next_observation        │
└──────────────────────────────────────────────────────────────┘
```

---

## Excluded Metrics

Excluded on the basis of the EDA. `metric_recommendations.json` in the EDA run directory
carries the current classification and counts; the lists below name the metrics, not how many.

### Always Zero

- `bsr_queue_ac_bk`, `bsr_queue_ac_vi`, `bsr_queue_ac_vo`
- `power_mgmt_bit`
- `bytes_received_at_ap`, `packets_received_at_ap`
- `bytes_received_ac_*`, `packets_received_ac_*`
- `packets_generated`, `bytes_generated`
- `mpdu_drops_queue_full`
- `avg_latency_ms`, `queue_delay_sum_ms`, `queue_delay_count`
- `position_z_m`

### Limited Variance

- `is_active`, `tx_power_dbm`, `last_rx_mcs`, etc.

> Under the radio configuration used in the paper (Nakagami zones + AP/STA power), the
> link-quality metrics `rssi_dbm`, `snr_db`, `rcpi`, `rsni`, `link_margin_db`
> carry enough variance to no longer count as low-variance. Check their current
> `status` in `metric_quality_analysis.json` rather than assuming.

---

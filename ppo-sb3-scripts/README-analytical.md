# Analytical Policies for TWT WiFi Scheduling

This document describes the **analytical model-based policies** used as baselines for comparing against PPO (reinforcement learning). These policies use queuing theory and WiFi physics to make deterministic scheduling decisions.

## Table of Contents

- [Analytical Policies for TWT WiFi Scheduling](#analytical-policies-for-twt-wifi-scheduling)
  - [Table of Contents](#table-of-contents)
  - [Architecture Overview](#architecture-overview)
    - [Key Design Principle](#key-design-principle)
  - [Core Components](#core-components)
    - [TWT\_Schedules - Action Space Mapping](#twt_schedules---action-space-mapping)
      - [Key Methods](#key-methods)
    - [QueuingModel - M/D/1 Queue Theory](#queuingmodel---md1-queue-theory)
      - [Calibration from the EDA](#calibration-from-the-eda)
      - [Service Rate Estimation](#service-rate-estimation)
      - [Queue Dynamics](#queue-dynamics)
    - [EnergyModel - Power Consumption](#energymodel---power-consumption)
      - [Power Constants (from NS-3 simulation)](#power-constants-from-ns-3-simulation)
      - [Key Methods](#key-methods-1)
    - [ThroughputModel - Collision Estimation](#throughputmodel---collision-estimation)
      - [Collision Probability](#collision-probability)
      - [Effective Throughput](#effective-throughput)
  - [Policy Implementations](#policy-implementations)
  - [Airtime Efficiency](#airtime-efficiency)
    - [Airtime Reward Values](#airtime-reward-values)
  - [Author](#author)

---

## Architecture Overview

The analytical policies follow this decision flow:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANALYTICAL POLICY FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Raw Observation (16 STAs × 13 features = 208 values)          │
│                          │                                      │
│                          ▼                                      │
│              ┌───────────────────┐                              │
│              │ ObservationParser │  Extract queue, FCS, etc.    │
│              └─────────┬─────────┘                              │
│                        │                                        │
│            ┌───────────┼───────────┐                            │
│            ▼           ▼           ▼                            │
│    ┌─────────────┐ ┌───────────┐ ┌───────────────┐              │
│    │ QueuingModel│ │EnergyModel│ │ThroughputModel│              │
│    │   (M/D/1)   │ │  (Power)  │ │ (Collision)   │              │
│    └──────┬──────┘ └─────┬─────┘ └──────┬────────┘              │
│           │              │              │                       │
│           └──────────────┼──────────────┘                       │
│                          ▼                                      │
│              ┌───────────────────────┐                          │
│              │ Policy-Specific Logic │                          │
│              │ (Throughput/Energy/   │                          │
│              │  Queue optimization)  │                          │
│              └───────────┬───────────┘                          │
│                          ▼                                      │
│                 Action: [schedule_idx, assignment_idx]          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principle

Unlike PPO which learns from trial-and-error, analytical policies use **pre-computed mathematical models** calibrated from observed training data. They receive **raw observations** (not normalized) and apply domain-specific formulas to select actions.

---

## Core Components

### TWT_Schedules - Action Space Mapping

Maps `schedule_idx` (0-19) to actual TWT parameters loaded from `table_schedule.json`.
The 1- and 2-group schedules are shown below; IDs 10-19 cover the unequal 2-group, 3-group
and 4-group configurations. `table_schedule.json` is the authority for all 20.

| schedule_idx | Groups | Duration per Group | Total Duration | Name        |
| ------------ | ------ | ------------------ | -------------- | ----------- |
| 4            | 1      | [20ms]             | 20ms           | S4_G1_D20   |
| 3            | 1      | [40ms]             | 40ms           | S3_G1_D40   |
| 2            | 1      | [60ms]             | 60ms           | S2_G1_D60   |
| 1            | 1      | [80ms]             | 80ms           | S1_G1_D80   |
| 0            | 1      | [90ms]             | 90ms           | S0_G1_D90   |
| 9            | 2      | [15ms, 15ms]       | 30ms           | S9_G2_D15x2 |
| 8            | 2      | [20ms, 20ms]       | 40ms           | S8_G2_D20x2 |
| 7            | 2      | [30ms, 30ms]       | 60ms           | S7_G2_D30x2 |
| 6            | 2      | [40ms, 40ms]       | 80ms           | S6_G2_D40x2 |
| 5            | 2      | [45ms, 45ms]       | 90ms           | S5_G2_D45x2 |

#### Key Methods

```python
TWT_Schedules.get_total_duration(schedule_idx) -> float
# Returns total wake duration in ms

TWT_Schedules.get_num_groups(schedule_idx) -> int  
# Returns number of TWT groups

TWT_Schedules.get_schedules_by_duration(num_groups=None, ascending=True) -> list
# Returns schedules sorted by total duration (shortest first by default)
# Helper; the baseline enumerates all 380 pairs instead
```

---

### QueuingModel - M/D/1 Queue Theory

Models the WiFi TWT system as a **deterministic service queue**:

- **λ (arrival rate)**: Packets arriving per beacon interval per STA
- **μ (service rate)**: Packets transmitted per beacon interval per STA
- **ρ = λ/μ (utilization)**: Must be < 1 for stable queue

#### Calibration from the EDA

Loaded at import from `../exploration-scripts/eda-data/<run_id>/derived_constants.json`, written by
`5-dial-constants.py` from the newest EDA run. There is no hardcoded fallback: a missing
file or key raises.

```python
MEAN_ARRIVAL_RATE = _EDA_REPORT["arrival_rate_per_bi_mean"]   # packets per BI per STA
BASELINE_RATE_PER_MS = _EDA_REPORT["service_rate_mean"]       # packets per ms per STA
MEAN_DUTY_CYCLE = _EDA_NORM["duty_cycle"]["mean"]             # fraction of the beacon interval
MEAN_WAKE_MS = MEAN_DUTY_CYCLE * 102.4                        # ms
MEAN_SERVICE_RATE = BASELINE_RATE_PER_MS * MEAN_WAKE_MS       # packets per BI per STA
```

These values change with every EDA run; read them from `derived_constants.json` rather than
from this document.

#### Service Rate Estimation

```python
QueuingModel.estimate_service_rate(wake_duration_ms, num_sta_in_group) -> float
```

Estimates packets transmitted per STA in a wake period, Eq. (11) of the paper:

```
mu(d, n) = BASELINE_RATE_PER_MS * d * (1 - 0.025 * (n - 1)) * 1.5
```

**Factors applied:**

1. **Base rate**: `BASELINE_RATE_PER_MS × wake_ms` packets, the rate being EDA-derived.
1. **Contention penalty**: `1 - 0.025 × (n - 1)`, so more STAs sharing a service period lowers efficiency.
   - 16 STAs: factor 0.625
   - 8 STAs: factor 0.825
   - 4 STAs: factor 0.925
1. **Aggregation gain**: 1.5× for A-MPDU efficiency.

The rate is linear in wake duration; there is no saturation term.

#### Queue Dynamics

```python
QueuingModel.estimate_queue_growth(current_queue, service_rate, arrival_rate)
# Q(t+1) = max(0, Q(t) + λ - μ)

QueuingModel.compute_utilization(wake_duration_ms, num_sta, num_groups)
# Returns ρ = λ/μ for stability analysis
```

---

### EnergyModel - Power Consumption

Models energy consumption per beacon interval:

```
E_per_BI = P_active × T_wake + P_sleep × T_sleep
```

#### Power Constants (from NS-3 simulation)

Current draws come from `twt-constants.h`; power is current × `SUPPLY_VOLTAGE_V` (3.0 V).

| State             | Current (mA) | Power (mW) |
| ----------------- | ------------ | ---------- |
| Transmit (P_TX)   | 232          | 696.0      |
| Receive (P_RX)    | 66           | 198.0      |
| Idle (P_IDLE)     | 50           | 150.0      |
| Sleep (P_SLEEP)   | 0.12         | 0.36       |
| Active (P_ACTIVE) | 66           | 198.0      |

`P_ACTIVE` aliases `P_RX`: an awake STA is receiving unless it holds the medium.

#### Key Methods

```python
EnergyModel.compute_energy_mj(wake_duration_ms) -> float
# Energy per beacon interval in millijoules

EnergyModel.compute_energy_per_packet(wake_duration_ms, packets_transmitted) -> float
# Energy efficiency: mJ per packet; helper, not used by the baseline

EnergyModel.optimal_wake_for_queue(queue_bytes, max_wake_ms=90) -> float
# Minimum wake duration that drains the queue; helper, not used by the baseline
```

---

### ThroughputModel - Collision Estimation

Models throughput considering WiFi contention. The baseline does not call these methods;
they remain as analysis helpers.

#### Collision Probability

```python
ThroughputModel.estimate_collision_prob(num_sta, cw=CW_MIN)
# P_collision ≈ 1 - (1 - 1/CW)^(n-1)
```

| STAs Contending | Collision Probability |
| --------------- | --------------------- |
| 4               | ~19%                  |
| 8               | ~38%                  |
| 16              | ~64%                  |

#### Effective Throughput

```python
ThroughputModel.effective_throughput(wake_duration_ms, num_sta, num_groups)
# Returns packets per BI across all STAs after collision losses
```

**Key insight**: Splitting into groups reduces per-group contention, improving effective throughput.

---

## Policy Implementations

All three policies share one implementation, `_SectionVABaseline`. They differ only in the
two objective weights.

At construction the baseline enumerates all 20 x 19 = 380 action pairs. For each pair it
applies the assignment pattern to get per-group STA counts, then computes `mu(d_k, n_k)` per
group with `estimate_service_rate()` and the pair's total energy with `compute_energy_mj()`.
It also records `rho_max`, the worst per-STA utilization, and marks a pair stable when
`rho_max < 0.85`.

At each step `predict()` converts the observed BSR index to a packet backlog, projects one
step of queue growth per STA as `max(0, queue + lambda - mu)`, sums that over STAs, and
selects the pair minimising

```
W_QUEUE * scale(backlog) + W_ENERGY * scale(energy)
```

where `scale()` is min-max normalization across the 380 candidates. Unstable pairs are
excluded whenever at least one stable pair exists.

| Policy                       | `W_QUEUE` | `W_ENERGY` |
| ---------------------------- | --------- | ---------- |
| `AnalyticalThroughputPolicy` | 0.20      | 0.10       |
| `AnalyticalEnergyPolicy`     | 0.10      | 0.35       |
| `AnalyticalQueuePolicy`      | 0.35      | 0.10       |

The policies are stateless: `predict()` depends only on the current observation, and
`reset()` is a no-op.

## Airtime Efficiency

Shorter schedules leave more of the beacon interval for non-TWT traffic and cut wake energy.
The baseline gets this from its energy term: `compute_energy_mj()` grows with wake duration,
so a shorter schedule scoring equally on backlog wins on energy.
`TWT_Schedules.get_schedules_by_duration()` is available for shortest-first iteration but the
baseline does not use it.

### Airtime Reward Values

The reward function applies a matching **airtime penalty**:

`R_air = tanh(1.5 × (1 - d/90) - 0.35)`, where `d` is the schedule's total wake duration.

| Schedule     | Duration | Airtime Reward |
| ------------ | -------- | -------------- |
| 4            | 20ms     | +0.67          |
| 9            | 30ms     | +0.57          |
| 3, 8         | 40ms     | +0.45          |
| 2, 7, 13     | 60ms     | +0.15          |
| 1, 6, 10, 11 | 80ms     | -0.18          |
| 0, 5         | 90ms     | -0.34          |

---

## Author

Ahmed Maksud <ahmed.maksud@email.ucr.edu>\
SHINE Lab, Texas State University

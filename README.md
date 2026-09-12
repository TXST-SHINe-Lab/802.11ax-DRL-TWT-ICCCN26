# RL-Based TWT Scheduler for Next-Gen WiFi

<p align="center">
  <img src="https://img.shields.io/badge/NS--3-3.44-blue" alt="NS-3 Version">
  <img src="https://img.shields.io/badge/Python-3.11-green" alt="Python Version">
  <img src="https://img.shields.io/badge/WiFi-802.11ax-orange" alt="WiFi Standard">
  <img src="https://img.shields.io/badge/RL-PPO-red" alt="RL Algorithm">
  <a href="https://ieeexplore.ieee.org/document/11662818"><img src="https://img.shields.io/badge/Paper-IEEE%20ICCCN%202026-00629B" alt="IEEE ICCCN 2026 Paper"></a>
</p>

A reinforcement learning framework for **Target Wake Time (TWT)** scheduling in IEEE 802.11ax (WiFi 6) networks.
This project integrates NS-3 network simulation with Stable-Baselines3 PPO to learn optimal power-saving schedules for heterogeneous IoT networks.

Code for the paper:

> **A. Maksud and M. M. Carvalho**, "Deep Reinforcement Learning-based Dynamic TWT Scheduling for Heterogeneous Wi-Fi Networks,"
> in *Proc. 35th International Conference on Computer Communications and Networks (ICCCN)*, Honolulu, HI, USA, Jul. 2026, pp. 1–9.
> [[IEEE Xplore](https://ieeexplore.ieee.org/document/11662818), [doi:10.1109/ICCCN69946.2026.11662818](https://doi.org/10.1109/ICCCN69946.2026.11662818)]

If you use this code, please cite the paper (see [Citation](#11-citation)).

**Author**: Ahmed Maksud — SHINE Lab, Texas State University
**PI**: Prof. Marcelo Menezes De Carvalho

---

> **Important**: This project requires the [NS3-NS3AI installation](https://github.com/ahmedmaksud/NS3-NS3AI--installation-and-tests) to be completed first.
> Both projects must be located in the same `NS3-project` directory structure:

```
~/NS3-project/
├── NS3-NS3AI--installation-and-tests/    ← complete this first
└── ns-allinone-3.44/
    └── ns-3.44/                          ← ns3-root
        └── contrib/ai/examples/
            └── twt/                      ← this repo cloned here
```

## Quick Start

```bash
# 1. Clone the repository into the ns3-ai examples directory
cd ~/NS3-project/ns-allinone-3.44/ns-3.44/contrib/ai/examples/
git clone https://github.com/ahmedmaksud/802.11ax-DRL-TWT-ICCCN26.git twt

# 2. Register the new directory with CMake
echo 'add_subdirectory(twt)' >> CMakeLists.txt

# 3. Apply NS-3 WiFi patches (once, before the first build)
cd twt/mod-files/
./twt-complete-setup.sh
./install_bsr_manager.sh
cd ..

# 4. Run the full pipeline (build → EDA → train → evaluate → plot)
./run_all.sh
```

---

## Table of Contents

- [RL-Based TWT Scheduler for Next-Gen WiFi](#rl-based-twt-scheduler-for-next-gen-wifi)
  - [Table of Contents](#table-of-contents)
  - [1. Overview](#1-overview)
  - [2. System Architecture](#2-system-architecture)
    - [Data-flow summary](#data-flow-summary)
  - [3. NS-3 Simulation Source Files](#3-ns-3-simulation-source-files)
    - [`twt-constants.h`](#twt-constantsh)
    - [`pb-twt-core.h`](#pb-twt-coreh)
    - [`twt-simulation-config.h` / `.cc`](#twt-simulation-configh--cc)
    - [`twt-trace-callbacks.h` / `.cc`](#twt-trace-callbacksh--cc)
    - [`twt-metrics.h` / `.cc`](#twt-metricsh--cc)
    - [`pb-twt-wrapper-ns3.h` / `.cc`](#pb-twt-wrapper-ns3h--cc)
    - [`pb-twt-interface.cc`](#pb-twt-interfacecc)
    - [`pb_twt_wrapper_py.py`](#pb_twt_wrapper_pypy)
    - [`twt-main-simulation.cc`](#twt-main-simulationcc)
  - [4. How Callbacks Work](#4-how-callbacks-work)
  - [5. Metrics Collection: Two-Level Design](#5-metrics-collection-two-level-design)
    - [BI-Level (Fine-Grained CSV)](#bi-level-fine-grained-csv)
    - [Call-Level (Controller-Step Snapshots)](#call-level-controller-step-snapshots)
    - [Why Cumulative Values?](#why-cumulative-values)
  - [6. C++–Python Bridge: NS3-AI Shared Memory](#6-cpython-bridge-ns3-ai-shared-memory)
    - [Bridge Stack (Top to Bottom)](#bridge-stack-top-to-bottom)
  - [7. NS-3 Source Modifications (`mod-files/`)](#7-ns-3-source-modifications-mod-files)
    - [Files](#files)
    - [`BsrManager`](#bsrmanager)
    - [Installation Order](#installation-order)
  - [8. One-Stop Pipeline: `run_all.sh`](#8-one-stop-pipeline-run_allsh)
    - [Stages](#stages)
    - [Troubleshooting the pipeline](#troubleshooting-the-pipeline)
  - [9. Inner README Progression](#9-inner-readme-progression)
    - [Step 1 — `test-scripts/`](#step-1--test-scripts)
    - [Step 2 — `exploration-scripts/`](#step-2--exploration-scripts)
    - [Step 3 — `ppo-sb3-scripts/`](#step-3--ppo-sb3-scripts)
  - [10. Quick Reference](#10-quick-reference)
    - [Project Structure](#project-structure)
    - [Key Configuration Points](#key-configuration-points)
    - [Build & Run](#build--run)
    - [License](#license)
  - [11. Citation](#11-citation)

---

## 1. Overview

This project implements a **closed-loop reinforcement learning controller** for IEEE 802.11ax (WiFi 6) Target Wake Time (TWT) scheduling.
An NS-3 simulation runs a heterogeneous WiFi network of 16 stations.
Every `TWT_UPDATE_INTERVAL_BI` beacon intervals the simulation pauses, ships per-STA metrics to a Python controller over shared memory, waits for a new TWT schedule, applies it, and resumes.
The Python side can be an RL agent (Stable-Baselines3 PPO/LSTM-PPO), an analytical heuristic, or a simple test harness.

**Key numbers:**

- 16 STAs, 4 heterogeneous traffic classes, up to 8 TWT groups
- Beacon interval: 102.4 ms
- Schedule update cadence: every 20 BIs (~2 s)
- Episode length: 38 update steps spanning 760 BIs (~77.8 s), within an 855 BI (~87.6 s) simulation
- Warm-up before metrics start: 75 BIs (~7.7 s)

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                  Python Controller Process                     │
│                                                                │
│  train_ppo_V1.py / episode_file_runner.py / simple-controller  │
│        │                                                       │
│  FileCommEnv (Gymnasium)                                       │
│        │ Python dict (env / action)                            │
│  TWTWrapper (pb_twt_wrapper_py.py)                             │
│        │ C-struct via NS3-AI                                   │
│        │ shared memory segments (/dev/shm/MySeg_*, ...)        │
└────────────────────┬───────────────────────────────────────────┘
                     │ POSIX shared memory
┌────────────────────┴───────────────────────────────────────────┐
│                  NS-3 Simulation Process                       │
│                                                                │
│  twt-main-simulation.cc  ←  command-line args                  │
│        │                                                       │
│  TwtNetworkSetup  (twt-simulation-config.cc)                   │
│    802.11ax HE network, 16 STAs, 4 device classes              │
│        │                                                       │
│  Trace callbacks  (twt-trace-callbacks.cc)                     │
│    PHY-state, MAC-queue, BSR, A-MPDU, link-measurement...      │
│        │  (fire into per-STA global arrays)                    │
│  TwtMetrics  (twt-metrics.cc)                                  │
│    PeriodicBiLevelLogging()  every BI → CSV                    │
│    LogAndSendCallLevelMetrics()  every 20 BI → EnvStruct       │
│        │                                                       │
│  TWTWrapper  (pb-twt-wrapper-ns3.cc)                           │
│    RequestTWTSchedule(env)  →  ActionStruct                    │
│        │                                                       │
│  TwtNetworkSetup::ApplyTWTSchedule(action)                     │
│    Sets per-STA wake interval / duration / SP offset           │
└────────────────────────────────────────────────────────────────┘
```

### Data-flow summary

```
PHY/MAC events
   │
   ├─► trace callbacks ──► per-STA global arrays
   │                              │
   │           BI tick            │
   │           └── LogBiLevelMetrics() ──────────────────► CSV (eda-data/)
   │
   └─► every 20 BI
          └── LogAndSendCallLevelMetrics()
                  │ PopulateStaObservationRaw() fills EnvStruct
                  │
                  RequestTWTSchedule(EnvStruct)
                          │ shared memory (NS3-AI)
                          ▼
                  Python side reads EnvStruct
                  Controller computes ActionStruct
                          │ shared memory (NS3-AI)
                          ▼
                  ApplyTWTSchedule(ActionStruct)
```

---

## 3. NS-3 Simulation Source Files

All files below live in the project root alongside `CMakeLists.txt`.

### `twt-constants.h`

**Role**: Single source of truth for every numerical constant in the simulation.
Nothing is hardcoded elsewhere; all timing, energy, and sizing parameters are `#define` macros defined here.

Key groups of constants:

| Group          | Examples                                                                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Network sizing | `MAX_NUM_STA = 16`, `MAX_NUM_TWT_GROUPS = 8`, `MAC_ADDR_LEN = 6`                                                                             |
| Timing         | `BEACON_INTERVAL_MS = 102.4`, `TWT_UPDATE_INTERVAL_BI = 20`, `DURATION_IN_UPDATE = 38`, `TWT_UPDATE_START_BI = 90`, `TWT_SETUP_TIME_BI = 75` |
| Metrics window | `METRICS_START_TIME_BI = 75`, `APP_START_TIME_MIN_BI = 25`, `APP_START_TIME_MAX_BI = 45`                                                     |
| Energy model   | `PHY_STATE_TX_MA = 232`, `PHY_STATE_RX_MA = 66`, `PHY_STATE_IDLE_MA = 50`, `PHY_STATE_SLEEP_MA = 0.12`, `BATTERY_VOLTAGE_V = 3.0`            |
| Traffic        | `MAX_QUEUE_SIZE_BYTES = 65536`, `PAYLOAD_SIZE_BYTES = 1400`                                                                                  |

To change simulation scale or timing, edit only this file.

---

### `pb-twt-core.h`

**Role**: Defines every C++ struct that crosses the NS-3 ↔ Python boundary.
Both the simulation and the Python bindings include this single header.

**Structs:**

**`StaRealisticMetrics`** — metrics the AP can observe through standard 802.11ax/k protocols without simulation omniscience:

- BSR (Buffer Status Report) per Access Category (BE, BK, VI, VO) — quantized 0–255 × scaling factor (256 bytes/unit)
- 802.11k link measurements: RCPI/RSNI raw values, converted RSSI (dBm) and SNR (dB), link margin, TX power
- AP-observable RX counters: `rx_fragment_count`, `fcs_error_count`, `bytes_received_at_ap`, per-AC byte/packet counts
- PHY frame metadata: last MCS, NSS, channel width, guard interval, Power Management bit, last-RX TSF timestamp
- TSPEC device characteristics: device class, nominal MSDU size, mean data rate (kbps), delay bound (ms), user priority
- Airtime used (µs), observation timestamp and sequence number

**`StaOracleMetrics`** — simulation-internal ground truth (not deployable on real hardware):

- Energy: `total_energy_consumed_mj`, `current_times_time_ma_ms` (integral), `awake_time_ms`, `sleep_time_ms`, `duty_cycle`
- STA-local TX counters: `tx_fragment_count`, `tx_failed_count`, `tx_retry_count`, `ack_failure_count`
- Application layer: `packets_generated`, `packets_enqueued`, `bytes_generated`
- Queue drops: `mpdu_drops_expired`, `mpdu_drops_queue_full`, `psdu_response_timeouts`
- TX-side: `packets_transmitted`, `bytes_transmitted`, A-MPDU statistics
- Queue state: `queue_size_packets`, `queue_size_bytes`, `queue_max_size`
- Latency: `avg_latency_ms`, `queue_delay_sum_ms`, `queue_delay_count`
- Position: `position_x_m`, `position_y_m`, `position_z_m`

**`StaEnvStruct`** — pairs one `StaRealisticMetrics` + one `StaOracleMetrics` for a single STA.

**`EnvStruct`** — top-level observation sent C++ → Python:

- Metadata: `num_sta`, `simulation_time_sec`, `observation_timestamp_ms`, `observation_count`, `beacon_interval_ms`
- `sta_observations[MAX_NUM_STA]` — array of `StaEnvStruct`

**`TwtGroupConfig`** — TWT timing for one group: `group_id`, `twt_wake_interval_ms`, `twt_wake_duration_ms`, `twt_sp_offset_ms`, `num_stas_assigned`.

**`StaGroupAssignment`** — maps one STA to a group: `sta_id`, `assigned_twt_group`, `enable_twt`.

**`ActionStruct`** — TWT schedule sent Python → C++:

- `num_sta`, `num_active_twt_groups`, `action_timestamp_ms`
- `group_configs[MAX_NUM_TWT_GROUPS]` — per-group timing parameters
- `sta_assignments[MAX_NUM_STA]` — per-STA group membership

> **Realistic vs Oracle**: Only `StaRealisticMetrics` fields are used as RL observations in the default training setup.
> `StaOracleMetrics` fields are used for reward computation and EDA validation.
> This split ensures the agent trains on information a real AP could observe, while the oracle data is used only server-side for reward shaping.

---

### `twt-simulation-config.h` / `.cc`

**Role**: Defines the network topology, per-device traffic profiles, and the `ApplyTWTSchedule()` method that translates an `ActionStruct` into live NS-3 TWT API calls.

**`DeviceClass` enum** — four traffic archetypes:

| Value | Name                     | Data Rate | Pattern           | Notes                        |
| ----- | ------------------------ | --------- | ----------------- | ---------------------------- |
| 0     | `DEVICE_IOT_SENSOR`      | 256 Kbps  | Periodic, sparse  | Battery-powered, best-effort |
| 1     | `DEVICE_VIDEO_CAMERA`    | 2 Mbps    | Constant bit-rate | Mains-powered, video AC      |
| 2     | `DEVICE_VOICE_ASSISTANT` | Low       | Bursty            | Latency-critical, voice AC   |
| 3     | `DEVICE_VIDEO_STREAMING` | High      | Elastic           | Interactive, video AC        |

**`StaApplicationConfig` struct** — per-STA application parameters: data rate (kbps), packet size (bytes), inter-packet interval (ms), start/stop times, device class, QoS user priority, delay bound, and mean data rate for TSPEC.

**`TwtSimulationConfig` struct** — top-level simulation knobs: `simId`, `randSeed`, `parallelSim` (for concurrent multi-seed runs), `nStations`, `simulationTime_ms`, `p2pLinkDelay_ms`, `enableStateLogs`, `enablePcap`, `beaconInterval_s`.

**`TwtNetworkSetup` class** — orchestrates:

1. Node creation (AP + STAs)
1. WiFi 802.11ax HE channel + PHY setup
1. MAC association and application installation per device class
1. Mobility model placement
1. `ApplyTWTSchedule(const ActionStruct& action)` — iterates each STA, looks up its `StaGroupAssignment`, reads the corresponding `TwtGroupConfig`, and calls the NS-3 TWT API (`SetUnsolicitedTwtSchedule`) to set wake interval, duration, and service-period offset. STAs marked `enable_twt = false` have TWT disabled (normal EDCA).

---

### `twt-trace-callbacks.h` / `.cc`

**Role**: Implements all NS-3 trace sink functions and holds the global per-STA accumulator arrays that these sinks write into.

**How trace sinks work in NS-3**: NS-3 uses a publish/subscribe callback system (`TraceSource` / `TraceCallback`).
Objects like `WifiPhy`, `WifiMac`, and `Application` expose named trace sources.
A trace sink is a C++ function (or lambda) that is *connected* to one of these sources at simulation setup time.
Whenever the subscribed event fires during the simulation (e.g., a PHY-state transition), NS-3 calls the sink with the event data.
The sinks here accumulate data into global arrays indexed by STA ID; `TwtMetrics` reads those arrays on demand.

**Global per-STA arrays** (all indexed `[MAX_NUM_STA]`):

| Array                                                                 | Source      | What it tracks                    |
| --------------------------------------------------------------------- | ----------- | --------------------------------- |
| `timeElapsedForSta_ms_TI`                                             | Clock       | Total elapsed simulated time      |
| `awakeTimeElapsedForSta_ms_TI`                                        | PHY state   | Time in TX/RX/IDLE/CCA_BUSY       |
| `sleepTimeElapsedForSta_ms_TI`                                        | PHY state   | Time in SLEEP                     |
| `current_mA_TimesTime_ms_ForSta_TI`                                   | PHY state   | Integral mA × ms for energy       |
| `packetsGeneratedByAppForSta`                                         | App layer   | UDP packets created               |
| `packetsEnqueuedAtMacForSta`                                          | MAC queue   | Packets entering MAC queue        |
| `packetsTransmittedByPhyForSta`                                       | PHY TX      | Packets sent over air             |
| `bytesTransmittedByPhyForSta`                                         | PHY TX      | Bytes sent over air               |
| `uplinkTimeoutsForSta`                                                | MAC         | PSDU response timeouts            |
| `uplinkExpiredMpduForSta`                                             | MAC         | MPDU drops due to lifetime        |
| `uplinkFailedEnqueueMpduForSta`                                       | MAC         | MPDU drops due to queue full      |
| `bsrQueueBytesAcBeForSta` / `BkForSta` / `ViForSta` / `VoForSta`      | BSR manager | Buffer status per AC (raw bytes)  |
| `txSuccessCountForSta` / `txRetryCountForSta` / `txFailedCountForSta` | MAC stats   | 802.11k-style link counters       |
| `rxSuccessCountForSta` / `rxErrorCountForSta`                         | MAC stats   | AP-side RX success/failure        |
| `lastRssiDbmForSta` / `lastSnrDbForSta`                               | PHY RX      | RSSI / SNR of last received frame |
| `lastTxMcsForSta` / `lastTxRate100KbpsForSta`                         | PHY TX      | MCS index and data rate           |

**Energy model**: `TI_currentModel_mA` — a `std::map<std::string, double>` from PHY state name (e.g., `"TX"`, `"RX"`, `"IDLE"`, `"CCA_BUSY"`, `"SLEEP"`) to current draw (mA).
Values come from `twt-constants.h`.
Each PHY-state-change callback multiplies the time spent in the previous state by its mA value and accumulates into `current_mA_TimesTime_ms_ForSta_TI`.

**Trace files written to disk** (when `enableStateLogs = true`):

| File                       | Contents                      |
| -------------------------- | ----------------------------- |
| `e2eTraceFile`             | End-to-end packet latency     |
| `macQueueSizeTraceFile`    | MAC queue depth over time     |
| `ampduTraceFile`           | A-MPDU aggregation events     |
| `bsrTraceFile`             | BSR values per AC per BI      |
| `phyStateTraceFile`        | PHY state machine transitions |
| `txRxStatsTraceFile`       | Per-packet TX/RX stats        |
| `linkMeasurementTraceFile` | RSSI/SNR per STA              |
| `timeoutDropTraceFile`     | Timeout and drop events       |

---

### `twt-metrics.h` / `.cc`

**Role**: Packages the raw per-STA accumulator arrays into structured observations and logs.
It is the bridge between the trace callback layer and the IPC layer.

**`TwtMetrics` class** — key methods:

**`LogBiLevelMetrics()`**\
Called every beacon interval starting at `METRICS_START_TIME_BI`.
Reads all global trace arrays and writes a row to the BI-level CSV under `data-log/`.
Produces the fine-grained time series used by `exploration-scripts/` for EDA.
All values logged are cumulative (not deltas); the EDA pipeline and Python environment compute deltas in post-processing.

**`LogAndSendCallLevelMetrics()` → `EnvStruct`**\
Called every `TWT_UPDATE_INTERVAL_BI` BIs at each controller-update step.
Calls `PopulateStaObservationRaw()` for each active STA, then returns the filled `EnvStruct`.
This `EnvStruct` is handed directly to `TWTWrapper:: RequestTWTSchedule()` for IPC.

**`PopulateStaObservationRaw(StaEnvStruct& sta, uint32_t staId)`**\
Reads every global trace array for `staId` and fills the corresponding fields in both the `StaRealisticMetrics` and `StaOracleMetrics` sub-structs.
All values are cumulative snapshots taken at the moment of the call.
The Python side (specifically `file_comm_env.py`) computes step-deltas by subtracting the previous observation.

> The call-level observation is also written as a row to the call-level CSV (`data-log/call-level-metrics-*.csv`), providing per-step snapshots used by `test-scripts/verify-call-level-metrics.py`.

---

### `pb-twt-wrapper-ns3.h` / `.cc`

**Role**: The C++ half of the IPC bridge — an `ns3::Object` subclass named `TWTWrapper` that wraps the NS3-AI shared-memory interface.

**NS3-AI interface**: Uses `Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>`.
At initialization the wrapper calls `Ns3AiMsgInterface::Get()` and retrieves the typed interface pointer.
NS-3 is the *consumer* of shared memory here (`SetIsMemoryCreator(false)`); Python creates the segments.

**`Initialize()`**:

- Marks NS-3 as the non-creator side (`SetIsMemoryCreator(false)`)
- Disables vector mode (`SetUseVector(false)`) — simple fixed-size struct
- Enables finish handling (`SetHandleFinish(true)`) for clean teardown

**`RequestTWTSchedule(const EnvStruct& env)` → `ActionStruct`**:

1. `CppSendBegin()` — acquires write lock on the Cpp→Python segment
1. `*GetCpp2PyStruct() = env` — copies the `EnvStruct` into shared memory
1. `CppSendEnd()` — releases write lock; Python unblocks
1. `CppRecvBegin()` — blocks until Python writes the `ActionStruct`
1. `action = *GetPy2CppStruct()` — reads response
1. `CppRecvEnd()` — signals completion

The call blocks the NS-3 simulation thread until Python responds.
If Python crashes or times out, `CppRecvBegin()` will hang; the `run_all.sh` pipeline handles cleanup of stale shared-memory segments between runs.

**Optional CSV logging**: `EnableLogging(true, path)` writes a row per step — timestamp, success flag, `num_sta`, simulation time, number of active TWT groups, TX/RX counts — useful for debugging IPC reliability.

---

### `pb-twt-interface.cc`

**Role**: pybind11 module that exposes every C++ struct from `pb-twt-core.h` to Python as importable classes.
Built during `./ns3 build` alongside the NS3-AI module and installed into the Python environment.

Exposes:

- `StaRealisticMetrics` — all ~40 fields as read/write Python attributes
- `StaOracleMetrics` — all ~29 fields as read/write Python attributes
- `StaEnvStruct` — `.realistic` and `.oracle` sub-objects
- `EnvStruct` — metadata fields + `.sta_observations` as a Python list
- `TwtGroupConfig` — group timing parameters
- `StaGroupAssignment` — per-STA group mapping
- `ActionStruct` — full schedule struct with `.group_configs` and `.sta_assignments` lists
- Module-level constants: `MAX_NUM_STA`, `MAX_NUM_TWT_GROUPS`, `MAC_ADDR_LEN`

The `sta_mac` field in `StaRealisticMetrics` is exposed as `py.bytes` with getter/setter that do proper `memcpy`.
The `sta_observations` list in `EnvStruct` uses `reference` return policy so mutations on the Python side write directly into the shared-memory struct.

---

### `pb_twt_wrapper_py.py`

**Role**: Python-side counterpart to `pb-twt-wrapper-ns3.cc`.
Hides all NS3-AI implementation details behind a simple `reset()` / `step()` / `close()` API.
Any Python controller (RL agent, heuristic, test script) talks only to this wrapper.

**Class `TWTWrapper`**:

```python
wrapper = TWTWrapper(verbose=True)
env_dict = wrapper.reset(seed=9000)   # spawns NS-3 binary, returns first obs
while True:
    action_dict = controller.decide(env_dict)
    env_dict, done = wrapper.step(action_dict)
    if done:
        break
wrapper.close()
```

**`reset(seed)`**:

1. Imports `pb_twt_interface_py` (the pybind11 module) via `_import_bindings()`
1. Creates `Ns3AiMsgInterface` as the memory *creator* (Python side)
1. Spawns the NS-3 binary (`twt-main-simulation`) as a subprocess
1. Waits for the first `EnvStruct` from the simulation
1. Converts it to a Python dict via `_env_to_dict()`

**`step(action_dict)`**:

1. `_dict_to_action(action_dict)` — fills an `ActionStruct` from Python dict
1. Writes `ActionStruct` into shared memory and signals NS-3
1. Reads next `EnvStruct` from shared memory (blocks until NS-3 sends)
1. Converts to dict; checks `done` flag
1. Logs both env and action to CSV if `enable_logging=True`

**Dict format** — `env_dict` keys:

- `num_sta`, `simulation_time_sec`, `observation_count`, `beacon_interval_ms`
- `sta_observations` — list of dicts, each with `realistic` and `oracle` sub-dicts containing all fields from the corresponding structs

**`action_dict` keys**:

- `num_active_twt_groups`, `action_timestamp_ms`
- `group_configs` — list of dicts (`group_id`, `twt_wake_interval_ms`, `twt_wake_duration_ms`, `twt_sp_offset_ms`, `num_stas_assigned`)
- `sta_assignments` — list of dicts (`sta_id`, `assigned_twt_group`, `enable_twt`)

**CSV logging**: Two CSV files per run written to `data-log/`:

- `py-wrapper-env-<timestamp>.csv` — one row per STA per step, all realistic
  - oracle fields prefixed
- `py-wrapper-action-<timestamp>.csv` — one row per STA per step, group timing and assignment fields

> `episode_file_runner.py` in `ppo-sb3-scripts/` adds another layer: it acts as a JSON relay between the RL environment's `FileCommEnv` (which reads/writes JSON files) and `TWTWrapper` (which manages the shared-memory connection).
> This decoupling lets `FileCommEnv` be a standard Gymnasium environment without any NS3-AI imports.

---

### `twt-main-simulation.cc`

**Role**: The NS-3 `main()` entry point.
Wires together all components and drives the simulation loop.

**Command-line arguments** (all overridable at runtime):

| Argument           | Default   | Description                                 |
| ------------------ | --------- | ------------------------------------------- |
| `simId`            | 0         | Simulation ID (used for output file naming) |
| `randSeed`         | 1         | Random seed for reproducibility             |
| `parallelSim`      | false     | Enables unique shared-memory segment names  |
| `scenario`         | "default" | Scenario tag (affects traffic)              |
| `nStations`        | 16        | Number of STAs                              |
| `simulationTime`   | computed  | Total simulation duration (ms)              |
| `p2pLinkDelay`     | 5         | AP backhaul delay (ms)                      |
| `enableStateLogs`  | false     | Write PHY/MAC trace files to disk           |
| `enablePcap`       | false     | Write PCAP capture files                    |
| `enableDynamicTWT` | true      | Enable periodic TWT updates                 |

**Startup sequence**:

1. Parse command-line args → populate `TwtSimulationConfig`
1. Create `TwtNetworkSetup` → build topology, install apps
1. Call `ConnectTraceCallbacks()` → hook all PHY/MAC/App trace sources
1. Schedule `PeriodicBiLevelLogging()` at `METRICS_START_TIME_BI` (called every BI thereafter via self-reschedule)
1. Schedule `PeriodicTWTUpdate()` at `TWT_UPDATE_START_BI` (called every `TWT_UPDATE_INTERVAL_BI` BIs for `DURATION_IN_UPDATE` steps)
1. `Simulator::Run()` — simulation runs until all events are consumed

**`PeriodicBiLevelLogging()`** (every BI):

```
g_metrics->LogBiLevelMetrics()   // append row to BI-level CSV
Simulator::Schedule(next_BI, ...)  // reschedule itself
```

**`PeriodicTWTUpdate()`** (every 20 BIs, 38 times total):

```
EnvStruct env = g_metrics->LogAndSendCallLevelMetrics()
ActionStruct action = g_twtWrapper->RequestTWTSchedule(env)  // blocks for Python
g_networkSetup->ApplyTWTSchedule(action)
if (++stepCount < DURATION_IN_UPDATE)
    Simulator::Schedule(next_update, PeriodicTWTUpdate)
```

---

## 4. How Callbacks Work

NS-3 trace callbacks are the nervous system of the simulation.
Here is the complete path from a hardware event to a Python observation field:

```
1. PHY state changes (e.g., STA wakes for its TWT SP)
      │
      └─ WifiPhy::TraceConnectWithoutContext("State", PhyStateCallback)
              │
              PhyStateCallback(staId, start, duration, newState)
                  │
                  currentModel = TI_currentModel_mA[prevState]
                  current_mA_TimesTime_ms_ForSta_TI[staId] += currentModel × durationMs
                  awakeTimeElapsedForSta_ms_TI[staId] += durationMs  (if not SLEEP)
                  sleepTimeElapsedForSta_ms_TI[staId] += durationMs  (if SLEEP)

2. STA sends uplink data (A-MPDU transmitted)
      │
      └─ WifiMac::TraceConnectWithoutContext("AmpduTx", AmpduTxCallback)
              │
              AmpduTxCallback(staId, nMpdus, bytes, txVector)
                  packetsTransmittedByPhyForSta[staId] += nMpdus
                  bytesTransmittedByPhyForSta[staId]   += bytes

3. AP receives BSR from STA (via BsrManager hook in qos-frame-exchange-manager.cc)
      │
      └─ BsrManager::RecordBsr(staAddress, tid, queueSize)
              │
              bsrQueueBytesAcBeForSta[staId] = queueSize × scalingFactor  (if AC_BE TID)
              (similarly for BK, VI, VO)

4. MPDU drop (timeout or queue full)
      │
      └─ WifiMac::TraceConnectWithoutContext("DroppedMpdu", DroppedMpduCallback)
              │
              uplinkExpiredMpduForSta[staId]++   (if DropReason == LIFETIME)
              uplinkFailedEnqueueMpduForSta[staId]++  (if DropReason == QUEUE_FULL)
```

At every controller-update step, `PopulateStaObservationRaw()` reads these arrays atomically (NS-3 is single-threaded) and copies values into the `EnvStruct` to be sent to Python.
Because all values are cumulative, the Python environment (`file_comm_env.py`) subtracts the previous step's snapshot to obtain per-step deltas (e.g., `delta_bytes_transmitted`, `delta_packets_dropped`).

---

## 5. Metrics Collection: Two-Level Design

The project uses a two-level logging architecture optimized for different consumers:

### BI-Level (Fine-Grained CSV)

- **Trigger**: Every beacon interval, starting at BI 75
- **Method**: `TwtMetrics::LogBiLevelMetrics()`
- **Output**: `data-log/bi-level-metrics-<simId>.csv`
- **Consumer**: `exploration-scripts/` EDA pipeline; `test-scripts/plot-bi-metrics.py`
- **Content**: One row per STA per BI, all cumulative counters captured at that instant. Produces ~16 × (total BIs − 75) rows per run.

### Call-Level (Controller-Step Snapshots)

- **Trigger**: Every 20 BIs (= each controller-update step)
- **Method**: `TwtMetrics::LogAndSendCallLevelMetrics()`
- **Output**: `data-log/call-level-metrics-<simId>.csv` + `EnvStruct` over IPC
- **Consumer**: Python controller (real-time); `test-scripts/` verification scripts
- **Content**: One row per STA per step, same cumulative counters. Produces 16 × 38 = 608 rows per episode.

### Why Cumulative Values?

The simulation never resets its counters, so both levels emit monotonically increasing values.
Delta computation (throughput/step, energy/step, etc.) is intentionally kept in Python, where it is simpler to vectorize and debug.
The `file_comm_env.py` observation builder computes step-deltas and normalises them using a `VecNormalize` wrapper.

---

## 6. C++–Python Bridge: NS3-AI Shared Memory

The IPC layer uses the **ns3-ai** module's `Ns3AiMsgInterfaceImpl` template, which creates three POSIX shared-memory segments per simulation instance:

| Segment            | Direction    | Content                    |
| ------------------ | ------------ | -------------------------- |
| `MySeg_<id>`       | Both         | Control flags and metadata |
| `MyCpp2PyMsg_<id>` | C++ → Python | `EnvStruct` (~8 KB)        |
| `MyPy2CppMsg_<id>` | Python → C++ | `ActionStruct` (~0.5 KB)   |

The `<id>` suffix is derived from `simId` when `parallelSim = true`, allowing multiple simulation instances to run concurrently (used in multi-seed training).

**Synchronisation**: Each segment has an associated lockable object (`MyLockable_<id>`).
The protocol is strictly alternating:

1. C++ writes `EnvStruct`, signals Python
1. C++ blocks on `CppRecvBegin()`
1. Python reads `EnvStruct`, computes `ActionStruct`
1. Python writes `ActionStruct`, signals C++
1. C++ unblocks, reads `ActionStruct`, simulation resumes

**Stale-segment cleanup**: If a previous run crashed, leftover segments under `/dev/shm/` will cause the next run to deadlock.
`run_all.sh` runs `rm -f /dev/shm/MySeg_* /dev/shm/MyCpp2PyMsg_* /dev/shm/MyPy2CppMsg_* /dev/shm/MyLockable_*` before each stage.

### Bridge Stack (Top to Bottom)

```
RL Agent / Heuristic / Test Script
    │
    │ Python dict (env_dict / action_dict)
    ▼
TWTWrapper  (pb_twt_wrapper_py.py)
    │  _env_to_dict() / _dict_to_action()
    │  reads/writes EnvStruct / ActionStruct objects (pybind11)
    ▼
pb_twt_interface_py  (pb-twt-interface.cc, built by ./ns3 build)
    │  C-level struct layout identical on both sides
    ▼
POSIX shared memory  (/dev/shm/)
    ▼
TWTWrapper  (pb-twt-wrapper-ns3.cc)
    │  CppSendBegin/End  CppRecvBegin/End
    ▼
Ns3AiMsgInterfaceImpl<EnvStruct, ActionStruct>
    ▼
twt-main-simulation.cc  (RequestTWTSchedule → ApplyTWTSchedule)
```

---

## 7. NS-3 Source Modifications (`mod-files/`)

The standard NS-3 3.44 WiFi module does not expose real-time BSR data or the TWT agreement API needed by this project.
The `mod-files/` directory contains the patches that must be applied **before the first `./ns3 build`**.

### Files

| File                                             | Purpose                                                                                                                                                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bsr-manager.h` / `bsr-manager.cc`               | New singleton class `BsrManager` — maintains an in-memory per-STA BSR table, updated in real time by a hook in the frame-exchange manager                                                      |
| `sta-wifi-mac.h` / `sta-wifi-mac.cc`             | Modified STA MAC — adds `GetTimeTillNextBeacon()` and enables unilateral TWT MLME calls                                                                                                        |
| `wifi-twt-agreement.h` / `wifi-twt-agreement.cc` | TWT agreement data structures from the [gtgnan/wifiTwt](https://github.com/gtgnan/wifiTwt) repository, patched for NS-3.44 API compatibility (IsRunning→IsPending, removed GetNavDurationLeft) |
| `install_bsr_manager.sh`                         | Installs `BsrManager` into `src/wifi/model/`                                                                                                                                                   |
| `twt-complete-setup.sh`                          | Complete one-shot NS-3 WiFi module setup                                                                                                                                                       |

### `BsrManager`

`BsrManager` is a singleton (`BsrManager::GetInstance()`) with a single method `RecordBsr(Mac48Address, tid, queueSize)`.
The installation script patches `src/wifi/model/qos-frame-exchange-manager.cc` to call this method every time the AP receives a QoS Data frame with a Buffer Status Report.
This makes the current BSR value available to trace callbacks in-process without file I/O.

### Installation Order

Run the two scripts from the `mod-files/` directory **before configuring and building NS-3**:

```bash
cd mod-files/

# Step 1: Install TWT agreement files and patch STA/AP WiFi MAC
./twt-complete-setup.sh

# Step 2: Install BsrManager into the WiFi module
./install_bsr_manager.sh
```

`twt-complete-setup.sh` performs:

1. Downloads `wifi-twt-agreement.h/.cc` from GitHub (or uses local copies) and copies them to `src/wifi/model/`
1. Patches `sta-wifi-mac.cc` to add `GetTimeTillNextBeacon()`
1. Applies NS-3.44 API fixes (`IsRunning` → `IsPending`, removes `GetNavDurationLeft`)
1. Updates `src/wifi/CMakeLists.txt`

`install_bsr_manager.sh` performs:

1. Copies `bsr-manager.h/.cc` to `src/wifi/model/`
1. Adds both files to `src/wifi/CMakeLists.txt`
1. Patches `qos-frame-exchange-manager.cc` — inserts the `BsrManager:: RecordBsr()` call after the existing `SetBufferStatus()` call in `PreProcessFrame()`
1. Backs up original files before any modification

After both scripts complete, run the full NS-3 build as described in the pipeline section below.

---

## 8. One-Stop Pipeline: `run_all.sh`

`run_all.sh` encodes the complete workflow from a fresh checkout to trained and evaluated RL models.
It resolves all paths relative to the script's own location, so it works regardless of where NS-3 is installed.

### Stages

```
Stage 0 — Cleanup
  pkill -f twt-main-simulation
  rm -f /dev/shm/MySeg_* ...

Stage 1 — Build
  ./ns3 clean
  source ~/NS3-project/EHRL/bin/activate
  pip install stable-baselines3 sb3-contrib
  ./ns3 configure \
      --enable-examples --enable-tests \
      -DNS3_PYTHON_BINDINGS=ON \
      -DPython3_EXECUTABLE=".../EHRL/bin/python" \
      -DNS3_BINDINGS_INSTALL_DIR=".../site-packages" \
      ...
  ./ns3 build

Stage 2 — EDA Data Collection  (exploration-scripts/)
  mkdir -p eda-data && rm -rf eda-data/*
  ./1-collect-data.sh           # runs many NS-3 instances with random policies
  python3.11 2-validate-logvstap.py  # sanity-checks log vs TAP metrics
  ./3-prepare-data.sh           # converts logs → Parquet / merged CSV
  python3.11 4-eda-analysis.py  # correlation heatmaps, distribution plots
  python3.11 5-dial-constants.py # derive reward NORM z-scores from this EDA

Stage 3 — Verification  (test-scripts/)
  python3.11 verify-call-level-metrics.py   # checks IPC data consistency
  python3.11 verify-ns3-to-python.py        # checks struct field alignment

Stage 4 — Cleanup between EDA and RL

Stage 5 — RL Training  (ppo-sb3-scripts/)
  ./run_training.sh --training-script train_lstm_ppo_V1.py
  # Trains three presets: throughput / energy / queue
  # Checkpoints → checkpoints/<run_id>/
  # TensorBoard logs → tb_logs/<run_id>/

Stage 6 — Cleanup between training and evaluation

Stage 7 — Evaluation
  ./run_eval.sh --training-script train_lstm_ppo_V1.py
  # Loads each checkpoint, evaluates over multiple seeds
  # Results → eval_results/eval_<run_id>.json

Stage 8 — Cleanup after evaluation

Stage 9 — Plotting
  python3.11 plot_training.py   → plots/training_curves_*.png
  python3.11 plot_evaluation.py → plots/eval_comparison_*.png
  python3.11 analyze_reward_signal.py → reward component breakdown

Stage 10 — Same pass for the MLP-PPO variant
  # run_all.sh then repeats stages 5–9 with train_ppo_V1.py
```

> To run a single stage in isolation, `cd` to the appropriate sub-directory and invoke the script or Python file directly.
> Each sub-directory has its own README with full usage details.

### Troubleshooting the pipeline

- **Deadlock on start**: stale `/dev/shm/MySeg_*` segments — run the cleanup commands at the top of `run_all.sh` manually.
- **`ImportError: pb_twt_interface_py`**: pybind11 module not installed — re-run `./ns3 build` and confirm the virtualenv path in `configure` matches.
- **Build failure after mod-files**: re-run `install_bsr_manager.sh` and `twt-complete-setup.sh` if NS-3 source was reset.

---

## 9. Inner README Progression

Work through the sub-directory READMEs in this order to build up a complete understanding of the system:

### Step 1 — `test-scripts/`

**`test-scripts/README.md`**

Start here.
Covers the lowest-level smoke tests that verify the simulation binary, the IPC bridge, and the metric pipeline each work correctly in isolation before running any training.

- `simple-controller.py` — exercises the full NS-3 ↔ Python ping-pong loop with a fixed TWT policy; confirms that `EnvStruct` and `ActionStruct` fields arrive correctly on both sides.
- `verify-call-level-metrics.py` — checks that call-level CSV values are monotonically increasing and that deltas are non-negative.
- `verify-ns3-to-python.py` — validates that every field in the pybind11 dict matches the corresponding C++ struct field observed via direct NS3-AI reads.
- `plot-bi-metrics.py` — plots per-STA energy, queue size, and throughput from a BI-level CSV to sanity-check callback accumulation.

### Step 2 — `exploration-scripts/`

**`exploration-scripts/README.md`**

After the unit tests pass, use this pipeline to build intuition about the state and action spaces before training.

- Runs the simulation hundreds of times with random TWT assignments and policies, collecting BI-level metric logs.
- Validates the logs, stacks them, and computes statistical summaries (correlation matrices, feature distributions, pairwise scatter plots).
- The resulting EDA data and plots reveal which features are most informative for which reward objective — critical context for reward function design.

### Step 3 — `ppo-sb3-scripts/`

**`ppo-sb3-scripts/README-ppo.md`** — Training and evaluation pipeline
**`ppo-sb3-scripts/README-analytical.md`** — Analytical baselines reference
**`ppo-sb3-scripts/README-metrics.md`** — Observation and reward metrics

The full RL training and evaluation pipeline.
Builds on the verified simulation and the EDA insights from the previous two steps.

---

## 10. Quick Reference

### Project Structure

```
twt/
├── twt-main-simulation.cc      ← NS-3 main() entry point
├── twt-constants.h             ← All simulation constants
├── pb-twt-core.h               ← EnvStruct, ActionStruct, RealisticMetrics, OracleMetrics
├── twt-trace-callbacks.h/.cc   ← PHY/MAC trace sinks + accumulator arrays
├── twt-metrics.h/.cc           ← BI-level and call-level metric packaging
├── twt-simulation-config.h/.cc ← Network topology, device classes, ApplyTWTSchedule()
├── pb-twt-wrapper-ns3.h/.cc    ← C++ NS3-AI shared-memory IPC
├── pb-twt-interface.cc         ← pybind11 bindings for all structs
├── pb_twt_wrapper_py.py        ← Python NS3-AI wrapper (reset/step/close)
├── CMakeLists.txt              ← NS-3 CMake build integration
├── mod-files/                  ← NS-3 WiFi source patches (run before build)
│   ├── bsr-manager.h/.cc
│   ├── sta-wifi-mac.h/.cc
│   ├── wifi-twt-agreement.h/.cc
│   ├── install_bsr_manager.sh
│   └── twt-complete-setup.sh
├── test-scripts/               ← Smoke tests and metric verification
├── exploration-scripts/        ← Random-action EDA pipeline
├── run_all.sh                  ← One-stop pipeline script
└── ppo-sb3-scripts/            ← RL training, evaluation, plotting
```

### Key Configuration Points

| What to change                             | Where                                  |
| ------------------------------------------ | -------------------------------------- |
| Number of STAs, TWT groups, episode length | `twt-constants.h`                      |
| Energy model (mA per PHY state)            | `twt-constants.h` — `PHY_STATE_*_MA`   |
| Traffic patterns per device class          | `twt-simulation-config.cc`             |
| Reward function weights                    | `ppo-sb3-scripts/reward_functions.py`  |
| RL hyperparameters (LR, clip, entropy)     | `ppo-sb3-scripts/train_lstm_ppo_V1.py` |
| Training timesteps / seeds                 | `ppo-sb3-scripts/run_training.sh`      |
| Evaluation episodes / baselines            | `ppo-sb3-scripts/run_eval.sh`          |

### Build & Run

```bash
# 1. Apply NS-3 patches (once)
cd mod-files/
./twt-complete-setup.sh
./install_bsr_manager.sh
cd ..

# 2. Build
cd ../../../..   # NS-3 root
./ns3 configure --enable-examples --enable-tests \
    -DNS3_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$(which python3.11)"
./ns3 build

# 3. Full pipeline
cd contrib/ai/examples/twt
./run_all.sh
```

### License

GNU General Public License v2 — see individual source files.

**Author**: Ahmed Maksud — [ahmed.maksud@email.ucr.edu](mailto:ahmed.maksud@email.ucr.edu)\
**PI**: Marcelo Menezes De Carvalho — SHINE Lab, Texas State University

---

## 11. Citation

If you use this code in your research, please cite:

```bibtex
@INPROCEEDINGS{11662818,
  author={Maksud, Ahmed and Carvalho, Marcelo M.},
  booktitle={2026 35th International Conference on Computer Communications and Networks (ICCCN)}, 
  title={Deep Reinforcement Learning-based Dynamic TWT Scheduling for Heterogeneous Wi-Fi Networks}, 
  year={2026},
  volume={},
  number={},
  pages={1-9},
  keywords={Schedules;Scheduling;Energy;Training;Long short term memory;Information rates;Modeling;Optimization;Throughput;Timing;Target Wake Time;IEEE 802.11ax;Deep Reinforcement Learning;Proximal Policy Optimization;Wi-Fi Power Management;IoT Networks},
  doi={10.1109/ICCCN69946.2026.11662818}}
```

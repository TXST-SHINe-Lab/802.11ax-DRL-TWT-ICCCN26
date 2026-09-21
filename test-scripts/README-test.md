# TWT Simulation Smoke-Test Scripts

This directory contains **smoke-test scripts** that verify the full NS-3 TWT simulation
pipeline before any RL training or EDA data collection. Run these scripts first whenever
the C++ code changes, the pybind11 interface is updated, or a new machine is set up.

The scripts should run before the EDA in `exploration-scripts/` and the
PPO training in `ppo-sb3-scripts/`. They answer a single question: *is the simulation
producing correct, internally-consistent logs?*

---

## Overview

The smoke-test pipeline has five stages:

```
1. simple-controller.py       ← drives the NS-3 simulator with fixed configs
         │  produces 13 CSV log types (in ../data-log/)
         ▼
2. summary-metrics.py         ← text report: scan every log type for obvious problems
         │
         ▼
3. plot-bi-metrics.py         ← visual inspection of per-BI metrics
         │
         ├─► 4. verify-ns3-to-python.py      ← NS3 call-log == py-wrapper-env?
         │
         └─► 5. verify-call-level-metrics.py ← BI aggregates == call-level? E2E == BI?
```

All five scripts read logs from `../data-log/` (created automatically by the simulation).

---

## Quick Start

```bash
# 1. Activate the Python virtual environment
source ~/NS3-project/EHRL/bin/activate

# 2. From the project root, run the simulation with five fixed configs
cd test-scripts/
python simple-controller.py

# (if NS-3 is not already built, build it first from ns-3.44/)

# 3. Print a comprehensive text report
python summary-metrics.py

# 4. Plot per-BI metrics for all STAs and save PNGs to ../data-log/
python plot-bi-metrics.py

# 5. Verify that NS-3's internal logs match what Python received
python verify-ns3-to-python.py

# 6. Verify that BI-level and call-level logs are numerically consistent
python verify-call-level-metrics.py
```

Expected final output of steps 5 and 6:

```
Overall: ALL TESTS PASSED
```

---

## Files

### `simple-config.json`

Defines **five TWT schedule configurations** used to drive the smoke test. Each
configuration is applied to all 16 STAs for a fixed number of BI steps.
`simple-controller.py` cycles through them sequentially.

| #   | Groups | STAs per group      | Wake window (ms)  | SP offset (ms)             |
| --- | ------ | ------------------- | ----------------- | -------------------------- |
| 1   | 1 (G0) | all 16 → G0         | 80                | 0                          |
| 2   | 2      | 0–7 → G0, 8–15 → G1 | G0: 30, G1: 50    | G0: 0, G1: 30              |
| 3   | 4      | 4 STAs each         | 10 / 20 / 25 / 35 | 0 / 10 / 30 / 55           |
| 4   | 2      | 0–7 → G0, 8–15 → G1 | G0: 50, G1: 30    | G0: 0, G1: 50              |
| 5   | 8      | 2 STAs each         | 10 each           | 10-ms increments (0…70 ms) |

The five configs are designed to exercise different group-count / window-size / offset
combinations without requiring the RL agent.

---

### `simple-controller.py`

**Purpose**: Run the complete C++↔Python bridge with a deterministic sequence of fixed
actions. Confirms the simulator launches, executes update steps, produces logs, and
terminates cleanly.

**Arguments**

| Flag          | Default              | Description                          |
| ------------- | -------------------- | ------------------------------------ |
| `--config`    | `simple-config.json` | Path to JSON config file             |
| `--seed`      | `9000`               | RNG seed passed to `wrapper.reset()` |
| `--max-steps` | `1000`               | Safety cap on simulation steps       |

**Flow**

1. Loads all configs from `simple-config.json`.
1. Calls `TWTWrapper.reset(seed)` to start NS-3.
1. Loops `wrapper.step(action_dict)`, cycling through the five configs in order.
1. Prints color-coded step progress (`RUNNING`, `DONE`).
1. Calls `wrapper.close()` and prints a completion banner with the location of the
   log files.

**Outputs** (written by NS-3 and the Python wrapper to `../data-log/`)

| File pattern                       | Writer | Contents                                                |
| ---------------------------------- | ------ | ------------------------------------------------------- |
| `ns3-BI-log-*.csv`                 | NS-3   | Per-BI per-STA PHY + energy + packet + BSR metrics      |
| `ns3-call-log-*.csv`               | NS-3   | Per-call-interval per-STA observations                  |
| `ns3-link-measurement-trace-*.csv` | NS-3   | RSSI, SNR, link margin over time                        |
| `ns3-ampdu-trace-*.csv`            | NS-3   | AMPDU aggregation statistics                            |
| `ns3-bsr-trace-*.csv`              | NS-3   | BSR trace                                               |
| `ns3-macqueuesize-trace-*.csv`     | NS-3   | MAC queue size trace                                    |
| `ns3-e2e-trace-*.csv`              | NS-3   | End-to-end packet trace (PHY_TX events)                 |
| `ns3-phystate-trace-*.csv`         | NS-3   | Physical layer state log                                |
| `ns3-timeout-drop-trace-*.csv`     | NS-3   | Timeout-based MPDU drops                                |
| `ns3-txrx-stats-trace-*.csv`       | NS-3   | TX/RX fragment and retry statistics                     |
| `ns3-twt-wrapper-*.csv`            | NS-3   | TWT wrapper events (config update BIs, actions)         |
| `py-wrapper-env-*.csv`             | Python | State received by Python controller (per step, per STA) |
| `py-wrapper-action-*.csv`          | Python | Actions applied by Python controller (per step)         |

**Example**

```bash
python simple-controller.py --config simple-config.json --seed 42
```

---

### `summary-metrics.py`

**Purpose**: Print a comprehensive human-readable report covering every log type.
Use this as a first-pass sanity check after running `simple-controller.py`.

**Arguments**: None (no CLI flags).

**Reads** (latest file of each pattern in `../data-log/`): all 13 log types listed above.

**Report sections**

| Section            | Highlights                                                                                                                         |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| BI-Level Metrics   | Dataset size, per-STA avg RSSI / SNR / link margin, total RX packets, FCS errors, cumulative energy; optional BSR and TX/RX tables |
| Call-Level Metrics | Total call count, per-call start time, duration, avg RSSI, SNR, RX packets (first 10 calls)                                        |
| Link Measurements  | RSSI / SNR / link margin distributions from the link-measurement trace                                                             |
| AMPDU Stats        | AMPDU aggregation counts                                                                                                           |
| MAC Queue          | MAC queue size trace summary                                                                                                       |
| Timeout Drops      | MPDU timeout drop trace summary                                                                                                    |
| Python Controller  | Summary of the Python `py-wrapper-env` and `py-wrapper-action` logs                                                                |

**Example**

```bash
python summary-metrics.py
```

---

### `plot-bi-metrics.py`

**Purpose**: Visualize per-BI metrics for every STA as matplotlib figures, with
call-level overlays and vertical markers at each TWT config-update point.

**Arguments**

| Flag        | Default                   | Description                          |
| ----------- | ------------------------- | ------------------------------------ |
| `--file`    | latest `ns3-BI-log-*.csv` | Path to a specific BI log CSV        |
| `--save`    | `True`                    | Save PNG files to `../data-log/`     |
| `--no-save` | —                         | Suppress PNG output                  |
| `--show`    | `False`                   | Display plots interactively (blocks) |
| `--sta`     | all                       | Plot only the specified STA ID       |

**Reads**

| File                    | Purpose                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------- |
| `ns3-BI-log-*.csv`      | Primary BI-level data                                                                         |
| `ns3-call-log-*.csv`    | Call-level overlay (window averages / snapshots)                                              |
| `ns3-twt-wrapper-*.csv` | Config-update BI indices for vertical markers                                                 |
| `../twt-constants.h` | Parsed at runtime for `BEACON_INTERVAL_MS`, `TWT_UPDATE_INTERVAL_BI`, `TWT_UPDATE_START_BI` |

**Important**: All `oracle_*` columns in `ns3-BI-log-*.csv` are **cumulative**.
The script computes `.diff()` per STA before plotting so every y-axis shows a
per-BI delta.

**Per-STA figure** (2 × 2 panel, saved as `sta{N}_metrics_*.png`)

| Panel        | Metric                       | Overlay                     |
| ------------ | ---------------------------- | --------------------------- |
| Top-left     | Throughput (Mbps)            | Call-level window average   |
| Top-right    | Energy efficiency (bytes/mJ) | Call-level window average   |
| Bottom-left  | Power consumption (mW)       | Call-level window average   |
| Bottom-right | BSR queue size (KB)          | Call-level snapshot markers |

Red dashed vertical lines mark each TWT config-update BI; a per-STA summary
banner is appended below the 2 × 2 grid.

**Network overview figure** (saved as `network_overview_*.png`)

| Panel        | Content                                     |
| ------------ | ------------------------------------------- |
| Top-left     | Total network throughput (sum over STAs)    |
| Top-right    | Per-STA throughput overlay                  |
| Bottom-left  | Total network energy (sum over STAs) per BI |
| Bottom-right | Network efficiency (Mbps / mJ)              |

**Example**

```bash
python plot-bi-metrics.py --show          # interactive, all STAs
python plot-bi-metrics.py --sta 3         # STA 3 only
python plot-bi-metrics.py --no-save --show
```

---

### `verify-ns3-to-python.py`

**Purpose**: Confirm that NS-3's internal call-level log matches the state actually
received by the Python controller via the ns3-ai shared-memory bridge. Catches serialization
bugs, field mapping errors, and off-by-one alignment issues.

**Arguments**: None (processes every timestamp group found in `../data-log/`).

**Reads** (per timestamp group)

| File                      | Role                                      |
| ------------------------- | ----------------------------------------- |
| `ns3-call-log-*.csv`      | Ground truth: what NS-3 logged internally |
| `py-wrapper-env-*.csv`    | What the Python controller received       |
| `ns3-BI-log-*.csv`        | Supplementary BI-level reference          |
| `py-wrapper-action-*.csv` | Supplementary action log                  |
| `ns3-twt-wrapper-*.csv`   | Timing reference                          |

**Test suite** (per timestamp group)

| Test        | Checks                                                                                                              |
| ----------- | ------------------------------------------------------------------------------------------------------------------- |
| `sta_count` | Same number of STAs in call-log and py-wrapper-env                                                                  |
| `timing`    | Call timestamps align with Python step indices                                                                      |
| `metrics`   | BSR (AC_BE/BK/VI/VO), RSSI, SNR, link margin, bytes/packets received at AP, airtime — values match within tolerance |
| `oracle`    | Energy, awake time, duty cycle, packets generated/transmitted, queue size — values match                            |

**Output** (printed to stdout):

```
Overall: ALL TESTS PASSED
```

Non-zero exit code on any failure.

**Example**

```bash
python verify-ns3-to-python.py
```

---

### `verify-call-level-metrics.py`

**Purpose**: Confirm that the three logging granularities agree with each other:

1. **BI vs Call**: Aggregating per-BI rows over a call window must equal the call-level log.
1. **E2E vs BI**: PHY_TX events in the E2E trace must match `total_packets_transmitted`
   delta in the BI log.

**Arguments**: None (processes every complete timestamp group in `../data-log/`).

**Reads** (per timestamp group)

| File                  | Role                              |
| --------------------- | --------------------------------- |
| `ns3-BI-log-*.csv`    | Fine-grained source               |
| `ns3-call-log-*.csv`  | Coarse-grained target             |
| `ns3-e2e-trace-*.csv` | Packet-level reference (optional) |

**Note**: `total_packets_generated` and `total_packets_transmitted` are **cumulative**
in the C++ code. The script computes per-interval deltas before comparing.

**Side-effect output**: `aggregated_from_bi_<timestamp>.csv` written to `../data-log/`
for manual inspection.

**Result table** (printed to stdout):

```
Timestamp            | BI vs Call      | E2E vs BI       | Overall
--------------------------------------------------------------------
20260201_120000      | PASSED          | PASSED          | PASSED
```

Non-zero exit code on any failure.

**Example**

```bash
python verify-call-level-metrics.py
```

---

## Log File Directory

All scripts read from and write to `../data-log/` (relative to this directory).
Files are named with a `YYYYMMDD_HHMMSS` timestamp so multiple runs co-exist
without overwriting each other. The verification scripts automatically group files
by matching timestamp suffixes.

```
../data-log/
├── ns3-BI-log-20260201_120000.csv
├── ns3-call-log-20260201_120000.csv
├── ns3-link-measurement-trace-20260201_120000.csv
├── ns3-ampdu-trace-20260201_120000.csv
├── ns3-bsr-trace-20260201_120000.csv
├── ns3-macqueuesize-trace-20260201_120000.csv
├── ns3-e2e-trace-20260201_120000.csv
├── ns3-phystate-trace-20260201_120000.csv
├── ns3-timeout-drop-trace-20260201_120000.csv
├── ns3-txrx-stats-trace-20260201_120000.csv
├── ns3-twt-wrapper-20260201_120000.csv
├── py-wrapper-env-20260201_120000.csv
├── py-wrapper-action-20260201_120000.csv
├── sta0_metrics_20260201_120001.png        ← from plot-bi-metrics.py
├── ...
├── network_overview_20260201_120001.png
└── aggregated_from_bi_20260201_120000.csv  ← from verify-call-level-metrics.py
```

---

## Important Caveats

- **Cumulative oracle columns**: All `oracle_*` fields in `ns3-BI-log-*.csv` are running
  totals — not per-BI values. Both `plot-bi-metrics.py` and `verify-call-level-metrics.py`
  compute `.diff()` per STA before any analysis.

- **INT8_MIN sentinel** (`-128`): Link quality columns (RSSI, SNR, link margin)
  use `−128` as a "no measurement yet" sentinel. `summary-metrics.py` and
  `verify-ns3-to-python.py` replace it with `NaN` before computing statistics.

- **`twt-constants.h` auto-parsed**: `plot-bi-metrics.py` reads
  `../twt-constants.h` at runtime with a regex to extract
  `BEACON_INTERVAL_MS`, `TWT_UPDATE_INTERVAL_BI`, and `TWT_UPDATE_START_BI`.
  If the header file moves, update the `parse_twt_constants()` search path.

- **First call baseline**: The first call period is a snapshot with no prior state;
  call-level data is only logged from the *second* call onward (`call_index ≥ 1`).
  `verify-call-level-metrics.py` accounts for this offset.

- **Running order**: These scripts are read-only after `simple-controller.py` finishes.
  You can re-run steps 2–5 as many times as needed without launching a new simulation.

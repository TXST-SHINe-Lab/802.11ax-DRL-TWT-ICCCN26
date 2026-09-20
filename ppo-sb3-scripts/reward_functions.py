#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""Reward functions for the TWT WiFi scheduling RL environment.

All 15 metrics are read; 14 contribute to the 6 components corresponding to Eqs. (4) through (9).
last_rx_timestamp_us is read but unused, since Eq. (9) has no recency term.

Metric correlations found by the EDA:
- bsr_queue_index ≈ queue_size_bytes (both 111 unique values) → merged into QUEUE
- duty_cycle = awake / (awake + sleep) → redundant, use sleep ratio directly
- airtime correlates with bytes_transmitted → merged into THROUGHPUT
- fcs_error_count, rx_fragment_count → merged into CHANNEL QUALITY
- packets_transmitted, packets_enqueued → drain ratio in QUEUE

The 6 components, between them covering every metric:
1. THROUGHPUT: bytes_transmitted, packets_transmitted, airtime_used
2. QUEUE: queue_size_bytes/packets, bsr_queue_index, packets_enqueued, drain ratio
3. ENERGY: energy_mj, awake_time, sleep_time, duty_cycle
4. DROPS: drops_expired (with fairness analysis)
5. CHANNEL: fcs_error_count, rx_fragment_count
6. AIRTIME: total scheduled wake duration of the selected schedule

Normalization constants come from derived_constants.json, written by 5-dial-constants.py into the EDA run directory it was derived from; see run_paths.derived_constants_path() for how the current file is located.
"""

from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import json
import os
import run_paths

# --- Schedule table, source of the TWT wake durations used by the airtime penalty ---


def _load_schedule_table() -> Dict[int, int]:
    """Map schedule_id to total_duration_ms, read from table_schedule.json.

    Raises if the table is missing or empty; there is deliberately no fallback.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    schedule_path = os.path.join(
        script_dir, "..", "exploration-scripts", "table_schedule.json"
    )

    if not os.path.exists(schedule_path):
        raise FileNotFoundError(
            f"Schedule table not found: {schedule_path}\n"
            "Run exploration-scripts/generate_action_tables.py before training or "
            "evaluation."
        )
    with open(schedule_path, "r") as f:
        data = json.load(f)
    schedules = data.get("schedules", [])
    if not schedules:
        raise ValueError(f"Schedule table contains no schedules: {schedule_path}")
    return {s["schedule_id"]: s["total_duration_ms"] for s in schedules}


# Loaded once at import; every reward call reads this lookup.
SCHEDULE_DURATIONS = _load_schedule_table()
BEACON_INTERVAL_MS = 102.4  # ms, the standard beacon interval

MAX_SCHEDULE_DURATION_MS = float(max(SCHEDULE_DURATIONS.values()))


# --- Normalization constants, the z-score table ---
# Sole source: derived_constants.json, produced by the EDA dial step 5-dial-constants.py and
# located per-run by run_paths.derived_constants_path().

# The 15 NORM keys every reward component requires.
REQUIRED_NORM_KEYS = (
    "delta_bytes_transmitted",
    "delta_packets_transmitted",
    "delta_airtime_used_us",
    "queue_size_bytes",
    "queue_size_packets",
    "bsr_queue_index",
    "delta_packets_enqueued",
    "delta_energy_mj",
    "delta_awake_time_ms",
    "delta_sleep_time_ms",
    "duty_cycle",
    "delta_drops_expired",
    "fcs_error_count",
    "rx_fragment_count",
    "last_rx_timestamp_us",
)


REQUIRED_CENTER_KEYS = (
    "sleep_ratio_mean",
    "jain_bytes_mean",
    "jain_sleep_ratio_mean",
)

DERIVED_CONSTANTS_PATH = run_paths.derived_constants_path()


def _load_derived_constants() -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Load NORM and the EDA-tuned reward centers from derived_constants.json.

    Raises if the file is missing, unreadable, or missing any required key.
    There is deliberately no hardcoded fallback: run the EDA dial step first.
    Also prints the path it loaded, so a stale constants file is visible in the log.
    """
    dc_path = DERIVED_CONSTANTS_PATH
    if not os.path.exists(dc_path):
        raise FileNotFoundError(
            f"EDA constants file not found: {dc_path}\n"
            "Run the EDA dial step (exploration-scripts/5-dial-constants.py) "
            "before training or evaluation."
        )
    with open(dc_path) as f:
        derived = json.load(f)
    norm = derived.get("NORM", {})
    centers = derived.get("_report_only", {})

    missing = [k for k in REQUIRED_NORM_KEYS if k not in norm]
    if missing:
        raise KeyError(
            f"derived_constants.json is missing required NORM keys: {missing}\n"
            f"(source: {dc_path})"
        )
    missing = [k for k in REQUIRED_CENTER_KEYS if k not in centers]
    if missing:
        raise KeyError(
            f"derived_constants.json is missing required center keys: {missing}\n"
            f"(source: {dc_path})\n"
            "Re-run exploration-scripts/5-dial-constants.py -- older versions of the "
            "dial step did not emit the Jain-index centers."
        )
    print(f"[reward_functions] loaded EDA constants from {dc_path}")
    return norm, centers


NORM, _CENTERS = _load_derived_constants()

EXPECTED_SLEEP_RATIO = _CENTERS["sleep_ratio_mean"]
JAIN_BYTES_CENTER = _CENTERS["jain_bytes_mean"]
JAIN_SLEEP_RATIO_CENTER = _CENTERS["jain_sleep_ratio_mean"]

FAIRNESS_DENOM = 0.1
SLEEP_RATIO_DENOM = 0.1
WORST_SLEEP_DENOM = 0.15
DRAIN_DENOM = 0.2
COUNT_PENALTY_FRACTION = 0.25
DRAIN_RATIO_CENTER = 1.0

NUM_STA = 16


# --- Preset weights, each set of 6 components summing to 1.0 ---

PRESET_WEIGHTS = {
    "throughput": {
        "throughput": 0.35,  # Primary
        "queue": 0.20,  # Secondary, queue depth caps throughput
        "drops": 0.15,  # Secondary, a drop is throughput already paid for
        "energy": 0.10,  # Tertiary
        "airtime": 0.15,  # Penalizes long TWT schedules
        "channel": 0.05,  # Tertiary
    },
    "energy": {
        "energy": 0.35,  # Primary
        "throughput": 0.20,  # Secondary, some throughput is still required
        "drops": 0.10,  # Secondary, retransmissions waste energy
        "queue": 0.10,  # Tertiary
        "airtime": 0.20,  # Shorter schedules save energy
        "channel": 0.05,  # Tertiary
    },
    "queue": {
        "queue": 0.35,  # Primary, the latency proxy
        "drops": 0.20,  # Secondary, a drop is queue overflow
        "throughput": 0.20,  # Secondary, transmission is what drains the queue
        "energy": 0.10,  # Tertiary
        "airtime": 0.10,  # Shorter schedules mean lower latency
        "channel": 0.05,  # Tertiary
    },
}


# --- Helper functions ---


def get_sta_values(sta_deltas: List[Dict[str, float]], key: str) -> np.ndarray:
    """Extract one metric across all STAs; missing entries read as 0.0."""
    return np.array([sta.get(key, 0.0) for sta in sta_deltas])


def jains_fairness(values: np.ndarray) -> float:
    """Jain's fairness index in [0, 1], where 1.0 is perfect fairness.

    An empty or all-zero input returns 1.0: nothing was shared unfairly.
    """
    if len(values) == 0:
        return 1.0
    sum_x = np.sum(values)
    if sum_x == 0:
        return 1.0
    sum_x2 = np.sum(values**2)
    if sum_x2 == 0:
        return 1.0
    return (sum_x**2) / (len(values) * sum_x2)


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Divide, returning default when the denominator is not positive."""
    return a / b if b > 0 else default


def normalize(value: float, metric_name: str) -> float:
    """
    Scale a value to roughly [0, 1] by dividing by its NORM mean.
    Initial scaling only, so the metrics contribute proportionally.
    An unknown metric, or a non-positive mean, passes the value through unchanged.
    """
    if metric_name not in NORM:
        return value
    mean = NORM[metric_name]["mean"]
    if mean <= 0:
        return value
    return value / mean


def soft_clip(x: float, scale: float = 1.0) -> float:
    """
    Soft saturation using tanh to avoid hard clipping.
    Maps x ∈ (-∞, +∞) → (-scale, +scale) smoothly.
    - x = 0 → 0
    - x = ±1 → ±0.76 * scale
    - x = ±2 → ±0.96 * scale
    - x → ±∞ → ±scale (asymptotically)
    """
    return np.tanh(x) * scale


def zscore(value: float, metric_name: str) -> float:
    """
    Compute the z-score (value - mean) / std from the NORM table.
    Returns 0 if the metric is absent or its std is not positive.
    """
    if metric_name not in NORM:
        return 0.0
    mean = NORM[metric_name]["mean"]
    std = NORM[metric_name]["std"]
    if std <= 0:
        return 0.0
    return (value - mean) / std


def zscore_reward(
    value: float, metric_name: str, scale: float = 1.0, invert: bool = False
) -> float:
    """
    Convert a raw value to a reward by z-score normalization then soft clipping.

    - value at mean → 0
    - value 1 std above mean → tanh(1) * scale ≈ 0.76 * scale
    - value 2 std above mean → tanh(2) * scale ≈ 0.96 * scale

    Args:
        value: Raw metric value
        metric_name: Key in NORM dict
        scale: Max reward magnitude
        invert: If True, lower values are better, as for energy, drops and queue
    """
    z = zscore(value, metric_name)
    if invert:
        z = -z  # Below the mean now scores positive
    return soft_clip(z, scale)


def ratio_reward(
    value: float, expected: float, std: float, scale: float = 1.0
) -> float:
    """
    Convert a ratio to a reward by z-score normalization against an explicit mean and std.

    Args:
        value: Actual value, for example total_bytes
        expected: Expected value, for example mean * n_sta
        std: Standard deviation for normalization; a non-positive std returns 0.0
        scale: Max reward magnitude
    """
    if std <= 0:
        return 0.0
    z = (value - expected) / std
    return soft_clip(z, scale)


def inverse_ratio_reward(
    value: float, expected: float, std: float, scale: float = 1.0
) -> float:
    """
    As ratio_reward, but inverted so that lower values score higher.
    """
    if std <= 0:
        return 0.0
    z = (value - expected) / std
    return soft_clip(-z, scale)  # Negated, so below expected scores positive


def zscore_of(value: float, expected: float, std: float) -> float:
    """Raw z-score, unsaturated. Eqs (4)-(9) apply a single tanh to the term sum."""
    if std <= 0:
        return 0.0
    return (value - expected) / std


# --- Component 1: throughput ---
# Metrics: delta_bytes_transmitted, delta_packets_transmitted, delta_airtime_used_us


def compute_throughput_component(
    sta_deltas: List[Dict[str, float]], n_sta: int
) -> Dict:
    """
    Throughput: Maximize bytes transmitted efficiently.

    Uses:
    - delta_bytes_transmitted: Primary throughput measure
    - delta_packets_transmitted: Packet count
    - delta_airtime_used_us: Efficiency = bytes/airtime
    """
    bytes_tx = get_sta_values(sta_deltas, "delta_bytes_transmitted")
    packets_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")
    airtime = get_sta_values(sta_deltas, "delta_airtime_used_us")

    total_bytes = np.sum(bytes_tx)
    total_packets = np.sum(packets_tx)
    total_airtime = np.sum(airtime)

    # --- z(B): aggregate bytes ---
    expected_bytes = NORM["delta_bytes_transmitted"]["mean"] * n_sta
    std_bytes = NORM["delta_bytes_transmitted"]["std"] * np.sqrt(n_sta)
    bytes_reward = zscore_of(total_bytes, expected_bytes, std_bytes)

    # --- z(P): aggregate packets ---
    expected_packets = NORM["delta_packets_transmitted"]["mean"] * n_sta
    std_packets = NORM["delta_packets_transmitted"]["std"] * np.sqrt(n_sta)
    packets_reward = zscore_of(total_packets, expected_packets, std_packets)

    # --- z(B/U): airtime efficiency ---
    expected_eff = NORM["delta_bytes_transmitted"]["mean"] / (
        NORM["delta_airtime_used_us"]["mean"] + 1
    )
    std_eff = expected_eff * 0.5
    if total_airtime > 0:
        eff_reward = zscore_of(total_bytes / total_airtime, expected_eff, std_eff)
    else:
        eff_reward = 0.0 if total_bytes == 0 else zscore_of(0.0, expected_eff, std_eff)

    # --- (J(delta_b) - mu_J) / 0.1 ---
    fairness = jains_fairness(bytes_tx)
    fairness_bonus = (fairness - JAIN_BYTES_CENTER) / FAIRNESS_DENOM

    # --- -n_starv / (0.25 N) ---
    starvation_thresh = NORM["delta_bytes_transmitted"]["mean"] * 0.1
    starving = int(np.sum(bytes_tx < starvation_thresh))
    starvation_penalty = -starving / (n_sta * COUNT_PENALTY_FRACTION)

    reward = (
        bytes_reward + packets_reward + eff_reward + fairness_bonus + starvation_penalty
    )

    return {
        "reward": float(np.tanh(reward)),
        "total_bytes": total_bytes,
        "total_packets": total_packets,
        "total_airtime": total_airtime,
        "efficiency": safe_div(total_bytes, total_airtime),
        "fairness": fairness,
        "starving_stas": starving,
        "bytes_reward": bytes_reward,
        "packets_reward": packets_reward,
        "eff_reward": eff_reward,
        "fairness_bonus": fairness_bonus,
    }


# --- Component 2: queue, the latency proxy ---
# Metrics: queue_size_bytes, queue_size_packets, bsr_queue_index,
#          delta_packets_enqueued, delta_packets_transmitted


def compute_queue_component(sta_deltas: List[Dict[str, float]], n_sta: int) -> Dict:
    """
    Queue: Minimize queue buildup (latency proxy).

    Uses:
    - queue_size_bytes: Primary queue measure
    - queue_size_packets: Secondary (validates bytes)
    - bsr_queue_index: BSR report (0-254, correlates with queue)
    - delta_packets_enqueued + delta_packets_transmitted: Drain ratio
    """
    queue_bytes = get_sta_values(sta_deltas, "queue_size_bytes")
    queue_packets = get_sta_values(sta_deltas, "queue_size_packets")
    bsr_index = get_sta_values(sta_deltas, "bsr_queue_index")
    packets_enq = get_sta_values(sta_deltas, "delta_packets_enqueued")
    packets_tx = get_sta_values(sta_deltas, "delta_packets_transmitted")

    avg_queue = np.mean(queue_bytes)
    max_queue = np.max(queue_bytes) if len(queue_bytes) > 0 else 0
    avg_queue_packets = np.mean(queue_packets)

    # --- -z(q_bar) ---
    expected_queue = NORM["queue_size_bytes"]["mean"]
    std_queue = NORM["queue_size_bytes"]["std"] / np.sqrt(n_sta)
    queue_reward = -zscore_of(avg_queue, expected_queue, std_queue)

    # --- -z(q_bar_bsr) ---
    avg_bsr = np.mean(bsr_index)
    expected_bsr = NORM["bsr_queue_index"]["mean"]
    std_bsr = NORM["bsr_queue_index"]["std"] / np.sqrt(n_sta)
    bsr_reward = -zscore_of(avg_bsr, expected_bsr, std_bsr)

    # --- (rho - 1) / 0.2 ---
    total_enq = np.sum(packets_enq)
    total_tx = np.sum(packets_tx)
    drain_ratio = total_tx / total_enq if total_enq > 0 else DRAIN_RATIO_CENTER
    drain_reward = (drain_ratio - DRAIN_RATIO_CENTER) / DRAIN_DENOM

    # --- -n_high / (0.25 N) ---
    high_thresh = expected_queue + NORM["queue_size_bytes"]["std"]
    high_queue_stas = int(np.sum(queue_bytes > high_thresh))
    high_queue_penalty = -high_queue_stas / (n_sta * COUNT_PENALTY_FRACTION)

    reward = queue_reward + bsr_reward + drain_reward + high_queue_penalty

    return {
        "reward": float(np.tanh(reward)),
        "avg_queue_bytes": avg_queue,
        "avg_queue_packets": avg_queue_packets,
        "max_queue_bytes": max_queue,
        "avg_bsr_index": avg_bsr,
        "drain_ratio": safe_div(total_tx, total_enq),
        "high_queue_stas": high_queue_stas,
        "queue_reward": queue_reward,
        "bsr_reward": bsr_reward,
        "drain_reward": drain_reward,
    }


# --- Component 3: energy ---
# Metrics: delta_energy_mj, delta_awake_time_ms, delta_sleep_time_ms, duty_cycle


def compute_energy_component(sta_deltas: List[Dict[str, float]], n_sta: int) -> Dict:
    """
    Energy: Minimize consumption, maximize sleep.

    Uses:
    - delta_energy_mj: Direct energy consumption
    - delta_awake_time_ms, delta_sleep_time_ms: Sleep efficiency
    - duty_cycle: Verification (should = awake/(awake+sleep))
    """
    energy = get_sta_values(sta_deltas, "delta_energy_mj")
    awake = get_sta_values(sta_deltas, "delta_awake_time_ms")
    sleep = get_sta_values(sta_deltas, "delta_sleep_time_ms")
    duty = get_sta_values(sta_deltas, "duty_cycle")

    total_energy = np.sum(energy)

    # --- Sleep ratio per STA ---
    total_time = awake + sleep + 1e-8  # Epsilon to avoid divide by zero
    sleep_ratios = sleep / total_time
    avg_sleep_ratio = np.mean(sleep_ratios)
    min_sleep_ratio = np.min(sleep_ratios) if len(sleep_ratios) > 0 else 0

    # --- (s_r_bar - mu_s) / 0.1 ---
    sleep_reward = (avg_sleep_ratio - EXPECTED_SLEEP_RATIO) / SLEEP_RATIO_DENOM

    # --- -z(sum E_i) ---
    expected_energy = NORM["delta_energy_mj"]["mean"] * n_sta
    std_energy = NORM["delta_energy_mj"]["std"] * np.sqrt(n_sta)
    energy_reward = -zscore_of(total_energy, expected_energy, std_energy)

    # --- -z(d_bar) ---
    avg_duty = np.mean(duty)
    std_duty = NORM["duty_cycle"]["std"] / np.sqrt(n_sta)
    duty_reward = -zscore_of(avg_duty, NORM["duty_cycle"]["mean"], std_duty)

    # --- (J(s_r) - mu_J) / 0.1 ---
    sleep_fairness = jains_fairness(sleep_ratios)
    fairness_bonus = (sleep_fairness - JAIN_SLEEP_RATIO_CENTER) / FAIRNESS_DENOM

    # --- -(mu_s - s_r_min) / 0.15 ---
    worst_penalty = -(EXPECTED_SLEEP_RATIO - min_sleep_ratio) / WORST_SLEEP_DENOM

    reward = sleep_reward + energy_reward + duty_reward + fairness_bonus + worst_penalty

    return {
        "reward": float(np.tanh(reward)),
        "total_energy_mj": total_energy,
        "avg_sleep_ratio": float(avg_sleep_ratio),
        "min_sleep_ratio": float(min_sleep_ratio),
        "avg_duty_cycle": float(avg_duty),
        "sleep_fairness": sleep_fairness,
        "sleep_reward": sleep_reward,
        "energy_reward": energy_reward,
        "duty_reward": duty_reward,
        "worst_penalty": worst_penalty,
    }


# --- Component 4: drops ---
# Metrics: delta_drops_expired


def compute_drops_component(sta_deltas: List[Dict[str, float]], n_sta: int) -> Dict:
    """
    Drops: Minimize packet drops.

    Uses:
    - delta_drops_expired: Primary drop measure
    - Per-STA analysis for worst-case and fairness
    """
    drops = get_sta_values(sta_deltas, "delta_drops_expired")

    total_drops = np.sum(drops)
    max_drops = np.max(drops) if len(drops) > 0 else 0

    # --- -z(D) ---
    expected_drops = NORM["delta_drops_expired"]["mean"] * n_sta
    std_drops = NORM["delta_drops_expired"]["std"] * np.sqrt(n_sta)
    drop_reward = -zscore_of(total_drops, expected_drops, std_drops)

    # --- -z(D_max) ---
    worst_penalty = -zscore_of(
        max_drops,
        NORM["delta_drops_expired"]["mean"],
        NORM["delta_drops_expired"]["std"],
    )
    # --- -n_hd / (0.25 N) ---
    high_drop_thresh = (
        NORM["delta_drops_expired"]["mean"] + NORM["delta_drops_expired"]["std"]
    )
    high_drop_stas = int(np.sum(drops > high_drop_thresh))
    high_drop_penalty = -high_drop_stas / (n_sta * COUNT_PENALTY_FRACTION)

    reward = drop_reward + worst_penalty + high_drop_penalty

    return {
        "reward": float(np.tanh(reward)),
        "total_drops": total_drops,
        "max_drops": max_drops,
        "high_drop_stas": high_drop_stas,
        "drop_reward": drop_reward,
        "worst_penalty": worst_penalty,
    }


# --- Component 5: channel quality ---
# Metrics: fcs_error_count, rx_fragment_count, last_rx_timestamp_us


def compute_channel_component(sta_deltas: List[Dict[str, float]], n_sta: int) -> Dict:
    """
    Channel quality: Monitor errors and activity.

    Uses:
    - fcs_error_count: FCS errors (lower = better channel)
    - rx_fragment_count: RX activity (more = active link)
    - last_rx_timestamp_us: Read but unused; Eq. (9) has no recency term
    """
    fcs = get_sta_values(sta_deltas, "fcs_error_count")
    rx_frags = get_sta_values(sta_deltas, "rx_fragment_count")
    rx_ts = get_sta_values(sta_deltas, "last_rx_timestamp_us")

    avg_fcs = np.mean(fcs)
    avg_rx = np.mean(rx_frags)

    # --- -z(fcs_bar) ---
    expected_fcs = NORM["fcs_error_count"]["mean"]
    std_fcs = NORM["fcs_error_count"]["std"] / np.sqrt(n_sta)
    fcs_reward = -zscore_of(avg_fcs, expected_fcs, std_fcs)

    # --- +z(rx_bar) ---
    expected_rx = NORM["rx_fragment_count"]["mean"]
    std_rx = NORM["rx_fragment_count"]["std"] / np.sqrt(n_sta)
    rx_reward = zscore_of(avg_rx, expected_rx, std_rx)

    # --- -n_inactive / (0.25 N) ---
    inactive_stas = int(np.sum(rx_frags < 1))
    inactive_penalty = -inactive_stas / (n_sta * COUNT_PENALTY_FRACTION)

    reward = fcs_reward + rx_reward + inactive_penalty

    return {
        "reward": float(np.tanh(reward)),
        "avg_fcs_errors": avg_fcs,
        "avg_rx_fragments": avg_rx,
        "inactive_stas": inactive_stas,
        "fcs_reward": fcs_reward,
        "rx_reward": rx_reward,
    }


# --- Component 6: airtime, schedule duration efficiency ---
# Penalizes long TWT schedules, encouraging efficient use of the beacon interval.


def compute_airtime_component(schedule_idx: Optional[int], n_sta: int) -> Dict:
    """
    Airtime efficiency: penalize long TWT schedule durations.

    Shorter schedules are better because:
    - More time available for other traffic (non-TWT STAs, management)
    - Lower latency for responsive scheduling
    - Energy efficiency (less total wake time)

    Args:
        schedule_idx: Index into SCHEDULE_DURATIONS, 0 through 19; None yields a neutral 0.0
        n_sta: Number of STAs, unused, kept so every component shares one signature

    Returns:
        Dict with reward, duration_ms, utilization and duration_reward
    """
    if schedule_idx is None:
        # No action provided, return neutral reward
        return {
            "reward": 0.0,
            "duration_ms": 0,
            "utilization": 0.0,
            "duration_reward": 0.0,
        }

    # An unknown index falls back to 90 ms, the longest schedule, so a bad action is never rewarded.
    duration_ms = SCHEDULE_DURATIONS.get(schedule_idx, 90)

    # Fraction of the beacon interval the schedule occupies; reported only, not rewarded.
    utilization = duration_ms / BEACON_INTERVAL_MS

    # --- Duration reward, monotonically decreasing in duration ---
    # Normalized to 1.0 at 0 ms and 0.0 at max_duration, then shifted so the longest schedule scores negative.
    max_duration = MAX_SCHEDULE_DURATION_MS
    duration_normalized = 1.0 - (duration_ms / max_duration)
    # 0 ms → +0.82, 20 ms → +0.67, 40 ms → +0.45, 60 ms → +0.15, 90 ms → -0.34
    duration_reward = float(np.tanh(duration_normalized * 1.5 - 0.35))

    return {
        "reward": duration_reward,
        "duration_ms": duration_ms,
        "utilization": utilization,
        "duration_reward": duration_reward,
    }


# --- Main reward function ---


def compute_reward(
    sta_deltas: List[Dict[str, float]],
    preset: str = "throughput",
    action: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Compute the weighted reward from all 15 metrics across the 6 components.

    Args:
        sta_deltas: List of per-STA metric dicts, 15 metrics each
        preset: 'throughput', 'energy', or 'queue'
        action: (schedule_idx, assignment_idx); without it the airtime component scores 0.0

    Returns:
        Dict with 'total' reward and 'components' breakdown
    """
    if not sta_deltas:
        return {"total": 0.0, "components": {}, "details": {}}

    n_sta = len(sta_deltas)
    weights = PRESET_WEIGHTS.get(preset, PRESET_WEIGHTS["throughput"])

    # Compute all 6 components
    throughput = compute_throughput_component(sta_deltas, n_sta)
    queue = compute_queue_component(sta_deltas, n_sta)
    energy = compute_energy_component(sta_deltas, n_sta)
    drops = compute_drops_component(sta_deltas, n_sta)
    channel = compute_channel_component(sta_deltas, n_sta)

    # Airtime component - uses action's schedule_idx
    schedule_idx = action[0] if action is not None else None
    airtime = compute_airtime_component(schedule_idx, n_sta)

    # Weighted sum
    w_throughput = weights["throughput"] * throughput["reward"]
    w_queue = weights["queue"] * queue["reward"]
    w_energy = weights["energy"] * energy["reward"]
    w_drops = weights["drops"] * drops["reward"]
    w_channel = weights["channel"] * channel["reward"]
    w_airtime = weights["airtime"] * airtime["reward"]

    total = w_throughput + w_queue + w_energy + w_drops + w_channel + w_airtime

    return {
        "total": total,
        "components": {
            # Weighted contributions
            "w_throughput": w_throughput,
            "w_queue": w_queue,
            "w_energy": w_energy,
            "w_drops": w_drops,
            "w_channel": w_channel,
            "w_airtime": w_airtime,
            # Raw component rewards
            "raw_throughput": throughput["reward"],
            "raw_queue": queue["reward"],
            "raw_energy": energy["reward"],
            "raw_drops": drops["reward"],
            "raw_channel": channel["reward"],
            "raw_airtime": airtime["reward"],
        },
        # Detailed breakdown for logging/plotting
        "details": {
            # Throughput sub-rewards
            "throughput_bytes_reward": throughput.get("bytes_reward", 0),
            "throughput_packets_reward": throughput.get("packets_reward", 0),
            "throughput_eff_reward": throughput.get("eff_reward", 0),
            "throughput_fairness_bonus": throughput.get("fairness_bonus", 0),
            # Queue sub-rewards
            "queue_reward": queue.get("queue_reward", 0),
            "queue_packets_reward": queue.get("packets_reward", 0),
            "queue_bsr_reward": queue.get("bsr_reward", 0),
            "queue_drain_reward": queue.get("drain_reward", 0),
            "queue_max_penalty": queue.get("max_penalty", 0),
            # Energy sub-rewards
            "energy_sleep_reward": energy.get("sleep_reward", 0),
            "energy_consumption_reward": energy.get("energy_reward", 0),
            "energy_duty_reward": energy.get("duty_reward", 0),
            "energy_worst_penalty": energy.get("worst_penalty", 0),
            # Drops sub-rewards
            "drops_reward": drops.get("drop_reward", 0),
            "drops_worst_penalty": drops.get("worst_penalty", 0),
            "drops_fairness_penalty": drops.get("fairness_penalty", 0),
            # Channel sub-rewards
            "channel_fcs_reward": channel.get("fcs_reward", 0),
            "channel_rx_reward": channel.get("rx_reward", 0),
            "channel_recency_reward": channel.get("recency_reward", 0),
            # Airtime sub-rewards
            "airtime_duration_reward": airtime.get("duration_reward", 0),
        },
        # Raw metrics for plotting
        "metrics": {
            "total_bytes": throughput["total_bytes"],
            "total_packets": throughput["total_packets"],
            "total_airtime": throughput["total_airtime"],
            "throughput_efficiency": throughput["efficiency"],
            "throughput_fairness": throughput["fairness"],
            "starving_stas": throughput["starving_stas"],
            "avg_queue_bytes": queue["avg_queue_bytes"],
            "avg_queue_packets": queue["avg_queue_packets"],
            "max_queue_bytes": queue["max_queue_bytes"],
            "avg_bsr_index": queue["avg_bsr_index"],
            "drain_ratio": queue["drain_ratio"],
            "high_queue_stas": queue["high_queue_stas"],
            "total_energy_mj": energy["total_energy_mj"],
            "avg_sleep_ratio": energy["avg_sleep_ratio"],
            "min_sleep_ratio": energy["min_sleep_ratio"],
            "avg_duty_cycle": energy["avg_duty_cycle"],
            "sleep_fairness": energy["sleep_fairness"],
            "total_drops": drops["total_drops"],
            "max_drops": drops["max_drops"],
            "high_drop_stas": drops["high_drop_stas"],
            "avg_fcs_errors": channel["avg_fcs_errors"],
            "avg_rx_fragments": channel["avg_rx_fragments"],
            "inactive_stas": channel["inactive_stas"],
            "schedule_duration_ms": airtime["duration_ms"],
            "beacon_utilization": airtime["utilization"],
        },
        # Z-scores for checking if metrics are out of expected bounds
        # z > 2 or z < -2 indicates unusual values
        "zscores": {
            # Throughput z-scores (sum metrics: std * sqrt(n))
            "total_bytes_z": (
                throughput["total_bytes"]
                - NORM["delta_bytes_transmitted"]["mean"] * n_sta
            )
            / (NORM["delta_bytes_transmitted"]["std"] * np.sqrt(n_sta) + 1e-8),
            "total_packets_z": (
                throughput["total_packets"]
                - NORM["delta_packets_transmitted"]["mean"] * n_sta
            )
            / (NORM["delta_packets_transmitted"]["std"] * np.sqrt(n_sta) + 1e-8),
            "total_airtime_z": (
                throughput["total_airtime"]
                - NORM["delta_airtime_used_us"]["mean"] * n_sta
            )
            / (NORM["delta_airtime_used_us"]["std"] * np.sqrt(n_sta) + 1e-8),
            # Queue z-scores (avg metrics: std / sqrt(n))
            "avg_queue_bytes_z": (
                queue["avg_queue_bytes"] - NORM["queue_size_bytes"]["mean"]
            )
            / (NORM["queue_size_bytes"]["std"] / np.sqrt(n_sta) + 1e-8),
            "avg_queue_packets_z": (
                queue["avg_queue_packets"] - NORM["queue_size_packets"]["mean"]
            )
            / (NORM["queue_size_packets"]["std"] / np.sqrt(n_sta) + 1e-8),
            "avg_bsr_index_z": (
                queue["avg_bsr_index"] - NORM["bsr_queue_index"]["mean"]
            )
            / (NORM["bsr_queue_index"]["std"] / np.sqrt(n_sta) + 1e-8),
            # Energy z-scores
            "total_energy_z": (
                energy["total_energy_mj"] - NORM["delta_energy_mj"]["mean"] * n_sta
            )
            / (NORM["delta_energy_mj"]["std"] * np.sqrt(n_sta) + 1e-8),
            "avg_duty_cycle_z": (energy["avg_duty_cycle"] - NORM["duty_cycle"]["mean"])
            / (NORM["duty_cycle"]["std"] / np.sqrt(n_sta) + 1e-8),
            # Drops z-scores
            "total_drops_z": (
                drops["total_drops"] - NORM["delta_drops_expired"]["mean"] * n_sta
            )
            / (NORM["delta_drops_expired"]["std"] * np.sqrt(n_sta) + 1e-8),
            # Channel z-scores
            "avg_fcs_errors_z": (
                channel["avg_fcs_errors"] - NORM["fcs_error_count"]["mean"]
            )
            / (NORM["fcs_error_count"]["std"] / np.sqrt(n_sta) + 1e-8),
            "avg_rx_fragments_z": (
                channel["avg_rx_fragments"] - NORM["rx_fragment_count"]["mean"]
            )
            / (NORM["rx_fragment_count"]["std"] / np.sqrt(n_sta) + 1e-8),
        },
        "weights": weights,
        "n_sta": n_sta,
    }


# --- Reward function class ---


class RewardFunction:
    """Binds a preset to compute_reward so callers can invoke it as a plain callable."""

    def __init__(self, preset: str = "throughput"):
        if preset not in PRESET_WEIGHTS:
            raise ValueError(
                f"Unknown preset: {preset}. Use: {list(PRESET_WEIGHTS.keys())}"
            )
        self.preset = preset
        self.weights = PRESET_WEIGHTS[preset]

    def __call__(
        self,
        sta_deltas: List[Dict[str, float]],
        action: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Compute reward with optional action for airtime penalty.

        Args:
            sta_deltas: Per-STA metric deltas
            action: Optional (schedule_idx, assignment_idx) tuple for airtime penalty
        """
        return compute_reward(sta_deltas, self.preset, action=action)

    def get_weights(self) -> Dict[str, float]:
        return self.weights.copy()


# --- Registry ---

REWARD_REGISTRY = {
    # Primary presets (match PRESET_WEIGHTS keys)
    "throughput": lambda: RewardFunction("throughput"),
    "energy": lambda: RewardFunction("energy"),
    "queue": lambda: RewardFunction("queue"),
    # Aliases for argparse compatibility
    "balanced": lambda: RewardFunction("throughput"),
    "latency": lambda: RewardFunction("queue"),
    "throughput_focused": lambda: RewardFunction("throughput"),
    "energy_focused": lambda: RewardFunction("energy"),
    "latency_focused": lambda: RewardFunction("queue"),
    # Additional aliases
    "fairness_focused": lambda: RewardFunction(
        "throughput"
    ),  # throughput has fairness bonus
}


def get_reward_function(name: str, **kwargs) -> RewardFunction:
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown: {name}. Available: {list(REWARD_REGISTRY.keys())}")
    return REWARD_REGISTRY[name]()


# --- Reward logger, captures the full reward breakdown for plotting ---

import csv
import os
from datetime import datetime


class RewardLogger:
    """
    Logs detailed reward breakdown per step for later plotting/analysis.

    Supports two modes:
    1. Immediate: Creates CSV file on init, writes each step immediately
    2. Buffered: Collects data in memory, saves on demand (useful for env integration)
    """

    # CSV columns
    COLUMNS = [
        # Episode/step info
        "episode",
        "step",
        "timestamp",
        # Total and weighted components
        "total_reward",
        "w_throughput",
        "w_queue",
        "w_energy",
        "w_drops",
        "w_channel",
        "w_airtime",
        # Raw component rewards
        "raw_throughput",
        "raw_queue",
        "raw_energy",
        "raw_drops",
        "raw_channel",
        "raw_airtime",
        # Throughput sub-rewards
        "throughput_bytes_reward",
        "throughput_packets_reward",
        "throughput_eff_reward",
        "throughput_fairness_bonus",
        # Queue sub-rewards
        "queue_reward",
        "queue_packets_reward",
        "queue_bsr_reward",
        "queue_drain_reward",
        "queue_max_penalty",
        # Energy sub-rewards
        "energy_sleep_reward",
        "energy_consumption_reward",
        "energy_duty_reward",
        "energy_worst_penalty",
        # Drops sub-rewards
        "drops_reward",
        "drops_worst_penalty",
        "drops_fairness_penalty",
        # Channel sub-rewards
        "channel_fcs_reward",
        "channel_rx_reward",
        "channel_recency_reward",
        # Airtime sub-rewards
        "airtime_duration_reward",
        # Raw metrics
        "total_bytes",
        "total_packets",
        "total_airtime",
        "throughput_efficiency",
        "throughput_fairness",
        "starving_stas",
        "avg_queue_bytes",
        "avg_queue_packets",
        "max_queue_bytes",
        "avg_bsr_index",
        "drain_ratio",
        "high_queue_stas",
        "total_energy_mj",
        "avg_sleep_ratio",
        "min_sleep_ratio",
        "avg_duty_cycle",
        "sleep_fairness",
        "total_drops",
        "max_drops",
        "high_drop_stas",
        "avg_fcs_errors",
        "avg_rx_fragments",
        "inactive_stas",
        "schedule_duration_ms",
        "beacon_utilization",
        # Z-scores (for detecting out-of-bounds metrics, |z| > 2 is unusual)
        "total_bytes_z",
        "total_packets_z",
        "total_airtime_z",
        "avg_queue_bytes_z",
        "avg_queue_packets_z",
        "avg_bsr_index_z",
        "total_energy_z",
        "avg_duty_cycle_z",
        "total_drops_z",
        "avg_fcs_errors_z",
        "avg_rx_fragments_z",
        # Action (optional)
        "schedule_idx",
        "assignment_idx",
    ]

    def __init__(
        self,
        preset: str = "throughput",
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Args:
            preset: Reward preset name (for reference only)
            log_dir: Directory to save CSV files (if None, uses buffered mode)
            run_name: Optional run name for filename (default: timestamp)
        """
        self.preset = preset
        self.episode = 0
        self.step = 0
        self.data = []  # Buffered rows

        # If log_dir provided, write to file immediately
        self.immediate_mode = log_dir is not None
        self.log_path = None

        if self.immediate_mode:
            os.makedirs(log_dir, exist_ok=True)

            if run_name is None:
                run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

            self.log_path = os.path.join(log_dir, f"reward_log_{run_name}.csv")

            # Create file with header
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

            print(f"RewardLogger initialized (immediate mode): {self.log_path}")
        else:
            print(f"RewardLogger initialized (buffered mode) for preset: {preset}")

    def _build_row(self, reward_result: Dict, action: Optional[tuple] = None) -> List:
        """Build a row from reward_result dict."""
        components = reward_result.get("components", {})
        details = reward_result.get("details", {})
        metrics = reward_result.get("metrics", {})
        zscores = reward_result.get("zscores", {})

        row = [
            self.episode,
            self.step,
            datetime.now().isoformat(),
            reward_result.get("total", 0),
            # Weighted components
            components.get("w_throughput", 0),
            components.get("w_queue", 0),
            components.get("w_energy", 0),
            components.get("w_drops", 0),
            components.get("w_channel", 0),
            components.get("w_airtime", 0),
            # Raw components
            components.get("raw_throughput", 0),
            components.get("raw_queue", 0),
            components.get("raw_energy", 0),
            components.get("raw_drops", 0),
            components.get("raw_channel", 0),
            components.get("raw_airtime", 0),
            # Throughput sub-rewards
            details.get("throughput_bytes_reward", 0),
            details.get("throughput_packets_reward", 0),
            details.get("throughput_eff_reward", 0),
            details.get("throughput_fairness_bonus", 0),
            # Queue sub-rewards
            details.get("queue_reward", 0),
            details.get("queue_packets_reward", 0),
            details.get("queue_bsr_reward", 0),
            details.get("queue_drain_reward", 0),
            details.get("queue_max_penalty", 0),
            # Energy sub-rewards
            details.get("energy_sleep_reward", 0),
            details.get("energy_consumption_reward", 0),
            details.get("energy_duty_reward", 0),
            details.get("energy_worst_penalty", 0),
            # Drops sub-rewards
            details.get("drops_reward", 0),
            details.get("drops_worst_penalty", 0),
            details.get("drops_fairness_penalty", 0),
            # Channel sub-rewards
            details.get("channel_fcs_reward", 0),
            details.get("channel_rx_reward", 0),
            details.get("channel_recency_reward", 0),
            # Airtime sub-rewards
            details.get("airtime_duration_reward", 0),
            # Raw metrics
            metrics.get("total_bytes", 0),
            metrics.get("total_packets", 0),
            metrics.get("total_airtime", 0),
            metrics.get("throughput_efficiency", 0),
            metrics.get("throughput_fairness", 0),
            metrics.get("starving_stas", 0),
            metrics.get("avg_queue_bytes", 0),
            metrics.get("avg_queue_packets", 0),
            metrics.get("max_queue_bytes", 0),
            metrics.get("avg_bsr_index", 0),
            metrics.get("drain_ratio", 0),
            metrics.get("high_queue_stas", 0),
            metrics.get("total_energy_mj", 0),
            metrics.get("avg_sleep_ratio", 0),
            metrics.get("min_sleep_ratio", 0),
            metrics.get("avg_duty_cycle", 0),
            metrics.get("sleep_fairness", 0),
            metrics.get("total_drops", 0),
            metrics.get("max_drops", 0),
            metrics.get("high_drop_stas", 0),
            metrics.get("avg_fcs_errors", 0),
            metrics.get("avg_rx_fragments", 0),
            metrics.get("inactive_stas", 0),
            metrics.get("schedule_duration_ms", 0),
            metrics.get("beacon_utilization", 0),
            # Z-scores (|z| > 2 indicates unusual values)
            zscores.get("total_bytes_z", 0),
            zscores.get("total_packets_z", 0),
            zscores.get("total_airtime_z", 0),
            zscores.get("avg_queue_bytes_z", 0),
            zscores.get("avg_queue_packets_z", 0),
            zscores.get("avg_bsr_index_z", 0),
            zscores.get("total_energy_z", 0),
            zscores.get("avg_duty_cycle_z", 0),
            zscores.get("total_drops_z", 0),
            zscores.get("avg_fcs_errors_z", 0),
            zscores.get("avg_rx_fragments_z", 0),
            # Action
            action[0] if action else -1,
            action[1] if action else -1,
        ]
        if len(row) != len(self.COLUMNS):
            raise AssertionError(
                f"RewardLogger row/header mismatch: {len(row)} values vs "
                f"{len(self.COLUMNS)} columns"
            )
        return row

    def log_step(
        self,
        reward_result: Dict,
        episode: Optional[int] = None,
        step: Optional[int] = None,
        action: Optional[tuple] = None,
    ):
        """
        Log a single step's reward breakdown.

        Args:
            reward_result: Output from compute_reward()
            episode: Episode number (uses internal counter if None)
            step: Step number (uses internal counter if None)
            action: Optional (schedule_idx, assignment_idx) tuple
        """
        if episode is not None:
            self.episode = episode
        if step is not None:
            self.step = step
        else:
            self.step += 1

        row = self._build_row(reward_result, action)

        if self.immediate_mode:
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        else:
            self.data.append(row)

    def new_episode(self):
        """Call at the start of a new episode."""
        self.episode += 1
        self.step = 0

    def save(self, filepath: str):
        """Save buffered data to CSV file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.COLUMNS)
            writer.writerows(self.data)
        print(f"RewardLogger: Saved {len(self.data)} entries to {filepath}")

    def clear(self):
        """Clear buffered data."""
        self.data = []

    def get_log_path(self) -> str:
        """Return the path to the log file (immediate mode only)."""
        return self.log_path if self.log_path else ""

    def __len__(self):
        """Return number of buffered entries."""
        return len(self.data)


# --- Metrics summary, all 15 in use ---
#
# Component 1, throughput, 3 metrics:
#   delta_bytes_transmitted    → Primary throughput
#   delta_packets_transmitted  → Packet count
#   delta_airtime_used_us      → Efficiency, bytes per us
#
# Component 2, queue, 5 metrics:
#   queue_size_bytes           → Primary queue measure
#   queue_size_packets         → Secondary queue measure, validates bytes
#   bsr_queue_index            → BSR report, 0 through 254
#   delta_packets_enqueued     → Drain ratio numerator
#   delta_packets_transmitted  → Drain ratio denominator, shared with throughput
#
# Component 3, energy, 4 metrics:
#   delta_energy_mj            → Direct energy
#   delta_awake_time_ms        → Sleep ratio
#   delta_sleep_time_ms        → Sleep ratio
#   duty_cycle                 → Verification
#
# Component 4, drops, 1 metric with deep analysis:
#   delta_drops_expired        → With fairness, worst case, high-drop count
#
# Component 5, channel, 3 metrics read, 2 used:
#   fcs_error_count            → Channel quality
#   rx_fragment_count          → Link activity
#   last_rx_timestamp_us       → read but unused, Eq. (9) has no recency term
#
# All 15 are read during reward computation; 14 contribute to a component.
#
# Correlations handled:
# - bsr_queue_index ≈ queue_size_bytes → both in QUEUE, BSR as a secondary check
# - duty_cycle = f(awake, sleep) → used as verification in ENERGY
# - airtime ∝ bytes_tx → efficiency metric in THROUGHPUT

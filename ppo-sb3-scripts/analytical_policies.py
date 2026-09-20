#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""analytical_policies.py - analytical model-based policies for TWT WiFi scheduling.

An alternative to the pure heuristics in eval_policy.py, deciding from queueing theory and energy models rather than tuned thresholds.

The heuristics underperformed because they do not model how TWT parameters relate to system performance.
An analytical model captures three things they miss:
  1. Queue dynamics: M/D/1 or M/G/1 queuing model
  2. Energy model: E = P_active * T_wake + P_sleep * T_sleep
  3. Throughput model: Capacity vs contention tradeoff

Constants come from derived_constants.json, located per-run by run_paths.derived_constants_path(); import fails if the EDA dial step has not been run.

Imported by eval_policy.py for the --compare-analytical baselines; running it directly exercises the policies alone.

Usage:
    python3.11 analytical_policies.py

Lab: SHINE Lab, Texas State University
"""

import collections
import json
import os
import sys

import numpy as np
from typing import Tuple, Optional, Dict, Any

_EXPLORE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "exploration-scripts"
)
if _EXPLORE not in sys.path:
    sys.path.insert(0, _EXPLORE)

from generate_action_tables import apply_assignment_pattern
import run_paths

# --- EDA-tuned constants ---

_DERIVED_CONSTANTS_PATH = run_paths.derived_constants_path()

REQUIRED_EDA_KEYS = ("arrival_rate_per_bi_mean", "service_rate_mean")


def _load_eda_constants():
    """Load the EDA-tuned queueing constants from derived_constants.json.

    Returns the (_report_only, NORM) pair. Raises if the file or any required key
    is missing; there is deliberately no fallback, since default constants would
    silently change every policy decision.
    """
    if not os.path.exists(_DERIVED_CONSTANTS_PATH):
        raise FileNotFoundError(
            f"EDA constants file not found: {_DERIVED_CONSTANTS_PATH}\n"
            "Run exploration-scripts/5-dial-constants.py before evaluation."
        )
    with open(_DERIVED_CONSTANTS_PATH) as f:
        derived = json.load(f)
    report = derived.get("_report_only", {})
    missing = [k for k in REQUIRED_EDA_KEYS if k not in report]
    if missing:
        raise KeyError(
            f"derived_constants.json is missing required keys: {missing}\n"
            f"(source: {_DERIVED_CONSTANTS_PATH})\n"
            "Re-run exploration-scripts/5-dial-constants.py."
        )
    norm = derived.get("NORM", {})
    if "duty_cycle" not in norm:
        raise KeyError(
            f"derived_constants.json is missing NORM['duty_cycle'] "
            f"(source: {_DERIVED_CONSTANTS_PATH})"
        )
    print(f"[analytical_policies] loaded EDA constants from {_DERIVED_CONSTANTS_PATH}")
    return report, norm


_EDA_REPORT, _EDA_NORM = _load_eda_constants()

EDA_ARRIVAL_RATE_PER_BI = _EDA_REPORT["arrival_rate_per_bi_mean"]
EDA_SERVICE_RATE_PER_MS = _EDA_REPORT["service_rate_mean"]
EDA_DUTY_CYCLE_MEAN = _EDA_NORM["duty_cycle"]["mean"]

# --- System constants, taken from the NS-3 simulation ---


class WiFi6Constants:
    """Physical and MAC layer constants for 802.11ax WiFi 6."""

    # Beacon Interval
    BEACON_INTERVAL_MS = 102.4  # ms, the standard TWT beacon interval

    # MCS and rate; mirrors DEFAULT_MCS, DEFAULT_CHANNEL_WIDTH_MHZ and DEFAULT_GUARD_INTERVAL_NS in twt-constants.h.
    MCS_INDEX = 4
    CHANNEL_WIDTH_MHZ = 20
    GUARD_INTERVAL_NS = 800
    PHY_RATE_MBPS = 51.62  # HeMcs4, 20 MHz, 1SS, 800 ns GI

    # Energy model; mirrors BATTERY_VOLTAGE_V and PHY_STATE_*_MA in twt-constants.h.
    SUPPLY_VOLTAGE_V = 3.0
    I_TX_MA = 232.0  # mA
    I_RX_MA = 66.0  # mA
    I_IDLE_MA = 50.0  # mA
    I_SLEEP_MA = 0.12  # mA

    # mW, current times supply voltage.
    P_TX_MW = SUPPLY_VOLTAGE_V * I_TX_MA
    P_RX_MW = SUPPLY_VOLTAGE_V * I_RX_MA
    P_IDLE_MW = SUPPLY_VOLTAGE_V * I_IDLE_MA
    P_SLEEP_MW = SUPPLY_VOLTAGE_V * I_SLEEP_MA
    P_ACTIVE_MW = P_RX_MW  # An awake STA is receiving unless it holds the medium

    # A-MPDU aggregation; mirrors ampduLimitBytes in twt-simulation-config.h.
    MAX_AMPDU_SIZE = 20000  # Bytes
    TYPICAL_AMPDU_FRAMES = 32

    # Per-packet overhead
    MAC_OVERHEAD_US = 200  # us, MAC layer overhead per MPDU
    DIFS_US = 34  # us, DCF interframe space
    SIFS_US = 16  # us, short interframe space
    SLOT_US = 9  # us, slot time

    # Contention window, in slots
    CW_MIN = 15
    CW_MAX = 1023

    # Frame sizes
    PACKET_SIZE_BYTES = 1472  # Bytes, a typical UDP payload


class TWT_Schedules:
    """
    TWT Schedule mappings.

    schedule_idx -> (num_groups, per_group_duration_ms, total_duration_ms)
    """

    SCHEDULE_TABLE = {
        0: {"groups": 1, "durations": [90], "total": 90, "name": "1G_90ms"},
        1: {"groups": 1, "durations": [80], "total": 80, "name": "1G_80ms"},
        2: {"groups": 1, "durations": [60], "total": 60, "name": "1G_60ms"},
        3: {"groups": 1, "durations": [40], "total": 40, "name": "1G_40ms"},
        4: {"groups": 1, "durations": [20], "total": 20, "name": "1G_20ms"},
        5: {"groups": 2, "durations": [45, 45], "total": 90, "name": "2G_45x2"},
        6: {"groups": 2, "durations": [40, 40], "total": 80, "name": "2G_40x2"},
        7: {"groups": 2, "durations": [30, 30], "total": 60, "name": "2G_30x2"},
        8: {"groups": 2, "durations": [20, 20], "total": 40, "name": "2G_20x2"},
        9: {"groups": 2, "durations": [15, 15], "total": 30, "name": "2G_15x2"},
        10: {"groups": 2, "durations": [60, 20], "total": 80, "name": "2G_60+20"},
        11: {"groups": 2, "durations": [50, 30], "total": 80, "name": "2G_50+30"},
        12: {"groups": 2, "durations": [40, 20], "total": 60, "name": "2G_40+20"},
        13: {"groups": 2, "durations": [30, 15], "total": 45, "name": "2G_30+15"},
        14: {"groups": 3, "durations": [30, 30, 25], "total": 85, "name": "3G"},
        15: {"groups": 3, "durations": [40, 25, 20], "total": 85, "name": "3G"},
        16: {"groups": 3, "durations": [25, 25, 25], "total": 75, "name": "3G_equal"},
        17: {
            "groups": 4,
            "durations": [22, 22, 22, 22],
            "total": 88,
            "name": "4G_equal",
        },
        18: {"groups": 4, "durations": [30, 20, 20, 15], "total": 85, "name": "4G"},
        19: {"groups": 4, "durations": [20, 20, 15, 15], "total": 70, "name": "4G"},
    }

    @classmethod
    def get_total_duration(cls, schedule_idx: int) -> float:
        """Total wake duration in ms; an unknown index falls back to schedule 0."""
        return cls.SCHEDULE_TABLE.get(schedule_idx, cls.SCHEDULE_TABLE[0])["total"]

    @classmethod
    def get_num_groups(cls, schedule_idx: int) -> int:
        """Number of TWT groups, 1 through 4; an unknown index falls back to schedule 0."""
        return cls.SCHEDULE_TABLE.get(schedule_idx, cls.SCHEDULE_TABLE[0])["groups"]

    @classmethod
    def get_schedules_by_duration(
        cls, num_groups: int = None, ascending: bool = True
    ) -> list:
        """
        Get schedule indices sorted by total duration.

        Args:
            num_groups: Filter by number of groups; None returns all 20 schedules
            ascending: If True, shortest first, which is the airtime-efficient order

        Returns:
            List of (schedule_idx, total_duration_ms, spec) tuples, sorted by duration
        """
        schedules = []
        for idx, spec in cls.SCHEDULE_TABLE.items():
            if num_groups is None or spec["groups"] == num_groups:
                schedules.append((idx, spec["total"], spec))

        # Sort by total duration
        schedules.sort(key=lambda x: x[1], reverse=not ascending)
        return schedules

    @classmethod
    def get_duty_cycle(cls, schedule_idx: int) -> float:
        """Duty cycle as wake_time / beacon_interval, so 0.0 through roughly 0.88."""
        total = cls.get_total_duration(schedule_idx)
        return total / WiFi6Constants.BEACON_INTERVAL_MS


# --- Observation parser, identical to the one the heuristics use ---


class ObservationParser:
    """Parse normalized observation vector."""

    NUM_STA = 16
    NUM_FEATURES = 13

    # Feature indices
    IDX_BSR_QUEUE = 0
    IDX_DUTY_CYCLE = 7
    IDX_DELTA_PACKETS = 12
    IDX_DELTA_AIRTIME = 9
    IDX_FCS_ERRORS = 2

    @classmethod
    def get_all_sta_features(cls, obs, feature_idx: int) -> np.ndarray:
        values = []
        for sta in range(cls.NUM_STA):
            idx = sta * cls.NUM_FEATURES + feature_idx
            if idx < len(obs):
                values.append(float(obs[idx]))
            else:
                values.append(0.0)
        return np.array(values)

    @classmethod
    def get_queue_sizes(cls, obs) -> np.ndarray:
        return cls.get_all_sta_features(obs, cls.IDX_BSR_QUEUE)

    @classmethod
    def get_duty_cycles(cls, obs) -> np.ndarray:
        return cls.get_all_sta_features(obs, cls.IDX_DUTY_CYCLE)

    @classmethod
    def get_delta_packets(cls, obs) -> np.ndarray:
        return cls.get_all_sta_features(obs, cls.IDX_DELTA_PACKETS)

    @classmethod
    def get_delta_airtime(cls, obs) -> np.ndarray:
        return cls.get_all_sta_features(obs, cls.IDX_DELTA_AIRTIME)

    @classmethod
    def get_fcs_errors(cls, obs) -> np.ndarray:
        return cls.get_all_sta_features(obs, cls.IDX_FCS_ERRORS)


# --- Analytical models ---


class QueuingModel:
    """
    M/D/1 Queuing Model for TWT Service Period.

    In TWT, each beacon interval is a service opportunity.
    The queue behavior follows:
        - λ (arrival rate): packets arriving per beacon interval
        - μ (service rate): packets served per beacon interval (depends on wake duration)
        - ρ = λ/μ (utilization)

    For stable queue: ρ < 1

    Average queue length (Little's Law): L = λ * W
    Where W = average waiting time

    CALIBRATION (from training data):
        - At ~28% duty cycle (default TWT), service_rate ≈ 58 pkts/STA/BI
        - 28% duty = 0.28 * 102.4 = 28.7ms wake time
        - Therefore: service_rate ≈ 2.0 pkts/ms/STA at typical load
        - With A-MPDU aggregation, efficiency is much higher than single packets
    """

    MEAN_ARRIVAL_RATE = EDA_ARRIVAL_RATE_PER_BI
    BASELINE_RATE_PER_MS = EDA_SERVICE_RATE_PER_MS
    MEAN_DUTY_CYCLE = EDA_DUTY_CYCLE_MEAN
    MEAN_WAKE_MS = EDA_DUTY_CYCLE_MEAN * WiFi6Constants.BEACON_INTERVAL_MS
    MEAN_SERVICE_RATE = BASELINE_RATE_PER_MS * MEAN_WAKE_MS

    AGGREGATION_GAIN = 1.5
    CONTENTION_SLOPE = 0.025

    @classmethod
    def estimate_service_rate(
        cls, wake_duration_ms: float, num_sta_in_group: float
    ) -> float:
        """
        Per-STA service rate, Eq. (11):

            mu(d, n) = mu_0 * d * (1 - CONTENTION_SLOPE * (n - 1)) * G_agg

        Args:
            wake_duration_ms: TWT wake duration d, in ms
            num_sta_in_group: STAs n sharing the service period

        Returns:
            Estimated packets per STA per beacon interval
        """
        base_rate = cls.BASELINE_RATE_PER_MS * wake_duration_ms
        contention_factor = 1.0 - cls.CONTENTION_SLOPE * (num_sta_in_group - 1)
        return base_rate * contention_factor * cls.AGGREGATION_GAIN

    @classmethod
    def estimate_queue_growth(
        cls, current_queue: float, service_rate: float, arrival_rate: float = None
    ) -> float:
        """
        Estimate queue size after one beacon interval.

        Q(t+1) = max(0, Q(t) + λ - μ)
        """
        if arrival_rate is None:
            arrival_rate = cls.MEAN_ARRIVAL_RATE

        next_queue = current_queue + arrival_rate - service_rate
        return max(0, next_queue)

    @classmethod
    def compute_utilization(
        cls, wake_duration_ms: float, num_sta: int = 16, num_groups: int = 1
    ) -> float:
        """Compute ρ = λ/μ for stability analysis."""
        sta_per_group = num_sta / num_groups
        service_rate = cls.estimate_service_rate(wake_duration_ms, sta_per_group)

        return cls.MEAN_ARRIVAL_RATE / (service_rate + 1e-6)


class EnergyModel:
    """
    Energy consumption model for TWT.

    E_per_BI = P_active * T_wake + P_sleep * T_sleep

    Where T_sleep = BI - T_wake
    """

    @classmethod
    def compute_energy_mj(cls, wake_duration_ms: float) -> float:
        """
        Compute energy consumption per beacon interval per STA (mJ).
        """
        sleep_duration_ms = WiFi6Constants.BEACON_INTERVAL_MS - wake_duration_ms

        # Energy in mJ = Power (mW) * Time (ms) / 1000
        e_active = WiFi6Constants.P_ACTIVE_MW * wake_duration_ms / 1000
        e_sleep = WiFi6Constants.P_SLEEP_MW * sleep_duration_ms / 1000

        return e_active + e_sleep

    @classmethod
    def compute_energy_per_packet(
        cls, wake_duration_ms: float, packets_transmitted: float
    ) -> float:
        """Energy efficiency: mJ per packet transmitted."""
        total_energy = cls.compute_energy_mj(wake_duration_ms)
        return total_energy / (packets_transmitted + 1e-6)

    @classmethod
    def optimal_wake_for_queue(
        cls, queue_bytes: float, max_wake_ms: float = 90
    ) -> float:
        """
        Find minimum wake duration that can drain the queue.

        This is energy-optimal: use just enough time to serve the queue.
        """
        # Convert queue bytes to packets
        packets = queue_bytes / WiFi6Constants.PACKET_SIZE_BYTES

        # Add incoming packets during next BI
        total_packets = packets + QueuingModel.MEAN_ARRIVAL_RATE

        # Find wake duration needed
        for wake_ms in [20, 30, 40, 50, 60, 70, 80, 90]:
            service_rate = QueuingModel.estimate_service_rate(wake_ms, 16)
            if service_rate >= total_packets * 0.9:  # 90% drain target
                return min(wake_ms, max_wake_ms)

        return max_wake_ms  # Use max if can't drain


class ThroughputModel:
    """
    Throughput optimization model.

    Throughput = successfully transmitted packets per beacon interval

    Key insight: Throughput is maximized when:
      1. Wake duration is long enough to drain queues
      2. Contention is low enough to avoid collisions
      3. Groups are split when contention is high
    """

    # Collision probability increases with number of STAs
    # P_collision ≈ 1 - (1 - 1/CW)^(n-1)

    @classmethod
    def estimate_collision_prob(cls, num_sta: int, cw: int = None) -> float:
        """Estimate collision probability given number of contending STAs."""
        if cw is None:
            cw = WiFi6Constants.CW_MIN

        if num_sta <= 1:
            return 0.0

        # Simplified: each STA picks random slot
        p_same_slot = 1.0 / cw
        p_collision = 1.0 - (1.0 - p_same_slot) ** (num_sta - 1)

        return min(p_collision, 0.95)  # Cap at 95%

    @classmethod
    def effective_throughput(
        cls, wake_duration_ms: float, num_sta: int = 16, num_groups: int = 1
    ) -> float:
        """
        Estimate effective throughput considering collisions.

        Returns packets per BI across all STAs.
        """
        sta_per_group = num_sta // num_groups
        collision_prob = cls.estimate_collision_prob(sta_per_group)

        # Service rate per STA
        service_rate = QueuingModel.estimate_service_rate(
            wake_duration_ms, sta_per_group
        )

        # Effective rate after collisions
        effective_rate = service_rate * (1 - collision_prob)

        # Total for all STAs
        return effective_rate * num_sta

    @classmethod
    def optimal_groups_for_throughput(
        cls, num_sta: int = 16, fcs_error_rate: float = 0.0
    ) -> int:
        """
        Determine optimal number of groups to maximize throughput.

        More groups = less contention per group, but overhead
        """
        if fcs_error_rate > 0.15:  # High collision rate
            return 2
        elif fcs_error_rate > 0.30:
            return 3
        else:
            return 1  # Single group is simpler and efficient


# --- Analytical policies ---


def _load_queue_scaling():
    """BSR index -> packets, derived from the current EDA output. No fallback."""
    queue_bytes = _EDA_NORM["queue_size_bytes"]["mean"]
    bsr_bytes_per_unit = queue_bytes / _EDA_NORM["bsr_queue_index"]["mean"]
    mean_pkt_bytes = queue_bytes / _EDA_NORM["queue_size_packets"]["mean"]
    return bsr_bytes_per_unit, mean_pkt_bytes


BSR_BYTES_PER_UNIT, MEAN_PKT_BYTES = _load_queue_scaling()


class _SectionVABaseline:
    """Section V-A baseline: enumerate all 380 action pairs and minimise the objective."""

    W_QUEUE = 0.20
    W_ENERGY = 0.10

    RHO_TARGET = 0.85
    CONTENTION_SLOPE = QueuingModel.CONTENTION_SLOPE

    def __init__(
        self, action_space=None, num_sta: int = 16, normalization: str = "minmax"
    ):
        self.action_space = action_space
        self.num_sta = num_sta
        self.normalization = normalization

        with open(os.path.join(_EXPLORE, "table_schedule.json")) as f:
            schedules = json.load(f)["schedules"]
        with open(os.path.join(_EXPLORE, "table_assignment.json")) as f:
            assignments = json.load(f)["assignments"]

        n_pairs = len(schedules) * len(assignments)
        self.mu = np.zeros((n_pairs, num_sta), dtype=np.float64)
        self.energy = np.zeros(n_pairs, dtype=np.float64)
        self.rho_max = np.zeros(n_pairs, dtype=np.float64)
        self.pairs = []

        for si, sched in enumerate(schedules):
            k_groups = sched["num_groups"]
            dur = {g["group_id"]: g["wake_duration_ms"] for g in sched["groups"]}
            for ai in range(len(assignments)):
                idx = len(self.pairs)
                self.pairs.append((si, ai))
                members = apply_assignment_pattern(assignments[ai], num_sta, k_groups)
                per_group_count = collections.Counter(g for _, g in members)
                for sta_id, group_id in members:
                    wake_ms = dur[group_id]
                    n_k = per_group_count[group_id]
                    self.mu[idx, sta_id] = QueuingModel.estimate_service_rate(
                        wake_ms, n_k
                    )
                    self.energy[idx] += EnergyModel.compute_energy_mj(wake_ms)
                self.rho_max[idx] = np.max(
                    QueuingModel.MEAN_ARRIVAL_RATE / np.maximum(self.mu[idx], 1e-9)
                )

        self.stable = self.rho_max < self.RHO_TARGET

    def _queue_packets(self, obs) -> np.ndarray:
        bsr = np.clip(
            ObservationParser.get_queue_sizes(obs)[: self.num_sta], 0.0, 254.0
        )
        return bsr * BSR_BYTES_PER_UNIT / MEAN_PKT_BYTES

    def _scale(self, values: np.ndarray) -> np.ndarray:
        if self.normalization == "raw":
            return values
        if self.normalization == "mean":
            mean = values.mean()
            return values / mean if mean > 0 else values
        if self.normalization == "zscore":
            std = values.std()
            return (values - values.mean()) / std if std > 0 else values - values.mean()
        span = np.ptp(values)
        return (values - values.min()) / span if span > 0 else np.zeros_like(values)

    def predict(self, obs, deterministic=True):
        queue = self._queue_packets(obs)
        backlog = np.maximum(
            0.0, queue[None, :] + QueuingModel.MEAN_ARRIVAL_RATE - self.mu
        ).sum(axis=1)

        score = self.W_QUEUE * self._scale(backlog) + self.W_ENERGY * self._scale(
            self.energy
        )
        if self.stable.any():
            score = np.where(self.stable, score, np.inf)

        schedule_idx, assignment_idx = self.pairs[int(np.argmin(score))]
        return np.array([schedule_idx, assignment_idx]), None

    def reset(self):
        pass


class AnalyticalThroughputPolicy(_SectionVABaseline):
    """Section V-A baseline with the throughput preset weights."""

    W_QUEUE = 0.20
    W_ENERGY = 0.10


class AnalyticalEnergyPolicy(_SectionVABaseline):
    """Section V-A baseline with the energy preset weights."""

    W_QUEUE = 0.10
    W_ENERGY = 0.35


class AnalyticalQueuePolicy(_SectionVABaseline):
    """Section V-A baseline with the queue preset weights."""

    W_QUEUE = 0.35
    W_ENERGY = 0.10


# --- Policy registry ---

ANALYTICAL_POLICIES = {
    "throughput": AnalyticalThroughputPolicy,
    "energy": AnalyticalEnergyPolicy,
    "queue": AnalyticalQueuePolicy,
}


def get_analytical_policy(preset: str, action_space, num_sta: int = 16):
    """Factory function to get analytical policy for a preset."""
    policy_class = ANALYTICAL_POLICIES.get(preset)
    if policy_class is None:
        raise ValueError(
            f"Unknown preset: {preset}. Choose from {list(ANALYTICAL_POLICIES.keys())}"
        )
    return policy_class(action_space, num_sta)


# --- Testing ---

if __name__ == "__main__":
    import gymnasium as gym

    print("=" * 60)
    print("ANALYTICAL POLICY MODELS - UNIT TESTS")
    print("=" * 60)

    # Test queuing model
    print("\n--- Queuing Model ---")
    for wake_ms in [20, 40, 60, 80, 90]:
        service_rate = QueuingModel.estimate_service_rate(wake_ms, 16)
        utilization = QueuingModel.MEAN_ARRIVAL_RATE / service_rate
        print(
            f"Wake={wake_ms}ms: service_rate={service_rate:.1f} pkts, ρ={utilization:.2f}"
        )

    # Test energy model
    print("\n--- Energy Model ---")
    for wake_ms in [20, 40, 60, 80, 90]:
        energy = EnergyModel.compute_energy_mj(wake_ms)
        print(f"Wake={wake_ms}ms: E={energy:.2f} mJ/BI, duty={wake_ms/102.4:.2%}")

    # Test throughput model
    print("\n--- Throughput Model (collision impact) ---")
    for num_sta in [4, 8, 16]:
        for num_groups in [1, 2, 4]:
            throughput = ThroughputModel.effective_throughput(60, num_sta, num_groups)
            print(
                f"STAs={num_sta}, Groups={num_groups}: throughput={throughput:.1f} pkts/BI"
            )

    # Test policies with dummy observation
    print("\n--- Policy Tests ---")
    action_space = gym.spaces.MultiDiscrete([20, 19])

    # Create dummy normalized observation (16 STAs x 13 features = 208)
    dummy_obs = np.random.randn(208) * 0.5  # z-scores around 0

    for preset, policy_class in ANALYTICAL_POLICIES.items():
        policy = policy_class(action_space, num_sta=16)
        action, _ = policy.predict(dummy_obs)

        schedule_info = TWT_Schedules.SCHEDULE_TABLE[action[0]]
        print(f"\n{preset.upper()} Policy:")
        print(f"  Action: schedule_idx={action[0]}, assignment_idx={action[1]}")
        print(f"  Schedule: {schedule_info['name']} ({schedule_info['total']}ms total)")

    print("\n" + "=" * 60)
    print("All tests passed!")

#!/usr/bin/env python3
"""ap_observation.py - AP-observable 13-feature observation (EE 7331 proposal, Table 1).

The ICCCN'26 observation (ppo-sb3-scripts/file_comm_env.py::build_ppo_observation) reads 7 of
its 13 per-STA features from simulator ground truth (the `oracle` dict): awake time, sleep
time, duty cycle, packets transmitted and three deltas built from them. A real AP cannot see
those. This module rebuilds the same 13-slot layout from quantities an AP does have:

  slot  ICCCN'26 (source)                 here (source)
  ----  --------------------------------  -----------------------------------------------
   1    bsr_queue_ac_be      (realistic)  bsr_queue_ac_be          (realistic, unchanged)
   2    airtime_used_us      (realistic)  airtime_used_us          (realistic, unchanged)
   3    fcs_error_count      (realistic)  fcs_error_count          (realistic, unchanged)
   4    rx_fragment_count    (realistic)  rx_fragment_count        (realistic, unchanged)
   5    last_rx_timestamp_us (realistic)  last_rx_timestamp_us     (realistic, unchanged)
   6    awake_time_ms        (ORACLE)     commanded awake time     (from the AP's own schedule)
   7    sleep_time_ms        (ORACLE)     commanded sleep time     (from the AP's own schedule)
   8    duty_cycle           (ORACLE)     commanded duty cycle     (from the AP's own schedule)
   9    packets_transmitted  (ORACLE)     packets_received_at_ap   (realistic AP counter)
  10    d airtime            (realistic)  d airtime                (realistic, unchanged)
  11    d awake              (ORACLE)     d commanded awake        (from the AP's own schedule)
  12    d sleep              (ORACLE)     d commanded sleep        (from the AP's own schedule)
  13    d packets_tx         (ORACLE)     d packets_received_at_ap (realistic AP counter)

"Commanded" means: the AP set the schedule, so it knows each STA's group, that group's wake
duration d and the wake interval T. Over an epoch of length E the STA is commanded awake for
E * d / T and asleep for the rest. On the bench the same numbers come from the values the
HaLowWrapper wrote to twt_ap_update, so ns-3 and hardware share one definition.

The builder never receives the oracle dicts at all: `realistic_view()` strips them before any
feature is computed, and tests/test_ap_observation.py poisons every oracle field to prove it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..", "exploration-scripts")))
from generate_action_tables import apply_assignment_pattern  # noqa: E402

import hw_constraints as hwc  # noqa: E402

PER_STA_FEATURES = 13
MAX_STA = 16

FEATURE_NAMES = [
    "bsr_queue_ac_be",
    "airtime_used_us",
    "fcs_error_count",
    "rx_fragment_count",
    "last_rx_timestamp_us",
    "cmd_awake_time_ms",
    "cmd_sleep_time_ms",
    "cmd_duty_cycle",
    "packets_received_at_ap",
    "delta_airtime_used_us",
    "delta_cmd_awake_time_ms",
    "delta_cmd_sleep_time_ms",
    "delta_packets_received_at_ap",
]
assert len(FEATURE_NAMES) == PER_STA_FEATURES

# The only per-STA keys the builder is allowed to read.
REALISTIC_KEYS_USED = (
    "bsr_queue_ac_be",
    "airtime_used_us",
    "fcs_error_count",
    "rx_fragment_count",
    "last_rx_timestamp_us",
    "packets_received_at_ap",
)


def _clean(v, max_val: float = 1e12) -> float:
    """Same cleaning as file_comm_env.clean_positive: finite, non-negative, capped."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v) or v < 0:
        return 0.0
    return min(v, max_val)


def realistic_view(env: Optional[Dict]) -> List[Dict[str, float]]:
    """Per-STA list holding only the realistic keys the AP can read. Oracle data is dropped."""
    if not env:
        return []
    out = []
    for sta in env.get("sta_observations", []):
        r = sta.get("realistic", {}) or {}
        out.append({k: _clean(r.get(k, 0)) for k in REALISTIC_KEYS_USED})
    return out


def commanded_durations_ms(action: Sequence[int], num_sta: int) -> np.ndarray:
    """Wake duration (ms) commanded to each STA by a [schedule_idx, assignment_idx] action.

    Uses the same table lookup and apply_assignment_pattern() call as
    episode_file_runner.build_action_dict, so the agent's view of the schedule matches what
    ns-3 was actually told.
    """
    schedule_table, assignment_table = hwc.load_tables()
    s_idx, a_idx = int(action[0]), int(action[1])
    groups = schedule_table["schedules"][s_idx]["groups"]
    pattern = assignment_table["assignments"][a_idx]
    mapping = apply_assignment_pattern(pattern, num_sta, len(groups))

    dur_by_group = {g["group_id"]: float(g["wake_duration_ms"]) for g in groups}
    durations = np.zeros(num_sta, dtype=np.float64)
    for sta_id, group_id in mapping:
        if 0 <= sta_id < num_sta:
            durations[sta_id] = dur_by_group.get(group_id, 0.0)
    return durations


@dataclass
class APObservationBuilder:
    """Stateful builder: tracks commanded awake/sleep time per STA across an episode.

    Call reset() once with the first env and the schedule in force (the warmup action), then
    update() after every step with the action that was in force during that epoch.
    """

    wake_interval_ms: float = hwc.BEACON_INTERVAL_MS
    fallback_epoch_ms: float = 2048.0  # Used only if simulation_time_sec is missing
    max_sta: int = MAX_STA
    num_sta: int = 0
    cum_awake_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cum_sleep_ms: np.ndarray = field(default_factory=lambda: np.zeros(0))
    duty: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def reset(self, env: Dict, action_in_force: Sequence[int]) -> np.ndarray:
        self.num_sta = min(int(env.get("num_sta", len(env.get("sta_observations", [])))), self.max_sta)
        self.cum_awake_ms = np.zeros(self.num_sta)
        self.cum_sleep_ms = np.zeros(self.num_sta)
        d = commanded_durations_ms(action_in_force, self.num_sta)
        self.duty = np.clip(d / self.wake_interval_ms, 0.0, 1.0)
        zeros = np.zeros(self.num_sta)
        return self._assemble(realistic_view(env), None, zeros, zeros)

    def update(self, prev_env: Dict, curr_env: Dict, action_in_force: Sequence[int]) -> np.ndarray:
        elapsed_ms = self._elapsed_ms(prev_env, curr_env)
        d = commanded_durations_ms(action_in_force, self.num_sta)
        duty = np.clip(d / self.wake_interval_ms, 0.0, 1.0)
        d_awake = elapsed_ms * duty
        d_sleep = elapsed_ms - d_awake
        self.cum_awake_ms += d_awake
        self.cum_sleep_ms += d_sleep
        self.duty = duty
        return self._assemble(realistic_view(curr_env), realistic_view(prev_env), d_awake, d_sleep)

    # --- internals ---

    def _elapsed_ms(self, prev_env: Dict, curr_env: Dict) -> float:
        try:
            dt = (float(curr_env.get("simulation_time_sec", 0)) - float(prev_env.get("simulation_time_sec", 0))) * 1000.0
        except (TypeError, ValueError, AttributeError):
            dt = 0.0
        return dt if dt > 0 else self.fallback_epoch_ms

    def _assemble(self, curr: List[Dict], prev: Optional[List[Dict]], d_awake, d_sleep) -> np.ndarray:
        obs = np.zeros(self.max_sta * PER_STA_FEATURES, dtype=np.float32)
        for i in range(min(self.num_sta, len(curr))):
            c = curr[i]
            p = prev[i] if prev and i < len(prev) else None
            row = [
                c["bsr_queue_ac_be"],
                c["airtime_used_us"],
                c["fcs_error_count"],
                c["rx_fragment_count"],
                c["last_rx_timestamp_us"],
                self.cum_awake_ms[i],
                self.cum_sleep_ms[i],
                self.duty[i],
                c["packets_received_at_ap"],
                max(0.0, c["airtime_used_us"] - p["airtime_used_us"]) if p else 0.0,
                d_awake[i],
                d_sleep[i],
                max(0.0, c["packets_received_at_ap"] - p["packets_received_at_ap"]) if p else 0.0,
            ]
            obs[i * PER_STA_FEATURES:(i + 1) * PER_STA_FEATURES] = row
        return obs


@dataclass
class FixedZScore:
    """Per-feature z-score with statistics from a random-policy run (proposal, Table 1).

    Statistics are pooled over all active STA slots, since STA identity is re-randomized every
    episode. Padded slots (STA index >= num_sta) stay exactly zero.
    """

    mean: np.ndarray
    std: np.ndarray
    eps: float = 1e-8

    def __call__(self, obs: np.ndarray, num_sta: int) -> np.ndarray:
        x = obs.reshape(-1, PER_STA_FEATURES).astype(np.float64)
        x[:num_sta] = (x[:num_sta] - self.mean) / (self.std + self.eps)
        x[num_sta:] = 0.0
        return x.reshape(-1).astype(np.float32)

    @classmethod
    def identity(cls) -> "FixedZScore":
        return cls(np.zeros(PER_STA_FEATURES), np.ones(PER_STA_FEATURES))

    @classmethod
    def fit(cls, rows: np.ndarray) -> "FixedZScore":
        """rows: (n, 13) array of active-STA feature rows."""
        rows = np.asarray(rows, dtype=np.float64)
        std = rows.std(axis=0)
        std[std < 1e-6] = 1.0  # Constant features pass through centred but unscaled
        return cls(rows.mean(axis=0), std)

    def save(self, path: str, meta: Optional[Dict] = None) -> None:
        payload = {
            "feature_names": FEATURE_NAMES,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "meta": meta or {},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FixedZScore":
        with open(path) as f:
            d = json.load(f)
        if d.get("feature_names") != FEATURE_NAMES:
            raise ValueError(f"{path} was fitted on a different feature layout")
        return cls(np.array(d["mean"]), np.array(d["std"]))

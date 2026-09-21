#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""eval_policy.py - evaluate a trained PPO policy for TWT WiFi scheduling.

Benchmarks a trained PPO model against any combination of three baselines, each requested by its own flag:
  - Random, via --compare-random
  - Preset-aligned adaptive heuristics, via --compare-heuristic
  - Analytical model-based policies from analytical_policies.py, via --compare-analytical

Every policy in a run is scored under the same --reward-type, so the numbers are comparable.
PPO alone runs when no --compare flag is given; run_eval.sh drives the usual suite across all three presets.

Usage:
    python3.11 eval_policy.py checkpoints/<run_id>/ppo_V1_twt_*/ppo_V1_twt_final.zip
    python3.11 eval_policy.py --policy random --n-episodes 10
    python3.11 eval_policy.py model.zip --compare-heuristic throughput
    python3.11 eval_policy.py model.zip --compare-analytical queue --reward-type queue

Lab: SHINE Lab, Texas State University
"""

import os
import sys
import json
import inspect
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict
import numpy as np
import gymnasium as gym

# Add current directory for local imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

# Import Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    print(f"Warning: stable-baselines3 not installed. PPO evaluation unavailable.")
    PPO = None
    VecNormalize = None

# Import file-based environment for NS-3 communication
from file_comm_env import FileCommEnv
import run_paths

# --- Baseline policies ---


class RandomPolicy:
    """Random action policy for baseline comparison."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=False):
        action = self.action_space.sample()
        return action, None


# --- Observation parsing helper ---


class ObservationParser:
    """
    Parse PPO observation vector into per-STA metrics.

    PPO Observation (13 features per STA):
        0: bsr_queue_ac_be      - BSR queue INDEX (0-254), NOT bytes! Mean ~100
        1: airtime_used_us      - Cumulative airtime used (microseconds)
        2: fcs_error_count      - FCS error count (collision indicator). Mean ~102
        3: rx_fragment_count    - Received fragment count. Mean ~7.5
        4: last_rx_timestamp_us - Last RX timestamp
        5: awake_time_ms        - Cumulative awake time (ms)
        6: sleep_time_ms        - Cumulative sleep time (ms)
        7: duty_cycle           - Duty cycle (0-1). Mean ~0.28
        8: packets_transmitted  - Cumulative packets transmitted
        9: delta_airtime_used_us  - Mean ~13909 per STA
        10: delta_awake_time_ms   - Mean ~64.5 per STA
        11: delta_sleep_time_ms   - Mean ~211.5 per STA
        12: delta_packets_transmitted - Mean ~58 per STA

    NORM Constants (from reward_functions.py):
        bsr_queue_index: mean=100.7, std=114.4, max=254
        fcs_error_count: mean=102, std=190, max=500
        duty_cycle: mean=0.28, std=0.10
        delta_packets_transmitted: mean=58.1, std=55.9
        delta_airtime_used_us: mean=13909, std=22947
    """

    NUM_STA = 16
    NUM_FEATURES = 13

    # Feature indices
    IDX_BSR_QUEUE = 0
    IDX_AIRTIME = 1
    IDX_FCS_ERRORS = 2
    IDX_RX_FRAGMENTS = 3
    IDX_LAST_RX_TS = 4
    IDX_AWAKE_TIME = 5
    IDX_SLEEP_TIME = 6
    IDX_DUTY_CYCLE = 7
    IDX_PACKETS_TX = 8
    IDX_DELTA_AIRTIME = 9
    IDX_DELTA_AWAKE = 10
    IDX_DELTA_SLEEP = 11
    IDX_DELTA_PACKETS = 12

    @classmethod
    def get_sta_feature(cls, obs, sta_idx: int, feature_idx: int) -> float:
        """Get a specific feature for a specific STA."""
        idx = sta_idx * cls.NUM_FEATURES + feature_idx
        if idx < len(obs):
            return float(obs[idx])
        return 0.0

    @classmethod
    def get_all_sta_features(cls, obs, feature_idx: int) -> np.ndarray:
        """Get a specific feature across all STAs."""
        values = []
        for sta in range(cls.NUM_STA):
            values.append(cls.get_sta_feature(obs, sta, feature_idx))
        return np.array(values)

    @classmethod
    def get_queue_sizes(cls, obs) -> np.ndarray:
        """Get BSR queue sizes for all STAs."""
        return cls.get_all_sta_features(obs, cls.IDX_BSR_QUEUE)

    @classmethod
    def get_duty_cycles(cls, obs) -> np.ndarray:
        """Get duty cycles for all STAs."""
        return cls.get_all_sta_features(obs, cls.IDX_DUTY_CYCLE)

    @classmethod
    def get_fcs_errors(cls, obs) -> np.ndarray:
        """Get FCS error counts for all STAs."""
        return cls.get_all_sta_features(obs, cls.IDX_FCS_ERRORS)

    @classmethod
    def get_delta_packets(cls, obs) -> np.ndarray:
        """Get delta packets transmitted for all STAs."""
        return cls.get_all_sta_features(obs, cls.IDX_DELTA_PACKETS)

    @classmethod
    def get_delta_airtime(cls, obs) -> np.ndarray:
        """Get delta airtime used for all STAs."""
        return cls.get_all_sta_features(obs, cls.IDX_DELTA_AIRTIME)


# --- Preset-aligned adaptive heuristics ---
#
# These 3 heuristics correspond to the 3 reward presets in reward_functions.py:
#   1. ThroughputAdaptiveHeuristic - for 'throughput' preset
#   2. EnergyAdaptiveHeuristic     - for 'energy' preset
#   3. QueueAdaptiveHeuristic      - for 'queue' preset (latency)
#
# Each heuristic optimizes for the same objectives as its corresponding preset.


class ThroughputAdaptiveHeuristic:
    """
    Adaptive heuristic for 'throughput' preset.

    Matches reward_functions.py PRESET_WEIGHTS['throughput']:
        throughput: 0.35 (PRIMARY)
        queue: 0.20 (SECONDARY - queue affects throughput)
        drops: 0.15 (SECONDARY - drops = lost throughput)
        airtime: 0.15 (penalizes long TWT schedules)
        energy: 0.10 (TERTIARY)
        channel: 0.05 (TERTIARY)

    Strategy:
    - Maximize bytes transmitted (use long wake durations)
    - Split groups when contention detected (FCS errors high, efficiency low)
    - Minimize drops by ensuring adequate airtime
    - Fairness bonus: split if queue variance is high

    NORM constants:
    - bsr_queue_index: mean=100.7, std=114.4
    - delta_packets_transmitted: mean=58.1, std=55.9
    - delta_airtime_used_us: mean=13909, std=22947
    - fcs_error_count: mean=102 (total ~1632 for 16 STAs)
    """

    BSR_MEAN = 100.7
    BSR_STD = 114.4
    FCS_TOTAL_MEAN = 102.0 * 16  # ~1632
    EXPECTED_EFFICIENCY = 58.1 / 13909.0  # ~0.0042 packets/us

    def __init__(self, action_space, num_sta: int = 16):
        self.action_space = action_space
        self.num_sta = num_sta

    def predict(self, obs, deterministic=False):
        queues = ObservationParser.get_queue_sizes(obs)
        fcs_errors = ObservationParser.get_fcs_errors(obs)
        delta_packets = ObservationParser.get_delta_packets(obs)
        delta_airtime = ObservationParser.get_delta_airtime(obs)

        avg_queue = np.mean(queues)
        max_queue = np.max(queues)
        std_queue = np.std(queues)
        total_fcs = np.sum(fcs_errors)
        total_packets = np.sum(delta_packets)
        total_airtime = np.sum(delta_airtime)

        # Efficiency: packets per microsecond
        efficiency = (
            total_packets / (total_airtime + 1e-6)
            if total_airtime > 0
            else self.EXPECTED_EFFICIENCY
        )

        # Note: This policy receives RAW observations (not normalized)
        # VecNormalize is only applied to PPO policy during evaluation

        # --- Assess conditions (using raw value thresholds) ---
        high_load = avg_queue > 120 or max_queue > 180
        very_high_load = avg_queue > 170 or max_queue > 220
        high_contention = total_fcs > 2000
        unfair = std_queue > 80
        low_efficiency = efficiency < 0.002

        # --- Decision logic (prioritize throughput) ---
        if very_high_load:
            # Maximum throughput mode
            if high_contention or (low_efficiency and unfair):
                # Split to reduce contention
                schedule_idx = 5  # 2 groups x 45ms (90ms total airtime)
                assignment_idx = 4  # Round-robin 2 groups
            else:
                # Keep together with max duration
                schedule_idx = 0  # 90ms single group
                assignment_idx = 0
        elif high_load:
            # High throughput mode
            if high_contention:
                schedule_idx = 5  # 2 groups x 45ms
                assignment_idx = 4
            elif unfair:
                schedule_idx = 6  # 2 groups x 40ms
                assignment_idx = 8  # Even/odd split for fairness
            else:
                schedule_idx = 1  # 80ms single group
                assignment_idx = 0
        else:
            # Moderate load: still prioritize throughput over energy
            if unfair:
                schedule_idx = 7  # 2 groups x 30ms
                assignment_idx = 4
            else:
                schedule_idx = 2  # 60ms - good throughput
                assignment_idx = 0

        return np.array([schedule_idx, assignment_idx]), None


class EnergyAdaptiveHeuristic:
    """
    Adaptive heuristic for 'energy' preset.

    Matches reward_functions.py PRESET_WEIGHTS['energy']:
        energy: 0.35 (PRIMARY)
        throughput: 0.20 (SECONDARY - need some throughput)
        airtime: 0.20 (shorter schedules save energy)
        drops: 0.10 (SECONDARY - retx waste energy)
        queue: 0.10 (TERTIARY)
        channel: 0.05 (TERTIARY)

    Strategy:
    - Minimize wake duration to maximize sleep (save energy)
    - Only increase duration when queue is critically high
    - Avoid splitting groups (overhead wastes energy)
    - Target duty_cycle < 0.28 (NORM mean)

    NORM constants:
    - duty_cycle: mean=0.28, std=0.10 (0.38 = +1 std is "high")
    - bsr_queue_index: mean=100.7, std=114.4
    - delta_energy_mj: mean=33.0, std=65.0
    """

    DUTY_MEAN = 0.28
    DUTY_HIGH = 0.38  # mean + 1 std
    DUTY_LOW = 0.18  # mean - 1 std (target for energy saving)
    BSR_MEAN = 100.7
    BSR_STD = 114.4

    def __init__(self, action_space, num_sta: int = 16):
        self.action_space = action_space
        self.num_sta = num_sta

    def predict(self, obs, deterministic=False):
        queues = ObservationParser.get_queue_sizes(obs)
        duty_cycles = ObservationParser.get_duty_cycles(obs)

        avg_queue = np.mean(queues)
        max_queue = np.max(queues)
        avg_duty = np.mean(duty_cycles)
        min_duty = np.min(duty_cycles)

        # Note: This policy receives RAW observations (not normalized)
        # VecNormalize is only applied to PPO policy during evaluation

        # --- Assess conditions (using raw value thresholds) ---
        critical_queue = max_queue > 220  # Near max BSR
        high_queue = avg_queue > 150
        moderate_queue = avg_queue > 100
        high_duty = avg_duty > self.DUTY_HIGH  # > 0.38
        low_duty = avg_duty < self.DUTY_LOW  # < 0.18

        # --- Decision logic (prioritize energy saving) ---
        if critical_queue:
            # Must provide service to avoid drops (drops waste energy via retx)
            schedule_idx = 2  # 60ms - minimum to handle critical
            assignment_idx = 0  # Keep together (simpler = less overhead)
        elif high_queue:
            # Balance energy vs queue buildup
            schedule_idx = 3  # 40ms
            assignment_idx = 0
        elif moderate_queue:
            # Can reduce wake time
            if high_duty:
                # Already using too much energy, reduce aggressively
                schedule_idx = 4  # 20ms minimum
            else:
                schedule_idx = 3  # 40ms safe
            assignment_idx = 0
        else:
            # Low queue - aggressive energy saving
            if low_duty:
                # Already efficient, maintain
                schedule_idx = 3  # 40ms
            else:
                # Reduce to minimum
                schedule_idx = 4  # 20ms
            assignment_idx = 0

        return np.array([schedule_idx, assignment_idx]), None


class QueueAdaptiveHeuristic:
    """
    Adaptive heuristic for 'queue' preset (latency optimization).

    Matches reward_functions.py PRESET_WEIGHTS['queue']:
        queue: 0.35 (PRIMARY - latency proxy)
        drops: 0.20 (SECONDARY - drops = queue overflow)
        throughput: 0.20 (SECONDARY - need TX to drain)
        energy: 0.10 (TERTIARY)
        airtime: 0.10 (shorter schedules mean lower latency)
        channel: 0.05 (TERTIARY)

    Strategy:
    - Minimize queue sizes to reduce latency
    - React quickly to queue buildup
    - Split groups when variance is high (some STAs starving)
    - Drain ratio is key: packets_tx / packets_enqueued

    NORM constants:
    - bsr_queue_index: mean=100.7, std=114.4
    - queue_size_bytes: mean=25776, std=29288
    - delta_packets_transmitted: mean=58.1, std=55.9
    - delta_packets_enqueued: mean=64.5, std=72.1
    - Expected drain ratio: ~0.90
    """

    BSR_MEAN = 100.7
    BSR_STD = 114.4
    EXPECTED_DRAIN_RATIO = 0.901  # packets_tx / packets_enqueued

    def __init__(self, action_space, num_sta: int = 16):
        self.action_space = action_space
        self.num_sta = num_sta

    def predict(self, obs, deterministic=False):
        queues = ObservationParser.get_queue_sizes(obs)
        delta_packets = ObservationParser.get_delta_packets(obs)

        avg_queue = np.mean(queues)
        max_queue = np.max(queues)
        std_queue = np.std(queues)
        total_packets = np.sum(delta_packets)

        # Note: This policy receives RAW observations (not normalized)
        # VecNormalize is only applied to PPO policy during evaluation

        # --- Assess conditions (using raw value thresholds) ---
        very_high_queue = avg_queue > 180 or max_queue > 220
        high_queue = avg_queue > 130 or max_queue > 180
        moderate_queue = avg_queue > 80
        unfair = std_queue > 80
        low_throughput = total_packets < 800  # 16 STAs * 58.1 * 0.85

        # --- Decision logic (prioritize queue draining) ---
        if very_high_queue:
            # Emergency: drain queues immediately
            if unfair:
                # Some STAs starving - split for fairness
                schedule_idx = 5  # 2 groups x 45ms
                assignment_idx = 4  # Round-robin
            else:
                # All high - max single duration
                schedule_idx = 0  # 90ms
                assignment_idx = 0
        elif high_queue:
            # High queue - focus on draining
            if unfair:
                schedule_idx = 6  # 2 groups x 40ms
                assignment_idx = 8  # Even/odd for fairness
            elif low_throughput:
                # Not draining fast enough
                schedule_idx = 0  # 90ms - max time
                assignment_idx = 0
            else:
                schedule_idx = 1  # 80ms
                assignment_idx = 0
        elif moderate_queue:
            # Moderate queue - balanced approach
            if unfair:
                schedule_idx = 7  # 2 groups x 30ms
                assignment_idx = 4
            else:
                schedule_idx = 2  # 60ms
                assignment_idx = 0
        else:
            # Low queue - can reduce, but don't be too aggressive
            schedule_idx = 3  # 40ms - safe for latency
            assignment_idx = 0

        return np.array([schedule_idx, assignment_idx]), None


# Dictionary of heuristics matching reward presets
# Each heuristic is designed to match its corresponding PPO reward preset
HEURISTIC_POLICIES = {
    "throughput": ThroughputAdaptiveHeuristic,  # For 'throughput' preset
    "energy": EnergyAdaptiveHeuristic,  # For 'energy' preset
    "queue": QueueAdaptiveHeuristic,  # For 'queue' preset
}

# Import analytical model-based policies (more principled than heuristics)
try:
    from analytical_policies import (
        AnalyticalThroughputPolicy,
        AnalyticalEnergyPolicy,
        AnalyticalQueuePolicy,
        ANALYTICAL_POLICIES,
    )

    ANALYTICAL_AVAILABLE = True
except ImportError:
    ANALYTICAL_AVAILABLE = False
    ANALYTICAL_POLICIES = {}
    print("Note: analytical_policies.py not found. Analytical baselines disabled.")


# --- Evaluation functions ---


def evaluate_policy(
    policy,
    env: FileCommEnv,
    n_episodes: int = 10,
    seeds: Optional[List[int]] = None,
    deterministic: bool = True,
    verbose: bool = True,
    policy_name: str = "policy",
    vec_normalize: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate a policy on the TWT environment.

    Args:
        policy: Policy with predict(obs, deterministic) method
        env: Gymnasium environment instance
        n_episodes: Number of episodes to evaluate
        seeds: List of seeds for each episode (or None for random)
        deterministic: Use deterministic actions
        verbose: Print progress
        policy_name: Name for logging
        vec_normalize: Optional VecNormalize wrapper for observation normalization
                       (required for PPO models trained with VecNormalize)

    Returns:
        Dictionary with evaluation results
    """
    supports_recurrent_state = "state" in inspect.signature(policy.predict).parameters

    if seeds is None:
        seeds = [np.random.randint(1000, 99999) for _ in range(n_episodes)]

    if len(seeds) < n_episodes:
        seeds = seeds + [
            np.random.randint(1000, 99999) for _ in range(n_episodes - len(seeds))
        ]

    # Storage for metrics
    episode_rewards = []
    episode_lengths = []
    episode_metrics = defaultdict(list)
    action_distribution = defaultdict(int)

    # Helper function to normalize observation if vec_normalize is provided
    def normalize_obs(obs):
        if vec_normalize is not None:
            # VecNormalize expects shape (n_envs, obs_dim), so add batch dim
            obs_batch = np.array([obs])
            normalized = vec_normalize.normalize_obs(obs_batch)
            return normalized[0]  # Remove batch dim
        return obs

    for ep_idx in range(n_episodes):
        seed = seeds[ep_idx]

        if verbose:
            print(f"[{policy_name}] Episode {ep_idx + 1}/{n_episodes} (seed={seed})")

        obs, info = env.reset(seed=seed)

        # Reset round-robin policy if applicable
        if hasattr(policy, "reset"):
            policy.reset()

        lstm_states = None
        episode_start = True

        episode_reward = 0.0
        episode_length = 0
        ep_metrics = defaultdict(float)

        done = False
        while not done:
            # Normalize observation for PPO (if vec_normalize provided)
            obs_normalized = normalize_obs(obs)

            # Get action from policy
            if supports_recurrent_state:
                action, lstm_states = policy.predict(
                    obs_normalized,
                    state=lstm_states,
                    episode_start=np.array([episode_start]),
                    deterministic=deterministic,
                )
                episode_start = False
            else:
                action, _ = policy.predict(obs_normalized, deterministic=deterministic)

            # Track action distribution
            action_key = f"({action[0]},{action[1]})"
            action_distribution[action_key] += 1

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Accumulate step metrics
            for key in [
                "total_bytes_tx",
                "total_energy_mj",
                "avg_queue_bytes",
                "total_drops",
            ]:
                if key in info:
                    ep_metrics[key] = info[key]  # Take last value (cumulative)

        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        for key, value in ep_metrics.items():
            episode_metrics[key].append(value)

        if verbose:
            print(f"    Reward: {episode_reward:.4f}, Steps: {episode_length}")

    # Compute summary statistics
    results = {
        "policy_name": policy_name,
        "n_episodes": n_episodes,
        "seeds": seeds,
        # Reward statistics
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        # Episode length
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        # Raw data
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        # Per-metric statistics
        "metrics": {},
        # Action distribution
        "action_distribution": dict(action_distribution),
        "unique_actions": len(action_distribution),
    }

    # Compute per-metric statistics
    for key, values in episode_metrics.items():
        if values:
            results["metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }

    return results


def compare_policies(
    policies: Dict[str, Any],
    env: gym.Env,
    n_episodes: int = 10,
    seeds: Optional[List[int]] = None,
    verbose: bool = True,
    vec_normalize: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple policies on the same episodes.

    Args:
        policies: Dict mapping policy names to policy objects
        env: Gymnasium environment instance
        n_episodes: Number of episodes
        seeds: List of seeds (same for all policies for fair comparison)
        verbose: Print progress
        vec_normalize: Optional VecNormalize wrapper for PPO observation normalization

    Returns:
        Dictionary mapping policy names to results
    """
    if seeds is None:
        seeds = [np.random.randint(1000, 99999) for _ in range(n_episodes)]

    all_results = {}

    for policy_name, policy in policies.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {policy_name}")
        print(f"{'='*60}")

        # Only use vec_normalize for PPO - other policies use raw observations
        # This ensures Analytical/Heuristic policies are reproducible across different PPO models
        if policy_name == "PPO":
            use_vec_normalize = vec_normalize
        else:
            use_vec_normalize = None  # Analytical/Heuristic/Random use raw observations

        results = evaluate_policy(
            policy=policy,
            env=env,
            n_episodes=n_episodes,
            seeds=seeds,
            deterministic=True,
            verbose=verbose,
            policy_name=policy_name,
            vec_normalize=use_vec_normalize,
        )

        all_results[policy_name] = results

    return all_results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("POLICY COMPARISON RESULTS")
    print("=" * 80)

    # Header
    print(
        f"{'Policy':<20} {'Mean Reward':>15} {'Std':>10} {'Mean Bytes TX':>15} {'Mean Energy':>12}"
    )
    print("-" * 80)

    for policy_name, result in results.items():
        mean_reward = result.get("mean_reward", 0)
        std_reward = result.get("std_reward", 0)

        metrics = result.get("metrics", {})
        mean_bytes = metrics.get("total_bytes_tx", {}).get("mean", 0)
        mean_energy = metrics.get("total_energy_mj", {}).get("mean", 0)

        print(
            f"{policy_name:<20} {mean_reward:>15.4f} {std_reward:>10.4f} "
            f"{mean_bytes:>15.0f} {mean_energy:>12.2f}"
        )

    print("=" * 80)


def save_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON."""

    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_json = convert(results)

    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Results saved to: {output_path}")


# --- Main evaluation script ---


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO policy for TWT WiFi Scheduling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model/policy specification
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to trained PPO model (.zip)",
    )
    parser.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats (.pkl). If not provided, will look for vecnormalize.pkl in same dir as model.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="ppo",
        choices=["ppo", "random"],
        help="Policy type to evaluate (ppo requires model_path)",
    )

    # Evaluation config
    parser.add_argument(
        "--n-episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed (will generate n_episodes seeds from this)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions",
    )

    # Comparison options
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="Also compare with random baseline policy",
    )
    parser.add_argument(
        "--compare-heuristic",
        type=str,
        nargs="*",
        default=None,
        choices=list(HEURISTIC_POLICIES.keys()),
        help="Compare with heuristic policies: throughput, energy, queue",
    )
    parser.add_argument(
        "--compare-analytical",
        type=str,
        nargs="*",
        default=None,
        choices=["throughput", "energy", "queue"],
        help="Compare with analytical model-based policies (more principled than heuristics)",
    )

    # Environment config
    parser.add_argument(
        "--reward-type",
        type=str,
        default="throughput",
        choices=["throughput", "energy", "queue"],
        help="Reward type (should match training preset)",
    )

    # Output config
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file for results"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for auto-generated output filename (e.g., 'lstm_ppo_V1_twt_')",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    args = parser.parse_args()

    # Generate seeds
    if args.seed is not None:
        np.random.seed(args.seed)
    seeds = [np.random.randint(1000, 99999) for _ in range(args.n_episodes)]

    print("=" * 60)
    print("TWT Policy Evaluation")
    print("=" * 60)
    print(f"Episodes: {args.n_episodes}")
    print(f"Seeds: {seeds[:5]}..." if len(seeds) > 5 else f"Seeds: {seeds}")
    print(f"Reward type: {args.reward_type}")

    # Create file-based environment (fresh process per episode)
    print("\nCreating file-based environment...")
    env = FileCommEnv(
        comm_dir="/tmp/twt_comm_eval",
        seed=seeds[0] if seeds else 1000,
        warmup_steps=3,
        reward_type=args.reward_type,
        timeout=120.0,
        verbose=False,
    )

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Build policies to evaluate
    policies = {}
    vec_env = None  # Will hold VecNormalize if loaded

    if args.policy == "ppo" or args.model_path:
        if args.model_path and os.path.exists(args.model_path):
            if PPO is None:
                print("Error: stable-baselines3 not installed for PPO evaluation")
                sys.exit(1)
            print(f"\nLoading PPO model from: {args.model_path}")

            # Check for VecNormalize stats
            vecnorm_path = args.vecnormalize
            if vecnorm_path is None:
                # Try to find it in same directory as model
                model_dir = os.path.dirname(args.model_path)
                candidate = os.path.join(model_dir, "vecnormalize.pkl")
                if os.path.exists(candidate):
                    vecnorm_path = candidate

            # Wrap environment for VecNormalize if stats exist
            dummy_env = DummyVecEnv([lambda: env])

            if vecnorm_path and os.path.exists(vecnorm_path):
                print(f"Loading VecNormalize stats from: {vecnorm_path}")
                vec_env = VecNormalize.load(vecnorm_path, dummy_env)
                vec_env.training = False  # Don't update stats during eval
                vec_env.norm_reward = False  # Don't normalize rewards during eval
            else:
                print("Warning: No VecNormalize stats found. Using raw observations.")
                vec_env = dummy_env

            if "lstm" in os.path.basename(args.model_path).lower():
                from sb3_contrib import RecurrentPPO

                ppo_model = RecurrentPPO.load(args.model_path, env=vec_env)
            else:
                ppo_model = PPO.load(args.model_path, env=vec_env)
            policies["PPO"] = ppo_model
        elif args.policy == "ppo":
            print("Error: --model-path required for PPO evaluation")
            sys.exit(1)

    if args.policy == "random":
        policies["Random"] = RandomPolicy(env.action_space)

    # Add random baseline if --compare-random is set
    if args.compare_random and "Random" not in policies:
        policies["Random"] = RandomPolicy(env.action_space)

    # Add heuristic policies
    if args.compare_heuristic:
        for heuristic_name in args.compare_heuristic:
            if heuristic_name in HEURISTIC_POLICIES:
                policy_class = HEURISTIC_POLICIES[heuristic_name]
                display_name = f"Heuristic_{heuristic_name}"
                policies[display_name] = policy_class(env.action_space)

    # Add analytical policies if requested
    if args.compare_analytical:
        if not ANALYTICAL_AVAILABLE:
            print("Warning: Analytical policies not available. Skipping.")
        else:
            for analytical_name in args.compare_analytical:
                if analytical_name in ANALYTICAL_POLICIES:
                    policy_class = ANALYTICAL_POLICIES[analytical_name]
                    display_name = f"Analytical_{analytical_name}"
                    policies[display_name] = policy_class(env.action_space)
                    print(f"Added analytical policy: {display_name}")

    if not policies:
        print("Error: No policy specified")
        sys.exit(1)

    # Run evaluation
    try:
        all_results = compare_policies(
            policies=policies,
            env=env,
            n_episodes=args.n_episodes,
            seeds=seeds,
            verbose=args.verbose,
            vec_normalize=vec_env if isinstance(vec_env, VecNormalize) else None,
        )

        # Add metadata for identification
        all_results["metadata"] = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "reward_type": args.reward_type,
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "model_path": args.model_path,
            "policies_evaluated": list(policies.keys()),
        }

        # Print comparison table
        print_comparison_table(all_results)

        # Save results
        if args.output:
            save_results(all_results, args.output)
        else:
            # Auto-generate output path with identifier for easy recognition
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = run_paths.artifact_dir("eval_results", create=True)

            # Use "random" as identifier if only random policy, otherwise use reward_type
            if args.policy == "random" and not args.model_path:
                file_id = "random"
            else:
                file_id = args.reward_type

            # Include prefix if provided (e.g., 'lstm_ppo_V1_twt_throughput_timestamp.json')
            if args.output_prefix:
                output_path = os.path.join(
                    output_dir, f"eval_{args.output_prefix}{file_id}_{timestamp}.json"
                )
            else:
                output_path = os.path.join(
                    output_dir, f"eval_{file_id}_{timestamp}.json"
                )
            save_results(all_results, output_path)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")

    finally:
        env.close()

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

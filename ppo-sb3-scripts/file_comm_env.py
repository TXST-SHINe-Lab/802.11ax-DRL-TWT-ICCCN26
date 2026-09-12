#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""file_comm_env.py - file-based communication Gym environment.

Talks to NS-3 through JSON files orchestrated by a shell script, rather than over shared memory directly.
Each episode spawns a fresh NS-3 process, so no shared-memory segment outlives the episode that created it.

Architecture:
    Python (Gym API) <---> JSON Files <---> Shell Script <---> NS-3

Files used (in comm_dir):
    - response.json: Episode runner writes observations
    - action.json:   Python writes actions
    - done.flag:     Signals episode completion

Imported by the training scripts; running it directly drives one episode of random actions as a smoke test.

Usage:
    python3.11 file_comm_env.py
"""

import os
import sys
import json
import time
import subprocess
import signal
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, List
import shutil

# Add parent for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", "exploration-scripts"))

from reward_functions import get_reward_function, RewardLogger, compute_reward

# --- Data cleaning and delta computation ---

# Sentinels NS-3 emits for values it never measured.
INVALID_SENTINEL = 65535  # 0xFFFF, an uninitialised counter
INVALID_RSSI = -200  # dBm, below any real measurement


def clean_value(value: float, default: float = 0.0) -> float:
    """Clean a single value: handle NaN, Inf, and sentinel values."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if value == INVALID_SENTINEL or value == 65535:
            return default
        if np.isnan(value) or np.isinf(value):
            return default
        return float(value)
    return default


def clean_positive(value: float, max_val: float = 1e9) -> float:
    """Clean a value that should be non-negative, with optional upper bound."""
    value = clean_value(value, default=0.0)
    return float(np.clip(value, 0.0, max_val))


def clean_rssi(value: float) -> float:
    """Clean RSSI value: handle invalid values and clamp to reasonable range."""
    value = clean_value(value, default=-100.0)
    if value < INVALID_RSSI or value < -100:
        return -100.0
    return np.clip(value, -100.0, 0.0)


def compute_sta_deltas(prev_env: Dict, curr_env: Dict) -> List[Dict[str, float]]:
    """
    Compute per-STA delta metrics from previous and current observations.

    Converts cumulative NS-3 counters to per-step deltas for reward computation.
    Covers all metrics, both the ones PPO observes and the hidden oracle ones.

    Args:
        prev_env: Previous environment dict from wrapper
        curr_env: Current environment dict from wrapper

    Returns:
        List of dicts (one per STA) with delta'd and instantaneous metrics for reward
    """
    curr_sta_obs = curr_env.get("sta_observations", [])
    prev_sta_obs = prev_env.get("sta_observations", []) if prev_env else []

    sta_deltas = []

    for i, curr_sta in enumerate(curr_sta_obs):
        prev_sta = prev_sta_obs[i] if i < len(prev_sta_obs) else {}

        curr_oracle = curr_sta.get("oracle", {})
        prev_oracle = prev_sta.get("oracle", {}) if prev_sta else {}
        curr_realistic = curr_sta.get("realistic", {})
        prev_realistic = prev_sta.get("realistic", {}) if prev_sta else {}

        # --- PPO-visible metrics, also present in the observation ---
        # Realistic raw
        curr_airtime = clean_positive(curr_realistic.get("airtime_used_us", 0))
        prev_airtime = clean_positive(prev_realistic.get("airtime_used_us", 0))

        # Oracle raw, TWT-related
        curr_awake = clean_positive(curr_oracle.get("awake_time_ms", 0))
        prev_awake = clean_positive(prev_oracle.get("awake_time_ms", 0))

        curr_sleep = clean_positive(curr_oracle.get("sleep_time_ms", 0))
        prev_sleep = clean_positive(prev_oracle.get("sleep_time_ms", 0))

        curr_packets_tx = clean_positive(curr_oracle.get("packets_transmitted", 0))
        prev_packets_tx = clean_positive(prev_oracle.get("packets_transmitted", 0))

        # --- Hidden oracle metrics, used by the reward only, never observed by PPO ---
        curr_bytes_tx = clean_positive(curr_oracle.get("bytes_transmitted", 0))
        prev_bytes_tx = clean_positive(prev_oracle.get("bytes_transmitted", 0))

        curr_energy = clean_positive(curr_oracle.get("total_energy_consumed_mj", 0))
        prev_energy = clean_positive(prev_oracle.get("total_energy_consumed_mj", 0))

        curr_drops_exp = clean_positive(curr_oracle.get("mpdu_drops_expired", 0))
        prev_drops_exp = clean_positive(prev_oracle.get("mpdu_drops_expired", 0))

        curr_enqueued = clean_positive(curr_oracle.get("packets_enqueued", 0))
        prev_enqueued = clean_positive(prev_oracle.get("packets_enqueued", 0))

        # Every metric the reward function may read, in one dict.
        # Deltas are clamped at zero because a counter reset would otherwise read as a large negative step.
        sta_delta = {
            # --- PPO-visible deltas ---
            "delta_airtime_used_us": max(0.0, curr_airtime - prev_airtime),
            "delta_awake_time_ms": max(0.0, curr_awake - prev_awake),
            "delta_sleep_time_ms": max(0.0, curr_sleep - prev_sleep),
            "delta_packets_transmitted": max(0.0, curr_packets_tx - prev_packets_tx),
            # --- PPO-visible instantaneous ---
            # bsr_queue_index is a BSR index in 0 through 254, not bytes; it correlates with queue_size_bytes.
            "bsr_queue_index": clean_positive(
                curr_realistic.get("bsr_queue_ac_be", 0), 255
            ),
            "fcs_error_count": clean_positive(curr_realistic.get("fcs_error_count", 0)),
            "rx_fragment_count": clean_positive(
                curr_realistic.get("rx_fragment_count", 0)
            ),
            "last_rx_timestamp_us": clean_positive(
                curr_realistic.get("last_rx_timestamp_us", 0)
            ),
            "duty_cycle": float(
                np.clip(clean_value(curr_oracle.get("duty_cycle", 0)), 0, 1.0)
            ),
            # --- Hidden oracle deltas, reward only ---
            "delta_bytes_transmitted": max(0.0, curr_bytes_tx - prev_bytes_tx),
            "delta_energy_mj": max(0.0, curr_energy - prev_energy),
            "delta_drops_expired": max(0.0, curr_drops_exp - prev_drops_exp),
            "delta_packets_enqueued": max(0.0, curr_enqueued - prev_enqueued),
            # --- Hidden oracle instantaneous, reward only ---
            "queue_size_bytes": clean_positive(
                curr_oracle.get("queue_size_bytes", 0), 1e8
            ),
            "queue_size_packets": clean_positive(
                curr_oracle.get("queue_size_packets", 0)
            ),
        }

        sta_deltas.append(sta_delta)

    return sta_deltas


def build_ppo_observation(
    prev_env: Dict, curr_env: Dict, num_sta: int = 16
) -> np.ndarray:
    """
    Build the full PPO observation vector with raw features + deltas.

    PPO Observation (13 features per STA = 208 dims total):

    REALISTIC RAW (5):
      1. bsr_queue_ac_be
      2. airtime_used_us
      3. fcs_error_count
      4. rx_fragment_count
      5. last_rx_timestamp_us

    ORACLE RAW (4 - TWT related):
      6. awake_time_ms
      7. sleep_time_ms
      8. duty_cycle
      9. packets_transmitted

    DELTAS (4 - computed from cumulative):
      10. delta_airtime_used_us
      11. delta_awake_time_ms
      12. delta_sleep_time_ms
      13. delta_packets_transmitted

    Args:
        prev_env: Previous environment dict (for delta computation)
        curr_env: Current environment dict
        num_sta: Number of STAs (default 16)

    Returns:
        np.ndarray of shape (num_sta * 13,) = (208,)
    """
    curr_sta_obs = curr_env.get("sta_observations", [])
    prev_sta_obs = prev_env.get("sta_observations", []) if prev_env else []

    obs = []

    for i in range(num_sta):
        if i < len(curr_sta_obs):
            curr_sta = curr_sta_obs[i]
            prev_sta = prev_sta_obs[i] if i < len(prev_sta_obs) else {}

            curr_realistic = curr_sta.get("realistic", {})
            curr_oracle = curr_sta.get("oracle", {})
            prev_realistic = prev_sta.get("realistic", {}) if prev_sta else {}
            prev_oracle = prev_sta.get("oracle", {}) if prev_sta else {}

            # --- Realistic raw, features 1 through 5 ---
            obs.append(clean_positive(curr_realistic.get("bsr_queue_ac_be", 0)))
            obs.append(clean_positive(curr_realistic.get("airtime_used_us", 0)))
            obs.append(clean_positive(curr_realistic.get("fcs_error_count", 0)))
            obs.append(clean_positive(curr_realistic.get("rx_fragment_count", 0)))
            obs.append(clean_positive(curr_realistic.get("last_rx_timestamp_us", 0)))

            # --- Oracle raw, TWT-related, features 6 through 9 ---
            curr_awake = clean_positive(curr_oracle.get("awake_time_ms", 0))
            curr_sleep = clean_positive(curr_oracle.get("sleep_time_ms", 0))
            curr_duty = float(
                np.clip(clean_value(curr_oracle.get("duty_cycle", 0)), 0, 1.0)
            )
            curr_packets_tx = clean_positive(curr_oracle.get("packets_transmitted", 0))

            obs.append(curr_awake)
            obs.append(curr_sleep)
            obs.append(curr_duty)
            obs.append(curr_packets_tx)

            # --- Deltas against the previous step, features 10 through 13 ---
            prev_airtime = clean_positive(prev_realistic.get("airtime_used_us", 0))
            prev_awake = clean_positive(prev_oracle.get("awake_time_ms", 0))
            prev_sleep = clean_positive(prev_oracle.get("sleep_time_ms", 0))
            prev_packets_tx = clean_positive(prev_oracle.get("packets_transmitted", 0))

            curr_airtime = clean_positive(curr_realistic.get("airtime_used_us", 0))

            obs.append(max(0.0, curr_airtime - prev_airtime))  # delta_airtime_used_us
            obs.append(max(0.0, curr_awake - prev_awake))  # delta_awake_time_ms
            obs.append(max(0.0, curr_sleep - prev_sleep))  # delta_sleep_time_ms
            obs.append(
                max(0.0, curr_packets_tx - prev_packets_tx)
            )  # delta_packets_transmitted
        else:
            # Fewer STAs than MAX_STA, so pad to keep the observation a fixed 208 dims.
            obs.extend([0.0] * 13)

    return np.array(obs, dtype=np.float32)


class FileCommEnv(gym.Env):
    """
    Gymnasium environment that communicates with NS-3 via files.

    Each episode:
    1. Python calls reset() -> spawns episode runner
    2. Episode runner starts NS-3, writes initial obs to response.json
    3. Python reads obs, computes action, writes to action.json
    4. Episode runner reads action, steps NS-3, writes next obs
    5. Repeat until done
    6. NS-3 exits, episode runner exits, memory cleaned

    PPO Observation (13 features per STA):
    - 5 realistic raw + 4 oracle raw (TWT-related) + 4 deltas

    The reward reads every metric, including the hidden oracle ones the observation omits.
    """

    metadata = {"render_modes": []}

    # 5 realistic raw + 4 oracle raw + 4 deltas, so the observation is MAX_STA * 13 = 208 dims.
    PER_STA_FEATURES = 13
    MAX_STA = 16

    def __init__(
        self,
        comm_dir: str = "/tmp/twt_comm",
        runner_script: str = None,
        seed: int = 1000,
        warmup_steps: int = 2,
        reward_type: str = "balanced",
        timeout: float = 120.0,
        verbose: bool = False,
    ):
        super().__init__()

        self.comm_dir = comm_dir
        self.runner_script = runner_script or os.path.join(
            _script_dir, "episode_file_runner.py"
        )
        self.base_seed = seed
        self.current_seed = seed
        self.warmup_steps = warmup_steps
        self.reward_type = reward_type
        self.timeout = timeout
        self.verbose = verbose

        # File paths
        self.response_file = os.path.join(comm_dir, "response.json")
        self.action_file = os.path.join(comm_dir, "action.json")
        self.done_flag = os.path.join(comm_dir, "done.flag")

        # Action tables
        self._load_action_tables()

        # Action space: [schedule_idx, assignment_idx]
        self.num_schedules = len(self.schedule_table["schedules"])
        self.num_assignments = len(self.assignment_table["assignments"])
        self.action_space = spaces.MultiDiscrete(
            [self.num_schedules, self.num_assignments]
        )

        # Observation space: flattened per-STA features
        self.observation_dim = self.MAX_STA * self.PER_STA_FEATURES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32
        )

        # State
        self.episode_count = 0
        self.step_count = 0
        self.current_process = None
        self.prev_env = None
        self.reward_fn = get_reward_function(reward_type)

        # Reward logging for detailed analysis
        self.reward_logger = RewardLogger(preset=reward_type)
        self.last_reward_breakdown = None  # Surfaced in the step() info dict

        # Cumulative tracking for logging
        self.cumulative_reward = 0.0
        self.cumulative_bytes_tx = 0
        self.cumulative_energy_mj = 0.0
        self.cumulative_drops = 0
        self.avg_queue_bytes = (
            0.0  # Bytes, stands in for latency, which NS-3 does not report per step
        )

        # Ensure comm dir exists and is clean
        os.makedirs(comm_dir, exist_ok=True)
        self._clean_comm_dir()

        self._log(
            f"FileCommEnv initialized: {self.num_schedules}x{self.num_assignments} actions"
        )

    def _log(self, msg: str):
        if self.verbose:
            print(f"[FileCommEnv] {msg}")

    def _load_action_tables(self):
        """Load the schedule and assignment tables written by generate_action_tables.py.

        Raises if either file is missing; there is deliberately no fallback, since a
        default table would silently change the action space.
        """
        exploration_dir = os.path.join(_script_dir, "..", "exploration-scripts")

        with open(os.path.join(exploration_dir, "table_schedule.json")) as f:
            self.schedule_table = json.load(f)
        with open(os.path.join(exploration_dir, "table_assignment.json")) as f:
            self.assignment_table = json.load(f)

    def _clean_comm_dir(self):
        """Remove all communication files."""
        for f in [self.response_file, self.action_file, self.done_flag]:
            if os.path.exists(f):
                os.remove(f)

    def _write_json(self, filepath: str, data: Dict):
        """Write JSON atomically."""
        tmp = filepath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.rename(tmp, filepath)

    def _read_json_wait(self, filepath: str, timeout: float = None) -> Optional[Dict]:
        """Poll for a JSON file, read it, and delete it. Returns None on timeout.

        A partially written file raises and is retried, so the writer does not need to
        write atomically for reads to be safe.
        """
        timeout = timeout or self.timeout
        start = time.time()

        while time.time() - start < timeout:
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                    os.remove(
                        filepath
                    )  # Consume it so the next poll blocks on a fresh write
                    return data
                except (json.JSONDecodeError, IOError):
                    pass  # Half-written, retry on the next tick
            time.sleep(0.02)

        return None

    def _kill_process(self):
        """Kill the episode runner if running and release its shared memory.

        Without the /dev/shm sweep the next reset() fails with
        boost::interprocess_exception::library_error.
        """
        if self.current_process is not None:
            try:
                os.killpg(os.getpgid(self.current_process.pid), signal.SIGTERM)
                self.current_process.wait(timeout=5)
            except:
                try:
                    os.killpg(os.getpgid(self.current_process.pid), signal.SIGKILL)
                except:
                    pass
            self.current_process = None

        # SIGKILL leaves the segments behind, so sweep them unconditionally.
        try:
            subprocess.run(["bash", "-c", "rm -f /dev/shm/My*"], timeout=2)
        except:
            pass

    def _spawn_episode(self) -> bool:
        """Spawn a new episode via Python runner script."""
        self._kill_process()
        self._clean_comm_dir()

        # reset() already advanced base_seed, so one seed maps to one simulation.
        self.current_seed = self.base_seed
        self.episode_count += 1

        cmd = [
            "python3.11",
            self.runner_script,
            "--seed",
            str(self.current_seed),
            "--warmup-steps",
            str(self.warmup_steps),
            "--comm-dir",
            self.comm_dir,
        ]

        if self.verbose:
            cmd.append("--verbose")

        self._log(
            f"Spawning episode {self.episode_count} with seed {self.current_seed}"
        )

        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=None if self.verbose else subprocess.DEVNULL,
                stderr=None if self.verbose else subprocess.DEVNULL,
                preexec_fn=os.setsid,  # New process group for cleanup
            )

            # Wait for initial observation
            response = self._read_json_wait(self.response_file, timeout=self.timeout)

            if response is None:
                self._log("No initial response from episode runner")
                self._kill_process()
                return False

            if response.get("type") == "error":
                self._log(f"Episode error: {response.get('error')}")
                self._kill_process()
                return False

            if response.get("type") != "obs":
                self._log(f"Unexpected response type: {response.get('type')}")
                self._kill_process()
                return False

            # Store initial observation
            self._last_obs = np.array(response.get("obs", []), dtype=np.float32)
            self._last_env = response.get("env", {})
            self.prev_env = self._last_env
            self.step_count = 0

            self._log(f"Episode started, got {self._last_env.get('num_sta', '?')} STAs")
            return True

        except Exception as e:
            self._log(f"Failed to spawn episode: {e}")
            self._kill_process()
            return False

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Spawn a fresh episode and return its first observation.

        An explicit seed pins the simulation; omitting one advances base_seed by 1 so
        successive episodes differ. Spawning is retried three times before giving up,
        after which a zero observation and {"error": "spawn_failed"} are returned rather
        than raising.
        """
        if seed is not None:
            self.base_seed = seed
        else:
            self.base_seed += 1
        # Reset cumulative trackers
        self.cumulative_reward = 0.0
        self.cumulative_bytes_tx = 0
        self.cumulative_energy_mj = 0.0
        self.cumulative_drops = 0
        self.avg_queue_bytes = 0.0

        # Try to spawn, with retries
        for attempt in range(3):
            if self._spawn_episode():
                # No prev_env on reset, so every delta feature starts at 0.
                obs = build_ppo_observation(None, self._last_env, self.MAX_STA)
                return obs, {
                    "seed": self.current_seed,
                    "num_sta": self._last_env.get("num_sta", 16),
                }

            self._log(f"Spawn attempt {attempt + 1} failed, retrying...")
            time.sleep(1.0)

        # All attempts failed
        self._log("All spawn attempts failed!")
        return np.zeros(self.observation_dim, dtype=np.float32), {
            "error": "spawn_failed"
        }

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Apply one [schedule_idx, assignment_idx] action and return the Gym step tuple.

        Also accumulates the episode totals and writes a RewardLogger row.
        A response timeout ends the episode with reward 0 and info["error"] == "timeout"
        rather than raising, so a stalled NS-3 does not abort training.
        """
        self.step_count += 1

        # Convert action to list
        if hasattr(action, "tolist"):
            action = action.tolist()

        schedule_idx = int(action[0])
        assignment_idx = int(action[1])

        # Clip rather than reject; MultiDiscrete can emit an index the table no longer has.
        schedule_idx = max(0, min(schedule_idx, self.num_schedules - 1))
        assignment_idx = max(0, min(assignment_idx, self.num_assignments - 1))

        # Write action
        self._write_json(
            self.action_file,
            {
                "schedule_idx": schedule_idx,
                "assignment_idx": assignment_idx,
            },
        )

        # Wait for response
        response = self._read_json_wait(self.response_file, timeout=60.0)

        if response is None:
            # Timeout - episode failed, but still return accumulated metrics
            self._log(f"Timeout waiting for step {self.step_count} response")
            self._kill_process()
            return (
                np.zeros(self.observation_dim, dtype=np.float32),
                0.0,
                True,
                False,
                {
                    "error": "timeout",
                    "step": self.step_count,
                    "reason": "timeout",
                    "cumulative_reward": self.cumulative_reward,
                    "total_steps": self.step_count,
                    "total_bytes_tx": self.cumulative_bytes_tx,
                    "total_energy_mj": self.cumulative_energy_mj,
                    "total_drops": self.cumulative_drops,
                    "avg_queue_bytes": self.avg_queue_bytes,
                },
            )

        done = response.get("done", False)

        if done or response.get("type") == "done":
            # Episode ended normally
            self._log(f"Episode done after {self.step_count} steps")
            self._kill_process()
            # Return cumulative metrics in info for logging
            return (
                np.zeros(self.observation_dim, dtype=np.float32),
                0.0,
                True,
                False,
                {
                    "step": self.step_count,
                    "reason": "episode_done",
                    "cumulative_reward": self.cumulative_reward,
                    "total_steps": self.step_count,
                    "total_bytes_tx": self.cumulative_bytes_tx,
                    "total_energy_mj": self.cumulative_energy_mj,
                    "total_drops": self.cumulative_drops,
                    "avg_queue_bytes": self.avg_queue_bytes,
                },
            )

        # Extract observation and compute reward
        current_env = response.get("env", {})

        obs = build_ppo_observation(self.prev_env, current_env, self.MAX_STA)

        # The reward sees the hidden oracle metrics too; the action goes along for the airtime penalty and the log row.
        reward = self._compute_reward(
            self.prev_env, current_env, action=(schedule_idx, assignment_idx)
        )

        # Track cumulative metrics
        self.cumulative_reward += reward
        self._update_cumulative_metrics(current_env)

        # Update state
        self.prev_env = current_env

        # Build info dict with reward breakdown if available
        info = {"step": self.step_count}
        if self.last_reward_breakdown is not None:
            info["reward_breakdown"] = self.last_reward_breakdown

        return obs, reward, False, False, info

    def _update_cumulative_metrics(self, env: Dict):
        """Refresh the episode totals reported in the terminal step's info dict."""
        sta_obs_list = env.get("sta_observations", [])

        total_bytes = 0
        total_energy = 0.0
        total_drops = 0
        total_queue_bytes = 0

        for sta_obs in sta_obs_list:
            oracle = sta_obs.get("oracle", {})
            total_bytes += oracle.get("bytes_transmitted", 0)
            total_energy += oracle.get("total_energy_consumed_mj", 0)
            total_drops += oracle.get("mpdu_drops_expired", 0)
            total_queue_bytes += oracle.get("queue_size_bytes", 0)
            # mpdu_drops_queue_full is not summed here; it is always zero in the NS-3 data.

        # Assigned, not added to: NS-3 already reports these as running totals.
        self.cumulative_bytes_tx = total_bytes
        self.cumulative_energy_mj = total_energy
        self.cumulative_drops = total_drops
        # Mean queue bytes across STAs, the latency proxy.
        self.avg_queue_bytes = total_queue_bytes / max(len(sta_obs_list), 1)

    def _compute_reward(
        self, prev_env: Dict, current_env: Dict, action: Optional[tuple] = None
    ) -> float:
        """Compute the reward for one environment transition from delta-based metrics.

        Also writes the breakdown to reward_logger and caches it in last_reward_breakdown.
        Returns 0.0 on a missing transition or any reward-function error, so a bad step
        costs nothing rather than killing the run.
        """
        if not prev_env or not current_env:
            self.last_reward_breakdown = None
            return 0.0

        try:
            # Compute per-STA deltas from prev and current env
            sta_deltas = compute_sta_deltas(prev_env, current_env)

            if not sta_deltas:
                self.last_reward_breakdown = None
                return 0.0

            # Call reward function with sta_deltas and action for airtime penalty
            result = self.reward_fn(sta_deltas, action=action)
            reward = result.get("total", 0.0)

            # Store breakdown for info dict
            self.last_reward_breakdown = result

            # Log to RewardLogger for detailed CSV output
            self.reward_logger.log_step(
                reward_result=result,
                episode=self.episode_count,
                step=self.step_count,
                action=action,
            )

            return float(reward)
        except Exception as e:
            self._log(f"Reward computation error: {e}")
            self.last_reward_breakdown = None
            return 0.0

    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Zero-pad or truncate obs to observation_dim, which must match the space."""
        if len(obs) == self.observation_dim:
            return obs

        padded = np.zeros(self.observation_dim, dtype=np.float32)
        min_len = min(len(obs), self.observation_dim)
        padded[:min_len] = obs[:min_len]
        return padded

    def close(self):
        """Kill the runner, release shared memory, and remove the comm files."""
        self._kill_process()
        self._clean_comm_dir()
        self._log("Environment closed")

    def get_reward_logger(self) -> RewardLogger:
        """Expose the reward logger so callers can save or clear it directly."""
        return self.reward_logger

    def save_reward_log(self, filepath: str):
        """Write the accumulated per-step reward breakdown to a CSV."""
        self.reward_logger.save(filepath)
        self._log(f"Reward log saved to {filepath}")

    def clear_reward_log(self):
        """Drop the accumulated rows, for example between training runs."""
        self.reward_logger.clear()

    def render(self):
        pass


def make_file_comm_env(
    seed: int = 1000,
    warmup_steps: int = 2,
    reward_type: str = "balanced",
    verbose: bool = False,
    **kwargs,
) -> FileCommEnv:
    """Construct a FileCommEnv; kwargs pass through to the constructor."""
    return FileCommEnv(
        seed=seed,
        warmup_steps=warmup_steps,
        reward_type=reward_type,
        verbose=verbose,
        **kwargs,
    )


if __name__ == "__main__":
    # Smoke test: one episode of up to 40 random actions. Run with python3.11 file_comm_env.py.
    env = FileCommEnv(seed=5000, verbose=True)

    obs, info = env.reset()
    print(f"Reset: obs_sum={obs.sum():.1f}, info={info}")

    for i in range(40):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, done={done}")
        if done:
            break

    env.close()
    print("Test complete!")

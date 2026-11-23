#!/usr/bin/env python3
"""hw_env.py - Constraint-aware TWT environment (EE 7331 proposal, Section 2 and Table 1).

HwConstrainedTWTEnv subclasses the ICCCN'26 FileCommEnv and changes exactly four things:

  1. Observation: the AP-observable 13-feature layout from ap_observation.py replaces the
     simulator-ground-truth features, then a fixed z-score from a random-policy run.
  2. Actions: action_masks() hands sb3-contrib MaskablePPO the legal-action mask (schedules
     whose windows exceed 65.28 ms are removed). An illegal action reaching step() raises by
     default, so a masking bug cannot hide; illegal_action="clip" is only for running the
     published ICCCN'26 policy as a baseline, and counts every clip.
  3. Episode: truncation after `num_epochs` decision epochs (20 in the proposal), counted here,
     independent of how long ns-3 was built to run.
  4. Runner: hw_episode_runner.py, which also clips the hardcoded warmup schedule (schedule 0,
     90 ms, illegal on the bench) so no illegal schedule ever reaches ns-3.

The reward is untouched: FileCommEnv._compute_reward still reads the oracle metrics, which is
the privileged reward of the proposal (training only). The policy never sees them.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PPO_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "ppo-sb3-scripts"))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _PPO_DIR)

from file_comm_env import FileCommEnv  # noqa: E402

import hw_constraints as hwc  # noqa: E402
from ap_observation import APObservationBuilder, FixedZScore, PER_STA_FEATURES, MAX_STA  # noqa: E402

DEFAULT_RUNNER = os.path.join(_THIS_DIR, "hw_episode_runner.py")


class IllegalActionError(RuntimeError):
    """Raised when a masked policy emits an action the bench cannot execute."""


class HwConstrainedTWTEnv(FileCommEnv):
    """FileCommEnv with AP-observable inputs, a legal-action mask and 20-epoch truncation."""

    def __init__(
        self,
        comm_dir: str = "/tmp/twt_comm_hw",
        seed: int = 1000,
        reward_type: str = "energy",
        num_epochs: int = 20,
        warmup_steps: int = 2,
        obs_stats_path: Optional[str] = None,
        wake_interval_ms: float = hwc.BEACON_INTERVAL_MS,
        epoch_s: float = 25.0,
        step_timeout: float = 600.0,
        spawn_timeout: float = 600.0,
        runner_action_timeout: float = 86400.0,
        illegal_action: str = "raise",
        runner_script: Optional[str] = None,
        python_executable: Optional[str] = None,
        max_sp_ms: float = hwc.MAX_SP_DURATION_MS,
        verbose: bool = False,
    ):
        if illegal_action not in ("raise", "clip"):
            raise ValueError("illegal_action must be 'raise' or 'clip'")
        super().__init__(
            comm_dir=comm_dir,
            runner_script=runner_script or DEFAULT_RUNNER,
            seed=seed,
            warmup_steps=warmup_steps,
            reward_type=reward_type,
            timeout=spawn_timeout,
            verbose=verbose,
        )
        self.num_epochs = int(num_epochs)
        self.step_timeout = float(step_timeout)
        # How long the runner waits for the next action. Large on purpose: during periodic
        # evaluation the training episode sits paused mid-epoch until evaluation finishes.
        self.runner_action_timeout = float(runner_action_timeout)
        self.illegal_action = illegal_action
        self.python_executable = python_executable or sys.executable
        self.max_sp_ms = float(max_sp_ms)

        self._mask = hwc.multidiscrete_action_mask(self.max_sp_ms)
        self._legal = set(hwc.legal_schedules(self.max_sp_ms))
        # The runner warms up on (0, 0); hw_episode_runner clips it the same way.
        self.warmup_action = hwc.clip_action((0, 0), self.max_sp_ms)

        self.builder = APObservationBuilder(
            wake_interval_ms=wake_interval_ms, fallback_epoch_ms=epoch_s * 1000.0, max_sta=MAX_STA
        )
        self.normalizer = FixedZScore.load(obs_stats_path) if obs_stats_path else FixedZScore.identity()
        self.obs_stats_path = obs_stats_path

        self.num_sta = 0
        self.epoch = 0
        self.clipped_actions = 0  # Only ever non-zero with illegal_action="clip"
        self.last_raw_obs: Optional[np.ndarray] = None
        self._episode_metrics: Dict[str, float] = {}

    # --- MaskablePPO hook -----------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Flat [20 schedule | 19 assignment] mask; constant, since the limit is static."""
        return self._mask.copy()

    # --- Gym API --------------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.base_seed = seed
        else:
            self.base_seed += 1
        self.cumulative_reward = 0.0
        self.cumulative_bytes_tx = 0
        self.cumulative_energy_mj = 0.0
        self.cumulative_drops = 0
        self.avg_queue_bytes = 0.0
        self.epoch = 0
        self._episode_metrics = {}

        for attempt in range(3):
            if self._spawn_episode():
                raw = self.builder.reset(self._last_env, self.warmup_action)
                self.num_sta = self.builder.num_sta
                self._start_metrics = self._oracle_totals(self._last_env)
                return self._finish_obs(raw), {"seed": self.current_seed, "num_sta": self.num_sta}
            self._log(f"Spawn attempt {attempt + 1} failed, retrying...")
            time.sleep(1.0)

        raise RuntimeError(
            f"ns-3 episode failed to start three times (seed {self.base_seed}). "
            "Run the runner by hand with --verbose to see the ns-3 error."
        )

    def step(self, action: Sequence[int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = [int(a) for a in (action.tolist() if hasattr(action, "tolist") else action)]
        s_idx = int(np.clip(action[0], 0, self.num_schedules - 1))
        a_idx = int(np.clip(action[1], 0, self.num_assignments - 1))

        if s_idx not in self._legal:
            if self.illegal_action == "raise":
                raise IllegalActionError(
                    f"Schedule {s_idx} {hwc.schedule_durations(s_idx)} ms exceeds the "
                    f"{self.max_sp_ms} ms SP limit; the action mask should have removed it."
                )
            s_idx = hwc.nearest_legal_schedule(s_idx, self.max_sp_ms)
            self.clipped_actions += 1

        self.step_count += 1
        self.epoch += 1
        self._write_json(self.action_file, {"schedule_idx": s_idx, "assignment_idx": a_idx})
        response = self._read_json_wait(self.response_file, timeout=self.step_timeout)

        if response is None or response.get("type") == "error":
            err = "timeout" if response is None else response.get("error", "error")
            self._kill_process()
            raise RuntimeError(
                f"ns-3 step {self.epoch} failed ({err}). With long epochs, raise --step-timeout."
            )

        if response.get("done", False) or response.get("type") == "done":
            # ns-3 ran out before num_epochs: it was built with too few update cycles.
            self._kill_process()
            raise RuntimeError(
                f"ns-3 ended after {self.epoch} epochs but num_epochs={self.num_epochs}. "
                "Rebuild with DURATION_IN_UPDATE >= num_epochs + warmup_steps + 1 "
                "(ns3-patch/apply_hw_patch.py does this)."
            )

        current_env = response.get("env", {})
        raw = self.builder.update(self.prev_env, current_env, (s_idx, a_idx))
        reward = self._compute_reward(self.prev_env, current_env, action=(s_idx, a_idx))

        self.cumulative_reward += reward
        self._update_cumulative_metrics(current_env)
        self.prev_env = current_env

        truncated = self.epoch >= self.num_epochs
        info = {
            "step": self.step_count,
            "epoch": self.epoch,
            "executed_action": (s_idx, a_idx),
            "clipped_actions": self.clipped_actions,
        }
        if self.last_reward_breakdown is not None:
            info["reward_breakdown"] = self.last_reward_breakdown
        if truncated:
            info["episode_metrics"] = self._episode_summary(current_env)
            self._kill_process()

        return self._finish_obs(raw), float(reward), False, truncated, info

    # --- helpers --------------------------------------------------------------------------

    def _finish_obs(self, raw: np.ndarray) -> np.ndarray:
        self.last_raw_obs = raw
        return self.normalizer(raw, self.num_sta)

    @staticmethod
    def _oracle_totals(env: Dict) -> Dict[str, float]:
        """Ground-truth totals for EVALUATION LOGGING ONLY. Never part of the observation."""
        t = {"energy_mj": 0.0, "bytes_tx": 0.0, "packets_tx": 0.0, "drops": 0.0, "packets_rx_ap": 0.0}
        for sta in (env or {}).get("sta_observations", []):
            o = sta.get("oracle", {}) or {}
            r = sta.get("realistic", {}) or {}
            t["energy_mj"] += float(o.get("total_energy_consumed_mj", 0) or 0)
            t["bytes_tx"] += float(o.get("bytes_transmitted", 0) or 0)
            t["packets_tx"] += float(o.get("packets_transmitted", 0) or 0)
            t["drops"] += float(o.get("mpdu_drops_expired", 0) or 0)
            t["packets_rx_ap"] += float(r.get("packets_received_at_ap", 0) or 0)
        return t

    def _episode_summary(self, env: Dict) -> Dict[str, float]:
        """Per-episode totals over the agent's epochs (warmup excluded)."""
        end = self._oracle_totals(env)
        out = {k: max(0.0, end[k] - self._start_metrics.get(k, 0.0)) for k in end}
        delivered = out["packets_rx_ap"]
        out["energy_per_delivered_packet_mj"] = out["energy_mj"] / delivered if delivered > 0 else float("nan")
        out["return"] = self.cumulative_reward
        out["epochs"] = self.epoch
        out["num_sta"] = self.num_sta
        out["seed"] = self.current_seed
        out["clipped_actions"] = self.clipped_actions
        return out

    def _kill_process(self):
        """Stop the runner and remove ONLY this episode's shared-memory segments.

        FileCommEnv._kill_process runs `rm -f /dev/shm/My*`, which also deletes the segments of
        any other run on the same machine and crashes it. Segment names carry the seed
        (pb_twt_wrapper_py.TWTWrapper.reset), so they can be removed one by one instead,
        which is what makes parallel seeds safe. On macOS there is no /dev/shm and the
        removal is a harmless no-op.
        """
        if self.current_process is not None:
            try:
                os.killpg(os.getpgid(self.current_process.pid), 15)
                self.current_process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(os.getpgid(self.current_process.pid), 9)
                except Exception:  # noqa: BLE001
                    pass
            self.current_process = None
        seed = getattr(self, "current_seed", None)
        if seed is not None:
            for name in (f"MySeg_{seed}", f"MyCpp2PyMsg_{seed}", f"MyPy2CppMsg_{seed}", f"MyLockable_{seed}"):
                path = os.path.join("/dev/shm", name)
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass

    def _spawn_episode(self) -> bool:
        """FileCommEnv._spawn_episode with a configurable interpreter (it hardcodes python3.11)."""
        self._kill_process()
        self._clean_comm_dir()
        self.current_seed = self.base_seed
        self.episode_count += 1

        cmd = [
            self.python_executable,
            self.runner_script,
            "--seed", str(self.current_seed),
            "--warmup-steps", str(self.warmup_steps),
            "--comm-dir", self.comm_dir,
            "--action-timeout", str(int(self.runner_action_timeout)),
        ]
        if self.verbose:
            cmd.append("--verbose")

        try:
            self.current_process = subprocess.Popen(
                cmd,
                stdout=None if self.verbose else subprocess.DEVNULL,
                stderr=None if self.verbose else subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
            response = self._read_json_wait(self.response_file, timeout=self.timeout)
            if response is None or response.get("type") != "obs":
                self._log(f"Bad initial response: {response}")
                self._kill_process()
                return False
            self._last_env = response.get("env", {})
            self.prev_env = self._last_env
            self.step_count = 0
            return True
        except Exception as e:  # noqa: BLE001
            self._log(f"Failed to spawn episode: {e}")
            self._kill_process()
            return False


def make_hw_env(**kwargs) -> HwConstrainedTWTEnv:
    return HwConstrainedTWTEnv(**kwargs)

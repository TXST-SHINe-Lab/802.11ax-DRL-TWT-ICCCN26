#!/usr/bin/env python3
"""policies.py - One act() interface for every policy that gets evaluated.

Each policy is driven one episode at a time by evaluation.run_episodes():
    policy.reset()                       at the start of an episode
    action = policy.act(obs, env)        every epoch; obs is the env's observation

Implemented here:
    MaskedMlpAgent        trained MaskablePPO (MLP-PPO-HW)
    MaskedLstmAgent       trained RecurrentPPO with the masked policy (LSTM-PPO-HW)
    FixedSchedulePolicy   one fixed [schedule, assignment], e.g. the best fixed legal schedule
    RandomLegalPolicy     uniform over the legal actions (used to fit observation statistics)
    MD1Policy             the M/D/1 analytical scheduler of [5] (energy preset), legal actions only
    PublishedPolicy       the published ICCCN'26 MLP-PPO, unchanged, fed its own (simulator)
                          observation; run with illegal_action="clip" (proposal, Section 3.1)
    load_run_agent()      the right trained agent for a train_hw.py run directory

The trained agents normalize observations with the VecNormalize statistics saved at training
time, frozen, exactly as SB3 does inside a VecNormalize(training=False) wrapper.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, Sequence

import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Saved MLP models reference EnhancedFeatureExtractor from ppo-sb3-scripts/train_ppo_V1.py.
sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..", "ppo-sb3-scripts")))
sys.path.insert(0, _THIS_DIR)


class _VecNormObs:
    """Apply saved VecNormalize observation statistics without a VecEnv."""

    def __init__(self, path: Optional[str]):
        self.vn = None
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                self.vn = pickle.load(f)
            self.vn.training = False

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.vn is None or not self.vn.norm_obs:
            return obs
        return self.vn.normalize_obs(obs[None, :])[0].astype(np.float32)


class MaskedMlpAgent:
    name = "MLP-PPO-HW"

    def __init__(self, model_path: str, vecnorm_path: Optional[str] = None, deterministic: bool = True):
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path, device="cpu")
        self.norm = _VecNormObs(vecnorm_path)
        self.deterministic = deterministic

    def reset(self):
        pass

    def act(self, obs, env):
        action, _ = self.model.predict(self.norm(obs), action_masks=env.action_masks(), deterministic=self.deterministic)
        return [int(action[0]), int(action[1])]


class MaskedLstmAgent:
    name = "LSTM-PPO-HW"

    def __init__(self, model_path: str, vecnorm_path: Optional[str] = None, deterministic: bool = True):
        from sb3_contrib import RecurrentPPO
        import masked_lstm_policy  # noqa: F401  (registers the class for unpickling)
        self.model = RecurrentPPO.load(model_path, device="cpu")
        self.norm = _VecNormObs(vecnorm_path)
        self.deterministic = deterministic
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def reset(self):
        self.state = None
        self.episode_start = np.ones((1,), dtype=bool)

    def act(self, obs, env):
        action, self.state = self.model.predict(
            self.norm(obs)[None, :], state=self.state, episode_start=self.episode_start, deterministic=self.deterministic
        )
        self.episode_start = np.zeros((1,), dtype=bool)
        return [int(action[0][0]), int(action[0][1])]


class FixedSchedulePolicy:
    def __init__(self, schedule_idx: int, assignment_idx: int):
        self.action = [int(schedule_idx), int(assignment_idx)]
        self.name = f"fixed_s{schedule_idx}_a{assignment_idx}"

    def reset(self):
        pass

    def act(self, obs, env):
        return list(self.action)


class RandomLegalPolicy:
    name = "random_legal"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def act(self, obs, env):
        m = env.action_masks()
        n_s = env.num_schedules
        s = int(self.rng.choice(np.flatnonzero(m[:n_s])))
        a = int(self.rng.choice(np.flatnonzero(m[n_s:])))
        return [s, a]


# --- Baselines of proposal Section 3.1 ----------------------------------------------------


class MD1Policy:
    """The M/D/1 analytical scheduler of [5] (analytical_policies.AnalyticalEnergyPolicy).

    Same model, weights and decision rule as the published baseline: it scores every joint
    action by predicted M/D/1 backlog (from each STA's BSR queue index) and energy, prefers
    actions whose utilisation stays below 0.85, and takes the minimum. The only change is the
    action set: it chooses among the bench-legal actions, so the min-max scaling is over those.
    Like in [5] it reads the raw (un-normalised) observation; the only feature it uses is the
    BSR queue index, which the AP observes.

    Its queueing constants come from derived_constants.json (TWT_DERIVED_CONSTANTS), so they
    are the ones dialed for the bench setting.
    """

    name = "M/D/1"

    def __init__(self):
        self.base = None
        self.legal_idx = None

    def reset(self):
        pass

    def _build(self, env):
        from analytical_policies import AnalyticalEnergyPolicy

        self.base = AnalyticalEnergyPolicy(None, env.num_sta)
        legal = set(int(s) for s in np.flatnonzero(env.action_masks()[: env.num_schedules]))
        self.legal_idx = np.array([i for i, (s, _) in enumerate(self.base.pairs) if s in legal])

    def act(self, obs, env):
        if self.base is None or self.base.num_sta != env.num_sta:
            self._build(env)
        b, idx = self.base, self.legal_idx
        from analytical_policies import QueuingModel

        queue = b._queue_packets(env.last_raw_obs)
        backlog = np.maximum(0.0, queue[None, :] + QueuingModel.MEAN_ARRIVAL_RATE - b.mu[idx]).sum(axis=1)
        score = b.W_QUEUE * b._scale(backlog) + b.W_ENERGY * b._scale(b.energy[idx])
        stable = b.stable[idx]
        if stable.any():
            score = np.where(stable, score, np.inf)
        s, a = b.pairs[int(idx[int(np.argmin(score))])]
        return [int(s), int(a)]


class PublishedPolicy:
    """The published ICCCN'26 MLP-PPO (energy preset), used as given, not retrained.

    It sees exactly the observation it was trained on: file_comm_env.build_ppo_observation()
    of the raw ns-3 dicts (16 STA slots, including the simulator-only awake/sleep/duty/packet
    features), normalised with its own vecnormalize.pkl. Its actions are not masked; run it
    in an env with illegal_action="clip" so an illegal schedule is replaced by the nearest
    legal one (the env counts these in episode_metrics["clipped_actions"]).
    """

    name = "Published MLP-PPO"

    def __init__(self, model_path: str, vecnorm_path: Optional[str] = None, deterministic: bool = True):
        from stable_baselines3 import PPO
        import train_ppo_V1

        # Models saved by running train_ppo_V1.py as a script may refer to __main__.EnhancedFeatureExtractor.
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and not hasattr(main_mod, "EnhancedFeatureExtractor"):
            setattr(main_mod, "EnhancedFeatureExtractor", train_ppo_V1.EnhancedFeatureExtractor)
        self.model = PPO.load(model_path, device="cpu")
        if vecnorm_path is None:
            cand = os.path.join(os.path.dirname(os.path.abspath(model_path)), "vecnormalize.pkl")
            vecnorm_path = cand if os.path.exists(cand) else None
        if vecnorm_path is None:
            print("WARNING: published policy loaded without vecnormalize.pkl; its inputs will be unnormalised")
        self.norm = _VecNormObs(vecnorm_path)
        self.deterministic = deterministic
        self._prev_env = None

    def reset(self):
        self._prev_env = None

    def observation(self, env) -> np.ndarray:
        """Ahmed's observation, built the way FileCommEnv builds it (None as prev on reset)."""
        from file_comm_env import FileCommEnv, build_ppo_observation

        curr = env.prev_env  # HwConstrainedTWTEnv keeps the latest ns-3 dict here
        obs = build_ppo_observation(self._prev_env, curr, FileCommEnv.MAX_STA)
        self._prev_env = curr
        return obs

    def act(self, obs, env):
        action, _ = self.model.predict(self.norm(self.observation(env)), deterministic=self.deterministic)
        return [int(action[0]), int(action[1])]


def load_run_agent(run_dir: str, deterministic: bool = True):
    """Trained agent of a train_hw.py run directory, named '<ALGO>-PPO-HW seed <n>'."""
    import json

    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    algo = cfg["args"]["algo"]
    model = os.path.join(run_dir, "model_final.zip")
    vecnorm = os.path.join(run_dir, "vecnormalize.pkl")
    agent = (MaskedMlpAgent if algo == "mlp" else MaskedLstmAgent)(model, vecnorm, deterministic)
    agent.name = f"{agent.name} seed {cfg['args']['seed']}"
    agent.algo, agent.run_seed, agent.config = algo, cfg["args"]["seed"], cfg
    return agent

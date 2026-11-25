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

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

import hw_constraints as hwc
from masked_lstm_policy import MaskedRecurrentActorCriticPolicy

MASK = hwc.multidiscrete_action_mask()
ILLEGAL = set(hwc.illegal_schedules())


class ToyEnv(gym.Env):
    """208-dim noise observations, MultiDiscrete([20, 19]); reward favours the illegal schedules,
    so an unmasked policy would drift towards them."""

    observation_space = spaces.Box(-np.inf, np.inf, (208,), np.float32)
    action_space = spaces.MultiDiscrete([20, 19])

    def __init__(self):
        self.t = 0
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None, options=None):
        self.t = 0
        return self.rng.normal(size=208).astype(np.float32), {}

    def step(self, a):
        self.t += 1
        r = 1.0 if int(a[0]) in ILLEGAL else 0.0
        return self.rng.normal(size=208).astype(np.float32), r, False, self.t >= 20, {}

    def action_masks(self):
        return MASK.copy()


def test_recurrent_policy_never_samples_illegal_and_survives_training(tmp_path):
    from sb3_contrib import RecurrentPPO
    model = RecurrentPPO(MaskedRecurrentActorCriticPolicy, ToyEnv(), n_steps=128, batch_size=64, n_epochs=2,
                         policy_kwargs=dict(lstm_hidden_size=32, net_arch=dict(pi=[32], vf=[32]),
                                            action_mask=MASK.tolist()), seed=0, device="cpu")
    model.learn(512)

    obs = np.random.default_rng(1).normal(size=(4000, 208)).astype(np.float32)
    acts, _ = model.predict(obs, state=None, episode_start=np.ones(4000, dtype=bool), deterministic=False)
    assert not (set(acts[:, 0].tolist()) & ILLEGAL)
    assert len(set(acts[:, 0].tolist())) > 10  # still explores the legal schedules

    # Illegal schedules carry ~zero probability, entropy stays finite.
    with th.no_grad():
        o = th.as_tensor(obs[:8])
        n_layers, n_env, hid = model.policy.lstm_hidden_state_shape
        state = (th.zeros(n_layers, 8, hid), th.zeros(n_layers, 8, hid))
        dist, _ = model.policy.get_distribution(o, state, th.ones(8))
        probs = th.softmax(dist.distribution[0].logits, dim=-1)
        assert float(probs[:, list(ILLEGAL)].max()) == 0.0
        assert th.isfinite(dist.entropy()).all()

    path = tmp_path / "m.zip"
    model.save(path)
    loaded = RecurrentPPO.load(path, device="cpu")
    assert loaded.policy._action_mask_list == MASK.tolist()
    acts2, _ = loaded.predict(obs[:500], state=None, episode_start=np.ones(500, dtype=bool), deterministic=False)
    assert not (set(acts2[:, 0].tolist()) & ILLEGAL)


def test_maskable_ppo_never_samples_illegal():
    from sb3_contrib import MaskablePPO
    model = MaskablePPO("MlpPolicy", ToyEnv(), n_steps=128, batch_size=64, n_epochs=2,
                        policy_kwargs=dict(net_arch=[32]), seed=0, device="cpu")
    model.learn(512)
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(2000):
        a, _ = model.predict(rng.normal(size=208).astype(np.float32), action_masks=MASK, deterministic=False)
        seen.add(int(a[0]))
    assert not (seen & ILLEGAL)


def test_policy_rejects_bad_masks():
    from sb3_contrib import RecurrentPPO
    with pytest.raises(ValueError):
        RecurrentPPO(MaskedRecurrentActorCriticPolicy, ToyEnv(), policy_kwargs=dict(action_mask=[True] * 10), device="cpu")
    bad = MASK.copy()
    bad[:20] = False
    with pytest.raises(ValueError):
        RecurrentPPO(MaskedRecurrentActorCriticPolicy, ToyEnv(), policy_kwargs=dict(action_mask=bad.tolist()), device="cpu")

"""Train both agents briefly on the mock runner, reload them, evaluate on held-out seeds.

Checks plumbing only (masking under training, saving, VecNormalize reload, recurrent state
handling in evaluation). The mock runner is not ns-3; its returns mean nothing.
"""

import json
import os

import pytest

import seeds
from conftest import MOCK_RUNNER
from evaluation import run_episodes
from hw_env import HwConstrainedTWTEnv
from policies import MaskedLstmAgent, MaskedMlpAgent
import train_hw


@pytest.mark.parametrize("algo,agent_cls", [("mlp", MaskedMlpAgent), ("lstm", MaskedLstmAgent)])
def test_train_reload_evaluate(tmp_path, algo, agent_cls):
    run_dir = train_hw.main([
        "--algo", algo, "--seed", "1", "--mock", "--total-timesteps", "256",
        "--eval-every", "128", "--eval-episodes", "2", "--checkpoint-every", "100000",
        "--out-dir", str(tmp_path / "runs"), "--comm-root", str(tmp_path / "comm"),
    ])
    for f in ("config.json", "model_final.zip", "vecnormalize.pkl", "train_episodes.jsonl", "eval_curve.jsonl"):
        assert os.path.exists(os.path.join(run_dir, f)), f

    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    assert cfg["illegal_schedules"] == [0, 1] and cfg["legal_joint_actions"] == 342
    assert cfg["runner"].startswith("mock")

    train_rows = [json.loads(l) for l in open(os.path.join(run_dir, "train_episodes.jsonl"))]
    base = seeds.train_env_base(1)
    assert [r["seed"] for r in train_rows[:3]] == [base + 1, base + 2, base + 3]

    curve = [json.loads(l) for l in open(os.path.join(run_dir, "eval_curve.jsonl"))]
    assert curve[-1]["timesteps"] >= 256  # final-model evaluation is always logged
    assert all(c["seeds"] == seeds.eval_curve_seeds(2) for c in curve)

    agent = agent_cls(os.path.join(run_dir, "model_final.zip"), os.path.join(run_dir, "vecnormalize.pkl"))
    env = HwConstrainedTWTEnv(comm_dir=str(tmp_path / "comm_eval"), runner_script=MOCK_RUNNER,
                              num_epochs=20, step_timeout=30, spawn_timeout=30)
    try:
        rows = run_episodes(agent, env, seeds.test_seeds(2))
    finally:
        env.close()
    assert len(rows) == 2
    for r in rows:
        assert r["epochs"] == 20 and r["clipped_actions"] == 0
        assert all(a[0] not in (0, 1) for a in r["actions"])

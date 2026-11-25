#!/usr/bin/env python3
"""train_hw.py - Train MLP-PPO-HW (MaskablePPO) or LSTM-PPO-HW (RecurrentPPO + mask).

EE 7331 proposal, Sections 3 and 4. One invocation = one run (one agent, one seed, one
learning rate). Everything a run produces lands in its own directory:

    <out-dir>/<algo>_lr<lr>_seed<seed>/
        config.json            every argument, hyperparameter, seed range and library version
        train_episodes.jsonl   one row per training episode (return, energy, drops, ...)
        eval_curve.jsonl       every --eval-every steps: 10 deterministic episodes, fixed seeds
        model_final.zip        final policy
        vecnormalize.pkl       running observation/reward statistics needed to load it
        checkpoints/           intermediate models, every --checkpoint-every steps

Defaults reproduce the published ICCCN'26 settings (hyperparams.json of the energy runs):
lr 3e-4, minibatch 64, 10 epochs, clip 0.2, GAE 0.95, gamma 0.99, entropy 0.01, vf 0.5,
grad clip 0.5, rollouts of 256 (MLP) / 128 (LSTM), 2x256 heads, 15k steps, VecNormalize on
observations and rewards (clip 10).

Examples:
    # final run, real ns-3 (after ns3-patch/apply_hw_patch.py and a rebuild)
    python3 train_hw.py --algo mlp --seed 1 --obs-stats obs_stats.json
    # learning-rate sweep run (sweep seeds 101-103 only)
    python3 train_hw.py --algo lstm --seed 101 --purpose sweep --learning-rate 1e-4 \\
        --total-timesteps 3000 --obs-stats obs_stats.json
    # smoke test on the stand-in runner (NOT ns-3; numbers are meaningless)
    python3 train_hw.py --algo lstm --seed 1 --mock --total-timesteps 400 --eval-every 200
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

import torch as th  # noqa: E402
import torch.nn as nn  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import hw_constraints as hwc  # noqa: E402
import seeds as seedplan  # noqa: E402
from evaluation import run_episodes  # noqa: E402
from hw_env import HwConstrainedTWTEnv  # noqa: E402
from masked_lstm_policy import MaskedRecurrentActorCriticPolicy  # noqa: E402

MOCK_RUNNER = os.path.join(_THIS_DIR, "testing", "mock_ns3_runner.py")

ALGO_DEFAULTS = {"mlp": {"n_steps": 256}, "lstm": {"n_steps": 128}}


# --- Environment construction -------------------------------------------------------------


def make_env(args, base_seed: int, comm_dir: str) -> HwConstrainedTWTEnv:
    return HwConstrainedTWTEnv(
        comm_dir=comm_dir,
        seed=base_seed,
        reward_type=args.reward_preset,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        obs_stats_path=args.obs_stats,
        wake_interval_ms=args.wake_interval_ms,
        epoch_s=args.epoch_s,
        step_timeout=args.step_timeout,
        spawn_timeout=args.step_timeout,
        illegal_action="raise",
        runner_script=MOCK_RUNNER if args.mock else None,
        python_executable=args.python,
        verbose=args.verbose,
    )


class SeedPlanWrapper(Monitor):
    """Monitor that ignores the seed SB3 pushes into the first reset().

    SB3 seeds a VecEnv with the run seed (1, 2, ...), which would make ns-3 episodes start at
    seed 1 for run 1, seed 2 for run 2 and so on, so neighbouring runs would share most of their
    traffic. Dropping it lets HwConstrainedTWTEnv walk its own range from seeds.py instead
    (base+1, base+2, ...). Model/torch seeding is unaffected: that comes from PPO(seed=...).
    """

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        return super().reset(**kwargs)


# --- Callbacks ----------------------------------------------------------------------------


class EpisodeLogCallback(BaseCallback):
    """Write one JSON row per finished training episode (training returns include exploration)."""

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.n = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            m = info.get("episode_metrics")
            if m is not None:
                self.n += 1
                row = {"episode": self.n, "timesteps": int(self.num_timesteps), **m}
                with open(self.path, "a") as f:
                    f.write(json.dumps(row) + "\n")
        return True


class _FrozenAgent:
    """Wraps the model being trained so evaluation.run_episodes() can drive it."""

    def __init__(self, model, algo, vec_env: VecNormalize):
        self.model, self.algo, self.vec_env = model, algo, vec_env
        self.name = f"{algo}-in-training"
        self.state, self.start = None, np.ones((1,), dtype=bool)

    def reset(self):
        self.state, self.start = None, np.ones((1,), dtype=bool)

    def _norm(self, obs):
        return self.vec_env.normalize_obs(obs[None, :]).astype(np.float32) if self.vec_env.norm_obs else obs[None, :]

    def act(self, obs, env):
        o = self._norm(obs)
        if self.algo == "mlp":
            a, _ = self.model.predict(o, action_masks=env.action_masks()[None, :], deterministic=True)
        else:
            a, self.state = self.model.predict(o, state=self.state, episode_start=self.start, deterministic=True)
            self.start = np.zeros((1,), dtype=bool)
        return [int(a[0][0]), int(a[0][1])]


class PeriodicEvalCallback(BaseCallback):
    """Every `every` steps: deterministic episodes on fixed, held-out eval-curve seeds."""

    def __init__(self, eval_env, vec_env, algo, every, seeds, path):
        super().__init__()
        self.eval_env, self.vec_env, self.algo = eval_env, vec_env, algo
        self.every, self.seeds, self.path = every, list(seeds), path
        self.next_at = every

    def _evaluate(self):
        t0 = time.time()
        rows = run_episodes(_FrozenAgent(self.model, self.algo, self.vec_env), self.eval_env, self.seeds)
        rets = [r["return"] for r in rows]
        out = {
            "timesteps": int(self.num_timesteps),
            "mean_return": float(np.mean(rets)),
            "returns": rets,
            "seeds": self.seeds,
            "energy_mj": [r["energy_mj"] for r in rows],
            "packets_rx_ap": [r["packets_rx_ap"] for r in rows],
            "drops": [r["drops"] for r in rows],
            "energy_per_delivered_packet_mj": [r["energy_per_delivered_packet_mj"] for r in rows],
            "wall_s": round(time.time() - t0, 1),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(out) + "\n")
        if self.verbose:
            print(f"[eval] t={self.num_timesteps} mean_return={out['mean_return']:.3f}")

    # Evaluate at rollout boundaries, AFTER the PPO update: SB3 calls on_rollout_start after
    # train(), whereas on_step / on_rollout_end fire before it, so an evaluation there would
    # score the pre-update weights. The logged `timesteps` is the exact count trained on.
    def _on_rollout_start(self) -> None:
        if self.num_timesteps >= self.next_at:
            self._evaluate()
            while self.next_at <= self.num_timesteps:
                self.next_at += self.every

    def _on_training_end(self) -> None:
        self._evaluate()  # Always score the final model

    def _on_step(self) -> bool:
        return True


# --- Model construction -------------------------------------------------------------------


def build_model(args, vec_env, mask):
    common = dict(
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device="cpu",
        verbose=1 if args.verbose else 0,
    )
    if args.algo == "mlp":
        from sb3_contrib import MaskablePPO
        sys.path.insert(0, os.path.normpath(os.path.join(_THIS_DIR, "..", "ppo-sb3-scripts")))
        from train_ppo_V1 import EnhancedFeatureExtractor  # the ICCCN'26 extractor, unchanged
        policy_kwargs = dict(
            features_extractor_class=EnhancedFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.GELU,
            ortho_init=True,
        )
        return MaskablePPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, **common)

    from sb3_contrib import RecurrentPPO
    policy_kwargs = dict(
        lstm_hidden_size=256,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=True,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        action_mask=mask.tolist(),
    )
    return RecurrentPPO(MaskedRecurrentActorCriticPolicy, vec_env, policy_kwargs=policy_kwargs, **common)


# --- Provenance ---------------------------------------------------------------------------


def versions():
    import gymnasium, sb3_contrib, stable_baselines3
    repo = os.path.normpath(os.path.join(_THIS_DIR, ".."))
    try:
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:  # noqa: BLE001
        commit, dirty = "", None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": th.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "repo_commit": commit,
        "repo_dirty": dirty,
    }


# --- Main ---------------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["mlp", "lstm"], required=True)
    p.add_argument("--seed", type=int, required=True, help="Run seed (1-10 final, 101-103 sweep)")
    p.add_argument("--purpose", choices=["train", "sweep"], default="train")
    p.add_argument("--total-timesteps", type=int, default=15_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=None, help="Default 256 (mlp) / 128 (lstm)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--no-vecnorm", action="store_true", help="Disable VecNormalize (default: on, as in [5])")
    p.add_argument("--reward-preset", default="energy")
    p.add_argument("--num-epochs", type=int, default=20, help="Decision epochs per episode")
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--epoch-s", type=float, default=25.0, help="Epoch length ns-3 was built with (recorded; fallback only)")
    p.add_argument("--wake-interval-ms", type=float, default=hwc.BEACON_INTERVAL_MS)
    p.add_argument("--obs-stats", default=None, help="obs_stats.json from collect_obs_stats.py")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=seedplan.N_EVAL_CURVE)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--step-timeout", type=float, default=900.0, help="Seconds to wait for one ns-3 epoch")
    p.add_argument("--out-dir", default=os.path.join(_THIS_DIR, "runs"))
    p.add_argument("--comm-root", default=None, help="Directory for the JSON file protocol (default under /tmp)")
    p.add_argument("--python", default=None, help="Interpreter for the episode runner (default: this one)")
    p.add_argument("--mock", action="store_true", help="Use the test-only stand-in runner instead of ns-3")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.n_steps is None:
        args.n_steps = ALGO_DEFAULTS[args.algo]["n_steps"]
    if args.n_steps % args.batch_size:
        p.error("--n-steps must be a multiple of --batch-size")
    if args.purpose == "sweep" and args.seed not in seedplan.SWEEP_RUN_SEEDS:
        p.error(f"sweep runs must use seeds {seedplan.SWEEP_RUN_SEEDS}")
    if args.purpose == "train" and args.seed in seedplan.SWEEP_RUN_SEEDS:
        p.error("sweep seeds are reserved for the sweep")
    if not args.mock and not args.obs_stats:
        print("WARNING: no --obs-stats given; observations are fed un-z-scored (VecNormalize still applies).")
    return args


def main(argv=None):
    args = parse_args(argv)
    th.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    run_name = f"{args.algo}_lr{args.learning_rate:g}_seed{args.seed}" + ("_sweep" if args.purpose == "sweep" else "")
    run_dir = os.path.join(args.out_dir, run_name)
    if os.path.exists(os.path.join(run_dir, "model_final.zip")):
        print(f"{run_dir} already has a final model; delete it to rerun.")
        return run_dir
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    comm_root = args.comm_root or os.path.join("/tmp", f"twt_hw_{os.getpid()}")

    base = seedplan.train_env_base(args.seed, args.purpose)
    eval_seeds = seedplan.eval_curve_seeds(args.eval_episodes)

    train_env = make_env(args, base, os.path.join(comm_root, f"{run_name}_train"))
    eval_env = make_env(args, base, os.path.join(comm_root, f"{run_name}_eval"))
    mask = train_env.action_masks()

    vec_env = DummyVecEnv([lambda: SeedPlanWrapper(train_env)])
    vec_env = VecNormalize(vec_env, norm_obs=not args.no_vecnorm, norm_reward=not args.no_vecnorm,
                           clip_obs=10.0, clip_reward=10.0, gamma=args.gamma)

    model = build_model(args, vec_env, mask)

    config = {
        "run_name": run_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "train_env_seed_base": base,
        "eval_curve_seeds": eval_seeds,
        "action_mask": mask.astype(int).tolist(),
        "illegal_schedules": list(hwc.illegal_schedules()),
        "legal_joint_actions": hwc.num_legal_joint_actions(),
        "max_sp_ms": hwc.MAX_SP_DURATION_MS,
        "runner": "mock (TEST ONLY)" if args.mock else "ns-3 via hw_episode_runner.py",
        "versions": versions(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    callbacks = CallbackList([
        EpisodeLogCallback(os.path.join(run_dir, "train_episodes.jsonl")),
        PeriodicEvalCallback(eval_env, vec_env, args.algo, args.eval_every, eval_seeds,
                             os.path.join(run_dir, "eval_curve.jsonl")),
        CheckpointCallback(save_freq=args.checkpoint_every, save_path=os.path.join(run_dir, "checkpoints"),
                           name_prefix=args.algo, save_vecnormalize=True),
    ])

    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=False)
    finally:
        model.save(os.path.join(run_dir, "model_last.zip"))
        vec_env.save(os.path.join(run_dir, "vecnormalize_last.pkl"))
        train_env.close()
        eval_env.close()

    os.replace(os.path.join(run_dir, "model_last.zip"), os.path.join(run_dir, "model_final.zip"))
    os.replace(os.path.join(run_dir, "vecnormalize_last.pkl"), os.path.join(run_dir, "vecnormalize.pkl"))
    config["wall_clock_s"] = round(time.time() - t0, 1)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Done: {run_dir} ({config['wall_clock_s']} s)")
    return run_dir


if __name__ == "__main__":
    main()

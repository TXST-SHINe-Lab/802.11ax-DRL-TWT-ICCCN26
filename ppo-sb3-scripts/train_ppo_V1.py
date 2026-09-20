#!/usr/bin/env python3
# Copyright (c) 2025 Texas State University
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Ahmed Maksud <ahmed.maksud@email.ucr.edu>
# PI: Marcelo Menezes De Carvalho <mmcarvalho@txstate.edu>

"""train_ppo_V1.py - PPO V1 training script for TWT WiFi scheduling.

Trains a feed-forward PPO agent with Stable-Baselines3 on the TWT Gymnasium environment.
See train_lstm_ppo_V1.py for the recurrent variant.

Features:
- MultiDiscrete action space support (native in SB3)
- Configurable hyperparameters
- TensorBoard logging
- Checkpoint saving
- Resumable training

Usage:
    python3.11 train_ppo_V1.py --total-timesteps 100000 --seed 42
    python3.11 train_ppo_V1.py --resume checkpoints/ppo_V1_twt_best.zip

Lab: SHINE Lab, Texas State University
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np

# Add current directory for local imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)

# Import PyTorch
import torch
import torch.nn as nn

# Import Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ImportError as e:
    print(f"Error: stable-baselines3 not installed. Run: pip install stable-baselines3")
    print(f"Import error: {e}")
    sys.exit(1)

# Each episode runs in its own process talking over JSON files, so no shared memory survives between episodes.
from file_comm_env import FileCommEnv, make_file_comm_env

# --- Custom feature extractor, moderate capacity ---


class EnhancedFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor with LayerNorm and GELU activation.
    Moderate capacity - balanced between speed and performance.
    """

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]

        # Three 256-wide blocks, each LayerNorm then GELU.
        self.net = nn.Sequential(
            nn.Linear(n_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        """Apply orthogonal initialization, gain sqrt(2), for training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# --- Default hyperparameters, moderate-capacity configuration ---
# Every one of these can be overridden from the command line; see the parser at the bottom.

DEFAULT_PPO_HYPERPARAMS = {
    # Core PPO params
    "learning_rate": 3e-4,
    "n_steps": 256,  # Steps per rollout, so one update every 256 env steps
    "batch_size": 64,  # Must divide n_steps
    "n_epochs": 10,  # Passes over each rollout
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    # Entropy and value function
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # Network architecture, moderate capacity
    "policy_kwargs": {
        "features_extractor_class": EnhancedFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),  # Policy head  # Value head
        "activation_fn": nn.GELU,
        "ortho_init": True,
    },
}


# --- Custom callbacks ---


class TBMetricsCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics: Dict[str, list] = {
            "total_bytes_tx": [],
            "total_energy_mj": [],
            "avg_queue_bytes": [],  # Latency proxy; avg_latency_ms was always zero in the NS-3 data
            "total_drops": [],
        }

    def _on_step(self) -> bool:
        """Collect episode metrics after each step; always returns True to keep training."""
        # Extract info from the environment
        for info in self.locals.get("infos", []):
            # Monitor wraps per-episode reward and length under the 'episode' key.
            if "episode" in info and isinstance(info["episode"], dict):
                ep_info = info["episode"]
                self.episode_rewards.append(ep_info.get("r", 0))
                self.episode_lengths.append(ep_info.get("l", 0))

            # Also check for our custom cumulative_reward
            if "cumulative_reward" in info:
                self.episode_rewards.append(info["cumulative_reward"])

            # Log custom metrics
            for metric_name in self.episode_metrics.keys():
                if metric_name in info:
                    self.episode_metrics[metric_name].append(info[metric_name])

        return True

    def _on_rollout_end(self) -> None:
        """Record each metric's mean over the last 100 episodes to TensorBoard."""
        if self.episode_rewards:
            self.logger.record(
                "rollout/ep_rew_mean", np.mean(self.episode_rewards[-100:])
            )

        for metric_name, values in self.episode_metrics.items():
            if values:
                self.logger.record(f"custom/{metric_name}_mean", np.mean(values[-100:]))


class EpisodeLoggerCallback(BaseCallback):
    """
    Callback that logs episode information to a JSON Lines file.
    """

    def __init__(
        self, log_dir: str, run_timestamp: str, log_freq: int = 1, verbose: int = 0
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.run_timestamp = run_timestamp
        self.log_freq = log_freq
        self.episode_count = 0
        self.log_file = None

    def _on_training_start(self) -> None:
        """Open training_log_<timestamp>.jsonl and write the metadata line."""
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = self.run_timestamp
        log_path = os.path.join(self.log_dir, f"training_log_{timestamp}.jsonl")
        self.log_file = open(log_path, "w")

        # First line of the file, distinguished by the _metadata key.
        metadata = {
            "_metadata": {
                "timestamp": timestamp,
                "total_timesteps": self.locals.get("total_timesteps", 0),
            }
        }
        self.log_file.write(json.dumps(metadata) + "\n")

    def _on_step(self) -> bool:
        """Append one JSON line per finished episode, every log_freq episodes."""
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.episode_count += 1

                if self.episode_count % self.log_freq == 0:
                    info = self.locals.get("infos", [{}])[idx]

                    # On auto-reset the SB3 VecEnv wrappers move the final step's info under 'terminal_info'.
                    terminal_info = info.get("terminal_info", info)

                    # Monitor's own episode stats, used unless the env reported its own total.
                    ep_reward = 0.0
                    ep_length = 0
                    if "episode" in info:
                        ep_reward = info["episode"].get("r", 0.0)
                        ep_length = info["episode"].get("l", 0)

                    # The env's cumulative_reward wins: it excludes the zero-reward terminal step.
                    if "cumulative_reward" in terminal_info:
                        ep_reward = terminal_info["cumulative_reward"]

                    log_entry = {
                        "episode": self.episode_count,
                        "timestep": self.num_timesteps,
                        "reward": float(ep_reward),
                        "steps": terminal_info.get("total_steps", ep_length),
                        "total_bytes_tx": terminal_info.get(
                            "cumulative_bytes_tx",
                            terminal_info.get("total_bytes_tx", 0),
                        ),
                        "total_energy_mj": terminal_info.get(
                            "cumulative_energy_mj",
                            terminal_info.get("total_energy_mj", 0),
                        ),
                        "avg_queue_bytes": terminal_info.get(
                            "avg_queue_bytes", 0
                        ),  # Latency proxy; avg_latency_ms was always 0
                        "total_drops": terminal_info.get(
                            "cumulative_drops", terminal_info.get("total_drops", 0)
                        ),
                    }

                    if self.log_file:
                        self.log_file.write(json.dumps(log_entry) + "\n")
                        self.log_file.flush()

        return True

    def _on_training_end(self) -> None:
        """Close the JSONL file."""
        if self.log_file:
            self.log_file.close()


class RewardLoggerCallback(BaseCallback):
    """
    Callback that saves detailed reward breakdown logs at specified intervals.

    The RewardLogger in the environment collects per-step reward data.
    This callback periodically saves that data to CSV files for analysis.
    """

    def __init__(
        self,
        log_dir: str,
        run_timestamp: str,
        save_freq: int = 10000,  # Timesteps between saves
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.run_timestamp = run_timestamp
        self.save_freq = save_freq
        self.last_save = 0
        self.save_count = 0

    def _on_training_start(self) -> None:
        """Create reward_logs/ and stamp the timestamp shared by every part file."""
        self.reward_log_dir = os.path.join(self.log_dir, "reward_logs")
        os.makedirs(self.reward_log_dir, exist_ok=True)
        self.training_start_timestamp = self.run_timestamp

    def _on_step(self) -> bool:
        """Flush the reward log once save_freq timesteps have passed since the last save."""
        if self.num_timesteps - self.last_save >= self.save_freq:
            self._save_reward_log()
            self.last_save = self.num_timesteps
        return True

    def _save_reward_log(self) -> None:
        """Write the buffered reward rows to a numbered part CSV, then clear the buffer.

        Failures are swallowed: losing a diagnostic log must not abort a training run.
        """
        try:
            env = self.training_env

            # Peel VecNormalize, then DummyVecEnv, then Monitor, down to the base FileCommEnv.
            while hasattr(env, "venv"):
                env = env.venv
            while hasattr(env, "envs"):
                env = env.envs[0]
            while hasattr(env, "env"):
                env = env.env

            # A different env class, or one built without a logger, simply has nothing to save.
            if hasattr(env, "save_reward_log") and hasattr(env, "reward_logger"):
                if len(env.reward_logger.data) > 0:
                    self.save_count += 1
                    filename = f"reward_log_{self.training_start_timestamp}_part{self.save_count:03d}.csv"
                    filepath = os.path.join(self.reward_log_dir, filename)
                    env.save_reward_log(filepath)

                    if self.verbose >= 1:
                        print(
                            f"[RewardLogger] Saved {len(env.reward_logger.data)} entries to {filename}"
                        )

                    # Clear after saving; the buffer grows unbounded otherwise.
                    env.clear_reward_log()
        except Exception as e:
            if self.verbose >= 1:
                print(f"[RewardLogger] Warning: Could not save reward log: {e}")

    def _on_training_end(self) -> None:
        """Flush whatever remains in the buffer, however few timesteps since the last save."""
        self._save_reward_log()
        if self.verbose >= 1:
            print(f"[RewardLogger] Training complete. Total saves: {self.save_count}")


class ProgressCallback(BaseCallback):
    """
    Simple callback to print training progress.
    """

    def __init__(self, print_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_count = 0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        """Print episode count and the mean of the last 10 rewards every print_freq steps."""
        for info in self.locals.get("infos", []):
            # Monitor's episode stats, when the Monitor wrapper is in the stack.
            if "episode" in info and isinstance(info["episode"], dict):
                self.episode_count += 1
                self.episode_rewards.append(info["episode"].get("r", 0))
            # Otherwise fall back to the env's own end-of-episode total.
            elif "cumulative_reward" in info and info.get("total_steps", 0) > 0:
                self.episode_count += 1
                self.episode_rewards.append(info["cumulative_reward"])

        if self.num_timesteps % self.print_freq == 0:
            mean_reward = (
                np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            )
            print(
                f"[Step {self.num_timesteps}] "
                f"Episodes: {self.episode_count} | "
                f"Mean Reward (last 10): {mean_reward:.4f}"
            )

        return True


# --- Training function ---


def train_ppo(
    # Training config
    total_timesteps: int = 100000,
    seed: int = 42,
    # Environment config
    reward_type: str = "per_sta",
    reward_preset: str = "balanced",
    reward_scale: float = 1.0,
    # PPO hyperparameters - Moderate Capacity defaults
    learning_rate: float = 3e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    # Network architecture - Moderate Capacity
    net_arch_pi: list = [256, 256],
    net_arch_vf: list = [256, 256],
    # Feature extractor
    use_custom_extractor: bool = True,  # False falls back to the SB3 default MLP extractor
    features_dim: int = 256,  # EnhancedFeatureExtractor output width
    # Output config
    output_dir: str = "checkpoints",
    tensorboard_log: str = "tb_logs",
    save_freq: int = 10000,
    # Resume training
    resume_path: Optional[str] = None,
    # Normalization
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    # Misc
    verbose: int = 1,
    device: str = "auto",
) -> PPO:
    """
    Train a PPO agent on the TWT environment.

    Args:
        total_timesteps: Total training timesteps
        seed: Random seed
        reward_type: Reward type for environment
        reward_preset: Reward preset name, one of throughput, energy, queue
        reward_scale: Reward scaling factor
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size, must divide n_steps
        n_epochs: PPO epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Gradient clipping
        net_arch_pi: Policy network architecture
        net_arch_vf: Value network architecture
        use_custom_extractor: Use EnhancedFeatureExtractor rather than the SB3 default
        features_dim: Feature extractor output width
        output_dir: Directory for checkpoints
        tensorboard_log: Directory for TensorBoard logs
        save_freq: Checkpoint save frequency in timesteps
        resume_path: Path to resume training from
        normalize_obs: Wrap the env in VecNormalize for observations
        normalize_reward: Wrap the env in VecNormalize for rewards
        clip_obs: VecNormalize observation clip, in standard deviations
        clip_reward: VecNormalize reward clip, in standard deviations
        verbose: Verbosity level
        device: Device for training (auto, cpu, cuda)

    Returns:
        Trained PPO model
    """
    # Absolute paths throughout: NS-3 changes the working directory out from under us.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ppo_V1_twt_{reward_preset}_{timestamp}"

    # Ensure paths are absolute
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_script_dir, output_dir)
    if not os.path.isabs(tensorboard_log):
        tensorboard_log = os.path.join(_script_dir, tensorboard_log)

    checkpoint_dir = os.path.join(output_dir, run_name)
    log_dir = os.path.join(tensorboard_log, run_name)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("PPO V1 Training for TWT WiFi Scheduling")
    print("=" * 60)
    print(f"Run name: {run_name}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Seed: {seed}")
    print(f"Reward type: {reward_type}")
    print(f"Reward preset: {reward_preset}")
    print(f"Normalize obs: {normalize_obs} (clip={clip_obs})")
    print(f"Normalize reward: {normalize_reward} (clip={clip_reward})")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"TensorBoard log: {log_dir}")
    print("=" * 60)

    # Create environment.
    # File-based comms with NS-3, one process per episode, so no shared memory carries over.
    print("\nCreating file-based environment (fresh process per episode)...")

    file_env = FileCommEnv(
        comm_dir="/tmp/twt_comm",
        seed=seed,
        warmup_steps=3,
        reward_type=reward_preset,
        timeout=120.0,
        verbose=False,
    )

    # Monitor supplies the 'episode' info dict, with 'r' for reward and 'l' for length.
    monitor_env = Monitor(file_env)

    # SB3 requires a VecEnv even for a single environment.
    vec_env = DummyVecEnv([lambda: monitor_env])

    # VecNormalize keeps running mean and variance; the callbacks unwrap it to reach FileCommEnv.
    if normalize_obs or normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=clip_obs,
            clip_reward=clip_reward,
            gamma=gamma,
        )
        print(f"Applied VecNormalize: obs={normalize_obs}, reward={normalize_reward}")

    print(f"Action space: {file_env.action_space}")
    print(f"Observation space: {file_env.observation_space}")

    # Build PPO hyperparameters, moderate-capacity configuration.
    policy_kwargs = {
        "net_arch": dict(pi=net_arch_pi, vf=net_arch_vf),
        "activation_fn": nn.GELU,  # Better gradient flow than ReLU
        "ortho_init": True,  # Orthogonal initialization for stability
    }

    if use_custom_extractor:
        policy_kwargs["features_extractor_class"] = EnhancedFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {"features_dim": features_dim}
        print(f"Using EnhancedFeatureExtractor with features_dim={features_dim}")

    print(f"Network architecture: pi={net_arch_pi}, vf={net_arch_vf}")
    print(f"Activation: GELU, Ortho init: True")

    # Create or load model
    if resume_path and os.path.exists(resume_path):
        print(f"\nResuming training from: {resume_path}")
        model = PPO.load(
            resume_path,
            env=vec_env,
            device=device,
            tensorboard_log=log_dir,
        )
    else:
        print("\nCreating new PPO V1 model...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            seed=seed,
            device=device,
            verbose=verbose,
        )

    # Recorded alongside the checkpoint so a run can be reproduced from its directory alone.
    hyperparams = {
        "timestamp": timestamp,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "reward_type": reward_type,
        "reward_preset": reward_preset,
        "reward_scale": reward_scale,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "net_arch_pi": net_arch_pi,
        "net_arch_vf": net_arch_vf,
        "use_custom_extractor": use_custom_extractor,
        "features_dim": features_dim,
        "activation_fn": "GELU",
        "ortho_init": True,
        "normalize_obs": normalize_obs,
        "normalize_reward": normalize_reward,
        "clip_obs": clip_obs,
        "clip_reward": clip_reward,
        "action_space": str(vec_env.action_space),
        "observation_space": str(vec_env.observation_space),
    }

    hyperparams_path = os.path.join(checkpoint_dir, "hyperparams.json")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=2)
    print(f"Saved hyperparameters to: {hyperparams_path}")

    # Create callbacks
    callbacks = [
        # Save checkpoints
        CheckpointCallback(
            save_freq=save_freq,
            save_path=checkpoint_dir,
            name_prefix="ppo_V1_twt",
            verbose=1,
        ),
        # Custom metrics logging
        TBMetricsCallback(verbose=0),
        # Episode logging
        EpisodeLoggerCallback(
            log_dir=checkpoint_dir,
            run_timestamp=timestamp,
            log_freq=1,
            verbose=0,
        ),
        # Detailed reward logging for analysis
        RewardLoggerCallback(
            log_dir=checkpoint_dir,
            run_timestamp=timestamp,
            save_freq=5000,  # Timesteps between part files
            verbose=1,
        ),
        # Progress printing
        ProgressCallback(print_freq=1000, verbose=1),
    ]

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Both handlers fall through to the save below, so an interrupted run still leaves a usable model.
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
            reset_num_timesteps=resume_path is None,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback

        traceback.print_exc()

    # Save final model
    final_path = os.path.join(checkpoint_dir, "ppo_V1_twt_final.zip")
    model.save(final_path)
    print(f"\nSaved final model to: {final_path}")

    # Inference must reuse these statistics; a model loaded without them sees unnormalised observations.
    if isinstance(vec_env, VecNormalize):
        vecnorm_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
        vec_env.save(vecnorm_path)
        print(f"Saved VecNormalize stats to: {vecnorm_path}")

        # The same numbers again as JSON, readable without unpickling.
        norm_stats = {
            "obs_mean": vec_env.obs_rms.mean.tolist(),
            "obs_var": vec_env.obs_rms.var.tolist(),
            "obs_count": int(vec_env.obs_rms.count),
            "ret_mean": float(vec_env.ret_rms.mean) if normalize_reward else None,
            "ret_var": float(vec_env.ret_rms.var) if normalize_reward else None,
            "clip_obs": clip_obs,
            "clip_reward": clip_reward,
        }
        norm_stats_path = os.path.join(checkpoint_dir, "normalization_stats.json")
        with open(norm_stats_path, "w") as f:
            json.dump(norm_stats, f, indent=2)
        print(f"Saved normalization stats to: {norm_stats_path}")

    # Cleanup
    vec_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Model saved to: {final_path}")
    print(f"TensorBoard logs: tensorboard --logdir {log_dir}")
    print(f"Checkpoints: {checkpoint_dir}")

    return model


# --- Main entry point ---


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent for TWT WiFi Scheduling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training config
    parser.add_argument(
        "--total-timesteps", type=int, default=100000, help="Total training timesteps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Environment config
    parser.add_argument(
        "--reward-type",
        type=str,
        default="per_sta",
        choices=["per_sta", "throughput", "energy", "multi"],
        help="Reward type: per_sta (new), throughput, energy, multi",
    )
    parser.add_argument(
        "--reward-preset",
        type=str,
        default="throughput",
        choices=[
            "throughput",
            "energy",
            "queue",
            "balanced",
            "latency",
            "throughput_focused",
            "energy_focused",
            "latency_focused",
            "fairness_focused",
        ],
        help="Reward preset: throughput/energy/queue or aliases like *_focused",
    )
    parser.add_argument(
        "--reward-scale", type=float, default=1.0, help="Reward scaling factor"
    )

    # PPO hyperparameters - Moderate Capacity defaults
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--n-steps", type=int, default=256, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="PPO epochs per update"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument(
        "--ent-coef", type=float, default=0.01, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="Gradient clipping"
    )

    # Network architecture - Moderate Capacity
    parser.add_argument(
        "--net-arch",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Network architecture (hidden layer sizes)",
    )
    parser.add_argument(
        "--no-custom-extractor",
        action="store_true",
        default=False,
        help="Disable custom EnhancedFeatureExtractor (use simple MLP)",
    )
    parser.add_argument(
        "--features-dim",
        type=int,
        default=256,
        help="Feature extractor output dimension",
    )

    # Output config
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint output directory",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="tb_logs",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--save-freq", type=int, default=10000, help="Checkpoint save frequency"
    )

    # Resume training
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to model to resume training from"
    )

    # Normalization
    parser.add_argument(
        "--normalize-obs",
        action="store_true",
        default=True,
        help="Enable observation normalization via VecNormalize",
    )
    parser.add_argument(
        "--no-normalize-obs",
        action="store_false",
        dest="normalize_obs",
        help="Disable observation normalization",
    )
    parser.add_argument(
        "--normalize-reward",
        action="store_true",
        default=True,
        help="Enable reward normalization via VecNormalize",
    )
    parser.add_argument(
        "--no-normalize-reward",
        action="store_false",
        dest="normalize_reward",
        help="Disable reward normalization",
    )
    parser.add_argument(
        "--clip-obs",
        type=float,
        default=10.0,
        help="Observation clipping range for normalization",
    )
    parser.add_argument(
        "--clip-reward",
        type=float,
        default=10.0,
        help="Reward clipping range for normalization",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for training",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")

    args = parser.parse_args()

    # net_arch feeds both heads, so pi and vf always share a shape here.
    train_ppo(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        reward_type=args.reward_type,
        reward_preset=args.reward_preset,
        reward_scale=args.reward_scale,
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
        net_arch_pi=args.net_arch,
        net_arch_vf=args.net_arch,
        use_custom_extractor=not args.no_custom_extractor,
        features_dim=args.features_dim,
        output_dir=args.output_dir,
        tensorboard_log=args.tensorboard_log,
        save_freq=args.save_freq,
        resume_path=args.resume,
        normalize_obs=args.normalize_obs,
        normalize_reward=args.normalize_reward,
        clip_obs=args.clip_obs,
        clip_reward=args.clip_reward,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

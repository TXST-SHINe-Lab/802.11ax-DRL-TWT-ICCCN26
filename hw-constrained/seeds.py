#!/usr/bin/env python3
"""seeds.py - One place that fixes every seed used in the study (proposal, Sections 3.3 and 4).

Five disjoint ns-3 seed ranges, so no episode is ever used for two purposes:

  obs_stats    random-policy episodes that fit the observation z-score
  sweep        learning-rate sweep runs            SWEEP_BASE + (run_seed - 100) * SPAN + episode
  train        final 10-seed training runs         TRAIN_BASE + run_seed * SPAN + episode
  fixed_search episodes on which all 342 legal fixed actions are simulated to pick the best
               fixed schedule (proposal, Section 3.1); training-distribution traffic, own range
  eval_curve   10 fixed episodes for the learning curves, identical for every run
  test         50 held-out episodes for the final comparison (H1), never seen in training

FileCommEnv.reset() advances its base seed by one per episode, so a run with base B uses
B+1, B+2, ... A 15k-step run has 750 episodes, far below SPAN, so ranges never overlap.
"""

SPAN = 10_000
SWEEP_BASE = 1_000_000
TRAIN_BASE = 2_000_000
FIXED_SEARCH_BASE = 6_000_000
OBS_STATS_BASE = 7_000_000
EVAL_CURVE_BASE = 8_000_000
TEST_BASE = 9_000_000

N_EVAL_CURVE = 10
N_TEST = 50

FINAL_RUN_SEEDS = list(range(1, 11))  # 10 runs per agent
SWEEP_RUN_SEEDS = [101, 102, 103]  # 3 seeds per learning rate, never reused below


def train_env_base(run_seed: int, purpose: str = "train") -> int:
    """First ns-3 seed (minus one) of a run. Ranges are 1M wide, so they cannot overlap."""
    if purpose == "train":
        if not 1 <= run_seed < 100:
            raise ValueError("final-run seeds must be in 1..99")
        return TRAIN_BASE + run_seed * SPAN
    if purpose == "sweep":
        if run_seed not in SWEEP_RUN_SEEDS:
            raise ValueError(f"sweep seeds must be one of {SWEEP_RUN_SEEDS}")
        return SWEEP_BASE + (run_seed - 100) * SPAN
    raise ValueError(purpose)


def eval_curve_seeds(n: int = N_EVAL_CURVE):
    return [EVAL_CURVE_BASE + i for i in range(n)]


def test_seeds(n: int = N_TEST):
    return [TEST_BASE + i for i in range(n)]


def obs_stats_seeds(n: int):
    return [OBS_STATS_BASE + i for i in range(n)]


def fixed_search_seeds(n: int, start: int = 0):
    return [FIXED_SEARCH_BASE + i for i in range(start, start + n)]

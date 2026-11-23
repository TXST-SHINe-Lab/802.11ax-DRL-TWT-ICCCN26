import numpy as np
import pytest

import hw_constraints as hwc
import seeds


def test_only_90_and_80_ms_schedules_are_illegal():
    # Proposal, Section 5: "only schedules 0 and 1 (single 90 and 80 ms windows) exceed 65.28 ms"
    assert hwc.illegal_schedules() == (0, 1)
    assert hwc.num_legal_joint_actions() == 18 * 19 == 342


def test_mask_layout_matches_maskableppo_multidiscrete():
    m = hwc.multidiscrete_action_mask()
    assert m.shape == (20 + 19,)
    assert not m[0] and not m[1] and m[2:20].all()
    assert m[20:].all()


def test_nearest_legal_schedule():
    assert hwc.nearest_legal_schedule(0) == 2
    assert hwc.nearest_legal_schedule(1) == 2
    for s in hwc.legal_schedules():
        assert hwc.nearest_legal_schedule(s) == s
    assert hwc.clip_action([0, 7]) == (2, 7)
    assert hwc.clip_action([99, -3]) == (19, 0)  # out-of-range indices are clipped first


@pytest.mark.parametrize("ms,expected", [(65.28, 65.28), (20.0, 20.224), (0.256, 0.256), (65.3, 65.536)])
def test_round_up_to_256us(ms, expected):
    assert hwc.round_up_to_granularity_ms(ms) == pytest.approx(expected)


def test_window_that_exceeds_limit_only_after_rounding_is_illegal():
    assert hwc.round_up_to_granularity_ms(65.28) <= hwc.MAX_SP_DURATION_MS
    assert hwc.round_up_to_granularity_ms(65.29) > hwc.MAX_SP_DURATION_MS


def test_seed_ranges_are_disjoint():
    span = seeds.SPAN
    ranges = []
    for s in seeds.SWEEP_RUN_SEEDS:
        b = seeds.train_env_base(s, "sweep")
        ranges.append((b, b + span))
    for s in seeds.FINAL_RUN_SEEDS:
        b = seeds.train_env_base(s, "train")
        ranges.append((b, b + span))
    ranges.append((seeds.OBS_STATS_BASE, seeds.OBS_STATS_BASE + 1000))
    ranges.append((seeds.EVAL_CURVE_BASE, seeds.EVAL_CURVE_BASE + seeds.N_EVAL_CURVE))
    ranges.append((seeds.TEST_BASE, seeds.TEST_BASE + seeds.N_TEST))
    ranges.sort()
    for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
        assert a1 <= b0, f"overlap {a0}-{a1} and {b0}-{b1}"
    with pytest.raises(ValueError):
        seeds.train_env_base(101, "train")
    with pytest.raises(ValueError):
        seeds.train_env_base(1, "sweep")

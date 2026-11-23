import copy

import numpy as np
import pytest

import hw_constraints as hwc
from ap_observation import (APObservationBuilder, FEATURE_NAMES, FixedZScore, PER_STA_FEATURES,
                            commanded_durations_ms)


def fake_env(t_s, n=5, scale=1.0):
    stas = []
    for i in range(n):
        stas.append({
            "realistic": {
                "bsr_queue_ac_be": 10.0 * i,
                "airtime_used_us": 1e4 * (i + 1) * t_s * scale,
                "fcs_error_count": 2.0 * t_s,
                "rx_fragment_count": 1.0 * t_s,
                "last_rx_timestamp_us": t_s * 1e6,
                "packets_received_at_ap": 50.0 * (i + 1) * t_s * scale,
            },
            "oracle": {
                "awake_time_ms": 111.0, "sleep_time_ms": 222.0, "duty_cycle": 0.33,
                "packets_transmitted": 444.0, "total_energy_consumed_mj": 5.0,
                "queue_size_bytes": 1234.0,
            },
        })
    return {"num_sta": n, "simulation_time_sec": t_s, "sta_observations": stas}


def poison_oracle(env):
    env = copy.deepcopy(env)
    for sta in env["sta_observations"]:
        for k in sta["oracle"]:
            sta["oracle"][k] = 9.9e30
    return env


def run(builder, envs, actions):
    out = [builder.reset(envs[0], actions[0])]
    for prev, curr, a in zip(envs, envs[1:], actions[1:]):
        out.append(builder.update(prev, curr, a))
    return out


def test_observation_never_reads_oracle_fields():
    envs = [fake_env(t) for t in (10.0, 35.0, 60.0)]
    acts = [(2, 0), (5, 4), (17, 6)]
    clean = run(APObservationBuilder(), envs, acts)
    poisoned = run(APObservationBuilder(), [poison_oracle(e) for e in envs], acts)
    for a, b in zip(clean, poisoned):
        np.testing.assert_array_equal(a, b)


def test_commanded_time_bookkeeping():
    b = APObservationBuilder()
    e0, e1 = fake_env(10.0), fake_env(35.0)  # 25 s epoch
    b.reset(e0, (2, 0))  # warmup: 60 ms, everyone in group 0
    obs = b.update(e0, e1, (7, 4)).reshape(-1, PER_STA_FEATURES)  # 2 x 30 ms, round-robin
    i_awake, i_sleep, i_duty = FEATURE_NAMES.index("cmd_awake_time_ms"), FEATURE_NAMES.index("cmd_sleep_time_ms"), FEATURE_NAMES.index("cmd_duty_cycle")
    expected_awake = 25000.0 * 30.0 / hwc.BEACON_INTERVAL_MS
    np.testing.assert_allclose(obs[:5, i_awake], expected_awake, rtol=1e-6)
    np.testing.assert_allclose(obs[:5, i_awake] + obs[:5, i_sleep], 25000.0, rtol=1e-6)
    np.testing.assert_allclose(obs[:5, i_duty], 30.0 / hwc.BEACON_INTERVAL_MS, rtol=1e-6)
    i_dpk = FEATURE_NAMES.index("delta_packets_received_at_ap")
    np.testing.assert_allclose(obs[:5, i_dpk], [50.0 * (i + 1) * 25.0 for i in range(5)])


def test_padding_slots_are_zero_and_stay_zero_after_zscore():
    b = APObservationBuilder()
    obs = b.reset(fake_env(10.0), (2, 0))
    assert obs.shape == (16 * 13,)
    assert not obs.reshape(16, 13)[5:].any()
    z = FixedZScore(np.full(13, 3.0), np.full(13, 2.0))
    zo = z(obs, 5).reshape(16, 13)
    assert not zo[5:].any()
    np.testing.assert_allclose(zo[:5], (obs.reshape(16, 13)[:5] - 3.0) / 2.0, rtol=1e-5)


def test_commanded_durations_follow_assignment_pattern():
    # schedule 10 = 60 + 20 ms; assignment 13 = weighted 75/25 -> 4 STAs in group 0, 1 in group 1
    d = commanded_durations_ms((10, 13), 5)
    assert sorted(d.tolist()) == [20.0, 60.0, 60.0, 60.0, 60.0]


def test_zscore_save_load_roundtrip(tmp_path):
    rows = np.random.default_rng(0).normal(size=(100, 13)) * 5 + 2
    z = FixedZScore.fit(rows)
    p = tmp_path / "s.json"
    z.save(str(p))
    z2 = FixedZScore.load(str(p))
    np.testing.assert_allclose(z.mean, z2.mean)
    np.testing.assert_allclose(z.std, z2.std)

"""End-to-end tests of the constrained env over the real FileCommEnv file protocol,
using the test-only mock runner in place of ns-3."""

import os

import numpy as np
import pytest

import hw_constraints as hwc
from conftest import MOCK_RUNNER
from hw_env import HwConstrainedTWTEnv, IllegalActionError
from policies import FixedSchedulePolicy, RandomLegalPolicy
from evaluation import run_episodes


@pytest.fixture
def env_factory(tmp_path, monkeypatch):
    envs = []

    def make(**kw):
        kw.setdefault("comm_dir", str(tmp_path / f"comm{len(envs)}"))
        kw.setdefault("runner_script", MOCK_RUNNER)
        kw.setdefault("num_epochs", 20)
        kw.setdefault("step_timeout", 30)
        kw.setdefault("spawn_timeout", 30)
        e = HwConstrainedTWTEnv(**kw)
        envs.append(e)
        return e

    yield make
    for e in envs:
        e.close()


def test_episode_truncates_after_num_epochs(env_factory):
    env = env_factory(num_epochs=7)
    env.reset(seed=5)
    for k in range(7):
        _, r, term, trunc, info = env.step([2, 0])
        assert np.isfinite(r) and not term
        assert trunc == (k == 6)
    m = info["episode_metrics"]
    assert m["epochs"] == 7 and m["num_sta"] == 5 and m["energy_mj"] > 0


def test_illegal_action_raises_by_default(env_factory):
    env = env_factory()
    env.reset(seed=6)
    with pytest.raises(IllegalActionError):
        env.step([0, 3])


def test_clip_mode_never_sends_illegal_schedule(env_factory, tmp_path, monkeypatch):
    log = tmp_path / "actions.log"
    monkeypatch.setenv("MOCK_ACTION_LOG", str(log))
    env = env_factory(illegal_action="clip", num_epochs=5)
    env.reset(seed=7)
    for s in (0, 1, 0, 4, 1):
        _, _, _, trunc, info = env.step([s, 0])
    assert info["clipped_actions"] == 4
    received = [int(x) for x in log.read_text().split()]
    assert received == [2, 2, 2, 4, 2]


def test_same_seed_same_episode(env_factory):
    env = env_factory(num_epochs=5)
    a = run_episodes(FixedSchedulePolicy(7, 4), env, [42])[0]
    b = run_episodes(FixedSchedulePolicy(7, 4), env, [42])[0]
    assert a["return"] == pytest.approx(b["return"]) and a["energy_mj"] == pytest.approx(b["energy_mj"])


def test_random_legal_policy_stays_legal(env_factory, tmp_path, monkeypatch):
    log = tmp_path / "actions.log"
    monkeypatch.setenv("MOCK_ACTION_LOG", str(log))
    env = env_factory(num_epochs=20)
    run_episodes(RandomLegalPolicy(seed=3), env, [1, 2])
    received = {int(x) for x in log.read_text().split()}
    assert not received & set(hwc.illegal_schedules())


def test_runner_failure_surfaces_as_error(env_factory, monkeypatch):
    monkeypatch.setenv("MOCK_FAIL_AT", "3")
    env = env_factory()
    env.reset(seed=8)
    env.step([2, 0])
    env.step([2, 0])
    with pytest.raises(RuntimeError):
        env.step([2, 0])


def test_kill_process_leaves_other_runs_shared_memory_alone(env_factory):
    if not os.path.isdir("/dev/shm"):
        pytest.skip("no /dev/shm on this OS")
    other = "/dev/shm/MySeg_424242"
    open(other, "w").close()
    try:
        env = env_factory(num_epochs=2)
        env.reset(seed=9)
        env.step([2, 0])
        env.step([2, 0])  # truncation -> _kill_process
        env.close()
        assert os.path.exists(other)
    finally:
        os.remove(other)


def test_hw_runner_clips_warmup_schedule():
    import hw_episode_runner  # noqa: F401  (patches episode_file_runner on import)
    import episode_file_runner as efr
    s_tab, a_tab = hwc.load_tables()
    d = efr.build_action_dict(0, 0, s_tab, a_tab, 5)
    assert [g["twt_wake_duration_ms"] for g in d["twt_group_configs"]] == [60.0]
    d = efr.build_action_dict(17, 6, s_tab, a_tab, 5)
    assert [g["twt_wake_duration_ms"] for g in d["twt_group_configs"]] == [22.0] * 4

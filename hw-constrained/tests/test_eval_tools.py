"""Statistics, fixed-action de-duplication and the shared-memory name tag."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyze  # noqa: E402
import best_fixed  # noqa: E402
import hw_constraints as hwc  # noqa: E402


def test_iqm_drops_outer_quartiles():
    assert analyze.iqm([1, 2, 3, 4, 100, -100, 5, 6]) == np.mean([2, 3, 4, 5])
    assert analyze.iqm([3.0]) == 3.0


def test_probability_of_improvement():
    assert analyze.prob_improvement([2, 3], [0, 1]) == 1.0
    assert analyze.prob_improvement([1], [1]) == 0.5
    lo, hi = analyze.poi_ci([5, 6, 7], [1, 2, 3], 200, np.random.default_rng(0))
    assert lo == hi == 1.0


def test_bootstrap_ci_brackets_the_statistic():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    lo, hi = analyze.bootstrap_ci(x, np.mean, 2000, rng)
    assert lo < x.mean() < hi


def test_fixed_actions_deduplicated_cover_all_legal_actions():
    groups = best_fixed.distinct_actions(hwc.legal_schedules(), 5)
    covered = sorted(tuple(a) for v in groups.values() for a in v)
    assert covered == sorted((s, a) for s in hwc.legal_schedules() for a in range(hwc.num_assignments()))
    assert len(groups) < len(covered)
    # A one-group schedule puts every STA in group 0 whatever the assignment.
    one_group = [s for s in hwc.legal_schedules() if hwc.load_tables()[0]["schedules"][s]["num_groups"] == 1]
    for s in one_group:
        assert sum(1 for sp in groups if sp.startswith(f"fixed:{s}:")) == 1


def test_shm_names_get_the_tag():
    import hw_episode_runner
    import pb_twt_wrapper_py as ptw

    original = ptw.TWTWrapper.reset
    try:
        hw_episode_runner.tag_shm_names("p42")

        class Stop(Exception):
            pass

        class FakeExp:
            _created = False

            def __init__(self, *a, **k):
                raise Stop(k["segName"])

        sys.modules["ns3ai_utils"] = type(sys)("ns3ai_utils")
        sys.modules["ns3ai_utils"].Experiment = FakeExp
        w = ptw.TWTWrapper.__new__(ptw.TWTWrapper)
        w.initialized, w.verbose, w.enable_logging = False, False, False
        w._import_bindings, w._log, w.ns3_root, w.py_binding = (lambda: None), (lambda *a, **k: None), "/tmp", None
        try:
            w.reset(seed=8000001)
        except Exception:  # noqa: BLE001  (the fake Experiment stops reset; the wrapper re-raises)
            pass
        assert w._segmentName == "MySeg_8000001_p42"
        assert w._lockableName == "MyLockable_8000001_p42"
    finally:
        ptw.TWTWrapper.reset = original
        sys.modules.pop("ns3ai_utils", None)

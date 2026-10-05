"""Evaluation protocol, threshold selection, bootstrap and the test-once guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fleet_signal.eval.bootstrap import bootstrap_ci, paired_difference
from fleet_signal.eval.official import OfficialResultExists, write_demo, write_official
from fleet_signal.eval.protocol import EvalConfig, ScoredSet, breakdown, meets_bar
from fleet_signal.eval.threshold import select_threshold, sweep, threshold_candidates
from fleet_signal.incidents.grouping import IncidentParams

ECFG = EvalConfig.load()
P1 = IncidentParams(open_n=1, close_m=5, cooldown_c=0)


def _scores(run_id: str, asset_id: str, n: int, spikes: dict[int, float]) -> pd.DataFrame:
    score = np.zeros(n)
    for k, v in spikes.items():
        score[k] = v
    return pd.DataFrame(
        {"run_id": run_id, "asset_id": asset_id, "seq": np.arange(n), "score": score}
    )


def _fault(run_id: str, asset_id: str, start: int, end: int, ftype: str = "overheating"):
    return {
        "run_id": run_id,
        "asset_id": asset_id,
        "fault_type": ftype,
        "variant": "linear",
        "progressive": ftype in ECFG.progressive,
        "fault_start_seq": start,
        "fault_end_seq": end,
    }


@pytest.fixture
def toy() -> ScoredSet:
    """r1: normal, one false spike. r2: fault [100, 200) on a, detected at 103 (latency 3)."""
    scored = pd.concat(
        [
            _scores("r1", "a", 300, {50: 5.0}),
            _scores("r2", "a", 300, {103: 5.0, 104: 5.0}),
        ]
    )
    faults = pd.DataFrame([_fault("r2", "a", 100, 200)])
    return ScoredSet(scored, faults, ["r1", "r2"], ECFG, n_assets=1)


def test_metric_definitions_on_toy(toy: ScoredSet) -> None:
    res = toy.evaluate(1.0, P1)
    s = res.summary
    assert s["n_incidents"] == 2 and s["n_false_incidents"] == 1
    assert s["precision"] == pytest.approx(0.5)
    assert s["recall"] == pytest.approx(1.0)
    assert s["progressive_latency_median"] == 3
    # normal time: r1 all 300 s + r2 outside [100, 200 + grace)
    normal_s = 300 + (300 - (100 + ECFG.grace))
    assert s["normal_fleet_minutes"] == pytest.approx(normal_s / 60)
    assert s["fp_per_10min"] == pytest.approx(1 / (normal_s / 600))
    assert res.per_fault.iloc[0]["latency_events"] == 3


def test_nan_scores_never_alert_and_do_not_count_as_normal_time() -> None:
    scored = _scores("r1", "a", 100, {50: 5.0})
    scored.loc[scored["seq"] < 60, "score"] = np.nan  # e.g. insufficient history
    ss = ScoredSet(scored, pd.DataFrame(columns=list(_fault("x", "a", 0, 1))), ["r1"], ECFG, 1)
    s = ss.evaluate(1.0, P1).summary
    assert s["n_incidents"] == 0
    assert s["normal_fleet_minutes"] == pytest.approx(40 / 60)


def test_detection_outside_grace_is_missed_and_false() -> None:
    late = 200 + ECFG.grace + 5
    ss = ScoredSet(
        _scores("r2", "a", 300, {late: 5.0}),
        pd.DataFrame([_fault("r2", "a", 100, 200)]),
        ["r2"],
        ECFG,
        1,
    )
    s = ss.evaluate(1.0, P1).summary
    assert s["recall"] == 0.0 and s["n_false_incidents"] == 1


def test_incident_on_wrong_asset_does_not_detect() -> None:
    scored = pd.concat([_scores("r2", "a", 300, {}), _scores("r2", "b", 300, {120: 5.0})])
    ss = ScoredSet(scored, pd.DataFrame([_fault("r2", "a", 100, 200)]), ["r2"], ECFG, 2)
    s = ss.evaluate(1.0, P1).summary
    assert s["recall"] == 0.0 and s["n_false_incidents"] == 1


def test_already_open_incident_counts_tp_but_not_as_detection() -> None:
    """An alarm that started before the fault overlaps it (TP) but did not detect it."""
    spikes = {k: 5.0 for k in range(90, 150)}
    ss = ScoredSet(
        _scores("r2", "a", 300, spikes),
        pd.DataFrame([_fault("r2", "a", 100, 200)]),
        ["r2"],
        ECFG,
        1,
    )
    s = ss.evaluate(1.0, P1).summary
    assert s["n_false_incidents"] == 0 and s["recall"] == 0.0


def test_breakdown_and_bar(toy: ScoredSet) -> None:
    res = toy.evaluate(1.0, P1)
    table = breakdown(res.per_fault, ["fault_type"])
    assert table.loc[0, "recall"] == 1.0
    bar = meets_bar(res.summary, ECFG)
    assert bar == {"precision": False, "recall": True, "false_alerts": True, "latency": True}


# ---------------------------------------------------------------- threshold selection


def test_candidates_cover_upper_tail_and_never_alert() -> None:
    scores = np.random.default_rng(0).normal(size=10_000)
    c = threshold_candidates(scores, 40)
    assert c.max() > scores.max()
    assert np.all(np.diff(c) > 0)


def test_select_threshold_respects_budget_then_maximises_recall() -> None:
    curve = pd.DataFrame(
        {
            "threshold": [1.0, 2.0, 3.0, 4.0],
            "recall": [1.0, 0.9, 0.9, 0.5],
            "fp_per_10min": [5.0, 1.4, 1.0, 0.1],
            "progressive_latency_median": [1, 4, 4, 9],
        }
    )
    pick = select_threshold(curve, budget=1.5)
    assert pick["feasible"] and pick["threshold"] == 3.0  # tie on recall/latency -> higher thr


def test_select_threshold_enforces_min_precision() -> None:
    curve = pd.DataFrame(
        {
            "threshold": [1.0, 2.0, 3.0],
            "recall": [1.0, 0.9, 0.7],
            "precision": [0.3, 0.6, 0.95],
            "f1": [0.46, 0.72, 0.81],
            "fp_per_10min": [1.0, 0.5, 0.0],
            "progressive_latency_median": [1, 1, 1],
        }
    )
    assert select_threshold(curve, budget=1.5, min_precision=0.8)["threshold"] == 3.0
    assert select_threshold(curve, budget=1.5, min_precision=0.0)["threshold"] == 1.0


def test_select_threshold_infeasible_is_flagged() -> None:
    curve = pd.DataFrame(
        {
            "threshold": [1.0, 2.0],
            "recall": [1.0, 0.9],
            "fp_per_10min": [5.0, 3.0],
            "progressive_latency_median": [1, 1],
        }
    )
    pick = select_threshold(curve, budget=1.5)
    assert not pick["feasible"] and pick["threshold"] == 2.0


def test_sweep_monotone_in_threshold(toy: ScoredSet) -> None:
    curve = sweep(toy, P1, np.array([0.5, 1.0, 6.0]))
    assert curve["n_incidents"].is_monotonic_decreasing


# ---------------------------------------------------------------- bootstrap


def test_paired_difference_of_identical_detectors_is_zero(toy: ScoredSet) -> None:
    pr = toy.evaluate(1.0, P1).per_run
    d = paired_difference(pr, pr, 1, 200, 0)
    assert d["recall"] == {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}


def test_bootstrap_ci_brackets_point_estimate(toy: ScoredSet) -> None:
    pr = toy.evaluate(1.0, P1).per_run
    ci = bootstrap_ci(pr, 1, 500, 0)
    lo, hi = ci["fp_per_10min"]
    assert lo <= toy.evaluate(1.0, P1).summary["fp_per_10min"] <= hi


# ---------------------------------------------------------------- test-once guard


def test_official_result_cannot_be_overwritten(tmp_path: Path) -> None:
    write_official("rule", "v1", {"recall": 0.9}, root=tmp_path)
    with pytest.raises(OfficialResultExists):
        write_official("rule", "v1", {"recall": 0.99}, root=tmp_path)
    assert '"recall": 0.9' in (tmp_path / "test_rule_v1.json").read_text()
    write_official("rule", "v2", {"recall": 0.8}, root=tmp_path)  # new model version is fine


def test_demo_results_go_elsewhere(tmp_path: Path) -> None:
    path = write_demo("threshold_demo", {"threshold": 1.0}, root=tmp_path / "demo")
    assert path.parent.name == "demo"

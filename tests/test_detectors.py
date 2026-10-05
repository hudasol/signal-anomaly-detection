"""Baseline detectors: scoring contract, rule semantics, robust-z fitting, evidence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from fleet_signal.data.config import GenerationConfig
from fleet_signal.data.generator import generate_run
from fleet_signal.data.splits import PlannedRun
from fleet_signal.detectors.rule import NOT_APPLICABLE, RuleDetector
from fleet_signal.detectors.stats import StatsDetector, robust_scale
from fleet_signal.eval.protocol import EvalConfig, ScoredSet
from fleet_signal.features.build import FEATURE_COLUMNS, build_features
from fleet_signal.incidents.grouping import IncidentParams


def _rows(n: int = 3, **values: object) -> pd.DataFrame:
    """Scorable feature rows with neutral values, overridden by keyword."""
    base: dict[str, object] = {c: 0.0 for c in FEATURE_COLUMNS}
    base.update(
        run_id="r1", asset_id="rover-01", asset_type="rover", seq=0, mode="moving",
        mode_age=300.0, link_pct=80.0, link_min_m=70.0, link_std_m=2.0,
        history=500, history_ok=True, max_gap_l=1.0, gap_ok=True,
    )  # fmt: skip
    base.update(values)
    df = pd.DataFrame([base] * n)
    df["seq"] = np.arange(n)
    return df


# ---------------------------------------------------------------- rule baseline


def test_rule_quiet_on_neutral_rows_and_fires_over_limit() -> None:
    det = RuleDetector()
    assert (det.score(_rows()) < 0).all()
    hot = _rows(temp_slope_l=6.0)  # rover limit 3.0 C/min, relative scale -> (6-3)/3 = 1
    assert det.score(hot) == pytest.approx([1.0, 1.0, 1.0])
    assert det.evidence(hot, k=1)[0][0]["signal"] == "temp_rise"


def test_rule_limits_depend_on_asset_type() -> None:
    det = RuleDetector()
    drone = _rows(asset_type="drone", asset_id="drone-01", temp_slope_l=5.0)  # drone limit 5.5
    assert (det.score(drone) < 0).all()


def test_rule_conditions_respected() -> None:
    det = RuleDetector()
    # battery rule needs the long window inside one mode
    young = _rows(batt_slope_l=-5.0, mode_age=60.0)
    settled = _rows(batt_slope_l=-5.0, mode_age=300.0)
    assert det.score(young).max() < det.score(settled).min()
    # freeze rules only while moving / returning
    parked = _rows(speed_unchanged=50.0, mode="idle")
    moving = _rows(speed_unchanged=50.0, mode="moving")
    assert det.score(parked).max() < 0 < det.score(moving).min()


def test_rule_battery_limit_is_mode_specific() -> None:
    det = RuleDetector()
    # -0.8 %/min is normal while moving (limit -1.1) but abnormal while charging (limit +0.5)
    assert (det.score(_rows(batt_slope_l=-0.8, mode="moving")) < 0).all()
    assert (det.score(_rows(batt_slope_l=-0.8, mode="charging")) > 0).all()


def test_rule_not_applicable_score() -> None:
    det = RuleDetector(rules=[{"name": "x", "feature": "temp_unchanged", "op": ">", "limit": 5,
                               "scale": "relative", "when": {"mode_in": ["moving"]}}])  # fmt: skip
    assert (det.score(_rows(mode="charging")) == NOT_APPLICABLE).all()


def test_unscorable_rows_get_nan_not_normal() -> None:
    rows = _rows(temp_slope_l=99.0)
    rows.loc[0, "history_ok"] = False
    rows.loc[1, "gap_ok"] = False
    rows.loc[2, "temp_c"] = np.nan
    det = RuleDetector()
    assert np.isnan(det.score(rows)).all()
    assert det.evidence(rows) == [[], [], []]


# ---------------------------------------------------------------- robust z baseline


def test_robust_scale_floors() -> None:
    assert robust_scale(np.zeros(100)) == pytest.approx(1e-3)
    x = np.random.default_rng(0).normal(0, 2, 100_000)
    assert robust_scale(x) == pytest.approx(2.0, rel=0.05)


def test_stats_detector_learns_per_mode_envelope() -> None:
    rng = np.random.default_rng(1)
    moving = _rows(500, mode="moving")
    moving["batt_slope_l"] = rng.normal(-0.8, 0.05, 500)
    charging = _rows(500, mode="charging")
    charging["batt_slope_l"] = rng.normal(4.0, 0.2, 500)
    for df in (moving, charging):
        for col in ("temp_c", "speed_mps", "link_pct"):
            df[col] = rng.normal(50, 1, 500)
    det = StatsDetector().fit(pd.concat([moving, charging], ignore_index=True))

    probe = _rows(2, batt_slope_l=-0.8, temp_c=50.0, speed_mps=50.0, link_pct=50.0)
    probe["mode"] = ["moving", "charging"]
    z = det._contributions(probe)
    assert z.loc[0, "batt_slope_l"] < 1.0  # normal while moving
    assert z.loc[1, "batt_slope_l"] > 10.0  # draining while charging
    assert det.evidence(probe, k=1)[1][0]["signal"] == "batt_slope_l"


def test_stats_detector_must_be_fitted() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        StatsDetector().score(_rows())


# ---------------------------------------------------------------- end to end on generated data


def test_rule_detects_a_generated_position_jump(small_cfg: GenerationConfig) -> None:
    cfg = replace(small_cfg, data={**small_cfg.data, "run_duration_s": 600})
    run = PlannedRun("r2003", 2003, "validation", "motion_anomaly", 0, "jump")
    result = generate_run(cfg, run)
    feats = build_features(result.telemetry)
    scored = feats[["run_id", "asset_id", "seq"]].assign(score=RuleDetector().score(feats))
    faults = pd.DataFrame([result.fault])
    ss = ScoredSet(scored, faults, ["r2003"], EvalConfig.load(), n_assets=3)
    res = ss.evaluate(0.0, IncidentParams(2, 5, 30))
    assert res.summary["recall"] == 1.0
    assert res.per_fault.iloc[0]["latency_events"] <= 3

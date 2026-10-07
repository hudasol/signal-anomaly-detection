"""Exceeds-the-bar pieces: drift monitor, prioritisation, ablation knobs, shifted fleets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fleet_signal.data.config import load_config
from fleet_signal.eval.exceeds import SHIFTS, AblatedFastPath, shifted_config
from fleet_signal.features.build import FEATURE_COLUMNS
from fleet_signal.incidents.priority import WEIGHTS, severity, time_to_critical
from fleet_signal.monitoring.drift import DriftMonitor, psi


def _rows(n: int, seed: int = 0, noise: float = 1.0, **kw: Any) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base: dict[str, Any] = {c: 0.0 for c in FEATURE_COLUMNS}
    base.update(asset_type="rover", mode="moving", mode_age=300.0, history=500,
                history_ok=True, max_gap_l=1.0, gap_ok=True, run_id=f"r{seed}",
                asset_id="rover-01", battery_pct=60.0, temp_c=35.0, link_pct=80.0)  # fmt: skip
    df = pd.DataFrame([base] * n)
    df["seq"] = np.arange(n)
    for c in ("temp_c", "speed_mps", "link_pct", "temp_slope_l", "batt_slope_l", "link_std_m",
              "speed_mismatch", "spd_d3", "batt_res3", "temp_res3", "batt_res10", "temp_res10",
              "spd_d10"):  # fmt: skip
        df[c] = df[c] + rng.normal(0, 1.0, n)
    df["temp_res3"] = rng.normal(0, noise, n)
    for k, v in kw.items():
        df[k] = v
    return df


# ---------------------------------------------------------------- drift


def test_psi_is_small_for_the_same_distribution_and_large_for_a_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    edges = np.quantile(ref, np.linspace(0, 1, 11)[1:-1])
    props = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=10) / len(ref)
    assert psi(props, edges, rng.normal(size=2000)) < 0.05
    assert psi(props, edges, rng.normal(1.5, 1, size=2000)) > 1.0


def test_drift_monitor_flags_noisier_sensors_and_passes_normal(tmp_path) -> None:
    mon = DriftMonitor().fit(_rows(5000, seed=1))
    mon.calibrate([_rows(600, seed=s) for s in range(2, 12)])
    assert mon.score(_rows(600, seed=50))["status"] in ("ok", "caution")
    noisy = mon.score(_rows(600, seed=51, noise=2.0))
    assert noisy["status"] == "drift" and noisy["top"][0][0] == "temp_res3"
    path = tmp_path / "drift.json"
    mon.save(path)
    again = DriftMonitor.load(path)
    assert again.score(_rows(600, seed=51, noise=2.0))["score"] == pytest.approx(noisy["score"])


def test_drift_monitor_needs_enough_rows() -> None:
    mon = DriftMonitor().fit(_rows(5000, seed=1))
    assert mon.score(_rows(20, seed=3))["status"] == "insufficient_data"


# ---------------------------------------------------------------- prioritisation


def test_time_to_critical_takes_the_soonest_signal() -> None:
    row = pd.Series({"battery_pct": 30.0, "batt_slope_m": -2.0, "temp_c": 50.0,
                     "temp_slope_m": 1.0, "link_pct": 80.0, "link_slope_m": 0.0})  # fmt: skip
    assert time_to_critical(row) == (10.0, "battery")  # (30-10)/2 vs (70-50)/1 = 20
    row["temp_c"] = 72.0
    assert time_to_critical(row) == (0.0, "temperature")  # already past critical
    flat = pd.Series({"battery_pct": 60.0, "batt_slope_m": 0.5, "temp_c": 40.0,
                      "temp_slope_m": -0.1, "link_pct": 90.0, "link_slope_m": 0.2})  # fmt: skip
    assert time_to_critical(flat) == (None, None)


def test_severity_is_the_stated_weighted_sum() -> None:
    from fleet_signal.detectors.fastpath import FastPathDetector
    from fleet_signal.detectors.hybrid import HybridDetector
    from fleet_signal.detectors.rule import RuleDetector

    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    train = _rows(2000, seed=1)
    det = HybridDetector([RuleDetector(), FastPathDetector()], "h").fit(train)
    feats = _rows(20, seed=2, battery_pct=12.0, batt_slope_m=-1.0)  # 2 min to critical
    scores = np.full(20, 3.0)
    s = severity(det, 1.0, feats, scores, 15)
    assert s.urgency == 1.0 and s.time_to_critical_min == pytest.approx(2.0)
    assert s.strength == pytest.approx(1.0)  # mean score/threshold 3 -> 1
    expected = 0.40 * s.urgency + 0.35 * s.strength + 0.25 * s.breadth
    assert s.severity == pytest.approx(round(expected, 3)) and s.level == "P1"
    calm = severity(det, 1.0, _rows(20, seed=3), np.full(20, 1.1), 15)
    assert calm.level == "P3"


# ---------------------------------------------------------------- ablation + shifted fleets


def test_ablated_fast_path_drops_signals_and_covariate() -> None:
    train = _rows(2000, seed=1)
    keep_batt = AblatedFastPath(("batt_res3", "batt_res10")).fit(train)
    z = keep_batt.zscores(_rows(5, seed=2))
    assert np.isnan(z[:, 2:]).all() and np.isfinite(z[:, :2]).all()
    no_cov = AblatedFastPath(("batt_res3",), covariate=False).fit(train)
    assert all(np.allclose(m[:, 0], 0.0) for m in no_cov.models.values())  # slope fixed at 0


def test_shifted_configs_change_only_what_they_say() -> None:
    base = load_config()
    for name in SHIFTS:
        cfg = shifted_config(name, seed_offset=100)
        assert cfg.splits["splits"]["test"]["seed_start"] == SHIFTS[name]["seed_start"] + 100
        assert cfg.version != base.version
    hot = shifted_config("hot_climate")
    assert hot.data["ambient_c"] == [40.0, 50.0] and base.data["ambient_c"] == [22.0, 42.0]
    noisy = shifted_config("noisy_sensors")
    for t, prof in noisy.data["profiles"].items():
        assert prof["noise"]["temp"] == 2 * base.data["profiles"][t]["noise"]["temp"]
    aged = shifted_config("aged_batteries")
    for t, prof in aged.data["profiles"].items():
        assert prof["drain_load"] == pytest.approx(1.3 * base.data["profiles"][t]["drain_load"])

"""v2: fast-path features and detector, hybrid, detectability, v2 ship rule, partial generation."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fleet_signal.data.config import GenerationConfig, load_config
from fleet_signal.data.generator import generate_dataset
from fleet_signal.data.splits import build_run_plan
from fleet_signal.data.telemetry import load_telemetry
from fleet_signal.detectors.fastpath import NOT_APPLICABLE, FastPathDetector
from fleet_signal.detectors.hybrid import HybridDetector
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.eval.decision import ship_decision_v2
from fleet_signal.eval.detectability import with_detectability
from fleet_signal.eval.protocol import expected_latency
from fleet_signal.eval.select_v2 import _rank_key, meets_validation_bar
from fleet_signal.features.build import (
    FEATURE_COLUMNS,
    events_since_any_change,
    residual_slope,
)

# ---------------------------------------------------------------- features


def test_residual_slope_cancels_a_steady_trend_and_shows_a_new_one_after_k() -> None:
    t = np.arange(200, dtype=float)
    y = -0.05 * t  # steady drain
    r = residual_slope(t, y, 3, 30)
    assert np.allclose(r[40:], 0.0, atol=1e-9)
    y2 = y.copy()
    y2[150:] -= 0.04 * (t[150:] - 150)  # extra drain starts at index 150 (0 at onset)
    r2 = residual_slope(t, y2, 3, 30)
    assert np.allclose(r2[40:151], 0.0, atol=1e-9)  # NaN before 33 events of history
    assert r2[153] == pytest.approx(-0.04)  # full size 3 events after onset
    assert -0.04 < r2[151] < 0  # partly visible earlier


def test_residual_slope_is_causal() -> None:
    t = np.arange(120, dtype=float)
    y = np.random.default_rng(0).normal(size=120)
    a = residual_slope(t, y, 10, 30)
    y[100:] += 50  # change the future
    b = residual_slope(t, y, 10, 30)
    assert np.allclose(a[:100], b[:100], equal_nan=True)


def test_position_freeze_needs_both_coordinates_to_repeat() -> None:
    x = np.array([1.0, 1.0, 1.0, 2.0, 2.0])
    y = np.array([5.0, 6.0, 6.0, 6.0, 6.0])
    assert events_since_any_change([x, y], 300).tolist() == [0, 0, 1, 0, 1]
    # the v1 key x*1e6+y could not tell (x+0.1, y-1e5) from (x, y); a tuple compare can
    assert events_since_any_change([np.array([0.0, 0.1]), np.array([1e5, 0.0])], 9).tolist() == [
        0,
        0,
    ]


# ---------------------------------------------------------------- fast path detector


def _rows(n: int, **kw: Any) -> pd.DataFrame:
    rng = np.random.default_rng(kw.pop("seed", 0))
    base: dict[str, Any] = {c: 0.0 for c in FEATURE_COLUMNS}
    base.update(asset_type="rover", mode="moving", mode_age=300.0, history=500,
                history_ok=True, max_gap_l=1.0, gap_ok=True, run_id="r1",
                asset_id="rover-01")  # fmt: skip
    df = pd.DataFrame([base] * n)
    df["seq"] = np.arange(n)
    for c in ("batt_res3", "batt_res10", "temp_res3", "temp_res10"):
        df[c] = rng.normal(0, 0.3, n)
    for c in ("spd_d3", "spd_d10"):
        df[c] = rng.normal(0, 0.2, n)
    for k, v in kw.items():
        df[k] = v
    return df


def test_fast_path_flags_extra_drain_and_ignores_the_opposite_direction() -> None:
    det = FastPathDetector().fit(_rows(2000))
    normal = det.score(_rows(200, seed=1))
    drain = det.score(_rows(5, seed=2, batt_res3=-3.0))
    less = det.score(_rows(5, seed=3, batt_res3=+3.0, batt_res10=0.0, temp_res3=0.0,
                           temp_res10=0.0))  # fmt: skip
    assert np.median(drain) > np.quantile(normal, 0.999)
    assert np.max(less) < 5  # a drop in drain is not an alarm
    ev = det.evidence(_rows(1, seed=4, batt_res3=-3.0), k=1)[0][0]
    assert ev["signal"] == "batt_res3"


def test_fast_path_learns_the_manoeuvre_covariate() -> None:
    train = _rows(3000)
    train["batt_res3"] = train["batt_res3"] - 2.0 * train["spd_d3"]  # drain follows speed
    det = FastPathDetector().fit(train)
    manoeuvre = _rows(5, seed=5, spd_d3=1.5, batt_res3=-3.0)  # explained by the speed-up
    unexplained = _rows(5, seed=6, spd_d3=0.0, batt_res3=-3.0)
    assert det.score(manoeuvre).max() < det.score(unexplained).min()


def test_fast_path_is_silent_right_after_a_mode_change() -> None:
    det = FastPathDetector().fit(_rows(2000))
    young = _rows(3, mode_age=10.0, batt_res3=-9.0)
    assert np.all(det.score(young) == NOT_APPLICABLE)


def test_fast_path_must_be_fitted() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        FastPathDetector().score(_rows(3))


# ---------------------------------------------------------------- hybrid


def test_hybrid_takes_the_strongest_part_and_explains_with_it() -> None:
    train = _rows(2000)
    train["mode_age"] = 300.0
    hyb = HybridDetector([RuleDetector(), FastPathDetector()], "hybrid_rule_fast").fit(train)
    quiet = hyb.score(_rows(50, seed=7))
    drain = _rows(1, seed=8, batt_res3=-4.0)
    s = hyb.score(drain)
    assert s[0] > np.quantile(quiet, 0.99)
    assert hyb.evidence(drain, k=1)[0][0]["signal"].startswith("fast:")
    assert hyb.params["parts"] == ["rule", "fast"]
    assert set(hyb.calibration) == {"rule", "fast"}


# ---------------------------------------------------------------- detectability + latency


def _fault(ftype: str, variant: str, params: dict[str, Any], progressive: bool = True):
    return {"run_id": "r1", "asset_id": "rover-01", "fault_type": ftype, "variant": variant,
            "progressive": progressive, "params_json": json.dumps(params)}  # fmt: skip


def test_expected_to_catch_is_physical_and_declared(cfg: GenerationConfig) -> None:
    faults = pd.DataFrame(
        [
            _fault("battery_drain", "step", {"extra_pct_per_min": 2.0}),
            _fault("battery_drain", "step", {"extra_pct_per_min": 1.0}),
            _fault("overheating", "linear", {"rate_c_per_min": 14.0}),
            _fault("link_degradation", "decline", {"decline_pct_per_min": 30.0}),
            _fault("sensor_freeze", "speed_mps", {}, progressive=False),
        ]
    )
    got = with_detectability(faults, cfg)["fast_detectable"].tolist()
    assert got == [True, False, False, False, False]


def test_expected_latency_counts_misses_as_infinite() -> None:
    pf = pd.DataFrame({"fast_detectable": [True, True, True, False],
                       "detected": [True, False, False, True],
                       "latency_events": [2, None, None, 1]})  # fmt: skip
    out = expected_latency(pf)
    assert out["n_expected"] == 3 and out["n_expected_detected"] == 1
    assert out["expected_latency_median"] == np.inf  # 2 of 3 missed: median is a miss


# ---------------------------------------------------------------- v2 ship rule and selection


def _d(diff: float, lo: float, hi: float) -> dict[str, float]:
    return {"diff": diff, "ci_low": lo, "ci_high": hi}


def test_v2_ship_rule() -> None:
    lat_win = {"n_paired": 20, "diff": -30, "ci_low": -50, "ci_high": -10}
    lat_none = {"n_paired": 20, "diff": -5, "ci_low": -20, "ci_high": 5}
    # not worse on recall + faster -> candidate ships
    d = ship_decision_v2("c", "rule", _d(0.04, -0.03, 0.1), _d(0.0, -0.1, 0.1), lat_win)
    assert d["ship"] == "c"
    # could be worse on recall -> baseline, even if faster
    d = ship_decision_v2("c", "rule", _d(-0.02, -0.12, 0.08), _d(0.1, 0.02, 0.2), lat_win)
    assert d["ship"] == "rule"
    # not worse but no significant gain -> baseline
    d = ship_decision_v2("c", "rule", _d(0.02, -0.04, 0.1), _d(0.05, -0.02, 0.1), lat_none)
    assert d["ship"] == "rule"
    # precision win alone is enough
    d = ship_decision_v2("c", "rule", _d(0.0, -0.04, 0.05), _d(0.1, 0.03, 0.2), lat_none)
    assert d["ship"] == "c"


def test_selection_prefers_candidates_that_meet_the_bar_on_validation() -> None:
    bar = {"recall": 0.85, "precision": 0.75, "false_alerts_per_10min": 2.0,
           "median_latency_events": 3}  # fmt: skip
    fast_ok = {"recall": 0.93, "precision": 0.9, "fp_per_10min": 0.1,
               "expected_latency_median": 1.5, "progressive_latency_median": 7.0}  # fmt: skip
    slow_hi = {"recall": 0.97, "precision": 0.9, "fp_per_10min": 0.1,
               "expected_latency_median": 5.0, "progressive_latency_median": 18.0}  # fmt: skip
    for r in (fast_ok, slow_hi):
        r["meets_bar"] = meets_validation_bar(r, bar)
    assert fast_ok["meets_bar"] and not slow_hi["meets_bar"]
    assert min([slow_hi, fast_ok], key=_rank_key) is fast_ok


# ---------------------------------------------------------------- generate test later


def test_generating_splits_separately_gives_identical_runs(small_cfg, tmp_path) -> None:
    plan = build_run_plan(small_cfg)
    val = [r for r in plan if r.split == "validation"][:2]
    test = [r for r in plan if r.split == "test"][:1]
    generate_dataset(small_cfg, tmp_path / "a", runs=val)
    generate_dataset(small_cfg, tmp_path / "b", runs=val + test)
    ids = [r.run_id for r in val]
    a = load_telemetry(data_dir=tmp_path / "a", run_ids=ids)
    b = load_telemetry(data_dir=tmp_path / "b", run_ids=ids)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert set(load_telemetry(data_dir=tmp_path / "a")["run_id"]) == set(ids)


def test_v2_test_seeds_are_new() -> None:
    cfg = load_config()
    test_seeds = {r.seed for r in build_run_plan(cfg) if r.split == "test"}
    assert min(test_seeds) == 4000 and not test_seeds & set(range(3000, 3060))


def test_regenerating_data_removes_a_stale_feature_cache(tmp_path) -> None:
    """Found reproducing v2 from the README in a fresh clone: generating the test split after
    the freeze reused the train/validation-only feature cache, so test had no features."""
    import yaml

    from fleet_signal.data.cli import main as data_cli
    from fleet_signal.data.config import DEFAULT_DATA_CONFIG

    raw = yaml.safe_load(DEFAULT_DATA_CONFIG.read_text())
    raw["run_duration_s"] = 200
    small = tmp_path / "data.yaml"
    small.write_text(yaml.safe_dump(raw))
    out = tmp_path / "data"
    data_cli(["generate", "--config", str(small), "--out", str(out), "--only-splits", "train"])
    version_dir = next(p for p in out.iterdir() if p.is_dir())
    cache = version_dir / "features" / "x.parquet"
    cache.parent.mkdir()
    cache.write_text("stale")
    data_cli(["generate", "--config", str(small), "--out", str(out), "--only-splits", "train"])
    assert not cache.parent.exists()

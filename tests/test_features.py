"""Preprocessing: hand-computed values, causality, run isolation, label isolation."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fleet_signal
from fleet_signal.data.config import GenerationConfig
from fleet_signal.data.generator import generate_run
from fleet_signal.data.splits import PlannedRun
from fleet_signal.features.build import (
    FEATURE_COLUMNS,
    FeatureConfig,
    build_features,
    events_since_change,
    rolling_slope,
    scorable,
)
from fleet_signal.features.store import SplitLeakError, assert_only_split

FCFG = FeatureConfig.load()


def _tel(
    n: int = 200, run_id: str = "r1", asset: str = "rover-01", **cols: np.ndarray
) -> pd.DataFrame:
    """A synthetic, perfectly controlled telemetry stream for one asset."""
    seq = np.arange(n)
    df = pd.DataFrame(
        {
            "run_id": run_id,
            "asset_id": asset,
            "asset_type": "rover",
            "seq": seq,
            "timestamp_utc": pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(seq, unit="s"),
            "x_m": seq * 1.0,
            "y_m": np.zeros(n),
            "z_m": np.zeros(n),
            "speed_mps": np.ones(n),
            "heading_deg": np.zeros(n),
            "battery_pct": 90.0 - 0.01 * seq,
            "temperature_c": 30.0 + 0.05 * seq,
            "link_quality_pct": 80.0 + (seq % 2),
            "mode": "moving",
        }
    )
    for k, v in cols.items():
        df[k] = v
    return df


# ---------------------------------------------------------------- primitives


def test_rolling_slope_exact_on_line_and_nan_before_window() -> None:
    t = np.arange(50, dtype=float)
    y = 3.0 + 0.5 * t
    s = rolling_slope(t, y, 10)
    assert np.isnan(s[:9]).all()
    assert np.allclose(s[9:], 0.5)


def test_rolling_slope_uses_real_time_across_missing_events() -> None:
    t = np.array([0, 1, 2, 5, 6, 7], dtype=float)  # events 3 and 4 were lost
    y = 2.0 * t
    assert np.allclose(rolling_slope(t, y, 4)[3:], 2.0)


def test_events_since_change() -> None:
    v = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3])
    assert events_since_change(v, cap=300).tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]
    assert events_since_change(v, cap=2).tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 2]


# ---------------------------------------------------------------- hand-computed values


def test_trend_features_match_hand_values() -> None:
    f = build_features(_tel(), FCFG)
    row = f.iloc[-1]
    assert row["temp_slope_l"] == pytest.approx(0.05 * 60)  # 0.05 C/s -> 3 C/min
    assert row["batt_slope_m"] == pytest.approx(-0.01 * 60)
    assert row["range_slope_m"] == pytest.approx(60.0)  # moving away at 1 m/s


def test_motion_consistency_features() -> None:
    n = 200
    x = np.arange(n, dtype=float)  # 1 m per event, consistent with speed 1
    x[150:] += 40.0  # a 40 m jump at event 150
    f = build_features(_tel(n, x_m=x), FCFG).set_index("seq")
    assert f.loc[140, "implied_speed"] == pytest.approx(1.0)
    assert f.loc[140, "speed_mismatch"] == pytest.approx(0.0)
    assert f.loc[140, "jump_excess"] == pytest.approx(0.0)
    assert f.loc[150, "jump_excess"] == pytest.approx(40.0)
    assert f.loc[154, "jump_excess"] == pytest.approx(40.0)  # held for the 5-step window
    assert f.loc[155, "jump_excess"] == pytest.approx(0.0)


def test_freeze_features_count_repeats() -> None:
    n = 200
    temp = 30.0 + 0.05 * np.arange(n)
    temp[160:] = temp[160]
    f = build_features(_tel(n, temperature_c=temp), FCFG).set_index("seq")
    assert f.loc[159, "temp_unchanged"] == 0
    assert f.loc[199, "temp_unchanged"] == 39
    assert f.loc[199, "pos_unchanged"] == 0


def test_mode_age_and_one_hots() -> None:
    mode = np.array(["moving"] * 150 + ["returning"] * 50)
    f = build_features(_tel(200, mode=mode), FCFG).set_index("seq")
    assert f.loc[149, "mode_age"] == 149
    assert f.loc[150, "mode_age"] == 0
    assert f.loc[199, "mode_age"] == 49
    assert f.loc[199, "mode_returning"] == 1.0 and f.loc[199, "mode_moving"] == 0.0
    assert f.loc[199, "type_rover"] == 1.0 and f.loc[199, "type_drone"] == 0.0


# ---------------------------------------------------------------- isolation and causality


def test_windows_never_cross_runs_or_assets() -> None:
    a = _tel(200, run_id="r1")
    b = _tel(200, run_id="r2", temperature_c=np.full(200, 99.0))
    c = _tel(200, run_id="r1", asset="drone-01", temperature_c=np.full(200, -5.0))
    f = build_features(pd.concat([a, b, c]), FCFG)
    for (_run, _asset), g in f.groupby(["run_id", "asset_id"]):
        g = g.sort_values("seq")
        assert g["history"].tolist() == list(range(1, 201))
        assert g["temp_slope_l"].iloc[: FCFG.long - 1].isna().all()
    r2 = f[f["run_id"] == "r2"]
    assert (r2["temp_slope_l"].dropna() == 0).all()  # never sees r1's rising temperature


def test_features_are_causal(small_cfg: GenerationConfig) -> None:
    """Features at event i must not change if events after i are removed."""
    run = PlannedRun("r2005", 2005, "validation", "overheating", 0, "linear")
    tel = generate_run(small_cfg, run).telemetry
    full = build_features(tel, FCFG).set_index(["asset_id", "seq"])
    cut = tel[tel["seq"] < 200]
    part = build_features(cut, FCFG).set_index(["asset_id", "seq"])
    pd.testing.assert_frame_equal(
        full.loc[part.index, list(FEATURE_COLUMNS)], part[list(FEATURE_COLUMNS)]
    )


def test_feature_order_and_input_order_independent(small_cfg: GenerationConfig) -> None:
    run = PlannedRun("r1000", 1000, "train", "normal", None, None)
    tel = generate_run(small_cfg, run).telemetry
    a = build_features(tel, FCFG)
    b = build_features(tel.sample(frac=1.0, random_state=0), FCFG)
    pd.testing.assert_frame_equal(a, b)


def test_scorable_requires_history_and_no_gaps() -> None:
    n = 400
    tel = _tel(n)
    tel = tel[~tel["seq"].between(170, 175)]  # 6 lost events -> gap of 7
    f = build_features(tel, FCFG).set_index("seq")
    ok = scorable(f)
    assert not ok.loc[FCFG.min_history - 2]  # 119 events of history
    assert ok.loc[FCFG.min_history - 1]  # exactly min_history events
    assert ok.loc[169]
    assert not ok.loc[176]  # long window now contains a 7-event gap
    assert not ok.loc[176 + FCFG.long - 7]
    assert ok.loc[399]  # gap has left the window


def test_missing_columns_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        build_features(_tel().drop(columns=["temperature_c"]), FCFG)


def test_schema_hash_tracks_config() -> None:
    assert FCFG.schema_hash != replace(FCFG, long=60).schema_hash


# ---------------------------------------------------------------- label isolation


def _imports(path: Path) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
    return mods


@pytest.mark.parametrize("package", ["features", "detectors"])
def test_feature_and_detector_code_never_import_ground_truth(package: str) -> None:
    root = Path(fleet_signal.__file__).parent / package
    files = list(root.glob("*.py"))
    assert files
    for path in files:
        bad = {m for m in _imports(path) if "ground_truth" in m or m.endswith(".faults")}
        assert not bad, f"{path.name} imports {bad}"


def test_split_guard_rejects_foreign_runs(cfg: GenerationConfig) -> None:
    df = pd.DataFrame({"run_id": ["r1000", "r1001"]})
    assert_only_split(df, "train", cfg)
    with pytest.raises(SplitLeakError):
        assert_only_split(pd.DataFrame({"run_id": ["r1000", "r3001"]}), "train", cfg)

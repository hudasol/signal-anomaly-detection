"""Generator: determinism, schema, label isolation and that each fault really happens."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fleet_signal.data import telemetry as telemetry_module
from fleet_signal.data.config import FAULT_TYPES, GenerationConfig
from fleet_signal.data.generator import TELEMETRY_COLUMNS, generate_dataset, generate_run
from fleet_signal.data.ground_truth import load_episodes, load_faults, load_runs
from fleet_signal.data.splits import PlannedRun, allowed_variants
from fleet_signal.data.telemetry import load_telemetry

BLACKBOX_MODES = {"idle", "moving", "returning", "charging"}


def _run(scenario: str, variant: str | None, seed: int = 2001, asset: int | None = 0, split="test"):
    return PlannedRun(
        run_id=f"r{seed}",
        seed=seed,
        split=split,
        scenario=scenario,
        fault_asset_index=None if scenario == "normal" else asset,
        variant=variant,
    )


def _all_fault_cases(cfg: GenerationConfig) -> list[tuple[str, str]]:
    return [(f, v) for f in FAULT_TYPES for v in allowed_variants(cfg, f, include_test_only=True)]


# ---------------------------------------------------------------- determinism


def test_same_seed_gives_identical_run(small_cfg: GenerationConfig) -> None:
    run = _run("overheating", "linear")
    a, b = generate_run(small_cfg, run), generate_run(small_cfg, run)
    pd.testing.assert_frame_equal(a.telemetry, b.telemetry)
    assert a.fault == b.fault
    assert a.episodes == b.episodes


def test_different_seeds_differ(small_cfg: GenerationConfig) -> None:
    a = generate_run(small_cfg, _run("normal", None, seed=1000)).telemetry
    b = generate_run(small_cfg, _run("normal", None, seed=1001)).telemetry
    assert not np.allclose(a["temperature_c"].to_numpy()[:100], b["temperature_c"].to_numpy()[:100])


# ---------------------------------------------------------------- schema


def test_schema_and_bounds_match_blackbox_contract(small_cfg: GenerationConfig) -> None:
    tel = generate_run(small_cfg, _run("normal", None, seed=1000)).telemetry
    assert tuple(tel.columns) == TELEMETRY_COLUMNS
    assert set(tel["mode"]) <= BLACKBOX_MODES
    assert set(tel["asset_type"]) == {"drone", "rover", "quadruped"}
    assert tel["battery_pct"].between(0, 100).all()
    assert tel["link_quality_pct"].between(0, 100).all()
    assert ((tel["heading_deg"] >= 0) & (tel["heading_deg"] < 360)).all()
    assert (tel["speed_mps"] >= 0).all()
    assert tel[["x_m", "y_m", "z_m", "temperature_c"]].notna().all().all()
    for _, g in tel.groupby("asset_id"):
        assert g["seq"].is_monotonic_increasing
        assert g["timestamp_utc"].is_monotonic_increasing


def test_timestamps_follow_seq(small_cfg: GenerationConfig) -> None:
    tel = generate_run(small_cfg, _run("normal", None, seed=1000)).telemetry
    t0 = pd.Timestamp(small_cfg.data["start_epoch_utc"]) + pd.Timedelta(
        seconds=1000 * small_cfg.data["run_spacing_s"]
    )
    expected = t0 + pd.to_timedelta(tel["seq"], unit="s")
    assert (tel["timestamp_utc"] == expected).all()


def test_telemetry_has_no_label_columns(small_cfg: GenerationConfig) -> None:
    tel = generate_run(small_cfg, _run("battery_drain", "step")).telemetry
    banned = ("fault", "label", "scenario", "variant", "anomal", "episode", "split")
    for col in tel.columns:
        assert not any(word in col.lower() for word in banned), col


def test_normal_runs_contain_difficult_normal_behaviour(small_cfg: GenerationConfig) -> None:
    cfg = replace(small_cfg, data={**small_cfg.data, "run_duration_s": 1200})
    kinds: set[str] = set()
    modes: set[str] = set()
    for seed in range(1000, 1006):
        result = generate_run(cfg, _run("normal", None, seed=seed))
        kinds |= {e["episode"] for e in result.episodes}
        modes |= set(result.telemetry["mode"])
    assert {"charging", "returning", "hard_manoeuvre", "noisy_link"} <= kinds
    assert modes == BLACKBOX_MODES


# ---------------------------------------------------------------- fault isolation


@pytest.mark.parametrize("fault_type", FAULT_TYPES)
def test_fault_changes_nothing_before_onset_or_on_other_assets(
    small_cfg: GenerationConfig, fault_type: str
) -> None:
    """Counterfactual: a fault run equals the same seed's normal run until the fault starts."""
    variant = allowed_variants(small_cfg, fault_type, include_test_only=False)[0]
    faulty = generate_run(small_cfg, _run(fault_type, variant, asset=1))
    clean = generate_run(small_cfg, _run("normal", None)).telemetry
    tel = faulty.telemetry
    fault = faulty.fault
    assert fault is not None

    other = tel["asset_id"] != fault["asset_id"]
    pd.testing.assert_frame_equal(
        tel[other].reset_index(drop=True),
        clean[clean["asset_id"] != fault["asset_id"]].reset_index(drop=True),
    )
    before = (tel["asset_id"] == fault["asset_id"]) & (tel["seq"] < fault["fault_start_seq"])
    clean_before = (clean["asset_id"] == fault["asset_id"]) & (
        clean["seq"] < fault["fault_start_seq"]
    )
    pd.testing.assert_frame_equal(
        tel[before].reset_index(drop=True), clean[clean_before].reset_index(drop=True)
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"{c[0]}-{c[1]}")
        for c in [
            ("overheating", "linear"),
            ("overheating", "runaway"),
            ("battery_drain", "step"),
            ("battery_drain", "accelerating"),
            ("link_degradation", "decline"),
            ("link_degradation", "oscillation"),
            ("link_degradation", "intermittent"),
            ("sensor_freeze", "temperature_c"),
            ("sensor_freeze", "battery_pct"),
            ("sensor_freeze", "link_quality_pct"),
            ("sensor_freeze", "speed_mps"),
            ("sensor_freeze", "position"),
            ("sensor_freeze", "multi"),
            ("motion_anomaly", "jump"),
            ("motion_anomaly", "speed_mismatch"),
            ("motion_anomaly", "drift"),
        ]
    ],
)
def test_each_fault_variant_is_observable(
    small_cfg: GenerationConfig, case: tuple[str, str]
) -> None:
    """Each injected fault must produce the behaviour its label claims, vs the clean twin."""
    fault_type, variant = case
    faulty = generate_run(small_cfg, _run(fault_type, variant, asset=0))
    f = faulty.fault
    assert f is not None and f["fault_type"] == fault_type and f["variant"] == variant
    assert 0 <= f["fault_start_seq"] < f["fault_end_seq"] <= small_cfg.n_ticks

    def window(df: pd.DataFrame) -> pd.DataFrame:
        on = (df["asset_id"] == f["asset_id"]) & df["seq"].between(
            f["fault_start_seq"] + 1, f["fault_end_seq"] - 1
        )
        return df[on].set_index("seq")

    w = window(faulty.telemetry)
    c = window(generate_run(small_cfg, _run("normal", None)).telemetry)
    common = w.index.intersection(c.index)
    w, c = w.loc[common], c.loc[common]
    params = json.loads(f["params_json"])

    if fault_type == "overheating":
        assert (w["temperature_c"] - c["temperature_c"]).iloc[-1] > 1.0
    elif fault_type == "battery_drain":
        assert (c["battery_pct"] - w["battery_pct"]).iloc[-1] > 0.5
    elif fault_type == "link_degradation":
        assert (c["link_quality_pct"] - w["link_quality_pct"]).mean() > 3.0
    elif fault_type == "sensor_freeze":
        fields = params["fields"]
        assert (w[fields].nunique() == 1).all(), "frozen fields must not change"
        assert (c[fields].nunique() > 1).any(), "clean twin should have changed"
    elif variant == "jump":
        offset = np.hypot(w["x_m"] - c["x_m"], w["y_m"] - c["y_m"])
        assert np.allclose(offset, params["jump_m"], atol=0.05)
    elif variant == "drift":
        offset = np.hypot(w["x_m"] - c["x_m"], w["y_m"] - c["y_m"])
        assert offset.is_monotonic_increasing and offset.iloc[-1] > 10
    elif variant == "speed_mismatch":
        moving = c["speed_mps"] > 0.5
        assert moving.any()
        ratio = (w.loc[moving, "speed_mps"] / c.loc[moving, "speed_mps"]).median()
        assert abs(ratio - params["factor"]) < 0.05


def test_sensor_faults_start_while_moving_when_possible(small_cfg: GenerationConfig) -> None:
    for variant in ("speed_mps", "temperature_c"):
        result = generate_run(small_cfg, _run("sensor_freeze", variant, asset=0))
        start = result.fault["fault_start_seq"]  # type: ignore[index]
        tel = result.telemetry
        row = tel[(tel["asset_id"] == result.fault["asset_id"]) & (tel["seq"] <= start)].iloc[-1]  # type: ignore[index]
        assert row["mode"] in ("moving", "returning")


# ---------------------------------------------------------------- dataset files


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory, small_cfg: GenerationConfig) -> Path:
    out = tmp_path_factory.mktemp("data")
    runs = [
        _run("normal", None, seed=1000, split="train"),
        _run("overheating", "linear", seed=2000, split="validation"),
        _run("sensor_freeze", "multi", seed=3000, split="test", asset=2),
    ]
    generate_dataset(small_cfg, out, runs=runs)
    return out


def test_dataset_layout_separates_ground_truth(
    tiny_dataset: Path, small_cfg: GenerationConfig
) -> None:
    version_dir = tiny_dataset / small_cfg.version
    assert (tiny_dataset / "CURRENT").read_text().strip() == small_cfg.version
    assert (version_dir / "telemetry.parquet").exists()
    for name in ("runs.parquet", "faults.parquet", "episodes.parquet"):
        assert (version_dir / "ground_truth" / name).exists()
    manifest = json.loads((version_dir / "manifest.json").read_text())
    assert manifest["data_version"] == small_cfg.version
    assert manifest["telemetry_columns"] == list(TELEMETRY_COLUMNS)


def test_loaders_round_trip(tiny_dataset: Path) -> None:
    tel = load_telemetry(data_dir=tiny_dataset)
    assert set(tel["run_id"]) == {"r1000", "r2000", "r3000"}
    assert tuple(tel.columns) == TELEMETRY_COLUMNS
    only = load_telemetry(data_dir=tiny_dataset, run_ids=["r2000"])
    assert set(only["run_id"]) == {"r2000"}
    runs = load_runs(data_dir=tiny_dataset)
    assert dict(zip(runs["run_id"], runs["split"], strict=True)) == {
        "r1000": "train",
        "r2000": "validation",
        "r3000": "test",
    }
    faults = load_faults(data_dir=tiny_dataset)
    assert set(faults["run_id"]) == {"r2000", "r3000"}
    assert len(load_episodes(data_dir=tiny_dataset)) > 0


def test_missing_dataset_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="signal-data generate"):
        load_telemetry(data_dir=tmp_path)


def test_telemetry_loader_cannot_reach_ground_truth() -> None:
    """The module feature code reads data through must not import ground truth."""
    tree = ast.parse(Path(telemetry_module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    assert not any("ground_truth" in m for m in imported)


def test_version_changes_with_config(small_cfg: GenerationConfig, cfg: GenerationConfig) -> None:
    assert small_cfg.version != cfg.version

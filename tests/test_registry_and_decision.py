"""Artifacts fail safe; the ship rule matches PLAN §7.3; paired latency comparison."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.eval.bootstrap import paired_latency
from fleet_signal.eval.decision import best_baseline, ship_decision
from fleet_signal.features.build import FeatureConfig
from fleet_signal.registry import (
    ArtifactError,
    EvaluatedModelError,
    ModelArtifact,
    load_artifact,
    read_registry,
    register,
    save_artifact,
    sha256_file,
    write_registry,
)

FCFG = FeatureConfig.load()


def _art(threshold: float = 0.5) -> ModelArtifact:
    return ModelArtifact(
        detector=RuleDetector(),
        detector_name="rule",
        threshold=threshold,
        incident_params={"open_n": 2, "close_m": 5, "cooldown_c": 30},
        feature_config=FCFG.as_dict(),
        feature_schema_hash=FCFG.schema_hash,
        data_version="v-test",
        train_run_ids=["r1000"],
        validation_summary={"recall": 0.9},
        git_sha=None,
    )


def test_artifact_round_trip_and_version(tmp_path: Path) -> None:
    art = _art()
    path = save_artifact(art, tmp_path)
    loaded = load_artifact(path)
    assert loaded.model_version == art.model_version
    assert loaded.threshold == 0.5
    assert (path.parent / "metadata.json").exists()
    # version is content-derived: a different threshold is a different model
    assert _art(0.6).model_version != art.model_version


def test_missing_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="not found"):
        load_artifact(tmp_path / "nope.joblib")


def test_corrupt_artifact_raises(tmp_path: Path) -> None:
    bad = tmp_path / "model.joblib"
    bad.write_bytes(b"not a pickle")
    with pytest.raises(ArtifactError, match="could not be loaded"):
        load_artifact(bad)


def test_feature_schema_mismatch_raises(tmp_path: Path) -> None:
    path = save_artifact(_art(), tmp_path)
    other = replace(FCFG, long=60)
    with pytest.raises(ArtifactError, match="schema mismatch"):
        load_artifact(path, other)


def test_registry_records_sha_and_threshold(tmp_path: Path) -> None:
    art = _art()
    path = save_artifact(art, tmp_path)
    reg_path = tmp_path / "registry.json"
    register(art, path, reg_path)
    entry = read_registry(reg_path)["models"]["rule"]
    assert entry["model_version"] == art.model_version
    assert entry["threshold"] == 0.5
    assert len(entry["artifact_sha256"]) == 64


# ---------------------------------------------------------------- ship rule


def _ci(diff: float, lo: float, hi: float) -> dict[str, float]:
    return {"diff": diff, "ci_low": lo, "ci_high": hi}


NO_LAT = {
    "n_paired": 0,
    "median_diff": float("nan"),
    "ci_low": float("nan"),
    "ci_high": float("nan"),
}


def test_ml_ships_on_significant_recall_gain() -> None:
    d = ship_decision("lof", "rule", _ci(0.1, 0.02, 0.2), NO_LAT)
    assert d["ship"] == "lof"


def test_ml_ships_on_significant_latency_gain_without_recall_loss() -> None:
    lat = {"n_paired": 20, "median_diff": -40.0, "ci_low": -60.0, "ci_high": -20.0}
    assert ship_decision("lof", "rule", _ci(0.0, -0.1, 0.1), lat)["ship"] == "lof"


def test_faster_but_less_recall_does_not_ship_ml() -> None:
    lat = {"n_paired": 20, "median_diff": -40.0, "ci_low": -60.0, "ci_high": -20.0}
    d = ship_decision("lof", "rule", _ci(-0.05, -0.15, 0.05), lat)
    assert d["ship"] == "rule"
    assert "loses recall" in d["reason"]


def test_inconclusive_ships_baseline() -> None:
    assert ship_decision("lof", "rule", _ci(0.03, -0.05, 0.1), NO_LAT)["ship"] == "rule"


def test_best_baseline_by_recall_then_precision() -> None:
    s = {"rule": {"recall": 0.9, "precision": 0.8}, "stats": {"recall": 0.9, "precision": 0.9}}
    assert best_baseline(s, ("rule", "stats")) == "stats"


def test_paired_latency_uses_only_faults_both_detected() -> None:
    def pf(lat: list[float | None]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "run_id": ["r1", "r2", "r3"],
                "asset_id": "a",
                "fault_type": "overheating",
                "detected": [x is not None for x in lat],
                "latency_events": lat,
            }
        )

    out = paired_latency(pf([10, 20, None]), pf([30, 50, 5]), ("overheating",), 200, 0)
    assert out["n_paired"] == 2
    assert out["median_diff"] == pytest.approx(-25.0)


def test_artifact_bytes_are_deterministic(tmp_path: Path) -> None:
    a = save_artifact(_art(), tmp_path / "a")
    b = save_artifact(_art(), tmp_path / "b")
    assert sha256_file(a) == sha256_file(b)
    meta = json.loads((a.parent / "metadata.json").read_text())
    assert meta["created_utc"] and len(meta["detector_code_hash"]) == 10


def test_evaluated_model_cannot_be_overwritten(tmp_path: Path) -> None:
    art = _art()
    path = save_artifact(art, tmp_path)
    reg_path = tmp_path / "registry.json"
    register(art, path, reg_path)
    reg = read_registry(reg_path)
    reg["models"]["rule"]["official_test_result"] = "results/official/test_rule_x.json"
    write_registry(reg, reg_path)
    with pytest.raises(EvaluatedModelError):
        register(_art(0.7), save_artifact(_art(0.7), tmp_path), reg_path)

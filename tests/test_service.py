"""Inference contract, fail-safe states, streaming == batch, API, and end-to-end replay."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fleet_signal.data.config import GenerationConfig
from fleet_signal.data.generator import generate_run
from fleet_signal.data.splits import PlannedRun
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.eval.protocol import EvalConfig, ScoredSet
from fleet_signal.features.build import FeatureConfig, build_features
from fleet_signal.incidents.grouping import IncidentParams, group_alerts
from fleet_signal.incidents.tracker import IncidentTracker
from fleet_signal.registry import ModelArtifact, save_artifact
from fleet_signal.service.app import create_app
from fleet_signal.service.replay import replay_asset
from fleet_signal.service.scorer import Scorer

FCFG = FeatureConfig.load()
PARAMS = {"open_n": 2, "close_m": 5, "cooldown_c": 30}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    art = ModelArtifact(
        detector=RuleDetector(),
        detector_name="rule",
        threshold=0.0,
        incident_params=PARAMS,
        feature_config=FCFG.as_dict(),
        feature_schema_hash=FCFG.schema_hash,
        data_version="v-test",
        train_run_ids=[],
        validation_summary={},
        git_sha=None,
    )
    return save_artifact(art, tmp_path_factory.mktemp("models"))


@pytest.fixture(scope="module")
def jump_run(small_cfg: GenerationConfig):
    cfg = replace(small_cfg, data={**small_cfg.data, "run_duration_s": 600})
    return generate_run(cfg, PlannedRun("r2003", 2003, "validation", "motion_anomaly", 0, "jump"))


def _asset(result, asset: str = "drone-01") -> pd.DataFrame:
    tel = result.telemetry
    return tel[tel["asset_id"] == asset].reset_index(drop=True)


# ---------------------------------------------------------------- fail-safe states


def test_missing_model_is_unavailable_not_normal(tmp_path: Path, jump_run) -> None:
    scorer = Scorer(tmp_path / "deleted.joblib")
    r = scorer.score_events(_asset(jump_run).head(300))
    assert r.status == "unavailable"
    assert r.decision is None and r.score is None
    assert "not found" in (r.reason or "")


def test_corrupt_model_is_unavailable(tmp_path: Path, jump_run) -> None:
    bad = tmp_path / "model.joblib"
    bad.write_bytes(b"\x00garbage")
    assert Scorer(bad).score_events(_asset(jump_run).head(300)).status == "unavailable"


def test_short_history_is_insufficient_data(artifact: Path, jump_run) -> None:
    r = Scorer(artifact).score_events(_asset(jump_run).head(FCFG.min_history - 1))
    assert r.status == "insufficient_data" and r.decision is None
    assert r.history == FCFG.min_history - 1


def test_gap_in_window_is_degraded(artifact: Path, jump_run) -> None:
    ev = _asset(jump_run).head(300)
    ev = ev[~ev["seq"].between(250, 260)]
    r = Scorer(artifact).score_events(ev)
    assert r.status == "degraded" and r.decision is None and "gap" in (r.reason or "")


def test_non_finite_input_is_degraded(artifact: Path, jump_run) -> None:
    ev = _asset(jump_run).head(300).copy()
    ev.loc[ev.index[-1], "temperature_c"] = np.nan
    assert Scorer(artifact).score_events(ev).status == "degraded"


def test_mixed_assets_rejected(artifact: Path, jump_run) -> None:
    mixed = pd.concat(
        [_asset(jump_run, "drone-01").head(200), _asset(jump_run, "rover-01").head(200)]
    )
    r = Scorer(artifact).score_events(mixed)
    assert r.status == "degraded" and "one asset" in (r.reason or "")


def test_ok_response_contract(artifact: Path, jump_run) -> None:
    r = Scorer(artifact).score_events(_asset(jump_run).head(300))
    assert r.status == "ok"
    assert r.decision in ("normal", "anomalous")
    assert isinstance(r.score, float) and r.threshold == 0.0
    assert r.model_version and r.model_version.startswith("rule-")
    assert 1 <= len(r.evidence) <= 3
    assert all({"signal", "contribution"} <= set(e) for e in r.evidence)
    assert all(e["contribution"] >= 0 for e in r.evidence[1:])


def test_blackbox_nested_position_events_accepted(artifact: Path, jump_run) -> None:
    flat = _asset(jump_run).head(300)
    nested = []
    for row in flat.to_dict("records"):
        row["position"] = {"x_m": row.pop("x_m"), "y_m": row.pop("y_m"), "z_m": row.pop("z_m")}
        row.pop("run_id")
        nested.append(row)
    s = Scorer(artifact)
    assert s.score_events(nested).score == pytest.approx(s.score_events(flat).score)


# ---------------------------------------------------------------- streaming == batch


def test_tracker_matches_batch_grouping_on_random_sequences() -> None:
    rng = np.random.default_rng(0)
    for trial in range(300):
        p = IncidentParams(
            int(rng.integers(1, 4)), int(rng.integers(1, 8)), int(rng.integers(0, 12))
        )
        alert = rng.random(int(rng.integers(5, 120))) < rng.uniform(0.05, 0.6)
        batch = group_alerts(alert, p)
        tr = IncidentTracker(p)
        for i, a in enumerate(alert):
            tr.update(i, bool(a))
        tr.finish(len(alert) - 1)
        got = [(t.start_seq, t.open_seqs, t.last_alert_seq, t.close_seq) for t in tr.incidents]
        want = [(b.start, b.opens, b.last_alert, b.close) for b in batch]
        assert got == want, (trial, p, alert.astype(int).tolist())


def test_replay_strict_window_path_equals_batch_path(artifact: Path, jump_run) -> None:
    scorer = Scorer(artifact)
    ev = _asset(jump_run)
    fast, _, inc_fast = replay_asset(scorer, ev, strict=False)
    strict, _, inc_strict = replay_asset(scorer, ev, strict=True)
    assert [r.status for r in fast] == [r.status for r in strict]
    a = np.array([r.score if r.score is not None else np.nan for r in fast])
    b = np.array([r.score if r.score is not None else np.nan for r in strict])
    assert np.allclose(a, b, equal_nan=True)
    assert inc_fast == inc_strict


def test_replay_incidents_match_evaluation(artifact: Path, jump_run) -> None:
    """The live path opens incidents exactly where the evaluation harness says it does."""
    scorer = Scorer(artifact)
    ev = _asset(jump_run)
    _, _, incidents = replay_asset(scorer, ev)
    feats = build_features(ev)
    scored = feats[["run_id", "asset_id", "seq"]].assign(
        score=scorer.artifact.detector.score(feats)
    )
    ss = ScoredSet(scored, pd.DataFrame([jump_run.fault]), ["r2003"], EvalConfig.load(), 1)
    res = ss.evaluate(0.0, IncidentParams(**PARAMS))
    assert [i["open_seqs"] for i in incidents] == res.incidents["open_seqs"].tolist()
    assert res.summary["recall"] == 1.0


def test_replay_with_missing_model_never_alerts(tmp_path: Path, jump_run) -> None:
    results, lifecycle, incidents = replay_asset(Scorer(tmp_path / "nope.joblib"), _asset(jump_run))
    assert {r.status for r in results} == {"unavailable"}
    assert lifecycle == [] and incidents == []


# ---------------------------------------------------------------- HTTP API


def _payload(df: pd.DataFrame) -> dict:
    return {"events": df.drop(columns=["timestamp_utc"]).to_dict("records")}


def test_api_score_and_health(artifact: Path, jump_run) -> None:
    client = TestClient(create_app(artifact))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model").json()["threshold"] == 0.0
    r = client.post("/score", json=_payload(_asset(jump_run).head(300)))
    assert r.status_code == 200 and r.json()["status"] == "ok"
    short = client.post("/score", json=_payload(_asset(jump_run).head(10)))
    assert short.status_code == 200 and short.json()["status"] == "insufficient_data"


def test_api_without_model_returns_503_unavailable(tmp_path: Path, jump_run) -> None:
    client = TestClient(create_app(tmp_path / "renamed.joblib"))
    assert client.get("/health").status_code == 503
    r = client.post("/score", json=_payload(_asset(jump_run).head(300)))
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable" and r.json()["decision"] is None


def test_api_rejects_empty_request(artifact: Path) -> None:
    client = TestClient(create_app(artifact))
    assert client.post("/score", json={"events": []}).status_code == 422

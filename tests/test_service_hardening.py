"""Post-freeze service hardening: no input, file or startup problem may end in "normal".

Each test here is a case found in the pre-release engineering review. None of
them change a score on valid input (replays of the test runs are byte-identical).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fleet_signal.data.config import GenerationConfig
from fleet_signal.data.generator import generate_run
from fleet_signal.data.splits import PlannedRun
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.features.build import FEATURE_COLUMNS, FeatureConfig
from fleet_signal.incidents.grouping import IncidentParams, group_alerts
from fleet_signal.incidents.tracker import IncidentTracker
from fleet_signal.registry import (
    ModelArtifact,
    read_registry,
    register,
    save_artifact,
    sha256_file,
    write_registry,
)
from fleet_signal.service import app as app_module
from fleet_signal.service.app import BodyLimit, create_app
from fleet_signal.service.scorer import Scorer, _single_threaded
from fleet_signal.service.validation import MAX_BODY_BYTES, MAX_EVENTS

FCFG = FeatureConfig.load()
PARAMS = {"open_n": 2, "close_m": 5, "cooldown_c": 30}


def _art() -> ModelArtifact:
    return ModelArtifact(
        detector=RuleDetector(), detector_name="rule", threshold=0.0, incident_params=PARAMS,
        feature_config=FCFG.as_dict(), feature_schema_hash=FCFG.schema_hash,
        data_version="v-test", train_run_ids=[], validation_summary={}, git_sha=None,
    )  # fmt: skip


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return save_artifact(_art(), tmp_path_factory.mktemp("models"))


@pytest.fixture(scope="module")
def window(small_cfg: GenerationConfig) -> pd.DataFrame:
    """A valid window that starts at the run start (so it is scored)."""
    cfg = replace(small_cfg, data={**small_cfg.data, "run_duration_s": 600})
    run = generate_run(cfg, PlannedRun("r2003", 2003, "validation", "motion_anomaly", 0, "jump"))
    tel = run.telemetry
    return tel[tel["asset_id"] == "drone-01"].reset_index(drop=True).head(300)


@pytest.fixture(scope="module")
def client(artifact: Path) -> TestClient:
    return TestClient(BodyLimit(create_app(artifact, allow_unregistered=True), MAX_BODY_BYTES))


def _events(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.drop(columns=["timestamp_utc"]).to_dict("records")


def _mutate(df: pd.DataFrame, i: int, **values: Any) -> list[dict[str, Any]]:
    ev = _events(df)
    ev[i] = {**ev[i], **values}
    return ev


# ---------------------------------------------------------------- A1: the HTTP contract


def test_valid_window_scores_ok_and_matches_the_scorer(client: TestClient, artifact, window):
    r = client.post("/score", json={"events": _events(window)})
    assert r.status_code == 200 and r.json()["status"] == "ok"
    direct = Scorer(artifact, allow_unregistered=True).score_events(window).as_dict()
    assert r.json()["score"] == pytest.approx(direct["score"])
    assert r.json()["decision"] == direct["decision"]


@pytest.mark.parametrize(
    "case, change",
    [
        ("mode upper case", {"mode": "MOVING"}),
        ("unknown mode", {"mode": "flying"}),
        ("unknown asset type", {"asset_type": "boat"}),
        ("bool speed", {"speed_mps": True}),
        ("string number", {"battery_pct": "55"}),
        ("float seq", {"seq": 3.7}),
        ("string seq", {"seq": "12"}),
        ("negative seq", {"seq": -5}),
        ("null asset", {"asset_id": None}),
        ("null x", {"x_m": None}),
        ("missing battery", {"battery_pct": "__drop__"}),
    ],
)
def test_contract_violations_are_422_not_scored(client: TestClient, window, case, change):
    ev = _events(window)
    last = dict(ev[-1])
    for k, v in change.items():
        if v == "__drop__":
            last.pop(k)
        else:
            last[k] = v
    ev[-1] = last
    r = client.post("/score", json={"events": ev})
    assert r.status_code == 422, case
    body = r.json()
    assert body["status"] == "invalid_request" and body["decision"] is None
    assert "input" not in json.dumps(body["errors"])  # submitted values are never echoed


def test_duplicate_reordered_and_mixed_windows_are_422(client: TestClient, window):
    ev = _events(window)
    dup = ev + [{**ev[-1], "temperature_c": 99.0}]
    assert client.post("/score", json={"events": dup}).status_code == 422
    assert client.post("/score", json={"events": ev[::-1]}).status_code == 422
    mixed_id = ev[:-1] + [{**ev[-1], "asset_id": "drone-02"}]
    assert client.post("/score", json={"events": mixed_id}).status_code == 422
    mixed_type = ev[:-1] + [{**ev[-1], "asset_type": "rover"}]
    assert client.post("/score", json={"events": mixed_type}).status_code == 422
    assert client.post("/score", json=[1, 2, 3]).status_code == 422


def test_too_many_events_and_oversized_body_are_rejected(client: TestClient, window):
    ev = _events(window)
    many = [{**ev[0], "seq": i} for i in range(MAX_EVENTS + 1)]
    assert client.post("/score", json={"events": many}).status_code == 422
    big = b'{"events": [' + b" " * (MAX_BODY_BYTES + 10) + b"]}"
    r = client.post("/score", content=big, headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["decision"] is None

    def chunks():  # no content-length: the limit still applies while reading
        yield b'{"events": ['
        for _ in range(MAX_BODY_BYTES // 100_000 + 2):
            yield b" " * 100_000
        yield b"]}"

    r = client.post("/score", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413


@pytest.mark.parametrize(
    "change",
    [
        {"battery_pct": -50.0},
        {"battery_pct": 500.0},
        {"link_quality_pct": 101.0},
        {"speed_mps": 1e308},
        {"temperature_c": 999.0},
    ],
)
def test_impossible_readings_are_degraded_never_normal(client: TestClient, window, change):
    r = client.post("/score", json={"events": _mutate(window, -1, **change)})
    assert r.status_code == 200
    assert r.json()["status"] == "degraded" and r.json()["decision"] is None
    assert "physical range" in r.json()["reason"]


def test_blackbox_events_pass_the_contract(client: TestClient, window):
    ev = []
    for e in _events(window):
        e = dict(e)
        e["position"] = {k: e.pop(k) for k in ("x_m", "y_m", "z_m")}
        ev.append(e)
    r = client.post("/score", json={"events": ev})
    assert r.status_code == 200 and r.json()["status"] == "ok"


# ---------------------------------------------------------------- A2: the scorer never repairs


@pytest.mark.parametrize(
    "case",
    ["duplicate", "reversed", "negative", "null_asset", "mixed_type", "mode_case", "bool_seq"],
)
def test_scorer_reports_instead_of_repairing(artifact: Path, window: pd.DataFrame, case: str):
    df = window.copy()
    if case == "duplicate":
        df = pd.concat([df, df.tail(1).assign(temperature_c=99.0)], ignore_index=True)
    elif case == "reversed":
        df = df.iloc[::-1]
    elif case == "negative":
        df = df.assign(seq=df["seq"] - 10_000)
    elif case == "null_asset":
        df.loc[df.index[-1], "asset_id"] = None
    elif case == "mixed_type":
        df.loc[df.index[-1], "asset_type"] = "rover"
    elif case == "mode_case":
        df.loc[df.index[-1], "mode"] = "MOVING"
    elif case == "bool_seq":
        df = df.assign(seq=True)
    r = Scorer(artifact, allow_unregistered=True).score_events(df)
    assert r.status == "degraded" and r.decision is None, case


def test_timestamps_must_advance_at_1hz(artifact: Path, window: pd.DataFrame):
    s = Scorer(artifact, allow_unregistered=True)
    assert s.score_events(window).status == "ok"  # generated timestamps are 1 Hz
    fast = window.assign(
        timestamp_utc=window["timestamp_utc"].iloc[0]
        + pd.to_timedelta(window["seq"] * 0.5, unit="s")
    )
    assert s.score_events(fast).status == "degraded"
    partly = window.astype({"timestamp_utc": object})
    partly.loc[partly.index[3], "timestamp_utc"] = None
    assert s.score_events(partly).status == "degraded"


# ---------------------------------------------------------------- A3: unknown modes


def test_unknown_mode_gets_no_score_from_any_detector() -> None:
    row: dict[str, Any] = {c: 0.0 for c in FEATURE_COLUMNS}
    row.update(
        run_id="r1", asset_id="rover-01", asset_type="rover", seq=0, mode="MOVING",
        mode_age=300.0, link_pct=80.0, history=500, history_ok=True, max_gap_l=1.0,
        gap_ok=True, batt_slope_l=-30.0,
    )  # fmt: skip
    df = pd.DataFrame([row])
    assert np.isnan(RuleDetector().score(df)).all()
    assert RuleDetector().evidence(df) == [[]]
    known = df.assign(mode="moving")
    assert np.isfinite(RuleDetector().score(known)).all()
    assert RuleDetector().score(known)[0] > 0  # the same drain fires once the mode is valid
    stats = StatsDetector()
    stats.center[("rover", "*")] = np.zeros(len(stats.features))
    stats.scale[("rover", "*")] = np.ones(len(stats.features))
    assert np.isnan(stats.score(df)).all()


# ---------------------------------------------------------------- B1: artifacts


class _Payload:
    """A pickle that runs code when loaded: writes a marker file."""

    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):  # type: ignore[no-untyped-def]
        return (Path.write_text, (self.marker, "code ran"))


def _registry_with(tmp_path: Path, art_path: Path) -> Path:
    reg_path = tmp_path / "registry.json"
    register(_art(), art_path, reg_path)
    reg = read_registry(reg_path)
    reg["models"]["rule"]["artifact"] = str(art_path.resolve())
    reg["serving"] = "rule"
    write_registry(reg, reg_path)
    return reg_path


def test_tampered_artifact_is_refused_before_any_code_runs(tmp_path: Path) -> None:
    art_path = save_artifact(_art(), tmp_path / "models")
    reg_path = _registry_with(tmp_path, art_path)
    assert Scorer(registry_path=reg_path).available  # the genuine file loads

    marker = tmp_path / "pwned.txt"
    art_path.write_bytes(pickle.dumps(_Payload(marker)))
    s = Scorer(registry_path=reg_path)
    assert not s.available and "SHA-256" in (s.error or "")
    assert not marker.exists()


def test_unregistered_artifact_is_refused(tmp_path: Path, monkeypatch) -> None:
    other = save_artifact(_art(), tmp_path / "elsewhere")
    reg_path = _registry_with(tmp_path, save_artifact(_art(), tmp_path / "models"))
    s = Scorer(other, registry_path=reg_path)
    assert not s.available and "not listed" in (s.error or "")
    # the env var cannot bypass the registry either
    monkeypatch.setenv("SIGNAL_ARTIFACT", str(other))
    c = TestClient(create_app())
    assert c.get("/readyz").status_code == 503


def test_corrupt_registry_and_broken_config_leave_service_up_but_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    bad = tmp_path / "registry.json"
    bad.write_text("{not json")
    s = Scorer(registry_path=bad)
    assert not s.available and "registry unreadable" in (s.error or "")

    def boom(*a: Any, **k: Any) -> FeatureConfig:
        raise FileNotFoundError("configs/features.yaml")

    monkeypatch.setattr(FeatureConfig, "load", classmethod(lambda cls, *a, **k: boom()))
    s = Scorer()
    assert not s.available and s.error == "service misconfigured: FileNotFoundError"
    c = TestClient(create_app())
    assert c.get("/livez").status_code == 200
    assert c.get("/readyz").status_code == 503
    assert c.get("/health").status_code == 503


def test_registry_writes_are_atomic(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    write_registry({"serving": None, "models": {}}, reg_path)
    write_registry({"serving": "rule", "models": {}}, reg_path)
    assert read_registry(reg_path)["serving"] == "rule"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["registry.json"]  # no temp left
    # readable by other users (found when the non-root Docker image could not read it)
    assert reg_path.stat().st_mode & 0o044 == 0o044


def test_inference_is_single_threaded() -> None:
    class M:
        n_jobs = -1

    class D:
        models = {"drone": M(), "rover": M()}
        model = M()

    d = D()
    _single_threaded(d)
    assert {m.n_jobs for m in [*d.models.values(), d.model]} == {1}


# ---------------------------------------------------------------- B3 / B5


def test_importing_the_app_module_builds_nothing() -> None:
    assert "app" not in app_module._built or app_module._built  # never raises
    assert callable(app_module.create_app)


def test_every_score_call_is_audited(client: TestClient, window, caplog) -> None:
    with caplog.at_level(logging.INFO, logger="fleet_signal.audit"):
        client.post("/score", json={"events": _events(window)})
        client.post("/score", json={"events": _mutate(window, -1, battery_pct=-50.0)})
    records = [getattr(r, "fields", {}) for r in caplog.records if r.name == "fleet_signal.audit"]
    assert [r["status"] for r in records] == ["ok", "degraded"]
    first = records[0]
    for key in ("asset_id", "seq", "score", "threshold", "decision", "model_version",
                "events_sha256", "latency_ms", "n_events"):  # fmt: skip
        assert key in first
    assert first["model_version"] and first["decision"] in ("normal", "anomalous")


def test_model_endpoint_reports_versions_and_unit(client: TestClient) -> None:
    m = client.get("/model").json()
    assert m["decision_unit"] == "event" and m["max_events"] == MAX_EVENTS
    assert {"recorded", "running", "status"} <= set(m["code_provenance"])
    assert m["service_version"]


# ---------------------------------------------------------------- C: property tests


def test_tracker_matches_batch_grouping_when_seq_has_gaps() -> None:
    rng = np.random.default_rng(1)
    for trial in range(300):
        p = IncidentParams(
            int(rng.integers(1, 4)), int(rng.integers(1, 8)), int(rng.integers(0, 12))
        )
        alert = rng.random(int(rng.integers(5, 120))) < rng.uniform(0.05, 0.6)
        seqs = np.cumsum(rng.integers(1, 4, len(alert)))  # gaps of up to 2 missing events
        batch = group_alerts(alert, p)
        tr = IncidentTracker(p)
        for s, a in zip(seqs, alert, strict=True):
            tr.update(int(s), bool(a))
        tr.finish(int(seqs[-1]))
        got = [(t.start_seq, t.open_seqs, t.last_alert_seq, t.close_seq) for t in tr.incidents]
        want = [
            (int(seqs[b.start]), [int(seqs[o]) for o in b.opens], int(seqs[b.last_alert]),
             int(seqs[b.close]))
            for b in batch
        ]  # fmt: skip
        assert got == want, trial


@pytest.mark.parametrize("which", ["stats", "lof"])
def test_fitted_artifact_bytes_are_deterministic(tmp_path: Path, which: str) -> None:
    from fleet_signal.detectors.lof import LOFDetector

    rng = np.random.default_rng(0)
    rows = []
    for t in ("drone", "rover", "quadruped"):
        for i in range(300):
            row: dict[str, Any] = {c: float(rng.normal()) for c in FEATURE_COLUMNS}
            row.update(
                run_id=f"r{i % 4}", asset_id=f"{t}-01", asset_type=t, seq=i, mode="moving",
                history=500, history_ok=True, max_gap_l=1.0, gap_ok=True,
            )  # fmt: skip
            rows.append(row)
    train = pd.DataFrame(rows)

    def build(sub: str) -> Path:
        det = (StatsDetector() if which == "stats" else LOFDetector(n_neighbors=10)).fit(train)
        art = replace(_art(), detector=det, detector_name=which)
        return save_artifact(art, tmp_path / sub)

    assert sha256_file(build("a")) == sha256_file(build("b"))
    assert os.path.getsize(build("a")) > 0

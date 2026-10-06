"""End to end with FITTED detectors (PLAN §11): generate -> features -> fit on train ->
select threshold on validation -> freeze artifact -> evaluate test once -> replay through
the service path, which must reproduce the evaluation's incidents exactly."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fleet_signal.data.config import GenerationConfig
from fleet_signal.data.generator import generate_dataset
from fleet_signal.data.ground_truth import load_faults
from fleet_signal.data.splits import PlannedRun
from fleet_signal.data.telemetry import load_telemetry
from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.lof import LOFDetector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.eval.official import OfficialResultExists, write_official
from fleet_signal.eval.protocol import EvalConfig, ScoredSet
from fleet_signal.eval.threshold import select_threshold, sweep
from fleet_signal.features.build import FeatureConfig, build_features
from fleet_signal.features.store import SplitLeakError, assert_only_split
from fleet_signal.incidents.grouping import IncidentParams
from fleet_signal.registry import ModelArtifact, load_artifact, save_artifact
from fleet_signal.service.replay import replay_asset
from fleet_signal.service.scorer import Scorer

PARAMS = IncidentParams(2, 5, 30)


def _run(seed: int, split: str, scenario: str = "normal", variant: str | None = None,
         asset: int | None = None) -> PlannedRun:  # fmt: skip
    return PlannedRun(f"r{seed}", seed, split, scenario, asset, variant)


PLAN = [
    *[_run(s, "train") for s in (1000, 1001, 1002, 1003)],
    _run(2000, "validation"),
    _run(2001, "validation", "motion_anomaly", "jump", 0),
    _run(2002, "validation", "overheating", "linear", 1),
    _run(2003, "validation", "sensor_freeze", "speed_mps", 2),
    _run(3000, "test"),
    _run(3001, "test", "motion_anomaly", "drift", 2),
    _run(3002, "test", "overheating", "runaway", 0),
]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory, small_cfg: GenerationConfig):
    cfg = replace(small_cfg, data={**small_cfg.data, "run_duration_s": 600})
    data_dir = tmp_path_factory.mktemp("data")
    generate_dataset(cfg, data_dir, runs=PLAN)
    feats = build_features(load_telemetry(data_dir=data_dir), FeatureConfig.load())
    faults = load_faults(data_dir=data_dir)
    ids = {sp: [r.run_id for r in PLAN if r.split == sp] for sp in ("train", "validation", "test")}
    by_split = {sp: feats[feats["run_id"].isin(v)] for sp, v in ids.items()}
    return cfg, data_dir, by_split, faults, ids


@pytest.mark.parametrize(
    "make", [StatsDetector, lambda: LOFDetector(n_neighbors=10, max_train=2_000)],
    ids=["stats", "lof"],
)  # fmt: skip
def test_full_pipeline_with_fitted_detector(pipeline, make, tmp_path: Path, cfg) -> None:
    _, _, by_split, faults, ids = pipeline
    ecfg = EvalConfig.load()

    # fit on train only, guarded
    assert_only_split(by_split["train"], "train", cfg)
    with pytest.raises(SplitLeakError):
        assert_only_split(pd.concat([by_split["train"], by_split["test"]]), "train", cfg)
    det: Detector = make().fit(by_split["train"])

    # threshold chosen on validation (no constraint on precision with this tiny set)
    val = by_split["validation"]
    val_ss = ScoredSet(val[["run_id", "asset_id", "seq"]].assign(score=det.score(val)),
                       faults, ids["validation"], ecfg, n_assets=3)  # fmt: skip
    pick = select_threshold(sweep(val_ss, PARAMS), budget=ecfg.budget)
    assert pick["recall"] > 0, "validation selection found nothing"

    # freeze
    fcfg = FeatureConfig.load()
    art = ModelArtifact(
        detector=det, detector_name=det.name, threshold=float(pick["threshold"]),
        incident_params=PARAMS.as_dict(), feature_config=fcfg.as_dict(),
        feature_schema_hash=fcfg.schema_hash, data_version="e2e", train_run_ids=ids["train"],
        validation_summary=pick, git_sha=None,
    )  # fmt: skip
    path = save_artifact(art, tmp_path / "models")
    frozen = load_artifact(path)

    # test, once
    test = by_split["test"]
    test_scored = test[["run_id", "asset_id", "seq"]].assign(score=frozen.detector.score(test))
    res = ScoredSet(test_scored, faults, ids["test"], ecfg, 3).evaluate(
        frozen.threshold, frozen.params
    )
    assert res.summary["n_detected"] >= 1, "comparison below would be trivially empty"
    write_official(det.name, frozen.model_version, res.summary, root=tmp_path / "official")
    with pytest.raises(OfficialResultExists):
        write_official(det.name, frozen.model_version, res.summary, root=tmp_path / "official")

    # the live path (scorer + streaming tracker) reproduces evaluation's incidents exactly
    tel = load_telemetry(data_dir=pipeline[1], run_ids=ids["test"])
    scorer = Scorer(path, allow_unregistered=True)
    for (run_id, asset_id), events in tel.groupby(["run_id", "asset_id"]):
        _, _, incidents = replay_asset(scorer, events)
        expected = res.incidents[
            (res.incidents["run_id"] == run_id) & (res.incidents["asset_id"] == asset_id)
        ]["open_seqs"].tolist()
        assert [i["open_seqs"] for i in incidents] == expected, (run_id, asset_id)


@pytest.fixture
def cfg(pipeline):
    return pipeline[0]

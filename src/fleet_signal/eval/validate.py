"""Validation-only selection and reporting for detectors.

Fits each detector on train-normal features, scores validation, chooses the
shared incident parameters and each detector's threshold under the
false-alert budget, and writes everything to results/validation/.

The test split is never loaded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.data.config import REPO_ROOT, GenerationConfig, load_config
from fleet_signal.data.ground_truth import load_faults
from fleet_signal.data.splits import split_run_ids
from fleet_signal.detectors.base import Detector
from fleet_signal.eval.bootstrap import bootstrap_ci, paired_difference
from fleet_signal.eval.detectability import with_detectability
from fleet_signal.eval.protocol import EvalConfig, EvalResult, ScoredSet, breakdown
from fleet_signal.eval.threshold import select_incident_params, select_threshold, sweep
from fleet_signal.features.store import assert_only_split, load_features
from fleet_signal.incidents.grouping import IncidentParams

VALIDATION_DIR = REPO_ROOT / "results" / "validation"


def seen_variants(gen_cfg: GenerationConfig) -> set[tuple[str, str]]:
    """Variants that appear in validation (everything not marked test_only)."""
    return {
        (ftype, v)
        for ftype, spec in gen_cfg.data["faults"].items()
        if isinstance(spec, dict) and "variants" in spec
        for v, vs in spec["variants"].items()
        if not vs["test_only"]
    }


def fit_on_train(detectors: list[Detector], gen_cfg: GenerationConfig) -> pd.DataFrame:
    train = load_features("train", gen_cfg=gen_cfg)
    assert_only_split(train, "train", gen_cfg)
    for det in detectors:
        det.fit(train)
    return train


def scored_frame(det: Detector, feats: pd.DataFrame) -> pd.DataFrame:
    return feats[["run_id", "asset_id", "seq"]].assign(score=det.score(feats))


def build_scored_set(
    det: Detector, split: str, gen_cfg: GenerationConfig, ecfg: EvalConfig
) -> ScoredSet:
    feats = load_features(split, gen_cfg=gen_cfg)
    assert_only_split(feats, split, gen_cfg)
    ids = split_run_ids(gen_cfg)[split]
    return ScoredSet(
        scored_frame(det, feats),
        with_detectability(load_faults(), gen_cfg),
        ids,
        ecfg,
        n_assets=len(gen_cfg.data["fleet"]),
        seen_variants=seen_variants(gen_cfg),
    )


@dataclass
class Selection:
    params: IncidentParams
    params_table: pd.DataFrame
    picks: dict[str, dict[str, Any]]
    curves: dict[str, pd.DataFrame]
    results: dict[str, EvalResult]


BASELINE_NAMES: tuple[str, ...] = ("rule", "stats")


def select_on_validation(
    detectors: list[Detector],
    gen_cfg: GenerationConfig | None = None,
    ecfg: EvalConfig | None = None,
    params: IncidentParams | None = None,
) -> Selection:
    """Fit on train, choose incident params (unless given) and thresholds on validation.

    Incident params are chosen from the baselines only, so adding or swapping the ML
    detector can never re-tune the grouping the baselines are judged with.
    """
    gen_cfg = gen_cfg or load_config()
    ecfg = ecfg or EvalConfig.load()
    fit_on_train(detectors, gen_cfg)
    sets = {d.name: build_scored_set(d, "validation", gen_cfg, ecfg) for d in detectors}
    if params is None:
        basis = {k: v for k, v in sets.items() if k in BASELINE_NAMES} or sets
        params, table = select_incident_params(basis, ecfg)
    else:
        table = pd.DataFrame([params.as_dict()])
    curves, picks, results = {}, {}, {}
    for name, ss in sets.items():
        curves[name] = sweep(ss, params)
        picks[name] = select_threshold(curves[name], ecfg.budget, ecfg.min_precision)
        results[name] = ss.evaluate(picks[name]["threshold"], params)
    return Selection(params, table, picks, curves, results)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_validation_report(
    sel: Selection, gen_cfg: GenerationConfig, ecfg: EvalConfig, out_dir: Path = VALIDATION_DIR
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_assets = len(gen_cfg.data["fleet"])
    sel.params_table.to_csv(out_dir / "incident_params_grid.csv", index=False)
    report: dict[str, Any] = {
        "split": "validation",
        "data_version": gen_cfg.version,
        "incident_params": sel.params.as_dict(),
        "false_alert_budget_per_10min": ecfg.budget,
        "min_precision": ecfg.min_precision,
        "detectors": {},
    }
    for name, res in sel.results.items():
        sel.curves[name].to_csv(out_dir / f"{name}_threshold_curve.csv", index=False)
        res.per_fault.to_csv(out_dir / f"{name}_per_fault.csv", index=False)
        res.incidents.to_csv(out_dir / f"{name}_incidents.csv", index=False)
        report["detectors"][name] = {
            "selected": sel.picks[name],
            "summary": res.summary,
            "ci95": bootstrap_ci(res.per_run, n_assets, ecfg.n_resamples, ecfg.bootstrap_seed),
            "by_fault_type": breakdown(res.per_fault, ["fault_type"]).to_dict("records"),
            "by_variant": breakdown(res.per_fault, ["fault_type", "variant"]).to_dict("records"),
            "window_confusion": res.window_confusion,
        }
    names = list(sel.results)
    report["paired"] = {
        f"{a}_minus_{b}": paired_difference(
            sel.results[a].per_run,
            sel.results[b].per_run,
            n_assets,
            ecfg.n_resamples,
            ecfg.bootstrap_seed,
        )
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    report = _jsonable(report)
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    return report

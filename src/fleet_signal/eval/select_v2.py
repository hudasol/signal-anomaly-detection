"""v2 model selection (PLAN_V2 §5). Validation only; the test split is never loaded.

1. Baselines (rule, robust z) and the shared incident grouping, chosen on validation
   exactly as in v1.
2. LOF neighbours grid (k in 10/20/30, 30k train rows per type).
3. The fast path, fitted on train.
4. Candidate systems: lof, fast, hybrid_lof_fast, hybrid_rule_lof_fast,
   hybrid_rule_fast, each with an incident opening after 1 or 2 consecutive alerts
   (close and cooldown as chosen for the baselines). LOF is always evaluated as the
   ML detector; the SHIPPED candidate need not contain it (see PROCESS_LOG
   2026-10-08: a first pass restricted candidates to LOF systems; on validation the
   single hybrid threshold is set by LOF and slows the fast path to 4-5 events, so
   that restriction would pick a system that fails the latency bar. Changed on
   validation evidence, before the v2 test set existed).
5. Pick: feasible (false incidents <= budget AND precision >= min_precision),
   then meets the brief's bar on validation (recall >= 0.85 and expected-to-catch
   latency <= 3), then highest recall, then lowest expected-to-catch latency
   (misses count as infinite), then lowest all-progressive latency.
6. Best baseline = the baseline with the higher VALIDATION recall (v1 picked it on
   test; fixed here).

Writes results/validation/v2_selection.json and v2_candidates.csv.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.data.config import load_config
from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.fastpath import FastPathDetector
from fleet_signal.detectors.hybrid import HybridDetector
from fleet_signal.detectors.lof import LOFDetector
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.eval.protocol import EvalConfig
from fleet_signal.eval.threshold import select_threshold, sweep
from fleet_signal.eval.validate import (
    VALIDATION_DIR,
    build_scored_set,
    fit_on_train,
    select_on_validation,
    write_validation_report,
)
from fleet_signal.incidents.grouping import IncidentParams

ELIGIBLE = ("lof", "fast", "hybrid_lof_fast", "hybrid_rule_lof_fast", "hybrid_rule_fast")
OPEN_N = (1, 2)
LOF_K = (10, 20, 30)
LOF_MAX_TRAIN = 30_000
SELECTION = VALIDATION_DIR / "v2_selection.json"


def build_candidate(name: str, k: int) -> Detector:
    """A fresh, unfitted candidate (used by selection and by signal-train)."""
    lof = LOFDetector(n_neighbors=k, max_train=LOF_MAX_TRAIN)
    if name == "lof":
        return lof
    if name == "fast":
        return FastPathDetector()
    parts: dict[str, list[Detector]] = {
        "hybrid_lof_fast": [lof, FastPathDetector()],
        "hybrid_rule_lof_fast": [RuleDetector(), lof, FastPathDetector()],
        "hybrid_rule_fast": [RuleDetector(), FastPathDetector()],
    }
    return HybridDetector(parts[name], name=name)


def _finite(v: Any) -> bool:
    return v is not None and bool(np.isfinite(v))


def meets_validation_bar(row: dict[str, Any], bar: dict[str, float]) -> bool:
    exp = row["expected_latency_median"]
    return bool(
        row["recall"] >= bar["recall"]
        and row["precision"] >= bar["precision"]
        and row["fp_per_10min"] <= bar["false_alerts_per_10min"]
        and _finite(exp)
        and exp <= bar["median_latency_events"]
    )


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    exp = row["expected_latency_median"]
    allp = row["progressive_latency_median"]
    return (
        0.0 if row["meets_bar"] else 1.0,
        -row["recall"],
        exp if exp is not None and np.isfinite(exp) else 1e12,
        allp if allp is not None and np.isfinite(allp) else 1e12,
    )


def run(out_dir: Path = VALIDATION_DIR) -> dict[str, Any]:
    gen_cfg, ecfg = load_config(), EvalConfig.load()
    base = select_on_validation([RuleDetector(), StatsDetector()], gen_cfg, ecfg)
    shared = base.params
    best_baseline = max(
        ("rule", "stats"),
        key=lambda n: (base.results[n].summary["recall"], base.results[n].summary["precision"]),
    )

    # LOF neighbours: chosen alone at the shared grouping (as in v1).
    lof_rows = []
    for k in LOF_K:
        sel = select_on_validation(
            [LOFDetector(n_neighbors=k, max_train=LOF_MAX_TRAIN)], gen_cfg, ecfg, shared
        )
        s, pick = sel.results["lof"].summary, sel.picks["lof"]
        lof_rows.append({"k": k, "feasible": bool(pick["feasible"]), **_brief(s)})
    lof_df = pd.DataFrame(lof_rows)
    ok = lof_df[lof_df["feasible"]] if lof_df["feasible"].any() else lof_df
    best_k = int(ok.sort_values(["recall", "precision"], ascending=False).iloc[0]["k"])

    # Fit every distinct part once on train, then assemble candidates from fitted parts.
    lof = LOFDetector(n_neighbors=best_k, max_train=LOF_MAX_TRAIN)
    fast, rule = FastPathDetector(), RuleDetector()
    train = fit_on_train([lof, fast, rule], gen_cfg)
    cands: dict[str, Detector] = {
        "lof": lof,
        "fast": fast,
        "hybrid_lof_fast": HybridDetector([lof, fast], "hybrid_lof_fast", prefit=True),
        "hybrid_rule_lof_fast": HybridDetector(
            [rule, lof, fast], "hybrid_rule_lof_fast", prefit=True
        ),
        "hybrid_rule_fast": HybridDetector([rule, fast], "hybrid_rule_fast", prefit=True),
    }
    for det in cands.values():
        if isinstance(det, HybridDetector):
            det.fit(train)  # calibration only (parts already fitted)

    rows: list[dict[str, Any]] = []
    for name, det in cands.items():
        ss = build_scored_set(det, "validation", gen_cfg, ecfg)
        for n in OPEN_N:
            params = replace(shared, open_n=n)
            pick = select_threshold(sweep(ss, params), ecfg.budget, ecfg.min_precision)
            s = ss.evaluate(pick["threshold"], params).summary
            rows.append(
                {
                    "candidate": name,
                    "eligible": name in ELIGIBLE,
                    "open_n": n,
                    "feasible": bool(pick["feasible"]),
                    **_brief(s),
                }
            )
            rows[-1]["meets_bar"] = meets_validation_bar(rows[-1], ecfg.bar)
    grid = pd.DataFrame(rows)
    eligible = [r for r in rows if r["eligible"] and r["feasible"]]
    if not eligible:
        raise SystemExit("no eligible candidate is feasible on validation")
    chosen = min(eligible, key=_rank_key)

    out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_dir / "v2_candidates.csv", index=False)
    lof_df.to_csv(out_dir / "v2_lof_grid.csv", index=False)
    summary = {
        "data_version": gen_cfg.version,
        "shared_incident_params": shared.as_dict(),
        "best_baseline": best_baseline,
        "lof_k": best_k,
        "chosen": {
            "name": chosen["candidate"],
            "open_n": int(chosen["open_n"]),
            "validation": {k: chosen[k] for k in _brief({}).keys() if k in chosen},
        },
        "rule": (
            "feasible, then meets the bar on validation, then recall, then expected-to-catch "
            "latency, then all-progressive latency"
        ),
    }
    (out_dir / "v2_selection.json").write_text(json.dumps(_clean(summary), indent=2))
    write_validation_report(base, gen_cfg, ecfg, out_dir)
    return {"grid": grid, "lof_grid": lof_df, **summary}


def chosen_params(sel: dict[str, Any]) -> IncidentParams:
    p = dict(sel["shared_incident_params"])
    p["open_n"] = int(sel["chosen"]["open_n"])
    return IncidentParams(**p)


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    keys = ("threshold", "precision", "per_fault_precision", "recall", "fp_per_10min",
            "progressive_latency_median", "expected_latency_median", "n_expected",
            "n_expected_detected")  # fmt: skip
    return {k: s.get(k) for k in keys}


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return _clean(o.item())
    return o

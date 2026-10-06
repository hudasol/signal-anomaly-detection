"""Reproducible ML model selection and ablation. Validation only.

1. Isolation Forest grid (raw and z-score inputs).
2. LOF grid (the PLAN §6 escalation, because Isolation Forest lost on validation).
3. Pick the ML candidate with the selection rule used everywhere: feasible
   (false alerts and precision constraints), then highest recall, then lowest
   median progressive latency.
4. Ablation: refit the chosen model with one feature group removed at a time.

Incident params are fixed to those chosen from the baselines.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from fleet_signal.data.config import load_config
from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.iforest import IsolationForestDetector
from fleet_signal.detectors.lof import LOFDetector
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.eval.protocol import EvalConfig
from fleet_signal.eval.validate import VALIDATION_DIR, select_on_validation
from fleet_signal.incidents.grouping import IncidentParams

LOF_GROUPS = ("level", "trend", "volatility", "freeze", "motion")


def _row(
    kind: str, params: dict[str, Any], det: Detector, ecfg: EvalConfig, inc: IncidentParams
) -> dict[str, Any]:
    sel = select_on_validation([det], ecfg=ecfg, params=inc)
    s, pick = sel.results[det.name].summary, sel.picks[det.name]
    return {
        "model": kind,
        **{k: (",".join(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "feasible": bool(pick["feasible"]),
        "threshold": s["threshold"],
        "precision": s["precision"],
        "recall": s["recall"],
        "fp_per_10min": s["fp_per_10min"],
        "latency_median": s["progressive_latency_median"],
    }


def run(out_dir: Path = VALIDATION_DIR) -> dict[str, Any]:
    gen_cfg, ecfg = load_config(), EvalConfig.load()
    base = select_on_validation([RuleDetector(), StatsDetector()], gen_cfg, ecfg)
    inc = base.params
    candidates: list[tuple[str, dict[str, Any], Callable[[], Detector]]] = []
    for space in ("raw", "zscore"):
        for ms in (256, 2048):
            for mf in (0.5, 1.0):
                p: dict[str, Any] = {"input_space": space, "max_samples": ms, "max_features": mf}
                candidates.append(("iforest", p, partial(IsolationForestDetector, **p)))
    for k in (10, 20, 30):
        for mt in (15_000, 30_000):
            p = {"n_neighbors": k, "max_train": mt}  # same dict[str, Any] variable
            candidates.append(("lof", p, partial(LOFDetector, **p)))

    rows = [_row(kind, p, make(), ecfg, inc) for kind, p, make in candidates]
    grid = pd.DataFrame(rows)
    feasible = grid[grid["feasible"]].assign(_lat=grid["latency_median"].fillna(1e9))
    best = feasible.sort_values(["recall", "_lat"], ascending=[False, True]).iloc[0]
    # Integer hyperparameters stay ints; max_features stays a float (1.0 = all features,
    # whereas int 1 would mean a single feature to scikit-learn).
    int_keys = {"n_neighbors", "max_train", "max_samples"}
    keys = ("n_neighbors", "max_train") if best["model"] == "lof" else (
        "max_samples", "max_features")  # fmt: skip
    chosen: dict[str, Any] = {
        "model": str(best["model"]),
        "params": {k: int(best[k]) if k in int_keys else float(best[k]) for k in keys},
    }
    if best["model"] == "iforest":
        chosen["params"]["input_space"] = best["input_space"]

    ablation: list[dict[str, Any]] = []
    if chosen["model"] == "lof":
        all_groups = LOF_GROUPS
        ablation.append(
            _row("lof", {"dropped": "none"}, LOFDetector(**chosen["params"]), ecfg, inc)
        )
        for g in all_groups:
            kept = tuple(x for x in all_groups if x != g)
            det = LOFDetector(**chosen["params"], groups=kept)
            ablation.append(_row("lof", {"dropped": g}, det, ecfg, inc))

    out_dir.mkdir(parents=True, exist_ok=True)
    grid.drop(columns=["_lat"], errors="ignore").to_csv(out_dir / "ml_model_grid.csv", index=False)
    pd.DataFrame(ablation).to_csv(out_dir / "ml_ablation.csv", index=False)
    summary = {"incident_params": inc.as_dict(), "chosen": chosen}
    (out_dir / "ml_selection.json").write_text(json.dumps(summary, indent=2))
    return {"grid": grid, "ablation": pd.DataFrame(ablation), **summary}

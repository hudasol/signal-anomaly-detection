"""Threshold and incident-parameter selection. Validation data only.

Selection rule (PLAN §7.2, amended 2026-10-06): among thresholds whose
false-incident rate is within the budget AND whose precision reaches
min_precision, take the one with the highest fault-event recall; break ties by
lower median progressive latency, then by the higher (more conservative)
threshold. If no threshold satisfies both constraints, take the best F1 among
thresholds within the false-alert budget and mark the selection infeasible.
The caller must never pass test data here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.eval.protocol import EvalConfig, ScoredSet
from fleet_signal.incidents.grouping import IncidentParams


def threshold_candidates(scores: np.ndarray, n: int) -> np.ndarray:
    """Quantiles concentrated in the upper tail, where useful thresholds live."""
    if scores.size == 0:
        return np.array([])
    tail = np.logspace(-0.5, -5, n)  # fraction of events above the threshold
    qs = np.clip(1.0 - tail, 0.0, 1.0)
    cands = np.quantile(scores, qs)
    cands = np.append(cands, np.nextafter(scores.max(), np.inf))  # "never alert"
    return np.unique(cands)


def sweep(
    ss: ScoredSet, params: IncidentParams, candidates: np.ndarray | None = None
) -> pd.DataFrame:
    cands = (
        candidates
        if candidates is not None
        else threshold_candidates(ss.scores(), ss.ecfg.n_candidates)
    )
    return pd.DataFrame([ss.evaluate(float(t), params).summary for t in cands])


def select_threshold(
    curve: pd.DataFrame, budget: float, min_precision: float = 0.0
) -> dict[str, Any]:
    precision = curve["precision"].fillna(0.0) if "precision" in curve else 1.0
    feasible = curve[(curve["fp_per_10min"] <= budget) & (precision >= min_precision)]
    if feasible.empty:
        within = curve[curve["fp_per_10min"] <= budget]
        pool = within if not within.empty else curve
        if "f1" in pool and pool["f1"].notna().any():
            best = pool.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
        else:
            best = pool.sort_values(["fp_per_10min", "threshold"], ascending=[True, False]).iloc[0]
        return {**{str(k): v for k, v in best.to_dict().items()}, "feasible": False}
    lat = feasible["progressive_latency_median"].fillna(np.inf)
    ranked = feasible.assign(_lat=lat).sort_values(
        ["recall", "_lat", "threshold"], ascending=[False, True, False]
    )
    top = ranked.iloc[0].drop("_lat").to_dict()
    return {**{str(k): v for k, v in top.items()}, "feasible": True}


def select_incident_params(
    sets: dict[str, ScoredSet], ecfg: EvalConfig
) -> tuple[IncidentParams, pd.DataFrame]:
    """One set of incident params for all detectors, chosen on validation.

    Order: most detectors with a feasible threshold, then highest mean selected recall,
    then lower mean latency, then fewer false incidents, then the smaller grid point.
    """
    rows: list[dict[str, Any]] = []
    for params in ecfg.param_grid():
        picks = {
            name: select_threshold(sweep(ss, params), ecfg.budget, ecfg.min_precision)
            for name, ss in sets.items()
        }
        rows.append(
            {
                **params.as_dict(),
                "mean_recall": float(np.mean([p["recall"] for p in picks.values()])),
                "mean_latency": _nanmean([p["progressive_latency_median"] for p in picks.values()]),
                "mean_fp_per_10min": float(np.mean([p["fp_per_10min"] for p in picks.values()])),
                "n_feasible": sum(bool(p["feasible"]) for p in picks.values()),
                **{f"recall_{k}": v["recall"] for k, v in picks.items()},
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["n_feasible", "mean_recall", "mean_latency", "mean_fp_per_10min",
         "open_n", "close_m", "cooldown_c"],
        ascending=[False, False, True, True, True, True, True],
    ).iloc[0]  # fmt: skip
    chosen = IncidentParams(int(best["open_n"]), int(best["close_m"]), int(best["cooldown_c"]))
    return chosen, table


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

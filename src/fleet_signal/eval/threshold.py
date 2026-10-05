"""Threshold and incident-parameter selection. Validation data only.

Selection rule (PLAN §7.2): among thresholds whose false-incident rate is within
the budget, take the one with the highest fault-event recall; break ties by
lower median progressive latency, then by the higher (more conservative)
threshold. If no threshold fits the budget, take the one with the lowest
false-incident rate and mark the selection infeasible. The caller must never
pass test data here.
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


def select_threshold(curve: pd.DataFrame, budget: float) -> dict[str, Any]:
    feasible = curve[curve["fp_per_10min"] <= budget]
    if feasible.empty:
        best = curve.sort_values(["fp_per_10min", "threshold"], ascending=[True, False]).iloc[0]
        return {**best.to_dict(), "feasible": False}
    lat = feasible["progressive_latency_median"].fillna(np.inf)
    ranked = feasible.assign(_lat=lat).sort_values(
        ["recall", "_lat", "threshold"], ascending=[False, True, False]
    )
    return {**ranked.iloc[0].drop("_lat").to_dict(), "feasible": True}


def select_incident_params(
    sets: dict[str, ScoredSet], ecfg: EvalConfig
) -> tuple[IncidentParams, pd.DataFrame]:
    """One set of incident params for all detectors: best mean selected recall on validation.

    Ties: lower mean latency, then fewer mean false incidents, then the smaller grid point.
    """
    rows: list[dict[str, Any]] = []
    for params in ecfg.param_grid():
        picks = {
            name: select_threshold(sweep(ss, params), ecfg.budget) for name, ss in sets.items()
        }
        rows.append(
            {
                **params.as_dict(),
                "mean_recall": float(np.mean([p["recall"] for p in picks.values()])),
                "mean_latency": float(
                    np.nanmean([p["progressive_latency_median"] for p in picks.values()])
                ),
                "mean_fp_per_10min": float(np.mean([p["fp_per_10min"] for p in picks.values()])),
                "all_feasible": all(p["feasible"] for p in picks.values()),
                **{f"recall_{k}": v["recall"] for k, v in picks.items()},
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["all_feasible", "mean_recall", "mean_latency", "mean_fp_per_10min",
         "open_n", "close_m", "cooldown_c"],
        ascending=[False, False, True, True, True, True, True],
    ).iloc[0]  # fmt: skip
    chosen = IncidentParams(int(best["open_n"]), int(best["close_m"]), int(best["cooldown_c"]))
    return chosen, table

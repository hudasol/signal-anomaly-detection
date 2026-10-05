"""Bootstrap confidence intervals, resampling whole runs (not events).

Events inside one run are strongly correlated, so the run is the independent
unit. For a paired comparison both detectors are evaluated on the same
resampled runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fleet_signal.eval.protocol import rates

METRICS = ("precision", "recall", "f1", "fp_per_10min")


def _resample_ids(run_ids: np.ndarray, n: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(run_ids, size=len(run_ids), replace=True) for _ in range(n)]


def bootstrap_ci(
    per_run: pd.DataFrame, n_assets: int, n_resamples: int, seed: int, alpha: float = 0.05
) -> dict[str, tuple[float, float]]:
    pr = per_run.set_index("run_id")
    samples = {m: [] for m in METRICS}
    for ids in _resample_ids(pr.index.to_numpy(), n_resamples, seed):
        r = rates(pr.loc[ids], n_assets)
        for m in METRICS:
            samples[m].append(r[m])
    return {
        m: (float(np.nanquantile(v, alpha / 2)), float(np.nanquantile(v, 1 - alpha / 2)))
        for m, v in samples.items()
    }


def paired_difference(
    per_run_a: pd.DataFrame,
    per_run_b: pd.DataFrame,
    n_assets: int,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """CI of (A - B) for each metric, resampling the same runs for both."""
    a = per_run_a.set_index("run_id")
    b = per_run_b.set_index("run_id")
    if set(a.index) != set(b.index):
        raise ValueError("paired comparison needs both detectors on the same runs")
    point_a, point_b = rates(a, n_assets), rates(b, n_assets)
    diffs = {m: [] for m in METRICS}
    for ids in _resample_ids(a.index.to_numpy(), n_resamples, seed):
        ra, rb = rates(a.loc[ids], n_assets), rates(b.loc[ids], n_assets)
        for m in METRICS:
            diffs[m].append(ra[m] - rb[m])
    out: dict[str, dict[str, float]] = {}
    for m in METRICS:
        lo, hi = np.nanquantile(diffs[m], [alpha / 2, 1 - alpha / 2])
        out[m] = {"diff": point_a[m] - point_b[m], "ci_low": float(lo), "ci_high": float(hi)}
    return out

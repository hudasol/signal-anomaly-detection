"""Drift monitor: is this telemetry still like the data the model was trained on?

A detector's thresholds were chosen for train/validation-like telemetry. If the
fleet starts behaving differently (hotter climate, noisier sensors, aged batteries,
new firmware), scores and false-alarm rates move with it, often without any single
alert. This monitor compares the distribution of a set of features in a window of
telemetry (one asset, e.g. its last 650 events or a whole run) against the train
distribution, per (asset type, mode), with the Population Stability Index:

    PSI = sum over bins (cur - ref) * ln(cur / ref)        (10 train-quantile bins)

Each feature's PSI is the row-weighted mean over the modes present with >= MIN_ROWS
rows. Features differ a lot in how much they move between normal runs (ambient
temperature changes per run; sensor-noise residuals hardly move), so each feature is
judged against its own normal spread: its threshold is the largest PSI seen on
**validation normal** runs. The window's drift score is the largest
PSI / feature-threshold ratio: `caution` at 1, `drift` at 2.

(Round 1 used the mean PSI over all features against one threshold. On the shifted
fleets it barely reacted to doubled sensor noise, because features with large normal
swings dominated the mean. Recorded in PROCESS_LOG; the per-feature version was then
evaluated on freshly generated fleets.) A `caution` or `drift` status means: the model's
evaluated error rates may not hold here; treat its decisions with caution. It does
not change any decision.

Faults also move distributions; this is not a fault detector and is calibrated to
ignore nothing but normal variation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.features.build import scorable

DRIFT_FEATURES: tuple[str, ...] = (
    "temp_c",
    "speed_mps",
    "link_pct",
    "temp_slope_l",
    "batt_slope_l",
    "link_std_m",
    "speed_mismatch",
    "batt_res3",
    "temp_res3",
    "spd_d3",
)
N_BINS = 10
MIN_ROWS = 100
EPS = 1e-4


def psi(ref_props: np.ndarray, edges: np.ndarray, values: np.ndarray) -> float:
    """PSI of `values` against reference bin proportions (bins given by inner edges)."""
    idx = np.searchsorted(edges, values, side="right")
    cur = np.bincount(idx, minlength=len(ref_props)) / max(len(values), 1)
    ref, cur = np.clip(ref_props, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


@dataclass
class DriftMonitor:
    features: tuple[str, ...] = DRIFT_FEATURES
    # (asset_type, mode, feature) -> (inner edges, reference proportions)
    reference: dict[str, dict[str, Any]] = field(default_factory=dict)
    caution: float = float("nan")  # on the ratio scale: 1.0 once calibrated
    drift: float = float("nan")
    feature_threshold: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def _key(t: str, m: str, f: str) -> str:
        return f"{t}|{m}|{f}"

    def fit(self, train: pd.DataFrame) -> DriftMonitor:
        train = train[scorable(train)]
        for (t, m), g in train.groupby(["asset_type", "mode"]):
            if len(g) < MIN_ROWS:
                continue
            for f in self.features:
                x = g[f].to_numpy(dtype=float)
                edges = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)[1:-1]))
                props = np.bincount(np.searchsorted(edges, x, side="right"),
                                    minlength=len(edges) + 1) / len(x)  # fmt: skip
                self.reference[self._key(str(t), str(m), f)] = {
                    "edges": edges.tolist(), "props": props.tolist()}  # fmt: skip
        return self

    def feature_psi(self, window: pd.DataFrame) -> dict[str, float]:
        """Row-weighted PSI per feature for one asset's window."""
        window = window[scorable(window)]
        per_feature: dict[str, list[tuple[float, int]]] = {f: [] for f in self.features}
        for (t, m), g in window.groupby(["asset_type", "mode"]):
            if len(g) < MIN_ROWS:
                continue
            for f in self.features:
                ref = self.reference.get(self._key(str(t), str(m), f))
                if ref is None:
                    continue
                value = psi(np.asarray(ref["props"]), np.asarray(ref["edges"]),
                            g[f].to_numpy(dtype=float))  # fmt: skip
                per_feature[f].append((value, len(g)))
        return {
            f: float(np.average([v for v, _ in vals], weights=[n for _, n in vals]))
            for f, vals in per_feature.items() if vals
        }  # fmt: skip

    def score(self, window: pd.DataFrame) -> dict[str, Any]:
        """Drift score of one asset's window (largest PSI / own-threshold ratio), with the
        features that moved most relative to their normal spread."""
        feat_psi = self.feature_psi(window)
        if not feat_psi:
            return {"score": float("nan"), "status": "insufficient_data", "top": []}
        if not self.feature_threshold:
            return {"score": float("nan"), "status": "uncalibrated", "top": []}
        ratio = {f: v / self.feature_threshold[f] for f, v in feat_psi.items()
                 if f in self.feature_threshold}  # fmt: skip
        s = float(max(ratio.values()))
        top = sorted(ratio.items(), key=lambda kv: -kv[1])[:3]
        return {"score": s, "status": self.status(s), "top": [[f, round(v, 2)] for f, v in top],
                "mean_psi": float(np.mean(list(feat_psi.values())))}  # fmt: skip

    def status(self, s: float) -> str:
        if not np.isfinite(self.caution):
            return "uncalibrated"
        return "drift" if s >= self.drift else "caution" if s >= self.caution else "ok"

    def calibrate(self, normal_windows: list[pd.DataFrame]) -> DriftMonitor:
        per = [self.feature_psi(w) for w in normal_windows]
        for f in self.features:
            vals = [p[f] for p in per if f in p and np.isfinite(p[f])]
            if vals:
                self.feature_threshold[f] = max(float(max(vals)), EPS)
        self.caution, self.drift = 1.0, 2.0
        return self

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"features": list(self.features), "caution": self.caution,
                                    "drift": self.drift, "reference": self.reference,
                                    "feature_threshold": self.feature_threshold}))  # fmt: skip

    @classmethod
    def load(cls, path: Path) -> DriftMonitor:
        d = json.loads(Path(path).read_text())
        return cls(tuple(d["features"]), d["reference"], d["caution"], d["drift"],
                   d.get("feature_threshold", {}))  # fmt: skip

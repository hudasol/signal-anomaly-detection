"""Isolation Forest detector (PLAN §6).

One forest per asset type, trained on train-normal windows only (novelty
detection: it learns the normal envelope, it never sees a fault). The raw
score is `-score_samples`, rescaled per asset type against that type's train
score distribution so one threshold works across the three models:

    score = (raw - median_train) / (q99.9_train - median_train)

so 0 is a typical normal window and 1 is the edge of the train-normal tail.

Isolation Forest has no faithful per-feature attribution. Evidence is
therefore reported as the features with the largest robust deviation from
train-normal for that asset type and mode, labelled as evidence, not as the
model's internal reasoning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.features.build import FEATURE_GROUPS, scorable

DEFAULT_GROUPS: tuple[str, ...] = ("level", "trend", "volatility", "freeze", "motion", "context")
TYPE_COLUMNS = tuple(c for c in FEATURE_GROUPS["context"] if c.startswith("type_"))


class IsolationForestDetector(Detector):
    name = "iforest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: int | float = 1024,
        max_features: float = 1.0,
        groups: tuple[str, ...] = DEFAULT_GROUPS,
        random_state: int = 0,
        input_space: str = "zscore",
    ) -> None:
        self.params: dict[str, Any] = {
            "n_estimators": n_estimators,
            "max_samples": max_samples,
            "max_features": max_features,
            "groups": tuple(groups),
            "random_state": random_state,
            "input_space": input_space,
        }
        cols = [c for g in groups for c in FEATURE_GROUPS[g]]
        # One model per asset type, so the asset-type one-hots carry no information.
        self.features = [c for c in cols if c not in TYPE_COLUMNS]
        self.models: dict[str, IsolationForest] = {}
        self.calibration: dict[str, tuple[float, float]] = {}
        # Train-normal robust stats per (asset type, mode): used as the z-score input space
        # and to report evidence.
        self.evidence_model = StatsDetector(
            {"feature_groups": [g for g in groups if g != "context"] or ["level"],
             "exclude": [], "min_group_rows": 200}
        )  # fmt: skip
        self.fitted_runs: list[str] = []

    def fit(self, train: pd.DataFrame) -> IsolationForestDetector:
        train = train[scorable(train)]
        if train.empty:
            raise ValueError("no scorable training rows")
        self.fitted_runs = sorted(train["run_id"].unique())
        self.evidence_model.fit(train)
        p = self.params
        for type_key, g in train.groupby("asset_type"):
            asset_type = str(type_key)
            x = self._inputs(g)
            model = IsolationForest(
                n_estimators=p["n_estimators"],
                max_samples=min(p["max_samples"], len(x))
                if isinstance(p["max_samples"], int)
                else p["max_samples"],
                max_features=p["max_features"],
                random_state=p["random_state"],
                n_jobs=-1,
            ).fit(x)
            raw = -model.score_samples(x)
            med, hi = float(np.median(raw)), float(np.quantile(raw, 0.999))
            self.models[asset_type] = model
            self.calibration[asset_type] = (med, max(hi - med, 1e-9))
        return self

    def _inputs(self, feats: pd.DataFrame) -> np.ndarray:
        """Model inputs: raw features, or robust z per (asset type, mode) from train stats."""
        if self.params["input_space"] == "raw":
            return feats[self.features].to_numpy(dtype=float)
        z = self.evidence_model._z(feats)
        z = np.clip(np.nan_to_num(z, nan=0.0), -50.0, 50.0)
        extra = [c for c in self.features if c not in self.evidence_model.features]
        return np.hstack([z, feats[extra].to_numpy(dtype=float)]) if extra else z

    def _raw_score(self, feats: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("IsolationForestDetector used before fit()")
        out = np.full(len(feats), np.nan)
        types = feats["asset_type"].to_numpy()
        for asset_type, model in self.models.items():
            rows = types == asset_type
            if not rows.any():
                continue
            raw = -model.score_samples(self._inputs(feats.loc[rows]))
            med, span = self.calibration[asset_type]
            out[rows] = (raw - med) / span
        return out

    def _contributions(self, feats: pd.DataFrame) -> pd.DataFrame:
        return self.evidence_model._contributions(feats)

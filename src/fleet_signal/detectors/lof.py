"""Local Outlier Factor in novelty mode: the PLAN §6 escalation candidate.

Tried once because Isolation Forest did not beat the baselines on validation.
One model per asset type, fit on a fixed-seed subsample of train-normal rows
(LOF scoring cost grows with training size). Inputs are robust z-scores per
(asset type, mode), clipped, because LOF is distance-based and needs features
on a common scale. Score calibration and evidence work as for IsolationForest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.features.build import scorable


class LOFDetector(Detector):
    name = "lof"

    def __init__(
        self,
        n_neighbors: int = 20,
        max_train: int = 30_000,
        groups: tuple[str, ...] = ("level", "trend", "volatility", "freeze", "motion"),
        random_state: int = 0,
    ) -> None:
        self.params: dict[str, Any] = {
            "n_neighbors": n_neighbors,
            "max_train": max_train,
            "groups": tuple(groups),
            "random_state": random_state,
        }
        self.space = StatsDetector(
            {"feature_groups": list(groups), "exclude": [], "min_group_rows": 200}
        )
        self.models: dict[str, LocalOutlierFactor] = {}
        self.calibration: dict[str, tuple[float, float]] = {}
        self.fitted_runs: list[str] = []

    def _inputs(self, feats: pd.DataFrame) -> np.ndarray:
        z = self.space.zscores(feats)
        return np.clip(np.nan_to_num(z, nan=0.0), -20.0, 20.0)

    def fit(self, train: pd.DataFrame) -> LOFDetector:
        train = train[scorable(train)]
        self.fitted_runs = sorted(train["run_id"].unique())
        self.space.fit(train)
        rng = np.random.default_rng(self.params["random_state"])
        for type_key, g in train.groupby("asset_type"):
            asset_type = str(type_key)
            if len(g) > self.params["max_train"]:
                g = g.iloc[np.sort(rng.choice(len(g), self.params["max_train"], replace=False))]
            x = self._inputs(g)
            model = LocalOutlierFactor(
                n_neighbors=self.params["n_neighbors"], novelty=True, n_jobs=-1
            ).fit(x)
            raw = -model.score_samples(x)
            med, hi = float(np.median(raw)), float(np.quantile(raw, 0.999))
            self.models[asset_type] = model
            self.calibration[asset_type] = (med, max(hi - med, 1e-9))
        return self

    def _raw_score(self, feats: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("LOFDetector used before fit()")
        out = np.full(len(feats), np.nan)
        types = feats["asset_type"].to_numpy()
        for asset_type, model in self.models.items():
            rows = types == asset_type
            if rows.any():
                raw = -model.score_samples(self._inputs(feats.loc[rows]))
                med, span = self.calibration[asset_type]
                out[rows] = (raw - med) / span
        return out

    def _contributions(self, feats: pd.DataFrame) -> pd.DataFrame:
        return self.space._contributions(feats)

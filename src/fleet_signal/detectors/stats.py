"""Statistical baseline: robust z-scores against the train-normal envelope.

For each (asset_type, mode) the median and a robust scale of every feature are
learned from train-normal rows. A row's score is the largest |z| over its
features. Groups with too few train rows fall back to per-asset-type stats.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from fleet_signal.detectors.base import Detector, load_detector_config
from fleet_signal.features.build import FEATURE_GROUPS, scorable


def robust_scale(x: np.ndarray) -> float:
    """1.4826 * MAD, floored so near-constant or discrete features cannot explode."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    mad = 1.4826 * float(np.median(np.abs(x - np.median(x))))
    spread = float(np.quantile(x, 0.99) - np.quantile(x, 0.01)) / 6.0
    return max(mad, spread, 1e-3)


class StatsDetector(Detector):
    name = "stats"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg if cfg is not None else load_detector_config()["stats"]
        cols = [c for g in cfg["feature_groups"] for c in FEATURE_GROUPS[g]]
        self.features = [c for c in cols if c not in set(cfg.get("exclude", []))]
        self.min_group_rows = int(cfg["min_group_rows"])
        self.center: dict[tuple[str, str], np.ndarray] = {}
        self.scale: dict[tuple[str, str], np.ndarray] = {}
        self.fitted_runs: list[str] = []

    def fit(self, train: pd.DataFrame) -> StatsDetector:
        train = train[scorable(train)]
        if train.empty:
            raise ValueError("no scorable training rows")
        self.fitted_runs = sorted(train["run_id"].unique())
        for type_key, by_type in train.groupby("asset_type"):
            asset_type = str(type_key)
            x_all = by_type[self.features].to_numpy(dtype=float)
            fallback = (
                np.median(x_all, axis=0),
                np.array([robust_scale(x_all[:, j]) for j in range(x_all.shape[1])]),
            )
            self.center[(asset_type, "*")], self.scale[(asset_type, "*")] = fallback
            for mode_key, g in by_type.groupby("mode"):
                mode = str(mode_key)
                if len(g) < self.min_group_rows:
                    continue
                x = g[self.features].to_numpy(dtype=float)
                self.center[(asset_type, mode)] = np.median(x, axis=0)
                self.scale[(asset_type, mode)] = np.array(
                    [robust_scale(x[:, j]) for j in range(x.shape[1])]
                )
        return self

    def _z(self, feats: pd.DataFrame) -> np.ndarray:
        if not self.center:
            raise RuntimeError("StatsDetector used before fit()")
        x = feats[self.features].to_numpy(dtype=float)
        z = np.full_like(x, np.nan)
        groups = (
            pd.DataFrame({"t": feats["asset_type"].to_numpy(), "m": feats["mode"].to_numpy()})
            .groupby(["t", "m"])
            .indices
        )
        for raw_key, rows in groups.items():
            key = cast(tuple[str, str], raw_key)
            stat_key = key if key in self.center else (key[0], "*")
            if stat_key not in self.center:
                continue  # unknown asset type: leave NaN (no evidence either way)
            z[rows] = (x[rows] - self.center[stat_key]) / self.scale[stat_key]
        return z

    def _contributions(self, feats: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.abs(self._z(feats)), index=feats.index, columns=self.features)

    def _raw_score(self, feats: pd.DataFrame) -> np.ndarray:
        z = np.abs(self._z(feats))
        out = np.full(len(feats), np.nan)  # rows with no statistics (unknown type) stay NaN
        has = ~np.isnan(z).all(axis=1)
        out[has] = np.nanmax(z[has], axis=1)
        return out

"""v2 fast path: abrupt change against the asset's own recent trend.

Inputs are the "fast" residual features (features/build.py): the slope over the last
3 or 10 events minus the slope over the 30 events before them, for battery and
temperature. A steady trend cancels out; a drain or heating that started within the
last few events does not.

Normal behaviour still moves these residuals (a hard manoeuvre raises speed and
battery drain together; mode changes move every trend), so per (asset type, mode),
on train-normal rows only:

  1. regress each residual on the matching speed change (`spd_d3` / `spd_d10`),
  2. take the robust centre and scale of what is left,
  3. score = the largest one-sided z (battery: more drain than expected;
     temperature: hotter than expected).

Rows within `min_mode_age` events of a mode change are not applicable (their
baseline window spans two modes): they get NOT_APPLICABLE, so the fast path stays
silent there and other detectors cover them.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.stats import robust_scale
from fleet_signal.features.build import scorable

NOT_APPLICABLE = -10.0

# (feature, sign that means "worse", covariate)
SIGNALS: tuple[tuple[str, float, str], ...] = (
    ("batt_res3", -1.0, "spd_d3"),
    ("batt_res10", -1.0, "spd_d10"),
    ("temp_res3", 1.0, "spd_d3"),
    ("temp_res10", 1.0, "spd_d10"),
)


class FastPathDetector(Detector):
    name = "fast"

    def __init__(self, min_mode_age: int = 45, min_group_rows: int = 200) -> None:
        self.params: dict[str, Any] = {
            "min_mode_age": min_mode_age,
            "min_group_rows": min_group_rows,
            "signals": [s[0] for s in SIGNALS],
        }
        # (asset_type, mode | "*") -> per signal (slope, intercept, centre, scale)
        self.models: dict[tuple[str, str], np.ndarray] = {}
        self.fitted_runs: list[str] = []

    def _fit_group(self, g: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(SIGNALS), 4))
        for j, (feat, _sign, cov) in enumerate(SIGNALS):
            y, x = g[feat].to_numpy(dtype=float), g[cov].to_numpy(dtype=float)
            slope, icept = np.polyfit(x, y, 1) if np.ptp(x) > 0 else (0.0, float(np.median(y)))
            resid = y - (slope * x + icept)
            out[j] = (slope, icept, float(np.median(resid)), robust_scale(resid))
        return out

    def fit(self, train: pd.DataFrame) -> FastPathDetector:
        train = train[scorable(train) & (train["mode_age"] >= self.params["min_mode_age"])]
        if train.empty:
            raise ValueError("no scorable training rows")
        self.fitted_runs = sorted(train["run_id"].unique())
        for t_key, by_type in train.groupby("asset_type"):
            t = str(t_key)
            self.models[(t, "*")] = self._fit_group(by_type)
            for m_key, g in by_type.groupby("mode"):
                if len(g) >= self.params["min_group_rows"]:
                    self.models[(t, str(m_key))] = self._fit_group(g)
        return self

    def zscores(self, feats: pd.DataFrame) -> np.ndarray:
        """One-sided z per signal (rows x signals); NaN where not applicable."""
        if not self.models:
            raise RuntimeError("FastPathDetector used before fit()")
        z = np.full((len(feats), len(SIGNALS)), np.nan)
        keys = pd.DataFrame({"t": feats["asset_type"].to_numpy(), "m": feats["mode"].to_numpy()})
        for key, rows in keys.groupby(["t", "m"]).indices.items():
            t, m = (str(v) for v in cast(tuple[str, str], key))
            model = self.models.get((t, m), self.models.get((t, "*")))
            if model is None:
                continue
            sub = feats.iloc[rows]
            for j, (feat, sign, cov) in enumerate(SIGNALS):
                slope, icept, centre, scale = model[j]
                resid = sub[feat].to_numpy(dtype=float) - (
                    slope * sub[cov].to_numpy(dtype=float) + icept
                )
                z[rows, j] = sign * (resid - centre) / scale
        young = feats["mode_age"].to_numpy(dtype=float) < self.params["min_mode_age"]
        z[young] = np.nan
        return z

    def _contributions(self, feats: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.zscores(feats), index=feats.index, columns=[s[0] for s in SIGNALS])

    def _raw_score(self, feats: pd.DataFrame) -> np.ndarray:
        z = self.zscores(feats)
        none = np.isnan(z).all(axis=1)
        return np.where(none, NOT_APPLICABLE, np.nanmax(np.where(none[:, None], 0.0, z), axis=1))

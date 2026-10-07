"""v2 hybrid: several detectors, one alert when ANY of them is far outside normal.

Each part's score is put on a common scale using its own TRAIN-normal scores
(median -> 0, 99.9th percentile -> 1), then the hybrid score is the largest
part. One threshold on that score is then chosen on validation like any other
detector's. Evidence comes from the part that drove the score.

The calibration only makes parts comparable; it is computed on train, never on
validation or test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.detectors.base import Detector
from fleet_signal.features.build import scorable


class HybridDetector(Detector):
    def __init__(
        self, parts: list[Detector], name: str, prefit: bool = False, calib_q: float = 0.999
    ) -> None:
        self.parts = parts
        self.name = name
        self.prefit = prefit
        self.calibration: dict[str, tuple[float, float]] = {}
        self.params: dict[str, Any] = {
            "parts": [p.name for p in parts],
            "part_params": [getattr(p, "params", None) for p in parts],
            "calib_q": calib_q,
        }
        rules = [getattr(p, "rules", None) for p in parts if getattr(p, "rules", None)]
        self.rules = rules[0] if rules else None

    @property
    def fitted_runs(self) -> list[str]:
        runs: set[str] = set()
        for p in self.parts:
            runs |= set(getattr(p, "fitted_runs", []))
        return sorted(runs)

    def fit(self, train: pd.DataFrame) -> HybridDetector:
        if not self.prefit:
            for p in self.parts:
                p.fit(train)
        rows = train[scorable(train)]
        for p in self.parts:
            s = p.score(rows)
            s = s[np.isfinite(s)]
            med, hi = float(np.median(s)), float(np.quantile(s, self.params["calib_q"]))
            self.calibration[p.name] = (med, max(hi - med, 1e-9))
        self.prefit = False  # a pickled hybrid always refits its parts if fitted again
        return self

    def _normalised(self, feats: pd.DataFrame) -> np.ndarray:
        if not self.calibration:
            raise RuntimeError("HybridDetector used before fit()")
        cols = []
        for p in self.parts:
            med, span = self.calibration[p.name]
            cols.append((p.score(feats) - med) / span)
        return np.column_stack(cols)

    def _raw_score(self, feats: pd.DataFrame) -> np.ndarray:
        n = self._normalised(feats)
        none = np.isnan(n).all(axis=1)
        return np.where(none, np.nan, np.nanmax(np.where(none[:, None], -np.inf, n), axis=1))

    def _contributions(self, feats: pd.DataFrame) -> pd.DataFrame:
        cols = self.params["parts"]
        return pd.DataFrame(self._normalised(feats), index=feats.index, columns=cols)

    def evidence(self, feats: pd.DataFrame, k: int = 3) -> list[list[dict[str, Any]]]:
        """Evidence of the part with the highest normalised score, prefixed with its name."""
        result: list[list[dict[str, Any]]] = [[] for _ in range(len(feats))]
        ok = self._usable(feats)
        if not ok.any():
            return result
        n = self._normalised(feats.loc[ok])
        lead = np.argmax(np.where(np.isnan(n), -np.inf, n), axis=1)
        positions = np.flatnonzero(ok)
        for j, part in enumerate(self.parts):
            mine = positions[lead == j]
            if len(mine) == 0:
                continue
            ev = part.evidence(feats.iloc[mine], k=k)
            for pos, items in zip(mine, ev, strict=True):
                result[pos] = [{**e, "signal": f"{part.name}:{e['signal']}"} for e in items]
        return result

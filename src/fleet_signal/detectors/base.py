"""Common detector interface.

A detector turns feature rows into one anomaly score per row (higher = more
anomalous) and can explain a row with its top contributing signals. Rows that
are not scorable (too little history, gaps, non-finite features) get NaN, never
a low "normal-looking" score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fleet_signal.data.config import REPO_ROOT
from fleet_signal.features.build import scorable

DEFAULT_DETECTOR_CONFIG = REPO_ROOT / "configs" / "detectors.yaml"


def load_detector_config(path: Path = DEFAULT_DETECTOR_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


class Detector(ABC):
    name: str = "detector"

    def fit(self, train: pd.DataFrame) -> Detector:
        """Learn from train-normal feature rows. Default: nothing to learn."""
        return self

    @abstractmethod
    def _raw_score(self, feats: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def _contributions(self, feats: pd.DataFrame) -> pd.DataFrame:
        """Per-row, per-signal contribution (same index as feats), larger = more anomalous."""

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        ok = scorable(feats).to_numpy()
        out = np.full(len(feats), np.nan)
        if ok.any():
            out[ok] = self._raw_score(feats.loc[ok])
        return out

    def evidence(self, feats: pd.DataFrame, k: int = 3) -> list[list[dict[str, float]]]:
        """Top-k contributing signals per row (empty list for unscorable rows)."""
        ok = scorable(feats).to_numpy()
        result: list[list[dict[str, float]]] = [[] for _ in range(len(feats))]
        if not ok.any():
            return result
        contrib = self._contributions(feats.loc[ok])
        values = contrib.to_numpy()
        names = contrib.columns.to_numpy()
        top = np.argsort(-np.nan_to_num(values, nan=-np.inf), axis=1)[:, :k]
        for row_pos, (orig, idx) in enumerate(zip(np.flatnonzero(ok), top, strict=True)):
            # The strongest signal always, then only others that push towards "anomalous"
            # (contribution >= 0). For the rule detector a negative value means "below its
            # written limit", which is not evidence.
            result[orig] = [
                {"signal": str(names[j]), "contribution": float(values[row_pos, j])}
                for rank, j in enumerate(idx)
                if np.isfinite(values[row_pos, j]) and (rank == 0 or values[row_pos, j] >= 0)
            ]
        return result

"""Transparent rule baseline: hand-written limits on a few physical signals.

Each rule compares one feature against a limit for the asset type (and mode,
for battery). Its score is the exceedance in units of `scale`, so a value of 0
is exactly at the written limit. The detector score is the worst rule.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.detectors.base import Detector, load_detector_config

NOT_APPLICABLE = -10.0  # score when no rule applies to a row


class RuleDetector(Detector):
    name = "rule"

    def __init__(self, rules: list[dict[str, Any]] | None = None) -> None:
        self.rules = rules if rules is not None else load_detector_config()["rule"]["rules"]

    def _limit(self, rule: dict[str, Any], feats: pd.DataFrame) -> np.ndarray:
        types = feats["asset_type"].to_numpy()
        if "limit_by_mode" in rule:
            modes = feats["mode"].to_numpy()
            table = rule["limit_by_mode"]
            return np.array(
                [table.get(t, {}).get(m, np.nan) for t, m in zip(types, modes, strict=True)],
                dtype=float,
            )
        limit = rule["limit"]
        if isinstance(limit, dict):
            return np.array([limit.get(t, np.nan) for t in types], dtype=float)
        return np.full(len(feats), float(limit))

    def _applies(self, rule: dict[str, Any], feats: pd.DataFrame) -> np.ndarray:
        ok = np.ones(len(feats), dtype=bool)
        when = rule.get("when", {})
        if "mode_in" in when:
            ok &= feats["mode"].isin(when["mode_in"]).to_numpy()
        if "mode_age_min" in when:
            ok &= feats["mode_age"].to_numpy() >= when["mode_age_min"]
        if "link_below" in when:
            ok &= feats["link_pct"].to_numpy() < when["link_below"]
        return ok

    def _contributions(self, feats: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, np.ndarray] = {}
        for rule in self.rules:
            x = feats[rule["feature"]].to_numpy(dtype=float)
            limit = self._limit(rule, feats)
            scale = np.abs(limit) if rule["scale"] == "relative" else float(rule["scale"])
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (x - limit) / scale if rule["op"] == ">" else (limit - x) / scale
            s = np.where(self._applies(rule, feats) & np.isfinite(s), s, np.nan)
            out[rule["name"]] = s
        return pd.DataFrame(out, index=feats.index)

    @property
    def known_asset_types(self) -> set[str]:
        types: set[str] = set()
        for rule in self.rules:
            table = rule.get("limit_by_mode", rule.get("limit"))
            if isinstance(table, dict):
                types |= set(table)
        return types

    def _raw_score(self, feats: pd.DataFrame) -> np.ndarray:
        c = self._contributions(feats).to_numpy()
        all_nan = np.isnan(c).all(axis=1)
        best = np.nanmax(np.where(all_nan[:, None], NOT_APPLICABLE, c), axis=1)
        # An asset type the rules were never written for gets no score (never "normal").
        known = self.known_asset_types
        if known:
            unknown = ~feats["asset_type"].isin(known).to_numpy()
            best = np.where(unknown, np.nan, best)
        return best

"""Incident prioritisation: a transparent severity score for each incident when it opens.

    severity = 0.40 * urgency + 0.35 * strength + 0.25 * breadth          (each 0..1)
    P1 if severity >= 0.60,  P2 if >= 0.35,  else P3

* **urgency**: time-to-critical at the current trend, the soonest of
    battery  (battery_pct - 10 %)   / drain rate      (10 % is below every asset's
                                                        return-to-base level, 25-30 %)
    temp     (70 degC - temp_c)     / heating rate    (70 degC is ~15 degC above the
                                                        hottest train-normal reading)
    link     (link_pct - 20 %)      / decline rate    (20 % is the rule's dropout limit)
  using the 30-event slopes. urgency = 1 if <= 5 min, 0.6 if <= 15, 0.3 if <= 60, else 0.
  Already past a critical level counts as 0 minutes.
* **strength** (sustained anomaly strength): mean of score / threshold over the last
  10 scored events up to the open, mapped from [1, 3] to [0, 1].
* **breadth**: how many signal families are anomalous at the open (battery,
  temperature, link, motion/position, frozen field), counted from the shipped
  detector's own parts (rule margins > 0, fast-path z beyond its alert level),
  capped at 3 and divided by 3.

The weights and cut-offs are hand-set and stated here so an operator can recompute
any severity by hand. They were checked post-hoc on the v2 test (do true incidents
rank above false ones?), not tuned on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CRITICAL = {"battery": 10.0, "temp": 70.0, "link": 20.0}
WEIGHTS = {"urgency": 0.40, "strength": 0.35, "breadth": 0.25}
LEVELS = ((0.60, "P1"), (0.35, "P2"))
RULE_FAMILY = {
    "temp_rise": "temperature", "battery_drain": "battery", "link_dropout": "link",
    "link_unstable": "link", "frozen_temperature": "frozen", "frozen_battery": "frozen",
    "frozen_link": "frozen", "frozen_speed": "frozen", "frozen_position": "frozen",
    "speed_position_mismatch": "motion", "position_jump": "motion",
}  # fmt: skip
FAST_FAMILY = {"batt_res3": "battery", "batt_res10": "battery",
               "temp_res3": "temperature", "temp_res10": "temperature"}  # fmt: skip
FAST_ALERT_Z = 5.0  # a fast-path signal counts toward breadth beyond this z


@dataclass
class Severity:
    severity: float
    level: str
    urgency: float
    strength: float
    breadth: float
    time_to_critical_min: float | None
    critical_signal: str | None
    families: list[str]

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def time_to_critical(row: pd.Series) -> tuple[float | None, str | None]:
    """Minutes until the first signal reaches its critical level at the current trend."""
    cands: list[tuple[float, str]] = []
    batt, drain = float(row["battery_pct"]), -float(row["batt_slope_m"])
    if batt <= CRITICAL["battery"]:
        cands.append((0.0, "battery"))
    elif drain > 0:
        cands.append(((batt - CRITICAL["battery"]) / drain, "battery"))
    temp, heat = float(row["temp_c"]), float(row["temp_slope_m"])
    if temp >= CRITICAL["temp"]:
        cands.append((0.0, "temperature"))
    elif heat > 0:
        cands.append(((CRITICAL["temp"] - temp) / heat, "temperature"))
    link, decline = float(row["link_pct"]), -float(row["link_slope_m"])
    if link <= CRITICAL["link"]:
        cands.append((0.0, "link"))
    elif decline > 0:
        cands.append(((link - CRITICAL["link"]) / decline, "link"))
    if not cands:
        return None, None
    m, sig = min(cands)
    return float(m), sig


def _urgency(ttc: float | None) -> float:
    if ttc is None:
        return 0.0
    return 1.0 if ttc <= 5 else 0.6 if ttc <= 15 else 0.3 if ttc <= 60 else 0.0


def families(detector: Any, row: pd.DataFrame) -> list[str]:
    """Signal families anomalous at this row, from the detector's own parts."""
    parts = getattr(detector, "parts", [detector])
    fams: set[str] = set()
    for p in parts:
        if p.name == "rule":
            c = p._contributions(row).iloc[0]
            fams |= {RULE_FAMILY[k] for k, v in c.items() if np.isfinite(v) and v > 0}
        elif p.name == "fast":
            z = p._contributions(row).iloc[0]
            fams |= {FAST_FAMILY[k] for k, v in z.items() if np.isfinite(v) and v > FAST_ALERT_Z}
    return sorted(fams)


def severity(
    detector: Any, threshold: float, feats: pd.DataFrame, scores: np.ndarray, open_pos: int
) -> Severity:
    """Severity of an incident opening at position `open_pos` of one asset's feature rows."""
    row = feats.iloc[[open_pos]]
    ttc, sig = time_to_critical(row.iloc[0])
    recent = scores[max(0, open_pos - 9) : open_pos + 1]
    recent = recent[np.isfinite(recent)]
    if len(recent) == 0 or threshold == 0:
        strength = 0.0
    else:
        ratio = (
            float(np.mean(recent)) / abs(threshold)
            if threshold > 0
            else 1.0 + float(np.mean(recent) - threshold)
        )
        strength = float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))
    fams = families(detector, row)
    breadth = min(len(fams), 3) / 3.0
    u = _urgency(ttc)
    s = WEIGHTS["urgency"] * u + WEIGHTS["strength"] * strength + WEIGHTS["breadth"] * breadth
    level = next((name for cut, name in LEVELS if s >= cut), "P3")
    return Severity(round(s, 3), level, u, round(strength, 3), round(breadth, 3),
                    None if ttc is None else round(ttc, 1), sig, fams)  # fmt: skip

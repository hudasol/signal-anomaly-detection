"""Window features from raw telemetry.

One function, `build_features`, turns telemetry events into one feature row per
event, computed only from that asset's own past events in the same run
(trailing windows). It is used unchanged at training, evaluation and inference
time.

Rules this module guarantees (and tests check):

* windows never cross a run or an asset boundary;
* no feature looks at future events;
* nothing here reads ground truth: the input is telemetry only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fleet_signal.data.config import REPO_ROOT

DEFAULT_FEATURE_CONFIG = REPO_ROOT / "configs" / "features.yaml"

KEY_COLUMNS: tuple[str, ...] = ("run_id", "asset_id", "asset_type", "seq", "timestamp_utc", "mode")
MODES: tuple[str, ...] = ("idle", "moving", "returning", "charging")
ASSET_TYPES: tuple[str, ...] = ("drone", "rover", "quadruped")

# Feature groups, named so an ablation can drop one group at a time.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "level": ("temp_c", "battery_pct", "link_pct", "speed_mps", "range_m", "alt_m"),
    "trend": (
        "temp_slope_s",
        "temp_slope_m",
        "temp_slope_l",
        "batt_slope_s",
        "batt_slope_m",
        "batt_slope_l",
        "link_slope_m",
        "link_slope_l",
        "range_slope_m",
    ),
    "volatility": ("temp_std_m", "link_std_m", "link_min_m", "link_absdiff_s"),
    "freeze": (
        "temp_unchanged",
        "batt_unchanged",
        "link_unchanged",
        "speed_unchanged",
        "pos_unchanged",
    ),
    "motion": ("implied_speed", "speed_mismatch", "jump_excess"),
    "context": (
        "mode_age",
        *(f"mode_{m}" for m in MODES),
        *(f"type_{t}" for t in ASSET_TYPES),
    ),
}
FEATURE_COLUMNS: tuple[str, ...] = tuple(c for cols in FEATURE_GROUPS.values() for c in cols)
STATUS_COLUMNS: tuple[str, ...] = ("history", "history_ok", "max_gap_l", "gap_ok")


@dataclass(frozen=True)
class FeatureConfig:
    feature_version: str
    short: int
    mid: int
    long: int
    min_history: int
    max_gap: int
    unchanged_cap: int
    mismatch_median: int

    @classmethod
    def load(cls, path: Path = DEFAULT_FEATURE_CONFIG) -> FeatureConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            feature_version=raw["feature_version"],
            short=raw["windows"]["short"],
            mid=raw["windows"]["mid"],
            long=raw["windows"]["long"],
            min_history=raw["min_history"],
            max_gap=raw["max_gap"],
            unchanged_cap=raw["unchanged_cap"],
            mismatch_median=raw["mismatch_median"],
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @property
    def schema_hash(self) -> str:
        """Identifies config + feature list + this source file. Stored with every model."""
        h = hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode())
        h.update(json.dumps(FEATURE_COLUMNS).encode())
        h.update(Path(__file__).read_bytes())
        return h.hexdigest()[:10]


# ---------------------------------------------------------------- primitives


def rolling_slope(t: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """OLS slope of y on t over the trailing w points (inclusive). NaN until w points exist."""
    n = len(y)
    out = np.full(n, np.nan)
    if n < w:
        return out
    t = t - t[0]
    c = lambda a: np.concatenate(([0.0], np.cumsum(a)))  # noqa: E731
    st, sy, stt, sty = c(t), c(y), c(t * t), c(t * y)
    sx = st[w:] - st[:-w]
    sxx = stt[w:] - stt[:-w]
    sxy = sty[w:] - sty[:-w]
    sy_w = sy[w:] - sy[:-w]
    denom = w * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        out[w - 1 :] = (w * sxy - sx * sy_w) / denom
    return out


def events_since_change(values: np.ndarray, cap: int) -> np.ndarray:
    """0 when the value just changed (or at the first event), k after k identical repeats."""
    n = len(values)
    if n == 0:
        return np.zeros(0)
    changed = np.ones(n, dtype=bool)
    changed[1:] = values[1:] != values[:-1]
    idx = np.arange(n)
    last_change = np.maximum.accumulate(np.where(changed, idx, 0))
    return np.minimum(idx - last_change, cap).astype(float)


def _roll(s: pd.Series, w: int, fn: str) -> np.ndarray:
    return getattr(s.rolling(w, min_periods=w), fn)().to_numpy()


# ---------------------------------------------------------------- per asset-run


def _features_one(g: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    g = g.sort_values("seq")
    t = g["seq"].to_numpy(dtype=float)
    temp = g["temperature_c"].to_numpy(dtype=float)
    batt = g["battery_pct"].to_numpy(dtype=float)
    link = g["link_quality_pct"].to_numpy(dtype=float)
    speed = g["speed_mps"].to_numpy(dtype=float)
    x, y, z = (g[c].to_numpy(dtype=float) for c in ("x_m", "y_m", "z_m"))
    mode = g["mode"].to_numpy()
    n = len(g)
    s, m, ln = cfg.short, cfg.mid, cfg.long
    per_min = 60.0  # slopes are reported per minute

    f: dict[str, np.ndarray] = {}
    rng_m = np.hypot(x, y)
    f["temp_c"], f["battery_pct"], f["link_pct"] = temp, batt, link
    f["speed_mps"], f["range_m"], f["alt_m"] = speed, rng_m, z

    f["temp_slope_s"] = rolling_slope(t, temp, s) * per_min
    f["temp_slope_m"] = rolling_slope(t, temp, m) * per_min
    f["temp_slope_l"] = rolling_slope(t, temp, ln) * per_min
    f["batt_slope_s"] = rolling_slope(t, batt, s) * per_min
    f["batt_slope_m"] = rolling_slope(t, batt, m) * per_min
    f["batt_slope_l"] = rolling_slope(t, batt, ln) * per_min
    f["link_slope_m"] = rolling_slope(t, link, m) * per_min
    f["link_slope_l"] = rolling_slope(t, link, ln) * per_min
    f["range_slope_m"] = rolling_slope(t, rng_m, m) * per_min

    temp_s, link_s = pd.Series(temp), pd.Series(link)
    f["temp_std_m"] = _roll(temp_s, m, "std")
    f["link_std_m"] = _roll(link_s, m, "std")
    f["link_min_m"] = _roll(link_s, m, "min")
    absdiff = pd.Series(np.abs(np.diff(link, prepend=np.nan)))
    f["link_absdiff_s"] = _roll(absdiff, s, "mean")

    cap = cfg.unchanged_cap
    f["temp_unchanged"] = events_since_change(temp, cap)
    f["batt_unchanged"] = events_since_change(batt, cap)
    f["link_unchanged"] = events_since_change(link, cap)
    f["speed_unchanged"] = events_since_change(speed, cap)
    pos_key = x * 1e6 + y  # unchanged only if both x and y repeat
    f["pos_unchanged"] = events_since_change(pos_key, cap)

    dt = np.diff(t, prepend=np.nan)
    step = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
    implied = step / dt
    f["implied_speed"] = implied
    k = cfg.mismatch_median
    f["speed_mismatch"] = _roll(pd.Series(np.abs(speed - implied)), k, "median")
    # Distance moved beyond what the reported speed allows, worst of the last k steps.
    excess = pd.Series(step - speed * dt)
    f["jump_excess"] = _roll(excess, k, "max")

    mode_changed = np.ones(n, dtype=bool)
    mode_changed[1:] = mode[1:] != mode[:-1]
    idx = np.arange(n)
    last_mode_change = np.maximum.accumulate(np.where(mode_changed, t, -np.inf)).astype(float)
    f["mode_age"] = np.minimum(t - last_mode_change, 600.0)
    for md in MODES:
        f[f"mode_{md}"] = (mode == md).astype(float)
    asset_type = g["asset_type"].iloc[0]
    for at in ASSET_TYPES:
        f[f"type_{at}"] = np.full(n, float(asset_type == at))

    history = idx + 1
    gaps = pd.Series(dt).fillna(1.0)
    max_gap = _roll(gaps, ln, "max")
    status = {
        "history": history,
        "history_ok": history >= cfg.min_history,
        "max_gap_l": max_gap,
        "gap_ok": np.nan_to_num(max_gap, nan=np.inf) <= cfg.max_gap,
    }

    out = g[list(KEY_COLUMNS)].reset_index(drop=True)
    feats = pd.DataFrame(f)[list(FEATURE_COLUMNS)]
    return pd.concat([out, feats, pd.DataFrame(status)], axis=1)


def build_features(tel: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Feature rows for every event. Each (run_id, asset_id) is processed independently."""
    cfg = cfg or FeatureConfig.load()
    missing = {
        "run_id",
        "asset_id",
        "asset_type",
        "seq",
        "temperature_c",
        "battery_pct",
        "link_quality_pct",
        "speed_mps",
        "x_m",
        "y_m",
        "z_m",
        "mode",
    } - set(tel.columns)
    if missing:
        raise ValueError(f"telemetry is missing columns: {sorted(missing)}")
    parts = [_features_one(g, cfg) for _, g in tel.groupby(["run_id", "asset_id"], sort=True)]
    if not parts:
        return pd.DataFrame(columns=[*KEY_COLUMNS, *FEATURE_COLUMNS, *STATUS_COLUMNS])
    return pd.concat(parts, ignore_index=True)


def scorable(features: pd.DataFrame) -> pd.Series:
    """Rows a detector may score: enough history, no long gap, all features finite."""
    finite = np.isfinite(features[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    return features["history_ok"] & features["gap_ok"] & finite

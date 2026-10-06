"""Input validation at the inference boundary.

Two layers, so a bad input can never be scored as "normal":

1. **Contract** (HTTP only, pydantic, answered with 422): field types, known
   `mode` / `asset_type` values (exact, case-sensitive), integer `seq >= 0`,
   one asset per request, `seq` strictly increasing, bounded request size.
   A request that breaks the contract is malformed; it is rejected, not scored.

2. **Window checks** (`check_window`, used by the scorer for HTTP *and* replay,
   answered with status `degraded`): the same contract checks on a DataFrame, plus
   physical plausibility (battery and link in 0-100, finite numbers, speed and
   temperature inside generous physical limits) and, when every event carries a
   timestamp, that timestamps advance at the 1 Hz rate the features assume.
   An impossible reading is a data-quality problem the operator should see,
   so it is reported, never scored.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, Strict, model_validator

from fleet_signal.features.build import ASSET_TYPES, MODES

MAX_EVENTS = 1000  # 650 needed; headroom for clients that send a little more
MAX_BODY_BYTES = 2_000_000  # ~1000 events with room to spare
MAX_SEQ = 10**12

# Physical plausibility limits. Generous on purpose: they reject impossible
# readings, not unusual ones (unusual is the detector's job).
LIMITS: dict[str, tuple[float, float]] = {
    "battery_pct": (0.0, 100.0),
    "link_quality_pct": (0.0, 100.0),
    "temperature_c": (-60.0, 200.0),
    "speed_mps": (0.0, 100.0),
    "x_m": (-1e5, 1e5),
    "y_m": (-1e5, 1e5),
    "z_m": (-1e3, 1e4),
}
NUMERIC = ("x_m", "y_m", "z_m", "speed_mps", "battery_pct", "temperature_c", "link_quality_pct")
TICK_S = 1.0  # features assume one event per second
TICK_TOLERANCE_S = 0.25  # per step; generated telemetry is exactly 1.0 s

# Strict: no bools, no numeric strings, no NaN/inf. Ints are accepted for floats.
Num = Annotated[float, Strict(), Field(allow_inf_nan=False)]
Seq = Annotated[int, Strict(), Field(ge=0, le=MAX_SEQ)]
Mode = Literal["idle", "moving", "returning", "charging"]
AssetType = Literal["drone", "rover", "quadruped"]

if set(MODES) != {"idle", "moving", "returning", "charging"} or set(ASSET_TYPES) != {
    "drone",
    "rover",
    "quadruped",
}:  # the Literal types above must match the feature code
    raise RuntimeError("validation.Mode / AssetType are out of sync with features.build")


class Position(BaseModel):
    model_config = ConfigDict(extra="ignore")
    x_m: Num
    y_m: Num
    z_m: Num


class Event(BaseModel):
    """One telemetry event: flat Signal form, or Blackbox form with a nested `position`."""

    model_config = ConfigDict(extra="ignore")
    asset_id: Annotated[str, Strict(), Field(min_length=1, max_length=128)]
    asset_type: AssetType
    seq: Seq
    mode: Mode
    speed_mps: Num
    battery_pct: Num
    temperature_c: Num
    link_quality_pct: Num
    x_m: Num | None = None
    y_m: Num | None = None
    z_m: Num | None = None
    position: Position | None = None
    timestamp_utc: str | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def _position(self) -> Event:
        if self.position is not None:
            self.x_m, self.y_m, self.z_m = self.position.x_m, self.position.y_m, self.position.z_m
        if self.x_m is None or self.y_m is None or self.z_m is None:
            raise ValueError("position required: x_m, y_m, z_m (flat) or `position` (Blackbox)")
        return self

    def flat(self) -> dict[str, Any]:
        return self.model_dump(exclude={"position"}, exclude_none=True)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[Event] = Field(
        ...,
        min_length=1,
        max_length=MAX_EVENTS,
        description=(
            "One asset's telemetry, oldest first, seq strictly increasing. Flat Signal events "
            "or Blackbox-contract events (nested `position`). Send 650 events, or every event "
            f"since the run started. At most {MAX_EVENTS}."
        ),
    )

    @model_validator(mode="after")
    def _one_asset_in_order(self) -> ScoreRequest:
        ids = {e.asset_id for e in self.events}
        types = {e.asset_type for e in self.events}
        if len(ids) != 1 or len(types) != 1:
            raise ValueError("a request must contain exactly one asset_id and one asset_type")
        seqs = [e.seq for e in self.events]
        if any(b <= a for a, b in zip(seqs, seqs[1:], strict=False)):
            raise ValueError("seq must be strictly increasing (no duplicates, oldest first)")
        return self


# ---------------------------------------------------------------- window checks


def _is_int_like(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return False
    v = s.to_numpy(dtype=float)
    return bool(np.isfinite(v).all() and (v == np.floor(v)).all())


def check_window(df: pd.DataFrame) -> str | None:
    """Return a short reason if the window must not be scored, else None.

    Expects the columns of `REQUIRED` in the scorer to be present.
    """
    if df["asset_id"].isna().any() or df["asset_id"].nunique(dropna=False) != 1:
        return "a window must contain exactly one asset_id"
    if df["asset_type"].isna().any() or df["asset_type"].nunique(dropna=False) != 1:
        return "a window must contain exactly one asset_type"
    if not set(df["asset_type"]) <= set(ASSET_TYPES):
        return f"unknown asset_type; expected one of {list(ASSET_TYPES)}"
    if df["mode"].isna().any() or not set(df["mode"]) <= set(MODES):
        return f"unknown mode; expected one of {list(MODES)} (case-sensitive)"
    if not _is_int_like(df["seq"]):
        return "seq must be integers"
    seq = df["seq"].to_numpy(dtype=float)
    if (seq < 0).any() or (seq > MAX_SEQ).any():
        return "seq out of range"
    if (np.diff(seq) <= 0).any():
        return "seq must be strictly increasing (no duplicates, oldest first)"
    for col in NUMERIC:
        s = df[col]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            return f"{col} must be numeric"
        v = s.to_numpy(dtype=float)
        if not np.isfinite(v).all():
            return f"{col} has missing or non-finite values"
        lo, hi = LIMITS[col]
        bad = (v < lo) | (v > hi)
        if bad.any():
            at = int(df["seq"].iloc[int(np.flatnonzero(bad)[0])])
            return f"{col} outside physical range [{lo:g}, {hi:g}] at seq {at}"
    return _check_ticks(df)


def _check_ticks(df: pd.DataFrame) -> str | None:
    """If every event has a timestamp, they must advance 1 s per seq step (features assume it)."""
    if "timestamp_utc" not in df:
        return None
    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if ts.isna().all():
        return None  # no timestamps sent: seq is the clock
    if ts.isna().any():
        return "timestamp_utc missing or unparseable on some events"
    dt = np.diff(ts.astype("int64").to_numpy()) / 1e9
    dseq = np.diff(df["seq"].to_numpy(dtype=float)) * TICK_S
    if len(dt) and (np.abs(dt - dseq) > TICK_TOLERANCE_S).any():
        return "timestamps do not advance at 1 Hz per seq step; features assume 1 Hz telemetry"
    return None

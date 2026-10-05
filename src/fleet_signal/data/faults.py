"""Fault scenarios: sampling parameters and applying them during simulation.

Faults come in two kinds:

* Physical faults (overheating, battery_drain, link_degradation) change what
  the asset actually experiences: extra heat, extra drain, worse radio.
* Sensor faults (sensor_freeze, motion_anomaly) leave the asset's true
  state alone and corrupt what it reports.

All randomness for a fault is drawn once, at sampling time, from a dedicated
RNG stream. The simulation of normal behaviour never consumes it, so before
the fault starts a fault run is identical to the same seed's normal run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fleet_signal.data.config import GenerationConfig

SCALAR_SENSOR_FIELDS: tuple[str, ...] = (
    "temperature_c",
    "battery_pct",
    "link_quality_pct",
    "speed_mps",
)
POSITION_FIELDS: tuple[str, ...] = ("x_m", "y_m", "z_m")


def _u(rng: np.random.Generator, lo_hi: list[float] | tuple[float, float]) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _i(rng: np.random.Generator, lo_hi: list[int] | tuple[int, int]) -> int:
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


@dataclass
class Fault:
    """One injected fault on one asset. `start`/`end` are tick indices, end exclusive."""

    fault_type: str
    variant: str
    asset_index: int
    progressive: bool
    planned_onset: int
    trigger_on_motion: bool
    trigger_timeout: int
    duration: int | None  # None = persists to end of run
    n_ticks: int
    params: dict[str, Any]
    start: int | None = None
    end: int | None = None
    _frozen: dict[str, float] = field(default_factory=dict)

    # ---- lifecycle -------------------------------------------------------

    def maybe_start(self, t: int, asset_is_moving: bool) -> None:
        if self.start is not None or t < self.planned_onset:
            return
        # Sensor faults wait for motion (a frozen speed on a parked asset is invisible and
        # would be a meaningless label), but never past the timeout or 80% of the run.
        deadline = min(self.planned_onset + self.trigger_timeout, int(0.8 * self.n_ticks))
        if not self.trigger_on_motion or asset_is_moving or t >= deadline:
            self.start = t
            end = self.n_ticks if self.duration is None else t + self.duration
            self.end = min(end, self.n_ticks)

    def active(self, t: int) -> bool:
        return self.start is not None and self.start <= t < self.end  # type: ignore[operator]

    def _minutes(self, t: int) -> float:
        return (t - self.start) / 60.0  # type: ignore[operator]

    # ---- physical effects ---------------------------------------------------

    def temp_offset_c(self, t: int) -> float:
        if self.fault_type != "overheating" or not self.active(t):
            return 0.0
        m = self._minutes(t)
        rate = self.params["rate_c_per_min"]
        offset = rate * m
        if self.variant == "runaway":
            offset += rate * self.params["runaway_coeff"] * m * m
        return min(offset, self.params["cap_c"])

    def extra_drain_pct_per_s(self, t: int) -> float:
        if self.fault_type != "battery_drain" or not self.active(t):
            return 0.0
        per_min = self.params["extra_pct_per_min"]
        if self.variant == "accelerating":
            per_min += self.params["accel_pct_per_min2"] * self._minutes(t)
        return per_min / 60.0

    def modify_link(self, t: int, lq: float, jitter: float) -> float:
        """`jitter` is a non-negative noise value supplied by the caller's normal stream."""
        if self.fault_type != "link_degradation" or not self.active(t):
            return lq
        p = self.params
        if self.variant == "decline":
            floor = p["dropout_floor"] + jitter
            return max(lq - p["decline_pct_per_min"] * self._minutes(t), floor)
        if self.variant == "oscillation":
            k = t - self.start  # type: ignore[operator]
            growth = min(1.0, k / 60.0)
            phase = 0.5 + 0.5 * math.sin(2 * math.pi * k / p["osc_period_s"] - math.pi / 2)
            return lq - p["osc_amp_pct"] * growth * phase
        # intermittent: slow decline plus scheduled short dropouts
        k = t - self.start  # type: ignore[operator]
        lq = lq - 0.3 * p["decline_pct_per_min"] * self._minutes(t)
        for s, e, floor in p["dropouts"]:
            if s <= k < e:
                return floor + jitter
        return lq

    # ---- sensor effects -------------------------------------------------------

    def corrupt_report(self, t: int, report: dict[str, float]) -> None:
        """Mutate a reported event in place."""
        if not self.active(t):
            return
        if self.fault_type == "sensor_freeze":
            for name in self.params["fields"]:
                if name not in self._frozen:
                    self._frozen[name] = report[name]
                report[name] = self._frozen[name]
        elif self.fault_type == "motion_anomaly":
            k = t - self.start  # type: ignore[operator]
            if self.variant == "jump":
                report["x_m"] = round(report["x_m"] + self.params["dx"], 2)
                report["y_m"] = round(report["y_m"] + self.params["dy"], 2)
            elif self.variant == "drift":
                report["x_m"] = round(report["x_m"] + self.params["vx"] * k, 2)
                report["y_m"] = round(report["y_m"] + self.params["vy"] * k, 2)
            elif self.variant == "speed_mismatch":
                report["speed_mps"] = round(report["speed_mps"] * self.params["factor"], 2)

    def public_params(self) -> dict[str, Any]:
        """Parameters for the ground-truth record (JSON-serialisable)."""
        return {k: v for k, v in self.params.items() if k != "dropouts"} | (
            {"n_dropouts": len(self.params["dropouts"])} if "dropouts" in self.params else {}
        )


def sample_fault(
    fault_type: str,
    variant: str,
    asset_index: int,
    cfg: GenerationConfig,
    range_set: str,
    rng: np.random.Generator,
) -> Fault:
    """Draw a fault's onset and parameters from the configured ranges.

    The variant is chosen by the run plan (balanced per split), not here.
    """
    fcfg = cfg.data["faults"]
    spec = fcfg[fault_type]
    n_ticks = cfg.n_ticks
    tick_hz = cfg.data["tick_hz"]

    if variant not in spec["variants"]:
        raise ValueError(f"unknown variant {variant!r} for {fault_type}")
    r = spec["ranges"][range_set]
    onset = int(_u(rng, fcfg["onset_frac"]) * n_ticks)
    params: dict[str, Any] = {}
    duration: int | None = None

    if fault_type == "overheating":
        params = {"rate_c_per_min": _u(rng, r["rate_c_per_min"]), "cap_c": r["cap_c"]}
        if variant == "runaway":
            params["runaway_coeff"] = _u(rng, r["runaway_coeff"])
    elif fault_type == "battery_drain":
        params = {"extra_pct_per_min": _u(rng, r["extra_pct_per_min"])}
        if variant == "accelerating":
            params["accel_pct_per_min2"] = _u(rng, r["accel_pct_per_min2"])
    elif fault_type == "link_degradation":
        params = {
            "decline_pct_per_min": _u(rng, r["decline_pct_per_min"]),
            "dropout_floor": _u(rng, r["dropout_floor"]),
        }
        if variant == "oscillation":
            params["osc_amp_pct"] = _u(rng, r["osc_amp_pct"])
            params["osc_period_s"] = _u(rng, r["osc_period_s"])
            duration = _i(rng, r["osc_duration_s"]) * tick_hz
        elif variant == "intermittent":
            dropouts: list[tuple[int, int, float]] = []
            k = _i(rng, r["intermittent_gap_s"])
            while k < n_ticks:
                length = _i(rng, r["intermittent_len_s"])
                dropouts.append((k, k + length, _u(rng, (0.0, 10.0))))
                k += length + _i(rng, r["intermittent_gap_s"])
            params["dropouts"] = dropouts
    elif fault_type == "sensor_freeze":
        if variant == "position":
            fields = list(POSITION_FIELDS)
        elif variant == "multi":
            idx = rng.choice(len(SCALAR_SENSOR_FIELDS), size=2, replace=False)
            fields = sorted(SCALAR_SENSOR_FIELDS[int(i)] for i in idx)
        else:
            fields = [variant]
        params = {"fields": fields}
        duration = _i(rng, r["duration_s"]) * tick_hz
    elif fault_type == "motion_anomaly":
        duration = _i(rng, r["duration_s"]) * tick_hz
        angle = _u(rng, (0.0, 2 * math.pi))
        if variant == "jump":
            mag = _u(rng, r["jump_m"])
            params = {"jump_m": mag, "dx": mag * math.cos(angle), "dy": mag * math.sin(angle)}
        elif variant == "drift":
            v = _u(rng, r["drift_mps"]) / tick_hz
            params = {"drift_mps": v, "vx": v * math.cos(angle), "vy": v * math.sin(angle)}
        else:
            low = rng.random() < 0.5
            factor = _u(rng, r["mismatch_low"] if low else r["mismatch_high"])
            params = {"factor": factor}
    else:
        raise ValueError(f"unknown fault type: {fault_type}")

    return Fault(
        fault_type=fault_type,
        variant=variant,
        asset_index=asset_index,
        progressive=bool(spec["progressive"]),
        planned_onset=onset,
        trigger_on_motion=fault_type in ("sensor_freeze", "motion_anomaly"),
        trigger_timeout=int(fcfg["sensor_trigger_timeout_s"] * tick_hz),
        duration=duration,
        n_ticks=n_ticks,
        params=params,
    )

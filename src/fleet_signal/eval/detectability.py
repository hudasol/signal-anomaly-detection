"""Which progressive faults a detector can be *expected to catch* within the latency bar.

The brief's latency criterion applies to "faults the detector is expected to catch".
v2 makes that set explicit and physical, declared before the v2 test set existed
(PLAN_V2 §4):

    A progressive fault is fast-detectable if, 3 events after its labelled start,
    the change it has caused in the affected signal is at least 5 times the
    standard deviation of a 3-event difference of that signal under sensor noise
    and quantisation alone.

5x is the margin a detector needs to alert at the false-alert budget without
alerting on noise; 3 events is the brief's bar. The rule uses only the fault's
parameters (from the ground truth, evaluation only) and the sensor noise in
configs/data.yaml; no detector output is involved, so it cannot be tuned to a
detector. Faults outside the set still count for recall and are still reported
in the all-progressive latency.

Under this rule, with the configured noise, only abrupt battery drains
(extra drain >= ~1.5 %/min) qualify: overheating would need > ~21 degC/min
(temperature noise 0.15 degC, 0.1 degC resolution) and link changes are smaller
than link noise within 3 events.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from fleet_signal.data.config import GenerationConfig

HORIZON_EVENTS = 3
SNR_REQUIRED = 5.0
# Reporting resolution of each signal (data/sim.py `_q` steps).
RESOLUTION = {"battery": 0.01, "temp": 0.1, "link": 1.0}


def _diff_noise(sigma: float, resolution: float) -> float:
    """Std of a k-event difference of a noisy, quantised reading (independent noise)."""
    return math.sqrt(2.0) * math.sqrt(sigma**2 + resolution**2 / 12.0)


def signal_and_noise(fault: dict[str, Any], gen_cfg: GenerationConfig) -> tuple[float, float]:
    """(change caused after HORIZON_EVENTS events, noise of a HORIZON_EVENTS-event difference)."""
    data = gen_cfg.data
    asset_type = _asset_type(fault, data)
    noise = data["profiles"][asset_type]["noise"]
    p = json.loads(fault["params_json"]) if isinstance(fault["params_json"], str) else {}
    minutes = HORIZON_EVENTS / 60.0
    ftype = fault["fault_type"]
    if ftype == "battery_drain":
        rate = float(p.get("extra_pct_per_min", 0.0))
        accel = float(p.get("accel_pct_per_min2", 0.0)) if fault["variant"] == "accelerating" else 0
        sig = rate * minutes + 0.5 * accel * minutes**2
        return sig, _diff_noise(noise["battery"], RESOLUTION["battery"])
    if ftype == "overheating":
        rate = float(p.get("rate_c_per_min", 0.0))
        coeff = float(p.get("runaway_coeff", 0.0)) if fault["variant"] == "runaway" else 0.0
        sig = rate * minutes + rate * coeff * minutes**2
        return sig, _diff_noise(noise["temp"], RESOLUTION["temp"])
    if ftype == "link_degradation":
        link = data["link"]
        fade_step = link["fade_std"] * math.sqrt(2 * (1 - link["fade_phi"]))  # AR(1) increment
        nz = math.sqrt(_diff_noise(link["white_std"], RESOLUTION["link"]) ** 2 + fade_step**2)
        k = HORIZON_EVENTS
        if fault["variant"] == "decline":
            sig = float(p.get("decline_pct_per_min", 0.0)) * minutes
        elif fault["variant"] == "oscillation":
            amp, period = float(p.get("osc_amp_pct", 0.0)), float(p.get("osc_period_s", 1.0))
            growth = min(1.0, k / 60.0)
            phase = 0.5 + 0.5 * math.sin(2 * math.pi * k / period - math.pi / 2)
            sig = amp * growth * phase
        else:  # intermittent: slow decline; dropouts are scheduled after a gap of >= 6 s
            dropouts = p.get("dropouts") or []
            early = any(int(s) < k for s, _e, _f in dropouts)
            sig = 100.0 if early else 0.3 * float(p.get("decline_pct_per_min", 0.0)) * minutes
        return sig, nz
    return 0.0, 1.0  # not progressive


def _asset_type(fault: dict[str, Any], data: dict[str, Any]) -> str:
    for asset in data["fleet"]:
        if asset["asset_id"] == fault["asset_id"]:
            return str(asset["asset_type"])
    raise ValueError(f"asset {fault['asset_id']} is not in the configured fleet")


def with_detectability(faults: pd.DataFrame, gen_cfg: GenerationConfig) -> pd.DataFrame:
    """Add `fast_detectable`, `signal_3`, `noise_3` columns (evaluation only)."""
    out = faults.copy()
    if out.empty:
        return out.assign(fast_detectable=pd.Series(dtype=bool))
    records: list[dict[str, Any]] = [
        {str(k): v for k, v in r.items()} for r in out.to_dict("records")
    ]
    sn = [signal_and_noise(r, gen_cfg) for r in records]
    out["signal_3"] = [s for s, _ in sn]
    out["noise_3"] = [n for _, n in sn]
    out["fast_detectable"] = [
        bool(r["progressive"]) and s >= SNR_REQUIRED * n
        for r, (s, n) in zip(records, sn, strict=True)
    ]
    return out

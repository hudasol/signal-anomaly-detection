"""Normal behaviour of one simulated asset.

The physics is deliberately simple but causal, so that signals relate to each
other the way an operator would expect:

* load comes from what the asset is doing (parked, inspecting, moving, ...);
* battery drains with load and rises while charging;
* temperature follows load (and charging) through a first-order lag;
* link quality falls with distance from base, with slow fading and noise;
* reported position integrates the reported heading and speed.

Internal states map onto the blackbox-telemetry modes:
parked/inspect -> idle, moving -> moving, returning -> returning,
charging -> charging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fleet_signal.data.faults import Fault

STATE_TO_MODE: dict[str, str] = {
    "parked": "idle",
    "inspect": "idle",
    "moving": "moving",
    "returning": "returning",
    "charging": "charging",
}
MOVING_STATES = ("moving", "returning")


def _q(value: float, step: float) -> float:
    """Quantise like a real sensor would (fixed resolution)."""
    return round(round(value / step) * step, 6)


@dataclass
class Episode:
    kind: str
    start: int
    end: int  # exclusive


@dataclass
class AssetSim:
    asset_id: str
    asset_type: str
    base_xy: tuple[float, float]
    profile: dict[str, Any]
    mission: dict[str, Any]
    link_cfg: dict[str, Any]
    episode_cfg: dict[str, Any]
    rng: np.random.Generator
    n_ticks: int
    ambient_c: float
    packet_loss_prob: float
    fault: Fault | None = None
    episodes: list[Episode] = field(default_factory=list)
    states: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        p, m, rng = self.profile, self.mission, self.rng
        self.base = np.array(self.base_xy, dtype=float)
        self.pos = self.base.copy()
        self.z = 0.0
        self.speed = 0.0
        self.heading = float(rng.uniform(0.0, 360.0))
        self.cruise = float(rng.uniform(*p["cruise_speed"]))
        self.cruise_alt = float(rng.uniform(*p["cruise_alt_m"]))
        self.hover_alt = float(rng.uniform(*p["hover_alt_m"]))
        self.battery = float(rng.uniform(*m["start_battery_pct"]))
        self.temp = self.ambient_c + p["heat_gain_c"] * p["loads"]["parked"]
        self.state = "parked"
        self.until = int(rng.integers(m["initial_park_s"][0], m["initial_park_s"][1] + 1))
        self.target = self.base.copy()
        self.charge_target = 95.0
        self.fade = 0.0

        e = self.episode_cfg
        self.scheduled_rtb: int | None = None
        if rng.random() < e["scheduled_rtb_prob"]:
            self.scheduled_rtb = int(rng.uniform(*e["scheduled_rtb_frac"]) * self.n_ticks)
        self.noisy_link: tuple[int, int] | None = None
        if rng.random() < e["noisy_link_prob"]:
            length = int(rng.integers(e["noisy_link_s"][0], e["noisy_link_s"][1] + 1))
            start = int(rng.integers(0, max(1, self.n_ticks - length)))
            self.noisy_link = (start, start + length)
            self.episodes.append(Episode("noisy_link", start, start + length))
        self.manoeuvre: dict[str, float] | None = None

    # ------------------------------------------------------------------

    def _new_waypoint(self) -> np.ndarray:
        radius = self.profile["work_radius_m"]
        r = radius * math.sqrt(self.rng.uniform(self.mission["waypoint_min_frac"] ** 2, 1.0))
        a = self.rng.uniform(0.0, 2 * math.pi)
        return self.base + np.array([r * math.cos(a), r * math.sin(a)])

    def _at(self, point: np.ndarray) -> bool:
        return float(np.linalg.norm(point - self.pos)) <= self.mission["arrival_tol_m"]

    def _transitions(self, t: int) -> None:
        rng, m = self.rng, self.mission
        scheduled = self.scheduled_rtb is not None and t >= self.scheduled_rtb
        low = self.battery <= self.profile["rtb_battery_pct"]
        if self.state in ("parked", "inspect", "moving") and (low or scheduled):
            if scheduled:
                self.scheduled_rtb = None
            if self._at(self.base):
                self.state = "charging"
                self.charge_target = float(rng.uniform(*m["charge_target_pct"]))
            else:
                self.state = "returning"
                self.target = self.base.copy()
            return
        if self.state == "parked" and t >= self.until:
            self.state = "moving"
            self.target = self._new_waypoint()
        elif self.state == "moving" and self._at(self.target):
            self.state = "inspect"
            self.until = t + int(rng.integers(m["inspect_s"][0], m["inspect_s"][1] + 1))
        elif self.state == "inspect" and t >= self.until:
            self.state = "moving"
            self.target = self._new_waypoint()
        elif self.state == "returning" and self._at(self.base):
            self.state = "charging"
            self.charge_target = float(rng.uniform(*m["charge_target_pct"]))
        elif self.state == "charging" and self.battery >= self.charge_target:
            self.state = "parked"
            lo, hi = m["post_charge_park_s"]
            self.until = t + int(rng.integers(lo, hi + 1))

    def _kinematics(self, t: int) -> bool:
        """Advance position/speed/heading/altitude. Returns True if manoeuvring."""
        p, rng, e = self.profile, self.rng, self.episode_cfg
        manoeuvring = False
        if self.state in MOVING_STATES:
            delta = self.target - self.pos
            dist = float(np.linalg.norm(delta))
            desired_heading = math.degrees(math.atan2(delta[1], delta[0])) % 360.0
            v_target = min(self.cruise, max(0.5, 0.5 * dist))
            if self.manoeuvre is None and rng.random() < e["hard_manoeuvre_rate_per_s"]:
                length = int(rng.integers(e["hard_manoeuvre_s"][0], e["hard_manoeuvre_s"][1] + 1))
                self.manoeuvre = {
                    "start": t,
                    "len": length,
                    "mult": float(rng.uniform(*e["hard_manoeuvre_speed_mult"])),
                    "swerve": float(rng.uniform(*e["hard_manoeuvre_swerve_deg"]))
                    * (1 if rng.random() < 0.5 else -1),
                }
                self.episodes.append(Episode("hard_manoeuvre", t, t + length))
            if self.manoeuvre is not None:
                k = t - self.manoeuvre["start"]
                v_target = self.cruise * self.manoeuvre["mult"]
                desired_heading += self.manoeuvre["swerve"] * math.sin(
                    math.pi * k / self.manoeuvre["len"]
                )
                manoeuvring = True
                if k + 1 >= self.manoeuvre["len"]:
                    self.manoeuvre = None
            alt_target = self.cruise_alt
        elif self.state == "inspect":
            dist = math.inf
            v_target = float(rng.uniform(0.0, p["hover_drift_speed"]))
            desired_heading = (self.heading + float(rng.normal(0.0, 30.0))) % 360.0
            alt_target = self.hover_alt
        else:
            dist = math.inf
            v_target = 0.0
            desired_heading = self.heading
            alt_target = 0.0
        if self.state not in MOVING_STATES:
            self.manoeuvre = None

        # Turn toward the desired heading at a limited rate.
        diff = (desired_heading - self.heading + 180.0) % 360.0 - 180.0
        turn = max(-p["turn_rate"], min(p["turn_rate"], diff))
        self.heading = (self.heading + turn) % 360.0

        # Accelerate toward the target speed at a limited rate.
        dv = max(-p["accel"], min(p["accel"], v_target - self.speed))
        self.speed = max(0.0, self.speed + dv)

        # Move; snap onto the target instead of overshooting it.
        step = self.speed
        if self.state in MOVING_STATES and step >= dist:
            self.pos = self.target.copy()
        else:
            rad = math.radians(self.heading)
            self.pos = self.pos + step * np.array([math.cos(rad), math.sin(rad)])

        dz = max(-p["climb_rate"], min(p["climb_rate"], alt_target - self.z))
        self.z = self.z + dz
        return manoeuvring

    def _energy_and_heat(self, t: int, manoeuvring: bool) -> None:
        p = self.profile
        load = p["loads"]["manoeuvre"] if manoeuvring else p["loads"][self.state]
        drain = (p["drain_idle"] + p["drain_load"] * load) / 60.0
        if self.state == "charging":
            taper = 1.0 if self.battery < 80.0 else max(0.15, (100.0 - self.battery) / 20.0)
            self.battery += p["charge_rate"] / 60.0 * taper - p["drain_idle"] / 60.0
        else:
            self.battery -= drain
        if self.fault is not None:
            self.battery -= self.fault.extra_drain_pct_per_s(t)
        self.battery = min(100.0, max(0.0, self.battery))

        target = self.ambient_c + p["heat_gain_c"] * load
        if self.state == "charging":
            target += p["charge_heat_c"]
        self.temp += (target - self.temp) / p["thermal_tau_s"]

    def _link(self, t: int) -> float:
        lc, e, rng = self.link_cfg, self.episode_cfg, self.rng
        noisy = self.noisy_link is not None and self.noisy_link[0] <= t < self.noisy_link[1]
        fade_std = e["noisy_link_fade_std"] if noisy else lc["fade_std"]
        white_std = e["noisy_link_white_std"] if noisy else lc["white_std"]
        phi = lc["fade_phi"]
        self.fade = phi * self.fade + math.sqrt(1 - phi * phi) * fade_std * float(rng.normal())
        dist = float(np.linalg.norm(self.pos - self.base))
        ratio = dist / self.profile["link_range_m"]
        lq = 100.0 * (1.0 - lc["path_loss_coeff"] * ratio**1.5)
        lq += self.fade + white_std * float(rng.normal())
        jitter = abs(float(rng.normal(0.0, 1.0)))
        if self.fault is not None:
            lq = self.fault.modify_link(t, lq, jitter)
        return lq

    # ------------------------------------------------------------------

    def step(self, t: int) -> dict[str, Any] | None:
        """Advance one tick. Returns the reported event, or None if it was lost in transit."""
        self._transitions(t)
        if self.fault is not None:
            self.fault.maybe_start(t, self.state in MOVING_STATES)
        manoeuvring = self._kinematics(t)
        self._energy_and_heat(t, manoeuvring)
        lq = self._link(t)
        self.states.append(self.state)

        n, rng = self.profile["noise"], self.rng
        temp = self.temp + (self.fault.temp_offset_c(t) if self.fault is not None else 0.0)
        heading = _q((self.heading + float(rng.normal(0.0, n["heading"]))) % 360.0, 0.1) % 360.0
        report: dict[str, Any] = {
            "x_m": _q(self.pos[0] + float(rng.normal(0.0, n["gps_m"])), 0.01),
            "y_m": _q(self.pos[1] + float(rng.normal(0.0, n["gps_m"])), 0.01),
            # GPS altitude noise can read slightly below ground; that is realistic.
            "z_m": _q(self.z + float(rng.normal(0.0, n["alt_m"])), 0.01),
            "speed_mps": _q(max(0.0, self.speed + float(rng.normal(0.0, n["speed"]))), 0.01),
            "heading_deg": heading,
            "battery_pct": _q(
                min(100.0, max(0.0, self.battery + float(rng.normal(0.0, n["battery"])))), 0.01
            ),
            "temperature_c": _q(temp + float(rng.normal(0.0, n["temp"])), 0.1),
            "link_quality_pct": _q(min(100.0, max(0.0, lq)), 1.0),
            "mode": STATE_TO_MODE[self.state],
        }
        if self.fault is not None:
            self.fault.corrupt_report(t, report)
        lost = rng.random() < self.packet_loss_prob
        return None if lost else report

    def mode_episodes(self) -> list[Episode]:
        """Contiguous returning/charging stretches, from the full (lossless) state history."""
        out: list[Episode] = []
        start = 0
        for i in range(1, len(self.states) + 1):
            if i == len(self.states) or self.states[i] != self.states[start]:
                if self.states[start] in ("returning", "charging"):
                    out.append(Episode(self.states[start], start, i))
                start = i
        return out

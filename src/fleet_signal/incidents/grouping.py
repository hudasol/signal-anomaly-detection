"""Turning per-event alert decisions into incidents.

Rules, per asset, in event order (only scorable events count):

* open: N consecutive alerting events open an incident; the incident's open
  time is the N-th of those events.
* stay open: while open, any alerting event resets the close counter.
* close: M consecutive non-alerting events close it; the close time is the
  M-th quiet event (or the last event of the run).
* cooldown: if a new opening happens within C events after a close, it
  re-opens the same incident instead of creating a new one. The re-open time
  is recorded, so detection of a second fault is still credited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class IncidentParams:
    open_n: int
    close_m: int
    cooldown_c: int

    def as_dict(self) -> dict[str, int]:
        return {"open_n": self.open_n, "close_m": self.close_m, "cooldown_c": self.cooldown_c}


@dataclass
class Incident:
    start: int  # index of the first alerting event of the opening run
    opens: list[int] = field(default_factory=list)  # index of every open / re-open
    last_alert: int = 0
    close: int = 0


def _runs(alert: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of consecutive True values."""
    if not alert.any():
        return []
    padded = np.concatenate(([False], alert, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def group_alerts(alert: np.ndarray, params: IncidentParams) -> list[Incident]:
    """Group one asset-run's boolean alert sequence into incidents (indices into `alert`)."""
    n_events = len(alert)
    n, m, c = params.open_n, params.close_m, params.cooldown_c
    incidents: list[Incident] = []
    cur: Incident | None = None
    for s, e in _runs(np.asarray(alert, dtype=bool)):
        if cur is not None and s - cur.last_alert - 1 < m:
            cur.last_alert = e  # still open: the quiet gap was shorter than M
            continue
        if e - s + 1 < n:
            continue  # too short to open anything
        open_idx = s + n - 1
        if cur is not None and open_idx - (cur.last_alert + m) <= c:
            cur.opens.append(open_idx)  # re-open inside the cooldown
            cur.last_alert = e
            continue
        if cur is not None:
            incidents.append(cur)
        cur = Incident(start=s, opens=[open_idx], last_alert=e)
    if cur is not None:
        incidents.append(cur)
    for inc in incidents:
        inc.close = min(inc.last_alert + m, n_events - 1)
    return incidents

"""Streaming incident tracker: the online twin of `group_alerts`.

Feed it one decision per event, in order. It applies exactly the same
open / stay-open / close / cooldown rules as the batch grouping used in
evaluation (property tests check they agree on random sequences, with and
without seq gaps), so incidents built from live decisions match what was measured.

Used by `signal-replay`. The HTTP `/score` endpoint is stateless and returns the
per-event decision only; whoever consumes those decisions runs this tracker
(one per asset) to get incidents.

An event that could not be scored (insufficient data, degraded, model
unavailable) is passed as `alert=False`: it never opens an incident, and it
counts toward closing one, exactly as in evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fleet_signal.incidents.grouping import IncidentParams


@dataclass
class TrackedIncident:
    incident_id: int
    start_seq: int
    open_seq: int
    open_seqs: list[int] = field(default_factory=list)
    last_alert_seq: int = 0
    close_seq: int | None = None
    peak_score: float = float("-inf")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class IncidentTracker:
    def __init__(self, params: IncidentParams) -> None:
        self.p = params
        self.idx = -1
        self.run = 0  # consecutive alerts while closed
        self.run_start_seq = 0
        self.current: TrackedIncident | None = None
        self.is_open = False
        self._last_alert_idx = -(10**9)
        self.incidents: list[TrackedIncident] = []

    def update(self, seq: int, alert: bool, score: float | None = None) -> list[dict[str, Any]]:
        """Process one event; returns lifecycle events ('opened', 'reopened', 'closed')."""
        self.idx += 1
        i, n, m, c = self.idx, self.p.open_n, self.p.close_m, self.p.cooldown_c
        out: list[dict[str, Any]] = []
        if alert:
            if self.is_open:
                assert self.current is not None
                self._last_alert_idx = i
                self.current.last_alert_seq = seq
                if score is not None:
                    self.current.peak_score = max(self.current.peak_score, score)
                return out
            if self.run == 0:
                self.run_start_seq = seq
            self.run += 1
            if self.run >= n:
                cur = self.current
                if cur is not None and i - (self._last_alert_idx + m) <= c:
                    cur.open_seqs.append(seq)
                    cur.close_seq = None
                    out.append({"event": "reopened", "seq": seq, "incident_id": cur.incident_id})
                else:
                    cur = TrackedIncident(
                        incident_id=len(self.incidents) + 1,
                        start_seq=self.run_start_seq,
                        open_seq=seq,
                        open_seqs=[seq],
                    )
                    self.incidents.append(cur)
                    self.current = cur
                    out.append({"event": "opened", "seq": seq, "incident_id": cur.incident_id})
                self.is_open = True
                self._last_alert_idx = i
                cur.last_alert_seq = seq
                if score is not None:
                    cur.peak_score = max(cur.peak_score, score)
                self.run = 0
            return out
        self.run = 0
        if self.is_open and i - self._last_alert_idx >= m:
            assert self.current is not None
            self.is_open = False
            self.current.close_seq = seq
            out.append({"event": "closed", "seq": seq, "incident_id": self.current.incident_id})
        return out

    def finish(self, last_seq: int) -> list[dict[str, Any]]:
        """End of stream: close anything still open at the last event."""
        if self.is_open and self.current is not None:
            self.is_open = False
            self.current.close_seq = last_seq
            return [{"event": "closed", "seq": last_seq, "incident_id": self.current.incident_id}]
        return []

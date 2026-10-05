"""Incident grouping: open / stay-open / close / cooldown on hand-built alert sequences."""

from __future__ import annotations

import numpy as np

from fleet_signal.incidents.grouping import IncidentParams, group_alerts

P = IncidentParams(open_n=2, close_m=3, cooldown_c=4)


def _a(s: str) -> np.ndarray:
    return np.array([c == "1" for c in s])


def _summ(alert: str, params: IncidentParams = P) -> list[tuple[int, list[int], int, int]]:
    return [(i.start, i.opens, i.last_alert, i.close) for i in group_alerts(_a(alert), params)]


def test_no_alerts_no_incidents() -> None:
    assert _summ("0000000") == []


def test_needs_n_consecutive_to_open() -> None:
    assert _summ("0101010100") == []
    assert _summ("0110000000") == [(1, [2], 2, 5)]


def test_open_n_one_opens_on_first_alert() -> None:
    assert _summ("0010000000", IncidentParams(1, 3, 0)) == [(2, [2], 2, 5)]


def test_short_quiet_gap_keeps_incident_open() -> None:
    # gap of 2 quiet events < M=3: same incident, and a single alert is enough while open
    assert _summ("1100100000") == [(0, [1], 4, 7)]


def test_close_after_m_quiet_events() -> None:
    # gap of exactly M=3 closes the first incident
    inc = _summ("110001100000000000", IncidentParams(2, 3, 0))
    assert inc == [(0, [1], 1, 4), (5, [6], 6, 9)]


def test_reopen_within_cooldown_merges_and_records_reopen() -> None:
    # closes at 1+3=4; re-opens at 7, which is 3 <= C=4 after the close -> same incident
    assert _summ("110000110000000") == [(0, [1, 7], 7, 10)]


def test_reopen_after_cooldown_is_new_incident() -> None:
    alert = "11" + "0" * 12 + "11" + "0" * 6
    assert _summ(alert) == [(0, [1], 1, 4), (14, [15], 15, 18)]


def test_short_burst_during_cooldown_does_not_reopen() -> None:
    assert _summ("1100001000000") == [(0, [1], 1, 4)]


def test_incident_open_at_end_of_run_closes_at_last_event() -> None:
    assert _summ("000011") == [(4, [5], 5, 5)]


def test_one_long_fault_is_one_incident_not_spam() -> None:
    noisy = "11" + "1101" * 50 + "0" * 10  # flickers but never quiet for M events
    assert len(group_alerts(_a(noisy), P)) == 1

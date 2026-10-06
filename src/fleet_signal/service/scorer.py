"""Scoring one asset's telemetry window: the inference boundary.

Every response carries an explicit `status`. The service answers "normal" only
for a validated, full-context window scored by a verified model:

    unavailable        no verified model: artifact missing, not in the registry,
                       SHA-256 mismatch, unreadable, or built for another schema
    insufficient_data  not enough context to compute the features the model was
                       evaluated with (see below)
    degraded           the window cannot be trusted: a gap, an impossible or
                       non-finite reading, unknown asset type or mode, seq not
                       strictly increasing, more than one asset, timestamps not 1 Hz
    ok                 scored; decision is "normal" or "anomalous"

Only `ok` carries a decision. The decision is per EVENT (this window's last
event crossed the threshold or not). Incidents (open after N alerts, close after
M quiet events) are built from the stream of decisions by
`fleet_signal.incidents.tracker.IncidentTracker`, as `signal-replay` does; the
false-incident rate in EVALUATION.md is for incidents, not for these decisions.

Context rule. Some features look back up to 600 events ("seconds in this mode",
events since a field last changed). Computed from a shorter window they are cut
off and the score is NOT the score evaluation measured: a battery rule that needs
two minutes in one mode can never fire on a 2-minute window, which would turn an
alert into "normal". So a window is scored only if it holds HISTORY_BUFFER
events, or it starts at the asset's first event of the run (seq <= max_gap), in
which case nothing earlier exists to be cut off. Anything else is
`insufficient_data`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.features.build import FEATURE_COLUMNS, FeatureConfig, build_features
from fleet_signal.registry import (
    REGISTRY_PATH,
    ArtifactError,
    ModelArtifact,
    code_provenance,
    load_artifact,
    registered_artifact,
)
from fleet_signal.service.validation import check_window

log = logging.getLogger(__name__)

# Longest look-back any feature needs: mode_age is capped at 600 s.
HISTORY_BUFFER = 650
REQUIRED = (
    "asset_id",
    "asset_type",
    "seq",
    "x_m",
    "y_m",
    "z_m",
    "speed_mps",
    "battery_pct",
    "temperature_c",
    "link_quality_pct",
    "mode",
)


@dataclass
class ScoreResult:
    status: str
    asset_id: str | None = None
    seq: int | None = None
    score: float | None = None
    threshold: float | None = None
    decision: str | None = None
    model_version: str | None = None
    detector: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    history: int = 0
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def normalise_events(events: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Accept flat Signal events or Blackbox-contract events (nested `position`)."""
    if isinstance(events, pd.DataFrame):
        df = events.copy()
    else:
        rows = []
        for e in events:
            if not isinstance(e, dict):
                raise TypeError("each event must be an object")
            e = dict(e)
            pos = e.pop("position", None)
            if isinstance(pos, dict):
                e.update({k: pos.get(k) for k in ("x_m", "y_m", "z_m")})
            rows.append(e)
        df = pd.DataFrame(rows)
    if "run_id" not in df:
        df["run_id"] = "live"
    if "timestamp_utc" not in df:
        df["timestamp_utc"] = pd.NaT
    return df


def _identify(df: pd.DataFrame) -> dict[str, Any]:
    """Best-effort asset_id / seq of the last event, for reporting a rejected window."""
    out: dict[str, Any] = {}
    last = df.iloc[-1]
    if isinstance(last["asset_id"], str):
        out["asset_id"] = last["asset_id"]
    seq = last["seq"]
    if isinstance(seq, (int, np.integer)) and not isinstance(seq, (bool, np.bool_)):
        out["seq"] = int(seq)
    out["history"] = len(df)
    return out


def _single_threaded(detector: Any) -> None:
    """Inference runs one window at a time inside a web worker: do not fan out
    across every CPU (LOF was fitted with n_jobs=-1). Scores are unchanged."""
    models = list(getattr(detector, "models", {}).values())
    models += [m for m in (getattr(detector, "model", None),) if m is not None]
    for m in models:
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1


class Scorer:
    """Loads a verified artifact once and scores windows.

    Only an artifact listed in the model registry is loaded, and its SHA-256 is
    checked before it is unpickled. `allow_unregistered=True` skips the
    registry (tests and local experiments only; not reachable from the API or env).
    """

    def __init__(
        self,
        artifact_path: Path | None = None,
        threshold_override: float | None = None,
        fcfg: FeatureConfig | None = None,
        *,
        allow_unregistered: bool = False,
        registry_path: Path = REGISTRY_PATH,
    ) -> None:
        self.artifact: ModelArtifact | None = None
        self.error: str | None = None
        self.provenance: dict[str, Any] = {}
        self.threshold_override = threshold_override
        self._fcfg: FeatureConfig | None = fcfg
        try:
            if self._fcfg is None:
                self._fcfg = FeatureConfig.load()
            if allow_unregistered and artifact_path is not None:
                path, sha = Path(artifact_path), None
            else:
                path, sha = registered_artifact(
                    Path(artifact_path) if artifact_path else None, registry_path
                )
            self.artifact = load_artifact(path, self._fcfg, expected_sha256=sha)
            _single_threaded(self.artifact.detector)
            self.provenance = code_provenance(path)
        except ArtifactError as exc:
            self.error = str(exc)
            log.error("model unavailable: %s", exc)
        except Exception as exc:  # config missing or broken: unavailable, never a crash
            self.error = f"service misconfigured: {type(exc).__name__}"
            log.exception("model unavailable: startup failed")

    @property
    def fcfg(self) -> FeatureConfig:
        if self._fcfg is None:
            raise ArtifactError(self.error or "feature config not loaded")
        return self._fcfg

    @property
    def available(self) -> bool:
        return self.artifact is not None

    @property
    def threshold(self) -> float | None:
        if self.artifact is None:
            return None
        return (
            self.threshold_override
            if self.threshold_override is not None
            else self.artifact.threshold
        )

    def result(self, **kw: Any) -> ScoreResult:
        art = self.artifact
        return ScoreResult(
            model_version=art.model_version if art else None,
            detector=art.detector_name if art else None,
            threshold=self.threshold,
            **kw,
        )

    def score_events(self, events: list[dict[str, Any]] | pd.DataFrame) -> ScoreResult:
        """Score the LAST event of one asset's window."""
        if not self.available:
            return self.result(status="unavailable", reason=self.error)
        try:
            df = normalise_events(events)
        except (TypeError, ValueError):
            return self.result(status="degraded", reason="events are not a list of objects")
        if df.empty:
            return self.result(status="degraded", reason="no events")
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            return self.result(status="degraded", reason=f"missing fields: {missing}")
        # Never repair a window (no sorting, no de-duplication): a window that is not
        # one asset, strictly increasing seq, plausible values, is reported, not scored.
        problem = check_window(df)
        if problem is not None:
            return self.result(status="degraded", reason=problem, **_identify(df))
        df = df.tail(HISTORY_BUFFER)
        last = df.iloc[-1]
        asset_id, seq, n = str(last["asset_id"]), int(last["seq"]), len(df)
        if n < self.fcfg.min_history:
            return self.result(
                status="insufficient_data", asset_id=asset_id, seq=seq, history=n,
                reason=f"{n} events of history, need {self.fcfg.min_history}",
            )  # fmt: skip
        starts_at_run_start = int(df["seq"].iloc[0]) <= self.fcfg.max_gap
        if n < HISTORY_BUFFER and not starts_at_run_start:
            return self.result(
                status="insufficient_data", asset_id=asset_id, seq=seq, history=n,
                reason=(f"{n} events sent mid-run; send {HISTORY_BUFFER} (or the whole run so "
                        "far) so look-back features match evaluation"),
            )  # fmt: skip
        try:
            feats = build_features(df, self.fcfg).tail(1)
        except (ValueError, TypeError, KeyError):
            log.exception("feature computation failed for %s seq %s", asset_id, seq)
            return self.result(status="degraded", asset_id=asset_id, seq=seq, history=n,
                              reason="features could not be computed from this window")  # fmt: skip
        return self._score_feature_row(feats, asset_id, seq, n)

    def _score_feature_row(
        self, feats: pd.DataFrame, asset_id: str, seq: int, history: int
    ) -> ScoreResult:
        row = feats.iloc[0]
        values = feats[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        if not bool(row["gap_ok"]):
            gap = f"telemetry gap of {row['max_gap_l']:.0f} events in window"
            return self.result(status="degraded", asset_id=asset_id, seq=seq, history=history,
                              reason=gap)  # fmt: skip
        if not np.isfinite(values).all():
            bad = [c for c, v in zip(FEATURE_COLUMNS, values[0], strict=True) if not np.isfinite(v)]
            return self.result(status="degraded", asset_id=asset_id, seq=seq, history=history,
                              reason=f"non-finite features: {bad[:5]}")  # fmt: skip
        art, thr = self.artifact, self.threshold
        if art is None or thr is None:  # cannot happen after the availability check
            return self.result(status="unavailable", reason=self.error)
        det = art.detector
        score = float(det.score(feats)[0])
        if not np.isfinite(score):
            return self.result(status="degraded", asset_id=asset_id, seq=seq, history=history,
                              reason="detector returned no score")  # fmt: skip
        return self.result(
            status="ok", asset_id=asset_id, seq=seq, history=history, score=score,
            decision="anomalous" if score >= thr else "normal",
            evidence=det.evidence(feats, k=3)[0],
        )  # fmt: skip

    def score_feature_rows(self, feats: pd.DataFrame) -> list[ScoreResult]:
        """Batch path for replay: one result per pre-computed (causal) feature row."""
        art, thr = self.artifact, self.threshold
        if art is None or thr is None:
            return [self.result(status="unavailable", reason=self.error) for _ in range(len(feats))]
        det = art.detector
        scores = det.score(feats)
        evidence = det.evidence(feats, k=3)
        out: list[ScoreResult] = []
        for i, (_, row) in enumerate(feats.iterrows()):
            base = {"asset_id": str(row["asset_id"]), "seq": int(row["seq"]),
                    "history": int(row["history"])}  # fmt: skip
            if not bool(row["history_ok"]):
                out.append(self.result(status="insufficient_data", **base,
                                      reason=f"need {self.fcfg.min_history} events"))  # fmt: skip
            elif not bool(row["gap_ok"]):
                out.append(self.result(status="degraded", **base, reason="telemetry gap in window"))
            elif not np.isfinite(scores[i]):
                out.append(self.result(status="degraded", **base, reason="non-finite features"))
            else:
                s = float(scores[i])
                out.append(self.result(status="ok", **base, score=s, evidence=evidence[i],
                                      decision="anomalous" if s >= thr else "normal"))  # fmt: skip
        return out

"""FastAPI inference service.

    uvicorn fleet_signal.service.app:app            # serves the registry's shipped model
    SIGNAL_ARTIFACT=models/lof/.../model.joblib uvicorn fleet_signal.service.app:app

Endpoints:
    GET  /health   200 if a model is loaded, 503 + reason if not
    GET  /model    frozen threshold, incident params, versions
    POST /score    one asset's recent events (oldest first) -> status, score,
                   threshold, decision, model version, evidence

/score returns 503 with status "unavailable" when no model is loaded. It
never falls back to "normal".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from fleet_signal.service.scorer import HISTORY_BUFFER, Scorer


class ScoreRequest(BaseModel):
    events: list[dict[str, Any]] = Field(
        ...,
        description=(
            "One asset's telemetry, oldest first. Flat Signal events or blackbox-telemetry "
            f"events (nested `position`). Send up to {HISTORY_BUFFER} events."
        ),
        min_length=1,
    )


def create_app(artifact_path: Path | None = None) -> FastAPI:
    env = os.environ.get("SIGNAL_ARTIFACT")
    scorer = Scorer(artifact_path or (Path(env) if env else None))
    app = FastAPI(title="Signal", version="1.0.0")
    app.state.scorer = scorer

    @app.get("/health")
    def health() -> JSONResponse:
        s: Scorer = app.state.scorer
        if not s.available:
            return JSONResponse({"status": "unavailable", "reason": s.error}, status_code=503)
        assert s.artifact is not None
        return JSONResponse({"status": "ok", "model_version": s.artifact.model_version})

    @app.get("/model")
    def model() -> JSONResponse:
        s: Scorer = app.state.scorer
        if not s.available:
            return JSONResponse({"status": "unavailable", "reason": s.error}, status_code=503)
        assert s.artifact is not None
        meta = s.artifact.metadata()
        keep = ("model_version", "detector_name", "threshold", "incident_params",
                "data_version", "feature_schema_hash", "created_utc", "git_sha")  # fmt: skip
        return JSONResponse({k: meta[k] for k in keep} | {"history_buffer": HISTORY_BUFFER})

    @app.post("/score")
    def score(req: ScoreRequest) -> JSONResponse:
        s: Scorer = app.state.scorer
        result = s.score_events(req.events).as_dict()
        return JSONResponse(result, status_code=503 if result["status"] == "unavailable" else 200)

    return app


app = create_app()

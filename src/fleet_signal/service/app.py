"""FastAPI inference service.

    uvicorn fleet_signal.service.app:app            # serves the registry's shipped model
    SIGNAL_ARTIFACT=models/lof/<version>/model.joblib uvicorn fleet_signal.service.app:app

`SIGNAL_ARTIFACT` may only name an artifact listed in `models/registry.json`; its
SHA-256 is checked before it is loaded. Anything else leaves the service
`unavailable`.

Endpoints:
    GET  /livez    200 while the process is up (liveness)
    GET  /readyz   200 if a verified model is loaded, else 503 + reason (readiness)
    GET  /health   same as /readyz (kept for existing clients)
    GET  /model    frozen threshold, incident params, versions, code provenance
    POST /score    one asset's recent events (oldest first) -> status, score,
                   threshold, per-event decision, model version, evidence

Malformed requests (wrong types, unknown mode or asset type, several assets,
seq not strictly increasing, more than MAX_EVENTS events, body over
MAX_BODY_BYTES) get 422 / 413 and are never scored. Without a verified model,
/score returns 503 `unavailable`. It never falls back to "normal".
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fleet_signal.service.logs import audit, configure_logging
from fleet_signal.service.scorer import HISTORY_BUFFER, Scorer
from fleet_signal.service.validation import MAX_BODY_BYTES, MAX_EVENTS, ScoreRequest

try:
    __version__ = version("fleet-signal")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "unknown"


class BodyLimit:
    """Reject request bodies over `limit` bytes with 413 before anything parses them.

    The body is read (at most `limit` bytes) before the app is called, then handed
    to the app unchanged, so an oversized or endless body never reaches JSON parsing.
    """

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app, self.limit = app, limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length" and value.isdigit() and int(value) > self.limit:
                await self._reject(scope, send)
                return
        chunks: list[bytes] = []
        size = 0
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                return  # client went away
            chunk = msg.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                await self._reject(scope, send)
                return
            chunks.append(chunk)
            if not msg.get("more_body", False):
                break
        body, sent = b"".join(chunks), False

        async def replay() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        resp = JSONResponse(
            {"status": "invalid_request", "decision": None,
             "reason": f"request body over {self.limit} bytes"},
            status_code=413,
        )  # fmt: skip
        await resp(scope, _empty_receive, send)


async def _empty_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


def create_app(artifact_path: Path | None = None, *, allow_unregistered: bool = False) -> FastAPI:
    """Never raises: any startup problem leaves the service up but `unavailable`."""
    configure_logging()
    env = os.environ.get("SIGNAL_ARTIFACT")
    path = artifact_path or (Path(env) if env else None)
    scorer = Scorer(path, allow_unregistered=allow_unregistered and artifact_path is not None)
    app = FastAPI(title="Signal", version=__version__)
    app.state.scorer = scorer

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations and messages only: never echo the submitted values back.
        errors = [
            {"loc": [str(p) for p in e.get("loc", ())], "msg": str(e.get("msg", ""))}
            for e in exc.errors()[:20]
        ]
        return JSONResponse(
            {"status": "invalid_request", "decision": None, "errors": errors}, status_code=422
        )

    def _ready() -> JSONResponse:
        s: Scorer = app.state.scorer
        if s.artifact is None:
            return JSONResponse({"status": "unavailable", "reason": s.error}, status_code=503)
        return JSONResponse({"status": "ok", "model_version": s.artifact.model_version})

    @app.get("/livez")
    def livez() -> dict[str, str]:
        return {"status": "alive", "version": __version__}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        return _ready()

    @app.get("/health")
    def health() -> JSONResponse:
        return _ready()

    @app.get("/model")
    def model() -> JSONResponse:
        s: Scorer = app.state.scorer
        if s.artifact is None:
            return JSONResponse({"status": "unavailable", "reason": s.error}, status_code=503)
        meta = s.artifact.metadata()
        keep = ("model_version", "detector_name", "threshold", "incident_params",
                "data_version", "feature_schema_hash")  # fmt: skip
        return JSONResponse(
            {k: meta[k] for k in keep}
            | {
                "history_buffer": HISTORY_BUFFER,
                "max_events": MAX_EVENTS,
                "service_version": __version__,
                "code_provenance": s.provenance,
                "decision_unit": "event",
            }
        )

    @app.post("/score")
    def score(req: ScoreRequest) -> JSONResponse:  # sync: runs in the worker threadpool
        s: Scorer = app.state.scorer
        t0 = time.perf_counter()
        events = [e.flat() for e in req.events]
        result = s.score_events(events).as_dict()
        digest = json.dumps(events, sort_keys=True, default=str).encode()
        audit(_audit_record(result, digest, time.perf_counter() - t0, len(events)))
        return JSONResponse(result, status_code=503 if result["status"] == "unavailable" else 200)

    return app


def _audit_record(result: dict[str, Any], body: bytes, seconds: float, n: int) -> dict[str, Any]:
    keep = ("status", "asset_id", "seq", "score", "threshold", "decision", "model_version",
            "reason")  # fmt: skip
    return {k: result.get(k) for k in keep} | {
        "n_events": n,
        "events_sha256": hashlib.sha256(body).hexdigest()[:16],
        "latency_ms": round(seconds * 1000, 2),
        "top_signal": (result.get("evidence") or [{}])[0].get("signal"),
    }


def build() -> ASGIApp:
    """The served ASGI app: the FastAPI app behind the request-size limit."""
    return BodyLimit(create_app(), MAX_BODY_BYTES)


_built: dict[str, ASGIApp] = {}


def __getattr__(name: str) -> Any:
    # `uvicorn fleet_signal.service.app:app` builds the app on first access, not at import,
    # so importing this module never loads a model or touches the filesystem.
    if name == "app":
        if "app" not in _built:
            _built["app"] = build()
        return _built["app"]
    raise AttributeError(name)


__all__ = ["BodyLimit", "build", "create_app"]

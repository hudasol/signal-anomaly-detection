"""Structured (JSON-lines) logging for the service, and the decision audit log.

Every /score call writes one audit record: when, which asset and seq, what was
decided, by which model version, at which threshold, and a hash of the request
body. That is enough to reconstruct what the live service said about an asset
after the fact (for example, after a missed fault), without storing telemetry.

    SIGNAL_LOG_LEVEL     default INFO
    SIGNAL_AUDIT_LOG     optional file path; audit records are also appended there
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

AUDIT_LOGGER = "fleet_signal.audit"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


_configured = False


def configure_logging() -> None:
    """Idempotent. JSON lines to stderr for everything under `fleet_signal`."""
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    pkg = logging.getLogger("fleet_signal")
    pkg.addHandler(handler)
    pkg.setLevel(os.environ.get("SIGNAL_LOG_LEVEL", "INFO").upper())
    pkg.propagate = False
    audit_file = os.environ.get("SIGNAL_AUDIT_LOG")
    if audit_file:
        fh = logging.FileHandler(audit_file)
        fh.setFormatter(JsonFormatter())
        logging.getLogger(AUDIT_LOGGER).addHandler(fh)
    _configured = True


def audit(fields: dict[str, Any]) -> None:
    logging.getLogger(AUDIT_LOGGER).info("score", extra={"fields": fields})

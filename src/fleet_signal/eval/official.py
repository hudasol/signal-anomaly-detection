"""The test-once guard.

The official test result is written exactly once per detector + model version.
A second attempt raises instead of overwriting, so "tune against test errors,
re-run, report the better number" is not possible without deleting a committed
file, which would show in git history. Demonstration runs (e.g. moving the
threshold in the acceptance demo) write to results/demo/ and never touch the
official file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fleet_signal.data.config import REPO_ROOT

OFFICIAL_DIR = REPO_ROOT / "results" / "official"
DEMO_DIR = REPO_ROOT / "results" / "demo"


class OfficialResultExists(RuntimeError):
    pass


def official_path(detector: str, model_version: str, root: Path = OFFICIAL_DIR) -> Path:
    return Path(root) / f"test_{detector}_{model_version}.json"


def write_official(
    detector: str, model_version: str, payload: dict[str, Any], root: Path = OFFICIAL_DIR
) -> Path:
    path = official_path(detector, model_version, root)
    if path.exists():
        raise OfficialResultExists(
            f"{path} already exists. The test set is evaluated once per model version; "
            "use --demo for exploratory runs."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    # "x" mode: fail even if another process created the file in the meantime.
    with path.open("x") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def write_demo(name: str, payload: dict[str, Any], root: Path = DEMO_DIR) -> Path:
    path = Path(root) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path

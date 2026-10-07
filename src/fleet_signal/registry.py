"""Model artifacts and the model registry.

An artifact bundles everything needed to reproduce a decision: the fitted
detector, the feature config and schema hash, the frozen threshold and
incident params, the data version and train runs it was fitted on, and a
model version derived from all of those.

    models/<detector>/<model_version>/model.joblib     (committed: the evaluated files)
    models/registry.json                               (committed)

The registry ties artifact -> data version -> feature schema -> threshold ->
validation report -> official test result, and records the artifact's
SHA-256. Serving and the official test only load an artifact listed in the
registry, and check its SHA-256 BEFORE unpickling it: joblib/pickle can run
code while loading, so an unverified file is never opened.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from fleet_signal.data.config import REPO_ROOT
from fleet_signal.detectors.base import Detector
from fleet_signal.features.build import FeatureConfig
from fleet_signal.incidents.grouping import IncidentParams

MODELS_DIR = REPO_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"
ARTIFACT_FORMAT = 1

log = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    """The artifact is missing, unreadable, or does not match the running code."""


@dataclass
class ModelArtifact:
    detector: Detector
    detector_name: str
    threshold: float
    incident_params: dict[str, int]
    feature_config: dict[str, Any]
    feature_schema_hash: str
    data_version: str
    train_run_ids: list[str]
    validation_summary: dict[str, Any]
    git_sha: str | None
    created_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    artifact_format: int = ARTIFACT_FORMAT

    @property
    def model_version(self) -> str:
        """Content-derived: same detector params + threshold + data + schema => same version."""
        ident = {
            "detector": self.detector_name,
            "params": getattr(self.detector, "params", {}),
            "rules": getattr(self.detector, "rules", None),
            "threshold": round(self.threshold, 9),
            "incident_params": self.incident_params,
            "feature_schema_hash": self.feature_schema_hash,
            "data_version": self.data_version,
            "train_run_ids": self.train_run_ids,
        }
        digest = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode())
        return f"{self.detector_name}-{digest.hexdigest()[:10]}"

    @property
    def params(self) -> IncidentParams:
        return IncidentParams(**self.incident_params)

    def metadata(self) -> dict[str, Any]:
        meta = {k: v for k, v in asdict(self).items() if k != "detector"}
        meta["model_version"] = self.model_version
        return meta


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(art: ModelArtifact, root: Path = MODELS_DIR) -> Path:
    return Path(root) / art.detector_name / art.model_version / "model.joblib"


CODE_SOURCES: tuple[str, ...] = ("detectors", "features")


def detector_code_hash() -> str:
    """Hash of the code that turns telemetry into a score (detectors + feature builder).

    Recorded in metadata.json for every artifact saved since 2026-10-06 (the three
    evaluated artifacts predate it). It is deliberately NOT part of `model_version`:
    the version identifies what was fitted and frozen; this records which code ran it.
    """
    h = hashlib.sha256()
    pkg = Path(__file__).parent
    for sub in CODE_SOURCES:
        for f in sorted((pkg / sub).glob("*.py")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def save_artifact(art: ModelArtifact, root: Path = MODELS_DIR) -> Path:
    """Write the artifact. The pickle holds no timestamp or git SHA, so the same model
    gives the same bytes; creation time, git SHA and code hash go to metadata.json."""
    path = artifact_path(art, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(replace(art, created_utc="", git_sha=None), path)
    meta = art.metadata() | {"detector_code_hash": detector_code_hash()}
    (path.parent / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load_artifact(
    path: Path, fcfg: FeatureConfig | None = None, expected_sha256: str | None = None
) -> ModelArtifact:
    """Load and validate an artifact; any problem raises ArtifactError (never returns a guess).

    With `expected_sha256`, the file's hash is checked BEFORE it is unpickled, so a
    tampered or substituted file is refused without running any of its code.
    """
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"model artifact not found: {path}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ArtifactError(f"artifact SHA-256 does not match the registry: {path}")
    try:
        art = joblib.load(path)
    except Exception as exc:  # corrupt, truncated, or from an incompatible library
        raise ArtifactError(f"model artifact could not be loaded: {exc}") from exc
    if not isinstance(art, ModelArtifact):
        raise ArtifactError("file is not a Signal model artifact")
    if art.artifact_format != ARTIFACT_FORMAT:
        raise ArtifactError(f"artifact format {art.artifact_format} != {ARTIFACT_FORMAT}")
    expected = (fcfg or FeatureConfig.load()).schema_hash
    if art.feature_schema_hash != expected:
        raise ArtifactError(
            f"feature schema mismatch: artifact {art.feature_schema_hash}, running code {expected}"
        )
    return art


# ---------------------------------------------------------------- registry


def read_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    if not Path(path).exists():
        return {"serving": None, "models": {}}
    try:
        reg = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"model registry unreadable: {path}") from exc
    if not isinstance(reg, dict) or not isinstance(reg.get("models", {}), dict):
        raise ArtifactError(f"model registry malformed: {path}")
    return reg


def write_registry(reg: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    """Atomic: write a temp file in the same directory, then rename over the old one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".registry-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(reg, indent=2, default=str) + "\n")
        # mkstemp creates 0600; the registry must stay readable by the service user
        # (the Docker image runs as non-root), so apply the normal umask-based mode.
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class EvaluatedModelError(RuntimeError):
    """Refusing to replace an artifact that already has an official test result."""


def is_frozen_and_present(name: str, model_version: str, reg_path: Path = REGISTRY_PATH) -> bool:
    """True if `name` already holds this exact evaluated artifact on disk (SHA verified)."""
    entry = read_registry(reg_path)["models"].get(name)
    if not entry or not entry.get("official_test_result"):
        return False
    path = REPO_ROOT / entry["artifact"]
    return (
        entry["model_version"] == model_version
        and path.exists()
        and sha256_file(path) == entry["artifact_sha256"]
    )


def register(art: ModelArtifact, path_on_disk: Path, reg_path: Path = REGISTRY_PATH) -> None:
    reg = read_registry(reg_path)
    old = reg["models"].get(art.detector_name)
    if old and old.get("official_test_result"):
        raise EvaluatedModelError(
            f"{art.detector_name} {old['model_version']} has an official test result "
            f"({old['official_test_result']}); refusing to overwrite its registry entry. "
            "The evaluated artifact is committed under models/; restore it with git."
        )
    rel = Path(path_on_disk).resolve()
    try:
        rel = rel.relative_to(REPO_ROOT)
    except ValueError:
        pass
    reg["models"][art.detector_name] = {
        "model_version": art.model_version,
        "artifact": str(rel),
        "artifact_sha256": sha256_file(path_on_disk),
        "threshold": art.threshold,
        "incident_params": art.incident_params,
        "data_version": art.data_version,
        "feature_version": art.feature_config["feature_version"],
        "feature_schema_hash": art.feature_schema_hash,
        "n_train_runs": len(art.train_run_ids),
        "validation_report": "results/validation/validation_report.json",
        "validation_summary": {
            k: art.validation_summary.get(k)
            for k in ("precision", "recall", "fp_per_10min", "progressive_latency_median")
        },
        "official_test_result": reg["models"]
        .get(art.detector_name, {})
        .get("official_test_result"),
        "git_sha": art.git_sha,
        "detector_code_hash": detector_code_hash(),
        "created_utc": art.created_utc,
    }
    write_registry(reg, reg_path)


def serving_artifact_path(reg_path: Path = REGISTRY_PATH) -> Path:
    return registered_artifact(None, reg_path)[0]


def registered_artifact(path: Path | None, reg_path: Path = REGISTRY_PATH) -> tuple[Path, str]:
    """(path, expected SHA-256) for the serving model (path=None) or a registered path.

    An artifact that is not listed in the registry is refused: serving never loads
    an arbitrary file (pickles execute code on load).
    """
    reg = read_registry(reg_path)
    models = reg.get("models", {})
    if path is None:
        name = reg.get("serving")
        if not name or name not in models:
            raise ArtifactError("no serving model set in the registry")
        entry = models[name]
        return REPO_ROOT / entry["artifact"], str(entry["artifact_sha256"])
    want = Path(path).resolve()
    for entry in models.values():
        if (REPO_ROOT / entry["artifact"]).resolve() == want:
            return want, str(entry["artifact_sha256"])
    raise ArtifactError(f"artifact is not listed in the model registry: {path}")


def code_provenance(path: Path) -> dict[str, Any]:
    """Compare the code hash recorded next to an artifact with the running code.

    Informational: the three evaluated artifacts predate the recorded hash, and the
    scoring code has had post-freeze service fixes that do not change scores (tested).
    """
    now = detector_code_hash()
    meta_path = Path(path).parent / "metadata.json"
    recorded = None
    if meta_path.exists():
        try:
            recorded = json.loads(meta_path.read_text()).get("detector_code_hash")
        except (OSError, ValueError):
            recorded = None
    if recorded is None:
        status = "not recorded"
    else:
        status = "match" if recorded == now else "differs"
    if status == "differs":
        log.warning("artifact %s was saved with detector code %s; running %s", path, recorded, now)
    return {"recorded": recorded, "running": now, "status": status}

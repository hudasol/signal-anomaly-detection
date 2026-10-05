"""Model artifacts and the model registry.

An artifact bundles everything needed to reproduce a decision: the fitted
detector, the feature config and schema hash, the frozen threshold and
incident params, the data version and train runs it was fitted on, and a
model version derived from all of those.

    models/<detector>/<model_version>/model.joblib     (not committed; rebuild with signal-train)
    models/registry.json                               (committed)

The registry ties artifact -> data version -> feature schema -> threshold ->
validation report -> official test result, and records the artifact's
SHA-256 so a rebuilt file can be checked against the one that was evaluated.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
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


def save_artifact(art: ModelArtifact, root: Path = MODELS_DIR) -> Path:
    path = artifact_path(art, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, path)
    (path.parent / "metadata.json").write_text(json.dumps(art.metadata(), indent=2, default=str))
    return path


def load_artifact(path: Path, fcfg: FeatureConfig | None = None) -> ModelArtifact:
    """Load and validate an artifact; any problem raises ArtifactError (never returns a guess)."""
    path = Path(path)
    if not path.exists():
        raise ArtifactError(f"model artifact not found: {path}")
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
    return json.loads(Path(path).read_text())


def write_registry(reg: dict[str, Any], path: Path = REGISTRY_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(reg, indent=2, default=str) + "\n")


def register(art: ModelArtifact, path_on_disk: Path, reg_path: Path = REGISTRY_PATH) -> None:
    reg = read_registry(reg_path)
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
        "created_utc": art.created_utc,
    }
    write_registry(reg, reg_path)


def serving_artifact_path(reg_path: Path = REGISTRY_PATH) -> Path:
    reg = read_registry(reg_path)
    name = reg.get("serving")
    if not name or name not in reg.get("models", {}):
        raise ArtifactError("no serving model set in the registry")
    return REPO_ROOT / reg["models"][name]["artifact"]

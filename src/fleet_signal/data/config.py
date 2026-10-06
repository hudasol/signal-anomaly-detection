"""Loading and hashing the data-generation and split configs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _find_root() -> Path:
    """Where configs/, models/ and data/ live.

    SIGNAL_HOME if set; else the source checkout this file sits in (editable
    install); else the nearest directory at or above the working directory that
    holds configs/data.yaml (a regular install, e.g. the Docker image).
    """
    env = os.environ.get("SIGNAL_HOME")
    if env:
        return Path(env).resolve()
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "configs" / "data.yaml").exists():
        return checkout
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / "configs" / "data.yaml").exists():
            return d
    return checkout


REPO_ROOT = _find_root()
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs" / "data.yaml"
DEFAULT_SPLITS_CONFIG = REPO_ROOT / "configs" / "splits.yaml"
DEFAULT_DATA_DIR = REPO_ROOT / "data"

FAULT_TYPES: tuple[str, ...] = (
    "overheating",
    "battery_drain",
    "link_degradation",
    "sensor_freeze",
    "motion_anomaly",
)
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
# Source files whose behaviour determines the generated data.
GENERATOR_SOURCES: tuple[str, ...] = ("sim.py", "faults.py", "generator.py", "splits.py")


@dataclass(frozen=True)
class GenerationConfig:
    """Both configs together, plus a content hash that versions the dataset."""

    data: dict[str, Any]
    splits: dict[str, Any]

    @property
    def version(self) -> str:
        """Hash of both configs AND the generator source, so any change gets a new version."""
        payload = json.dumps({"data": self.data, "splits": self.splits}, sort_keys=True)
        h = hashlib.sha256(payload.encode())
        for name in GENERATOR_SOURCES:
            h.update((Path(__file__).parent / name).read_bytes())
        return f"v{self.data['generator_version']}-{h.hexdigest()[:10]}"

    @property
    def n_ticks(self) -> int:
        return int(self.data["run_duration_s"] * self.data["tick_hz"])


def load_config(
    data_path: Path = DEFAULT_DATA_CONFIG,
    splits_path: Path = DEFAULT_SPLITS_CONFIG,
    overrides: dict[str, Any] | None = None,
) -> GenerationConfig:
    """Load configs. `overrides` replaces top-level keys of the data config (tests use this)."""
    data = yaml.safe_load(Path(data_path).read_text())
    splits = yaml.safe_load(Path(splits_path).read_text())
    if overrides:
        data = {**data, **overrides}
    return GenerationConfig(data=data, splits=splits)

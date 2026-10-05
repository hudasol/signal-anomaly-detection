"""Loading ground truth. For evaluation, error analysis and plotting only.

Nothing in feature construction, training or the inference service may import
this module. The feature pipeline sees telemetry only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fleet_signal.data.config import DEFAULT_DATA_DIR
from fleet_signal.data.telemetry import resolve_version_dir


def _gt(version: str | None, data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(resolve_version_dir(version, data_dir) / "ground_truth" / name)


def load_runs(version: str | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """run_id, seed, split, scenario."""
    return _gt(version, data_dir, "runs.parquet")


def load_faults(version: str | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """One row per injected fault: asset, type, variant, window, parameters."""
    return _gt(version, data_dir, "faults.parquet")


def load_episodes(version: str | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Difficult-normal episodes (charging, returning, hard manoeuvre, noisy link)."""
    return _gt(version, data_dir, "episodes.parquet")

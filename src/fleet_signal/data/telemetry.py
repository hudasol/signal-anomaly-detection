"""Loading telemetry. This module is the only data entry point for feature and model code.

It must never import `fleet_signal.data.ground_truth`; a test enforces that.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fleet_signal.data.config import DEFAULT_DATA_DIR


def resolve_version_dir(version: str | None = None, data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Directory for a data version; defaults to the one recorded in data/CURRENT."""
    data_dir = Path(data_dir)
    if version is None:
        current = data_dir / "CURRENT"
        if not current.exists():
            raise FileNotFoundError(
                f"no dataset found in {data_dir}; run `signal-data generate` first"
            )
        version = current.read_text().strip()
    path = data_dir / version
    if not (path / "telemetry.parquet").exists():
        raise FileNotFoundError(f"telemetry for data version {version} not found at {path}")
    return path


def load_telemetry(
    version: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    run_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Telemetry events, sorted by run, asset and time. Optionally filtered to some runs."""
    path = resolve_version_dir(version, data_dir) / "telemetry.parquet"
    filters = [("run_id", "in", run_ids)] if run_ids is not None else None
    df = pd.read_parquet(path, filters=filters)
    return df.sort_values(["run_id", "asset_id", "seq"], ignore_index=True)

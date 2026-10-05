"""Cached feature tables and split guards.

Features are cached next to the data version they were built from:
`data/<data_version>/features/<feature_version>-<schema_hash>.parquet`.
Any change to the feature config or code gets a new file.

Split membership comes from `split_run_ids`, which carries no scenario
information, so nothing here can see labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fleet_signal.data.config import DEFAULT_DATA_DIR, GenerationConfig, load_config
from fleet_signal.data.splits import split_run_ids
from fleet_signal.data.telemetry import load_telemetry, resolve_version_dir
from fleet_signal.features.build import FeatureConfig, build_features


class SplitLeakError(RuntimeError):
    """Raised when rows from the wrong split reach a fitting step."""


def feature_cache_path(
    fcfg: FeatureConfig, version: str | None = None, data_dir: Path = DEFAULT_DATA_DIR
) -> Path:
    base = resolve_version_dir(version, data_dir)
    return base / "features" / f"{fcfg.feature_version}-{fcfg.schema_hash}.parquet"


def load_features(
    split: str | None = None,
    fcfg: FeatureConfig | None = None,
    version: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    gen_cfg: GenerationConfig | None = None,
) -> pd.DataFrame:
    """Features for one split (or all), building and caching them on first use."""
    fcfg = fcfg or FeatureConfig.load()
    path = feature_cache_path(fcfg, version, data_dir)
    if path.exists():
        feats = pd.read_parquet(path)
    else:
        feats = build_features(load_telemetry(version, data_dir), fcfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(path, index=False)
    if split is None:
        return feats
    ids = split_run_ids(gen_cfg or load_config())[split]
    return feats[feats["run_id"].isin(ids)].reset_index(drop=True)


def assert_only_split(
    df: pd.DataFrame, split: str, gen_cfg: GenerationConfig | None = None
) -> None:
    """Refuse to continue if any row belongs to a run outside `split`."""
    allowed = set(split_run_ids(gen_cfg or load_config())[split])
    bad = sorted(set(df["run_id"]) - allowed)
    if bad:
        raise SplitLeakError(f"{len(bad)} run(s) outside '{split}' reached this step: {bad[:5]}")

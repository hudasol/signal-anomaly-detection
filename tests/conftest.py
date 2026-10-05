from __future__ import annotations

import pytest

from fleet_signal.data.config import GenerationConfig, load_config


@pytest.fixture(scope="session")
def cfg() -> GenerationConfig:
    """The real committed config."""
    return load_config()


@pytest.fixture(scope="session")
def small_cfg() -> GenerationConfig:
    """Same config with 5-minute runs, so generation tests stay fast."""
    return load_config(overrides={"run_duration_s": 300})

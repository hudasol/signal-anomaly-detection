"""Split logic: disjoint seeds/runs/time, correct composition, no scenario leakage."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from fleet_signal.data.config import FAULT_TYPES, GenerationConfig
from fleet_signal.data.splits import allowed_variants, build_run_plan, split_run_ids


def test_seeds_and_run_ids_are_disjoint_across_splits(cfg: GenerationConfig) -> None:
    plan = build_run_plan(cfg)
    by_split: dict[str, set[int]] = {}
    for run in plan:
        by_split.setdefault(run.split, set()).add(run.seed)
    assert by_split["train"].isdisjoint(by_split["validation"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["validation"].isdisjoint(by_split["test"])
    assert len({r.run_id for r in plan}) == len(plan)


def test_runs_are_disjoint_in_time_test_after_everything(cfg: GenerationConfig) -> None:
    plan = build_run_plan(cfg)
    epoch = pd.Timestamp(cfg.data["start_epoch_utc"])
    spacing = cfg.data["run_spacing_s"]
    assert cfg.data["run_duration_s"] <= spacing  # runs never overlap
    start = {r.split: [] for r in plan}
    for r in plan:
        start[r.split].append(epoch + pd.Timedelta(seconds=r.seed * spacing))
    assert max(start["train"]) < min(start["validation"])
    assert max(start["validation"]) < min(start["test"])


def test_composition_matches_config(cfg: GenerationConfig) -> None:
    plan = build_run_plan(cfg)
    for split, spec in cfg.splits["splits"].items():
        counts = Counter(r.scenario for r in plan if r.split == split)
        assert dict(counts) == spec["composition"]


def test_train_is_normal_only(cfg: GenerationConfig) -> None:
    assert {r.scenario for r in build_run_plan(cfg) if r.split == "train"} == {"normal"}


def test_faults_balanced_over_assets_and_variants(cfg: GenerationConfig) -> None:
    plan = build_run_plan(cfg)
    n_assets = len(cfg.data["fleet"])
    for split in ("validation", "test"):
        include = cfg.splits["splits"][split]["include_test_only_variants"]
        for fault in FAULT_TYPES:
            runs = [r for r in plan if r.split == split and r.scenario == fault]
            assets = Counter(r.fault_asset_index for r in runs)
            assert max(assets.values()) - min(assets.values()) <= 1
            assert len(assets) == n_assets
            variants = Counter(r.variant for r in runs)
            allowed = allowed_variants(cfg, fault, include)
            assert set(variants) <= set(allowed)
            expected_present = min(len(allowed), len(runs))
            assert len(variants) == expected_present
            assert max(variants.values()) - min(variants.values()) <= 1


def test_test_only_variants_never_in_validation(cfg: GenerationConfig) -> None:
    plan = build_run_plan(cfg)
    for run in plan:
        if run.split != "validation" or run.variant is None:
            continue
        assert not cfg.data["faults"][run.scenario]["variants"][run.variant]["test_only"]


def test_test_set_contains_unseen_variants(cfg: GenerationConfig) -> None:
    plan = build_run_plan(cfg)
    seen = {(r.scenario, r.variant) for r in plan if r.split == "validation" and r.variant}
    unseen = {(r.scenario, r.variant) for r in plan if r.split == "test" and r.variant} - seen
    assert unseen, "test must include fault variants never used for threshold selection"


def test_plan_is_deterministic(cfg: GenerationConfig) -> None:
    assert build_run_plan(cfg) == build_run_plan(cfg)


def test_split_run_ids_match_plan_and_carry_no_scenario(cfg: GenerationConfig) -> None:
    plan = build_run_plan(cfg)
    ids = split_run_ids(cfg)
    for split, run_ids in ids.items():
        assert sorted(run_ids) == sorted(r.run_id for r in plan if r.split == split)
        for run_id in run_ids:
            assert not any(name in run_id for name in (*FAULT_TYPES, "normal"))


def test_run_order_does_not_reveal_scenario(cfg: GenerationConfig) -> None:
    """Scenarios are shuffled within a split, not grouped by seed."""
    val = [r.scenario for r in build_run_plan(cfg) if r.split == "validation"]
    first_ten = set(val[:10])
    assert len(first_ten) > 1

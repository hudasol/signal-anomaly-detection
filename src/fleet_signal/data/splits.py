"""Expanding splits.yaml into a concrete run plan.

Two views exist on purpose:

* `build_run_plan` returns every run with its scenario. Only the generator and
  the evaluation code use it; it is ground truth.
* `split_run_ids` returns run ids per split with no scenario information. This
  is the only split view the feature and training code are allowed to use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fleet_signal.data.config import FAULT_TYPES, SPLIT_NAMES, GenerationConfig


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    seed: int
    split: str
    scenario: str  # "normal" or a fault type
    fault_asset_index: int | None  # index into the fleet list, None for normal runs
    variant: str | None  # fault variant, None for normal runs


def run_id_for_seed(seed: int) -> str:
    return f"r{seed}"


def build_run_plan(cfg: GenerationConfig) -> list[PlannedRun]:
    """Deterministically assign scenarios and faulted assets to seeds."""
    n_assets = len(cfg.data["fleet"])
    rng = np.random.default_rng(cfg.splits["assignment_seed"])
    plan: list[PlannedRun] = []

    for split in SPLIT_NAMES:
        spec = cfg.splits["splits"][split]
        composition: dict[str, int] = spec["composition"]
        unknown = set(composition) - {"normal", *FAULT_TYPES}
        if unknown:
            raise ValueError(f"unknown scenarios in split {split}: {sorted(unknown)}")
        if split == "train" and set(composition) != {"normal"}:
            raise ValueError("train must contain normal runs only")

        scenarios: list[str] = []
        assets: list[int | None] = []
        variants: list[str | None] = []
        for scenario, count in composition.items():
            scenarios.extend([scenario] * count)
            if scenario == "normal":
                assets.extend([None] * count)
                variants.extend([None] * count)
                continue
            # Balanced: each fault type hits each asset equally often, and each allowed
            # variant appears equally often. The variant order is shuffled separately so
            # variant and asset are not locked together.
            assets.extend([i % n_assets for i in range(count)])
            allowed = allowed_variants(cfg, scenario, spec["include_test_only_variants"])
            balanced = [allowed[i % len(allowed)] for i in range(count)]
            variants.extend(balanced[j] for j in rng.permutation(count))

        order = rng.permutation(len(scenarios))
        for offset, idx in enumerate(order):
            seed = spec["seed_start"] + offset
            plan.append(
                PlannedRun(
                    run_id=run_id_for_seed(seed),
                    seed=seed,
                    split=split,
                    scenario=scenarios[idx],
                    fault_asset_index=assets[idx],
                    variant=variants[idx],
                )
            )
    _check_disjoint(plan)
    return plan


def allowed_variants(cfg: GenerationConfig, fault_type: str, include_test_only: bool) -> list[str]:
    spec = cfg.data["faults"][fault_type]["variants"]
    return [name for name, v in spec.items() if include_test_only or not v["test_only"]]


def split_run_ids(cfg: GenerationConfig) -> dict[str, list[str]]:
    """Run ids per split, without scenarios. Safe for feature/training code."""
    out: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        spec = cfg.splits["splits"][split]
        n = sum(spec["composition"].values())
        out[split] = [run_id_for_seed(spec["seed_start"] + i) for i in range(n)]
    return out


def _check_disjoint(plan: list[PlannedRun]) -> None:
    seeds = [r.seed for r in plan]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed ranges overlap between splits")

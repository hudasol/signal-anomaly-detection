"""`signal-train`: fit detectors on train, freeze thresholds chosen on validation, save artifacts.

    signal-train                    rule, stats and lof -> models/ + models/registry.json

Deterministic: the same data version and code give the same model versions and
thresholds, and (since 2026-10-06) the same artifact bytes; creation time and
git SHA live in metadata.json, not in the pickle.

Evaluated models are never overwritten. If the registry entry already has an
official test result and the committed artifact on disk matches its SHA-256,
training keeps it untouched; replacing it is refused.
"""

from __future__ import annotations

import argparse
import json
import time

from fleet_signal.data.config import load_config
from fleet_signal.data.splits import split_run_ids
from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.lof import LOFDetector
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.eval.protocol import EvalConfig
from fleet_signal.eval.validate import VALIDATION_DIR, select_on_validation
from fleet_signal.features.build import FeatureConfig
from fleet_signal.registry import (
    EvaluatedModelError,
    ModelArtifact,
    git_sha,
    is_frozen_and_present,
    read_registry,
    register,
    save_artifact,
)

ML_SELECTION = VALIDATION_DIR / "ml_selection.json"

FACTORIES: dict[str, type[Detector]] = {
    "rule": RuleDetector,
    "stats": StatsDetector,
    "lof": LOFDetector,
}


def _make(name: str) -> Detector:
    """The ML detector is built with the hyperparameters chosen by `signal-eval select-model`."""
    if name == "lof" and ML_SELECTION.exists():
        chosen = json.loads(ML_SELECTION.read_text())["chosen"]
        if chosen["model"] != "lof":
            raise SystemExit(f"model selection chose {chosen['model']}, not lof")
        return LOFDetector(**chosen["params"])
    return FACTORIES[name]()


def train_and_freeze(names: list[str]) -> list[ModelArtifact]:
    gen_cfg, ecfg, fcfg = load_config(), EvalConfig.load(), FeatureConfig.load()
    detectors = [_make(n) for n in names]
    sel = select_on_validation(detectors, gen_cfg, ecfg)  # fits on train, selects on validation
    sha = git_sha()
    arts = []
    for det in detectors:
        pick = sel.picks[det.name]
        art = ModelArtifact(
            detector=det,
            detector_name=det.name,
            threshold=float(pick["threshold"]),
            incident_params=sel.params.as_dict(),
            feature_config=fcfg.as_dict(),
            feature_schema_hash=fcfg.schema_hash,
            data_version=gen_cfg.version,
            train_run_ids=split_run_ids(gen_cfg)["train"],
            validation_summary={**sel.results[det.name].summary, "feasible": pick["feasible"]},
            git_sha=sha,
        )
        if is_frozen_and_present(det.name, art.model_version):
            print(f"{art.model_version}: evaluated artifact present and verified; kept as is")
        else:
            register_guard(det.name)
            path = save_artifact(art)
            register(art, path)
        arts.append(art)
    return arts


def register_guard(name: str) -> None:
    """Fail BEFORE writing a file if this would replace an evaluated model."""
    entry = read_registry()["models"].get(name)
    if entry and entry.get("official_test_result"):
        raise EvaluatedModelError(
            f"{name}: the evaluated artifact {entry['artifact']} is missing or does not match "
            "its registry SHA-256, and this registry entry has an official test result. "
            "Restore it with `git checkout -- models/` instead of retraining."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="signal-train", description=__doc__)
    parser.add_argument("--detectors", default="rule,stats,lof")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    arts = train_and_freeze([n.strip() for n in args.detectors.split(",") if n.strip()])
    for art in arts:
        print(f"{art.model_version:<22} threshold={art.threshold:.4f} params={art.incident_params}")
    print(f"done in {time.perf_counter() - started:.1f}s; registry: models/registry.json")


if __name__ == "__main__":
    main()

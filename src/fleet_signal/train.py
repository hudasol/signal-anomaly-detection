"""`signal-train`: fit detectors on train, freeze thresholds chosen on validation, save artifacts.

    signal-train                    rule, stats and lof -> models/ + models/registry.json

Deterministic: the same data version and code give the same model versions and
thresholds. Artifact bytes differ between builds only because each artifact
records its own creation time and git SHA; the registry stores the SHA-256 of
the exact file that was evaluated.
"""

from __future__ import annotations

import argparse
import time

from fleet_signal.data.config import load_config
from fleet_signal.data.splits import split_run_ids
from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.lof import LOFDetector
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.eval.protocol import EvalConfig
from fleet_signal.eval.validate import select_on_validation
from fleet_signal.features.build import FeatureConfig
from fleet_signal.registry import ModelArtifact, git_sha, register, save_artifact

FACTORIES: dict[str, type[Detector]] = {
    "rule": RuleDetector,
    "stats": StatsDetector,
    "lof": LOFDetector,
}


def train_and_freeze(names: list[str]) -> list[ModelArtifact]:
    gen_cfg, ecfg, fcfg = load_config(), EvalConfig.load(), FeatureConfig.load()
    detectors = [FACTORIES[n]() for n in names]
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
        path = save_artifact(art)
        register(art, path)
        arts.append(art)
    return arts


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

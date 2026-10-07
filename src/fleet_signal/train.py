"""`signal-train`: fit detectors on train, freeze thresholds chosen on validation, save artifacts.

    signal-train    rule, stats, lof and the v2-selected system -> models/, models/registry.json

v2: run `signal-eval select-v2` first. Baselines and LOF use the shared incident
grouping; the selected system uses its own validation-chosen grouping (open after
1 or 2 alerts) and threshold.

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
from fleet_signal.eval.validate import select_on_validation
from fleet_signal.features.build import FeatureConfig
from fleet_signal.incidents.grouping import IncidentParams
from fleet_signal.registry import (
    EvaluatedModelError,
    ModelArtifact,
    git_sha,
    is_frozen_and_present,
    read_registry,
    register,
    save_artifact,
)

FACTORIES: dict[str, type[Detector]] = {
    "rule": RuleDetector,
    "stats": StatsDetector,
    "lof": LOFDetector,
}


def _selection() -> dict[str, object]:
    from fleet_signal.eval.select_v2 import SELECTION

    if not SELECTION.exists():
        raise SystemExit("run `signal-eval select-v2` first (results/validation/v2_selection.json)")
    return dict(json.loads(SELECTION.read_text()))


def _make(name: str, sel: dict[str, object]) -> Detector:
    from fleet_signal.eval.select_v2 import build_candidate

    if name in FACTORIES and name != "lof":
        return FACTORIES[name]()
    return build_candidate(name, int(sel["lof_k"]))  # type: ignore[call-overload]


def train_and_freeze(names: list[str] | None = None) -> list[ModelArtifact]:
    from fleet_signal.eval.select_v2 import chosen_params

    gen_cfg, ecfg, fcfg = load_config(), EvalConfig.load(), FeatureConfig.load()
    sel_json = _selection()
    chosen = str(sel_json["chosen"]["name"])  # type: ignore[index]
    names = names or list(dict.fromkeys(["rule", "stats", "lof", chosen]))
    shared = IncidentParams(**sel_json["shared_incident_params"])  # type: ignore[arg-type]
    detectors, sels = [], {}
    for n in names:
        det = _make(n, sel_json)
        params = chosen_params(sel_json) if n == chosen else shared
        # fits on train, selects the threshold on validation (deterministic)
        sels[n] = select_on_validation([det], gen_cfg, ecfg, params)
        detectors.append(det)
    sha = git_sha()
    arts = []
    for det in detectors:
        sel = sels[det.name]
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
    parser.add_argument("--detectors", default=None, help="default: rule,stats,lof,<selected>")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    names = [n.strip() for n in args.detectors.split(",") if n.strip()] if args.detectors else None
    arts = train_and_freeze(names)
    for art in arts:
        print(f"{art.model_version:<22} threshold={art.threshold:.4f} params={art.incident_params}")
    print(f"done in {time.perf_counter() - started:.1f}s; registry: models/registry.json")


if __name__ == "__main__":
    main()

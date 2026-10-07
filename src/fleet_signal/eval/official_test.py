"""The official, run-once test evaluation.

Order of operations (every check happens before the test split is read):

1. load each frozen artifact from the registry and verify its SHA-256;
2. check it was trained on the current data version;
3. check no official result already exists for any of them;
4. only then load test features, score, and evaluate at the frozen threshold
   and incident params;
5. write per-detector results, the comparison, and the ship decision with the
   test-once guard; point the registry's `serving` entry at the shipped model.

Post-hoc analysis (threshold curves on test, used for the sensitivity plot and
the demo) is computed afterwards, written to results/official/posthoc/, and
labelled as post-hoc. It never feeds back into a frozen artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fleet_signal.data.config import REPO_ROOT, load_config
from fleet_signal.data.ground_truth import load_faults
from fleet_signal.data.splits import split_run_ids
from fleet_signal.eval.bootstrap import bootstrap_ci, paired_difference, paired_latency
from fleet_signal.eval.decision import ship_decision_v2
from fleet_signal.eval.detectability import with_detectability
from fleet_signal.eval.official import OFFICIAL_DIR, official_path, write_official
from fleet_signal.eval.protocol import EvalConfig, ScoredSet, breakdown, meets_bar
from fleet_signal.eval.threshold import sweep
from fleet_signal.eval.validate import _jsonable, scored_frame, seen_variants
from fleet_signal.features.store import assert_only_split, load_features
from fleet_signal.registry import (
    ArtifactError,
    ModelArtifact,
    git_sha,
    load_artifact,
    read_registry,
    write_registry,
)


def _load_frozen(names: list[str], data_version: str) -> dict[str, ModelArtifact]:
    reg = read_registry()
    arts: dict[str, ModelArtifact] = {}
    for name in names:
        entry = reg["models"].get(name)
        if entry is None:
            raise ArtifactError(f"{name} is not in the registry; run signal-train first")
        path = REPO_ROOT / entry["artifact"]
        art = load_artifact(path, expected_sha256=entry["artifact_sha256"])  # verified first
        if art.data_version != data_version:
            raise ArtifactError(f"{name} was trained on {art.data_version}, data is {data_version}")
        arts[name] = art
    return arts


def comparison_key(arts: dict[str, ModelArtifact]) -> str:
    joined = "|".join(sorted(a.model_version for a in arts.values()))
    return hashlib.sha256(joined.encode()).hexdigest()[:10]


def v2_names() -> tuple[tuple[str, ...], str, str]:
    """(detectors to evaluate, the selected candidate, the best baseline) from v2 selection."""
    from fleet_signal.eval.select_v2 import SELECTION

    sel = json.loads(SELECTION.read_text())
    cand = sel["chosen"]["name"]
    names = tuple(dict.fromkeys(("rule", "stats", "lof", cand)))
    return names, cand, sel["best_baseline"]


def run_official_test(names: tuple[str, ...] | None = None) -> dict[str, Any]:
    gen_cfg, ecfg = load_config(), EvalConfig.load()
    default_names, candidate, chosen_baseline = v2_names()
    names = names or default_names
    arts = _load_frozen(list(names), gen_cfg.version)
    key = comparison_key(arts)
    targets = [official_path(n, a.model_version) for n, a in arts.items()]
    targets.append(official_path("report", key))
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise FileExistsError(f"official results already exist, refusing to re-run: {existing}")

    # ---- the test split is touched for the first time here ----
    test = load_features("test", gen_cfg=gen_cfg)
    assert_only_split(test, "test", gen_cfg)
    test_ids = split_run_ids(gen_cfg)["test"]
    faults = with_detectability(load_faults(), gen_cfg)
    n_assets = len(gen_cfg.data["fleet"])
    seen = seen_variants(gen_cfg)

    results, payloads = {}, {}
    for name, art in arts.items():
        ss = ScoredSet(scored_frame(art.detector, test), faults, test_ids, ecfg, n_assets, seen)
        res = ss.evaluate(art.threshold, art.params)
        results[name] = (ss, res)
        per_fault = res.per_fault
        payloads[name] = _jsonable(
            {
                "split": "test",
                "official": True,
                "detector": name,
                "model_version": art.model_version,
                "artifact_sha256": read_registry()["models"][name]["artifact_sha256"],
                "frozen_threshold": art.threshold,
                "incident_params": art.incident_params,
                "data_version": art.data_version,
                "feature_schema_hash": art.feature_schema_hash,
                "evaluated_utc": datetime.now(UTC).isoformat(),
                "git_sha": git_sha(),
                "summary": res.summary,
                "meets_bar": meets_bar(res.summary, ecfg),
                "ci95": bootstrap_ci(res.per_run, n_assets, ecfg.n_resamples, ecfg.bootstrap_seed),
                "by_fault_type": breakdown(per_fault, ["fault_type"]).to_dict("records"),
                "by_variant": breakdown(per_fault, ["fault_type", "variant"]).to_dict("records"),
                "by_seen_variant": breakdown(per_fault, ["seen_variant"]).to_dict("records"),
                "by_asset": breakdown(per_fault, ["asset_id"]).to_dict("records"),
                "window_confusion": res.window_confusion,
                "per_fault": per_fault.to_dict("records"),
                "incidents": res.incidents.to_dict("records"),
            }
        )

    summaries = {n: results[n][1].summary for n in arts}
    paired = {
        f"{a}_minus_{b}": paired_difference(
            results[a][1].per_run, results[b][1].per_run, n_assets,
            ecfg.n_resamples, ecfg.bootstrap_seed,
        )
        for i, a in enumerate(arts) for b in list(arts)[i + 1 :]
    }  # fmt: skip
    decision: dict[str, Any] | None = None
    if candidate in arts and chosen_baseline in arts:
        base = chosen_baseline  # chosen on VALIDATION recall (v2 fix)
        diff = paired_difference(
            results[candidate][1].per_run, results[base][1].per_run, n_assets,
            ecfg.n_resamples, ecfg.bootstrap_seed,
        )  # fmt: skip
        lat = paired_latency(
            results[candidate][1].per_fault, results[base][1].per_fault, ecfg.progressive,
            ecfg.n_resamples, ecfg.bootstrap_seed,
        )  # fmt: skip
        decision = ship_decision_v2(candidate, base, diff["recall"], diff["precision"], lat)
        decision["candidate_minus_baseline"] = diff
        decision["latency_candidate_minus_baseline"] = lat

    report = _jsonable(
        {
            "split": "test",
            "official": True,
            "comparison_key": key,
            "data_version": gen_cfg.version,
            "incident_params": {n: a.incident_params for n, a in arts.items()},
            "false_alert_budget_per_10min": ecfg.budget,
            "min_precision": ecfg.min_precision,
            "bar": ecfg.bar,
            "detectors": {
                n: {
                    "model_version": arts[n].model_version,
                    "selected": {**summaries[n], "threshold": arts[n].threshold},
                    "summary": summaries[n],
                    "meets_bar": payloads[n]["meets_bar"],
                    "ci95": payloads[n]["ci95"],
                    "by_fault_type": payloads[n]["by_fault_type"],
                    "by_seen_variant": payloads[n]["by_seen_variant"],
                    "window_confusion": payloads[n]["window_confusion"],
                }
                for n in arts
            },
            "paired": paired,
            "decision": decision,
        }
    )

    written = [write_official(n, arts[n].model_version, payloads[n]) for n in arts]
    written.append(write_official("report", key, report))

    reg = read_registry()
    for n, art in arts.items():
        rel = official_path(n, art.model_version).relative_to(REPO_ROOT)
        reg["models"][n]["official_test_result"] = str(rel)
    if decision is not None:
        reg["serving"] = decision["ship"]
        reg["ship_decision"] = {k: decision[k] for k in ("ship", "best_baseline", "reason")}
    reg["official_test_report"] = str(official_path("report", key).relative_to(REPO_ROOT))
    write_registry(reg)

    _posthoc(results, key)
    return {"report": report, "written": [str(p) for p in written]}


def _posthoc(results: dict[str, Any], key: str, root: Path = OFFICIAL_DIR / "posthoc") -> None:
    """Threshold curves on test, AFTER the official result is recorded. Analysis only."""
    root.mkdir(parents=True, exist_ok=True)
    for name, (ss, res) in results.items():
        params = res.summary
        from fleet_signal.incidents.grouping import IncidentParams

        p = IncidentParams(params["open_n"], params["close_m"], params["cooldown_c"])
        sweep(ss, p).to_csv(root / f"{name}_threshold_curve.csv", index=False)
    (root / "README.md").write_text(
        "Post-hoc analysis on the test split, computed after the official result for "
        f"comparison {key} was written. Used for the threshold-sensitivity plot and the demo. "
        "Nothing here was used to choose a threshold, feature or model.\n"
    )
    report = json.loads(official_path("report", key).read_text())
    (root / "test_report_for_plots.json").write_text(json.dumps(report, indent=2))

"""Acceptance-demo helpers. Read-only with respect to the official result.

* `audit()`: prove the split: seed ranges, which runs each frozen artifact was
  fitted on, which runs selection used, and that the test runs appear in neither.
* `show()`: the official comparison, printed from the saved report file.
* `threshold_demo()`: evaluate a frozen model on test at a DIFFERENT threshold,
  to show the trade-off. Writes to results/demo/ only, never to the official
  files, and the frozen artifact is not modified.
"""

from __future__ import annotations

import json
from typing import Any

from fleet_signal.data.config import REPO_ROOT, load_config
from fleet_signal.data.ground_truth import load_faults
from fleet_signal.data.splits import split_run_ids
from fleet_signal.eval.official import OFFICIAL_DIR, write_demo
from fleet_signal.eval.protocol import EvalConfig, ScoredSet
from fleet_signal.eval.validate import VALIDATION_DIR, scored_frame, seen_variants
from fleet_signal.features.store import load_features
from fleet_signal.registry import load_artifact, read_registry


def audit() -> list[str]:
    gen_cfg = load_config()
    ids = split_run_ids(gen_cfg)
    reg = read_registry()
    lines = [f"data version {gen_cfg.version}"]
    for split, spec in gen_cfg.splits["splits"].items():
        n = len(ids[split])
        lo, hi = spec["seed_start"], spec["seed_start"] + n - 1
        lines.append(f"  {split:<10} seeds {lo}-{hi}  ({n} runs)")
    train, val, test = set(ids["train"]), set(ids["validation"]), set(ids["test"])
    lines.append(f"pairwise overlap of run ids: train/val {len(train & val)}, "
                 f"train/test {len(train & test)}, val/test {len(val & test)}")  # fmt: skip
    for entry in reg["models"].values():
        art = load_artifact(REPO_ROOT / entry["artifact"])
        fitted = set(art.train_run_ids)
        fitted_on = set(getattr(art.detector, "fitted_runs", [])) or fitted
        lines.append(
            f"{entry['model_version']}: fitted on {len(fitted_on)} runs, all in train: "
            f"{fitted_on <= train}; any test run used: {bool(fitted_on & test)}"
        )
    vrep = json.loads((VALIDATION_DIR / "validation_report.json").read_text())
    used = set()
    for name in vrep["detectors"]:
        pf = VALIDATION_DIR / f"{name}_per_fault.csv"
        if pf.exists():
            used |= {line.split(",")[0] for line in pf.read_text().splitlines()[1:]}
    lines.append(f"runs referenced in the validation selection outputs: {len(used)}, all in "
                 f"validation: {used <= val}; any test run: {bool(used & test)}")  # fmt: skip
    if "official_test_report" in reg:
        lines.append(f"official test report: {reg['official_test_report']}")
    return lines


def show() -> list[str]:
    reg = read_registry()
    path = REPO_ROOT / reg["official_test_report"]
    rep = json.loads(path.read_text())
    cols = ["precision", "recall", "f1", "fp_per_10min", "progressive_latency_median"]
    out = [f"OFFICIAL TEST ({path.name}), frozen thresholds, incident params "
           f"{rep['incident_params']}", ""]  # fmt: skip
    out.append("detector  model version      " + "".join(c[:12].rjust(13) for c in cols))
    for name, d in rep["detectors"].items():
        s = d["summary"]
        out.append(f"{name:<9} {d['model_version']:<18}" + "".join(
            f"{s[c]:13.3f}" for c in cols))  # fmt: skip
    dec = rep.get("decision")
    if dec:
        r, lat = dec["recall_ml_minus_baseline"], dec["latency_ml_minus_baseline"]
        ci = f"(CI {r['ci_low']:+.3f}, {r['ci_high']:+.3f})"
        out += ["", f"LOF - rule recall {r['diff']:+.3f} {ci}",
                f"LOF - rule latency {lat['median_diff']:+.1f} events (CI {lat['ci_low']:+.1f}, "
                f"{lat['ci_high']:+.1f}, n={lat['n_paired']})",
                f"SHIP: {dec['ship']} - {dec['reason']}"]  # fmt: skip
    return out


def threshold_demo(detector: str, threshold: float) -> dict[str, Any]:
    gen_cfg, ecfg = load_config(), EvalConfig.load()
    entry = read_registry()["models"][detector]
    art = load_artifact(REPO_ROOT / entry["artifact"])
    test = load_features("test", gen_cfg=gen_cfg)
    ss = ScoredSet(
        scored_frame(art.detector, test), load_faults(), split_run_ids(gen_cfg)["test"],
        ecfg, len(gen_cfg.data["fleet"]), seen_variants(gen_cfg),
    )  # fmt: skip
    keys = ["precision", "recall", "fp_per_10min", "n_false_incidents",
            "progressive_latency_median"]  # fmt: skip
    frozen = ss.evaluate(art.threshold, art.params).summary
    demo = ss.evaluate(threshold, art.params).summary
    payload = {
        "DEMO_ONLY": True,
        "note": "Exploratory threshold change on test for the acceptance demo. Not the official "
        "result; the frozen artifact and results/official/ are untouched.",
        "detector": detector,
        "model_version": art.model_version,
        "frozen_threshold": art.threshold,
        "demo_threshold": threshold,
        "frozen": {k: frozen[k] for k in keys},
        "demo": {k: demo[k] for k in keys},
    }
    write_demo(f"threshold_{detector}_{threshold:g}", payload)
    return payload


def official_files() -> list[str]:
    return sorted(p.name for p in OFFICIAL_DIR.glob("test_*.json"))

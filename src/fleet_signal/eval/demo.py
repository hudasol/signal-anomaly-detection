"""Acceptance-demo helpers. Read-only with respect to the official result.

* `audit()`: prove the split: seed ranges, which runs each frozen artifact was
  fitted on, which runs selection used, and that the test runs appear in neither.
* `show()`: the official comparison, printed from the saved report file.
* `threshold_demo()`: evaluate a frozen model on test at a DIFFERENT threshold,
  to show the trade-off. Writes to results/demo/ only, never to the official
  files, and the frozen artifact is not modified.
* `fragmentation()`: post-hoc, from the SAVED official files: how many incidents
  each detected fault produced, and precision counted per fault.
* `rule_envelopes()`: the train-normal envelopes the rule limits were set from.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

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
        art = load_artifact(REPO_ROOT / entry["artifact"], expected_sha256=entry["artifact_sha256"])
        fitted_on = set(getattr(art.detector, "fitted_runs", []))
        if not fitted_on:  # the rule detector has no fit
            lines.append(
                f"{entry['model_version']}: no fit; limits hand-set from train-normal envelopes "
                "(results/validation/rule_envelopes_train.csv); threshold chosen on validation"
            )
            continue
        lines.append(
            f"{entry['model_version']}: fitted on {len(fitted_on)} runs, all in train: "
            f"{fitted_on <= train}; any test run used: {bool(fitted_on & test)}"
        )
    vrep = json.loads((VALIDATION_DIR / "validation_report.json").read_text())
    used: set[str] = set()
    for name in vrep["detectors"]:
        for kind in ("per_fault", "incidents"):
            f = VALIDATION_DIR / f"{name}_{kind}.csv"
            if f.exists():
                used |= set(pd.read_csv(f, usecols=["run_id"])["run_id"])
    lines.append(f"runs referenced in the validation selection outputs: {len(used)}, all in "
                 f"validation: {used <= val}; any test run: {bool(used & test)}")  # fmt: skip
    if "official_test_report" in reg:
        lines.append(f"official test report: {reg['official_test_report']}")
    lines += code_provenance(reg)
    return lines


SCORING_CODE = ("src/fleet_signal/detectors", "src/fleet_signal/features",
                "src/fleet_signal/incidents", "src/fleet_signal/service")  # fmt: skip


def code_provenance(reg: dict[str, Any]) -> list[str]:
    """Live, from git: what scoring code changed since the models were frozen."""
    import subprocess

    from fleet_signal.registry import detector_code_hash

    frozen = reg.get("provenance", {}).get("frozen_code_commit")
    out = [f"detector_code_hash now: {detector_code_hash()}"]
    if not frozen:
        return out
    try:
        log = subprocess.run(
            ["git", "log", "--format=%h %s", f"{frozen}..HEAD", "--", *SCORING_CODE],
            capture_output=True, text=True, check=True, cwd=REPO_ROOT,
        ).stdout.strip().splitlines()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return out + ["(git history unavailable)"]
    out.append(f"scoring-code commits since freeze ({frozen}): {len(log)}")
    out += [f"  {line[:100]}" for line in log]
    return out


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
    art = load_artifact(REPO_ROOT / entry["artifact"], expected_sha256=entry["artifact_sha256"])
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


def fragmentation_table(per_fault: pd.DataFrame, incidents: pd.DataFrame) -> dict[str, Any]:
    """Incidents per detected fault, and precision counted per fault instead of per incident.

    Per-fault precision = detected faults / (detected faults + false incidents): a fault
    that produced six incidents counts once, so splitting a fault into many incidents
    cannot raise it.
    """
    tp = incidents[incidents["true_positive"]]
    per = tp.groupby(["run_id", "asset_id"]).size()
    detected = int(per_fault["detected"].sum())
    false_inc = int((~incidents["true_positive"]).sum())
    return {
        "detected_faults": detected,
        "true_incidents": int(len(tp)),
        "false_incidents": false_inc,
        "incidents_per_detected_fault": float(per.mean()) if len(per) else float("nan"),
        "max_incidents_on_one_fault": int(per.max()) if len(per) else 0,
        "faults_with_more_than_one_incident": int((per > 1).sum()),
        "incident_precision": float(len(tp) / len(incidents)) if len(incidents) else float("nan"),
        "per_fault_precision": (
            float(detected / (detected + false_inc)) if detected + false_inc else float("nan")
        ),
    }


def fragmentation() -> dict[str, Any]:
    """Computed from the saved official result files only; written to posthoc/."""
    out: dict[str, Any] = {
        "POST_HOC": True,
        "note": "Computed after the official result from the saved files; selects nothing.",
        "test": {},
        "validation": {},
    }
    reg = read_registry()
    for name, entry in reg["models"].items():
        off = json.loads((REPO_ROOT / entry["official_test_result"]).read_text())
        out["test"][name] = fragmentation_table(
            pd.DataFrame(off["per_fault"]), pd.DataFrame(off["incidents"])
        )
        pf, inc = VALIDATION_DIR / f"{name}_per_fault.csv", VALIDATION_DIR / f"{name}_incidents.csv"
        if pf.exists() and inc.exists():
            out["validation"][name] = fragmentation_table(pd.read_csv(pf), pd.read_csv(inc))
    path = OFFICIAL_DIR / "posthoc" / "fragmentation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


ENVELOPE_FEATURES: tuple[str, ...] = (
    "temp_slope_l", "link_min_m", "link_std_m", "speed_mismatch", "jump_excess",
)  # fmt: skip


def rule_envelopes() -> pd.DataFrame:
    """Train-normal envelopes the rule limits in configs/detectors.yaml were set from."""
    from fleet_signal.features.build import scorable

    tr = load_features("train")
    tr = tr[scorable(tr)]
    rows: list[dict[str, Any]] = []
    for at, g in tr.groupby("asset_type"):
        for feat in ENVELOPE_FEATURES:
            v = g[feat]
            rows.append({"asset_type": at, "mode": "*", "feature": feat, "condition": "all",
                         "min": v.min(), "q0.1%": v.quantile(0.001), "median": v.median(),
                         "q99.9%": v.quantile(0.999), "max": v.max()})  # fmt: skip
        settled = g[g["mode_age"] >= 120]
        for md, gm in settled.groupby("mode"):
            v = gm["batt_slope_l"]
            rows.append({"asset_type": at, "mode": md, "feature": "batt_slope_l",
                         "condition": "mode_age>=120", "min": v.min(),
                         "q0.1%": v.quantile(0.001), "median": v.median(),
                         "q99.9%": v.quantile(0.999), "max": v.max()})  # fmt: skip
        moving = g[g["mode"].isin(["moving", "returning"])]
        for feat in ("temp_unchanged", "batt_unchanged", "speed_unchanged", "pos_unchanged"):
            rows.append({"asset_type": at, "mode": "moving|returning", "feature": feat,
                         "condition": "moving", "max": moving[feat].max()})  # fmt: skip
        low = moving[moving["link_pct"] < 99]
        rows.append({"asset_type": at, "mode": "moving|returning", "feature": "link_unchanged",
                     "condition": "moving, link<99",
                     "max": low["link_unchanged"].max()})  # fmt: skip
    df = pd.DataFrame(rows)
    df.to_csv(VALIDATION_DIR / "rule_envelopes_train.csv", index=False, float_format="%.3f")
    return df

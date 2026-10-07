"""Exceeds-the-bar analyses for v2. None of these change a frozen model or an official result.

    frozen_validation_report()   threshold curves on VALIDATION for all four frozen artifacts
                                 (results/validation/v2_frozen/), so the shipped operating point
                                 can be justified on the data it was chosen on
    ablation()                   VALIDATION: remove parts / signals / the speed covariate from the
                                 shipped design and re-select its threshold
    shifted_fleets()             generalisation: generate fleets with a different distribution
                                 (hot climate, noisier sensors, aged batteries), evaluate the
                                 frozen models at their frozen thresholds, and run the drift monitor

Validation-only work selects nothing new for shipping; shifted-fleet results are post-hoc.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.data.config import REPO_ROOT, GenerationConfig, load_config
from fleet_signal.data.ground_truth import load_faults
from fleet_signal.data.splits import build_run_plan, split_run_ids
from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.fastpath import SIGNALS, FastPathDetector
from fleet_signal.detectors.hybrid import HybridDetector
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.eval.detectability import with_detectability
from fleet_signal.eval.protocol import EvalConfig, ScoredSet
from fleet_signal.eval.threshold import select_threshold, sweep
from fleet_signal.eval.validate import (
    VALIDATION_DIR,
    build_scored_set,
    fit_on_train,
    scored_frame,
    seen_variants,
)
from fleet_signal.incidents.grouping import IncidentParams
from fleet_signal.registry import ModelArtifact, load_artifact, read_registry

EXCEEDS_DIR = REPO_ROOT / "results" / "exceeds"
FROZEN_VAL_DIR = VALIDATION_DIR / "v2_frozen"
SUMMARY_KEYS = ("threshold", "precision", "per_fault_precision", "recall", "fp_per_10min",
                "n_false_incidents", "progressive_latency_median", "expected_latency_median",
                "n_expected", "n_expected_detected")  # fmt: skip


def _brief(s: dict[str, Any]) -> dict[str, Any]:
    return {k: s.get(k) for k in SUMMARY_KEYS}


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def frozen_artifacts() -> dict[str, ModelArtifact]:
    """Every registered v2 artifact, SHA-256 verified before loading."""
    reg = read_registry()
    return {
        name: load_artifact(REPO_ROOT / e["artifact"], expected_sha256=e["artifact_sha256"])
        for name, e in reg["models"].items()
    }


# ---------------------------------------------------------------- 1. threshold sensitivity


def frozen_validation_report(out_dir: Path = FROZEN_VAL_DIR) -> dict[str, Any]:
    """Validation threshold curves of the four frozen artifacts, each at its own grouping."""
    gen_cfg, ecfg = load_config(), EvalConfig.load()
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"split": "validation", "data_version": gen_cfg.version,
                              "note": "frozen v2 artifacts re-scored on validation",
                              "false_alert_budget_per_10min": ecfg.budget,
                              "min_precision": ecfg.min_precision,
                              "detectors": {}}  # fmt: skip
    for name, art in frozen_artifacts().items():
        ss = build_scored_set(art.detector, "validation", gen_cfg, ecfg)
        curve = sweep(ss, art.params)
        curve.to_csv(out_dir / f"{name}_threshold_curve.csv", index=False)
        res = ss.evaluate(art.threshold, art.params)
        report["detectors"][name] = {
            "model_version": art.model_version,
            "selected": {**_brief(res.summary), **art.params.as_dict()},
            "summary": res.summary,
            "by_fault_type": [],
        }
    report = _clean(report)
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- 2. ablation


class AblatedFastPath(FastPathDetector):
    """The shipped fast path with signals removed and/or without the speed covariate."""

    def __init__(self, keep: tuple[str, ...], covariate: bool = True) -> None:
        super().__init__()
        self.params.update({"keep": list(keep), "covariate": covariate})

    def _fit_group(self, g: pd.DataFrame) -> np.ndarray:
        out = super()._fit_group(g)
        if not self.params["covariate"]:
            for j, (feat, _sign, _cov) in enumerate(SIGNALS):
                y = g[feat].to_numpy(dtype=float)
                med = float(np.median(y))
                resid = y - med
                from fleet_signal.detectors.stats import robust_scale

                out[j] = (0.0, med, float(np.median(resid)), robust_scale(resid))
        return out

    def zscores(self, feats: pd.DataFrame) -> np.ndarray:
        z = super().zscores(feats)
        for j, (feat, _s, _c) in enumerate(SIGNALS):
            if feat not in self.params["keep"]:
                z[:, j] = np.nan
        return z


ALL_SIGNALS = tuple(s[0] for s in SIGNALS)
ABLATIONS: tuple[tuple[str, str, Any], ...] = (
    ("shipped: rule + fast", "", lambda: [RuleDetector(), FastPathDetector()]),
    ("- fast path (rule only)", "fast", lambda: [RuleDetector()]),
    ("- rule (fast path only)", "rule", lambda: [FastPathDetector()]),
    ("- battery residuals", "batt_res3, batt_res10",
     lambda: [RuleDetector(), AblatedFastPath(("temp_res3", "temp_res10"))]),
    ("- temperature residuals", "temp_res3, temp_res10",
     lambda: [RuleDetector(), AblatedFastPath(("batt_res3", "batt_res10"))]),
    ("- 3-event residuals", "batt_res3, temp_res3",
     lambda: [RuleDetector(), AblatedFastPath(("batt_res10", "temp_res10"))]),
    ("- 10-event residuals", "batt_res10, temp_res10",
     lambda: [RuleDetector(), AblatedFastPath(("batt_res3", "temp_res3"))]),
    ("- speed covariate", "spd_d3, spd_d10 (regression)",
     lambda: [RuleDetector(), AblatedFastPath(ALL_SIGNALS, covariate=False)]),
)  # fmt: skip


def ablation(out_dir: Path = EXCEEDS_DIR) -> pd.DataFrame:
    """VALIDATION only. Each variant is refitted on train and gets its own threshold under the
    same constraints and grouping as the shipped system (open on 1 alert)."""
    gen_cfg, ecfg = load_config(), EvalConfig.load()
    sel = json.loads((VALIDATION_DIR / "v2_selection.json").read_text())
    p = dict(sel["shared_incident_params"])
    params = IncidentParams(**{**p, "open_n": int(sel["chosen"]["open_n"])})
    rows = []
    for label, removed, make in ABLATIONS:
        parts: list[Detector] = make()
        det: Detector = parts[0] if len(parts) == 1 else HybridDetector(parts, "ablation")
        fit_on_train([det], gen_cfg)
        ss = build_scored_set(det, "validation", gen_cfg, ecfg)
        pick = select_threshold(sweep(ss, params), ecfg.budget, ecfg.min_precision)
        s = ss.evaluate(pick["threshold"], params).summary
        rows.append({"variant": label, "removed": removed, "feasible": bool(pick["feasible"]),
                     **_brief(s)})  # fmt: skip
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ablation_validation.csv", index=False)
    return df


# ---------------------------------------------------------------- 3. drift monitor


DRIFT_REFERENCE = EXCEEDS_DIR / "drift_reference.json"


def fit_drift_monitor() -> Any:
    """Reference on TRAIN, thresholds calibrated on VALIDATION normal runs (per asset-run)."""
    from fleet_signal.data.ground_truth import load_runs
    from fleet_signal.features.store import load_features
    from fleet_signal.monitoring.drift import DriftMonitor

    gen_cfg = load_config()
    mon = DriftMonitor().fit(load_features("train", gen_cfg=gen_cfg))
    val = load_features("validation", gen_cfg=gen_cfg)
    runs = load_runs()
    normal = set(runs[(runs["split"] == "validation") & (runs["scenario"] == "normal")]["run_id"])
    windows = [g for (r, _a), g in val.groupby(["run_id", "asset_id"]) if r in normal]
    mon.calibrate(windows)
    EXCEEDS_DIR.mkdir(parents=True, exist_ok=True)
    mon.save(DRIFT_REFERENCE)
    return mon


def drift_table(mon: Any, feats: pd.DataFrame, normal_runs: set[str]) -> pd.DataFrame:
    rows = []
    for (r, a), g in feats.groupby(["run_id", "asset_id"]):
        res = mon.score(g)
        rows.append({"run_id": r, "asset_id": a, "normal_run": r in normal_runs,
                     "score": res["score"], "status": res["status"],
                     "top": ", ".join(f for f, _ in res["top"])})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. shifted fleets


SHIFTS: dict[str, dict[str, Any]] = {
    # name: (description, seed_start, mutation of the data config)
    "hot_climate": {"seed_start": 5000, "what": "ambient 40-50 degC instead of 22-42"},
    "noisy_sensors": {"seed_start": 6000, "what": "temperature and battery sensor noise x2"},
    "aged_batteries": {"seed_start": 7000,
                       "what": "drain under load x1.3, charge rate x0.8 (all asset types)"},
}  # fmt: skip
SHIFT_DATA_DIR = REPO_ROOT / "data_shift"


def shifted_config(name: str, seed_offset: int = 0) -> GenerationConfig:
    base = load_config()
    data, splits = copy.deepcopy(base.data), copy.deepcopy(base.splits)
    if name == "hot_climate":
        data["ambient_c"] = [40.0, 50.0]
    elif name == "noisy_sensors":
        for prof in data["profiles"].values():
            prof["noise"]["temp"] *= 2
            prof["noise"]["battery"] *= 2
    elif name == "aged_batteries":
        for prof in data["profiles"].values():
            prof["drain_load"] *= 1.3
            prof["charge_rate"] *= 0.8
    else:
        raise ValueError(name)
    splits["splits"]["test"]["seed_start"] = SHIFTS[name]["seed_start"] + seed_offset
    return GenerationConfig(data=data, splits=splits)


def shifted_fleets(out_dir: Path = EXCEEDS_DIR, seed_offset: int = 100) -> dict[str, Any]:
    """Post-hoc generalisation test: frozen models, frozen thresholds, new distributions.

    Each shifted fleet has the v2 test composition (15 normal + 45 fault runs, wide ranges,
    unseen variants) on new seeds; only the stated physical parameter differs. The v2 test
    (unshifted) is the reference row. Round 1 used seed offset 0 (results/exceeds/round1_*);
    the reported round uses offset 100 (fresh fleets, see the drift monitor's note).
    """
    from fleet_signal.data.generator import generate_dataset
    from fleet_signal.data.ground_truth import load_runs
    from fleet_signal.data.telemetry import load_telemetry
    from fleet_signal.features.build import FeatureConfig, build_features
    from fleet_signal.features.store import load_features

    ecfg, fcfg = EvalConfig.load(), FeatureConfig.load()
    arts = frozen_artifacts()
    mon = fit_drift_monitor()
    rows, drift_rows = [], []

    def evaluate(label: str, cfg: GenerationConfig, feats: pd.DataFrame, faults: pd.DataFrame,
                 runs: pd.DataFrame) -> None:  # fmt: skip
        ids = sorted(runs.loc[runs["split"] == "test", "run_id"])
        normal = set(runs[(runs["split"] == "test") & (runs["scenario"] == "normal")]["run_id"])
        feats = feats[feats["run_id"].isin(ids)]
        for name in ("rule", "lof", "hybrid_rule_fast"):
            art = arts[name]
            ss = ScoredSet(scored_frame(art.detector, feats), with_detectability(faults, cfg),
                           ids, ecfg, len(cfg.data["fleet"]), seen_variants(cfg))  # fmt: skip
            s = ss.evaluate(art.threshold, art.params).summary
            rows.append({"fleet": label, "detector": name, **_brief(s)})
        d = drift_table(mon, feats, normal).assign(fleet=label)
        drift_rows.append(d)

    base_cfg = load_config()
    evaluate("v2 test (reference)", base_cfg, load_features("test", gen_cfg=base_cfg),
             load_faults(), load_runs())  # fmt: skip
    for name in SHIFTS:
        cfg = shifted_config(name, seed_offset)
        plan = [r for r in build_run_plan(cfg) if r.split == "test"]
        ddir = SHIFT_DATA_DIR / name
        generate_dataset(cfg, ddir, runs=plan)
        tel = load_telemetry(data_dir=ddir)
        evaluate(name, cfg, build_features(tel, fcfg), load_faults(data_dir=ddir),
                 load_runs(data_dir=ddir))  # fmt: skip

    perf = pd.DataFrame(rows)
    drift = pd.concat(drift_rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    perf.to_csv(out_dir / "shifted_fleets_performance.csv", index=False)
    drift.to_csv(out_dir / "shifted_fleets_drift.csv", index=False)
    summary = (
        drift.groupby("fleet")
        .agg(
            asset_runs=("status", "size"),
            ok=("status", lambda s: int((s == "ok").sum())),
            caution=("status", lambda s: int((s == "caution").sum())),
            drift=("status", lambda s: int((s == "drift").sum())),
            median_score=("score", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "shifted_fleets_drift_summary.csv", index=False)
    meta = {"shifts": SHIFTS, "seed_offset": seed_offset,
            "drift_feature_thresholds": mon.feature_threshold}  # fmt: skip
    (out_dir / "shifted_fleets.json").write_text(json.dumps(_clean(meta), indent=2))
    return {"performance": perf, "drift_summary": summary, "drift": drift, **meta}


# ---------------------------------------------------------------- 5. shadow replay


def shadow_replay(shadow: str = "lof", out_dir: Path = EXCEEDS_DIR) -> dict[str, Any]:
    """Replay every v2 test asset-run through the SHIPPED model and a SHADOW model, exactly as
    the service would (registry artifacts, SHA-verified), without acting on the shadow's output.
    Every decision of both is stored for review (shadow_decisions.parquet); the summary says
    where they disagree."""
    from fleet_signal.data.telemetry import load_telemetry
    from fleet_signal.registry import REGISTRY_PATH, registered_artifact
    from fleet_signal.service.replay import replay_asset
    from fleet_signal.service.scorer import Scorer

    reg = read_registry()
    shipped = Scorer()
    shadow_path, _ = registered_artifact(REPO_ROOT / reg["models"][shadow]["artifact"],
                                         REGISTRY_PATH)  # fmt: skip
    shadow_scorer = Scorer(shadow_path)
    if shipped.artifact is None or shadow_scorer.artifact is None:
        raise SystemExit("a model is unavailable")
    tel = load_telemetry(run_ids=split_run_ids(load_config())["test"])
    rows: list[tuple[Any, ...]] = []
    inc_rows: list[dict[str, Any]] = []
    for (run_id, asset_id), ev in tel.groupby(["run_id", "asset_id"]):
        a, _, ia = replay_asset(shipped, ev)
        b, _, ib = replay_asset(shadow_scorer, ev)
        for x, y in zip(a, b, strict=True):
            rows.append((run_id, asset_id, x.seq, x.status, x.decision, x.score,
                         y.status, y.decision, y.score))  # fmt: skip
        inc_rows.append({"run_id": run_id, "asset_id": asset_id,
                         "shipped_incidents": len(ia), "shadow_incidents": len(ib)})  # fmt: skip
    cols = ["run_id", "asset_id", "seq", "shipped_status", "shipped_decision", "shipped_score",
            "shadow_status", "shadow_decision", "shadow_score"]  # fmt: skip
    dec = pd.DataFrame(rows, columns=cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    dec.to_parquet(out_dir / "shadow_decisions.parquet", index=False)
    both = dec[(dec.shipped_status == "ok") & (dec.shadow_status == "ok")]
    alert_a, alert_b = both.shipped_decision == "anomalous", both.shadow_decision == "anomalous"
    summary = {
        "shipped": shipped.artifact.model_version,
        "shadow": shadow_scorer.artifact.model_version,
        "decisions_stored": len(dec),
        "events_scored_by_both": len(both),
        "agree": float((alert_a == alert_b).mean()),
        "shipped_alert_shadow_quiet": int((alert_a & ~alert_b).sum()),
        "shadow_alert_shipped_quiet": int((~alert_a & alert_b).sum()),
        "shipped_incidents": int(sum(r["shipped_incidents"] for r in inc_rows)),
        "shadow_incidents": int(sum(r["shadow_incidents"] for r in inc_rows)),
    }
    (out_dir / "shadow_summary.json").write_text(json.dumps(_clean(summary), indent=2))
    return summary


# ---------------------------------------------------------------- 6. incident prioritisation


def prioritisation_check(out_dir: Path = EXCEEDS_DIR) -> dict[str, Any]:
    """Post-hoc on the v2 test: severity of every incident the shipped model opened (at its
    first open), split by true vs false incident. Checks the formula, does not tune it."""
    import glob

    from fleet_signal.features.store import load_features
    from fleet_signal.incidents.priority import severity

    gen_cfg = load_config()
    reg = read_registry()
    name = reg["serving"]
    art = frozen_artifacts()[name]
    official = json.loads(Path(glob.glob(str(REPO_ROOT / reg["models"][name][
        "official_test_result"]))[0]).read_text())  # fmt: skip
    inc = pd.DataFrame(official["incidents"])
    test = load_features("test", gen_cfg=gen_cfg)
    rows = []
    for (run_id, asset_id), g in test.groupby(["run_id", "asset_id"]):
        mine = inc[(inc["run_id"] == run_id) & (inc["asset_id"] == asset_id)]
        if mine.empty:
            continue
        g = g.sort_values("seq").reset_index(drop=True)
        scores = art.detector.score(g)
        pos = {int(s): i for i, s in enumerate(g["seq"])}
        for _, r in mine.iterrows():
            sev = severity(art.detector, art.threshold, g, scores, pos[int(r["open_seq"])])
            rows.append({"run_id": run_id, "asset_id": asset_id, "open_seq": int(r["open_seq"]),
                         "true_positive": bool(r["true_positive"]), **sev.as_dict()})  # fmt: skip
    df = pd.DataFrame(rows)
    df["families"] = df["families"].map(lambda f: ",".join(f))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "incident_priority_test.csv", index=False)
    tp, fp = df[df.true_positive], df[~df.true_positive]
    # probability a random true incident outranks a random false one (ties count half)
    auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in tp.severity for b in fp.severity]))
    table = df.groupby(["true_positive", "level"]).size().unstack(fill_value=0)
    summary = {"incidents": len(df), "true": len(tp), "false": len(fp),
               "levels": {("true" if bool(k) else "false"): v
                          for k, v in table.to_dict("index").items()},
               "mean_severity_true": float(tp.severity.mean()),
               "mean_severity_false": float(fp.severity.mean()),
               "rank_auc_true_over_false": auc}  # fmt: skip
    (out_dir / "incident_priority_summary.json").write_text(json.dumps(_clean(summary), indent=2))
    return summary

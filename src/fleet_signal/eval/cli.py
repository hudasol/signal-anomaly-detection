"""`signal-eval`: validation-only selection and reporting.

    signal-eval validate        fit on train, select on validation -> results/validation/
    signal-eval validate --detectors rule
    signal-eval select-model    ML grid + ablation on validation
    signal-eval select-v2       v2: fast path + hybrids + grouping, chosen on validation
    signal-eval test            OFFICIAL test, once per frozen model version
    signal-eval plots           re-render all figures from saved outputs
    signal-eval audit           prove the split
    signal-eval show            official comparison from the saved report
    signal-eval demo-threshold --detector lof --threshold 1.5   DEMO ONLY trade-off
    signal-eval fragmentation   post-hoc incidents-per-fault and per-fault precision
    signal-eval envelopes       train envelopes the rule limits were set from

The official test run is a separate command added when the model is frozen.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from fleet_signal.data.config import load_config
from fleet_signal.detectors.base import Detector
from fleet_signal.detectors.lof import LOFDetector
from fleet_signal.detectors.rule import RuleDetector
from fleet_signal.detectors.stats import StatsDetector
from fleet_signal.eval.plots import plot_recall_by_fault, plot_threshold_sensitivity
from fleet_signal.eval.protocol import EvalConfig
from fleet_signal.eval.validate import VALIDATION_DIR, select_on_validation, write_validation_report

DETECTORS: dict[str, type[Detector]] = {
    "rule": RuleDetector,
    "stats": StatsDetector,
    "lof": LOFDetector,
}


def _fmt(v: object) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


DOC_FIGURES = VALIDATION_DIR.parents[1] / "docs" / "figures"
# Example runs shown in the docs: a normal and an overheating validation run, and two
# post-hoc v2 test cases discussed in EVALUATION.md (a fast catch and a late one).
# The v1 test figure (r3015) stays in docs/figures as it was; v1 test data is v1's.
DOC_RUNS: tuple[tuple[str, str], ...] = (
    ("r2001", "drone-01"),
    ("r2011", "drone-01"),
    ("r4009", "drone-01"),
    ("r4000", "quad-01"),
)


def render_doc_figures() -> list[str]:
    """Regenerate every figure under docs/figures/ (no hand-copied images)."""
    import shutil

    from fleet_signal.data.ground_truth import load_faults, load_runs
    from fleet_signal.data.plots import plot_asset_run
    from fleet_signal.data.telemetry import load_telemetry
    from fleet_signal.eval.official import OFFICIAL_DIR

    DOC_FIGURES.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for name in ("test_threshold_sensitivity.png", "test_recall_by_fault_type.png"):
        if (OFFICIAL_DIR / name).exists():
            shutil.copyfile(OFFICIAL_DIR / name, DOC_FIGURES / name)
            out.append(str(DOC_FIGURES / name))
    runs, faults = load_runs(), load_faults()
    tel = load_telemetry(run_ids=[r for r, _ in DOC_RUNS])
    for run_id, asset in DOC_RUNS:
        split = str(runs.loc[runs["run_id"] == run_id, "split"].iloc[0])
        f = faults[(faults["run_id"] == run_id) & (faults["asset_id"] == asset)]
        fault = f.iloc[0] if len(f) else None
        stem = f"{split}_{run_id}_{asset}"
        if fault is not None:
            stem += f"_{fault['fault_type']}_{fault['variant']}"
        out.append(str(plot_asset_run(tel, run_id, asset, DOC_FIGURES / f"{stem}.png", fault,
                                      f" · {split}")))  # fmt: skip
    return out


def render_all_plots() -> list[str]:
    """Every evaluation figure, drawn only from files already saved on disk."""
    from fleet_signal.eval.official import OFFICIAL_DIR

    out = [
        plot_threshold_sensitivity(VALIDATION_DIR, VALIDATION_DIR / "threshold_sensitivity.png"),
        plot_recall_by_fault(VALIDATION_DIR, VALIDATION_DIR / "recall_by_fault_type.png"),
    ]
    posthoc = OFFICIAL_DIR / "posthoc"
    if (posthoc / "test_report_for_plots.json").exists():
        out.append(plot_threshold_sensitivity(
            posthoc, OFFICIAL_DIR / "test_threshold_sensitivity.png",
            report_name="test_report_for_plots.json",
            title="Test: post-hoc threshold sweep (markers = frozen operating points)",
        ))  # fmt: skip
        out.append(plot_recall_by_fault(
            posthoc, OFFICIAL_DIR / "test_recall_by_fault_type.png",
            report_name="test_report_for_plots.json",
            title="Recall by fault type at the frozen thresholds (official test)",
        ))  # fmt: skip
    return [str(p) for p in out]


def _exceeds(only: str) -> None:
    import json as _json

    from fleet_signal.eval import exceeds as ex

    todo = ["thresholds", "ablation", "shifted", "shadow", "priority"] if only == "all" else [only]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        if "thresholds" in todo:
            rep = ex.frozen_validation_report()
            print(
                "validation, frozen artifacts:",
                {n: round(d["selected"]["recall"], 3) for n, d in rep["detectors"].items()},
            )
        if "ablation" in todo:
            print(ex.ablation().round(3).to_string(index=False))
        if "shifted" in todo:
            r = ex.shifted_fleets()
            print(r["performance"].round(3).to_string(index=False))
            print(r["drift_summary"].round(3).to_string(index=False))
        if "shadow" in todo:
            print(_json.dumps(ex.shadow_replay(), indent=1))
        if "priority" in todo:
            print(_json.dumps(ex.prioritisation_check(), indent=1))
    for path in render_exceeds_figures():
        print(path)


def render_exceeds_figures() -> list[str]:
    from fleet_signal.eval.exceeds import EXCEEDS_DIR, FROZEN_VAL_DIR
    from fleet_signal.eval.plots import plot_shifted_fleets

    out = []
    if (FROZEN_VAL_DIR / "validation_report.json").exists():
        out.append(plot_threshold_sensitivity(
            FROZEN_VAL_DIR, DOC_FIGURES / "validation_v2_threshold_sensitivity.png",
            title="Validation: threshold sweep of the frozen v2 models (markers = frozen points)",
        ))  # fmt: skip
    if (EXCEEDS_DIR / "shifted_fleets_performance.csv").exists():
        out.append(
            plot_shifted_fleets(
                EXCEEDS_DIR / "shifted_fleets_performance.csv",
                DOC_FIGURES / "generalisation_shifted_fleets.png",
            )
        )
    return [str(p) for p in out]


def _official() -> None:
    from fleet_signal.eval.official_test import run_official_test

    out = run_official_test()
    report = out["report"]
    render_all_plots()
    cols = ["precision", "recall", "f1", "fp_per_10min", "n_false_incidents",
            "progressive_latency_median"]  # fmt: skip
    print("OFFICIAL TEST RESULT (frozen thresholds, run once)")
    print("detector".ljust(10) + "".join(c[:14].rjust(15) for c in cols) + "   meets bar")
    for name, det in report["detectors"].items():
        bar = ",".join(k for k, ok in det["meets_bar"].items() if not ok) or "all"
        failed = "" if bar == "all" else "fails: "
        print(name.ljust(10) + "".join(_fmt(det["summary"][c]).rjust(15) for c in cols)
              + f"   {failed}{bar}")  # fmt: skip
    if report.get("decision"):
        print(f"\nship: {report['decision']['ship']}  ({report['decision']['reason']})")
    print("\n" + "\n".join(out["written"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="signal-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="select incident params and thresholds on validation")
    val.add_argument("--detectors", default="rule,stats,lof")
    sub.add_parser("select-model", help="ML grid (Isolation Forest, LOF) + ablation on validation")
    sub.add_parser("select-v2", help="v2 selection on validation: fast path, hybrids, grouping")
    sub.add_parser("test", help="OFFICIAL run-once test evaluation of the frozen models")
    sub.add_parser("plots", help="re-render every evaluation plot from saved outputs")
    sub.add_parser("audit", help="prove the split: which runs fitted / selected / tested")
    sub.add_parser("show", help="print the official comparison from the saved report")
    sub.add_parser("fragmentation", help="post-hoc: incidents per fault, per-fault precision")
    sub.add_parser("envelopes", help="export the train envelopes the rule limits came from")
    sub.add_parser("replay-check", help="post-hoc: service replay of every test run == official")
    ex = sub.add_parser("exceeds", help="exceeds-the-bar analyses (validation + post-hoc)")
    ex.add_argument(
        "--only", default="all", help="all | thresholds | ablation | shifted | shadow | priority"
    )
    demo = sub.add_parser("demo-threshold", help="DEMO ONLY: a frozen model at another threshold")
    demo.add_argument("--detector", required=True, help="a detector in models/registry.json")
    demo.add_argument("--threshold", required=True, type=float)
    args = parser.parse_args(argv)

    if args.command == "fragmentation":
        from fleet_signal.eval.demo import fragmentation

        frag = fragmentation()
        print("POST-HOC (from saved official files)")
        keys = ["incident_precision", "per_fault_precision", "incidents_per_detected_fault",
                "max_incidents_on_one_fault", "faults_with_more_than_one_incident"]  # fmt: skip
        for split in ("test", "validation"):
            print(f"\n{split}")
            for name, row in frag[split].items():
                print(f"  {name:<6}" + "  ".join(f"{k}={row[k]:.3g}" for k in keys))
        return
    if args.command == "envelopes":
        from fleet_signal.eval.demo import rule_envelopes

        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(rule_envelopes().round(3).to_string(index=False))
        return
    if args.command in ("audit", "show"):
        from fleet_signal.eval import demo as demo_mod

        print("\n".join(demo_mod.audit() if args.command == "audit" else demo_mod.show()))
        return
    if args.command == "demo-threshold":
        from fleet_signal.eval.demo import threshold_demo

        out = threshold_demo(args.detector, args.threshold)
        print(f"DEMO ONLY ({out['model_version']}); official result untouched")
        for label in ("frozen", "demo"):
            thr = out[f"{label}_threshold"]
            vals = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in out[label].items())  # fmt: skip
            print(f"  {label:<6} threshold {thr:8.3f}  {vals}")
        return

    if args.command == "plots":
        for path in [*render_all_plots(), *render_doc_figures(), *render_exceeds_figures()]:
            print(path)
        return

    if args.command == "test":
        _official()
        return

    if args.command == "exceeds":
        _exceeds(args.only)
        return

    if args.command == "replay-check":
        from fleet_signal.eval.demo import replay_check

        out = replay_check()
        n_bad = len(out["mismatches"])
        print(f"{out['model_version']}: {out['asset_runs']} test asset-runs replayed through "
              f"the service; mismatches with the official incidents: {n_bad}")  # fmt: skip
        return

    if args.command == "select-v2":
        from fleet_signal.eval.select_v2 import run as run_v2

        out = run_v2()
        cols = ["candidate", "open_n", "feasible", "meets_bar", "threshold", "precision",
                "recall", "fp_per_10min", "progressive_latency_median",
                "expected_latency_median", "n_expected", "n_expected_detected"]  # fmt: skip
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(out["lof_grid"].round(3).to_string(index=False))
            print()
            print(out["grid"].reindex(columns=cols).round(3).to_string(index=False))
        print(f"\nchosen: {out['chosen']['name']} (open after {out['chosen']['open_n']}); "
              f"best baseline on validation: {out['best_baseline']}")  # fmt: skip
        return

    if args.command == "select-model":
        from fleet_signal.eval.model_selection import run

        out = run()
        cols = [
            "model",
            "input_space",
            "max_samples",
            "max_features",
            "n_neighbors",
            "max_train",
            "feasible",
            "precision",
            "recall",
            "fp_per_10min",
            "latency_median",
        ]
        grid = out["grid"].reindex(columns=cols)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(grid.round(3).to_string(index=False))
            print("\nchosen:", out["chosen"])
            if len(out["ablation"]):
                print("\nablation (one feature group removed):")
                print(out["ablation"].round(3).to_string(index=False))
        return

    gen_cfg, ecfg = load_config(), EvalConfig.load()
    names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    started = time.perf_counter()
    sel = select_on_validation([DETECTORS[n]() for n in names], gen_cfg, ecfg)
    report = write_validation_report(sel, gen_cfg, ecfg, VALIDATION_DIR)
    render_all_plots()

    print(f"validation selection in {time.perf_counter() - started:.1f}s")
    print(f"incident params (shared): {report['incident_params']}")
    cols = ["threshold", "feasible", "precision", "recall", "f1", "fp_per_10min",
            "n_false_incidents", "progressive_latency_median"]  # fmt: skip
    print("\n" + "detector".ljust(10) + "".join(c[:14].rjust(15) for c in cols))
    for name, det in report["detectors"].items():
        row = {**det["summary"], "feasible": det["selected"]["feasible"]}
        print(name.ljust(10) + "".join(_fmt(row[c]).rjust(15) for c in cols))
    print(f"\nwritten to {VALIDATION_DIR}")


if __name__ == "__main__":
    main()

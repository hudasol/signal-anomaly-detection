"""`signal-eval`: validation-only selection and reporting.

    signal-eval validate        fit on train, select on validation -> results/validation/
    signal-eval validate --detectors rule
    signal-eval select-model    ML grid + ablation on validation
    signal-eval test            OFFICIAL test, once per frozen model version
    signal-eval plots           re-render all figures from saved outputs

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
    sub.add_parser("test", help="OFFICIAL run-once test evaluation of the frozen models")
    sub.add_parser("plots", help="re-render every evaluation plot from saved outputs")
    args = parser.parse_args(argv)

    if args.command == "plots":
        for path in render_all_plots():
            print(path)
        return

    if args.command == "test":
        _official()
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

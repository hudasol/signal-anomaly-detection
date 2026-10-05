"""`signal-eval`: validation-only selection and reporting.

    signal-eval validate        fit on train, select on validation -> results/validation/
    signal-eval validate --detectors rule
    signal-eval select-model    ML grid + ablation on validation

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="signal-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    val = sub.add_parser("validate", help="select incident params and thresholds on validation")
    val.add_argument("--detectors", default="rule,stats,lof")
    sub.add_parser("select-model", help="ML grid (Isolation Forest, LOF) + ablation on validation")
    args = parser.parse_args(argv)

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
    plot_threshold_sensitivity(VALIDATION_DIR, VALIDATION_DIR / "threshold_sensitivity.png")
    plot_recall_by_fault(VALIDATION_DIR, VALIDATION_DIR / "recall_by_fault_type.png")

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

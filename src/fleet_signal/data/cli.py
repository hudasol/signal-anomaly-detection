"""`signal-data`: generate, summarise and plot the dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from fleet_signal.data.config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_DATA_DIR,
    DEFAULT_SPLITS_CONFIG,
    REPO_ROOT,
    load_config,
)
from fleet_signal.data.generator import generate_dataset
from fleet_signal.data.ground_truth import load_episodes, load_faults, load_runs
from fleet_signal.data.telemetry import load_telemetry, resolve_version_dir

DEFAULT_PLOT_DIR = REPO_ROOT / "results" / "data_sanity"


def _cmd_generate(args: argparse.Namespace) -> None:
    cfg = load_config(Path(args.config), Path(args.splits))
    started = time.perf_counter()
    path = generate_dataset(cfg, Path(args.out))
    print(f"data version {cfg.version} written to {path} in {time.perf_counter() - started:.1f}s")


def _cmd_summary(args: argparse.Namespace) -> None:
    version_dir = resolve_version_dir(args.version, Path(args.data_dir))
    manifest = json.loads((version_dir / "manifest.json").read_text())
    runs = load_runs(args.version, Path(args.data_dir))
    faults = load_faults(args.version, Path(args.data_dir)).merge(runs, on="run_id")
    episodes = load_episodes(args.version, Path(args.data_dir)).merge(runs, on="run_id")
    print(f"data version: {manifest['data_version']}  events: {manifest['n_events']:,}")
    print("\nseed ranges per split:")
    for split, (lo, hi) in manifest["seeds_per_split"].items():
        print(f"  {split:<11} {lo}-{hi}")
    print("\nruns per split and scenario:")
    print(pd.crosstab(runs["scenario"], runs["split"]).to_string())
    print("\nfault variants per split:")
    print(pd.crosstab([faults["fault_type"], faults["variant"]], faults["split"]).to_string())
    print("\nfault windows (seconds):")
    faults["len_s"] = faults["fault_end_seq"] - faults["fault_start_seq"]
    print(faults.groupby("fault_type")["len_s"].describe()[["min", "50%", "max"]].to_string())
    print("\ndifficult-normal episodes per split:")
    print(pd.crosstab(episodes["episode"], episodes["split"]).to_string())


def _cmd_plot(args: argparse.Namespace) -> None:
    from fleet_signal.data.plots import plot_asset_run

    data_dir = Path(args.data_dir)
    runs = load_runs(args.version, data_dir)
    faults = load_faults(args.version, data_dir)
    out_dir = Path(args.out)

    targets: list[tuple[str, str | None]] = []
    if args.run:
        targets.append((args.run, args.asset))
    else:
        # Gallery: first validation and first test run of each fault variant, plus one normal run.
        merged = faults.merge(runs, on="run_id")
        for _, group in merged.groupby(["split", "fault_type", "variant"]):
            targets.append((group.iloc[0]["run_id"], None))
        normal = runs[(runs["split"] == "validation") & (runs["scenario"] == "normal")]
        targets.append((normal.iloc[0]["run_id"], None))

    tel = load_telemetry(args.version, data_dir, run_ids=sorted({r for r, _ in targets}))
    for run_id, asset in targets:
        fault_rows = faults[faults["run_id"] == run_id]
        fault = fault_rows.iloc[0] if len(fault_rows) else None
        assets = [asset] if asset else ([fault["asset_id"]] if fault is not None else ["drone-01"])
        for asset_id in assets:
            on_asset = fault if fault is not None and fault["asset_id"] == asset_id else None
            split = runs.loc[runs["run_id"] == run_id, "split"].iloc[0]
            name = f"{split}_{run_id}_{asset_id}"
            if on_asset is not None:
                name += f"_{on_asset['fault_type']}_{on_asset['variant']}"
            path = plot_asset_run(
                tel, run_id, asset_id, out_dir / f"{name}.png", on_asset, f" · {split}"
            )
            print(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="signal-data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate every run in configs/splits.yaml")
    gen.add_argument("--config", default=str(DEFAULT_DATA_CONFIG))
    gen.add_argument("--splits", default=str(DEFAULT_SPLITS_CONFIG))
    gen.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    gen.set_defaults(func=_cmd_generate)

    summ = sub.add_parser("summary", help="print split, scenario and episode counts")
    summ.add_argument("--version", default=None)
    summ.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    summ.set_defaults(func=_cmd_summary)

    plot = sub.add_parser("plot", help="sanity plots (one run, or a gallery of every variant)")
    plot.add_argument("--run", default=None)
    plot.add_argument("--asset", default=None)
    plot.add_argument("--version", default=None)
    plot.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    plot.add_argument("--out", default=str(DEFAULT_PLOT_DIR))
    plot.set_defaults(func=_cmd_plot)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

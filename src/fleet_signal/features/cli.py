"""`signal-features`: build and inspect the cached feature table."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from fleet_signal.features.build import FEATURE_GROUPS, FeatureConfig, scorable
from fleet_signal.features.store import feature_cache_path, load_features


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="signal-features", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build (or reuse) the feature cache for the current data version")
    desc = sub.add_parser("describe", help="train-split feature distributions per asset type")
    desc.add_argument("--group", choices=sorted(FEATURE_GROUPS), default="trend")
    args = parser.parse_args(argv)

    fcfg = FeatureConfig.load()
    if args.command == "build":
        started = time.perf_counter()
        feats = load_features(fcfg=fcfg)
        print(
            f"{len(feats):,} rows, {sum(len(v) for v in FEATURE_GROUPS.values())} features, "
            f"{scorable(feats).mean():.1%} scorable, {time.perf_counter() - started:.1f}s"
        )
        print(feature_cache_path(fcfg))
    else:
        train = load_features("train", fcfg)
        train = train[scorable(train)]
        cols = list(FEATURE_GROUPS[args.group])
        with pd.option_context("display.width", 200, "display.max_columns", 50):
            print(train.groupby("asset_type")[cols].describe(percentiles=[0.5, 0.999]).T)


if __name__ == "__main__":
    main()

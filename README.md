# Signal

Evaluated anomaly detection and early warning for fleet telemetry from a simulated inspection fleet (drone, rover, quadruped). Signal compares a rule baseline, a statistical baseline and an ML detector under leakage-safe splits and a frozen threshold, and serves the result as incidents with evidence.

Task 02 of the systems/ML mentorship track. It is a separate component next to [blackbox-telemetry](https://github.com/hudasol/blackbox-telemetry); Blackbox is not modified. Signal's events use the Blackbox telemetry contract (same field names, units and modes), so Blackbox telemetry can be replayed through it.

**Status:** in progress. Design: [`docs/PLAN.md`](docs/PLAN.md). Build history and plan changes: [`docs/PROCESS_LOG.md`](docs/PROCESS_LOG.md).

## Setup

Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Data

One command generates every run listed in `configs/splits.yaml`, using `configs/data.yaml`:

```bash
signal-data generate        # ~20 s; writes data/<version>/ and data/CURRENT
signal-data summary         # seed ranges, runs per split/scenario, variants, episodes
signal-data plot            # sanity plots of every fault variant -> results/data_sanity/
signal-data plot --run r2011 --asset drone-01   # one run, one asset
```

Generation is deterministic: the same config and code produce byte-identical files. The data version (e.g. `v1.0.0-9e903f9008`) is a hash of both configs and the generator source, so any change produces a new version.

```
data/<version>/
  telemetry.parquet        the only file feature/model code reads
  manifest.json            version, git SHA, seed ranges, counts, full config
  ground_truth/            evaluation and error analysis only
    runs.parquet           run_id -> seed, split, scenario
    faults.parquet         fault asset, type, variant, window, parameters
    episodes.parquet       difficult-normal episodes (charging, returning, hard manoeuvre, noisy link)
```

| Split | Seeds | Runs |
|---|---|---|
| train | 1000–1039 | 40 normal |
| validation | 2000–2039 | 10 normal + 30 fault (6 per type) |
| test | 3000–3059 | 15 normal + 45 fault (9 per type), wider ranges, unseen variants |

Each run is 20 minutes of 1 Hz telemetry for all three assets (~503k events in total).

## Features

```bash
signal-features build              # build/reuse the feature cache (~6 s)
signal-features describe --group trend   # train-split distributions per asset type
```

35 causal features per event (trailing windows of 10/30/120 events per asset and run): levels, slopes, volatility, events-since-change, speed/position consistency, mode context. An asset needs 120 events of history before it can be scored.

## Validation (baselines)

```bash
signal-eval validate               # fit on train, select on validation -> results/validation/
```

Fits each detector on train-normal features, chooses the shared incident parameters and each detector's threshold on validation (false alerts ≤ 1.5 per 10 min and precision ≥ 0.80, then highest recall), and writes curves, per-fault tables, incidents and plots. The test split is never loaded by this command.

Commands for the ML model, the official test run, inference and replay are added as each part lands.

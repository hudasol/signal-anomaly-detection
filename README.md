# Signal

Evaluated anomaly detection and early warning for fleet telemetry from a simulated inspection fleet (drone, rover, quadruped). Signal compares a rule baseline, a statistical baseline and an ML detector under leakage-safe splits and a frozen threshold, and serves the result as incidents with evidence.

Task 02 of the systems/ML mentorship track. It is a separate component next to [blackbox-telemetry](https://github.com/hudasol/blackbox-telemetry); Blackbox is not modified.

**Status:** in progress. See [`docs/PLAN.md`](docs/PLAN.md) for the design and [`docs/PROCESS_LOG.md`](docs/PROCESS_LOG.md) for how it is being built.

## Setup

Python 3.12+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Commands for data generation, training, evaluation, inference and replay are added as each part lands.

# Signal

Anomaly detection and early warning for fleet telemetry from a simulated inspection fleet (drone, rover, quadruped). Signal consumes telemetry, scores each asset's recent behaviour, groups alerts into incidents, and tells the operator which signals drove each decision. It compares a transparent rule baseline, a robust-statistics baseline and an ML detector under leakage-safe splits, with thresholds frozen on validation and a test set evaluated once.

Task 02 of the mentorship track, built next to [blackbox-telemetry](https://github.com/hudasol/blackbox-telemetry) (not modified). Signal's events use the Blackbox telemetry contract, and the API accepts Blackbox events directly.

## Result (official test, run once)

| | Precision | Recall | False incidents / 10 min | Median progressive latency |
|---|---|---|---|---|
| **Rule baseline (shipped)** | 0.81 | 0.84 | 0.12 | 71 events |
| Robust z baseline | 0.79 | 0.38 | 0.05 | 534.5 events |
| LOF (ML) | 0.97 | 0.89 | 0.02 | 16 events |

**The rule baseline ships**, by the decision rule declared before the test: LOF's recall and latency gains were not statistically established on 45 test faults. LOF is, however, significantly better on precision and false alerts, which that rule did not consider. The recommendation is to run LOF in shadow mode next to the rule (see [EVALUATION.md §5](docs/EVALUATION.md#5-ship-decision)). No detector meets the brief's 3-event latency bar for slow progressive faults ([§8](docs/EVALUATION.md#8-error-analysis), pattern 5).

## Docs

| | |
|---|---|
| [PLAN.md](docs/PLAN.md) | written before any code; changes are recorded in the process log, not edited in |
| [DATA_CARD.md](docs/DATA_CARD.md) | how telemetry is generated, labels, splits, distributions, leakage controls, limitations |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | shipped model and ML candidate, features, thresholds, intended use and non-use, failure cases |
| [EVALUATION.md](docs/EVALUATION.md) | protocol, test results with CIs, per-fault breakdown, comparison, threshold choice, ablation, error analysis |
| [PROCESS_LOG.md](docs/PROCESS_LOG.md) | decisions, plan changes, bugs found, section by section |
| [RETROSPECTIVE.md](docs/RETROSPECTIVE.md) | what went wrong and what I'd change |
| [DEMO.md](docs/DEMO.md) | the acceptance demo, command by command |

## Setup

Python 3.12+. No Docker or external services are needed for the pipeline.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Reproduce everything

Each step is deterministic. Total time is a few minutes.

```bash
signal-data generate        # 1. dataset: 140 runs, ~503k events, ground truth stored separately   (~20 s)
signal-features build       # 2. causal window features, cached per data version                   (~6 s)
signal-eval validate        # 3. fit on train; incident params + thresholds on VALIDATION          (~60 s)
signal-eval select-model    # 4. ML grid (Isolation Forest, LOF) + feature-group ablation          (~5 min)
signal-train                # 5. freeze rule / stats / lof artifacts + models/registry.json        (~50 s)
signal-eval test            # 6. OFFICIAL test: runs once per model version, then refuses
signal-eval plots           #    re-render every figure from saved outputs
```

The committed `results/official/` already holds the official result, so step 6 will refuse to run in this repo, by design. To reproduce it, run it in a fresh clone after deleting `results/official/`, or compare against the committed files. Rebuilt artifacts get the same model versions and thresholds; the registry records the SHA-256 of the artifact that was evaluated.

## Inference

**Replay a stored run** through the shipped (or any frozen) detector, as the service would:

```bash
signal-replay --run r3006 --show-truth                     # shipped model, all assets; ground truth shown afterwards
signal-replay --run r3006 --detector lof --asset drone-01
signal-replay --run r3006 --detector lof --threshold 1.5   # DEMO ONLY threshold override
signal-replay --run r3006 --strict                         # window-by-window through the service code path
```

Every decision is logged to `results/replay/<run>_<model_version>.jsonl`.

**HTTP service:**

```bash
uvicorn fleet_signal.service.app:app --port 8000           # serves the registry's shipped model
SIGNAL_ARTIFACT=models/lof/<version>/model.joblib uvicorn fleet_signal.service.app:app

curl localhost:8000/health
curl localhost:8000/model
curl -X POST localhost:8000/score -H 'content-type: application/json' \
     -d '{"events": [ ...one asset, oldest first, up to 650 events... ]}'
```

Responses carry `status` (`ok`, `insufficient_data`, `degraded`, `unavailable`), `score`, `threshold`, `decision`, `model_version` and `evidence`. Without a usable model, `/score` returns HTTP 503 `unavailable`; it never answers `normal` without a score.

## Tests

```bash
pytest                       # 126 tests, ~13 s
ruff check . && ruff format --check .
```

The tests pin down preprocessing (`test_features.py`), split logic (`test_splits.py`), generator and label isolation (`test_generator.py`), threshold logic and metrics (`test_eval.py`), incident grouping (`test_incidents.py`), detectors (`test_detectors.py`), artifacts and the ship rule (`test_registry_and_decision.py`), and the inference contract with end-to-end replay (`test_service.py`). CI runs lint and tests on every push: [GitHub Actions](https://github.com/hudasol/signal-anomaly-detection/actions).

## Exceeds the bar

| Item | What | Risk it addresses |
|---|---|---|
| Threshold sensitivity | validation and post-hoc test curves, operating points marked ([figure](docs/figures/test_threshold_sensitivity.png)) | an arbitrary or cherry-picked operating point |
| Model versioning | `models/registry.json` ties artifact SHA-256, data version, feature schema, threshold, validation report and official result | not being able to say which model and data produced a number |
| Ablation | LOF refit without each feature group ([EVALUATION §7](docs/EVALUATION.md#7-ablation-validation-exceeds-the-bar-item)) | assuming more features are better |

## Layout

```
configs/         data.yaml  splits.yaml  features.yaml  detectors.yaml  eval.yaml
src/fleet_signal/
  data/          generator, faults, splits, telemetry loader, ground-truth loader (eval only)
  features/      build_features(), cache, split guard
  detectors/     rule, stats (robust z), lof, iforest
  incidents/     batch grouping + streaming tracker
  eval/          protocol, threshold selection, bootstrap, ship decision, official test, plots
  service/       scorer (failure states), FastAPI app, replay worker
  registry.py    artifacts + registry      train.py   signal-train
results/validation/   selection outputs (committed)
results/official/     the run-once test result (committed); posthoc/ = analysis after the fact
models/registry.json  committed; artifacts are rebuilt by signal-train
```

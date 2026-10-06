# Signal

Anomaly detection and early warning for fleet telemetry from a simulated inspection fleet (drone, rover, quadruped). Signal consumes telemetry, scores each asset's recent behaviour, groups alerts into incidents, and tells the operator which signals drove each decision. It compares a transparent rule baseline, a robust-statistics baseline and an ML detector under leakage-safe splits, with thresholds frozen on validation and a test set evaluated once.

Task 02 of the mentorship track, built next to [blackbox-telemetry](https://github.com/hudasol/blackbox-telemetry) (not modified). Signal's events use the Blackbox telemetry contract, and the API accepts Blackbox events directly.

## Result (official test, run once)

> **The shipped system does not meet the brief's bar.** The rule baseline ships by the pre-declared decision rule, but it misses recall by one fault (0.844 vs 0.85) and latency by a wide margin (71 events vs 3). LOF meets every bar item except latency. No detector meets the latency bar.

| | Precision | Recall | False incidents / 10 min | Median progressive latency |
|---|---|---|---|---|
| **Rule baseline (shipped)** | 0.81 (per fault 0.76) | 0.84 ❌ | 0.12 | 71 events ❌ |
| Robust z baseline | 0.79 (per fault 0.77) | 0.38 ❌ | 0.05 | 534.5 events ❌ |
| LOF (ML) | 0.97 (per fault 0.95) | 0.89 | 0.02 | 16 events ❌ |

Precision in brackets counts each fault once; incident precision is flattered when one fault is split into several incidents (LOF did this for 17 of its 40 detected faults; [EVALUATION §2](docs/EVALUATION.md#incident-fragmentation-and-per-fault-precision-post-hoc-from-the-saved-files)).

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

The results were produced with **Python 3.12.3**; `requirements.lock` pins every package by version and hash (numpy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1, pyarrow 25.0.1, fastapi 0.142.2, …). `requirements.runtime.lock` is the runtime-only subset the Docker image installs.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.lock && pip install --no-deps -e .
```

Or with Docker (base image pinned by digest; runs as a non-root user; has a health check):

```bash
docker build -t signal .                           # runtime image: the API only
docker run --rm -p 8000:8000 signal                # serves the shipped model
docker build --target test -t signal-test .        # runtime + test tools + tests
docker run --rm --network none signal-test         # the test suite, offline
```

## Reproduce everything

Each step is deterministic. Total time is a few minutes.

```bash
signal-data generate        # 1. dataset: 140 runs, ~503k events, ground truth stored separately   (~20 s)
signal-features build       # 2. causal window features, cached per data version                   (~6 s)
signal-eval validate        # 3. fit on train; incident params + thresholds on VALIDATION          (~60 s)
signal-eval select-model    # 4. ML grid (Isolation Forest, LOF) + feature-group ablation          (~5 min)
signal-train                # 5. freeze artifacts; KEEPS the committed evaluated ones (SHA-verified) (~50 s)
signal-eval test            # 6. OFFICIAL test: runs once per model version, then refuses
signal-eval plots           #    re-render every figure from saved outputs
```

The committed `results/official/` already holds the official result, so step 6 refuses to run in this repo, by design. To reproduce it, run it in a scratch clone after deleting `results/official/` and compare with the committed files; this was done and reproduced every number and decision exactly. The **exact evaluated artifacts are committed** under `models/` (their SHA-256 is in `models/registry.json` and in each official result file), and `signal-train` will not overwrite them.

Post-hoc analysis and audit helpers (read-only):

```bash
signal-eval show            # the official comparison, from the saved report
signal-eval audit           # proves which runs each model was fitted / selected / tested on
signal-eval fragmentation   # incidents per detected fault, per-fault precision
signal-eval envelopes       # the train-normal envelopes the rule limits came from
signal-eval demo-threshold --detector lof --threshold 1.5   # DEMO ONLY, writes results/demo/
```

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
SIGNAL_ARTIFACT=models/lof/<version>/model.joblib uvicorn fleet_signal.service.app:app   # must be a registered artifact

curl localhost:8000/livez        # process up
curl localhost:8000/readyz       # a verified model is loaded (503 + reason if not); /health is an alias
curl localhost:8000/model        # threshold, incident params, versions, code provenance
curl -X POST localhost:8000/score -H 'content-type: application/json' \
     -d '{"events": [ ...one asset, oldest first: 650 events, or all since the run started... ]}'
```

Responses carry `status` (`ok`, `insufficient_data`, `degraded`, `unavailable`), `score`, `threshold`, `decision`, `model_version` and `evidence`. The service answers `normal` only for a **validated, full-context window scored by a verified model**:

| Problem | Answer |
|---|---|
| malformed request: wrong types, unknown `mode` or `asset_type` (case-sensitive), several assets, `seq` not strictly increasing or negative, more than 1,000 events | HTTP 422 `invalid_request`, not scored |
| request body over 2 MB | HTTP 413, rejected before parsing |
| impossible reading (battery or link outside 0–100, speed over 100 m/s, …), a gap, timestamps not 1 Hz | `degraded`, no decision |
| fewer than 120 events, or a mid-run window shorter than 650 (look-back features would be cut off) | `insufficient_data`, no decision |
| no model, or the model file is not in the registry or fails its SHA-256 check (checked *before* loading) | HTTP 503 `unavailable` |

`decision` is **per event**: whether this window's last event crossed the threshold. The evaluated false-alarm rate (0.12 false incidents per 10 min for the rule) is for **incidents**, which are built from the stream of decisions (open after 2 alerts, close after 5 quiet events) by the incident tracker, as `signal-replay` does. Every `/score` call writes one JSON audit line (asset, seq, status, score, threshold, decision, model version, a hash of the events); set `SIGNAL_AUDIT_LOG=<file>` to keep them in a file. Cost and scaling notes are in [MODEL_CARD.md](docs/MODEL_CARD.md#scaling-and-protocol).

## Tests

```bash
pytest                       # 173 tests, ~20 s
ruff check . && ruff format --check .
mypy                         # type check (clean)
```

The tests pin down preprocessing (`test_features.py`), split logic (`test_splits.py`), generator and label isolation (`test_generator.py`), threshold logic and metrics (`test_eval.py`), incident grouping (`test_incidents.py`), detectors (`test_detectors.py`), artifacts and the ship rule (`test_registry_and_decision.py`), the inference contract with replay (`test_service.py`), input validation, artifact verification and the other service hardening (`test_service_hardening.py`), and the full pipeline with fitted detectors (`test_end_to_end.py`). CI installs the hash-pinned environment on Python 3.12.3 and runs ruff, mypy, pytest and a dependency audit, then builds both Docker images, runs the tests inside one with no network and checks the other serves the model as a non-root user, on every push: [GitHub Actions](https://github.com/hudasol/signal-anomaly-detection/actions) ([verified run on 0b690d3](https://github.com/hudasol/signal-anomaly-detection/actions/runs/37394108168): pinned install, ruff, mypy, 133 tests).

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
  service/       validation, scorer (failure states), FastAPI app, audit log, replay worker
  registry.py    artifacts + registry      train.py   signal-train
results/validation/   selection outputs (committed)
results/official/     the run-once test result (committed); posthoc/ = analysis after the fact
models/               the exact evaluated artifacts + registry.json (committed, SHA-verified)
requirements.lock     exact environment (hashes)   requirements.runtime.lock   runtime subset
Dockerfile            runtime image (default) + test image (--target test)
```

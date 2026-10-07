# Signal

Anomaly detection and early warning for fleet telemetry from a simulated inspection fleet (drone, rover, quadruped). Signal consumes telemetry, scores each asset's recent behaviour, groups alerts into incidents, and tells the operator which signals drove each decision. It compares a transparent rule baseline, a robust-statistics baseline and an ML detector under leakage-safe splits, with thresholds frozen on validation and a test set evaluated once.

Task 02 of the mentorship track, built next to [blackbox-telemetry](https://github.com/hudasol/blackbox-telemetry) (not modified). Signal's events use the Blackbox telemetry contract, and the API accepts Blackbox events directly.

## Result (official v2 test, run once)

> **v2 meets 7 of the 8 acceptance criteria on a fresh, pre-registered test set. The latency criterion is still not met by the project's own original definition.** The shipped system is the **rule baseline plus a fast-path residual detector**, with incidents opening on the first alert. On test seeds 4000–4059 (generated only after the v2 models were frozen and pushed) it reached precision 0.85, recall 0.98 and 0.12 false incidents per 10 min. Latency: **1 event** median on the faults that are physically catchable within 3 events (abrupt battery drains, 5 of 5), but **11 events** over all progressive faults. My original plan (PLAN §7.1) declared every fault type "expected to catch" and said narrowing that set later would count as a failure, not a redefinition. The narrower, physics-based set was declared for v2 after v1 had failed (though before v2's test existed), so by the original rule latency is a miss. Slow overheating and link decline cannot be seen in 3 seconds at this sensor noise.

| v2 test | Precision | Recall | False incidents / 10 min | Expected-to-catch latency | All-progressive latency |
|---|---|---|---|---|---|
| **Rule + fast path (shipped)** | 0.85 (per fault 0.79) | **0.98** | 0.12 | **1 event** (5 of 5) | 11 ❌ (bar: 3) |
| Rule baseline | 0.83 | 0.93 | 0.11 | 188 (4 of 5) | 76 |
| LOF (ML) | 0.92 | 0.87 | 0.06 | 7 (4 of 5) | 25 |
| Robust z baseline | 0.63 ❌ | 0.31 ❌ | 0.09 | missed 4 of 5 | 403 |

**Why it ships:** under the ship rule declared before the v2 test, it is not worse than the rule on recall (paired CI +0.000 to +0.119) and significantly faster. Most of that speed comes from battery drain (188 → 2 events) and overheating (64 → 11). The paired median gain is only 1 event, and the rule alone also passes the precision / recall / false-alert bar on this test set (EVALUATION §2, §4).

**v1** (tag `iteration-1`) shipped the rule baseline alone and missed two criteria on its own test set (recall 0.844, latency 71). That result is unchanged in [docs/v1/](docs/v1/EVALUATION_v1.md). v2 was built from v1's test-set error analysis, so v1's test set is spent and v2 is judged only on new data: [docs/PLAN_V2.md](docs/PLAN_V2.md) is the pre-registration.

### Against the brief's acceptance criteria

| | v1 | **v2** |
|---|---|---|
| Dataset: normal + 5 fault types, repeatable, ground truth stored | ✅ | ✅ |
| Leakage-safe split, documented | ✅ | ✅ (v2 test generated after the freeze was pushed) |
| Baselines and ML on the same test, run once | ✅ | ✅ |
| Precision ≥ 0.75, recall ≥ 0.85, ≤ 2 false / 10 min | ❌ recall 0.844 | ✅ 0.85 / 0.98 / 0.12 |
| Median latency ≤ 3 events for faults expected to be caught | ❌ | ❌ **11** over all progressive faults (the original plan's definition) · ✅ 1 event on the narrower physics-based set declared for v2 |
| Defensible advantage at a comparable operating point, or the baseline ships | ✅ baseline shipped | ✅ faster at the same false-alert rate |
| Reproducible; saved result has model version and frozen threshold | ✅ | ✅ |
| Tests pass, errors analysed, honest failure states | ✅ | ✅ 187 tests |

Evidence per criterion, the interpretation questions decided without the mentor's sign-off, and the limits of the result: [EVALUATION.md](docs/EVALUATION.md#acceptance-criteria-the-briefs-definition-of-done).

## Docs

| | |
|---|---|
| [PLAN.md](docs/PLAN.md) | written before any code; changes are recorded in the process log, not edited in |
| [PLAN_V2.md](docs/PLAN_V2.md) | v2 pre-registration, committed with the frozen v2 models before the v2 test set existed |
| [DATA_CARD.md](docs/DATA_CARD.md) | how telemetry is generated, labels, splits, distributions, leakage controls, limitations |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | shipped v2 model, features, thresholds, intended use and non-use, failure cases |
| [EVALUATION.md](docs/EVALUATION.md) | v2: acceptance criteria, protocol, test results with CIs, per-fault breakdown, comparison, error analysis, limits |
| [PROCESS_LOG.md](docs/PROCESS_LOG.md) | decisions, plan changes, bugs found, section by section |
| [RETROSPECTIVE.md](docs/RETROSPECTIVE.md) | what went wrong and what I'd change |
| [v1/](docs/v1/EVALUATION_v1.md) | v1's evaluation and model card, archived unchanged (paths only) |
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
signal-data generate --only-splits train,validation   # 1. train + validation only (~15 s)
signal-features build        # 2. causal window features, cached per data version     (~5 s)
signal-eval select-v2        # 3. VALIDATION: baselines, LOF grid, fast path, hybrids, grouping (~3.5 min)
signal-train                 # 4. freeze rule, robust z, LOF and the selected system  (~35 s)
                             #    (v2: this state was committed and pushed in d121e2a)
signal-data generate         # 5. now the test split too (train/validation byte-identical)
signal-features build
signal-eval test             # 6. OFFICIAL test: runs once per model version, then refuses
signal-eval plots            #    re-render every figure from saved outputs
```

The committed `results/official/` already holds the official v2 result, so step 6 refuses to run in this repo, by design. To reproduce it, run the steps in a scratch clone after deleting `results/official/` and compare with the committed files. The **exact evaluated artifacts are committed** under `models/` (SHA-256 in `models/registry.json` and in each official result file). v1 is reproducible from tag `iteration-1`.

Post-hoc analysis and audit helpers (read-only):

```bash
signal-eval show            # the official comparison, from the saved report
signal-eval audit           # proves which runs each model was fitted / selected / tested on
signal-eval fragmentation   # incidents per detected fault, per-fault precision
signal-eval envelopes       # the train-normal envelopes the rule limits came from
signal-eval replay-check    # every test run replayed through the service == the official incidents
signal-eval demo-threshold --detector hybrid_rule_fast --threshold 2.0   # DEMO ONLY, writes results/demo/
```

## Inference

**Replay a stored run** through the shipped (or any frozen) detector, as the service would:

```bash
signal-replay --run r4009 --show-truth                     # shipped model, all assets; ground truth shown afterwards
signal-replay --run r4009 --detector rule --asset drone-01 # the v1 baseline on the same run
signal-replay --run r4009 --threshold 2.0                  # DEMO ONLY threshold override
signal-replay --run r4009 --strict                         # window-by-window through the service code path
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

`decision` is **per event**: whether this window's last event crossed the threshold. The evaluated false-alarm rate (0.12 false incidents per 10 min for the shipped model) is for **incidents**, which are built from the stream of decisions (for the shipped model: open on the first alert, close after 5 quiet events) by the incident tracker, as `signal-replay` does. Every `/score` call writes one JSON audit line (asset, seq, status, score, threshold, decision, model version, a hash of the events); set `SIGNAL_AUDIT_LOG=<file>` to keep them in a file. Cost and scaling notes are in [MODEL_CARD.md](docs/MODEL_CARD.md#scaling-and-protocol).

## Tests

```bash
pytest                       # 187 tests, ~25 s
ruff check . && ruff format --check .
mypy                         # type check (clean)
```

The tests pin down preprocessing (`test_features.py`), split logic (`test_splits.py`), generator and label isolation (`test_generator.py`), threshold logic and metrics (`test_eval.py`), incident grouping (`test_incidents.py`), detectors (`test_detectors.py`), artifacts and the ship rule (`test_registry_and_decision.py`), the inference contract with replay (`test_service.py`), input validation, artifact verification and the other service hardening (`test_service_hardening.py`), the v2 fast path, hybrid, detectability and ship rule (`test_v2.py`), and the full pipeline with fitted detectors (`test_end_to_end.py`). CI installs the hash-pinned environment on Python 3.12.3 and runs ruff, mypy, pytest and a dependency audit, then builds both Docker images, runs the tests inside one with no network and checks the other serves the model as a non-root user, on every push: [GitHub Actions](https://github.com/hudasol/signal-anomaly-detection/actions) ([verified run on 5fba586](https://github.com/hudasol/signal-anomaly-detection/actions/runs/37689594485), the v2 state: hash-pinned install, ruff, mypy, 187 tests, dependency audit, both Docker images built, tests offline in one, the other serving the v2 model as non-root).

## Exceeds the bar

| Item | What | Risk it addresses |
|---|---|---|
| Threshold sensitivity | validation and post-hoc test curves, operating points marked ([figure](docs/figures/test_threshold_sensitivity.png)) | an arbitrary or cherry-picked operating point |
| Model versioning | `models/registry.json` ties artifact SHA-256, data version, feature schema, threshold, validation report and official result | not being able to say which model and data produced a number |
| Ablation | LOF refit without each feature group (v1, [v1 EVALUATION §7](docs/v1/EVALUATION_v1.md#7-ablation-validation-exceeds-the-bar-item)) | assuming more features are better |
| Pre-registered second test | v2 frozen and pushed before its test set was generated ([PLAN_V2](docs/PLAN_V2.md)); v1's result kept beside it | improving a model by re-using a test set you have studied |

## Layout

```
configs/         data.yaml  splits.yaml  features.yaml  detectors.yaml  eval.yaml
src/fleet_signal/
  data/          generator, faults, splits, telemetry loader, ground-truth loader (eval only)
  features/      build_features(), cache, split guard
  detectors/     rule, stats (robust z), lof, iforest, fastpath (v2), hybrid (v2)
  incidents/     batch grouping + streaming tracker
  eval/          protocol, threshold selection, bootstrap, ship decision, official test, plots
  service/       validation, scorer (failure states), FastAPI app, audit log, replay worker
  registry.py    artifacts + registry      train.py   signal-train
results/validation/   v2 selection outputs (committed)
results/official/     the v2 run-once test result (committed); posthoc/ = analysis after the fact
results/v1/           v1's validation and official results, unchanged
models/               the exact evaluated artifacts (v1 and v2) + registry.json (v2) + registry_v1.json
requirements.lock     exact environment (hashes)   requirements.runtime.lock   runtime subset
Dockerfile            runtime image (default) + test image (--target test)
```

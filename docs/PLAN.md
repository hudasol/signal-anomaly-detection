# Signal — Plan

Task 02 · anomaly detection and early warning for fleet telemetry
Written before implementation. Changes after this commit are recorded in `docs/PROCESS_LOG.md`, not silently edited here.

---

## 1. Decision

**What the detector supports.** An operator watching a mixed fleet (drone, rover, quadruped) needs to know *which asset to inspect now* and *why*. Signal advises; it never commands an asset, changes its mode, or suppresses Blackbox data.

**What counts as an anomaly.** Sustained behaviour on one asset that departs from that asset type's normal operating envelope in a way that would plausibly need operator attention. Specifically:

- a single out-of-range reading is **not** an anomaly on its own;
- a pattern across time (slope, persistence, oscillation) or across signals (e.g. moving but position not changing) **is**;
- known-normal hard cases (charging, return-to-base, hard manoeuvre, noisy-but-functional link) are **not** anomalies and count as false alerts if flagged.

**Output unit.** An *incident*: `asset_id`, open time, close time, peak score, decision, model version, top contributing signals. Individual window scores are internal; the operator sees incidents.

---

## 2. Data

### 2.1 Telemetry schema

One event per asset per tick at a fixed rate of **1 Hz**, matching Blackbox's event shape:

`run_id, seed, asset_id, asset_type, ts, battery_pct, temp_c, link_quality, pos_x, pos_y, speed_mps, mode`

`mode ∈ {idle, transit, inspect, charging, return_to_base}`

### 2.2 Generation

`signal-data generate --config configs/data.yaml` produces every run deterministically from a seed list. Same seed → byte-identical output (tested).

- Each run: **20 min (1,200 events) × 3 assets**.
- Normal behaviour per asset type: mission profile of transit/inspect/idle segments, battery discharge proportional to load, temperature following load with lag, link quality falling with distance from base plus noise, position integrated from speed and heading.
- Difficult-normal episodes are part of normal behaviour and appear in every split: **charging** (battery rises, temp rises), **return-to-base** (speed and heading change), **hard manoeuvre** (speed/temp spikes), **noisy link** (high-variance but non-degrading link).

### 2.3 Fault scenarios

One fault per fault run, injected on one randomly chosen asset at a random onset between 25% and 60% of the run. Each has a labelled `fault_start` and `fault_end`.

| Fault | Type | Injected behaviour (parameters randomised per run) |
|---|---|---|
| overheating | progressive | temperature drift added on top of normal, rate drawn from a range |
| rapid battery drain | progressive | discharge slope multiplied by a factor drawn from a range |
| link degradation | progressive | sustained decline or growing oscillation in link quality, optionally ending in dropout |
| sensor freeze | abrupt | one field (random choice) stops changing while the asset keeps moving |
| position / motion anomaly | abrupt | position jump, or reported speed disagreeing with displacement |

Parameters (magnitude, rate, duration, affected field) are sampled, so test faults are not copies of training faults. The test split uses **wider parameter ranges** than validation, including weaker faults (see §3).

### 2.4 Label isolation

- Telemetry goes to `data/<version>/telemetry.parquet`. Ground truth goes to a **separate** `data/<version>/ground_truth.parquet` (`run_id, asset_id, fault_type, fault_start, fault_end`).
- The feature pipeline only ever reads `telemetry.parquet`. A test asserts the feature table contains no column derived from ground truth and that the feature code never imports the ground-truth loader.
- `mode` is a real telemetry field an operator would see, so it may be used as a feature. No mode value is ever introduced by a fault.

---

## 3. Split

Splits are by **seed / run**, never by row. Recorded in a committed `configs/splits.yaml`.

| Split | Seeds | Contents | Used for |
|---|---|---|---|
| train | 1000–1039 | 40 normal runs (incl. difficult-normal episodes) | fitting models and scalers |
| validation | 2000–2039 | 10 normal + 30 fault runs (6 per fault type) | features, hyperparameters, incident params, threshold |
| test | 3000–3059 | 15 normal + 45 fault runs (9 per fault type), wider fault parameter ranges | official result, run **once** |

- Train is normal-only: the models are novelty detectors that learn the normal envelope.
- Rolling windows are computed per `(run_id, asset_id)` and never cross run boundaries.
- Scalers and any statistics are fit on **train only**.
- Tests assert: no `run_id` or seed appears in two splits; window construction never spans two runs.

**Test-once protocol.** `signal-eval --split test` writes `results/official/test_<model_version>.json`, including the frozen threshold, model version, data version and git SHA. If that file already exists the command refuses to run. Demo threshold changes go through a separate `--demo` path that writes elsewhere and never touches the official result.

---

## 4. Features

Computed over a trailing window of **W = 30 events** per asset (W is tuned on validation from {15, 30, 60}). The same `build_features()` function is used in training, evaluation and the service.

| Group | Features |
|---|---|
| level | last value of temp, battery, link, speed |
| trend | OLS slope of temp, battery, link over the window |
| volatility | rolling std of link and temp; count of link drops below a floor |
| freeze | events since each field last changed |
| motion consistency | `|reported_speed − displacement/Δt|`; max single-step displacement |
| context | one-hot `mode`, `asset_type` |

Feature groups are named so an ablation can remove them one at a time (exceeds-the-bar candidate).

---

## 5. Baselines

Both baselines go through the same incident grouping and threshold-selection procedure as the model.

**Rule baseline.** Transparent per-asset-type limits on level and slope (e.g. temp slope above X °C/min outside `charging`, battery slope steeper than Y %/min outside `charging`, link below Z for K events, any field unchanged for F events while speed > 0, implied speed mismatch above S). Constants are set on validation; the score is the number of rules firing, scaled by how far over the limit.

**Statistical baseline.** Robust z-score of every feature against train-normal median and MAD, per asset type and mode. Score = max |z| across features.

The ML model has to justify replacing these.

---

## 6. Model

**Choice: Isolation Forest** (scikit-learn), trained on train-normal windows only, scored with `score_samples`. No `contamination` setting; the threshold is chosen separately (§7).

**Why it fits:**
- does not assume a distribution, and the features mix bounded, skewed and count variables;
- handles cross-signal interactions that independent per-feature limits miss;
- cheap to train and score, small artifact, deterministic with a fixed `random_state`.

**Assumptions and where they may break:**
- anomalies are few and isolatable in feature space → difficult-normal episodes are also rare, so they may look anomalous (the main expected false-positive source);
- axis-aligned splits → slow faults that only show up as a combination may be weak early on, which hurts latency;
- trained on normal only → it cannot tell fault *types* apart; it only says "unusual".

**Hyperparameters** (`n_estimators`, `max_samples`, `max_features`) are tuned on validation only.

**Evidence / top signals.** Isolation Forest has no native per-feature attribution. Evidence is reported as the features with the largest robust deviation from train-normal for that window, and labelled as *evidence*, not as the model's internal reasoning.

**Escalation rule.** If Isolation Forest does not beat the baselines on validation, try One-Class SVM or LOF (novelty mode) once. No deep learning unless both classical models fail for a reason that a sequence model would plausibly fix.

---

## 7. Metrics, threshold and comparison

### 7.1 Metric definitions

- **Fault event** = one injected fault on one asset in one run.
- **Detected** = an incident opens on the faulted asset between `fault_start` and `fault_end + grace` (grace = 10 events).
- **Recall** = detected fault events / all fault events.
- **Precision** = incidents that overlap a fault window (on the right asset) / all incidents.
- **False incident rate** = incidents with no overlapping fault, per **10 minutes of fleet time** (all 3 assets together), measured on normal runs plus the non-fault portions of fault runs.
- **Detection latency** = events from `fault_start` to incident open, for detected progressive faults (overheating, battery drain, link degradation). Report median and p90.
- Window-level precision/recall is reported as a secondary diagnostic only.

**Expected-to-catch set**, declared now: all five fault types. Latency bar applies to the three progressive ones. If a fault type is later dropped from this set, that is recorded as a failure in `EVALUATION.md`, not a redefinition.

### 7.2 Threshold selection

For each detector separately, on validation:
1. sweep the score threshold;
2. keep thresholds where the false-incident rate is **≤ 1.5 per 10 min** (margin under the bar of 2);
3. pick the one with the highest event recall; tie-break on lower median latency.

The threshold, incident parameters and window size are written into the model artifact and frozen before test.

### 7.3 Honest comparison

All three detectors are compared on the same test runs at **matched false-alert rate**: each uses its own validation-chosen threshold under the same false-alert budget. A precision–recall / false-alert curve per detector is also reported.

**Ship rule.** The ML detector ships only if, on test, its event recall is higher than the best baseline by a margin whose 95% bootstrap CI (resampled over runs) excludes zero, *or* its median latency is meaningfully lower with no recall loss. Otherwise the conclusion says the baseline ships.

---

## 8. Incident grouping

Per asset, on the stream of window decisions:

- **open**: N consecutive windows above threshold (N chosen on validation from {1, 2, 3}).
- **close**: M consecutive windows below threshold (M chosen from {5, 10, 20}).
- **cooldown**: if a new crossing occurs within C events of a close (C from {30, 60}), it reopens the same incident instead of creating a new one.

The same N, M, C are used for all three detectors so grouping is not what separates them. N directly adds latency (N − 1 events), which is weighed against the 3-event latency bar.

---

## 9. Failure behaviour

The service never returns `normal` when it cannot score safely. Every response carries `status`:

| Situation | `status` | `decision` |
|---|---|---|
| model artifact missing, corrupt or wrong schema version | `unavailable` | `null` |
| fewer than W events of history for the asset | `insufficient_data` | `null` |
| required field missing / NaN / time gap > 3 events inside the window | `degraded` | `null` |
| scored successfully | `ok` | `normal` or `anomalous` |

Each row has a test. The demo will delete the artifact and show `unavailable`.

---

## 10. Service and artifact

- **Interface:** FastAPI `POST /score` (window of events in → status, score, threshold, decision, model version, top signals) plus a replay CLI `signal-replay --run <run_id>` that streams a stored run through the same scorer and incident grouper and logs every decision.
- **Artifact:** one `joblib` bundle containing model, feature schema, scaler stats, threshold, W/N/M/C, model version, data version hash, train seeds, git SHA.
- **Manifest:** `models/registry.json` linking artifact → data version → feature schema → threshold → evaluation report (exceeds-the-bar candidate).
- Signal is a separate component. An optional adapter can replay telemetry exported from Blackbox; Blackbox itself is not modified.

---

## 11. Tests

pytest, run in GitHub Actions CI:

- preprocessing: window construction, no cross-run windows, feature values on hand-computed fixtures;
- split: no overlap of runs/seeds; feature table has no ground-truth columns;
- threshold: selection respects the false-alert budget; frozen value round-trips through the artifact;
- incident grouping: open/close/cooldown on synthetic decision sequences;
- inference contract: response schema; `unavailable` / `insufficient_data` / `degraded` paths;
- determinism: same seed → identical data;
- end-to-end: small generate → train → evaluate → replay run.

---

## 12. Exceeds-the-bar targets

1. **Threshold sensitivity**: precision / recall / false-alert curves across thresholds with the chosen operating point marked. *Risk addressed:* an arbitrary or cherry-picked operating point.
2. **Model versioning manifest** (§10). *Risk addressed:* not being able to tie a result to the exact model, data and threshold that produced it.
3. *If time allows:* **ablation** by feature group, or a **generalisation test** holding out one asset type.

---

## 13. Risks and open questions

- **Designer bias.** I write both the fault generator and the rule baseline, so the rules may fit the generator unrealistically well. Mitigation: wider test parameter ranges, and rule constants set on validation only. Even so, a rule win here is weaker evidence than it looks; this goes in the limitations.
- **Latency bar vs. progressive faults.** A slow drift starting at `fault_start` may be physically indistinguishable from noise for the first few events. Trailing-window slopes also lag. The 3-event median may be hard to meet honestly; if missed, it is reported as missed, not fixed by redefining `fault_start`.
- **Difficult-normal false positives.** Charging and hard manoeuvres are rare in normal data, so a novelty detector may flag them. `mode` as a feature is the main mitigation.
- **Open with Awaiz:** (a) is per-event recall the intended unit for "fault-window detection"? (b) are wider/unseen fault parameter ranges in test acceptable?

---

## 14. Repository layout

```
configs/        data.yaml, splits.yaml, model.yaml
src/fleet_signal/
  data/         generator, faults, ground truth, loaders
  features/     build_features()
  detectors/    rule.py, stats.py, iforest.py
  incidents/    grouping
  eval/         metrics, threshold selection, reports, plots
  service/      FastAPI app, replay CLI, artifact loading
tests/
docs/           PLAN, DATA_CARD, MODEL_CARD, EVALUATION, PROCESS_LOG, RETROSPECTIVE
data/ models/ results/   generated, git-ignored except official results and registry
```

The package is named `fleet_signal` because `signal` would shadow Python's standard-library module.

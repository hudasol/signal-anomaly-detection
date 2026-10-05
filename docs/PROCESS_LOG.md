# Process log

Dated record of what was done, what changed from `PLAN.md` and why. Plan changes are recorded here rather than silently edited into the plan.

## 2026-10-05

- **PLAN.md committed** before any implementation (commit `bd42450`).
- **Section 1: project setup.** `pyproject.toml` (Python ≥3.12), `fleet_signal` package skeleton, GitHub Actions CI running `ruff check`, `ruff format --check` and `pytest` on Python 3.12. The package is `fleet_signal`, not `signal`, because `signal` would shadow the standard-library module.

## 2026-10-06

### Section 2: data generator

Built `signal-data generate`: 140 runs × 3 assets × 1,200 events at 1 Hz, about 503k events, written with ground truth in a separate folder.

**How normal behaviour is modelled.** Each asset runs a simple mission (park → move to waypoint → inspect → … → return → charge) with causal physics. Load comes from activity; battery drains with load and rises while charging; temperature follows load through a first-order lag; link quality falls with distance from base, with slow fading and white noise; reported position integrates heading and speed, plus GPS noise. Sensors are quantised (temperature 0.1 °C, battery 0.01 %, link 1 %), so exact repeated values also happen in normal data. That keeps "time since change" from being a trivial freeze detector.

**Difficult normal.** Charging and returning (triggered by low battery, plus an unplanned return in about half of asset-runs), hard manoeuvres (speed ×1.3–1.8 and a swerve, about 3 per moving asset per run) and noisy-link episodes. They appear in every split and are logged to `ground_truth/episodes.parquet` for false-positive analysis, never as features.

**Faults.** One fault per fault run, on one asset, starting at 25–60 % of the run. Overheating and battery drain persist to the end of the run. Link degradation either declines to dropout (persists) or oscillates and then recovers. Sensor freeze and motion anomalies last 40–300 s, then revert. Sensor faults wait until the asset is moving before starting (up to a deadline), because a frozen speed on a parked asset is indistinguishable from normal and would be a meaningless label.

**Changes from PLAN.md:**

1. **Telemetry schema now follows the Blackbox contract** instead of the names in PLAN §2.1: `temperature_c`, `link_quality_pct` on 0–100 (not 0–1), `heading_deg`, `z_m`, `seq`, `timestamp_utc`, and modes `idle / moving / returning / charging`. PLAN's `inspect` maps to `idle` (a hovering drone is idle but loaded, and `z_m` distinguishes it). *Why:* Signal should be able to replay Blackbox telemetry without a translation layer. I did not reuse the Blackbox simulator itself: it is a random walk (temperature moves ±0.5 °C per tick independently of everything else), so it cannot express causal faults or a meaningful normal envelope.
2. **The test set includes fault variants that never appear in validation**, not only wider parameter ranges: runaway overheating, accelerating drain, intermittent link dropouts, position freeze, multi-field freeze and position drift. About 40 % of test fault runs are unseen variants. *Why:* the brief warns against a test set that only contains anomalies matching the training examples, and I wrote both the generator and the rules (designer bias, PLAN §13). Evaluation will therefore report results separately for seen and unseen variants. This is also open question (b) for Awaiz.
3. **Ground truth is three files, not one**: `runs`, `faults` and `episodes`, all under `ground_truth/`.

**Bugs found while building:**

- *Unbalanced variants.* At first each fault variant was drawn at random per run. `signal-data summary` showed that the test set had zero multi-field freezes and only one drift case. Fixed by assigning variants in the run plan, balanced and shuffled separately from asset assignment, with a test.
- *Source silently not committed.* The `.gitignore` pattern `data/` (meant for generated data) also matched `src/fleet_signal/data/`, so the generator code was missing from the first commit attempt. Caught by checking the commit's file list before pushing; patterns are now anchored to the repo root (`/data/`), and the commit was verified from a fresh clone.
- *Data version did not track code changes.* The version hash only covered the YAML configs, so changing the generator produced a different dataset under the same version string. The hash now also covers the generator source files.

**Leakage controls built in:**

- `run_id` is `r<seed>` and carries no scenario; scenarios are shuffled across seeds within a split.
- Seeds are disjoint across splits, and runs are disjoint in time, with test after validation after train.
- Telemetry and ground truth live in separate files and separate loader modules. A test fails if the telemetry loader ever imports the ground-truth module.
- Each concern (run, fault, each asset) has its own RNG stream. A test checks that a fault run is **identical** to the same seed's normal run before the fault starts and on the other assets. Faults cannot leak backwards in time or across assets.
- A test checks that every one of the 16 fault variants actually produces the behaviour its label claims, against that clean twin run.

**Verified:** regenerating the full dataset into a second directory gives byte-identical files (SHA-256 of all four parquet files matched). 44 tests pass.

**Observations for later sections:**

- Some test-range faults are deliberately weak. Example: a quadruped battery fault of +1 %/min while idle looks like normal moving drain. It is only visible relative to mode. Expect these as false negatives for every detector; they belong in the error analysis, not in a retuned test set.
- A position jump is only directly visible at its two edges (offset on, offset off). In between, motion is self-consistent. Detection relies on the jump itself.
- **Latency risk (PLAN §13) is now concrete.** At 1 Hz, "≤ 3 events" means ≤ 3 seconds. A linear overheating at 4 °C/min adds about 0.2 °C in 3 s, about the same as one noise step. Early detection of slow progressive faults may not be achievable honestly; this will be measured, not assumed.

### Section 3: features

`build_features()` turns telemetry into one row per event using only that asset's own past events in the same run. The same function runs at training, evaluation and inference. 35 features in 6 named groups (level, trend, volatility, freeze, motion, context), so an ablation can drop one group at a time. A full rebuild over 503k events takes about 6 s; results are cached under the data version and a feature-schema hash.

**Change from PLAN §4:** instead of tuning one window W ∈ {15, 30, 60}, features use three fixed scales: 10, 30 and 120 events. *Why:* the two things a window has to do pull in opposite directions. Short windows react quickly, which matters for latency. Long windows separate a sustained drift from a normal transient: after an asset starts moving, its temperature rises at up to about 5 °C/min for a minute or two, which is inside the overheating range. Giving the detectors both scales lets them use each, rather than me picking one number. The scorable threshold is `min_history = 120` events. Before that, the service will return `insufficient_data`.

**New features not in the plan:**

- `mode_age` (seconds since the mode last changed). Normal transients happen right after mode changes; faults do not care about mode.
- `range_m` and `range_slope_m` (distance from the comms origin and its trend). Link quality only means something relative to distance.

**Guards and tests:**

- The OLS slope uses real timestamps, so lost packets do not distort it.
- A causality test checks that features at event *i* are identical when every later event is removed.
- No window crosses a run or an asset boundary.
- Hand-computed values are checked for slopes, jumps, freeze counts and mode age.
- An AST test fails if any file under `features/` or `detectors/` imports ground truth or the fault module.
- `assert_only_split()` refuses to fit on rows from runs outside the train split.

### Section 4: evaluation harness

All metric definitions are in `eval/protocol.py` and `configs/eval.yaml`, and every detector goes through the same code. Incident grouping: open after N consecutive alerts, stay open while quiet gaps are shorter than M, close after M quiet events. A re-open within the cooldown C is recorded on the same incident, not spammed as a new one.

Two definition choices worth stating:

- **An alarm that was already open before a fault started is a true positive (it overlaps the fault) but does not count as detecting it.** Only an open or re-open inside `[fault_start, fault_end + grace)` counts. Otherwise a noisy detector could get credit for faults it never reacted to.
- **False-alert time is scorable time only.** Events without enough history never alert, so they are not counted as "normal time with no false alarms".

The official test result can be written only once per detector and model version (opened with `"x"` mode, and the test checks that a second write fails). Demo runs go to `results/demo/`. Confidence intervals bootstrap whole runs, not events, because events within a run are correlated.

### Section 5: baselines on validation

**Rule baseline:** 11 hand-written rules. Their limits were set from **train-normal envelopes only**, about 10–20 % beyond the largest normal value, never from fault runs. Two things I learned while setting them:

- A "link declining" rule does not work. Even when the asset is not moving away, normal link slopes reach −33 %/min because of fading, which overlaps the fault's 8–20 %/min. The rule only catches outright dropout; normal link never goes below 25 %.
- Battery-slope limits only make sense once the asset has been in one mode for 120 s. Before that, the long window mixes two modes. The rule has an explicit `mode_age ≥ 120` condition.

**Statistical baseline:** robust z-score per (asset type, mode) against train median and MAD; score = max |z|.

**Change from PLAN §7.2: threshold selection now also requires precision ≥ 0.80.** The first sweep showed that "highest recall within 1.5 false alerts per 10 min" picks thresholds with about 0.3 precision. Validation has about 650 fleet-minutes of normal time but only 30 faults, so the false-alert budget still allows about 97 false incidents. That is far below the brief's 0.75 precision bar. The rule is now: false alerts ≤ 1.5 per 10 min **and** precision ≥ 0.80, then the highest recall. This was decided on validation, before any test data was touched.

**Shared incident parameters** (chosen on validation across both baselines): N = 2, M = 5, C = 30.

**Validation results** (not the official result; test is untouched):

| | Precision | Recall | False incidents / 10 min | Median progressive latency |
|---|---|---|---|---|
| Rule | 0.88 (CI 0.78–0.97) | 0.93 (0.83–1.00) | 0.08 | 81.5 events |
| Robust z | 0.83 (0.62–1.00) | 0.40 (0.23–0.58) | 0.06 | 474 events |

The rule baseline beats robust z on recall by 0.53 (paired bootstrap CI 0.32–0.73).

**What this tells me before building the ML model:**

1. **The rule baseline is strong.** At almost exactly its written limits it reaches precision 0.88 and recall 0.93. The ML model has a real bar to clear, and "ship the rule" is a live outcome.
2. **The max-|z| baseline fails on slow faults.** It catches 0 of 6 overheating faults. With about 30 heavy-tailed features, the normal maximum |z| is so large that the threshold ends up near 150. Only abrupt, extreme events (jumps, freezes) clear it.
3. **Latency is the real problem.** The rule's median latency on progressive faults is about 80 events, against a bar of 3. Overheating needs the 120 s slope to exceed the normal warm-up envelope. At 4–10 °C/min that takes about a minute. This is the PLAN §13 risk confirmed.
4. **First error pattern (rule false positives):** 4 of the 5 false incidents start at seq 120–142, the first moment an asset becomes scorable. The asset is still warming up from its parked start, and the 120 s temperature slope looks like overheating. Not tuned away; recorded for the error analysis.
5. **Rule misses:** a weak link decline on a quadruped (it never hit the dropout floor within its window) and a battery-drain step on a quadruped. Both fit the "weak faults" observation from Section 2.

### Section 6: ML detector (validation only)

Reproduce with `signal-eval select-model`, which writes `results/validation/ml_model_grid.csv`, `ml_ablation.csv` and `ml_selection.json`. Incident params stay fixed at those chosen from the baselines (N=2, M=5, C=30), so adding the ML model cannot re-tune the grouping the baselines are judged with.

**Isolation Forest (PLAN's first choice) lost badly.** One forest per asset type, eight settings (raw or mode-conditioned z-score inputs; `max_samples` ∈ {256, 2048}; `max_features` ∈ {0.5, 1.0}). The best feasible setting reached recall 0.27 at precision ≥ 0.8. It even missed position jumps, which the rule catches every time. My reading: most faults here show up as *one* feature going extreme (a jump, a frozen counter, a slope). With about 33 dimensions, an extreme value in one of them only shortens isolation paths in the trees that happen to split on it. Meanwhile heavy-tailed normal features (link slopes from fading) widen the normal range. Feeding it z-scores did not help; no z-score setting met the precision constraint at all. This is the assumption failure PLAN §6 flagged ("anomalies few and isolatable"), just in a different way than expected.

**Escalation (PLAN §6): LOF in novelty mode.** One model per asset type, fit on up to 30k train-normal rows, inputs are robust z-scores per (asset type, mode), clipped at ±20. Grid: `n_neighbors` ∈ {10, 20, 30} × `max_train` ∈ {15k, 30k}. All six settings met the constraints with recall 0.77–0.83. The selection rule (recall, then latency) picked `n_neighbors=20, max_train=30000`: precision 0.90, recall 0.83, 0.06 false alerts per 10 min, **median progressive latency 25 events**. Why LOF fits better: it is distance-based on standardised features, so one extreme coordinate moves a point far from all its neighbours, which is exactly the shape of these faults.

**Change from PLAN §6:** the ML detector is LOF, not Isolation Forest. Allowed by the plan's escalation rule; decided on validation only.

**Ablation (exceeds-the-bar item: what actually matters).** Refit with one feature group removed:

| Dropped | Precision | Recall | Latency |
|---|---|---|---|
| none | 0.90 | 0.83 | 25 |
| trend | 0.94 | **0.63** | **172** |
| motion | 0.81 | 0.67 | 24 |
| volatility | 0.89 | 0.77 | 18 |
| freeze | 0.86 | 0.83 | 23.5 |
| level | 0.83 | 0.87 | 17 |

Trend features carry the model: without them recall drops 0.20 and latency goes up about 7×. Freeze features add almost nothing to LOF, because the trend and motion features already move when a field freezes. Dropping *level* scores slightly better, but that is one fault out of 30, well inside the confidence interval. **I kept the pre-declared feature set** rather than chase a one-fault difference on 30 validation faults.

**Three-way validation comparison** (same incident params, each at its own validation-selected threshold):

| | Precision | Recall | False alerts / 10 min | Median progressive latency |
|---|---|---|---|---|
| Rule | 0.88 | 0.93 | 0.08 | 81.5 |
| Robust z | 0.83 | 0.40 | 0.06 | 474 |
| LOF | 0.90 | 0.83 | 0.06 | 25 |

Rule minus LOF recall: +0.10 (paired bootstrap CI 0.00 to 0.22). The trade-off going into the test: LOF is about 3× faster on progressive faults but catches fewer of them. Under the pre-declared ship rule (PLAN §7.3), a recall loss means LOF does not ship unless the test result says otherwise.

### Section 7: freeze and the official test (run once)

**Freeze.** `signal-train` fits rule, stats and LOF on train, takes their thresholds from the validation selection, and writes one artifact per detector (`models/<detector>/<model_version>/model.joblib`) plus `models/registry.json`. Two checks before touching test:

- Two training runs gave identical model versions and thresholds. Artifact bytes differ only by the embedded creation time; the registry stores the SHA-256 of the exact file that was evaluated.
- A dry run of the frozen artifacts on **validation** reproduced the committed validation report exactly, for all three detectors.

The code and registry were committed (`05fcfcb`) **before** the test split was read. LOF artifacts are about 25 MB (LOF keeps its training points), so artifacts are not committed; `signal-train` rebuilds them deterministically.

**Official test result** (`signal-eval test`, frozen thresholds, incident params N=2 / M=5 / C=30; 60 runs, 45 faults, 969.5 normal fleet-minutes):

| | Precision | Recall | F1 | False incidents / 10 min | Median progressive latency | Bar |
|---|---|---|---|---|---|---|
| Rule | 0.81 | 0.84 | 0.83 | 0.12 | 71 events | fails recall (by 0.006), latency |
| Robust z | 0.79 | 0.38 | 0.51 | 0.05 | 534.5 events | fails recall, latency |
| LOF | **0.97** | **0.89** | **0.93** | **0.02** | **16 events** | fails latency only |

**Ship decision (pre-declared rule, PLAN §7.3): the rule baseline.** LOF's recall gain over the rule is +0.044 with a paired CI of −0.08 to +0.16, so it is not established. Its latency gain is a median 24 events faster over the 20 progressive faults both caught, with a CI of −53.5 to **+3.5**, which just includes zero, so that is not established either.

**What the pre-declared rule missed.** On the same test runs, LOF is significantly better than the rule on **precision** (+0.16, paired CI +0.05 to +0.28) and on **false alerts** (0.10 fewer per 10 min, CI 0.02 to 0.20), with no recall loss. My ship rule only looked at recall and latency, because I wrote it thinking about missed faults, not operator alarm fatigue. That was a gap in the rule, not a property of the result. I am not changing the official decision after seeing the test; this is recorded here and in the retrospective, and the recommendation is discussed in EVALUATION.md.

**Seen vs unseen fault variants on test** (recall): rule 0.76 seen / 1.00 unseen; LOF 0.83 / 1.00. The variants held back from validation turned out *easier*, not harder: runaway heating, accelerating drain and position drift are more extreme than their validation counterparts. What made the test hard was the wider parameter ranges, which produced weaker versions of the *seen* variants.

**Concrete errors found** (details in EVALUATION.md):

- 9 of the rule's 12 false incidents start within 45 s of the asset first becoming scorable (warm-up temperature slope).
- LOF false positive in r3003: a 3-second charging → returning → charging blip. The trailing battery slope still reflects charging but is judged against "returning" statistics, giving z ≈ 33.
- A speed freeze in r3025 was missed by every detector. My generator's fallback forced it to start while the drone was **charging**, so a speed frozen at 0 is unobservable. This is a label-validity bug in the data design.
- Battery drain in r3015 was missed by every detector. The whole fault ran during charging above 80 %, where the normal charge rate already tapers, so the net +1.5 %/min sits inside normal.

**Plots** are re-rendered from saved outputs only, with `signal-eval plots`. Test threshold curves were computed *after* the official result was written and live in `results/official/posthoc/`, labelled post-hoc.

### Section 8: service, replay and failure states

- **`Scorer`** is the inference boundary. Every response has a `status`: `unavailable` (artifact missing, corrupt or schema mismatch), `insufficient_data` (fewer than 120 events), `degraded` (gap in the window, non-finite features, mixed assets, missing fields) or `ok`. Only `ok` carries a decision. There is no code path that answers "normal" without a score.
- **`IncidentTracker`** is the streaming twin of the batch grouping. A property test runs 300 random alert sequences with random N, M, C and checks the tracker produces exactly the same incidents as `group_alerts`. What the service does live is what evaluation measured.
- **FastAPI** (`uvicorn fleet_signal.service.app:app`): `GET /health`, `GET /model`, `POST /score`. It accepts flat Signal events or Blackbox-contract events with a nested `position`. When no model is loaded, `/score` returns HTTP 503 with `status: unavailable`.
- **`signal-replay`** streams a stored run through the scorer and tracker and logs every decision to `results/replay/*.jsonl`. Ground truth is loaded only after the replay ends (`--show-truth`). A test checks that the window-by-window service path (`--strict`) gives identical scores and incidents to the batch path, and another that replay opens incidents exactly where the evaluation harness does.

**Checked live:** uvicorn with the shipped model scored seq 700 of test run r3006 as `anomalous` (evidence: `temp_rise`). After renaming the artifact, `/health` and `/score` both return 503 `unavailable`. Replaying r3006 reproduces the official latencies exactly (rule 61 events, LOF 10).

**One change while building:** evidence for the rule detector used to list rules that had *not* fired, with negative margins, which would mislead an operator. Evidence now shows the strongest signal plus any others with a non-negative contribution. This affects only the explanation, not scores or decisions.

### Section 9: documentation

DATA_CARD, MODEL_CARD, EVALUATION and RETROSPECTIVE written from the saved outputs, not from memory. Writing them found three things:

1. **Fail-safe bug (fixed).** For an asset type the rules have no limits for, the rule detector fell back to its "nothing fired" score (−10), which the service would have reported as **normal**. Unknown asset types now get no score from every detector, so the service answers `degraded`. A test was added. Verified afterwards: the frozen artifacts reproduce the committed validation report exactly, because the fix only touches asset types absent from the data.
2. **Wrong number in this log.** The Section 7 entry said 1,105 normal fleet-minutes; the saved test result says 969.5. Corrected.
3. **Stale threshold claim.** My first EVALUATION draft quoted incident counts from an early sweep that used different incident params. Rewritten from `results/validation/*_threshold_curve.csv`. In the process I found the precision constraint cost LOF two validation faults: one step lower it had recall 0.90 at precision 0.79.

Also verified for the error analysis: the r3015 battery-drain miss happens at 60–72 % charge, where charging should be fast. It is missed because no feature knows that normal charge rate depends on state of charge, not because of taper at that moment.

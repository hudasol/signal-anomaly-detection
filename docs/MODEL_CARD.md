# Model card: Signal anomaly detector

| | |
|---|---|
| **Shipped model** | Rule baseline, `rule-64d39edc1d` (`"serving": "rule"` in `models/registry.json`) |
| **ML candidate (recommended for shadow mode)** | LOF, `lof-a588a4d11b` |
| Data version | `v1.0.0-9e903f9008` (train seeds 1000–1039) |
| Feature version / schema | `1.0.0` / `704b8f1fda` |
| Incident grouping | open after 2 alerts, close after 5 quiet events, re-open within 30 = same incident |
| Artifacts | the **exact evaluated files** are committed under `models/` (SHA-256 in `models/registry.json` matches the official result files); `signal-train` verifies and keeps them and refuses to overwrite an evaluated model |

**The shipped model does not meet the brief's bar:** it misses recall by one fault (0.844 vs 0.85) and latency by a wide margin (71 events vs 3). See EVALUATION.md.

Why the rule ships even though LOF scored higher is explained in EVALUATION.md §5. In short: the pre-declared ship rule required a statistically established recall or latency gain, and 45 test faults were not enough to establish either.

## Intended use

An **advisory** early-warning signal for an operator watching a mixed inspection fleet (drone, rover, quadruped). For each asset it scores the latest telemetry window, groups alerts into incidents, and says which signals drove the decision. Intended action: *look at this asset now*.

## Not intended for

- **Controlling assets.** No automatic return-to-base, shutdown or mission change. It never writes to Blackbox or to an asset.
- **Diagnosis.** The detectors say "unusual", not "battery cell failure". Evidence lists signals, not causes.
- **Safety-critical alarms.** It is not certified, not validated on real hardware, and it is slow on slow faults: median latency over the progressive faults it detects is 71 events (rule) and 16 events (LOF) at 1 Hz, but LOF needs a median of 206 events for link degradation and 365 for slow link declines.
- **Asset types, sensors or operating envelopes it was not built on.** Other platforms, other telemetry rates, night or sub-zero ambient.
- **Real telemetry, without re-validation.** All evidence comes from synthetic data (DATA_CARD.md).

## Inputs and outputs

**Input:** one asset's recent events on the Blackbox telemetry contract, oldest first. Send **650 events** (the longest look-back any feature uses, "seconds in this mode", is capped at 600), or everything since the asset's first event of the run. A shorter mid-run window would cut off look-back features and change the score, so it is answered `insufficient_data` instead. `POST /score`, or `signal-replay` for stored runs.

**Output:**

```json
{"status": "ok", "asset_id": "drone-01", "seq": 700, "score": 0.25, "threshold": -0.087,
 "decision": "anomalous", "model_version": "rule-64d39edc1d", "detector": "rule",
 "evidence": [{"signal": "temp_rise", "contribution": 0.25}], "history": 650, "reason": null}
```

`status` is one of:

| Status | When | Decision |
|---|---|---|
| `ok` | scored | `normal` / `anomalous` |
| `insufficient_data` | fewer than 120 events, or a mid-run window shorter than 650 events (look-back features would be cut off) | none |
| `degraded` | gap of more than 3 events in the last 120, non-finite feature, missing field, mixed assets | none |
| `unavailable` | model artifact missing, corrupt, or built for a different feature schema (HTTP 503) | none |

The service never returns `normal` when it could not score.

## Features (35, causal, per run and asset)

Computed by `fleet_signal.features.build_features`, identical at training, evaluation and inference (tested).

| Group | Features |
|---|---|
| level | temperature, battery, link, speed, distance from origin, altitude |
| trend | OLS slopes of temperature and battery (10/30/120 events), link (30/120), distance (30) |
| volatility | temperature std, link std / min (30), mean absolute link change (10) |
| freeze | events since temperature, battery, link, speed, position last changed |
| motion | speed implied by position, rolling median of reported vs implied speed, distance beyond what reported speed allows |
| context | seconds in current mode, mode one-hot, asset-type one-hot |

## Shipped model: rule baseline

Eleven transparent rules, each comparing one feature to a limit per asset type (battery: per type and mode, only once the asset has been in the mode for 120 s). Limits were set by hand from **train-normal envelopes only**, at or beyond the largest train-normal value: about 5–25 % beyond it for the continuous signals (e.g. drone link std 14.0 vs a normal maximum of 13.3), and 2–8× the normal maximum for the freeze counters (position freeze: 3 repeats, where normal is 0) (e.g. 8 identical speed readings while moving vs a normal maximum of 2–3), see `results/validation/rule_envelopes_train.csv` (`configs/detectors.yaml`). Rules cover temperature rise, battery drain, link dropout or instability, five frozen-field rules, speed/position mismatch and position jumps. Score = worst rule margin in units of its scale; frozen threshold **−0.087** (chosen on validation; 0 would be exactly at the written limits).

Evidence = the rule(s) that fired, with their margin.

## ML candidate: Local Outlier Factor (novelty)

One LOF per asset type (`n_neighbors = 20`), fit on up to 30,000 train-normal rows. Inputs are robust z-scores of the level, trend, volatility, freeze and motion features against train median and MAD per (asset type, mode), clipped at ±20. Raw LOF scores are rescaled per type against the scores of LOF's own training points (median → 0, 99.9th percentile → 1). Because those points are scored by the model fitted on them, this reference is optimistic, so "1" is **not** a calibrated train-normal percentile; the scale only makes the three per-type models comparable. The operating threshold (**2.279**) was chosen on validation, so results do not depend on the calibration being exact. *(An earlier version claimed 0 = typical normal and 1 = the train-normal 99.9th percentile; corrected.)* The ±20 clip on the inputs also flattens extreme counters (EVALUATION §8 pattern 3, r3025).

**Assumptions:** normal windows form dense regions per (asset type, mode); a fault moves a window far from all its normal neighbours, often along one feature. **Where they break:** short mode transitions (statistics of the wrong mode) and long normal stationary periods (event counters keep growing); see EVALUATION.md §8 patterns 1–2.

Evidence = features with the largest robust deviation from train-normal. This is **not** LOF's internal reasoning (LOF has no per-feature attribution) and is labelled as evidence.

**Isolation Forest**, the model originally planned, was evaluated and rejected on validation (best feasible recall 0.27).

## Performance (official test, run once)

| | Precision | Recall | False incidents / 10 min | Median progressive latency |
|---|---|---|---|---|
| Rule (shipped) | 0.81 (per fault 0.76) | 0.84 ❌ | 0.12 | 71 events ❌ |
| LOF | 0.97 (per fault 0.95) | 0.89 | 0.02 | 16 events ❌ |

Precision in brackets is per fault (a fault split into several incidents counts once): the rule splits 10 of its 38 detected faults into more than one incident, LOF 17 of 40.

Full tables, confidence intervals and per-fault results are in EVALUATION.md.

## Known failure cases

- **Warm-up after start (rule):** first 1–2 minutes of motion can look like overheating; 9 of 12 rule false incidents.
- **Brief mode flips (LOF):** a 4-second charging → idle → returning → charging flip produced LOF's highest false score.
- **Faults during charging:** extra drain while charging is missed by every detector (no state-of-charge context). A speed frozen at 0 during charging is missed by the rule (its freeze rules only run while moving) and LOF (input clipping), though robust z catches it.
- **Fragmented incidents:** one long fault can become several incidents when its alerts pause for longer than the cooldown (LOF: up to 6 on one fault).
- **Slow link decline (rule misses, LOF late):** fading masks declines below about 20 %/min.
- **Latency:** slow progressive faults are caught after tens of seconds (LOF) to minutes (rule), never within the brief's 3 events.
- **Asset types outside {drone, rover, quadruped}:** no limits or statistics exist, so every detector returns no score and the service answers `degraded` (found while writing this card: the rule detector used to fall back to a "nothing fired" score, which would have read as normal; fixed and tested).

## Limitations

Synthetic training and evaluation data generated by the same author who wrote the rules (designer bias). 40 train runs, 30 validation and 45 test faults; wide confidence intervals. No drift monitoring: a fleet whose normal behaviour changes (new firmware, a hotter season) will raise false alerts without warning. Thresholds are global per detector, not per asset.

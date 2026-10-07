# Model card: Signal anomaly detector (v2)

| | |
|---|---|
| **Shipped model** | **Rule baseline + fast-path residual detector**, `hybrid_rule_fast-a0918187ba` (`"serving": "hybrid_rule_fast"` in `models/registry.json`) |
| Threshold / grouping | 1.175 on the hybrid score; an incident opens on the first alert, closes after 5 quiet events, re-opens within 30 events as the same incident |
| Also evaluated | rule `rule-ac784bbf45`, robust z `stats-b688d29f11`, LOF (the ML detector) `lof-f9f000d78f` |
| Data version | `v1.0.0-ee836a6bb4` (train seeds 1000–1039; v2 test seeds 4000–4059) |
| Feature version / schema | `2.0.0` / `b539bdefb8` (41 features) |
| Artifacts | the exact evaluated files are committed under `models/` (SHA-256 in `models/registry.json` matches the official result files); serving checks the SHA-256 before loading |
| Previous version | v1 (shipped the rule baseline alone): [v1/MODEL_CARD_v1.md](v1/MODEL_CARD_v1.md), tag `iteration-1` |

**Official v2 test (run once):** precision 0.85, recall 0.98, 0.12 false incidents per 10 min, median latency **1 event** on the faults expected to be caught within 3 events, **11 events** over all progressive faults. It meets 7 of the 8 acceptance criteria. The latency criterion is met only on a narrower physics-based set declared for v2, not by the original plan's definition, so it counts as missed (EVALUATION.md).

## Intended use

An **advisory** early-warning signal for an operator watching a mixed inspection fleet (drone, rover, quadruped). For each asset it scores the latest telemetry window, groups alerts into incidents, and says which signals drove the decision. Intended action: *look at this asset now*. It is fastest on abrupt battery drain (median 2 events on test) and catches overheating in about 11 events.

## Not intended for

- **Controlling assets.** No automatic return-to-base, shutdown or mission change. It never writes to Blackbox or to an asset.
- **Diagnosis.** "Unusual battery drain" is a symptom, not a cause.
- **Safety-critical alarms.** Not certified, not validated on hardware, and slow on slow faults: link decline takes a median of 76 events, and a fault starting at the exact moment of a mode change can be caught late (EVALUATION §7 pattern 2).
- **Asset types, sensors, telemetry rates or operating envelopes it was not built on.** The fast path assumes 1 Hz and the noise levels of this simulator.
- **Real telemetry, without re-validation.** All evidence is synthetic, and the fast path matches the simulator's battery model by design (designer bias, EVALUATION §8).

## Inputs and outputs

**Input:** one asset's recent events on the Blackbox telemetry contract, oldest first, `seq` strictly increasing. Send **650 events** (the longest look-back any feature uses, "seconds in this mode", is capped at 600), or everything since the asset's first event of the run; at most 1,000. A shorter mid-run window would cut off look-back features and change the score, so it is answered `insufficient_data` instead. `POST /score`, or `signal-replay` for stored runs. Telemetry is assumed to be **1 Hz**: features measure time in events, so when timestamps are sent they must advance one second per `seq` step.

**Output:**

```json
{"status": "ok", "asset_id": "drone-01", "seq": 699, "score": 2.23, "threshold": 1.175,
 "decision": "anomalous", "model_version": "hybrid_rule_fast-a0918187ba",
 "detector": "hybrid_rule_fast", "history": 650, "reason": null,
 "evidence": [{"signal": "fast:batt_res3", "contribution": 10.63},
              {"signal": "fast:batt_res10", "contribution": 4.44},
              {"signal": "fast:temp_res3", "contribution": 0.71}]}
```

`decision` is **per event**. Incidents (for the shipped model: open on the first alert, close after 5 quiet events, re-open within 30 = same incident) are built from the stream of decisions by the incident tracker, as `signal-replay` does. The false-incident rates in this card and in EVALUATION.md are for incidents, not for individual decisions.

| Answer | When | Decision |
|---|---|---|
| `ok` | validated, full-context window, verified model | `normal` / `anomalous` |
| `insufficient_data` | fewer than 120 events, or a mid-run window shorter than 650 events (look-back features would be cut off) | none |
| `degraded` | gap of more than 3 events in the last 120; an impossible reading (battery or link outside 0–100, speed outside 0–100 m/s, temperature outside −60 to 200 °C, non-finite); timestamps not 1 Hz; and, for replay or direct use of the scorer, the contract problems below | none |
| `unavailable` (HTTP 503) | model missing, not listed in the registry, failing its SHA-256 check (checked *before* the file is unpickled), corrupt, or built for a different feature schema; registry or config unreadable | none |
| HTTP 422 `invalid_request` | malformed request: wrong types (e.g. `"12"` or `3.7` for `seq`, `true` for a number), unknown `mode` or `asset_type` (exact, case-sensitive), more than one asset, `seq` negative, duplicated or out of order, more than 1,000 events | none |
| HTTP 413 | request body over 2 MB, rejected before parsing | none |

The service answers `normal` only for a validated, full-context window scored by a verified model. Detectors also refuse to score an unknown asset type or mode (score NaN → `degraded`): an unknown mode used to switch off every mode-gated rule, so a battery draining at 30 %/min scored `normal` with `mode: "MOVING"` (found in the pre-release engineering review; fixed and tested).

## Scaling and protocol

Measured once on the development container (not a saved benchmark), one asset, 650-event windows sent sequentially, including input validation: `/score` p50 ≈ 43 ms, p95 ≈ 57 ms for the shipped hybrid (v1 rule and LOF: p50 ≈ 28 ms). Each call sends about 180 KB of JSON and recomputes every feature for the window. At 200 assets reporting at 1 Hz that is about 35 MB/s of mostly repeated data and about 9 CPU-seconds per second: workable for a demo, wasteful for a fleet. The next step is incremental feature state per asset (send only the newest event). The service holds no per-asset state, so it can run as several replicas; incident state lives with the consumer of decisions.

The feature-schema hash covers the feature builder's raw source, so even a formatting change there needs re-freezing. The v1 position-freeze key (`x·10⁶ + y`) was replaced in v2 by comparing x and y as a pair.

## Features (41, causal, per run and asset)

Computed by `fleet_signal.features.build_features`, identical at training, evaluation and inference (tested; all 180 test asset-runs replayed through the service reproduce the official incidents: `signal-eval replay-check`).

| Group | Features |
|---|---|
| level | temperature, battery, link, speed, distance from origin, altitude |
| trend | OLS slopes of temperature and battery (10/30/120 events), link (30/120), distance (30) |
| volatility | temperature std, link std / min (30), mean absolute link change (10) |
| freeze | events since temperature, battery, link, speed, position last changed |
| motion | speed implied by position, rolling median of reported vs implied speed, distance beyond what reported speed allows |
| **fast (v2)** | battery and temperature: slope over the last 3 and 10 events minus the slope over the 30 before (`batt_res3/10`, `temp_res3/10`); matching speed change (`spd_d3/10`) |
| context | seconds in current mode, mode one-hot, asset-type one-hot |

## Shipped model

**Rule part.** Eleven transparent rules, the same rules and limits as v1 (re-frozen as `rule-ac784bbf45` because the feature schema changed): each compares one feature to a limit set by hand from train-normal envelopes (temperature rise, battery drain per mode, link dropout and instability, five frozen-field rules, speed/position mismatch, position jumps). Score = the worst rule margin.

**Fast-path part** (`detectors/fastpath.py`). For each residual, per asset type and mode, on train-normal rows only: a linear fit on the matching speed change (a hard manoeuvre raises speed and drain together), then a robust centre and scale of what is left. The score is the largest one-sided z: more drain than expected, or hotter than expected. It is silent for 45 events after a mode change, because its 30-event baseline would span two modes.

**Combining** (`detectors/hybrid.py`). Each part's score is rescaled with its own train-normal scores (median → 0, 99.9th percentile → 1), and the hybrid score is the larger. One threshold (1.175) and the grouping (open on 1 alert) were chosen on validation. Evidence comes from the part that drove the score, prefixed `rule:` or `fast:`.

**Why this and not LOF.** On validation LOF alone missed the latency bar (expected-to-catch latency 64.5 events when opening on 1 alert, recall 0.90; recall 0.83 when opening on 2), and combined with the fast path it slowed the fast path to 4–5 events, because one shared threshold is set by LOF. Rule + fast was the only system that met the whole bar on validation (EVALUATION §6, PLAN_V2 §5). LOF stays the evaluated ML detector: on test it had the best precision (0.92) and fewest false alarms (0.06 / 10 min), with recall 0.87.

## Performance (official v2 test, run once)

| | Precision | Recall | False / 10 min | Expected-to-catch latency | All progressive |
|---|---|---|---|---|---|
| **Rule + fast (shipped)** | 0.85 (per fault 0.79) | 0.98 | 0.12 | 1 (5 of 5) | 11 |
| Rule | 0.83 (per fault 0.79) | 0.93 | 0.11 | 188 (4 of 5) | 76 |
| LOF | 0.92 (per fault 0.87) | 0.87 | 0.06 | 7 (4 of 5) | 25 |

Against the rule on the same test (paired 95 % CI): recall +0.044 (0.000 to +0.119), precision +0.02 (not significant), latency significantly lower (median paired difference −1 event, CI −37 to −1). The large gains are on battery drain (188 → 2 events) and overheating (64 → 11).

## Known failure cases

- **Onset at a mode change (r4000):** a drain that starts the same second the asset changes mode is hidden from the fast path (silent for 45 events, then its baseline already contains the drain). The rule part caught it 187 events later.
- **Saturated readings (r4023):** a link reading frozen at 100 % while charging at base looks exactly like normal saturation (normal runs of 100 reach 272 readings) and was missed by every detector.
- **Incident closes while the fault continues (r4009 and 13 more of 44 detected test faults):** the fast path flags a change and then adapts to the new rate within about 30 events, and slope checks go quiet once heating levels off. Treat a closed incident on an asset that alerted recently as "check again", not "resolved".
- **Single-event false alarms:** opening on the first alert turns some lone alerts into incidents (6 of 12 false incidents on test).
- **Charge taper (rule part):** near full charge the charging slope drops, which the rule's battery check can read as extra drain (r4048).
- **Slow drifts:** link decline (median 76 events) and slow overheating stay far above 3 events.
- **Fragmentation:** one fault can become several incidents (13 of 44 detected faults; up to 5).
- **Asset types or modes outside the trained set:** every detector returns no score and the service answers `degraded`; a mis-cased or unknown mode is rejected with 422.

## Limitations

Synthetic data from a generator I wrote, and the fast path was designed knowing its battery model. 40 train runs, 45 test faults, only 5 of them in the expected-to-catch set: wide confidence intervals. The ship rule changed between v1 and v2 after seeing v1's result (declared before the v2 test). Train and validation are shared with v1. No drift monitoring. Thresholds are global per detector.

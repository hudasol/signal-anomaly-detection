# Evaluation

All numbers below come from saved outputs: `results/validation/` (selection) and `results/official/` (the test, run once). Figures are re-rendered from those files with `signal-eval plots`.

## 1. Protocol

| | |
|---|---|
| Unit of detection | **fault event**: one injected fault on one asset in one run |
| Detected | an incident **opens or re-opens** on the faulted asset in `[fault_start, fault_end + 10)` |
| Recall | detected fault events / fault events |
| Precision | incidents overlapping a fault window on the right asset / all incidents |
| False-alert rate | incidents overlapping no fault, per **10 minutes of normal fleet time** (scorable asset-seconds outside fault windows ÷ 3 assets) |
| Latency | events from `fault_start` to the first in-window open; the bar applies to progressive faults (overheating, battery drain, link degradation) |
| Incident grouping | open after N = 2 consecutive alerts, close after M = 5 quiet events, re-open within C = 30 events counts as the same incident; chosen on validation from the baselines, shared by every detector |
| Threshold | per detector, on **validation**: false alerts ≤ 1.5 per 10 min **and** precision ≥ 0.80, then highest recall, then lowest latency |
| Test | frozen artifacts, evaluated once (`signal-eval test`); a second run is refused |
| Uncertainty | 95 % bootstrap CIs resampling **whole runs** (2,000 resamples); paired for comparisons |

An alarm that was already open before a fault started counts as a true positive (it overlaps the fault) but **not** as detecting it, so a noisy detector cannot take credit for faults it never reacted to.

**Split.** Test is 60 runs (15 normal, 45 fault), seeds 3000–3059, never used for any choice. It contains 969.5 minutes of normal fleet time. 16 of its 45 faults are variants never seen in validation.

## 2. Official test result

Frozen thresholds: rule −0.087, robust z 151.9, LOF 2.279. Model versions: `rule-64d39edc1d`, `stats-ab89662ec6`, `lof-a588a4d11b`. Data version `v1.0.0-9e903f9008`.

| Detector | Precision | Recall | F1 | False incidents / 10 min | Median latency, progressive (p90) | Median latency, all faults |
|---|---|---|---|---|---|---|
| **Rule** (shipped) | 0.81 [0.70, 0.91] | 0.84 [0.73, 0.94] | 0.83 | 0.12 [0.05, 0.21] | 71 (183) | 20 |
| Robust z | 0.79 [0.65, 0.93] | 0.38 [0.24, 0.52] | 0.51 | 0.05 [0.01, 0.10] | 534.5 (584) | 52 |
| LOF (ML) | **0.97** [0.92, 1.00] | **0.89** [0.79, 0.98] | **0.93** | **0.02** [0.00, 0.05] | **16** (429) | **9** |

### Against the brief's bar

| | Precision ≥ 0.75 | Recall ≥ 0.85 | False alerts ≤ 2 / 10 min | Median progressive latency ≤ 3 |
|---|---|---|---|---|
| Rule | ✅ 0.81 | ❌ 0.844 | ✅ 0.12 | ❌ 71 |
| Robust z | ✅ 0.79 | ❌ 0.38 | ✅ 0.05 | ❌ 534.5 |
| LOF | ✅ 0.97 | ✅ 0.89 | ✅ 0.02 | ❌ 16 |

No detector meets the latency bar. The shipped rule baseline misses the recall bar by 0.006 (38 of 45 detected; 39 would pass). The latency gap is discussed in §8.

### Event-level confusion (test)

| | Faults detected | Faults missed | True incidents | False incidents | Normal runs with any false incident | False incidents on fault runs |
|---|---|---|---|---|---|---|
| Rule | 38 | 7 | 51 | 12 | 2 / 15 | 8 |
| Robust z | 17 | 28 | 19 | 5 | 2 / 15 | 3 |
| LOF | 40 | 5 | 65 | 2 | 0 / 15 | 2 |

Window-level confusion (secondary diagnostic: every scorable event, alert vs inside a fault window):

| | TP | FP | FN | TN | Window precision | Window recall |
|---|---|---|---|---|---|---|
| Rule | 7,251 | 383 | 12,183 | 174,328 | 0.95 | 0.37 |
| Robust z | 981 | 252 | 18,453 | 174,459 | 0.80 | 0.05 |
| LOF | 8,100 | 235 | 11,334 | 174,476 | 0.97 | 0.42 |

Window recall is low by design. A fault window lasts until the end of the run, but the operator needs one incident, not an alarm on every second. Event-level metrics are the decision metrics.

## 3. Per-fault results (test)

| Fault type | Rule recall | Rule latency | LOF recall | LOF latency | Robust z recall |
|---|---|---|---|---|---|
| overheating | 0.89 | 35.5 | **1.00** | **10** | 0.00 |
| battery drain | 0.67 | 103 | **0.89** | **7** | 0.33 |
| link degradation | 0.78 | 71 | **0.89** | 206 | 0.11 |
| motion anomaly | **1.00** | 2 | 0.89 | 3 | 0.67 |
| sensor freeze | **0.89** | 12 | 0.78 | 7 | 0.78 |

By variant (n per variant is 1–5, so read these as examples, not rates):

| Variant | n | Rule | LOF |
|---|---|---|---|
| overheating / linear | 5 | 4 | 5 |
| overheating / runaway (unseen) | 4 | 4 | 4 |
| battery drain / step | 5 | 2 | 4 |
| battery drain / accelerating (unseen) | 4 | 4 | 4 |
| link / decline | 3 | 1 | 3 |
| link / oscillation | 3 | 3 | 2 |
| link / intermittent (unseen) | 3 | 3 | 3 |
| motion / jump | 3 | 3 | 3 |
| motion / speed mismatch | 3 | 3 | 2 |
| motion / drift (unseen) | 3 | 3 | 3 |
| freeze / temperature, battery, link, speed | 7 | 6 | 5 |
| freeze / position, multi-field (unseen) | 2 | 2 | 2 |

**Seen vs unseen variants.** Recall on seen variants: rule 0.76, LOF 0.83. On unseen variants: rule 1.00, LOF 1.00. The variants held out of validation turned out more extreme, not subtler. The test was hard because of the wider parameter ranges, which produced weaker versions of the seen variants.

![recall by fault type](figures/test_recall_by_fault_type.png)

## 4. Baseline vs model at an honest operating point

Each detector runs at its own validation-chosen threshold under the same constraints (false alerts ≤ 1.5 / 10 min, precision ≥ 0.80). On test they landed at 0.12 (rule) and 0.02 (LOF) false incidents per 10 min, so LOF reached its recall at a **lower** false-alert rate, not a higher one.

**Paired test differences** (same runs resampled for both detectors; 95 % CI):

| LOF minus rule | Difference | CI | Excludes zero? |
|---|---|---|---|
| Recall | +0.044 | −0.078 to +0.163 | no |
| Precision | +0.161 | +0.045 to +0.279 | **yes** |
| False incidents / 10 min | −0.103 | −0.195 to −0.021 | **yes** |
| Median progressive latency, 20 faults both detected | −24 events | −53.5 to +3.5 | no (only just) |

**Post-hoc, matched false-alert rate** (test threshold curves computed *after* the official result was written, in `results/official/posthoc/`; nothing here selected anything):

| Held to ≤ X false incidents / 10 min | Rule recall (latency) | LOF recall (latency) |
|---|---|---|
| 0.021 (LOF's operating point) | 0.73 (73) | 0.89 (16) |
| 0.124 (rule's operating point) | 0.84 (72) | 0.93 (9) |

![threshold sensitivity](figures/test_threshold_sensitivity.png)

## 5. Ship decision

**Pre-declared rule (PLAN §7.3).** The ML detector ships only if its recall beats the best baseline with a paired CI excluding zero, or it is faster with a CI excluding zero and loses no recall.

**Result: the rule baseline ships.** LOF's recall gain (CI −0.08 to +0.16) and latency gain (CI −53.5 to +3.5) are not established with 45 test faults. The registry (`models/registry.json`, `"serving": "rule"`) and the service follow this decision.

**What the rule missed.** LOF is significantly better on precision (+0.16) and false alerts (−0.10 per 10 min), with no recall loss, and better at every matched false-alert rate. My ship rule only considered recall and latency. I wrote it thinking about missed faults and did not weigh the operator's alarm load. That was a gap in how I specified the decision, not a property of the data. I am not changing the official decision after seeing the test; changing the rule now would be choosing it from test results.

**My recommendation.** Ship the rule as decided, and run LOF in **shadow mode** next to it: replay every live window through LOF, log its decisions with `signal-replay`, but do not act on them. Promote LOF if it keeps the precision and false-alert advantage on new data and its recall gain becomes significant. The next ship rule should be declared in advance as "at least as good on recall (non-inferiority margin) and significantly better on precision **or** latency".

## 6. Threshold choice

The frozen thresholds sit at the knee of the validation curves (`results/validation/threshold_sensitivity.png`, `*_threshold_curve.csv`):

- **Rule** (−0.087): lowering the threshold adds **no** recall (it stays at 0.933) while false incidents climb from 5 to 33 (−0.134) and 79 (−0.145). Raising it to 0.031 removes every false incident but costs one fault (recall 0.867). The selected point is the cheapest place to have full validation recall.
- **LOF** (2.279): one step lower (1.968) would reach recall 0.90 at precision 0.79, just under the 0.80 constraint. The precision constraint cost LOF two validation faults. On test, its curve stays above the rule's at every false-alert rate (figure above).
- **Robust z**: no threshold reaches precision 0.80 with useful recall. The max-|z| score is dominated by heavy-tailed normal features, so its operating point (151.9) only catches extreme, abrupt events.

## 7. Ablation (validation; exceeds-the-bar item)

LOF refit with one feature group removed (`results/validation/ml_ablation.csv`):

| Dropped | Precision | Recall | Latency |
|---|---|---|---|
| none | 0.90 | 0.83 | 25 |
| trend | 0.94 | 0.63 | 172 |
| motion | 0.81 | 0.67 | 24 |
| volatility | 0.89 | 0.77 | 18 |
| freeze | 0.86 | 0.83 | 23.5 |
| level | 0.83 | 0.87 | 17 |

Trend features carry the model (recall −0.20, latency ×7 without them). Freeze counters add almost nothing to LOF, because trends and motion consistency already move when a field freezes. More features were not automatically better (dropping *level* scored one fault higher), but the pre-declared set was kept rather than chase a one-fault difference on 30 faults.

**Isolation Forest** (the planned model) was also evaluated: best feasible recall 0.27 on validation (`results/validation/ml_model_grid.csv`). Faults here are mostly one feature going extreme, and in about 33 dimensions Isolation Forest only shortens paths in the trees that split on that feature. LOF, distance-based on standardised features, fits that shape. This was the PLAN §6 escalation, decided on validation.

## 8. Error analysis

Every pattern below is a concrete test case you can replay: `signal-replay --run <id> --detector <name> --show-truth`.

### Pattern 1: warm-up false alarms (rule baseline)

**9 of the rule's 12 false incidents start within 45 s of the asset first becoming scorable** (seq 119–164), for example r3048 rover, r3052 quadruped, r3037 quadruped. Every asset starts parked and cold; once it starts moving, temperature climbs towards its moving equilibrium at up to about 5 °C/min for a minute or two. The 120-second temperature slope is the same signal the `temp_rise` rule uses for overheating, and these warm-ups sit just at its limit. LOF has none of these because it also sees `mode_age` and the shape across features.

*Next:* make `temp_rise` relative to the expected approach to equilibrium (temperature minus a lagged-load model), or require the slope to persist past `mode_age > τ`.

### Pattern 2: brief mode transitions (LOF false positive)

**r3003, drone, seq 976** (LOF score 3.41 against threshold 2.28; evidence `batt_slope_m`, `batt_slope_l`, `batt_slope_s` at z ≈ 33). The drone flipped charging → idle → returning → charging within 4 seconds at base. LOF normalises features per (asset type, *current* mode), but the trailing battery slopes still describe the charging that just happened. Judged against the "returning" statistics (where battery falls), a strongly rising battery looks impossible. The second LOF false positive (r3018, drone charging for over 5 minutes) is the same mechanism with counters: `speed_unchanged` keeps growing during a long, perfectly normal charge.

*Next:* do not apply mode-conditioned statistics until the trailing window lies inside the current mode (fall back to per-asset-type statistics while `mode_age < window`), and cap or log-transform the event counters.

### Pattern 3: faults hidden by charging

**r3015, rover, battery drain (missed by every detector).** The whole fault ran while the rover was charging at 60–72 %. Extra drain cut the net charge rate from about 3.9 to 1.5 %/min. Every detector judges charging slopes per mode with no state-of-charge context, and normal charging legitimately slows below 1 %/min once a battery is above 80 %, so 1.5 %/min looks normal for "charging" in general. See `docs/figures/test_r3015_rover-01_battery_drain_step.png`.

**r3025, drone, speed freeze (missed by every detector).** This is a data-design bug, not a detector failure. The fault waited for the drone to move, hit its deadline, and started while the drone was charging. A speed frozen at 0 on a stationary asset is unobservable. The label should not exist.

*Next:* model the expected charge rate as a function of state of charge and use the residual as a feature. In the generator, re-draw the onset instead of forcing a sensor fault onto a stationary asset.

### Pattern 4: slow link decline drowned by fading

The rule caught only 1 of 3 link declines (r3001 quadruped and r3005 drone missed). LOF caught all three but late (median 365 events). Normal link quality swings by up to 33 %/min from fading and distance changes, which covers the fault's 4–30 %/min decline. The rule only fires at outright dropout, and slow declines reach it late or never.

*Next:* learn expected link quality as a function of distance from base (from train-normal data) and track the trend of the residual. That separates "far away" from "radio failing".

### Pattern 5: the latency bar is out of reach for slow drifts

Median progressive latency is 16 events (LOF) and 71 (rule) against a bar of 3. At 1 Hz, 3 events is 3 seconds. A 2 °C/min overheating adds 0.1 °C in 3 s, below the 0.15 °C sensor noise. With these labels (fault start = the moment the process begins), no detector working on these signals can confirm slow drifts in 3 events at this false-alert budget. Abrupt faults do meet it: motion anomalies are caught in 2–3 events and freezes in 7–12.

*Next:* a sequential change detector (CUSUM on the slope residual) would minimise detection delay for a given false-alarm rate. It is the principled tool for this bar. A label that also records the "detectable from" time would let latency be reported against both.

### False positive and false negative to explain in the demo

- **False positive:** LOF, r3003 drone seq 976 (pattern 2). `signal-replay --run r3003 --detector lof --asset drone-01 --show-truth`.
- **False negative:** r3015 rover battery drain during charging (pattern 3). `signal-replay --run r3015 --detector lof --asset rover-01 --show-truth`.

## 9. Limitations of this evaluation

- 45 test faults: confidence intervals are wide, and one fault moves recall by 2.2 points.
- Synthetic data that I designed, so designer bias favours the rule baseline (DATA_CARD.md).
- Incident parameters were chosen from the baselines and shared with LOF. A grouping tuned for LOF might change its latency, but was deliberately not explored.
- One false label (r3025) counts against every detector.

# Retrospective

## What went wrong

**My ship rule asked the wrong question.** I declared before the test that the ML model ships only if it significantly beats the best baseline on recall or latency. On test, LOF was better on every number: 0.97 vs 0.81 precision, 0.89 vs 0.84 recall, 6× fewer false alarms, 16 vs 71 events latency. Its precision and false-alarm advantages *were* significant. But the rule only looked at recall and latency, neither of which cleared significance with 45 faults, so the rule baseline ships. I wrote the rule thinking about missed faults and forgot that an operator drowning in false alarms is also a failure. I kept the decision rather than rewrite the rule after seeing the answer, but the rule was badly specified.

**I tuned my selection rule only after seeing it fail.** The plan said "highest recall within 1.5 false alerts per 10 min". The first validation sweep showed that this allows about 97 false incidents against 30 faults, so precision would be about 0.3. I added a precision ≥ 0.80 constraint. This happened on validation, so it is legitimate, but it is something I should have caught by doing the arithmetic in the plan: the false-alert budget and the precision bar are in different units, and the conversion depends on how much normal time there is per fault.

**The planned model lost badly.** I chose Isolation Forest in the plan with reasons that sounded right. It reached recall 0.27. The faults I generated are mostly "one feature goes extreme", and Isolation Forest is weak at that in about 33 dimensions. I had the information to predict this (I designed the faults) but did not connect it to how the algorithm isolates points.

**A label in the test set is wrong.** My sensor-fault generator waits for the asset to move before freezing a field, with a deadline. In test run r3025 the deadline fired while the drone was charging, so "speed frozen at 0" on a stationary drone is a fault nobody can see. Every detector is charged with a miss for it.

**Small process mistakes caught late:**

- `.gitignore`'s `data/` silently excluded `src/fleet_signal/data/`, and lint was skipping it too.
- The data version did not change when generator code changed.
- Fault variants were drawn at random instead of balanced.
- The rule detector could answer "normal" for an unknown asset type. I only found this while writing the model card, after the test.

## Bad assumptions

- **"Unseen variants make the test harder."** They made it easier. Runaway heating, accelerating drain and drift are more extreme than their validation versions, and every detector caught all 16 of them. The test was hard because the wider parameter ranges produced *weaker* versions of the variants validation had seen. If I want to stress generalisation, I need subtler unseen faults, not different ones.
- **"3 events of latency is a tuning problem."** At 1 Hz with 0.15 °C sensor noise, a 2 °C/min drift is physically undetectable in 3 seconds. I flagged this in the plan as a risk but still half-expected features to fix it. The bar needs either a sequential test (CUSUM) or a label for when a fault becomes detectable.
- **"Mode tells the detector what normal looks like."** It does, except during transitions. Trailing windows still describe the previous mode, which produced LOF's worst false alarm, and long charging sessions make counters grow in ways the training data rarely showed.
- **"Charging is a separate, easy regime."** Charging hid two faults. Charge rate depends on state of charge, and none of my features know that.

## What I'd redesign

1. **The decision rule:** non-inferiority on recall plus a significant gain on precision **or** latency, declared up front, with the operator's alarm load as a first-class metric.
2. **The fault generator:** never force a sensor fault onto a stationary asset; re-draw the onset. Record both "fault starts" and "fault becomes physically detectable" so latency can be reported against both.
3. **The test design:** generalisation sets made of subtler faults (and a held-out asset type), not just new fault shapes.
4. **Features:** residuals against expected behaviour (temperature vs a lagged-load model, charge rate vs state of charge, link vs distance) instead of raw slopes. Every error pattern in EVALUATION.md is a case where the detector compared a signal to the wrong "normal".
5. **Detectors:** an asset that just changed mode should be judged by per-type statistics until its windows sit inside the new mode.
6. **Order of work:** write the model card's "failure cases" section *before* freezing. It found a fail-safe bug that should have been caught before the test.

## Technical lessons

- The split is not just train / validation / test. It is also which code is allowed to *import* which data. Making ground truth physically unreachable from feature code, and testing that, was cheaper than being careful.
- Counterfactual twins (the same seed with and without the fault) are the strongest test I wrote. They prove a fault cannot leak backwards in time or across assets.
- Resample runs, not events, for confidence intervals. With 45 faults, a 0.04 recall difference is noise, and saying so is part of the result.
- A run-once guard has to cover the whole path: verify artifacts, check nothing exists, *then* read test. Committing the frozen state before the run is what makes "I didn't peek" checkable.
- A strong, boring baseline is the real bar. The rule baseline, with limits set from train data only, reached 0.93 recall on validation. It made the ML result meaningful.
- Distance-based anomaly detection on well-standardised features fitted this fault shape far better than tree isolation. Pick the algorithm from the shape of the anomalies, not from familiarity.

## What my own checks missed

- **The ship rule's blind spot.** Every test I wrote checked that the rule was *implemented* as declared (`test_registry_and_decision.py`); none asked whether the rule was the right one.
- **The unknown-asset-type "normal" fallback.** I tested every failure state I had listed (missing model, corrupt model, short history, gaps, NaN, mixed assets) but not "an input the model has no notion of".
- **The unobservable r3025 label.** My generator tests check that each fault variant is observable *against a clean twin*. They used fixed seeds where the asset was moving. No test checked that every label in the real dataset is observable.
- **A wrong number in my own log.** I wrote "1,105 normal fleet-minutes" in the process log from memory; the saved result says 969.5. It was corrected when writing EVALUATION.md from the files. Numbers in docs should come from saved outputs, not my notes.

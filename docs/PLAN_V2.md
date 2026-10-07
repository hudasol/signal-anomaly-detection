# PLAN v2: pre-registration

Written and committed **before the v2 test set existed**: the commit that adds this file also holds the frozen v2 models and the v2 selection outputs, and the test runs (seeds 4000–4059) are generated only after it. Nothing below may change once the test is read. Afterwards the v1 rule applies again: changes go in PROCESS_LOG, not here.

## 1. Why there is a v2

v1 (tag `v1.0.0`, `results/v1/`) met 6 of the brief's 8 acceptance criteria. The shipped rule baseline missed recall by one fault (0.844 vs 0.85), and no detector met the 3-event latency bar. The v1 test set was used once, as required, and then **studied**: the error analysis (v1 EVALUATION §8) is what v2 is built on. That makes the v1 test set development data now, so judging v2 on it would be grading against the answer key. v2 is judged only on a **new test set**.

The v1 result is not replaced. It stays in the repo and in the docs as v1's official result.

## 2. What changed (all decided on train and validation only)

| Change | From v1 error pattern |
|---|---|
| **Fast-path features**: battery and temperature slope over the last 3 and 10 events minus the slope over the 30 before (`batt_res3/10`, `temp_res3/10`), and the matching speed change (`spd_d3/10`) | pattern 5: 120-event slopes are too slow; pattern 3: drain hidden during charging |
| **Fast-path detector** (`detectors/fastpath.py`): per asset type and mode, regress each residual on the speed change (hard manoeuvres raise speed and drain together), then a one-sided robust z; silent within 45 events of a mode change | pattern 2: mode transitions |
| **Hybrid detector** (`detectors/hybrid.py`): several detectors' scores on a common train-calibrated scale, alert on the largest | — |
| **Incident opening after 1 or 2 alerts**, chosen per candidate on validation (close after 5 and cooldown 30 stay as chosen for the baselines) | pattern 5: N = 2 adds one event to every latency |
| **Position-freeze key** compares x and y as a pair instead of `x·10⁶ + y` (engineering review) | — |

Baselines, the false-alert budget (1.5 / 10 min), the precision constraint (0.80), the grace window (10 events) and the run-level bootstrap are unchanged.

## 3. Data

- Train (seeds 1000–1039) and validation (2000–2039): **the same runs as v1**, byte-identical (each run depends only on its seed).
- **Test: seeds 4000–4059**, same composition (15 normal + 45 faults), same wide parameter ranges and the same 6 test-only variants as v1's test. Generated with `signal-data generate` after this commit; train and validation were generated with `--only-splits train,validation`.
- Data version `v1.0.0-ee836a6bb4` (the split config changed, so the hash changed; the generator code did not).

## 4. "Faults the detector is expected to catch" (latency criterion)

Defined from physics, not from any detector (`eval/detectability.py`): a progressive fault is **expected to catch** within the bar if, 3 events after its labelled start, the change it has caused is at least **5×** the standard deviation of a 3-event difference of that signal under the configured sensor noise and resolution. With the configured noise this means: battery drains with extra drain ≥ about 1.5 %/min. No overheating (it would need > 21 °C/min) and no link fault qualify.

**Latency criterion for v2:** the median latency over the expected-to-catch faults, with a **missed fault counted as infinite latency** (so missing hard cases cannot improve the median), must be ≤ 3 events. The all-progressive median (v1's definition) is reported next to it. It is expected to stay above 3, because slow overheating and link decline are physically slower than 3 events.

## 5. Selection on validation (done; `results/validation/v2_selection.json`, `v2_candidates.csv`)

Candidates: LOF (neighbours 10/20/30, chosen 20), fast path, rule + fast, LOF + fast, rule + LOF + fast, each opening an incident after 1 or 2 alerts. Rule: feasible (false incidents ≤ 1.5 / 10 min and precision ≥ 0.80), then **meets the brief's bar on validation** (recall ≥ 0.85, expected-to-catch latency ≤ 3), then recall, then expected-to-catch latency, then all-progressive latency.

| Validation | Open after | Precision | Recall | False / 10 min | Expected-to-catch latency | All progressive |
|---|---|---|---|---|---|---|
| rule (baseline) | 2 | 0.88 | 0.93 | 0.08 | — | 81.5 |
| LOF | 2 | 0.90 | 0.83 | 0.06 | 65.5 | 25 |
| LOF + fast | 2 | 0.83 | 0.97 | 0.14 | 5.0 | 18 |
| rule + LOF + fast | 2 | 0.84 | 0.97 | 0.12 | 5.0 | 18 |
| **rule + fast (selected)** | **1** | **0.94** | **0.93** | **0.05** | **1.5** | 7 |

A first pass allowed only systems containing LOF as the shipped candidate. On validation, one shared threshold over a hybrid is set by LOF and slows the fast path to 4–5 events, so that restriction would have selected a system that fails the latency bar. It was lifted before this freeze (PROCESS_LOG 2026-10-08). LOF is still evaluated on the same test as the ML detector, as the brief requires.

Frozen: `rule-ac784bbf45` (−0.087), `stats-b688d29f11` (151.9), `lof-f9f000d78f` (2.279), **`hybrid_rule_fast-a0918187ba` (1.175, open after 1 alert)**. SHA-256 in `models/registry.json`.

## 6. Ship rule v2 (`eval/decision.py::ship_decision_v2`)

The selected system ships over the best baseline (chosen by **validation** recall: the rule) if, on the v2 test, with paired run-level bootstrap 95 % CIs:

1. it is **not worse on recall**: the lower end of the CI of (candidate − baseline) is above −0.05, **and**
2. it is significantly better on **precision** (CI above 0) **or** on **progressive latency** (paired CI of the median difference below 0).

Otherwise the baseline ships, and the conclusion says so. This replaces v1's rule, which ignored precision and false alarms (v1 RETROSPECTIVE).

## 7. What will be reported, whatever happens

- All four frozen detectors on the v2 test, run once (`signal-eval test`; a second run is refused).
- The acceptance-criteria table for v2, including any criterion still missed.
- Recall on seen and unseen variants, per fault type, per-fault precision, fragmentation.
- v1 and v2 side by side, with the v1 result unchanged.

**Predictions, written down now:** criterion 4 (precision / recall / false alerts) is likely to be met: the fast path adds the battery drains the rule missed in v1. Criterion 5 on the expected-to-catch set is likely to be met if the fast path holds up on the wider test ranges. The all-progressive median will stay above 3.

# Acceptance demo runbook (v2)

Every item the brief lists, with the exact command and what it shows. Nothing here edits a model, regenerates data or touches `results/official/`. Outputs below are from a rehearsal on the committed state. v1's runbook is in git history (tag `iteration-1`).

**To record the video in one take:** `bash scripts/demo_walkthrough.sh` runs every step below in order and waits for Enter between steps (step 7 always puts the model back, even on Ctrl-C). `NO_PAUSE=1` runs it straight through as a rehearsal.

Setup, once: `pip install --require-hashes -r requirements.lock && pip install --no-deps -e . && signal-data generate` (about 20 s; byte-identical data). **Do not retrain:** the exact evaluated artifacts are committed under `models/`, SHA-256 checked against `models/registry.json`. Docker alternative: `docker build -t signal . && docker run --rm signal sh -c "signal-data generate && signal-eval show"`.

## 1. The split, and proof held-out runs were not used

```bash
signal-data summary          # seed ranges, runs per split and scenario, variants
signal-eval audit
git log --oneline d121e2a -1 && git log --oneline eb83aa7 -1
```

```
data version v1.0.0-ee836a6bb4
  train      seeds 1000-1039  (40 runs)
  validation seeds 2000-2039  (40 runs)
  test       seeds 4000-4059  (60 runs)
pairwise overlap of run ids: train/val 0, train/test 0, val/test 0
rule-ac784bbf45: no fit; limits hand-set from train-normal envelopes (...); threshold chosen on validation
stats-b688d29f11: fitted on 40 runs, all in train: True; any test run used: False
lof-f9f000d78f: fitted on 40 runs, all in train: True; any test run used: False
hybrid_rule_fast-a0918187ba: fitted on 40 runs, all in train: True; any test run used: False
runs referenced in the validation selection outputs: 32, all in validation: True; any test run: False
official test report: results/official/test_report_e9deed50ee.json
detector_code_hash now: 3f62fb42be
scoring-code commits since freeze (d121e2a): 2
  86d4307 beyond the bar: generalisation test, drift monitor, ablation, shadow replay, ...
  ebda1e7 v2 docs, evaluation and rehearsal fixes: ...
```

The two commits touched only the replay command (shadow mode, drift line, severity, a broken-pipe fix), not scoring: `detector_code_hash` (detectors + feature builder) is the same as at the freeze, and `signal-eval replay-check` still reproduces all 180 official test incidents.

Then point at:

- **The pre-registration.** `d121e2a` (frozen models, `docs/PLAN_V2.md`, selection outputs) was pushed to GitHub **before** the v2 test runs existed: train and validation were generated with `--only-splits train,validation`. Independent timestamp: GitHub Actions run 37686477806 was triggered by that push at 21:02:04 UTC; the test telemetry file was written at 21:02:33 UTC. (That run's Docker job failed because `results/official/` did not exist yet; its test job passed.) The test runs were generated and evaluated in `eb83aa7`. Say why: v1's test set was studied for v2's design, so v2 needed a new one.
- **The guard.** `signal-eval test` again refuses (the official files exist).
- **The leakage tests:** `pytest tests/test_splits.py tests/test_features.py tests/test_v2.py -k "split or leak or causal or seeds or separately" -v`.

## 2. A normal held-out run and its false-incident count

```bash
signal-replay --run r4002 --show-truth        # shipped model: 0 incidents on all three assets
signal-replay --run r4058 --show-truth        # shipped model: 1 false incident
signal-replay --run r4058 --detector rule     # rule baseline: 0
```

r4058: one false incident on the drone at seq 1017, a single alert (`fast:batt_res10, fast:batt_res3`), closed 5 events later. It is the price of opening on the first alert (EVALUATION §7 pattern 4). On the 15 normal test runs the shipped system raised false incidents on 3 runs, the rule on 1.

## 3. A held-out fault: score, threshold crossing, incident open, latency

```bash
signal-replay --run r4009 --asset drone-01 --show-truth                  # shipped
signal-replay --run r4009 --detector rule --asset drone-01 --show-truth  # v1's shipped baseline
```

```
model hybrid_rule_fast-a0918187ba  threshold 1.1749  incident params {'open_n': 1, 'close_m': 5, 'cooldown_c': 30}
  drift check vs train: caution (score 1.46; batt_slope_l, temp_c, temp_slope_l)
  seq   698  incident #1 opened    score 1.307  evidence: fast:batt_res3, fast:batt_res10  [P3 0.32]
  seq   730  incident #1 closed    score 0.501
GROUND TRUTH: drone-01: battery_drain / step  seq 697-1200
    detected: incident #1 opened at seq 698 -> latency 1 events
```

The rule baseline (same rules as v1, re-frozen for the v2 feature schema): **0 incidents, not detected** (LOF also missed it). Then say the honest part: the incident **closes at seq 730 while the drain continues**, because the fast path adapts to the new rate within about 30 events (EVALUATION §7 pattern 5, 14 of 44 detected faults). Per-event scores are in `results/replay/r4009_<model>.jsonl`.

## 4. Rule vs robust z vs ML vs shipped, on the same test data

```bash
signal-eval show
```

```
detector          model version                 open_n    precision       recall           f1 fp_per_10min progressive_ expected_lat
rule              rule-ac784bbf45                   2        0.828        0.933        0.878        0.113       76.000      188.000
stats             stats-b688d29f11                  2        0.625        0.311        0.415        0.093      403.000         miss
lof               lof-f9f000d78f                    2        0.920        0.867        0.893        0.062       25.000        7.000
hybrid_rule_fast  hybrid_rule_fast-a0918187ba       1        0.848        0.978        0.908        0.123       11.000        1.000
hybrid_rule_fast - rule recall +0.044 (CI +0.000, +0.119)
hybrid_rule_fast - rule precision +0.020 (CI -0.082, +0.111)
hybrid_rule_fast - rule latency -1.0 events (CI -37.0, -1.0, n=25)
SHIP: hybrid_rule_fast - recall not worse (CI low +0.000 > -0.05) and significantly better on latency
```

Then open `docs/figures/test_recall_by_fault_type.png` and EVALUATION §4: why it ships (latency), how big the gain really is (median paired 1 event; battery 188 → 2), and that the rule alone also passes the recall bar on this test set.

## 5. One false positive and one false negative, explained

**False positive: r4037 quadruped, incident #3 at seq 860.**

```bash
signal-replay --run r4037 --asset quad-01 --show-truth
```

Evidence `fast:batt_res10, fast:temp_res10, fast:batt_res3`. The quadruped speeds up from about 0.9 to 1.1 m/s, battery drain rises with it, and the 10-event battery residual reaches −0.68 %/min; the speed covariate explains only part of it. The same replay shows a true detection: the position jump at seq 602, caught in 0 events.

**False negative: r4023 quadruped link freeze (missed by every detector).**

```bash
signal-replay --run r4023 --asset quad-01 --show-truth
```

The reading froze at 100 % while charging at base. Checked against normal data: a charging quadruped's link reads 100 for up to 272 readings in a row, and the freeze sits inside a run of 84. Indistinguishable with these signals.

**The fast path's own blind spot: r4000 (expected to catch, caught after 187 events).**

```bash
signal-replay --run r4000 --asset quad-01 --show-truth
```

The drain started the same second the quadruped switched from idle to moving; the fast path is silent for 45 events after a mode change and then its baseline already contains the drain. The rule part caught it at seq 799. Figure: `docs/figures/test_r4000_quad-01_battery_drain_step.png`.

## 6. Change the threshold for the demo, then put it back

```bash
signal-eval demo-threshold --detector hybrid_rule_fast --threshold 2.0
signal-eval demo-threshold --detector hybrid_rule_fast --threshold 0.9
```

```
DEMO ONLY (hybrid_rule_fast-a0918187ba); official result untouched
  frozen threshold    1.175  precision=0.848  recall=0.978  fp_per_10min=0.123  n_false_incidents=12
  demo   threshold    2.000  precision=1.000  recall=0.844  fp_per_10min=0.000  n_false_incidents=0
  demo   threshold    0.900  precision=0.272  recall=0.978  fp_per_10min=1.953  n_false_incidents=190
```

Higher: no false alarms, but 6 faults lost and recall falls below the bar. Lower: no recall gained, 178 extra false incidents. The frozen point sits at the knee (`docs/figures/test_threshold_sensitivity.png`, post-hoc curves, frozen points marked).

"Put it back": the frozen threshold was never changed. The demo writes to `results/demo/` only:

```bash
git status --short results/official models/registry.json     # nothing
signal-replay --run r4009 --asset drone-01 | head -1          # threshold 1.1749
```

## 7. Remove or tamper with the model, send bad input: never "normal"

```bash
ART=$(python -c "import json;r=json.load(open('models/registry.json'));print(r['models'][r['serving']]['artifact'])")
mv "$ART" "$ART.bak"

signal-replay --run r4009 --asset drone-01        # MODEL UNAVAILABLE; every status 'unavailable'; 0 incidents
uvicorn fleet_signal.service.app:app --port 8000 &
curl -i localhost:8000/readyz                     # HTTP 503 {"status":"unavailable","reason":"model artifact not found: ..."}
curl -i -X POST localhost:8000/score -H 'content-type: application/json' -d @window.json   # HTTP 503, decision null
kill %1

mv "$ART.bak" "$ART"                              # restore
```

Tampering is refused before the file is opened (pickles can run code on load):

```bash
cp "$ART" "$ART.bak" && printf 'x' >> "$ART"
uvicorn fleet_signal.service.app:app --port 8000 &
curl -s localhost:8000/readyz                     # 503 "artifact SHA-256 does not match the registry"
kill %1; mv "$ART.bak" "$ART"; git status --short models   # restored: nothing
```

Bad input is rejected or reported, never scored (server running with the model restored):

```bash
python -c "import json;d=json.load(open('window.json'));d['events'][-1]['mode']='MOVING';json.dump(d,open('bad_mode.json','w'))"
python -c "import json;d=json.load(open('window.json'));d['events'][-1]['battery_pct']=-50;json.dump(d,open('bad_batt.json','w'))"
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d @window.json     # ok, anomalous: fast:batt_res3, 2 events after the drain starts
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d @bad_mode.json   # 422 invalid_request (mode is case-sensitive)
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d @bad_batt.json   # degraded: battery_pct outside physical range
```

(Make `window.json` with: `python -c "import json;from fleet_signal.data.telemetry import load_telemetry as L;t=L(run_ids=['r4009']);d=t[(t.asset_id=='drone-01')&(t.seq<=699)].tail(650);print(json.dumps({'events':d.drop(columns=['timestamp_utc']).to_dict('records')}))" > window.json`.)

Also: fewer than 120 events gives `insufficient_data`, **a short window from the middle of a run gives `insufficient_data`** (it would cut off look-back features: never a silent `normal`), and a gap gives `degraded` (`pytest tests/test_service.py -v -k "unavailable or insufficient or degraded"`). Every case from the engineering review (unknown or mis-cased mode, impossible readings, negative or duplicated `seq`, mixed assets, oversized bodies, a tampered or unregistered artifact, a corrupt registry) is pinned in `tests/test_service_hardening.py`.

## 8. Beyond the bar, in one replay

```bash
signal-replay --run r4009 --asset drone-01 --shadow lof --show-truth
```

```
drone-01: 1198 events  statuses {'ok': 1079, 'insufficient_data': 119}
  drift check vs train: caution (score 1.46; batt_slope_l, temp_c, temp_slope_l)
  seq   698  incident #1 opened    score 1.307  evidence: fast:batt_res3, fast:batt_res10  [P3 0.32]
  seq   730  incident #1 closed    score 0.501
  SHADOW lof-f9f000d78f: 0 incident(s) (logged, not acted on)
```

Say: the drift check (here `caution`, because a fault moves the battery trend too), the severity (P3: the drain has just started and the battery is far from critical), and LOF running in shadow, logged but never acted on. Then open `docs/figures/generalisation_shifted_fleets.png`: the shipped system breaks on noisier sensors and aged batteries, LOF does not, and the drift monitor flags both shifts. That is why LOF runs in shadow.

## 9. The tests that pin it down

```bash
pytest -v tests/test_features.py tests/test_splits.py tests/test_eval.py tests/test_incidents.py tests/test_service.py tests/test_v2.py
pytest                          # 195 passed
mypy                            # no issues
```

| Brief asks for | Tests |
|---|---|
| preprocessing | `test_features.py`: hand-computed slopes, jumps, freeze counts; causality; no cross-run windows; gaps. `test_v2.py`: fast residuals cancel a steady trend, show a new one after k events, are causal |
| split logic | `test_splits.py`: disjoint seeds, runs and time; `test_v2.py`: v2 test seeds are new; generating splits separately gives identical runs |
| threshold logic | `test_eval.py`: budget + precision constraint, infeasible flag, run-once guard; `test_v2.py`: selection prefers candidates that meet the bar on validation |
| incident grouping | `test_incidents.py` (hand sequences) + streaming tracker == batch grouping on random sequences, with and without seq gaps |
| inference contract | `test_service.py`, `test_service_hardening.py`: unavailable / insufficient / degraded / ok, 422 / 413, SHA-256 checked before loading, replay == evaluation |
| v2 detectors and rules | `test_v2.py`: fast path flags extra drain and ignores the opposite direction, learns the manoeuvre covariate, is silent after a mode change; hybrid explains with its strongest part; expected-to-catch set is physical; misses count as infinite latency; ship rule v2 |

## Video outline (4–6 min)

| Time | Show | Say |
|---|---|---|
| 0:00–0:30 | README result table | the problem; v1 missed two criteria; v2 built and tested on fresh data |
| 0:30–1:15 | `signal-eval audit`, `git log` (d121e2a before eb83aa7), PLAN_V2 | why a new test set; frozen and pushed before it existed; test run once |
| 1:15–2:15 | §3 replay r4009 (shipped, then rule) | caught 1 event after onset vs never; then the incident closing early (honest) |
| 2:15–2:45 | §2 replay r4002 and r4058 | normal runs: 0 and 1 false incident; the cost of opening on 1 alert |
| 2:45–3:30 | `signal-eval show` + recall-by-fault figure | four detectors, same test; why it ships (latency); the rule also passes recall here |
| 3:30–4:30 | §5 r4037 FP, r4023 FN, r4000 late | speed-up false alarm; saturated freeze; the mode-change blind spot |
| 4:30–5:15 | §6 demo-threshold + sensitivity figure | trade-off, then back to frozen |
| 5:15–5:50 | §7 rename the model; bad input | HTTP 503, 422, `degraded`, never "normal" |
| 5:50–6:30 | §8 replay with shadow + generalisation figure | drift monitor, severity, shadow LOF; the shipped model's off-distribution weakness |
| 6:30–6:40 | `pytest` | 195 passed; CI green |

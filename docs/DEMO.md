# Acceptance demo runbook

Every item the brief lists, with the exact command and what it shows. Nothing here edits a model, regenerates data or touches `results/official/`. Outputs shown are from a rehearsal on the committed state.

Setup, once: `pip install --require-hashes -r requirements.lock && pip install --no-deps -e . && signal-data generate` (about 20 s; byte-identical data). **Do not retrain:** the exact evaluated model artifacts are committed under `models/`, and their SHA-256 matches `models/registry.json` and the official result files. (`signal-train` would verify and keep them anyway, and it refuses to overwrite an evaluated model.) Docker alternative: `docker build -t signal . && docker run --rm signal sh -c "signal-data generate && signal-eval show"` (runs as a non-root user).

## 1. The split, and proof held-out runs were not used

```bash
signal-data summary          # seed ranges, runs per split and scenario, variants
signal-eval audit
```

```
data version v1.0.0-9e903f9008
  train      seeds 1000-1039  (40 runs)
  validation seeds 2000-2039  (40 runs)
  test       seeds 3000-3059  (60 runs)
pairwise overlap of run ids: train/val 0, train/test 0, val/test 0
rule-64d39edc1d: no fit; limits hand-set from train-normal envelopes (results/validation/rule_envelopes_train.csv); threshold chosen on validation
stats-ab89662ec6: fitted on 40 runs, all in train: True; any test run used: False
lof-a588a4d11b: fitted on 40 runs, all in train: True; any test run used: False
runs referenced in the validation selection outputs: 32, all in validation: True; any test run: False
official test report: results/official/test_report_4530f6a108.json
detector_code_hash now: 422f446750
scoring-code commits since freeze (05fcfcb): 6
  6a0bbf8 service hardening: never 'normal' on bad input, verified artifacts, limits, audit log, Docke
  62e6158 repro + code: pinned environment, mypy in CI, true end-to-end test, Dockerfile, small fixes
  cb1df60 fix(service): never score a truncated mid-run window (was a silent-normal path)
  1cdbaaf feat(demo): acceptance runbook, audit/show/demo-threshold commands
  07aab1d docs: data card, model card, evaluation, retrospective, full README
  079b2d5 feat(service): scorer with explicit failure states, streaming incidents, API, replay
```

Then point at:

- **Git history.** The frozen registry was committed in `05fcfcb` *before* the official result `6683640`.
- **The guard.** Running `signal-eval test` again refuses (the official files already exist).
- **Disclosure, said out loud:** three test runs (r3000, r3032, r3037) were viewed in a data-sanity plot while building the generator, before any limit or threshold was set; nothing was chosen from them (EVALUATION §1). The gallery is now validation-only.
- **The leakage tests:** `pytest tests/test_splits.py tests/test_features.py -k "split or leak or ground_truth or causal or cross" -v`.

## 2. A normal held-out run and its false-incident count

```bash
signal-replay --run r3048 --show-truth        # shipped rule baseline
signal-replay --run r3048 --detector lof --show-truth
```

Rule: **3 false incidents**. Two are warm-up false alarms in the first minute of motion: rover seq 122 and quadruped seq 151, evidence `temp_rise` (error pattern 1). The third is on the drone at seq 973. LOF: **0**. For a clean example: `signal-replay --run r3002 --show-truth`.

## 3. A held-out fault: score, threshold crossing, incident open, latency

```bash
signal-replay --run r3006 --detector lof --asset drone-01 --show-truth
signal-replay --run r3006 --asset drone-01 --show-truth          # shipped rule, same run
```

```
model lof-a588a4d11b  threshold 2.2792  incident params {'open_n': 2, 'close_m': 5, 'cooldown_c': 30}
  seq   634  incident #1 opened    score 2.348  evidence: temp_slope_s, temp_std_m, link_unchanged
  seq   899  incident #1 closed    score 2.042
  seq   906  incident #1 reopened  score 3.409  evidence: temp_c, temp_slope_l, link_absdiff_s
GROUND TRUTH: drone-01: overheating / linear  seq 624-1200
    detected: incident #1 opened at seq 634 -> latency 10 events
```

The rule opens at seq 685: **latency 61**. Per-event scores and decisions are in `results/replay/r3006_<model>.jsonl` (open it to show the score crossing the threshold the event before the incident opens: N = 2).

## 4. Rule vs robust z vs ML on the same test data

```bash
signal-eval show
```

```
detector  model version          precision       recall           f1 fp_per_10min progressive_
rule      rule-64d39edc1d           0.810        0.844        0.827        0.124       71.000
stats     stats-ab89662ec6          0.792        0.378        0.511        0.052      534.500
lof       lof-a588a4d11b            0.970        0.889        0.928        0.021       16.000

LOF - rule recall +0.044 (CI -0.078, +0.163)
LOF - rule latency -24.0 events (CI -53.5, +3.5, n=20)
SHIP: rule - recall difference ... does not show ML is better; latency advantage not established
```

Then open `docs/figures/test_recall_by_fault_type.png` and EVALUATION.md §5: why the rule ships, and the gap in that rule (LOF's significant precision and false-alert advantage).

## 5. One false positive and one false negative, explained

**False positive: LOF, r3003 drone, alerts at seq 976–978, incident opens at 977.**

```bash
signal-replay --run r3003 --detector lof --show-truth
```

The incident on drone-01 has evidence `batt_slope_l, batt_slope_m, batt_slope_s` (z ≈ 33). The drone flipped charging → idle → returning → charging within 4 s at base. LOF judges features against the statistics of the *current* mode ("returning", where battery falls), but the trailing battery slopes still describe the charge that just happened. The same run also shows a true detection: the quadruped's real fault caught in 6 events.

**False negative: r3015 rover battery drain (every detector misses it).**

```bash
signal-replay --run r3015 --detector lof --asset rover-01 --show-truth
```

0 incidents. Open `docs/figures/test_r3015_rover-01_battery_drain_step.png`: the fault starts while the rover is charging at 60 %, and extra drain cuts the charge rate from about 3.9 to 1.5 %/min. No feature knows that charge rate depends on state of charge, and normal charging does drop that low near full, so a slow charge looks normal. *Next:* a charge-rate-vs-state-of-charge residual feature.

(Second FN, with a different cause: r3025, a speed freeze that started while the drone was charging. The rule and LOF missed it but robust z caught it at 153 events via `speed_unchanged`, showing two detector gaps: the rule's freeze checks only run while moving, and LOF clips its inputs at ±20. Run it with `signal-replay --run r3025 --detector stats --asset drone-01 --show-truth`.)

## 6. Change the threshold for the demo, then put it back

```bash
signal-eval demo-threshold --detector lof --threshold 1.5
```

```
DEMO ONLY (lof-a588a4d11b); official result untouched
  frozen threshold    2.279  precision=0.970  recall=0.889  fp_per_10min=0.021  n_false_incidents=2   progressive_latency_median=16.0
  demo   threshold    1.500  precision=0.709  recall=0.956  fp_per_10min=0.237  n_false_incidents=23  progressive_latency_median=9.5
```

Lower threshold: +3 faults caught and 6 events faster, but 21 more false incidents and precision falls below the 0.75 bar. The other direction: `signal-eval demo-threshold --detector rule --threshold 0.1` gives precision 0.98 but recall 0.76. Show the whole trade-off with `docs/figures/test_threshold_sensitivity.png` (post-hoc curves, frozen points marked).

"Put it back": the frozen threshold was never changed. The demo writes to `results/demo/` only, `git status results/official models/registry.json` is clean, and a replay without `--threshold` uses 2.279 again:

```bash
git status --short results/official models/registry.json     # nothing
signal-replay --run r3006 --detector lof --asset drone-01 | head -1   # threshold 2.2792
```

## 7. Remove or tamper with the model, send bad input: never "normal"

```bash
ART=$(python -c "import json;r=json.load(open('models/registry.json'));print(r['models'][r['serving']]['artifact'])")
mv "$ART" "$ART.bak"

signal-replay --run r3006 --asset drone-01        # MODEL UNAVAILABLE; every status 'unavailable'; 0 incidents
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
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d @window.json     # ok, anomalous
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d @bad_mode.json   # 422 invalid_request (mode is case-sensitive)
curl -s -X POST localhost:8000/score -H 'content-type: application/json' -d @bad_batt.json   # degraded: battery_pct outside physical range
```

(Make `window.json` with: `python -c "import json;from fleet_signal.data.telemetry import load_telemetry as L;t=L(run_ids=['r3006']);d=t[(t.asset_id=='drone-01')&(t.seq<=700)].tail(650);print(json.dumps({'events':d.drop(columns=['timestamp_utc']).to_dict('records')}))" > window.json`.)

Also: fewer than 120 events gives `insufficient_data`, **a short window from the middle of a run gives `insufficient_data`** (it would cut off look-back features; e.g. r3029 rover at seq 796: 650 events → `anomalous`, last 120 only → `insufficient_data`, never a silent `normal`), and a gap gives `degraded` (`pytest tests/test_service.py -v -k "unavailable or insufficient or degraded"`). Every case from the engineering review (unknown or mis-cased mode, impossible readings, negative or duplicated `seq`, mixed assets, oversized bodies, a tampered or unregistered artifact, a corrupt registry) is pinned in `tests/test_service_hardening.py`.

## 8. The tests that pin it down

```bash
pytest -v tests/test_features.py tests/test_splits.py tests/test_eval.py tests/test_incidents.py tests/test_service.py
pytest                          # 173 passed
mypy                            # no issues
```

| Brief asks for | Tests |
|---|---|
| preprocessing | `test_features.py`: hand-computed slopes, jumps, freeze counts; causality; no cross-run windows; gaps |
| split logic | `test_splits.py`: disjoint seeds, runs and time; composition; no scenario in run ids |
| threshold logic | `test_eval.py`: budget + precision constraint, infeasible flag, run-once guard |
| incident grouping | `test_incidents.py` (hand sequences) + `test_service.py::test_tracker_matches_batch_grouping_on_random_sequences` |
| inference contract | `test_service.py`: unavailable / insufficient / degraded / ok, HTTP 503, replay == evaluation |
| service hardening | `test_service_hardening.py`: 422 on malformed input, `degraded` on impossible readings, 413 on oversized bodies, SHA-256 checked before loading, registry-only artifacts, audit log |

## Video outline (4–6 min)

| Time | Show | Say |
|---|---|---|
| 0:00–0:30 | README result table | the problem, what was built, the one-line conclusion |
| 0:30–1:15 | `signal-eval audit`, splits.yaml, git log | split by seed and time, test run once, registry committed before the test |
| 1:15–2:15 | §3 replay r3006 (LOF then rule) | score crossing, incident opens, latency 10 vs 61; incident closes and re-opens |
| 2:15–2:45 | §2 replay r3048 | normal run: rule 3 false alarms (2 warm-up, 1 on the drone at seq 973), LOF 0 |
| 2:45–3:30 | `signal-eval show` + recall-by-fault figure | three detectors on the same test; why the rule ships; the ship-rule gap |
| 3:30–4:30 | §5 FP r3003 and FN r3015 | mode-transition false alarm; drain hidden by charging; what I'd change |
| 4:30–5:15 | §6 demo-threshold + sensitivity figure | trade-off, then back to frozen (nothing changed) |
| 5:15–5:50 | §7 rename the model; then `bad_mode.json` and `bad_batt.json` | HTTP 503 unavailable, never "normal"; restore; 422 and `degraded` on bad input |
| 5:50–6:00 | `pytest` | 173 passed; CI green |

# retrospective

## v2 - going after the two criteria v1 missed

v1 hit 6 of the 8 acceptance criteria. recall was one fault short (0.844) and latency was way off (71 events vs 3). i couldnt fix that on v1s test set because id already used it once and then studied every error in it. retuning against it would just be grading myself on the answer key. so v2 got a brand new test set (seeds 4000-4059), and the v2 models were frozen and pushed to github before that test data even existed. thats the part im proudest of - anyone can check the order in the git history.

**what worked.** i did the physics before writing code. at 1 Hz battery is super precise (0.01 noise) so an extra drain shows up within 2-3 events if you compare against the assets own recent trend. temperature and link are too noisy for that. so i built a "fast path" for sudden changes and paired it with the rule baseline. on the new test: precision 0.85, recall 0.98, 0.12 false alarms per 10 min, and 1 event median latency on the faults that are physically catchable in 3 events. battery drain went from 188 events (rule) to 2.

**what i had to be honest about.**

- latency is still a miss. over all progressive faults the median is 11 events, because slow overheating and link decline just cant be seen in 3 seconds at this noise. for v2 i defined a narrower "expected to catch" set from physics (before the v2 test, misses counted as infinite), and on that set its 1 event. but my own original plan said every fault type is expected to catch and that narrowing it later counts as a failure, not a redefinition. i only noticed that line while checking the final docs, after id already written "meets every criterion". so its 7 of 8, not 8 of 8.
- the rule baseline alone also passed the recall bar on the new test (0.93). so v2s real win is speed, not recall. the median paired gain is only 1 event, and most of the big gains are on battery and overheating.
- i changed a selection rule on validation. at first only systems containing LOF could ship, and that would have picked one that failed latency on validation. i dropped that restriction before the test existed and logged it, but it was still a call i made after seeing validation numbers.
- while rehearsing the demo i noticed an incident open 1 event after a drain started and then close 30 events later while the drain was still going. the fast path adapts to the new rate. 14 of 44 caught faults go quiet before they end. i wouldnt have found it without actually running the demo end to end.
- r4000: a drain started the exact second the asset changed mode, and the fast path is blind right after mode changes. caught 187 events late.
- i built the fast path knowing exactly how my generator drains batteries. thats the strongest designer bias in the whole project.

**what id do next.** compare against an expected value (drain for this load, temperature for this load) instead of the assets own recent trend. that fixes the mode-change blind spot and the incidents closing early. keep an incident open until the state is back to normal, not just the trend. and get real telemetry, because everything here is my simulator.

## v1

### what went wrong

**my ship rule asked the wrong question.** before the test i said the ML model ships only if its significantly better than the best baseline on recall or latency. on test LOF was better on every number - 0.97 vs 0.81 precision, 0.89 vs 0.84 recall, 6x fewer false alarms, 16 vs 71 events latency. its precision and false alarm wins were significant. but the rule only looked at recall and latency, and with 45 faults neither one cleared significance, so the rule baseline ships. i wrote that rule thinking about missed faults and forgot that an operator drowning in false alarms is also a failure. i kept the decision instead of rewriting the rule after seeing the answer, but the rule itself was badly specified.

**i fixed my selection rule only after it failed.** the plan said highest recall within 1.5 false alerts per 10 min. the first validation sweep showed that allows about 97 false incidents against 30 faults, so precision would be around 0.3. so i added precision >= 0.80. it happened on validation so its legit, but i should have caught it with basic arithmetic in the plan - the false alert budget and the precision bar are in different units and the conversion depends on how much normal time there is per fault.

**the model i planned lost badly.** i picked isolation forest in the plan for reasons that sounded right. it got recall 0.27. the faults i generated are mostly "one feature goes extreme" and isolation forest is weak at that in about 33 dimensions. i had the info to predict this (i designed the faults) but didnt connect it to how the algorithm actually isolates points.

**i wrote a confident wrong explanation into five documents.** in test run r3025 a speed freeze started while the drone was charging. i said the label was invalid ("a fault nobody can see, every detector missed it") without checking. the review showed robust z caught it, through `speed_unchanged` growing to 153 against a normal max of 14. the real lesson was about my detectors - the rule only checks frozen fields while moving, and LOF's input clipping flattens extreme counters. blaming the data was the easier story. i should have checked all three detectors per fault results before writing anything.

**i looked at test data before the freeze.** my generator sanity gallery drew one example of every fault variant from validation and test, and i looked at three test runs while checking the generator. i didnt choose anything from them, but "test was never touched before the freeze" wasnt true, and i only disclosed it after the review. the gallery should have been validation only from day one.

**the service could still say "normal" when it should alert - twice.** first, the scorer took any window of 120+ events. a short mid run window cuts off look back features ("seconds in this mode"), so the rule's battery check could never fire. on one test run 42 of 57 sampled decisions changed (anomalous became normal). my tests missed it because they always sent windows from the start of the run.

then the engineering review found more of the same. `mode: "MOVING"` instead of `"moving"` silently switched off every mode based rule, so a battery draining at 30 %/min came back ok / normal. battery -50, battery 500, speed 1e308 all scored normal. a negative seq got past my start of run check. a duplicate seq with temperature 999 flipped the decision because i was quietly keeping the last copy. i had fixed unknown asset types but never asked the same question about mode. the deeper mistake was that the scorer tried to repair bad input (sort it, dedupe it, take the first asset type) instead of refusing it. now theres a strict schema at the API (422) and the same checks inside the scorer (degraded), and it never repairs anything.

**the service would load any model file it was pointed at.** `joblib.load` ran before the SHA check, and pickles can run code when you load them - a harmless test pickle wrote a file. the SHA was there the whole time, i was just checking it after opening the file. now only registered files load and the hash is checked first.

**one fault often became several incidents.** LOF split 17 of its 40 detected test faults into multiple incidents (up to 6), which flatters incident precision. it was already visible on validation and i didnt measure it until the review.

**the API and the evaluation werent measuring the same thing.** evaluation counts incidents. `/score` returns one decision per event with no memory between calls. so the 0.12 false incidents per 10 min isnt what someone calling the API sees, and my tracker docstring said the service used it when it didnt. now the docs say which number applies to what.

**small process mistakes caught late:**

- `.gitignore`'s `data/` silently excluded `src/fleet_signal/data/`, and lint was skipping it too.
- the data version didnt change when generator code changed.
- fault variants were drawn at random instead of balanced.
- the rule detector could answer normal for an unknown asset type. found it while writing the model card, after the test.
- some numbers in my docs were wrong (event count, which false alarms were warm up, a threshold curve claim). i wrote them from memory or an old sweep instead of the saved files.
- running the demo setup would have rewritten the registry hashes and broken the link to the evaluated files. the evaluated artifacts are now committed and protected.
- my first multi-stage Dockerfile built the test image by default, because docker builds the last stage. caught it by checking `whoami` and which tools were in the image.

### bad assumptions

- **"unseen variants make the test harder."** they made it easier. runaway heating, accelerating drain and drift are more extreme than their validation versions, and every detector caught all 16. the test was hard because the wider ranges made *weaker* versions of variants validation had already seen. if i want to stress generalisation i need subtler unseen faults, not just different ones.
- **"3 events of latency is a tuning problem."** at 1 Hz with 0.15 °C sensor noise, a 2 °C/min drift is physically undetectable in 3 seconds. i flagged this in the plan as a risk but still half expected features to fix it. that bar needs a sequential test (CUSUM) or a label for when a fault actually becomes detectable.
- **"mode tells the detector what normal looks like."** it does, except during transitions. trailing windows still describe the previous mode - that gave LOF its worst false alarm. and long charging sessions make counters grow in ways training rarely showed.
- **"charging is a separate easy regime."** charging hid two faults. charge rate depends on state of charge and none of my features know that.
- **"the input will look like my data."** every service test i wrote sent clean windows from my own generator. a real client sends wrong case, wrong types, duplicates, retries. the inference boundary has to assume the input is wrong until proven otherwise.

### what id redesign

1. **the decision rule:** non-inferiority on recall plus a significant gain on precision **or** latency, declared up front, with the operators alarm load as a first class metric.
2. **the fault generator:** record both "fault starts" and "fault becomes physically detectable", so latency can be reported against both.
3. **the test design:** generalisation sets made of subtler faults (and a held out asset type), not just new fault shapes.
4. **features:** residuals against expected behaviour (temperature vs a lagged load model, charge rate vs state of charge, link vs distance) instead of raw slopes. every error pattern in EVALUATION.md is a case where the detector compared a signal to the wrong "normal".
5. **detectors:** an asset that just changed mode should be judged by per type stats until its windows sit inside the new mode.
6. **order of work:** write the model card's failure cases section *before* freezing - it found a fail safe bug. and get an independent review before the test, not after. the reviews found two more fail safe problems and an honesty problem i missed.
7. **incident grouping:** choose open, close and cooldown on per fault precision, and keep an incident open while the asset is still inside an unresolved fault, so one fault stays one incident.
8. **the service contract first:** write the request schema and the "what do we answer for bad input" table before the scorer, and test it with hostile input, not just my own data. same for the protocol - resending 650 events every second is fine for a demo and wasteful for 200 assets. id keep feature state per asset on the server.

### technical lessons

- the split isnt just train / validation / test. its also which code is allowed to *import* which data. making ground truth physically unreachable from feature code, and testing that, was cheaper than being careful.
- counterfactual twins (same seed with and without the fault) are the strongest test i wrote. they prove a fault cant leak backwards in time or across assets.
- resample runs, not events, for confidence intervals. with 45 faults a 0.04 recall difference is noise, and saying so is part of the result.
- a run once guard has to cover the whole path: verify artifacts, check nothing exists, *then* read test. committing the frozen state before the run is what makes "i didnt peek" checkable.
- a strong boring baseline is the real bar. the rule baseline, with limits set from train data only, hit 0.93 recall on validation. that made the ML result mean something.
- distance based anomaly detection on well standardised features fit this fault shape way better than tree isolation. pick the algorithm from the shape of the anomalies, not from familiarity.
- "never answers normal" is a property of the whole input space, not of the cases i listed. if theres a field the model branches on (mode, asset type), an unexpected value there is a silent normal waiting to happen.
- verify before you open. a hash checked after loading a pickle protects nothing.

### what my own checks missed

- **the ship rules blind spot.** every test i wrote checked the rule was *implemented* as declared (`test_registry_and_decision.py`). none asked if it was the right rule.
- **the unknown asset type "normal" fallback.** i tested every failure state i had listed (missing model, corrupt model, short history, gaps, NaN, mixed assets) but not "an input the model has no idea about".
- **the short window silent normal.** my service tests always sent windows starting at the beginning of a run, so look back features were never cut off. the review found it by sending a short window from the middle of a run.
- **unknown mode, impossible values, bad seq.** same blind spot again, one field over. i fixed asset type and didnt generalise the lesson to the other fields the model branches on.
- **my own explanation of r3025.** i checked that the rule and LOF missed it and assumed every detector did. a one line query over the three per fault result files would have shown otherwise.
- **fragmentation.** i counted incidents, not incidents per fault, so a metric that flatters splitting faults went unnoticed.
- **a wrong number in my own log.** i wrote "1,105 normal fleet minutes" in the process log from memory. the saved result says 969.5. caught it when writing EVALUATION.md from the files. numbers in docs should come from saved outputs, not my notes.

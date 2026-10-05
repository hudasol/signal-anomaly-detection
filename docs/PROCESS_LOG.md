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

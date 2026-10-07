"""`signal-replay`: stream a stored run through the detector, as the service would.

    signal-replay --run r3007                      shipped model, every asset
    signal-replay --run r3007 --detector lof       a specific frozen detector
    signal-replay --run r3007 --threshold 1.5      DEMO ONLY: try another threshold
    signal-replay --run r3007 --strict             score window-by-window via the service path
    signal-replay --run r3007 --show-truth         afterwards, print the labelled fault window

Every decision is written to results/replay/<run>_<model_version>.jsonl (a
shadow-replay log: what the detector said, when, and why). The detector never
sees ground truth; `--show-truth` loads it only after replay has finished, for
the operator's comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fleet_signal.data.config import REPO_ROOT
from fleet_signal.data.telemetry import load_telemetry
from fleet_signal.eval.protocol import EvalConfig
from fleet_signal.features.build import build_features
from fleet_signal.incidents.tracker import IncidentTracker
from fleet_signal.registry import REGISTRY_PATH, read_registry
from fleet_signal.service.scorer import HISTORY_BUFFER, Scorer, ScoreResult
from fleet_signal.service.validation import check_window

REPLAY_DIR = REPO_ROOT / "results" / "replay"


def replay_asset(
    scorer: Scorer, events: pd.DataFrame, strict: bool = False
) -> tuple[list[ScoreResult], list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (per-event results, lifecycle events, incidents) for one asset's run."""
    events = events.sort_values("seq").reset_index(drop=True)
    problem = check_window(events) if len(events) else "no events"
    if problem is not None and scorer.available:
        # The same checks the service applies: a run that fails them is not scored.
        results = [
            scorer.result(status="degraded", asset_id=str(a), seq=int(q), reason=problem)
            for a, q in zip(events["asset_id"], events["seq"], strict=True)
        ]
    elif strict:
        results = [
            scorer.score_events(events.iloc[max(0, i + 1 - HISTORY_BUFFER) : i + 1])
            for i in range(len(events))
        ]
    elif scorer.available:
        results = scorer.score_feature_rows(build_features(events, scorer.fcfg))
    else:
        results = [scorer.score_events(events.iloc[:1]) for _ in range(len(events))]

    lifecycle: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    if scorer.artifact is not None:
        tracker = IncidentTracker(scorer.artifact.params)
        for r, event_seq in zip(results, events["seq"], strict=True):
            alert = r.status == "ok" and r.decision == "anomalous"
            seq = int(r.seq) if r.seq is not None else int(event_seq)
            for ev in tracker.update(seq, alert, r.score):
                lifecycle.append({**ev, "asset_id": r.asset_id, "score": r.score,
                                  "evidence": r.evidence})  # fmt: skip
        if results and results[-1].seq is not None:
            lifecycle.extend(tracker.finish(int(results[-1].seq)))
        incidents = [i.as_dict() for i in tracker.incidents]
    return results, lifecycle, incidents


def _artifact_for(detector: str | None) -> Path | None:
    if detector is None:
        return None
    reg = read_registry(REGISTRY_PATH)
    if detector not in reg["models"]:
        raise SystemExit(f"{detector} not in registry; run signal-train")
    return REPO_ROOT / reg["models"][detector]["artifact"]


def _drift_monitor() -> Any:
    from fleet_signal.monitoring.drift import DriftMonitor

    ref = REPO_ROOT / "results" / "exceeds" / "drift_reference.json"
    return DriftMonitor.load(ref) if ref.exists() else None


def _severities(
    scorer: Scorer, feats: pd.DataFrame, lifecycle: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Severity (incidents/priority.py) for every opened / re-opened incident."""
    from fleet_signal.incidents.priority import severity

    art = scorer.artifact
    if art is None or scorer.threshold is None or not lifecycle:
        return {}
    feats = feats.reset_index(drop=True)
    scores = art.detector.score(feats)
    pos = {int(q): i for i, q in enumerate(feats["seq"])}
    return {
        int(ev["seq"]): severity(art.detector, scorer.threshold, feats, scores,
                                 pos[int(ev["seq"])]).as_dict()
        for ev in lifecycle if ev["event"] != "closed" and int(ev["seq"]) in pos
    }  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="signal-replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--run", required=True)
    ap.add_argument("--asset", default=None)
    ap.add_argument("--detector", default=None, help="rule | stats | lof (default: shipped)")
    ap.add_argument("--artifact", default=None, help="explicit artifact path")
    ap.add_argument("--threshold", type=float, default=None, help="DEMO ONLY override")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--show-truth", action="store_true")
    ap.add_argument(
        "--shadow",
        default=None,
        help="also run this registered detector in SHADOW mode: logged, never acted on",
    )
    args = ap.parse_args(argv)

    path = Path(args.artifact) if args.artifact else _artifact_for(args.detector)
    scorer = Scorer(path, threshold_override=args.threshold)
    tel = load_telemetry(run_ids=[args.run])
    if tel.empty:
        raise SystemExit(f"run {args.run} not found")
    assets = [args.asset] if args.asset else sorted(tel["asset_id"].unique())

    if not scorer.available:
        print(f"MODEL UNAVAILABLE: {scorer.error}")
    else:
        art = scorer.artifact
        if art is None:
            raise SystemExit("model unavailable")
        demo = "  (DEMO threshold override)" if args.threshold is not None else ""
        print(f"model {art.model_version}  threshold {scorer.threshold:.4f}{demo}  "
              f"incident params {art.incident_params}")  # fmt: skip

    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    tag = scorer.artifact.model_version if scorer.artifact else "unavailable"
    if args.threshold is not None:
        tag += f"_thr{args.threshold:g}"
    log_path = REPLAY_DIR / f"{args.run}_{tag}.jsonl"
    all_incidents: dict[str, list[dict[str, Any]]] = {}
    shadow = Scorer(_artifact_for(args.shadow)) if args.shadow else None
    monitor = _drift_monitor()
    with log_path.open("w") as log:
        for asset in assets:
            events = tel[tel["asset_id"] == asset]
            results, lifecycle, incidents = replay_asset(scorer, events, strict=args.strict)
            feats = build_features(events.sort_values("seq"), scorer.fcfg)
            sev = _severities(scorer, feats, lifecycle)
            for r in results:
                log.write(json.dumps(r.as_dict()) + "\n")
            for ev in lifecycle:
                log.write(json.dumps({"lifecycle": ev, "severity": sev.get(ev["seq"])},
                                     default=str) + "\n")  # fmt: skip
            statuses = pd.Series([r.status for r in results]).value_counts().to_dict()
            print(f"\n{asset}: {len(results)} events  statuses {statuses}")
            if monitor is not None:
                d = monitor.score(feats)
                top = ", ".join(f for f, _ in d["top"])
                print(f"  drift check vs train: {d['status']} (score {d['score']:.2f}; {top})")
            for ev in lifecycle:
                top = ", ".join(e["signal"] for e in ev.get("evidence") or [])
                score = f"score {ev['score']:.3f}" if ev.get("score") is not None else ""
                why = f"evidence: {top}" if top and ev["event"] != "closed" else ""
                s = sev.get(ev["seq"]) if ev["event"] != "closed" else None
                pri = f"  [{s['level']} {s['severity']:.2f}]" if s else ""
                print(
                    f"  seq {ev['seq']:>5}  incident #{ev['incident_id']} {ev['event']:<9}"
                    f" {score}  {why}{pri}"
                )
            if shadow is not None and shadow.artifact is not None:
                sres, _slc, sinc = replay_asset(shadow, events, strict=args.strict)
                for r in sres:
                    log.write(json.dumps({"shadow": r.as_dict()}) + "\n")
                opens = [i["open_seqs"][0] for i in sinc]
                print(
                    f"  SHADOW {shadow.artifact.model_version}: {len(sinc)} incident(s)"
                    f"{' opening at ' + str(opens) if opens else ''} (logged, not acted on)"
                )
            all_incidents[asset] = incidents
    n_inc = sum(len(v) for v in all_incidents.values())
    print(f"\n{n_inc} incident(s). Decisions logged to {log_path}")

    if args.show_truth:
        from fleet_signal.data.ground_truth import load_faults, load_runs  # evaluation only

        runs = load_runs()
        split = runs.loc[runs["run_id"] == args.run, "split"]
        faults = load_faults()
        grace = EvalConfig.load().grace
        f = faults[faults["run_id"] == args.run]
        print(f"\nGROUND TRUTH (shown after replay; split: {split.iloc[0] if len(split) else '?'})")
        if f.empty:
            print("  normal run: every incident above is a false incident")
        for _, row in f.iterrows():
            print(f"  {row['asset_id']}: {row['fault_type']} / {row['variant']}  "
                  f"seq {row['fault_start_seq']}-{row['fault_end_seq']}")  # fmt: skip
            if row["asset_id"] not in all_incidents:
                print(f"    ({row['asset_id']} was not replayed; drop --asset to include it)")
                print("    every incident above is on a non-faulted asset: false incident")
                continue
            for inc in all_incidents.get(row["asset_id"], []):
                opens = [o for o in inc["open_seqs"] if o >= row["fault_start_seq"]]
                if opens and opens[0] < row["fault_end_seq"] + grace:
                    print(f"    detected: incident #{inc['incident_id']} opened at seq {opens[0]}"
                          f" -> latency {opens[0] - row['fault_start_seq']} events")  # fmt: skip
                    break
            else:
                print("    not detected")


def cli(argv: list[str] | None = None) -> None:
    """Entry point: exits quietly when the output is piped into `head`."""
    import os
    import sys

    try:
        main(argv)
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)


if __name__ == "__main__":
    cli()

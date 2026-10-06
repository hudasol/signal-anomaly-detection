"""Evaluation protocol: metric definitions, applied identically to every detector.

Definitions (PLAN §7, configs/eval.yaml):

* fault event: one injected fault on one asset in one run.
* detected: an incident opens or re-opens on the faulted asset in
  [fault_start, fault_end + grace).
* recall: detected fault events / fault events.
* precision: incidents overlapping [fault_start, fault_end + grace) on the
  faulted asset / all incidents.
* false incident rate: incidents that overlap no fault, per 10 minutes of
  normal fleet time. Fleet time = scorable asset-seconds outside fault windows
  divided by the number of assets (3 assets for 1 s = 1 fleet-second).
* latency: events from fault_start to the first in-window open. The bar
  applies to progressive faults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from fleet_signal.data.config import REPO_ROOT
from fleet_signal.incidents.grouping import IncidentParams, group_alerts

DEFAULT_EVAL_CONFIG = REPO_ROOT / "configs" / "eval.yaml"


@dataclass(frozen=True)
class EvalConfig:
    grace: int
    budget: float
    min_precision: float
    bar: dict[str, float]
    progressive: tuple[str, ...]
    grid: dict[str, list[int]]
    n_candidates: int
    n_resamples: int
    bootstrap_seed: int

    @classmethod
    def load(cls, path: Path = DEFAULT_EVAL_CONFIG) -> EvalConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            grace=raw["grace_events"],
            budget=raw["false_alert_budget_per_10min"],
            min_precision=raw["min_precision"],
            bar=raw["bar"],
            progressive=tuple(raw["progressive_faults"]),
            grid=raw["incident_grid"],
            n_candidates=raw["threshold_candidates"],
            n_resamples=raw["bootstrap"]["n_resamples"],
            bootstrap_seed=raw["bootstrap"]["seed"],
        )

    def param_grid(self) -> list[IncidentParams]:
        return [
            IncidentParams(n, m, c)
            for n in self.grid["open_n"]
            for m in self.grid["close_m"]
            for c in self.grid["cooldown_c"]
        ]


@dataclass
class _AssetRun:
    run_id: str
    asset_id: str
    seq: np.ndarray
    score: np.ndarray
    fault: dict[str, Any] | None  # fault on this asset, if any


@dataclass
class EvalResult:
    summary: dict[str, Any]
    per_run: pd.DataFrame  # counts per run; bootstrap resamples these rows
    per_fault: pd.DataFrame
    incidents: pd.DataFrame
    window_confusion: dict[str, int]


class ScoredSet:
    """Scores for every event of a set of runs, with their ground truth, ready to evaluate.

    Built once per (detector, split); evaluating a threshold + incident params is then cheap,
    which is what makes the validation sweep affordable.
    """

    def __init__(
        self,
        scored: pd.DataFrame,
        faults: pd.DataFrame,
        run_ids: list[str],
        ecfg: EvalConfig,
        n_assets: int,
        seen_variants: set[tuple[str, str]] | None = None,
    ) -> None:
        self.ecfg = ecfg
        self.n_assets = n_assets
        self.run_ids = sorted(run_ids)
        self.seen = seen_variants
        scored = scored[scored["run_id"].isin(self.run_ids)]
        faults = faults[faults["run_id"].isin(self.run_ids)]
        self.faults = faults.reset_index(drop=True)
        by_asset: dict[tuple[str, str], dict[str, Any]] = {
            (str(r["run_id"]), str(r["asset_id"])): {str(k): v for k, v in r.items()}
            for r in faults.to_dict("records")
        }
        self.groups: list[_AssetRun] = []
        for (run_id, asset_id), g in scored.groupby(["run_id", "asset_id"], sort=True):
            g = g.sort_values("seq")
            self.groups.append(
                _AssetRun(
                    str(run_id),
                    str(asset_id),
                    g["seq"].to_numpy(),
                    g["score"].to_numpy(dtype=float),
                    by_asset.get((str(run_id), str(asset_id))),
                )
            )

    def scores(self) -> np.ndarray:
        """All finite scores, used to place threshold candidates."""
        allscores = np.concatenate([g.score for g in self.groups]) if self.groups else np.array([])
        return allscores[np.isfinite(allscores)]

    def evaluate(self, threshold: float, params: IncidentParams) -> EvalResult:
        grace = self.ecfg.grace
        per_run: dict[str, dict[str, float]] = {
            r: {"n_faults": 0, "n_detected": 0, "n_incidents": 0, "n_tp": 0, "n_fp": 0,
                "normal_asset_s": 0.0}
            for r in self.run_ids
        }  # fmt: skip
        fault_rows: list[dict[str, Any]] = []
        inc_rows: list[dict[str, Any]] = []
        tp_w = fp_w = fn_w = tn_w = 0

        for g in self.groups:
            finite = np.isfinite(g.score)
            alert = np.where(finite, g.score, -np.inf) >= threshold
            incs = group_alerts(alert, params)
            f = g.fault
            if f is not None:
                fs, fe = f["fault_start_seq"], f["fault_end_seq"]
                in_window = (g.seq >= fs) & (g.seq < fe + grace)
                label = (g.seq >= fs) & (g.seq < fe)
            else:
                in_window = np.zeros(len(g.seq), dtype=bool)
                label = in_window
            run = per_run[g.run_id]
            run["normal_asset_s"] += float((finite & ~in_window).sum())
            # window-level confusion (secondary diagnostic): scorable events only
            tp_w += int((alert & label & finite).sum())
            fp_w += int((alert & ~label & finite).sum())
            fn_w += int((~alert & label & finite).sum())
            tn_w += int((~alert & ~label & finite).sum())

            first_open: int | None = None
            for inc in incs:
                start, last = int(g.seq[inc.start]), int(g.seq[inc.last_alert])
                opens = [int(g.seq[i]) for i in inc.opens]
                tp = f is not None and start < fe + grace and last >= fs
                run["n_incidents"] += 1
                run["n_tp" if tp else "n_fp"] += 1
                if f is not None:
                    hits = [o for o in opens if fs <= o < fe + grace]
                    if hits and (first_open is None or min(hits) < first_open):
                        first_open = min(hits)
                inc_rows.append(
                    {
                        "run_id": g.run_id,
                        "asset_id": g.asset_id,
                        "start_seq": start,
                        "open_seq": opens[0],
                        "open_seqs": opens,
                        "last_alert_seq": last,
                        "close_seq": int(g.seq[inc.close]),
                        "peak_score": float(np.nanmax(g.score[inc.start : inc.last_alert + 1])),
                        "true_positive": bool(tp),
                    }
                )
            if f is not None:
                run["n_faults"] += 1
                run["n_detected"] += int(first_open is not None)
                fault_rows.append(
                    {
                        "run_id": g.run_id,
                        "asset_id": g.asset_id,
                        "fault_type": f["fault_type"],
                        "variant": f["variant"],
                        "progressive": bool(f["progressive"]),
                        "seen_variant": (
                            None
                            if self.seen is None
                            else (f["fault_type"], f["variant"]) in self.seen
                        ),
                        "fault_start_seq": fs,
                        "fault_end_seq": fe,
                        "detected": first_open is not None,
                        "open_seq": first_open,
                        "latency_events": None if first_open is None else first_open - fs,
                    }
                )

        per_run_df = pd.DataFrame.from_dict(per_run, orient="index").rename_axis("run_id")
        per_fault = pd.DataFrame(fault_rows)
        summary = summarise(per_run_df, per_fault, self.n_assets, self.ecfg)
        summary.update({"threshold": float(threshold), **params.as_dict()})
        confusion = {"tp": tp_w, "fp": fp_w, "fn": fn_w, "tn": tn_w}
        return EvalResult(
            summary, per_run_df.reset_index(), per_fault, pd.DataFrame(inc_rows), confusion
        )


def rates(per_run: pd.DataFrame, n_assets: int) -> dict[str, float]:
    """Headline rates from per-run counts (also used on bootstrap resamples)."""
    n_inc = per_run["n_incidents"].sum()
    n_faults = per_run["n_faults"].sum()
    precision = per_run["n_tp"].sum() / n_inc if n_inc else float("nan")
    recall = per_run["n_detected"].sum() / n_faults if n_faults else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    fleet_10min = per_run["normal_asset_s"].sum() / n_assets / 600.0
    fp_rate = per_run["n_fp"].sum() / fleet_10min if fleet_10min else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "fp_per_10min": fp_rate}


def summarise(
    per_run: pd.DataFrame, per_fault: pd.DataFrame, n_assets: int, ecfg: EvalConfig
) -> dict[str, Any]:
    out: dict[str, Any] = rates(per_run, n_assets)
    out["n_faults"] = int(per_run["n_faults"].sum())
    out["n_detected"] = int(per_run["n_detected"].sum())
    out["n_incidents"] = int(per_run["n_incidents"].sum())
    out["n_false_incidents"] = int(per_run["n_fp"].sum())
    out["normal_fleet_minutes"] = float(per_run["normal_asset_s"].sum() / n_assets / 60.0)
    # Precision counted per fault (a fault split into many incidents counts once).
    det, fp = out["n_detected"], out["n_false_incidents"]
    out["per_fault_precision"] = float(det / (det + fp)) if det + fp else float("nan")
    lat = per_fault.loc[
        per_fault["detected"] & per_fault["fault_type"].isin(ecfg.progressive), "latency_events"
    ] if len(per_fault) else pd.Series(dtype=float)  # fmt: skip
    out["progressive_latency_median"] = float(lat.median()) if len(lat) else float("nan")
    out["progressive_latency_p90"] = float(lat.quantile(0.9)) if len(lat) else float("nan")
    return out


def breakdown(per_fault: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Recall and latency per fault type (or type + variant, or seen/unseen)."""
    if per_fault.empty:
        return pd.DataFrame()
    g = per_fault.groupby(by, dropna=False)
    out = g.agg(
        n=("detected", "size"),
        detected=("detected", "sum"),
        latency_median=("latency_events", lambda s: _quantile(s, 0.5)),
        latency_p90=("latency_events", lambda s: _quantile(s, 0.9)),
    )
    out["recall"] = out["detected"] / out["n"]
    return out.reset_index()


def _quantile(s: pd.Series, q: float) -> float:
    """Quantile of the finite values; NaN (without a warning) when nothing was detected."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.quantile(q)) if len(v) else float("nan")


def meets_bar(summary: dict[str, Any], ecfg: EvalConfig) -> dict[str, bool]:
    b = ecfg.bar
    return {
        "precision": summary["precision"] >= b["precision"],
        "recall": summary["recall"] >= b["recall"],
        "false_alerts": summary["fp_per_10min"] <= b["false_alerts_per_10min"],
        "latency": summary["progressive_latency_median"] <= b["median_latency_events"],
    }

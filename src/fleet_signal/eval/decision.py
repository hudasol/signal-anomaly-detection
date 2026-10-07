"""The ship decision, exactly as pre-declared in PLAN §7.3.

The ML detector ships only if, on test, either
  (a) its fault-event recall beats the best baseline and the 95% paired
      bootstrap CI of the difference excludes zero, or
  (b) its median progressive latency is lower than the best baseline's with a
      paired CI that excludes zero, AND it loses no recall (point estimate).
Otherwise the best baseline ships. The "best baseline" is the baseline with the
higher test recall (ties: higher precision).
"""

from __future__ import annotations

from typing import Any


def best_baseline(summaries: dict[str, dict[str, Any]], baselines: tuple[str, ...]) -> str:
    present = [b for b in baselines if b in summaries]
    return max(present, key=lambda n: (summaries[n]["recall"], summaries[n]["precision"]))


def ship_decision(
    ml_name: str,
    baseline_name: str,
    recall_diff: dict[str, float],
    latency_diff: dict[str, float],
) -> dict[str, Any]:
    """recall_diff: ML minus baseline. latency_diff: ML minus baseline (negative = ML faster)."""
    recall_win = recall_diff["ci_low"] > 0
    latency_win = latency_diff.get("n_paired", 0) > 0 and latency_diff["ci_high"] < 0
    no_recall_loss = recall_diff["diff"] >= 0
    if recall_win:
        ship, reason = ml_name, "ML recall is higher and the paired 95% CI excludes zero"
    elif latency_win and no_recall_loss:
        ship, reason = ml_name, "ML is faster (paired CI excludes zero) with no recall loss"
    else:
        parts = []
        if not recall_win:
            parts.append(
                f"recall difference {recall_diff['diff']:+.3f} "
                f"(CI {recall_diff['ci_low']:+.3f} to {recall_diff['ci_high']:+.3f}) "
                "does not show ML is better"
            )
        if latency_win and not no_recall_loss:
            parts.append("ML is faster but loses recall, which the rule does not allow")
        elif not latency_win:
            parts.append("latency advantage not established")
        ship, reason = baseline_name, "; ".join(parts)
    return {
        "ship": ship,
        "ml": ml_name,
        "best_baseline": baseline_name,
        "reason": reason,
        "criteria": {
            "recall_ci_excludes_zero_in_ml_favour": bool(recall_win),
            "latency_ci_excludes_zero_in_ml_favour": bool(latency_win),
            "ml_no_recall_loss": bool(no_recall_loss),
        },
    }


# ---------------------------------------------------------------- v2 (PLAN_V2 §6)

RECALL_MARGIN = 0.05  # non-inferiority margin on recall


def ship_decision_v2(
    candidate: str,
    baseline_name: str,
    recall_diff: dict[str, float],
    precision_diff: dict[str, float],
    latency_diff: dict[str, float],
    margin: float = RECALL_MARGIN,
) -> dict[str, Any]:
    """Declared before the v2 test set existed. Differences are candidate minus baseline.

    The candidate ships if it is NOT WORSE on recall (lower end of the paired 95% CI
    above -margin) AND it is significantly better on precision (CI above zero) OR on
    progressive latency (paired CI below zero). Otherwise the baseline ships.
    `baseline_name` is the baseline with the higher VALIDATION recall.
    """
    non_inferior = recall_diff["ci_low"] > -margin
    precision_win = precision_diff["ci_low"] > 0
    latency_win = latency_diff.get("n_paired", 0) > 0 and latency_diff["ci_high"] < 0
    ships = non_inferior and (precision_win or latency_win)
    if ships:
        wins = [w for w, ok in (("precision", precision_win), ("latency", latency_win)) if ok]
        reason = (
            f"recall not worse (CI low {recall_diff['ci_low']:+.3f} > -{margin}) and "
            f"significantly better on {' and '.join(wins)}"
        )
    else:
        parts = []
        if not non_inferior:
            parts.append(f"recall may be worse (CI low {recall_diff['ci_low']:+.3f})")
        if not (precision_win or latency_win):
            parts.append("no significant precision or latency gain")
        reason = "; ".join(parts)
    return {
        "ship": candidate if ships else baseline_name,
        "ml": candidate,
        "best_baseline": baseline_name,
        "reason": reason,
        "rule": "v2: recall non-inferior (0.05) AND precision or latency significantly better",
        "criteria": {
            "recall_non_inferior": bool(non_inferior),
            "precision_ci_excludes_zero_in_candidate_favour": bool(precision_win),
            "latency_ci_excludes_zero_in_candidate_favour": bool(latency_win),
        },
    }

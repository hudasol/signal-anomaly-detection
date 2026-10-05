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

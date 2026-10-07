"""Evaluation plots. Every plot is drawn from saved CSV/JSON outputs, not live objects."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# Fixed categorical order (validated: dataviz validate_palette.js, light mode).
DETECTOR_COLORS = {"rule": "#2a78d6", "stats": "#eb6834", "lof": "#1baf7a",
                   "hybrid_rule_fast": "#7a4fc9"}  # fmt: skip
DETECTOR_MARKERS = {"rule": "o", "stats": "s", "lof": "^", "hybrid_rule_fast": "D"}
# Where each operating-point label sits (points from the marker), so labels never overlap.
LABEL_OFFSETS = {"hybrid_rule_fast": (40, -4), "rule": (40, -30), "lof": (40, -58),
                 "stats": (40, -20)}  # fmt: skip
DETECTOR_LABELS = {
    "rule": "Rule baseline",
    "stats": "Robust z baseline",
    "lof": "LOF (ML)",
    "hybrid_rule_fast": "Rule + fast path (shipped v2)",
}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def plot_threshold_sensitivity(
    results_dir: Path,
    out_path: Path,
    x_max: float = 4.0,
    report_name: str = "validation_report.json",
    title: str = "Validation threshold sweep (markers = selected operating points)",
) -> Path:
    """Recall vs false-alert rate, and precision vs recall, with the selected points marked."""
    report = json.loads((results_dir / report_name).read_text())
    budget = report["false_alert_budget_per_10min"]
    min_precision = report.get("min_precision", 0.0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for i, (name, det) in enumerate(report["detectors"].items()):
        curve = pd.read_csv(results_dir / f"{name}_threshold_curve.csv").sort_values("threshold")
        color = DETECTOR_COLORS.get(name, INK_2)
        marker = DETECTOR_MARKERS.get(name, "o")
        label = DETECTOR_LABELS.get(name, name)
        sel = det["selected"]
        ax1.plot(curve["fp_per_10min"], curve["recall"], color=color, linewidth=2, label=label)
        ax1.scatter(
            [sel["fp_per_10min"]], [sel["recall"]], s=70, color=color, marker=marker,
            edgecolor=SURFACE, linewidth=2, zorder=5,
        )  # fmt: skip
        ax1.annotate(
            f"{label}\nrecall {sel['recall']:.2f}",
            (sel["fp_per_10min"], sel["recall"]),
            textcoords="offset points", xytext=LABEL_OFFSETS.get(name, (14, -10 - 30 * i)),
            fontsize=8, color=INK_2,
            arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8},
        )  # fmt: skip
        pr = curve.dropna(subset=["precision"])
        ax2.plot(pr["recall"], pr["precision"], color=color, linewidth=2, label=label)
        ax2.scatter(
            [sel["recall"]], [sel["precision"]], s=70, color=color, marker=marker,
            edgecolor=SURFACE, linewidth=2, zorder=5,
        )  # fmt: skip
    ax1.axvline(budget, color=MUTED, linestyle="--", linewidth=1)
    ax1.text(budget, 0.02, f" budget {budget}/10 min", color=MUTED, fontsize=8)
    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("False incidents per 10 min of normal fleet time", color=INK_2, fontsize=9)
    ax1.set_ylabel("Fault-event recall", color=INK_2, fontsize=9)
    ax1.set_title("Recall vs false-alert rate", color=INK, fontsize=10, loc="left")
    ax2.axhline(min_precision, color=MUTED, linestyle="--", linewidth=1)
    ax2.text(0.01, min_precision + 0.01, f"min precision {min_precision}", color=MUTED, fontsize=8)
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Fault-event recall", color=INK_2, fontsize=9)
    ax2.set_ylabel("Incident precision", color=INK_2, fontsize=9)
    ax2.set_title("Precision vs recall", color=INK, fontsize=10, loc="left")
    for ax in (ax1, ax2):
        _style(ax)
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def plot_recall_by_fault(
    results_dir: Path,
    out_path: Path,
    report_name: str = "validation_report.json",
    title: str = "Recall by fault type at the selected thresholds (validation)",
) -> Path:
    """Recall per fault type, one bar per detector, at the selected operating points."""
    report = json.loads((results_dir / report_name).read_text())
    frames = []
    for name, det in report["detectors"].items():
        df = pd.DataFrame(det["by_fault_type"])
        df["detector"] = name
        frames.append(df)
    data = pd.concat(frames)
    types = sorted(data["fault_type"].unique())
    names = list(report["detectors"])
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    for i, name in enumerate(names):
        d = data[data["detector"] == name].set_index("fault_type").reindex(types)
        xs = [j + (i - (len(names) - 1) / 2) * width for j in range(len(types))]
        ax.bar(
            xs, d["recall"], width=width * 0.92, color=DETECTOR_COLORS.get(name, INK_2),
            label=DETECTOR_LABELS.get(name, name),
        )  # fmt: skip
        for x, v in zip(xs, d["recall"], strict=True):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([t.replace("_", " ") for t in types], fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Fault-event recall", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=len(names))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=SURFACE)
    plt.close(fig)
    return out_path

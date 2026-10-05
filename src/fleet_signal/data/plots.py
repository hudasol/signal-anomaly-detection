"""Sanity plots of generated runs, read back from the saved dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
FAULT = "#d03b3b"
MODE_COLORS = {
    "idle": "#c3c2b7",
    "moving": "#1baf7a",
    "returning": "#eda100",
    "charging": "#4a3aa7",
}

PANELS: list[tuple[str, str]] = [
    ("temperature_c", "Temperature (°C)"),
    ("battery_pct", "Battery (%)"),
    ("link_quality_pct", "Link quality (%)"),
    ("speed_mps", "Reported speed (m/s)"),
    ("implied_speed_mps", "Speed implied by position (m/s)"),
]


def plot_asset_run(
    tel: pd.DataFrame,
    run_id: str,
    asset_id: str,
    out_path: Path,
    fault: pd.Series | None = None,
    title_note: str = "",
) -> Path:
    df = tel[(tel["run_id"] == run_id) & (tel["asset_id"] == asset_id)].sort_values("seq")
    df = df.assign(
        implied_speed_mps=np.hypot(df["x_m"].diff(), df["y_m"].diff()) / df["seq"].diff()
    )
    t = df["seq"].to_numpy()

    fig, axes = plt.subplots(
        len(PANELS) + 1,
        1,
        figsize=(11, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3] * len(PANELS) + [0.6]},
        facecolor=SURFACE,
    )
    for ax, (col, label) in zip(axes[:-1], PANELS, strict=True):
        ax.set_facecolor(SURFACE)
        ax.plot(t, df[col].to_numpy(), color=SERIES, linewidth=1.2)
        ax.set_ylabel(label, color=INK_2, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        if fault is not None:
            ax.axvspan(fault["fault_start_seq"], fault["fault_end_seq"], color=FAULT, alpha=0.12)
            ax.axvline(fault["fault_start_seq"], color=FAULT, linewidth=1.0)

    strip = axes[-1]
    for mode, color in MODE_COLORS.items():
        mask = (df["mode"] == mode).to_numpy()
        strip.fill_between(t, 0, 1, where=mask, color=color, step="mid", linewidth=0)
    strip.set_yticks([])
    strip.set_ylabel("Mode", color=INK_2, fontsize=9, rotation=0, ha="right", va="center")
    strip.set_xlabel("Seconds since run start", color=INK_2, fontsize=9)
    strip.tick_params(colors=MUTED, labelsize=8)

    handles = [Patch(color=c, label=m) for m, c in MODE_COLORS.items()]
    if fault is not None:
        handles.append(Patch(color=FAULT, alpha=0.25, label="fault window (ground truth)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8)

    scenario = (
        f"{fault['fault_type']} / {fault['variant']}" if fault is not None else "no fault on asset"
    )
    fig.suptitle(
        f"{run_id} · {asset_id} · {scenario}{title_note}", color=INK, fontsize=11, x=0.01, ha="left"
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out_path

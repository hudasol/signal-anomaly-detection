"""Generate the full dataset: telemetry plus separately stored ground truth.

Output layout (one directory per data version):

    data/<version>/
        telemetry.parquet          features may be built from this only
        manifest.json              provenance: version, config hash, counts
        ground_truth/
            runs.parquet           run -> split, scenario   (evaluation only)
            faults.parquet         fault windows             (evaluation only)
            episodes.parquet       difficult-normal episodes (evaluation only)
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fleet_signal.data.config import DEFAULT_DATA_DIR, GenerationConfig
from fleet_signal.data.faults import Fault, sample_fault
from fleet_signal.data.sim import AssetSim
from fleet_signal.data.splits import PlannedRun, build_run_plan

TELEMETRY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "asset_id",
    "asset_type",
    "seq",
    "timestamp_utc",
    "x_m",
    "y_m",
    "z_m",
    "speed_mps",
    "heading_deg",
    "battery_pct",
    "temperature_c",
    "link_quality_pct",
    "mode",
)


@dataclass
class RunResult:
    telemetry: pd.DataFrame
    fault: dict[str, Any] | None
    episodes: list[dict[str, Any]]


def _rng_streams(seed: int) -> dict[str, np.random.Generator]:
    """Independent streams per concern, so a fault can never perturb normal behaviour."""
    run_ss, fault_ss, *asset_ss = np.random.SeedSequence(seed).spawn(5)
    return {
        "run": np.random.default_rng(run_ss),
        "fault": np.random.default_rng(fault_ss),
        **{f"asset{i}": np.random.default_rng(s) for i, s in enumerate(asset_ss)},
    }


def generate_run(cfg: GenerationConfig, run: PlannedRun) -> RunResult:
    d = cfg.data
    n_ticks = cfg.n_ticks
    streams = _rng_streams(run.seed)
    ambient = float(streams["run"].uniform(*d["ambient_c"]))
    t0 = pd.Timestamp(d["start_epoch_utc"]) + pd.Timedelta(seconds=run.seed * d["run_spacing_s"])
    tick = pd.Timedelta(seconds=1.0 / d["tick_hz"])

    fault: Fault | None = None
    if run.scenario != "normal":
        split_spec = cfg.splits["splits"][run.split]
        fault = sample_fault(
            run.scenario,
            run.variant,  # type: ignore[arg-type]
            run.fault_asset_index,  # type: ignore[arg-type]
            cfg,
            split_spec["fault_ranges"],
            streams["fault"],
        )

    frames: list[pd.DataFrame] = []
    episodes: list[dict[str, Any]] = []
    for i, asset in enumerate(d["fleet"]):
        sim = AssetSim(
            asset_id=asset["asset_id"],
            asset_type=asset["asset_type"],
            base_xy=tuple(asset["base_xy"]),
            profile=d["profiles"][asset["asset_type"]],
            mission=d["mission"],
            link_cfg=d["link"],
            episode_cfg=d["episodes"],
            rng=streams[f"asset{i}"],
            n_ticks=n_ticks,
            ambient_c=ambient,
            packet_loss_prob=d["packet_loss_prob"],
            fault=fault if fault is not None and fault.asset_index == i else None,
        )
        rows: list[dict[str, Any]] = []
        for t in range(n_ticks):
            event = sim.step(t)
            if event is not None:
                event["seq"] = t
                rows.append(event)
        df = pd.DataFrame(rows)
        df["timestamp_utc"] = t0 + df["seq"] * tick
        df["run_id"] = run.run_id
        df["asset_id"] = asset["asset_id"]
        df["asset_type"] = asset["asset_type"]
        frames.append(df[list(TELEMETRY_COLUMNS)])
        for ep in sorted(sim.episodes + sim.mode_episodes(), key=lambda e: (e.start, e.kind)):
            end = min(ep.end, n_ticks)
            episodes.append(
                {
                    "run_id": run.run_id,
                    "asset_id": asset["asset_id"],
                    "episode": ep.kind,
                    "start_utc": t0 + ep.start * tick,
                    "end_utc": t0 + end * tick,
                    "start_seq": ep.start,
                    "end_seq": end,
                }
            )

    fault_row: dict[str, Any] | None = None
    if fault is not None:
        if fault.start is None:
            raise RuntimeError(f"{run.run_id}: fault never started")
        fault_row = {
            "run_id": run.run_id,
            "asset_id": d["fleet"][fault.asset_index]["asset_id"],
            "fault_type": fault.fault_type,
            "variant": fault.variant,
            "progressive": fault.progressive,
            "fault_start_utc": t0 + fault.start * tick,
            "fault_end_utc": t0 + fault.end * tick,  # type: ignore[operator]
            "fault_start_seq": fault.start,
            "fault_end_seq": fault.end,
            "params_json": json.dumps(fault.public_params(), sort_keys=True),
        }
    return RunResult(
        telemetry=pd.concat(frames, ignore_index=True), fault=fault_row, episodes=episodes
    )


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate_dataset(
    cfg: GenerationConfig,
    out_dir: Path = DEFAULT_DATA_DIR,
    runs: list[PlannedRun] | None = None,
) -> Path:
    """Generate every planned run and write telemetry and ground truth separately."""
    plan = runs if runs is not None else build_run_plan(cfg)
    version_dir = Path(out_dir) / cfg.version
    gt_dir = version_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    telemetry: list[pd.DataFrame] = []
    faults: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for run in plan:
        result = generate_run(cfg, run)
        telemetry.append(result.telemetry)
        if result.fault is not None:
            faults.append(result.fault)
        episodes.extend(result.episodes)

    tel = pd.concat(telemetry, ignore_index=True)
    tel.to_parquet(version_dir / "telemetry.parquet", index=False)

    runs_df = pd.DataFrame(
        [
            {"run_id": r.run_id, "seed": r.seed, "split": r.split, "scenario": r.scenario}
            for r in plan
        ]
    )
    runs_df.to_parquet(gt_dir / "runs.parquet", index=False)
    pd.DataFrame(faults, columns=_FAULT_COLUMNS).to_parquet(gt_dir / "faults.parquet", index=False)
    pd.DataFrame(episodes, columns=_EPISODE_COLUMNS).to_parquet(
        gt_dir / "episodes.parquet", index=False
    )

    manifest = {
        "data_version": cfg.version,
        "generator_version": cfg.data["generator_version"],
        "git_sha": _git_sha(),
        "n_runs": len(plan),
        "n_events": len(tel),
        "runs_per_split": runs_df.groupby("split").size().to_dict(),
        "seeds_per_split": {
            split: [int(g["seed"].min()), int(g["seed"].max())]
            for split, g in runs_df.groupby("split")
        },
        "telemetry_columns": list(TELEMETRY_COLUMNS),
        "config": {"data": cfg.data, "splits": cfg.splits},
    }
    (version_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (Path(out_dir) / "CURRENT").write_text(cfg.version + "\n")
    return version_dir


_FAULT_COLUMNS = [
    "run_id",
    "asset_id",
    "fault_type",
    "variant",
    "progressive",
    "fault_start_utc",
    "fault_end_utc",
    "fault_start_seq",
    "fault_end_seq",
    "params_json",
]
_EPISODE_COLUMNS = ["run_id", "asset_id", "episode", "start_utc", "end_utc", "start_seq", "end_seq"]

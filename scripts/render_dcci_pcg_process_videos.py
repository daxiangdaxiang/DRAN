#!/usr/bin/env python3
"""Render staged DRAN videos with D-CCI-PCG process initialization frames."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DATASETS = [
    ("parking-garage", "data/parking-garage.g2o", "3d"),
    ("sphere", "data/sphere2500.g2o", "3d"),
    ("torus", "data/torus3D.g2o", "3d"),
    ("CSAIL", "data/CSAIL.g2o", "xy"),
    ("inter", "data/input_INTEL_g2o.g2o", "xy"),
    ("manhattan", "data/input_M3500_g2o.g2o", "xy"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dcci-process-root",
        type=Path,
        default=Path("results/dcci_pcg_process_20260620"),
        help="root containing <dataset>/initialization_estimates with D-CCI-PCG trace frames",
    )
    parser.add_argument(
        "--dran-run-root",
        type=Path,
        default=Path("results/phase41_video_dran_six_dcci_pcg_init_20260620/runs"),
        help="root containing <dataset>/iteration_estimates and iteration_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_phase41_dran_nature/figures/videos_dcci_pcg_process"),
    )
    parser.add_argument("--dataset", action="append", help="dataset name to render; repeatable")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--num-robots", type=int, default=5)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def check_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {description}: {path}")


def validate_dcci_process_trace(init_dir: Path) -> dict[str, object]:
    """Validate that an initialization directory contains sampled PCG frames."""
    check_dir(init_dir, "D-CCI-PCG initialization trace directory")
    metadata_path = init_dir / "init_metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"missing D-CCI-PCG process metadata: {metadata_path}")

    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError(
            f"D-CCI-PCG process trace requires at least two metadata rows: "
            f"{metadata_path}")

    missing = []
    for row in rows:
        try:
            frame_index = int(row["frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"D-CCI-PCG process metadata row has invalid frame: {row}") from exc
        pose_path = init_dir / f"init_{frame_index:04d}.txt"
        if not pose_path.is_file():
            missing.append(pose_path)
    if missing:
        raise FileNotFoundError(
            "D-CCI-PCG process metadata references missing pose frame: "
            f"{missing[0]}")

    pcg_iters = [row.get("pcg_iter", "") for row in rows]
    return {
        "metadata_rows": len(rows),
        "pose_frames": len(rows),
        "first_pcg_iter": pcg_iters[0],
        "last_pcg_iter": pcg_iters[-1],
        "metadata_path": str(metadata_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = set(args.dataset or [])
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "dpgo_matplotlib_cache"))

    for name, graph, projection in DATASETS:
        if selected and name not in selected:
            continue
        init_dir = args.dcci_process_root / name / "initialization_estimates"
        iter_dir = args.dran_run_root / name / "iteration_estimates"
        iter_summary = args.dran_run_root / name / "iteration_summary.csv"
        trace_summary = validate_dcci_process_trace(init_dir)
        check_dir(iter_dir, "DRAN iteration estimate directory")
        if not iter_summary.is_file():
            raise FileNotFoundError(f"missing DRAN iteration summary: {iter_summary}")

        output = output_dir / f"dran_{name}_staged.mp4"
        cmd = [
            args.python,
            "scripts/animate_pose_graph_iterations.py",
            "--graph",
            graph,
            "--initialization-dir",
            str(init_dir),
            "--estimate-dir",
            str(iter_dir),
            "--iteration-summary",
            str(iter_summary),
            "--title",
            f"DRAN {name}",
            "--method-label",
            "DRAN",
            "--initialization-stage-label",
            "D-CCI-PCG initialization",
            "--optimization-stage-label",
            "DRAN optimization",
            "--projection",
            projection,
            "--output",
            str(output),
            "--fps",
            str(args.fps),
            "--dpi",
            str(args.dpi),
            "--num-robots",
            str(args.num_robots),
        ]
        print(
            "[render-dcci-video] "
            f"dataset={name} dcci_frames={trace_summary['pose_frames']} "
            f"pcg={trace_summary['first_pcg_iter']}..{trace_summary['last_pcg_iter']} "
            + " ".join(cmd),
            flush=True)
        subprocess.run(cmd, cwd=repo_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

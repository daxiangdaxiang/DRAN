#!/usr/bin/env python3
"""Write objective-matched CCI handoff references for DCI certificate gates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

try:
  from scripts import analyze_distributed_chordal_init as dci  # type: ignore
  from scripts import run_two_stage_dci_certificate_sweep as sweep  # type: ignore
except ImportError:  # pragma: no cover - used when executed as a script.
  import analyze_distributed_chordal_init as dci  # type: ignore
  import run_two_stage_dci_certificate_sweep as sweep  # type: ignore


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description=(
          "Compute centralized chordal-initialization handoff costs for use "
          "as external DCI certificate-stop references."))
  parser.add_argument(
      "--dataset",
      action="append",
      required=True,
      help="Dataset graph, either NAME=PATH or PATH. Can be repeated.")
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="dpgo")
  parser.add_argument(
      "--robot-topology",
      choices=["chain", "complete", "none"],
      default="chain",
      help=(
          "Only used to reuse the same dataset/partition spec builder as DCI "
          "sweeps; CCI itself is centralized."))
  return parser.parse_args(argv)


def _write_outputs(output_dir: Path, rows: list[dict], full_stats: list[dict],
                   report: dict):
  output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = output_dir / "cci_reference_summary.csv"
  fieldnames = [
      "dataset",
      "graph_path",
      "reference_method",
      "reference_handoff_cost",
      "all_edges_evaluated",
      "evaluated_edge_count",
      "missing_edge_count",
      "pose_count",
      "edge_count",
      "dimension",
      "num_robots",
      "robot_count",
      "anchor_pose",
      "weighted",
      "cost_mode",
      "wall_time_sec",
      "rotation_solver",
      "rotation_iterations",
      "rotation_residual_norm",
      "translation_solver",
      "translation_iterations",
      "translation_residual_norm",
  ]
  with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
  (output_dir / "cci_reference_report.json").write_text(
      json.dumps({
          **report,
          "rows": rows,
          "full_stats": full_stats,
      },
                 indent=2,
                 sort_keys=True) + "\n",
      encoding="utf-8")


def build_reference_rows(dataset_specs: list[dict], weighted: bool,
                         cost_mode: str, num_robots: int):
  rows = []
  full_stats = []
  for spec in dataset_specs:
    start = time.perf_counter()
    poses, stats = dci.solve_centralized_chordal_initialization(
        graph_edges=spec["graph_edges"],
        pose_ids=spec["pose_ids"],
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=spec.get("anchor_pose"))
    handoff = dci._pose_set_total_chordal_cost(
        spec["graph_edges"], poses, weighted=weighted, cost_mode=cost_mode)
    wall_time_sec = time.perf_counter() - start
    dim = dci.pose_dimension_from_edges(spec["graph_edges"])
    robot_ids = sorted(set(spec["robot_of"].values()))
    row = {
        "dataset": str(spec["dataset"]),
        "graph_path": str(spec["graph_path"]),
        "reference_method": "centralized_chordal_initialization",
        "reference_handoff_cost": float(handoff["total_cost"]),
        "all_edges_evaluated": bool(handoff["all_edges_evaluated"]),
        "evaluated_edge_count": int(handoff["evaluated_edge_count"]),
        "missing_edge_count": int(handoff["missing_edge_count"]),
        "pose_count": int(len(spec["pose_ids"])),
        "edge_count": int(len(spec["graph_edges"])),
        "dimension": int(dim),
        "num_robots": int(num_robots),
        "robot_count": int(len(robot_ids)),
        "anchor_pose": int(stats.get("anchor_pose", spec["pose_ids"][0])),
        "weighted": bool(weighted),
        "cost_mode": str(cost_mode),
        "wall_time_sec": float(wall_time_sec),
        "rotation_solver": stats["rotation_stats"].get("solver"),
        "rotation_iterations": int(
            stats["rotation_stats"].get("iterations", 0)),
        "rotation_residual_norm": float(
            stats["rotation_stats"].get("residual_norm", 0.0)),
        "translation_solver": stats["translation_stats"].get("solver"),
        "translation_iterations": int(
            stats["translation_stats"].get("iterations", 0)),
        "translation_residual_norm": float(
            stats["translation_stats"].get("residual_norm", 0.0)),
    }
    rows.append(row)
    full_stats.append({
        "dataset": row["dataset"],
        "graph_path": row["graph_path"],
        "handoff_cost": handoff,
        "solver_stats": stats,
        "wall_time_sec": float(wall_time_sec),
    })
  return rows, full_stats


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  dataset_specs = sweep.build_dataset_specs(
      dataset_tokens=args.dataset,
      num_robots=args.num_robots,
      robot_topology=args.robot_topology)
  rows, full_stats = build_reference_rows(
      dataset_specs=dataset_specs,
      weighted=args.weighted,
      cost_mode=args.cost_mode,
      num_robots=args.num_robots)
  report = {
      "model": "cci_reference_summary",
      "row_count": int(len(rows)),
      "dataset_count": int(len(dataset_specs)),
      "reference_method": "centralized_chordal_initialization",
      "weighted": bool(args.weighted),
      "cost_mode": str(args.cost_mode),
      "num_robots": int(args.num_robots),
  }
  _write_outputs(args.output_dir, rows, full_stats, report)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

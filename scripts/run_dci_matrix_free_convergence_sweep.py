#!/usr/bin/env python3
"""Run controlled matrix-free DCI certificate convergence sweeps."""

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
      description="Run matrix-free DCI convergence sweeps over PCG budgets.")
  parser.add_argument(
      "--dataset",
      action="append",
      required=True,
      help="Dataset graph, either NAME=PATH or PATH. Can be repeated.")
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--iteration-budget",
      action="append",
      type=int,
      required=True,
      help="PCG iteration budget used for both rotation and translation.")
  parser.add_argument("--tolerance", type=float, default=1e-6)
  parser.add_argument("--damping", type=float, default=0.0)
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="dpgo")
  parser.add_argument(
      "--edge-owner-policy",
      choices=["lower_robot", "first_endpoint_robot", "second_endpoint_robot"],
      default="lower_robot")
  parser.add_argument(
      "--robot-topology",
      choices=["chain", "complete", "none"],
      default="chain")
  parser.add_argument("--hop-radius", type=int, default=1)
  parser.add_argument(
      "--communication-model",
      choices=["separator_owner_star", "separator_tree"],
      default="separator_owner_star")
  return parser.parse_args(argv)


def _write_report(output_dir: Path, rows: list[dict], full_stats: list[dict],
                  report: dict):
  output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = output_dir / "matrix_free_convergence_summary.csv"
  if rows:
    fieldnames = []
    for row in rows:
      for key in row:
        if key not in fieldnames:
          fieldnames.append(key)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)
  else:
    summary_path.write_text("", encoding="utf-8")
  report_payload = {
      **report,
      "rows": rows,
      "full_stats": full_stats,
  }
  (output_dir / "matrix_free_convergence_report.json").write_text(
      json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  budgets = [int(value) for value in args.iteration_budget]
  if any(value < 0 for value in budgets):
    raise ValueError("iteration budgets must be non-negative")
  dataset_specs = sweep.build_dataset_specs(
      dataset_tokens=args.dataset,
      num_robots=args.num_robots,
      robot_topology=args.robot_topology)
  rows = []
  full_stats = []
  for spec in dataset_specs:
    previous_total_gap = None
    previous_handoff_cost = None
    for budget in budgets:
      start = time.perf_counter()
      row = dci.two_stage_dci_certificate_sweep_row(
          dataset=str(spec["dataset"]),
          graph_edges=spec["graph_edges"],
          pose_ids=spec["pose_ids"],
          robot_of=spec["robot_of"],
          weighted=args.weighted,
          cost_mode=args.cost_mode,
          anchor_pose=spec.get("anchor_pose"),
          solver="pcg",
          rotation_iterations=budget,
          translation_iterations=budget,
          tolerance=args.tolerance,
          damping=args.damping,
          edge_owner_policy=args.edge_owner_policy,
          robot_topology_edges=spec.get("robot_topology_edges"),
          hop_radius=spec.get("hop_radius", args.hop_radius),
          communication_model=args.communication_model,
          hessian_storage="sparse",
          certificate_mode="matrix_free_dual")
      wall_time_sec = time.perf_counter() - start
      stats = row.pop("stats")
      total_gap = float(row["total_schur_energy_gap"])
      handoff_cost = float(row["handoff_cost"])
      row.update({
          "iteration_budget": int(budget),
          "wall_time_sec": float(wall_time_sec),
          "hessian_storage": "sparse",
          "certificate_mode": "matrix_free_dual",
          "dense_preflight_active": False,
          "total_schur_energy_gap_delta_from_previous": (
              "" if previous_total_gap is None
              else float(total_gap - previous_total_gap)),
          "handoff_cost_delta_from_previous": (
              "" if previous_handoff_cost is None
              else float(handoff_cost - previous_handoff_cost)),
      })
      previous_total_gap = total_gap
      previous_handoff_cost = handoff_cost
      rows.append(row)
      full_stats.append({
          "dataset": row["dataset"],
          "iteration_budget": int(budget),
          "wall_time_sec": float(wall_time_sec),
          "stats": stats,
      })
  report = {
      "model": "matrix_free_dci_convergence_sweep",
      "row_count": int(len(rows)),
      "dataset_count": int(len(dataset_specs)),
      "iteration_budgets": budgets,
      "weighted": bool(args.weighted),
      "cost_mode": args.cost_mode,
      "edge_owner_policy": args.edge_owner_policy,
      "hop_radius": int(max(0, int(args.hop_radius))),
      "communication_model": args.communication_model,
      "hessian_storage": "sparse",
      "certificate_mode": "matrix_free_dual",
  }
  _write_report(args.output_dir, rows, full_stats, report)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

#!/usr/bin/env python3
"""Run DCI sweeps with independent rotation/translation solver budgets."""

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
          "Run matrix-free DCI sweeps over independent rotation and "
          "translation PCG iteration budgets. This is a stage-attribution "
          "diagnostic, not a new default initializer."))
  parser.add_argument(
      "--dataset",
      action="append",
      required=True,
      help="Dataset graph, either NAME=PATH or PATH. Can be repeated.")
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--rotation-budget",
      action="append",
      type=int,
      required=True,
      help="Rotation-stage PCG iteration budget. Can be repeated.")
  parser.add_argument(
      "--translation-budget",
      action="append",
      type=int,
      required=True,
      help="Translation-stage PCG iteration budget. Can be repeated.")
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
  parser.add_argument(
      "--schur-preconditioner",
      choices=["none", "diagonal"],
      default="none")
  return parser.parse_args(argv)


def _finite_float(value) -> float:
  try:
    result = float(value)
  except (TypeError, ValueError):
    return float("nan")
  return result


def _best_row(rows: list[dict], key: str):
  finite = [
      row for row in rows
      if _finite_float(row.get(key)) == _finite_float(row.get(key))
  ]
  if not finite:
    return None
  return min(finite, key=lambda row: _finite_float(row.get(key)))


def _budget_pair(row: dict | None):
  if row is None:
    return None
  return {
      "rotation_iteration_budget": int(row["rotation_iteration_budget"]),
      "translation_iteration_budget": int(row["translation_iteration_budget"]),
  }


def _dataset_summaries(rows: list[dict]) -> list[dict]:
  groups: dict[str, list[dict]] = {}
  for row in rows:
    groups.setdefault(str(row.get("dataset", "")), []).append(row)
  summaries = []
  for dataset, dataset_rows in sorted(groups.items()):
    best_handoff = _best_row(dataset_rows, "handoff_cost")
    best_gap = _best_row(dataset_rows, "total_schur_energy_gap")
    summaries.append({
        "dataset": dataset,
        "row_count": int(len(dataset_rows)),
        "best_handoff_cost_pair": _budget_pair(best_handoff),
        "best_handoff_cost": (
            None if best_handoff is None
            else float(best_handoff["handoff_cost"])),
        "best_total_gap_pair": _budget_pair(best_gap),
        "best_total_schur_energy_gap": (
            None if best_gap is None
            else float(best_gap["total_schur_energy_gap"])),
    })
  return summaries


def _write_report(output_dir: Path, rows: list[dict], full_stats: list[dict],
                  report: dict):
  output_dir.mkdir(parents=True, exist_ok=True)
  summary_path = output_dir / "stage_budget_summary.csv"
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
  (output_dir / "stage_budget_report.json").write_text(
      json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  rotation_budgets = [int(value) for value in args.rotation_budget]
  translation_budgets = [int(value) for value in args.translation_budget]
  if any(value < 0 for value in rotation_budgets + translation_budgets):
    raise ValueError("stage budgets must be non-negative")
  dataset_specs = sweep.build_dataset_specs(
      dataset_tokens=args.dataset,
      num_robots=args.num_robots,
      robot_topology=args.robot_topology)
  rows = []
  full_stats = []
  for spec in dataset_specs:
    for rotation_budget in rotation_budgets:
      for translation_budget in translation_budgets:
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
            rotation_iterations=rotation_budget,
            translation_iterations=translation_budget,
            tolerance=args.tolerance,
            damping=args.damping,
            edge_owner_policy=args.edge_owner_policy,
            robot_topology_edges=spec.get("robot_topology_edges"),
            hop_radius=spec.get("hop_radius", args.hop_radius),
            communication_model=args.communication_model,
            schur_preconditioner=args.schur_preconditioner,
            hessian_storage="sparse",
            certificate_mode="matrix_free_dual")
        wall_time_sec = time.perf_counter() - start
        stats = row.pop("stats")
        row.update({
            "rotation_iteration_budget": int(rotation_budget),
            "translation_iteration_budget": int(translation_budget),
            "total_iteration_budget": int(rotation_budget + translation_budget),
            "stage_budget_pair": f"{rotation_budget}:{translation_budget}",
            "wall_time_sec": float(wall_time_sec),
            "hessian_storage": "sparse",
            "certificate_mode": "matrix_free_dual",
            "dense_preflight_active": False,
        })
        rows.append(row)
        full_stats.append({
            "dataset": row["dataset"],
            "rotation_iteration_budget": int(rotation_budget),
            "translation_iteration_budget": int(translation_budget),
            "wall_time_sec": float(wall_time_sec),
            "stats": stats,
        })
  report = {
      "model": "dci_stage_budget_sweep",
      "row_count": int(len(rows)),
      "dataset_count": int(len(dataset_specs)),
      "rotation_iteration_budgets": rotation_budgets,
      "translation_iteration_budgets": translation_budgets,
      "dataset_summaries": _dataset_summaries(rows),
      "weighted": bool(args.weighted),
      "cost_mode": args.cost_mode,
      "edge_owner_policy": args.edge_owner_policy,
      "hop_radius": int(max(0, int(args.hop_radius))),
      "communication_model": args.communication_model,
      "schur_preconditioner": args.schur_preconditioner,
      "hessian_storage": "sparse",
      "certificate_mode": "matrix_free_dual",
  }
  _write_report(args.output_dir, rows, full_stats, report)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze DCI stage-budget sweeps with Pareto dominance checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _float_or_nan(value) -> float:
  if value is None or value == "":
    return float("nan")
  try:
    return float(value)
  except (TypeError, ValueError):
    return float("nan")


def _int_or_zero(value) -> int:
  try:
    return int(float(value))
  except (TypeError, ValueError):
    return 0


def _is_finite(value: float) -> bool:
  return value == value


def read_summary_rows(summary_path: Path) -> list[dict]:
  with Path(summary_path).open(newline="", encoding="utf-8") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def _parsed_row(row: dict) -> dict:
  rotation_budget = _int_or_zero(row.get("rotation_iteration_budget"))
  translation_budget = _int_or_zero(row.get("translation_iteration_budget"))
  rotation_comm_mb = _float_or_nan(row.get("rotation_comm_mb"))
  translation_comm_mb = _float_or_nan(row.get("translation_comm_mb"))
  total_comm_mb = rotation_comm_mb + translation_comm_mb
  if not _is_finite(total_comm_mb):
    total_comm_mb = _float_or_nan(row.get("total_comm_mb"))
  pair = row.get("stage_budget_pair") or f"{rotation_budget}:{translation_budget}"
  return {
      "dataset": str(row.get("dataset", "")),
      "rotation_iteration_budget": rotation_budget,
      "translation_iteration_budget": translation_budget,
      "stage_budget_pair": str(pair),
      "is_equal_budget": bool(rotation_budget == translation_budget),
      "handoff_cost": _float_or_nan(row.get("handoff_cost")),
      "total_schur_energy_gap": _float_or_nan(row.get("total_schur_energy_gap")),
      "rotation_comm_mb": rotation_comm_mb,
      "translation_comm_mb": translation_comm_mb,
      "total_comm_mb": total_comm_mb,
  }


def _dominates(candidate: dict, other: dict, tolerance: float) -> bool:
  candidate_cost = candidate["handoff_cost"]
  other_cost = other["handoff_cost"]
  candidate_comm = candidate["total_comm_mb"]
  other_comm = other["total_comm_mb"]
  if not all(_is_finite(value) for value in [
      candidate_cost, other_cost, candidate_comm, other_comm]):
    return False
  not_worse = (
      candidate_cost <= other_cost + tolerance and
      candidate_comm <= other_comm + tolerance)
  strictly_better = (
      candidate_cost < other_cost - tolerance or
      candidate_comm < other_comm - tolerance)
  return bool(not_worse and strictly_better)


def analyze_rows(rows: list[dict], tolerance: float = 1e-9) -> dict:
  groups: dict[str, list[dict]] = {}
  for row in rows:
    parsed = _parsed_row(row)
    groups.setdefault(parsed["dataset"], []).append(parsed)

  dataset_reports = []
  for dataset, dataset_rows in sorted(groups.items()):
    frontier = []
    dominated_equal = []
    for row in dataset_rows:
      dominators = [
          other for other in dataset_rows
          if other is not row and _dominates(other, row, tolerance)
      ]
      dominated = bool(dominators)
      row_report = {**row, "pareto_dominated": dominated}
      if not dominated:
        frontier.append(row_report)
      if row["is_equal_budget"]:
        non_equal_dominators = [
            other for other in dominators if not other["is_equal_budget"]
        ]
        for dominator in non_equal_dominators:
          dominated_equal.append({
              "dominated_pair": row["stage_budget_pair"],
              "dominating_pair": dominator["stage_budget_pair"],
              "dominated_handoff_cost": float(row["handoff_cost"]),
              "dominating_handoff_cost": float(dominator["handoff_cost"]),
              "dominated_total_comm_mb": float(row["total_comm_mb"]),
              "dominating_total_comm_mb": float(dominator["total_comm_mb"]),
          })
    frontier.sort(
        key=lambda item: (item["total_comm_mb"], item["handoff_cost"],
                          item["stage_budget_pair"]))
    dataset_reports.append({
        "dataset": dataset,
        "row_count": int(len(dataset_rows)),
        "pareto_frontier_count": int(len(frontier)),
        "pareto_frontier": frontier,
        "non_equal_dominates_equal": bool(dominated_equal),
        "equal_budget_dominance_events": dominated_equal,
    })
  return {
      "model": "dci_stage_budget_pareto_analysis",
      "dataset_count": int(len(dataset_reports)),
      "row_count": int(len(rows)),
      "objective_axes": ["handoff_cost", "total_comm_mb"],
      "datasets": dataset_reports,
  }


def write_analysis(report: dict, output_dir: Path):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "stage_budget_pareto_report.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  lines = ["# DCI Stage-Budget Pareto Analysis", ""]
  for dataset in report["datasets"]:
    lines.extend([
        f"## {dataset['dataset']}",
        "",
        f"- rows: {dataset['row_count']}",
        f"- pareto_frontier_count: {dataset['pareto_frontier_count']}",
        f"- non_equal_dominates_equal: {dataset['non_equal_dominates_equal']}",
        "",
    ])
    if dataset["equal_budget_dominance_events"]:
      lines.append("Dominated equal-budget rows:")
      lines.append("")
      for event in dataset["equal_budget_dominance_events"]:
        lines.append(
            f"- {event['dominating_pair']} dominates equal-budget "
            f"{event['dominated_pair']} "
            f"(cost {event['dominating_handoff_cost']} <= "
            f"{event['dominated_handoff_cost']}, comm "
            f"{event['dominating_total_comm_mb']} <= "
            f"{event['dominated_total_comm_mb']})")
      lines.append("")
    lines.extend([
        "| pair | handoff_cost | total_comm_mb | equal_budget |",
        "| --- | ---: | ---: | --- |",
    ])
    for row in dataset["pareto_frontier"]:
      lines.append(
          f"| {row['stage_budget_pair']} | {row['handoff_cost']} | "
          f"{row['total_comm_mb']} | {row['is_equal_budget']} |")
    lines.append("")
  (output_dir / "stage_budget_pareto_report.md").write_text(
      "\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description="Analyze DCI stage-budget Pareto fronts.")
  parser.add_argument("--summary", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--tolerance", type=float, default=1e-9)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  rows = read_summary_rows(args.summary)
  report = analyze_rows(rows, tolerance=float(args.tolerance))
  write_analysis(report, args.output_dir)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

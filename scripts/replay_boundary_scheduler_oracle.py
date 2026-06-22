#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


SUMMARY_FIELDS = [
    "run_dir",
    "iteration_file",
    "cost_field",
    "num_rows",
    "initial_cost",
    "final_cost",
    "cost_drop",
    "total_comm_mb",
    "total_selected_updates",
    "total_candidate_updates",
    "avg_selected_fraction",
    "avg_predicted_gain_fraction",
    "avg_disagreement_per_update",
    "diagnostic_notes",
]

CANDIDATE_FIELDS = [
    "boundary_candidate_count",
    "object_comm_candidate_count",
    "boundary_budget_candidates",
]

APPROXIMATE_CANDIDATE_FIELDS = [
    "receiver_boundary_updates",
    "consensus_rejected_updates",
    "consensus_correction_updates",
]

SELECTED_FIELDS = [
    "receiver_boundary_updates",
    "object_comm_poses",
]


def read_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def find_iterations_path(run_dir: Path) -> Path:
  for name in ("iteration_summary.csv", "iterations.csv"):
    path = run_dir / name
    if path.exists():
      return path
  raise FileNotFoundError(
      f"missing iteration_summary.csv or iterations.csv under {run_dir}"
  )


def as_float(value: str | int | float | None, default: float = math.nan) -> float:
  if value is None or value == "":
    return default
  try:
    return float(value)
  except (TypeError, ValueError):
    return default


def fmt(value: float | int | str | None) -> str:
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  if isinstance(value, int):
    return str(value)
  if math.isnan(value) or math.isinf(value):
    return ""
  text = f"{value:.6f}".rstrip("0").rstrip(".")
  return text if text else "0"


def has_numeric(rows: list[dict[str, str]], field: str) -> bool:
  return any(not math.isnan(as_float(row.get(field))) for row in rows)


def sum_field(rows: list[dict[str, str]], field: str) -> float | None:
  if not any(row.get(field) not in (None, "") for row in rows):
    return None
  total = 0.0
  for row in rows:
    value = as_float(row.get(field))
    if not math.isnan(value):
      total += value
  return total


def mean_field(rows: list[dict[str, str]], field: str) -> float | None:
  values = [as_float(row.get(field)) for row in rows]
  values = [value for value in values if not math.isnan(value)]
  if not values:
    return None
  return sum(values) / len(values)


def choose_cost_field(rows: list[dict[str, str]], notes: list[str]) -> str:
  if has_numeric(rows, "chordal_measurement_cost"):
    return "chordal_measurement_cost"
  notes.append("missing chordal_measurement_cost")
  if has_numeric(rows, "measurement_cost"):
    return "measurement_cost"
  notes.append("missing measurement_cost")
  return "measurement_cost"


def sort_rows_by_iteration(rows: list[dict[str, str]]) -> list[dict[str, str]]:
  indexed = list(enumerate(rows))

  def key(item: tuple[int, dict[str, str]]) -> tuple[float, int]:
    index, row = item
    iteration = as_float(row.get("iter"))
    if math.isnan(iteration):
      iteration = float(index)
    return iteration, index

  return [row for _, row in sorted(indexed, key=key)]


def first_available_sum(
    rows: list[dict[str, str]],
    fields: list[str],
    notes: list[str],
    label: str,
) -> tuple[float, str | None]:
  for field in fields:
    total = sum_field(rows, field)
    if total is not None:
      return total, field
  notes.append(f"missing {label}")
  return math.nan, None


def candidate_update_sum(rows: list[dict[str, str]], notes: list[str]) -> float:
  for field in CANDIDATE_FIELDS:
    total = sum_field(rows, field)
    if total is not None:
      return total

  approximate_total = 0.0
  used_fields: list[str] = []
  for field in APPROXIMATE_CANDIDATE_FIELDS:
    total = sum_field(rows, field)
    if total is not None:
      approximate_total += total
      used_fields.append(field)
  if used_fields:
    notes.append("approximate candidates from " + "+".join(used_fields))
    return approximate_total

  notes.append("missing candidate update fields")
  return math.nan


def final_total_comm_mb(rows: list[dict[str, str]], notes: list[str]) -> float:
  final_row = rows[-1]
  total = as_float(final_row.get("total_comm_mb"))
  if not math.isnan(total):
    return total
  notes.append("missing total_comm_mb")
  total = as_float(final_row.get("cumulative_comm_mb"))
  if not math.isnan(total):
    return total
  notes.append("missing cumulative_comm_mb")
  return math.nan


def summarize_run(run_dir: Path) -> dict[str, str]:
  iterations_path = find_iterations_path(run_dir)
  rows = read_rows(iterations_path)
  if not rows:
    raise ValueError(f"{iterations_path} is empty")
  rows = sort_rows_by_iteration(rows)

  notes = [
      "aggregate-level oracle; iteration_summary.csv has aggregate fields only, "
      "not per-message candidates"
  ]
  cost_field = choose_cost_field(rows, notes)
  initial_cost = as_float(rows[0].get(cost_field))
  final_cost = as_float(rows[-1].get(cost_field))
  cost_drop = initial_cost - final_cost

  selected_updates, _ = first_available_sum(rows, SELECTED_FIELDS, notes, "selected update fields")
  candidate_updates = candidate_update_sum(rows, notes)
  total_comm_mb = final_total_comm_mb(rows, notes)

  selected_fraction = math.nan
  if not math.isnan(selected_updates) and not math.isnan(candidate_updates) and candidate_updates != 0.0:
    selected_fraction = selected_updates / candidate_updates

  predicted_gain_fraction = mean_field(rows, "boundary_predictive_selected_gain_fraction")
  if predicted_gain_fraction is None:
    notes.append("missing boundary_predictive_selected_gain_fraction")

  disagreement = sum_field(rows, "receiver_boundary_disagreement_sum")
  disagreement_per_update = math.nan
  if disagreement is None:
    notes.append("missing receiver_boundary_disagreement_sum")
  elif not math.isnan(selected_updates) and selected_updates != 0.0:
    disagreement_per_update = disagreement / selected_updates

  return {
      "run_dir": str(run_dir),
      "iteration_file": iterations_path.name,
      "cost_field": cost_field,
      "num_rows": fmt(len(rows)),
      "initial_cost": fmt(initial_cost),
      "final_cost": fmt(final_cost),
      "cost_drop": fmt(cost_drop),
      "total_comm_mb": fmt(total_comm_mb),
      "total_selected_updates": fmt(selected_updates),
      "total_candidate_updates": fmt(candidate_updates),
      "avg_selected_fraction": fmt(selected_fraction),
      "avg_predicted_gain_fraction": fmt(predicted_gain_fraction),
      "avg_disagreement_per_update": fmt(disagreement_per_update),
      "diagnostic_notes": "; ".join(notes),
  }


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Summarize aggregate-level boundary scheduler oracle signals from iteration_summary.csv."
  )
  parser.add_argument("run_dirs", nargs="+", type=Path, metavar="RUN_DIR")
  parser.add_argument("--output", required=True, type=Path, help="Output summary CSV path.")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    rows = [summarize_run(run_dir) for run_dir in args.run_dirs]
    write_summary(rows, args.output)
  except (OSError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

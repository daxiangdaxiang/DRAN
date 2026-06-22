#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


SUMMARY_FIELDS = [
    "run_dir",
    "run_name",
    "num_iters",
    "final_iter",
    "cost_field",
    "initial_cost",
    "final_cost",
    "best_cost",
    "cost_drop",
    "relative_cost_drop",
    "optimal_cost",
    "final_cost_gap",
    "init_comm_mb",
    "outer_comm_mb",
    "total_comm_mb",
    "final_comm_poses",
    "cost_drop_per_total_mb",
    "cost_gap_per_total_mb",
    "final_primal_residual",
    "final_dual_residual",
    "final_gradnorm",
    "final_consensus_cost",
    "final_known_object_copies",
    "final_relay_object_copies",
    "final_active_consensus_pairs",
    "final_frame_sync_objective",
    "stale_cache_skipped_total",
    "innovation_gate_skipped_total",
    "innovation_gate_forced_total",
    "rejected_updates_total",
    "payload_blocks_total",
    "diagnostic_note",
]

REJECTED_UPDATE_FIELDS = [
    "consensus_rejected_updates",
    "main_merit_rejected_updates",
    "gauge_merit_rejected_updates",
]


def read_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def as_float(value: str | None, default: float = math.nan) -> float:
  if value is None or value == "":
    return default
  try:
    return float(value)
  except ValueError:
    return default


def as_int(value: str | None, default: int = 0) -> int:
  number = as_float(value)
  if math.isnan(number):
    return default
  return int(round(number))


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


def choose_cost_field(rows: list[dict[str, str]]) -> str:
  if has_numeric(rows, "chordal_measurement_cost"):
    return "chordal_measurement_cost"
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


def sum_field(rows: list[dict[str, str]], field: str) -> float | None:
  if not any(field in row and row.get(field) not in (None, "") for row in rows):
    return None
  total = 0.0
  for row in rows:
    value = as_float(row.get(field))
    if not math.isnan(value):
      total += value
  return total


def sum_fields(rows: list[dict[str, str]], fields: list[str]) -> float | None:
  if not any(field in row and row.get(field) not in (None, "") for row in rows for field in fields):
    return None
  total = 0.0
  for row in rows:
    for field in fields:
      value = as_float(row.get(field))
      if not math.isnan(value):
        total += value
  return total


def final_or_blank(row: dict[str, str], field: str) -> str:
  value = as_float(row.get(field))
  return fmt(value)


def safe_div(numerator: float, denominator: float) -> float:
  if math.isnan(numerator) or math.isnan(denominator) or denominator == 0:
    return math.nan
  return numerator / denominator


def summarize_run(run_dir: Path, optimal_cost: float | None = None) -> dict[str, str]:
  iterations_path = run_dir / "iterations.csv"
  rows = read_rows(iterations_path)
  if not rows:
    raise ValueError(f"{iterations_path} is empty")

  rows = sort_rows_by_iteration(rows)
  iter0_rows = [row for row in rows if as_int(row.get("iter"), -1) == 0]
  initial_row = iter0_rows[0] if iter0_rows else rows[0]
  final_row = rows[-1]
  cost_field = choose_cost_field(rows)

  costs = [as_float(row.get(cost_field)) for row in rows]
  valid_costs = [value for value in costs if not math.isnan(value)]
  initial_cost = as_float(initial_row.get(cost_field))
  final_cost = as_float(final_row.get(cost_field))
  best_cost = min(valid_costs) if valid_costs else math.nan
  cost_drop = initial_cost - final_cost

  init_comm_mb = as_float(initial_row.get("comm_mb"))
  if math.isnan(init_comm_mb):
    init_comm_mb = as_float(initial_row.get("cumulative_comm_mb"), 0.0)

  total_comm_mb = as_float(final_row.get("cumulative_comm_mb"))
  if math.isnan(total_comm_mb):
    summed_comm = sum_field(rows, "comm_mb")
    total_comm_mb = summed_comm if summed_comm is not None else math.nan
  outer_comm_mb = total_comm_mb - init_comm_mb

  final_comm_poses = as_float(final_row.get("cumulative_object_comm_poses"))
  if math.isnan(final_comm_poses):
    summed_poses = sum_field(rows, "object_comm_poses")
    final_comm_poses = summed_poses if summed_poses is not None else math.nan

  payload_blocks_total = sum_field(rows, "object_comm_payload_blocks")
  stale_cache_skipped_total = sum_field(rows, "stale_cache_skipped")
  innovation_gate_skipped_total = sum_field(rows, "innovation_gate_skipped")
  innovation_gate_forced_total = sum_field(rows, "innovation_gate_forced")
  rejected_updates_total = sum_fields(rows, REJECTED_UPDATE_FIELDS)

  optimal = math.nan if optimal_cost is None else float(optimal_cost)
  final_cost_gap = final_cost - optimal if not math.isnan(optimal) else math.nan

  diagnostic_notes: list[str] = []
  if cost_field == "measurement_cost":
    diagnostic_notes.append("missing chordal_measurement_cost")
  if payload_blocks_total is None:
    diagnostic_notes.append("missing object_comm_payload_blocks")

  row = {
      "run_dir": str(run_dir),
      "run_name": run_dir.name,
      "num_iters": fmt(len(rows)),
      "final_iter": fmt(as_float(final_row.get("iter"))),
      "cost_field": cost_field,
      "initial_cost": fmt(initial_cost),
      "final_cost": fmt(final_cost),
      "best_cost": fmt(best_cost),
      "cost_drop": fmt(cost_drop),
      "relative_cost_drop": fmt(safe_div(cost_drop, initial_cost)),
      "optimal_cost": fmt(optimal),
      "final_cost_gap": fmt(final_cost_gap),
      "init_comm_mb": fmt(init_comm_mb),
      "outer_comm_mb": fmt(outer_comm_mb),
      "total_comm_mb": fmt(total_comm_mb),
      "final_comm_poses": fmt(final_comm_poses),
      "cost_drop_per_total_mb": fmt(safe_div(cost_drop, total_comm_mb)),
      "cost_gap_per_total_mb": fmt(safe_div(final_cost_gap, total_comm_mb)),
      "final_primal_residual": final_or_blank(final_row, "primal_residual"),
      "final_dual_residual": final_or_blank(final_row, "dual_residual"),
      "final_gradnorm": final_or_blank(final_row, "total_gradnorm"),
      "final_consensus_cost": final_or_blank(final_row, "consensus_cost"),
      "final_known_object_copies": final_or_blank(final_row, "known_object_copies"),
      "final_relay_object_copies": final_or_blank(final_row, "relay_object_copies"),
      "final_active_consensus_pairs": final_or_blank(final_row, "active_consensus_pairs"),
      "final_frame_sync_objective": final_or_blank(final_row, "frame_sync_objective"),
      "stale_cache_skipped_total": fmt(stale_cache_skipped_total),
      "innovation_gate_skipped_total": fmt(innovation_gate_skipped_total),
      "innovation_gate_forced_total": fmt(innovation_gate_forced_total),
      "rejected_updates_total": fmt(rejected_updates_total),
      "payload_blocks_total": fmt(payload_blocks_total),
      "diagnostic_note": "; ".join(diagnostic_notes),
  }
  return row


def build_summaries(run_dirs: list[Path], optimal_cost: float | None = None) -> list[dict[str, str]]:
  return [summarize_run(run_dir, optimal_cost) for run_dir in run_dirs]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def parse_args(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Summarize DRAN-Object iteration logs for phase-1 diagnostics."
  )
  parser.add_argument(
      "--run-dir",
      action="append",
      required=True,
      type=Path,
      help="DRAN-Object output directory containing iterations.csv. Repeatable.",
  )
  parser.add_argument(
      "--output",
      required=True,
      type=Path,
      help="Output summary CSV path.",
  )
  parser.add_argument(
      "--optimal-cost",
      type=float,
      default=None,
      help="Optional centralized reference cost used to compute final gap.",
  )
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    rows = build_summaries(args.run_dir, args.optimal_cost)
    write_csv(args.output, rows)
  except (OSError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


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
    "object_pose_rows",
    "has_receiver_boundary",
    "receiver_boundary_disagreement_sum_total",
    "receiver_boundary_cache_delta_sum_total",
    "object_comm_payload_blocks_total",
    "payload_blocks_total",
    "comm_mb_total",
    "total_comm_mb",
    "boundary_disagreement_per_mb",
    "boundary_cache_delta_per_mb",
    "selected_proxy_efficiency",
    "final_known_object_copies",
    "final_relay_object_copies",
    "final_active_consensus_pairs",
    "gt_trajectory_translation_rmse",
    "gt_trajectory_rotation_rmse_rad",
    "gt_trajectory_rotation_rmse_deg",
    "gt_object_translation_rmse",
    "gt_object_rotation_rmse_rad",
    "gt_object_rotation_rmse_deg",
    "gt_object_copy_translation_rmse",
    "gt_object_copy_rotation_rmse_deg",
    "gt_object_mean_translation_rmse",
    "gt_object_mean_rotation_rmse_deg",
    "diagnostic_note",
]


def read_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


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


def safe_div(numerator: float, denominator: float) -> float:
  if math.isnan(numerator) or math.isnan(denominator) or denominator == 0.0:
    return math.nan
  return numerator / denominator


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


def sum_field(rows: list[dict[str, str]], field: str, missing_default: float | None = None) -> float | None:
  present = any(field in row and row.get(field) not in (None, "") for row in rows)
  if not present:
    return missing_default
  total = 0.0
  for row in rows:
    value = as_float(row.get(field))
    if not math.isnan(value):
      total += value
  return total


def final_or_blank(row: dict[str, str], field: str) -> str:
  value = as_float(row.get(field))
  return fmt(value)


def count_object_pose_rows(path: Path) -> int:
  count = 0
  with path.open(encoding="utf-8") as handle:
    for raw in handle:
      tokens = raw.strip().split()
      if tokens and tokens[0] == "object":
        count += 1
  return count


def read_gt_eval(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  with path.open(encoding="utf-8") as handle:
    data = json.load(handle)
  if not isinstance(data, dict):
    return {}
  return data


def gt_value(gt_eval: dict[str, Any], *keys: str) -> str:
  for key in keys:
    value = gt_eval.get(key)
    if value is not None:
      return fmt(as_float(value))
  return ""


def summarize_run(run_dir: Path) -> dict[str, str]:
  iterations_path = run_dir / "iterations.csv"
  object_poses_path = run_dir / "object_poses.txt"
  rows = read_rows(iterations_path)
  if not rows:
    raise ValueError(f"{iterations_path} is empty")
  if not object_poses_path.exists():
    raise ValueError(f"missing {object_poses_path}")

  rows = sort_rows_by_iteration(rows)
  initial_row = rows[0]
  final_row = rows[-1]
  cost_field = choose_cost_field(rows)
  costs = [as_float(row.get(cost_field)) for row in rows]
  valid_costs = [value for value in costs if not math.isnan(value)]
  initial_cost = as_float(initial_row.get(cost_field))
  final_cost = as_float(final_row.get(cost_field))
  best_cost = min(valid_costs) if valid_costs else math.nan
  cost_drop = initial_cost - final_cost

  has_receiver_boundary = (
      sum_field(rows, "receiver_boundary_disagreement_sum", None) is not None
      or sum_field(rows, "receiver_boundary_cache_delta_sum", None) is not None
  )
  boundary_disagreement = sum_field(rows, "receiver_boundary_disagreement_sum", 0.0)
  boundary_cache_delta = sum_field(rows, "receiver_boundary_cache_delta_sum", 0.0)
  payload_blocks = sum_field(rows, "object_comm_payload_blocks", 0.0)

  cumulative_comm = as_float(final_row.get("cumulative_comm_mb"))
  comm_mb_sum = sum_field(rows, "comm_mb", None)
  total_comm_mb = cumulative_comm
  if math.isnan(total_comm_mb):
    total_comm_mb = comm_mb_sum if comm_mb_sum is not None else math.nan

  gt_eval = read_gt_eval(run_dir / "gt_eval.json")
  diagnostic_notes: list[str] = []
  if not has_receiver_boundary:
    diagnostic_notes.append("missing receiver_boundary diagnostics")
  if "object_comm_payload_blocks" not in rows[0]:
    diagnostic_notes.append("missing object_comm_payload_blocks")
  if not gt_eval:
    diagnostic_notes.append("missing gt_eval.json")

  boundary_disagreement_value = float(boundary_disagreement or 0.0)
  boundary_cache_delta_value = float(boundary_cache_delta or 0.0)
  payload_blocks_value = float(payload_blocks or 0.0)

  return {
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
      "object_pose_rows": fmt(count_object_pose_rows(object_poses_path)),
      "has_receiver_boundary": "true" if has_receiver_boundary else "false",
      "receiver_boundary_disagreement_sum_total": fmt(boundary_disagreement_value),
      "receiver_boundary_cache_delta_sum_total": fmt(boundary_cache_delta_value),
      "object_comm_payload_blocks_total": fmt(payload_blocks_value),
      "payload_blocks_total": fmt(payload_blocks_value),
      "comm_mb_total": fmt(total_comm_mb),
      "total_comm_mb": fmt(total_comm_mb),
      "boundary_disagreement_per_mb": fmt(safe_div(boundary_disagreement_value, total_comm_mb)),
      "boundary_cache_delta_per_mb": fmt(safe_div(boundary_cache_delta_value, total_comm_mb)),
      "selected_proxy_efficiency": fmt(safe_div(boundary_disagreement_value, payload_blocks_value)),
      "final_known_object_copies": final_or_blank(final_row, "known_object_copies"),
      "final_relay_object_copies": final_or_blank(final_row, "relay_object_copies"),
      "final_active_consensus_pairs": final_or_blank(final_row, "active_consensus_pairs"),
      "gt_trajectory_translation_rmse": gt_value(gt_eval, "trajectory_translation_rmse"),
      "gt_trajectory_rotation_rmse_rad": gt_value(gt_eval, "trajectory_rotation_rmse_rad"),
      "gt_trajectory_rotation_rmse_deg": gt_value(gt_eval, "trajectory_rotation_rmse_deg"),
      "gt_object_translation_rmse": gt_value(gt_eval, "object_translation_rmse"),
      "gt_object_rotation_rmse_rad": gt_value(gt_eval, "object_rotation_rmse_rad"),
      "gt_object_rotation_rmse_deg": gt_value(gt_eval, "object_rotation_rmse_deg"),
      "gt_object_copy_translation_rmse": gt_value(gt_eval, "object_copy_translation_rmse"),
      "gt_object_copy_rotation_rmse_deg": gt_value(gt_eval, "object_copy_rotation_rmse_deg"),
      "gt_object_mean_translation_rmse": gt_value(gt_eval, "object_mean_translation_rmse"),
      "gt_object_mean_rotation_rmse_deg": gt_value(gt_eval, "object_mean_rotation_rmse_deg"),
      "diagnostic_note": "; ".join(diagnostic_notes),
  }


def build_summaries(run_dirs: list[Path]) -> list[dict[str, str]]:
  return [summarize_run(run_dir) for run_dir in run_dirs]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def parse_args(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Summarize DRAN-Object boundary-response diagnostics as an offline Schur-style oracle."
  )
  parser.add_argument(
      "--run-dir",
      action="append",
      required=True,
      type=Path,
      help="DRAN-Object run directory containing iterations.csv and object_poses.txt. Repeatable.",
  )
  parser.add_argument("--output", required=True, type=Path, help="Output summary CSV path.")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    write_csv(args.output, build_summaries(args.run_dir))
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

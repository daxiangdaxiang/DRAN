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
    "iteration_file",
    "num_rows",
    "cost_field",
    "initial_cost",
    "final_cost",
    "best_cost",
    "cost_drop",
    "total_comm_mb",
    "total_payload_blocks",
    "total_predicted_gain",
    "total_selected_predicted_gain",
    "total_stiffness",
    "total_interface_state_grad_norm",
    "interface_state_total_comm_mb",
    "mean_selected_gain_fraction",
    "cost_drop_per_comm_mb",
    "cost_drop_per_selected_gain",
    "selected_gain_to_next_cost_drop_corr",
    "comm_to_next_cost_drop_corr",
    "signal_source",
    "diagnostic_notes",
]

PREDICTIVE_TOTAL_GAIN_FIELD = "boundary_predictive_gain_sum"
PREDICTIVE_SELECTED_GAIN_FIELD = "boundary_predictive_selected_gain_sum"
PREDICTIVE_STIFFNESS_FIELD = "boundary_predictive_stiffness_sum"
PREDICTIVE_FRACTION_FIELD = "boundary_predictive_selected_gain_fraction"
BOUNDARY_MODEL_SELECTED_GAIN_FIELD = "boundary_model_merit_decrease"
INTERFACE_STATE_PAYLOAD_FIELD = "interface_state_payload_blocks"
INTERFACE_STATE_GRAD_FIELD = "interface_state_grad_norm"
INTERFACE_STATE_STIFFNESS_FIELD = "interface_state_stiffness_sum"
INTERFACE_STATE_COMM_FIELD = "interface_state_comm_mb"
INTERFACE_STATE_CUMULATIVE_COMM_FIELD = "cumulative_interface_state_comm_mb"


def read_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def find_iterations_path(run_dir: Path) -> Path:
  for name in ("iterations.csv", "iteration_summary.csv"):
    path = run_dir / name
    if path.exists():
      return path
  raise FileNotFoundError(
      f"missing iterations.csv or iteration_summary.csv under {run_dir}"
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


def safe_div(numerator: float, denominator: float) -> float:
  if math.isnan(numerator) or math.isnan(denominator) or denominator == 0.0:
    return math.nan
  return numerator / denominator


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


def sort_rows_by_iteration(rows: list[dict[str, str]]) -> list[dict[str, str]]:
  indexed = list(enumerate(rows))

  def key(item: tuple[int, dict[str, str]]) -> tuple[float, int]:
    index, row = item
    iteration = as_float(row.get("iter"))
    if math.isnan(iteration):
      iteration = float(index)
    return iteration, index

  return [row for _, row in sorted(indexed, key=key)]


def choose_cost_field(rows: list[dict[str, str]], notes: list[str]) -> str:
  for field in ("chordal_measurement_cost", "measurement_cost", "cost", "global_cost"):
    if has_numeric(rows, field):
      return field
  notes.append("missing recognized cost field")
  return "measurement_cost"


def final_total_comm_mb(rows: list[dict[str, str]], notes: list[str]) -> float:
  final_row = rows[-1]
  for field in ("total_comm_mb", "cumulative_comm_mb", "cumulative_object_comm_mb"):
    value = as_float(final_row.get(field))
    if not math.isnan(value):
      return value
  for field in ("comm_mb", "iter_comm_mb"):
    total = sum_field(rows, field)
    if total is not None:
      return total
  notes.append("missing communication MB fields")
  return math.nan


def total_payload_blocks(rows: list[dict[str, str]], notes: list[str]) -> float:
  for field in (
      INTERFACE_STATE_PAYLOAD_FIELD,
      "object_comm_payload_blocks",
      "object_comm_poses",
      "comm_pose_count",
      "iter_comm_poses",
  ):
    total = sum_field(rows, field)
    if total is not None:
      return total
  notes.append("missing payload block/count fields")
  return math.nan


def pearson(xs: list[float], ys: list[float]) -> float:
  pairs = [
      (x, y)
      for x, y in zip(xs, ys)
      if not math.isnan(x) and not math.isnan(y)
  ]
  if len(pairs) < 2:
    return math.nan
  x_values = [x for x, _ in pairs]
  y_values = [y for _, y in pairs]
  x_mean = sum(x_values) / len(x_values)
  y_mean = sum(y_values) / len(y_values)
  x_centered = [x - x_mean for x in x_values]
  y_centered = [y - y_mean for y in y_values]
  numerator = sum(x * y for x, y in zip(x_centered, y_centered))
  x_denom = math.sqrt(sum(x * x for x in x_centered))
  y_denom = math.sqrt(sum(y * y for y in y_centered))
  if x_denom == 0.0 or y_denom == 0.0:
    return math.nan
  return numerator / (x_denom * y_denom)


def aligned_next_cost_drop_correlation(
    rows: list[dict[str, str]],
    cost_field: str,
    signal_field: str | None,
) -> float:
  if signal_field is None:
    return math.nan
  drops: list[float] = []
  signals: list[float] = []
  for index in range(1, len(rows)):
    previous_cost = as_float(rows[index - 1].get(cost_field))
    current_cost = as_float(rows[index].get(cost_field))
    signal = as_float(rows[index].get(signal_field))
    if math.isnan(previous_cost) or math.isnan(current_cost):
      continue
    drops.append(previous_cost - current_cost)
    signals.append(signal)
  return pearson(signals, drops)


def aligned_comm_cost_drop_correlation(
    rows: list[dict[str, str]],
    cost_field: str,
) -> float:
  drops: list[float] = []
  comms: list[float] = []
  for index in range(1, len(rows)):
    previous_cost = as_float(rows[index - 1].get(cost_field))
    current_cost = as_float(rows[index].get(cost_field))
    if math.isnan(previous_cost) or math.isnan(current_cost):
      continue
    comm = as_float(rows[index].get("comm_mb"))
    if math.isnan(comm):
      comm = as_float(rows[index].get("iter_comm_mb"))
    if math.isnan(comm):
      continue
    drops.append(previous_cost - current_cost)
    comms.append(comm)
  return pearson(comms, drops)


def extract_interface_signal(
    rows: list[dict[str, str]], notes: list[str]
) -> tuple[float, float, float, float, str, str | None]:
  if has_numeric(rows, PREDICTIVE_SELECTED_GAIN_FIELD) or has_numeric(
      rows, PREDICTIVE_TOTAL_GAIN_FIELD
  ):
    total_gain = sum_field(rows, PREDICTIVE_TOTAL_GAIN_FIELD)
    selected_gain = sum_field(rows, PREDICTIVE_SELECTED_GAIN_FIELD)
    stiffness = sum_field(rows, PREDICTIVE_STIFFNESS_FIELD)
    mean_fraction = mean_field(rows, PREDICTIVE_FRACTION_FIELD)
    if total_gain is None:
      total_gain = selected_gain
      notes.append("missing boundary_predictive_gain_sum")
    if selected_gain is None:
      selected_gain = 0.0
      notes.append("missing boundary_predictive_selected_gain_sum")
    if stiffness is None:
      stiffness = math.nan
      notes.append("missing boundary_predictive_stiffness_sum")
    if mean_fraction is None:
      mean_fraction = math.nan
      notes.append("missing boundary_predictive_selected_gain_fraction")
    return (
        float(total_gain or 0.0),
        float(selected_gain or 0.0),
        stiffness,
        mean_fraction,
        "object boundary_predictive fields",
        PREDICTIVE_SELECTED_GAIN_FIELD,
    )

  if has_numeric(rows, INTERFACE_STATE_PAYLOAD_FIELD) or has_numeric(
      rows, INTERFACE_STATE_GRAD_FIELD
  ):
    stiffness = sum_field(rows, INTERFACE_STATE_STIFFNESS_FIELD)
    if stiffness is None:
      stiffness = math.nan
      notes.append("missing interface_state_stiffness_sum")
    notes.append(
        "interface_state diagnostic fields are not a predicted-gain signal"
    )
    return (
        math.nan,
        math.nan,
        stiffness,
        math.nan,
        "interface_state diagnostic fields",
        None,
    )

  if has_numeric(rows, BOUNDARY_MODEL_SELECTED_GAIN_FIELD):
    selected_gain = sum_field(rows, BOUNDARY_MODEL_SELECTED_GAIN_FIELD) or 0.0
    notes.append(
        "diagnostic fallback: boundary_model merit is local receiver merit, not "
        "a true compact interface-state signal"
    )
    return (
        selected_gain,
        selected_gain,
        math.nan,
        math.nan,
        "boundary_model merit fallback",
        BOUNDARY_MODEL_SELECTED_GAIN_FIELD,
    )

  notes.append("missing interface signal fields")
  return math.nan, math.nan, math.nan, math.nan, "missing", None


def interface_state_total_comm_mb(rows: list[dict[str, str]]) -> float:
  total = sum_field(rows, INTERFACE_STATE_COMM_FIELD)
  if total is not None:
    return total
  final_value = as_float(rows[-1].get(INTERFACE_STATE_CUMULATIVE_COMM_FIELD))
  return final_value


def summarize_run(run_dir: Path) -> dict[str, str]:
  iterations_path = find_iterations_path(run_dir)
  rows = read_rows(iterations_path)
  if not rows:
    raise ValueError(f"{iterations_path} is empty")
  rows = sort_rows_by_iteration(rows)

  notes: list[str] = [
      "offline oracle; aggregate CSV fields cannot prove per-message causality"
  ]
  cost_field = choose_cost_field(rows, notes)
  costs = [as_float(row.get(cost_field)) for row in rows]
  valid_costs = [value for value in costs if not math.isnan(value)]
  initial_cost = as_float(rows[0].get(cost_field))
  final_cost = as_float(rows[-1].get(cost_field))
  best_cost = min(valid_costs) if valid_costs else math.nan
  cost_drop = initial_cost - final_cost

  total_comm = final_total_comm_mb(rows, notes)
  payload_blocks = total_payload_blocks(rows, notes)
  interface_grad_norm = sum_field(rows, INTERFACE_STATE_GRAD_FIELD)
  interface_comm_mb = interface_state_total_comm_mb(rows)
  (
      total_gain,
      selected_gain,
      stiffness,
      mean_fraction,
      signal_source,
      signal_field,
  ) = extract_interface_signal(rows, notes)

  selected_gain_corr = aligned_next_cost_drop_correlation(
      rows, cost_field, signal_field
  )
  comm_corr = aligned_comm_cost_drop_correlation(rows, cost_field)

  return {
      "run_dir": str(run_dir),
      "run_name": run_dir.name,
      "iteration_file": iterations_path.name,
      "num_rows": fmt(len(rows)),
      "cost_field": cost_field,
      "initial_cost": fmt(initial_cost),
      "final_cost": fmt(final_cost),
      "best_cost": fmt(best_cost),
      "cost_drop": fmt(cost_drop),
      "total_comm_mb": fmt(total_comm),
      "total_payload_blocks": fmt(payload_blocks),
      "total_predicted_gain": fmt(total_gain),
      "total_selected_predicted_gain": fmt(selected_gain),
      "total_stiffness": fmt(stiffness),
      "total_interface_state_grad_norm": fmt(interface_grad_norm),
      "interface_state_total_comm_mb": fmt(interface_comm_mb),
      "mean_selected_gain_fraction": fmt(mean_fraction),
      "cost_drop_per_comm_mb": fmt(safe_div(cost_drop, total_comm)),
      "cost_drop_per_selected_gain": fmt(safe_div(cost_drop, selected_gain)),
      "selected_gain_to_next_cost_drop_corr": fmt(selected_gain_corr),
      "comm_to_next_cost_drop_corr": fmt(comm_corr),
      "signal_source": signal_source,
      "diagnostic_notes": "; ".join(notes),
  }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def parse_args(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Summarize aggregate compact-interface-state signals from DRAN "
          "iteration CSVs before changing optimizer code."
      )
  )
  parser.add_argument(
      "--run-dir",
      action="append",
      required=True,
      type=Path,
      help="Run directory containing iterations.csv or iteration_summary.csv.",
  )
  parser.add_argument("--output", required=True, type=Path)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    rows = [summarize_run(run_dir) for run_dir in args.run_dir]
    write_csv(args.output, rows)
  except (OSError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

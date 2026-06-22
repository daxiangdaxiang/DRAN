#!/usr/bin/env python3
"""Select DCI stage-budget rows by certificate gates and minimum communication."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REFERENCE_HANDOFF_COST_COLUMNS = (
    "reference_handoff_cost",
    "cci_handoff_cost",
    "centralized_handoff_cost",
    "optimal_cost",
    "handoff_cost",
)


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


def _finite(value: float) -> bool:
  return value == value and math.isfinite(value)


def read_summary_rows(summary_path: Path) -> list[dict]:
  with Path(summary_path).open(newline="", encoding="utf-8") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def read_reference_summary(reference_path: Path) -> dict[str, float]:
  references: dict[str, float] = {}
  with Path(reference_path).open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
      dataset = str(row.get("dataset", ""))
      if not dataset:
        continue
      for column in REFERENCE_HANDOFF_COST_COLUMNS:
        value = _float_or_nan(row.get(column))
        if _finite(value):
          references[dataset] = value
          break
  return references


def _parsed_row(row: dict) -> dict:
  rotation_budget = _int_or_zero(row.get("rotation_iteration_budget"))
  translation_budget = _int_or_zero(row.get("translation_iteration_budget"))
  rotation_comm_mb = _float_or_nan(row.get("rotation_comm_mb"))
  translation_comm_mb = _float_or_nan(row.get("translation_comm_mb"))
  total_comm_mb = rotation_comm_mb + translation_comm_mb
  if not _finite(total_comm_mb):
    total_comm_mb = _float_or_nan(row.get("total_comm_mb"))
  pair = row.get("stage_budget_pair") or f"{rotation_budget}:{translation_budget}"
  signed_projection_cost_delta = _float_or_nan(
      row.get("rotation_projection_cost_delta"))
  absolute_projection_cost_delta = _float_or_nan(
      row.get("rotation_abs_projection_cost_delta"))
  if _finite(signed_projection_cost_delta):
    projection_cost_increase = max(0.0, signed_projection_cost_delta)
  else:
    projection_cost_increase = absolute_projection_cost_delta
  return {
      "dataset": str(row.get("dataset", "")),
      "rotation_iteration_budget": rotation_budget,
      "translation_iteration_budget": translation_budget,
      "stage_budget_pair": str(pair),
      "handoff_cost": _float_or_nan(row.get("handoff_cost")),
      "rotation_schur_residual_norm": _float_or_nan(
          row.get("rotation_schur_residual_norm")),
      "translation_schur_residual_norm": _float_or_nan(
          row.get("translation_schur_residual_norm")),
      "rotation_max_projection_correction_norm": _float_or_nan(
          row.get("rotation_max_projection_correction_norm")),
      "rotation_projection_cost_delta": signed_projection_cost_delta,
      "rotation_abs_projection_cost_delta": absolute_projection_cost_delta,
      "rotation_projection_cost_increase": projection_cost_increase,
      "solver": str(row.get("solver", "")),
      "rotation_solver_model": str(row.get("rotation_solver_model", "")),
      "translation_solver_model": str(
          row.get("translation_solver_model", "")),
      "rotation_schur_preconditioner": str(
          row.get("rotation_schur_preconditioner", "")),
      "translation_schur_preconditioner": str(
          row.get("translation_schur_preconditioner", "")),
      "rotation_fixed_step_acceleration": str(
          row.get("rotation_fixed_step_acceleration", "")),
      "translation_fixed_step_acceleration": str(
          row.get("translation_fixed_step_acceleration", "")),
      "rotation_chebyshev_certificate_policy_decision": str(
          row.get("rotation_chebyshev_certificate_policy_decision", "")),
      "translation_chebyshev_certificate_policy_decision": str(
          row.get("translation_chebyshev_certificate_policy_decision", "")),
      "rotation_chebyshev_bound_source": str(
          row.get("rotation_chebyshev_bound_source", "")),
      "translation_chebyshev_bound_source": str(
          row.get("translation_chebyshev_bound_source", "")),
      "rotation_chebyshev_block_certificate_payload_mb": _float_or_nan(
          row.get("rotation_chebyshev_block_certificate_payload_mb")),
      "translation_chebyshev_block_certificate_payload_mb": _float_or_nan(
          row.get("translation_chebyshev_block_certificate_payload_mb")),
      "rotation_schur_preconditioner_private_solve_count": _int_or_zero(
          row.get("rotation_schur_preconditioner_private_solve_count")),
      "translation_schur_preconditioner_private_solve_count": _int_or_zero(
          row.get("translation_schur_preconditioner_private_solve_count")),
      "rotation_comm_mb": rotation_comm_mb,
      "translation_comm_mb": translation_comm_mb,
      "total_comm_mb": total_comm_mb,
  }


def _passes(value: float, maximum: float | None) -> bool:
  if maximum is None:
    return True
  return _finite(value) and value <= maximum


def _failure_reasons(
    row: dict,
    gates: dict[str, float | None],
    missing_reference_handoff_cost: bool = False) -> list[str]:
  reasons = []
  if missing_reference_handoff_cost:
    reasons.append("missing_reference_handoff_cost")
  if not _passes(row["rotation_schur_residual_norm"],
                 gates["max_rotation_residual"]):
    reasons.append("rotation_residual")
  if not _passes(row["translation_schur_residual_norm"],
                 gates["max_translation_residual"]):
    reasons.append("translation_residual")
  if not _passes(row["rotation_max_projection_correction_norm"],
                 gates["max_rotation_projection_correction"]):
    reasons.append("rotation_projection")
  if not _passes(row["rotation_projection_cost_increase"],
                 gates["max_rotation_projection_cost_delta"]):
    reasons.append("rotation_projection_cost")
  if not _passes(row["handoff_cost"], gates["max_handoff_cost"]):
    reasons.append("handoff_cost")
  return reasons


def _best_finite(rows: list[dict], key: str) -> float | None:
  values = [row[key] for row in rows if _finite(row[key])]
  if not values:
    return None
  return min(values)


def _derived_gates_for_dataset(
    dataset_rows: list[dict],
    base_gates: dict[str, float | None],
    gate_source: str,
    slacks: dict[str, float],
    reference_handoff_cost: float | None = None) -> dict[str, float | None]:
  if gate_source == "reference_summary_relative":
    dataset_gates = dict(base_gates)
    if reference_handoff_cost is not None and _finite(reference_handoff_cost):
      if dataset_gates["max_handoff_cost"] is None:
        dataset_gates["max_handoff_cost"] = float(
            reference_handoff_cost *
            (1.0 + slacks["handoff_reference_relative_slack"]))
      if dataset_gates["max_rotation_projection_cost_delta"] is None:
        dataset_gates["max_rotation_projection_cost_delta"] = float(
            reference_handoff_cost *
            slacks["projection_cost_reference_relative_slack"])
    return dataset_gates
  if gate_source != "best_observed_plus_slack":
    return dict(base_gates)
  best_rotation = _best_finite(dataset_rows, "rotation_schur_residual_norm")
  best_translation = _best_finite(
      dataset_rows, "translation_schur_residual_norm")
  best_projection = _best_finite(
      dataset_rows, "rotation_max_projection_correction_norm")
  best_projection_cost = _best_finite(
      dataset_rows, "rotation_projection_cost_increase")
  best_handoff = _best_finite(dataset_rows, "handoff_cost")
  return {
      "max_rotation_residual": (
          None if best_rotation is None
          else float(best_rotation + slacks["rotation_residual_slack"])),
      "max_translation_residual": (
          None if best_translation is None
          else float(best_translation + slacks["translation_residual_slack"])),
      "max_rotation_projection_correction": (
          None if best_projection is None
          else float(best_projection + slacks[
              "rotation_projection_correction_slack"])),
      "max_rotation_projection_cost_delta": (
          None if best_projection_cost is None
          else float(best_projection_cost + slacks[
              "rotation_projection_cost_delta_slack"])),
      "max_handoff_cost": (
          None if best_handoff is None
          else float(best_handoff + slacks["handoff_cost_slack"])),
  }


def analyze_rows(
    rows: list[dict],
    gates: dict[str, float | None],
    gate_source: str = "manual",
    slacks: dict[str, float] | None = None,
    reference_handoff_costs: dict[str, float] | None = None) -> dict:
  slacks = slacks or {
      "rotation_residual_slack": 0.0,
      "translation_residual_slack": 0.0,
      "rotation_projection_correction_slack": 0.0,
      "rotation_projection_cost_delta_slack": 0.0,
      "handoff_cost_slack": 0.0,
      "handoff_reference_relative_slack": 0.0,
      "projection_cost_reference_relative_slack": 0.0,
  }
  reference_handoff_costs = reference_handoff_costs or {}
  groups: dict[str, list[dict]] = {}
  for row in rows:
    parsed = _parsed_row(row)
    groups.setdefault(parsed["dataset"], []).append(parsed)

  dataset_reports = []
  for dataset, dataset_rows in sorted(groups.items()):
    reference_handoff_cost = reference_handoff_costs.get(dataset)
    missing_reference_handoff_cost = (
        gate_source == "reference_summary_relative" and
        not (reference_handoff_cost is not None and
             _finite(reference_handoff_cost)))
    dataset_gates = _derived_gates_for_dataset(
        dataset_rows=dataset_rows,
        base_gates=gates,
        gate_source=gate_source,
        slacks=slacks,
        reference_handoff_cost=reference_handoff_cost)
    row_reports = []
    failure_counts = {
        "rotation_residual": 0,
        "translation_residual": 0,
        "rotation_projection": 0,
        "rotation_projection_cost": 0,
        "handoff_cost": 0,
        "missing_reference_handoff_cost": 0,
    }
    for row in dataset_rows:
      reasons = _failure_reasons(
          row,
          dataset_gates,
          missing_reference_handoff_cost=missing_reference_handoff_cost)
      for reason in reasons:
        failure_counts[reason] += 1
      row_reports.append({
          **row,
          "feasible": bool(not reasons),
          "failure_reasons": reasons,
      })
    feasible_rows = [row for row in row_reports if row["feasible"]]
    selected = None
    if feasible_rows:
      selected = min(
          feasible_rows,
          key=lambda row: (
              row["total_comm_mb"],
              row["handoff_cost"],
              row["rotation_iteration_budget"] + row["translation_iteration_budget"],
              row["stage_budget_pair"]))
    dataset_reports.append({
        "dataset": dataset,
        "row_count": int(len(row_reports)),
        "feasible_count": int(len(feasible_rows)),
        "feasible": bool(selected is not None),
        "selection_rule": "min_total_comm_then_handoff_cost",
        "selected_pair": None if selected is None else selected["stage_budget_pair"],
        "selected_row": selected,
        "reference_handoff_cost": reference_handoff_cost,
        "gates": dataset_gates,
        "failure_counts": failure_counts,
        "rows": row_reports,
    })
  return {
      "model": "dci_stage_certificate_stop_analysis",
      "dataset_count": int(len(dataset_reports)),
      "row_count": int(len(rows)),
      "gate_source": gate_source,
      "gates": gates,
      "slacks": slacks,
      "reference_handoff_costs": reference_handoff_costs,
      "datasets": dataset_reports,
  }


def write_analysis(report: dict, output_dir: Path):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  (output_dir / "stage_certificate_stop_report.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  lines = ["# DCI Stage Certificate Stop Analysis", ""]
  lines.extend([
      "Default gates:",
      "",
      f"- max_rotation_residual: {report['gates']['max_rotation_residual']}",
      f"- max_translation_residual: {report['gates']['max_translation_residual']}",
      f"- max_rotation_projection_correction: {report['gates']['max_rotation_projection_correction']}",
      f"- max_rotation_projection_cost_delta: {report['gates']['max_rotation_projection_cost_delta']}",
      f"- max_handoff_cost: {report['gates']['max_handoff_cost']}",
      f"- gate_source: {report['gate_source']}",
      f"- slacks: {report['slacks']}",
      "",
  ])
  for dataset in report["datasets"]:
    lines.extend([
        f"## {dataset['dataset']}",
        "",
        f"- feasible: {dataset['feasible']}",
        f"- feasible_count: {dataset['feasible_count']}",
        f"- selection_rule: {dataset['selection_rule']}",
        f"- selected_pair: {dataset['selected_pair']}",
        f"- reference_handoff_cost: {dataset['reference_handoff_cost']}",
        f"- gates: {dataset['gates']}",
        f"- failure_counts: {dataset['failure_counts']}",
        "",
        "| pair | feasible | handoff_cost | total_comm_mb | rotation_residual | translation_residual | rotation_projection | rotation_projection_cost | failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in dataset["rows"]:
      failures = ",".join(row["failure_reasons"]) or "none"
      lines.append(
          f"| {row['stage_budget_pair']} | {row['feasible']} | "
          f"{row['handoff_cost']} | {row['total_comm_mb']} | "
          f"{row['rotation_schur_residual_norm']} | "
          f"{row['translation_schur_residual_norm']} | "
          f"{row['rotation_max_projection_correction_norm']} | "
          f"{row['rotation_projection_cost_increase']} | "
          f"{failures} |")
    lines.append("")
  (output_dir / "stage_certificate_stop_report.md").write_text(
      "\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description="Select DCI stage-budget rows by certificate gates.")
  parser.add_argument("--summary", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--reference-summary", type=Path, default=None)
  parser.add_argument("--max-rotation-residual", type=float, default=None)
  parser.add_argument("--max-translation-residual", type=float, default=None)
  parser.add_argument(
      "--max-rotation-projection-correction", type=float, default=None)
  parser.add_argument(
      "--max-rotation-projection-cost-delta", type=float, default=None)
  parser.add_argument("--max-handoff-cost", type=float, default=None)
  parser.add_argument(
      "--derive-gates-from",
      choices=["manual", "best_observed"],
      default="manual")
  parser.add_argument("--rotation-residual-slack", type=float, default=0.0)
  parser.add_argument("--translation-residual-slack", type=float, default=0.0)
  parser.add_argument(
      "--rotation-projection-correction-slack", type=float, default=0.0)
  parser.add_argument(
      "--rotation-projection-cost-delta-slack", type=float, default=0.0)
  parser.add_argument("--handoff-cost-slack", type=float, default=0.0)
  parser.add_argument(
      "--handoff-reference-relative-slack", type=float, default=0.0)
  parser.add_argument(
      "--projection-cost-reference-relative-slack", type=float, default=0.0)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  gates = {
      "max_rotation_residual": args.max_rotation_residual,
      "max_translation_residual": args.max_translation_residual,
      "max_rotation_projection_correction":
          args.max_rotation_projection_correction,
      "max_rotation_projection_cost_delta":
          args.max_rotation_projection_cost_delta,
      "max_handoff_cost": args.max_handoff_cost,
  }
  gate_source = (
      "best_observed_plus_slack"
      if args.derive_gates_from == "best_observed" else "manual")
  reference_handoff_costs = {}
  if args.reference_summary is not None:
    gate_source = "reference_summary_relative"
    reference_handoff_costs = read_reference_summary(args.reference_summary)
  slacks = {
      "rotation_residual_slack": float(args.rotation_residual_slack),
      "translation_residual_slack": float(args.translation_residual_slack),
      "rotation_projection_correction_slack": float(
          args.rotation_projection_correction_slack),
      "rotation_projection_cost_delta_slack": float(
          args.rotation_projection_cost_delta_slack),
      "handoff_cost_slack": float(args.handoff_cost_slack),
      "handoff_reference_relative_slack": float(
          args.handoff_reference_relative_slack),
      "projection_cost_reference_relative_slack": float(
          args.projection_cost_reference_relative_slack),
  }
  rows = read_summary_rows(args.summary)
  report = analyze_rows(
      rows,
      gates,
      gate_source=gate_source,
      slacks=slacks,
      reference_handoff_costs=reference_handoff_costs)
  write_analysis(report, args.output_dir)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

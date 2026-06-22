#!/usr/bin/env python3
"""Reference-free DCI budget-grid stability gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import apply_dci_downstream_basin_gate as candidate_io


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description=(
          "Reject DCI initialization candidates when the same budget grid "
          "shows that increasing only rotation or only translation budget "
          "still gives a significant handoff/residual improvement."))
  parser.add_argument(
      "--candidate-summary",
      action="append",
      type=Path,
      required=True,
      help="reference_gate_experiment_summary.csv or stage_budget_summary.csv.")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--min-handoff-improvement", type=float, default=1e-3)
  parser.add_argument(
      "--min-rotation-residual-improvement", type=float, default=1e-6)
  parser.add_argument(
      "--min-translation-residual-improvement", type=float, default=1e-6)
  return parser.parse_args(argv)


def _float_or_none(value):
  if value is None or value == "":
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  if math.isfinite(result):
    return result
  return None


def _parse_pair(pair: str) -> tuple[int | None, int | None]:
  if ":" not in str(pair):
    return None, None
  lhs, rhs = str(pair).split(":", 1)
  try:
    return int(float(lhs)), int(float(rhs))
  except ValueError:
    return None, None


def normalize_rows(rows: list[dict]) -> list[dict]:
  normalized_rows = []
  for row in rows:
    normalized = candidate_io.normalize_candidate_row(row)
    rotation_budget, translation_budget = _parse_pair(
        normalized.get("selected_pair", ""))
    if rotation_budget is None:
      rotation_budget = _float_or_none(row.get("rotation_iteration_budget"))
      rotation_budget = None if rotation_budget is None else int(rotation_budget)
    if translation_budget is None:
      translation_budget = _float_or_none(
          row.get("translation_iteration_budget"))
      translation_budget = (
          None if translation_budget is None else int(translation_budget))
    normalized.update({
        "rotation_budget": rotation_budget,
        "translation_budget": translation_budget,
    })
    normalized_rows.append(normalized)
  return normalized_rows


def _same_dataset(lhs: dict, rhs: dict) -> bool:
  return str(lhs.get("dataset", "")) == str(rhs.get("dataset", ""))


def _handoff_improvement(candidate: dict, challenger: dict) -> float:
  return (float(candidate["candidate_handoff_cost"]) -
          float(challenger["candidate_handoff_cost"]))


def _residual_improvement(candidate: dict, challenger: dict,
                          residual_key: str) -> float:
  return float(candidate[residual_key]) - float(challenger[residual_key])


def _axis_challengers(row: dict, rows: list[dict],
                      axis: str) -> list[dict]:
  challengers = []
  for other in rows:
    if other is row or not _same_dataset(row, other):
      continue
    if axis == "rotation":
      if row["translation_budget"] != other["translation_budget"]:
        continue
      if other["rotation_budget"] is None or row["rotation_budget"] is None:
        continue
      if other["rotation_budget"] > row["rotation_budget"]:
        challengers.append(other)
    elif axis == "translation":
      if row["rotation_budget"] != other["rotation_budget"]:
        continue
      if (other["translation_budget"] is None or
          row["translation_budget"] is None):
        continue
      if other["translation_budget"] > row["translation_budget"]:
        challengers.append(other)
    else:
      raise ValueError(f"unknown axis: {axis}")
  return challengers


def _axis_improvement_report(row: dict, challengers: list[dict],
                             axis: str,
                             min_handoff_improvement: float,
                             min_rotation_residual_improvement: float,
                             min_translation_residual_improvement: float
                             ) -> tuple[bool, str, dict | None]:
  if not challengers:
    return False, "budget_boundary", None
  residual_key = (
      "candidate_rotation_residual"
      if axis == "rotation" else "candidate_translation_residual")
  residual_threshold = (
      min_rotation_residual_improvement
      if axis == "rotation" else min_translation_residual_improvement)
  best = None
  best_score = -float("inf")
  for challenger in challengers:
    handoff_gain = _handoff_improvement(row, challenger)
    residual_gain = _residual_improvement(row, challenger, residual_key)
    improves = (handoff_gain > min_handoff_improvement or
                residual_gain > residual_threshold)
    score = max(handoff_gain, residual_gain)
    if improves and score > best_score:
      best = {
          "challenger_pair": challenger.get("selected_pair", ""),
          "challenger_handoff_cost": challenger[
              "candidate_handoff_cost"],
          "handoff_improvement": handoff_gain,
          "residual_improvement": residual_gain,
      }
      best_score = score
  if best is None:
    return False, "stable_against_grid", None
  return True, "axis_improvement", best


def analyze_rows(rows: list[dict],
                 min_handoff_improvement: float = 1e-3,
                 min_rotation_residual_improvement: float = 1e-6,
                 min_translation_residual_improvement: float = 1e-6
                 ) -> list[dict]:
  normalized_rows = normalize_rows(rows)
  output = []
  for row in normalized_rows:
    rotation_challengers = _axis_challengers(row, normalized_rows, "rotation")
    translation_challengers = _axis_challengers(
        row, normalized_rows, "translation")
    rotation_improves, rotation_status, rotation_best = (
        _axis_improvement_report(
            row, rotation_challengers, "rotation",
            min_handoff_improvement,
            min_rotation_residual_improvement,
            min_translation_residual_improvement))
    translation_improves, translation_status, translation_best = (
        _axis_improvement_report(
            row, translation_challengers, "translation",
            min_handoff_improvement,
            min_rotation_residual_improvement,
            min_translation_residual_improvement))
    failures = []
    if rotation_improves:
      failures.append("rotation_axis_improvement")
    if translation_improves:
      failures.append("translation_axis_improvement")
    annotated = dict(row)
    annotated.update({
        "rotation_axis_status": rotation_status,
        "translation_axis_status": translation_status,
        "rotation_axis_best_challenger": (
            "" if rotation_best is None else rotation_best[
                "challenger_pair"]),
        "translation_axis_best_challenger": (
            "" if translation_best is None else translation_best[
                "challenger_pair"]),
        "rotation_axis_handoff_improvement": (
            "" if rotation_best is None else rotation_best[
                "handoff_improvement"]),
        "translation_axis_handoff_improvement": (
            "" if translation_best is None else translation_best[
                "handoff_improvement"]),
        "rotation_axis_residual_improvement": (
            "" if rotation_best is None else rotation_best[
                "residual_improvement"]),
        "translation_axis_residual_improvement": (
            "" if translation_best is None else translation_best[
                "residual_improvement"]),
        "grid_stability_feasible": not failures,
        "grid_stability_failure_reasons": ",".join(failures) or "none",
        "grid_stability_on_budget_boundary": (
            rotation_status == "budget_boundary" or
            translation_status == "budget_boundary"),
    })
    output.append(annotated)
  return output


def select_min_comm_stable(rows: list[dict]) -> dict | None:
  feasible = [row for row in rows if row.get("grid_stability_feasible", False)]
  if not feasible:
    return None
  return min(
      feasible,
      key=lambda row: (
          float(row["candidate_total_comm_mb"]),
          float(row["candidate_handoff_cost"]),
          str(row.get("dataset", "")),
          str(row.get("selected_pair", "")),
      ))


def _write_csv(path: Path, rows: list[dict], selected: dict | None = None):
  selected_id = None if selected is None else id(selected)
  output_rows = []
  fieldnames = []
  for row in rows:
    csv_row = dict(row)
    csv_row["selected"] = bool(id(row) == selected_id)
    output_rows.append(csv_row)
    for key in csv_row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(output_rows)


def _format_float(value) -> str:
  if isinstance(value, float):
    return f"{value:.12g}"
  return str(value)


def _write_markdown(path: Path, rows: list[dict], selected: dict | None, args):
  lines = [
      "# DCI Grid Stability Gate",
      "",
      "Reference-free diagnostic: a row is rejected if another row in the "
      "same DCI budget grid, with only a larger rotation or only a larger "
      "translation budget, still gives a significant handoff or residual "
      "improvement.",
      "",
      f"- min handoff improvement: `{args.min_handoff_improvement}`",
      f"- min rotation residual improvement: "
      f"`{args.min_rotation_residual_improvement}`",
      f"- min translation residual improvement: "
      f"`{args.min_translation_residual_improvement}`",
      "",
  ]
  if selected is None:
    lines.extend(["No grid-stable row was selected.", ""])
  else:
    boundary = "yes" if selected[
        "grid_stability_on_budget_boundary"] else "no"
    lines.extend([
        "Selected row:",
        "",
        "| Dataset | Pair | Comm MB | Handoff cost | Boundary? |",
        "| --- | ---: | ---: | ---: | --- |",
        "| {dataset} | {pair} | {comm} | {cost} | {boundary} |"
        .format(
            dataset=selected.get("dataset", ""),
            pair=selected.get("selected_pair", ""),
            comm=_format_float(selected["candidate_total_comm_mb"]),
            cost=_format_float(selected["candidate_handoff_cost"]),
            boundary=boundary),
        "",
        "Note: a selected row on the budget boundary is not a convergence "
        "proof; it only means this grid has no higher-budget axis challenger.",
        "",
    ])
  lines.extend([
      "All rows:",
      "",
      "| Dataset | Pair | Feasible | Failure reason | Rot axis | Trans axis | "
      "Boundary? |",
      "| --- | ---: | --- | --- | --- | --- | --- |",
  ])
  for row in rows:
    boundary = "yes" if row["grid_stability_on_budget_boundary"] else "no"
    lines.append(
        "| {dataset} | {pair} | {feasible} | {reason} | {rot} | {trans} | "
        "{boundary} |".format(
            dataset=row.get("dataset", ""),
            pair=row.get("selected_pair", ""),
            feasible=row["grid_stability_feasible"],
            reason=row["grid_stability_failure_reasons"],
            rot=row["rotation_axis_status"],
            trans=row["translation_axis_status"],
            boundary=boundary))
  lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, rows: list[dict],
                  selected: dict | None, args) -> dict:
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_csv(output_dir / "dci_grid_stability_gate.csv", rows, selected)
  _write_markdown(output_dir / "dci_grid_stability_gate.md",
                  rows, selected, args)
  report = {
      "model": "dci_grid_stability_gate",
      "candidate_summary_paths": [
          str(path) for path in args.candidate_summary
      ],
      "min_handoff_improvement": float(args.min_handoff_improvement),
      "min_rotation_residual_improvement": float(
          args.min_rotation_residual_improvement),
      "min_translation_residual_improvement": float(
          args.min_translation_residual_improvement),
      "row_count": int(len(rows)),
      "feasible_count": int(sum(
          1 for row in rows if row.get("grid_stability_feasible", False))),
      "selected_row": selected,
      "rows": rows,
  }
  (output_dir / "dci_grid_stability_gate.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return report


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  rows = analyze_rows(
      candidate_io.read_candidate_rows(args.candidate_summary),
      min_handoff_improvement=args.min_handoff_improvement,
      min_rotation_residual_improvement=(
          args.min_rotation_residual_improvement),
      min_translation_residual_improvement=(
          args.min_translation_residual_improvement))
  selected = select_min_comm_stable(rows)
  write_outputs(args.output_dir, rows, selected, args)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

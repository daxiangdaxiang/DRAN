#!/usr/bin/env python3
"""Plan adaptive DCI budget-grid expansion from reference-free stability data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import analyze_dci_grid_stability_gate as stability
from scripts import apply_dci_downstream_basin_gate as candidate_io


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description=(
          "Given a DCI candidate budget grid, select the current grid-stable "
          "row and recommend the next rotation/translation budgets when "
          "the selected row is only accepted on a boundary with large "
          "previous marginal gains."))
  parser.add_argument(
      "--candidate-summary", action="append", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--min-handoff-improvement", type=float, default=1e-3)
  parser.add_argument(
      "--min-rotation-residual-improvement", type=float, default=1e-6)
  parser.add_argument(
      "--min-translation-residual-improvement", type=float, default=1e-6)
  return parser.parse_args(argv)


def analyze_candidate_rows(rows: list[dict],
                           min_handoff_improvement: float = 1e-3,
                           min_rotation_residual_improvement: float = 1e-6,
                           min_translation_residual_improvement: float = 1e-6
                           ) -> list[dict]:
  return stability.analyze_rows(
      rows,
      min_handoff_improvement=min_handoff_improvement,
      min_rotation_residual_improvement=(
          min_rotation_residual_improvement),
      min_translation_residual_improvement=(
          min_translation_residual_improvement))


def select_min_comm_grid_stable(rows: list[dict]) -> dict | None:
  return stability.select_min_comm_stable(rows)


def _same_dataset(lhs: dict, rhs: dict) -> bool:
  return str(lhs.get("dataset", "")) == str(rhs.get("dataset", ""))


def _axis_previous_rows(selected: dict, rows: list[dict],
                        axis: str) -> list[dict]:
  previous = []
  for row in rows:
    if row is selected or not _same_dataset(selected, row):
      continue
    if axis == "rotation":
      if row["translation_budget"] != selected["translation_budget"]:
        continue
      if row["rotation_budget"] < selected["rotation_budget"]:
        previous.append(row)
    elif axis == "translation":
      if row["rotation_budget"] != selected["rotation_budget"]:
        continue
      if row["translation_budget"] < selected["translation_budget"]:
        previous.append(row)
    else:
      raise ValueError(f"unknown axis: {axis}")
  previous.sort(key=lambda row: (
      row["rotation_budget"] if axis == "rotation"
      else row["translation_budget"]),
                reverse=True)
  return previous


def _marginal_gain(selected: dict, previous: dict,
                   axis: str) -> dict:
  residual_key = (
      "candidate_rotation_residual"
      if axis == "rotation" else "candidate_translation_residual")
  budget_key = "rotation_budget" if axis == "rotation" else (
      "translation_budget")
  return {
      "previous_pair": previous["selected_pair"],
      "base_pair": selected["selected_pair"],
      "axis": axis,
      "axis_step": int(selected[budget_key] - previous[budget_key]),
      "handoff_improvement": (
          float(previous["candidate_handoff_cost"]) -
          float(selected["candidate_handoff_cost"])),
      "residual_improvement": (
          float(previous[residual_key]) - float(selected[residual_key])),
  }


def _gain_is_large(gain: dict, axis: str,
                   min_handoff_improvement: float,
                   min_rotation_residual_improvement: float,
                   min_translation_residual_improvement: float) -> bool:
  residual_threshold = (
      min_rotation_residual_improvement
      if axis == "rotation" else min_translation_residual_improvement)
  return (float(gain["handoff_improvement"]) > min_handoff_improvement or
          float(gain["residual_improvement"]) > residual_threshold)


def _pair(rotation_budget: int, translation_budget: int) -> str:
  return f"{int(rotation_budget)}:{int(translation_budget)}"


def plan_expansions_for_selected(
    selected: dict,
    rows: list[dict],
    min_handoff_improvement: float = 1e-3,
    min_rotation_residual_improvement: float = 1e-6,
    min_translation_residual_improvement: float = 1e-6) -> list[dict]:
  recommendations = []
  axis_plans = {}
  for axis in ("rotation", "translation"):
    previous_rows = _axis_previous_rows(selected, rows, axis)
    if not previous_rows:
      continue
    previous = previous_rows[0]
    gain = _marginal_gain(selected, previous, axis)
    if not _gain_is_large(
        gain, axis, min_handoff_improvement,
        min_rotation_residual_improvement,
        min_translation_residual_improvement):
      continue
    if axis == "rotation":
      next_rotation = int(selected["rotation_budget"] + gain["axis_step"])
      next_translation = int(selected["translation_budget"])
      reason = "rotation_boundary_gain"
    else:
      next_rotation = int(selected["rotation_budget"])
      next_translation = int(selected["translation_budget"] + gain["axis_step"])
      reason = "translation_boundary_gain"
    plan = {
        **gain,
        "dataset": selected.get("dataset", ""),
        "recommended_pair": _pair(next_rotation, next_translation),
        "recommended_rotation_budget": next_rotation,
        "recommended_translation_budget": next_translation,
        "reason": reason,
    }
    axis_plans[axis] = plan
    recommendations.append(plan)
  if "rotation" in axis_plans and "translation" in axis_plans:
    recommendations.append({
        "dataset": selected.get("dataset", ""),
        "axis": "combined",
        "previous_pair": selected["selected_pair"],
        "base_pair": selected["selected_pair"],
        "axis_step": "",
        "handoff_improvement": "",
        "residual_improvement": "",
        "recommended_rotation_budget": axis_plans["rotation"][
            "recommended_rotation_budget"],
        "recommended_translation_budget": axis_plans["translation"][
            "recommended_translation_budget"],
        "recommended_pair": _pair(
            axis_plans["rotation"]["recommended_rotation_budget"],
            axis_plans["translation"]["recommended_translation_budget"]),
        "reason": "combined_boundary_gain",
    })
  return recommendations


def _write_csv(path: Path, rows: list[dict]):
  fieldnames = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _format_float(value) -> str:
  if isinstance(value, float):
    return f"{value:.12g}"
  return str(value)


def _write_markdown(path: Path, selected: dict | None,
                    recommendations: list[dict], args):
  lines = [
      "# DCI Adaptive Grid Expansion Plan",
      "",
      "Reference-free continuation policy for a finite DCI budget grid. "
      "If the selected stable row lies on a boundary and the previous segment "
      "still gave a large marginal gain, expand that axis.",
      "",
      f"- min handoff improvement: `{args.min_handoff_improvement}`",
      f"- min rotation residual improvement: "
      f"`{args.min_rotation_residual_improvement}`",
      f"- min translation residual improvement: "
      f"`{args.min_translation_residual_improvement}`",
      "",
  ]
  if selected is None:
    lines.extend(["No grid-stable base row was found.", ""])
  else:
    lines.extend([
        "Base row:",
        "",
        "| Dataset | Pair | Comm MB | Handoff cost |",
        "| --- | ---: | ---: | ---: |",
        "| {dataset} | {pair} | {comm} | {cost} |".format(
            dataset=selected.get("dataset", ""),
            pair=selected.get("selected_pair", ""),
            comm=_format_float(selected["candidate_total_comm_mb"]),
            cost=_format_float(selected["candidate_handoff_cost"])),
        "",
    ])
  if not recommendations:
    lines.extend([
        "No expansion recommended by the current marginal thresholds.",
        "",
    ])
  else:
    lines.extend([
        "Recommended next evaluations:",
        "",
        "| Pair | Reason | Previous | Handoff gain | Residual gain |",
        "| ---: | --- | ---: | ---: | ---: |",
    ])
    for row in recommendations:
      lines.append(
          "| {pair} | {reason} | {previous} | {handoff} | {residual} |"
          .format(
              pair=row["recommended_pair"],
              reason=row["reason"],
              previous=row["previous_pair"],
              handoff=_format_float(row["handoff_improvement"]),
              residual=_format_float(row["residual_improvement"])))
    lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, selected: dict | None,
                  recommendations: list[dict], analyzed_rows: list[dict],
                  args) -> dict:
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_csv(output_dir / "dci_adaptive_grid_expansion_plan.csv",
             recommendations)
  _write_markdown(output_dir / "dci_adaptive_grid_expansion_plan.md",
                  selected, recommendations, args)
  report = {
      "model": "dci_adaptive_grid_expansion_plan",
      "candidate_summary_paths": [
          str(path) for path in args.candidate_summary
      ],
      "min_handoff_improvement": float(args.min_handoff_improvement),
      "min_rotation_residual_improvement": float(
          args.min_rotation_residual_improvement),
      "min_translation_residual_improvement": float(
          args.min_translation_residual_improvement),
      "selected_row": selected,
      "recommendations": recommendations,
      "analyzed_rows": analyzed_rows,
  }
  (output_dir / "dci_adaptive_grid_expansion_plan.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return report


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  analyzed_rows = analyze_candidate_rows(
      candidate_io.read_candidate_rows(args.candidate_summary),
      min_handoff_improvement=args.min_handoff_improvement,
      min_rotation_residual_improvement=(
          args.min_rotation_residual_improvement),
      min_translation_residual_improvement=(
          args.min_translation_residual_improvement))
  selected = select_min_comm_grid_stable(analyzed_rows)
  recommendations = [] if selected is None else plan_expansions_for_selected(
      selected,
      analyzed_rows,
      min_handoff_improvement=args.min_handoff_improvement,
      min_rotation_residual_improvement=(
          args.min_rotation_residual_improvement),
      min_translation_residual_improvement=(
          args.min_translation_residual_improvement))
  write_outputs(args.output_dir, selected, recommendations, analyzed_rows, args)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

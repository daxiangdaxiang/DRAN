#!/usr/bin/env python3
"""Analyze matrix-free DCI convergence sweep summaries."""

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


def _is_monotone_nonincreasing(values: list[float], tolerance: float) -> bool:
  finite = [value for value in values if value == value]
  return all(
      finite[index] <= finite[index - 1] + tolerance
      for index in range(1, len(finite)))


def _dominant_stage(rotation_gap: float, translation_gap: float,
                    tolerance: float) -> str:
  if rotation_gap != rotation_gap and translation_gap != translation_gap:
    return "unknown"
  if translation_gap != translation_gap:
    return "rotation"
  if rotation_gap != rotation_gap:
    return "translation"
  if abs(rotation_gap - translation_gap) <= tolerance:
    return "balanced"
  return "translation" if translation_gap > rotation_gap else "rotation"


def read_summary_rows(summary_path: Path) -> list[dict]:
  with Path(summary_path).open(newline="", encoding="utf-8") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def analyze_rows(rows: list[dict], tolerance: float = 1e-9) -> dict:
  groups: dict[str, list[dict]] = {}
  for row in rows:
    dataset = str(row.get("dataset", ""))
    groups.setdefault(dataset, []).append(row)
  dataset_reports = []
  for dataset, dataset_rows in sorted(groups.items()):
    parsed_rows = []
    for row in dataset_rows:
      budget = int(float(row.get("iteration_budget", 0)))
      rotation_gap = _float_or_nan(row.get("rotation_schur_energy_gap"))
      translation_gap = _float_or_nan(row.get("translation_schur_energy_gap"))
      total_gap = _float_or_nan(row.get("total_schur_energy_gap"))
      handoff_cost = _float_or_nan(row.get("handoff_cost"))
      wall_time_sec = _float_or_nan(row.get("wall_time_sec"))
      parsed_rows.append({
          "dataset": dataset,
          "iteration_budget": budget,
          "rotation_schur_energy_gap": rotation_gap,
          "translation_schur_energy_gap": translation_gap,
          "total_schur_energy_gap": total_gap,
          "handoff_cost": handoff_cost,
          "wall_time_sec": wall_time_sec,
          "dominant_stage": _dominant_stage(
              rotation_gap, translation_gap, tolerance),
      })
    parsed_rows.sort(key=lambda item: item["iteration_budget"])
    total_gaps = [row["total_schur_energy_gap"] for row in parsed_rows]
    handoff_costs = [row["handoff_cost"] for row in parsed_rows]
    rotation_gaps = [row["rotation_schur_energy_gap"] for row in parsed_rows]
    translation_gaps = [
        row["translation_schur_energy_gap"] for row in parsed_rows]
    total_gap_monotone = _is_monotone_nonincreasing(total_gaps, tolerance)
    handoff_monotone = _is_monotone_nonincreasing(handoff_costs, tolerance)

    def best_index(values: list[float]):
      finite = [(index, value) for index, value in enumerate(values)
                if value == value]
      if not finite:
        return None
      return min(finite, key=lambda item: item[1])[0]

    best_total_gap_index = best_index(total_gaps)
    best_handoff_index = best_index(handoff_costs)
    dominant_counts: dict[str, int] = {}
    for row in parsed_rows:
      dominant = row["dominant_stage"]
      dominant_counts[dominant] = int(dominant_counts.get(dominant, 0) + 1)
    dominant_stage = max(
        sorted(dominant_counts.items()), key=lambda item: item[1])[0]
    dataset_reports.append({
        "dataset": dataset,
        "row_count": int(len(parsed_rows)),
        "iteration_budgets": [
            int(row["iteration_budget"]) for row in parsed_rows],
        "total_gap_monotone_nonincreasing": bool(total_gap_monotone),
        "handoff_cost_monotone_nonincreasing": bool(handoff_monotone),
        "rotation_gap_monotone_nonincreasing":
            bool(_is_monotone_nonincreasing(rotation_gaps, tolerance)),
        "translation_gap_monotone_nonincreasing":
            bool(_is_monotone_nonincreasing(translation_gaps, tolerance)),
        "dominant_stage": dominant_stage,
        "dominant_stage_counts": dominant_counts,
        "best_total_gap_budget": (
            None if best_total_gap_index is None
            else int(parsed_rows[best_total_gap_index]["iteration_budget"])),
        "best_handoff_cost_budget": (
            None if best_handoff_index is None
            else int(parsed_rows[best_handoff_index]["iteration_budget"])),
        "nonmonotone_total_gap": bool(not total_gap_monotone),
        "nonmonotone_handoff_cost": bool(not handoff_monotone),
        "rows": parsed_rows,
    })
  return {
      "model": "matrix_free_convergence_analysis",
      "dataset_count": int(len(dataset_reports)),
      "row_count": int(len(rows)),
      "datasets": dataset_reports,
  }


def write_analysis(report: dict, output_dir: Path):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  json_path = output_dir / "matrix_free_convergence_analysis.json"
  json_path.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  lines = ["# Matrix-Free DCI Convergence Analysis", ""]
  for dataset in report["datasets"]:
    flags = []
    if dataset["nonmonotone_total_gap"]:
      flags.append("nonmonotone_total_gap")
    if dataset["nonmonotone_handoff_cost"]:
      flags.append("nonmonotone_handoff_cost")
    flag_text = ", ".join(flags) if flags else "no_nonmonotone_flags"
    lines.extend([
        f"## {dataset['dataset']}",
        "",
        f"- budgets: {dataset['iteration_budgets']}",
        f"- dominant_stage: {dataset['dominant_stage']}",
        f"- total_gap_monotone_nonincreasing: {dataset['total_gap_monotone_nonincreasing']}",
        f"- handoff_cost_monotone_nonincreasing: {dataset['handoff_cost_monotone_nonincreasing']}",
        f"- best_total_gap_budget: {dataset['best_total_gap_budget']}",
        f"- best_handoff_cost_budget: {dataset['best_handoff_cost_budget']}",
        f"- flags: {flag_text}",
        "",
        "| budget | rotation_gap | translation_gap | total_gap | handoff_cost | dominant_stage |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in dataset["rows"]:
      lines.append(
          f"| {row['iteration_budget']} | "
          f"{row['rotation_schur_energy_gap']} | "
          f"{row['translation_schur_energy_gap']} | "
          f"{row['total_schur_energy_gap']} | "
          f"{row['handoff_cost']} | "
          f"{row['dominant_stage']} |")
    lines.append("")
  (output_dir / "matrix_free_convergence_analysis.md").write_text(
      "\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description="Analyze matrix-free DCI convergence sweep CSV files.")
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

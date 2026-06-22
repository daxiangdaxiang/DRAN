#!/usr/bin/env python3
"""Apply a calibrated downstream-basin gate to DCI initialization candidates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


GATE_KEYS = (
    "max_handoff_gap_to_reference",
    "max_rotation_residual",
    "max_translation_residual",
    "max_projection_cost_increase",
)


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description=(
          "Apply thresholds from analyze_dci_downstream_basin_gate.py to "
          "reference-gate or stage-budget DCI candidate summaries."))
  parser.add_argument("--gate-report", type=Path, required=True)
  parser.add_argument(
      "--candidate-summary",
      action="append",
      type=Path,
      required=True,
      help=(
          "CSV with candidate DCI rows. Can be a reference_gate_experiment "
          "summary or a stage_budget_summary; may be repeated."))
  parser.add_argument("--output-dir", type=Path, required=True)
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


def _float_or_inf(value) -> float:
  result = _float_or_none(value)
  return float("inf") if result is None else float(result)


def _first_present(row: dict, *names: str):
  for name in names:
    value = row.get(name)
    if value not in (None, ""):
      return value
  return None


def _read_csv_rows(path: Path) -> list[dict]:
  with Path(path).open(newline="", encoding="utf-8") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def load_gate(path: Path) -> dict[str, float]:
  report = json.loads(Path(path).read_text(encoding="utf-8"))
  selected = report.get("selected_candidate") or report
  gate = {}
  missing = []
  for key in GATE_KEYS:
    value = _float_or_none(selected.get(key))
    if value is None:
      missing.append(key)
    else:
      gate[key] = float(value)
  if missing:
    raise ValueError(f"gate report missing thresholds: {', '.join(missing)}")
  return gate


def read_candidate_rows(paths: list[Path]) -> list[dict]:
  rows = []
  for path in paths:
    for row in _read_csv_rows(path):
      row = dict(row)
      row["source_candidate_summary"] = str(path)
      rows.append(row)
  return rows


def _projection_cost_increase(row: dict) -> float:
  value = _first_present(row, "selected_projection_cost_increase")
  if value is not None:
    return _float_or_inf(value)
  signed_delta = _float_or_none(row.get("rotation_projection_cost_delta"))
  if signed_delta is not None:
    return max(0.0, signed_delta)
  return _float_or_inf(row.get("rotation_abs_projection_cost_delta"))


def _candidate_total_comm_mb(row: dict) -> float:
  selected_total = _float_or_none(row.get("selected_total_comm_mb"))
  if selected_total is not None:
    return float(selected_total)
  total = _float_or_none(row.get("total_comm_mb"))
  if total is not None:
    return float(total)
  rotation = _float_or_none(row.get("rotation_comm_mb")) or 0.0
  translation = _float_or_none(row.get("translation_comm_mb")) or 0.0
  return float(rotation) + float(translation)


def _handoff_gap_to_reference(row: dict) -> float:
  explicit = _float_or_none(row.get("selected_handoff_gap_to_reference"))
  if explicit is not None:
    return float(explicit)
  handoff = _float_or_none(_first_present(
      row, "selected_handoff_cost", "handoff_cost"))
  reference = _float_or_none(row.get("reference_handoff_cost"))
  if handoff is None or reference is None:
    return float("inf")
  return float(handoff) - float(reference)


def normalize_candidate_row(row: dict) -> dict:
  pair = _first_present(row, "selected_pair", "stage_budget_pair") or ""
  handoff_cost = _float_or_inf(_first_present(
      row, "selected_handoff_cost", "handoff_cost"))
  normalized = dict(row)
  normalized.update({
      "selected_pair": str(pair),
      "candidate_handoff_cost": handoff_cost,
      "candidate_handoff_gap_to_reference": _handoff_gap_to_reference(row),
      "candidate_rotation_residual": _float_or_inf(_first_present(
          row, "selected_rotation_residual", "rotation_schur_residual_norm")),
      "candidate_translation_residual": _float_or_inf(_first_present(
          row, "selected_translation_residual",
          "translation_schur_residual_norm")),
      "candidate_projection_cost_increase": _projection_cost_increase(row),
      "candidate_total_comm_mb": _candidate_total_comm_mb(row),
  })
  return normalized


def apply_gate_to_candidate_rows(rows: list[dict],
                                 gate: dict[str, float]) -> list[dict]:
  output = []
  for row in rows:
    normalized = normalize_candidate_row(row)
    failures = []
    if (normalized["candidate_handoff_gap_to_reference"] >
        gate["max_handoff_gap_to_reference"]):
      failures.append("handoff_gap_to_reference")
    if normalized["candidate_rotation_residual"] > gate[
        "max_rotation_residual"]:
      failures.append("rotation_residual")
    if normalized["candidate_translation_residual"] > gate[
        "max_translation_residual"]:
      failures.append("translation_residual")
    if normalized["candidate_projection_cost_increase"] > gate[
        "max_projection_cost_increase"]:
      failures.append("projection_cost_increase")
    normalized.update({
        "gate_feasible": not failures,
        "gate_failure_reasons": ",".join(failures) or "none",
    })
    output.append(normalized)
  return output


def select_min_comm_feasible(rows: list[dict]) -> dict | None:
  feasible = [row for row in rows if row.get("gate_feasible", False)]
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


def _write_markdown(path: Path, rows: list[dict], selected: dict | None,
                    gate: dict[str, float]):
  lines = [
      "# Downstream Basin Gate Application",
      "",
      "Applied calibrated downstream-basin thresholds to DCI initialization "
      "candidate rows.",
      "",
      "Gate thresholds:",
      "",
      "| Metric | Threshold |",
      "| --- | ---: |",
  ]
  for key in GATE_KEYS:
    lines.append(f"| `{key}` | `{_format_float(gate[key])}` |")
  lines.append("")
  if selected is None:
    lines.extend(["No candidate passed the gate.", ""])
  else:
    lines.extend([
        "Selected candidate:",
        "",
        "| Dataset | Pair | Comm MB | Handoff gap | Rot residual | "
        "Trans residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| {dataset} | {pair} | {comm} | {gap} | {rot} | {trans} |"
        .format(
            dataset=selected.get("dataset", ""),
            pair=selected.get("selected_pair", ""),
            comm=_format_float(selected["candidate_total_comm_mb"]),
            gap=_format_float(
                selected["candidate_handoff_gap_to_reference"]),
            rot=_format_float(selected["candidate_rotation_residual"]),
            trans=_format_float(
                selected["candidate_translation_residual"])),
        "",
    ])
  lines.extend([
      "All candidates:",
      "",
      "| Dataset | Pair | Feasible | Failure reason | Comm MB | Handoff gap | "
      "Rot residual | Trans residual |",
      "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
  ])
  for row in rows:
    lines.append(
        "| {dataset} | {pair} | {feasible} | {reason} | {comm} | {gap} | "
        "{rot} | {trans} |".format(
            dataset=row.get("dataset", ""),
            pair=row.get("selected_pair", ""),
            feasible=row.get("gate_feasible", ""),
            reason=row.get("gate_failure_reasons", ""),
            comm=_format_float(row["candidate_total_comm_mb"]),
            gap=_format_float(row["candidate_handoff_gap_to_reference"]),
            rot=_format_float(row["candidate_rotation_residual"]),
            trans=_format_float(row["candidate_translation_residual"])))
  lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, rows: list[dict],
                  selected: dict | None, gate: dict[str, float], args) -> dict:
  output_dir.mkdir(parents=True, exist_ok=True)
  _write_csv(output_dir / "downstream_basin_gate_application.csv",
             rows, selected)
  _write_markdown(
      output_dir / "downstream_basin_gate_application.md",
      rows, selected, gate)
  report = {
      "model": "dci_downstream_basin_gate_application",
      "gate_report_path": str(args.gate_report),
      "candidate_summary_paths": [
          str(path) for path in args.candidate_summary
      ],
      "gate": gate,
      "row_count": int(len(rows)),
      "feasible_count": int(sum(
          1 for row in rows if row.get("gate_feasible", False))),
      "selected_row": selected,
      "rows": rows,
  }
  (output_dir / "downstream_basin_gate_application.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return report


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  gate = load_gate(args.gate_report)
  rows = apply_gate_to_candidate_rows(
      read_candidate_rows(args.candidate_summary), gate)
  selected = select_min_comm_feasible(rows)
  write_outputs(args.output_dir, rows, selected, gate, args)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

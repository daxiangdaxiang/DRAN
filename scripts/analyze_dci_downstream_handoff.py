#!/usr/bin/env python3
"""Compare downstream optimization after different DCI initializations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _parse_label_path(value: str) -> tuple[str, Path]:
  if "=" not in value:
    raise argparse.ArgumentTypeError("expected LABEL=PATH")
  label, path = value.split("=", 1)
  label = label.strip()
  if not label:
    raise argparse.ArgumentTypeError("label must be nonempty")
  return label, Path(path)


def _read_csv_rows(path: Path) -> list[dict]:
  with path.open(encoding="utf-8") as handle:
    return list(csv.DictReader(handle))


def _as_float(value, default: float = float("nan")) -> float:
  try:
    if value in (None, ""):
      return default
    return float(value)
  except (TypeError, ValueError):
    return default


def _as_int(value, default: int = 0) -> int:
  try:
    if value in (None, ""):
      return default
    return int(float(value))
  except (TypeError, ValueError):
    return default


def _delta(lhs: float, rhs: float) -> float:
  return round(float(lhs) - float(rhs), 12)


def _summary_by_dataset(root: Path) -> dict[str, dict]:
  rows = _read_csv_rows(root / "summary.csv")
  return {str(row["dataset"]): row for row in rows}


def _iteration_endpoints(summary_row: dict) -> tuple[dict, dict]:
  iter_path = Path(summary_row["iter_summary_path"])
  rows = _read_csv_rows(iter_path)
  if not rows:
    raise ValueError(f"empty iteration summary: {iter_path}")
  return rows[0], rows[-1]


def _init_summary_by_dataset(path: Path) -> dict[str, dict]:
  return {str(row["dataset"]): row for row in _read_csv_rows(path)}


def _init_value(row: dict, *names: str) -> str:
  for name in names:
    value = row.get(name)
    if value not in (None, ""):
      return value
  return ""


def compare_handoff_roots(
    baseline_root: Path,
    case_roots: dict[str, Path],
    init_summary_paths: dict[str, Path] | None = None) -> list[dict]:
  baseline_rows = _summary_by_dataset(Path(baseline_root))
  init_summary_paths = init_summary_paths or {}
  init_summaries = {
      label: _init_summary_by_dataset(path)
      for label, path in init_summary_paths.items()
  }
  output_rows = []
  for label, root in case_roots.items():
    case_rows = _summary_by_dataset(Path(root))
    for dataset, case_summary in sorted(case_rows.items()):
      if dataset not in baseline_rows:
        continue
      baseline_summary = baseline_rows[dataset]
      baseline_initial, baseline_final = _iteration_endpoints(
          baseline_summary)
      case_initial, case_final = _iteration_endpoints(case_summary)
      init_row = init_summaries.get(label, {}).get(dataset, {})

      baseline_initial_cost = _as_float(baseline_initial.get("global_cost"))
      case_initial_cost = _as_float(case_initial.get("global_cost"))
      baseline_final_cost = _as_float(baseline_final.get("global_cost"))
      case_final_cost = _as_float(case_final.get("global_cost"))
      baseline_gradient = _as_float(baseline_final.get("gradient"))
      case_gradient = _as_float(case_final.get("gradient"))
      downstream_comm_mb = _as_float(
          case_summary.get("total_comm_mb"),
          _as_float(case_final.get("cumulative_comm_mb"), 0.0))
      baseline_downstream_comm_mb = _as_float(
          baseline_summary.get("total_comm_mb"),
          _as_float(baseline_final.get("cumulative_comm_mb"), 0.0))
      init_comm_mb = _as_float(
          _init_value(init_row, "selected_total_comm_mb", "total_comm_mb"),
          0.0)
      init_handoff_cost = _as_float(
          _init_value(init_row, "selected_handoff_cost", "handoff_cost"),
          case_initial_cost)
      reference_handoff_cost = _as_float(
          init_row.get("reference_handoff_cost"))
      selected_handoff_gap_to_reference = _as_float(
          init_row.get("selected_handoff_gap_to_reference"))
      selected_projection_cost_increase = _as_float(
          init_row.get("selected_projection_cost_increase"))
      selected_rotation_residual = _as_float(
          init_row.get("selected_rotation_residual"))
      selected_translation_residual = _as_float(
          init_row.get("selected_translation_residual"))

      output_rows.append({
          "dataset": dataset,
          "case": label,
          "case_root": str(root),
          "selected_pair": _init_value(init_row, "selected_pair"),
          "init_handoff_cost": init_handoff_cost,
          "reference_handoff_cost": reference_handoff_cost,
          "selected_handoff_gap_to_reference": (
              selected_handoff_gap_to_reference),
          "selected_projection_cost_increase": (
              selected_projection_cost_increase),
          "selected_rotation_residual": selected_rotation_residual,
          "selected_translation_residual": selected_translation_residual,
          "baseline_initial_cost": baseline_initial_cost,
          "case_initial_cost": case_initial_cost,
          "initial_cost_gap_to_baseline": _delta(
              case_initial_cost, baseline_initial_cost),
          "baseline_final_cost": baseline_final_cost,
          "case_final_cost": case_final_cost,
          "final_cost_gap_to_baseline": _delta(
              case_final_cost, baseline_final_cost),
          "baseline_gradient": baseline_gradient,
          "case_gradient": case_gradient,
          "gradient_gap_to_baseline": _delta(case_gradient,
                                             baseline_gradient),
          "baseline_downstream_comm_mb": baseline_downstream_comm_mb,
          "downstream_comm_mb": downstream_comm_mb,
          "init_comm_mb": init_comm_mb,
          "total_comm_with_init_mb": round(init_comm_mb + downstream_comm_mb,
                                           12),
          "baseline_wall_time_sec": _as_float(
              baseline_summary.get("wall_time_sec"), 0.0),
          "case_wall_time_sec": _as_float(
              case_summary.get("wall_time_sec"), 0.0),
          "baseline_solver_time_per_node_sec": _as_float(
              baseline_summary.get("solver_time_per_node_sec"), 0.0),
          "case_solver_time_per_node_sec": _as_float(
              case_summary.get("solver_time_per_node_sec"), 0.0),
          "baseline_final_iter": _as_int(baseline_final.get("iter")),
          "case_final_iter": _as_int(case_final.get("iter")),
      })
  return output_rows


def _write_csv(path: Path, rows: list[dict]):
  path.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = []
  for row in rows:
    for key in row:
      if key not in fieldnames:
        fieldnames.append(key)
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _format_float(value) -> str:
  if isinstance(value, float):
    return f"{value:.12g}"
  return str(value)


def _write_markdown(path: Path, rows: list[dict], baseline_root: Path):
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = [
      "# DCI Downstream Handoff Audit",
      "",
      f"Baseline root: `{baseline_root}`",
      "",
      "| Dataset | Case | Init pair | Init cost gap | Ref gap | Rot residual | "
      "Trans residual | Final cost gap | Gradient gap | Init MB | "
      "Downstream MB | Total MB |",
      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
      "---: | ---: | ---: |",
  ]
  for row in rows:
    lines.append(
        "| {dataset} | {case} | {selected_pair} | {init_gap} | "
        "{ref_gap} | {rot_res} | {trans_res} | {final_gap} | {grad_gap} | "
        "{init_mb} | {down_mb} | {total_mb} |"
        .format(
            dataset=row["dataset"],
            case=row["case"],
            selected_pair=row["selected_pair"],
            init_gap=_format_float(row["initial_cost_gap_to_baseline"]),
            ref_gap=_format_float(
                row["selected_handoff_gap_to_reference"]),
            rot_res=_format_float(row["selected_rotation_residual"]),
            trans_res=_format_float(row["selected_translation_residual"]),
            final_gap=_format_float(row["final_cost_gap_to_baseline"]),
            grad_gap=_format_float(row["gradient_gap_to_baseline"]),
            init_mb=_format_float(row["init_comm_mb"]),
            down_mb=_format_float(row["downstream_comm_mb"]),
            total_mb=_format_float(row["total_comm_with_init_mb"])))
  lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline-root", required=True)
  parser.add_argument("--case", action="append", type=_parse_label_path,
                      required=True, help="LABEL=downstream_result_root")
  parser.add_argument("--init-summary", action="append",
                      type=_parse_label_path, default=[],
                      help="LABEL=reference_gate_or_selected_init_summary.csv")
  parser.add_argument("--output-csv", required=True)
  parser.add_argument("--output-md", default=None)
  args = parser.parse_args(argv)

  case_roots = dict(args.case)
  init_summary_paths = dict(args.init_summary)
  rows = compare_handoff_roots(
      baseline_root=Path(args.baseline_root),
      case_roots=case_roots,
      init_summary_paths=init_summary_paths)
  _write_csv(Path(args.output_csv), rows)
  if args.output_md:
    _write_markdown(Path(args.output_md), rows, Path(args.baseline_root))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

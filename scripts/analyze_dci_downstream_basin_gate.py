#!/usr/bin/env python3
"""Calibrate DCI initialization gates against downstream basin labels."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


BASIN_GATE_METRICS = (
    ("max_handoff_gap_to_reference", "selected_handoff_gap_to_reference"),
    ("max_rotation_residual", "selected_rotation_residual"),
    ("max_translation_residual", "selected_translation_residual"),
    ("max_projection_cost_increase", "selected_projection_cost_increase"),
)


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description=(
          "Search initialization-stage thresholds that predict whether a "
          "fixed downstream optimizer enters the CCI-initialized basin."))
  parser.add_argument(
      "--handoff-comparison",
      action="append",
      type=Path,
      required=True,
      help="handoff_comparison.csv from analyze_dci_downstream_handoff.py.")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--max-final-cost-gap", type=float, default=0.01)
  parser.add_argument("--max-gradient-gap", type=float, default=None)
  parser.add_argument("--max-false-positive", type=int, default=0)
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


def _read_csv_rows(path: Path) -> list[dict]:
  with Path(path).open(newline="", encoding="utf-8") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def read_handoff_rows(paths: list[Path]) -> list[dict]:
  rows = []
  for path in paths:
    for row in _read_csv_rows(path):
      row = dict(row)
      row["source_handoff_comparison"] = str(path)
      rows.append(row)
  return rows


def _abs_or_inf(value) -> float:
  result = _float_or_none(value)
  return float("inf") if result is None else abs(float(result))


def annotate_downstream_basin_labels(
    rows: list[dict],
    max_final_cost_gap: float,
    max_gradient_gap: float | None = None) -> list[dict]:
  annotated = []
  for row in rows:
    final_gap = _abs_or_inf(row.get("final_cost_gap_to_baseline"))
    gradient_gap = _abs_or_inf(row.get("gradient_gap_to_baseline"))
    reasons = []
    if final_gap > float(max_final_cost_gap):
      reasons.append("final_cost_gap")
    if max_gradient_gap is not None and gradient_gap > float(max_gradient_gap):
      reasons.append("gradient_gap")
    feasible = not reasons
    annotated_row = dict(row)
    annotated_row.update({
        "downstream_basin_final_cost_gap_abs": final_gap,
        "downstream_basin_gradient_gap_abs": gradient_gap,
        "downstream_basin_max_final_cost_gap": float(max_final_cost_gap),
        "downstream_basin_max_gradient_gap": (
            "" if max_gradient_gap is None else float(max_gradient_gap)),
        "downstream_basin_feasible": bool(feasible),
        "downstream_basin_failure_reasons": ",".join(reasons) or "none",
    })
    annotated.append(annotated_row)
  return annotated


def metric_keys() -> tuple[str, ...]:
  return tuple(key for key, _ in BASIN_GATE_METRICS)


def _gate_metrics(row: dict) -> dict[str, float]:
  return {
      metric_key: _float_or_inf(row.get(row_key))
      for metric_key, row_key in BASIN_GATE_METRICS
  }


def candidate_thresholds_from_feasible_rows(rows: list[dict]) -> list[dict]:
  candidates = []
  feasible_metrics = []
  for row in rows:
    if not row.get("downstream_basin_feasible", False):
      continue
    thresholds = _gate_metrics(row)
    if not all(math.isfinite(value) for value in thresholds.values()):
      continue
    feasible_metrics.append(thresholds)
    candidates.append({
        "source_dataset": str(row.get("dataset", "")),
        "source_case": str(row.get("case", "")),
        "source_pair": str(row.get("selected_pair", "")),
        **thresholds,
    })
  per_metric_values = [
      sorted({metrics[key] for metrics in feasible_metrics})
      for key in metric_keys()
  ]
  if all(per_metric_values):
    for values in itertools.product(*per_metric_values):
      candidates.append({
          "source_dataset": "mixed_downstream_basin",
          "source_case": "metric_grid",
          "source_pair": "metric_grid",
          **dict(zip(metric_keys(), values)),
      })
  unique = {}
  for candidate in candidates:
    key = tuple(candidate[name] for name in metric_keys())
    unique.setdefault(key, candidate)
  return list(unique.values())


def gate_passes(row: dict, thresholds: dict[str, float]) -> bool:
  metrics = _gate_metrics(row)
  for key, threshold in thresholds.items():
    value = metrics[key]
    if not (math.isfinite(value) and value <= float(threshold)):
      return False
  return True


def score_candidate(candidate: dict, rows: list[dict]) -> dict:
  thresholds = {key: float(candidate[key]) for key in metric_keys()}
  tp = fp = tn = fn = 0
  accepted = []
  rejected = []
  for row in rows:
    label = bool(row.get("downstream_basin_feasible", False))
    predicted = gate_passes(row, thresholds)
    if label and predicted:
      tp += 1
    elif (not label) and predicted:
      fp += 1
    elif label and not predicted:
      fn += 1
    else:
      tn += 1
    target = accepted if predicted else rejected
    target.append({
        "dataset": row.get("dataset", ""),
        "case": row.get("case", ""),
        "selected_pair": row.get("selected_pair", ""),
        "downstream_basin_feasible": label,
        "final_cost_gap_to_baseline": row.get("final_cost_gap_to_baseline", ""),
        "gradient_gap_to_baseline": row.get("gradient_gap_to_baseline", ""),
    })
  precision = "" if (tp + fp) == 0 else tp / float(tp + fp)
  recall = "" if (tp + fn) == 0 else tp / float(tp + fn)
  return {
      **candidate,
      "true_positive": int(tp),
      "false_positive": int(fp),
      "false_negative": int(fn),
      "true_negative": int(tn),
      "accepted_count": int(tp + fp),
      "downstream_basin_feasible_count": int(tp + fn),
      "precision": precision,
      "recall": recall,
      "accepted_rows": accepted,
      "rejected_rows": rejected,
  }


def select_candidate(scored: list[dict],
                     max_false_positive: int) -> dict | None:
  eligible = [
      row for row in scored
      if int(row["false_positive"]) <= int(max_false_positive)
  ]
  if not eligible:
    return None
  return max(
      eligible,
      key=lambda row: (
          int(row["true_positive"]),
          -int(row["false_positive"]),
          *[-float(row[key]) for key in metric_keys()],
      ))


def _write_csv(path: Path, rows: list[dict]):
  fieldnames = []
  for row in rows:
    for key in row:
      if key in {"accepted_rows", "rejected_rows"}:
        continue
      if key not in fieldnames:
        fieldnames.append(key)
  if "selected" not in fieldnames:
    fieldnames.append("selected")
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)


def _format_float(value) -> str:
  if isinstance(value, float):
    return f"{value:.12g}"
  return str(value)


def _write_markdown(path: Path, selected: dict | None, rows: list[dict], args):
  lines = [
      "# Downstream Basin Gate",
      "",
      "A DCI row is labeled feasible only if the same downstream optimizer "
      "stays within the configured CCI-baseline final-cost and gradient gaps.",
      "",
      f"- max final cost gap: `{args.max_final_cost_gap}`",
      f"- max gradient gap: `{args.max_gradient_gap}`",
      f"- max false positives: `{args.max_false_positive}`",
      "",
  ]
  if selected is None:
    lines.extend(["No zero-false-positive candidate was selected.", ""])
  else:
    lines.extend([
        "Selected thresholds:",
        "",
        "| Metric | Threshold |",
        "| --- | ---: |",
    ])
    for key in metric_keys():
      lines.append(f"| `{key}` | `{_format_float(selected[key])}` |")
    lines.extend([
        "",
        "Accepted rows:",
        "",
        "| Dataset | Case | Pair | Basin label | Final gap | Gradient gap |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ])
    for row in selected["accepted_rows"]:
      lines.append(
          "| {dataset} | {case} | {pair} | {label} | {final_gap} | "
          "{grad_gap} |".format(
              dataset=row["dataset"],
              case=row["case"],
              pair=row["selected_pair"],
              label=row["downstream_basin_feasible"],
              final_gap=row["final_cost_gap_to_baseline"],
              grad_gap=row["gradient_gap_to_baseline"]))
    lines.append("")
  lines.extend([
      "All labeled rows:",
      "",
      "| Dataset | Case | Pair | Basin feasible | Failure reason | Final gap | "
      "Gradient gap |",
      "| --- | --- | ---: | --- | --- | ---: | ---: |",
  ])
  for row in rows:
    lines.append(
        "| {dataset} | {case} | {pair} | {feasible} | {reason} | "
        "{final_gap} | {grad_gap} |".format(
            dataset=row.get("dataset", ""),
            case=row.get("case", ""),
            pair=row.get("selected_pair", ""),
            feasible=row.get("downstream_basin_feasible", ""),
            reason=row.get("downstream_basin_failure_reasons", ""),
            final_gap=row.get("final_cost_gap_to_baseline", ""),
            grad_gap=row.get("gradient_gap_to_baseline", "")))
  lines.append("")
  path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, scored: list[dict],
                  selected: dict | None, rows: list[dict], args) -> dict:
  output_dir.mkdir(parents=True, exist_ok=True)
  selected_id = None if selected is None else id(selected)
  csv_rows = []
  for row in scored:
    csv_row = dict(row)
    csv_row.pop("accepted_rows", None)
    csv_row.pop("rejected_rows", None)
    csv_row["selected"] = bool(id(row) == selected_id)
    csv_rows.append(csv_row)
  _write_csv(output_dir / "downstream_basin_gate_summary.csv", csv_rows)
  _write_csv(output_dir / "downstream_basin_labeled_rows.csv", rows)
  _write_markdown(
      output_dir / "downstream_basin_gate_report.md", selected, rows, args)
  report = {
      "model": "dci_downstream_basin_gate",
      "handoff_comparison_paths": [
          str(path) for path in args.handoff_comparison
      ],
      "row_count": int(len(rows)),
      "candidate_count": int(len(scored)),
      "downstream_basin_feasible_count": int(sum(
          1 for row in rows if row.get("downstream_basin_feasible", False))),
      "max_final_cost_gap": float(args.max_final_cost_gap),
      "max_gradient_gap": args.max_gradient_gap,
      "max_false_positive": int(args.max_false_positive),
      "selected_candidate": selected,
      "candidates": scored,
  }
  (output_dir / "downstream_basin_gate_report.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return report


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  rows = annotate_downstream_basin_labels(
      read_handoff_rows(args.handoff_comparison),
      max_final_cost_gap=args.max_final_cost_gap,
      max_gradient_gap=args.max_gradient_gap)
  candidates = candidate_thresholds_from_feasible_rows(rows)
  scored = [score_candidate(candidate, rows) for candidate in candidates]
  selected = select_candidate(scored, args.max_false_positive)
  write_outputs(args.output_dir, scored, selected, rows, args)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

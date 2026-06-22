#!/usr/bin/env python3
"""Calibrate DCI proxy stopping gates against CCI-reference gate labels."""

from __future__ import annotations

import argparse
import csv
import itertools
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

PROXY_METRIC_PROFILES = {
    "raw": (
        ("max_rotation_residual", "rotation_schur_residual_norm"),
        ("max_translation_residual", "translation_schur_residual_norm"),
        ("max_rotation_projection_correction",
         "rotation_max_projection_correction_norm"),
        ("max_rotation_projection_cost_increase", None),
    ),
    "schur_energy": (
        ("max_rotation_schur_energy_gap", "rotation_schur_energy_gap"),
        ("max_translation_schur_energy_gap", "translation_schur_energy_gap"),
        ("max_rotation_projection_correction",
         "rotation_max_projection_correction_norm"),
        ("max_rotation_projection_cost_increase", None),
    ),
    "schur_energy_ratio": (
        ("max_rotation_schur_energy_ratio", None),
        ("max_translation_schur_energy_ratio", None),
        ("max_rotation_projection_correction",
         "rotation_max_projection_correction_norm"),
        ("max_rotation_projection_cost_increase", None),
    ),
}


def parse_args(argv: list[str] | None = None):
  parser = argparse.ArgumentParser(
      description=(
          "Compare reference-free DCI proxy gates against CCI-reference "
          "handoff/projection labels from existing stage-budget rows."))
  parser.add_argument(
      "--stage-summary",
      action="append",
      type=Path,
      required=True,
      help="stage_budget_summary.csv. Can be repeated.")
  parser.add_argument("--reference-summary", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument(
      "--handoff-reference-relative-slack", type=float, default=0.1)
  parser.add_argument(
      "--projection-cost-reference-relative-slack", type=float, default=0.1)
  parser.add_argument("--max-false-positive", type=int, default=0)
  parser.add_argument(
      "--proxy-metric-profile",
      choices=sorted(PROXY_METRIC_PROFILES),
      default="raw",
      help=(
          "Proxy metric family used for threshold search. 'raw' uses "
          "Schur residual norms; 'schur_energy' uses Schur energy gaps; "
          "'schur_energy_ratio' divides Schur energy gaps by the current "
          "handoff cost."))
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


def _float_or_inf(value):
  result = _float_or_none(value)
  return float("inf") if result is None else float(result)


def _read_csv_rows(path: Path) -> list[dict]:
  with Path(path).open(newline="", encoding="utf-8") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def read_reference_handoff_costs(path: Path) -> dict[str, float]:
  references: dict[str, float] = {}
  with Path(path).open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
      dataset = str(row.get("dataset", ""))
      if not dataset:
        continue
      for column in REFERENCE_HANDOFF_COST_COLUMNS:
        value = _float_or_none(row.get(column))
        if value is not None:
          references[dataset] = float(value)
          break
  return references


def read_stage_rows(paths: list[Path]) -> list[dict]:
  rows = []
  for path in paths:
    for row in _read_csv_rows(path):
      row = dict(row)
      row["source_stage_summary"] = str(path)
      rows.append(row)
  return rows


def projection_cost_increase(row: dict) -> float:
  signed_delta = _float_or_none(row.get("rotation_projection_cost_delta"))
  abs_delta = _float_or_none(row.get("rotation_abs_projection_cost_delta"))
  if signed_delta is not None:
    return max(0.0, float(signed_delta))
  if abs_delta is not None:
    return float(abs_delta)
  return float("inf")


def _ratio_or_inf(numerator, denominator) -> float:
  numerator_value = _float_or_none(numerator)
  denominator_value = _float_or_none(denominator)
  if numerator_value is None or denominator_value is None:
    return float("inf")
  if denominator_value <= 0.0:
    return float("inf")
  return float(numerator_value) / float(denominator_value)


def annotate_reference_labels(
    rows: list[dict],
    references: dict[str, float],
    handoff_reference_relative_slack: float,
    projection_cost_reference_relative_slack: float) -> list[dict]:
  annotated = []
  for row in rows:
    dataset = str(row.get("dataset", ""))
    reference = references.get(dataset)
    handoff_cost = _float_or_none(row.get("handoff_cost"))
    projection_increase = projection_cost_increase(row)
    feasible = False
    reasons = []
    max_handoff = None
    max_projection = None
    if reference is None:
      reasons.append("missing_reference")
    else:
      max_handoff = reference * (1.0 + handoff_reference_relative_slack)
      max_projection = reference * projection_cost_reference_relative_slack
      if handoff_cost is None or handoff_cost > max_handoff:
        reasons.append("handoff_cost")
      if (not math.isfinite(projection_increase) or
          projection_increase > max_projection):
        reasons.append("rotation_projection_cost")
    feasible = not reasons
    annotated_row = dict(row)
    annotated_row.update({
        "reference_handoff_cost": (
            "" if reference is None else float(reference)),
        "reference_gate_max_handoff_cost": (
            "" if max_handoff is None else float(max_handoff)),
        "reference_gate_max_rotation_projection_cost_delta": (
            "" if max_projection is None else float(max_projection)),
        "reference_gate_rotation_projection_cost_increase":
            float(projection_increase),
        "reference_gate_feasible": bool(feasible),
        "reference_gate_failure_reasons": ",".join(reasons) or "none",
    })
    annotated.append(annotated_row)
  return annotated


def proxy_metric_keys(profile: str) -> tuple[str, ...]:
  return tuple(key for key, _ in PROXY_METRIC_PROFILES[profile])


def _proxy_metrics(row: dict, profile: str = "raw") -> dict[str, float]:
  metrics = {}
  for metric_key, row_key in PROXY_METRIC_PROFILES[profile]:
    if metric_key == "max_rotation_projection_cost_increase":
      metrics[metric_key] = projection_cost_increase(row)
    elif metric_key == "max_rotation_schur_energy_ratio":
      metrics[metric_key] = _ratio_or_inf(
          row.get("rotation_schur_energy_gap"), row.get("handoff_cost"))
    elif metric_key == "max_translation_schur_energy_ratio":
      metrics[metric_key] = _ratio_or_inf(
          row.get("translation_schur_energy_gap"), row.get("handoff_cost"))
    else:
      metrics[metric_key] = _float_or_inf(row.get(row_key))
  return metrics


def proxy_passes(row: dict, thresholds: dict[str, float],
                 profile: str = "raw") -> bool:
  metrics = _proxy_metrics(row, profile)
  for key, threshold in thresholds.items():
    value = metrics[key]
    if not (math.isfinite(value) and value <= threshold):
      return False
  return True


def candidate_thresholds_from_feasible_rows(
    rows: list[dict],
    profile: str = "raw") -> list[dict]:
  candidates = []
  finite_feasible_metrics = []
  for row in rows:
    if not row.get("reference_gate_feasible", False):
      continue
    thresholds = _proxy_metrics(row, profile)
    if not all(math.isfinite(value) for value in thresholds.values()):
      continue
    finite_feasible_metrics.append(thresholds)
    candidates.append({
        "source_dataset": str(row.get("dataset", "")),
        "source_pair": str(row.get("stage_budget_pair", "")),
        **thresholds,
    })
  per_metric_values = [
      sorted({metrics[key] for metrics in finite_feasible_metrics})
      for key in proxy_metric_keys(profile)
  ]
  if all(per_metric_values):
    for values in itertools.product(*per_metric_values):
      candidates.append({
          "source_dataset": "mixed_feasible_proxy",
          "source_pair": "metric_grid",
          **dict(zip(proxy_metric_keys(profile), values)),
      })
  unique = {}
  for candidate in candidates:
    key = tuple(candidate[name] for name in proxy_metric_keys(profile))
    unique.setdefault(key, candidate)
  return list(unique.values())


def score_candidate(candidate: dict, rows: list[dict],
                    profile: str = "raw") -> dict:
  thresholds = {
      key: float(candidate[key])
      for key in proxy_metric_keys(profile)
  }
  tp = fp = tn = fn = 0
  accepted = []
  rejected = []
  for row in rows:
    reference_feasible = bool(row.get("reference_gate_feasible", False))
    proxy_feasible = proxy_passes(row, thresholds, profile)
    if reference_feasible and proxy_feasible:
      tp += 1
    elif (not reference_feasible) and proxy_feasible:
      fp += 1
    elif reference_feasible and not proxy_feasible:
      fn += 1
    else:
      tn += 1
    target = accepted if proxy_feasible else rejected
    target.append({
        "dataset": row.get("dataset", ""),
        "stage_budget_pair": row.get("stage_budget_pair", ""),
        "reference_gate_feasible": reference_feasible,
    })
  precision = None if (tp + fp) == 0 else tp / float(tp + fp)
  recall = None if (tp + fn) == 0 else tp / float(tp + fn)
  return {
      **candidate,
      "true_positive": int(tp),
      "false_positive": int(fp),
      "false_negative": int(fn),
      "true_negative": int(tn),
      "accepted_count": int(tp + fp),
      "reference_feasible_count": int(tp + fn),
      "precision": "" if precision is None else float(precision),
      "recall": "" if recall is None else float(recall),
      "accepted_rows": accepted,
      "rejected_rows": rejected,
  }


def select_candidate(scored: list[dict], max_false_positive: int,
                     profile: str = "raw") -> dict | None:
  eligible = [
      row for row in scored
      if int(row["false_positive"]) <= int(max_false_positive)
  ]
  if not eligible:
    return None
  metric_keys = proxy_metric_keys(profile)
  return max(
      eligible,
      key=lambda row: (
          int(row["true_positive"]),
          -int(row["false_positive"]),
          *[-float(row[key]) for key in metric_keys],
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
  _write_csv(output_dir / "proxy_gate_calibration_summary.csv", csv_rows)
  report = {
      "model": "dci_proxy_gate_calibration",
      "stage_summary_paths": [str(path) for path in args.stage_summary],
      "reference_summary_path": str(args.reference_summary),
      "row_count": int(len(rows)),
      "candidate_count": int(len(scored)),
      "proxy_metric_profile": str(args.proxy_metric_profile),
      "reference_feasible_count": int(sum(
          1 for row in rows if row.get("reference_gate_feasible", False))),
      "max_false_positive": int(args.max_false_positive),
      "handoff_reference_relative_slack": float(
          args.handoff_reference_relative_slack),
      "projection_cost_reference_relative_slack": float(
          args.projection_cost_reference_relative_slack),
      "selected_candidate": selected,
      "candidates": scored,
  }
  (output_dir / "proxy_gate_calibration_report.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return report


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  references = read_reference_handoff_costs(args.reference_summary)
  rows = annotate_reference_labels(
      read_stage_rows(args.stage_summary),
      references,
      handoff_reference_relative_slack=(
          args.handoff_reference_relative_slack),
      projection_cost_reference_relative_slack=(
          args.projection_cost_reference_relative_slack))
  candidates = candidate_thresholds_from_feasible_rows(
      rows, profile=args.proxy_metric_profile)
  scored = [
      score_candidate(candidate, rows, profile=args.proxy_metric_profile)
      for candidate in candidates
  ]
  selected = select_candidate(
      scored, args.max_false_positive, profile=args.proxy_metric_profile)
  write_outputs(args.output_dir, scored, selected, rows, args)
  print(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

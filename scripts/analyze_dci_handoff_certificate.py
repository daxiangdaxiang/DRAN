#!/usr/bin/env python3
"""Audit decentralized chordal initialization handoff readiness.

The certificate is intentionally diagnostic. It does not change the optimizer
or decide a default algorithm path. It makes the two-stage boundary explicit:
does the initialization stage expose enough separator evidence to justify
handoff to the downstream Riemannian optimizer?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CertificateThresholds:
  min_topology_coverage: float = 0.95
  min_residual_mass_coverage: float = 0.99
  min_payload_coverage: float = 0.98
  max_missing_bridge_ratio: float = 0.02
  max_component_delta_ratio: float = 0.02
  max_missing_bridge_blocks: int | None = None
  max_component_delta: int | None = None
  max_structural_criticality: float = 0.05
  max_rotation_normal_residual: float | None = None
  max_translation_normal_residual: float | None = None


def _float_metric(report: dict[str, Any], name: str, default: float = 0.0):
  value = report.get(name, default)
  if value is None:
    return float(default)
  return float(value)


def _int_metric(report: dict[str, Any], name: str, default: int = 0):
  value = report.get(name, default)
  if value is None:
    return int(default)
  return int(value)


def _min_gate(name: str, value: float, threshold: float):
  return {
      "name": name,
      "value": float(value),
      "threshold": float(threshold),
      "direction": ">=",
      "passed": float(value) >= float(threshold),
  }


def _max_gate(name: str, value: float, threshold: float):
  return {
      "name": name,
      "value": float(value),
      "threshold": float(threshold),
      "direction": "<=",
      "passed": float(value) <= float(threshold),
  }


def _normal_summary_full_block_edges(report: dict[str, Any]):
  value = report.get("separator_normal_summary_full_block_edges")
  if value is not None:
    return int(value)
  graph = report.get("separator_normal_summary_block_graph") or {}
  value = graph.get("full_block_edge_count", 0)
  return int(value or 0)


def _nested_float(report: dict[str, Any], path: tuple[str, ...]):
  value = report
  for key in path:
    if not isinstance(value, dict) or key not in value:
      return None
    value = value[key]
  if value is None:
    return None
  return float(value)


def evaluate_report_certificate(
    report: dict[str, Any],
    thresholds: CertificateThresholds = CertificateThresholds(),
):
  """Return conservative handoff-readiness gates for one DCI report."""

  full_block_edges = max(1, _normal_summary_full_block_edges(report))
  missing_bridge_blocks = _int_metric(
      report, "separator_normal_summary_missing_bridge_block_edges")
  component_delta = _int_metric(
      report, "separator_normal_summary_component_count_delta")
  gates = [
      _min_gate(
          "topology_separator_coverage",
          _float_metric(report, "topology_separator_coverage"),
          thresholds.min_topology_coverage),
      _min_gate(
          "separator_residual_mass_coverage",
          _float_metric(report, "separator_residual_mass_coverage"),
          thresholds.min_residual_mass_coverage),
      _min_gate(
          "separator_normal_summary_payload_coverage",
          _float_metric(report, "separator_normal_summary_payload_coverage"),
          thresholds.min_payload_coverage),
      _max_gate(
          "missing_bridge_block_ratio",
          missing_bridge_blocks / float(full_block_edges),
          thresholds.max_missing_bridge_ratio),
      _max_gate(
          "component_count_delta_ratio",
          component_delta / float(full_block_edges),
          thresholds.max_component_delta_ratio),
      _max_gate(
          "structural_criticality",
          _float_metric(report, "separator_normal_summary_structural_criticality"),
          thresholds.max_structural_criticality),
  ]
  if thresholds.max_missing_bridge_blocks is not None:
    gates.append(_max_gate(
        "missing_bridge_blocks",
        missing_bridge_blocks,
        thresholds.max_missing_bridge_blocks))
  if thresholds.max_component_delta is not None:
    gates.append(_max_gate(
        "component_count_delta",
        component_delta,
        thresholds.max_component_delta))
  if thresholds.max_rotation_normal_residual is not None:
    rotation_residual = _nested_float(
        report,
        ("distributed_stats", "rotation_stats", "final_normal_residual"))
    gates.append(_max_gate(
        "rotation_normal_residual",
        float("inf") if rotation_residual is None else rotation_residual,
        thresholds.max_rotation_normal_residual))
  if thresholds.max_translation_normal_residual is not None:
    translation_residual = _nested_float(
        report,
        ("distributed_stats", "translation_stats", "final_normal_residual"))
    gates.append(_max_gate(
        "translation_normal_residual",
        float("inf") if translation_residual is None else translation_residual,
        thresholds.max_translation_normal_residual))
  status = "ready" if all(gate["passed"] for gate in gates) else "not_ready"
  return {
      "dataset": report.get("dataset") or report.get("name"),
      "relay_scheduler": report.get("relay_scheduler"),
      "status": status,
      "gates": gates,
      "normal_summary_full_block_edges": full_block_edges,
      "missing_bridge_blocks": missing_bridge_blocks,
      "component_count_delta": component_delta,
      "actual_dci_comm_mb": _float_metric(report, "actual_dci_comm_mb"),
      "selected_total_cost": report.get("selected_total_cost"),
      "full_graph_distributed_total_cost":
          report.get("full_graph_distributed_total_cost"),
  }


def _parse_float(row: dict[str, str], key: str):
  value = row.get(key, "")
  if value == "":
    return None
  return float(value)


def evaluate_handoff_transfer_csv(
    comparison_csv: Path,
    attenuation_ratio_threshold: float = 0.2,
):
  """Evaluate whether initialization gains survive downstream optimization."""

  rows = []
  with Path(comparison_csv).open(newline="") as stream:
    for row in csv.DictReader(stream):
      init_delta = _parse_float(row, "init_delta_adaptive_minus_selected")
      b20_delta = _parse_float(row, "b20_delta_adaptive_minus_selected")
      reference_delta = _parse_float(row, "b20_delta_adaptive_minus_no_external")

      transfer_ratio = None
      if init_delta is not None and b20_delta is not None and init_delta < 0:
        transfer_ratio = b20_delta / init_delta

      if init_delta is None or b20_delta is None:
        transfer_status = "missing"
      elif init_delta >= 0:
        transfer_status = "no_initial_gain"
      elif b20_delta >= 0:
        transfer_status = "not_transferred"
      elif transfer_ratio is not None and transfer_ratio < attenuation_ratio_threshold:
        transfer_status = "attenuated"
      else:
        transfer_status = "transferred"

      if reference_delta is None:
        reference_gap_status = "missing"
      elif reference_delta > 0:
        reference_gap_status = "worse_than_reference"
      elif reference_delta < 0:
        reference_gap_status = "better_than_reference"
      else:
        reference_gap_status = "matched_reference"

      rows.append({
          "dataset": row.get("dataset"),
          "init_delta": init_delta,
          "downstream_delta": b20_delta,
          "transfer_ratio": transfer_ratio,
          "transfer_status": transfer_status,
          "reference_delta": reference_delta,
          "reference_gap_status": reference_gap_status,
      })
  return rows


def _infer_dataset_from_report_path(path: Path):
  stem = Path(path).stem
  for suffix in (
      "_projected_merit",
      "_gradient_energy",
      "_size",
  ):
    if stem.endswith(suffix) and len(stem) > len(suffix):
      return stem[:-len(suffix)]
  return Path(path).parent.name


def load_report_json(path: Path):
  path = Path(path)
  with path.open() as stream:
    report = json.load(stream)
  report.setdefault("dataset", _infer_dataset_from_report_path(path))
  report.setdefault("source_path", str(path))
  return report


def _variant_label(report: dict[str, Any], fallback_index: int):
  for key in (
      "variant",
      "name",
      "interface_schur_rotation_coarse_component_selection_mode",
      "interface_schur_ritz_mode",
      "relay_scheduler",
  ):
    value = report.get(key)
    if value:
      return str(value)
  source_path = report.get("source_path")
  if source_path:
    return Path(str(source_path)).stem
  return f"report_{fallback_index}"


def _cost_metric(report: dict[str, Any]):
  for key in (
      "distributed_total_cost",
      "full_graph_distributed_total_cost",
      "selected_total_cost",
  ):
    value = report.get(key)
    if value is not None:
      return float(value)
  return None


def _residual_metrics(report: dict[str, Any]):
  rotation = _nested_float(
      report, ("distributed_stats", "rotation_stats", "final_normal_residual"))
  translation = _nested_float(
      report,
      ("distributed_stats", "translation_stats", "final_normal_residual"))
  joint = None
  if rotation is not None and translation is not None:
    joint = math.sqrt(rotation * rotation + translation * translation)
  return {
      "rotation_normal_residual": rotation,
      "translation_normal_residual": translation,
      "joint_normal_residual": joint,
  }


def _best_variants(rows: list[dict[str, Any]],
                   metric: str,
                   tolerance: float):
  scored = [
      (row["variant"], float(row[metric]))
      for row in rows
      if row.get(metric) is not None and math.isfinite(float(row[metric]))
  ]
  if not scored:
    return [], None
  best_value = min(value for _, value in scored)
  absolute_tolerance = max(float(tolerance), float(tolerance) * abs(best_value))
  return [
      variant
      for variant, value in scored
      if value <= best_value + absolute_tolerance
  ], best_value


def evaluate_metric_alignment_reports(
    reports: list[dict[str, Any]],
    residual_metric: str = "joint_normal_residual",
    cost_metric: str = "measurement_cost",
    tolerance: float = 1e-9,
):
  """Check whether residual-based handoff ranking agrees with measurement cost."""

  if residual_metric not in {
      "rotation_normal_residual",
      "translation_normal_residual",
      "joint_normal_residual",
  }:
    raise ValueError(f"unsupported residual metric: {residual_metric}")
  grouped: dict[str, list[dict[str, Any]]] = {}
  for index, report in enumerate(reports):
    dataset = str(report.get("dataset") or report.get("name") or "unknown")
    residuals = _residual_metrics(report)
    row = {
        "dataset": dataset,
        "variant": _variant_label(report, index),
        "measurement_cost": _cost_metric(report),
        **residuals,
    }
    grouped.setdefault(dataset, []).append(row)

  results = []
  for dataset in sorted(grouped):
    rows = grouped[dataset]
    if len(rows) < 2:
      status = "insufficient_reports"
      best_cost_variants: list[str] = []
      best_residual_variants: list[str] = []
      best_cost = None
      best_residual = None
    else:
      best_cost_variants, best_cost = _best_variants(
          rows, cost_metric, tolerance)
      best_residual_variants, best_residual = _best_variants(
          rows, residual_metric, tolerance)
      if not best_cost_variants or not best_residual_variants:
        status = "missing_metric"
      elif set(best_cost_variants).intersection(best_residual_variants):
        status = "aligned"
      else:
        status = "residual_cost_conflict"
    results.append({
        "dataset": dataset,
        "status": status,
        "cost_metric": cost_metric,
        "residual_metric": residual_metric,
        "best_cost": best_cost,
        "best_residual": best_residual,
        "best_cost_variants": best_cost_variants,
        "best_residual_variants": best_residual_variants,
        "rows": rows,
    })
  return results


def _format_gate(gate: dict[str, Any]):
  marker = "PASS" if gate["passed"] else "FAIL"
  return (
      f"| {gate['name']} | {gate['value']:.12g} | "
      f"{gate['direction']} {gate['threshold']:.12g} | {marker} |"
  )


def render_markdown(report_certificates: list[dict[str, Any]],
                    transfer_rows: list[dict[str, Any]],
                    metric_alignments: list[dict[str, Any]] | None = None):
  lines = [
      "# DCI Handoff Certificate Audit",
      "",
      "This diagnostic separates initialization basin evidence from downstream "
      "optimizer behavior. A failed certificate is not an optimizer failure; it "
      "means the initialization stage did not expose enough separator evidence "
      "under the chosen thresholds.",
      "",
  ]
  if report_certificates:
    lines.extend([
        "## Initialization Reports",
        "",
    ])
    for item in report_certificates:
      lines.extend([
          f"### {item.get('dataset') or 'unknown'}",
          "",
          f"- scheduler: `{item.get('relay_scheduler')}`",
          f"- status: `{item['status']}`",
          f"- actual DCI comm MB: `{item['actual_dci_comm_mb']:.6f}`",
          "",
          "| Gate | Value | Threshold | Result |",
          "| --- | ---: | --- | --- |",
      ])
      lines.extend(_format_gate(gate) for gate in item["gates"])
      lines.append("")
  if transfer_rows:
    lines.extend([
        "## Downstream Transfer",
        "",
        "| Dataset | Init delta | B20 delta | Transfer ratio | Transfer | Reference gap |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ])
    for row in transfer_rows:
      ratio = row["transfer_ratio"]
      ratio_text = "" if ratio is None else f"{ratio:.6f}"
      lines.append(
          f"| {row['dataset']} | {row['init_delta']:.6f} | "
          f"{row['downstream_delta']:.6f} | {ratio_text} | "
          f"{row['transfer_status']} | {row['reference_gap_status']} |"
      )
    lines.append("")
  if metric_alignments:
    lines.extend([
        "## Measurement-Cost Alignment",
        "",
        "This table checks whether the report with the lowest normal residual "
        "is also among the reports with the lowest final measurement cost.",
        "",
        "| Dataset | Status | Best cost variants | Best residual variants | Metric |",
        "| --- | --- | --- | --- | --- |",
    ])
    for item in metric_alignments:
      lines.append(
          f"| {item['dataset']} | {item['status']} | "
          f"`{', '.join(item['best_cost_variants'])}` | "
          f"`{', '.join(item['best_residual_variants'])}` | "
          f"`{item['residual_metric']}` |")
    lines.append("")
  return "\n".join(lines)


def parse_args():
  parser = argparse.ArgumentParser(
      description="Audit DCI handoff readiness and downstream transfer.")
  parser.add_argument("--report-json", action="append", default=[],
                      help="DCI report.json path. Can be passed multiple times.")
  parser.add_argument("--comparison-csv", type=Path, default=None,
                      help="CSV comparing selected-normal and adaptive handoff.")
  parser.add_argument("--output", type=Path, default=None,
                      help="Optional markdown output path.")
  parser.add_argument("--min-topology-coverage", type=float, default=0.95)
  parser.add_argument("--min-residual-mass-coverage", type=float, default=0.99)
  parser.add_argument("--min-payload-coverage", type=float, default=0.98)
  parser.add_argument("--max-missing-bridge-ratio", type=float, default=0.02)
  parser.add_argument("--max-component-delta-ratio", type=float, default=0.02)
  parser.add_argument("--max-missing-bridge-blocks", type=int, default=None)
  parser.add_argument("--max-component-delta", type=int, default=None)
  parser.add_argument("--max-structural-criticality", type=float, default=0.05)
  parser.add_argument("--max-rotation-normal-residual", type=float, default=None)
  parser.add_argument("--max-translation-normal-residual", type=float,
                      default=None)
  parser.add_argument("--attenuation-ratio-threshold", type=float, default=0.2)
  parser.add_argument(
      "--alignment-residual-metric",
      choices=[
          "rotation_normal_residual",
          "translation_normal_residual",
          "joint_normal_residual",
      ],
      default="joint_normal_residual")
  return parser.parse_args()


def main():
  args = parse_args()
  thresholds = CertificateThresholds(
      min_topology_coverage=args.min_topology_coverage,
      min_residual_mass_coverage=args.min_residual_mass_coverage,
      min_payload_coverage=args.min_payload_coverage,
      max_missing_bridge_ratio=args.max_missing_bridge_ratio,
      max_component_delta_ratio=args.max_component_delta_ratio,
      max_missing_bridge_blocks=args.max_missing_bridge_blocks,
      max_component_delta=args.max_component_delta,
      max_structural_criticality=args.max_structural_criticality,
      max_rotation_normal_residual=args.max_rotation_normal_residual,
      max_translation_normal_residual=args.max_translation_normal_residual,
  )
  loaded_reports = [
      load_report_json(Path(path)) for path in args.report_json
  ]
  report_certificates = [
      evaluate_report_certificate(report, thresholds)
      for report in loaded_reports
  ]
  metric_alignments = evaluate_metric_alignment_reports(
      loaded_reports,
      residual_metric=args.alignment_residual_metric,
  )
  transfer_rows = []
  if args.comparison_csv is not None:
    transfer_rows = evaluate_handoff_transfer_csv(
        args.comparison_csv,
        attenuation_ratio_threshold=args.attenuation_ratio_threshold)
  markdown = render_markdown(
      report_certificates, transfer_rows, metric_alignments)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown + "\n")
  else:
    print(markdown)


if __name__ == "__main__":
  main()

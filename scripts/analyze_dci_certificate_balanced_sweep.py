#!/usr/bin/env python3
"""Evaluate DCI candidate reports with structural and solve-quality gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_dci_handoff_certificate as handoff_cert


def _nested_float(report: dict[str, Any], path: tuple[str, ...]):
  value = report
  for key in path:
    if not isinstance(value, dict) or key not in value:
      return None
    value = value[key]
  if value is None:
    return None
  return float(value)


def _label_for_report(report: dict[str, Any], index: int):
  if report.get("name"):
    return str(report["name"])
  budget = report.get("relay_byte_budget_mb")
  rot = report.get("rotation_iterations")
  trans = report.get("translation_iterations")
  if budget is None:
    budget_label = "unbounded"
  else:
    budget_label = f"{float(budget):.6g}MB"
  pieces = [budget_label]
  if rot is not None or trans is not None:
    pieces.append(f"pcg{rot}_{trans}")
  return "_".join(pieces) if pieces else f"report_{index}"


def evaluate_certificate_balanced_reports(
    reports: list[dict[str, Any]],
    thresholds: handoff_cert.CertificateThresholds,
):
  rows = []
  for index, report in enumerate(reports):
    certificate = handoff_cert.evaluate_report_certificate(report, thresholds)
    failed_gates = [
        gate["name"] for gate in certificate["gates"] if not gate["passed"]
    ]
    row = {
        "label": _label_for_report(report, index),
        "dataset": report.get("dataset"),
        "status": certificate["status"],
        "failed_gates": failed_gates,
        "actual_dci_comm_mb":
            float(report.get("actual_dci_comm_mb") or 0.0),
        "relay_byte_budget_mb": report.get("relay_byte_budget_mb"),
        "rotation_iterations": report.get("rotation_iterations"),
        "translation_iterations": report.get("translation_iterations"),
        "cost": report.get("full_graph_distributed_total_cost"),
        "selected": report.get("selected"),
        "topology_separator_coverage":
            report.get("topology_separator_coverage"),
        "separator_normal_summary_payload_coverage":
            report.get("separator_normal_summary_payload_coverage"),
        "missing_bridge_blocks":
            report.get("separator_normal_summary_missing_bridge_block_edges"),
        "component_count_delta":
            report.get("separator_normal_summary_component_count_delta"),
        "rotation_normal_residual":
            _nested_float(
                report,
                ("distributed_stats", "rotation_stats",
                 "final_normal_residual")),
        "translation_normal_residual":
            _nested_float(
                report,
                ("distributed_stats", "translation_stats",
                 "final_normal_residual")),
    }
    rows.append(row)
  ready_rows = [row for row in rows if row["status"] == "ready"]
  selected = None
  if ready_rows:
    selected = min(
        ready_rows,
        key=lambda row: (
            float(row["actual_dci_comm_mb"]),
            float("inf") if row["cost"] is None else float(row["cost"]),
            row["label"],
        ))
  return rows, selected


def render_certificate_balanced_markdown(rows, selected):
  lines = [
      "# DCI Certificate-Balanced Sweep",
      "",
      "This diagnostic evaluates actual DCI candidate reports with both "
      "separator-structure gates and optional linear-solve residual gates.",
      "",
  ]
  if selected is None:
    lines.append("selected configuration: `none`")
  else:
    lines.append(f"selected configuration: `{selected['label']}`")
  lines.extend([
      "",
      "| Label | Status | DCI MB | Cost | Rot residual | Trans residual | Failed gates |",
      "| --- | --- | ---: | ---: | ---: | ---: | --- |",
  ])
  for row in rows:
    cost = "" if row["cost"] is None else f"{float(row['cost']):.6f}"
    rot = (
        "" if row["rotation_normal_residual"] is None
        else f"{float(row['rotation_normal_residual']):.6f}"
    )
    trans = (
        "" if row["translation_normal_residual"] is None
        else f"{float(row['translation_normal_residual']):.6f}"
    )
    failed = ", ".join(row["failed_gates"])
    lines.append(
        f"| {row['label']} | {row['status']} | "
        f"{float(row['actual_dci_comm_mb']):.6f} | {cost} | "
        f"{rot} | {trans} | {failed} |"
    )
  lines.append("")
  return "\n".join(lines)


def _write_csv(path: Path, rows):
  fieldnames = [
      "label",
      "dataset",
      "status",
      "actual_dci_comm_mb",
      "relay_byte_budget_mb",
      "rotation_iterations",
      "translation_iterations",
      "cost",
      "selected",
      "topology_separator_coverage",
      "separator_normal_summary_payload_coverage",
      "missing_bridge_blocks",
      "component_count_delta",
      "rotation_normal_residual",
      "translation_normal_residual",
      "failed_gates",
  ]
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      out = dict(row)
      out["failed_gates"] = ";".join(row["failed_gates"])
      writer.writerow({key: out.get(key) for key in fieldnames})


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--report-json", action="append", default=[],
                      help="Actual DCI report.json. Can be repeated.")
  parser.add_argument("--output-md", type=Path, default=None)
  parser.add_argument("--output-csv", type=Path, default=None)
  parser.add_argument("--min-topology-coverage", type=float, default=0.95)
  parser.add_argument("--min-residual-mass-coverage", type=float, default=0.99)
  parser.add_argument("--min-payload-coverage", type=float, default=0.98)
  parser.add_argument("--max-missing-bridge-ratio", type=float, default=0.02)
  parser.add_argument("--max-component-delta-ratio", type=float, default=0.02)
  parser.add_argument("--max-structural-criticality", type=float, default=0.05)
  parser.add_argument("--max-rotation-normal-residual", type=float,
                      default=None)
  parser.add_argument("--max-translation-normal-residual", type=float,
                      default=None)
  return parser.parse_args()


def main():
  args = parse_args()
  thresholds = handoff_cert.CertificateThresholds(
      min_topology_coverage=args.min_topology_coverage,
      min_residual_mass_coverage=args.min_residual_mass_coverage,
      min_payload_coverage=args.min_payload_coverage,
      max_missing_bridge_ratio=args.max_missing_bridge_ratio,
      max_component_delta_ratio=args.max_component_delta_ratio,
      max_structural_criticality=args.max_structural_criticality,
      max_rotation_normal_residual=args.max_rotation_normal_residual,
      max_translation_normal_residual=args.max_translation_normal_residual,
  )
  reports = [
      handoff_cert.load_report_json(Path(path)) for path in args.report_json
  ]
  rows, selected = evaluate_certificate_balanced_reports(reports, thresholds)
  if args.output_csv is not None:
    _write_csv(args.output_csv, rows)
  markdown = render_certificate_balanced_markdown(rows, selected)
  if args.output_md is not None:
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown + "\n")
  else:
    print(markdown)


if __name__ == "__main__":
  main()

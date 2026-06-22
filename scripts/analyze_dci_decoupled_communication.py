#!/usr/bin/env python3
"""Estimate communication targets for certificate-balanced DCI variants.

This is a diagnostic model, not an implemented optimizer. It quantifies the
potential gain from decoupling one-time separator evidence acquisition from
later solve-quality continuation messages.
"""

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


def _float(report: dict[str, Any], key: str, default: float = 0.0):
  value = report.get(key, default)
  if value is None:
    return float(default)
  return float(value)


def _nested_float(report: dict[str, Any], path: tuple[str, ...],
                  default: float = 0.0):
  value = report
  for key in path:
    if not isinstance(value, dict) or key not in value:
      return float(default)
    value = value[key]
  if value is None:
    return float(default)
  return float(value)


def _round_mb(value: float):
  return round(float(value), 6)


def estimate_decoupled_communication(evidence_report: dict[str, Any],
                                     solve_report: dict[str, Any]):
  """Estimate two decoupled communication targets from existing reports.

  evidence_report supplies the one-time relay payload required to satisfy the
  structural certificate. solve_report supplies the high-quality solve result.
  """

  evidence_relay_mb = _float(evidence_report, "relay_extra_comm_mb")
  if evidence_relay_mb <= 0.0:
    evidence_relay_mb = _float(
        evidence_report, "relay_scheduler_predicted_selected_relay_mb")
  normal_summary_mb = _float(
      solve_report, "normal_equation_summary_comm_mb",
      _float(evidence_report, "normal_equation_summary_comm_mb"))
  linear_total_mb = _nested_float(
      solve_report,
      ("actual_linear_communication", "linear_solve_total_estimated_mb"))
  pcg_reduction_mb = _nested_float(
      solve_report,
      ("actual_linear_communication", "pcg_global_reduction_mb"))
  certificate_mb = _nested_float(
      solve_report,
      ("certificate_communication", "residual_delta_consensus_mb"))
  current_actual_mb = _float(solve_report, "actual_dci_comm_mb")
  current_relay_mb = _float(solve_report, "relay_extra_comm_mb")

  decoupled_relay_only_mb = (
      evidence_relay_mb + normal_summary_mb + linear_total_mb + certificate_mb)
  compact_continuation_mb = (
      evidence_relay_mb + normal_summary_mb + pcg_reduction_mb + certificate_mb)
  return {
      "decoupled_model":
          "one_time_evidence_relay_plus_compact_solve_continuation",
      "current_actual_mb": _round_mb(current_actual_mb),
      "current_relay_mb": _round_mb(current_relay_mb),
      "evidence_relay_mb": _round_mb(evidence_relay_mb),
      "normal_summary_mb": _round_mb(normal_summary_mb),
      "linear_total_mb": _round_mb(linear_total_mb),
      "pcg_reduction_mb": _round_mb(pcg_reduction_mb),
      "certificate_mb": _round_mb(certificate_mb),
      "decoupled_relay_only_mb": _round_mb(decoupled_relay_only_mb),
      "relay_savings_mb": _round_mb(current_actual_mb - decoupled_relay_only_mb),
      "compact_continuation_mb": _round_mb(compact_continuation_mb),
      "compact_savings_mb":
          _round_mb(current_actual_mb - compact_continuation_mb),
      "solve_cost": solve_report.get("full_graph_distributed_total_cost"),
      "solve_rotation_residual": _nested_float(
          solve_report,
          ("distributed_stats", "rotation_stats", "final_normal_residual"),
          default=float("nan")),
      "solve_translation_residual": _nested_float(
          solve_report,
          ("distributed_stats", "translation_stats", "final_normal_residual"),
          default=float("nan")),
  }


def render_markdown(estimate: dict[str, Any]):
  return "\n".join([
      "# DCI Decoupled Communication Diagnostic",
      "",
      "This diagnostic estimates communication if separator evidence is "
      "transmitted once and later solve-quality iterations use compact "
      "continuation messages. It is a target model, not an implemented "
      "optimizer.",
      "",
      "| Model | MB | Savings vs current |",
      "| --- | ---: | ---: |",
      f"| current reported DCI | {estimate['current_actual_mb']:.6f} | 0.000000 |",
      f"| decoupled relay only | {estimate['decoupled_relay_only_mb']:.6f} | "
      f"{estimate['relay_savings_mb']:.6f} |",
      f"| compact continuation target | {estimate['compact_continuation_mb']:.6f} | "
      f"{estimate['compact_savings_mb']:.6f} |",
      "",
      "Components:",
      "",
      f"- evidence relay MB: `{estimate['evidence_relay_mb']:.6f}`",
      f"- normal-summary MB: `{estimate['normal_summary_mb']:.6f}`",
      f"- full linear-solve MB: `{estimate['linear_total_mb']:.6f}`",
      f"- PCG global-reduction MB: `{estimate['pcg_reduction_mb']:.6f}`",
      f"- certificate MB: `{estimate['certificate_mb']:.6f}`",
      f"- solve cost: `{estimate['solve_cost']}`",
      f"- solve residuals: rotation `{estimate['solve_rotation_residual']:.6f}`, "
      f"translation `{estimate['solve_translation_residual']:.6f}`",
      "",
  ])


def _write_csv(path: Path, estimate: dict[str, Any]):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=sorted(estimate))
    writer.writeheader()
    writer.writerow(estimate)


def _load(path: Path):
  with Path(path).open() as stream:
    return json.load(stream)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--evidence-report", required=True,
                      help="Structural-ready low-iteration DCI report.")
  parser.add_argument("--solve-report", required=True,
                      help="High-quality solve DCI report.")
  parser.add_argument("--output-md", type=Path, default=None)
  parser.add_argument("--output-csv", type=Path, default=None)
  return parser.parse_args()


def main():
  args = parse_args()
  estimate = estimate_decoupled_communication(
      _load(Path(args.evidence_report)),
      _load(Path(args.solve_report)))
  if args.output_csv is not None:
    _write_csv(args.output_csv, estimate)
  markdown = render_markdown(estimate)
  if args.output_md is not None:
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown + "\n")
  else:
    print(markdown)


if __name__ == "__main__":
  main()

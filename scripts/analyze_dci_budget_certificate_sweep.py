#!/usr/bin/env python3
"""Sweep relay communication budgets against DCI handoff certificates.

This diagnostic asks a narrow initialization-stage question: how much relay
payload is needed before the selected separator evidence satisfies the
handoff certificate? It deliberately does not solve the DCI candidate or run
the downstream optimizer.
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

from scripts import analyze_dci_handoff_certificate as handoff_cert
from scripts import run_budgeted_dci as bc_dci
from scripts.evaluate_pgo import parse_g2o_graph


def _budget_label(budget_mb):
  return "unbounded" if budget_mb is None else f"{float(budget_mb):.6g}"


def _gate_map(certificate: dict[str, Any]):
  return {gate["name"]: gate for gate in certificate["gates"]}


def _ratio(numerator: float, denominator: float):
  denominator = float(denominator)
  if denominator <= 0.0:
    return 0.0
  return float(numerator) / denominator


def evaluate_one_budget(
    *,
    graph_edges,
    pose_ids,
    robot_of,
    baseline_poses,
    available_robot_pairs,
    max_relay_hops,
    relay_scheduler,
    relay_budget_mb,
    thresholds,
    rotation_iterations,
    translation_iterations,
    weighted,
    cost_mode,
    dataset=None,
):
  selected_edges, topology_evidence = bc_dci.topology_edge_evidence(
      graph_edges=graph_edges,
      robot_of=robot_of,
      available_robot_pairs=available_robot_pairs,
      max_relay_hops=max_relay_hops,
      relay_scheduler=relay_scheduler,
      relay_byte_budget_mb=relay_budget_mb,
      baseline_poses=baseline_poses,
      rotation_iterations=rotation_iterations,
      translation_iterations=translation_iterations,
      weighted=weighted,
      cost_mode=cost_mode)
  dim = bc_dci.pose_dimension_from_edges(graph_edges)
  active_ids = bc_dci.active_pose_ids(graph_edges, robot_of, 1)
  full_cost = bc_dci.cost_breakdown(
      graph_edges, baseline_poses, robot_of, active_ids, weighted, cost_mode)
  selected_active_ids = bc_dci.active_pose_ids(
      selected_edges, robot_of, 1)
  selected_cost = bc_dci.cost_breakdown(
      selected_edges, baseline_poses, robot_of, selected_active_ids,
      weighted, cost_mode)
  full_separator_edges = bc_dci._separator_edge_count(graph_edges, robot_of)
  covered_separator_edges = int(topology_evidence["covered_separator_edges"])
  topology_coverage = (
      1.0 if full_separator_edges == 0
      else float(covered_separator_edges) / float(full_separator_edges)
  )
  residual_coverage = bc_dci.compute_separator_residual_mass_coverage(
      full_cost, selected_cost, topology_coverage)
  payload_coverage = bc_dci.compute_separator_normal_summary_payload_coverage(
      full_graph_edges=graph_edges,
      selected_edges=selected_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      baseline_poses=baseline_poses,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode)
  structure = bc_dci.compute_separator_normal_summary_block_graph_certificate(
      full_graph_edges=graph_edges,
      selected_edges=selected_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      baseline_poses=baseline_poses,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode)
  full_block_edges = max(
      1, int(structure["separator_normal_summary_full_block_edges"]))
  report = {
      "dataset": dataset,
      "relay_scheduler": relay_scheduler,
      "topology_separator_coverage": topology_coverage,
      **residual_coverage,
      **payload_coverage,
      "separator_normal_summary_full_block_edges":
          int(structure["separator_normal_summary_full_block_edges"]),
      "separator_normal_summary_covered_block_edges":
          int(structure["separator_normal_summary_covered_block_edges"]),
      "separator_normal_summary_missing_block_edges":
          int(structure["separator_normal_summary_missing_block_edges"]),
      "separator_normal_summary_missing_bridge_block_edges":
          int(structure[
              "separator_normal_summary_missing_bridge_block_edges"]),
      "separator_normal_summary_component_count_delta":
          int(structure["separator_normal_summary_component_count_delta"]),
      "separator_normal_summary_structural_criticality":
          float(structure[
              "separator_normal_summary_structural_criticality"]),
      "actual_dci_comm_mb":
          float(topology_evidence.get(
              "relay_scheduler_predicted_selected_relay_mb", 0.0)),
  }
  certificate = handoff_cert.evaluate_report_certificate(report, thresholds)
  gates = _gate_map(certificate)
  return {
      "dataset": dataset,
      "relay_scheduler": relay_scheduler,
      "budget_mb": relay_budget_mb,
      "certificate_status": certificate["status"],
      "topology_separator_coverage": topology_coverage,
      "separator_residual_mass_coverage":
          residual_coverage["separator_residual_mass_coverage"],
      "separator_normal_summary_payload_coverage":
          payload_coverage["separator_normal_summary_payload_coverage"],
      "missing_bridge_block_ratio":
          gates["missing_bridge_block_ratio"]["value"],
      "component_count_delta_ratio":
          gates["component_count_delta_ratio"]["value"],
      "structural_criticality":
          float(structure[
              "separator_normal_summary_structural_criticality"]),
      "missing_bridge_blocks":
          int(structure[
              "separator_normal_summary_missing_bridge_block_edges"]),
      "component_count_delta":
          int(structure["separator_normal_summary_component_count_delta"]),
      "full_block_edges": full_block_edges,
      "relay_scheduler_selected_relay_edges":
          int(topology_evidence["relay_scheduler_selected_relay_edges"]),
      "relay_scheduler_reachable_relay_edges":
          int(topology_evidence["relay_scheduler_reachable_relay_edges"]),
      "relay_scheduler_predicted_selected_relay_mb":
          float(topology_evidence.get(
              "relay_scheduler_predicted_selected_relay_mb", 0.0)),
      "relay_scheduler_pose_summary_units":
          int(topology_evidence.get("relay_scheduler_pose_summary_units", 0)),
      "relay_scheduler_structural_bridge_gain":
          int(topology_evidence.get("relay_scheduler_structural_bridge_gain", 0)),
      "certificate": certificate,
  }


def evaluate_budget_sweep_for_graph_data(
    *,
    graph_edges,
    pose_ids,
    robot_of,
    baseline_poses,
    available_robot_pairs,
    max_relay_hops,
    relay_scheduler,
    relay_budgets_mb,
    thresholds=handoff_cert.CertificateThresholds(),
    rotation_iterations=80,
    translation_iterations=100,
    weighted=False,
    cost_mode="dpgo",
    dataset=None,
):
  rows = []
  for budget_mb in relay_budgets_mb:
    rows.append(evaluate_one_budget(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        baseline_poses=baseline_poses,
        available_robot_pairs=available_robot_pairs,
        max_relay_hops=max_relay_hops,
        relay_scheduler=relay_scheduler,
        relay_budget_mb=budget_mb,
        thresholds=thresholds,
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        weighted=weighted,
        cost_mode=cost_mode,
        dataset=dataset))
  first_ready = next(
      (row for row in rows if row["certificate_status"] == "ready"), None)
  return rows, first_ready


def render_budget_sweep_markdown(rows, first_ready):
  lines = [
      "# DCI Budget-Certificate Sweep",
      "",
      "This diagnostic estimates the relay budget needed for the initialization "
      "stage to satisfy the handoff certificate. It does not run candidate DCI "
      "optimization or downstream DRAN/Phase41 refinement.",
      "",
  ]
  if first_ready is None:
    lines.append("first ready budget: `none in sweep`")
  else:
    lines.append(f"first ready budget: `{_budget_label(first_ready['budget_mb'])}`")
  lines.extend([
      "",
      "| Dataset | Budget MB | Status | Topology cov | Payload cov | Missing bridge ratio | Component delta ratio | Structural criticality | Relay MB | Relay edges |",
      "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ])
  for row in rows:
    lines.append(
        f"| {row.get('dataset') or ''} | {_budget_label(row['budget_mb'])} | "
        f"{row['certificate_status']} | "
        f"{row['topology_separator_coverage']:.6f} | "
        f"{row['separator_normal_summary_payload_coverage']:.6f} | "
        f"{row['missing_bridge_block_ratio']:.6f} | "
        f"{row['component_count_delta_ratio']:.6f} | "
        f"{row['structural_criticality']:.6f} | "
        f"{row['relay_scheduler_predicted_selected_relay_mb']:.6f} | "
        f"{row['relay_scheduler_selected_relay_edges']} |"
    )
  lines.append("")
  return "\n".join(lines)


def _write_csv(path: Path, rows):
  fieldnames = [
      "dataset",
      "relay_scheduler",
      "budget_mb",
      "certificate_status",
      "topology_separator_coverage",
      "separator_residual_mass_coverage",
      "separator_normal_summary_payload_coverage",
      "missing_bridge_block_ratio",
      "component_count_delta_ratio",
      "structural_criticality",
      "missing_bridge_blocks",
      "component_count_delta",
      "full_block_edges",
      "relay_scheduler_predicted_selected_relay_mb",
      "relay_scheduler_selected_relay_edges",
      "relay_scheduler_reachable_relay_edges",
      "relay_scheduler_pose_summary_units",
      "relay_scheduler_structural_bridge_gain",
  ]
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: row.get(key) for key in fieldnames})


def parse_budget_list(text: str):
  budgets = []
  for token in text.split(","):
    stripped = token.strip()
    if not stripped:
      continue
    if stripped.lower() in {"none", "inf", "unbounded"}:
      budgets.append(None)
    else:
      budgets.append(float(stripped))
  if not budgets:
    raise ValueError("at least one budget is required")
  return budgets


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", required=True)
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--baseline-estimate", required=True)
  parser.add_argument("--baseline-format", default="manual_matrix",
                      choices=["auto", "g2o", "pose", "matrix", "manual_matrix"])
  parser.add_argument("--topology-file", default=None)
  parser.add_argument("--topology-round", type=int, default=0)
  parser.add_argument("--topology-round-window", type=int, default=1)
  parser.add_argument("--topology-max-relay-hops", type=int, default=0)
  parser.add_argument("--relay-scheduler",
                      default="pose_summary_adaptive_bridge_density_budget")
  parser.add_argument("--relay-budgets-mb", required=True,
                      help="Comma-separated budgets; use unbounded for None.")
  parser.add_argument("--rotation-iterations", type=int, default=80)
  parser.add_argument("--translation-iterations", type=int, default=100)
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="dpgo", choices=["legacy", "dpgo"])
  parser.add_argument("--output-csv", type=Path, default=None)
  parser.add_argument("--output-md", type=Path, default=None)
  parser.add_argument("--min-topology-coverage", type=float, default=0.95)
  parser.add_argument("--min-residual-mass-coverage", type=float, default=0.99)
  parser.add_argument("--min-payload-coverage", type=float, default=0.98)
  parser.add_argument("--max-missing-bridge-ratio", type=float, default=0.02)
  parser.add_argument("--max-component-delta-ratio", type=float, default=0.02)
  parser.add_argument("--max-structural-criticality", type=float, default=0.05)
  return parser.parse_args()


def main():
  args = parse_args()
  graph_vertices, graph_edges = parse_g2o_graph(Path(args.graph))
  pose_ids = bc_dci.graph_pose_ids(graph_vertices, graph_edges)
  robot_of, _ = bc_dci.build_contiguous_robot_map(pose_ids, args.num_robots)
  baseline_poses, _ = bc_dci.load_oracle_pose_set(
      Path(args.baseline_estimate), args.baseline_format)
  available_pairs = (
      bc_dci.load_topology_pairs(
          Path(args.topology_file),
          args.topology_round,
          args.topology_round_window)
      if args.topology_file else None
  )
  thresholds = handoff_cert.CertificateThresholds(
      min_topology_coverage=args.min_topology_coverage,
      min_residual_mass_coverage=args.min_residual_mass_coverage,
      min_payload_coverage=args.min_payload_coverage,
      max_missing_bridge_ratio=args.max_missing_bridge_ratio,
      max_component_delta_ratio=args.max_component_delta_ratio,
      max_structural_criticality=args.max_structural_criticality)
  rows, first_ready = evaluate_budget_sweep_for_graph_data(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      baseline_poses=baseline_poses,
      available_robot_pairs=available_pairs,
      max_relay_hops=args.topology_max_relay_hops,
      relay_scheduler=args.relay_scheduler,
      relay_budgets_mb=parse_budget_list(args.relay_budgets_mb),
      thresholds=thresholds,
      rotation_iterations=args.rotation_iterations,
      translation_iterations=args.translation_iterations,
      weighted=args.weighted,
      cost_mode=args.cost_mode,
      dataset=Path(args.graph).stem)
  if args.output_csv is not None:
    _write_csv(args.output_csv, rows)
  markdown = render_budget_sweep_markdown(rows, first_ready)
  if args.output_md is not None:
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown + "\n")
  else:
    print(markdown)


if __name__ == "__main__":
  main()

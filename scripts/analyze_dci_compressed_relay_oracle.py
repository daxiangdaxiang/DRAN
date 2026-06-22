#!/usr/bin/env python3
"""Diagnose compressed relay evidence for BC-DCI topology coverage.

This script does not change the DCI solver. It asks whether sparse-topology
initialization failures are caused by edge-wise relay accounting that could be
replaced by a separator-pose summary model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
  from scripts.analyze_shape_drift_oracle import (  # type: ignore
      build_contiguous_robot_map,
      graph_pose_ids,
      pose_dimension_from_edges,
  )
  from scripts.evaluate_pgo import Edge, parse_g2o_graph  # type: ignore
  from scripts.run_budgeted_dci import (  # type: ignore
      _normalize_pair,
      _separator_edge_count,
      _shortest_robot_path,
      load_topology_pairs,
      topology_edge_evidence,
  )
except ImportError:  # pragma: no cover - used when executed as a script.
  from analyze_shape_drift_oracle import (  # type: ignore
      build_contiguous_robot_map,
      graph_pose_ids,
      pose_dimension_from_edges,
  )
  from evaluate_pgo import Edge, parse_g2o_graph  # type: ignore
  from run_budgeted_dci import (  # type: ignore
      _normalize_pair,
      _separator_edge_count,
      _shortest_robot_path,
      load_topology_pairs,
      topology_edge_evidence,
  )


def estimate_pose_summary_relay_unit_mb(rotation_iterations: int,
                                        translation_iterations: int,
                                        extra_hops: int,
                                        dimension: int):
  if extra_hops <= 0:
    return 0.0
  bytes_count = (
      int(extra_hops) *
      (
          int(rotation_iterations) * int(dimension * dimension) * 8 +
          int(translation_iterations) * int(dimension) * 8
      )
  )
  return float(bytes_count) / (1024.0 * 1024.0)


def _relay_pose_units(edge: Edge, robot_of: dict[int, int], path: list[int]):
  ri = int(robot_of[edge.i])
  rj = int(robot_of[edge.j])
  path_tuple = tuple(int(robot) for robot in path)
  return {
      (int(edge.i), ri, rj, path_tuple),
      (int(edge.j), rj, ri, tuple(reversed(path_tuple))),
  }


def compute_pose_summary_relay_oracle(
    graph_edges: list[Edge],
    robot_of: dict[int, int],
    available_robot_pairs: set[tuple[int, int]],
    max_relay_hops: int,
    relay_byte_budget_mb: float,
    rotation_iterations: int,
    translation_iterations: int,
):
  if max_relay_hops < 0:
    raise ValueError("max_relay_hops must be nonnegative")
  available = {_normalize_pair(src, dst) for src, dst in available_robot_pairs}
  dim = pose_dimension_from_edges(graph_edges)
  max_path_hops = max(1, int(max_relay_hops) + 1)
  full_separator_edges = _separator_edge_count(graph_edges, robot_of)
  direct_edges = 0
  unreachable_edges = 0
  candidates = []

  for edge_index, edge in enumerate(graph_edges):
    ri = robot_of.get(edge.i)
    rj = robot_of.get(edge.j)
    if ri is None or rj is None or ri == rj:
      continue
    path = _shortest_robot_path(int(ri), int(rj), available, max_path_hops)
    if path is None:
      unreachable_edges += 1
      continue
    path_hops = len(path) - 1
    if path_hops == 1:
      direct_edges += 1
      continue
    extra_hops = path_hops - 1
    units = _relay_pose_units(edge, robot_of, path)
    unit_costs = {
        unit: estimate_pose_summary_relay_unit_mb(
            rotation_iterations=rotation_iterations,
            translation_iterations=translation_iterations,
            extra_hops=extra_hops,
            dimension=dim,
        )
        for unit in units
    }
    candidates.append({
        "edge_index": int(edge_index),
        "path": [int(robot) for robot in path],
        "path_hops": int(path_hops),
        "extra_hops": int(extra_hops),
        "units": units,
        "unit_costs": unit_costs,
    })

  budget_mb = max(0.0, float(relay_byte_budget_mb))
  spent_mb = 0.0
  selected_indices: list[int] = []
  selected_units = set()
  remaining = list(candidates)
  while remaining:
    best = None
    best_cost = None
    for candidate in remaining:
      incremental_cost = sum(
          float(candidate["unit_costs"][unit])
          for unit in candidate["units"]
          if unit not in selected_units
      )
      if spent_mb + incremental_cost > budget_mb + 1e-15:
        continue
      key = (incremental_cost, int(candidate["edge_index"]))
      if best is None or key < best_cost:
        best = candidate
        best_cost = key
    if best is None:
      break
    spent_mb += float(best_cost[0])
    selected_indices.append(int(best["edge_index"]))
    selected_units.update(best["units"])
    remaining = [
        candidate for candidate in remaining
        if int(candidate["edge_index"]) != int(best["edge_index"])
    ]

  covered_edges = direct_edges + len(selected_indices)
  coverage = (
      1.0 if full_separator_edges == 0
      else float(covered_edges) / float(full_separator_edges)
  )
  return {
      "full_separator_edges": int(full_separator_edges),
      "direct_separator_edges": int(direct_edges),
      "reachable_relay_edges": int(len(candidates)),
      "unreachable_separator_edges": int(unreachable_edges),
      "selected_relay_edges": int(len(selected_indices)),
      "skipped_relay_edges": int(len(candidates) - len(selected_indices)),
      "covered_separator_edges": int(covered_edges),
      "topology_separator_coverage": coverage,
      "selected_pose_summary_units": int(len(selected_units)),
      "predicted_relay_mb": float(spent_mb),
      "relay_byte_budget_mb": float(budget_mb),
      "selected_edge_indices": selected_indices,
      "model": "pose_summary_relay_oracle",
  }


def build_report(args):
  vertices, graph_edges = parse_g2o_graph(Path(args.graph))
  pose_ids = graph_pose_ids(vertices, graph_edges)
  robot_of, ranges = build_contiguous_robot_map(pose_ids, args.num_robots)
  available_pairs = load_topology_pairs(
      Path(args.topology_file), args.topology_round, args.topology_round_window)
  _, edgewise = topology_edge_evidence(
      graph_edges=graph_edges,
      robot_of=robot_of,
      available_robot_pairs=available_pairs,
      max_relay_hops=args.topology_max_relay_hops,
      relay_scheduler="edge_coverage_budget",
      relay_byte_budget_mb=args.relay_byte_budget_mb,
      rotation_iterations=args.rotation_iterations,
      translation_iterations=args.translation_iterations,
  )
  compressed = compute_pose_summary_relay_oracle(
      graph_edges=graph_edges,
      robot_of=robot_of,
      available_robot_pairs=available_pairs,
      max_relay_hops=args.topology_max_relay_hops,
      relay_byte_budget_mb=args.relay_byte_budget_mb,
      rotation_iterations=args.rotation_iterations,
      translation_iterations=args.translation_iterations,
  )
  full_separator_edges = _separator_edge_count(graph_edges, robot_of)
  edgewise_coverage = (
      1.0 if full_separator_edges == 0
      else float(edgewise["covered_separator_edges"]) / float(full_separator_edges)
  )
  return {
      "graph": str(Path(args.graph)),
      "num_robots": int(args.num_robots),
      "robot_index_ranges": ranges,
      "topology_file": str(Path(args.topology_file)),
      "topology_round": int(args.topology_round),
      "topology_round_window": int(args.topology_round_window),
      "topology_max_relay_hops": int(args.topology_max_relay_hops),
      "rotation_iterations": int(args.rotation_iterations),
      "translation_iterations": int(args.translation_iterations),
      "relay_byte_budget_mb": float(args.relay_byte_budget_mb),
      "edgewise_coverage": edgewise_coverage,
      "edgewise_selected_relay_edges": int(
          edgewise["relay_scheduler_selected_relay_edges"]),
      "edgewise_skipped_relay_edges": int(
          edgewise["relay_scheduler_skipped_relay_edges"]),
      "edgewise_predicted_relay_mb": float(
          edgewise["relay_scheduler_predicted_selected_relay_mb"]),
      "pose_summary_coverage": compressed["topology_separator_coverage"],
      "pose_summary_selected_relay_edges": compressed["selected_relay_edges"],
      "pose_summary_skipped_relay_edges": compressed["skipped_relay_edges"],
      "pose_summary_units": compressed["selected_pose_summary_units"],
      "pose_summary_predicted_relay_mb": compressed["predicted_relay_mb"],
      "full_separator_edges": int(full_separator_edges),
      "direct_separator_edges": int(compressed["direct_separator_edges"]),
      "method": "compressed_relay_oracle",
      "compressed_relay": compressed,
      "edgewise_relay": edgewise,
  }


def _summary_row(report: dict):
  return {
      "graph": report["graph"],
      "edgewise_coverage": report["edgewise_coverage"],
      "pose_summary_coverage": report["pose_summary_coverage"],
      "edgewise_selected_relay_edges": report["edgewise_selected_relay_edges"],
      "pose_summary_selected_relay_edges":
          report["pose_summary_selected_relay_edges"],
      "edgewise_predicted_relay_mb": report["edgewise_predicted_relay_mb"],
      "pose_summary_predicted_relay_mb":
          report["pose_summary_predicted_relay_mb"],
      "pose_summary_units": report["pose_summary_units"],
      "relay_byte_budget_mb": report["relay_byte_budget_mb"],
      "full_separator_edges": report["full_separator_edges"],
      "direct_separator_edges": report["direct_separator_edges"],
  }


def write_summary_row(path: Path, row: dict):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  exists = path.exists()
  with path.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
    if not exists:
      writer.writeheader()
    writer.writerow(row)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", required=True)
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--topology-file", required=True)
  parser.add_argument("--topology-round", type=int, default=0)
  parser.add_argument("--topology-round-window", type=int, default=1)
  parser.add_argument("--topology-max-relay-hops", type=int, default=3)
  parser.add_argument("--relay-byte-budget-mb", type=float, required=True)
  parser.add_argument("--rotation-iterations", type=int, default=80)
  parser.add_argument("--translation-iterations", type=int, default=100)
  parser.add_argument("--output-json", default=None)
  parser.add_argument("--output-summary-row", default=None)
  args = parser.parse_args()

  report = build_report(args)
  if args.output_json:
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
  if args.output_summary_row:
    write_summary_row(Path(args.output_summary_row), _summary_row(report))
  print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()

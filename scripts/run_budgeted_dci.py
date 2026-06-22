#!/usr/bin/env python3
"""Run budgeted certified decentralized chordal initialization.

This runner turns the BC-DCI diagnostic into an explicit initialization stage:
it estimates the communication cost of a low-budget distributed chordal
candidate before solving it, skips low-value candidates, and otherwise accepts
the candidate only through the residual-delta certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from heapq import heappop, heappush
from pathlib import Path

import numpy as np

try:
  from scripts.analyze_distributed_chordal_init import (  # type: ignore
      compute_candidate_residual_certificate,
      estimate_dci_boundary_pose_solve_communication,
      estimate_dci_compact_pcg_solve_communication,
      estimate_dci_interface_schur_solve_communication,
      estimate_dci_linear_solve_communication,
      estimate_separator_normal_equation_summary_communication,
      estimate_residual_certificate_communication,
      separator_boundary_pose_unit_count,
      separator_normal_summary_block_graph_diagnostics,
      solve_distributed_chordal_initialization,
      solve_summary_chordal_initialization,
      write_manual_matrix_pose_set,
  )
  from scripts.analyze_shape_drift_oracle import (  # type: ignore
      active_pose_ids,
      build_contiguous_robot_map,
      cost_breakdown,
      graph_pose_ids,
      load_oracle_pose_set,
      pose_dimension_from_edges,
  )
  from scripts.evaluate_pgo import (  # type: ignore
      Edge,
      edge_chordal_cost,
      edge_weights_from_info,
      parse_g2o_graph,
  )
except ImportError:  # pragma: no cover - used when executed as a script.
  from analyze_distributed_chordal_init import (  # type: ignore
      compute_candidate_residual_certificate,
      estimate_dci_boundary_pose_solve_communication,
      estimate_dci_compact_pcg_solve_communication,
      estimate_dci_interface_schur_solve_communication,
      estimate_dci_linear_solve_communication,
      estimate_separator_normal_equation_summary_communication,
      estimate_residual_certificate_communication,
      separator_boundary_pose_unit_count,
      separator_normal_summary_block_graph_diagnostics,
      solve_distributed_chordal_initialization,
      solve_summary_chordal_initialization,
      write_manual_matrix_pose_set,
  )
  from analyze_shape_drift_oracle import (  # type: ignore
      active_pose_ids,
      build_contiguous_robot_map,
      cost_breakdown,
      graph_pose_ids,
      load_oracle_pose_set,
      pose_dimension_from_edges,
  )
  from evaluate_pgo import (  # type: ignore
      Edge,
      edge_chordal_cost,
      edge_weights_from_info,
      parse_g2o_graph,
  )


def _separator_edge_count(graph_edges: list[Edge], robot_of: dict[int, int]):
  return sum(
      1 for edge in graph_edges
      if edge.i in robot_of and edge.j in robot_of and robot_of[edge.i] != robot_of[edge.j]
  )


def _normalize_pair(src: int, dst: int):
  return (src, dst) if src <= dst else (dst, src)


def _separator_pose_block_edge(edge: Edge, robot_of: dict[int, int]):
  ri = robot_of.get(edge.i)
  rj = robot_of.get(edge.j)
  if ri is None or rj is None or ri == rj:
    return None
  return (int(edge.i), int(edge.j)) if int(edge.i) <= int(edge.j) else (
      int(edge.j), int(edge.i))


def _separator_pose_block_graph(graph_edges: list[Edge],
                                robot_of: dict[int, int]):
  nodes: set[int] = set()
  edges: set[tuple[int, int]] = set()
  for edge in graph_edges:
    block_edge = _separator_pose_block_edge(edge, robot_of)
    if block_edge is None:
      continue
    nodes.add(block_edge[0])
    nodes.add(block_edge[1])
    edges.add(block_edge)
  return nodes, edges


def _edge_normal_block_weight(edge: Edge, weighted: bool, cost_mode: str):
  tau, kappa = edge_weights_from_info(edge, cost_mode, weighted)
  dim = int(edge.dim)
  return max(0.0, float(dim) * float(tau) + float(dim * dim) * float(kappa))


def _pose_normal_stiffness(graph_edges: list[Edge],
                           edge_indices: set[int],
                           robot_of: dict[int, int],
                           weighted: bool,
                           cost_mode: str):
  stiffness: dict[int, float] = {}
  for edge_index in edge_indices:
    edge = graph_edges[int(edge_index)]
    weight = _edge_normal_block_weight(edge, weighted, cost_mode)
    if edge.i in robot_of:
      stiffness[int(edge.i)] = stiffness.get(int(edge.i), 0.0) + weight
    if edge.j in robot_of:
      stiffness[int(edge.j)] = stiffness.get(int(edge.j), 0.0) + weight
  return stiffness


def _pose_normal_edge_weights(graph_edges: list[Edge],
                              edge_indices: set[int],
                              robot_of: dict[int, int],
                              weighted: bool,
                              cost_mode: str):
  edge_weights: dict[tuple[int, int], float] = {}
  for edge_index in edge_indices:
    edge = graph_edges[int(edge_index)]
    block_edge = _separator_pose_block_edge(edge, robot_of)
    if block_edge is None:
      continue
    edge_weights[block_edge] = (
        edge_weights.get(block_edge, 0.0) +
        _edge_normal_block_weight(edge, weighted, cost_mode)
    )
  return edge_weights


def _candidate_endpoint_leverage(candidate: dict,
                                 pose_stiffness: dict[int, float],
                                 weighted_gain: float):
  block_edge = candidate.get("pose_block_edge")
  if block_edge is None:
    return 0.0
  i, j = block_edge
  inverse_stiffness = (
      1.0 / (1.0 + float(pose_stiffness.get(int(i), 0.0))) +
      1.0 / (1.0 + float(pose_stiffness.get(int(j), 0.0)))
  )
  return float(weighted_gain) * inverse_stiffness


def _candidate_endpoint_guarded_leverage(candidate: dict,
                                         pose_stiffness: dict[int, float],
                                         weighted_gain: float):
  block_edge = candidate.get("pose_block_edge")
  if block_edge is None:
    return 0.0
  i, j = block_edge
  inverse_stiffness = (
      1.0 / (1.0 + float(pose_stiffness.get(int(i), 0.0))) +
      1.0 / (1.0 + float(pose_stiffness.get(int(j), 0.0)))
  )
  return float(weighted_gain) * (1.0 + min(1.0, inverse_stiffness))


def _pose_block_union_find(edges: set[tuple[int, int]]):
  parent: dict[int, int] = {}
  rank: dict[int, int] = {}

  def ensure(node: int):
    node = int(node)
    if node not in parent:
      parent[node] = node
      rank[node] = 0
    return node

  def find(node: int):
    node = ensure(node)
    if parent[node] != node:
      parent[node] = find(parent[node])
    return parent[node]

  def union(i: int, j: int):
    ri = find(i)
    rj = find(j)
    if ri == rj:
      return
    if rank[ri] < rank[rj]:
      ri, rj = rj, ri
    parent[rj] = ri
    if rank[ri] == rank[rj]:
      rank[ri] += 1

  for i, j in edges:
    union(i, j)
  return set(parent.keys()), find, union


def _candidate_selected_normal_leverage(candidate: dict,
                                        selected_pose_block_nodes: set[int],
                                        find_pose_block_root,
                                        weighted_gain: float):
  block_edge = candidate.get("pose_block_edge")
  if block_edge is None:
    return 0.0
  i, j = int(block_edge[0]), int(block_edge[1])
  missing_endpoints = int(i not in selected_pose_block_nodes) + int(
      j not in selected_pose_block_nodes)
  component_gap = 0
  if missing_endpoints == 0 and find_pose_block_root(i) != find_pose_block_root(j):
    component_gap = 1
  normal_gap = min(2, missing_endpoints + component_gap)
  return float(weighted_gain) * float(1 + normal_gap)


def _candidate_block_jacobi_leverage(candidate: dict,
                                     pose_stiffness: dict[int, float],
                                     selected_normal_gain: float):
  block_edge = candidate.get("pose_block_edge")
  if block_edge is None:
    return 0.0
  i, j = int(block_edge[0]), int(block_edge[1])
  edge_weight = max(0.0, float(candidate.get("normal_block_weight", 0.0)))
  inverse_stiffness = (
      1.0 / (1.0 + float(pose_stiffness.get(i, 0.0))) +
      1.0 / (1.0 + float(pose_stiffness.get(j, 0.0)))
  )
  jacobi_q = edge_weight * inverse_stiffness
  bounded_leverage = jacobi_q / (1.0 + jacobi_q)
  return float(selected_normal_gain) * (1.0 + bounded_leverage)


def _selected_interface_path_resistance(
    src: int,
    dst: int,
    selected_pose_block_edge_weight: dict[tuple[int, int], float],
    max_hops: int = 4):
  if int(src) == int(dst):
    return 0.0
  adjacency: dict[int, list[tuple[int, float]]] = {}
  for edge, weight in selected_pose_block_edge_weight.items():
    i, j = int(edge[0]), int(edge[1])
    resistance = 1.0 / (1.0 + max(0.0, float(weight)))
    adjacency.setdefault(i, []).append((j, resistance))
    adjacency.setdefault(j, []).append((i, resistance))
  if int(src) not in adjacency or int(dst) not in adjacency:
    return 0.0
  queue: list[tuple[float, int, int]] = []
  heappush(queue, (0.0, 0, int(src)))
  best: dict[tuple[int, int], float] = {(int(src), 0): 0.0}
  while queue:
    distance, hops, node = heappop(queue)
    if node == int(dst):
      return float(distance)
    if hops >= int(max_hops):
      continue
    if distance > best.get((node, hops), float("inf")) + 1e-18:
      continue
    for nbr, edge_resistance in adjacency.get(node, []):
      next_state = (int(nbr), hops + 1)
      next_distance = float(distance) + float(edge_resistance)
      if next_distance + 1e-18 < best.get(next_state, float("inf")):
        best[next_state] = next_distance
        heappush(queue, (next_distance, hops + 1, int(nbr)))
  return 0.0


def _candidate_interface_leverage(
    candidate: dict,
    selected_pose_block_edge_weight: dict[tuple[int, int], float],
    selected_normal_gain: float):
  block_edge = candidate.get("pose_block_edge")
  if block_edge is None:
    return 0.0
  path_resistance = _selected_interface_path_resistance(
      int(block_edge[0]), int(block_edge[1]), selected_pose_block_edge_weight)
  if path_resistance <= 0.0:
    return float(selected_normal_gain)
  edge_weight = max(0.0, float(candidate.get("normal_block_weight", 0.0)))
  schur_q = edge_weight * path_resistance
  bounded_interface_gain = schur_q / (1.0 + schur_q)
  return float(selected_normal_gain) * (1.0 + bounded_interface_gain)


def _bounded_interface_nodes(src: int,
                             dst: int,
                             selected_pose_block_edge_weight: dict[tuple[int, int], float],
                             max_hops: int = 3,
                             max_nodes: int = 36):
  adjacency: dict[int, set[int]] = {}
  for edge in selected_pose_block_edge_weight:
    i, j = int(edge[0]), int(edge[1])
    adjacency.setdefault(i, set()).add(j)
    adjacency.setdefault(j, set()).add(i)
  if int(src) not in adjacency or int(dst) not in adjacency:
    return set()
  nodes = {int(src), int(dst)}
  queue = deque([(int(src), 0), (int(dst), 0)])
  while queue and len(nodes) <= int(max_nodes):
    node, hops = queue.popleft()
    if hops >= int(max_hops):
      continue
    for nbr in sorted(adjacency.get(node, ())):
      if nbr in nodes:
        continue
      nodes.add(int(nbr))
      queue.append((int(nbr), hops + 1))
      if len(nodes) > int(max_nodes):
        return set()
  return nodes


def _selected_interface_effective_resistance(
    src: int,
    dst: int,
    selected_pose_block_edge_weight: dict[tuple[int, int], float],
    max_hops: int = 3,
    max_nodes: int = 36):
  if int(src) == int(dst):
    return 0.0
  nodes = _bounded_interface_nodes(
      src, dst, selected_pose_block_edge_weight, max_hops, max_nodes)
  if not nodes or int(src) not in nodes or int(dst) not in nodes:
    return 0.0
  indexed_nodes = sorted(nodes)
  index = {node: offset for offset, node in enumerate(indexed_nodes)}
  laplacian = np.zeros((len(indexed_nodes), len(indexed_nodes)), dtype=float)
  edge_count = 0
  for edge, weight in selected_pose_block_edge_weight.items():
    i, j = int(edge[0]), int(edge[1])
    if i not in index or j not in index:
      continue
    conductance = 1.0 + max(0.0, float(weight))
    ii, jj = index[i], index[j]
    laplacian[ii, ii] += conductance
    laplacian[jj, jj] += conductance
    laplacian[ii, jj] -= conductance
    laplacian[jj, ii] -= conductance
    edge_count += 1
  if edge_count == 0:
    return 0.0
  b = np.zeros(len(indexed_nodes), dtype=float)
  b[index[int(src)]] = 1.0
  b[index[int(dst)]] = -1.0
  ground = len(indexed_nodes) - 1
  keep = [idx for idx in range(len(indexed_nodes)) if idx != ground]
  reduced_laplacian = laplacian[np.ix_(keep, keep)]
  reduced_b = b[keep]
  try:
    reduced_x = np.linalg.solve(
        reduced_laplacian + 1e-12 * np.eye(len(keep)), reduced_b)
  except np.linalg.LinAlgError:
    reduced_x = np.linalg.lstsq(
        reduced_laplacian + 1e-9 * np.eye(len(keep)), reduced_b,
        rcond=None)[0]
  x = np.zeros(len(indexed_nodes), dtype=float)
  x[keep] = reduced_x
  resistance = float(b @ x)
  if not np.isfinite(resistance) or resistance <= 0.0:
    return 0.0
  return resistance


def _candidate_schur_leverage(
    candidate: dict,
    selected_pose_block_edge_weight: dict[tuple[int, int], float],
    selected_normal_gain: float):
  block_edge = candidate.get("pose_block_edge")
  if block_edge is None:
    return 0.0
  resistance = _selected_interface_effective_resistance(
      int(block_edge[0]), int(block_edge[1]), selected_pose_block_edge_weight)
  if resistance <= 0.0:
    return float(selected_normal_gain)
  edge_weight = max(0.0, float(candidate.get("normal_block_weight", 0.0)))
  schur_q = edge_weight * resistance
  bounded_schur_gain = schur_q / (1.0 + schur_q)
  return float(selected_normal_gain) * (1.0 + bounded_schur_gain)


def choose_adaptive_bridge_density_subscheduler(normal_evidence: dict,
                                                density_evidence: dict):
  normal_bridge_gain = int(
      normal_evidence.get("relay_scheduler_structural_bridge_gain", 0))
  density_bridge_gain = int(
      density_evidence.get("relay_scheduler_structural_bridge_gain", 0))
  normal_gain = float(
      normal_evidence.get(
          "relay_scheduler_structural_normal_leverage_gain", 0.0))
  density_gain = float(
      density_evidence.get(
          "relay_scheduler_structural_normal_leverage_gain", 0.0))
  if density_bridge_gain >= normal_bridge_gain and density_gain > normal_gain:
    return "pose_summary_normal_density_budget"
  return "pose_summary_normal_leverage_budget"


def _pose_block_component_count(nodes: set[int], edges: set[tuple[int, int]]):
  if not nodes:
    return 0
  adjacency = {node: set() for node in nodes}
  for i, j in edges:
    if i not in nodes or j not in nodes:
      continue
    adjacency[i].add(j)
    adjacency[j].add(i)
  seen = set()
  component_count = 0
  for start in sorted(nodes):
    if start in seen:
      continue
    component_count += 1
    stack = [start]
    seen.add(start)
    while stack:
      node = stack.pop()
      for nbr in adjacency[node]:
        if nbr in seen:
          continue
        seen.add(nbr)
        stack.append(nbr)
  return component_count


def _pose_block_bridge_edges(nodes: set[int], edges: set[tuple[int, int]]):
  adjacency = {node: set() for node in nodes}
  for i, j in edges:
    if i not in nodes or j not in nodes:
      continue
    adjacency[i].add(j)
    adjacency[j].add(i)
  bridges: set[tuple[int, int]] = set()
  discovery: dict[int, int] = {}
  low: dict[int, int] = {}
  time = 0

  def visit(node: int, parent: int | None):
    nonlocal time
    discovery[node] = time
    low[node] = time
    time += 1
    for nbr in sorted(adjacency[node]):
      if nbr == parent:
        continue
      if nbr not in discovery:
        visit(nbr, node)
        low[node] = min(low[node], low[nbr])
        if low[nbr] > discovery[node]:
          bridges.add(_normalize_pair(node, nbr))
      else:
        low[node] = min(low[node], discovery[nbr])

  for node in sorted(nodes):
    if node not in discovery:
      visit(node, None)
  return bridges


def load_topology_pairs(path: Path, round_index: int = 0, round_window: int = 1):
  if round_window < 1:
    raise ValueError("round_window must be positive")
  first_round = int(round_index)
  last_round_exclusive = first_round + int(round_window)
  pairs: set[tuple[int, int]] = set()
  with Path(path).open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      row_round = int(row.get("round", 0))
      if row_round < first_round or row_round >= last_round_exclusive:
        continue
      pairs.add(_normalize_pair(int(row["src"]), int(row["dst"])))
  return pairs


def _shortest_robot_path(src: int,
                         dst: int,
                         available_robot_pairs: set[tuple[int, int]],
                         max_path_hops: int):
  if src == dst:
    return [src]
  adjacency: dict[int, list[int]] = {}
  for i, j in available_robot_pairs:
    a, b = _normalize_pair(i, j)
    adjacency.setdefault(a, []).append(b)
    adjacency.setdefault(b, []).append(a)
  queue = deque([(src, [src])])
  seen = {src}
  while queue:
    node, path = queue.popleft()
    if len(path) - 1 >= max_path_hops:
      continue
    for nbr in sorted(adjacency.get(node, [])):
      if nbr in seen:
        continue
      next_path = path + [nbr]
      if nbr == dst:
        return next_path
      seen.add(nbr)
      queue.append((nbr, next_path))
  return None


def _pose_summary_relay_unit_mb(rotation_iterations: int,
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


def _pose_summary_units(edge: Edge, robot_of: dict[int, int], path: list[int]):
  ri = int(robot_of[edge.i])
  rj = int(robot_of[edge.j])
  path_tuple = tuple(int(robot) for robot in path)
  return {
      (int(edge.i), ri, rj, path_tuple),
      (int(edge.j), rj, ri, tuple(reversed(path_tuple))),
  }


def topology_edge_evidence(graph_edges: list[Edge],
                           robot_of: dict[int, int],
                           available_robot_pairs: set[tuple[int, int]] | None,
                           max_relay_hops: int = 0,
                           relay_scheduler: str = "none",
                           relay_byte_budget_mb: float | None = None,
                           baseline_poses: dict[int, object] | None = None,
                           rotation_iterations: int = 0,
                           translation_iterations: int = 0,
                           weighted: bool = False,
                           cost_mode: str = "dpgo"):
  scheduler_start = time.perf_counter()
  if available_robot_pairs is None:
    separator_edges = _separator_edge_count(graph_edges, robot_of)
    return list(graph_edges), {
        "covered_separator_edges": separator_edges,
        "direct_separator_edges": separator_edges,
        "relayed_separator_edges": 0,
        "relay_extra_hop_count": 0,
        "relay_scheduler": relay_scheduler,
        "relay_scheduler_selected_relay_edges": 0,
        "relay_scheduler_skipped_relay_edges": 0,
        "relay_scheduler_reachable_relay_edges": 0,
        "relay_scheduler_byte_budget_mb": relay_byte_budget_mb,
        "relay_scheduler_predicted_selected_relay_mb": 0.0,
        "relay_scheduler_pose_summary_units": 0,
        "relay_scheduler_structural_bridge_gain": 0,
        "relay_scheduler_structural_component_gain": 0,
        "relay_scheduler_structural_weighted_gain": 0.0,
        "relay_scheduler_structural_leverage_gain": 0.0,
        "relay_scheduler_structural_guarded_leverage_gain": 0.0,
        "relay_scheduler_structural_normal_leverage_gain": 0.0,
        "relay_scheduler_structural_jacobi_leverage_gain": 0.0,
        "relay_scheduler_structural_interface_leverage_gain": 0.0,
        "relay_scheduler_structural_schur_leverage_gain": 0.0,
        "relay_scheduler_structural_guarded_schur_leverage_gain": 0.0,
        "relay_scheduler_structural_lazy_guarded_schur_leverage_gain": 0.0,
        "relay_scheduler_adaptive_selected_subscheduler": None,
        "relay_scheduler_adaptive_normal_bridge_gain": 0,
        "relay_scheduler_adaptive_density_bridge_gain": 0,
        "relay_scheduler_adaptive_normal_normal_leverage_gain": 0.0,
        "relay_scheduler_adaptive_density_normal_leverage_gain": 0.0,
        "relay_scheduler_schur_eval_count": 0,
        "relay_scheduler_wall_sec": time.perf_counter() - scheduler_start,
        "separator_paths": [],
        "max_relay_hops": max_relay_hops,
    }
  if max_relay_hops < 0:
    raise ValueError("max_relay_hops must be nonnegative")
  if relay_scheduler not in {
      "none",
      "edge_residual_budget",
      "edge_coverage_budget",
      "pose_summary_budget",
      "pose_summary_criticality_budget",
      "pose_summary_weighted_criticality_budget",
      "pose_summary_leverage_budget",
      "pose_summary_guarded_leverage_budget",
      "pose_summary_normal_leverage_budget",
      "pose_summary_normal_density_budget",
      "pose_summary_bridge_normal_budget",
      "pose_summary_adaptive_bridge_density_budget",
      "pose_summary_jacobi_leverage_budget",
      "pose_summary_interface_leverage_budget",
      "pose_summary_schur_leverage_budget",
      "pose_summary_guarded_schur_leverage_budget",
      "pose_summary_lazy_guarded_schur_leverage_budget",
  }:
    raise ValueError(f"unsupported relay scheduler: {relay_scheduler}")
  if relay_scheduler == "edge_residual_budget" and baseline_poses is None:
    raise ValueError("baseline_poses is required for edge_residual_budget")
  if relay_scheduler == "pose_summary_adaptive_bridge_density_budget":
    normal_filtered, normal_evidence = topology_edge_evidence(
        graph_edges=graph_edges,
        robot_of=robot_of,
        available_robot_pairs=available_robot_pairs,
        max_relay_hops=max_relay_hops,
        relay_scheduler="pose_summary_normal_leverage_budget",
        relay_byte_budget_mb=relay_byte_budget_mb,
        baseline_poses=baseline_poses,
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        weighted=weighted,
        cost_mode=cost_mode)
    density_filtered, density_evidence = topology_edge_evidence(
        graph_edges=graph_edges,
        robot_of=robot_of,
        available_robot_pairs=available_robot_pairs,
        max_relay_hops=max_relay_hops,
        relay_scheduler="pose_summary_normal_density_budget",
        relay_byte_budget_mb=relay_byte_budget_mb,
        baseline_poses=baseline_poses,
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        weighted=weighted,
        cost_mode=cost_mode)
    selected_subscheduler = choose_adaptive_bridge_density_subscheduler(
        normal_evidence, density_evidence)
    if selected_subscheduler == "pose_summary_normal_density_budget":
      selected_filtered = density_filtered
      selected_evidence = dict(density_evidence)
    else:
      selected_filtered = normal_filtered
      selected_evidence = dict(normal_evidence)
    selected_evidence.update({
        "relay_scheduler": relay_scheduler,
        "relay_scheduler_adaptive_selected_subscheduler": selected_subscheduler,
        "relay_scheduler_adaptive_normal_bridge_gain":
            normal_evidence.get("relay_scheduler_structural_bridge_gain", 0),
        "relay_scheduler_adaptive_density_bridge_gain":
            density_evidence.get("relay_scheduler_structural_bridge_gain", 0),
        "relay_scheduler_adaptive_normal_normal_leverage_gain":
            normal_evidence.get(
                "relay_scheduler_structural_normal_leverage_gain", 0.0),
        "relay_scheduler_adaptive_density_normal_leverage_gain":
            density_evidence.get(
                "relay_scheduler_structural_normal_leverage_gain", 0.0),
        "relay_scheduler_schur_eval_count": (
            int(normal_evidence.get("relay_scheduler_schur_eval_count", 0)) +
            int(density_evidence.get("relay_scheduler_schur_eval_count", 0))
        ),
        "relay_scheduler_wall_sec": time.perf_counter() - scheduler_start,
    })
    return selected_filtered, selected_evidence
  available = {_normalize_pair(src, dst) for src, dst in available_robot_pairs}
  max_path_hops = max(1, int(max_relay_hops) + 1)
  dim = pose_dimension_from_edges(graph_edges)
  selected_edge_indices = set()
  direct_paths = []
  relay_candidates = []

  def relay_cost_mb(path_hops: int):
    return float(estimate_relay_extra_communication(
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        relay_extra_hop_count=max(0, path_hops - 1),
        dimension=dim,
    )["relay_extra_exchange_mb"])

  direct_separator_edges = 0
  for edge_index, edge in enumerate(graph_edges):
    ri = robot_of.get(edge.i)
    rj = robot_of.get(edge.j)
    if ri is None or rj is None or ri == rj:
      selected_edge_indices.add(edge_index)
      continue
    path = _shortest_robot_path(ri, rj, available, max_path_hops)
    if path is None:
      continue
    path_hops = len(path) - 1
    if path_hops == 1:
      direct_separator_edges += 1
      selected_edge_indices.add(edge_index)
      direct_paths.append({
          "edge_index": edge_index,
          "robots": path,
          "path_hops": path_hops,
      })
      continue
    cost = 0.0
    if baseline_poses is not None:
      cost, ok = edge_chordal_cost(edge, baseline_poses, weighted, cost_mode)
      if not ok:
        cost = 0.0
    extra_mb = relay_cost_mb(path_hops)
    unit_mb = _pose_summary_relay_unit_mb(
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        extra_hops=path_hops - 1,
        dimension=dim,
    )
    pose_units = _pose_summary_units(edge, robot_of, path)
    relay_candidates.append({
        "edge_index": edge_index,
        "robots": path,
        "path_hops": path_hops,
        "extra_hops": path_hops - 1,
        "baseline_cost": float(cost),
        "relay_extra_comm_mb": extra_mb,
        "value_per_mb": float(cost) / max(extra_mb, 1e-12),
        "pose_summary_units": pose_units,
        "pose_summary_unit_costs": {unit: unit_mb for unit in pose_units},
        "pose_block_edge": _separator_pose_block_edge(edge, robot_of),
        "normal_block_weight": _edge_normal_block_weight(
            edge, weighted, cost_mode),
    })

  selected_relay_paths = []
  selected_pose_summary_units = set()
  selected_pose_summary_mb = None
  selected_pose_block_edges = {
      block_edge for block_edge in (
          _separator_pose_block_edge(graph_edges[edge_index], robot_of)
          for edge_index in selected_edge_indices
      )
      if block_edge is not None
  }
  selected_pose_block_nodes, find_pose_block_root, union_pose_block = (
      _pose_block_union_find(selected_pose_block_edges))
  full_pose_block_nodes, full_pose_block_edges = _separator_pose_block_graph(
      graph_edges, robot_of)
  full_pose_block_bridges = _pose_block_bridge_edges(
      full_pose_block_nodes, full_pose_block_edges)
  pose_stiffness = _pose_normal_stiffness(
      graph_edges, selected_edge_indices, robot_of, weighted, cost_mode)
  selected_pose_block_edge_weight = _pose_normal_edge_weights(
      graph_edges, selected_edge_indices, robot_of, weighted, cost_mode)
  structural_bridge_gain = 0
  structural_component_gain = 0
  structural_weighted_gain = 0.0
  structural_leverage_gain = 0.0
  structural_guarded_leverage_gain = 0.0
  structural_normal_leverage_gain = 0.0
  structural_jacobi_leverage_gain = 0.0
  structural_interface_leverage_gain = 0.0
  structural_schur_leverage_gain = 0.0
  structural_guarded_schur_leverage_gain = 0.0
  structural_lazy_guarded_schur_leverage_gain = 0.0
  schur_eval_count = 0
  if relay_scheduler == "none":
    selected_relay_paths = relay_candidates
  elif relay_scheduler == "edge_residual_budget":
    budget_mb = float("inf") if relay_byte_budget_mb is None else max(
        0.0, float(relay_byte_budget_mb))
    spent_mb = 0.0
    for candidate in sorted(
        relay_candidates,
        key=lambda item: (-item["value_per_mb"], item["edge_index"])):
      extra_mb = float(candidate["relay_extra_comm_mb"])
      if spent_mb + extra_mb <= budget_mb + 1e-15:
        selected_relay_paths.append(candidate)
        spent_mb += extra_mb
  elif relay_scheduler == "edge_coverage_budget":
    budget_mb = float("inf") if relay_byte_budget_mb is None else max(
        0.0, float(relay_byte_budget_mb))
    spent_mb = 0.0
    for candidate in sorted(
        relay_candidates,
        key=lambda item: (
            float(item["relay_extra_comm_mb"]),
            -float(item["value_per_mb"]),
            item["edge_index"],
        )):
      extra_mb = float(candidate["relay_extra_comm_mb"])
      if spent_mb + extra_mb <= budget_mb + 1e-15:
        selected_relay_paths.append(candidate)
        spent_mb += extra_mb
  else:
    budget_mb = float("inf") if relay_byte_budget_mb is None else max(
        0.0, float(relay_byte_budget_mb))
    spent_mb = 0.0
    remaining = list(relay_candidates)
    while remaining:
      best = None
      best_key = None
      best_incremental_mb = 0.0
      best_bridge_gain = 0
      best_component_gain = 0
      best_weighted_gain = 0.0
      best_leverage_gain = 0.0
      best_guarded_leverage_gain = 0.0
      best_normal_leverage_gain = 0.0
      best_jacobi_leverage_gain = 0.0
      best_interface_leverage_gain = 0.0
      best_schur_leverage_gain = 0.0
      best_guarded_schur_leverage_gain = 0.0
      best_lazy_guarded_schur_leverage_gain = 0.0
      if relay_scheduler == "pose_summary_lazy_guarded_schur_leverage_budget":
        feasible = []
        for candidate in remaining:
          incremental_mb = sum(
              float(candidate["pose_summary_unit_costs"][unit])
              for unit in candidate["pose_summary_units"]
              if unit not in selected_pose_summary_units
          )
          if spent_mb + incremental_mb > budget_mb + 1e-15:
            continue
          block_edge = candidate.get("pose_block_edge")
          bridge_gain = int(
              block_edge in full_pose_block_bridges and
              block_edge not in selected_pose_block_edges)
          component_gain = bridge_gain
          weighted_gain = (
              float(candidate.get("normal_block_weight", 0.0)) *
              float(1 + bridge_gain + component_gain)
          )
          leverage_gain = _candidate_endpoint_leverage(
              candidate, pose_stiffness, weighted_gain)
          guarded_leverage_gain = _candidate_endpoint_guarded_leverage(
              candidate, pose_stiffness, weighted_gain)
          normal_leverage_gain = _candidate_selected_normal_leverage(
              candidate, selected_pose_block_nodes, find_pose_block_root,
              weighted_gain)
          feasible.append({
              "candidate": candidate,
              "incremental_mb": incremental_mb,
              "bridge_gain": bridge_gain,
              "component_gain": component_gain,
              "weighted_gain": weighted_gain,
              "normal_leverage_gain": normal_leverage_gain,
          })
        if not feasible:
          break
        best_prefix = min(
            (
                item["incremental_mb"],
                -item["normal_leverage_gain"],
                -item["weighted_gain"],
            )
            for item in feasible
        )
        tied = [
            item for item in feasible
            if (
                item["incremental_mb"],
                -item["normal_leverage_gain"],
                -item["weighted_gain"],
            ) == best_prefix
        ]
        if len(tied) > 1:
          for item in tied:
            item["lazy_guarded_schur_leverage_gain"] = _candidate_schur_leverage(
                item["candidate"], selected_pose_block_edge_weight,
                item["normal_leverage_gain"])
          schur_eval_count += len(tied)
        else:
          tied[0]["lazy_guarded_schur_leverage_gain"] = tied[0][
              "normal_leverage_gain"]
        chosen = min(
            tied,
            key=lambda item: (
                -item["lazy_guarded_schur_leverage_gain"],
                -item["bridge_gain"],
                -item["component_gain"],
                -float(item["candidate"]["value_per_mb"]),
                int(item["candidate"]["edge_index"]),
            ))
        best = chosen["candidate"]
        best_incremental_mb = chosen["incremental_mb"]
        best_bridge_gain = chosen["bridge_gain"]
        best_component_gain = chosen["component_gain"]
        best_weighted_gain = chosen["weighted_gain"]
        best_normal_leverage_gain = chosen["normal_leverage_gain"]
        best_lazy_guarded_schur_leverage_gain = chosen[
            "lazy_guarded_schur_leverage_gain"]
      else:
        for candidate in remaining:
          incremental_mb = sum(
              float(candidate["pose_summary_unit_costs"][unit])
              for unit in candidate["pose_summary_units"]
              if unit not in selected_pose_summary_units
          )
          if spent_mb + incremental_mb > budget_mb + 1e-15:
            continue
          if relay_scheduler in {
              "pose_summary_criticality_budget",
              "pose_summary_weighted_criticality_budget",
              "pose_summary_leverage_budget",
              "pose_summary_guarded_leverage_budget",
              "pose_summary_normal_leverage_budget",
              "pose_summary_normal_density_budget",
              "pose_summary_bridge_normal_budget",
              "pose_summary_jacobi_leverage_budget",
              "pose_summary_interface_leverage_budget",
              "pose_summary_schur_leverage_budget",
              "pose_summary_guarded_schur_leverage_budget",
          }:
            block_edge = candidate.get("pose_block_edge")
            bridge_gain = int(
                block_edge in full_pose_block_bridges and
                block_edge not in selected_pose_block_edges)
            component_gain = bridge_gain
            weighted_gain = (
                float(candidate.get("normal_block_weight", 0.0)) *
                float(1 + bridge_gain + component_gain)
            )
            leverage_gain = 0.0
            guarded_leverage_gain = 0.0
            normal_leverage_gain = 0.0
            jacobi_leverage_gain = 0.0
            interface_leverage_gain = 0.0
            schur_leverage_gain = 0.0
            guarded_schur_leverage_gain = 0.0
            if relay_scheduler == "pose_summary_leverage_budget":
              leverage_gain = _candidate_endpoint_leverage(
                  candidate, pose_stiffness, weighted_gain)
            elif relay_scheduler == "pose_summary_guarded_leverage_budget":
              guarded_leverage_gain = _candidate_endpoint_guarded_leverage(
                  candidate, pose_stiffness, weighted_gain)
            if relay_scheduler in {
                "pose_summary_normal_leverage_budget",
                "pose_summary_normal_density_budget",
                "pose_summary_bridge_normal_budget",
                "pose_summary_jacobi_leverage_budget",
                "pose_summary_interface_leverage_budget",
                "pose_summary_schur_leverage_budget",
                "pose_summary_guarded_schur_leverage_budget",
            }:
              normal_leverage_gain = _candidate_selected_normal_leverage(
                  candidate, selected_pose_block_nodes, find_pose_block_root,
                  weighted_gain)
            if relay_scheduler == "pose_summary_jacobi_leverage_budget":
              jacobi_leverage_gain = _candidate_block_jacobi_leverage(
                  candidate, pose_stiffness, normal_leverage_gain)
            elif relay_scheduler == "pose_summary_interface_leverage_budget":
              interface_leverage_gain = _candidate_interface_leverage(
                  candidate, selected_pose_block_edge_weight,
                  normal_leverage_gain)
            elif relay_scheduler in {
                "pose_summary_schur_leverage_budget",
                "pose_summary_guarded_schur_leverage_budget",
            }:
              schur_leverage_gain = _candidate_schur_leverage(
                  candidate, selected_pose_block_edge_weight,
                  normal_leverage_gain)
              schur_eval_count += 1
              guarded_schur_leverage_gain = schur_leverage_gain
            if relay_scheduler == "pose_summary_leverage_budget":
              key = (
                  incremental_mb,
                  -leverage_gain,
                  -weighted_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_guarded_leverage_budget":
              key = (
                  incremental_mb,
                  -guarded_leverage_gain,
                  -weighted_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_normal_leverage_budget":
              key = (
                  incremental_mb,
                  -normal_leverage_gain,
                  -weighted_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_normal_density_budget":
              normal_density_gain = normal_leverage_gain / max(
                  incremental_mb, 1e-12)
              key = (
                  -normal_density_gain,
                  incremental_mb,
                  -normal_leverage_gain,
                  -weighted_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_bridge_normal_budget":
              key = (
                  -bridge_gain,
                  -component_gain,
                  incremental_mb,
                  -normal_leverage_gain,
                  -weighted_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_jacobi_leverage_budget":
              key = (
                  incremental_mb,
                  -jacobi_leverage_gain,
                  -normal_leverage_gain,
                  -weighted_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_interface_leverage_budget":
              key = (
                  incremental_mb,
                  -interface_leverage_gain,
                  -normal_leverage_gain,
                  -weighted_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_schur_leverage_budget":
              key = (
                  incremental_mb,
                  -schur_leverage_gain,
                  -normal_leverage_gain,
                  -weighted_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_guarded_schur_leverage_budget":
              key = (
                  incremental_mb,
                  -normal_leverage_gain,
                  -weighted_gain,
                  -schur_leverage_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            elif relay_scheduler == "pose_summary_weighted_criticality_budget":
              key = (
                  incremental_mb,
                  -weighted_gain,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
            else:
              key = (
                  incremental_mb,
                  -bridge_gain,
                  -component_gain,
                  -float(candidate["value_per_mb"]),
                  int(candidate["edge_index"]),
              )
          else:
            bridge_gain = 0
            component_gain = 0
            weighted_gain = 0.0
            leverage_gain = 0.0
            guarded_leverage_gain = 0.0
            normal_leverage_gain = 0.0
            jacobi_leverage_gain = 0.0
            interface_leverage_gain = 0.0
            schur_leverage_gain = 0.0
            guarded_schur_leverage_gain = 0.0
            key = (
                incremental_mb,
                -float(candidate["value_per_mb"]),
                int(candidate["edge_index"]),
            )
          if best is None or key < best_key:
            best = candidate
            best_key = key
            best_incremental_mb = incremental_mb
            best_bridge_gain = bridge_gain
            best_component_gain = component_gain
            best_weighted_gain = weighted_gain
            best_leverage_gain = leverage_gain
            best_guarded_leverage_gain = guarded_leverage_gain
            best_normal_leverage_gain = normal_leverage_gain
            best_jacobi_leverage_gain = jacobi_leverage_gain
            best_interface_leverage_gain = interface_leverage_gain
            best_schur_leverage_gain = schur_leverage_gain
            best_guarded_schur_leverage_gain = guarded_schur_leverage_gain
      if best is None:
        break
      selected_relay_paths.append(best)
      spent_mb += best_incremental_mb
      selected_pose_summary_units.update(best["pose_summary_units"])
      block_edge = best.get("pose_block_edge")
      if block_edge is not None:
        selected_pose_block_nodes.add(int(block_edge[0]))
        selected_pose_block_nodes.add(int(block_edge[1]))
        union_pose_block(int(block_edge[0]), int(block_edge[1]))
        selected_pose_block_edges.add(block_edge)
      best_edge_index = int(best["edge_index"])
      best_weight = float(best.get("normal_block_weight", 0.0))
      best_edge = graph_edges[best_edge_index]
      if best_edge.i in robot_of:
        pose_stiffness[int(best_edge.i)] = (
            pose_stiffness.get(int(best_edge.i), 0.0) + best_weight)
      if best_edge.j in robot_of:
        pose_stiffness[int(best_edge.j)] = (
            pose_stiffness.get(int(best_edge.j), 0.0) + best_weight)
      if block_edge is not None:
        selected_pose_block_edge_weight[block_edge] = (
            selected_pose_block_edge_weight.get(block_edge, 0.0) +
            best_weight)
      structural_bridge_gain += int(best_bridge_gain)
      structural_component_gain += int(best_component_gain)
      structural_weighted_gain += float(best_weighted_gain)
      structural_leverage_gain += float(best_leverage_gain)
      structural_guarded_leverage_gain += float(best_guarded_leverage_gain)
      structural_normal_leverage_gain += float(best_normal_leverage_gain)
      structural_jacobi_leverage_gain += float(best_jacobi_leverage_gain)
      structural_interface_leverage_gain += float(best_interface_leverage_gain)
      structural_schur_leverage_gain += float(best_schur_leverage_gain)
      structural_guarded_schur_leverage_gain += float(
          best_guarded_schur_leverage_gain)
      structural_lazy_guarded_schur_leverage_gain += float(
          best_lazy_guarded_schur_leverage_gain)
      remaining = [
          candidate for candidate in remaining
          if int(candidate["edge_index"]) != int(best["edge_index"])
      ]
    selected_pose_summary_mb = spent_mb
  for candidate in selected_relay_paths:
    selected_edge_indices.add(int(candidate["edge_index"]))
  selected_relay_indices = {int(item["edge_index"]) for item in selected_relay_paths}
  filtered = [
      edge for edge_index, edge in enumerate(graph_edges)
      if edge_index in selected_edge_indices
  ]
  relay_extra_hop_count = sum(int(item["extra_hops"]) for item in selected_relay_paths)
  relay_extra_mb = sum(float(item["relay_extra_comm_mb"]) for item in selected_relay_paths)
  if selected_pose_summary_mb is not None:
    relay_extra_mb = selected_pose_summary_mb
  separator_paths = direct_paths + [
      {
          "edge_index": int(item["edge_index"]),
          "robots": item["robots"],
          "path_hops": int(item["path_hops"]),
      }
      for item in selected_relay_paths
  ]
  return filtered, {
      "covered_separator_edges": direct_separator_edges + len(selected_relay_paths),
      "direct_separator_edges": direct_separator_edges,
      "relayed_separator_edges": len(selected_relay_paths),
      "relay_extra_hop_count": relay_extra_hop_count,
      "relay_scheduler": relay_scheduler,
      "relay_scheduler_selected_relay_edges": len(selected_relay_paths),
      "relay_scheduler_skipped_relay_edges": (
          len(relay_candidates) - len(selected_relay_paths)
      ),
      "relay_scheduler_reachable_relay_edges": len(relay_candidates),
      "relay_scheduler_byte_budget_mb": relay_byte_budget_mb,
      "relay_scheduler_predicted_selected_relay_mb": relay_extra_mb,
      "relay_scheduler_pose_summary_units": len(selected_pose_summary_units),
      "relay_scheduler_structural_bridge_gain": structural_bridge_gain,
      "relay_scheduler_structural_component_gain": structural_component_gain,
      "relay_scheduler_structural_weighted_gain": structural_weighted_gain,
      "relay_scheduler_structural_leverage_gain": structural_leverage_gain,
      "relay_scheduler_structural_guarded_leverage_gain":
          structural_guarded_leverage_gain,
      "relay_scheduler_structural_normal_leverage_gain":
          structural_normal_leverage_gain,
      "relay_scheduler_structural_jacobi_leverage_gain":
          structural_jacobi_leverage_gain,
      "relay_scheduler_structural_interface_leverage_gain":
          structural_interface_leverage_gain,
      "relay_scheduler_structural_schur_leverage_gain":
          structural_schur_leverage_gain,
      "relay_scheduler_structural_guarded_schur_leverage_gain":
          structural_guarded_schur_leverage_gain,
      "relay_scheduler_structural_lazy_guarded_schur_leverage_gain":
          structural_lazy_guarded_schur_leverage_gain,
      "relay_scheduler_schur_eval_count": schur_eval_count,
      "relay_scheduler_wall_sec": time.perf_counter() - scheduler_start,
      "relay_scheduler_selected_edge_indices": sorted(selected_relay_indices),
      "separator_paths": separator_paths,
      "max_relay_hops": max_relay_hops,
  }


def filter_edges_for_topology(graph_edges: list[Edge],
                              robot_of: dict[int, int],
                              available_robot_pairs: set[tuple[int, int]] | None):
  filtered, _ = topology_edge_evidence(
      graph_edges, robot_of, available_robot_pairs, max_relay_hops=0)
  return filtered


def estimate_relay_extra_communication(rotation_iterations: int,
                                       translation_iterations: int,
                                       relay_extra_hop_count: int,
                                       dimension: int):
  if relay_extra_hop_count <= 0:
    return {
        "relay_extra_hop_count": int(relay_extra_hop_count),
        "relay_extra_exchange_bytes": 0,
        "relay_extra_exchange_mb": 0.0,
        "model": "extra_separator_block_exchange_hops",
    }
  rot_bytes = (
      int(rotation_iterations) * int(relay_extra_hop_count) *
      2 * int(dimension * dimension) * 8
  )
  trans_bytes = (
      int(translation_iterations) * int(relay_extra_hop_count) *
      2 * int(dimension) * 8
  )
  total_bytes = rot_bytes + trans_bytes
  return {
      "relay_extra_hop_count": int(relay_extra_hop_count),
      "rotation_relay_extra_exchange_bytes": rot_bytes,
      "translation_relay_extra_exchange_bytes": trans_bytes,
      "relay_extra_exchange_bytes": total_bytes,
      "relay_extra_exchange_mb": float(total_bytes) / (1024.0 * 1024.0),
      "model": "extra_separator_block_exchange_hops",
  }


def estimate_topology_routed_normal_summary_communication(
    logical_payload_bytes: int | float,
    separator_paths: list[dict],
):
  logical_bytes = max(0.0, float(logical_payload_bytes))
  path_count = len(separator_paths)
  if logical_bytes <= 0.0 or path_count == 0:
    logical_int = int(round(logical_bytes))
    return {
        "model": "equal_separator_edge_payload_path_hop_routing",
        "logical_payload_bytes": logical_int,
        "logical_payload_mb": logical_int / (1024.0 * 1024.0),
        "routed_payload_bytes": logical_int,
        "routed_payload_mb": logical_int / (1024.0 * 1024.0),
        "extra_relay_payload_bytes": 0,
        "extra_relay_payload_mb": 0.0,
        "separator_path_count": path_count,
        "direct_path_count": 0,
        "relayed_path_count": 0,
        "total_path_hops": 0,
        "mean_path_hops": 0.0,
        "path_payload_bytes": 0.0,
    }
  total_path_hops = 0
  direct_path_count = 0
  for item in separator_paths:
    hops = max(1, int(item.get("path_hops", 1)))
    total_path_hops += hops
    if hops == 1:
      direct_path_count += 1
  path_payload_bytes = logical_bytes / float(path_count)
  routed_bytes = int(round(path_payload_bytes * float(total_path_hops)))
  logical_int = int(round(logical_bytes))
  extra_bytes = max(0, routed_bytes - logical_int)
  return {
      "model": "equal_separator_edge_payload_path_hop_routing",
      "logical_payload_bytes": logical_int,
      "logical_payload_mb": logical_int / (1024.0 * 1024.0),
      "routed_payload_bytes": routed_bytes,
      "routed_payload_mb": routed_bytes / (1024.0 * 1024.0),
      "extra_relay_payload_bytes": extra_bytes,
      "extra_relay_payload_mb": extra_bytes / (1024.0 * 1024.0),
      "separator_path_count": path_count,
      "direct_path_count": direct_path_count,
      "relayed_path_count": path_count - direct_path_count,
      "total_path_hops": total_path_hops,
      "mean_path_hops": float(total_path_hops) / float(path_count),
      "path_payload_bytes": path_payload_bytes,
  }


def estimate_budgeted_candidate_communication(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    rotation_iterations: int,
    translation_iterations: int,
    linear_solver: str,
    relay_extra_hop_count: int = 0,
    relay_extra_comm_mb: float | None = None,
    boundary_pose_unit_count: int | None = None,
    solve_communication_model: str = "boundary_pose_exchange",
    normal_summary_comm_mb: float = 0.0,
):
  if solve_communication_model not in {
      "boundary_pose_exchange",
      "compact_continuation_diagnostic",
  }:
    raise ValueError(
        f"unsupported solve_communication_model: {solve_communication_model}")
  dim = pose_dimension_from_edges(graph_edges)
  robot_count = len({robot_of[pose_id] for pose_id in pose_ids if pose_id in robot_of})
  separator_count = _separator_edge_count(graph_edges, robot_of)
  if linear_solver in {"pcg", "compact_pcg"}:
    rot_reductions = 2 * max(0, rotation_iterations)
    trans_reductions = 2 * max(0, translation_iterations)
  else:
    rot_reductions = 0
    trans_reductions = 0
  rotation_stats = {
      "iterations": max(0, rotation_iterations),
      "global_reduction_count": rot_reductions,
  }
  translation_stats = {
      "iterations": max(0, translation_iterations),
      "global_reduction_count": trans_reductions,
  }
  if linear_solver == "compact_pcg":
    linear_comm = estimate_dci_compact_pcg_solve_communication(
        rotation_stats,
        translation_stats,
        robot_count,
    )
  elif boundary_pose_unit_count is None:
    linear_comm = estimate_dci_linear_solve_communication(
        rotation_stats,
        translation_stats,
        robot_count,
        separator_count,
        dim,
    )
  else:
    linear_comm = estimate_dci_boundary_pose_solve_communication(
        rotation_stats,
        translation_stats,
        robot_count,
        int(boundary_pose_unit_count),
        dim,
    )
  cert_comm = estimate_residual_certificate_communication(robot_count)
  if relay_extra_comm_mb is None:
    relay_comm = estimate_relay_extra_communication(
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        relay_extra_hop_count=relay_extra_hop_count,
        dimension=dim,
    )
  else:
    relay_bytes = int(round(float(relay_extra_comm_mb) * 1024.0 * 1024.0))
    relay_comm = {
        "relay_extra_hop_count": int(relay_extra_hop_count),
        "relay_extra_exchange_bytes": relay_bytes,
        "relay_extra_exchange_mb": float(relay_extra_comm_mb),
        "model": "selected_topology_relay_payload",
    }
  effective_linear_mb = float(linear_comm.get(
      "linear_solve_total_estimated_mb", 0.0))
  if solve_communication_model == "compact_continuation_diagnostic":
    effective_linear_mb = float(linear_comm.get("pcg_global_reduction_mb", 0.0))
  total_mb = (
      effective_linear_mb +
      max(0.0, float(normal_summary_comm_mb)) +
      float(cert_comm.get("residual_delta_consensus_mb", 0.0)) +
      float(relay_comm.get("relay_extra_exchange_mb", 0.0))
  )
  return {
      "linear_solve": linear_comm,
      "certificate": cert_comm,
      "relay_extra": relay_comm,
      "normal_summary_comm_mb": max(0.0, float(normal_summary_comm_mb)),
      "effective_linear_solve_comm_mb": effective_linear_mb,
      "total_estimated_mb": total_mb,
      "solve_communication_model": solve_communication_model,
      "model": "pre_gate_max_iteration_budget",
  }


def compute_value_score(baseline_cost: dict,
                        predicted_comm_mb: float,
                        mode: str):
  if predicted_comm_mb <= 0.0:
    return float("inf")
  if mode == "raw_cost":
    return float(baseline_cost["total_cost"]) / predicted_comm_mb
  if mode == "mean_edge_residual":
    mean_edge_cost = (
        float(baseline_cost["total_cost"]) /
        max(1.0, float(baseline_cost.get("total_edges", 0)))
    )
    comm_per_separator = (
        predicted_comm_mb /
        max(1.0, float(baseline_cost.get("separator_edges", 0)))
    )
    return mean_edge_cost / max(comm_per_separator, 1e-12)
  if mode == "separator_private_contrast":
    mean_separator_cost = (
        float(baseline_cost.get("separator_cost", 0.0)) /
        max(1.0, float(baseline_cost.get("separator_edges", 0)))
    )
    mean_private_cost = (
        float(baseline_cost.get("private_cost", 0.0)) /
        max(1.0, float(baseline_cost.get("private_edges", 0)))
    )
    comm_per_separator = (
        predicted_comm_mb /
        max(1.0, float(baseline_cost.get("separator_edges", 0)))
    )
    return max(0.0, mean_separator_cost - mean_private_cost) / max(
        comm_per_separator, 1e-12)
  raise ValueError(f"unsupported value score mode: {mode}")


def compute_candidate_precert(baseline_cost: dict,
                              mode: str = "none",
                              min_sep_private_ratio: float = 0.0):
  if mode == "none":
    return {
        "candidate_precert_mode": mode,
        "candidate_precert_accept": True,
        "candidate_precert_sep_private_ratio": None,
        "candidate_precert_min_sep_private_ratio": min_sep_private_ratio,
    }
  if mode == "separator_private_ratio":
    mean_separator_cost = (
        float(baseline_cost.get("separator_cost", 0.0)) /
        max(1.0, float(baseline_cost.get("separator_edges", 0)))
    )
    mean_private_cost = (
        float(baseline_cost.get("private_cost", 0.0)) /
        max(1.0, float(baseline_cost.get("private_edges", 0)))
    )
    if mean_private_cost <= 1e-12:
      ratio = float("inf") if mean_separator_cost > 0.0 else 0.0
    else:
      ratio = mean_separator_cost / mean_private_cost
    return {
        "candidate_precert_mode": mode,
        "candidate_precert_accept": ratio >= min_sep_private_ratio,
        "candidate_precert_sep_private_ratio": ratio,
        "candidate_precert_mean_separator_cost": mean_separator_cost,
        "candidate_precert_mean_private_cost": mean_private_cost,
        "candidate_precert_min_sep_private_ratio": min_sep_private_ratio,
    }
  raise ValueError(f"unsupported candidate pre-cert mode: {mode}")


def compute_separator_residual_mass_coverage(full_baseline_cost: dict,
                                             selected_baseline_cost: dict,
                                             edge_count_coverage: float):
  full_cost = float(full_baseline_cost.get("separator_cost", 0.0))
  selected_cost = float(selected_baseline_cost.get("separator_cost", 0.0))
  if full_cost <= 1e-12:
    coverage = float(edge_count_coverage)
  else:
    coverage = max(0.0, min(1.0, selected_cost / full_cost))
  return {
      "separator_residual_mass_coverage": coverage,
      "full_separator_residual_cost": full_cost,
      "covered_separator_residual_cost": selected_cost,
      "missing_separator_residual_cost": max(0.0, full_cost - selected_cost),
  }


def compute_separator_normal_summary_payload_coverage(
    full_graph_edges: list[Edge],
    selected_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    baseline_poses: dict[int, object],
    dim: int,
    weighted: bool,
    cost_mode: str):
  rotations = {
      pose_id: baseline_poses[pose_id][:dim, :dim]
      for pose_id in pose_ids
      if pose_id in baseline_poses
  }
  anchor_pose = pose_ids[0] if pose_ids else None
  if anchor_pose is None:
    return {
        "separator_normal_summary_payload_coverage": 1.0,
        "full_separator_normal_summary_payload_bytes": 0,
        "covered_separator_normal_summary_payload_bytes": 0,
        "missing_separator_normal_summary_payload_bytes": 0,
  }
  full_summary = estimate_separator_normal_equation_summary_communication(
      graph_edges=full_graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      rotations=rotations,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose)
  selected_summary = estimate_separator_normal_equation_summary_communication(
      graph_edges=selected_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      rotations=rotations,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose)
  full_bytes = int(full_summary.get("total_summary_bytes", 0))
  selected_bytes = int(selected_summary.get("total_summary_bytes", 0))
  coverage = 1.0 if full_bytes <= 0 else max(
      0.0, min(1.0, float(selected_bytes) / float(full_bytes)))
  return {
      "separator_normal_summary_payload_coverage": coverage,
      "full_separator_normal_summary_payload_bytes": full_bytes,
      "covered_separator_normal_summary_payload_bytes": selected_bytes,
      "missing_separator_normal_summary_payload_bytes":
          max(0, full_bytes - selected_bytes),
  }


def compute_separator_normal_summary_block_graph_certificate(
    full_graph_edges: list[Edge],
    selected_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    baseline_poses: dict[int, object],
    dim: int,
    weighted: bool,
    cost_mode: str):
  rotations = {
      pose_id: baseline_poses[pose_id][:dim, :dim]
      for pose_id in pose_ids
      if pose_id in baseline_poses
  }
  anchor_pose = pose_ids[0] if pose_ids else None
  if anchor_pose is None:
    diagnostics = {
        "full_node_count": 0,
        "selected_node_count": 0,
        "full_diagonal_block_count": 0,
        "covered_diagonal_block_count": 0,
        "missing_diagonal_block_count": 0,
        "full_block_edge_count": 0,
        "covered_block_edge_count": 0,
        "missing_block_edge_count": 0,
        "missing_bridge_block_edge_count": 0,
        "full_component_count": 0,
        "selected_component_count": 0,
        "component_count_delta": 0,
        "structural_criticality": 0.0,
        "rotation": {},
        "translation": {},
        "model": "separator_normal_summary_block_graph_diagnostics",
    }
  else:
    diagnostics = separator_normal_summary_block_graph_diagnostics(
        full_graph_edges=full_graph_edges,
        selected_edges=selected_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        rotations=rotations,
        dim=dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose)
  return {
      "separator_normal_summary_block_graph": diagnostics,
      "separator_normal_summary_full_block_edges":
          int(diagnostics["full_block_edge_count"]),
      "separator_normal_summary_covered_block_edges":
          int(diagnostics["covered_block_edge_count"]),
      "separator_normal_summary_missing_block_edges":
          int(diagnostics["missing_block_edge_count"]),
      "separator_normal_summary_missing_bridge_block_edges":
          int(diagnostics["missing_bridge_block_edge_count"]),
      "separator_normal_summary_component_count_delta":
          int(diagnostics["component_count_delta"]),
      "separator_normal_summary_structural_criticality":
          float(diagnostics["structural_criticality"]),
  }


def run_budgeted_dci_for_graph_data(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    baseline_poses: dict[int, object],
    output_selected_estimate: Path,
    rotation_iterations: int,
    translation_iterations: int,
    value_threshold: float,
    weighted: bool,
    cost_mode: str,
    linear_solver: str,
    value_score_mode: str = "raw_cost",
    active_hops: int = 1,
    relaxation: float = 1.0,
    damping: float = 1e-12,
    interface_schur_preconditioner: str = "block_jacobi",
    interface_schur_rotation_preconditioner: str | None = None,
    interface_schur_translation_preconditioner: str | None = None,
    interface_schur_coarse_initial_guess: bool = False,
    interface_schur_coarse_basis_mode: str = "robot_coordinate",
    interface_schur_coarse_component_limit: int | None = 32,
    interface_schur_rotation_coarse_basis_mode: str | None = None,
    interface_schur_translation_coarse_basis_mode: str | None = None,
    interface_schur_rotation_coarse_component_limit: int | None = None,
    interface_schur_translation_coarse_component_limit: int | None = None,
    interface_schur_coarse_component_selection_mode: str = "size",
    interface_schur_rotation_coarse_component_selection_mode: str | None = None,
    interface_schur_translation_coarse_component_selection_mode:
        str | None = None,
    interface_schur_residual_deflation_rank: int = 0,
    interface_schur_residual_deflation_pilot_iterations: int = 0,
    interface_schur_ritz_rank: int = 0,
    interface_schur_ritz_probe_iterations: int = 0,
    interface_schur_rotation_ritz_rank: int | None = None,
    interface_schur_translation_ritz_rank: int | None = None,
    interface_schur_rotation_ritz_probe_iterations: int | None = None,
    interface_schur_translation_ritz_probe_iterations: int | None = None,
    interface_schur_rotation_ritz_rank_selection_mode: str = "fixed",
    interface_schur_translation_ritz_rank_selection_mode: str = "fixed",
    interface_schur_rotation_ritz_value_threshold: float | None = None,
    interface_schur_translation_ritz_value_threshold: float | None = None,
    interface_schur_rotation_ritz_energy_capture_fraction:
        float | None = None,
    interface_schur_translation_ritz_energy_capture_fraction:
        float | None = None,
    interface_schur_translation_budget_candidates:
        list[tuple[int, int]] | None = None,
    interface_schur_ritz_mode: str = "preconditioned_operator",
    available_robot_pairs: set[tuple[int, int]] | None = None,
    min_topology_separator_coverage: float = 0.0,
    topology_round_window: int = 1,
    max_relay_hops: int = 0,
    relay_scheduler: str = "none",
    relay_byte_budget_mb: float | None = None,
    relay_cost_rotation_iterations: int | None = None,
    relay_cost_translation_iterations: int | None = None,
    candidate_precert_mode: str = "none",
    candidate_precert_min_sep_private_ratio: float = 0.0,
    candidate_solver: str = "edge_subgraph",
    handoff_gate_mode: str = "evidence_cost",
    solve_communication_model: str = "boundary_pose_exchange",
    normal_summary_communication_model: str = "logical",
    summary_selection_mode: str = "all",
    summary_max_offdiag_block_edges: int | None = None,
    summary_refinement_max_offdiag_block_edges: int | None = None,
    omitted_force_correction_rounds: int = 1,
    omitted_force_correction_portfolio: bool = False,
    omitted_force_correction_rank_portfolio: bool = False,
    omitted_force_correction_curvature_model: str = "none",
    omitted_force_correction_curvature_rank: int = 1,
    omitted_force_correction_curvature_rank_scheduler: str = "fixed",
    omitted_force_correction_curvature_rank_max_condition: float | None = None,
    omitted_force_correction_curvature_rank_condition_policy: str = "fixed",
    omitted_force_correction_curvature_rank_condition_mad_scale: float = 3.0,
    omitted_force_correction_curvature_payload_model: str = "global_projected",
    omitted_force_correction_diagnose_subspace_miss: bool = False,
    omitted_force_correction_max_comm_mb: float | None = None,
    omitted_force_correction_min_marginal_cost_per_mb: float | None = None,
    central_equivalence_diagnostic: bool = False,
    central_equivalence_iterative_diagnostic_iterations: int = 0,
):
  if candidate_solver not in {"edge_subgraph", "normal_summary"}:
    raise ValueError(f"unsupported candidate_solver: {candidate_solver}")
  if handoff_gate_mode not in {"evidence_cost", "full_graph_cost"}:
    raise ValueError(f"unsupported handoff_gate_mode: {handoff_gate_mode}")
  if linear_solver == "compact_pcg" and candidate_solver != "normal_summary":
    raise ValueError("compact_pcg requires candidate_solver='normal_summary'")
  if (interface_schur_translation_budget_candidates is not None and
      candidate_solver != "normal_summary"):
    raise ValueError(
        "translation budget candidates require candidate_solver='normal_summary'")
  if normal_summary_communication_model not in {
      "logical",
      "topology_routed_equal_edge",
  }:
    raise ValueError(
        "unsupported normal_summary_communication_model: "
        f"{normal_summary_communication_model}")
  if summary_selection_mode not in {
      "all",
      "structural_spanning",
      "residual_force_refinement",
      "omitted_force_correction",
  }:
    raise ValueError(f"unsupported summary_selection_mode: {summary_selection_mode}")
  if summary_selection_mode != "all" and candidate_solver != "normal_summary":
    raise ValueError("summary selection requires candidate_solver='normal_summary'")
  if solve_communication_model not in {
      "boundary_pose_exchange",
      "compact_continuation_diagnostic",
  }:
    raise ValueError(
        f"unsupported solve_communication_model: {solve_communication_model}")
  relay_cost_rotation_iterations = (
      rotation_iterations if relay_cost_rotation_iterations is None
      else int(relay_cost_rotation_iterations)
  )
  relay_cost_translation_iterations = (
      translation_iterations if relay_cost_translation_iterations is None
      else int(relay_cost_translation_iterations)
  )
  pose_ids = sorted(pose_ids)
  dci_edges, topology_evidence = topology_edge_evidence(
      graph_edges=graph_edges,
      robot_of=robot_of,
      available_robot_pairs=available_robot_pairs,
      max_relay_hops=max_relay_hops,
      relay_scheduler=relay_scheduler,
      relay_byte_budget_mb=relay_byte_budget_mb,
      baseline_poses=baseline_poses,
      rotation_iterations=relay_cost_rotation_iterations,
      translation_iterations=relay_cost_translation_iterations,
      weighted=weighted,
      cost_mode=cost_mode)
  active_ids = active_pose_ids(graph_edges, robot_of, active_hops)
  baseline_cost = cost_breakdown(
      graph_edges, baseline_poses, robot_of, active_ids, weighted, cost_mode)
  dci_active_ids = active_pose_ids(dci_edges, robot_of, active_hops)
  dci_baseline_cost = cost_breakdown(
      dci_edges, baseline_poses, robot_of, dci_active_ids, weighted, cost_mode)
  dim = pose_dimension_from_edges(graph_edges)
  full_separator_edges = _separator_edge_count(graph_edges, robot_of)
  topology_separator_edges = int(topology_evidence["covered_separator_edges"])
  topology_separator_coverage = (
      1.0 if full_separator_edges == 0
      else float(topology_separator_edges) / float(full_separator_edges)
  )
  evidence_mass = compute_separator_residual_mass_coverage(
      full_baseline_cost=baseline_cost,
      selected_baseline_cost=dci_baseline_cost,
      edge_count_coverage=topology_separator_coverage)
  normal_summary_coverage = compute_separator_normal_summary_payload_coverage(
      full_graph_edges=graph_edges,
      selected_edges=dci_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      baseline_poses=baseline_poses,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode)
  normal_summary_structure = (
      compute_separator_normal_summary_block_graph_certificate(
          full_graph_edges=graph_edges,
          selected_edges=dci_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          baseline_poses=baseline_poses,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode))
  boundary_pose_unit_count = separator_boundary_pose_unit_count(dci_edges, robot_of)
  use_boundary_pose_linear_accounting = (
      relay_scheduler in {
          "pose_summary_budget",
          "pose_summary_criticality_budget",
          "pose_summary_weighted_criticality_budget",
          "pose_summary_leverage_budget",
          "pose_summary_guarded_leverage_budget",
          "pose_summary_normal_leverage_budget",
          "pose_summary_normal_density_budget",
          "pose_summary_bridge_normal_budget",
          "pose_summary_adaptive_bridge_density_budget",
          "pose_summary_jacobi_leverage_budget",
          "pose_summary_interface_leverage_budget",
          "pose_summary_schur_leverage_budget",
          "pose_summary_guarded_schur_leverage_budget",
          "pose_summary_lazy_guarded_schur_leverage_budget",
      } or
      candidate_solver == "normal_summary"
  )

  def normal_summary_accounting(logical_payload_bytes):
    logical_bytes = max(0, int(round(float(logical_payload_bytes))))
    routed = estimate_topology_routed_normal_summary_communication(
        logical_payload_bytes=logical_bytes,
        separator_paths=topology_evidence.get("separator_paths", []),
    )
    logical_mb = float(logical_bytes) / (1024.0 * 1024.0)
    routed_mb = float(routed.get("routed_payload_mb", logical_mb))
    effective_mb = (
        routed_mb
        if normal_summary_communication_model == "topology_routed_equal_edge"
        else logical_mb
    )
    return {
        "model": normal_summary_communication_model,
        "logical_bytes": logical_bytes,
        "logical_mb": logical_mb,
        "routed_mb": routed_mb,
        "effective_mb": effective_mb,
        "routing": routed,
    }

  def curvature_payload_accounting(logical_payload_bytes):
    logical_bytes = max(0, int(round(float(logical_payload_bytes))))
    routed = estimate_topology_routed_normal_summary_communication(
        logical_payload_bytes=logical_bytes,
        separator_paths=topology_evidence.get("separator_paths", []),
    )
    routed = {
        **routed,
        "model": "pairwise_curvature_payload_path_hop_routing",
    }
    logical_mb = float(logical_bytes) / (1024.0 * 1024.0)
    routed_mb = float(routed.get("routed_payload_mb", logical_mb))
    use_routed = (
        normal_summary_communication_model == "topology_routed_equal_edge" and
        omitted_force_correction_curvature_payload_model == "pairwise_projected"
    )
    return {
        "model": (
            "topology_routed_pairwise_projected_curvature"
            if use_routed else "logical_curvature_payload"),
        "logical_bytes": logical_bytes,
        "logical_mb": logical_mb,
        "routed_mb": routed_mb,
        "effective_mb": routed_mb if use_routed else logical_mb,
        "routing": routed,
    }

  def curvature_payload_comm_multiplier():
    if not (
        normal_summary_communication_model == "topology_routed_equal_edge" and
        omitted_force_correction_curvature_payload_model == "pairwise_projected"
    ):
      return 1.0
    separator_paths = topology_evidence.get("separator_paths", [])
    if not separator_paths:
      return 1.0
    total_path_hops = sum(
        max(1, int(item.get("path_hops", 1))) for item in separator_paths)
    return float(total_path_hops) / float(len(separator_paths))

  relay_extra_hop_count = int(topology_evidence["relay_extra_hop_count"])
  relay_extra_selected_mb = float(
      topology_evidence.get("relay_scheduler_predicted_selected_relay_mb", 0.0)
  )
  relay_extra_comm = {
      "relay_extra_hop_count": relay_extra_hop_count,
      "relay_extra_exchange_bytes":
          int(round(relay_extra_selected_mb * 1024.0 * 1024.0)),
      "relay_extra_exchange_mb": relay_extra_selected_mb,
      "model": "selected_topology_relay_payload",
  }
  candidate_precert = compute_candidate_precert(
      baseline_cost=dci_baseline_cost,
      mode=candidate_precert_mode,
      min_sep_private_ratio=candidate_precert_min_sep_private_ratio)

  def write_baseline_and_report(reason: str,
                                predicted_mb: float = 0.0,
                                value_score: float = 0.0,
                                predicted_comm: dict | None = None):
    selected_poses = {
        pose_id: baseline_poses[pose_id]
        for pose_id in pose_ids
        if pose_id in baseline_poses
    }
    write_manual_matrix_pose_set(Path(output_selected_estimate), selected_poses, dim)
    return {
        "pre_gate_decision": reason,
        "candidate_solved": False,
        "selected": "baseline_gauge_selector",
        "candidate_solver": candidate_solver,
        "solve_communication_model": solve_communication_model,
        "summary_selection_mode": summary_selection_mode,
        "summary_max_offdiag_block_edges": summary_max_offdiag_block_edges,
        "summary_refinement_max_offdiag_block_edges":
            summary_refinement_max_offdiag_block_edges,
        "omitted_force_correction_rounds": omitted_force_correction_rounds,
        "omitted_force_correction_requested_curvature_rank":
            int(omitted_force_correction_curvature_rank),
        "omitted_force_correction_selected_curvature_rank":
            int(omitted_force_correction_curvature_rank),
        "omitted_force_correction_rank_portfolio_requested":
            bool(omitted_force_correction_rank_portfolio),
        "omitted_force_correction_rank_portfolio_enabled": False,
        "omitted_force_correction_rank_portfolio": None,
        "omitted_force_correction_curvature_rank_scheduler":
            omitted_force_correction_curvature_rank_scheduler,
        "omitted_force_correction_curvature_rank_max_condition":
            omitted_force_correction_curvature_rank_max_condition,
        "omitted_force_correction_curvature_rank_condition_policy":
            omitted_force_correction_curvature_rank_condition_policy,
        "omitted_force_correction_curvature_rank_condition_mad_scale":
            float(omitted_force_correction_curvature_rank_condition_mad_scale),
        "omitted_force_correction_diagnose_subspace_miss":
            bool(omitted_force_correction_diagnose_subspace_miss),
        "relay_cost_rotation_iterations": relay_cost_rotation_iterations,
        "relay_cost_translation_iterations": relay_cost_translation_iterations,
        "selected_total_cost": baseline_cost["total_cost"],
        "baseline_total_cost": baseline_cost["total_cost"],
        "dci_baseline_total_cost": dci_baseline_cost["total_cost"],
        "distributed_total_cost": None,
        "full_graph_distributed_total_cost": None,
        "certificate_total_delta": None,
        "evidence_global_consensus_accept": False,
        "global_consensus_accept": False,
        "handoff_gate_mode": handoff_gate_mode,
        "handoff_gate_accept": False,
        "full_graph_handoff_delta": None,
        "handoff_gate_reject_reason": reason,
        "predicted_candidate_comm_mb": predicted_mb,
        "actual_dci_comm_mb": 0.0,
        "value_upper_bound_cost_per_mb": value_score,
        "value_score": value_score,
        "value_score_mode": value_score_mode,
        "value_threshold": value_threshold,
        "output_path": str(output_selected_estimate),
        "predicted_communication": predicted_comm or {},
        "topology_separator_edges": topology_separator_edges,
        "full_separator_edges": full_separator_edges,
        "boundary_pose_unit_count": boundary_pose_unit_count,
        "topology_separator_coverage": topology_separator_coverage,
        **evidence_mass,
        **normal_summary_coverage,
        **normal_summary_structure,
        "normal_summary_communication_model": normal_summary_communication_model,
        "normal_equation_summary_logical_comm_mb": 0.0,
        "normal_equation_summary_routed_comm_mb": 0.0,
        "topology_routed_normal_summary_communication":
            normal_summary_accounting(0)["routing"],
        "min_topology_separator_coverage": min_topology_separator_coverage,
        "topology_available_pairs": (
            sorted(_normalize_pair(src, dst) for src, dst in available_robot_pairs)
            if available_robot_pairs is not None else None
        ),
        "topology_round_window": topology_round_window,
        "max_relay_hops": max_relay_hops,
        "direct_separator_edges": topology_evidence["direct_separator_edges"],
        "relayed_separator_edges": topology_evidence["relayed_separator_edges"],
        "relay_extra_hop_count": relay_extra_hop_count,
        "relay_scheduler": topology_evidence["relay_scheduler"],
        "relay_scheduler_selected_relay_edges":
            topology_evidence["relay_scheduler_selected_relay_edges"],
        "relay_scheduler_skipped_relay_edges":
            topology_evidence["relay_scheduler_skipped_relay_edges"],
        "relay_scheduler_reachable_relay_edges":
            topology_evidence["relay_scheduler_reachable_relay_edges"],
        "relay_scheduler_byte_budget_mb":
            topology_evidence["relay_scheduler_byte_budget_mb"],
        "relay_scheduler_predicted_selected_relay_mb":
            topology_evidence["relay_scheduler_predicted_selected_relay_mb"],
        "predicted_relay_extra_comm_mb": (
            float(predicted_comm.get("relay_extra", {}).get(
                "relay_extra_exchange_mb", 0.0))
            if predicted_comm else 0.0
        ),
        "relay_extra_comm_mb": 0.0,
        "relay_evidence": topology_evidence,
        **candidate_precert,
    }

  if topology_separator_edges == 0:
    return write_baseline_and_report("skip_no_separator_topology")
  if topology_separator_coverage < min_topology_separator_coverage:
    return write_baseline_and_report("skip_insufficient_topology_coverage")
  predicted_normal_summary_comm_mb = 0.0
  predicted_normal_summary_accounting = normal_summary_accounting(0)
  if candidate_solver == "normal_summary":
    predicted_normal_summary_accounting = normal_summary_accounting(
        normal_summary_coverage.get(
            "covered_separator_normal_summary_payload_bytes", 0.0))
    predicted_normal_summary_comm_mb = predicted_normal_summary_accounting[
        "effective_mb"]
  predicted_comm = estimate_budgeted_candidate_communication(
      graph_edges=dci_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      rotation_iterations=rotation_iterations,
      translation_iterations=translation_iterations,
      linear_solver=linear_solver,
      relay_extra_hop_count=relay_extra_hop_count,
      relay_extra_comm_mb=relay_extra_selected_mb,
      boundary_pose_unit_count=(
          boundary_pose_unit_count if use_boundary_pose_linear_accounting else None
      ),
      solve_communication_model=solve_communication_model,
      normal_summary_comm_mb=predicted_normal_summary_comm_mb,
  )
  predicted_mb = float(predicted_comm["total_estimated_mb"])
  value_score = compute_value_score(
      baseline_cost=dci_baseline_cost,
      predicted_comm_mb=predicted_mb,
      mode=value_score_mode)
  should_run_candidate = value_score >= value_threshold

  output_selected_estimate = Path(output_selected_estimate)
  if not should_run_candidate:
    selected_poses = {
        pose_id: baseline_poses[pose_id]
        for pose_id in pose_ids
        if pose_id in baseline_poses
    }
    write_manual_matrix_pose_set(output_selected_estimate, selected_poses, dim)
    return {
        "pre_gate_decision": "skip",
        "candidate_solved": False,
        "selected": "baseline_gauge_selector",
        "candidate_solver": candidate_solver,
        "solve_communication_model": solve_communication_model,
        "summary_selection_mode": summary_selection_mode,
        "summary_max_offdiag_block_edges": summary_max_offdiag_block_edges,
        "summary_refinement_max_offdiag_block_edges":
            summary_refinement_max_offdiag_block_edges,
        "omitted_force_correction_rounds": omitted_force_correction_rounds,
        "omitted_force_correction_requested_curvature_rank":
            int(omitted_force_correction_curvature_rank),
        "omitted_force_correction_selected_curvature_rank":
            int(omitted_force_correction_curvature_rank),
        "omitted_force_correction_rank_portfolio_requested":
            bool(omitted_force_correction_rank_portfolio),
        "omitted_force_correction_rank_portfolio_enabled": False,
        "omitted_force_correction_rank_portfolio": None,
        "omitted_force_correction_curvature_rank_scheduler":
            omitted_force_correction_curvature_rank_scheduler,
        "omitted_force_correction_curvature_rank_max_condition":
            omitted_force_correction_curvature_rank_max_condition,
        "omitted_force_correction_curvature_rank_condition_policy":
            omitted_force_correction_curvature_rank_condition_policy,
        "omitted_force_correction_curvature_rank_condition_mad_scale":
            float(omitted_force_correction_curvature_rank_condition_mad_scale),
        "omitted_force_correction_diagnose_subspace_miss":
            bool(omitted_force_correction_diagnose_subspace_miss),
        "relay_cost_rotation_iterations": relay_cost_rotation_iterations,
        "relay_cost_translation_iterations": relay_cost_translation_iterations,
        "selected_total_cost": baseline_cost["total_cost"],
        "baseline_total_cost": baseline_cost["total_cost"],
        "dci_baseline_total_cost": dci_baseline_cost["total_cost"],
        "distributed_total_cost": None,
        "full_graph_distributed_total_cost": None,
        "certificate_total_delta": None,
        "evidence_global_consensus_accept": False,
        "global_consensus_accept": False,
        "handoff_gate_mode": handoff_gate_mode,
        "handoff_gate_accept": False,
        "full_graph_handoff_delta": None,
        "handoff_gate_reject_reason": "pre_gate_skip",
        "predicted_candidate_comm_mb": predicted_mb,
        "actual_dci_comm_mb": 0.0,
        "value_upper_bound_cost_per_mb": value_score,
        "value_score": value_score,
        "value_score_mode": value_score_mode,
        "value_threshold": value_threshold,
        "output_path": str(output_selected_estimate),
        "predicted_communication": predicted_comm,
        "topology_separator_edges": topology_separator_edges,
        "full_separator_edges": full_separator_edges,
        "boundary_pose_unit_count": boundary_pose_unit_count,
        "topology_separator_coverage": topology_separator_coverage,
        **evidence_mass,
        **normal_summary_coverage,
        **normal_summary_structure,
        "normal_summary_communication_model": normal_summary_communication_model,
        "normal_equation_summary_logical_comm_mb": 0.0,
        "normal_equation_summary_routed_comm_mb": 0.0,
        "topology_routed_normal_summary_communication":
            normal_summary_accounting(0)["routing"],
        "min_topology_separator_coverage": min_topology_separator_coverage,
        "topology_available_pairs": (
            sorted(_normalize_pair(src, dst) for src, dst in available_robot_pairs)
            if available_robot_pairs is not None else None
        ),
        "topology_round_window": topology_round_window,
        "max_relay_hops": max_relay_hops,
        "direct_separator_edges": topology_evidence["direct_separator_edges"],
        "relayed_separator_edges": topology_evidence["relayed_separator_edges"],
        "relay_extra_hop_count": relay_extra_hop_count,
        "relay_scheduler": topology_evidence["relay_scheduler"],
        "relay_scheduler_selected_relay_edges":
            topology_evidence["relay_scheduler_selected_relay_edges"],
        "relay_scheduler_skipped_relay_edges":
            topology_evidence["relay_scheduler_skipped_relay_edges"],
        "relay_scheduler_reachable_relay_edges":
            topology_evidence["relay_scheduler_reachable_relay_edges"],
        "relay_scheduler_byte_budget_mb":
            topology_evidence["relay_scheduler_byte_budget_mb"],
        "relay_scheduler_predicted_selected_relay_mb":
            topology_evidence["relay_scheduler_predicted_selected_relay_mb"],
        "predicted_relay_extra_comm_mb": (
            float(predicted_comm.get("relay_extra", {}).get(
                "relay_extra_exchange_mb", 0.0))
        ),
        "relay_extra_comm_mb": 0.0,
        "relay_evidence": topology_evidence,
        **candidate_precert,
    }
  if not candidate_precert["candidate_precert_accept"]:
    selected_poses = {
        pose_id: baseline_poses[pose_id]
        for pose_id in pose_ids
        if pose_id in baseline_poses
    }
    write_manual_matrix_pose_set(output_selected_estimate, selected_poses, dim)
    return {
        "pre_gate_decision": "skip_precert",
        "candidate_solved": False,
        "selected": "baseline_gauge_selector",
        "candidate_solver": candidate_solver,
        "solve_communication_model": solve_communication_model,
        "summary_selection_mode": summary_selection_mode,
        "summary_max_offdiag_block_edges": summary_max_offdiag_block_edges,
        "summary_refinement_max_offdiag_block_edges":
            summary_refinement_max_offdiag_block_edges,
        "omitted_force_correction_rounds": omitted_force_correction_rounds,
        "omitted_force_correction_requested_curvature_rank":
            int(omitted_force_correction_curvature_rank),
        "omitted_force_correction_selected_curvature_rank":
            int(omitted_force_correction_curvature_rank),
        "omitted_force_correction_rank_portfolio_requested":
            bool(omitted_force_correction_rank_portfolio),
        "omitted_force_correction_rank_portfolio_enabled": False,
        "omitted_force_correction_rank_portfolio": None,
        "omitted_force_correction_curvature_rank_scheduler":
            omitted_force_correction_curvature_rank_scheduler,
        "omitted_force_correction_curvature_rank_max_condition":
            omitted_force_correction_curvature_rank_max_condition,
        "omitted_force_correction_curvature_rank_condition_policy":
            omitted_force_correction_curvature_rank_condition_policy,
        "omitted_force_correction_curvature_rank_condition_mad_scale":
            float(omitted_force_correction_curvature_rank_condition_mad_scale),
        "omitted_force_correction_diagnose_subspace_miss":
            bool(omitted_force_correction_diagnose_subspace_miss),
        "relay_cost_rotation_iterations": relay_cost_rotation_iterations,
        "relay_cost_translation_iterations": relay_cost_translation_iterations,
        "selected_total_cost": baseline_cost["total_cost"],
        "baseline_total_cost": baseline_cost["total_cost"],
        "dci_baseline_total_cost": dci_baseline_cost["total_cost"],
        "distributed_total_cost": None,
        "full_graph_distributed_total_cost": None,
        "certificate_total_delta": None,
        "global_consensus_accept": False,
        "predicted_candidate_comm_mb": predicted_mb,
        "actual_dci_comm_mb": 0.0,
        "value_upper_bound_cost_per_mb": value_score,
        "value_score": value_score,
        "value_score_mode": value_score_mode,
        "value_threshold": value_threshold,
        "output_path": str(output_selected_estimate),
        "predicted_communication": predicted_comm,
        "topology_separator_edges": topology_separator_edges,
        "full_separator_edges": full_separator_edges,
        "boundary_pose_unit_count": boundary_pose_unit_count,
        "topology_separator_coverage": topology_separator_coverage,
        **evidence_mass,
        **normal_summary_coverage,
        **normal_summary_structure,
        "normal_summary_communication_model": normal_summary_communication_model,
        "normal_equation_summary_logical_comm_mb": 0.0,
        "normal_equation_summary_routed_comm_mb": 0.0,
        "topology_routed_normal_summary_communication":
            normal_summary_accounting(0)["routing"],
        "min_topology_separator_coverage": min_topology_separator_coverage,
        "topology_available_pairs": (
            sorted(_normalize_pair(src, dst) for src, dst in available_robot_pairs)
            if available_robot_pairs is not None else None
        ),
        "topology_round_window": topology_round_window,
        "max_relay_hops": max_relay_hops,
        "direct_separator_edges": topology_evidence["direct_separator_edges"],
        "relayed_separator_edges": topology_evidence["relayed_separator_edges"],
        "relay_extra_hop_count": relay_extra_hop_count,
        "relay_scheduler": topology_evidence["relay_scheduler"],
        "relay_scheduler_selected_relay_edges":
            topology_evidence["relay_scheduler_selected_relay_edges"],
        "relay_scheduler_skipped_relay_edges":
            topology_evidence["relay_scheduler_skipped_relay_edges"],
        "relay_scheduler_reachable_relay_edges":
            topology_evidence["relay_scheduler_reachable_relay_edges"],
        "relay_scheduler_byte_budget_mb":
            topology_evidence["relay_scheduler_byte_budget_mb"],
        "relay_scheduler_predicted_selected_relay_mb":
            topology_evidence["relay_scheduler_predicted_selected_relay_mb"],
        "predicted_relay_extra_comm_mb": (
            float(predicted_comm.get("relay_extra", {}).get(
                "relay_extra_exchange_mb", 0.0))
        ),
        "relay_extra_comm_mb": 0.0,
        "relay_evidence": topology_evidence,
        **candidate_precert,
    }

  distributed_poses, distributed_stats = solve_distributed_chordal_initialization(
      graph_edges=dci_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      rotation_iterations=rotation_iterations,
      translation_iterations=translation_iterations,
      weighted=weighted,
      cost_mode=cost_mode,
      relaxation=relaxation,
      damping=damping,
      linear_solver=linear_solver,
  ) if candidate_solver == "edge_subgraph" else solve_summary_chordal_initialization(
      graph_edges=dci_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      rotation_iterations=rotation_iterations,
      translation_iterations=translation_iterations,
      weighted=weighted,
      cost_mode=cost_mode,
      relaxation=relaxation,
      damping=damping,
      linear_solver=linear_solver,
      interface_schur_preconditioner=interface_schur_preconditioner,
      interface_schur_rotation_preconditioner=(
          interface_schur_rotation_preconditioner),
      interface_schur_translation_preconditioner=(
          interface_schur_translation_preconditioner),
      interface_schur_coarse_initial_guess=(
          interface_schur_coarse_initial_guess),
      interface_schur_coarse_basis_mode=interface_schur_coarse_basis_mode,
      interface_schur_coarse_component_limit=(
          interface_schur_coarse_component_limit),
      interface_schur_rotation_coarse_basis_mode=(
          interface_schur_rotation_coarse_basis_mode),
      interface_schur_translation_coarse_basis_mode=(
          interface_schur_translation_coarse_basis_mode),
      interface_schur_rotation_coarse_component_limit=(
          interface_schur_rotation_coarse_component_limit),
      interface_schur_translation_coarse_component_limit=(
          interface_schur_translation_coarse_component_limit),
      interface_schur_coarse_component_selection_mode=(
          interface_schur_coarse_component_selection_mode),
      interface_schur_rotation_coarse_component_selection_mode=(
          interface_schur_rotation_coarse_component_selection_mode),
      interface_schur_translation_coarse_component_selection_mode=(
          interface_schur_translation_coarse_component_selection_mode),
      interface_schur_residual_deflation_rank=(
          interface_schur_residual_deflation_rank),
      interface_schur_residual_deflation_pilot_iterations=(
          interface_schur_residual_deflation_pilot_iterations),
      interface_schur_ritz_rank=interface_schur_ritz_rank,
      interface_schur_ritz_probe_iterations=(
          interface_schur_ritz_probe_iterations),
      interface_schur_rotation_ritz_rank=(
          interface_schur_rotation_ritz_rank),
      interface_schur_translation_ritz_rank=(
          interface_schur_translation_ritz_rank),
      interface_schur_rotation_ritz_probe_iterations=(
          interface_schur_rotation_ritz_probe_iterations),
      interface_schur_translation_ritz_probe_iterations=(
          interface_schur_translation_ritz_probe_iterations),
      interface_schur_rotation_ritz_rank_selection_mode=(
          interface_schur_rotation_ritz_rank_selection_mode),
      interface_schur_translation_ritz_rank_selection_mode=(
          interface_schur_translation_ritz_rank_selection_mode),
      interface_schur_rotation_ritz_value_threshold=(
          interface_schur_rotation_ritz_value_threshold),
      interface_schur_translation_ritz_value_threshold=(
          interface_schur_translation_ritz_value_threshold),
      interface_schur_rotation_ritz_energy_capture_fraction=(
          interface_schur_rotation_ritz_energy_capture_fraction),
      interface_schur_translation_ritz_energy_capture_fraction=(
          interface_schur_translation_ritz_energy_capture_fraction),
      interface_schur_translation_budget_candidates=(
          interface_schur_translation_budget_candidates),
      interface_schur_ritz_mode=interface_schur_ritz_mode,
      summary_selection_mode=summary_selection_mode,
      summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
      summary_refinement_max_offdiag_block_edges=(
          summary_refinement_max_offdiag_block_edges),
      omitted_force_correction_rounds=omitted_force_correction_rounds,
      omitted_force_correction_portfolio=omitted_force_correction_portfolio,
      omitted_force_correction_rank_portfolio=(
          omitted_force_correction_rank_portfolio),
      omitted_force_correction_curvature_model=(
          omitted_force_correction_curvature_model),
      omitted_force_correction_curvature_rank=(
          omitted_force_correction_curvature_rank),
      omitted_force_correction_curvature_rank_scheduler=(
          omitted_force_correction_curvature_rank_scheduler),
      omitted_force_correction_curvature_rank_max_condition=(
          omitted_force_correction_curvature_rank_max_condition),
      omitted_force_correction_curvature_rank_condition_policy=(
          omitted_force_correction_curvature_rank_condition_policy),
      omitted_force_correction_curvature_rank_condition_mad_scale=(
          omitted_force_correction_curvature_rank_condition_mad_scale),
      omitted_force_correction_curvature_payload_model=(
          omitted_force_correction_curvature_payload_model),
      omitted_force_correction_diagnose_subspace_miss=(
          omitted_force_correction_diagnose_subspace_miss),
      omitted_force_correction_max_comm_mb=(
          omitted_force_correction_max_comm_mb),
      omitted_force_correction_curvature_comm_multiplier=(
          curvature_payload_comm_multiplier()),
      omitted_force_correction_min_marginal_cost_per_mb=(
          omitted_force_correction_min_marginal_cost_per_mb),
      central_equivalence_diagnostic=central_equivalence_diagnostic,
      central_equivalence_iterative_diagnostic_iterations=(
          central_equivalence_iterative_diagnostic_iterations),
  )
  distributed_cost = cost_breakdown(
      dci_edges, distributed_poses, robot_of, dci_active_ids, weighted, cost_mode)
  full_graph_distributed_cost = cost_breakdown(
      graph_edges, distributed_poses, robot_of, active_ids, weighted, cost_mode)
  certificate = compute_candidate_residual_certificate(
      graph_edges=dci_edges,
      baseline=baseline_poses,
      candidate=distributed_poses,
      robot_of=robot_of,
      weighted=weighted,
      cost_mode=cost_mode,
  )
  evidence_accept = bool(certificate["global_consensus_accept"])
  full_graph_handoff_delta = (
      float(full_graph_distributed_cost["total_cost"]) -
      float(baseline_cost["total_cost"]))
  if handoff_gate_mode == "evidence_cost":
    selected_is_candidate = evidence_accept
    handoff_gate_reject_reason = (
        "" if selected_is_candidate else "evidence_cost_not_decreased")
  else:
    selected_is_candidate = (
        evidence_accept and full_graph_handoff_delta < -1e-12)
    if not evidence_accept:
      handoff_gate_reject_reason = "evidence_cost_not_decreased"
    elif full_graph_handoff_delta >= -1e-12:
      handoff_gate_reject_reason = "full_graph_cost_not_decreased"
    else:
      handoff_gate_reject_reason = ""
  selected_poses = distributed_poses if selected_is_candidate else baseline_poses
  ordered_selected = {
      pose_id: selected_poses[pose_id]
      for pose_id in pose_ids
      if pose_id in selected_poses
  }
  write_manual_matrix_pose_set(output_selected_estimate, ordered_selected, dim)
  linear_comm = distributed_stats.get("communication_estimate", {})
  translation_budget_portfolio_comm = distributed_stats.get(
      "translation_budget_portfolio_communication_estimate")
  partial_measurement_portfolio_comm = distributed_stats.get(
      "partial_measurement_portfolio_communication_estimate")
  coarse_measurement_portfolio_comm = distributed_stats.get(
      "coarse_measurement_portfolio_communication_estimate")
  measurement_portfolio_comm = distributed_stats.get(
      "measurement_portfolio_communication_estimate")
  if translation_budget_portfolio_comm:
    linear_comm = translation_budget_portfolio_comm
  elif partial_measurement_portfolio_comm:
    linear_comm = partial_measurement_portfolio_comm
  elif coarse_measurement_portfolio_comm:
    linear_comm = coarse_measurement_portfolio_comm
  elif measurement_portfolio_comm:
    linear_comm = measurement_portfolio_comm
  elif use_boundary_pose_linear_accounting and linear_solver == "interface_schur_pcg":
    linear_comm = estimate_dci_interface_schur_solve_communication(
        distributed_stats.get("rotation_stats", {}),
        distributed_stats.get("translation_stats", {}),
        len({robot_of[pose_id] for pose_id in pose_ids if pose_id in robot_of}),
    )
  elif use_boundary_pose_linear_accounting and linear_solver != "compact_pcg":
    linear_comm = estimate_dci_boundary_pose_solve_communication(
        distributed_stats.get("rotation_stats", {}),
        distributed_stats.get("translation_stats", {}),
        len({robot_of[pose_id] for pose_id in pose_ids if pose_id in robot_of}),
        boundary_pose_unit_count,
        dim,
    )
  cert_comm = certificate.get("communication_estimate", {})
  normal_summary_comm = distributed_stats.get(
      "normal_equation_summary_communication", {})
  actual_normal_summary_accounting = normal_summary_accounting(
      normal_summary_comm.get("total_summary_bytes", 0)
      if use_boundary_pose_linear_accounting else 0)
  normal_summary_mb = actual_normal_summary_accounting["effective_mb"]
  correction_comm = distributed_stats.get("omitted_force_correction", {}).get(
      "communication_estimate", {})
  omitted_force_comm_mb = float(
      correction_comm.get("total_omitted_force_payload_mb", 0.0))
  actual_curvature_accounting = curvature_payload_accounting(
      correction_comm.get("total_curvature_payload_bytes", 0))
  curvature_comm_mb = float(actual_curvature_accounting["effective_mb"])
  correction_comm_mb = omitted_force_comm_mb + curvature_comm_mb
  rank_portfolio = distributed_stats.get("omitted_force_correction_rank_portfolio")
  rank_portfolio_enabled = bool(
      rank_portfolio and rank_portfolio.get("enabled", False))
  selected_curvature_rank = int(distributed_stats.get(
      "omitted_force_correction_selected_curvature_rank",
      omitted_force_correction_curvature_rank))
  effective_linear_solve_comm_mb = float(
      linear_comm.get("linear_solve_total_estimated_mb", 0.0))
  if solve_communication_model == "compact_continuation_diagnostic":
    effective_linear_solve_comm_mb = float(
        linear_comm.get("pcg_global_reduction_mb", 0.0))
  actual_mb = (
      effective_linear_solve_comm_mb +
      float(cert_comm.get("residual_delta_consensus_mb", 0.0)) +
      float(relay_extra_comm.get("relay_extra_exchange_mb", 0.0)) +
      normal_summary_mb +
      correction_comm_mb
  )
  return {
      "pre_gate_decision": "run",
      "candidate_solved": True,
      "selected": (
          "distributed_chordal_pcg" if selected_is_candidate
          else "baseline_gauge_selector"
      ),
      "candidate_solver": candidate_solver,
      "solve_communication_model": solve_communication_model,
      "summary_selection_mode": summary_selection_mode,
      "summary_max_offdiag_block_edges": summary_max_offdiag_block_edges,
      "summary_refinement_max_offdiag_block_edges":
          summary_refinement_max_offdiag_block_edges,
      "omitted_force_correction_rounds": omitted_force_correction_rounds,
      "omitted_force_correction_curvature_model":
          omitted_force_correction_curvature_model,
      "omitted_force_correction_curvature_rank":
          int(omitted_force_correction_curvature_rank),
      "omitted_force_correction_requested_curvature_rank":
          int(omitted_force_correction_curvature_rank),
      "omitted_force_correction_selected_curvature_rank":
          selected_curvature_rank,
      "omitted_force_correction_curvature_rank_scheduler":
          omitted_force_correction_curvature_rank_scheduler,
      "omitted_force_correction_curvature_rank_max_condition":
          omitted_force_correction_curvature_rank_max_condition,
      "omitted_force_correction_curvature_rank_condition_policy":
          omitted_force_correction_curvature_rank_condition_policy,
      "omitted_force_correction_curvature_rank_condition_mad_scale":
          float(omitted_force_correction_curvature_rank_condition_mad_scale),
      "omitted_force_correction_diagnose_subspace_miss":
          bool(omitted_force_correction_diagnose_subspace_miss),
      "omitted_force_correction_rank_portfolio_requested":
          bool(omitted_force_correction_rank_portfolio),
      "omitted_force_correction_rank_portfolio_enabled":
          rank_portfolio_enabled,
      "omitted_force_correction_curvature_payload_model":
          omitted_force_correction_curvature_payload_model,
      "omitted_force_correction_max_comm_mb":
          omitted_force_correction_max_comm_mb,
      "omitted_force_correction_curvature_comm_multiplier":
          curvature_payload_comm_multiplier(),
      "omitted_force_correction_min_marginal_cost_per_mb":
          omitted_force_correction_min_marginal_cost_per_mb,
      "relay_cost_rotation_iterations": relay_cost_rotation_iterations,
      "relay_cost_translation_iterations": relay_cost_translation_iterations,
      "selected_total_cost": (
          full_graph_distributed_cost["total_cost"] if selected_is_candidate
          else baseline_cost["total_cost"]
      ),
      "baseline_total_cost": baseline_cost["total_cost"],
      "dci_baseline_total_cost": dci_baseline_cost["total_cost"],
      "distributed_total_cost": distributed_cost["total_cost"],
      "full_graph_distributed_total_cost":
          full_graph_distributed_cost["total_cost"],
      "certificate_total_delta": certificate["total_delta"],
      "evidence_global_consensus_accept": evidence_accept,
      "global_consensus_accept": selected_is_candidate,
      "handoff_gate_mode": handoff_gate_mode,
      "handoff_gate_accept": bool(selected_is_candidate),
      "full_graph_handoff_delta": full_graph_handoff_delta,
      "handoff_gate_reject_reason": handoff_gate_reject_reason,
      "predicted_candidate_comm_mb": predicted_mb,
      "actual_dci_comm_mb": actual_mb,
      "value_upper_bound_cost_per_mb": value_score,
      "value_score": value_score,
      "value_score_mode": value_score_mode,
      "value_threshold": value_threshold,
      "linear_solver": linear_solver,
      "interface_schur_preconditioner": interface_schur_preconditioner,
      "interface_schur_rotation_preconditioner": distributed_stats.get(
          "interface_schur_rotation_preconditioner"),
      "interface_schur_translation_preconditioner": distributed_stats.get(
          "interface_schur_translation_preconditioner"),
      "interface_schur_coarse_initial_guess": bool(
          interface_schur_coarse_initial_guess),
      "interface_schur_coarse_basis_mode": interface_schur_coarse_basis_mode,
      "interface_schur_coarse_component_limit": (
          None if interface_schur_coarse_component_limit is None
          else int(interface_schur_coarse_component_limit)),
      "interface_schur_rotation_coarse_basis_mode": distributed_stats.get(
          "interface_schur_rotation_coarse_basis_mode"),
      "interface_schur_translation_coarse_basis_mode": distributed_stats.get(
          "interface_schur_translation_coarse_basis_mode"),
      "interface_schur_rotation_coarse_component_limit": distributed_stats.get(
          "interface_schur_rotation_coarse_component_limit"),
      "interface_schur_translation_coarse_component_limit": distributed_stats.get(
          "interface_schur_translation_coarse_component_limit"),
      "interface_schur_coarse_component_selection_mode": distributed_stats.get(
          "interface_schur_coarse_component_selection_mode"),
      "interface_schur_rotation_coarse_component_selection_mode":
          distributed_stats.get(
              "interface_schur_rotation_coarse_component_selection_mode"),
      "interface_schur_translation_coarse_component_selection_mode":
          distributed_stats.get(
              "interface_schur_translation_coarse_component_selection_mode"),
      "interface_schur_residual_deflation_rank": int(
          interface_schur_residual_deflation_rank),
      "interface_schur_residual_deflation_pilot_iterations": int(
          interface_schur_residual_deflation_pilot_iterations),
      "interface_schur_ritz_rank": int(interface_schur_ritz_rank),
      "interface_schur_ritz_probe_iterations": int(
          interface_schur_ritz_probe_iterations),
      "interface_schur_rotation_ritz_rank": distributed_stats.get(
          "interface_schur_rotation_ritz_rank"),
      "interface_schur_translation_ritz_rank": distributed_stats.get(
          "interface_schur_translation_ritz_rank"),
      "interface_schur_rotation_ritz_probe_iterations": distributed_stats.get(
          "interface_schur_rotation_ritz_probe_iterations"),
      "interface_schur_translation_ritz_probe_iterations": distributed_stats.get(
          "interface_schur_translation_ritz_probe_iterations"),
      "interface_schur_rotation_ritz_rank_selection_mode":
          distributed_stats.get(
              "interface_schur_rotation_ritz_rank_selection_mode"),
      "interface_schur_translation_ritz_rank_selection_mode":
          distributed_stats.get(
              "interface_schur_translation_ritz_rank_selection_mode"),
      "interface_schur_rotation_ritz_value_threshold": distributed_stats.get(
          "interface_schur_rotation_ritz_value_threshold"),
      "interface_schur_translation_ritz_value_threshold": distributed_stats.get(
          "interface_schur_translation_ritz_value_threshold"),
      "interface_schur_rotation_ritz_energy_capture_fraction":
          distributed_stats.get(
              "interface_schur_rotation_ritz_energy_capture_fraction"),
      "interface_schur_translation_ritz_energy_capture_fraction":
          distributed_stats.get(
              "interface_schur_translation_ritz_energy_capture_fraction"),
      "interface_schur_translation_budget_candidates": (
          interface_schur_translation_budget_candidates),
      "interface_schur_ritz_mode": interface_schur_ritz_mode,
      "central_equivalence_diagnostic": bool(central_equivalence_diagnostic),
      "central_equivalence_iterative_diagnostic_iterations": int(
          central_equivalence_iterative_diagnostic_iterations),
      "output_path": str(output_selected_estimate),
      "predicted_communication": predicted_comm,
      "actual_linear_communication": linear_comm,
      "effective_linear_solve_comm_mb": effective_linear_solve_comm_mb,
      "certificate_communication": cert_comm,
      "relay_extra_communication": relay_extra_comm,
      "normal_equation_summary_communication": normal_summary_comm,
      "normal_equation_summary_comm_mb": normal_summary_mb,
      "omitted_force_correction_force_comm_mb": omitted_force_comm_mb,
      "omitted_force_correction_curvature_comm_mb": curvature_comm_mb,
      "omitted_force_correction_logical_comm_mb":
          float(correction_comm.get("total_correction_payload_mb", 0.0)),
      "omitted_force_correction_effective_comm_mb": correction_comm_mb,
      "topology_routed_omitted_force_curvature_communication":
          actual_curvature_accounting["routing"],
      "summary_model_representativeness":
          distributed_stats.get("summary_model_representativeness"),
      "private_interface_residual_split":
          distributed_stats.get("private_interface_residual_split"),
      "omitted_force_correction":
          distributed_stats.get("omitted_force_correction"),
      "omitted_force_correction_portfolio":
          distributed_stats.get("omitted_force_correction_portfolio"),
      "omitted_force_correction_rank_portfolio":
          rank_portfolio,
      "translation_budget_portfolio":
          distributed_stats.get("translation_budget_portfolio"),
      "distributed_stats": distributed_stats,
      "topology_separator_edges": topology_separator_edges,
      "full_separator_edges": full_separator_edges,
      "boundary_pose_unit_count": boundary_pose_unit_count,
      "topology_separator_coverage": topology_separator_coverage,
      **evidence_mass,
      **normal_summary_coverage,
      **normal_summary_structure,
      "normal_summary_communication_model": normal_summary_communication_model,
      "normal_equation_summary_logical_comm_mb":
          actual_normal_summary_accounting["logical_mb"],
      "normal_equation_summary_routed_comm_mb":
          actual_normal_summary_accounting["routed_mb"],
      "topology_routed_normal_summary_communication":
          actual_normal_summary_accounting["routing"],
      "min_topology_separator_coverage": min_topology_separator_coverage,
      "topology_available_pairs": (
          sorted(_normalize_pair(src, dst) for src, dst in available_robot_pairs)
          if available_robot_pairs is not None else None
      ),
      "topology_round_window": topology_round_window,
      "max_relay_hops": max_relay_hops,
      "direct_separator_edges": topology_evidence["direct_separator_edges"],
      "relayed_separator_edges": topology_evidence["relayed_separator_edges"],
      "relay_extra_hop_count": relay_extra_hop_count,
      "relay_scheduler": topology_evidence["relay_scheduler"],
      "relay_scheduler_selected_relay_edges":
          topology_evidence["relay_scheduler_selected_relay_edges"],
      "relay_scheduler_skipped_relay_edges":
          topology_evidence["relay_scheduler_skipped_relay_edges"],
      "relay_scheduler_reachable_relay_edges":
          topology_evidence["relay_scheduler_reachable_relay_edges"],
      "relay_scheduler_byte_budget_mb":
          topology_evidence["relay_scheduler_byte_budget_mb"],
      "relay_scheduler_predicted_selected_relay_mb":
          topology_evidence["relay_scheduler_predicted_selected_relay_mb"],
      "predicted_relay_extra_comm_mb": (
          float(predicted_comm.get("relay_extra", {}).get(
              "relay_extra_exchange_mb", 0.0))
      ),
      "relay_extra_comm_mb": relay_extra_comm["relay_extra_exchange_mb"],
      "relay_evidence": topology_evidence,
      **candidate_precert,
  }


def build_report(args):
  graph_vertices, graph_edges = parse_g2o_graph(Path(args.graph))
  pose_ids = graph_pose_ids(graph_vertices, graph_edges)
  robot_of, ranges = build_contiguous_robot_map(pose_ids, args.num_robots)
  baseline_poses, baseline_format = load_oracle_pose_set(
      Path(args.baseline_estimate), args.baseline_format)
  available_pairs = (
      load_topology_pairs(
          Path(args.topology_file),
          args.topology_round,
          args.topology_round_window)
      if args.topology_file else None
  )
  report = run_budgeted_dci_for_graph_data(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      baseline_poses=baseline_poses,
      output_selected_estimate=Path(args.output_selected_estimate),
      rotation_iterations=args.rotation_iterations,
      translation_iterations=args.translation_iterations,
      value_threshold=args.value_threshold,
      weighted=args.weighted,
      cost_mode=args.cost_mode,
      linear_solver=args.linear_solver,
      interface_schur_preconditioner=args.interface_schur_preconditioner,
      interface_schur_rotation_preconditioner=(
          args.interface_schur_rotation_preconditioner),
      interface_schur_translation_preconditioner=(
          args.interface_schur_translation_preconditioner),
      interface_schur_coarse_initial_guess=(
          args.interface_schur_coarse_initial_guess),
      interface_schur_coarse_basis_mode=args.interface_schur_coarse_basis_mode,
      interface_schur_coarse_component_limit=(
          args.interface_schur_coarse_component_limit),
      interface_schur_rotation_coarse_basis_mode=(
          args.interface_schur_rotation_coarse_basis_mode),
      interface_schur_translation_coarse_basis_mode=(
          args.interface_schur_translation_coarse_basis_mode),
      interface_schur_rotation_coarse_component_limit=(
          args.interface_schur_rotation_coarse_component_limit),
      interface_schur_translation_coarse_component_limit=(
          args.interface_schur_translation_coarse_component_limit),
      interface_schur_coarse_component_selection_mode=(
          args.interface_schur_coarse_component_selection_mode),
      interface_schur_rotation_coarse_component_selection_mode=(
          args.interface_schur_rotation_coarse_component_selection_mode),
      interface_schur_translation_coarse_component_selection_mode=(
          args.interface_schur_translation_coarse_component_selection_mode),
      interface_schur_residual_deflation_rank=(
          args.interface_schur_residual_deflation_rank),
      interface_schur_residual_deflation_pilot_iterations=(
          args.interface_schur_residual_deflation_pilot_iterations),
      interface_schur_ritz_rank=args.interface_schur_ritz_rank,
      interface_schur_ritz_probe_iterations=(
          args.interface_schur_ritz_probe_iterations),
      interface_schur_rotation_ritz_rank=(
          args.interface_schur_rotation_ritz_rank),
      interface_schur_translation_ritz_rank=(
          args.interface_schur_translation_ritz_rank),
      interface_schur_rotation_ritz_probe_iterations=(
          args.interface_schur_rotation_ritz_probe_iterations),
      interface_schur_translation_ritz_probe_iterations=(
          args.interface_schur_translation_ritz_probe_iterations),
      interface_schur_rotation_ritz_rank_selection_mode=(
          args.interface_schur_rotation_ritz_rank_selection_mode),
      interface_schur_translation_ritz_rank_selection_mode=(
          args.interface_schur_translation_ritz_rank_selection_mode),
      interface_schur_rotation_ritz_value_threshold=(
          args.interface_schur_rotation_ritz_value_threshold),
      interface_schur_translation_ritz_value_threshold=(
          args.interface_schur_translation_ritz_value_threshold),
      interface_schur_rotation_ritz_energy_capture_fraction=(
          args.interface_schur_rotation_ritz_energy_capture_fraction),
      interface_schur_translation_ritz_energy_capture_fraction=(
          args.interface_schur_translation_ritz_energy_capture_fraction),
      interface_schur_translation_budget_candidates=(
          args.interface_schur_translation_budget_candidates),
      interface_schur_ritz_mode=args.interface_schur_ritz_mode,
      value_score_mode=args.value_score_mode,
      active_hops=args.active_hops,
      relaxation=args.relaxation,
      damping=args.damping,
      available_robot_pairs=available_pairs,
      min_topology_separator_coverage=args.min_topology_separator_coverage,
      topology_round_window=args.topology_round_window,
      max_relay_hops=args.topology_max_relay_hops,
      relay_scheduler=args.relay_scheduler,
      relay_byte_budget_mb=args.relay_byte_budget_mb,
      relay_cost_rotation_iterations=args.relay_cost_rotation_iterations,
      relay_cost_translation_iterations=args.relay_cost_translation_iterations,
      candidate_precert_mode=args.candidate_precert_mode,
      candidate_precert_min_sep_private_ratio=
          args.candidate_precert_min_sep_private_ratio,
      candidate_solver=args.candidate_solver,
      handoff_gate_mode=args.handoff_gate_mode,
      solve_communication_model=args.solve_communication_model,
      normal_summary_communication_model=
          args.normal_summary_communication_model,
      summary_selection_mode=args.summary_selection_mode,
      summary_max_offdiag_block_edges=args.summary_max_offdiag_block_edges,
      summary_refinement_max_offdiag_block_edges=
          args.summary_refinement_max_offdiag_block_edges,
      omitted_force_correction_rounds=args.omitted_force_correction_rounds,
      omitted_force_correction_portfolio=(
          args.omitted_force_correction_portfolio),
      omitted_force_correction_rank_portfolio=(
          args.omitted_force_correction_rank_portfolio),
      omitted_force_correction_curvature_model=(
          args.omitted_force_correction_curvature_model),
      omitted_force_correction_curvature_rank=(
          args.omitted_force_correction_curvature_rank),
      omitted_force_correction_curvature_rank_scheduler=(
          args.omitted_force_correction_curvature_rank_scheduler),
      omitted_force_correction_curvature_rank_max_condition=(
          args.omitted_force_correction_curvature_rank_max_condition),
      omitted_force_correction_curvature_rank_condition_policy=(
          args.omitted_force_correction_curvature_rank_condition_policy),
      omitted_force_correction_curvature_rank_condition_mad_scale=(
          args.omitted_force_correction_curvature_rank_condition_mad_scale),
      omitted_force_correction_curvature_payload_model=(
          args.omitted_force_correction_curvature_payload_model),
      omitted_force_correction_diagnose_subspace_miss=(
          args.omitted_force_correction_diagnose_subspace_miss),
      omitted_force_correction_max_comm_mb=(
          args.omitted_force_correction_max_comm_mb),
      omitted_force_correction_min_marginal_cost_per_mb=(
          args.omitted_force_correction_min_marginal_cost_per_mb),
      central_equivalence_diagnostic=args.central_equivalence_diagnostic,
      central_equivalence_iterative_diagnostic_iterations=(
          args.central_equivalence_iterative_diagnostic_iterations),
  )
  report.update({
      "graph": str(Path(args.graph)),
      "num_robots": args.num_robots,
      "robot_index_ranges": ranges,
      "baseline_estimate": str(Path(args.baseline_estimate)),
      "baseline_format": baseline_format,
      "rotation_iterations": args.rotation_iterations,
      "translation_iterations": args.translation_iterations,
      "linear_solver": args.linear_solver,
      "interface_schur_preconditioner": args.interface_schur_preconditioner,
      "interface_schur_rotation_preconditioner":
          report.get("interface_schur_rotation_preconditioner"),
      "interface_schur_translation_preconditioner":
          report.get("interface_schur_translation_preconditioner"),
      "interface_schur_coarse_initial_guess":
          args.interface_schur_coarse_initial_guess,
      "interface_schur_coarse_basis_mode":
          args.interface_schur_coarse_basis_mode,
      "interface_schur_coarse_component_limit":
          args.interface_schur_coarse_component_limit,
      "interface_schur_rotation_coarse_basis_mode":
          report.get("interface_schur_rotation_coarse_basis_mode"),
      "interface_schur_translation_coarse_basis_mode":
          report.get("interface_schur_translation_coarse_basis_mode"),
      "interface_schur_rotation_coarse_component_limit":
          report.get("interface_schur_rotation_coarse_component_limit"),
      "interface_schur_translation_coarse_component_limit":
          report.get("interface_schur_translation_coarse_component_limit"),
      "interface_schur_coarse_component_selection_mode":
          report.get("interface_schur_coarse_component_selection_mode"),
      "interface_schur_rotation_coarse_component_selection_mode":
          report.get("interface_schur_rotation_coarse_component_selection_mode"),
      "interface_schur_translation_coarse_component_selection_mode":
          report.get(
              "interface_schur_translation_coarse_component_selection_mode"),
      "interface_schur_residual_deflation_rank":
          args.interface_schur_residual_deflation_rank,
      "interface_schur_residual_deflation_pilot_iterations":
          args.interface_schur_residual_deflation_pilot_iterations,
      "interface_schur_ritz_rank":
          args.interface_schur_ritz_rank,
      "interface_schur_ritz_probe_iterations":
          args.interface_schur_ritz_probe_iterations,
      "interface_schur_rotation_ritz_rank":
          report.get("interface_schur_rotation_ritz_rank"),
      "interface_schur_translation_ritz_rank":
          report.get("interface_schur_translation_ritz_rank"),
      "interface_schur_rotation_ritz_probe_iterations":
          report.get("interface_schur_rotation_ritz_probe_iterations"),
      "interface_schur_translation_ritz_probe_iterations":
          report.get("interface_schur_translation_ritz_probe_iterations"),
      "interface_schur_rotation_ritz_rank_selection_mode":
          report.get("interface_schur_rotation_ritz_rank_selection_mode"),
      "interface_schur_translation_ritz_rank_selection_mode":
          report.get("interface_schur_translation_ritz_rank_selection_mode"),
      "interface_schur_rotation_ritz_value_threshold":
          report.get("interface_schur_rotation_ritz_value_threshold"),
      "interface_schur_translation_ritz_value_threshold":
          report.get("interface_schur_translation_ritz_value_threshold"),
      "interface_schur_rotation_ritz_energy_capture_fraction":
          report.get("interface_schur_rotation_ritz_energy_capture_fraction"),
      "interface_schur_translation_ritz_energy_capture_fraction":
          report.get(
              "interface_schur_translation_ritz_energy_capture_fraction"),
      "interface_schur_translation_budget_candidates":
          report.get("interface_schur_translation_budget_candidates"),
      "interface_schur_ritz_mode":
          args.interface_schur_ritz_mode,
      "central_equivalence_diagnostic":
          args.central_equivalence_diagnostic,
      "central_equivalence_iterative_diagnostic_iterations":
          args.central_equivalence_iterative_diagnostic_iterations,
      "value_score_mode": args.value_score_mode,
      "cost_mode": args.cost_mode,
      "weighted": args.weighted,
      "topology_file": str(Path(args.topology_file)) if args.topology_file else "",
      "topology_round": args.topology_round,
      "topology_round_window": args.topology_round_window,
      "topology_max_relay_hops": args.topology_max_relay_hops,
      "relay_scheduler": args.relay_scheduler,
      "relay_byte_budget_mb": args.relay_byte_budget_mb,
      "relay_cost_rotation_iterations": (
          args.rotation_iterations if args.relay_cost_rotation_iterations is None
          else args.relay_cost_rotation_iterations
      ),
      "relay_cost_translation_iterations": (
          args.translation_iterations
          if args.relay_cost_translation_iterations is None
          else args.relay_cost_translation_iterations
      ),
      "candidate_precert_mode": args.candidate_precert_mode,
      "candidate_precert_min_sep_private_ratio":
          args.candidate_precert_min_sep_private_ratio,
      "candidate_solver": args.candidate_solver,
      "solve_communication_model": args.solve_communication_model,
      "normal_summary_communication_model":
          args.normal_summary_communication_model,
      "summary_selection_mode": args.summary_selection_mode,
      "summary_max_offdiag_block_edges": args.summary_max_offdiag_block_edges,
      "summary_refinement_max_offdiag_block_edges":
          args.summary_refinement_max_offdiag_block_edges,
      "omitted_force_correction_rounds": args.omitted_force_correction_rounds,
      "omitted_force_correction_portfolio_enabled":
          args.omitted_force_correction_portfolio,
      "omitted_force_correction_rank_portfolio_requested":
          args.omitted_force_correction_rank_portfolio,
      "omitted_force_correction_rank_portfolio_enabled":
          report.get("omitted_force_correction_rank_portfolio_enabled", False),
      "omitted_force_correction_curvature_model":
          args.omitted_force_correction_curvature_model,
      "omitted_force_correction_curvature_rank":
          args.omitted_force_correction_curvature_rank,
      "omitted_force_correction_requested_curvature_rank":
          args.omitted_force_correction_curvature_rank,
      "omitted_force_correction_selected_curvature_rank":
          report.get(
              "omitted_force_correction_selected_curvature_rank",
              args.omitted_force_correction_curvature_rank),
      "omitted_force_correction_curvature_rank_scheduler":
          args.omitted_force_correction_curvature_rank_scheduler,
      "omitted_force_correction_curvature_rank_max_condition":
          args.omitted_force_correction_curvature_rank_max_condition,
      "omitted_force_correction_curvature_rank_condition_policy":
          args.omitted_force_correction_curvature_rank_condition_policy,
      "omitted_force_correction_curvature_rank_condition_mad_scale":
          args.omitted_force_correction_curvature_rank_condition_mad_scale,
      "omitted_force_correction_diagnose_subspace_miss":
          args.omitted_force_correction_diagnose_subspace_miss,
      "omitted_force_correction_curvature_payload_model":
          args.omitted_force_correction_curvature_payload_model,
      "omitted_force_correction_max_comm_mb":
          args.omitted_force_correction_max_comm_mb,
      "omitted_force_correction_min_marginal_cost_per_mb":
          args.omitted_force_correction_min_marginal_cost_per_mb,
      "min_topology_separator_coverage": args.min_topology_separator_coverage,
      "handoff_gate_mode": args.handoff_gate_mode,
      "method": "budgeted_certified_dci",
  })
  return report


def _summary_row(report: dict):
  rank_portfolio = report.get("omitted_force_correction_rank_portfolio") or {}
  correction_portfolio = report.get("omitted_force_correction_portfolio") or {}
  translation_budget_portfolio = (
      report.get("translation_budget_portfolio") or {})
  correction = report.get("omitted_force_correction") or {}
  private_interface_split = (
      report.get("private_interface_residual_split") or
      report.get("distributed_stats", {}).get("private_interface_residual_split") or
      {})
  private_interface_combined = private_interface_split.get("combined", {})
  private_interface_rotation = private_interface_split.get("rotation", {})
  private_interface_translation = private_interface_split.get("translation", {})
  correction_rotation = correction.get("rotation") or {}
  correction_translation = correction.get("translation") or {}
  rotation_subspace = correction_rotation.get("reference_subspace_error") or {}
  translation_subspace = (
      correction_translation.get("reference_subspace_error") or {})
  return {
      "graph": report.get("graph", ""),
      "selected": report["selected"],
      "pre_gate_decision": report["pre_gate_decision"],
      "candidate_solved": report["candidate_solved"],
      "baseline_total_cost": report["baseline_total_cost"],
      "dci_baseline_total_cost": report.get("dci_baseline_total_cost"),
      "distributed_total_cost": report["distributed_total_cost"],
      "full_graph_distributed_total_cost":
          report.get("full_graph_distributed_total_cost"),
      "selected_total_cost": report["selected_total_cost"],
      "certificate_total_delta": report["certificate_total_delta"],
      "evidence_global_consensus_accept":
          report.get("evidence_global_consensus_accept"),
      "global_consensus_accept": report["global_consensus_accept"],
      "handoff_gate_mode": report.get("handoff_gate_mode"),
      "handoff_gate_accept": report.get("handoff_gate_accept"),
      "full_graph_handoff_delta": report.get("full_graph_handoff_delta"),
      "handoff_gate_reject_reason": report.get("handoff_gate_reject_reason"),
      "predicted_candidate_comm_mb": report["predicted_candidate_comm_mb"],
      "actual_dci_comm_mb": report["actual_dci_comm_mb"],
      "value_upper_bound_cost_per_mb":
          report["value_upper_bound_cost_per_mb"],
      "value_score": report["value_score"],
      "value_score_mode": report["value_score_mode"],
      "value_threshold": report["value_threshold"],
      "interface_schur_preconditioner":
          report.get("interface_schur_preconditioner"),
      "interface_schur_rotation_preconditioner":
          report.get("interface_schur_rotation_preconditioner"),
      "interface_schur_translation_preconditioner":
          report.get("interface_schur_translation_preconditioner"),
      "interface_schur_coarse_initial_guess":
          report.get("interface_schur_coarse_initial_guess"),
      "interface_schur_coarse_basis_mode":
          report.get("interface_schur_coarse_basis_mode"),
      "interface_schur_coarse_component_limit":
          report.get("interface_schur_coarse_component_limit"),
      "interface_schur_rotation_coarse_basis_mode":
          report.get("interface_schur_rotation_coarse_basis_mode"),
      "interface_schur_translation_coarse_basis_mode":
          report.get("interface_schur_translation_coarse_basis_mode"),
      "interface_schur_rotation_coarse_component_limit":
          report.get("interface_schur_rotation_coarse_component_limit"),
      "interface_schur_translation_coarse_component_limit":
          report.get("interface_schur_translation_coarse_component_limit"),
      "interface_schur_coarse_component_selection_mode":
          report.get("interface_schur_coarse_component_selection_mode"),
      "interface_schur_rotation_coarse_component_selection_mode":
          report.get("interface_schur_rotation_coarse_component_selection_mode"),
      "interface_schur_translation_coarse_component_selection_mode":
          report.get(
              "interface_schur_translation_coarse_component_selection_mode"),
      "interface_schur_residual_deflation_rank":
          report.get("interface_schur_residual_deflation_rank"),
      "interface_schur_residual_deflation_pilot_iterations":
          report.get("interface_schur_residual_deflation_pilot_iterations"),
      "interface_schur_ritz_rank":
          report.get("interface_schur_ritz_rank"),
      "interface_schur_ritz_probe_iterations":
          report.get("interface_schur_ritz_probe_iterations"),
      "interface_schur_rotation_ritz_rank":
          report.get("interface_schur_rotation_ritz_rank"),
      "interface_schur_translation_ritz_rank":
          report.get("interface_schur_translation_ritz_rank"),
      "interface_schur_rotation_ritz_probe_iterations":
          report.get("interface_schur_rotation_ritz_probe_iterations"),
      "interface_schur_translation_ritz_probe_iterations":
          report.get("interface_schur_translation_ritz_probe_iterations"),
      "interface_schur_rotation_ritz_rank_selection_mode":
          report.get("interface_schur_rotation_ritz_rank_selection_mode"),
      "interface_schur_translation_ritz_rank_selection_mode":
          report.get("interface_schur_translation_ritz_rank_selection_mode"),
      "interface_schur_rotation_ritz_value_threshold":
          report.get("interface_schur_rotation_ritz_value_threshold"),
      "interface_schur_translation_ritz_value_threshold":
          report.get("interface_schur_translation_ritz_value_threshold"),
      "interface_schur_rotation_ritz_energy_capture_fraction":
          report.get("interface_schur_rotation_ritz_energy_capture_fraction"),
      "interface_schur_translation_ritz_energy_capture_fraction":
          report.get(
              "interface_schur_translation_ritz_energy_capture_fraction"),
      "interface_schur_translation_budget_candidates":
          report.get("interface_schur_translation_budget_candidates"),
      "interface_schur_ritz_mode":
          report.get("interface_schur_ritz_mode"),
      "translation_budget_portfolio_candidate_count":
          translation_budget_portfolio.get("candidate_count"),
      "translation_budget_portfolio_selected_rank":
          translation_budget_portfolio.get("selected_rank"),
      "translation_budget_portfolio_selected_probe_iterations":
          translation_budget_portfolio.get("selected_probe_iterations"),
      "translation_budget_portfolio_selected_cost":
          translation_budget_portfolio.get("selected_measurement_cost"),
      "topology_separator_edges": report.get("topology_separator_edges"),
      "full_separator_edges": report.get("full_separator_edges"),
      "boundary_pose_unit_count": report.get("boundary_pose_unit_count"),
      "topology_separator_coverage":
          report.get("topology_separator_coverage"),
      "separator_residual_mass_coverage":
          report.get("separator_residual_mass_coverage"),
      "full_separator_residual_cost":
          report.get("full_separator_residual_cost"),
      "covered_separator_residual_cost":
          report.get("covered_separator_residual_cost"),
      "missing_separator_residual_cost":
          report.get("missing_separator_residual_cost"),
      "separator_normal_summary_payload_coverage":
          report.get("separator_normal_summary_payload_coverage"),
      "full_separator_normal_summary_payload_bytes":
          report.get("full_separator_normal_summary_payload_bytes"),
      "covered_separator_normal_summary_payload_bytes":
          report.get("covered_separator_normal_summary_payload_bytes"),
      "missing_separator_normal_summary_payload_bytes":
          report.get("missing_separator_normal_summary_payload_bytes"),
      "separator_normal_summary_full_block_edges":
          report.get("separator_normal_summary_full_block_edges"),
      "separator_normal_summary_covered_block_edges":
          report.get("separator_normal_summary_covered_block_edges"),
      "separator_normal_summary_missing_block_edges":
          report.get("separator_normal_summary_missing_block_edges"),
      "separator_normal_summary_missing_bridge_block_edges":
          report.get("separator_normal_summary_missing_bridge_block_edges"),
      "separator_normal_summary_component_count_delta":
          report.get("separator_normal_summary_component_count_delta"),
      "separator_normal_summary_structural_criticality":
          report.get("separator_normal_summary_structural_criticality"),
      "min_topology_separator_coverage":
          report.get("min_topology_separator_coverage"),
      "topology_round": report.get("topology_round"),
      "topology_round_window": report.get("topology_round_window"),
      "topology_max_relay_hops": report.get("topology_max_relay_hops"),
      "max_relay_hops": report.get("max_relay_hops"),
      "direct_separator_edges": report.get("direct_separator_edges"),
      "relayed_separator_edges": report.get("relayed_separator_edges"),
      "relay_extra_hop_count": report.get("relay_extra_hop_count"),
      "relay_scheduler": report.get("relay_scheduler"),
      "relay_scheduler_selected_relay_edges":
          report.get("relay_scheduler_selected_relay_edges"),
      "relay_scheduler_skipped_relay_edges":
          report.get("relay_scheduler_skipped_relay_edges"),
      "relay_scheduler_reachable_relay_edges":
          report.get("relay_scheduler_reachable_relay_edges"),
      "relay_scheduler_byte_budget_mb":
          report.get("relay_scheduler_byte_budget_mb"),
      "relay_cost_rotation_iterations":
          report.get("relay_cost_rotation_iterations"),
      "relay_cost_translation_iterations":
          report.get("relay_cost_translation_iterations"),
      "relay_scheduler_predicted_selected_relay_mb":
          report.get("relay_scheduler_predicted_selected_relay_mb"),
      "relay_scheduler_pose_summary_units":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_pose_summary_units"),
      "relay_scheduler_structural_bridge_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_bridge_gain"),
      "relay_scheduler_structural_component_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_component_gain"),
      "relay_scheduler_structural_weighted_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_weighted_gain"),
      "relay_scheduler_structural_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_leverage_gain"),
      "relay_scheduler_structural_guarded_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_guarded_leverage_gain"),
      "relay_scheduler_structural_normal_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_normal_leverage_gain"),
      "relay_scheduler_structural_jacobi_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_jacobi_leverage_gain"),
      "relay_scheduler_structural_interface_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_interface_leverage_gain"),
      "relay_scheduler_structural_schur_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_schur_leverage_gain"),
      "relay_scheduler_structural_guarded_schur_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_guarded_schur_leverage_gain"),
      "relay_scheduler_structural_lazy_guarded_schur_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_structural_lazy_guarded_schur_leverage_gain"),
      "relay_scheduler_adaptive_selected_subscheduler":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_adaptive_selected_subscheduler"),
      "relay_scheduler_adaptive_normal_bridge_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_adaptive_normal_bridge_gain"),
      "relay_scheduler_adaptive_density_bridge_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_adaptive_density_bridge_gain"),
      "relay_scheduler_adaptive_normal_normal_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_adaptive_normal_normal_leverage_gain"),
      "relay_scheduler_adaptive_density_normal_leverage_gain":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_adaptive_density_normal_leverage_gain"),
      "relay_scheduler_schur_eval_count":
          report.get("relay_evidence", {}).get(
              "relay_scheduler_schur_eval_count"),
      "relay_scheduler_wall_sec":
          report.get("relay_evidence", {}).get("relay_scheduler_wall_sec"),
      "predicted_relay_extra_comm_mb":
          report.get("predicted_relay_extra_comm_mb"),
      "relay_extra_comm_mb": report.get("relay_extra_comm_mb"),
      "normal_equation_summary_comm_mb":
          report.get("normal_equation_summary_comm_mb"),
      "normal_summary_communication_model":
          report.get("normal_summary_communication_model"),
      "normal_equation_summary_logical_comm_mb":
          report.get("normal_equation_summary_logical_comm_mb"),
      "normal_equation_summary_routed_comm_mb":
          report.get("normal_equation_summary_routed_comm_mb"),
      "normal_summary_routed_extra_relay_mb":
          report.get("topology_routed_normal_summary_communication", {}).get(
              "extra_relay_payload_mb"),
      "normal_summary_mean_path_hops":
          report.get("topology_routed_normal_summary_communication", {}).get(
              "mean_path_hops"),
      "effective_linear_solve_comm_mb":
          report.get("effective_linear_solve_comm_mb"),
      "summary_model_representative":
          report.get("summary_model_representativeness", {}).get(
              "combined", {}).get("normal_model_representative"),
      "summary_model_full_residual_norm":
          report.get("summary_model_representativeness", {}).get(
              "combined", {}).get("full_model_residual_norm"),
      "summary_model_selected_residual_norm":
          report.get("summary_model_representativeness", {}).get(
              "combined", {}).get("selected_model_residual_norm"),
      "summary_model_omitted_residual_norm":
          report.get("summary_model_representativeness", {}).get(
              "combined", {}).get("omitted_model_residual_norm"),
      "summary_model_relative_omitted_residual":
          report.get("summary_model_representativeness", {}).get(
              "combined", {}).get("relative_omitted_model_residual"),
      "private_interface_full_private_residual_norm":
          private_interface_combined.get("full_model_private_residual_norm"),
      "private_interface_full_interface_residual_norm":
          private_interface_combined.get("full_model_interface_residual_norm"),
      "private_interface_full_interface_energy_fraction":
          private_interface_combined.get(
              "full_model_interface_residual_energy_fraction"),
      "private_interface_omitted_private_residual_norm":
          private_interface_combined.get("omitted_model_private_residual_norm"),
      "private_interface_omitted_interface_residual_norm":
          private_interface_combined.get("omitted_model_interface_residual_norm"),
      "private_interface_omitted_interface_energy_fraction":
          private_interface_combined.get(
              "omitted_model_interface_residual_energy_fraction"),
      "private_interface_rotation_full_interface_energy_fraction":
          private_interface_rotation.get(
              "full_model_interface_residual_energy_fraction"),
      "private_interface_translation_full_interface_energy_fraction":
          private_interface_translation.get(
              "full_model_interface_residual_energy_fraction"),
      "omitted_force_correction_comm_mb":
          report.get("omitted_force_correction_effective_comm_mb"),
      "omitted_force_correction_logical_comm_mb":
          report.get("omitted_force_correction_logical_comm_mb"),
      "omitted_force_correction_force_comm_mb":
          report.get("omitted_force_correction_force_comm_mb"),
      "omitted_force_correction_curvature_comm_mb":
          report.get("omitted_force_correction_curvature_comm_mb"),
      "omitted_force_correction_curvature_routed_extra_mb":
          report.get(
              "topology_routed_omitted_force_curvature_communication", {}).get(
                  "extra_relay_payload_mb"),
      "omitted_force_correction_curvature_mean_path_hops":
          report.get(
              "topology_routed_omitted_force_curvature_communication", {}).get(
                  "mean_path_hops"),
      "omitted_force_correction_logical_force_comm_mb":
          report.get("omitted_force_correction", {}).get(
              "communication_estimate", {}).get(
                  "total_omitted_force_payload_mb"),
      "omitted_force_correction_logical_curvature_comm_mb":
          report.get("omitted_force_correction", {}).get(
              "communication_estimate", {}).get("total_curvature_payload_mb"),
      "omitted_force_correction_total_logical_comm_mb":
          report.get("omitted_force_correction", {}).get(
              "communication_estimate", {}).get("total_correction_payload_mb"),
      "omitted_force_correction_rounds":
          report.get("omitted_force_correction_rounds"),
      "omitted_force_correction_curvature_model":
          report.get("omitted_force_correction_curvature_model"),
      "omitted_force_correction_curvature_rank":
          report.get("omitted_force_correction_curvature_rank"),
      "omitted_force_correction_requested_curvature_rank":
          report.get("omitted_force_correction_requested_curvature_rank"),
      "omitted_force_correction_selected_curvature_rank":
          report.get("omitted_force_correction_selected_curvature_rank"),
      "omitted_force_correction_rank_portfolio_requested":
          report.get("omitted_force_correction_rank_portfolio_requested"),
      "omitted_force_correction_rank_portfolio_enabled":
          report.get("omitted_force_correction_rank_portfolio_enabled"),
      "omitted_force_correction_curvature_rank_scheduler":
          report.get("omitted_force_correction_curvature_rank_scheduler"),
      "omitted_force_correction_curvature_rank_max_condition":
          report.get("omitted_force_correction_curvature_rank_max_condition"),
      "omitted_force_correction_curvature_rank_condition_policy":
          report.get(
              "omitted_force_correction_curvature_rank_condition_policy"),
      "omitted_force_correction_curvature_rank_condition_mad_scale":
          report.get(
              "omitted_force_correction_curvature_rank_condition_mad_scale"),
      "omitted_force_correction_diagnose_subspace_miss":
          report.get("omitted_force_correction_diagnose_subspace_miss"),
      "omitted_force_correction_curvature_payload_model":
          report.get("omitted_force_correction_curvature_payload_model"),
      "omitted_force_correction_max_comm_mb":
          report.get("omitted_force_correction_max_comm_mb"),
      "omitted_force_correction_curvature_comm_multiplier":
          report.get("omitted_force_correction_curvature_comm_multiplier"),
      "omitted_force_correction_min_marginal_cost_per_mb":
          report.get("omitted_force_correction_min_marginal_cost_per_mb"),
      "omitted_force_correction_accepted_rotation_rounds":
          correction_rotation.get("accepted_rounds"),
      "omitted_force_correction_accepted_translation_rounds":
          correction_translation.get("accepted_rounds"),
      "omitted_force_correction_portfolio_selected_rounds":
          correction_portfolio.get("selected_rounds"),
      "omitted_force_correction_portfolio_selected_cost":
          correction_portfolio.get("selected_measurement_cost"),
      "omitted_force_correction_portfolio_last_cost":
          correction_portfolio.get("last_measurement_cost"),
      "omitted_force_correction_rotation_missed_error_fraction":
          rotation_subspace.get("missed_error_energy_fraction"),
      "omitted_force_correction_rotation_explainable_error_fraction":
          rotation_subspace.get("explainable_error_energy_fraction"),
      "omitted_force_correction_rotation_reference_error_norm":
          rotation_subspace.get("error_norm"),
      "omitted_force_correction_rotation_subspace_rank":
          rotation_subspace.get("subspace_rank"),
      "omitted_force_correction_translation_missed_error_fraction":
          translation_subspace.get("missed_error_energy_fraction"),
      "omitted_force_correction_translation_explainable_error_fraction":
          translation_subspace.get("explainable_error_energy_fraction"),
      "omitted_force_correction_translation_reference_error_norm":
          translation_subspace.get("error_norm"),
      "omitted_force_correction_translation_subspace_rank":
          translation_subspace.get("subspace_rank"),
      "omitted_force_correction_rank_portfolio_selected_rank":
          rank_portfolio.get("selected_rank"),
      "omitted_force_correction_rank_portfolio_selected_cost":
          rank_portfolio.get("selected_measurement_cost"),
      "candidate_precert_mode": report.get("candidate_precert_mode"),
      "candidate_precert_accept": report.get("candidate_precert_accept"),
      "candidate_precert_sep_private_ratio":
          report.get("candidate_precert_sep_private_ratio"),
      "candidate_precert_min_sep_private_ratio":
          report.get("candidate_precert_min_sep_private_ratio"),
      "candidate_solver": report.get("candidate_solver"),
      "solve_communication_model": report.get("solve_communication_model"),
      "summary_selection_mode": report.get("summary_selection_mode"),
      "summary_max_offdiag_block_edges":
          report.get("summary_max_offdiag_block_edges"),
      "summary_refinement_max_offdiag_block_edges":
          report.get("summary_refinement_max_offdiag_block_edges"),
      "output_path": report["output_path"],
  }


def _parse_translation_budget_candidates(text: str | None):
  if text is None or text == "":
    return None
  candidates = []
  for raw_item in str(text).split(","):
    item = raw_item.strip()
    if not item:
      continue
    pieces = item.split(":")
    if len(pieces) != 2:
      raise argparse.ArgumentTypeError(
          "translation budget candidates must use rank:probe format")
    try:
      rank = int(pieces[0])
      probe_iterations = int(pieces[1])
    except ValueError as exc:
      raise argparse.ArgumentTypeError(
          "translation budget rank/probe must be integers") from exc
    if rank < 0 or probe_iterations < 0:
      raise argparse.ArgumentTypeError(
          "translation budget rank/probe must be nonnegative")
    candidates.append((rank, probe_iterations))
  if not candidates:
    raise argparse.ArgumentTypeError(
        "translation budget candidate list must be nonempty")
  return candidates


def main():
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
  parser.add_argument("--relay-scheduler", default="none",
                      choices=[
                          "none",
                          "edge_residual_budget",
                          "edge_coverage_budget",
                          "pose_summary_budget",
                          "pose_summary_criticality_budget",
                          "pose_summary_weighted_criticality_budget",
                          "pose_summary_leverage_budget",
                          "pose_summary_guarded_leverage_budget",
                          "pose_summary_normal_leverage_budget",
                          "pose_summary_normal_density_budget",
                          "pose_summary_bridge_normal_budget",
                          "pose_summary_adaptive_bridge_density_budget",
                          "pose_summary_jacobi_leverage_budget",
                          "pose_summary_interface_leverage_budget",
                          "pose_summary_schur_leverage_budget",
                          "pose_summary_guarded_schur_leverage_budget",
                          "pose_summary_lazy_guarded_schur_leverage_budget",
                      ])
  parser.add_argument("--relay-byte-budget-mb", type=float, default=None)
  parser.add_argument("--relay-cost-rotation-iterations", type=int, default=None)
  parser.add_argument("--relay-cost-translation-iterations", type=int,
                      default=None)
  parser.add_argument("--candidate-precert-mode", default="none",
                      choices=["none", "separator_private_ratio"])
  parser.add_argument("--candidate-precert-min-sep-private-ratio",
                      type=float, default=0.0)
  parser.add_argument("--candidate-solver", default="edge_subgraph",
                      choices=["edge_subgraph", "normal_summary"])
  parser.add_argument("--handoff-gate-mode", default="evidence_cost",
                      choices=["evidence_cost", "full_graph_cost"])
  parser.add_argument("--min-topology-separator-coverage", type=float,
                      default=0.0)
  parser.add_argument("--output-selected-estimate", required=True)
  parser.add_argument("--output-json", default=None)
  parser.add_argument("--output-summary-row", default=None)
  parser.add_argument("--active-hops", type=int, default=1)
  parser.add_argument("--rotation-iterations", type=int, default=80)
  parser.add_argument("--translation-iterations", type=int, default=100)
  parser.add_argument("--value-threshold", type=float, default=0.1)
  parser.add_argument("--value-score-mode", default="raw_cost",
                      choices=[
                          "raw_cost",
                          "mean_edge_residual",
                          "separator_private_contrast",
                      ])
  parser.add_argument("--linear-solver", default="pcg",
                      choices=[
                          "block_jacobi",
                          "pcg",
                          "compact_pcg",
                          "interface_schur_pcg",
                      ])
  parser.add_argument("--interface-schur-preconditioner",
                      default="block_jacobi",
                      choices=[
                          "diagonal",
                          "block_jacobi",
                          "block_jacobi+coarse",
                          "block_jacobi+balanced_coarse",
                          "overlap_schwarz",
                      ])
  parser.add_argument("--interface-schur-rotation-preconditioner",
                      default=None,
                      choices=[
                          "diagonal",
                          "block_jacobi",
                          "block_jacobi+coarse",
                          "block_jacobi+balanced_coarse",
                          "overlap_schwarz",
                      ])
  parser.add_argument("--interface-schur-translation-preconditioner",
                      default=None,
                      choices=[
                          "diagonal",
                          "block_jacobi",
                          "block_jacobi+coarse",
                          "block_jacobi+balanced_coarse",
                          "overlap_schwarz",
                      ])
  parser.add_argument("--interface-schur-coarse-initial-guess",
                      action="store_true")
  parser.add_argument("--interface-schur-coarse-basis-mode",
                      default="robot_coordinate",
                      choices=[
                          "robot_coordinate",
                          "component_coordinate",
                          "robot_component_coordinate",
                      ])
  parser.add_argument("--interface-schur-coarse-component-limit",
                      type=int, default=32)
  parser.add_argument("--interface-schur-rotation-coarse-basis-mode",
                      default=None,
                      choices=[
                          "robot_coordinate",
                          "component_coordinate",
                          "robot_component_coordinate",
                      ])
  parser.add_argument("--interface-schur-translation-coarse-basis-mode",
                      default=None,
                      choices=[
                          "robot_coordinate",
                          "component_coordinate",
                          "robot_component_coordinate",
                      ])
  parser.add_argument("--interface-schur-rotation-coarse-component-limit",
                      type=int, default=None)
  parser.add_argument("--interface-schur-translation-coarse-component-limit",
                      type=int, default=None)
  parser.add_argument("--interface-schur-coarse-component-selection-mode",
                      default="size",
                      choices=["size", "gradient_energy", "projected_merit"])
  parser.add_argument(
      "--interface-schur-rotation-coarse-component-selection-mode",
      default=None,
      choices=["size", "gradient_energy", "projected_merit"])
  parser.add_argument(
      "--interface-schur-translation-coarse-component-selection-mode",
      default=None,
      choices=["size", "gradient_energy", "projected_merit"])
  parser.add_argument("--interface-schur-residual-deflation-rank",
                      type=int, default=0)
  parser.add_argument(
      "--interface-schur-residual-deflation-pilot-iterations",
      type=int, default=0)
  parser.add_argument("--interface-schur-ritz-rank",
                      type=int, default=0)
  parser.add_argument("--interface-schur-ritz-probe-iterations",
                      type=int, default=0)
  parser.add_argument("--interface-schur-rotation-ritz-rank",
                      type=int, default=None)
  parser.add_argument("--interface-schur-translation-ritz-rank",
                      type=int, default=None)
  parser.add_argument("--interface-schur-rotation-ritz-probe-iterations",
                      type=int, default=None)
  parser.add_argument("--interface-schur-translation-ritz-probe-iterations",
                      type=int, default=None)
  parser.add_argument("--interface-schur-rotation-ritz-rank-selection-mode",
                      default="fixed",
                      choices=["fixed", "value_threshold", "energy_capture"])
  parser.add_argument("--interface-schur-translation-ritz-rank-selection-mode",
                      default="fixed",
                      choices=["fixed", "value_threshold", "energy_capture"])
  parser.add_argument("--interface-schur-rotation-ritz-value-threshold",
                      type=float, default=None)
  parser.add_argument("--interface-schur-translation-ritz-value-threshold",
                      type=float, default=None)
  parser.add_argument("--interface-schur-rotation-ritz-energy-capture-fraction",
                      type=float, default=None)
  parser.add_argument(
      "--interface-schur-translation-ritz-energy-capture-fraction",
      type=float, default=None)
  parser.add_argument("--interface-schur-translation-budget-candidates",
                      type=_parse_translation_budget_candidates,
                      default=None,
                      help=("Comma-separated rank:probe list for a diagnostic "
                            "translation-only measurement portfolio, e.g. "
                            "1:4,2:8,4:8. Default disables the portfolio."))
  parser.add_argument("--interface-schur-ritz-mode",
                      default="preconditioned_operator",
                      choices=[
                          "preconditioned_operator",
                          "generalized",
                          "m_orthogonal_lanczos",
                          "harmonic",
                          "low_harmonic_hybrid",
                          "rot_low_trans_harmonic",
                          "rot_harmonic_trans_low",
                          "portfolio",
                          "measurement_portfolio",
                          "coarse_measurement_portfolio",
                          "partial_measurement_portfolio",
                      ])
  parser.add_argument("--solve-communication-model",
                      default="boundary_pose_exchange",
                      choices=[
                          "boundary_pose_exchange",
                          "compact_continuation_diagnostic",
                      ])
  parser.add_argument("--normal-summary-communication-model",
                      default="logical",
                      choices=[
                          "logical",
                          "topology_routed_equal_edge",
                      ])
  parser.add_argument("--summary-selection-mode", default="all",
                      choices=[
                          "all",
                          "structural_spanning",
                          "residual_force_refinement",
                          "omitted_force_correction",
                      ])
  parser.add_argument("--summary-max-offdiag-block-edges", type=int,
                      default=None)
  parser.add_argument("--summary-refinement-max-offdiag-block-edges", type=int,
                      default=None)
  parser.add_argument("--omitted-force-correction-rounds", type=int, default=1)
  parser.add_argument("--omitted-force-correction-portfolio",
                      action="store_true")
  parser.add_argument("--omitted-force-correction-rank-portfolio",
                      action="store_true")
  parser.add_argument("--omitted-force-correction-curvature-model",
                      default="none",
                      choices=["none", "block_gershgorin",
                               "directional_secant", "subspace_secant"])
  parser.add_argument("--omitted-force-correction-curvature-rank", type=int,
                      default=1)
  parser.add_argument("--omitted-force-correction-curvature-rank-scheduler",
                      default="fixed",
                      choices=["fixed", "projected_residual"])
  parser.add_argument(
      "--omitted-force-correction-curvature-rank-max-condition",
      type=float,
      default=None)
  parser.add_argument(
      "--omitted-force-correction-curvature-rank-condition-policy",
      default="fixed",
      choices=["fixed", "median_mad"])
  parser.add_argument(
      "--omitted-force-correction-curvature-rank-condition-mad-scale",
      type=float,
      default=3.0)
  parser.add_argument("--omitted-force-correction-curvature-payload-model",
                      default="global_projected",
                      choices=["global_projected", "pairwise_projected"])
  parser.add_argument("--omitted-force-correction-diagnose-subspace-miss",
                      action="store_true")
  parser.add_argument("--omitted-force-correction-max-comm-mb", type=float,
                      default=None)
  parser.add_argument("--central-equivalence-diagnostic",
                      action="store_true",
                      help=("Materialize an exact dense interface-Schur "
                            "reference and report Schur-energy gaps. "
                            "Diagnostic only; default off."))
  parser.add_argument("--central-equivalence-iterative-diagnostic-iterations",
                      type=int,
                      default=0,
                      help=("Run an iterative Schur-dual diagnostic for this "
                            "many PCG iterations without materializing dense "
                            "Schur. Diagnostic only; 0 disables it."))
  parser.add_argument("--omitted-force-correction-min-marginal-cost-per-mb",
                      type=float, default=None)
  parser.add_argument("--relaxation", type=float, default=1.0)
  parser.add_argument("--damping", type=float, default=1e-12)
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="dpgo", choices=["legacy", "dpgo"])
  args = parser.parse_args()

  report = build_report(args)
  payload = json.dumps(report, indent=2, sort_keys=True)
  if args.output_json:
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")
  if args.output_summary_row:
    out = Path(args.output_summary_row)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    row = _summary_row(report)
    with out.open("a", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
      if not exists:
        writer.writeheader()
      writer.writerow(row)
  print(payload)


if __name__ == "__main__":
  main()

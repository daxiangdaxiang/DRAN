#!/usr/bin/env python3
"""Build graph-derived basin skeleton candidates for DCI research.

This module is intentionally structural. It does not solve chordal
initialization and does not tune PCG/rank/threshold parameters. It extracts a
deterministic set of graph/measurement-derived primal candidates that can later
define a reduced interface map ``x_B ~= Z y``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np

try:
  from scripts.evaluate_pgo import (  # type: ignore
      Edge,
      edge_chordal_cost,
      parse_g2o_graph,
  )
  from scripts.analyze_shape_drift_oracle import (  # type: ignore
      build_contiguous_robot_map,
      graph_pose_ids,
      pose_dimension_from_edges,
      project_rotation_block,
  )
  from scripts.analyze_distributed_chordal_init import (  # type: ignore
      assemble_rotation_system,
      assemble_translation_system,
      build_graph_local_normal_systems_for_interface_schur,
      rotation_projection_cost_certificate,
      rotation_projection_safety_certificate,
      solve_centralized_chordal_initialization,
      solve_local_interface_schur_fixed_step,
      solve_local_interface_schur_pcg,
      sum_local_interface_schur_contributions,
  )
except ImportError:  # pragma: no cover - used when executed as a script.
  from evaluate_pgo import Edge, edge_chordal_cost, parse_g2o_graph  # type: ignore
  from analyze_shape_drift_oracle import (  # type: ignore
      build_contiguous_robot_map,
      graph_pose_ids,
      pose_dimension_from_edges,
      project_rotation_block,
  )
  from analyze_distributed_chordal_init import (  # type: ignore
      assemble_rotation_system,
      assemble_translation_system,
      build_graph_local_normal_systems_for_interface_schur,
      rotation_projection_cost_certificate,
      rotation_projection_safety_certificate,
      solve_centralized_chordal_initialization,
      solve_local_interface_schur_fixed_step,
      solve_local_interface_schur_pcg,
      sum_local_interface_schur_contributions,
  )


def _robot_pair(a: int, b: int) -> tuple[int, int]:
  return (a, b) if a <= b else (b, a)


def _normalize_cycle(cycle: list[int]) -> list[int]:
  if not cycle:
    return []
  candidates = []
  for seq in (cycle, list(reversed(cycle))):
    min_index = min(range(len(seq)), key=lambda idx: seq[idx])
    rotated = seq[min_index:] + seq[:min_index]
    candidates.append(rotated)
  return min(candidates)


def _tree_path(parent: dict[int, int | None], start: int, end: int) -> list[int]:
  start_ancestors: dict[int, int] = {}
  node: int | None = start
  depth = 0
  while node is not None:
    start_ancestors[node] = depth
    node = parent.get(node)
    depth += 1

  end_path: list[int] = []
  node = end
  while node not in start_ancestors:
    end_path.append(node)
    next_node = parent.get(node)
    if next_node is None:
      return []
    node = next_node

  lca = node
  start_path: list[int] = []
  node = start
  while node != lca:
    start_path.append(node)
    next_node = parent.get(node)
    if next_node is None:
      return []
    node = next_node
  start_path.append(lca)
  return start_path + list(reversed(end_path))


def _graph_bridges_and_articulations(
    nodes: list[int],
    adjacency: dict[int, set[int]],
) -> tuple[list[list[int]], list[int]]:
  discovery: dict[int, int] = {}
  lowlink: dict[int, int] = {}
  parent: dict[int, int | None] = {}
  bridges: set[tuple[int, int]] = set()
  articulations: set[int] = set()
  time = 0

  def visit(node: int):
    nonlocal time
    discovery[node] = time
    lowlink[node] = time
    time += 1
    child_count = 0
    for neighbor in sorted(adjacency.get(node, ())):
      if neighbor not in discovery:
        parent[neighbor] = node
        child_count += 1
        visit(neighbor)
        lowlink[node] = min(lowlink[node], lowlink[neighbor])
        if lowlink[neighbor] > discovery[node]:
          bridges.add(_robot_pair(node, neighbor))
        if parent.get(node) is None:
          if child_count > 1:
            articulations.add(node)
        elif lowlink[neighbor] >= discovery[node]:
          articulations.add(node)
      elif neighbor != parent.get(node):
        lowlink[node] = min(lowlink[node], discovery[neighbor])

  for node in nodes:
    if node not in discovery:
      parent[node] = None
      visit(node)

  return [list(edge) for edge in sorted(bridges)], sorted(articulations)


def _cycle_basis(nodes: list[int],
                 adjacency: dict[int, set[int]]) -> tuple[list[list[int]], set[tuple[int, int]]]:
  parent: dict[int, int | None] = {}
  tree_edges: set[tuple[int, int]] = set()

  for root in nodes:
    if root in parent:
      continue
    parent[root] = None
    stack = [root]
    while stack:
      node = stack.pop()
      for neighbor in sorted(adjacency.get(node, ()), reverse=True):
        if neighbor not in parent:
          parent[neighbor] = node
          tree_edges.add(_robot_pair(node, neighbor))
          stack.append(neighbor)

  all_edges = {
      _robot_pair(node, neighbor)
      for node in nodes
      for neighbor in adjacency.get(node, ())
      if node < neighbor
  }
  cycles: list[list[int]] = []
  cycle_edges: set[tuple[int, int]] = set()
  for edge in sorted(all_edges - tree_edges):
    path = _tree_path(parent, edge[0], edge[1])
    if len(path) < 2:
      continue
    cycle = _normalize_cycle(path)
    cycles.append(cycle)
    for idx, node in enumerate(path):
      next_node = path[(idx + 1) % len(path)]
      cycle_edges.add(_robot_pair(node, next_node))
  return cycles, cycle_edges


def build_basin_skeleton(
    graph_edges: Iterable[Edge],
    pose_ids: Iterable[int],
    robot_of: dict[int, int],
) -> dict:
  """Return deterministic graph-derived skeleton candidates.

  The returned ``z_map`` is a pose-block candidate map. Later solver stages can
  expand each listed pose block into rotation/translation coordinates.
  """
  pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  robot_nodes = sorted({int(robot) for robot in robot_of.values()})
  pair_counts: dict[tuple[int, int], int] = defaultdict(int)
  pair_pose_ids: dict[tuple[int, int], set[int]] = defaultdict(set)

  for edge in graph_edges:
    if edge.i not in robot_of or edge.j not in robot_of:
      continue
    robot_i = int(robot_of[edge.i])
    robot_j = int(robot_of[edge.j])
    if robot_i == robot_j:
      continue
    pair = _robot_pair(robot_i, robot_j)
    pair_counts[pair] += 1
    pair_pose_ids[pair].update((int(edge.i), int(edge.j)))

  adjacency: dict[int, set[int]] = {robot: set() for robot in robot_nodes}
  for robot_i, robot_j in pair_counts:
    adjacency.setdefault(robot_i, set()).add(robot_j)
    adjacency.setdefault(robot_j, set()).add(robot_i)

  bridges, articulations = _graph_bridges_and_articulations(
      robot_nodes, adjacency)
  bridge_pairs = {tuple(edge) for edge in bridges}
  cycles, cycle_pairs = _cycle_basis(robot_nodes, adjacency)

  shared_separator_pose_ids = sorted({
      pose_id
      for pose_set in pair_pose_ids.values()
      for pose_id in pose_set
  })
  bridge_pose_ids = sorted({
      pose_id
      for pair in bridge_pairs
      for pose_id in pair_pose_ids.get(pair, set())
  })
  articulation_pose_ids = sorted({
      pose_id
      for pose_id in shared_separator_pose_ids
      if robot_of.get(pose_id) in set(articulations)
  })
  cycle_pose_ids = sorted({
      pose_id
      for pair in cycle_pairs
      for pose_id in pair_pose_ids.get(pair, set())
  })
  skeleton_pose_ids = sorted(
      set(shared_separator_pose_ids) | set(bridge_pose_ids) |
      set(articulation_pose_ids) | set(cycle_pose_ids))

  reason_order = ("articulation", "bridge", "cycle", "shared_separator")
  pose_reasons: dict[int, set[str]] = {pose_id: set() for pose_id in skeleton_pose_ids}
  for pose_id in articulation_pose_ids:
    pose_reasons.setdefault(pose_id, set()).add("articulation")
  for pose_id in bridge_pose_ids:
    pose_reasons.setdefault(pose_id, set()).add("bridge")
  for pose_id in cycle_pose_ids:
    pose_reasons.setdefault(pose_id, set()).add("cycle")
  for pose_id in shared_separator_pose_ids:
    pose_reasons.setdefault(pose_id, set()).add("shared_separator")

  z_map = []
  for block_index, pose_id in enumerate(skeleton_pose_ids):
    reasons = [
        reason for reason in reason_order
        if reason in pose_reasons.get(pose_id, set())
    ]
    z_map.append({
        "block_index": block_index,
        "pose_id": int(pose_id),
        "robot": int(robot_of[pose_id]),
        "reasons": reasons,
    })

  robot_graph_edges = []
  for pair in sorted(pair_counts):
    robot_graph_edges.append({
        "robots": [int(pair[0]), int(pair[1])],
        "edge_count": int(pair_counts[pair]),
        "separator_pose_ids": sorted(int(pose_id) for pose_id in pair_pose_ids[pair]),
    })

  return {
      "model": "dci_basin_skeleton",
      "pose_count": len(pose_ids),
      "robot_count": len(robot_nodes),
      "robot_graph_edges": robot_graph_edges,
      "bridge_robot_edges": bridges,
      "articulation_robots": articulations,
      "cycle_basis": cycles,
      "shared_separator_pose_ids": shared_separator_pose_ids,
      "bridge_pose_ids": bridge_pose_ids,
      "articulation_pose_ids": articulation_pose_ids,
      "cycle_pose_ids": cycle_pose_ids,
      "object_candidate_ids": [],
      "skeleton_pose_ids": skeleton_pose_ids,
      "skeleton_pose_count": len(skeleton_pose_ids),
      "z_map": z_map,
  }


def select_initial_skeleton_pose_ids(structural_report: dict,
                                     policy: str = "shared_separator") -> list[int]:
  """Select an initial reduced skeleton from structural candidates."""
  if policy == "shared_separator":
    return sorted(int(pose_id) for pose_id in structural_report.get(
        "shared_separator_pose_ids", []))
  if policy == "bridge_articulation":
    return sorted(
        set(int(pose_id) for pose_id in structural_report.get("bridge_pose_ids", [])) |
        set(int(pose_id) for pose_id in structural_report.get("articulation_pose_ids", [])))
  if policy == "bridge_articulation_cycle":
    return sorted(
        set(int(pose_id) for pose_id in structural_report.get("bridge_pose_ids", [])) |
        set(int(pose_id) for pose_id in structural_report.get("articulation_pose_ids", [])) |
        set(int(pose_id) for pose_id in structural_report.get("cycle_pose_ids", [])))
  raise ValueError(f"unsupported initial skeleton policy: {policy}")


def _variable_interface_pose_ids(
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    anchor_pose: int | None = None,
) -> list[int]:
  pose_ids = sorted(int(pose_id) for pose_id in interface_pose_ids)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  if interface_variable_count % int(block_dim) != 0:
    raise ValueError("interface variable count must be divisible by block_dim")
  expected_pose_count = int(interface_variable_count) // int(block_dim)
  if expected_pose_count > len(pose_ids):
    raise ValueError("more interface variable blocks than interface poses")
  if anchor_pose is not None and int(anchor_pose) in pose_ids:
    pose_ids = [pose_id for pose_id in pose_ids if pose_id != int(anchor_pose)]
  if len(pose_ids) == expected_pose_count:
    return pose_ids
  if len(pose_ids) > expected_pose_count:
    # The default chordal initializer anchors the first pose. When the anchor
    # is an interface endpoint it has no variable block and must not enter Z.
    return pose_ids[-expected_pose_count:]
  raise ValueError("unable to infer variable-carrying interface pose ids")


def _pose_block_selection_basis(
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    skeleton_pose_ids: Iterable[int],
    anchor_pose: int | None = None,
) -> tuple[np.ndarray, list[int]]:
  variable_pose_ids = _variable_interface_pose_ids(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=interface_variable_count,
      block_dim=block_dim,
      anchor_pose=anchor_pose,
  )
  skeleton_pose_set = {int(pose_id) for pose_id in skeleton_pose_ids}
  selected_pose_ids = [
      pose_id for pose_id in variable_pose_ids if pose_id in skeleton_pose_set
  ]
  rows = int(interface_variable_count)
  cols = len(selected_pose_ids) * int(block_dim)
  basis = np.zeros((rows, cols), dtype=float)
  pose_to_block = {
      pose_id: block_index
      for block_index, pose_id in enumerate(variable_pose_ids)
  }
  out_col = 0
  for pose_id in selected_pose_ids:
    block_index = pose_to_block[pose_id]
    row_start = block_index * int(block_dim)
    for local_col in range(int(block_dim)):
      basis[row_start + local_col, out_col] = 1.0
      out_col += 1
  return basis, selected_pose_ids


def robot_coordinate_skeleton_basis(
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    robot_of: dict[int, int],
    selected_robot_ids: Iterable[int] | None = None,
    anchor_pose: int | None = None,
) -> dict:
  """Build a robot-coordinate primal skeleton basis.

  Each selected robot contributes one coordinate block shared by all of its
  variable-carrying interface poses. This is a reduced primal state, not a PCG
  preconditioner: solving with this basis constrains same-robot separator pose
  blocks to move together in the selected coordinates.
  """
  block_dim = int(block_dim)
  variable_pose_ids = _variable_interface_pose_ids(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=interface_variable_count,
      block_dim=block_dim,
      anchor_pose=anchor_pose,
  )
  if selected_robot_ids is None:
    selected_robots = sorted({
        int(robot_of[pose_id])
        for pose_id in variable_pose_ids
        if pose_id in robot_of
    })
  else:
    selected_robots = sorted({int(robot) for robot in selected_robot_ids})
  robot_to_column_block = {
      robot: index
      for index, robot in enumerate(selected_robots)
  }
  rows = int(interface_variable_count)
  cols = len(selected_robots) * block_dim
  basis = np.zeros((rows, cols), dtype=float)
  entries: list[dict] = []
  for pose_block_index, pose_id in enumerate(variable_pose_ids):
    robot = robot_of.get(pose_id)
    if robot is None:
      continue
    robot = int(robot)
    column_block = robot_to_column_block.get(robot)
    if column_block is None:
      continue
    row_start = pose_block_index * block_dim
    col_start = column_block * block_dim
    for coord in range(block_dim):
      row = row_start + coord
      col = col_start + coord
      basis[row, col] = 1.0
      entries.append({
          "pose_id": int(pose_id),
          "robot": int(robot),
          "coordinate": int(coord),
          "row": int(row),
          "column": int(col),
      })
  return {
      "model": "robot_coordinate_skeleton_basis",
      "basis": basis,
      "basis_row_count": int(basis.shape[0]),
      "basis_column_count": int(basis.shape[1]),
      "block_dim": block_dim,
      "variable_interface_pose_ids": variable_pose_ids,
      "selected_robot_ids": selected_robots,
      "entries": entries,
  }


def cycle_coordinate_skeleton_basis(
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    robot_of: dict[int, int],
    cycle_basis: Iterable[Iterable[int]],
    anchor_pose: int | None = None,
) -> dict:
  """Build coordinate modes shared by all interface poses on each robot cycle."""
  block_dim = int(block_dim)
  variable_pose_ids = _variable_interface_pose_ids(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=interface_variable_count,
      block_dim=block_dim,
      anchor_pose=anchor_pose,
  )
  selected_cycles: list[list[int]] = []
  for cycle in cycle_basis:
    normalized = _normalize_cycle([int(robot) for robot in cycle])
    if normalized and normalized not in selected_cycles:
      selected_cycles.append(normalized)
  rows = int(interface_variable_count)
  cols = len(selected_cycles) * block_dim
  basis = np.zeros((rows, cols), dtype=float)
  entries: list[dict] = []
  cycle_sets = [set(cycle) for cycle in selected_cycles]
  for pose_block_index, pose_id in enumerate(variable_pose_ids):
    robot = robot_of.get(pose_id)
    if robot is None:
      continue
    robot = int(robot)
    row_start = pose_block_index * block_dim
    for cycle_index, cycle_robots in enumerate(cycle_sets):
      if robot not in cycle_robots:
        continue
      col_start = cycle_index * block_dim
      for coord in range(block_dim):
        row = row_start + coord
        col = col_start + coord
        basis[row, col] = 1.0
        entries.append({
            "pose_id": int(pose_id),
            "robot": int(robot),
            "cycle_index": int(cycle_index),
            "cycle": selected_cycles[cycle_index],
            "coordinate": int(coord),
            "row": int(row),
            "column": int(col),
        })
  return {
      "model": "cycle_coordinate_skeleton_basis",
      "basis": basis,
      "basis_row_count": int(basis.shape[0]),
      "basis_column_count": int(basis.shape[1]),
      "block_dim": block_dim,
      "variable_interface_pose_ids": variable_pose_ids,
      "selected_cycles": selected_cycles,
      "entries": entries,
  }


def separator_component_coordinate_skeleton_basis(
    graph_edges: list[Edge],
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    robot_of: dict[int, int],
    anchor_pose: int | None = None,
) -> dict:
  """Build coordinate modes shared on connected separator-edge components."""
  block_dim = int(block_dim)
  variable_pose_ids = _variable_interface_pose_ids(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=interface_variable_count,
      block_dim=block_dim,
      anchor_pose=anchor_pose,
  )
  variable_pose_set = set(variable_pose_ids)
  adjacency = {pose_id: set() for pose_id in variable_pose_ids}
  for edge in graph_edges:
    pose_i = int(edge.i)
    pose_j = int(edge.j)
    if pose_i not in variable_pose_set or pose_j not in variable_pose_set:
      continue
    robot_i = robot_of.get(pose_i)
    robot_j = robot_of.get(pose_j)
    if robot_i is None or robot_j is None:
      continue
    if int(robot_i) == int(robot_j):
      continue
    adjacency[pose_i].add(pose_j)
    adjacency[pose_j].add(pose_i)

  components: list[list[int]] = []
  seen: set[int] = set()
  for pose_id in variable_pose_ids:
    if pose_id in seen:
      continue
    stack = [pose_id]
    seen.add(pose_id)
    component: list[int] = []
    while stack:
      current = stack.pop()
      component.append(current)
      for neighbor in sorted(adjacency[current]):
        if neighbor in seen:
          continue
        seen.add(neighbor)
        stack.append(neighbor)
    components.append(sorted(component))
  components.sort(key=lambda item: (item[0] if item else -1, len(item)))

  rows = int(interface_variable_count)
  cols = len(components) * block_dim
  basis = np.zeros((rows, cols), dtype=float)
  entries: list[dict] = []
  pose_to_block = {
      pose_id: pose_index
      for pose_index, pose_id in enumerate(variable_pose_ids)
  }
  for component_index, component in enumerate(components):
    col_start = component_index * block_dim
    for pose_id in component:
      row_start = pose_to_block[pose_id] * block_dim
      for coord in range(block_dim):
        row = row_start + coord
        col = col_start + coord
        basis[row, col] = 1.0
        entries.append({
            "pose_id": int(pose_id),
            "robot": int(robot_of.get(pose_id, -1)),
            "component_index": int(component_index),
            "component": [int(item) for item in component],
            "coordinate": int(coord),
            "row": int(row),
            "column": int(col),
        })
  return {
      "model": "separator_component_coordinate_skeleton_basis",
      "basis": basis,
      "basis_row_count": int(basis.shape[0]),
      "basis_column_count": int(basis.shape[1]),
      "block_dim": block_dim,
      "variable_interface_pose_ids": variable_pose_ids,
      "separator_components": components,
      "entries": entries,
  }


def separator_component_geneo_skeleton_basis(
    graph_edges: list[Edge],
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    robot_of: dict[int, int],
    full_schur: np.ndarray,
    full_rhs: np.ndarray,
    max_modes_per_component: int,
    anchor_pose: int | None = None,
) -> dict:
  """Build component-local generalized-eigen separator modes.

  This diagnostic uses the dense interface Schur operator to choose local
  separator-component modes by projected solution magnitude. It is a model for
  a future component-local distributed setup, not the final communication
  accounting.
  """
  block_dim = int(block_dim)
  variable_pose_ids = _variable_interface_pose_ids(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=interface_variable_count,
      block_dim=block_dim,
      anchor_pose=anchor_pose,
  )
  full_schur = np.asarray(full_schur, dtype=float)
  full_rhs = np.asarray(full_rhs, dtype=float).reshape(-1)
  row_count = int(interface_variable_count)
  if full_schur.shape != (row_count, row_count):
    raise ValueError("full_schur shape must match interface variable count")
  if len(full_rhs) != row_count:
    raise ValueError("full_rhs length must match interface variable count")
  mode_limit = max(0, int(max_modes_per_component))

  variable_pose_set = set(variable_pose_ids)
  adjacency = {pose_id: set() for pose_id in variable_pose_ids}
  for edge in graph_edges:
    pose_i = int(edge.i)
    pose_j = int(edge.j)
    if pose_i not in variable_pose_set or pose_j not in variable_pose_set:
      continue
    robot_i = robot_of.get(pose_i)
    robot_j = robot_of.get(pose_j)
    if robot_i is None or robot_j is None:
      continue
    if int(robot_i) == int(robot_j):
      continue
    adjacency[pose_i].add(pose_j)
    adjacency[pose_j].add(pose_i)

  components: list[list[int]] = []
  seen: set[int] = set()
  for pose_id in variable_pose_ids:
    if pose_id in seen:
      continue
    stack = [pose_id]
    seen.add(pose_id)
    component: list[int] = []
    while stack:
      current = stack.pop()
      component.append(current)
      for neighbor in sorted(adjacency[current]):
        if neighbor in seen:
          continue
        seen.add(neighbor)
        stack.append(neighbor)
    components.append(sorted(component))
  components.sort(key=lambda item: (item[0] if item else -1, len(item)))

  pose_to_block = {
      pose_id: pose_index
      for pose_index, pose_id in enumerate(variable_pose_ids)
  }
  columns: list[np.ndarray] = []
  component_records: list[dict] = []
  for component_index, component in enumerate(components):
    rows: list[int] = []
    for pose_id in component:
      row_start = pose_to_block[pose_id] * block_dim
      rows.extend(range(row_start, row_start + block_dim))
    rows_array = np.asarray(rows, dtype=int)
    if rows_array.size == 0 or mode_limit <= 0:
      component_records.append({
          "component_index": int(component_index),
          "component": [int(item) for item in component],
          "row_count": int(rows_array.size),
          "selected_mode_count": 0,
          "selected_mode_rows": [],
          "selected_eigenvalues": [],
          "selected_scores": [],
      })
      continue
    local_schur = full_schur[rows_array[:, None], rows_array]
    local_schur = 0.5 * (local_schur + local_schur.T)
    local_rhs = full_rhs[rows_array]
    try:
      values, vectors = np.linalg.eigh(local_schur)
    except np.linalg.LinAlgError:
      values, vectors = np.linalg.eig(local_schur)
      values = np.real(values)
      vectors = np.real(vectors)
    rhs_coeff = vectors.T @ local_rhs
    safe_values = np.maximum(np.abs(values), 1e-12)
    scores = np.abs(rhs_coeff) / safe_values
    order = np.lexsort((values, -scores))
    selected = order[:min(mode_limit, len(order))]
    selected_rows: list[int] = []
    selected_values: list[float] = []
    selected_scores: list[float] = []
    for selected_index in selected:
      local_vector = vectors[:, selected_index]
      lifted = np.zeros(row_count, dtype=float)
      lifted[rows_array] = local_vector
      norm = float(np.linalg.norm(lifted))
      if norm <= 1e-12:
        continue
      lifted /= norm
      columns.append(lifted)
      selected_rows.append(int(rows_array[int(np.argmax(np.abs(local_vector)))]))
      selected_values.append(float(values[selected_index]))
      selected_scores.append(float(scores[selected_index]))
    component_records.append({
        "component_index": int(component_index),
        "component": [int(item) for item in component],
        "row_count": int(rows_array.size),
        "selected_mode_count": int(len(selected_rows)),
        "selected_mode_rows": selected_rows,
        "selected_eigenvalues": selected_values,
        "selected_scores": selected_scores,
    })

  if columns:
    basis = _orthonormalize_basis_columns(
        np.column_stack(columns), row_count=row_count, tolerance=1e-12)
  else:
    basis = np.zeros((row_count, 0), dtype=float)
  return {
      "model": "separator_component_geneo_skeleton_basis",
      "basis": basis,
      "basis_row_count": int(basis.shape[0]),
      "basis_column_count": int(basis.shape[1]),
      "block_dim": block_dim,
      "variable_interface_pose_ids": variable_pose_ids,
      "separator_components": components,
      "component_count": int(len(components)),
      "max_modes_per_component": int(mode_limit),
      "component_records": component_records,
  }


def separator_component_merit_skeleton_basis(
    graph_edges: list[Edge],
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    robot_of: dict[int, int],
    full_schur: np.ndarray,
    full_rhs: np.ndarray,
    max_selected_components: int = 0,
    anchor_pose: int | None = None,
    payload_bytes_per_coordinate: int = 8,
) -> dict:
  """Select separator component-coordinate atoms by projected Schur merit."""
  block_dim = int(block_dim)
  row_count = int(interface_variable_count)
  full_schur = np.asarray(full_schur, dtype=float)
  full_rhs = np.asarray(full_rhs, dtype=float).reshape(-1)
  if full_schur.shape != (row_count, row_count):
    raise ValueError("full_schur shape must match interface variable count")
  if len(full_rhs) != row_count:
    raise ValueError("full_rhs length must match interface variable count")
  variable_pose_ids = _variable_interface_pose_ids(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=row_count,
      block_dim=block_dim,
      anchor_pose=anchor_pose,
  )
  variable_pose_set = set(variable_pose_ids)
  adjacency = {pose_id: set() for pose_id in variable_pose_ids}
  for edge in graph_edges:
    pose_i = int(edge.i)
    pose_j = int(edge.j)
    if pose_i not in variable_pose_set or pose_j not in variable_pose_set:
      continue
    robot_i = robot_of.get(pose_i)
    robot_j = robot_of.get(pose_j)
    if robot_i is None or robot_j is None:
      continue
    if int(robot_i) == int(robot_j):
      continue
    adjacency[pose_i].add(pose_j)
    adjacency[pose_j].add(pose_i)

  components: list[list[int]] = []
  seen: set[int] = set()
  for pose_id in variable_pose_ids:
    if pose_id in seen:
      continue
    stack = [pose_id]
    seen.add(pose_id)
    component: list[int] = []
    while stack:
      current = stack.pop()
      component.append(current)
      for neighbor in sorted(adjacency[current]):
        if neighbor in seen:
          continue
        seen.add(neighbor)
        stack.append(neighbor)
    components.append(sorted(component))
  components.sort(key=lambda item: (item[0] if item else -1, len(item)))

  pose_to_block = {
      pose_id: pose_index
      for pose_index, pose_id in enumerate(variable_pose_ids)
  }
  candidate_records: list[dict] = []
  for component_index, component in enumerate(components):
    local_basis = np.zeros((row_count, block_dim), dtype=float)
    for pose_id in component:
      row_start = pose_to_block[pose_id] * block_dim
      for coord in range(block_dim):
        local_basis[row_start + coord, coord] = 1.0
    projected_schur = local_basis.T @ full_schur @ local_basis
    projected_schur = 0.5 * (projected_schur + projected_schur.T)
    projected_rhs = local_basis.T @ full_rhs
    try:
      coeff = np.linalg.solve(projected_schur, projected_rhs)
      singular = False
    except np.linalg.LinAlgError:
      coeff = np.linalg.pinv(projected_schur, rcond=1e-12) @ projected_rhs
      singular = True
    projected_merit = 0.5 * float(projected_rhs @ coeff)
    payload_bytes = int(block_dim * max(1, len(component)) *
                        int(payload_bytes_per_coordinate))
    value_per_byte = (
        projected_merit / float(payload_bytes)
        if payload_bytes > 0 else 0.0)
    candidate_records.append({
        "component_index": int(component_index),
        "component": [int(item) for item in component],
        "basis": local_basis,
        "basis_column_count": int(block_dim),
        "payload_bytes": int(payload_bytes),
        "projected_merit": float(projected_merit),
        "value_per_byte": float(value_per_byte),
        "projected_singular": bool(singular),
        "selected": False,
    })

  ordered_records = sorted(
      candidate_records,
      key=lambda item: (
          -float(item["value_per_byte"]),
          -float(item["projected_merit"]),
          item["component"][0] if item["component"] else -1))
  selected_limit = int(max_selected_components)
  if selected_limit <= 0:
    selected_limit = len(ordered_records)
  selected_records = ordered_records[:selected_limit]
  selected_ids = {int(record["component_index"]) for record in selected_records}
  for record in ordered_records:
    record["selected"] = int(record["component_index"]) in selected_ids

  if selected_records:
    basis = np.column_stack([record["basis"] for record in selected_records])
  else:
    basis = np.zeros((row_count, 0), dtype=float)
  public_records = []
  for record in ordered_records:
    public = {
        key: value for key, value in record.items() if key != "basis"
    }
    public_records.append(public)
  return {
      "model": "separator_component_merit_skeleton_basis",
      "basis": basis,
      "basis_row_count": int(basis.shape[0]),
      "basis_column_count": int(basis.shape[1]),
      "block_dim": block_dim,
      "variable_interface_pose_ids": variable_pose_ids,
      "separator_components": components,
      "selected_components": [
          [int(item) for item in record["component"]]
          for record in selected_records
      ],
      "rejected_components": [
          [int(item) for item in record["component"]]
          for record in ordered_records if not bool(record["selected"])
      ],
      "component_count": int(len(components)),
      "selected_component_count": int(len(selected_records)),
      "max_selected_components": (
          0 if int(max_selected_components) <= 0
          else int(max_selected_components)),
      "component_records": public_records,
  }


def pose_block_promotion_basis(
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    omitted_force_blocks: Iterable[dict],
    max_promoted_pose_blocks: int,
    anchor_pose: int | None = None,
) -> dict:
  """Build full pose-block columns for the largest omitted-force pose blocks."""
  add_budget = max(0, int(max_promoted_pose_blocks))
  candidates = [
      block for block in omitted_force_blocks
      if (not bool(block.get("selected", False)) and
          float(block.get("block_norm", 0.0)) > 0.0)
  ]
  candidates.sort(key=lambda item: (
      -float(item.get("block_norm", 0.0)),
      int(item["pose_id"]),
  ))
  promoted_pose_ids = [
      int(item["pose_id"]) for item in candidates[:add_budget]
  ]
  basis, selected_pose_ids = _pose_block_selection_basis(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=interface_variable_count,
      block_dim=block_dim,
      skeleton_pose_ids=promoted_pose_ids,
      anchor_pose=anchor_pose,
  )
  return {
      "model": "pose_block_promotion_basis",
      "basis": basis,
      "basis_row_count": int(basis.shape[0]),
      "basis_column_count": int(basis.shape[1]),
      "block_dim": int(block_dim),
      "promoted_pose_ids": selected_pose_ids,
      "candidate_pose_ids": [
          int(item["pose_id"]) for item in candidates
      ],
      "max_promoted_pose_blocks": add_budget,
  }


def _orthonormalize_basis_columns(
    basis: np.ndarray,
    row_count: int,
    tolerance: float = 1e-12,
) -> np.ndarray:
  basis_array = np.asarray(basis, dtype=float)
  if basis_array.size == 0:
    return np.zeros((int(row_count), 0), dtype=float)
  if basis_array.ndim == 1:
    basis_array = basis_array.reshape(-1, 1)
  if basis_array.ndim != 2 or basis_array.shape[0] != int(row_count):
    raise ValueError("basis row count mismatch")
  columns: list[np.ndarray] = []
  for column in range(basis_array.shape[1]):
    candidate = basis_array[:, column].copy()
    for existing in columns:
      candidate -= float(existing @ candidate) * existing
    norm = float(np.linalg.norm(candidate))
    if norm > float(tolerance):
      columns.append(candidate / norm)
  if not columns:
    return np.zeros((int(row_count), 0), dtype=float)
  return np.column_stack(columns)


def _append_independent_columns_to_orthonormal_basis(
    current_basis: np.ndarray,
    candidate_basis: np.ndarray,
    row_count: int,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, int]:
  current = _orthonormalize_basis_columns(
      np.asarray(current_basis, dtype=float),
      row_count=row_count,
      tolerance=tolerance,
  )
  candidates = np.asarray(candidate_basis, dtype=float)
  if candidates.size == 0:
    return current, 0
  if candidates.ndim == 1:
    candidates = candidates.reshape(-1, 1)
  if candidates.ndim != 2 or candidates.shape[0] != int(row_count):
    raise ValueError("candidate basis row count mismatch")
  columns: list[np.ndarray] = [
      current[:, column].copy() for column in range(current.shape[1])
  ]
  added = 0
  for column in range(candidates.shape[1]):
    candidate = candidates[:, column].copy()
    for existing in columns:
      candidate -= float(existing @ candidate) * existing
    norm = float(np.linalg.norm(candidate))
    if norm > float(tolerance):
      columns.append(candidate / norm)
      added += 1
  if not columns:
    return np.zeros((int(row_count), 0), dtype=float), 0
  return np.column_stack(columns), int(added)


def _basis_quadratic_merit(
    full_schur: np.ndarray,
    full_rhs: np.ndarray,
    basis: np.ndarray,
) -> dict:
  full_schur = np.asarray(full_schur, dtype=float)
  full_schur = 0.5 * (full_schur + full_schur.T)
  full_rhs = np.asarray(full_rhs, dtype=float).reshape(-1)
  basis_array = np.asarray(basis, dtype=float)
  if basis_array.size == 0:
    lifted = np.zeros_like(full_rhs)
    residual = full_rhs.copy()
    return {
        "coefficients": np.zeros(0, dtype=float),
        "lifted_solution": lifted,
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "merit": 0.0,
        "singular": False,
    }
  if basis_array.ndim == 1:
    basis_array = basis_array.reshape(-1, 1)
  reduced = basis_array.T @ full_schur @ basis_array
  reduced = 0.5 * (reduced + reduced.T)
  rhs = basis_array.T @ full_rhs
  try:
    coefficients = np.linalg.solve(reduced, rhs)
    singular = False
  except np.linalg.LinAlgError:
    coefficients = np.linalg.pinv(reduced, rcond=1e-12) @ rhs
    singular = True
  lifted = basis_array @ coefficients
  residual = full_rhs - full_schur @ lifted
  merit = float(full_rhs @ lifted - 0.5 * lifted @ full_schur @ lifted)
  return {
      "coefficients": coefficients,
      "lifted_solution": lifted,
      "residual": residual,
      "residual_norm": float(np.linalg.norm(residual)),
      "merit": merit,
      "singular": bool(singular),
  }


def pose_block_value_per_byte_promotion_basis(
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    omitted_force_blocks: Iterable[dict],
    full_schur: np.ndarray,
    full_rhs: np.ndarray,
    base_basis: np.ndarray,
    min_value_per_byte: float,
    max_promoted_pose_blocks: int,
    payload_bytes_per_coordinate: int = 8,
    min_relative_value_fraction: float | None = None,
    anchor_pose: int | None = None,
    tolerance: float = 1e-12,
) -> dict:
  """Promote pose blocks by exact incremental Schur merit per payload byte."""
  full_schur = np.asarray(full_schur, dtype=float)
  full_rhs = np.asarray(full_rhs, dtype=float).reshape(-1)
  row_count = len(full_rhs)
  current_basis = _orthonormalize_basis_columns(
      np.asarray(base_basis, dtype=float),
      row_count=row_count,
      tolerance=tolerance,
  )
  current_eval = _basis_quadratic_merit(full_schur, full_rhs, current_basis)
  candidate_blocks = [
      block for block in omitted_force_blocks
      if (not bool(block.get("selected", False)) and
          float(block.get("block_norm", 0.0)) > 0.0)
  ]
  candidate_blocks.sort(key=lambda item: (
      -float(item.get("block_norm", 0.0)),
      int(item["pose_id"]),
  ))
  remaining = {int(block["pose_id"]): dict(block) for block in candidate_blocks}
  accepted: list[dict] = []
  rejected: list[dict] = []
  promoted_pose_ids: list[int] = []
  max_blocks = max(0, int(max_promoted_pose_blocks))
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))
  threshold = float(min_value_per_byte)
  relative_fraction = (
      None if min_relative_value_fraction is None
      else max(0.0, float(min_relative_value_fraction)))
  first_value_per_byte: float | None = None
  final_effective_threshold = threshold
  tol = float(tolerance)

  def evaluate_pose(pose_id: int) -> dict:
    candidate_basis_report = pose_block_promotion_basis(
        interface_pose_ids=interface_pose_ids,
        interface_variable_count=interface_variable_count,
        block_dim=block_dim,
        omitted_force_blocks=[{
            "pose_id": int(pose_id),
            "selected": False,
            "block_norm": float(remaining[pose_id].get("block_norm", 0.0)),
        }],
        max_promoted_pose_blocks=1,
        anchor_pose=anchor_pose,
    )
    combined_basis, added_columns = (
        _append_independent_columns_to_orthonormal_basis(
            current_basis,
            candidate_basis_report["basis"],
            row_count=row_count,
            tolerance=tol,
        )
    )
    if added_columns <= 0:
      return {
          "pose_id": int(pose_id),
          "reason": "dependent",
          "incremental_merit": 0.0,
          "payload_bytes": 0,
          "value_per_byte": 0.0,
          "combined_basis": current_basis,
          "combined_eval": current_eval,
          "candidate_basis_column_count": 0,
      }
    combined_eval = _basis_quadratic_merit(full_schur, full_rhs, combined_basis)
    incremental_merit = float(combined_eval["merit"] - current_eval["merit"])
    payload_bytes = int(added_columns * payload_per_coord)
    value_per_byte = (
        incremental_merit / float(payload_bytes)
        if payload_bytes > 0 else 0.0)
    return {
        "pose_id": int(pose_id),
        "reason": "candidate",
        "incremental_merit": incremental_merit,
        "payload_bytes": payload_bytes,
        "value_per_byte": value_per_byte,
        "combined_basis": combined_basis,
        "combined_eval": combined_eval,
        "candidate_basis_column_count": int(candidate_basis_report[
            "basis_column_count"]),
    }

  while remaining and len(accepted) < max_blocks:
    evaluations = [evaluate_pose(pose_id) for pose_id in sorted(remaining)]
    evaluations.sort(key=lambda item: (
        -float(item["value_per_byte"]),
        -float(item["incremental_merit"]),
        int(item["pose_id"]),
    ))
    best = evaluations[0]
    relative_threshold = (
        None if first_value_per_byte is None or relative_fraction is None
        else relative_fraction * first_value_per_byte)
    effective_threshold = threshold
    if relative_threshold is not None:
      effective_threshold = max(effective_threshold, relative_threshold)
    final_effective_threshold = float(effective_threshold)
    stop_for_low_value = float(best["value_per_byte"]) < effective_threshold
    if float(best["incremental_merit"]) <= tol or stop_for_low_value:
      for item in evaluations:
        if item["reason"] == "dependent":
          reason = "dependent"
        elif (relative_threshold is not None and
              float(item["value_per_byte"]) < relative_threshold):
          reason = "below_relative_value_threshold"
        else:
          reason = "below_value_threshold"
        rejected.append({
            "pose_id": int(item["pose_id"]),
            "reason": reason,
            "incremental_merit": float(item["incremental_merit"]),
            "payload_bytes": int(item["payload_bytes"]),
            "value_per_byte": float(item["value_per_byte"]),
            "effective_value_threshold": float(effective_threshold),
            "relative_value_threshold": (
                None if relative_threshold is None
                else float(relative_threshold)),
        })
      remaining.clear()
      break
    current_basis = np.asarray(best["combined_basis"], dtype=float)
    current_eval = dict(best["combined_eval"])
    if first_value_per_byte is None:
      first_value_per_byte = float(best["value_per_byte"])
    promoted_pose_ids.append(int(best["pose_id"]))
    accepted.append({
        "pose_id": int(best["pose_id"]),
        "incremental_merit": float(best["incremental_merit"]),
        "payload_bytes": int(best["payload_bytes"]),
        "value_per_byte": float(best["value_per_byte"]),
        "effective_value_threshold": float(effective_threshold),
        "relative_value_threshold": (
            None if relative_threshold is None else float(relative_threshold)),
        "residual_norm_after_accept": float(current_eval["residual_norm"]),
    })
    remaining.pop(int(best["pose_id"]), None)

  for pose_id in sorted(remaining):
    item = evaluate_pose(pose_id)
    rejected.append({
        "pose_id": int(pose_id),
        "reason": "budget_exhausted",
        "incremental_merit": float(item["incremental_merit"]),
        "payload_bytes": int(item["payload_bytes"]),
        "value_per_byte": float(item["value_per_byte"]),
    })

  return {
      "model": "pose_block_value_per_byte_promotion_basis",
      "basis": current_basis,
      "basis_row_count": int(current_basis.shape[0]),
      "basis_column_count": int(current_basis.shape[1]),
      "block_dim": int(block_dim),
      "promoted_pose_ids": promoted_pose_ids,
      "accepted_promotions": accepted,
      "rejected_promotions": rejected,
      "candidate_pose_ids": [int(block["pose_id"]) for block in candidate_blocks],
      "min_value_per_byte": threshold,
      "min_relative_value_fraction": relative_fraction,
      "first_value_per_byte": (
          None if first_value_per_byte is None
          else float(first_value_per_byte)),
      "final_effective_value_threshold": float(final_effective_threshold),
      "max_promoted_pose_blocks": max_blocks,
      "payload_bytes_per_coordinate": payload_per_coord,
      "base_basis_column_count": int(
          _orthonormalize_basis_columns(
              np.asarray(base_basis, dtype=float),
              row_count=row_count,
              tolerance=tolerance,
          ).shape[1]),
      "final_omitted_force_norm": float(current_eval["residual_norm"]),
      "final_merit": float(current_eval["merit"]),
      "final_singular": bool(current_eval["singular"]),
  }


def local_schur_residual_packets(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_state: np.ndarray,
    interface_pose_ids: Iterable[int] | None = None,
    block_dim: int | None = None,
    variable_count: int | None = None,
    damping: float = 0.0,
    payload_bytes_per_coordinate: int = 8,
) -> dict:
  """Compute robot-local Schur residual packets for an interface state.

  Each robot can form ``r_B^r = b_B^r - S_BB^r x_B`` from its local Schur
  contribution. Summing these packets recovers the exact full-interface
  omitted force when all local systems participate.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
  interface_state = np.asarray(interface_state, dtype=float).reshape(-1)
  if interface_state.shape[0] != len(interface_indices):
    raise ValueError("interface_state length must match interface_indices")
  if variable_count is None:
    inferred = int(np.max(interface_indices)) + 1
    for system in local_systems:
      hessian = system.get("hessian")
      if hessian is not None:
        inferred = max(inferred, int(hessian.shape[0]))
    variable_count = inferred
  variable_count = int(variable_count)
  if variable_count <= 0:
    raise ValueError("variable_count must be positive")
  if int(np.min(interface_indices)) < 0 or int(np.max(interface_indices)) >= variable_count:
    raise ValueError("interface index outside variable dimension")

  packets: list[dict] = []
  aggregate_residual = np.zeros(len(interface_indices), dtype=float)
  local_singular_count = 0
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    gradient = system.get("gradient")
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if hessian is None or gradient is None:
      raise ValueError(f"local system {system_index} is missing hessian/gradient")
    if hasattr(hessian, "toarray"):
      hessian = hessian.toarray()
    hessian = np.asarray(hessian, dtype=float)
    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    if hessian.shape != (variable_count, variable_count):
      raise ValueError(
          f"local system {system_index} hessian has wrong shape")
    if gradient.shape[0] != variable_count:
      raise ValueError(
          f"local system {system_index} gradient has wrong dimension")
    if private_indices.size:
      if (int(np.min(private_indices)) < 0 or
          int(np.max(private_indices)) >= variable_count):
        raise ValueError(
            f"local system {system_index} private index outside dimension")
      if set(int(index) for index in private_indices).intersection(
          int(index) for index in interface_indices):
        raise ValueError(
            f"local system {system_index} private/interface overlap")

    h_bb = hessian[interface_indices[:, None], interface_indices]
    g_b = gradient[interface_indices]
    if private_indices.size == 0:
      local_schur = h_bb
      local_rhs = g_b
    else:
      h_pp = hessian[private_indices[:, None], private_indices]
      if damping > 0.0:
        h_pp = h_pp + float(damping) * np.eye(len(private_indices))
      h_pb = hessian[private_indices[:, None], interface_indices]
      h_bp = hessian[interface_indices[:, None], private_indices]
      g_p = gradient[private_indices]
      try:
        private_interface_response = np.linalg.solve(h_pp, h_pb)
        private_rhs_response = np.linalg.solve(h_pp, g_p)
      except np.linalg.LinAlgError:
        local_singular_count += 1
        h_pp_pinv = np.linalg.pinv(h_pp, rcond=1e-12)
        private_interface_response = h_pp_pinv @ h_pb
        private_rhs_response = h_pp_pinv @ g_p
      local_schur = h_bb - h_bp @ private_interface_response
      local_rhs = g_b - h_bp @ private_rhs_response
    local_schur = 0.5 * (local_schur + local_schur.T)
    residual = local_rhs - local_schur @ interface_state
    aggregate_residual += residual
    packets.append({
        "robot": system.get("robot", system_index),
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "coordinate_count": int(len(residual)),
        "payload_bytes": int(len(residual) * payload_per_coord),
    })

  aggregate_pose_blocks: list[dict] = []
  if interface_pose_ids is not None and block_dim is not None:
    pose_ids = [int(pose_id) for pose_id in interface_pose_ids]
    block_dim = int(block_dim)
    if block_dim <= 0:
      raise ValueError("block_dim must be positive")
    if len(pose_ids) * block_dim != len(interface_indices):
      raise ValueError(
          "interface_pose_ids and block_dim do not match interface indices")
    for block_index, pose_id in enumerate(pose_ids):
      start = block_index * block_dim
      block = aggregate_residual[start:start + block_dim]
      aggregate_pose_blocks.append({
          "pose_id": int(pose_id),
          "block_norm": float(np.linalg.norm(block)),
          "block_residual": block,
      })

  payload_bytes = int(sum(int(packet["payload_bytes"]) for packet in packets))
  return {
      "model": "local_schur_residual_packets",
      "packets": packets,
      "packet_count": int(len(packets)),
      "payload_bytes_per_coordinate": payload_per_coord,
      "payload_bytes": payload_bytes,
      "interface_variable_count": int(len(interface_indices)),
      "aggregate_residual": aggregate_residual,
      "aggregate_residual_norm": float(np.linalg.norm(aggregate_residual)),
      "aggregate_pose_blocks": aggregate_pose_blocks,
      "local_singular_count": int(local_singular_count),
  }


def local_schur_sparse_residual_packets(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_state: np.ndarray,
    interface_pose_ids: Iterable[int],
    block_dim: int,
    variable_count: int | None = None,
    damping: float = 0.0,
    block_norm_threshold: float = 0.0,
    top_k_blocks_per_robot: int | None = None,
    payload_bytes_per_coordinate: int = 8,
    payload_index_bytes: int = 4,
) -> dict:
  """Send only selected separator-block residuals from each local packet."""
  pose_ids = [int(pose_id) for pose_id in interface_pose_ids]
  block_dim = int(block_dim)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  dense = local_schur_residual_packets(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_state=interface_state,
      interface_pose_ids=pose_ids,
      block_dim=block_dim,
      variable_count=variable_count,
      damping=damping,
      payload_bytes_per_coordinate=payload_bytes_per_coordinate,
  )
  interface_variable_count = int(dense["interface_variable_count"])
  if len(pose_ids) * block_dim != interface_variable_count:
    raise ValueError(
        "interface_pose_ids and block_dim do not match interface variables")
  threshold = max(0.0, float(block_norm_threshold))
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))
  index_bytes = max(0, int(payload_index_bytes))
  top_k = None if top_k_blocks_per_robot is None else max(0, int(top_k_blocks_per_robot))

  aggregate_residual = np.zeros(interface_variable_count, dtype=float)
  sparse_packets: list[dict] = []
  selected_block_count = 0
  dropped_block_count = 0
  dropped_residual_norm_bound = 0.0
  payload_bytes = 0

  for packet in dense["packets"]:
    residual = np.asarray(packet["residual"], dtype=float).reshape(-1)
    blocks: list[dict] = []
    dropped_blocks: list[dict] = []
    for block_index, pose_id in enumerate(pose_ids):
      start = block_index * block_dim
      block = residual[start:start + block_dim]
      block_norm = float(np.linalg.norm(block))
      block_record = {
          "pose_id": int(pose_id),
          "block_index": int(block_index),
          "residual_block": block,
          "block_norm": block_norm,
      }
      if block_norm > threshold:
        blocks.append(block_record)
      else:
        dropped_blocks.append({**block_record, "reason": "below_threshold"})
    blocks.sort(key=lambda item: (-float(item["block_norm"]), int(item["pose_id"])))
    if top_k is not None and len(blocks) > top_k:
      kept = blocks[:top_k]
      for item in blocks[top_k:]:
        dropped_blocks.append({**item, "reason": "top_k_dropped"})
      blocks = kept
    for item in blocks:
      start = int(item["block_index"]) * block_dim
      aggregate_residual[start:start + block_dim] += np.asarray(
          item["residual_block"], dtype=float)
    packet_dropped_bound = float(
        sum(float(item["block_norm"]) for item in dropped_blocks))
    dropped_residual_norm_bound += packet_dropped_bound
    block_payload = len(blocks) * (block_dim * payload_per_coord + index_bytes)
    selected_block_count += int(len(blocks))
    dropped_block_count += int(len(dropped_blocks))
    payload_bytes += int(block_payload)
    sparse_packets.append({
        "robot": packet["robot"],
        "blocks": blocks,
        "dropped_blocks": dropped_blocks,
        "selected_block_count": int(len(blocks)),
        "dropped_block_count": int(len(dropped_blocks)),
        "dropped_residual_norm_bound": packet_dropped_bound,
        "payload_bytes": int(block_payload),
    })

  full_aggregate = np.asarray(dense["aggregate_residual"], dtype=float).reshape(-1)
  error = aggregate_residual - full_aggregate
  return {
      "model": "local_schur_sparse_residual_packets",
      "dense_packet_report": dense,
      "packets": sparse_packets,
      "packet_count": int(len(sparse_packets)),
      "selected_block_count": int(selected_block_count),
      "dropped_block_count": int(dropped_block_count),
      "dropped_residual_norm_bound": float(dropped_residual_norm_bound),
      "payload_bytes_per_coordinate": payload_per_coord,
      "payload_index_bytes": index_bytes,
      "payload_bytes": int(payload_bytes),
      "dense_payload_bytes": int(dense["payload_bytes"]),
      "block_norm_threshold": threshold,
      "top_k_blocks_per_robot": top_k,
      "interface_variable_count": interface_variable_count,
      "aggregate_residual": aggregate_residual,
      "aggregate_residual_norm": float(np.linalg.norm(aggregate_residual)),
      "full_aggregate_residual": full_aggregate,
      "full_aggregate_residual_norm": float(np.linalg.norm(full_aggregate)),
      "sparse_residual_error": error,
      "sparse_residual_error_norm": float(np.linalg.norm(error)),
  }


def sparse_basin_gate_decision(
    before_sparse_report: dict,
    after_sparse_report: dict,
    max_omitted_force_increase: float = 0.0,
) -> dict:
  """Certify whether a sparse residual gate decision is insensitive to drops."""
  before_norm = float(before_sparse_report.get("aggregate_residual_norm", 0.0))
  after_norm = float(after_sparse_report.get("aggregate_residual_norm", 0.0))
  before_bound = max(
      0.0, float(before_sparse_report.get("dropped_residual_norm_bound", 0.0)))
  after_bound = max(
      0.0, float(after_sparse_report.get("dropped_residual_norm_bound", 0.0)))
  max_increase = max(0.0, float(max_omitted_force_increase))
  before_lower = max(0.0, before_norm - before_bound)
  before_upper = before_norm + before_bound
  after_lower = max(0.0, after_norm - after_bound)
  after_upper = after_norm + after_bound
  if after_upper <= before_lower + max_increase:
    decision = "safe_accept"
  elif after_lower > before_upper + max_increase:
    decision = "safe_reject"
  else:
    decision = "uncertain"
  return {
      "model": "sparse_basin_gate_decision",
      "decision": decision,
      "max_omitted_force_increase": max_increase,
      "before_sparse_norm": before_norm,
      "after_sparse_norm": after_norm,
      "before_dropped_bound": before_bound,
      "after_dropped_bound": after_bound,
      "before_lower_bound": float(before_lower),
      "before_upper_bound": float(before_upper),
      "after_lower_bound": float(after_lower),
      "after_upper_bound": float(after_upper),
      "accept_margin_lower_bound": float(
          before_lower + max_increase - after_upper),
      "reject_margin_lower_bound": float(
          after_lower - before_upper - max_increase),
  }


def streaming_sparse_basin_gate_decision(
    before_exact_sparse_report: dict,
    after_exact_sparse_report: dict,
    block_dim: int,
    payload_bytes_per_coordinate: int = 8,
    payload_index_bytes: int = 4,
    max_omitted_force_increase: float = 0.0,
) -> dict:
  """Stream residual blocks until the sparse basin gate is certified."""
  block_dim = int(block_dim)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  interface_variable_count = int(before_exact_sparse_report[
      "interface_variable_count"])
  if int(after_exact_sparse_report["interface_variable_count"]) != interface_variable_count:
    raise ValueError("before/after interface dimensions differ")
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))
  index_bytes = max(0, int(payload_index_bytes))
  block_payload_bytes = int(block_dim * payload_per_coord + index_bytes)

  def collect_blocks(report: dict, side: str) -> list[dict]:
    blocks: list[dict] = []
    for packet in report.get("packets", []):
      for block in packet.get("blocks", []):
        block_index = int(block["block_index"])
        residual_block = np.asarray(
            block["residual_block"], dtype=float).reshape(-1)
        if residual_block.shape[0] != block_dim:
          raise ValueError("residual block dimension mismatch")
        blocks.append({
            "side": side,
            "robot": packet.get("robot"),
            "pose_id": int(block["pose_id"]),
            "block_index": block_index,
            "residual_block": residual_block,
            "block_norm": float(block["block_norm"]),
        })
    return blocks

  all_blocks = (
      collect_blocks(before_exact_sparse_report, "before") +
      collect_blocks(after_exact_sparse_report, "after"))
  all_blocks.sort(key=lambda item: (
      -float(item["block_norm"]),
      0 if item["side"] == "before" else 1,
      str(item["robot"]),
      int(item["pose_id"]),
      int(item["block_index"]),
  ))
  before_aggregate = np.zeros(interface_variable_count, dtype=float)
  after_aggregate = np.zeros(interface_variable_count, dtype=float)
  before_dropped_bound = float(
      sum(float(block["block_norm"]) for block in all_blocks
          if block["side"] == "before"))
  after_dropped_bound = float(
      sum(float(block["block_norm"]) for block in all_blocks
          if block["side"] == "after"))
  selected_blocks: list[dict] = []
  trace: list[dict] = []
  payload_bytes = 0

  def make_report(side: str) -> dict:
    if side == "before":
      aggregate = before_aggregate
      dropped_bound = before_dropped_bound
    else:
      aggregate = after_aggregate
      dropped_bound = after_dropped_bound
    return {
        "aggregate_residual": aggregate.copy(),
        "aggregate_residual_norm": float(np.linalg.norm(aggregate)),
        "dropped_residual_norm_bound": float(dropped_bound),
    }

  decision = sparse_basin_gate_decision(
      make_report("before"),
      make_report("after"),
      max_omitted_force_increase=max_omitted_force_increase,
  )
  if decision["decision"] not in {"safe_accept", "safe_reject"}:
    for block in all_blocks:
      start = int(block["block_index"]) * block_dim
      if start < 0 or start + block_dim > interface_variable_count:
        raise ValueError("block index outside interface residual")
      if block["side"] == "before":
        before_aggregate[start:start + block_dim] += block["residual_block"]
        before_dropped_bound -= float(block["block_norm"])
      else:
        after_aggregate[start:start + block_dim] += block["residual_block"]
        after_dropped_bound -= float(block["block_norm"])
      before_dropped_bound = max(0.0, before_dropped_bound)
      after_dropped_bound = max(0.0, after_dropped_bound)
      selected = {
          **block,
          "payload_bytes": block_payload_bytes,
      }
      selected_blocks.append(selected)
      payload_bytes += block_payload_bytes
      decision = sparse_basin_gate_decision(
          make_report("before"),
          make_report("after"),
          max_omitted_force_increase=max_omitted_force_increase,
      )
      trace.append({
          "selected_block_count": int(len(selected_blocks)),
          "payload_bytes": int(payload_bytes),
          "last_block": selected,
          "decision": decision,
      })
      if decision["decision"] in {"safe_accept", "safe_reject"}:
        break

  before_report = make_report("before")
  after_report = make_report("after")
  before_full = np.asarray(
      before_exact_sparse_report.get("full_aggregate_residual",
                                     before_exact_sparse_report.get(
                                         "aggregate_residual",
                                         np.zeros(interface_variable_count))),
      dtype=float).reshape(-1)
  after_full = np.asarray(
      after_exact_sparse_report.get("full_aggregate_residual",
                                    after_exact_sparse_report.get(
                                        "aggregate_residual",
                                        np.zeros(interface_variable_count))),
      dtype=float).reshape(-1)
  exact_decision = (
      "accept"
      if np.linalg.norm(after_full) <=
      np.linalg.norm(before_full) + max(0.0, float(max_omitted_force_increase))
      else "reject")
  return {
      "model": "streaming_sparse_basin_gate_decision",
      "decision": decision,
      "exact_decision": exact_decision,
      "before_stream_report": before_report,
      "after_stream_report": after_report,
      "selected_blocks": selected_blocks,
      "selected_block_count": int(len(selected_blocks)),
      "candidate_block_count": int(len(all_blocks)),
      "payload_bytes": int(payload_bytes),
      "payload_bytes_per_block": block_payload_bytes,
      "payload_bytes_per_coordinate": payload_per_coord,
      "payload_index_bytes": index_bytes,
      "trace": trace,
  }


def topology_round_streaming_sparse_basin_gate_decision(
    before_exact_sparse_report: dict,
    after_exact_sparse_report: dict,
    block_dim: int,
    payload_bytes_per_coordinate: int = 8,
    payload_index_bytes: int = 4,
    max_omitted_force_increase: float = 0.0,
) -> dict:
  """Round-based local-priority residual streaming without global ordering."""
  block_dim = int(block_dim)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  interface_variable_count = int(before_exact_sparse_report[
      "interface_variable_count"])
  if int(after_exact_sparse_report["interface_variable_count"]) != interface_variable_count:
    raise ValueError("before/after interface dimensions differ")
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))
  index_bytes = max(0, int(payload_index_bytes))
  block_payload_bytes = int(block_dim * payload_per_coord + index_bytes)

  queues: dict[object, list[dict]] = {}

  def add_blocks(report: dict, side: str):
    for packet in report.get("packets", []):
      robot = packet.get("robot")
      for block in packet.get("blocks", []):
        residual_block = np.asarray(
            block["residual_block"], dtype=float).reshape(-1)
        if residual_block.shape[0] != block_dim:
          raise ValueError("residual block dimension mismatch")
        queues.setdefault(robot, []).append({
            "side": side,
            "robot": robot,
            "pose_id": int(block["pose_id"]),
            "block_index": int(block["block_index"]),
            "residual_block": residual_block,
            "block_norm": float(block["block_norm"]),
        })

  add_blocks(before_exact_sparse_report, "before")
  add_blocks(after_exact_sparse_report, "after")
  for robot_queue in queues.values():
    robot_queue.sort(key=lambda item: (
        -float(item["block_norm"]),
        0 if item["side"] == "before" else 1,
        int(item["pose_id"]),
        int(item["block_index"]),
    ))

  before_aggregate = np.zeros(interface_variable_count, dtype=float)
  after_aggregate = np.zeros(interface_variable_count, dtype=float)
  before_dropped_bound = float(
      sum(float(block["block_norm"]) for queue in queues.values()
          for block in queue if block["side"] == "before"))
  after_dropped_bound = float(
      sum(float(block["block_norm"]) for queue in queues.values()
          for block in queue if block["side"] == "after"))
  selected_blocks: list[dict] = []
  round_trace: list[dict] = []
  payload_bytes = 0

  def make_report(side: str) -> dict:
    if side == "before":
      aggregate = before_aggregate
      dropped_bound = before_dropped_bound
    else:
      aggregate = after_aggregate
      dropped_bound = after_dropped_bound
    return {
        "aggregate_residual": aggregate.copy(),
        "aggregate_residual_norm": float(np.linalg.norm(aggregate)),
        "dropped_residual_norm_bound": float(dropped_bound),
    }

  def apply_block(block: dict):
    nonlocal before_dropped_bound, after_dropped_bound, payload_bytes
    start = int(block["block_index"]) * block_dim
    if start < 0 or start + block_dim > interface_variable_count:
      raise ValueError("block index outside interface residual")
    if block["side"] == "before":
      before_aggregate[start:start + block_dim] += block["residual_block"]
      before_dropped_bound -= float(block["block_norm"])
    else:
      after_aggregate[start:start + block_dim] += block["residual_block"]
      after_dropped_bound -= float(block["block_norm"])
    before_dropped_bound = max(0.0, before_dropped_bound)
    after_dropped_bound = max(0.0, after_dropped_bound)
    selected = {**block, "payload_bytes": block_payload_bytes}
    selected_blocks.append(selected)
    payload_bytes += block_payload_bytes
    return selected

  decision = sparse_basin_gate_decision(
      make_report("before"),
      make_report("after"),
      max_omitted_force_increase=max_omitted_force_increase,
  )
  round_count = 0
  robot_order = sorted(queues, key=lambda value: str(value))
  while decision["decision"] not in {"safe_accept", "safe_reject"}:
    sent_this_round: list[dict] = []
    for robot in robot_order:
      queue = queues.get(robot, [])
      if not queue:
        continue
      sent_this_round.append(apply_block(queue.pop(0)))
    if not sent_this_round:
      break
    round_count += 1
    decision = sparse_basin_gate_decision(
        make_report("before"),
        make_report("after"),
        max_omitted_force_increase=max_omitted_force_increase,
    )
    round_trace.append({
        "round": int(round_count),
        "sent_blocks": sent_this_round,
        "selected_block_count": int(len(selected_blocks)),
        "payload_bytes": int(payload_bytes),
        "decision": decision,
    })

  before_full = np.asarray(
      before_exact_sparse_report.get("full_aggregate_residual",
                                     before_exact_sparse_report.get(
                                         "aggregate_residual",
                                         np.zeros(interface_variable_count))),
      dtype=float).reshape(-1)
  after_full = np.asarray(
      after_exact_sparse_report.get("full_aggregate_residual",
                                    after_exact_sparse_report.get(
                                        "aggregate_residual",
                                        np.zeros(interface_variable_count))),
      dtype=float).reshape(-1)
  exact_decision = (
      "accept"
      if np.linalg.norm(after_full) <=
      np.linalg.norm(before_full) + max(0.0, float(max_omitted_force_increase))
      else "reject")
  return {
      "model": "topology_round_streaming_sparse_basin_gate_decision",
      "decision": decision,
      "exact_decision": exact_decision,
      "before_stream_report": make_report("before"),
      "after_stream_report": make_report("after"),
      "selected_blocks": selected_blocks,
      "selected_block_count": int(len(selected_blocks)),
      "round_count": int(round_count),
      "payload_bytes": int(payload_bytes),
      "payload_bytes_per_block": block_payload_bytes,
      "candidate_block_count": int(sum(len(queue) for queue in queues.values()) +
                                   len(selected_blocks)),
      "round_trace": round_trace,
  }


def receiver_pulled_streaming_sparse_basin_gate_decision(
    before_exact_sparse_report: dict,
    after_exact_sparse_report: dict,
    block_dim: int,
    payload_bytes_per_coordinate: int = 8,
    payload_index_bytes: int = 4,
    max_omitted_force_increase: float = 0.0,
) -> dict:
  """Pull one highest-priority local queue head at a time until certified."""
  block_dim = int(block_dim)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  interface_variable_count = int(before_exact_sparse_report[
      "interface_variable_count"])
  if int(after_exact_sparse_report["interface_variable_count"]) != interface_variable_count:
    raise ValueError("before/after interface dimensions differ")
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))
  index_bytes = max(0, int(payload_index_bytes))
  block_payload_bytes = int(block_dim * payload_per_coord + index_bytes)

  queues: dict[object, list[dict]] = {}

  def add_blocks(report: dict, side: str):
    for packet in report.get("packets", []):
      robot = packet.get("robot")
      for block in packet.get("blocks", []):
        residual_block = np.asarray(
            block["residual_block"], dtype=float).reshape(-1)
        if residual_block.shape[0] != block_dim:
          raise ValueError("residual block dimension mismatch")
        queues.setdefault(robot, []).append({
            "side": side,
            "robot": robot,
            "pose_id": int(block["pose_id"]),
            "block_index": int(block["block_index"]),
            "residual_block": residual_block,
            "block_norm": float(block["block_norm"]),
        })

  add_blocks(before_exact_sparse_report, "before")
  add_blocks(after_exact_sparse_report, "after")
  for robot_queue in queues.values():
    robot_queue.sort(key=lambda item: (
        -float(item["block_norm"]),
        0 if item["side"] == "before" else 1,
        int(item["pose_id"]),
        int(item["block_index"]),
    ))
  candidate_block_count = int(sum(len(queue) for queue in queues.values()))

  before_aggregate = np.zeros(interface_variable_count, dtype=float)
  after_aggregate = np.zeros(interface_variable_count, dtype=float)
  before_dropped_bound = float(
      sum(float(block["block_norm"]) for queue in queues.values()
          for block in queue if block["side"] == "before"))
  after_dropped_bound = float(
      sum(float(block["block_norm"]) for queue in queues.values()
          for block in queue if block["side"] == "after"))
  selected_blocks: list[dict] = []
  pull_trace: list[dict] = []
  payload_bytes = 0

  def make_report(side: str) -> dict:
    if side == "before":
      aggregate = before_aggregate
      dropped_bound = before_dropped_bound
    else:
      aggregate = after_aggregate
      dropped_bound = after_dropped_bound
    return {
        "aggregate_residual": aggregate.copy(),
        "aggregate_residual_norm": float(np.linalg.norm(aggregate)),
        "dropped_residual_norm_bound": float(dropped_bound),
    }

  def apply_block(block: dict) -> dict:
    nonlocal before_dropped_bound, after_dropped_bound, payload_bytes
    start = int(block["block_index"]) * block_dim
    if start < 0 or start + block_dim > interface_variable_count:
      raise ValueError("block index outside interface residual")
    if block["side"] == "before":
      before_aggregate[start:start + block_dim] += block["residual_block"]
      before_dropped_bound -= float(block["block_norm"])
    else:
      after_aggregate[start:start + block_dim] += block["residual_block"]
      after_dropped_bound -= float(block["block_norm"])
    before_dropped_bound = max(0.0, before_dropped_bound)
    after_dropped_bound = max(0.0, after_dropped_bound)
    selected = {**block, "payload_bytes": block_payload_bytes}
    selected_blocks.append(selected)
    payload_bytes += block_payload_bytes
    return selected

  def head_key(robot: object, block: dict):
    return (
        -float(block["block_norm"]),
        0 if block["side"] == "before" else 1,
        str(robot),
        int(block["pose_id"]),
        int(block["block_index"]),
    )

  decision = sparse_basin_gate_decision(
      make_report("before"),
      make_report("after"),
      max_omitted_force_increase=max_omitted_force_increase,
  )
  pull_count = 0
  while decision["decision"] not in {"safe_accept", "safe_reject"}:
    active_heads = [
        (head_key(robot, queue[0]), robot)
        for robot, queue in queues.items() if queue
    ]
    if not active_heads:
      break
    _, selected_robot = min(active_heads, key=lambda item: item[0])
    selected = apply_block(queues[selected_robot].pop(0))
    pull_count += 1
    decision = sparse_basin_gate_decision(
        make_report("before"),
        make_report("after"),
        max_omitted_force_increase=max_omitted_force_increase,
    )
    pull_trace.append({
        "pull": int(pull_count),
        "selected_block": selected,
        "selected_block_count": int(len(selected_blocks)),
        "payload_bytes": int(payload_bytes),
        "decision": decision,
    })

  before_full = np.asarray(
      before_exact_sparse_report.get("full_aggregate_residual",
                                     before_exact_sparse_report.get(
                                         "aggregate_residual",
                                         np.zeros(interface_variable_count))),
      dtype=float).reshape(-1)
  after_full = np.asarray(
      after_exact_sparse_report.get("full_aggregate_residual",
                                    after_exact_sparse_report.get(
                                        "aggregate_residual",
                                        np.zeros(interface_variable_count))),
      dtype=float).reshape(-1)
  exact_decision = (
      "accept"
      if np.linalg.norm(after_full) <=
      np.linalg.norm(before_full) + max(0.0, float(max_omitted_force_increase))
      else "reject")
  return {
      "model": "receiver_pulled_streaming_sparse_basin_gate_decision",
      "decision": decision,
      "exact_decision": exact_decision,
      "before_stream_report": make_report("before"),
      "after_stream_report": make_report("after"),
      "selected_blocks": selected_blocks,
      "selected_block_count": int(len(selected_blocks)),
      "pull_count": int(pull_count),
      "payload_bytes": int(payload_bytes),
      "payload_bytes_per_block": block_payload_bytes,
      "candidate_block_count": candidate_block_count,
      "pull_trace": pull_trace,
  }


def receiver_pulled_top_residual_block_decision(
    exact_sparse_report: dict,
    block_dim: int,
    selected_pose_ids: Iterable[int] = (),
    payload_bytes_per_coordinate: int = 8,
    payload_index_bytes: int = 4,
    allow_top_set: bool = False,
    tolerance: float = 1e-12,
) -> dict:
  """Pull residual blocks until the largest omitted-force pose is certified."""
  block_dim = int(block_dim)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  interface_variable_count = int(exact_sparse_report["interface_variable_count"])
  if interface_variable_count % block_dim != 0:
    raise ValueError("interface variable count must be divisible by block_dim")
  block_count = int(interface_variable_count // block_dim)
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))
  index_bytes = max(0, int(payload_index_bytes))
  block_payload_bytes = int(block_dim * payload_per_coord + index_bytes)
  selected_pose_set = {int(pose_id) for pose_id in selected_pose_ids}

  queues: dict[object, list[dict]] = {}
  block_pose_ids: dict[int, int] = {}

  def note_pose(block: dict):
    block_index = int(block["block_index"])
    if block_index < 0 or block_index >= block_count:
      raise ValueError("block index outside interface residual")
    pose_id = int(block["pose_id"])
    previous = block_pose_ids.get(block_index)
    if previous is not None and previous != pose_id:
      raise ValueError("inconsistent pose id for residual block")
    block_pose_ids[block_index] = pose_id

  for packet in exact_sparse_report.get("packets", []):
    robot = packet.get("robot")
    for block in packet.get("blocks", []):
      residual_block = np.asarray(
          block["residual_block"], dtype=float).reshape(-1)
      if residual_block.shape[0] != block_dim:
        raise ValueError("residual block dimension mismatch")
      note_pose(block)
      queues.setdefault(robot, []).append({
          "side": "residual",
          "robot": robot,
          "pose_id": int(block["pose_id"]),
          "block_index": int(block["block_index"]),
          "residual_block": residual_block,
          "block_norm": float(block["block_norm"]),
      })
    for block in packet.get("dropped_blocks", []):
      note_pose(block)

  for robot_queue in queues.values():
    robot_queue.sort(key=lambda item: (
        -float(item["block_norm"]),
        int(item["pose_id"]),
        int(item["block_index"]),
    ))
  candidate_block_count = int(sum(len(queue) for queue in queues.values()))

  aggregate = np.zeros(interface_variable_count, dtype=float)
  dropped_bound_by_block = np.zeros(block_count, dtype=float)
  for queue in queues.values():
    for block in queue:
      dropped_bound_by_block[int(block["block_index"])] += float(
          block["block_norm"])

  selected_blocks: list[dict] = []
  pull_trace: list[dict] = []
  payload_bytes = 0
  tol = float(tolerance)

  def block_records() -> list[dict]:
    records: list[dict] = []
    for block_index in range(block_count):
      pose_id = block_pose_ids.get(block_index)
      if pose_id is None:
        continue
      start = block_index * block_dim
      residual_block = aggregate[start:start + block_dim]
      norm = float(np.linalg.norm(residual_block))
      bound = max(0.0, float(dropped_bound_by_block[block_index]))
      records.append({
          "pose_id": int(pose_id),
          "block_index": int(block_index),
          "selected": int(pose_id) in selected_pose_set,
          "residual_norm": norm,
          "dropped_bound": bound,
          "lower_bound": max(0.0, norm - bound),
          "upper_bound": norm + bound,
      })
    records.sort(key=lambda item: (
        item["selected"],
        -float(item["lower_bound"]),
        -float(item["upper_bound"]),
        int(item["pose_id"]),
    ))
    return records

  def make_decision() -> dict:
    records = [record for record in block_records() if not record["selected"]]
    if not records:
      return {
          "decision": "no_candidate",
          "top_pose_id": None,
          "top_lower_bound": 0.0,
          "next_upper_bound": 0.0,
          "margin": 0.0,
          "candidate_bounds": records,
      }
    if bool(allow_top_set):
      lower_sorted = sorted(records, key=lambda item: (
          -float(item["lower_bound"]),
          -float(item["upper_bound"]),
          int(item["pose_id"]),
      ))
      for end in range(1, len(lower_sorted) + 1):
        top_set = lower_sorted[:end]
        outside = lower_sorted[end:]
        min_top_lower = min(float(record["lower_bound"]) for record in top_set)
        max_outside_upper = max(
            [float(record["upper_bound"]) for record in outside] + [0.0])
        margin = min_top_lower - max_outside_upper
        if margin > tol:
          top_pose_ids = [int(record["pose_id"]) for record in top_set]
          if len(top_pose_ids) == 1:
            top = top_set[0]
            return {
                "decision": "safe_top",
                "top_pose_id": int(top["pose_id"]),
                "top_pose_ids": top_pose_ids,
                "top_block_index": int(top["block_index"]),
                "top_lower_bound": float(top["lower_bound"]),
                "top_upper_bound": float(top["upper_bound"]),
                "next_upper_bound": float(max_outside_upper),
                "margin": float(margin),
                "candidate_bounds": records,
            }
          return {
              "decision": "safe_top_set",
              "top_pose_id": int(top_pose_ids[0]),
              "top_pose_ids": top_pose_ids,
              "top_lower_bound": float(min_top_lower),
              "next_upper_bound": float(max_outside_upper),
              "margin": float(margin),
              "candidate_bounds": records,
          }
    top = records[0]
    next_upper = max(
        [float(record["upper_bound"]) for record in records[1:]] + [0.0])
    margin = float(top["lower_bound"]) - next_upper
    decision = "safe_top" if margin > tol else "uncertain"
    return {
        "decision": decision,
        "top_pose_id": int(top["pose_id"]),
        "top_block_index": int(top["block_index"]),
        "top_lower_bound": float(top["lower_bound"]),
        "top_upper_bound": float(top["upper_bound"]),
        "next_upper_bound": float(next_upper),
        "margin": float(margin),
        "candidate_bounds": records,
    }

  def apply_block(block: dict) -> dict:
    nonlocal payload_bytes
    block_index = int(block["block_index"])
    start = block_index * block_dim
    aggregate[start:start + block_dim] += block["residual_block"]
    dropped_bound_by_block[block_index] -= float(block["block_norm"])
    dropped_bound_by_block[block_index] = max(
        0.0, float(dropped_bound_by_block[block_index]))
    selected = {**block, "payload_bytes": block_payload_bytes}
    selected_blocks.append(selected)
    payload_bytes += block_payload_bytes
    return selected

  def head_key(robot: object, block: dict):
    return (
        -float(block["block_norm"]),
        str(robot),
        int(block["pose_id"]),
        int(block["block_index"]),
    )

  decision = make_decision()
  pull_count = 0
  while decision["decision"] == "uncertain":
    active_heads = [
        (head_key(robot, queue[0]), robot)
        for robot, queue in queues.items() if queue
    ]
    if not active_heads:
      break
    _, selected_robot = min(active_heads, key=lambda item: item[0])
    selected = apply_block(queues[selected_robot].pop(0))
    pull_count += 1
    decision = make_decision()
    pull_trace.append({
        "pull": int(pull_count),
        "selected_block": selected,
        "selected_block_count": int(len(selected_blocks)),
        "payload_bytes": int(payload_bytes),
        "decision": decision,
    })

  full_residual = np.asarray(
      exact_sparse_report.get("full_aggregate_residual",
                              exact_sparse_report.get(
                                  "aggregate_residual",
                                  np.zeros(interface_variable_count))),
      dtype=float).reshape(-1)
  exact_records: list[dict] = []
  for block_index in range(block_count):
    pose_id = block_pose_ids.get(block_index)
    if pose_id is None or int(pose_id) in selected_pose_set:
      continue
    start = block_index * block_dim
    block = full_residual[start:start + block_dim]
    exact_records.append({
        "pose_id": int(pose_id),
        "block_index": int(block_index),
        "block_norm": float(np.linalg.norm(block)),
    })
  exact_records.sort(key=lambda item: (
      -float(item["block_norm"]),
      int(item["pose_id"]),
  ))
  exact_top_pose_id = (
      None if not exact_records else int(exact_records[0]["pose_id"]))
  return {
      "model": "receiver_pulled_top_residual_block_decision",
      "decision": decision,
      "exact_top_pose_id": exact_top_pose_id,
      "exact_top_block_norm": (
          0.0 if not exact_records else float(exact_records[0]["block_norm"])),
      "exact_block_ranking": exact_records,
      "selected_blocks": selected_blocks,
      "selected_block_count": int(len(selected_blocks)),
      "pull_count": int(pull_count),
      "payload_bytes": int(payload_bytes),
      "payload_bytes_per_block": block_payload_bytes,
      "candidate_block_count": candidate_block_count,
      "pull_trace": pull_trace,
      "final_aggregate_residual": aggregate,
      "final_dropped_bound_by_block": dropped_bound_by_block,
  }


def pose_block_basin_certificate_promotion_basis(
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    omitted_force_blocks: Iterable[dict],
    full_schur: np.ndarray,
    full_rhs: np.ndarray,
    base_basis: np.ndarray,
    min_value_per_byte: float,
    max_promoted_pose_blocks: int,
    payload_bytes_per_coordinate: int = 8,
    max_omitted_force_increase: float = 0.0,
    anchor_pose: int | None = None,
    tolerance: float = 1e-12,
) -> dict:
  """Promote pose blocks only when Schur merit and handoff residual agree."""
  full_schur = np.asarray(full_schur, dtype=float)
  full_rhs = np.asarray(full_rhs, dtype=float).reshape(-1)
  row_count = len(full_rhs)
  current_basis = _orthonormalize_basis_columns(
      np.asarray(base_basis, dtype=float),
      row_count=row_count,
      tolerance=tolerance,
  )
  current_eval = _basis_quadratic_merit(full_schur, full_rhs, current_basis)
  candidate_blocks = [
      block for block in omitted_force_blocks
      if (not bool(block.get("selected", False)) and
          float(block.get("block_norm", 0.0)) > 0.0)
  ]
  candidate_blocks.sort(key=lambda item: (
      -float(item.get("block_norm", 0.0)),
      int(item["pose_id"]),
  ))
  remaining = {int(block["pose_id"]): dict(block) for block in candidate_blocks}
  accepted: list[dict] = []
  rejected: list[dict] = []
  rejected_pose_ids: set[int] = set()
  promoted_pose_ids: list[int] = []
  max_blocks = max(0, int(max_promoted_pose_blocks))
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))
  threshold = float(min_value_per_byte)
  omitted_increase_limit = max(0.0, float(max_omitted_force_increase))
  tol = float(tolerance)

  def evaluate_pose(pose_id: int) -> dict:
    candidate_basis_report = pose_block_promotion_basis(
        interface_pose_ids=interface_pose_ids,
        interface_variable_count=interface_variable_count,
        block_dim=block_dim,
        omitted_force_blocks=[{
            "pose_id": int(pose_id),
            "selected": False,
            "block_norm": float(remaining[pose_id].get("block_norm", 0.0)),
        }],
        max_promoted_pose_blocks=1,
        anchor_pose=anchor_pose,
    )
    combined_basis, added_columns = (
        _append_independent_columns_to_orthonormal_basis(
            current_basis,
            candidate_basis_report["basis"],
            row_count=row_count,
            tolerance=tol,
        )
    )
    if added_columns <= 0:
      return {
          "pose_id": int(pose_id),
          "reason": "dependent",
          "incremental_merit": 0.0,
          "payload_bytes": 0,
          "value_per_byte": 0.0,
          "omitted_force_before": float(current_eval["residual_norm"]),
          "omitted_force_after": float(current_eval["residual_norm"]),
          "omitted_force_delta": 0.0,
          "combined_basis": current_basis,
          "combined_eval": current_eval,
          "candidate_basis_column_count": 0,
      }
    combined_eval = _basis_quadratic_merit(full_schur, full_rhs, combined_basis)
    incremental_merit = float(combined_eval["merit"] - current_eval["merit"])
    payload_bytes = int(added_columns * payload_per_coord)
    value_per_byte = (
        incremental_merit / float(payload_bytes)
        if payload_bytes > 0 else 0.0)
    omitted_force_before = float(current_eval["residual_norm"])
    omitted_force_after = float(combined_eval["residual_norm"])
    return {
        "pose_id": int(pose_id),
        "reason": "candidate",
        "incremental_merit": incremental_merit,
        "payload_bytes": payload_bytes,
        "value_per_byte": value_per_byte,
        "omitted_force_before": omitted_force_before,
        "omitted_force_after": omitted_force_after,
        "omitted_force_delta": omitted_force_after - omitted_force_before,
        "combined_basis": combined_basis,
        "combined_eval": combined_eval,
        "candidate_basis_column_count": int(candidate_basis_report[
            "basis_column_count"]),
    }

  def rejection_reason(item: dict) -> str:
    if item["reason"] == "dependent":
      return "dependent"
    if float(item["incremental_merit"]) <= tol:
      return "nonpositive_merit"
    if float(item["value_per_byte"]) < threshold:
      return "below_value_threshold"
    if float(item["omitted_force_delta"]) > omitted_increase_limit + tol:
      return "omitted_force_worse"
    return "candidate_not_selected"

  while remaining and len(accepted) < max_blocks:
    evaluations = [evaluate_pose(pose_id) for pose_id in sorted(remaining)]
    feasible = [
        item for item in evaluations
        if rejection_reason(item) == "candidate_not_selected"
    ]
    if not feasible:
      for item in evaluations:
        rejected.append({
            "pose_id": int(item["pose_id"]),
            "reason": rejection_reason(item),
            "incremental_merit": float(item["incremental_merit"]),
            "payload_bytes": int(item["payload_bytes"]),
            "value_per_byte": float(item["value_per_byte"]),
            "omitted_force_before": float(item["omitted_force_before"]),
            "omitted_force_after": float(item["omitted_force_after"]),
            "omitted_force_delta": float(item["omitted_force_delta"]),
        })
        rejected_pose_ids.add(int(item["pose_id"]))
      remaining.clear()
      break
    feasible.sort(key=lambda item: (
        -float(item["value_per_byte"]),
        float(item["omitted_force_delta"]),
        -float(item["incremental_merit"]),
        int(item["pose_id"]),
    ))
    best = feasible[0]
    for item in evaluations:
      if int(item["pose_id"]) == int(best["pose_id"]):
        continue
      reason = rejection_reason(item)
      if reason == "candidate_not_selected":
        continue
      rejected.append({
          "pose_id": int(item["pose_id"]),
          "reason": reason,
          "incremental_merit": float(item["incremental_merit"]),
          "payload_bytes": int(item["payload_bytes"]),
          "value_per_byte": float(item["value_per_byte"]),
          "omitted_force_before": float(item["omitted_force_before"]),
          "omitted_force_after": float(item["omitted_force_after"]),
          "omitted_force_delta": float(item["omitted_force_delta"]),
      })
      rejected_pose_ids.add(int(item["pose_id"]))
    current_basis = np.asarray(best["combined_basis"], dtype=float)
    current_eval = dict(best["combined_eval"])
    promoted_pose_ids.append(int(best["pose_id"]))
    accepted.append({
        "pose_id": int(best["pose_id"]),
        "incremental_merit": float(best["incremental_merit"]),
        "payload_bytes": int(best["payload_bytes"]),
        "value_per_byte": float(best["value_per_byte"]),
        "omitted_force_before": float(best["omitted_force_before"]),
        "omitted_force_after": float(best["omitted_force_after"]),
        "omitted_force_delta": float(best["omitted_force_delta"]),
        "residual_norm_after_accept": float(current_eval["residual_norm"]),
    })
    remaining.pop(int(best["pose_id"]), None)

  for pose_id in sorted(remaining):
    if int(pose_id) in rejected_pose_ids:
      continue
    item = evaluate_pose(pose_id)
    rejected.append({
        "pose_id": int(pose_id),
        "reason": "budget_exhausted",
        "incremental_merit": float(item["incremental_merit"]),
        "payload_bytes": int(item["payload_bytes"]),
        "value_per_byte": float(item["value_per_byte"]),
        "omitted_force_before": float(item["omitted_force_before"]),
        "omitted_force_after": float(item["omitted_force_after"]),
        "omitted_force_delta": float(item["omitted_force_delta"]),
    })

  return {
      "model": "pose_block_basin_certificate_promotion_basis",
      "basis": current_basis,
      "basis_row_count": int(current_basis.shape[0]),
      "basis_column_count": int(current_basis.shape[1]),
      "block_dim": int(block_dim),
      "promoted_pose_ids": promoted_pose_ids,
      "accepted_promotions": accepted,
      "rejected_promotions": rejected,
      "candidate_pose_ids": [int(block["pose_id"]) for block in candidate_blocks],
      "min_value_per_byte": threshold,
      "max_omitted_force_increase": omitted_increase_limit,
      "max_promoted_pose_blocks": max_blocks,
      "payload_bytes_per_coordinate": payload_per_coord,
      "base_basis_column_count": int(
          _orthonormalize_basis_columns(
              np.asarray(base_basis, dtype=float),
              row_count=row_count,
              tolerance=tolerance,
          ).shape[1]),
      "final_omitted_force_norm": float(current_eval["residual_norm"]),
      "final_merit": float(current_eval["merit"]),
      "final_singular": bool(current_eval["singular"]),
  }


def pose_block_topk_basin_certificate_promotion_basis(
    interface_pose_ids: Iterable[int],
    interface_variable_count: int,
    block_dim: int,
    omitted_force_blocks: Iterable[dict],
    full_schur: np.ndarray,
    full_rhs: np.ndarray,
    base_basis: np.ndarray,
    candidate_top_k: int,
    payload_bytes_per_coordinate: int = 8,
    max_omitted_force_increase: float = 0.0,
    anchor_pose: int | None = None,
    tolerance: float = 1e-12,
) -> dict:
  """Promote a coupled top-k residual block set under a basin certificate."""
  full_schur = np.asarray(full_schur, dtype=float)
  full_rhs = np.asarray(full_rhs, dtype=float).reshape(-1)
  row_count = len(full_rhs)
  current_basis = _orthonormalize_basis_columns(
      np.asarray(base_basis, dtype=float),
      row_count=row_count,
      tolerance=tolerance,
  )
  current_eval = _basis_quadratic_merit(full_schur, full_rhs, current_basis)
  max_candidates = max(1, int(candidate_top_k))
  payload_per_coord = max(1, int(payload_bytes_per_coordinate))
  omitted_increase_limit = max(0.0, float(max_omitted_force_increase))
  tol = float(tolerance)
  candidate_blocks = [
      dict(block) for block in omitted_force_blocks
      if (not bool(block.get("selected", False)) and
          float(block.get("block_norm", 0.0)) > 0.0)
  ]
  candidate_blocks.sort(key=lambda item: (
      -float(item.get("block_norm", 0.0)),
      int(item["pose_id"]),
  ))
  selected_blocks = candidate_blocks[:max_candidates]
  if not selected_blocks:
    return {
        "model": "pose_block_topk_basin_certificate_promotion_basis",
        "basis": current_basis,
        "basis_row_count": int(current_basis.shape[0]),
        "basis_column_count": int(current_basis.shape[1]),
        "block_dim": int(block_dim),
        "promoted_pose_ids": [],
        "candidate_pose_ids": [],
        "accepted_promotions": [],
        "rejected_promotions": [],
        "candidate_top_k": max_candidates,
        "final_omitted_force_norm": float(current_eval["residual_norm"]),
        "final_merit": float(current_eval["merit"]),
        "final_singular": bool(current_eval["singular"]),
    }

  candidate_basis_report = pose_block_promotion_basis(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=interface_variable_count,
      block_dim=block_dim,
      omitted_force_blocks=selected_blocks,
      max_promoted_pose_blocks=len(selected_blocks),
      anchor_pose=anchor_pose,
  )
  combined_basis, added_columns = _append_independent_columns_to_orthonormal_basis(
      current_basis,
      candidate_basis_report["basis"],
      row_count=row_count,
      tolerance=tol,
  )
  payload_bytes = int(max(0, added_columns) * payload_per_coord)
  if added_columns <= 0:
    return {
        "model": "pose_block_topk_basin_certificate_promotion_basis",
        "basis": current_basis,
        "basis_row_count": int(current_basis.shape[0]),
        "basis_column_count": int(current_basis.shape[1]),
        "block_dim": int(block_dim),
        "promoted_pose_ids": [],
        "candidate_pose_ids": [int(block["pose_id"]) for block in selected_blocks],
        "accepted_promotions": [],
        "rejected_promotions": [{
            "pose_ids": [int(block["pose_id"]) for block in selected_blocks],
            "reason": "dependent",
            "incremental_merit": 0.0,
            "payload_bytes": 0,
            "omitted_force_before": float(current_eval["residual_norm"]),
            "omitted_force_after": float(current_eval["residual_norm"]),
            "omitted_force_delta": 0.0,
        }],
        "candidate_top_k": max_candidates,
        "final_omitted_force_norm": float(current_eval["residual_norm"]),
        "final_merit": float(current_eval["merit"]),
        "final_singular": bool(current_eval["singular"]),
    }

  combined_eval = _basis_quadratic_merit(full_schur, full_rhs, combined_basis)
  incremental_merit = float(combined_eval["merit"] - current_eval["merit"])
  omitted_force_before = float(current_eval["residual_norm"])
  omitted_force_after = float(combined_eval["residual_norm"])
  omitted_force_delta = omitted_force_after - omitted_force_before
  selected_pose_ids = [
      int(pose_id) for pose_id in candidate_basis_report["promoted_pose_ids"]
  ]
  accepted = (
      incremental_merit > tol and
      omitted_force_delta <= omitted_increase_limit + tol)
  promotion_record = {
      "pose_ids": selected_pose_ids,
      "incremental_merit": float(incremental_merit),
      "payload_bytes": int(payload_bytes),
      "omitted_force_before": float(omitted_force_before),
      "omitted_force_after": float(omitted_force_after),
      "omitted_force_delta": float(omitted_force_delta),
  }
  if accepted:
    final_basis = combined_basis
    final_eval = combined_eval
    promoted_pose_ids = selected_pose_ids
    accepted_promotions = [promotion_record]
    rejected_promotions: list[dict] = []
  else:
    reason = (
        "nonpositive_merit" if incremental_merit <= tol
        else "omitted_force_worse")
    final_basis = current_basis
    final_eval = current_eval
    promoted_pose_ids = []
    accepted_promotions = []
    rejected_promotions = [{**promotion_record, "reason": reason}]

  return {
      "model": "pose_block_topk_basin_certificate_promotion_basis",
      "basis": final_basis,
      "basis_row_count": int(final_basis.shape[0]),
      "basis_column_count": int(final_basis.shape[1]),
      "block_dim": int(block_dim),
      "promoted_pose_ids": promoted_pose_ids,
      "candidate_pose_ids": [int(block["pose_id"]) for block in selected_blocks],
      "accepted_promotions": accepted_promotions,
      "rejected_promotions": rejected_promotions,
      "candidate_top_k": max_candidates,
      "payload_bytes_per_coordinate": payload_per_coord,
      "max_omitted_force_increase": omitted_increase_limit,
      "base_basis_column_count": int(current_basis.shape[1]),
      "final_omitted_force_norm": float(final_eval["residual_norm"]),
      "final_merit": float(final_eval["merit"]),
      "final_singular": bool(final_eval["singular"]),
  }


def project_local_interface_schur_with_basis(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    basis: np.ndarray,
    basis_metadata: dict | None = None,
    variable_count: int | None = None,
    damping: float = 0.0,
) -> dict:
  """Project the exact local-Schur interface system onto a supplied basis."""
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  full_schur, full_rhs, schur_stats = sum_local_interface_schur_contributions(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping,
  )
  basis_array = np.asarray(basis, dtype=float)
  if basis_array.ndim != 2:
    raise ValueError("skeleton basis must be a two-dimensional array")
  if basis_array.shape[0] != len(interface_indices):
    raise ValueError("skeleton basis row count must match interface variables")
  reduced_schur = basis_array.T @ full_schur @ basis_array
  reduced_schur = 0.5 * (reduced_schur + reduced_schur.T)
  reduced_rhs = basis_array.T @ full_rhs
  metadata = {} if basis_metadata is None else dict(basis_metadata)
  metadata.pop("basis", None)
  return {
      "model": "basis_reduced_interface_schur",
      "basis_model": str(metadata.get("model", "custom_basis")),
      "basis": basis_array,
      "basis_column_count": int(basis_array.shape[1]),
      "basis_row_count": int(basis_array.shape[0]),
      "basis_metadata": metadata,
      "reduced_schur": reduced_schur,
      "reduced_rhs": reduced_rhs,
      "full_schur": full_schur,
      "full_rhs": full_rhs,
      "full_schur_shape": [int(full_schur.shape[0]), int(full_schur.shape[1])],
      "schur_stats": schur_stats,
  }


def rank_aware_structural_span_basis(
    candidate_basis_reports: Iterable[dict],
    full_schur: np.ndarray,
    full_rhs: np.ndarray,
    tolerance: float = 1e-12,
) -> dict:
  """Select an independent structural span by incremental Schur merit.

  Candidate reports provide physically meaningful columns. The returned basis
  is an orthonormal basis for the accepted span, so it is rank-safe even when
  structural candidates are redundant.
  """
  full_schur = np.asarray(full_schur, dtype=float)
  full_schur = 0.5 * (full_schur + full_schur.T)
  full_rhs = np.asarray(full_rhs, dtype=float).reshape(-1)
  if full_schur.ndim != 2 or full_schur.shape[0] != full_schur.shape[1]:
    raise ValueError("full_schur must be a square matrix")
  if full_schur.shape[0] != len(full_rhs):
    raise ValueError("full_schur and full_rhs dimension mismatch")

  candidates: list[dict] = []
  for report_index, report in enumerate(candidate_basis_reports):
    basis = np.asarray(report.get("basis"), dtype=float)
    if basis.size == 0:
      continue
    if basis.ndim == 1:
      basis = basis.reshape(-1, 1)
    if basis.ndim != 2 or basis.shape[0] != len(full_rhs):
      raise ValueError("candidate basis row count must match full_rhs")
    source_model = str(report.get("model", f"candidate_{report_index}"))
    for column in range(basis.shape[1]):
      candidates.append({
          "candidate_index": int(report_index),
          "source_model": source_model,
          "source_column": int(column),
          "vector": basis[:, column].copy(),
      })

  q_columns: list[np.ndarray] = []
  accepted: list[dict] = []
  rejected: list[dict] = []
  active = set(range(len(candidates)))
  scale = max(
      1.0,
      float(np.linalg.norm(full_schur, ord=np.inf)) if full_schur.size else 1.0,
      float(np.linalg.norm(full_rhs)),
  )
  tol = float(tolerance)

  def orthogonal_residual(vector: np.ndarray) -> np.ndarray:
    residual = np.asarray(vector, dtype=float).reshape(-1).copy()
    for q_col in q_columns:
      residual -= float(q_col @ residual) * q_col
    return residual

  def solve_current_basis() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    if not q_columns:
      lifted = np.zeros_like(full_rhs)
      return np.zeros(0, dtype=float), lifted, full_rhs.copy(), False
    basis = np.column_stack(q_columns)
    reduced = basis.T @ full_schur @ basis
    reduced = 0.5 * (reduced + reduced.T)
    rhs = basis.T @ full_rhs
    try:
      coeff = np.linalg.solve(reduced, rhs)
      singular = False
    except np.linalg.LinAlgError:
      coeff = np.linalg.pinv(reduced, rcond=1e-12) @ rhs
      singular = True
    lifted = basis @ coeff
    residual = full_rhs - full_schur @ lifted
    return coeff, lifted, residual, singular

  _, _, residual, final_singular = solve_current_basis()
  while active:
    best: dict | None = None
    dependent_now: list[int] = []
    for index in sorted(active):
      candidate = candidates[index]
      direction = orthogonal_residual(candidate["vector"])
      direction_norm = float(np.linalg.norm(direction))
      if direction_norm <= tol * scale:
        dependent_now.append(index)
        continue
      direction /= direction_norm
      stiffness = float(direction @ full_schur @ direction)
      if stiffness <= tol * scale:
        merit = 0.0
      else:
        numerator = float(direction @ residual)
        merit = 0.5 * numerator * numerator / stiffness
      record = {
          "candidate_index": int(candidate["candidate_index"]),
          "source_model": candidate["source_model"],
          "source_column": int(candidate["source_column"]),
          "candidate_column": int(index),
          "direction_norm": direction_norm,
          "stiffness": stiffness,
          "incremental_merit": float(merit),
          "direction": direction,
      }
      if best is None or (
          record["incremental_merit"] > best["incremental_merit"] + tol):
        best = record

    for index in dependent_now:
      if index in active:
        candidate = candidates[index]
        rejected.append({
            "candidate_index": int(candidate["candidate_index"]),
            "source_model": candidate["source_model"],
            "source_column": int(candidate["source_column"]),
            "candidate_column": int(index),
            "reason": "dependent",
            "incremental_merit": 0.0,
        })
        active.remove(index)

    if best is None or best["incremental_merit"] <= tol * scale:
      break

    active.remove(int(best["candidate_column"]))
    q_columns.append(np.asarray(best["direction"], dtype=float).reshape(-1))
    _, lifted, residual, final_singular = solve_current_basis()
    accepted.append({
        "candidate_index": int(best["candidate_index"]),
        "source_model": best["source_model"],
        "source_column": int(best["source_column"]),
        "candidate_column": int(best["candidate_column"]),
        "stiffness": float(best["stiffness"]),
        "incremental_merit": float(best["incremental_merit"]),
        "residual_norm_after_accept": float(np.linalg.norm(residual)),
    })

  for index in sorted(active):
    candidate = candidates[index]
    direction = orthogonal_residual(candidate["vector"])
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= tol * scale:
      reason = "dependent"
      stiffness = 0.0
      merit = 0.0
    else:
      direction /= direction_norm
      stiffness = float(direction @ full_schur @ direction)
      if stiffness <= tol * scale:
        reason = "nonpositive_stiffness"
        merit = 0.0
      else:
        numerator = float(direction @ residual)
        merit = 0.5 * numerator * numerator / stiffness
        reason = (
            "zero_incremental_merit"
            if merit <= tol * scale else "not_selected")
    rejected.append({
        "candidate_index": int(candidate["candidate_index"]),
        "source_model": candidate["source_model"],
        "source_column": int(candidate["source_column"]),
        "candidate_column": int(index),
        "reason": reason,
        "stiffness": float(stiffness),
        "incremental_merit": float(merit),
    })

  if q_columns:
    basis = np.column_stack(q_columns)
  else:
    basis = np.zeros((len(full_rhs), 0), dtype=float)
  coeff, lifted, final_residual, final_singular = solve_current_basis()
  return {
      "model": "rank_aware_structural_span_basis",
      "basis_model": "rank_aware_structural_span_basis",
      "basis": basis,
      "basis_row_count": int(basis.shape[0]),
      "basis_column_count": int(basis.shape[1]),
      "candidate_column_count": int(len(candidates)),
      "accepted_column_count": int(len(accepted)),
      "rejected_column_count": int(len(rejected)),
      "accepted_columns": accepted,
      "rejected_columns": rejected,
      "final_reduced_solution": coeff,
      "final_lifted_solution": lifted,
      "final_omitted_force": final_residual,
      "final_omitted_force_norm": float(np.linalg.norm(final_residual)),
      "reduced_singular": bool(final_singular),
      "selection_tolerance": tol,
  }


def rank_omitted_force_blocks(
    omitted_force: np.ndarray,
    interface_pose_ids: Iterable[int],
    block_dim: int,
    selected_interface_pose_ids: Iterable[int] = (),
    anchor_pose: int | None = None,
) -> list[dict]:
  """Rank interface pose blocks by omitted KKT force.

  Missing blocks with large force are the first structural enrichment
  candidates. Selected blocks are retained in the report for diagnostics but
  rank behind unselected blocks.
  """
  omitted_force = np.asarray(omitted_force, dtype=float).reshape(-1)
  variable_pose_ids = _variable_interface_pose_ids(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=len(omitted_force),
      block_dim=block_dim,
      anchor_pose=anchor_pose,
  )
  selected_set = {int(pose_id) for pose_id in selected_interface_pose_ids}
  rows: list[dict] = []
  for block_index, pose_id in enumerate(variable_pose_ids):
    start = block_index * int(block_dim)
    block = omitted_force[start:start + int(block_dim)]
    block_norm = float(np.linalg.norm(block))
    selected = int(pose_id) in selected_set
    rows.append({
        "pose_id": int(pose_id),
        "block_index": int(block_index),
        "selected": bool(selected),
        "block_norm": block_norm,
        "block_force": block.tolist(),
    })
  rows.sort(key=lambda item: (
      item["selected"],
      -float(item["block_norm"]),
      int(item["pose_id"]),
  ))
  for rank, item in enumerate(rows):
    item["rank"] = int(rank)
  return rows


def enrich_skeleton_by_omitted_force(
    skeleton_pose_ids: Iterable[int],
    omitted_force_blocks: Iterable[dict],
    max_add: int = 1,
) -> dict:
  """Add the largest missing omitted-force blocks to a skeleton pose set."""
  current = sorted({int(pose_id) for pose_id in skeleton_pose_ids})
  current_set = set(current)
  add_budget = max(0, int(max_add))
  candidates = [
      block for block in omitted_force_blocks
      if (not bool(block.get("selected", False)) and
          int(block["pose_id"]) not in current_set and
          float(block.get("block_norm", 0.0)) > 0.0)
  ]
  candidates.sort(key=lambda item: (
      -float(item.get("block_norm", 0.0)),
      int(item["pose_id"]),
  ))
  added = [int(item["pose_id"]) for item in candidates[:add_budget]]
  return {
      "model": "basin_skeleton_omitted_force_enrichment",
      "reason": "omitted_force",
      "input_skeleton_pose_ids": current,
      "added_pose_ids": added,
      "skeleton_pose_ids": sorted(current_set | set(added)),
      "candidate_count": int(len(candidates)),
      "max_add": int(add_budget),
  }


def run_omitted_force_enrichment_iterations(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_pose_ids: Iterable[int],
    block_dim: int,
    initial_skeleton_pose_ids: Iterable[int],
    max_iterations: int = 3,
    max_add_per_iteration: int = 1,
    stop_omitted_force_norm: float = 1e-12,
    variable_count: int | None = None,
    damping: float = 0.0,
    anchor_pose: int | None = None,
) -> dict:
  """Run a deterministic solve-rank-enrich diagnostic loop."""
  current = sorted({int(pose_id) for pose_id in initial_skeleton_pose_ids})
  history: list[dict] = []
  final_solve: dict | None = None
  converged = False
  stopped_reason = "max_iterations"
  iteration_count = max(0, int(max_iterations))

  for iteration in range(iteration_count):
    solve = solve_skeleton_reduced_interface_schur(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=interface_pose_ids,
        block_dim=block_dim,
        skeleton_pose_ids=current,
        variable_count=variable_count,
        damping=damping,
        anchor_pose=anchor_pose,
    )
    final_solve = solve
    top_block = solve["omitted_force_blocks"][0] if solve["omitted_force_blocks"] else None
    if float(solve["omitted_force_norm"]) <= float(stop_omitted_force_norm):
      converged = True
      stopped_reason = "omitted_force_tolerance"
      history.append({
          "iteration": int(iteration),
          "skeleton_pose_ids": current,
          "selected_interface_pose_ids": solve["selected_interface_pose_ids"],
          "omitted_force_norm": float(solve["omitted_force_norm"]),
          "interface_solution_error_norm": float(
              solve["interface_solution_error_norm"]),
          "top_omitted_force_pose_id": None if top_block is None else int(top_block["pose_id"]),
          "top_omitted_force_norm": 0.0 if top_block is None else float(top_block["block_norm"]),
          "added_pose_ids": [],
      })
      break

    enriched = enrich_skeleton_by_omitted_force(
        skeleton_pose_ids=current,
        omitted_force_blocks=solve["omitted_force_blocks"],
        max_add=max_add_per_iteration,
    )
    added = list(enriched["added_pose_ids"])
    history.append({
        "iteration": int(iteration),
        "skeleton_pose_ids": current,
        "selected_interface_pose_ids": solve["selected_interface_pose_ids"],
        "omitted_force_norm": float(solve["omitted_force_norm"]),
        "interface_solution_error_norm": float(
            solve["interface_solution_error_norm"]),
        "top_omitted_force_pose_id": None if top_block is None else int(top_block["pose_id"]),
        "top_omitted_force_norm": 0.0 if top_block is None else float(top_block["block_norm"]),
        "added_pose_ids": added,
    })
    if not added:
      stopped_reason = "no_enrichment_candidate"
      break
    current = list(enriched["skeleton_pose_ids"])

  if final_solve is None:
    final_solve = solve_skeleton_reduced_interface_schur(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=interface_pose_ids,
        block_dim=block_dim,
        skeleton_pose_ids=current,
        variable_count=variable_count,
        damping=damping,
        anchor_pose=anchor_pose,
    )
    converged = (
        float(final_solve["omitted_force_norm"]) <=
        float(stop_omitted_force_norm))
    stopped_reason = (
        "omitted_force_tolerance" if converged else "max_iterations")

  return {
      "model": "basin_skeleton_omitted_force_enrichment_iterations",
      "initial_skeleton_pose_ids": sorted(
          {int(pose_id) for pose_id in initial_skeleton_pose_ids}),
      "final_skeleton_pose_ids": current,
      "history": history,
      "iteration_count": int(len(history)),
      "max_iterations": int(iteration_count),
      "max_add_per_iteration": int(max_add_per_iteration),
      "stop_omitted_force_norm": float(stop_omitted_force_norm),
      "converged": bool(converged),
      "stopped_reason": stopped_reason,
      "final_solve": final_solve,
  }


def run_graph_stage_omitted_force_enrichment(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    initial_policy: str = "bridge_articulation",
    max_iterations: int = 3,
    max_add_per_iteration: int = 1,
    stop_omitted_force_norm: float = 1e-12,
) -> dict:
  """Build graph-local Schur systems and run skeleton enrichment."""
  pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  structural = build_basin_skeleton(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
  )
  initial_skeleton = select_initial_skeleton_pose_ids(
      structural, policy=initial_policy)
  local_systems, interface_indices, build_stats = (
      build_graph_local_normal_systems_for_interface_schur(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          stage=stage,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          rotations=rotations,
          edge_owner_policy=edge_owner_policy,
          hessian_storage=hessian_storage,
      )
  )
  variable_interface_pose_ids = _variable_interface_pose_ids(
      interface_pose_ids=build_stats["interface_pose_ids"],
      interface_variable_count=len(interface_indices),
      block_dim=build_stats["block_dim"],
      anchor_pose=anchor_pose,
  )
  variable_initial = [
      pose_id for pose_id in initial_skeleton
      if pose_id in set(variable_interface_pose_ids)
  ]
  enrichment = run_omitted_force_enrichment_iterations(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_pose_ids=build_stats["interface_pose_ids"],
      block_dim=build_stats["block_dim"],
      initial_skeleton_pose_ids=initial_skeleton,
      max_iterations=max_iterations,
      max_add_per_iteration=max_add_per_iteration,
      stop_omitted_force_norm=stop_omitted_force_norm,
      anchor_pose=anchor_pose,
  )
  return {
      "model": "graph_stage_basin_skeleton_omitted_force_enrichment",
      "stage": stage,
      "initial_policy": initial_policy,
      "structural_skeleton": structural,
      "initial_skeleton_pose_ids": initial_skeleton,
      "variable_initial_skeleton_pose_ids": variable_initial,
      "full_interface_pose_ids": list(build_stats["interface_pose_ids"]),
      "variable_interface_pose_ids": variable_interface_pose_ids,
      "full_interface_pose_count": int(len(build_stats["interface_pose_ids"])),
      "variable_interface_pose_count": int(len(variable_interface_pose_ids)),
      "initial_variable_fraction": (
          float(len(variable_initial)) / float(len(variable_interface_pose_ids))
          if variable_interface_pose_ids else 0.0),
      "build_stats": build_stats,
      "enrichment": enrichment,
  }


def run_graph_stage_primal_skeleton_basis_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promoted_pose_block_count: int = 0,
    promotion_min_value_per_byte: float = 0.0,
    promotion_min_relative_value_fraction: float | None = None,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_payload_bytes_per_coordinate: int = 8,
) -> dict:
  """Build and solve one graph-stage primal-skeleton basis diagnostic."""
  pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  structural = build_basin_skeleton(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
  )
  local_systems, interface_indices, build_stats = (
      build_graph_local_normal_systems_for_interface_schur(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          stage=stage,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          rotations=rotations,
          edge_owner_policy=edge_owner_policy,
          hessian_storage=hessian_storage,
      )
  )
  promotion_report: dict | None = None
  if basis_mode == "robot_coordinate":
    basis_report = robot_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        anchor_pose=anchor_pose,
    )
  elif basis_mode == "cycle_coordinate":
    basis_report = cycle_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        cycle_basis=structural["cycle_basis"],
        anchor_pose=anchor_pose,
    )
  elif basis_mode == "rank_aware_robot_cycle_span":
    robot_basis_report = robot_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        anchor_pose=anchor_pose,
    )
    cycle_basis_report = cycle_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        cycle_basis=structural["cycle_basis"],
        anchor_pose=anchor_pose,
    )
    full_schur, full_rhs, _ = sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )
    basis_report = rank_aware_structural_span_basis(
        candidate_basis_reports=[robot_basis_report, cycle_basis_report],
        full_schur=full_schur,
        full_rhs=full_rhs,
    )
  elif basis_mode == "rank_aware_robot_cycle_span_plus_pose_promote":
    robot_basis_report = robot_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        anchor_pose=anchor_pose,
    )
    cycle_basis_report = cycle_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        cycle_basis=structural["cycle_basis"],
        anchor_pose=anchor_pose,
    )
    full_schur, full_rhs, _ = sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )
    structural_basis_report = rank_aware_structural_span_basis(
        candidate_basis_reports=[robot_basis_report, cycle_basis_report],
        full_schur=full_schur,
        full_rhs=full_rhs,
    )
    structural_solve = solve_skeleton_reduced_interface_schur_with_basis(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        basis=structural_basis_report["basis"],
        basis_metadata=structural_basis_report,
        anchor_pose=anchor_pose,
    )
    promotion_basis_report = pose_block_promotion_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        omitted_force_blocks=structural_solve["omitted_force_blocks"],
        max_promoted_pose_blocks=promoted_pose_block_count,
        anchor_pose=anchor_pose,
    )
    basis_report = rank_aware_structural_span_basis(
        candidate_basis_reports=[structural_basis_report, promotion_basis_report],
        full_schur=full_schur,
        full_rhs=full_rhs,
    )
    promotion_report = {
        "base_basis_model": structural_basis_report["model"],
        "base_basis_column_count": int(structural_basis_report["basis_column_count"]),
        "base_omitted_force_norm": float(structural_solve["omitted_force_norm"]),
        "base_interface_solution_error_norm": float(
            structural_solve["interface_solution_error_norm"]),
        "promoted_pose_ids": promotion_basis_report["promoted_pose_ids"],
        "promotion_basis_column_count": int(
            promotion_basis_report["basis_column_count"]),
        "max_promoted_pose_blocks": int(promoted_pose_block_count),
    }
  elif basis_mode == "rank_aware_robot_cycle_span_plus_value_promote":
    robot_basis_report = robot_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        anchor_pose=anchor_pose,
    )
    cycle_basis_report = cycle_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        cycle_basis=structural["cycle_basis"],
        anchor_pose=anchor_pose,
    )
    full_schur, full_rhs, _ = sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )
    structural_basis_report = rank_aware_structural_span_basis(
        candidate_basis_reports=[robot_basis_report, cycle_basis_report],
        full_schur=full_schur,
        full_rhs=full_rhs,
    )
    structural_solve = solve_skeleton_reduced_interface_schur_with_basis(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        basis=structural_basis_report["basis"],
        basis_metadata=structural_basis_report,
        anchor_pose=anchor_pose,
    )
    basis_report = pose_block_value_per_byte_promotion_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        omitted_force_blocks=structural_solve["omitted_force_blocks"],
        full_schur=full_schur,
        full_rhs=full_rhs,
        base_basis=structural_basis_report["basis"],
        min_value_per_byte=promotion_min_value_per_byte,
        min_relative_value_fraction=promotion_min_relative_value_fraction,
        max_promoted_pose_blocks=promoted_pose_block_count,
        payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
        anchor_pose=anchor_pose,
    )
    promotion_report = {
        "base_basis_model": structural_basis_report["model"],
        "base_basis_column_count": int(structural_basis_report["basis_column_count"]),
        "base_omitted_force_norm": float(structural_solve["omitted_force_norm"]),
        "base_interface_solution_error_norm": float(
            structural_solve["interface_solution_error_norm"]),
        "promoted_pose_ids": basis_report["promoted_pose_ids"],
        "accepted_promotions": basis_report["accepted_promotions"],
        "rejected_promotions": basis_report["rejected_promotions"],
        "min_value_per_byte": float(promotion_min_value_per_byte),
        "min_relative_value_fraction": (
            None if promotion_min_relative_value_fraction is None
            else float(promotion_min_relative_value_fraction)),
        "first_value_per_byte": basis_report["first_value_per_byte"],
        "final_effective_value_threshold": float(
            basis_report["final_effective_value_threshold"]),
        "payload_bytes_per_coordinate": int(
            promotion_payload_bytes_per_coordinate),
        "max_promoted_pose_blocks": int(promoted_pose_block_count),
    }
  elif basis_mode == "rank_aware_robot_cycle_span_plus_basin_promote":
    robot_basis_report = robot_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        anchor_pose=anchor_pose,
    )
    cycle_basis_report = cycle_coordinate_skeleton_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        cycle_basis=structural["cycle_basis"],
        anchor_pose=anchor_pose,
    )
    full_schur, full_rhs, _ = sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )
    structural_basis_report = rank_aware_structural_span_basis(
        candidate_basis_reports=[robot_basis_report, cycle_basis_report],
        full_schur=full_schur,
        full_rhs=full_rhs,
    )
    structural_solve = solve_skeleton_reduced_interface_schur_with_basis(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=build_stats["interface_pose_ids"],
        block_dim=build_stats["block_dim"],
        basis=structural_basis_report["basis"],
        basis_metadata=structural_basis_report,
        anchor_pose=anchor_pose,
    )
    basis_report = pose_block_basin_certificate_promotion_basis(
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        omitted_force_blocks=structural_solve["omitted_force_blocks"],
        full_schur=full_schur,
        full_rhs=full_rhs,
        base_basis=structural_basis_report["basis"],
        min_value_per_byte=promotion_min_value_per_byte,
        max_omitted_force_increase=promotion_max_omitted_force_increase,
        max_promoted_pose_blocks=promoted_pose_block_count,
        payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
        anchor_pose=anchor_pose,
    )
    promotion_report = {
        "base_basis_model": structural_basis_report["model"],
        "base_basis_column_count": int(structural_basis_report["basis_column_count"]),
        "base_omitted_force_norm": float(structural_solve["omitted_force_norm"]),
        "base_interface_solution_error_norm": float(
            structural_solve["interface_solution_error_norm"]),
        "promoted_pose_ids": basis_report["promoted_pose_ids"],
        "accepted_promotions": basis_report["accepted_promotions"],
        "rejected_promotions": basis_report["rejected_promotions"],
        "min_value_per_byte": float(promotion_min_value_per_byte),
        "max_omitted_force_increase": float(promotion_max_omitted_force_increase),
        "payload_bytes_per_coordinate": int(
            promotion_payload_bytes_per_coordinate),
        "max_promoted_pose_blocks": int(promoted_pose_block_count),
    }
  else:
    raise ValueError(f"unsupported primal skeleton basis mode: {basis_mode}")

  solve = solve_skeleton_reduced_interface_schur_with_basis(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_pose_ids=build_stats["interface_pose_ids"],
      block_dim=build_stats["block_dim"],
      basis=basis_report["basis"],
      basis_metadata=basis_report,
      anchor_pose=anchor_pose,
  )
  full_count = int(len(interface_indices))
  column_count = int(basis_report["basis_column_count"])
  result = {
      "model": "graph_stage_primal_skeleton_basis_diagnostic",
      "stage": stage,
      "basis_mode": basis_mode,
      "basis_model": basis_report["model"],
      "basis_column_count": column_count,
      "basis_row_count": int(basis_report["basis_row_count"]),
      "full_interface_variable_count": full_count,
      "basis_compression_ratio": (
          float(column_count) / float(full_count) if full_count else 0.0),
      "structural_skeleton": structural,
      "basis": basis_report,
      "build_stats": build_stats,
      "solve": solve,
  }
  if promotion_report is not None:
    result["promotion"] = promotion_report
  return result


def run_graph_stage_residual_packet_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promoted_pose_block_count: int = 0,
    promotion_min_value_per_byte: float = 0.0,
    promotion_min_relative_value_fraction: float | None = None,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_payload_bytes_per_coordinate: int = 8,
) -> dict:
  """Compare graph-stage omitted force with distributed residual packets."""
  basis_diagnostic = run_graph_stage_primal_skeleton_basis_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      basis_mode=basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
  )
  pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  local_systems, interface_indices, build_stats = (
      build_graph_local_normal_systems_for_interface_schur(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          stage=stage,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          rotations=rotations,
          edge_owner_policy=edge_owner_policy,
          hessian_storage=hessian_storage,
      )
  )
  interface_state = np.asarray(
      basis_diagnostic["solve"]["lifted_interface_solution"],
      dtype=float).reshape(-1)
  packet_report = local_schur_residual_packets(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_state=interface_state,
      interface_pose_ids=_variable_interface_pose_ids(
          build_stats["interface_pose_ids"],
          len(interface_indices),
          build_stats["block_dim"],
          anchor_pose=anchor_pose),
      block_dim=build_stats["block_dim"],
      variable_count=build_stats["variable_count"],
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
  )
  omitted_force = np.asarray(
      basis_diagnostic["solve"]["omitted_force"], dtype=float).reshape(-1)
  packet_residual = np.asarray(
      packet_report["aggregate_residual"], dtype=float).reshape(-1)
  residual_error = packet_residual - omitted_force
  return {
      "model": "graph_stage_residual_packet_diagnostic",
      "stage": stage,
      "basis_mode": basis_mode,
      "basis_diagnostic": basis_diagnostic,
      "packet_report": packet_report,
      "residual_packet_error": residual_error,
      "residual_packet_error_norm": float(np.linalg.norm(residual_error)),
      "omitted_force_norm": float(np.linalg.norm(omitted_force)),
      "packet_aggregate_residual_norm": float(np.linalg.norm(packet_residual)),
  }


def run_graph_stage_sparse_residual_packet_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promoted_pose_block_count: int = 0,
    promotion_min_value_per_byte: float = 0.0,
    promotion_min_relative_value_fraction: float | None = None,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_top_k_blocks_per_robot: int | None = None,
    sparse_payload_index_bytes: int = 4,
) -> dict:
  """Compare omitted force with sparse distributed residual packets."""
  basis_diagnostic = run_graph_stage_primal_skeleton_basis_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      basis_mode=basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
  )
  pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  local_systems, interface_indices, build_stats = (
      build_graph_local_normal_systems_for_interface_schur(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          stage=stage,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          rotations=rotations,
          edge_owner_policy=edge_owner_policy,
          hessian_storage=hessian_storage,
      )
  )
  interface_state = np.asarray(
      basis_diagnostic["solve"]["lifted_interface_solution"],
      dtype=float).reshape(-1)
  sparse_report = local_schur_sparse_residual_packets(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_state=interface_state,
      interface_pose_ids=_variable_interface_pose_ids(
          build_stats["interface_pose_ids"],
          len(interface_indices),
          build_stats["block_dim"],
          anchor_pose=anchor_pose),
      block_dim=build_stats["block_dim"],
      variable_count=build_stats["variable_count"],
      block_norm_threshold=sparse_block_norm_threshold,
      top_k_blocks_per_robot=sparse_top_k_blocks_per_robot,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
  )
  omitted_force = np.asarray(
      basis_diagnostic["solve"]["omitted_force"], dtype=float).reshape(-1)
  sparse_residual = np.asarray(
      sparse_report["aggregate_residual"], dtype=float).reshape(-1)
  residual_error = sparse_residual - omitted_force
  return {
      "model": "graph_stage_sparse_residual_packet_diagnostic",
      "stage": stage,
      "basis_mode": basis_mode,
      "basis_diagnostic": basis_diagnostic,
      "sparse_packet_report": sparse_report,
      "full_packet_payload_bytes": int(sparse_report["dense_payload_bytes"]),
      "sparse_payload_bytes": int(sparse_report["payload_bytes"]),
      "sparse_payload_ratio": (
          float(sparse_report["payload_bytes"]) /
          float(sparse_report["dense_payload_bytes"])
          if int(sparse_report["dense_payload_bytes"]) > 0 else 0.0),
      "sparse_residual_error": residual_error,
      "sparse_residual_error_norm": float(np.linalg.norm(residual_error)),
      "omitted_force_norm": float(np.linalg.norm(omitted_force)),
      "sparse_aggregate_residual_norm": float(np.linalg.norm(sparse_residual)),
  }


def run_graph_stage_sparse_basin_decision_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    before_basis_mode: str,
    after_basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promoted_pose_block_count: int = 0,
    promotion_min_value_per_byte: float = 0.0,
    promotion_min_relative_value_fraction: float | None = None,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_top_k_blocks_per_robot: int | None = None,
    sparse_payload_index_bytes: int = 4,
) -> dict:
  """Certify a before/after basin decision with sparse residual packets."""
  before = run_graph_stage_sparse_residual_packet_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      basis_mode=before_basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      sparse_block_norm_threshold=sparse_block_norm_threshold,
      sparse_top_k_blocks_per_robot=sparse_top_k_blocks_per_robot,
      sparse_payload_index_bytes=sparse_payload_index_bytes,
  )
  after = run_graph_stage_sparse_residual_packet_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      basis_mode=after_basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      sparse_block_norm_threshold=sparse_block_norm_threshold,
      sparse_top_k_blocks_per_robot=sparse_top_k_blocks_per_robot,
      sparse_payload_index_bytes=sparse_payload_index_bytes,
  )
  decision = sparse_basin_gate_decision(
      before["sparse_packet_report"],
      after["sparse_packet_report"],
      max_omitted_force_increase=promotion_max_omitted_force_increase,
  )
  before_exact = float(before["omitted_force_norm"])
  after_exact = float(after["omitted_force_norm"])
  max_increase = max(0.0, float(promotion_max_omitted_force_increase))
  exact_decision = "accept" if after_exact <= before_exact + max_increase else "reject"
  return {
      "model": "graph_stage_sparse_basin_decision_diagnostic",
      "stage": stage,
      "before_basis_mode": before_basis_mode,
      "after_basis_mode": after_basis_mode,
      "before_sparse": before,
      "after_sparse": after,
      "decision": decision,
      "exact_decision": exact_decision,
      "before_exact_omitted_force_norm": before_exact,
      "after_exact_omitted_force_norm": after_exact,
      "exact_omitted_force_delta": float(after_exact - before_exact),
      "total_sparse_payload_bytes": int(
          before["sparse_payload_bytes"] + after["sparse_payload_bytes"]),
      "total_full_payload_bytes": int(
          before["full_packet_payload_bytes"] +
          after["full_packet_payload_bytes"]),
  }


def run_graph_stage_adaptive_sparse_basin_decision_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    before_basis_mode: str,
    after_basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promoted_pose_block_count: int = 0,
    promotion_min_value_per_byte: float = 0.0,
    promotion_min_relative_value_fraction: float | None = None,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
    max_top_k_blocks_per_robot: int | None = None,
) -> dict:
  """Search for the smallest per-robot top-k sparse packet with safe decision."""
  exact_candidate = run_graph_stage_sparse_basin_decision_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      before_basis_mode=before_basis_mode,
      after_basis_mode=after_basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      sparse_block_norm_threshold=sparse_block_norm_threshold,
      sparse_top_k_blocks_per_robot=None,
      sparse_payload_index_bytes=sparse_payload_index_bytes,
  )
  exact_packets = (
      exact_candidate["before_sparse"]["sparse_packet_report"]["packets"] +
      exact_candidate["after_sparse"]["sparse_packet_report"]["packets"])
  inferred_max_top_k = max(
      [int(packet["selected_block_count"]) for packet in exact_packets] + [0])
  if max_top_k_blocks_per_robot is None:
    max_top_k = inferred_max_top_k
  else:
    max_top_k = min(max(0, int(max_top_k_blocks_per_robot)), inferred_max_top_k)

  candidates: list[dict] = []
  selected_candidate: dict | None = None
  selected_top_k: int | None = None
  for top_k in range(max_top_k + 1):
    candidate = run_graph_stage_sparse_basin_decision_diagnostic(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        stage=stage,
        before_basis_mode=before_basis_mode,
        after_basis_mode=after_basis_mode,
        dim=dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
        rotations=rotations,
        edge_owner_policy=edge_owner_policy,
        hessian_storage=hessian_storage,
        promoted_pose_block_count=promoted_pose_block_count,
        promotion_min_value_per_byte=promotion_min_value_per_byte,
        promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
        promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
        promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
        sparse_block_norm_threshold=sparse_block_norm_threshold,
        sparse_top_k_blocks_per_robot=top_k,
        sparse_payload_index_bytes=sparse_payload_index_bytes,
    )
    candidate["top_k_blocks_per_robot"] = int(top_k)
    candidates.append(candidate)
    if candidate["decision"]["decision"] in {"safe_accept", "safe_reject"}:
      selected_candidate = candidate
      selected_top_k = int(top_k)
      break

  if selected_candidate is None:
    exact_candidate["top_k_blocks_per_robot"] = None
    selected_candidate = exact_candidate
    selected_top_k = None
    candidates.append(exact_candidate)

  return {
      "model": "graph_stage_adaptive_sparse_basin_decision_diagnostic",
      "stage": stage,
      "before_basis_mode": before_basis_mode,
      "after_basis_mode": after_basis_mode,
      "selected_top_k_blocks_per_robot": selected_top_k,
      "selected_candidate": selected_candidate,
      "candidates": candidates,
      "exact_candidate": exact_candidate,
      "searched_top_k_max": int(max_top_k),
      "inferred_max_top_k_blocks_per_robot": int(inferred_max_top_k),
      "selected_payload_ratio": (
          float(selected_candidate["total_sparse_payload_bytes"]) /
          float(selected_candidate["total_full_payload_bytes"])
          if int(selected_candidate["total_full_payload_bytes"]) > 0 else 0.0),
  }


def run_graph_stage_streaming_sparse_basin_decision_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    before_basis_mode: str,
    after_basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promoted_pose_block_count: int = 0,
    promotion_min_value_per_byte: float = 0.0,
    promotion_min_relative_value_fraction: float | None = None,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
) -> dict:
  """Stream residual blocks in norm order until the basin decision is safe."""
  exact_sparse = run_graph_stage_sparse_basin_decision_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      before_basis_mode=before_basis_mode,
      after_basis_mode=after_basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      sparse_block_norm_threshold=sparse_block_norm_threshold,
      sparse_top_k_blocks_per_robot=None,
      sparse_payload_index_bytes=sparse_payload_index_bytes,
  )
  block_dim = int(exact_sparse["before_sparse"]["basis_diagnostic"][
      "build_stats"]["block_dim"])
  streaming = streaming_sparse_basin_gate_decision(
      exact_sparse["before_sparse"]["sparse_packet_report"],
      exact_sparse["after_sparse"]["sparse_packet_report"],
      block_dim=block_dim,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
      max_omitted_force_increase=promotion_max_omitted_force_increase,
  )
  adaptive = run_graph_stage_adaptive_sparse_basin_decision_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      before_basis_mode=before_basis_mode,
      after_basis_mode=after_basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      sparse_block_norm_threshold=sparse_block_norm_threshold,
      sparse_payload_index_bytes=sparse_payload_index_bytes,
  )
  total_full_payload = int(exact_sparse["total_full_payload_bytes"])
  return {
      "model": "graph_stage_streaming_sparse_basin_decision_diagnostic",
      "stage": stage,
      "before_basis_mode": before_basis_mode,
      "after_basis_mode": after_basis_mode,
      "exact_sparse": exact_sparse,
      "streaming_decision": streaming,
      "adaptive_top_k": adaptive,
      "adaptive_top_k_payload_bytes": int(
          adaptive["selected_candidate"]["total_sparse_payload_bytes"]),
      "exact_sparse_payload_bytes": int(exact_sparse["total_sparse_payload_bytes"]),
      "total_full_payload_bytes": total_full_payload,
      "streaming_payload_ratio": (
          float(streaming["payload_bytes"]) / float(total_full_payload)
          if total_full_payload > 0 else 0.0),
  }


def run_graph_stage_topology_round_streaming_sparse_basin_decision_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    before_basis_mode: str,
    after_basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promoted_pose_block_count: int = 0,
    promotion_min_value_per_byte: float = 0.0,
    promotion_min_relative_value_fraction: float | None = None,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
) -> dict:
  """Round-based local streaming diagnostic for graph-stage residual packets."""
  exact_sparse = run_graph_stage_sparse_basin_decision_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      before_basis_mode=before_basis_mode,
      after_basis_mode=after_basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      sparse_block_norm_threshold=sparse_block_norm_threshold,
      sparse_top_k_blocks_per_robot=None,
      sparse_payload_index_bytes=sparse_payload_index_bytes,
  )
  block_dim = int(exact_sparse["before_sparse"]["basis_diagnostic"][
      "build_stats"]["block_dim"])
  global_streaming = streaming_sparse_basin_gate_decision(
      exact_sparse["before_sparse"]["sparse_packet_report"],
      exact_sparse["after_sparse"]["sparse_packet_report"],
      block_dim=block_dim,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
      max_omitted_force_increase=promotion_max_omitted_force_increase,
  )
  topology_streaming = topology_round_streaming_sparse_basin_gate_decision(
      exact_sparse["before_sparse"]["sparse_packet_report"],
      exact_sparse["after_sparse"]["sparse_packet_report"],
      block_dim=block_dim,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
      max_omitted_force_increase=promotion_max_omitted_force_increase,
  )
  total_full_payload = int(exact_sparse["total_full_payload_bytes"])
  return {
      "model": "graph_stage_topology_round_streaming_sparse_basin_decision_diagnostic",
      "stage": stage,
      "before_basis_mode": before_basis_mode,
      "after_basis_mode": after_basis_mode,
      "exact_sparse": exact_sparse,
      "global_streaming_decision": global_streaming,
      "topology_streaming_decision": topology_streaming,
      "global_streaming_payload_bytes": int(global_streaming["payload_bytes"]),
      "topology_streaming_payload_bytes": int(topology_streaming["payload_bytes"]),
      "exact_sparse_payload_bytes": int(exact_sparse["total_sparse_payload_bytes"]),
      "total_full_payload_bytes": total_full_payload,
      "topology_payload_ratio": (
          float(topology_streaming["payload_bytes"]) / float(total_full_payload)
          if total_full_payload > 0 else 0.0),
  }


def run_graph_stage_receiver_pulled_streaming_sparse_basin_decision_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    before_basis_mode: str,
    after_basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promoted_pose_block_count: int = 0,
    promotion_min_value_per_byte: float = 0.0,
    promotion_min_relative_value_fraction: float | None = None,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
) -> dict:
  """Receiver-pulled local-queue streaming diagnostic for graph stages."""
  exact_sparse = run_graph_stage_sparse_basin_decision_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      before_basis_mode=before_basis_mode,
      after_basis_mode=after_basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promoted_pose_block_count=promoted_pose_block_count,
      promotion_min_value_per_byte=promotion_min_value_per_byte,
      promotion_min_relative_value_fraction=promotion_min_relative_value_fraction,
      promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      sparse_block_norm_threshold=sparse_block_norm_threshold,
      sparse_top_k_blocks_per_robot=None,
      sparse_payload_index_bytes=sparse_payload_index_bytes,
  )
  block_dim = int(exact_sparse["before_sparse"]["basis_diagnostic"][
      "build_stats"]["block_dim"])
  global_streaming = streaming_sparse_basin_gate_decision(
      exact_sparse["before_sparse"]["sparse_packet_report"],
      exact_sparse["after_sparse"]["sparse_packet_report"],
      block_dim=block_dim,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
      max_omitted_force_increase=promotion_max_omitted_force_increase,
  )
  receiver_pulled = receiver_pulled_streaming_sparse_basin_gate_decision(
      exact_sparse["before_sparse"]["sparse_packet_report"],
      exact_sparse["after_sparse"]["sparse_packet_report"],
      block_dim=block_dim,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
      max_omitted_force_increase=promotion_max_omitted_force_increase,
  )
  topology_streaming = topology_round_streaming_sparse_basin_gate_decision(
      exact_sparse["before_sparse"]["sparse_packet_report"],
      exact_sparse["after_sparse"]["sparse_packet_report"],
      block_dim=block_dim,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
      max_omitted_force_increase=promotion_max_omitted_force_increase,
  )
  total_full_payload = int(exact_sparse["total_full_payload_bytes"])
  return {
      "model": "graph_stage_receiver_pulled_streaming_sparse_basin_decision_diagnostic",
      "stage": stage,
      "before_basis_mode": before_basis_mode,
      "after_basis_mode": after_basis_mode,
      "exact_sparse": exact_sparse,
      "global_streaming_decision": global_streaming,
      "receiver_pulled_decision": receiver_pulled,
      "topology_streaming_decision": topology_streaming,
      "global_streaming_payload_bytes": int(global_streaming["payload_bytes"]),
      "receiver_pulled_payload_bytes": int(receiver_pulled["payload_bytes"]),
      "topology_streaming_payload_bytes": int(topology_streaming["payload_bytes"]),
      "exact_sparse_payload_bytes": int(exact_sparse["total_sparse_payload_bytes"]),
      "total_full_payload_bytes": total_full_payload,
      "receiver_pulled_payload_ratio": (
          float(receiver_pulled["payload_bytes"]) / float(total_full_payload)
          if total_full_payload > 0 else 0.0),
  }


def run_graph_stage_receiver_pulled_top_residual_enrichment_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
) -> dict:
  """Use receiver-pulled residual packets to choose one skeleton enrichment."""
  basis_diagnostic = run_graph_stage_primal_skeleton_basis_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      basis_mode=basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
  )
  pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  local_systems, interface_indices, build_stats = (
      build_graph_local_normal_systems_for_interface_schur(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          stage=stage,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          rotations=rotations,
          edge_owner_policy=edge_owner_policy,
          hessian_storage=hessian_storage,
      )
  )
  interface_state = np.asarray(
      basis_diagnostic["solve"]["lifted_interface_solution"],
      dtype=float).reshape(-1)
  variable_interface_pose_ids = _variable_interface_pose_ids(
      build_stats["interface_pose_ids"],
      len(interface_indices),
      build_stats["block_dim"],
      anchor_pose=anchor_pose)
  sparse_report = local_schur_sparse_residual_packets(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_state=interface_state,
      interface_pose_ids=variable_interface_pose_ids,
      block_dim=build_stats["block_dim"],
      variable_count=build_stats["variable_count"],
      block_norm_threshold=sparse_block_norm_threshold,
      top_k_blocks_per_robot=None,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
  )
  selected_pose_ids = basis_diagnostic["solve"].get(
      "selected_interface_pose_ids", [])
  top_decision = receiver_pulled_top_residual_block_decision(
      sparse_report,
      block_dim=build_stats["block_dim"],
      selected_pose_ids=selected_pose_ids,
      payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
      payload_index_bytes=sparse_payload_index_bytes,
  )
  centralized_top_block = (
      basis_diagnostic["solve"]["omitted_force_blocks"][0]
      if basis_diagnostic["solve"]["omitted_force_blocks"] else None)
  packet_pose_id = top_decision["decision"].get("top_pose_id")
  after_solve = None
  after_omitted_force_norm = float(
      basis_diagnostic["solve"]["omitted_force_norm"])
  if top_decision["decision"]["decision"] == "safe_top" and packet_pose_id is not None:
    promotion_basis = pose_block_promotion_basis(
        interface_pose_ids=variable_interface_pose_ids,
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        omitted_force_blocks=[{
            "pose_id": int(packet_pose_id),
            "selected": False,
            "block_norm": float(top_decision["decision"]["top_lower_bound"]),
        }],
        max_promoted_pose_blocks=1,
        anchor_pose=anchor_pose,
    )
    base_basis = np.asarray(basis_diagnostic["basis"]["basis"], dtype=float)
    combined_basis = np.column_stack(
        [base_basis, promotion_basis["basis"]]
        if base_basis.size else [promotion_basis["basis"]])
    combined_basis = _orthonormalize_basis_columns(
        combined_basis,
        row_count=len(interface_indices),
    )
    after_solve = solve_skeleton_reduced_interface_schur_with_basis(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=variable_interface_pose_ids,
        block_dim=build_stats["block_dim"],
        basis=combined_basis,
        basis_metadata={
            "model": "receiver_pulled_top_residual_enriched_basis",
            "base_basis_model": basis_diagnostic["basis_model"],
            "packet_selected_pose_id": int(packet_pose_id),
        },
        anchor_pose=anchor_pose,
    )
    after_omitted_force_norm = float(after_solve["omitted_force_norm"])

  return {
      "model": "graph_stage_receiver_pulled_top_residual_enrichment_diagnostic",
      "stage": stage,
      "basis_mode": basis_mode,
      "basis_diagnostic": basis_diagnostic,
      "sparse_packet_report": sparse_report,
      "top_residual_decision": top_decision,
      "centralized_top_pose_id": (
          None if centralized_top_block is None
          else int(centralized_top_block["pose_id"])),
      "centralized_top_block_norm": (
          0.0 if centralized_top_block is None
          else float(centralized_top_block["block_norm"])),
      "packet_selected_pose_id": (
          None if packet_pose_id is None else int(packet_pose_id)),
      "before_omitted_force_norm": float(
          basis_diagnostic["solve"]["omitted_force_norm"]),
      "after_omitted_force_norm": float(after_omitted_force_norm),
      "after_solve": after_solve,
      "top_residual_payload_bytes": int(top_decision["payload_bytes"]),
      "full_packet_payload_bytes": int(sparse_report["dense_payload_bytes"]),
      "sparse_packet_payload_bytes": int(sparse_report["payload_bytes"]),
  }


def run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    max_iterations: int = 3,
    stop_omitted_force_norm: float = 1e-12,
    allow_top_set_promotion: bool = False,
    promotion_policy: str = "top_residual",
    promotion_min_value_per_byte: float = 0.0,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_candidate_top_k: int = 4,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
) -> dict:
  """Iteratively enrich a graph-stage skeleton using pulled residual packets."""
  promotion_policy = str(promotion_policy)
  if promotion_policy not in {
      "top_residual", "basin_certificate", "topk_basin_certificate"}:
    raise ValueError(f"unsupported promotion_policy: {promotion_policy}")
  basis_diagnostic = run_graph_stage_primal_skeleton_basis_diagnostic(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage=stage,
      basis_mode=basis_mode,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      hessian_storage=hessian_storage,
      promotion_payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
  )
  pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  local_systems, interface_indices, build_stats = (
      build_graph_local_normal_systems_for_interface_schur(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          stage=stage,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          rotations=rotations,
          edge_owner_policy=edge_owner_policy,
          hessian_storage=hessian_storage,
      )
  )
  variable_interface_pose_ids = _variable_interface_pose_ids(
      build_stats["interface_pose_ids"],
      len(interface_indices),
      build_stats["block_dim"],
      anchor_pose=anchor_pose)
  current_basis = _orthonormalize_basis_columns(
      np.asarray(basis_diagnostic["basis"]["basis"], dtype=float),
      row_count=len(interface_indices),
  )
  promoted_pose_ids = [
      int(pose_id)
      for pose_id in basis_diagnostic.get("promotion", {}).get(
          "promoted_pose_ids", [])
  ]
  current_solve = solve_skeleton_reduced_interface_schur_with_basis(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_pose_ids=variable_interface_pose_ids,
      block_dim=build_stats["block_dim"],
      basis=current_basis,
      basis_metadata={
          "model": "receiver_pulled_top_residual_initial_basis",
          "base_basis_model": basis_diagnostic["basis_model"],
      },
      selected_interface_pose_ids=promoted_pose_ids,
      anchor_pose=anchor_pose,
  )
  initial_omitted_force_norm = float(current_solve["omitted_force_norm"])
  history: list[dict] = []
  total_top_payload = 0
  total_full_payload = 0
  total_sparse_payload = 0
  total_promotion_payload = 0
  converged = initial_omitted_force_norm <= float(stop_omitted_force_norm)
  stopped_reason = (
      "omitted_force_tolerance" if converged else "max_iterations")
  iteration_limit = max(0, int(max_iterations))

  for iteration in range(iteration_limit):
    before_norm = float(current_solve["omitted_force_norm"])
    if before_norm <= float(stop_omitted_force_norm):
      converged = True
      stopped_reason = "omitted_force_tolerance"
      break
    interface_state = np.asarray(
        current_solve["lifted_interface_solution"], dtype=float).reshape(-1)
    sparse_report = local_schur_sparse_residual_packets(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_state=interface_state,
        interface_pose_ids=variable_interface_pose_ids,
        block_dim=build_stats["block_dim"],
        variable_count=build_stats["variable_count"],
        block_norm_threshold=sparse_block_norm_threshold,
        top_k_blocks_per_robot=None,
        payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
        payload_index_bytes=sparse_payload_index_bytes,
    )
    top_decision = receiver_pulled_top_residual_block_decision(
        sparse_report,
        block_dim=build_stats["block_dim"],
        selected_pose_ids=promoted_pose_ids,
        payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
        payload_index_bytes=sparse_payload_index_bytes,
        allow_top_set=allow_top_set_promotion,
    )
    total_full_payload += int(sparse_report["dense_payload_bytes"])
    total_sparse_payload += int(sparse_report["payload_bytes"])
    centralized_top_block = (
        current_solve["omitted_force_blocks"][0]
        if current_solve["omitted_force_blocks"] else None)
    promotion_decision: dict
    if promotion_policy == "basin_certificate":
      max_blocks_this_round = (
          max(1, len(current_solve["omitted_force_blocks"]))
          if allow_top_set_promotion else 1)
      promotion_decision = pose_block_basin_certificate_promotion_basis(
          interface_pose_ids=variable_interface_pose_ids,
          interface_variable_count=len(interface_indices),
          block_dim=build_stats["block_dim"],
          omitted_force_blocks=current_solve["omitted_force_blocks"],
          full_schur=current_solve["full_schur"],
          full_rhs=current_solve["full_rhs"],
          base_basis=current_basis,
          min_value_per_byte=promotion_min_value_per_byte,
          max_omitted_force_increase=promotion_max_omitted_force_increase,
          max_promoted_pose_blocks=max_blocks_this_round,
          payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
          anchor_pose=anchor_pose,
      )
      packet_pose_ids = [
          int(pose_id)
          for pose_id in promotion_decision.get("promoted_pose_ids", [])
      ]
      promotion_payload = int(sum(
          int(item.get("payload_bytes", 0))
          for item in promotion_decision.get("accepted_promotions", [])))
      decision_label = (
          "safe_merit_promotion" if packet_pose_ids
          else "no_merit_safe_promotion")
    elif promotion_policy == "topk_basin_certificate":
      promotion_decision = pose_block_topk_basin_certificate_promotion_basis(
          interface_pose_ids=variable_interface_pose_ids,
          interface_variable_count=len(interface_indices),
          block_dim=build_stats["block_dim"],
          omitted_force_blocks=current_solve["omitted_force_blocks"],
          full_schur=current_solve["full_schur"],
          full_rhs=current_solve["full_rhs"],
          base_basis=current_basis,
          candidate_top_k=promotion_candidate_top_k,
          max_omitted_force_increase=promotion_max_omitted_force_increase,
          payload_bytes_per_coordinate=promotion_payload_bytes_per_coordinate,
          anchor_pose=anchor_pose,
      )
      packet_pose_ids = [
          int(pose_id)
          for pose_id in promotion_decision.get("promoted_pose_ids", [])
      ]
      promotion_payload = int(sum(
          int(item.get("payload_bytes", 0))
          for item in promotion_decision.get("accepted_promotions", [])))
      decision_label = (
          "safe_merit_promotion" if packet_pose_ids
          else "no_merit_safe_promotion")
    else:
      decision_label = str(top_decision["decision"]["decision"])
      if decision_label == "safe_top_set":
        packet_pose_ids = [
            int(pose_id)
            for pose_id in top_decision["decision"].get("top_pose_ids", [])
        ]
      else:
        packet_pose = top_decision["decision"].get("top_pose_id")
        packet_pose_ids = [] if packet_pose is None else [int(packet_pose)]
      lower_bound_by_pose = {
          int(record["pose_id"]): float(record["lower_bound"])
          for record in top_decision["decision"].get("candidate_bounds", [])
      }
      promotion_decision = pose_block_promotion_basis(
          interface_pose_ids=variable_interface_pose_ids,
          interface_variable_count=len(interface_indices),
          block_dim=build_stats["block_dim"],
          omitted_force_blocks=[
              {
                  "pose_id": int(pose_id),
                  "selected": False,
                  "block_norm": float(lower_bound_by_pose.get(
                      int(pose_id),
                      top_decision["decision"].get("top_lower_bound", 0.0))),
              }
              for pose_id in packet_pose_ids
          ],
          max_promoted_pose_blocks=len(packet_pose_ids),
          anchor_pose=anchor_pose,
      )
      promotion_payload = int(top_decision["payload_bytes"])
    total_top_payload += int(top_decision["payload_bytes"])
    total_promotion_payload += int(promotion_payload)
    promotion_succeeded = (
        decision_label == "safe_merit_promotion"
        if promotion_policy in {"basin_certificate", "topk_basin_certificate"}
        else decision_label in {"safe_top", "safe_top_set"})
    if not promotion_succeeded or not packet_pose_ids:
      history.append({
          "iteration": int(iteration),
          "before_omitted_force_norm": before_norm,
          "after_omitted_force_norm": before_norm,
          "centralized_top_pose_id": (
              None if centralized_top_block is None
              else int(centralized_top_block["pose_id"])),
          "packet_selected_pose_id": None,
          "packet_selected_pose_ids": [],
          "top_residual_decision": top_decision,
          "promotion_policy": promotion_policy,
          "promotion_decision": promotion_decision,
          "promotion_payload_bytes": int(promotion_payload),
          "top_residual_payload_bytes": int(top_decision["payload_bytes"]),
          "full_packet_payload_bytes": int(sparse_report["dense_payload_bytes"]),
          "sparse_packet_payload_bytes": int(sparse_report["payload_bytes"]),
          "promoted_pose_ids_after": list(promoted_pose_ids),
      })
      stopped_reason = (
          "no_merit_safe_promotion"
          if decision_label == "no_merit_safe_promotion" else
          "no_enrichment_candidate"
          if decision_label == "no_candidate" else "uncertain_top_residual")
      break

    if promotion_policy in {"basin_certificate", "topk_basin_certificate"}:
      next_basis = np.asarray(promotion_decision["basis"], dtype=float)
      added_columns = int(next_basis.shape[1] - current_basis.shape[1])
    else:
      next_basis, added_columns = (
          _append_independent_columns_to_orthonormal_basis(
              current_basis,
              promotion_decision["basis"],
              row_count=len(interface_indices),
          )
      )
    if added_columns <= 0:
      history.append({
          "iteration": int(iteration),
          "before_omitted_force_norm": before_norm,
          "after_omitted_force_norm": before_norm,
          "centralized_top_pose_id": (
              None if centralized_top_block is None
              else int(centralized_top_block["pose_id"])),
          "packet_selected_pose_id": int(packet_pose_ids[0]),
          "packet_selected_pose_ids": list(packet_pose_ids),
          "top_residual_decision": top_decision,
          "promotion_policy": promotion_policy,
          "promotion_decision": promotion_decision,
          "promotion_payload_bytes": int(promotion_payload),
          "top_residual_payload_bytes": int(top_decision["payload_bytes"]),
          "full_packet_payload_bytes": int(sparse_report["dense_payload_bytes"]),
          "sparse_packet_payload_bytes": int(sparse_report["payload_bytes"]),
          "promoted_pose_ids_after": list(promoted_pose_ids),
      })
      stopped_reason = "dependent_promotion"
      break

    for pose_id in packet_pose_ids:
      if int(pose_id) not in promoted_pose_ids:
        promoted_pose_ids.append(int(pose_id))
    next_solve = solve_skeleton_reduced_interface_schur_with_basis(
        local_systems=local_systems,
        interface_indices=interface_indices,
        interface_pose_ids=variable_interface_pose_ids,
        block_dim=build_stats["block_dim"],
        basis=next_basis,
        basis_metadata={
            "model": "receiver_pulled_top_residual_iterated_basis",
            "base_basis_model": basis_diagnostic["basis_model"],
            "promoted_pose_ids": list(promoted_pose_ids),
        },
        selected_interface_pose_ids=promoted_pose_ids,
        anchor_pose=anchor_pose,
    )
    after_norm = float(next_solve["omitted_force_norm"])
    history.append({
        "iteration": int(iteration),
        "before_omitted_force_norm": before_norm,
        "after_omitted_force_norm": after_norm,
        "centralized_top_pose_id": (
            None if centralized_top_block is None
            else int(centralized_top_block["pose_id"])),
        "centralized_top_block_norm": (
            0.0 if centralized_top_block is None
            else float(centralized_top_block["block_norm"])),
        "packet_selected_pose_id": int(packet_pose_ids[0]),
        "packet_selected_pose_ids": list(packet_pose_ids),
        "top_residual_decision": top_decision,
        "promotion_policy": promotion_policy,
        "promotion_decision": promotion_decision,
        "promotion_payload_bytes": int(promotion_payload),
        "top_residual_payload_bytes": int(top_decision["payload_bytes"]),
        "full_packet_payload_bytes": int(sparse_report["dense_payload_bytes"]),
        "sparse_packet_payload_bytes": int(sparse_report["payload_bytes"]),
        "promoted_pose_ids_after": list(promoted_pose_ids),
    })
    current_basis = next_basis
    current_solve = next_solve
    if after_norm <= float(stop_omitted_force_norm):
      converged = True
      stopped_reason = "omitted_force_tolerance"
      break
  else:
    stopped_reason = "max_iterations"

  final_omitted_force_norm = float(current_solve["omitted_force_norm"])
  return {
      "model": "graph_stage_receiver_pulled_top_residual_enrichment_iterations",
      "stage": stage,
      "basis_mode": basis_mode,
      "promotion_policy": promotion_policy,
      "promotion_candidate_top_k": int(promotion_candidate_top_k),
      "basis_diagnostic": basis_diagnostic,
      "initial_omitted_force_norm": initial_omitted_force_norm,
      "final_omitted_force_norm": final_omitted_force_norm,
      "final_solve": current_solve,
      "history": history,
      "iteration_count": int(len(history)),
      "max_iterations": iteration_limit,
      "stop_omitted_force_norm": float(stop_omitted_force_norm),
      "allow_top_set_promotion": bool(allow_top_set_promotion),
      "converged": bool(converged),
      "stopped_reason": stopped_reason,
      "promoted_pose_ids": list(promoted_pose_ids),
      "total_top_residual_payload_bytes": int(total_top_payload),
      "total_promotion_payload_bytes": int(total_promotion_payload),
      "total_full_packet_payload_bytes": int(total_full_payload),
      "total_sparse_packet_payload_bytes": int(total_sparse_payload),
      "final_basis_column_count": int(current_basis.shape[1]),
      "full_interface_variable_count": int(len(interface_indices)),
  }


def run_two_stage_receiver_pulled_top_residual_enrichment_summary(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    max_iterations: int = 3,
    stop_omitted_force_norm: float = 1e-12,
    allow_top_set_promotion: bool = False,
    promotion_policy: str = "top_residual",
    promotion_min_value_per_byte: float = 0.0,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_candidate_top_k: int = 4,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
) -> dict:
  """Run packet-driven DCI enrichment for rotation and translation stages.

  This is a stage-level diagnostic wrapper. It checks whether pulled packet
  enrichment can close the omitted Schur force in each chordal-initialization
  subproblem; it does not recover poses or evaluate the nonlinear handoff cost.
  """
  stage_dim = int(dim) if dim is not None else (
      int(graph_edges[0].dim) if graph_edges else 3)
  sorted_pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  stage_rotations = rotations
  if stage_rotations is None:
    stage_rotations = {
        int(pose_id): np.eye(stage_dim, dtype=float)
        for pose_id in sorted_pose_ids
    }

  stage_reports: dict[str, dict] = {}
  for stage in ("rotation", "translation"):
    stage_reports[stage] = (
        run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
            graph_edges=graph_edges,
            pose_ids=sorted_pose_ids,
            robot_of=robot_of,
            stage=stage,
            basis_mode=basis_mode,
            dim=stage_dim,
            weighted=weighted,
            cost_mode=cost_mode,
            anchor_pose=anchor_pose,
            rotations=stage_rotations,
            edge_owner_policy=edge_owner_policy,
            hessian_storage=hessian_storage,
            max_iterations=max_iterations,
            stop_omitted_force_norm=stop_omitted_force_norm,
            allow_top_set_promotion=allow_top_set_promotion,
            promotion_policy=promotion_policy,
            promotion_min_value_per_byte=promotion_min_value_per_byte,
            promotion_max_omitted_force_increase=(
                promotion_max_omitted_force_increase),
            promotion_candidate_top_k=promotion_candidate_top_k,
            promotion_payload_bytes_per_coordinate=(
                promotion_payload_bytes_per_coordinate),
            sparse_block_norm_threshold=sparse_block_norm_threshold,
            sparse_payload_index_bytes=sparse_payload_index_bytes,
        )
    )

  rotation_report = stage_reports["rotation"]
  translation_report = stage_reports["translation"]
  return {
      "model": "two_stage_receiver_pulled_top_residual_enrichment_summary",
      "stage_order": ["rotation", "translation"],
      "basis_mode": basis_mode,
      "dim": stage_dim,
      "weighted": bool(weighted),
      "cost_mode": cost_mode,
      "anchor_pose": anchor_pose,
      "max_iterations": int(max_iterations),
      "stop_omitted_force_norm": float(stop_omitted_force_norm),
      "allow_top_set_promotion": bool(allow_top_set_promotion),
      "promotion_policy": str(promotion_policy),
      "promotion_candidate_top_k": int(promotion_candidate_top_k),
      "stages": stage_reports,
      "rotation_converged": bool(rotation_report["converged"]),
      "translation_converged": bool(translation_report["converged"]),
      "all_stages_converged": bool(
          rotation_report["converged"] and translation_report["converged"]),
      "rotation_initial_omitted_force_norm": float(
          rotation_report["initial_omitted_force_norm"]),
      "rotation_final_omitted_force_norm": float(
          rotation_report["final_omitted_force_norm"]),
      "translation_initial_omitted_force_norm": float(
          translation_report["initial_omitted_force_norm"]),
      "translation_final_omitted_force_norm": float(
          translation_report["final_omitted_force_norm"]),
      "total_top_residual_payload_bytes": int(sum(
          int(report["total_top_residual_payload_bytes"])
          for report in stage_reports.values())),
      "total_promotion_payload_bytes": int(sum(
          int(report.get("total_promotion_payload_bytes", 0))
          for report in stage_reports.values())),
      "total_full_packet_payload_bytes": int(sum(
          int(report["total_full_packet_payload_bytes"])
          for report in stage_reports.values())),
      "total_sparse_packet_payload_bytes": int(sum(
          int(report["total_sparse_packet_payload_bytes"])
          for report in stage_reports.values())),
      "rotation_promoted_pose_ids": list(
          rotation_report["promoted_pose_ids"]),
      "translation_promoted_pose_ids": list(
          translation_report["promoted_pose_ids"]),
  }


def run_graph_stage_matrix_free_interface_pcg_solve(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "sparse",
    pcg_iterations: int = 100,
    pcg_tolerance: float = 1e-10,
    pcg_damping: float = 0.0,
    schur_preconditioner: str = "diagonal",
    communication_model: str = "separator_owner_star",
    robot_topology_edges: list[tuple[object, object]] | None = None,
) -> dict:
  """Solve one chordal-initialization stage with matrix-free local-Schur PCG."""
  sorted_pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  local_systems, interface_indices, build_stats = (
      build_graph_local_normal_systems_for_interface_schur(
          graph_edges=graph_edges,
          pose_ids=sorted_pose_ids,
          robot_of=robot_of,
          stage=stage,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          rotations=rotations,
          edge_owner_policy=edge_owner_policy,
          hessian_storage=hessian_storage,
      )
  )
  variable_interface_pose_ids = _variable_interface_pose_ids(
      build_stats["interface_pose_ids"],
      len(interface_indices),
      build_stats["block_dim"],
      anchor_pose=anchor_pose)
  zero_state = np.zeros(len(interface_indices), dtype=float)
  initial_packets = local_schur_residual_packets(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_state=zero_state,
      interface_pose_ids=variable_interface_pose_ids,
      block_dim=build_stats["block_dim"],
      variable_count=build_stats["variable_count"],
      damping=pcg_damping,
  )
  full_solution, solver_stats = solve_local_interface_schur_pcg(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=build_stats["variable_count"],
      iterations=pcg_iterations,
      tolerance=pcg_tolerance,
      damping=pcg_damping,
      schur_preconditioner=schur_preconditioner,
      communication_model=communication_model,
      robot_topology_edges=robot_topology_edges,
  )
  interface_solution = np.asarray(full_solution, dtype=float)[
      np.asarray(interface_indices, dtype=int)]
  final_norm = float(solver_stats["final_schur_residual"])
  converged = final_norm <= float(pcg_tolerance)
  return {
      "model": "graph_stage_matrix_free_interface_pcg_solve",
      "stage": stage,
      "interface_solver": "pcg",
      "hessian_storage": hessian_storage,
      "initial_omitted_force_norm": float(
          initial_packets["aggregate_residual_norm"]),
      "final_omitted_force_norm": final_norm,
      "final_solve": {
          "model": "matrix_free_interface_pcg_stage_solution",
          "lifted_interface_solution": interface_solution,
          "full_solution": np.asarray(full_solution, dtype=float),
          "omitted_force_norm": final_norm,
      },
      "solver_stats": solver_stats,
      "history": [],
      "iteration_count": int(solver_stats.get("iterations", 0)),
      "max_iterations": int(pcg_iterations),
      "stop_omitted_force_norm": float(pcg_tolerance),
      "converged": bool(converged),
      "stopped_reason": (
          "pcg_tolerance" if converged else "pcg_max_iterations"),
      "promoted_pose_ids": [],
      "total_top_residual_payload_bytes": 0,
      "total_promotion_payload_bytes": 0,
      "total_full_packet_payload_bytes": 0,
      "total_sparse_packet_payload_bytes": 0,
      "total_pcg_comm_mb": float(solver_stats.get("estimated_comm_mb", 0.0)),
      "total_pcg_comm_bytes": int(solver_stats.get("estimated_comm_bytes", 0)),
      "final_basis_column_count": int(len(interface_indices)),
      "full_interface_variable_count": int(len(interface_indices)),
      "build_stats": build_stats,
  }


def _global_coordinate_interface_coarse_basis(
    interface_variable_count: int,
    block_dim: int,
) -> np.ndarray:
  interface_variable_count = int(interface_variable_count)
  block_dim = max(1, int(block_dim))
  if interface_variable_count <= 0:
    return np.zeros((0, 0), dtype=float)
  column_count = min(block_dim, interface_variable_count)
  basis = np.zeros((interface_variable_count, column_count), dtype=float)
  for index in range(interface_variable_count):
    coordinate = int(index % block_dim)
    if coordinate < column_count:
      basis[index, coordinate] = 1.0
  active_columns = [
      column for column in range(column_count)
      if np.linalg.norm(basis[:, column]) > 0.0
  ]
  return basis[:, active_columns] if active_columns else np.zeros(
      (interface_variable_count, 0), dtype=float)


def _robot_coordinate_interface_coarse_basis(
    interface_pose_ids: list[int],
    robot_of: dict[int, int],
    block_dim: int,
) -> np.ndarray:
  block_dim = max(1, int(block_dim))
  pose_ids = [int(pose_id) for pose_id in interface_pose_ids]
  interface_variable_count = int(len(pose_ids) * block_dim)
  if interface_variable_count <= 0:
    return np.zeros((0, 0), dtype=float)
  robot_ids = sorted({int(robot_of.get(pose_id, -1)) for pose_id in pose_ids})
  columns: list[np.ndarray] = []
  for robot in robot_ids:
    if robot < 0:
      continue
    for coordinate in range(block_dim):
      column = np.zeros(interface_variable_count, dtype=float)
      for pose_index, pose_id in enumerate(pose_ids):
        if int(robot_of.get(pose_id, -1)) != robot:
          continue
        column[pose_index * block_dim + coordinate] = 1.0
      if np.linalg.norm(column) > 0.0:
        columns.append(column)
  if not columns:
    return np.zeros((interface_variable_count, 0), dtype=float)
  return np.column_stack(columns)


def run_graph_stage_matrix_free_interface_fixed_step_solve(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    stage: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    rotations: dict[int, np.ndarray] | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "sparse",
    iterations: int = 100,
    tolerance: float = 1e-10,
    damping: float = 0.0,
    schur_preconditioner: str = "diagonal",
    relaxation: float = 1.0,
    coarse_basis_mode: str = "none",
    adaptive_coarse_basis: str = "none",
    adaptive_coarse_rank: int = 0,
    adaptive_coarse_probe_count: int = 0,
    adaptive_coarse_seed: int = 0,
    component_modes_per_component: int = 1,
    component_max_components: int = 0,
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
    solver_label: str = "fixed_step",
    communication_model: str = "separator_owner_star",
    robot_topology_edges: list[tuple[object, object]] | None = None,
) -> dict:
  """Solve one stage with reduction-free fixed-step local-Schur iterations."""
  sorted_pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  local_systems, interface_indices, build_stats = (
      build_graph_local_normal_systems_for_interface_schur(
          graph_edges=graph_edges,
          pose_ids=sorted_pose_ids,
          robot_of=robot_of,
          stage=stage,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          rotations=rotations,
          edge_owner_policy=edge_owner_policy,
          hessian_storage=hessian_storage,
      )
  )
  variable_interface_pose_ids = _variable_interface_pose_ids(
      build_stats["interface_pose_ids"],
      len(interface_indices),
      build_stats["block_dim"],
      anchor_pose=anchor_pose)
  zero_state = np.zeros(len(interface_indices), dtype=float)
  initial_packets = local_schur_residual_packets(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_state=zero_state,
      interface_pose_ids=variable_interface_pose_ids,
      block_dim=build_stats["block_dim"],
      variable_count=build_stats["variable_count"],
      damping=damping,
  )
  coarse_basis_mode = str(coarse_basis_mode)
  geneo_basis_report = None
  merit_basis_report = None
  geneo_setup_schur_bytes = 0
  merit_setup_schur_bytes = 0
  if coarse_basis_mode == "none":
    coarse_basis = None
  elif coarse_basis_mode == "global_coordinate":
    coarse_basis = _global_coordinate_interface_coarse_basis(
        len(interface_indices), build_stats["block_dim"])
  elif coarse_basis_mode == "robot_coordinate":
    coarse_basis = _robot_coordinate_interface_coarse_basis(
        variable_interface_pose_ids, robot_of, build_stats["block_dim"])
  elif coarse_basis_mode == "separator_component_coordinate":
    basis_report = separator_component_coordinate_skeleton_basis(
        graph_edges=graph_edges,
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        anchor_pose=anchor_pose,
    )
    coarse_basis = basis_report["basis"]
  elif coarse_basis_mode == "separator_component_geneo":
    full_schur, full_rhs, _ = sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )
    geneo_setup_schur_bytes = int(full_schur.nbytes + full_rhs.nbytes)
    geneo_basis_report = separator_component_geneo_skeleton_basis(
        graph_edges=graph_edges,
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        full_schur=full_schur,
        full_rhs=full_rhs,
        max_modes_per_component=component_modes_per_component,
        anchor_pose=anchor_pose,
    )
    coarse_basis = geneo_basis_report["basis"]
  elif coarse_basis_mode == "separator_component_merit":
    full_schur, full_rhs, _ = sum_local_interface_schur_contributions(
        local_systems=local_systems,
        interface_indices=interface_indices,
    )
    merit_setup_schur_bytes = int(full_schur.nbytes + full_rhs.nbytes)
    merit_basis_report = separator_component_merit_skeleton_basis(
        graph_edges=graph_edges,
        interface_pose_ids=build_stats["interface_pose_ids"],
        interface_variable_count=len(interface_indices),
        block_dim=build_stats["block_dim"],
        robot_of=robot_of,
        full_schur=full_schur,
        full_rhs=full_rhs,
        max_selected_components=component_max_components,
        anchor_pose=anchor_pose,
    )
    coarse_basis = merit_basis_report["basis"]
  else:
    raise ValueError(f"unsupported coarse_basis_mode: {coarse_basis_mode}")
  full_solution, solver_stats = solve_local_interface_schur_fixed_step(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=build_stats["variable_count"],
      iterations=iterations,
      tolerance=tolerance,
      damping=damping,
      schur_preconditioner=schur_preconditioner,
      relaxation=relaxation,
      coarse_basis=coarse_basis,
      adaptive_coarse_basis=adaptive_coarse_basis,
      adaptive_coarse_rank=adaptive_coarse_rank,
      adaptive_coarse_probe_count=adaptive_coarse_probe_count,
      adaptive_coarse_seed=adaptive_coarse_seed,
      fixed_step_acceleration=fixed_step_acceleration,
      chebyshev_lambda_min=chebyshev_lambda_min,
      chebyshev_lambda_max=chebyshev_lambda_max,
      chebyshev_safety_monitor=chebyshev_safety_monitor,
      chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
      chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
      chebyshev_ritz_seed=chebyshev_ritz_seed,
      chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
      communication_model=communication_model,
      robot_topology_edges=robot_topology_edges,
  )
  solver_stats = dict(solver_stats)
  solver_stats["coarse_basis_mode"] = coarse_basis_mode
  solver_stats["component_modes_per_component"] = int(
      component_modes_per_component)
  solver_stats["component_max_components"] = int(component_max_components)
  if geneo_basis_report is not None:
    solver_stats["component_geneo_component_count"] = int(
        geneo_basis_report["component_count"])
    solver_stats["component_geneo_basis_column_count"] = int(
        geneo_basis_report["basis_column_count"])
    solver_stats["component_geneo_setup_dense_schur_bytes"] = int(
        geneo_setup_schur_bytes)
    solver_stats["component_geneo_setup_materializes_dense_schur"] = True
  if merit_basis_report is not None:
    solver_stats["component_merit_component_count"] = int(
        merit_basis_report["component_count"])
    solver_stats["component_merit_selected_component_count"] = int(
        merit_basis_report["selected_component_count"])
    solver_stats["component_merit_basis_column_count"] = int(
        merit_basis_report["basis_column_count"])
    solver_stats["component_merit_setup_dense_schur_bytes"] = int(
        merit_setup_schur_bytes)
    solver_stats["component_merit_setup_materializes_dense_schur"] = True
  interface_solution = np.asarray(full_solution, dtype=float)[
      np.asarray(interface_indices, dtype=int)]
  final_norm = float(solver_stats["final_schur_residual"])
  converged = final_norm <= float(tolerance)
  return {
      "model": "graph_stage_matrix_free_interface_fixed_step_solve",
      "stage": stage,
      "interface_solver": str(solver_label),
      "hessian_storage": hessian_storage,
      "initial_omitted_force_norm": float(
          initial_packets["aggregate_residual_norm"]),
      "final_omitted_force_norm": final_norm,
      "final_solve": {
          "model": "matrix_free_interface_fixed_step_stage_solution",
          "lifted_interface_solution": interface_solution,
          "full_solution": np.asarray(full_solution, dtype=float),
          "omitted_force_norm": final_norm,
      },
      "solver_stats": solver_stats,
      "history": [],
      "iteration_count": int(solver_stats.get("iterations", 0)),
      "max_iterations": int(iterations),
      "stop_omitted_force_norm": float(tolerance),
      "converged": bool(converged),
      "stopped_reason": (
          "fixed_step_tolerance" if converged else "fixed_step_max_iterations"),
      "promoted_pose_ids": [],
      "total_top_residual_payload_bytes": 0,
      "total_promotion_payload_bytes": 0,
      "total_full_packet_payload_bytes": 0,
      "total_sparse_packet_payload_bytes": 0,
      "total_pcg_comm_mb": float(solver_stats.get("estimated_comm_mb", 0.0)),
      "total_pcg_comm_bytes": int(solver_stats.get("estimated_comm_bytes", 0)),
      "final_basis_column_count": int(len(interface_indices)),
      "full_interface_variable_count": int(len(interface_indices)),
      "build_stats": build_stats,
  }


def _pcg_stage_full_solution_reuse_report(stage_report: dict) -> dict:
  solver_stats = dict(stage_report.get("solver_stats", {}))
  final_solve = stage_report.get("final_solve", {})
  solution = np.asarray(final_solve.get("full_solution", []), dtype=float)
  solver = str(stage_report.get("interface_solver", "pcg"))
  model = (
      "pcg_full_solution_reuse" if solver == "pcg"
      else f"{solver}_full_solution_reuse")
  return {
      "model": model,
      "uses_dense_backsubstitution": False,
      "solution": solution,
      "private_backsubstitution_count": int(
          solver_stats.get("private_backsubstitution_count", 0)),
      "unassigned_variable_count": int(
          solver_stats.get("unassigned_variable_count", 0)),
      "local_singular_count": int(
          solver_stats.get("local_singular_count", 0)),
      "final_normal_residual": float(
          solver_stats.get("final_normal_residual", 0.0)),
      "final_schur_residual": float(
          solver_stats.get("final_schur_residual", 0.0)),
      "materializes_dense_schur": bool(
          solver_stats.get("materializes_dense_schur", False)),
      "materializes_dense_local_hessian": bool(
          solver_stats.get("materializes_dense_local_hessian", False)),
      "uses_sparse_private_factorization": bool(
          solver_stats.get("uses_sparse_private_factorization", False)),
  }


def backsubstitute_full_solution_from_interface(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_solution: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0,
) -> dict:
  """Recover private variables conditioned on an interface Schur solution."""
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  interface_solution = np.asarray(interface_solution, dtype=float).reshape(-1)
  if len(interface_solution) != len(interface_indices):
    raise ValueError("interface_solution length must match interface_indices")
  if variable_count is None:
    inferred = int(np.max(interface_indices)) + 1 if interface_indices.size else 0
    for system in local_systems:
      hessian = system.get("hessian")
      if hessian is None:
        continue
      inferred = max(inferred, int(hessian.shape[0]))
    variable_count = inferred
  variable_count = int(variable_count)
  if variable_count < 0:
    raise ValueError("variable_count must be non-negative")
  if interface_indices.size:
    if (int(np.min(interface_indices)) < 0 or
        int(np.max(interface_indices)) >= variable_count):
      raise ValueError("interface index outside variable dimension")

  solution = np.zeros(variable_count, dtype=float)
  assigned = np.zeros(variable_count, dtype=bool)
  solution[interface_indices] = interface_solution
  assigned[interface_indices] = True
  global_hessian = np.zeros((variable_count, variable_count), dtype=float)
  global_gradient = np.zeros(variable_count, dtype=float)
  duplicate_private_count = 0
  private_backsubstitution_count = 0
  local_singular_count = 0

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    gradient = system.get("gradient")
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if hessian is None or gradient is None:
      raise ValueError(f"local system {system_index} is missing hessian/gradient")
    if hasattr(hessian, "toarray"):
      hessian = hessian.toarray()
    hessian = np.asarray(hessian, dtype=float)
    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    if hessian.shape != (variable_count, variable_count):
      raise ValueError(f"local system {system_index} hessian has wrong shape")
    if gradient.shape[0] != variable_count:
      raise ValueError(f"local system {system_index} gradient has wrong dimension")
    global_hessian += hessian
    global_gradient += gradient
    if private_indices.size == 0:
      continue
    if np.any(assigned[private_indices]):
      duplicate_private_count += int(np.count_nonzero(assigned[private_indices]))
      raise ValueError(f"local system {system_index} reuses private variables")
    h_pp = hessian[private_indices[:, None], private_indices]
    if damping > 0.0:
      h_pp = h_pp + float(damping) * np.eye(len(private_indices))
    h_pb = hessian[private_indices[:, None], interface_indices]
    g_p = gradient[private_indices]
    private_rhs = g_p - h_pb @ interface_solution
    try:
      private_solution = np.linalg.solve(h_pp, private_rhs)
    except np.linalg.LinAlgError:
      local_singular_count += 1
      private_solution = np.linalg.pinv(h_pp, rcond=1e-12) @ private_rhs
    solution[private_indices] = private_solution
    assigned[private_indices] = True
    private_backsubstitution_count += 1

  normal_residual = global_hessian @ solution - global_gradient
  return {
      "model": "full_solution_from_interface_backsubstitution",
      "solution": solution,
      "variable_count": int(variable_count),
      "interface_variable_count": int(len(interface_indices)),
      "private_backsubstitution_count": int(private_backsubstitution_count),
      "duplicate_private_variable_count": int(duplicate_private_count),
      "local_singular_count": int(local_singular_count),
      "unassigned_variable_count": int(np.count_nonzero(~assigned)),
      "final_normal_residual": float(np.linalg.norm(normal_residual)),
  }


def _rotations_from_full_stage_solution(
    solution: np.ndarray,
    pose_ids: list[int],
    offsets: dict[int, int],
    dim: int,
    anchor_pose: int,
) -> dict[int, np.ndarray]:
  rotations: dict[int, np.ndarray] = {}
  solution = np.asarray(solution, dtype=float).reshape(-1)
  for pose_id in pose_ids:
    pose_id = int(pose_id)
    if pose_id == int(anchor_pose):
      rotations[pose_id] = np.eye(dim, dtype=float)
      continue
    offset = int(offsets[pose_id])
    raw = solution[offset:offset + dim * dim].reshape((dim, dim))
    rotations[pose_id] = project_rotation_block(raw)
  return rotations


def _translations_from_full_stage_solution(
    solution: np.ndarray,
    pose_ids: list[int],
    offsets: dict[int, int],
    dim: int,
    anchor_pose: int,
) -> dict[int, np.ndarray]:
  translations: dict[int, np.ndarray] = {}
  solution = np.asarray(solution, dtype=float).reshape(-1)
  for pose_id in pose_ids:
    pose_id = int(pose_id)
    if pose_id == int(anchor_pose):
      translations[pose_id] = np.zeros(dim, dtype=float)
      continue
    offset = int(offsets[pose_id])
    translations[pose_id] = solution[offset:offset + dim]
  return translations


def _poses_from_stage_solutions(
    pose_ids: list[int],
    rotations: dict[int, np.ndarray],
    translations: dict[int, np.ndarray],
    dim: int,
) -> dict[int, np.ndarray]:
  poses: dict[int, np.ndarray] = {}
  for pose_id in pose_ids:
    pose_id = int(pose_id)
    pose = np.eye(4, dtype=float)
    pose[:dim, :dim] = rotations[pose_id]
    pose[:dim, 3] = translations[pose_id]
    if dim == 2:
      pose[2, 2] = 1.0
    poses[pose_id] = pose
  return poses


def _pose_set_total_chordal_cost(
    graph_edges: list[Edge],
    poses: dict[int, np.ndarray],
    weighted: bool,
    cost_mode: str,
) -> dict:
  total_cost = 0.0
  evaluated = 0
  missing = 0
  for edge in graph_edges:
    cost, ok = edge_chordal_cost(edge, poses, weighted, cost_mode)
    if ok:
      total_cost += float(cost)
      evaluated += 1
    else:
      missing += 1
  if missing:
    total_cost = float("inf")
  return {
      "total_cost": float(total_cost),
      "evaluated_edge_count": int(evaluated),
      "missing_edge_count": int(missing),
      "all_edges_evaluated": bool(missing == 0),
  }


def run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    max_iterations: int = 3,
    stop_omitted_force_norm: float = 1e-12,
    allow_top_set_promotion: bool = False,
    promotion_policy: str = "top_residual",
    promotion_min_value_per_byte: float = 0.0,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_candidate_top_k: int = 4,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
    interface_solver: str = "promotion",
    pcg_iterations: int = 100,
    pcg_tolerance: float = 1e-10,
    pcg_damping: float = 0.0,
    schur_preconditioner: str = "diagonal",
    pcg_communication_model: str = "separator_owner_star",
    adaptive_coarse_rank: int = 8,
    adaptive_coarse_probe_count: int = 0,
    adaptive_coarse_seed: int = 0,
    component_modes_per_component: int = 1,
    component_max_components: int = 0,
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
    robot_topology_edges: list[tuple[object, object]] | None = None,
) -> tuple[dict[int, np.ndarray], dict]:
  """Recover a pose handoff from two-stage packet-enriched DCI diagnostics."""
  interface_solver = str(interface_solver)
  if interface_solver not in {
      "promotion",
      "pcg",
      "fixed_step",
      "coarse_fixed_step",
      "separator_component_fixed_step",
      "separator_component_residual_krylov_fixed_step",
      "separator_component_geneo_fixed_step",
      "separator_component_geneo_residual_krylov_fixed_step",
      "separator_component_merit_fixed_step",
      "separator_component_merit_residual_krylov_fixed_step",
      "residual_krylov_fixed_step",
      "deterministic_ritz_fixed_step",
  }:
    raise ValueError(f"unsupported interface_solver: {interface_solver}")
  sorted_pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  if not sorted_pose_ids:
    return {}, {
        "model": "two_stage_receiver_pulled_top_residual_handoff_diagnostic",
        "pose_count": 0,
    }
  stage_dim = int(dim) if dim is not None else pose_dimension_from_edges(
      graph_edges)
  anchor_pose = sorted_pose_ids[0] if anchor_pose is None else int(anchor_pose)

  if interface_solver == "pcg":
    rotation_report = run_graph_stage_matrix_free_interface_pcg_solve(
        graph_edges=graph_edges,
        pose_ids=sorted_pose_ids,
        robot_of=robot_of,
        stage="rotation",
        dim=stage_dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
        rotations=None,
        edge_owner_policy=edge_owner_policy,
        hessian_storage=hessian_storage,
        pcg_iterations=pcg_iterations,
        pcg_tolerance=pcg_tolerance,
        pcg_damping=pcg_damping,
        schur_preconditioner=schur_preconditioner,
        communication_model=pcg_communication_model,
        robot_topology_edges=robot_topology_edges,
    )
  elif interface_solver in {
      "fixed_step",
      "coarse_fixed_step",
      "separator_component_fixed_step",
      "separator_component_residual_krylov_fixed_step",
      "separator_component_geneo_fixed_step",
      "separator_component_geneo_residual_krylov_fixed_step",
      "separator_component_merit_fixed_step",
      "separator_component_merit_residual_krylov_fixed_step",
      "residual_krylov_fixed_step",
      "deterministic_ritz_fixed_step"}:
    rotation_report = run_graph_stage_matrix_free_interface_fixed_step_solve(
        graph_edges=graph_edges,
        pose_ids=sorted_pose_ids,
        robot_of=robot_of,
        stage="rotation",
        dim=stage_dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
        rotations=None,
        edge_owner_policy=edge_owner_policy,
        hessian_storage=hessian_storage,
        iterations=pcg_iterations,
        tolerance=pcg_tolerance,
        damping=pcg_damping,
        schur_preconditioner=schur_preconditioner,
        relaxation=1.0,
        coarse_basis_mode=(
            "robot_coordinate" if interface_solver == "coarse_fixed_step"
            else "separator_component_coordinate"
            if interface_solver in {
                "separator_component_fixed_step",
                "separator_component_residual_krylov_fixed_step"}
            else "separator_component_geneo"
            if interface_solver in {
                "separator_component_geneo_fixed_step",
                "separator_component_geneo_residual_krylov_fixed_step"}
            else "separator_component_merit"
            if interface_solver in {
                "separator_component_merit_fixed_step",
                "separator_component_merit_residual_krylov_fixed_step"}
            else "none"),
        adaptive_coarse_basis=(
            "residual_krylov"
            if interface_solver in {
                "residual_krylov_fixed_step",
                "separator_component_residual_krylov_fixed_step",
                "separator_component_geneo_residual_krylov_fixed_step",
                "separator_component_merit_residual_krylov_fixed_step"}
            else "deterministic_ritz"
            if interface_solver == "deterministic_ritz_fixed_step"
            else "none"),
        adaptive_coarse_rank=(
            int(adaptive_coarse_rank)
            if interface_solver in {
                "residual_krylov_fixed_step",
                "separator_component_residual_krylov_fixed_step",
                "separator_component_geneo_residual_krylov_fixed_step",
                "separator_component_merit_residual_krylov_fixed_step",
                "deterministic_ritz_fixed_step"} else 0),
        adaptive_coarse_probe_count=(
            int(adaptive_coarse_probe_count)
            if interface_solver == "deterministic_ritz_fixed_step" else 0),
        adaptive_coarse_seed=(
            int(adaptive_coarse_seed)
            if interface_solver == "deterministic_ritz_fixed_step" else 0),
        component_modes_per_component=component_modes_per_component,
        component_max_components=component_max_components,
        fixed_step_acceleration=fixed_step_acceleration,
        chebyshev_lambda_min=chebyshev_lambda_min,
        chebyshev_lambda_max=chebyshev_lambda_max,
        chebyshev_safety_monitor=chebyshev_safety_monitor,
        chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
        chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
        chebyshev_ritz_seed=chebyshev_ritz_seed,
        chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
        solver_label=interface_solver,
        communication_model=pcg_communication_model,
        robot_topology_edges=robot_topology_edges,
    )
  else:
    rotation_report = run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
        graph_edges=graph_edges,
        pose_ids=sorted_pose_ids,
        robot_of=robot_of,
        stage="rotation",
        basis_mode=basis_mode,
        dim=stage_dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
        rotations=None,
        edge_owner_policy=edge_owner_policy,
        hessian_storage=hessian_storage,
        max_iterations=max_iterations,
        stop_omitted_force_norm=stop_omitted_force_norm,
        allow_top_set_promotion=allow_top_set_promotion,
        promotion_policy=promotion_policy,
        promotion_min_value_per_byte=promotion_min_value_per_byte,
        promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
        promotion_candidate_top_k=promotion_candidate_top_k,
        promotion_payload_bytes_per_coordinate=(
            promotion_payload_bytes_per_coordinate),
        sparse_block_norm_threshold=sparse_block_norm_threshold,
        sparse_payload_index_bytes=sparse_payload_index_bytes,
    )
  if interface_solver in {
      "pcg",
      "fixed_step",
      "coarse_fixed_step",
      "separator_component_fixed_step",
      "separator_component_residual_krylov_fixed_step",
      "separator_component_geneo_fixed_step",
      "separator_component_geneo_residual_krylov_fixed_step",
      "separator_component_merit_fixed_step",
      "separator_component_merit_residual_krylov_fixed_step",
      "residual_krylov_fixed_step",
      "deterministic_ritz_fixed_step",
  }:
    rotation_backsubstitution = _pcg_stage_full_solution_reuse_report(
        rotation_report)
  else:
    rot_local_systems, rot_interface_indices, rot_build_stats = (
        build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=sorted_pose_ids,
            robot_of=robot_of,
            stage="rotation",
            dim=stage_dim,
            weighted=weighted,
            cost_mode=cost_mode,
            anchor_pose=anchor_pose,
            rotations=None,
            edge_owner_policy=edge_owner_policy,
            hessian_storage=hessian_storage,
        )
    )
    rotation_backsubstitution = backsubstitute_full_solution_from_interface(
        local_systems=rot_local_systems,
        interface_indices=rot_interface_indices,
        interface_solution=rotation_report["final_solve"][
            "lifted_interface_solution"],
        variable_count=rot_build_stats["variable_count"],
    )
  _, _, rot_meta = assemble_rotation_system(
      graph_edges=graph_edges,
      pose_ids=sorted_pose_ids,
      dim=stage_dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
  )
  rotations = _rotations_from_full_stage_solution(
      rotation_backsubstitution["solution"],
      sorted_pose_ids,
      rot_meta["offsets"],
      stage_dim,
      anchor_pose,
  )

  if interface_solver == "pcg":
    translation_report = run_graph_stage_matrix_free_interface_pcg_solve(
        graph_edges=graph_edges,
        pose_ids=sorted_pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=stage_dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
        rotations=rotations,
        edge_owner_policy=edge_owner_policy,
        hessian_storage=hessian_storage,
        pcg_iterations=pcg_iterations,
        pcg_tolerance=pcg_tolerance,
        pcg_damping=pcg_damping,
        schur_preconditioner=schur_preconditioner,
        communication_model=pcg_communication_model,
        robot_topology_edges=robot_topology_edges,
    )
  elif interface_solver in {
      "fixed_step",
      "coarse_fixed_step",
      "separator_component_fixed_step",
      "separator_component_residual_krylov_fixed_step",
      "separator_component_geneo_fixed_step",
      "separator_component_geneo_residual_krylov_fixed_step",
      "separator_component_merit_fixed_step",
      "separator_component_merit_residual_krylov_fixed_step",
      "residual_krylov_fixed_step",
      "deterministic_ritz_fixed_step"}:
    translation_report = run_graph_stage_matrix_free_interface_fixed_step_solve(
        graph_edges=graph_edges,
        pose_ids=sorted_pose_ids,
        robot_of=robot_of,
        stage="translation",
        dim=stage_dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
        rotations=rotations,
        edge_owner_policy=edge_owner_policy,
        hessian_storage=hessian_storage,
        iterations=pcg_iterations,
        tolerance=pcg_tolerance,
        damping=pcg_damping,
        schur_preconditioner=schur_preconditioner,
        relaxation=1.0,
        coarse_basis_mode=(
            "robot_coordinate" if interface_solver == "coarse_fixed_step"
            else "separator_component_coordinate"
            if interface_solver in {
                "separator_component_fixed_step",
                "separator_component_residual_krylov_fixed_step"}
            else "separator_component_geneo"
            if interface_solver in {
                "separator_component_geneo_fixed_step",
                "separator_component_geneo_residual_krylov_fixed_step"}
            else "separator_component_merit"
            if interface_solver in {
                "separator_component_merit_fixed_step",
                "separator_component_merit_residual_krylov_fixed_step"}
            else "none"),
        adaptive_coarse_basis=(
            "residual_krylov"
            if interface_solver in {
                "residual_krylov_fixed_step",
                "separator_component_residual_krylov_fixed_step",
                "separator_component_geneo_residual_krylov_fixed_step",
                "separator_component_merit_residual_krylov_fixed_step"}
            else "deterministic_ritz"
            if interface_solver == "deterministic_ritz_fixed_step"
            else "none"),
        adaptive_coarse_rank=(
            int(adaptive_coarse_rank)
            if interface_solver in {
                "residual_krylov_fixed_step",
                "separator_component_residual_krylov_fixed_step",
                "separator_component_geneo_residual_krylov_fixed_step",
                "separator_component_merit_residual_krylov_fixed_step",
                "deterministic_ritz_fixed_step"} else 0),
        adaptive_coarse_probe_count=(
            int(adaptive_coarse_probe_count)
            if interface_solver == "deterministic_ritz_fixed_step" else 0),
        adaptive_coarse_seed=(
            int(adaptive_coarse_seed)
            if interface_solver == "deterministic_ritz_fixed_step" else 0),
        component_modes_per_component=component_modes_per_component,
        component_max_components=component_max_components,
        fixed_step_acceleration=fixed_step_acceleration,
        chebyshev_lambda_min=chebyshev_lambda_min,
        chebyshev_lambda_max=chebyshev_lambda_max,
        chebyshev_safety_monitor=chebyshev_safety_monitor,
        chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
        chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
        chebyshev_ritz_seed=chebyshev_ritz_seed,
        chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
        solver_label=interface_solver,
        communication_model=pcg_communication_model,
        robot_topology_edges=robot_topology_edges,
    )
  else:
    translation_report = run_graph_stage_receiver_pulled_top_residual_enrichment_iterations(
        graph_edges=graph_edges,
        pose_ids=sorted_pose_ids,
        robot_of=robot_of,
        stage="translation",
        basis_mode=basis_mode,
        dim=stage_dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
        rotations=rotations,
        edge_owner_policy=edge_owner_policy,
        hessian_storage=hessian_storage,
        max_iterations=max_iterations,
        stop_omitted_force_norm=stop_omitted_force_norm,
        allow_top_set_promotion=allow_top_set_promotion,
        promotion_policy=promotion_policy,
        promotion_min_value_per_byte=promotion_min_value_per_byte,
        promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
        promotion_candidate_top_k=promotion_candidate_top_k,
        promotion_payload_bytes_per_coordinate=(
            promotion_payload_bytes_per_coordinate),
        sparse_block_norm_threshold=sparse_block_norm_threshold,
        sparse_payload_index_bytes=sparse_payload_index_bytes,
    )
  if interface_solver in {
      "pcg",
      "fixed_step",
      "coarse_fixed_step",
      "separator_component_fixed_step",
      "separator_component_residual_krylov_fixed_step",
      "separator_component_geneo_fixed_step",
      "separator_component_geneo_residual_krylov_fixed_step",
      "separator_component_merit_fixed_step",
      "separator_component_merit_residual_krylov_fixed_step",
      "residual_krylov_fixed_step",
      "deterministic_ritz_fixed_step",
  }:
    translation_backsubstitution = _pcg_stage_full_solution_reuse_report(
        translation_report)
  else:
    trans_local_systems, trans_interface_indices, trans_build_stats = (
        build_graph_local_normal_systems_for_interface_schur(
            graph_edges=graph_edges,
            pose_ids=sorted_pose_ids,
            robot_of=robot_of,
            stage="translation",
            dim=stage_dim,
            weighted=weighted,
            cost_mode=cost_mode,
            anchor_pose=anchor_pose,
            rotations=rotations,
            edge_owner_policy=edge_owner_policy,
            hessian_storage=hessian_storage,
        )
    )
    translation_backsubstitution = backsubstitute_full_solution_from_interface(
        local_systems=trans_local_systems,
        interface_indices=trans_interface_indices,
        interface_solution=translation_report["final_solve"][
            "lifted_interface_solution"],
        variable_count=trans_build_stats["variable_count"],
    )
  _, _, trans_meta = assemble_translation_system(
      graph_edges=graph_edges,
      pose_ids=sorted_pose_ids,
      rotations=rotations,
      dim=stage_dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
  )
  translations = _translations_from_full_stage_solution(
      translation_backsubstitution["solution"],
      sorted_pose_ids,
      trans_meta["offsets"],
      stage_dim,
      anchor_pose,
  )
  poses = _poses_from_stage_solutions(
      sorted_pose_ids,
      rotations,
      translations,
      stage_dim,
  )
  handoff_cost = _pose_set_total_chordal_cost(
      graph_edges=graph_edges,
      poses=poses,
      weighted=weighted,
      cost_mode=cost_mode,
  )
  centralized_poses, centralized_stats = solve_centralized_chordal_initialization(
      graph_edges=graph_edges,
      pose_ids=sorted_pose_ids,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
  )
  centralized_handoff_cost = _pose_set_total_chordal_cost(
      graph_edges=graph_edges,
      poses=centralized_poses,
      weighted=weighted,
      cost_mode=cost_mode,
  )
  rotation_projection_safety = rotation_projection_safety_certificate(
      solution=rotation_backsubstitution["solution"],
      pose_ids=sorted_pose_ids,
      offsets=rot_meta["offsets"],
      dim=stage_dim,
      anchor_pose=anchor_pose,
  )
  rotation_projection_cost = rotation_projection_cost_certificate(
      graph_edges=graph_edges,
      solution=rotation_backsubstitution["solution"],
      pose_ids=sorted_pose_ids,
      offsets=rot_meta["offsets"],
      translations=translations,
      dim=stage_dim,
      anchor_pose=anchor_pose,
      weighted=weighted,
      cost_mode=cost_mode,
  )
  stage_reports = {
      "rotation": rotation_report,
      "translation": translation_report,
  }
  stage_summary = {
      "model": "two_stage_receiver_pulled_top_residual_enrichment_summary",
      "stage_order": ["rotation", "translation"],
      "basis_mode": basis_mode,
      "interface_solver": interface_solver,
      "adaptive_coarse_rank": int(adaptive_coarse_rank),
      "adaptive_coarse_probe_count": int(adaptive_coarse_probe_count),
      "adaptive_coarse_seed": int(adaptive_coarse_seed),
      "component_modes_per_component": int(component_modes_per_component),
      "component_max_components": int(component_max_components),
      "fixed_step_acceleration": str(fixed_step_acceleration),
      "chebyshev_lambda_min": (
          None if chebyshev_lambda_min is None
          else float(chebyshev_lambda_min)),
      "chebyshev_lambda_max": (
          None if chebyshev_lambda_max is None
          else float(chebyshev_lambda_max)),
      "chebyshev_safety_monitor": str(chebyshev_safety_monitor),
      "chebyshev_safety_growth_factor": float(
          chebyshev_safety_growth_factor),
      "promotion_policy": str(promotion_policy),
      "promotion_candidate_top_k": int(promotion_candidate_top_k),
      "all_stages_converged": bool(
          rotation_report["converged"] and translation_report["converged"]),
      "rotation_converged": bool(rotation_report["converged"]),
      "translation_converged": bool(translation_report["converged"]),
      "rotation_final_omitted_force_norm": float(
          rotation_report["final_omitted_force_norm"]),
      "translation_final_omitted_force_norm": float(
          translation_report["final_omitted_force_norm"]),
      "stages": stage_reports,
      "total_top_residual_payload_bytes": int(sum(
          int(report["total_top_residual_payload_bytes"])
          for report in stage_reports.values())),
      "total_promotion_payload_bytes": int(sum(
          int(report.get("total_promotion_payload_bytes", 0))
          for report in stage_reports.values())),
      "total_full_packet_payload_bytes": int(sum(
          int(report["total_full_packet_payload_bytes"])
          for report in stage_reports.values())),
      "total_sparse_packet_payload_bytes": int(sum(
          int(report["total_sparse_packet_payload_bytes"])
          for report in stage_reports.values())),
      "total_pcg_comm_mb": float(sum(
          float(report.get("total_pcg_comm_mb", 0.0))
          for report in stage_reports.values())),
      "total_pcg_comm_bytes": int(sum(
          int(report.get("total_pcg_comm_bytes", 0))
          for report in stage_reports.values())),
  }
  return poses, {
      "model": "two_stage_receiver_pulled_top_residual_handoff_diagnostic",
      "pose_count": int(len(sorted_pose_ids)),
      "dimension": int(stage_dim),
      "anchor_pose": int(anchor_pose),
      "basis_mode": basis_mode,
      "interface_solver": interface_solver,
      "promotion_policy": str(promotion_policy),
      "promotion_candidate_top_k": int(promotion_candidate_top_k),
      "pcg_iterations": int(pcg_iterations),
      "pcg_tolerance": float(pcg_tolerance),
      "pcg_damping": float(pcg_damping),
      "schur_preconditioner": schur_preconditioner,
      "pcg_communication_model": pcg_communication_model,
      "adaptive_coarse_rank": int(adaptive_coarse_rank),
      "adaptive_coarse_probe_count": int(adaptive_coarse_probe_count),
      "adaptive_coarse_seed": int(adaptive_coarse_seed),
      "component_modes_per_component": int(component_modes_per_component),
      "component_max_components": int(component_max_components),
      "fixed_step_acceleration": str(fixed_step_acceleration),
      "chebyshev_lambda_min": (
          None if chebyshev_lambda_min is None
          else float(chebyshev_lambda_min)),
      "chebyshev_lambda_max": (
          None if chebyshev_lambda_max is None
          else float(chebyshev_lambda_max)),
      "chebyshev_safety_monitor": str(chebyshev_safety_monitor),
      "chebyshev_safety_growth_factor": float(
          chebyshev_safety_growth_factor),
      "weighted": bool(weighted),
      "cost_mode": cost_mode,
      "stage_summary": stage_summary,
      "rotation_backsubstitution": rotation_backsubstitution,
      "translation_backsubstitution": translation_backsubstitution,
      "rotation_projection_safety": rotation_projection_safety,
      "rotation_projection_cost": rotation_projection_cost,
      "handoff_cost": handoff_cost,
      "centralized_handoff_cost": centralized_handoff_cost,
      "centralized_stats": centralized_stats,
      "handoff_cost_delta_to_centralized": float(
          handoff_cost["total_cost"] - centralized_handoff_cost["total_cost"]),
  }


def _estimated_stage_interface_variables(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    dim: int,
    anchor_pose: int,
) -> dict:
  pose_set = {int(pose_id) for pose_id in pose_ids}
  interface_pose_ids: set[int] = set()
  separator_edge_count = 0
  for edge in graph_edges:
    if int(edge.i) not in pose_set or int(edge.j) not in pose_set:
      continue
    if int(edge.i) not in robot_of or int(edge.j) not in robot_of:
      continue
    if int(robot_of[int(edge.i)]) == int(robot_of[int(edge.j)]):
      continue
    separator_edge_count += 1
    interface_pose_ids.add(int(edge.i))
    interface_pose_ids.add(int(edge.j))
  variable_interface_pose_ids = sorted(
      pose_id for pose_id in interface_pose_ids if int(pose_id) != int(anchor_pose))
  return {
      "estimated_interface_pose_count": int(len(interface_pose_ids)),
      "estimated_variable_interface_pose_count": int(
          len(variable_interface_pose_ids)),
      "estimated_rotation_interface_variables": int(
          len(variable_interface_pose_ids) * int(dim) * int(dim)),
      "estimated_translation_interface_variables": int(
          len(variable_interface_pose_ids) * int(dim)),
      "separator_edge_count": int(separator_edge_count),
      "estimated_interface_pose_ids": sorted(interface_pose_ids),
  }


def _json_safe(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, dict):
    return {str(key): _json_safe(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(item) for item in value]
  return value


def _compact_stage_progress(prefix: str, stage_report: dict) -> dict:
  history = list(stage_report.get("history", []))
  nonmonotone = 0
  best_norm = float(stage_report.get("initial_omitted_force_norm", 0.0))
  for item in history:
    before = float(item.get("before_omitted_force_norm", best_norm))
    after = float(item.get("after_omitted_force_norm", before))
    best_norm = min(best_norm, before, after)
    if after > before + 1e-12:
      nonmonotone += 1
  final_norm = float(stage_report.get("final_omitted_force_norm", best_norm))
  best_norm = min(best_norm, final_norm)
  full_count = int(stage_report.get("full_interface_variable_count", 0))
  basis_count = int(stage_report.get("final_basis_column_count", 0))
  return {
      f"{prefix}_promoted_pose_count": int(
          len(stage_report.get("promoted_pose_ids", []))),
      f"{prefix}_iteration_count": int(stage_report.get("iteration_count", 0)),
      f"{prefix}_stopped_reason": str(stage_report.get("stopped_reason", "")),
      f"{prefix}_nonmonotone_iteration_count": int(nonmonotone),
      f"{prefix}_best_omitted_force_norm": float(best_norm),
      f"{prefix}_final_over_best_omitted_force_ratio": float(
          final_norm / best_norm if best_norm > 0.0 else 0.0),
      f"{prefix}_final_basis_column_count": int(basis_count),
      f"{prefix}_full_interface_variable_count": int(full_count),
      f"{prefix}_basis_coverage_ratio": float(
          basis_count / full_count if full_count > 0 else 0.0),
  }


def packet_handoff_sweep_row(
    dataset: str,
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    anchor_pose: int | None = None,
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    max_iterations: int = 3,
    stop_omitted_force_norm: float = 1e-12,
    allow_top_set_promotion: bool = False,
    promotion_policy: str = "top_residual",
    promotion_min_value_per_byte: float = 0.0,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_candidate_top_k: int = 4,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
    max_interface_variables: int | None = None,
    interface_solver: str = "promotion",
    pcg_iterations: int = 100,
    pcg_tolerance: float = 1e-10,
    pcg_damping: float = 0.0,
    schur_preconditioner: str = "diagonal",
    pcg_communication_model: str = "separator_owner_star",
    adaptive_coarse_rank: int = 8,
    adaptive_coarse_probe_count: int = 0,
    adaptive_coarse_seed: int = 0,
    component_modes_per_component: int = 1,
    component_max_components: int = 0,
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
    robot_topology_edges: list[tuple[object, object]] | None = None,
    catch_errors: bool = True,
) -> dict:
  """Build one flat packet-enriched DCI handoff sweep row."""
  sorted_pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  stage_dim = int(dim) if dim is not None else pose_dimension_from_edges(
      graph_edges)
  anchor_pose = (
      sorted_pose_ids[0] if anchor_pose is None and sorted_pose_ids
      else int(anchor_pose) if anchor_pose is not None else 0)
  estimate = _estimated_stage_interface_variables(
      graph_edges=graph_edges,
      pose_ids=sorted_pose_ids,
      robot_of=robot_of,
      dim=stage_dim,
      anchor_pose=anchor_pose,
  )
  base_row = {
      "model": "packet_handoff_sweep_row",
      "dataset": str(dataset),
      "status": "ok",
      "pose_count": int(len(sorted_pose_ids)),
      "edge_count": int(len(graph_edges)),
      "dimension": int(stage_dim),
      "anchor_pose": int(anchor_pose),
      "basis_mode": basis_mode,
      "weighted": bool(weighted),
      "cost_mode": cost_mode,
      "edge_owner_policy": edge_owner_policy,
      "hessian_storage": hessian_storage,
      "interface_solver": str(interface_solver),
      "max_iterations": int(max_iterations),
      "stop_omitted_force_norm": float(stop_omitted_force_norm),
      "allow_top_set_promotion": bool(allow_top_set_promotion),
      "promotion_policy": str(promotion_policy),
      "promotion_min_value_per_byte": float(promotion_min_value_per_byte),
      "promotion_max_omitted_force_increase": float(
          promotion_max_omitted_force_increase),
      "promotion_candidate_top_k": int(promotion_candidate_top_k),
      "pcg_iterations": int(pcg_iterations),
      "pcg_tolerance": float(pcg_tolerance),
      "pcg_damping": float(pcg_damping),
      "schur_preconditioner": schur_preconditioner,
      "pcg_communication_model": pcg_communication_model,
      "adaptive_coarse_rank": int(adaptive_coarse_rank),
      "adaptive_coarse_probe_count": int(adaptive_coarse_probe_count),
      "adaptive_coarse_seed": int(adaptive_coarse_seed),
      "component_modes_per_component": int(component_modes_per_component),
      "component_max_components": int(component_max_components),
      "fixed_step_acceleration": str(fixed_step_acceleration),
      "chebyshev_lambda_min": (
          None if chebyshev_lambda_min is None
          else float(chebyshev_lambda_min)),
      "chebyshev_lambda_max": (
          None if chebyshev_lambda_max is None
          else float(chebyshev_lambda_max)),
      "chebyshev_safety_monitor": str(chebyshev_safety_monitor),
      "chebyshev_safety_growth_factor": float(
          chebyshev_safety_growth_factor),
      "chebyshev_ritz_probe_iterations": int(
          chebyshev_ritz_probe_iterations),
      "chebyshev_ritz_seed": int(chebyshev_ritz_seed),
      "chebyshev_ritz_safety_factor": float(
          chebyshev_ritz_safety_factor),
      **estimate,
  }
  max_estimated_interface_variables = max(
      int(estimate["estimated_rotation_interface_variables"]),
      int(estimate["estimated_translation_interface_variables"]),
  )
  base_row["estimated_max_interface_variables"] = int(
      max_estimated_interface_variables)
  if (max_interface_variables is not None and
      max_estimated_interface_variables > int(max_interface_variables)):
    return {
        **base_row,
        "status": "skipped_preflight",
        "skip_reason": "interface_variable_count_exceeds_limit",
        "max_interface_variables": int(max_interface_variables),
    }

  try:
    _, report = run_two_stage_receiver_pulled_top_residual_handoff_diagnostic(
        graph_edges=graph_edges,
        pose_ids=sorted_pose_ids,
        robot_of=robot_of,
        basis_mode=basis_mode,
        dim=stage_dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
        edge_owner_policy=edge_owner_policy,
        hessian_storage=hessian_storage,
        max_iterations=max_iterations,
        stop_omitted_force_norm=stop_omitted_force_norm,
        allow_top_set_promotion=allow_top_set_promotion,
        promotion_policy=promotion_policy,
        promotion_min_value_per_byte=promotion_min_value_per_byte,
        promotion_max_omitted_force_increase=promotion_max_omitted_force_increase,
        promotion_candidate_top_k=promotion_candidate_top_k,
        promotion_payload_bytes_per_coordinate=(
            promotion_payload_bytes_per_coordinate),
        sparse_block_norm_threshold=sparse_block_norm_threshold,
        sparse_payload_index_bytes=sparse_payload_index_bytes,
        interface_solver=interface_solver,
        pcg_iterations=pcg_iterations,
        pcg_tolerance=pcg_tolerance,
        pcg_damping=pcg_damping,
        schur_preconditioner=schur_preconditioner,
        pcg_communication_model=pcg_communication_model,
        adaptive_coarse_rank=adaptive_coarse_rank,
        adaptive_coarse_probe_count=adaptive_coarse_probe_count,
        adaptive_coarse_seed=adaptive_coarse_seed,
        component_modes_per_component=component_modes_per_component,
        component_max_components=component_max_components,
        fixed_step_acceleration=fixed_step_acceleration,
        chebyshev_lambda_min=chebyshev_lambda_min,
        chebyshev_lambda_max=chebyshev_lambda_max,
        chebyshev_safety_monitor=chebyshev_safety_monitor,
        chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
        chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
        chebyshev_ritz_seed=chebyshev_ritz_seed,
        chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
        robot_topology_edges=robot_topology_edges,
    )
  except Exception as exc:
    if not catch_errors:
      raise
    return {
        **base_row,
        "status": "failed",
        "failure_type": type(exc).__name__,
        "failure_message": str(exc),
    }

  stage_summary = report["stage_summary"]
  rotation_stage = stage_summary["stages"]["rotation"]
  translation_stage = stage_summary["stages"]["translation"]
  handoff = report["handoff_cost"]
  centralized = report["centralized_handoff_cost"]
  delta = float(report["handoff_cost_delta_to_centralized"])
  return {
      **base_row,
      "status": "ok",
      "all_stages_converged": bool(stage_summary["all_stages_converged"]),
      "rotation_converged": bool(stage_summary["rotation_converged"]),
      "translation_converged": bool(stage_summary["translation_converged"]),
      "rotation_final_omitted_force_norm": float(
          stage_summary["rotation_final_omitted_force_norm"]),
      "translation_final_omitted_force_norm": float(
          stage_summary["translation_final_omitted_force_norm"]),
      **_compact_stage_progress("rotation", rotation_stage),
      **_compact_stage_progress("translation", translation_stage),
      "rotation_normal_residual": float(
          report["rotation_backsubstitution"]["final_normal_residual"]),
      "translation_normal_residual": float(
          report["translation_backsubstitution"]["final_normal_residual"]),
      "handoff_cost": float(handoff["total_cost"]),
      "centralized_handoff_cost": float(centralized["total_cost"]),
      "handoff_cost_delta_to_centralized": delta,
      "handoff_cost_delta_to_centralized_abs": float(abs(delta)),
      "all_edges_evaluated": bool(handoff["all_edges_evaluated"]),
      "evaluated_edge_count": int(handoff["evaluated_edge_count"]),
      "missing_edge_count": int(handoff["missing_edge_count"]),
      "rotation_projection_max_correction_norm": float(
          report["rotation_projection_safety"][
              "max_projection_correction_norm"]),
      "rotation_projection_cost_delta": float(
          report["rotation_projection_cost"]["projected_minus_raw_cost_delta"]),
      "rotation_abs_projection_cost_delta": float(
          report["rotation_projection_cost"]["absolute_projection_cost_delta"]),
      "total_top_residual_payload_bytes": int(
          stage_summary["total_top_residual_payload_bytes"]),
      "total_promotion_payload_bytes": int(
          stage_summary["total_promotion_payload_bytes"]),
      "total_full_packet_payload_bytes": int(
          stage_summary["total_full_packet_payload_bytes"]),
      "total_sparse_packet_payload_bytes": int(
          stage_summary["total_sparse_packet_payload_bytes"]),
      "total_pcg_comm_mb": float(stage_summary.get("total_pcg_comm_mb", 0.0)),
      "total_pcg_comm_bytes": int(
          stage_summary.get("total_pcg_comm_bytes", 0)),
      "report": report,
  }


def write_packet_handoff_sweep(
    output_dir: Path,
    dataset_specs: list[dict],
    basis_mode: str,
    dim: int | None = None,
    weighted: bool = False,
    cost_mode: str = "dpgo",
    edge_owner_policy: str = "lower_robot",
    hessian_storage: str = "dense",
    max_iterations: int = 3,
    stop_omitted_force_norm: float = 1e-12,
    allow_top_set_promotion: bool = False,
    promotion_policy: str = "top_residual",
    promotion_min_value_per_byte: float = 0.0,
    promotion_max_omitted_force_increase: float = 0.0,
    promotion_candidate_top_k: int = 4,
    promotion_payload_bytes_per_coordinate: int = 8,
    sparse_block_norm_threshold: float = 0.0,
    sparse_payload_index_bytes: int = 4,
    max_interface_variables: int | None = None,
    interface_solver: str = "promotion",
    pcg_iterations: int = 100,
    pcg_tolerance: float = 1e-10,
    pcg_damping: float = 0.0,
    schur_preconditioner: str = "diagonal",
    pcg_communication_model: str = "separator_owner_star",
    adaptive_coarse_rank: int = 8,
    adaptive_coarse_probe_count: int = 0,
    adaptive_coarse_seed: int = 0,
    component_modes_per_component: int = 1,
    component_max_components: int = 0,
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
) -> dict:
  """Write CSV/JSON rows for packet-enriched DCI handoff diagnostics."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  rows: list[dict] = []
  full_reports: list[dict] = []
  for spec in dataset_specs:
    row = packet_handoff_sweep_row(
        dataset=str(spec["dataset"]),
        graph_edges=spec["graph_edges"],
        pose_ids=spec["pose_ids"],
        robot_of=spec["robot_of"],
        basis_mode=spec.get("basis_mode", basis_mode),
        dim=spec.get("dim", dim),
        weighted=spec.get("weighted", weighted),
        cost_mode=spec.get("cost_mode", cost_mode),
        anchor_pose=spec.get("anchor_pose"),
        edge_owner_policy=spec.get("edge_owner_policy", edge_owner_policy),
        hessian_storage=spec.get("hessian_storage", hessian_storage),
        max_iterations=spec.get("max_iterations", max_iterations),
        stop_omitted_force_norm=spec.get(
            "stop_omitted_force_norm", stop_omitted_force_norm),
        allow_top_set_promotion=spec.get(
            "allow_top_set_promotion", allow_top_set_promotion),
        promotion_policy=spec.get("promotion_policy", promotion_policy),
        promotion_min_value_per_byte=spec.get(
            "promotion_min_value_per_byte", promotion_min_value_per_byte),
        promotion_max_omitted_force_increase=spec.get(
            "promotion_max_omitted_force_increase",
            promotion_max_omitted_force_increase),
        promotion_candidate_top_k=spec.get(
            "promotion_candidate_top_k", promotion_candidate_top_k),
        promotion_payload_bytes_per_coordinate=spec.get(
            "promotion_payload_bytes_per_coordinate",
            promotion_payload_bytes_per_coordinate),
        sparse_block_norm_threshold=spec.get(
            "sparse_block_norm_threshold", sparse_block_norm_threshold),
        sparse_payload_index_bytes=spec.get(
            "sparse_payload_index_bytes", sparse_payload_index_bytes),
        max_interface_variables=spec.get(
            "max_interface_variables", max_interface_variables),
        interface_solver=spec.get("interface_solver", interface_solver),
        pcg_iterations=spec.get("pcg_iterations", pcg_iterations),
        pcg_tolerance=spec.get("pcg_tolerance", pcg_tolerance),
        pcg_damping=spec.get("pcg_damping", pcg_damping),
        schur_preconditioner=spec.get(
            "schur_preconditioner", schur_preconditioner),
        pcg_communication_model=spec.get(
            "pcg_communication_model", pcg_communication_model),
        adaptive_coarse_rank=spec.get(
            "adaptive_coarse_rank", adaptive_coarse_rank),
        adaptive_coarse_probe_count=spec.get(
            "adaptive_coarse_probe_count", adaptive_coarse_probe_count),
        adaptive_coarse_seed=spec.get(
            "adaptive_coarse_seed", adaptive_coarse_seed),
        component_modes_per_component=spec.get(
            "component_modes_per_component",
            component_modes_per_component),
        component_max_components=spec.get(
            "component_max_components", component_max_components),
        fixed_step_acceleration=spec.get(
            "fixed_step_acceleration", fixed_step_acceleration),
        chebyshev_lambda_min=spec.get(
            "chebyshev_lambda_min", chebyshev_lambda_min),
        chebyshev_lambda_max=spec.get(
            "chebyshev_lambda_max", chebyshev_lambda_max),
        chebyshev_safety_monitor=spec.get(
            "chebyshev_safety_monitor", chebyshev_safety_monitor),
        chebyshev_safety_growth_factor=spec.get(
            "chebyshev_safety_growth_factor",
            chebyshev_safety_growth_factor),
        chebyshev_ritz_probe_iterations=spec.get(
            "chebyshev_ritz_probe_iterations",
            chebyshev_ritz_probe_iterations),
        chebyshev_ritz_seed=spec.get(
            "chebyshev_ritz_seed", chebyshev_ritz_seed),
        chebyshev_ritz_safety_factor=spec.get(
            "chebyshev_ritz_safety_factor",
            chebyshev_ritz_safety_factor),
        robot_topology_edges=spec.get("robot_topology_edges"),
    )
    full_report = row.pop("report", None)
    rows.append(row)
    if full_report is not None:
      full_reports.append({
          "dataset": row["dataset"],
          "report": _json_safe(full_report),
      })
  csv_path = output_dir / "packet_handoff_sweep_summary.csv"
  if rows:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)
  else:
    csv_path.write_text("", encoding="utf-8")
  report = {
      "model": "packet_handoff_sweep",
      "row_count": int(len(rows)),
      "basis_mode": basis_mode,
      "weighted": bool(weighted),
      "cost_mode": cost_mode,
      "interface_solver": str(interface_solver),
      "max_iterations": int(max_iterations),
      "stop_omitted_force_norm": float(stop_omitted_force_norm),
      "allow_top_set_promotion": bool(allow_top_set_promotion),
      "promotion_policy": str(promotion_policy),
      "promotion_min_value_per_byte": float(promotion_min_value_per_byte),
      "promotion_max_omitted_force_increase": float(
          promotion_max_omitted_force_increase),
      "promotion_candidate_top_k": int(promotion_candidate_top_k),
      "pcg_iterations": int(pcg_iterations),
      "pcg_tolerance": float(pcg_tolerance),
      "pcg_damping": float(pcg_damping),
      "schur_preconditioner": schur_preconditioner,
      "pcg_communication_model": pcg_communication_model,
      "adaptive_coarse_rank": int(adaptive_coarse_rank),
      "adaptive_coarse_probe_count": int(adaptive_coarse_probe_count),
      "adaptive_coarse_seed": int(adaptive_coarse_seed),
      "component_modes_per_component": int(component_modes_per_component),
      "component_max_components": int(component_max_components),
      "fixed_step_acceleration": str(fixed_step_acceleration),
      "chebyshev_lambda_min": (
          None if chebyshev_lambda_min is None
          else float(chebyshev_lambda_min)),
      "chebyshev_lambda_max": (
          None if chebyshev_lambda_max is None
          else float(chebyshev_lambda_max)),
      "chebyshev_safety_monitor": str(chebyshev_safety_monitor),
      "chebyshev_safety_growth_factor": float(
          chebyshev_safety_growth_factor),
      "chebyshev_ritz_probe_iterations": int(
          chebyshev_ritz_probe_iterations),
      "chebyshev_ritz_seed": int(chebyshev_ritz_seed),
      "chebyshev_ritz_safety_factor": float(
          chebyshev_ritz_safety_factor),
      "max_interface_variables": (
          None if max_interface_variables is None
          else int(max_interface_variables)),
      "rows": rows,
      "full_reports": full_reports,
  }
  json_path = output_dir / "packet_handoff_sweep_report.json"
  json_path.write_text(
      json.dumps(_json_safe(report), indent=2, sort_keys=True),
      encoding="utf-8")
  return report


def project_local_interface_schur_to_skeleton(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_pose_ids: Iterable[int],
    block_dim: int,
    skeleton_pose_ids: Iterable[int],
    variable_count: int | None = None,
    damping: float = 0.0,
    anchor_pose: int | None = None,
) -> dict:
  """Project the exact local-Schur interface system onto skeleton pose blocks.

  This computes ``Z^T S Z`` and ``Z^T b`` using the existing local-Schur
  reference. It is a diagnostic bridge from the full-interface CCI reference to
  the new basin-skeleton route; it does not solve or tune an initializer.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  full_schur, full_rhs, schur_stats = sum_local_interface_schur_contributions(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping,
  )
  basis, selected_pose_ids = _pose_block_selection_basis(
      interface_pose_ids=interface_pose_ids,
      interface_variable_count=len(interface_indices),
      block_dim=block_dim,
      skeleton_pose_ids=skeleton_pose_ids,
      anchor_pose=anchor_pose,
  )
  reduced_schur = basis.T @ full_schur @ basis
  reduced_schur = 0.5 * (reduced_schur + reduced_schur.T)
  reduced_rhs = basis.T @ full_rhs
  return {
      "model": "skeleton_reduced_interface_schur",
      "basis": basis,
      "basis_column_count": int(basis.shape[1]),
      "basis_row_count": int(basis.shape[0]),
      "selected_interface_pose_ids": selected_pose_ids,
      "reduced_schur": reduced_schur,
      "reduced_rhs": reduced_rhs,
      "full_schur": full_schur,
      "full_rhs": full_rhs,
      "full_schur_shape": [int(full_schur.shape[0]), int(full_schur.shape[1])],
      "block_dim": int(block_dim),
      "schur_stats": schur_stats,
  }


def solve_skeleton_reduced_interface_schur_with_basis(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_pose_ids: Iterable[int],
    block_dim: int,
    basis: np.ndarray,
    basis_metadata: dict | None = None,
    selected_interface_pose_ids: Iterable[int] = (),
    variable_count: int | None = None,
    damping: float = 0.0,
    anchor_pose: int | None = None,
) -> dict:
  """Solve a reduced interface Schur system defined by an arbitrary basis."""
  projected = project_local_interface_schur_with_basis(
      local_systems=local_systems,
      interface_indices=interface_indices,
      basis=basis,
      basis_metadata=basis_metadata,
      variable_count=variable_count,
      damping=damping,
  )
  reduced_schur = projected["reduced_schur"]
  reduced_rhs = projected["reduced_rhs"]
  basis_array = projected["basis"]
  full_schur = projected["full_schur"]
  full_rhs = projected["full_rhs"]
  if reduced_rhs.size == 0:
    reduced_solution = np.zeros(0, dtype=float)
    reduced_singular = False
  else:
    try:
      reduced_solution = np.linalg.solve(reduced_schur, reduced_rhs)
      reduced_singular = False
    except np.linalg.LinAlgError:
      reduced_solution = np.linalg.pinv(reduced_schur, rcond=1e-12) @ reduced_rhs
      reduced_singular = True
  lifted_solution = basis_array @ reduced_solution
  omitted_force = full_rhs - full_schur @ lifted_solution
  selected_pose_ids = [
      int(pose_id)
      for pose_id in selected_interface_pose_ids
  ]
  omitted_force_blocks = rank_omitted_force_blocks(
      omitted_force=omitted_force,
      interface_pose_ids=interface_pose_ids,
      block_dim=block_dim,
      selected_interface_pose_ids=selected_pose_ids,
      anchor_pose=anchor_pose,
  )
  try:
    full_reference_solution = np.linalg.solve(full_schur, full_rhs)
    full_singular = False
  except np.linalg.LinAlgError:
    full_reference_solution = np.linalg.pinv(full_schur, rcond=1e-12) @ full_rhs
    full_singular = True
  solution_error = lifted_solution - full_reference_solution
  return {
      **projected,
      "model": "basis_reduced_interface_schur_solve",
      "selected_interface_pose_ids": selected_pose_ids,
      "reduced_solution": reduced_solution,
      "lifted_interface_solution": lifted_solution,
      "full_reference_solution": full_reference_solution,
      "reduced_singular": bool(reduced_singular),
      "full_singular": bool(full_singular),
      "omitted_force": omitted_force,
      "omitted_force_norm": float(np.linalg.norm(omitted_force)),
      "omitted_force_blocks": omitted_force_blocks,
      "interface_solution_error_norm": float(np.linalg.norm(solution_error)),
      "reduced_residual_norm": float(
          np.linalg.norm(reduced_schur @ reduced_solution - reduced_rhs)
          if reduced_rhs.size else 0.0),
  }


def solve_skeleton_reduced_interface_schur(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_pose_ids: Iterable[int],
    block_dim: int,
    skeleton_pose_ids: Iterable[int],
    variable_count: int | None = None,
    damping: float = 0.0,
    anchor_pose: int | None = None,
) -> dict:
  """Solve the reduced skeleton Schur system and lift it to full interface."""
  projected = project_local_interface_schur_to_skeleton(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_pose_ids=interface_pose_ids,
      block_dim=block_dim,
      skeleton_pose_ids=skeleton_pose_ids,
      variable_count=variable_count,
      damping=damping,
      anchor_pose=anchor_pose,
  )
  reduced_schur = projected["reduced_schur"]
  reduced_rhs = projected["reduced_rhs"]
  basis = projected["basis"]
  full_schur = projected["full_schur"]
  full_rhs = projected["full_rhs"]
  if reduced_rhs.size == 0:
    reduced_solution = np.zeros(0, dtype=float)
    reduced_singular = False
  else:
    try:
      reduced_solution = np.linalg.solve(reduced_schur, reduced_rhs)
      reduced_singular = False
    except np.linalg.LinAlgError:
      reduced_solution = np.linalg.pinv(reduced_schur, rcond=1e-12) @ reduced_rhs
      reduced_singular = True
  lifted_solution = basis @ reduced_solution
  omitted_force = full_rhs - full_schur @ lifted_solution
  omitted_force_blocks = rank_omitted_force_blocks(
      omitted_force=omitted_force,
      interface_pose_ids=interface_pose_ids,
      block_dim=block_dim,
      selected_interface_pose_ids=projected["selected_interface_pose_ids"],
      anchor_pose=anchor_pose,
  )
  try:
    full_reference_solution = np.linalg.solve(full_schur, full_rhs)
    full_singular = False
  except np.linalg.LinAlgError:
    full_reference_solution = np.linalg.pinv(full_schur, rcond=1e-12) @ full_rhs
    full_singular = True
  solution_error = lifted_solution - full_reference_solution
  return {
      **projected,
      "model": "skeleton_reduced_interface_schur_solve",
      "reduced_solution": reduced_solution,
      "lifted_interface_solution": lifted_solution,
      "full_reference_solution": full_reference_solution,
      "reduced_singular": bool(reduced_singular),
      "full_singular": bool(full_singular),
      "omitted_force": omitted_force,
      "omitted_force_norm": float(np.linalg.norm(omitted_force)),
      "omitted_force_blocks": omitted_force_blocks,
      "interface_solution_error_norm": float(np.linalg.norm(solution_error)),
      "reduced_residual_norm": float(
          np.linalg.norm(reduced_schur @ reduced_solution - reduced_rhs)
          if reduced_rhs.size else 0.0),
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
      description="Extract graph-derived DCI basin-skeleton candidates.")
  parser.add_argument("--g2o", type=Path, required=True)
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--output-json", type=Path, required=True)
  args = parser.parse_args(argv)

  vertices, graph_edges = parse_g2o_graph(args.g2o)
  pose_ids = graph_pose_ids(vertices, graph_edges)
  robot_of, ranges = build_contiguous_robot_map(pose_ids, args.num_robots)
  report = {
      "model": "dci_basin_skeleton_report",
      "dataset": args.g2o.name,
      "g2o_path": str(args.g2o),
      "num_robots": int(args.num_robots),
      "pose_count": len(pose_ids),
      "edge_count": len(graph_edges),
      "robot_index_ranges": [[int(start), int(end)] for start, end in ranges],
      "skeleton": build_basin_skeleton(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
      ),
  }
  args.output_json.parent.mkdir(parents=True, exist_ok=True)
  args.output_json.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())

#!/usr/bin/env python3
"""Simulate decentralized chordal initialization with block linear solvers.

This diagnostic separates the initialization problem from DRAN refinement. It
solves the chordal rotation and translation least-squares systems either
centrally or by robot-owned block-Jacobi iterations over the normal equations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
  from scipy.sparse import coo_matrix, identity  # type: ignore
  from scipy.sparse.linalg import splu  # type: ignore
except ImportError:  # pragma: no cover - scipy is available in the research env.
  coo_matrix = None
  identity = None
  splu = None

try:
  from scripts.evaluate_pgo import (  # type: ignore
      Edge,
      edge_chordal_cost,
      edge_weights_from_info,
      parse_g2o_graph,
  )
  from scripts.analyze_shape_drift_oracle import (  # type: ignore
      _solve_least_squares,
      active_pose_ids,
      build_contiguous_robot_map,
      cost_breakdown,
      graph_pose_ids,
      load_oracle_pose_set,
      pose_dimension_from_edges,
      project_rotation_block,
  )
except ImportError:  # pragma: no cover - used when executed as a script.
  from evaluate_pgo import (  # type: ignore
      Edge,
      edge_chordal_cost,
      edge_weights_from_info,
      parse_g2o_graph,
  )
  from analyze_shape_drift_oracle import (  # type: ignore
      _solve_least_squares,
      active_pose_ids,
      build_contiguous_robot_map,
      cost_breakdown,
      graph_pose_ids,
      load_oracle_pose_set,
      pose_dimension_from_edges,
      project_rotation_block,
  )


def _variable_pose_ids(pose_ids: list[int], anchor_pose: int):
  return [pose_id for pose_id in pose_ids if pose_id != anchor_pose]


def _pose_variable_offsets(pose_ids: list[int], anchor_pose: int, block_dim: int):
  offsets: dict[int, int] = {}
  offset = 0
  for pose_id in _variable_pose_ids(pose_ids, anchor_pose):
    offsets[pose_id] = offset
    offset += block_dim
  return offsets, offset


def _add_row(rows: list[int], cols: list[int], data: list[float],
             rhs: list[float], row: int,
             coefficients: list[tuple[int, float]], target: float):
  for col, value in coefficients:
    rows.append(row)
    cols.append(col)
    data.append(value)
  rhs.append(target)
  return row + 1


def assemble_rotation_system(graph_edges: list[Edge], pose_ids: list[int],
                             dim: int, weighted: bool, cost_mode: str,
                             anchor_pose: int):
  block_dim = dim * dim
  offsets, variable_count = _pose_variable_offsets(pose_ids, anchor_pose,
                                                   block_dim)
  pose_set = set(pose_ids)
  rows: list[int] = []
  cols: list[int] = []
  data: list[float] = []
  rhs: list[float] = []
  row = 0
  used_edges = 0

  for edge in graph_edges:
    if edge.i not in pose_set or edge.j not in pose_set:
      continue
    _, kappa = edge_weights_from_info(edge, cost_mode, weighted)
    if kappa <= 0.0:
      continue
    used_edges += 1
    weight = math.sqrt(kappa)
    rel_rot = edge.measurement[:dim, :dim]
    for r in range(dim):
      for c in range(dim):
        coefficients: list[tuple[int, float]] = []
        target = 0.0
        if edge.i == anchor_pose:
          target -= weight * float(rel_rot[r, c])
        else:
          base = offsets[edge.i]
          for k in range(dim):
            coefficients.append((base + r * dim + k,
                                 weight * float(rel_rot[k, c])))
        if edge.j == anchor_pose:
          target += weight * (1.0 if r == c else 0.0)
        else:
          coefficients.append((offsets[edge.j] + r * dim + c, -weight))
        row = _add_row(rows, cols, data, rhs, row, coefficients, target)

  if coo_matrix is None:
    matrix = None
  else:
    matrix = coo_matrix((data, (rows, cols)),
                        shape=(row, variable_count)).tocsr()
  return matrix, np.asarray(rhs, dtype=float), {
      "equation_count": row,
      "variable_count": variable_count,
      "used_edges": used_edges,
      "block_dim": block_dim,
      "offsets": offsets,
  }


def assemble_translation_system(graph_edges: list[Edge], pose_ids: list[int],
                                rotations: dict[int, np.ndarray], dim: int,
                                weighted: bool, cost_mode: str,
                                anchor_pose: int):
  offsets, variable_count = _pose_variable_offsets(pose_ids, anchor_pose, dim)
  pose_set = set(pose_ids)
  rows: list[int] = []
  cols: list[int] = []
  data: list[float] = []
  rhs: list[float] = []
  row = 0
  used_edges = 0

  for edge in graph_edges:
    if edge.i not in pose_set or edge.j not in pose_set:
      continue
    tau, _ = edge_weights_from_info(edge, cost_mode, weighted)
    if tau <= 0.0:
      continue
    used_edges += 1
    weight = math.sqrt(tau)
    offset = rotations[edge.i] @ edge.measurement[:dim, 3]
    for c in range(dim):
      coefficients: list[tuple[int, float]] = []
      target = -weight * float(offset[c])
      if edge.i != anchor_pose:
        coefficients.append((offsets[edge.i] + c, weight))
      if edge.j != anchor_pose:
        coefficients.append((offsets[edge.j] + c, -weight))
      row = _add_row(rows, cols, data, rhs, row, coefficients, target)

  if coo_matrix is None:
    matrix = None
  else:
    matrix = coo_matrix((data, (rows, cols)),
                        shape=(row, variable_count)).tocsr()
  return matrix, np.asarray(rhs, dtype=float), {
      "equation_count": row,
      "variable_count": variable_count,
      "used_edges": used_edges,
      "block_dim": dim,
      "offsets": offsets,
  }


def _solve_central_system(matrix, rhs: np.ndarray):
  if matrix is None:
    raise RuntimeError("scipy sparse support is required")
  coo = matrix.tocoo()
  solution, stats = _solve_least_squares(coo.row.tolist(), coo.col.tolist(),
                                         coo.data.tolist(), rhs.tolist(),
                                         matrix.shape[0], matrix.shape[1])
  return solution, stats


def normal_equation_block_summary(matrix, rhs: np.ndarray, block_dim: int,
                                  tolerance: float = 1e-12):
  if matrix is None:
    raise RuntimeError("scipy sparse support is required")
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  variable_count = int(matrix.shape[1])
  if variable_count % int(block_dim) != 0:
    raise ValueError("variable_count must be divisible by block_dim")
  hessian = (matrix.T @ matrix).tocoo()
  gradient = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1)
  block_count = variable_count // int(block_dim)
  hessian_blocks_raw: dict[tuple[int, int], np.ndarray] = {}
  hessian_blocks: dict[tuple[int, int], list[list[float]]] = {}
  gradient_blocks: dict[int, list[float]] = {}

  for row, col, value in zip(hessian.row, hessian.col, hessian.data):
    if abs(float(value)) <= tolerance:
      continue
    row_block = int(row) // int(block_dim)
    col_block = int(col) // int(block_dim)
    key = (row_block, col_block)
    if key not in hessian_blocks_raw:
      hessian_blocks_raw[key] = np.zeros((int(block_dim), int(block_dim)),
                                         dtype=float)
    hessian_blocks_raw[key][
        int(row) % int(block_dim),
        int(col) % int(block_dim),
    ] += float(value)

  for row_block in range(block_count):
    row_start = row_block * int(block_dim)
    row_end = row_start + int(block_dim)
    grad_block = gradient[row_start:row_end]
    if np.any(np.abs(grad_block) > tolerance):
      gradient_blocks[row_block] = grad_block.tolist()
  for key, block in hessian_blocks_raw.items():
    if np.any(np.abs(block) > tolerance):
      hessian_blocks[key] = block.tolist()

  hessian_scalar_count = len(hessian_blocks) * int(block_dim) * int(block_dim)
  gradient_scalar_count = len(gradient_blocks) * int(block_dim)
  return {
      "variable_count": variable_count,
      "block_dim": int(block_dim),
      "block_count": block_count,
      "hessian_blocks": hessian_blocks,
      "gradient_blocks": gradient_blocks,
      "hessian_block_count": len(hessian_blocks),
      "gradient_block_count": len(gradient_blocks),
      "hessian_scalar_count": hessian_scalar_count,
      "gradient_scalar_count": gradient_scalar_count,
      "payload_bytes": (hessian_scalar_count + gradient_scalar_count) * 8,
      "payload_mb": _bytes_to_mb(
          (hessian_scalar_count + gradient_scalar_count) * 8),
      "model": "dense_block_normal_equation_summary",
  }


def recount_normal_summary_payload(summary: dict):
  block_dim = int(summary["block_dim"])
  hessian_blocks = dict(summary.get("hessian_blocks", {}))
  gradient_blocks = dict(summary.get("gradient_blocks", {}))
  hessian_scalar_count = len(hessian_blocks) * block_dim * block_dim
  gradient_scalar_count = len(gradient_blocks) * block_dim
  payload_bytes = (hessian_scalar_count + gradient_scalar_count) * 8
  updated = dict(summary)
  updated["hessian_blocks"] = hessian_blocks
  updated["gradient_blocks"] = gradient_blocks
  updated["hessian_block_count"] = len(hessian_blocks)
  updated["gradient_block_count"] = len(gradient_blocks)
  updated["hessian_scalar_count"] = hessian_scalar_count
  updated["gradient_scalar_count"] = gradient_scalar_count
  updated["payload_bytes"] = payload_bytes
  updated["payload_mb"] = _bytes_to_mb(payload_bytes)
  return updated


def select_normal_summary_blocks(summary: dict,
                                 mode: str = "all",
                                 max_offdiag_block_edges: int | None = None):
  if mode not in {"all", "structural_spanning"}:
    raise ValueError(f"unsupported summary selection mode: {mode}")
  full = recount_normal_summary_payload(summary)
  if mode == "all" or max_offdiag_block_edges is None:
    selected = dict(full)
    selected["summary_selection"] = {
        "mode": mode,
        "max_offdiag_block_edges": max_offdiag_block_edges,
        "full_offdiag_block_edges": len(_normal_summary_block_graph(full)[2]),
        "selected_offdiag_block_edges": len(_normal_summary_block_graph(full)[2]),
    }
    return selected

  max_edges = max(0, int(max_offdiag_block_edges))
  nodes, _, block_edges = _normal_summary_block_graph(full)
  if max_edges >= len(block_edges):
    selected = dict(full)
    selected["summary_selection"] = {
        "mode": mode,
        "max_offdiag_block_edges": max_edges,
        "full_offdiag_block_edges": len(block_edges),
        "selected_offdiag_block_edges": len(block_edges),
    }
    return selected

  edge_scores: dict[tuple[int, int], float] = {edge: 0.0 for edge in block_edges}
  directed_blocks: dict[tuple[int, int], list[tuple[tuple[int, int], object]]] = {
      edge: [] for edge in block_edges
  }
  diagonal_blocks = {}
  for key, values in full.get("hessian_blocks", {}).items():
    row_block, col_block = _normal_summary_key_to_blocks(key)
    if row_block == col_block:
      diagonal_blocks[(row_block, col_block)] = values
      continue
    edge = (min(row_block, col_block), max(row_block, col_block))
    block = np.asarray(values, dtype=float)
    edge_scores[edge] = edge_scores.get(edge, 0.0) + float(np.linalg.norm(block))
    directed_blocks.setdefault(edge, []).append(((row_block, col_block), values))

  parent = {node: node for node in nodes}
  rank = {node: 0 for node in nodes}

  def find(node):
    if parent[node] != node:
      parent[node] = find(parent[node])
    return parent[node]

  def union(i, j):
    ri = find(i)
    rj = find(j)
    if ri == rj:
      return False
    if rank[ri] < rank[rj]:
      ri, rj = rj, ri
    parent[rj] = ri
    if rank[ri] == rank[rj]:
      rank[ri] += 1
    return True

  ordered_edges = sorted(
      block_edges,
      key=lambda edge: (-edge_scores.get(edge, 0.0), edge[0], edge[1]))
  selected_edges: list[tuple[int, int]] = []
  for edge in ordered_edges:
    if len(selected_edges) >= max_edges:
      break
    if union(edge[0], edge[1]):
      selected_edges.append(edge)
  for edge in ordered_edges:
    if len(selected_edges) >= max_edges:
      break
    if edge not in selected_edges:
      selected_edges.append(edge)

  selected_edge_set = set(selected_edges)
  selected_hessian = dict(diagonal_blocks)
  for edge in selected_edges:
    for key, values in directed_blocks.get(edge, []):
      selected_hessian[key] = values
  selected = {
      **full,
      "hessian_blocks": selected_hessian,
      "gradient_blocks": dict(full.get("gradient_blocks", {})),
  }
  selected = recount_normal_summary_payload(selected)
  selected["summary_selection"] = {
      "mode": mode,
      "max_offdiag_block_edges": max_edges,
      "full_offdiag_block_edges": len(block_edges),
      "selected_offdiag_block_edges": len(selected_edge_set),
      "selected_offdiag_block_edge_keys": [
          [int(edge[0]), int(edge[1])] for edge in selected_edges
      ],
  }
  return selected


def refine_normal_summary_blocks_by_residual_force(
    full_summary: dict,
    selected_summary: dict,
    solution: np.ndarray,
    max_offdiag_block_edges: int | None = None):
  full = recount_normal_summary_payload(full_summary)
  selected = recount_normal_summary_payload(selected_summary)
  full_edges = _normal_summary_block_graph(full)[2]
  selected_edges = _normal_summary_block_graph(selected)[2]
  if max_offdiag_block_edges is None:
    max_edges = len(full_edges)
  else:
    max_edges = max(0, int(max_offdiag_block_edges))
  seed_edge_count = len(selected_edges)
  if max_edges <= seed_edge_count:
    unchanged = dict(selected)
    unchanged["summary_selection"] = {
        **dict(selected.get("summary_selection", {})),
        "mode": "residual_force_refinement",
        "seed_offdiag_block_edges": seed_edge_count,
        "max_offdiag_block_edges": max_edges,
        "full_offdiag_block_edges": len(full_edges),
        "selected_offdiag_block_edges": seed_edge_count,
        "refinement_added_offdiag_block_edges": 0,
    }
    return unchanged

  block_dim = int(full["block_dim"])
  solution = np.asarray(solution, dtype=float).reshape(-1)
  directed_blocks: dict[tuple[int, int], list[tuple[tuple[int, int], object]]] = {}
  edge_scores: dict[tuple[int, int], float] = {}
  for key, values in full.get("hessian_blocks", {}).items():
    row_block, col_block = _normal_summary_key_to_blocks(key)
    if row_block == col_block:
      continue
    edge = (min(row_block, col_block), max(row_block, col_block))
    directed_blocks.setdefault(edge, []).append(((row_block, col_block), values))
    if edge in selected_edges:
      continue
    col_start = col_block * block_dim
    col_vec = solution[col_start:col_start + block_dim]
    force = np.asarray(values, dtype=float) @ col_vec
    edge_scores[edge] = edge_scores.get(edge, 0.0) + float(force @ force)

  missing_edges = sorted(
      (edge for edge in full_edges if edge not in selected_edges),
      key=lambda edge: (-edge_scores.get(edge, 0.0), edge[0], edge[1]))
  added_edges = missing_edges[:max(0, max_edges - seed_edge_count)]
  selected_hessian = dict(selected.get("hessian_blocks", {}))
  for edge in added_edges:
    for key, values in directed_blocks.get(edge, []):
      selected_hessian[key] = values
  refined = {
      **selected,
      "hessian_blocks": selected_hessian,
      "gradient_blocks": dict(selected.get("gradient_blocks", {})),
  }
  refined = recount_normal_summary_payload(refined)
  refined_edges = _normal_summary_block_graph(refined)[2]
  refined["summary_selection"] = {
      **dict(selected.get("summary_selection", {})),
      "mode": "residual_force_refinement",
      "seed_offdiag_block_edges": seed_edge_count,
      "max_offdiag_block_edges": max_edges,
      "full_offdiag_block_edges": len(full_edges),
      "selected_offdiag_block_edges": len(refined_edges),
      "refinement_added_offdiag_block_edges": len(added_edges),
      "selected_offdiag_block_edge_keys": [
          [int(edge[0]), int(edge[1])] for edge in sorted(refined_edges)
      ],
      "refinement_added_offdiag_block_edge_keys": [
          [int(edge[0]), int(edge[1])] for edge in added_edges
      ],
      "refinement_added_force_scores": [
          float(math.sqrt(max(0.0, edge_scores.get(edge, 0.0))))
          for edge in added_edges
      ],
  }
  return refined


def _normal_summary_key_to_blocks(key):
  if isinstance(key, str):
    stripped = key.strip().strip("()")
    parts = [part.strip() for part in stripped.split(",") if part.strip()]
    if len(parts) != 2:
      raise ValueError(f"invalid normal-summary block key: {key}")
    return int(parts[0]), int(parts[1])
  return int(key[0]), int(key[1])


def _normal_summary_block_graph(summary: dict):
  nodes: set[int] = set()
  diagonal_blocks: set[int] = set()
  block_edges: set[tuple[int, int]] = set()
  for key in summary.get("hessian_blocks", {}):
    row_block, col_block = _normal_summary_key_to_blocks(key)
    nodes.add(row_block)
    nodes.add(col_block)
    if row_block == col_block:
      diagonal_blocks.add(row_block)
    else:
      block_edges.add((
          min(row_block, col_block),
          max(row_block, col_block),
      ))
  for key in summary.get("gradient_blocks", {}):
    nodes.add(int(key))
  return nodes, diagonal_blocks, block_edges


def _component_count(nodes: set[int], edges: set[tuple[int, int]]):
  if not nodes:
    return 0
  adjacency = {node: set() for node in nodes}
  for i, j in edges:
    if i not in nodes or j not in nodes:
      continue
    adjacency[i].add(j)
    adjacency[j].add(i)
  seen = set()
  count = 0
  for start in sorted(nodes):
    if start in seen:
      continue
    count += 1
    stack = [start]
    seen.add(start)
    while stack:
      node = stack.pop()
      for nbr in adjacency[node]:
        if nbr in seen:
          continue
        seen.add(nbr)
        stack.append(nbr)
  return count


def _bridge_edges(nodes: set[int], edges: set[tuple[int, int]]):
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
          bridges.add((min(node, nbr), max(node, nbr)))
      else:
        low[node] = min(low[node], discovery[nbr])

  for node in sorted(nodes):
    if node not in discovery:
      visit(node, None)
  return bridges


def normal_summary_block_graph_diagnostics(full_summary: dict,
                                           selected_summary: dict):
  full_nodes, full_diagonal_blocks, full_edges = _normal_summary_block_graph(
      full_summary)
  selected_nodes, selected_diagonal_blocks, selected_edges = (
      _normal_summary_block_graph(selected_summary))
  nodes = set(full_nodes)
  covered_edges = full_edges & selected_edges
  missing_edges = full_edges - selected_edges
  covered_diagonal_blocks = full_diagonal_blocks & selected_diagonal_blocks
  missing_diagonal_blocks = full_diagonal_blocks - selected_diagonal_blocks
  full_component_count = _component_count(nodes, full_edges)
  selected_component_count = _component_count(nodes, covered_edges)
  component_delta = max(0, selected_component_count - full_component_count)
  full_bridges = _bridge_edges(nodes, full_edges)
  missing_bridges = missing_edges & full_bridges
  structural_criticality = (
      float(len(missing_bridges) + component_delta) /
      float(max(1, len(full_edges)))
  )
  return {
      "full_node_count": len(full_nodes),
      "selected_node_count": len(selected_nodes),
      "full_diagonal_block_count": len(full_diagonal_blocks),
      "covered_diagonal_block_count": len(covered_diagonal_blocks),
      "missing_diagonal_block_count": len(missing_diagonal_blocks),
      "full_block_edge_count": len(full_edges),
      "covered_block_edge_count": len(covered_edges),
      "missing_block_edge_count": len(missing_edges),
      "missing_bridge_block_edge_count": len(missing_bridges),
      "full_component_count": full_component_count,
      "selected_component_count": selected_component_count,
      "component_count_delta": component_delta,
      "structural_criticality": structural_criticality,
  }


def _combine_normal_summary_block_graph_diagnostics(rotation_diag: dict,
                                                    translation_diag: dict):
  summed_keys = [
      "full_node_count",
      "selected_node_count",
      "full_diagonal_block_count",
      "covered_diagonal_block_count",
      "missing_diagonal_block_count",
      "full_block_edge_count",
      "covered_block_edge_count",
      "missing_block_edge_count",
      "missing_bridge_block_edge_count",
      "full_component_count",
      "selected_component_count",
      "component_count_delta",
  ]
  combined = {
      key: int(rotation_diag.get(key, 0)) + int(translation_diag.get(key, 0))
      for key in summed_keys
  }
  combined["structural_criticality"] = (
      float(combined["missing_bridge_block_edge_count"] +
            combined["component_count_delta"]) /
      float(max(1, combined["full_block_edge_count"]))
  )
  return combined


def separator_normal_summary_block_graph_diagnostics(
    full_graph_edges: list[Edge],
    selected_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    rotations: dict[int, np.ndarray],
    dim: int,
    weighted: bool,
    cost_mode: str,
    anchor_pose: int):
  full_separator_edges = [
      edge for edge in full_graph_edges
      if edge.i in robot_of and edge.j in robot_of and
      robot_of[edge.i] != robot_of[edge.j]
  ]
  selected_separator_edges = [
      edge for edge in selected_edges
      if edge.i in robot_of and edge.j in robot_of and
      robot_of[edge.i] != robot_of[edge.j]
  ]

  full_rot_matrix, full_rot_rhs, full_rot_meta = assemble_rotation_system(
      full_separator_edges, pose_ids, dim, weighted, cost_mode, anchor_pose)
  selected_rot_matrix, selected_rot_rhs, selected_rot_meta = (
      assemble_rotation_system(
          selected_separator_edges, pose_ids, dim, weighted, cost_mode,
          anchor_pose))
  if full_rot_meta["block_dim"] != selected_rot_meta["block_dim"]:
    raise ValueError("rotation block dimensions do not match")
  full_rot_summary = normal_equation_block_summary(
      full_rot_matrix, full_rot_rhs, full_rot_meta["block_dim"])
  selected_rot_summary = normal_equation_block_summary(
      selected_rot_matrix, selected_rot_rhs, selected_rot_meta["block_dim"])
  rotation_diag = normal_summary_block_graph_diagnostics(
      full_rot_summary, selected_rot_summary)

  full_trans_matrix, full_trans_rhs, full_trans_meta = assemble_translation_system(
      full_separator_edges, pose_ids, rotations, dim, weighted, cost_mode,
      anchor_pose)
  selected_trans_matrix, selected_trans_rhs, selected_trans_meta = (
      assemble_translation_system(
          selected_separator_edges, pose_ids, rotations, dim, weighted,
          cost_mode, anchor_pose))
  if full_trans_meta["block_dim"] != selected_trans_meta["block_dim"]:
    raise ValueError("translation block dimensions do not match")
  full_trans_summary = normal_equation_block_summary(
      full_trans_matrix, full_trans_rhs, full_trans_meta["block_dim"])
  selected_trans_summary = normal_equation_block_summary(
      selected_trans_matrix, selected_trans_rhs, selected_trans_meta["block_dim"])
  translation_diag = normal_summary_block_graph_diagnostics(
      full_trans_summary, selected_trans_summary)
  combined = _combine_normal_summary_block_graph_diagnostics(
      rotation_diag, translation_diag)
  return {
      **combined,
      "rotation": rotation_diag,
      "translation": translation_diag,
      "model": "separator_normal_summary_block_graph_diagnostics",
  }


def reconstruct_normal_equation_from_summary(summary: dict):
  variable_count = int(summary["variable_count"])
  block_dim = int(summary["block_dim"])
  hessian = np.zeros((variable_count, variable_count), dtype=float)
  gradient = np.zeros(variable_count, dtype=float)
  for key, values in summary["hessian_blocks"].items():
    if isinstance(key, str):
      row_block, col_block = (int(part) for part in key.split(","))
    else:
      row_block, col_block = key
    row_start = int(row_block) * block_dim
    col_start = int(col_block) * block_dim
    hessian[
      row_start:row_start + block_dim,
      col_start:col_start + block_dim,
    ] = np.asarray(values, dtype=float)
  for key, values in summary["gradient_blocks"].items():
    block = int(key)
    start = block * block_dim
    gradient[start:start + block_dim] = np.asarray(values, dtype=float)
  return hessian, gradient


def reconstruct_sparse_normal_equation_from_summary(summary: dict):
  if coo_matrix is None:
    raise RuntimeError("scipy sparse support is required")
  variable_count = int(summary["variable_count"])
  block_dim = int(summary["block_dim"])
  rows: list[int] = []
  cols: list[int] = []
  data: list[float] = []
  gradient = np.zeros(variable_count, dtype=float)
  for key, values in summary["hessian_blocks"].items():
    if isinstance(key, str):
      row_block, col_block = (int(part) for part in key.split(","))
    else:
      row_block, col_block = key
    block = np.asarray(values, dtype=float)
    for row in range(block.shape[0]):
      for col in range(block.shape[1]):
        value = float(block[row, col])
        if value == 0.0:
          continue
        rows.append(int(row_block) * block_dim + row)
        cols.append(int(col_block) * block_dim + col)
        data.append(value)
  for key, values in summary["gradient_blocks"].items():
    block_id = int(key)
    start = block_id * block_dim
    gradient[start:start + block_dim] = np.asarray(values, dtype=float)
  hessian = coo_matrix(
      (data, (rows, cols)), shape=(variable_count, variable_count)).tocsc()
  return hessian, gradient


def sum_local_interface_schur_contributions(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0):
  """Sums robot-local interface Schur systems in global coordinates.

  This diagnostic is the algebraic target for a decentralized CCI-equivalent
  initializer. Each local system contributes

    S_r = H_BB^r - H_BP^r (H_PP^r)^(-1) H_PB^r
    b_r = g_B^r - H_BP^r (H_PP^r)^(-1) g_P^r

  over robot-private variables P_r and shared interface variables B. When the
  private variables are disjoint across robots, summing these local
  contributions is exactly the centralized interface Schur system.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
  if variable_count is None:
    inferred = int(np.max(interface_indices)) + 1
    for system in local_systems:
      hessian = system.get("hessian")
      if hessian is None:
        continue
      inferred = max(inferred, int(hessian.shape[0]))
    variable_count = inferred
  variable_count = int(variable_count)
  if variable_count <= 0:
    raise ValueError("variable_count must be positive")
  if int(np.min(interface_indices)) < 0 or int(np.max(interface_indices)) >= variable_count:
    raise ValueError("interface index outside variable dimension")

  schur = np.zeros((len(interface_indices), len(interface_indices)), dtype=float)
  rhs = np.zeros(len(interface_indices), dtype=float)
  local_private_variable_count_sum = 0
  local_singular_count = 0
  local_condition_max = 0.0

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
    local_private_variable_count_sum += int(private_indices.size)

    h_bb = hessian[interface_indices[:, None], interface_indices]
    g_b = gradient[interface_indices]
    if private_indices.size == 0:
      schur += h_bb
      rhs += g_b
      continue

    h_pp = hessian[private_indices[:, None], private_indices]
    if damping > 0.0:
      h_pp = h_pp + float(damping) * np.eye(len(private_indices))
    h_pb = hessian[private_indices[:, None], interface_indices]
    h_bp = hessian[interface_indices[:, None], private_indices]
    g_p = gradient[private_indices]
    condition = float(np.linalg.cond(h_pp))
    if np.isfinite(condition):
      local_condition_max = max(local_condition_max, condition)
    try:
      private_interface_response = np.linalg.solve(h_pp, h_pb)
      private_rhs_response = np.linalg.solve(h_pp, g_p)
    except np.linalg.LinAlgError:
      local_singular_count += 1
      h_pp_pinv = np.linalg.pinv(h_pp, rcond=1e-12)
      private_interface_response = h_pp_pinv @ h_pb
      private_rhs_response = h_pp_pinv @ g_p
    schur += h_bb - h_bp @ private_interface_response
    rhs += g_b - h_bp @ private_rhs_response

  schur = 0.5 * (schur + schur.T)
  return schur, rhs, {
      "model": "sum_local_interface_schur_contributions",
      "local_system_count": int(len(local_systems)),
      "interface_variable_count": int(len(interface_indices)),
      "variable_count": int(variable_count),
      "local_private_variable_count_sum": int(local_private_variable_count_sum),
      "local_singular_count": int(local_singular_count),
      "local_hpp_condition_max": float(local_condition_max),
      "damping": float(damping),
  }


def normalized_interface_schur_spectral_certificate(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0,
    lambda_min_bound: float = 1e-4,
    lambda_max_bound: float = 2.0,
    positive_tolerance: float = 1e-10):
  """Dense diagnostic for graph-normalized Chebyshev spectral coverage.

  This is not a deployment solver. It materializes the interface Schur matrix
  to audit whether the heuristic Chebyshev interval covers the normalized
  operator ``D^{-1/2} S D^{-1/2}``, where ``D = abs(diag(S))``.
  """
  schur, _, schur_stats = sum_local_interface_schur_contributions(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping)
  schur = np.asarray(schur, dtype=float)
  if schur.ndim != 2 or schur.shape[0] != schur.shape[1]:
    raise ValueError("Schur matrix must be square")
  lambda_min_bound = float(lambda_min_bound)
  lambda_max_bound = float(lambda_max_bound)
  positive_tolerance = float(positive_tolerance)
  if (not np.isfinite(lambda_min_bound) or
      not np.isfinite(lambda_max_bound) or
      lambda_min_bound <= 0.0 or lambda_max_bound <= lambda_min_bound):
    raise ValueError(
        "spectral bounds must satisfy 0 < lambda_min_bound < lambda_max_bound")
  if not np.isfinite(positive_tolerance) or positive_tolerance < 0.0:
    raise ValueError("positive_tolerance must be nonnegative")

  symmetry_error = float(np.max(np.abs(schur - schur.T))) if schur.size else 0.0
  structural_tolerance = max(1e-10, 10.0 * float(positive_tolerance))
  raw_diagonal = np.diag(schur)
  offdiag = schur.copy()
  np.fill_diagonal(offdiag, 0.0)
  offdiag_abs_sum = np.sum(np.abs(offdiag), axis=1)
  diagonal_dominance_margin = raw_diagonal - offdiag_abs_sum
  positive_offdiag = offdiag[offdiag > structural_tolerance]
  sdd_m_matrix_is_symmetric = symmetry_error <= structural_tolerance
  sdd_m_matrix_positive_diagonal = bool(
      np.all(raw_diagonal > structural_tolerance))
  sdd_m_matrix_z_offdiag = bool(positive_offdiag.size == 0)
  sdd_m_matrix_diagonal_dominance = bool(
      np.all(diagonal_dominance_margin >= -structural_tolerance))
  sdd_m_matrix_theorem_applies = bool(
      sdd_m_matrix_is_symmetric and
      sdd_m_matrix_positive_diagonal and
      sdd_m_matrix_z_offdiag and
      sdd_m_matrix_diagonal_dominance)

  diagonal = np.maximum(np.abs(raw_diagonal), 1e-12)
  sqrt_diagonal = np.sqrt(diagonal)
  normalized = (schur / sqrt_diagonal[:, None]) / sqrt_diagonal[None, :]
  normalized = 0.5 * (normalized + normalized.T)
  eigenvalues = np.linalg.eigvalsh(normalized)
  positive_eigenvalues = eigenvalues[eigenvalues > positive_tolerance]
  if positive_eigenvalues.size:
    normalized_lambda_min = float(positive_eigenvalues[0])
    normalized_lambda_max = float(positive_eigenvalues[-1])
  else:
    normalized_lambda_min = 0.0
    normalized_lambda_max = 0.0
  lambda_min_covered = (
      normalized_lambda_min == 0.0 or
      lambda_min_bound <= normalized_lambda_min)
  lambda_max_covered = normalized_lambda_max <= lambda_max_bound
  sdd_m_matrix_theorem_lambda_max_bound = (
      2.0 if sdd_m_matrix_theorem_applies else 0.0)
  sdd_m_matrix_theorem_lambda_max_covered = bool(
      sdd_m_matrix_theorem_applies and
      normalized_lambda_max <= 2.0 + structural_tolerance)
  lambda_max_bound_source = (
      "sdd_m_matrix_theorem"
      if (sdd_m_matrix_theorem_applies and lambda_max_bound >= 2.0)
      else "numeric_eigendecomposition")
  return {
      "model": "normalized_interface_schur_spectral_certificate",
      "materializes_dense_schur": True,
      "normalization": "abs_diagonal",
      "structural_tolerance": float(structural_tolerance),
      "interface_variable_count": int(schur.shape[0]),
      "positive_tolerance": float(positive_tolerance),
      "positive_eigenvalue_count": int(positive_eigenvalues.size),
      "normalized_lambda_min": float(normalized_lambda_min),
      "normalized_lambda_max": float(normalized_lambda_max),
      "normalized_lambda_raw_min": (
          float(eigenvalues[0]) if eigenvalues.size else 0.0),
      "normalized_lambda_raw_max": (
          float(eigenvalues[-1]) if eigenvalues.size else 0.0),
      "lambda_min_bound": float(lambda_min_bound),
      "lambda_max_bound": float(lambda_max_bound),
      "lambda_min_covered": bool(lambda_min_covered),
      "lambda_max_covered": bool(lambda_max_covered),
      "lambda_max_bound_source": lambda_max_bound_source,
      "interval_covered": bool(lambda_min_covered and lambda_max_covered),
      "sdd_m_matrix_structure_checked": True,
      "sdd_m_matrix_is_symmetric": bool(sdd_m_matrix_is_symmetric),
      "sdd_m_matrix_positive_diagonal": bool(
          sdd_m_matrix_positive_diagonal),
      "sdd_m_matrix_z_offdiag": bool(sdd_m_matrix_z_offdiag),
      "sdd_m_matrix_diagonal_dominance": bool(
          sdd_m_matrix_diagonal_dominance),
      "sdd_m_matrix_theorem_applies": bool(
          sdd_m_matrix_theorem_applies),
      "sdd_m_matrix_symmetry_error": float(symmetry_error),
      "sdd_m_matrix_min_diagonal": (
          float(np.min(raw_diagonal)) if raw_diagonal.size else 0.0),
      "sdd_m_matrix_max_positive_offdiag": (
          float(np.max(positive_offdiag)) if positive_offdiag.size else 0.0),
      "sdd_m_matrix_min_diagonal_dominance_margin": (
          float(np.min(diagonal_dominance_margin))
          if diagonal_dominance_margin.size else 0.0),
      "sdd_m_matrix_bad_diagonal_count": int(
          np.count_nonzero(raw_diagonal <= structural_tolerance)),
      "sdd_m_matrix_positive_offdiag_count": int(positive_offdiag.size),
      "sdd_m_matrix_bad_diagonal_dominance_count": int(
          np.count_nonzero(
              diagonal_dominance_margin < -structural_tolerance)),
      "sdd_m_matrix_theorem_lambda_max_bound": float(
          sdd_m_matrix_theorem_lambda_max_bound),
      "sdd_m_matrix_theorem_lambda_max_covered": bool(
          sdd_m_matrix_theorem_lambda_max_covered),
      "schur_stats": schur_stats,
  }


def dirichlet_interface_schur_spectral_certificate(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0,
    lambda_max_bound: float = 2.0,
    positive_tolerance: float = 1e-10):
  """Dense Dirichlet-transfer certificate for interface Schur spectra.

  If the full normal matrix satisfies ``H <= lambda * D`` with
  ``D = diag(H)``, then the Schur complement after minimizing private
  variables satisfies ``S <= lambda * D_B``, where ``D_B`` is the pre-elimination
  boundary diagonal. This diagnostic checks the premise by dense eigensolve and
  reports the transferred boundary-diagonal normalized Schur spectrum.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
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
  lambda_max_bound = float(lambda_max_bound)
  positive_tolerance = float(positive_tolerance)
  if not np.isfinite(lambda_max_bound) or lambda_max_bound <= 0.0:
    raise ValueError("lambda_max_bound must be positive and finite")
  if not np.isfinite(positive_tolerance) or positive_tolerance < 0.0:
    raise ValueError("positive_tolerance must be nonnegative")

  full_hessian = np.zeros((variable_count, variable_count), dtype=float)
  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    gradient = system.get("gradient")
    if hessian is None or gradient is None:
      raise ValueError(f"local system {system_index} is missing hessian/gradient")
    if hasattr(hessian, "toarray"):
      hessian = hessian.toarray()
    hessian = np.asarray(hessian, dtype=float)
    if hessian.shape != (variable_count, variable_count):
      raise ValueError(
          f"local system {system_index} hessian has wrong shape")
    full_hessian += hessian
  full_hessian = 0.5 * (full_hessian + full_hessian.T)

  schur, _, schur_stats = sum_local_interface_schur_contributions(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping)
  schur = np.asarray(schur, dtype=float)
  if schur.ndim != 2 or schur.shape[0] != schur.shape[1]:
    raise ValueError("Schur matrix must be square")

  def normalized_eigenspectrum(matrix, diagonal):
    matrix = np.asarray(matrix, dtype=float)
    diagonal = np.asarray(diagonal, dtype=float).reshape(-1)
    if matrix.shape != (diagonal.shape[0], diagonal.shape[0]):
      raise ValueError("matrix/diagonal dimension mismatch")
    safe_diagonal = np.maximum(np.abs(diagonal), 1e-12)
    sqrt_diagonal = np.sqrt(safe_diagonal)
    normalized = (matrix / sqrt_diagonal[:, None]) / sqrt_diagonal[None, :]
    normalized = 0.5 * (normalized + normalized.T)
    eigenvalues = np.linalg.eigvalsh(normalized)
    positive = eigenvalues[eigenvalues > positive_tolerance]
    return {
        "lambda_min": float(positive[0]) if positive.size else 0.0,
        "lambda_max": float(eigenvalues[-1]) if eigenvalues.size else 0.0,
        "raw_lambda_min": float(eigenvalues[0]) if eigenvalues.size else 0.0,
        "raw_lambda_max": float(eigenvalues[-1]) if eigenvalues.size else 0.0,
        "positive_eigenvalue_count": int(positive.size),
        "diagonal_min": float(np.min(safe_diagonal))
            if safe_diagonal.size else 0.0,
        "diagonal_max": float(np.max(safe_diagonal))
            if safe_diagonal.size else 0.0,
    }

  full_diagonal = np.diag(full_hessian)
  boundary_diagonal = full_diagonal[interface_indices]
  schur_diagonal = np.diag(schur)
  full_spectrum = normalized_eigenspectrum(full_hessian, full_diagonal)
  boundary_spectrum = normalized_eigenspectrum(schur, boundary_diagonal)
  schur_spectrum = normalized_eigenspectrum(schur, schur_diagonal)
  structural_tolerance = max(1e-10, 10.0 * float(positive_tolerance))
  dirichlet_transfer_bound_applies = bool(
      full_spectrum["lambda_max"] <= lambda_max_bound + structural_tolerance)
  lambda_max_bound_source = (
      "dirichlet_full_diagonal_numeric"
      if dirichlet_transfer_bound_applies else "numeric_only")
  return {
      "model": "dirichlet_interface_schur_spectral_certificate",
      "materializes_dense_schur": True,
      "materializes_dense_full_hessian": True,
      "normalization": "pre_elimination_boundary_diagonal",
      "interface_variable_count": int(len(interface_indices)),
      "variable_count": int(variable_count),
      "damping": float(damping),
      "positive_tolerance": float(positive_tolerance),
      "structural_tolerance": float(structural_tolerance),
      "lambda_max_bound": float(lambda_max_bound),
      "lambda_max_bound_source": lambda_max_bound_source,
      "dirichlet_transfer_bound_applies": bool(
          dirichlet_transfer_bound_applies),
      "full_diagonal_normalized_lambda_min": float(
          full_spectrum["lambda_min"]),
      "full_diagonal_normalized_lambda_max": float(
          full_spectrum["lambda_max"]),
      "full_diagonal_normalized_raw_lambda_min": float(
          full_spectrum["raw_lambda_min"]),
      "full_diagonal_normalized_positive_eigenvalue_count": int(
          full_spectrum["positive_eigenvalue_count"]),
      "boundary_diagonal_normalized_lambda_min": float(
          boundary_spectrum["lambda_min"]),
      "boundary_diagonal_normalized_lambda_max": float(
          boundary_spectrum["lambda_max"]),
      "boundary_diagonal_normalized_raw_lambda_min": float(
          boundary_spectrum["raw_lambda_min"]),
      "boundary_diagonal_normalized_positive_eigenvalue_count": int(
          boundary_spectrum["positive_eigenvalue_count"]),
      "schur_diagonal_normalized_lambda_min": float(
          schur_spectrum["lambda_min"]),
      "schur_diagonal_normalized_lambda_max": float(
          schur_spectrum["lambda_max"]),
      "schur_diagonal_normalized_raw_lambda_min": float(
          schur_spectrum["raw_lambda_min"]),
      "schur_diagonal_normalized_positive_eigenvalue_count": int(
          schur_spectrum["positive_eigenvalue_count"]),
      "boundary_diagonal_min": float(boundary_spectrum["diagonal_min"]),
      "boundary_diagonal_max": float(boundary_spectrum["diagonal_max"]),
      "schur_diagonal_min": float(schur_spectrum["diagonal_min"]),
      "schur_diagonal_max": float(schur_spectrum["diagonal_max"]),
      "schur_stats": schur_stats,
  }


def block_gershgorin_interface_schur_spectral_certificate(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    block_dim: int,
    variable_count: int | None = None,
    damping: float = 0.0,
    lambda_min_bound: float = 1e-8,
    lambda_max_bound: float = 2.0,
    positive_tolerance: float = 1e-10):
  """Dense weighted block-Gershgorin certificate for Schur spectra.

  For block diagonal ``D = blockdiag(S_ii)``, define
  ``A = D^{-1/2} S D^{-1/2}`` and
  ``C_ij = ||A_ij||_2`` for ``i != j``.  If every ``S_ii`` is SPD and
  ``rho(C) <= 1``, weighted block Gershgorin/Collatz-Wielandt gives
  ``lambda_max(A) <= 1 + rho(C) <= 2``.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  block_dim = int(block_dim)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  if len(interface_indices) % block_dim != 0:
    raise ValueError("interface variable count must be divisible by block_dim")
  lambda_min_bound = float(lambda_min_bound)
  lambda_max_bound = float(lambda_max_bound)
  positive_tolerance = float(positive_tolerance)
  if (not np.isfinite(lambda_min_bound) or
      not np.isfinite(lambda_max_bound) or
      lambda_min_bound <= 0.0 or lambda_max_bound <= lambda_min_bound):
    raise ValueError(
        "spectral bounds must satisfy 0 < lambda_min_bound < lambda_max_bound")
  if not np.isfinite(positive_tolerance) or positive_tolerance < 0.0:
    raise ValueError("positive_tolerance must be nonnegative")

  schur, _, schur_stats = sum_local_interface_schur_contributions(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping)
  schur = np.asarray(schur, dtype=float)
  schur = 0.5 * (schur + schur.T)
  block_count = len(interface_indices) // block_dim
  structural_tolerance = max(1e-10, 10.0 * positive_tolerance)

  inv_sqrt_blocks: list[np.ndarray] = []
  diagonal_block_min_eigenvalue = float("inf")
  diagonal_block_max_condition = 0.0
  bad_diagonal_block_count = 0
  for block_index in range(block_count):
    block_slice = slice(
        block_index * block_dim, (block_index + 1) * block_dim)
    diagonal_block = schur[block_slice, block_slice]
    diagonal_block = 0.5 * (diagonal_block + diagonal_block.T)
    eigenvalues, eigenvectors = np.linalg.eigh(diagonal_block)
    min_eigenvalue = float(eigenvalues[0]) if eigenvalues.size else 0.0
    max_eigenvalue = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    diagonal_block_min_eigenvalue = min(
        diagonal_block_min_eigenvalue, min_eigenvalue)
    if min_eigenvalue <= structural_tolerance:
      bad_diagonal_block_count += 1
    if min_eigenvalue > structural_tolerance:
      condition = max_eigenvalue / min_eigenvalue
      if np.isfinite(condition):
        diagonal_block_max_condition = max(
            diagonal_block_max_condition, float(condition))
    safe = np.maximum(eigenvalues, 1e-12)
    inv_sqrt_blocks.append((eigenvectors / np.sqrt(safe)) @ eigenvectors.T)

  normalized = np.zeros_like(schur)
  coupling = np.zeros((block_count, block_count), dtype=float)
  for row_block in range(block_count):
    row_slice = slice(row_block * block_dim, (row_block + 1) * block_dim)
    left = inv_sqrt_blocks[row_block]
    for col_block in range(block_count):
      col_slice = slice(col_block * block_dim, (col_block + 1) * block_dim)
      right = inv_sqrt_blocks[col_block]
      normalized_block = left @ schur[row_slice, col_slice] @ right
      normalized[row_slice, col_slice] = normalized_block
      if row_block != col_block:
        coupling[row_block, col_block] = float(
            np.linalg.norm(normalized_block, ord=2))
  normalized = 0.5 * (normalized + normalized.T)
  eigenvalues = np.linalg.eigvalsh(normalized)
  positive_eigenvalues = eigenvalues[eigenvalues > positive_tolerance]
  normalized_lambda_min = (
      float(positive_eigenvalues[0]) if positive_eigenvalues.size else 0.0)
  normalized_lambda_max = (
      float(eigenvalues[-1]) if eigenvalues.size else 0.0)
  row_sums = np.sum(coupling, axis=1) if coupling.size else np.zeros(0)
  coupling_eigenvalues = (
      np.linalg.eigvals(coupling) if coupling.size else np.zeros(0))
  spectral_radius = (
      float(np.max(np.abs(coupling_eigenvalues)))
      if coupling_eigenvalues.size else 0.0)
  theorem_bound = 1.0 + spectral_radius
  diagonal_blocks_spd = bool(bad_diagonal_block_count == 0)
  theorem_applies = bool(
      diagonal_blocks_spd and
      spectral_radius <= 1.0 + structural_tolerance)
  lambda_min_covered = (
      normalized_lambda_min == 0.0 or
      lambda_min_bound <= normalized_lambda_min)
  lambda_max_covered = normalized_lambda_max <= lambda_max_bound
  bound_source = (
      "weighted_block_gershgorin"
      if (theorem_applies and theorem_bound <=
          lambda_max_bound + structural_tolerance)
      else "numeric_only")
  return {
      "model": "block_gershgorin_interface_schur_spectral_certificate",
      "materializes_dense_schur": True,
      "normalization": "schur_block_diagonal",
      "interface_variable_count": int(len(interface_indices)),
      "block_dim": int(block_dim),
      "block_count": int(block_count),
      "positive_tolerance": float(positive_tolerance),
      "structural_tolerance": float(structural_tolerance),
      "normalized_lambda_min": float(normalized_lambda_min),
      "normalized_lambda_max": float(normalized_lambda_max),
      "normalized_lambda_raw_min": (
          float(eigenvalues[0]) if eigenvalues.size else 0.0),
      "normalized_lambda_raw_max": (
          float(eigenvalues[-1]) if eigenvalues.size else 0.0),
      "positive_eigenvalue_count": int(positive_eigenvalues.size),
      "lambda_min_bound": float(lambda_min_bound),
      "lambda_max_bound": float(lambda_max_bound),
      "lambda_min_covered": bool(lambda_min_covered),
      "lambda_max_covered": bool(lambda_max_covered),
      "interval_covered": bool(lambda_min_covered and lambda_max_covered),
      "lambda_max_bound_source": bound_source,
      "diagonal_blocks_spd": bool(diagonal_blocks_spd),
      "bad_diagonal_block_count": int(bad_diagonal_block_count),
      "diagonal_block_min_eigenvalue": float(
          diagonal_block_min_eigenvalue
          if np.isfinite(diagonal_block_min_eigenvalue) else 0.0),
      "diagonal_block_max_condition": float(diagonal_block_max_condition),
      "block_row_sum_max": float(np.max(row_sums)) if row_sums.size else 0.0,
      "block_row_sum_mean": float(np.mean(row_sums)) if row_sums.size else 0.0,
      "block_coupling_spectral_radius": float(spectral_radius),
      "weighted_block_gershgorin_lambda_max_bound": float(theorem_bound),
      "weighted_block_gershgorin_theorem_applies": bool(theorem_applies),
      "weighted_block_gershgorin_theorem_lambda_max_covered": bool(
          theorem_applies and
          normalized_lambda_max <= theorem_bound + structural_tolerance),
      "schur_stats": schur_stats,
  }


def distributed_block_perron_schur_spectral_certificate(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    block_dim: int,
    variable_count: int | None = None,
    damping: float = 0.0,
    lambda_max_bound: float = 2.0,
    perron_iterations: int = 32,
    perron_vector_mode: str = "lazy_power",
    resolvent_shift: float = 1e-2,
    positive_tolerance: float = 1e-10):
  """Block-Gershgorin Schur certificate from distributed block packets.

  This is the non-dense counterpart of
  ``block_gershgorin_interface_schur_spectral_certificate``.  Each local robot
  contributes only block Schur packets.  The certificate then runs a Perron
  power iteration on the nonnegative block-coupling graph and uses the
  Collatz-Wielandt upper bound ``max_i (Cx)_i / x_i``.  Any positive vector
  gives a valid upper bound, so the returned theorem flag is a certificate, not
  a heuristic eigenvalue estimate.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  block_dim = int(block_dim)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  if len(interface_indices) % block_dim != 0:
    raise ValueError("interface variable count must be divisible by block_dim")
  lambda_max_bound = float(lambda_max_bound)
  perron_iterations = max(0, int(perron_iterations))
  perron_vector_mode = str(perron_vector_mode)
  resolvent_shift = float(resolvent_shift)
  positive_tolerance = float(positive_tolerance)
  damping = float(damping)
  if not np.isfinite(lambda_max_bound) or lambda_max_bound <= 0.0:
    raise ValueError("lambda_max_bound must be positive and finite")
  if perron_vector_mode not in {"lazy_power", "resolvent", "cg_resolvent"}:
    raise ValueError(
        f"unsupported perron_vector_mode: {perron_vector_mode}")
  if not np.isfinite(resolvent_shift) or resolvent_shift < 0.0:
    raise ValueError("resolvent_shift must be nonnegative and finite")
  if not np.isfinite(positive_tolerance) or positive_tolerance < 0.0:
    raise ValueError("positive_tolerance must be nonnegative")
  if not np.isfinite(damping) or damping < 0.0:
    raise ValueError("damping must be nonnegative")

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
  if (int(np.min(interface_indices)) < 0 or
      int(np.max(interface_indices)) >= variable_count):
    raise ValueError("interface index outside variable dimension")

  block_count = len(interface_indices) // block_dim
  structural_tolerance = max(1e-10, 10.0 * positive_tolerance)
  diagonal_blocks = [
      np.zeros((block_dim, block_dim), dtype=float)
      for _ in range(block_count)
  ]
  offdiag_blocks: dict[tuple[int, int], np.ndarray] = {}
  diagonal_block_packet_count = 0
  offdiag_block_packet_count = 0
  local_private_variable_count_sum = 0
  local_singular_count = 0
  local_condition_max = 0.0
  uses_sparse_private_factorization = False
  dense_private_fallback_count = 0
  dense_candidate_block_pair_count = 0
  active_candidate_block_pair_count = 0
  candidate_block_pair_count = 0
  active_interface_block_visit_count = 0
  local_interface_component_count = 0
  component_schur_assembly_count = 0
  component_private_solve_count = 0
  component_schur_block_pair_loop_count = 0

  def dense_block(value):
    if hasattr(value, "toarray"):
      return np.asarray(value.toarray(), dtype=float)
    return np.asarray(value, dtype=float)

  def matrix_vector_product(matrix, vector):
    return np.asarray(matrix @ vector, dtype=float)

  block_index_arrays = [
      interface_indices[
          block_index * block_dim:(block_index + 1) * block_dim]
      for block_index in range(block_count)
  ]
  interface_global_block_ids = [
      int(block_indices[0]) // block_dim for block_indices in block_index_arrays
  ]

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    if hessian is None:
      raise ValueError(f"local system {system_index} is missing hessian")
    is_sparse_hessian = (
        hasattr(hessian, "tocsr") and not isinstance(hessian, np.ndarray))
    if is_sparse_hessian:
      hessian = hessian.tocsr()
    else:
      hessian = np.asarray(hessian, dtype=float)
    if hessian.shape != (variable_count, variable_count):
      raise ValueError(
          f"local system {system_index} hessian has wrong shape")
    active_blocks = []
    for block_index, block_indices in enumerate(block_index_arrays):
      if is_sparse_hessian:
        active = (
            hessian[block_indices, :].nnz > 0 or
            hessian[:, block_indices].nnz > 0)
      else:
        active = (
            np.count_nonzero(hessian[block_indices, :]) > 0 or
            np.count_nonzero(hessian[:, block_indices]) > 0)
      if active:
        active_blocks.append(block_index)
    dense_candidate_block_pair_count += block_count * block_count
    active_candidate_block_pair_count += len(active_blocks) * len(active_blocks)
    active_interface_block_visit_count += len(active_blocks)
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if private_indices.size:
      if (int(np.min(private_indices)) < 0 or
          int(np.max(private_indices)) >= variable_count):
        raise ValueError(
            f"local system {system_index} private index outside dimension")
      if set(int(index) for index in private_indices).intersection(
          int(index) for index in interface_indices):
        raise ValueError(
            f"local system {system_index} private/interface overlap")
    local_private_variable_count_sum += int(private_indices.size)
    private_global_blocks = sorted(set(
        int(index) // block_dim for index in private_indices))
    active_interface_global_blocks = {
        interface_global_block_ids[block_index]: block_index
        for block_index in active_blocks
    }
    component_nodes = set(active_interface_global_blocks)
    component_nodes.update(private_global_blocks)
    adjacency = {node: set() for node in component_nodes}
    if component_nodes:
      if is_sparse_hessian:
        nonzero_rows, nonzero_cols = hessian.nonzero()
      else:
        nonzero_rows, nonzero_cols = np.nonzero(hessian)
      for row_index, col_index in zip(nonzero_rows, nonzero_cols):
        row_block = int(row_index) // block_dim
        col_block = int(col_index) // block_dim
        if row_block not in component_nodes or col_block not in component_nodes:
          continue
        adjacency[row_block].add(col_block)
        adjacency[col_block].add(row_block)
    component_interface_blocks: list[list[int]] = []
    visited_component_nodes: set[int] = set()
    for node in sorted(component_nodes):
      if node in visited_component_nodes:
        continue
      stack = [node]
      visited_component_nodes.add(node)
      component = []
      while stack:
        current = stack.pop()
        component.append(current)
        for neighbor in adjacency.get(current, set()):
          if neighbor in visited_component_nodes:
            continue
          visited_component_nodes.add(neighbor)
          stack.append(neighbor)
      local_interface_blocks = sorted(
          active_interface_global_blocks[global_block]
          for global_block in component
          if global_block in active_interface_global_blocks)
      if local_interface_blocks:
        component_interface_blocks.append(local_interface_blocks)
    local_interface_component_count += len(component_interface_blocks)
    candidate_block_pair_count += sum(
        len(component) * len(component)
        for component in component_interface_blocks)

    private_solver = None
    if private_indices.size:
      h_pp = hessian[private_indices[:, None], private_indices]
      if is_sparse_hessian:
        if splu is None or identity is None:
          raise RuntimeError("scipy sparse support is required")
        h_pp = h_pp.tocsc()
        if damping > 0.0:
          h_pp = h_pp + damping * identity(h_pp.shape[0], format="csc")
        try:
          private_factor = splu(h_pp)
          uses_sparse_private_factorization = True

          def private_solver(rhs, factor=private_factor):
            return factor.solve(dense_block(rhs))
        except RuntimeError:
          local_singular_count += 1
          dense_private_fallback_count += 1
          dense_h_pp = h_pp.toarray()

          def private_solver(rhs, dense_h_pp=dense_h_pp):
            return np.linalg.pinv(dense_h_pp, rcond=1e-12) @ dense_block(rhs)
      else:
        h_pp = np.asarray(h_pp, dtype=float)
        if damping > 0.0:
          h_pp = h_pp + damping * np.eye(len(private_indices))
        condition = float(np.linalg.cond(h_pp))
        if np.isfinite(condition):
          local_condition_max = max(local_condition_max, condition)
        try:
          h_pp_inv = np.linalg.inv(h_pp)
        except np.linalg.LinAlgError:
          local_singular_count += 1
          h_pp_inv = np.linalg.pinv(h_pp, rcond=1e-12)

        def private_solver(rhs, h_pp_inv=h_pp_inv):
          return h_pp_inv @ dense_block(rhs)

    for component in component_interface_blocks:
      component_indices = np.concatenate([
          block_index_arrays[block_index] for block_index in component
      ]).astype(int)
      component_h_bb = dense_block(
          hessian[component_indices[:, None], component_indices])
      if private_solver is not None:
        component_h_pb = hessian[private_indices[:, None], component_indices]
        component_h_bp = hessian[component_indices[:, None], private_indices]
        private_response = private_solver(component_h_pb)
        component_schur = component_h_bb - matrix_vector_product(
            component_h_bp, private_response)
        component_private_solve_count += 1
      else:
        component_schur = component_h_bb
      component_schur = np.asarray(component_schur, dtype=float).reshape(
          len(component) * block_dim, len(component) * block_dim)
      component_schur_assembly_count += 1
      component_schur_block_pair_loop_count += len(component) * len(component)
      for row_position, row_block in enumerate(component):
        row_start = row_position * block_dim
        row_stop = row_start + block_dim
        for col_position, col_block in enumerate(component):
          col_start = col_position * block_dim
          col_stop = col_start + block_dim
          schur_block = component_schur[
              row_start:row_stop, col_start:col_stop]
          schur_block = np.asarray(schur_block, dtype=float).reshape(
              block_dim, block_dim)
          if row_block == col_block:
            diagonal_blocks[row_block] += schur_block
            if float(np.linalg.norm(schur_block, ord="fro")) > positive_tolerance:
              diagonal_block_packet_count += 1
          elif float(np.linalg.norm(schur_block, ord="fro")) > positive_tolerance:
            key = (row_block, col_block)
            if key not in offdiag_blocks:
              offdiag_blocks[key] = np.zeros((block_dim, block_dim), dtype=float)
            offdiag_blocks[key] += schur_block
            offdiag_block_packet_count += 1

  inv_sqrt_blocks: list[np.ndarray] = []
  diagonal_block_min_eigenvalue = float("inf")
  diagonal_block_max_condition = 0.0
  bad_diagonal_block_count = 0
  for diagonal_block in diagonal_blocks:
    diagonal_block = 0.5 * (diagonal_block + diagonal_block.T)
    eigenvalues, eigenvectors = np.linalg.eigh(diagonal_block)
    min_eigenvalue = float(eigenvalues[0]) if eigenvalues.size else 0.0
    max_eigenvalue = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    diagonal_block_min_eigenvalue = min(
        diagonal_block_min_eigenvalue, min_eigenvalue)
    if min_eigenvalue <= structural_tolerance:
      bad_diagonal_block_count += 1
    if min_eigenvalue > structural_tolerance:
      condition = max_eigenvalue / min_eigenvalue
      if np.isfinite(condition):
        diagonal_block_max_condition = max(
            diagonal_block_max_condition, float(condition))
    safe = np.maximum(eigenvalues, 1e-12)
    inv_sqrt_blocks.append((eigenvectors / np.sqrt(safe)) @ eigenvectors.T)

  coupling_edges: dict[tuple[int, int], float] = {}
  row_sums = np.zeros(block_count, dtype=float)
  for (row_block, col_block), schur_block in offdiag_blocks.items():
    normalized_block = (
        inv_sqrt_blocks[row_block] @ schur_block @ inv_sqrt_blocks[col_block])
    weight = float(np.linalg.norm(normalized_block, ord=2))
    if weight > positive_tolerance:
      coupling_edges[(row_block, col_block)] = weight
      row_sums[row_block] += weight

  def coupling_matvec(vector):
    output = np.zeros(block_count, dtype=float)
    for (row_block, col_block), weight in coupling_edges.items():
      output[row_block] += weight * vector[col_block]
    return output

  x = np.ones(block_count, dtype=float)
  if block_count:
    x /= float(block_count)
  positive_floor = max(1e-15, positive_tolerance)
  perron_scalar_message_count = 0
  perron_global_reduction_count = 0
  cg_resolvent_iterations = 0
  cg_resolvent_final_residual = 0.0
  lazy_shift = max(1.0, float(np.max(row_sums)) if row_sums.size else 1.0)
  if perron_vector_mode == "lazy_power":
    for _ in range(perron_iterations):
      y = coupling_matvec(x) + lazy_shift * x
      perron_scalar_message_count += len(coupling_edges)
      if float(np.linalg.norm(y, ord=1)) <= positive_floor:
        break
      y = np.maximum(y, positive_floor)
      x = y / max(float(np.linalg.norm(y, ord=1)), positive_floor)
  elif perron_vector_mode == "resolvent":
    x = np.ones(block_count, dtype=float)
    for _ in range(perron_iterations):
      y = (np.ones(block_count, dtype=float) + coupling_matvec(x)) / (
          1.0 + resolvent_shift)
      perron_scalar_message_count += len(coupling_edges)
      if not np.all(np.isfinite(y)):
        break
      x = np.maximum(y, positive_floor)
  else:
    rhs = np.ones(block_count, dtype=float)

    def resolvent_matvec(vector):
      return (1.0 + resolvent_shift) * vector - coupling_matvec(vector)

    x = np.zeros(block_count, dtype=float)
    residual = rhs - resolvent_matvec(x)
    perron_scalar_message_count += len(coupling_edges)
    direction = residual.copy()
    residual_norm_sq = float(residual @ residual)
    perron_global_reduction_count += 1
    cg_tolerance_sq = max(positive_floor, 1e-12) ** 2 * max(
        1.0, float(rhs @ rhs))
    if residual_norm_sq <= cg_tolerance_sq:
      cg_resolvent_final_residual = math.sqrt(max(0.0, residual_norm_sq))
    else:
      for iteration in range(perron_iterations):
        matvec_direction = resolvent_matvec(direction)
        perron_scalar_message_count += len(coupling_edges)
        direction_curvature = float(direction @ matvec_direction)
        perron_global_reduction_count += 1
        if (not np.isfinite(direction_curvature) or
            direction_curvature <= positive_floor):
          break
        step = residual_norm_sq / direction_curvature
        x = x + step * direction
        residual = residual - step * matvec_direction
        next_residual_norm_sq = float(residual @ residual)
        perron_global_reduction_count += 1
        cg_resolvent_iterations = iteration + 1
        cg_resolvent_final_residual = math.sqrt(
            max(0.0, next_residual_norm_sq))
        if next_residual_norm_sq <= cg_tolerance_sq:
          residual_norm_sq = next_residual_norm_sq
          break
        beta = next_residual_norm_sq / max(residual_norm_sq, positive_floor)
        direction = residual + beta * direction
        residual_norm_sq = next_residual_norm_sq
    x = np.maximum(x, positive_floor)

  cx = coupling_matvec(x)
  perron_scalar_message_count += len(coupling_edges)
  safe_x = np.maximum(x, positive_floor)
  ratios = cx / safe_x if block_count else np.zeros(0)
  collatz_upper = float(np.max(ratios)) if ratios.size else 0.0
  collatz_lower = float(np.min(ratios)) if ratios.size else 0.0
  rayleigh = (
      float(x @ cx / max(float(x @ x), positive_floor))
      if block_count else 0.0)
  diagonal_blocks_spd = bool(bad_diagonal_block_count == 0)
  theorem_bound = 1.0 + collatz_upper
  theorem_applies = bool(
      diagonal_blocks_spd and
      collatz_upper <= 1.0 + structural_tolerance)
  bound_source = (
      "distributed_collatz_perron"
      if theorem_applies and theorem_bound <=
      lambda_max_bound + structural_tolerance
      else "numeric_only")
  double_bytes = 8
  block_packet_bytes = block_dim * block_dim * double_bytes
  diagonal_payload_bytes = diagonal_block_packet_count * block_packet_bytes
  offdiag_payload_bytes = offdiag_block_packet_count * block_packet_bytes
  perron_payload_bytes = perron_scalar_message_count * double_bytes
  certificate_payload_bytes = (
      diagonal_payload_bytes + offdiag_payload_bytes + perron_payload_bytes)
  return {
      "model": "distributed_block_perron_schur_spectral_certificate",
      "materializes_dense_schur": False,
      "materializes_dense_coupling": False,
      "normalization": "schur_block_diagonal",
      "block_packet_construction": "component_dense_schur",
      "interface_variable_count": int(len(interface_indices)),
      "block_dim": int(block_dim),
      "block_count": int(block_count),
      "positive_tolerance": float(positive_tolerance),
      "structural_tolerance": float(structural_tolerance),
      "lambda_max_bound": float(lambda_max_bound),
      "lambda_max_bound_source": bound_source,
      "diagonal_blocks_spd": bool(diagonal_blocks_spd),
      "bad_diagonal_block_count": int(bad_diagonal_block_count),
      "diagonal_block_min_eigenvalue": float(
          diagonal_block_min_eigenvalue
          if np.isfinite(diagonal_block_min_eigenvalue) else 0.0),
      "diagonal_block_max_condition": float(diagonal_block_max_condition),
      "block_row_sum_max": float(np.max(row_sums)) if row_sums.size else 0.0,
      "block_row_sum_mean": float(np.mean(row_sums)) if row_sums.size else 0.0,
      "block_coupling_edge_count": int(len(coupling_edges)),
      "dense_candidate_block_pair_count": int(dense_candidate_block_pair_count),
      "active_candidate_block_pair_count": int(
          active_candidate_block_pair_count),
      "candidate_block_pair_count": int(candidate_block_pair_count),
      "active_interface_block_visit_count": int(
          active_interface_block_visit_count),
      "local_interface_component_count": int(local_interface_component_count),
      "component_schur_assembly_count": int(component_schur_assembly_count),
      "component_private_solve_count": int(component_private_solve_count),
      "component_schur_block_pair_loop_count": int(
          component_schur_block_pair_loop_count),
      "skipped_inactive_block_pair_count": int(
          dense_candidate_block_pair_count - active_candidate_block_pair_count),
      "skipped_cross_component_block_pair_count": int(
          active_candidate_block_pair_count - candidate_block_pair_count),
      "block_coupling_collatz_lower_bound": float(collatz_lower),
      "block_coupling_collatz_upper_bound": float(collatz_upper),
      "block_coupling_perron_rayleigh": float(rayleigh),
      "perron_iterations": int(perron_iterations),
      "perron_vector_mode": perron_vector_mode,
      "perron_lazy_shift": float(lazy_shift),
      "resolvent_shift": float(resolvent_shift),
      "perron_global_reduction_count": int(perron_global_reduction_count),
      "cg_resolvent_iterations": int(cg_resolvent_iterations),
      "cg_resolvent_final_residual": float(cg_resolvent_final_residual),
      "perron_scalar_message_count": int(perron_scalar_message_count),
      "weighted_block_gershgorin_lambda_max_bound": float(theorem_bound),
      "weighted_block_gershgorin_theorem_applies": bool(theorem_applies),
      "weighted_block_gershgorin_theorem_lambda_max_covered": bool(
          theorem_applies and theorem_bound <=
          lambda_max_bound + structural_tolerance),
      "diagonal_block_packet_count": int(diagonal_block_packet_count),
      "offdiag_block_packet_count": int(offdiag_block_packet_count),
      "certificate_payload_bytes": int(certificate_payload_bytes),
      "diagonal_payload_bytes": int(diagonal_payload_bytes),
      "offdiag_payload_bytes": int(offdiag_payload_bytes),
      "perron_payload_bytes": int(perron_payload_bytes),
      "local_private_variable_count_sum": int(local_private_variable_count_sum),
      "local_singular_count": int(local_singular_count),
      "local_condition_max": float(local_condition_max),
      "uses_sparse_private_factorization": bool(uses_sparse_private_factorization),
      "dense_private_fallback_count": int(dense_private_fallback_count),
  }


def estimate_distributed_block_perron_schur_certificate_preflight(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    block_dim: int,
    variable_count: int | None = None,
    perron_iterations: int = 32,
    perron_vector_mode: str = "lazy_power"):
  """Conservative preflight payload bound for the distributed certificate.

  The estimate is structural: it walks the same active interface components as
  the certificate, but does not solve private Schur responses.  Therefore it can
  upper-bound block-packet and Perron scalar payload before spending the
  certificate setup cost.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  block_dim = int(block_dim)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  if len(interface_indices) % block_dim != 0:
    raise ValueError("interface variable count must be divisible by block_dim")
  perron_iterations = max(0, int(perron_iterations))
  perron_vector_mode = str(perron_vector_mode)
  if perron_vector_mode not in {"lazy_power", "resolvent", "cg_resolvent"}:
    raise ValueError(
        f"unsupported perron_vector_mode: {perron_vector_mode}")
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
  if (int(np.min(interface_indices)) < 0 or
      int(np.max(interface_indices)) >= variable_count):
    raise ValueError("interface index outside variable dimension")

  block_count = len(interface_indices) // block_dim
  block_index_arrays = [
      interface_indices[
          block_index * block_dim:(block_index + 1) * block_dim]
      for block_index in range(block_count)
  ]
  interface_global_block_ids = [
      int(block_indices[0]) // block_dim for block_indices in block_index_arrays
  ]
  dense_candidate_block_pair_count = 0
  active_candidate_block_pair_count = 0
  candidate_block_pair_count = 0
  component_diagonal_pair_count = 0
  active_interface_block_visit_count = 0
  local_interface_component_count = 0

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    if hessian is None:
      raise ValueError(f"local system {system_index} is missing hessian")
    is_sparse_hessian = (
        hasattr(hessian, "tocsr") and not isinstance(hessian, np.ndarray))
    if is_sparse_hessian:
      hessian = hessian.tocsr()
    else:
      hessian = np.asarray(hessian, dtype=float)
    if hessian.shape != (variable_count, variable_count):
      raise ValueError(
          f"local system {system_index} hessian has wrong shape")
    active_blocks = []
    for block_index, block_indices in enumerate(block_index_arrays):
      if is_sparse_hessian:
        active = (
            hessian[block_indices, :].nnz > 0 or
            hessian[:, block_indices].nnz > 0)
      else:
        active = (
            np.count_nonzero(hessian[block_indices, :]) > 0 or
            np.count_nonzero(hessian[:, block_indices]) > 0)
      if active:
        active_blocks.append(block_index)
    dense_candidate_block_pair_count += block_count * block_count
    active_candidate_block_pair_count += len(active_blocks) * len(active_blocks)
    active_interface_block_visit_count += len(active_blocks)
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if private_indices.size:
      if (int(np.min(private_indices)) < 0 or
          int(np.max(private_indices)) >= variable_count):
        raise ValueError(
            f"local system {system_index} private index outside dimension")
      if set(int(index) for index in private_indices).intersection(
          int(index) for index in interface_indices):
        raise ValueError(
            f"local system {system_index} private/interface overlap")
    private_global_blocks = sorted(set(
        int(index) // block_dim for index in private_indices))
    active_interface_global_blocks = {
        interface_global_block_ids[block_index]: block_index
        for block_index in active_blocks
    }
    component_nodes = set(active_interface_global_blocks)
    component_nodes.update(private_global_blocks)
    adjacency = {node: set() for node in component_nodes}
    if component_nodes:
      if is_sparse_hessian:
        nonzero_rows, nonzero_cols = hessian.nonzero()
      else:
        nonzero_rows, nonzero_cols = np.nonzero(hessian)
      for row_index, col_index in zip(nonzero_rows, nonzero_cols):
        row_block = int(row_index) // block_dim
        col_block = int(col_index) // block_dim
        if row_block not in component_nodes or col_block not in component_nodes:
          continue
        adjacency[row_block].add(col_block)
        adjacency[col_block].add(row_block)
    visited_component_nodes: set[int] = set()
    for node in sorted(component_nodes):
      if node in visited_component_nodes:
        continue
      stack = [node]
      visited_component_nodes.add(node)
      component = []
      while stack:
        current = stack.pop()
        component.append(current)
        for neighbor in adjacency.get(current, set()):
          if neighbor in visited_component_nodes:
            continue
          visited_component_nodes.add(neighbor)
          stack.append(neighbor)
      local_interface_blocks = sorted(
          active_interface_global_blocks[global_block]
          for global_block in component
          if global_block in active_interface_global_blocks)
      if local_interface_blocks:
        local_interface_component_count += 1
        component_size = len(local_interface_blocks)
        candidate_block_pair_count += component_size * component_size
        component_diagonal_pair_count += component_size

  double_bytes = 8
  block_packet_bytes = block_dim * block_dim * double_bytes
  block_packet_payload_upper_bytes = (
      candidate_block_pair_count * block_packet_bytes)
  coupling_edge_upper_count = max(
      0, candidate_block_pair_count - component_diagonal_pair_count)
  if perron_vector_mode == "cg_resolvent":
    perron_scalar_message_count_upper = (
        coupling_edge_upper_count * (perron_iterations + 2))
    perron_global_reduction_count_upper = 1 + 2 * perron_iterations
  else:
    perron_scalar_message_count_upper = (
        coupling_edge_upper_count * (perron_iterations + 1))
    perron_global_reduction_count_upper = 0
  perron_payload_upper_bytes = (
      perron_scalar_message_count_upper * double_bytes)
  payload_upper_bytes = (
      block_packet_payload_upper_bytes + perron_payload_upper_bytes)
  return {
      "model": "distributed_block_perron_schur_certificate_preflight",
      "interface_variable_count": int(len(interface_indices)),
      "block_dim": int(block_dim),
      "block_count": int(block_count),
      "perron_iterations": int(perron_iterations),
      "perron_vector_mode": perron_vector_mode,
      "dense_candidate_block_pair_count": int(dense_candidate_block_pair_count),
      "active_candidate_block_pair_count": int(
          active_candidate_block_pair_count),
      "candidate_block_pair_count": int(candidate_block_pair_count),
      "component_diagonal_pair_count": int(component_diagonal_pair_count),
      "coupling_edge_upper_count": int(coupling_edge_upper_count),
      "active_interface_block_visit_count": int(
          active_interface_block_visit_count),
      "local_interface_component_count": int(local_interface_component_count),
      "block_packet_payload_upper_bytes": int(
          block_packet_payload_upper_bytes),
      "perron_scalar_message_count_upper": int(
          perron_scalar_message_count_upper),
      "perron_payload_upper_bytes": int(perron_payload_upper_bytes),
      "payload_upper_bytes": int(payload_upper_bytes),
      "payload_upper_mb": _bytes_to_mb(payload_upper_bytes),
      "perron_global_reduction_count_upper": int(
          perron_global_reduction_count_upper),
  }


def _deterministic_spectral_probe_vector(size: int, seed: int):
  size = int(size)
  if size <= 0:
    raise ValueError("probe vector size must be positive")
  seed = int(seed)
  vector = np.asarray([
      math.sin((index + 1) * (seed + 1) * 0.6180339887498948) +
      math.cos((index + 1) * (seed + 3) * 0.4142135623730951)
      for index in range(size)
  ], dtype=float)
  norm = float(np.linalg.norm(vector))
  if norm <= 1e-14:
    vector = np.arange(1, size + 1, dtype=float)
    norm = float(np.linalg.norm(vector))
  return vector / max(norm, 1e-14)


def _matrix_free_ritz_largest_eigen_probe_from_matvec(
    matvec,
    size: int,
    probe_iterations: int,
    seed: int = 0,
    safety_factor: float = 1.05,
    tolerance: float = 1e-12):
  """Estimates the largest eigenvalue of a symmetric operator by Lanczos."""
  size = int(size)
  probe_iterations = max(1, int(probe_iterations))
  safety_factor = float(safety_factor)
  tolerance = float(tolerance)
  if size <= 0:
    raise ValueError("probe size must be positive")
  if not np.isfinite(safety_factor) or safety_factor < 1.0:
    raise ValueError("safety_factor must be finite and >= 1")
  if not np.isfinite(tolerance) or tolerance < 0.0:
    raise ValueError("tolerance must be nonnegative")

  q = _deterministic_spectral_probe_vector(size, seed)
  q_previous = np.zeros(size, dtype=float)
  basis: list[np.ndarray] = []
  alphas: list[float] = []
  betas: list[float] = []
  matvec_count = 0
  # The initial distributed norm and each Lanczos alpha/beta/reorthogonalizing
  # inner product need scalar reductions. The count is a communication model,
  # not a numerical necessity in this single-process diagnostic.
  global_reduction_count = 1
  for iteration in range(min(probe_iterations, size)):
    basis.append(q.copy())
    z = np.asarray(matvec(q), dtype=float).reshape(-1)
    if z.shape[0] != size:
      raise ValueError("probe matvec returned wrong dimension")
    matvec_count += 1
    alpha = float(q @ z)
    global_reduction_count += 1
    z = z - alpha * q
    if iteration > 0:
      z = z - betas[-1] * q_previous
    for old_q in basis:
      correction = float(old_q @ z)
      global_reduction_count += 1
      z = z - correction * old_q
    beta = float(np.linalg.norm(z))
    global_reduction_count += 1
    alphas.append(alpha)
    if beta <= tolerance or iteration == min(probe_iterations, size) - 1:
      break
    betas.append(beta)
    q_previous = q
    q = z / beta

  rank = len(alphas)
  tridiagonal = np.diag(np.asarray(alphas, dtype=float))
  if rank > 1:
    offdiag = np.asarray(betas[:rank - 1], dtype=float)
    tridiagonal += np.diag(offdiag, k=1) + np.diag(offdiag, k=-1)
  tridiagonal = 0.5 * (tridiagonal + tridiagonal.T)
  ritz_values = np.linalg.eigvalsh(tridiagonal) if rank else np.zeros(0)
  ritz_lambda_max = float(ritz_values[-1]) if ritz_values.size else 0.0
  safe_lambda_max = max(0.0, safety_factor * ritz_lambda_max)
  return {
      "model": "matrix_free_ritz_largest_eigen_probe",
      "probe_iterations": int(probe_iterations),
      "probe_basis_rank": int(rank),
      "seed": int(seed),
      "safety_factor": float(safety_factor),
      "ritz_lambda_max": float(ritz_lambda_max),
      "safe_lambda_max": float(safe_lambda_max),
      "ritz_values": [float(value) for value in ritz_values],
      "matvec_count": int(matvec_count),
      "global_reduction_count": int(global_reduction_count),
  }


def matrix_free_interface_schur_ritz_spectral_probe(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0,
    schur_preconditioner: str = "diagonal",
    dirichlet_diagonal_floor: float = 0.25,
    probe_iterations: int = 8,
    seed: int = 0,
    safety_factor: float = 1.05,
    communication_model: str = "separator_owner_star",
    robot_topology_edges: list[tuple[object, object]] | None = None):
  """Matrix-free Ritz probe for the normalized interface Schur spectrum.

  The probe targets ``M^{-1/2} S M^{-1/2}``, where ``M`` is the same diagonal
  used by the fixed-step Schur preconditioner. It is a diagnostic spectral
  estimator, not a theorem-level bound: the returned safe endpoint is the Ritz
  estimate multiplied by a user-controlled safety factor.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
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
  if schur_preconditioner not in {
      "none",
      "diagonal",
      "boundary_diagonal",
      "dirichlet_clipped_diagonal",
  }:
    raise ValueError(f"unsupported schur_preconditioner: {schur_preconditioner}")
  dirichlet_diagonal_floor = float(dirichlet_diagonal_floor)
  if (not np.isfinite(dirichlet_diagonal_floor) or
      dirichlet_diagonal_floor <= 0.0 or
      dirichlet_diagonal_floor > 1.0):
    raise ValueError("dirichlet_diagonal_floor must be in (0, 1]")

  local_data = []
  dense_local_hessian_materialized = False
  uses_sparse_private_factorization = False
  dense_private_fallback_count = 0
  local_singular_count = 0

  def matrix_vector_product(matrix, vector):
    return np.asarray(matrix @ vector, dtype=float).reshape(-1)

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    gradient = system.get("gradient")
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if hessian is None or gradient is None:
      raise ValueError(f"local system {system_index} is missing hessian/gradient")
    is_sparse_hessian = (
        hasattr(hessian, "tocsr") and not isinstance(hessian, np.ndarray))
    if is_sparse_hessian:
      hessian = hessian.tocsr()
    else:
      hessian = np.asarray(hessian, dtype=float)
      dense_local_hessian_materialized = True
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
    if private_indices.size == 0:
      local_data.append({
          "h_bb": h_bb,
          "h_pb": None,
          "h_bp": None,
          "private_solver": None,
      })
      continue
    h_pp = hessian[private_indices[:, None], private_indices]
    h_pb = hessian[private_indices[:, None], interface_indices]
    h_bp = hessian[interface_indices[:, None], private_indices]
    if is_sparse_hessian:
      if splu is None or identity is None:
        raise RuntimeError("scipy sparse support is required")
      h_pp = h_pp.tocsc()
      if damping > 0.0:
        h_pp = h_pp + float(damping) * identity(
            h_pp.shape[0], format="csc")
      try:
        private_factor = splu(h_pp)
        uses_sparse_private_factorization = True

        def private_solver(rhs, factor=private_factor):
          return factor.solve(np.asarray(rhs, dtype=float).reshape(-1))
      except RuntimeError:
        local_singular_count += 1
        dense_private_fallback_count += 1
        dense_h_pp = h_pp.toarray()
        dense_local_hessian_materialized = True

        def private_solver(rhs, dense_h_pp=dense_h_pp):
          return np.linalg.pinv(dense_h_pp, rcond=1e-12) @ np.asarray(
              rhs, dtype=float).reshape(-1)
    else:
      if damping > 0.0:
        h_pp = h_pp + float(damping) * np.eye(len(private_indices))
      try:
        h_pp_inv = np.linalg.inv(h_pp)
      except np.linalg.LinAlgError:
        local_singular_count += 1
        h_pp_inv = np.linalg.pinv(h_pp, rcond=1e-12)

      def private_solver(rhs, h_pp_inv=h_pp_inv):
        return h_pp_inv @ np.asarray(rhs, dtype=float).reshape(-1)
    local_data.append({
        "h_bb": h_bb,
        "h_pb": h_pb,
        "h_bp": h_bp,
        "private_solver": private_solver,
    })

  interface_matvec_count = 0

  def schur_matvec(vector):
    nonlocal interface_matvec_count
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if len(vector) != len(interface_indices):
      raise ValueError("interface vector dimension mismatch")
    output = np.zeros_like(vector)
    for data in local_data:
      output += matrix_vector_product(data["h_bb"], vector)
      private_solver = data["private_solver"]
      if private_solver is not None:
        private_rhs = matrix_vector_product(data["h_pb"], vector)
        output -= matrix_vector_product(data["h_bp"],
                                        private_solver(private_rhs))
    interface_matvec_count += 1
    return output

  preconditioner_private_solve_count = 0

  def build_diagonal_preconditioner():
    nonlocal preconditioner_private_solve_count
    diagonal = np.zeros(len(interface_indices), dtype=float)
    boundary_diagonal = np.zeros(len(interface_indices), dtype=float)
    for data in local_data:
      h_bb_diag = np.asarray(data["h_bb"].diagonal(), dtype=float).reshape(-1)
      diagonal += h_bb_diag
      boundary_diagonal += h_bb_diag
      if schur_preconditioner == "boundary_diagonal":
        continue
      private_solver = data["private_solver"]
      if private_solver is None:
        continue
      for col in range(len(interface_indices)):
        basis = np.zeros(len(interface_indices), dtype=float)
        basis[col] = 1.0
        private_rhs = matrix_vector_product(data["h_pb"], basis)
        response = private_solver(private_rhs)
        preconditioner_private_solve_count += 1
        diagonal[col] -= matrix_vector_product(
            data["h_bp"][col:col + 1, :], response)[0]
    if schur_preconditioner == "dirichlet_clipped_diagonal":
      boundary_diagonal = np.maximum(np.abs(boundary_diagonal), 1e-12)
      diagonal = np.maximum(
          np.maximum(np.abs(diagonal), 1e-12),
          float(dirichlet_diagonal_floor) * boundary_diagonal)
    return np.maximum(np.abs(diagonal), 1e-12)

  if schur_preconditioner in {
      "diagonal",
      "boundary_diagonal",
      "dirichlet_clipped_diagonal",
  }:
    preconditioner_diagonal = build_diagonal_preconditioner()
  else:
    preconditioner_diagonal = np.ones(len(interface_indices), dtype=float)
  sqrt_diagonal = np.sqrt(np.maximum(preconditioner_diagonal, 1e-12))

  def normalized_schur_matvec(vector):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    return schur_matvec(vector / sqrt_diagonal) / sqrt_diagonal

  probe_stats = _matrix_free_ritz_largest_eigen_probe_from_matvec(
      normalized_schur_matvec,
      size=len(interface_indices),
      probe_iterations=probe_iterations,
      seed=seed,
      safety_factor=safety_factor)
  communication_stats = estimate_local_interface_schur_pcg_communication(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_matvec_count=interface_matvec_count,
      pcg_global_reduction_count=int(probe_stats["global_reduction_count"]),
      rhs_reduction_count=0,
      scalar_bytes=8,
      communication_model=communication_model,
      robot_topology_edges=robot_topology_edges)
  return {
      "model": "matrix_free_interface_schur_ritz_spectral_probe",
      "materializes_dense_schur": False,
      "materializes_dense_local_hessian": bool(
          dense_local_hessian_materialized),
      "uses_sparse_private_factorization": bool(
          uses_sparse_private_factorization),
      "dense_private_fallback_count": int(dense_private_fallback_count),
      "local_system_count": int(len(local_systems)),
      "interface_variable_count": int(len(interface_indices)),
      "variable_count": int(variable_count),
      "local_singular_count": int(local_singular_count),
      "damping": float(damping),
      "schur_preconditioner": schur_preconditioner,
      "dirichlet_diagonal_floor": float(dirichlet_diagonal_floor),
      "schur_preconditioner_private_solve_count": int(
          preconditioner_private_solve_count),
      "schur_preconditioner_diagonal_min": float(
          np.min(preconditioner_diagonal)) if preconditioner_diagonal.size else 0.0,
      "schur_preconditioner_diagonal_max": float(
          np.max(preconditioner_diagonal)) if preconditioner_diagonal.size else 0.0,
      "probe_iterations": int(probe_stats["probe_iterations"]),
      "probe_basis_rank": int(probe_stats["probe_basis_rank"]),
      "seed": int(seed),
      "safety_factor": float(safety_factor),
      "ritz_lambda_max": float(probe_stats["ritz_lambda_max"]),
      "safe_lambda_max": float(probe_stats["safe_lambda_max"]),
      "ritz_values": list(probe_stats["ritz_values"]),
      "interface_matvec_count": int(interface_matvec_count),
      "global_reduction_count": int(probe_stats["global_reduction_count"]),
      **communication_stats,
  }


def local_interface_schur_residual_envelope(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_state: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0):
  """Bounds the exact interface residual by robot-local residual norms.

  For a candidate interface state x_B, each robot can form its own Schur
  residual contribution r_r = b_r - S_r x_B. The centralized residual is
  sum_r r_r, so sum_r ||r_r|| is a topology-local upper envelope on its norm.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
  interface_state = np.asarray(interface_state, dtype=float).reshape(-1)
  if interface_state.shape[0] != len(interface_indices):
    raise ValueError("interface_state dimension mismatch")
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

  exact_residual = np.zeros(len(interface_indices), dtype=float)
  local_residual_norms = []
  local_singular_count = 0
  local_private_variable_count_sum = 0

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
    local_private_variable_count_sum += int(private_indices.size)

    h_bb = hessian[interface_indices[:, None], interface_indices]
    g_b = gradient[interface_indices]
    local_matvec = h_bb @ interface_state
    local_rhs = g_b.copy()
    if private_indices.size:
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
      local_matvec -= h_bp @ (private_interface_response @ interface_state)
      local_rhs -= h_bp @ private_rhs_response
    local_residual = local_rhs - local_matvec
    exact_residual += local_residual
    local_residual_norms.append(float(np.linalg.norm(local_residual)))

  local_residual_norms_array = np.asarray(local_residual_norms, dtype=float)
  return {
      "model": "local_interface_schur_residual_envelope",
      "materializes_dense_schur": False,
      "local_system_count": int(len(local_systems)),
      "interface_variable_count": int(len(interface_indices)),
      "variable_count": int(variable_count),
      "local_private_variable_count_sum": int(local_private_variable_count_sum),
      "local_singular_count": int(local_singular_count),
      "damping": float(damping),
      "exact_residual": exact_residual,
      "local_residual_norms": local_residual_norms_array,
      "exact_residual_norm": float(np.linalg.norm(exact_residual)),
      "envelope_residual_norm": float(np.sum(local_residual_norms_array)),
      "global_reduction_count": 0,
  }


def solve_local_interface_schur_reference(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0):
  """Solves a CCI-equivalent local-Schur reference system.

  This is a correctness reference for the new DCI route. It still solves the
  assembled interface Schur system densely, so it is not the final deployment
  communication algorithm. Its purpose is to verify the decomposition:
  local Schur contributions + one interface solve + local private
  back-substitution should reproduce the centralized normal-equation solution.
  """
  schur, rhs, schur_stats = sum_local_interface_schur_contributions(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping)
  variable_count = int(schur_stats["variable_count"])
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)

  try:
    interface_solution = np.linalg.solve(schur, rhs)
    interface_singular = False
  except np.linalg.LinAlgError:
    interface_solution = np.linalg.pinv(schur, rcond=1e-12) @ rhs
    interface_singular = True

  solution = np.zeros(variable_count, dtype=float)
  assigned = np.zeros(variable_count, dtype=bool)
  solution[interface_indices] = interface_solution
  assigned[interface_indices] = True
  global_hessian = np.zeros((variable_count, variable_count), dtype=float)
  global_gradient = np.zeros(variable_count, dtype=float)
  duplicate_private_count = 0
  private_backsubstitution_count = 0

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    gradient = system.get("gradient")
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if hasattr(hessian, "toarray"):
      hessian = hessian.toarray()
    hessian = np.asarray(hessian, dtype=float)
    gradient = np.asarray(gradient, dtype=float).reshape(-1)
    global_hessian += hessian
    global_gradient += gradient
    if private_indices.size == 0:
      continue
    already_assigned = assigned[private_indices]
    if np.any(already_assigned):
      duplicate_private_count += int(np.count_nonzero(already_assigned))
      raise ValueError(
          f"local system {system_index} reuses private variables")
    h_pp = hessian[private_indices[:, None], private_indices]
    if damping > 0.0:
      h_pp = h_pp + float(damping) * np.eye(len(private_indices))
    h_pb = hessian[private_indices[:, None], interface_indices]
    g_p = gradient[private_indices]
    private_rhs = g_p - h_pb @ interface_solution
    try:
      private_solution = np.linalg.solve(h_pp, private_rhs)
    except np.linalg.LinAlgError:
      private_solution = np.linalg.pinv(h_pp, rcond=1e-12) @ private_rhs
    solution[private_indices] = private_solution
    assigned[private_indices] = True
    private_backsubstitution_count += 1

  normal_residual = global_hessian @ solution - global_gradient
  unassigned_variable_count = int(np.count_nonzero(~assigned))
  return solution, {
      **schur_stats,
      "model": "local_interface_schur_reference_solve",
      "interface_singular": bool(interface_singular),
      "schur_condition": float(np.linalg.cond(schur)),
      "interface_solution_norm": float(np.linalg.norm(interface_solution)),
      "private_backsubstitution_count": int(private_backsubstitution_count),
      "duplicate_private_variable_count": int(duplicate_private_count),
      "unassigned_variable_count": int(unassigned_variable_count),
      "final_normal_residual": float(np.linalg.norm(normal_residual)),
  }


def local_interface_schur_gap_certificate(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    candidate_solution: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0):
  """Measures a candidate solution's gap to the local-Schur reference.

  This is the certificate used by the KKT-consistent DCI route. It separates
  interface error, exact Schur residual/energy, and private-variable
  consistency conditioned on the candidate interface state.
  """
  schur, rhs, schur_stats = sum_local_interface_schur_contributions(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping)
  variable_count = int(schur_stats["variable_count"])
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  candidate_solution = np.asarray(candidate_solution, dtype=float).reshape(-1)
  if len(candidate_solution) != variable_count:
    raise ValueError("candidate_solution dimension does not match variable_count")

  try:
    reference_interface = np.linalg.solve(schur, rhs)
    schur_singular = False
  except np.linalg.LinAlgError:
    reference_interface = np.linalg.pinv(schur, rcond=1e-12) @ rhs
    schur_singular = True
  reference_solution, _ = solve_local_interface_schur_reference(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping)

  candidate_interface = candidate_solution[interface_indices]
  interface_error = candidate_interface - reference_interface
  schur_residual = schur @ candidate_interface - rhs
  try:
    dual_solution = np.linalg.solve(schur, schur_residual)
  except np.linalg.LinAlgError:
    dual_solution = np.linalg.pinv(schur, rcond=1e-12) @ schur_residual
  schur_energy_gap = float(interface_error.T @ schur @ interface_error)
  schur_residual_dual_energy = float(schur_residual.T @ dual_solution)

  global_hessian = np.zeros((variable_count, variable_count), dtype=float)
  global_gradient = np.zeros(variable_count, dtype=float)
  assigned_private = np.zeros(variable_count, dtype=bool)
  private_consistency_sq = 0.0
  private_consistency_max = 0.0
  private_variable_count = 0
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
      raise ValueError(
          f"local system {system_index} hessian has wrong shape")
    if gradient.shape[0] != variable_count:
      raise ValueError(
          f"local system {system_index} gradient has wrong dimension")
    global_hessian += hessian
    global_gradient += gradient
    if private_indices.size == 0:
      continue
    if np.any(assigned_private[private_indices]):
      duplicate_private_count += int(np.count_nonzero(
          assigned_private[private_indices]))
      raise ValueError(
          f"local system {system_index} reuses private variables")
    h_pp = hessian[private_indices[:, None], private_indices]
    if damping > 0.0:
      h_pp = h_pp + float(damping) * np.eye(len(private_indices))
    h_pb = hessian[private_indices[:, None], interface_indices]
    g_p = gradient[private_indices]
    conditional_rhs = g_p - h_pb @ candidate_interface
    try:
      conditional_private = np.linalg.solve(h_pp, conditional_rhs)
    except np.linalg.LinAlgError:
      local_singular_count += 1
      conditional_private = np.linalg.pinv(h_pp, rcond=1e-12) @ conditional_rhs
    private_error = candidate_solution[private_indices] - conditional_private
    private_consistency_sq += float(private_error @ private_error)
    private_consistency_max = max(
        private_consistency_max, float(np.linalg.norm(private_error)))
    private_variable_count += int(private_indices.size)
    private_backsubstitution_count += 1
    assigned_private[private_indices] = True

  normal_residual = global_hessian @ candidate_solution - global_gradient
  return {
      "model": "local_interface_schur_gap_certificate",
      "diagnostic_model": "local_schur_energy_certificate",
      "uses_dense_schur_for_certificate": True,
      "local_system_count": int(len(local_systems)),
      "interface_variable_count": int(len(interface_indices)),
      "private_variable_count": int(private_variable_count),
      "variable_count": int(variable_count),
      "damping": float(damping),
      "schur_singular": bool(schur_singular),
      "schur_condition": float(np.linalg.cond(schur)),
      "schur_energy_gap": max(0.0, schur_energy_gap),
      "schur_residual_norm": float(np.linalg.norm(schur_residual)),
      "schur_residual_dual_energy": max(0.0, schur_residual_dual_energy),
      "interface_reference_norm": float(np.linalg.norm(reference_interface)),
      "interface_reference_error_norm": float(np.linalg.norm(interface_error)),
      "full_reference_solution_error_norm": float(
          np.linalg.norm(candidate_solution - reference_solution)),
      "local_private_consistency_error_norm": float(
          math.sqrt(max(0.0, private_consistency_sq))),
      "local_private_consistency_error_max": float(private_consistency_max),
      "private_backsubstitution_count": int(private_backsubstitution_count),
      "duplicate_private_variable_count": int(duplicate_private_count),
      "local_singular_count": int(
          local_singular_count + schur_stats.get("local_singular_count", 0)),
      "unassigned_private_variable_count": int(
          np.count_nonzero(~assigned_private) - len(interface_indices)),
      "final_normal_residual": float(np.linalg.norm(normal_residual)),
  }


def matrix_free_local_interface_schur_dual_certificate(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    candidate_solution: np.ndarray,
    variable_count: int | None = None,
    damping: float = 0.0,
    dual_iterations: int = 100,
    dual_tolerance: float = 1e-10):
  """Estimates Schur energy with a matrix-free dual solve.

  This avoids dense Schur assembly. It computes the exact local-Schur residual
  `r = S x_B - b` by summing local Schur matvec contributions, then estimates
  the dual energy `r^T S^{-1} r` by PCG on the same matrix-free operator.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
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
  candidate_solution = np.asarray(candidate_solution, dtype=float).reshape(-1)
  if len(candidate_solution) != variable_count:
    raise ValueError("candidate_solution dimension does not match variable_count")

  local_data = []
  schur_rhs = np.zeros(len(interface_indices), dtype=float)
  global_gradient = np.zeros(variable_count, dtype=float)
  local_private_variable_count_sum = 0
  local_singular_count = 0
  dense_local_hessian_materialized = False
  uses_sparse_private_factorization = False
  dense_private_fallback_count = 0

  def matrix_vector_product(matrix, vector):
    return np.asarray(matrix @ vector, dtype=float).reshape(-1)

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    gradient = system.get("gradient")
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if hessian is None or gradient is None:
      raise ValueError(f"local system {system_index} is missing hessian/gradient")
    is_sparse_hessian = (
        hasattr(hessian, "tocsr") and not isinstance(hessian, np.ndarray))
    if is_sparse_hessian:
      hessian = hessian.tocsr()
    else:
      hessian = np.asarray(hessian, dtype=float)
      dense_local_hessian_materialized = True
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
    global_gradient += gradient
    local_private_variable_count_sum += int(private_indices.size)

    h_bb = hessian[interface_indices[:, None], interface_indices]
    g_b = gradient[interface_indices]
    if private_indices.size == 0:
      schur_rhs += g_b
      local_data.append({
          "private_indices": private_indices,
          "hessian": hessian,
          "gradient": gradient,
          "h_bb": h_bb,
          "h_pb": None,
          "h_bp": None,
          "private_solver": None,
      })
      continue

    h_pp = hessian[private_indices[:, None], private_indices]
    h_pb = hessian[private_indices[:, None], interface_indices]
    h_bp = hessian[interface_indices[:, None], private_indices]
    g_p = gradient[private_indices]
    if is_sparse_hessian:
      if splu is None or identity is None:
        raise RuntimeError("scipy sparse support is required")
      h_pp = h_pp.tocsc()
      if damping > 0.0:
        h_pp = h_pp + float(damping) * identity(
            h_pp.shape[0], format="csc")
      try:
        private_factor = splu(h_pp)
        uses_sparse_private_factorization = True

        def private_solver(rhs, factor=private_factor):
          return factor.solve(np.asarray(rhs, dtype=float).reshape(-1))
      except RuntimeError:
        local_singular_count += 1
        dense_private_fallback_count += 1
        dense_h_pp = h_pp.toarray()
        dense_local_hessian_materialized = True

        def private_solver(rhs, dense_h_pp=dense_h_pp):
          return np.linalg.pinv(dense_h_pp, rcond=1e-12) @ np.asarray(
              rhs, dtype=float).reshape(-1)
    else:
      if damping > 0.0:
        h_pp = h_pp + float(damping) * np.eye(len(private_indices))
      try:
        h_pp_inv = np.linalg.inv(h_pp)
      except np.linalg.LinAlgError:
        local_singular_count += 1
        h_pp_inv = np.linalg.pinv(h_pp, rcond=1e-12)

      def private_solver(rhs, h_pp_inv=h_pp_inv):
        return h_pp_inv @ np.asarray(rhs, dtype=float).reshape(-1)

    schur_rhs += g_b - matrix_vector_product(h_bp, private_solver(g_p))
    local_data.append({
        "private_indices": private_indices,
        "hessian": hessian,
        "gradient": gradient,
        "h_bb": h_bb,
        "h_pb": h_pb,
        "h_bp": h_bp,
        "private_solver": private_solver,
    })

  interface_matvec_count = 0

  def schur_matvec(vector):
    nonlocal interface_matvec_count
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if len(vector) != len(interface_indices):
      raise ValueError("interface vector dimension mismatch")
    output = np.zeros_like(vector)
    for data in local_data:
      output += matrix_vector_product(data["h_bb"], vector)
      private_solver = data["private_solver"]
      if private_solver is not None:
        private_rhs = matrix_vector_product(data["h_pb"], vector)
        output -= matrix_vector_product(data["h_bp"],
                                        private_solver(private_rhs))
    interface_matvec_count += 1
    return output

  candidate_interface = candidate_solution[interface_indices]
  schur_residual = schur_matvec(candidate_interface) - schur_rhs
  dual_solution, dual_stats = _pcg_solve_from_matvec(
      schur_matvec,
      schur_residual,
      iterations=max(0, int(dual_iterations)),
      tolerance=float(dual_tolerance))
  schur_residual_dual_energy = float(schur_residual @ dual_solution)

  private_consistency_sq = 0.0
  private_consistency_max = 0.0
  private_variable_count = 0
  assigned_private = np.zeros(variable_count, dtype=bool)
  normal_residual = -global_gradient.copy()
  for system_index, data in enumerate(local_data):
    normal_residual += matrix_vector_product(data["hessian"], candidate_solution)
    private_indices = data["private_indices"]
    if private_indices.size == 0:
      continue
    if np.any(assigned_private[private_indices]):
      raise ValueError(
          f"local system {system_index} reuses private variables")
    conditional_rhs = (
        data["gradient"][private_indices] -
        matrix_vector_product(data["h_pb"], candidate_interface))
    conditional_private = data["private_solver"](conditional_rhs)
    private_error = candidate_solution[private_indices] - conditional_private
    private_consistency_sq += float(private_error @ private_error)
    private_consistency_max = max(
        private_consistency_max, float(np.linalg.norm(private_error)))
    private_variable_count += int(private_indices.size)
    assigned_private[private_indices] = True

  return {
      "model": "matrix_free_local_interface_schur_dual_certificate",
      "diagnostic_model": "matrix_free_local_schur_dual_energy_certificate",
      "uses_dense_schur_for_certificate": False,
      "materializes_dense_local_hessian": bool(
          dense_local_hessian_materialized),
      "uses_sparse_private_factorization": bool(
          uses_sparse_private_factorization),
      "dense_private_fallback_count": int(dense_private_fallback_count),
      "local_system_count": int(len(local_systems)),
      "interface_variable_count": int(len(interface_indices)),
      "private_variable_count": int(private_variable_count),
      "variable_count": int(variable_count),
      "local_private_variable_count_sum": int(local_private_variable_count_sum),
      "damping": float(damping),
      "dual_iterations": int(dual_stats.get("iterations", 0)),
      "dual_final_residual": float(dual_stats.get("final_residual", 0.0)),
      "dual_global_reduction_count": int(
          dual_stats.get("global_reduction_count", 0)),
      "interface_matvec_count": int(interface_matvec_count),
      "schur_residual_norm": float(np.linalg.norm(schur_residual)),
      "schur_residual_dual_energy": max(0.0, schur_residual_dual_energy),
      "schur_energy_gap": max(0.0, schur_residual_dual_energy),
      "interface_reference_error_norm": float("nan"),
      "full_reference_solution_error_norm": float("nan"),
      "local_private_consistency_error_norm": float(
          math.sqrt(max(0.0, private_consistency_sq))),
      "local_private_consistency_error_max": float(private_consistency_max),
      "local_singular_count": int(local_singular_count),
      "unassigned_private_variable_count": int(
          np.count_nonzero(~assigned_private) - len(interface_indices)),
      "final_normal_residual": float(np.linalg.norm(normal_residual)),
  }


def estimate_local_interface_schur_pcg_communication(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    interface_matvec_count: int,
    pcg_global_reduction_count: int,
    rhs_reduction_count: int = 1,
    scalar_bytes: int = 8,
    communication_model: str = "separator_owner_star",
    robot_topology_edges: list[tuple[object, object]] | None = None):
  """Estimates explicit separator-owner payload for local-Schur PCG.

  The owner-star model sends each non-owner scalar contribution directly to the
  deterministic owner. The separator-tree model routes those contributions over
  shortest paths in `robot_topology_edges`. PCG scalar dot-product reductions
  are counted separately as one scalar sent by each non-root active robot.
  """
  if communication_model not in {"separator_owner_star", "separator_tree"}:
    raise ValueError(f"unsupported communication_model: {communication_model}")
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    return {
        "communication_model": communication_model,
        "separator_owner_count": 0,
        "separator_shared_scalar_count": 0,
      "separator_owner_pair_payload_bytes": {},
      "separator_owner_interface_payload_bytes": 0,
      "separator_tree_hop_count_per_round": 0,
      "global_scalar_reduction_pair_payload_bytes": {},
      "global_scalar_reduction_hop_count_per_reduction": 0,
      "global_scalar_reduction_root_robot": None,
      "global_scalar_reduction_payload_bytes": 0,
      "estimated_comm_bytes": 0,
      "estimated_comm_mb": 0.0,
      "active_robot_count": 0,
  }
  scalar_bytes = int(scalar_bytes)
  if scalar_bytes <= 0:
    raise ValueError("scalar_bytes must be positive")
  interface_matvec_count = max(0, int(interface_matvec_count))
  pcg_global_reduction_count = max(0, int(pcg_global_reduction_count))
  rhs_reduction_count = max(0, int(rhs_reduction_count))

  participants_by_scalar: list[list[object]] = []
  active_robots: set[object] = set()
  for global_index in interface_indices:
    participants = []
    for system_index, system in enumerate(local_systems):
      robot = system.get("robot", system_index)
      hessian = system.get("hessian")
      gradient = system.get("gradient")
      if hessian is None or gradient is None:
        continue
      gradient = np.asarray(gradient, dtype=float).reshape(-1)
      if int(global_index) >= hessian.shape[0]:
        continue
      if hasattr(hessian, "tocsr") and not isinstance(hessian, np.ndarray):
        hessian_csr = hessian.tocsr()
        hessian_csc = hessian.tocsc()
        row_active = (
            hessian_csr.getrow(int(global_index)).count_nonzero() > 0)
        col_active = (
            hessian_csc.getcol(int(global_index)).count_nonzero() > 0)
      else:
        hessian = np.asarray(hessian, dtype=float)
        row_active = np.any(np.abs(hessian[int(global_index), :]) > 1e-12)
        col_active = np.any(np.abs(hessian[:, int(global_index)]) > 1e-12)
      grad_active = (
          int(global_index) < len(gradient) and
          abs(float(gradient[int(global_index)])) > 1e-12)
      if row_active or col_active or grad_active:
        participants.append(robot)
        active_robots.add(robot)
    participants = sorted(set(participants), key=lambda value: str(value))
    participants_by_scalar.append(participants)

  pair_payload_bytes: dict[str, int] = {}
  owner_set = set()
  shared_scalar_count = 0
  per_scalar_exchange_count = 0
  separator_tree_hop_count_per_round = 0
  exchange_rounds = interface_matvec_count + rhs_reduction_count

  topology_adjacency: dict[object, set[object]] = {}
  for robot in active_robots:
    topology_adjacency.setdefault(robot, set())
  for edge in robot_topology_edges or []:
    if len(edge) != 2:
      raise ValueError("robot_topology_edges must contain robot pairs")
    first, second = edge
    topology_adjacency.setdefault(first, set()).add(second)
    topology_adjacency.setdefault(second, set()).add(first)

  def shortest_path(sender, owner):
    if sender == owner:
      return [sender]
    if communication_model == "separator_owner_star":
      return [sender, owner]
    frontier = [sender]
    parent = {sender: None}
    while frontier:
      current = frontier.pop(0)
      for neighbor in sorted(topology_adjacency.get(current, set()),
                             key=lambda value: str(value)):
        if neighbor in parent:
          continue
        parent[neighbor] = current
        if neighbor == owner:
          path = [owner]
          cursor = owner
          while parent[cursor] is not None:
            cursor = parent[cursor]
            path.append(cursor)
          path.reverse()
          return path
        frontier.append(neighbor)
    raise ValueError(
        f"no robot topology path from {sender} to separator owner {owner}")

  for participants in participants_by_scalar:
    if len(participants) <= 1:
      continue
    owner = participants[0]
    owner_set.add(owner)
    shared_scalar_count += 1
    for sender in participants[1:]:
      path = shortest_path(sender, owner)
      if len(path) <= 1:
        continue
      payload = int(exchange_rounds * scalar_bytes)
      for start, end in zip(path[:-1], path[1:]):
        key = f"{start}->{end}"
        pair_payload_bytes[key] = int(pair_payload_bytes.get(key, 0) + payload)
        separator_tree_hop_count_per_round += 1
      per_scalar_exchange_count += 1

  interface_payload_bytes = int(sum(pair_payload_bytes.values()))
  active_robot_count = len(active_robots)
  global_reduction_pair_payload_bytes: dict[str, int] = {}
  global_reduction_hop_count_per_reduction = 0
  global_reduction_root_robot = None
  if active_robots:
    sorted_active_robots = sorted(active_robots, key=lambda value: str(value))
    global_reduction_root_robot = sorted_active_robots[0]
    payload = int(pcg_global_reduction_count * scalar_bytes)
    for sender in sorted_active_robots[1:]:
      path = shortest_path(sender, global_reduction_root_robot)
      if len(path) <= 1:
        continue
      global_reduction_hop_count_per_reduction += int(len(path) - 1)
      for start, end in zip(path[:-1], path[1:]):
        key = f"{start}->{end}"
        global_reduction_pair_payload_bytes[key] = int(
            global_reduction_pair_payload_bytes.get(key, 0) + payload)
  global_scalar_payload_bytes = int(
      sum(global_reduction_pair_payload_bytes.values()))
  total_bytes = int(interface_payload_bytes + global_scalar_payload_bytes)
  return {
      "communication_model": communication_model,
      "separator_owner_count": int(len(owner_set)),
      "separator_shared_scalar_count": int(shared_scalar_count),
      "separator_owner_pair_payload_bytes": pair_payload_bytes,
      "separator_owner_interface_payload_bytes": int(interface_payload_bytes),
      "separator_owner_scalar_exchange_count_per_round":
          int(per_scalar_exchange_count),
      "separator_tree_hop_count_per_round":
          int(separator_tree_hop_count_per_round),
      "global_scalar_reduction_pair_payload_bytes":
          global_reduction_pair_payload_bytes,
      "global_scalar_reduction_hop_count_per_reduction":
          int(global_reduction_hop_count_per_reduction),
      "global_scalar_reduction_root_robot": global_reduction_root_robot,
      "global_scalar_reduction_payload_bytes": int(global_scalar_payload_bytes),
      "estimated_comm_bytes": int(total_bytes),
      "estimated_comm_mb": _bytes_to_mb(total_bytes),
      "active_robot_count": int(active_robot_count),
  }


def limited_hop_local_interface_schur_matvec_diagnostic(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    vector: np.ndarray,
    robot_topology_edges: list[tuple[object, object]],
    hop_radius: int,
    damping: float = 0.0,
    scalar_bytes: int = 8,
    tolerance: float = 1e-12):
  """Diagnoses interface matvec error under finite-hop contribution gathering.

  For each interface scalar, the deterministic owner is the smallest robot id
  among robots with a nonzero local Schur contribution to that scalar. The
  approximate matvec contains only contributions from robots within
  `hop_radius` graph hops of that owner. This models delayed/limited-range
  reductions without changing the exact local Schur model.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  vector = np.asarray(vector, dtype=float).reshape(-1)
  if len(interface_indices) == 0:
    raise ValueError("interface_indices must be non-empty")
  if len(vector) != len(interface_indices):
    raise ValueError("interface vector dimension mismatch")
  hop_radius = max(0, int(hop_radius))
  scalar_bytes = int(scalar_bytes)
  if scalar_bytes <= 0:
    raise ValueError("scalar_bytes must be positive")

  local_contributions = []
  robots = set()
  for system_index, system in enumerate(local_systems):
    robot = system.get("robot", system_index)
    robots.add(robot)
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
    if int(np.max(interface_indices)) >= hessian.shape[0]:
      raise ValueError("interface index outside local hessian dimension")
    h_bb = hessian[interface_indices[:, None], interface_indices]
    contribution = h_bb @ vector
    if private_indices.size:
      h_pp = hessian[private_indices[:, None], private_indices]
      if damping > 0.0:
        h_pp = h_pp + float(damping) * np.eye(len(private_indices))
      h_pb = hessian[private_indices[:, None], interface_indices]
      h_bp = hessian[interface_indices[:, None], private_indices]
      try:
        private_response = np.linalg.solve(h_pp, h_pb @ vector)
      except np.linalg.LinAlgError:
        private_response = np.linalg.pinv(h_pp, rcond=1e-12) @ (h_pb @ vector)
      contribution = contribution - h_bp @ private_response
    local_contributions.append((robot, np.asarray(contribution, dtype=float)))

  adjacency: dict[object, set[object]] = {robot: set() for robot in robots}
  for first, second in robot_topology_edges:
    adjacency.setdefault(first, set()).add(second)
    adjacency.setdefault(second, set()).add(first)

  def distance(sender, owner):
    if sender == owner:
      return 0
    frontier = [owner]
    distances = {owner: 0}
    while frontier:
      current = frontier.pop(0)
      for neighbor in sorted(adjacency.get(current, set()),
                             key=lambda value: str(value)):
        if neighbor in distances:
          continue
        distances[neighbor] = distances[current] + 1
        if neighbor == sender:
          return distances[neighbor]
        frontier.append(neighbor)
    return math.inf

  exact = np.zeros(len(interface_indices), dtype=float)
  approx = np.zeros(len(interface_indices), dtype=float)
  total_participant_count = 0
  covered_participant_count = 0
  payload_bytes = 0
  owners = set()
  for scalar_offset in range(len(interface_indices)):
    participants = [
        (robot, float(contribution[scalar_offset]))
        for robot, contribution in local_contributions
        if abs(float(contribution[scalar_offset])) > tolerance
    ]
    participants = sorted(participants, key=lambda item: str(item[0]))
    if not participants:
      continue
    owner = participants[0][0]
    owners.add(owner)
    for robot, value in participants:
      exact[scalar_offset] += value
      total_participant_count += 1
      hop_distance = distance(robot, owner)
      if hop_distance <= hop_radius:
        approx[scalar_offset] += value
        covered_participant_count += 1
        if robot != owner:
          payload_bytes += int(hop_distance * scalar_bytes)

  error = approx - exact
  return {
      "model": "limited_hop_local_interface_schur_matvec",
      "hop_radius": int(hop_radius),
      "exact_matvec": [float(value) for value in exact],
      "approx_matvec": [float(value) for value in approx],
      "matvec_error": [float(value) for value in error],
      "matvec_error_norm": float(np.linalg.norm(error)),
      "covered_participant_count": int(covered_participant_count),
      "total_participant_count": int(total_participant_count),
      "coverage_fraction": (
          1.0 if total_participant_count == 0
          else float(covered_participant_count / total_participant_count)),
      "limited_hop_payload_bytes": int(payload_bytes),
      "limited_hop_payload_mb": _bytes_to_mb(payload_bytes),
      "separator_owner_count": int(len(owners)),
  }


def solve_limited_hop_local_interface_schur_pcg(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    variable_count: int | None = None,
    iterations: int = 100,
    tolerance: float = 1e-10,
    damping: float = 0.0,
    robot_topology_edges: list[tuple[object, object]] | None = None,
    hop_radius: int = 1):
  """Solves the interface system with a finite-hop Schur matvec diagnostic.

  The right-hand side is kept exact on purpose: this isolates the solver error
  caused only by incomplete finite-hop matvec reductions. Dense Schur assembly
  is used only after the approximate solve to report reference residuals and
  solution error, not to form the PCG search directions.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
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
  if robot_topology_edges is None:
    robot_topology_edges = []

  exact_schur, exact_rhs, schur_stats = sum_local_interface_schur_contributions(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping)

  limited_hop_matvec_count = 0
  max_limited_hop_matvec_error_norm = 0.0
  limited_hop_payload_bytes = 0
  covered_participant_count = 0
  total_participant_count = 0
  min_coverage_fraction = 1.0

  def finite_hop_matvec(vector):
    nonlocal limited_hop_matvec_count
    nonlocal max_limited_hop_matvec_error_norm
    nonlocal limited_hop_payload_bytes
    nonlocal covered_participant_count
    nonlocal total_participant_count
    nonlocal min_coverage_fraction
    diagnostic = limited_hop_local_interface_schur_matvec_diagnostic(
        local_systems=local_systems,
        interface_indices=interface_indices,
        vector=vector,
        robot_topology_edges=robot_topology_edges,
        hop_radius=hop_radius,
        damping=damping,
        scalar_bytes=8)
    limited_hop_matvec_count += 1
    max_limited_hop_matvec_error_norm = max(
        max_limited_hop_matvec_error_norm,
        float(diagnostic["matvec_error_norm"]))
    limited_hop_payload_bytes += int(diagnostic["limited_hop_payload_bytes"])
    covered_participant_count += int(diagnostic["covered_participant_count"])
    total_participant_count += int(diagnostic["total_participant_count"])
    min_coverage_fraction = min(
        min_coverage_fraction, float(diagnostic["coverage_fraction"]))
    return np.asarray(diagnostic["approx_matvec"], dtype=float)

  interface_solution, pcg_stats = _pcg_solve_from_matvec(
      finite_hop_matvec,
      exact_rhs,
      iterations=max(0, int(iterations)),
      tolerance=float(tolerance))

  solution = np.zeros(variable_count, dtype=float)
  assigned = np.zeros(variable_count, dtype=bool)
  solution[interface_indices] = interface_solution
  assigned[interface_indices] = True
  global_hessian = np.zeros((variable_count, variable_count), dtype=float)
  global_gradient = np.zeros(variable_count, dtype=float)
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
      raise ValueError(
          f"local system {system_index} hessian has wrong shape")
    if gradient.shape[0] != variable_count:
      raise ValueError(
          f"local system {system_index} gradient has wrong dimension")
    global_hessian += hessian
    global_gradient += gradient
    if private_indices.size == 0:
      continue
    if np.any(assigned[private_indices]):
      raise ValueError(
          f"local system {system_index} reuses private variables")
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

  reference_solution, _ = solve_local_interface_schur_reference(
      local_systems=local_systems,
      interface_indices=interface_indices,
      variable_count=variable_count,
      damping=damping)
  exact_schur_residual = exact_schur @ interface_solution - exact_rhs
  normal_residual = global_hessian @ solution - global_gradient
  gap_certificate = local_interface_schur_gap_certificate(
      local_systems=local_systems,
      interface_indices=interface_indices,
      candidate_solution=solution,
      variable_count=variable_count,
      damping=damping)
  coverage_fraction = (
      1.0 if total_participant_count == 0
      else float(covered_participant_count / total_participant_count))
  return solution, {
      "model": "limited_hop_local_interface_schur_pcg",
      "materializes_dense_schur": False,
      "materializes_dense_schur_for_diagnostics": True,
      "uses_exact_rhs": True,
      "local_system_count": int(len(local_systems)),
      "interface_variable_count": int(len(interface_indices)),
      "variable_count": int(variable_count),
      "damping": float(damping),
      "hop_radius": int(max(0, int(hop_radius))),
      "iterations": int(pcg_stats.get("iterations", 0)),
      "pcg_final_residual": float(pcg_stats.get("final_residual", 0.0)),
      "pcg_global_reduction_count": int(
          pcg_stats.get("global_reduction_count", 0)),
      "limited_hop_matvec_count": int(limited_hop_matvec_count),
      "max_limited_hop_matvec_error_norm": float(
          max_limited_hop_matvec_error_norm),
      "final_exact_schur_residual": float(np.linalg.norm(exact_schur_residual)),
      "final_normal_residual": float(np.linalg.norm(normal_residual)),
      "reference_solution_error_norm": float(
          np.linalg.norm(solution - reference_solution)),
      "schur_energy_gap": float(gap_certificate["schur_energy_gap"]),
      "schur_residual_norm": float(gap_certificate["schur_residual_norm"]),
      "schur_residual_dual_energy": float(
          gap_certificate["schur_residual_dual_energy"]),
      "interface_reference_error_norm": float(
          gap_certificate["interface_reference_error_norm"]),
      "full_reference_solution_error_norm": float(
          gap_certificate["full_reference_solution_error_norm"]),
      "local_private_consistency_error_norm": float(
          gap_certificate["local_private_consistency_error_norm"]),
      "interface_solution_norm": float(np.linalg.norm(interface_solution)),
      "private_backsubstitution_count": int(private_backsubstitution_count),
      "unassigned_variable_count": int(np.count_nonzero(~assigned)),
      "local_singular_count": int(
          local_singular_count + schur_stats.get("local_singular_count", 0)),
      "covered_participant_count": int(covered_participant_count),
      "total_participant_count": int(total_participant_count),
      "coverage_fraction": float(coverage_fraction),
      "min_coverage_fraction": float(min_coverage_fraction),
      "limited_hop_payload_bytes": int(limited_hop_payload_bytes),
      "limited_hop_payload_mb": _bytes_to_mb(limited_hop_payload_bytes),
      "communication_model": "limited_hop",
  }


def solve_local_interface_schur_pcg(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    variable_count: int | None = None,
    iterations: int = 100,
    tolerance: float = 1e-10,
    damping: float = 0.0,
    schur_preconditioner: str = "none",
    communication_model: str = "separator_owner_star",
    robot_topology_edges: list[tuple[object, object]] | None = None):
  """Solves the local-Schur interface system by matrix-free PCG.

  Unlike `solve_local_interface_schur_reference`, this function never
  materializes the dense interface Schur matrix. Each matvec asks every local
  system for its contribution `S_r v`, then sums those interface contributions.
  This emulates the distributed interface-reduction path required by a
  deployable KKT-consistent DCI initializer.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
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
  if schur_preconditioner not in {"none", "diagonal"}:
    raise ValueError(f"unsupported schur_preconditioner: {schur_preconditioner}")

  local_data = []
  schur_rhs = np.zeros(len(interface_indices), dtype=float)
  global_gradient = np.zeros(variable_count, dtype=float)
  local_private_variable_count_sum = 0
  local_singular_count = 0
  local_condition_max = 0.0
  dense_local_hessian_materialized = False
  uses_sparse_private_factorization = False
  dense_private_fallback_count = 0

  def matrix_vector_product(matrix, vector):
    return np.asarray(matrix @ vector, dtype=float).reshape(-1)

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    gradient = system.get("gradient")
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if hessian is None or gradient is None:
      raise ValueError(f"local system {system_index} is missing hessian/gradient")
    is_sparse_hessian = (
        hasattr(hessian, "tocsr") and not isinstance(hessian, np.ndarray))
    if is_sparse_hessian:
      hessian = hessian.tocsr()
    else:
      hessian = np.asarray(hessian, dtype=float)
      dense_local_hessian_materialized = True
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
    global_gradient += gradient
    local_private_variable_count_sum += int(private_indices.size)

    h_bb = hessian[interface_indices[:, None], interface_indices]
    g_b = gradient[interface_indices]
    if private_indices.size == 0:
      schur_rhs += g_b
      local_data.append({
          "private_indices": private_indices,
          "hessian": hessian,
          "gradient": gradient,
          "h_bb": h_bb,
          "h_pb": None,
          "h_bp": None,
          "private_solver": None,
      })
      continue

    h_pp = hessian[private_indices[:, None], private_indices]
    h_pb = hessian[private_indices[:, None], interface_indices]
    h_bp = hessian[interface_indices[:, None], private_indices]
    g_p = gradient[private_indices]
    if is_sparse_hessian:
      if splu is None or identity is None:
        raise RuntimeError("scipy sparse support is required")
      h_pp = h_pp.tocsc()
      if damping > 0.0:
        h_pp = h_pp + float(damping) * identity(
            h_pp.shape[0], format="csc")
      try:
        private_factor = splu(h_pp)
        uses_sparse_private_factorization = True

        def private_solver(rhs, factor=private_factor):
          return factor.solve(np.asarray(rhs, dtype=float).reshape(-1))
      except RuntimeError:
        local_singular_count += 1
        dense_private_fallback_count += 1
        dense_h_pp = h_pp.toarray()
        dense_local_hessian_materialized = True

        def private_solver(rhs, dense_h_pp=dense_h_pp):
          return np.linalg.pinv(dense_h_pp, rcond=1e-12) @ np.asarray(
              rhs, dtype=float).reshape(-1)
    else:
      if damping > 0.0:
        h_pp = h_pp + float(damping) * np.eye(len(private_indices))
      condition = float(np.linalg.cond(h_pp))
      if np.isfinite(condition):
        local_condition_max = max(local_condition_max, condition)
      try:
        h_pp_inv = np.linalg.inv(h_pp)
      except np.linalg.LinAlgError:
        local_singular_count += 1
        h_pp_inv = np.linalg.pinv(h_pp, rcond=1e-12)

      def private_solver(rhs, h_pp_inv=h_pp_inv):
        return h_pp_inv @ np.asarray(rhs, dtype=float).reshape(-1)

    schur_rhs += g_b - matrix_vector_product(h_bp, private_solver(g_p))
    local_data.append({
        "private_indices": private_indices,
        "hessian": hessian,
        "gradient": gradient,
        "h_bb": h_bb,
        "h_pb": h_pb,
        "h_bp": h_bp,
        "private_solver": private_solver,
    })

  interface_matvec_count = 0

  def schur_matvec(vector):
    nonlocal interface_matvec_count
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if len(vector) != len(interface_indices):
      raise ValueError("interface vector dimension mismatch")
    output = np.zeros_like(vector)
    for data in local_data:
      h_bb = data["h_bb"]
      output += matrix_vector_product(h_bb, vector)
      private_solver = data["private_solver"]
      if private_solver is not None:
        private_rhs = matrix_vector_product(data["h_pb"], vector)
        output -= matrix_vector_product(data["h_bp"],
                                        private_solver(private_rhs))
    interface_matvec_count += 1
    return output

  preconditioner_private_solve_count = 0
  schur_preconditioner_diagonal_min = 0.0
  schur_preconditioner_diagonal_max = 0.0

  def build_diagonal_preconditioner():
    nonlocal preconditioner_private_solve_count
    diagonal = np.zeros(len(interface_indices), dtype=float)
    for data in local_data:
      h_bb = data["h_bb"]
      h_bb_diag = np.asarray(h_bb.diagonal(), dtype=float).reshape(-1)
      diagonal += h_bb_diag
      private_solver = data["private_solver"]
      if private_solver is None:
        continue
      for col in range(len(interface_indices)):
        basis = np.zeros(len(interface_indices), dtype=float)
        basis[col] = 1.0
        private_rhs = matrix_vector_product(data["h_pb"], basis)
        response = private_solver(private_rhs)
        preconditioner_private_solve_count += 1
        diagonal[col] -= matrix_vector_product(
            data["h_bp"][col:col + 1, :], response)[0]
    scale = np.maximum(np.abs(diagonal), 1e-12)
    return scale

  preconditioner = None
  if schur_preconditioner == "diagonal":
    preconditioner_diagonal = build_diagonal_preconditioner()
    schur_preconditioner_diagonal_min = float(
        np.min(preconditioner_diagonal)) if preconditioner_diagonal.size else 0.0
    schur_preconditioner_diagonal_max = float(
        np.max(preconditioner_diagonal)) if preconditioner_diagonal.size else 0.0

    def preconditioner(residual, diagonal=preconditioner_diagonal):
      return np.asarray(residual, dtype=float).reshape(-1) / diagonal

  interface_solution, pcg_stats = _pcg_solve_from_matvec(
      schur_matvec,
      schur_rhs,
      iterations=max(0, int(iterations)),
      tolerance=float(tolerance),
      preconditioner=preconditioner)
  final_schur_residual = float(
      np.linalg.norm(schur_matvec(interface_solution) - schur_rhs))

  solution = np.zeros(variable_count, dtype=float)
  assigned = np.zeros(variable_count, dtype=bool)
  solution[interface_indices] = interface_solution
  assigned[interface_indices] = True
  private_backsubstitution_count = 0
  for system_index, data in enumerate(local_data):
    private_indices = data["private_indices"]
    if private_indices.size == 0:
      continue
    if np.any(assigned[private_indices]):
      raise ValueError(
          f"local system {system_index} reuses private variables")
    private_rhs = (
        data["gradient"][private_indices] -
        matrix_vector_product(data["h_pb"], interface_solution))
    solution[private_indices] = data["private_solver"](private_rhs)
    assigned[private_indices] = True
    private_backsubstitution_count += 1

  normal_residual = -global_gradient.copy()
  for data in local_data:
    normal_residual += matrix_vector_product(data["hessian"], solution)
  solve_reductions = int(pcg_stats.get("global_reduction_count", 0))
  rhs_reduction_count = 1
  topology_reduction_count = (
      rhs_reduction_count + interface_matvec_count + solve_reductions)
  communication_stats = estimate_local_interface_schur_pcg_communication(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_matvec_count=interface_matvec_count,
      pcg_global_reduction_count=solve_reductions,
      rhs_reduction_count=rhs_reduction_count,
      scalar_bytes=8,
      communication_model=communication_model,
      robot_topology_edges=robot_topology_edges)
  return solution, {
      "model": "local_interface_schur_pcg",
      "materializes_dense_schur": False,
      "materializes_dense_local_hessian": bool(
          dense_local_hessian_materialized),
      "uses_sparse_private_factorization": bool(
          uses_sparse_private_factorization),
      "dense_private_fallback_count": int(dense_private_fallback_count),
      "local_system_count": int(len(local_systems)),
      "interface_variable_count": int(len(interface_indices)),
      "variable_count": int(variable_count),
      "local_private_variable_count_sum": int(local_private_variable_count_sum),
      "local_singular_count": int(local_singular_count),
      "local_hpp_condition_max": float(local_condition_max),
      "damping": float(damping),
      "schur_preconditioner": schur_preconditioner,
      "schur_preconditioner_private_solve_count": int(
          preconditioner_private_solve_count),
      "schur_preconditioner_diagonal_min": float(
          schur_preconditioner_diagonal_min),
      "schur_preconditioner_diagonal_max": float(
          schur_preconditioner_diagonal_max),
      "iterations": int(pcg_stats.get("iterations", 0)),
      "final_schur_residual": final_schur_residual,
      "pcg_final_residual": float(pcg_stats.get("final_residual", 0.0)),
      "pcg_global_reduction_count": solve_reductions,
      "rhs_reduction_count": int(rhs_reduction_count),
      "interface_matvec_count": int(interface_matvec_count),
      "topology_reduction_count": int(topology_reduction_count),
      "interface_solution_norm": float(np.linalg.norm(interface_solution)),
      "private_backsubstitution_count": int(private_backsubstitution_count),
      "unassigned_variable_count": int(np.count_nonzero(~assigned)),
      "final_normal_residual": float(np.linalg.norm(normal_residual)),
      **communication_stats,
  }


def solve_local_interface_schur_fixed_step(
    local_systems: list[dict],
    interface_indices: np.ndarray,
    variable_count: int | None = None,
    iterations: int = 100,
    tolerance: float = 1e-10,
    damping: float = 0.0,
    schur_preconditioner: str = "diagonal",
    relaxation: float = 1.0,
    coarse_basis: np.ndarray | None = None,
    coarse_regularization: float = 0.0,
    adaptive_coarse_basis: str = "none",
    adaptive_coarse_rank: int = 0,
    adaptive_coarse_probe_count: int = 0,
    adaptive_coarse_seed: int = 0,
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    dirichlet_diagonal_floor: float = 0.25,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
    chebyshev_certificate_block_dim: int = 1,
    chebyshev_certificate_iterations: int = 128,
    chebyshev_certificate_vector_mode: str = "cg_resolvent",
    chebyshev_certificate_resolvent_shift: float = 0.0,
    chebyshev_certificate_max_payload_mb: float | None = None,
    chebyshev_certificate_preflight_hard_cap_mb: float | None = None,
    communication_model: str = "separator_owner_star",
    robot_topology_edges: list[tuple[object, object]] | None = None):
  """Solves the local-Schur interface system with no global dot reductions.

  This is a deployment diagnostic for reduction-light DCI. It uses the same
  local Schur matvec and private recovery target as PCG, but replaces CG's
  dot-product driven alpha/beta updates with a fixed-step preconditioned
  Richardson iteration:

      x_{k+1} = x_k + relaxation * M^{-1}(b - S x_k).

  The residual norm is computed locally in this diagnostic report, but no
  scalar dot-product reduction is charged to the communication model. The
  optional Chebyshev mode keeps the same matvec communication pattern but
  replaces the Richardson polynomial with a fixed spectral-interval polynomial.
  """
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  interface_indices = np.asarray(
      sorted(set(int(index) for index in interface_indices)), dtype=int)
  if interface_indices.size == 0:
    raise ValueError("interface_indices must be non-empty")
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
  if schur_preconditioner not in {
      "none",
      "diagonal",
      "boundary_diagonal",
      "dirichlet_clipped_diagonal",
  }:
    raise ValueError(f"unsupported schur_preconditioner: {schur_preconditioner}")
  dirichlet_diagonal_floor = float(dirichlet_diagonal_floor)
  if (not np.isfinite(dirichlet_diagonal_floor) or
      dirichlet_diagonal_floor <= 0.0 or
      dirichlet_diagonal_floor > 1.0):
    raise ValueError("dirichlet_diagonal_floor must be in (0, 1]")
  relaxation = float(relaxation)
  if not np.isfinite(relaxation) or relaxation <= 0.0:
    raise ValueError("relaxation must be a positive finite value")
  coarse_regularization = float(coarse_regularization)
  if not np.isfinite(coarse_regularization) or coarse_regularization < 0.0:
    raise ValueError("coarse_regularization must be nonnegative")
  adaptive_coarse_basis = str(adaptive_coarse_basis)
  if adaptive_coarse_basis not in {
      "none", "residual_krylov", "deterministic_ritz"}:
    raise ValueError(f"unsupported adaptive_coarse_basis: {adaptive_coarse_basis}")
  adaptive_coarse_rank = max(0, int(adaptive_coarse_rank))
  adaptive_coarse_probe_count = max(0, int(adaptive_coarse_probe_count))
  adaptive_coarse_seed = int(adaptive_coarse_seed)
  fixed_step_acceleration = str(fixed_step_acceleration)
  if fixed_step_acceleration not in {
      "none",
      "chebyshev",
      "chebyshev_auto_gershgorin",
      "chebyshev_graph_normalized",
      "chebyshev_ritz_probe",
      "chebyshev_block_gershgorin_certificate",
      "chebyshev_block_gershgorin_budgeted_certificate",
  }:
    raise ValueError(
        f"unsupported fixed_step_acceleration: {fixed_step_acceleration}")
  chebyshev_ritz_probe_iterations = max(1, int(
      chebyshev_ritz_probe_iterations))
  chebyshev_ritz_seed = int(chebyshev_ritz_seed)
  chebyshev_ritz_safety_factor = float(chebyshev_ritz_safety_factor)
  if (not np.isfinite(chebyshev_ritz_safety_factor) or
      chebyshev_ritz_safety_factor < 1.0):
    raise ValueError("chebyshev_ritz_safety_factor must be finite and >= 1")
  chebyshev_certificate_block_dim = int(chebyshev_certificate_block_dim)
  chebyshev_certificate_iterations = max(
      0, int(chebyshev_certificate_iterations))
  chebyshev_certificate_vector_mode = str(chebyshev_certificate_vector_mode)
  chebyshev_certificate_resolvent_shift = float(
      chebyshev_certificate_resolvent_shift)
  if chebyshev_certificate_max_payload_mb is None:
    chebyshev_certificate_max_payload_mb_value = float("inf")
  else:
    chebyshev_certificate_max_payload_mb_value = float(
        chebyshev_certificate_max_payload_mb)
    if (not np.isfinite(chebyshev_certificate_max_payload_mb_value) or
        chebyshev_certificate_max_payload_mb_value < 0.0):
      raise ValueError(
          "chebyshev_certificate_max_payload_mb must be nonnegative")
  if chebyshev_certificate_preflight_hard_cap_mb is None:
    chebyshev_certificate_preflight_hard_cap_mb_value = (
        chebyshev_certificate_max_payload_mb_value)
  else:
    chebyshev_certificate_preflight_hard_cap_mb_value = float(
        chebyshev_certificate_preflight_hard_cap_mb)
    if (not np.isfinite(chebyshev_certificate_preflight_hard_cap_mb_value) or
        chebyshev_certificate_preflight_hard_cap_mb_value < 0.0):
      raise ValueError(
          "chebyshev_certificate_preflight_hard_cap_mb must be nonnegative")
  if chebyshev_certificate_block_dim <= 0:
    raise ValueError("chebyshev_certificate_block_dim must be positive")
  chebyshev_lambda_min_value = 0.0
  chebyshev_lambda_max_value = 0.0
  chebyshev_bound_source = "none"
  chebyshev_bound_setup_matvec_count = 0
  chebyshev_bound_global_reduction_count = 0
  chebyshev_gershgorin_lambda_min = 0.0
  chebyshev_gershgorin_lambda_max = 0.0
  chebyshev_ritz_probe_basis_rank = 0
  chebyshev_ritz_probe_lambda_max = 0.0
  chebyshev_ritz_probe_safe_lambda_max = 0.0
  chebyshev_block_certificate: dict = {}
  chebyshev_certificate_preflight: dict = {}
  chebyshev_certificate_policy_decision = "not_applicable"
  if fixed_step_acceleration == "chebyshev":
    if chebyshev_lambda_min is None or chebyshev_lambda_max is None:
      raise ValueError("chebyshev spectral bounds must be provided")
    chebyshev_lambda_min_value = float(chebyshev_lambda_min)
    chebyshev_lambda_max_value = float(chebyshev_lambda_max)
    chebyshev_bound_source = "manual"
    if (not np.isfinite(chebyshev_lambda_min_value) or
        not np.isfinite(chebyshev_lambda_max_value) or
        chebyshev_lambda_min_value <= 0.0 or
        chebyshev_lambda_max_value <= chebyshev_lambda_min_value):
      raise ValueError(
          "chebyshev spectral bounds must satisfy "
          "0 < chebyshev_lambda_min < chebyshev_lambda_max")
  elif fixed_step_acceleration == "chebyshev_graph_normalized":
    chebyshev_lambda_min_value = (
        1e-8 if chebyshev_lambda_min is None
        else float(chebyshev_lambda_min))
    chebyshev_lambda_max_value = (
        (2.0 / dirichlet_diagonal_floor)
        if (chebyshev_lambda_max is None and
            schur_preconditioner == "dirichlet_clipped_diagonal")
        else 2.0 if chebyshev_lambda_max is None
        else float(chebyshev_lambda_max))
    chebyshev_bound_source = (
        "dirichlet_clipped_graph_normalized"
        if schur_preconditioner == "dirichlet_clipped_diagonal"
        else "graph_normalized_heuristic")
    if (not np.isfinite(chebyshev_lambda_min_value) or
        not np.isfinite(chebyshev_lambda_max_value) or
        chebyshev_lambda_min_value <= 0.0 or
        chebyshev_lambda_max_value <= chebyshev_lambda_min_value):
      raise ValueError(
          "chebyshev graph-normalized heuristic bounds must satisfy "
          "0 < chebyshev_lambda_min < chebyshev_lambda_max")
  elif fixed_step_acceleration == "chebyshev_ritz_probe":
    chebyshev_lambda_min_value = (
        1e-8 if chebyshev_lambda_min is None
        else float(chebyshev_lambda_min))
    if (not np.isfinite(chebyshev_lambda_min_value) or
        chebyshev_lambda_min_value <= 0.0):
      raise ValueError("chebyshev_lambda_min must be positive and finite")
  elif fixed_step_acceleration in {
      "chebyshev_block_gershgorin_certificate",
      "chebyshev_block_gershgorin_budgeted_certificate",
  }:
    chebyshev_lambda_min_value = (
        1e-8 if chebyshev_lambda_min is None
        else float(chebyshev_lambda_min))
    if (not np.isfinite(chebyshev_lambda_min_value) or
        chebyshev_lambda_min_value <= 0.0):
      raise ValueError("chebyshev_lambda_min must be positive and finite")
  chebyshev_safety_monitor = str(chebyshev_safety_monitor)
  if chebyshev_safety_monitor not in {
      "none",
      "residual_growth",
      "local_residual_envelope",
  }:
    raise ValueError(
        f"unsupported chebyshev_safety_monitor: {chebyshev_safety_monitor}")
  chebyshev_safety_growth_factor = float(chebyshev_safety_growth_factor)
  if (not np.isfinite(chebyshev_safety_growth_factor) or
      chebyshev_safety_growth_factor <= 1.0):
    raise ValueError("chebyshev_safety_growth_factor must be finite and > 1")

  local_data = []
  schur_rhs = np.zeros(len(interface_indices), dtype=float)
  global_gradient = np.zeros(variable_count, dtype=float)
  local_private_variable_count_sum = 0
  local_singular_count = 0
  local_condition_max = 0.0
  dense_local_hessian_materialized = False
  uses_sparse_private_factorization = False
  dense_private_fallback_count = 0

  def matrix_vector_product(matrix, vector):
    return np.asarray(matrix @ vector, dtype=float).reshape(-1)

  for system_index, system in enumerate(local_systems):
    hessian = system.get("hessian")
    gradient = system.get("gradient")
    private_indices = np.asarray(
        system.get("private_indices", []), dtype=int).reshape(-1)
    private_indices = np.asarray(
        sorted(set(int(index) for index in private_indices)), dtype=int)
    if hessian is None or gradient is None:
      raise ValueError(f"local system {system_index} is missing hessian/gradient")
    is_sparse_hessian = (
        hasattr(hessian, "tocsr") and not isinstance(hessian, np.ndarray))
    if is_sparse_hessian:
      hessian = hessian.tocsr()
    else:
      hessian = np.asarray(hessian, dtype=float)
      dense_local_hessian_materialized = True
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
    global_gradient += gradient
    local_private_variable_count_sum += int(private_indices.size)

    h_bb = hessian[interface_indices[:, None], interface_indices]
    g_b = gradient[interface_indices]
    if private_indices.size == 0:
      local_schur_rhs = g_b.copy()
      schur_rhs += g_b
      local_data.append({
          "private_indices": private_indices,
          "hessian": hessian,
          "gradient": gradient,
          "h_bb": h_bb,
          "schur_rhs": local_schur_rhs,
          "h_pb": None,
          "h_bp": None,
          "private_solver": None,
      })
      continue

    h_pp = hessian[private_indices[:, None], private_indices]
    h_pb = hessian[private_indices[:, None], interface_indices]
    h_bp = hessian[interface_indices[:, None], private_indices]
    g_p = gradient[private_indices]
    if is_sparse_hessian:
      if splu is None or identity is None:
        raise RuntimeError("scipy sparse support is required")
      h_pp = h_pp.tocsc()
      if damping > 0.0:
        h_pp = h_pp + float(damping) * identity(
            h_pp.shape[0], format="csc")
      try:
        private_factor = splu(h_pp)
        uses_sparse_private_factorization = True

        def private_solver(rhs, factor=private_factor):
          return factor.solve(np.asarray(rhs, dtype=float).reshape(-1))
      except RuntimeError:
        local_singular_count += 1
        dense_private_fallback_count += 1
        dense_h_pp = h_pp.toarray()
        dense_local_hessian_materialized = True

        def private_solver(rhs, dense_h_pp=dense_h_pp):
          return np.linalg.pinv(dense_h_pp, rcond=1e-12) @ np.asarray(
              rhs, dtype=float).reshape(-1)
    else:
      if damping > 0.0:
        h_pp = h_pp + float(damping) * np.eye(len(private_indices))
      condition = float(np.linalg.cond(h_pp))
      if np.isfinite(condition):
        local_condition_max = max(local_condition_max, condition)
      try:
        h_pp_inv = np.linalg.inv(h_pp)
      except np.linalg.LinAlgError:
        local_singular_count += 1
        h_pp_inv = np.linalg.pinv(h_pp, rcond=1e-12)

      def private_solver(rhs, h_pp_inv=h_pp_inv):
        return h_pp_inv @ np.asarray(rhs, dtype=float).reshape(-1)

    local_schur_rhs = g_b - matrix_vector_product(h_bp, private_solver(g_p))
    schur_rhs += local_schur_rhs
    local_data.append({
        "private_indices": private_indices,
        "hessian": hessian,
        "gradient": gradient,
        "h_bb": h_bb,
        "schur_rhs": local_schur_rhs,
        "h_pb": h_pb,
        "h_bp": h_bp,
        "private_solver": private_solver,
    })

  interface_matvec_count = 0

  def schur_matvec(vector):
    nonlocal interface_matvec_count
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if len(vector) != len(interface_indices):
      raise ValueError("interface vector dimension mismatch")
    output = np.zeros_like(vector)
    for data in local_data:
      h_bb = data["h_bb"]
      output += matrix_vector_product(h_bb, vector)
      private_solver = data["private_solver"]
      if private_solver is not None:
        private_rhs = matrix_vector_product(data["h_pb"], vector)
        output -= matrix_vector_product(data["h_bp"],
                                        private_solver(private_rhs))
    interface_matvec_count += 1
    return output

  chebyshev_safety_envelope_evaluation_count = 0

  def local_schur_contribution_matvec(data, vector):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    local_output = matrix_vector_product(data["h_bb"], vector)
    private_solver = data["private_solver"]
    if private_solver is not None:
      private_rhs = matrix_vector_product(data["h_pb"], vector)
      local_output -= matrix_vector_product(
          data["h_bp"], private_solver(private_rhs))
    return local_output

  def local_residual_envelope_metric(vector):
    nonlocal chebyshev_safety_envelope_evaluation_count
    vector = np.asarray(vector, dtype=float).reshape(-1)
    envelope = 0.0
    for data in local_data:
      local_residual = (
          data["schur_rhs"] - local_schur_contribution_matvec(data, vector))
      envelope += float(np.linalg.norm(local_residual))
    chebyshev_safety_envelope_evaluation_count += 1
    return float(envelope)

  preconditioner_private_solve_count = 0
  schur_preconditioner_diagonal_min = 0.0
  schur_preconditioner_diagonal_max = 0.0
  dirichlet_clipped_diagonal_clipped_count = 0
  dirichlet_clipped_diagonal_ratio_min = 0.0
  dirichlet_clipped_diagonal_ratio_max = 0.0

  def build_diagonal_preconditioner():
    nonlocal preconditioner_private_solve_count
    nonlocal dirichlet_clipped_diagonal_clipped_count
    nonlocal dirichlet_clipped_diagonal_ratio_min
    nonlocal dirichlet_clipped_diagonal_ratio_max
    diagonal = np.zeros(len(interface_indices), dtype=float)
    boundary_diagonal = np.zeros(len(interface_indices), dtype=float)
    for data in local_data:
      h_bb = data["h_bb"]
      h_bb_diag = np.asarray(h_bb.diagonal(), dtype=float).reshape(-1)
      diagonal += h_bb_diag
      boundary_diagonal += h_bb_diag
      if schur_preconditioner == "boundary_diagonal":
        continue
      private_solver = data["private_solver"]
      if private_solver is None:
        continue
      for col in range(len(interface_indices)):
        basis = np.zeros(len(interface_indices), dtype=float)
        basis[col] = 1.0
        private_rhs = matrix_vector_product(data["h_pb"], basis)
        response = private_solver(private_rhs)
        preconditioner_private_solve_count += 1
        diagonal[col] -= matrix_vector_product(
            data["h_bp"][col:col + 1, :], response)[0]
    if schur_preconditioner == "dirichlet_clipped_diagonal":
      boundary_diagonal = np.maximum(np.abs(boundary_diagonal), 1e-12)
      schur_diagonal = np.maximum(np.abs(diagonal), 1e-12)
      floor_diagonal = float(dirichlet_diagonal_floor) * boundary_diagonal
      clipped = schur_diagonal < floor_diagonal
      dirichlet_clipped_diagonal_clipped_count = int(np.count_nonzero(clipped))
      ratios = schur_diagonal / boundary_diagonal
      dirichlet_clipped_diagonal_ratio_min = (
          float(np.min(ratios)) if ratios.size else 0.0)
      dirichlet_clipped_diagonal_ratio_max = (
          float(np.max(ratios)) if ratios.size else 0.0)
      diagonal = np.maximum(schur_diagonal, floor_diagonal)
    return np.maximum(np.abs(diagonal), 1e-12)

  def preconditioner(residual):
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if schur_preconditioner in {
        "diagonal",
        "boundary_diagonal",
        "dirichlet_clipped_diagonal",
    }:
      return residual / preconditioner_diagonal
    return residual.copy()

  preconditioner_diagonal = np.ones(len(interface_indices), dtype=float)
  if schur_preconditioner in {
      "diagonal",
      "boundary_diagonal",
      "dirichlet_clipped_diagonal",
  }:
    preconditioner_diagonal = build_diagonal_preconditioner()
    schur_preconditioner_diagonal_min = float(
        np.min(preconditioner_diagonal)) if preconditioner_diagonal.size else 0.0
    schur_preconditioner_diagonal_max = float(
        np.max(preconditioner_diagonal)) if preconditioner_diagonal.size else 0.0

  if fixed_step_acceleration == "chebyshev_auto_gershgorin":
    scale = (
        preconditioner_diagonal
        if schur_preconditioner == "diagonal"
        else np.ones(len(interface_indices), dtype=float))
    scale = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-12)
    sqrt_scale = np.sqrt(scale)
    row_abs_sums = np.zeros(len(interface_indices), dtype=float)
    normalized_diagonal = np.zeros(len(interface_indices), dtype=float)
    setup_start_matvec_count = int(interface_matvec_count)
    for column in range(len(interface_indices)):
      basis = np.zeros(len(interface_indices), dtype=float)
      basis[column] = 1.0
      schur_column = schur_matvec(basis)
      normalized_column = schur_column / (sqrt_scale * sqrt_scale[column])
      row_abs_sums += np.abs(normalized_column)
      normalized_diagonal[column] = normalized_column[column]
    chebyshev_bound_setup_matvec_count = int(
        interface_matvec_count - setup_start_matvec_count)
    if row_abs_sums.size:
      chebyshev_gershgorin_lambda_max = float(np.max(row_abs_sums))
      offdiag_abs = row_abs_sums - np.abs(normalized_diagonal)
      chebyshev_gershgorin_lambda_min = float(
          max(0.0, np.min(normalized_diagonal - offdiag_abs)))
    user_lambda_max = (
        None if chebyshev_lambda_max is None
        else float(chebyshev_lambda_max))
    chebyshev_lambda_max_value = max(
        chebyshev_gershgorin_lambda_max,
        1e-12 if user_lambda_max is None else user_lambda_max)
    if chebyshev_lambda_min is None:
      chebyshev_lambda_min_value = max(
          1e-12, min(
              chebyshev_gershgorin_lambda_min
              if chebyshev_gershgorin_lambda_min > 0.0 else
              chebyshev_lambda_max_value * 1e-8,
              0.5 * chebyshev_lambda_max_value))
    else:
      chebyshev_lambda_min_value = float(chebyshev_lambda_min)
    chebyshev_bound_source = "normalized_gershgorin"
    if (not np.isfinite(chebyshev_lambda_min_value) or
        not np.isfinite(chebyshev_lambda_max_value) or
        chebyshev_lambda_min_value <= 0.0 or
        chebyshev_lambda_max_value <= chebyshev_lambda_min_value):
      raise ValueError(
          "chebyshev auto Gershgorin bounds must produce "
          "0 < chebyshev_lambda_min < chebyshev_lambda_max")

  if fixed_step_acceleration == "chebyshev_ritz_probe":
    scale = (
        preconditioner_diagonal
        if schur_preconditioner in {
            "diagonal",
            "boundary_diagonal",
            "dirichlet_clipped_diagonal",
        }
        else np.ones(len(interface_indices), dtype=float))
    scale = np.maximum(np.asarray(scale, dtype=float).reshape(-1), 1e-12)
    sqrt_scale = np.sqrt(scale)

    def normalized_schur_matvec(vector):
      vector = np.asarray(vector, dtype=float).reshape(-1)
      return schur_matvec(vector / sqrt_scale) / sqrt_scale

    setup_start_matvec_count = int(interface_matvec_count)
    probe_stats = _matrix_free_ritz_largest_eigen_probe_from_matvec(
        normalized_schur_matvec,
        size=len(interface_indices),
        probe_iterations=chebyshev_ritz_probe_iterations,
        seed=chebyshev_ritz_seed,
        safety_factor=chebyshev_ritz_safety_factor)
    chebyshev_bound_setup_matvec_count = int(
        interface_matvec_count - setup_start_matvec_count)
    chebyshev_bound_global_reduction_count = int(
        probe_stats["global_reduction_count"])
    chebyshev_ritz_probe_basis_rank = int(probe_stats["probe_basis_rank"])
    chebyshev_ritz_probe_lambda_max = float(probe_stats["ritz_lambda_max"])
    chebyshev_ritz_probe_safe_lambda_max = float(
        probe_stats["safe_lambda_max"])
    user_lambda_max = (
        None if chebyshev_lambda_max is None
        else float(chebyshev_lambda_max))
    chebyshev_lambda_max_value = max(
        chebyshev_ritz_probe_safe_lambda_max,
        0.0 if user_lambda_max is None else user_lambda_max)
    chebyshev_bound_source = "matrix_free_ritz_probe"
    if (not np.isfinite(chebyshev_lambda_max_value) or
        chebyshev_lambda_max_value <= chebyshev_lambda_min_value):
      raise ValueError(
          "chebyshev Ritz probe bounds must produce "
          "0 < chebyshev_lambda_min < chebyshev_lambda_max")

  if fixed_step_acceleration in {
      "chebyshev_block_gershgorin_certificate",
      "chebyshev_block_gershgorin_budgeted_certificate",
  }:
    run_block_certificate = True
    if fixed_step_acceleration == (
        "chebyshev_block_gershgorin_budgeted_certificate"):
      chebyshev_certificate_preflight = (
          estimate_distributed_block_perron_schur_certificate_preflight(
              local_systems=local_systems,
              interface_indices=interface_indices,
              block_dim=chebyshev_certificate_block_dim,
              variable_count=variable_count,
              perron_iterations=chebyshev_certificate_iterations,
              perron_vector_mode=chebyshev_certificate_vector_mode))
      if (float(chebyshev_certificate_preflight["payload_upper_mb"]) >
          chebyshev_certificate_preflight_hard_cap_mb_value):
        run_block_certificate = False
        chebyshev_certificate_policy_decision = "skip_budget"
        chebyshev_lambda_max_value = (
            (2.0 / dirichlet_diagonal_floor)
            if (chebyshev_lambda_max is None and
                schur_preconditioner == "dirichlet_clipped_diagonal")
            else 2.0 if chebyshev_lambda_max is None
            else float(chebyshev_lambda_max))
        chebyshev_bound_source = (
            "distributed_block_gershgorin_certificate_budget_skipped")
      else:
        if np.isinf(chebyshev_certificate_max_payload_mb_value):
          chebyshev_certificate_policy_decision = "run_unbounded_budget"
        elif (float(chebyshev_certificate_preflight["payload_upper_mb"]) <=
              chebyshev_certificate_max_payload_mb_value):
          chebyshev_certificate_policy_decision = "run_budget_ok"
        else:
          chebyshev_certificate_policy_decision = "run_postcheck_pending"
    if run_block_certificate:
      if chebyshev_certificate_policy_decision == "not_applicable":
        chebyshev_certificate_policy_decision = "run_explicit"
      chebyshev_block_certificate = (
          distributed_block_perron_schur_spectral_certificate(
              local_systems=local_systems,
              interface_indices=interface_indices,
              block_dim=chebyshev_certificate_block_dim,
              variable_count=variable_count,
              damping=damping,
              lambda_max_bound=(
                  2.0 if chebyshev_lambda_max is None
                  else float(chebyshev_lambda_max)),
              perron_iterations=chebyshev_certificate_iterations,
              perron_vector_mode=chebyshev_certificate_vector_mode,
              resolvent_shift=chebyshev_certificate_resolvent_shift))
      chebyshev_bound_global_reduction_count = int(
          chebyshev_block_certificate.get("perron_global_reduction_count", 0))
      if bool(chebyshev_block_certificate.get(
          "weighted_block_gershgorin_theorem_applies", False)):
        chebyshev_lambda_max_value = float(
            chebyshev_block_certificate[
                "weighted_block_gershgorin_lambda_max_bound"])
        chebyshev_bound_source = "distributed_block_gershgorin_certificate"
      else:
        chebyshev_lambda_max_value = (
            2.0 if chebyshev_lambda_max is None
            else float(chebyshev_lambda_max))
        chebyshev_bound_source = (
            "distributed_block_gershgorin_certificate_failed_fallback")
      if chebyshev_certificate_policy_decision == "run_postcheck_pending":
        if (float(chebyshev_block_certificate.get(
            "certificate_payload_bytes", 0)) / 1024.0 / 1024.0 <=
            chebyshev_certificate_max_payload_mb_value):
          chebyshev_certificate_policy_decision = "run_postcheck_budget_ok"
        else:
          chebyshev_certificate_policy_decision = "run_postcheck_over_budget"
    if not run_block_certificate and chebyshev_bound_source == "none":
      chebyshev_lambda_max_value = (
          2.0 if chebyshev_lambda_max is None
          else float(chebyshev_lambda_max))
      chebyshev_bound_source = (
          "distributed_block_gershgorin_certificate_budget_skipped")
    if (not np.isfinite(chebyshev_lambda_max_value) or
        chebyshev_lambda_max_value <= chebyshev_lambda_min_value):
      raise ValueError(
          "chebyshev block-Gershgorin certificate bounds must produce "
          "0 < chebyshev_lambda_min < chebyshev_lambda_max")

  coarse_initial_guess_used = False
  coarse_basis_column_count = 0
  coarse_basis_rank = 0
  coarse_setup_matvec_count = 0
  adaptive_coarse_generation_matvec_count = 0
  coarse_operator_condition = 0.0
  coarse_initial_residual = float(np.linalg.norm(schur_rhs))
  interface_solution = np.zeros(len(interface_indices), dtype=float)
  residual = schur_rhs.copy()
  adaptive_coarse_ritz_selected_rank = 0
  adaptive_coarse_ritz_min_value = 0.0
  adaptive_coarse_ritz_max_value = 0.0
  adaptive_coarse_ritz_selection = "none"
  generated_adaptive_coarse_basis = None

  def deterministic_probe_matrix(row_count: int, column_count: int, seed: int):
    columns: list[np.ndarray] = []
    for column in range(column_count):
      if column == 0:
        vector = np.ones(row_count, dtype=float)
      elif column == 1:
        vector = np.asarray(
            [1.0 if row % 2 == 0 else -1.0 for row in range(row_count)],
            dtype=float)
      else:
        vector = np.empty(row_count, dtype=float)
        stride = 2 * column + 1
        offset = 2 * int(seed) + column + 1
        for row in range(row_count):
          value = ((row + 1) * stride + offset) % 4
          vector[row] = 1.0 if value in {0, 1} else -1.0
      columns.append(vector)
    return np.column_stack(columns) if columns else np.empty((row_count, 0))

  def independent_columns(raw_basis: np.ndarray, tolerance_qr: float = 1e-10):
    raw_basis = np.asarray(raw_basis, dtype=float)
    if raw_basis.ndim == 1:
      raw_basis = raw_basis.reshape(-1, 1)
    if raw_basis.shape[1] == 0:
      return np.empty((raw_basis.shape[0], 0), dtype=float)
    q_basis, r_factor = np.linalg.qr(raw_basis)
    if r_factor.size:
      diag = np.abs(np.diag(r_factor))
      independent = diag > tolerance_qr
    else:
      independent = np.zeros(raw_basis.shape[1], dtype=bool)
    return q_basis[:, independent]

  if adaptive_coarse_basis == "residual_krylov":
    krylov_columns: list[np.ndarray] = []
    if adaptive_coarse_rank > 0 and np.linalg.norm(schur_rhs) > 0.0:
      vector = preconditioner(schur_rhs)
      for _ in range(min(adaptive_coarse_rank, len(interface_indices))):
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-14:
          break
        krylov_columns.append(vector / norm)
        vector = preconditioner(schur_matvec(vector))
        adaptive_coarse_generation_matvec_count += 1
    if krylov_columns:
      generated_adaptive_coarse_basis = np.column_stack(krylov_columns)
  elif adaptive_coarse_basis == "deterministic_ritz":
    interface_dim = len(interface_indices)
    requested_rank = min(adaptive_coarse_rank, interface_dim)
    probe_count = adaptive_coarse_probe_count
    if probe_count <= 0:
      probe_count = max(requested_rank, 2 * requested_rank)
    probe_count = min(max(probe_count, requested_rank), interface_dim)
    if requested_rank > 0 and probe_count > 0:
      raw_probes = deterministic_probe_matrix(
          interface_dim, probe_count, adaptive_coarse_seed)
      probe_space = independent_columns(raw_probes)
      if probe_space.shape[1] < requested_rank:
        fallback = np.eye(interface_dim, dtype=float)
        probe_space = independent_columns(
            np.column_stack([probe_space, fallback]))
      if probe_space.shape[1] > 0:
        probe_space = probe_space[:, :probe_count]
        schur_probe_columns = []
        for column in range(probe_space.shape[1]):
          schur_probe_columns.append(schur_matvec(probe_space[:, column]))
        adaptive_coarse_generation_matvec_count = int(
            len(schur_probe_columns))
        schur_probe = np.column_stack(schur_probe_columns)
        projected_operator = probe_space.T @ schur_probe
        projected_operator = 0.5 * (
            projected_operator + projected_operator.T)
        try:
          eigvals, eigvecs = np.linalg.eigh(projected_operator)
        except np.linalg.LinAlgError:
          eigvals, eigvecs = np.linalg.eig(projected_operator)
          eigvals = np.real(eigvals)
          eigvecs = np.real(eigvecs)
        projected_rhs = probe_space.T @ schur_rhs
        ritz_rhs = eigvecs.T @ projected_rhs
        safe_eigvals = np.maximum(np.abs(eigvals), 1e-12)
        solution_scores = np.abs(ritz_rhs) / safe_eigvals
        order = np.lexsort((eigvals, -solution_scores))
        selected = order[:requested_rank]
        if selected.size:
          ritz_values = eigvals[selected]
          adaptive_coarse_ritz_selected_rank = int(selected.size)
          adaptive_coarse_ritz_min_value = float(np.min(ritz_values))
          adaptive_coarse_ritz_max_value = float(np.max(ritz_values))
          adaptive_coarse_ritz_selection = "rhs_solution_magnitude"
          generated_adaptive_coarse_basis = probe_space @ eigvecs[:, selected]
  if generated_adaptive_coarse_basis is not None:
    if coarse_basis is None:
      coarse_basis = generated_adaptive_coarse_basis
    else:
      base = np.asarray(coarse_basis, dtype=float)
      if base.ndim == 1:
        base = base.reshape(-1, 1)
      adaptive = np.asarray(generated_adaptive_coarse_basis, dtype=float)
      if adaptive.ndim == 1:
        adaptive = adaptive.reshape(-1, 1)
      coarse_basis = np.column_stack([base, adaptive])
  if coarse_basis is not None:
    raw_coarse_basis = np.asarray(coarse_basis, dtype=float)
    if raw_coarse_basis.ndim == 1:
      raw_coarse_basis = raw_coarse_basis.reshape(-1, 1)
    if raw_coarse_basis.shape[0] != len(interface_indices):
      raise ValueError("coarse_basis row count must match interface dimension")
    coarse_basis_column_count = int(raw_coarse_basis.shape[1])
    if coarse_basis_column_count > 0:
      coarse_space = independent_columns(raw_coarse_basis)
      coarse_basis_rank = int(coarse_space.shape[1])
      if coarse_basis_rank > 0:
        schur_coarse_columns = []
        for column in range(coarse_basis_rank):
          schur_coarse_columns.append(schur_matvec(coarse_space[:, column]))
        coarse_setup_matvec_count = int(coarse_basis_rank)
        schur_coarse = np.column_stack(schur_coarse_columns)
        coarse_operator = coarse_space.T @ schur_coarse
        if coarse_regularization > 0.0:
          coarse_operator = (
              coarse_operator +
              coarse_regularization * np.eye(coarse_basis_rank))
        condition = float(np.linalg.cond(coarse_operator))
        if np.isfinite(condition):
          coarse_operator_condition = condition
        coarse_rhs = coarse_space.T @ schur_rhs
        try:
          coarse_solution = np.linalg.solve(coarse_operator, coarse_rhs)
        except np.linalg.LinAlgError:
          coarse_solution = (
              np.linalg.pinv(coarse_operator, rcond=1e-12) @ coarse_rhs)
        interface_solution = coarse_space @ coarse_solution
        residual = schur_rhs - schur_coarse @ coarse_solution
        coarse_initial_residual = float(np.linalg.norm(residual))
        coarse_initial_guess_used = True
  used_iterations = 0
  final_schur_residual = float(np.linalg.norm(residual))
  chebyshev_safety_triggered = False
  chebyshev_safety_trigger_iteration = -1
  chebyshev_safety_rejected_residual = 0.0
  chebyshev_safety_accepted_residual = final_schur_residual
  chebyshev_safety_metric = "none"
  if chebyshev_safety_monitor == "residual_growth":
    chebyshev_safety_metric = "exact_residual_norm"
  elif chebyshev_safety_monitor == "local_residual_envelope":
    chebyshev_safety_metric = "local_residual_envelope"
  chebyshev_safety_rejected_metric = 0.0
  chebyshev_safety_accepted_metric = (
      float(final_schur_residual)
      if chebyshev_safety_metric != "local_residual_envelope"
      else local_residual_envelope_metric(interface_solution))
  chebyshev_safety_accepted_iterations = 0
  if fixed_step_acceleration in {
      "chebyshev",
      "chebyshev_auto_gershgorin",
      "chebyshev_graph_normalized",
      "chebyshev_ritz_probe",
      "chebyshev_block_gershgorin_certificate",
      "chebyshev_block_gershgorin_budgeted_certificate",
  }:
    chebyshev_center = 0.5 * (
        chebyshev_lambda_max_value + chebyshev_lambda_min_value)
    chebyshev_radius = 0.5 * (
        chebyshev_lambda_max_value - chebyshev_lambda_min_value)
    chebyshev_alpha = 0.0
    chebyshev_direction = np.zeros_like(interface_solution)
    for it in range(max(0, int(iterations))):
      if final_schur_residual <= float(tolerance):
        break
      previous_solution = interface_solution.copy()
      previous_residual = residual.copy()
      previous_residual_norm = final_schur_residual
      previous_safety_metric = chebyshev_safety_accepted_metric
      preconditioned_residual = preconditioner(residual)
      if it == 0:
        alpha = 1.0 / chebyshev_center
        beta = 0.0
        chebyshev_direction = preconditioned_residual
      else:
        alpha = 1.0 / (
            chebyshev_center -
            (chebyshev_radius * chebyshev_radius * chebyshev_alpha) / 4.0)
        beta = (
            chebyshev_radius * chebyshev_radius *
            alpha * chebyshev_alpha / 4.0)
        chebyshev_direction = (
            preconditioned_residual + beta * chebyshev_direction)
      interface_solution += alpha * chebyshev_direction
      residual = schur_rhs - schur_matvec(interface_solution)
      final_schur_residual = float(np.linalg.norm(residual))
      if chebyshev_safety_metric == "local_residual_envelope":
        current_safety_metric = local_residual_envelope_metric(
            interface_solution)
      else:
        current_safety_metric = final_schur_residual
      if (chebyshev_safety_monitor != "none" and
          current_safety_metric >
          chebyshev_safety_growth_factor * previous_safety_metric):
        chebyshev_safety_triggered = True
        chebyshev_safety_trigger_iteration = int(it)
        chebyshev_safety_rejected_residual = float(final_schur_residual)
        chebyshev_safety_accepted_residual = float(previous_residual_norm)
        chebyshev_safety_rejected_metric = float(current_safety_metric)
        chebyshev_safety_accepted_metric = float(previous_safety_metric)
        interface_solution = previous_solution
        residual = previous_residual
        final_schur_residual = float(previous_residual_norm)
        break
      used_iterations = it + 1
      chebyshev_safety_accepted_iterations = used_iterations
      chebyshev_safety_accepted_residual = float(final_schur_residual)
      chebyshev_safety_accepted_metric = float(current_safety_metric)
      chebyshev_alpha = alpha
  else:
    for it in range(max(0, int(iterations))):
      if final_schur_residual <= float(tolerance):
        break
      interface_solution += relaxation * preconditioner(residual)
      residual = schur_rhs - schur_matvec(interface_solution)
      final_schur_residual = float(np.linalg.norm(residual))
      used_iterations = it + 1

  solution = np.zeros(variable_count, dtype=float)
  assigned = np.zeros(variable_count, dtype=bool)
  solution[interface_indices] = interface_solution
  assigned[interface_indices] = True
  private_backsubstitution_count = 0
  for system_index, data in enumerate(local_data):
    private_indices = data["private_indices"]
    if private_indices.size == 0:
      continue
    if np.any(assigned[private_indices]):
      raise ValueError(
          f"local system {system_index} reuses private variables")
    private_rhs = (
        data["gradient"][private_indices] -
        matrix_vector_product(data["h_pb"], interface_solution))
    solution[private_indices] = data["private_solver"](private_rhs)
    assigned[private_indices] = True
    private_backsubstitution_count += 1

  normal_residual = -global_gradient.copy()
  for data in local_data:
    normal_residual += matrix_vector_product(data["hessian"], solution)
  rhs_reduction_count = 1
  communication_stats = estimate_local_interface_schur_pcg_communication(
      local_systems=local_systems,
      interface_indices=interface_indices,
      interface_matvec_count=interface_matvec_count,
      pcg_global_reduction_count=chebyshev_bound_global_reduction_count,
      rhs_reduction_count=rhs_reduction_count,
      scalar_bytes=8,
      communication_model=communication_model,
      robot_topology_edges=robot_topology_edges)
  certificate_payload_bytes = int(
      chebyshev_block_certificate.get("certificate_payload_bytes", 0))
  if certificate_payload_bytes:
    communication_stats["certificate_payload_bytes"] = certificate_payload_bytes
    communication_stats["certificate_payload_mb"] = _bytes_to_mb(
        certificate_payload_bytes)
    communication_stats["estimated_comm_bytes"] = int(
        communication_stats.get("estimated_comm_bytes", 0) +
        certificate_payload_bytes)
    communication_stats["estimated_comm_mb"] = _bytes_to_mb(
        int(communication_stats["estimated_comm_bytes"]))
  total_global_reduction_count = int(chebyshev_bound_global_reduction_count)
  return solution, {
      "model": "local_interface_schur_fixed_step",
      "materializes_dense_schur": False,
      "materializes_dense_local_hessian": bool(
          dense_local_hessian_materialized),
      "uses_sparse_private_factorization": bool(
          uses_sparse_private_factorization),
      "dense_private_fallback_count": int(dense_private_fallback_count),
      "local_system_count": int(len(local_systems)),
      "interface_variable_count": int(len(interface_indices)),
      "variable_count": int(variable_count),
      "local_private_variable_count_sum": int(local_private_variable_count_sum),
      "local_singular_count": int(local_singular_count),
      "local_hpp_condition_max": float(local_condition_max),
      "damping": float(damping),
      "schur_preconditioner": schur_preconditioner,
      "schur_preconditioner_private_solve_count": int(
          preconditioner_private_solve_count),
      "schur_preconditioner_diagonal_min": float(
          schur_preconditioner_diagonal_min),
      "schur_preconditioner_diagonal_max": float(
          schur_preconditioner_diagonal_max),
      "dirichlet_diagonal_floor": float(dirichlet_diagonal_floor),
      "dirichlet_clipped_diagonal_clipped_count": int(
          dirichlet_clipped_diagonal_clipped_count),
      "dirichlet_clipped_diagonal_ratio_min": float(
          dirichlet_clipped_diagonal_ratio_min),
      "dirichlet_clipped_diagonal_ratio_max": float(
          dirichlet_clipped_diagonal_ratio_max),
      "relaxation": float(relaxation),
      "coarse_initial_guess_used": bool(coarse_initial_guess_used),
      "coarse_basis_column_count": int(coarse_basis_column_count),
      "coarse_basis_rank": int(coarse_basis_rank),
      "coarse_regularization": float(coarse_regularization),
      "adaptive_coarse_basis": adaptive_coarse_basis,
      "adaptive_coarse_rank": int(adaptive_coarse_rank),
      "adaptive_coarse_probe_count": int(adaptive_coarse_probe_count),
      "adaptive_coarse_seed": int(adaptive_coarse_seed),
      "adaptive_coarse_generation_matvec_count": int(
          adaptive_coarse_generation_matvec_count),
      "adaptive_coarse_ritz_selected_rank": int(
          adaptive_coarse_ritz_selected_rank),
      "adaptive_coarse_ritz_min_value": float(
          adaptive_coarse_ritz_min_value),
      "adaptive_coarse_ritz_max_value": float(
          adaptive_coarse_ritz_max_value),
      "adaptive_coarse_ritz_selection": adaptive_coarse_ritz_selection,
      "fixed_step_acceleration": fixed_step_acceleration,
      "chebyshev_lambda_min": float(chebyshev_lambda_min_value),
      "chebyshev_lambda_max": float(chebyshev_lambda_max_value),
      "chebyshev_bound_source": chebyshev_bound_source,
      "chebyshev_bound_setup_matvec_count": int(
          chebyshev_bound_setup_matvec_count),
      "chebyshev_bound_global_reduction_count": int(
          chebyshev_bound_global_reduction_count),
      "chebyshev_gershgorin_lambda_min": float(
          chebyshev_gershgorin_lambda_min),
      "chebyshev_gershgorin_lambda_max": float(
          chebyshev_gershgorin_lambda_max),
      "chebyshev_ritz_probe_iterations": int(
          chebyshev_ritz_probe_iterations),
      "chebyshev_ritz_seed": int(chebyshev_ritz_seed),
      "chebyshev_ritz_safety_factor": float(
          chebyshev_ritz_safety_factor),
      "chebyshev_ritz_probe_basis_rank": int(
          chebyshev_ritz_probe_basis_rank),
      "chebyshev_ritz_probe_lambda_max": float(
          chebyshev_ritz_probe_lambda_max),
      "chebyshev_ritz_probe_safe_lambda_max": float(
          chebyshev_ritz_probe_safe_lambda_max),
      "chebyshev_certificate_block_dim": int(
          chebyshev_certificate_block_dim),
      "chebyshev_certificate_iterations": int(
          chebyshev_certificate_iterations),
      "chebyshev_certificate_vector_mode": (
          chebyshev_certificate_vector_mode),
      "chebyshev_certificate_resolvent_shift": float(
          chebyshev_certificate_resolvent_shift),
      "chebyshev_certificate_max_payload_mb": float(
          chebyshev_certificate_max_payload_mb_value),
      "chebyshev_certificate_preflight_hard_cap_mb": float(
          chebyshev_certificate_preflight_hard_cap_mb_value),
      "chebyshev_certificate_policy_decision": (
          chebyshev_certificate_policy_decision),
      "chebyshev_certificate_preflight_payload_upper_bytes": int(
          chebyshev_certificate_preflight.get("payload_upper_bytes", 0)),
      "chebyshev_certificate_preflight_payload_upper_mb": float(
          chebyshev_certificate_preflight.get("payload_upper_mb", 0.0)),
      "chebyshev_certificate_preflight_block_packet_payload_upper_bytes": int(
          chebyshev_certificate_preflight.get(
              "block_packet_payload_upper_bytes", 0)),
      "chebyshev_certificate_preflight_perron_payload_upper_bytes": int(
          chebyshev_certificate_preflight.get(
              "perron_payload_upper_bytes", 0)),
      "chebyshev_certificate_preflight_global_reduction_upper_count": int(
          chebyshev_certificate_preflight.get(
              "perron_global_reduction_count_upper", 0)),
      "chebyshev_block_certificate_theorem_applies": bool(
          chebyshev_block_certificate.get(
              "weighted_block_gershgorin_theorem_applies", False)),
      "chebyshev_block_certificate_payload_bytes": int(
          chebyshev_block_certificate.get("certificate_payload_bytes", 0)),
      "chebyshev_block_certificate_payload_mb": _bytes_to_mb(int(
          chebyshev_block_certificate.get("certificate_payload_bytes", 0))),
      "chebyshev_block_certificate_collatz_upper": float(
          chebyshev_block_certificate.get(
              "block_coupling_collatz_upper_bound", 0.0)),
      "chebyshev_block_certificate_component_assemblies": int(
          chebyshev_block_certificate.get("component_schur_assembly_count", 0)),
      "chebyshev_block_certificate_cg_iterations": int(
          chebyshev_block_certificate.get("cg_resolvent_iterations", 0)),
      "chebyshev_safety_monitor": chebyshev_safety_monitor,
      "chebyshev_safety_growth_factor": float(
          chebyshev_safety_growth_factor),
      "chebyshev_safety_triggered": bool(chebyshev_safety_triggered),
      "chebyshev_safety_trigger_iteration": int(
          chebyshev_safety_trigger_iteration),
      "chebyshev_safety_rejected_residual": float(
          chebyshev_safety_rejected_residual),
      "chebyshev_safety_accepted_residual": float(
          chebyshev_safety_accepted_residual),
      "chebyshev_safety_metric": chebyshev_safety_metric,
      "chebyshev_safety_rejected_metric": float(
          chebyshev_safety_rejected_metric),
      "chebyshev_safety_accepted_metric": float(
          chebyshev_safety_accepted_metric),
      "chebyshev_safety_envelope_evaluation_count": int(
          chebyshev_safety_envelope_evaluation_count),
      "chebyshev_safety_accepted_iterations": int(
          chebyshev_safety_accepted_iterations),
      "coarse_setup_matvec_count": int(coarse_setup_matvec_count),
      "coarse_operator_condition": float(coarse_operator_condition),
      "coarse_initial_residual": float(coarse_initial_residual),
      "iterations": int(used_iterations),
      "final_schur_residual": final_schur_residual,
      "fixed_step_global_reduction_count": 0,
      "global_reduction_count": int(total_global_reduction_count),
      "rhs_reduction_count": int(rhs_reduction_count),
      "interface_matvec_count": int(interface_matvec_count),
      "topology_reduction_count": int(
          rhs_reduction_count + interface_matvec_count +
          total_global_reduction_count),
      "interface_solution_norm": float(np.linalg.norm(interface_solution)),
      "private_backsubstitution_count": int(private_backsubstitution_count),
      "unassigned_variable_count": int(np.count_nonzero(~assigned)),
      "final_normal_residual": float(np.linalg.norm(normal_residual)),
      **communication_stats,
  }


def build_graph_local_normal_systems_for_interface_schur(
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
    hessian_storage: str = "dense"):
  """Builds robot-local normal systems for the KKT-consistent Schur target.

  Every measurement edge is assigned to exactly one robot-local system. The
  local systems keep global variable coordinates so their Schur contributions
  can be summed without an additional indexing convention. Robot-owned poses
  incident to inter-robot measurements become interface variables; remaining
  robot-owned poses are private variables for that robot.
  """
  if stage not in {"rotation", "translation"}:
    raise ValueError(f"unsupported local normal system stage: {stage}")
  if edge_owner_policy not in {
      "lower_robot",
      "first_endpoint_robot",
      "second_endpoint_robot",
  }:
    raise ValueError(f"unsupported edge_owner_policy: {edge_owner_policy}")
  if hessian_storage not in {"dense", "sparse"}:
    raise ValueError(f"unsupported hessian_storage: {hessian_storage}")
  if dim is None:
    dim = pose_dimension_from_edges(graph_edges)
  dim = int(dim)
  anchor_pose = pose_ids[0] if anchor_pose is None else int(anchor_pose)
  pose_set = set(int(pose_id) for pose_id in pose_ids)
  for pose_id in pose_ids:
    if int(pose_id) not in robot_of:
      raise ValueError(f"pose {pose_id} is missing robot ownership")
  if stage == "translation" and rotations is None:
    raise ValueError("translation local normal systems require rotations")

  robots = {
      robot_of[int(pose_id)]
      for pose_id in pose_ids
      if int(pose_id) in robot_of
  }
  owned_edges: dict[object, list[Edge]] = {robot: [] for robot in robots}
  interface_pose_ids: set[int] = set()
  assigned_edge_count = 0
  skipped_edge_count = 0

  def choose_owner(edge: Edge):
    first_robot = robot_of[int(edge.i)]
    second_robot = robot_of[int(edge.j)]
    if edge_owner_policy == "first_endpoint_robot":
      return first_robot
    if edge_owner_policy == "second_endpoint_robot":
      return second_robot
    return min(first_robot, second_robot)

  for edge in graph_edges:
    if int(edge.i) not in pose_set or int(edge.j) not in pose_set:
      skipped_edge_count += 1
      continue
    if int(edge.i) not in robot_of or int(edge.j) not in robot_of:
      raise ValueError("edge endpoint is missing robot ownership")
    first_robot = robot_of[int(edge.i)]
    second_robot = robot_of[int(edge.j)]
    if first_robot != second_robot:
      interface_pose_ids.add(int(edge.i))
      interface_pose_ids.add(int(edge.j))
    owner = choose_owner(edge)
    if owner not in owned_edges:
      owned_edges[owner] = []
      robots.add(owner)
    owned_edges[owner].append(edge)
    assigned_edge_count += 1

  if stage == "rotation":
    _, _, meta = assemble_rotation_system(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        dim=dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose)
  else:
    _, _, meta = assemble_translation_system(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        rotations=rotations if rotations is not None else {},
        dim=dim,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose)
  offsets = meta["offsets"]
  block_dim = int(meta["block_dim"])
  variable_count = int(meta["variable_count"])
  interface_indices = _interface_variable_indices_from_offsets(
      offsets, block_dim, interface_pose_ids)

  local_systems = []
  robot_ids = sorted(robots, key=lambda value: str(value))
  private_pose_count = 0
  local_hessian_nonzero_count = 0
  for robot in robot_ids:
    robot_edges = owned_edges.get(robot, [])
    if stage == "rotation":
      matrix, rhs, local_meta = assemble_rotation_system(
          graph_edges=robot_edges,
          pose_ids=pose_ids,
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose)
    else:
      matrix, rhs, local_meta = assemble_translation_system(
          graph_edges=robot_edges,
          pose_ids=pose_ids,
          rotations=rotations if rotations is not None else {},
          dim=dim,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose)
    if int(local_meta["variable_count"]) != variable_count:
      raise ValueError("local and global variable counts do not match")
    local_normal = matrix.T @ matrix
    if hessian_storage == "sparse":
      hessian = local_normal.tocsr()
      local_hessian_nonzero_count += int(hessian.nnz)
    else:
      hessian = local_normal.toarray()
      local_hessian_nonzero_count += int(np.count_nonzero(hessian))
    gradient = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1)
    private_pose_ids = [
        int(pose_id)
        for pose_id in pose_ids
        if (int(pose_id) != anchor_pose and
            robot_of[int(pose_id)] == robot and
            int(pose_id) not in interface_pose_ids and
            int(pose_id) in offsets)
    ]
    private_indices = []
    for pose_id in private_pose_ids:
      start = int(offsets[pose_id])
      private_indices.extend(range(start, start + block_dim))
    private_pose_count += len(private_pose_ids)
    local_systems.append({
        "robot": robot,
        "hessian": hessian,
        "gradient": gradient,
        "private_indices": np.asarray(private_indices, dtype=int),
        "private_pose_ids": private_pose_ids,
        "owned_edge_count": int(len(robot_edges)),
    })

  return local_systems, interface_indices, {
      "model": "graph_local_normal_systems_for_interface_schur",
      "stage": stage,
      "edge_owner_policy": edge_owner_policy,
      "robot_ids": list(robot_ids),
      "local_system_count": int(len(local_systems)),
      "assigned_edge_count": int(assigned_edge_count),
      "skipped_edge_count": int(skipped_edge_count),
      "interface_pose_ids": sorted(int(pose_id) for pose_id in interface_pose_ids),
      "interface_pose_count": int(len(interface_pose_ids)),
      "interface_variable_count": int(len(interface_indices)),
      "private_pose_count": int(private_pose_count),
      "variable_count": int(variable_count),
      "block_dim": int(block_dim),
      "hessian_storage": hessian_storage,
      "materializes_dense_local_hessian": bool(hessian_storage == "dense"),
      "local_hessian_nonzero_count": int(local_hessian_nonzero_count),
  }


def graph_local_interface_schur_gap_decomposition(
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
    solver: str = "pcg",
    iterations: int = 100,
    tolerance: float = 1e-10,
    damping: float = 0.0,
    robot_topology_edges: list[tuple[object, object]] | None = None,
    hop_radius: int = 1,
    communication_model: str = "separator_owner_star",
    schur_preconditioner: str = "none",
    hessian_storage: str = "dense",
    certificate_mode: str = "dense",
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    dirichlet_diagonal_floor: float = 0.25,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
    chebyshev_certificate_block_dim: int | None = None,
    chebyshev_certificate_iterations: int = 128,
    chebyshev_certificate_vector_mode: str = "cg_resolvent",
    chebyshev_certificate_resolvent_shift: float = 0.0,
    chebyshev_certificate_max_payload_mb: float | None = None,
    chebyshev_certificate_preflight_hard_cap_mb: float | None = None):
  """Runs a graph-derived local-Schur solve and reports a gap certificate.

  This is a research diagnostic for DCI-vs-CCI attribution. It builds the
  robot-local normal systems for one chordal-initialization stage, runs the
  requested interface solver, and certifies the result against the exact
  local-Schur reference.
  """
  if solver not in {"reference", "pcg", "limited_hop_pcg", "fixed_step"}:
    raise ValueError(f"unsupported graph local-Schur solver: {solver}")
  if certificate_mode not in {"dense", "matrix_free_dual"}:
    raise ValueError(f"unsupported certificate_mode: {certificate_mode}")
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
          hessian_storage=hessian_storage)
  )
  variable_count = int(build_stats["variable_count"])
  if len(interface_indices) == 0:
    raise ValueError("graph local-Schur gap decomposition requires separators")

  if solver == "reference":
    solution, solver_stats = solve_local_interface_schur_reference(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=variable_count,
        damping=damping)
  elif solver == "pcg":
    solution, solver_stats = solve_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=variable_count,
        iterations=iterations,
        tolerance=tolerance,
        damping=damping,
        schur_preconditioner=schur_preconditioner,
        communication_model=communication_model,
        robot_topology_edges=robot_topology_edges)
  elif solver == "fixed_step":
    solution, solver_stats = solve_local_interface_schur_fixed_step(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=variable_count,
        iterations=iterations,
        tolerance=tolerance,
        damping=damping,
        schur_preconditioner=schur_preconditioner,
        fixed_step_acceleration=fixed_step_acceleration,
        chebyshev_lambda_min=chebyshev_lambda_min,
        chebyshev_lambda_max=chebyshev_lambda_max,
        dirichlet_diagonal_floor=dirichlet_diagonal_floor,
        chebyshev_safety_monitor=chebyshev_safety_monitor,
        chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
        chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
        chebyshev_ritz_seed=chebyshev_ritz_seed,
        chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
        chebyshev_certificate_block_dim=(
            int(build_stats["block_dim"])
            if chebyshev_certificate_block_dim is None
            else int(chebyshev_certificate_block_dim)),
        chebyshev_certificate_iterations=chebyshev_certificate_iterations,
        chebyshev_certificate_vector_mode=(
            chebyshev_certificate_vector_mode),
        chebyshev_certificate_resolvent_shift=(
            chebyshev_certificate_resolvent_shift),
        chebyshev_certificate_max_payload_mb=(
            chebyshev_certificate_max_payload_mb),
        chebyshev_certificate_preflight_hard_cap_mb=(
            chebyshev_certificate_preflight_hard_cap_mb),
        communication_model=communication_model,
        robot_topology_edges=robot_topology_edges)
  else:
    solution, solver_stats = solve_limited_hop_local_interface_schur_pcg(
        local_systems=local_systems,
        interface_indices=interface_indices,
        variable_count=variable_count,
        iterations=iterations,
        tolerance=tolerance,
        damping=damping,
        robot_topology_edges=robot_topology_edges,
        hop_radius=hop_radius)

  if certificate_mode == "matrix_free_dual":
    certificate = matrix_free_local_interface_schur_dual_certificate(
        local_systems=local_systems,
        interface_indices=interface_indices,
        candidate_solution=solution,
        variable_count=variable_count,
        damping=damping,
        dual_iterations=iterations,
        dual_tolerance=tolerance)
  else:
    certificate = local_interface_schur_gap_certificate(
        local_systems=local_systems,
        interface_indices=interface_indices,
        candidate_solution=solution,
        variable_count=variable_count,
        damping=damping)
  stats = {
      "model": "graph_local_interface_schur_gap_decomposition",
      "stage": stage,
      "solver": solver,
      "certificate_mode": certificate_mode,
      "weighted": bool(weighted),
      "cost_mode": cost_mode,
      "edge_owner_policy": edge_owner_policy,
      "hessian_storage": hessian_storage,
      "iterations": int(iterations),
      "tolerance": float(tolerance),
      "damping": float(damping),
      "hop_radius": int(max(0, int(hop_radius))),
      "schur_preconditioner": schur_preconditioner,
      "build_stats": build_stats,
      "solver_stats": solver_stats,
      "certificate": certificate,
      "local_system_count": int(build_stats["local_system_count"]),
      "interface_pose_count": int(build_stats["interface_pose_count"]),
      "interface_variable_count": int(build_stats["interface_variable_count"]),
      "private_pose_count": int(build_stats["private_pose_count"]),
      "variable_count": variable_count,
      "block_dim": int(build_stats["block_dim"]),
      "schur_energy_gap": float(certificate["schur_energy_gap"]),
      "schur_residual_norm": float(certificate["schur_residual_norm"]),
      "schur_residual_dual_energy": float(
          certificate["schur_residual_dual_energy"]),
      "interface_reference_error_norm": float(
          certificate["interface_reference_error_norm"]),
      "full_reference_solution_error_norm": float(
          certificate["full_reference_solution_error_norm"]),
      "local_private_consistency_error_norm": float(
          certificate["local_private_consistency_error_norm"]),
  }
  for key in [
      "estimated_comm_mb",
      "limited_hop_payload_mb",
      "coverage_fraction",
      "max_limited_hop_matvec_error_norm",
  ]:
    if key in solver_stats:
      stats[key] = solver_stats[key]
  return solution, stats


def two_stage_graph_local_chordal_gap_decomposition(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    weighted: bool,
    cost_mode: str,
    anchor_pose: int | None = None,
    solver: str = "pcg",
    rotation_iterations: int = 100,
    translation_iterations: int = 100,
    tolerance: float = 1e-10,
    damping: float = 0.0,
    edge_owner_policy: str = "lower_robot",
    robot_topology_edges: list[tuple[object, object]] | None = None,
    hop_radius: int = 1,
    communication_model: str = "separator_owner_star",
    schur_preconditioner: str = "none",
    hessian_storage: str = "dense",
    certificate_mode: str = "dense",
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    dirichlet_diagonal_floor: float = 0.25,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
    chebyshev_certificate_iterations: int = 128,
    chebyshev_certificate_vector_mode: str = "cg_resolvent",
    chebyshev_certificate_resolvent_shift: float = 0.0,
    chebyshev_certificate_max_payload_mb: float | None = None,
    chebyshev_certificate_preflight_hard_cap_mb: float | None = None):
  """Runs and certifies the two chordal-initialization stages.

  Rotation and translation are certified separately because translation is
  solved after projecting the rotation-stage output. This helper is the
  dataset-sweep entry point for attributing DCI/CCI handoff gap before
  nonlinear DRAN starts.
  """
  pose_ids = sorted(int(pose_id) for pose_id in pose_ids)
  if not pose_ids:
    return {}, {
        "model": "two_stage_graph_local_chordal_gap_decomposition",
        "pose_count": 0,
  }
  anchor_pose = pose_ids[0] if anchor_pose is None else int(anchor_pose)
  dim = pose_dimension_from_edges(graph_edges)

  rot_solution, rot_stats = graph_local_interface_schur_gap_decomposition(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage="rotation",
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=None,
      edge_owner_policy=edge_owner_policy,
      solver=solver,
      iterations=rotation_iterations,
      tolerance=tolerance,
      damping=damping,
      robot_topology_edges=robot_topology_edges,
      hop_radius=hop_radius,
      communication_model=communication_model,
      schur_preconditioner=schur_preconditioner,
      hessian_storage=hessian_storage,
      certificate_mode=certificate_mode,
      fixed_step_acceleration=fixed_step_acceleration,
      chebyshev_lambda_min=chebyshev_lambda_min,
      chebyshev_lambda_max=chebyshev_lambda_max,
      dirichlet_diagonal_floor=dirichlet_diagonal_floor,
      chebyshev_safety_monitor=chebyshev_safety_monitor,
      chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
      chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
      chebyshev_ritz_seed=chebyshev_ritz_seed,
      chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
      chebyshev_certificate_iterations=chebyshev_certificate_iterations,
      chebyshev_certificate_vector_mode=chebyshev_certificate_vector_mode,
      chebyshev_certificate_resolvent_shift=(
          chebyshev_certificate_resolvent_shift),
      chebyshev_certificate_max_payload_mb=(
          chebyshev_certificate_max_payload_mb),
      chebyshev_certificate_preflight_hard_cap_mb=(
          chebyshev_certificate_preflight_hard_cap_mb))
  _, _, rot_meta = assemble_rotation_system(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose)
  rotation_projection_safety = rotation_projection_safety_certificate(
      solution=rot_solution,
      pose_ids=pose_ids,
      offsets=rot_meta["offsets"],
      dim=dim,
      anchor_pose=anchor_pose)
  rotations = _rotations_from_solution(
      rot_solution, pose_ids, rot_meta["offsets"], dim, anchor_pose)

  trans_solution, trans_stats = graph_local_interface_schur_gap_decomposition(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      stage="translation",
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      rotations=rotations,
      edge_owner_policy=edge_owner_policy,
      solver=solver,
      iterations=translation_iterations,
      tolerance=tolerance,
      damping=damping,
      robot_topology_edges=robot_topology_edges,
      hop_radius=hop_radius,
      communication_model=communication_model,
      schur_preconditioner=schur_preconditioner,
      hessian_storage=hessian_storage,
      certificate_mode=certificate_mode,
      fixed_step_acceleration=fixed_step_acceleration,
      chebyshev_lambda_min=chebyshev_lambda_min,
      chebyshev_lambda_max=chebyshev_lambda_max,
      dirichlet_diagonal_floor=dirichlet_diagonal_floor,
      chebyshev_safety_monitor=chebyshev_safety_monitor,
      chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
      chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
      chebyshev_ritz_seed=chebyshev_ritz_seed,
      chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
      chebyshev_certificate_iterations=chebyshev_certificate_iterations,
      chebyshev_certificate_vector_mode=chebyshev_certificate_vector_mode,
      chebyshev_certificate_resolvent_shift=(
          chebyshev_certificate_resolvent_shift),
      chebyshev_certificate_max_payload_mb=(
          chebyshev_certificate_max_payload_mb),
      chebyshev_certificate_preflight_hard_cap_mb=(
          chebyshev_certificate_preflight_hard_cap_mb))
  _, _, trans_meta = assemble_translation_system(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      rotations=rotations,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose)
  translations = _translations_from_solution(
      trans_solution, pose_ids, trans_meta["offsets"], dim, anchor_pose)
  rotation_projection_cost = rotation_projection_cost_certificate(
      graph_edges=graph_edges,
      solution=rot_solution,
      pose_ids=pose_ids,
      offsets=rot_meta["offsets"],
      translations=translations,
      dim=dim,
      anchor_pose=anchor_pose,
      weighted=weighted,
      cost_mode=cost_mode)
  poses = _poses_from_rotations_translations(pose_ids, rotations, translations,
                                             dim)
  handoff_cost = _pose_set_total_chordal_cost(
      graph_edges=graph_edges,
      poses=poses,
      weighted=weighted,
      cost_mode=cost_mode)
  total_schur_energy_gap = (
      float(rot_stats["schur_energy_gap"]) +
      float(trans_stats["schur_energy_gap"]))
  total_schur_residual_norm = math.sqrt(
      float(rot_stats["schur_residual_norm"]) ** 2 +
      float(trans_stats["schur_residual_norm"]) ** 2)
  return poses, {
      "model": "two_stage_graph_local_chordal_gap_decomposition",
      "pose_count": int(len(pose_ids)),
      "dimension": int(dim),
      "anchor_pose": int(anchor_pose),
      "solver": solver,
      "weighted": bool(weighted),
      "cost_mode": cost_mode,
      "edge_owner_policy": edge_owner_policy,
      "hessian_storage": hessian_storage,
      "certificate_mode": certificate_mode,
      "rotation_iterations": int(rotation_iterations),
      "translation_iterations": int(translation_iterations),
      "tolerance": float(tolerance),
      "damping": float(damping),
      "hop_radius": int(max(0, int(hop_radius))),
      "schur_preconditioner": schur_preconditioner,
      "rotation": rot_stats,
      "rotation_projection_safety": rotation_projection_safety,
      "rotation_projection_cost": rotation_projection_cost,
      "translation": trans_stats,
      "handoff_cost": handoff_cost,
      "total_schur_energy_gap": float(total_schur_energy_gap),
      "total_schur_residual_norm": float(total_schur_residual_norm),
  }


def two_stage_dci_certificate_sweep_row(
    dataset: str,
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    weighted: bool,
    cost_mode: str,
    anchor_pose: int | None = None,
    solver: str = "pcg",
    rotation_iterations: int = 100,
    translation_iterations: int = 100,
    tolerance: float = 1e-10,
    damping: float = 0.0,
    edge_owner_policy: str = "lower_robot",
    robot_topology_edges: list[tuple[object, object]] | None = None,
    hop_radius: int = 1,
    communication_model: str = "separator_owner_star",
    schur_preconditioner: str = "none",
    hessian_storage: str = "dense",
    certificate_mode: str = "dense",
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    dirichlet_diagonal_floor: float = 0.25,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
    chebyshev_certificate_iterations: int = 128,
    chebyshev_certificate_vector_mode: str = "cg_resolvent",
    chebyshev_certificate_resolvent_shift: float = 0.0,
    chebyshev_certificate_max_payload_mb: float | None = None,
    chebyshev_certificate_preflight_hard_cap_mb: float | None = None):
  """Builds one flat summary row for the two-stage DCI certificate sweep."""
  _, stats = two_stage_graph_local_chordal_gap_decomposition(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
      solver=solver,
      rotation_iterations=rotation_iterations,
      translation_iterations=translation_iterations,
      tolerance=tolerance,
      damping=damping,
      edge_owner_policy=edge_owner_policy,
      robot_topology_edges=robot_topology_edges,
      hop_radius=hop_radius,
      communication_model=communication_model,
      schur_preconditioner=schur_preconditioner,
      hessian_storage=hessian_storage,
      certificate_mode=certificate_mode,
      fixed_step_acceleration=fixed_step_acceleration,
      chebyshev_lambda_min=chebyshev_lambda_min,
      chebyshev_lambda_max=chebyshev_lambda_max,
      dirichlet_diagonal_floor=dirichlet_diagonal_floor,
      chebyshev_safety_monitor=chebyshev_safety_monitor,
      chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
      chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
      chebyshev_ritz_seed=chebyshev_ritz_seed,
      chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
      chebyshev_certificate_iterations=chebyshev_certificate_iterations,
      chebyshev_certificate_vector_mode=chebyshev_certificate_vector_mode,
      chebyshev_certificate_resolvent_shift=(
          chebyshev_certificate_resolvent_shift),
      chebyshev_certificate_max_payload_mb=(
          chebyshev_certificate_max_payload_mb),
      chebyshev_certificate_preflight_hard_cap_mb=(
          chebyshev_certificate_preflight_hard_cap_mb))
  rotation = stats["rotation"]
  rotation_projection = stats["rotation_projection_safety"]
  rotation_projection_cost = stats["rotation_projection_cost"]
  translation = stats["translation"]
  handoff = stats["handoff_cost"]
  row = {
      "dataset": str(dataset),
      "solver": solver,
      "pose_count": int(stats["pose_count"]),
      "dimension": int(stats["dimension"]),
      "anchor_pose": int(stats["anchor_pose"]),
      "weighted": bool(weighted),
      "cost_mode": cost_mode,
      "edge_owner_policy": edge_owner_policy,
      "hessian_storage": hessian_storage,
      "certificate_mode": certificate_mode,
      "rotation_iterations": int(rotation_iterations),
      "translation_iterations": int(translation_iterations),
      "tolerance": float(tolerance),
      "damping": float(damping),
      "hop_radius": int(max(0, int(hop_radius))),
      "schur_preconditioner": schur_preconditioner,
      "fixed_step_acceleration": fixed_step_acceleration,
      "chebyshev_certificate_max_payload_mb": (
          "" if chebyshev_certificate_max_payload_mb is None
          else float(chebyshev_certificate_max_payload_mb)),
      "chebyshev_certificate_preflight_hard_cap_mb": (
          "" if chebyshev_certificate_preflight_hard_cap_mb is None
          else float(chebyshev_certificate_preflight_hard_cap_mb)),
      "rotation_solver_model": str(
          rotation["solver_stats"].get("model", "")),
      "translation_solver_model": str(
          translation["solver_stats"].get("model", "")),
      "rotation_fixed_step_acceleration": str(
          rotation["solver_stats"].get("fixed_step_acceleration", "")),
      "translation_fixed_step_acceleration": str(
          translation["solver_stats"].get("fixed_step_acceleration", "")),
      "rotation_chebyshev_certificate_policy_decision": str(
          rotation["solver_stats"].get(
              "chebyshev_certificate_policy_decision", "")),
      "translation_chebyshev_certificate_policy_decision": str(
          translation["solver_stats"].get(
              "chebyshev_certificate_policy_decision", "")),
      "rotation_chebyshev_bound_source": str(
          rotation["solver_stats"].get("chebyshev_bound_source", "")),
      "translation_chebyshev_bound_source": str(
          translation["solver_stats"].get("chebyshev_bound_source", "")),
      "rotation_chebyshev_block_certificate_payload_mb": float(
          rotation["solver_stats"].get(
              "chebyshev_block_certificate_payload_mb", 0.0)),
      "translation_chebyshev_block_certificate_payload_mb": float(
          translation["solver_stats"].get(
              "chebyshev_block_certificate_payload_mb", 0.0)),
      "rotation_schur_preconditioner": str(
          rotation["solver_stats"].get("schur_preconditioner", "none")),
      "translation_schur_preconditioner": str(
          translation["solver_stats"].get("schur_preconditioner", "none")),
      "rotation_schur_preconditioner_private_solve_count": int(
          rotation["solver_stats"].get(
              "schur_preconditioner_private_solve_count", 0)),
      "translation_schur_preconditioner_private_solve_count": int(
          translation["solver_stats"].get(
              "schur_preconditioner_private_solve_count", 0)),
      "rotation_schur_energy_gap": float(rotation["schur_energy_gap"]),
      "translation_schur_energy_gap": float(translation["schur_energy_gap"]),
      "total_schur_energy_gap": float(stats["total_schur_energy_gap"]),
      "rotation_schur_residual_norm": float(rotation["schur_residual_norm"]),
      "translation_schur_residual_norm": float(
          translation["schur_residual_norm"]),
      "total_schur_residual_norm": float(stats["total_schur_residual_norm"]),
      "rotation_interface_reference_error_norm": float(
          rotation["interface_reference_error_norm"]),
      "translation_interface_reference_error_norm": float(
          translation["interface_reference_error_norm"]),
      "rotation_private_consistency_error_norm": float(
          rotation["local_private_consistency_error_norm"]),
      "translation_private_consistency_error_norm": float(
          translation["local_private_consistency_error_norm"]),
      "rotation_max_projection_correction_norm": float(
          rotation_projection["max_projection_correction_norm"]),
      "rotation_mean_projection_correction_norm": float(
          rotation_projection["mean_projection_correction_norm"]),
      "rotation_sum_projection_correction_norm": float(
          rotation_projection["sum_projection_correction_norm"]),
      "rotation_max_orthogonality_error": float(
          rotation_projection["max_orthogonality_error"]),
      "rotation_mean_orthogonality_error": float(
          rotation_projection["mean_orthogonality_error"]),
      "rotation_max_determinant_deviation": float(
          rotation_projection["max_determinant_deviation"]),
      "rotation_mean_determinant_deviation": float(
          rotation_projection["mean_determinant_deviation"]),
      "rotation_raw_handoff_cost": float(
          rotation_projection_cost["raw_rotation_cost"]["total_cost"]),
      "rotation_projected_handoff_cost": float(
          rotation_projection_cost["projected_rotation_cost"]["total_cost"]),
      "rotation_projection_cost_delta": float(
          rotation_projection_cost["projected_minus_raw_cost_delta"]),
      "rotation_abs_projection_cost_delta": float(
          rotation_projection_cost["absolute_projection_cost_delta"]),
      "rotation_relative_projection_cost_delta": float(
          rotation_projection_cost["relative_projection_cost_delta"]),
      "handoff_cost": float(handoff["total_cost"]),
      "evaluated_edge_count": int(handoff["evaluated_edge_count"]),
      "missing_edge_count": int(handoff["missing_edge_count"]),
      "all_edges_evaluated": bool(handoff["all_edges_evaluated"]),
      "rotation_comm_mb": float(rotation.get("estimated_comm_mb", 0.0)),
      "translation_comm_mb": float(translation.get("estimated_comm_mb", 0.0)),
      "rotation_coverage_fraction": float(
          rotation.get("coverage_fraction", 1.0)),
      "translation_coverage_fraction": float(
          translation.get("coverage_fraction", 1.0)),
      "rotation_limited_hop_matvec_error_norm": float(
          rotation.get("max_limited_hop_matvec_error_norm", 0.0)),
      "translation_limited_hop_matvec_error_norm": float(
          translation.get("max_limited_hop_matvec_error_norm", 0.0)),
  }
  row["stats"] = stats
  return row


def write_two_stage_dci_certificate_sweep(
    output_dir: Path,
    dataset_specs: list[dict],
    weighted: bool,
    cost_mode: str,
    solver: str = "pcg",
    rotation_iterations: int = 100,
    translation_iterations: int = 100,
    tolerance: float = 1e-10,
    damping: float = 0.0,
    edge_owner_policy: str = "lower_robot",
    robot_topology_edges: list[tuple[object, object]] | None = None,
    hop_radius: int = 1,
    communication_model: str = "separator_owner_star",
    schur_preconditioner: str = "none",
    hessian_storage: str = "dense",
    certificate_mode: str = "dense",
    fixed_step_acceleration: str = "none",
    chebyshev_lambda_min: float | None = None,
    chebyshev_lambda_max: float | None = None,
    dirichlet_diagonal_floor: float = 0.25,
    chebyshev_safety_monitor: str = "none",
    chebyshev_safety_growth_factor: float = 10.0,
    chebyshev_ritz_probe_iterations: int = 8,
    chebyshev_ritz_seed: int = 0,
    chebyshev_ritz_safety_factor: float = 1.05,
    chebyshev_certificate_iterations: int = 128,
    chebyshev_certificate_vector_mode: str = "cg_resolvent",
    chebyshev_certificate_resolvent_shift: float = 0.0,
    chebyshev_certificate_max_payload_mb: float | None = None,
    chebyshev_certificate_preflight_hard_cap_mb: float | None = None):
  """Writes a CSV/JSON report for multiple two-stage DCI certificate rows."""
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  full_stats = []
  for spec in dataset_specs:
    row = two_stage_dci_certificate_sweep_row(
        dataset=str(spec["dataset"]),
        graph_edges=spec["graph_edges"],
        pose_ids=spec["pose_ids"],
        robot_of=spec["robot_of"],
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=spec.get("anchor_pose"),
        solver=solver,
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        tolerance=tolerance,
        damping=damping,
        edge_owner_policy=edge_owner_policy,
        robot_topology_edges=(
            spec.get("robot_topology_edges", robot_topology_edges)),
        hop_radius=spec.get("hop_radius", hop_radius),
        communication_model=communication_model,
        schur_preconditioner=schur_preconditioner,
        hessian_storage=spec.get("hessian_storage", hessian_storage),
        certificate_mode=spec.get("certificate_mode", certificate_mode),
        fixed_step_acceleration=fixed_step_acceleration,
        chebyshev_lambda_min=chebyshev_lambda_min,
        chebyshev_lambda_max=chebyshev_lambda_max,
        dirichlet_diagonal_floor=dirichlet_diagonal_floor,
        chebyshev_safety_monitor=chebyshev_safety_monitor,
        chebyshev_safety_growth_factor=chebyshev_safety_growth_factor,
        chebyshev_ritz_probe_iterations=chebyshev_ritz_probe_iterations,
        chebyshev_ritz_seed=chebyshev_ritz_seed,
        chebyshev_ritz_safety_factor=chebyshev_ritz_safety_factor,
        chebyshev_certificate_iterations=chebyshev_certificate_iterations,
        chebyshev_certificate_vector_mode=chebyshev_certificate_vector_mode,
        chebyshev_certificate_resolvent_shift=(
            chebyshev_certificate_resolvent_shift),
        chebyshev_certificate_max_payload_mb=(
            chebyshev_certificate_max_payload_mb),
        chebyshev_certificate_preflight_hard_cap_mb=(
            chebyshev_certificate_preflight_hard_cap_mb))
    stats = row.pop("stats")
    rows.append(row)
    full_stats.append({
        "dataset": row["dataset"],
        "stats": stats,
    })
  csv_path = output_dir / "two_stage_dci_certificate_summary.csv"
  if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)
  else:
    csv_path.write_text("", encoding="utf-8")
  report = {
      "model": "two_stage_dci_certificate_sweep",
      "row_count": int(len(rows)),
      "weighted": bool(weighted),
      "cost_mode": cost_mode,
      "solver": solver,
      "rotation_iterations": int(rotation_iterations),
      "translation_iterations": int(translation_iterations),
      "tolerance": float(tolerance),
      "damping": float(damping),
      "edge_owner_policy": edge_owner_policy,
      "hessian_storage": hessian_storage,
      "certificate_mode": certificate_mode,
      "fixed_step_acceleration": fixed_step_acceleration,
      "chebyshev_certificate_max_payload_mb": (
          None if chebyshev_certificate_max_payload_mb is None
          else float(chebyshev_certificate_max_payload_mb)),
      "chebyshev_certificate_preflight_hard_cap_mb": (
          None if chebyshev_certificate_preflight_hard_cap_mb is None
          else float(chebyshev_certificate_preflight_hard_cap_mb)),
      "hop_radius": int(max(0, int(hop_radius))),
      "rows": rows,
      "full_stats": full_stats,
  }
  (output_dir / "two_stage_dci_certificate_report.json").write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n",
      encoding="utf-8")
  return report


def _normal_model_residual_norm(hessian, gradient: np.ndarray,
                                solution: np.ndarray):
  residual = _normal_model_residual_vector(hessian, gradient, solution)
  return float(np.linalg.norm(residual))


def _normal_model_residual_vector(hessian, gradient: np.ndarray,
                                  solution: np.ndarray):
  gradient = np.asarray(gradient, dtype=float).reshape(-1)
  solution = np.asarray(solution, dtype=float).reshape(-1)
  return np.asarray(hessian @ solution, dtype=float).reshape(-1) - gradient


def _residual_split_stats(residual: np.ndarray,
                          interface_mask: np.ndarray,
                          prefix: str,
                          tolerance: float = 1e-30):
  residual = np.asarray(residual, dtype=float).reshape(-1)
  interface = residual[interface_mask]
  private = residual[~interface_mask]
  total_norm = float(np.linalg.norm(residual))
  private_norm = float(np.linalg.norm(private))
  interface_norm = float(np.linalg.norm(interface))
  denominator = max(total_norm * total_norm, tolerance)
  return {
      f"{prefix}_residual_norm": total_norm,
      f"{prefix}_private_residual_norm": private_norm,
      f"{prefix}_interface_residual_norm": interface_norm,
      f"{prefix}_private_residual_energy_fraction":
          float((private_norm * private_norm) / denominator),
      f"{prefix}_interface_residual_energy_fraction":
          float((interface_norm * interface_norm) / denominator),
  }


def normal_model_private_interface_residual_split(
    full_hessian,
    full_gradient: np.ndarray,
    selected_hessian,
    selected_gradient: np.ndarray,
    solution: np.ndarray,
    offsets: dict[int, int],
    block_dim: int,
    interface_pose_ids: set[int],
    tolerance: float = 1e-30):
  """Splits selected/full/omitted normal residuals by private/interface blocks."""
  full_gradient = np.asarray(full_gradient, dtype=float).reshape(-1)
  selected_gradient = np.asarray(selected_gradient, dtype=float).reshape(-1)
  solution = np.asarray(solution, dtype=float).reshape(-1)
  block_dim = int(block_dim)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  if len(solution) % block_dim != 0:
    raise ValueError("solution dimension must be divisible by block_dim")
  if full_gradient.shape != solution.shape:
    raise ValueError("full_gradient and solution must have the same shape")
  if selected_gradient.shape != solution.shape:
    raise ValueError("selected_gradient and solution must have the same shape")

  interface_blocks = set()
  for pose_id in interface_pose_ids:
    if pose_id not in offsets:
      continue
    offset = int(offsets[pose_id])
    if offset < 0 or offset + block_dim > len(solution):
      raise ValueError("offset outside solution dimension")
    interface_blocks.add(offset // block_dim)

  block_count = len(solution) // block_dim
  interface_mask = np.zeros(len(solution), dtype=bool)
  for block_id in interface_blocks:
    start = int(block_id) * block_dim
    interface_mask[start:start + block_dim] = True

  full_residual = _normal_model_residual_vector(
      full_hessian, full_gradient, solution)
  selected_residual = _normal_model_residual_vector(
      selected_hessian, selected_gradient, solution)
  omitted_residual = _normal_model_residual_vector(
      full_hessian - selected_hessian,
      full_gradient - selected_gradient,
      solution)
  return {
      "model": "private_interface_normal_residual_split",
      "block_dim": block_dim,
      "variable_count": int(len(solution)),
      "block_count": int(block_count),
      "private_block_count": int(block_count - len(interface_blocks)),
      "interface_block_count": int(len(interface_blocks)),
      "interface_pose_count": int(len(interface_blocks)),
      **_residual_split_stats(
          selected_residual, interface_mask, "selected_model", tolerance),
      **_residual_split_stats(
          full_residual, interface_mask, "full_model", tolerance),
      **_residual_split_stats(
          omitted_residual, interface_mask, "omitted_model", tolerance),
  }


def normal_model_representativeness_certificate(full_hessian,
                                                full_gradient: np.ndarray,
                                                selected_hessian,
                                                selected_gradient: np.ndarray,
                                                solution: np.ndarray,
                                                relative_tolerance:
                                                float = 1e-6,
                                                absolute_tolerance:
                                                float = 1e-10):
  """Checks whether a selected normal model represents the full normal model."""
  full_gradient = np.asarray(full_gradient, dtype=float).reshape(-1)
  selected_gradient = np.asarray(selected_gradient, dtype=float).reshape(-1)
  solution = np.asarray(solution, dtype=float).reshape(-1)
  selected_residual = _normal_model_residual_norm(
      selected_hessian, selected_gradient, solution)
  full_residual = _normal_model_residual_norm(
      full_hessian, full_gradient, solution)
  omitted_hessian = full_hessian - selected_hessian
  omitted_gradient = full_gradient - selected_gradient
  omitted_residual = _normal_model_residual_norm(
      omitted_hessian, omitted_gradient, solution)
  gradient_norm = float(np.linalg.norm(full_gradient))
  denominator = max(gradient_norm, absolute_tolerance)
  relative_omitted = omitted_residual / denominator
  return {
      "selected_model_residual_norm": selected_residual,
      "full_model_residual_norm": full_residual,
      "omitted_model_residual_norm": omitted_residual,
      "full_gradient_norm": gradient_norm,
      "relative_omitted_model_residual": relative_omitted,
      "relative_tolerance": float(relative_tolerance),
      "absolute_tolerance": float(absolute_tolerance),
      "normal_model_representative": bool(
          omitted_residual <= absolute_tolerance or
          relative_omitted <= relative_tolerance),
      "model": "full_vs_selected_normal_model_residual",
  }


def combine_normal_model_representativeness(rotation_cert: dict,
                                            translation_cert: dict):
  selected = math.sqrt(
      float(rotation_cert.get("selected_model_residual_norm", 0.0))**2 +
      float(translation_cert.get("selected_model_residual_norm", 0.0))**2)
  full = math.sqrt(
      float(rotation_cert.get("full_model_residual_norm", 0.0))**2 +
      float(translation_cert.get("full_model_residual_norm", 0.0))**2)
  omitted = math.sqrt(
      float(rotation_cert.get("omitted_model_residual_norm", 0.0))**2 +
      float(translation_cert.get("omitted_model_residual_norm", 0.0))**2)
  gradient_norm = math.sqrt(
      float(rotation_cert.get("full_gradient_norm", 0.0))**2 +
      float(translation_cert.get("full_gradient_norm", 0.0))**2)
  absolute_tolerance = max(
      float(rotation_cert.get("absolute_tolerance", 0.0)),
      float(translation_cert.get("absolute_tolerance", 0.0)),
  )
  denominator = max(gradient_norm, absolute_tolerance, 1e-30)
  return {
      "selected_model_residual_norm": selected,
      "full_model_residual_norm": full,
      "omitted_model_residual_norm": omitted,
      "full_gradient_norm": gradient_norm,
      "relative_omitted_model_residual": omitted / denominator,
      "relative_tolerance": max(
          float(rotation_cert.get("relative_tolerance", 0.0)),
          float(translation_cert.get("relative_tolerance", 0.0)),
      ),
      "absolute_tolerance": absolute_tolerance,
      "normal_model_representative": bool(
          rotation_cert.get("normal_model_representative", False) and
          translation_cert.get("normal_model_representative", False)),
      "model": "combined_full_vs_selected_normal_model_residual",
  }


def combine_private_interface_residual_split(rotation_split: dict,
                                             translation_split: dict):
  combined = {
      "model": "combined_private_interface_normal_residual_split",
      "block_count": int(rotation_split.get("block_count", 0)) +
          int(translation_split.get("block_count", 0)),
      "private_block_count": int(rotation_split.get("private_block_count", 0)) +
          int(translation_split.get("private_block_count", 0)),
      "interface_block_count":
          int(rotation_split.get("interface_block_count", 0)) +
          int(translation_split.get("interface_block_count", 0)),
  }
  for prefix in ("selected_model", "full_model", "omitted_model"):
    private = math.sqrt(
        float(rotation_split.get(f"{prefix}_private_residual_norm", 0.0))**2 +
        float(translation_split.get(f"{prefix}_private_residual_norm", 0.0))**2)
    interface = math.sqrt(
        float(rotation_split.get(f"{prefix}_interface_residual_norm", 0.0))**2 +
        float(translation_split.get(f"{prefix}_interface_residual_norm", 0.0))**2)
    total = math.sqrt(private * private + interface * interface)
    denominator = max(total * total, 1e-30)
    combined.update({
        f"{prefix}_residual_norm": total,
        f"{prefix}_private_residual_norm": private,
        f"{prefix}_interface_residual_norm": interface,
        f"{prefix}_private_residual_energy_fraction":
            float((private * private) / denominator),
        f"{prefix}_interface_residual_energy_fraction":
            float((interface * interface) / denominator),
    })
  return combined


def estimate_omitted_force_vector_communication(force: np.ndarray,
                                                block_dim: int,
                                                tolerance: float = 1e-12):
  force = np.asarray(force, dtype=float).reshape(-1)
  block_dim = int(block_dim)
  nonzero_blocks = 0
  for start in range(0, len(force), block_dim):
    if float(np.linalg.norm(force[start:start + block_dim])) > tolerance:
      nonzero_blocks += 1
  payload_bytes = nonzero_blocks * block_dim * 8
  return {
      "nonzero_force_blocks": nonzero_blocks,
      "block_dim": block_dim,
      "omitted_force_payload_bytes": payload_bytes,
      "omitted_force_payload_mb": _bytes_to_mb(payload_bytes),
      "model": "omitted_force_block_vector_payload",
  }


def _as_sparse_hessian(hessian):
  if hasattr(hessian, "tocsc"):
    return hessian.tocsc()
  if coo_matrix is None:
    raise RuntimeError("scipy sparse support is required")
  return coo_matrix(np.asarray(hessian, dtype=float)).tocsc()


def _omitted_block_gershgorin_majorizer(full_hessian,
                                        selected_hessian,
                                        block_dim: int,
                                        tolerance: float = 1e-12):
  if coo_matrix is None:
    raise RuntimeError("scipy sparse support is required")
  full_hessian = _as_sparse_hessian(full_hessian)
  selected_hessian = _as_sparse_hessian(selected_hessian)
  block_dim = int(block_dim)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  if full_hessian.shape != selected_hessian.shape:
    raise ValueError("full and selected Hessian shapes do not match")
  variable_count = int(full_hessian.shape[0])
  if variable_count % block_dim != 0:
    raise ValueError("Hessian dimension must be divisible by block_dim")
  omitted = (full_hessian - selected_hessian).tocoo()
  block_count = variable_count // block_dim
  offdiag_blocks: dict[tuple[int, int], np.ndarray] = {}
  for row, col, value in zip(omitted.row, omitted.col, omitted.data):
    value = float(value)
    if abs(value) <= tolerance:
      continue
    row_block = int(row) // block_dim
    col_block = int(col) // block_dim
    if row_block == col_block:
      continue
    key = (row_block, col_block)
    if key not in offdiag_blocks:
      offdiag_blocks[key] = np.zeros((block_dim, block_dim), dtype=float)
    offdiag_blocks[key][int(row) % block_dim, int(col) % block_dim] += value

  block_row_sums = np.zeros(block_count, dtype=float)
  for (row_block, _), block in offdiag_blocks.items():
    norm = float(np.linalg.norm(block, ord=2))
    if norm > tolerance:
      block_row_sums[row_block] += norm

  rows = []
  cols = []
  data = []
  nonzero_blocks = 0
  for block_id, scale in enumerate(block_row_sums):
    if float(scale) <= tolerance:
      continue
    nonzero_blocks += 1
    for local in range(block_dim):
      index = block_id * block_dim + local
      rows.append(index)
      cols.append(index)
      data.append(float(scale))
  majorizer = coo_matrix(
      (data, (rows, cols)), shape=full_hessian.shape).tocsc()
  payload_bytes = nonzero_blocks * 8
  return majorizer, {
      "model": "omitted_block_gershgorin_scalar_majorizer",
      "block_dim": block_dim,
      "nonzero_curvature_blocks": nonzero_blocks,
      "offdiag_block_count": len(offdiag_blocks),
      "curvature_payload_bytes": payload_bytes,
      "curvature_payload_mb": _bytes_to_mb(payload_bytes),
  }


def _orthonormalize_directions(vectors: list[np.ndarray],
                               max_rank: int,
                               tolerance: float = 1e-12):
  basis = []
  for vector in vectors:
    candidate = np.asarray(vector, dtype=float).reshape(-1).copy()
    for existing in basis:
      candidate -= float(existing @ candidate) * existing
    norm = float(np.linalg.norm(candidate))
    if norm <= tolerance:
      continue
    basis.append(candidate / norm)
    if len(basis) >= max_rank:
      break
  if not basis:
    return np.zeros((len(vectors[0]) if vectors else 0, 0), dtype=float)
  return np.column_stack(basis)


def subspace_error_certificate(reference_solution: np.ndarray,
                               solution: np.ndarray,
                               subspace_vectors: list[np.ndarray],
                               reference_label: str = "reference_solution",
                               tolerance: float = 1e-12):
  reference = np.asarray(reference_solution, dtype=float).reshape(-1)
  solution = np.asarray(solution, dtype=float).reshape(-1)
  if reference.shape != solution.shape:
    raise ValueError("reference_solution and solution must have the same shape")
  error = reference - solution
  error_norm = float(np.linalg.norm(error))
  basis = _orthonormalize_directions(
      subspace_vectors, len(error), tolerance=tolerance)
  if basis.shape[1] == 0:
    explainable = np.zeros_like(error)
  else:
    explainable = basis @ (basis.T @ error)
  missed = error - explainable
  explainable_norm = float(np.linalg.norm(explainable))
  missed_norm = float(np.linalg.norm(missed))
  denominator = max(error_norm * error_norm, tolerance)
  return {
      "model": "reference_error_subspace_projection",
      "reference_label": reference_label,
      "subspace_rank": int(basis.shape[1]),
      "subspace_vector_count": len(subspace_vectors),
      "solution_norm": float(np.linalg.norm(solution)),
      "reference_norm": float(np.linalg.norm(reference)),
      "error_norm": error_norm,
      "explainable_error_norm": explainable_norm,
      "missed_error_norm": missed_norm,
      "explainable_error_energy_fraction":
          float((explainable_norm * explainable_norm) / denominator),
      "missed_error_energy_fraction":
          float((missed_norm * missed_norm) / denominator),
  }


def _projected_pairwise_omitted_curvature(omitted_hessian,
                                          basis: np.ndarray,
                                          block_dim: int,
                                          tolerance: float = 1e-12):
  basis = np.asarray(basis, dtype=float)
  rank = int(basis.shape[1])
  if rank == 0:
    return {
        "matrix": np.zeros((0, 0), dtype=float),
        "projected_block_pair_count": 0,
        "projected_scalar_count": 0,
        "payload_bytes": 0,
        "payload_mb": 0.0,
        "payload_model": "pairwise_projected",
        "projected_source": "pairwise_block_aggregation",
        "directed_scalar_entry_count": 0,
    }
  block_dim = int(block_dim)
  if block_dim <= 0:
    raise ValueError("block_dim must be positive")
  if basis.shape[0] % block_dim != 0:
    raise ValueError("basis row count must be divisible by block_dim")
  omitted = _as_sparse_hessian(omitted_hessian).tocoo()
  active_pairs = set()
  projected = np.zeros((rank, rank), dtype=float)
  directed_entries = 0
  for row, col, value in zip(omitted.row, omitted.col, omitted.data):
    value = float(value)
    if abs(value) <= tolerance:
      continue
    projected += value * np.outer(basis[int(row), :], basis[int(col), :])
    row_block = int(row) // block_dim
    col_block = int(col) // block_dim
    if row_block == col_block:
      continue
    row_slice = basis[row_block * block_dim:(row_block + 1) * block_dim, :]
    col_slice = basis[col_block * block_dim:(col_block + 1) * block_dim, :]
    if (float(np.linalg.norm(row_slice)) <= tolerance or
        float(np.linalg.norm(col_slice)) <= tolerance):
      continue
    active_pairs.add((min(row_block, col_block), max(row_block, col_block)))
    directed_entries += 1
  scalar_per_pair = rank * (rank + 1) // 2
  scalar_count = len(active_pairs) * scalar_per_pair
  payload_bytes = scalar_count * 8
  return {
      "matrix": projected,
      "projected_block_pair_count": len(active_pairs),
      "projected_scalar_count": scalar_count,
      "payload_bytes": payload_bytes,
      "payload_mb": _bytes_to_mb(payload_bytes),
      "payload_model": "pairwise_projected",
      "projected_source": "pairwise_block_aggregation",
      "directed_scalar_entry_count": directed_entries,
  }


def _best_projected_step_residual_norm(full_hessian,
                                       full_gradient: np.ndarray,
                                       current: np.ndarray,
                                       delta: np.ndarray,
                                       before_norm: float,
                                       line_search_scales):
  best_scale = 0.0
  best_norm = float(before_norm)
  for scale in line_search_scales:
    candidate = current + float(scale) * delta
    candidate_residual = np.asarray(
        full_hessian @ candidate, dtype=float).reshape(-1) - full_gradient
    candidate_norm = float(np.linalg.norm(candidate_residual))
    if candidate_norm < best_norm:
      best_norm = candidate_norm
      best_scale = float(scale)
  return best_scale, best_norm


def apply_omitted_force_correction(full_hessian,
                                   full_gradient: np.ndarray,
                                   selected_hessian,
                                   selected_gradient: np.ndarray,
                                   blocks: dict[int, np.ndarray],
                                   solution: np.ndarray,
                                   iterations: int,
                                   linear_solver: str,
                                   relaxation: float,
                                   damping: float,
                                   block_dim: int | None = None,
                                   correction_rounds: int = 1,
                                   line_search_scales: tuple[float, ...] | None = None,
                                   curvature_model: str = "none",
                                   curvature_rank: int = 1,
                                   curvature_payload_model: str = "global_projected",
                                   curvature_rank_scheduler: str = "fixed",
                                   curvature_rank_max_condition: float | None = None,
                                   curvature_rank_condition_policy: str = "fixed",
                                   curvature_rank_condition_mad_scale: float = 3.0,
                                   reference_solution: np.ndarray | None = None,
                                   reference_label: str = "reference_solution"):
  if curvature_model not in {
      "none",
      "block_gershgorin",
      "directional_secant",
      "subspace_secant",
  }:
    raise ValueError(f"unsupported curvature_model: {curvature_model}")
  if curvature_payload_model not in {"global_projected", "pairwise_projected"}:
    raise ValueError(
        f"unsupported curvature_payload_model: {curvature_payload_model}")
  if curvature_rank_scheduler not in {"fixed", "projected_residual"}:
    raise ValueError(
        f"unsupported curvature_rank_scheduler: {curvature_rank_scheduler}")
  if curvature_rank_condition_policy not in {"fixed", "median_mad"}:
    raise ValueError(
        "unsupported curvature_rank_condition_policy: "
        f"{curvature_rank_condition_policy}")
  if (curvature_rank_max_condition is not None and
      float(curvature_rank_max_condition) <= 0.0):
    raise ValueError("curvature_rank_max_condition must be positive")
  if float(curvature_rank_condition_mad_scale) < 0.0:
    raise ValueError("curvature_rank_condition_mad_scale must be nonnegative")
  max_projected_condition = (
      None if curvature_rank_max_condition is None
      else float(curvature_rank_max_condition))
  condition_mad_scale = float(curvature_rank_condition_mad_scale)
  curvature_rank = max(1, int(curvature_rank))
  selected_hessian = _as_sparse_hessian(selected_hessian)
  full_hessian = _as_sparse_hessian(full_hessian)
  full_gradient = np.asarray(full_gradient, dtype=float).reshape(-1)
  selected_gradient = np.asarray(selected_gradient, dtype=float).reshape(-1)
  current = np.asarray(solution, dtype=float).reshape(-1)
  if line_search_scales is None:
    line_search_scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
  if block_dim is None:
    block_dim = max(1, len(current))
  if curvature_model == "block_gershgorin":
    curvature_hessian, curvature_comm = _omitted_block_gershgorin_majorizer(
        full_hessian, selected_hessian, block_dim)
    correction_hessian = selected_hessian + curvature_hessian
  else:
    correction_hessian = selected_hessian
    curvature_comm = {
        "model": "none",
        "block_dim": int(block_dim),
        "nonzero_curvature_blocks": 0,
        "offdiag_block_count": 0,
        "curvature_payload_bytes": 0,
        "curvature_payload_mb": 0.0,
    }
  omitted_hessian = full_hessian - selected_hessian
  before = normal_model_representativeness_certificate(
      full_hessian=full_hessian,
      full_gradient=full_gradient,
      selected_hessian=selected_hessian,
      selected_gradient=selected_gradient,
      solution=current)
  round_reports = []
  total_payload_bytes = 0
  accepted_rounds = 0
  last_delta = np.zeros_like(current)
  last_solve_stats = {}
  last_full_residual = np.asarray(
      full_hessian @ current, dtype=float).reshape(-1) - full_gradient
  last_selected_residual = np.asarray(
      selected_hessian @ current, dtype=float).reshape(-1) - selected_gradient
  last_omitted_force = last_full_residual - last_selected_residual
  dynamic_curvature_payload_bytes = 0
  global_projected_curvature_payload_bytes = 0
  pairwise_projected_curvature_payload_bytes = 0
  pairwise_projected_block_pairs = 0
  pairwise_projected_scalar_count = 0
  pairwise_projection_errors: list[float] = []
  subspace_basis_history: list[np.ndarray] = []
  accepted_subspace_vectors: list[np.ndarray] = []
  max_observed_subspace_rank = 0

  for round_index in range(max(0, int(correction_rounds))):
    full_residual = np.asarray(
        full_hessian @ current, dtype=float).reshape(-1) - full_gradient
    selected_residual = np.asarray(
        selected_hessian @ current, dtype=float).reshape(-1) - selected_gradient
    omitted_force = full_residual - selected_residual
    comm = estimate_omitted_force_vector_communication(omitted_force, block_dim)
    total_payload_bytes += int(comm["omitted_force_payload_bytes"])
    before_norm = float(np.linalg.norm(full_residual))
    delta, solve_stats = solve_distributed_hessian_system(
        correction_hessian,
        -full_residual,
        blocks,
        iterations,
        linear_solver,
        relaxation,
        damping)
    directional_secant_scale = 1.0
    selected_directional_curvature = 0.0
    omitted_directional_curvature = 0.0
    subspace_rank = 0
    subspace_candidate_rank = 0
    subspace_payload_bytes = 0
    subspace_rank_selection = {
        "enabled": False,
        "scheduler": curvature_rank_scheduler,
        "candidate_count": 0,
        "selected_rank": 0,
        "selected_index": 0,
        "selection_rule": "fixed_requested_rank",
    }
    subspace_pairwise_projection_error = 0.0
    round_subspace_basis = None
    subspace_projected_source = (
        "pairwise_block_aggregation"
        if curvature_payload_model == "pairwise_projected"
        else "global_projected_matrix")
    if curvature_model == "directional_secant":
      selected_directional_curvature = float(
          delta @ np.asarray(selected_hessian @ delta, dtype=float).reshape(-1))
      omitted_directional_curvature = float(
          delta @ np.asarray(omitted_hessian @ delta, dtype=float).reshape(-1))
      dynamic_curvature_payload_bytes += 8
      positive_omitted = max(0.0, omitted_directional_curvature)
      denominator = selected_directional_curvature + positive_omitted
      if selected_directional_curvature > 1e-30 and denominator > 1e-30:
        directional_secant_scale = max(
            0.0, min(1.0, selected_directional_curvature / denominator))
        delta = directional_secant_scale * delta
      solve_stats = {
          **solve_stats,
          "directional_secant_scale": directional_secant_scale,
          "selected_directional_curvature": selected_directional_curvature,
          "omitted_directional_curvature": omitted_directional_curvature,
      }
    elif curvature_model == "subspace_secant":
      basis = _orthonormalize_directions(
          [*subspace_basis_history, delta], curvature_rank)
      subspace_candidate_rank = int(basis.shape[1])
      if (curvature_rank_scheduler == "projected_residual" and
          subspace_candidate_rank > 1):
        rank_candidates = []
        best_index = 0
        best_norm = math.inf
        best_payload_bytes = math.inf
        for prefix_rank in range(1, subspace_candidate_rank + 1):
          prefix_basis = basis[:, :prefix_rank]
          prefix_selected_projected = prefix_basis.T @ (
              selected_hessian @ prefix_basis)
          prefix_global_omitted_projected = prefix_basis.T @ (
              omitted_hessian @ prefix_basis)
          prefix_pairwise_payload = _projected_pairwise_omitted_curvature(
              omitted_hessian, prefix_basis, block_dim)
          if curvature_payload_model == "pairwise_projected":
            prefix_omitted_projected = np.asarray(
                prefix_pairwise_payload["matrix"], dtype=float)
            prefix_payload_bytes = int(prefix_pairwise_payload["payload_bytes"])
          else:
            prefix_omitted_projected = prefix_global_omitted_projected
            prefix_payload_bytes = prefix_rank * (prefix_rank + 1) // 2 * 8
          prefix_full_projected = (
              prefix_selected_projected + prefix_omitted_projected)
          prefix_condition = float(np.linalg.cond(
              prefix_full_projected + damping * np.eye(prefix_rank)))
          prefix_residual_projected = prefix_basis.T @ full_residual
          try:
            prefix_step = np.linalg.solve(
                prefix_full_projected + damping * np.eye(prefix_rank),
                -prefix_residual_projected)
          except np.linalg.LinAlgError:
            prefix_step, *_ = np.linalg.lstsq(
                prefix_full_projected + damping * np.eye(prefix_rank),
                -prefix_residual_projected,
                rcond=None)
          prefix_delta = prefix_basis @ prefix_step
          prefix_best_scale, prefix_best_norm = (
              _best_projected_step_residual_norm(
                  full_hessian, full_gradient, current, prefix_delta,
                  before_norm, line_search_scales))
          improvement = max(0.0, before_norm - prefix_best_norm)
          value_per_byte = (
              improvement / float(prefix_payload_bytes)
              if prefix_payload_bytes > 0 else math.inf)
          candidate = {
              "rank": prefix_rank,
              "best_line_search_scale": prefix_best_scale,
              "predicted_full_residual_norm": prefix_best_norm,
              "predicted_full_residual_improvement": improvement,
              "curvature_payload_bytes": prefix_payload_bytes,
              "projected_condition": prefix_condition,
              "value_per_byte": value_per_byte,
          }
          rank_candidates.append(candidate)
        adaptive_projected_condition = None
        effective_max_projected_condition = max_projected_condition
        if curvature_rank_condition_policy == "median_mad":
          projected_conditions = np.asarray([
              float(item["projected_condition"]) for item in rank_candidates
          ], dtype=float)
          median_condition = float(np.median(projected_conditions))
          mad_condition = float(np.median(
              np.abs(projected_conditions - median_condition)))
          adaptive_projected_condition = (
              median_condition + condition_mad_scale * mad_condition)
          effective_max_projected_condition = adaptive_projected_condition
          if max_projected_condition is not None:
            effective_max_projected_condition = min(
                max_projected_condition, adaptive_projected_condition)
        for candidate in rank_candidates:
          condition = float(candidate["projected_condition"])
          candidate["condition_feasible"] = (
              effective_max_projected_condition is None or
              condition <= effective_max_projected_condition + 1e-12)
        feasible_indices = [
            index for index, item in enumerate(rank_candidates)
            if bool(item["condition_feasible"])
        ]
        condition_fallback_used = not feasible_indices
        eligible_indices = feasible_indices
        if condition_fallback_used:
          eligible_indices = list(range(len(rank_candidates)))
        for index in eligible_indices:
          candidate = rank_candidates[index]
          prefix_best_norm = float(candidate["predicted_full_residual_norm"])
          prefix_payload_bytes = int(candidate["curvature_payload_bytes"])
          if (prefix_best_norm < best_norm - 1e-12 or
              (abs(prefix_best_norm - best_norm) <= 1e-12 and
               prefix_payload_bytes < best_payload_bytes)):
            best_index = index
            best_norm = prefix_best_norm
            best_payload_bytes = prefix_payload_bytes
        selected_rank = int(rank_candidates[best_index]["rank"])
        basis = basis[:, :selected_rank]
        subspace_rank_selection = {
            "enabled": True,
            "scheduler": curvature_rank_scheduler,
            "selection_rule":
                ("minimum_feasible_predicted_full_residual_then_curvature_payload"
                 if effective_max_projected_condition is not None
                 else "minimum_predicted_full_residual_then_curvature_payload"),
            "condition_guard_enabled":
                effective_max_projected_condition is not None,
            "condition_policy": curvature_rank_condition_policy,
            "condition_mad_scale": condition_mad_scale,
            "fixed_projected_condition": max_projected_condition,
            "adaptive_projected_condition": adaptive_projected_condition,
            "max_projected_condition": effective_max_projected_condition,
            "condition_feasible_candidate_count": len(feasible_indices),
            "condition_fallback_used": bool(condition_fallback_used),
            "eligible_candidate_count": len(feasible_indices),
            "candidate_count": len(rank_candidates),
            "selected_rank": selected_rank,
            "selected_index": best_index,
            "candidate_ranks": [
                int(item["rank"]) for item in rank_candidates
            ],
            "candidate_predicted_full_residual_norms": [
                float(item["predicted_full_residual_norm"])
                for item in rank_candidates
            ],
            "candidate_curvature_payload_bytes": [
                int(item["curvature_payload_bytes"])
                for item in rank_candidates
            ],
            "candidate_projected_conditions": [
                float(item["projected_condition"]) for item in rank_candidates
            ],
            "candidate_condition_feasible": [
                bool(item["condition_feasible"]) for item in rank_candidates
            ],
            "candidate_value_per_byte": [
                float(item["value_per_byte"]) for item in rank_candidates
            ],
            "candidates": rank_candidates,
        }
      subspace_rank = int(basis.shape[1])
      max_observed_subspace_rank = max(max_observed_subspace_rank, subspace_rank)
      if subspace_rank > 0:
        round_subspace_basis = basis.copy()
        selected_projected = basis.T @ (
            selected_hessian @ basis)
        global_omitted_projected = basis.T @ (
            omitted_hessian @ basis)
        pairwise_payload = _projected_pairwise_omitted_curvature(
            omitted_hessian, basis, block_dim)
        pairwise_omitted_projected = np.asarray(
            pairwise_payload["matrix"], dtype=float)
        subspace_pairwise_projection_error = float(
            np.linalg.norm(pairwise_omitted_projected - global_omitted_projected))
        pairwise_projection_errors.append(subspace_pairwise_projection_error)
        if curvature_payload_model == "pairwise_projected":
          omitted_projected = pairwise_omitted_projected
          full_projected = selected_projected + omitted_projected
        else:
          omitted_projected = global_omitted_projected
          full_projected = selected_projected + omitted_projected
        residual_projected = basis.T @ full_residual
        try:
          projected_step = np.linalg.solve(
              full_projected + damping * np.eye(subspace_rank),
              -residual_projected)
        except np.linalg.LinAlgError:
          projected_step, *_ = np.linalg.lstsq(
              full_projected + damping * np.eye(subspace_rank),
              -residual_projected,
              rcond=None)
        delta = basis @ projected_step
        symmetric_scalar_count = subspace_rank * (subspace_rank + 1) // 2
        global_payload_bytes = symmetric_scalar_count * 8
        pairwise_payload_bytes = int(pairwise_payload["payload_bytes"])
        subspace_payload_bytes = (
            pairwise_payload_bytes
            if curvature_payload_model == "pairwise_projected"
            else global_payload_bytes)
        dynamic_curvature_payload_bytes += subspace_payload_bytes
        global_projected_curvature_payload_bytes += global_payload_bytes
        pairwise_projected_curvature_payload_bytes += pairwise_payload_bytes
        pairwise_projected_block_pairs += int(
            pairwise_payload["projected_block_pair_count"])
        pairwise_projected_scalar_count += int(
            pairwise_payload["projected_scalar_count"])
        selected_directional_curvature = float(
            delta @ np.asarray(selected_hessian @ delta, dtype=float).reshape(-1))
        omitted_directional_curvature = float(
            delta @ np.asarray(omitted_hessian @ delta, dtype=float).reshape(-1))
        solve_stats = {
            **solve_stats,
            "subspace_rank": subspace_rank,
            "subspace_candidate_rank": subspace_candidate_rank,
            "subspace_rank_scheduler": curvature_rank_scheduler,
            "subspace_rank_selection": subspace_rank_selection,
            "subspace_projected_condition": float(np.linalg.cond(
                full_projected + damping * np.eye(subspace_rank))),
            "subspace_curvature_payload_bytes": subspace_payload_bytes,
            "subspace_global_projected_payload_bytes": global_payload_bytes,
            "subspace_pairwise_projected_payload_bytes": pairwise_payload_bytes,
            "subspace_projected_block_pair_count":
                int(pairwise_payload["projected_block_pair_count"]),
            "subspace_projected_source": subspace_projected_source,
            "subspace_pairwise_projection_error":
                subspace_pairwise_projection_error,
            "subspace_pairwise_projection_relative_error": (
                subspace_pairwise_projection_error /
                max(1e-30, float(np.linalg.norm(global_omitted_projected)))),
            "selected_directional_curvature": selected_directional_curvature,
            "omitted_directional_curvature": omitted_directional_curvature,
            "omitted_projected_frobenius_norm":
                float(np.linalg.norm(omitted_projected)),
            "global_omitted_projected_frobenius_norm":
                float(np.linalg.norm(global_omitted_projected)),
            "pairwise_omitted_projected_frobenius_norm":
                float(np.linalg.norm(pairwise_omitted_projected)),
        }
    best_scale = 0.0
    best_candidate = current
    best_norm = before_norm
    for scale in line_search_scales:
      candidate = current + float(scale) * delta
      candidate_residual = np.asarray(
          full_hessian @ candidate, dtype=float).reshape(-1) - full_gradient
      candidate_norm = float(np.linalg.norm(candidate_residual))
      if candidate_norm < best_norm:
        best_norm = candidate_norm
        best_scale = float(scale)
        best_candidate = candidate
    accepted = best_scale > 0.0 and best_norm < before_norm - 1e-12
    round_reports.append({
        "round": round_index,
        "accepted": bool(accepted),
        "line_search_scale": best_scale,
        "before_full_residual_norm": before_norm,
        "after_full_residual_norm": best_norm if accepted else before_norm,
        "delta_norm": float(np.linalg.norm(delta)),
        "omitted_force_norm": float(np.linalg.norm(omitted_force)),
        "directional_secant_scale": directional_secant_scale,
        "selected_directional_curvature": selected_directional_curvature,
        "omitted_directional_curvature": omitted_directional_curvature,
        "subspace_rank": subspace_rank,
        "subspace_candidate_rank": subspace_candidate_rank,
        "subspace_rank_scheduler": (
            curvature_rank_scheduler if curvature_model == "subspace_secant"
            else "fixed"),
        "subspace_curvature_payload_bytes": subspace_payload_bytes,
        "subspace_curvature_payload_model": curvature_payload_model,
        "subspace_projected_source": subspace_projected_source,
        "subspace_pairwise_projection_error": (
            solve_stats.get("subspace_pairwise_projection_error", 0.0)
            if curvature_model == "subspace_secant" else 0.0),
        "subspace_global_projected_payload_bytes": (
            solve_stats.get("subspace_global_projected_payload_bytes", 0)
            if curvature_model == "subspace_secant" else 0),
        "subspace_pairwise_projected_payload_bytes": (
            solve_stats.get("subspace_pairwise_projected_payload_bytes", 0)
            if curvature_model == "subspace_secant" else 0),
        "subspace_projected_block_pair_count": (
            solve_stats.get("subspace_projected_block_pair_count", 0)
            if curvature_model == "subspace_secant" else 0),
        "communication_estimate": comm,
        "solve_stats": solve_stats,
    })
    last_delta = best_scale * delta
    last_solve_stats = solve_stats
    last_full_residual = full_residual
    last_selected_residual = selected_residual
    last_omitted_force = omitted_force
    if not accepted:
      break
    current = best_candidate
    accepted_rounds += 1
    if round_subspace_basis is not None and round_subspace_basis.shape[1] > 0:
      for col in range(round_subspace_basis.shape[1]):
        accepted_subspace_vectors.append(round_subspace_basis[:, col].copy())
    if curvature_model == "subspace_secant" and float(
        np.linalg.norm(last_delta)) > 1e-12:
      subspace_basis_history.append(last_delta.copy())
      if len(subspace_basis_history) > curvature_rank - 1:
        subspace_basis_history = subspace_basis_history[-(curvature_rank - 1):]

  after = normal_model_representativeness_certificate(
      full_hessian=full_hessian,
      full_gradient=full_gradient,
      selected_hessian=selected_hessian,
      selected_gradient=selected_gradient,
      solution=current)
  reference_subspace_error = None
  if reference_solution is not None:
    reference_subspace_error = subspace_error_certificate(
        reference_solution=reference_solution,
        solution=current,
        subspace_vectors=accepted_subspace_vectors,
        reference_label=reference_label)
  static_curvature_payload_bytes = (
      int(curvature_comm.get("curvature_payload_bytes", 0))
      if round_reports and curvature_model == "block_gershgorin"
      else 0)
  total_curvature_payload_bytes = (
      static_curvature_payload_bytes + dynamic_curvature_payload_bytes)
  if curvature_model == "directional_secant":
    reported_curvature_comm = {
        "model": "directional_secant_omitted_curvature_scalar",
        "scalar_count": len(round_reports),
        "curvature_payload_bytes": dynamic_curvature_payload_bytes,
        "curvature_payload_mb": _bytes_to_mb(dynamic_curvature_payload_bytes),
    }
  elif curvature_model == "subspace_secant":
    max_pairwise_projection_error = (
        max(pairwise_projection_errors) if pairwise_projection_errors else 0.0)
    sum_pairwise_projection_error = float(sum(pairwise_projection_errors))
    reported_curvature_comm = {
        "model": "subspace_secant_omitted_curvature_projection",
        "payload_model": curvature_payload_model,
        "projected_source": (
            "pairwise_block_aggregation"
            if curvature_payload_model == "pairwise_projected"
            else "global_projected_matrix"),
        "max_subspace_rank": max_observed_subspace_rank,
        "requested_subspace_rank": curvature_rank,
        "attempted_rounds": len(round_reports),
        "global_projected_payload_bytes":
            global_projected_curvature_payload_bytes,
        "global_projected_payload_mb":
            _bytes_to_mb(global_projected_curvature_payload_bytes),
        "pairwise_projected_payload_bytes":
            pairwise_projected_curvature_payload_bytes,
        "pairwise_projected_payload_mb":
            _bytes_to_mb(pairwise_projected_curvature_payload_bytes),
        "projected_block_pair_count": pairwise_projected_block_pairs,
        "projected_scalar_count": pairwise_projected_scalar_count,
        "pairwise_projection_error_count": len(pairwise_projection_errors),
        "max_pairwise_projection_error": max_pairwise_projection_error,
        "mean_pairwise_projection_error": (
            sum_pairwise_projection_error / len(pairwise_projection_errors)
            if pairwise_projection_errors else 0.0),
        "pairwise_projection_error_sum": sum_pairwise_projection_error,
        "curvature_payload_bytes": dynamic_curvature_payload_bytes,
        "curvature_payload_mb": _bytes_to_mb(dynamic_curvature_payload_bytes),
    }
  elif curvature_model == "block_gershgorin":
    reported_curvature_comm = {
        **curvature_comm,
        "curvature_payload_bytes": static_curvature_payload_bytes,
        "curvature_payload_mb": _bytes_to_mb(static_curvature_payload_bytes),
    }
  else:
    reported_curvature_comm = curvature_comm
  comm = {
      "nonzero_force_blocks": sum(
          int(report.get("communication_estimate", {}).get(
              "nonzero_force_blocks", 0))
          for report in round_reports),
      "block_dim": int(block_dim),
      "omitted_force_payload_bytes": total_payload_bytes,
      "omitted_force_payload_mb": _bytes_to_mb(total_payload_bytes),
      "curvature_payload_bytes": int(total_curvature_payload_bytes),
      "curvature_payload_mb": _bytes_to_mb(total_curvature_payload_bytes),
      "model": "omitted_force_block_vector_payload",
  }
  comm["total_correction_payload_bytes"] = (
      int(comm["omitted_force_payload_bytes"]) +
      int(comm["curvature_payload_bytes"]))
  comm["total_correction_payload_mb"] = _bytes_to_mb(
      comm["total_correction_payload_bytes"])
  return current, {
      "model": "omitted_force_vector_correction",
      "curvature_model": curvature_model,
      "curvature_rank": curvature_rank,
      "curvature_rank_scheduler": curvature_rank_scheduler,
      "curvature_rank_max_condition": max_projected_condition,
      "curvature_rank_condition_policy": curvature_rank_condition_policy,
      "curvature_rank_condition_mad_scale": condition_mad_scale,
      "curvature_payload_model": curvature_payload_model,
      "curvature_communication_estimate": reported_curvature_comm,
      "reference_subspace_error": reference_subspace_error,
      "before": before,
      "after": after,
      "requested_rounds": int(correction_rounds),
      "attempted_rounds": len(round_reports),
      "accepted_rounds": accepted_rounds,
      "rounds": round_reports,
      "delta_norm": float(np.linalg.norm(last_delta)),
      "omitted_force_norm": float(np.linalg.norm(last_omitted_force)),
      "full_residual_norm": float(np.linalg.norm(last_full_residual)),
      "selected_residual_norm": float(np.linalg.norm(last_selected_residual)),
      "total_solve_iterations": sum(
          int(report.get("solve_stats", {}).get("iterations", 0))
          for report in round_reports),
      "solve_stats": last_solve_stats,
      "communication_estimate": comm,
  }


def _robot_blocks_from_offsets(offsets: dict[int, int], block_dim: int,
                               robot_of: dict[int, int]):
  by_robot: dict[int, list[int]] = {}
  for pose_id, offset in sorted(offsets.items()):
    robot = robot_of[pose_id]
    by_robot.setdefault(robot, []).extend(range(offset, offset + block_dim))
  return {robot: np.asarray(cols, dtype=int) for robot, cols in by_robot.items()}


def block_jacobi_normal_solve(matrix, rhs: np.ndarray,
                              blocks: dict[int, np.ndarray],
                              iterations: int,
                              relaxation: float = 1.0,
                              damping: float = 1e-12,
                              tolerance: float = 1e-10):
  if matrix is None or identity is None or splu is None:
    raise RuntimeError("scipy sparse support is required")
  hessian = (matrix.T @ matrix).tocsc()
  gradient = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1)
  return block_jacobi_hessian_solve(hessian, gradient, blocks, iterations,
                                    relaxation, damping, tolerance)


def block_jacobi_hessian_solve(hessian, gradient: np.ndarray,
                               blocks: dict[int, np.ndarray],
                               iterations: int,
                               relaxation: float = 1.0,
                               damping: float = 1e-12,
                               tolerance: float = 1e-10):
  if hessian is None or identity is None or splu is None:
    raise RuntimeError("scipy sparse support is required")
  if hessian.shape[1] == 0:
    return np.zeros(0), {
        "iterations": 0,
        "final_normal_residual": 0.0,
        "block_count": 0,
        "method": "block_jacobi",
    }
  hessian = hessian.tocsc()
  gradient = np.asarray(gradient, dtype=float).reshape(-1)
  x = np.zeros(hessian.shape[1], dtype=float)
  factors = {}
  for robot, cols in blocks.items():
    hbb = hessian[cols[:, None], cols].tocsc()
    if damping > 0.0:
      hbb = hbb + damping * identity(hbb.shape[0], format="csc")
    factors[robot] = splu(hbb)

  final_residual = float(np.linalg.norm(hessian @ x - gradient))
  used_iterations = 0
  for it in range(max(0, iterations)):
    previous = x.copy()
    for robot, cols in blocks.items():
      h_rows = hessian[cols, :]
      rhs_block = gradient[cols] - np.asarray(h_rows @ previous).reshape(-1)
      hbb_x = np.asarray(hessian[cols[:, None], cols] @ previous[cols]).reshape(-1)
      rhs_block += hbb_x
      solved = factors[robot].solve(rhs_block)
      x[cols] = (1.0 - relaxation) * previous[cols] + relaxation * solved
    used_iterations = it + 1
    final_residual = float(np.linalg.norm(hessian @ x - gradient))
    if final_residual <= tolerance:
      break

  return x, {
      "iterations": used_iterations,
      "final_normal_residual": final_residual,
      "block_count": len(blocks),
      "method": "block_jacobi",
      "relaxation": relaxation,
      "damping": damping,
  }


def _build_block_preconditioner(hessian, blocks: dict[int, np.ndarray],
                                damping: float):
  factors = {}
  for robot, cols in blocks.items():
    hbb = hessian[cols[:, None], cols].tocsc()
    if damping > 0.0:
      hbb = hbb + damping * identity(hbb.shape[0], format="csc")
    factors[robot] = splu(hbb)
  return factors


def _apply_block_preconditioner(factors, blocks: dict[int, np.ndarray],
                                residual: np.ndarray):
  output = np.zeros_like(residual)
  for robot, cols in blocks.items():
    output[cols] = factors[robot].solve(residual[cols])
  return output


def block_pcg_normal_solve(matrix, rhs: np.ndarray,
                           blocks: dict[int, np.ndarray],
                           iterations: int,
                           damping: float = 1e-12,
                           tolerance: float = 1e-10):
  if matrix is None or identity is None or splu is None:
    raise RuntimeError("scipy sparse support is required")
  hessian = (matrix.T @ matrix).tocsc()
  gradient = np.asarray(matrix.T @ rhs, dtype=float).reshape(-1)
  return block_pcg_hessian_solve(hessian, gradient, blocks, iterations,
                                 damping, tolerance)


def block_pcg_hessian_solve(hessian, gradient: np.ndarray,
                            blocks: dict[int, np.ndarray],
                            iterations: int,
                            damping: float = 1e-12,
                            tolerance: float = 1e-10):
  if hessian is None or identity is None or splu is None:
    raise RuntimeError("scipy sparse support is required")
  if hessian.shape[1] == 0:
    return np.zeros(0), {
        "iterations": 0,
        "final_normal_residual": 0.0,
        "block_count": 0,
        "method": "block_pcg",
    }
  hessian = hessian.tocsc()
  gradient = np.asarray(gradient, dtype=float).reshape(-1)
  factors = _build_block_preconditioner(hessian, blocks, damping)
  x = np.zeros(hessian.shape[1], dtype=float)
  residual = gradient - np.asarray(hessian @ x).reshape(-1)
  z = _apply_block_preconditioner(factors, blocks, residual)
  direction = z.copy()
  rz_old = float(residual @ z)
  final_residual = float(np.linalg.norm(residual))
  used_iterations = 0
  if final_residual <= tolerance or abs(rz_old) <= 1e-30:
    return x, {
        "iterations": 0,
        "final_normal_residual": final_residual,
        "block_count": len(blocks),
        "method": "block_pcg",
        "damping": damping,
        "global_reduction_count": 0,
    }

  for it in range(max(0, iterations)):
    h_direction = np.asarray(hessian @ direction).reshape(-1)
    denom = float(direction @ h_direction)
    if abs(denom) <= 1e-30:
      break
    alpha = rz_old / denom
    x += alpha * direction
    residual -= alpha * h_direction
    final_residual = float(np.linalg.norm(residual))
    used_iterations = it + 1
    if final_residual <= tolerance:
      break
    z = _apply_block_preconditioner(factors, blocks, residual)
    rz_new = float(residual @ z)
    if abs(rz_old) <= 1e-30:
      break
    beta = rz_new / rz_old
    direction = z + beta * direction
    rz_old = rz_new

  return x, {
      "iterations": used_iterations,
      "final_normal_residual": final_residual,
      "block_count": len(blocks),
      "method": "block_pcg",
      "damping": damping,
      "global_reduction_count": 2 * used_iterations,
  }


def _pcg_solve_from_matvec(matvec,
                           rhs: np.ndarray,
                           iterations: int,
                           tolerance: float = 1e-10,
                           preconditioner=None,
                           initial_solution: np.ndarray | None = None,
                           collect_residual_history: int = 0):
  rhs = np.asarray(rhs, dtype=float).reshape(-1)
  if initial_solution is None:
    x = np.zeros_like(rhs)
  else:
    x = np.asarray(initial_solution, dtype=float).reshape(-1).copy()
    if x.shape != rhs.shape:
      raise ValueError("initial_solution dimension does not match rhs")
  max_history = max(0, int(collect_residual_history))
  residual_history: list[np.ndarray] = []

  def record_residual(vector):
    if max_history > 0 and len(residual_history) < max_history:
      residual_history.append(np.asarray(vector, dtype=float).reshape(-1).copy())

  residual = rhs - np.asarray(matvec(x), dtype=float).reshape(-1)
  record_residual(residual)
  z = preconditioner(residual) if preconditioner is not None else residual.copy()
  direction = z.copy()
  rz_old = float(residual @ z)
  final_residual = float(np.linalg.norm(residual))
  used_iterations = 0
  if final_residual <= tolerance or abs(rz_old) <= 1e-30:
    stats = {
        "iterations": 0,
        "final_residual": final_residual,
        "global_reduction_count": 0,
    }
    if max_history > 0:
      stats["residual_history"] = residual_history
    return x, stats
  for it in range(max(0, iterations)):
    mat_direction = np.asarray(matvec(direction), dtype=float).reshape(-1)
    denom = float(direction @ mat_direction)
    if abs(denom) <= 1e-30:
      break
    alpha = rz_old / denom
    x += alpha * direction
    residual -= alpha * mat_direction
    record_residual(residual)
    final_residual = float(np.linalg.norm(residual))
    used_iterations = it + 1
    if final_residual <= tolerance:
      break
    z = preconditioner(residual) if preconditioner is not None else residual.copy()
    rz_new = float(residual @ z)
    if abs(rz_old) <= 1e-30:
      break
    beta = rz_new / rz_old
    direction = z + beta * direction
    rz_old = rz_new
  stats = {
      "iterations": used_iterations,
      "final_residual": final_residual,
      "global_reduction_count": 2 * used_iterations,
  }
  if max_history > 0:
    stats["residual_history"] = residual_history
  return x, stats


def _interface_variable_indices_from_offsets(offsets: dict[int, int],
                                             block_dim: int,
                                             interface_pose_ids: set[int]):
  indices: list[int] = []
  for pose_id in sorted(interface_pose_ids):
    if pose_id not in offsets:
      continue
    offset = int(offsets[pose_id])
    indices.extend(range(offset, offset + int(block_dim)))
  return np.asarray(sorted(set(indices)), dtype=int)


def _interface_variable_blocks_from_offsets(offsets: dict[int, int],
                                            block_dim: int,
                                            interface_pose_ids: set[int]):
  blocks: list[np.ndarray] = []
  for pose_id in sorted(interface_pose_ids):
    if pose_id not in offsets:
      continue
    offset = int(offsets[pose_id])
    blocks.append(np.arange(offset, offset + int(block_dim), dtype=int))
  return blocks


def _interface_robot_coordinate_coarse_basis(offsets: dict[int, int],
                                             block_dim: int,
                                             interface_pose_ids: set[int],
                                             robot_of: dict[int, int],
                                             interface_indices: np.ndarray):
  """Builds robot-wise coordinate-constant modes on interface variables."""
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  if interface_indices.size == 0:
    return np.zeros((0, 0), dtype=float)
  interface_position = {
      int(global_index): local_index
      for local_index, global_index in enumerate(interface_indices)
  }
  vectors: list[np.ndarray] = []
  robots = sorted({
      int(robot_of[pose_id])
      for pose_id in interface_pose_ids
      if pose_id in offsets and pose_id in robot_of
  })
  for robot in robots:
    for coord in range(int(block_dim)):
      vector = np.zeros(len(interface_indices), dtype=float)
      for pose_id in sorted(interface_pose_ids):
        if pose_id not in offsets or pose_id not in robot_of:
          continue
        if int(robot_of[pose_id]) != robot:
          continue
        global_index = int(offsets[pose_id]) + coord
        local_index = interface_position.get(global_index)
        if local_index is not None:
          vector[local_index] = 1.0
      vectors.append(vector)
  return _orthonormalize_directions(
      vectors, max_rank=len(vectors), tolerance=1e-12)


def _interface_summary_components(separator_summary: dict,
                                  allowed_blocks: set[int]):
  adjacency = _interface_summary_adjacency(separator_summary, allowed_blocks)
  nodes = set(adjacency)
  seen = set()
  components: list[list[int]] = []
  for start in sorted(nodes):
    if start in seen:
      continue
    stack = [start]
    seen.add(start)
    component: list[int] = []
    while stack:
      node = stack.pop()
      component.append(node)
      for nbr in sorted(adjacency[node]):
        if nbr in seen:
          continue
        seen.add(nbr)
        stack.append(nbr)
    components.append(sorted(component))
  components.sort(key=lambda item: (-len(item), item[0] if item else -1))
  return components


def _interface_summary_adjacency(separator_summary: dict,
                                 allowed_blocks: set[int]):
  nodes = {int(block) for block in allowed_blocks}
  _, _, summary_edges = _normal_summary_block_graph(separator_summary)
  adjacency = {node: set() for node in nodes}
  for src, dst in summary_edges:
    src = int(src)
    dst = int(dst)
    if src not in nodes or dst not in nodes:
      continue
    adjacency[src].add(dst)
    adjacency[dst].add(src)
  return adjacency


def _sort_interface_summary_components(
    components: list[list[int]],
    separator_summary: dict,
    block_dim: int,
    component_selection_mode: str):
  if component_selection_mode not in {
      "size",
      "gradient_energy",
      "projected_merit",
  }:
    raise ValueError(
        f"unsupported component selection mode: {component_selection_mode}")
  components = [list(component) for component in components]
  components.sort(key=lambda item: (-len(item), item[0] if item else -1))
  gradient_blocks = separator_summary.get("gradient_blocks", {})

  if component_selection_mode == "gradient_energy":
    def component_gradient_energy(component):
      energy = 0.0
      for block_id in component:
        block = gradient_blocks.get(int(block_id), None)
        if block is None:
          continue
        vector = np.asarray(block, dtype=float).reshape(-1)
        energy += float(vector @ vector)
      return energy

    components.sort(
        key=lambda item: (
            -component_gradient_energy(item),
            -len(item),
            item[0] if item else -1))
  elif component_selection_mode == "projected_merit":
    hessian_lookup: dict[tuple[int, int], np.ndarray] = {}
    for key, values in separator_summary.get("hessian_blocks", {}).items():
      row_block, col_block = _normal_summary_key_to_blocks(key)
      hessian_lookup[(row_block, col_block)] = np.asarray(
          values, dtype=float).reshape(block_dim, block_dim)

    def component_projected_merit(component):
      projected_gradient = np.zeros(block_dim, dtype=float)
      projected_hessian = np.zeros((block_dim, block_dim), dtype=float)
      for block_id in component:
        gradient_block = gradient_blocks.get(int(block_id), None)
        if gradient_block is not None:
          projected_gradient += np.asarray(
              gradient_block, dtype=float).reshape(block_dim)
      for row_block in component:
        for col_block in component:
          block = hessian_lookup.get((int(row_block), int(col_block)))
          if block is not None:
            projected_hessian += block
      projected_hessian = 0.5 * (projected_hessian + projected_hessian.T)
      scale = max(1.0, float(np.linalg.norm(projected_hessian, ord=np.inf)))
      regularized = (
          projected_hessian + (1e-12 * scale) * np.eye(block_dim))
      try:
        step = np.linalg.pinv(regularized, rcond=1e-12) @ projected_gradient
      except np.linalg.LinAlgError:
        return 0.0
      merit = float(projected_gradient @ step)
      if not math.isfinite(merit):
        return 0.0
      return max(0.0, merit)

    components.sort(
        key=lambda item: (
            -component_projected_merit(item),
            -len(item),
            item[0] if item else -1))
  return components


def _interface_component_coordinate_coarse_basis(
    offsets: dict[int, int],
    block_dim: int,
    interface_pose_ids: set[int],
    interface_indices: np.ndarray,
    separator_summary: dict,
    component_limit: int | None = None,
    component_selection_mode: str = "size"):
  """Builds coordinate-constant coarse modes on summary graph components."""
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  if interface_indices.size == 0:
    return np.zeros((0, 0), dtype=float)
  block_dim = int(block_dim)
  interface_position = {
      int(global_index): local_index
      for local_index, global_index in enumerate(interface_indices)
  }
  interface_blocks = {
      int(offsets[pose_id]) // block_dim
      for pose_id in interface_pose_ids
      if pose_id in offsets
  }
  components = _interface_summary_components(separator_summary, interface_blocks)
  components = _sort_interface_summary_components(
      components, separator_summary, block_dim, component_selection_mode)
  if component_limit is not None:
    limit = int(component_limit)
    components = [] if limit <= 0 else components[:limit]
  vectors: list[np.ndarray] = []
  for component in components:
    for coord in range(block_dim):
      vector = np.zeros(len(interface_indices), dtype=float)
      for block_id in component:
        global_index = int(block_id) * block_dim + coord
        local_index = interface_position.get(global_index)
        if local_index is not None:
          vector[local_index] = 1.0
      vectors.append(vector)
  if not vectors:
    return np.zeros((len(interface_indices), 0), dtype=float)
  return _orthonormalize_directions(
      vectors, max_rank=len(vectors), tolerance=1e-12)


def _interface_component_endpoint_coordinate_coarse_basis(
    offsets: dict[int, int],
    block_dim: int,
    interface_pose_ids: set[int],
    interface_indices: np.ndarray,
    separator_summary: dict,
    component_limit: int | None = None,
    component_selection_mode: str = "size"):
  """Builds coordinate modes on endpoint blocks of summary graph components."""
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  if interface_indices.size == 0:
    return np.zeros((0, 0), dtype=float)
  block_dim = int(block_dim)
  interface_position = {
      int(global_index): local_index
      for local_index, global_index in enumerate(interface_indices)
  }
  interface_blocks = {
      int(offsets[pose_id]) // block_dim
      for pose_id in interface_pose_ids
      if pose_id in offsets
  }
  adjacency = _interface_summary_adjacency(separator_summary, interface_blocks)
  components = _interface_summary_components(separator_summary, interface_blocks)
  components = _sort_interface_summary_components(
      components, separator_summary, block_dim, component_selection_mode)
  if component_limit is not None:
    limit = int(component_limit)
    components = [] if limit <= 0 else components[:limit]
  vectors: list[np.ndarray] = []
  for component in components:
    endpoints = [
        int(block_id)
        for block_id in component
        if len(adjacency.get(int(block_id), set())) <= 1
    ]
    if not endpoints:
      endpoints = sorted({int(component[0]), int(component[-1])})
    for block_id in sorted(set(endpoints)):
      for coord in range(block_dim):
        vector = np.zeros(len(interface_indices), dtype=float)
        global_index = int(block_id) * block_dim + coord
        local_index = interface_position.get(global_index)
        if local_index is not None:
          vector[local_index] = 1.0
          vectors.append(vector)
  if not vectors:
    return np.zeros((len(interface_indices), 0), dtype=float)
  return _orthonormalize_directions(
      vectors, max_rank=len(vectors), tolerance=1e-12)


def _combine_interface_coarse_bases(bases: list[np.ndarray],
                                    row_count: int):
  vectors: list[np.ndarray] = []
  for basis in bases:
    array = np.asarray(basis, dtype=float)
    if array.size == 0:
      continue
    if array.ndim == 1:
      array = array.reshape(-1, 1)
    if array.shape[0] != int(row_count):
      raise ValueError("coarse basis row count mismatch")
    vectors.extend(array[:, col] for col in range(array.shape[1]))
  if not vectors:
    return np.zeros((int(row_count), 0), dtype=float)
  return _orthonormalize_directions(
      vectors, max_rank=len(vectors), tolerance=1e-12)


def _residual_deflation_basis_from_history(residual_history,
                                           max_rank: int):
  vectors = [
      np.asarray(vector, dtype=float).reshape(-1)
      for vector in residual_history
      if np.asarray(vector, dtype=float).size > 0
  ]
  if not vectors:
    return np.zeros((0, 0), dtype=float)
  row_count = len(vectors[0])
  for vector in vectors:
    if len(vector) != row_count:
      raise ValueError("residual history vectors must have matching dimensions")
  vectors.sort(key=lambda vector: float(vector @ vector), reverse=True)
  return _orthonormalize_directions(
      vectors, max_rank=max(0, int(max_rank)), tolerance=1e-12)


def _ritz_low_mode_basis_from_operator(matvec,
                                       size: int,
                                       max_rank: int,
                                       probe_iterations: int,
                                       seed_vectors: list[np.ndarray] | None = None,
                                       tolerance: float = 1e-12):
  """Approximates low Ritz modes of a linear operator from an Arnoldi probe."""
  size = int(size)
  max_rank = max(0, int(max_rank))
  requested_iterations = max(0, int(probe_iterations))
  if size <= 0 or max_rank <= 0 or requested_iterations <= 0:
    return np.zeros((max(0, size), 0), dtype=float), {
        "ritz_requested_rank": int(max_rank),
        "ritz_basis_rank": 0,
        "ritz_probe_requested_iterations": int(requested_iterations),
        "ritz_probe_iterations": 0,
        "ritz_projected_dimension": 0,
        "ritz_probe_global_reduction_count": 0,
        "ritz_selected_values": [],
    }

  def add_orthogonalized(vector, basis):
    candidate = np.asarray(vector, dtype=float).reshape(-1).copy()
    if len(candidate) != size:
      raise ValueError("Ritz seed vector dimension mismatch")
    for existing in basis:
      candidate -= float(existing @ candidate) * existing
    norm = float(np.linalg.norm(candidate))
    if norm <= tolerance:
      return False
    basis.append(candidate / norm)
    return True

  basis: list[np.ndarray] = []
  for seed in seed_vectors or []:
    if add_orthogonalized(seed, basis):
      break
  if not basis:
    seed = np.ones(size, dtype=float)
    if not add_orthogonalized(seed, basis):
      return np.zeros((size, 0), dtype=float), {
          "ritz_requested_rank": int(max_rank),
          "ritz_basis_rank": 0,
          "ritz_probe_requested_iterations": int(requested_iterations),
          "ritz_probe_iterations": 0,
          "ritz_projected_dimension": 0,
          "ritz_probe_global_reduction_count": 0,
          "ritz_selected_values": [],
      }

  target_dimension = min(size, max(max_rank, requested_iterations))
  operator_columns: list[np.ndarray | None] = []
  matvec_count = 0
  global_reductions = 0
  index = 0
  while (len(basis) < target_dimension and
         matvec_count < requested_iterations and
         index < len(basis)):
    vector = basis[index]
    image = np.asarray(matvec(vector), dtype=float).reshape(-1)
    if len(image) != size:
      raise ValueError("Ritz operator output dimension mismatch")
    operator_columns.append(image.copy())
    matvec_count += 1
    candidate = image.copy()
    for existing in basis:
      candidate -= float(existing @ candidate) * existing
      global_reductions += 1
    norm = float(np.linalg.norm(candidate))
    global_reductions += 1
    if norm > tolerance:
      basis.append(candidate / norm)
    index += 1

  while (len(operator_columns) < len(basis) and
         matvec_count < requested_iterations):
    image = np.asarray(matvec(basis[len(operator_columns)]),
                       dtype=float).reshape(-1)
    if len(image) != size:
      raise ValueError("Ritz operator output dimension mismatch")
    operator_columns.append(image.copy())
    matvec_count += 1

  projected_dimension = min(len(basis), len(operator_columns))
  if projected_dimension <= 0:
    return np.zeros((size, 0), dtype=float), {
        "ritz_requested_rank": int(max_rank),
        "ritz_basis_rank": 0,
        "ritz_probe_requested_iterations": int(requested_iterations),
        "ritz_probe_iterations": int(matvec_count),
        "ritz_projected_dimension": 0,
        "ritz_probe_global_reduction_count": int(global_reductions),
        "ritz_selected_values": [],
    }
  q_matrix = np.column_stack(basis[:projected_dimension])
  aq_matrix = np.column_stack(operator_columns[:projected_dimension])
  projected = q_matrix.T @ aq_matrix
  projected = 0.5 * (projected + projected.T)
  values, vectors = np.linalg.eigh(projected)
  order = sorted(
      range(len(values)),
      key=lambda idx: (abs(float(values[idx])), float(values[idx])))
  selected_vectors = []
  selected_values = []
  for idx in order[:max_rank]:
    selected_vectors.append(q_matrix @ vectors[:, idx])
    selected_values.append(float(values[idx]))
  basis_matrix = _orthonormalize_directions(
      selected_vectors, max_rank=max_rank, tolerance=tolerance)
  return basis_matrix, {
      "ritz_requested_rank": int(max_rank),
      "ritz_basis_rank": int(basis_matrix.shape[1]),
      "ritz_probe_requested_iterations": int(requested_iterations),
      "ritz_probe_iterations": int(matvec_count),
      "ritz_projected_dimension": int(projected_dimension),
      "ritz_probe_global_reduction_count": int(global_reductions),
      "ritz_selected_values": selected_values[:int(basis_matrix.shape[1])],
  }


def _harmonic_ritz_low_mode_basis_from_operator(
    matvec,
    size: int,
    max_rank: int,
    probe_iterations: int,
    seed_vectors: list[np.ndarray] | None = None,
    tolerance: float = 1e-12):
  """Approximates near-zero harmonic Ritz modes of a linear operator."""
  size = int(size)
  max_rank = max(0, int(max_rank))
  requested_iterations = max(0, int(probe_iterations))
  empty_stats = {
      "ritz_mode": "harmonic",
      "ritz_requested_rank": int(max_rank),
      "ritz_basis_rank": 0,
      "ritz_probe_requested_iterations": int(requested_iterations),
      "ritz_probe_iterations": 0,
      "ritz_projected_dimension": 0,
      "ritz_probe_global_reduction_count": 0,
      "ritz_selected_values": [],
  }
  if size <= 0 or max_rank <= 0 or requested_iterations <= 0:
    return np.zeros((max(0, size), 0), dtype=float), empty_stats

  def add_orthogonalized(vector, basis):
    candidate = np.asarray(vector, dtype=float).reshape(-1).copy()
    if len(candidate) != size:
      raise ValueError("harmonic Ritz seed vector dimension mismatch")
    for existing in basis:
      candidate -= float(existing @ candidate) * existing
    norm = float(np.linalg.norm(candidate))
    if norm <= tolerance:
      return False
    basis.append(candidate / norm)
    return True

  basis: list[np.ndarray] = []
  for seed in seed_vectors or []:
    if add_orthogonalized(seed, basis):
      break
  if not basis:
    if not add_orthogonalized(np.ones(size, dtype=float), basis):
      return np.zeros((size, 0), dtype=float), empty_stats

  target_dimension = min(size, max(max_rank, requested_iterations))
  operator_columns: list[np.ndarray | None] = []
  matvec_count = 0
  global_reductions = 0
  index = 0
  while (len(basis) < target_dimension and
         matvec_count < requested_iterations and
         index < len(basis)):
    vector = basis[index]
    image = np.asarray(matvec(vector), dtype=float).reshape(-1)
    if len(image) != size:
      raise ValueError("harmonic Ritz operator output dimension mismatch")
    operator_columns.append(image.copy())
    matvec_count += 1
    candidate = image.copy()
    for existing in basis:
      candidate -= float(existing @ candidate) * existing
      global_reductions += 1
    norm = float(np.linalg.norm(candidate))
    global_reductions += 1
    if norm > tolerance:
      basis.append(candidate / norm)
    index += 1

  while (len(operator_columns) < len(basis) and
         matvec_count < requested_iterations):
    image = np.asarray(matvec(basis[len(operator_columns)]),
                       dtype=float).reshape(-1)
    if len(image) != size:
      raise ValueError("harmonic Ritz operator output dimension mismatch")
    operator_columns.append(image.copy())
    matvec_count += 1

  projected_dimension = min(len(basis), len(operator_columns))
  if projected_dimension <= 0:
    stats = {**empty_stats, **{
        "ritz_probe_iterations": int(matvec_count),
        "ritz_probe_global_reduction_count": int(global_reductions),
    }}
    return np.zeros((size, 0), dtype=float), stats

  q_matrix = np.column_stack(basis[:projected_dimension])
  aq_matrix = np.column_stack(operator_columns[:projected_dimension])
  normal_projected = aq_matrix.T @ aq_matrix
  normal_projected = 0.5 * (normal_projected + normal_projected.T)
  cross_projected = aq_matrix.T @ q_matrix
  cross_projected = 0.5 * (cross_projected + cross_projected.T)
  cross_values, cross_vectors = np.linalg.eigh(cross_projected)
  positive = np.asarray([
      idx for idx, value in enumerate(cross_values)
      if float(abs(value)) > tolerance
  ], dtype=int)
  if positive.size == 0:
    stats = {**empty_stats, **{
        "ritz_probe_iterations": int(matvec_count),
        "ritz_projected_dimension": int(projected_dimension),
        "ritz_probe_global_reduction_count": int(global_reductions),
    }}
    return np.zeros((size, 0), dtype=float), stats
  inverse_sqrt_abs = (
      cross_vectors[:, positive] @
      np.diag(1.0 / np.sqrt(np.abs(cross_values[positive]))) @
      cross_vectors[:, positive].T)
  standard_projected = (
      inverse_sqrt_abs @ normal_projected @ inverse_sqrt_abs)
  standard_projected = 0.5 * (standard_projected + standard_projected.T)
  values, vectors = np.linalg.eigh(standard_projected)
  order = sorted(
      range(len(values)),
      key=lambda idx: (abs(float(values[idx])), float(values[idx])))
  selected_vectors = []
  selected_values = []
  for idx in order[:max_rank]:
    coeff = inverse_sqrt_abs @ vectors[:, idx]
    selected_vectors.append(q_matrix @ coeff)
    selected_values.append(float(values[idx]))
  basis_matrix = _orthonormalize_directions(
      selected_vectors, max_rank=max_rank, tolerance=tolerance)
  return basis_matrix, {
      "ritz_mode": "harmonic",
      "ritz_requested_rank": int(max_rank),
      "ritz_basis_rank": int(basis_matrix.shape[1]),
      "ritz_probe_requested_iterations": int(requested_iterations),
      "ritz_probe_iterations": int(matvec_count),
      "ritz_projected_dimension": int(projected_dimension),
      "ritz_probe_global_reduction_count": int(global_reductions),
      "ritz_selected_values": selected_values[:int(basis_matrix.shape[1])],
  }


def _generalized_ritz_low_mode_basis_from_operators(
    stiffness_matvec,
    mass_matvec,
    probe_matvec,
    size: int,
    max_rank: int,
    probe_iterations: int,
    seed_vectors: list[np.ndarray] | None = None,
    tolerance: float = 1e-12):
  """Approximates low generalized Ritz modes S z = lambda M z."""
  size = int(size)
  max_rank = max(0, int(max_rank))
  requested_iterations = max(0, int(probe_iterations))
  empty_stats = {
      "ritz_mode": "generalized",
      "ritz_requested_rank": int(max_rank),
      "ritz_basis_rank": 0,
      "ritz_probe_requested_iterations": int(requested_iterations),
      "ritz_probe_iterations": 0,
      "ritz_projected_dimension": 0,
      "ritz_probe_global_reduction_count": 0,
      "ritz_selected_values": [],
  }
  if size <= 0 or max_rank <= 0 or requested_iterations <= 0:
    return np.zeros((max(0, size), 0), dtype=float), empty_stats

  def add_orthogonalized(vector, basis):
    candidate = np.asarray(vector, dtype=float).reshape(-1).copy()
    if len(candidate) != size:
      raise ValueError("generalized Ritz seed vector dimension mismatch")
    for existing in basis:
      candidate -= float(existing @ candidate) * existing
    norm = float(np.linalg.norm(candidate))
    if norm <= tolerance:
      return False
    basis.append(candidate / norm)
    return True

  basis: list[np.ndarray] = []
  for seed in seed_vectors or []:
    if add_orthogonalized(seed, basis):
      break
  if not basis:
    if not add_orthogonalized(np.ones(size, dtype=float), basis):
      return np.zeros((size, 0), dtype=float), empty_stats

  target_dimension = min(size, max(max_rank, requested_iterations))
  probe_columns: list[np.ndarray | None] = []
  matvec_count = 0
  global_reductions = 0
  index = 0
  while (len(basis) < target_dimension and
         matvec_count < requested_iterations and
         index < len(basis)):
    vector = basis[index]
    image = np.asarray(probe_matvec(vector), dtype=float).reshape(-1)
    if len(image) != size:
      raise ValueError("generalized Ritz probe output dimension mismatch")
    probe_columns.append(image.copy())
    matvec_count += 1
    candidate = image.copy()
    for existing in basis:
      candidate -= float(existing @ candidate) * existing
      global_reductions += 1
    norm = float(np.linalg.norm(candidate))
    global_reductions += 1
    if norm > tolerance:
      basis.append(candidate / norm)
    index += 1

  while (len(probe_columns) < len(basis) and
         matvec_count < requested_iterations):
    image = np.asarray(probe_matvec(basis[len(probe_columns)]),
                       dtype=float).reshape(-1)
    if len(image) != size:
      raise ValueError("generalized Ritz probe output dimension mismatch")
    probe_columns.append(image.copy())
    matvec_count += 1

  projected_dimension = min(len(basis), max(1, matvec_count))
  if projected_dimension <= 0:
    stats = {**empty_stats, **{
        "ritz_probe_iterations": int(matvec_count),
        "ritz_probe_global_reduction_count": int(global_reductions),
    }}
    return np.zeros((size, 0), dtype=float), stats

  q_matrix = np.column_stack(basis[:projected_dimension])
  s_columns = np.column_stack([
      np.asarray(stiffness_matvec(q_matrix[:, i]), dtype=float).reshape(-1)
      for i in range(projected_dimension)
  ])
  m_columns = np.column_stack([
      np.asarray(mass_matvec(q_matrix[:, i]), dtype=float).reshape(-1)
      for i in range(projected_dimension)
  ])
  s_projected = 0.5 * (q_matrix.T @ s_columns + s_columns.T @ q_matrix)
  m_projected = 0.5 * (q_matrix.T @ m_columns + m_columns.T @ q_matrix)
  m_values, m_vectors = np.linalg.eigh(m_projected)
  positive = np.asarray([
      idx for idx, value in enumerate(m_values)
      if float(value) > tolerance
  ], dtype=int)
  if positive.size == 0:
    stats = {**empty_stats, **{
        "ritz_probe_iterations": int(matvec_count),
        "ritz_projected_dimension": int(projected_dimension),
        "ritz_probe_global_reduction_count": int(global_reductions),
    }}
    return np.zeros((size, 0), dtype=float), stats
  m_inverse_sqrt = (
      m_vectors[:, positive] @
      np.diag(1.0 / np.sqrt(m_values[positive])) @
      m_vectors[:, positive].T)
  standard_projected = m_inverse_sqrt @ s_projected @ m_inverse_sqrt
  standard_projected = 0.5 * (
      standard_projected + standard_projected.T)
  values, vectors = np.linalg.eigh(standard_projected)
  order = sorted(
      range(len(values)),
      key=lambda idx: (abs(float(values[idx])), float(values[idx])))
  selected_vectors = []
  selected_values = []
  for idx in order[:max_rank]:
    coeff = m_inverse_sqrt @ vectors[:, idx]
    selected_vectors.append(q_matrix @ coeff)
    selected_values.append(float(values[idx]))
  basis_matrix = _orthonormalize_directions(
      selected_vectors, max_rank=max_rank, tolerance=tolerance)
  return basis_matrix, {
      "ritz_mode": "generalized",
      "ritz_requested_rank": int(max_rank),
      "ritz_basis_rank": int(basis_matrix.shape[1]),
      "ritz_probe_requested_iterations": int(requested_iterations),
      "ritz_probe_iterations": int(matvec_count),
      "ritz_projected_dimension": int(projected_dimension),
      "ritz_probe_global_reduction_count": int(global_reductions),
      "ritz_selected_values": selected_values[:int(basis_matrix.shape[1])],
  }


def _m_orthogonal_lanczos_ritz_low_mode_basis_from_operators(
    stiffness_matvec,
    mass_matvec,
    preconditioned_matvec,
    size: int,
    max_rank: int,
    probe_iterations: int,
    seed_vectors: list[np.ndarray] | None = None,
    tolerance: float = 1e-12):
  """Approximates low generalized Ritz modes with an M-orthogonal Lanczos probe."""
  size = int(size)
  max_rank = max(0, int(max_rank))
  requested_iterations = max(0, int(probe_iterations))
  empty_stats = {
      "ritz_mode": "m_orthogonal_lanczos",
      "ritz_requested_rank": int(max_rank),
      "ritz_basis_rank": 0,
      "ritz_probe_requested_iterations": int(requested_iterations),
      "ritz_probe_iterations": 0,
      "ritz_projected_dimension": 0,
      "ritz_probe_global_reduction_count": 0,
      "ritz_lanczos_m_orthogonality_error": 0.0,
      "ritz_selected_values": [],
  }
  if size <= 0 or max_rank <= 0 or requested_iterations <= 0:
    return np.zeros((max(0, size), 0), dtype=float), empty_stats

  def m_inner(left, right):
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    return float(left @ np.asarray(mass_matvec(right), dtype=float).reshape(-1))

  def add_m_orthogonalized(vector, basis):
    candidate = np.asarray(vector, dtype=float).reshape(-1).copy()
    if len(candidate) != size:
      raise ValueError("M-orthogonal Lanczos seed vector dimension mismatch")
    reduction_count = 0
    for existing in basis:
      candidate -= m_inner(existing, candidate) * existing
      reduction_count += 1
    norm_sq = m_inner(candidate, candidate)
    reduction_count += 1
    if norm_sq <= tolerance * tolerance:
      return False, reduction_count
    basis.append(candidate / np.sqrt(max(norm_sq, 0.0)))
    return True, reduction_count

  basis: list[np.ndarray] = []
  global_reductions = 0
  for seed in seed_vectors or []:
    added, reductions = add_m_orthogonalized(seed, basis)
    global_reductions += reductions
    if added:
      break
  if not basis:
    added, reductions = add_m_orthogonalized(np.ones(size, dtype=float), basis)
    global_reductions += reductions
    if not added:
      stats = {**empty_stats, **{
          "ritz_probe_global_reduction_count": int(global_reductions),
      }}
      return np.zeros((size, 0), dtype=float), stats

  target_dimension = min(size, max(max_rank, requested_iterations))
  matvec_count = 0
  previous = np.zeros(size, dtype=float)
  previous_beta = 0.0
  index = 0
  while (index < len(basis) and len(basis) < target_dimension and
         matvec_count < requested_iterations):
    current = basis[index]
    image = np.asarray(preconditioned_matvec(current), dtype=float).reshape(-1)
    if len(image) != size:
      raise ValueError("M-orthogonal Lanczos probe output dimension mismatch")
    matvec_count += 1
    alpha = m_inner(current, image)
    global_reductions += 1
    candidate = image - alpha * current
    if index > 0:
      candidate -= previous_beta * previous
    for existing in basis:
      candidate -= m_inner(existing, candidate) * existing
      global_reductions += 1
    beta_sq = m_inner(candidate, candidate)
    global_reductions += 1
    if beta_sq > tolerance * tolerance:
      previous = current
      previous_beta = np.sqrt(max(beta_sq, 0.0))
      basis.append(candidate / previous_beta)
    index += 1

  while (matvec_count < requested_iterations and
         matvec_count < len(basis)):
    image = np.asarray(preconditioned_matvec(basis[matvec_count]),
                       dtype=float).reshape(-1)
    if len(image) != size:
      raise ValueError("M-orthogonal Lanczos probe output dimension mismatch")
    matvec_count += 1

  projected_dimension = min(len(basis), max(1, matvec_count))
  if projected_dimension <= 0:
    stats = {**empty_stats, **{
        "ritz_probe_iterations": int(matvec_count),
        "ritz_probe_global_reduction_count": int(global_reductions),
    }}
    return np.zeros((size, 0), dtype=float), stats

  q_matrix = np.column_stack(basis[:projected_dimension])
  s_columns = np.column_stack([
      np.asarray(stiffness_matvec(q_matrix[:, i]), dtype=float).reshape(-1)
      for i in range(projected_dimension)
  ])
  projected = 0.5 * (q_matrix.T @ s_columns + s_columns.T @ q_matrix)
  values, vectors = np.linalg.eigh(projected)
  order = sorted(
      range(len(values)),
      key=lambda idx: (abs(float(values[idx])), float(values[idx])))
  selected_vectors = []
  selected_values = []
  for idx in order[:max_rank]:
    selected_vectors.append(q_matrix @ vectors[:, idx])
    selected_values.append(float(values[idx]))
  basis_matrix = _orthonormalize_directions(
      selected_vectors, max_rank=max_rank, tolerance=tolerance)
  m_columns = np.column_stack([
      np.asarray(mass_matvec(q_matrix[:, i]), dtype=float).reshape(-1)
      for i in range(projected_dimension)
  ])
  m_gram = q_matrix.T @ m_columns
  orthogonality_error = float(
      np.linalg.norm(m_gram - np.eye(projected_dimension), ord="fro"))
  return basis_matrix, {
      "ritz_mode": "m_orthogonal_lanczos",
      "ritz_requested_rank": int(max_rank),
      "ritz_basis_rank": int(basis_matrix.shape[1]),
      "ritz_probe_requested_iterations": int(requested_iterations),
      "ritz_probe_iterations": int(matvec_count),
      "ritz_projected_dimension": int(projected_dimension),
      "ritz_probe_global_reduction_count": int(global_reductions),
      "ritz_lanczos_m_orthogonality_error": orthogonality_error,
      "ritz_selected_values": selected_values[:int(basis_matrix.shape[1])],
  }


def _coarse_basis_quadratic_merit(basis: np.ndarray,
                                  stiffness_matvec,
                                  rhs: np.ndarray,
                                  damping: float,
                                  tolerance: float = 1e-12):
  """Returns projected quadratic decrease for a Schur coarse basis."""
  rhs = np.asarray(rhs, dtype=float).reshape(-1)
  basis_array = np.asarray(basis, dtype=float)
  if basis_array.size == 0:
    return {
        "merit": 0.0,
        "residual_norm": float(np.linalg.norm(rhs)),
        "basis_rank": 0,
        "schur_matvec_count": 0,
        "operator_condition": 0.0,
    }
  if basis_array.ndim == 1:
    basis_array = basis_array.reshape(-1, 1)
  if basis_array.ndim != 2 or basis_array.shape[0] != len(rhs):
    raise ValueError("coarse merit basis row count mismatch")
  basis_array = _orthonormalize_directions(
      [basis_array[:, col] for col in range(basis_array.shape[1])],
      max_rank=basis_array.shape[1],
      tolerance=tolerance)
  rank = int(basis_array.shape[1])
  if rank == 0:
    return {
        "merit": 0.0,
        "residual_norm": float(np.linalg.norm(rhs)),
        "basis_rank": 0,
        "schur_matvec_count": 0,
        "operator_condition": 0.0,
    }
  schur_columns = np.column_stack([
      np.asarray(stiffness_matvec(basis_array[:, col]), dtype=float).reshape(-1)
      for col in range(rank)
  ])
  projected = basis_array.T @ schur_columns
  projected = 0.5 * (projected + projected.T)
  if damping > 0.0:
    projected = projected + damping * np.eye(rank)
  rhs_projected = basis_array.T @ rhs
  condition = float(np.linalg.cond(projected))
  if np.isfinite(condition) and condition < 1e12:
    coeff = np.linalg.solve(projected, rhs_projected)
  else:
    coeff = np.linalg.pinv(projected, rcond=1e-12) @ rhs_projected
  coarse_solution = basis_array @ coeff
  merit = float(
      rhs_projected @ coeff - 0.5 * coeff @ projected @ coeff)
  residual_norm = float(np.linalg.norm(schur_columns @ coeff - rhs))
  return {
      "merit": merit,
      "residual_norm": residual_norm,
      "basis_rank": rank,
      "schur_matvec_count": int(rank),
      "operator_condition": condition,
  }


def _select_ritz_portfolio_basis(candidates,
                                 stiffness_matvec,
                                 rhs: np.ndarray,
                                 base_coarse_basis: np.ndarray | None,
                                 max_rank: int,
                                 damping: float,
                                 tolerance: float = 1e-12):
  """Selects a Ritz candidate by projected Schur quadratic decrease."""
  rhs = np.asarray(rhs, dtype=float).reshape(-1)
  size = len(rhs)
  max_rank = max(0, int(max_rank))
  if base_coarse_basis is None:
    base_basis = np.zeros((size, 0), dtype=float)
  else:
    base_basis = np.asarray(base_coarse_basis, dtype=float)
    if base_basis.ndim == 1:
      base_basis = base_basis.reshape(-1, 1)
    if base_basis.ndim != 2 or base_basis.shape[0] != size:
      raise ValueError("base coarse basis row count mismatch")
  base_basis = _combine_interface_coarse_bases([base_basis], size)
  base_merit = _coarse_basis_quadratic_merit(
      base_basis, stiffness_matvec, rhs, damping, tolerance)
  best_basis = np.zeros((size, 0), dtype=float)
  best_score = -math.inf
  best_mode = ""
  candidate_records = []
  scoring_matvec_count = int(base_merit["schur_matvec_count"])
  total_probe_iterations = 0
  total_requested_iterations = 0
  total_projected_dimension = 0
  total_probe_reductions = 0
  max_lanczos_orthogonality_error = 0.0

  for order, (mode, candidate_basis, candidate_stats) in enumerate(candidates):
    candidate_basis = np.asarray(candidate_basis, dtype=float)
    if candidate_basis.size == 0:
      candidate_basis = np.zeros((size, 0), dtype=float)
    elif candidate_basis.ndim == 1:
      candidate_basis = candidate_basis.reshape(-1, 1)
    if candidate_basis.ndim != 2 or candidate_basis.shape[0] != size:
      raise ValueError("Ritz portfolio candidate row count mismatch")
    candidate_basis = _orthonormalize_directions(
        [candidate_basis[:, col] for col in range(candidate_basis.shape[1])],
        max_rank=max_rank,
        tolerance=tolerance)
    combined = _combine_interface_coarse_bases(
        [base_basis, candidate_basis], size)
    merit = _coarse_basis_quadratic_merit(
        combined, stiffness_matvec, rhs, damping, tolerance)
    incremental_merit = float(merit["merit"] - base_merit["merit"])
    scoring_matvec_count += int(merit["schur_matvec_count"])
    total_probe_iterations += int(candidate_stats.get(
        "ritz_probe_iterations", 0))
    total_requested_iterations += int(candidate_stats.get(
        "ritz_probe_requested_iterations", 0))
    total_projected_dimension += int(candidate_stats.get(
        "ritz_projected_dimension", 0))
    total_probe_reductions += int(candidate_stats.get(
        "ritz_probe_global_reduction_count", 0))
    max_lanczos_orthogonality_error = max(
        max_lanczos_orthogonality_error,
        float(candidate_stats.get("ritz_lanczos_m_orthogonality_error", 0.0)))
    candidate_records.append({
        "mode": mode,
        "basis_rank": int(candidate_basis.shape[1]),
        "total_merit": float(merit["merit"]),
        "incremental_merit": incremental_merit,
        "residual_norm": float(merit["residual_norm"]),
        "operator_condition": float(merit["operator_condition"]),
        "ritz_selected_values": [
            float(value)
            for value in candidate_stats.get("ritz_selected_values", [])
        ],
    })
    if (candidate_basis.shape[1] > 0 and
        (incremental_merit > best_score + tolerance or
         (abs(incremental_merit - best_score) <= tolerance and
          (best_mode == "" or order == 0)))):
      best_score = incremental_merit
      best_basis = candidate_basis
      best_mode = mode

  if best_mode == "":
    best_score = 0.0
  return best_basis, {
      "ritz_mode": "portfolio",
      "ritz_requested_rank": int(max_rank),
      "ritz_basis_rank": int(best_basis.shape[1]),
      "ritz_probe_requested_iterations": int(total_requested_iterations),
      "ritz_probe_iterations": int(total_probe_iterations),
      "ritz_projected_dimension": int(total_projected_dimension),
      "ritz_probe_global_reduction_count": int(total_probe_reductions),
      "ritz_selected_values": [],
      "ritz_portfolio_selected_mode": best_mode,
      "ritz_portfolio_selected_incremental_merit": float(best_score),
      "ritz_portfolio_base_merit": float(base_merit["merit"]),
      "ritz_portfolio_base_residual_norm": float(base_merit["residual_norm"]),
      "ritz_portfolio_candidate_count": int(len(candidate_records)),
      "ritz_portfolio_scoring_schur_matvec_count": int(scoring_matvec_count),
      "ritz_portfolio_candidates": candidate_records,
      "ritz_lanczos_m_orthogonality_error": float(
          max_lanczos_orthogonality_error),
  }


def _apply_ritz_rank_selection(basis: np.ndarray,
                               stats: dict,
                               mode: str = "fixed",
                               value_threshold: float | None = None,
                               energy_capture_fraction: float | None = None,
                               min_rank: int = 1):
  """Selects an effective Ritz rank without re-running the probe."""
  basis_array = np.asarray(basis, dtype=float)
  if basis_array.size == 0:
    if basis_array.ndim == 1:
      basis_array = basis_array.reshape(-1, 0)
    requested_rank = 0
  else:
    if basis_array.ndim == 1:
      basis_array = basis_array.reshape(-1, 1)
    if basis_array.ndim != 2:
      raise ValueError("Ritz basis must be a two-dimensional array")
    requested_rank = int(basis_array.shape[1])
  selected_stats = dict(stats)
  if mode not in {"fixed", "value_threshold", "energy_capture"}:
    raise ValueError(f"unsupported Ritz rank selection mode: {mode}")
  min_rank = max(0, int(min_rank))
  selected_rank = requested_rank
  if mode == "value_threshold":
    if value_threshold is None or not np.isfinite(float(value_threshold)):
      raise ValueError(
          "ritz value_threshold rank selection requires a finite threshold")
    values = [
        float(value)
        for value in selected_stats.get("ritz_selected_values", [])
    ]
    if len(values) < requested_rank:
      selected_rank = requested_rank
    else:
      selected_rank = sum(
          1 for value in values[:requested_rank]
          if value <= float(value_threshold))
      selected_rank = max(min_rank, selected_rank)
      selected_rank = min(requested_rank, selected_rank)
  elif mode == "energy_capture":
    if (energy_capture_fraction is None or
        not np.isfinite(float(energy_capture_fraction))):
      raise ValueError(
          "ritz energy_capture rank selection requires a finite fraction")
    fraction = float(energy_capture_fraction)
    if fraction < 0.0 or fraction > 1.0:
      raise ValueError("ritz energy_capture fraction must be in [0, 1]")
    contributions = [
        max(0.0, float(value))
        for value in selected_stats.get("ritz_energy_contributions", [])
    ]
    if len(contributions) < requested_rank:
      selected_rank = requested_rank
    else:
      total_energy = float(sum(contributions[:requested_rank]))
      target_energy = fraction * total_energy
      cumulative_energy = 0.0
      selected_rank = requested_rank
      for index, energy in enumerate(contributions[:requested_rank], start=1):
        cumulative_energy += energy
        if cumulative_energy + 1e-15 >= target_energy:
          selected_rank = index
          break
      selected_rank = max(min_rank, selected_rank)
      selected_rank = min(requested_rank, selected_rank)
  selected_basis = basis_array[:, :selected_rank]
  selected_stats["ritz_deflation_requested_rank_before_selection"] = (
      int(requested_rank))
  selected_stats["ritz_deflation_selected_rank"] = int(selected_rank)
  selected_stats["ritz_rank_selection_mode"] = mode
  selected_stats["ritz_value_threshold"] = (
      None if value_threshold is None else float(value_threshold))
  selected_stats["ritz_energy_capture_fraction"] = (
      None if energy_capture_fraction is None
      else float(energy_capture_fraction))
  selected_stats["ritz_basis_rank"] = int(selected_rank)
  selected_stats["ritz_selected_values"] = [
      float(value)
      for value in selected_stats.get("ritz_selected_values", [])
  ][:selected_rank]
  contributions = [
      max(0.0, float(value))
      for value in selected_stats.get("ritz_energy_contributions", [])
  ]
  selected_stats["ritz_energy_capture_total"] = float(
      sum(contributions[:requested_rank]))
  selected_stats["ritz_energy_capture_selected"] = float(
      sum(contributions[:selected_rank]))
  selected_stats["ritz_energy_contributions"] = (
      contributions[:selected_rank])
  return selected_basis, selected_stats


def _build_interface_coarse_basis(offsets: dict[int, int],
                                  block_dim: int,
                                  interface_pose_ids: set[int],
                                  robot_of: dict[int, int],
                                  interface_indices: np.ndarray,
                                  separator_summary: dict,
                                  mode: str,
                                  component_limit: int | None,
                                  component_selection_mode: str = "size"):
  if mode not in {
      "robot_coordinate",
      "component_coordinate",
      "component_endpoint_coordinate",
      "robot_component_coordinate",
  }:
    raise ValueError(f"unsupported interface coarse basis mode: {mode}")
  robot_basis = np.zeros((len(interface_indices), 0), dtype=float)
  component_basis = np.zeros((len(interface_indices), 0), dtype=float)
  endpoint_basis = np.zeros((len(interface_indices), 0), dtype=float)
  if mode in {"robot_coordinate", "robot_component_coordinate"}:
    robot_basis = _interface_robot_coordinate_coarse_basis(
        offsets, block_dim, interface_pose_ids, robot_of, interface_indices)
  if mode in {"component_coordinate", "robot_component_coordinate"}:
    component_basis = _interface_component_coordinate_coarse_basis(
        offsets=offsets,
        block_dim=block_dim,
        interface_pose_ids=interface_pose_ids,
        interface_indices=interface_indices,
        separator_summary=separator_summary,
        component_limit=component_limit,
        component_selection_mode=component_selection_mode)
  if mode == "component_endpoint_coordinate":
    endpoint_basis = _interface_component_endpoint_coordinate_coarse_basis(
        offsets=offsets,
        block_dim=block_dim,
        interface_pose_ids=interface_pose_ids,
        interface_indices=interface_indices,
        separator_summary=separator_summary,
        component_limit=component_limit,
        component_selection_mode=component_selection_mode)
  if mode == "robot_coordinate":
    basis = robot_basis
  elif mode == "component_coordinate":
    basis = component_basis
  elif mode == "component_endpoint_coordinate":
    basis = endpoint_basis
  else:
    basis = _combine_interface_coarse_bases(
        [robot_basis, component_basis], len(interface_indices))
  return basis, {
      "coarse_basis_mode": mode,
      "coarse_component_limit": (
          None if component_limit is None else int(component_limit)),
      "coarse_component_selection_mode": component_selection_mode,
      "robot_coordinate_basis_rank": int(robot_basis.shape[1]),
      "component_coordinate_basis_rank": int(component_basis.shape[1]),
      "endpoint_coordinate_basis_rank": int(endpoint_basis.shape[1]),
  }


def interface_schur_hessian_solve(hessian,
                                  gradient: np.ndarray,
                                  interface_indices: np.ndarray,
                                  iterations: int,
                                  damping: float = 1e-12,
                                  tolerance: float = 1e-10,
                                  interface_blocks: list[np.ndarray] | None = None,
                                  schur_preconditioner: str = "diagonal",
                                  coarse_basis: np.ndarray | None = None,
                                  use_coarse_initial_guess: bool = False,
                                  residual_deflation_rank: int = 0,
                                  residual_deflation_pilot_iterations:
                                  int | None = None,
                                  ritz_deflation_rank: int = 0,
                                  ritz_probe_iterations: int | None = None,
                                  ritz_mode: str = "preconditioned_operator",
                                  ritz_rank_selection_mode: str = "fixed",
                                  ritz_value_threshold: float | None = None,
                                  ritz_energy_capture_fraction:
                                  float | None = None,
                                  ritz_rank_selection_min_rank: int = 1,
                                  coarse_regularization: float = 0.0):
  """Solves H x = g by eliminating private variables and PCG-solving interface."""
  if hessian is None or identity is None or splu is None:
    raise RuntimeError("scipy sparse support is required")
  hessian = _as_sparse_hessian(hessian)
  gradient = np.asarray(gradient, dtype=float).reshape(-1)
  variable_count = int(hessian.shape[0])
  if hessian.shape[0] != hessian.shape[1]:
    raise ValueError("hessian must be square")
  if len(gradient) != variable_count:
    raise ValueError("gradient dimension does not match hessian")
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  if interface_indices.size:
    if int(np.min(interface_indices)) < 0 or int(np.max(interface_indices)) >= variable_count:
      raise ValueError("interface index outside hessian dimension")
  interface_indices = np.asarray(sorted(set(int(i) for i in interface_indices)),
                                 dtype=int)
  interface_mask = np.zeros(variable_count, dtype=bool)
  interface_mask[interface_indices] = True
  private_indices = np.nonzero(~interface_mask)[0].astype(int)

  if variable_count == 0:
    return np.zeros(0), {
        "iterations": 0,
        "final_normal_residual": 0.0,
        "method": "interface_schur_pcg",
        "interface_variable_count": 0,
        "private_variable_count": 0,
    }
  if len(interface_indices) == 0:
    system = hessian
    if damping > 0.0:
      system = system + damping * identity(variable_count, format="csc")
    solution = splu(system.tocsc()).solve(gradient)
    final_residual = float(np.linalg.norm(hessian @ solution - gradient))
    return solution, {
        "iterations": 0,
        "schur_iterations": 0,
        "final_normal_residual": final_residual,
        "final_schur_residual": 0.0,
        "method": "interface_schur_pcg",
        "interface_variable_count": 0,
        "private_variable_count": variable_count,
        "private_factor_damping": damping,
        "global_reduction_count": 0,
    }
  if len(private_indices) == 0:
    solution, stats = block_pcg_hessian_solve(
        hessian, gradient, {0: interface_indices}, iterations, damping,
        tolerance)
    stats = {
        **stats,
        "method": "interface_schur_pcg",
        "base_method": stats.get("method"),
        "schur_iterations": stats.get("iterations", 0),
        "interface_variable_count": len(interface_indices),
        "private_variable_count": 0,
        "private_factor_damping": damping,
        "schur_preconditioner": "full_block",
        "schur_preconditioner_block_count": 1,
    }
    return solution, stats

  h_pp = hessian[private_indices[:, None], private_indices].tocsc()
  if damping > 0.0:
    h_pp = h_pp + damping * identity(h_pp.shape[0], format="csc")
  h_pb = hessian[private_indices[:, None], interface_indices].tocsc()
  h_bp = hessian[interface_indices[:, None], private_indices].tocsc()
  h_bb = hessian[interface_indices[:, None], interface_indices].tocsc()
  g_p = gradient[private_indices]
  g_b = gradient[interface_indices]
  private_factor = splu(h_pp)
  private_rhs_response = private_factor.solve(g_p)
  schur_rhs = g_b - np.asarray(h_bp @ private_rhs_response).reshape(-1)

  def schur_matvec(vector):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    private_response = private_factor.solve(
        np.asarray(h_pb @ vector).reshape(-1))
    return (np.asarray(h_bb @ vector).reshape(-1) -
            np.asarray(h_bp @ private_response).reshape(-1))

  if schur_preconditioner not in {
      "diagonal",
      "block_jacobi",
      "block_jacobi+coarse",
      "block_jacobi+balanced_coarse",
      "overlap_schwarz",
  }:
    raise ValueError(f"unsupported schur_preconditioner: {schur_preconditioner}")
  if ritz_mode not in {
      "preconditioned_operator",
      "generalized",
      "m_orthogonal_lanczos",
      "harmonic",
      "portfolio",
      "low_harmonic_hybrid",
  }:
    raise ValueError(f"unsupported ritz_mode: {ritz_mode}")
  if ritz_rank_selection_mode not in {
      "fixed",
      "value_threshold",
      "energy_capture",
  }:
    raise ValueError(
        "unsupported Ritz rank selection mode: "
        f"{ritz_rank_selection_mode}")
  if int(ritz_rank_selection_min_rank) < 0:
    raise ValueError("ritz_rank_selection_min_rank must be nonnegative")
  if (ritz_rank_selection_mode == "value_threshold" and
      (ritz_value_threshold is None or
       not np.isfinite(float(ritz_value_threshold)))):
    raise ValueError(
        "ritz value-threshold rank selection requires a finite threshold")
  if ritz_rank_selection_mode == "energy_capture":
    if (ritz_energy_capture_fraction is None or
        not np.isfinite(float(ritz_energy_capture_fraction))):
      raise ValueError(
          "ritz energy-capture rank selection requires a finite fraction")
    if (float(ritz_energy_capture_fraction) < 0.0 or
        float(ritz_energy_capture_fraction) > 1.0):
      raise ValueError(
          "ritz energy-capture fraction must be in [0, 1]")
  coarse_regularization = float(coarse_regularization)
  if coarse_regularization < 0.0:
    raise ValueError("coarse_regularization must be nonnegative")
  residual_deflation_rank = max(0, int(residual_deflation_rank))
  if (residual_deflation_rank > 0 and
      schur_preconditioner not in {
          "block_jacobi+coarse",
          "block_jacobi+balanced_coarse",
      }):
    raise ValueError(
        "residual_deflation_rank requires a coarse block-Jacobi preconditioner")
  ritz_deflation_rank = max(0, int(ritz_deflation_rank))
  if (ritz_deflation_rank > 0 and
      schur_preconditioner not in {
          "block_jacobi+coarse",
          "block_jacobi+balanced_coarse",
      }):
    raise ValueError(
        "ritz_deflation_rank requires a coarse block-Jacobi preconditioner")
  interface_position = {
      int(global_index): local_index
      for local_index, global_index in enumerate(interface_indices)
  }
  local_blocks: list[np.ndarray] = []
  if interface_blocks is not None:
    for block in interface_blocks:
      local = [
          interface_position[int(global_index)]
          for global_index in np.asarray(block, dtype=int).reshape(-1)
          if int(global_index) in interface_position
      ]
      if local:
        local_blocks.append(np.asarray(local, dtype=int))
  if not local_blocks:
    local_blocks = [
        np.asarray([index], dtype=int) for index in range(len(interface_indices))
    ]

  schur_diag = np.maximum(np.abs(h_bb.diagonal()), 1e-12)
  block_factors = []
  overlap_factors = []
  overlap_counts = np.zeros(len(interface_indices), dtype=float)
  overlap_schwarz_setup_schur_matvec_count = 0
  overlap_schwarz_setup_local_schur_column_count = 0
  overlap_schwarz_duplicate_group_count = 0
  overlap_schwarz_overlap_level = 0
  uses_block_jacobi = schur_preconditioner in {
      "block_jacobi",
      "block_jacobi+coarse",
      "block_jacobi+balanced_coarse",
  }
  uses_coarse_space = schur_preconditioner in {
      "block_jacobi+coarse",
      "block_jacobi+balanced_coarse",
  }
  if uses_block_jacobi:
    for block in local_blocks:
      h_bb_block = h_bb[block[:, None], block].tocsc()
      h_pb_block = h_pb[:, block].tocsc()
      h_bp_block = h_bp[block, :].tocsc()
      private_response = private_factor.solve(h_pb_block.toarray())
      schur_block = (
          h_bb_block.toarray() -
          np.asarray(h_bp_block @ private_response, dtype=float))
      if damping > 0.0:
        schur_block = schur_block + damping * np.eye(len(block))
      block_factors.append((block, schur_block, np.linalg.inv(schur_block)))
  if schur_preconditioner == "overlap_schwarz":
    overlap_schwarz_overlap_level = 1
    adjacency = []
    h_bb_abs = abs(h_bb)
    for seed in local_blocks:
      group = set(int(index) for index in seed)
      for candidate in local_blocks:
        candidate_set = set(int(index) for index in candidate)
        connected = False
        for seed_index in seed:
          for candidate_index in candidate:
            if (seed_index == candidate_index or
                h_bb_abs[int(seed_index), int(candidate_index)] != 0 or
                h_bb_abs[int(candidate_index), int(seed_index)] != 0):
              connected = True
              break
          if connected:
            break
        if connected:
          group.update(candidate_set)
      adjacency.append(np.asarray(sorted(group), dtype=int))
    unique_adjacency = []
    seen_groups: set[tuple[int, ...]] = set()
    for group in adjacency:
      key = tuple(int(index) for index in group)
      if key in seen_groups:
        overlap_schwarz_duplicate_group_count += 1
        continue
      seen_groups.add(key)
      unique_adjacency.append(group)
    for group in unique_adjacency:
      if len(group) == 0:
        continue
      h_bb_group = h_bb[group[:, None], group].tocsc()
      h_pb_group = h_pb[:, group].tocsc()
      h_bp_group = h_bp[group, :].tocsc()
      private_response = private_factor.solve(h_pb_group.toarray())
      local_operator = (
          h_bb_group.toarray() -
          np.asarray(h_bp_group @ private_response, dtype=float))
      local_operator = 0.5 * (local_operator + local_operator.T)
      if damping > 0.0:
        local_operator = local_operator + damping * np.eye(len(group))
      condition = float(np.linalg.cond(local_operator))
      if np.isfinite(condition) and condition < 1e12:
        inverse = np.linalg.inv(local_operator)
      else:
        inverse = np.linalg.pinv(local_operator, rcond=1e-12)
      overlap_factors.append((group, inverse))
      overlap_counts[group] += 1.0
      overlap_schwarz_setup_local_schur_column_count += int(len(group))
    overlap_counts = np.maximum(overlap_counts, 1.0)

  def base_preconditioner(residual):
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if schur_preconditioner == "overlap_schwarz":
      output = np.zeros_like(residual)
      for group, inverse in overlap_factors:
        weights = 1.0 / np.sqrt(overlap_counts[group])
        output[group] += weights * (inverse @ (weights * residual[group]))
      return output
    if uses_block_jacobi:
      output = np.zeros_like(residual)
      for block, _, inverse in block_factors:
        output[block] = inverse @ residual[block]
      return output
    return residual / schur_diag

  def base_mass_matvec(vector):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    if uses_block_jacobi:
      output = np.zeros_like(vector)
      for block, schur_block, _ in block_factors:
        output[block] = schur_block @ vector[block]
      return output
    return schur_diag * vector

  coarse_column_count = 0
  coarse_rank = 0
  coarse_operator_condition = 0.0
  coarse_inverse = None
  coarse_space = np.zeros((len(interface_indices), 0), dtype=float)
  coarse_initial_guess = None
  coarse_initial_residual = 0.0
  coarse_initial_guess_norm = 0.0
  residual_deflation_basis_rank = 0
  residual_deflation_pilot_used_iterations = 0
  residual_deflation_pilot_global_reductions = 0
  residual_deflation_pilot_final_residual = 0.0
  ritz_deflation_basis_rank = 0
  ritz_deflation_requested_rank_before_selection = 0
  ritz_deflation_selected_rank = 0
  ritz_probe_used_iterations = 0
  ritz_probe_requested_iterations = 0
  ritz_probe_global_reductions = 0
  ritz_lanczos_m_orthogonality_error = 0.0
  ritz_projected_dimension = 0
  ritz_selected_values: list[float] = []
  ritz_rhs_coefficients: list[float] = []
  ritz_energy_contributions: list[float] = []
  ritz_energy_capture_total = 0.0
  ritz_energy_capture_selected = 0.0
  ritz_portfolio_selected_mode = ""
  ritz_portfolio_selected_incremental_merit = 0.0
  ritz_portfolio_base_merit = 0.0
  ritz_portfolio_base_residual_norm = 0.0
  ritz_portfolio_candidate_count = 0
  ritz_portfolio_scoring_schur_matvec_count = 0
  ritz_portfolio_candidates: list[dict] = []
  ritz_hybrid_candidate_count = 0
  ritz_hybrid_candidate_modes: list[str] = []
  ritz_hybrid_candidate_ranks: list[int] = []
  balanced_coarse_preconditioner_schur_matvec_count = 0
  if coarse_basis is not None:
    coarse_array = np.asarray(coarse_basis, dtype=float)
    if coarse_array.ndim == 1:
      coarse_array = coarse_array.reshape(-1, 1)
    if coarse_array.ndim != 2:
      raise ValueError("coarse_basis must be a two-dimensional array")
    if coarse_array.shape[0] != len(interface_indices):
      raise ValueError(
          "coarse_basis row count must match interface variable count")
    coarse_column_count = int(coarse_array.shape[1])
    coarse_vectors = [coarse_array[:, i] for i in range(coarse_array.shape[1])]
    coarse_space = _orthonormalize_directions(
        coarse_vectors, max_rank=coarse_column_count, tolerance=1e-12)
    coarse_rank = int(coarse_space.shape[1])
  if residual_deflation_rank > 0:
    if residual_deflation_pilot_iterations is None:
      pilot_iterations = min(max(1, residual_deflation_rank),
                             max(1, int(iterations)))
    else:
      pilot_iterations = max(1, int(residual_deflation_pilot_iterations))
    _, pilot_stats = _pcg_solve_from_matvec(
        schur_matvec,
        schur_rhs,
        pilot_iterations,
        tolerance,
        preconditioner=base_preconditioner,
        collect_residual_history=pilot_iterations + 1)
    residual_vectors = pilot_stats.get("residual_history", [])
    residual_basis = _residual_deflation_basis_from_history(
        residual_vectors, residual_deflation_rank)
    residual_deflation_basis_rank = int(residual_basis.shape[1])
    if residual_deflation_basis_rank > 0:
      coarse_space = _combine_interface_coarse_bases(
          [coarse_space, residual_basis], len(interface_indices))
      coarse_rank = int(coarse_space.shape[1])
      coarse_column_count += int(len(residual_vectors))
    residual_deflation_pilot_used_iterations = int(
        pilot_stats.get("iterations", 0))
    residual_deflation_pilot_global_reductions = int(
        pilot_stats.get("global_reduction_count", 0))
    residual_deflation_pilot_final_residual = float(
        pilot_stats.get("final_residual", 0.0))
  if ritz_deflation_rank > 0:
    if ritz_probe_iterations is None:
      requested_probe_iterations = min(
          len(interface_indices),
          max(ritz_deflation_rank + 2, int(iterations)))
    else:
      requested_probe_iterations = max(1, int(ritz_probe_iterations))

    def preconditioned_schur_matvec(vector):
      return base_preconditioner(schur_matvec(vector))

    seed_vectors = [
        base_preconditioner(schur_rhs),
        schur_rhs,
        np.ones(len(interface_indices), dtype=float),
    ]
    if ritz_mode == "portfolio":
      portfolio_candidates = []
      ordinary_basis, ordinary_stats = _ritz_low_mode_basis_from_operator(
          preconditioned_schur_matvec,
          size=len(interface_indices),
          max_rank=ritz_deflation_rank,
          probe_iterations=requested_probe_iterations,
          seed_vectors=seed_vectors)
      portfolio_candidates.append((
          "preconditioned_operator",
          ordinary_basis,
          {**ordinary_stats, "ritz_mode": "preconditioned_operator"}))
      generalized_basis, generalized_stats = (
          _generalized_ritz_low_mode_basis_from_operators(
              stiffness_matvec=schur_matvec,
              mass_matvec=base_mass_matvec,
              probe_matvec=preconditioned_schur_matvec,
              size=len(interface_indices),
              max_rank=ritz_deflation_rank,
              probe_iterations=requested_probe_iterations,
              seed_vectors=seed_vectors))
      portfolio_candidates.append((
          "generalized", generalized_basis, generalized_stats))
      lanczos_basis, lanczos_stats = (
          _m_orthogonal_lanczos_ritz_low_mode_basis_from_operators(
              stiffness_matvec=schur_matvec,
              mass_matvec=base_mass_matvec,
              preconditioned_matvec=preconditioned_schur_matvec,
              size=len(interface_indices),
              max_rank=ritz_deflation_rank,
              probe_iterations=requested_probe_iterations,
              seed_vectors=seed_vectors))
      portfolio_candidates.append((
          "m_orthogonal_lanczos", lanczos_basis, lanczos_stats))
      harmonic_basis, harmonic_stats = (
          _harmonic_ritz_low_mode_basis_from_operator(
              preconditioned_schur_matvec,
              size=len(interface_indices),
              max_rank=ritz_deflation_rank,
              probe_iterations=requested_probe_iterations,
              seed_vectors=seed_vectors))
      portfolio_candidates.append((
          "harmonic", harmonic_basis, harmonic_stats))
      ritz_basis, ritz_stats = _select_ritz_portfolio_basis(
          candidates=portfolio_candidates,
          stiffness_matvec=schur_matvec,
          rhs=schur_rhs,
          base_coarse_basis=coarse_space,
          max_rank=ritz_deflation_rank,
          damping=damping)
    elif ritz_mode == "harmonic":
      ritz_basis, ritz_stats = _harmonic_ritz_low_mode_basis_from_operator(
          preconditioned_schur_matvec,
          size=len(interface_indices),
          max_rank=ritz_deflation_rank,
          probe_iterations=requested_probe_iterations,
          seed_vectors=seed_vectors)
    elif ritz_mode == "low_harmonic_hybrid":
      ordinary_basis, ordinary_stats = _ritz_low_mode_basis_from_operator(
          preconditioned_schur_matvec,
          size=len(interface_indices),
          max_rank=ritz_deflation_rank,
          probe_iterations=requested_probe_iterations,
          seed_vectors=seed_vectors)
      harmonic_basis, harmonic_stats = (
          _harmonic_ritz_low_mode_basis_from_operator(
              preconditioned_schur_matvec,
              size=len(interface_indices),
              max_rank=ritz_deflation_rank,
              probe_iterations=requested_probe_iterations,
              seed_vectors=seed_vectors))
      ritz_basis = _combine_interface_coarse_bases(
          [ordinary_basis, harmonic_basis], len(interface_indices))
      ritz_hybrid_candidate_modes = [
          "preconditioned_operator",
          "harmonic",
      ]
      ritz_hybrid_candidate_ranks = [
          int(ordinary_basis.shape[1]),
          int(harmonic_basis.shape[1]),
      ]
      ritz_hybrid_candidate_count = len(ritz_hybrid_candidate_modes)
      ritz_stats = {
          "ritz_mode": "low_harmonic_hybrid",
          "ritz_probe_iterations": (
              int(ordinary_stats.get("ritz_probe_iterations", 0)) +
              int(harmonic_stats.get("ritz_probe_iterations", 0))),
          "ritz_probe_requested_iterations": int(requested_probe_iterations),
          "ritz_probe_global_reduction_count": (
              int(ordinary_stats.get("ritz_probe_global_reduction_count", 0)) +
              int(harmonic_stats.get("ritz_probe_global_reduction_count", 0))),
          "ritz_projected_dimension": (
              int(ordinary_stats.get("ritz_projected_dimension", 0)) +
              int(harmonic_stats.get("ritz_projected_dimension", 0))),
          "ritz_selected_values": [
              float(value)
              for value in ordinary_stats.get("ritz_selected_values", [])
          ] + [
              float(value)
              for value in harmonic_stats.get("ritz_selected_values", [])
          ],
          "ritz_hybrid_candidate_count": int(ritz_hybrid_candidate_count),
          "ritz_hybrid_candidate_modes": list(ritz_hybrid_candidate_modes),
          "ritz_hybrid_candidate_ranks": list(ritz_hybrid_candidate_ranks),
      }
    elif ritz_mode == "m_orthogonal_lanczos":
      ritz_basis, ritz_stats = (
          _m_orthogonal_lanczos_ritz_low_mode_basis_from_operators(
              stiffness_matvec=schur_matvec,
              mass_matvec=base_mass_matvec,
              preconditioned_matvec=preconditioned_schur_matvec,
              size=len(interface_indices),
              max_rank=ritz_deflation_rank,
              probe_iterations=requested_probe_iterations,
              seed_vectors=seed_vectors))
    elif ritz_mode == "generalized":
      ritz_basis, ritz_stats = _generalized_ritz_low_mode_basis_from_operators(
          stiffness_matvec=schur_matvec,
          mass_matvec=base_mass_matvec,
          probe_matvec=preconditioned_schur_matvec,
          size=len(interface_indices),
          max_rank=ritz_deflation_rank,
          probe_iterations=requested_probe_iterations,
          seed_vectors=seed_vectors)
    else:
      ritz_basis, ritz_stats = _ritz_low_mode_basis_from_operator(
          preconditioned_schur_matvec,
          size=len(interface_indices),
          max_rank=ritz_deflation_rank,
          probe_iterations=requested_probe_iterations,
          seed_vectors=seed_vectors)
    if ritz_basis.size:
      rhs_coefficients = np.asarray(ritz_basis.T @ schur_rhs,
                                    dtype=float).reshape(-1)
      ritz_stats["ritz_rhs_coefficients"] = [
          float(value) for value in rhs_coefficients
      ]
      values = [
          abs(float(value))
          for value in ritz_stats.get("ritz_selected_values", [])
      ]
      energy_contributions = []
      for index, coefficient in enumerate(rhs_coefficients):
        theta = values[index] if index < len(values) else 1.0
        theta = max(theta, 1e-12)
        energy_contributions.append(
            float((float(coefficient) * float(coefficient)) / theta))
      ritz_stats["ritz_energy_contributions"] = energy_contributions
    ritz_basis, ritz_stats = _apply_ritz_rank_selection(
        ritz_basis,
        ritz_stats,
        mode=ritz_rank_selection_mode,
        value_threshold=ritz_value_threshold,
        energy_capture_fraction=ritz_energy_capture_fraction,
        min_rank=ritz_rank_selection_min_rank)
    ritz_deflation_basis_rank = int(ritz_basis.shape[1])
    if ritz_deflation_basis_rank > 0:
      coarse_space = _combine_interface_coarse_bases(
          [coarse_space, ritz_basis], len(interface_indices))
      coarse_rank = int(coarse_space.shape[1])
      coarse_column_count += int(ritz_deflation_basis_rank)
    ritz_probe_used_iterations = int(
        ritz_stats.get("ritz_probe_iterations", 0))
    ritz_probe_requested_iterations = int(
        ritz_stats.get("ritz_probe_requested_iterations", 0))
    ritz_probe_global_reductions = int(
        ritz_stats.get("ritz_probe_global_reduction_count", 0))
    ritz_deflation_requested_rank_before_selection = int(
        ritz_stats.get("ritz_deflation_requested_rank_before_selection",
                       ritz_deflation_basis_rank))
    ritz_deflation_selected_rank = int(
        ritz_stats.get("ritz_deflation_selected_rank",
                       ritz_deflation_basis_rank))
    ritz_projected_dimension = int(
        ritz_stats.get("ritz_projected_dimension", 0))
    ritz_lanczos_m_orthogonality_error = float(
        ritz_stats.get("ritz_lanczos_m_orthogonality_error", 0.0))
    ritz_selected_values = [
        float(value) for value in ritz_stats.get("ritz_selected_values", [])
    ]
    ritz_rhs_coefficients = [
        float(value) for value in ritz_stats.get("ritz_rhs_coefficients", [])
    ][:ritz_deflation_basis_rank]
    ritz_energy_contributions = [
        float(value)
        for value in ritz_stats.get("ritz_energy_contributions", [])
    ][:ritz_deflation_basis_rank]
    ritz_energy_capture_total = float(
        ritz_stats.get("ritz_energy_capture_total", 0.0))
    ritz_energy_capture_selected = float(
        ritz_stats.get("ritz_energy_capture_selected", 0.0))
    ritz_portfolio_selected_mode = str(
        ritz_stats.get("ritz_portfolio_selected_mode", ""))
    ritz_portfolio_selected_incremental_merit = float(
        ritz_stats.get("ritz_portfolio_selected_incremental_merit", 0.0))
    ritz_portfolio_base_merit = float(
        ritz_stats.get("ritz_portfolio_base_merit", 0.0))
    ritz_portfolio_base_residual_norm = float(
        ritz_stats.get("ritz_portfolio_base_residual_norm", 0.0))
    ritz_portfolio_candidate_count = int(
        ritz_stats.get("ritz_portfolio_candidate_count", 0))
    ritz_portfolio_scoring_schur_matvec_count = int(
        ritz_stats.get("ritz_portfolio_scoring_schur_matvec_count", 0))
    ritz_portfolio_candidates = list(
        ritz_stats.get("ritz_portfolio_candidates", []))
    ritz_hybrid_candidate_count = int(
        ritz_stats.get("ritz_hybrid_candidate_count",
                       ritz_hybrid_candidate_count))
    ritz_hybrid_candidate_modes = [
        str(mode) for mode in ritz_stats.get(
            "ritz_hybrid_candidate_modes", ritz_hybrid_candidate_modes)
    ]
    ritz_hybrid_candidate_ranks = [
        int(rank) for rank in ritz_stats.get(
            "ritz_hybrid_candidate_ranks", ritz_hybrid_candidate_ranks)
    ]
  if uses_coarse_space and coarse_rank > 0:
    schur_coarse_columns = np.column_stack([
        schur_matvec(coarse_space[:, i]) for i in range(coarse_rank)
    ])
    coarse_operator = coarse_space.T @ schur_coarse_columns
    if damping > 0.0:
      coarse_operator = coarse_operator + damping * np.eye(coarse_rank)
    if coarse_regularization > 0.0:
      coarse_operator = (
          coarse_operator + coarse_regularization * np.eye(coarse_rank))
    coarse_operator_condition = float(np.linalg.cond(coarse_operator))
    if np.isfinite(coarse_operator_condition) and coarse_operator_condition < 1e12:
      coarse_inverse = np.linalg.inv(coarse_operator)
    else:
      coarse_inverse = np.linalg.pinv(coarse_operator, rcond=1e-12)
    if use_coarse_initial_guess:
      coarse_initial_guess = coarse_space @ (
          coarse_inverse @ (coarse_space.T @ schur_rhs))
      coarse_initial_guess_norm = float(np.linalg.norm(coarse_initial_guess))
      coarse_initial_residual = float(np.linalg.norm(
          schur_matvec(coarse_initial_guess) - schur_rhs))

  def preconditioner(residual):
    nonlocal balanced_coarse_preconditioner_schur_matvec_count
    if (schur_preconditioner == "block_jacobi+balanced_coarse" and
        coarse_inverse is not None):
      coarse_correction = coarse_space @ (
          coarse_inverse @ (coarse_space.T @ residual))
      balanced_residual = residual - schur_matvec(coarse_correction)
      local_correction = base_preconditioner(balanced_residual)
      local_coarse_leak = coarse_space @ (
          coarse_inverse @ (
              coarse_space.T @ schur_matvec(local_correction)))
      balanced_coarse_preconditioner_schur_matvec_count += 2
      return coarse_correction + local_correction - local_coarse_leak
    output = base_preconditioner(residual)
    if coarse_inverse is not None:
      output = output + coarse_space @ (coarse_inverse @ (coarse_space.T @ residual))
    return output

  x_b, schur_stats = _pcg_solve_from_matvec(
      schur_matvec,
      schur_rhs,
      iterations,
      tolerance,
      preconditioner=preconditioner,
      initial_solution=coarse_initial_guess)
  x_p = private_factor.solve(
      g_p - np.asarray(h_pb @ x_b).reshape(-1))
  solution = np.zeros(variable_count, dtype=float)
  solution[private_indices] = x_p
  solution[interface_indices] = x_b
  final_normal_residual = float(np.linalg.norm(hessian @ solution - gradient))
  final_schur_residual = float(np.linalg.norm(schur_matvec(x_b) - schur_rhs))
  solve_global_reduction_count = int(schur_stats["global_reduction_count"])
  total_global_reduction_count = (
      solve_global_reduction_count +
      residual_deflation_pilot_global_reductions +
      ritz_probe_global_reductions)
  return solution, {
      "iterations": int(schur_stats["iterations"]),
      "schur_iterations": int(schur_stats["iterations"]),
      "final_normal_residual": final_normal_residual,
      "final_schur_residual": final_schur_residual,
      "interface_variable_count": int(len(interface_indices)),
      "private_variable_count": int(len(private_indices)),
      "method": "interface_schur_pcg",
      "private_factor_damping": damping,
      "schur_preconditioner": schur_preconditioner,
      "schur_preconditioner_block_count": (
          len(overlap_factors) if schur_preconditioner == "overlap_schwarz"
          else
          len(block_factors) if uses_block_jacobi
          else len(interface_indices)),
      "schur_preconditioner_overlap_level": int(
          overlap_schwarz_overlap_level),
      "overlap_schwarz_setup_schur_matvec_count": int(
          overlap_schwarz_setup_schur_matvec_count),
      "overlap_schwarz_setup_local_schur_column_count": int(
          overlap_schwarz_setup_local_schur_column_count),
      "overlap_schwarz_duplicate_group_count": int(
          overlap_schwarz_duplicate_group_count),
      "balanced_coarse_preconditioner_schur_matvec_count": int(
          balanced_coarse_preconditioner_schur_matvec_count),
      "coarse_basis_column_count": int(coarse_column_count),
      "coarse_basis_rank": int(coarse_rank),
      "coarse_initial_guess_used": bool(coarse_initial_guess is not None),
      "coarse_regularization": float(coarse_regularization),
      "coarse_initial_guess_norm": float(coarse_initial_guess_norm),
      "coarse_initial_residual": float(coarse_initial_residual),
      "coarse_setup_schur_matvec_count": int(coarse_rank),
      "coarse_operator_scalar_count": int(
          coarse_rank * (coarse_rank + 1) // 2),
      "coarse_operator_condition": float(coarse_operator_condition),
      "residual_deflation_requested_rank": int(residual_deflation_rank),
      "residual_deflation_basis_rank": int(residual_deflation_basis_rank),
      "residual_deflation_pilot_iterations": int(
          residual_deflation_pilot_used_iterations),
      "residual_deflation_pilot_global_reduction_count": int(
          residual_deflation_pilot_global_reductions),
      "residual_deflation_pilot_final_residual": float(
          residual_deflation_pilot_final_residual),
      "ritz_deflation_requested_rank": int(ritz_deflation_rank),
      "ritz_mode": ritz_mode,
      "ritz_rank_selection_mode": ritz_rank_selection_mode,
      "ritz_value_threshold": (
          None if ritz_value_threshold is None else float(ritz_value_threshold)),
      "ritz_energy_capture_fraction": (
          None if ritz_energy_capture_fraction is None
          else float(ritz_energy_capture_fraction)),
      "ritz_deflation_requested_rank_before_selection": int(
          ritz_deflation_requested_rank_before_selection),
      "ritz_deflation_selected_rank": int(ritz_deflation_selected_rank),
      "ritz_deflation_basis_rank": int(ritz_deflation_basis_rank),
      "ritz_probe_iterations": int(ritz_probe_used_iterations),
      "ritz_probe_requested_iterations": int(ritz_probe_requested_iterations),
      "ritz_probe_projected_dimension": int(ritz_projected_dimension),
      "ritz_probe_global_reduction_count": int(ritz_probe_global_reductions),
      "ritz_lanczos_m_orthogonality_error": float(
          ritz_lanczos_m_orthogonality_error),
      "ritz_selected_values": ritz_selected_values,
      "ritz_rhs_coefficients": ritz_rhs_coefficients,
      "ritz_energy_contributions": ritz_energy_contributions,
      "ritz_energy_capture_total": float(ritz_energy_capture_total),
      "ritz_energy_capture_selected": float(ritz_energy_capture_selected),
      "ritz_portfolio_selected_mode": ritz_portfolio_selected_mode,
      "ritz_portfolio_selected_incremental_merit": float(
          ritz_portfolio_selected_incremental_merit),
      "ritz_portfolio_base_merit": float(ritz_portfolio_base_merit),
      "ritz_portfolio_base_residual_norm": float(
          ritz_portfolio_base_residual_norm),
      "ritz_portfolio_candidate_count": int(ritz_portfolio_candidate_count),
      "ritz_portfolio_scoring_schur_matvec_count": int(
          ritz_portfolio_scoring_schur_matvec_count),
      "ritz_portfolio_candidates": ritz_portfolio_candidates,
      "ritz_hybrid_candidate_count": int(ritz_hybrid_candidate_count),
      "ritz_hybrid_candidate_modes": ritz_hybrid_candidate_modes,
      "ritz_hybrid_candidate_ranks": ritz_hybrid_candidate_ranks,
      "solve_global_reduction_count": solve_global_reduction_count,
      "global_reduction_count": total_global_reduction_count,
  }


def interface_schur_central_equivalence_diagnostic(
    hessian,
    gradient: np.ndarray,
    solution: np.ndarray,
    interface_indices: np.ndarray,
    damping: float = 1e-12):
  """Measures exact interface-Schur distance to the centralized linear solve.

  This is an offline diagnostic, not a deployable communication routine. It
  materializes the dense interface Schur matrix so tests and research reports
  can evaluate the central-equivalence target directly.
  """
  if identity is None or splu is None:
    raise RuntimeError("scipy sparse support is required")
  hessian = _as_sparse_hessian(hessian)
  gradient = np.asarray(gradient, dtype=float).reshape(-1)
  solution = np.asarray(solution, dtype=float).reshape(-1)
  variable_count = int(hessian.shape[0])
  if hessian.shape[0] != hessian.shape[1]:
    raise ValueError("hessian must be square")
  if len(gradient) != variable_count:
    raise ValueError("gradient dimension does not match hessian")
  if len(solution) != variable_count:
    raise ValueError("solution dimension does not match hessian")
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  if interface_indices.size:
    if (int(np.min(interface_indices)) < 0 or
        int(np.max(interface_indices)) >= variable_count):
      raise ValueError("interface index outside hessian dimension")
  interface_indices = np.asarray(sorted(set(int(i) for i in interface_indices)),
                                 dtype=int)
  interface_mask = np.zeros(variable_count, dtype=bool)
  interface_mask[interface_indices] = True
  private_indices = np.nonzero(~interface_mask)[0].astype(int)

  if variable_count == 0:
    return {
        "diagnostic_model": "exact_dense_schur_reference",
        "interface_variable_count": 0,
        "private_variable_count": 0,
        "schur_energy_gap": 0.0,
        "schur_residual_norm": 0.0,
        "schur_residual_dual_energy": 0.0,
        "private_backsubstitution_error_norm": 0.0,
    }

  if len(interface_indices) == 0:
    system = hessian
    if damping > 0.0:
      system = system + damping * identity(variable_count, format="csc")
    reference = splu(system.tocsc()).solve(gradient)
    error = solution - reference
    energy_gap = float(error.T @ np.asarray(hessian @ error).reshape(-1))
    residual = np.asarray(hessian @ solution).reshape(-1) - gradient
    return {
        "diagnostic_model": "exact_dense_schur_reference",
        "interface_variable_count": 0,
        "private_variable_count": variable_count,
        "schur_energy_gap": max(0.0, energy_gap),
        "schur_residual_norm": float(np.linalg.norm(residual)),
        "schur_residual_dual_energy": max(0.0, energy_gap),
        "private_backsubstitution_error_norm": 0.0,
    }

  if len(private_indices) == 0:
    dense_hessian = hessian.toarray()
    reference = np.linalg.solve(
        dense_hessian + damping * np.eye(variable_count), gradient)
    error = solution - reference
    residual = dense_hessian @ solution - gradient
    dual = np.linalg.solve(
        dense_hessian + damping * np.eye(variable_count), residual)
    return {
        "diagnostic_model": "exact_dense_schur_reference",
        "interface_variable_count": int(len(interface_indices)),
        "private_variable_count": 0,
        "schur_energy_gap": max(0.0, float(error.T @ dense_hessian @ error)),
        "schur_residual_norm": float(np.linalg.norm(residual)),
        "schur_residual_dual_energy": max(0.0, float(residual.T @ dual)),
        "private_backsubstitution_error_norm": 0.0,
    }

  h_pp = hessian[private_indices[:, None], private_indices].tocsc()
  if damping > 0.0:
    h_pp = h_pp + damping * identity(h_pp.shape[0], format="csc")
  h_pb = hessian[private_indices[:, None], interface_indices].tocsc()
  h_bp = hessian[interface_indices[:, None], private_indices].tocsc()
  h_bb = hessian[interface_indices[:, None], interface_indices].tocsc()
  g_p = gradient[private_indices]
  g_b = gradient[interface_indices]
  private_factor = splu(h_pp)
  private_rhs_response = private_factor.solve(g_p)
  schur_rhs = g_b - np.asarray(h_bp @ private_rhs_response).reshape(-1)
  schur_columns = []
  for index in range(len(interface_indices)):
    basis = np.zeros(len(interface_indices), dtype=float)
    basis[index] = 1.0
    private_response = private_factor.solve(
        np.asarray(h_pb @ basis).reshape(-1))
    schur_columns.append(
        np.asarray(h_bb @ basis).reshape(-1) -
        np.asarray(h_bp @ private_response).reshape(-1))
  schur = np.column_stack(schur_columns)
  schur = 0.5 * (schur + schur.T)
  try:
    reference_interface = np.linalg.solve(schur, schur_rhs)
  except np.linalg.LinAlgError:
    reference_interface = np.linalg.pinv(schur, rcond=1e-12) @ schur_rhs
  candidate_interface = solution[interface_indices]
  interface_error = candidate_interface - reference_interface
  schur_residual = schur @ candidate_interface - schur_rhs
  try:
    dual_solution = np.linalg.solve(schur, schur_residual)
  except np.linalg.LinAlgError:
    dual_solution = np.linalg.pinv(schur, rcond=1e-12) @ schur_residual
  backsub_private = private_factor.solve(
      g_p - np.asarray(h_pb @ candidate_interface).reshape(-1))
  private_error = solution[private_indices] - backsub_private
  energy_gap = float(interface_error.T @ schur @ interface_error)
  dual_energy = float(schur_residual.T @ dual_solution)
  return {
      "diagnostic_model": "exact_dense_schur_reference",
      "interface_variable_count": int(len(interface_indices)),
      "private_variable_count": int(len(private_indices)),
      "schur_energy_gap": max(0.0, energy_gap),
      "schur_residual_norm": float(np.linalg.norm(schur_residual)),
      "schur_residual_dual_energy": max(0.0, dual_energy),
      "private_backsubstitution_error_norm": float(np.linalg.norm(private_error)),
      "interface_reference_norm": float(np.linalg.norm(reference_interface)),
      "interface_error_norm": float(np.linalg.norm(interface_error)),
      "schur_condition": float(np.linalg.cond(schur)),
  }


def interface_schur_iterative_central_equivalence_diagnostic(
    hessian,
    gradient: np.ndarray,
    solution: np.ndarray,
    interface_indices: np.ndarray,
    iterations: int,
    damping: float = 1e-12,
    tolerance: float = 1e-10,
    interface_blocks: list[np.ndarray] | None = None,
    schur_preconditioner: str = "block_jacobi"):
  """Estimates r_B^T S^{-1} r_B by Schur matvecs and PCG.

  Unlike interface_schur_central_equivalence_diagnostic(), this routine does
  not materialize the dense Schur matrix. It is still a diagnostic because it
  spends extra Schur matvecs after the candidate solve.
  """
  if identity is None or splu is None:
    raise RuntimeError("scipy sparse support is required")
  hessian = _as_sparse_hessian(hessian)
  gradient = np.asarray(gradient, dtype=float).reshape(-1)
  solution = np.asarray(solution, dtype=float).reshape(-1)
  variable_count = int(hessian.shape[0])
  if hessian.shape[0] != hessian.shape[1]:
    raise ValueError("hessian must be square")
  if len(gradient) != variable_count:
    raise ValueError("gradient dimension does not match hessian")
  if len(solution) != variable_count:
    raise ValueError("solution dimension does not match hessian")
  interface_indices = np.asarray(interface_indices, dtype=int).reshape(-1)
  if interface_indices.size:
    if (int(np.min(interface_indices)) < 0 or
        int(np.max(interface_indices)) >= variable_count):
      raise ValueError("interface index outside hessian dimension")
  interface_indices = np.asarray(sorted(set(int(i) for i in interface_indices)),
                                 dtype=int)
  interface_mask = np.zeros(variable_count, dtype=bool)
  interface_mask[interface_indices] = True
  private_indices = np.nonzero(~interface_mask)[0].astype(int)
  iterations = max(0, int(iterations))
  if schur_preconditioner not in {"diagonal", "block_jacobi"}:
    raise ValueError(
        "unsupported iterative diagnostic schur_preconditioner: "
        f"{schur_preconditioner}")

  if variable_count == 0 or len(interface_indices) == 0:
    residual = np.asarray(hessian @ solution).reshape(-1) - gradient
    return {
        "diagnostic_model": "iterative_schur_dual_energy",
        "materializes_dense_schur": False,
        "interface_variable_count": int(len(interface_indices)),
        "private_variable_count": int(len(private_indices)),
        "schur_residual_norm": float(np.linalg.norm(residual)),
        "schur_residual_dual_energy_estimate": 0.0,
        "dual_solve_iterations": 0,
        "dual_solve_final_residual": 0.0,
        "private_backsubstitution_error_norm": 0.0,
    }

  if len(private_indices) == 0:
    system = hessian
    if damping > 0.0:
      system = system + damping * identity(variable_count, format="csc")
    h_diag = np.maximum(np.abs(system.diagonal()), 1e-12)

    def matvec(vector):
      return np.asarray(system @ vector).reshape(-1)

    def preconditioner(residual):
      return np.asarray(residual, dtype=float).reshape(-1) / h_diag

    residual = matvec(solution) - gradient
    dual, dual_stats = _pcg_solve_from_matvec(
        matvec, residual, iterations, tolerance, preconditioner=preconditioner)
    return {
        "diagnostic_model": "iterative_schur_dual_energy",
        "materializes_dense_schur": False,
        "interface_variable_count": int(len(interface_indices)),
        "private_variable_count": 0,
        "schur_residual_norm": float(np.linalg.norm(residual)),
        "schur_residual_dual_energy_estimate": max(
            0.0, float(residual @ dual)),
        "dual_solve_iterations": int(dual_stats.get("iterations", 0)),
        "dual_solve_final_residual": float(
            dual_stats.get("final_residual", 0.0)),
        "private_backsubstitution_error_norm": 0.0,
    }

  h_pp = hessian[private_indices[:, None], private_indices].tocsc()
  if damping > 0.0:
    h_pp = h_pp + damping * identity(h_pp.shape[0], format="csc")
  h_pb = hessian[private_indices[:, None], interface_indices].tocsc()
  h_bp = hessian[interface_indices[:, None], private_indices].tocsc()
  h_bb = hessian[interface_indices[:, None], interface_indices].tocsc()
  g_p = gradient[private_indices]
  g_b = gradient[interface_indices]
  private_factor = splu(h_pp)
  schur_rhs = g_b - np.asarray(
      h_bp @ private_factor.solve(g_p)).reshape(-1)

  def schur_matvec(vector):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    private_response = private_factor.solve(
        np.asarray(h_pb @ vector).reshape(-1))
    return (np.asarray(h_bb @ vector).reshape(-1) -
            np.asarray(h_bp @ private_response).reshape(-1))

  interface_position = {
      int(global_index): local_index
      for local_index, global_index in enumerate(interface_indices)
  }
  local_blocks: list[np.ndarray] = []
  if interface_blocks is not None:
    for block in interface_blocks:
      local = [
          interface_position[int(global_index)]
          for global_index in np.asarray(block, dtype=int).reshape(-1)
          if int(global_index) in interface_position
      ]
      if local:
        local_blocks.append(np.asarray(local, dtype=int))
  if not local_blocks:
    local_blocks = [
        np.asarray([index], dtype=int) for index in range(len(interface_indices))
    ]

  schur_diag = np.maximum(np.abs(h_bb.diagonal()), 1e-12)
  block_factors = []
  if schur_preconditioner == "block_jacobi":
    for block in local_blocks:
      h_bb_block = h_bb[block[:, None], block].tocsc()
      h_pb_block = h_pb[:, block].tocsc()
      h_bp_block = h_bp[block, :].tocsc()
      private_response = private_factor.solve(h_pb_block.toarray())
      schur_block = (
          h_bb_block.toarray() -
          np.asarray(h_bp_block @ private_response, dtype=float))
      if damping > 0.0:
        schur_block = schur_block + damping * np.eye(len(block))
      block_factors.append((block, np.linalg.inv(schur_block)))

  def preconditioner(residual):
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if schur_preconditioner == "block_jacobi":
      output = np.zeros_like(residual)
      for block, inverse in block_factors:
        output[block] = inverse @ residual[block]
      return output
    return residual / schur_diag

  candidate_interface = solution[interface_indices]
  schur_residual = schur_matvec(candidate_interface) - schur_rhs
  dual, dual_stats = _pcg_solve_from_matvec(
      schur_matvec,
      schur_residual,
      iterations,
      tolerance,
      preconditioner=preconditioner)
  backsub_private = private_factor.solve(
      g_p - np.asarray(h_pb @ candidate_interface).reshape(-1))
  private_error = solution[private_indices] - backsub_private
  return {
      "diagnostic_model": "iterative_schur_dual_energy",
      "materializes_dense_schur": False,
      "interface_variable_count": int(len(interface_indices)),
      "private_variable_count": int(len(private_indices)),
      "schur_residual_norm": float(np.linalg.norm(schur_residual)),
      "schur_residual_dual_energy_estimate": max(
          0.0, float(schur_residual @ dual)),
      "dual_solve_iterations": int(dual_stats.get("iterations", 0)),
      "dual_solve_final_residual": float(
          dual_stats.get("final_residual", 0.0)),
      "dual_solve_global_reduction_count": int(
          dual_stats.get("global_reduction_count", 0)),
      "private_backsubstitution_error_norm": float(np.linalg.norm(private_error)),
  }


def solve_distributed_normal_system(matrix, rhs: np.ndarray,
                                    blocks: dict[int, np.ndarray],
                                    iterations: int,
                                    linear_solver: str,
                                    relaxation: float,
                                    damping: float):
  if linear_solver == "block_jacobi":
    return block_jacobi_normal_solve(matrix, rhs, blocks, iterations,
                                     relaxation, damping)
  if linear_solver == "pcg":
    return block_pcg_normal_solve(matrix, rhs, blocks, iterations, damping)
  raise ValueError(f"unsupported linear_solver: {linear_solver}")


def solve_distributed_hessian_system(hessian, gradient: np.ndarray,
                                     blocks: dict[int, np.ndarray],
                                     iterations: int,
                                     linear_solver: str,
                                     relaxation: float,
                                     damping: float):
  if linear_solver == "block_jacobi":
    return block_jacobi_hessian_solve(hessian, gradient, blocks, iterations,
                                      relaxation, damping)
  if linear_solver == "pcg":
    return block_pcg_hessian_solve(hessian, gradient, blocks, iterations,
                                   damping)
  if linear_solver == "compact_pcg":
    solution, stats = block_pcg_hessian_solve(hessian, gradient, blocks,
                                             iterations, damping)
    stats["method"] = "compact_pcg"
    stats["base_method"] = "block_pcg"
    return solution, stats
  raise ValueError(f"unsupported linear_solver: {linear_solver}")


def _split_private_separator_edges(graph_edges: list[Edge],
                                   robot_of: dict[int, int]):
  private_edges = []
  separator_edges = []
  for edge in graph_edges:
    if edge.i not in robot_of or edge.j not in robot_of:
      private_edges.append(edge)
    elif robot_of[edge.i] == robot_of[edge.j]:
      private_edges.append(edge)
    else:
      separator_edges.append(edge)
  return private_edges, separator_edges


def _summary_normal_equation_for_system(assembler,
                                        private_edges: list[Edge],
                                        separator_edges: list[Edge],
                                        pose_ids: list[int],
                                        dim: int,
                                        weighted: bool,
                                        cost_mode: str,
                                        anchor_pose: int,
                                        *assembler_extra_args,
                                        summary_selection_mode: str = "all",
                                        summary_max_offdiag_block_edges:
                                        int | None = None):
  private_matrix, private_rhs, meta = assembler(
      private_edges, pose_ids, *assembler_extra_args, dim, weighted, cost_mode,
      anchor_pose)
  separator_matrix, separator_rhs, _ = assembler(
      separator_edges, pose_ids, *assembler_extra_args, dim, weighted,
      cost_mode, anchor_pose)
  private_hessian = (private_matrix.T @ private_matrix).tocsc()
  private_gradient = np.asarray(private_matrix.T @ private_rhs,
                                dtype=float).reshape(-1)
  separator_summary = normal_equation_block_summary(
      separator_matrix, separator_rhs, meta["block_dim"])
  full_separator_summary = recount_normal_summary_payload(separator_summary)
  separator_summary = select_normal_summary_blocks(
      full_separator_summary,
      mode=summary_selection_mode,
      max_offdiag_block_edges=summary_max_offdiag_block_edges,
  )
  separator_hessian, separator_gradient = (
      reconstruct_sparse_normal_equation_from_summary(separator_summary))
  full_separator_hessian, full_separator_gradient = (
      reconstruct_sparse_normal_equation_from_summary(full_separator_summary))
  return (
      private_hessian + separator_hessian,
      private_gradient + separator_gradient,
      meta,
      separator_summary,
      private_hessian + full_separator_hessian,
      private_gradient + full_separator_gradient,
      full_separator_summary,
  )


def _replace_separator_summary_in_total_system(full_hessian,
                                               full_gradient: np.ndarray,
                                               full_separator_summary: dict,
                                               selected_separator_summary:
                                               dict):
  full_separator_hessian, full_separator_gradient = (
      reconstruct_sparse_normal_equation_from_summary(full_separator_summary))
  selected_separator_hessian, selected_separator_gradient = (
      reconstruct_sparse_normal_equation_from_summary(selected_separator_summary))
  return (
      full_hessian - full_separator_hessian + selected_separator_hessian,
      full_gradient - full_separator_gradient + selected_separator_gradient,
  )


def _rotations_from_solution(solution: np.ndarray, pose_ids: list[int],
                             offsets: dict[int, int], dim: int,
                             anchor_pose: int):
  rotations: dict[int, np.ndarray] = {}
  for pose_id in pose_ids:
    if pose_id == anchor_pose:
      rotations[pose_id] = np.eye(dim)
      continue
    offset = offsets[pose_id]
    raw = solution[offset:offset + dim * dim].reshape((dim, dim))
    rotations[pose_id] = project_rotation_block(raw)
  return rotations


def _raw_rotations_from_solution(solution: np.ndarray, pose_ids: list[int],
                                 offsets: dict[int, int], dim: int,
                                 anchor_pose: int):
  rotations: dict[int, np.ndarray] = {}
  for pose_id in pose_ids:
    if pose_id == anchor_pose:
      rotations[pose_id] = np.eye(dim)
      continue
    offset = offsets[pose_id]
    rotations[pose_id] = np.asarray(
        solution[offset:offset + dim * dim], dtype=float).reshape((dim, dim))
  return rotations


def rotation_projection_safety_certificate(
    solution: np.ndarray,
    pose_ids: list[int],
    offsets: dict[int, int],
    dim: int,
    anchor_pose: int):
  """Measures how far relaxed rotation blocks move under SO(d) projection."""
  correction_norms = []
  orthogonality_errors = []
  determinant_deviations = []
  for pose_id in pose_ids:
    if int(pose_id) == int(anchor_pose):
      continue
    offset = int(offsets[int(pose_id)])
    raw = np.asarray(
        solution[offset:offset + dim * dim], dtype=float).reshape((dim, dim))
    projected = project_rotation_block(raw)
    correction_norms.append(float(np.linalg.norm(raw - projected, ord="fro")))
    orthogonality_errors.append(float(
        np.linalg.norm(raw.T @ raw - np.eye(dim), ord="fro")))
    determinant_deviations.append(float(abs(np.linalg.det(raw) - 1.0)))

  if correction_norms:
    correction_array = np.asarray(correction_norms, dtype=float)
    orthogonality_array = np.asarray(orthogonality_errors, dtype=float)
    determinant_array = np.asarray(determinant_deviations, dtype=float)
    max_projection_correction = float(np.max(correction_array))
    mean_projection_correction = float(np.mean(correction_array))
    sum_projection_correction = float(np.sum(correction_array))
    max_orthogonality_error = float(np.max(orthogonality_array))
    mean_orthogonality_error = float(np.mean(orthogonality_array))
    max_determinant_deviation = float(np.max(determinant_array))
    mean_determinant_deviation = float(np.mean(determinant_array))
  else:
    max_projection_correction = 0.0
    mean_projection_correction = 0.0
    sum_projection_correction = 0.0
    max_orthogonality_error = 0.0
    mean_orthogonality_error = 0.0
    max_determinant_deviation = 0.0
    mean_determinant_deviation = 0.0

  return {
      "model": "rotation_projection_safety_certificate",
      "pose_count": int(len(pose_ids)),
      "evaluated_pose_count": int(len(correction_norms)),
      "max_projection_correction_norm": max_projection_correction,
      "mean_projection_correction_norm": mean_projection_correction,
      "sum_projection_correction_norm": sum_projection_correction,
      "max_orthogonality_error": max_orthogonality_error,
      "mean_orthogonality_error": mean_orthogonality_error,
      "max_determinant_deviation": max_determinant_deviation,
      "mean_determinant_deviation": mean_determinant_deviation,
  }


def rotation_projection_cost_certificate(
    graph_edges: list[Edge],
    solution: np.ndarray,
    pose_ids: list[int],
    offsets: dict[int, int],
    translations: dict[int, np.ndarray],
    dim: int,
    anchor_pose: int,
    weighted: bool,
    cost_mode: str):
  """Measures measurement-cost change caused by SO(d) projection."""
  raw_rotations = _raw_rotations_from_solution(
      solution=solution,
      pose_ids=pose_ids,
      offsets=offsets,
      dim=dim,
      anchor_pose=anchor_pose)
  projected_rotations = {
      pose_id: project_rotation_block(rotation)
      for pose_id, rotation in raw_rotations.items()
  }
  raw_poses = _poses_from_rotations_translations(
      pose_ids=pose_ids,
      rotations=raw_rotations,
      translations=translations,
      dim=dim)
  projected_poses = _poses_from_rotations_translations(
      pose_ids=pose_ids,
      rotations=projected_rotations,
      translations=translations,
      dim=dim)
  raw_cost = _pose_set_total_chordal_cost(
      graph_edges=graph_edges,
      poses=raw_poses,
      weighted=weighted,
      cost_mode=cost_mode)
  projected_cost = _pose_set_total_chordal_cost(
      graph_edges=graph_edges,
      poses=projected_poses,
      weighted=weighted,
      cost_mode=cost_mode)
  raw_total = float(raw_cost["total_cost"])
  projected_total = float(projected_cost["total_cost"])
  signed_delta = float(projected_total - raw_total)
  return {
      "model": "rotation_projection_cost_certificate",
      "raw_rotation_cost": raw_cost,
      "projected_rotation_cost": projected_cost,
      "projected_minus_raw_cost_delta": signed_delta,
      "absolute_projection_cost_delta": float(abs(signed_delta)),
      "relative_projection_cost_delta": (
          float(abs(signed_delta) / max(1e-30, abs(raw_total)))
          if math.isfinite(raw_total) and math.isfinite(projected_total)
          else math.inf),
  }


def _translations_from_solution(solution: np.ndarray, pose_ids: list[int],
                                offsets: dict[int, int], dim: int,
                                anchor_pose: int):
  translations: dict[int, np.ndarray] = {}
  for pose_id in pose_ids:
    if pose_id == anchor_pose:
      translations[pose_id] = np.zeros(dim)
      continue
    offset = offsets[pose_id]
    translations[pose_id] = solution[offset:offset + dim]
  return translations


def _poses_from_rotations_translations(pose_ids: list[int],
                                       rotations: dict[int, np.ndarray],
                                       translations: dict[int, np.ndarray],
                                       dim: int):
  poses: dict[int, np.ndarray] = {}
  for pose_id in pose_ids:
    pose = np.eye(4, dtype=float)
    pose[:dim, :dim] = rotations[pose_id]
    pose[:dim, 3] = translations[pose_id]
    if dim == 2:
      pose[2, 2] = 1.0
    poses[pose_id] = pose
  return poses


def write_manual_matrix_pose_set(path: Path, poses: dict[int, np.ndarray],
                                 dim: int):
  ordered = sorted(poses)
  if not ordered:
    raise ValueError("cannot write an empty pose set")
  dense = np.zeros((dim, len(ordered) * (dim + 1)), dtype=float)
  for idx, pose_id in enumerate(ordered):
    col = idx * (dim + 1)
    pose = poses[pose_id]
    dense[:, col:col + dim] = pose[:dim, :dim]
    dense[:, col + dim] = pose[:dim, 3]
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(path, dense, fmt="%.17g")


def solve_centralized_chordal_initialization(graph_edges: list[Edge],
                                             pose_ids: list[int],
                                             weighted: bool,
                                             cost_mode: str,
                                             anchor_pose: int | None = None):
  pose_ids = sorted(pose_ids)
  if not pose_ids:
    return {}, {"pose_count": 0}
  anchor_pose = pose_ids[0] if anchor_pose is None else anchor_pose
  dim = pose_dimension_from_edges(graph_edges)
  rot_matrix, rot_rhs, rot_meta = assemble_rotation_system(
      graph_edges, pose_ids, dim, weighted, cost_mode, anchor_pose)
  rot_solution, rot_stats = _solve_central_system(rot_matrix, rot_rhs)
  rotations = _rotations_from_solution(rot_solution, pose_ids,
                                       rot_meta["offsets"], dim, anchor_pose)

  trans_matrix, trans_rhs, trans_meta = assemble_translation_system(
      graph_edges, pose_ids, rotations, dim, weighted, cost_mode, anchor_pose)
  trans_solution, trans_stats = _solve_central_system(trans_matrix, trans_rhs)
  translations = _translations_from_solution(trans_solution, pose_ids,
                                             trans_meta["offsets"], dim,
                                             anchor_pose)
  poses = _poses_from_rotations_translations(pose_ids, rotations, translations,
                                             dim)
  return poses, {
      "pose_count": len(pose_ids),
      "dimension": dim,
      "anchor_pose": anchor_pose,
      "rotation_stats": {**rot_stats, **{
          "equation_count": rot_meta["equation_count"],
          "variable_count": rot_meta["variable_count"],
      }},
      "translation_stats": {**trans_stats, **{
          "equation_count": trans_meta["equation_count"],
          "variable_count": trans_meta["variable_count"],
      }},
      "method": "centralized_lsqr",
  }


def _separator_edge_count(graph_edges: list[Edge], robot_of: dict[int, int]):
  count = 0
  for edge in graph_edges:
    if edge.i in robot_of and edge.j in robot_of and robot_of[edge.i] != robot_of[edge.j]:
      count += 1
  return count


def separator_boundary_pose_unit_count(graph_edges: list[Edge],
                                       robot_of: dict[int, int]):
  units = set()
  for edge in graph_edges:
    if edge.i not in robot_of or edge.j not in robot_of:
      continue
    ri = robot_of[edge.i]
    rj = robot_of[edge.j]
    if ri == rj:
      continue
    units.add((int(edge.i), int(ri), int(rj)))
    units.add((int(edge.j), int(rj), int(ri)))
  return len(units)


def _robot_count_for_poses(pose_ids: list[int], robot_of: dict[int, int]):
  return len({robot_of[pose_id] for pose_id in pose_ids if pose_id in robot_of})


def _scalar_allreduce_bytes(reduction_count: int, robot_count: int,
                            scalar_count: int = 1):
  if reduction_count <= 0 or robot_count <= 1 or scalar_count <= 0:
    return 0
  return int(reduction_count) * 2 * (int(robot_count) - 1) * int(scalar_count) * 8


def _bytes_to_mb(byte_count: int):
  return float(byte_count) / (1024.0 * 1024.0)


def _separator_block_exchange_bytes(iteration_count: int,
                                    separator_edge_count: int,
                                    block_dim: int):
  if iteration_count <= 0 or separator_edge_count <= 0 or block_dim <= 0:
    return 0
  return int(iteration_count) * int(separator_edge_count) * 2 * int(block_dim) * 8


def _directed_separator_unit_exchange_bytes(iteration_count: int,
                                            separator_unit_count: int,
                                            block_dim: int):
  if iteration_count <= 0 or separator_unit_count <= 0 or block_dim <= 0:
    return 0
  return int(iteration_count) * int(separator_unit_count) * int(block_dim) * 8


def estimate_dci_linear_solve_communication(rotation_stats: dict,
                                            translation_stats: dict,
                                            robot_count: int,
                                            separator_edge_count: int,
                                            dimension: int):
  rot_reductions = int(rotation_stats.get("global_reduction_count", 0))
  trans_reductions = int(translation_stats.get("global_reduction_count", 0))
  rot_bytes = _scalar_allreduce_bytes(rot_reductions, robot_count)
  trans_bytes = _scalar_allreduce_bytes(trans_reductions, robot_count)
  rot_exchange_bytes = _separator_block_exchange_bytes(
      int(rotation_stats.get("iterations", 0)), separator_edge_count,
      dimension * dimension)
  trans_exchange_bytes = _separator_block_exchange_bytes(
      int(translation_stats.get("iterations", 0)), separator_edge_count,
      dimension)
  reduction_total_bytes = rot_bytes + trans_bytes
  exchange_total_bytes = rot_exchange_bytes + trans_exchange_bytes
  total_bytes = reduction_total_bytes + exchange_total_bytes
  return {
      "robot_count": int(robot_count),
      "separator_edge_count": int(separator_edge_count),
      "rotation_global_reduction_count": rot_reductions,
      "translation_global_reduction_count": trans_reductions,
      "rotation_global_reduction_bytes": rot_bytes,
      "translation_global_reduction_bytes": trans_bytes,
      "rotation_separator_exchange_bytes": rot_exchange_bytes,
      "translation_separator_exchange_bytes": trans_exchange_bytes,
      "separator_exchange_bytes": exchange_total_bytes,
      "pcg_global_reduction_bytes": reduction_total_bytes,
      "rotation_global_reduction_mb": _bytes_to_mb(rot_bytes),
      "translation_global_reduction_mb": _bytes_to_mb(trans_bytes),
      "rotation_separator_exchange_mb": _bytes_to_mb(rot_exchange_bytes),
      "translation_separator_exchange_mb": _bytes_to_mb(trans_exchange_bytes),
      "separator_exchange_mb": _bytes_to_mb(exchange_total_bytes),
      "pcg_global_reduction_mb": _bytes_to_mb(reduction_total_bytes),
      "linear_solve_total_estimated_bytes": total_bytes,
      "linear_solve_total_estimated_mb": _bytes_to_mb(total_bytes),
      "model": "separator_block_exchange_plus_tree_allreduce_scalar_doubles",
  }


def estimate_dci_boundary_pose_solve_communication(
    rotation_stats: dict,
    translation_stats: dict,
    robot_count: int,
    boundary_pose_unit_count: int,
    dimension: int):
  rot_reductions = int(rotation_stats.get("global_reduction_count", 0))
  trans_reductions = int(translation_stats.get("global_reduction_count", 0))
  rot_bytes = _scalar_allreduce_bytes(rot_reductions, robot_count)
  trans_bytes = _scalar_allreduce_bytes(trans_reductions, robot_count)
  rot_exchange_bytes = _directed_separator_unit_exchange_bytes(
      int(rotation_stats.get("iterations", 0)), boundary_pose_unit_count,
      dimension * dimension)
  trans_exchange_bytes = _directed_separator_unit_exchange_bytes(
      int(translation_stats.get("iterations", 0)), boundary_pose_unit_count,
      dimension)
  reduction_total_bytes = rot_bytes + trans_bytes
  exchange_total_bytes = rot_exchange_bytes + trans_exchange_bytes
  total_bytes = reduction_total_bytes + exchange_total_bytes
  return {
      "robot_count": int(robot_count),
      "boundary_pose_unit_count": int(boundary_pose_unit_count),
      "rotation_global_reduction_count": rot_reductions,
      "translation_global_reduction_count": trans_reductions,
      "rotation_global_reduction_bytes": rot_bytes,
      "translation_global_reduction_bytes": trans_bytes,
      "rotation_separator_exchange_bytes": rot_exchange_bytes,
      "translation_separator_exchange_bytes": trans_exchange_bytes,
      "separator_exchange_bytes": exchange_total_bytes,
      "pcg_global_reduction_bytes": reduction_total_bytes,
      "rotation_global_reduction_mb": _bytes_to_mb(rot_bytes),
      "translation_global_reduction_mb": _bytes_to_mb(trans_bytes),
      "rotation_separator_exchange_mb": _bytes_to_mb(rot_exchange_bytes),
      "translation_separator_exchange_mb": _bytes_to_mb(trans_exchange_bytes),
      "separator_exchange_mb": _bytes_to_mb(exchange_total_bytes),
      "pcg_global_reduction_mb": _bytes_to_mb(reduction_total_bytes),
      "linear_solve_total_estimated_bytes": total_bytes,
      "linear_solve_total_estimated_mb": _bytes_to_mb(total_bytes),
      "model": "boundary_pose_exchange_plus_tree_allreduce_scalar_doubles",
  }


def estimate_dci_interface_schur_solve_communication(rotation_stats: dict,
                                                     translation_stats: dict,
                                                     robot_count: int):
  rot_reductions = int(rotation_stats.get("global_reduction_count", 0))
  trans_reductions = int(translation_stats.get("global_reduction_count", 0))
  rot_interface_variables = int(
      rotation_stats.get("interface_variable_count", 0))
  trans_interface_variables = int(
      translation_stats.get("interface_variable_count", 0))
  rot_iterations = int(rotation_stats.get("iterations", 0))
  trans_iterations = int(translation_stats.get("iterations", 0))
  rot_pilot_iterations = int(
      rotation_stats.get("residual_deflation_pilot_iterations", 0))
  trans_pilot_iterations = int(
      translation_stats.get("residual_deflation_pilot_iterations", 0))
  rot_ritz_iterations = int(rotation_stats.get("ritz_probe_iterations", 0))
  trans_ritz_iterations = int(
      translation_stats.get("ritz_probe_iterations", 0))
  rot_ritz_portfolio_scoring_matvecs = int(
      rotation_stats.get("ritz_portfolio_scoring_schur_matvec_count", 0))
  trans_ritz_portfolio_scoring_matvecs = int(
      translation_stats.get("ritz_portfolio_scoring_schur_matvec_count", 0))
  rot_overlap_setup_matvecs = int(
      rotation_stats.get("overlap_schwarz_setup_schur_matvec_count", 0))
  trans_overlap_setup_matvecs = int(
      translation_stats.get("overlap_schwarz_setup_schur_matvec_count", 0))
  rot_overlap_local_columns = int(rotation_stats.get(
      "overlap_schwarz_setup_local_schur_column_count", 0))
  trans_overlap_local_columns = int(translation_stats.get(
      "overlap_schwarz_setup_local_schur_column_count", 0))
  rot_balanced_coarse_matvecs = int(rotation_stats.get(
      "balanced_coarse_preconditioner_schur_matvec_count", 0))
  trans_balanced_coarse_matvecs = int(translation_stats.get(
      "balanced_coarse_preconditioner_schur_matvec_count", 0))
  rot_coarse_rank = int(rotation_stats.get("coarse_basis_rank", 0))
  trans_coarse_rank = int(translation_stats.get("coarse_basis_rank", 0))
  rot_reduction_bytes = _scalar_allreduce_bytes(rot_reductions, robot_count)
  trans_reduction_bytes = _scalar_allreduce_bytes(trans_reductions, robot_count)
  rot_interface_bytes = (
      max(0, (rot_iterations + rot_pilot_iterations + rot_ritz_iterations +
              rot_ritz_portfolio_scoring_matvecs +
              rot_overlap_setup_matvecs +
              rot_balanced_coarse_matvecs)) *
      max(0, rot_interface_variables) * 8)
  trans_interface_bytes = (
      max(0, (trans_iterations + trans_pilot_iterations +
              trans_ritz_iterations + trans_ritz_portfolio_scoring_matvecs +
              trans_overlap_setup_matvecs +
              trans_balanced_coarse_matvecs)) *
      max(0, trans_interface_variables) * 8)
  rot_pilot_interface_bytes = (
      max(0, rot_pilot_iterations) * max(0, rot_interface_variables) * 8)
  trans_pilot_interface_bytes = (
      max(0, trans_pilot_iterations) * max(0, trans_interface_variables) * 8)
  rot_ritz_interface_bytes = (
      max(0, rot_ritz_iterations) * max(0, rot_interface_variables) * 8)
  trans_ritz_interface_bytes = (
      max(0, trans_ritz_iterations) * max(0, trans_interface_variables) * 8)
  rot_ritz_portfolio_scoring_bytes = (
      max(0, rot_ritz_portfolio_scoring_matvecs) *
      max(0, rot_interface_variables) * 8)
  trans_ritz_portfolio_scoring_bytes = (
      max(0, trans_ritz_portfolio_scoring_matvecs) *
      max(0, trans_interface_variables) * 8)
  rot_overlap_setup_bytes = (
      max(0, rot_overlap_setup_matvecs) *
      max(0, rot_interface_variables) * 8)
  trans_overlap_setup_bytes = (
      max(0, trans_overlap_setup_matvecs) *
      max(0, trans_interface_variables) * 8)
  rot_balanced_coarse_preconditioner_bytes = (
      max(0, rot_balanced_coarse_matvecs) *
      max(0, rot_interface_variables) * 8)
  trans_balanced_coarse_preconditioner_bytes = (
      max(0, trans_balanced_coarse_matvecs) *
      max(0, trans_interface_variables) * 8)
  rot_coarse_vector_bytes = (
      max(0, rot_coarse_rank) * max(0, rot_interface_variables) * 8)
  trans_coarse_vector_bytes = (
      max(0, trans_coarse_rank) * max(0, trans_interface_variables) * 8)
  rot_coarse_scalar_count = max(0, rot_coarse_rank) * (
      max(0, rot_coarse_rank) + 1) // 2
  trans_coarse_scalar_count = max(0, trans_coarse_rank) * (
      max(0, trans_coarse_rank) + 1) // 2
  rot_coarse_reduction_bytes = _scalar_allreduce_bytes(
      rot_coarse_scalar_count, robot_count)
  trans_coarse_reduction_bytes = _scalar_allreduce_bytes(
      trans_coarse_scalar_count, robot_count)
  reduction_total_bytes = rot_reduction_bytes + trans_reduction_bytes
  interface_total_bytes = rot_interface_bytes + trans_interface_bytes
  coarse_setup_bytes = (
      rot_coarse_vector_bytes +
      trans_coarse_vector_bytes +
      rot_coarse_reduction_bytes +
      trans_coarse_reduction_bytes)
  total_bytes = (
      reduction_total_bytes + interface_total_bytes + coarse_setup_bytes)
  return {
      "robot_count": int(robot_count),
      "rotation_global_reduction_count": rot_reductions,
      "translation_global_reduction_count": trans_reductions,
      "rotation_interface_variable_count": rot_interface_variables,
      "translation_interface_variable_count": trans_interface_variables,
      "rotation_coarse_basis_rank": rot_coarse_rank,
      "translation_coarse_basis_rank": trans_coarse_rank,
      "rotation_residual_deflation_pilot_iterations": rot_pilot_iterations,
      "translation_residual_deflation_pilot_iterations":
          trans_pilot_iterations,
      "rotation_ritz_probe_iterations": rot_ritz_iterations,
      "translation_ritz_probe_iterations": trans_ritz_iterations,
      "rotation_ritz_portfolio_scoring_schur_matvec_count":
          rot_ritz_portfolio_scoring_matvecs,
      "translation_ritz_portfolio_scoring_schur_matvec_count":
          trans_ritz_portfolio_scoring_matvecs,
      "rotation_overlap_schwarz_setup_schur_matvec_count":
          rot_overlap_setup_matvecs,
      "translation_overlap_schwarz_setup_schur_matvec_count":
          trans_overlap_setup_matvecs,
      "rotation_overlap_schwarz_setup_local_schur_column_count":
          rot_overlap_local_columns,
      "translation_overlap_schwarz_setup_local_schur_column_count":
          trans_overlap_local_columns,
      "rotation_balanced_coarse_preconditioner_schur_matvec_count":
          rot_balanced_coarse_matvecs,
      "translation_balanced_coarse_preconditioner_schur_matvec_count":
          trans_balanced_coarse_matvecs,
      "rotation_global_reduction_bytes": rot_reduction_bytes,
      "translation_global_reduction_bytes": trans_reduction_bytes,
      "rotation_interface_vector_bytes": rot_interface_bytes,
      "translation_interface_vector_bytes": trans_interface_bytes,
      "rotation_residual_deflation_pilot_interface_vector_bytes":
          rot_pilot_interface_bytes,
      "translation_residual_deflation_pilot_interface_vector_bytes":
          trans_pilot_interface_bytes,
      "rotation_ritz_probe_interface_vector_bytes":
          rot_ritz_interface_bytes,
      "translation_ritz_probe_interface_vector_bytes":
          trans_ritz_interface_bytes,
      "rotation_ritz_portfolio_scoring_interface_vector_bytes":
          rot_ritz_portfolio_scoring_bytes,
      "translation_ritz_portfolio_scoring_interface_vector_bytes":
          trans_ritz_portfolio_scoring_bytes,
      "rotation_overlap_schwarz_setup_interface_vector_bytes":
          rot_overlap_setup_bytes,
      "translation_overlap_schwarz_setup_interface_vector_bytes":
          trans_overlap_setup_bytes,
      "rotation_balanced_coarse_preconditioner_interface_vector_bytes":
          rot_balanced_coarse_preconditioner_bytes,
      "translation_balanced_coarse_preconditioner_interface_vector_bytes":
          trans_balanced_coarse_preconditioner_bytes,
      "rotation_coarse_setup_vector_bytes": rot_coarse_vector_bytes,
      "translation_coarse_setup_vector_bytes": trans_coarse_vector_bytes,
      "rotation_coarse_operator_reduction_bytes": rot_coarse_reduction_bytes,
      "translation_coarse_operator_reduction_bytes": (
          trans_coarse_reduction_bytes),
      "coarse_setup_bytes": coarse_setup_bytes,
      "interface_vector_exchange_bytes": interface_total_bytes,
      "pcg_global_reduction_bytes": reduction_total_bytes,
      "rotation_global_reduction_mb": _bytes_to_mb(rot_reduction_bytes),
      "translation_global_reduction_mb": _bytes_to_mb(trans_reduction_bytes),
      "rotation_interface_vector_mb": _bytes_to_mb(rot_interface_bytes),
      "translation_interface_vector_mb": _bytes_to_mb(trans_interface_bytes),
      "residual_deflation_pilot_interface_vector_mb": _bytes_to_mb(
          rot_pilot_interface_bytes + trans_pilot_interface_bytes),
      "ritz_probe_interface_vector_mb": _bytes_to_mb(
          rot_ritz_interface_bytes + trans_ritz_interface_bytes),
      "ritz_portfolio_scoring_interface_vector_mb": _bytes_to_mb(
          rot_ritz_portfolio_scoring_bytes +
          trans_ritz_portfolio_scoring_bytes),
      "overlap_schwarz_setup_interface_vector_mb": _bytes_to_mb(
          rot_overlap_setup_bytes + trans_overlap_setup_bytes),
      "balanced_coarse_preconditioner_interface_vector_mb": _bytes_to_mb(
          rot_balanced_coarse_preconditioner_bytes +
          trans_balanced_coarse_preconditioner_bytes),
      "rotation_coarse_setup_vector_mb": _bytes_to_mb(
          rot_coarse_vector_bytes),
      "translation_coarse_setup_vector_mb": _bytes_to_mb(
          trans_coarse_vector_bytes),
      "rotation_coarse_operator_reduction_mb": _bytes_to_mb(
          rot_coarse_reduction_bytes),
      "translation_coarse_operator_reduction_mb": _bytes_to_mb(
          trans_coarse_reduction_bytes),
      "coarse_setup_mb": _bytes_to_mb(coarse_setup_bytes),
      "interface_vector_exchange_mb": _bytes_to_mb(interface_total_bytes),
      "pcg_global_reduction_mb": _bytes_to_mb(reduction_total_bytes),
      "linear_solve_total_estimated_bytes": total_bytes,
      "linear_solve_total_estimated_mb": _bytes_to_mb(total_bytes),
      "model": "interface_schur_vector_exchange_plus_tree_allreduce",
  }


def estimate_dci_compact_pcg_solve_communication(rotation_stats: dict,
                                                 translation_stats: dict,
                                                 robot_count: int):
  rot_reductions = int(rotation_stats.get("global_reduction_count", 0))
  trans_reductions = int(translation_stats.get("global_reduction_count", 0))
  rot_bytes = _scalar_allreduce_bytes(rot_reductions, robot_count)
  trans_bytes = _scalar_allreduce_bytes(trans_reductions, robot_count)
  reduction_total_bytes = rot_bytes + trans_bytes
  return {
      "robot_count": int(robot_count),
      "rotation_global_reduction_count": rot_reductions,
      "translation_global_reduction_count": trans_reductions,
      "rotation_global_reduction_bytes": rot_bytes,
      "translation_global_reduction_bytes": trans_bytes,
      "rotation_separator_exchange_bytes": 0,
      "translation_separator_exchange_bytes": 0,
      "separator_exchange_bytes": 0,
      "pcg_global_reduction_bytes": reduction_total_bytes,
      "rotation_global_reduction_mb": _bytes_to_mb(rot_bytes),
      "translation_global_reduction_mb": _bytes_to_mb(trans_bytes),
      "rotation_separator_exchange_mb": 0.0,
      "translation_separator_exchange_mb": 0.0,
      "separator_exchange_mb": 0.0,
      "pcg_global_reduction_mb": _bytes_to_mb(reduction_total_bytes),
      "linear_solve_total_estimated_bytes": reduction_total_bytes,
      "linear_solve_total_estimated_mb": _bytes_to_mb(reduction_total_bytes),
      "model": "normal_summary_compact_pcg_global_reductions",
  }


def estimate_separator_normal_equation_summary_communication(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    rotations: dict[int, np.ndarray],
    dim: int,
    weighted: bool,
    cost_mode: str,
    anchor_pose: int):
  separator_edges = [
      edge for edge in graph_edges
      if edge.i in robot_of and edge.j in robot_of and
      robot_of[edge.i] != robot_of[edge.j]
  ]
  if not separator_edges:
    return {
        "separator_edge_count": 0,
        "rotation_summary_mb": 0.0,
        "translation_summary_mb": 0.0,
        "total_summary_mb": 0.0,
        "rotation_summary_bytes": 0,
        "translation_summary_bytes": 0,
        "total_summary_bytes": 0,
        "model": "separator_dense_block_normal_equation_summary",
    }
  rot_matrix, rot_rhs, rot_meta = assemble_rotation_system(
      separator_edges, pose_ids, dim, weighted, cost_mode, anchor_pose)
  rot_summary = normal_equation_block_summary(
      rot_matrix, rot_rhs, rot_meta["block_dim"])
  trans_matrix, trans_rhs, trans_meta = assemble_translation_system(
      separator_edges, pose_ids, rotations, dim, weighted, cost_mode,
      anchor_pose)
  trans_summary = normal_equation_block_summary(
      trans_matrix, trans_rhs, trans_meta["block_dim"])
  total_bytes = int(rot_summary["payload_bytes"]) + int(
      trans_summary["payload_bytes"])
  return {
      "separator_edge_count": len(separator_edges),
      "rotation_summary": {
          key: value for key, value in rot_summary.items()
          if key not in {"hessian_blocks", "gradient_blocks"}
      },
      "translation_summary": {
          key: value for key, value in trans_summary.items()
          if key not in {"hessian_blocks", "gradient_blocks"}
      },
      "rotation_summary_bytes": int(rot_summary["payload_bytes"]),
      "translation_summary_bytes": int(trans_summary["payload_bytes"]),
      "total_summary_bytes": total_bytes,
      "rotation_summary_mb": _bytes_to_mb(int(rot_summary["payload_bytes"])),
      "translation_summary_mb": _bytes_to_mb(int(trans_summary["payload_bytes"])),
      "total_summary_mb": _bytes_to_mb(total_bytes),
      "model": "separator_dense_block_normal_equation_summary",
  }


def normal_summary_communication_from_summaries(separator_edge_count: int,
                                                rotation_summary: dict,
                                                translation_summary: dict):
  rot_bytes = int(rotation_summary.get("payload_bytes", 0))
  trans_bytes = int(translation_summary.get("payload_bytes", 0))
  total_bytes = rot_bytes + trans_bytes
  return {
      "separator_edge_count": int(separator_edge_count),
      "rotation_summary": {
          key: value for key, value in rotation_summary.items()
          if key not in {"hessian_blocks", "gradient_blocks"}
      },
      "translation_summary": {
          key: value for key, value in translation_summary.items()
          if key not in {"hessian_blocks", "gradient_blocks"}
      },
      "rotation_summary_bytes": rot_bytes,
      "translation_summary_bytes": trans_bytes,
      "total_summary_bytes": total_bytes,
      "rotation_summary_mb": _bytes_to_mb(rot_bytes),
      "translation_summary_mb": _bytes_to_mb(trans_bytes),
      "total_summary_mb": _bytes_to_mb(total_bytes),
      "model": "separator_dense_block_normal_equation_summary",
  }


def estimate_residual_certificate_communication(robot_count: int):
  bytes_count = _scalar_allreduce_bytes(1, robot_count)
  return {
      "robot_count": int(robot_count),
      "residual_delta_allreduce_count": 1 if robot_count > 1 else 0,
      "residual_delta_consensus_bytes": bytes_count,
      "residual_delta_consensus_mb": _bytes_to_mb(bytes_count),
      "model": "tree_allreduce_one_scalar_delta",
  }


def solve_distributed_chordal_initialization(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    rotation_iterations: int,
    translation_iterations: int,
    weighted: bool,
    cost_mode: str,
    anchor_pose: int | None = None,
    relaxation: float = 1.0,
    damping: float = 1e-12,
    linear_solver: str = "block_jacobi",
):
  pose_ids = sorted(pose_ids)
  if not pose_ids:
    return {}, {"pose_count": 0}
  anchor_pose = pose_ids[0] if anchor_pose is None else anchor_pose
  dim = pose_dimension_from_edges(graph_edges)

  rot_matrix, rot_rhs, rot_meta = assemble_rotation_system(
      graph_edges, pose_ids, dim, weighted, cost_mode, anchor_pose)
  rot_blocks = _robot_blocks_from_offsets(rot_meta["offsets"],
                                          rot_meta["block_dim"], robot_of)
  rot_solution, rot_stats = solve_distributed_normal_system(
      rot_matrix, rot_rhs, rot_blocks, rotation_iterations, linear_solver,
      relaxation, damping)
  rotations = _rotations_from_solution(rot_solution, pose_ids,
                                       rot_meta["offsets"], dim, anchor_pose)

  trans_matrix, trans_rhs, trans_meta = assemble_translation_system(
      graph_edges, pose_ids, rotations, dim, weighted, cost_mode, anchor_pose)
  trans_blocks = _robot_blocks_from_offsets(trans_meta["offsets"],
                                            trans_meta["block_dim"], robot_of)
  trans_solution, trans_stats = solve_distributed_normal_system(
      trans_matrix, trans_rhs, trans_blocks, translation_iterations,
      linear_solver, relaxation, damping)
  translations = _translations_from_solution(trans_solution, pose_ids,
                                             trans_meta["offsets"], dim,
                                             anchor_pose)
  poses = _poses_from_rotations_translations(pose_ids, rotations, translations,
                                             dim)
  robot_count = _robot_count_for_poses(pose_ids, robot_of)
  separator_count = _separator_edge_count(graph_edges, robot_of)
  normal_summary_comm = estimate_separator_normal_equation_summary_communication(
      graph_edges=graph_edges,
      pose_ids=pose_ids,
      robot_of=robot_of,
      rotations=rotations,
      dim=dim,
      weighted=weighted,
      cost_mode=cost_mode,
      anchor_pose=anchor_pose,
  )
  return poses, {
      "pose_count": len(pose_ids),
      "dimension": dim,
      "anchor_pose": anchor_pose,
      "separator_edge_count": separator_count,
      "rotation_stats": {**rot_stats, **{
          "equation_count": rot_meta["equation_count"],
          "variable_count": rot_meta["variable_count"],
      }},
      "translation_stats": {**trans_stats, **{
          "equation_count": trans_meta["equation_count"],
          "variable_count": trans_meta["variable_count"],
      }},
      "method": "distributed_chordal_linear_solve",
      "linear_solver": linear_solver,
      "communication_estimate": estimate_dci_linear_solve_communication(
          rot_stats, trans_stats, robot_count, separator_count, dim),
      "normal_equation_summary_communication": normal_summary_comm,
  }


def _pose_set_total_chordal_cost(graph_edges: list[Edge],
                                 poses: dict[int, np.ndarray],
                                 weighted: bool,
                                 cost_mode: str):
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
    total_cost = math.inf
  return {
      "total_cost": float(total_cost),
      "evaluated_edge_count": evaluated,
      "missing_edge_count": missing,
      "all_edges_evaluated": missing == 0,
  }


def _measurement_portfolio_communication(candidates: list[dict],
                                         robot_count: int):
  candidate_linear_bytes = 0
  for candidate in candidates:
    candidate_linear_bytes += int(
        candidate.get("communication_estimate", {}).get(
            "linear_solve_total_estimated_bytes", 0))
  certificate_bytes = _scalar_allreduce_bytes(len(candidates), robot_count)
  total_bytes = candidate_linear_bytes + certificate_bytes
  return {
      "robot_count": int(robot_count),
      "candidate_count": int(len(candidates)),
      "measurement_portfolio_candidate_linear_bytes": int(
          candidate_linear_bytes),
      "measurement_portfolio_candidate_linear_mb": _bytes_to_mb(
          candidate_linear_bytes),
      "measurement_portfolio_certificate_bytes": int(certificate_bytes),
      "measurement_portfolio_certificate_comm_mb": _bytes_to_mb(
          certificate_bytes),
      "linear_solve_total_estimated_bytes": int(total_bytes),
      "linear_solve_total_estimated_mb": _bytes_to_mb(total_bytes),
      "model":
          "measurement_portfolio_candidate_solves_plus_cost_allreduce",
  }


def _translation_budget_portfolio_communication(candidates: list[dict],
                                                robot_count: int):
  aggregate = _measurement_portfolio_communication(candidates, robot_count)
  return {
      "robot_count": aggregate["robot_count"],
      "candidate_count": aggregate["candidate_count"],
      "translation_budget_portfolio_candidate_count":
          aggregate["candidate_count"],
      "translation_budget_portfolio_candidate_linear_bytes":
          aggregate["measurement_portfolio_candidate_linear_bytes"],
      "translation_budget_portfolio_candidate_linear_mb":
          aggregate["measurement_portfolio_candidate_linear_mb"],
      "translation_budget_portfolio_certificate_bytes":
          aggregate["measurement_portfolio_certificate_bytes"],
      "translation_budget_portfolio_certificate_comm_mb":
          aggregate["measurement_portfolio_certificate_comm_mb"],
      "linear_solve_total_estimated_bytes":
          aggregate["linear_solve_total_estimated_bytes"],
      "linear_solve_total_estimated_mb":
          aggregate["linear_solve_total_estimated_mb"],
      "model":
          "translation_budget_candidate_solves_plus_cost_allreduce",
  }


def _coarse_measurement_portfolio_communication(proxy_candidates: list[dict],
                                                selected_communication: dict,
                                                robot_count: int):
  proxy_bytes = 0
  for candidate in proxy_candidates:
    proxy_bytes += int(
        candidate.get("communication_estimate", {}).get(
            "linear_solve_total_estimated_bytes", 0))
  selected_bytes = int(
      selected_communication.get("linear_solve_total_estimated_bytes", 0))
  certificate_bytes = _scalar_allreduce_bytes(len(proxy_candidates), robot_count)
  total_bytes = proxy_bytes + selected_bytes + certificate_bytes
  return {
      "robot_count": int(robot_count),
      "candidate_count": int(len(proxy_candidates)),
      "coarse_measurement_portfolio_proxy_bytes": int(proxy_bytes),
      "coarse_measurement_portfolio_proxy_mb": _bytes_to_mb(proxy_bytes),
      "coarse_measurement_portfolio_selected_full_solve_bytes": int(
          selected_bytes),
      "coarse_measurement_portfolio_selected_full_solve_mb": _bytes_to_mb(
          selected_bytes),
      "coarse_measurement_portfolio_certificate_bytes": int(certificate_bytes),
      "coarse_measurement_portfolio_certificate_comm_mb": _bytes_to_mb(
          certificate_bytes),
      "linear_solve_total_estimated_bytes": int(total_bytes),
      "linear_solve_total_estimated_mb": _bytes_to_mb(total_bytes),
      "model":
          "coarse_measurement_proxy_candidates_plus_selected_full_solve",
  }


def _partial_measurement_portfolio_communication(proxy_candidates: list[dict],
                                                 selected_communication: dict,
                                                 robot_count: int):
  aggregate = _coarse_measurement_portfolio_communication(
      proxy_candidates, selected_communication, robot_count)
  return {
      "robot_count": aggregate["robot_count"],
      "candidate_count": aggregate["candidate_count"],
      "partial_measurement_portfolio_proxy_bytes":
          aggregate["coarse_measurement_portfolio_proxy_bytes"],
      "partial_measurement_portfolio_proxy_mb":
          aggregate["coarse_measurement_portfolio_proxy_mb"],
      "partial_measurement_portfolio_selected_full_solve_bytes":
          aggregate["coarse_measurement_portfolio_selected_full_solve_bytes"],
      "partial_measurement_portfolio_selected_full_solve_mb":
          aggregate["coarse_measurement_portfolio_selected_full_solve_mb"],
      "partial_measurement_portfolio_certificate_bytes":
          aggregate["coarse_measurement_portfolio_certificate_bytes"],
      "partial_measurement_portfolio_certificate_comm_mb":
          aggregate["coarse_measurement_portfolio_certificate_comm_mb"],
      "linear_solve_total_estimated_bytes":
          aggregate["linear_solve_total_estimated_bytes"],
      "linear_solve_total_estimated_mb":
          aggregate["linear_solve_total_estimated_mb"],
      "model":
          "partial_measurement_proxy_candidates_plus_selected_full_solve",
  }


def solve_summary_chordal_initialization(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    rotation_iterations: int,
    translation_iterations: int,
    weighted: bool,
    cost_mode: str,
    anchor_pose: int | None = None,
    relaxation: float = 1.0,
    damping: float = 1e-12,
    linear_solver: str = "pcg",
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
    omitted_force_correction_curvature_comm_multiplier: float = 1.0,
    omitted_force_correction_min_marginal_cost_per_mb: float | None = None,
    central_equivalence_diagnostic: bool = False,
    central_equivalence_iterative_diagnostic_iterations: int = 0,
):
  allowed_interface_schur_preconditioners = {
      "diagonal",
      "block_jacobi",
      "block_jacobi+coarse",
      "block_jacobi+balanced_coarse",
      "overlap_schwarz",
  }
  if interface_schur_preconditioner not in allowed_interface_schur_preconditioners:
    raise ValueError(
        "unsupported interface_schur_preconditioner: "
        f"{interface_schur_preconditioner}")
  rotation_preconditioner = (
      interface_schur_preconditioner
      if interface_schur_rotation_preconditioner is None
      else interface_schur_rotation_preconditioner)
  translation_preconditioner = (
      interface_schur_preconditioner
      if interface_schur_translation_preconditioner is None
      else interface_schur_translation_preconditioner)
  if rotation_preconditioner not in allowed_interface_schur_preconditioners:
    raise ValueError(
        "unsupported interface_schur_rotation_preconditioner: "
        f"{rotation_preconditioner}")
  if translation_preconditioner not in allowed_interface_schur_preconditioners:
    raise ValueError(
        "unsupported interface_schur_translation_preconditioner: "
        f"{translation_preconditioner}")
  allowed_coarse_basis_modes = {
      "robot_coordinate",
      "component_coordinate",
      "robot_component_coordinate",
  }
  if interface_schur_coarse_basis_mode not in allowed_coarse_basis_modes:
    raise ValueError(
        "unsupported interface_schur_coarse_basis_mode: "
        f"{interface_schur_coarse_basis_mode}")
  rotation_coarse_basis_mode = (
      interface_schur_coarse_basis_mode
      if interface_schur_rotation_coarse_basis_mode is None
      else interface_schur_rotation_coarse_basis_mode)
  translation_coarse_basis_mode = (
      interface_schur_coarse_basis_mode
      if interface_schur_translation_coarse_basis_mode is None
      else interface_schur_translation_coarse_basis_mode)
  if rotation_coarse_basis_mode not in allowed_coarse_basis_modes:
    raise ValueError(
        "unsupported interface_schur_rotation_coarse_basis_mode: "
        f"{rotation_coarse_basis_mode}")
  if translation_coarse_basis_mode not in allowed_coarse_basis_modes:
    raise ValueError(
        "unsupported interface_schur_translation_coarse_basis_mode: "
        f"{translation_coarse_basis_mode}")
  if (interface_schur_coarse_component_limit is not None and
      int(interface_schur_coarse_component_limit) < 0):
    raise ValueError("interface_schur_coarse_component_limit must be nonnegative")
  rotation_coarse_component_limit = (
      interface_schur_coarse_component_limit
      if interface_schur_rotation_coarse_component_limit is None
      else interface_schur_rotation_coarse_component_limit)
  translation_coarse_component_limit = (
      interface_schur_coarse_component_limit
      if interface_schur_translation_coarse_component_limit is None
      else interface_schur_translation_coarse_component_limit)
  if (rotation_coarse_component_limit is not None and
      int(rotation_coarse_component_limit) < 0):
    raise ValueError(
        "interface_schur_rotation_coarse_component_limit must be nonnegative")
  if (translation_coarse_component_limit is not None and
      int(translation_coarse_component_limit) < 0):
    raise ValueError(
        "interface_schur_translation_coarse_component_limit must be nonnegative")
  allowed_component_selection_modes = {
      "size",
      "gradient_energy",
      "projected_merit",
  }
  if interface_schur_coarse_component_selection_mode not in (
      allowed_component_selection_modes):
    raise ValueError(
        "unsupported interface_schur_coarse_component_selection_mode: "
        f"{interface_schur_coarse_component_selection_mode}")
  rotation_coarse_component_selection_mode = (
      interface_schur_coarse_component_selection_mode
      if interface_schur_rotation_coarse_component_selection_mode is None
      else interface_schur_rotation_coarse_component_selection_mode)
  translation_coarse_component_selection_mode = (
      interface_schur_coarse_component_selection_mode
      if interface_schur_translation_coarse_component_selection_mode is None
      else interface_schur_translation_coarse_component_selection_mode)
  if rotation_coarse_component_selection_mode not in allowed_component_selection_modes:
    raise ValueError(
        "unsupported interface_schur_rotation_coarse_component_selection_mode: "
        f"{rotation_coarse_component_selection_mode}")
  if translation_coarse_component_selection_mode not in allowed_component_selection_modes:
    raise ValueError(
        "unsupported interface_schur_translation_coarse_component_selection_mode: "
        f"{translation_coarse_component_selection_mode}")
  if int(interface_schur_residual_deflation_rank) < 0:
    raise ValueError("interface_schur_residual_deflation_rank must be nonnegative")
  if int(interface_schur_residual_deflation_pilot_iterations) < 0:
    raise ValueError(
        "interface_schur_residual_deflation_pilot_iterations must be nonnegative")
  if int(interface_schur_ritz_rank) < 0:
    raise ValueError("interface_schur_ritz_rank must be nonnegative")
  if int(interface_schur_ritz_probe_iterations) < 0:
    raise ValueError("interface_schur_ritz_probe_iterations must be nonnegative")
  if (interface_schur_rotation_ritz_rank is not None and
      int(interface_schur_rotation_ritz_rank) < 0):
    raise ValueError("interface_schur_rotation_ritz_rank must be nonnegative")
  if (interface_schur_translation_ritz_rank is not None and
      int(interface_schur_translation_ritz_rank) < 0):
    raise ValueError(
        "interface_schur_translation_ritz_rank must be nonnegative")
  if (interface_schur_rotation_ritz_probe_iterations is not None and
      int(interface_schur_rotation_ritz_probe_iterations) < 0):
    raise ValueError(
        "interface_schur_rotation_ritz_probe_iterations must be nonnegative")
  if (interface_schur_translation_ritz_probe_iterations is not None and
      int(interface_schur_translation_ritz_probe_iterations) < 0):
    raise ValueError(
        "interface_schur_translation_ritz_probe_iterations must be nonnegative")
  for label, mode, threshold, energy_fraction in [
      (
          "interface_schur_rotation_ritz_rank_selection_mode",
          interface_schur_rotation_ritz_rank_selection_mode,
          interface_schur_rotation_ritz_value_threshold,
          interface_schur_rotation_ritz_energy_capture_fraction),
      (
          "interface_schur_translation_ritz_rank_selection_mode",
          interface_schur_translation_ritz_rank_selection_mode,
          interface_schur_translation_ritz_value_threshold,
          interface_schur_translation_ritz_energy_capture_fraction),
  ]:
    if mode not in {"fixed", "value_threshold", "energy_capture"}:
      raise ValueError(f"unsupported {label}: {mode}")
    if mode == "value_threshold" and (
        threshold is None or not np.isfinite(float(threshold))):
      raise ValueError(f"{label} requires a finite value threshold")
    if mode == "energy_capture":
      if (energy_fraction is None or
          not np.isfinite(float(energy_fraction))):
        raise ValueError(f"{label} requires a finite energy fraction")
      if float(energy_fraction) < 0.0 or float(energy_fraction) > 1.0:
        raise ValueError(f"{label} energy fraction must be in [0, 1]")
  rotation_ritz_rank = int(interface_schur_ritz_rank)
  translation_ritz_rank = int(interface_schur_ritz_rank)
  if interface_schur_rotation_ritz_rank is not None:
    rotation_ritz_rank = int(interface_schur_rotation_ritz_rank)
  if interface_schur_translation_ritz_rank is not None:
    translation_ritz_rank = int(interface_schur_translation_ritz_rank)
  rotation_ritz_probe_iterations = int(interface_schur_ritz_probe_iterations)
  translation_ritz_probe_iterations = int(interface_schur_ritz_probe_iterations)
  if interface_schur_rotation_ritz_probe_iterations is not None:
    rotation_ritz_probe_iterations = int(
        interface_schur_rotation_ritz_probe_iterations)
  if interface_schur_translation_ritz_probe_iterations is not None:
    translation_ritz_probe_iterations = int(
        interface_schur_translation_ritz_probe_iterations)
  if interface_schur_ritz_mode not in {
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
  }:
    raise ValueError(
        "unsupported interface_schur_ritz_mode: "
        f"{interface_schur_ritz_mode}")
  if omitted_force_correction_curvature_model not in {
      "none",
      "block_gershgorin",
      "directional_secant",
      "subspace_secant",
  }:
    raise ValueError(
        "unsupported omitted_force_correction_curvature_model: "
        f"{omitted_force_correction_curvature_model}")
  if omitted_force_correction_curvature_payload_model not in {
      "global_projected",
      "pairwise_projected",
  }:
    raise ValueError(
        "unsupported omitted_force_correction_curvature_payload_model: "
        f"{omitted_force_correction_curvature_payload_model}")
  if omitted_force_correction_curvature_rank_scheduler not in {
      "fixed",
      "projected_residual",
  }:
    raise ValueError(
        "unsupported omitted_force_correction_curvature_rank_scheduler: "
        f"{omitted_force_correction_curvature_rank_scheduler}")
  if omitted_force_correction_curvature_rank_condition_policy not in {
      "fixed",
      "median_mad",
  }:
    raise ValueError(
        "unsupported omitted_force_correction_curvature_rank_condition_policy: "
        f"{omitted_force_correction_curvature_rank_condition_policy}")
  if float(omitted_force_correction_curvature_rank_condition_mad_scale) < 0.0:
    raise ValueError(
        "omitted_force_correction_curvature_rank_condition_mad_scale "
        "must be nonnegative")
  if summary_selection_mode not in {
      "all",
      "structural_spanning",
      "residual_force_refinement",
      "omitted_force_correction",
  }:
    raise ValueError(f"unsupported summary_selection_mode: {summary_selection_mode}")
  if (omitted_force_correction_max_comm_mb is not None and
      float(omitted_force_correction_max_comm_mb) < 0.0):
    raise ValueError("omitted_force_correction_max_comm_mb must be nonnegative")
  if (omitted_force_correction_min_marginal_cost_per_mb is not None and
      float(omitted_force_correction_min_marginal_cost_per_mb) < 0.0):
    raise ValueError(
        "omitted_force_correction_min_marginal_cost_per_mb must be nonnegative")
  curvature_comm_multiplier = max(
      0.0, float(omitted_force_correction_curvature_comm_multiplier))
  min_marginal_cost_per_mb = (
      None if omitted_force_correction_min_marginal_cost_per_mb is None
      else float(omitted_force_correction_min_marginal_cost_per_mb))
  pose_ids = sorted(pose_ids)
  if not pose_ids:
    return {}, {"pose_count": 0}
  if interface_schur_translation_budget_candidates is not None:
    if interface_schur_translation_ritz_rank_selection_mode != "fixed":
      raise ValueError(
          "translation budget portfolio cannot also use translation Ritz "
          "rank selection")
    budget_candidates = []
    for candidate in interface_schur_translation_budget_candidates:
      if len(candidate) != 2:
        raise ValueError(
            "translation budget candidates must be (rank, probe_iterations)")
      rank = int(candidate[0])
      probe_iterations = int(candidate[1])
      if rank < 0 or probe_iterations < 0:
        raise ValueError("translation budget candidates must be nonnegative")
      budget_candidates.append((rank, probe_iterations))
    if not budget_candidates:
      raise ValueError("translation budget candidate list must be nonempty")
    if interface_schur_ritz_mode in {
        "measurement_portfolio",
        "coarse_measurement_portfolio",
        "partial_measurement_portfolio",
    }:
      raise ValueError(
          "translation budget portfolio cannot wrap measurement portfolios")

    candidates = []
    best_index = 0
    best_cost = math.inf
    best_comm_mb = math.inf
    robot_count = _robot_count_for_poses(pose_ids, robot_of)
    for rank, probe_iterations in budget_candidates:
      candidate_poses, candidate_stats = solve_summary_chordal_initialization(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          rotation_iterations=rotation_iterations,
          translation_iterations=translation_iterations,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
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
          interface_schur_coarse_basis_mode=(
              interface_schur_coarse_basis_mode),
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
          interface_schur_translation_ritz_rank=rank,
          interface_schur_rotation_ritz_probe_iterations=(
              interface_schur_rotation_ritz_probe_iterations),
          interface_schur_translation_ritz_probe_iterations=probe_iterations,
          interface_schur_ritz_mode=interface_schur_ritz_mode,
          summary_selection_mode=summary_selection_mode,
          summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
          summary_refinement_max_offdiag_block_edges=(
              summary_refinement_max_offdiag_block_edges),
          omitted_force_correction_rounds=omitted_force_correction_rounds,
          omitted_force_correction_portfolio=False,
          omitted_force_correction_rank_portfolio=False,
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
              curvature_comm_multiplier),
          omitted_force_correction_min_marginal_cost_per_mb=(
              omitted_force_correction_min_marginal_cost_per_mb),
      )
      measurement = _pose_set_total_chordal_cost(
          graph_edges, candidate_poses, weighted, cost_mode)
      communication = candidate_stats.get("communication_estimate", {})
      comm_mb = float(
          communication.get("linear_solve_total_estimated_mb", math.inf))
      candidate = {
          "rank": int(rank),
          "probe_iterations": int(probe_iterations),
          "measurement_cost": float(measurement["total_cost"]),
          "all_edges_evaluated": bool(measurement["all_edges_evaluated"]),
          "evaluated_edge_count": int(measurement["evaluated_edge_count"]),
          "missing_edge_count": int(measurement["missing_edge_count"]),
          "linear_solve_comm_mb": comm_mb,
          "communication_estimate": communication,
          "rotation_residual": float(
              candidate_stats.get("rotation_stats", {}).get(
                  "final_normal_residual", math.inf)),
          "translation_residual": float(
              candidate_stats.get("translation_stats", {}).get(
                  "final_normal_residual", math.inf)),
          "stats": candidate_stats,
          "poses": candidate_poses,
      }
      candidates.append(candidate)
      cost = float(candidate["measurement_cost"])
      if (cost < best_cost - 1e-12 or
          (abs(cost - best_cost) <= 1e-12 and comm_mb < best_comm_mb)):
        best_index = len(candidates) - 1
        best_cost = cost
        best_comm_mb = comm_mb

    selected = candidates[best_index]
    selected_stats = dict(selected["stats"])
    selected_poses = selected["poses"]
    aggregate_communication = _translation_budget_portfolio_communication(
        candidates, robot_count)
    candidate_records = [
        {key: value for key, value in candidate.items()
         if key not in {"communication_estimate", "stats", "poses"}}
        for candidate in candidates
    ]
    selected_stats["communication_estimate"] = aggregate_communication
    selected_stats["translation_budget_portfolio_communication_estimate"] = (
        aggregate_communication)
    selected_stats["translation_budget_portfolio_selected_candidate_communication"] = (
        selected["communication_estimate"])
    selected_stats["translation_budget_portfolio"] = {
        "selection_rule": "minimum_full_measurement_cost_then_candidate_comm",
        "candidate_count": int(len(candidates)),
        "candidates": candidate_records,
        "selected_index": int(best_index),
        "selected_rank": int(selected["rank"]),
        "selected_probe_iterations": int(selected["probe_iterations"]),
        "selected_measurement_cost": float(selected["measurement_cost"]),
        "selected_linear_solve_comm_mb": float(selected["linear_solve_comm_mb"]),
        "candidate_measurement_costs": [
            float(candidate["measurement_cost"]) for candidate in candidates
        ],
        "candidate_linear_solve_comm_mb": [
            float(candidate["linear_solve_comm_mb"]) for candidate in candidates
        ],
        "communication_estimate": aggregate_communication,
        "selected_candidate_communication_estimate": (
            selected["communication_estimate"]),
    }
    return selected_poses, selected_stats
  if interface_schur_ritz_mode == "partial_measurement_portfolio":
    candidate_modes = [
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    ]
    proxy_rotation_iterations = min(max(int(rotation_iterations), 0), 1)
    proxy_translation_iterations = min(max(int(translation_iterations), 0), 1)
    proxy_candidates = []
    best_index = 0
    best_proxy_cost = math.inf
    best_proxy_comm_mb = math.inf
    robot_count = _robot_count_for_poses(pose_ids, robot_of)
    for mode in candidate_modes:
      proxy_poses, proxy_stats = solve_summary_chordal_initialization(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          rotation_iterations=proxy_rotation_iterations,
          translation_iterations=proxy_translation_iterations,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          relaxation=relaxation,
          damping=damping,
          linear_solver=linear_solver,
          interface_schur_preconditioner=interface_schur_preconditioner,
          interface_schur_rotation_preconditioner=(
              interface_schur_rotation_preconditioner),
          interface_schur_translation_preconditioner=(
              interface_schur_translation_preconditioner),
          interface_schur_coarse_initial_guess=True,
          interface_schur_coarse_basis_mode=(
              interface_schur_coarse_basis_mode),
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
          interface_schur_ritz_mode=mode,
          summary_selection_mode=summary_selection_mode,
          summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
          summary_refinement_max_offdiag_block_edges=(
              summary_refinement_max_offdiag_block_edges),
          omitted_force_correction_rounds=omitted_force_correction_rounds,
          omitted_force_correction_portfolio=False,
          omitted_force_correction_rank_portfolio=False,
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
              curvature_comm_multiplier),
          omitted_force_correction_min_marginal_cost_per_mb=(
              omitted_force_correction_min_marginal_cost_per_mb),
      )
      proxy_measurement = _pose_set_total_chordal_cost(
          graph_edges, proxy_poses, weighted, cost_mode)
      proxy_communication = proxy_stats.get("communication_estimate", {})
      proxy_comm_mb = float(
          proxy_communication.get("linear_solve_total_estimated_mb", math.inf))
      candidate = {
          "mode": mode,
          "proxy_measurement_cost": float(proxy_measurement["total_cost"]),
          "all_edges_evaluated": bool(proxy_measurement["all_edges_evaluated"]),
          "evaluated_edge_count": int(proxy_measurement["evaluated_edge_count"]),
          "missing_edge_count": int(proxy_measurement["missing_edge_count"]),
          "proxy_linear_solve_comm_mb": proxy_comm_mb,
          "communication_estimate": proxy_communication,
          "rotation_proxy_residual": float(
              proxy_stats.get("rotation_stats", {}).get(
                  "final_normal_residual", math.inf)),
          "translation_proxy_residual": float(
              proxy_stats.get("translation_stats", {}).get(
                  "final_normal_residual", math.inf)),
      }
      proxy_candidates.append(candidate)
      proxy_cost = float(candidate["proxy_measurement_cost"])
      if (proxy_cost < best_proxy_cost - 1e-12 or
          (abs(proxy_cost - best_proxy_cost) <= 1e-12 and
           proxy_comm_mb < best_proxy_comm_mb)):
        best_index = len(proxy_candidates) - 1
        best_proxy_cost = proxy_cost
        best_proxy_comm_mb = proxy_comm_mb

    selected_mode = str(proxy_candidates[best_index]["mode"])
    selected_poses, selected_stats = solve_summary_chordal_initialization(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
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
        interface_schur_coarse_basis_mode=(
            interface_schur_coarse_basis_mode),
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
        interface_schur_ritz_mode=selected_mode,
        summary_selection_mode=summary_selection_mode,
        summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
        summary_refinement_max_offdiag_block_edges=(
            summary_refinement_max_offdiag_block_edges),
        omitted_force_correction_rounds=omitted_force_correction_rounds,
        omitted_force_correction_portfolio=False,
        omitted_force_correction_rank_portfolio=False,
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
            curvature_comm_multiplier),
        omitted_force_correction_min_marginal_cost_per_mb=(
            omitted_force_correction_min_marginal_cost_per_mb),
    )
    selected_communication = selected_stats.get("communication_estimate", {})
    selected_measurement = _pose_set_total_chordal_cost(
        graph_edges, selected_poses, weighted, cost_mode)
    aggregate_communication = _partial_measurement_portfolio_communication(
        proxy_candidates, selected_communication, robot_count)
    proxy_records = [
        {key: value for key, value in candidate.items()
         if key != "communication_estimate"}
        for candidate in proxy_candidates
    ]
    selected_stats = dict(selected_stats)
    selected_stats["interface_schur_ritz_mode"] = (
        "partial_measurement_portfolio")
    selected_stats["partial_measurement_portfolio_selected_mode"] = (
        selected_mode)
    selected_stats["partial_measurement_portfolio_selected_index"] = int(
        best_index)
    selected_stats["partial_measurement_portfolio_selected_proxy_cost"] = float(
        best_proxy_cost)
    selected_stats["partial_measurement_portfolio_selected_full_cost"] = float(
        selected_measurement["total_cost"])
    selected_stats["partial_measurement_portfolio_candidate_count"] = int(
        len(proxy_candidates))
    selected_stats["partial_measurement_portfolio_proxy_rotation_iterations"] = (
        int(proxy_rotation_iterations))
    selected_stats[
        "partial_measurement_portfolio_proxy_translation_iterations"] = int(
            proxy_translation_iterations)
    selected_stats["partial_measurement_portfolio_proxy_comm_mb"] = float(
        aggregate_communication["partial_measurement_portfolio_proxy_mb"])
    selected_stats["partial_measurement_portfolio_selected_full_comm_mb"] = float(
        aggregate_communication[
            "partial_measurement_portfolio_selected_full_solve_mb"])
    selected_stats["partial_measurement_portfolio_communication_estimate"] = (
        aggregate_communication)
    selected_stats["partial_measurement_portfolio_selected_candidate_communication"] = (
        selected_communication)
    selected_stats["communication_estimate"] = aggregate_communication
    selected_stats["partial_measurement_portfolio"] = {
        "selection_rule":
            "minimum_partial_refinement_measurement_cost_then_proxy_comm",
        "candidate_modes": candidate_modes,
        "candidate_count": int(len(proxy_candidates)),
        "selected_mode": selected_mode,
        "selected_index": int(best_index),
        "selected_proxy_cost": float(best_proxy_cost),
        "selected_full_cost": float(selected_measurement["total_cost"]),
        "selected_proxy_linear_solve_comm_mb": float(best_proxy_comm_mb),
        "proxy_rotation_iterations": int(proxy_rotation_iterations),
        "proxy_translation_iterations": int(proxy_translation_iterations),
        "candidate_proxy_costs": [
            float(candidate["proxy_measurement_cost"])
            for candidate in proxy_candidates
        ],
        "candidate_proxy_linear_solve_comm_mb": [
            float(candidate["proxy_linear_solve_comm_mb"])
            for candidate in proxy_candidates
        ],
        "candidates": proxy_records,
        "communication_estimate": aggregate_communication,
        "selected_candidate_communication_estimate": selected_communication,
    }
    return selected_poses, selected_stats
  if interface_schur_ritz_mode == "coarse_measurement_portfolio":
    candidate_modes = [
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    ]
    proxy_candidates = []
    best_index = 0
    best_proxy_cost = math.inf
    best_proxy_comm_mb = math.inf
    robot_count = _robot_count_for_poses(pose_ids, robot_of)
    for mode in candidate_modes:
      proxy_poses, proxy_stats = solve_summary_chordal_initialization(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          rotation_iterations=0,
          translation_iterations=0,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
          relaxation=relaxation,
          damping=damping,
          linear_solver=linear_solver,
          interface_schur_preconditioner=interface_schur_preconditioner,
          interface_schur_rotation_preconditioner=(
              interface_schur_rotation_preconditioner),
          interface_schur_translation_preconditioner=(
              interface_schur_translation_preconditioner),
          interface_schur_coarse_initial_guess=True,
          interface_schur_coarse_basis_mode=(
              interface_schur_coarse_basis_mode),
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
          interface_schur_ritz_mode=mode,
          summary_selection_mode=summary_selection_mode,
          summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
          summary_refinement_max_offdiag_block_edges=(
              summary_refinement_max_offdiag_block_edges),
          omitted_force_correction_rounds=omitted_force_correction_rounds,
          omitted_force_correction_portfolio=False,
          omitted_force_correction_rank_portfolio=False,
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
              curvature_comm_multiplier),
          omitted_force_correction_min_marginal_cost_per_mb=(
              omitted_force_correction_min_marginal_cost_per_mb),
      )
      proxy_measurement = _pose_set_total_chordal_cost(
          graph_edges, proxy_poses, weighted, cost_mode)
      proxy_communication = proxy_stats.get("communication_estimate", {})
      proxy_comm_mb = float(
          proxy_communication.get("linear_solve_total_estimated_mb", math.inf))
      candidate = {
          "mode": mode,
          "proxy_measurement_cost": float(proxy_measurement["total_cost"]),
          "all_edges_evaluated": bool(proxy_measurement["all_edges_evaluated"]),
          "evaluated_edge_count": int(proxy_measurement["evaluated_edge_count"]),
          "missing_edge_count": int(proxy_measurement["missing_edge_count"]),
          "proxy_linear_solve_comm_mb": proxy_comm_mb,
          "communication_estimate": proxy_communication,
          "rotation_proxy_residual": float(
              proxy_stats.get("rotation_stats", {}).get(
                  "final_normal_residual", math.inf)),
          "translation_proxy_residual": float(
              proxy_stats.get("translation_stats", {}).get(
                  "final_normal_residual", math.inf)),
      }
      proxy_candidates.append(candidate)
      proxy_cost = float(candidate["proxy_measurement_cost"])
      if (proxy_cost < best_proxy_cost - 1e-12 or
          (abs(proxy_cost - best_proxy_cost) <= 1e-12 and
           proxy_comm_mb < best_proxy_comm_mb)):
        best_index = len(proxy_candidates) - 1
        best_proxy_cost = proxy_cost
        best_proxy_comm_mb = proxy_comm_mb

    selected_mode = str(proxy_candidates[best_index]["mode"])
    selected_poses, selected_stats = solve_summary_chordal_initialization(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
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
        interface_schur_coarse_basis_mode=(
            interface_schur_coarse_basis_mode),
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
        interface_schur_ritz_mode=selected_mode,
        summary_selection_mode=summary_selection_mode,
        summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
        summary_refinement_max_offdiag_block_edges=(
            summary_refinement_max_offdiag_block_edges),
        omitted_force_correction_rounds=omitted_force_correction_rounds,
        omitted_force_correction_portfolio=False,
        omitted_force_correction_rank_portfolio=False,
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
            curvature_comm_multiplier),
        omitted_force_correction_min_marginal_cost_per_mb=(
            omitted_force_correction_min_marginal_cost_per_mb),
    )
    selected_communication = selected_stats.get("communication_estimate", {})
    selected_measurement = _pose_set_total_chordal_cost(
        graph_edges, selected_poses, weighted, cost_mode)
    aggregate_communication = _coarse_measurement_portfolio_communication(
        proxy_candidates, selected_communication, robot_count)
    proxy_records = [
        {key: value for key, value in candidate.items()
         if key != "communication_estimate"}
        for candidate in proxy_candidates
    ]
    selected_stats = dict(selected_stats)
    selected_stats["interface_schur_ritz_mode"] = (
        "coarse_measurement_portfolio")
    selected_stats["coarse_measurement_portfolio_selected_mode"] = (
        selected_mode)
    selected_stats["coarse_measurement_portfolio_selected_index"] = int(
        best_index)
    selected_stats["coarse_measurement_portfolio_selected_proxy_cost"] = float(
        best_proxy_cost)
    selected_stats["coarse_measurement_portfolio_selected_full_cost"] = float(
        selected_measurement["total_cost"])
    selected_stats["coarse_measurement_portfolio_candidate_count"] = int(
        len(proxy_candidates))
    selected_stats["coarse_measurement_portfolio_proxy_comm_mb"] = float(
        aggregate_communication["coarse_measurement_portfolio_proxy_mb"])
    selected_stats["coarse_measurement_portfolio_selected_full_comm_mb"] = float(
        aggregate_communication[
            "coarse_measurement_portfolio_selected_full_solve_mb"])
    selected_stats["coarse_measurement_portfolio_communication_estimate"] = (
        aggregate_communication)
    selected_stats["coarse_measurement_portfolio_selected_candidate_communication"] = (
        selected_communication)
    selected_stats["communication_estimate"] = aggregate_communication
    selected_stats["coarse_measurement_portfolio"] = {
        "selection_rule":
            "minimum_zero_iteration_coarse_measurement_cost_then_proxy_comm",
        "candidate_modes": candidate_modes,
        "candidate_count": int(len(proxy_candidates)),
        "selected_mode": selected_mode,
        "selected_index": int(best_index),
        "selected_proxy_cost": float(best_proxy_cost),
        "selected_full_cost": float(selected_measurement["total_cost"]),
        "selected_proxy_linear_solve_comm_mb": float(best_proxy_comm_mb),
        "candidate_proxy_costs": [
            float(candidate["proxy_measurement_cost"])
            for candidate in proxy_candidates
        ],
        "candidate_proxy_linear_solve_comm_mb": [
            float(candidate["proxy_linear_solve_comm_mb"])
            for candidate in proxy_candidates
        ],
        "candidates": proxy_records,
        "communication_estimate": aggregate_communication,
        "selected_candidate_communication_estimate": selected_communication,
    }
    return selected_poses, selected_stats
  if interface_schur_ritz_mode == "measurement_portfolio":
    candidate_modes = [
        "preconditioned_operator",
        "generalized",
        "m_orthogonal_lanczos",
        "harmonic",
    ]
    candidates = []
    best_index = 0
    best_cost = math.inf
    best_comm_mb = math.inf
    best_poses = None
    best_stats = None
    robot_count = _robot_count_for_poses(pose_ids, robot_of)
    for mode in candidate_modes:
      candidate_poses, candidate_stats = solve_summary_chordal_initialization(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          rotation_iterations=rotation_iterations,
          translation_iterations=translation_iterations,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
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
          interface_schur_coarse_basis_mode=(
              interface_schur_coarse_basis_mode),
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
          interface_schur_ritz_mode=mode,
          summary_selection_mode=summary_selection_mode,
          summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
          summary_refinement_max_offdiag_block_edges=(
              summary_refinement_max_offdiag_block_edges),
          omitted_force_correction_rounds=omitted_force_correction_rounds,
          omitted_force_correction_portfolio=False,
          omitted_force_correction_rank_portfolio=False,
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
              curvature_comm_multiplier),
          omitted_force_correction_min_marginal_cost_per_mb=(
              omitted_force_correction_min_marginal_cost_per_mb),
      )
      measurement = _pose_set_total_chordal_cost(
          graph_edges, candidate_poses, weighted, cost_mode)
      communication = candidate_stats.get("communication_estimate", {})
      comm_mb = float(
          communication.get("linear_solve_total_estimated_mb", math.inf))
      candidate = {
          "mode": mode,
          "measurement_cost": float(measurement["total_cost"]),
          "all_edges_evaluated": bool(measurement["all_edges_evaluated"]),
          "evaluated_edge_count": int(measurement["evaluated_edge_count"]),
          "missing_edge_count": int(measurement["missing_edge_count"]),
          "linear_solve_comm_mb": comm_mb,
          "communication_estimate": communication,
          "rotation_final_normal_residual": float(
              candidate_stats.get("rotation_stats", {}).get(
                  "final_normal_residual", math.inf)),
          "translation_final_normal_residual": float(
              candidate_stats.get("translation_stats", {}).get(
                  "final_normal_residual", math.inf)),
      }
      candidates.append(candidate)
      measurement_cost = float(candidate["measurement_cost"])
      if (measurement_cost < best_cost - 1e-12 or
          (abs(measurement_cost - best_cost) <= 1e-12 and
           comm_mb < best_comm_mb)):
        best_index = len(candidates) - 1
        best_cost = measurement_cost
        best_comm_mb = comm_mb
        best_poses = candidate_poses
        best_stats = candidate_stats
    if best_poses is None or best_stats is None:
      raise RuntimeError("measurement portfolio failed to evaluate candidates")
    selected_mode = str(candidates[best_index]["mode"])
    selected_communication = best_stats.get("communication_estimate", {})
    aggregate_communication = _measurement_portfolio_communication(
        candidates, robot_count)
    candidate_records = [
        {key: value for key, value in candidate.items()
         if key != "communication_estimate"}
        for candidate in candidates
    ]
    best_stats = dict(best_stats)
    best_stats["interface_schur_ritz_mode"] = "measurement_portfolio"
    best_stats["measurement_portfolio_selected_mode"] = selected_mode
    best_stats["measurement_portfolio_selected_index"] = int(best_index)
    best_stats["measurement_portfolio_selected_cost"] = float(best_cost)
    best_stats["measurement_portfolio_candidate_count"] = int(len(candidates))
    best_stats["measurement_portfolio_certificate_comm_mb"] = float(
        aggregate_communication["measurement_portfolio_certificate_comm_mb"])
    best_stats["measurement_portfolio_selected_candidate_communication"] = (
        selected_communication)
    best_stats["measurement_portfolio_communication_estimate"] = (
        aggregate_communication)
    best_stats["communication_estimate"] = aggregate_communication
    best_stats["measurement_portfolio"] = {
        "selection_rule": "minimum_chordal_measurement_cost_then_linear_comm",
        "candidate_modes": candidate_modes,
        "candidate_count": int(len(candidates)),
        "selected_mode": selected_mode,
        "selected_index": int(best_index),
        "selected_cost": float(best_cost),
        "selected_linear_solve_comm_mb": float(best_comm_mb),
        "candidate_costs": [
            float(candidate["measurement_cost"]) for candidate in candidates
        ],
        "candidate_linear_solve_comm_mb": [
            float(candidate["linear_solve_comm_mb"]) for candidate in candidates
        ],
        "candidates": candidate_records,
        "communication_estimate": aggregate_communication,
        "selected_candidate_communication_estimate": selected_communication,
    }
    return best_poses, best_stats
  if (summary_selection_mode == "omitted_force_correction" and
      omitted_force_correction_portfolio and
      omitted_force_correction_rank_portfolio and
      omitted_force_correction_curvature_model == "subspace_secant" and
      int(omitted_force_correction_curvature_rank) > 1):
    max_rank = max(1, int(omitted_force_correction_curvature_rank))
    rank_candidates = []
    best_index = 0
    best_cost = math.inf
    best_comm_mb = math.inf
    best_poses = None
    best_stats = None
    for rank in range(1, max_rank + 1):
      candidate_poses, candidate_stats = solve_summary_chordal_initialization(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          rotation_iterations=rotation_iterations,
          translation_iterations=translation_iterations,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
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
          interface_schur_coarse_basis_mode=(
              interface_schur_coarse_basis_mode),
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
          interface_schur_ritz_mode=interface_schur_ritz_mode,
          summary_selection_mode=summary_selection_mode,
          summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
          summary_refinement_max_offdiag_block_edges=(
              summary_refinement_max_offdiag_block_edges),
          omitted_force_correction_rounds=omitted_force_correction_rounds,
          omitted_force_correction_portfolio=True,
          omitted_force_correction_rank_portfolio=False,
          omitted_force_correction_curvature_model=(
              omitted_force_correction_curvature_model),
          omitted_force_correction_curvature_rank=rank,
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
              curvature_comm_multiplier),
          omitted_force_correction_min_marginal_cost_per_mb=(
              omitted_force_correction_min_marginal_cost_per_mb),
      )
      portfolio = candidate_stats.get("omitted_force_correction_portfolio", {})
      measurement_cost = float(
          portfolio.get("selected_measurement_cost", math.inf))
      correction_comm_mb = float(
          portfolio.get("selected_correction_comm_mb", math.inf))
      candidate = {
          "rank": int(rank),
          "selected_rounds": int(portfolio.get("selected_rounds", 0)),
          "selected_measurement_cost": measurement_cost,
          "selected_correction_comm_mb": correction_comm_mb,
          "selected_omitted_force_comm_mb": float(
              portfolio.get("selected_omitted_force_comm_mb", 0.0)),
          "selected_curvature_comm_mb": float(
              portfolio.get("selected_curvature_comm_mb", 0.0)),
          "eligible_round_candidate_count": int(
              portfolio.get("eligible_candidate_count", 0)),
          "round_candidate_count": int(portfolio.get("candidate_count", 0)),
      }
      rank_candidates.append(candidate)
      if (measurement_cost < best_cost - 1e-12 or
          (abs(measurement_cost - best_cost) <= 1e-12 and
           correction_comm_mb < best_comm_mb)):
        best_index = len(rank_candidates) - 1
        best_cost = measurement_cost
        best_comm_mb = correction_comm_mb
        best_poses = candidate_poses
        best_stats = candidate_stats
    if best_poses is None or best_stats is None:
      raise RuntimeError("rank portfolio failed to evaluate any candidate")
    selected_rank = int(rank_candidates[best_index]["rank"])
    best_stats["omitted_force_correction_selected_curvature_rank"] = (
        selected_rank)
    portfolio = best_stats.get("omitted_force_correction_portfolio")
    if portfolio is not None:
      portfolio["selected_curvature_rank"] = selected_rank
      portfolio["requested_curvature_rank"] = max_rank
      portfolio["rank_portfolio_enabled"] = True
    best_stats["omitted_force_correction_rank_portfolio"] = {
        "enabled": True,
        "selection_rule":
            "minimum_rank_portfolio_measurement_cost_then_correction_comm",
        "candidate_count": len(rank_candidates),
        "rank_candidates": [int(item["rank"]) for item in rank_candidates],
        "requested_max_rank": max_rank,
        "selected_rank": selected_rank,
        "selected_index": best_index,
        "selected_measurement_cost":
            float(rank_candidates[best_index]["selected_measurement_cost"]),
        "selected_correction_comm_mb":
            float(rank_candidates[best_index]["selected_correction_comm_mb"]),
        "candidate_measurement_costs": [
            float(item["selected_measurement_cost"])
            for item in rank_candidates
        ],
        "candidate_correction_comm_mb": [
            float(item["selected_correction_comm_mb"])
            for item in rank_candidates
        ],
        "candidate_selected_rounds": [
            int(item["selected_rounds"]) for item in rank_candidates
        ],
        "candidates": rank_candidates,
    }
    return best_poses, best_stats
  if summary_selection_mode == "omitted_force_correction" and (
      omitted_force_correction_portfolio):
    max_rounds = max(0, int(omitted_force_correction_rounds))
    candidates = []
    best_index = 0
    best_cost = math.inf
    best_comm_mb = math.inf
    for rounds in range(max_rounds + 1):
      candidate_poses, candidate_stats = solve_summary_chordal_initialization(
          graph_edges=graph_edges,
          pose_ids=pose_ids,
          robot_of=robot_of,
          rotation_iterations=rotation_iterations,
          translation_iterations=translation_iterations,
          weighted=weighted,
          cost_mode=cost_mode,
          anchor_pose=anchor_pose,
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
          interface_schur_coarse_basis_mode=(
              interface_schur_coarse_basis_mode),
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
          interface_schur_ritz_mode=interface_schur_ritz_mode,
          summary_selection_mode=summary_selection_mode,
          summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
          summary_refinement_max_offdiag_block_edges=(
              summary_refinement_max_offdiag_block_edges),
          omitted_force_correction_rounds=rounds,
          omitted_force_correction_portfolio=False,
          omitted_force_correction_rank_portfolio=False,
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
      )
      measurement = _pose_set_total_chordal_cost(
          graph_edges, candidate_poses, weighted, cost_mode)
      correction_comm = candidate_stats.get(
          "omitted_force_correction", {}).get("communication_estimate", {})
      force_comm_mb = float(
          correction_comm.get("total_omitted_force_payload_mb", 0.0))
      curvature_comm_mb = float(
          correction_comm.get("total_curvature_payload_mb", 0.0))
      logical_comm_mb = float(
          correction_comm.get(
              "total_correction_payload_mb",
              force_comm_mb + curvature_comm_mb))
      comm_mb = force_comm_mb + curvature_comm_multiplier * curvature_comm_mb
      budget_feasible = (
          omitted_force_correction_max_comm_mb is None or
          comm_mb <= float(omitted_force_correction_max_comm_mb) + 1e-12)
      full_residual = float(
          candidate_stats.get("summary_model_representativeness", {}).get(
              "combined", {}).get("full_model_residual_norm", math.inf))
      candidate = {
          "rounds": rounds,
          "measurement_cost": measurement["total_cost"],
          "all_edges_evaluated": measurement["all_edges_evaluated"],
          "evaluated_edge_count": measurement["evaluated_edge_count"],
          "missing_edge_count": measurement["missing_edge_count"],
          "full_model_residual_norm": full_residual,
          "omitted_force_comm_mb": force_comm_mb,
          "curvature_comm_mb": curvature_comm_mb,
          "logical_correction_comm_mb": logical_comm_mb,
          "correction_comm_mb": comm_mb,
          "comm_budget_feasible": bool(budget_feasible),
      }
      candidates.append(candidate)
    marginal_improvements = [0.0]
    marginal_comm_mb = [0.0]
    marginal_value_per_mb = [math.inf]
    marginal_value_feasible = [True]
    prefix_feasible = True
    for index in range(1, len(candidates)):
      previous = candidates[index - 1]
      current_candidate = candidates[index]
      improvement = (
          float(previous["measurement_cost"]) -
          float(current_candidate["measurement_cost"]))
      incremental_comm = max(
          0.0,
          float(current_candidate["correction_comm_mb"]) -
          float(previous["correction_comm_mb"]))
      if incremental_comm > 1e-15:
        value_per_mb = improvement / incremental_comm
      elif improvement > 0.0:
        value_per_mb = math.inf
      else:
        value_per_mb = 0.0
      marginal_improvements.append(float(improvement))
      marginal_comm_mb.append(float(incremental_comm))
      marginal_value_per_mb.append(float(value_per_mb))
      if min_marginal_cost_per_mb is not None:
        prefix_feasible = (
            prefix_feasible and
            value_per_mb >= min_marginal_cost_per_mb - 1e-12)
      marginal_value_feasible.append(bool(prefix_feasible))
    for index, candidate in enumerate(candidates):
      candidate["marginal_cost_improvement"] = marginal_improvements[index]
      candidate["marginal_correction_comm_mb"] = marginal_comm_mb[index]
      candidate["marginal_cost_improvement_per_mb"] = marginal_value_per_mb[index]
      candidate["marginal_value_feasible"] = marginal_value_feasible[index]
    eligible_indices = [
        index for index, candidate in enumerate(candidates)
        if (bool(candidate.get("comm_budget_feasible", True)) and
            bool(candidate.get("marginal_value_feasible", True)))
    ]
    if not eligible_indices:
      eligible_indices = list(range(len(candidates)))
    best_index = eligible_indices[0]
    best_cost = math.inf
    best_comm_mb = math.inf
    for index in eligible_indices:
      candidate = candidates[index]
      cost = float(candidate["measurement_cost"])
      comm_mb = float(candidate["correction_comm_mb"])
      if (cost < best_cost - 1e-12 or
          (abs(cost - best_cost) <= 1e-12 and comm_mb < best_comm_mb)):
        best_index = index
        best_cost = cost
        best_comm_mb = comm_mb
    selected_rounds = int(candidates[best_index]["rounds"])
    best_poses, best_stats = solve_summary_chordal_initialization(
        graph_edges=graph_edges,
        pose_ids=pose_ids,
        robot_of=robot_of,
        rotation_iterations=rotation_iterations,
        translation_iterations=translation_iterations,
        weighted=weighted,
        cost_mode=cost_mode,
        anchor_pose=anchor_pose,
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
        interface_schur_coarse_basis_mode=(
            interface_schur_coarse_basis_mode),
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
        interface_schur_ritz_mode=interface_schur_ritz_mode,
        summary_selection_mode=summary_selection_mode,
        summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
        summary_refinement_max_offdiag_block_edges=(
            summary_refinement_max_offdiag_block_edges),
        omitted_force_correction_rounds=selected_rounds,
        omitted_force_correction_portfolio=False,
        omitted_force_correction_rank_portfolio=False,
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
    )
    best_stats["omitted_force_correction_requested_rounds"] = max_rounds
    best_stats["omitted_force_correction_portfolio"] = {
        "enabled": True,
        "selection_rule": (
            "minimum_dci_measurement_cost_under_comm_budget_then_comm"
            if omitted_force_correction_max_comm_mb is not None
            else "minimum_dci_measurement_cost_then_comm"),
        "comm_budget_enabled":
            omitted_force_correction_max_comm_mb is not None,
        "comm_budget_mb": (
            None if omitted_force_correction_max_comm_mb is None
            else float(omitted_force_correction_max_comm_mb)),
        "curvature_comm_multiplier": curvature_comm_multiplier,
        "marginal_value_gate_enabled": min_marginal_cost_per_mb is not None,
        "min_marginal_cost_improvement_per_mb": min_marginal_cost_per_mb,
        "eligible_candidate_count": len(eligible_indices),
        "candidate_count": len(candidates),
        "selected_rounds": selected_rounds,
        "requested_rounds": max_rounds,
        "selected_curvature_rank":
            int(omitted_force_correction_curvature_rank),
        "requested_curvature_rank":
            int(omitted_force_correction_curvature_rank),
        "curvature_rank_scheduler":
            omitted_force_correction_curvature_rank_scheduler,
        "curvature_rank_max_condition":
            omitted_force_correction_curvature_rank_max_condition,
        "curvature_rank_condition_policy":
            omitted_force_correction_curvature_rank_condition_policy,
        "curvature_rank_condition_mad_scale":
            float(omitted_force_correction_curvature_rank_condition_mad_scale),
        "diagnose_subspace_miss":
            bool(omitted_force_correction_diagnose_subspace_miss),
        "rank_portfolio_enabled": False,
        "selected_measurement_cost":
            float(candidates[best_index]["measurement_cost"]),
        "last_measurement_cost": float(candidates[-1]["measurement_cost"]),
        "selected_omitted_force_comm_mb":
            float(candidates[best_index]["omitted_force_comm_mb"]),
        "selected_curvature_comm_mb":
            float(candidates[best_index]["curvature_comm_mb"]),
        "selected_logical_correction_comm_mb":
            float(candidates[best_index]["logical_correction_comm_mb"]),
        "last_omitted_force_comm_mb":
            float(candidates[-1]["omitted_force_comm_mb"]),
        "last_curvature_comm_mb":
            float(candidates[-1]["curvature_comm_mb"]),
        "last_logical_correction_comm_mb":
            float(candidates[-1]["logical_correction_comm_mb"]),
        "selected_correction_comm_mb":
            float(candidates[best_index]["correction_comm_mb"]),
        "last_correction_comm_mb":
            float(candidates[-1]["correction_comm_mb"]),
        "candidate_measurement_costs": [
            float(item["measurement_cost"]) for item in candidates
        ],
        "candidate_full_model_residual_norms": [
            float(item["full_model_residual_norm"]) for item in candidates
        ],
        "candidate_omitted_force_comm_mb": [
            float(item["omitted_force_comm_mb"]) for item in candidates
        ],
        "candidate_curvature_comm_mb": [
            float(item["curvature_comm_mb"]) for item in candidates
        ],
        "candidate_logical_correction_comm_mb": [
            float(item["logical_correction_comm_mb"]) for item in candidates
        ],
        "candidate_correction_comm_mb": [
            float(item["correction_comm_mb"]) for item in candidates
        ],
        "candidate_comm_budget_feasible": [
            bool(item["comm_budget_feasible"]) for item in candidates
        ],
        "candidate_marginal_cost_improvement": [
            float(item["marginal_cost_improvement"]) for item in candidates
        ],
        "candidate_marginal_correction_comm_mb": [
            float(item["marginal_correction_comm_mb"]) for item in candidates
        ],
        "candidate_marginal_cost_improvement_per_mb": [
            float(item["marginal_cost_improvement_per_mb"])
            for item in candidates
        ],
        "candidate_marginal_value_feasible": [
            bool(item["marginal_value_feasible"]) for item in candidates
        ],
        "candidates": candidates,
    }
    return best_poses, best_stats
  anchor_pose = pose_ids[0] if anchor_pose is None else anchor_pose
  dim = pose_dimension_from_edges(graph_edges)
  private_edges, separator_edges = _split_private_separator_edges(
      graph_edges, robot_of)
  uses_rotation_interface_coarse_preconditioner = rotation_preconditioner in {
      "block_jacobi+coarse",
      "block_jacobi+balanced_coarse",
  }
  uses_translation_interface_coarse_preconditioner = translation_preconditioner in {
      "block_jacobi+coarse",
      "block_jacobi+balanced_coarse",
  }
  interface_pose_ids = {
      int(edge.i) for edge in separator_edges
      if edge.i in robot_of and edge.j in robot_of and robot_of[edge.i] != robot_of[edge.j]
  } | {
      int(edge.j) for edge in separator_edges
      if edge.i in robot_of and edge.j in robot_of and robot_of[edge.i] != robot_of[edge.j]
  }
  seed_selection_mode = (
      "structural_spanning"
      if summary_selection_mode in {
          "residual_force_refinement",
          "omitted_force_correction",
      }
      else summary_selection_mode)
  refinement_max_edges = (
      summary_max_offdiag_block_edges
      if summary_refinement_max_offdiag_block_edges is None
      else summary_refinement_max_offdiag_block_edges)
  rotation_ritz_mode = interface_schur_ritz_mode
  translation_ritz_mode = interface_schur_ritz_mode
  if interface_schur_ritz_mode == "rot_low_trans_harmonic":
    rotation_ritz_mode = "preconditioned_operator"
    translation_ritz_mode = "harmonic"
  elif interface_schur_ritz_mode == "rot_harmonic_trans_low":
    rotation_ritz_mode = "harmonic"
    translation_ritz_mode = "preconditioned_operator"

  (
      rot_hessian,
      rot_gradient,
      rot_meta,
      rot_separator_summary,
      full_rot_hessian,
      full_rot_gradient,
      full_rot_separator_summary,
  ) = (
      _summary_normal_equation_for_system(
          assemble_rotation_system,
          private_edges,
          separator_edges,
          pose_ids,
          dim,
          weighted,
          cost_mode,
          anchor_pose,
          summary_selection_mode=seed_selection_mode,
          summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
      ))
  rot_blocks = _robot_blocks_from_offsets(rot_meta["offsets"],
                                          rot_meta["block_dim"], robot_of)
  rot_interface_indices = _interface_variable_indices_from_offsets(
      rot_meta["offsets"], rot_meta["block_dim"], interface_pose_ids)
  rot_interface_blocks = _interface_variable_blocks_from_offsets(
      rot_meta["offsets"], rot_meta["block_dim"], interface_pose_ids)
  rot_interface_coarse_basis = None
  rot_coarse_info = {
      "coarse_basis_mode": rotation_coarse_basis_mode,
      "coarse_component_limit": (
          None if rotation_coarse_component_limit is None
          else int(rotation_coarse_component_limit)),
      "coarse_component_selection_mode":
          rotation_coarse_component_selection_mode,
      "robot_coordinate_basis_rank": 0,
      "component_coordinate_basis_rank": 0,
  }
  if uses_rotation_interface_coarse_preconditioner:
    rot_interface_coarse_basis, rot_coarse_info = _build_interface_coarse_basis(
        offsets=rot_meta["offsets"],
        block_dim=rot_meta["block_dim"],
        interface_pose_ids=interface_pose_ids,
        robot_of=robot_of,
        interface_indices=rot_interface_indices,
        separator_summary=rot_separator_summary,
        mode=rotation_coarse_basis_mode,
        component_limit=rotation_coarse_component_limit,
        component_selection_mode=rotation_coarse_component_selection_mode)
  if linear_solver == "interface_schur_pcg":
    rot_solution, rot_stats = interface_schur_hessian_solve(
        rot_hessian, rot_gradient, rot_interface_indices, rotation_iterations,
        damping, tolerance=1e-10, interface_blocks=rot_interface_blocks,
        schur_preconditioner=rotation_preconditioner,
        coarse_basis=rot_interface_coarse_basis,
        use_coarse_initial_guess=interface_schur_coarse_initial_guess,
        residual_deflation_rank=interface_schur_residual_deflation_rank,
        residual_deflation_pilot_iterations=(
            interface_schur_residual_deflation_pilot_iterations),
        ritz_deflation_rank=rotation_ritz_rank,
        ritz_probe_iterations=(
            rotation_ritz_probe_iterations
            if rotation_ritz_probe_iterations > 0 else None),
        ritz_mode=rotation_ritz_mode,
        ritz_rank_selection_mode=(
            interface_schur_rotation_ritz_rank_selection_mode),
        ritz_value_threshold=interface_schur_rotation_ritz_value_threshold,
        ritz_energy_capture_fraction=(
            interface_schur_rotation_ritz_energy_capture_fraction))
    rot_stats = {**rot_stats, **rot_coarse_info}
  else:
    rot_solution, rot_stats = solve_distributed_hessian_system(
        rot_hessian, rot_gradient, rot_blocks, rotation_iterations,
        linear_solver, relaxation, damping)
  if summary_selection_mode == "residual_force_refinement":
    rot_separator_summary = refine_normal_summary_blocks_by_residual_force(
        full_summary=full_rot_separator_summary,
        selected_summary=rot_separator_summary,
        solution=rot_solution,
        max_offdiag_block_edges=refinement_max_edges,
    )
    rot_hessian, rot_gradient = _replace_separator_summary_in_total_system(
        full_rot_hessian,
        full_rot_gradient,
        full_rot_separator_summary,
        rot_separator_summary)
    if uses_rotation_interface_coarse_preconditioner:
      rot_interface_coarse_basis, rot_coarse_info = (
          _build_interface_coarse_basis(
              offsets=rot_meta["offsets"],
              block_dim=rot_meta["block_dim"],
              interface_pose_ids=interface_pose_ids,
              robot_of=robot_of,
              interface_indices=rot_interface_indices,
              separator_summary=rot_separator_summary,
              mode=rotation_coarse_basis_mode,
              component_limit=rotation_coarse_component_limit,
              component_selection_mode=rotation_coarse_component_selection_mode))
    if linear_solver == "interface_schur_pcg":
      rot_solution, rot_stats = interface_schur_hessian_solve(
          rot_hessian, rot_gradient, rot_interface_indices,
          rotation_iterations, damping, tolerance=1e-10,
          interface_blocks=rot_interface_blocks,
          schur_preconditioner=rotation_preconditioner,
          coarse_basis=rot_interface_coarse_basis,
          use_coarse_initial_guess=interface_schur_coarse_initial_guess,
          residual_deflation_rank=interface_schur_residual_deflation_rank,
          residual_deflation_pilot_iterations=(
              interface_schur_residual_deflation_pilot_iterations),
          ritz_deflation_rank=rotation_ritz_rank,
          ritz_probe_iterations=(
              rotation_ritz_probe_iterations
              if rotation_ritz_probe_iterations > 0 else None),
          ritz_mode=rotation_ritz_mode,
          ritz_rank_selection_mode=(
              interface_schur_rotation_ritz_rank_selection_mode),
          ritz_value_threshold=interface_schur_rotation_ritz_value_threshold,
          ritz_energy_capture_fraction=(
              interface_schur_rotation_ritz_energy_capture_fraction))
      rot_stats = {**rot_stats, **rot_coarse_info}
    else:
      rot_solution, rot_stats = solve_distributed_hessian_system(
          rot_hessian, rot_gradient, rot_blocks, rotation_iterations,
          linear_solver, relaxation, damping)
  rot_correction_stats = None
  if summary_selection_mode == "omitted_force_correction":
    rot_reference_solution = None
    rot_reference_stats = None
    if omitted_force_correction_diagnose_subspace_miss:
      rot_reference_solution, rot_reference_stats = solve_distributed_hessian_system(
          full_rot_hessian, full_rot_gradient, rot_blocks, rotation_iterations,
          linear_solver, relaxation, damping)
    rot_solution, rot_correction_stats = apply_omitted_force_correction(
        full_hessian=full_rot_hessian,
        full_gradient=full_rot_gradient,
        selected_hessian=rot_hessian,
        selected_gradient=rot_gradient,
        blocks=rot_blocks,
        solution=rot_solution,
        iterations=rotation_iterations,
        linear_solver=linear_solver,
        relaxation=relaxation,
        damping=damping,
        block_dim=rot_meta["block_dim"],
        correction_rounds=omitted_force_correction_rounds,
        curvature_model=omitted_force_correction_curvature_model,
        curvature_rank=omitted_force_correction_curvature_rank,
        curvature_rank_scheduler=(
            omitted_force_correction_curvature_rank_scheduler),
        curvature_rank_max_condition=(
            omitted_force_correction_curvature_rank_max_condition),
        curvature_rank_condition_policy=(
            omitted_force_correction_curvature_rank_condition_policy),
        curvature_rank_condition_mad_scale=(
            omitted_force_correction_curvature_rank_condition_mad_scale),
        curvature_payload_model=(
            omitted_force_correction_curvature_payload_model),
        reference_solution=rot_reference_solution,
        reference_label="full_normal_solution")
    if rot_reference_stats is not None:
      rot_correction_stats["reference_solve_stats"] = rot_reference_stats
    rot_stats = {
        **rot_stats,
        "omitted_force_correction_iterations":
            rot_correction_stats.get("total_solve_iterations", 0),
        "omitted_force_correction_accepted_rounds":
            rot_correction_stats.get("accepted_rounds", 0),
        "omitted_force_correction_delta_norm":
            rot_correction_stats.get("delta_norm", 0.0),
    }
  rotations = _rotations_from_solution(rot_solution, pose_ids,
                                       rot_meta["offsets"], dim, anchor_pose)

  (
      trans_hessian,
      trans_gradient,
      trans_meta,
      trans_separator_summary,
      full_trans_hessian,
      full_trans_gradient,
      full_trans_separator_summary,
  ) = (
      _summary_normal_equation_for_system(
          assemble_translation_system,
          private_edges,
          separator_edges,
          pose_ids,
          dim,
          weighted,
          cost_mode,
          anchor_pose,
          rotations,
          summary_selection_mode=seed_selection_mode,
          summary_max_offdiag_block_edges=summary_max_offdiag_block_edges,
      ))
  trans_blocks = _robot_blocks_from_offsets(trans_meta["offsets"],
                                            trans_meta["block_dim"], robot_of)
  trans_interface_indices = _interface_variable_indices_from_offsets(
      trans_meta["offsets"], trans_meta["block_dim"], interface_pose_ids)
  trans_interface_blocks = _interface_variable_blocks_from_offsets(
      trans_meta["offsets"], trans_meta["block_dim"], interface_pose_ids)
  trans_interface_coarse_basis = None
  trans_coarse_info = {
      "coarse_basis_mode": translation_coarse_basis_mode,
      "coarse_component_limit": (
          None if translation_coarse_component_limit is None
          else int(translation_coarse_component_limit)),
      "coarse_component_selection_mode":
          translation_coarse_component_selection_mode,
      "robot_coordinate_basis_rank": 0,
      "component_coordinate_basis_rank": 0,
  }
  if uses_translation_interface_coarse_preconditioner:
    trans_interface_coarse_basis, trans_coarse_info = (
        _build_interface_coarse_basis(
            offsets=trans_meta["offsets"],
            block_dim=trans_meta["block_dim"],
            interface_pose_ids=interface_pose_ids,
            robot_of=robot_of,
            interface_indices=trans_interface_indices,
            separator_summary=trans_separator_summary,
            mode=translation_coarse_basis_mode,
            component_limit=translation_coarse_component_limit,
            component_selection_mode=(
                translation_coarse_component_selection_mode)))
  if linear_solver == "interface_schur_pcg":
    trans_solution, trans_stats = interface_schur_hessian_solve(
        trans_hessian, trans_gradient, trans_interface_indices,
        translation_iterations, damping, tolerance=1e-10,
        interface_blocks=trans_interface_blocks,
        schur_preconditioner=translation_preconditioner,
        coarse_basis=trans_interface_coarse_basis,
        use_coarse_initial_guess=interface_schur_coarse_initial_guess,
        residual_deflation_rank=interface_schur_residual_deflation_rank,
        residual_deflation_pilot_iterations=(
            interface_schur_residual_deflation_pilot_iterations),
        ritz_deflation_rank=translation_ritz_rank,
        ritz_probe_iterations=(
            translation_ritz_probe_iterations
            if translation_ritz_probe_iterations > 0 else None),
        ritz_mode=translation_ritz_mode,
        ritz_rank_selection_mode=(
            interface_schur_translation_ritz_rank_selection_mode),
        ritz_value_threshold=interface_schur_translation_ritz_value_threshold,
        ritz_energy_capture_fraction=(
            interface_schur_translation_ritz_energy_capture_fraction))
    trans_stats = {**trans_stats, **trans_coarse_info}
  else:
    trans_solution, trans_stats = solve_distributed_hessian_system(
        trans_hessian, trans_gradient, trans_blocks, translation_iterations,
        linear_solver, relaxation, damping)
  if summary_selection_mode == "residual_force_refinement":
    trans_separator_summary = refine_normal_summary_blocks_by_residual_force(
        full_summary=full_trans_separator_summary,
        selected_summary=trans_separator_summary,
        solution=trans_solution,
        max_offdiag_block_edges=refinement_max_edges,
    )
    trans_hessian, trans_gradient = _replace_separator_summary_in_total_system(
        full_trans_hessian,
        full_trans_gradient,
        full_trans_separator_summary,
        trans_separator_summary)
    if uses_translation_interface_coarse_preconditioner:
      trans_interface_coarse_basis, trans_coarse_info = (
          _build_interface_coarse_basis(
              offsets=trans_meta["offsets"],
              block_dim=trans_meta["block_dim"],
              interface_pose_ids=interface_pose_ids,
              robot_of=robot_of,
              interface_indices=trans_interface_indices,
              separator_summary=trans_separator_summary,
              mode=translation_coarse_basis_mode,
              component_limit=translation_coarse_component_limit,
              component_selection_mode=(
                  translation_coarse_component_selection_mode)))
    if linear_solver == "interface_schur_pcg":
      trans_solution, trans_stats = interface_schur_hessian_solve(
          trans_hessian, trans_gradient, trans_interface_indices,
          translation_iterations, damping, tolerance=1e-10,
          interface_blocks=trans_interface_blocks,
          schur_preconditioner=translation_preconditioner,
          coarse_basis=trans_interface_coarse_basis,
          use_coarse_initial_guess=interface_schur_coarse_initial_guess,
          residual_deflation_rank=interface_schur_residual_deflation_rank,
          residual_deflation_pilot_iterations=(
              interface_schur_residual_deflation_pilot_iterations),
          ritz_deflation_rank=translation_ritz_rank,
          ritz_probe_iterations=(
              translation_ritz_probe_iterations
              if translation_ritz_probe_iterations > 0 else None),
          ritz_mode=translation_ritz_mode,
          ritz_rank_selection_mode=(
              interface_schur_translation_ritz_rank_selection_mode),
          ritz_value_threshold=(
              interface_schur_translation_ritz_value_threshold),
          ritz_energy_capture_fraction=(
              interface_schur_translation_ritz_energy_capture_fraction))
      trans_stats = {**trans_stats, **trans_coarse_info}
    else:
      trans_solution, trans_stats = solve_distributed_hessian_system(
          trans_hessian, trans_gradient, trans_blocks, translation_iterations,
          linear_solver, relaxation, damping)
  trans_correction_stats = None
  if summary_selection_mode == "omitted_force_correction":
    trans_reference_solution = None
    trans_reference_stats = None
    if omitted_force_correction_diagnose_subspace_miss:
      trans_reference_solution, trans_reference_stats = (
          solve_distributed_hessian_system(
              full_trans_hessian, full_trans_gradient, trans_blocks,
              translation_iterations, linear_solver, relaxation, damping))
    trans_solution, trans_correction_stats = apply_omitted_force_correction(
        full_hessian=full_trans_hessian,
        full_gradient=full_trans_gradient,
        selected_hessian=trans_hessian,
        selected_gradient=trans_gradient,
        blocks=trans_blocks,
        solution=trans_solution,
        iterations=translation_iterations,
        linear_solver=linear_solver,
        relaxation=relaxation,
        damping=damping,
        block_dim=trans_meta["block_dim"],
        correction_rounds=omitted_force_correction_rounds,
        curvature_model=omitted_force_correction_curvature_model,
        curvature_rank=omitted_force_correction_curvature_rank,
        curvature_rank_scheduler=(
            omitted_force_correction_curvature_rank_scheduler),
        curvature_rank_max_condition=(
            omitted_force_correction_curvature_rank_max_condition),
        curvature_rank_condition_policy=(
            omitted_force_correction_curvature_rank_condition_policy),
        curvature_rank_condition_mad_scale=(
            omitted_force_correction_curvature_rank_condition_mad_scale),
        curvature_payload_model=(
            omitted_force_correction_curvature_payload_model),
        reference_solution=trans_reference_solution,
        reference_label="full_normal_solution")
    if trans_reference_stats is not None:
      trans_correction_stats["reference_solve_stats"] = trans_reference_stats
    trans_stats = {
        **trans_stats,
        "omitted_force_correction_iterations":
            trans_correction_stats.get("total_solve_iterations", 0),
        "omitted_force_correction_accepted_rounds":
            trans_correction_stats.get("accepted_rounds", 0),
        "omitted_force_correction_delta_norm":
            trans_correction_stats.get("delta_norm", 0.0),
    }
  translations = _translations_from_solution(trans_solution, pose_ids,
                                             trans_meta["offsets"], dim,
                                             anchor_pose)
  if central_equivalence_diagnostic:
    rot_stats = {
        **rot_stats,
        "central_equivalence":
            interface_schur_central_equivalence_diagnostic(
                full_rot_hessian,
                full_rot_gradient,
                rot_solution,
                rot_interface_indices,
                damping=damping),
    }
    trans_stats = {
        **trans_stats,
        "central_equivalence":
            interface_schur_central_equivalence_diagnostic(
                full_trans_hessian,
                full_trans_gradient,
                trans_solution,
                trans_interface_indices,
                damping=damping),
    }
  central_equivalence_iterative_diagnostic_iterations = max(
      0, int(central_equivalence_iterative_diagnostic_iterations))
  if central_equivalence_iterative_diagnostic_iterations > 0:
    rot_stats = {
        **rot_stats,
        "central_equivalence_iterative":
            interface_schur_iterative_central_equivalence_diagnostic(
                full_rot_hessian,
                full_rot_gradient,
                rot_solution,
                rot_interface_indices,
                iterations=central_equivalence_iterative_diagnostic_iterations,
                damping=damping,
                tolerance=1e-10,
                interface_blocks=rot_interface_blocks,
                schur_preconditioner="block_jacobi"),
    }
    trans_stats = {
        **trans_stats,
        "central_equivalence_iterative":
            interface_schur_iterative_central_equivalence_diagnostic(
                full_trans_hessian,
                full_trans_gradient,
                trans_solution,
                trans_interface_indices,
                iterations=central_equivalence_iterative_diagnostic_iterations,
                damping=damping,
                tolerance=1e-10,
                interface_blocks=trans_interface_blocks,
                schur_preconditioner="block_jacobi"),
    }
  rotation_model_cert = normal_model_representativeness_certificate(
      full_hessian=full_rot_hessian,
      full_gradient=full_rot_gradient,
      selected_hessian=rot_hessian,
      selected_gradient=rot_gradient,
      solution=rot_solution)
  translation_model_cert = normal_model_representativeness_certificate(
      full_hessian=full_trans_hessian,
      full_gradient=full_trans_gradient,
      selected_hessian=trans_hessian,
      selected_gradient=trans_gradient,
      solution=trans_solution)
  summary_model_cert = {
      "rotation": rotation_model_cert,
      "translation": translation_model_cert,
      "combined": combine_normal_model_representativeness(
          rotation_model_cert,
          translation_model_cert),
  }
  rotation_interface_split = normal_model_private_interface_residual_split(
      full_hessian=full_rot_hessian,
      full_gradient=full_rot_gradient,
      selected_hessian=rot_hessian,
      selected_gradient=rot_gradient,
      solution=rot_solution,
      offsets=rot_meta["offsets"],
      block_dim=rot_meta["block_dim"],
      interface_pose_ids=interface_pose_ids)
  translation_interface_split = normal_model_private_interface_residual_split(
      full_hessian=full_trans_hessian,
      full_gradient=full_trans_gradient,
      selected_hessian=trans_hessian,
      selected_gradient=trans_gradient,
      solution=trans_solution,
      offsets=trans_meta["offsets"],
      block_dim=trans_meta["block_dim"],
      interface_pose_ids=interface_pose_ids)
  private_interface_residual_split = {
      "model": "combined_private_interface_normal_residual_split",
      "rotation": rotation_interface_split,
      "translation": translation_interface_split,
      "combined": combine_private_interface_residual_split(
          rotation_interface_split,
          translation_interface_split),
  }
  poses = _poses_from_rotations_translations(pose_ids, rotations, translations,
                                             dim)
  robot_count = _robot_count_for_poses(pose_ids, robot_of)
  separator_count = len(separator_edges)
  normal_summary_comm = normal_summary_communication_from_summaries(
      separator_count,
      rot_separator_summary,
      trans_separator_summary)
  if linear_solver == "compact_pcg":
    linear_comm = estimate_dci_compact_pcg_solve_communication(
        rot_stats,
        trans_stats,
        robot_count)
  elif linear_solver == "interface_schur_pcg":
    linear_comm = estimate_dci_interface_schur_solve_communication(
        rot_stats,
        trans_stats,
        robot_count)
  else:
    linear_comm = estimate_dci_boundary_pose_solve_communication(
        rot_stats,
        trans_stats,
        robot_count,
        separator_boundary_pose_unit_count(graph_edges, robot_of),
        dim)
  correction_comm = {
      "rotation": (
          rot_correction_stats.get("communication_estimate", {})
          if rot_correction_stats else {}),
      "translation": (
          trans_correction_stats.get("communication_estimate", {})
          if trans_correction_stats else {}),
      "total_omitted_force_payload_bytes": (
          int(rot_correction_stats.get("communication_estimate", {}).get(
              "omitted_force_payload_bytes", 0))
          if rot_correction_stats else 0) + (
              int(trans_correction_stats.get("communication_estimate", {}).get(
                  "omitted_force_payload_bytes", 0))
              if trans_correction_stats else 0),
      "total_curvature_payload_bytes": (
          int(rot_correction_stats.get("communication_estimate", {}).get(
              "curvature_payload_bytes", 0))
          if rot_correction_stats else 0) + (
              int(trans_correction_stats.get("communication_estimate", {}).get(
                  "curvature_payload_bytes", 0))
              if trans_correction_stats else 0),
  }
  correction_comm["total_omitted_force_payload_mb"] = _bytes_to_mb(
      correction_comm["total_omitted_force_payload_bytes"])
  correction_comm["total_curvature_payload_mb"] = _bytes_to_mb(
      correction_comm["total_curvature_payload_bytes"])
  correction_comm["total_correction_payload_bytes"] = (
      int(correction_comm["total_omitted_force_payload_bytes"]) +
      int(correction_comm["total_curvature_payload_bytes"]))
  correction_comm["total_correction_payload_mb"] = _bytes_to_mb(
      correction_comm["total_correction_payload_bytes"])
  correction_comm["model"] = "omitted_force_vector_correction_payload"
  return poses, {
      "pose_count": len(pose_ids),
      "dimension": dim,
      "anchor_pose": anchor_pose,
      "separator_edge_count": separator_count,
      "private_edge_count": len(private_edges),
      "summary_selection_mode": summary_selection_mode,
      "summary_max_offdiag_block_edges": summary_max_offdiag_block_edges,
      "summary_refinement_max_offdiag_block_edges": (
          summary_refinement_max_offdiag_block_edges),
      "interface_schur_preconditioner": interface_schur_preconditioner,
      "interface_schur_rotation_preconditioner": rotation_preconditioner,
      "interface_schur_translation_preconditioner":
          translation_preconditioner,
      "interface_schur_coarse_initial_guess": bool(
          interface_schur_coarse_initial_guess),
      "interface_schur_coarse_basis_mode": interface_schur_coarse_basis_mode,
      "interface_schur_coarse_component_limit": (
          None if interface_schur_coarse_component_limit is None
          else int(interface_schur_coarse_component_limit)),
      "interface_schur_rotation_coarse_basis_mode":
          rotation_coarse_basis_mode,
      "interface_schur_translation_coarse_basis_mode":
          translation_coarse_basis_mode,
      "interface_schur_rotation_coarse_component_limit": (
          None if rotation_coarse_component_limit is None
          else int(rotation_coarse_component_limit)),
      "interface_schur_translation_coarse_component_limit": (
          None if translation_coarse_component_limit is None
          else int(translation_coarse_component_limit)),
      "interface_schur_coarse_component_selection_mode":
          interface_schur_coarse_component_selection_mode,
      "interface_schur_rotation_coarse_component_selection_mode":
          rotation_coarse_component_selection_mode,
      "interface_schur_translation_coarse_component_selection_mode":
          translation_coarse_component_selection_mode,
      "interface_schur_residual_deflation_rank": int(
          interface_schur_residual_deflation_rank),
      "interface_schur_residual_deflation_pilot_iterations": int(
          interface_schur_residual_deflation_pilot_iterations),
      "interface_schur_ritz_rank": int(interface_schur_ritz_rank),
      "interface_schur_ritz_probe_iterations": int(
          interface_schur_ritz_probe_iterations),
      "interface_schur_rotation_ritz_rank": int(rotation_ritz_rank),
      "interface_schur_translation_ritz_rank": int(translation_ritz_rank),
      "interface_schur_rotation_ritz_probe_iterations": int(
          rotation_ritz_probe_iterations),
      "interface_schur_translation_ritz_probe_iterations": int(
          translation_ritz_probe_iterations),
      "interface_schur_rotation_ritz_rank_selection_mode":
          interface_schur_rotation_ritz_rank_selection_mode,
      "interface_schur_translation_ritz_rank_selection_mode":
          interface_schur_translation_ritz_rank_selection_mode,
      "interface_schur_rotation_ritz_value_threshold": (
          None if interface_schur_rotation_ritz_value_threshold is None
          else float(interface_schur_rotation_ritz_value_threshold)),
      "interface_schur_translation_ritz_value_threshold": (
          None if interface_schur_translation_ritz_value_threshold is None
          else float(interface_schur_translation_ritz_value_threshold)),
      "interface_schur_rotation_ritz_energy_capture_fraction": (
          None if interface_schur_rotation_ritz_energy_capture_fraction is None
          else float(interface_schur_rotation_ritz_energy_capture_fraction)),
      "interface_schur_translation_ritz_energy_capture_fraction": (
          None
          if interface_schur_translation_ritz_energy_capture_fraction is None
          else float(interface_schur_translation_ritz_energy_capture_fraction)),
      "interface_schur_ritz_mode": interface_schur_ritz_mode,
      "central_equivalence_diagnostic": bool(central_equivalence_diagnostic),
      "central_equivalence_iterative_diagnostic_iterations": int(
          central_equivalence_iterative_diagnostic_iterations),
      "omitted_force_correction_rounds": int(omitted_force_correction_rounds),
      "omitted_force_correction_curvature_model":
          omitted_force_correction_curvature_model,
      "omitted_force_correction_curvature_rank":
          int(omitted_force_correction_curvature_rank),
      "omitted_force_correction_curvature_rank_scheduler":
          omitted_force_correction_curvature_rank_scheduler,
      "omitted_force_correction_curvature_rank_max_condition":
          omitted_force_correction_curvature_rank_max_condition,
      "omitted_force_correction_curvature_rank_condition_policy":
          omitted_force_correction_curvature_rank_condition_policy,
      "omitted_force_correction_curvature_rank_condition_mad_scale":
          float(omitted_force_correction_curvature_rank_condition_mad_scale),
      "omitted_force_correction_curvature_payload_model":
          omitted_force_correction_curvature_payload_model,
      "omitted_force_correction_diagnose_subspace_miss":
          bool(omitted_force_correction_diagnose_subspace_miss),
      "rotation_stats": {**rot_stats, **{
          "variable_count": rot_meta["variable_count"],
          "block_dim": rot_meta["block_dim"],
      }},
      "translation_stats": {**trans_stats, **{
          "variable_count": trans_meta["variable_count"],
          "block_dim": trans_meta["block_dim"],
      }},
      "rotation_separator_summary": {
          key: value for key, value in rot_separator_summary.items()
          if key not in {"hessian_blocks", "gradient_blocks"}
      },
      "translation_separator_summary": {
          key: value for key, value in trans_separator_summary.items()
          if key not in {"hessian_blocks", "gradient_blocks"}
      },
      "full_rotation_separator_summary": {
          key: value for key, value in full_rot_separator_summary.items()
          if key not in {"hessian_blocks", "gradient_blocks"}
      },
      "full_translation_separator_summary": {
          key: value for key, value in full_trans_separator_summary.items()
          if key not in {"hessian_blocks", "gradient_blocks"}
      },
      "summary_model_representativeness": summary_model_cert,
      "private_interface_residual_split": private_interface_residual_split,
      "omitted_force_correction": {
          "rotation": rot_correction_stats,
          "translation": trans_correction_stats,
          "communication_estimate": correction_comm,
      },
      "method": "summary_chordal_normal_equation_solve",
      "linear_solver": linear_solver,
      "communication_estimate": linear_comm,
      "normal_equation_summary_communication": normal_summary_comm,
  }


def _pose_rmse(reference: dict[int, np.ndarray], estimate: dict[int, np.ndarray]):
  trans_errors = []
  rot_errors = []
  for pose_id in sorted(set(reference) & set(estimate)):
    trans_errors.append(float(np.linalg.norm(reference[pose_id][:3, 3] -
                                            estimate[pose_id][:3, 3])))
    delta = reference[pose_id][:3, :3].T @ estimate[pose_id][:3, :3]
    cos_theta = (float(np.trace(delta)) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    rot_errors.append(math.degrees(math.acos(cos_theta)))
  if not trans_errors:
    return {"translation_rmse": 0.0, "rotation_rmse_deg": 0.0}
  return {
      "translation_rmse": float(np.sqrt(np.mean(np.square(trans_errors)))),
      "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rot_errors)))),
  }


def compute_candidate_residual_certificate(
    graph_edges: list[Edge],
    baseline: dict[int, np.ndarray],
    candidate: dict[int, np.ndarray],
    robot_of: dict[int, int],
    weighted: bool,
    cost_mode: str,
    tolerance: float = 1e-12,
):
  private_baseline: dict[str, float] = {}
  private_candidate: dict[str, float] = {}
  separator_baseline: dict[str, float] = {}
  separator_candidate: dict[str, float] = {}
  baseline_total = 0.0
  candidate_total = 0.0
  missing_edges = 0

  def add(mapping: dict[str, float], key: str, value: float):
    mapping[key] = mapping.get(key, 0.0) + value

  for edge in graph_edges:
    baseline_cost, baseline_ok = edge_chordal_cost(edge, baseline, weighted,
                                                   cost_mode)
    candidate_cost, candidate_ok = edge_chordal_cost(edge, candidate, weighted,
                                                     cost_mode)
    if not baseline_ok or not candidate_ok:
      missing_edges += 1
      continue
    baseline_total += baseline_cost
    candidate_total += candidate_cost
    ri = robot_of.get(edge.i)
    rj = robot_of.get(edge.j)
    if ri is not None and rj is not None and ri != rj:
      key = f"{min(ri, rj)}-{max(ri, rj)}"
      add(separator_baseline, key, baseline_cost)
      add(separator_candidate, key, candidate_cost)
    elif ri is not None:
      key = str(ri)
      add(private_baseline, key, baseline_cost)
      add(private_candidate, key, candidate_cost)

  private_delta = {
      key: private_candidate.get(key, 0.0) - private_baseline.get(key, 0.0)
      for key in sorted(set(private_baseline) | set(private_candidate),
                        key=lambda x: int(x))
  }
  separator_delta = {
      key: separator_candidate.get(key, 0.0) - separator_baseline.get(key, 0.0)
      for key in sorted(set(separator_baseline) | set(separator_candidate))
  }
  local_deltas = list(private_delta.values()) + list(separator_delta.values())
  total_delta = candidate_total - baseline_total
  strict_local_accept = (
      bool(local_deltas) and all(delta <= tolerance for delta in local_deltas)
  )
  global_consensus_accept = total_delta < -tolerance
  robot_count = len(set(robot_of.values()))
  return {
      "baseline_total_cost": baseline_total,
      "candidate_total_cost": candidate_total,
      "total_delta": total_delta,
      "private_baseline_by_robot": private_baseline,
      "private_candidate_by_robot": private_candidate,
      "private_delta_by_robot": private_delta,
      "separator_baseline_by_pair": separator_baseline,
      "separator_candidate_by_pair": separator_candidate,
      "separator_delta_by_pair": separator_delta,
      "strict_local_accept": strict_local_accept,
      "global_consensus_accept": global_consensus_accept,
      "accepted_by": (
          "global_consensus" if global_consensus_accept
          else "baseline"
      ),
      "missing_edges": missing_edges,
      "communication_estimate": estimate_residual_certificate_communication(
          robot_count),
  }


def materialize_certified_hybrid_estimate(
    graph_edges: list[Edge],
    pose_ids: list[int],
    robot_of: dict[int, int],
    baseline: dict[int, np.ndarray],
    candidate: dict[int, np.ndarray],
    weighted: bool,
    cost_mode: str,
    output_path: Path,
):
  certificate = compute_candidate_residual_certificate(
      graph_edges=graph_edges,
      baseline=baseline,
      candidate=candidate,
      robot_of=robot_of,
      weighted=weighted,
      cost_mode=cost_mode,
  )
  selected = (
      "distributed_chordal_pcg"
      if certificate["global_consensus_accept"]
      else "baseline_gauge_selector"
  )
  selected_poses = candidate if selected == "distributed_chordal_pcg" else baseline
  dim = pose_dimension_from_edges(graph_edges)
  ordered_selected = {
      pose_id: selected_poses[pose_id]
      for pose_id in sorted(pose_ids)
      if pose_id in selected_poses
  }
  write_manual_matrix_pose_set(output_path, ordered_selected, dim)
  return {
      "selected": selected,
      "selected_total_cost": (
          certificate["candidate_total_cost"]
          if selected == "distributed_chordal_pcg"
          else certificate["baseline_total_cost"]
      ),
      "output_path": str(output_path),
      "certificate": certificate,
      "pose_count": len(ordered_selected),
  }


def build_report(args):
  graph_vertices, graph_edges = parse_g2o_graph(Path(args.graph))
  pose_ids = graph_pose_ids(graph_vertices, graph_edges)
  robot_of, ranges = build_contiguous_robot_map(pose_ids, args.num_robots)
  active_ids = active_pose_ids(graph_edges, robot_of, args.active_hops)

  central_poses, central_stats = solve_centralized_chordal_initialization(
      graph_edges, pose_ids, args.weighted, args.cost_mode)
  distributed_poses, distributed_stats = solve_distributed_chordal_initialization(
      graph_edges, pose_ids, robot_of, args.rotation_iterations,
      args.translation_iterations, args.weighted, args.cost_mode,
      relaxation=args.relaxation, damping=args.damping,
      linear_solver=args.linear_solver)

  central_cost = cost_breakdown(graph_edges, central_poses, robot_of,
                                active_ids, args.weighted, args.cost_mode)
  distributed_cost = cost_breakdown(graph_edges, distributed_poses, robot_of,
                                    active_ids, args.weighted, args.cost_mode)
  baseline_report = None
  certificate_report = None
  if args.baseline_estimate:
    baseline_poses, baseline_fmt = load_oracle_pose_set(
        Path(args.baseline_estimate), args.baseline_format)
    baseline_cost = cost_breakdown(graph_edges, baseline_poses, robot_of,
                                   active_ids, args.weighted, args.cost_mode)
    certificate_report = compute_candidate_residual_certificate(
        graph_edges=graph_edges,
        baseline=baseline_poses,
        candidate=distributed_poses,
        robot_of=robot_of,
        weighted=args.weighted,
        cost_mode=args.cost_mode,
    )
    baseline_report = {
        "path": str(Path(args.baseline_estimate)),
        "format": baseline_fmt,
        "total_cost": baseline_cost["total_cost"],
        "separator_cost": baseline_cost["separator_cost"],
        "private_cost": baseline_cost["private_cost"],
        "gap_to_distributed": baseline_cost["total_cost"] -
            distributed_cost["total_cost"],
        "gap_to_central": baseline_cost["total_cost"] -
            central_cost["total_cost"],
    }

  report = {
      "graph": str(Path(args.graph)),
      "num_robots": args.num_robots,
      "robot_index_ranges": ranges,
      "weighted": args.weighted,
      "cost_mode": args.cost_mode,
      "active_hops": args.active_hops,
      "rotation_iterations": args.rotation_iterations,
      "translation_iterations": args.translation_iterations,
      "relaxation": args.relaxation,
      "damping": args.damping,
      "linear_solver": args.linear_solver,
      "central_total_cost": central_cost["total_cost"],
      "central_separator_cost": central_cost["separator_cost"],
      "central_private_cost": central_cost["private_cost"],
      "distributed_total_cost": distributed_cost["total_cost"],
      "distributed_separator_cost": distributed_cost["separator_cost"],
      "distributed_private_cost": distributed_cost["private_cost"],
      "distributed_minus_central_cost":
          distributed_cost["total_cost"] - central_cost["total_cost"],
      "distributed_pose_rmse_to_central":
          _pose_rmse(central_poses, distributed_poses),
      "central_stats": central_stats,
      "distributed_stats": distributed_stats,
      "dci_linear_solve_communication":
          distributed_stats.get("communication_estimate", {}),
      "central_breakdown": central_cost,
      "distributed_breakdown": distributed_cost,
  }
  if baseline_report is not None:
    report["baseline"] = baseline_report
  if certificate_report is not None:
    selected_is_candidate = certificate_report["global_consensus_accept"]
    report["hybrid_certificate"] = {
        **certificate_report,
        "selected": (
            "distributed_chordal_pcg" if selected_is_candidate
            else "baseline_gauge_selector"
        ),
        "selected_total_cost": (
            distributed_cost["total_cost"]
            if selected_is_candidate else baseline_report["total_cost"]
        ),
        "strict_local_selected": (
            "distributed_chordal_pcg"
            if certificate_report["strict_local_accept"]
          else "baseline_gauge_selector"
        ),
    }
    linear_comm = report["dci_linear_solve_communication"]
    cert_comm = certificate_report.get("communication_estimate", {})
    report["dci_hybrid_total_estimated_comm_mb"] = (
        float(linear_comm.get("linear_solve_total_estimated_mb", 0.0)) +
        float(cert_comm.get("residual_delta_consensus_mb", 0.0))
    )
    if args.output_selected_estimate:
      report["hybrid_materialized_estimate"] = (
          materialize_certified_hybrid_estimate(
              graph_edges=graph_edges,
              pose_ids=pose_ids,
              robot_of=robot_of,
              baseline=baseline_poses,
              candidate=distributed_poses,
              weighted=args.weighted,
              cost_mode=args.cost_mode,
              output_path=Path(args.output_selected_estimate),
          )
      )
  return report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", required=True)
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--active-hops", type=int, default=1)
  parser.add_argument("--rotation-iterations", type=int, default=50)
  parser.add_argument("--translation-iterations", type=int, default=50)
  parser.add_argument("--relaxation", type=float, default=1.0)
  parser.add_argument("--damping", type=float, default=1e-12)
  parser.add_argument("--linear-solver", default="block_jacobi",
                      choices=[
                          "block_jacobi",
                          "pcg",
                          "compact_pcg",
                          "interface_schur_pcg",
                      ])
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="dpgo", choices=["legacy", "dpgo"])
  parser.add_argument("--baseline-estimate", default=None)
  parser.add_argument("--baseline-format", default="manual_matrix",
                      choices=["auto", "g2o", "pose", "matrix", "manual_matrix"])
  parser.add_argument("--output-json", default=None)
  parser.add_argument("--output-summary-row", default=None)
  parser.add_argument("--output-selected-estimate", default=None)
  args = parser.parse_args()

  report = build_report(args)
  payload = json.dumps(report, indent=2, sort_keys=True)
  if args.output_json:
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(payload + "\n", encoding="utf-8")
  if args.output_summary_row:
    row = {
        "graph": report["graph"],
        "central_total_cost": report["central_total_cost"],
        "distributed_total_cost": report["distributed_total_cost"],
        "distributed_minus_central_cost":
            report["distributed_minus_central_cost"],
        "distributed_translation_rmse_to_central":
            report["distributed_pose_rmse_to_central"]["translation_rmse"],
        "distributed_rotation_rmse_to_central_deg":
            report["distributed_pose_rmse_to_central"]["rotation_rmse_deg"],
        "dci_rotation_global_reduction_mb":
            report["dci_linear_solve_communication"].get(
                "rotation_global_reduction_mb", 0.0),
        "dci_translation_global_reduction_mb":
            report["dci_linear_solve_communication"].get(
                "translation_global_reduction_mb", 0.0),
        "dci_pcg_global_reduction_mb":
            report["dci_linear_solve_communication"].get(
                "pcg_global_reduction_mb", 0.0),
        "dci_separator_exchange_mb":
            report["dci_linear_solve_communication"].get(
                "separator_exchange_mb", 0.0),
        "dci_linear_solve_total_estimated_mb":
            report["dci_linear_solve_communication"].get(
                "linear_solve_total_estimated_mb", 0.0),
    }
    if "baseline" in report:
      row["baseline_total_cost"] = report["baseline"]["total_cost"]
      row["baseline_gap_to_distributed"] = (
          report["baseline"]["gap_to_distributed"])
      row["baseline_gap_to_central"] = report["baseline"]["gap_to_central"]
    if "hybrid_certificate" in report:
      row["certificate_selected"] = report["hybrid_certificate"]["selected"]
      row["certificate_selected_total_cost"] = (
          report["hybrid_certificate"]["selected_total_cost"])
      row["strict_local_selected"] = (
          report["hybrid_certificate"]["strict_local_selected"])
      row["dci_certificate_consensus_mb"] = (
          report["hybrid_certificate"].get("communication_estimate", {}).get(
              "residual_delta_consensus_mb", 0.0))
      row["dci_total_estimated_comm_mb"] = report.get(
          "dci_hybrid_total_estimated_comm_mb",
          row["dci_linear_solve_total_estimated_mb"])
    out = Path(args.output_summary_row)
    out.parent.mkdir(parents=True, exist_ok=True)
    exists = out.exists()
    with out.open("a", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
      if not exists:
        writer.writeheader()
      writer.writerow(row)
  print(payload)


if __name__ == "__main__":
  main()

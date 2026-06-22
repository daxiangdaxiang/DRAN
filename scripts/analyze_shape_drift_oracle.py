#!/usr/bin/env python3
"""Diagnose robot-local shape drift after decentralized chordal initialization.

The script is intentionally diagnostic. It does not change an optimizer state.
It compares a deployment initialization estimate against a reference estimate,
then reports how much residual remains after removing per-robot gauge
differences and how much an active-boundary substitution oracle could explain.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np

try:
  from scipy.sparse import coo_matrix  # type: ignore
  from scipy.sparse.linalg import lsqr  # type: ignore
except ImportError:  # pragma: no cover - scipy is available in the research env.
  coo_matrix = None
  lsqr = None

try:
  from scripts.evaluate_pgo import (  # type: ignore
      Edge,
      edge_chordal_cost,
      edge_weights_from_info,
      invert_pose,
      load_pose_set,
      parse_g2o_graph,
      pose_to_matrix,
  )
  from scripts import split_g2o_to_distributed_mapper as dm_split  # type: ignore
except ImportError:  # pragma: no cover - used when executed as a script.
  from evaluate_pgo import (  # type: ignore
      Edge,
      edge_chordal_cost,
      edge_weights_from_info,
      invert_pose,
      load_pose_set,
      parse_g2o_graph,
      pose_to_matrix,
  )
  import split_g2o_to_distributed_mapper as dm_split  # type: ignore


def build_contiguous_robot_map(ids: Iterable[int], num_robots: int):
  sorted_ids = sorted(ids)
  if num_robots <= 0:
    raise ValueError("num_robots must be positive")
  if len(sorted_ids) < num_robots:
    raise ValueError(
        f"num_poses={len(sorted_ids)} is smaller than num_robots={num_robots}")

  poses_per_robot = len(sorted_ids) // num_robots
  robot_of: dict[int, int] = {}
  ranges: list[tuple[int, int]] = []
  for robot in range(num_robots):
    start = robot * poses_per_robot
    end = (robot + 1) * poses_per_robot if robot < num_robots - 1 else len(sorted_ids)
    ranges.append((start, end))
    for idx in range(start, end):
      robot_of[sorted_ids[idx]] = robot
  return robot_of, ranges


def graph_pose_ids(vertices: dict[int, np.ndarray], edges: Iterable[Edge]):
  if vertices:
    return sorted(vertices)
  ids: set[int] = set()
  for edge in edges:
    ids.add(edge.i)
    ids.add(edge.j)
  return sorted(ids)


def active_pose_ids(edges: Iterable[Edge], robot_of: dict[int, int], hops: int) -> set[int]:
  hops = max(0, int(hops))
  active: set[int] = set()
  private_adj: dict[int, set[int]] = defaultdict(set)

  for edge in edges:
    if edge.i not in robot_of or edge.j not in robot_of:
      continue
    if robot_of[edge.i] == robot_of[edge.j]:
      private_adj[edge.i].add(edge.j)
      private_adj[edge.j].add(edge.i)
    else:
      active.add(edge.i)
      active.add(edge.j)

  frontier = deque((pose_id, 0) for pose_id in sorted(active))
  while frontier:
    pose_id, depth = frontier.popleft()
    if depth >= hops:
      continue
    for neighbor in private_adj.get(pose_id, ()):
      if neighbor not in active:
        active.add(neighbor)
        frontier.append((neighbor, depth + 1))
  return active


def cost_breakdown(
    edges: Iterable[Edge],
    poses: dict[int, np.ndarray],
    robot_of: dict[int, int],
    active_ids: set[int],
    weighted: bool,
    cost_mode: str,
):
  totals = {
      "total_cost": 0.0,
      "private_cost": 0.0,
      "separator_cost": 0.0,
      "active_incident_cost": 0.0,
      "inactive_incident_cost": 0.0,
      "total_edges": 0,
      "private_edges": 0,
      "separator_edges": 0,
      "active_incident_edges": 0,
      "inactive_incident_edges": 0,
      "missing_edges": 0,
  }
  for edge in edges:
    cost, ok = edge_chordal_cost(edge, poses, weighted, cost_mode)
    if not ok:
      totals["missing_edges"] += 1
      continue
    totals["total_cost"] += cost
    totals["total_edges"] += 1
    is_separator = (
        edge.i in robot_of and edge.j in robot_of and robot_of[edge.i] != robot_of[edge.j]
    )
    active_incident = edge.i in active_ids or edge.j in active_ids
    if is_separator:
      totals["separator_cost"] += cost
      totals["separator_edges"] += 1
    else:
      totals["private_cost"] += cost
      totals["private_edges"] += 1
    if active_incident:
      totals["active_incident_cost"] += cost
      totals["active_incident_edges"] += 1
    else:
      totals["inactive_incident_cost"] += cost
      totals["inactive_incident_edges"] += 1
  return totals


def transform_pose_set(transform: np.ndarray, poses: dict[int, np.ndarray]):
  return {pose_id: transform @ pose for pose_id, pose in poses.items()}


def align_reference_to_estimate(
    reference: dict[int, np.ndarray], estimate: dict[int, np.ndarray]
):
  common = sorted(set(reference) & set(estimate))
  if not common:
    raise RuntimeError("no common poses between estimate and reference")
  anchor = common[0]
  transform = estimate[anchor] @ invert_pose(reference[anchor])
  return transform_pose_set(transform, reference), anchor


def substitute_active_poses(
    base: dict[int, np.ndarray],
    replacement: dict[int, np.ndarray],
    active_ids: set[int],
):
  output = dict(base)
  for pose_id in active_ids:
    if pose_id in replacement:
      output[pose_id] = replacement[pose_id]
  return output


def pose_dimension_from_edges(edges: Iterable[Edge]) -> int:
  dim = 0
  for edge in edges:
    dim = max(dim, int(edge.dim))
  return dim if dim in {2, 3} else 3


def project_rotation_block(raw: np.ndarray) -> np.ndarray:
  u, _, vt = np.linalg.svd(raw)
  rot = u @ vt
  if np.linalg.det(rot) < 0.0:
    u[:, -1] *= -1.0
    rot = u @ vt
  return rot


def make_pose_from_block(template: np.ndarray, rotation: np.ndarray,
                         translation: np.ndarray, dim: int) -> np.ndarray:
  pose = np.array(template, dtype=float, copy=True)
  pose[:dim, :dim] = project_rotation_block(rotation)
  pose[:dim, 3] = translation
  if dim == 2:
    pose[0:2, 2] = 0.0
    pose[2, 0:2] = 0.0
    pose[2, 2] = 1.0
    pose[2, 3] = 0.0
  pose[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
  return pose


def interpolate_pose_blocks(start: np.ndarray, target: np.ndarray, alpha: float,
                            dim: int) -> np.ndarray:
  rotation = (1.0 - alpha) * start[:dim, :dim] + alpha * target[:dim, :dim]
  translation = (1.0 - alpha) * start[:dim, 3] + alpha * target[:dim, 3]
  return make_pose_from_block(start, rotation, translation, dim)


def _solve_least_squares(rows: list[int], cols: list[int], data: list[float],
                         rhs: list[float], row_count: int,
                         variable_count: int):
  b = np.asarray(rhs, dtype=float)
  if variable_count == 0:
    return np.zeros(0), {"solver": "none", "iterations": 0, "residual_norm": 0.0}
  if row_count == 0:
    return np.zeros(variable_count), {
        "solver": "empty",
        "iterations": 0,
        "residual_norm": 0.0,
    }
  if coo_matrix is not None and lsqr is not None:
    matrix = coo_matrix((data, (rows, cols)),
                        shape=(row_count, variable_count)).tocsr()
    result = lsqr(matrix, b, atol=1e-10, btol=1e-10,
                  iter_lim=max(200, 4 * variable_count))
    return result[0], {
        "solver": "scipy_lsqr",
        "iterations": int(result[2]),
        "residual_norm": float(result[3]),
    }

  dense = np.zeros((row_count, variable_count), dtype=float)
  for row, col, value in zip(rows, cols, data):
    dense[row, col] += value
  solution, residuals, _, _ = np.linalg.lstsq(dense, b, rcond=None)
  residual_norm = float(np.sqrt(residuals[0])) if residuals.size else float(
      np.linalg.norm(dense @ solution - b))
  return solution, {
      "solver": "dense_lstsq",
      "iterations": 0,
      "residual_norm": residual_norm,
  }


def continuous_fixed_reference_shape_oracle(
    graph_edges: list[Edge],
    estimate: dict[int, np.ndarray],
    reference: dict[int, np.ndarray],
    robot_of: dict[int, int],
    active_ids: set[int],
    weighted: bool,
    cost_mode: str,
    damping_weight: float = 1e-6,
    alpha_values: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625,
                                       0.03125, 0.015625, 0.0),
):
  """Solve a per-robot active-set chordal LS with neighbour states fixed.

  This is an oracle diagnostic, not a deployable initialization algorithm:
  separator neighbours are clamped to the supplied reference while each robot
  updates only its own active separator-neighbourhood variables.
  """
  dim = pose_dimension_from_edges(graph_edges)
  block_dim = dim * dim + dim
  updated = {pose_id: np.array(pose, dtype=float, copy=True)
             for pose_id, pose in estimate.items()}
  per_robot = []
  current_global_cost = cost_breakdown(graph_edges, updated, robot_of,
                                       active_ids, weighted,
                                       cost_mode)["total_cost"]

  def variable_index(offsets: dict[int, int], pose_id: int,
                     is_translation: bool, row: int, col: int = 0):
    base = offsets[pose_id]
    if is_translation:
      return base + dim * dim + row
    return base + row * dim + col

  robots = sorted(set(robot_of.values()))
  for robot in robots:
    local_active = sorted(
        pose_id for pose_id in active_ids
        if robot_of.get(pose_id) == robot and pose_id in estimate
    )
    if not local_active:
      per_robot.append({
          "robot": robot,
          "active_pose_count": 0,
          "updated_pose_count": 0,
          "edge_count": 0,
          "equation_count": 0,
          "solver": "none",
          "iterations": 0,
          "residual_norm": 0.0,
      })
      continue

    offsets = {pose_id: idx * block_dim
               for idx, pose_id in enumerate(local_active)}
    local_active_set = set(local_active)
    variable_count = len(local_active) * block_dim
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs: list[float] = []
    row = 0
    used_edges = 0

    def is_local_variable(pose_id: int) -> bool:
      return pose_id in local_active_set and robot_of.get(pose_id) == robot

    def fixed_pose(pose_id: int) -> np.ndarray:
      if robot_of.get(pose_id) == robot:
        return updated[pose_id]
      return reference.get(pose_id, estimate[pose_id])

    def add_equation(coefficients: list[tuple[int, float]], constant: float,
                     weight: float):
      nonlocal row
      if weight <= 0.0:
        return
      sqrt_weight = math.sqrt(weight)
      for col, value in coefficients:
        rows.append(row)
        cols.append(col)
        data.append(sqrt_weight * value)
      rhs.append(-sqrt_weight * constant)
      row += 1

    for edge in graph_edges:
      if edge.i not in robot_of or edge.j not in robot_of:
        continue
      touches_local_active = (
          (edge.i in local_active_set and robot_of[edge.i] == robot) or
          (edge.j in local_active_set and robot_of[edge.j] == robot)
      )
      if not touches_local_active:
        continue
      used_edges += 1
      tau, kappa = edge_weights_from_info(edge, cost_mode, weighted)
      tij = edge.measurement[:dim, 3]
      rij = edge.measurement[:dim, :dim]
      i_is_var = is_local_variable(edge.i)
      j_is_var = is_local_variable(edge.j)
      fixed_i = None if i_is_var else fixed_pose(edge.i)
      fixed_j = None if j_is_var else fixed_pose(edge.j)

      for a in range(dim):
        coefficients: list[tuple[int, float]] = []
        constant = 0.0
        if j_is_var:
          coefficients.append((
              variable_index(offsets, edge.j, True, a), 1.0))
        else:
          constant += float(fixed_j[a, 3])
        if i_is_var:
          coefficients.append((
              variable_index(offsets, edge.i, True, a), -1.0))
          for k in range(dim):
            coefficients.append((
                variable_index(offsets, edge.i, False, a, k), -float(tij[k])))
        else:
          constant -= float(fixed_i[a, 3])
          constant -= float(fixed_i[a, :dim] @ tij)
        add_equation(coefficients, constant, tau)

      for a in range(dim):
        for b in range(dim):
          coefficients = []
          constant = 0.0
          if j_is_var:
            coefficients.append((
                variable_index(offsets, edge.j, False, a, b), 1.0))
          else:
            constant += float(fixed_j[a, b])
          if i_is_var:
            for k in range(dim):
              coefficients.append((
                  variable_index(offsets, edge.i, False, a, k),
                  -float(rij[k, b])))
          else:
            constant -= float(fixed_i[a, :dim] @ rij[:, b])
          add_equation(coefficients, constant, kappa)

    if damping_weight > 0.0:
      for pose_id in local_active:
        base_pose = updated[pose_id]
        for a in range(dim):
          for b in range(dim):
            add_equation(
                [(variable_index(offsets, pose_id, False, a, b), 1.0)],
                -float(base_pose[a, b]),
                damping_weight,
            )
        for a in range(dim):
          add_equation(
              [(variable_index(offsets, pose_id, True, a), 1.0)],
              -float(base_pose[a, 3]),
              damping_weight,
          )

    solution, solve_stats = _solve_least_squares(rows, cols, data, rhs, row,
                                                 variable_count)
    solved_poses: dict[int, np.ndarray] = {}
    for pose_id in local_active:
      offset = offsets[pose_id]
      raw_rot = solution[offset:offset + dim * dim].reshape((dim, dim))
      raw_t = solution[offset + dim * dim:offset + block_dim]
      solved_poses[pose_id] = make_pose_from_block(updated[pose_id], raw_rot,
                                                   raw_t, dim)

    best_alpha = 0.0
    best_cost = current_global_cost
    best_state = updated
    for alpha in alpha_values:
      candidate = dict(updated)
      for pose_id in local_active:
        candidate[pose_id] = interpolate_pose_blocks(updated[pose_id],
                                                     solved_poses[pose_id],
                                                     alpha, dim)
      candidate_cost = cost_breakdown(graph_edges, candidate, robot_of,
                                      active_ids, weighted,
                                      cost_mode)["total_cost"]
      if candidate_cost < best_cost - 1e-12:
        best_alpha = alpha
        best_cost = candidate_cost
        best_state = candidate
    accepted = best_alpha > 0.0
    if accepted:
      updated = best_state
      current_global_cost = best_cost

    per_robot.append({
        "robot": robot,
        "active_pose_count": len(local_active),
        "updated_pose_count": len(local_active) if accepted else 0,
        "edge_count": used_edges,
        "equation_count": row,
        "accepted": accepted,
        "selected_alpha": best_alpha,
        "global_cost_after_robot": current_global_cost,
        **solve_stats,
    })

  stats = {
      "active_pose_count": len(active_ids),
      "updated_pose_count": int(sum(row["updated_pose_count"] for row in per_robot)),
      "accepted_robot_count": int(sum(1 for row in per_robot
                                      if row.get("accepted"))),
      "damping_weight": damping_weight,
      "per_robot": per_robot,
  }
  return updated, stats


def rotation_error_deg(reference: np.ndarray, estimate: np.ndarray) -> float:
  delta = invert_pose(reference) @ estimate
  cos_theta = (float(np.trace(delta[:3, :3])) - 1.0) * 0.5
  cos_theta = max(-1.0, min(1.0, cos_theta))
  return math.degrees(math.acos(cos_theta))


def per_robot_shape_drift(
    estimate: dict[int, np.ndarray],
    reference: dict[int, np.ndarray],
    robot_of: dict[int, int],
    pose_filter: set[int] | None = None,
):
  by_robot: dict[int, list[int]] = defaultdict(list)
  for pose_id in sorted(set(estimate) & set(reference) & set(robot_of)):
    if pose_filter is not None and pose_id not in pose_filter:
      continue
    by_robot[robot_of[pose_id]].append(pose_id)

  trans_errors: list[float] = []
  rot_errors: list[float] = []
  robot_rows = []
  for robot, pose_ids in sorted(by_robot.items()):
    if not pose_ids:
      continue
    anchor = pose_ids[0]
    transform = reference[anchor] @ invert_pose(estimate[anchor])
    robot_trans: list[float] = []
    robot_rot: list[float] = []
    for pose_id in pose_ids:
      aligned = transform @ estimate[pose_id]
      t_err = float(np.linalg.norm(aligned[:3, 3] - reference[pose_id][:3, 3]))
      r_err = rotation_error_deg(reference[pose_id], aligned)
      robot_trans.append(t_err)
      robot_rot.append(r_err)
      trans_errors.append(t_err)
      rot_errors.append(r_err)
    robot_rows.append({
        "robot": robot,
        "pose_count": len(pose_ids),
        "translation_rmse": rmse(robot_trans),
        "rotation_rmse_deg": rmse(robot_rot),
        "translation_max": max(robot_trans) if robot_trans else 0.0,
        "rotation_max_deg": max(robot_rot) if robot_rot else 0.0,
    })

  return {
      "pose_count": len(trans_errors),
      "translation_rmse": rmse(trans_errors),
      "translation_mean": float(np.mean(trans_errors)) if trans_errors else 0.0,
      "translation_max": max(trans_errors) if trans_errors else 0.0,
      "rotation_rmse_deg": rmse(rot_errors),
      "rotation_mean_deg": float(np.mean(rot_errors)) if rot_errors else 0.0,
      "rotation_max_deg": max(rot_errors) if rot_errors else 0.0,
      "per_robot": robot_rows,
  }


def rmse(values: Iterable[float]) -> float:
  arr = np.asarray(list(values), dtype=float)
  if arr.size == 0:
    return 0.0
  return float(np.sqrt(np.mean(arr * arr)))


def fraction(numerator: float, denominator: float) -> float:
  if abs(denominator) <= 1e-12:
    return 0.0
  return numerator / denominator


def build_oracle_report_from_data(
    graph_edges: list[Edge],
    estimate: dict[int, np.ndarray],
    reference: dict[int, np.ndarray],
    robot_of: dict[int, int],
    active_hops: int,
    weighted: bool,
    cost_mode: str,
    run_continuous_shape_oracle: bool = False,
    shape_damping: float = 1e-6,
):
  active_ids = active_pose_ids(graph_edges, robot_of, active_hops)
  reference_aligned, alignment_anchor = align_reference_to_estimate(reference, estimate)
  active_substitution = substitute_active_poses(estimate, reference_aligned, active_ids)

  estimate_cost = cost_breakdown(graph_edges, estimate, robot_of, active_ids,
                                 weighted, cost_mode)
  reference_cost = cost_breakdown(graph_edges, reference_aligned, robot_of, active_ids,
                                  weighted, cost_mode)
  substitution_cost = cost_breakdown(graph_edges, active_substitution, robot_of,
                                     active_ids, weighted, cost_mode)
  shape = per_robot_shape_drift(estimate, reference_aligned, robot_of)
  active_shape = per_robot_shape_drift(estimate, reference_aligned, robot_of,
                                       active_ids)
  inactive_ids = set(robot_of) - active_ids
  inactive_shape = per_robot_shape_drift(estimate, reference_aligned, robot_of,
                                         inactive_ids)

  total_gap = estimate_cost["total_cost"] - reference_cost["total_cost"]
  substitution_reduction = (
      estimate_cost["total_cost"] - substitution_cost["total_cost"]
  )
  separator_gap = estimate_cost["separator_cost"] - reference_cost["separator_cost"]
  active_gap = (
      estimate_cost["active_incident_cost"] -
      reference_cost["active_incident_cost"]
  )

  report = {
      "num_robots": len(set(robot_of.values())),
      "pose_count": len(robot_of),
      "edge_count": len(graph_edges),
      "active_hops": active_hops,
      "active_pose_count": len(active_ids),
      "alignment_anchor_id": alignment_anchor,
      "cost_mode": cost_mode,
      "weighted": weighted,
      "estimate_total_cost": estimate_cost["total_cost"],
      "reference_total_cost": reference_cost["total_cost"],
      "active_substitution_total_cost": substitution_cost["total_cost"],
      "total_gap": total_gap,
      "active_substitution_reduction": substitution_reduction,
      "active_substitution_gap_explained_fraction": fraction(
          substitution_reduction, total_gap),
      "estimate_private_cost": estimate_cost["private_cost"],
      "reference_private_cost": reference_cost["private_cost"],
      "estimate_separator_cost": estimate_cost["separator_cost"],
      "reference_separator_cost": reference_cost["separator_cost"],
      "separator_gap": separator_gap,
      "separator_gap_fraction_of_total_gap": fraction(separator_gap, total_gap),
      "estimate_active_incident_cost": estimate_cost["active_incident_cost"],
      "reference_active_incident_cost": reference_cost["active_incident_cost"],
      "active_incident_gap": active_gap,
      "active_incident_gap_fraction_of_total_gap": fraction(active_gap, total_gap),
      "shape_translation_rmse": shape["translation_rmse"],
      "shape_translation_mean": shape["translation_mean"],
      "shape_translation_max": shape["translation_max"],
      "shape_rotation_rmse_deg": shape["rotation_rmse_deg"],
      "shape_rotation_mean_deg": shape["rotation_mean_deg"],
      "shape_rotation_max_deg": shape["rotation_max_deg"],
      "active_shape_translation_rmse": active_shape["translation_rmse"],
      "active_shape_rotation_rmse_deg": active_shape["rotation_rmse_deg"],
      "inactive_shape_translation_rmse": inactive_shape["translation_rmse"],
      "inactive_shape_rotation_rmse_deg": inactive_shape["rotation_rmse_deg"],
      "per_robot_shape": shape["per_robot"],
      "estimate_breakdown": estimate_cost,
      "reference_breakdown": reference_cost,
      "active_substitution_breakdown": substitution_cost,
  }
  if run_continuous_shape_oracle:
    continuous_state, continuous_stats = continuous_fixed_reference_shape_oracle(
        graph_edges=graph_edges,
        estimate=estimate,
        reference=reference_aligned,
        robot_of=robot_of,
        active_ids=active_ids,
        weighted=weighted,
        cost_mode=cost_mode,
        damping_weight=shape_damping,
    )
    continuous_cost = cost_breakdown(graph_edges, continuous_state, robot_of,
                                     active_ids, weighted, cost_mode)
    continuous_reduction = (
        estimate_cost["total_cost"] - continuous_cost["total_cost"]
    )
    continuous_separator_reduction = (
        estimate_cost["separator_cost"] - continuous_cost["separator_cost"]
    )
    report.update({
        "continuous_shape_oracle_enabled": True,
        "continuous_shape_oracle_total_cost": continuous_cost["total_cost"],
        "continuous_shape_oracle_reduction": continuous_reduction,
        "continuous_shape_oracle_gap_explained_fraction": fraction(
            continuous_reduction, total_gap),
        "continuous_shape_oracle_separator_cost": continuous_cost["separator_cost"],
        "continuous_shape_oracle_separator_reduction":
            continuous_separator_reduction,
        "continuous_shape_oracle_private_cost": continuous_cost["private_cost"],
        "continuous_shape_oracle_active_incident_cost":
            continuous_cost["active_incident_cost"],
        "continuous_shape_oracle_stats": continuous_stats,
        "continuous_shape_oracle_breakdown": continuous_cost,
    })
  else:
    report["continuous_shape_oracle_enabled"] = False
  return report


def centralized_chordal_reference_from_graph(graph_path: Path):
  vertices, vertex_order, edges, _, vertex_kind, edge_kind = dm_split.parse_g2o(graph_path)
  if vertex_order:
    ordered_vertices = sorted(vertex_order)
  else:
    ordered_vertices = sorted(
        {int(tokens[1]) for tokens in edges} | {int(tokens[2]) for tokens in edges}
    )
  if not ordered_vertices:
    raise RuntimeError(f"{graph_path} contains no vertices or edges")
  is3d = dm_split.is_se3_g2o(vertex_kind, edge_kind)
  pose_tuples = dm_split.chordal_initialized_vertices(edges, is3d, ordered_vertices)
  return {
      gid: pose_to_matrix(*pose_tuples[gid])
      for gid in sorted(pose_tuples)
  }


def parse_manual_matrix_estimate(path: Path):
  dense = np.loadtxt(path)
  if dense.ndim != 2 or dense.shape[0] not in {2, 3}:
    raise ValueError(
        f"manual_matrix estimate must be 2 x 3n or 3 x 4n, got {dense.shape}")
  d = int(dense.shape[0])
  block_dim = d + 1
  if dense.shape[1] % block_dim != 0:
    raise ValueError(
        f"manual_matrix columns must be divisible by {block_dim}, got {dense.shape[1]}")
  num_poses = dense.shape[1] // block_dim
  poses: dict[int, np.ndarray] = {}
  for pose_id in range(num_poses):
    col = pose_id * block_dim
    mat = np.eye(4, dtype=float)
    mat[:d, :d] = dense[:, col:col + d]
    mat[:d, 3] = dense[:, col + d]
    poses[pose_id] = mat
  return poses


def load_oracle_pose_set(path: Path, fmt: str):
  if fmt == "manual_matrix":
    return parse_manual_matrix_estimate(path), fmt
  return load_pose_set(path, fmt)


def build_oracle_report(args):
  graph_vertices, graph_edges = parse_g2o_graph(Path(args.graph))
  estimate, estimate_fmt = load_oracle_pose_set(Path(args.estimate), args.estimate_format)
  if args.reference_from_centralized_chordal:
    reference = centralized_chordal_reference_from_graph(Path(args.graph))
    reference_fmt = "centralized_chordal_from_graph"
  else:
    if not args.reference:
      raise ValueError(
          "--reference is required unless --reference-from-centralized-chordal is set")
    reference, reference_fmt = load_oracle_pose_set(Path(args.reference), args.reference_format)
  pose_ids = graph_pose_ids(graph_vertices, graph_edges)
  robot_of, ranges = build_contiguous_robot_map(pose_ids, args.num_robots)
  report = build_oracle_report_from_data(
      graph_edges=graph_edges,
      estimate=estimate,
      reference=reference,
      robot_of=robot_of,
      active_hops=args.active_hops,
      weighted=args.weighted,
      cost_mode=args.cost_mode,
      run_continuous_shape_oracle=args.continuous_shape_oracle,
      shape_damping=args.shape_damping,
  )
  report.update({
      "graph": str(Path(args.graph)),
      "estimate": str(Path(args.estimate)),
      "reference": (
          "centralized_chordal_from_graph"
          if args.reference_from_centralized_chordal
          else str(Path(args.reference))
      ),
      "estimate_format": estimate_fmt,
      "reference_format": reference_fmt,
      "graph_vertices": len(graph_vertices),
      "graph_pose_ids": len(pose_ids),
      "robot_index_ranges": ranges,
  })
  return report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", required=True)
  parser.add_argument("--estimate", required=True)
  parser.add_argument("--reference", default=None)
  parser.add_argument("--reference-from-centralized-chordal", action="store_true")
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--active-hops", type=int, default=1)
  parser.add_argument("--estimate-format", default="manual_matrix",
                      choices=["auto", "g2o", "pose", "matrix", "manual_matrix"])
  parser.add_argument("--reference-format", default="manual_matrix",
                      choices=["auto", "g2o", "pose", "matrix", "manual_matrix"])
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="dpgo", choices=["legacy", "dpgo"])
  parser.add_argument("--continuous-shape-oracle", action="store_true")
  parser.add_argument("--shape-damping", type=float, default=1e-6)
  parser.add_argument("--output-json", default=None)
  args = parser.parse_args()

  report = build_oracle_report(args)
  payload = json.dumps(report, indent=2, sort_keys=True)
  if args.output_json:
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(payload + "\n", encoding="utf-8")
  print(payload)


if __name__ == "__main__":
  main()

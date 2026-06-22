#!/usr/bin/env python3
"""Diagnose robot-separator graph gauge synchronization for DCI.

This script is an offline diagnostic for decentralized chordal initialization.
It estimates one robot-level left gauge correction per robot from separator
edges only, then evaluates how that correction changes the global chordal cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
  from scripts.evaluate_pgo import (  # type: ignore
      Edge,
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


def _robot_variable_offsets(robots: list[int], block_dim: int, anchor_robot: int):
  offsets: dict[int, int] = {}
  offset = 0
  for robot in robots:
    if robot == anchor_robot:
      continue
    offsets[robot] = offset
    offset += block_dim
  return offsets, offset


def _add_weighted_equation(rows: list[int], cols: list[int], data: list[float],
                           rhs: list[float], row: int,
                           coefficients: list[tuple[int, float]],
                           target: float, weight: float):
  if weight <= 0.0:
    return row
  sqrt_weight = math.sqrt(weight)
  for col, value in coefficients:
    rows.append(row)
    cols.append(col)
    data.append(sqrt_weight * value)
  rhs.append(sqrt_weight * target)
  return row + 1


def _solve_robot_rotations(graph_edges: list[Edge], estimate: dict[int, np.ndarray],
                           robot_of: dict[int, int], robots: list[int],
                           dim: int, weighted: bool, cost_mode: str,
                           anchor_robot: int):
  block_dim = dim * dim
  offsets, variable_count = _robot_variable_offsets(robots, block_dim,
                                                    anchor_robot)
  rows: list[int] = []
  cols: list[int] = []
  data: list[float] = []
  rhs: list[float] = []
  row = 0
  separator_edges = 0

  def add_robot_block(coefficients: list[tuple[int, float]], robot: int,
                      row_idx: int, col_idx: int, value: float):
    if robot == anchor_robot:
      return value * (1.0 if row_idx == col_idx else 0.0)
    coefficients.append(
        (offsets[robot] + row_idx * dim + col_idx, value))
    return 0.0

  for edge in graph_edges:
    if edge.i not in robot_of or edge.j not in robot_of:
      continue
    ri = robot_of[edge.i]
    rj = robot_of[edge.j]
    if ri == rj or edge.i not in estimate or edge.j not in estimate:
      continue
    separator_edges += 1
    _, kappa = edge_weights_from_info(edge, cost_mode, weighted)
    rot_i = estimate[edge.i][:dim, :dim]
    rot_j = estimate[edge.j][:dim, :dim]
    rel_rot = edge.measurement[:dim, :dim]
    relative_correction = rot_i @ rel_rot @ rot_j.T
    for a in range(dim):
      for b in range(dim):
        coefficients: list[tuple[int, float]] = []
        constant = 0.0
        constant += add_robot_block(coefficients, rj, a, b, 1.0)
        for k in range(dim):
          constant += add_robot_block(coefficients, ri, a, k,
                                      -float(relative_correction[k, b]))
        row = _add_weighted_equation(rows, cols, data, rhs, row,
                                     coefficients, -constant, kappa)

  solution, solve_stats = _solve_least_squares(rows, cols, data, rhs, row,
                                               variable_count)
  rotations: dict[int, np.ndarray] = {}
  for robot in robots:
    if robot == anchor_robot:
      rotations[robot] = np.eye(dim)
      continue
    offset = offsets[robot]
    raw = solution[offset:offset + block_dim].reshape((dim, dim))
    rotations[robot] = project_rotation_block(raw)

  solve_stats.update({
      "separator_edge_count": separator_edges,
      "rotation_equation_count": row,
  })
  return rotations, solve_stats


def _solve_robot_translations(graph_edges: list[Edge],
                              estimate: dict[int, np.ndarray],
                              robot_of: dict[int, int], robots: list[int],
                              rotations: dict[int, np.ndarray], dim: int,
                              weighted: bool, cost_mode: str,
                              anchor_robot: int):
  offsets, variable_count = _robot_variable_offsets(robots, dim, anchor_robot)
  rows: list[int] = []
  cols: list[int] = []
  data: list[float] = []
  rhs: list[float] = []
  row = 0

  def add_translation_coefficients(coefficients: list[tuple[int, float]],
                                   robot: int, coord: int, value: float):
    if robot == anchor_robot:
      return
    coefficients.append((offsets[robot] + coord, value))

  for edge in graph_edges:
    if edge.i not in robot_of or edge.j not in robot_of:
      continue
    ri = robot_of[edge.i]
    rj = robot_of[edge.j]
    if ri == rj or edge.i not in estimate or edge.j not in estimate:
      continue
    tau, _ = edge_weights_from_info(edge, cost_mode, weighted)
    rot_i = estimate[edge.i][:dim, :dim]
    ti = estimate[edge.i][:dim, 3]
    tj = estimate[edge.j][:dim, 3]
    rel_t = edge.measurement[:dim, 3]
    target = rotations[ri] @ (ti + rot_i @ rel_t) - rotations[rj] @ tj
    for coord in range(dim):
      coefficients: list[tuple[int, float]] = []
      add_translation_coefficients(coefficients, rj, coord, 1.0)
      add_translation_coefficients(coefficients, ri, coord, -1.0)
      row = _add_weighted_equation(rows, cols, data, rhs, row,
                                   coefficients, float(target[coord]), tau)

  solution, solve_stats = _solve_least_squares(rows, cols, data, rhs, row,
                                               variable_count)
  translations: dict[int, np.ndarray] = {}
  for robot in robots:
    if robot == anchor_robot:
      translations[robot] = np.zeros(dim)
      continue
    offset = offsets[robot]
    translations[robot] = solution[offset:offset + dim]

  solve_stats.update({"translation_equation_count": row})
  return translations, solve_stats


def _interpolate_rotation(rotation: np.ndarray, alpha: float):
  return project_rotation_block((1.0 - alpha) * np.eye(rotation.shape[0]) +
                                alpha * rotation)


def apply_robot_gauge_corrections(estimate: dict[int, np.ndarray],
                                  robot_of: dict[int, int],
                                  rotations: dict[int, np.ndarray],
                                  translations: dict[int, np.ndarray],
                                  dim: int, alpha: float = 1.0):
  corrected: dict[int, np.ndarray] = {}
  for pose_id, pose in estimate.items():
    robot = robot_of.get(pose_id)
    if robot is None:
      corrected[pose_id] = np.array(pose, dtype=float, copy=True)
      continue
    q = _interpolate_rotation(rotations.get(robot, np.eye(dim)), alpha)
    c = alpha * translations.get(robot, np.zeros(dim))
    out = np.array(pose, dtype=float, copy=True)
    out[:dim, :dim] = q @ pose[:dim, :dim]
    out[:dim, 3] = q @ pose[:dim, 3] + c
    if dim == 2:
      out[0:2, 2] = 0.0
      out[2, 0:2] = 0.0
      out[2, 2] = 1.0
      out[2, 3] = 0.0
    out[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    corrected[pose_id] = out
  return corrected


def separator_graph_gauge_sync(graph_edges: list[Edge],
                               estimate: dict[int, np.ndarray],
                               robot_of: dict[int, int],
                               weighted: bool,
                               cost_mode: str,
                               anchor_robot: int | None = None):
  dim = pose_dimension_from_edges(graph_edges)
  robots = sorted(set(robot_of.values()))
  if not robots:
    return dict(estimate), {
        "robot_correction_count": 0,
        "separator_edge_count": 0,
        "selected_alpha": 1.0,
    }
  if anchor_robot is None:
    anchor_robot = robots[0]
  rotations, rot_stats = _solve_robot_rotations(
      graph_edges, estimate, robot_of, robots, dim, weighted, cost_mode,
      anchor_robot)
  translations, trans_stats = _solve_robot_translations(
      graph_edges, estimate, robot_of, robots, rotations, dim, weighted,
      cost_mode, anchor_robot)
  corrected = apply_robot_gauge_corrections(estimate, robot_of, rotations,
                                            translations, dim, alpha=1.0)
  stats = {
      "robot_correction_count": len(robots),
      "separator_edge_count": rot_stats["separator_edge_count"],
      "anchor_robot": anchor_robot,
      "dimension": dim,
      "rotation_solver": rot_stats,
      "translation_solver": trans_stats,
      "selected_alpha": 1.0,
  }
  return corrected, stats


def build_sync_report(args):
  graph_vertices, graph_edges = parse_g2o_graph(Path(args.graph))
  estimate, estimate_fmt = load_oracle_pose_set(Path(args.estimate),
                                                args.estimate_format)
  pose_ids = graph_pose_ids(graph_vertices, graph_edges)
  robot_of, ranges = build_contiguous_robot_map(pose_ids, args.num_robots)
  active_ids = active_pose_ids(graph_edges, robot_of, args.active_hops)
  before = cost_breakdown(graph_edges, estimate, robot_of, active_ids,
                          args.weighted, args.cost_mode)
  raw_corrected, raw_stats = separator_graph_gauge_sync(
      graph_edges, estimate, robot_of, args.weighted, args.cost_mode,
      args.anchor_robot)

  dim = pose_dimension_from_edges(graph_edges)
  # Reuse solved corrections for diagnostic backtracking without rebuilding LS.
  robots = sorted(set(robot_of.values()))
  anchor = args.anchor_robot if args.anchor_robot is not None else robots[0]
  rotations, _ = _solve_robot_rotations(graph_edges, estimate, robot_of,
                                        robots, dim, args.weighted,
                                        args.cost_mode, anchor)
  translations, _ = _solve_robot_translations(graph_edges, estimate, robot_of,
                                              robots, rotations, dim,
                                              args.weighted, args.cost_mode,
                                              anchor)

  alphas = [1.0]
  if args.backtracking:
    alphas = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0]
  best_alpha = 1.0
  best_state = raw_corrected
  best_cost = cost_breakdown(graph_edges, raw_corrected, robot_of, active_ids,
                             args.weighted, args.cost_mode)["total_cost"]
  for alpha in alphas:
    candidate = apply_robot_gauge_corrections(estimate, robot_of, rotations,
                                              translations, dim, alpha)
    candidate_cost = cost_breakdown(graph_edges, candidate, robot_of,
                                    active_ids, args.weighted,
                                    args.cost_mode)["total_cost"]
    if candidate_cost < best_cost - 1e-12:
      best_alpha = alpha
      best_cost = candidate_cost
      best_state = candidate

  after = cost_breakdown(graph_edges, best_state, robot_of, active_ids,
                         args.weighted, args.cost_mode)
  raw_after = cost_breakdown(graph_edges, raw_corrected, robot_of, active_ids,
                             args.weighted, args.cost_mode)
  report = {
      "graph": str(Path(args.graph)),
      "estimate": str(Path(args.estimate)),
      "estimate_format": estimate_fmt,
      "num_robots": args.num_robots,
      "robot_index_ranges": ranges,
      "weighted": args.weighted,
      "cost_mode": args.cost_mode,
      "active_hops": args.active_hops,
      "initial_total_cost": before["total_cost"],
      "initial_private_cost": before["private_cost"],
      "initial_separator_cost": before["separator_cost"],
      "raw_sync_total_cost": raw_after["total_cost"],
      "raw_sync_private_cost": raw_after["private_cost"],
      "raw_sync_separator_cost": raw_after["separator_cost"],
      "final_total_cost": after["total_cost"],
      "final_private_cost": after["private_cost"],
      "final_separator_cost": after["separator_cost"],
      "total_reduction": before["total_cost"] - after["total_cost"],
      "separator_reduction": before["separator_cost"] - after["separator_cost"],
      "private_change": after["private_cost"] - before["private_cost"],
      "selected_alpha": best_alpha,
      "backtracking": args.backtracking,
      "sync_stats": raw_stats,
      "initial_breakdown": before,
      "raw_sync_breakdown": raw_after,
      "final_breakdown": after,
  }
  return report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", required=True)
  parser.add_argument("--estimate", required=True)
  parser.add_argument("--num-robots", type=int, default=5)
  parser.add_argument("--active-hops", type=int, default=1)
  parser.add_argument("--estimate-format", default="manual_matrix",
                      choices=["auto", "g2o", "pose", "matrix", "manual_matrix"])
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="dpgo", choices=["legacy", "dpgo"])
  parser.add_argument("--anchor-robot", type=int, default=None)
  parser.add_argument("--backtracking", action="store_true")
  parser.add_argument("--output-json", default=None)
  args = parser.parse_args()

  report = build_sync_report(args)
  payload = json.dumps(report, indent=2, sort_keys=True)
  if args.output_json:
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(payload + "\n", encoding="utf-8")
  print(payload)


if __name__ == "__main__":
  main()

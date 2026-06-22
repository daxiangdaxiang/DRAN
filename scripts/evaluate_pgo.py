#!/usr/bin/env python3
"""Evaluate SE2/SE3 pose graphs and pose outputs.

Supported pose inputs:
  - g2o files containing VERTEX_SE2 lines
  - g2o files containing VERTEX_SE3:QUAT lines
  - plain pose rows: id x y z qx qy qz qw
  - matrix text formats, parsed best-effort:
      * SE-Sync xhat as a 3 x 4n matrix: [t_1 ... t_n R_1 ... R_n]
      * DPGO-MM estimates as a 4n x 3 matrix: [t rows; R row blocks]
      * id followed by a row-major 3x4 or 4x4 transform
      * or a bare 3x4 / 4x4 transform with an implicit line-based id

The matrix parser is intentionally permissive.  It assumes one pose per line,
whitespace-separated values, and row-major layout for 12/16 numeric entries
unless it detects the SE-Sync full-matrix layout.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Edge:
  i: int
  j: int
  measurement: np.ndarray
  info: np.ndarray | None
  dim: int


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
  q = np.array([qx, qy, qz, qw], dtype=float)
  n = np.linalg.norm(q)
  if n == 0.0:
    return np.eye(3)
  q /= n
  x, y, z, w = q
  xx = x * x
  yy = y * y
  zz = z * z
  xy = x * y
  xz = x * z
  yz = y * z
  wx = w * x
  wy = w * y
  wz = w * z
  return np.array(
      [
          [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
          [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
          [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
      ],
      dtype=float,
  )


def rot_to_quat(rot: np.ndarray) -> np.ndarray:
  m = np.asarray(rot, dtype=float)
  tr = float(np.trace(m))
  if tr > 0.0:
    s = math.sqrt(tr + 1.0) * 2.0
    qw = 0.25 * s
    qx = (m[2, 1] - m[1, 2]) / s
    qy = (m[0, 2] - m[2, 0]) / s
    qz = (m[1, 0] - m[0, 1]) / s
  elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
    s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
    qw = (m[2, 1] - m[1, 2]) / s
    qx = 0.25 * s
    qy = (m[0, 1] + m[1, 0]) / s
    qz = (m[0, 2] + m[2, 0]) / s
  elif m[1, 1] > m[2, 2]:
    s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
    qw = (m[0, 2] - m[2, 0]) / s
    qx = (m[0, 1] + m[1, 0]) / s
    qy = 0.25 * s
    qz = (m[1, 2] + m[2, 1]) / s
  else:
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    qw = (m[1, 0] - m[0, 1]) / s
    qx = (m[0, 2] + m[2, 0]) / s
    qy = (m[1, 2] + m[2, 1]) / s
    qz = 0.25 * s
  q = np.array([qx, qy, qz, qw], dtype=float)
  n = np.linalg.norm(q)
  if n == 0.0:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
  return q / n


def pose_to_matrix(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
  t = np.array([x, y, z], dtype=float)
  r = quat_to_rot(qx, qy, qz, qw)
  tmat = np.eye(4, dtype=float)
  tmat[:3, :3] = r
  tmat[:3, 3] = t
  return tmat


def pose2_to_matrix(x: float, y: float, theta: float) -> np.ndarray:
  c = math.cos(theta)
  s = math.sin(theta)
  tmat = np.eye(4, dtype=float)
  tmat[0, 0] = c
  tmat[0, 1] = -s
  tmat[1, 0] = s
  tmat[1, 1] = c
  tmat[0, 3] = x
  tmat[1, 3] = y
  return tmat


def unpack_upper_triangle_6(values: list[float]) -> np.ndarray:
  if len(values) != 21:
    raise ValueError("expected 21 upper-triangle values for a 6x6 information matrix")
  info = np.zeros((6, 6), dtype=float)
  idx = 0
  for row in range(6):
    for col in range(row, 6):
      info[row, col] = values[idx]
      info[col, row] = values[idx]
      idx += 1
  return info


def unpack_upper_triangle_3(values: list[float]) -> np.ndarray:
  if len(values) != 6:
    raise ValueError("expected 6 upper-triangle values for a 3x3 information matrix")
  info = np.zeros((3, 3), dtype=float)
  idx = 0
  for row in range(3):
    for col in range(row, 3):
      info[row, col] = values[idx]
      info[col, row] = values[idx]
      idx += 1
  return info


def matrix_to_pose(mat: np.ndarray):
  rot = np.asarray(mat[:3, :3], dtype=float)
  trans = np.asarray(mat[:3, 3], dtype=float)
  quat = rot_to_quat(rot)
  return trans, quat


def invert_pose(mat: np.ndarray) -> np.ndarray:
  inv = np.eye(4, dtype=float)
  rot = mat[:3, :3]
  trans = mat[:3, 3]
  inv[:3, :3] = rot.T
  inv[:3, 3] = -rot.T @ trans
  return inv


def compose_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  return a @ b


def parse_g2o_graph(path: Path):
  vertices = {}
  edges = []
  with path.open("r", encoding="utf-8") as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      tokens = line.split()
      kind = tokens[0]
      if kind == "VERTEX_SE3:QUAT":
        gid = int(tokens[1])
        vertices[gid] = pose_to_matrix(*map(float, tokens[2:9]))
      elif kind == "VERTEX_SE2":
        gid = int(tokens[1])
        vertices[gid] = pose2_to_matrix(*map(float, tokens[2:5]))
      elif kind == "EDGE_SE3:QUAT":
        gid1 = int(tokens[1])
        gid2 = int(tokens[2])
        meas = pose_to_matrix(*map(float, tokens[3:10]))
        info = None
        if len(tokens) >= 31:
          info_vals = list(map(float, tokens[10:31]))
          info = unpack_upper_triangle_6(info_vals)
        edges.append(Edge(gid1, gid2, meas, info, 3))
      elif kind == "EDGE_SE2":
        gid1 = int(tokens[1])
        gid2 = int(tokens[2])
        meas = pose2_to_matrix(*map(float, tokens[3:6]))
        info = None
        if len(tokens) >= 12:
          info_vals = list(map(float, tokens[6:12]))
          info = unpack_upper_triangle_3(info_vals)
        edges.append(Edge(gid1, gid2, meas, info, 2))
  return vertices, edges


def parse_pose_rows(path: Path):
  poses = {}
  with path.open("r", encoding="utf-8") as f:
    for line_idx, raw_line in enumerate(f):
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      tokens = line.split()
      if len(tokens) < 8:
        continue
      try:
        gid = int(tokens[0])
        values = list(map(float, tokens[1:8]))
      except ValueError:
        continue
      poses[gid] = pose_to_matrix(*values)
  return poses


def parse_matrix_text(path: Path):
  try:
    dense = np.loadtxt(path)
    if dense.ndim == 2 and dense.shape[0] == 3 and dense.shape[1] % 4 == 0:
      num_poses = dense.shape[1] // 4
      poses = {}
      for gid in range(num_poses):
        mat = np.eye(4, dtype=float)
        mat[:3, 3] = dense[:, gid]
        mat[:3, :3] = dense[:, num_poses + 3 * gid:num_poses + 3 * (gid + 1)]
        poses[gid] = mat
      return poses
    if dense.ndim == 2 and dense.shape[0] == 2 and dense.shape[1] % 3 == 0:
      num_poses = dense.shape[1] // 3
      poses = {}
      for gid in range(num_poses):
        mat = np.eye(4, dtype=float)
        mat[:2, 3] = dense[:, gid]
        mat[:2, :2] = dense[:, num_poses + 2 * gid:num_poses + 2 * (gid + 1)]
        poses[gid] = mat
      return poses
    if dense.ndim == 2 and dense.shape[1] == 3 and dense.shape[0] % 4 == 0:
      num_poses = dense.shape[0] // 4
      poses = {}
      for gid in range(num_poses):
        mat = np.eye(4, dtype=float)
        mat[:3, 3] = dense[gid, :]
        mat[:3, :3] = dense[num_poses + 3 * gid:num_poses + 3 * (gid + 1), :]
        poses[gid] = mat
      return poses
  except ValueError:
    pass

  poses = {}
  auto_id = 0
  with path.open("r", encoding="utf-8") as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      tokens = line.split()
      numbers = []
      all_numeric = True
      for tok in tokens:
        try:
          numbers.append(float(tok))
        except ValueError:
          all_numeric = False
          break
      if not all_numeric:
        continue
      gid = None
      values = None
      if len(numbers) == 12:
        values = numbers
      elif len(numbers) == 13:
        gid = int(numbers[0])
        values = numbers[1:]
      elif len(numbers) == 16:
        values = numbers
      elif len(numbers) == 17:
        gid = int(numbers[0])
        values = numbers[1:]
      else:
        continue
      if gid is None:
        gid = auto_id
        auto_id += 1
      mat = np.eye(4, dtype=float)
      if len(values) == 12:
        mat[:3, :4] = np.array(values, dtype=float).reshape(3, 4)
      elif len(values) == 16:
        mat[:, :] = np.array(values, dtype=float).reshape(4, 4)
      else:
        continue
      poses[gid] = mat
  return poses


def detect_format(path: Path):
  saw_g2o = False
  saw_pose_rows = False
  saw_matrix = False
  with path.open("r", encoding="utf-8") as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      tokens = line.split()
      if tokens[0] in {"VERTEX_SE2", "EDGE_SE2", "VERTEX_SE3:QUAT", "EDGE_SE3:QUAT"}:
        saw_g2o = True
        break
      if len(tokens) == 8:
        try:
          int(tokens[0])
          list(map(float, tokens[1:8]))
          saw_pose_rows = True
          continue
        except ValueError:
          pass
      numeric_count = 0
      for tok in tokens:
        try:
          float(tok)
          numeric_count += 1
        except ValueError:
          numeric_count = -1
          break
      if numeric_count in {12, 13, 16, 17}:
        saw_matrix = True
  if saw_g2o:
    return "g2o"
  if saw_pose_rows:
    return "pose"
  if saw_matrix:
    return "matrix"
  return "pose"


def load_pose_set(path: Path, fmt: str):
  if fmt == "auto":
    fmt = detect_format(path)
  if fmt == "g2o":
    poses, _ = parse_g2o_graph(path)
    return poses, fmt
  if fmt == "pose":
    return parse_pose_rows(path), fmt
  if fmt == "matrix":
    return parse_matrix_text(path), fmt
  raise ValueError(f"unsupported format: {fmt}")


def edge_weights_from_info(edge: Edge, cost_mode: str, weighted: bool) -> tuple[float, float]:
  if edge.info is None:
    return 1.0, 1.0
  if cost_mode == "dpgo":
    if edge.dim == 2:
      trans_info = edge.info[:2, :2]
      try:
        trans_cov = np.linalg.inv(trans_info)
        tau = 2.0 / float(np.trace(trans_cov))
        kappa = float(edge.info[2, 2])
        return tau, kappa
      except np.linalg.LinAlgError:
        return 1.0, 1.0
    trans_info = edge.info[:3, :3]
    rot_info = edge.info[3:6, 3:6]
    try:
      trans_cov = np.linalg.inv(trans_info)
      rot_cov = np.linalg.inv(rot_info)
      tau = 3.0 / float(np.trace(trans_cov))
      kappa = 3.0 / (2.0 * float(np.trace(rot_cov)))
      return tau, kappa
    except np.linalg.LinAlgError:
      return 1.0, 1.0
  if weighted:
    trans_dim = 2 if edge.dim == 2 else 3
    wt = float(np.mean(np.diag(edge.info)[:trans_dim]))
    wr = float(edge.info[2, 2]) if edge.dim == 2 else float(np.mean(np.diag(edge.info)[3:6]))
    return wt, wr
  return 1.0, 1.0


def edge_chordal_cost(edge: Edge, poses: dict[int, np.ndarray], weighted: bool, cost_mode: str) -> tuple[float, bool]:
  if edge.i not in poses or edge.j not in poses:
    return 0.0, False
  ti = poses[edge.i][:3, 3]
  tj = poses[edge.j][:3, 3]
  ri = poses[edge.i][:3, :3]
  rj = poses[edge.j][:3, :3]
  rij = edge.measurement[:3, :3]
  tij = edge.measurement[:3, 3]
  rt = tj - ti - ri @ tij
  rR = rj - ri @ rij
  trans_cost = float(rt @ rt)
  rot_cost = float(np.sum(rR * rR))
  wt, wr = edge_weights_from_info(edge, cost_mode, weighted)
  trans_cost *= wt
  rot_cost *= wr
  scale = 1.0 if cost_mode == "dpgo" else 0.5
  return scale * (trans_cost + rot_cost), True


def compute_chordal_cost(graph_edges, poses, weighted: bool, cost_mode: str):
  total = 0.0
  used = 0
  missing = 0
  for edge in graph_edges:
    cost, ok = edge_chordal_cost(edge, poses, weighted, cost_mode)
    if ok:
      total += cost
      used += 1
    else:
      missing += 1
  return {
      "chordal_cost": total,
      "chordal_cost_per_edge": total / used if used else 0.0,
      "chordal_edges_used": used,
      "chordal_edges_missing": missing,
      "chordal_cost_mode": cost_mode,
  }


def first_common_alignment(estimate: dict[int, np.ndarray], reference: dict[int, np.ndarray]):
  common = sorted(set(estimate) & set(reference))
  if not common:
    raise RuntimeError("no common pose ids between estimate and reference")
  anchor = common[0]
  align = reference[anchor] @ invert_pose(estimate[anchor])
  return align, common


def pose_error_metrics(estimate: dict[int, np.ndarray], reference: dict[int, np.ndarray]):
  align, common = first_common_alignment(estimate, reference)
  trans_errors = []
  rot_errors_deg = []
  for gid in common:
    aligned = compose_pose(align, estimate[gid])
    delta = invert_pose(reference[gid]) @ aligned
    trans_errors.append(float(np.linalg.norm(delta[:3, 3])))
    cos_theta = (np.trace(delta[:3, :3]) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    rot_errors_deg.append(math.degrees(math.acos(cos_theta)))
  trans_errors = np.asarray(trans_errors, dtype=float)
  rot_errors_deg = np.asarray(rot_errors_deg, dtype=float)
  return {
      "common_poses": len(common),
      "alignment_anchor_id": common[0],
      "translation_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
      "translation_mean": float(np.mean(trans_errors)),
      "translation_max": float(np.max(trans_errors)),
      "rotation_rmse_deg": float(np.sqrt(np.mean(rot_errors_deg ** 2))),
      "rotation_mean_deg": float(np.mean(rot_errors_deg)),
      "rotation_max_deg": float(np.max(rot_errors_deg)),
  }


def build_report(args):
  graph_vertices, graph_edges = parse_g2o_graph(Path(args.graph))
  estimate_format = args.estimate_format or args.format
  reference_format = args.reference_format or args.format
  estimate, estimate_fmt = load_pose_set(Path(args.estimate), estimate_format)
  reference, reference_fmt = load_pose_set(Path(args.reference), reference_format)

  report = {
      "graph": str(Path(args.graph)),
      "estimate": str(Path(args.estimate)),
      "reference": str(Path(args.reference)),
      "format": args.format,
      "estimate_format": estimate_fmt,
      "reference_format": reference_fmt,
      "graph_vertices": len(graph_vertices),
      "graph_edges": len(graph_edges),
      "estimate_poses": len(estimate),
      "reference_poses": len(reference),
  }
  report.update(compute_chordal_cost(graph_edges, estimate, args.weighted, args.cost_mode))
  try:
    report.update(pose_error_metrics(estimate, reference))
    report["pose_error_available"] = True
  except RuntimeError as exc:
    report["pose_error_available"] = False
    report["pose_error_error"] = str(exc)
  return report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--graph", required=True)
  parser.add_argument("--estimate", required=True)
  parser.add_argument("--reference", required=True)
  parser.add_argument("--format", default="auto", choices=["auto", "g2o", "pose", "matrix"])
  parser.add_argument("--estimate-format", default=None, choices=["auto", "g2o", "pose", "matrix"])
  parser.add_argument("--reference-format", default=None, choices=["auto", "g2o", "pose", "matrix"])
  parser.add_argument("--weighted", action="store_true")
  parser.add_argument("--cost-mode", default="legacy", choices=["legacy", "dpgo"],
                      help="legacy uses 0.5 unweighted/simplified weighted chordal cost; dpgo uses DPGO/MESA kappa/tau weights without 0.5.")
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  report = build_report(args)
  if args.json:
    print(json.dumps(report, indent=2, sort_keys=True))
    return

  print("PGO evaluation summary")
  print(f"graph: {report['graph']}")
  print(f"estimate: {report['estimate']} ({report['estimate_format']})")
  print(f"reference: {report['reference']} ({report['reference_format']})")
  print(
      f"chordal cost: {report['chordal_cost']:.6f} "
      f"(per-edge {report['chordal_cost_per_edge']:.6f}, used {report['chordal_edges_used']}, "
      f"missing {report['chordal_edges_missing']}, weighted={args.weighted}, mode={args.cost_mode})"
  )
  if report.get("pose_error_available", True):
    print(
        f"pose error after first-pose alignment: "
        f"translation rmse {report['translation_rmse']:.6f}, mean {report['translation_mean']:.6f}, max {report['translation_max']:.6f}; "
        f"rotation rmse {report['rotation_rmse_deg']:.6f} deg, mean {report['rotation_mean_deg']:.6f} deg, "
        f"max {report['rotation_max_deg']:.6f} deg; common poses {report['common_poses']}"
    )
  else:
    print(f"pose error after first-pose alignment: unavailable ({report.get('pose_error_error', 'unknown error')})")


if __name__ == "__main__":
  main()

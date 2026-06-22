#!/usr/bin/env python3
"""Evaluate DRAN-object semantic pose output against robot-local g2o GT.

The estimate is the `object_poses.txt` written by
`run_object_drone_experiment.sh`:

  type robot_id local_vertex_id object_id x y z qx qy qz qw

Ground truth is a directory of `rob_*_gt.g2o` files.  The script aligns the
estimate to GT by the first common trajectory pose, then reports trajectory and
object errors.  Object errors are reported both per robot-local object copy and
after averaging aligned object copies per object id.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
  q = np.asarray([qx, qy, qz, qw], dtype=float)
  n = np.linalg.norm(q)
  if n == 0.0:
    return np.eye(3)
  q = q / n
  x, y, z, w = q
  return np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ], dtype=float)


def pose_to_matrix(values: list[float]) -> np.ndarray:
  x, y, z, qx, qy, qz, qw = values
  mat = np.eye(4, dtype=float)
  mat[:3, :3] = quat_to_rot(qx, qy, qz, qw)
  mat[:3, 3] = [x, y, z]
  return mat


def invert_pose(mat: np.ndarray) -> np.ndarray:
  inv = np.eye(4, dtype=float)
  inv[:3, :3] = mat[:3, :3].T
  inv[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
  return inv


def rot_error_deg(a: np.ndarray, b: np.ndarray) -> float:
  delta = a[:3, :3].T @ b[:3, :3]
  cos_theta = (np.trace(delta) - 1.0) * 0.5
  cos_theta = max(-1.0, min(1.0, float(cos_theta)))
  return math.degrees(math.acos(cos_theta))


def sorted_robot_files(directory: Path):
  def robot_index(path: Path) -> int:
    match = re.match(r"rob_(\d+)(?:_gt)?\.g2o$", path.name)
    return int(match.group(1)) if match else 10**9
  return sorted(directory.glob("rob_*_gt.g2o"), key=lambda p: (robot_index(p), p.name))


def infer_object_start(vertex_ids: list[int], edges: list[tuple[int, int]]) -> int:
  if not vertex_ids:
    return 0
  consecutive = {tuple(sorted(edge)) for edge in edges}
  trajectory_end = vertex_ids[0]
  for prev, cur in zip(vertex_ids, vertex_ids[1:]):
    if cur != prev + 1 or (prev, cur) not in consecutive:
      break
    trajectory_end = cur
  return trajectory_end + 1


def parse_gt_dir(directory: Path, num_objects: int | None):
  trajectories = {}
  objects = {}
  for path in sorted_robot_files(directory):
    match = re.match(r"rob_(\d+)(?:_gt)?\.g2o$", path.name)
    if not match:
      continue
    robot_id = int(match.group(1))
    vertices = {}
    edges = []
    with path.open("r", encoding="utf-8") as f:
      for raw in f:
        tokens = raw.strip().split()
        if not tokens:
          continue
        if tokens[0] == "VERTEX_SE3:QUAT":
          vertices[int(tokens[1])] = pose_to_matrix(list(map(float, tokens[2:9])))
        elif tokens[0] == "EDGE_SE3:QUAT":
          edges.append((int(tokens[1]), int(tokens[2])))
    vertex_ids = sorted(vertices)
    object_count = num_objects
    if object_count is None:
      object_start = infer_object_start(vertex_ids, edges)
      object_count = max(vertex_ids) - object_start + 1 if vertex_ids else 0
    else:
      object_start = max(vertex_ids) + 1 - object_count
    for local_id, pose in vertices.items():
      if local_id < object_start:
        trajectories[(robot_id, local_id)] = pose
      elif local_id < object_start + object_count:
        objects[(robot_id, local_id, local_id - object_start)] = pose
  return trajectories, objects


def parse_estimate(path: Path):
  trajectories = {}
  objects = {}
  with path.open("r", encoding="utf-8") as f:
    for raw in f:
      tokens = raw.strip().split()
      if not tokens or tokens[0] == "type":
        continue
      if len(tokens) != 11:
        continue
      kind = tokens[0]
      robot_id = int(tokens[1])
      local_id = int(tokens[2])
      object_id = int(tokens[3])
      pose = pose_to_matrix(list(map(float, tokens[4:11])))
      if kind == "trajectory":
        trajectories[(robot_id, local_id)] = pose
      elif kind == "object":
        objects[(robot_id, local_id, object_id)] = pose
  return trajectories, objects


def metric_summary(trans_errors: list[float], rot_errors: list[float], prefix: str):
  if not trans_errors:
    return {
        f"{prefix}_count": 0,
        f"{prefix}_translation_rmse": None,
        f"{prefix}_translation_mean": None,
        f"{prefix}_translation_max": None,
        f"{prefix}_rotation_rmse_deg": None,
        f"{prefix}_rotation_mean_deg": None,
        f"{prefix}_rotation_max_deg": None,
    }
  t = np.asarray(trans_errors, dtype=float)
  r = np.asarray(rot_errors, dtype=float)
  return {
      f"{prefix}_count": int(len(t)),
      f"{prefix}_translation_rmse": float(np.sqrt(np.mean(t * t))),
      f"{prefix}_translation_mean": float(np.mean(t)),
      f"{prefix}_translation_max": float(np.max(t)),
      f"{prefix}_rotation_rmse_deg": float(np.sqrt(np.mean(r * r))),
      f"{prefix}_rotation_mean_deg": float(np.mean(r)),
      f"{prefix}_rotation_max_deg": float(np.max(r)),
  }


def average_translations(poses: list[np.ndarray]) -> np.ndarray:
  avg = np.eye(4, dtype=float)
  avg[:3, 3] = np.mean([pose[:3, 3] for pose in poses], axis=0)
  avg[:3, :3] = poses[0][:3, :3]
  return avg


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--estimate", required=True)
  parser.add_argument("--gt-dir", required=True)
  parser.add_argument("--num-objects", type=int, default=None)
  parser.add_argument("--json", action="store_true")
  args = parser.parse_args()

  est_traj, est_obj = parse_estimate(Path(args.estimate))
  gt_traj, gt_obj = parse_gt_dir(Path(args.gt_dir), args.num_objects)
  common_traj = sorted(set(est_traj) & set(gt_traj))
  if not common_traj:
    raise RuntimeError("no common trajectory poses between estimate and GT")
  anchor = common_traj[0]
  align = gt_traj[anchor] @ invert_pose(est_traj[anchor])

  traj_t = []
  traj_r = []
  for key in common_traj:
    aligned = align @ est_traj[key]
    gt = gt_traj[key]
    traj_t.append(float(np.linalg.norm(aligned[:3, 3] - gt[:3, 3])))
    traj_r.append(rot_error_deg(gt, aligned))

  common_obj = sorted(set(est_obj) & set(gt_obj))
  obj_t = []
  obj_r = []
  aligned_obj_by_id = defaultdict(list)
  gt_obj_by_id = defaultdict(list)
  for key in common_obj:
    aligned = align @ est_obj[key]
    gt = gt_obj[key]
    obj_t.append(float(np.linalg.norm(aligned[:3, 3] - gt[:3, 3])))
    obj_r.append(rot_error_deg(gt, aligned))
    aligned_obj_by_id[key[2]].append(aligned)
    gt_obj_by_id[key[2]].append(gt)

  mean_obj_t = []
  mean_obj_r = []
  for object_id in sorted(set(aligned_obj_by_id) & set(gt_obj_by_id)):
    aligned_mean = average_translations(aligned_obj_by_id[object_id])
    gt_mean = average_translations(gt_obj_by_id[object_id])
    mean_obj_t.append(float(np.linalg.norm(aligned_mean[:3, 3] - gt_mean[:3, 3])))
    mean_obj_r.append(rot_error_deg(gt_mean, aligned_mean))

  report = {
      "estimate": str(Path(args.estimate)),
      "gt_dir": str(Path(args.gt_dir)),
      "alignment_anchor": {"robot_id": anchor[0], "local_vertex_id": anchor[1]},
      "estimate_trajectory_poses": len(est_traj),
      "gt_trajectory_poses": len(gt_traj),
      "estimate_object_poses": len(est_obj),
      "gt_object_poses": len(gt_obj),
  }
  report.update(metric_summary(traj_t, traj_r, "trajectory"))
  report.update(metric_summary(obj_t, obj_r, "object_copy"))
  report.update(metric_summary(mean_obj_t, mean_obj_r, "object_mean"))

  if args.json:
    print(json.dumps(report, indent=2, sort_keys=True))
    return
  print("Object DRAN GT evaluation")
  for key, value in report.items():
    print(f"{key}: {value}")


if __name__ == "__main__":
  main()

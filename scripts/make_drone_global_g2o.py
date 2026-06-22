#!/usr/bin/env python3
"""Merge a directory of rob_*.g2o files into one centralized SE3 g2o.

Drone robot files use local vertex ids in each rob_*.g2o by default.  The
default id mode therefore treats (robot_id, local_vertex_id) as the unique pose
key and rewrites all vertices into one contiguous global id range.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


def parse_g2o(path: Path):
  vertices = []
  vertex_order = []
  edges = []
  passthrough = []
  with path.open("r", encoding="utf-8") as f:
    for raw_line in f:
      line = raw_line.strip()
      if not line or line.startswith("#"):
        passthrough.append(raw_line)
        continue
      tokens = line.split()
      kind = tokens[0]
      if kind.startswith("VERTEX"):
        gid = int(tokens[1])
        vertices.append((gid, tokens))
        vertex_order.append(gid)
      elif kind.startswith("EDGE"):
        edges.append(tokens)
      else:
        passthrough.append(raw_line)
  return vertices, vertex_order, edges, passthrough


def pose_key(tokens):
  return tuple(tokens)


def edge_key(tokens):
  return tuple(tokens)


def max_vertex_id(vertices, edges):
  ids = [gid for gid, _ in vertices]
  for tokens in edges:
    ids.extend([int(tokens[1]), int(tokens[2])])
  if not ids:
    return 0
  return max(ids)


def infer_object_start(vertex_order, edges):
  if not vertex_order:
    return 0
  consecutive_edges = set()
  for tokens in edges:
    lo = min(int(tokens[1]), int(tokens[2]))
    hi = max(int(tokens[1]), int(tokens[2]))
    consecutive_edges.add((lo, hi))
  trajectory_end = vertex_order[0]
  for prev, cur in zip(vertex_order, vertex_order[1:]):
    if cur != prev + 1:
      break
    if (prev, cur) not in consecutive_edges:
      break
    trajectory_end = cur
  return trajectory_end + 1


def sorted_robot_files(input_dir: Path):
  def robot_index(path: Path):
    match = re.match(r"rob_(\d+)(?:_gt)?\.g2o$", path.name)
    return int(match.group(1)) if match else 10**9

  files = sorted(input_dir.glob("rob_*.g2o"), key=lambda p: (robot_index(p), p.name))
  return files


def merge_directory(input_dir: Path, output_path: Path, start_index: int, id_mode: str, num_objects: int):
  files = sorted_robot_files(input_dir)
  if not files:
    raise RuntimeError(f"no rob_*.g2o files found in {input_dir}")

  merged_vertices = {}
  merged_vertex_order = []
  merged_edges = []
  edge_counter = Counter()
  file_remaps = {}
  duplicate_vertex_lines = 0
  conflicting_vertex_lines = 0
  passthrough = []
  skipped_edges = 0
  skipped_object_object_edges = 0

  parsed_files = []
  inferred_counts = []
  for path in files:
    vertices, vertex_order, edges, file_passthrough = parse_g2o(path)
    object_start = infer_object_start(vertex_order, edges)
    max_id = max_vertex_id(vertices, edges)
    inferred_count = max_id - object_start + 1 if max_id >= object_start else 0
    inferred_counts.append(inferred_count)
    parsed_files.append((path, vertices, vertex_order, edges, file_passthrough, object_start))

  shared_num_objects = num_objects
  if id_mode == "object-aware" and shared_num_objects == 0:
    shared_num_objects = min(inferred_counts) if inferred_counts else 0

  for path, vertices, vertex_order, edges, file_passthrough, inferred_object_start in parsed_files:
    robot_match = re.match(r"rob_(\d+)(?:_gt)?\.g2o$", path.name)
    robot_idx = int(robot_match.group(1)) if robot_match else len(file_remaps)
    if not passthrough:
      passthrough = file_passthrough
    if id_mode == "object-aware":
      object_start = max_vertex_id(vertices, edges) + 1 - shared_num_objects
    else:
      object_start = inferred_object_start

    def local_key(gid):
      if id_mode == "shared-global":
        return gid
      if id_mode == "object-aware" and object_start <= gid < object_start + shared_num_objects:
        return ("object", gid - object_start)
      return (robot_idx, gid)

    local_to_key = {}
    seen_local_vertices = {}
    for gid, tokens in vertices:
      key = local_key(gid)
      local_to_key[gid] = key
      if gid in seen_local_vertices:
        duplicate_vertex_lines += 1
        if pose_key(seen_local_vertices[gid]) != pose_key(tokens):
          conflicting_vertex_lines += 1
      else:
        seen_local_vertices[gid] = tokens
      if key not in merged_vertices:
        merged_vertices[key] = tokens
        merged_vertex_order.append(key)
      else:
        duplicate_vertex_lines += 1
        if pose_key(merged_vertices[key]) != pose_key(tokens):
          conflicting_vertex_lines += 1
    file_remaps[path.name] = local_to_key
    for tokens in edges:
      src = int(tokens[1])
      dst = int(tokens[2])
      if src not in local_to_key or dst not in local_to_key:
        skipped_edges += 1
        continue
      if (id_mode == "object-aware" and
          object_start <= src < object_start + shared_num_objects and
          object_start <= dst < object_start + shared_num_objects):
        skipped_object_object_edges += 1
        continue
      rewritten = list(tokens)
      rewritten[1] = local_to_key[src]
      rewritten[2] = local_to_key[dst]
      merged_edges.append(rewritten)
      edge_counter[edge_key(rewritten)] += 1

  remap = {}
  for offset, key in enumerate(merged_vertex_order):
    remap[key] = start_index + offset

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as out:
    for key in merged_vertex_order:
      tokens = merged_vertices[key]
      mapped_id = remap[key]
      out.write(
        f"{tokens[0]} {mapped_id} " + " ".join(tokens[2:]) + "\n"
      )
    for tokens in merged_edges:
      src = tokens[1]
      dst = tokens[2]
      if src not in remap or dst not in remap:
        continue
      out.write(
        f"{tokens[0]} {remap[src]} {remap[dst]} " + " ".join(tokens[3:]) + "\n"
      )

  duplicate_edge_lines = sum(count - 1 for count in edge_counter.values() if count > 1)
  duplicate_edge_pairs = sum(1 for count in edge_counter.values() if count > 1)
  summary = {
    "input_dir": str(input_dir),
    "output": str(output_path),
    "files": len(files),
    "vertices_unique": len(merged_vertex_order),
    "vertex_duplicate_lines": duplicate_vertex_lines,
    "vertex_conflicts": conflicting_vertex_lines,
    "edges_total": len(merged_edges),
    "edges_skipped_missing_vertices": skipped_edges,
    "edges_skipped_object_object": skipped_object_object_edges,
    "edge_duplicate_lines": duplicate_edge_lines,
    "edge_duplicate_entries": duplicate_edge_pairs,
    "start_index": start_index,
    "id_mode": id_mode,
    "shared_objects": shared_num_objects,
  }
  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--input-dir", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--start-index", type=int, default=0)
  parser.add_argument(
      "--id-mode",
      choices=["robot-local", "shared-global", "object-aware"],
      default="robot-local",
      help="Use robot-local ids, preserve shared global ids, or merge inferred object suffix vertices across robots.",
  )
  parser.add_argument("--num-objects", type=int, default=0,
                      help="Shared object suffix size for --id-mode object-aware; 0 infers the common suffix size.")
  args = parser.parse_args()

  summary = merge_directory(Path(args.input_dir), Path(args.output), args.start_index, args.id_mode, args.num_objects)
  print("Merged drone g2o summary")
  for key in [
      "input_dir",
      "output",
      "files",
      "vertices_unique",
      "vertex_duplicate_lines",
      "vertex_conflicts",
      "edges_total",
      "edge_duplicate_lines",
      "edge_duplicate_entries",
      "edges_skipped_missing_vertices",
      "edges_skipped_object_object",
      "start_index",
      "id_mode",
      "shared_objects",
  ]:
    print(f"{key}: {summary[key]}")


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path


SUMMARY_FIELDS = [
    "run_dir",
    "run_name",
    "topology_source",
    "topology_rounds",
    "num_robots",
    "num_objects",
    "undirected_edges",
    "candidate_messages",
    "total_impact",
    "top_10pct_impact_share",
    "top_1pct_impact_share",
    "total_existing_comm_poses",
    "outer_existing_comm_poses",
    "oracle_total_budget_impact_share",
    "oracle_outer_budget_impact_share",
    "payload_bytes",
    "oracle_total_budget_mb",
    "oracle_outer_budget_mb",
    "max_translation_error",
    "max_rotation_error_rad",
    "mean_translation_error",
    "mean_rotation_error_rad",
    "diagnostic_note",
]


class Pose:
  def __init__(
      self,
      t: tuple[float, float, float],
      q: tuple[float, float, float, float],
  ) -> None:
    self.t = t
    self.q = q


def fmt(value: float | int | str | None) -> str:
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  if isinstance(value, int):
    return str(value)
  if math.isnan(value) or math.isinf(value):
    return ""
  text = f"{value:.6f}".rstrip("0").rstrip(".")
  return text if text else "0"


def as_float(value: str | None, default: float = math.nan) -> float:
  if value is None or value == "":
    return default
  try:
    return float(value)
  except ValueError:
    return default


def as_int(value: str | None, default: int = 0) -> int:
  value_float = as_float(value)
  if math.isnan(value_float):
    return default
  return int(round(value_float))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


def normalize_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
  norm = math.sqrt(sum(component * component for component in q))
  if norm == 0.0:
    return (0.0, 0.0, 0.0, 1.0)
  return tuple(component / norm for component in q)  # type: ignore[return-value]


def rotation_angle_rad(a: Pose, b: Pose) -> float:
  qa = normalize_quat(a.q)
  qb = normalize_quat(b.q)
  dot = abs(sum(x * y for x, y in zip(qa, qb)))
  dot = max(-1.0, min(1.0, dot))
  return 2.0 * math.acos(dot)


def translation_error(a: Pose, b: Pose) -> float:
  return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a.t, b.t)))


def parse_object_poses(path: Path) -> dict[tuple[int, int], Pose]:
  objects: dict[tuple[int, int], Pose] = {}
  with path.open(encoding="utf-8") as handle:
    for raw in handle:
      tokens = raw.strip().split()
      if not tokens or tokens[0] == "type":
        continue
      if len(tokens) != 11 or tokens[0] != "object":
        continue
      robot_id = int(tokens[1])
      object_id = int(tokens[3])
      values = list(map(float, tokens[4:11]))
      objects[(robot_id, object_id)] = Pose(
          (values[0], values[1], values[2]),
          (values[3], values[4], values[5], values[6]),
      )
  return objects


def ring_edges(num_robots: int, hops: int) -> list[tuple[int, int, float]]:
  edges = set()
  hops = max(1, min(hops, max(1, num_robots - 1)))
  for i in range(num_robots):
    for offset in range(1, hops + 1):
      j = (i + offset) % num_robots
      if i != j:
        edges.add(tuple(sorted((i, j))))
  return [(i, j, 1.0) for i, j in sorted(edges)]


def complete_edges(num_robots: int) -> list[tuple[int, int, float]]:
  return [(i, j, 1.0) for i in range(num_robots) for j in range(i + 1, num_robots)]


def parse_run_log(run_dir: Path) -> dict[str, str]:
  path = run_dir / "run.log"
  info: dict[str, str] = {}
  if not path.exists():
    return info
  with path.open(encoding="utf-8", errors="ignore") as handle:
    for raw in handle:
      loaded = re.search(r"Loaded\s+(\d+)\s+robots,\s+(\d+)\s+shared objects,\s+dimension\s+(\d+)", raw)
      if loaded:
        info["num_robots"] = loaded.group(1)
        info["num_objects"] = loaded.group(2)
        info["dimension"] = loaded.group(3)
      if "Object consensus topology:" in raw:
        head, *parts = raw.strip().rstrip(".").split("|")
        info["topology"] = head.split(":", 1)[1].strip()
        for part in parts:
          if "=" not in part:
            continue
          key, value = part.split("=", 1)
          info[key.strip()] = value.strip()
  return info


def load_topology_edges(path: Path, weight_mode: str) -> tuple[list[tuple[int, int, float]], int]:
  rows = read_csv_rows(path)
  edges: list[tuple[int, int, float]] = []
  rounds = set()
  for row in rows:
    src = int(row["src"])
    dst = int(row["dst"])
    weight = 1.0 if weight_mode == "unit" else float(row.get("weight", "1"))
    edges.append((min(src, dst), max(src, dst), weight))
    rounds.add(int(row.get("round", "0")))
  unique_edges = sorted(set(edges))
  return unique_edges, len(rounds) if rounds else 1


def infer_topology_from_log(run_dir: Path, num_robots: int) -> tuple[list[tuple[int, int, float]], str, int]:
  info = parse_run_log(run_dir)
  topology_file = info.get("topology_file", "<none>")
  weight_mode = info.get("topology_weight_mode", "unit")
  if topology_file and topology_file != "<none>":
    path = Path(topology_file)
    if not path.is_absolute():
      path = Path.cwd() / path
    edges, rounds = load_topology_edges(path, weight_mode)
    return edges, f"file:{topology_file}", rounds

  topology = info.get("topology", "ring")
  if topology == "complete":
    return complete_edges(num_robots), "complete", 1
  hops = as_int(info.get("ring_hops"), 1)
  return ring_edges(num_robots, hops), "ring", 1


def score_candidates(
    objects: dict[tuple[int, int], Pose],
    edges: list[tuple[int, int, float]],
    rotation_scale: float,
) -> list[dict[str, float | int | str]]:
  object_ids_by_robot: dict[int, set[int]] = {}
  for robot_id, object_id in objects:
    object_ids_by_robot.setdefault(robot_id, set()).add(object_id)

  candidates: list[dict[str, float | int | str]] = []
  for src, dst, weight in edges:
    common_objects = sorted(
        object_ids_by_robot.get(src, set()) & object_ids_by_robot.get(dst, set())
    )
    for object_id in common_objects:
      src_pose = objects[(src, object_id)]
      dst_pose = objects[(dst, object_id)]
      trans = translation_error(src_pose, dst_pose)
      rot = rotation_angle_rad(src_pose, dst_pose)
      impact = weight * (trans * trans + (rotation_scale * rot) * (rotation_scale * rot))
      for a, b in ((src, dst), (dst, src)):
        candidates.append({
            "src": a,
            "dst": b,
            "object_id": object_id,
            "weight": weight,
            "translation_error": trans,
            "rotation_error_rad": rot,
            "impact": impact,
        })
  candidates.sort(key=lambda item: (-float(item["impact"]), int(item["src"]), int(item["dst"]), int(item["object_id"])))
  return candidates


def impact_share(candidates: list[dict[str, float | int | str]], budget: int) -> float:
  total = sum(float(item["impact"]) for item in candidates)
  if total <= 0.0 or budget <= 0:
    return 0.0
  kept = sum(float(item["impact"]) for item in candidates[:min(budget, len(candidates))])
  return kept / total


def top_fraction_share(candidates: list[dict[str, float | int | str]], fraction: float) -> float:
  if not candidates:
    return 0.0
  count = max(1, int(math.ceil(len(candidates) * fraction)))
  return impact_share(candidates, count)


def read_existing_comm_budget(run_dir: Path) -> tuple[int, int]:
  path = run_dir / "iterations.csv"
  rows = read_csv_rows(path)
  if not rows:
    return 0, 0
  final = rows[-1]
  total = as_int(final.get("cumulative_object_comm_poses"), -1)
  if total < 0:
    total = sum(as_int(row.get("object_comm_poses"), 0) for row in rows)
  init = 0
  for row in rows:
    if as_int(row.get("iter"), -1) == 0:
      init = as_int(row.get("object_comm_poses"), 0)
      break
  return total, max(0, total - init)


def summarize_run(
    run_dir: Path,
    payload_bytes: int = 96,
    rotation_scale: float = 1.0,
) -> dict[str, str]:
  objects = parse_object_poses(run_dir / "object_poses.txt")
  if not objects:
    raise ValueError(f"no object rows found in {run_dir / 'object_poses.txt'}")

  info = parse_run_log(run_dir)
  num_robots = as_int(info.get("num_robots"), max(robot for robot, _ in objects) + 1)
  num_objects = as_int(info.get("num_objects"), len({obj for _, obj in objects}))
  edges, topology_source, topology_rounds = infer_topology_from_log(run_dir, num_robots)
  candidates = score_candidates(objects, edges, rotation_scale)
  total_budget, outer_budget = read_existing_comm_budget(run_dir)
  impacts = [float(item["impact"]) for item in candidates]
  trans_errors = [float(item["translation_error"]) for item in candidates]
  rot_errors = [float(item["rotation_error_rad"]) for item in candidates]

  total_impact = sum(impacts)
  notes = [
      "final-state proxy only",
      "no per-iteration send mask",
  ]
  if not info:
    notes.append("missing run.log metadata")

  return {
      "run_dir": str(run_dir),
      "run_name": run_dir.name,
      "topology_source": topology_source,
      "topology_rounds": fmt(topology_rounds),
      "num_robots": fmt(num_robots),
      "num_objects": fmt(num_objects),
      "undirected_edges": fmt(len(edges)),
      "candidate_messages": fmt(len(candidates)),
      "total_impact": fmt(total_impact),
      "top_10pct_impact_share": fmt(top_fraction_share(candidates, 0.10)),
      "top_1pct_impact_share": fmt(top_fraction_share(candidates, 0.01)),
      "total_existing_comm_poses": fmt(total_budget),
      "outer_existing_comm_poses": fmt(outer_budget),
      "oracle_total_budget_impact_share": fmt(impact_share(candidates, total_budget)),
      "oracle_outer_budget_impact_share": fmt(impact_share(candidates, outer_budget)),
      "payload_bytes": fmt(payload_bytes),
      "oracle_total_budget_mb": fmt(total_budget * payload_bytes / (1024 * 1024)),
      "oracle_outer_budget_mb": fmt(outer_budget * payload_bytes / (1024 * 1024)),
      "max_translation_error": fmt(max(trans_errors) if trans_errors else math.nan),
      "max_rotation_error_rad": fmt(max(rot_errors) if rot_errors else math.nan),
      "mean_translation_error": fmt(
          sum(trans_errors) / len(trans_errors) if trans_errors else math.nan
      ),
      "mean_rotation_error_rad": fmt(
          sum(rot_errors) / len(rot_errors) if rot_errors else math.nan
      ),
      "diagnostic_note": "; ".join(notes),
  }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def parse_args(argv: list[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Offline final-state innovation replay for DRAN-object runs."
  )
  parser.add_argument("--run-dir", action="append", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--payload-bytes", type=int, default=96)
  parser.add_argument("--rotation-scale", type=float, default=1.0)
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    rows = [
        summarize_run(run_dir, args.payload_bytes, args.rotation_scale)
        for run_dir in args.run_dir
    ]
    write_csv(args.output, rows)
  except (OSError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

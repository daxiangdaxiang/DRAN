#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_dran_object_innovation.py"


def load_module():
  spec = importlib.util.spec_from_file_location("object_innovation", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def write_iterations(path: Path, rows: list[dict[str, object]]) -> None:
  fields: list[str] = []
  for row in rows:
    for key in row:
      if key not in fields:
        fields.append(key)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)


def write_object_poses(path: Path, rows: list[tuple[int, int, float, float, float]]) -> None:
  with path.open("w") as handle:
    handle.write("type robot_id local_vertex_id object_id x y z qx qy qz qw\n")
    for robot_id, object_id, x, y, z in rows:
      handle.write(
          f"object {robot_id} {100 + object_id} {object_id} "
          f"{x} {y} {z} 0 0 0 1\n"
      )


class ObjectInnovationReplayTest(unittest.TestCase):
  def test_scores_directed_object_messages(self) -> None:
    module = load_module()
    objects = {
        (0, 0): module.Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (1, 0): module.Pose((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (0, 1): module.Pose((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (1, 1): module.Pose((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
    }

    candidates = module.score_candidates(objects, [(0, 1, 1.0)], rotation_scale=1.0)

    self.assertEqual(len(candidates), 4)
    self.assertEqual(sum(1 for item in candidates if item["impact"] > 0), 2)
    self.assertEqual(module.fmt(sum(item["impact"] for item in candidates)), "2")
    best = candidates[0]
    self.assertEqual(best["src"], 0)
    self.assertEqual(best["dst"], 1)
    self.assertEqual(best["object_id"], 0)
    self.assertEqual(module.fmt(best["translation_error"]), "1")
    self.assertEqual(module.fmt(best["impact"]), "1")

  def test_summarize_run_infers_ring_topology_and_budget(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "ring_run"
      run_dir.mkdir()
      write_object_poses(run_dir / "object_poses.txt", [
          (0, 0, 1.0, 0.0, 0.0),
          (1, 0, 0.0, 1.0, 0.0),
          (2, 0, -1.0, 0.0, 0.0),
          (3, 0, 0.0, -1.0, 0.0),
      ])
      write_iterations(run_dir / "iterations.csv", [
          {"iter": "0", "object_comm_poses": "1", "cumulative_object_comm_poses": "1"},
          {"iter": "1", "object_comm_poses": "1", "cumulative_object_comm_poses": "2"},
      ])
      (run_dir / "run.log").write_text(
          "Loaded 4 robots, 1 shared objects, dimension 3.\n"
          "Object consensus topology: ring | ring_hops = 1 | "
          "topology_file = <none> | topology_weight_mode = unit.\n"
      )

      row = module.summarize_run(run_dir)

      self.assertEqual(row["run_name"], "ring_run")
      self.assertEqual(row["topology_source"], "ring")
      self.assertEqual(row["num_robots"], "4")
      self.assertEqual(row["num_objects"], "1")
      self.assertEqual(row["undirected_edges"], "4")
      self.assertEqual(row["candidate_messages"], "8")
      self.assertEqual(row["total_existing_comm_poses"], "2")
      self.assertEqual(row["outer_existing_comm_poses"], "1")
      self.assertEqual(row["oracle_total_budget_impact_share"], "0.25")
      self.assertEqual(row["oracle_outer_budget_impact_share"], "0.125")

  def test_cli_writes_summary_csv(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      run_dir = tmp_path / "cli_run"
      run_dir.mkdir()
      output = tmp_path / "innovation.csv"
      write_object_poses(run_dir / "object_poses.txt", [
          (0, 0, 0.0, 0.0, 0.0),
          (1, 0, 1.0, 0.0, 0.0),
          (0, 1, 0.0, 0.0, 0.0),
          (1, 1, 0.0, 0.0, 0.0),
      ])
      write_iterations(run_dir / "iterations.csv", [
          {"iter": "0", "object_comm_poses": "1", "cumulative_object_comm_poses": "1"},
          {"iter": "1", "object_comm_poses": "1", "cumulative_object_comm_poses": "2"},
      ])
      (run_dir / "run.log").write_text(
          "Loaded 2 robots, 2 shared objects, dimension 3.\n"
          "Object consensus topology: ring | ring_hops = 1 | "
          "topology_file = <none> | topology_weight_mode = unit.\n"
      )

      code = module.main(["--run-dir", str(run_dir), "--output", str(output)])

      self.assertEqual(code, 0)
      rows = list(csv.DictReader(output.open()))
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]["run_name"], "cli_run")
      self.assertEqual(rows[0]["candidate_messages"], "4")
      self.assertEqual(rows[0]["oracle_total_budget_impact_share"], "1")


if __name__ == "__main__":
  unittest.main()

#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_dran_boundary_response.py"


def load_module():
  spec = importlib.util.spec_from_file_location("boundary_response", SCRIPT)
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


def write_object_poses(path: Path, count: int) -> None:
  with path.open("w") as handle:
    handle.write("type robot_id local_vertex_id object_id x y z qx qy qz qw\n")
    for idx in range(count):
      handle.write(f"object 0 {100 + idx} {idx} {idx}.0 0 0 0 0 0 1\n")


class BoundaryResponseOracleTest(unittest.TestCase):
  def test_summarize_run_reports_boundary_response_efficiency(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "boundary_run"
      run_dir.mkdir()
      write_iterations(run_dir / "iterations.csv", [
          {
              "iter": "0",
              "chordal_measurement_cost": "10.0",
              "measurement_cost": "11.0",
              "object_comm_payload_blocks": "4",
              "comm_mb": "0.5",
              "cumulative_comm_mb": "0.5",
              "receiver_boundary_disagreement_sum": "2.0",
              "receiver_boundary_cache_delta_sum": "1.0",
          },
          {
              "iter": "1",
              "chordal_measurement_cost": "7.0",
              "measurement_cost": "8.0",
              "object_comm_payload_blocks": "6",
              "comm_mb": "0.25",
              "cumulative_comm_mb": "0.75",
              "receiver_boundary_disagreement_sum": "3.0",
              "receiver_boundary_cache_delta_sum": "2.0",
              "known_object_copies": "2",
              "relay_object_copies": "1",
          },
      ])
      write_object_poses(run_dir / "object_poses.txt", 3)
      (run_dir / "gt_eval.json").write_text(json.dumps({
          "trajectory_translation_rmse": 0.12,
          "trajectory_rotation_rmse_rad": 0.03,
          "object_translation_rmse": 0.4,
          "object_rotation_rmse_rad": 0.05,
      }))

      row = module.summarize_run(run_dir)

      self.assertEqual(row["run_name"], "boundary_run")
      self.assertEqual(row["has_receiver_boundary"], "true")
      self.assertEqual(row["cost_field"], "chordal_measurement_cost")
      self.assertEqual(row["final_cost"], "7")
      self.assertEqual(row["cost_drop"], "3")
      self.assertEqual(row["object_pose_rows"], "3")
      self.assertEqual(row["payload_blocks_total"], "10")
      self.assertEqual(row["total_comm_mb"], "0.75")
      self.assertEqual(row["receiver_boundary_disagreement_sum_total"], "5")
      self.assertEqual(row["receiver_boundary_cache_delta_sum_total"], "3")
      self.assertEqual(row["boundary_disagreement_per_mb"], "6.666667")
      self.assertEqual(row["boundary_cache_delta_per_mb"], "4")
      self.assertEqual(row["selected_proxy_efficiency"], "0.5")
      self.assertEqual(row["gt_trajectory_translation_rmse"], "0.12")
      self.assertEqual(row["gt_object_rotation_rmse_rad"], "0.05")

  def test_legacy_schema_marks_missing_receiver_boundary(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "legacy_run"
      run_dir.mkdir()
      write_iterations(run_dir / "iterations.csv", [
          {
              "iter": "0",
              "measurement_cost": "4.0",
              "object_comm_payload_blocks": "2",
              "comm_mb": "0.2",
              "cumulative_comm_mb": "0.2",
          },
          {
              "iter": "1",
              "measurement_cost": "3.0",
              "object_comm_payload_blocks": "3",
              "comm_mb": "0.3",
              "cumulative_comm_mb": "0.5",
          },
      ])
      write_object_poses(run_dir / "object_poses.txt", 1)

      row = module.summarize_run(run_dir)

      self.assertEqual(row["has_receiver_boundary"], "false")
      self.assertEqual(row["receiver_boundary_disagreement_sum_total"], "0")
      self.assertEqual(row["receiver_boundary_cache_delta_sum_total"], "0")
      self.assertEqual(row["boundary_disagreement_per_mb"], "0")
      self.assertEqual(row["selected_proxy_efficiency"], "0")
      self.assertIn("missing receiver_boundary diagnostics", row["diagnostic_note"])

  def test_cli_accepts_multiple_run_dirs_and_writes_csv(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      first = tmp_path / "first"
      second = tmp_path / "second"
      first.mkdir()
      second.mkdir()
      output = tmp_path / "boundary.csv"
      for run_dir, cost in ((first, "5.0"), (second, "6.0")):
        write_iterations(run_dir / "iterations.csv", [
            {
                "iter": "0",
                "measurement_cost": cost,
                "object_comm_payload_blocks": "1",
                "comm_mb": "0.1",
                "receiver_boundary_disagreement_sum": "1.0",
                "receiver_boundary_cache_delta_sum": "0.5",
            },
        ])
        write_object_poses(run_dir / "object_poses.txt", 1)

      code = module.main([
          "--run-dir",
          str(first),
          "--run-dir",
          str(second),
          "--output",
          str(output),
      ])

      self.assertEqual(code, 0)
      rows = list(csv.DictReader(output.open()))
      self.assertEqual([row["run_name"] for row in rows], ["first", "second"])
      self.assertEqual(rows[0]["boundary_disagreement_per_mb"], "10")


if __name__ == "__main__":
  unittest.main()

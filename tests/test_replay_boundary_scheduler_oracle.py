#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "replay_boundary_scheduler_oracle.py"


def load_module():
  spec = importlib.util.spec_from_file_location("boundary_scheduler_oracle", SCRIPT)
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


class ReplayBoundarySchedulerOracleTest(unittest.TestCase):
  def test_summarize_run_uses_full_boundary_predictive_fields(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "full_run"
      run_dir.mkdir()
      write_iterations(run_dir / "iteration_summary.csv", [
          {
              "iter": "0",
              "chordal_measurement_cost": "10.0",
              "measurement_cost": "20.0",
              "total_comm_mb": "1.0",
              "receiver_boundary_updates": "4",
              "boundary_candidate_count": "10",
              "boundary_predictive_selected_gain_fraction": "0.25",
              "receiver_boundary_disagreement_sum": "8.0",
          },
          {
              "iter": "1",
              "chordal_measurement_cost": "7.0",
              "measurement_cost": "15.0",
              "total_comm_mb": "1.5",
              "receiver_boundary_updates": "6",
              "boundary_candidate_count": "15",
              "boundary_predictive_selected_gain_fraction": "0.75",
              "receiver_boundary_disagreement_sum": "12.0",
          },
      ])

      row = module.summarize_run(run_dir)

      self.assertEqual(row["run_dir"], str(run_dir))
      self.assertEqual(row["iteration_file"], "iteration_summary.csv")
      self.assertEqual(row["cost_field"], "chordal_measurement_cost")
      self.assertEqual(row["num_rows"], "2")
      self.assertEqual(row["initial_cost"], "10")
      self.assertEqual(row["final_cost"], "7")
      self.assertEqual(row["cost_drop"], "3")
      self.assertEqual(row["total_comm_mb"], "1.5")
      self.assertEqual(row["total_selected_updates"], "10")
      self.assertEqual(row["total_candidate_updates"], "25")
      self.assertEqual(row["avg_selected_fraction"], "0.4")
      self.assertEqual(row["avg_predicted_gain_fraction"], "0.5")
      self.assertEqual(row["avg_disagreement_per_update"], "2")
      self.assertIn("aggregate-level oracle", row["diagnostic_notes"])

  def test_summarize_run_falls_back_for_missing_boundary_fields(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "fallback_run"
      run_dir.mkdir()
      write_iterations(run_dir / "iteration_summary.csv", [
          {
              "iter": "0",
              "measurement_cost": "5.0",
              "cumulative_comm_mb": "0.2",
              "object_comm_poses": "3",
              "receiver_boundary_updates": "2",
              "consensus_rejected_updates": "1",
              "consensus_correction_updates": "4",
          },
          {
              "iter": "1",
              "measurement_cost": "4.0",
              "cumulative_comm_mb": "0.5",
              "object_comm_poses": "5",
              "receiver_boundary_updates": "3",
              "consensus_rejected_updates": "2",
              "consensus_correction_updates": "0",
          },
      ])

      row = module.summarize_run(run_dir)

      self.assertEqual(row["iteration_file"], "iteration_summary.csv")
      self.assertEqual(row["cost_field"], "measurement_cost")
      self.assertEqual(row["total_comm_mb"], "0.5")
      self.assertEqual(row["total_selected_updates"], "5")
      self.assertEqual(row["total_candidate_updates"], "12")
      self.assertEqual(row["avg_selected_fraction"], "0.416667")
      self.assertEqual(row["avg_predicted_gain_fraction"], "")
      self.assertEqual(row["avg_disagreement_per_update"], "")
      self.assertIn("missing chordal_measurement_cost", row["diagnostic_notes"])
      self.assertIn("missing boundary_predictive_selected_gain_fraction", row["diagnostic_notes"])
      self.assertIn("approximate candidates", row["diagnostic_notes"])

  def test_cli_writes_one_row_per_run(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      first = tmp_path / "first"
      second = tmp_path / "second"
      first.mkdir()
      second.mkdir()
      output = tmp_path / "oracle.csv"
      write_iterations(first / "iteration_summary.csv", [
          {"measurement_cost": "3", "total_comm_mb": "0.1", "object_comm_poses": "1"},
      ])
      write_iterations(second / "iteration_summary.csv", [
          {"measurement_cost": "4", "total_comm_mb": "0.2", "object_comm_poses": "2"},
      ])

      code = module.main([str(first), str(second), "--output", str(output)])

      self.assertEqual(code, 0)
      rows = list(csv.DictReader(output.open()))
      self.assertEqual([row["run_dir"] for row in rows], [str(first), str(second)])
      self.assertEqual(rows[1]["final_cost"], "4")

  def test_summarize_run_accepts_object_iterations_csv(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "object_run"
      run_dir.mkdir()
      write_iterations(run_dir / "iterations.csv", [
          {
              "iter": "0",
              "chordal_measurement_cost": "8",
              "cumulative_comm_mb": "0.1",
              "receiver_boundary_updates": "2",
          },
          {
              "iter": "1",
              "chordal_measurement_cost": "6",
              "cumulative_comm_mb": "0.3",
              "receiver_boundary_updates": "4",
          },
      ])

      row = module.summarize_run(run_dir)

      self.assertEqual(row["iteration_file"], "iterations.csv")
      self.assertEqual(row["final_cost"], "6")
      self.assertEqual(row["total_comm_mb"], "0.3")

  def test_missing_iteration_summary_raises_clear_error(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "missing"
      run_dir.mkdir()

      with self.assertRaisesRegex(FileNotFoundError, "missing iteration_summary.csv or iterations.csv"):
        module.summarize_run(run_dir)


if __name__ == "__main__":
  unittest.main()

#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_interface_state_oracle.py"


def load_module():
  spec = importlib.util.spec_from_file_location("interface_state_oracle", SCRIPT)
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


class InterfaceStateOracleTest(unittest.TestCase):
  def test_object_predictive_fields_measure_interface_signal(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "object_predictive"
      run_dir.mkdir()
      write_iterations(run_dir / "iterations.csv", [
          {
              "iter": "0",
              "chordal_measurement_cost": "10.0",
              "cumulative_comm_mb": "0.1",
              "object_comm_payload_blocks": "1",
              "boundary_predictive_gain_sum": "0.0",
              "boundary_predictive_selected_gain_sum": "0.0",
              "boundary_predictive_stiffness_sum": "0.0",
              "boundary_predictive_selected_gain_fraction": "0.0",
          },
          {
              "iter": "1",
              "chordal_measurement_cost": "8.0",
              "cumulative_comm_mb": "0.6",
              "object_comm_payload_blocks": "4",
              "boundary_predictive_gain_sum": "3.0",
              "boundary_predictive_selected_gain_sum": "2.0",
              "boundary_predictive_stiffness_sum": "5.0",
              "boundary_predictive_selected_gain_fraction": "0.6666667",
          },
          {
              "iter": "2",
              "chordal_measurement_cost": "7.0",
              "cumulative_comm_mb": "0.85",
              "object_comm_payload_blocks": "2",
              "boundary_predictive_gain_sum": "2.0",
              "boundary_predictive_selected_gain_sum": "1.0",
              "boundary_predictive_stiffness_sum": "4.0",
              "boundary_predictive_selected_gain_fraction": "0.5",
          },
      ])

      row = module.summarize_run(run_dir)

      self.assertEqual(row["run_name"], "object_predictive")
      self.assertEqual(row["iteration_file"], "iterations.csv")
      self.assertEqual(row["cost_field"], "chordal_measurement_cost")
      self.assertEqual(row["initial_cost"], "10")
      self.assertEqual(row["final_cost"], "7")
      self.assertEqual(row["cost_drop"], "3")
      self.assertEqual(row["total_comm_mb"], "0.85")
      self.assertEqual(row["total_payload_blocks"], "7")
      self.assertEqual(row["total_predicted_gain"], "5")
      self.assertEqual(row["total_selected_predicted_gain"], "3")
      self.assertEqual(row["total_stiffness"], "9")
      self.assertEqual(row["mean_selected_gain_fraction"], "0.388889")
      self.assertEqual(row["cost_drop_per_comm_mb"], "3.529412")
      self.assertEqual(row["cost_drop_per_selected_gain"], "1")
      self.assertEqual(row["selected_gain_to_next_cost_drop_corr"], "1")
      self.assertIn("object boundary_predictive fields", row["signal_source"])

  def test_boundary_model_fields_are_marked_as_diagnostic_fallback(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "pure_boundary_model"
      run_dir.mkdir()
      write_iterations(run_dir / "iteration_summary.csv", [
          {
              "iter": "0",
              "cost": "5.0",
              "iter_comm_mb": "0.1",
              "boundary_model_merit_decrease": "0.0",
              "boundary_model_merit_grad_decrease": "0.0",
          },
          {
              "iter": "1",
              "cost": "4.5",
              "iter_comm_mb": "0.2",
              "boundary_model_merit_decrease": "0.25",
              "boundary_model_merit_grad_decrease": "0.5",
          },
      ])

      row = module.summarize_run(run_dir)

      self.assertEqual(row["iteration_file"], "iteration_summary.csv")
      self.assertEqual(row["cost_field"], "cost")
      self.assertEqual(row["total_comm_mb"], "0.3")
      self.assertEqual(row["total_selected_predicted_gain"], "0.25")
      self.assertEqual(row["selected_gain_to_next_cost_drop_corr"], "")
      self.assertIn("boundary_model merit fallback", row["signal_source"])
      self.assertIn("diagnostic fallback", row["diagnostic_notes"])

  def test_direct_interface_state_fields_are_summarized_without_gain_claim(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "interface_state"
      run_dir.mkdir()
      write_iterations(run_dir / "iterations.csv", [
          {
              "iter": "0",
              "cost": "5.0",
              "comm_mb": "0.1",
              "interface_state_payload_blocks": "2",
              "interface_state_grad_norm": "3.0",
              "interface_state_stiffness_sum": "10.0",
              "interface_state_comm_mb": "0.2",
          },
          {
              "iter": "1",
              "cost": "4.0",
              "comm_mb": "0.2",
              "interface_state_payload_blocks": "4",
              "interface_state_grad_norm": "5.0",
              "interface_state_stiffness_sum": "20.0",
              "interface_state_comm_mb": "0.4",
          },
      ])

      row = module.summarize_run(run_dir)

      self.assertEqual(row["signal_source"], "interface_state diagnostic fields")
      self.assertEqual(row["total_payload_blocks"], "6")
      self.assertEqual(row["total_stiffness"], "30")
      self.assertEqual(row["total_predicted_gain"], "")
      self.assertEqual(row["total_selected_predicted_gain"], "")
      self.assertEqual(row["total_interface_state_grad_norm"], "8")
      self.assertEqual(row["interface_state_total_comm_mb"], "0.6")
      self.assertIn("not a predicted-gain signal", row["diagnostic_notes"])

  def test_cli_writes_summary_rows(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      first = tmp_path / "first"
      second = tmp_path / "second"
      first.mkdir()
      second.mkdir()
      output = tmp_path / "interface_oracle.csv"
      write_iterations(first / "iterations.csv", [
          {
              "measurement_cost": "3.0",
              "cumulative_comm_mb": "0.1",
              "boundary_predictive_selected_gain_sum": "1.0",
          },
      ])
      write_iterations(second / "iterations.csv", [
          {
              "measurement_cost": "4.0",
              "cumulative_comm_mb": "0.2",
              "boundary_predictive_selected_gain_sum": "2.0",
          },
      ])

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
      self.assertEqual(rows[1]["total_selected_predicted_gain"], "2")


if __name__ == "__main__":
  unittest.main()

#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_dran_object_diagnostics.py"


def load_module():
  spec = importlib.util.spec_from_file_location("object_diag", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


class ObjectDiagnosticsTest(unittest.TestCase):
  def test_legacy_iterations_summary(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "legacy_run"
      run_dir.mkdir()
      write_csv(run_dir / "iterations.csv", [
          {
              "iter": "0",
              "measurement_cost": "10.0",
              "consensus_cost": "0.5",
              "primal_residual": "2.0",
              "dual_residual": "2.0",
              "total_gradnorm": "3.0",
              "object_comm_poses": "100",
              "comm_mb": "1.0",
              "cumulative_object_comm_poses": "100",
              "cumulative_comm_mb": "1.0",
          },
          {
              "iter": "1",
              "measurement_cost": "7.5",
              "consensus_cost": "0.25",
              "primal_residual": "1.0",
              "dual_residual": "0.5",
              "total_gradnorm": "2.0",
              "object_comm_poses": "10",
              "comm_mb": "0.1",
              "cumulative_object_comm_poses": "110",
              "cumulative_comm_mb": "1.1",
          },
      ])

      row = module.summarize_run(run_dir, optimal_cost=6.0)

      self.assertEqual(row["run_name"], "legacy_run")
      self.assertEqual(row["cost_field"], "measurement_cost")
      self.assertEqual(row["num_iters"], "2")
      self.assertEqual(row["initial_cost"], "10")
      self.assertEqual(row["final_cost"], "7.5")
      self.assertEqual(row["best_cost"], "7.5")
      self.assertEqual(row["cost_drop"], "2.5")
      self.assertEqual(row["final_cost_gap"], "1.5")
      self.assertEqual(row["init_comm_mb"], "1")
      self.assertEqual(row["outer_comm_mb"], "0.1")
      self.assertEqual(row["total_comm_mb"], "1.1")
      self.assertEqual(row["final_comm_poses"], "110")
      self.assertEqual(row["cost_drop_per_total_mb"], "2.272727")
      self.assertEqual(row["stale_cache_skipped_total"], "")
      self.assertEqual(row["rejected_updates_total"], "")

  def test_extended_iterations_summary(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp) / "extended_run"
      run_dir.mkdir()
      write_csv(run_dir / "iterations.csv", [
          {
              "iter": "0",
              "measurement_cost": "9.5",
              "chordal_measurement_cost": "9.0",
              "consensus_cost": "0.4",
              "object_comm_poses": "20",
              "object_comm_payload_blocks": "40",
              "comm_mb": "0.5",
              "cumulative_object_comm_poses": "20",
              "cumulative_comm_mb": "0.5",
              "known_object_copies": "42",
              "relay_object_copies": "12",
              "active_consensus_pairs": "21",
              "stale_cache_skipped": "3",
              "innovation_gate_skipped": "2",
              "innovation_gate_forced": "5",
              "consensus_rejected_updates": "1",
              "main_merit_rejected_updates": "2",
              "gauge_merit_rejected_updates": "4",
              "frame_sync_objective": "0.125",
          },
          {
              "iter": "1",
              "measurement_cost": "8.5",
              "chordal_measurement_cost": "7.0",
              "consensus_cost": "0.3",
              "object_comm_poses": "4",
              "object_comm_payload_blocks": "8",
              "comm_mb": "0.125",
              "cumulative_object_comm_poses": "24",
              "cumulative_comm_mb": "0.625",
              "known_object_copies": "42",
              "relay_object_copies": "12",
              "active_consensus_pairs": "20",
              "stale_cache_skipped": "5",
              "innovation_gate_skipped": "7",
              "innovation_gate_forced": "11",
              "consensus_rejected_updates": "0",
              "main_merit_rejected_updates": "1",
              "gauge_merit_rejected_updates": "0",
              "frame_sync_objective": "0.1",
          },
      ])

      row = module.summarize_run(run_dir)

      self.assertEqual(row["cost_field"], "chordal_measurement_cost")
      self.assertEqual(row["initial_cost"], "9")
      self.assertEqual(row["final_cost"], "7")
      self.assertEqual(row["payload_blocks_total"], "48")
      self.assertEqual(row["stale_cache_skipped_total"], "8")
      self.assertEqual(row["innovation_gate_skipped_total"], "9")
      self.assertEqual(row["innovation_gate_forced_total"], "16")
      self.assertEqual(row["rejected_updates_total"], "8")
      self.assertEqual(row["final_known_object_copies"], "42")
      self.assertEqual(row["final_relay_object_copies"], "12")
      self.assertEqual(row["final_active_consensus_pairs"], "20")
      self.assertEqual(row["final_frame_sync_objective"], "0.1")

  def test_cli_writes_output_csv(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      run_dir = tmp_path / "cli_run"
      run_dir.mkdir()
      output = tmp_path / "diagnostics.csv"
      write_csv(run_dir / "iterations.csv", [
          {
              "iter": "0",
              "measurement_cost": "4.0",
              "object_comm_poses": "2",
              "comm_mb": "0.25",
              "cumulative_object_comm_poses": "2",
              "cumulative_comm_mb": "0.25",
          },
          {
              "iter": "1",
              "measurement_cost": "3.0",
              "object_comm_poses": "1",
              "comm_mb": "0.25",
              "cumulative_object_comm_poses": "3",
              "cumulative_comm_mb": "0.5",
          },
      ])

      code = module.main([
          "--run-dir",
          str(run_dir),
          "--output",
          str(output),
          "--optimal-cost",
          "2.5",
      ])

      self.assertEqual(code, 0)
      rows = list(csv.DictReader(output.open()))
      self.assertEqual(len(rows), 1)
      self.assertEqual(rows[0]["run_name"], "cli_run")
      self.assertEqual(rows[0]["final_cost_gap"], "0.5")


if __name__ == "__main__":
  unittest.main()

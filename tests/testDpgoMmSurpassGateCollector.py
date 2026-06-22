#!/usr/bin/env python3
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_comparison(path: Path) -> None:
  rows = [
      {
          "dataset": "alpha",
          "updates": "8",
          "variant": "dpgo_mm_amm",
          "status": "ok",
          "global_cost": "10.0",
          "gradient": "2.0",
          "total_comm_mb": "1.0",
          "wall_time_sec": "3.0",
          "solver_time_per_node_sec": "0.3",
          "run_root": "runs/alpha/amm",
          "summary_csv": "runs/alpha/amm/summary.csv",
      },
      {
          "dataset": "alpha",
          "updates": "8",
          "variant": "manual_reduced",
          "status": "ok",
          "global_cost": "9.5",
          "gradient": "1.5",
          "total_comm_mb": "1.0",
          "wall_time_sec": "4.0",
          "solver_time_per_node_sec": "0.4",
          "run_root": "runs/alpha/manual",
          "summary_csv": "runs/alpha/manual/summary.csv",
      },
      {
          "dataset": "beta",
          "updates": "8",
          "variant": "dpgo_mm_amm",
          "status": "ok",
          "global_cost": "5.0",
          "gradient": "1.0",
          "total_comm_mb": "2.0",
          "wall_time_sec": "2.0",
          "solver_time_per_node_sec": "0.2",
          "run_root": "runs/beta/amm",
          "summary_csv": "runs/beta/amm/summary.csv",
      },
      {
          "dataset": "beta",
          "updates": "8",
          "variant": "manual_reduced",
          "status": "ok",
          "global_cost": "5.5",
          "gradient": "1.2",
          "total_comm_mb": "2.4",
          "wall_time_sec": "2.5",
          "solver_time_per_node_sec": "0.25",
          "run_root": "runs/beta/manual",
          "summary_csv": "runs/beta/manual/summary.csv",
      },
  ]
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as handle:
    return [dict(row) for row in csv.DictReader(handle)]


class DpgoMmSurpassGateCollectorTest(unittest.TestCase):
  def test_collects_gate_a_rows_pairwise_and_aggregate(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      comparison = tmp_path / "comparison.csv"
      out_dir = tmp_path / "out"
      write_comparison(comparison)

      result = subprocess.run(
          [
              sys.executable,
              "scripts/collect_dpgo_mm_surpass_gates.py",
              "--comparison",
              f"A:smoke:{comparison}",
              "--out-dir",
              str(out_dir),
          ],
          cwd=ROOT,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=True,
      )

      self.assertIn("Wrote 4 normalized rows", result.stdout)

      normalized = read_csv(out_dir / "normalized_gate_rows.csv")
      manual_alpha = next(
          row for row in normalized
          if row["dataset"] == "alpha" and row["method"] == "manual_reduced")
      self.assertEqual(manual_alpha["gate"], "A")
      self.assertEqual(manual_alpha["claim_scope"], "same-init optimizer")
      self.assertEqual(manual_alpha["init_mode"],
                       "centralized_chordal_same_init")
      self.assertEqual(manual_alpha["dist_init"], "false")
      self.assertEqual(manual_alpha["init_comm_mb"], "0")
      self.assertEqual(manual_alpha["outer_comm_mb"], "1")
      self.assertEqual(manual_alpha["deployable_flag"], "diagnostic_same_init")

      pairwise = read_csv(out_dir / "pairwise_to_dpgo_mm_amm.csv")
      alpha = next(row for row in pairwise if row["dataset"] == "alpha")
      beta = next(row for row in pairwise if row["dataset"] == "beta")
      self.assertEqual(alpha["cost_better"], "true")
      self.assertEqual(alpha["same_or_lower_comm"], "true")
      self.assertEqual(alpha["same_comm_cost_win"], "true")
      self.assertEqual(beta["cost_better"], "false")
      self.assertEqual(beta["same_or_lower_comm"], "false")
      self.assertEqual(beta["same_comm_cost_win"], "false")

      aggregate = read_csv(out_dir / "aggregate_by_method.csv")
      manual = next(row for row in aggregate
                    if row["method"] == "manual_reduced")
      self.assertEqual(manual["num_datasets"], "2")
      self.assertEqual(manual["cost_win_count"], "1")
      self.assertEqual(manual["same_comm_cost_win_count"], "1")
      summary = (out_dir / "summary.md").read_text()
      self.assertIn("DPGO-MM AMM Surpass Gate Summary", summary)
      self.assertIn("| A | smoke | manual_reduced | 1/2 | 1/2 |", summary)

  def test_preserves_deployment_payload_and_convergence_fields(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      comparison = tmp_path / "comparison.csv"
      out_dir = tmp_path / "out"
      rows = [
          {
              "dataset": "sphere",
              "outer_updates": "20",
              "variant": "dpgo_mm_amm",
              "status": "ok",
              "final_cost": "100",
              "gradient": "3",
              "init_mode": "deployment_init",
              "dist_init": "true",
              "init_comm_mb": "0.25",
              "outer_comm_mb": "1.0",
              "total_comm_mb": "1.25",
              "payload_breakdown": "init_pose=0.25;outer_pose=1.0",
              "convergence_iter": "18",
              "convergence_comm_mb": "1.1",
              "topology": "ring",
              "wall_time_sec": "10",
              "solver_time_per_node_sec": "1",
          },
          {
              "dataset": "sphere",
              "outer_updates": "20",
              "variant": "manual_reduced",
              "status": "ok",
              "final_cost": "99",
              "gradient": "2",
              "init_mode": "deployment_init",
              "dist_init": "true",
              "init_comm_mb": "0.20",
              "outer_comm_mb": "0.9",
              "total_comm_mb": "1.10",
              "payload_breakdown": "init_pose=0.20;outer_pose=0.9",
              "convergence_iter": "16",
              "convergence_comm_mb": "0.95",
              "topology": "ring",
              "wall_time_sec": "12",
              "solver_time_per_node_sec": "1.5",
          },
      ]
      with comparison.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

      subprocess.run(
          [
              sys.executable,
              "scripts/collect_dpgo_mm_surpass_gates.py",
              "--comparison",
              f"B:deploy:{comparison}",
              "--out-dir",
              str(out_dir),
          ],
          cwd=ROOT,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=True,
      )

      normalized = read_csv(out_dir / "normalized_gate_rows.csv")
      manual = next(row for row in normalized
                    if row["method"] == "manual_reduced")
      self.assertEqual(manual["init_comm_mb"], "0.20")
      self.assertEqual(manual["outer_comm_mb"], "0.9")
      self.assertEqual(manual["total_comm_mb"], "1.10")
      self.assertEqual(manual["payload_breakdown"],
                       "init_pose=0.20;outer_pose=0.9")
      self.assertEqual(manual["convergence_iter"], "16")
      self.assertEqual(manual["convergence_comm_mb"], "0.95")
      self.assertEqual(manual["topology"], "ring")
      self.assertEqual(manual["deployable_flag"], "candidate")

      pairwise = read_csv(out_dir / "pairwise_to_dpgo_mm_amm.csv")
      compared = next(row for row in pairwise
                      if row["method"] == "manual_reduced")
      self.assertEqual(compared["payload_breakdown"],
                       "init_pose=0.20;outer_pose=0.9")
      self.assertEqual(compared["convergence_iter"], "16")
      self.assertEqual(compared["convergence_comm_mb"], "0.95")

  def test_rejects_invalid_comparison_spec(self):
    with tempfile.TemporaryDirectory() as tmp:
      result = subprocess.run(
          [
              sys.executable,
              "scripts/collect_dpgo_mm_surpass_gates.py",
              "--comparison",
              "Z:bad:/tmp/missing.csv",
              "--out-dir",
              tmp,
          ],
          cwd=ROOT,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )

      self.assertNotEqual(result.returncode, 0)
      self.assertIn("gate must be one of A, B, or C", result.stdout)


if __name__ == "__main__":
  unittest.main()

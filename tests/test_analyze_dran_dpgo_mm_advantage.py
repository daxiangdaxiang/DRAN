#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_dran_dpgo_mm_advantage.py"


def load_module():
  spec = importlib.util.spec_from_file_location("advantage", SCRIPT)
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


class AdvantageAnalysisTest(unittest.TestCase):
  def test_payload_normalization_and_dpgo_mm_proxy(self) -> None:
    module = load_module()
    with tempfile.TemporaryDirectory() as tmp:
      tmp_path = Path(tmp)
      data_dir = tmp_path / "data"
      data_dir.mkdir()
      (data_dir / "toy3d.g2o").write_text(
          "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n"
          "EDGE_SE3:QUAT 0 1 0 0 0 0 0 0 1 1 0 0 0 1 0 0 1\n"
      )
      metrics = tmp_path / "normalized_metrics.csv"
      write_csv(metrics, [
          {
              "problem_type": "six",
              "dataset": "toy3d",
              "method": "DRAN",
              "setting": "fixed_20",
              "budget": "20",
              "status": "ok",
              "cost": "12.0",
              "cost_gap_abs": "2.0",
              "total_comm_poses": "10",
              "outer_comm_mb": "0.00152587890625",
              "total_comm_mb": "0.00152587890625",
              "final_iter": "19",
          },
          {
              "problem_type": "six",
              "dataset": "toy3d",
              "method": "DPGO-MM",
              "setting": "fixed_20",
              "budget": "20",
              "status": "ok",
              "cost": "10.0",
              "cost_gap_abs": "0.0",
              "total_comm_poses": "4",
              "outer_comm_mb": "0.0003662109375",
              "total_comm_mb": "0.0003662109375",
              "final_iter": "19",
          },
      ])
      curves = tmp_path / "curves.csv"
      write_csv(curves, [
          {
              "problem_type": "six",
              "dataset": "toy3d",
              "method": "DPGO-MM",
              "setting": "fixed_20",
              "budget": "20",
              "iter": "0",
              "cumulative_comm_mb": "0.000091552734375",
          },
          {
              "problem_type": "six",
              "dataset": "toy3d",
              "method": "DPGO-MM",
              "setting": "fixed_20",
              "budget": "20",
              "iter": "1",
              "cumulative_comm_mb": "0.00018310546875",
          },
      ])
      rows = module.build_summary(
          normalized_metrics=metrics,
          curve_paths=[curves],
          data_dir=data_dir,
          setting="fixed_20",
          budget="20",
          init_proxy_rounds=900,
      )
      self.assertEqual(len(rows), 1)
      row = rows[0]
      self.assertEqual(row["dimension"], "3")
      self.assertEqual(row["dpgo_mm_payload_bytes_per_pose"], "96")
      self.assertEqual(row["dran_payload_normalized_comm_mb"], "0.000915527")
      self.assertEqual(row["dpgo_mm_iter0_comm_poses"], "1")
      self.assertEqual(row["proxy_init_comm_mb"], "0.082397461")
      self.assertEqual(row["dpgo_mm_outer_comm_mb"], "0.000366211")
      self.assertEqual(row["proxy_total_comm_mb"], "0.082763672")
      self.assertEqual(row["final_gap_dran_minus_dpgo_mm"], "2")
      self.assertEqual(row["reported_comm_mb_dran_minus_dpgo_mm"], "0.001159668")
      self.assertEqual(row["payload_normalized_comm_mb_dran_minus_dpgo_mm"], "0.000549316")
      self.assertEqual(row["proxy_total_comm_mb_dran_minus_dpgo_mm"], "-0.081237793")
      self.assertEqual(row["dpgo_mm_dominates_reported"], "true")
      self.assertEqual(row["dpgo_mm_dominates_payload_normalized"], "true")
      self.assertEqual(row["dpgo_mm_dominates_with_init_proxy"], "false")


if __name__ == "__main__":
  unittest.main()

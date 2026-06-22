#!/usr/bin/env python3
import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ObjectJacobiAdaptiveBudgetSmokeTest(unittest.TestCase):
    def test_adaptive_budget_stops_object_jacobi_after_minimum_rounds(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "object-based-multi-robot-example"
        data_dir = (
            root
            / "data"
            / "chordal_dataset"
            / "drone_g2o_file"
            / "range_10_tau_0.01_ang_error_1"
        )
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")
        if not data_dir.exists():
            self.skipTest(f"missing data dir: {data_dir}")

        env = os.environ.copy()
        env["DPGO_OBJECT_INIT_SUMMARY_OUTPUT"] = ""
        env["OBJECT_INIT_OBJECT_JACOBI_ITERS"] = "10"
        env["OBJECT_INIT_OBJECT_JACOBI_SCOPE"] = "all"
        env["OBJECT_INIT_OBJECT_JACOBI_ADAPTIVE_BUDGET"] = "true"
        env["OBJECT_INIT_OBJECT_JACOBI_MIN_ITERS"] = "2"
        env["OBJECT_INIT_OBJECT_JACOBI_BUDGET_STEP_TOL"] = "999"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "iters.csv"
            pose_path = tmp / "poses.txt"
            init_summary_path = tmp / "init_summary.csv"
            env["DPGO_OBJECT_INIT_SUMMARY_OUTPUT"] = str(init_summary_path)
            result = subprocess.run(
                [
                    str(binary),
                    str(data_dir),
                    "21",
                    "0",
                    "0",
                    "1",
                    "0.5",
                    str(csv_path),
                    "0.001",
                    "100",
                    str(pose_path),
                    "ring",
                    "1",
                    "1",
                    "0.01",
                    "distributed_chordal_object_jacobi",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            with init_summary_path.open() as handle:
                rows = list(csv.DictReader(handle))

        self.assertIn("object_jacobi_adaptive_budget = true", result.stdout)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["object_jacobi_adaptive_budget"], "true")
        self.assertEqual(row["object_jacobi_rotation_iters"], "2")
        self.assertEqual(row["object_jacobi_translation_iters"], "2")
        self.assertEqual(
            row["object_jacobi_rotation_stop_reason"], "adaptive_step_tol"
        )
        self.assertEqual(
            row["object_jacobi_translation_stop_reason"], "adaptive_step_tol"
        )
        self.assertLess(float(row["object_jacobi_comm_mb"]), 1.0)


if __name__ == "__main__":
    unittest.main()

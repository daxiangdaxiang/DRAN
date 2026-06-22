#!/usr/bin/env python3
import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ObjectInnovationGateModeSmokeTest(unittest.TestCase):
    def test_high_threshold_suppresses_noncoverage_object_messages(self):
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
        env["OBJECT_COMMUNICATION_POLICY"] = "innovation_gated"
        env["OBJECT_INNOVATION_THRESHOLD"] = "999"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "iters.csv"
            pose_path = tmp / "poses.txt"
            result = subprocess.run(
                [
                    str(binary),
                    str(data_dir),
                    "21",
                    "0",
                    "2",
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
                    "raw",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = list(csv.DictReader(csv_path.open()))

        self.assertIn("communication_policy = innovation_gated", result.stdout)
        self.assertEqual(rows[0]["object_comm_poses"], "2520")
        self.assertEqual(
            rows[0]["receiver_boundary_updates"], rows[0]["object_comm_poses"]
        )
        self.assertEqual(rows[1]["object_comm_poses"], "0")
        self.assertEqual(rows[1]["receiver_boundary_updates"], "0")
        self.assertEqual(float(rows[1]["receiver_boundary_cache_delta_sum"]), 0.0)
        self.assertEqual(float(rows[1]["receiver_boundary_disagreement_sum"]), 0.0)
        self.assertGreater(int(rows[1]["innovation_gate_skipped"]), 0)
        self.assertEqual(rows[1]["innovation_gate_forced"], "0")


if __name__ == "__main__":
    unittest.main()

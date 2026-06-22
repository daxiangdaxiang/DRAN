#!/usr/bin/env python3
import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


class ObjectReceiverImpactDiagnosticsSmokeTest(unittest.TestCase):
    def test_receiver_boundary_metrics_are_logged_for_object_messages(self):
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

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "iters.csv"
            pose_path = tmp / "poses.txt"
            subprocess.run(
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
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            with csv_path.open() as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames or []

        required_fields = {
            "receiver_boundary_updates",
            "receiver_boundary_cold_starts",
            "receiver_boundary_cache_delta_sum",
            "receiver_boundary_cache_delta_max",
            "receiver_boundary_disagreement_sum",
            "receiver_boundary_disagreement_max",
        }
        self.assertTrue(required_fields.issubset(set(fieldnames)))
        self.assertEqual(
            rows[0]["receiver_boundary_updates"], rows[0]["object_comm_poses"]
        )
        self.assertGreaterEqual(
            int(rows[0]["receiver_boundary_cold_starts"]), 0
        )
        self.assertGreater(
            float(rows[0]["receiver_boundary_disagreement_sum"]), 0.0
        )
        self.assertGreater(
            float(rows[0]["receiver_boundary_disagreement_max"]), 0.0
        )
        self.assertGreaterEqual(
            float(rows[0]["receiver_boundary_cache_delta_sum"]), 0.0
        )
        self.assertGreaterEqual(
            float(rows[0]["receiver_boundary_cache_delta_max"]), 0.0
        )


if __name__ == "__main__":
    unittest.main()

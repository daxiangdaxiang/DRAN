#!/usr/bin/env python3
import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ObjectBoundaryBudgetModeSmokeTest(unittest.TestCase):
    def run_object_drone(self, extra_env):
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
        env.update(extra_env)

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
            with csv_path.open() as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                fieldnames = reader.fieldnames or []

        return result.stdout, rows, fieldnames

    def test_high_budget_preserves_first_round_forced_coverage(self):
        stdout, rows, fieldnames = self.run_object_drone(
            {
                "OBJECT_COMMUNICATION_POLICY": "boundary_budget",
                "OBJECT_BOUNDARY_BUDGET_POSE_FRACTION": "1.0",
            }
        )

        required_fields = {
            "boundary_budget_candidates",
            "boundary_budget_selected",
            "boundary_budget_forced",
            "boundary_budget_skipped",
            "boundary_budget_score_sum",
            "boundary_budget_selected_score_sum",
        }
        self.assertTrue(required_fields.issubset(set(fieldnames)))
        self.assertIn("communication_policy = boundary_budget", stdout)
        self.assertEqual(rows[0]["object_comm_poses"], "2520")
        self.assertEqual(rows[0]["boundary_budget_forced"], "2520")
        self.assertEqual(rows[0]["boundary_budget_skipped"], "0")
        self.assertEqual(
            rows[0]["receiver_boundary_updates"], rows[0]["object_comm_poses"]
        )

    def test_low_budget_skips_lower_scored_second_round_candidates(self):
        _, rows, _ = self.run_object_drone(
            {
                "OBJECT_COMMUNICATION_POLICY": "boundary_budget",
                "OBJECT_BOUNDARY_BUDGET_MAX_POSES": "1",
            }
        )

        second = rows[1]
        self.assertGreater(int(second["boundary_budget_candidates"]), 0)
        self.assertLess(
            int(second["boundary_budget_selected"]),
            int(second["boundary_budget_candidates"]),
        )
        self.assertGreater(int(second["boundary_budget_skipped"]), 0)
        self.assertGreaterEqual(
            float(second["boundary_budget_score_sum"]),
            float(second["boundary_budget_selected_score_sum"]),
        )

    def test_predictive_boundary_budget_logs_predicted_gain(self):
        stdout, rows, fieldnames = self.run_object_drone(
            {
                "OBJECT_COMMUNICATION_POLICY": "boundary_predictive",
                "OBJECT_BOUNDARY_BUDGET_MAX_POSES": "1",
            }
        )

        required_fields = {
            "boundary_predictive_gain_sum",
            "boundary_predictive_selected_gain_sum",
            "boundary_predictive_stiffness_sum",
        }
        self.assertTrue(required_fields.issubset(set(fieldnames)))
        self.assertIn("communication_policy = boundary_predictive", stdout)
        second = rows[1]
        self.assertGreater(int(second["boundary_budget_candidates"]), 0)
        self.assertLess(
            int(second["boundary_budget_selected"]),
            int(second["boundary_budget_candidates"]),
        )
        self.assertGreater(float(second["boundary_predictive_gain_sum"]), 0.0)
        self.assertGreaterEqual(
            float(second["boundary_predictive_gain_sum"]),
            float(second["boundary_predictive_selected_gain_sum"]),
        )
        self.assertGreater(
            float(second["boundary_predictive_stiffness_sum"]), 0.0
        )

    def test_predictive_gain_fraction_selects_until_target_gain(self):
        _, rows, fieldnames = self.run_object_drone(
            {
                "OBJECT_COMMUNICATION_POLICY": "boundary_predictive",
                "OBJECT_BOUNDARY_PREDICTIVE_GAIN_FRACTION": "0.2",
            }
        )

        required_fields = {
            "boundary_predictive_gain_target_sum",
            "boundary_predictive_selected_gain_fraction",
        }
        self.assertTrue(required_fields.issubset(set(fieldnames)))
        second = rows[1]
        selected = int(second["boundary_budget_selected"])
        candidates = int(second["boundary_budget_candidates"])
        self.assertGreater(selected, 0)
        self.assertLess(selected, candidates)
        self.assertGreaterEqual(
            float(second["boundary_predictive_selected_gain_sum"]),
            float(second["boundary_predictive_gain_target_sum"]),
        )

        self.assertGreaterEqual(
            float(second["boundary_predictive_selected_gain_fraction"]), 0.2
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


class BoundaryJacobiCorrectionModeSmokeTest(unittest.TestCase):
    def test_boundary_jacobi_mode_applies_non_rtr_corrections(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_JACOBI_STEP"] = "0.25"
        env["DRAN_BOUNDARY_JACOBI_MAX_BLOCK_NORM"] = "0.05"
        env["DRAN_BOUNDARY_JACOBI_REQUIRE_DECREASE"] = "1"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "4",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_jacobi_correction",
                    "1",
                    "0.001",
                    "7",
                    "0.02",
                    "0.8",
                    "0",
                    "RTR",
                    "0.0001",
                    "0",
                    "0.5",
                    "",
                    "0",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            header = csv_path.read_text().splitlines()[0]

        self.assertIn(
            "configured_mode = decentralized_boundary_jacobi_correction",
            result.stdout,
        )
        self.assertIn("boundary_jacobi_attempted", header)
        attempts = [
            int(match)
            for match in re.findall(
                r"boundary_jacobi_attempted = ([0-9]+)", result.stdout
            )
        ]
        accepted = [
            int(match)
            for match in re.findall(
                r"boundary_jacobi_accepted = ([0-9]+)", result.stdout
            )
        ]
        blocks = [
            int(match)
            for match in re.findall(
                r"boundary_jacobi_blocks = ([0-9]+)", result.stdout
            )
        ]
        self.assertTrue(attempts, result.stdout)
        self.assertTrue(accepted, result.stdout)
        self.assertTrue(blocks, result.stdout)
        self.assertGreater(max(attempts), 0)
        self.assertGreater(max(accepted), 0)
        self.assertGreater(max(blocks), 0)
        self.assertIn(
            "boundary_jacobi_total_accepted=", result.stdout
        )

    def test_boundary_jacobi_rejected_zero_step_reports_no_blocks(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_JACOBI_STEP"] = "0"
        env["DRAN_BOUNDARY_JACOBI_MAX_BLOCK_NORM"] = "0.05"
        env["DRAN_BOUNDARY_JACOBI_REQUIRE_DECREASE"] = "1"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "2",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_jacobi_correction",
                    "1",
                    "0.001",
                    "7",
                    "0.02",
                    "0.8",
                    "0",
                    "RTR",
                    "0.0001",
                    "0",
                    "0.5",
                    "",
                    "0",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = csv_path.read_text().splitlines()

        header = rows[0].split(",")
        last = rows[-1].split(",")
        values = dict(zip(header, last))
        self.assertGreater(int(values["boundary_jacobi_attempted"]), 0)
        self.assertEqual(int(values["boundary_jacobi_accepted"]), 0)
        self.assertGreater(int(values["boundary_jacobi_rejected"]), 0)
        self.assertEqual(int(values["boundary_jacobi_blocks"]), 0)
        self.assertIn("boundary_jacobi_total_rejected=", result.stdout)

    def test_boundary_jacobi_mode_runs_with_acceleration_state(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_JACOBI_STEP"] = "0.25"
        env["DRAN_BOUNDARY_JACOBI_MAX_BLOCK_NORM"] = "0.05"
        env["DRAN_BOUNDARY_JACOBI_REQUIRE_DECREASE"] = "1"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "2",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_jacobi_correction",
                    "1",
                    "0.001",
                    "7",
                    "0.02",
                    "0.8",
                    "0",
                    "RTR",
                    "0.0001",
                    "1",
                    "0.5",
                    "",
                    "0",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

        self.assertIn("SUMMARY configured_update_mode=", result.stdout)
        self.assertIn("boundary_jacobi_total_accepted=", result.stdout)


if __name__ == "__main__":
    unittest.main()

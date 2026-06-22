#!/usr/bin/env python3
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


class MajorizedBoundaryModeSmokeTest(unittest.TestCase):
    def test_majorized_boundary_mode_runs_extra_local_refinement(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR"] = "1"
        env["DRAN_ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL"] = "0"
        env["DRAN_ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES"] = "1"
        env["DRAN_ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS"] = "1"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "7",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_majorized_boundary_async_schur_late_feedback",
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

        self.assertIn(
            "configured_mode = decentralized_majorized_boundary_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        candidate_counts = [
            int(match)
            for match in re.findall(
                r"majorized_boundary_candidates = ([0-9]+)", result.stdout
            )
        ]
        selected_counts = [
            int(match)
            for match in re.findall(
                r"majorized_boundary_selected = ([0-9]+)", result.stdout
            )
        ]
        self.assertTrue(candidate_counts, result.stdout)
        self.assertTrue(selected_counts, result.stdout)
        self.assertGreater(max(candidate_counts), 0)
        self.assertGreater(max(selected_counts), 0)


if __name__ == "__main__":
    unittest.main()

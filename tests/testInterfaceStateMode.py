#!/usr/bin/env python3
import csv
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


class InterfaceStateModeSmokeTest(unittest.TestCase):
    def test_interface_state_mode_logs_compact_state_without_changing_effective_mode(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

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
                    "decentralized_interface_state_async_schur_late_feedback",
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
                env=os.environ.copy(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = list(csv.DictReader(csv_path.open()))

        self.assertIn(
            "configured_mode = decentralized_interface_state_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        for field in (
            "interface_state_payload_blocks",
            "interface_state_grad_norm",
            "interface_state_stiffness_sum",
            "interface_state_freshness_sum",
            "interface_state_comm_mb",
            "cumulative_interface_state_comm_mb",
        ):
            self.assertIn(field, rows[0])

        payload_blocks = [
            int(row["interface_state_payload_blocks"]) for row in rows
        ]
        comm_mb = [float(row["interface_state_comm_mb"]) for row in rows]
        grad_norms = [float(row["interface_state_grad_norm"]) for row in rows]
        stiffness = [float(row["interface_state_stiffness_sum"]) for row in rows]
        self.assertGreater(max(payload_blocks), 0)
        self.assertGreater(max(comm_mb), 0.0)
        self.assertGreater(max(grad_norms), 0.0)
        self.assertGreater(max(stiffness), 0.0)
        self.assertIn("interface_state_total_payload_blocks=", result.stdout)
        self.assertIn("interface_state_total_comm_mb=", result.stdout)

        summary_match = re.search(
            r"total_comm_mb=([0-9.eE+-]+).*interface_state_total_comm_mb=([0-9.eE+-]+)",
            result.stdout,
        )
        self.assertIsNotNone(summary_match, result.stdout)
        self.assertGreater(float(summary_match.group(1)), 0.0)
        self.assertGreater(float(summary_match.group(2)), 0.0)


if __name__ == "__main__":
    unittest.main()

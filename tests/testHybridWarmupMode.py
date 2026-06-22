#!/usr/bin/env python3
import subprocess
import tempfile
import unittest
from pathlib import Path


class HybridWarmupModeSmokeTest(unittest.TestCase):
    def test_first_round_uses_decentralized_effective_mode(self):
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
                    "1",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_hybrid_warmup_async_schur_late_feedback",
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
                    "1",
                ],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

        self.assertIn(
            "configured_mode = decentralized_hybrid_warmup_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn("effective_mode = decentralized", result.stdout)


if __name__ == "__main__":
    unittest.main()

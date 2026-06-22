#!/usr/bin/env python3
import csv
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


class InterfaceDeltaModeSmokeTest(unittest.TestCase):
    def test_interface_delta_mode_uses_sparse_delta_payloads(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_INTERFACE_DELTA_CODEC"] = "sparse"
        env["DRAN_INTERFACE_DELTA_TOPK"] = "4"
        env["DRAN_INTERFACE_DELTA_MAX_RECON_ERROR"] = "100"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "6",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_interface_delta_async_schur_late_feedback",
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

            rows = list(csv.DictReader(csv_path.open()))

        self.assertIn(
            "configured_mode = decentralized_interface_delta_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        for field in (
            "interface_delta_candidate_pose_blocks",
            "interface_delta_cache_miss_pose_blocks",
            "interface_delta_rejected_pose_blocks",
            "interface_delta_sparse_pose_blocks",
            "interface_delta_tangent_pose_blocks",
            "interface_delta_full_pose_blocks",
            "interface_delta_delta_pose_blocks",
            "interface_delta_delta_entries",
            "interface_delta_comm_mb",
            "cumulative_interface_delta_comm_mb",
        ):
            self.assertIn(field, rows[0])

        delta_blocks = [
            int(row["interface_delta_delta_pose_blocks"]) for row in rows
        ]
        delta_entries = [
            int(row["interface_delta_delta_entries"]) for row in rows
        ]
        candidates = [
            int(row["interface_delta_candidate_pose_blocks"]) for row in rows
        ]
        rejected = [
            int(row["interface_delta_rejected_pose_blocks"]) for row in rows
        ]
        self.assertGreater(max(delta_blocks), 0)
        self.assertGreater(max(delta_entries), 0)
        self.assertGreater(max(candidates), 0)
        self.assertGreaterEqual(max(rejected), 0)

        summary_match = re.search(
            r"total_comm_mb=([0-9.eE+-]+).*"
            r"interface_delta_total_candidate_pose_blocks=([0-9]+).*"
            r"interface_delta_total_delta_pose_blocks=([0-9]+).*"
            r"interface_delta_total_comm_mb=([0-9.eE+-]+)",
            result.stdout,
        )
        self.assertIsNotNone(summary_match, result.stdout)
        self.assertGreater(int(summary_match.group(2)), 0)
        self.assertGreater(int(summary_match.group(3)), 0)
        self.assertAlmostEqual(
            float(summary_match.group(1)),
            float(summary_match.group(4)),
            places=9,
        )

    def test_interface_delta_tangent_codec_uses_manifold_payloads(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_INTERFACE_DELTA_CODEC"] = "tangent"
        env["DRAN_INTERFACE_DELTA_MAX_RECON_ERROR"] = "100"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "6",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_interface_delta_async_schur_late_feedback",
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

            rows = list(csv.DictReader(csv_path.open()))

        self.assertIn("interface_delta_codec = tangent", result.stdout)
        self.assertTrue(rows)
        tangent_blocks = [
            int(row["interface_delta_tangent_pose_blocks"]) for row in rows
        ]
        sparse_blocks = [
            int(row["interface_delta_sparse_pose_blocks"]) for row in rows
        ]
        self.assertGreater(max(tangent_blocks), 0)
        self.assertEqual(max(sparse_blocks), 0)
        self.assertRegex(
            result.stdout,
            r"interface_delta_total_tangent_pose_blocks=[1-9][0-9]*",
        )

    def test_interface_delta_hybrid_codec_selects_valid_payloads(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_INTERFACE_DELTA_CODEC"] = "hybrid"
        env["DRAN_INTERFACE_DELTA_TOPK"] = "4"
        env["DRAN_INTERFACE_DELTA_MAX_RECON_ERROR"] = "100"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "6",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_interface_delta_async_schur_late_feedback",
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

            rows = list(csv.DictReader(csv_path.open()))

        self.assertIn("interface_delta_codec = hybrid", result.stdout)
        total_delta_blocks = sum(
            int(row["interface_delta_delta_pose_blocks"]) for row in rows
        )
        total_kind_blocks = sum(
            int(row["interface_delta_sparse_pose_blocks"]) +
            int(row["interface_delta_tangent_pose_blocks"])
            for row in rows
        )
        self.assertGreater(total_delta_blocks, 0)
        self.assertEqual(total_delta_blocks, total_kind_blocks)
        self.assertRegex(
            result.stdout,
            r"interface_delta_total_sparse_pose_blocks=[0-9]+.*"
            r"interface_delta_total_tangent_pose_blocks=[0-9]+",
        )

    def test_tangent_codec_does_not_require_sparse_topk(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        for codec in ("tangent", "hybrid"):
            env = os.environ.copy()
            env["DRAN_INTERFACE_DELTA_CODEC"] = codec
            env["DRAN_INTERFACE_DELTA_TOPK"] = "0"
            env["DRAN_INTERFACE_DELTA_MAX_RECON_ERROR"] = "100"

            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = Path(tmpdir) / "iters.csv"
                result = subprocess.run(
                    [
                        str(binary),
                        "2",
                        str(root / "data" / "tinyGrid3D.g2o"),
                        "6",
                        "-1",
                        "-1",
                        str(csv_path),
                        "1",
                        "10",
                        "0.0001",
                        "100",
                        "2",
                        "decentralized_interface_delta_async_schur_late_feedback",
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

                rows = list(csv.DictReader(csv_path.open()))

            tangent_blocks = sum(
                int(row["interface_delta_tangent_pose_blocks"]) for row in rows
            )
            sparse_blocks = sum(
                int(row["interface_delta_sparse_pose_blocks"]) for row in rows
            )
            self.assertGreater(tangent_blocks, 0, codec)
            self.assertEqual(sparse_blocks, 0, codec)
            self.assertRegex(
                result.stdout,
                r"interface_delta_total_tangent_pose_blocks=[1-9][0-9]*",
            )


if __name__ == "__main__":
    unittest.main()

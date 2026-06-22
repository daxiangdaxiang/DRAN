#!/usr/bin/env python3
import csv
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


class BoundaryModelPredictionModeSmokeTest(unittest.TestCase):
    def test_local_amm_mode_tries_guarded_local_state_candidates(self):
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
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_local_amm_async_schur_late_feedback",
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
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = list(csv.DictReader(csv_path.open()))

        self.assertIn(
            "configured_mode = decentralized_local_amm_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        for field in [
            "local_amm_attempted",
            "local_amm_accepted",
            "local_amm_rejected",
            "local_amm_cost_decrease",
            "local_amm_grad_decrease",
        ]:
            self.assertIn(field, rows[0])
        attempts = [int(row["local_amm_attempted"]) for row in rows]
        accepted = [int(row["local_amm_accepted"]) for row in rows]
        rejected = [int(row["local_amm_rejected"]) for row in rows]
        self.assertGreater(sum(attempts), 0)
        self.assertEqual(sum(attempts), sum(accepted) + sum(rejected))
        self.assertRegex(result.stdout, r"local_amm_total_attempted=[1-9][0-9]*")

    def test_local_amm_manual_solver_reports_baseline_comparator(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_LOCAL_SOLVER"] = "manual_newton"
        env["DRAN_LOCAL_AMM_COMPARE_BASELINE"] = "true"

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
                    "2",
                    "12",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_local_amm_async_schur_late_feedback",
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

        self.assertIn("local_solver = manual_newton", result.stdout)
        self.assertTrue(rows)
        for field in [
            "local_amm_baseline_compared",
            "local_amm_selected",
            "local_amm_baseline_selected",
            "local_amm_vs_baseline_cost_delta",
            "local_amm_vs_baseline_grad_delta",
        ]:
            self.assertIn(field, rows[0])
        compared = sum(int(row["local_amm_baseline_compared"]) for row in rows)
        selected = sum(int(row["local_amm_selected"]) for row in rows)
        baseline_selected = sum(
            int(row["local_amm_baseline_selected"]) for row in rows
        )
        self.assertGreater(compared, 0)
        self.assertEqual(compared, selected + baseline_selected)
        self.assertRegex(
            result.stdout, r"local_amm_total_baseline_compared=[1-9][0-9]*"
        )

    def test_amm_boundary_mode_uses_guarded_model_candidates(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_AMM_BOUNDARY_BETA"] = "0.6"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_amm_boundary_async_schur_late_feedback",
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
            "configured_mode = decentralized_amm_boundary_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        predictions = [int(row["boundary_model_predictions"]) for row in rows]
        evaluations = [
            int(row["boundary_model_merit_evaluations"]) for row in rows
        ]
        self.assertGreater(max(predictions), 0)
        self.assertGreater(max(evaluations), 0)

    def test_boundary_schur_response_mode_applies_coupled_corrections(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_STEP"] = "0.2"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_GAIN"] = "1.0"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM"] = "0.03"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_DAMPING"] = "0.1"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS"] = "8"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE"] = "0"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_schur_response_async_schur_late_feedback",
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
            "configured_mode = decentralized_boundary_schur_response_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        attempts = [int(row["boundary_batch_response_attempted"]) for row in rows]
        accepted = [int(row["boundary_batch_response_accepted"]) for row in rows]
        blocks = [int(row["boundary_batch_response_blocks"]) for row in rows]
        step_norm = [float(row["boundary_batch_response_step_norm"]) for row in rows]
        grad_delta_norm = [
            float(row["boundary_batch_response_grad_delta_norm"]) for row in rows
        ]
        self.assertGreater(max(attempts), 0)
        self.assertGreater(max(accepted), 0)
        self.assertGreater(max(blocks), 0)
        self.assertGreater(max(step_norm), 0.0)
        self.assertGreater(max(grad_delta_norm), 0.0)
        self.assertRegex(
            result.stdout,
            r"boundary_batch_response_total_accepted=[1-9][0-9]*",
        )

    def test_boundary_schur_response_can_skip_baseline_refine(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_STEP"] = "0.2"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM"] = "0.03"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS"] = "8"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE"] = "0"
        env["DRAN_BOUNDARY_SCHUR_RESPONSE_SKIP_BASELINE_REFINE"] = "1"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_schur_response_async_schur_late_feedback",
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

        self.assertTrue(rows)
        active_rows = [
            row
            for row in rows
            if int(row["boundary_batch_response_attempted"]) > 0
        ]
        self.assertTrue(active_rows)
        for row in active_rows:
            response_ms = float(row["boundary_batch_response_ms"])
            stage_ms = float(row["boundary_batch_model_refine_ms"])
            self.assertGreater(response_ms, 0.0)
            self.assertLessEqual(stage_ms, response_ms * 1.25 + 0.25)

    def test_boundary_batch_response_mode_applies_aggregated_corrections(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_BATCH_RESPONSE_STEP"] = "0.15"
        env["DRAN_BOUNDARY_BATCH_RESPONSE_GAIN"] = "1.0"
        env["DRAN_BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM"] = "0.03"
        env["DRAN_BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE"] = "0"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_batch_response_async_schur_late_feedback",
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

            with csv_path.open() as f:
                raw_rows = list(csv.reader(f))
            rows = list(csv.DictReader(csv_path.open()))

        self.assertIn(
            "configured_mode = decentralized_boundary_batch_response_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        self.assertTrue(raw_rows)
        header_width = len(raw_rows[0])
        self.assertTrue(
            all(len(row) == header_width for row in raw_rows[1:]),
            "iteration CSV header/data column counts differ",
        )
        for field in (
            "boundary_batch_response_attempted",
            "boundary_batch_response_accepted",
            "boundary_batch_response_rejected",
            "boundary_batch_response_blocks",
            "boundary_batch_response_step_norm",
            "boundary_batch_response_grad_delta_norm",
            "boundary_batch_response_cost_decrease",
            "boundary_batch_response_ms",
            "cumulative_boundary_batch_response_accepted",
            "cumulative_boundary_batch_response_ms",
        ):
            self.assertIn(field, rows[0])

        attempts = [int(row["boundary_batch_response_attempted"]) for row in rows]
        accepted = [int(row["boundary_batch_response_accepted"]) for row in rows]
        blocks = [int(row["boundary_batch_response_blocks"]) for row in rows]
        step_norm = [float(row["boundary_batch_response_step_norm"]) for row in rows]
        grad_delta_norm = [
            float(row["boundary_batch_response_grad_delta_norm"]) for row in rows
        ]
        self.assertGreater(max(attempts), 0)
        self.assertGreater(max(accepted), 0)
        self.assertGreater(max(blocks), 0)
        self.assertGreater(max(step_norm), 0.0)
        self.assertGreater(max(grad_delta_norm), 0.0)
        self.assertRegex(
            result.stdout,
            r"boundary_batch_response_total_accepted=[1-9][0-9]*",
        )

    def test_budgeted_boundary_batch_model_mode_drops_low_priority_refreshes(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER"] = "1"
        env["DRAN_BOUNDARY_BATCH_BUDGET_FRACTION"] = "0.5"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_budgeted_boundary_batch_model_async_schur_late_feedback",
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
            "configured_mode = decentralized_budgeted_boundary_batch_model_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        for field in (
            "boundary_batch_model_budget_candidates",
            "boundary_batch_model_budget_selected",
            "boundary_batch_model_budget_dropped",
            "cumulative_boundary_batch_model_budget_candidates",
            "cumulative_boundary_batch_model_budget_selected",
            "cumulative_boundary_batch_model_budget_dropped",
        ):
            self.assertIn(field, rows[0])

        candidates = [
            int(row["boundary_batch_model_budget_candidates"]) for row in rows
        ]
        selected = [
            int(row["boundary_batch_model_budget_selected"]) for row in rows
        ]
        dropped = [
            int(row["boundary_batch_model_budget_dropped"]) for row in rows
        ]
        self.assertGreater(max(candidates), 0)
        self.assertGreater(max(selected), 0)
        self.assertGreater(max(dropped), 0)
        self.assertTrue(
            any(c > s for c, s in zip(candidates, selected)),
            result.stdout,
        )
        self.assertRegex(
            result.stdout,
            r"boundary_batch_model_total_budget_dropped=[1-9][0-9]*",
        )

    def test_boundary_batch_model_mode_refines_with_aggregated_neighbor_models(self):
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
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_batch_model_async_schur_late_feedback",
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
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = list(csv.DictReader(csv_path.open()))

        self.assertIn(
            "configured_mode = decentralized_boundary_batch_model_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        for field in (
            "boundary_batch_model_receivers",
            "boundary_batch_model_candidate_poses",
            "boundary_batch_model_merit_evaluations",
            "boundary_batch_model_merit_accepted",
            "boundary_batch_model_merit_rejected",
            "boundary_batch_model_refine_ms",
            "cumulative_boundary_batch_model_receivers",
            "cumulative_boundary_batch_model_candidate_poses",
            "cumulative_boundary_batch_model_merit_evaluations",
        ):
            self.assertIn(field, rows[0])

        receivers = [int(row["boundary_batch_model_receivers"]) for row in rows]
        candidates = [
            int(row["boundary_batch_model_candidate_poses"]) for row in rows
        ]
        merit_evals = [
            int(row["boundary_batch_model_merit_evaluations"]) for row in rows
        ]
        accepted = [
            int(row["boundary_batch_model_merit_accepted"]) for row in rows
        ]
        rejected = [
            int(row["boundary_batch_model_merit_rejected"]) for row in rows
        ]
        self.assertGreater(max(receivers), 0)
        self.assertGreater(max(candidates), 0)
        self.assertGreater(max(merit_evals), 0)
        self.assertGreater(max(a + r for a, r in zip(accepted, rejected)), 0)
        self.assertRegex(
            result.stdout,
            r"boundary_batch_model_total_receivers=[1-9][0-9]*",
        )
        self.assertRegex(
            result.stdout,
            r"boundary_batch_model_total_candidate_poses=[1-9][0-9]*",
        )

    def test_boundary_response_mode_applies_receiver_side_corrections(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_RESPONSE_STEP"] = "0.15"
        env["DRAN_BOUNDARY_RESPONSE_GAIN"] = "1.0"
        env["DRAN_BOUNDARY_RESPONSE_MAX_BLOCK_NORM"] = "0.03"
        env["DRAN_BOUNDARY_RESPONSE_REQUIRE_DECREASE"] = "0"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_response_async_schur_late_feedback",
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
            "configured_mode = decentralized_boundary_response_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        for field in (
            "boundary_model_packet_payload_blocks",
            "boundary_response_attempted",
            "boundary_response_accepted",
            "boundary_response_rejected",
            "boundary_response_blocks",
            "boundary_response_step_norm",
            "boundary_response_grad_delta_norm",
            "boundary_response_cost_decrease",
            "boundary_response_ms",
            "cumulative_boundary_response_accepted",
            "cumulative_boundary_response_ms",
        ):
            self.assertIn(field, rows[0])

        attempts = [int(row["boundary_response_attempted"]) for row in rows]
        accepted = [int(row["boundary_response_accepted"]) for row in rows]
        blocks = [int(row["boundary_response_blocks"]) for row in rows]
        step_norm = [float(row["boundary_response_step_norm"]) for row in rows]
        grad_delta_norm = [
            float(row["boundary_response_grad_delta_norm"]) for row in rows
        ]
        packet_blocks = [
            int(row["boundary_model_packet_payload_blocks"]) for row in rows
        ]
        self.assertGreater(max(packet_blocks), 0)
        self.assertGreater(max(attempts), 0)
        self.assertGreater(max(accepted), 0)
        self.assertGreater(max(blocks), 0)
        self.assertGreater(max(step_norm), 0.0)
        self.assertGreater(max(grad_delta_norm), 0.0)
        self.assertRegex(
            result.stdout,
            r"boundary_response_total_attempted=[1-9][0-9]*",
        )
        self.assertRegex(
            result.stdout,
            r"boundary_response_total_accepted=[1-9][0-9]*",
        )

    def test_boundary_model_packet_mode_uses_explicit_model_refresh_packets(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_MODEL_STEP"] = "0.25"
        env["DRAN_BOUNDARY_MODEL_MAX_BLOCK_NORM"] = "0.05"
        env["DRAN_BOUNDARY_MODEL_GAIN"] = "1.0"

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "iters.csv"
            result = subprocess.run(
                [
                    str(binary),
                    "2",
                    str(root / "data" / "tinyGrid3D.g2o"),
                    "8",
                    "-1",
                    "-1",
                    str(csv_path),
                    "1",
                    "10",
                    "0.0001",
                    "100",
                    "2",
                    "decentralized_boundary_model_packet_async_schur_late_feedback",
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
            "configured_mode = decentralized_boundary_model_packet_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertTrue(rows)
        for field in (
            "boundary_model_packet_payload_blocks",
            "boundary_model_packet_model_pose_blocks",
            "boundary_model_packet_grad_norm",
            "boundary_model_packet_stiffness_sum",
            "boundary_model_packet_freshness_sum",
            "boundary_model_packet_comm_mb",
            "cumulative_boundary_model_packet_payload_blocks",
            "cumulative_boundary_model_packet_comm_mb",
        ):
            self.assertIn(field, rows[0])

        payload_blocks = [
            int(row["boundary_model_packet_payload_blocks"]) for row in rows
        ]
        model_pose_blocks = [
            int(row["boundary_model_packet_model_pose_blocks"]) for row in rows
        ]
        packet_comm_mb = [
            float(row["boundary_model_packet_comm_mb"]) for row in rows
        ]
        grad_norm = [
            float(row["boundary_model_packet_grad_norm"]) for row in rows
        ]
        self.assertGreater(max(payload_blocks), 0)
        self.assertGreater(max(model_pose_blocks), 0)
        self.assertGreater(max(packet_comm_mb), 0.0)
        self.assertGreater(max(grad_norm), 0.0)
        self.assertRegex(
            result.stdout,
            r"boundary_model_packet_total_payload_blocks=[1-9][0-9]*",
        )
        self.assertRegex(
            result.stdout,
            r"boundary_model_packet_total_model_pose_blocks=[1-9][0-9]*",
        )
        self.assertRegex(
            result.stdout,
            r"boundary_model_packet_total_comm_mb=[0-9.eE+-]+",
        )

    def test_boundary_model_mode_logs_predictions_without_extra_rtr_mode(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_MODEL_STEP"] = "0.25"
        env["DRAN_BOUNDARY_MODEL_MAX_BLOCK_NORM"] = "0.05"
        env["DRAN_BOUNDARY_MODEL_REQUIRE_DECREASE"] = "1"

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
                    "decentralized_boundary_model_async_schur_late_feedback",
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
            "configured_mode = decentralized_boundary_model_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn(
            "effective_mode = decentralized_residual_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn("boundary_model_predictions", header)
        self.assertIn("boundary_model_computed_step_norm", header)
        self.assertIn("boundary_model_compute_ms", header)
        predictions = [
            int(match)
            for match in re.findall(
                r"boundary_model_predictions = ([0-9]+)", result.stdout
            )
        ]
        self.assertTrue(predictions, result.stdout)
        self.assertGreater(max(predictions), 0)
        self.assertIn("boundary_model_total_predictions=", result.stdout)
        self.assertIn(
            "boundary_model_total_computed_step_norm=", result.stdout
        )
        self.assertIn("boundary_model_total_compute_ms=", result.stdout)
        self.assertIn("boundary_model_merit_evaluations", header)
        self.assertIn("boundary_model_merit_accepted", header)
        self.assertIn("boundary_model_merit_rejected", header)
        self.assertIn("boundary_model_merit_grad_decrease", header)
        self.assertIn("boundary_model_total_merit_evaluations=", result.stdout)
        self.assertIn(
            "boundary_model_total_merit_grad_decrease=", result.stdout
        )
        merit_evals = [
            int(match)
            for match in re.findall(
                r"boundary_model_merit_evaluations = ([0-9]+)",
                result.stdout,
            )
        ]
        self.assertTrue(merit_evals, result.stdout)
        self.assertGreater(max(merit_evals), 0)

    def test_boundary_model_mode_runs_with_acceleration_state(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "multi-robot-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        env["DRAN_BOUNDARY_MODEL_STEP"] = "0.25"
        env["DRAN_BOUNDARY_MODEL_MAX_BLOCK_NORM"] = "0.05"
        env["DRAN_BOUNDARY_MODEL_REQUIRE_DECREASE"] = "1"

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
                    "decentralized_boundary_model_async_schur_late_feedback",
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
        self.assertIn("boundary_model_total_predictions=", result.stdout)


if __name__ == "__main__":
    unittest.main()

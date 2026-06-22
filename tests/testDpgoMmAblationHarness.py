#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DpgoMmAblationHarnessTest(unittest.TestCase):
    def test_manual_replay_help_exposes_fresh_neighbor_exchange(self):
        binary = ROOT / "build" / "bin" / "manual-dpgo-mm-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        result = subprocess.run(
            [str(binary), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--local_gradient_correction_fresh_neighbor_exchange",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_fresh_min_pose_delta",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_fresh_delta_mode",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_fresh_budget_fraction",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_fresh_max_poses_per_receiver",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_neighborhood_step",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_curvature_step",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_boundary_only",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_coupled_direction",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_coupled_direction_topk_entries",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_coupled_direction_min_score",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_coupled_direction_min_score_ratio",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_coupled_direction_byte_budget_mb",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_block_jacobi",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_compact_schur",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_compact_schur_max_private_cols",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_compact_schur_max_boundary_poses",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_compact_schur_damping",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_compact_schur_gradient_guard",
            result.stdout,
        )
        self.assertIn(
            "--local_gradient_correction_compact_schur_max_gradient_increase_ratio",
            result.stdout,
        )
        self.assertIn("--post_exchange_period", result.stdout)
        self.assertIn("--post_exchange_min_pose_delta", result.stdout)
        self.assertIn("--post_exchange_delta_mode", result.stdout)
        self.assertIn("--post_exchange_budget_fraction", result.stdout)
        self.assertIn("--post_exchange_max_poses_per_receiver", result.stdout)
        self.assertIn("--local_max_iterations_accepted", result.stdout)
        self.assertIn("--amm_accepted_delta", result.stdout)
        self.assertIn("--amm_oscillation_count_period", result.stdout)
        self.assertIn("--amm_max_oscillations", result.stdout)
        self.assertIn("--amm_baseline_surrogate_state", result.stdout)
        self.assertIn("--amm_dpgo_recursive_simple_state", result.stdout)
        self.assertIn("--amm_recursive_simple_reanchor_period", result.stdout)
        self.assertIn("--amm_prox_reset_skip_refined_solve", result.stdout)
        self.assertIn("--amm_dpgo_proximal_fallback_only", result.stdout)
        self.assertIn("--amm_lazy_plain_after_certificate", result.stdout)
        self.assertIn("--amm_surrogate_first_exact_evaluation", result.stdout)
        self.assertIn("--amm_dpgo_strict_refined_gate", result.stdout)
        self.assertIn(
            "--amm_dpgo_recover_translations_after_proximal",
            result.stdout,
        )
        self.assertIn(
            "--amm_dpgo_refined_starts_at_recovered_proximal",
            result.stdout,
        )
        self.assertIn("--feh_translation_recovery_initial_guess",
                      result.stdout)
        self.assertIn("--fused_candidate_evaluation", result.stdout)
        self.assertIn("--lazy_candidate_gradient_evaluation", result.stdout)
        self.assertIn("--lazy_solver_start_gradient_evaluation",
                      result.stdout)
        self.assertIn("--lazy_surrogate_candidate_gradient_evaluation",
                      result.stdout)
        self.assertIn("--reduced_rotation_curvature_cauchy_candidate",
                      result.stdout)
        self.assertIn("--reduced_rotation_curvature_fallback_candidate",
                      result.stdout)
        self.assertIn("--reduced_rotation_gradient_boundary_candidate",
                      result.stdout)
        self.assertIn("--print_iteration_summary", result.stdout)
        self.assertIn("--reduced_rotation_surrogate_tcg_accept",
                      result.stdout)
        self.assertIn(
            "--reduced_rotation_skip_redundant_candidate_projection",
            result.stdout,
        )
        self.assertIn("--comm_topology_interface_model", result.stdout)
        self.assertIn("--comm_topology_interface_model_payload",
                      result.stdout)
        self.assertIn("--comm_topology_interface_model_local_merit",
                      result.stdout)
        self.assertIn("--comm_topology_boundary_candidate", result.stdout)
        self.assertIn("--comm_topology_boundary_surrogate", result.stdout)
        self.assertIn(
            "--comm_topology_boundary_surrogate_stale_gain",
            result.stdout,
        )
        self.assertIn("--comm_topology_reduced_interface_model",
                      result.stdout)
        self.assertIn("--comm_topology_reduced_interface_weight",
                      result.stdout)
        self.assertIn(
            "--comm_topology_reduced_interface_max_local_iters",
            result.stdout,
        )
        self.assertIn("--comm_topology_reduced_interface_candidate",
                      result.stdout)

    def test_manual_replay_accepts_case_insensitive_post_delta_mode(self):
        binary = ROOT / "build" / "bin" / "manual-dpgo-mm-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        result = subprocess.run(
            [
                str(binary),
                "--dataset", "data/tinyGrid3D.g2o",
                "--num_nodes", "2",
                "--iters", "0",
                "--post_exchange_delta_mode", "Weighted",
                "--save", "false",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("post_exchange_delta_mode=weighted", result.stdout)

    def test_manual_replay_parses_compact_schur_options(self):
        binary = ROOT / "build" / "bin" / "manual-dpgo-mm-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        result = subprocess.run(
            [
                str(binary),
                "--dataset", "data/tinyGrid3D.g2o",
                "--num_nodes", "2",
                "--iters", "0",
                "--local_gradient_correction_compact_schur", "true",
                "--local_gradient_correction_compact_schur_max_boundary_poses",
                "7",
                "--local_gradient_correction_compact_schur_max_private_cols",
                "11",
                "--local_gradient_correction_compact_schur_damping",
                "0.03",
                "--local_gradient_correction_compact_schur_gradient_guard",
                "true",
                "--local_gradient_correction_compact_schur_max_gradient_increase_ratio",
                "0.15",
                "--save", "false",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("local_gradient_correction_compact_schur=true",
                      result.stdout)
        self.assertIn(
            "local_gradient_correction_compact_schur_max_boundary_poses=7",
            result.stdout,
        )
        self.assertIn(
            "local_gradient_correction_compact_schur_max_private_cols=11",
            result.stdout,
        )
        self.assertIn("local_gradient_correction_compact_schur_damping=0.03",
                      result.stdout)
        self.assertIn(
            "local_gradient_correction_compact_schur_gradient_guard=true",
            result.stdout,
        )
        self.assertIn(
            "local_gradient_correction_compact_schur_max_gradient_increase_ratio=0.15",
            result.stdout,
        )

    def test_manual_replay_rejects_negative_compact_schur_damping(self):
        binary = ROOT / "build" / "bin" / "manual-dpgo-mm-example"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        result = subprocess.run(
            [
                str(binary),
                "--dataset", "data/tinyGrid3D.g2o",
                "--num_nodes", "2",
                "--iters", "0",
                "--local_gradient_correction_compact_schur_damping",
                "-0.1",
                "--save", "false",
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "localGradientCorrectionCompactSchurDamping must be finite and nonnegative",
            result.stdout,
        )

    def test_dist_pgo_exposes_internal_ablation_options(self):
        binary = ROOT / "baselines" / "DPGO_MM" / "C++" / "build-local" / "bin" / "dist_pgo"
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")

        env = os.environ.copy()
        lib_dir = binary.parents[1] / "lib"
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
        result = subprocess.run(
            [str(binary), "--help"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        for option in (
            "--scheme",
            "--preconditioner",
            "--local_max_iterations",
            "--local_max_iterations_accepted",
            "--init_red_rot_iters",
            "--init_rot_iters",
            "--init_red_trans_iters",
            "--init_trans_iters",
            "--trace_amm",
        ):
            self.assertIn(option, result.stdout)

    def test_ablation_matrix_dry_run_lists_core_variants(self):
        script = ROOT / "scripts" / "run_dpgo_mm_ablation_matrix.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "MAX_ITERS": "3",
                "ABLATIONS": "baseline,no_amm,no_preconditioner,weak_init,centralized_init_no_amm",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("baseline,sphere,", result.stdout)
        self.assertIn("--scheme amm", result.stdout)
        self.assertIn("no_amm,sphere,", result.stdout)
        self.assertIn("--scheme mm", result.stdout)
        self.assertIn("no_preconditioner,sphere,", result.stdout)
        self.assertIn("--preconditioner none", result.stdout)
        self.assertIn("weak_init,sphere,", result.stdout)
        self.assertIn("--init_red_rot_iters 10", result.stdout)
        self.assertIn("--init_trans_iters 25", result.stdout)
        self.assertIn("centralized_init_no_amm,sphere,", result.stdout)
        self.assertIn("--dist_init false", result.stdout)

    def test_dpgo_mm_runner_dry_run_prints_command_without_execution(self):
        script = ROOT / "scripts" / "run_dpgo_mm_six.sh"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            marker = tmp_path / "executed"
            mock_bin = tmp_path / "dist_pgo"
            mock_bin.write_text(
                "#!/usr/bin/env bash\n"
                f"touch {marker}\n"
                "exit 77\n")
            os.chmod(mock_bin, 0o755)

            result = subprocess.run(
                [
                    str(script),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "BIN": str(mock_bin),
                    "DRY_RUN": "true",
                    "DATASETS": "sphere",
                    "MAX_ITERS": "5",
                    "DIST_INIT": "false",
                    "SCHEME": "amm",
                    "ACCELERATED": "true",
                    "PRECONDITIONER": "none",
                    "OUT_ROOT": str(tmp_path / "out"),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            self.assertFalse(marker.exists())
            self.assertIn("DRY_RUN dataset=sphere", result.stdout)
            self.assertIn("data/sphere2500.g2o", result.stdout)
            self.assertIn("--iters 4", result.stdout)
            self.assertIn("--dist_init false", result.stdout)
            self.assertIn("--scheme amm", result.stdout)
            self.assertIn("--accelerated true", result.stdout)
            self.assertIn("--preconditioner none", result.stdout)

    def test_dran_six_dry_run_prints_active_parameter_snapshot(self):
        script = ROOT / "scripts" / "run_six_dataset_experiments.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "MAX_ITERS": "7",
                "RTR_ITERS": "2",
                "UPDATE_MODE":
                    "decentralized_boundary_schur_response_async_schur_late_feedback",
                "BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS": "13",
                "BOUNDARY_SCHUR_RESPONSE_DAMPING": "0.04",
                "INTERFACE_DELTA_CODEC": "tangent",
                "INTERFACE_DELTA_TOPK": "3",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("DRY_RUN dataset=sphere", result.stdout)
        self.assertIn("path=data/sphere2500.g2o", result.stdout)
        self.assertIn("max_iters=7", result.stdout)
        self.assertIn("rtr_iters=2", result.stdout)
        self.assertIn(
            "update_mode=decentralized_boundary_schur_response_async_schur_late_feedback",
            result.stdout,
        )
        self.assertIn("boundary_schur_response_max_blocks=13", result.stdout)
        self.assertIn("boundary_schur_response_damping=0.04", result.stdout)
        self.assertIn("interface_delta_codec=tangent", result.stdout)
        self.assertIn("interface_delta_topk=3", result.stdout)

    def test_manual_replay_comparison_dry_run_aligns_update_count(self):
        script = ROOT / "scripts" / "compare_manual_dpgo_mm_replay.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere,torus",
                "MAX_UPDATES": "4",
                "BIN": "/tmp/manual-release",
                "SCHEME": "amm",
                "ACCELERATED": "true",
                "LOCAL_MAX_ITERATIONS_ACCEPTED": "2",
                "AMM_DPGO_SURROGATE_PARITY": "true",
                "AMM_DPGO_RECURSIVE_SIMPLE_STATE": "true",
                "AMM_RECURSIVE_SIMPLE_REANCHOR_PERIOD": "10",
                "AMM_DPGO_STRICT_REFINED_GATE": "true",
                "AMM_DPGO_RECOVER_TRANSLATIONS_AFTER_PROXIMAL": "true",
                "AMM_DPGO_REFINED_STARTS_AT_RECOVERED_PROXIMAL": "true",
                "DPGO_MM_AMM_TRACE": "true",
                "TRACE_AMM": "true",
                "LOCAL_GRADIENT_CORRECTION": "true",
                "LOCAL_GRADIENT_CORRECTION_STEPS": "0.0001,0.001",
                "LOCAL_GRADIENT_CORRECTION_SHARED_STEP": "true",
                "LOCAL_GRADIENT_CORRECTION_NEIGHBORHOOD_STEP": "true",
                "LOCAL_GRADIENT_CORRECTION_CURVATURE_STEP": "true",
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION": "true",
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_BUDGET_FRACTION":
                    "0.5",
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_MAX_PACKETS_PER_RECEIVER":
                    "8",
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_TOPK_ENTRIES":
                    "2",
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_MIN_SCORE":
                    "0.03",
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_MIN_SCORE_RATIO":
                    "0.2",
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_BYTE_BUDGET_MB":
                    "0.001",
                "LOCAL_GRADIENT_CORRECTION_BOUNDARY_ONLY": "true",
                "LOCAL_GRADIENT_CORRECTION_BLOCK_JACOBI": "true",
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR": "true",
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_MAX_BOUNDARY_POSES":
                    "5",
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_MAX_PRIVATE_COLS":
                    "12",
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_DAMPING": "0.02",
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_GRADIENT_GUARD":
                    "true",
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_MAX_GRADIENT_INCREASE_RATIO":
                    "0.2",
                "LOCAL_GRADIENT_CORRECTION_FRESH_NEIGHBOR_EXCHANGE": "true",
                "LOCAL_GRADIENT_CORRECTION_FRESH_MIN_POSE_DELTA": "0.02",
                "LOCAL_GRADIENT_CORRECTION_FRESH_DELTA_MODE": "weighted",
                "LOCAL_GRADIENT_CORRECTION_FRESH_BUDGET_FRACTION": "0.25",
                "LOCAL_GRADIENT_CORRECTION_FRESH_MAX_POSES_PER_RECEIVER": "6",
                "LOCAL_GRADIENT_CORRECTION_INNER_ROUNDS": "3",
                "LOCAL_GRADIENT_CORRECTION_MAX_ROUNDS": "3",
                "POST_EXCHANGE_PERIOD": "2",
                "POST_EXCHANGE_MIN_POSE_DELTA": "0.01",
                "POST_EXCHANGE_DELTA_MODE": "weighted",
                "POST_EXCHANGE_BUDGET_FRACTION": "0.5",
                "POST_EXCHANGE_MAX_POSES_PER_RECEIVER": "7",
                "SYNC_AMM_REFERENCE_AFTER_LOCAL_GRADIENT_CORRECTION": "false",
                "GLOBAL_GRADIENT_CORRECTION": "true",
                "GLOBAL_GRADIENT_CORRECTION_STEPS": "0.0001,0.001",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        lines = result.stdout.splitlines()
        by_variant_dataset = {}
        for line in lines:
            parts = line.split(",", 2)
            if len(parts) >= 2:
                by_variant_dataset[(parts[0], parts[1])] = line

        for dataset in ("sphere", "torus"):
            mm = by_variant_dataset[("dpgo_mm_mm", dataset)]
            amm = by_variant_dataset[("dpgo_mm_amm", dataset)]
            manual = by_variant_dataset[("manual_reduced", dataset)]
            self.assertIn("updates=4", mm)
            self.assertIn("MAX_ITERS=5", mm)
            self.assertIn("SCHEME=mm", mm)
            self.assertIn("ACCELERATED=false", mm)
            self.assertIn("AMM_TRACE=true", mm)
            self.assertNotIn("/tmp/manual-release", mm)
            self.assertIn("updates=4", amm)
            self.assertIn("MAX_ITERS=5", amm)
            self.assertIn("SCHEME=amm", amm)
            self.assertIn("ACCELERATED=true", amm)
            self.assertIn("AMM_TRACE=true", amm)
            self.assertNotIn("/tmp/manual-release", amm)
            self.assertIn("updates=4", manual)
            self.assertIn("MAX_ITERS=4", manual)
            self.assertIn("BIN=/tmp/manual-release", manual)
            self.assertIn("SCHEME=amm", manual)
            self.assertIn("ACCELERATED=true", manual)
            self.assertIn("LOCAL_SOLVER=reduced_rotation", manual)
            self.assertIn("LOCAL_MAX_ITERATIONS_ACCEPTED=2", manual)
            self.assertIn("LOCAL_MAX_TCG_ITERATIONS=10", manual)
            self.assertIn("AMM_DPGO_SURROGATE_PARITY=true", manual)
            self.assertIn("AMM_DPGO_RECURSIVE_SIMPLE_STATE=true", manual)
            self.assertIn("AMM_RECURSIVE_SIMPLE_REANCHOR_PERIOD=10", manual)
            self.assertIn("AMM_DPGO_STRICT_REFINED_GATE=true", manual)
            self.assertIn(
                "AMM_DPGO_RECOVER_TRANSLATIONS_AFTER_PROXIMAL=true",
                manual,
            )
            self.assertIn(
                "AMM_DPGO_REFINED_STARTS_AT_RECOVERED_PROXIMAL=true",
                manual,
            )
            self.assertIn("TRACE_AMM=true", manual)
            self.assertIn("LOCAL_GRADIENT_CORRECTION=true", manual)
            self.assertIn("LOCAL_GRADIENT_CORRECTION_STEPS=0.0001\\,0.001",
                          manual)
            self.assertIn("LOCAL_GRADIENT_CORRECTION_SHARED_STEP=true",
                          manual)
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_NEIGHBORHOOD_STEP=true",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_CURVATURE_STEP=true",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION=true",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_BUDGET_FRACTION=0.5",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_MAX_PACKETS_PER_RECEIVER=8",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_TOPK_ENTRIES=2",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_MIN_SCORE=0.03",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_MIN_SCORE_RATIO=0.2",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COUPLED_DIRECTION_BYTE_BUDGET_MB=0.001",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_BOUNDARY_ONLY=true",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_BLOCK_JACOBI=true",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR=true",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_MAX_BOUNDARY_POSES=5",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_MAX_PRIVATE_COLS=12",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_DAMPING=0.02",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_GRADIENT_GUARD=true",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_COMPACT_SCHUR_MAX_GRADIENT_INCREASE_RATIO=0.2",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_FRESH_NEIGHBOR_EXCHANGE=true",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_FRESH_MIN_POSE_DELTA=0.02",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_FRESH_DELTA_MODE=weighted",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_FRESH_BUDGET_FRACTION=0.25",
                manual,
            )
            self.assertIn(
                "LOCAL_GRADIENT_CORRECTION_FRESH_MAX_POSES_PER_RECEIVER=6",
                manual,
            )
            self.assertIn("LOCAL_GRADIENT_CORRECTION_INNER_ROUNDS=3",
                          manual)
            self.assertIn("LOCAL_GRADIENT_CORRECTION_MAX_ROUNDS=3",
                          manual)
            self.assertIn("POST_EXCHANGE_PERIOD=2", manual)
            self.assertIn("POST_EXCHANGE_MIN_POSE_DELTA=0.01", manual)
            self.assertIn("POST_EXCHANGE_DELTA_MODE=weighted", manual)
            self.assertIn("POST_EXCHANGE_BUDGET_FRACTION=0.5", manual)
            self.assertIn("POST_EXCHANGE_MAX_POSES_PER_RECEIVER=7", manual)
            self.assertIn(
                "SYNC_AMM_REFERENCE_AFTER_LOCAL_GRADIENT_CORRECTION=false",
                manual,
            )
            self.assertIn("GLOBAL_GRADIENT_CORRECTION=true", manual)
            self.assertIn("GLOBAL_GRADIENT_CORRECTION_STEPS=0.0001\\,0.001",
                          manual)

    def test_manual_replay_pairwise_table_keeps_header_first(self):
        script = ROOT / "scripts" / "compare_manual_dpgo_mm_replay.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "comparison.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "dataset", "updates", "variant", "status", "global_cost",
                    "gradient", "total_comm_mb", "wall_time_sec",
                    "solver_time_per_node_sec", "run_root", "summary_csv",
                ])
                writer.writerow(["b", "4", "manual_reduced", "ok", "9", "", "", "", "", "", ""])
                writer.writerow(["b", "4", "dpgo_mm_amm", "ok", "10", "", "", "", "", "", ""])
                writer.writerow(["a", "4", "manual_reduced", "ok", "12", "", "", "", "", "", ""])
                writer.writerow(["a", "4", "dpgo_mm_mm", "ok", "11", "", "", "", "", "", ""])

            subprocess.run(
                [
                    str(script),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "DRY_RUN": "true",
                    "PAIRWISE_ONLY": "true",
                    "OUT_ROOT": str(root),
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            lines = (root / "pairwise_cost_delta.csv").read_text().splitlines()
            self.assertGreaterEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith("dataset,updates,baseline"))

    def test_convergence_analyzer_reads_manual_replay_root(self):
        script = ROOT / "scripts" / "compare_six_convergence.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for method_dir, costs in {
                "dpgo_mm_mm": [10.0, 9.0, 8.9995, 8.9992],
                "dpgo_mm_amm": [10.0, 8.5, 8.4998, 8.4997],
                "manual_reduced": [10.0, 8.0, 7.9996, 7.9995],
            }.items():
                run_dir = root / method_dir / "sphere" / "runs" / "sphere"
                run_dir.mkdir(parents=True)
                with (run_dir / "iteration_summary.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([
                        "iter",
                        "global_cost",
                        "cumulative_comm_pose_count",
                        "cumulative_comm_mb",
                        "time",
                    ])
                    for idx, cost in enumerate(costs):
                        writer.writerow([idx, cost, 100 * (idx + 1), 0.1 * (idx + 1), idx * 0.5])

            output = root / "convergence.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--manual-replay-root",
                    str(root),
                    "--datasets",
                    "sphere",
                    "--threshold",
                    "0.001",
                    "--window",
                    "2",
                    "-o",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row["method"] for row in rows},
                             {"DPGO-MM MM", "DPGO-MM AMM", "Manual-reduced"})
            manual = next(row for row in rows if row["method"] == "Manual-reduced")
            self.assertEqual(manual["converged"], "1")
            self.assertEqual(manual["convergence_iter"], "3")
            self.assertEqual(manual["convergence_cost"], "7.9995")

    def test_phase7_advantage_analyzer_reports_pairwise_overtake(self):
        script = ROOT / "scripts" / "analyze_phase7_dpgo_mm_advantage.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "normalized_metrics.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "problem_type",
                    "dataset",
                    "method",
                    "setting",
                    "budget",
                    "status",
                    "cost",
                    "optimal_cost",
                    "total_comm_mb",
                ])
                writer.writerow(["six", "sphere", "SE-Sync", "reference", "", "GlobalOpt", "5.0", "5.0", ""])
                writer.writerow(["six", "sphere", "DPGO-MM", "fixed_3", "3", "ok", "6.0", "5.0", "0.3"])
                writer.writerow(["six", "sphere", "Manual-reduced", "fixed_3", "3", "ok", "7.0", "5.0", "0.3"])
                writer.writerow(["six", "sphere", "DRAN", "fixed_3", "3", "ok", "6.5", "5.0", "0.6"])

            with (root / "curves.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "problem_type",
                    "dataset",
                    "method",
                    "setting",
                    "budget",
                    "cost",
                    "iter",
                    "cumulative_comm_mb",
                ])
                for method, costs, comms in (
                    ("DPGO-MM", [10.0, 8.0, 6.0], [0.1, 0.2, 0.3]),
                    ("Manual-reduced", [9.0, 7.5, 7.0], [0.1, 0.2, 0.3]),
                    ("DRAN", [8.5, 7.2, 6.5], [0.2, 0.4, 0.6]),
                ):
                    for idx, (cost, comm) in enumerate(zip(costs, comms)):
                        writer.writerow(["six", "sphere", method, "fixed_3", "3", cost, idx, comm])

            manual_amm_root = root / "manual_amm"
            manual_amm_run = manual_amm_root / "runs" / "sphere"
            manual_amm_run.mkdir(parents=True)
            manual_amm_iters = manual_amm_run / "iteration_summary.csv"
            with manual_amm_iters.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "iter",
                    "global_cost",
                    "gradient",
                    "cumulative_comm_mb",
                    "amm_trace_count",
                    "amm_accelerated_accepted_count",
                    "amm_restart_count",
                    "amm_local_merit_rejected_count",
                ])
                writer.writerow([0, 8.8, 1.0, 0.1, 1, 0, 0, 1])
                writer.writerow([1, 7.0, 0.7, 0.2, 1, 1, 0, 0])
                writer.writerow([2, 6.5, 0.5, 0.3, 1, 0, 1, 0])
            manual_amm_summary = manual_amm_root / "summary.csv"
            with manual_amm_summary.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow([
                    "dataset",
                    "method",
                    "status",
                    "global_cost",
                    "gradient",
                    "total_comm_mb",
                    "iter_summary_path",
                ])
                writer.writerow([
                    "sphere",
                    "manual_reduced_amm",
                    "ok",
                    "6.5",
                    "0.5",
                    "0.3",
                    str(manual_amm_iters),
                ])

            out_dir = root / "analysis"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--result-root",
                    str(root),
                    "--output-dir",
                    str(out_dir),
                    "--datasets",
                    "sphere",
                    "--extra-manual-summary",
                    f"Manual-reduced-AMM={manual_amm_summary}",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            pair_rows = list(csv.DictReader((out_dir / "phase7_pairwise_vs_dpgo_mm.csv").open()))
            manual = next(row for row in pair_rows if row["method"] == "Manual-reduced")
            self.assertEqual(manual["method_minus_baseline_cost"], "1")
            self.assertEqual(manual["method_comm_ratio_vs_baseline"], "1")
            self.assertEqual(manual["baseline_first_overtake_iter"], "2")
            progress_rows = list(csv.DictReader((out_dir / "phase7_method_progress.csv").open()))
            dpgo_mm = next(row for row in progress_rows if row["method"] == "DPGO-MM")
            self.assertEqual(dpgo_mm["final_gap"], "1")
            self.assertEqual(dpgo_mm["gap_closed"], "4")
            manual_amm = next(row for row in progress_rows
                              if row["method"] == "Manual-reduced-AMM")
            self.assertEqual(manual_amm["final_cost"], "6.5")
            self.assertEqual(manual_amm["cost_decrease_per_outer_mb"], "11.5")
            counter_rows = list(csv.DictReader(
                (out_dir / "phase7_amm_counters.csv").open()))
            manual_amm_counter = next(
                row for row in counter_rows
                if row["method"] == "Manual-reduced-AMM")
            self.assertEqual(manual_amm_counter["amm_trace_count"], "3")
            self.assertEqual(
                manual_amm_counter["amm_accelerated_accepted_count"], "1")
            self.assertEqual(manual_amm_counter["amm_restart_count"], "1")

    def test_chordal_fair_dry_run_includes_manual_reduced_manifest_arm(self):
        script = ROOT / "scripts" / "run_chordal_fair_benchmarks.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run(
                [
                    str(script),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "DRY_RUN": "true",
                    "OUT_ROOT": str(root),
                    "RUN_DRONE": "false",
                    "RUN_SESYNC": "false",
                    "RUN_DRAN": "false",
                    "RUN_DPGO_FIRST": "false",
                    "RUN_DPGO_SECOND": "false",
                    "RUN_DPGO_MM": "false",
                    "RUN_MESA": "false",
                    "RUN_DISTRIBUTED_MAPPER": "false",
                    "RUN_MANUAL_REDUCED": "true",
                    "BUDGETS": "2",
                    "NUM_ROBOTS": "5",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            self.assertIn("Manual-reduced", result.stdout)
            rows = list(csv.DictReader((root / "manifest.csv").open()))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["problem_type"], "six")
            self.assertEqual(row["setting"], "fixed_2")
            self.assertEqual(row["method"], "Manual-reduced")
            self.assertEqual(row["budget"], "2")
            self.assertEqual(row["status"], "dry_run")
            self.assertIn("run_manual_dpgo_mm_six.sh", row["command"])
            self.assertIn("LOCAL_SOLVER=reduced_rotation", row["command"])
            self.assertIn("LOCAL_MAX_TCG_ITERATIONS=10", row["command"])
            self.assertIn("ADAPTIVE_REDUCED_TCG=true", row["command"])
            self.assertIn("LOCAL_STATE_EXTRAPOLATION=true", row["command"])

    def test_chordal_fair_manual_reduced_is_opt_in(self):
        script = ROOT / "scripts" / "run_chordal_fair_benchmarks.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(
                [
                    str(script),
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "DRY_RUN": "true",
                    "OUT_ROOT": str(root),
                    "RUN_DRONE": "false",
                    "RUN_SESYNC": "false",
                    "RUN_DRAN": "false",
                    "RUN_DPGO_FIRST": "false",
                    "RUN_DPGO_SECOND": "false",
                    "RUN_DPGO_MM": "false",
                    "RUN_MESA": "false",
                    "RUN_DISTRIBUTED_MAPPER": "false",
                    "BUDGETS": "2",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = list(csv.DictReader((root / "manifest.csv").open()))
            self.assertEqual(rows, [])

    def test_manual_replay_script_caps_nested_linear_algebra_threads(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        env = os.environ.copy()
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env.pop(key, None)
        env.update({
            "DRY_RUN": "true",
            "DATASETS": "sphere",
            "LOCAL_SOLVER": "reduced_rotation",
        })
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("THREAD_CONFIG", result.stdout)
        self.assertIn("OMP_NUM_THREADS=1", result.stdout)
        self.assertIn("OPENBLAS_NUM_THREADS=1", result.stdout)
        self.assertIn("MKL_NUM_THREADS=1", result.stdout)

    def test_manual_replay_script_forwards_adaptive_reduced_tcg(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "ADAPTIVE_REDUCED_TCG": "true",
                "ADAPTIVE_REDUCED_TCG_MAX_ITERATIONS": "50",
                "ADAPTIVE_REDUCED_TCG_GRADIENT_RATIO": "0.15",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--adaptive_reduced_tcg true", result.stdout)
        self.assertIn("--adaptive_reduced_tcg_max_iterations 50", result.stdout)
        self.assertIn("--adaptive_reduced_tcg_gradient_ratio 0.15", result.stdout)
        self.assertIn("--local_state_extrapolation false", result.stdout)

    def test_manual_replay_script_forwards_feh_pcg_schur_options(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_schur",
                "FEH_LINEAR_REL_TOL": "1e-7",
                "FEH_LINEAR_ABS_TOL": "1e-11",
                "FEH_LINEAR_MAX_ITERS": "123",
                "FEH_LINEAR_BLOCK_JACOBI": "false",
                "FEH_SPARSE_MATVEC": "false",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_schur", result.stdout)
        self.assertIn("--feh_linear_rel_tol 1e-7", result.stdout)
        self.assertIn("--feh_linear_abs_tol 1e-11", result.stdout)
        self.assertIn("--feh_linear_max_iters 123", result.stdout)
        self.assertIn("--feh_linear_block_jacobi false", result.stdout)
        self.assertIn("--feh_sparse_matvec false", result.stdout)

    def test_manual_replay_script_forwards_feh_pcg_full_backend(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_LINEAR_MAX_ITERS": "321",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_linear_max_iters 321", result.stdout)

    def test_manual_replay_script_forwards_feh_translation_recovery_polish(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_TRANSLATION_RECOVERY_POLISH": "true",
                "FEH_TRANSLATION_RECOVERY_POLISH_GRADIENT_GUARD": "true",
                "FEH_TRANSLATION_RECOVERY_POLISH_MAX_GRADIENT_INCREASE_RATIO": "0.25",
                "FEH_TRANSLATION_RECOVERY_POLISH_BACKTRACKING": "true",
                "FEH_TRANSLATION_RECOVERY_POLISH_BACKTRACKING_STEPS": "6",
                "FEH_TRANSLATION_RECOVERY_POLISH_MERIT_SELECTOR": "true",
                "FEH_TRANSLATION_RECOVERY_POLISH_MERIT_MIN_COST_RECOVERY_RATIO": "0.85",
                "FEH_TRANSLATION_RECOVERY_POLISH_SELECTED_ONLY": "true",
                "FEH_TRANSLATION_RECOVERY_STEP_TRIALS": "true",
                "FEH_STEP_PARETO_SELECTOR": "true",
                "FEH_STEP_PARETO_MIN_DECREASE_RATIO": "0.8",
                "FEH_ACTIVE_SEPARATOR_LM_SCHUR": "true",
                "FEH_ACTIVE_SEPARATOR_LM_SCHUR_SCORE_MODE": "model_decrease",
                "FEH_ACTIVE_SEPARATOR_LM_SCHUR_GRADIENT_GUARD": "true",
                "FEH_ACTIVE_SEPARATOR_LM_SCHUR_MAX_GRADIENT_INCREASE_RATIO": "0.1",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_translation_recovery_polish true", result.stdout)
        self.assertIn(
            "--feh_translation_recovery_polish_gradient_guard true",
            result.stdout,
        )
        self.assertIn(
            "--feh_translation_recovery_polish_max_gradient_increase_ratio 0.25",
            result.stdout,
        )
        self.assertIn(
            "--feh_translation_recovery_polish_backtracking true",
            result.stdout,
        )
        self.assertIn(
            "--feh_translation_recovery_polish_backtracking_steps 6",
            result.stdout,
        )
        self.assertIn(
            "--feh_translation_recovery_polish_merit_selector true",
            result.stdout,
        )
        self.assertIn(
            "--feh_translation_recovery_polish_merit_min_cost_recovery_ratio 0.85",
            result.stdout,
        )
        self.assertIn(
            "--feh_translation_recovery_polish_selected_only true",
            result.stdout,
        )
        self.assertIn(
            "--feh_translation_recovery_step_trials true",
            result.stdout,
        )
        self.assertIn("--feh_step_pareto_selector true", result.stdout)
        self.assertIn(
            "--feh_step_pareto_min_decrease_ratio 0.8",
            result.stdout,
        )
        self.assertIn(
            "--feh_active_separator_lm_schur true",
            result.stdout,
        )
        self.assertIn(
            "--feh_active_separator_lm_schur_score_mode model_decrease",
            result.stdout,
        )
        self.assertIn(
            "--feh_active_separator_lm_schur_gradient_guard true",
            result.stdout,
        )
        self.assertIn(
            "--feh_active_separator_lm_schur_max_gradient_increase_ratio 0.1",
            result.stdout,
        )

    def test_manual_replay_script_accepts_output_dir_alias(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "OUTPUT_DIR": "results/manual_output_dir_alias_dryrun",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--output_dir "
            + str(ROOT / "results" / "manual_output_dir_alias_dryrun" / "runs" / "sphere"),
            result.stdout,
        )
        self.assertIn(
            "Saved summary: "
            + str(ROOT / "results" / "manual_output_dir_alias_dryrun" / "summary.csv"),
            result.stdout,
        )

    def test_manual_replay_script_forwards_feh_translation_schur_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_PRECOND_TRANSLATION_SCHUR": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_precond_translation_schur true", result.stdout)

    def test_manual_replay_script_forwards_feh_translation_block_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_PRECOND_TRANSLATION_BLOCK": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_precond_translation_block true", result.stdout)

    def test_manual_replay_script_forwards_feh_translation_sparse_schur_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_PRECOND_TRANSLATION_SPARSE_SCHUR": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_precond_translation_sparse_schur true",
                      result.stdout)

    def test_manual_replay_script_forwards_feh_translation_local_schur_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_PRECOND_TRANSLATION_LOCAL_SCHUR": "true",
                "FEH_PRECOND_TRANSLATION_LOCAL_SCHUR_MAX_POSES": "12",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_precond_translation_local_schur true",
                      result.stdout)
        self.assertIn("--feh_precond_translation_local_schur_max_poses 12",
                      result.stdout)

    def test_manual_replay_script_forwards_feh_laplacian_deflation_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_PRECOND_LAPLACIAN_DEFLATION": "true",
                "FEH_PRECOND_LAPLACIAN_DEFLATION_BASIS_SIZE": "6",
                "FEH_PRECOND_LAPLACIAN_DEFLATION_MAX_EIGEN_POSES": "128",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_precond_laplacian_deflation true",
                      result.stdout)
        self.assertIn("--feh_precond_laplacian_deflation_basis_size 6",
                      result.stdout)
        self.assertIn("--feh_precond_laplacian_deflation_max_eigen_poses 128",
                      result.stdout)

    def test_manual_replay_script_forwards_feh_rqn_warm_start_options(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_RQN_WARM_START": "true",
                "FEH_RQN_MEMORY_SIZE": "7",
                "FEH_RQN_MIN_CURVATURE_RATIO": "1e-9",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_rqn_warm_start true", result.stdout)
        self.assertIn("--feh_rqn_memory_size 7", result.stdout)
        self.assertIn("--feh_rqn_min_curvature_ratio 1e-9", result.stdout)

    def test_manual_replay_script_forwards_feh_rqn_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_PRECOND_RQN_MEMORY": "true",
                "FEH_RQN_MEMORY_SIZE": "6",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_precond_rqn_memory true", result.stdout)
        self.assertIn("--feh_rqn_memory_size 6", result.stdout)

    def test_manual_replay_script_forwards_feh_local_chain_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_PRECOND_LOCAL_CHAIN": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_precond_local_chain true", result.stdout)

    def test_manual_replay_script_forwards_feh_schwarz_presmoothing(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_SCHWARZ_PRESMOOTHING": "true",
                "FEH_SCHWARZ_SWEEPS": "2",
                "FEH_SCHWARZ_MAX_BLOCK_NORM": "0.5",
                "FEH_SCHWARZ_BLOCK_MODE": "edge_pair",
                "FEH_LOCAL_PORTFOLIO": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_schwarz_presmoothing true", result.stdout)
        self.assertIn("--feh_schwarz_sweeps 2", result.stdout)
        self.assertIn("--feh_schwarz_max_block_norm 0.5", result.stdout)
        self.assertIn("--feh_schwarz_block_mode edge_pair", result.stdout)
        self.assertIn("--feh_local_portfolio true", result.stdout)

    def test_manual_replay_dry_run_does_not_require_built_binary(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        missing_bin = ROOT / "build" / "bin" / "definitely-missing-manual-dpgo"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "BIN": str(missing_bin),
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("DRY_RUN=1", result.stdout)
        self.assertIn(str(missing_bin), result.stdout)
        self.assertIn("--feh_linear_backend pcg_full", result.stdout)

    def test_manual_replay_script_forwards_feh_reduced_rotation_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_PRECOND_REDUCED_ROTATION": "true",
                "REDUCED_ROTATION_PRECONDITIONER": "cholesky",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_precond_reduced_rotation true", result.stdout)
        self.assertIn("--reduced_rotation_preconditioner cholesky", result.stdout)

    def test_manual_replay_script_forwards_feh_reduced_rotation_initial_guess(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_REDUCED_ROTATION_INITIAL_GUESS": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_reduced_rotation_initial_guess true", result.stdout)

    def test_manual_replay_script_forwards_feh_translation_recovery_initial_guess(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "LOCAL_SOLVER": "full_equiv_hybrid",
                "FEH_LINEAR_BACKEND": "pcg_full",
                "FEH_TRANSLATION_RECOVERY_INITIAL_GUESS": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_translation_recovery_initial_guess true",
                      result.stdout)

    def test_feh_preconditioner_ablation_dry_run_lists_fair_variants(self):
        script = ROOT / "scripts" / "run_feh_preconditioner_ablation.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "MAX_ITERS": "3",
                "OUT_ROOT": "results/test_feh_preconditioner_ablation",
                "FEH_RQN_MEMORY_SIZE": "9",
                "FEH_RQN_MIN_CURVATURE_RATIO": "1e-7",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("variant=block_jacobi dataset=sphere", result.stdout)
        self.assertIn("variant=local_chain dataset=sphere", result.stdout)
        self.assertIn("variant=laplacian_deflation dataset=sphere",
                      result.stdout)
        self.assertIn("variant=rqn_memory dataset=sphere", result.stdout)
        self.assertIn("variant=translation_block dataset=sphere", result.stdout)
        self.assertIn("variant=translation_sparse_schur dataset=sphere",
                      result.stdout)
        self.assertIn("variant=translation_sparse_schur_laplacian dataset=sphere",
                      result.stdout)
        self.assertIn("variant=translation_local_schur dataset=sphere",
                      result.stdout)
        self.assertIn("variant=translation_local_schur_full dataset=sphere",
                      result.stdout)
        self.assertIn("variant=translation_schur dataset=sphere", result.stdout)
        self.assertIn("variant=rr_cholesky dataset=sphere", result.stdout)
        self.assertIn("variant=rr_jacobi dataset=sphere", result.stdout)
        self.assertIn("variant=rr_schur_jacobi dataset=sphere", result.stdout)
        self.assertIn("--local_solver full_equiv_hybrid", result.stdout)
        self.assertIn("--feh_linear_backend pcg_full", result.stdout)
        self.assertIn("--feh_sparse_matvec true", result.stdout)
        self.assertIn("--feh_linear_max_iters 200", result.stdout)
        self.assertIn("--feh_precond_local_chain true", result.stdout)
        self.assertIn("--feh_precond_translation_block true", result.stdout)
        self.assertIn("--feh_precond_translation_sparse_schur true",
                      result.stdout)
        self.assertIn("--feh_precond_laplacian_deflation true", result.stdout)
        self.assertIn("--feh_precond_laplacian_deflation_basis_size 4",
                      result.stdout)
        self.assertIn("--feh_precond_laplacian_deflation_max_eigen_poses 512",
                      result.stdout)
        self.assertIn("--feh_precond_rqn_memory true", result.stdout)
        self.assertIn("--feh_rqn_memory_size 9", result.stdout)
        self.assertIn("--feh_rqn_min_curvature_ratio 1e-7", result.stdout)
        self.assertIn("--feh_precond_translation_local_schur true",
                      result.stdout)
        self.assertIn("--feh_precond_translation_local_schur_max_poses 32",
                      result.stdout)
        self.assertIn("--feh_precond_translation_local_schur_max_poses 0",
                      result.stdout)
        self.assertIn("--feh_precond_translation_schur true", result.stdout)
        self.assertIn("--feh_precond_reduced_rotation true", result.stdout)
        self.assertIn("--reduced_rotation_preconditioner schur_jacobi",
                      result.stdout)
        self.assertIn("--feh_reduced_rotation_initial_guess false",
                      result.stdout)

    def test_manual_replay_script_forwards_baseline_surrogate_state(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "inter",
                "SCHEME": "amm",
                "LOCAL_SOLVER": "reduced_rotation",
                "LOCAL_MAX_ITERATIONS": "10",
                "LOCAL_MAX_ITERATIONS_ACCEPTED": "1",
                "AMM_ACCEPTED_DELTA": "0.0007",
                "AMM_OSCILLATION_COUNT_PERIOD": "9",
                "AMM_MAX_OSCILLATIONS": "4",
                "AMM_DPGO_SURROGATE_PARITY": "true",
                "AMM_BASELINE_SURROGATE_STATE": "true",
                "AMM_DPGO_RECURSIVE_SIMPLE_STATE": "true",
                "AMM_RECURSIVE_SIMPLE_REANCHOR_PERIOD": "10",
                "AMM_DPGO_STRICT_REFINED_GATE": "true",
                "AMM_DPGO_RECOVER_TRANSLATIONS_AFTER_PROXIMAL": "true",
                "AMM_DPGO_REFINED_STARTS_AT_RECOVERED_PROXIMAL": "true",
                "TRACE_AMM": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--local_max_iterations 10", result.stdout)
        self.assertIn("--local_max_iterations_accepted 1", result.stdout)
        self.assertIn("--amm_accepted_delta 0.0007", result.stdout)
        self.assertIn("--amm_oscillation_count_period 9", result.stdout)
        self.assertIn("--amm_max_oscillations 4", result.stdout)
        self.assertIn("--amm_dpgo_surrogate_parity true", result.stdout)
        self.assertIn("--amm_baseline_surrogate_state true", result.stdout)
        self.assertIn("--amm_dpgo_recursive_simple_state true", result.stdout)
        self.assertIn("--amm_recursive_simple_reanchor_period 10", result.stdout)
        self.assertIn("--amm_dpgo_strict_refined_gate true", result.stdout)
        self.assertIn(
            "--amm_dpgo_recover_translations_after_proximal true",
            result.stdout,
        )
        self.assertIn(
            "--amm_dpgo_refined_starts_at_recovered_proximal true",
            result.stdout,
        )
        self.assertIn("--trace_amm true", result.stdout)

    def test_manual_replay_script_forwards_distributed_init(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "DIST_INIT": "true",
                "DIST_INIT_REFINEMENT_ROUNDS": "3",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--dist_init true", result.stdout)
        self.assertIn("--dist_init_refinement_rounds 3", result.stdout)

    def test_manual_replay_script_forwards_local_extrapolation(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "LOCAL_STATE_EXTRAPOLATION": "true",
                "LOCAL_STATE_EXTRAPOLATION_GAMMA": "0.4",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--local_state_extrapolation true", result.stdout)
        self.assertIn("--local_state_extrapolation_gamma 0.4", result.stdout)

    def test_manual_replay_script_forwards_local_extrapolation_gammas(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "LOCAL_STATE_EXTRAPOLATION": "true",
                "LOCAL_STATE_EXTRAPOLATION_GAMMAS": "0.1,0.25,0.5",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--local_state_extrapolation true", result.stdout)
        self.assertIn("--local_state_extrapolation_gammas", result.stdout)
        self.assertIn("0.1\\,0.25\\,0.5", result.stdout)

    def test_manual_replay_script_forwards_local_model_g_extrapolation(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "LOCAL_MODEL_G_EXTRAPOLATION": "true",
                "LOCAL_MODEL_G_EXTRAPOLATION_GAMMAS": "0.25,0.5",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--local_model_g_extrapolation true", result.stdout)
        self.assertIn("--local_model_g_extrapolation_gammas", result.stdout)
        self.assertIn("0.25\\,0.5", result.stdout)

    def test_manual_replay_script_forwards_candidate_tie_tolerance(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "LOCAL_CANDIDATE_COST_TIE_TOLERANCE": "1e-3",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--local_candidate_cost_tie_tolerance 1e-3", result.stdout)

    def test_manual_replay_script_forwards_coupled_state_g_extrapolation(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "COUPLED_STATE_G_EXTRAPOLATION": "true",
                "COUPLED_STATE_G_EXTRAPOLATION_GAMMAS": "0.25,0.5",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--coupled_state_g_extrapolation true", result.stdout)
        self.assertIn("--coupled_state_g_extrapolation_gammas", result.stdout)
        self.assertIn("0.25\\,0.5", result.stdout)

    def test_manual_replay_script_forwards_reduced_rotation_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_PRECONDITIONER": "jacobi",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--reduced_rotation_preconditioner jacobi", result.stdout)

    def test_manual_replay_script_forwards_schur_jacobi_preconditioner(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_PRECONDITIONER": "schur_jacobi",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--reduced_rotation_preconditioner schur_jacobi",
                      result.stdout)

    def test_manual_replay_script_forwards_reduced_rotation_portfolio(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_PRECONDITIONER": "portfolio",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--reduced_rotation_preconditioner portfolio",
                      result.stdout)

    def test_manual_replay_script_forwards_certified_adaptive_portfolio(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_PRECONDITIONER": "adaptive_portfolio",
                "REDUCED_ADAPTIVE_PORTFOLIO_CERTIFIED_FAST_PATH": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--reduced_adaptive_portfolio_certified_fast_path true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_reduced_direct_objective(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_DIRECT_OBJECTIVE": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--reduced_rotation_direct_objective true",
                      result.stdout)

    def test_manual_replay_script_forwards_fused_candidate_evaluation(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "FUSED_CANDIDATE_EVALUATION": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--fused_candidate_evaluation true", result.stdout)

    def test_manual_replay_script_forwards_lazy_candidate_gradient(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "LAZY_CANDIDATE_GRADIENT_EVALUATION": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--lazy_candidate_gradient_evaluation true",
                      result.stdout)

    def test_manual_replay_script_forwards_lazy_solver_start_gradient(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "LAZY_SOLVER_START_GRADIENT_EVALUATION": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--lazy_solver_start_gradient_evaluation true",
                      result.stdout)

    def test_manual_replay_script_forwards_lazy_surrogate_candidate_gradient(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "LAZY_SURROGATE_CANDIDATE_GRADIENT_EVALUATION": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--lazy_surrogate_candidate_gradient_evaluation true",
                      result.stdout)

    def test_manual_replay_script_forwards_curvature_cauchy_candidate(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_CURVATURE_CAUCHY_CANDIDATE": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--reduced_rotation_curvature_cauchy_candidate true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_curvature_fallback_candidate(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_CURVATURE_FALLBACK_CANDIDATE": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--reduced_rotation_curvature_fallback_candidate true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_skip_redundant_projection(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_SKIP_REDUNDANT_CANDIDATE_PROJECTION": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--reduced_rotation_skip_redundant_candidate_projection true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_gradient_boundary_candidate(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_GRADIENT_BOUNDARY_CANDIDATE": "false",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--reduced_rotation_gradient_boundary_candidate false",
            result.stdout,
        )

    def test_manual_replay_script_forwards_surrogate_tcg_accept(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_SURROGATE_TCG_ACCEPT": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--reduced_rotation_surrogate_tcg_accept true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_amm_prox_reset_skip_refine(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "AMM_PROX_RESET_SKIP_REFINED_SOLVE": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--amm_prox_reset_skip_refined_solve true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_amm_dpgo_proximal_fallback_only(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "AMM_DPGO_PROXIMAL_FALLBACK_ONLY": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--amm_dpgo_proximal_fallback_only true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_amm_lazy_plain_after_certificate(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "AMM_LAZY_PLAIN_AFTER_CERTIFICATE": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--amm_lazy_plain_after_certificate true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_amm_surrogate_first_exact_evaluation(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "AMM_SURROGATE_FIRST_EXACT_EVALUATION": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "--amm_surrogate_first_exact_evaluation true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_print_iteration_summary(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "PRINT_ITERATION_SUMMARY": "false",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--print_iteration_summary false", result.stdout)

    def test_manual_replay_script_forwards_lifted_delta_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "COMM_TOPOLOGY_LIFTED_DELTA_COMPRESSION": "true",
                "COMM_TOPOLOGY_LIFTED_DELTA_MAX_RECONSTRUCTION_ERROR": "0.0003",
                "COMM_TOPOLOGY_LIFTED_DELTA_RANK": "1",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--comm_topology_lifted_delta_compression true",
                      result.stdout)
        self.assertIn(
            "--comm_topology_lifted_delta_max_reconstruction_error 0.0003",
            result.stdout,
        )
        self.assertIn("--comm_topology_lifted_delta_rank 1", result.stdout)

    def test_manual_replay_script_forwards_value_scheduler_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "COMM_TOPOLOGY_VALUE_SCHEDULER": "true",
                "COMM_TOPOLOGY_VALUE_SCHEDULER_MODE":
                    "static_sensitivity_staleness",
                "COMM_TOPOLOGY_VALUE_BYTE_BUDGET_MB": "0.55",
                "COMM_TOPOLOGY_VALUE_MIN_SCORE_RATIO": "0.2",
                "COMM_TOPOLOGY_VALUE_BUDGET_PACING": "false",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--comm_topology_value_scheduler true", result.stdout)
        self.assertIn(
            "--comm_topology_value_scheduler_mode "
            "static_sensitivity_staleness",
            result.stdout,
        )
        self.assertIn("--comm_topology_value_byte_budget_mb 0.55",
                      result.stdout)
        self.assertIn("--comm_topology_value_min_score_ratio 0.2",
                      result.stdout)
        self.assertIn("--comm_topology_value_budget_pacing false",
                      result.stdout)

    def test_manual_replay_script_forwards_interface_model_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "COMM_TOPOLOGY_INTERFACE_MODEL": "true",
                "COMM_TOPOLOGY_INTERFACE_MODEL_PAYLOAD":
                    "direction_block_diag_stiffness",
                "COMM_TOPOLOGY_INTERFACE_MODEL_LOCAL_MERIT": "false",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--comm_topology_interface_model true",
                      result.stdout)
        self.assertIn(
            "--comm_topology_interface_model_payload "
            "direction_block_diag_stiffness",
            result.stdout,
        )
        self.assertIn("--comm_topology_interface_model_local_merit false",
                      result.stdout)

    def test_manual_replay_script_forwards_topology_boundary_candidate_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "COMM_TOPOLOGY_BOUNDARY_CANDIDATE": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--comm_topology_boundary_candidate true",
                      result.stdout)

    def test_manual_replay_script_forwards_topology_boundary_surrogate_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "COMM_TOPOLOGY_BOUNDARY_SURROGATE": "true",
                "COMM_TOPOLOGY_BOUNDARY_SURROGATE_STALE_GAIN": "0.5",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--comm_topology_boundary_surrogate true",
                      result.stdout)
        self.assertIn(
            "--comm_topology_boundary_surrogate_stale_gain 0.5",
            result.stdout,
        )

    def test_manual_replay_script_forwards_topology_reduced_interface_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere",
                "COMM_TOPOLOGY_REDUCED_INTERFACE_MODEL": "true",
                "COMM_TOPOLOGY_REDUCED_INTERFACE_WEIGHT": "0.1",
                "COMM_TOPOLOGY_REDUCED_INTERFACE_MAX_LOCAL_ITERS": "7",
                "COMM_TOPOLOGY_REDUCED_INTERFACE_CANDIDATE": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--comm_topology_reduced_interface_model true",
                      result.stdout)
        self.assertIn("--comm_topology_reduced_interface_weight 0.1",
                      result.stdout)
        self.assertIn(
            "--comm_topology_reduced_interface_max_local_iters 7",
            result.stdout,
        )
        self.assertIn("--comm_topology_reduced_interface_candidate true",
                      result.stdout)

    def test_manual_replay_script_forwards_packed_tcg_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_PACKED_TCG": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "DRAN_REDUCED_ROTATION_PACKED_TCG=true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_packed_cholesky_project_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT": "true",
                "REDUCED_ROTATION_PACKED_TCG": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "DRAN_REDUCED_ROTATION_PACKED_CHOLESKY_PROJECT=true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_packed_riemannian_hvp_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_PACKED_TCG": "true",
                "REDUCED_ROTATION_PACKED_RIEMANNIAN_HVP": "true",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "DRAN_REDUCED_ROTATION_PACKED_RIEMANNIAN_HVP=true",
            result.stdout,
        )

    def test_manual_replay_script_forwards_compact_hvp_env(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_ROTATION_COMPACT_HVP": "true",
                "REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP": "false",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn(
            "DRAN_REDUCED_ROTATION_COMPACT_HVP=true",
            result.stdout,
        )
        self.assertIn(
            "DRAN_REDUCED_ROTATION_PRECOMPUTED_RESPONSE_HVP=false",
            result.stdout,
        )

    def test_manual_replay_script_forwards_reduced_surrogate_mode(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        default_result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        self.assertIn("--reduced_surrogate_mode true_local",
                      default_result.stdout)
        self.assertNotIn("--reduced_surrogate_mode edge_tight_quadratic",
                         default_result.stdout)

        edge_tight_result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "REDUCED_SURROGATE_MODE": "edge_tight_quadratic",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        self.assertIn("--reduced_surrogate_mode edge_tight_quadratic",
                      edge_tight_result.stdout)

    def test_manual_replay_script_forwards_vm_bmm_aa_flags(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        default_result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        self.assertIn("--surrogate_mode legacy", default_result.stdout)
        self.assertIn("--edge_split_theta_mode constant",
                      default_result.stdout)
        self.assertIn("--edge_split_theta_default 0.5",
                      default_result.stdout)
        self.assertIn("--edge_split_theta_min 0.15", default_result.stdout)
        self.assertIn("--edge_split_theta_max 0.85", default_result.stdout)
        self.assertIn("--edge_split_theta_candidates 0.2\\,0.35\\,0.5\\,0.65\\,0.8",
                      default_result.stdout)
        self.assertIn("--mm_accelerator nesterov_legacy",
                      default_result.stdout)
        self.assertIn("--mm_safeguard local_surrogate",
                      default_result.stdout)
        self.assertIn("--debug_surrogate_bound_check false",
                      default_result.stdout)
        self.assertIn("--debug_surrogate_bound_samples 8",
                      default_result.stdout)

        weighted_result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "LOCAL_SOLVER": "reduced_rotation",
                "SURROGATE_MODE": "weighted_edge_split",
                "EDGE_SPLIT_THETA_MODE": "degree",
                "EDGE_SPLIT_THETA_DEFAULT": "0.35",
                "EDGE_SPLIT_THETA_MIN": "0.1",
                "EDGE_SPLIT_THETA_MAX": "0.9",
                "EDGE_SPLIT_THETA_CANDIDATES": "0.25,0.5,0.75",
                "MM_ACCELERATOR": "anderson",
                "MM_SAFEGUARD": "local_surrogate_plus_boundary",
                "DEBUG_SURROGATE_BOUND_CHECK": "true",
                "DEBUG_SURROGATE_BOUND_SAMPLES": "4",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )
        self.assertIn("--surrogate_mode weighted_edge_split",
                      weighted_result.stdout)
        self.assertIn("--edge_split_theta_mode degree",
                      weighted_result.stdout)
        self.assertIn("--edge_split_theta_default 0.35",
                      weighted_result.stdout)
        self.assertIn("--edge_split_theta_min 0.1", weighted_result.stdout)
        self.assertIn("--edge_split_theta_max 0.9", weighted_result.stdout)
        self.assertIn("--edge_split_theta_candidates 0.25\\,0.5\\,0.75",
                      weighted_result.stdout)
        self.assertIn("--mm_accelerator anderson", weighted_result.stdout)
        self.assertIn("--mm_safeguard local_surrogate_plus_boundary",
                      weighted_result.stdout)
        self.assertIn("--debug_surrogate_bound_check true",
                      weighted_result.stdout)
        self.assertIn("--debug_surrogate_bound_samples 4",
                      weighted_result.stdout)

    def test_manual_replay_script_allows_amm_dry_run(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "SCHEME": "amm",
                "ACCELERATED": "true",
                "LOCAL_SOLVER": "reduced_rotation",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--scheme amm", result.stdout)
        self.assertIn("--accelerated true", result.stdout)

    def test_manual_replay_script_forwards_mixed_surrogate_portfolio(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "SCHEME": "amm",
                "ACCELERATED": "true",
                "AMM_DPGO_SURROGATE_PARITY": "true",
                "AMM_DPGO_MIXED_SURROGATE_PORTFOLIO": "true",
                "AMM_MIXED_SURROGATE_SKIP_SIMPLE_AFTER_TRUE_LOCAL_STREAK": "2",
                "AMM_MIXED_SURROGATE_FORCE_SIMPLE_EVERY_SKIPPED_ROUNDS": "3",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--amm_dpgo_surrogate_parity true", result.stdout)
        self.assertIn("--amm_dpgo_mixed_surrogate_portfolio true",
                      result.stdout)
        self.assertIn(
            "--amm_mixed_surrogate_skip_simple_after_true_local_streak 2",
            result.stdout,
        )
        self.assertIn(
            "--amm_mixed_surrogate_force_simple_every_skipped_rounds 3",
            result.stdout,
        )
        self.assertIn("--amm_local_merit_filter true", result.stdout)

    def test_manual_replay_script_forwards_amm_gamma_scale(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "SCHEME": "amm",
                "ACCELERATED": "true",
                "AMM_GAMMA_SCALE": "0.25",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--amm_gamma_scale 0.25", result.stdout)

    def test_manual_replay_script_forwards_global_state_extrapolation(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "GLOBAL_STATE_EXTRAPOLATION": "true",
                "GLOBAL_STATE_EXTRAPOLATION_GAMMAS": "0.25,0.5",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--global_state_extrapolation true", result.stdout)
        self.assertIn("--global_state_extrapolation_gammas 0.25\\,0.5",
                      result.stdout)

    def test_manual_replay_script_forwards_amm_proximal_start(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "SCHEME": "amm",
                "ACCELERATED": "true",
                "AMM_PROXIMAL_START": "false",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--amm_proximal_start false", result.stdout)

    def test_manual_replay_script_forwards_amm_cooldown(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "SCHEME": "amm",
                "ACCELERATED": "true",
                "AMM_COOLDOWN_AFTER_REJECTED": "2",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--amm_cooldown_after_rejected 2", result.stdout)

    def test_manual_replay_script_forwards_amm_local_merit_tie_tolerance(self):
        script = ROOT / "scripts" / "run_manual_dpgo_mm_six.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "CSAIL",
                "SCHEME": "amm",
                "ACCELERATED": "true",
                "AMM_LOCAL_MERIT_COST_TIE_TOLERANCE": "0.0001",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("--amm_local_merit_cost_tie_tolerance 0.0001",
                      result.stdout)

    def test_dran_six_script_dry_run_filters_datasets(self):
        script = ROOT / "scripts" / "run_six_dataset_experiments.sh"
        result = subprocess.run(
            [
                str(script),
            ],
            cwd=ROOT,
            env={
                **os.environ,
                "DRY_RUN": "true",
                "DATASETS": "sphere,torus",
                "MAX_ITERS": "3",
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

        self.assertIn("DRY_RUN dataset=sphere path=data/sphere2500.g2o", result.stdout)
        self.assertIn("DRY_RUN dataset=torus path=data/torus3D.g2o", result.stdout)
        self.assertNotIn("parking-garage", result.stdout)

    def test_ablation_contribution_analyzer_computes_positive_degradation_share(self):
        script = ROOT / "scripts" / "analyze_dpgo_mm_ablation_contributions.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = {
                "baseline": ("100", "10", "100", "1.0", "0.5", "5.0"),
                "no_amm": ("110", "12", "100", "1.0", "0.6", "5.5"),
                "weak_init": ("130", "18", "100", "1.0", "0.7", "5.6"),
                "no_preconditioner": ("95", "9", "100", "1.0", "0.55", "5.4"),
            }
            for ablation, values in rows.items():
                path = root / ablation
                path.mkdir()
                with (path / "summary.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([
                        "dataset", "method", "status", "scheme", "accelerated",
                        "dist_init", "preconditioner", "local_max_iterations",
                        "local_max_iterations_accepted", "local_max_tcg_iterations",
                        "init_red_rot_iters", "init_rot_iters",
                        "init_red_trans_iters", "init_trans_iters", "global_cost",
                        "gradient", "total_comm_poses", "total_comm_mb",
                        "solver_time_per_node_sec", "wall_time_sec", "output_dir",
                        "result_path", "estimate_path", "log_path",
                        "final_objective", "final_gradient", "iter_summary_path",
                    ])
                    writer.writerow([
                        "sphere", ablation, "ok", "amm", "true", "true",
                        "regularized_cholesky", "10", "1", "10000",
                        "100", "400", "150", "250", *values, "", "", "",
                        "", "", "", "",
                    ])

            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--root",
                    str(root),
                    "--share-ablations",
                    "no_amm,weak_init,no_preconditioner",
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            with (root / "ablation_contributions.csv").open(newline="") as stream:
                by_ablation = {row["ablation"]: row for row in csv.DictReader(stream)}

            self.assertAlmostEqual(float(by_ablation["no_amm"]["contribution_share"]), 0.25)
            self.assertAlmostEqual(float(by_ablation["weak_init"]["contribution_share"]), 0.75)
            self.assertAlmostEqual(float(by_ablation["no_preconditioner"]["contribution_share"]), 0.0)
            self.assertIn("not confirmed", by_ablation["no_preconditioner"]["note"])

    def test_amm_trace_comparator_uses_manual_surrogate_fields(self):
        script = ROOT / "scripts" / "compare_amm_traces.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dpgo = root / "dpgo.csv"
            manual = root / "manual.csv"
            out_dir = root / "out"
            with dpgo.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "iter", "robot", "local_iter", "fobj", "F0", "F1",
                    "gamma", "Gkh_initial", "minG",
                    "Gk_after_accelerated", "Gkh_after_restart_check",
                    "final_Gk", "phi_lhs", "phi_rhs", "refined",
                    "prox_reset", "hard_restart", "soft_restart",
                    "restart_used_xakh", "phi_fallback",
                    "soft_restart_hits0", "soft_restart_hits1",
                    "num_oscillations",
                ])
                writer.writeheader()
                writer.writerow({
                    "iter": "2", "robot": "1", "local_iter": "1",
                    "fobj": "10", "F0": "10", "F1": "10", "gamma": "0.2",
                    "Gkh_initial": "9.5", "minG": "9.4",
                    "Gk_after_accelerated": "9.2",
                    "Gkh_after_restart_check": "9.5", "final_Gk": "9.1",
                    "phi_lhs": "0.9", "phi_rhs": "0.01", "refined": "1",
                    "prox_reset": "0", "hard_restart": "0",
                    "soft_restart": "0", "restart_used_xakh": "0",
                    "phi_fallback": "0", "soft_restart_hits0": "2",
                    "soft_restart_hits1": "3", "num_oscillations": "1",
                })
            with manual.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "iter", "robot", "local_iter", "fobj", "F0", "F1",
                    "gamma", "Gkh_initial", "surrogate_Gkh_initial",
                    "minG", "Gk_after_accelerated",
                    "surrogate_Gk_after_accelerated",
                    "Gkh_after_restart_check",
                    "surrogate_Gkh_after_restart_check",
                    "final_Gk", "surrogate_final_Gk", "phi_lhs",
                    "phi_rhs", "refined", "prox_reset", "hard_restart",
                    "soft_restart", "restart_used_xakh", "phi_fallback",
                    "soft_restart_hits0", "soft_restart_hits1",
                    "num_oscillations",
                ])
                writer.writeheader()
                writer.writerow({
                    "iter": "2", "robot": "1", "local_iter": "1",
                    "fobj": "10", "F0": "10", "F1": "10", "gamma": "0.2",
                    "Gkh_initial": "19.5", "surrogate_Gkh_initial": "9.6",
                    "minG": "9.4", "Gk_after_accelerated": "19.2",
                    "surrogate_Gk_after_accelerated": "9.4",
                    "Gkh_after_restart_check": "19.5",
                    "surrogate_Gkh_after_restart_check": "9.7",
                    "final_Gk": "19.1", "surrogate_final_Gk": "9.0",
                    "phi_lhs": "0.8", "phi_rhs": "0.02", "refined": "0",
                    "prox_reset": "0", "hard_restart": "0",
                    "soft_restart": "0", "restart_used_xakh": "0",
                    "phi_fallback": "0", "soft_restart_hits0": "2",
                    "soft_restart_hits1": "3", "num_oscillations": "1",
                })

            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--dpgo-trace",
                    str(dpgo),
                    "--manual-trace",
                    str(manual),
                    "--out-dir",
                    str(out_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            self.assertIn("Wrote", result.stdout)
            pairwise = list(csv.DictReader((out_dir / "trace_pairwise.csv").open()))
            gk = next(row for row in pairwise
                      if row["field"] == "Gk_after_accelerated")
            self.assertEqual(gk["manual_field"], "surrogate_Gk_after_accelerated")
            self.assertAlmostEqual(float(gk["delta_manual_minus_dpgo"]), 0.2)
            refined = next(row for row in pairwise
                           if row["field"] == "refined")
            self.assertEqual(refined["bool_mismatch"], "1")
            iter_sums = list(csv.DictReader((out_dir / "iter_sums.csv").open()))
            self.assertEqual(iter_sums[0]["iter"], "2")
            self.assertAlmostEqual(
                float(iter_sums[0]["delta_manual_minus_dpgo_fobj_sum"]), 0.0)
            self.assertAlmostEqual(
                float(iter_sums[0][
                    "delta_manual_minus_dpgo_common_final_Gk_sum"
                ]),
                -0.1,
            )
            summary = (out_dir / "summary.md").read_text()
            self.assertIn("AMM Trace Parity Report", summary)
            self.assertIn("Iteration Sum Checks", summary)

    def test_collect_feh_preconditioner_ablation_exports_translation_initial_guess(self):
        script = ROOT / "scripts" / "collect_feh_preconditioner_ablation.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_root = root / "variant"
            run_root.mkdir()
            iter_path = run_root / "iteration_summary.csv"
            with iter_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "iter",
                    "full_equiv_hybrid_translation_recovery_initial_guess_candidate_count",
                    "full_equiv_hybrid_translation_recovery_initial_guess_used_count",
                    "full_equiv_hybrid_translation_recovery_initial_guess_rejected_count",
                    "local_gradient_compact_schur_candidate_count",
                    "local_gradient_compact_schur_accepted_count",
                    "local_gradient_compact_schur_guard_rejected_count",
                    "local_gradient_compact_schur_gradient_guard_rejected_count",
                    "local_gradient_compact_schur_gradient_change_sum",
                ])
                writer.writeheader()
                writer.writerow({
                    "iter": "0",
                    "full_equiv_hybrid_translation_recovery_initial_guess_candidate_count": "0",
                    "full_equiv_hybrid_translation_recovery_initial_guess_used_count": "0",
                    "full_equiv_hybrid_translation_recovery_initial_guess_rejected_count": "0",
                    "local_gradient_compact_schur_candidate_count": "0",
                    "local_gradient_compact_schur_accepted_count": "0",
                    "local_gradient_compact_schur_guard_rejected_count": "0",
                    "local_gradient_compact_schur_gradient_guard_rejected_count": "0",
                    "local_gradient_compact_schur_gradient_change_sum": "0",
                })
                writer.writerow({
                    "iter": "1",
                    "full_equiv_hybrid_translation_recovery_initial_guess_candidate_count": "5",
                    "full_equiv_hybrid_translation_recovery_initial_guess_used_count": "4",
                    "full_equiv_hybrid_translation_recovery_initial_guess_rejected_count": "1",
                    "local_gradient_compact_schur_candidate_count": "7",
                    "local_gradient_compact_schur_accepted_count": "6",
                    "local_gradient_compact_schur_guard_rejected_count": "1",
                    "local_gradient_compact_schur_gradient_guard_rejected_count": "2",
                    "local_gradient_compact_schur_gradient_change_sum": "0.25",
                })

            with (run_root / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "dataset", "status", "global_cost", "gradient",
                    "total_comm_poses", "total_comm_mb",
                    "solver_time_per_node_sec", "wall_time_sec",
                    "iter_summary_path",
                ])
                writer.writeheader()
                writer.writerow({
                    "dataset": "sphere",
                    "status": "ok",
                    "global_cost": "1.0",
                    "gradient": "2.0",
                    "total_comm_poses": "3",
                    "total_comm_mb": "4.0",
                    "solver_time_per_node_sec": "5.0",
                    "wall_time_sec": "6.0",
                    "iter_summary_path": str(iter_path),
                })

            output = root / "collected.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    f"test_variant={run_root}",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = list(csv.DictReader(output.open()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0][
                    "full_equiv_hybrid_translation_recovery_initial_guess_candidate_count"
                ],
                "5",
            )
            self.assertEqual(
                rows[0][
                    "full_equiv_hybrid_translation_recovery_initial_guess_used_count"
                ],
                "4",
            )
            self.assertEqual(
                rows[0][
                    "full_equiv_hybrid_translation_recovery_initial_guess_rejected_count"
                ],
                "1",
            )
            self.assertEqual(
                rows[0]["local_gradient_compact_schur_candidate_count"],
                "7",
            )
            self.assertEqual(
                rows[0]["local_gradient_compact_schur_accepted_count"],
                "6",
            )
            self.assertEqual(
                rows[0]["local_gradient_compact_schur_guard_rejected_count"],
                "1",
            )
            self.assertEqual(
                rows[0][
                    "local_gradient_compact_schur_gradient_guard_rejected_count"
                ],
                "2",
            )
            self.assertEqual(
                rows[0]["local_gradient_compact_schur_gradient_change_sum"],
                "0.25",
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class ObjectInitTopologyOverrideSmokeTest(unittest.TestCase):
    def test_object_drone_runner_dry_run_prints_reproducible_command(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "run_object_drone_experiment.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            marker = tmp / "executed"
            mock_bin = tmp / "object-based-multi-robot-example"
            mock_bin.write_text(
                "#!/usr/bin/env bash\n"
                f"touch {marker}\n"
                "exit 79\n")
            os.chmod(mock_bin, 0o755)

            result = subprocess.run(
                [str(script)],
                cwd=root,
                env={
                    **os.environ,
                    "BIN": str(mock_bin),
                    "DRY_RUN": "true",
                    "DATA_DIR": str(tmp / "data"),
                    "OUT_ROOT": str(tmp / "out"),
                    "NUM_ROBOTS": "21",
                    "NUM_OBJECTS": "0",
                    "MAX_ITERS": "3",
                    "OBJECT_TOPOLOGY": "ring",
                    "OBJECT_RING_HOPS": "2",
                    "OBJECT_INIT_MODE": "distributed_chordal_object_jacobi",
                    "OBJECT_INIT_TOPOLOGY": "ring",
                    "OBJECT_INIT_RING_HOPS": "1",
                    "OBJECT_INIT_OBJECT_JACOBI_ITERS": "2",
                    "OBJECT_CONSENSUS_MODE": "prox_mixing",
                    "OBJECT_CONSENSUS_ALPHA": "0.3",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            self.assertFalse(marker.exists())
            self.assertIn("DRY_RUN data_dir=", result.stdout)
            self.assertIn("out_root=", result.stdout)
            self.assertIn("object_topology=ring", result.stdout)
            self.assertIn("object_ring_hops=2", result.stdout)
            self.assertIn("object_init_mode=distributed_chordal_object_jacobi",
                          result.stdout)
            self.assertIn("object_init_topology=ring", result.stdout)
            self.assertIn("object_init_ring_hops=1", result.stdout)
            self.assertIn("object_consensus_mode=prox_mixing", result.stdout)
            self.assertIn("object_consensus_alpha=0.3", result.stdout)
            self.assertIn("command=", result.stdout)

    def test_manual_object_runner_dry_run_prints_reproducible_command(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "run_manual_object_dran_drone.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            marker = tmp / "executed"
            mock_bin = tmp / "manual-object-dran-example"
            mock_bin.write_text(
                "#!/usr/bin/env bash\n"
                f"touch {marker}\n"
                "exit 80\n")
            os.chmod(mock_bin, 0o755)

            result = subprocess.run(
                [str(script)],
                cwd=root,
                env={
                    **os.environ,
                    "BIN": str(mock_bin),
                    "DRY_RUN": "true",
                    "DATA_DIR": str(tmp / "data"),
                    "OUT_ROOT": str(tmp / "out"),
                    "MAX_ITERS": "2",
                    "OBJECT_TOPOLOGY": "ring",
                    "OBJECT_RING_HOPS": "1",
                    "OBJECT_COMMUNICATION_POLICY": "triggered",
                    "LOCAL_SOLVER": "reduced_rotation",
                    "REDUCED_ROTATION_PRECONDITIONER": "portfolio",
                    "GT_DIR": str(tmp / "gt"),
                    "COMM_TOPOLOGY_FILE": str(tmp / "topology_edges.csv"),
                    "COMM_WEIGHT_MODE": "matrix",
                    "OBJECT_INTERFACE_RESPONSE_TRIGGER": "predicted_decrease",
                    "OBJECT_INTERFACE_MIN_PREDICTED_DECREASE": "0.25",
                    "OBJECT_INTERFACE_MIN_INNOVATION_SCORE": "10",
                    "OBJECT_INITIALIZATION_ANCHOR_MODE": "information",
                    "OBJECT_INITIALIZATION_CONSENSUS_ROUNDS": "3",
                    "OBJECT_INITIALIZATION_OBSERVED_ANCHOR_WEIGHT": "2.5",
                    "OBJECT_INITIALIZATION_RELAY_ANCHOR_WEIGHT": "0.25",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            self.assertFalse(marker.exists())
            self.assertIn("DRY_RUN data_dir=", result.stdout)
            self.assertIn("out_root=", result.stdout)
            self.assertIn("topology=ring", result.stdout)
            self.assertIn("communication_policy=triggered", result.stdout)
            self.assertIn("local_solver=reduced_rotation", result.stdout)
            self.assertIn("reduced_rotation_preconditioner=portfolio",
                          result.stdout)
            self.assertIn("gt_dir=", result.stdout)
            self.assertIn("comm_topology_file=", result.stdout)
            self.assertIn("comm_weight_mode=matrix", result.stdout)
            self.assertIn("object_interface_response_trigger=predicted_decrease",
                          result.stdout)
            self.assertIn("object_interface_min_predicted_decrease=0.25",
                          result.stdout)
            self.assertIn("object_interface_min_innovation_score=10",
                          result.stdout)
            self.assertIn("object_initialization_anchor_mode=information",
                          result.stdout)
            self.assertIn("object_initialization_consensus_rounds=3",
                          result.stdout)
            self.assertIn("object_initialization_observed_anchor_weight=2.5",
                          result.stdout)
            self.assertIn("object_initialization_relay_anchor_weight=0.25",
                          result.stdout)
            self.assertIn("--topology_file", result.stdout)
            self.assertIn("--topology_weight_mode", result.stdout)
            self.assertIn("--object_interface_response_trigger", result.stdout)
            self.assertIn("--object_interface_min_predicted_decrease",
                          result.stdout)
            self.assertIn("--object_interface_min_innovation_score",
                          result.stdout)
            self.assertIn("--object_initialization_anchor_mode",
                          result.stdout)
            self.assertIn("--object_initialization_consensus_rounds",
                          result.stdout)
            self.assertIn("--object_initialization_observed_anchor_weight",
                          result.stdout)
            self.assertIn("--object_initialization_relay_anchor_weight",
                          result.stdout)
            self.assertIn("command=", result.stdout)

    def test_manual_object_runner_writes_gt_eval_when_gt_dir_is_set(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "run_manual_object_dran_drone.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out = tmp / "out"
            gt = tmp / "gt"
            gt.mkdir()
            (gt / "rob_0_gt.g2o").write_text(
                "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\n")
            mock_bin = tmp / "manual-object-dran-example"
            mock_bin.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "out_dir=''\n"
                "while [[ $# -gt 0 ]]; do\n"
                "  case \"$1\" in\n"
                "    --output_dir) out_dir=\"$2\"; shift 2 ;;\n"
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                "mkdir -p \"$out_dir\"\n"
                "cat > \"$out_dir/object_poses.txt\" <<'EOF'\n"
                "type robot_id local_vertex_id object_id x y z qx qy qz qw\n"
                "trajectory 0 0 -1 0 0 0 0 0 0 1\n"
                "EOF\n")
            os.chmod(mock_bin, 0o755)

            result = subprocess.run(
                [str(script)],
                cwd=root,
                env={
                    **os.environ,
                    "BIN": str(mock_bin),
                    "DATA_DIR": str(tmp / "data"),
                    "OUT_ROOT": str(out),
                    "GT_DIR": str(gt),
                    "NUM_ROBOTS": "1",
                    "NUM_OBJECTS": "0",
                    "MAX_ITERS": "0",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            gt_eval = out / "gt_eval.json"
            self.assertTrue(gt_eval.exists(), result.stdout)
            self.assertIn("Saved GT evaluation", result.stdout)
            self.assertIn('"trajectory_count": 1', gt_eval.read_text())

    def test_ring_initialization_topology_can_override_random_outer_topology(self):
        root = Path(__file__).resolve().parents[1]
        binary = root / "build" / "bin" / "object-based-multi-robot-example"
        data_dir = (
            root
            / "data"
            / "chordal_dataset"
            / "drone_g2o_file"
            / "range_10_tau_0.01_ang_error_1"
        )
        random_topology = (
            root
            / "results"
            / "topologies"
            / "drone_random_tv_p02_r5"
            / "topology_edges.csv"
        )
        if not binary.exists():
            self.skipTest(f"missing binary: {binary}")
        if not data_dir.exists():
            self.skipTest(f"missing data dir: {data_dir}")
        if not random_topology.exists():
            self.skipTest(f"missing topology: {random_topology}")

        env = os.environ.copy()
        env["OBJECT_INIT_OBJECT_JACOBI_ITERS"] = "2"
        env["OBJECT_INIT_OBJECT_JACOBI_SCOPE"] = "all"
        env["OBJECT_INIT_TOPOLOGY"] = "ring"
        env["OBJECT_INIT_RING_HOPS"] = "1"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "iters.csv"
            pose_path = tmp / "poses.txt"
            init_summary_path = tmp / "init_summary.csv"
            env["DPGO_OBJECT_INIT_SUMMARY_OUTPUT"] = str(init_summary_path)
            subprocess.run(
                [
                    str(binary),
                    str(data_dir),
                    "21",
                    "0",
                    "1",
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
                    "distributed_chordal_object_jacobi",
                    str(random_topology),
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = list(csv.DictReader(init_summary_path.open()))
            iter_rows = list(csv.DictReader(csv_path.open()))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["init_topology_source"], "default")
        self.assertEqual(row["init_topology"], "ring")
        self.assertEqual(row["init_ring_hops"], "1")
        self.assertEqual(row["object_jacobi_active_directed_pairs"], "2520")
        self.assertGreaterEqual(len(iter_rows), 1)

    def test_outer_consensus_cost_stop_terminates_before_max_iters(self):
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
        env["OBJECT_INIT_OBJECT_JACOBI_ITERS"] = "1"
        env["OBJECT_INIT_OBJECT_JACOBI_SCOPE"] = "all"
        env["OBJECT_OUTER_STOP_CONSENSUS_COST"] = "1000000000"
        env["OBJECT_OUTER_STOP_MIN_ITERS"] = "1"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "iters.csv"
            pose_path = tmp / "poses.txt"
            init_summary_path = tmp / "init_summary.csv"
            env["DPGO_OBJECT_INIT_SUMMARY_OUTPUT"] = str(init_summary_path)
            result = subprocess.run(
                [
                    str(binary),
                    str(data_dir),
                    "21",
                    "0",
                    "3",
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
                    "distributed_chordal_object_jacobi",
                ],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            iter_rows = list(csv.DictReader(csv_path.open()))

        self.assertEqual(len(iter_rows), 1, result.stdout)
        self.assertIn("Stopping outer iterations after 1 iterations",
                      result.stdout)


if __name__ == "__main__":
    unittest.main()

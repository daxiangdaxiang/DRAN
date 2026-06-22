#!/usr/bin/env python3
import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DranObjectFairComparisonTest(unittest.TestCase):
    def test_fair_runner_dry_run_prints_track_and_common_conditions(self):
        script = ROOT / "scripts" / "run_dran_object_fair_compare.sh"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            old_marker = tmp / "old_executed"
            manual_marker = tmp / "manual_executed"
            old_bin = tmp / "old"
            manual_bin = tmp / "manual"
            old_bin.write_text(f"#!/usr/bin/env bash\ntouch {old_marker}\n")
            manual_bin.write_text(f"#!/usr/bin/env bash\ntouch {manual_marker}\n")
            os.chmod(old_bin, 0o755)
            os.chmod(manual_bin, 0o755)

            result = subprocess.run(
                [str(script)],
                cwd=ROOT,
                env={
                    **os.environ,
                    "DRY_RUN": "true",
                    "TRACK": "optimizer_parity",
                    "OLD_BIN": str(old_bin),
                    "MANUAL_BIN": str(manual_bin),
                    "OUT_ROOT": str(tmp / "out"),
                    "DATA_DIR": str(tmp / "data"),
                    "GT_DIR": str(tmp / "gt"),
                    "MAX_ITERS": "20",
                    "OBJECT_TOPOLOGY": "ring",
                    "OBJECT_RING_HOPS": "1",
                    "OBJECT_POSE_TOL": "0.001",
                    "OBJECT_MAX_AGE": "5",
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            self.assertFalse(old_marker.exists())
            self.assertFalse(manual_marker.exists())
            self.assertIn("FAIR_COMPARE track=optimizer_parity",
                          result.stdout)
            self.assertIn("metric=measurement_cost", result.stdout)
            self.assertIn("data_dir=", result.stdout)
            self.assertIn("gt_dir=", result.stdout)
            self.assertIn("topology=ring", result.stdout)
            self.assertIn("ring_hops=1", result.stdout)
            self.assertIn("object_pose_tol=0.001", result.stdout)
            self.assertIn("object_max_age=5", result.stdout)
            self.assertIn("METHOD old_dran_object_optimizer_parity",
                          result.stdout)
            self.assertIn("METHOD manual_object_reduced_optimizer_parity",
                          result.stdout)
            self.assertIn("OBJECT_INIT_MODE=centralized_chordal",
                          result.stdout)
            self.assertIn("OBJECT_INITIALIZATION=centralized_chordal",
                          result.stdout)

    def test_summary_normalizes_old_and_manual_result_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            old = tmp / "old"
            manual = tmp / "manual"
            old.mkdir()
            manual.mkdir()
            with (old / "iterations.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "iter",
                        "measurement_cost",
                        "total_comm_mb",
                        "time",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "iter": "0",
                    "measurement_cost": "0.2",
                    "total_comm_mb": "0.1",
                    "time": "1.0",
                })
                writer.writerow({
                    "iter": "19",
                    "measurement_cost": "0.114356",
                    "total_comm_mb": "0.938049",
                    "time": "8.926",
                })
            with (manual / "iteration_summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "iter",
                        "measurement_cost",
                        "cumulative_comm_mb",
                        "time",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "iter": "0",
                    "measurement_cost": "0.3",
                    "cumulative_comm_mb": "0.2",
                    "time": "0.0",
                })
                writer.writerow({
                    "iter": "20",
                    "measurement_cost": "0.1140237283",
                    "cumulative_comm_mb": "1.018433",
                    "time": "3.4321",
                })
            for path, traj_rmse, object_rmse in [
                (old, 0.129306, 0.105688),
                (manual, 0.160050, 0.133332),
            ]:
                (path / "gt_eval.json").write_text(json.dumps({
                    "trajectory_translation_rmse": traj_rmse,
                    "trajectory_rotation_rmse_deg": 1.0,
                    "object_mean_translation_rmse": object_rmse,
                    "object_mean_rotation_rmse_deg": 0.4,
                }))
            manifest = tmp / "manifest.csv"
            with manifest.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["method", "track", "result_dir"],
                )
                writer.writeheader()
                writer.writerow({
                    "method": "old_dran_object",
                    "track": "optimizer_parity",
                    "result_dir": str(old),
                })
                writer.writerow({
                    "method": "manual_object_reduced",
                    "track": "optimizer_parity",
                    "result_dir": str(manual),
                })
            out = tmp / "summary.csv"

            subprocess.run(
                [
                    "python",
                    "scripts/summarize_dran_object_fair.py",
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )

            rows = list(csv.DictReader(out.open()))
            self.assertEqual([row["method"] for row in rows],
                             ["old_dran_object",
                              "manual_object_reduced"])
            self.assertEqual(rows[0]["final_measurement_cost"], "0.114356")
            self.assertEqual(rows[0]["final_comm_mb"], "0.938049")
            self.assertEqual(rows[1]["final_measurement_cost"], "0.1140237283")
            self.assertEqual(rows[1]["final_comm_mb"], "1.018433")
            self.assertEqual(rows[0]["trajectory_translation_rmse"],
                             "0.129306")
            self.assertEqual(rows[1]["object_mean_translation_rmse"],
                             "0.133332")


if __name__ == "__main__":
    unittest.main()

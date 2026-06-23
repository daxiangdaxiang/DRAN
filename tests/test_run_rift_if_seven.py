import csv
import importlib.util
import os
from pathlib import Path


def load_runner_module():
  repo_root = Path(__file__).resolve().parents[1]
  module_path = repo_root / "scripts" / "run_rift_if_seven.py"
  spec = importlib.util.spec_from_file_location("run_rift_if_seven", module_path)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def test_rift_if_seven_defaults_to_pr14_methods_and_datasets():
  module = load_runner_module()

  args = module.parse_args([])

  assert module.methods(args) == [
      "ted_cci_rift_if",
      "ted_cci_sr_direct",
      "dpcg_cci",
  ]
  assert args.interface_backend == "rift_auto"
  assert args.use_rotation_multi_rhs == "true"
  datasets = dict(module.dataset_specs(args))
  assert datasets["ais2klinik"] == Path("data/ais2klinik.g2o")


def test_rift_if_seven_writes_pr14_aliases_and_guarded_command(tmp_path):
  module = load_runner_module()
  fake_bench = tmp_path / "fake_bench.py"
  fake_bench.write_text(
      "#!/usr/bin/env python3\n"
      "print('BENCH_DCCI dataset=tiny g2o=data/tinyGrid3D.g2o dimension=2 "
      "poses=8 edges=9 robots=2 initialization_mode=ted_cci_rift_if "
      "cci_cost=1 method_cost=1 cost_abs_gap=0 cost_rel_gap=0 "
      "method_relative_pose_matrix_diff=0 method_comm_rounds=4 "
      "method_comm_mb=0.01 centralized_ms=1 method_ms=2 "
      "rift_selected_backend=rift_exact rift_directed_messages_sent=4 "
      "rift_actual_message_bytes=128 rift_symbolic_ms=0.1 "
      "rift_message_qr_ms=0.2 rift_belief_solve_ms=0.3 "
      "rift_final_interface_residual=1e-12 rift_used_global_matrix=0 "
      "rift_used_direct_solver=0 rift_used_collective=0')\n",
      encoding="utf-8")
  os.chmod(fake_bench, 0o755)
  output_dir = tmp_path / "out"

  rc = module.main([
      "--bench-bin", str(fake_bench),
      "--output-dir", str(output_dir),
      "--dataset", "tiny=data/tinyGrid3D.g2o",
      "--methods", "ted_cci_rift_if",
  ])

  assert rc == 0
  command = (output_dir / "commands.txt").read_text(encoding="utf-8")
  assert "--interface-backend rift_auto" in command
  assert "--use-rotation-multi-rhs true" in command
  assert "--forbid-direct-interface-solver true" in command
  assert "--forbid-global-interface-matrix true" in command
  assert "--forbid-collectives true" in command
  with (output_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
  assert len(rows) == 1
  row = rows[0]
  assert row["status"] == "ok"
  assert row["dim"] == "2"
  assert row["selected_backend"] == "rift_exact"
  assert row["rift_cost"] == "1"
  assert row["pose_diff"] == "0"
  assert row["directed_messages"] == "4"
  assert row["actual_message_bytes"] == "128"
  assert row["used_global_matrix"] == "0"
  assert row["used_direct_solver"] == "0"
  assert row["used_collective"] == "0"

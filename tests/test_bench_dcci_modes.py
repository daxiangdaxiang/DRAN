from pathlib import Path
import subprocess


def parse_bench_line(output: str) -> dict[str, str]:
  for line in output.splitlines():
    if not line.startswith("BENCH_DCCI"):
      continue
    fields = {}
    for token in line.split()[1:]:
      if "=" in token:
        key, value = token.split("=", 1)
        fields[key] = value
    return fields
  raise AssertionError(f"missing BENCH_DCCI line in output:\n{output}")


def test_bench_dcci_reports_ted_direct_mode_metrics():
  bench = Path("build/bin/bench-dcci")
  if not bench.exists():
    raise AssertionError("build/bin/bench-dcci must be built before this test")

  proc = subprocess.run([
      str(bench),
      "--dimension", "2",
      "--poses", "8",
      "--robots", "4",
      "--initialization-mode", "ted_cci_sr_direct",
  ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  row = parse_bench_line(proc.stdout)
  assert row["initialization_mode"] == "ted_cci_sr_direct"
  assert float(row["method_ms"]) >= 0.0
  assert float(row["method_cost_abs_gap"]) < 1e-8
  assert float(row["method_relative_pose_matrix_diff"]) < 1e-8
  assert float(row["method_rotation_equivalence_error"]) < 1e-8
  assert float(row["method_translation_equivalence_error"]) < 1e-8
  assert int(row["ted_factor_messages"]) > 0
  assert int(row["ted_solution_messages"]) > 0
  assert float(row["ted_local_qr_ms"]) >= 0.0
  assert float(row["ted_interface_solve_ms"]) >= 0.0
  assert float(row["ted_comm_mb"]) > 0.0


def test_bench_dcci_compare_all_includes_dpcg_and_ted_modes():
  bench = Path("build/bin/bench-dcci")
  if not bench.exists():
    raise AssertionError("build/bin/bench-dcci must be built before this test")

  proc = subprocess.run([
      str(bench),
      "--dimension", "2",
      "--poses", "8",
      "--robots", "4",
      "--compare-all",
  ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  modes = {
      parse_bench_line(line)["initialization_mode"]
      for line in proc.stdout.splitlines()
      if line.startswith("BENCH_DCCI")
  }
  assert "dpcg_cci" in modes
  assert "ted_cci_sr_direct" in modes
  assert "ted_cci_sr_hierarchical" in modes
  assert "ted_cci_sr_auto" in modes


def test_bench_dcci_reports_ted_auto_backend_metrics():
  bench = Path("build/bin/bench-dcci")
  if not bench.exists():
    raise AssertionError("build/bin/bench-dcci must be built before this test")

  proc = subprocess.run([
      str(bench),
      "--dimension", "2",
      "--poses", "8",
      "--robots", "4",
      "--initialization-mode", "ted_cci_sr_auto",
  ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  row = parse_bench_line(proc.stdout)
  assert row["initialization_mode"] == "ted_cci_sr_auto"
  assert row["ted_condensation_backend"] in {
      "dense_householder",
      "sparse_spqr",
  }
  assert row["ted_interface_backend"] in {
      "dense_householder",
      "sparse_spqr",
  }
  assert int(row["ted_spqr_rank_deficient_fallbacks"]) == 0
  assert int(row["ted_peak_interface_cols"]) >= 0
  assert int(row["ted_factor_dense_bytes_upward"]) > 0
  assert int(row["ted_factor_sparse_triplet_bytes_upward"]) > 0
  assert int(row["ted_factor_sparse_triplet_nonzeros"]) > 0
  assert int(row["ted_bytes_upward"]) <= min(
      int(row["ted_factor_dense_bytes_upward"]),
      int(row["ted_factor_sparse_triplet_bytes_upward"]),
  )
  assert float(row["method_cost_abs_gap"]) < 1e-8
  assert float(row["method_relative_pose_matrix_diff"]) < 1e-8


def test_bench_dcci_reports_optional_nonlinear_refinement_cost():
  bench = Path("build/bin/bench-dcci")
  if not bench.exists():
    raise AssertionError("build/bin/bench-dcci must be built before this test")

  proc = subprocess.run([
      str(bench),
      "--dimension", "2",
      "--poses", "6",
      "--robots", "3",
      "--initialization-mode", "ted_cci_sr_direct",
      "--nonlinear-refinement-iters", "1",
  ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  row = parse_bench_line(proc.stdout)
  assert row["nonlinear_refinement_iters"] == "1"
  assert row["final_pgo_cost_after_nonlinear_refinement_available"] == "1"
  assert float(row["final_pgo_cost_after_nonlinear_refinement"]) >= 0.0
  assert float(row["nonlinear_refinement_ms"]) >= 0.0


def test_bench_dcci_reports_rift_pr14_fields_and_comm_rounds():
  bench = Path("build/bin/bench-dcci")
  if not bench.exists():
    raise AssertionError("build/bin/bench-dcci must be built before this test")

  proc = subprocess.run([
      str(bench),
      "--dimension", "2",
      "--poses", "8",
      "--robots", "4",
      "--initialization-mode", "ted_cci_rift_if",
      "--interface-backend", "rift_auto",
      "--use-rotation-multi-rhs", "true",
      "--forbid-direct-interface-solver", "true",
      "--forbid-global-interface-matrix", "true",
      "--forbid-collectives", "true",
  ], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  row = parse_bench_line(proc.stdout)
  assert row["initialization_mode"] == "ted_cci_rift_if"
  assert row["selected_backend"] in {
      "rift_exact",
      "rift_cak",
      "rift_async_schur",
  }
  assert row["rift_selected_backend"] == row["selected_backend"]
  assert row["dim"] == row["dimension"]
  assert row["rift_cost"] == row["method_cost"]
  assert row["pose_diff"] == row["method_relative_pose_matrix_diff"]
  assert row["directed_messages"] == row["rift_directed_messages_sent"]
  assert row["actual_message_bytes"] == row["rift_actual_message_bytes"]
  assert row["symbolic_ms"] == row["rift_symbolic_ms"]
  assert row["message_qr_ms"] == row["rift_message_qr_ms"]
  assert row["belief_solve_ms"] == row["rift_belief_solve_ms"]
  assert row["final_interface_residual"] == row["rift_final_interface_residual"]
  assert row["used_global_matrix"] == "0"
  assert row["used_direct_solver"] == "0"
  assert row["used_collective"] == "0"
  assert row["method_comm_rounds"] == row["directed_messages"]
  assert float(row["method_cost_abs_gap"]) < 1e-8
  assert float(row["method_relative_pose_matrix_diff"]) < 1e-8

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


FEH_ITER_COLUMNS = [
    "full_equiv_hybrid_linear_pcg_iteration_count",
    "full_equiv_hybrid_linear_initial_residual",
    "full_equiv_hybrid_linear_final_residual",
    "full_equiv_hybrid_linear_full_residual",
    "full_equiv_hybrid_linear_solve_time_sec",
    "full_equiv_hybrid_step_trial_count",
    "full_equiv_hybrid_step_trial_accepted_count",
    "full_equiv_hybrid_step_trial_rejected_count",
    "full_equiv_hybrid_step_predicted_decrease_sum",
    "full_equiv_hybrid_step_actual_decrease_sum",
    "full_equiv_hybrid_step_rho_sum",
    "full_equiv_hybrid_step_rho_count",
    "full_equiv_hybrid_step_accepted_scale_sum",
    "full_equiv_hybrid_step_accepted_predicted_decrease_sum",
    "full_equiv_hybrid_step_accepted_actual_decrease_sum",
    "full_equiv_hybrid_step_accepted_rho_sum",
    "full_equiv_hybrid_step_accepted_rho_count",
    "full_equiv_hybrid_sparse_matvec_count",
    "full_equiv_hybrid_translation_schur_preconditioner_application_count",
    "full_equiv_hybrid_translation_schur_preconditioner_factorization_count",
    "full_equiv_hybrid_translation_schur_preconditioner_fallback_count",
    "full_equiv_hybrid_local_chain_preconditioner_application_count",
    "full_equiv_hybrid_local_chain_preconditioner_factorization_count",
    "full_equiv_hybrid_local_chain_preconditioner_fallback_count",
    "full_equiv_hybrid_translation_block_preconditioner_application_count",
    "full_equiv_hybrid_translation_block_preconditioner_factorization_count",
    "full_equiv_hybrid_translation_block_preconditioner_fallback_count",
    "full_equiv_hybrid_translation_sparse_schur_preconditioner_application_count",
    "full_equiv_hybrid_translation_sparse_schur_preconditioner_factorization_count",
    "full_equiv_hybrid_translation_sparse_schur_preconditioner_fallback_count",
    "full_equiv_hybrid_translation_local_schur_preconditioner_application_count",
    "full_equiv_hybrid_translation_local_schur_preconditioner_factorization_count",
    "full_equiv_hybrid_translation_local_schur_preconditioner_fallback_count",
    "full_equiv_hybrid_translation_local_schur_preconditioner_active_pose_count",
    "full_equiv_hybrid_translation_local_schur_preconditioner_active_column_count",
    "full_equiv_hybrid_laplacian_deflation_preconditioner_application_count",
    "full_equiv_hybrid_laplacian_deflation_preconditioner_factorization_count",
    "full_equiv_hybrid_laplacian_deflation_preconditioner_fallback_count",
    "full_equiv_hybrid_laplacian_deflation_preconditioner_basis_dimension",
    "full_equiv_hybrid_reduced_rotation_preconditioner_application_count",
    "full_equiv_hybrid_reduced_rotation_preconditioner_factorization_count",
    "full_equiv_hybrid_rqn_preconditioner_application_count",
    "full_equiv_hybrid_reduced_rotation_initial_guess_candidate_count",
    "full_equiv_hybrid_reduced_rotation_initial_guess_used_count",
    "full_equiv_hybrid_reduced_rotation_initial_guess_rejected_count",
    "full_equiv_hybrid_translation_recovery_initial_guess_candidate_count",
    "full_equiv_hybrid_translation_recovery_initial_guess_used_count",
    "full_equiv_hybrid_translation_recovery_initial_guess_rejected_count",
    "full_equiv_hybrid_local_portfolio_candidate_count",
    "full_equiv_hybrid_local_portfolio_selected_unsmoothed_count",
    "full_equiv_hybrid_local_portfolio_selected_schwarz_only_count",
    "full_equiv_hybrid_local_portfolio_selected_schwarz_feh_count",
    "full_equiv_hybrid_schwarz_smoothing_sweep_count",
    "full_equiv_hybrid_schwarz_smoothing_candidate_count",
    "full_equiv_hybrid_schwarz_smoothing_accepted_count",
    "full_equiv_hybrid_schwarz_smoothing_rejected_count",
    "full_equiv_hybrid_schwarz_smoothing_cost_decrease",
    "full_equiv_hybrid_schwarz_smoothing_time_sec",
    "local_gradient_compact_schur_candidate_count",
    "local_gradient_compact_schur_accepted_count",
    "local_gradient_compact_schur_guard_rejected_count",
    "local_gradient_compact_schur_gradient_guard_rejected_count",
    "local_gradient_compact_schur_solve_failure_count",
    "local_gradient_compact_schur_boundary_col_count",
    "local_gradient_compact_schur_private_col_count",
    "local_gradient_compact_schur_gradient_change_sum",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return list(reader)


def parse_variant_root(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        variant, root = spec.split("=", 1)
        return variant, Path(root)
    root = Path(spec)
    return root.name, root


def final_iteration_row(path_value: str) -> dict[str, str]:
    if not path_value:
        return {}
    rows = read_csv_rows(Path(path_value))
    return rows[-1] if rows else {}


def collect_rows(variant_roots: list[tuple[str, Path]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for variant, root in variant_roots:
        summary_path = root / "summary.csv"
        for summary_row in read_csv_rows(summary_path):
            iter_row = final_iteration_row(summary_row.get("iter_summary_path", ""))
            row = {
                "variant": variant,
                "dataset": summary_row.get("dataset", ""),
                "status": summary_row.get("status", ""),
                "global_cost": summary_row.get("global_cost", ""),
                "gradient": summary_row.get("gradient", ""),
                "total_comm_poses": summary_row.get("total_comm_poses", ""),
                "total_comm_mb": summary_row.get("total_comm_mb", ""),
                "solver_time_per_node_sec": summary_row.get(
                    "solver_time_per_node_sec", ""
                ),
                "wall_time_sec": summary_row.get("wall_time_sec", ""),
                "result_path": summary_row.get("result_path", ""),
                "output_dir": summary_row.get("output_dir", ""),
                "iteration_summary_path": summary_row.get("iter_summary_path", ""),
                "log_path": summary_row.get("log_path", ""),
                "summary_path": str(summary_path),
            }
            for column in FEH_ITER_COLUMNS:
                row[column] = iter_row.get(column, "")
            rows.append(row)
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Collect FE-Hybrid preconditioner ablation summaries."
    )
    parser.add_argument(
        "variant_roots",
        nargs="+",
        help="Variant root specs as name=/path/to/run_root or bare run roots.",
    )
    parser.add_argument("-o", "--output", default="-", help="Output CSV path.")
    args = parser.parse_args(argv)

    variant_roots = [parse_variant_root(spec) for spec in args.variant_roots]
    rows = collect_rows(variant_roots)

    fieldnames = [
        "variant",
        "dataset",
        "status",
        "global_cost",
        "gradient",
        "total_comm_poses",
        "total_comm_mb",
        "solver_time_per_node_sec",
        "wall_time_sec",
        *FEH_ITER_COLUMNS,
        "result_path",
        "output_dir",
        "iteration_summary_path",
        "log_path",
        "summary_path",
    ]

    output = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    try:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if output is not sys.stdout:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

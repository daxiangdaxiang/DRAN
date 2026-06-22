#include <DPGO/ManualDpgoMm.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool parseBool(const std::string &value) {
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "on";
}

std::vector<double> parseDoubleList(const std::string &value) {
  std::vector<double> result;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t comma = value.find(',', start);
    const std::size_t end =
        comma == std::string::npos ? value.size() : comma;
    if (end > start) {
      result.push_back(std::stod(value.substr(start, end - start)));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return result;
}

DPGO::ManualDpgoMmPostExchangeDeltaMode parsePostExchangeDeltaMode(
    const std::string &value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (normalized == "absolute") {
    return DPGO::ManualDpgoMmPostExchangeDeltaMode::Absolute;
  }
  if (normalized == "weighted") {
    return DPGO::ManualDpgoMmPostExchangeDeltaMode::Weighted;
  }
  throw std::invalid_argument(
      "post_exchange_delta_mode must be absolute or weighted");
}

const char *postExchangeDeltaModeName(
    DPGO::ManualDpgoMmPostExchangeDeltaMode mode) {
  switch (mode) {
    case DPGO::ManualDpgoMmPostExchangeDeltaMode::Absolute:
      return "absolute";
    case DPGO::ManualDpgoMmPostExchangeDeltaMode::Weighted:
      return "weighted";
  }
  return "unknown";
}

DPGO::ManualDpgoMmCommunicationDeltaMode parseCommunicationDeltaMode(
    const std::string &value) {
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (normalized == "sparse") {
    return DPGO::ManualDpgoMmCommunicationDeltaMode::Sparse;
  }
  if (normalized == "tangent") {
    return DPGO::ManualDpgoMmCommunicationDeltaMode::Tangent;
  }
  if (normalized == "hybrid") {
    return DPGO::ManualDpgoMmCommunicationDeltaMode::Hybrid;
  }
  throw std::invalid_argument(
      "comm_topology_delta_mode must be sparse, tangent, or hybrid");
}

const char *communicationDeltaModeName(
    DPGO::ManualDpgoMmCommunicationDeltaMode mode) {
  switch (mode) {
    case DPGO::ManualDpgoMmCommunicationDeltaMode::Sparse:
      return "sparse";
    case DPGO::ManualDpgoMmCommunicationDeltaMode::Tangent:
      return "tangent";
    case DPGO::ManualDpgoMmCommunicationDeltaMode::Hybrid:
      return "hybrid";
  }
  return "unknown";
}

void printUsage(const char *program) {
  std::cout << "Usage: " << program
            << " --dataset path.g2o --num_nodes N [options]\n"
            << "Options:\n"
            << "  --iters N\n"
            << "  --scheme auto|mm|amm\n"
            << "  --accelerated true|false\n"
            << "  --local_solver manual_full|full_equiv_hybrid|reduced_rotation\n"
            << "  --reduced_rotation_preconditioner none|jacobi|schur_jacobi|cholesky|portfolio|adaptive_portfolio\n"
            << "    portfolio evaluates none|jacobi|schur_jacobi|cholesky; adaptive_portfolio uses Cholesky-anchored local fast-paths with portfolio recalibration\n"
            << "  --reduced_surrogate_mode true_local|dpgo_simple|edge_tight_quadratic\n"
            << "  --surrogate_mode legacy|weighted_edge_split|adaptive_spectral|variable_projected_schur\n"
            << "  --edge_split_theta_mode constant|degree|curvature|adaptive_conditioned\n"
            << "  --edge_split_theta_default value\n"
            << "  --edge_split_theta_min value\n"
            << "  --edge_split_theta_max value\n"
            << "  --edge_split_theta_candidates a,b,c\n"
            << "  --mm_accelerator none|nesterov_legacy|anderson|squarem\n"
            << "  --mm_safeguard local_surrogate|local_surrogate_plus_boundary|debug_global\n"
            << "  --debug_surrogate_bound_check true|false\n"
            << "  --debug_surrogate_bound_samples N\n"
            << "  --feh_linear_backend manual_full_rtr|sparse_direct_schur|pcg_schur|pcg_full\n"
            << "  --feh_schur_warm_start true|false\n"
            << "  --feh_final_polish_rtr true|false\n"
            << "  --feh_schur_damping value\n"
            << "  --feh_linear_rel_tol value\n"
            << "  --feh_linear_abs_tol value\n"
            << "  --feh_linear_max_iters N\n"
            << "  --feh_linear_block_jacobi true|false\n"
            << "  --feh_sparse_matvec true|false\n"
            << "  --feh_precond_translation_schur true|false\n"
            << "  --feh_precond_local_chain true|false\n"
            << "  --feh_precond_translation_block true|false\n"
            << "  --feh_precond_translation_sparse_schur true|false\n"
            << "  --feh_precond_translation_local_schur true|false\n"
            << "  --feh_precond_translation_local_schur_max_poses N\n"
            << "  --feh_precond_laplacian_deflation true|false\n"
            << "  --feh_precond_laplacian_deflation_basis_size N\n"
            << "  --feh_precond_laplacian_deflation_max_eigen_poses N\n"
            << "  --feh_precond_reduced_rotation true|false\n"
            << "  --feh_reduced_rotation_initial_guess true|false\n"
            << "  --feh_translation_recovery_initial_guess true|false\n"
            << "  --feh_rqn_warm_start true|false\n"
            << "  --feh_precond_rqn_memory true|false\n"
            << "  --feh_rqn_memory_size N\n"
            << "  --feh_rqn_min_curvature_ratio value\n"
            << "  --feh_warm_start_max_norm value\n"
            << "  --feh_warm_start_backtracking_steps N\n"
            << "  --feh_select_best_backtracking true|false\n"
            << "  --feh_step_pareto_selector true|false\n"
            << "  --feh_step_pareto_min_decrease_ratio value\n"
            << "  --feh_backtracking_shared_pose_prox_weight value\n"
            << "  --feh_projected_model_rho_guard true|false\n"
            << "  --feh_projected_model_rho_eta value\n"
            << "  --feh_active_separator_correction true|false\n"
            << "  --feh_active_separator_step_cap value\n"
            << "  --feh_active_separator_block_jacobi true|false\n"
            << "  --feh_active_separator_compact_schur true|false\n"
            << "  --feh_active_separator_lm_schur true|false\n"
            << "  --feh_active_separator_lm_schur_max_boundary_poses N\n"
            << "  --feh_active_separator_lm_schur_max_private_cols N\n"
            << "  --feh_active_separator_lm_schur_damping value\n"
            << "  --feh_active_separator_lm_schur_backtracking_steps N\n"
            << "  --feh_active_separator_lm_schur_min_cost_improvement value\n"
            << "  --feh_active_separator_lm_schur_max_rounds N\n"
            << "  --feh_active_separator_lm_schur_score_mode gradient|model_decrease\n"
            << "  --feh_active_separator_lm_schur_gradient_guard true|false\n"
            << "  --feh_active_separator_lm_schur_max_gradient_increase_ratio value\n"
            << "  --feh_translation_recovery_polish true|false\n"
            << "  --feh_translation_recovery_polish_gradient_guard true|false\n"
            << "  --feh_translation_recovery_polish_max_gradient_increase_ratio value\n"
            << "  --feh_translation_recovery_polish_backtracking true|false\n"
            << "  --feh_translation_recovery_polish_backtracking_steps N\n"
            << "  --feh_translation_recovery_polish_merit_selector true|false\n"
            << "  --feh_translation_recovery_polish_merit_min_cost_recovery_ratio value\n"
            << "  --feh_translation_recovery_polish_selected_only true|false\n"
            << "  --feh_translation_recovery_step_trials true|false\n"
            << "  --feh_local_portfolio true|false\n"
            << "  --manual_full_portfolio true|false\n"
            << "  --feh_schwarz_presmoothing true|false\n"
            << "  --feh_schwarz_sweeps N\n"
            << "  --feh_schwarz_max_block_norm value\n"
            << "  --feh_schwarz_block_mode per_pose|edge_pair\n"
            << "  --local_max_iterations N\n"
            << "  --local_max_iterations_accepted N\n"
            << "  --local_max_tcg_iterations N\n"
            << "  --local_grad_norm_tol value\n"
            << "  --trust_region_initial_radius value\n"
            << "  --record_local_model_diagnostics true|false\n"
            << "  --parallel_local_solves true|false\n"
            << "  --adaptive_reduced_tcg true|false\n"
            << "  --adaptive_reduced_tcg_max_iterations N\n"
            << "  --adaptive_reduced_tcg_gradient_ratio value\n"
            << "  --reduced_rotation_tcg_relative_tolerance value\n"
            << "  --local_state_extrapolation true|false\n"
            << "  --local_state_extrapolation_gamma value\n"
            << "  --local_state_extrapolation_gammas a,b,c\n"
            << "  --local_state_extrapolation_use_active_surrogate true|false\n"
            << "  --local_boundary_proximal_candidate true|false\n"
            << "  --local_boundary_proximal_weight value\n"
            << "  --local_boundary_proximal_weights a,b,c\n"
            << "  --local_model_g_extrapolation true|false\n"
            << "  --local_model_g_extrapolation_gamma value\n"
            << "  --local_model_g_extrapolation_gammas a,b,c\n"
            << "  --coupled_state_g_extrapolation true|false\n"
            << "  --coupled_state_g_extrapolation_gamma value\n"
            << "  --coupled_state_g_extrapolation_gammas a,b,c\n"
            << "  --local_anderson_acceleration true|false\n"
            << "  --local_anderson_max_alpha value\n"
            << "  --local_squarem_max_alpha value\n"
            << "  --local_candidate_cost_tie_tolerance value\n"
            << "  --global_state_extrapolation true|false\n"
            << "  --global_state_extrapolation_gamma value\n"
            << "  --global_state_extrapolation_gammas a,b,c\n"
            << "  --global_anderson_acceleration true|false\n"
            << "  --global_anderson_max_alpha value\n"
            << "  --local_gradient_correction true|false\n"
            << "  --local_gradient_correction_step value\n"
            << "  --local_gradient_correction_steps a,b,c\n"
            << "  --local_gradient_correction_shared_step true|false\n"
            << "  --local_gradient_correction_neighborhood_step true|false\n"
            << "  --local_gradient_correction_curvature_step true|false\n"
            << "  --local_gradient_correction_coupled_direction true|false\n"
            << "  --local_gradient_correction_coupled_direction_budget_fraction value\n"
            << "  --local_gradient_correction_coupled_direction_max_packets_per_receiver N\n"
            << "  --local_gradient_correction_coupled_direction_topk_entries N\n"
            << "  --local_gradient_correction_coupled_direction_min_score value\n"
            << "  --local_gradient_correction_coupled_direction_min_score_ratio value\n"
            << "  --local_gradient_correction_coupled_direction_byte_budget_mb value\n"
            << "  --local_gradient_correction_boundary_only true|false\n"
            << "  --local_gradient_correction_block_jacobi true|false\n"
            << "  --local_gradient_correction_compact_schur true|false\n"
            << "  --local_gradient_correction_compact_schur_max_boundary_poses N\n"
            << "  --local_gradient_correction_compact_schur_max_private_cols N\n"
            << "  --local_gradient_correction_compact_schur_damping value\n"
            << "  --local_gradient_correction_compact_schur_min_cost_decrease value\n"
            << "  --local_gradient_correction_compact_schur_gradient_guard true|false\n"
            << "  --local_gradient_correction_compact_schur_max_gradient_increase_ratio value\n"
            << "  --local_gradient_correction_fresh_neighbor_exchange true|false\n"
            << "  --local_gradient_correction_fresh_min_pose_delta value\n"
            << "  --local_gradient_correction_fresh_delta_mode absolute|weighted\n"
            << "  --local_gradient_correction_fresh_budget_fraction value\n"
            << "  --local_gradient_correction_fresh_max_poses_per_receiver N\n"
            << "  --local_gradient_correction_inner_rounds N\n"
            << "  --local_gradient_correction_max_rounds N\n"
            << "  --post_exchange_period N\n"
            << "  --post_exchange_min_pose_delta value\n"
            << "  --post_exchange_delta_mode absolute|weighted\n"
            << "  --post_exchange_budget_fraction value\n"
            << "  --post_exchange_max_poses_per_receiver N\n"
            << "  --comm_topology_file path/to/topology_edges.csv\n"
            << "  --comm_topology_initial_full true|false\n"
            << "  --comm_topology_max_relay_hops N\n"
            << "  --comm_topology_hop_budget_fraction value\n"
            << "  --comm_topology_stale_aware_relay_score true|false\n"
            << "  --comm_topology_stale_aware_relay_age_gain value\n"
            << "  --comm_topology_delta_compression true|false\n"
            << "  --comm_topology_delta_mode sparse|tangent|hybrid\n"
            << "  --comm_topology_delta_topk N\n"
            << "  --comm_topology_delta_max_reconstruction_error value\n"
            << "  --comm_topology_lifted_delta_compression true|false\n"
            << "  --comm_topology_lifted_delta_max_reconstruction_error value\n"
            << "  --comm_topology_lifted_delta_rank N\n"
            << "  --comm_topology_value_scheduler true|false\n"
            << "  --comm_topology_value_scheduler_mode static_sensitivity_staleness|boundary_residual_staleness\n"
            << "  --comm_topology_value_byte_budget_mb value\n"
            << "  --comm_topology_value_min_score_ratio value\n"
            << "  --comm_topology_value_budget_pacing true|false\n"
            << "  --comm_topology_interface_model true|false\n"
            << "  --comm_topology_interface_model_payload direction_block|diag_stiffness|direction_block_diag_stiffness\n"
            << "  --comm_topology_interface_model_local_merit true|false\n"
            << "  --comm_topology_boundary_candidate true|false\n"
            << "  --comm_topology_boundary_surrogate true|false\n"
            << "  --comm_topology_boundary_surrogate_stale_gain value\n"
            << "  --comm_topology_reduced_interface_model true|false\n"
            << "  --comm_topology_reduced_interface_weight value\n"
            << "  --comm_topology_reduced_interface_max_local_iters N\n"
            << "  --comm_topology_reduced_interface_candidate true|false\n"
            << "  --stale_boundary_proximal true|false\n"
            << "  --stale_boundary_proximal_weight value\n"
            << "  --stale_boundary_prediction true|false\n"
            << "  --stale_boundary_prediction_gain value\n"
            << "  --sync_amm_reference_after_local_gradient_correction true|false\n"
            << "  --global_gradient_correction true|false\n"
            << "  --global_gradient_correction_step value\n"
            << "  --global_gradient_correction_steps a,b,c\n"
            << "  --reset_local_history_after_global_correction true|false\n"
            << "  --sync_amm_reference_after_global_correction true|false\n"
            << "  --amm_eta0 value\n"
            << "  --amm_eta1 value\n"
            << "  --amm_psi value\n"
            << "  --amm_phi value\n"
            << "  --amm_gamma_scale value\n"
            << "  --amm_gamma_scales a,b,c\n"
            << "  --amm_accepted_delta value\n"
            << "  --amm_oscillation_count_period N\n"
            << "  --amm_max_oscillations N\n"
            << "  --amm_proximal_start true|false\n"
            << "  --amm_dpgo_restart_fallback true|false\n"
            << "  --amm_dpgo_surrogate_parity true|false\n"
            << "  --amm_baseline_surrogate_state true|false\n"
            << "  --amm_dpgo_recursive_simple_state true|false\n"
            << "  --amm_recursive_simple_reanchor_period N\n"
            << "  --amm_dpgo_strict_refined_gate true|false\n"
            << "  --amm_dpgo_recover_translations_after_proximal true|false\n"
            << "  --amm_dpgo_refined_starts_at_recovered_proximal true|false\n"
            << "  --amm_dpgo_mixed_surrogate_portfolio true|false\n"
            << "  --amm_mixed_surrogate_skip_simple_after_true_local_streak N\n"
            << "  --amm_mixed_surrogate_force_simple_every_skipped_rounds N\n"
            << "  --amm_cooldown_after_rejected N\n"
            << "  --amm_prox_reset_skip_refined_solve true|false\n"
            << "  --amm_dpgo_proximal_fallback_only true|false\n"
            << "  --amm_lazy_plain_after_certificate true|false\n"
            << "  --amm_surrogate_first_exact_evaluation true|false\n"
            << "  --amm_max_soft_restart_hits0 N\n"
            << "  --amm_max_soft_restart_hits1 N\n"
            << "  --amm_local_merit_filter true|false\n"
            << "  --amm_local_merit_cost_tie_tolerance value\n"
            << "  --trace_amm true|false\n"
            << "  --profile_optimizer true|false\n"
            << "  --candidate_evaluation_cache true|false\n"
            << "  --candidate_object_reuse true|false\n"
            << "  --reduced_adaptive_portfolio_certified_fast_path true|false\n"
            << "  --reduced_rotation_direct_objective true|false\n"
            << "  --fused_candidate_evaluation true|false\n"
            << "  --lazy_candidate_gradient_evaluation true|false\n"
            << "  --lazy_solver_start_gradient_evaluation true|false\n"
            << "  --lazy_surrogate_candidate_gradient_evaluation true|false\n"
            << "  --reduced_rotation_curvature_cauchy_candidate true|false\n"
            << "  --reduced_rotation_curvature_fallback_candidate true|false\n"
            << "  --reduced_rotation_gradient_boundary_candidate true|false\n"
            << "  --reduced_rotation_surrogate_tcg_accept true|false\n"
            << "  --reduced_rotation_skip_redundant_candidate_projection true|false\n"
            << "  --print_iteration_summary true|false\n"
            << "  --output_dir path\n"
            << "  --external_initial_estimate path/to/manual_matrix.txt\n"
            << "  --dist_init true|false\n"
            << "  --dist_init_refinement_rounds N\n"
            << "  --dist_init_fixed_neighbor_chordal_refit true|false\n"
            << "  --save_iteration_estimates true|false\n"
            << "  --save true|false\n";
}

}  // namespace

int main(int argc, char **argv) {
  std::string dataset;
  std::string scheme = "mm";
  std::string localSolver = "reduced_rotation";
  bool accelerated = false;
  DPGO::ManualDpgoMmOptions options;

  try {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto requireValue = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--dataset") {
      dataset = requireValue(arg);
    } else if (arg == "--num_nodes" || arg == "--num_robots") {
      options.numRobots =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--iters") {
      options.maxIterations =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--scheme") {
      scheme = requireValue(arg);
    } else if (arg == "--accelerated") {
      accelerated = parseBool(requireValue(arg));
    } else if (arg == "--local_solver") {
      localSolver = requireValue(arg);
    } else if (arg == "--reduced_rotation_preconditioner") {
      options.reducedRotationPreconditioner =
          DPGO::parseManualDpgoMmReducedRotationPreconditioner(
              requireValue(arg));
    } else if (arg == "--reduced_surrogate_mode") {
      options.reducedSurrogateMode =
          DPGO::parseManualDpgoMmReducedSurrogateMode(requireValue(arg));
    } else if (arg == "--surrogate_mode") {
      options.surrogateMode =
          DPGO::parseManualDpgoMmSurrogateMode(requireValue(arg));
    } else if (arg == "--edge_split_theta_mode") {
      options.edgeSplitThetaMode =
          DPGO::parseManualDpgoMmEdgeSplitThetaMode(requireValue(arg));
    } else if (arg == "--edge_split_theta_default") {
      options.edgeSplitThetaDefault = std::stod(requireValue(arg));
    } else if (arg == "--edge_split_theta_min") {
      options.edgeSplitThetaMin = std::stod(requireValue(arg));
    } else if (arg == "--edge_split_theta_max") {
      options.edgeSplitThetaMax = std::stod(requireValue(arg));
    } else if (arg == "--edge_split_theta_candidates") {
      options.edgeSplitThetaCandidates = parseDoubleList(requireValue(arg));
    } else if (arg == "--mm_accelerator") {
      options.mmAcceleratorMode =
          DPGO::parseManualDpgoMmMmAcceleratorMode(requireValue(arg));
    } else if (arg == "--mm_safeguard") {
      options.mmSafeguard =
          DPGO::parseManualDpgoMmMmSafeguard(requireValue(arg));
    } else if (arg == "--debug_surrogate_bound_check") {
      options.debugSurrogateBoundCheck = parseBool(requireValue(arg));
    } else if (arg == "--debug_surrogate_bound_samples") {
      options.debugSurrogateBoundSamples =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_linear_backend") {
      options.fullEquivHybridBackend =
          DPGO::parseManualDpgoMmFullEquivHybridBackend(requireValue(arg));
    } else if (arg == "--feh_schur_warm_start") {
      options.fullEquivHybridSchurWarmStart = parseBool(requireValue(arg));
    } else if (arg == "--feh_final_polish_rtr") {
      options.fullEquivHybridFinalPolishRtr = parseBool(requireValue(arg));
    } else if (arg == "--feh_schur_damping") {
      options.fullEquivHybridSchurDamping = std::stod(requireValue(arg));
    } else if (arg == "--feh_linear_rel_tol") {
      options.fullEquivHybridLinearRelativeTolerance =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_linear_abs_tol") {
      options.fullEquivHybridLinearAbsoluteTolerance =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_linear_max_iters") {
      options.fullEquivHybridLinearMaxIterations =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_linear_block_jacobi") {
      options.fullEquivHybridLinearBlockJacobiPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_sparse_matvec") {
      options.fullEquivHybridSparseMatrixVectorProduct =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_precond_translation_schur") {
      options.fullEquivHybridLinearTranslationSchurPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_precond_local_chain") {
      options.fullEquivHybridLocalChainPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_precond_translation_block") {
      options.fullEquivHybridTranslationBlockPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_precond_translation_sparse_schur") {
      options.fullEquivHybridTranslationSparseSchurPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_precond_translation_local_schur") {
      options.fullEquivHybridTranslationLocalSchurPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_precond_translation_local_schur_max_poses") {
      options.fullEquivHybridTranslationLocalSchurMaxActivePoses =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_precond_laplacian_deflation") {
      options.fullEquivHybridLaplacianDeflationPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_precond_laplacian_deflation_basis_size") {
      options.fullEquivHybridLaplacianDeflationBasisSize =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_precond_laplacian_deflation_max_eigen_poses") {
      options.fullEquivHybridLaplacianDeflationMaxEigenPoses =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_precond_reduced_rotation") {
      options.fullEquivHybridReducedRotationPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_reduced_rotation_initial_guess") {
      options.fullEquivHybridReducedRotationInitialGuess =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_translation_recovery_initial_guess") {
      options.fullEquivHybridTranslationRecoveryInitialGuess =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_rqn_warm_start") {
      options.fullEquivHybridRqnWarmStart = parseBool(requireValue(arg));
    } else if (arg == "--feh_precond_rqn_memory") {
      options.fullEquivHybridRqnMemoryPreconditioner =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_rqn_memory_size") {
      options.fullEquivHybridRqnMemorySize =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_rqn_min_curvature_ratio") {
      options.fullEquivHybridRqnMinCurvatureRatio =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_warm_start_max_norm") {
      options.fullEquivHybridWarmStartMaxNorm = std::stod(requireValue(arg));
    } else if (arg == "--feh_warm_start_backtracking_steps") {
      options.fullEquivHybridWarmStartBacktrackingSteps =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_select_best_backtracking") {
      options.fullEquivHybridSelectBestBacktrackingTrial =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_step_pareto_selector") {
      options.fullEquivHybridStepParetoSelector = parseBool(requireValue(arg));
    } else if (arg == "--feh_step_pareto_min_decrease_ratio") {
      options.fullEquivHybridStepParetoMinDecreaseRatio =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_backtracking_shared_pose_prox_weight") {
      options.fullEquivHybridBacktrackingSharedPoseProxWeight =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_projected_model_rho_guard") {
      options.fullEquivHybridProjectedModelRhoGuard =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_projected_model_rho_eta") {
      options.fullEquivHybridProjectedModelRhoEta =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_active_separator_correction") {
      options.fullEquivHybridActiveSeparatorCorrection =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_active_separator_step_cap") {
      options.fullEquivHybridActiveSeparatorStepCap =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_active_separator_block_jacobi") {
      options.fullEquivHybridActiveSeparatorBlockJacobi =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_active_separator_compact_schur") {
      options.fullEquivHybridActiveSeparatorCompactSchur =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_active_separator_lm_schur") {
      options.fullEquivHybridActiveSeparatorLmSchur =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_active_separator_lm_schur_max_boundary_poses") {
      options.fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_active_separator_lm_schur_max_private_cols") {
      options.fullEquivHybridActiveSeparatorLmSchurMaxPrivateCols =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_active_separator_lm_schur_damping") {
      options.fullEquivHybridActiveSeparatorLmSchurDamping =
          std::stod(requireValue(arg));
    } else if (
        arg == "--feh_active_separator_lm_schur_backtracking_steps") {
      options.fullEquivHybridActiveSeparatorLmSchurBacktrackingSteps =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (
        arg == "--feh_active_separator_lm_schur_min_cost_improvement") {
      options.fullEquivHybridActiveSeparatorLmSchurMinCostImprovement =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_active_separator_lm_schur_max_rounds") {
      options.fullEquivHybridActiveSeparatorLmSchurMaxRounds =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_active_separator_lm_schur_score_mode") {
      options.fullEquivHybridActiveSeparatorLmSchurScoreMode =
          DPGO::parseManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode(
              requireValue(arg));
    } else if (arg == "--feh_active_separator_lm_schur_gradient_guard") {
      options.fullEquivHybridActiveSeparatorLmSchurGradientGuard =
          parseBool(requireValue(arg));
    } else if (
        arg ==
        "--feh_active_separator_lm_schur_max_gradient_increase_ratio") {
      options.fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_translation_recovery_polish") {
      options.fullEquivHybridTranslationRecoveryPolish =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_translation_recovery_polish_gradient_guard") {
      options.fullEquivHybridTranslationRecoveryPolishGradientGuard =
          parseBool(requireValue(arg));
    } else if (
        arg ==
        "--feh_translation_recovery_polish_max_gradient_increase_ratio") {
      options.fullEquivHybridTranslationRecoveryPolishMaxGradientIncreaseRatio =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_translation_recovery_polish_backtracking") {
      options.fullEquivHybridTranslationRecoveryPolishBacktracking =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_translation_recovery_polish_backtracking_steps") {
      options.fullEquivHybridTranslationRecoveryPolishBacktrackingSteps =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_translation_recovery_polish_merit_selector") {
      options.fullEquivHybridTranslationRecoveryPolishMeritSelector =
          parseBool(requireValue(arg));
    } else if (
        arg ==
        "--feh_translation_recovery_polish_merit_min_cost_recovery_ratio") {
      options
          .fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_translation_recovery_polish_selected_only") {
      options.fullEquivHybridTranslationRecoveryPolishSelectedOnly =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_translation_recovery_step_trials") {
      options.fullEquivHybridTranslationRecoveryStepTrials =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_local_portfolio") {
      options.fullEquivHybridLocalPortfolio = parseBool(requireValue(arg));
    } else if (arg == "--manual_full_portfolio") {
      options.manualFullPortfolio = parseBool(requireValue(arg));
    } else if (arg == "--feh_schwarz_presmoothing") {
      options.fullEquivHybridSchwarzPreSmoothing =
          parseBool(requireValue(arg));
    } else if (arg == "--feh_schwarz_sweeps") {
      options.fullEquivHybridSchwarzSweeps =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--feh_schwarz_max_block_norm") {
      options.fullEquivHybridSchwarzMaxBlockNorm =
          std::stod(requireValue(arg));
    } else if (arg == "--feh_schwarz_block_mode") {
      options.fullEquivHybridSchwarzBlockMode =
          DPGO::parseManualDpgoMmFullEquivHybridSchwarzBlockMode(
              requireValue(arg));
    } else if (arg == "--local_max_iterations") {
      options.trustRegionIterations =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--local_max_iterations_accepted") {
      options.trustRegionAcceptedIterations =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--local_max_tcg_iterations") {
      options.trustRegionMaxInnerIterations =
          std::stoi(requireValue(arg));
    } else if (arg == "--local_grad_norm_tol") {
      options.trustRegionTolerance = std::stod(requireValue(arg));
    } else if (arg == "--trust_region_initial_radius") {
      options.trustRegionInitialRadius = std::stod(requireValue(arg));
    } else if (arg == "--record_local_model_diagnostics") {
      options.recordLocalModelDiagnostics = parseBool(requireValue(arg));
    } else if (arg == "--parallel_local_solves") {
      options.parallelLocalSolves = parseBool(requireValue(arg));
    } else if (arg == "--adaptive_reduced_tcg") {
      options.adaptiveReducedTcg = parseBool(requireValue(arg));
    } else if (arg == "--adaptive_reduced_tcg_max_iterations") {
      options.adaptiveReducedTcgMaxIterations = std::stoi(requireValue(arg));
    } else if (arg == "--adaptive_reduced_tcg_gradient_ratio") {
      options.adaptiveReducedTcgGradientRatio = std::stod(requireValue(arg));
    } else if (arg == "--reduced_rotation_tcg_relative_tolerance") {
      options.reducedRotationTcgRelativeTolerance =
          std::stod(requireValue(arg));
    } else if (arg == "--local_state_extrapolation") {
      options.localStateExtrapolation = parseBool(requireValue(arg));
    } else if (arg == "--local_state_extrapolation_gamma") {
      options.localStateExtrapolationGamma = std::stod(requireValue(arg));
    } else if (arg == "--local_state_extrapolation_gammas") {
      options.localStateExtrapolationGammas =
          parseDoubleList(requireValue(arg));
    } else if (arg == "--local_state_extrapolation_use_active_surrogate") {
      options.localStateExtrapolationUseActiveSurrogate =
          parseBool(requireValue(arg));
    } else if (arg == "--local_boundary_proximal_candidate") {
      options.localBoundaryProximalCandidate =
          parseBool(requireValue(arg));
    } else if (arg == "--local_boundary_proximal_weight") {
      options.localBoundaryProximalWeight = std::stod(requireValue(arg));
    } else if (arg == "--local_boundary_proximal_weights") {
      options.localBoundaryProximalWeights =
          parseDoubleList(requireValue(arg));
    } else if (arg == "--local_model_g_extrapolation") {
      options.localModelGExtrapolation = parseBool(requireValue(arg));
    } else if (arg == "--local_model_g_extrapolation_gamma") {
      options.localModelGExtrapolationGamma = std::stod(requireValue(arg));
    } else if (arg == "--local_model_g_extrapolation_gammas") {
      options.localModelGExtrapolationGammas =
          parseDoubleList(requireValue(arg));
    } else if (arg == "--coupled_state_g_extrapolation") {
      options.coupledStateGExtrapolation = parseBool(requireValue(arg));
    } else if (arg == "--coupled_state_g_extrapolation_gamma") {
      options.coupledStateGExtrapolationGamma =
          std::stod(requireValue(arg));
    } else if (arg == "--coupled_state_g_extrapolation_gammas") {
      options.coupledStateGExtrapolationGammas =
          parseDoubleList(requireValue(arg));
    } else if (arg == "--local_anderson_acceleration") {
      options.localAndersonAcceleration = parseBool(requireValue(arg));
    } else if (arg == "--local_anderson_max_alpha") {
      options.localAndersonMaxAlpha = std::stod(requireValue(arg));
    } else if (arg == "--local_squarem_max_alpha") {
      options.localSquaremMaxAlpha = std::stod(requireValue(arg));
    } else if (arg == "--local_candidate_cost_tie_tolerance") {
      options.localCandidateCostTieTolerance = std::stod(requireValue(arg));
    } else if (arg == "--global_state_extrapolation") {
      options.globalStateExtrapolation = parseBool(requireValue(arg));
    } else if (arg == "--global_state_extrapolation_gamma") {
      options.globalStateExtrapolationGamma = std::stod(requireValue(arg));
    } else if (arg == "--global_state_extrapolation_gammas") {
      options.globalStateExtrapolationGammas =
          parseDoubleList(requireValue(arg));
    } else if (arg == "--global_anderson_acceleration") {
      options.globalAndersonAcceleration = parseBool(requireValue(arg));
    } else if (arg == "--global_anderson_max_alpha") {
      options.globalAndersonMaxAlpha = std::stod(requireValue(arg));
    } else if (arg == "--local_gradient_correction") {
      options.localGradientCorrection = parseBool(requireValue(arg));
    } else if (arg == "--local_gradient_correction_step") {
      options.localGradientCorrectionStep = std::stod(requireValue(arg));
    } else if (arg == "--local_gradient_correction_steps") {
      options.localGradientCorrectionSteps =
          parseDoubleList(requireValue(arg));
    } else if (arg == "--local_gradient_correction_shared_step") {
      options.localGradientCorrectionSharedStep =
          parseBool(requireValue(arg));
    } else if (arg == "--local_gradient_correction_neighborhood_step") {
      options.localGradientCorrectionNeighborhoodStep =
          parseBool(requireValue(arg));
    } else if (arg == "--local_gradient_correction_curvature_step") {
      options.localGradientCorrectionCurvatureStep =
          parseBool(requireValue(arg));
    } else if (arg == "--local_gradient_correction_coupled_direction") {
      options.localGradientCorrectionCoupledDirection =
          parseBool(requireValue(arg));
    } else if (
        arg ==
        "--local_gradient_correction_coupled_direction_budget_fraction") {
      options.localGradientCorrectionCoupledDirectionBudgetFraction =
          std::stod(requireValue(arg));
    } else if (
        arg ==
        "--local_gradient_correction_coupled_direction_max_packets_per_receiver") {
      options.localGradientCorrectionCoupledDirectionMaxPacketsPerReceiver =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (
        arg ==
        "--local_gradient_correction_coupled_direction_topk_entries") {
      options.localGradientCorrectionCoupledDirectionTopKEntries =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (
        arg ==
        "--local_gradient_correction_coupled_direction_min_score") {
      options.localGradientCorrectionCoupledDirectionMinScore =
          std::stod(requireValue(arg));
    } else if (
        arg ==
        "--local_gradient_correction_coupled_direction_min_score_ratio") {
      options.localGradientCorrectionCoupledDirectionMinScoreRatio =
          std::stod(requireValue(arg));
    } else if (
        arg ==
        "--local_gradient_correction_coupled_direction_byte_budget_mb") {
      options.localGradientCorrectionCoupledDirectionByteBudgetMb =
          std::stod(requireValue(arg));
    } else if (arg == "--local_gradient_correction_boundary_only") {
      options.localGradientCorrectionBoundaryOnly =
          parseBool(requireValue(arg));
    } else if (arg == "--local_gradient_correction_block_jacobi") {
      options.localGradientCorrectionBlockJacobi =
          parseBool(requireValue(arg));
    } else if (arg == "--local_gradient_correction_compact_schur") {
      options.localGradientCorrectionCompactSchur =
          parseBool(requireValue(arg));
    } else if (
        arg == "--local_gradient_correction_compact_schur_max_boundary_poses") {
      options.localGradientCorrectionCompactSchurMaxBoundaryPoses =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (
        arg == "--local_gradient_correction_compact_schur_max_private_cols") {
      options.localGradientCorrectionCompactSchurMaxPrivateCols =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--local_gradient_correction_compact_schur_damping") {
      options.localGradientCorrectionCompactSchurDamping =
          std::stod(requireValue(arg));
    } else if (
        arg ==
        "--local_gradient_correction_compact_schur_min_cost_decrease") {
      options.localGradientCorrectionCompactSchurMinCostDecrease =
          std::stod(requireValue(arg));
    } else if (
        arg ==
        "--local_gradient_correction_compact_schur_gradient_guard") {
      options.localGradientCorrectionCompactSchurGradientGuard =
          parseBool(requireValue(arg));
    } else if (
        arg ==
        "--local_gradient_correction_compact_schur_max_gradient_increase_ratio") {
      options.localGradientCorrectionCompactSchurMaxGradientIncreaseRatio =
          std::stod(requireValue(arg));
    } else if (arg == "--local_gradient_correction_fresh_neighbor_exchange") {
      options.localGradientCorrectionFreshNeighborExchange =
          parseBool(requireValue(arg));
    } else if (arg == "--local_gradient_correction_fresh_min_pose_delta") {
      options.localGradientCorrectionFreshMinPoseDelta =
          std::stod(requireValue(arg));
    } else if (arg == "--local_gradient_correction_fresh_delta_mode") {
      options.localGradientCorrectionFreshDeltaMode =
          parsePostExchangeDeltaMode(requireValue(arg));
    } else if (arg == "--local_gradient_correction_fresh_budget_fraction") {
      options.localGradientCorrectionFreshBudgetFraction =
          std::stod(requireValue(arg));
    } else if (
        arg == "--local_gradient_correction_fresh_max_poses_per_receiver") {
      options.localGradientCorrectionFreshMaxPosesPerReceiver =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--local_gradient_correction_inner_rounds") {
      options.localGradientCorrectionInnerRounds =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--local_gradient_correction_max_rounds") {
      options.localGradientCorrectionMaxRounds =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--post_exchange_period") {
      options.postExchangePeriod =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--post_exchange_min_pose_delta") {
      options.postExchangeMinPoseDelta = std::stod(requireValue(arg));
    } else if (arg == "--post_exchange_delta_mode") {
      options.postExchangeDeltaMode =
          parsePostExchangeDeltaMode(requireValue(arg));
    } else if (arg == "--post_exchange_budget_fraction") {
      options.postExchangeBudgetFraction = std::stod(requireValue(arg));
    } else if (arg == "--post_exchange_max_poses_per_receiver") {
      options.postExchangeMaxPosesPerReceiver =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--comm_topology_file") {
      options.communicationTopologyFile = requireValue(arg);
    } else if (arg == "--comm_topology_initial_full") {
      options.communicationTopologyInitialFull = parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_max_relay_hops") {
      options.communicationTopologyMaxRelayHops =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--comm_topology_hop_budget_fraction") {
      options.communicationTopologyHopBudgetFraction =
          std::stod(requireValue(arg));
    } else if (arg == "--comm_topology_stale_aware_relay_score") {
      options.communicationTopologyStaleAwareRelayScore =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_stale_aware_relay_age_gain") {
      options.communicationTopologyStaleAwareRelayAgeGain =
          std::stod(requireValue(arg));
    } else if (arg == "--comm_topology_delta_compression") {
      options.communicationTopologyDeltaCompression =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_delta_mode") {
      options.communicationTopologyDeltaMode =
          parseCommunicationDeltaMode(requireValue(arg));
    } else if (arg == "--comm_topology_delta_topk") {
      options.communicationTopologyDeltaTopK =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--comm_topology_delta_max_reconstruction_error") {
      options.communicationTopologyDeltaMaxReconstructionError =
          std::stod(requireValue(arg));
    } else if (arg == "--comm_topology_lifted_delta_compression") {
      options.communicationTopologyLiftedDeltaCompression =
          parseBool(requireValue(arg));
    } else if (
        arg == "--comm_topology_lifted_delta_max_reconstruction_error") {
      options.communicationTopologyLiftedDeltaMaxReconstructionError =
          std::stod(requireValue(arg));
    } else if (arg == "--comm_topology_lifted_delta_rank") {
      options.communicationTopologyLiftedDeltaRank =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--comm_topology_value_scheduler") {
      options.communicationTopologyValueScheduler =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_value_scheduler_mode") {
      options.communicationTopologyValueSchedulerMode = requireValue(arg);
    } else if (arg == "--comm_topology_value_byte_budget_mb") {
      options.communicationTopologyValueByteBudgetMb =
          std::stod(requireValue(arg));
    } else if (arg == "--comm_topology_value_min_score_ratio") {
      options.communicationTopologyValueMinScoreRatio =
          std::stod(requireValue(arg));
    } else if (arg == "--comm_topology_value_budget_pacing") {
      options.communicationTopologyValueBudgetPacing =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_interface_model") {
      options.communicationTopologyInterfaceModel =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_interface_model_payload") {
      options.communicationTopologyInterfaceModelPayload = requireValue(arg);
    } else if (arg == "--comm_topology_interface_model_local_merit") {
      options.communicationTopologyInterfaceModelLocalMerit =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_boundary_candidate") {
      options.communicationTopologyBoundaryCandidate =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_boundary_surrogate") {
      options.communicationTopologyBoundarySurrogate =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_boundary_surrogate_stale_gain") {
      options.communicationTopologyBoundarySurrogateStaleGain =
          std::stod(requireValue(arg));
    } else if (arg == "--comm_topology_reduced_interface_model") {
      options.communicationTopologyReducedInterfaceModel =
          parseBool(requireValue(arg));
    } else if (arg == "--comm_topology_reduced_interface_weight") {
      options.communicationTopologyReducedInterfaceWeight =
          std::stod(requireValue(arg));
    } else if (arg ==
               "--comm_topology_reduced_interface_max_local_iters") {
      options.communicationTopologyReducedInterfaceMaxLocalIterations =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--comm_topology_reduced_interface_candidate") {
      options.communicationTopologyReducedInterfaceCandidate =
          parseBool(requireValue(arg));
    } else if (arg == "--stale_boundary_proximal") {
      options.staleBoundaryProximal = parseBool(requireValue(arg));
    } else if (arg == "--stale_boundary_proximal_weight") {
      options.staleBoundaryProximalWeight = std::stod(requireValue(arg));
    } else if (arg == "--stale_boundary_prediction") {
      options.staleBoundaryPrediction = parseBool(requireValue(arg));
    } else if (arg == "--stale_boundary_prediction_gain") {
      options.staleBoundaryPredictionGain = std::stod(requireValue(arg));
    } else if (arg == "--sync_amm_reference_after_local_gradient_correction") {
      options.syncAmmReferenceAfterLocalGradientCorrection =
          parseBool(requireValue(arg));
    } else if (arg == "--global_gradient_correction") {
      options.globalGradientCorrection = parseBool(requireValue(arg));
    } else if (arg == "--global_gradient_correction_step") {
      options.globalGradientCorrectionStep = std::stod(requireValue(arg));
    } else if (arg == "--global_gradient_correction_steps") {
      options.globalGradientCorrectionSteps =
          parseDoubleList(requireValue(arg));
    } else if (arg == "--reset_local_history_after_global_correction") {
      options.resetLocalHistoryAfterGlobalCorrection =
          parseBool(requireValue(arg));
    } else if (arg == "--sync_amm_reference_after_global_correction") {
      options.syncAmmReferenceAfterGlobalCorrection =
          parseBool(requireValue(arg));
    } else if (arg == "--amm_eta0") {
      options.ammEta0 = std::stod(requireValue(arg));
    } else if (arg == "--amm_eta1") {
      options.ammEta1 = std::stod(requireValue(arg));
    } else if (arg == "--amm_psi") {
      options.ammPsi = std::stod(requireValue(arg));
    } else if (arg == "--amm_phi") {
      options.ammPhi = std::stod(requireValue(arg));
    } else if (arg == "--amm_gamma_scale") {
      options.ammGammaScale = std::stod(requireValue(arg));
    } else if (arg == "--amm_gamma_scales") {
      options.ammGammaScales = parseDoubleList(requireValue(arg));
    } else if (arg == "--amm_accepted_delta") {
      options.ammAcceptedDelta = std::stod(requireValue(arg));
    } else if (arg == "--amm_oscillation_count_period") {
      options.ammOscillationCountPeriod = std::stoi(requireValue(arg));
    } else if (arg == "--amm_max_oscillations") {
      options.ammMaxOscillations = std::stoi(requireValue(arg));
    } else if (arg == "--amm_proximal_start") {
      options.ammProximalStart = parseBool(requireValue(arg));
    } else if (arg == "--amm_dpgo_restart_fallback") {
      options.ammDpgoRestartFallback = parseBool(requireValue(arg));
    } else if (arg == "--amm_dpgo_surrogate_parity") {
      options.ammDpgoSurrogateParity = parseBool(requireValue(arg));
    } else if (arg == "--amm_baseline_surrogate_state") {
      options.ammBaselineSurrogateState = parseBool(requireValue(arg));
    } else if (arg == "--amm_dpgo_recursive_simple_state") {
      options.ammDpgoRecursiveSimpleState = parseBool(requireValue(arg));
    } else if (arg == "--amm_recursive_simple_reanchor_period") {
      options.ammRecursiveSimpleReanchorPeriod =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--amm_dpgo_strict_refined_gate") {
      options.ammDpgoStrictRefinedGate = parseBool(requireValue(arg));
    } else if (arg == "--amm_dpgo_recover_translations_after_proximal") {
      options.ammDpgoRecoverTranslationsAfterProximal =
          parseBool(requireValue(arg));
    } else if (arg == "--amm_dpgo_refined_starts_at_recovered_proximal") {
      options.ammDpgoRefinedStartsAtRecoveredProximal =
          parseBool(requireValue(arg));
    } else if (arg == "--amm_dpgo_mixed_surrogate_portfolio") {
      options.ammDpgoMixedSurrogatePortfolio = parseBool(requireValue(arg));
    } else if (
        arg == "--amm_mixed_surrogate_skip_simple_after_true_local_streak") {
      options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak =
          std::stoi(requireValue(arg));
    } else if (
        arg == "--amm_mixed_surrogate_force_simple_every_skipped_rounds") {
      options.ammMixedSurrogateForceSimpleEverySkippedRounds =
          std::stoi(requireValue(arg));
    } else if (arg == "--amm_cooldown_after_rejected") {
      options.ammCooldownAfterRejected = std::stoi(requireValue(arg));
    } else if (arg == "--amm_prox_reset_skip_refined_solve") {
      options.ammProxResetSkipRefinedSolve = parseBool(requireValue(arg));
    } else if (arg == "--amm_dpgo_proximal_fallback_only") {
      options.ammDpgoProximalFallbackOnly = parseBool(requireValue(arg));
    } else if (arg == "--amm_lazy_plain_after_certificate") {
      options.ammLazyPlainAfterCertificate = parseBool(requireValue(arg));
    } else if (arg == "--amm_surrogate_first_exact_evaluation") {
      options.ammSurrogateFirstExactEvaluation =
          parseBool(requireValue(arg));
    } else if (arg == "--amm_max_soft_restart_hits0") {
      options.ammMaxSoftRestartHits0 = std::stoi(requireValue(arg));
    } else if (arg == "--amm_max_soft_restart_hits1") {
      options.ammMaxSoftRestartHits1 = std::stoi(requireValue(arg));
    } else if (arg == "--amm_local_merit_filter") {
      options.ammLocalMeritFilter = parseBool(requireValue(arg));
    } else if (arg == "--amm_local_merit_cost_tie_tolerance") {
      options.ammLocalMeritCostTieTolerance = std::stod(requireValue(arg));
    } else if (arg == "--trace_amm") {
      options.traceAmm = parseBool(requireValue(arg));
    } else if (arg == "--profile_optimizer") {
      options.profileOptimizer = parseBool(requireValue(arg));
    } else if (arg == "--candidate_evaluation_cache") {
      options.candidateEvaluationCache = parseBool(requireValue(arg));
    } else if (arg == "--candidate_object_reuse") {
      options.candidateObjectReuse = parseBool(requireValue(arg));
    } else if (arg == "--reduced_adaptive_portfolio_certified_fast_path") {
      options.reducedAdaptivePortfolioCertifiedFastPath =
          parseBool(requireValue(arg));
    } else if (arg == "--reduced_rotation_direct_objective") {
      options.reducedRotationDirectObjective = parseBool(requireValue(arg));
    } else if (arg == "--fused_candidate_evaluation") {
      options.fusedCandidateEvaluation = parseBool(requireValue(arg));
    } else if (arg == "--lazy_candidate_gradient_evaluation") {
      options.lazyCandidateGradientEvaluation =
          parseBool(requireValue(arg));
    } else if (arg == "--lazy_solver_start_gradient_evaluation") {
      options.lazySolverStartGradientEvaluation =
          parseBool(requireValue(arg));
    } else if (arg == "--lazy_surrogate_candidate_gradient_evaluation") {
      options.lazySurrogateCandidateGradientEvaluation =
          parseBool(requireValue(arg));
    } else if (arg == "--reduced_rotation_curvature_cauchy_candidate") {
      options.reducedRotationCurvatureCauchyCandidate =
          parseBool(requireValue(arg));
    } else if (arg == "--reduced_rotation_curvature_fallback_candidate") {
      options.reducedRotationCurvatureFallbackCandidate =
          parseBool(requireValue(arg));
    } else if (arg == "--reduced_rotation_gradient_boundary_candidate") {
      options.reducedRotationGradientBoundaryCandidate =
          parseBool(requireValue(arg));
    } else if (arg == "--reduced_rotation_surrogate_tcg_accept") {
      options.reducedRotationSurrogateTcgAccept = parseBool(requireValue(arg));
    } else if (arg ==
               "--reduced_rotation_skip_redundant_candidate_projection") {
      options.reducedRotationSkipRedundantCandidateProjection =
          parseBool(requireValue(arg));
    } else if (arg == "--print_iteration_summary") {
      options.printIterationSummary = parseBool(requireValue(arg));
    } else if (arg == "--output_dir") {
      options.outputDirectory = requireValue(arg);
    } else if (arg == "--external_initial_estimate") {
      options.externalInitialEstimatePath = requireValue(arg);
    } else if (arg == "--save_iteration_estimates") {
      options.saveIterationEstimates = parseBool(requireValue(arg));
    } else if (arg == "--save") {
      options.save = parseBool(requireValue(arg));
    } else if (arg == "--dist_init") {
      const bool distInit = parseBool(requireValue(arg));
      options.centralizedChordalInit = !distInit;
    } else if (arg == "--dist_init_refinement_rounds") {
      options.distributedInitializationRefinementRounds =
          static_cast<unsigned>(std::stoul(requireValue(arg)));
    } else if (arg == "--dist_init_fixed_neighbor_chordal_refit") {
      options.distributedInitializationFixedNeighborChordalRefit =
          parseBool(requireValue(arg));
    } else if (arg == "--preconditioner" ||
               arg == "--local_preconditioned_grad_norm_tol" ||
               arg == "--tnt_kappa" || arg == "--tnt_theta" ||
               arg == "--loss" || arg == "--init_red_rot_iters" ||
               arg == "--init_rot_iters" ||
               arg == "--init_red_trans_iters" ||
               arg == "--init_trans_iters") {
      const std::string ignored = requireValue(arg);
      (void)ignored;
    } else {
      throw std::invalid_argument("Unknown option: " + arg);
    }
  }

  if (dataset.empty()) {
    printUsage(argv[0]);
    return 1;
  }

  options.scheme = DPGO::parseManualDpgoMmScheme(scheme, accelerated);
  options.localSolver = DPGO::parseManualDpgoMmLocalSolver(localSolver);

    std::cout << "ABLATION_CONFIG method=manual_dpgo_mm"
              << " scheme=" << DPGO::manualDpgoMmSchemeName(options.scheme)
              << " local_solver="
              << DPGO::manualDpgoMmLocalSolverName(options.localSolver)
              << " reduced_rotation_preconditioner="
              << DPGO::manualDpgoMmReducedRotationPreconditionerName(
                     options.reducedRotationPreconditioner)
              << " reduced_surrogate_mode="
              << DPGO::manualDpgoMmReducedSurrogateModeName(
                     options.reducedSurrogateMode)
              << " surrogate_mode="
              << DPGO::manualDpgoMmSurrogateModeName(options.surrogateMode)
              << " edge_split_theta_mode="
              << DPGO::manualDpgoMmEdgeSplitThetaModeName(
                     options.edgeSplitThetaMode)
              << " edge_split_theta_default="
              << options.edgeSplitThetaDefault
              << " edge_split_theta_min=" << options.edgeSplitThetaMin
              << " edge_split_theta_max=" << options.edgeSplitThetaMax
              << " mm_accelerator="
              << DPGO::manualDpgoMmMmAcceleratorModeName(
                     options.mmAcceleratorMode)
              << " mm_safeguard="
              << DPGO::manualDpgoMmMmSafeguardName(options.mmSafeguard)
              << " debug_surrogate_bound_check="
              << (options.debugSurrogateBoundCheck ? "true" : "false")
              << " debug_surrogate_bound_samples="
              << options.debugSurrogateBoundSamples
              << " feh_linear_backend="
              << DPGO::manualDpgoMmFullEquivHybridBackendName(
                     options.fullEquivHybridBackend)
              << " dist_init="
              << (options.centralizedChordalInit
                      ? "centralized_chordal_same_init"
                      : "distributed_local_chordal")
              << " dist_init_refinement_rounds="
              << options.distributedInitializationRefinementRounds
              << " dist_init_fixed_neighbor_chordal_refit="
              << (options.distributedInitializationFixedNeighborChordalRefit
                      ? "true"
                      : "false")
              << " external_initial_estimate="
              << (options.externalInitialEstimatePath.empty()
                      ? "none"
                      : options.externalInitialEstimatePath)
              << " local_max_iterations=" << options.trustRegionIterations
              << " local_max_iterations_accepted="
              << options.trustRegionAcceptedIterations
              << " local_max_tcg_iterations="
              << options.trustRegionMaxInnerIterations
              << " feh_schur_warm_start="
              << (options.fullEquivHybridSchurWarmStart ? "true" : "false")
              << " feh_final_polish_rtr="
              << (options.fullEquivHybridFinalPolishRtr ? "true" : "false")
              << " feh_schur_damping="
              << options.fullEquivHybridSchurDamping
              << " feh_linear_rel_tol="
              << options.fullEquivHybridLinearRelativeTolerance
              << " feh_linear_abs_tol="
              << options.fullEquivHybridLinearAbsoluteTolerance
              << " feh_linear_max_iters="
              << options.fullEquivHybridLinearMaxIterations
              << " feh_linear_block_jacobi="
              << (options.fullEquivHybridLinearBlockJacobiPreconditioner
                      ? "true"
                      : "false")
              << " feh_sparse_matvec="
              << (options.fullEquivHybridSparseMatrixVectorProduct
                      ? "true"
                      : "false")
              << " feh_precond_translation_schur="
              << (options.fullEquivHybridLinearTranslationSchurPreconditioner
                      ? "true"
                      : "false")
              << " feh_precond_local_chain="
              << (options.fullEquivHybridLocalChainPreconditioner ? "true"
                                                                   : "false")
              << " feh_precond_translation_block="
              << (options.fullEquivHybridTranslationBlockPreconditioner
                      ? "true"
                      : "false")
              << " feh_precond_translation_sparse_schur="
              << (options.fullEquivHybridTranslationSparseSchurPreconditioner
                      ? "true"
                      : "false")
              << " feh_precond_translation_local_schur="
              << (options.fullEquivHybridTranslationLocalSchurPreconditioner
                      ? "true"
                      : "false")
              << " feh_precond_translation_local_schur_max_poses="
              << options.fullEquivHybridTranslationLocalSchurMaxActivePoses
              << " feh_precond_laplacian_deflation="
              << (options.fullEquivHybridLaplacianDeflationPreconditioner
                      ? "true"
                      : "false")
              << " feh_precond_laplacian_deflation_basis_size="
              << options.fullEquivHybridLaplacianDeflationBasisSize
              << " feh_precond_laplacian_deflation_max_eigen_poses="
              << options.fullEquivHybridLaplacianDeflationMaxEigenPoses
              << " feh_precond_reduced_rotation="
              << (options.fullEquivHybridReducedRotationPreconditioner
                      ? "true"
                      : "false")
              << " feh_reduced_rotation_initial_guess="
              << (options.fullEquivHybridReducedRotationInitialGuess ? "true"
                                                                      : "false")
              << " feh_translation_recovery_initial_guess="
              << (options.fullEquivHybridTranslationRecoveryInitialGuess
                      ? "true"
                      : "false")
              << " feh_rqn_warm_start="
              << (options.fullEquivHybridRqnWarmStart ? "true" : "false")
              << " feh_precond_rqn_memory="
              << (options.fullEquivHybridRqnMemoryPreconditioner ? "true"
                                                                 : "false")
              << " feh_rqn_memory_size="
              << options.fullEquivHybridRqnMemorySize
              << " feh_rqn_min_curvature_ratio="
              << options.fullEquivHybridRqnMinCurvatureRatio
              << " feh_warm_start_max_norm="
              << options.fullEquivHybridWarmStartMaxNorm
              << " feh_warm_start_backtracking_steps="
              << options.fullEquivHybridWarmStartBacktrackingSteps
              << " feh_select_best_backtracking="
              << (options.fullEquivHybridSelectBestBacktrackingTrial
                      ? "true"
                      : "false")
              << " feh_step_pareto_selector="
              << (options.fullEquivHybridStepParetoSelector ? "true"
                                                             : "false")
              << " feh_step_pareto_min_decrease_ratio="
              << options.fullEquivHybridStepParetoMinDecreaseRatio
              << " feh_backtracking_shared_pose_prox_weight="
              << options.fullEquivHybridBacktrackingSharedPoseProxWeight
              << " feh_projected_model_rho_guard="
              << (options.fullEquivHybridProjectedModelRhoGuard ? "true"
                                                                 : "false")
              << " feh_projected_model_rho_eta="
              << options.fullEquivHybridProjectedModelRhoEta
              << " feh_active_separator_correction="
              << (options.fullEquivHybridActiveSeparatorCorrection ? "true"
                                                                    : "false")
              << " feh_active_separator_step_cap="
              << options.fullEquivHybridActiveSeparatorStepCap
              << " feh_active_separator_block_jacobi="
              << (options.fullEquivHybridActiveSeparatorBlockJacobi ? "true"
                                                                     : "false")
              << " feh_active_separator_compact_schur="
              << (options.fullEquivHybridActiveSeparatorCompactSchur ? "true"
                                                                      : "false")
              << " feh_active_separator_lm_schur="
              << (options.fullEquivHybridActiveSeparatorLmSchur ? "true"
                                                                : "false")
              << " feh_active_separator_lm_schur_max_boundary_poses="
              << options.fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses
              << " feh_active_separator_lm_schur_max_private_cols="
              << options.fullEquivHybridActiveSeparatorLmSchurMaxPrivateCols
              << " feh_active_separator_lm_schur_damping="
              << options.fullEquivHybridActiveSeparatorLmSchurDamping
              << " feh_active_separator_lm_schur_backtracking_steps="
              << options
                     .fullEquivHybridActiveSeparatorLmSchurBacktrackingSteps
              << " feh_active_separator_lm_schur_min_cost_improvement="
              << options
                     .fullEquivHybridActiveSeparatorLmSchurMinCostImprovement
              << " feh_active_separator_lm_schur_max_rounds="
              << options.fullEquivHybridActiveSeparatorLmSchurMaxRounds
              << " feh_active_separator_lm_schur_score_mode="
              << DPGO::
                     manualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreModeName(
                         options
                             .fullEquivHybridActiveSeparatorLmSchurScoreMode)
              << " feh_active_separator_lm_schur_gradient_guard="
              << (options.fullEquivHybridActiveSeparatorLmSchurGradientGuard
                      ? "true"
                      : "false")
              << " feh_active_separator_lm_schur_max_gradient_increase_ratio="
              << options
                     .fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio
              << " feh_translation_recovery_polish="
              << (options.fullEquivHybridTranslationRecoveryPolish ? "true"
                                                                   : "false")
              << " feh_translation_recovery_polish_gradient_guard="
              << (options.fullEquivHybridTranslationRecoveryPolishGradientGuard
                      ? "true"
                      : "false")
              << " feh_translation_recovery_polish_max_gradient_increase_ratio="
              << options
                     .fullEquivHybridTranslationRecoveryPolishMaxGradientIncreaseRatio
              << " feh_translation_recovery_polish_backtracking="
              << (options.fullEquivHybridTranslationRecoveryPolishBacktracking
                      ? "true"
                      : "false")
              << " feh_translation_recovery_polish_backtracking_steps="
              << options
                     .fullEquivHybridTranslationRecoveryPolishBacktrackingSteps
              << " feh_translation_recovery_polish_merit_selector="
              << (options.fullEquivHybridTranslationRecoveryPolishMeritSelector
                      ? "true"
                      : "false")
              << " feh_translation_recovery_polish_merit_min_cost_recovery_ratio="
              << options
                     .fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio
              << " feh_translation_recovery_polish_selected_only="
              << (options.fullEquivHybridTranslationRecoveryPolishSelectedOnly
                      ? "true"
                      : "false")
              << " feh_translation_recovery_step_trials="
              << (options.fullEquivHybridTranslationRecoveryStepTrials
                      ? "true"
                      : "false")
              << " feh_local_portfolio="
              << (options.fullEquivHybridLocalPortfolio ? "true" : "false")
              << " feh_schwarz_presmoothing="
              << (options.fullEquivHybridSchwarzPreSmoothing ? "true"
                                                              : "false")
              << " feh_schwarz_sweeps="
              << options.fullEquivHybridSchwarzSweeps
              << " feh_schwarz_max_block_norm="
              << options.fullEquivHybridSchwarzMaxBlockNorm
              << " feh_schwarz_block_mode="
              << DPGO::manualDpgoMmFullEquivHybridSchwarzBlockModeName(
                     options.fullEquivHybridSchwarzBlockMode)
              << " record_local_model_diagnostics="
              << (options.recordLocalModelDiagnostics ? "true" : "false")
              << " parallel_local_solves="
              << (options.parallelLocalSolves ? "true" : "false")
              << " adaptive_reduced_tcg="
              << (options.adaptiveReducedTcg ? "true" : "false")
              << " adaptive_reduced_tcg_max_iterations="
              << options.adaptiveReducedTcgMaxIterations
              << " adaptive_reduced_tcg_gradient_ratio="
              << options.adaptiveReducedTcgGradientRatio
              << " reduced_rotation_tcg_relative_tolerance="
              << options.reducedRotationTcgRelativeTolerance
              << " local_state_extrapolation="
              << (options.localStateExtrapolation ? "true" : "false")
              << " local_state_extrapolation_gamma="
              << options.localStateExtrapolationGamma
              << " local_state_extrapolation_use_active_surrogate="
              << (options.localStateExtrapolationUseActiveSurrogate ? "true"
                                                                    : "false")
              << " local_boundary_proximal_candidate="
              << (options.localBoundaryProximalCandidate ? "true" : "false")
              << " local_boundary_proximal_weight="
              << options.localBoundaryProximalWeight;
    if (!options.localStateExtrapolationGammas.empty()) {
      std::cout << " local_state_extrapolation_gammas=";
      for (std::size_t idx = 0;
           idx < options.localStateExtrapolationGammas.size(); ++idx) {
        if (idx > 0) {
          std::cout << ",";
        }
        std::cout << options.localStateExtrapolationGammas[idx];
      }
    }
    if (!options.localBoundaryProximalWeights.empty()) {
      std::cout << " local_boundary_proximal_weights=";
      for (std::size_t idx = 0;
           idx < options.localBoundaryProximalWeights.size(); ++idx) {
        if (idx > 0) {
          std::cout << ",";
        }
        std::cout << options.localBoundaryProximalWeights[idx];
      }
    }
    std::cout << " local_model_g_extrapolation="
              << (options.localModelGExtrapolation ? "true" : "false")
              << " local_model_g_extrapolation_gamma="
              << options.localModelGExtrapolationGamma;
    if (!options.localModelGExtrapolationGammas.empty()) {
      std::cout << " local_model_g_extrapolation_gammas=";
      for (std::size_t idx = 0;
           idx < options.localModelGExtrapolationGammas.size(); ++idx) {
        if (idx > 0) {
          std::cout << ",";
        }
        std::cout << options.localModelGExtrapolationGammas[idx];
      }
    }
    std::cout << " coupled_state_g_extrapolation="
              << (options.coupledStateGExtrapolation ? "true" : "false")
              << " coupled_state_g_extrapolation_gamma="
              << options.coupledStateGExtrapolationGamma;
    if (!options.coupledStateGExtrapolationGammas.empty()) {
      std::cout << " coupled_state_g_extrapolation_gammas=";
      for (std::size_t idx = 0;
           idx < options.coupledStateGExtrapolationGammas.size(); ++idx) {
        if (idx > 0) {
          std::cout << ",";
        }
        std::cout << options.coupledStateGExtrapolationGammas[idx];
      }
    }
    std::cout << " local_anderson_acceleration="
              << (options.localAndersonAcceleration ? "true" : "false")
              << " local_anderson_max_alpha="
              << options.localAndersonMaxAlpha;
    std::cout << " local_squarem_max_alpha="
              << options.localSquaremMaxAlpha;
    std::cout << " local_candidate_cost_tie_tolerance="
              << options.localCandidateCostTieTolerance;
    std::cout << " global_state_extrapolation="
              << (options.globalStateExtrapolation ? "true" : "false")
              << " global_state_extrapolation_gamma="
              << options.globalStateExtrapolationGamma;
    if (!options.globalStateExtrapolationGammas.empty()) {
      std::cout << " global_state_extrapolation_gammas=";
      for (std::size_t idx = 0;
           idx < options.globalStateExtrapolationGammas.size(); ++idx) {
        if (idx > 0) {
          std::cout << ",";
        }
        std::cout << options.globalStateExtrapolationGammas[idx];
      }
    }
    std::cout << " global_anderson_acceleration="
              << (options.globalAndersonAcceleration ? "true" : "false")
              << " global_anderson_max_alpha="
              << options.globalAndersonMaxAlpha;
    std::cout << " local_gradient_correction="
              << (options.localGradientCorrection ? "true" : "false")
              << " local_gradient_correction_step="
              << options.localGradientCorrectionStep;
    if (!options.localGradientCorrectionSteps.empty()) {
      std::cout << " local_gradient_correction_steps=";
      for (std::size_t idx = 0;
           idx < options.localGradientCorrectionSteps.size(); ++idx) {
        if (idx > 0) {
          std::cout << ",";
        }
        std::cout << options.localGradientCorrectionSteps[idx];
      }
    }
    std::cout << " local_gradient_correction_shared_step="
              << (options.localGradientCorrectionSharedStep ? "true"
                                                            : "false");
    std::cout << " local_gradient_correction_neighborhood_step="
              << (options.localGradientCorrectionNeighborhoodStep ? "true"
                                                                  : "false");
    std::cout << " local_gradient_correction_curvature_step="
              << (options.localGradientCorrectionCurvatureStep ? "true"
                                                               : "false");
    std::cout << " local_gradient_correction_coupled_direction="
              << (options.localGradientCorrectionCoupledDirection ? "true"
                                                                  : "false");
    std::cout
        << " local_gradient_correction_coupled_direction_budget_fraction="
        << options.localGradientCorrectionCoupledDirectionBudgetFraction;
    std::cout
        << " local_gradient_correction_coupled_direction_max_packets_per_receiver="
        << options.localGradientCorrectionCoupledDirectionMaxPacketsPerReceiver;
    std::cout << " local_gradient_correction_coupled_direction_topk_entries="
              << options.localGradientCorrectionCoupledDirectionTopKEntries;
    std::cout << " local_gradient_correction_coupled_direction_min_score="
              << options.localGradientCorrectionCoupledDirectionMinScore;
    std::cout
        << " local_gradient_correction_coupled_direction_min_score_ratio="
        << options.localGradientCorrectionCoupledDirectionMinScoreRatio;
    std::cout << " local_gradient_correction_coupled_direction_byte_budget_mb="
              << options.localGradientCorrectionCoupledDirectionByteBudgetMb;
    std::cout << " local_gradient_correction_boundary_only="
              << (options.localGradientCorrectionBoundaryOnly ? "true"
                                                              : "false");
    std::cout << " local_gradient_correction_block_jacobi="
              << (options.localGradientCorrectionBlockJacobi ? "true"
                                                             : "false");
    std::cout << " local_gradient_correction_compact_schur="
              << (options.localGradientCorrectionCompactSchur ? "true"
                                                              : "false");
    std::cout
        << " local_gradient_correction_compact_schur_max_boundary_poses="
        << options.localGradientCorrectionCompactSchurMaxBoundaryPoses;
    std::cout << " local_gradient_correction_compact_schur_max_private_cols="
              << options.localGradientCorrectionCompactSchurMaxPrivateCols;
    std::cout << " local_gradient_correction_compact_schur_damping="
              << options.localGradientCorrectionCompactSchurDamping;
    std::cout
        << " local_gradient_correction_compact_schur_min_cost_decrease="
        << options.localGradientCorrectionCompactSchurMinCostDecrease;
    std::cout
        << " local_gradient_correction_compact_schur_gradient_guard="
        << (options.localGradientCorrectionCompactSchurGradientGuard
                ? "true"
                : "false");
    std::cout
        << " local_gradient_correction_compact_schur_max_gradient_increase_ratio="
        << options.localGradientCorrectionCompactSchurMaxGradientIncreaseRatio;
    std::cout << " local_gradient_correction_fresh_neighbor_exchange="
              << (options.localGradientCorrectionFreshNeighborExchange
                      ? "true"
                      : "false");
    std::cout << " local_gradient_correction_fresh_min_pose_delta="
              << options.localGradientCorrectionFreshMinPoseDelta;
    std::cout << " local_gradient_correction_fresh_delta_mode="
              << postExchangeDeltaModeName(
                     options.localGradientCorrectionFreshDeltaMode);
    std::cout << " local_gradient_correction_fresh_budget_fraction="
              << options.localGradientCorrectionFreshBudgetFraction;
    std::cout << " local_gradient_correction_fresh_max_poses_per_receiver="
              << options.localGradientCorrectionFreshMaxPosesPerReceiver;
    std::cout << " local_gradient_correction_inner_rounds="
              << options.localGradientCorrectionInnerRounds;
    std::cout << " local_gradient_correction_max_rounds="
              << options.localGradientCorrectionMaxRounds;
    std::cout << " post_exchange_period=" << options.postExchangePeriod;
    std::cout << " post_exchange_min_pose_delta="
              << options.postExchangeMinPoseDelta;
    std::cout << " post_exchange_delta_mode="
              << postExchangeDeltaModeName(options.postExchangeDeltaMode);
    std::cout << " post_exchange_budget_fraction="
              << options.postExchangeBudgetFraction;
    std::cout << " post_exchange_max_poses_per_receiver="
              << options.postExchangeMaxPosesPerReceiver;
    std::cout << " comm_topology_file="
              << (options.communicationTopologyFile.empty()
                      ? "none"
                      : options.communicationTopologyFile);
    std::cout << " comm_topology_initial_full="
              << (options.communicationTopologyInitialFull ? "true"
                                                           : "false");
    std::cout << " comm_topology_max_relay_hops="
              << options.communicationTopologyMaxRelayHops;
    std::cout << " comm_topology_hop_budget_fraction="
              << options.communicationTopologyHopBudgetFraction;
    std::cout << " comm_topology_stale_aware_relay_score="
              << (options.communicationTopologyStaleAwareRelayScore ? "true"
                                                                    : "false");
    std::cout << " comm_topology_stale_aware_relay_age_gain="
              << options.communicationTopologyStaleAwareRelayAgeGain;
    std::cout << " comm_topology_delta_compression="
              << (options.communicationTopologyDeltaCompression ? "true"
                                                                : "false");
    std::cout << " comm_topology_delta_mode="
              << communicationDeltaModeName(
                     options.communicationTopologyDeltaMode);
    std::cout << " comm_topology_delta_topk="
              << options.communicationTopologyDeltaTopK;
    std::cout << " comm_topology_delta_max_reconstruction_error="
              << options.communicationTopologyDeltaMaxReconstructionError;
    std::cout << " comm_topology_lifted_delta_compression="
              << (options.communicationTopologyLiftedDeltaCompression ? "true"
                                                                      : "false");
    std::cout << " comm_topology_lifted_delta_max_reconstruction_error="
              << options
                     .communicationTopologyLiftedDeltaMaxReconstructionError;
    std::cout << " comm_topology_lifted_delta_rank="
              << options.communicationTopologyLiftedDeltaRank;
    std::cout << " comm_topology_value_scheduler="
              << (options.communicationTopologyValueScheduler ? "true"
                                                              : "false");
    std::cout << " comm_topology_value_scheduler_mode="
              << options.communicationTopologyValueSchedulerMode;
    std::cout << " comm_topology_value_byte_budget_mb="
              << options.communicationTopologyValueByteBudgetMb;
    std::cout << " comm_topology_value_min_score_ratio="
              << options.communicationTopologyValueMinScoreRatio;
    std::cout << " comm_topology_value_budget_pacing="
              << (options.communicationTopologyValueBudgetPacing ? "true"
                                                                 : "false");
    std::cout << " comm_topology_interface_model="
              << (options.communicationTopologyInterfaceModel ? "true"
                                                              : "false");
    std::cout << " comm_topology_interface_model_payload="
              << options.communicationTopologyInterfaceModelPayload;
    std::cout << " comm_topology_interface_model_local_merit="
              << (options.communicationTopologyInterfaceModelLocalMerit
                      ? "true"
                      : "false");
    std::cout << " comm_topology_boundary_candidate="
              << (options.communicationTopologyBoundaryCandidate ? "true"
                                                                 : "false");
    std::cout << " comm_topology_boundary_surrogate="
              << (options.communicationTopologyBoundarySurrogate ? "true"
                                                                 : "false");
    std::cout << " comm_topology_boundary_surrogate_stale_gain="
              << options.communicationTopologyBoundarySurrogateStaleGain;
    std::cout << " comm_topology_reduced_interface_model="
              << (options.communicationTopologyReducedInterfaceModel
                      ? "true"
                      : "false");
    std::cout << " comm_topology_reduced_interface_weight="
              << options.communicationTopologyReducedInterfaceWeight;
    std::cout << " comm_topology_reduced_interface_max_local_iters="
              << options.communicationTopologyReducedInterfaceMaxLocalIterations;
    std::cout << " comm_topology_reduced_interface_candidate="
              << (options.communicationTopologyReducedInterfaceCandidate
                      ? "true"
                      : "false");
    std::cout << " stale_boundary_proximal="
              << (options.staleBoundaryProximal ? "true" : "false");
    std::cout << " stale_boundary_proximal_weight="
              << options.staleBoundaryProximalWeight;
    std::cout << " stale_boundary_prediction="
              << (options.staleBoundaryPrediction ? "true" : "false");
    std::cout << " stale_boundary_prediction_gain="
              << options.staleBoundaryPredictionGain;
    std::cout << " sync_amm_reference_after_local_gradient_correction="
              << (options.syncAmmReferenceAfterLocalGradientCorrection
                      ? "true"
                      : "false");
    std::cout << " global_gradient_correction="
              << (options.globalGradientCorrection ? "true" : "false")
              << " global_gradient_correction_step="
              << options.globalGradientCorrectionStep;
    if (!options.globalGradientCorrectionSteps.empty()) {
      std::cout << " global_gradient_correction_steps=";
      for (std::size_t idx = 0;
           idx < options.globalGradientCorrectionSteps.size(); ++idx) {
        if (idx > 0) {
          std::cout << ",";
        }
        std::cout << options.globalGradientCorrectionSteps[idx];
      }
    }
    std::cout << " reset_local_history_after_global_correction="
              << (options.resetLocalHistoryAfterGlobalCorrection ? "true"
                                                                  : "false")
              << " sync_amm_reference_after_global_correction="
              << (options.syncAmmReferenceAfterGlobalCorrection ? "true"
                                                                 : "false");
    std::cout << " amm_eta0=" << options.ammEta0
              << " amm_eta1=" << options.ammEta1
              << " amm_psi=" << options.ammPsi
              << " amm_phi=" << options.ammPhi
              << " amm_gamma_scale=" << options.ammGammaScale
              << " amm_accepted_delta=" << options.ammAcceptedDelta
              << " amm_oscillation_count_period="
              << options.ammOscillationCountPeriod
              << " amm_max_oscillations=" << options.ammMaxOscillations;
    if (!options.ammGammaScales.empty()) {
      std::cout << " amm_gamma_scales=";
      for (std::size_t idx = 0; idx < options.ammGammaScales.size(); ++idx) {
        if (idx > 0) {
          std::cout << ",";
        }
        std::cout << options.ammGammaScales[idx];
      }
    }
    std::cout << " amm_proximal_start="
              << (options.ammProximalStart ? "true" : "false")
              << " amm_dpgo_restart_fallback="
              << (options.ammDpgoRestartFallback ? "true" : "false")
              << " amm_dpgo_surrogate_parity="
              << (options.ammDpgoSurrogateParity ? "true" : "false")
              << " amm_baseline_surrogate_state="
              << (options.ammBaselineSurrogateState ? "true" : "false")
              << " amm_dpgo_recursive_simple_state="
              << (options.ammDpgoRecursiveSimpleState ? "true" : "false")
              << " amm_recursive_simple_reanchor_period="
              << options.ammRecursiveSimpleReanchorPeriod
              << " amm_dpgo_strict_refined_gate="
              << (options.ammDpgoStrictRefinedGate ? "true" : "false")
              << " amm_dpgo_recover_translations_after_proximal="
              << (options.ammDpgoRecoverTranslationsAfterProximal ? "true"
                                                                  : "false")
              << " amm_dpgo_refined_starts_at_recovered_proximal="
              << (options.ammDpgoRefinedStartsAtRecoveredProximal ? "true"
                                                                  : "false")
              << " amm_dpgo_mixed_surrogate_portfolio="
              << (options.ammDpgoMixedSurrogatePortfolio ? "true" : "false")
              << " amm_mixed_surrogate_skip_simple_after_true_local_streak="
              << options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak
              << " amm_mixed_surrogate_force_simple_every_skipped_rounds="
              << options.ammMixedSurrogateForceSimpleEverySkippedRounds
              << " amm_cooldown_after_rejected="
              << options.ammCooldownAfterRejected
              << " amm_prox_reset_skip_refined_solve="
              << (options.ammProxResetSkipRefinedSolve ? "true" : "false")
              << " amm_dpgo_proximal_fallback_only="
              << (options.ammDpgoProximalFallbackOnly ? "true" : "false")
              << " amm_lazy_plain_after_certificate="
              << (options.ammLazyPlainAfterCertificate ? "true" : "false")
              << " amm_surrogate_first_exact_evaluation="
              << (options.ammSurrogateFirstExactEvaluation ? "true"
                                                           : "false")
              << " amm_max_soft_restart_hits0="
              << options.ammMaxSoftRestartHits0
              << " amm_max_soft_restart_hits1="
              << options.ammMaxSoftRestartHits1
              << " amm_local_merit_filter="
              << (options.ammLocalMeritFilter ? "true" : "false")
              << " amm_local_merit_cost_tie_tolerance="
              << options.ammLocalMeritCostTieTolerance
              << " trace_amm=" << (options.traceAmm ? "true" : "false")
              << " profile_optimizer="
              << (options.profileOptimizer ? "true" : "false")
              << " candidate_evaluation_cache="
              << (options.candidateEvaluationCache ? "true" : "false")
              << " candidate_object_reuse="
              << (options.candidateObjectReuse ? "true" : "false")
              << " reduced_adaptive_portfolio_certified_fast_path="
              << (options.reducedAdaptivePortfolioCertifiedFastPath ? "true"
                                                                     : "false")
              << " reduced_rotation_direct_objective="
              << (options.reducedRotationDirectObjective ? "true" : "false")
              << " fused_candidate_evaluation="
              << (options.fusedCandidateEvaluation ? "true" : "false")
              << " lazy_candidate_gradient_evaluation="
              << (options.lazyCandidateGradientEvaluation ? "true" : "false")
              << " lazy_solver_start_gradient_evaluation="
              << (options.lazySolverStartGradientEvaluation ? "true"
                                                            : "false")
              << " lazy_surrogate_candidate_gradient_evaluation="
              << (options.lazySurrogateCandidateGradientEvaluation ? "true"
                                                                   : "false")
              << " reduced_rotation_curvature_cauchy_candidate="
              << (options.reducedRotationCurvatureCauchyCandidate ? "true"
                                                                  : "false")
              << " reduced_rotation_curvature_fallback_candidate="
              << (options.reducedRotationCurvatureFallbackCandidate ? "true"
                                                                    : "false")
              << " reduced_rotation_gradient_boundary_candidate="
              << (options.reducedRotationGradientBoundaryCandidate ? "true"
                                                                   : "false")
              << " reduced_rotation_surrogate_tcg_accept="
              << (options.reducedRotationSurrogateTcgAccept ? "true"
                                                            : "false")
              << " reduced_rotation_skip_redundant_candidate_projection="
              << (options.reducedRotationSkipRedundantCandidateProjection
                      ? "true"
                      : "false")
              << " print_iteration_summary="
              << (options.printIterationSummary ? "true" : "false")
              << " manual_full_portfolio="
              << (options.manualFullPortfolio ? "true" : "false")
              << " edge_split_theta_candidates=";
    for (std::size_t idx = 0; idx < options.edgeSplitThetaCandidates.size();
         ++idx) {
      if (idx > 0) {
        std::cout << ",";
      }
      std::cout << options.edgeSplitThetaCandidates[idx];
    }
    std::cout << std::endl;
    const auto result = DPGO::runManualDpgoMm(dataset, options);
    std::cout << std::setprecision(20);
    std::cout << "final objective: " << result.finalObjective << std::endl;
    std::cout << "final gradient: " << result.finalGradient << std::endl;
    if (options.debugSurrogateBoundCheck) {
      std::cout << "surrogate bound checks: "
                << result.surrogateBoundCheckCount
                << " violations: " << result.surrogateBoundViolationCount
                << " min margin: " << result.surrogateBoundMinMargin
                << std::endl;
    }
    std::cout << "time: " << result.elapsedSeconds << " s/node."
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "manual_dpgo_mm failed: " << e.what() << std::endl;
    return 2;
  }

  return 0;
}

#!/usr/bin/env bash
set -euo pipefail

# Runs the 5-robot multi-robot example on six benchmark datasets and records:
# 1) raw stdout/stderr logs, 2) per-iteration CSVs, 3) final summary CSV.
# Default method favors faster loss decrease with a modest communication
# increase: residual-aware colored decentralized RTR.
# Dataset aliases:
#   inter     -> data/input_INTEL_g2o.g2o
#   manhattan -> data/input_M3500_g2o.g2o

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/bin/multi-robot-example}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/residual_colored_six_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"
DATASET_FILTER="${DATASETS:-}"

NUM_ROBOTS="${NUM_ROBOTS:-5}"
MAX_ITERS="${MAX_ITERS:-300}"
GRAD_TOL="${GRAD_TOL:-0.01}"
LOSS_TOL="${LOSS_TOL:-0.0002}"
RTR_ITERS="${RTR_ITERS:-1}"
RTR_MAX_INNER="${RTR_MAX_INNER:-50}"
RTR_TOL="${RTR_TOL:-0.0001}"
RTR_RADIUS="${RTR_RADIUS:-100}"
STABLE_LOSS_ROUNDS="${STABLE_LOSS_ROUNDS:-5}"
UPDATE_MODE="${UPDATE_MODE:-decentralized_residual_async_schur_late_feedback}"
DECENTRALIZED_REFRESH_PERIOD="${DECENTRALIZED_REFRESH_PERIOD:-1}"
EVENT_POSE_TOL="${EVENT_POSE_TOL:-0.001}"
EVENT_MAX_AGE="${EVENT_MAX_AGE:-7}"
EVENT_LOCAL_GRAD_TOL="${EVENT_LOCAL_GRAD_TOL:-0.02}"
EVENT_BUDGET_FRACTION="${EVENT_BUDGET_FRACTION:-0.8}"
EVENT_MAX_POSES_PER_NEIGHBOR="${EVENT_MAX_POSES_PER_NEIGHBOR:-0}"
LOCAL_ALGORITHM="${LOCAL_ALGORITHM:-RTR}"
LOCAL_SOLVER="${LOCAL_SOLVER:-${DRAN_LOCAL_SOLVER:-roptlib}}"
RELAXATION_RANK="${RELAXATION_RANK:-${DRAN_RELAXATION_RANK:-}}"
LOCAL_AMM_COMPARE_BASELINE="${LOCAL_AMM_COMPARE_BASELINE:-${DRAN_LOCAL_AMM_COMPARE_BASELINE:-false}}"
LOCAL_AMM_REQUIRE_BEAT_BASELINE="${LOCAL_AMM_REQUIRE_BEAT_BASELINE:-${DRAN_LOCAL_AMM_REQUIRE_BEAT_BASELINE:-true}}"
LOCAL_AMM_REQUIRE_DECREASE="${LOCAL_AMM_REQUIRE_DECREASE:-${DRAN_LOCAL_AMM_REQUIRE_DECREASE:-true}}"
LOCAL_AMM_REQUIRE_GRAD_NONINCREASE="${LOCAL_AMM_REQUIRE_GRAD_NONINCREASE:-${DRAN_LOCAL_AMM_REQUIRE_GRAD_NONINCREASE:-false}}"
EVENT_RESIDUAL_TOL="${EVENT_RESIDUAL_TOL:-0.0001}"
ACCELERATION="${ACCELERATION:-0}"
NEIGHBOR_DIRECTION_GAIN="${NEIGHBOR_DIRECTION_GAIN:-0.5}"
HYBRID_WARMUP_ROUNDS="${HYBRID_WARMUP_ROUNDS:-0}"
ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR="${ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR:-${DRAN_ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR:-1}}"
ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL="${ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL:-${DRAN_ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL:-${EVENT_RESIDUAL_TOL}}}"
ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES="${ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES:-${DRAN_ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES:-1}}"
ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS="${ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS:-${DRAN_ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS:-0}}"
BOUNDARY_JACOBI_STEP="${BOUNDARY_JACOBI_STEP:-${DRAN_BOUNDARY_JACOBI_STEP:-0.2}}"
BOUNDARY_JACOBI_MAX_BLOCK_NORM="${BOUNDARY_JACOBI_MAX_BLOCK_NORM:-${DRAN_BOUNDARY_JACOBI_MAX_BLOCK_NORM:-0.02}}"
BOUNDARY_JACOBI_REQUIRE_DECREASE="${BOUNDARY_JACOBI_REQUIRE_DECREASE:-${DRAN_BOUNDARY_JACOBI_REQUIRE_DECREASE:-1}}"
BOUNDARY_JACOBI_BACKTRACKING_STEPS="${BOUNDARY_JACOBI_BACKTRACKING_STEPS:-${DRAN_BOUNDARY_JACOBI_BACKTRACKING_STEPS:-5}}"
BOUNDARY_MODEL_STEP="${BOUNDARY_MODEL_STEP:-${DRAN_BOUNDARY_MODEL_STEP:-0.2}}"
BOUNDARY_MODEL_MAX_BLOCK_NORM="${BOUNDARY_MODEL_MAX_BLOCK_NORM:-${DRAN_BOUNDARY_MODEL_MAX_BLOCK_NORM:-0.02}}"
BOUNDARY_MODEL_REQUIRE_DECREASE="${BOUNDARY_MODEL_REQUIRE_DECREASE:-${DRAN_BOUNDARY_MODEL_REQUIRE_DECREASE:-1}}"
BOUNDARY_MODEL_BACKTRACKING_STEPS="${BOUNDARY_MODEL_BACKTRACKING_STEPS:-${DRAN_BOUNDARY_MODEL_BACKTRACKING_STEPS:-5}}"
BOUNDARY_MODEL_GAIN="${BOUNDARY_MODEL_GAIN:-${DRAN_BOUNDARY_MODEL_GAIN:-1.0}}"
DEFAULT_BOUNDARY_PACKET_STEP_MODE="gradient"
if [[ "${UPDATE_MODE}" == "decentralized_boundary_precond_packet_async_schur_late_feedback" ]]; then
  DEFAULT_BOUNDARY_PACKET_STEP_MODE="preconditioned"
fi
BOUNDARY_PACKET_STEP_MODE="${BOUNDARY_PACKET_STEP_MODE:-${DRAN_BOUNDARY_PACKET_STEP_MODE:-${DEFAULT_BOUNDARY_PACKET_STEP_MODE}}}"
BOUNDARY_BATCH_RESPONSE_STEP="${BOUNDARY_BATCH_RESPONSE_STEP:-${DRAN_BOUNDARY_BATCH_RESPONSE_STEP:-${DRAN_BOUNDARY_RESPONSE_STEP:-${BOUNDARY_MODEL_STEP}}}}"
BOUNDARY_BATCH_RESPONSE_GAIN="${BOUNDARY_BATCH_RESPONSE_GAIN:-${DRAN_BOUNDARY_BATCH_RESPONSE_GAIN:-${DRAN_BOUNDARY_RESPONSE_GAIN:-1.0}}}"
BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM="${BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM:-${DRAN_BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM:-${DRAN_BOUNDARY_RESPONSE_MAX_BLOCK_NORM:-${BOUNDARY_MODEL_MAX_BLOCK_NORM}}}}"
BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE="${BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE:-${DRAN_BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE:-${DRAN_BOUNDARY_RESPONSE_REQUIRE_DECREASE:-1}}}"
BOUNDARY_BATCH_RESPONSE_BACKTRACKING_STEPS="${BOUNDARY_BATCH_RESPONSE_BACKTRACKING_STEPS:-${DRAN_BOUNDARY_BATCH_RESPONSE_BACKTRACKING_STEPS:-${DRAN_BOUNDARY_RESPONSE_BACKTRACKING_STEPS:-${BOUNDARY_MODEL_BACKTRACKING_STEPS}}}}"
BOUNDARY_BATCH_RESPONSE_POST_REFINE="${BOUNDARY_BATCH_RESPONSE_POST_REFINE:-${DRAN_BOUNDARY_BATCH_RESPONSE_POST_REFINE:-0}}"
BOUNDARY_SCHUR_RESPONSE_STEP="${BOUNDARY_SCHUR_RESPONSE_STEP:-${DRAN_BOUNDARY_SCHUR_RESPONSE_STEP:-${BOUNDARY_BATCH_RESPONSE_STEP}}}"
BOUNDARY_SCHUR_RESPONSE_GAIN="${BOUNDARY_SCHUR_RESPONSE_GAIN:-${DRAN_BOUNDARY_SCHUR_RESPONSE_GAIN:-${BOUNDARY_BATCH_RESPONSE_GAIN}}}"
BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM="${BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM:-${DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM:-${BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM}}}"
BOUNDARY_SCHUR_RESPONSE_DAMPING="${BOUNDARY_SCHUR_RESPONSE_DAMPING:-${DRAN_BOUNDARY_SCHUR_RESPONSE_DAMPING:-0.01}}"
BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS="${BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS:-${DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS:-80}}"
BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE="${BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE:-${DRAN_BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE:-${BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE}}}"
BOUNDARY_SCHUR_RESPONSE_BACKTRACKING_STEPS="${BOUNDARY_SCHUR_RESPONSE_BACKTRACKING_STEPS:-${DRAN_BOUNDARY_SCHUR_RESPONSE_BACKTRACKING_STEPS:-${BOUNDARY_BATCH_RESPONSE_BACKTRACKING_STEPS}}}"
BOUNDARY_SCHUR_RESPONSE_POST_REFINE="${BOUNDARY_SCHUR_RESPONSE_POST_REFINE:-${DRAN_BOUNDARY_SCHUR_RESPONSE_POST_REFINE:-${BOUNDARY_BATCH_RESPONSE_POST_REFINE}}}"
BOUNDARY_SCHUR_RESPONSE_SKIP_BASELINE_REFINE="${BOUNDARY_SCHUR_RESPONSE_SKIP_BASELINE_REFINE:-${DRAN_BOUNDARY_SCHUR_RESPONSE_SKIP_BASELINE_REFINE:-0}}"
BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_PERIOD="${BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_PERIOD:-${DRAN_BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_PERIOD:-0}}"
BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_OFFSET="${BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_OFFSET:-${DRAN_BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_OFFSET:-0}}"
BOUNDARY_PACKET_SCHUR_DAMPING_GAIN="${BOUNDARY_PACKET_SCHUR_DAMPING_GAIN:-${DRAN_BOUNDARY_PACKET_SCHUR_DAMPING_GAIN:-0.01}}"
BOUNDARY_PACKET_SCHUR_DAMPING_MAX="${BOUNDARY_PACKET_SCHUR_DAMPING_MAX:-${DRAN_BOUNDARY_PACKET_SCHUR_DAMPING_MAX:-1.0}}"
BOUNDARY_BATCH_BUDGET_FRACTION="${BOUNDARY_BATCH_BUDGET_FRACTION:-${DRAN_BOUNDARY_BATCH_BUDGET_FRACTION:-1.0}}"
BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER="${BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER:-${DRAN_BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER:-0}}"
INTERFACE_DELTA_TOPK="${INTERFACE_DELTA_TOPK:-${DRAN_INTERFACE_DELTA_TOPK:-4}}"
INTERFACE_DELTA_MAX_RECON_ERROR="${INTERFACE_DELTA_MAX_RECON_ERROR:-${DRAN_INTERFACE_DELTA_MAX_RECON_ERROR:-0.0005}}"
INTERFACE_DELTA_CODEC="${INTERFACE_DELTA_CODEC:-${DRAN_INTERFACE_DELTA_CODEC:-hybrid}}"

export DRAN_ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR="${ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR}"
export DRAN_ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL="${ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL}"
export DRAN_ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES="${ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES}"
export DRAN_ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS="${ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS}"
export DRAN_BOUNDARY_JACOBI_STEP="${BOUNDARY_JACOBI_STEP}"
export DRAN_BOUNDARY_JACOBI_MAX_BLOCK_NORM="${BOUNDARY_JACOBI_MAX_BLOCK_NORM}"
export DRAN_BOUNDARY_JACOBI_REQUIRE_DECREASE="${BOUNDARY_JACOBI_REQUIRE_DECREASE}"
export DRAN_BOUNDARY_JACOBI_BACKTRACKING_STEPS="${BOUNDARY_JACOBI_BACKTRACKING_STEPS}"
export DRAN_BOUNDARY_MODEL_STEP="${BOUNDARY_MODEL_STEP}"
export DRAN_BOUNDARY_MODEL_MAX_BLOCK_NORM="${BOUNDARY_MODEL_MAX_BLOCK_NORM}"
export DRAN_BOUNDARY_MODEL_REQUIRE_DECREASE="${BOUNDARY_MODEL_REQUIRE_DECREASE}"
export DRAN_BOUNDARY_MODEL_BACKTRACKING_STEPS="${BOUNDARY_MODEL_BACKTRACKING_STEPS}"
export DRAN_BOUNDARY_MODEL_GAIN="${BOUNDARY_MODEL_GAIN}"
export DRAN_BOUNDARY_PACKET_STEP_MODE="${BOUNDARY_PACKET_STEP_MODE}"
export DRAN_BOUNDARY_BATCH_RESPONSE_STEP="${BOUNDARY_BATCH_RESPONSE_STEP}"
export DRAN_BOUNDARY_BATCH_RESPONSE_GAIN="${BOUNDARY_BATCH_RESPONSE_GAIN}"
export DRAN_BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM="${BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM}"
export DRAN_BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE="${BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE}"
export DRAN_BOUNDARY_BATCH_RESPONSE_BACKTRACKING_STEPS="${BOUNDARY_BATCH_RESPONSE_BACKTRACKING_STEPS}"
export DRAN_BOUNDARY_BATCH_RESPONSE_POST_REFINE="${BOUNDARY_BATCH_RESPONSE_POST_REFINE}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_STEP="${BOUNDARY_SCHUR_RESPONSE_STEP}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_GAIN="${BOUNDARY_SCHUR_RESPONSE_GAIN}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM="${BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_DAMPING="${BOUNDARY_SCHUR_RESPONSE_DAMPING}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS="${BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE="${BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_BACKTRACKING_STEPS="${BOUNDARY_SCHUR_RESPONSE_BACKTRACKING_STEPS}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_POST_REFINE="${BOUNDARY_SCHUR_RESPONSE_POST_REFINE}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_SKIP_BASELINE_REFINE="${BOUNDARY_SCHUR_RESPONSE_SKIP_BASELINE_REFINE}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_PERIOD="${BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_PERIOD}"
export DRAN_BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_OFFSET="${BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_OFFSET}"
export DRAN_BOUNDARY_PACKET_SCHUR_DAMPING_GAIN="${BOUNDARY_PACKET_SCHUR_DAMPING_GAIN}"
export DRAN_BOUNDARY_PACKET_SCHUR_DAMPING_MAX="${BOUNDARY_PACKET_SCHUR_DAMPING_MAX}"
export DRAN_BOUNDARY_BATCH_BUDGET_FRACTION="${BOUNDARY_BATCH_BUDGET_FRACTION}"
export DRAN_BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER="${BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER}"
export DRAN_INTERFACE_DELTA_TOPK="${INTERFACE_DELTA_TOPK}"
export DRAN_INTERFACE_DELTA_MAX_RECON_ERROR="${INTERFACE_DELTA_MAX_RECON_ERROR}"
export DRAN_INTERFACE_DELTA_CODEC="${INTERFACE_DELTA_CODEC}"
export DRAN_LOCAL_SOLVER="${LOCAL_SOLVER}"
if [[ -n "${RELAXATION_RANK}" ]]; then
  export DRAN_RELAXATION_RANK="${RELAXATION_RANK}"
fi
export DRAN_LOCAL_AMM_COMPARE_BASELINE="${LOCAL_AMM_COMPARE_BASELINE}"
export DRAN_LOCAL_AMM_REQUIRE_BEAT_BASELINE="${LOCAL_AMM_REQUIRE_BEAT_BASELINE}"
export DRAN_LOCAL_AMM_REQUIRE_DECREASE="${LOCAL_AMM_REQUIRE_DECREASE}"
export DRAN_LOCAL_AMM_REQUIRE_GRAD_NONINCREASE="${LOCAL_AMM_REQUIRE_GRAD_NONINCREASE}"

declare -A DATASET_PATHS=(
  ["parking-garage"]="data/parking-garage.g2o"
  ["sphere"]="data/sphere2500.g2o"
  ["torus"]="data/torus3D.g2o"
  ["CSAIL"]="data/CSAIL.g2o"
  ["inter"]="data/input_INTEL_g2o.g2o"
  ["manhattan"]="data/input_M3500_g2o.g2o"
)

DATASET_ORDER=("parking-garage" "sphere" "torus" "CSAIL" "inter" "manhattan")
if [[ -n "${DATASET_FILTER}" ]]; then
  IFS=',' read -r -a DATASET_ORDER <<< "${DATASET_FILTER}"
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  for dataset in "${DATASET_ORDER[@]}"; do
    rel_path="${DATASET_PATHS[${dataset}]:-}"
    if [[ -z "${rel_path}" ]]; then
      echo "DRY_RUN unknown_dataset=${dataset}"
      continue
    fi
    echo "DRY_RUN dataset=${dataset} path=${rel_path} bin=${BIN} num_robots=${NUM_ROBOTS} max_iters=${MAX_ITERS} relaxation_rank=${RELAXATION_RANK:-default} grad_tol=${GRAD_TOL} loss_tol=${LOSS_TOL} stable_loss_rounds=${STABLE_LOSS_ROUNDS} rtr_iters=${RTR_ITERS} rtr_max_inner=${RTR_MAX_INNER} rtr_tol=${RTR_TOL} rtr_radius=${RTR_RADIUS} update_mode=${UPDATE_MODE} decentralized_refresh_period=${DECENTRALIZED_REFRESH_PERIOD} event_pose_tol=${EVENT_POSE_TOL} event_max_age=${EVENT_MAX_AGE} event_local_grad_tol=${EVENT_LOCAL_GRAD_TOL} event_budget_fraction=${EVENT_BUDGET_FRACTION} event_max_poses_per_neighbor=${EVENT_MAX_POSES_PER_NEIGHBOR} local_algorithm=${LOCAL_ALGORITHM} local_solver=${LOCAL_SOLVER} local_amm_compare_baseline=${LOCAL_AMM_COMPARE_BASELINE} local_amm_require_beat_baseline=${LOCAL_AMM_REQUIRE_BEAT_BASELINE} local_amm_require_decrease=${LOCAL_AMM_REQUIRE_DECREASE} local_amm_require_grad_nonincrease=${LOCAL_AMM_REQUIRE_GRAD_NONINCREASE} hybrid_warmup_rounds=${HYBRID_WARMUP_ROUNDS} adaptive_extra_rtr=${ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR} adaptive_residual_tol=${ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL} adaptive_min_separator_poses=${ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES} adaptive_max_robots=${ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS} boundary_jacobi_step=${BOUNDARY_JACOBI_STEP} boundary_jacobi_max_block_norm=${BOUNDARY_JACOBI_MAX_BLOCK_NORM} boundary_jacobi_require_decrease=${BOUNDARY_JACOBI_REQUIRE_DECREASE} boundary_model_step=${BOUNDARY_MODEL_STEP} boundary_model_max_block_norm=${BOUNDARY_MODEL_MAX_BLOCK_NORM} boundary_model_require_decrease=${BOUNDARY_MODEL_REQUIRE_DECREASE} boundary_model_gain=${BOUNDARY_MODEL_GAIN} boundary_packet_step_mode=${BOUNDARY_PACKET_STEP_MODE} boundary_batch_response_step=${BOUNDARY_BATCH_RESPONSE_STEP} boundary_batch_response_gain=${BOUNDARY_BATCH_RESPONSE_GAIN} boundary_batch_response_max_block_norm=${BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM} boundary_batch_response_require_decrease=${BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE} boundary_schur_response_step=${BOUNDARY_SCHUR_RESPONSE_STEP} boundary_schur_response_gain=${BOUNDARY_SCHUR_RESPONSE_GAIN} boundary_schur_response_max_block_norm=${BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM} boundary_schur_response_damping=${BOUNDARY_SCHUR_RESPONSE_DAMPING} boundary_schur_response_max_blocks=${BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS} boundary_schur_response_require_decrease=${BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE} boundary_packet_schur_damping_gain=${BOUNDARY_PACKET_SCHUR_DAMPING_GAIN} boundary_packet_schur_damping_max=${BOUNDARY_PACKET_SCHUR_DAMPING_MAX} boundary_batch_budget_fraction=${BOUNDARY_BATCH_BUDGET_FRACTION} boundary_batch_max_poses_per_receiver=${BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER} interface_delta_codec=${INTERFACE_DELTA_CODEC} interface_delta_topk=${INTERFACE_DELTA_TOPK} interface_delta_max_recon_error=${INTERFACE_DELTA_MAX_RECON_ERROR}"
  done
  exit 0
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build it first, for example: cmake -S . -B build && make -C build multi-robot-example" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/iterations" "${OUT_ROOT}/poses"

SUMMARY_CSV="${OUT_ROOT}/final_summary.csv"

printf "dataset,status,final_iter,final_cost,final_gradnorm,total_comm_poses,total_comm_mb,loss_tol_iter,loss_tol_comm_poses,loss_tol_comm_mb,grad_tol_iter,grad_tol_comm_poses,grad_tol_comm_mb,convergence_iter,convergence_comm_poses,convergence_comm_mb,cumulative_parallel_compute_ms,adaptive_total_refine_robots,adaptive_total_refine_calls,adaptive_total_refine_ms,local_amm_total_attempted,local_amm_total_accepted,local_amm_total_rejected,local_amm_total_cost_decrease,local_amm_total_grad_decrease,local_amm_total_step_norm,local_amm_total_ms,local_amm_total_baseline_compared,local_amm_total_selected,local_amm_total_baseline_selected,local_amm_total_vs_baseline_cost_delta,local_amm_total_vs_baseline_grad_delta,majorized_boundary_total_candidates,majorized_boundary_total_selected,majorized_boundary_total_score,majorized_boundary_total_selected_score,boundary_jacobi_total_attempted,boundary_jacobi_total_accepted,boundary_jacobi_total_rejected,boundary_jacobi_total_blocks,boundary_jacobi_total_step_norm,boundary_jacobi_total_cost_decrease,boundary_jacobi_total_ms,boundary_model_total_predictions,boundary_model_total_computed_step_norm,boundary_model_total_computed_cost_decrease,boundary_model_total_compute_ms,boundary_model_total_merit_evaluations,boundary_model_total_merit_accepted,boundary_model_total_merit_rejected,boundary_model_total_merit_decrease,boundary_model_total_merit_grad_decrease,boundary_model_total_merit_ms,boundary_model_packet_total_payload_blocks,boundary_model_packet_total_model_pose_blocks,boundary_model_packet_total_grad_norm,boundary_model_packet_total_stiffness_sum,boundary_model_packet_total_freshness_sum,boundary_model_packet_total_preconditioned_step_norm,boundary_model_packet_total_schur_sensitivity_sum,boundary_model_packet_total_reduced_preconditioner_sum,boundary_model_packet_total_comm_mb,boundary_model_packet_payload_bytes_per_block,boundary_response_total_attempted,boundary_response_total_accepted,boundary_response_total_rejected,boundary_response_total_blocks,boundary_response_total_step_norm,boundary_response_total_grad_delta_norm,boundary_response_total_cost_decrease,boundary_response_total_ms,boundary_batch_response_total_attempted,boundary_batch_response_total_accepted,boundary_batch_response_total_rejected,boundary_batch_response_total_blocks,boundary_batch_response_total_step_norm,boundary_batch_response_total_grad_delta_norm,boundary_batch_response_total_cost_decrease,boundary_batch_response_total_ms,boundary_batch_model_total_receivers,boundary_batch_model_total_candidate_poses,boundary_batch_model_total_merit_evaluations,boundary_batch_model_total_merit_accepted,boundary_batch_model_total_merit_rejected,boundary_batch_model_total_merit_decrease,boundary_batch_model_total_merit_grad_decrease,boundary_batch_model_total_refine_ms,boundary_batch_model_total_budget_candidates,boundary_batch_model_total_budget_selected,boundary_batch_model_total_budget_dropped,interface_state_total_payload_blocks,interface_state_payload_bytes_per_block,interface_state_total_comm_mb,interface_state_total_compute_ms,interface_delta_total_candidate_pose_blocks,interface_delta_total_cache_miss_pose_blocks,interface_delta_total_rejected_pose_blocks,interface_delta_total_sparse_pose_blocks,interface_delta_total_tangent_pose_blocks,interface_delta_total_full_pose_blocks,interface_delta_total_delta_pose_blocks,interface_delta_total_delta_entries,interface_delta_total_comm_mb,log_path,csv_path,pose_path\n" > "${SUMMARY_CSV}"

write_summary_row() {
  local row=("$@")
  local IFS=,
  printf "%s\n" "${row[*]}" >> "${SUMMARY_CSV}"
}

write_empty_summary_row() {
  local dataset="$1"
  local status="$2"
  local log_path="$3"
  local iter_csv="$4"
  local pose_path="$5"
  local row=("${dataset}" "${status}")
  for _ in {1..101}; do
    row+=("")
  done
  row+=("${log_path}" "${iter_csv}" "${pose_path}")
  write_summary_row "${row[@]}"
}

summary_value() {
  local key="$1"
  local line="$2"
  awk -v key="${key}" '{
    for (i = 1; i <= NF; ++i) {
      split($i, kv, "=")
      if (kv[1] == key) {
        print kv[2]
        exit
      }
    }
  }' <<< "${line}"
}

for dataset in "${DATASET_ORDER[@]}"; do
  rel_path="${DATASET_PATHS[${dataset}]:-}"
  if [[ -z "${rel_path}" ]]; then
    log_path="${OUT_ROOT}/logs/${dataset}.log"
    iter_csv="${OUT_ROOT}/iterations/${dataset}.csv"
    pose_path="${OUT_ROOT}/poses/${dataset}.txt"
    mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/iterations" "${OUT_ROOT}/poses"
    echo "[${dataset}] unknown dataset alias" | tee "${log_path}"
    write_empty_summary_row "${dataset}" "unknown_dataset" "${log_path}" \
      "${iter_csv}" "${pose_path}"
    continue
  fi
  data_path="${ROOT_DIR}/${rel_path}"
  log_path="${OUT_ROOT}/logs/${dataset}.log"
  iter_csv="${OUT_ROOT}/iterations/${dataset}.csv"
  pose_path="${OUT_ROOT}/poses/${dataset}.txt"

  if [[ ! -f "${data_path}" ]]; then
    echo "[${dataset}] missing dataset: ${data_path}" | tee "${log_path}"
    write_empty_summary_row "${dataset}" "missing_dataset" "${log_path}" \
      "${iter_csv}" "${pose_path}"
    continue
  fi

  echo "=== Running ${dataset}: ${rel_path} ==="
  echo "CONFIG bin=${BIN} num_robots=${NUM_ROBOTS} max_iters=${MAX_ITERS} relaxation_rank=${RELAXATION_RANK:-default} rtr_iters=${RTR_ITERS} rtr_max_inner=${RTR_MAX_INNER} rtr_tol=${RTR_TOL} update_mode=${UPDATE_MODE} local_algorithm=${LOCAL_ALGORITHM} local_solver=${LOCAL_SOLVER} local_amm_compare_baseline=${LOCAL_AMM_COMPARE_BASELINE} local_amm_require_beat_baseline=${LOCAL_AMM_REQUIRE_BEAT_BASELINE} local_amm_require_decrease=${LOCAL_AMM_REQUIRE_DECREASE} local_amm_require_grad_nonincrease=${LOCAL_AMM_REQUIRE_GRAD_NONINCREASE} hybrid_warmup_rounds=${HYBRID_WARMUP_ROUNDS} adaptive_extra_rtr=${ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR} adaptive_residual_tol=${ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL} adaptive_min_separator_poses=${ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES} adaptive_max_robots=${ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS} boundary_jacobi_step=${BOUNDARY_JACOBI_STEP} boundary_jacobi_max_block_norm=${BOUNDARY_JACOBI_MAX_BLOCK_NORM} boundary_jacobi_require_decrease=${BOUNDARY_JACOBI_REQUIRE_DECREASE} boundary_jacobi_backtracking_steps=${BOUNDARY_JACOBI_BACKTRACKING_STEPS} boundary_model_step=${BOUNDARY_MODEL_STEP} boundary_model_max_block_norm=${BOUNDARY_MODEL_MAX_BLOCK_NORM} boundary_model_require_decrease=${BOUNDARY_MODEL_REQUIRE_DECREASE} boundary_model_backtracking_steps=${BOUNDARY_MODEL_BACKTRACKING_STEPS} boundary_model_gain=${BOUNDARY_MODEL_GAIN} boundary_packet_step_mode=${BOUNDARY_PACKET_STEP_MODE} boundary_batch_response_step=${BOUNDARY_BATCH_RESPONSE_STEP} boundary_batch_response_gain=${BOUNDARY_BATCH_RESPONSE_GAIN} boundary_batch_response_max_block_norm=${BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM} boundary_batch_response_require_decrease=${BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE} boundary_batch_response_backtracking_steps=${BOUNDARY_BATCH_RESPONSE_BACKTRACKING_STEPS} boundary_batch_response_post_refine=${BOUNDARY_BATCH_RESPONSE_POST_REFINE} boundary_schur_response_step=${BOUNDARY_SCHUR_RESPONSE_STEP} boundary_schur_response_gain=${BOUNDARY_SCHUR_RESPONSE_GAIN} boundary_schur_response_max_block_norm=${BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM} boundary_schur_response_damping=${BOUNDARY_SCHUR_RESPONSE_DAMPING} boundary_schur_response_max_blocks=${BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS} boundary_schur_response_require_decrease=${BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE} boundary_schur_response_backtracking_steps=${BOUNDARY_SCHUR_RESPONSE_BACKTRACKING_STEPS} boundary_schur_response_post_refine=${BOUNDARY_SCHUR_RESPONSE_POST_REFINE} boundary_schur_response_skip_baseline_refine=${BOUNDARY_SCHUR_RESPONSE_SKIP_BASELINE_REFINE} boundary_schur_response_baseline_refine_period=${BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_PERIOD} boundary_schur_response_baseline_refine_offset=${BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_OFFSET} boundary_packet_schur_damping_gain=${BOUNDARY_PACKET_SCHUR_DAMPING_GAIN} boundary_packet_schur_damping_max=${BOUNDARY_PACKET_SCHUR_DAMPING_MAX} boundary_batch_budget_fraction=${BOUNDARY_BATCH_BUDGET_FRACTION} boundary_batch_max_poses_per_receiver=${BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER} interface_delta_codec=${INTERFACE_DELTA_CODEC} interface_delta_topk=${INTERFACE_DELTA_TOPK} interface_delta_max_recon_error=${INTERFACE_DELTA_MAX_RECON_ERROR}"
  set +e
  "${BIN}" "${NUM_ROBOTS}" "${data_path}" "${MAX_ITERS}" "${GRAD_TOL}" \
    "${LOSS_TOL}" "${iter_csv}" "${RTR_ITERS}" "${RTR_MAX_INNER}" \
    "${RTR_TOL}" "${RTR_RADIUS}" "${STABLE_LOSS_ROUNDS}" \
    "${UPDATE_MODE}" "${DECENTRALIZED_REFRESH_PERIOD}" "${EVENT_POSE_TOL}" \
    "${EVENT_MAX_AGE}" "${EVENT_LOCAL_GRAD_TOL}" \
    "${EVENT_BUDGET_FRACTION}" "${EVENT_MAX_POSES_PER_NEIGHBOR}" \
    "${LOCAL_ALGORITHM}" "${EVENT_RESIDUAL_TOL}" "${ACCELERATION}" \
    "${NEIGHBOR_DIRECTION_GAIN}" "${pose_path}" \
    "${HYBRID_WARMUP_ROUNDS}" 2>&1 | tee "${log_path}"
  run_status=${PIPESTATUS[0]}
  set -e

  summary_line="$(grep '^SUMMARY' "${log_path}" | tail -n 1 || true)"
  if [[ -z "${summary_line}" ]]; then
    write_empty_summary_row "${dataset}" "failed" "${log_path}" "${iter_csv}" \
      "${pose_path}"
    echo "[${dataset}] failed with status ${run_status}; no SUMMARY line found." >&2
    continue
  fi

  summary_row=(
    "${dataset}"
    "ok"
    "$(summary_value final_iter "${summary_line}")"
    "$(summary_value final_cost "${summary_line}")"
    "$(summary_value final_gradnorm "${summary_line}")"
    "$(summary_value total_comm_poses "${summary_line}")"
    "$(summary_value total_comm_mb "${summary_line}")"
    "$(summary_value loss_tol_iter "${summary_line}")"
    "$(summary_value loss_tol_comm_poses "${summary_line}")"
    "$(summary_value loss_tol_comm_mb "${summary_line}")"
    "$(summary_value grad_tol_iter "${summary_line}")"
    "$(summary_value grad_tol_comm_poses "${summary_line}")"
    "$(summary_value grad_tol_comm_mb "${summary_line}")"
    "$(summary_value convergence_iter "${summary_line}")"
    "$(summary_value convergence_comm_poses "${summary_line}")"
    "$(summary_value convergence_comm_mb "${summary_line}")"
    "$(summary_value cumulative_parallel_compute_ms "${summary_line}")"
    "$(summary_value adaptive_total_refine_robots "${summary_line}")"
    "$(summary_value adaptive_total_refine_calls "${summary_line}")"
    "$(summary_value adaptive_total_refine_ms "${summary_line}")"
    "$(summary_value local_amm_total_attempted "${summary_line}")"
    "$(summary_value local_amm_total_accepted "${summary_line}")"
    "$(summary_value local_amm_total_rejected "${summary_line}")"
    "$(summary_value local_amm_total_cost_decrease "${summary_line}")"
    "$(summary_value local_amm_total_grad_decrease "${summary_line}")"
    "$(summary_value local_amm_total_step_norm "${summary_line}")"
    "$(summary_value local_amm_total_ms "${summary_line}")"
    "$(summary_value local_amm_total_baseline_compared "${summary_line}")"
    "$(summary_value local_amm_total_selected "${summary_line}")"
    "$(summary_value local_amm_total_baseline_selected "${summary_line}")"
    "$(summary_value local_amm_total_vs_baseline_cost_delta "${summary_line}")"
    "$(summary_value local_amm_total_vs_baseline_grad_delta "${summary_line}")"
    "$(summary_value majorized_boundary_total_candidates "${summary_line}")"
    "$(summary_value majorized_boundary_total_selected "${summary_line}")"
    "$(summary_value majorized_boundary_total_score "${summary_line}")"
    "$(summary_value majorized_boundary_total_selected_score "${summary_line}")"
    "$(summary_value boundary_jacobi_total_attempted "${summary_line}")"
    "$(summary_value boundary_jacobi_total_accepted "${summary_line}")"
    "$(summary_value boundary_jacobi_total_rejected "${summary_line}")"
    "$(summary_value boundary_jacobi_total_blocks "${summary_line}")"
    "$(summary_value boundary_jacobi_total_step_norm "${summary_line}")"
    "$(summary_value boundary_jacobi_total_cost_decrease "${summary_line}")"
    "$(summary_value boundary_jacobi_total_ms "${summary_line}")"
    "$(summary_value boundary_model_total_predictions "${summary_line}")"
    "$(summary_value boundary_model_total_computed_step_norm "${summary_line}")"
    "$(summary_value boundary_model_total_computed_cost_decrease "${summary_line}")"
    "$(summary_value boundary_model_total_compute_ms "${summary_line}")"
    "$(summary_value boundary_model_total_merit_evaluations "${summary_line}")"
    "$(summary_value boundary_model_total_merit_accepted "${summary_line}")"
    "$(summary_value boundary_model_total_merit_rejected "${summary_line}")"
    "$(summary_value boundary_model_total_merit_decrease "${summary_line}")"
    "$(summary_value boundary_model_total_merit_grad_decrease "${summary_line}")"
    "$(summary_value boundary_model_total_merit_ms "${summary_line}")"
    "$(summary_value boundary_model_packet_total_payload_blocks "${summary_line}")"
    "$(summary_value boundary_model_packet_total_model_pose_blocks "${summary_line}")"
    "$(summary_value boundary_model_packet_total_grad_norm "${summary_line}")"
    "$(summary_value boundary_model_packet_total_stiffness_sum "${summary_line}")"
    "$(summary_value boundary_model_packet_total_freshness_sum "${summary_line}")"
    "$(summary_value boundary_model_packet_total_preconditioned_step_norm "${summary_line}")"
    "$(summary_value boundary_model_packet_total_schur_sensitivity_sum "${summary_line}")"
    "$(summary_value boundary_model_packet_total_reduced_preconditioner_sum "${summary_line}")"
    "$(summary_value boundary_model_packet_total_comm_mb "${summary_line}")"
    "$(summary_value boundary_model_packet_payload_bytes_per_block "${summary_line}")"
    "$(summary_value boundary_response_total_attempted "${summary_line}")"
    "$(summary_value boundary_response_total_accepted "${summary_line}")"
    "$(summary_value boundary_response_total_rejected "${summary_line}")"
    "$(summary_value boundary_response_total_blocks "${summary_line}")"
    "$(summary_value boundary_response_total_step_norm "${summary_line}")"
    "$(summary_value boundary_response_total_grad_delta_norm "${summary_line}")"
    "$(summary_value boundary_response_total_cost_decrease "${summary_line}")"
    "$(summary_value boundary_response_total_ms "${summary_line}")"
    "$(summary_value boundary_batch_response_total_attempted "${summary_line}")"
    "$(summary_value boundary_batch_response_total_accepted "${summary_line}")"
    "$(summary_value boundary_batch_response_total_rejected "${summary_line}")"
    "$(summary_value boundary_batch_response_total_blocks "${summary_line}")"
    "$(summary_value boundary_batch_response_total_step_norm "${summary_line}")"
    "$(summary_value boundary_batch_response_total_grad_delta_norm "${summary_line}")"
    "$(summary_value boundary_batch_response_total_cost_decrease "${summary_line}")"
    "$(summary_value boundary_batch_response_total_ms "${summary_line}")"
    "$(summary_value boundary_batch_model_total_receivers "${summary_line}")"
    "$(summary_value boundary_batch_model_total_candidate_poses "${summary_line}")"
    "$(summary_value boundary_batch_model_total_merit_evaluations "${summary_line}")"
    "$(summary_value boundary_batch_model_total_merit_accepted "${summary_line}")"
    "$(summary_value boundary_batch_model_total_merit_rejected "${summary_line}")"
    "$(summary_value boundary_batch_model_total_merit_decrease "${summary_line}")"
    "$(summary_value boundary_batch_model_total_merit_grad_decrease "${summary_line}")"
    "$(summary_value boundary_batch_model_total_refine_ms "${summary_line}")"
    "$(summary_value boundary_batch_model_total_budget_candidates "${summary_line}")"
    "$(summary_value boundary_batch_model_total_budget_selected "${summary_line}")"
    "$(summary_value boundary_batch_model_total_budget_dropped "${summary_line}")"
    "$(summary_value interface_state_total_payload_blocks "${summary_line}")"
    "$(summary_value interface_state_payload_bytes_per_block "${summary_line}")"
    "$(summary_value interface_state_total_comm_mb "${summary_line}")"
    "$(summary_value interface_state_total_compute_ms "${summary_line}")"
    "$(summary_value interface_delta_total_candidate_pose_blocks "${summary_line}")"
    "$(summary_value interface_delta_total_cache_miss_pose_blocks "${summary_line}")"
    "$(summary_value interface_delta_total_rejected_pose_blocks "${summary_line}")"
    "$(summary_value interface_delta_total_sparse_pose_blocks "${summary_line}")"
    "$(summary_value interface_delta_total_tangent_pose_blocks "${summary_line}")"
    "$(summary_value interface_delta_total_full_pose_blocks "${summary_line}")"
    "$(summary_value interface_delta_total_delta_pose_blocks "${summary_line}")"
    "$(summary_value interface_delta_total_delta_entries "${summary_line}")"
    "$(summary_value interface_delta_total_comm_mb "${summary_line}")"
    "${log_path}"
    "${iter_csv}"
    "${pose_path}"
  )
  write_summary_row "${summary_row[@]}"
done

echo "Saved final summary: ${SUMMARY_CSV}"
echo "Saved per-dataset logs and iteration CSVs under: ${OUT_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-${ROOT_DIR}/build/bin/manual-object-dran-example}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/manual_object_dran_drone_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"
GT_DIR="${GT_DIR:-}"

NUM_ROBOTS="${NUM_ROBOTS:-21}"
NUM_OBJECTS="${NUM_OBJECTS:-0}"
MAX_ITERS="${MAX_ITERS:-20}"
BETA="${BETA:-1}"
OBJECT_TOPOLOGY="${OBJECT_TOPOLOGY:-ring}"
OBJECT_RING_HOPS="${OBJECT_RING_HOPS:-1}"
COMM_TOPOLOGY_FILE="${COMM_TOPOLOGY_FILE:-}"
COMM_WEIGHT_MODE="${COMM_WEIGHT_MODE:-unit}"
OBJECT_COMMUNICATION_POLICY="${OBJECT_COMMUNICATION_POLICY:-triggered}"
OBJECT_INITIALIZATION="${OBJECT_INITIALIZATION:-centralized_chordal}"
OBJECT_INITIALIZATION_ANCHOR_MODE="${OBJECT_INITIALIZATION_ANCHOR_MODE:-constant}"
OBJECT_INITIALIZATION_CONSENSUS_ROUNDS="${OBJECT_INITIALIZATION_CONSENSUS_ROUNDS:-0}"
OBJECT_INITIALIZATION_OBSERVED_ANCHOR_WEIGHT="${OBJECT_INITIALIZATION_OBSERVED_ANCHOR_WEIGHT:-1}"
OBJECT_INITIALIZATION_RELAY_ANCHOR_WEIGHT="${OBJECT_INITIALIZATION_RELAY_ANCHOR_WEIGHT:-0}"
OBJECT_INTERFACE_MODE="${OBJECT_INTERFACE_MODE:-copy_baseline}"
OBJECT_INTERFACE_RESPONSE_STRATEGY="${OBJECT_INTERFACE_RESPONSE_STRATEGY:-serial}"
OBJECT_INTERFACE_RESPONSE_TRIGGER="${OBJECT_INTERFACE_RESPONSE_TRIGGER:-periodic}"
OBJECT_INTERFACE_RESPONSE_PERIOD="${OBJECT_INTERFACE_RESPONSE_PERIOD:-1}"
OBJECT_INTERFACE_MIN_PREDICTED_DECREASE="${OBJECT_INTERFACE_MIN_PREDICTED_DECREASE:-0}"
OBJECT_INTERFACE_MIN_INNOVATION_SCORE="${OBJECT_INTERFACE_MIN_INNOVATION_SCORE:-0}"
OBJECT_INTERFACE_SCHUR_DAMPING="${OBJECT_INTERFACE_SCHUR_DAMPING:-0.01}"
OBJECT_INTERFACE_STEP_GAIN="${OBJECT_INTERFACE_STEP_GAIN:-1}"
OBJECT_INTERFACE_MAX_OBJECTS_PER_ROBOT="${OBJECT_INTERFACE_MAX_OBJECTS_PER_ROBOT:-0}"
OBJECT_INTERFACE_MAX_PRIVATE_COLS="${OBJECT_INTERFACE_MAX_PRIVATE_COLS:-64}"
OBJECT_INTERFACE_MAX_BLOCK_STEP_NORM="${OBJECT_INTERFACE_MAX_BLOCK_STEP_NORM:-0.25}"
OBJECT_INTERFACE_REQUIRE_DECREASE="${OBJECT_INTERFACE_REQUIRE_DECREASE:-true}"
OBJECT_POSE_TOL="${OBJECT_POSE_TOL:-0.001}"
OBJECT_MAX_AGE="${OBJECT_MAX_AGE:-5}"
OBJECT_TRANSLATION_PROX_WEIGHT="${OBJECT_TRANSLATION_PROX_WEIGHT:-0}"
OBJECT_PROJECT_TO_SE_AFTER_LOCAL_SOLVE="${OBJECT_PROJECT_TO_SE_AFTER_LOCAL_SOLVE:-true}"
OBJECT_COUPLED_TRANSLATION_TRUST_REGION="${OBJECT_COUPLED_TRANSLATION_TRUST_REGION:-true}"
OBJECT_TRANSLATION_ELIMINATION_PROX_WEIGHT="${OBJECT_TRANSLATION_ELIMINATION_PROX_WEIGHT:-0}"
LOCAL_SOLVER="${LOCAL_SOLVER:-manual_full}"
OBJECT_FULL_SOLVER_OUTER_ITERS="${OBJECT_FULL_SOLVER_OUTER_ITERS:-0}"
OBJECT_LOCAL_TR_ITERS="${OBJECT_LOCAL_TR_ITERS:-1}"
OBJECT_LOCAL_TCG_ITERS="${OBJECT_LOCAL_TCG_ITERS:-20}"
OBJECT_LOCAL_TR_TOL="${OBJECT_LOCAL_TR_TOL:-0.001}"
OBJECT_TR_RADIUS="${OBJECT_TR_RADIUS:-10}"
REDUCED_ROTATION_PRECONDITIONER="${REDUCED_ROTATION_PRECONDITIONER:-none}"

if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
  cmd=(
    "${BIN}"
    --data_dir "${DATA_DIR}"
    --num_robots "${NUM_ROBOTS}"
    --num_objects "${NUM_OBJECTS}"
    --iters "${MAX_ITERS}"
    --beta "${BETA}"
    --topology "${OBJECT_TOPOLOGY}"
    --ring_hops "${OBJECT_RING_HOPS}"
    --topology_file "${COMM_TOPOLOGY_FILE}"
    --topology_weight_mode "${COMM_WEIGHT_MODE}"
    --communication_policy "${OBJECT_COMMUNICATION_POLICY}"
    --object_initialization "${OBJECT_INITIALIZATION}"
    --object_initialization_anchor_mode "${OBJECT_INITIALIZATION_ANCHOR_MODE}"
    --object_initialization_consensus_rounds "${OBJECT_INITIALIZATION_CONSENSUS_ROUNDS}"
    --object_initialization_observed_anchor_weight "${OBJECT_INITIALIZATION_OBSERVED_ANCHOR_WEIGHT}"
    --object_initialization_relay_anchor_weight "${OBJECT_INITIALIZATION_RELAY_ANCHOR_WEIGHT}"
    --object_interface_mode "${OBJECT_INTERFACE_MODE}"
    --object_interface_response_strategy "${OBJECT_INTERFACE_RESPONSE_STRATEGY}"
    --object_interface_response_trigger "${OBJECT_INTERFACE_RESPONSE_TRIGGER}"
    --object_interface_response_period "${OBJECT_INTERFACE_RESPONSE_PERIOD}"
    --object_interface_min_predicted_decrease "${OBJECT_INTERFACE_MIN_PREDICTED_DECREASE}"
    --object_interface_min_innovation_score "${OBJECT_INTERFACE_MIN_INNOVATION_SCORE}"
    --object_interface_schur_damping "${OBJECT_INTERFACE_SCHUR_DAMPING}"
    --object_interface_step_gain "${OBJECT_INTERFACE_STEP_GAIN}"
    --object_interface_max_objects_per_robot "${OBJECT_INTERFACE_MAX_OBJECTS_PER_ROBOT}"
    --object_interface_max_private_cols "${OBJECT_INTERFACE_MAX_PRIVATE_COLS}"
    --object_interface_max_block_step_norm "${OBJECT_INTERFACE_MAX_BLOCK_STEP_NORM}"
    --object_interface_require_decrease "${OBJECT_INTERFACE_REQUIRE_DECREASE}"
    --object_pose_tol "${OBJECT_POSE_TOL}"
    --object_max_age "${OBJECT_MAX_AGE}"
    --translation_prox_weight "${OBJECT_TRANSLATION_PROX_WEIGHT}"
    --project_to_se_after_local_solve "${OBJECT_PROJECT_TO_SE_AFTER_LOCAL_SOLVE}"
    --coupled_translation_trust_region "${OBJECT_COUPLED_TRANSLATION_TRUST_REGION}"
    --translation_elimination_prox_weight "${OBJECT_TRANSLATION_ELIMINATION_PROX_WEIGHT}"
    --local_solver "${LOCAL_SOLVER}"
    --full_solver_outer_iterations "${OBJECT_FULL_SOLVER_OUTER_ITERS}"
    --local_max_iterations "${OBJECT_LOCAL_TR_ITERS}"
    --local_max_tcg_iterations "${OBJECT_LOCAL_TCG_ITERS}"
    --local_grad_norm_tol "${OBJECT_LOCAL_TR_TOL}"
    --trust_region_initial_radius "${OBJECT_TR_RADIUS}"
    --reduced_rotation_preconditioner "${REDUCED_ROTATION_PRECONDITIONER}"
    --output_dir "${OUT_ROOT}"
    --save true
  )
  printf 'DRY_RUN data_dir=%s out_root=%s gt_dir=%s num_robots=%s num_objects=%s max_iters=%s beta=%s topology=%s ring_hops=%s comm_topology_file=%s comm_weight_mode=%s communication_policy=%s object_initialization=%s object_initialization_anchor_mode=%s object_initialization_consensus_rounds=%s object_initialization_observed_anchor_weight=%s object_initialization_relay_anchor_weight=%s object_interface_mode=%s object_interface_response_strategy=%s object_interface_response_trigger=%s object_interface_response_period=%s object_interface_min_predicted_decrease=%s object_interface_min_innovation_score=%s object_pose_tol=%s object_max_age=%s local_solver=%s reduced_rotation_preconditioner=%s command=' \
    "${DATA_DIR}" "${OUT_ROOT}" "${GT_DIR}" "${NUM_ROBOTS}" "${NUM_OBJECTS}" \
    "${MAX_ITERS}" "${BETA}" "${OBJECT_TOPOLOGY}" "${OBJECT_RING_HOPS}" \
    "${COMM_TOPOLOGY_FILE}" "${COMM_WEIGHT_MODE}" \
    "${OBJECT_COMMUNICATION_POLICY}" "${OBJECT_INITIALIZATION}" \
    "${OBJECT_INITIALIZATION_ANCHOR_MODE}" \
    "${OBJECT_INITIALIZATION_CONSENSUS_ROUNDS}" \
    "${OBJECT_INITIALIZATION_OBSERVED_ANCHOR_WEIGHT}" \
    "${OBJECT_INITIALIZATION_RELAY_ANCHOR_WEIGHT}" \
    "${OBJECT_INTERFACE_MODE}" "${OBJECT_INTERFACE_RESPONSE_STRATEGY}" \
    "${OBJECT_INTERFACE_RESPONSE_TRIGGER}" "${OBJECT_INTERFACE_RESPONSE_PERIOD}" \
    "${OBJECT_INTERFACE_MIN_PREDICTED_DECREASE}" \
    "${OBJECT_INTERFACE_MIN_INNOVATION_SCORE}" \
    "${OBJECT_POSE_TOL}" \
    "${OBJECT_MAX_AGE}" "${LOCAL_SOLVER}" "${REDUCED_ROTATION_PRECONDITIONER}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build it first: cmake -S . -B build && cmake --build build --target manual-object-dran-example -j2" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
LOG_PATH="${OUT_ROOT}/run.log"

echo "CONFIG bin=${BIN} data_dir=${DATA_DIR} num_robots=${NUM_ROBOTS} num_objects=${NUM_OBJECTS} max_iters=${MAX_ITERS} beta=${BETA} topology=${OBJECT_TOPOLOGY} ring_hops=${OBJECT_RING_HOPS} comm_topology_file=${COMM_TOPOLOGY_FILE} comm_weight_mode=${COMM_WEIGHT_MODE} communication_policy=${OBJECT_COMMUNICATION_POLICY} object_initialization=${OBJECT_INITIALIZATION} object_initialization_anchor_mode=${OBJECT_INITIALIZATION_ANCHOR_MODE} object_initialization_consensus_rounds=${OBJECT_INITIALIZATION_CONSENSUS_ROUNDS} object_initialization_observed_anchor_weight=${OBJECT_INITIALIZATION_OBSERVED_ANCHOR_WEIGHT} object_initialization_relay_anchor_weight=${OBJECT_INITIALIZATION_RELAY_ANCHOR_WEIGHT} object_interface_mode=${OBJECT_INTERFACE_MODE} object_interface_response_strategy=${OBJECT_INTERFACE_RESPONSE_STRATEGY} object_interface_response_trigger=${OBJECT_INTERFACE_RESPONSE_TRIGGER} object_interface_response_period=${OBJECT_INTERFACE_RESPONSE_PERIOD} object_interface_min_predicted_decrease=${OBJECT_INTERFACE_MIN_PREDICTED_DECREASE} object_interface_min_innovation_score=${OBJECT_INTERFACE_MIN_INNOVATION_SCORE} object_interface_schur_damping=${OBJECT_INTERFACE_SCHUR_DAMPING} object_interface_step_gain=${OBJECT_INTERFACE_STEP_GAIN} object_interface_max_objects_per_robot=${OBJECT_INTERFACE_MAX_OBJECTS_PER_ROBOT} object_interface_max_private_cols=${OBJECT_INTERFACE_MAX_PRIVATE_COLS} object_interface_max_block_step_norm=${OBJECT_INTERFACE_MAX_BLOCK_STEP_NORM} object_interface_require_decrease=${OBJECT_INTERFACE_REQUIRE_DECREASE} object_pose_tol=${OBJECT_POSE_TOL} object_max_age=${OBJECT_MAX_AGE} translation_prox_weight=${OBJECT_TRANSLATION_PROX_WEIGHT} project_to_se_after_local_solve=${OBJECT_PROJECT_TO_SE_AFTER_LOCAL_SOLVE} coupled_translation_trust_region=${OBJECT_COUPLED_TRANSLATION_TRUST_REGION} translation_elimination_prox_weight=${OBJECT_TRANSLATION_ELIMINATION_PROX_WEIGHT} local_solver=${LOCAL_SOLVER} full_solver_outer_iters=${OBJECT_FULL_SOLVER_OUTER_ITERS} local_tr_iters=${OBJECT_LOCAL_TR_ITERS} local_tcg_iters=${OBJECT_LOCAL_TCG_ITERS} local_tr_tol=${OBJECT_LOCAL_TR_TOL} tr_radius=${OBJECT_TR_RADIUS} reduced_rotation_preconditioner=${REDUCED_ROTATION_PRECONDITIONER}"

"${BIN}" \
  --data_dir "${DATA_DIR}" \
  --num_robots "${NUM_ROBOTS}" \
  --num_objects "${NUM_OBJECTS}" \
  --iters "${MAX_ITERS}" \
  --beta "${BETA}" \
  --topology "${OBJECT_TOPOLOGY}" \
  --ring_hops "${OBJECT_RING_HOPS}" \
  --topology_file "${COMM_TOPOLOGY_FILE}" \
  --topology_weight_mode "${COMM_WEIGHT_MODE}" \
  --communication_policy "${OBJECT_COMMUNICATION_POLICY}" \
  --object_initialization "${OBJECT_INITIALIZATION}" \
  --object_initialization_anchor_mode "${OBJECT_INITIALIZATION_ANCHOR_MODE}" \
  --object_initialization_consensus_rounds "${OBJECT_INITIALIZATION_CONSENSUS_ROUNDS}" \
  --object_initialization_observed_anchor_weight "${OBJECT_INITIALIZATION_OBSERVED_ANCHOR_WEIGHT}" \
  --object_initialization_relay_anchor_weight "${OBJECT_INITIALIZATION_RELAY_ANCHOR_WEIGHT}" \
  --object_interface_mode "${OBJECT_INTERFACE_MODE}" \
  --object_interface_response_strategy "${OBJECT_INTERFACE_RESPONSE_STRATEGY}" \
  --object_interface_response_trigger "${OBJECT_INTERFACE_RESPONSE_TRIGGER}" \
  --object_interface_response_period "${OBJECT_INTERFACE_RESPONSE_PERIOD}" \
  --object_interface_min_predicted_decrease "${OBJECT_INTERFACE_MIN_PREDICTED_DECREASE}" \
  --object_interface_min_innovation_score "${OBJECT_INTERFACE_MIN_INNOVATION_SCORE}" \
  --object_interface_schur_damping "${OBJECT_INTERFACE_SCHUR_DAMPING}" \
  --object_interface_step_gain "${OBJECT_INTERFACE_STEP_GAIN}" \
  --object_interface_max_objects_per_robot "${OBJECT_INTERFACE_MAX_OBJECTS_PER_ROBOT}" \
  --object_interface_max_private_cols "${OBJECT_INTERFACE_MAX_PRIVATE_COLS}" \
  --object_interface_max_block_step_norm "${OBJECT_INTERFACE_MAX_BLOCK_STEP_NORM}" \
  --object_interface_require_decrease "${OBJECT_INTERFACE_REQUIRE_DECREASE}" \
  --object_pose_tol "${OBJECT_POSE_TOL}" \
  --object_max_age "${OBJECT_MAX_AGE}" \
  --translation_prox_weight "${OBJECT_TRANSLATION_PROX_WEIGHT}" \
  --project_to_se_after_local_solve "${OBJECT_PROJECT_TO_SE_AFTER_LOCAL_SOLVE}" \
  --coupled_translation_trust_region "${OBJECT_COUPLED_TRANSLATION_TRUST_REGION}" \
  --translation_elimination_prox_weight "${OBJECT_TRANSLATION_ELIMINATION_PROX_WEIGHT}" \
  --local_solver "${LOCAL_SOLVER}" \
  --full_solver_outer_iterations "${OBJECT_FULL_SOLVER_OUTER_ITERS}" \
  --local_max_iterations "${OBJECT_LOCAL_TR_ITERS}" \
  --local_max_tcg_iterations "${OBJECT_LOCAL_TCG_ITERS}" \
  --local_grad_norm_tol "${OBJECT_LOCAL_TR_TOL}" \
  --trust_region_initial_radius "${OBJECT_TR_RADIUS}" \
  --reduced_rotation_preconditioner "${REDUCED_ROTATION_PRECONDITIONER}" \
  --output_dir "${OUT_ROOT}" \
  --save true | tee "${LOG_PATH}"

echo "RESULT_DIR ${OUT_ROOT}"

if [[ -n "${GT_DIR}" ]]; then
  GT_EVAL_PATH="${OUT_ROOT}/gt_eval.json"
  python3 "${ROOT_DIR}/scripts/evaluate_object_drone_gt.py" \
    --estimate "${OUT_ROOT}/object_poses.txt" \
    --gt-dir "${GT_DIR}" \
    --json > "${GT_EVAL_PATH}"
  echo "Saved GT evaluation: ${GT_EVAL_PATH}"
fi

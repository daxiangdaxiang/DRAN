#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRACK="${TRACK:-optimizer_parity}"
DRY_RUN="${DRY_RUN:-false}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/dran_object_fair_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1}"
GT_DIR="${GT_DIR:-${ROOT_DIR}/data/chordal_dataset/drone_g2o_file/range_100_tau_0_ang_error_0}"
OLD_BIN="${OLD_BIN:-${ROOT_DIR}/build/bin/object-based-multi-robot-example}"
MANUAL_BIN="${MANUAL_BIN:-${ROOT_DIR}/build/bin/manual-object-dran-example}"

NUM_ROBOTS="${NUM_ROBOTS:-21}"
NUM_OBJECTS="${NUM_OBJECTS:-0}"
MAX_ITERS="${MAX_ITERS:-20}"
BETA="${BETA:-1}"
ETA="${ETA:-0.5}"
OBJECT_TOPOLOGY="${OBJECT_TOPOLOGY:-ring}"
OBJECT_RING_HOPS="${OBJECT_RING_HOPS:-1}"
COMM_TOPOLOGY_FILE="${COMM_TOPOLOGY_FILE:-}"
COMM_WEIGHT_MODE="${COMM_WEIGHT_MODE:-unit}"
OBJECT_POSE_TOL="${OBJECT_POSE_TOL:-0.001}"
OBJECT_MAX_AGE="${OBJECT_MAX_AGE:-5}"
OBJECT_COMMUNICATION_POLICY="${OBJECT_COMMUNICATION_POLICY:-triggered}"

RUN_OLD="${RUN_OLD:-true}"
RUN_MANUAL_REDUCED="${RUN_MANUAL_REDUCED:-true}"
RUN_MANUAL_FULL="${RUN_MANUAL_FULL:-false}"

MANUAL_REDUCED_TCG="${MANUAL_REDUCED_TCG:-10}"
MANUAL_REDUCED_PRECONDITIONER="${MANUAL_REDUCED_PRECONDITIONER:-portfolio}"
MANUAL_FULL_TCG="${MANUAL_FULL_TCG:-20}"

DEPLOYMENT_OLD_INIT_MODE="${DEPLOYMENT_OLD_INIT_MODE:-distributed_chordal_object_jacobi}"
DEPLOYMENT_OLD_JACOBI_ITERS="${DEPLOYMENT_OLD_JACOBI_ITERS:-10}"
DEPLOYMENT_OLD_JACOBI_SCALE="${DEPLOYMENT_OLD_JACOBI_SCALE:-10}"
DEPLOYMENT_OLD_JACOBI_SCOPE="${DEPLOYMENT_OLD_JACOBI_SCOPE:-objects_only}"
DEPLOYMENT_MANUAL_INIT="${DEPLOYMENT_MANUAL_INIT:-neighbor_average}"
DEPLOYMENT_MANUAL_ANCHOR_MODE="${DEPLOYMENT_MANUAL_ANCHOR_MODE:-information}"
DEPLOYMENT_MANUAL_CONSENSUS_ROUNDS="${DEPLOYMENT_MANUAL_CONSENSUS_ROUNDS:-5}"
DEPLOYMENT_MANUAL_OBSERVED_ANCHOR="${DEPLOYMENT_MANUAL_OBSERVED_ANCHOR:-1}"
DEPLOYMENT_MANUAL_RELAY_ANCHOR="${DEPLOYMENT_MANUAL_RELAY_ANCHOR:-0}"

old_script="${ROOT_DIR}/scripts/run_object_drone_experiment.sh"
manual_script="${ROOT_DIR}/scripts/run_manual_object_dran_drone.sh"
manifest="${OUT_ROOT}/manifest.csv"
summary="${OUT_ROOT}/summary.csv"

common_line() {
  printf 'FAIR_COMPARE track=%s metric=measurement_cost data_dir=%s gt_dir=%s num_robots=%s num_objects=%s max_iters=%s beta=%s topology=%s ring_hops=%s comm_topology_file=%s comm_weight_mode=%s communication_policy=%s object_pose_tol=%s object_max_age=%s\n' \
    "${TRACK}" "${DATA_DIR}" "${GT_DIR}" "${NUM_ROBOTS}" "${NUM_OBJECTS}" \
    "${MAX_ITERS}" "${BETA}" "${OBJECT_TOPOLOGY}" "${OBJECT_RING_HOPS}" \
    "${COMM_TOPOLOGY_FILE:-<none>}" "${COMM_WEIGHT_MODE}" \
    "${OBJECT_COMMUNICATION_POLICY}" "${OBJECT_POSE_TOL}" "${OBJECT_MAX_AGE}"
}

print_command() {
  local method="$1"
  shift
  printf 'METHOD %s command=' "${method}"
  printf '%q ' "$@"
  printf '\n'
}

append_manifest() {
  local method="$1"
  local track="$2"
  local result_dir="$3"
  local init_mode="$4"
  local solver="$5"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${method}" "${track}" "${result_dir}" "${init_mode}" "${solver}" \
    "${DATA_DIR}" "${GT_DIR}" "${MAX_ITERS}" "${OBJECT_TOPOLOGY}" \
    "${OBJECT_RING_HOPS}" >> "${manifest}"
}

old_common_env=(
  "BIN=${OLD_BIN}"
  "DATA_DIR=${DATA_DIR}"
  "GT_DIR=${GT_DIR}"
  "NUM_ROBOTS=${NUM_ROBOTS}"
  "NUM_OBJECTS=${NUM_OBJECTS}"
  "MAX_ITERS=${MAX_ITERS}"
  "BETA=${BETA}"
  "ETA=${ETA}"
  "OBJECT_TOPOLOGY=${OBJECT_TOPOLOGY}"
  "OBJECT_RING_HOPS=${OBJECT_RING_HOPS}"
  "COMM_TOPOLOGY_FILE=${COMM_TOPOLOGY_FILE}"
  "COMM_WEIGHT_MODE=${COMM_WEIGHT_MODE}"
  "OBJECT_COMMUNICATION_POLICY=${OBJECT_COMMUNICATION_POLICY}"
  "OBJECT_POSE_TOL=${OBJECT_POSE_TOL}"
  "OBJECT_MAX_AGE=${OBJECT_MAX_AGE}"
)

manual_common_env=(
  "BIN=${MANUAL_BIN}"
  "DATA_DIR=${DATA_DIR}"
  "GT_DIR=${GT_DIR}"
  "NUM_ROBOTS=${NUM_ROBOTS}"
  "NUM_OBJECTS=${NUM_OBJECTS}"
  "MAX_ITERS=${MAX_ITERS}"
  "BETA=${BETA}"
  "OBJECT_TOPOLOGY=${OBJECT_TOPOLOGY}"
  "OBJECT_RING_HOPS=${OBJECT_RING_HOPS}"
  "COMM_TOPOLOGY_FILE=${COMM_TOPOLOGY_FILE}"
  "COMM_WEIGHT_MODE=${COMM_WEIGHT_MODE}"
  "OBJECT_COMMUNICATION_POLICY=${OBJECT_COMMUNICATION_POLICY}"
  "OBJECT_POSE_TOL=${OBJECT_POSE_TOL}"
  "OBJECT_MAX_AGE=${OBJECT_MAX_AGE}"
)

run_old_optimizer_parity() {
  local result_dir="${OUT_ROOT}/old_dran_object_optimizer_parity"
  local env_args=(
    "${old_common_env[@]}"
    "OUT_ROOT=${result_dir}"
    "OBJECT_INIT_MODE=centralized_chordal"
    "OBJECT_LOCAL_TR_ITERS=1"
    "OBJECT_LOCAL_TR_TOL=0.001"
  )
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    print_command "old_dran_object_optimizer_parity" \
      env "${env_args[@]}" "${old_script}"
    return
  fi
  env "${env_args[@]}" "${old_script}"
  append_manifest "old_dran_object" "optimizer_parity" "${result_dir}" \
    "centralized_chordal" "legacy_rtr1"
}

run_manual_reduced_optimizer_parity() {
  local result_dir="${OUT_ROOT}/manual_object_reduced_optimizer_parity"
  local env_args=(
    "${manual_common_env[@]}"
    "OUT_ROOT=${result_dir}"
    "OBJECT_INITIALIZATION=centralized_chordal"
    "OBJECT_INITIALIZATION_ANCHOR_MODE=constant"
    "OBJECT_INITIALIZATION_CONSENSUS_ROUNDS=0"
    "LOCAL_SOLVER=reduced_rotation"
    "REDUCED_ROTATION_PRECONDITIONER=${MANUAL_REDUCED_PRECONDITIONER}"
    "OBJECT_LOCAL_TCG_ITERS=${MANUAL_REDUCED_TCG}"
  )
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    print_command "manual_object_reduced_optimizer_parity" \
      env "${env_args[@]}" "${manual_script}"
    return
  fi
  env "${env_args[@]}" "${manual_script}"
  append_manifest "manual_object_reduced" "optimizer_parity" "${result_dir}" \
    "centralized_chordal" "reduced_rotation"
}

run_manual_full_optimizer_parity() {
  local result_dir="${OUT_ROOT}/manual_object_full_optimizer_parity"
  local env_args=(
    "${manual_common_env[@]}"
    "OUT_ROOT=${result_dir}"
    "OBJECT_INITIALIZATION=centralized_chordal"
    "OBJECT_INITIALIZATION_ANCHOR_MODE=constant"
    "OBJECT_INITIALIZATION_CONSENSUS_ROUNDS=0"
    "LOCAL_SOLVER=manual_full"
    "OBJECT_LOCAL_TCG_ITERS=${MANUAL_FULL_TCG}"
  )
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    print_command "manual_object_full_optimizer_parity" \
      env "${env_args[@]}" "${manual_script}"
    return
  fi
  env "${env_args[@]}" "${manual_script}"
  append_manifest "manual_object_full" "optimizer_parity" "${result_dir}" \
    "centralized_chordal" "manual_full"
}

run_old_deployment_init() {
  local result_dir="${OUT_ROOT}/old_dran_object_deployment_init"
  local env_args=(
    "${old_common_env[@]}"
    "OUT_ROOT=${result_dir}"
    "OBJECT_INIT_MODE=${DEPLOYMENT_OLD_INIT_MODE}"
    "OBJECT_INIT_OBJECT_JACOBI_ITERS=${DEPLOYMENT_OLD_JACOBI_ITERS}"
    "OBJECT_INIT_OBJECT_JACOBI_CONSENSUS_SCALE=${DEPLOYMENT_OLD_JACOBI_SCALE}"
    "OBJECT_INIT_OBJECT_JACOBI_SCOPE=${DEPLOYMENT_OLD_JACOBI_SCOPE}"
    "OBJECT_LOCAL_TR_ITERS=20"
    "OBJECT_LOCAL_TR_TOL=0.001"
  )
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    print_command "old_dran_object_deployment_init" \
      env "${env_args[@]}" "${old_script}"
    return
  fi
  env "${env_args[@]}" "${old_script}"
  append_manifest "old_dran_object" "deployment_init" "${result_dir}" \
    "${DEPLOYMENT_OLD_INIT_MODE}" "legacy_rtr20"
}

run_manual_deployment_init() {
  local result_dir="${OUT_ROOT}/manual_object_reduced_deployment_init"
  local env_args=(
    "${manual_common_env[@]}"
    "OUT_ROOT=${result_dir}"
    "OBJECT_INITIALIZATION=${DEPLOYMENT_MANUAL_INIT}"
    "OBJECT_INITIALIZATION_ANCHOR_MODE=${DEPLOYMENT_MANUAL_ANCHOR_MODE}"
    "OBJECT_INITIALIZATION_CONSENSUS_ROUNDS=${DEPLOYMENT_MANUAL_CONSENSUS_ROUNDS}"
    "OBJECT_INITIALIZATION_OBSERVED_ANCHOR_WEIGHT=${DEPLOYMENT_MANUAL_OBSERVED_ANCHOR}"
    "OBJECT_INITIALIZATION_RELAY_ANCHOR_WEIGHT=${DEPLOYMENT_MANUAL_RELAY_ANCHOR}"
    "LOCAL_SOLVER=reduced_rotation"
    "REDUCED_ROTATION_PRECONDITIONER=${MANUAL_REDUCED_PRECONDITIONER}"
    "OBJECT_LOCAL_TCG_ITERS=${MANUAL_REDUCED_TCG}"
  )
  if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    print_command "manual_object_reduced_deployment_init" \
      env "${env_args[@]}" "${manual_script}"
    return
  fi
  env "${env_args[@]}" "${manual_script}"
  append_manifest "manual_object_reduced" "deployment_init" "${result_dir}" \
    "${DEPLOYMENT_MANUAL_INIT}:${DEPLOYMENT_MANUAL_ANCHOR_MODE}" \
    "reduced_rotation"
}

run_track() {
  case "$1" in
    optimizer_parity)
      if [[ "${RUN_OLD}" != "false" ]]; then
        run_old_optimizer_parity
      fi
      if [[ "${RUN_MANUAL_REDUCED}" != "false" ]]; then
        run_manual_reduced_optimizer_parity
      fi
      if [[ "${RUN_MANUAL_FULL}" == "true" ]]; then
        run_manual_full_optimizer_parity
      fi
      ;;
    deployment_init)
      if [[ "${RUN_OLD}" != "false" ]]; then
        run_old_deployment_init
      fi
      if [[ "${RUN_MANUAL_REDUCED}" != "false" ]]; then
        run_manual_deployment_init
      fi
      ;;
    *)
      echo "Unknown TRACK: $1" >&2
      exit 2
      ;;
  esac
}

common_line
if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
  if [[ "${TRACK}" == "all" ]]; then
    TRACK="optimizer_parity"
    common_line
    run_track "optimizer_parity"
    TRACK="deployment_init"
    common_line
    run_track "deployment_init"
  else
    run_track "${TRACK}"
  fi
  exit 0
fi

mkdir -p "${OUT_ROOT}"
printf 'method,track,result_dir,init_mode,solver,data_dir,gt_dir,max_iters,topology,ring_hops\n' \
  > "${manifest}"

if [[ "${TRACK}" == "all" ]]; then
  TRACK="optimizer_parity"
  run_track "optimizer_parity"
  TRACK="deployment_init"
  run_track "deployment_init"
else
  run_track "${TRACK}"
fi

python3 "${ROOT_DIR}/scripts/summarize_dran_object_fair.py" \
  --manifest "${manifest}" \
  --output "${summary}"
echo "MANIFEST ${manifest}"
echo "SUMMARY ${summary}"

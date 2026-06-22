#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/chordal_fair_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"
SMOKE="${SMOKE:-false}"
BUDGETS="${BUDGETS:-}"
NUM_ROBOTS="${NUM_ROBOTS:-5}"
DRONE_DATA_DIR="${DRONE_DATA_DIR:-${ROOT_DIR}/data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1}"
GT_DIR="${GT_DIR:-${ROOT_DIR}/data/chordal_dataset/drone_g2o_file/range_100_tau_0_ang_error_0}"
DRAN_OBJECT_LOCAL_TR_ITERS="${DRAN_OBJECT_LOCAL_TR_ITERS:-20}"
DRAN_OBJECT_LOCAL_TR_TOL="${DRAN_OBJECT_LOCAL_TR_TOL:-0.001}"

RUN_SIX="${RUN_SIX:-true}"
RUN_DRONE="${RUN_DRONE:-true}"
RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-false}"

RUN_DRAN="${RUN_DRAN:-true}"
RUN_DPGO_FIRST="${RUN_DPGO_FIRST:-true}"
RUN_DPGO_SECOND="${RUN_DPGO_SECOND:-true}"
RUN_DPGO_MM="${RUN_DPGO_MM:-true}"
# Opt-in research replay arm. It is disabled by default so legacy chordal fair
# runs keep the original baseline set unless the seven-method gate asks for it.
RUN_MANUAL_REDUCED="${RUN_MANUAL_REDUCED:-false}"
RUN_MESA="${RUN_MESA:-true}"
RUN_DISTRIBUTED_MAPPER="${RUN_DISTRIBUTED_MAPPER:-true}"
RUN_SESYNC="${RUN_SESYNC:-true}"
RUN_DRAN_OBJECT="${RUN_DRAN_OBJECT:-true}"

MANUAL_REDUCED_BIN="${MANUAL_REDUCED_BIN:-}"
MANUAL_REDUCED_METHOD="${MANUAL_REDUCED_METHOD:-Manual-reduced}"
MANUAL_REDUCED_METHOD_NAME="${MANUAL_REDUCED_METHOD_NAME:-manual_reduced}"
MANUAL_REDUCED_SCHEME="${MANUAL_REDUCED_SCHEME:-mm}"
MANUAL_REDUCED_ACCELERATED="${MANUAL_REDUCED_ACCELERATED:-false}"
MANUAL_REDUCED_LOCAL_SOLVER="${MANUAL_REDUCED_LOCAL_SOLVER:-reduced_rotation}"
MANUAL_REDUCED_LOCAL_MAX_ITERATIONS="${MANUAL_REDUCED_LOCAL_MAX_ITERATIONS:-1}"
MANUAL_REDUCED_LOCAL_MAX_TCG_ITERATIONS="${MANUAL_REDUCED_LOCAL_MAX_TCG_ITERATIONS:-10}"
MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG="${MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG:-true}"
MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG_MAX_ITERATIONS="${MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG_MAX_ITERATIONS:-50}"
MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG_GRADIENT_RATIO="${MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG_GRADIENT_RATIO:-0.15}"
MANUAL_REDUCED_LOCAL_STATE_EXTRAPOLATION="${MANUAL_REDUCED_LOCAL_STATE_EXTRAPOLATION:-true}"
MANUAL_REDUCED_LOCAL_STATE_EXTRAPOLATION_GAMMAS="${MANUAL_REDUCED_LOCAL_STATE_EXTRAPOLATION_GAMMAS:-0.1,0.25,0.5}"
MANUAL_REDUCED_REDUCED_ROTATION_PRECONDITIONER="${MANUAL_REDUCED_REDUCED_ROTATION_PRECONDITIONER:-none}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ -z "${BUDGETS}" ]]; then
  if [[ "${SMOKE}" == "true" || "${SMOKE}" == "1" ]]; then
    BUDGETS="2,5"
  else
    BUDGETS="5,10,20,50,100,300"
  fi
fi

mkdir -p "${OUT_ROOT}"
MANIFEST="${OUT_ROOT}/manifest.csv"
printf "problem_type,setting,method,budget,out_root,status,command\n" > "${MANIFEST}"
SIX_REFERENCE_DONE=false
DRONE_REFERENCE_DONE=false
LAST_RUN_STATUS=0

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

bool_enabled() {
  local value="$1"
  [[ "${value}" == "true" || "${value}" == "1" || "${value}" == "on" ]]
}

command_string() {
  printf '%q ' "$@"
}

record_manifest() {
  local problem_type="$1"
  local setting="$2"
  local method="$3"
  local budget="$4"
  local run_root="$5"
  local status="$6"
  shift 6
  local command_text
  command_text="$(command_string "$@")"
  {
    csv_escape "${problem_type}"; printf ','
    csv_escape "${setting}"; printf ','
    csv_escape "${method}"; printf ','
    csv_escape "${budget}"; printf ','
    csv_escape "${run_root}"; printf ','
    csv_escape "${status}"; printf ','
    csv_escape "${command_text}"; printf '\n'
  } >> "${MANIFEST}"
}

run_case() {
  local problem_type="$1"
  local setting="$2"
  local method="$3"
  local budget="$4"
  local run_root="$5"
  shift 5
  local cmd=("$@")

  echo "=== ${problem_type} ${setting} ${method} budget=${budget} ==="
  echo "$(command_string "${cmd[@]}")"
  if bool_enabled "${DRY_RUN}"; then
    LAST_RUN_STATUS=0
    record_manifest "${problem_type}" "${setting}" "${method}" "${budget}" \
      "${run_root}" "dry_run" "${cmd[@]}"
    return 0
  fi

  set +e
  "${cmd[@]}"
  local status=$?
  set -e
  if [[ ${status} -eq 0 ]]; then
    record_manifest "${problem_type}" "${setting}" "${method}" "${budget}" \
      "${run_root}" "ok" "${cmd[@]}"
  else
    record_manifest "${problem_type}" "${setting}" "${method}" "${budget}" \
      "${run_root}" "failed_${status}" "${cmd[@]}"
  fi
  LAST_RUN_STATUS=${status}
  return 0
}

run_six_budget() {
  local budget="$1"
  local setting="fixed_${budget}"
  local common_disable_stop=(
    "MAX_ITERS=${budget}"
    "NUM_ROBOTS=${NUM_ROBOTS}"
    "GRAD_TOL=-1"
    "LOSS_TOL=-1"
    "STABLE_LOSS_ROUNDS=$((budget + 1))"
  )

  if bool_enabled "${RUN_SESYNC}" && ! bool_enabled "${SIX_REFERENCE_DONE}"; then
    local run_root="${OUT_ROOT}/six/sesync"
    run_case "six" "reference" "SE-Sync" "" "${run_root}" \
      env "OUT_ROOT=${run_root}" "MODE=six" \
      "${ROOT_DIR}/scripts/run_sesync_baselines.sh"
    if [[ ${LAST_RUN_STATUS} -eq 0 ]]; then
      SIX_REFERENCE_DONE=true
    fi
  fi

  if bool_enabled "${RUN_DRAN}"; then
    local run_root="${OUT_ROOT}/six/${setting}/dran"
    run_case "six" "${setting}" "DRAN" "${budget}" "${run_root}" \
      env "OUT_ROOT=${run_root}" "${common_disable_stop[@]}" \
      "UPDATE_MODE=decentralized_residual_async_schur_late_feedback" \
      "LOCAL_ALGORITHM=RTR" \
      "${ROOT_DIR}/scripts/run_six_dataset_experiments.sh"
  fi

  if bool_enabled "${RUN_DPGO_FIRST}"; then
    local run_root="${OUT_ROOT}/six/${setting}/dpgo_first"
    run_case "six" "${setting}" "DPGO-first" "${budget}" "${run_root}" \
      env "OUT_ROOT=${run_root}" "${common_disable_stop[@]}" \
      "${ROOT_DIR}/scripts/run_first_order_baseline_six.sh"
  fi

  if bool_enabled "${RUN_DPGO_SECOND}"; then
    local run_root="${OUT_ROOT}/six/${setting}/dpgo_second"
    run_case "six" "${setting}" "DPGO-second" "${budget}" "${run_root}" \
      env "OUT_ROOT=${run_root}" "${common_disable_stop[@]}" \
      "${ROOT_DIR}/scripts/run_dpgo_like_rtr_baseline_six.sh"
  fi

  if bool_enabled "${RUN_DPGO_MM}"; then
    local run_root="${OUT_ROOT}/six/${setting}/dpgo_mm"
    run_case "six" "${setting}" "DPGO-MM" "${budget}" "${run_root}" \
      env "OUT_ROOT=${run_root}" "MAX_ITERS=${budget}" \
      "NUM_ROBOTS=${NUM_ROBOTS}" "${ROOT_DIR}/scripts/run_dpgo_mm_six.sh"
  fi

  if bool_enabled "${RUN_MANUAL_REDUCED}"; then
    local run_root="${OUT_ROOT}/six/${setting}/manual_reduced"
    local manual_iters="${budget}"
    if [[ "${budget}" =~ ^[0-9]+$ && "${budget}" -gt 0 ]]; then
      # run_manual_dpgo_mm_six.sh writes iter=0 plus MAX_ITERS accepted update
      # rows. Use B-1 internally so the external budget B means the same
      # available communication-update rows as DPGO-MM in fair comparisons.
      manual_iters="$((budget - 1))"
    fi
    local manual_bin_env=()
    if [[ -n "${MANUAL_REDUCED_BIN}" ]]; then
      manual_bin_env=("BIN=${MANUAL_REDUCED_BIN}")
    elif [[ -x "${ROOT_DIR}/build-release/bin/manual-dpgo-mm-example" ]]; then
      manual_bin_env=("BIN=${ROOT_DIR}/build-release/bin/manual-dpgo-mm-example")
    fi
    run_case "six" "${setting}" "${MANUAL_REDUCED_METHOD}" "${budget}" "${run_root}" \
      env "OUT_ROOT=${run_root}" "MAX_ITERS=${manual_iters}" \
      "NUM_ROBOTS=${NUM_ROBOTS}" "${manual_bin_env[@]}" \
      "METHOD_NAME=${MANUAL_REDUCED_METHOD_NAME}" \
      "SCHEME=${MANUAL_REDUCED_SCHEME}" \
      "ACCELERATED=${MANUAL_REDUCED_ACCELERATED}" \
      "LOCAL_SOLVER=${MANUAL_REDUCED_LOCAL_SOLVER}" \
      "LOCAL_MAX_ITERATIONS=${MANUAL_REDUCED_LOCAL_MAX_ITERATIONS}" \
      "LOCAL_MAX_TCG_ITERATIONS=${MANUAL_REDUCED_LOCAL_MAX_TCG_ITERATIONS}" \
      "REDUCED_ROTATION_PRECONDITIONER=${MANUAL_REDUCED_REDUCED_ROTATION_PRECONDITIONER}" \
      "ADAPTIVE_REDUCED_TCG=${MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG}" \
      "ADAPTIVE_REDUCED_TCG_MAX_ITERATIONS=${MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG_MAX_ITERATIONS}" \
      "ADAPTIVE_REDUCED_TCG_GRADIENT_RATIO=${MANUAL_REDUCED_ADAPTIVE_REDUCED_TCG_GRADIENT_RATIO}" \
      "LOCAL_STATE_EXTRAPOLATION=${MANUAL_REDUCED_LOCAL_STATE_EXTRAPOLATION}" \
      "LOCAL_STATE_EXTRAPOLATION_GAMMAS=${MANUAL_REDUCED_LOCAL_STATE_EXTRAPOLATION_GAMMAS}" \
      "${ROOT_DIR}/scripts/run_manual_dpgo_mm_six.sh"
  fi

  if bool_enabled "${RUN_MESA}"; then
    local run_root="${OUT_ROOT}/six/${setting}/mesa"
    run_case "six" "${setting}" "MESA" "${budget}" "${run_root}" \
      env "OUT_ROOT=${run_root}" "MAX_ITERS=${budget}" \
      "FIXED_ITERS=true" "NUM_ROBOTS=${NUM_ROBOTS}" \
      "MESA_METHOD=chordal-mesa" "PARTITION_STRATEGY=dpgo" \
      "CHORDAL_INITIALIZATION=true" \
      "${ROOT_DIR}/scripts/run_mesa_chordal_six.sh"
  fi

  if bool_enabled "${RUN_DISTRIBUTED_MAPPER}"; then
    local run_root="${OUT_ROOT}/six/${setting}/distributed_mapper"
    run_case "six" "${setting}" "Distributed-Mapper" "${budget}" \
      "${run_root}" env "OUT_ROOT=${run_root}" "MAX_ITERS=${budget}" \
      "NUM_ROBOTS=${NUM_ROBOTS}" "CHORDAL_INITIALIZATION=true" \
      "${ROOT_DIR}/scripts/run_distributed_mapper_baseline2_5robots.sh"
  fi
}

run_drone_budget() {
  local budget="$1"
  local setting="fixed_${budget}"

  if bool_enabled "${RUN_SESYNC}" && ! bool_enabled "${DRONE_REFERENCE_DONE}"; then
    local run_root="${OUT_ROOT}/drone/sesync_object_aware"
    run_case "drone_object" "reference" "SE-Sync-object-aware" "" \
      "${run_root}" env "OUT_ROOT=${run_root}" "MODE=drone" \
      "DRONE_ID_MODE=object-aware" "DRONE_INPUT_DIR=${DRONE_DATA_DIR}" \
      "${ROOT_DIR}/scripts/run_sesync_baselines.sh"
    if [[ ${LAST_RUN_STATUS} -eq 0 ]]; then
      DRONE_REFERENCE_DONE=true
    fi
  fi

  if bool_enabled "${RUN_DRAN_OBJECT}"; then
    local run_root="${OUT_ROOT}/drone/${setting}/dran_object_centralized_init"
    run_case "drone_object" "${setting}" "DRAN-Object-centralized-init" \
      "${budget}" "${run_root}" env "OUT_ROOT=${run_root}" \
      "MAX_ITERS=${budget}" "OBJECT_LOCAL_OBJECTIVE=chordal" \
      "OBJECT_CONSENSUS_MODE=penalty" "OBJECT_INIT_MODE=centralized_chordal" \
      "OBJECT_LOCAL_TR_ITERS=${DRAN_OBJECT_LOCAL_TR_ITERS}" \
      "OBJECT_LOCAL_TR_TOL=${DRAN_OBJECT_LOCAL_TR_TOL}" \
      "DATA_DIR=${DRONE_DATA_DIR}" "GT_DIR=${GT_DIR}" \
      "${ROOT_DIR}/scripts/run_object_drone_experiment.sh"

    run_root="${OUT_ROOT}/drone/${setting}/dran_object_deployment_init"
    run_case "drone_object" "${setting}" "DRAN-Object-deployment-init" \
      "${budget}" "${run_root}" env "OUT_ROOT=${run_root}" \
      "MAX_ITERS=${budget}" "OBJECT_LOCAL_OBJECTIVE=chordal" \
      "OBJECT_CONSENSUS_MODE=penalty" \
      "OBJECT_INIT_MODE=distributed_chordal_object_jacobi" \
      "OBJECT_INIT_OBJECT_JACOBI_SCOPE=all" \
      "OBJECT_INIT_OBJECT_JACOBI_CONSENSUS_SCALE=10" \
      "OBJECT_INIT_OBJECT_JACOBI_ITERS=20" \
      "OBJECT_LOCAL_TR_ITERS=${DRAN_OBJECT_LOCAL_TR_ITERS}" \
      "OBJECT_LOCAL_TR_TOL=${DRAN_OBJECT_LOCAL_TR_TOL}" \
      "DATA_DIR=${DRONE_DATA_DIR}" "GT_DIR=${GT_DIR}" \
      "${ROOT_DIR}/scripts/run_object_drone_experiment.sh"
  fi
}

run_robustness_smoke() {
  if ! bool_enabled "${RUN_ROBUSTNESS}"; then
    return 0
  fi
  local topo_root="${OUT_ROOT}/topologies/drone_random_tv_p02_r5"
  run_case "drone_object" "topology_generation" "topology-random-tv-p02" "" \
    "${topo_root}" python3 "${ROOT_DIR}/scripts/communication_topology.py" \
    --num-robots 21 --topology random --probability 0.2 --time-varying \
    --rounds 5 --weighting metropolis --seed 7 --output-dir "${topo_root}"

  local budget="${ROBUSTNESS_ITERS:-10}"
  local run_root="${OUT_ROOT}/drone/robust_random_tv_p02/dran_object"
  run_case "drone_object" "robust_random_tv_p02" "DRAN-Object-centralized-init" \
    "${budget}" "${run_root}" env "OUT_ROOT=${run_root}" \
    "MAX_ITERS=${budget}" "OBJECT_LOCAL_OBJECTIVE=chordal" \
    "OBJECT_CONSENSUS_MODE=penalty" "OBJECT_INIT_MODE=centralized_chordal" \
    "OBJECT_LOCAL_TR_ITERS=${DRAN_OBJECT_LOCAL_TR_ITERS}" \
    "OBJECT_LOCAL_TR_TOL=${DRAN_OBJECT_LOCAL_TR_TOL}" \
    "COMM_TOPOLOGY_FILE=${topo_root}/topology_edges.csv" \
    "COMM_WEIGHT_MODE=unit" "DATA_DIR=${DRONE_DATA_DIR}" "GT_DIR=${GT_DIR}" \
    "${ROOT_DIR}/scripts/run_object_drone_experiment.sh"
}

IFS=',' read -r -a BUDGET_ARRAY <<< "${BUDGETS}"
for budget in "${BUDGET_ARRAY[@]}"; do
  budget="${budget// /}"
  if [[ -z "${budget}" ]]; then
    continue
  fi
  if bool_enabled "${RUN_SIX}"; then
    run_six_budget "${budget}"
  fi
  if bool_enabled "${RUN_DRONE}"; then
    run_drone_budget "${budget}"
  fi
done

run_robustness_smoke

echo "Saved manifest: ${MANIFEST}"
echo "Result root: ${OUT_ROOT}"
if bool_enabled "${DRY_RUN}"; then
  echo "Dry run only; no benchmark scripts were executed."
fi

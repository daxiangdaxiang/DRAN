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
EVENT_RESIDUAL_TOL="${EVENT_RESIDUAL_TOL:-0.0001}"
ACCELERATION="${ACCELERATION:-0}"
NEIGHBOR_DIRECTION_GAIN="${NEIGHBOR_DIRECTION_GAIN:-0.5}"

declare -A DATASETS=(
  ["parking-garage"]="data/parking-garage.g2o"
  ["sphere"]="data/sphere2500.g2o"
  ["torus"]="data/torus3D.g2o"
  ["CSAIL"]="data/CSAIL.g2o"
  ["inter"]="data/input_INTEL_g2o.g2o"
  ["manhattan"]="data/input_M3500_g2o.g2o"
)

DATASET_ORDER=("parking-garage" "sphere" "torus" "CSAIL" "inter" "manhattan")

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build it first, for example: cmake -S . -B build && make -C build multi-robot-example" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/iterations" "${OUT_ROOT}/poses"

SUMMARY_CSV="${OUT_ROOT}/final_summary.csv"

printf "dataset,status,final_iter,final_cost,final_gradnorm,total_comm_poses,total_comm_mb,loss_tol_iter,loss_tol_comm_poses,loss_tol_comm_mb,grad_tol_iter,grad_tol_comm_poses,grad_tol_comm_mb,convergence_iter,convergence_comm_poses,convergence_comm_mb,cumulative_parallel_compute_ms,log_path,csv_path,pose_path\n" > "${SUMMARY_CSV}"

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
  rel_path="${DATASETS[${dataset}]}"
  data_path="${ROOT_DIR}/${rel_path}"
  log_path="${OUT_ROOT}/logs/${dataset}.log"
  iter_csv="${OUT_ROOT}/iterations/${dataset}.csv"
  pose_path="${OUT_ROOT}/poses/${dataset}.txt"

  if [[ ! -f "${data_path}" ]]; then
    echo "[${dataset}] missing dataset: ${data_path}" | tee "${log_path}"
    printf "%s,missing_dataset,,,,,,,,,,,,,,,,%s,%s,%s\n" \
      "${dataset}" "${log_path}" "${iter_csv}" "${pose_path}" >> "${SUMMARY_CSV}"
    continue
  fi

  echo "=== Running ${dataset}: ${rel_path} ==="
  echo "CONFIG bin=${BIN} num_robots=${NUM_ROBOTS} max_iters=${MAX_ITERS} rtr_iters=${RTR_ITERS} rtr_max_inner=${RTR_MAX_INNER} rtr_tol=${RTR_TOL} update_mode=${UPDATE_MODE} local_algorithm=${LOCAL_ALGORITHM}"
  set +e
  "${BIN}" "${NUM_ROBOTS}" "${data_path}" "${MAX_ITERS}" "${GRAD_TOL}" \
    "${LOSS_TOL}" "${iter_csv}" "${RTR_ITERS}" "${RTR_MAX_INNER}" \
    "${RTR_TOL}" "${RTR_RADIUS}" "${STABLE_LOSS_ROUNDS}" \
    "${UPDATE_MODE}" "${DECENTRALIZED_REFRESH_PERIOD}" "${EVENT_POSE_TOL}" \
    "${EVENT_MAX_AGE}" "${EVENT_LOCAL_GRAD_TOL}" \
    "${EVENT_BUDGET_FRACTION}" "${EVENT_MAX_POSES_PER_NEIGHBOR}" \
    "${LOCAL_ALGORITHM}" "${EVENT_RESIDUAL_TOL}" "${ACCELERATION}" \
    "${NEIGHBOR_DIRECTION_GAIN}" "${pose_path}" 2>&1 | tee "${log_path}"
  run_status=${PIPESTATUS[0]}
  set -e

  summary_line="$(grep '^SUMMARY' "${log_path}" | tail -n 1 || true)"
  if [[ -z "${summary_line}" ]]; then
    printf "%s,failed,,,,,,,,,,,,,,,,%s,%s,%s\n" \
      "${dataset}" "${log_path}" "${iter_csv}" "${pose_path}" >> "${SUMMARY_CSV}"
    echo "[${dataset}] failed with status ${run_status}; no SUMMARY line found." >&2
    continue
  fi

  printf "%s,ok,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${dataset}" \
    "$(summary_value final_iter "${summary_line}")" \
    "$(summary_value final_cost "${summary_line}")" \
    "$(summary_value final_gradnorm "${summary_line}")" \
    "$(summary_value total_comm_poses "${summary_line}")" \
    "$(summary_value total_comm_mb "${summary_line}")" \
    "$(summary_value loss_tol_iter "${summary_line}")" \
    "$(summary_value loss_tol_comm_poses "${summary_line}")" \
    "$(summary_value loss_tol_comm_mb "${summary_line}")" \
    "$(summary_value grad_tol_iter "${summary_line}")" \
    "$(summary_value grad_tol_comm_poses "${summary_line}")" \
    "$(summary_value grad_tol_comm_mb "${summary_line}")" \
    "$(summary_value convergence_iter "${summary_line}")" \
    "$(summary_value convergence_comm_poses "${summary_line}")" \
    "$(summary_value convergence_comm_mb "${summary_line}")" \
    "$(summary_value cumulative_parallel_compute_ms "${summary_line}")" \
    "${log_path}" \
    "${iter_csv}" \
    "${pose_path}" >> "${SUMMARY_CSV}"
done

echo "Saved final summary: ${SUMMARY_CSV}"
echo "Saved per-dataset logs and iteration CSVs under: ${OUT_ROOT}"

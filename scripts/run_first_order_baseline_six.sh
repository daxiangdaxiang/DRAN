#!/usr/bin/env bash
set -euo pipefail

# DPGO-first baseline from baselines/DPGO's original multi-robot example.
# It uses selected-robot communication and local RGD optimization.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPGO_ROOT="${DPGO_ROOT:-${ROOT_DIR}/baselines/DPGO}"
if [[ -z "${BIN:-}" ]]; then
  if [[ -x "${DPGO_ROOT}/build-local/bin/multi-robot-example" ]]; then
    BIN="${DPGO_ROOT}/build-local/bin/multi-robot-example"
  else
    BIN="${DPGO_ROOT}/build/bin/multi-robot-example"
  fi
fi
if [[ -z "${DPGO_LD_LIBRARY_PATH:-}" ]]; then
  if [[ -d "${DPGO_ROOT}/build-local/lib" ]]; then
    DPGO_LD_LIBRARY_PATH="${DPGO_ROOT}/build-local/lib"
  else
    DPGO_LD_LIBRARY_PATH="${DPGO_ROOT}/build/lib"
  fi
fi
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/first_order_baseline_six_$(date +%Y%m%d_%H%M%S)}"
DATASET_FILTER="${DATASET_FILTER:-${DATASETS:-}}"

NUM_ROBOTS="${NUM_ROBOTS:-5}"
MAX_ITERS="${MAX_ITERS:-300}"
ACCELERATION="${ACCELERATION:-true}"
DPGO_GRAD_STOP_TOL="${DPGO_GRAD_STOP_TOL:-${GRAD_TOL:-0.1}}"

declare -A DATASETS=(
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

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build it first: cmake -S ${DPGO_ROOT} -B ${DPGO_ROOT}/build && cmake --build ${DPGO_ROOT}/build --target multi-robot-example -j2" >&2
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
    printf "%s,missing_dataset,,,,,,,,,,,,,,,%s,%s,%s\n" \
      "${dataset}" "${log_path}" "${iter_csv}" "${pose_path}" >> "${SUMMARY_CSV}"
    continue
  fi

  echo "=== First-order baseline ${dataset}: ${rel_path} ==="
  set +e
  DPGO_GRAD_STOP_TOL="${DPGO_GRAD_STOP_TOL}" \
  LD_LIBRARY_PATH="${DPGO_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
    "${BIN}" "${NUM_ROBOTS}" "${data_path}" "${iter_csv}" "${MAX_ITERS}" RGD "${ACCELERATION}" 2>&1 | tee "${log_path}"
  run_status=${PIPESTATUS[0]}
  set -e

  summary_line="$(grep '^FINAL_SUMMARY' "${log_path}" | tail -n 1 || true)"
  if [[ -z "${summary_line}" ]]; then
    printf "%s,failed,,,,,,,,,,,,,,,%s,%s,%s\n" \
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
    "" "" "" "" "" "" "" "" "" "" \
    "${log_path}" \
    "${iter_csv}" \
    "${pose_path}" >> "${SUMMARY_CSV}"
done

echo "Saved final summary: ${SUMMARY_CSV}"
echo "Saved per-dataset logs and iteration CSVs under: ${OUT_ROOT}"

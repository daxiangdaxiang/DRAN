#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

pick_binary() {
  local candidate
  for candidate in \
    "${ROOT_DIR}/baselines/SESync/C++/build/bin/SE-Sync" \
    "${ROOT_DIR}/baselines/SESync/C++/build-local/bin/SE-Sync"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

BIN="${BIN:-$(pick_binary || true)}"
SESync_LD_LIBRARY_PATH="${SESync_LD_LIBRARY_PATH:-${ROOT_DIR}/baselines/SESync/C++/build-local/lib:${ROOT_DIR}/baselines/SESync/C++/build/lib}"
MODE="${MODE:-six}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/sesync_$(date +%Y%m%d_%H%M%S)}"
THREADS="${THREADS:-4}"
VERBOSE="${VERBOSE:-0}"
DRONE_INPUT_DIR="${DRONE_INPUT_DIR:-${ROOT_DIR}/data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1}"
DRONE_ID_MODE="${DRONE_ID_MODE:-robot-local}"
DRONE_NUM_OBJECTS="${DRONE_NUM_OBJECTS:-0}"
EVAL_COST_MODE="${EVAL_COST_MODE:-dpgo}"

DRONE_GENERATOR="${DRONE_GENERATOR:-${ROOT_DIR}/scripts/make_drone_global_g2o.py}"
EVALUATOR="${EVALUATOR:-${ROOT_DIR}/scripts/evaluate_pgo.py}"

declare -A DATASETS=(
  ["parking-garage"]="data/parking-garage.g2o"
  ["sphere"]="data/sphere2500.g2o"
  ["torus"]="data/torus3D.g2o"
  ["CSAIL"]="data/CSAIL.g2o"
  ["inter"]="data/input_INTEL_g2o.g2o"
  ["manhattan"]="data/input_M3500_g2o.g2o"
)

DATASET_ORDER=("parking-garage" "sphere" "torus" "CSAIL" "inter" "manhattan")

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/poses" "${OUT_ROOT}/inputs" "${OUT_ROOT}/eval"

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build it first: cmake -S ${ROOT_DIR}/baselines/SESync/C++ -B ${ROOT_DIR}/baselines/SESync/C++/build && cmake --build ${ROOT_DIR}/baselines/SESync/C++/build --target SE-Sync -j2" >&2
  exit 1
fi

if [[ "${MODE}" != "six" && "${MODE}" != "drone" ]]; then
  echo "Error: unsupported MODE=${MODE}. Expected six or drone." >&2
  exit 1
fi

SUMMARY_CSV="${OUT_ROOT}/summary.csv"
printf "dataset,mode,input,output,num_poses,num_measurements,elapsed_sec,status,sdpval,fxhat,duality_gap,suboptimality_bound,optimal_cost,eval_path,log_path,poses_path\n" > "${SUMMARY_CSV}"

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

run_sesync_case() {
  local dataset="$1"
  local input_path="$2"
  local output_path="$3"
  local log_path="$4"

  echo "=== Running ${dataset}: ${input_path#${ROOT_DIR}/} ==="
  set +e
  if [[ "${VERBOSE}" == "1" ]]; then
    LD_LIBRARY_PATH="${SESync_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
      "${BIN}" --input "${input_path}" --output "${output_path}" --threads "${THREADS}" --verbose 2>&1 | tee "${log_path}"
  else
    LD_LIBRARY_PATH="${SESync_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
      "${BIN}" --input "${input_path}" --output "${output_path}" --threads "${THREADS}" --quiet 2>&1 | tee "${log_path}"
  fi
  run_status=${PIPESTATUS[0]}
  set -e

  local summary_line
  summary_line="$(grep '^SUMMARY' "${log_path}" | tail -n 1 || true)"
  if [[ -z "${summary_line}" ]]; then
    printf "%s,%s,%s,%s,,,,,,,,,,,%s,%s\n" \
      "${dataset}" "${MODE}" "${input_path}" "${output_path}" "${log_path}" "${output_path}" >> "${SUMMARY_CSV}"
    echo "[${dataset}] failed with status ${run_status}; no SUMMARY line found." >&2
    return 0
  fi

  local eval_path="${OUT_ROOT}/eval/${dataset}.json"
  local optimal_cost=""
  if [[ -f "${EVALUATOR}" ]]; then
    python3 "${EVALUATOR}" --graph "${input_path}" \
      --estimate "${output_path}" --estimate-format matrix \
      --reference "${input_path}" --reference-format g2o \
      --cost-mode "${EVAL_COST_MODE}" \
      --json > "${eval_path}" || true
    if [[ -s "${eval_path}" ]]; then
      optimal_cost="$(python3 - "${eval_path}" <<'PY'
import json
import sys
with open(sys.argv[1]) as f:
    print(json.load(f).get("chordal_cost", ""))
PY
)"
    fi
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${dataset}" \
    "${MODE}" \
    "${input_path}" \
    "${output_path}" \
    "$(summary_value num_poses "${summary_line}")" \
    "$(summary_value num_measurements "${summary_line}")" \
    "$(summary_value elapsed_sec "${summary_line}")" \
    "$(summary_value status "${summary_line}")" \
    "$(summary_value sdpval "${summary_line}")" \
    "$(summary_value fxhat "${summary_line}")" \
    "$(summary_value duality_gap "${summary_line}")" \
    "$(summary_value suboptimality_bound "${summary_line}")" \
    "${optimal_cost}" \
    "${eval_path}" \
    "${log_path}" \
    "${output_path}" >> "${SUMMARY_CSV}"
}

case "${MODE}" in
  six)
    for dataset in "${DATASET_ORDER[@]}"; do
      rel_path="${DATASETS[${dataset}]}"
      input_path="${ROOT_DIR}/${rel_path}"
      output_path="${OUT_ROOT}/poses/${dataset}.txt"
      log_path="${OUT_ROOT}/logs/${dataset}.log"

      if [[ ! -f "${input_path}" ]]; then
        echo "[${dataset}] missing dataset: ${input_path}" | tee "${log_path}"
        printf "%s,%s,%s,%s,,,,,,,,,,,%s,%s\n" \
          "${dataset}" "${MODE}" "${input_path}" "${output_path}" "${log_path}" "${output_path}" >> "${SUMMARY_CSV}"
        continue
      fi

      run_sesync_case "${dataset}" "${input_path}" "${output_path}" "${log_path}"
    done
    ;;

  drone)
    if [[ ! -f "${DRONE_GENERATOR}" ]]; then
      echo "Error: missing drone generator script: ${DRONE_GENERATOR}" >&2
      echo "Expected scripts/make_drone_global_g2o.py to exist before running MODE=drone." >&2
      exit 1
    fi

    drone_input="${OUT_ROOT}/inputs/drone_global.g2o"
    drone_log="${OUT_ROOT}/logs/drone_generate.log"
    echo "=== Generating drone global g2o ==="
    set +e
    python3 "${DRONE_GENERATOR}" --input-dir "${DRONE_INPUT_DIR}" \
      --output "${drone_input}" --id-mode "${DRONE_ID_MODE}" \
      --num-objects "${DRONE_NUM_OBJECTS}" 2>&1 | tee "${drone_log}"
    gen_status=${PIPESTATUS[0]}
    set -e

    if [[ ${gen_status} -ne 0 ]]; then
      echo "Error: drone generator failed with status ${gen_status}. See ${drone_log}" >&2
      exit ${gen_status}
    fi

    if [[ ! -f "${drone_input}" ]]; then
      echo "Error: drone generator did not create ${drone_input}" >&2
      exit 1
    fi

    run_sesync_case "drone" "${drone_input}" "${OUT_ROOT}/poses/drone.txt" "${OUT_ROOT}/logs/drone.log"
    ;;
esac

echo "Saved summary CSV: ${SUMMARY_CSV}"
echo "Saved logs and pose matrices under: ${OUT_ROOT}"

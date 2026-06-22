#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pick_binary() {
  local candidate
  for candidate in \
    "${ROOT_DIR}/baselines/mesa_ws2/mesa/build-workerC/experiments/run-dist-batch" \
    "${ROOT_DIR}/baselines/mesa_ws2/mesa/build-local/experiments/run-dist-batch" \
    "${ROOT_DIR}/baselines/mesa_ws2/mesa/build/experiments/run-dist-batch" \
    "${ROOT_DIR}/baselines/mesa_ws2/mesa/build/bin/run-dist-batch"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

pick_converter() {
  local candidate
  for candidate in \
    "${ROOT_DIR}/baselines/mesa_ws2/mesa/build-workerC/experiments/g2o-2-mr-jrl" \
    "${ROOT_DIR}/baselines/mesa_ws2/mesa/build-local/experiments/g2o-2-mr-jrl" \
    "${ROOT_DIR}/baselines/mesa_ws2/mesa/build/experiments/g2o-2-mr-jrl" \
    "${ROOT_DIR}/baselines/mesa_ws2/mesa/build/bin/g2o-2-mr-jrl"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

RUNNER="${RUNNER:-$(pick_binary || true)}"
CONVERTER="${CONVERTER:-$(pick_converter || true)}"
MESA_LD_LIBRARY_PATH="${MESA_LD_LIBRARY_PATH:-${ROOT_DIR}/baselines/mesa_ws2/mesa/build-local/thirdparty/dc2-pgo:${ROOT_DIR}/baselines/mesa_ws2/mesa/build-local/thirdparty/dc2-pgo/thirdparty/ROPTLIB:${ROOT_DIR}/baselines/mesa_ws2/mesa/build/thirdparty/dc2-pgo:${ROOT_DIR}/baselines/mesa_ws2/mesa/build/thirdparty/dc2-pgo/thirdparty/ROPTLIB}"
NUM_ROBOTS="${NUM_ROBOTS:-5}"
MAX_ITERS="${MAX_ITERS:-}"
CHORDAL_INITIALIZATION="${CHORDAL_INITIALIZATION:-true}"
PARTITION_STRATEGY="${PARTITION_STRATEGY:-dpgo}"
MESA_METHOD="${MESA_METHOD:-chordal-mesa}"
FIXED_ITERS="${FIXED_ITERS:-false}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/mesa_chordal_six_$(date +%Y%m%d_%H%M%S)}"
DATASET_FILTER="${DATASET_FILTER:-${DATASETS:-}}"
if [[ "${OUT_ROOT}" != /* ]]; then
  OUT_ROOT="${ROOT_DIR}/${OUT_ROOT}"
fi
LOG_ROOT="${OUT_ROOT}/logs"
RUN_ROOT="${OUT_ROOT}/runs"
JRL_ROOT="${JRL_ROOT:-${OUT_ROOT}/jrl_cache}"
if [[ "${JRL_ROOT}" != /* ]]; then
  JRL_ROOT="${ROOT_DIR}/${JRL_ROOT}"
fi
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

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

detect_g2o_dimension() {
  local g2o_path="$1"
  awk '
    $1 ~ /^(VERTEX|EDGE)_SE3/ { print "3"; exit }
    $1 ~ /^(VERTEX|EDGE)_SE2/ { print "2"; exit }
  ' "${g2o_path}"
}

if [[ -z "${RUNNER}" || ! -x "${RUNNER}" ]]; then
  echo "Missing executable: ${RUNNER:-<unset>}" >&2
  echo "Expected one of:" >&2
  echo "  ${ROOT_DIR}/baselines/mesa_ws2/mesa/build-workerC/experiments/run-dist-batch" >&2
  echo "  ${ROOT_DIR}/baselines/mesa_ws2/mesa/build-local/experiments/run-dist-batch" >&2
  echo "  ${ROOT_DIR}/baselines/mesa_ws2/mesa/build/experiments/run-dist-batch" >&2
  echo "  ${ROOT_DIR}/baselines/mesa_ws2/mesa/build/bin/run-dist-batch" >&2
  exit 1
fi

if [[ -z "${CONVERTER}" || ! -x "${CONVERTER}" ]]; then
  echo "Missing executable: ${CONVERTER:-<unset>}" >&2
  echo "Expected one of:" >&2
  echo "  ${ROOT_DIR}/baselines/mesa_ws2/mesa/build-workerC/experiments/g2o-2-mr-jrl" >&2
  echo "  ${ROOT_DIR}/baselines/mesa_ws2/mesa/build-local/experiments/g2o-2-mr-jrl" >&2
  echo "  ${ROOT_DIR}/baselines/mesa_ws2/mesa/build/experiments/g2o-2-mr-jrl" >&2
  echo "  ${ROOT_DIR}/baselines/mesa_ws2/mesa/build/bin/g2o-2-mr-jrl" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}" "${JRL_ROOT}"

printf 'dataset,method,status,global_cost,mean_residual,gradient,total_comm_poses,total_comm_mb,output_dir,jrl_path,log_path,result_path,final_metrics_path,comm_path,iter_summary_path\n' > "${SUMMARY_CSV}"

read_cbor_mean_residual() {
  local metrics_path="$1"
  python3 - "${metrics_path}" <<'PY'
import math
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()

def read_uint(data, idx, ai):
    if ai < 24:
        return ai, idx
    if ai == 24:
        return data[idx], idx + 1
    if ai == 25:
        return struct.unpack_from('>H', data, idx)[0], idx + 2
    if ai == 26:
        return struct.unpack_from('>I', data, idx)[0], idx + 4
    if ai == 27:
        return struct.unpack_from('>Q', data, idx)[0], idx + 8
    raise ValueError('unsupported CBOR integer length')

def decode(data, idx=0):
    initial = data[idx]
    idx += 1
    major = initial >> 5
    ai = initial & 0x1F

    if major == 0:
      val, idx = read_uint(data, idx, ai)
      return val, idx
    if major == 1:
      val, idx = read_uint(data, idx, ai)
      return -1 - val, idx
    if major == 3:
      length, idx = read_uint(data, idx, ai)
      text = data[idx:idx + length].decode('utf-8')
      return text, idx + length
    if major == 4:
      length, idx = read_uint(data, idx, ai)
      items = []
      for _ in range(length):
        item, idx = decode(data, idx)
        items.append(item)
      return items, idx
    if major == 5:
      length, idx = read_uint(data, idx, ai)
      items = {}
      for _ in range(length):
        key, idx = decode(data, idx)
        value, idx = decode(data, idx)
        items[key] = value
      return items, idx
    if major == 7:
      if ai == 20:
        return False, idx
      if ai == 21:
        return True, idx
      if ai == 22 or ai == 23:
        return None, idx
      if ai == 26:
        return struct.unpack_from('>f', data, idx)[0], idx + 4
      if ai == 27:
        return struct.unpack_from('>d', data, idx)[0], idx + 8
    raise ValueError(f'unsupported CBOR major type {major} additional info {ai}')

obj, end = decode(data)
if not isinstance(obj, dict):
    print('')
    sys.exit(0)
value = obj.get('mean_residual', '')
if value is None:
    value = ''
print(value)
PY
}

for dataset in "${DATASET_ORDER[@]}"; do
  rel_path="${DATASETS[${dataset}]}"
  data_path="${ROOT_DIR}/${rel_path}"
  run_dir="${RUN_ROOT}/${dataset}"
  log_path="${LOG_ROOT}/${dataset}.log"
  init_tag="rawinit"
  if [[ "${CHORDAL_INITIALIZATION}" == "true" || "${CHORDAL_INITIALIZATION}" == "1" ]]; then
    init_tag="chordalinit"
  fi
  jrl_path="${JRL_ROOT}/${init_tag}_${PARTITION_STRATEGY}_r${NUM_ROBOTS}_${dataset}.jrl"
  final_results_path=""
  final_metrics_path=""
  comm_path=""
  iter_summary_path=""
  method_name="${MESA_METHOD}"

  mkdir -p "${run_dir}"

  if [[ ! -f "${data_path}" ]]; then
    echo "[${dataset}] missing dataset: ${data_path}" | tee "${log_path}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${dataset}" "${method_name}" "missing_dataset" "" "" "" "" "" "${run_dir}" "${jrl_path}" "${log_path}" "" "" "" "" >> "${SUMMARY_CSV}"
    continue
  fi

  g2o_dim="$(detect_g2o_dimension "${data_path}")"
  if [[ "${g2o_dim}" != "2" && "${g2o_dim}" != "3" ]]; then
    echo "[${dataset}] unsupported or unrecognized g2o type: ${data_path}" | tee "${log_path}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${dataset}" "${method_name}" "unsupported_g2o" "" "" "" "" "" "${run_dir}" "${jrl_path}" "${log_path}" "" "" "" "" >> "${SUMMARY_CSV}"
    continue
  fi

  if [[ ! -f "${jrl_path}" ]]; then
    echo "Converting ${rel_path} -> ${jrl_path}"
    set +e
    LD_LIBRARY_PATH="${MESA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
    "${CONVERTER}" -i "${data_path}" -n "${init_tag}_${dataset}" -o "${jrl_path}" -p "${NUM_ROBOTS}" \
      --chordal_initialization "${CHORDAL_INITIALIZATION}" \
      --partition_strategy "${PARTITION_STRATEGY}" 2>&1 | tee "${log_path}"
    convert_status=${PIPESTATUS[0]}
    set -e
    if [[ ${convert_status} -ne 0 ]]; then
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${dataset}" "${method_name}" "convert_failed" "" "" "" "" "" "${run_dir}" "${jrl_path}" "${log_path}" "" "" "" "" >> "${SUMMARY_CSV}"
      continue
    fi
  fi

  echo "=== MESA ${method_name} ${dataset}: ${rel_path} ==="
  runner_args=("${RUNNER}" -i "${jrl_path}" -m "${method_name}" -o "${run_dir}")
  if [[ "${g2o_dim}" == "3" ]]; then
    runner_args+=(--is3d)
  fi
  if [[ -n "${MAX_ITERS}" ]]; then
    runner_args+=(--max_iters "${MAX_ITERS}")
    if [[ "${FIXED_ITERS}" == "true" || "${FIXED_ITERS}" == "1" ]]; then
      runner_args+=(--fixed_iters)
    fi
  fi
  set +e
  LD_LIBRARY_PATH="${MESA_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" \
    "${runner_args[@]}" 2>&1 | tee "${log_path}"
  run_status=${PIPESTATUS[0]}
  set -e

  global_cost=""
  mean_residual=""
  result_dir="$(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1 || true)"
  if [[ -n "${result_dir}" ]]; then
    final_results_path="${result_dir}/final_results.jrr.cbor"
    final_metrics_path="${result_dir}/final_metrics.jrm.cbor"
    comm_path="${result_dir}/communication_counts.txt"
    iter_summary_path="${result_dir}/iteration_summary.csv"
  fi
  if [[ -f "${final_metrics_path}" ]]; then
    mean_residual="$(read_cbor_mean_residual "${final_metrics_path}")"
  fi
  total_comm=""
  if [[ -f "${comm_path}" ]]; then
    total_comm="$(awk '{for (i = 1; i <= NF; ++i) last = $i} END {print last}' "${comm_path}")"
  fi
  total_comm_mb=""
  if [[ -f "${iter_summary_path}" ]]; then
    global_cost="$(awk -F, 'NR > 1 {value=$2} END {print value}' "${iter_summary_path}")"
    mean_residual="$(awk -F, 'NR > 1 {value=$3} END {print value}' "${iter_summary_path}")"
    total_comm="$(awk -F, 'NR > 1 {value=$6} END {print value}' "${iter_summary_path}")"
    total_comm_mb="$(awk -F, 'NR > 1 {value=$7} END {print value}' "${iter_summary_path}")"
  fi

  status="ok"
  if [[ ${run_status} -ne 0 || -z "${global_cost}" ]]; then
    status="failed"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${dataset}" \
    "${method_name}" \
    "${status}" \
    "${global_cost}" \
    "${mean_residual}" \
    "" \
    "${total_comm}" \
    "${total_comm_mb}" \
    "${run_dir}" \
    "${jrl_path}" \
    "${log_path}" \
    "${final_results_path}" \
    "${final_metrics_path}" \
    "${comm_path}" \
    "${iter_summary_path}" >> "${SUMMARY_CSV}"
done

echo "Saved summary: ${SUMMARY_CSV}"
echo "Saved logs under: ${LOG_ROOT}"
echo "Saved run directories under: ${RUN_ROOT}"
echo "Cached JRL files under: ${JRL_ROOT}"

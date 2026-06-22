#!/usr/bin/env bash
set -euo pipefail

# Baseline2 for the cloned distributed-mapper project.
# Runs the bundled example datasets and records:
# - per-iteration communication CSVs
# - raw logs
# - final summary CSV

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DM_ROOT="${DM_ROOT:-${ROOT_DIR}/baselines/distributed-mapper/distributed_mapper_core/cpp}"
BIN="${BIN:-${DM_ROOT}/build/runDistributedMapper}"
DM_LD_LIBRARY_PATH="${DM_LD_LIBRARY_PATH:-${DM_ROOT}/build}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/distributed_mapper_baseline2_$(date +%Y%m%d_%H%M%S)}"
MAX_ITERS="${MAX_ITERS:-1000}"

declare -A DATASETS=(
  ["2robots"]="2:baselines/distributed-mapper/distributed_mapper_core/data/example_2robots"
  ["4robots"]="4:baselines/distributed-mapper/distributed_mapper_core/data/example_4robots"
  ["9robots"]="9:baselines/distributed-mapper/distributed_mapper_core/data/example_9robots"
  ["16robots"]="16:baselines/distributed-mapper/distributed_mapper_core/data/example_16robots"
  ["25robots"]="25:baselines/distributed-mapper/distributed_mapper_core/data/example_25robots"
  ["36robots"]="36:baselines/distributed-mapper/distributed_mapper_core/data/example_36robots"
  ["49robots"]="49:baselines/distributed-mapper/distributed_mapper_core/data/example_49robots"
)

DATASET_ORDER=("2robots" "4robots" "9robots" "16robots" "25robots" "36robots" "49robots")

if [[ ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN}" >&2
  echo "Build it first: cmake -S ${DM_ROOT} -B ${DM_ROOT}/build && cmake --build ${DM_ROOT}/build -j2" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/comm_traces" "${OUT_ROOT}/trace_files"

SUMMARY_CSV="${OUT_ROOT}/final_summary.csv"
printf "dataset,nr_robots,distributed_error,centralized_error,gn_error,total_comm_msgs,total_comm_bytes,total_comm_mb,comm_csv,log_path\n" > "${SUMMARY_CSV}"

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
  entry="${DATASETS[${dataset}]}"
  IFS=":" read -r nr_robots rel_path <<< "${entry}"
  data_dir="${ROOT_DIR}/${rel_path}"
  log_path="${OUT_ROOT}/logs/${dataset}.log"
  comm_csv="${OUT_ROOT}/comm_traces/${dataset}.csv"
  trace_base="${OUT_ROOT}/trace_files/${dataset}"

  if [[ ! -d "${data_dir}" ]]; then
    echo "[${dataset}] missing dataset directory: ${data_dir}" | tee "${log_path}"
    printf "%s,%s,,,,,,%s,%s\n" "${dataset}" "${nr_robots}" "${comm_csv}" "${log_path}" >> "${SUMMARY_CSV}"
    continue
  fi

  echo "=== Running ${dataset}: ${rel_path} ==="
  set +e
  LD_LIBRARY_PATH="${DM_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}" "${BIN}" --nrRobots "${nr_robots}" \
    --dataDir "${data_dir}/" \
    --maxIter "${MAX_ITERS}" \
    --traceFile "${trace_base}" \
    --commCsv "${comm_csv}" > "${log_path}" 2>&1
  run_status=$?
  set -e

  summary_line="$(grep '^SUMMARY' "${log_path}" | tail -n 1 || true)"
  if [[ -z "${summary_line}" ]]; then
    printf "%s,%s,,,,,,%s,%s\n" "${dataset}" "${nr_robots}" "${comm_csv}" "${log_path}" >> "${SUMMARY_CSV}"
    echo "[${dataset}] failed with status ${run_status}; no SUMMARY line found." >&2
    continue
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${dataset}" \
    "${nr_robots}" \
    "$(summary_value distributed_error "${summary_line}")" \
    "$(summary_value centralized_error "${summary_line}")" \
    "$(summary_value gn_error "${summary_line}")" \
    "$(summary_value total_comm_msgs "${summary_line}")" \
    "$(summary_value total_comm_bytes "${summary_line}")" \
    "$(summary_value total_comm_mb "${summary_line}")" \
    "${comm_csv}" \
    "${log_path}" >> "${SUMMARY_CSV}"
done

echo "Saved final summary: ${SUMMARY_CSV}"
echo "Saved logs and communication traces under: ${OUT_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DM_ROOT="$ROOT_DIR/baselines/distributed-mapper"
pick_binary() {
  local candidate
  for candidate in \
    "$DM_ROOT/distributed_mapper_core/cpp/build/runDistributedMapper" \
    "$DM_ROOT/distributed_mapper_core/cpp/build-local/runDistributedMapper"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

RUNNER="${RUNNER:-$(pick_binary || true)}"
BUILD_DIR="$(dirname "${RUNNER:-$DM_ROOT/distributed_mapper_core/cpp/build/runDistributedMapper}")"
DM_LD_LIBRARY_PATH="${DM_LD_LIBRARY_PATH:-$BUILD_DIR}"
SPLITTER="$ROOT_DIR/scripts/split_g2o_to_distributed_mapper.py"

NR_ROBOTS="${NUM_ROBOTS:-5}"
MAX_ITERS="${MAX_ITERS:-300}"
RTHRESH="${RTHRESH:--1}"
PTHRESH="${PTHRESH:--1}"
CHORDAL_INITIALIZATION="${CHORDAL_INITIALIZATION:-true}"
OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/results/distributed_mapper_baseline2_5robots_$(date +%Y%m%d_%H%M%S)}"
DATASET_FILTER="${DATASET_FILTER:-${DATASETS:-}}"

mkdir -p "$OUT_ROOT" "$OUT_ROOT/iterations" "$OUT_ROOT/comm" "$OUT_ROOT/logs" "$OUT_ROOT/splits"

DATASETS=(
  "parking-garage:parking-garage.g2o"
  "sphere:sphere2500.g2o"
  "torus:torus3D.g2o"
  "CSAIL:CSAIL.g2o"
  "inter:input_INTEL_g2o.g2o"
  "manhattan:input_M3500_g2o.g2o"
)

if [[ -n "${DATASET_FILTER}" ]]; then
  IFS=',' read -r -a REQUESTED_DATASETS <<< "${DATASET_FILTER}"
  FILTERED_DATASETS=()
  for requested in "${REQUESTED_DATASETS[@]}"; do
    found=false
    for item in "${DATASETS[@]}"; do
      if [[ "${item%%:*}" == "${requested}" ]]; then
        FILTERED_DATASETS+=("${item}")
        found=true
        break
      fi
    done
    if [[ "${found}" != "true" ]]; then
      echo "Unknown dataset filter entry: ${requested}" >&2
      exit 1
    fi
  done
  DATASETS=("${FILTERED_DATASETS[@]}")
fi

FINAL_SUMMARY="$OUT_ROOT/final_summary.tsv"
printf "dataset\tfinal_iter\tfinal_cost\tfinal_gradnorm\ttotal_comm_poses\ttotal_comm_kb\ttotal_comm_mb\tlog_path\tcsv_path\tcomm_path\tsplit_dir\tstatus\n" > "$FINAL_SUMMARY"

if [[ -z "$RUNNER" || ! -x "$RUNNER" ]]; then
  echo "runDistributedMapper not found. Checked cpp/build and cpp/build-local."
  exit 1
fi

parse_summary() {
  local summary_line="$1"
  python3 - "$summary_line" <<'PY'
import re
import sys

line = sys.argv[1]
pairs = dict(re.findall(r'(\w+)=([^\s]+)', line))
fields = [
    "final_iter",
    "final_cost",
    "final_gradnorm",
    "total_comm_msgs",
    "total_comm_kb",
    "total_comm_mb",
]
print(",".join(pairs.get(field, "") for field in fields))
PY
}

for item in "${DATASETS[@]}"; do
  dataset="${item%%:*}"
  g2o_file="${item##*:}"
  source_file="$ROOT_DIR/data/$g2o_file"
  split_dir="$OUT_ROOT/splits/$dataset"
  log_file="$OUT_ROOT/logs/$dataset.log"
  comm_csv="$OUT_ROOT/comm/$dataset.tsv"
  loss_csv="$OUT_ROOT/iterations/$dataset.tsv"

  set +e
  python3 "$SPLITTER" --input "$source_file" --output-dir "$split_dir" --robots "$NR_ROBOTS" \
    --chordal-initialization "$CHORDAL_INITIALIZATION" >"$log_file" 2>&1
  split_status=$?
  set -e
  if [[ $split_status -ne 0 ]]; then
    printf "%s\t\t\t\t\t\t\t%s\t%s\t%s\t%s\tsplit_failed_%s\n" \
      "$dataset" "$log_file" "$loss_csv" "$comm_csv" "$split_dir" "$split_status" >> "$FINAL_SUMMARY"
    continue
  fi

  set +e
  LD_LIBRARY_PATH="$DM_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}" "$RUNNER" \
    --nrRobots "$NR_ROBOTS" \
    --dataDir "$split_dir/" \
    --traceFile "$OUT_ROOT/$dataset" \
    --commCsv "$comm_csv" \
    --lossCsv "$loss_csv" \
    --rthresh "$RTHRESH" \
    --pthresh "$PTHRESH" \
    --maxIter "$MAX_ITERS" \
    >>"$log_file" 2>&1
  run_status=$?
  set -e

  summary_line=$(grep '^SUMMARY ' "$log_file" | tail -n 1 || true)
  if [[ -z "$summary_line" ]]; then
    echo "Missing SUMMARY line for $dataset" >&2
    printf "%s\t\t\t\t\t\t\t%s\t%s\t%s\t%s\tmissing_summary_%s\n" \
      "$dataset" "$log_file" "$loss_csv" "$comm_csv" "$split_dir" "$run_status" >> "$FINAL_SUMMARY"
    continue
  fi

  parsed=$(parse_summary "$summary_line")
  IFS=',' read -r final_iter final_cost final_gradnorm total_comm_poses total_comm_kb total_comm_mb <<< "$parsed"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$dataset" \
    "$final_iter" \
    "$final_cost" \
    "$final_gradnorm" \
    "$total_comm_poses" \
    "$total_comm_kb" \
    "$total_comm_mb" \
    "$log_file" \
    "$loss_csv" \
    "$comm_csv" \
    "$split_dir" \
    "ok" >> "$FINAL_SUMMARY"
done

echo "$FINAL_SUMMARY"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pick_binary() {
  local candidate
  for candidate in \
    "${ROOT_DIR}/baselines/DPGO_MM/C++/build-local/bin/dist_pgo" \
    "${ROOT_DIR}/baselines/DPGO_MM/release/bin/dist_pgo" \
    "${ROOT_DIR}/baselines/DPGO_MM/C++/build/bin/dist_pgo"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

BIN="${BIN:-$(pick_binary || true)}"
NUM_ROBOTS="${NUM_ROBOTS:-5}"
MAX_ITERS="${MAX_ITERS:-300}"
if (( MAX_ITERS > 0 )); then
  DPGO_MM_SOLVER_ITERS="${DPGO_MM_SOLVER_ITERS:-$((MAX_ITERS - 1))}"
else
  DPGO_MM_SOLVER_ITERS="${DPGO_MM_SOLVER_ITERS:-0}"
fi
LOSS="${LOSS:-trivial}"
DIST_INIT="${DIST_INIT:-true}"
ACCELERATED="${ACCELERATED:-true}"
SCHEME="${SCHEME:-${DPGO_MM_SCHEME:-auto}}"
PRECONDITIONER="${PRECONDITIONER:-${DPGO_MM_PRECONDITIONER:-regularized_cholesky}}"
LOCAL_MAX_ITERATIONS="${LOCAL_MAX_ITERATIONS:-${DPGO_MM_LOCAL_MAX_ITERATIONS:-10}}"
LOCAL_MAX_ITERATIONS_ACCEPTED="${LOCAL_MAX_ITERATIONS_ACCEPTED:-${DPGO_MM_LOCAL_MAX_ITERATIONS_ACCEPTED:-1}}"
LOCAL_MAX_TCG_ITERATIONS="${LOCAL_MAX_TCG_ITERATIONS:-${DPGO_MM_LOCAL_MAX_TCG_ITERATIONS:-10000}}"
LOCAL_GRAD_NORM_TOL="${LOCAL_GRAD_NORM_TOL:-${DPGO_MM_LOCAL_GRAD_NORM_TOL:-0.001}}"
LOCAL_PRECONDITIONED_GRAD_NORM_TOL="${LOCAL_PRECONDITIONED_GRAD_NORM_TOL:-${DPGO_MM_LOCAL_PRECONDITIONED_GRAD_NORM_TOL:-0.0001}}"
TNT_KAPPA="${TNT_KAPPA:-${DPGO_MM_TNT_KAPPA:-0.05}}"
TNT_THETA="${TNT_THETA:-${DPGO_MM_TNT_THETA:-0.9}}"
INIT_RED_ROT_ITERS="${INIT_RED_ROT_ITERS:-${DPGO_MM_INIT_RED_ROT_ITERS:-100}}"
INIT_ROT_ITERS="${INIT_ROT_ITERS:-${DPGO_MM_INIT_ROT_ITERS:-400}}"
INIT_RED_TRANS_ITERS="${INIT_RED_TRANS_ITERS:-${DPGO_MM_INIT_RED_TRANS_ITERS:-150}}"
INIT_TRANS_ITERS="${INIT_TRANS_ITERS:-${DPGO_MM_INIT_TRANS_ITERS:-250}}"
AMM_TRACE="${AMM_TRACE:-${DPGO_MM_AMM_TRACE:-false}}"
SAVE="${SAVE:-true}"
DATASET_FILTER="${DATASETS:-}"
DRY_RUN="${DRY_RUN:-false}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/dpgo_mm_six_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${OUT_ROOT}/logs"
RUN_ROOT="${OUT_ROOT}/runs"
SUMMARY_CSV="${OUT_ROOT}/summary.csv"

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

result_suffix="mm"
if [[ "${SCHEME}" == "amm" || ( "${SCHEME}" == "auto" && "${ACCELERATED}" == "true" ) ]]; then
  result_suffix="amm"
fi

if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
  for dataset in "${DATASET_ORDER[@]}"; do
    rel_path="${DATASET_PATHS[${dataset}]:-}"
    if [[ -z "${rel_path}" ]]; then
      echo "DRY_RUN unknown_dataset=${dataset}"
      continue
    fi
    data_path="${ROOT_DIR}/${rel_path}"
    run_dir="${RUN_ROOT}/${dataset}"
    cmd=(
      "${BIN:-<missing-dist-pgo>}"
      --dataset "${data_path}"
      --num_nodes "${NUM_ROBOTS}"
      --iters "${DPGO_MM_SOLVER_ITERS}"
      --loss "${LOSS}"
      --dist_init "${DIST_INIT}"
      --accelerated "${ACCELERATED}"
      --scheme "${SCHEME}"
      --preconditioner "${PRECONDITIONER}"
      --local_max_iterations "${LOCAL_MAX_ITERATIONS}"
      --local_max_iterations_accepted "${LOCAL_MAX_ITERATIONS_ACCEPTED}"
      --local_max_tcg_iterations "${LOCAL_MAX_TCG_ITERATIONS}"
      --local_grad_norm_tol "${LOCAL_GRAD_NORM_TOL}"
      --local_preconditioned_grad_norm_tol "${LOCAL_PRECONDITIONED_GRAD_NORM_TOL}"
      --tnt_kappa "${TNT_KAPPA}"
      --tnt_theta "${TNT_THETA}"
      --init_red_rot_iters "${INIT_RED_ROT_ITERS}"
      --init_rot_iters "${INIT_ROT_ITERS}"
      --init_red_trans_iters "${INIT_RED_TRANS_ITERS}"
      --init_trans_iters "${INIT_TRANS_ITERS}"
      --trace_amm "${AMM_TRACE}"
      --save "${SAVE}"
    )
    printf 'DRY_RUN dataset=%s path=%s run_dir=%s result_suffix=%s command=' \
      "${dataset}" "${rel_path}" "${run_dir}" "${result_suffix}"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  done
  exit 0
fi

if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
  echo "Missing executable: ${BIN:-<unset>}" >&2
  echo "Expected one of:" >&2
  echo "  ${ROOT_DIR}/baselines/DPGO_MM/C++/build-local/bin/dist_pgo" >&2
  echo "  ${ROOT_DIR}/baselines/DPGO_MM/release/bin/dist_pgo" >&2
  echo "  ${ROOT_DIR}/baselines/DPGO_MM/C++/build/bin/dist_pgo" >&2
  exit 1
fi

BIN_ROOT="$(cd "$(dirname "${BIN}")/.." && pwd)"
export LD_LIBRARY_PATH="${BIN_ROOT}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"

printf 'dataset,method,status,scheme,accelerated,dist_init,preconditioner,local_max_iterations,local_max_iterations_accepted,local_max_tcg_iterations,init_red_rot_iters,init_rot_iters,init_red_trans_iters,init_trans_iters,global_cost,gradient,total_comm_poses,total_comm_mb,solver_time_per_node_sec,wall_time_sec,output_dir,result_path,estimate_path,log_path,final_objective,final_gradient,iter_summary_path,amm_trace_path\n' > "${SUMMARY_CSV}"

for dataset in "${DATASET_ORDER[@]}"; do
  rel_path="${DATASET_PATHS[${dataset}]:-}"
  data_path="${ROOT_DIR}/${rel_path}"
  run_dir="${RUN_ROOT}/${dataset}"
  log_path="${LOG_ROOT}/${dataset}.log"
  base_name="$(basename "${data_path}" .g2o)"
  result_path="${run_dir}/results_chordal_${base_name}_${NUM_ROBOTS}_${result_suffix}.txt"
  estimate_path="${run_dir}/estimates_${LOSS}.txt"
  iter_summary_path="${run_dir}/iteration_summary.csv"
  amm_trace_path="${run_dir}/amm_trace.csv"
  method_name="${METHOD_NAME:-dpgo_mm_${result_suffix}}"

  mkdir -p "${run_dir}"

  if [[ ! -f "${data_path}" ]]; then
    echo "[${dataset}] missing dataset: ${data_path}" | tee "${log_path}"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${dataset}" "${method_name}" "missing_dataset" "${SCHEME}" "${ACCELERATED}" \
      "${DIST_INIT}" "${PRECONDITIONER}" "${LOCAL_MAX_ITERATIONS}" \
      "${LOCAL_MAX_ITERATIONS_ACCEPTED}" "${LOCAL_MAX_TCG_ITERATIONS}" \
      "${INIT_RED_ROT_ITERS}" "${INIT_ROT_ITERS}" "${INIT_RED_TRANS_ITERS}" \
      "${INIT_TRANS_ITERS}" "" "" "" "" "" "" "${run_dir}" "" "" \
      "${log_path}" "" "" "" "" >> "${SUMMARY_CSV}"
    continue
  fi

  echo "=== DPGO-MM ${dataset}: ${rel_path} ==="
  start_ns="$(date +%s%N)"
  set +e
  (
    cd "${run_dir}"
      "${BIN}" \
      --dataset "${data_path}" \
      --num_nodes "${NUM_ROBOTS}" \
      --iters "${DPGO_MM_SOLVER_ITERS}" \
      --loss "${LOSS}" \
      --dist_init "${DIST_INIT}" \
      --accelerated "${ACCELERATED}" \
      --scheme "${SCHEME}" \
      --preconditioner "${PRECONDITIONER}" \
      --local_max_iterations "${LOCAL_MAX_ITERATIONS}" \
      --local_max_iterations_accepted "${LOCAL_MAX_ITERATIONS_ACCEPTED}" \
      --local_max_tcg_iterations "${LOCAL_MAX_TCG_ITERATIONS}" \
      --local_grad_norm_tol "${LOCAL_GRAD_NORM_TOL}" \
      --local_preconditioned_grad_norm_tol "${LOCAL_PRECONDITIONED_GRAD_NORM_TOL}" \
      --tnt_kappa "${TNT_KAPPA}" \
      --tnt_theta "${TNT_THETA}" \
      --init_red_rot_iters "${INIT_RED_ROT_ITERS}" \
      --init_rot_iters "${INIT_ROT_ITERS}" \
      --init_red_trans_iters "${INIT_RED_TRANS_ITERS}" \
      --init_trans_iters "${INIT_TRANS_ITERS}" \
      --trace_amm "${AMM_TRACE}" \
      --save "${SAVE}"
  ) 2>&1 | tee "${log_path}"
  run_status=${PIPESTATUS[0]}
  set -e
  end_ns="$(date +%s%N)"
  wall_time_sec="$(awk -v start="${start_ns}" -v end="${end_ns}" 'BEGIN {printf "%.9f", (end - start) / 1000000000.0}')"

  final_objective="$(awk -F': *' '/^final objective:/ {value=$2} END {print value}' "${log_path}")"
  final_gradient="$(awk -F': *' '/^final gradient:/ {value=$2} END {print value}' "${log_path}")"
  solver_time_per_node_sec="$(awk '/^time:/ {value=$2} END {print value}' "${log_path}")"
  total_comm_poses=""
  total_comm_mb=""
  if [[ -f "${iter_summary_path}" ]]; then
    total_comm_poses="$(awk -F, 'NR > 1 {value=$7} END {print value}' "${iter_summary_path}")"
    total_comm_mb="$(awk -F, 'NR > 1 {value=$8} END {print value}' "${iter_summary_path}")"
  fi

  status="ok"
  if [[ ${run_status} -ne 0 || -z "${final_objective}" || -z "${final_gradient}" ]]; then
    status="failed"
  fi

  if [[ "${AMM_TRACE}" != "true" && "${AMM_TRACE}" != "1" ]]; then
    amm_trace_path=""
  elif [[ ! -f "${amm_trace_path}" ]]; then
    amm_trace_path=""
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${dataset}" \
    "${method_name}" \
    "${status}" \
    "${SCHEME}" \
    "${ACCELERATED}" \
    "${DIST_INIT}" \
    "${PRECONDITIONER}" \
    "${LOCAL_MAX_ITERATIONS}" \
    "${LOCAL_MAX_ITERATIONS_ACCEPTED}" \
    "${LOCAL_MAX_TCG_ITERATIONS}" \
    "${INIT_RED_ROT_ITERS}" \
    "${INIT_ROT_ITERS}" \
    "${INIT_RED_TRANS_ITERS}" \
    "${INIT_TRANS_ITERS}" \
    "${final_objective}" \
    "${final_gradient}" \
    "${total_comm_poses}" \
    "${total_comm_mb}" \
    "${solver_time_per_node_sec}" \
    "${wall_time_sec}" \
    "${run_dir}" \
    "${result_path}" \
    "${estimate_path}" \
    "${log_path}" \
    "${final_objective}" \
    "${final_gradient}" \
    "${iter_summary_path}" \
    "${amm_trace_path}" >> "${SUMMARY_CSV}"
done

echo "Saved summary: ${SUMMARY_CSV}"
echo "Saved logs under: ${LOG_ROOT}"
echo "Saved run directories under: ${RUN_ROOT}"

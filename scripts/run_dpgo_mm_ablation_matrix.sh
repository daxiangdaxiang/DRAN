#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/dpgo_mm_ablation_matrix_$(date +%Y%m%d_%H%M%S)}"
MAX_ITERS="${MAX_ITERS:-20}"
DATASETS="${DATASETS:-sphere,torus}"
ABLATIONS="${ABLATIONS:-baseline,no_amm,no_preconditioner,weak_init,no_tnt,centralized_init,centralized_init_no_amm}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${OUT_ROOT}"

run_case() {
  local ablation="$1"
  local scheme="amm"
  local accelerated="true"
  local dist_init="true"
  local preconditioner="regularized_cholesky"
  local local_max_iterations="10"
  local local_max_iterations_accepted="1"
  local init_red_rot_iters="100"
  local init_rot_iters="400"
  local init_red_trans_iters="150"
  local init_trans_iters="250"

  case "${ablation}" in
    baseline)
      ;;
    no_amm)
      scheme="mm"
      accelerated="false"
      ;;
    no_preconditioner)
      preconditioner="none"
      ;;
    no_tnt)
      local_max_iterations="0"
      local_max_iterations_accepted="0"
      ;;
    weak_init)
      init_red_rot_iters="10"
      init_rot_iters="40"
      init_red_trans_iters="15"
      init_trans_iters="25"
      ;;
    centralized_init)
      dist_init="false"
      ;;
    centralized_init_no_amm)
      scheme="mm"
      accelerated="false"
      dist_init="false"
      ;;
    *)
      echo "Unknown ablation: ${ablation}" >&2
      return 2
      ;;
  esac

  local case_root="${OUT_ROOT}/${ablation}"
  local command="./scripts/run_dpgo_mm_six.sh"
  local rendered="${command} --scheme ${scheme} --accelerated ${accelerated} --dist_init ${dist_init} --preconditioner ${preconditioner} --local_max_iterations ${local_max_iterations} --local_max_iterations_accepted ${local_max_iterations_accepted} --init_red_rot_iters ${init_red_rot_iters} --init_rot_iters ${init_rot_iters} --init_red_trans_iters ${init_red_trans_iters} --init_trans_iters ${init_trans_iters}"

  if [[ "${DRY_RUN}" == "true" ]]; then
    IFS=',' read -r -a dry_datasets <<< "${DATASETS}"
    for dataset in "${dry_datasets[@]}"; do
      printf '%s,%s,%s\n' "${ablation}" "${dataset}" "${rendered}"
    done
    return 0
  fi

  env \
    OUT_ROOT="${case_root}" \
    METHOD_NAME="dpgo_mm_${ablation}" \
    MAX_ITERS="${MAX_ITERS}" \
    DATASETS="${DATASETS}" \
    SCHEME="${scheme}" \
    ACCELERATED="${accelerated}" \
    DIST_INIT="${dist_init}" \
    PRECONDITIONER="${preconditioner}" \
    LOCAL_MAX_ITERATIONS="${local_max_iterations}" \
    LOCAL_MAX_ITERATIONS_ACCEPTED="${local_max_iterations_accepted}" \
    INIT_RED_ROT_ITERS="${init_red_rot_iters}" \
    INIT_ROT_ITERS="${init_rot_iters}" \
    INIT_RED_TRANS_ITERS="${init_red_trans_iters}" \
    INIT_TRANS_ITERS="${init_trans_iters}" \
    "${ROOT_DIR}/scripts/run_dpgo_mm_six.sh"
}

IFS=',' read -r -a ablations <<< "${ABLATIONS}"
for ablation in "${ablations[@]}"; do
  run_case "${ablation}"
done

if [[ "${DRY_RUN}" != "true" ]]; then
  echo "Saved DPGO-MM ablation matrix under: ${OUT_ROOT}"
fi

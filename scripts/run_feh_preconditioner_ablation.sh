#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${RUNNER:-${ROOT_DIR}/scripts/run_manual_dpgo_mm_six.sh}"

DATASET_FILTER="${DATASETS:-parking-garage,sphere,torus,CSAIL,inter,manhattan}"
VARIANT_FILTER="${VARIANTS:-block_jacobi,local_chain,laplacian_deflation,rqn_memory,translation_block,translation_sparse_schur,translation_sparse_schur_laplacian,translation_local_schur,translation_local_schur_full,translation_schur,rr_cholesky,rr_jacobi,rr_schur_jacobi}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/results/feh_preconditioner_ablation_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"

NUM_ROBOTS="${NUM_ROBOTS:-5}"
MAX_ITERS="${MAX_ITERS:-20}"
SCHEME="${SCHEME:-mm}"
ACCELERATED="${ACCELERATED:-false}"
DIST_INIT="${DIST_INIT:-false}"
FEH_LINEAR_REL_TOL="${FEH_LINEAR_REL_TOL:-1e-8}"
FEH_LINEAR_ABS_TOL="${FEH_LINEAR_ABS_TOL:-1e-12}"
FEH_LINEAR_MAX_ITERS="${FEH_LINEAR_MAX_ITERS:-200}"
FEH_SCHUR_DAMPING="${FEH_SCHUR_DAMPING:-1e-6}"
FEH_SPARSE_MATVEC="${FEH_SPARSE_MATVEC:-true}"
FEH_PRECOND_TRANSLATION_LOCAL_SCHUR_MAX_POSES="${FEH_PRECOND_TRANSLATION_LOCAL_SCHUR_MAX_POSES:-32}"
FEH_PRECOND_LAPLACIAN_DEFLATION_BASIS_SIZE="${FEH_PRECOND_LAPLACIAN_DEFLATION_BASIS_SIZE:-4}"
FEH_PRECOND_LAPLACIAN_DEFLATION_MAX_EIGEN_POSES="${FEH_PRECOND_LAPLACIAN_DEFLATION_MAX_EIGEN_POSES:-512}"
FEH_FINAL_POLISH_RTR="${FEH_FINAL_POLISH_RTR:-false}"
FEH_RQN_WARM_START="${FEH_RQN_WARM_START:-false}"
FEH_PRECOND_RQN_MEMORY="${FEH_PRECOND_RQN_MEMORY:-false}"
FEH_RQN_MEMORY_SIZE="${FEH_RQN_MEMORY_SIZE:-5}"
FEH_RQN_MIN_CURVATURE_RATIO="${FEH_RQN_MIN_CURVATURE_RATIO:-1e-8}"
SAVE="${SAVE:-true}"

if [[ "${OUT_ROOT}" != /* ]]; then
  OUT_ROOT="${ROOT_DIR}/${OUT_ROOT}"
fi

sanitize() {
  local value="$1"
  printf '%s' "${value}" | tr '[:upper:]' '[:lower:]'
}

is_true() {
  local value
  value="$(sanitize "$1")"
  [[ "${value}" == "1" || "${value}" == "true" || "${value}" == "yes" || "${value}" == "on" ]]
}

variant_config() {
  local variant="$1"
  case "${variant}" in
    block_jacobi)
      printf '%s\n' "true false false false false false false false false none"
      ;;
    local_chain)
      printf '%s\n' "true false true false false false false false false none"
      ;;
    laplacian_deflation)
      printf '%s\n' "true false false false false false true false false none"
      ;;
    rqn_memory)
      printf '%s\n' "true false false false false false false true false none"
      ;;
    translation_block)
      printf '%s\n' "true false false true false false false false false none"
      ;;
    translation_sparse_schur)
      printf '%s\n' "true false false false true false false false false none"
      ;;
    translation_sparse_schur_laplacian)
      printf '%s\n' "true false false false true false true false false none"
      ;;
    translation_local_schur)
      printf '%s\n' "true false false false false true false false false none"
      ;;
    translation_local_schur_full)
      printf '%s\n' "true false false false false true false false false none"
      ;;
    translation_schur)
      printf '%s\n' "false true false false false false false false false none"
      ;;
    rr_cholesky)
      printf '%s\n' "false false false false false false false false true cholesky"
      ;;
    rr_jacobi)
      printf '%s\n' "false false false false false false false false true jacobi"
      ;;
    rr_schur_jacobi)
      printf '%s\n' "false false false false false false false false true schur_jacobi"
      ;;
    *)
      echo "Unknown FE-Hybrid preconditioner variant: ${variant}" >&2
      exit 2
      ;;
  esac
}

IFS=',' read -r -a VARIANTS_ARRAY <<< "${VARIANT_FILTER}"
mkdir -p "${OUT_ROOT}"

variant_specs=()
for variant in "${VARIANTS_ARRAY[@]}"; do
  read -r block_jacobi translation_schur local_chain translation_block translation_sparse_schur translation_local_schur laplacian_deflation rqn_memory reduced_rotation reduced_mode <<< "$(variant_config "${variant}")"
  translation_local_schur_max_poses="${FEH_PRECOND_TRANSLATION_LOCAL_SCHUR_MAX_POSES}"
  if [[ "${variant}" == "translation_local_schur_full" ]]; then
    translation_local_schur_max_poses="0"
  fi
  variant_root="${OUT_ROOT}/variants/${variant}"
  variant_specs+=("${variant}=${variant_root}")

  echo "variant=${variant} dataset=${DATASET_FILTER}"
  OUT_ROOT="${variant_root}" \
  METHOD_NAME="feh_${variant}" \
  DATASETS="${DATASET_FILTER}" \
  NUM_ROBOTS="${NUM_ROBOTS}" \
  MAX_ITERS="${MAX_ITERS}" \
  SCHEME="${SCHEME}" \
  ACCELERATED="${ACCELERATED}" \
  DIST_INIT="${DIST_INIT}" \
  LOCAL_SOLVER="full_equiv_hybrid" \
  FEH_LINEAR_BACKEND="pcg_full" \
  FEH_LINEAR_REL_TOL="${FEH_LINEAR_REL_TOL}" \
  FEH_LINEAR_ABS_TOL="${FEH_LINEAR_ABS_TOL}" \
  FEH_LINEAR_MAX_ITERS="${FEH_LINEAR_MAX_ITERS}" \
  FEH_SCHUR_DAMPING="${FEH_SCHUR_DAMPING}" \
  FEH_SPARSE_MATVEC="${FEH_SPARSE_MATVEC}" \
  FEH_LINEAR_BLOCK_JACOBI="${block_jacobi}" \
  FEH_PRECOND_TRANSLATION_SCHUR="${translation_schur}" \
  FEH_PRECOND_LOCAL_CHAIN="${local_chain}" \
  FEH_PRECOND_TRANSLATION_BLOCK="${translation_block}" \
  FEH_PRECOND_TRANSLATION_SPARSE_SCHUR="${translation_sparse_schur}" \
  FEH_PRECOND_TRANSLATION_LOCAL_SCHUR="${translation_local_schur}" \
  FEH_PRECOND_TRANSLATION_LOCAL_SCHUR_MAX_POSES="${translation_local_schur_max_poses}" \
  FEH_PRECOND_LAPLACIAN_DEFLATION="${laplacian_deflation}" \
  FEH_PRECOND_LAPLACIAN_DEFLATION_BASIS_SIZE="${FEH_PRECOND_LAPLACIAN_DEFLATION_BASIS_SIZE}" \
  FEH_PRECOND_LAPLACIAN_DEFLATION_MAX_EIGEN_POSES="${FEH_PRECOND_LAPLACIAN_DEFLATION_MAX_EIGEN_POSES}" \
  FEH_PRECOND_REDUCED_ROTATION="${reduced_rotation}" \
  REDUCED_ROTATION_PRECONDITIONER="${reduced_mode}" \
  FEH_REDUCED_ROTATION_INITIAL_GUESS="false" \
  FEH_RQN_WARM_START="${FEH_RQN_WARM_START}" \
  FEH_PRECOND_RQN_MEMORY="${rqn_memory}" \
  FEH_RQN_MEMORY_SIZE="${FEH_RQN_MEMORY_SIZE}" \
  FEH_RQN_MIN_CURVATURE_RATIO="${FEH_RQN_MIN_CURVATURE_RATIO}" \
  FEH_FINAL_POLISH_RTR="${FEH_FINAL_POLISH_RTR}" \
  SAVE="${SAVE}" \
  DRY_RUN="${DRY_RUN}" \
  "${RUNNER}"
done

summary_path="${OUT_ROOT}/preconditioner_summary.csv"
python3 "${ROOT_DIR}/scripts/collect_feh_preconditioner_ablation.py" \
  "${variant_specs[@]}" \
  -o "${summary_path}"

echo "Saved FE-Hybrid preconditioner summary: ${summary_path}"
if is_true "${DRY_RUN}"; then
  echo "DRY_RUN complete; no solver binary execution was attempted."
fi

/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef PCG_SOLVER_H
#define PCG_SOLVER_H

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace DPGO {

struct PCGParams {
  unsigned max_iters = 200;
  double rel_tol = 1e-8;
  double abs_tol = 1e-12;
  bool verbose = false;
  std::function<void(unsigned, const Eigen::VectorXd &, double, bool)>
      iteration_callback;
};

struct PCGResult {
  Eigen::VectorXd x;
  unsigned iters = 0;
  double final_residual_norm = 0.0;
  bool converged = false;
};

namespace pcg_internal {

inline void validateVectorSize(const Eigen::VectorXd &v, int expected,
                               const char *name) {
  if (v.size() != expected) {
    throw std::invalid_argument(std::string(name) +
                                " returned vector with wrong size");
  }
}

inline double checkedSqrt(double value) {
  return std::sqrt(std::max(0.0, value));
}

}  // namespace pcg_internal

template <typename ApplyH, typename ApplyMInv, typename Dot>
PCGResult solvePCG(const Eigen::VectorXd &b, ApplyH applyH, ApplyMInv applyMInv,
                   Dot dot, const PCGParams &params) {
  if (params.rel_tol < 0.0 || params.abs_tol < 0.0) {
    throw std::invalid_argument("PCG tolerances must be non-negative");
  }

  PCGResult result;
  result.x = Eigen::VectorXd::Zero(b.size());

  if (b.size() == 0) {
    result.converged = true;
    return result;
  }

  Eigen::VectorXd r = b;
  double bNorm = pcg_internal::checkedSqrt(dot(b, b));
  const double tolerance = std::max(params.abs_tol, params.rel_tol * bNorm);
  result.final_residual_norm = pcg_internal::checkedSqrt(dot(r, r));
  if (result.final_residual_norm <= tolerance) {
    result.converged = true;
    return result;
  }

  Eigen::VectorXd z = applyMInv(r);
  pcg_internal::validateVectorSize(z, b.size(), "applyMInv");
  Eigen::VectorXd p = z;
  double rz = dot(r, z);
  if (!std::isfinite(rz) || rz <= 0.0) {
    throw std::invalid_argument(
        "PCG requires a positive-definite preconditioner");
  }

  for (unsigned iter = 0; iter < params.max_iters; ++iter) {
    Eigen::VectorXd Hp = applyH(p);
    pcg_internal::validateVectorSize(Hp, b.size(), "applyH");
    const double denom = dot(p, Hp);
    if (!std::isfinite(denom) || denom <= 0.0) {
      throw std::invalid_argument("PCG requires an SPD linear operator");
    }

    const double alpha = rz / denom;
    result.x += alpha * p;
    r -= alpha * Hp;
    result.iters = iter + 1;
    result.final_residual_norm = pcg_internal::checkedSqrt(dot(r, r));
    result.converged = result.final_residual_norm <= tolerance;

    if (params.verbose) {
      std::cout << "PCG iter=" << result.iters
                << " residual=" << result.final_residual_norm << std::endl;
    }

    if (params.iteration_callback) {
      params.iteration_callback(result.iters, result.x,
                                result.final_residual_norm,
                                result.converged);
    }

    if (result.converged) {
      break;
    }

    z = applyMInv(r);
    pcg_internal::validateVectorSize(z, b.size(), "applyMInv");
    const double rzNext = dot(r, z);
    if (!std::isfinite(rzNext) || rzNext < 0.0) {
      throw std::invalid_argument(
          "PCG requires a positive-definite preconditioner");
    }
    const double beta = rzNext / rz;
    p = z + beta * p;
    rz = rzNext;
  }

  return result;
}

}  // namespace DPGO

#endif

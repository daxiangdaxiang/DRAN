/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef MANUALQUADRATICOPTIMIZER_H
#define MANUALQUADRATICOPTIMIZER_H

#include <DPGO/DPGO_types.h>
#include <DPGO/QuadraticProblem.h>

namespace DPGO {

/**
 * Lightweight handwritten trust-region optimizer for QuadraticProblem.
 *
 * This class deliberately does not call ROPTLIB solvers. It uses the local
 * quadratic matrix, tangent projection on LiftedSE, truncated CG, and a
 * projection retraction. The Hessian model is an embedded approximation used
 * for diagnostics and accelerator research, not a drop-in replacement for
 * ROPTLIB's full RTRNewton implementation.
 */
class ManualQuadraticOptimizer {
 public:
  explicit ManualQuadraticOptimizer(QuadraticProblem *p);

  Matrix optimize(const Matrix &Y);

  ROPTResult getOptResult() const { return result; }

  void setVerbose(bool flag) { verbose = flag; }
  void setTrustRegionIterations(unsigned iter) { maxIterations = iter; }
  void setTrustRegionAcceptedIterations(unsigned iter) {
    maxAcceptedIterations = iter;
  }
  void setTrustRegionTolerance(double tol) { gradientTolerance = tol; }
  void setTrustRegionInitialRadius(double radius) { initialRadius = radius; }
  void setTrustRegionMaxInnerIterations(int iter) { maxCgIterations = iter; }
  void setMaxIterations(unsigned iter) { maxIterations = iter; }
  void setMaxCgIterations(int iter) { maxCgIterations = iter; }
  void setGradientTolerance(double tol) { gradientTolerance = tol; }
  void setInitialDamping(double damping) { initialDamping = damping; }

 private:
  QuadraticProblem *problem;
  bool verbose;
  unsigned maxIterations;
  unsigned maxAcceptedIterations;
  int maxCgIterations;
  double gradientTolerance;
  double initialRadius;
  double initialDamping;
  ROPTResult result;

  Matrix projectTangent(const Matrix &Y, const Matrix &Z) const;
  Matrix retract(const Matrix &Y, const Matrix &eta) const;
  Matrix hessianEta(const Matrix &Y, const Matrix &eta,
                    const SparseMatrix &Q) const;
  Matrix polishTranslations(const Matrix &Y, const SparseMatrix &Q) const;
  Matrix polishMajorizedRotationBlocks(const Matrix &Y,
                                       const SparseMatrix &Q) const;
  Matrix solveSurrogateDirection(const Matrix &Y, const Matrix &grad,
                                 double radius,
                                 const SparseMatrix &Q) const;
  Matrix solveTruncatedCg(const Matrix &Y, const Matrix &grad, double radius,
                          double damping, const SparseMatrix &Q) const;
  static double innerProduct(const Matrix &A, const Matrix &B);
  static double boundaryTau(const Matrix &eta, const Matrix &p,
                            double radius);
};

}  // namespace DPGO

#endif

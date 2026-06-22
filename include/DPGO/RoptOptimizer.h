/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef ROPTOPTIMIZER_H
#define ROPTOPTIMIZER_H

#include <DPGO/DPGO_types.h>

#include "Problems/Problem.h"

namespace DPGO {

class RoptOptimizer {
 public:
  RoptOptimizer(ROPTLIB::Problem *p, unsigned int nIn, unsigned int dIn,
                unsigned int rIn);

  Matrix optimize(const Matrix &Y);

  void setProblem(ROPTLIB::Problem *p) { problem = p; }
  void setVerbose(bool v) { verbose = v; }
  void setAlgorithm(ROPTALG alg) { algorithm = alg; }
  void setGradientDescentStepsize(double s) { gradientDescentStepsize = s; }
  void setTrustRegionIterations(unsigned int iter) {
    trustRegionIterations = iter;
  }
  void setTrustRegionTolerance(double tol) { trustRegionTolerance = tol; }
  void setTrustRegionInitialRadius(double radius) {
    trustRegionInitialRadius = radius;
  }
  void setTrustRegionMaxInnerIterations(int iter) {
    trustRegionMaxInnerIterations = iter;
  }

  ROPTResult getOptResult() const { return result; }
  double f(const Matrix &Y) const;
  Matrix RieGrad(const Matrix &Y) const;
  double RieGradNorm(const Matrix &Y) const;

 private:
  ROPTLIB::Problem *problem;
  unsigned int n;
  unsigned int d;
  unsigned int r;
  ROPTALG algorithm;
  ROPTResult result;
  double gradientDescentStepsize;
  unsigned int trustRegionIterations;
  double trustRegionTolerance;
  double trustRegionInitialRadius;
  int trustRegionMaxInnerIterations;
  bool verbose;
  bool lastOptimizationFailed;

  Matrix trustRegion(const Matrix &Yinit);
  Matrix gradientDescent(const Matrix &Yinit);
};

}  // namespace DPGO

#endif

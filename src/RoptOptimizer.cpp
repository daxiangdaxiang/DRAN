/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/RoptOptimizer.h>

#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include "RSD.h"
#include "RTRNewton.h"

namespace DPGO {

RoptOptimizer::RoptOptimizer(ROPTLIB::Problem *p, unsigned int nIn,
                             unsigned int dIn, unsigned int rIn)
    : problem(p),
      n(nIn),
      d(dIn),
      r(rIn),
      algorithm(ROPTALG::RTR),
      gradientDescentStepsize(1e-3),
      trustRegionIterations(1),
      trustRegionTolerance(1e-2),
      trustRegionInitialRadius(1e1),
      trustRegionMaxInnerIterations(50),
      verbose(false),
      lastOptimizationFailed(false) {
  result.success = false;
}

double RoptOptimizer::f(const Matrix &Y) const {
  assert(problem != nullptr);
  assert((unsigned int)Y.rows() == r);
  assert((unsigned int)Y.cols() == (d + 1) * n);
  LiftedSEVariable var(r, d, n);
  var.setData(Y);
  return problem->f(var.var());
}

Matrix RoptOptimizer::RieGrad(const Matrix &Y) const {
  assert(problem != nullptr);
  assert((unsigned int)Y.rows() == r);
  assert((unsigned int)Y.cols() == (d + 1) * n);

  LiftedSEVariable var(r, d, n);
  LiftedSEVector eGrad(r, d, n);
  LiftedSEVector rGrad(r, d, n);
  LiftedSEManifold manifold(r, d, n);

  var.setData(Y);
  problem->EucGrad(var.var(), eGrad.vec());
  manifold.getManifold()->Projection(var.var(), eGrad.vec(), rGrad.vec());
  return rGrad.getData();
}

double RoptOptimizer::RieGradNorm(const Matrix &Y) const {
  return RieGrad(Y).norm();
}

Matrix RoptOptimizer::optimize(const Matrix &Y) {
  assert(problem != nullptr);
  result = ROPTResult();
  lastOptimizationFailed = false;
  result.fInit = f(Y);
  result.gradNormInit = RieGradNorm(Y);
  const auto startTime = std::chrono::high_resolution_clock::now();

  Matrix YOpt;
  if (algorithm == ROPTALG::RTR) {
    YOpt = trustRegion(Y);
  } else {
    assert(algorithm == ROPTALG::RGD);
    YOpt = gradientDescent(Y);
  }

  const auto elapsed = std::chrono::high_resolution_clock::now() - startTime;
  result.elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  result.fOpt = f(YOpt);
  result.gradNormOpt = RieGradNorm(YOpt);
  result.relativeChange = std::sqrt((YOpt - Y).squaredNorm() / n);
  result.success = !lastOptimizationFailed;
  return YOpt;
}

Matrix RoptOptimizer::trustRegion(const Matrix &Yinit) {
  const double gn0 = RieGradNorm(Yinit);
  if (gn0 < trustRegionTolerance) {
    return Yinit;
  }

  auto runTrustRegion = [&](double initialRadius, double maxRadius,
                            unsigned int maxIterations,
                            bool *accepted) {
    LiftedSEVariable varInit(r, d, n);
    varInit.setData(Yinit);
    varInit.var()->NewMemoryOnWrite();

    ROPTLIB::RTRNewton solver(problem, varInit.var());
    solver.Stop_Criterion = ROPTLIB::StopCrit::GRAD_F;
    solver.Tolerance = trustRegionTolerance;
    solver.initial_Delta = initialRadius;
    solver.maximum_Delta = maxRadius;
    solver.Debug = verbose ? ROPTLIB::DEBUGINFO::ITERRESULT
                           : ROPTLIB::DEBUGINFO::NOOUTPUT;
    solver.Max_Iteration = static_cast<int>(maxIterations);
    solver.Min_Inner_Iter = 0;
    solver.Max_Inner_Iter = trustRegionMaxInnerIterations;
    solver.TimeBound = 5.0;
    solver.Run();
    if (accepted != nullptr) {
      *accepted = solver.latestStepAccepted();
    }
    result.tCGStatus = solver.gettCGStatus();

    const auto *Yopt =
        dynamic_cast<const ROPTLIB::ProductElement *>(solver.GetXopt());
    LiftedSEVariable varOpt(r, d, n);
    Yopt->CopyTo(varOpt.var());
    return varOpt.getData();
  };

  if (trustRegionIterations == 1) {
    double radius = trustRegionInitialRadius;
    int rejectedSteps = 0;
    while (true) {
      bool accepted = false;
      Matrix candidate = runTrustRegion(radius, radius, 1, &accepted);
      if (accepted) {
        result.rtrAcceptedRadius = radius;
        result.rtrRejectedSteps = static_cast<unsigned int>(rejectedSteps);
        return candidate;
      }
      if (rejectedSteps > 10) {
        result.rtrAcceptedRadius = radius;
        result.rtrRejectedSteps = static_cast<unsigned int>(rejectedSteps);
        lastOptimizationFailed = true;
        return Yinit;
      }
      radius /= 4.0;
      ++rejectedSteps;
      if (verbose) {
        std::cout << "RTR step rejected. Shrinking trust-region radius to "
                  << radius << std::endl;
      }
    }
  }
  result.rtrAcceptedRadius = trustRegionInitialRadius;
  return runTrustRegion(trustRegionInitialRadius,
                        5.0 * trustRegionInitialRadius,
                        trustRegionIterations, nullptr);
}

Matrix RoptOptimizer::gradientDescent(const Matrix &Yinit) {
  LiftedSEVariable varInit(r, d, n);
  LiftedSEVariable varNext(r, d, n);
  LiftedSEVector rGrad(r, d, n);
  LiftedSEManifold manifold(r, d, n);

  varInit.setData(Yinit);
  problem->EucGrad(varInit.var(), rGrad.vec());
  manifold.getManifold()->Projection(varInit.var(), rGrad.vec(), rGrad.vec());
  manifold.getManifold()->ScaleTimesVector(varInit.var(),
                                           -gradientDescentStepsize,
                                           rGrad.vec(), rGrad.vec());
  manifold.getManifold()->Retraction(varInit.var(), rGrad.vec(),
                                     varNext.var());
  return varNext.getData();
}

}  // namespace DPGO

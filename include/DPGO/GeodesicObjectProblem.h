/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef GEODESICOBJECTPROBLEM_H
#define GEODESICOBJECTPROBLEM_H

#include <DPGO/DPGO_types.h>
#include <DPGO/manifold/LiftedSEManifold.h>

#include <vector>

#include "Problems/Problem.h"

namespace DPGO {

struct GeodesicMeasurementTerm {
  size_t firstPose{0};
  size_t secondPose{0};
  Matrix relativeRotation;
  Matrix relativeTranslation;
  double rotationalPrecision{1.0};
  double translationalPrecision{1.0};
};

struct GeodesicConsensusTerm {
  size_t pose{0};
  Matrix targetPose;
  double beta{1.0};
  double weight{1.0};
};

std::vector<GeodesicConsensusTerm> makeGeodesicProximalTerms(
    const Matrix &referenceState, double beta);

Matrix limitGeodesicStateStep(const Matrix &previousState,
                              const Matrix &trialState,
                              double maxBlockStep,
                              double *appliedScale = nullptr,
                              double *maxBlockDelta = nullptr);

class GeodesicObjectProblem : public ROPTLIB::Problem {
 public:
  explicit GeodesicObjectProblem(size_t numPoses);
  ~GeodesicObjectProblem() override;

  size_t num_poses() const { return n; }
  unsigned int dimension() const { return d; }
  unsigned int relaxation_rank() const { return r; }

  void setMeasurements(const std::vector<GeodesicMeasurementTerm> &terms);
  void setConsensusTerms(const std::vector<GeodesicConsensusTerm> &terms);
  void clearConsensusTerms();

  double f(const Matrix &Y) const;
  double f(ROPTLIB::Variable *x) const override;
  void EucGrad(ROPTLIB::Variable *x, ROPTLIB::Vector *g) const override;
  // Returns a Gauss-Newton Hessian-vector approximation J^T W J eta. This is
  // deliberate for the geodesic local solver; it is not the exact second
  // derivative of the nonlinear residual objective away from zero residuals.
  void EucHessianEta(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                     ROPTLIB::Vector *Hv) const override;

  Matrix RieGrad(const Matrix &Y) const;
  double RieGradNorm(const Matrix &Y) const;

 private:
  const size_t n;
  const unsigned int d{3};
  const unsigned int r{3};
  std::vector<GeodesicMeasurementTerm> measurements;
  std::vector<GeodesicConsensusTerm> consensusTerms;
  LiftedSEManifold *M;

  Matrix readElement(const ROPTLIB::Element *element) const;
};

}  // namespace DPGO

#endif

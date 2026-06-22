/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DCCI_LINEAR_OPERATORS_H
#define DCCI_LINEAR_OPERATORS_H

#include <DPGO/DPGO_types.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <cstddef>
#include <vector>

namespace DPGO {

class ChordalRotationLinearOperator {
 public:
  ChordalRotationLinearOperator(
      size_t dimension, size_t num_poses,
      const std::vector<RelativeSEMeasurement> &measurements,
      bool use_measurement_weight = false);

  Eigen::VectorXd buildRHS() const;
  Eigen::VectorXd applyH(const Eigen::VectorXd &x) const;
  Eigen::VectorXd buildJacobiPreconditionerDiagonal() const;

  size_t dimension() const { return dimension_; }
  size_t numPoses() const { return numPoses_; }
  size_t reducedDimension() const;

 private:
  double effectiveKappa(const RelativeSEMeasurement &measurement) const;
  void validateMeasurement(const RelativeSEMeasurement &measurement) const;
  Eigen::Map<const Matrix> inputBlock(const Eigen::VectorXd &x,
                                      size_t pose) const;
  Eigen::Map<Matrix> outputBlock(Eigen::VectorXd &x, size_t pose) const;

  size_t dimension_;
  size_t numPoses_;
  std::vector<RelativeSEMeasurement> measurements_;
  bool useMeasurementWeight_;
};

class ChordalTranslationLinearOperator {
 public:
  ChordalTranslationLinearOperator(
      size_t dimension, size_t num_poses,
      const std::vector<RelativeSEMeasurement> &measurements,
      const Matrix &projected_rotations,
      bool use_measurement_weight = false);

  Eigen::VectorXd buildRHS() const;
  Eigen::VectorXd applyH(const Eigen::VectorXd &x) const;
  Eigen::VectorXd buildJacobiPreconditionerDiagonal() const;

  size_t dimension() const { return dimension_; }
  size_t numPoses() const { return numPoses_; }
  size_t reducedDimension() const;

 private:
  double effectiveTau(const RelativeSEMeasurement &measurement) const;
  void validateMeasurement(const RelativeSEMeasurement &measurement) const;
  Eigen::Map<const Vector> inputBlock(const Eigen::VectorXd &x,
                                      size_t pose) const;
  Eigen::Map<Vector> outputBlock(Eigen::VectorXd &x, size_t pose) const;
  Matrix rotationBlock(size_t pose) const;

  size_t dimension_;
  size_t numPoses_;
  std::vector<RelativeSEMeasurement> measurements_;
  Matrix projectedRotations_;
  bool useMeasurementWeight_;
};

}  // namespace DPGO

#endif

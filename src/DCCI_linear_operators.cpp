/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DCCI_linear_operators.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace DPGO {

namespace {

constexpr size_t kAnchorPose = 0;

}  // namespace

ChordalRotationLinearOperator::ChordalRotationLinearOperator(
    size_t dimension, size_t num_poses,
    const std::vector<RelativeSEMeasurement> &measurements,
    bool use_measurement_weight)
    : dimension_(dimension),
      numPoses_(num_poses),
      measurements_(measurements),
      useMeasurementWeight_(use_measurement_weight) {
  if (dimension_ == 0) {
    throw std::invalid_argument(
        "ChordalRotationLinearOperator requires positive dimension");
  }
  if (numPoses_ == 0) {
    throw std::invalid_argument(
        "ChordalRotationLinearOperator requires at least one pose");
  }
  for (const auto &measurement : measurements_) {
    validateMeasurement(measurement);
  }
}

size_t ChordalRotationLinearOperator::reducedDimension() const {
  return (numPoses_ - 1) * dimension_ * dimension_;
}

Eigen::VectorXd ChordalRotationLinearOperator::buildRHS() const {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(reducedDimension());

  for (const auto &measurement : measurements_) {
    const double kappa = effectiveKappa(measurement);
    if (measurement.p1 == kAnchorPose && measurement.p2 != kAnchorPose) {
      outputBlock(q, measurement.p2) += kappa * measurement.R;
    } else if (measurement.p2 == kAnchorPose && measurement.p1 != kAnchorPose) {
      outputBlock(q, measurement.p1) += kappa * measurement.R.transpose();
    }
  }

  return q;
}

Eigen::VectorXd ChordalRotationLinearOperator::applyH(
    const Eigen::VectorXd &x) const {
  if (static_cast<size_t>(x.size()) != reducedDimension()) {
    throw std::invalid_argument("ChordalRotationLinearOperator input size " +
                                std::to_string(x.size()) +
                                " does not match reduced dimension " +
                                std::to_string(reducedDimension()));
  }

  Eigen::VectorXd y = Eigen::VectorXd::Zero(reducedDimension());
  const Matrix zeroBlock = Matrix::Zero(dimension_, dimension_);

  for (const auto &measurement : measurements_) {
    const Matrix Pi = measurement.p1 == kAnchorPose
                          ? zeroBlock
                          : Matrix(inputBlock(x, measurement.p1));
    const Matrix Pj = measurement.p2 == kAnchorPose
                          ? zeroBlock
                          : Matrix(inputBlock(x, measurement.p2));
    const Matrix E = Pi * measurement.R - Pj;
    const double kappa = effectiveKappa(measurement);

    if (measurement.p1 != kAnchorPose) {
      outputBlock(y, measurement.p1) +=
          kappa * E * measurement.R.transpose();
    }
    if (measurement.p2 != kAnchorPose) {
      outputBlock(y, measurement.p2) += -kappa * E;
    }
  }

  return y;
}

Eigen::VectorXd
ChordalRotationLinearOperator::buildJacobiPreconditionerDiagonal() const {
  Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(reducedDimension());

  for (const auto &measurement : measurements_) {
    const double kappa = effectiveKappa(measurement);

    if (measurement.p1 != kAnchorPose) {
      Eigen::Map<Matrix> block = outputBlock(diagonal, measurement.p1);
      for (size_t c = 0; c < dimension_; ++c) {
        const double rowNormSquared = measurement.R.row(c).squaredNorm();
        for (size_t r = 0; r < dimension_; ++r) {
          block(r, c) += kappa * rowNormSquared;
        }
      }
    }

    if (measurement.p2 != kAnchorPose) {
      outputBlock(diagonal, measurement.p2).array() += kappa;
    }
  }

  return diagonal;
}

double ChordalRotationLinearOperator::effectiveKappa(
    const RelativeSEMeasurement &measurement) const {
  return useMeasurementWeight_ ? measurement.kappa * measurement.weight
                               : measurement.kappa;
}

void ChordalRotationLinearOperator::validateMeasurement(
    const RelativeSEMeasurement &measurement) const {
  if (measurement.p1 >= numPoses_ || measurement.p2 >= numPoses_) {
    throw std::invalid_argument(
        "ChordalRotationLinearOperator measurement pose index out of range");
  }
  if (static_cast<size_t>(measurement.R.rows()) != dimension_ ||
      static_cast<size_t>(measurement.R.cols()) != dimension_) {
    throw std::invalid_argument(
        "ChordalRotationLinearOperator measurement rotation has wrong size");
  }
  if (static_cast<size_t>(measurement.t.rows()) != dimension_ ||
      measurement.t.cols() != 1) {
    throw std::invalid_argument(
        "ChordalRotationLinearOperator measurement translation has wrong size");
  }
  if (!std::isfinite(measurement.kappa)) {
    throw std::invalid_argument(
        "ChordalRotationLinearOperator measurement kappa must be finite");
  }
}

Eigen::Map<const Matrix> ChordalRotationLinearOperator::inputBlock(
    const Eigen::VectorXd &x, size_t pose) const {
  if (pose == kAnchorPose || pose >= numPoses_) {
    throw std::invalid_argument(
        "ChordalRotationLinearOperator requested invalid input block");
  }
  const size_t d2 = dimension_ * dimension_;
  const size_t offset = (pose - 1) * d2;
  return Eigen::Map<const Matrix>(x.data() + offset, dimension_, dimension_);
}

Eigen::Map<Matrix> ChordalRotationLinearOperator::outputBlock(
    Eigen::VectorXd &x, size_t pose) const {
  if (pose == kAnchorPose || pose >= numPoses_) {
    throw std::invalid_argument(
        "ChordalRotationLinearOperator requested invalid output block");
  }
  const size_t d2 = dimension_ * dimension_;
  const size_t offset = (pose - 1) * d2;
  return Eigen::Map<Matrix>(x.data() + offset, dimension_, dimension_);
}

ChordalTranslationLinearOperator::ChordalTranslationLinearOperator(
    size_t dimension, size_t num_poses,
    const std::vector<RelativeSEMeasurement> &measurements,
    const Matrix &projected_rotations, bool use_measurement_weight)
    : dimension_(dimension),
      numPoses_(num_poses),
      measurements_(measurements),
      projectedRotations_(projected_rotations),
      useMeasurementWeight_(use_measurement_weight) {
  if (dimension_ == 0) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator requires positive dimension");
  }
  if (numPoses_ == 0) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator requires at least one pose");
  }
  if (static_cast<size_t>(projectedRotations_.rows()) != dimension_ ||
      static_cast<size_t>(projectedRotations_.cols()) != dimension_ * numPoses_) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator projected rotations have wrong size");
  }
  for (const auto &measurement : measurements_) {
    validateMeasurement(measurement);
  }
}

size_t ChordalTranslationLinearOperator::reducedDimension() const {
  return (numPoses_ - 1) * dimension_;
}

Eigen::VectorXd ChordalTranslationLinearOperator::buildRHS() const {
  Eigen::VectorXd q = Eigen::VectorXd::Zero(reducedDimension());

  for (const auto &measurement : measurements_) {
    const double tau = effectiveTau(measurement);
    const Vector a = rotationBlock(measurement.p1) * measurement.t;

    if (measurement.p1 != kAnchorPose) {
      outputBlock(q, measurement.p1) += -tau * a;
    }
    if (measurement.p2 != kAnchorPose) {
      outputBlock(q, measurement.p2) += tau * a;
    }
  }

  return q;
}

Eigen::VectorXd ChordalTranslationLinearOperator::applyH(
    const Eigen::VectorXd &x) const {
  if (static_cast<size_t>(x.size()) != reducedDimension()) {
    throw std::invalid_argument("ChordalTranslationLinearOperator input size " +
                                std::to_string(x.size()) +
                                " does not match reduced dimension " +
                                std::to_string(reducedDimension()));
  }

  Eigen::VectorXd y = Eigen::VectorXd::Zero(reducedDimension());
  const Vector zeroBlock = Vector::Zero(dimension_);

  for (const auto &measurement : measurements_) {
    const Vector pi = measurement.p1 == kAnchorPose
                          ? zeroBlock
                          : Vector(inputBlock(x, measurement.p1));
    const Vector pj = measurement.p2 == kAnchorPose
                          ? zeroBlock
                          : Vector(inputBlock(x, measurement.p2));
    const Vector u = pj - pi;
    const double tau = effectiveTau(measurement);

    if (measurement.p1 != kAnchorPose) {
      outputBlock(y, measurement.p1) += -tau * u;
    }
    if (measurement.p2 != kAnchorPose) {
      outputBlock(y, measurement.p2) += tau * u;
    }
  }

  return y;
}

Eigen::VectorXd
ChordalTranslationLinearOperator::buildJacobiPreconditionerDiagonal() const {
  Eigen::VectorXd diagonal = Eigen::VectorXd::Zero(reducedDimension());

  for (const auto &measurement : measurements_) {
    const double tau = effectiveTau(measurement);
    if (measurement.p1 != kAnchorPose) {
      outputBlock(diagonal, measurement.p1).array() += tau;
    }
    if (measurement.p2 != kAnchorPose) {
      outputBlock(diagonal, measurement.p2).array() += tau;
    }
  }

  return diagonal;
}

double ChordalTranslationLinearOperator::effectiveTau(
    const RelativeSEMeasurement &measurement) const {
  return useMeasurementWeight_ ? measurement.tau * measurement.weight
                               : measurement.tau;
}

void ChordalTranslationLinearOperator::validateMeasurement(
    const RelativeSEMeasurement &measurement) const {
  if (measurement.p1 >= numPoses_ || measurement.p2 >= numPoses_) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator measurement pose index out of range");
  }
  if (static_cast<size_t>(measurement.R.rows()) != dimension_ ||
      static_cast<size_t>(measurement.R.cols()) != dimension_) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator measurement rotation has wrong size");
  }
  if (static_cast<size_t>(measurement.t.rows()) != dimension_ ||
      measurement.t.cols() != 1) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator measurement translation has wrong size");
  }
  if (!std::isfinite(measurement.tau)) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator measurement tau must be finite");
  }
}

Eigen::Map<const Vector> ChordalTranslationLinearOperator::inputBlock(
    const Eigen::VectorXd &x, size_t pose) const {
  if (pose == kAnchorPose || pose >= numPoses_) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator requested invalid input block");
  }
  const size_t offset = (pose - 1) * dimension_;
  return Eigen::Map<const Vector>(x.data() + offset, dimension_);
}

Eigen::Map<Vector> ChordalTranslationLinearOperator::outputBlock(
    Eigen::VectorXd &x, size_t pose) const {
  if (pose == kAnchorPose || pose >= numPoses_) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator requested invalid output block");
  }
  const size_t offset = (pose - 1) * dimension_;
  return Eigen::Map<Vector>(x.data() + offset, dimension_);
}

Matrix ChordalTranslationLinearOperator::rotationBlock(size_t pose) const {
  if (pose >= numPoses_) {
    throw std::invalid_argument(
        "ChordalTranslationLinearOperator requested invalid rotation block");
  }
  return projectedRotations_.block(0, pose * dimension_, dimension_,
                                  dimension_);
}

}  // namespace DPGO

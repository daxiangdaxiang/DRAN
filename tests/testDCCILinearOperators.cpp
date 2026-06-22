#include <DPGO/DCCI_linear_operators.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

struct RotationOperatorError {
  double applyHRelError{0.0};
  double rhsRelError{0.0};
  double jacobiRelError{0.0};
};

struct TranslationOperatorError {
  double applyHRelError{0.0};
  double rhsRelError{0.0};
  double jacobiRelError{0.0};
};

Matrix rotation2(double theta) {
  Matrix R(2, 2);
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  R << c, -s, s, c;
  return R;
}

Matrix rotation3(double roll, double pitch, double yaw) {
  const Eigen::AngleAxisd Rx(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd Ry(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd Rz(yaw, Eigen::Vector3d::UnitZ());
  return (Rz * Ry * Rx).toRotationMatrix();
}

RelativeSEMeasurement makeMeasurement(size_t i, size_t j, const Matrix &Ri,
                                      const Vector &ti, const Matrix &Rj,
                                      const Vector &tj, double kappa,
                                      double tau) {
  return RelativeSEMeasurement(0, 0, i, j, Ri.transpose() * Rj,
                               Ri.transpose() * (tj - ti), kappa, tau);
}

std::vector<RelativeSEMeasurement> makeSynthetic2DGraph(size_t *numPoses) {
  *numPoses = 4;
  const std::vector<Matrix> rotations = {rotation2(0.0), rotation2(0.25),
                                         rotation2(0.55), rotation2(0.80)};
  std::vector<Vector> translations(*numPoses, Vector::Zero(2));
  translations[0] << 0.0, 0.0;
  translations[1] << 1.0, 0.2;
  translations[2] << 2.1, 0.7;
  translations[3] << 3.0, 1.1;

  std::vector<RelativeSEMeasurement> measurements;
  measurements.emplace_back(makeMeasurement(0, 1, rotations[0],
                                            translations[0], rotations[1],
                                            translations[1], 3.0, 2.0));
  measurements.emplace_back(makeMeasurement(1, 2, rotations[1],
                                            translations[1], rotations[2],
                                            translations[2], 2.0, 4.0));
  measurements.emplace_back(makeMeasurement(2, 3, rotations[2],
                                            translations[2], rotations[3],
                                            translations[3], 5.0, 3.0));
  measurements.emplace_back(makeMeasurement(0, 2, rotations[0],
                                            translations[0], rotations[2],
                                            translations[2], 4.0, 2.5));
  measurements.emplace_back(makeMeasurement(1, 3, rotations[1],
                                            translations[1], rotations[3],
                                            translations[3], 2.5, 3.5));
  measurements.emplace_back(makeMeasurement(3, 0, rotations[3],
                                            translations[3], rotations[0],
                                            translations[0], 1.7, 2.2));
  return measurements;
}

std::vector<RelativeSEMeasurement> makeSynthetic3DGraph(size_t *numPoses) {
  *numPoses = 4;
  const std::vector<Matrix> rotations = {
      rotation3(0.0, 0.0, 0.0), rotation3(0.10, -0.05, 0.20),
      rotation3(0.18, 0.08, 0.45), rotation3(0.22, -0.12, 0.70)};
  std::vector<Vector> translations(*numPoses, Vector::Zero(3));
  translations[0] << 0.0, 0.0, 0.0;
  translations[1] << 1.0, 0.2, 0.1;
  translations[2] << 1.8, 0.9, 0.4;
  translations[3] << 2.7, 1.4, 0.8;

  std::vector<RelativeSEMeasurement> measurements;
  measurements.emplace_back(makeMeasurement(0, 1, rotations[0],
                                            translations[0], rotations[1],
                                            translations[1], 3.0, 2.0));
  measurements.emplace_back(makeMeasurement(1, 2, rotations[1],
                                            translations[1], rotations[2],
                                            translations[2], 2.0, 4.0));
  measurements.emplace_back(makeMeasurement(2, 3, rotations[2],
                                            translations[2], rotations[3],
                                            translations[3], 5.0, 3.0));
  measurements.emplace_back(makeMeasurement(0, 2, rotations[0],
                                            translations[0], rotations[2],
                                            translations[2], 4.0, 2.5));
  measurements.emplace_back(makeMeasurement(1, 3, rotations[1],
                                            translations[1], rotations[3],
                                            translations[3], 2.5, 3.5));
  measurements.emplace_back(makeMeasurement(3, 0, rotations[3],
                                            translations[3], rotations[0],
                                            translations[0], 1.7, 2.2));
  return measurements;
}

Vector deterministicVector(size_t size) {
  Vector x(size);
  for (size_t i = 0; i < size; ++i) {
    x(i) = std::sin(0.31 * static_cast<double>(i + 1)) +
           0.25 * std::cos(0.17 * static_cast<double>(i + 1));
  }
  return x;
}

Matrix extractRotations(const Matrix &T, size_t d, size_t numPoses) {
  Matrix rotations(d, d * numPoses);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    rotations.block(0, pose * d, d, d) =
        T.block(0, pose * (d + 1), d, d);
  }
  return rotations;
}

double relativeError(const Vector &actual, const Vector &expected) {
  return (actual - expected).norm() / std::max(1.0, expected.norm());
}

TranslationOperatorError compareTranslationOperator(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements,
    bool useMeasurementWeight = false) {
  SparseMatrix B1, B2, B3;
  constructBMatrices(measurements, B1, B2, B3, useMeasurementWeight);
  SparseMatrix B1red = B1.rightCols((numPoses - 1) * d);
  B1red.makeCompressed();

  const Matrix T = chordalInitialization(d, numPoses, measurements);
  Matrix projectedRotations = extractRotations(T, d, numPoses);
  Eigen::Map<Vector> rvec(projectedRotations.data(), d * d * numPoses);
  const Vector x = deterministicVector((numPoses - 1) * d);

  const ChordalTranslationLinearOperator op(d, numPoses, measurements,
                                            projectedRotations,
                                            useMeasurementWeight);
  const Vector expectedHx = B1red.transpose() * (B1red * x);
  const Vector expectedRhs = -B1red.transpose() * (B2 * rvec);
  const SparseMatrix H = B1red.transpose() * B1red;
  const Vector expectedJacobi = H.diagonal();

  TranslationOperatorError error;
  error.applyHRelError = relativeError(op.applyH(x), expectedHx);
  error.rhsRelError = relativeError(op.buildRHS(), expectedRhs);
  error.jacobiRelError =
      relativeError(op.buildJacobiPreconditionerDiagonal(), expectedJacobi);
  return error;
}

RotationOperatorError compareRotationOperator(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements,
    bool useMeasurementWeight = false) {
  SparseMatrix B1, B2, B3;
  constructBMatrices(measurements, B1, B2, B3, useMeasurementWeight);
  const size_t d2 = d * d;
  SparseMatrix B3red = B3.rightCols((numPoses - 1) * d2);
  B3red.makeCompressed();

  Matrix I = Matrix::Identity(d, d);
  Eigen::Map<Vector> IdVec(I.data(), d2);
  const Vector cR = B3.leftCols(d2) * IdVec;
  const Vector x = deterministicVector((numPoses - 1) * d2);

  const ChordalRotationLinearOperator op(d, numPoses, measurements,
                                         useMeasurementWeight);
  const Vector expectedHx = B3red.transpose() * (B3red * x);
  const Vector expectedRhs = -B3red.transpose() * cR;
  const SparseMatrix H = B3red.transpose() * B3red;
  const Vector expectedJacobi = H.diagonal();

  RotationOperatorError error;
  error.applyHRelError = relativeError(op.applyH(x), expectedHx);
  error.rhsRelError = relativeError(op.buildRHS(), expectedRhs);
  error.jacobiRelError =
      relativeError(op.buildJacobiPreconditionerDiagonal(), expectedJacobi);
  return error;
}

void expectTranslationOperatorMatchesExplicitB1B2(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements,
    bool useMeasurementWeight = false) {
  const TranslationOperatorError error =
      compareTranslationOperator(d, numPoses, measurements,
                                 useMeasurementWeight);
  std::cout << "DCCI translation operator d=" << d
            << " applyH_rel_error=" << error.applyHRelError
            << " rhs_rel_error=" << error.rhsRelError
            << " jacobi_rel_error=" << error.jacobiRelError << std::endl;
  EXPECT_LT(error.applyHRelError, 1e-10);
  EXPECT_LT(error.rhsRelError, 1e-10);
  EXPECT_LT(error.jacobiRelError, 1e-10);
}

void expectRotationOperatorMatchesExplicitB3(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements,
    bool useMeasurementWeight = false) {
  const RotationOperatorError error =
      compareRotationOperator(d, numPoses, measurements, useMeasurementWeight);
  std::cout << "DCCI rotation operator d=" << d
            << " applyH_rel_error=" << error.applyHRelError
            << " rhs_rel_error=" << error.rhsRelError
            << " jacobi_rel_error=" << error.jacobiRelError << std::endl;
  EXPECT_LT(error.applyHRelError, 1e-10);
  EXPECT_LT(error.rhsRelError, 1e-10);
  EXPECT_LT(error.jacobiRelError, 1e-10);
}

}  // namespace

TEST(testDPGO, DCCIRotationOperatorMatchesExplicitB3In2D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic2DGraph(&numPoses);
  expectRotationOperatorMatchesExplicitB3(2, numPoses, measurements);
}

TEST(testDPGO, DCCIRotationOperatorMatchesExplicitB3In3D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic3DGraph(&numPoses);
  expectRotationOperatorMatchesExplicitB3(3, numPoses, measurements);
}

TEST(testDPGO, DCCIRotationOperatorIgnoresMeasurementWeightByDefault) {
  size_t numPoses = 0;
  auto measurements = makeSynthetic3DGraph(&numPoses);
  measurements[1].weight = 0.05;
  measurements[3].weight = 0.25;
  expectRotationOperatorMatchesExplicitB3(3, numPoses, measurements);
}

TEST(testDPGO, DCCIRotationOperatorMatchesExplicitWeightedB3) {
  size_t numPoses = 0;
  auto measurements = makeSynthetic3DGraph(&numPoses);
  measurements[1].weight = 0.05;
  measurements[3].weight = 0.25;
  measurements[5].weight = 0.60;
  expectRotationOperatorMatchesExplicitB3(3, numPoses, measurements, true);
}

TEST(testDPGO, DCCITranslationOperatorMatchesExplicitB1B2In2D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic2DGraph(&numPoses);
  expectTranslationOperatorMatchesExplicitB1B2(2, numPoses, measurements);
}

TEST(testDPGO, DCCITranslationOperatorMatchesExplicitB1B2In3D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic3DGraph(&numPoses);
  expectTranslationOperatorMatchesExplicitB1B2(3, numPoses, measurements);
}

TEST(testDPGO, DCCITranslationOperatorIgnoresMeasurementWeightByDefault) {
  size_t numPoses = 0;
  auto measurements = makeSynthetic3DGraph(&numPoses);
  measurements[0].weight = 0.15;
  measurements[4].weight = 0.35;
  expectTranslationOperatorMatchesExplicitB1B2(3, numPoses, measurements);
}

TEST(testDPGO, DCCITranslationOperatorMatchesExplicitWeightedB1B2) {
  size_t numPoses = 0;
  auto measurements = makeSynthetic3DGraph(&numPoses);
  measurements[0].weight = 0.15;
  measurements[2].weight = 0.45;
  measurements[4].weight = 0.35;
  expectTranslationOperatorMatchesExplicitB1B2(3, numPoses, measurements,
                                               true);
}

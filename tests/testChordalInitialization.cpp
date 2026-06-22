#include <DPGO/DPGO_utils.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Geometry>

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

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

Matrix poseRotation(const Matrix &T, size_t d, size_t pose) {
  return T.block(0, pose * (d + 1), d, d);
}

Vector poseTranslation(const Matrix &T, size_t d, size_t pose) {
  return T.block(0, pose * (d + 1) + d, d, 1);
}

double measurementResidual(const Matrix &T, size_t d,
                           const std::vector<RelativeSEMeasurement> &edges) {
  double residual = 0.0;
  for (const auto &edge : edges) {
    const Matrix Ri = poseRotation(T, d, edge.p1);
    const Matrix Rj = poseRotation(T, d, edge.p2);
    const Vector ti = poseTranslation(T, d, edge.p1);
    const Vector tj = poseTranslation(T, d, edge.p2);
    residual += edge.kappa * (Ri * edge.R - Rj).squaredNorm();
    residual += edge.tau * (tj - ti - Ri * edge.t).squaredNorm();
  }
  return residual;
}

void expectValidChordalEstimate(
    size_t d, size_t numPoses, const Matrix &T,
    const std::vector<RelativeSEMeasurement> &measurements) {
  ASSERT_EQ(T.rows(), static_cast<int>(d));
  ASSERT_EQ(T.cols(), static_cast<int>(numPoses * (d + 1)));
  ASSERT_TRUE(T.allFinite());

  const Matrix I = Matrix::Identity(d, d);
  EXPECT_LE((poseRotation(T, d, 0) - I).norm(), 1e-10);
  EXPECT_LE(poseTranslation(T, d, 0).norm(), 1e-10);

  for (size_t pose = 0; pose < numPoses; ++pose) {
    const Matrix R = poseRotation(T, d, pose);
    EXPECT_LE((R.transpose() * R - I).norm(), 1e-8);
    EXPECT_NEAR(R.determinant(), 1.0, 1e-8);
    EXPECT_TRUE(poseTranslation(T, d, pose).allFinite());
  }

  const double residual = measurementResidual(T, d, measurements);
  EXPECT_TRUE(std::isfinite(residual));
  EXPECT_LE(residual, 1e-8);
}

}  // namespace

TEST(testDPGO, ChordalInitialization2DChainWithLoopClosure) {
  const size_t d = 2;
  const size_t numPoses = 4;
  const std::vector<Matrix> rotations = {rotation2(0.0), rotation2(0.25),
                                         rotation2(0.55), rotation2(0.80)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
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

  const Matrix T = chordalInitialization(d, numPoses, measurements);
  expectValidChordalEstimate(d, numPoses, T, measurements);
}

TEST(testDPGO, ChordalInitialization3DChainWithLoopClosure) {
  const size_t d = 3;
  const size_t numPoses = 4;
  const std::vector<Matrix> rotations = {
      rotation3(0.0, 0.0, 0.0), rotation3(0.10, -0.05, 0.20),
      rotation3(0.18, 0.08, 0.45), rotation3(0.22, -0.12, 0.70)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
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

  const Matrix T = chordalInitialization(d, numPoses, measurements);
  expectValidChordalEstimate(d, numPoses, T, measurements);
}

TEST(testDPGO, ChordalInitializationIgnoresMeasurementWeightByDefault) {
  const size_t d = 3;
  const size_t numPoses = 4;
  const std::vector<Matrix> rotations = {
      rotation3(0.0, 0.0, 0.0), rotation3(0.10, -0.05, 0.20),
      rotation3(0.18, 0.08, 0.45), rotation3(0.22, -0.12, 0.70)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  translations[0] << 0.0, 0.0, 0.0;
  translations[1] << 1.0, 0.2, 0.1;
  translations[2] << 1.8, 0.9, 0.4;
  translations[3] << 2.7, 1.4, 0.8;

  std::vector<RelativeSEMeasurement> weighted;
  weighted.emplace_back(makeMeasurement(0, 1, rotations[0], translations[0],
                                        rotations[1], translations[1], 3.0,
                                        2.0));
  weighted.emplace_back(makeMeasurement(1, 2, rotations[1], translations[1],
                                        rotations[2], translations[2], 2.0,
                                        4.0));
  weighted.emplace_back(makeMeasurement(2, 3, rotations[2], translations[2],
                                        rotations[3], translations[3], 5.0,
                                        3.0));
  weighted.emplace_back(makeMeasurement(0, 2, rotations[0], translations[0],
                                        rotations[2], translations[2], 4.0,
                                        2.5));
  weighted.emplace_back(makeMeasurement(1, 3, rotations[1], translations[1],
                                        rotations[3], translations[3], 2.5,
                                        3.5));
  weighted[1].weight = 0.05;
  weighted[3].weight = 0.30;

  auto unitWeight = weighted;
  for (auto &measurement : unitWeight) {
    measurement.weight = 1.0;
  }

  const Matrix defaultWeighted = chordalInitialization(d, numPoses, weighted);
  const Matrix defaultUnit = chordalInitialization(d, numPoses, unitWeight);
  EXPECT_LE((defaultWeighted - defaultUnit).norm(), 1e-12);
}

TEST(testDPGO, ChordalInitializationUsesMeasurementWeightWhenEnabled) {
  const size_t d = 3;
  const size_t numPoses = 4;
  const std::vector<Matrix> rotations = {
      rotation3(0.0, 0.0, 0.0), rotation3(0.10, -0.05, 0.20),
      rotation3(0.18, 0.08, 0.45), rotation3(0.22, -0.12, 0.70)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  translations[0] << 0.0, 0.0, 0.0;
  translations[1] << 1.0, 0.2, 0.1;
  translations[2] << 1.8, 0.9, 0.4;
  translations[3] << 2.7, 1.4, 0.8;

  std::vector<RelativeSEMeasurement> weighted;
  weighted.emplace_back(makeMeasurement(0, 1, rotations[0], translations[0],
                                        rotations[1], translations[1], 3.0,
                                        2.0));
  weighted.emplace_back(makeMeasurement(1, 2, rotations[1], translations[1],
                                        rotations[2], translations[2], 2.0,
                                        4.0));
  weighted.emplace_back(makeMeasurement(2, 3, rotations[2], translations[2],
                                        rotations[3], translations[3], 5.0,
                                        3.0));
  weighted.emplace_back(makeMeasurement(0, 2, rotations[0], translations[0],
                                        rotations[2], translations[2], 4.0,
                                        2.5));
  weighted.emplace_back(makeMeasurement(1, 3, rotations[1], translations[1],
                                        rotations[3], translations[3], 2.5,
                                        3.5));
  weighted[1].weight = 0.05;
  weighted[3].weight = 0.30;
  weighted[4].weight = 0.65;

  auto scaled = weighted;
  for (auto &measurement : scaled) {
    measurement.kappa *= measurement.weight;
    measurement.tau *= measurement.weight;
    measurement.weight = 1.0;
  }

  const Matrix explicitWeighted =
      chordalInitialization(d, numPoses, weighted, true);
  const Matrix scaledDefault = chordalInitialization(d, numPoses, scaled);
  EXPECT_LE((explicitWeighted - scaledDefault).norm(), 1e-10);
}

TEST(testDPGO, ChordalInitializationIsDeterministicForSameInput) {
  const size_t d = 3;
  const size_t numPoses = 4;
  const std::vector<Matrix> rotations = {
      rotation3(0.0, 0.0, 0.0), rotation3(0.10, -0.05, 0.20),
      rotation3(0.18, 0.08, 0.45), rotation3(0.22, -0.12, 0.70)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
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
  measurements[1].weight = 0.05;
  measurements[3].weight = 0.30;

  const Matrix first = chordalInitialization(d, numPoses, measurements, true);
  const Matrix second = chordalInitialization(d, numPoses, measurements, true);
  EXPECT_LE((first - second).norm(), 1e-12);
}

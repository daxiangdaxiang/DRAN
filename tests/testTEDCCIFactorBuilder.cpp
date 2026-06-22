#include <DPGO/TEDCCI.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Geometry>

#include <cmath>
#include <map>
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

std::vector<RelativeSEMeasurement> makeGraph(size_t d, size_t *numPoses) {
  *numPoses = 4;
  std::vector<Matrix> rotations(*numPoses);
  std::vector<Vector> translations(*numPoses, Vector::Zero(d));
  if (d == 2) {
    rotations = {rotation2(0.0), rotation2(0.25), rotation2(0.55),
                 rotation2(0.80)};
    translations[0] << 0.0, 0.0;
    translations[1] << 1.0, 0.2;
    translations[2] << 2.1, 0.7;
    translations[3] << 3.0, 1.1;
  } else {
    rotations = {rotation3(0.0, 0.0, 0.0), rotation3(0.10, -0.05, 0.20),
                 rotation3(0.18, 0.08, 0.45),
                 rotation3(0.22, -0.12, 0.70)};
    translations[0] << 0.0, 0.0, 0.0;
    translations[1] << 1.0, 0.2, 0.1;
    translations[2] << 1.8, 0.9, 0.4;
    translations[3] << 2.7, 1.4, 0.8;
  }

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
  measurements.emplace_back(makeMeasurement(3, 0, rotations[3],
                                            translations[3], rotations[0],
                                            translations[0], 1.7, 2.2));
  return measurements;
}

Matrix deterministicBlock(size_t d, size_t pose) {
  Matrix X(d, d);
  for (size_t col = 0; col < d; ++col) {
    for (size_t row = 0; row < d; ++row) {
      X(row, col) =
          std::sin(0.21 * static_cast<double>((pose + 1) * (row + 1))) +
          0.3 * std::cos(0.17 * static_cast<double>((col + 2) * (pose + 1)));
    }
  }
  return X;
}

Vector deterministicVector(size_t d, size_t pose) {
  Vector x(d);
  for (size_t row = 0; row < d; ++row) {
    x(row) = std::sin(0.37 * static_cast<double>(pose + row + 1)) +
             0.2 * std::cos(0.19 * static_cast<double>((pose + 1) * (row + 2)));
  }
  return x;
}

Vector stackRotationVariables(const LinearFactorBlock &factor, size_t d,
                              const std::map<PoseKey, Matrix> &blocks) {
  Vector x(factor.keys.size() * d * d);
  for (size_t k = 0; k < factor.keys.size(); ++k) {
    const Matrix block = blocks.at(factor.keys[k]);
    x.segment(k * d * d, d * d) =
        Eigen::Map<const Vector>(block.data(), d * d);
  }
  return x;
}

Vector stackTranslationVariables(const LinearFactorBlock &factor, size_t d,
                                 const std::map<PoseKey, Vector> &blocks) {
  Vector x(factor.keys.size() * d);
  for (size_t k = 0; k < factor.keys.size(); ++k) {
    x.segment(k * d, d) = blocks.at(factor.keys[k]);
  }
  return x;
}

double effectiveWeight(double precision, double weight, bool useWeight) {
  return std::sqrt(precision * (useWeight ? weight : 1.0));
}

void expectRotationFactorsMatchDirectResiduals(size_t d, bool useWeight) {
  size_t numPoses = 0;
  auto measurements = makeGraph(d, &numPoses);
  measurements[1].weight = 0.05;
  measurements[3].weight = 0.30;
  measurements[4].weight = 0.65;

  TEDCCIParams params;
  params.use_measurement_weight = useWeight;
  const auto factors =
      CCIFactorBuilder::BuildRotationFactors(d, measurements, params);

  ASSERT_EQ(factors.size(), measurements.size());
  std::map<PoseKey, Matrix> blocks;
  blocks[PoseKey{0, 0}] = Matrix::Identity(d, d);
  for (size_t pose = 1; pose < numPoses; ++pose) {
    blocks[PoseKey{0, static_cast<int>(pose)}] = deterministicBlock(d, pose);
  }

  for (size_t e = 0; e < measurements.size(); ++e) {
    const auto &m = measurements[e];
    const Vector x = stackRotationVariables(factors[e], d, blocks);
    const Vector actual = factors[e].A * x - factors[e].b;
    const Matrix residual =
        effectiveWeight(m.kappa, m.weight, useWeight) *
        (blocks[PoseKey{0, static_cast<int>(m.p1)}] * m.R -
         blocks[PoseKey{0, static_cast<int>(m.p2)}]);
    const Vector expected =
        Eigen::Map<const Vector>(residual.data(), static_cast<int>(d * d));
    EXPECT_LT((actual - expected).norm(), 1e-10);
  }
}

void expectTranslationFactorsMatchDirectResiduals(size_t d, bool useWeight) {
  size_t numPoses = 0;
  auto measurements = makeGraph(d, &numPoses);
  measurements[1].weight = 0.05;
  measurements[3].weight = 0.30;
  measurements[4].weight = 0.65;

  std::map<PoseKey, Matrix> rotations;
  std::map<PoseKey, Vector> translations;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    rotations[PoseKey{0, static_cast<int>(pose)}] =
        pose == 0 ? Matrix::Identity(d, d) : deterministicBlock(d, pose);
    translations[PoseKey{0, static_cast<int>(pose)}] =
        pose == 0 ? Vector::Zero(d) : deterministicVector(d, pose);
  }

  TEDCCIParams params;
  params.use_measurement_weight = useWeight;
  const auto factors = CCIFactorBuilder::BuildTranslationFactors(
      d, measurements, rotations, params);

  ASSERT_EQ(factors.size(), measurements.size());
  for (size_t e = 0; e < measurements.size(); ++e) {
    const auto &m = measurements[e];
    const Vector x = stackTranslationVariables(factors[e], d, translations);
    const Vector actual = factors[e].A * x - factors[e].b;
    const Vector expected =
        effectiveWeight(m.tau, m.weight, useWeight) *
        (translations[PoseKey{0, static_cast<int>(m.p2)}] -
         translations[PoseKey{0, static_cast<int>(m.p1)}] -
         rotations[PoseKey{0, static_cast<int>(m.p1)}] * m.t);
    EXPECT_LT((actual - expected).norm(), 1e-10);
  }
}

}  // namespace

TEST(testDPGO, TEDCCIRotationFactorsMatchDirectResiduals2D) {
  expectRotationFactorsMatchDirectResiduals(2, false);
}

TEST(testDPGO, TEDCCIRotationFactorsMatchDirectResiduals3DWeighted) {
  expectRotationFactorsMatchDirectResiduals(3, true);
}

TEST(testDPGO, TEDCCITranslationFactorsMatchDirectResiduals2DWeighted) {
  expectTranslationFactorsMatchDirectResiduals(2, true);
}

TEST(testDPGO, TEDCCITranslationFactorsMatchDirectResiduals3D) {
  expectTranslationFactorsMatchDirectResiduals(3, false);
}

#include <DPGO/DPGO_utils.h>
#include <DPGO/TEDCCI.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Geometry>

#include <algorithm>
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

RelativeSEMeasurement makeMeasurement(size_t i, size_t j,
                                      const std::vector<int> &owner,
                                      const Matrix &Ri, const Vector &ti,
                                      const Matrix &Rj, const Vector &tj,
                                      double kappa, double tau) {
  return RelativeSEMeasurement(owner.at(i), owner.at(j), i, j,
                               Ri.transpose() * Rj,
                               Ri.transpose() * (tj - ti), kappa, tau);
}

std::vector<int> contiguousOwner(size_t numPoses, int robots) {
  std::vector<int> owner(numPoses);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    owner[pose] = std::min<int>(
        static_cast<int>((pose * static_cast<size_t>(robots)) / numPoses),
        robots - 1);
  }
  return owner;
}

std::vector<TEDCCIPartition> makePartitions(
    size_t numPoses, int robots,
    const std::vector<RelativeSEMeasurement> &measurements,
    const TEDCCIParams &params) {
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  std::vector<RelativeSEMeasurement> local;
  std::vector<RelativeSEMeasurement> shared;
  for (const auto &measurement : measurements) {
    if (measurement.r1 == measurement.r2) {
      local.push_back(measurement);
    } else {
      shared.push_back(measurement);
    }
  }

  std::vector<TEDCCIPartition> partitions;
  for (int robot = 0; robot < robots; ++robot) {
    std::vector<PoseKey> localPoses;
    for (size_t pose = 0; pose < numPoses; ++pose) {
      if (owner[pose] == robot) {
        localPoses.push_back(PoseKey{robot, static_cast<int>(pose)});
      }
    }
    std::vector<RelativeSEMeasurement> robotLocal;
    for (const auto &measurement : local) {
      if (static_cast<int>(measurement.r1) == robot) {
        robotLocal.push_back(measurement);
      }
    }
    partitions.push_back(BoundarySelector::SelectBoundaryVariables(
        robot, localPoses, robotLocal, shared, params));
  }
  return partitions;
}

void expectPoseEstimateClose(const Matrix &actual, const Matrix &expected,
                             size_t d, size_t numPoses,
                             double tolerance = 1e-8) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  double maxRot = 0.0;
  double maxTrans = 0.0;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    maxRot = std::max(
        maxRot,
        (actual.block(0, pose * (d + 1), d, d) -
         expected.block(0, pose * (d + 1), d, d))
            .norm());
    maxTrans = std::max(
        maxTrans,
        (actual.block(0, pose * (d + 1) + d, d, 1) -
         expected.block(0, pose * (d + 1) + d, d, 1))
            .norm());
  }
  EXPECT_LT(maxRot, tolerance);
  EXPECT_LT(maxTrans, tolerance);
}

TEST(testDPGO, TEDCCISingleProcessDirectMatchesCentralized2DChainSplit) {
  const size_t d = 2;
  const size_t numPoses = 5;
  const int robots = 2;
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  const std::vector<Matrix> rotations = {
      rotation2(0.0), rotation2(0.15), rotation2(0.35), rotation2(0.62),
      rotation2(0.88)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  translations[0] << 0.0, 0.0;
  translations[1] << 0.9, 0.1;
  translations[2] << 1.8, 0.4;
  translations[3] << 2.6, 0.9;
  translations[4] << 3.5, 1.1;

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < numPoses; ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0 + pose,
        3.0 + 0.5 * pose));
  }
  measurements.push_back(makeMeasurement(0, 3, owner, rotations[0],
                                         translations[0], rotations[3],
                                         translations[3], 4.0, 2.5));

  TEDCCIParams params;
  params.use_measurement_weight = false;
  const auto partitions = makePartitions(numPoses, robots, measurements, params);
  TEDCCIStats stats;
  const Matrix ted = TEDCCISolver::InitializeSingleProcessDirect(
      d, numPoses, measurements, partitions, params, &stats);
  const Matrix centralized = chordalInitialization(d, numPoses, measurements);
  expectPoseEstimateClose(ted, centralized, d, numPoses);
  EXPECT_GT(stats.num_interface_vars, 0);
}

TEST(testDPGO, TEDCCISingleProcessAsyncDDMatchesCentralized2DChainSplit) {
  const size_t d = 2;
  const size_t numPoses = 5;
  const int robots = 2;
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  const std::vector<Matrix> rotations = {
      rotation2(0.0), rotation2(0.15), rotation2(0.35), rotation2(0.62),
      rotation2(0.88)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  translations[0] << 0.0, 0.0;
  translations[1] << 0.9, 0.1;
  translations[2] << 1.8, 0.4;
  translations[3] << 2.6, 0.9;
  translations[4] << 3.5, 1.1;

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < numPoses; ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0 + pose,
        3.0 + 0.5 * pose));
  }
  measurements.push_back(makeMeasurement(0, 3, owner, rotations[0],
                                         translations[0], rotations[3],
                                         translations[3], 4.0, 2.5));

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_ASYNC_DD;
  params.use_measurement_weight = false;
  params.async_dd_max_iters = 1000;
  params.async_dd_rel_tol = 1e-12;
  const auto partitions = makePartitions(numPoses, robots, measurements,
                                         params);
  TEDCCIStats stats;
  const Matrix ted = TEDCCISolver::InitializeSingleProcessDirect(
      d, numPoses, measurements, partitions, params, &stats);
  const Matrix centralized = chordalInitialization(d, numPoses, measurements);

  expectPoseEstimateClose(ted, centralized, d, numPoses, 1e-6);
  EXPECT_TRUE(stats.async_dd_converged);
  EXPECT_GT(stats.async_dd_iterations, 0);
}

TEST(testDPGO, TEDCCISingleProcessDirectMatchesCentralized3DWeightedLoops) {
  const size_t d = 3;
  const size_t numPoses = 6;
  const int robots = 3;
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  const std::vector<Matrix> rotations = {
      rotation3(0.0, 0.0, 0.0), rotation3(0.06, -0.03, 0.14),
      rotation3(0.12, 0.02, 0.31), rotation3(0.16, -0.07, 0.48),
      rotation3(0.20, 0.04, 0.63), rotation3(0.25, -0.05, 0.82)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  translations[0] << 0.0, 0.0, 0.0;
  translations[1] << 0.9, 0.1, 0.2;
  translations[2] << 1.8, 0.5, 0.1;
  translations[3] << 2.5, 0.9, 0.4;
  translations[4] << 3.4, 1.1, 0.8;
  translations[5] << 4.2, 1.5, 1.0;

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < numPoses; ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0 + pose,
        3.0 + 0.25 * pose));
  }
  measurements.push_back(makeMeasurement(1, 4, owner, rotations[1],
                                         translations[1], rotations[4],
                                         translations[4], 4.0, 2.5));
  measurements.push_back(makeMeasurement(0, 5, owner, rotations[0],
                                         translations[0], rotations[5],
                                         translations[5], 3.5, 2.0));
  measurements[2].weight = 0.35;
  measurements[4].weight = 0.70;
  measurements[6].weight = 0.50;

  TEDCCIParams params;
  params.use_measurement_weight = true;
  const auto partitions = makePartitions(numPoses, robots, measurements, params);
  TEDCCIStats stats;
  const Matrix ted = TEDCCISolver::InitializeSingleProcessDirect(
      d, numPoses, measurements, partitions, params, &stats);
  const Matrix centralized =
      chordalInitialization(d, numPoses, measurements, true);
  expectPoseEstimateClose(ted, centralized, d, numPoses);
  EXPECT_GT(stats.num_condensed_rows, 0);
}

TEST(testDPGO, TEDCCISolverInitializeRoutesCentralizedAndHierarchicalModes) {
  const size_t d = 2;
  const size_t numPoses = 8;
  const int robots = 4;
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  std::vector<Matrix> rotations;
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  for (size_t pose = 0; pose < numPoses; ++pose) {
    rotations.push_back(rotation2(0.08 * static_cast<double>(pose)));
    translations[pose] << 0.7 * pose, 0.15 * pose;
  }

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < numPoses; ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0 + pose,
        3.0 + 0.2 * pose));
  }
  measurements.push_back(makeMeasurement(1, 4, owner, rotations[1],
                                         translations[1], rotations[4],
                                         translations[4], 4.0, 2.5));
  measurements.push_back(makeMeasurement(3, 6, owner, rotations[3],
                                         translations[3], rotations[6],
                                         translations[6], 3.5, 2.0));

  TEDCCIParams params;
  params.use_measurement_weight = false;
  const auto partitions = makePartitions(numPoses, robots, measurements,
                                         params);
  const Matrix centralized = chordalInitialization(d, numPoses, measurements);

  params.mode = CCIInitMode::CENTRALIZED_CCI;
  const Matrix viaCentralizedMode = TEDCCISolver::Initialize(
      d, numPoses, measurements, partitions, params);
  expectPoseEstimateClose(viaCentralizedMode, centralized, d, numPoses);

  params.mode = CCIInitMode::TED_CCI_SR_HIERARCHICAL;
  TEDCCIStats hierarchicalStats;
  const Matrix hierarchical = TEDCCISolver::Initialize(
      d, numPoses, measurements, partitions, params, &hierarchicalStats);
  expectPoseEstimateClose(hierarchical, centralized, d, numPoses);
  EXPECT_GT(hierarchicalStats.num_factor_messages, 0);
  EXPECT_GT(hierarchicalStats.max_separator_size, 0);
}

}  // namespace

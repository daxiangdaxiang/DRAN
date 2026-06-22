#include <DPGO/DCCI_chordal_initialization.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/InProcessDCCICommunicator.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
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

PoseID poseID(size_t pose) {
  return PoseID{0, static_cast<unsigned>(pose)};
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

std::vector<DistributedPartition> makeContiguousPartitions(size_t numPoses,
                                                           int numRobots) {
  std::vector<DistributedPartition> partitions(numRobots);
  std::unordered_map<PoseID, int, PoseIDHash> owner;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    const int robot = std::min<int>(
        static_cast<int>((pose * static_cast<size_t>(numRobots)) / numPoses),
        numRobots - 1);
    owner[poseID(pose)] = robot;
  }
  for (int robot = 0; robot < numRobots; ++robot) {
    partitions[robot].my_robot_id = robot;
    partitions[robot].num_robots = numRobots;
    partitions[robot].owner_robot = owner;
  }
  for (size_t pose = 0; pose < numPoses; ++pose) {
    const PoseID id = poseID(pose);
    DistributedPartition &partition = partitions[owner[id]];
    partition.global_to_local[id] = partition.global_pose_ids.size();
    partition.global_pose_ids.push_back(id);
  }
  return partitions;
}

DCCIParams tightDCCIParams() {
  DCCIParams params;
  params.rotation_pcg.max_iters = 500;
  params.rotation_pcg.rel_tol = 1e-14;
  params.rotation_pcg.abs_tol = 1e-14;
  params.translation_pcg = params.rotation_pcg;
  return params;
}

double maxRotationDiff(const Matrix &A, const Matrix &B, size_t d,
                       size_t numPoses) {
  double maxDiff = 0.0;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    maxDiff = std::max(maxDiff,
                       (A.block(0, pose * (d + 1), d, d) -
                        B.block(0, pose * (d + 1), d, d))
                           .norm());
  }
  return maxDiff;
}

double maxTranslationDiff(const Matrix &A, const Matrix &B, size_t d,
                          size_t numPoses) {
  double maxDiff = 0.0;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    maxDiff = std::max(maxDiff,
                       (A.block(0, pose * (d + 1) + d, d, 1) -
                        B.block(0, pose * (d + 1) + d, d, 1))
                           .norm());
  }
  return maxDiff;
}

void expectDCCIEqualsCentralized(
    size_t d, const std::vector<RelativeSEMeasurement> &measurements,
    size_t numPoses, int numRobots) {
  const auto partitions = makeContiguousPartitions(numPoses, numRobots);
  InProcessDCCICommunicator comm(partitions);
  DCCIStats stats;
  const Matrix distributed = distributedChordalInitialization(
      d, numPoses, measurements, partitions.front(), comm, tightDCCIParams(),
      &stats);
  const Matrix centralized = chordalInitialization(d, numPoses, measurements);
  const double rotDiff = maxRotationDiff(distributed, centralized, d, numPoses);
  const double transDiff =
      maxTranslationDiff(distributed, centralized, d, numPoses);
  const double relPoseDiff =
      (distributed - centralized).norm() / std::max(1.0, centralized.norm());

  std::cout << "DCCI init d=" << d << " robots=" << numRobots
            << " rot_diff=" << rotDiff << " trans_diff=" << transDiff
            << " rel_pose_diff=" << relPoseDiff
            << " rot_iters=" << stats.rotation_pcg.iters
            << " trans_iters=" << stats.translation_pcg.iters
            << " scalar_reductions="
            << stats.communication.num_scalar_reductions << std::endl;

  EXPECT_LT(rotDiff, 1e-8);
  EXPECT_LT(transDiff, 1e-8);
  EXPECT_LT(relPoseDiff, 1e-8);
  EXPECT_TRUE(stats.rotation_pcg.converged);
  EXPECT_TRUE(stats.translation_pcg.converged);
  EXPECT_GT(stats.communication.num_scalar_reductions, 0u);
}

}  // namespace

TEST(testDPGO, DCCIInitializationMatchesCentralized2D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic2DGraph(&numPoses);
  for (int numRobots : {1, 2, 4}) {
    expectDCCIEqualsCentralized(2, measurements, numPoses, numRobots);
  }
}

TEST(testDPGO, DCCIInitializationMatchesCentralized3D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic3DGraph(&numPoses);
  for (int numRobots : {1, 2, 4}) {
    expectDCCIEqualsCentralized(3, measurements, numPoses, numRobots);
  }
}

TEST(testDPGO, DCCIInitializationTraceRecordsProcessFrames) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic3DGraph(&numPoses);
  const auto partitions = makeContiguousPartitions(numPoses, 2);
  InProcessDCCICommunicator comm(partitions);
  DCCIStats stats;
  std::vector<DCCITraceFrame> frames;
  DCCITraceOptions trace;
  trace.frames = &frames;
  trace.frame_stride = 1;
  trace.max_frames_per_stage = 32;

  const Matrix distributed = distributedChordalInitialization(
      3, numPoses, measurements, partitions.front(), comm, tightDCCIParams(),
      &stats, &trace);

  ASSERT_GE(frames.size(), 2u);
  for (const auto &frame : frames) {
    EXPECT_EQ(frame.poses.rows(), 3);
    EXPECT_EQ(frame.poses.cols(), static_cast<int>(numPoses * 4));
    EXPECT_TRUE(std::isfinite(frame.residual_norm));
    EXPECT_EQ(frame.stage, DCCITraceStage::TranslationPCG);
  }
  const Matrix finalDiff = frames.back().poses - distributed;
  EXPECT_LT(finalDiff.norm(), 1e-12);
}

TEST(testDPGO, DCCIInitializationMatchesWeightedCentralized3D) {
  size_t numPoses = 0;
  auto measurements = makeSynthetic3DGraph(&numPoses);
  measurements[0].weight = 0.20;
  measurements[2].weight = 0.55;
  measurements[4].weight = 0.75;

  const int numRobots = 4;
  const auto partitions = makeContiguousPartitions(numPoses, numRobots);
  InProcessDCCICommunicator comm(partitions);
  DCCIParams params = tightDCCIParams();
  params.use_measurement_weight = true;
  DCCIStats stats;
  const Matrix distributed = distributedChordalInitialization(
      3, numPoses, measurements, partitions.front(), comm, params, &stats);
  const Matrix centralized = chordalInitialization(3, numPoses, measurements,
                                                  true);

  const double rotDiff = maxRotationDiff(distributed, centralized, 3, numPoses);
  const double transDiff =
      maxTranslationDiff(distributed, centralized, 3, numPoses);
  const double relPoseDiff =
      (distributed - centralized).norm() / std::max(1.0, centralized.norm());
  std::cout << "DCCI weighted init d=3 robots=" << numRobots
            << " rot_diff=" << rotDiff << " trans_diff=" << transDiff
            << " rel_pose_diff=" << relPoseDiff
            << " rot_iters=" << stats.rotation_pcg.iters
            << " trans_iters=" << stats.translation_pcg.iters << std::endl;

  EXPECT_LT(rotDiff, 1e-8);
  EXPECT_LT(transDiff, 1e-8);
  EXPECT_LT(relPoseDiff, 1e-8);
  EXPECT_TRUE(stats.rotation_pcg.converged);
  EXPECT_TRUE(stats.translation_pcg.converged);
}

TEST(testDPGO, DCCIInitializationRejectsDisconnectedGraph) {
  const size_t d = 2;
  const size_t numPoses = 4;
  const Matrix R = Matrix::Identity(d, d);
  Vector z = Vector::Zero(d);
  std::vector<RelativeSEMeasurement> measurements;
  measurements.emplace_back(makeMeasurement(0, 1, R, z, R, z, 1.0, 1.0));
  measurements.emplace_back(makeMeasurement(2, 3, R, z, R, z, 1.0, 1.0));
  const auto partitions = makeContiguousPartitions(numPoses, 2);
  InProcessDCCICommunicator comm(partitions);
  EXPECT_THROW(distributedChordalInitialization(
                   d, numPoses, measurements, partitions.front(), comm,
                   tightDCCIParams(), nullptr),
               std::invalid_argument);
}

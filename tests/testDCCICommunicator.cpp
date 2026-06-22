#include <DPGO/DCCI_linear_operators.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/InProcessDCCICommunicator.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Geometry>

#include <cmath>
#include <iostream>
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

PoseID poseID(size_t pose) {
  return PoseID{0, static_cast<unsigned>(pose)};
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

Vector deterministicVector(size_t size) {
  Vector x(size);
  for (size_t i = 0; i < size; ++i) {
    x(i) = std::sin(0.23 * static_cast<double>(i + 1)) -
           0.4 * std::cos(0.37 * static_cast<double>(i + 1));
  }
  return x;
}

Matrix reducedRotationBlock(const Vector &x, size_t d, size_t pose) {
  if (pose == 0) {
    return Matrix::Zero(d, d);
  }
  return Eigen::Map<const Matrix>(x.data() + (pose - 1) * d * d, d, d);
}

Vector reducedTranslationBlock(const Vector &x, size_t d, size_t pose) {
  if (pose == 0) {
    return Vector::Zero(d);
  }
  return Eigen::Map<const Vector>(x.data() + (pose - 1) * d, d);
}

void addRotationBlock(Vector *y, size_t d, size_t pose, const Matrix &block) {
  if (pose == 0) {
    return;
  }
  Eigen::Map<Matrix>(y->data() + (pose - 1) * d * d, d, d) += block;
}

void addTranslationBlock(Vector *y, size_t d, size_t pose,
                         const Vector &block) {
  if (pose == 0) {
    return;
  }
  Eigen::Map<Vector>(y->data() + (pose - 1) * d, d) += block;
}

int ownerOf(const DistributedPartition &partition, size_t pose) {
  return partition.owner_robot.at(poseID(pose));
}

Vector distributedRotationApplyH(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements,
    InProcessDCCICommunicator *comm) {
  Vector y = Vector::Zero((numPoses - 1) * d * d);

  for (int robot = 0; robot < comm->numRobots(); ++robot) {
    const auto &partition = comm->partition(robot);
    std::vector<PoseID> requested;
    for (const auto &measurement : measurements) {
      const bool ownsP1 = ownerOf(partition, measurement.p1) == robot;
      const bool ownsP2 = ownerOf(partition, measurement.p2) == robot;
      if (!ownsP1 && !ownsP2) {
        continue;
      }
      if (!ownsP1 && measurement.p1 != 0) {
        requested.push_back(poseID(measurement.p1));
      }
      if (!ownsP2 && measurement.p2 != 0) {
        requested.push_back(poseID(measurement.p2));
      }
    }
    const auto ghosts = comm->exchangeRotationBlocks(robot, requested);

    auto blockFor = [&](size_t pose) -> Matrix {
      if (pose == 0) {
        return Matrix::Zero(d, d);
      }
      const PoseID id = poseID(pose);
      if (ownerOf(partition, pose) == robot) {
        return comm->rotationBlock(id);
      }
      return ghosts.at(id);
    };

    for (const auto &measurement : measurements) {
      const bool ownsP1 = ownerOf(partition, measurement.p1) == robot;
      const bool ownsP2 = ownerOf(partition, measurement.p2) == robot;
      if (!ownsP1 && !ownsP2) {
        continue;
      }
      const Matrix E = blockFor(measurement.p1) * measurement.R -
                       blockFor(measurement.p2);
      if (ownsP1 && measurement.p1 != 0) {
        addRotationBlock(&y, d, measurement.p1,
                         measurement.kappa * E * measurement.R.transpose());
      }
      if (ownsP2 && measurement.p2 != 0) {
        addRotationBlock(&y, d, measurement.p2, -measurement.kappa * E);
      }
    }
  }

  return y;
}

Vector distributedTranslationApplyH(
    size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements,
    InProcessDCCICommunicator *comm) {
  Vector y = Vector::Zero((numPoses - 1) * d);

  for (int robot = 0; robot < comm->numRobots(); ++robot) {
    const auto &partition = comm->partition(robot);
    std::vector<PoseID> requested;
    for (const auto &measurement : measurements) {
      const bool ownsP1 = ownerOf(partition, measurement.p1) == robot;
      const bool ownsP2 = ownerOf(partition, measurement.p2) == robot;
      if (!ownsP1 && !ownsP2) {
        continue;
      }
      if (!ownsP1 && measurement.p1 != 0) {
        requested.push_back(poseID(measurement.p1));
      }
      if (!ownsP2 && measurement.p2 != 0) {
        requested.push_back(poseID(measurement.p2));
      }
    }
    const auto ghosts = comm->exchangeTranslationBlocks(robot, requested);

    auto blockFor = [&](size_t pose) -> Vector {
      if (pose == 0) {
        return Vector::Zero(d);
      }
      const PoseID id = poseID(pose);
      if (ownerOf(partition, pose) == robot) {
        return comm->translationBlock(id);
      }
      return ghosts.at(id);
    };

    for (const auto &measurement : measurements) {
      const bool ownsP1 = ownerOf(partition, measurement.p1) == robot;
      const bool ownsP2 = ownerOf(partition, measurement.p2) == robot;
      if (!ownsP1 && !ownsP2) {
        continue;
      }
      const Vector u = blockFor(measurement.p2) - blockFor(measurement.p1);
      if (ownsP1 && measurement.p1 != 0) {
        addTranslationBlock(&y, d, measurement.p1, -measurement.tau * u);
      }
      if (ownsP2 && measurement.p2 != 0) {
        addTranslationBlock(&y, d, measurement.p2, measurement.tau * u);
      }
    }
  }

  return y;
}

Matrix extractRotations(const Matrix &T, size_t d, size_t numPoses) {
  Matrix rotations(d, d * numPoses);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    rotations.block(0, pose * d, d, d) =
        T.block(0, pose * (d + 1), d, d);
  }
  return rotations;
}

void expectDistributedApplyHMatchesCentralized(
    size_t d, const std::vector<RelativeSEMeasurement> &measurements,
    size_t numPoses, int numRobots) {
  const Vector xRot = deterministicVector((numPoses - 1) * d * d);
  const Vector xTrans = deterministicVector((numPoses - 1) * d);
  const auto partitions = makeContiguousPartitions(numPoses, numRobots);
  InProcessDCCICommunicator comm(partitions);

  for (size_t pose = 0; pose < numPoses; ++pose) {
    comm.setRotationBlock(poseID(pose), reducedRotationBlock(xRot, d, pose));
    comm.setTranslationBlock(poseID(pose),
                             reducedTranslationBlock(xTrans, d, pose));
  }

  const ChordalRotationLinearOperator rotationOp(d, numPoses, measurements);
  const Matrix T = chordalInitialization(d, numPoses, measurements);
  const Matrix rotations = extractRotations(T, d, numPoses);
  const ChordalTranslationLinearOperator translationOp(d, numPoses,
                                                       measurements, rotations);

  const Vector yRot = distributedRotationApplyH(d, numPoses, measurements,
                                                &comm);
  const Vector yTrans = distributedTranslationApplyH(d, numPoses, measurements,
                                                     &comm);
  const double rotErr =
      (yRot - rotationOp.applyH(xRot)).norm() /
      std::max(1.0, rotationOp.applyH(xRot).norm());
  const double transErr =
      (yTrans - translationOp.applyH(xTrans)).norm() /
      std::max(1.0, translationOp.applyH(xTrans).norm());

  std::cout << "DCCI in-process communicator d=" << d
            << " robots=" << numRobots << " rotation_rel_error=" << rotErr
            << " translation_rel_error=" << transErr
            << " block_messages=" << comm.stats().num_block_messages
            << " bytes_sent=" << comm.stats().bytes_sent << std::endl;

  EXPECT_LT(rotErr, 1e-12);
  EXPECT_LT(transErr, 1e-12);
  if (numRobots == 1) {
    EXPECT_EQ(comm.stats().num_block_messages, 0u);
    EXPECT_EQ(comm.stats().bytes_sent, 0u);
  } else {
    EXPECT_GT(comm.stats().num_block_messages, 0u);
    EXPECT_GT(comm.stats().bytes_sent, 0u);
  }
}

}  // namespace

TEST(testDPGO, DCCIInProcessCommunicatorAllReduceTracksStats) {
  const auto partitions = makeContiguousPartitions(4, 2);
  InProcessDCCICommunicator comm(partitions);
  EXPECT_DOUBLE_EQ(comm.allReduceSum(1.25), 1.25);
  EXPECT_DOUBLE_EQ(comm.allReduceSum(-0.5), -0.5);
  EXPECT_EQ(comm.stats().num_scalar_reductions, 2u);
  EXPECT_EQ(comm.stats().num_block_messages, 0u);
}

TEST(testDPGO, DCCIInProcessCommunicatorDistributedApplyHMatchesCentralized2D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic2DGraph(&numPoses);
  for (int numRobots : {1, 2, 4}) {
    expectDistributedApplyHMatchesCentralized(2, measurements, numPoses,
                                              numRobots);
  }
}

TEST(testDPGO, DCCIInProcessCommunicatorDistributedApplyHMatchesCentralized3D) {
  size_t numPoses = 0;
  const auto measurements = makeSynthetic3DGraph(&numPoses);
  for (int numRobots : {1, 2, 4}) {
    expectDistributedApplyHMatchesCentralized(3, measurements, numPoses,
                                              numRobots);
  }
}

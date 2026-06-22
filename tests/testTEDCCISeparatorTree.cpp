#include <DPGO/TEDCCI.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

Matrix deterministicMatrix(int rows, int cols, double phase) {
  Matrix A(rows, cols);
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      A(r, c) = std::sin(phase + 0.11 * static_cast<double>((r + 1) * (c + 2))) +
                0.25 * std::cos(0.23 * static_cast<double>(r + c + 3));
    }
  }
  return A;
}

Vector deterministicVector(int rows, double phase) {
  Vector b(rows);
  for (int r = 0; r < rows; ++r) {
    b(r) = std::cos(phase + 0.17 * static_cast<double>(r + 1)) +
           0.12 * std::sin(0.37 * static_cast<double>(r + 2));
  }
  return b;
}

Matrix rotation2(double theta) {
  Matrix R(2, 2);
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  R << c, -s, s, c;
  return R;
}

CondensedFactor makeRobotFactor(int robot, int blockDim) {
  CondensedFactor factor;
  factor.owner_robot_id = robot;
  factor.factor_type = FactorType::TRANSLATION;
  factor.boundary_keys = {PoseKey{robot, 10}, PoseKey{robot, 20}};
  factor.Abar = deterministicMatrix(7,
                                    static_cast<int>(factor.boundary_keys.size()) *
                                        blockDim,
                                    0.2 + 0.4 * robot);
  factor.bbar = deterministicVector(7, 0.3 + 0.2 * robot);
  return factor;
}

LinearFactorBlock makeCrossFactor(int robotA, int robotB, int blockDim) {
  LinearFactorBlock factor;
  factor.keys = {PoseKey{robotA, 20}, PoseKey{robotB, 20}};
  factor.A = deterministicMatrix(3,
                                 static_cast<int>(factor.keys.size()) *
                                     blockDim,
                                 1.1 + 0.3 * robotA);
  factor.b = deterministicVector(3, 1.5 + 0.1 * robotB);
  return factor;
}

std::vector<TEDCCIPartition> makePartitions(int robots) {
  std::vector<TEDCCIPartition> partitions;
  for (int robot = 0; robot < robots; ++robot) {
    TEDCCIPartition partition;
    partition.local_robot_id = robot;
    partition.local_poses = {PoseKey{robot, 10}, PoseKey{robot, 20}};
    partition.boundary_poses = partition.local_poses;
    partitions.push_back(partition);
  }
  return partitions;
}

RelativeSEMeasurement makeMeasurement(size_t i, size_t j,
                                      const std::vector<int> &owner,
                                      const std::vector<Matrix> &rotations,
                                      double kappa, double tau) {
  const Vector zero = Vector::Zero(2);
  return RelativeSEMeasurement(owner.at(i), owner.at(j), i, j,
                               rotations[i].transpose() * rotations[j],
                               zero, kappa, tau);
}

std::vector<TEDCCIPartition> makeTedPartitions(
    size_t numPoses, int robots,
    const std::vector<RelativeSEMeasurement> &measurements,
    const TEDCCIParams &params) {
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
      if (static_cast<int>((pose * static_cast<size_t>(robots)) / numPoses) ==
          robot) {
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

std::vector<PoseKey> withoutAnchor(std::vector<PoseKey> keys) {
  const PoseKey anchor{0, 0};
  keys.erase(std::remove(keys.begin(), keys.end(), anchor), keys.end());
  return keys;
}

void expectSeparatorTreeMatchesInterfaceDirectSolver(int robots) {
  const int blockDim = 2;
  std::vector<CondensedFactor> condensed;
  for (int robot = 0; robot < robots; ++robot) {
    condensed.push_back(makeRobotFactor(robot, blockDim));
  }
  std::vector<LinearFactorBlock> cross;
  for (int robot = 0; robot + 1 < robots; ++robot) {
    cross.push_back(makeCrossFactor(robot, robot + 1, blockDim));
  }

  TEDCCIParams params;
  TEDCCIStats directStats;
  const Vector direct = InterfaceDirectSolver::Solve(
      condensed, cross, blockDim, params, &directStats);

  const SeparatorTree tree =
      SeparatorTree::BuildFromRobotGraph(makePartitions(robots), params);
  TEDCCIStats hierarchicalStats;
  const Vector hierarchical = tree.SolveHierarchical(
      condensed, cross, blockDim, &hierarchicalStats);

  EXPECT_LT((hierarchical - direct).norm(), 1e-8);
  EXPECT_GT(hierarchicalStats.max_separator_size, 0);
  EXPECT_LE(hierarchicalStats.max_separator_size, directStats.num_interface_vars);
  EXPECT_GT(hierarchicalStats.num_factor_messages, 0);
}

}  // namespace

TEST(testDPGO, TEDCCISeparatorTreeMatchesInterfaceDirectSolverForRobotScales) {
  expectSeparatorTreeMatchesInterfaceDirectSolver(2);
  expectSeparatorTreeMatchesInterfaceDirectSolver(4);
  expectSeparatorTreeMatchesInterfaceDirectSolver(8);
}

TEST(testDPGO, TEDCCISeparatorTreeMatchesDirectOnRealRotationFactors) {
  const int d = 2;
  const int blockDim = d * d;
  const size_t numPoses = 8;
  const int robots = 4;
  const std::vector<int> owner = {0, 0, 1, 1, 2, 2, 3, 3};
  std::vector<Matrix> rotations;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    rotations.push_back(rotation2(0.12 * static_cast<double>(pose)));
  }

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < numPoses; ++pose) {
    measurements.push_back(makeMeasurement(pose, pose + 1, owner, rotations,
                                           2.0 + pose, 3.0));
  }
  measurements.push_back(makeMeasurement(1, 4, owner, rotations, 4.0, 2.0));
  measurements.push_back(makeMeasurement(3, 6, owner, rotations, 3.5, 2.5));

  TEDCCIParams params;
  params.use_measurement_weight = false;
  const auto partitions =
      makeTedPartitions(numPoses, robots, measurements, params);

  std::vector<RelativeSEMeasurement> shared;
  std::vector<CondensedFactor> condensed;
  for (const auto &partition : partitions) {
    std::vector<RelativeSEMeasurement> local;
    for (const auto &measurement : measurements) {
      if (measurement.r1 == measurement.r2 &&
          static_cast<int>(measurement.r1) == partition.local_robot_id) {
        local.push_back(measurement);
      } else if (measurement.r1 != measurement.r2) {
        shared.push_back(measurement);
      }
    }
    const auto factors = CCIFactorBuilder::BuildRotationFactors(
        d, local, params);
    CondensedFactor factor;
    factor.owner_robot_id = partition.local_robot_id;
    factor.factor_type = FactorType::ROTATION;
    BackSubstitutionCache cache;
    LocalQRCondensation::Condense(
        factors, partition.interior_poses,
        withoutAnchor(partition.boundary_poses), blockDim, &factor, &cache);
    condensed.push_back(std::move(factor));
  }
  std::sort(shared.begin(), shared.end(),
            [](const RelativeSEMeasurement &a,
               const RelativeSEMeasurement &b) {
              return std::tie(a.p1, a.p2, a.r1, a.r2) <
                     std::tie(b.p1, b.p2, b.r1, b.r2);
            });
  shared.erase(std::unique(shared.begin(), shared.end(),
                           [](const RelativeSEMeasurement &a,
                              const RelativeSEMeasurement &b) {
                             return a.p1 == b.p1 && a.p2 == b.p2 &&
                                    a.r1 == b.r1 && a.r2 == b.r2;
                           }),
               shared.end());
  const auto crossFactors =
      CCIFactorBuilder::BuildRotationFactors(d, shared, params);

  TEDCCIStats directStats;
  const Vector direct = InterfaceDirectSolver::Solve(
      condensed, crossFactors, blockDim, params, &directStats);
  const SeparatorTree tree =
      SeparatorTree::BuildFromRobotGraph(partitions, params);
  TEDCCIStats hierarchicalStats;
  const Vector hierarchical = tree.SolveHierarchical(
      condensed, crossFactors, blockDim, &hierarchicalStats);

  EXPECT_LT((hierarchical - direct).norm(), 1e-8);
  EXPECT_GT(hierarchicalStats.bytes_sent_upward, 0u);
  EXPECT_GT(hierarchicalStats.bytes_sent_downward, 0u);
}

TEST(testDPGO, TEDCCISeparatorTreePlansDeterministicParentFailureResend) {
  TEDCCIParams params;
  const SeparatorTree tree =
      SeparatorTree::BuildFromRobotGraph(makePartitions(4), params);

  const SeparatorFailureRecoveryPlan plan =
      tree.PlanParentNodeFailureRecovery(/*failed_node_id=*/0);

  ASSERT_EQ(plan.failed_node_id, 0);
  EXPECT_EQ(plan.replacement_robot_id, 1);
  ASSERT_EQ(plan.resend_requests.size(), 2u);
  EXPECT_EQ(plan.resend_requests[0].sender_node_id, 1);
  EXPECT_EQ(plan.resend_requests[0].sender_robot_id, 0);
  EXPECT_EQ(plan.resend_requests[0].receiver_node_id, 0);
  EXPECT_EQ(plan.resend_requests[0].receiver_robot_id, 1);
  EXPECT_EQ(plan.resend_requests[1].sender_node_id, 4);
  EXPECT_EQ(plan.resend_requests[1].sender_robot_id, 2);
  EXPECT_EQ(plan.resend_requests[1].receiver_node_id, 0);
  EXPECT_EQ(plan.resend_requests[1].receiver_robot_id, 1);

  std::map<int, CondensedFactor> latest;
  latest[1] = makeRobotFactor(1, 2);
  latest[1].owner_robot_id = 1;
  latest[4] = makeRobotFactor(4, 2);
  latest[4].owner_robot_id = 4;

  const std::vector<FactorMessage> messages =
      tree.MakeRecoveryFactorMessages(plan, latest, /*epoch=*/42,
                                      /*topology_epoch=*/7);

  ASSERT_EQ(messages.size(), 2u);
  EXPECT_EQ(messages[0].sender_robot_id, 0);
  EXPECT_EQ(messages[0].receiver_robot_id, 1);
  EXPECT_EQ(messages[0].epoch, 42u);
  EXPECT_EQ(messages[0].topology_epoch, 7u);
  EXPECT_EQ(messages[0].factor_type, FactorType::TRANSLATION);
  EXPECT_EQ(messages[0].checksum, ComputeFactorMessageChecksum(messages[0]));
  EXPECT_EQ(messages[1].sender_robot_id, 2);
  EXPECT_EQ(messages[1].receiver_robot_id, 1);
  EXPECT_EQ(messages[1].epoch, 42u);
  EXPECT_EQ(messages[1].topology_epoch, 7u);
  EXPECT_EQ(messages[1].factor_type, FactorType::TRANSLATION);
  EXPECT_EQ(messages[1].checksum, ComputeFactorMessageChecksum(messages[1]));
}

TEST(testDPGO, TEDCCISeparatorTreeRuntimePlansViewEpochOwnerChangeResend) {
  TEDCCIParams params;
  const SeparatorTree tree =
      SeparatorTree::BuildFromRobotGraph(makePartitions(4), params);
  SeparatorTreeRuntime runtime(tree);

  const std::vector<SeparatorNodeOwnership> fullView =
      runtime.ElectOwners({0, 1, 2, 3}, /*topology_epoch=*/10);
  const std::vector<SeparatorNodeOwnership> recoveredView =
      runtime.ElectOwners({1, 2, 3}, /*topology_epoch=*/11);

  ASSERT_EQ(fullView.size(), tree.nodes().size());
  ASSERT_EQ(recoveredView.size(), tree.nodes().size());
  EXPECT_EQ(fullView[0].owner_robot_id, 0);
  EXPECT_EQ(fullView[0].topology_epoch, 10u);
  EXPECT_EQ(recoveredView[0].owner_robot_id, 1);
  EXPECT_EQ(recoveredView[0].topology_epoch, 11u);
  EXPECT_EQ(recoveredView[1].owner_robot_id, 1);
  EXPECT_EQ(recoveredView[2].owner_robot_id, -1);
  EXPECT_EQ(runtime.ParentOwnerForChildNode(4, recoveredView), 1);

  const SeparatorFailureRecoveryPlan plan =
      runtime.PlanParentOwnerChangeResend(fullView, recoveredView,
                                          /*parent_node_id=*/0);

  EXPECT_EQ(plan.failed_node_id, 0);
  EXPECT_EQ(plan.replacement_robot_id, 1);
  EXPECT_EQ(plan.topology_epoch, 11u);
  ASSERT_EQ(plan.resend_requests.size(), 1u);
  EXPECT_EQ(plan.resend_requests[0].sender_node_id, 4);
  EXPECT_EQ(plan.resend_requests[0].sender_robot_id, 2);
  EXPECT_EQ(plan.resend_requests[0].receiver_node_id, 0);
  EXPECT_EQ(plan.resend_requests[0].receiver_robot_id, 1);
}

TEST(testDPGO, TEDCCISeparatorTreeMessageControllerReplaysCachedLatestFactor) {
  TEDCCIParams params;
  const SeparatorTree tree =
      SeparatorTree::BuildFromRobotGraph(makePartitions(4), params);
  SeparatorTreeMessageController controller(tree);

  const std::vector<SeparatorNodeOwnership> fullView =
      controller.ElectOwners({0, 1, 2, 3}, /*topology_epoch=*/20);
  const std::vector<SeparatorNodeOwnership> recoveredView =
      controller.ElectOwners({1, 2, 3}, /*topology_epoch=*/21);

  CondensedFactor latest = makeRobotFactor(4, 2);
  latest.owner_robot_id = 2;
  latest.epoch = 8;
  latest.bbar.setConstant(8.0);
  CondensedFactor stale = latest;
  stale.epoch = 7;
  stale.bbar.setConstant(7.0);

  controller.UpdateLatestNodeFactor(/*node_id=*/4, latest);
  controller.UpdateLatestNodeFactor(/*node_id=*/4, stale);

  const std::vector<FactorMessage> messages =
      controller.MakeOwnerChangeRecoveryMessages(
          fullView, recoveredView, /*parent_node_id=*/0, /*epoch=*/42);

  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages[0].sender_robot_id, 2);
  EXPECT_EQ(messages[0].receiver_robot_id, 1);
  EXPECT_EQ(messages[0].epoch, 42u);
  EXPECT_EQ(messages[0].topology_epoch, 21u);
  EXPECT_EQ(messages[0].factor_type, FactorType::TRANSLATION);
  EXPECT_TRUE(messages[0].b.isApprox(latest.bbar));
  EXPECT_EQ(messages[0].checksum, ComputeFactorMessageChecksum(messages[0]));
}

TEST(testDPGO, TEDCCISeparatorTreeMessageControllerSendsReplayThroughExchange) {
  TEDCCIParams params;
  const SeparatorTree tree =
      SeparatorTree::BuildFromRobotGraph(makePartitions(4), params);
  SeparatorTreeMessageController controller(tree);
  InProcessAsyncFactorExchange exchange;

  const std::vector<SeparatorNodeOwnership> fullView =
      controller.ElectOwners({0, 1, 2, 3}, /*topology_epoch=*/30);
  const std::vector<SeparatorNodeOwnership> recoveredView =
      controller.ElectOwners({1, 2, 3}, /*topology_epoch=*/31);

  CondensedFactor latest = makeRobotFactor(4, 2);
  latest.owner_robot_id = 2;
  latest.epoch = 9;
  controller.UpdateLatestNodeFactor(/*node_id=*/4, latest);

  const std::size_t sent = controller.SendOwnerChangeRecoveryMessages(
      fullView, recoveredView, /*parent_node_id=*/0, /*epoch=*/43,
      &exchange);

  EXPECT_EQ(sent, 1u);
  std::vector<FactorMessage> messages = exchange.PollFactorMessages(1);
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages[0].sender_robot_id, 2);
  EXPECT_EQ(messages[0].receiver_robot_id, 1);
  EXPECT_EQ(messages[0].epoch, 43u);
  EXPECT_EQ(messages[0].topology_epoch, 31u);
  EXPECT_EQ(messages[0].checksum, ComputeFactorMessageChecksum(messages[0]));
}

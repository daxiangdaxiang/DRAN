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

RelativeSEMeasurement makeMeasurement(size_t i, size_t j,
                                      const std::vector<int> &owner,
                                      const Matrix &Ri, const Vector &ti,
                                      const Matrix &Rj, const Vector &tj,
                                      double kappa, double tau) {
  return RelativeSEMeasurement(owner.at(i), owner.at(j), i, j,
                               Ri.transpose() * Rj,
                               Ri.transpose() * (tj - ti), kappa, tau);
}

std::vector<TEDCCIPartition> makeTwoRobotPartitions(
    const std::vector<RelativeSEMeasurement> &local0,
    const std::vector<RelativeSEMeasurement> &local1,
    const std::vector<RelativeSEMeasurement> &shared,
    const TEDCCIParams &params) {
  const std::vector<PoseKey> robot0 = {PoseKey{0, 0}, PoseKey{0, 1}};
  const std::vector<PoseKey> robot1 = {PoseKey{1, 2}, PoseKey{1, 3}};
  return {
      BoundarySelector::SelectBoundaryVariables(0, robot0, local0, shared,
                                                params),
      BoundarySelector::SelectBoundaryVariables(1, robot1, local1, shared,
                                                params),
  };
}

struct TinyTwoRobotGraph {
  int dimension = 2;
  int num_poses = 4;
  std::vector<int> owner;
  std::vector<RelativeSEMeasurement> local0;
  std::vector<RelativeSEMeasurement> local1;
  std::vector<RelativeSEMeasurement> shared;
  std::vector<RelativeSEMeasurement> all;
};

TinyTwoRobotGraph makeTinyTwoRobotGraph() {
  TinyTwoRobotGraph graph;
  graph.owner = {0, 0, 1, 1};
  const std::vector<Matrix> rotations = {
      rotation2(0.0), rotation2(0.22), rotation2(0.47), rotation2(0.76)};
  std::vector<Vector> translations(4, Vector::Zero(2));
  translations[0] << 0.0, 0.0;
  translations[1] << 1.0, 0.1;
  translations[2] << 2.1, 0.6;
  translations[3] << 3.0, 1.0;

  graph.local0.push_back(makeMeasurement(
      0, 1, graph.owner, rotations[0], translations[0], rotations[1],
      translations[1], 2.0, 3.0));
  graph.local1.push_back(makeMeasurement(
      2, 3, graph.owner, rotations[2], translations[2], rotations[3],
      translations[3], 2.5, 3.5));
  graph.shared.push_back(makeMeasurement(
      1, 2, graph.owner, rotations[1], translations[1], rotations[2],
      translations[2], 4.0, 2.0));
  graph.shared.push_back(makeMeasurement(
      0, 3, graph.owner, rotations[0], translations[0], rotations[3],
      translations[3], 1.8, 2.2));

  graph.all = graph.local0;
  graph.all.insert(graph.all.end(), graph.local1.begin(), graph.local1.end());
  graph.all.insert(graph.all.end(), graph.shared.begin(), graph.shared.end());
  return graph;
}

void pumpNetwork(TEDCCIOrchestrator *first, TEDCCIOrchestrator *second,
                 int rounds = 8) {
  for (int round = 0; round < rounds; ++round) {
    first->ProcessIncomingMessages();
    second->ProcessIncomingMessages();
  }
}

void pumpNetwork(std::vector<TEDCCIOrchestrator *> robots, int rounds = 10) {
  for (int round = 0; round < rounds; ++round) {
    for (TEDCCIOrchestrator *robot : robots) {
      robot->ProcessIncomingMessages();
    }
  }
}

void expectPoseEstimateClose(const Matrix &actual, const Matrix &expected,
                             int d, int numPoses) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  double maxRot = 0.0;
  double maxTrans = 0.0;
  for (int pose = 0; pose < numPoses; ++pose) {
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
  EXPECT_LT(maxRot, 1e-8);
  EXPECT_LT(maxTrans, 1e-8);
}

struct ThreeRobotPartitionedGraph {
  int dimension = 2;
  int num_poses = 6;
  std::vector<int> owner;
  std::vector<RelativeSEMeasurement> local0;
  std::vector<RelativeSEMeasurement> local1;
  std::vector<RelativeSEMeasurement> local2;
  std::vector<RelativeSEMeasurement> shared01;
  std::vector<RelativeSEMeasurement> component01;
};

ThreeRobotPartitionedGraph makeThreeRobotPartitionedGraph() {
  ThreeRobotPartitionedGraph graph;
  graph.owner = {0, 0, 1, 1, 2, 2};
  const std::vector<Matrix> rotations = {
      rotation2(0.0), rotation2(0.15), rotation2(0.38),
      rotation2(0.54), rotation2(-0.10), rotation2(-0.02)};
  std::vector<Vector> translations(6, Vector::Zero(2));
  translations[0] << 0.0, 0.0;
  translations[1] << 0.9, 0.1;
  translations[2] << 1.8, 0.5;
  translations[3] << 2.7, 0.9;
  translations[4] << -0.4, 1.5;
  translations[5] << 0.4, 1.7;

  graph.local0.push_back(makeMeasurement(
      0, 1, graph.owner, rotations[0], translations[0], rotations[1],
      translations[1], 2.0, 3.0));
  graph.local1.push_back(makeMeasurement(
      2, 3, graph.owner, rotations[2], translations[2], rotations[3],
      translations[3], 2.5, 3.5));
  graph.local2.push_back(makeMeasurement(
      4, 5, graph.owner, rotations[4], translations[4], rotations[5],
      translations[5], 2.2, 2.8));
  graph.shared01.push_back(makeMeasurement(
      1, 2, graph.owner, rotations[1], translations[1], rotations[2],
      translations[2], 4.0, 2.0));

  graph.component01 = graph.local0;
  graph.component01.insert(graph.component01.end(), graph.local1.begin(),
                           graph.local1.end());
  graph.component01.insert(graph.component01.end(), graph.shared01.begin(),
                           graph.shared01.end());
  return graph;
}

struct ThreeRobotChainGraph {
  int dimension = 2;
  int num_poses = 6;
  std::vector<int> owner;
  std::vector<RelativeSEMeasurement> local0;
  std::vector<RelativeSEMeasurement> local1;
  std::vector<RelativeSEMeasurement> local2;
  std::vector<RelativeSEMeasurement> shared01;
  std::vector<RelativeSEMeasurement> shared12;
  std::vector<RelativeSEMeasurement> shared_all;
  std::vector<RelativeSEMeasurement> component01;
  std::vector<RelativeSEMeasurement> all;
};

ThreeRobotChainGraph makeThreeRobotChainGraph() {
  ThreeRobotChainGraph graph;
  graph.owner = {0, 0, 1, 1, 2, 2};
  const std::vector<Matrix> rotations = {
      rotation2(0.0), rotation2(0.12), rotation2(0.31),
      rotation2(0.50), rotation2(0.72), rotation2(0.88)};
  std::vector<Vector> translations(6, Vector::Zero(2));
  translations[0] << 0.0, 0.0;
  translations[1] << 0.8, 0.1;
  translations[2] << 1.7, 0.4;
  translations[3] << 2.4, 0.8;
  translations[4] << 3.1, 1.2;
  translations[5] << 3.8, 1.5;

  graph.local0.push_back(makeMeasurement(
      0, 1, graph.owner, rotations[0], translations[0], rotations[1],
      translations[1], 2.0, 3.0));
  graph.local1.push_back(makeMeasurement(
      2, 3, graph.owner, rotations[2], translations[2], rotations[3],
      translations[3], 2.5, 3.5));
  graph.local2.push_back(makeMeasurement(
      4, 5, graph.owner, rotations[4], translations[4], rotations[5],
      translations[5], 2.2, 2.8));
  graph.shared01.push_back(makeMeasurement(
      1, 2, graph.owner, rotations[1], translations[1], rotations[2],
      translations[2], 4.0, 2.0));
  graph.shared12.push_back(makeMeasurement(
      3, 4, graph.owner, rotations[3], translations[3], rotations[4],
      translations[4], 3.6, 2.4));

  graph.shared_all = graph.shared01;
  graph.shared_all.insert(graph.shared_all.end(), graph.shared12.begin(),
                          graph.shared12.end());

  graph.component01 = graph.local0;
  graph.component01.insert(graph.component01.end(), graph.local1.begin(),
                           graph.local1.end());
  graph.component01.insert(graph.component01.end(), graph.shared01.begin(),
                           graph.shared01.end());

  graph.all = graph.component01;
  graph.all.insert(graph.all.end(), graph.local2.begin(), graph.local2.end());
  graph.all.insert(graph.all.end(), graph.shared12.begin(),
                   graph.shared12.end());
  return graph;
}

std::vector<TEDCCIPartition> makeThreeRobotPartitions(
    const ThreeRobotPartitionedGraph &graph, const TEDCCIParams &params) {
  const std::vector<PoseKey> robot0 = {PoseKey{0, 0}, PoseKey{0, 1}};
  const std::vector<PoseKey> robot1 = {PoseKey{1, 2}, PoseKey{1, 3}};
  const std::vector<PoseKey> robot2 = {PoseKey{2, 4}, PoseKey{2, 5}};
  return {
      BoundarySelector::SelectBoundaryVariables(0, robot0, graph.local0,
                                                graph.shared01, params),
      BoundarySelector::SelectBoundaryVariables(1, robot1, graph.local1,
                                                graph.shared01, params),
      BoundarySelector::SelectBoundaryVariables(2, robot2, graph.local2, {},
                                                params),
  };
}

std::vector<TEDCCIPartition> makeThreeRobotPartitions(
    const ThreeRobotChainGraph &graph, const TEDCCIParams &params) {
  const std::vector<PoseKey> robot0 = {PoseKey{0, 0}, PoseKey{0, 1}};
  const std::vector<PoseKey> robot1 = {PoseKey{1, 2}, PoseKey{1, 3}};
  const std::vector<PoseKey> robot2 = {PoseKey{2, 4}, PoseKey{2, 5}};
  return {
      BoundarySelector::SelectBoundaryVariables(0, robot0, graph.local0,
                                                graph.shared_all, params),
      BoundarySelector::SelectBoundaryVariables(1, robot1, graph.local1,
                                                graph.shared_all, params),
      BoundarySelector::SelectBoundaryVariables(2, robot2, graph.local2,
                                                graph.shared_all, params),
  };
}

}  // namespace

TEST(testDPGO, TEDCCIOrchestratorFullyConnectedMatchesDirectSolver) {
  const TinyTwoRobotGraph graph = makeTinyTwoRobotGraph();
  TEDCCIParams params;
  params.use_measurement_weight = false;
  const auto partitions =
      makeTwoRobotPartitions(graph.local0, graph.local1, graph.shared, params);
  const Matrix direct = TEDCCISolver::InitializeSingleProcessDirect(
      graph.dimension, graph.num_poses, graph.all, partitions, params);

  InProcessAsyncFactorExchange exchange;
  TEDCCIOrchestrator robot0(0, params, &exchange);
  TEDCCIOrchestrator robot1(1, params, &exchange);
  robot0.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[0],
                            {0, 1});
  robot1.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[1],
                            {0, 1});
  robot0.SetLocalMeasurements(graph.local0, graph.shared);
  robot1.SetLocalMeasurements(graph.local1, {});

  robot0.StartEpoch(1);
  robot1.StartEpoch(1);
  pumpNetwork(&robot0, &robot1);

  ASSERT_TRUE(robot0.HasGlobalConsistentSolution());
  ASSERT_TRUE(robot1.HasGlobalConsistentSolution());
  expectPoseEstimateClose(robot0.GetGlobalConsistentInitialization(), direct,
                          graph.dimension, graph.num_poses);
  expectPoseEstimateClose(robot1.GetGlobalConsistentInitialization(), direct,
                          graph.dimension, graph.num_poses);
}

TEST(testDPGO, TEDCCIOrchestratorFallsBackDuringDisconnectThenRecovers) {
  const TinyTwoRobotGraph graph = makeTinyTwoRobotGraph();
  TEDCCIParams params;
  params.use_measurement_weight = false;
  const auto partitions =
      makeTwoRobotPartitions(graph.local0, graph.local1, graph.shared, params);
  const Matrix direct = TEDCCISolver::InitializeSingleProcessDirect(
      graph.dimension, graph.num_poses, graph.all, partitions, params);

  InProcessAsyncFactorExchange exchange(/*queue_when_link_down=*/true);
  exchange.MarkLinkDown(0, 1);
  TEDCCIOrchestrator robot0(0, params, &exchange);
  TEDCCIOrchestrator robot1(1, params, &exchange);
  robot0.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[0],
                            {0, 1});
  robot1.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[1],
                            {0, 1});
  robot0.SetLocalMeasurements(graph.local0, graph.shared);
  robot1.SetLocalMeasurements(graph.local1, {});

  robot0.StartEpoch(5);
  robot1.StartEpoch(5);
  pumpNetwork(&robot0, &robot1);

  EXPECT_TRUE(robot0.HasComponentConsistentSolution());
  EXPECT_TRUE(robot1.HasComponentConsistentSolution());
  EXPECT_FALSE(robot0.HasGlobalConsistentSolution());
  EXPECT_FALSE(robot1.HasGlobalConsistentSolution());
  EXPECT_GT(robot0.GetComponentConsistentInitialization().cols(), 0);
  EXPECT_GT(robot1.GetComponentConsistentInitialization().cols(), 0);

  exchange.MarkLinkUp(0, 1);
  pumpNetwork(&robot0, &robot1);

  ASSERT_TRUE(robot0.HasGlobalConsistentSolution());
  ASSERT_TRUE(robot1.HasGlobalConsistentSolution());
  expectPoseEstimateClose(robot0.GetGlobalConsistentInitialization(), direct,
                          graph.dimension, graph.num_poses);
  expectPoseEstimateClose(robot1.GetGlobalConsistentInitialization(), direct,
                          graph.dimension, graph.num_poses);
}

TEST(testDPGO, TEDCCIOrchestratorIgnoresStaleMessagesAfterGlobalSolve) {
  const TinyTwoRobotGraph graph = makeTinyTwoRobotGraph();
  TEDCCIParams params;
  params.use_measurement_weight = false;
  const auto partitions =
      makeTwoRobotPartitions(graph.local0, graph.local1, graph.shared, params);
  const Matrix direct = TEDCCISolver::InitializeSingleProcessDirect(
      graph.dimension, graph.num_poses, graph.all, partitions, params);

  InProcessAsyncFactorExchange exchange;
  TEDCCIOrchestrator robot0(0, params, &exchange);
  TEDCCIOrchestrator robot1(1, params, &exchange);
  robot0.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[0],
                            {0, 1});
  robot1.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[1],
                            {0, 1});
  robot0.SetLocalMeasurements(graph.local0, graph.shared);
  robot1.SetLocalMeasurements(graph.local1, {});

  robot0.StartEpoch(11);
  robot1.StartEpoch(11);
  pumpNetwork(&robot0, &robot1);
  ASSERT_TRUE(robot0.HasGlobalConsistentSolution());
  const Matrix before = robot0.GetGlobalConsistentInitialization();

  FactorMessage staleFactor;
  staleFactor.sender_robot_id = 1;
  staleFactor.receiver_robot_id = 0;
  staleFactor.epoch = 10;
  staleFactor.factor_type = FactorType::ROTATION;
  staleFactor.keys = partitions[1].boundary_poses;
  staleFactor.A = Matrix::Ones(1, static_cast<int>(staleFactor.keys.size()) *
                                      graph.dimension * graph.dimension);
  staleFactor.b = Vector::Constant(1, 1e6);
  staleFactor.checksum = ComputeFactorMessageChecksum(staleFactor);
  exchange.SendFactor(staleFactor);

  SolutionMessage staleSolution;
  staleSolution.sender_robot_id = 1;
  staleSolution.receiver_robot_id = 0;
  staleSolution.epoch = 9;
  staleSolution.factor_type = FactorType::TRANSLATION;
  staleSolution.keys = partitions[1].local_poses;
  staleSolution.x = Vector::Constant(
      static_cast<int>(staleSolution.keys.size()) * graph.dimension, -1e6);
  staleSolution.checksum = ComputeSolutionMessageChecksum(staleSolution);
  exchange.SendSolution(staleSolution);

  robot0.ProcessIncomingMessages();

  expectPoseEstimateClose(robot0.GetGlobalConsistentInitialization(), before,
                          graph.dimension, graph.num_poses);
  expectPoseEstimateClose(robot0.GetGlobalConsistentInitialization(), direct,
                          graph.dimension, graph.num_poses);
}

TEST(testDPGO, TEDCCIOrchestratorComputesReachableComponentDuringPartition) {
  const ThreeRobotPartitionedGraph graph = makeThreeRobotPartitionedGraph();
  TEDCCIParams params;
  params.use_measurement_weight = false;
  const auto partitions = makeThreeRobotPartitions(graph, params);
  const std::vector<TEDCCIPartition> componentPartitions = {
      partitions[0], partitions[1]};
  const Matrix componentDirect = TEDCCISolver::InitializeSingleProcessDirect(
      graph.dimension, 4, graph.component01, componentPartitions, params);

  InProcessAsyncFactorExchange exchange(/*queue_when_link_down=*/false);
  exchange.MarkLinkDown(0, 2);
  exchange.MarkLinkDown(1, 2);
  TEDCCIOrchestrator robot0(0, params, &exchange);
  TEDCCIOrchestrator robot1(1, params, &exchange);
  TEDCCIOrchestrator robot2(2, params, &exchange);
  robot0.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[0],
                            {0, 1, 2});
  robot1.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[1],
                            {0, 1, 2});
  robot2.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[2],
                            {0, 1, 2});
  robot0.SetLocalMeasurements(graph.local0, graph.shared01);
  robot1.SetLocalMeasurements(graph.local1, {});
  robot2.SetLocalMeasurements(graph.local2, {});

  robot0.StartEpoch(17);
  robot1.StartEpoch(17);
  robot2.StartEpoch(17);
  pumpNetwork({&robot0, &robot1, &robot2});

  ASSERT_TRUE(robot0.HasComponentConsistentSolution());
  ASSERT_TRUE(robot1.HasComponentConsistentSolution());
  EXPECT_FALSE(robot0.HasGlobalConsistentSolution());
  EXPECT_FALSE(robot1.HasGlobalConsistentSolution());
  {
    SCOPED_TRACE("robot0 component view");
    expectPoseEstimateClose(robot0.GetComponentConsistentInitialization(),
                            componentDirect, graph.dimension, 4);
  }
  {
    SCOPED_TRACE("robot1 component view");
    expectPoseEstimateClose(robot1.GetComponentConsistentInitialization(),
                            componentDirect, graph.dimension, 4);
  }
}

TEST(testDPGO, TEDCCIOrchestratorComponentSolveDoesNotPoisonRecoveredGlobal) {
  const ThreeRobotChainGraph graph = makeThreeRobotChainGraph();
  TEDCCIParams params;
  params.use_measurement_weight = false;
  const auto partitions = makeThreeRobotPartitions(graph, params);
  const std::vector<TEDCCIPartition> componentPartitions = {
      partitions[0], partitions[1]};
  const Matrix componentDirect = TEDCCISolver::InitializeSingleProcessDirect(
      graph.dimension, 4, graph.component01, componentPartitions, params);
  const Matrix globalDirect = TEDCCISolver::InitializeSingleProcessDirect(
      graph.dimension, graph.num_poses, graph.all, partitions, params);

  InProcessAsyncFactorExchange exchange(/*queue_when_link_down=*/true);
  exchange.MarkLinkDown(0, 2);
  exchange.MarkLinkDown(1, 2);
  TEDCCIOrchestrator robot0(0, params, &exchange);
  TEDCCIOrchestrator robot1(1, params, &exchange);
  TEDCCIOrchestrator robot2(2, params, &exchange);
  robot0.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[0],
                            {0, 1, 2});
  robot1.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[1],
                            {0, 1, 2});
  robot2.SetProblemMetadata(graph.dimension, graph.num_poses, partitions[2],
                            {0, 1, 2});
  robot0.SetLocalMeasurements(graph.local0, graph.shared01);
  robot1.SetLocalMeasurements(graph.local1, graph.shared12);
  robot2.SetLocalMeasurements(graph.local2, {});

  robot0.StartEpoch(23);
  robot1.StartEpoch(23);
  robot2.StartEpoch(23);
  pumpNetwork({&robot0, &robot1, &robot2});

  ASSERT_TRUE(robot0.HasComponentConsistentSolution());
  ASSERT_TRUE(robot1.HasComponentConsistentSolution());
  EXPECT_FALSE(robot0.HasGlobalConsistentSolution());
  EXPECT_FALSE(robot1.HasGlobalConsistentSolution());
  expectPoseEstimateClose(robot0.GetComponentConsistentInitialization(),
                          componentDirect, graph.dimension, 4);
  expectPoseEstimateClose(robot1.GetComponentConsistentInitialization(),
                          componentDirect, graph.dimension, 4);

  exchange.MarkLinkUp(0, 2);
  exchange.MarkLinkUp(1, 2);
  pumpNetwork({&robot0, &robot1, &robot2});

  ASSERT_TRUE(robot0.HasGlobalConsistentSolution());
  ASSERT_TRUE(robot1.HasGlobalConsistentSolution());
  ASSERT_TRUE(robot2.HasGlobalConsistentSolution());
  expectPoseEstimateClose(robot0.GetGlobalConsistentInitialization(),
                          globalDirect, graph.dimension, graph.num_poses);
  expectPoseEstimateClose(robot1.GetGlobalConsistentInitialization(),
                          globalDirect, graph.dimension, graph.num_poses);
  expectPoseEstimateClose(robot2.GetGlobalConsistentInitialization(),
                          globalDirect, graph.dimension, graph.num_poses);
}

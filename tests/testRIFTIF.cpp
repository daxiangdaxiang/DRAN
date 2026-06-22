#include <DPGO/DPGO_utils.h>
#include <DPGO/RIFTIF.h>
#include <DPGO/TEDCCI.h>

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

Matrix deterministicMatrix(int rows, int cols, double phase) {
  Matrix A(rows, cols);
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      A(r, c) =
          std::sin(phase + 0.11 * static_cast<double>((r + 1) * (c + 2))) +
          0.2 * std::cos(0.17 * static_cast<double>(r + c + 1));
    }
  }
  return A;
}

Vector deterministicVector(int rows, double phase) {
  Vector b(rows);
  for (int r = 0; r < rows; ++r) {
    b(r) = std::cos(phase + 0.23 * static_cast<double>(r + 1)) +
           0.1 * std::sin(0.31 * static_cast<double>(r + 2));
  }
  return b;
}

CondensedFactor makeCondensed(int robot, const std::vector<PoseKey> &keys,
                              int rows, int blockDim, double phase) {
  CondensedFactor factor;
  factor.owner_robot_id = robot;
  factor.factor_type = FactorType::TRANSLATION;
  factor.boundary_keys = keys;
  factor.Abar =
      deterministicMatrix(rows, static_cast<int>(keys.size()) * blockDim,
                          phase);
  factor.bbar = deterministicVector(rows, phase);
  return factor;
}

LinearFactorBlock makeCross(const std::vector<PoseKey> &keys, int rows,
                            int blockDim, double phase) {
  LinearFactorBlock factor;
  factor.keys = keys;
  factor.A =
      deterministicMatrix(rows, static_cast<int>(keys.size()) * blockDim,
                          phase);
  factor.b = deterministicVector(rows, phase);
  return factor;
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
  double maxDiff = 0.0;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    maxDiff = std::max(
        maxDiff,
        (actual.block(0, pose * (d + 1), d, d + 1) -
         expected.block(0, pose * (d + 1), d, d + 1))
            .norm());
  }
  EXPECT_LT(maxDiff, tolerance);
}

}  // namespace

TEST(testDPGO, RIFTIFBuildsInterfaceProblemFromTEDFactors) {
  const int blockDim = 2;
  const PoseKey a{0, 1};
  const PoseKey b{1, 2};
  const PoseKey c{2, 3};
  const auto c0 = makeCondensed(0, {a, b}, 5, blockDim, 0.1);
  const auto c1 = makeCondensed(1, {b, c}, 6, blockDim, 0.7);
  const auto cross = makeCross({a, c}, 4, blockDim, 1.3);

  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::TRANSLATION, 2, blockDim, {c0, c1}, {cross},
          /*use_rotation_multi_rhs=*/false);

  ASSERT_EQ(problem.factors.size(), 3u);
  EXPECT_EQ(problem.variables.size(), 3u);
  EXPECT_EQ(problem.block_dim, blockDim);
  EXPECT_EQ(problem.rhs_dim, 1);
  EXPECT_EQ(problem.factors[0].owner_robot, 0);
  EXPECT_EQ(problem.factors[2].A.rows(), cross.A.rows());
}

TEST(testDPGO, RIFTIFCliqueTreeCoversFactorsAndRunningIntersection) {
  const int blockDim = 1;
  const PoseKey a{0, 1};
  const PoseKey b{1, 2};
  const PoseKey c{2, 3};
  const PoseKey d{3, 4};
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::TRANSLATION, 2, blockDim,
          {makeCondensed(0, {a, b}, 3, blockDim, 0.1),
           makeCondensed(1, {b, c}, 3, blockDim, 0.3),
           makeCondensed(2, {c, d}, 3, blockDim, 0.5)},
          {makeCross({a, d}, 2, blockDim, 0.8)},
          /*use_rotation_multi_rhs=*/false);

  RIFTParams params;
  const InterfaceCliqueTree tree = InterfaceCliqueTreeBuilder::Build(
      problem, params);

  EXPECT_FALSE(tree.cliques.empty());
  EXPECT_TRUE(InterfaceCliqueTreeBuilder::VerifyRunningIntersection(tree));
  for (const auto &factor : problem.factors) {
    bool covered = false;
    for (const auto &clique : tree.cliques) {
      bool contains = true;
      for (const auto &key : factor.scope) {
        contains = contains &&
                   std::binary_search(clique.variables.begin(),
                                      clique.variables.end(), key);
      }
      covered = covered || contains;
    }
    EXPECT_TRUE(covered);
  }
}

TEST(testDPGO, RIFTIFRootlessSchedulerCompletesAllDirectedMessages) {
  InterfaceCliqueTree tree;
  tree.cliques.resize(3);
  for (int i = 0; i < 3; ++i) {
    tree.cliques[i].id = i;
  }
  tree.cliques[0].neighbors = {1};
  tree.cliques[1].neighbors = {0, 2};
  tree.cliques[2].neighbors = {1};

  RIFTRootlessScheduler scheduler;
  scheduler.Initialize(tree);
  while (!scheduler.AllDirectedMessagesComplete()) {
    const auto ready = scheduler.ReadyOutgoingMessages();
    ASSERT_FALSE(ready.empty());
    for (const auto &edge : ready) {
      RIFTMessage msg;
      msg.header.src = edge.src;
      msg.header.dst = edge.dst;
      scheduler.MarkMessageSent(edge.src, edge.dst);
      scheduler.OnMessageReceived(msg);
    }
  }
  EXPECT_TRUE(scheduler.CliqueBeliefReady(0));
  EXPECT_TRUE(scheduler.CliqueBeliefReady(1));
  EXPECT_TRUE(scheduler.CliqueBeliefReady(2));
}

TEST(testDPGO, RIFTIFExactMatchesDirectOracleOnSmallInterface) {
  const int blockDim = 2;
  const PoseKey a{0, 1};
  const PoseKey b{1, 2};
  const PoseKey c{2, 3};
  const auto c0 = makeCondensed(0, {a, b}, 8, blockDim, 0.1);
  const auto c1 = makeCondensed(1, {b, c}, 8, blockDim, 0.7);
  const auto cross = makeCross({c, a}, 5, blockDim, 1.3);

  TEDCCIParams tedParams;
  const Vector direct =
      InterfaceDirectSolver::Solve({c0, c1}, {cross}, blockDim, tedParams);

  RIFTParams riftParams;
  riftParams.forbid_global_interface_matrix = false;
  riftParams.forbid_direct_solver_in_deployment = false;
  riftParams.forbid_collectives = false;
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::TRANSLATION, 2, blockDim, {c0, c1}, {cross},
          /*use_rotation_multi_rhs=*/false);
  const InterfaceCliqueTree tree =
      InterfaceCliqueTreeBuilder::Build(problem, riftParams);
  DecentralizationGuard guard;
  RIFTStats stats;
  const Vector rift =
      RIFTExactSolver::Solve(problem, tree, riftParams, &stats, &guard);

  EXPECT_LT((rift - direct).norm(), 1e-8);
  EXPECT_EQ(stats.selected_backend, RIFTInterfaceBackend::RIFT_EXACT);
  EXPECT_FALSE(stats.used_direct_solver);
  EXPECT_FALSE(stats.used_global_matrix);
  EXPECT_FALSE(stats.used_collective);
  EXPECT_EQ(stats.directed_messages_sent,
            2 * static_cast<int>(tree.edges.size()));
}

TEST(testDPGO, RIFTIFExactPreservesRankDeficientEliminationResidual) {
  const int blockDim = 1;
  const PoseKey a{0, 1};
  const PoseKey b{1, 2};
  const PoseKey c{2, 3};

  CondensedFactor f0;
  f0.owner_robot_id = 0;
  f0.factor_type = FactorType::TRANSLATION;
  f0.boundary_keys = {a, b};
  f0.Abar.resize(1, 2);
  f0.Abar << 0.0, 1.0;
  f0.bbar.resize(1);
  f0.bbar << 1.0;

  CondensedFactor f1;
  f1.owner_robot_id = 1;
  f1.factor_type = FactorType::TRANSLATION;
  f1.boundary_keys = {b, c};
  f1.Abar.resize(2, 2);
  f1.Abar << 1.0, 0.0,
              0.0, 1.0;
  f1.bbar.resize(2);
  f1.bbar << 1.0, 2.0;

  TEDCCIParams tedParams;
  const Vector direct =
      InterfaceDirectSolver::Solve({f0, f1}, {}, blockDim, tedParams);

  RIFTParams riftParams;
  riftParams.forbid_global_interface_matrix = false;
  riftParams.forbid_direct_solver_in_deployment = false;
  riftParams.forbid_collectives = false;
  const InterfaceProblem problem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::TRANSLATION, 2, blockDim, {f0, f1}, {},
          /*use_rotation_multi_rhs=*/false);
  const InterfaceCliqueTree tree =
      InterfaceCliqueTreeBuilder::Build(problem, riftParams);
  RIFTStats stats;
  const Vector rift =
      RIFTExactSolver::Solve(problem, tree, riftParams, &stats);

  EXPECT_LT((rift - direct).norm(), 1e-10);
  EXPECT_NEAR(rift(1), 1.0, 1e-10);
  EXPECT_NEAR(rift(2), 2.0, 1e-10);
}

TEST(testDPGO, RIFTIFGuardDetectsDirectSolverUse) {
  DecentralizationGuard guard;
  guard.RecordDirectInterfaceSolverCall();
  EXPECT_THROW(guard.AssertNoDirectSolverCalled(), std::runtime_error);
}

TEST(testDPGO, RIFTIFRejectsUnimplementedMultiRHSAndFallbackBackends) {
  const int blockDim = 4;
  const PoseKey a{0, 1};
  CondensedFactor factor;
  factor.owner_robot_id = 0;
  factor.factor_type = FactorType::ROTATION;
  factor.boundary_keys = {a};
  factor.Abar = Matrix::Identity(blockDim, blockDim);
  factor.bbar = Vector::Ones(blockDim);

  EXPECT_THROW(InterfaceProblemBuilder::BuildFromTEDFactors(
                   FactorType::ROTATION, 2, blockDim, {factor}, {},
                   /*use_rotation_multi_rhs=*/true),
               std::invalid_argument);

  TEDCCIParams params;
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::RIFT_CAK;
  TEDCCIStats stats;
  EXPECT_THROW(SolveTEDInterfaceWithRIFTExact({factor}, {}, blockDim, {},
                                             params, &stats),
               std::invalid_argument);
}

TEST(testDPGO, TEDCCIRIFTIFMatchesCentralized2DChainSplit) {
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
  params.mode = CCIInitMode::TED_CCI_RIFT_IF;
  params.rift_interface_backend = RIFTInterfaceBackend::RIFT_EXACT;
  params.use_measurement_weight = false;
  const auto partitions = makePartitions(numPoses, robots, measurements,
                                         params);
  TEDCCIStats stats;
  const Matrix rift = TEDCCISolver::Initialize(
      static_cast<int>(d), static_cast<int>(numPoses), measurements,
      partitions, params, &stats);
  const Matrix centralized = chordalInitialization(d, numPoses, measurements);

  expectPoseEstimateClose(rift, centralized, d, numPoses, 1e-8);
  EXPECT_EQ(stats.effective_rotation_mode, CCIInitMode::TED_CCI_RIFT_IF);
  EXPECT_EQ(stats.effective_translation_mode, CCIInitMode::TED_CCI_RIFT_IF);
  EXPECT_EQ(stats.rift_selected_backend, RIFTInterfaceBackend::RIFT_EXACT);
  EXPECT_FALSE(stats.rift_used_direct_solver);
  EXPECT_FALSE(stats.rift_used_global_matrix);
  EXPECT_FALSE(stats.rift_used_collective);
}

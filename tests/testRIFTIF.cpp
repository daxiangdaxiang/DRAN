#include <DPGO/DPGO_utils.h>
#include <DPGO/RIFTIF.h>
#include <DPGO/TEDCCI.h>

#include <Eigen/Geometry>

#include <algorithm>
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

Matrix rotation3(double yaw, double pitch, double roll) {
  const Eigen::Matrix3d R =
      (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
       Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
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

Vector vectorizeForTest(const Matrix &M) {
  return Eigen::Map<const Vector>(M.data(), M.rows() * M.cols());
}

Vector flattenMultiRhsResidualForVectorizedOrder(const Matrix &residual,
                                                 int dimension) {
  Vector flattened(residual.rows() * dimension);
  for (int rhs = 0; rhs < dimension; ++rhs) {
    for (int row = 0; row < residual.rows(); ++row) {
      flattened(rhs + row * dimension) = residual(row, rhs);
    }
  }
  return flattened;
}

std::size_t estimatedTreeBytes(const InterfaceCliqueTree &tree) {
  std::size_t bytes = 0;
  for (const auto &edge : tree.edges) {
    bytes += 2u * edge.estimated_message_bytes;
  }
  return bytes;
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

TEST(testDPGO, RIFTIFRotationMultiRHSFactorMatchesVectorizedResiduals) {
  for (const int d : {2, 3}) {
    TEDCCIParams params;
    params.anchor_pose_id = 99;
    params.use_measurement_weight = false;
    const std::vector<int> owner = {0, 0, 1};
    const Matrix R0 = Matrix::Identity(d, d);
    const Matrix R1 = d == 2 ? rotation2(0.31) : rotation3(0.31, -0.08, 0.14);
    const Matrix R2 = d == 2 ? rotation2(0.72) : rotation3(0.72, 0.16, -0.22);
    Vector t0 = Vector::Zero(d);
    Vector t1 = Vector::Zero(d);
    Vector t2 = Vector::Zero(d);
    for (int i = 0; i < d; ++i) {
      t1(i) = 0.2 + 0.1 * i;
      t2(i) = 0.6 - 0.05 * i;
    }
    const RelativeSEMeasurement measurement =
        makeMeasurement(1, 2, owner, R1, t1, R2, t2, 2.0, 3.0);
    const LinearFactorBlock vectorized =
        CCIFactorBuilder::BuildRotationFactors(d, {measurement}, params)
            .front();
    const InterfaceFactor multi =
        BuildRotationMultiRHSFactorFromVectorized(
            0, vectorized.keys, vectorized.A, vectorized.b, d, 0);

    Vector xVec(static_cast<int>(vectorized.keys.size()) * d * d);
    Matrix yMulti(static_cast<int>(vectorized.keys.size()) * d, d);
    for (std::size_t keyIndex = 0; keyIndex < vectorized.keys.size();
         ++keyIndex) {
      const Matrix X = deterministicMatrix(d, d, 0.25 + keyIndex);
      xVec.segment(static_cast<int>(keyIndex) * d * d, d * d) =
          vectorizeForTest(X);
      yMulti.block(static_cast<int>(keyIndex) * d, 0, d, d) =
          X.transpose();
    }

    const Vector residualVec = vectorized.A * xVec - vectorized.b;
    const Matrix residualMulti = multi.A * yMulti - multi.B;
    const Vector residualMultiVec =
        flattenMultiRhsResidualForVectorizedOrder(residualMulti, d);
    EXPECT_LT((residualVec - residualMultiVec).norm(), 1e-12);
    EXPECT_EQ(multi.A.cols(),
              static_cast<int>(vectorized.keys.size()) * d);
    EXPECT_EQ(multi.B.cols(), d);
  }
}

TEST(testDPGO, RIFTIFRotationMultiRHSRejectsOffRowCoupling) {
  const int d = 3;
  TEDCCIParams params;
  params.anchor_pose_id = 99;
  params.use_measurement_weight = false;
  const std::vector<int> owner = {0, 1};
  Vector t0 = Vector::Zero(d);
  Vector t1 = Vector::Zero(d);
  const RelativeSEMeasurement measurement =
      makeMeasurement(0, 1, owner, Matrix::Identity(d, d), t0,
                      rotation3(0.2, 0.1, -0.1), t1, 2.0, 3.0);
  LinearFactorBlock vectorized =
      CCIFactorBuilder::BuildRotationFactors(d, {measurement}, params)
          .front();
  vectorized.A(0, 1) = 0.5;
  EXPECT_THROW(BuildRotationMultiRHSFactorFromVectorized(
                   0, vectorized.keys, vectorized.A, vectorized.b, d, 0),
               std::invalid_argument);
}

TEST(testDPGO, RIFTIFRotationMultiRHSPayloadEstimateIsSmallerIn3D) {
  const int d = 3;
  TEDCCIParams params;
  params.anchor_pose_id = 99;
  params.use_measurement_weight = false;
  const std::vector<int> owner = {0, 0, 1, 1};
  std::vector<Matrix> rotations = {
      Matrix::Identity(d, d), rotation3(0.2, 0.1, -0.1),
      rotation3(0.45, -0.2, 0.05), rotation3(0.7, 0.15, 0.2)};
  std::vector<Vector> translations(4, Vector::Zero(d));
  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < rotations.size(); ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0, 3.0));
  }
  measurements.push_back(makeMeasurement(0, 2, owner, rotations[0],
                                         translations[0], rotations[2],
                                         translations[2], 2.0, 3.0));
  const auto vectorFactors =
      CCIFactorBuilder::BuildRotationFactors(d, measurements, params);

  const InterfaceProblem vectorProblem =
      InterfaceProblemBuilder::BuildFromTEDFactors(
          FactorType::ROTATION, d, d * d, {}, vectorFactors,
          /*use_rotation_multi_rhs=*/false);
  std::vector<InterfaceFactor> multiFactors;
  RIFTFactorId nextId = 0;
  for (const auto &factor : vectorFactors) {
    multiFactors.push_back(BuildRotationMultiRHSFactorFromVectorized(
        nextId++, factor.keys, factor.A, factor.b, d,
        factor.keys.empty() ? -1 : factor.keys.front().robot_id));
  }
  const InterfaceProblem multiProblem =
      InterfaceProblemBuilder::BuildFromInterfaceFactors(
          FactorType::ROTATION, d, d, d, multiFactors);
  RIFTParams riftParams;
  const InterfaceCliqueTree vectorTree =
      InterfaceCliqueTreeBuilder::Build(vectorProblem, riftParams);
  const InterfaceCliqueTree multiTree =
      InterfaceCliqueTreeBuilder::Build(multiProblem, riftParams);

  EXPECT_LT(estimatedTreeBytes(multiTree), estimatedTreeBytes(vectorTree));
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
  for (const auto &clique : tree.cliques) {
    EXPECT_GE(clique.host_robot, 0);
    std::map<int, int> counts;
    for (const auto &key : clique.variables) {
      ++counts[key.robot_id];
    }
    ASSERT_FALSE(counts.empty());
    int expectedHost = counts.begin()->first;
    int expectedCount = counts.begin()->second;
    for (const auto &entry : counts) {
      if (entry.second > expectedCount ||
          (entry.second == expectedCount && entry.first < expectedHost)) {
        expectedHost = entry.first;
        expectedCount = entry.second;
      }
    }
    EXPECT_EQ(clique.host_robot, expectedHost);
  }
  for (const auto &edge : tree.edges) {
    ASSERT_GE(edge.a, 0);
    ASSERT_GE(edge.b, 0);
    ASSERT_LT(static_cast<std::size_t>(edge.a), tree.cliques.size());
    ASSERT_LT(static_cast<std::size_t>(edge.b), tree.cliques.size());
    EXPECT_GE(tree.cliques[edge.a].host_robot, 0);
    EXPECT_GE(tree.cliques[edge.b].host_robot, 0);
    EXPECT_GT(edge.estimated_message_bytes, 0u);
  }
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
  EXPECT_GT(stats.num_host_robots, 0);
  EXPECT_GT(stats.max_host_clique_load, 0);
  EXPECT_GE(stats.cross_host_tree_edges, 0);
  EXPECT_EQ(stats.estimated_route_hops,
            2 * stats.cross_host_tree_edges);
  EXPECT_LE(stats.estimated_routed_message_bytes,
            stats.estimated_message_bytes);
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

TEST(testDPGO, RIFTIFRejectsUnimplementedFallbackBackends) {
  const int blockDim = 4;
  const PoseKey a{0, 1};
  CondensedFactor factor;
  factor.owner_robot_id = 0;
  factor.factor_type = FactorType::ROTATION;
  factor.boundary_keys = {a};
  factor.Abar = Matrix::Identity(blockDim, blockDim);
  factor.bbar = Vector::Ones(blockDim);

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
  EXPECT_TRUE(params.rift_use_rotation_multi_rhs);
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

TEST(testDPGO, TEDCCIRIFTIFMultiRHSMatchesVectorizedRIFT3DChainSplit) {
  const size_t d = 3;
  const size_t numPoses = 5;
  const int robots = 2;
  const std::vector<int> owner = contiguousOwner(numPoses, robots);
  const std::vector<Matrix> rotations = {
      Matrix::Identity(d, d), rotation3(0.18, 0.04, -0.05),
      rotation3(0.39, -0.06, 0.08), rotation3(0.61, 0.09, -0.12),
      rotation3(0.83, -0.11, 0.16)};
  std::vector<Vector> translations(numPoses, Vector::Zero(d));
  translations[1] << 0.6, 0.1, -0.1;
  translations[2] << 1.3, 0.3, 0.2;
  translations[3] << 2.1, 0.5, 0.4;
  translations[4] << 2.8, 0.9, 0.5;

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < numPoses; ++pose) {
    measurements.push_back(makeMeasurement(
        pose, pose + 1, owner, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.5 + pose,
        3.5 + 0.25 * pose));
  }
  measurements.push_back(makeMeasurement(0, 3, owner, rotations[0],
                                         translations[0], rotations[3],
                                         translations[3], 4.0, 2.5));
  measurements.push_back(makeMeasurement(1, 4, owner, rotations[1],
                                         translations[1], rotations[4],
                                         translations[4], 3.0, 2.0));

  TEDCCIParams vectorParams;
  vectorParams.mode = CCIInitMode::TED_CCI_RIFT_IF;
  vectorParams.rift_interface_backend = RIFTInterfaceBackend::RIFT_EXACT;
  vectorParams.rift_use_rotation_multi_rhs = false;
  vectorParams.use_measurement_weight = false;
  const auto partitions = makePartitions(numPoses, robots, measurements,
                                         vectorParams);
  TEDCCIStats vectorStats;
  const Matrix vectorizedRift = TEDCCISolver::Initialize(
      static_cast<int>(d), static_cast<int>(numPoses), measurements,
      partitions, vectorParams, &vectorStats);

  TEDCCIParams multiParams = vectorParams;
  multiParams.rift_use_rotation_multi_rhs = true;
  TEDCCIStats multiStats;
  const Matrix multiRift = TEDCCISolver::Initialize(
      static_cast<int>(d), static_cast<int>(numPoses), measurements,
      partitions, multiParams, &multiStats);
  const Matrix centralized = chordalInitialization(d, numPoses, measurements);

  expectPoseEstimateClose(multiRift, vectorizedRift, d, numPoses, 1e-8);
  expectPoseEstimateClose(multiRift, centralized, d, numPoses, 1e-8);
  EXPECT_LT(multiStats.rift_estimated_message_bytes,
            vectorStats.rift_estimated_message_bytes);
  EXPECT_FALSE(multiStats.rift_used_direct_solver);
  EXPECT_FALSE(multiStats.rift_used_global_matrix);
  EXPECT_FALSE(multiStats.rift_used_collective);
}

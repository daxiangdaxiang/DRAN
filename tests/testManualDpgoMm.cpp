#include <DPGO/ManualDpgoMm.h>

#include <DPGO/DPGO_utils.h>
#include <DPGO/PGOAgent.h>

#include "gtest/gtest.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace DPGO;

namespace {

struct PartitionedTinyGrid {
  std::vector<RelativeSEMeasurement> dataset;
  unsigned numPoses{0};
  unsigned d{0};
  unsigned r{0};
  unsigned posesPerRobot{0};
  std::vector<std::vector<RelativeSEMeasurement>> odometry;
  std::vector<std::vector<RelativeSEMeasurement>> privateLoops;
  std::vector<std::vector<RelativeSEMeasurement>> sharedLoops;
  Matrix xChordal;
};

SparseMatrix paddedSparseMatrix(const SparseMatrix &input, std::size_t rows,
                                std::size_t cols) {
  SparseMatrix result(rows, cols);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<std::size_t>(input.nonZeros()));
  for (int k = 0; k < input.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(input, k); it; ++it) {
      triplets.emplace_back(it.row(), it.col(), it.value());
    }
  }
  result.setFromTriplets(triplets.begin(), triplets.end());
  result.makeCompressed();
  return result;
}

std::vector<std::string> splitCsvLine(const std::string &line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

std::map<std::string, std::string>
csvRowByHeader(const std::vector<std::string> &header,
               const std::vector<std::string> &row) {
  std::map<std::string, std::string> values;
  const std::size_t count = std::min(header.size(), row.size());
  for (std::size_t i = 0; i < count; ++i) {
    values[header[i]] = row[i];
  }
  return values;
}

std::string writeTinySe2Dataset(const std::string &prefix) {
  const std::string path =
      std::string("/tmp/") + prefix + "_" +
      std::to_string(static_cast<unsigned long long>(std::rand())) + ".g2o";
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write test dataset: " + path);
  }
  output << "EDGE_SE2 0 1 1.0 0.0 0.00 100 0 0 100 0 100\n";
  output << "EDGE_SE2 1 2 1.0 0.0 0.03 100 0 0 100 0 100\n";
  output << "EDGE_SE2 2 3 1.0 0.0 0.08 100 0 0 100 0 100\n";
  output << "EDGE_SE2 3 4 1.0 0.0 -0.04 100 0 0 100 0 100\n";
  output << "EDGE_SE2 4 5 1.0 0.0 0.02 100 0 0 100 0 100\n";
  output << "EDGE_SE2 1 4 3.0 0.1 0.05 50 0 0 50 0 50\n";
  return path;
}

std::string writeTinySe2DatasetWithFixedRecords(const std::string &prefix) {
  const std::string path =
      std::string("/tmp/") + prefix + "_" +
      std::to_string(static_cast<unsigned long long>(std::rand())) + ".g2o";
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write test dataset: " + path);
  }
  output << "VERTEX_SE2 0 0.0 0.0 0.0\n";
  output << "FIX 0\n";
  output << "VERTEX_SE2 1 1.0 0.0 0.0\n";
  output << "FIX! 1\n";
  output << "EDGE_SE2 0 1 1.0 0.0 0.0 100 0 0 100 0 100\n";
  return path;
}

std::string writeCyclicRobotGaugeSe2Dataset(const std::string &prefix) {
  const std::string path =
      std::string("/tmp/") + prefix + "_" +
      std::to_string(static_cast<unsigned long long>(std::rand())) + ".g2o";
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write test dataset: " + path);
  }
  output << "EDGE_SE2 0 1 1.0 0.0 0.0 100 0 0 100 0 100\n";
  output << "EDGE_SE2 2 3 1.0 0.0 0.0 100 0 0 100 0 100\n";
  output << "EDGE_SE2 4 5 1.0 0.0 0.0 100 0 0 100 0 100\n";
  output << "EDGE_SE2 1 2 12.0 0.0 0.0 100 0 0 100 0 100\n";
  output << "EDGE_SE2 3 4 9.0 0.0 0.0 100 0 0 100 0 100\n";
  output << "EDGE_SE2 5 0 -21.0 0.0 0.0 100 0 0 100 0 100\n";
  return path;
}

std::string writeMatrixTextFile(const std::string &prefix,
                                const Matrix &matrix) {
  const std::string path =
      std::string("/tmp/") + prefix + "_" +
      std::to_string(static_cast<unsigned long long>(std::rand())) + ".txt";
  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("Unable to write test matrix: " + path);
  }
  output << std::setprecision(16) << matrix << "\n";
  return path;
}

ManualDpgoMmOptions tinySe2LiftedDeltaOptions() {
  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.printIterationSummary = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  return options;
}

void expectSparseNear(const SparseMatrix &actual,
                      const SparseMatrix &expected, double tol) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  SparseMatrix diff = actual - expected;
  diff.prune(tol);
  EXPECT_EQ(diff.nonZeros(), 0) << Matrix(diff);
}

PartitionedTinyGrid buildPartitionedTinyGrid(
    const std::string &datasetPath, const ManualDpgoMmOptions &options) {
  PartitionedTinyGrid data;
  size_t numPosesSize = 0;
  data.dataset = read_g2o_file(datasetPath, numPosesSize);
  data.numPoses = static_cast<unsigned>(numPosesSize);
  data.d = static_cast<unsigned>(data.dataset.front().t.rows());
  data.r = data.d;
  data.posesPerRobot = data.numPoses / options.numRobots;

  std::map<unsigned, PoseID> poseMap;
  for (unsigned robot = 0; robot < options.numRobots; ++robot) {
    const unsigned startIdx = robot * data.posesPerRobot;
    unsigned endIdx = (robot + 1) * data.posesPerRobot;
    if (robot + 1 == options.numRobots) {
      endIdx = data.numPoses;
    }
    for (unsigned idx = startIdx; idx < endIdx; ++idx) {
      poseMap[idx] = std::make_pair(robot, idx - startIdx);
    }
  }

  data.odometry.resize(options.numRobots);
  data.privateLoops.resize(options.numRobots);
  data.sharedLoops.resize(options.numRobots);
  for (const auto &mIn : data.dataset) {
    const PoseID src = poseMap.at(mIn.p1);
    const PoseID dst = poseMap.at(mIn.p2);
    RelativeSEMeasurement m(src.first, dst.first, src.second, dst.second,
                            mIn.R, mIn.t, mIn.kappa, mIn.tau);
    if (src.first == dst.first) {
      if (src.second + 1 == dst.second) {
        data.odometry[src.first].push_back(m);
      } else {
        data.privateLoops[src.first].push_back(m);
      }
    } else {
      data.sharedLoops[src.first].push_back(m);
      data.sharedLoops[dst.first].push_back(m);
    }
  }

  const Matrix tChordal =
      chordalInitialization(data.d, data.numPoses, data.dataset);
  data.xChordal = fixedStiefelVariable(data.d, data.r) * tChordal;
  return data;
}

SparseMatrix referenceManualLocalQ(const PartitionedTinyGrid &data,
                                   unsigned robot,
                                   unsigned localPoseCount) {
  std::vector<RelativeSEMeasurement> privateMeasurements =
      data.odometry[robot];
  privateMeasurements.insert(privateMeasurements.end(),
                             data.privateLoops[robot].begin(),
                             data.privateLoops[robot].end());

  const std::size_t blockDim = data.d + 1;
  SparseMatrix q = paddedSparseMatrix(
      constructConnectionLaplacianSE(privateMeasurements),
      localPoseCount * blockDim, localPoseCount * blockDim);

  for (const auto &m : data.sharedLoops[robot]) {
    Matrix T = Matrix::Zero(data.d + 1, data.d + 1);
    T.block(0, 0, data.d, data.d) = m.R;
    T.block(0, data.d, data.d, 1) = m.t;
    T(data.d, data.d) = 1.0;

    Matrix Omega = Matrix::Zero(data.d + 1, data.d + 1);
    for (unsigned row = 0; row < data.d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(data.d, data.d) = m.weight * m.tau;

    std::size_t idx = 0;
    Matrix W;
    if (m.r1 == robot) {
      idx = m.p1;
      W = T * Omega * T.transpose();
    } else {
      idx = m.p2;
      W = Omega;
    }
    for (std::size_t col = 0; col < blockDim; ++col) {
      for (std::size_t row = 0; row < blockDim; ++row) {
        q.coeffRef(idx * blockDim + row, idx * blockDim + col) += W(row, col);
      }
    }
  }
  q.makeCompressed();
  return q;
}

SparseMatrix referenceManualLocalG(const PartitionedTinyGrid &data,
                                   const ManualDpgoMmOptions &options,
                                   unsigned robot,
                                   unsigned localPoseCount) {
  SparseMatrix g(data.r, (data.d + 1) * localPoseCount);
  for (const auto &m : data.sharedLoops[robot]) {
    Matrix T = Matrix::Zero(data.d + 1, data.d + 1);
    T.block(0, 0, data.d, data.d) = m.R;
    T.block(0, data.d, data.d, 1) = m.t;
    T(data.d, data.d) = 1.0;

    Matrix Omega = Matrix::Zero(data.d + 1, data.d + 1);
    for (unsigned row = 0; row < data.d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(data.d, data.d) = m.weight * m.tau;

    Matrix neighborPose;
    std::size_t idx = 0;
    Matrix L;
    if (m.r1 == robot) {
      const unsigned neighborStart = m.r2 * data.posesPerRobot;
      neighborPose = data.xChordal.block(
          0, (neighborStart + m.p2) * (data.d + 1), data.r, data.d + 1);
      idx = m.p1;
      L = -neighborPose * Omega * T.transpose();
    } else {
      const unsigned neighborStart = m.r1 * data.posesPerRobot;
      neighborPose = data.xChordal.block(
          0, (neighborStart + m.p1) * (data.d + 1), data.r, data.d + 1);
      idx = m.p2;
      L = -neighborPose * T * Omega;
    }

    for (std::size_t col = 0; col < data.d + 1; ++col) {
      for (std::size_t row = 0; row < data.r; ++row) {
        g.coeffRef(row, idx * (data.d + 1) + col) += L(row, col);
      }
    }
  }
  g.makeCompressed();
  (void)options;
  return g;
}

std::vector<std::unique_ptr<PGOAgent>> buildReferenceAgents(
    const std::string &datasetPath, const ManualDpgoMmOptions &options) {
  size_t numPosesSize = 0;
  const std::vector<RelativeSEMeasurement> dataset =
      read_g2o_file(datasetPath, numPosesSize);
  const unsigned numPoses = static_cast<unsigned>(numPosesSize);
  const unsigned d = static_cast<unsigned>(dataset.front().t.rows());
  const unsigned r = d;
  const unsigned posesPerRobot = numPoses / options.numRobots;

  std::map<unsigned, PoseID> poseMap;
  for (unsigned robot = 0; robot < options.numRobots; ++robot) {
    const unsigned startIdx = robot * posesPerRobot;
    unsigned endIdx = (robot + 1) * posesPerRobot;
    if (robot + 1 == options.numRobots) {
      endIdx = numPoses;
    }
    for (unsigned idx = startIdx; idx < endIdx; ++idx) {
      poseMap[idx] = std::make_pair(robot, idx - startIdx);
    }
  }

  std::vector<std::vector<RelativeSEMeasurement>> odometry(options.numRobots);
  std::vector<std::vector<RelativeSEMeasurement>> privateLoops(
      options.numRobots);
  std::vector<std::vector<RelativeSEMeasurement>> sharedLoops(
      options.numRobots);
  for (const auto &mIn : dataset) {
    const PoseID src = poseMap.at(mIn.p1);
    const PoseID dst = poseMap.at(mIn.p2);
    RelativeSEMeasurement m(src.first, dst.first, src.second, dst.second,
                            mIn.R, mIn.t, mIn.kappa, mIn.tau);
    if (src.first == dst.first) {
      if (src.second + 1 == dst.second) {
        odometry[src.first].push_back(m);
      } else {
        privateLoops[src.first].push_back(m);
      }
    } else {
      sharedLoops[src.first].push_back(m);
      sharedLoops[dst.first].push_back(m);
    }
  }

  std::vector<std::unique_ptr<PGOAgent>> agents;
  agents.reserve(options.numRobots);
  for (unsigned robot = 0; robot < options.numRobots; ++robot) {
    PGOAgentParameters params(d, r, options.numRobots);
    params.acceleration = false;
    params.verbose = false;
    auto agent = std::make_unique<PGOAgent>(robot, params);
    agent->setPoseGraph(odometry[robot], privateLoops[robot],
                        sharedLoops[robot]);
    agents.push_back(std::move(agent));
  }

  const Matrix TChordal = chordalInitialization(d, numPoses, dataset);
  const Matrix XChordal = fixedStiefelVariable(d, r) * TChordal;
  for (unsigned robot = 0; robot < options.numRobots; ++robot) {
    const unsigned startIdx = robot * posesPerRobot;
    unsigned endIdx = (robot + 1) * posesPerRobot;
    if (robot + 1 == options.numRobots) {
      endIdx = numPoses;
    }
    agents[robot]->setX(XChordal.block(
        0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)));
  }

  for (auto &receiver : agents) {
    for (unsigned senderID : receiver->getNeighbors()) {
      PoseDict poses;
      for (unsigned poseIndex : receiver->getNeighborPublicPoses(senderID)) {
        Matrix pose;
        if (agents[senderID]->getSharedPose(poseIndex, pose)) {
          poses[std::make_pair(senderID, poseIndex)] = pose;
        }
      }
      if (!poses.empty()) {
        receiver->setNeighborStatus(agents[senderID]->getStatus());
        receiver->updateNeighborPoses(senderID, poses);
      }
    }
  }

  return agents;
}

}  // namespace

TEST(testDPGO, ReadG2oIgnoresFixedPoseRecords) {
  const std::string datasetPath =
      writeTinySe2DatasetWithFixedRecords("tiny_fixed_records");
  ASSERT_EXIT(
      {
        size_t numPoses = 0;
        const std::vector<RelativeSEMeasurement> measurements =
            read_g2o_file(datasetPath, numPoses);
        if (numPoses != 2 || measurements.size() != 1 ||
            measurements.front().p1 != 0 || measurements.front().p2 != 1) {
          std::exit(2);
        }
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST(testDPGO, ManualDpgoMmParsesIndependentSolverAxes) {
  EXPECT_EQ(parseManualDpgoMmScheme("mm", true), ManualDpgoMmScheme::MM);
  EXPECT_EQ(parseManualDpgoMmScheme("amm", false), ManualDpgoMmScheme::AMM);
  EXPECT_EQ(parseManualDpgoMmScheme("auto", true), ManualDpgoMmScheme::AMM);
  EXPECT_EQ(parseManualDpgoMmScheme("auto", false), ManualDpgoMmScheme::MM);

  EXPECT_EQ(parseManualDpgoMmLocalSolver("manual_full"),
            ManualDpgoMmLocalSolver::ManualFull);
  EXPECT_EQ(parseManualDpgoMmLocalSolver("full_pose"),
            ManualDpgoMmLocalSolver::ManualFull);
  EXPECT_EQ(parseManualDpgoMmLocalSolver("full_equiv_hybrid"),
            ManualDpgoMmLocalSolver::FullEquivHybrid);
  EXPECT_EQ(parseManualDpgoMmLocalSolver("fe_hybrid"),
            ManualDpgoMmLocalSolver::FullEquivHybrid);
  EXPECT_EQ(parseManualDpgoMmLocalSolver("reduced_rotation"),
            ManualDpgoMmLocalSolver::ReducedRotation);
  EXPECT_EQ(parseManualDpgoMmLocalSolver("translation_eliminated"),
            ManualDpgoMmLocalSolver::ReducedRotation);
  EXPECT_EQ(parseManualDpgoMmFullEquivHybridBackend("manual_full_rtr"),
            ManualDpgoMmFullEquivHybridBackend::ManualFullRtr);
  EXPECT_EQ(parseManualDpgoMmFullEquivHybridBackend("sparse_direct_schur"),
            ManualDpgoMmFullEquivHybridBackend::SparseDirectSchur);
  EXPECT_EQ(parseManualDpgoMmFullEquivHybridBackend("PCG_SCHUR"),
            ManualDpgoMmFullEquivHybridBackend::PcgSchur);
  EXPECT_EQ(manualDpgoMmFullEquivHybridBackendName(
                ManualDpgoMmFullEquivHybridBackend::PcgSchur),
            "pcg_schur");
  EXPECT_EQ(parseManualDpgoMmFullEquivHybridBackend("pcg_full"),
            ManualDpgoMmFullEquivHybridBackend::PcgFull);
  EXPECT_EQ(manualDpgoMmFullEquivHybridBackendName(
                ManualDpgoMmFullEquivHybridBackend::PcgFull),
            "pcg_full");

  EXPECT_EQ(parseManualDpgoMmReducedRotationPreconditioner("none"),
            ManualDpgoMmReducedRotationPreconditioner::None);
  EXPECT_EQ(parseManualDpgoMmReducedRotationPreconditioner("jacobi"),
            ManualDpgoMmReducedRotationPreconditioner::Jacobi);
  EXPECT_EQ(parseManualDpgoMmReducedRotationPreconditioner("cholesky"),
            ManualDpgoMmReducedRotationPreconditioner::Cholesky);
  EXPECT_EQ(
      parseManualDpgoMmReducedRotationPreconditioner("regularized_cholesky"),
      ManualDpgoMmReducedRotationPreconditioner::Cholesky);
  EXPECT_EQ(
      parseManualDpgoMmReducedRotationPreconditioner("schur_jacobi"),
      ManualDpgoMmReducedRotationPreconditioner::SchurJacobi);
  EXPECT_EQ(parseManualDpgoMmReducedRotationPreconditioner(
                "reduced_schur_jacobi"),
            ManualDpgoMmReducedRotationPreconditioner::SchurJacobi);
  EXPECT_EQ(parseManualDpgoMmReducedRotationPreconditioner("portfolio"),
            ManualDpgoMmReducedRotationPreconditioner::Portfolio);
  EXPECT_EQ(
      parseManualDpgoMmReducedRotationPreconditioner("adaptive_portfolio"),
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio);
  EXPECT_EQ(manualDpgoMmReducedRotationPreconditionerName(
                ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio),
            "adaptive_portfolio");

  EXPECT_EQ(parseManualDpgoMmReducedSurrogateMode("true_local"),
            ManualDpgoMmReducedSurrogateMode::TrueLocal);
  EXPECT_EQ(parseManualDpgoMmReducedSurrogateMode("none"),
            ManualDpgoMmReducedSurrogateMode::TrueLocal);
  EXPECT_EQ(parseManualDpgoMmReducedSurrogateMode("dpgo_simple"),
            ManualDpgoMmReducedSurrogateMode::DpgoSimple);
  EXPECT_EQ(parseManualDpgoMmReducedSurrogateMode("edge_tight_quadratic"),
            ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic);
  EXPECT_EQ(parseManualDpgoMmReducedSurrogateMode("EDGE-TIGHT-QUADRATIC"),
            ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic);
  EXPECT_EQ(manualDpgoMmReducedSurrogateModeName(
                ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic),
            "edge_tight_quadratic");
  EXPECT_THROW(parseManualDpgoMmReducedSurrogateMode("unknown_surrogate"),
               std::invalid_argument);

  EXPECT_EQ(parseManualDpgoMmSurrogateMode("legacy"),
            ManualDpgoMmSurrogateMode::Legacy);
  EXPECT_EQ(parseManualDpgoMmSurrogateMode("weighted_edge_split"),
            ManualDpgoMmSurrogateMode::WeightedEdgeSplit);
  EXPECT_EQ(parseManualDpgoMmSurrogateMode("WEIGHTED-EDGE-SPLIT"),
            ManualDpgoMmSurrogateMode::WeightedEdgeSplit);
  EXPECT_EQ(parseManualDpgoMmSurrogateMode("adaptive_spectral"),
            ManualDpgoMmSurrogateMode::AdaptiveSpectral);
  EXPECT_EQ(parseManualDpgoMmSurrogateMode("variable_projected_schur"),
            ManualDpgoMmSurrogateMode::VariableProjectedSchur);
  EXPECT_EQ(manualDpgoMmSurrogateModeName(
                ManualDpgoMmSurrogateMode::WeightedEdgeSplit),
            "weighted_edge_split");
  EXPECT_THROW(parseManualDpgoMmSurrogateMode("unknown_surrogate_family"),
               std::invalid_argument);

  EXPECT_EQ(parseManualDpgoMmEdgeSplitThetaMode("constant"),
            ManualDpgoMmEdgeSplitThetaMode::Constant);
  EXPECT_EQ(parseManualDpgoMmEdgeSplitThetaMode("degree"),
            ManualDpgoMmEdgeSplitThetaMode::Degree);
  EXPECT_EQ(parseManualDpgoMmEdgeSplitThetaMode("curvature"),
            ManualDpgoMmEdgeSplitThetaMode::Curvature);
  EXPECT_EQ(parseManualDpgoMmEdgeSplitThetaMode("adaptive_conditioned"),
            ManualDpgoMmEdgeSplitThetaMode::AdaptiveConditioned);
  EXPECT_EQ(manualDpgoMmEdgeSplitThetaModeName(
                ManualDpgoMmEdgeSplitThetaMode::AdaptiveConditioned),
            "adaptive_conditioned");
  EXPECT_THROW(parseManualDpgoMmEdgeSplitThetaMode("bad_theta_mode"),
               std::invalid_argument);

  EXPECT_EQ(parseManualDpgoMmMmAcceleratorMode("none"),
            ManualDpgoMmMmAcceleratorMode::None);
  EXPECT_EQ(parseManualDpgoMmMmAcceleratorMode("nesterov_legacy"),
            ManualDpgoMmMmAcceleratorMode::NesterovLegacy);
  EXPECT_EQ(parseManualDpgoMmMmAcceleratorMode("anderson"),
            ManualDpgoMmMmAcceleratorMode::Anderson);
  EXPECT_EQ(parseManualDpgoMmMmAcceleratorMode("squarem"),
            ManualDpgoMmMmAcceleratorMode::Squarem);
  EXPECT_EQ(manualDpgoMmMmAcceleratorModeName(
                ManualDpgoMmMmAcceleratorMode::NesterovLegacy),
            "nesterov_legacy");
  EXPECT_THROW(parseManualDpgoMmMmAcceleratorMode("bad_accelerator"),
               std::invalid_argument);

  EXPECT_EQ(parseManualDpgoMmMmSafeguard("local_surrogate"),
            ManualDpgoMmMmSafeguard::LocalSurrogate);
  EXPECT_EQ(parseManualDpgoMmMmSafeguard("local_surrogate_plus_boundary"),
            ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary);
  EXPECT_EQ(parseManualDpgoMmMmSafeguard("debug_global"),
            ManualDpgoMmMmSafeguard::DebugGlobal);
  EXPECT_EQ(manualDpgoMmMmSafeguardName(
                ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary),
            "local_surrogate_plus_boundary");
  EXPECT_THROW(parseManualDpgoMmMmSafeguard("bad_safeguard"),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmUsesDpgoMmPosePayloadBytes) {
  EXPECT_EQ(manualDpgoMmPosePayloadBytes(2), 3u * 2u * sizeof(double));
  EXPECT_EQ(manualDpgoMmPosePayloadBytes(3), 4u * 3u * sizeof(double));
}

TEST(testDPGO, ManualDpgoMmQuadraticStatsMatchQuadraticProblem) {
  const unsigned d = 3;
  const unsigned r = 3;
  const unsigned n = 2;
  QuadraticProblem problem(n, d, r);

  SparseMatrix Q((d + 1) * n, (d + 1) * n);
  for (unsigned pose = 0; pose < n; ++pose) {
    const unsigned base = pose * (d + 1);
    for (unsigned col = 0; col < d + 1; ++col) {
      Q.insert(base + col, base + col) = 1.0 + 0.1 * col + pose;
    }
  }
  Q.insert(0, d + 1) = 0.2;
  Q.insert(d + 1, 0) = 0.2;
  problem.setQ(Q);

  SparseMatrix G(r, (d + 1) * n);
  G.insert(0, d) = 0.5;
  G.insert(1, d + 1 + d) = -0.25;
  problem.setG(G);

  Matrix X = Matrix::Zero(r, (d + 1) * n);
  for (int row = 0; row < X.rows(); ++row) {
    for (int col = 0; col < X.cols(); ++col) {
      X(row, col) = std::sin(0.3 * row + 0.7 * col + 0.2);
    }
  }
  LiftedSEManifold manifold(r, d, n);
  X = manifold.project(X);

  const ManualDpgoMmQuadraticStats stats =
      evaluateManualDpgoMmQuadraticStats(problem, X);

  EXPECT_NEAR(stats.globalCost, 2.0 * problem.f(X), 1e-10);
  EXPECT_NEAR(stats.gradient, 2.0 * problem.RieGradNorm(X), 1e-10);
}

TEST(testDPGO, ManualDpgoMmDefaultsToPlainMm) {
  ManualDpgoMmOptions options;
  EXPECT_EQ(options.scheme, ManualDpgoMmScheme::MM);
  EXPECT_TRUE(options.centralizedChordalInit);
  EXPECT_FALSE(options.recordLocalModelDiagnostics);
  EXPECT_FALSE(options.manualFullPortfolio);
  EXPECT_EQ(options.reducedSurrogateMode,
            ManualDpgoMmReducedSurrogateMode::TrueLocal);
  EXPECT_EQ(options.surrogateMode, ManualDpgoMmSurrogateMode::Legacy);
  EXPECT_EQ(options.edgeSplitThetaMode,
            ManualDpgoMmEdgeSplitThetaMode::Constant);
  EXPECT_DOUBLE_EQ(options.edgeSplitThetaDefault, 0.5);
  EXPECT_DOUBLE_EQ(options.edgeSplitThetaMin, 0.15);
  EXPECT_DOUBLE_EQ(options.edgeSplitThetaMax, 0.85);
  ASSERT_EQ(options.edgeSplitThetaCandidates.size(), 5u);
  EXPECT_DOUBLE_EQ(options.edgeSplitThetaCandidates[0], 0.2);
  EXPECT_DOUBLE_EQ(options.edgeSplitThetaCandidates[1], 0.35);
  EXPECT_DOUBLE_EQ(options.edgeSplitThetaCandidates[2], 0.5);
  EXPECT_DOUBLE_EQ(options.edgeSplitThetaCandidates[3], 0.65);
  EXPECT_DOUBLE_EQ(options.edgeSplitThetaCandidates[4], 0.8);
  EXPECT_EQ(options.mmAcceleratorMode,
            ManualDpgoMmMmAcceleratorMode::NesterovLegacy);
  EXPECT_EQ(options.mmSafeguard, ManualDpgoMmMmSafeguard::LocalSurrogate);
  EXPECT_FALSE(options.debugSurrogateBoundCheck);
  EXPECT_EQ(options.debugSurrogateBoundSamples, 8u);
  EXPECT_FALSE(options.localStateExtrapolationUseActiveSurrogate);
  EXPECT_FALSE(options.localBoundaryProximalCandidate);
  EXPECT_DOUBLE_EQ(options.localBoundaryProximalWeight, 0.1);
  EXPECT_TRUE(options.localBoundaryProximalWeights.empty());
  EXPECT_FALSE(options.communicationTopologyBoundaryCandidate);
  EXPECT_FALSE(options.communicationTopologyBoundarySurrogate);
  EXPECT_DOUBLE_EQ(options.communicationTopologyBoundarySurrogateStaleGain,
                   1.0);
  EXPECT_FALSE(options.communicationTopologyReducedInterfaceModel);
  EXPECT_DOUBLE_EQ(options.communicationTopologyReducedInterfaceWeight,
                   0.05);
  EXPECT_EQ(options.communicationTopologyReducedInterfaceMaxLocalIterations,
            0u);
  EXPECT_FALSE(options.communicationTopologyReducedInterfaceCandidate);
  EXPECT_FALSE(options.ammDpgoSurrogateParity);
  EXPECT_FALSE(options.ammDpgoMixedSurrogatePortfolio);
  EXPECT_FALSE(options.ammProxResetSkipRefinedSolve);
  EXPECT_FALSE(options.ammDpgoProximalFallbackOnly);
  EXPECT_FALSE(options.ammLazyPlainAfterCertificate);
  EXPECT_FALSE(options.ammSurrogateFirstExactEvaluation);
  EXPECT_TRUE(options.externalInitialEstimatePath.empty());
  EXPECT_TRUE(options.printIterationSummary);
}

TEST(testDPGO, ManualDpgoMmExternalInitialEstimateOverridesInitialState) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 0;
  baseOptions.save = false;
  baseOptions.saveIterationEstimates = true;
  baseOptions.printIterationSummary = false;
  baseOptions.parallelLocalSolves = false;

  const ManualDpgoMmRunResult seed =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  ASSERT_EQ(seed.iterations.size(), 1u);
  ASSERT_GT(seed.finalEstimate.rows(), 0);
  ASSERT_GT(seed.finalEstimate.cols(), 0);

  Matrix external = seed.finalEstimate;
  const unsigned d = static_cast<unsigned>(external.rows());
  const unsigned blockDim = d + 1;
  ASSERT_GE(external.cols(), static_cast<int>(2 * blockDim));
  external.col(blockDim + d).array() += 0.125;
  const std::string externalPath =
      writeMatrixTextFile("manual_dpgo_mm_external_init", external);

  ManualDpgoMmOptions externalOptions = baseOptions;
  externalOptions.externalInitialEstimatePath = externalPath;
  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", externalOptions);

  ASSERT_EQ(result.iterations.size(), 1u);
  ASSERT_EQ(result.finalEstimate.rows(), external.rows());
  ASSERT_EQ(result.finalEstimate.cols(), external.cols());
  EXPECT_LT((result.finalEstimate - external).norm(), 1e-12);
  ASSERT_EQ(result.initializationEstimates.size(), 1u);
  EXPECT_LT((result.initializationEstimates.front() - external).norm(),
            1e-12);
}

TEST(testDPGO, ManualDpgoMmExternalInitialEstimateRejectsWrongShape) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  const Matrix bad = Matrix::Zero(2, 3);
  const std::string badPath =
      writeMatrixTextFile("manual_dpgo_mm_bad_external_init", bad);

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 0;
  options.save = false;
  options.printIterationSummary = false;
  options.externalInitialEstimatePath = badPath;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmWeightedEdgeSplitIdentityMajorizes) {
  Matrix a(3, 4);
  Matrix b(3, 4);
  Matrix ak(3, 4);
  Matrix bk(3, 4);
  for (int row = 0; row < a.rows(); ++row) {
    for (int col = 0; col < a.cols(); ++col) {
      a(row, col) = std::sin(0.3 * row + 0.5 * col + 0.1);
      b(row, col) = std::cos(0.2 * row - 0.4 * col + 0.7);
      ak(row, col) = std::sin(0.7 * row + 0.1 * col - 0.2);
      bk(row, col) = std::cos(0.4 * row + 0.6 * col + 0.3);
    }
  }

  for (const double theta : {0.2, 0.35, 0.5, 0.65, 0.8}) {
    const ManualDpgoMmWeightedEdgeSplitStats stats =
        evaluateManualDpgoMmWeightedEdgeSplit(a, b, ak, bk, theta);
    EXPECT_GE(stats.gap, -1e-12);
    EXPECT_NEAR(stats.gap, stats.identityGap, 1e-12);
    EXPECT_NEAR(stats.surrogate, stats.objective + stats.identityGap, 1e-12);
  }

  const ManualDpgoMmWeightedEdgeSplitStats anchor =
      evaluateManualDpgoMmWeightedEdgeSplit(ak, bk, ak, bk, 0.35);
  EXPECT_NEAR(anchor.surrogate, anchor.objective, 1e-12);
  EXPECT_NEAR(anchor.gap, 0.0, 1e-12);
  EXPECT_NEAR(anchor.identityGap, 0.0, 1e-12);

  EXPECT_THROW(evaluateManualDpgoMmWeightedEdgeSplit(a, b, ak, bk, 0.0),
               std::invalid_argument);
  EXPECT_THROW(evaluateManualDpgoMmWeightedEdgeSplit(a, b, ak, bk, 1.0),
               std::invalid_argument);
  EXPECT_THROW(evaluateManualDpgoMmWeightedEdgeSplit(a, b, ak.leftCols(3),
                                                     bk, 0.5),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmDegreeEdgeSplitThetaBalancesRobotDegrees) {
  const double theta =
      evaluateManualDpgoMmDegreeEdgeSplitTheta(16.0, 4.0, 0.15, 0.85);
  EXPECT_NEAR(theta, 4.0 / 6.0, 1e-12);

  EXPECT_NEAR(evaluateManualDpgoMmDegreeEdgeSplitTheta(4.0, 16.0, 0.15,
                                                       0.85),
              2.0 / 6.0, 1e-12);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmDegreeEdgeSplitTheta(100.0, 1.0, 0.4,
                                                            0.6),
                   0.6);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmDegreeEdgeSplitTheta(1.0, 100.0, 0.4,
                                                            0.6),
                   0.4);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmDegreeEdgeSplitTheta(0.0, 0.0, 0.15,
                                                            0.85),
                   0.5);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmDegreeEdgeSplitTheta(
                       std::numeric_limits<double>::quiet_NaN(), 4.0, 0.15,
                       0.85),
                   0.5);
}

TEST(testDPGO, ManualDpgoMmCurvatureEdgeSplitThetaBalancesEdgeCurvature) {
  const double theta =
      evaluateManualDpgoMmCurvatureEdgeSplitTheta(25.0, 4.0, 0.1, 0.9);
  EXPECT_NEAR(theta, 5.0 / 7.0, 1e-12);

  EXPECT_NEAR(evaluateManualDpgoMmCurvatureEdgeSplitTheta(4.0, 25.0, 0.1,
                                                          0.9),
              2.0 / 7.0, 1e-12);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmCurvatureEdgeSplitTheta(100.0, 1.0,
                                                               0.25, 0.75),
                   0.75);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmCurvatureEdgeSplitTheta(1.0, 100.0,
                                                               0.25, 0.75),
                   0.25);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmCurvatureEdgeSplitTheta(0.0, 4.0, 0.1,
                                                               0.9),
                   0.5);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmCurvatureEdgeSplitTheta(
                       4.0, std::numeric_limits<double>::infinity(), 0.1,
                       0.9),
                   0.5);
}

TEST(testDPGO, ManualDpgoMmAdaptiveConditionedThetaUsesCandidateScore) {
  const std::vector<double> candidates{0.2, 0.35, 0.5, 0.65, 0.8};

  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmAdaptiveConditionedEdgeSplitTheta(
                       25.0, 1.0, 25.0, 1.0, candidates, 0.15, 0.85),
                   0.8);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmAdaptiveConditionedEdgeSplitTheta(
                       1.0, 25.0, 1.0, 25.0, candidates, 0.15, 0.85),
                   0.2);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmAdaptiveConditionedEdgeSplitTheta(
                       4.0, 4.0, 9.0, 9.0, candidates, 0.15, 0.85),
                   0.5);

  EXPECT_NEAR(evaluateManualDpgoMmAdaptiveConditionedEdgeSplitTheta(
                  16.0, 4.0, 0.0, 0.0, candidates, 0.15, 0.85),
              4.0 / 6.0, 1e-12);
  EXPECT_DOUBLE_EQ(evaluateManualDpgoMmAdaptiveConditionedEdgeSplitTheta(
                       0.0, 0.0, 0.0, 0.0, candidates, 0.15, 0.85),
                   0.5);
}

TEST(testDPGO, ManualDpgoMmScalarYoungMajorizerMajorizesCrossBlock) {
  Matrix cross(3, 3);
  cross << 1.0, -0.2, 0.3, 0.4, 0.7, -0.1, -0.3, 0.2, 0.5;

  for (int sample = 0; sample < 8; ++sample) {
    Matrix deltaAlpha(4, 3);
    Matrix deltaBeta(4, 3);
    for (int row = 0; row < deltaAlpha.rows(); ++row) {
      for (int col = 0; col < deltaAlpha.cols(); ++col) {
        deltaAlpha(row, col) =
            std::sin(0.31 * sample + 0.17 * row + 0.13 * col);
        deltaBeta(row, col) =
            std::cos(0.23 * sample - 0.11 * row + 0.19 * col);
      }
    }
    const double gap =
        evaluateManualDpgoMmScalarYoungMajorizerGap(deltaAlpha, deltaBeta,
                                                    cross, 0.8);
    EXPECT_GE(gap, -1e-10);
  }

  Matrix zeroAlpha = Matrix::Zero(2, 3);
  Matrix zeroBeta = Matrix::Zero(2, 3);
  EXPECT_NEAR(evaluateManualDpgoMmScalarYoungMajorizerGap(
                  zeroAlpha, zeroBeta, cross, 1.0),
              0.0, 1e-12);
  EXPECT_THROW(evaluateManualDpgoMmScalarYoungMajorizerGap(
                   zeroAlpha, zeroBeta, cross, 0.0),
               std::invalid_argument);
  EXPECT_THROW(evaluateManualDpgoMmScalarYoungMajorizerGap(
                   zeroAlpha.leftCols(2), zeroBeta, cross, 1.0),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmBlockGershgorinMajorizerMajorizesSchurBlocks) {
  Matrix schur(6, 6);
  schur << 3.0, 0.2, -0.4, 0.1, 0.3, -0.2,
      0.2, 2.4, 0.5, -0.2, 0.1, 0.4,
      -0.4, 0.5, 2.8, 0.6, -0.3, 0.2,
      0.1, -0.2, 0.6, 2.2, 0.5, -0.1,
      0.3, 0.1, -0.3, 0.5, 2.6, 0.7,
      -0.2, 0.4, 0.2, -0.1, 0.7, 3.1;
  schur = 0.5 * (schur + schur.transpose()).eval();

  for (int sample = 0; sample < 8; ++sample) {
    Matrix delta(5, 6);
    for (int row = 0; row < delta.rows(); ++row) {
      for (int col = 0; col < delta.cols(); ++col) {
        delta(row, col) =
            std::sin(0.19 * sample + 0.31 * row - 0.13 * col);
      }
    }
    const double gap =
        evaluateManualDpgoMmBlockGershgorinMajorizerGap(delta, schur, 2);
    EXPECT_GE(gap, -1e-10);
  }

  Matrix zero = Matrix::Zero(3, 6);
  EXPECT_NEAR(evaluateManualDpgoMmBlockGershgorinMajorizerGap(zero, schur, 2),
              0.0, 1e-12);
  EXPECT_THROW(evaluateManualDpgoMmBlockGershgorinMajorizerGap(zero, schur, 0),
               std::invalid_argument);
  EXPECT_THROW(evaluateManualDpgoMmBlockGershgorinMajorizerGap(
                   zero.leftCols(5), schur, 2),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmAmmUsesDeadbandedLocalMeritByDefault) {
  ManualDpgoMmOptions options;
  EXPECT_GT(options.ammLocalMeritCostTieTolerance, 0.0);
  EXPECT_TRUE(options.ammLocalMeritFilter);
  EXPECT_FALSE(options.globalStateExtrapolation);
}

TEST(testDPGO, ManualDpgoMmRunsTinyGridFullAndReducedSolvers) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.recordLocalModelDiagnostics = true;

  options.localSolver = ManualDpgoMmLocalSolver::ManualFull;
  const ManualDpgoMmRunResult full =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  const ManualDpgoMmRunResult fullEquivHybrid =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  const ManualDpgoMmRunResult reduced =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Cholesky;
  const ManualDpgoMmRunResult choleskyReduced =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::SchurJacobi;
  const ManualDpgoMmRunResult schurJacobiReduced =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(full.iterations.size(), 2u);
  ASSERT_EQ(fullEquivHybrid.iterations.size(), 2u);
  ASSERT_EQ(reduced.iterations.size(), 2u);
  ASSERT_EQ(choleskyReduced.iterations.size(), 2u);
  ASSERT_EQ(schurJacobiReduced.iterations.size(), 2u);
  EXPECT_EQ(full.iterations.front().localModelFailures, 0u);
  EXPECT_EQ(full.iterations.back().localModelFailures, 0u);
  EXPECT_EQ(full.iterations.back().localOptimizationFailures, 0u);
  EXPECT_EQ(fullEquivHybrid.iterations.back().localModelFailures, 0u);
  EXPECT_EQ(fullEquivHybrid.iterations.back().localOptimizationFailures, 0u);
  EXPECT_EQ(reduced.iterations.front().localModelFailures, 0u);
  EXPECT_EQ(reduced.iterations.back().localModelFailures, 0u);
  EXPECT_EQ(reduced.iterations.back().localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(full.iterations.back().localModelCostBefore));
  EXPECT_TRUE(std::isfinite(full.iterations.back().localModelCostAfter));
  EXPECT_TRUE(std::isfinite(
      fullEquivHybrid.iterations.back().localModelCostBefore));
  EXPECT_TRUE(std::isfinite(
      fullEquivHybrid.iterations.back().localModelCostAfter));
  EXPECT_TRUE(std::isfinite(reduced.iterations.back().localModelCostBefore));
  EXPECT_TRUE(std::isfinite(reduced.iterations.back().localModelCostAfter));
  EXPECT_LE(full.iterations.back().localModelCostAfter,
            full.iterations.back().localModelCostBefore + 1e-8);
  EXPECT_LE(fullEquivHybrid.iterations.back().localModelCostAfter,
            fullEquivHybrid.iterations.back().localModelCostBefore + 1e-8);
  EXPECT_LE(reduced.iterations.back().localModelCostAfter,
            reduced.iterations.back().localModelCostBefore + 1e-8);
  EXPECT_TRUE(std::isfinite(full.finalObjective));
  EXPECT_TRUE(std::isfinite(fullEquivHybrid.finalObjective));
  EXPECT_TRUE(std::isfinite(reduced.finalObjective));
  EXPECT_TRUE(std::isfinite(choleskyReduced.finalObjective));
  EXPECT_TRUE(std::isfinite(choleskyReduced.finalGradient));
  EXPECT_TRUE(std::isfinite(schurJacobiReduced.finalObjective));
  EXPECT_TRUE(std::isfinite(schurJacobiReduced.finalGradient));
  EXPECT_LE(full.finalObjective, full.iterations.front().globalCost);
  EXPECT_LE(fullEquivHybrid.finalObjective,
            fullEquivHybrid.iterations.front().globalCost);
  EXPECT_LE(reduced.finalObjective, reduced.iterations.front().globalCost);
  EXPECT_LE(choleskyReduced.finalObjective,
            choleskyReduced.iterations.front().globalCost);
  EXPECT_LE(schurJacobiReduced.finalObjective,
            schurJacobiReduced.iterations.front().globalCost);
  EXPECT_NEAR(full.finalObjective, fullEquivHybrid.finalObjective, 1e-10);
  EXPECT_NEAR(full.finalGradient, fullEquivHybrid.finalGradient, 1e-10);
  EXPECT_EQ(full.iterations.back().cumulativeCommPoseCount,
            fullEquivHybrid.iterations.back().cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(full.iterations.back().cumulativeCommMb,
                   fullEquivHybrid.iterations.back().cumulativeCommMb);
  EXPECT_EQ(full.iterations.back().cumulativeCommPoseCount,
            reduced.iterations.back().cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(full.iterations.back().cumulativeCommMb,
                   reduced.iterations.back().cumulativeCommMb);
  EXPECT_EQ(reduced.iterations.back().cumulativeCommPoseCount,
            schurJacobiReduced.iterations.back().cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(reduced.iterations.back().cumulativeCommMb,
                   schurJacobiReduced.iterations.back().cumulativeCommMb);
}

TEST(testDPGO, ManualDpgoMmReducedPortfolioIncludesSchurJacobiCandidate) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.surrogateMode = ManualDpgoMmSurrogateMode::Legacy;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;
  options.profileOptimizer = true;

  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  const ManualDpgoMmRunResult portfolio =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(portfolio.iterations.size(), 2u);
  EXPECT_EQ(portfolio.optimizerReducedSolveCount, 8u);
  EXPECT_TRUE(std::isfinite(portfolio.finalObjective));
  EXPECT_TRUE(std::isfinite(portfolio.finalGradient));
}

TEST(testDPGO, ManualDpgoMmCanSkipLocalModelDiagnostics) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.printIterationSummary = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.recordLocalModelDiagnostics = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_EQ(result.iterations.back().localModelFailures, 0u);
  EXPECT_EQ(result.iterations.back().localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isnan(result.iterations.back().localModelCostBefore));
  EXPECT_TRUE(std::isnan(result.iterations.back().localModelCostAfter));
  EXPECT_TRUE(std::isnan(result.iterations.back().localModelGradientBefore));
  EXPECT_TRUE(std::isnan(result.iterations.back().localModelGradientAfter));
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmEdgeTightQuadraticReportsDiagnosticsOnTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.reducedSurrogateMode =
      ManualDpgoMmReducedSurrogateMode::EdgeTightQuadratic;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));

  std::size_t evalCount = 0;
  double surrogateCostSum = 0.0;
  double trueCostSum = 0.0;
  double minGap = std::numeric_limits<double>::infinity();
  for (const auto &row : result.iterations) {
    evalCount += row.edgeTightQuadraticEvalCount;
    surrogateCostSum += row.edgeTightQuadraticSurrogateCostSum;
    trueCostSum += row.edgeTightQuadraticTrueCostSum;
    minGap = std::min(minGap, row.edgeTightQuadraticMajorizationGapMin);
  }

  EXPECT_GT(evalCount, 0u);
  EXPECT_TRUE(std::isfinite(surrogateCostSum));
  EXPECT_TRUE(std::isfinite(trueCostSum));
  EXPECT_TRUE(std::isfinite(minGap));
  EXPECT_GE(minGap, -1e-8);
}

TEST(testDPGO, ManualDpgoMmWeightedEdgeSplitThetaHalfMatchesDpgoSimple) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions dpgoSimple;
  dpgoSimple.numRobots = 2;
  dpgoSimple.maxIterations = 2;
  dpgoSimple.scheme = ManualDpgoMmScheme::MM;
  dpgoSimple.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  dpgoSimple.reducedSurrogateMode =
      ManualDpgoMmReducedSurrogateMode::DpgoSimple;
  dpgoSimple.save = false;
  dpgoSimple.trustRegionIterations = 1;
  dpgoSimple.trustRegionMaxInnerIterations = 8;
  dpgoSimple.parallelLocalSolves = false;

  ManualDpgoMmOptions weighted = dpgoSimple;
  weighted.reducedSurrogateMode = ManualDpgoMmReducedSurrogateMode::TrueLocal;
  weighted.surrogateMode = ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
  weighted.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Constant;
  weighted.edgeSplitThetaDefault = 0.5;

  const ManualDpgoMmRunResult dpgoSimpleResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", dpgoSimple);
  const ManualDpgoMmRunResult weightedResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", weighted);

  ASSERT_EQ(dpgoSimpleResult.iterations.size(),
            weightedResult.iterations.size());
  ASSERT_FALSE(dpgoSimpleResult.iterations.empty());
  EXPECT_NEAR(weightedResult.finalObjective, dpgoSimpleResult.finalObjective,
              1e-10);
  EXPECT_NEAR(weightedResult.finalGradient, dpgoSimpleResult.finalGradient,
              1e-10);
  EXPECT_EQ(weightedResult.iterations.back().cumulativeCommPoseCount,
            dpgoSimpleResult.iterations.back().cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(weightedResult.iterations.back().cumulativeCommMb,
                   dpgoSimpleResult.iterations.back().cumulativeCommMb);
  for (std::size_t idx = 0; idx < dpgoSimpleResult.iterations.size(); ++idx) {
    EXPECT_NEAR(weightedResult.iterations[idx].globalCost,
                dpgoSimpleResult.iterations[idx].globalCost, 1e-10);
    EXPECT_NEAR(weightedResult.iterations[idx].gradient,
                dpgoSimpleResult.iterations[idx].gradient, 1e-10);
  }
}

TEST(testDPGO, ManualDpgoMmWeightedEdgeSplitNonSymmetricThetaRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.surrogateMode = ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
  options.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Constant;
  options.edgeSplitThetaDefault = 0.35;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_EQ(result.iterations.back().localModelFailures, 0u);
  EXPECT_EQ(result.iterations.back().localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
  EXPECT_EQ(result.iterations.back().cumulativeCommPoseCount, 14u);
  EXPECT_DOUBLE_EQ(result.iterations.back().cumulativeCommMb,
                   2.0 * result.iterations.front().iterCommMb);
}

TEST(testDPGO,
     ManualDpgoMmWeightedEdgeSplitDebugBoundCheckReportsNoViolations) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.surrogateMode = ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
  options.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Constant;
  options.edgeSplitThetaDefault = 0.35;
  options.debugSurrogateBoundCheck = true;
  options.debugSurrogateBoundSamples = 4;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  EXPECT_GT(result.surrogateBoundCheckCount, 0u);
  EXPECT_EQ(result.surrogateBoundViolationCount, 0u);
  EXPECT_TRUE(std::isfinite(result.surrogateBoundMinMargin));
  EXPECT_GE(result.surrogateBoundMinMargin, -1e-10);
}

TEST(testDPGO,
     ManualDpgoMmWeightedEdgeSplitDebugOffLeavesBoundCheckDisabled) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.surrogateMode = ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
  options.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Constant;
  options.edgeSplitThetaDefault = 0.35;
  options.debugSurrogateBoundCheck = false;
  options.debugSurrogateBoundSamples = 4;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  EXPECT_EQ(result.surrogateBoundCheckCount, 0u);
  EXPECT_EQ(result.surrogateBoundViolationCount, 0u);
  EXPECT_DOUBLE_EQ(result.surrogateBoundMinMargin, 0.0);
}

TEST(testDPGO, ManualDpgoMmWeightedEdgeSplitDegreeModeAffectsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions constant;
  constant.numRobots = 2;
  constant.maxIterations = 2;
  constant.scheme = ManualDpgoMmScheme::MM;
  constant.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  constant.surrogateMode = ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
  constant.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Constant;
  constant.edgeSplitThetaDefault = 0.5;
  constant.debugSurrogateBoundCheck = true;
  constant.debugSurrogateBoundSamples = 2;
  constant.save = false;
  constant.trustRegionIterations = 1;
  constant.trustRegionMaxInnerIterations = 8;
  constant.parallelLocalSolves = false;

  ManualDpgoMmOptions degree = constant;
  degree.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Degree;

  const ManualDpgoMmRunResult constantResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", constant);
  const ManualDpgoMmRunResult degreeResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", degree);

  ASSERT_EQ(constantResult.iterations.size(), degreeResult.iterations.size());
  EXPECT_EQ(degreeResult.surrogateBoundViolationCount, 0u);
  EXPECT_GT(degreeResult.surrogateBoundCheckCount, 0u);
  EXPECT_TRUE(std::isfinite(degreeResult.finalObjective));
  EXPECT_TRUE(std::isfinite(degreeResult.finalGradient));
  EXPECT_GT(std::abs(degreeResult.finalObjective -
                     constantResult.finalObjective),
            1e-8);
}

TEST(testDPGO, ManualDpgoMmWeightedEdgeSplitCurvatureModeAffectsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions constant;
  constant.numRobots = 2;
  constant.maxIterations = 2;
  constant.scheme = ManualDpgoMmScheme::MM;
  constant.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  constant.surrogateMode = ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
  constant.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Constant;
  constant.edgeSplitThetaDefault = 0.5;
  constant.debugSurrogateBoundCheck = true;
  constant.debugSurrogateBoundSamples = 2;
  constant.save = false;
  constant.trustRegionIterations = 1;
  constant.trustRegionMaxInnerIterations = 8;
  constant.parallelLocalSolves = false;

  ManualDpgoMmOptions curvature = constant;
  curvature.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Curvature;

  const ManualDpgoMmRunResult constantResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", constant);
  const ManualDpgoMmRunResult curvatureResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", curvature);

  ASSERT_EQ(constantResult.iterations.size(),
            curvatureResult.iterations.size());
  EXPECT_EQ(curvatureResult.surrogateBoundViolationCount, 0u);
  EXPECT_GT(curvatureResult.surrogateBoundCheckCount, 0u);
  EXPECT_TRUE(std::isfinite(curvatureResult.finalObjective));
  EXPECT_TRUE(std::isfinite(curvatureResult.finalGradient));
  EXPECT_GT(std::abs(curvatureResult.finalObjective -
                     constantResult.finalObjective),
            1e-8);
}

TEST(testDPGO, ManualDpgoMmWeightedEdgeSplitAdaptiveConditionedModeAffectsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions constant;
  constant.numRobots = 2;
  constant.maxIterations = 2;
  constant.scheme = ManualDpgoMmScheme::MM;
  constant.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  constant.surrogateMode = ManualDpgoMmSurrogateMode::WeightedEdgeSplit;
  constant.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Constant;
  constant.edgeSplitThetaDefault = 0.5;
  constant.debugSurrogateBoundCheck = true;
  constant.debugSurrogateBoundSamples = 2;
  constant.save = false;
  constant.trustRegionIterations = 1;
  constant.trustRegionMaxInnerIterations = 8;
  constant.parallelLocalSolves = false;

  ManualDpgoMmOptions adaptive = constant;
  adaptive.edgeSplitThetaMode =
      ManualDpgoMmEdgeSplitThetaMode::AdaptiveConditioned;

  const ManualDpgoMmRunResult constantResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", constant);
  const ManualDpgoMmRunResult adaptiveResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", adaptive);

  ASSERT_EQ(constantResult.iterations.size(),
            adaptiveResult.iterations.size());
  EXPECT_EQ(adaptiveResult.surrogateBoundViolationCount, 0u);
  EXPECT_GT(adaptiveResult.surrogateBoundCheckCount, 0u);
  EXPECT_TRUE(std::isfinite(adaptiveResult.finalObjective));
  EXPECT_TRUE(std::isfinite(adaptiveResult.finalGradient));
  EXPECT_GT(std::abs(adaptiveResult.finalObjective -
                     constantResult.finalObjective),
            1e-8);
}

TEST(testDPGO, ManualDpgoMmAdaptiveSpectralRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.surrogateMode = ManualDpgoMmSurrogateMode::AdaptiveSpectral;
  options.edgeSplitThetaMode = ManualDpgoMmEdgeSplitThetaMode::Constant;
  options.edgeSplitThetaDefault = 0.5;
  options.debugSurrogateBoundCheck = true;
  options.debugSurrogateBoundSamples = 2;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.surrogateBoundViolationCount, 0u);
  EXPECT_GT(result.surrogateBoundCheckCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmVariableProjectedSchurReportsReducedDiagnostics) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.surrogateMode = ManualDpgoMmSurrogateMode::VariableProjectedSchur;
  options.debugSurrogateBoundCheck = true;
  options.debugSurrogateBoundSamples = 2;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_GT(result.variableProjectedSchurCandidateCount, 0u);
  EXPECT_GT(result.variableProjectedSchurAcceptedCount, 0u);
  EXPECT_GT(result.surrogateBoundCheckCount, 0u);
  EXPECT_EQ(result.surrogateBoundViolationCount, 0u);
  EXPECT_TRUE(std::isfinite(result.surrogateBoundMinMargin));
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO,
     ManualDpgoMmLocalStateExtrapolationUsesActiveSurrogateMap) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.surrogateMode = ManualDpgoMmSurrogateMode::VariableProjectedSchur;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationUseActiveSurrogate = true;
  options.localStateExtrapolationGammas = {0.1, 0.25};
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  const std::size_t baseCandidateCalls =
      options.numRobots * options.maxIterations;
  const std::size_t extrapolatedCandidateCalls =
      options.numRobots * options.localStateExtrapolationGammas.size();
  EXPECT_GE(result.variableProjectedSchurCandidateCount,
            baseCandidateCalls + extrapolatedCandidateCalls);
  EXPECT_EQ(result.iterations.back().extrapolationAcceptedCount +
                result.iterations.back().extrapolationRejectedCount,
            extrapolatedCandidateCalls);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, FullEquivHybridSchurWarmStartKeepsFullModelGuard) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridSchurWarmStart = true;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_EQ(result.iterations.back().localModelFailures, 0u);
  EXPECT_EQ(result.iterations.back().localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_LE(result.iterations.back().localModelCostAfter,
            result.iterations.back().localModelCostBefore + 1e-8);
  EXPECT_GT(result.iterations.back().fullEquivHybridWarmStartCandidateCount,
            0u);
  EXPECT_EQ(result.iterations.back().fullEquivHybridWarmStartCandidateCount,
            result.iterations.back().fullEquivHybridWarmStartAcceptedCount +
                result.iterations.back()
                    .fullEquivHybridWarmStartGuardRejectedCount);
}

TEST(testDPGO, FullEquivHybridSparseDirectSchurBackendRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::SparseDirectSchur;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_LE(last.localModelCostAfter, last.localModelCostBefore + 1e-8);
  EXPECT_GT(last.fullEquivHybridSchurStepCandidateCount, 0u);
  EXPECT_EQ(last.fullEquivHybridSchurStepCandidateCount,
            last.fullEquivHybridSchurStepAcceptedCount +
                last.fullEquivHybridSchurStepGuardRejectedCount);
  EXPECT_GT(last.fullEquivHybridSchurStepAcceptedCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearFullResidual, 0.0);
  EXPECT_LT(last.fullEquivHybridLinearFullResidual, 1e-8);
  EXPECT_GT(last.fullEquivHybridLinearSolveTimeSec, 0.0);
}

TEST(testDPGO, FullEquivHybridPcgSchurBackendRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgSchur;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 100;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_GT(last.fullEquivHybridSchurStepAcceptedCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearPcgIterationCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearInitialResidual, 0.0);
  EXPECT_LT(last.fullEquivHybridLinearFinalResidual,
            last.fullEquivHybridLinearInitialResidual);
  EXPECT_GT(last.fullEquivHybridLinearFullResidual, 0.0);
  EXPECT_LT(last.fullEquivHybridLinearFullResidual, 1e-5);
  EXPECT_GT(last.fullEquivHybridLinearSolveTimeSec, 0.0);
  EXPECT_GT(last.fullEquivHybridStepTrialCount, 0u);
  EXPECT_EQ(last.fullEquivHybridStepTrialCount,
            last.fullEquivHybridStepTrialAcceptedCount +
                last.fullEquivHybridStepTrialRejectedCount);
  EXPECT_EQ(last.fullEquivHybridStepTrialAcceptedCount,
            last.fullEquivHybridSchurStepAcceptedCount);
  EXPECT_TRUE(std::isfinite(last.fullEquivHybridStepPredictedDecreaseSum));
  EXPECT_TRUE(std::isfinite(last.fullEquivHybridStepActualDecreaseSum));
  EXPECT_GT(last.fullEquivHybridStepRhoCount, 0u);
  EXPECT_TRUE(std::isfinite(last.fullEquivHybridStepRhoSum));
  EXPECT_GT(last.fullEquivHybridStepAcceptedScaleSum, 0.0);
  EXPECT_LE(last.fullEquivHybridStepAcceptedScaleSum,
            static_cast<double>(last.fullEquivHybridStepTrialAcceptedCount) +
                1e-12);
  EXPECT_TRUE(
      std::isfinite(last.fullEquivHybridStepAcceptedPredictedDecreaseSum));
  EXPECT_GT(last.fullEquivHybridStepAcceptedActualDecreaseSum, 0.0);
  EXPECT_GT(last.fullEquivHybridStepAcceptedRhoCount, 0u);
  EXPECT_TRUE(std::isfinite(last.fullEquivHybridStepAcceptedRhoSum));
}

TEST(testDPGO, FullEquivHybridPcgFullBackendRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_GT(last.fullEquivHybridSchurStepAcceptedCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearPcgIterationCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearInitialResidual, 0.0);
  EXPECT_LT(last.fullEquivHybridLinearFinalResidual,
            last.fullEquivHybridLinearInitialResidual);
  EXPECT_GT(last.fullEquivHybridLinearFullResidual, 0.0);
  EXPECT_LT(last.fullEquivHybridLinearFullResidual, 1e-5);
  EXPECT_GT(last.fullEquivHybridLinearSolveTimeSec, 0.0);
  EXPECT_EQ(last.fullEquivHybridActiveSeparatorCandidateCount, 0u);
  EXPECT_EQ(last.fullEquivHybridActiveSeparatorAcceptedCount, 0u);
  EXPECT_EQ(last.fullEquivHybridActiveSeparatorRejectedCount, 0u);
}

TEST(testDPGO, FullEquivHybridSelectBestBacktrackingRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridWarmStartBacktrackingSteps = 4;
  options.fullEquivHybridSelectBestBacktrackingTrial = true;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_GT(last.fullEquivHybridSchurStepAcceptedCount, 0u);
  EXPECT_GT(last.fullEquivHybridStepTrialCount, 0u);
  EXPECT_EQ(last.fullEquivHybridStepTrialCount,
            last.fullEquivHybridStepTrialAcceptedCount +
                last.fullEquivHybridStepTrialRejectedCount);
  EXPECT_EQ(last.fullEquivHybridStepTrialAcceptedCount,
            last.fullEquivHybridSchurStepAcceptedCount);
  EXPECT_GE(last.fullEquivHybridStepTrialCount,
            last.fullEquivHybridSchurStepAcceptedCount);
  EXPECT_GT(last.fullEquivHybridStepAcceptedActualDecreaseSum, 0.0);
  EXPECT_LE(last.localModelCostAfter, last.localModelCostBefore + 1e-9);
}

TEST(testDPGO, FullEquivHybridStepParetoSelectorRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridWarmStartBacktrackingSteps = 4;
  options.fullEquivHybridStepParetoSelector = true;
  options.fullEquivHybridStepParetoMinDecreaseRatio = 0.75;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_GT(last.fullEquivHybridSchurStepAcceptedCount, 0u);
  EXPECT_GT(last.fullEquivHybridStepTrialCount, 0u);
  EXPECT_EQ(last.fullEquivHybridStepTrialCount,
            last.fullEquivHybridStepTrialAcceptedCount +
                last.fullEquivHybridStepTrialRejectedCount);
  EXPECT_EQ(last.fullEquivHybridStepTrialAcceptedCount,
            last.fullEquivHybridSchurStepAcceptedCount);
  EXPECT_GE(last.fullEquivHybridStepParetoCandidateCount,
            last.fullEquivHybridStepParetoSelectedCount);
  EXPECT_EQ(last.fullEquivHybridStepParetoSelectedCount,
            last.fullEquivHybridStepTrialAcceptedCount);
  EXPECT_EQ(last.fullEquivHybridStepParetoGradientEvalCount,
            last.fullEquivHybridStepParetoCandidateCount);
  EXPECT_GT(last.fullEquivHybridStepParetoSelectedScaleSum, 0.0);
  EXPECT_LE(last.fullEquivHybridStepParetoSelectedDecreaseSum,
            last.fullEquivHybridStepParetoBestDecreaseSum + 1e-9);
  EXPECT_TRUE(
      std::isfinite(last.fullEquivHybridStepParetoSelectedGradientSum));
  EXPECT_TRUE(
      std::isfinite(last.fullEquivHybridStepParetoGradientEvalTimeSec));
  EXPECT_LE(last.localModelCostAfter, last.localModelCostBefore + 1e-9);
}

TEST(testDPGO, FullEquivHybridSharedProxBacktrackingRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridWarmStartBacktrackingSteps = 4;
  options.fullEquivHybridBacktrackingSharedPoseProxWeight = 1e-2;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_GT(last.fullEquivHybridSchurStepAcceptedCount, 0u);
  EXPECT_EQ(last.fullEquivHybridStepTrialCount,
            last.fullEquivHybridStepTrialAcceptedCount +
                last.fullEquivHybridStepTrialRejectedCount);
  EXPECT_EQ(last.fullEquivHybridStepTrialAcceptedCount,
            last.fullEquivHybridSchurStepAcceptedCount);
  EXPECT_LE(last.localModelCostAfter, last.localModelCostBefore + 1e-9);
}

TEST(testDPGO, FullEquivHybridProjectedModelRhoGuardRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridWarmStartBacktrackingSteps = 4;
  options.fullEquivHybridProjectedModelRhoGuard = true;
  options.fullEquivHybridProjectedModelRhoEta = 1e-4;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_GT(last.fullEquivHybridSchurStepAcceptedCount, 0u);
  EXPECT_EQ(last.fullEquivHybridStepTrialCount,
            last.fullEquivHybridStepTrialAcceptedCount +
                last.fullEquivHybridStepTrialRejectedCount);
  EXPECT_EQ(last.fullEquivHybridStepTrialAcceptedCount,
            last.fullEquivHybridSchurStepAcceptedCount);
  EXPECT_LE(last.localModelCostAfter, last.localModelCostBefore + 1e-9);
}

TEST(testDPGO, FullEquivHybridActiveSeparatorCorrectionRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorStepCap = 1e-3;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  EXPECT_GT(last.fullEquivHybridActiveSeparatorCandidateCount, 0u);
  EXPECT_EQ(last.fullEquivHybridActiveSeparatorCandidateCount,
            last.fullEquivHybridActiveSeparatorAcceptedCount +
                last.fullEquivHybridActiveSeparatorRejectedCount);
  EXPECT_TRUE(
      std::isfinite(last.fullEquivHybridActiveSeparatorStepSum));
  EXPECT_TRUE(std::isfinite(
      last.fullEquivHybridActiveSeparatorCostDecreaseSum));
  if (last.fullEquivHybridActiveSeparatorAcceptedCount > 0u) {
    EXPECT_GT(last.fullEquivHybridActiveSeparatorStepSum, 0.0);
    EXPECT_GT(last.fullEquivHybridActiveSeparatorCostDecreaseSum, 0.0);
  }
}

TEST(testDPGO, FullEquivHybridActiveSeparatorCompactSchurUsesPrivateCouplingTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorCompactSchur = true;
  options.fullEquivHybridActiveSeparatorStepCap = 1e-3;
  options.localGradientCorrectionCompactSchurMaxBoundaryPoses = 8;
  options.localGradientCorrectionCompactSchurMaxPrivateCols = 8;
  options.localGradientCorrectionCompactSchurDamping = 1e-6;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridActiveSeparatorCandidateCount, 0u);
  EXPECT_GT(last.localGradientCompactSchurCandidateCount, 0u);
  EXPECT_GT(last.localGradientCompactSchurBoundaryColCount, 0u);
  EXPECT_GT(last.localGradientCompactSchurPrivateColCount, 0u);
  EXPECT_EQ(last.localGradientCompactSchurCandidateCount,
            last.localGradientCompactSchurAcceptedCount +
                last.localGradientCompactSchurGuardRejectedCount +
                last.localGradientCompactSchurGradientGuardRejectedCount +
                last.localGradientCompactSchurSolveFailureCount);
  EXPECT_EQ(last.localGradientCompactSchurGradientGuardRejectedCount, 0u);
  EXPECT_DOUBLE_EQ(last.localGradientCompactSchurGradientChangeSum, 0.0);
}

TEST(testDPGO, FullEquivHybridActiveSeparatorCompactSchurGradientGuardRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorCompactSchur = true;
  options.fullEquivHybridActiveSeparatorStepCap = 1e-3;
  options.localGradientCorrectionCompactSchurGradientGuard = true;
  options.localGradientCorrectionCompactSchurMaxGradientIncreaseRatio = 0.0;
  options.localGradientCorrectionCompactSchurMaxBoundaryPoses = 8;
  options.localGradientCorrectionCompactSchurMaxPrivateCols = 8;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_GT(last.localGradientCompactSchurCandidateCount, 0u);
  EXPECT_TRUE(std::isfinite(
      last.localGradientCompactSchurGradientChangeSum));
  EXPECT_EQ(last.localGradientCompactSchurCandidateCount,
            last.localGradientCompactSchurAcceptedCount +
                last.localGradientCompactSchurGuardRejectedCount +
                last.localGradientCompactSchurGradientGuardRejectedCount +
                last.localGradientCompactSchurSolveFailureCount);
}

TEST(testDPGO, FullEquivHybridActiveSeparatorLmSchurRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 1;
  baseOptions.scheme = ManualDpgoMmScheme::MM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  baseOptions.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  baseOptions.fullEquivHybridLinearRelativeTolerance = 1e-8;
  baseOptions.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  baseOptions.fullEquivHybridLinearMaxIterations = 120;
  baseOptions.fullEquivHybridActiveSeparatorCorrection = true;
  baseOptions.fullEquivHybridActiveSeparatorStepCap = 1e-3;
  baseOptions.save = false;
  baseOptions.recordLocalModelDiagnostics = true;

  ManualDpgoMmOptions lmOptions = baseOptions;
  lmOptions.fullEquivHybridActiveSeparatorLmSchur = true;
  lmOptions.fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses = 8;
  lmOptions.fullEquivHybridActiveSeparatorLmSchurMaxPrivateCols = 8;
  lmOptions.fullEquivHybridActiveSeparatorLmSchurDamping = 1e-6;
  lmOptions.fullEquivHybridActiveSeparatorLmSchurBacktrackingSteps = 4;

  const ManualDpgoMmRunResult baseResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult lmResult =
      runManualDpgoMm("data/tinyGrid3D.g2o", lmOptions);

  ASSERT_EQ(lmResult.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = lmResult.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(lmResult.finalObjective));
  EXPECT_LE(lmResult.finalObjective, baseResult.finalObjective + 1e-9);
  EXPECT_GT(last.fullEquivHybridActiveSeparatorLmSchurCandidateCount, 0u);
  EXPECT_GT(last.fullEquivHybridActiveSeparatorLmSchurBoundaryColCount, 0u);
  EXPECT_GT(last.fullEquivHybridActiveSeparatorLmSchurPrivateColCount, 0u);
  EXPECT_EQ(last.fullEquivHybridActiveSeparatorLmSchurCandidateCount,
            last.fullEquivHybridActiveSeparatorLmSchurAcceptedCount +
                last.fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount +
                last.fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount +
                last.fullEquivHybridActiveSeparatorLmSchurSolveFailureCount +
                last.fullEquivHybridActiveSeparatorLmSchurFallbackCount);
}

TEST(testDPGO, FullEquivHybridActiveSeparatorLmSchurGradientGuardRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorStepCap = 1e-3;
  options.fullEquivHybridActiveSeparatorLmSchur = true;
  options.fullEquivHybridActiveSeparatorLmSchurGradientGuard = true;
  options.fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio = 0.0;
  options.fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses = 8;
  options.fullEquivHybridActiveSeparatorLmSchurMaxPrivateCols = 8;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(
      last.fullEquivHybridActiveSeparatorLmSchurGradientChangeSum));
  EXPECT_GT(last.fullEquivHybridActiveSeparatorLmSchurCandidateCount, 0u);
  EXPECT_EQ(last.fullEquivHybridActiveSeparatorLmSchurCandidateCount,
            last.fullEquivHybridActiveSeparatorLmSchurAcceptedCount +
                last.fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount +
                last.fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount +
                last.fullEquivHybridActiveSeparatorLmSchurSolveFailureCount +
                last.fullEquivHybridActiveSeparatorLmSchurFallbackCount);
}

TEST(testDPGO, FullEquivHybridActiveSeparatorLmSchurModelDecreaseScoreRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorStepCap = 1e-3;
  options.fullEquivHybridActiveSeparatorLmSchur = true;
  options.fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses = 1;
  options.fullEquivHybridActiveSeparatorLmSchurMaxPrivateCols = 8;
  options.fullEquivHybridActiveSeparatorLmSchurScoreMode =
      ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode::
          ModelDecrease;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_GT(last.fullEquivHybridActiveSeparatorLmSchurCandidateCount, 0u);
  EXPECT_GT(last.fullEquivHybridActiveSeparatorLmSchurBoundaryColCount, 0u);
}

TEST(testDPGO, ManualDpgoMmRejectsNegativeCompactSchurMinCostDecrease) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.localGradientCorrectionCompactSchurMinCostDecrease = -1e-3;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmRejectsNegativeCompactSchurGradientGuardRatio) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.localGradientCorrectionCompactSchur = true;
  options.localGradientCorrectionCompactSchurGradientGuard = true;
  options.localGradientCorrectionCompactSchurMaxGradientIncreaseRatio = -1e-3;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridActiveSeparatorLmSchurRejectsNegativeMinCostImprovement) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorLmSchur = true;
  options.fullEquivHybridActiveSeparatorLmSchurMinCostImprovement = -1e-3;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridActiveSeparatorLmSchurGradientGuardRequiresLmSchur) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorLmSchurGradientGuard = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridActiveSeparatorLmSchurRejectsNegativeGradientGuardRatio) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorLmSchur = true;
  options.fullEquivHybridActiveSeparatorLmSchurGradientGuard = true;
  options.fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio =
      -1e-3;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridActiveSeparatorLmSchurHonorsMaxRounds) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorLmSchur = true;
  options.fullEquivHybridActiveSeparatorLmSchurMaxRounds = 1;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_GT(
      result.iterations[1].fullEquivHybridActiveSeparatorLmSchurCandidateCount,
      0u);
  EXPECT_EQ(
      result.iterations[2].fullEquivHybridActiveSeparatorLmSchurCandidateCount,
      0u);
}

TEST(testDPGO, FullEquivHybridTranslationRecoveryPolishRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridTranslationRecoveryPolishAttemptCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishAttemptCount,
            last.fullEquivHybridTranslationRecoveryPolishAcceptedCount +
                last.fullEquivHybridTranslationRecoveryPolishRejectedCount);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount,
      0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount,
      0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount, 0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount,
      0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum,
            0.0);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritCandidateCount,
            0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount, 0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount,
      0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum,
            0.0);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum,
            0.0);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum,
      0.0);
  if (last.fullEquivHybridTranslationRecoveryPolishAcceptedCount > 0u) {
    EXPECT_GT(
        last.fullEquivHybridTranslationRecoveryPolishCostDecreaseSum, 0.0);
  }
  EXPECT_EQ(last.ammTranslationRecoveryAttemptCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridTranslationRecoveryPolishDefaultsOff) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishAttemptCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishAcceptedCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishRejectedCount, 0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount,
      0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount,
      0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount, 0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount,
      0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum,
            0.0);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritCandidateCount,
            0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount, 0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount,
      0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum,
            0.0);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum,
            0.0);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum,
      0.0);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishCostDecreaseSum, 0.0);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishGradientChangeSum,
            0.0);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridTranslationRecoveryPolishSelectedOnlyRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridTranslationRecoveryPolishSelectedOnly = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridTranslationRecoveryPolishAttemptCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishAttemptCount,
            last.fullEquivHybridTranslationRecoveryPolishAcceptedCount +
                last.fullEquivHybridTranslationRecoveryPolishRejectedCount);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishSelectedOnlyRequiresPolish) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryPolishSelectedOnly = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridTranslationRecoveryStepTrialsRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridTranslationRecoveryStepTrials = true;
  options.fullEquivHybridStepParetoSelector = true;
  options.fullEquivHybridStepParetoMinDecreaseRatio = 0.75;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridTranslationRecoveryStepTrialCandidateCount,
            0u);
  EXPECT_LE(
      last.fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount,
      last.fullEquivHybridTranslationRecoveryStepTrialCandidateCount);
  EXPECT_LE(last.fullEquivHybridTranslationRecoveryStepTrialSelectedCount,
            last.fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount);
  EXPECT_GT(last.fullEquivHybridStepTrialCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryStepTrialsRequiresFehBackend) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::ManualFullRtr;
  options.fullEquivHybridTranslationRecoveryStepTrials = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryStepTrialsRequiresFullEquivHybridSolver) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.fullEquivHybridTranslationRecoveryStepTrials = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridTranslationRecoveryPolishGradientGuardRuns) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridTranslationRecoveryPolishGradientGuard = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_GT(last.fullEquivHybridTranslationRecoveryPolishAttemptCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishAttemptCount,
            last.fullEquivHybridTranslationRecoveryPolishAcceptedCount +
                last.fullEquivHybridTranslationRecoveryPolishRejectedCount);
  EXPECT_LE(
      last.fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount,
      last.fullEquivHybridTranslationRecoveryPolishRejectedCount);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount,
      0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount, 0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount,
      0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum,
            0.0);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritCandidateCount,
            0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount, 0u);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount,
      0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum,
            0.0);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum,
            0.0);
  EXPECT_EQ(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum,
      0.0);
  EXPECT_TRUE(std::isfinite(
      last.fullEquivHybridTranslationRecoveryPolishGradientChangeSum));
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishBacktrackingRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridTranslationRecoveryPolishGradientGuard = true;
  options.fullEquivHybridTranslationRecoveryPolishBacktracking = true;
  options.fullEquivHybridTranslationRecoveryPolishBacktrackingSteps = 3;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_GT(last.fullEquivHybridTranslationRecoveryPolishAttemptCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishAttemptCount,
            last.fullEquivHybridTranslationRecoveryPolishAcceptedCount +
                last.fullEquivHybridTranslationRecoveryPolishRejectedCount);
  EXPECT_LE(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount,
      last.fullEquivHybridTranslationRecoveryPolishAcceptedCount);
  EXPECT_GE(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount,
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount);
  EXPECT_LE(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount,
      last.fullEquivHybridTranslationRecoveryPolishAttemptCount);
  if (last.fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount >
      0u) {
    EXPECT_GT(
        last.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum,
        0.0);
    EXPECT_LT(
        last.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum,
        static_cast<double>(
            last.fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount));
  }
  EXPECT_TRUE(std::isfinite(
      last.fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum));
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishMeritSelectorRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridTranslationRecoveryPolishMeritSelector = true;
  options.fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio =
      0.9;
  options.fullEquivHybridTranslationRecoveryPolishBacktrackingSteps = 3;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_GT(last.fullEquivHybridTranslationRecoveryPolishAttemptCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishAttemptCount,
            last.fullEquivHybridTranslationRecoveryPolishAcceptedCount +
                last.fullEquivHybridTranslationRecoveryPolishRejectedCount);
  EXPECT_GE(last.fullEquivHybridTranslationRecoveryPolishMeritCandidateCount,
            last.fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount +
                last.fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount);
  EXPECT_LE(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount +
          last.fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount,
      last.fullEquivHybridTranslationRecoveryPolishAcceptedCount);
  EXPECT_TRUE(std::isfinite(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum));
  EXPECT_TRUE(std::isfinite(
      last.fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum));
  EXPECT_TRUE(std::isfinite(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum));
  EXPECT_LE(
      last.fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum,
      last.fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum +
          1e-9);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishGradientGuardRequiresPolish) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryPolishGradientGuard = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishRejectsNegativeGradientRatio) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridTranslationRecoveryPolishGradientGuard = true;
  options.fullEquivHybridTranslationRecoveryPolishMaxGradientIncreaseRatio =
      -1e-3;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishBacktrackingRequiresGuard) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridTranslationRecoveryPolishBacktracking = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishMeritSelectorRequiresPolish) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryPolishMeritSelector = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishMeritSelectorRejectsBadRatio) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridTranslationRecoveryPolishMeritSelector = true;
  options.fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio =
      1.1;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridStepParetoSelectorRejectsBadRatio) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridStepParetoSelector = true;
  options.fullEquivHybridStepParetoMinDecreaseRatio = 1.1;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishBacktrackingRejectsZeroSteps) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridTranslationRecoveryPolishGradientGuard = true;
  options.fullEquivHybridTranslationRecoveryPolishBacktracking = true;
  options.fullEquivHybridTranslationRecoveryPolishBacktrackingSteps = 0;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridTranslationRecoveryPolishRejectsRqnMemory) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridRqnWarmStart = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryPolishRejectsRqnPreconditioner) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearBlockJacobiPreconditioner = false;
  options.fullEquivHybridTranslationRecoveryPolish = true;
  options.fullEquivHybridRqnMemoryPreconditioner = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridActiveSeparatorRejectsRqnMemoryOptions) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridRqnWarmStart = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridPcgFullRqnWarmStartRecordsHistory) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridRqnWarmStart = true;
  options.fullEquivHybridRqnMemorySize = 3;
  options.save = false;
  options.recordLocalModelDiagnostics = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridRqnUsedCount, 0u);
  EXPECT_GT(last.fullEquivHybridRqnAcceptedPairCount, 0u);
  EXPECT_GT(last.fullEquivHybridRqnMemorySize, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridPcgFullRqnWarmStartReducesNoJacobiPcgWork) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baselineOptions;
  baselineOptions.numRobots = 2;
  baselineOptions.maxIterations = 2;
  baselineOptions.scheme = ManualDpgoMmScheme::MM;
  baselineOptions.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  baselineOptions.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  baselineOptions.fullEquivHybridLinearRelativeTolerance = 1e-8;
  baselineOptions.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  baselineOptions.fullEquivHybridLinearMaxIterations = 120;
  baselineOptions.fullEquivHybridLinearBlockJacobiPreconditioner = false;
  baselineOptions.fullEquivHybridSparseMatrixVectorProduct = false;
  baselineOptions.save = false;

  ManualDpgoMmOptions rqnOptions = baselineOptions;
  rqnOptions.fullEquivHybridRqnWarmStart = true;
  rqnOptions.fullEquivHybridRqnMemorySize = 3;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", baselineOptions);
  const ManualDpgoMmRunResult rqn =
      runManualDpgoMm("data/tinyGrid3D.g2o", rqnOptions);

  ASSERT_EQ(baseline.iterations.size(), 3u);
  ASSERT_EQ(rqn.iterations.size(), 3u);
  const ManualDpgoMmIterationSummary &baselineLast =
      baseline.iterations.back();
  const ManualDpgoMmIterationSummary &rqnLast = rqn.iterations.back();
  EXPECT_EQ(baselineLast.localOptimizationFailures, 0u);
  EXPECT_EQ(rqnLast.localOptimizationFailures, 0u);
  EXPECT_EQ(baselineLast.fullEquivHybridSparseMatrixVectorProductCount, 0u);
  EXPECT_EQ(rqnLast.fullEquivHybridSparseMatrixVectorProductCount, 0u);
  EXPECT_GT(rqnLast.fullEquivHybridRqnUsedCount, 0u);
  EXPECT_LT(rqnLast.fullEquivHybridLinearPcgIterationCount,
            baselineLast.fullEquivHybridLinearPcgIterationCount);
  EXPECT_LE(rqn.finalObjective, baseline.finalObjective + 1e-6);
}

TEST(testDPGO, FullEquivHybridPcgFullRqnPreconditionerRecordsApplications) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridLinearBlockJacobiPreconditioner = false;
  options.fullEquivHybridSparseMatrixVectorProduct = false;
  options.fullEquivHybridRqnMemoryPreconditioner = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridRqnAcceptedPairCount, 0u);
  EXPECT_GT(last.fullEquivHybridRqnMemorySize, 0u);
  EXPECT_GT(last.fullEquivHybridRqnPreconditionerApplicationCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridPcgFullLocalChainPreconditionerRecordsApplications) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridLocalChainPreconditioner = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridLocalChainPreconditionerApplicationCount, 0u);
  EXPECT_GT(last.fullEquivHybridLocalChainPreconditionerFactorizationCount, 0u);
  EXPECT_EQ(last.fullEquivHybridLocalChainPreconditionerFallbackCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridSchwarzPreSmoothingRecordsAcceptedBlocks) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 80;
  options.fullEquivHybridSchwarzPreSmoothing = true;
  options.fullEquivHybridSchwarzSweeps = 1;
  options.fullEquivHybridSchwarzMaxBlockNorm = 1.0;
  options.recordLocalModelDiagnostics = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridSchwarzSmoothingCandidateCount, 0u);
  EXPECT_GT(last.fullEquivHybridSchwarzSmoothingAcceptedCount, 0u);
  EXPECT_EQ(last.fullEquivHybridSchwarzSmoothingCandidateCount,
            last.fullEquivHybridSchwarzSmoothingAcceptedCount +
                last.fullEquivHybridSchwarzSmoothingRejectedCount);
  EXPECT_GT(last.fullEquivHybridSchwarzSmoothingCostDecrease, 0.0);
  EXPECT_GT(last.fullEquivHybridSchurStepCandidateCount, 0u);
  EXPECT_LE(last.localModelCostAfter, last.localModelCostBefore + 1e-9);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridSchwarzEdgePairModeRecordsCoupledBlocks) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 80;
  options.fullEquivHybridSchwarzPreSmoothing = true;
  options.fullEquivHybridSchwarzSweeps = 1;
  options.fullEquivHybridSchwarzMaxBlockNorm = 1.0;
  options.fullEquivHybridSchwarzBlockMode =
      ManualDpgoMmFullEquivHybridSchwarzBlockMode::EdgePair;
  options.recordLocalModelDiagnostics = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridSchwarzSmoothingCandidateCount, 0u);
  EXPECT_GT(last.fullEquivHybridSchwarzSmoothingAcceptedCount, 0u);
  EXPECT_EQ(last.fullEquivHybridSchwarzSmoothingCandidateCount,
            last.fullEquivHybridSchwarzSmoothingAcceptedCount +
                last.fullEquivHybridSchwarzSmoothingRejectedCount);
  EXPECT_GT(last.fullEquivHybridSchwarzSmoothingCostDecrease, 0.0);
  EXPECT_LE(last.localModelCostAfter, last.localModelCostBefore + 1e-9);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridLocalPortfolioRecordsSelectionCounts) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 80;
  options.fullEquivHybridLocalPortfolio = true;
  options.fullEquivHybridSchwarzPreSmoothing = true;
  options.fullEquivHybridSchwarzSweeps = 1;
  options.fullEquivHybridSchwarzMaxBlockNorm = 1.0;
  options.fullEquivHybridSchwarzBlockMode =
      ManualDpgoMmFullEquivHybridSchwarzBlockMode::EdgePair;
  options.recordLocalModelDiagnostics = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  const std::size_t selectedCount =
      last.fullEquivHybridLocalPortfolioSelectedUnsmoothedCount +
      last.fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount +
      last.fullEquivHybridLocalPortfolioSelectedSchwarzFehCount;
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_EQ(last.fullEquivHybridLocalPortfolioCandidateCount,
            static_cast<std::size_t>(options.numRobots) * 3u);
  EXPECT_EQ(selectedCount, static_cast<std::size_t>(options.numRobots));
  EXPECT_GT(last.fullEquivHybridSchurStepCandidateCount, 0u);
  EXPECT_GT(last.fullEquivHybridSchwarzSmoothingCandidateCount, 0u);
  EXPECT_LE(last.localModelCostAfter, last.localModelCostBefore + 1e-9);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridRqnPreconditionerRejectsUnsupportedManualOptions) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgSchur;
  options.fullEquivHybridRqnMemoryPreconditioner = true;
  options.save = false;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);

  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationSparseSchurPreconditioner = true;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);

  options.fullEquivHybridTranslationSparseSchurPreconditioner = false;
  options.fullEquivHybridRqnMemoryPreconditioner = false;
  options.fullEquivHybridRqnWarmStart = true;
  options.fullEquivHybridLocalPortfolio = true;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);

  options.fullEquivHybridLocalPortfolio = false;
  options.localStateExtrapolation = true;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);

  options.fullEquivHybridRqnWarmStart = false;
  options.fullEquivHybridRqnMemoryPreconditioner = true;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);

  options.localStateExtrapolation = false;
  options.fullEquivHybridRqnMemoryPreconditioner = false;
  options.fullEquivHybridRqnWarmStart = true;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);

  options.fullEquivHybridRqnWarmStart = false;
  options.fullEquivHybridRqnMemoryPreconditioner = true;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridPcgFullReducedRotationInitialGuessIsGuarded) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridLinearBlockJacobiPreconditioner = false;
  options.fullEquivHybridReducedRotationInitialGuess = true;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Cholesky;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridReducedRotationInitialGuessUsedCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearPcgIterationCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearFullResidual, 0.0);
  EXPECT_LT(last.fullEquivHybridLinearFullResidual, 1e-5);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO,
     FullEquivHybridPcgFullReducedRotationPortfolioInitialGuessUsesBestMode) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 5;
  baseOptions.maxIterations = 1;
  baseOptions.scheme = ManualDpgoMmScheme::MM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  baseOptions.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  baseOptions.fullEquivHybridLinearRelativeTolerance = 1e-8;
  baseOptions.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  baseOptions.fullEquivHybridLinearMaxIterations = 120;
  baseOptions.fullEquivHybridLinearBlockJacobiPreconditioner = false;
  baseOptions.fullEquivHybridReducedRotationInitialGuess = true;
  baseOptions.save = false;

  ManualDpgoMmOptions choleskyOptions = baseOptions;
  choleskyOptions.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Cholesky;
  ManualDpgoMmOptions portfolioOptions = baseOptions;
  portfolioOptions.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;

  const ManualDpgoMmRunResult cholesky =
      runManualDpgoMm("data/tinyGrid3D.g2o", choleskyOptions);
  const ManualDpgoMmRunResult portfolio =
      runManualDpgoMm("data/tinyGrid3D.g2o", portfolioOptions);

  ASSERT_EQ(cholesky.iterations.size(), 2u);
  ASSERT_EQ(portfolio.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &choleskyLast =
      cholesky.iterations.back();
  const ManualDpgoMmIterationSummary &portfolioLast =
      portfolio.iterations.back();
  ASSERT_GT(choleskyLast.fullEquivHybridReducedRotationInitialGuessUsedCount,
            0u);
  ASSERT_GT(portfolioLast.fullEquivHybridReducedRotationInitialGuessUsedCount,
            0u);
  EXPECT_LE(portfolio.finalObjective, cholesky.finalObjective + 1e-10);
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryInitialGuessRunsInsidePcgFull) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridLinearBlockJacobiPreconditioner = false;
  options.fullEquivHybridTranslationRecoveryInitialGuess = true;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_GT(last.fullEquivHybridTranslationRecoveryInitialGuessCandidateCount,
            0u);
  EXPECT_GT(last.fullEquivHybridTranslationRecoveryInitialGuessUsedCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryInitialGuessCandidateCount,
            last.fullEquivHybridTranslationRecoveryInitialGuessUsedCount +
                last.fullEquivHybridTranslationRecoveryInitialGuessRejectedCount);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryPolishAttemptCount, 0u);
  EXPECT_EQ(last.fullEquivHybridTranslationRecoveryStepTrialCandidateCount,
            0u);
  EXPECT_GT(last.fullEquivHybridLinearPcgIterationCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO,
     FullEquivHybridTranslationRecoveryInitialGuessRejectsReducedInitialGuessMix) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridTranslationRecoveryInitialGuess = true;
  options.fullEquivHybridReducedRotationInitialGuess = true;
  options.save = false;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridPcgFullReducedRotationPreconditionerRunsTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.fullEquivHybridLinearBlockJacobiPreconditioner = false;
  options.fullEquivHybridReducedRotationPreconditioner = true;
  options.fullEquivHybridReducedRotationInitialGuess = false;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Cholesky;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_EQ(last.localModelFailures, 0u);
  EXPECT_EQ(last.localOptimizationFailures, 0u);
  EXPECT_EQ(last.fullEquivHybridReducedRotationInitialGuessCandidateCount, 0u);
  EXPECT_EQ(last.fullEquivHybridReducedRotationInitialGuessRejectedCount, 0u);
  EXPECT_EQ(last.fullEquivHybridReducedRotationInitialGuessUsedCount, 0u);
  EXPECT_GT(last.fullEquivHybridReducedRotationPreconditionerApplicationCount,
            0u);
  EXPECT_GT(last.fullEquivHybridLinearPcgIterationCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearFullResidual, 0.0);
  EXPECT_LT(last.fullEquivHybridLinearFullResidual, 1e-5);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, FullEquivHybridPcgSchurRejectedSolveStillReportsTime) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgSchur;
  options.fullEquivHybridLinearRelativeTolerance = 0.0;
  options.fullEquivHybridLinearAbsoluteTolerance = 0.0;
  options.fullEquivHybridLinearMaxIterations = 1;
  options.save = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_GT(last.fullEquivHybridSchurStepCandidateCount, 0u);
  EXPECT_GT(last.fullEquivHybridSchurStepGuardRejectedCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearPcgIterationCount, 0u);
  EXPECT_GT(last.fullEquivHybridLinearSolveTimeSec, 0.0);
}

TEST(testDPGO, FullEquivHybridSparseDirectSchurPolishMatchesManualFullTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 1;
  baseOptions.scheme = ManualDpgoMmScheme::MM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ManualFull;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 10;

  const ManualDpgoMmRunResult manualFull =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);

  ManualDpgoMmOptions fehOptions = baseOptions;
  fehOptions.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  fehOptions.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::SparseDirectSchur;
  fehOptions.fullEquivHybridFinalPolishRtr = true;

  const ManualDpgoMmRunResult feh =
      runManualDpgoMm("data/tinyGrid3D.g2o", fehOptions);

  ASSERT_EQ(feh.iterations.size(), manualFull.iterations.size());
  EXPECT_EQ(feh.iterations.back().localModelFailures, 0u);
  EXPECT_EQ(feh.iterations.back().localOptimizationFailures, 0u);
  EXPECT_GT(feh.iterations.back().fullEquivHybridSchurStepAcceptedCount, 0u);
  EXPECT_LE(feh.finalObjective, manualFull.finalObjective + 1e-8);
  EXPECT_LE(feh.finalGradient, manualFull.finalGradient + 1e-8);
}

TEST(testDPGO, ManualDpgoMmAdaptiveReducedTcgRefinesWeakLocalSolve) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.recordLocalModelDiagnostics = true;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult fixed =
      runManualDpgoMm("data/CSAIL.g2o", options);

  options.adaptiveReducedTcg = true;
  options.adaptiveReducedTcgMaxIterations = 50;
  options.adaptiveReducedTcgGradientRatio = 0.15;
  const ManualDpgoMmRunResult adaptive =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(fixed.iterations.size(), 2u);
  ASSERT_EQ(adaptive.iterations.size(), 2u);
  EXPECT_GT(adaptive.iterations.back().adaptiveRefinementCount, 0u);
  EXPECT_LT(adaptive.finalObjective, fixed.finalObjective - 1e-3);
  EXPECT_LT(adaptive.iterations.back().localModelGradientAfter,
            fixed.iterations.back().localModelGradientAfter);
  EXPECT_EQ(adaptive.iterations.back().cumulativeCommPoseCount,
            fixed.iterations.back().cumulativeCommPoseCount);
}

TEST(testDPGO, ManualDpgoMmReducedTcgToleranceIsConfigurable) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 50;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult loose =
      runManualDpgoMm("data/CSAIL.g2o", options);

  options.reducedRotationTcgRelativeTolerance = 1e-2;
  const ManualDpgoMmRunResult tight =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(loose.iterations.size(), 2u);
  ASSERT_EQ(tight.iterations.size(), 2u);
  EXPECT_LE(tight.finalObjective, loose.finalObjective + 1e-10);
  EXPECT_EQ(tight.iterations.back().cumulativeCommPoseCount,
            loose.iterations.back().cumulativeCommPoseCount);
}

TEST(testDPGO, ManualDpgoMmLocalExtrapolationAttemptsAfterFirstStep) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGamma = 0.5;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations[1].extrapolationAcceptedCount +
                result.iterations[1].extrapolationRejectedCount,
            0u);
  EXPECT_GT(result.iterations[2].extrapolationAcceptedCount +
                result.iterations[2].extrapolationRejectedCount,
            0u);
  EXPECT_EQ(result.iterations.back().localOptimizationFailures, 0u);
}

TEST(testDPGO, ManualDpgoMmLocalExtrapolationTriesMultipleGammas) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGammas = {0.25, 0.5};
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  const auto &secondUpdate = result.iterations[2];
  EXPECT_EQ(secondUpdate.extrapolationAcceptedCount +
                secondUpdate.extrapolationRejectedCount,
            options.numRobots * options.localStateExtrapolationGammas.size());
  EXPECT_EQ(secondUpdate.cumulativeCommPoseCount,
            result.iterations[1].cumulativeCommPoseCount +
                secondUpdate.commPoseCount);
}

TEST(testDPGO, FullEquivHybridLocalExtrapolationAttemptsAfterFirstStep) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.save = false;
  options.parallelLocalSolves = false;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGammas = {0.25, 0.5};

  ManualDpgoMmOptions noExtrapOptions = options;
  noExtrapOptions.localStateExtrapolation = false;
  const ManualDpgoMmRunResult noExtrap =
      runManualDpgoMm("data/tinyGrid3D.g2o", noExtrapOptions);

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  ASSERT_EQ(noExtrap.iterations.size(), result.iterations.size());
  EXPECT_EQ(result.iterations[1].extrapolationAcceptedCount +
                result.iterations[1].extrapolationRejectedCount,
            0u);
  const auto &secondUpdate = result.iterations[2];
  EXPECT_EQ(secondUpdate.extrapolationAcceptedCount +
                secondUpdate.extrapolationRejectedCount,
            options.numRobots * options.localStateExtrapolationGammas.size());
  EXPECT_EQ(secondUpdate.cumulativeCommPoseCount,
            result.iterations[1].cumulativeCommPoseCount +
                secondUpdate.commPoseCount);
  for (std::size_t i = 0; i < result.iterations.size(); ++i) {
    EXPECT_EQ(result.iterations[i].commPoseCount,
              noExtrap.iterations[i].commPoseCount);
    EXPECT_EQ(result.iterations[i].postExchangePoseCount,
              noExtrap.iterations[i].postExchangePoseCount);
    EXPECT_EQ(result.iterations[i].localGradientFreshPoseCount,
              noExtrap.iterations[i].localGradientFreshPoseCount);
    EXPECT_EQ(result.iterations[i].localGradientScalarCount,
              noExtrap.iterations[i].localGradientScalarCount);
  }
  EXPECT_EQ(secondUpdate.localOptimizationFailures, 0u);
}

TEST(testDPGO, ManualFullAmmRecordsAccelerationTrace) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ManualFull;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammGammaScales = {1.0};

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t traceCount = 0;
  for (const auto &row : result.iterations) {
    traceCount += row.ammTraceCount;
  }
  EXPECT_GT(traceCount, 0u);
  EXPECT_EQ(result.iterations.back().localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualFullPortfolioAttemptsLocalExtrapolatedCandidates) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ManualFull;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.manualFullPortfolio = true;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGammas = {0.25, 0.5};

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations[1].extrapolationAcceptedCount +
                result.iterations[1].extrapolationRejectedCount,
            0u);
  const auto &secondUpdate = result.iterations[2];
  EXPECT_EQ(secondUpdate.extrapolationAcceptedCount +
                secondUpdate.extrapolationRejectedCount,
            options.numRobots * options.localStateExtrapolationGammas.size());
  EXPECT_EQ(secondUpdate.localOptimizationFailures, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualFullRejectsReducedOnlyAmmSurrogateParityFlags) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ManualFull;
  options.ammDpgoSurrogateParity = true;
  options.save = false;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);

  options.ammDpgoSurrogateParity = false;
  options.ammDpgoMixedSurrogatePortfolio = true;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, FullEquivHybridAmmMixedSurrogateReportsSelectionSource) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 120;
  options.save = false;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_GE(result.iterations.size(), 2u);
  EXPECT_EQ(result.iterations.front().ammMixedSurrogateCandidateCount, 0u);
  for (std::size_t i = 1; i < result.iterations.size(); ++i) {
    const auto &row = result.iterations[i];
    EXPECT_EQ(row.ammMixedSurrogateCandidateCount, options.numRobots);
    const std::size_t selected =
        row.ammMixedSurrogateSimpleSelectedCount +
        row.ammMixedSurrogateTrueLocalSelectedCount +
        row.ammMixedSurrogateExtrapolatedSelectedCount +
        row.ammMixedSurrogateOtherSelectedCount;
    EXPECT_EQ(selected, options.numRobots);
    EXPECT_EQ(row.localOptimizationFailures, 0u);
  }
}

TEST(testDPGO,
     FullEquivHybridMixedSurrogateCanSkipAndForceSimpleRefresh) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 5;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 80;
  options.fullEquivHybridTranslationSparseSchurPreconditioner = true;
  options.fullEquivHybridSchwarzPreSmoothing = true;
  options.fullEquivHybridSchwarzSweeps = 1;
  options.fullEquivHybridSchwarzMaxBlockNorm = 1.0;
  options.fullEquivHybridSchwarzBlockMode =
      ManualDpgoMmFullEquivHybridSchwarzBlockMode::EdgePair;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorStepCap = 5e-2;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGammas = {0.5};
  options.save = false;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak = 1;
  options.ammMixedSurrogateForceSimpleEverySkippedRounds = 1;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_GE(result.iterations.size(), 4u);
  std::size_t skipped = 0;
  std::size_t forcedRefreshes = 0;
  for (std::size_t i = 1; i < result.iterations.size(); ++i) {
    const auto &row = result.iterations[i];
    skipped += row.ammMixedSurrogateSimpleSkippedCount;
    forcedRefreshes += row.ammMixedSurrogateSimpleForcedRefreshCount;
    const std::size_t selected =
        row.ammMixedSurrogateSimpleSelectedCount +
        row.ammMixedSurrogateTrueLocalSelectedCount +
        row.ammMixedSurrogateExtrapolatedSelectedCount +
        row.ammMixedSurrogateOtherSelectedCount;
    EXPECT_EQ(selected, options.numRobots);
    EXPECT_EQ(row.localOptimizationFailures, 0u);
  }
  EXPECT_GT(skipped, 0u);
  EXPECT_GT(forcedRefreshes, 0u);
}

TEST(testDPGO,
     FullEquivHybridMixedSurrogateReportsExtrapolatedIncumbentSource) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 5;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::FullEquivHybrid;
  options.fullEquivHybridBackend =
      ManualDpgoMmFullEquivHybridBackend::PcgFull;
  options.fullEquivHybridLinearRelativeTolerance = 1e-8;
  options.fullEquivHybridLinearAbsoluteTolerance = 1e-12;
  options.fullEquivHybridLinearMaxIterations = 80;
  options.fullEquivHybridTranslationSparseSchurPreconditioner = true;
  options.fullEquivHybridSchwarzPreSmoothing = true;
  options.fullEquivHybridSchwarzSweeps = 1;
  options.fullEquivHybridSchwarzMaxBlockNorm = 1.0;
  options.fullEquivHybridSchwarzBlockMode =
      ManualDpgoMmFullEquivHybridSchwarzBlockMode::EdgePair;
  options.fullEquivHybridActiveSeparatorCorrection = true;
  options.fullEquivHybridActiveSeparatorStepCap = 5e-2;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGammas = {0.5};
  options.save = false;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_GE(result.iterations.size(), 4u);
  std::size_t extrapolatedSelected = 0;
  std::size_t extrapolationAttempts = 0;
  std::size_t trueLocalSelected = 0;
  std::size_t trueLocalAccepted = 0;
  for (std::size_t i = 1; i < result.iterations.size(); ++i) {
    const auto &row = result.iterations[i];
    extrapolatedSelected += row.ammMixedSurrogateExtrapolatedSelectedCount;
    extrapolationAttempts += row.extrapolationAcceptedCount +
                             row.extrapolationRejectedCount;
    trueLocalSelected += row.ammMixedSurrogateTrueLocalSelectedCount;
    trueLocalAccepted += row.ammMixedSurrogateTrueLocalAcceptedCount;
    const std::size_t selected =
        row.ammMixedSurrogateSimpleSelectedCount +
        row.ammMixedSurrogateTrueLocalSelectedCount +
        row.ammMixedSurrogateExtrapolatedSelectedCount +
        row.ammMixedSurrogateOtherSelectedCount;
    EXPECT_EQ(selected, options.numRobots);
    EXPECT_EQ(row.localOptimizationFailures, 0u);
  }
  EXPECT_GT(extrapolationAttempts, 0u);
  EXPECT_GT(extrapolatedSelected, 0u);
  EXPECT_GT(trueLocalSelected, 0u);
  EXPECT_EQ(trueLocalAccepted, trueLocalSelected);
}

TEST(testDPGO, ManualDpgoMmLocalGExtrapolationAttemptsAfterFirstStep) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.localModelGExtrapolation = true;
  options.localModelGExtrapolationGammas = {0.25, 0.5};
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations[1].gExtrapolationAcceptedCount +
                result.iterations[1].gExtrapolationRejectedCount,
            0u);
  EXPECT_EQ(result.iterations[2].gExtrapolationAcceptedCount +
                result.iterations[2].gExtrapolationRejectedCount,
            options.numRobots * options.localModelGExtrapolationGammas.size());
  EXPECT_EQ(result.iterations[2].cumulativeCommPoseCount,
            result.iterations[1].cumulativeCommPoseCount +
                result.iterations[2].commPoseCount);
}

TEST(testDPGO, ManualDpgoMmCoupledExtrapolationAttemptsAfterFirstStep) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.coupledStateGExtrapolation = true;
  options.coupledStateGExtrapolationGammas = {0.25, 0.5};
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations[1].coupledExtrapolationAcceptedCount +
                result.iterations[1].coupledExtrapolationRejectedCount,
            0u);
  EXPECT_EQ(result.iterations[2].coupledExtrapolationAcceptedCount +
                result.iterations[2].coupledExtrapolationRejectedCount,
            options.numRobots *
                options.coupledStateGExtrapolationGammas.size());
  EXPECT_EQ(result.iterations[2].cumulativeCommPoseCount,
            result.iterations[1].cumulativeCommPoseCount +
                result.iterations[2].commPoseCount);
}

TEST(testDPGO, ManualDpgoMmCandidateGradientTieBreakKeepsCommunication) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGammas = {0.1, 0.25, 0.5};
  options.localCandidateCostTieTolerance = 1e-3;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations.back().cumulativeCommPoseCount,
            result.iterations[1].cumulativeCommPoseCount +
                result.iterations[2].commPoseCount);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmAndersonAccelerationAttemptsAfterFirstStep) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.localAndersonAcceleration = true;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations[1].andersonAcceptedCount +
                result.iterations[1].andersonRejectedCount,
            0u);
  EXPECT_EQ(result.iterations[2].andersonAcceptedCount +
                result.iterations[2].andersonRejectedCount,
            options.numRobots);
  EXPECT_EQ(result.iterations[2].cumulativeCommPoseCount,
            result.iterations[1].cumulativeCommPoseCount +
                result.iterations[2].commPoseCount);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmMmAcceleratorAndersonEnablesLocalAnderson) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.mmAcceleratorMode = ManualDpgoMmMmAcceleratorMode::Anderson;
  options.localAndersonAcceleration = false;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations[1].andersonAcceptedCount +
                result.iterations[1].andersonRejectedCount,
            0u);
  EXPECT_EQ(result.iterations[2].andersonAcceptedCount +
                result.iterations[2].andersonRejectedCount,
            options.numRobots);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmMmAcceleratorSquaremAttemptsAfterFirstStep) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.mmAcceleratorMode = ManualDpgoMmMmAcceleratorMode::Squarem;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_EQ(result.iterations[1].squaremAcceptedCount +
                result.iterations[1].squaremRejectedCount,
            0u);
  EXPECT_EQ(result.iterations[2].squaremAcceptedCount +
                result.iterations[2].squaremRejectedCount,
            options.numRobots);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmBoundarySafeguardReportsBoundaryProxy) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.mmSafeguard =
      ManualDpgoMmMmSafeguard::LocalSurrogatePlusBoundary;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGammas = {0.1, 0.25};
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  const ManualDpgoMmIterationSummary &last = result.iterations.back();
  EXPECT_TRUE(std::isfinite(last.boundaryEdgeCostBefore));
  EXPECT_TRUE(std::isfinite(last.boundaryEdgeCostAfter));
  EXPECT_TRUE(std::isfinite(last.separatorDeltaNorm));
  EXPECT_GE(last.boundaryEdgeCostBefore, 0.0);
  EXPECT_GE(last.boundaryEdgeCostAfter, 0.0);
  EXPECT_GE(last.separatorDeltaNorm, 0.0);
  EXPECT_EQ(last.extrapolationAcceptedCount +
                last.extrapolationRejectedCount,
            options.numRobots * options.localStateExtrapolationGammas.size());
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmBoundaryProximalCandidateAttemptsWhenEnabled) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.localBoundaryProximalCandidate = true;
  options.localBoundaryProximalWeights = {0.1, 1.0};
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_EQ(result.boundaryProximalCandidateCount,
            options.numRobots * options.localBoundaryProximalWeights.size());
  EXPECT_EQ(result.iterations.back().boundaryProximalCandidateCount,
            result.boundaryProximalCandidateCount);
  EXPECT_LE(result.boundaryProximalAcceptedCount,
            result.boundaryProximalCandidateCount);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmTopologyBoundaryCandidateDefaultOffMatchesBaseline) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseline;
  baseline.numRobots = 2;
  baseline.maxIterations = 1;
  baseline.scheme = ManualDpgoMmScheme::MM;
  baseline.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseline.save = false;
  baseline.trustRegionIterations = 1;
  baseline.trustRegionMaxInnerIterations = 5;
  baseline.parallelLocalSolves = false;
  baseline.printIterationSummary = false;

  const ManualDpgoMmRunResult expected =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseline);

  ManualDpgoMmOptions defaultOff = baseline;
  ASSERT_FALSE(defaultOff.communicationTopologyBoundaryCandidate);
  const ManualDpgoMmRunResult actual =
      runManualDpgoMm("data/tinyGrid3D.g2o", defaultOff);

  ASSERT_EQ(expected.iterations.size(), actual.iterations.size());
  for (std::size_t idx = 0; idx < expected.iterations.size(); ++idx) {
    EXPECT_EQ(expected.iterations[idx].commPoseCount,
              actual.iterations[idx].commPoseCount);
    EXPECT_DOUBLE_EQ(expected.iterations[idx].cumulativeCommMb,
                     actual.iterations[idx].cumulativeCommMb);
  }
  EXPECT_EQ(actual.boundaryProximalCandidateCount, 0u);
  EXPECT_DOUBLE_EQ(expected.finalObjective, actual.finalObjective);
  EXPECT_DOUBLE_EQ(expected.finalGradient, actual.finalGradient);
}

TEST(testDPGO, ManualDpgoMmTopologyBoundaryCandidateAttemptsLocally) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.communicationTopologyBoundaryCandidate = true;
  options.localBoundaryProximalWeights = {0.1, 1.0};
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;
  options.printIterationSummary = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_EQ(result.boundaryProximalCandidateCount,
            options.numRobots * options.localBoundaryProximalWeights.size());
  EXPECT_EQ(result.iterations.back().boundaryProximalCandidateCount,
            result.boundaryProximalCandidateCount);
  EXPECT_LE(result.boundaryProximalAcceptedCount,
            result.boundaryProximalCandidateCount);
  for (const ManualDpgoMmIterationSummary &row : result.iterations) {
    EXPECT_EQ(row.localGradientScalarCount, 0u);
    EXPECT_DOUBLE_EQ(row.localGradientCumulativeScalarCommMb, 0.0);
    EXPECT_EQ(row.globalGradientAcceptedCount, 0u);
    EXPECT_EQ(row.globalGradientRejectedCount, 0u);
  }
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmTopologyBoundaryCandidateRunsWithAmmLazyPlain) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammLazyPlainAfterCertificate = true;
  options.communicationTopologyBoundaryCandidate = true;
  options.localBoundaryProximalWeights = {0.1};
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.printIterationSummary = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_EQ(result.boundaryProximalCandidateCount, options.numRobots);
  EXPECT_EQ(result.iterations.back().boundaryProximalCandidateCount,
            result.boundaryProximalCandidateCount);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmTopologyBoundarySurrogateRunsWithoutExtraPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammLazyPlainAfterCertificate = true;
  options.communicationTopologyBoundarySurrogate = true;
  options.communicationTopologyBoundarySurrogateStaleGain = 1.0;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.printIterationSummary = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_EQ(result.boundaryProximalCandidateCount, 0u);
  for (const ManualDpgoMmIterationSummary &row : result.iterations) {
    EXPECT_EQ(row.localGradientScalarCount, 0u);
    EXPECT_DOUBLE_EQ(row.localGradientCumulativeScalarCommMb, 0.0);
    EXPECT_EQ(row.localGradientCoupledDirectionPacketCount, 0u);
    EXPECT_EQ(row.globalGradientAcceptedCount, 0u);
    EXPECT_EQ(row.globalGradientRejectedCount, 0u);
  }
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmRejectsInvalidTopologyBoundarySurrogateGain) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.communicationTopologyBoundarySurrogate = true;
  options.communicationTopologyBoundarySurrogateStaleGain = -1.0;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmReducedInterfaceModelDefaultOffMatchesBaseline) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseline;
  baseline.numRobots = 2;
  baseline.maxIterations = 1;
  baseline.scheme = ManualDpgoMmScheme::MM;
  baseline.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseline.save = false;
  baseline.trustRegionIterations = 1;
  baseline.trustRegionMaxInnerIterations = 5;
  baseline.parallelLocalSolves = false;
  baseline.printIterationSummary = false;

  const ManualDpgoMmRunResult expected =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseline);

  ManualDpgoMmOptions defaultOff = baseline;
  defaultOff.communicationTopologyReducedInterfaceWeight = 0.5;
  ASSERT_FALSE(defaultOff.communicationTopologyReducedInterfaceModel);
  const ManualDpgoMmRunResult actual =
      runManualDpgoMm("data/tinyGrid3D.g2o", defaultOff);

  ASSERT_EQ(expected.iterations.size(), actual.iterations.size());
  for (std::size_t idx = 0; idx < expected.iterations.size(); ++idx) {
    EXPECT_EQ(expected.iterations[idx].commPoseCount,
              actual.iterations[idx].commPoseCount);
    EXPECT_DOUBLE_EQ(expected.iterations[idx].cumulativeCommMb,
                     actual.iterations[idx].cumulativeCommMb);
  }
  EXPECT_EQ(actual.boundaryProximalCandidateCount, 0u);
  EXPECT_DOUBLE_EQ(expected.finalObjective, actual.finalObjective);
  EXPECT_DOUBLE_EQ(expected.finalGradient, actual.finalGradient);
}

TEST(testDPGO, ManualDpgoMmReducedInterfaceModelRunsWithoutExtraPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammLazyPlainAfterCertificate = true;
  options.communicationTopologyReducedInterfaceModel = true;
  options.communicationTopologyReducedInterfaceWeight = 0.05;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.printIterationSummary = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ManualDpgoMmOptions commBaseline = options;
  commBaseline.communicationTopologyReducedInterfaceModel = false;
  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", commBaseline);

  ASSERT_EQ(result.iterations.size(), 2u);
  ASSERT_EQ(baseline.iterations.size(), result.iterations.size());
  for (std::size_t idx = 0; idx < result.iterations.size(); ++idx) {
    EXPECT_EQ(result.iterations[idx].commPoseCount,
              baseline.iterations[idx].commPoseCount);
    EXPECT_DOUBLE_EQ(result.iterations[idx].cumulativeCommMb,
                     baseline.iterations[idx].cumulativeCommMb);
  }
  EXPECT_EQ(result.boundaryProximalCandidateCount, 0u);
  for (const ManualDpgoMmIterationSummary &row : result.iterations) {
    EXPECT_EQ(row.localGradientScalarCount, 0u);
    EXPECT_DOUBLE_EQ(row.localGradientCumulativeScalarCommMb, 0.0);
    EXPECT_EQ(row.localGradientCoupledDirectionPacketCount, 0u);
    EXPECT_EQ(row.globalGradientAcceptedCount, 0u);
    EXPECT_EQ(row.globalGradientRejectedCount, 0u);
  }
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmReducedInterfaceScheduleRunsWithoutExtraPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammLazyPlainAfterCertificate = true;
  options.communicationTopologyReducedInterfaceModel = true;
  options.communicationTopologyReducedInterfaceWeight = 0.05;
  options.communicationTopologyReducedInterfaceMaxLocalIterations = 1;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.printIterationSummary = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ManualDpgoMmOptions commBaseline = options;
  commBaseline.communicationTopologyReducedInterfaceModel = false;
  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", commBaseline);

  ASSERT_EQ(baseline.iterations.size(), result.iterations.size());
  for (std::size_t idx = 0; idx < result.iterations.size(); ++idx) {
    EXPECT_EQ(result.iterations[idx].commPoseCount,
              baseline.iterations[idx].commPoseCount);
    EXPECT_DOUBLE_EQ(result.iterations[idx].cumulativeCommMb,
                     baseline.iterations[idx].cumulativeCommMb);
  }
  EXPECT_EQ(result.boundaryProximalCandidateCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmReducedInterfaceCandidateRunsWithoutExtraPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammLazyPlainAfterCertificate = true;
  options.communicationTopologyReducedInterfaceModel = true;
  options.communicationTopologyReducedInterfaceWeight = 0.05;
  options.communicationTopologyReducedInterfaceMaxLocalIterations = 1;
  options.communicationTopologyReducedInterfaceCandidate = true;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.printIterationSummary = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ManualDpgoMmOptions commBaseline = options;
  commBaseline.communicationTopologyReducedInterfaceModel = false;
  commBaseline.communicationTopologyReducedInterfaceCandidate = false;
  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", commBaseline);

  ASSERT_EQ(baseline.iterations.size(), result.iterations.size());
  for (std::size_t idx = 0; idx < result.iterations.size(); ++idx) {
    EXPECT_EQ(result.iterations[idx].commPoseCount,
              baseline.iterations[idx].commPoseCount);
    EXPECT_DOUBLE_EQ(result.iterations[idx].cumulativeCommMb,
                     baseline.iterations[idx].cumulativeCommMb);
  }
  EXPECT_EQ(result.boundaryProximalCandidateCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmReducedInterfaceInvalidPathMatchesBaseline) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseline;
  baseline.numRobots = 1;
  baseline.maxIterations = 2;
  baseline.scheme = ManualDpgoMmScheme::AMM;
  baseline.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseline.ammDpgoSurrogateParity = false;
  baseline.communicationTopologyReducedInterfaceWeight = 0.5;
  baseline.save = false;
  baseline.trustRegionIterations = 1;
  baseline.trustRegionMaxInnerIterations = 5;
  baseline.parallelLocalSolves = false;
  baseline.printIterationSummary = false;

  const ManualDpgoMmRunResult expected =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseline);

  ManualDpgoMmOptions enabled = baseline;
  enabled.communicationTopologyReducedInterfaceModel = true;
  const ManualDpgoMmRunResult actual =
      runManualDpgoMm("data/tinyGrid3D.g2o", enabled);

  ASSERT_EQ(expected.iterations.size(), actual.iterations.size());
  for (std::size_t idx = 0; idx < expected.iterations.size(); ++idx) {
    EXPECT_EQ(expected.iterations[idx].commPoseCount,
              actual.iterations[idx].commPoseCount);
    EXPECT_DOUBLE_EQ(expected.iterations[idx].cumulativeCommMb,
                     actual.iterations[idx].cumulativeCommMb);
  }
  EXPECT_DOUBLE_EQ(expected.finalObjective, actual.finalObjective);
  EXPECT_DOUBLE_EQ(expected.finalGradient, actual.finalGradient);
}

TEST(testDPGO, ManualDpgoMmRejectsInvalidReducedInterfaceWeight) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.communicationTopologyReducedInterfaceModel = true;
  options.communicationTopologyReducedInterfaceWeight = -1.0;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmReducedInterfaceRejectsRecursiveSimpleState) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoRecursiveSimpleState = true;
  options.communicationTopologyReducedInterfaceModel = true;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmParallelLocalSolvesMatchSequentialOnTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.recordLocalModelDiagnostics = false;

  options.parallelLocalSolves = false;
  const ManualDpgoMmRunResult sequential =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.parallelLocalSolves = true;
  const ManualDpgoMmRunResult parallel =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(sequential.iterations.size(), parallel.iterations.size());
  EXPECT_NEAR(parallel.finalObjective, sequential.finalObjective, 1e-9);
  EXPECT_NEAR(parallel.finalGradient, sequential.finalGradient, 1e-9);
  EXPECT_EQ(parallel.iterations.back().cumulativeCommPoseCount,
            sequential.iterations.back().cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(parallel.iterations.back().cumulativeCommMb,
                   sequential.iterations.back().cumulativeCommMb);
}

TEST(testDPGO, ManualDpgoMmInitialLocalModelsMatchPgoAgent) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.save = false;

  const std::vector<ManualDpgoMmLocalModelSummary> manual =
      inspectManualDpgoMmInitialLocalModels("data/tinyGrid3D.g2o", options);
  const std::vector<std::unique_ptr<PGOAgent>> referenceAgents =
      buildReferenceAgents("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(manual.size(), referenceAgents.size());
  for (std::size_t robot = 0; robot < manual.size(); ++robot) {
    double referenceCost = 0.0;
    double referenceGrad = 0.0;
    ASSERT_TRUE(referenceAgents[robot]->evaluateLocalModel(referenceCost,
                                                           referenceGrad));
    EXPECT_EQ(manual[robot].robot, robot);
    EXPECT_TRUE(manual[robot].ok);
    EXPECT_NEAR(manual[robot].cost, referenceCost, 1e-9);
    EXPECT_NEAR(manual[robot].gradient, referenceGrad, 1e-9);
  }
}

TEST(testDPGO, ManualDpgoMmInitialSparseQGMatchesReferenceFormula) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.save = false;

  const PartitionedTinyGrid partition =
      buildPartitionedTinyGrid("data/tinyGrid3D.g2o", options);
  const std::vector<ManualDpgoMmLocalModelSnapshot> snapshots =
      inspectManualDpgoMmInitialLocalModelSnapshots("data/tinyGrid3D.g2o",
                                                   options);

  ASSERT_EQ(snapshots.size(), options.numRobots);
  for (const auto &snapshot : snapshots) {
    const unsigned robot = snapshot.robot;
    const unsigned startIdx = robot * partition.posesPerRobot;
    unsigned endIdx = (robot + 1) * partition.posesPerRobot;
    if (robot + 1 == options.numRobots) {
      endIdx = partition.numPoses;
    }
    const unsigned localPoseCount = endIdx - startIdx;

    EXPECT_TRUE(snapshot.ok);
    expectSparseNear(snapshot.q,
                     referenceManualLocalQ(partition, robot, localPoseCount),
                     1e-12);
    expectSparseNear(snapshot.g,
                     referenceManualLocalG(partition, options, robot,
                                           localPoseCount),
                     1e-9);
  }
}

TEST(testDPGO, ManualDpgoMmRunsAmmSchemeWithRestartAccounting) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 4u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
  EXPECT_EQ(result.iterations.front().ammAcceleratedAcceptedCount, 0u);
  EXPECT_GT(result.iterations[2].ammAcceleratedAcceptedCount +
                result.iterations[2].ammRestartCount +
                result.iterations[2].ammPhiFallbackCount +
                result.iterations[2].ammLocalMeritRejectedCount,
            0u);
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
}

TEST(testDPGO, ManualDpgoMmAmmPhiFallbackIsCounted) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammPhi = 1e9;
  options.ammLocalMeritFilter = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_GT(result.iterations[2].ammPhiFallbackCount, 0u);
  EXPECT_EQ(result.iterations[2].ammAcceleratedAcceptedCount, 0u);
}

TEST(testDPGO, ManualDpgoMmAmmZeroGammaScaleMatchesMm) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;

  options.scheme = ManualDpgoMmScheme::MM;
  const ManualDpgoMmRunResult mm =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.scheme = ManualDpgoMmScheme::AMM;
  options.ammGammaScale = 0.0;
  const ManualDpgoMmRunResult amm =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(mm.iterations.size(), amm.iterations.size());
  EXPECT_NEAR(amm.finalObjective, mm.finalObjective, 1e-12);
  EXPECT_NEAR(amm.finalGradient, mm.finalGradient, 1e-12);
  for (const auto &row : amm.iterations) {
    EXPECT_EQ(row.ammAcceleratedAcceptedCount, 0u);
    EXPECT_EQ(row.ammRestartCount, 0u);
  }
}

TEST(testDPGO, ManualDpgoMmAmmHardRestartIsIndependentOfPsi) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = false;
  options.ammPhi = 0.0;

  options.ammPsi = 0.0;
  const ManualDpgoMmRunResult noPsi =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.ammPsi = 1e9;
  const ManualDpgoMmRunResult largePsi =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(noPsi.iterations.size(), largePsi.iterations.size());
  for (std::size_t iter = 0; iter < noPsi.iterations.size(); ++iter) {
    EXPECT_EQ(noPsi.iterations[iter].ammHardRestartCount,
              largePsi.iterations[iter].ammHardRestartCount)
        << "iter=" << iter;
  }
}

TEST(testDPGO, ManualDpgoMmAmmProximalStartIsCounted) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = true;
  options.ammProximalStart = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 4u);
  EXPECT_GT(result.iterations[1].ammProximalStartCount, 0u);
  EXPECT_GE(result.iterations[2].ammProximalStartCount,
            result.iterations[1].ammProximalStartCount);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
}

TEST(testDPGO, ManualDpgoMmAmmTriesConfiguredGammaScales) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = false;
  options.ammGammaScales = {1.0, 0.5, 0.25};

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 4u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  std::size_t ammDecisions = 0;
  for (const auto &row : result.iterations) {
    ammDecisions += row.ammAcceleratedAcceptedCount;
    ammDecisions += row.ammRestartCount;
    ammDecisions += row.ammPhiFallbackCount;
    ammDecisions += row.ammLocalMeritRejectedCount;
  }
  EXPECT_GT(ammDecisions, 0u);
}

TEST(testDPGO, ManualDpgoMmAmmTraceMatchesDpgoMmFields) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.traceAmm = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_amm_trace_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = false;
  options.ammProximalStart = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 4u);

  std::ifstream summary(options.outputDirectory + "/iteration_summary.csv");
  ASSERT_TRUE(summary.good());
  std::string header;
  std::getline(summary, header);
  EXPECT_NE(header.find("initialization_comm_pose_count"), std::string::npos);
  EXPECT_NE(header.find("initialization_comm_mb"), std::string::npos);
  EXPECT_NE(header.find("outer_comm_pose_count"), std::string::npos);
  EXPECT_NE(header.find("outer_comm_mb"), std::string::npos);
  EXPECT_NE(header.find("amm_trace_count"), std::string::npos);
  EXPECT_NE(header.find("amm_prox_reset_count"), std::string::npos);
  EXPECT_NE(header.find("amm_mean_Gkh_initial"), std::string::npos);
  EXPECT_NE(header.find("amm_mean_final_Gk"), std::string::npos);

  std::ifstream trace(options.outputDirectory + "/amm_trace.csv");
  ASSERT_TRUE(trace.good());
  std::getline(trace, header);
  EXPECT_NE(header.find("Gkh_initial"), std::string::npos);
  EXPECT_NE(header.find("minG"), std::string::npos);
  EXPECT_NE(header.find("Gk_after_accelerated"), std::string::npos);
  EXPECT_NE(header.find("Gkh_after_restart_check"), std::string::npos);
  EXPECT_NE(header.find("restart_certificate_lhs"), std::string::npos);
  EXPECT_NE(header.find("restart_certificate_rhs"), std::string::npos);
  EXPECT_NE(header.find("restart_certificate_margin"), std::string::npos);
  EXPECT_NE(header.find("restart_certificate_passed"), std::string::npos);
  EXPECT_NE(header.find("restart_used_xakh"), std::string::npos);
}

TEST(testDPGO, ManualDpgoMmAmmRestartCertificateReportsFiniteMargins) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = false;
  options.ammProximalStart = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t certificateCount = 0;
  std::size_t passedCount = 0;
  std::size_t failedCount = 0;
  double minMargin = std::numeric_limits<double>::infinity();
  for (const auto &row : result.iterations) {
    certificateCount += row.ammRestartCertificateCount;
    passedCount += row.ammRestartCertificatePassedCount;
    failedCount += row.ammRestartCertificateFailedCount;
    if (row.ammRestartCertificateCount > 0) {
      minMargin = std::min(minMargin, row.ammRestartCertificateMinMargin);
    }
  }

  EXPECT_GT(certificateCount, 0u);
  EXPECT_EQ(certificateCount, passedCount + failedCount);
  EXPECT_TRUE(std::isfinite(minMargin));
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmInitialIterSummaryPrintsRestartCertificateZeros) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 0;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.parallelLocalSolves = false;

  testing::internal::CaptureStdout();
  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  const std::string output = testing::internal::GetCapturedStdout();

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_NE(output.find("ITER_SUMMARY iter=0"), std::string::npos);
  EXPECT_NE(output.find("amm_restart_certificate_count=0"),
            std::string::npos);
  EXPECT_NE(output.find("amm_restart_certificate_passed_count=0"),
            std::string::npos);
  EXPECT_NE(output.find("amm_restart_certificate_failed_count=0"),
            std::string::npos);
  EXPECT_NE(output.find("amm_restart_certificate_min_margin=0"),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmAmmRestartCertificateTraceMatchesSummary) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = true;
  options.traceAmm = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_amm_cert_trace_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 8;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = false;
  options.ammProximalStart = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 5u);

  struct CertificateAggregate {
    std::size_t count{0};
    std::size_t passed{0};
    std::size_t failed{0};
    double minMargin{std::numeric_limits<double>::infinity()};
  };
  std::map<unsigned, CertificateAggregate> byIter;

  std::ifstream trace(options.outputDirectory + "/amm_trace.csv");
  ASSERT_TRUE(trace.good());
  std::string line;
  ASSERT_TRUE(static_cast<bool>(std::getline(trace, line)));
  const std::vector<std::string> traceHeader = splitCsvLine(line);
  while (std::getline(trace, line)) {
    if (line.empty()) {
      continue;
    }
    const auto row = csvRowByHeader(traceHeader, splitCsvLine(line));
    ASSERT_TRUE(row.count("restart_certificate_valid"));
    ASSERT_TRUE(row.count("restart_certificate_passed"));
    const bool valid = std::stoi(row.at("restart_certificate_valid")) != 0;
    if (!valid) {
      continue;
    }
    const double lhs = std::stod(row.at("restart_certificate_lhs"));
    const double rhs = std::stod(row.at("restart_certificate_rhs"));
    const double margin = std::stod(row.at("restart_certificate_margin"));
    EXPECT_NEAR(margin, lhs - rhs,
                1e-9 * std::max(1.0, std::abs(lhs - rhs)));
    const bool hardRestart = std::stoi(row.at("hard_restart")) != 0;
    const bool softRestart = std::stoi(row.at("soft_restart")) != 0;
    const bool passed = std::stoi(row.at("restart_certificate_passed")) != 0;
    EXPECT_EQ(passed, margin >= -1e-12 && !hardRestart && !softRestart);

    CertificateAggregate &aggregate =
        byIter[static_cast<unsigned>(std::stoul(row.at("iter")))];
    ++aggregate.count;
    aggregate.passed += passed ? 1 : 0;
    aggregate.failed += passed ? 0 : 1;
    aggregate.minMargin = std::min(aggregate.minMargin, margin);
  }

  std::size_t totalCertificates = 0;
  for (const auto &entry : byIter) {
    totalCertificates += entry.second.count;
  }
  ASSERT_GT(totalCertificates, 0u);

  std::ifstream summary(options.outputDirectory + "/iteration_summary.csv");
  ASSERT_TRUE(summary.good());
  ASSERT_TRUE(static_cast<bool>(std::getline(summary, line)));
  const std::vector<std::string> summaryHeader = splitCsvLine(line);
  while (std::getline(summary, line)) {
    if (line.empty()) {
      continue;
    }
    const auto row = csvRowByHeader(summaryHeader, splitCsvLine(line));
    const unsigned iter = static_cast<unsigned>(std::stoul(row.at("iter")));
    const auto found = byIter.find(iter);
    if (found == byIter.end()) {
      continue;
    }
    const CertificateAggregate &aggregate = found->second;
    EXPECT_EQ(std::stoull(row.at("amm_restart_certificate_count")),
              aggregate.count);
    EXPECT_EQ(std::stoull(row.at("amm_restart_certificate_passed_count")),
              aggregate.passed);
    EXPECT_EQ(std::stoull(row.at("amm_restart_certificate_failed_count")),
              aggregate.failed);
    EXPECT_NEAR(std::stod(row.at("amm_restart_certificate_min_margin")),
                aggregate.minMargin,
                1e-9 * std::max(1.0, std::abs(aggregate.minMargin)));
  }
}

TEST(testDPGO, ManualDpgoMmAmmDpgoRestartFallbackUsesProxResetGate) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = false;
  options.ammProximalStart = true;
  options.ammDpgoRestartFallback = true;
  options.ammPsi = 1e9;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t proxResetCount = 0;
  for (const auto &row : result.iterations) {
    proxResetCount += row.ammProxResetCount;
  }
  EXPECT_GT(proxResetCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmAmmGammaPortfolioRespectsSurrogateRestartGate) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = false;
  options.ammDpgoRestartFallback = true;
  options.ammGammaScale = 10.0;

  const ManualDpgoMmRunResult singleGamma =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  options.ammGammaScales = {1.0, 0.1, 0.01};
  const ManualDpgoMmRunResult gammaPortfolio =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  auto hardRestartCount = [](const ManualDpgoMmRunResult &result) {
    std::size_t count = 0;
    for (const auto &row : result.iterations) {
      count += row.ammHardRestartCount;
    }
    return count;
  };
  EXPECT_LE(hardRestartCount(gammaPortfolio),
            hardRestartCount(singleGamma));
  EXPECT_TRUE(std::isfinite(gammaPortfolio.finalObjective));
  EXPECT_TRUE(std::isfinite(gammaPortfolio.finalGradient));
}

TEST(testDPGO, ManualDpgoMmAmmDpgoParityUsesSingleDpgoGamma) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = true;
  options.ammDpgoSurrogateParity = true;
  options.ammGammaScales = {0.0};

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t traceCount = 0;
  for (const auto &row : result.iterations) {
    traceCount += row.ammTraceCount;
  }
  EXPECT_GT(traceCount, 0u);
  ASSERT_GT(result.iterations.size(), 1u);
  EXPECT_GT(result.iterations[1].ammProximalStartCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmAmmDpgoParityBypassesExtraLocalMeritGate) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 6;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = true;
  options.ammDpgoSurrogateParity = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t localMeritRejected = 0;
  std::size_t acceleratedAccepted = 0;
  for (const auto &row : result.iterations) {
    localMeritRejected += row.ammLocalMeritRejectedCount;
    acceleratedAccepted += row.ammAcceleratedAcceptedCount;
  }
  EXPECT_EQ(localMeritRejected, 0u);
  EXPECT_GT(acceleratedAccepted, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO,
     ManualDpgoMmAmmDpgoParityDoesNotExplodeWithoutGlobalFeedback) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 10;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 20;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_FALSE(result.iterations.empty());
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective,
            2.0 * result.iterations.front().globalCost);
}

TEST(testDPGO, ManualDpgoMmAmmDpgoParityUsesSimpleSurrogateOnTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 10;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 20;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, 19.75);
}

TEST(testDPGO, ManualDpgoMmMixedSurrogatePortfolioRequiresParity) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.ammDpgoMixedSurrogatePortfolio = true;

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmMixedSurrogatePortfolioRunsOnTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
}

TEST(testDPGO, ManualDpgoMmMixedSurrogatePortfolioReportsSelectionSource) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.localStateExtrapolation = true;
  options.localStateExtrapolationGammas = {0.1, 0.25};

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_GE(result.iterations.size(), 2u);
  EXPECT_EQ(result.iterations.front().ammMixedSurrogateCandidateCount, 0u);
  for (std::size_t i = 1; i < result.iterations.size(); ++i) {
    const auto &row = result.iterations[i];
    EXPECT_EQ(row.ammMixedSurrogateCandidateCount, options.numRobots);
    const std::size_t selected =
        row.ammMixedSurrogateSimpleSelectedCount +
        row.ammMixedSurrogateTrueLocalSelectedCount +
        row.ammMixedSurrogateExtrapolatedSelectedCount +
        row.ammMixedSurrogateOtherSelectedCount;
    EXPECT_EQ(selected, options.numRobots);
  }
}

TEST(testDPGO, ManualDpgoMmMixedSurrogateCanSkipSimpleAfterTrueLocalStreak) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak = 1;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_GE(result.iterations.size(), 3u);
  std::size_t skipped = 0;
  for (std::size_t i = 1; i < result.iterations.size(); ++i) {
    const auto &row = result.iterations[i];
    skipped += row.ammMixedSurrogateSimpleSkippedCount;
    const std::size_t selected =
        row.ammMixedSurrogateSimpleSelectedCount +
        row.ammMixedSurrogateTrueLocalSelectedCount +
        row.ammMixedSurrogateExtrapolatedSelectedCount +
        row.ammMixedSurrogateOtherSelectedCount;
    EXPECT_EQ(selected, options.numRobots);
  }
  EXPECT_GT(skipped, 0u);
}

TEST(testDPGO,
     ManualDpgoMmMixedSurrogateCanForceSimpleRefreshAfterSkippedRounds) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 5;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak = 1;
  options.ammMixedSurrogateForceSimpleEverySkippedRounds = 1;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_GE(result.iterations.size(), 4u);
  std::size_t skipped = 0;
  std::size_t forcedRefreshes = 0;
  for (std::size_t i = 1; i < result.iterations.size(); ++i) {
    const auto &row = result.iterations[i];
    skipped += row.ammMixedSurrogateSimpleSkippedCount;
    forcedRefreshes += row.ammMixedSurrogateSimpleForcedRefreshCount;
  }
  EXPECT_GT(skipped, 0u);
  EXPECT_GT(forcedRefreshes, 0u);
}

TEST(testDPGO, ManualDpgoMmAmmDpgoParityImprovesOverPlainMmOnTinyGrid) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 10;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 20;
  options.parallelLocalSolves = false;

  options.scheme = ManualDpgoMmScheme::MM;
  const ManualDpgoMmRunResult mm =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.scheme = ManualDpgoMmScheme::AMM;
  options.ammDpgoSurrogateParity = true;
  const ManualDpgoMmRunResult amm =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  EXPECT_TRUE(std::isfinite(mm.finalObjective));
  EXPECT_TRUE(std::isfinite(amm.finalObjective));
  EXPECT_LT(amm.finalObjective, mm.finalObjective - 0.1);
}

TEST(testDPGO, ManualDpgoMmGlobalStateExtrapolationIsSafeguarded) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.globalStateExtrapolation = true;
  options.globalStateExtrapolationGammas = {0.25, 0.5};

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 4u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_LE(result.finalObjective, result.iterations.front().globalCost);
  std::size_t attempted = 0;
  for (const auto &row : result.iterations) {
    attempted += row.globalExtrapolationAcceptedCount;
    attempted += row.globalExtrapolationRejectedCount;
  }
  EXPECT_GT(attempted, 0u);
}

TEST(testDPGO, ManualDpgoMmLocalGradientCorrectionUsesLocalMeritOnly) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrection = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  const ManualDpgoMmRunResult corrected =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t accepted = 0;
  std::size_t rejected = 0;
  std::size_t globalAccepted = 0;
  std::size_t scalarCount = 0;
  for (const auto &row : corrected.iterations) {
    accepted += row.localGradientAcceptedCount;
    rejected += row.localGradientRejectedCount;
    globalAccepted += row.globalGradientAcceptedCount;
    scalarCount += row.localGradientScalarCount;
  }
  EXPECT_GT(accepted + rejected, 0u);
  EXPECT_GT(accepted, 0u);
  EXPECT_EQ(globalAccepted, 0u);
  EXPECT_EQ(scalarCount, 0u);
  EXPECT_TRUE(std::isfinite(baseline.finalObjective));
  EXPECT_TRUE(std::isfinite(corrected.finalObjective));
  EXPECT_LE(corrected.finalObjective, baseline.finalObjective + 1e-10);
}

TEST(testDPGO, ManualDpgoMmSharedLocalGradientCorrectionUsesAggregateMerit) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrection = true;
  options.localGradientCorrectionSharedStep = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  const std::size_t expectedRoundScalars =
      options.numRobots *
      (options.localGradientCorrectionSteps.size() + 2u);
  const std::size_t expectedTotalScalars =
      expectedRoundScalars * options.maxIterations;
  const double expectedTotalScalarMb =
      static_cast<double>(expectedTotalScalars * sizeof(double)) /
      (1024.0 * 1024.0);
  const ManualDpgoMmRunResult corrected =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t accepted = 0;
  std::size_t rejected = 0;
  std::size_t globalAccepted = 0;
  std::size_t scalarCount = 0;
  double scalarMb = 0.0;
  double finalScalarMb = 0.0;
  bool sawMixedPayload = false;
  for (const auto &row : corrected.iterations) {
    accepted += row.localGradientAcceptedCount;
    rejected += row.localGradientRejectedCount;
    globalAccepted += row.globalGradientAcceptedCount;
    scalarCount += row.localGradientScalarCount;
    scalarMb += row.localGradientIterScalarCommMb;
    finalScalarMb = row.localGradientCumulativeScalarCommMb;
    if (row.iter == 0) {
      EXPECT_EQ(row.localGradientScalarCount, 0u);
    } else {
      EXPECT_EQ(row.localGradientScalarCount, expectedRoundScalars);
    }
    EXPECT_NEAR(row.cumulativeCommMb,
                row.poseCumulativeCommMb +
                    row.localGradientCumulativeScalarCommMb,
                1e-15);
    sawMixedPayload = sawMixedPayload ||
                      (row.cumulativeCommMb > row.poseCumulativeCommMb);
  }
  EXPECT_GT(accepted + rejected, 0u);
  EXPECT_GT(accepted, 0u);
  EXPECT_EQ(globalAccepted, 0u);
  EXPECT_EQ(scalarCount, expectedTotalScalars);
  EXPECT_NEAR(scalarMb, expectedTotalScalarMb, 1e-15);
  EXPECT_NEAR(finalScalarMb, expectedTotalScalarMb, 1e-15);
  EXPECT_TRUE(sawMixedPayload);
  EXPECT_TRUE(std::isfinite(baseline.finalObjective));
  EXPECT_TRUE(std::isfinite(corrected.finalObjective));
  EXPECT_LE(corrected.finalObjective, baseline.finalObjective + 1e-10);
}

TEST(testDPGO, ManualDpgoMmNeighborhoodLocalGradientCorrectionUsesSparseMerit) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrection = true;
  options.localGradientCorrectionNeighborhoodStep = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  const ManualDpgoMmRunResult corrected =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t accepted = 0;
  std::size_t rejected = 0;
  std::size_t scalarCount = 0;
  bool sawSparseScalarPayload = false;
  for (const auto &row : corrected.iterations) {
    accepted += row.localGradientAcceptedCount;
    rejected += row.localGradientRejectedCount;
    scalarCount += row.localGradientScalarCount;
    if (row.iter > 0 && row.localGradientScalarCount > 0) {
      sawSparseScalarPayload = true;
      EXPECT_EQ(row.localGradientScalarCount %
                    (options.localGradientCorrectionSteps.size() + 2u),
                0u);
    }
    EXPECT_NEAR(row.cumulativeCommMb,
                row.poseCumulativeCommMb +
                    row.localGradientCumulativeScalarCommMb,
                1e-15);
  }
  EXPECT_GT(accepted + rejected, 0u);
  EXPECT_GT(accepted, 0u);
  EXPECT_GT(scalarCount, 0u);
  EXPECT_TRUE(sawSparseScalarPayload);
  EXPECT_TRUE(std::isfinite(baseline.finalObjective));
  EXPECT_TRUE(std::isfinite(corrected.finalObjective));
  EXPECT_LE(corrected.finalObjective, baseline.finalObjective + 1e-10);
}

TEST(testDPGO,
     ManualDpgoMmNeighborhoodCurvatureCorrectionUsesCurvatureScalars) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrection = true;
  options.localGradientCorrectionNeighborhoodStep = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  const ManualDpgoMmRunResult corrected =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t accepted = 0;
  std::size_t rejected = 0;
  std::size_t scalarCount = 0;
  bool sawCurvatureScalarPayload = false;
  for (const auto &row : corrected.iterations) {
    accepted += row.localGradientAcceptedCount;
    rejected += row.localGradientRejectedCount;
    scalarCount += row.localGradientScalarCount;
    if (row.iter > 0 && row.localGradientScalarCount > 0) {
      sawCurvatureScalarPayload = true;
      EXPECT_EQ(row.localGradientScalarCount % 2u, 0u);
      EXPECT_LT(row.localGradientScalarCount,
                2u * (options.localGradientCorrectionSteps.size() + 2u));
      EXPECT_GT(row.localGradientSelectedStep, 0.0);
    }
    EXPECT_NEAR(row.cumulativeCommMb,
                row.poseCumulativeCommMb +
                    row.localGradientCumulativeScalarCommMb,
                1e-15);
  }
  EXPECT_GT(accepted + rejected, 0u);
  EXPECT_GT(accepted, 0u);
  EXPECT_GT(scalarCount, 0u);
  EXPECT_TRUE(sawCurvatureScalarPayload);
  EXPECT_TRUE(std::isfinite(baseline.finalObjective));
  EXPECT_TRUE(std::isfinite(corrected.finalObjective));
  EXPECT_LE(corrected.finalObjective, baseline.finalObjective + 1e-10);
}

TEST(testDPGO,
     ManualDpgoMmBoundaryOnlyCurvatureCorrectionTouchesSeparatorModel) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionNeighborhoodStep = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionBoundaryOnly = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};

  const ManualDpgoMmRunResult corrected =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t accepted = 0;
  std::size_t scalarCount = 0;
  double maxSelectedStep = 0.0;
  for (const auto &row : corrected.iterations) {
    accepted += row.localGradientAcceptedCount;
    scalarCount += row.localGradientScalarCount;
    maxSelectedStep = std::max(maxSelectedStep,
                               row.localGradientSelectedStep);
    if (row.iter > 0 && row.localGradientScalarCount > 0) {
      EXPECT_EQ(row.localGradientScalarCount % 2u, 0u);
    }
  }
  EXPECT_GT(accepted, 0u);
  EXPECT_GT(scalarCount, 0u);
  EXPECT_GT(maxSelectedStep, 0.0);
  EXPECT_TRUE(std::isfinite(corrected.finalObjective));
}

TEST(testDPGO,
     ManualDpgoMmBlockJacobiCurvatureCorrectionUsesCurvatureScalars) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrection = true;
  options.localGradientCorrectionNeighborhoodStep = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionBlockJacobi = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  const ManualDpgoMmRunResult corrected =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t accepted = 0;
  std::size_t scalarCount = 0;
  for (const auto &row : corrected.iterations) {
    accepted += row.localGradientAcceptedCount;
    scalarCount += row.localGradientScalarCount;
    if (row.iter > 0 && row.localGradientScalarCount > 0) {
      EXPECT_EQ(row.localGradientScalarCount % 2u, 0u);
      EXPECT_GT(row.localGradientSelectedStep, 0.0);
    }
  }
  EXPECT_GT(accepted, 0u);
  EXPECT_GT(scalarCount, 0u);
  EXPECT_TRUE(std::isfinite(baseline.finalObjective));
  EXPECT_TRUE(std::isfinite(corrected.finalObjective));
  EXPECT_LE(corrected.finalObjective, baseline.finalObjective + 1e-10);
}

TEST(testDPGO,
     ManualDpgoMmCompactSchurCurvatureCorrectionUsesCurvatureScalars) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrection = true;
  options.localGradientCorrectionNeighborhoodStep = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionCompactSchur = true;
  options.localGradientCorrectionCompactSchurMaxPrivateCols = 8;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  const ManualDpgoMmRunResult corrected =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t accepted = 0;
  std::size_t scalarCount = 0;
  std::size_t compactSchurCandidates = 0;
  std::size_t compactSchurBoundaryCols = 0;
  std::size_t compactSchurPrivateCols = 0;
  for (const auto &row : corrected.iterations) {
    accepted += row.localGradientAcceptedCount;
    scalarCount += row.localGradientScalarCount;
    compactSchurCandidates += row.localGradientCompactSchurCandidateCount;
    compactSchurBoundaryCols += row.localGradientCompactSchurBoundaryColCount;
    compactSchurPrivateCols += row.localGradientCompactSchurPrivateColCount;
    if (row.iter > 0 && row.localGradientScalarCount > 0) {
      EXPECT_EQ(row.localGradientScalarCount % 2u, 0u);
      EXPECT_GT(row.localGradientSelectedStep, 0.0);
    }
  }
  EXPECT_GT(accepted, 0u);
  EXPECT_GT(scalarCount, 0u);
  EXPECT_GT(compactSchurCandidates, 0u);
  EXPECT_GT(compactSchurBoundaryCols, 0u);
  EXPECT_GT(compactSchurPrivateCols, 0u);
  EXPECT_TRUE(std::isfinite(baseline.finalObjective));
  EXPECT_TRUE(std::isfinite(corrected.finalObjective));
  EXPECT_LE(corrected.finalObjective, baseline.finalObjective + 1e-10);
}

TEST(testDPGO, ManualDpgoMmCompactSchurDoesNotOverrideBetterBlockJacobi) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionNeighborhoodStep = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionBlockJacobi = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};

  const ManualDpgoMmRunResult blockJacobi =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrectionCompactSchur = true;
  options.localGradientCorrectionCompactSchurMaxPrivateCols = 8;
  const ManualDpgoMmRunResult guardedSchur =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  std::size_t compactSchurCandidates = 0;
  std::size_t compactSchurGuardRejected = 0;
  for (const auto &row : guardedSchur.iterations) {
    compactSchurCandidates += row.localGradientCompactSchurCandidateCount;
    compactSchurGuardRejected +=
        row.localGradientCompactSchurGuardRejectedCount;
  }

  EXPECT_GT(compactSchurCandidates, 0u);
  EXPECT_GT(compactSchurGuardRejected, 0u);
  EXPECT_TRUE(std::isfinite(blockJacobi.finalObjective));
  EXPECT_TRUE(std::isfinite(guardedSchur.finalObjective));
  EXPECT_LE(guardedSchur.finalObjective,
            blockJacobi.finalObjective + 1e-10);
}

TEST(testDPGO,
     ManualDpgoMmCoupledCurvatureDirectionUsesBoundaryDirectionPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionCoupledDirection = true;
  options.localGradientCorrectionBlockJacobi = true;
  options.localGradientCorrectionStep = 0.01;

  const ManualDpgoMmRunResult coupled =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(coupled.iterations.size(), 3u);
  bool sawPackets = false;
  bool sawCandidate = false;
  for (std::size_t idx = 1; idx < coupled.iterations.size(); ++idx) {
    const auto &row = coupled.iterations[idx];
    if (row.localGradientCoupledDirectionPacketCount > 0) {
      sawPackets = true;
      EXPECT_EQ(row.localGradientCoupledDirectionScalarCount,
                row.localGradientCoupledDirectionPacketCount * 12u);
      EXPECT_EQ(row.localGradientCoupledDirectionSkippedPacketCount, 0u);
      EXPECT_EQ(row.localGradientScalarCount,
                row.localGradientCoupledDirectionScalarCount);
      EXPECT_NEAR(row.localGradientIterScalarCommMb,
                  row.localGradientCoupledDirectionScalarCommMb, 1e-15);
    }
    if (row.localGradientCoupledDirectionCandidateCount > 0) {
      sawCandidate = true;
    }
  }
  EXPECT_TRUE(sawPackets);
  EXPECT_TRUE(sawCandidate);
  EXPECT_TRUE(std::isfinite(coupled.finalObjective));
}

TEST(testDPGO,
     ManualDpgoMmCanBudgetCoupledCurvatureDirectionPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionCoupledDirection = true;
  options.localGradientCorrectionBlockJacobi = true;
  options.localGradientCorrectionStep = 0.01;

  const ManualDpgoMmRunResult full =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrectionCoupledDirectionBudgetFraction = 0.5;
  options.localGradientCorrectionCoupledDirectionMaxPacketsPerReceiver = 1;
  const ManualDpgoMmRunResult budgeted =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(full.iterations.size(), budgeted.iterations.size());
  bool sawBudgetedSkip = false;
  for (std::size_t idx = 1; idx < budgeted.iterations.size(); ++idx) {
    EXPECT_LT(budgeted.iterations[idx]
                  .localGradientCoupledDirectionPacketCount,
              full.iterations[idx].localGradientCoupledDirectionPacketCount);
    EXPECT_LT(budgeted.iterations[idx]
                  .localGradientCoupledDirectionScalarCount,
              full.iterations[idx].localGradientCoupledDirectionScalarCount);
    if (budgeted.iterations[idx]
            .localGradientCoupledDirectionSkippedPacketCount > 0) {
      sawBudgetedSkip = true;
    }
  }
  EXPECT_TRUE(sawBudgetedSkip);
  EXPECT_TRUE(std::isfinite(budgeted.finalObjective));

  options.localGradientCorrectionCoupledDirectionBudgetFraction = 1.1;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     ManualDpgoMmCanSparseCoupledCurvatureDirectionPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionCoupledDirection = true;
  options.localGradientCorrectionBlockJacobi = true;
  options.localGradientCorrectionStep = 0.01;

  const ManualDpgoMmRunResult full =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrectionCoupledDirectionTopKEntries = 2;
  const ManualDpgoMmRunResult sparse =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(full.iterations.size(), sparse.iterations.size());
  bool sawSparsePackets = false;
  for (std::size_t idx = 1; idx < sparse.iterations.size(); ++idx) {
    const auto &fullRow = full.iterations[idx];
    const auto &sparseRow = sparse.iterations[idx];
    if (sparseRow.localGradientCoupledDirectionPacketCount > 0) {
      sawSparsePackets = true;
      EXPECT_EQ(sparseRow.localGradientCoupledDirectionPacketCount,
                fullRow.localGradientCoupledDirectionPacketCount);
      EXPECT_EQ(sparseRow.localGradientCoupledDirectionSkippedPacketCount,
                fullRow.localGradientCoupledDirectionSkippedPacketCount);
      EXPECT_EQ(sparseRow.localGradientCoupledDirectionScalarCount,
                sparseRow.localGradientCoupledDirectionPacketCount * 4u);
      EXPECT_LT(sparseRow.localGradientCoupledDirectionScalarCount,
                fullRow.localGradientCoupledDirectionScalarCount);
      const double expectedMb =
          static_cast<double>(
              sparseRow.localGradientCoupledDirectionScalarCount *
              sizeof(double)) /
          (1024.0 * 1024.0);
      EXPECT_NEAR(sparseRow.localGradientCoupledDirectionScalarCommMb,
                  expectedMb, 1e-15);
      EXPECT_NEAR(sparseRow.localGradientIterScalarCommMb,
                  sparseRow.localGradientCoupledDirectionScalarCommMb,
                  1e-15);
    }
  }
  EXPECT_TRUE(sawSparsePackets);
  EXPECT_TRUE(std::isfinite(sparse.finalObjective));

  options.localGradientCorrectionCoupledDirectionTopKEntries = 99;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     ManualDpgoMmCanEventFilterCoupledCurvatureDirectionPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionCoupledDirection = true;
  options.localGradientCorrectionBlockJacobi = true;
  options.localGradientCorrectionStep = 0.01;

  const ManualDpgoMmRunResult full =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrectionCoupledDirectionMinScore = 1e100;
  const ManualDpgoMmRunResult filtered =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(full.iterations.size(), filtered.iterations.size());
  bool sawScoreFilteredPackets = false;
  for (std::size_t idx = 1; idx < filtered.iterations.size(); ++idx) {
    const auto &fullRow = full.iterations[idx];
    const auto &filteredRow = filtered.iterations[idx];
    if (fullRow.localGradientCoupledDirectionPacketCount > 0) {
      sawScoreFilteredPackets = true;
      EXPECT_EQ(filteredRow.localGradientCoupledDirectionPacketCount, 0u);
      EXPECT_EQ(filteredRow.localGradientCoupledDirectionScalarCount, 0u);
      EXPECT_EQ(filteredRow.localGradientCoupledDirectionScoreSkippedPacketCount,
                fullRow.localGradientCoupledDirectionPacketCount);
      EXPECT_EQ(filteredRow.localGradientCoupledDirectionSkippedPacketCount,
                filteredRow
                    .localGradientCoupledDirectionScoreSkippedPacketCount);
    }
  }
  EXPECT_TRUE(sawScoreFilteredPackets);
  EXPECT_TRUE(std::isfinite(filtered.finalObjective));

  options.localGradientCorrectionCoupledDirectionMinScore = -1.0;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
  options.localGradientCorrectionCoupledDirectionMinScore = 0.0;
  options.localGradientCorrectionCoupledDirectionMinScoreRatio = 1.1;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO,
     ManualDpgoMmCanByteBudgetCoupledCurvatureDirectionPackets) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionCurvatureStep = true;
  options.localGradientCorrectionCoupledDirection = true;
  options.localGradientCorrectionBlockJacobi = true;
  options.localGradientCorrectionStep = 0.01;

  const ManualDpgoMmRunResult full =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  const double oneDensePacketMb =
      static_cast<double>(12u * sizeof(double)) / (1024.0 * 1024.0);
  options.localGradientCorrectionCoupledDirectionByteBudgetMb =
      oneDensePacketMb;
  const ManualDpgoMmRunResult denseBudget =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrectionCoupledDirectionTopKEntries = 1;
  const ManualDpgoMmRunResult sparseBudget =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(full.iterations.size(), denseBudget.iterations.size());
  ASSERT_EQ(full.iterations.size(), sparseBudget.iterations.size());
  bool sawDenseByteBudget = false;
  bool sawSparseSendsMorePackets = false;
  for (std::size_t idx = 1; idx < full.iterations.size(); ++idx) {
    const auto &fullRow = full.iterations[idx];
    const auto &denseRow = denseBudget.iterations[idx];
    const auto &sparseRow = sparseBudget.iterations[idx];
    if (fullRow.localGradientCoupledDirectionPacketCount > 0) {
      sawDenseByteBudget = true;
      EXPECT_LT(denseRow.localGradientCoupledDirectionPacketCount,
                fullRow.localGradientCoupledDirectionPacketCount);
      EXPECT_GT(denseRow.localGradientCoupledDirectionByteBudgetSkippedPacketCount,
                0u);
      EXPECT_LE(denseRow.localGradientCoupledDirectionScalarCount, 24u);
      EXPECT_LE(sparseRow.localGradientCoupledDirectionScalarCount, 24u);
      if (sparseRow.localGradientCoupledDirectionPacketCount >
          denseRow.localGradientCoupledDirectionPacketCount) {
        sawSparseSendsMorePackets = true;
      }
    }
  }
  EXPECT_TRUE(sawDenseByteBudget);
  EXPECT_TRUE(sawSparseSendsMorePackets);
  EXPECT_TRUE(std::isfinite(denseBudget.finalObjective));
  EXPECT_TRUE(std::isfinite(sparseBudget.finalObjective));

  options.localGradientCorrectionCoupledDirectionByteBudgetMb = -1.0;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmFreshLocalGradientCorrectionCountsPreExchange) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionSharedStep = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};

  const ManualDpgoMmRunResult stale =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrectionFreshNeighborExchange = true;
  const ManualDpgoMmRunResult fresh =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(stale.iterations.size(), fresh.iterations.size());
  ASSERT_EQ(fresh.iterations.size(), 3u);
  EXPECT_EQ(fresh.iterations[0].localGradientFreshPoseCount, 0u);
  for (std::size_t idx = 1; idx < fresh.iterations.size(); ++idx) {
    EXPECT_EQ(fresh.iterations[idx].localGradientFreshPoseCount,
              stale.iterations[idx].commPoseCount);
    EXPECT_EQ(fresh.iterations[idx].commPoseCount,
              stale.iterations[idx].commPoseCount +
                  fresh.iterations[idx].localGradientFreshPoseCount);
    EXPECT_NEAR(fresh.iterations[idx].poseIterCommMb,
                stale.iterations[idx].poseIterCommMb +
                    fresh.iterations[idx].localGradientFreshPoseCommMb,
                1e-15);
  }
  EXPECT_GT(fresh.iterations.back().cumulativeCommPoseCount,
            stale.iterations.back().cumulativeCommPoseCount);
  EXPECT_GT(fresh.iterations.back().poseCumulativeCommMb,
            stale.iterations.back().poseCumulativeCommMb);
  EXPECT_TRUE(std::isfinite(fresh.finalObjective));
}

TEST(testDPGO, ManualDpgoMmLocalGradientInnerRoundsDoNotAddPoseExchange) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionSharedStep = true;
  options.localGradientCorrectionFreshNeighborExchange = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};

  const ManualDpgoMmRunResult oneInner =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrectionInnerRounds = 3;
  const ManualDpgoMmRunResult multiInner =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  const std::size_t expectedRoundScalars =
      options.numRobots *
      (options.localGradientCorrectionSteps.size() + 2u);
  ASSERT_EQ(oneInner.iterations.size(), multiInner.iterations.size());
  for (std::size_t idx = 1; idx < multiInner.iterations.size(); ++idx) {
    EXPECT_EQ(multiInner.iterations[idx].localGradientFreshPoseCount,
              oneInner.iterations[idx].localGradientFreshPoseCount);
    EXPECT_EQ(multiInner.iterations[idx].commPoseCount,
              oneInner.iterations[idx].commPoseCount);
    EXPECT_GE(multiInner.iterations[idx].localGradientInnerRoundCount, 1u);
    EXPECT_LE(multiInner.iterations[idx].localGradientInnerRoundCount, 3u);
    EXPECT_EQ(multiInner.iterations[idx].localGradientScalarCount,
              expectedRoundScalars *
                  multiInner.iterations[idx].localGradientInnerRoundCount);
    EXPECT_NEAR(multiInner.iterations[idx].cumulativeCommMb,
                multiInner.iterations[idx].poseCumulativeCommMb +
                    multiInner.iterations[idx]
                        .localGradientCumulativeScalarCommMb,
                1e-15);
  }
  EXPECT_TRUE(std::isfinite(multiInner.finalObjective));
}

TEST(testDPGO, ManualDpgoMmCanSkipPostExchangeAfterFreshLocalCorrection) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionSharedStep = true;
  options.localGradientCorrectionFreshNeighborExchange = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};

  const ManualDpgoMmRunResult defaultExchange =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.postExchangePeriod = 0;
  const ManualDpgoMmRunResult delayedExchange =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(defaultExchange.iterations.size(),
            delayedExchange.iterations.size());
  for (std::size_t idx = 1; idx < delayedExchange.iterations.size(); ++idx) {
    EXPECT_GT(defaultExchange.iterations[idx].postExchangePoseCount, 0u);
    EXPECT_EQ(delayedExchange.iterations[idx].postExchangePoseCount, 0u);
    EXPECT_EQ(delayedExchange.iterations[idx].localGradientFreshPoseCount,
              defaultExchange.iterations[idx].localGradientFreshPoseCount);
    EXPECT_EQ(delayedExchange.iterations[idx].commPoseCount,
              delayedExchange.iterations[idx].localGradientFreshPoseCount);
    EXPECT_LT(delayedExchange.iterations[idx].commPoseCount,
              defaultExchange.iterations[idx].commPoseCount);
    EXPECT_NEAR(delayedExchange.iterations[idx].poseIterCommMb,
                delayedExchange.iterations[idx]
                    .localGradientFreshPoseCommMb,
                1e-15);
  }
  EXPECT_LT(delayedExchange.iterations.back().poseCumulativeCommMb,
            defaultExchange.iterations.back().poseCumulativeCommMb);
  EXPECT_TRUE(std::isfinite(delayedExchange.finalObjective));

  options.localGradientCorrectionFreshNeighborExchange = false;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmCanBudgetFreshLocalGradientExchange) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionSharedStep = true;
  options.localGradientCorrectionFreshNeighborExchange = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  options.postExchangePeriod = 0;

  const ManualDpgoMmRunResult fullFresh =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.localGradientCorrectionFreshBudgetFraction = 0.5;
  options.localGradientCorrectionFreshMaxPosesPerReceiver = 1;
  const ManualDpgoMmRunResult budgetedFresh =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(fullFresh.iterations.size(), budgetedFresh.iterations.size());
  bool sawBudgetedSkip = false;
  for (std::size_t idx = 1; idx < budgetedFresh.iterations.size(); ++idx) {
    EXPECT_EQ(fullFresh.iterations[idx].postExchangePoseCount, 0u);
    EXPECT_EQ(budgetedFresh.iterations[idx].postExchangePoseCount, 0u);
    EXPECT_LE(budgetedFresh.iterations[idx].localGradientFreshPoseCount,
              fullFresh.iterations[idx].localGradientFreshPoseCount);
    EXPECT_EQ(budgetedFresh.iterations[idx].commPoseCount,
              budgetedFresh.iterations[idx].localGradientFreshPoseCount);
    if (budgetedFresh.iterations[idx].localGradientFreshSkippedPoseCount > 0) {
      sawBudgetedSkip = true;
      EXPECT_LT(budgetedFresh.iterations[idx].localGradientFreshPoseCount,
                fullFresh.iterations[idx].localGradientFreshPoseCount);
    }
  }
  EXPECT_TRUE(sawBudgetedSkip);
  EXPECT_LT(budgetedFresh.iterations.back().poseCumulativeCommMb,
            fullFresh.iterations.back().poseCumulativeCommMb);
  EXPECT_TRUE(std::isfinite(budgetedFresh.finalObjective));

  options.localGradientCorrectionFreshBudgetFraction = 1.1;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmCanThresholdPostExchangeByPoseDelta) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionSharedStep = true;
  options.localGradientCorrectionFreshNeighborExchange = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  options.postExchangeMinPoseDelta = 1e9;

  const ManualDpgoMmRunResult thresholded =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(thresholded.iterations.size(), 3u);
  for (std::size_t idx = 1; idx < thresholded.iterations.size(); ++idx) {
    EXPECT_GT(thresholded.iterations[idx].localGradientFreshPoseCount, 0u);
    EXPECT_EQ(thresholded.iterations[idx].postExchangePoseCount, 0u);
    EXPECT_GT(thresholded.iterations[idx].postExchangeSkippedPoseCount, 0u);
    EXPECT_EQ(thresholded.iterations[idx].commPoseCount,
              thresholded.iterations[idx].localGradientFreshPoseCount);
  }
  EXPECT_TRUE(std::isfinite(thresholded.finalObjective));
}

TEST(testDPGO, ManualDpgoMmCanUseWeightedPostExchangeDelta) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionSharedStep = true;
  options.localGradientCorrectionFreshNeighborExchange = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  options.postExchangeMinPoseDelta = 1.0;

  const ManualDpgoMmRunResult absolute =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.postExchangeDeltaMode =
      ManualDpgoMmPostExchangeDeltaMode::Weighted;
  const ManualDpgoMmRunResult weighted =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(absolute.iterations.size(), weighted.iterations.size());
  std::size_t absolutePostPoses = 0;
  std::size_t weightedPostPoses = 0;
  for (std::size_t idx = 1; idx < weighted.iterations.size(); ++idx) {
    absolutePostPoses += absolute.iterations[idx].postExchangePoseCount;
    weightedPostPoses += weighted.iterations[idx].postExchangePoseCount;
    EXPECT_EQ(weighted.iterations[idx].localGradientFreshPoseCount,
              absolute.iterations[idx].localGradientFreshPoseCount);
    EXPECT_EQ(weighted.iterations[idx].commPoseCount,
              weighted.iterations[idx].localGradientFreshPoseCount +
                  weighted.iterations[idx].postExchangePoseCount);
    EXPECT_EQ(absolute.iterations[idx].postExchangePoseCount, 0u);
    EXPECT_GT(weighted.iterations[idx].postExchangeSkippedPoseCount, 0u);
  }
  EXPECT_EQ(absolutePostPoses, 0u);
  EXPECT_GT(weightedPostPoses, 0u);
  EXPECT_TRUE(std::isfinite(weighted.finalObjective));
}

TEST(testDPGO, ManualDpgoMmCanBudgetPostExchangePerReceiver) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.localGradientCorrection = true;
  options.localGradientCorrectionSharedStep = true;
  options.localGradientCorrectionFreshNeighborExchange = true;
  options.localGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};

  const ManualDpgoMmRunResult defaultExchange =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.postExchangeBudgetFraction = 0.5;
  options.postExchangeMaxPosesPerReceiver = 1;
  const ManualDpgoMmRunResult budgeted =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(defaultExchange.iterations.size(), budgeted.iterations.size());
  bool sawBudgetedSkip = false;
  for (std::size_t idx = 1; idx < budgeted.iterations.size(); ++idx) {
    EXPECT_EQ(budgeted.iterations[idx].localGradientFreshPoseCount,
              defaultExchange.iterations[idx].localGradientFreshPoseCount);
    EXPECT_LE(budgeted.iterations[idx].postExchangePoseCount,
              defaultExchange.iterations[idx].postExchangePoseCount);
    EXPECT_EQ(budgeted.iterations[idx].commPoseCount,
              budgeted.iterations[idx].localGradientFreshPoseCount +
                  budgeted.iterations[idx].postExchangePoseCount);
    if (budgeted.iterations[idx].postExchangeSkippedPoseCount > 0) {
      sawBudgetedSkip = true;
      EXPECT_LT(budgeted.iterations[idx].postExchangePoseCount,
                defaultExchange.iterations[idx].postExchangePoseCount);
    }
  }
  EXPECT_TRUE(sawBudgetedSkip);
  EXPECT_TRUE(std::isfinite(budgeted.finalObjective));

  options.postExchangeBudgetFraction = 1.1;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmValueSchedulerDefaultOffMatchesBaseline) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseline;
  baseline.numRobots = 2;
  baseline.maxIterations = 2;
  baseline.scheme = ManualDpgoMmScheme::MM;
  baseline.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseline.save = false;
  baseline.trustRegionIterations = 1;
  baseline.trustRegionMaxInnerIterations = 5;
  baseline.parallelLocalSolves = false;
  baseline.printIterationSummary = false;
  baseline.postExchangeBudgetFraction = 0.5;
  baseline.postExchangeMaxPosesPerReceiver = 1;

  const ManualDpgoMmRunResult expected =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseline);

  ManualDpgoMmOptions defaultOff = baseline;
  defaultOff.communicationTopologyValueByteBudgetMb = 1e-6;
  defaultOff.communicationTopologyValueMinScoreRatio = 0.75;
  ASSERT_FALSE(defaultOff.communicationTopologyValueScheduler);
  const ManualDpgoMmRunResult actual =
      runManualDpgoMm("data/tinyGrid3D.g2o", defaultOff);

  ASSERT_EQ(expected.iterations.size(), actual.iterations.size());
  for (std::size_t idx = 0; idx < expected.iterations.size(); ++idx) {
    EXPECT_EQ(expected.iterations[idx].postExchangePoseCount,
              actual.iterations[idx].postExchangePoseCount);
    EXPECT_EQ(expected.iterations[idx].postExchangeSkippedPoseCount,
              actual.iterations[idx].postExchangeSkippedPoseCount);
    EXPECT_DOUBLE_EQ(expected.iterations[idx].poseCumulativeCommMb,
                     actual.iterations[idx].poseCumulativeCommMb);
  }
  EXPECT_DOUBLE_EQ(expected.finalObjective, actual.finalObjective);
  EXPECT_DOUBLE_EQ(expected.finalGradient, actual.finalGradient);
}

TEST(testDPGO, ManualDpgoMmValueSchedulerRespectsByteBudget) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions full;
  full.numRobots = 2;
  full.maxIterations = 2;
  full.scheme = ManualDpgoMmScheme::MM;
  full.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  full.save = false;
  full.trustRegionIterations = 1;
  full.trustRegionMaxInnerIterations = 5;
  full.parallelLocalSolves = false;
  full.printIterationSummary = false;

  const ManualDpgoMmRunResult fullExchange =
      runManualDpgoMm("data/tinyGrid3D.g2o", full);

  ManualDpgoMmOptions budgeted = full;
  budgeted.communicationTopologyValueScheduler = true;
  budgeted.communicationTopologyValueByteBudgetMb = 1e-7;
  const ManualDpgoMmRunResult valueBudgeted =
      runManualDpgoMm("data/tinyGrid3D.g2o", budgeted);

  ASSERT_EQ(fullExchange.iterations.size(),
            valueBudgeted.iterations.size());
  bool sawBudgetedSkip = false;
  for (std::size_t idx = 1; idx < valueBudgeted.iterations.size(); ++idx) {
    EXPECT_LE(valueBudgeted.iterations[idx].postExchangePoseCount,
              fullExchange.iterations[idx].postExchangePoseCount);
    if (valueBudgeted.iterations[idx].postExchangeSkippedPoseCount >
        fullExchange.iterations[idx].postExchangeSkippedPoseCount) {
      sawBudgetedSkip = true;
    }
  }
  EXPECT_TRUE(sawBudgetedSkip);
  EXPECT_LT(valueBudgeted.iterations.back().poseCumulativeCommMb,
            fullExchange.iterations.back().poseCumulativeCommMb);
  EXPECT_TRUE(std::isfinite(valueBudgeted.finalObjective));

  budgeted.communicationTopologyValueByteBudgetMb = -1.0;
  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", budgeted),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmValueSchedulerDoesNotUseGlobalGradientCorrection) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.printIterationSummary = false;
  options.communicationTopologyValueScheduler = true;
  options.communicationTopologyValueByteBudgetMb = 1e-7;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  for (const ManualDpgoMmIterationSummary &row : result.iterations) {
    EXPECT_EQ(row.localGradientScalarCount, 0u);
    EXPECT_DOUBLE_EQ(row.localGradientCumulativeScalarCommMb, 0.0);
    EXPECT_EQ(row.globalGradientAcceptedCount, 0u);
    EXPECT_EQ(row.globalGradientRejectedCount, 0u);
    EXPECT_EQ(row.localHistorySyncCount, 0u);
  }
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, ManualDpgoMmInterfaceModelDefaultOffMatchesBaseline) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseline;
  baseline.numRobots = 2;
  baseline.maxIterations = 2;
  baseline.scheme = ManualDpgoMmScheme::MM;
  baseline.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseline.save = false;
  baseline.trustRegionIterations = 1;
  baseline.trustRegionMaxInnerIterations = 5;
  baseline.parallelLocalSolves = false;
  baseline.printIterationSummary = false;

  const ManualDpgoMmRunResult expected =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseline);

  ManualDpgoMmOptions defaultOff = baseline;
  defaultOff.communicationTopologyInterfaceModelPayload =
      "direction_block_diag_stiffness";
  defaultOff.communicationTopologyInterfaceModelLocalMerit = false;
  ASSERT_FALSE(defaultOff.communicationTopologyInterfaceModel);
  const ManualDpgoMmRunResult actual =
      runManualDpgoMm("data/tinyGrid3D.g2o", defaultOff);

  ASSERT_EQ(expected.iterations.size(), actual.iterations.size());
  for (std::size_t idx = 0; idx < expected.iterations.size(); ++idx) {
    EXPECT_EQ(expected.iterations[idx].commPoseCount,
              actual.iterations[idx].commPoseCount);
    EXPECT_DOUBLE_EQ(expected.iterations[idx].poseCumulativeCommMb,
                     actual.iterations[idx].poseCumulativeCommMb);
    EXPECT_EQ(expected.iterations[idx].localGradientScalarCount,
              actual.iterations[idx].localGradientScalarCount);
    EXPECT_DOUBLE_EQ(expected.iterations[idx].cumulativeCommMb,
                     actual.iterations[idx].cumulativeCommMb);
  }
  EXPECT_DOUBLE_EQ(expected.finalObjective, actual.finalObjective);
  EXPECT_DOUBLE_EQ(expected.finalGradient, actual.finalGradient);
}

TEST(testDPGO, ManualDpgoMmInterfaceModelDiagPayloadCountsBytesOnly) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseline;
  baseline.numRobots = 2;
  baseline.maxIterations = 2;
  baseline.scheme = ManualDpgoMmScheme::MM;
  baseline.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseline.save = false;
  baseline.trustRegionIterations = 1;
  baseline.trustRegionMaxInnerIterations = 5;
  baseline.parallelLocalSolves = false;
  baseline.printIterationSummary = false;

  const ManualDpgoMmRunResult noPayload =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseline);

  ManualDpgoMmOptions diagPayload = baseline;
  diagPayload.communicationTopologyInterfaceModel = true;
  diagPayload.communicationTopologyInterfaceModelPayload = "diag_stiffness";
  const ManualDpgoMmRunResult withPayload =
      runManualDpgoMm("data/tinyGrid3D.g2o", diagPayload);

  ASSERT_EQ(noPayload.iterations.size(), withPayload.iterations.size());
  std::size_t packetCount = 0;
  std::size_t scalarCount = 0;
  for (std::size_t idx = 1; idx < withPayload.iterations.size(); ++idx) {
    const auto &row = withPayload.iterations[idx];
    packetCount += row.localGradientCoupledDirectionPacketCount;
    scalarCount += row.localGradientCoupledDirectionScalarCount;
    EXPECT_EQ(row.localGradientAcceptedCount, 0u);
    EXPECT_EQ(row.localGradientCoupledDirectionAcceptedCount, 0u);
    EXPECT_EQ(row.localGradientCoupledDirectionScalarCount,
              row.localGradientCoupledDirectionPacketCount);
    const double expectedIterScalarMb =
        static_cast<double>(row.localGradientCoupledDirectionScalarCount *
                            sizeof(double)) /
        (1024.0 * 1024.0);
    EXPECT_NEAR(row.localGradientIterScalarCommMb,
                expectedIterScalarMb, 1e-15);
    EXPECT_NEAR(row.cumulativeCommMb,
                row.poseCumulativeCommMb +
                    row.localGradientCumulativeScalarCommMb,
                1e-15);
  }
  EXPECT_GT(packetCount, 0u);
  EXPECT_EQ(scalarCount, packetCount);
  EXPECT_DOUBLE_EQ(noPayload.finalObjective, withPayload.finalObjective);
  EXPECT_DOUBLE_EQ(noPayload.finalGradient, withPayload.finalGradient);
  EXPECT_GT(withPayload.iterations.back().cumulativeCommMb,
            noPayload.iterations.back().cumulativeCommMb);
}

TEST(testDPGO, ManualDpgoMmRejectsUnknownInterfaceModelPayload) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.communicationTopologyInterfaceModel = true;
  options.communicationTopologyInterfaceModelPayload = "invalid_payload";

  EXPECT_THROW(runManualDpgoMm("data/tinyGrid3D.g2o", options),
               std::invalid_argument);
}

TEST(testDPGO, ManualDpgoMmLiftedDeltaDefaultOffMatchesFullPayload) {
  const std::string datasetPath =
      writeTinySe2Dataset("manual_dpgo_mm_lifted_delta_default_off");

  ManualDpgoMmOptions fullPayload = tinySe2LiftedDeltaOptions();
  const ManualDpgoMmRunResult full =
      runManualDpgoMm(datasetPath, fullPayload);

  ManualDpgoMmOptions defaultOff = tinySe2LiftedDeltaOptions();
  defaultOff.communicationTopologyDeltaCompression = true;
  defaultOff.communicationTopologyDeltaMode =
      ManualDpgoMmCommunicationDeltaMode::Sparse;
  defaultOff.communicationTopologyDeltaTopK = 0;
  const ManualDpgoMmRunResult compressed =
      runManualDpgoMm(datasetPath, defaultOff);

  ASSERT_FALSE(defaultOff.communicationTopologyLiftedDeltaCompression);
  ASSERT_EQ(full.iterations.size(), compressed.iterations.size());
  for (std::size_t idx = 0; idx < full.iterations.size(); ++idx) {
    EXPECT_DOUBLE_EQ(full.iterations[idx].poseCumulativeCommMb,
                     compressed.iterations[idx].poseCumulativeCommMb);
    EXPECT_DOUBLE_EQ(full.iterations[idx].cumulativeCommMb,
                     compressed.iterations[idx].cumulativeCommMb);
  }
  EXPECT_DOUBLE_EQ(full.finalObjective, compressed.finalObjective);
  EXPECT_DOUBLE_EQ(full.finalGradient, compressed.finalGradient);
}

TEST(testDPGO, ManualDpgoMmLiftedDeltaRejectsLargeReconstructionError) {
  const std::string datasetPath =
      writeTinySe2Dataset("manual_dpgo_mm_lifted_delta_reject");

  ManualDpgoMmOptions fullPayload = tinySe2LiftedDeltaOptions();
  const ManualDpgoMmRunResult full =
      runManualDpgoMm(datasetPath, fullPayload);

  ManualDpgoMmOptions rejected = tinySe2LiftedDeltaOptions();
  rejected.communicationTopologyDeltaCompression = true;
  rejected.communicationTopologyDeltaMode =
      ManualDpgoMmCommunicationDeltaMode::Sparse;
  rejected.communicationTopologyDeltaTopK = 0;
  rejected.communicationTopologyLiftedDeltaCompression = true;
  rejected.communicationTopologyLiftedDeltaMaxReconstructionError = 0.0;
  rejected.communicationTopologyLiftedDeltaRank = 1;
  const ManualDpgoMmRunResult lifted =
      runManualDpgoMm(datasetPath, rejected);

  ASSERT_EQ(full.iterations.size(), lifted.iterations.size());
  for (std::size_t idx = 0; idx < full.iterations.size(); ++idx) {
    EXPECT_DOUBLE_EQ(full.iterations[idx].poseCumulativeCommMb,
                     lifted.iterations[idx].poseCumulativeCommMb);
    EXPECT_DOUBLE_EQ(full.iterations[idx].cumulativeCommMb,
                     lifted.iterations[idx].cumulativeCommMb);
  }
  EXPECT_DOUBLE_EQ(full.finalObjective, lifted.finalObjective);
  EXPECT_DOUBLE_EQ(full.finalGradient, lifted.finalGradient);
}

TEST(testDPGO, ManualDpgoMmLiftedDeltaDoesNotAffectSE3TangentPath) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions tangent;
  tangent.numRobots = 2;
  tangent.maxIterations = 2;
  tangent.scheme = ManualDpgoMmScheme::MM;
  tangent.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  tangent.save = false;
  tangent.printIterationSummary = false;
  tangent.trustRegionIterations = 1;
  tangent.trustRegionMaxInnerIterations = 5;
  tangent.parallelLocalSolves = false;
  tangent.communicationTopologyDeltaCompression = true;
  tangent.communicationTopologyDeltaMode =
      ManualDpgoMmCommunicationDeltaMode::Tangent;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", tangent);

  ManualDpgoMmOptions liftedFallback = tangent;
  liftedFallback.communicationTopologyLiftedDeltaCompression = true;
  liftedFallback.communicationTopologyLiftedDeltaMaxReconstructionError =
      1e-12;
  liftedFallback.communicationTopologyLiftedDeltaRank = 1;
  const ManualDpgoMmRunResult lifted =
      runManualDpgoMm("data/tinyGrid3D.g2o", liftedFallback);

  ASSERT_EQ(baseline.iterations.size(), lifted.iterations.size());
  for (std::size_t idx = 0; idx < baseline.iterations.size(); ++idx) {
    EXPECT_DOUBLE_EQ(baseline.iterations[idx].poseCumulativeCommMb,
                     lifted.iterations[idx].poseCumulativeCommMb);
    EXPECT_DOUBLE_EQ(baseline.iterations[idx].cumulativeCommMb,
                     lifted.iterations[idx].cumulativeCommMb);
  }
  EXPECT_DOUBLE_EQ(baseline.finalObjective, lifted.finalObjective);
  EXPECT_DOUBLE_EQ(baseline.finalGradient, lifted.finalGradient);
}

TEST(testDPGO, ManualDpgoMmGlobalCorrectionPortfolioKeepsBestCandidate) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 10;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.globalGradientCorrection = true;
  options.globalGradientCorrectionSteps = {1e-4, 3e-4, 1e-3};

  const ManualDpgoMmRunResult gradientOnly =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  options.globalStateExtrapolation = true;
  options.globalStateExtrapolationGammas = {0.05, 0.1, 0.2, 0.3};
  const ManualDpgoMmRunResult portfolio =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  EXPECT_TRUE(std::isfinite(gradientOnly.finalObjective));
  EXPECT_TRUE(std::isfinite(portfolio.finalObjective));
  EXPECT_LE(portfolio.finalObjective, gradientOnly.finalObjective + 1e-10);
}

TEST(testDPGO, ManualDpgoMmCanResetAmmHistoryAfterGlobalCorrection) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammProximalStart = true;
  options.ammLocalMeritFilter = false;
  options.globalGradientCorrection = true;
  options.globalGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  options.resetLocalHistoryAfterGlobalCorrection = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 5u);
  std::size_t checkedAfterAcceptedCorrection = 0;
  for (std::size_t idx = 2; idx < result.iterations.size(); ++idx) {
    const auto &previous = result.iterations[idx - 1];
    const auto &current = result.iterations[idx];
    if (previous.globalGradientAcceptedCount > 0 ||
        previous.globalExtrapolationAcceptedCount > 0 ||
        previous.globalAndersonAcceptedCount > 0) {
      ++checkedAfterAcceptedCorrection;
      EXPECT_EQ(current.ammTraceCount, 0u);
      EXPECT_EQ(current.ammAcceleratedAcceptedCount, 0u);
    }
  }
  EXPECT_GT(checkedAfterAcceptedCorrection, 0u);
}

TEST(testDPGO, ManualDpgoMmCanSyncAmmReferenceAfterGlobalCorrection) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammProximalStart = true;
  options.ammLocalMeritFilter = false;
  options.globalGradientCorrection = true;
  options.globalGradientCorrectionSteps = {1e-4, 1e-3, 1e-2};
  options.syncAmmReferenceAfterGlobalCorrection = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 5u);
  std::size_t syncCount = 0;
  std::size_t acceleratedAfterSync = 0;
  for (std::size_t idx = 1; idx < result.iterations.size(); ++idx) {
    syncCount += result.iterations[idx].localHistorySyncCount;
    if (idx >= 2 && result.iterations[idx - 1].localHistorySyncCount > 0) {
      acceleratedAfterSync += result.iterations[idx].ammTraceCount;
    }
  }
  EXPECT_GT(syncCount, 0u);
  EXPECT_GT(acceleratedAfterSync, 0u);
}

TEST(testDPGO, ManualDpgoMmAmmCooldownSkipsAfterRejectedCandidate) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammLocalMeritFilter = true;
  options.ammProximalStart = false;
  options.ammCooldownAfterRejected = 1;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 5u);
  EXPECT_GT(result.iterations[2].ammLocalMeritRejectedCount +
                result.iterations[2].ammRestartCount,
            0u);
  EXPECT_GT(result.iterations[3].ammSkippedCount, 0u);
  EXPECT_EQ(result.iterations[3].ammProximalStartCount, 0u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
}

TEST(testDPGO, ManualDpgoMmDistributedLocalChordalInitRunsAndAccountsInitComm) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.printIterationSummary = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.centralizedChordalInit = false;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);

  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));
  const auto &init = result.iterations.front();
  const auto &final = result.iterations.back();
  EXPECT_TRUE(std::isfinite(init.globalCost));
  EXPECT_LT(init.globalCost, 200.0);
  EXPECT_GT(init.initializationCommPoseCount, 0u);
  EXPECT_GT(init.initializationCommMb, 0.0);
  EXPECT_EQ(init.outerCommPoseCount, 0u);
  EXPECT_EQ(init.outerCommMb, 0.0);
  EXPECT_EQ(final.initializationCommPoseCount,
            init.initializationCommPoseCount);
  EXPECT_EQ(final.outerCommPoseCount + final.initializationCommPoseCount,
            final.cumulativeCommPoseCount);
  EXPECT_NEAR(final.outerCommMb + final.initializationCommMb,
              final.cumulativeCommMb, 1e-12);
}

TEST(testDPGO, ManualDpgoMmDistributedLocalChordalInitUsesSharedLoopGauge) {
  std::ifstream dataset("data/CSAIL.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "CSAIL.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 5;
  options.maxIterations = 0;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Cholesky;
  options.save = false;
  options.printIterationSummary = false;
  options.trustRegionIterations = 1;
  options.trustRegionAcceptedIterations = 1;
  options.trustRegionMaxInnerIterations = 10;
  options.reducedRotationTcgRelativeTolerance = 0.2;
  options.parallelLocalSolves = false;
  options.centralizedChordalInit = false;
  options.distributedInitializationRefinementRounds = 4;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammLazyPlainAfterCertificate = true;
  options.ammSurrogateFirstExactEvaluation = true;
  options.ammMixedSurrogateSkipSimpleAfterTrueLocalStreak = 1;
  options.ammMixedSurrogateForceSimpleEverySkippedRounds = 0;
  options.fusedCandidateEvaluation = true;
  options.lazyCandidateGradientEvaluation = true;
  options.lazySolverStartGradientEvaluation = true;
  options.lazySurrogateCandidateGradientEvaluation = true;
  options.reducedRotationSkipRedundantCandidateProjection = true;
  options.reducedRotationGradientBoundaryCandidate = false;
  options.reducedRotationSurrogateTcgAccept = true;
  options.reducedRotationDirectObjective = true;

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/CSAIL.g2o", options);

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_LT(result.iterations.front().globalCost, 100.0);
  EXPECT_TRUE(std::isfinite(result.iterations.front().gradient));
  EXPECT_GT(result.iterations.front().initializationCommPoseCount, 0u);
}

TEST(testDPGO, ManualDpgoMmDistributedLocalChordalInitAveragesRobotGaugeLoop) {
  const std::string datasetPath =
      writeCyclicRobotGaugeSe2Dataset("manual_dist_init_robot_gauge_loop");

  ManualDpgoMmOptions options;
  options.numRobots = 3;
  options.maxIterations = 0;
  options.scheme = ManualDpgoMmScheme::MM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = false;
  options.printIterationSummary = false;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.centralizedChordalInit = false;
  options.distributedInitializationRefinementRounds = 0;

  const ManualDpgoMmRunResult result = runManualDpgoMm(datasetPath, options);
  std::remove(datasetPath.c_str());

  ASSERT_EQ(result.iterations.size(), 1u);
  EXPECT_TRUE(std::isfinite(result.iterations.front().globalCost));
  EXPECT_LT(result.iterations.front().globalCost, 300.0);
  EXPECT_GT(result.iterations.front().initializationCommPoseCount, 0u);
}

TEST(testDPGO, ManualDpgoMmCandidateEvaluationCacheDefaultsOffAndNoProfile) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.candidateEvaluationCache);
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = false;
  options.candidateEvaluationCache = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_cache_no_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  EXPECT_FALSE(profile.good());
}

TEST(testDPGO, ManualDpgoMmCandidateEvaluationCacheMatchesNoCache) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 3;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.ammDpgoSurrogateParity = true;
  baseOptions.ammDpgoMixedSurrogatePortfolio = true;

  ManualDpgoMmOptions cacheOptions = baseOptions;
  cacheOptions.candidateEvaluationCache = true;

  const ManualDpgoMmRunResult noCache =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult cache =
      runManualDpgoMm("data/tinyGrid3D.g2o", cacheOptions);

  ASSERT_EQ(noCache.iterations.size(), cache.iterations.size());
  EXPECT_DOUBLE_EQ(noCache.finalObjective, cache.finalObjective);
  EXPECT_DOUBLE_EQ(noCache.finalGradient, cache.finalGradient);
  ASSERT_FALSE(noCache.iterations.empty());
  const auto &noCacheLast = noCache.iterations.back();
  const auto &cacheLast = cache.iterations.back();
  EXPECT_EQ(noCacheLast.cumulativeCommPoseCount,
            cacheLast.cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(noCacheLast.cumulativeCommMb,
                   cacheLast.cumulativeCommMb);
}

TEST(testDPGO,
     ManualDpgoMmCandidateEvaluationCacheDoesNotReuseChangedCandidate) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 3;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.localStateExtrapolation = true;
  baseOptions.localStateExtrapolationGammas = {0.25, 0.5};
  baseOptions.ammGammaScales = {1.0, 0.5};

  ManualDpgoMmOptions cacheOptions = baseOptions;
  cacheOptions.candidateEvaluationCache = true;

  const ManualDpgoMmRunResult noCache =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult cache =
      runManualDpgoMm("data/tinyGrid3D.g2o", cacheOptions);

  ASSERT_EQ(noCache.finalEstimate.rows(), cache.finalEstimate.rows());
  ASSERT_EQ(noCache.finalEstimate.cols(), cache.finalEstimate.cols());
  EXPECT_DOUBLE_EQ(noCache.finalObjective, cache.finalObjective);
  EXPECT_DOUBLE_EQ(noCache.finalGradient, cache.finalGradient);
  EXPECT_LT((noCache.finalEstimate - cache.finalEstimate).norm(), 1e-12);
}

TEST(testDPGO, ManualDpgoMmCandidateEvaluationCacheReportsProfileCounts) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = true;
  options.candidateEvaluationCache = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_cache_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"candidate_evaluation_cache_hit_count\""),
            std::string::npos);
  EXPECT_NE(content.find("\"candidate_evaluation_cache_miss_count\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmCandidateObjectReuseDefaultsOffAndNoProfile) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.candidateObjectReuse);
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = false;
  options.candidateObjectReuse = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_candidate_reuse_no_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  EXPECT_FALSE(profile.good());
}

TEST(testDPGO, ManualDpgoMmCandidateObjectReuseMatchesNoReuse) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 3;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.ammDpgoSurrogateParity = true;
  baseOptions.ammDpgoMixedSurrogatePortfolio = true;

  ManualDpgoMmOptions reuseOptions = baseOptions;
  reuseOptions.candidateObjectReuse = true;

  const ManualDpgoMmRunResult noReuse =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult reuse =
      runManualDpgoMm("data/tinyGrid3D.g2o", reuseOptions);

  ASSERT_EQ(noReuse.iterations.size(), reuse.iterations.size());
  EXPECT_DOUBLE_EQ(noReuse.finalObjective, reuse.finalObjective);
  EXPECT_DOUBLE_EQ(noReuse.finalGradient, reuse.finalGradient);
  ASSERT_FALSE(noReuse.iterations.empty());
  const auto &noReuseLast = noReuse.iterations.back();
  const auto &reuseLast = reuse.iterations.back();
  EXPECT_EQ(noReuseLast.cumulativeCommPoseCount,
            reuseLast.cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(noReuseLast.cumulativeCommMb,
                   reuseLast.cumulativeCommMb);
}

TEST(testDPGO, ManualDpgoMmCandidateObjectReuseDoesNotReuseChangedCandidate) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 3;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.localStateExtrapolation = true;
  baseOptions.localStateExtrapolationGammas = {0.25, 0.5};
  baseOptions.ammGammaScales = {1.0, 0.5};

  ManualDpgoMmOptions reuseOptions = baseOptions;
  reuseOptions.candidateObjectReuse = true;

  const ManualDpgoMmRunResult noReuse =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult reuse =
      runManualDpgoMm("data/tinyGrid3D.g2o", reuseOptions);

  ASSERT_EQ(noReuse.finalEstimate.rows(), reuse.finalEstimate.rows());
  ASSERT_EQ(noReuse.finalEstimate.cols(), reuse.finalEstimate.cols());
  EXPECT_DOUBLE_EQ(noReuse.finalObjective, reuse.finalObjective);
  EXPECT_DOUBLE_EQ(noReuse.finalGradient, reuse.finalGradient);
  EXPECT_LT((noReuse.finalEstimate - reuse.finalEstimate).norm(), 1e-12);
}

TEST(testDPGO, ManualDpgoMmCandidateObjectReuseReportsProfileCounts) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = true;
  options.candidateObjectReuse = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_candidate_reuse_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"candidate_object_reuse\""),
            std::string::npos);
  EXPECT_NE(content.find("\"candidate_construction_sec\""),
            std::string::npos);
  EXPECT_NE(content.find("\"candidate_object_reuse_count\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmOptimizerProfileReportsAgentBreakdown) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_agent_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"agent_profiles\""), std::string::npos);
  EXPECT_NE(content.find("\"robot_id\": 0"), std::string::npos);
  EXPECT_NE(content.find("\"robot_id\": 1"), std::string::npos);
  EXPECT_NE(content.find("\"agent_optimize_local_model_sec\""),
            std::string::npos);
  EXPECT_NE(content.find("\"reduced_candidate_evaluation_count\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmOptimizerProfileReportsPortfolioAttribution) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;
  options.save = true;
  options.profileOptimizer = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_portfolio_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"reduced_portfolio\""), std::string::npos);
  EXPECT_NE(content.find("\"selection_count\""), std::string::npos);
  EXPECT_NE(content.find("\"winner_counts\""), std::string::npos);
  EXPECT_NE(content.find("\"candidate_counts\""), std::string::npos);
  EXPECT_NE(content.find("\"schur_jacobi\""), std::string::npos);
  EXPECT_NE(content.find("\"winner_cost_margin_sum\""), std::string::npos);
  EXPECT_NE(content.find("\"agent_profiles\""), std::string::npos);
}

TEST(testDPGO, ManualDpgoMmAdaptivePortfolioReportsFastPathProfile) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 4;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  options.save = true;
  options.profileOptimizer = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_adaptive_portfolio_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 5u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"reduced_adaptive_portfolio\""),
            std::string::npos);
  EXPECT_NE(content.find("\"full_selection_count\""), std::string::npos);
  EXPECT_NE(content.find("\"fast_path_count\""), std::string::npos);
  EXPECT_NE(content.find("\"certified_fast_path_count\""),
            std::string::npos);
  EXPECT_NE(content.find("\"fallback_count\""), std::string::npos);
  EXPECT_NE(content.find("\"reduced_portfolio\""), std::string::npos);
}

TEST(testDPGO, ManualDpgoMmCertifiedAdaptivePortfolioDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.reducedAdaptivePortfolioCertifiedFastPath);
}

TEST(testDPGO, ManualDpgoMmCertifiedAdaptivePortfolioMatchesDefault) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 12;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  baseOptions.save = false;
  baseOptions.profileOptimizer = true;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.ammDpgoSurrogateParity = true;
  baseOptions.ammDpgoMixedSurrogatePortfolio = true;

  ManualDpgoMmOptions certifiedOptions = baseOptions;
  certifiedOptions.reducedAdaptivePortfolioCertifiedFastPath = true;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult certified =
      runManualDpgoMm("data/tinyGrid3D.g2o", certifiedOptions);

  ASSERT_EQ(baseline.iterations.size(), certified.iterations.size());
  EXPECT_NEAR(baseline.finalObjective, certified.finalObjective, 1e-9);
  EXPECT_NEAR(baseline.finalGradient, certified.finalGradient, 1e-9);
  ASSERT_FALSE(baseline.iterations.empty());
  ASSERT_FALSE(certified.iterations.empty());
  EXPECT_EQ(baseline.iterations.back().cumulativeCommPoseCount,
            certified.iterations.back().cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(baseline.iterations.back().cumulativeCommMb,
                   certified.iterations.back().cumulativeCommMb);
  EXPECT_LE(certified.optimizerReducedSolveCount,
            baseline.optimizerReducedSolveCount);
}

TEST(testDPGO, ManualDpgoMmFusedCandidateEvaluationDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.fusedCandidateEvaluation);
}

TEST(testDPGO, ManualDpgoMmFusedCandidateEvaluationMatchesSeparateCalls) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 3;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  baseOptions.reducedAdaptivePortfolioCertifiedFastPath = true;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.ammDpgoSurrogateParity = true;
  baseOptions.ammDpgoMixedSurrogatePortfolio = true;

  ManualDpgoMmOptions fusedOptions = baseOptions;
  fusedOptions.fusedCandidateEvaluation = true;

  const ManualDpgoMmRunResult separate =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult fused =
      runManualDpgoMm("data/tinyGrid3D.g2o", fusedOptions);

  ASSERT_EQ(separate.iterations.size(), fused.iterations.size());
  EXPECT_DOUBLE_EQ(separate.finalObjective, fused.finalObjective);
  EXPECT_NEAR(separate.finalGradient, fused.finalGradient, 1e-12);
  ASSERT_FALSE(separate.iterations.empty());
  const auto &separateLast = separate.iterations.back();
  const auto &fusedLast = fused.iterations.back();
  EXPECT_EQ(separateLast.cumulativeCommPoseCount,
            fusedLast.cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(separateLast.cumulativeCommMb,
                   fusedLast.cumulativeCommMb);
  ASSERT_EQ(separate.finalEstimate.rows(), fused.finalEstimate.rows());
  ASSERT_EQ(separate.finalEstimate.cols(), fused.finalEstimate.cols());
  EXPECT_LT((separate.finalEstimate - fused.finalEstimate).norm(), 1e-12);
}

TEST(testDPGO, ManualDpgoMmFusedCandidateEvaluationNoProfileWhenDisabled) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 1;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = false;
  options.fusedCandidateEvaluation = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_fused_candidate_no_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 2u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  EXPECT_FALSE(profile.good());
}

TEST(testDPGO, ManualDpgoMmFusedCandidateEvaluationReportsProfileFlag) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = true;
  options.fusedCandidateEvaluation = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_fused_candidate_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"fused_candidate_evaluation\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmLazyCandidateGradientDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.lazyCandidateGradientEvaluation);
}

TEST(testDPGO, ManualDpgoMmLazyCandidateGradientMatchesEager) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 4;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  baseOptions.reducedAdaptivePortfolioCertifiedFastPath = true;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.ammDpgoSurrogateParity = true;
  baseOptions.ammDpgoMixedSurrogatePortfolio = true;

  ManualDpgoMmOptions lazyOptions = baseOptions;
  lazyOptions.lazyCandidateGradientEvaluation = true;

  const ManualDpgoMmRunResult eager =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult lazy =
      runManualDpgoMm("data/tinyGrid3D.g2o", lazyOptions);

  ASSERT_EQ(eager.iterations.size(), lazy.iterations.size());
  EXPECT_DOUBLE_EQ(eager.finalObjective, lazy.finalObjective);
  EXPECT_NEAR(eager.finalGradient, lazy.finalGradient, 1e-12);
  ASSERT_FALSE(eager.iterations.empty());
  const auto &eagerLast = eager.iterations.back();
  const auto &lazyLast = lazy.iterations.back();
  EXPECT_EQ(eagerLast.cumulativeCommPoseCount,
            lazyLast.cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(eagerLast.cumulativeCommMb,
                   lazyLast.cumulativeCommMb);
  ASSERT_EQ(eager.finalEstimate.rows(), lazy.finalEstimate.rows());
  ASSERT_EQ(eager.finalEstimate.cols(), lazy.finalEstimate.cols());
  EXPECT_LT((eager.finalEstimate - lazy.finalEstimate).norm(), 1e-12);
}

TEST(testDPGO, ManualDpgoMmLazyCandidateGradientReportsProfileCounts) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = true;
  options.lazyCandidateGradientEvaluation = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_lazy_candidate_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"lazy_candidate_gradient_evaluation\""),
            std::string::npos);
  EXPECT_NE(content.find("\"lazy_candidate_gradient_evaluation_count\""),
            std::string::npos);
  EXPECT_NE(content.find("\"lazy_candidate_gradient_evaluation_sec\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmLazySolverStartGradientDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.lazySolverStartGradientEvaluation);
}

TEST(testDPGO, ManualDpgoMmLazySolverStartGradientMatchesEager) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 4;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  baseOptions.reducedAdaptivePortfolioCertifiedFastPath = true;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.ammDpgoSurrogateParity = true;
  baseOptions.ammDpgoMixedSurrogatePortfolio = true;

  ManualDpgoMmOptions lazyOptions = baseOptions;
  lazyOptions.lazySolverStartGradientEvaluation = true;

  const ManualDpgoMmRunResult eager =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult lazy =
      runManualDpgoMm("data/tinyGrid3D.g2o", lazyOptions);

  ASSERT_EQ(eager.iterations.size(), lazy.iterations.size());
  EXPECT_DOUBLE_EQ(eager.finalObjective, lazy.finalObjective);
  EXPECT_NEAR(eager.finalGradient, lazy.finalGradient, 1e-12);
  ASSERT_FALSE(eager.iterations.empty());
  const auto &eagerLast = eager.iterations.back();
  const auto &lazyLast = lazy.iterations.back();
  EXPECT_EQ(eagerLast.cumulativeCommPoseCount,
            lazyLast.cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(eagerLast.cumulativeCommMb,
                   lazyLast.cumulativeCommMb);
  ASSERT_EQ(eager.finalEstimate.rows(), lazy.finalEstimate.rows());
  ASSERT_EQ(eager.finalEstimate.cols(), lazy.finalEstimate.cols());
  EXPECT_LT((eager.finalEstimate - lazy.finalEstimate).norm(), 1e-12);
}

TEST(testDPGO, ManualDpgoMmLazySolverStartGradientReportsProfileCounts) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = true;
  options.lazySolverStartGradientEvaluation = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_lazy_solver_start_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"lazy_solver_start_gradient_evaluation\""),
            std::string::npos);
  EXPECT_NE(content.find("\"solver_start_gradient_evaluation_count\""),
            std::string::npos);
  EXPECT_NE(content.find("\"solver_start_gradient_evaluation_sec\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmLazySurrogateCandidateGradientDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.lazySurrogateCandidateGradientEvaluation);
}

TEST(testDPGO, ManualDpgoMmLazySurrogateCandidateGradientMatchesEager) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 4;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  baseOptions.reducedAdaptivePortfolioCertifiedFastPath = true;
  baseOptions.lazyCandidateGradientEvaluation = true;
  baseOptions.lazySolverStartGradientEvaluation = true;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.ammDpgoSurrogateParity = true;
  baseOptions.ammDpgoMixedSurrogatePortfolio = true;

  ManualDpgoMmOptions lazyOptions = baseOptions;
  lazyOptions.lazySurrogateCandidateGradientEvaluation = true;

  const ManualDpgoMmRunResult eager =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult lazy =
      runManualDpgoMm("data/tinyGrid3D.g2o", lazyOptions);

  ASSERT_EQ(eager.iterations.size(), lazy.iterations.size());
  EXPECT_DOUBLE_EQ(eager.finalObjective, lazy.finalObjective);
  EXPECT_NEAR(eager.finalGradient, lazy.finalGradient, 1e-12);
  ASSERT_FALSE(eager.iterations.empty());
  const auto &eagerLast = eager.iterations.back();
  const auto &lazyLast = lazy.iterations.back();
  EXPECT_EQ(eagerLast.cumulativeCommPoseCount,
            lazyLast.cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(eagerLast.cumulativeCommMb,
                   lazyLast.cumulativeCommMb);
  ASSERT_EQ(eager.finalEstimate.rows(), lazy.finalEstimate.rows());
  ASSERT_EQ(eager.finalEstimate.cols(), lazy.finalEstimate.cols());
  EXPECT_LT((eager.finalEstimate - lazy.finalEstimate).norm(), 1e-12);
}

TEST(testDPGO, ManualDpgoMmLazySurrogateCandidateGradientReportsProfileFlag) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.save = true;
  options.profileOptimizer = true;
  options.lazyCandidateGradientEvaluation = true;
  options.lazySolverStartGradientEvaluation = true;
  options.lazySurrogateCandidateGradientEvaluation = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_lazy_surrogate_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"lazy_surrogate_candidate_gradient_evaluation\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmCurvatureCauchyCandidateDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.reducedRotationCurvatureCauchyCandidate);
}

TEST(testDPGO, ManualDpgoMmCurvatureCauchyCandidateRunsAndProfiles) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  options.save = true;
  options.profileOptimizer = true;
  options.reducedRotationCurvatureCauchyCandidate = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_curvature_cauchy_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(
      content.find("\"reduced_rotation_curvature_cauchy_candidate\""),
      std::string::npos);
  EXPECT_NE(content.find("\"reduced_curvature_cauchy_candidate_count\""),
            std::string::npos);
  EXPECT_NE(content.find("\"reduced_curvature_cauchy_accepted_count\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmCurvatureFallbackCandidateDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.reducedRotationCurvatureFallbackCandidate);
}

TEST(testDPGO, ManualDpgoMmCurvatureFallbackCandidateRunsAndProfiles) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  options.save = true;
  options.profileOptimizer = true;
  options.reducedRotationCurvatureFallbackCandidate = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_curvature_fallback_profile_test_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(
      content.find("\"reduced_rotation_curvature_fallback_candidate\""),
      std::string::npos);
  EXPECT_NE(
      content.find("\"reduced_curvature_cauchy_fallback_candidate_count\""),
      std::string::npos);
  EXPECT_NE(content.find(
                "\"reduced_curvature_cauchy_fallback_accepted_count\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmSkipRedundantCandidateProjectionDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.reducedRotationSkipRedundantCandidateProjection);
}

TEST(testDPGO, ManualDpgoMmSkipRedundantCandidateProjectionMatchesBaseline) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions baseOptions;
  baseOptions.numRobots = 2;
  baseOptions.maxIterations = 4;
  baseOptions.scheme = ManualDpgoMmScheme::AMM;
  baseOptions.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  baseOptions.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  baseOptions.reducedAdaptivePortfolioCertifiedFastPath = true;
  baseOptions.save = false;
  baseOptions.trustRegionIterations = 1;
  baseOptions.trustRegionMaxInnerIterations = 5;
  baseOptions.parallelLocalSolves = false;
  baseOptions.ammDpgoSurrogateParity = true;
  baseOptions.ammDpgoMixedSurrogatePortfolio = true;

  ManualDpgoMmOptions skipOptions = baseOptions;
  skipOptions.reducedRotationSkipRedundantCandidateProjection = true;

  const ManualDpgoMmRunResult baseline =
      runManualDpgoMm("data/tinyGrid3D.g2o", baseOptions);
  const ManualDpgoMmRunResult skipped =
      runManualDpgoMm("data/tinyGrid3D.g2o", skipOptions);

  ASSERT_EQ(baseline.iterations.size(), skipped.iterations.size());
  EXPECT_NEAR(baseline.finalObjective, skipped.finalObjective, 1e-10);
  EXPECT_NEAR(baseline.finalGradient, skipped.finalGradient, 1e-10);
  ASSERT_FALSE(baseline.iterations.empty());
  const auto &baselineLast = baseline.iterations.back();
  const auto &skippedLast = skipped.iterations.back();
  EXPECT_EQ(baselineLast.cumulativeCommPoseCount,
            skippedLast.cumulativeCommPoseCount);
  EXPECT_DOUBLE_EQ(baselineLast.cumulativeCommMb,
                   skippedLast.cumulativeCommMb);
  ASSERT_EQ(baseline.finalEstimate.rows(), skipped.finalEstimate.rows());
  ASSERT_EQ(baseline.finalEstimate.cols(), skipped.finalEstimate.cols());
  EXPECT_LT((baseline.finalEstimate - skipped.finalEstimate).norm(), 1e-10);
}

TEST(testDPGO, ManualDpgoMmSkipRedundantCandidateProjectionReportsProfile) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 2;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::AdaptivePortfolio;
  options.save = true;
  options.profileOptimizer = true;
  options.reducedRotationSkipRedundantCandidateProjection = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_skip_candidate_projection_profile_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(
      content.find(
          "\"reduced_rotation_skip_redundant_candidate_projection\""),
      std::string::npos);
  EXPECT_NE(content.find("\"reduced_candidate_projection_skip_count\""),
            std::string::npos);
}

TEST(testDPGO, ManualDpgoMmSurrogateFirstExactEvaluationDefaultDisabled) {
  ManualDpgoMmOptions options;
  EXPECT_FALSE(options.ammSurrogateFirstExactEvaluation);
}

TEST(testDPGO, ManualDpgoMmSurrogateFirstExactEvaluationRunsAndProfiles) {
  std::ifstream dataset("data/tinyGrid3D.g2o");
  if (!dataset.good()) {
    GTEST_SKIP() << "tinyGrid3D.g2o not available from current test cwd";
  }

  ManualDpgoMmOptions options;
  options.numRobots = 2;
  options.maxIterations = 3;
  options.scheme = ManualDpgoMmScheme::AMM;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Cholesky;
  options.save = true;
  options.profileOptimizer = true;
  options.trustRegionIterations = 1;
  options.trustRegionMaxInnerIterations = 5;
  options.parallelLocalSolves = false;
  options.ammDpgoSurrogateParity = true;
  options.ammDpgoMixedSurrogatePortfolio = true;
  options.ammLazyPlainAfterCertificate = true;
  options.ammSurrogateFirstExactEvaluation = true;
  options.outputDirectory =
      std::string("/tmp/manual_dpgo_mm_surrogate_first_exact_profile_") +
      std::to_string(static_cast<unsigned long long>(std::rand()));

  const ManualDpgoMmRunResult result =
      runManualDpgoMm("data/tinyGrid3D.g2o", options);
  ASSERT_EQ(result.iterations.size(), 4u);
  EXPECT_TRUE(std::isfinite(result.finalObjective));
  EXPECT_TRUE(std::isfinite(result.finalGradient));

  std::ifstream profile(options.outputDirectory + "/optimizer_profile.json");
  ASSERT_TRUE(profile.good());
  const std::string content((std::istreambuf_iterator<char>(profile)),
                            std::istreambuf_iterator<char>());
  EXPECT_NE(content.find("\"amm_surrogate_first_exact_evaluation\""),
            std::string::npos);
}

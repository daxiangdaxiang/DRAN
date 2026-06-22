#include <DPGO/DPGO_utils.h>
#include <DPGO/MPIAsyncFactorExchange.h>
#include <DPGO/RelativeSEMeasurement.h>
#include <DPGO/TEDCCI.h>

#include <Eigen/Geometry>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

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

struct SyntheticGraph {
  int dimension = 3;
  int num_poses = 0;
  std::vector<int> owner;
  std::vector<RelativeSEMeasurement> all;
  std::vector<RelativeSEMeasurement> shared;
  std::vector<std::vector<RelativeSEMeasurement>> local_by_robot;
};

SyntheticGraph makeSyntheticGraph(int dimension, int numRobots) {
  SyntheticGraph graph;
  graph.dimension = dimension;
  graph.num_poses = numRobots * 2;
  graph.owner.resize(graph.num_poses);
  graph.local_by_robot.resize(numRobots);
  for (int pose = 0; pose < graph.num_poses; ++pose) {
    graph.owner[pose] = std::min(pose / 2, numRobots - 1);
  }

  std::vector<Matrix> rotations(graph.num_poses);
  std::vector<Vector> translations(graph.num_poses, Vector::Zero(dimension));
  for (int pose = 0; pose < graph.num_poses; ++pose) {
    if (dimension == 2) {
      rotations[pose] =
          rotation2(0.12 * pose + 0.02 * std::sin(static_cast<double>(pose)));
      translations[pose] << 0.7 * pose, 0.15 * pose + 0.02 * pose * pose;
    } else {
      rotations[pose] =
          rotation3(0.03 * pose, -0.02 * pose, 0.09 * pose);
      translations[pose] << 0.8 * pose, 0.25 * pose, 0.03 * pose * pose;
    }
  }

  auto addEdge = [&](int i, int j, double kappa, double tau) {
    RelativeSEMeasurement measurement = makeMeasurement(
        i, j, graph.owner, rotations[i], translations[i], rotations[j],
        translations[j], kappa, tau);
    measurement.weight = 1.0 + 0.05 * static_cast<double>((i + j) % 3);
    graph.all.push_back(measurement);
    if (graph.owner[i] == graph.owner[j]) {
      graph.local_by_robot[graph.owner[i]].push_back(measurement);
    } else {
      graph.shared.push_back(measurement);
    }
  };

  for (int pose = 0; pose + 1 < graph.num_poses; ++pose) {
    addEdge(pose, pose + 1, 2.0 + 0.2 * pose, 3.0 + 0.1 * pose);
  }
  if (graph.num_poses > 4) {
    addEdge(0, graph.num_poses - 1, 2.4, 1.8);
    addEdge(1, graph.num_poses - 2, 1.7, 2.6);
  }
  return graph;
}

std::vector<TEDCCIPartition> makeTedPartitions(
    const SyntheticGraph &graph, const TEDCCIParams &params) {
  std::vector<std::vector<PoseKey>> localPoses(graph.local_by_robot.size());
  for (int pose = 0; pose < graph.num_poses; ++pose) {
    localPoses.at(graph.owner.at(pose)).push_back(
        PoseKey{graph.owner.at(pose), pose});
  }

  std::vector<TEDCCIPartition> partitions;
  for (std::size_t robot = 0; robot < localPoses.size(); ++robot) {
    partitions.push_back(BoundarySelector::SelectBoundaryVariables(
        static_cast<int>(robot), localPoses[robot],
        graph.local_by_robot[robot], graph.shared, params));
  }
  return partitions;
}

FactorMessage makeRingFactor(int rank, int size) {
  FactorMessage msg;
  msg.sender_robot_id = rank;
  msg.receiver_robot_id = (rank + 1) % size;
  msg.epoch = 31;
  msg.topology_epoch = 7;
  msg.factor_type = FactorType::ROTATION;
  msg.keys = {PoseKey{rank, rank}, PoseKey{msg.receiver_robot_id, rank + 10}};
  msg.A = Matrix::Constant(2, 4, 0.25 * static_cast<double>(rank + 1));
  msg.b = Vector::Constant(2, static_cast<double>(rank + 1));
  msg.checksum = ComputeFactorMessageChecksum(msg);
  return msg;
}

SolutionMessage makeRingSolution(int rank, int size) {
  SolutionMessage msg;
  msg.sender_robot_id = rank;
  msg.receiver_robot_id = (rank + size - 1) % size;
  msg.epoch = 37;
  msg.topology_epoch = 9;
  msg.factor_type = FactorType::TRANSLATION;
  msg.keys = {PoseKey{rank, rank + 20}};
  msg.x = Vector::Constant(3, static_cast<double>(rank + 4));
  msg.checksum = ComputeSolutionMessageChecksum(msg);
  return msg;
}

void drainExchange(MPIAsyncFactorExchange *exchange, int rank,
                   std::vector<FactorMessage> *factors,
                   std::vector<SolutionMessage> *solutions) {
  for (int attempt = 0; attempt < 2000; ++attempt) {
    const auto factorBatch = exchange->PollFactorMessages(rank);
    factors->insert(factors->end(), factorBatch.begin(), factorBatch.end());
    const auto solutionBatch = exchange->PollSolutionMessages(rank);
    solutions->insert(solutions->end(), solutionBatch.begin(),
                      solutionBatch.end());
    if (!factors->empty() && !solutions->empty()) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

void checkRingExchange(int rank, int size, MPIAsyncFactorExchange *exchange) {
  exchange->SendFactor(makeRingFactor(rank, size));
  exchange->SendSolution(makeRingSolution(rank, size));

  std::vector<FactorMessage> factors;
  std::vector<SolutionMessage> solutions;
  drainExchange(exchange, rank, &factors, &solutions);

  if (factors.size() != 1 || solutions.size() != 1) {
    throw std::runtime_error("MPI TED-CCI exchange did not receive ring messages");
  }
  const int expectedFactorSender = (rank + size - 1) % size;
  const int expectedSolutionSender = (rank + 1) % size;
  if (factors.front().sender_robot_id != expectedFactorSender ||
      factors.front().receiver_robot_id != rank ||
      factors.front().checksum !=
          ComputeFactorMessageChecksum(factors.front())) {
    throw std::runtime_error("MPI TED-CCI factor payload mismatch");
  }
  if (solutions.front().sender_robot_id != expectedSolutionSender ||
      solutions.front().receiver_robot_id != rank ||
      solutions.front().checksum !=
          ComputeSolutionMessageChecksum(solutions.front())) {
    throw std::runtime_error("MPI TED-CCI solution payload mismatch");
  }
}

void runTedCCIOrchestratorCheck(int rank, int size,
                                MPIAsyncFactorExchange *exchange) {
  TEDCCIParams params;
  params.use_measurement_weight = true;
  const SyntheticGraph graph = makeSyntheticGraph(3, size);
  const auto partitions = makeTedPartitions(graph, params);
  const Matrix direct = TEDCCISolver::InitializeSingleProcessDirect(
      graph.dimension, graph.num_poses, graph.all, partitions, params);

  std::vector<int> expectedRobots(size);
  for (int robot = 0; robot < size; ++robot) {
    expectedRobots[robot] = robot;
  }

  TEDCCIOrchestrator orchestrator(rank, params, exchange);
  orchestrator.SetProblemMetadata(graph.dimension, graph.num_poses,
                                  partitions.at(rank), expectedRobots);
  orchestrator.SetLocalMeasurements(graph.local_by_robot.at(rank),
                                    graph.shared);
  orchestrator.StartEpoch(101);

  for (int round = 0;
       round < 2000 && !orchestrator.HasGlobalConsistentSolution();
       ++round) {
    orchestrator.ProcessIncomingMessages();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!orchestrator.HasGlobalConsistentSolution()) {
    throw std::runtime_error("MPI TED-CCI orchestrator did not converge");
  }
  const Matrix actual = orchestrator.GetGlobalConsistentInitialization();
  const double relDiff =
      (actual - direct).norm() / std::max(1.0, direct.norm());
  if (relDiff >= 1e-8) {
    throw std::runtime_error("MPI TED-CCI result differs from direct solver");
  }
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    if (size < 2) {
      throw std::runtime_error("run_ted_cci_mpi requires at least 2 ranks");
    }
    MPIAsyncFactorExchange exchange(MPI_COMM_WORLD);
    checkRingExchange(rank, size, &exchange);
    runTedCCIOrchestratorCheck(rank, size, &exchange);
    if (rank == 0) {
      std::cout << "TED-CCI MPI point-to-point ranks=" << size
                << " status=ok" << std::endl;
    }
    MPI_Finalize();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << " error: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

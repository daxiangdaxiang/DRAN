#include <DPGO/DCCI_chordal_initialization.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/MPIDCCICommunicator.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Geometry>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

std::vector<RelativeSEMeasurement> makeSyntheticGraph(size_t d,
                                                      size_t *numPoses) {
  *numPoses = 8;
  std::vector<Matrix> rotations(*numPoses);
  std::vector<Vector> translations(*numPoses, Vector::Zero(d));
  for (size_t pose = 0; pose < *numPoses; ++pose) {
    if (d == 2) {
      rotations[pose] = rotation2(0.15 * static_cast<double>(pose) +
                                  0.03 * std::sin(static_cast<double>(pose)));
      translations[pose] << 0.9 * pose, 0.2 * pose + 0.05 * pose * pose;
    } else {
      rotations[pose] = rotation3(0.04 * pose, -0.025 * pose, 0.12 * pose);
      translations[pose] << 0.8 * pose, 0.3 * pose, 0.05 * pose * pose;
    }
  }

  std::vector<RelativeSEMeasurement> measurements;
  for (size_t pose = 0; pose + 1 < *numPoses; ++pose) {
    measurements.emplace_back(makeMeasurement(
        pose, pose + 1, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0 + pose,
        3.0 + 0.5 * pose));
  }
  measurements.emplace_back(makeMeasurement(0, 3, rotations[0],
                                            translations[0], rotations[3],
                                            translations[3], 4.0, 2.5));
  measurements.emplace_back(makeMeasurement(2, 6, rotations[2],
                                            translations[2], rotations[6],
                                            translations[6], 3.5, 2.8));
  measurements.emplace_back(makeMeasurement(7, 0, rotations[7],
                                            translations[7], rotations[0],
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

void seedLocalBlocks(size_t d, size_t numPoses,
                     const DistributedPartition &partition,
                     MPIDCCICommunicator *comm) {
  for (const PoseID &id : partition.global_pose_ids) {
    const size_t pose = id.second;
    Matrix R = Matrix::Identity(d, d) * (1.0 + static_cast<double>(pose));
    Vector t = Vector::Constant(d, static_cast<double>(pose));
    comm->setRotationBlock(id, R);
    comm->setTranslationBlock(id, t);
  }
}

void exerciseGhostExchange(size_t numPoses, const DistributedPartition &partition,
                           MPIDCCICommunicator *comm) {
  std::vector<PoseID> allPoseIds;
  allPoseIds.reserve(numPoses);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    allPoseIds.push_back(poseID(pose));
  }
  const auto rotationGhosts =
      comm->exchangeRotationBlocks(partition.my_robot_id, allPoseIds);
  const auto translationGhosts =
      comm->exchangeTranslationBlocks(partition.my_robot_id, allPoseIds);
  const size_t expectedGhosts = numPoses - partition.global_pose_ids.size();
  if (rotationGhosts.size() != expectedGhosts ||
      translationGhosts.size() != expectedGhosts) {
    throw std::runtime_error("MPI DCCI ghost exchange returned wrong count");
  }
}

double relativePoseDiff(const Matrix &A, const Matrix &B) {
  return (A - B).norm() / std::max(1.0, B.norm());
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::string dataset = "synthetic_3d";
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--dataset" && i + 1 < argc) {
        dataset = argv[++i];
      }
    }
    const size_t d = dataset == "synthetic_2d" ? 2 : 3;
    if (dataset != "synthetic_2d" && dataset != "synthetic_3d") {
      throw std::invalid_argument("unknown dataset: " + dataset);
    }

    size_t numPoses = 0;
    const auto measurements = makeSyntheticGraph(d, &numPoses);
    const auto partitions = makeContiguousPartitions(numPoses, size);
    MPIDCCICommunicator comm(partitions[rank], MPI_COMM_WORLD);

    seedLocalBlocks(d, numPoses, partitions[rank], &comm);
    exerciseGhostExchange(numPoses, partitions[rank], &comm);

    DCCIStats stats;
    const Matrix distributed = distributedChordalInitialization(
        d, numPoses, measurements, partitions[rank], comm, tightDCCIParams(),
        &stats);
    const Matrix centralized = chordalInitialization(d, numPoses, measurements);
    const double relDiff = relativePoseDiff(distributed, centralized);
    double maxRelDiff = 0.0;
    MPI_Allreduce(&relDiff, &maxRelDiff, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "DCCI MPI dataset=" << dataset << " ranks=" << size
                << " relative_pose_matrix_diff=" << maxRelDiff
                << " rotation_iters=" << stats.rotation_pcg.iters
                << " translation_iters=" << stats.translation_pcg.iters
                << " scalar_reductions="
                << stats.communication.num_scalar_reductions
                << " block_messages=" << stats.communication.num_block_messages
                << " bytes_sent=" << stats.communication.bytes_sent
                << std::endl;
    }
    MPI_Finalize();
    return maxRelDiff < 1e-8 ? 0 : 2;
  } catch (const std::exception &e) {
    std::cerr << "rank " << rank << " error: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
}

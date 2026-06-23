#include <DPGO/DCCI_chordal_initialization.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/InProcessDCCICommunicator.h>
#include <DPGO/QuadraticOptimizer.h>
#include <DPGO/QuadraticProblem.h>
#include <DPGO/RelativeSEMeasurement.h>
#include <DPGO/RIFTIF.h>
#include <DPGO/TEDCCI.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

PoseID poseID(size_t pose) {
  return PoseID{0, static_cast<unsigned>(pose)};
}

RelativeSEMeasurement makeMeasurement(size_t i, size_t j, const Matrix &Ri,
                                      const Vector &ti, const Matrix &Rj,
                                      const Vector &tj, double kappa,
                                      double tau, double weight) {
  RelativeSEMeasurement measurement(0, 0, i, j, Ri.transpose() * Rj,
                                    Ri.transpose() * (tj - ti), kappa, tau);
  measurement.weight = weight;
  return measurement;
}

std::vector<RelativeSEMeasurement> makeSyntheticGraph(size_t d, size_t n) {
  std::vector<Matrix> rotations(n);
  std::vector<Vector> translations(n, Vector::Zero(d));
  for (size_t pose = 0; pose < n; ++pose) {
    const double x = static_cast<double>(pose);
    if (d == 2) {
      rotations[pose] = rotation2(0.03 * x + 0.01 * std::sin(0.17 * x));
      translations[pose] << 0.8 * x, 0.25 * std::sin(0.2 * x) + 0.03 * x;
    } else {
      rotations[pose] = rotation3(0.015 * x, -0.010 * x, 0.035 * x);
      translations[pose] << 0.8 * x, 0.25 * std::sin(0.2 * x),
          0.15 * std::cos(0.13 * x);
    }
  }

  std::vector<RelativeSEMeasurement> measurements;
  measurements.reserve(2 * n);
  for (size_t pose = 0; pose + 1 < n; ++pose) {
    const double weight = 0.6 + 0.4 * (static_cast<double>((pose % 5) + 1) / 5.0);
    measurements.emplace_back(makeMeasurement(
        pose, pose + 1, rotations[pose], translations[pose],
        rotations[pose + 1], translations[pose + 1], 2.0 + (pose % 7),
        3.0 + 0.25 * (pose % 11), weight));
  }
  for (size_t pose = 0; pose + 7 < n; pose += 7) {
    const size_t target = std::min(n - 1, pose + 11);
    const double weight = 0.5 + 0.1 * static_cast<double>(pose % 3);
    measurements.emplace_back(makeMeasurement(
        pose, target, rotations[pose], translations[pose], rotations[target],
        translations[target], 4.0 + (pose % 5), 2.5 + 0.2 * (pose % 4),
        weight));
  }
  if (n > 3) {
    measurements.emplace_back(makeMeasurement(
        n - 1, 0, rotations[n - 1], translations[n - 1], rotations[0],
        translations[0], 1.7, 2.2, 0.75));
  }
  return measurements;
}

std::vector<int> contiguousOwner(size_t numPoses, int numRobots) {
  std::vector<int> owner(numPoses, 0);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    owner[pose] = std::min<int>(
        static_cast<int>((pose * static_cast<size_t>(numRobots)) / numPoses),
        numRobots - 1);
  }
  return owner;
}

std::vector<RelativeSEMeasurement> normalizeMeasurementOwners(
    const std::vector<RelativeSEMeasurement> &measurements,
    const std::vector<int> &owner) {
  std::vector<RelativeSEMeasurement> normalized = measurements;
  for (auto &measurement : normalized) {
    if (measurement.p1 >= owner.size() || measurement.p2 >= owner.size()) {
      throw std::invalid_argument("measurement pose index out of range");
    }
    measurement.r1 = static_cast<unsigned>(owner[measurement.p1]);
    measurement.r2 = static_cast<unsigned>(owner[measurement.p2]);
  }
  return normalized;
}

std::vector<DistributedPartition> makeContiguousPartitions(size_t numPoses,
                                                           int numRobots) {
  std::vector<DistributedPartition> partitions(numRobots);
  std::unordered_map<PoseID, int, PoseIDHash> owner;
  const std::vector<int> owners = contiguousOwner(numPoses, numRobots);
  for (size_t pose = 0; pose < numPoses; ++pose) {
    owner[poseID(pose)] = owners[pose];
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

std::vector<TEDCCIPartition> makeTedContiguousPartitions(
    size_t numPoses, int numRobots,
    const std::vector<RelativeSEMeasurement> &measurements,
    const TEDCCIParams &params) {
  std::vector<RelativeSEMeasurement> localMeasurements;
  std::vector<RelativeSEMeasurement> sharedMeasurements;
  for (const auto &measurement : measurements) {
    if (measurement.r1 == measurement.r2) {
      localMeasurements.push_back(measurement);
    } else {
      sharedMeasurements.push_back(measurement);
    }
  }

  std::vector<TEDCCIPartition> partitions;
  partitions.reserve(static_cast<size_t>(numRobots));
  const std::vector<int> owner = contiguousOwner(numPoses, numRobots);
  for (int robot = 0; robot < numRobots; ++robot) {
    std::vector<PoseKey> localPoses;
    for (size_t pose = 0; pose < numPoses; ++pose) {
      if (owner[pose] == robot) {
        localPoses.push_back(PoseKey{robot, static_cast<int>(pose)});
      }
    }

    std::vector<RelativeSEMeasurement> robotLocalMeasurements;
    for (const auto &measurement : localMeasurements) {
      if (static_cast<int>(measurement.r1) == robot) {
        robotLocalMeasurements.push_back(measurement);
      }
    }
    partitions.push_back(BoundarySelector::SelectBoundaryVariables(
        robot, localPoses, robotLocalMeasurements, sharedMeasurements, params));
  }
  return partitions;
}

DCCIParams makeParams(unsigned maxIters, double relTol, double absTol,
                      bool useWeight) {
  DCCIParams params;
  params.rotation_pcg.max_iters = maxIters;
  params.rotation_pcg.rel_tol = relTol;
  params.rotation_pcg.abs_tol = absTol;
  params.translation_pcg = params.rotation_pcg;
  params.use_measurement_weight = useWeight;
  return params;
}

template <typename Fn>
double timeMilliseconds(Fn &&fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  const auto finish = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

double relativeDiff(const Matrix &A, const Matrix &B) {
  return (A - B).norm() / std::max(1.0, B.norm());
}

struct Options {
  size_t dimension = 3;
  size_t poses = 64;
  int robots = 4;
  unsigned maxIters = 500;
  double relTol = 1e-12;
  double absTol = 1e-12;
  bool useWeight = false;
  unsigned nonlinearRefinementIters = 0;
  std::string initializationMode = "dpcg_cci";
  bool compareAll = false;
  std::string g2oPath;
  std::string datasetName = "synthetic";
  std::string outputDistributedEstimate;
  std::string outputCentralizedEstimate;
  std::string outputDcciProcessDir;
  unsigned dcciProcessFrameStride = 0;
  unsigned dcciProcessMaxFrames = 16;
  RIFTInterfaceBackend interfaceBackend = RIFTInterfaceBackend::DIRECT_ORACLE;
  bool useRotationMultiRhs = true;
  bool forbidDirectInterfaceSolver = false;
  bool forbidGlobalInterfaceMatrix = false;
  bool forbidCollectives = false;
};

enum class BenchInitMode {
  DPCG_CCI,
  TED_CCI_SR_AUTO,
  TED_CCI_SR_DIRECT,
  TED_CCI_SR_HIERARCHICAL,
  TED_CCI_RIFT_IF,
  TED_CCI_ASYNC_DD,
};

std::string canonicalModeName(std::string mode) {
  for (char &ch : mode) {
    if (ch == '-') {
      ch = '_';
    } else {
      ch = static_cast<char>(
          std::tolower(static_cast<unsigned char>(ch)));
    }
  }
  return mode;
}

BenchInitMode parseBenchMode(const std::string &rawMode) {
  const std::string mode = canonicalModeName(rawMode);
  if (mode == "dpcg_cci" || mode == "d_cci_pcg") {
    return BenchInitMode::DPCG_CCI;
  }
  if (mode == "ted_cci_sr_auto" || mode == "ted_cci_auto") {
    return BenchInitMode::TED_CCI_SR_AUTO;
  }
  if (mode == "ted_cci_sr_direct") {
    return BenchInitMode::TED_CCI_SR_DIRECT;
  }
  if (mode == "ted_cci_sr_hierarchical") {
    return BenchInitMode::TED_CCI_SR_HIERARCHICAL;
  }
  if (mode == "ted_cci_rift_if" || mode == "rift_if" ||
      mode == "ted_cci_rift_exact") {
    return BenchInitMode::TED_CCI_RIFT_IF;
  }
  if (mode == "ted_cci_async_dd") {
    return BenchInitMode::TED_CCI_ASYNC_DD;
  }
  throw std::invalid_argument("unknown initialization mode: " + rawMode);
}

std::string modeName(BenchInitMode mode) {
  switch (mode) {
    case BenchInitMode::DPCG_CCI:
      return "dpcg_cci";
    case BenchInitMode::TED_CCI_SR_AUTO:
      return "ted_cci_sr_auto";
    case BenchInitMode::TED_CCI_SR_DIRECT:
      return "ted_cci_sr_direct";
    case BenchInitMode::TED_CCI_SR_HIERARCHICAL:
      return "ted_cci_sr_hierarchical";
    case BenchInitMode::TED_CCI_RIFT_IF:
      return "ted_cci_rift_if";
    case BenchInitMode::TED_CCI_ASYNC_DD:
      return "ted_cci_async_dd";
  }
  return "unknown";
}

std::string cciModeName(CCIInitMode mode) {
  switch (mode) {
    case CCIInitMode::CENTRALIZED_CCI:
      return "centralized_cci";
    case CCIInitMode::DPCG_CCI:
      return "dpcg_cci";
    case CCIInitMode::TED_CCI_SR_AUTO:
      return "ted_cci_sr_auto";
    case CCIInitMode::TED_CCI_SR_DIRECT:
      return "ted_cci_sr_direct";
    case CCIInitMode::TED_CCI_SR_HIERARCHICAL:
      return "ted_cci_sr_hierarchical";
    case CCIInitMode::TED_CCI_RIFT_IF:
      return "ted_cci_rift_if";
    case CCIInitMode::TED_CCI_ASYNC_DD:
      return "ted_cci_async_dd";
    case CCIInitMode::LOCAL_ONLY_CCI:
      return "local_only_cci";
  }
  return "unknown";
}

std::string tedBackendName(TEDCCIBackend backend) {
  switch (backend) {
    case TEDCCIBackend::AUTO:
      return "auto";
    case TEDCCIBackend::DENSE_HOUSEHOLDER:
      return "dense_householder";
    case TEDCCIBackend::SPARSE_SPQR:
      return "sparse_spqr";
  }
  return "unknown";
}

RIFTInterfaceBackend parseRiftBackend(const std::string &rawBackend) {
  const std::string backend = canonicalModeName(rawBackend);
  if (backend == "direct_oracle") {
    return RIFTInterfaceBackend::DIRECT_ORACLE;
  }
  if (backend == "rift_exact" || backend == "exact") {
    return RIFTInterfaceBackend::RIFT_EXACT;
  }
  if (backend == "rift_auto" || backend == "auto") {
    return RIFTInterfaceBackend::RIFT_AUTO;
  }
  if (backend == "rift_cak" || backend == "cak") {
    return RIFTInterfaceBackend::RIFT_CAK;
  }
  if (backend == "rift_async_schur" || backend == "async_schur") {
    return RIFTInterfaceBackend::RIFT_ASYNC_SCHUR;
  }
  throw std::invalid_argument("unknown RIFT interface backend: " + rawBackend);
}

bool parseBool(const std::string &value) {
  const std::string canonical = canonicalModeName(value);
  if (canonical == "1" || canonical == "true" || canonical == "yes" ||
      canonical == "on") {
    return true;
  }
  if (canonical == "0" || canonical == "false" || canonical == "no" ||
      canonical == "off") {
    return false;
  }
  throw std::invalid_argument("expected boolean value, got: " + value);
}

CCIInitMode tedMode(BenchInitMode mode) {
  switch (mode) {
    case BenchInitMode::TED_CCI_SR_AUTO:
      return CCIInitMode::TED_CCI_SR_AUTO;
    case BenchInitMode::TED_CCI_SR_DIRECT:
      return CCIInitMode::TED_CCI_SR_DIRECT;
    case BenchInitMode::TED_CCI_SR_HIERARCHICAL:
      return CCIInitMode::TED_CCI_SR_HIERARCHICAL;
    case BenchInitMode::TED_CCI_RIFT_IF:
      return CCIInitMode::TED_CCI_RIFT_IF;
    case BenchInitMode::TED_CCI_ASYNC_DD:
      return CCIInitMode::TED_CCI_ASYNC_DD;
    case BenchInitMode::DPCG_CCI:
      break;
  }
  throw std::invalid_argument("DPCG_CCI is not a TED-CCI mode");
}

Matrix poseRotation(const Matrix &T, size_t d, size_t pose) {
  return T.block(0, pose * (d + 1), d, d);
}

Vector poseTranslation(const Matrix &T, size_t d, size_t pose) {
  return T.block(0, pose * (d + 1) + d, d, 1);
}

double maxRotationBlockDiff(const Matrix &A, const Matrix &B, size_t d) {
  const size_t numPoses = static_cast<size_t>(A.cols()) / (d + 1);
  double maxDiff = 0.0;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    maxDiff = std::max(
        maxDiff,
        (poseRotation(A, d, pose) - poseRotation(B, d, pose)).norm());
  }
  return maxDiff;
}

double maxTranslationBlockDiff(const Matrix &A, const Matrix &B, size_t d) {
  const size_t numPoses = static_cast<size_t>(A.cols()) / (d + 1);
  double maxDiff = 0.0;
  for (size_t pose = 0; pose < numPoses; ++pose) {
    maxDiff = std::max(
        maxDiff,
        (poseTranslation(A, d, pose) - poseTranslation(B, d, pose)).norm());
  }
  return maxDiff;
}

double totalChordalCost(const Matrix &T,
                        const std::vector<RelativeSEMeasurement> &measurements,
                        bool useWeight) {
  if (measurements.empty()) {
    return 0.0;
  }
  const size_t d = static_cast<size_t>(T.rows());
  double cost = 0.0;
  for (const auto &measurement : measurements) {
    const double weight = useWeight ? measurement.weight : 1.0;
    const Matrix Ri = poseRotation(T, d, measurement.p1);
    const Matrix Rj = poseRotation(T, d, measurement.p2);
    const Vector ti = poseTranslation(T, d, measurement.p1);
    const Vector tj = poseTranslation(T, d, measurement.p2);
    cost += weight * measurement.kappa *
            (Ri * measurement.R - Rj).squaredNorm();
    cost += weight * measurement.tau *
            (tj - ti - Ri * measurement.t).squaredNorm();
  }
  return cost;
}

std::vector<RelativeSEMeasurement> effectiveMeasurementsForQuadraticProblem(
    const std::vector<RelativeSEMeasurement> &measurements, bool useWeight) {
  if (!useWeight) {
    return measurements;
  }
  std::vector<RelativeSEMeasurement> effective = measurements;
  for (auto &measurement : effective) {
    measurement.kappa *= measurement.weight;
    measurement.tau *= measurement.weight;
    measurement.weight = 1.0;
  }
  return effective;
}

std::pair<double, double> refinedPgoCostAfterNonlinearOptimization(
    const Matrix &initial, size_t d, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements, bool useWeight,
    unsigned maxIters) {
  Matrix refined;
  const std::vector<RelativeSEMeasurement> effectiveMeasurements =
      effectiveMeasurementsForQuadraticProblem(measurements, useWeight);
  QuadraticProblem problem(numPoses, d, d);
  problem.setQ(constructConnectionLaplacianSE(effectiveMeasurements));
  QuadraticOptimizer optimizer(&problem);
  optimizer.setAlgorithm(ROPTALG::RTR);
  optimizer.setTrustRegionIterations(maxIters);
  optimizer.setTrustRegionTolerance(1e-8);
  optimizer.setTrustRegionMaxInnerIterations(50);
  optimizer.setVerbose(false);
  const double elapsedMs = timeMilliseconds([&]() {
    refined = optimizer.optimize(initial);
  });
  return {2.0 * problem.f(refined), elapsedMs};
}

void writeInterleavedEstimate(const std::string &path, const Matrix &X,
                              size_t d) {
  Matrix aligned = X;
  const size_t numPoses = static_cast<size_t>(X.cols()) / (d + 1);
  if (numPoses > 0) {
    const Vector t0 = aligned.col(static_cast<int>(d));
    for (size_t pose = 0; pose < numPoses; ++pose) {
      aligned.col(static_cast<int>(pose * (d + 1) + d)) -= t0;
    }
    const Matrix R0 = aligned.block(0, 0, static_cast<int>(d),
                                    static_cast<int>(d)).transpose();
    aligned = R0 * aligned;
  }

  std::ofstream output(path);
  if (!output.is_open()) {
    throw std::runtime_error("unable to write estimate: " + path);
  }
  output << std::setprecision(16) << aligned << "\n";
}

const char *stageName(DCCITraceStage stage) {
  switch (stage) {
    case DCCITraceStage::TranslationPCG:
      return "D-CCI-PCG translation solve";
  }
  return "D-CCI-PCG solve";
}

void writeDcciProcessFrames(const std::string &directory,
                            const std::vector<DCCITraceFrame> &frames,
                            size_t d) {
  if (directory.empty()) {
    return;
  }
  if (frames.empty()) {
    throw std::runtime_error("D-CCI trace requested but no frames were recorded");
  }
  const std::filesystem::path dir(directory);
  std::filesystem::create_directories(dir);

  std::ofstream metadata(dir / "init_metadata.csv");
  if (!metadata.is_open()) {
    throw std::runtime_error("unable to write D-CCI trace metadata: " +
                             (dir / "init_metadata.csv").string());
  }
  metadata << "frame,stage,pcg_iter,residual_norm\n";
  for (size_t idx = 0; idx < frames.size(); ++idx) {
    std::ostringstream name;
    name << "init_" << std::setw(4) << std::setfill('0') << idx << ".txt";
    const std::filesystem::path path = dir / name.str();
    writeInterleavedEstimate(path.string(), frames[idx].poses, d);
    metadata << idx << "," << stageName(frames[idx].stage) << ","
             << frames[idx].pcg_iteration << ","
             << std::setprecision(16) << frames[idx].residual_norm << "\n";
  }
}

Options parseOptions(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto requireValue = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return std::string(argv[++i]);
    };
    if (arg == "--dimension") {
      options.dimension = static_cast<size_t>(
          std::stoul(requireValue("--dimension")));
    } else if (arg == "--poses") {
      options.poses = static_cast<size_t>(std::stoul(requireValue("--poses")));
    } else if (arg == "--robots") {
      options.robots = std::stoi(requireValue("--robots"));
    } else if (arg == "--max-iters") {
      options.maxIters =
          static_cast<unsigned>(std::stoul(requireValue("--max-iters")));
    } else if (arg == "--rel-tol") {
      options.relTol = std::stod(requireValue("--rel-tol"));
    } else if (arg == "--abs-tol") {
      options.absTol = std::stod(requireValue("--abs-tol"));
    } else if (arg == "--use-measurement-weight") {
      options.useWeight = true;
    } else if (arg == "--nonlinear-refinement-iters") {
      options.nonlinearRefinementIters =
          static_cast<unsigned>(
              std::stoul(requireValue("--nonlinear-refinement-iters")));
    } else if (arg == "--initialization-mode") {
      options.initializationMode =
          canonicalModeName(requireValue("--initialization-mode"));
    } else if (arg == "--compare-all") {
      options.compareAll = true;
    } else if (arg == "--g2o") {
      options.g2oPath = requireValue("--g2o");
    } else if (arg == "--dataset-name") {
      options.datasetName = requireValue("--dataset-name");
    } else if (arg == "--output-distributed-estimate") {
      options.outputDistributedEstimate =
          requireValue("--output-distributed-estimate");
    } else if (arg == "--output-centralized-estimate") {
      options.outputCentralizedEstimate =
          requireValue("--output-centralized-estimate");
    } else if (arg == "--output-dcci-process-dir") {
      options.outputDcciProcessDir = requireValue("--output-dcci-process-dir");
    } else if (arg == "--dcci-process-frame-stride") {
      options.dcciProcessFrameStride =
          static_cast<unsigned>(std::stoul(
              requireValue("--dcci-process-frame-stride")));
    } else if (arg == "--dcci-process-max-frames") {
      options.dcciProcessMaxFrames =
          static_cast<unsigned>(std::stoul(
              requireValue("--dcci-process-max-frames")));
    } else if (arg == "--interface-backend") {
      options.interfaceBackend =
          parseRiftBackend(requireValue("--interface-backend"));
    } else if (arg == "--use-rotation-multi-rhs") {
      options.useRotationMultiRhs =
          parseBool(requireValue("--use-rotation-multi-rhs"));
    } else if (arg == "--forbid-direct-interface-solver") {
      options.forbidDirectInterfaceSolver =
          parseBool(requireValue("--forbid-direct-interface-solver"));
    } else if (arg == "--forbid-global-interface-matrix") {
      options.forbidGlobalInterfaceMatrix =
          parseBool(requireValue("--forbid-global-interface-matrix"));
    } else if (arg == "--forbid-collectives") {
      options.forbidCollectives =
          parseBool(requireValue("--forbid-collectives"));
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (options.g2oPath.empty()) {
    if (options.dimension != 2 && options.dimension != 3) {
      throw std::invalid_argument("dimension must be 2 or 3");
    }
    if (options.poses < 2) {
      throw std::invalid_argument("poses must be at least 2");
    }
  }
  if (options.robots <= 0) {
    throw std::invalid_argument("robots must be positive");
  }
  if (!options.compareAll) {
    (void)parseBenchMode(options.initializationMode);
  }
  return options;
}

struct BenchRunResult {
  BenchInitMode mode = BenchInitMode::DPCG_CCI;
  Matrix estimate;
  double methodMs = 0.0;
  double methodCost = 0.0;
  double methodCostAbsGap = 0.0;
  double methodCostRelGap = 0.0;
  double methodRelativePoseMatrixDiff = 0.0;
  double methodRotationEquivalenceError = 0.0;
  double methodTranslationEquivalenceError = 0.0;
  bool finalPgoCostAfterNonlinearRefinementAvailable = false;
  double finalPgoCostAfterNonlinearRefinement =
      std::numeric_limits<double>::quiet_NaN();
  double nonlinearRefinementMs = 0.0;
  DCCIStats dcciStats;
  TEDCCIStats tedStats;
  std::vector<DCCITraceFrame> traceFrames;
};

double bytesToMb(std::size_t bytes) {
  return static_cast<double>(bytes) / 1048576.0;
}

double dcciScalarReductionMb(const DCCIStats &stats, int robots) {
  return static_cast<double>(stats.communication.num_scalar_reductions) *
         static_cast<double>(robots) * sizeof(double) / 1048576.0;
}

double dcciCommMb(const DCCIStats &stats, int robots) {
  return dcciScalarReductionMb(stats, robots) +
         bytesToMb(stats.communication.bytes_sent);
}

double tedCommMb(const TEDCCIStats &stats) {
  return bytesToMb(stats.bytes_sent_upward + stats.bytes_sent_downward);
}

std::size_t methodCommRounds(const BenchRunResult &result) {
  if (result.mode == BenchInitMode::DPCG_CCI) {
    return result.dcciStats.communication.num_scalar_reductions;
  }
  if (result.mode == BenchInitMode::TED_CCI_ASYNC_DD) {
    return static_cast<std::size_t>(
        std::max(0, result.tedStats.async_dd_iterations));
  }
  return (result.tedStats.num_factor_messages > 0 ||
          result.tedStats.num_solution_messages > 0)
             ? 2u
             : 0u;
}

double methodCommMb(const BenchRunResult &result, int robots) {
  if (result.mode == BenchInitMode::DPCG_CCI) {
    return dcciCommMb(result.dcciStats, robots);
  }
  return tedCommMb(result.tedStats);
}

TEDCCIParams makeTedParams(const Options &options, BenchInitMode mode) {
  TEDCCIParams params;
  params.mode = tedMode(mode);
  params.use_measurement_weight = options.useWeight;
  params.async_dd_max_iters = static_cast<int>(options.maxIters);
  params.async_dd_rel_tol = options.relTol;
  params.rift_interface_backend = options.interfaceBackend;
  if (mode == BenchInitMode::TED_CCI_RIFT_IF &&
      params.rift_interface_backend == RIFTInterfaceBackend::DIRECT_ORACLE) {
    params.rift_interface_backend = RIFTInterfaceBackend::RIFT_EXACT;
  }
  params.rift_use_rotation_multi_rhs = options.useRotationMultiRhs;
  params.rift_forbid_direct_interface_solver =
      options.forbidDirectInterfaceSolver;
  params.rift_forbid_global_interface_matrix =
      options.forbidGlobalInterfaceMatrix;
  params.rift_forbid_collectives = options.forbidCollectives;
  return params;
}

BenchRunResult runInitializationMode(
    BenchInitMode mode, const Options &options, size_t numPoses,
    const std::vector<RelativeSEMeasurement> &measurements,
    const std::vector<DistributedPartition> &dcciPartitions,
    const std::vector<TEDCCIPartition> &tedPartitions,
    const Matrix &centralized, double cciCost, bool recordDcciProcessFrames) {
  BenchRunResult result;
  result.mode = mode;
  if (mode == BenchInitMode::DPCG_CCI) {
    InProcessDCCICommunicator comm(dcciPartitions);
    const DCCIParams params =
        makeParams(options.maxIters, options.relTol, options.absTol,
                   options.useWeight);
    DCCITraceOptions trace;
    if (recordDcciProcessFrames) {
      trace.frames = &result.traceFrames;
      trace.frame_stride = options.dcciProcessFrameStride;
      trace.max_frames_per_stage = options.dcciProcessMaxFrames;
    }
    result.methodMs = timeMilliseconds([&]() {
      result.estimate = distributedChordalInitialization(
          options.dimension, numPoses, measurements, dcciPartitions.front(),
          comm, params, &result.dcciStats,
          recordDcciProcessFrames ? &trace : nullptr);
    });
  } else {
    const TEDCCIParams params = makeTedParams(options, mode);
    result.methodMs = timeMilliseconds([&]() {
      result.estimate = TEDCCISolver::Initialize(
          static_cast<int>(options.dimension), static_cast<int>(numPoses),
          measurements, tedPartitions, params, &result.tedStats);
    });
  }

  result.methodCost =
      totalChordalCost(result.estimate, measurements, options.useWeight);
  result.methodCostAbsGap = std::abs(result.methodCost - cciCost);
  result.methodCostRelGap =
      result.methodCostAbsGap / std::max(1.0, std::abs(cciCost));
  result.methodRelativePoseMatrixDiff =
      relativeDiff(result.estimate, centralized);
  result.methodRotationEquivalenceError =
      maxRotationBlockDiff(result.estimate, centralized, options.dimension);
  result.methodTranslationEquivalenceError =
      maxTranslationBlockDiff(result.estimate, centralized, options.dimension);
  if (options.nonlinearRefinementIters > 0) {
    const auto refined = refinedPgoCostAfterNonlinearOptimization(
        result.estimate, options.dimension, numPoses, measurements,
        options.useWeight, options.nonlinearRefinementIters);
    result.finalPgoCostAfterNonlinearRefinementAvailable = true;
    result.finalPgoCostAfterNonlinearRefinement = refined.first;
    result.nonlinearRefinementMs = refined.second;
  }
  return result;
}

void printBenchLine(const Options &options, size_t numPoses, size_t numEdges,
                    const std::string &g2oLabel, double centralizedMs,
                    double cciCost, const BenchRunResult &result) {
  const double scalarReductionMb =
      dcciScalarReductionMb(result.dcciStats, options.robots);
  std::cout << std::setprecision(16)
            << "BENCH_DCCI"
            << " dataset=" << options.datasetName
            << " g2o=" << g2oLabel
            << " dimension=" << options.dimension
            << " poses=" << numPoses
            << " edges=" << numEdges
            << " robots=" << options.robots
            << " use_measurement_weight=" << options.useWeight
            << " initialization_mode=" << modeName(result.mode)
            << " centralized_ms=" << centralizedMs
            << " dcci_ms=" << result.methodMs
            << " cci_cost=" << cciCost
            << " dcci_cost=" << result.methodCost
            << " cost_abs_gap=" << result.methodCostAbsGap
            << " cost_rel_gap=" << result.methodCostRelGap
            << " relative_pose_matrix_diff="
            << result.methodRelativePoseMatrixDiff
            << " rotation_iters=" << result.dcciStats.rotation_pcg.iters
            << " translation_iters=" << result.dcciStats.translation_pcg.iters
            << " rotation_residual="
            << result.dcciStats.rotation_pcg.final_residual_norm
            << " translation_residual="
            << result.dcciStats.translation_pcg.final_residual_norm
            << " dcci_trace_frames=" << result.traceFrames.size()
            << " scalar_reductions="
            << result.dcciStats.communication.num_scalar_reductions
            << " estimated_scalar_reduction_mb=" << scalarReductionMb
            << " block_messages="
            << result.dcciStats.communication.num_block_messages
            << " bytes_sent=" << result.dcciStats.communication.bytes_sent
            << " method_ms=" << result.methodMs
            << " method_cost=" << result.methodCost
            << " method_cost_abs_gap=" << result.methodCostAbsGap
            << " method_cost_rel_gap=" << result.methodCostRelGap
            << " method_relative_pose_matrix_diff="
            << result.methodRelativePoseMatrixDiff
            << " method_rotation_equivalence_error="
            << result.methodRotationEquivalenceError
            << " method_translation_equivalence_error="
            << result.methodTranslationEquivalenceError
            << " nonlinear_refinement_iters="
            << options.nonlinearRefinementIters
            << " final_pgo_cost_after_nonlinear_refinement_available="
            << result.finalPgoCostAfterNonlinearRefinementAvailable
            << " final_pgo_cost_after_nonlinear_refinement="
            << result.finalPgoCostAfterNonlinearRefinement
            << " nonlinear_refinement_ms=" << result.nonlinearRefinementMs
            << " method_comm_rounds=" << methodCommRounds(result)
            << " method_comm_mb=" << methodCommMb(result, options.robots)
            << " ted_local_qr_ms=" << result.tedStats.local_qr_ms
            << " ted_interface_solve_ms="
            << result.tedStats.interface_solve_ms
            << " ted_num_interface_vars="
            << result.tedStats.num_interface_vars
            << " ted_num_condensed_rows="
            << result.tedStats.num_condensed_rows
            << " ted_max_separator_size="
            << result.tedStats.max_separator_size
            << " ted_factor_messages="
            << result.tedStats.num_factor_messages
            << " ted_solution_messages="
            << result.tedStats.num_solution_messages
            << " ted_bytes_upward=" << result.tedStats.bytes_sent_upward
            << " ted_bytes_downward=" << result.tedStats.bytes_sent_downward
            << " ted_comm_mb=" << tedCommMb(result.tedStats)
            << " ted_effective_rotation_backend="
            << cciModeName(result.tedStats.effective_rotation_mode)
            << " ted_effective_translation_backend="
            << cciModeName(result.tedStats.effective_translation_mode)
            << " ted_condensation_backend="
            << tedBackendName(result.tedStats.effective_condensation_backend)
            << " ted_interface_backend="
            << tedBackendName(result.tedStats.effective_interface_backend)
            << " ted_spqr_rank_deficient_fallbacks="
            << result.tedStats.spqr_rank_deficient_fallbacks
            << " ted_peak_interface_cols="
            << result.tedStats.peak_interface_cols
            << " ted_factor_dense_bytes_upward="
            << result.tedStats.factor_dense_bytes_upward
            << " ted_factor_sparse_triplet_bytes_upward="
            << result.tedStats.factor_sparse_triplet_bytes_upward
            << " ted_factor_sparse_triplet_nonzeros="
            << result.tedStats.factor_sparse_triplet_nonzeros
            << " ted_async_dd_iterations="
            << result.tedStats.async_dd_iterations
            << " ted_async_dd_converged="
            << result.tedStats.async_dd_converged
            << " ted_async_dd_initial_residual="
            << result.tedStats.async_dd_initial_residual
            << " ted_async_dd_final_residual="
            << result.tedStats.async_dd_final_residual
            << " rift_selected_backend="
            << RIFTInterfaceBackendName(result.tedStats.rift_selected_backend)
            << " rift_num_cliques=" << result.tedStats.rift_num_cliques
            << " rift_num_tree_edges=" << result.tedStats.rift_num_tree_edges
            << " rift_num_host_robots="
            << result.tedStats.rift_num_host_robots
            << " rift_max_host_clique_load="
            << result.tedStats.rift_max_host_clique_load
            << " rift_cross_host_tree_edges="
            << result.tedStats.rift_cross_host_tree_edges
            << " rift_estimated_route_hops="
            << result.tedStats.rift_estimated_route_hops
            << " rift_max_clique_blocks="
            << result.tedStats.rift_max_clique_blocks
            << " rift_max_separator_blocks="
            << result.tedStats.rift_max_separator_blocks
            << " rift_estimated_message_bytes="
            << result.tedStats.rift_estimated_message_bytes
            << " rift_estimated_routed_message_bytes="
            << result.tedStats.rift_estimated_routed_message_bytes
            << " rift_actual_message_bytes="
            << result.tedStats.rift_actual_message_bytes
            << " rift_directed_messages_sent="
            << result.tedStats.rift_directed_messages_sent
            << " rift_cak_iterations="
            << result.tedStats.rift_cak_iterations
            << " rift_cak_scalar_reductions="
            << result.tedStats.rift_cak_scalar_reductions
            << " rift_cak_scalar_reduction_bytes="
            << result.tedStats.rift_cak_scalar_reduction_bytes
            << " rift_cak_final_residual="
            << result.tedStats.rift_cak_final_residual
            << " rift_symbolic_ms=" << result.tedStats.rift_symbolic_ms
            << " rift_message_qr_ms="
            << result.tedStats.rift_message_qr_ms
            << " rift_belief_solve_ms="
            << result.tedStats.rift_belief_solve_ms
            << " rift_final_interface_residual="
            << result.tedStats.rift_final_interface_residual
            << " rift_used_global_matrix="
            << result.tedStats.rift_used_global_matrix
            << " rift_used_direct_solver="
            << result.tedStats.rift_used_direct_solver
            << " rift_used_collective="
            << result.tedStats.rift_used_collective
            << std::endl;
}

std::string modeOutputPath(const std::string &base, BenchInitMode mode) {
  if (base.empty()) {
    return "";
  }
  const std::filesystem::path path(base);
  const std::string stem = path.stem().string();
  const std::string extension = path.extension().string();
  const std::filesystem::path parent = path.parent_path();
  const std::string file =
      stem + "_" + modeName(mode) + (extension.empty() ? ".txt" : extension);
  return (parent / file).string();
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Options options = parseOptions(argc, argv);
    size_t numPoses = options.poses;
    std::vector<RelativeSEMeasurement> measurements;
    if (options.g2oPath.empty()) {
      measurements = makeSyntheticGraph(options.dimension, numPoses);
    } else {
      measurements = read_g2o_file(options.g2oPath, numPoses);
      if (measurements.empty()) {
        throw std::invalid_argument("g2o file has no supported measurements");
      }
      options.dimension = static_cast<size_t>(measurements.front().t.rows());
      options.poses = numPoses;
      if (options.datasetName == "synthetic") {
        options.datasetName = options.g2oPath;
      }
    }
    if (options.dimension != 2 && options.dimension != 3) {
      throw std::invalid_argument("detected dimension must be 2 or 3");
    }
    measurements = normalizeMeasurementOwners(
        measurements, contiguousOwner(numPoses, options.robots));
    const auto dcciPartitions =
        makeContiguousPartitions(numPoses, options.robots);
    TEDCCIParams tedPartitionParams;
    tedPartitionParams.use_measurement_weight = options.useWeight;
    const auto tedPartitions = makeTedContiguousPartitions(
        numPoses, options.robots, measurements, tedPartitionParams);

    Matrix centralized;
    const double centralizedMs = timeMilliseconds([&]() {
      centralized = chordalInitialization(options.dimension, numPoses,
                                          measurements, options.useWeight);
    });
    const double cciCost =
        totalChordalCost(centralized, measurements, options.useWeight);

    std::vector<BenchInitMode> modes;
    if (options.compareAll) {
      modes = {BenchInitMode::DPCG_CCI, BenchInitMode::TED_CCI_SR_AUTO,
               BenchInitMode::TED_CCI_SR_DIRECT,
               BenchInitMode::TED_CCI_SR_HIERARCHICAL};
    } else {
      modes.push_back(parseBenchMode(options.initializationMode));
    }

    const bool hasDpcgMode =
        std::find(modes.begin(), modes.end(), BenchInitMode::DPCG_CCI) !=
        modes.end();
    if (!options.outputDcciProcessDir.empty() && !hasDpcgMode) {
      throw std::invalid_argument(
          "--output-dcci-process-dir is only supported for dpcg_cci");
    }

    if (!options.outputCentralizedEstimate.empty()) {
      writeInterleavedEstimate(options.outputCentralizedEstimate, centralized,
                               options.dimension);
    }

    const std::string g2oLabel =
        options.g2oPath.empty() ? "synthetic" : options.g2oPath;
    for (BenchInitMode mode : modes) {
      const bool recordDcciProcessFrames =
          mode == BenchInitMode::DPCG_CCI &&
          !options.outputDcciProcessDir.empty();
      BenchRunResult result = runInitializationMode(
          mode, options, numPoses, measurements, dcciPartitions, tedPartitions,
          centralized, cciCost, recordDcciProcessFrames);

      if (!options.outputDistributedEstimate.empty()) {
        const std::string outputPath =
            modes.size() == 1
                ? options.outputDistributedEstimate
                : modeOutputPath(options.outputDistributedEstimate, mode);
        writeInterleavedEstimate(outputPath, result.estimate,
                                 options.dimension);
      }
      if (recordDcciProcessFrames) {
        writeDcciProcessFrames(options.outputDcciProcessDir, result.traceFrames,
                               options.dimension);
      }
      printBenchLine(options, numPoses, measurements.size(), g2oLabel,
                     centralizedMs, cciCost, result);
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "bench-dcci error: " << e.what() << std::endl;
    return 1;
  }
}

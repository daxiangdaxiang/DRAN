/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#include <DPGO/ManualObjectDran.h>

#include <DPGO/BoundarySchurPreconditioner.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/ObjectPGOData.h>
#include <DPGO/QuadraticOptimizer.h>
#include <DPGO/QuadraticProblem.h>
#include <DPGO/ReducedRotationQuadraticOptimizer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace DPGO {
namespace {

struct WeightedNeighbor {
  unsigned id{0};
  double weight{1.0};
};

using NeighborRound = std::vector<std::vector<WeightedNeighbor>>;
using NeighborStateCache = std::vector<std::vector<Matrix>>;
using NeighborObjectAges =
    std::vector<std::vector<std::vector<std::size_t>>>;
using ObjectReceiveSelection =
    std::vector<std::vector<std::vector<bool>>>;

struct ObjectInitializationResult {
  std::vector<Matrix> states;
  std::size_t observedObjectCopies{0};
  std::size_t relayOnlyObjectCopies{0};
  std::size_t initializedFromNeighborObjectCopies{0};
  std::size_t staleObjectCopies{0};
};

struct ObjectExchangeStats {
  std::size_t communicated{0};
  std::size_t staleObjectCommTriggers{0};
  std::vector<std::vector<bool>> selectedForSend;
  ObjectReceiveSelection selectedForReceiver;
};

std::vector<std::string> splitCsvLine(const std::string &line) {
  std::vector<std::string> tokens;
  std::string token;
  std::stringstream ss(line);
  while (std::getline(ss, token, ',')) {
    tokens.push_back(token);
  }
  return tokens;
}

void addWeightedNeighbor(std::vector<WeightedNeighbor> &neighbors,
                         unsigned id, double weight) {
  for (WeightedNeighbor &neighbor : neighbors) {
    if (neighbor.id == id) {
      neighbor.weight = weight;
      return;
    }
  }
  neighbors.push_back(WeightedNeighbor{id, weight});
}

std::vector<RelativeSEMeasurement> objectAwareLocalMeasurements(
    const ObjectPGORobotData &robot) {
  std::vector<RelativeSEMeasurement> measurements = robot.trajectoryMeasurements;
  measurements.reserve(measurements.size() +
                       robot.objectObservationMeasurements.size());
  for (const ObjectObservationMeasurement &obs :
       robot.objectObservationMeasurements) {
    RelativeSEMeasurement m = obs.measurement;
    m.p1 = obs.trajectoryLocalIndex;
    m.p2 = robot.trajectoryVertexIds.size() + obs.objectLocalIndex;
    m.r1 = robot.robotId;
    m.r2 = robot.robotId;
    measurements.push_back(std::move(m));
  }
  return measurements;
}

SparseMatrix fullConnectionLaplacian(
    const std::vector<RelativeSEMeasurement> &measurements,
    std::size_t numPoses, unsigned d) {
  SparseMatrix full((d + 1) * numPoses, (d + 1) * numPoses);
  if (measurements.empty()) {
    return full;
  }
  const SparseMatrix compact = constructConnectionLaplacianSE(measurements);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(static_cast<std::size_t>(compact.nonZeros()));
  for (int outer = 0; outer < compact.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(compact, outer); it; ++it) {
      if (it.row() < full.rows() && it.col() < full.cols()) {
        triplets.emplace_back(it.row(), it.col(), it.value());
      }
    }
  }
  full.setFromTriplets(triplets.begin(), triplets.end());
  full.makeCompressed();
  return full;
}

Matrix assembleObjectAwareChordalState(
    const ObjectPGODirectoryData &dataset, std::size_t robotId,
    const Matrix &globalT, const std::vector<std::size_t> &trajectoryOffsets,
    std::size_t objectOffset, unsigned d) {
  const ObjectPGORobotData &robot = dataset.robots[robotId];
  const std::size_t localNumPoses =
      robot.trajectoryVertexIds.size() + dataset.numObjects;
  Matrix state = Matrix::Zero(d, localNumPoses * (d + 1));
  for (std::size_t localIdx = 0; localIdx < robot.trajectoryVertexIds.size();
       ++localIdx) {
    const std::size_t globalIdx = trajectoryOffsets[robotId] + localIdx;
    state.block(0, localIdx * (d + 1), d, d + 1) =
        globalT.block(0, globalIdx * (d + 1), d, d + 1);
  }
  for (std::size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
    const std::size_t localIdx = robot.trajectoryVertexIds.size() + objectId;
    const std::size_t globalIdx = objectOffset + objectId;
    state.block(0, localIdx * (d + 1), d, d + 1) =
        globalT.block(0, globalIdx * (d + 1), d, d + 1);
  }
  return state;
}

std::vector<Matrix> centralizedObjectAwareChordalStates(
    const ObjectPGODirectoryData &dataset, unsigned d) {
  std::vector<std::size_t> trajectoryOffsets(dataset.robots.size(), 0);
  std::size_t trajectoryCount = 0;
  for (std::size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    trajectoryOffsets[robotId] = trajectoryCount;
    trajectoryCount += dataset.robots[robotId].trajectoryVertexIds.size();
  }
  const std::size_t objectOffset = trajectoryCount;
  const std::size_t totalPoses = trajectoryCount + dataset.numObjects;

  std::vector<RelativeSEMeasurement> globalMeasurements;
  for (std::size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    for (RelativeSEMeasurement m : robot.trajectoryMeasurements) {
      m.p1 = trajectoryOffsets[robotId] + m.p1;
      m.p2 = trajectoryOffsets[robotId] + m.p2;
      m.r1 = 0;
      m.r2 = 0;
      globalMeasurements.push_back(std::move(m));
    }
    for (const ObjectObservationMeasurement &obs :
         robot.objectObservationMeasurements) {
      RelativeSEMeasurement m = obs.measurement;
      m.p1 = trajectoryOffsets[robotId] + obs.trajectoryLocalIndex;
      m.p2 = objectOffset + obs.objectGlobalVertexId;
      m.r1 = 0;
      m.r2 = 0;
      globalMeasurements.push_back(std::move(m));
    }
  }

  const Matrix globalT = chordalInitialization(d, totalPoses, globalMeasurements);
  std::vector<Matrix> states;
  states.reserve(dataset.robots.size());
  for (std::size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    states.push_back(assembleObjectAwareChordalState(
        dataset, robotId, globalT, trajectoryOffsets, objectOffset, d));
  }
  return states;
}

NeighborRound makeRingNeighbors(unsigned numRobots, unsigned hops) {
  NeighborRound neighbors(numRobots);
  if (numRobots < 2) {
    return neighbors;
  }
  hops = std::max(1u, std::min(hops, numRobots - 1));
  for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
    for (unsigned offset = 1; offset <= hops; ++offset) {
      const unsigned prev = (robotId + numRobots - offset) % numRobots;
      const unsigned next = (robotId + offset) % numRobots;
      auto add = [&](unsigned id) {
        if (id != robotId) {
          addWeightedNeighbor(neighbors[robotId], id, 1.0);
        }
      };
      add(prev);
      add(next);
    }
  }
  return neighbors;
}

NeighborRound makeCompleteNeighbors(unsigned numRobots) {
  NeighborRound neighbors(numRobots);
  for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
    for (unsigned other = 0; other < numRobots; ++other) {
      if (other != robotId) {
        addWeightedNeighbor(neighbors[robotId], other, 1.0);
      }
    }
  }
  return neighbors;
}

std::vector<NeighborRound> readTopologySchedule(
    const std::string &path, unsigned numRobots,
    const std::string &weightMode) {
  std::vector<NeighborRound> schedule;
  if (path.empty()) {
    return schedule;
  }
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Could not open communication topology file: " +
                             path);
  }
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const std::vector<std::string> tokens = splitCsvLine(line);
    if (tokens.size() < 4 || tokens[0] == "round") {
      continue;
    }
    const unsigned roundIdx =
        static_cast<unsigned>(std::stoul(tokens[0]));
    const unsigned src = static_cast<unsigned>(std::stoul(tokens[1]));
    const unsigned dst = static_cast<unsigned>(std::stoul(tokens[2]));
    double weight = std::stod(tokens[3]);
    if (weightMode == "unit") {
      weight = 1.0;
    } else if (weightMode != "matrix") {
      throw std::runtime_error("Unsupported communication weight mode: " +
                               weightMode);
    }
    if (!std::isfinite(weight) || weight <= 0.0 || weight > 1.0) {
      throw std::runtime_error("Invalid communication weight in " + path);
    }
    if (src >= numRobots || dst >= numRobots || src == dst) {
      throw std::runtime_error("Invalid communication edge in " + path);
    }
    if (schedule.size() <= roundIdx) {
      schedule.resize(roundIdx + 1, NeighborRound(numRobots));
    }
    addWeightedNeighbor(schedule[roundIdx][src], dst, weight);
    addWeightedNeighbor(schedule[roundIdx][dst], src, weight);
  }
  for (std::size_t roundIdx = 0; roundIdx < schedule.size(); ++roundIdx) {
    bool hasEdge = false;
    for (const std::vector<WeightedNeighbor> &neighbors : schedule[roundIdx]) {
      if (!neighbors.empty()) {
        hasEdge = true;
        break;
      }
    }
    if (!hasEdge) {
      throw std::runtime_error(
          "Communication topology rounds must be zero-based and contiguous: " +
          path);
    }
  }
  return schedule;
}

NeighborRound unionNeighbors(const std::vector<NeighborRound> &schedule,
                             unsigned numRobots) {
  NeighborRound result(numRobots);
  for (const NeighborRound &round : schedule) {
    if (round.size() != numRobots) {
      throw std::runtime_error("Invalid topology schedule round size");
    }
    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      for (const WeightedNeighbor &neighbor : round[robotId]) {
        addWeightedNeighbor(result[robotId], neighbor.id, neighbor.weight);
      }
    }
  }
  return result;
}

std::vector<NeighborRound> makeTopologySchedule(
    unsigned numRobots, const ManualObjectDranOptions &options) {
  std::vector<NeighborRound> schedule =
      readTopologySchedule(options.topologyFile, numRobots,
                           options.topologyWeightMode);
  if (!options.topologyFile.empty() && schedule.empty()) {
    throw std::runtime_error("Communication topology file has no edges: " +
                             options.topologyFile);
  }
  if (!schedule.empty()) {
    return schedule;
  }
  schedule.push_back(options.topology == ManualObjectDranTopology::Complete
                         ? makeCompleteNeighbors(numRobots)
                         : makeRingNeighbors(numRobots, options.ringHops));
  return schedule;
}

std::vector<bool> observedObjects(const ObjectPGORobotData &robot,
                                  std::size_t numObjects) {
  std::vector<bool> observed(numObjects, false);
  for (const ObjectObservationMeasurement &obs :
       robot.objectObservationMeasurements) {
    if (obs.objectGlobalVertexId < numObjects) {
      observed[obs.objectGlobalVertexId] = true;
    }
  }
  return observed;
}

std::vector<std::vector<double>> objectObservationInformationScales(
    const ObjectPGODirectoryData &dataset, unsigned d) {
  std::vector<std::vector<double>> information(
      dataset.numRobots, std::vector<double>(dataset.numObjects, 0.0));
  double positiveSum = 0.0;
  std::size_t positiveCount = 0;
  for (std::size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
    for (const ObjectObservationMeasurement &obs :
         dataset.robots[robotId].objectObservationMeasurements) {
      if (obs.objectGlobalVertexId >= dataset.numObjects) {
        continue;
      }
      const double kappa = std::isfinite(obs.measurement.kappa)
                               ? std::max(0.0, obs.measurement.kappa)
                               : 0.0;
      const double tau = std::isfinite(obs.measurement.tau)
                             ? std::max(0.0, obs.measurement.tau)
                             : 0.0;
      information[robotId][obs.objectGlobalVertexId] +=
          static_cast<double>(d) * (kappa + tau);
    }
  }
  for (const std::vector<double> &robotInformation : information) {
    for (double value : robotInformation) {
      if (value > 0.0) {
        positiveSum += value;
        ++positiveCount;
      }
    }
  }
  if (positiveCount == 0 || positiveSum <= 0.0) {
    for (std::vector<double> &robotScales : information) {
      std::fill(robotScales.begin(), robotScales.end(), 1.0);
    }
    return information;
  }
  const double meanInformation =
      positiveSum / static_cast<double>(positiveCount);
  for (std::vector<double> &robotScales : information) {
    for (double &value : robotScales) {
      value = value > 0.0 ? value / meanInformation : 1.0;
    }
  }
  return information;
}

std::size_t countObservedObjectCopies(const ObjectPGODirectoryData &dataset) {
  std::size_t count = 0;
  for (const ObjectPGORobotData &robot : dataset.robots) {
    const std::vector<bool> observed =
        observedObjects(robot, dataset.numObjects);
    count += static_cast<std::size_t>(
        std::count(observed.begin(), observed.end(), true));
  }
  return count;
}

Matrix blockPose(const Matrix &state, std::size_t poseIndex, unsigned d) {
  return state.block(0, poseIndex * (d + 1), d, d + 1);
}

Matrix identityPoseBlock(unsigned d) {
  Matrix pose = Matrix::Zero(d, d + 1);
  pose.leftCols(d).setIdentity();
  return pose;
}

Matrix composePoseBlocks(const Matrix &left, const Matrix &right, unsigned d) {
  Matrix pose = Matrix::Zero(d, d + 1);
  pose.leftCols(d) = left.leftCols(d) * right.leftCols(d);
  pose.rightCols(1) =
      left.rightCols(1) + left.leftCols(d) * right.rightCols(1);
  return pose;
}

bool isValidPoseBlock(const Matrix &pose, unsigned d) {
  return pose.rows() == static_cast<int>(d) &&
         pose.cols() == static_cast<int>(d + 1) && pose.allFinite();
}

void setPoseBlock(Matrix &state, std::size_t poseIndex, const Matrix &pose,
                  unsigned d) {
  state.block(0, poseIndex * (d + 1), d, d + 1) = pose;
}

Matrix projectStateToSE(const Matrix &state, unsigned d) {
  Matrix projected = state;
  if (projected.rows() != static_cast<int>(d) || d == 0) {
    return projected;
  }
  const unsigned blockWidth = d + 1;
  const std::size_t numPoses =
      static_cast<std::size_t>(projected.cols()) / blockWidth;
  for (std::size_t poseId = 0; poseId < numPoses; ++poseId) {
    projected.block(0, poseId * blockWidth, d, d) =
        projectToRotationGroup(
            projected.block(0, poseId * blockWidth, d, d));
  }
  return projected;
}

std::vector<Matrix> projectStatesToSE(const std::vector<Matrix> &states,
                                      unsigned d) {
  std::vector<Matrix> projected;
  projected.reserve(states.size());
  for (const Matrix &state : states) {
    projected.push_back(projectStateToSE(state, d));
  }
  return projected;
}

Matrix assembleObjectAwareInputState(const ObjectPGODirectoryData &dataset,
                                     std::size_t robotId, unsigned d) {
  const ObjectPGORobotData &robot = dataset.robots[robotId];
  const std::size_t localNumPoses =
      robot.trajectoryVertexIds.size() + dataset.numObjects;
  Matrix state = Matrix::Zero(d, localNumPoses * (d + 1));
  for (std::size_t localIdx = 0; localIdx < robot.trajectoryVertexIds.size();
       ++localIdx) {
    const Matrix pose =
        localIdx < robot.trajectoryInitialPoses.size()
            ? robot.trajectoryInitialPoses[localIdx]
            : Matrix();
    setPoseBlock(state, localIdx,
                 isValidPoseBlock(pose, d) ? pose : identityPoseBlock(d), d);
  }
  for (std::size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
    const std::size_t localIdx = robot.trajectoryVertexIds.size() + objectId;
    const Matrix pose =
        objectId < robot.objectInitialPoses.size()
            ? robot.objectInitialPoses[objectId]
            : Matrix();
    setPoseBlock(state, localIdx,
                 isValidPoseBlock(pose, d) ? pose : identityPoseBlock(d), d);
  }
  return state;
}

std::vector<Matrix> localObjectAwareChordalStates(
    const ObjectPGODirectoryData &dataset, unsigned d) {
  std::vector<Matrix> states;
  states.reserve(dataset.robots.size());
  for (const ObjectPGORobotData &robot : dataset.robots) {
    const std::size_t localNumPoses =
        robot.trajectoryVertexIds.size() + dataset.numObjects;
    Matrix state = assembleObjectAwareInputState(dataset, robot.robotId, d);
    const std::vector<RelativeSEMeasurement> measurements =
        objectAwareLocalMeasurements(robot);
    if (!measurements.empty()) {
      std::vector<int> localToCompact(localNumPoses, -1);
      std::vector<std::size_t> compactToLocal;
      auto compactIndex = [&](std::size_t localIdx) {
        if (localIdx >= localToCompact.size()) {
          throw std::runtime_error(
              "Object-aware local measurement index is out of range");
        }
        int &index = localToCompact[localIdx];
        if (index < 0) {
          index = static_cast<int>(compactToLocal.size());
          compactToLocal.push_back(localIdx);
        }
        return static_cast<std::size_t>(index);
      };

      std::vector<RelativeSEMeasurement> compactMeasurements;
      compactMeasurements.reserve(measurements.size());
      for (RelativeSEMeasurement m : measurements) {
        m.p1 = compactIndex(m.p1);
        m.p2 = compactIndex(m.p2);
        compactMeasurements.push_back(std::move(m));
      }

      const Matrix chordal =
          chordalInitialization(d, compactToLocal.size(), compactMeasurements);
      Matrix gauge = identityPoseBlock(d);
      if (!compactToLocal.empty()) {
        const Matrix inputAnchor = blockPose(state, compactToLocal.front(), d);
        if (isValidPoseBlock(inputAnchor, d)) {
          gauge = inputAnchor;
        }
      }
      for (std::size_t compactIdx = 0; compactIdx < compactToLocal.size();
           ++compactIdx) {
        const Matrix pose =
            composePoseBlocks(gauge, blockPose(chordal, compactIdx, d), d);
        if (isValidPoseBlock(pose, d)) {
          setPoseBlock(state, compactToLocal[compactIdx], pose, d);
        }
      }
    }
    states.push_back(projectStateToSE(state, d));
  }
  return states;
}

ObjectInitializationResult initializeObjectStates(
    const ObjectPGODirectoryData &dataset, unsigned d,
    const NeighborRound &initializationNeighbors,
    const ManualObjectDranOptions &options) {
  ObjectInitializationResult result;
  result.observedObjectCopies = countObservedObjectCopies(dataset);
  result.relayOnlyObjectCopies =
      dataset.numRobots * dataset.numObjects - result.observedObjectCopies;
  if (options.objectInitialization ==
      ManualObjectDranObjectInitialization::CentralizedChordal) {
    result.states =
        projectStatesToSE(centralizedObjectAwareChordalStates(dataset, d), d);
    return result;
  }

  result.states = localObjectAwareChordalStates(dataset, d);
  std::vector<std::vector<bool>> initialized(
      dataset.numRobots, std::vector<bool>(dataset.numObjects, false));
  std::vector<std::vector<bool>> initializedFromNeighbor(
      dataset.numRobots, std::vector<bool>(dataset.numObjects, false));
  for (std::size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
    initialized[robotId] = observedObjects(dataset.robots[robotId],
                                           dataset.numObjects);
  }
  const std::vector<std::vector<bool>> observedByRobot = initialized;

  for (std::size_t pass = 0; pass < dataset.numRobots; ++pass) {
    bool changed = false;
    std::vector<Matrix> nextStates = result.states;
    std::vector<std::vector<bool>> nextInitialized = initialized;
    for (std::size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
      const ObjectPGORobotData &robot = dataset.robots[robotId];
      const std::size_t objectOffset = robot.trajectoryVertexIds.size();
      for (std::size_t objectId = 0; objectId < dataset.numObjects;
           ++objectId) {
        if (initialized[robotId][objectId]) {
          continue;
        }
        Matrix rotationSum = Matrix::Zero(d, d);
        Matrix translationSum = Matrix::Zero(d, 1);
        double weightSum = 0.0;
        for (const WeightedNeighbor &neighbor :
             initializationNeighbors[robotId]) {
          if (!initialized[neighbor.id][objectId]) {
            continue;
          }
          const ObjectPGORobotData &neighborRobot =
              dataset.robots[neighbor.id];
          const std::size_t neighborObjectIndex =
              neighborRobot.trajectoryVertexIds.size() + objectId;
          const Matrix neighborPose =
              blockPose(result.states[neighbor.id], neighborObjectIndex, d);
          rotationSum.noalias() +=
              neighbor.weight * neighborPose.leftCols(d);
          translationSum.noalias() +=
              neighbor.weight * neighborPose.rightCols(1);
          weightSum += neighbor.weight;
        }
        if (weightSum <= 0.0) {
          continue;
        }
        Matrix averaged = Matrix::Zero(d, d + 1);
        averaged.leftCols(d) = projectToRotationGroup(rotationSum);
        averaged.rightCols(1) = translationSum / weightSum;
        setPoseBlock(nextStates[robotId], objectOffset + objectId, averaged,
                     d);
        nextInitialized[robotId][objectId] = true;
        initializedFromNeighbor[robotId][objectId] = true;
        changed = true;
      }
    }
    result.states = projectStatesToSE(nextStates, d);
    initialized = std::move(nextInitialized);
    if (!changed) {
      break;
    }
  }

  const std::size_t consensusRounds =
      options.objectInitializationConsensusRounds;
  const double observedAnchorWeight =
      std::isfinite(options.objectInitializationObservedAnchorWeight)
          ? std::max(0.0, options.objectInitializationObservedAnchorWeight)
          : 0.0;
  const double relayAnchorWeight =
      std::isfinite(options.objectInitializationRelayAnchorWeight)
          ? std::max(0.0, options.objectInitializationRelayAnchorWeight)
          : 0.0;
  if (consensusRounds > 0 &&
      (observedAnchorWeight > 0.0 || relayAnchorWeight > 0.0 ||
       !initializationNeighbors.empty())) {
    const std::vector<Matrix> anchorStates = result.states;
    const bool useInformationAnchors =
        options.objectInitializationAnchorMode ==
        ManualObjectDranObjectInitializationAnchorMode::Information;
    const std::vector<std::vector<double>> anchorScales =
        useInformationAnchors
            ? objectObservationInformationScales(dataset, d)
            : std::vector<std::vector<double>>();
    for (std::size_t round = 0; round < consensusRounds; ++round) {
      std::vector<Matrix> nextStates = result.states;
      for (std::size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
        const ObjectPGORobotData &robot = dataset.robots[robotId];
        const std::size_t objectOffset = robot.trajectoryVertexIds.size();
        for (std::size_t objectId = 0; objectId < dataset.numObjects;
             ++objectId) {
          if (!initialized[robotId][objectId]) {
            continue;
          }
          Matrix rotationSum = Matrix::Zero(d, d);
          Matrix translationSum = Matrix::Zero(d, 1);
          double weightSum = 0.0;
          const std::size_t localObjectIndex = objectOffset + objectId;
          double anchorWeight =
              observedByRobot[robotId][objectId] ? observedAnchorWeight
                                                 : relayAnchorWeight;
          if (useInformationAnchors && observedByRobot[robotId][objectId]) {
            anchorWeight *= anchorScales[robotId][objectId];
          }
          if (anchorWeight > 0.0) {
            const Matrix anchor =
                blockPose(anchorStates[robotId], localObjectIndex, d);
            rotationSum.noalias() += anchorWeight * anchor.leftCols(d);
            translationSum.noalias() += anchorWeight * anchor.rightCols(1);
            weightSum += anchorWeight;
          }
          for (const WeightedNeighbor &neighbor :
               initializationNeighbors[robotId]) {
            if (!initialized[neighbor.id][objectId]) {
              continue;
            }
            const ObjectPGORobotData &neighborRobot =
                dataset.robots[neighbor.id];
            const std::size_t neighborObjectIndex =
                neighborRobot.trajectoryVertexIds.size() + objectId;
            const Matrix neighborPose =
                blockPose(result.states[neighbor.id], neighborObjectIndex, d);
            rotationSum.noalias() +=
                neighbor.weight * neighborPose.leftCols(d);
            translationSum.noalias() +=
                neighbor.weight * neighborPose.rightCols(1);
            weightSum += neighbor.weight;
          }
          if (weightSum <= 0.0) {
            continue;
          }
          Matrix averaged = Matrix::Zero(d, d + 1);
          averaged.leftCols(d) = projectToRotationGroup(rotationSum);
          averaged.rightCols(1) = translationSum / weightSum;
          setPoseBlock(nextStates[robotId], localObjectIndex, averaged, d);
        }
      }
      result.states = projectStatesToSE(nextStates, d);
    }
  }

  for (std::size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
    for (std::size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (initializedFromNeighbor[robotId][objectId]) {
        ++result.initializedFromNeighborObjectCopies;
      }
      if (!initialized[robotId][objectId]) {
        ++result.staleObjectCopies;
      }
    }
  }
  return result;
}

double measurementCost(const RelativeSEMeasurement &m, const Matrix &state,
                       unsigned d) {
  const Matrix pose1 = blockPose(state, m.p1, d);
  const Matrix pose2 = blockPose(state, m.p2, d);
  const Matrix R1 = pose1.leftCols(d);
  const Matrix R2 = pose2.leftCols(d);
  const Matrix t1 = pose1.rightCols(1);
  const Matrix t2 = pose2.rightCols(1);
  return m.kappa * (R1 * m.R - R2).squaredNorm() +
         m.tau * (t2 - t1 - R1 * m.t).squaredNorm();
}

double objectMeasurementResidualNorm(const ObjectPGORobotData &robot,
                                     const Matrix &state,
                                     std::size_t objectId, unsigned d) {
  double residualSquared = 0.0;
  for (const ObjectObservationMeasurement &obs :
       robot.objectObservationMeasurements) {
    if (obs.objectGlobalVertexId != objectId) {
      continue;
    }
    residualSquared += measurementCost(obs.measurement, state, d);
  }
  return std::sqrt(residualSquared);
}

double totalMeasurementCost(
    const std::vector<std::vector<RelativeSEMeasurement>> &measurements,
    const std::vector<Matrix> &states, unsigned d) {
  double cost = 0.0;
  for (std::size_t robotId = 0; robotId < states.size(); ++robotId) {
    for (const RelativeSEMeasurement &m : measurements[robotId]) {
      cost += measurementCost(m, states[robotId], d);
    }
  }
  return cost;
}

void addConsensusTerms(const ObjectPGODirectoryData &dataset,
                       const NeighborRound &neighbors,
                       const Matrix &localState,
                       const std::vector<Matrix> &targetStates,
                       const std::vector<SparseMatrix> &baseQs,
                       unsigned robotId, double beta, unsigned d,
                       SparseMatrix &Q, SparseMatrix &G,
                       std::size_t &activePairs, double &consensusCost) {
  const ObjectPGORobotData &robot = dataset.robots[robotId];
  const std::size_t objectOffset = robot.trajectoryVertexIds.size();
  const unsigned blockWidth = d + 1;
  Q = baseQs[robotId];
  Matrix denseG = Matrix::Zero(d, Q.cols());
  for (const WeightedNeighbor &neighbor : neighbors[robotId]) {
    const ObjectPGORobotData &neighborRobot = dataset.robots[neighbor.id];
    const std::size_t neighborObjectOffset =
        neighborRobot.trajectoryVertexIds.size();
    const double stiffness = beta * neighbor.weight * neighbor.weight;
    for (std::size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      const std::size_t localObjectIndex = objectOffset + objectId;
      const std::size_t neighborObjectIndex = neighborObjectOffset + objectId;
      const std::size_t colOffset = localObjectIndex * blockWidth;
      const Matrix target =
          blockPose(targetStates[neighbor.id], neighborObjectIndex, d);
      for (unsigned c = 0; c < blockWidth; ++c) {
        Q.coeffRef(colOffset + c, colOffset + c) += stiffness;
      }
      denseG.block(0, colOffset, d, blockWidth) += -stiffness * target;
      ++activePairs;
      consensusCost +=
          0.5 * stiffness *
          (blockPose(localState, localObjectIndex, d) - target)
              .squaredNorm();
    }
  }
  Q.makeCompressed();
  G = denseG.sparseView();
}

double objectBlockDiagonalStiffness(const SparseMatrix &Q,
                                    std::size_t objectIndex,
                                    unsigned d) {
  const unsigned blockWidth = d + 1;
  double stiffness = 0.0;
  const std::size_t colOffset = objectIndex * blockWidth;
  for (unsigned c = 0; c < blockWidth; ++c) {
    stiffness +=
        std::abs(Q.coeff(static_cast<int>(colOffset + c),
                         static_cast<int>(colOffset + c)));
  }
  return stiffness;
}

double objectNeighborDisagreement(const ObjectPGODirectoryData &dataset,
                                  const NeighborRound &neighbors,
                                  const Matrix &localState,
                                  const std::vector<Matrix> &targetStates,
                                  unsigned robotId, std::size_t objectId,
                                  unsigned d) {
  const ObjectPGORobotData &robot = dataset.robots[robotId];
  const std::size_t localObjectIndex =
      robot.trajectoryVertexIds.size() + objectId;
  const Matrix localObject = blockPose(localState, localObjectIndex, d);
  double weightedSquaredDisagreement = 0.0;
  double weightSum = 0.0;
  for (const WeightedNeighbor &neighbor : neighbors[robotId]) {
    const ObjectPGORobotData &neighborRobot = dataset.robots[neighbor.id];
    const std::size_t neighborObjectIndex =
        neighborRobot.trajectoryVertexIds.size() + objectId;
    const Matrix target =
        blockPose(targetStates[neighbor.id], neighborObjectIndex, d);
    weightedSquaredDisagreement +=
        neighbor.weight * (localObject - target).squaredNorm();
    weightSum += neighbor.weight;
  }
  if (weightSum <= 0.0) {
    return 0.0;
  }
  return std::sqrt(weightedSquaredDisagreement / weightSum);
}

std::size_t objectInterfaceAge(const NeighborRound &neighbors,
                               const NeighborObjectAges &ages,
                               unsigned robotId, std::size_t objectId) {
  std::size_t maxAge = 0;
  for (const WeightedNeighbor &neighbor : neighbors[robotId]) {
    maxAge = std::max(maxAge, ages[robotId][neighbor.id][objectId]);
  }
  return maxAge;
}

std::vector<std::vector<bool>> initialSelectedForSend(
    const ObjectPGODirectoryData &dataset, const NeighborRound &neighbors) {
  std::vector<std::vector<bool>> selected(
      dataset.numRobots, std::vector<bool>(dataset.numObjects, false));
  for (std::size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
    if (neighbors[robotId].empty()) {
      continue;
    }
    std::fill(selected[robotId].begin(), selected[robotId].end(), true);
  }
  return selected;
}

void addTranslationProxTerms(const Matrix &referenceState, double weight,
                             unsigned d, SparseMatrix &Q, SparseMatrix &G) {
  if (weight <= 0.0 || referenceState.cols() == 0) {
    return;
  }
  const unsigned blockWidth = d + 1;
  const std::size_t numPoses =
      static_cast<std::size_t>(referenceState.cols()) / blockWidth;
  Matrix denseG = Matrix(G);
  if (denseG.rows() == 0 || denseG.cols() == 0) {
    denseG = Matrix::Zero(referenceState.rows(), referenceState.cols());
  }
  for (std::size_t poseId = 0; poseId < numPoses; ++poseId) {
    const std::size_t col = poseId * blockWidth + d;
    Q.coeffRef(static_cast<int>(col), static_cast<int>(col)) += weight;
    denseG.col(static_cast<int>(col)).noalias() +=
        -weight * referenceState.col(static_cast<int>(col));
  }
  Q.makeCompressed();
  G = denseG.sparseView();
}

void configureReducedOptimizer(
    ReducedRotationQuadraticOptimizer &optimizer,
    const ManualObjectDranOptions &options,
    ManualDpgoMmReducedRotationPreconditioner mode, int innerIterations) {
  optimizer.setTrustRegionIterations(options.localTrustRegionIterations);
  optimizer.setTrustRegionMaxInnerIterations(innerIterations);
  optimizer.setTrustRegionTolerance(options.localTrustRegionTolerance);
  optimizer.setTrustRegionInitialRadius(options.localTrustRegionInitialRadius);
  optimizer.setUseJacobiPreconditioner(
      mode == ManualDpgoMmReducedRotationPreconditioner::Jacobi);
  optimizer.setUseSchurJacobiPreconditioner(
      mode == ManualDpgoMmReducedRotationPreconditioner::SchurJacobi);
  optimizer.setUseCholeskyPreconditioner(
      mode == ManualDpgoMmReducedRotationPreconditioner::Cholesky);
  optimizer.setRecordResultStats(true);
  optimizer.setValidateTranslationRecoveryCost(false);
  optimizer.setVerbose(false);
  optimizer.setUseCoupledTrustRegionNorm(
      options.coupledTranslationTrustRegion);
  optimizer.setTranslationEliminationProxWeight(
      options.translationEliminationProxWeight);
}

Matrix optimizeReduced(QuadraticProblem &problem, const Matrix &start,
                       const ManualObjectDranOptions &options,
                       ROPTResult &result) {
  auto run = [&](ManualDpgoMmReducedRotationPreconditioner mode) {
    ReducedRotationQuadraticOptimizer optimizer(&problem);
    configureReducedOptimizer(optimizer, options, mode,
                              options.localTrustRegionMaxInnerIterations);
    Matrix candidate = optimizer.optimize(start);
    return std::make_pair(candidate, optimizer.getOptResult());
  };

  if (options.reducedRotationPreconditioner ==
      ManualDpgoMmReducedRotationPreconditioner::Portfolio) {
    auto plain = run(ManualDpgoMmReducedRotationPreconditioner::None);
    auto jacobi = run(ManualDpgoMmReducedRotationPreconditioner::Jacobi);
    const double plainCost = problem.f(plain.first);
    const double jacobiCost = problem.f(jacobi.first);
    if (std::isfinite(jacobiCost) &&
        (!std::isfinite(plainCost) || jacobiCost < plainCost)) {
      result = jacobi.second;
      return jacobi.first;
    }
    result = plain.second;
    return plain.first;
  }

  auto candidate = run(options.reducedRotationPreconditioner);
  result = candidate.second;
  return candidate.first;
}

Matrix optimizeFullRtr(QuadraticProblem &problem, const Matrix &start,
                       const ManualObjectDranOptions &options,
                       ROPTResult &result) {
  QuadraticOptimizer optimizer(&problem);
  optimizer.setAlgorithm(ROPTALG::RTR);
  optimizer.setTrustRegionIterations(options.localTrustRegionIterations);
  optimizer.setTrustRegionMaxInnerIterations(
      options.localTrustRegionMaxInnerIterations);
  optimizer.setTrustRegionTolerance(options.localTrustRegionTolerance);
  optimizer.setTrustRegionInitialRadius(options.localTrustRegionInitialRadius);
  optimizer.setVerbose(false);
  Matrix candidate = optimizer.optimize(start);
  result = optimizer.getOptResult();
  return candidate;
}

Matrix optimizeLocal(QuadraticProblem &problem, const Matrix &start,
                     const ManualObjectDranOptions &options,
                     unsigned outerIteration, ROPTResult &result) {
  const bool useFull =
      (options.localSolver == ManualDpgoMmLocalSolver::ManualFull ||
       options.localSolver == ManualDpgoMmLocalSolver::FullEquivHybrid) &&
      (options.fullSolverOuterIterations == 0 ||
       outerIteration <= options.fullSolverOuterIterations);
  if (useFull) {
    return optimizeFullRtr(problem, start, options, result);
  }
  return optimizeReduced(problem, start, options, result);
}

NeighborStateCache initializeNeighborStateCache(
    const NeighborRound &neighbors, const std::vector<Matrix> &states) {
  NeighborStateCache cache(states.size(),
                           std::vector<Matrix>(states.size(), Matrix()));
  for (std::size_t receiverId = 0; receiverId < states.size(); ++receiverId) {
    for (const WeightedNeighbor &neighbor : neighbors[receiverId]) {
      cache[receiverId][neighbor.id] = states[neighbor.id];
    }
  }
  return cache;
}

NeighborObjectAges initializeNeighborObjectAges(std::size_t numRobots,
                                                std::size_t numObjects) {
  return NeighborObjectAges(
      numRobots, std::vector<std::vector<std::size_t>>(
                     numRobots, std::vector<std::size_t>(numObjects, 0)));
}

std::size_t countDirectedObjectPairs(const ObjectPGODirectoryData &dataset,
                                     const NeighborRound &neighbors) {
  std::size_t count = 0;
  for (std::size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
    count += neighbors[robotId].size() * dataset.numObjects;
  }
  return count;
}

ObjectExchangeStats exchangeObjectUpdates(
    const ObjectPGODirectoryData &dataset, const NeighborRound &neighbors,
    const ManualObjectDranOptions &options, const std::vector<Matrix> &states,
    NeighborStateCache &cache, NeighborObjectAges &ages, unsigned d,
    bool forceAll) {
  const unsigned blockWidth = d + 1;
  ObjectExchangeStats stats;
  stats.selectedForSend.assign(
      dataset.numRobots, std::vector<bool>(dataset.numObjects, false));
  stats.selectedForReceiver.assign(
      dataset.numRobots,
      std::vector<std::vector<bool>>(
          dataset.numRobots, std::vector<bool>(dataset.numObjects, false)));
  for (std::size_t senderId = 0; senderId < dataset.numRobots; ++senderId) {
    const ObjectPGORobotData &sender = dataset.robots[senderId];
    const std::size_t senderObjectOffset = sender.trajectoryVertexIds.size();
    for (const WeightedNeighbor &neighbor : neighbors[senderId]) {
      const std::size_t receiverId = neighbor.id;
      Matrix &receiverCache = cache[receiverId][senderId];
      for (std::size_t objectId = 0; objectId < dataset.numObjects;
           ++objectId) {
        const std::size_t senderObjectIndex = senderObjectOffset + objectId;
        const Matrix current = blockPose(states[senderId], senderObjectIndex, d);
        const Matrix lastSent =
            blockPose(receiverCache, senderObjectIndex, d);
        const bool stale = ages[senderId][receiverId][objectId] >=
                           options.objectMaxAge;
        const bool changed =
            (current - lastSent).norm() > options.objectPoseTolerance;
        const bool send =
            forceAll ||
            options.communicationPolicy ==
                ManualObjectDranCommunicationPolicy::All ||
            changed || stale;
        if (send) {
          if (stale) {
            ++stats.staleObjectCommTriggers;
          }
          receiverCache.block(0, senderObjectIndex * blockWidth, d,
                              blockWidth) = current;
          ages[senderId][receiverId][objectId] = 0;
          stats.selectedForSend[senderId][objectId] = true;
          stats.selectedForReceiver[receiverId][senderId][objectId] = true;
          ++stats.communicated;
        } else {
          ++ages[senderId][receiverId][objectId];
        }
      }
    }
  }
  return stats;
}

struct ObjectInterfaceCandidate {
  std::size_t objectId{0};
  double score{0.0};
};

bool receivedObjectUpdate(const ObjectReceiveSelection &received,
                          std::size_t receiverId, std::size_t objectId) {
  if (receiverId >= received.size()) {
    return false;
  }
  for (const std::vector<bool> &fromSender : received[receiverId]) {
    if (objectId < fromSender.size() && fromSender[objectId]) {
      return true;
    }
  }
  return false;
}

Matrix objectInterfaceSchurStep(const SparseMatrix &Q, const Matrix &gradient,
                                unsigned colStart, unsigned blockWidth,
                                const ManualObjectDranOptions &options) {
  Matrix step = solveBoundaryLocalSchurPreconditionedStep(
      Q, gradient, colStart, blockWidth, options.objectInterfaceSchurDamping,
      options.objectInterfaceMaxPrivateCols);
  step *= std::max(0.0, options.objectInterfaceStepGain);
  double stepNorm = step.norm();
  if (!step.allFinite() || stepNorm <= 0.0) {
    return Matrix::Zero(gradient.rows(), blockWidth);
  }
  if (options.objectInterfaceMaxBlockStepNorm > 0.0 &&
      stepNorm > options.objectInterfaceMaxBlockStepNorm) {
    step *= options.objectInterfaceMaxBlockStepNorm / stepNorm;
  }
  return step;
}

double objectInterfacePredictedDecrease(
    const ObjectPGORobotData &robot,
    const std::vector<ObjectInterfaceCandidate> &candidates,
    const SparseMatrix &Q, const Matrix &gradient,
    const ManualObjectDranOptions &options, unsigned blockWidth) {
  double predictedDecrease = 0.0;
  for (const ObjectInterfaceCandidate &candidate : candidates) {
    const std::size_t objectIndex =
        robot.trajectoryVertexIds.size() + candidate.objectId;
    const unsigned colStart =
        static_cast<unsigned>(objectIndex * blockWidth);
    const Matrix step =
        objectInterfaceSchurStep(Q, gradient, colStart, blockWidth, options);
    if (step.norm() <= 0.0) {
      continue;
    }
    const Matrix gradientBlock =
        gradient.block(0, colStart, gradient.rows(), blockWidth);
    const double linearDecrease = -(gradientBlock.array() * step.array()).sum();
    if (std::isfinite(linearDecrease) && linearDecrease > 0.0) {
      predictedDecrease += linearDecrease;
    }
  }
  return std::isfinite(predictedDecrease) ? predictedDecrease : 0.0;
}

void applySerialObjectInterfaceSchurResponseForRobot(
    const ObjectPGORobotData &robot,
    const std::vector<ObjectInterfaceCandidate> &candidates,
    const SparseMatrix &Q, QuadraticProblem &problem,
    const ManualObjectDranOptions &options, unsigned d,
    unsigned blockWidth, Matrix &current,
    ManualObjectDranObjectInterfaceResponseSummary &summary) {
  double currentCost = problem.f(current);
  Matrix gradient = problem.RieGrad(current);
  for (const ObjectInterfaceCandidate &candidate : candidates) {
    ++summary.candidateObjectBlocks;
    const std::size_t objectIndex =
        robot.trajectoryVertexIds.size() + candidate.objectId;
    const unsigned colStart =
        static_cast<unsigned>(objectIndex * blockWidth);
    Matrix step =
        objectInterfaceSchurStep(Q, gradient, colStart, blockWidth, options);
    const double stepNorm = step.norm();
    if (stepNorm <= 0.0) {
      ++summary.rejectedObjectBlocks;
      continue;
    }

    Matrix proposed = current;
    proposed.block(0, colStart, d, blockWidth).noalias() += step;
    proposed = projectStateToSE(proposed, d);
    const double after = problem.f(proposed);
    const bool accept =
        std::isfinite(after) &&
        (!options.objectInterfaceRequireDecrease ||
         after <= currentCost + 1e-10);
    if (accept) {
      current = std::move(proposed);
      ++summary.acceptedObjectBlocks;
      summary.acceptedModelDecrease += std::max(0.0, currentCost - after);
      summary.stepNorm += stepNorm;
      currentCost = after;
      gradient = problem.RieGrad(current);
    } else {
      ++summary.rejectedObjectBlocks;
    }
  }
}

void applyBatchObjectInterfaceSchurResponseForRobot(
    const ObjectPGORobotData &robot,
    const std::vector<ObjectInterfaceCandidate> &candidates,
    const SparseMatrix &Q, QuadraticProblem &problem,
    const Matrix &initialGradient, const ManualObjectDranOptions &options,
    unsigned d, unsigned blockWidth, Matrix &current,
    ManualObjectDranObjectInterfaceResponseSummary &summary) {
  ++summary.batchResponseRobots;
  const double before = problem.f(current);
  Matrix proposed = current;
  std::size_t validSteps = 0;
  std::size_t invalidSteps = 0;
  double batchStepNorm = 0.0;
  for (const ObjectInterfaceCandidate &candidate : candidates) {
    const std::size_t objectIndex =
        robot.trajectoryVertexIds.size() + candidate.objectId;
    const unsigned colStart =
        static_cast<unsigned>(objectIndex * blockWidth);
    Matrix step = objectInterfaceSchurStep(
        Q, initialGradient, colStart, blockWidth, options);
    const double stepNorm = step.norm();
    if (stepNorm <= 0.0) {
      ++invalidSteps;
      continue;
    }
    proposed.block(0, colStart, d, blockWidth).noalias() += step;
    ++validSteps;
    batchStepNorm += stepNorm;
  }
  if (validSteps == 0) {
    ++summary.rejectedRobotBatches;
    applySerialObjectInterfaceSchurResponseForRobot(
        robot, candidates, Q, problem, options, d, blockWidth, current,
        summary);
    return;
  }

  proposed = projectStateToSE(proposed, d);
  const double after = problem.f(proposed);
  const bool accept =
      std::isfinite(after) &&
      (!options.objectInterfaceRequireDecrease || after <= before + 1e-10);
  if (accept) {
    current = std::move(proposed);
    summary.candidateObjectBlocks += candidates.size();
    summary.acceptedObjectBlocks += validSteps;
    summary.rejectedObjectBlocks += invalidSteps;
    ++summary.acceptedRobotBatches;
    summary.acceptedModelDecrease += std::max(0.0, before - after);
    summary.stepNorm += batchStepNorm;
  } else {
    ++summary.rejectedRobotBatches;
    applySerialObjectInterfaceSchurResponseForRobot(
        robot, candidates, Q, problem, options, d, blockWidth, current,
        summary);
  }
}

ManualObjectDranObjectInterfaceResponseSummary
applyObjectInterfaceSchurResponse(
    const ObjectPGODirectoryData &dataset, const NeighborRound &neighbors,
    const ManualObjectDranOptions &options,
    const std::vector<SparseMatrix> &baseQs,
    const NeighborStateCache &neighborStateCache,
    const NeighborObjectAges &neighborObjectAges,
    const ObjectReceiveSelection &receivedObjects, std::vector<Matrix> &states,
    std::vector<double> &innovationAccumulators, unsigned iter, unsigned d,
    unsigned r) {
  ManualObjectDranObjectInterfaceResponseSummary summary;
  summary.iter = iter;
  if (options.objectInterfaceMode !=
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse) {
    return summary;
  }
  const std::size_t responsePeriod =
      std::max<std::size_t>(1, options.objectInterfaceResponsePeriod);
  if (iter % responsePeriod != 0) {
    return summary;
  }
  summary.scheduled = true;

  const unsigned blockWidth = d + 1;
  const double gain =
      std::isfinite(options.objectInterfaceStepGain)
          ? std::max(0.0, options.objectInterfaceStepGain)
          : 0.0;
  if (gain <= 0.0) {
    return summary;
  }

  for (std::size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const std::size_t objectOffset = robot.trajectoryVertexIds.size();
    SparseMatrix Q;
    SparseMatrix G;
    std::size_t activePairs = 0;
    double consensusCost = 0.0;
    addConsensusTerms(dataset, neighbors, states[robotId],
                      neighborStateCache[robotId], baseQs, robotId,
                      options.beta, d, Q, G, activePairs, consensusCost);
    addTranslationProxTerms(states[robotId], options.translationProxWeight, d,
                            Q, G);
    QuadraticProblem problem(robot.trajectoryVertexIds.size() +
                                 dataset.numObjects,
                             d, r);
    problem.setQ(Q);
    problem.setG(G);
    Matrix current = states[robotId];
    const Matrix initialGradient = problem.RieGrad(current);
    std::vector<ObjectInterfaceCandidate> candidates;
    candidates.reserve(dataset.numObjects);
    for (std::size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!receivedObjectUpdate(receivedObjects, robotId, objectId)) {
        continue;
      }
      const std::size_t objectIndex = objectOffset + objectId;
      const double gradientScore =
          blockPose(initialGradient, objectIndex, d).norm();
      const double residualScore =
          objectMeasurementResidualNorm(robot, current, objectId, d);
      const double disagreementScore =
          objectNeighborDisagreement(dataset, neighbors, current,
                                     neighborStateCache[robotId], robotId,
                                     objectId, d);
      const double ageScore =
          static_cast<double>(objectInterfaceAge(
              neighbors, neighborObjectAges, robotId, objectId));
      const double score =
          gradientScore + residualScore + disagreementScore + 1e-3 * ageScore;
      if (std::isfinite(score) && score > 0.0) {
        candidates.push_back(ObjectInterfaceCandidate{objectId, score});
      }
    }
    if (candidates.empty()) {
      continue;
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const ObjectInterfaceCandidate &a,
                 const ObjectInterfaceCandidate &b) {
                if (a.score == b.score) {
                  return a.objectId < b.objectId;
                }
                return a.score > b.score;
              });
    if (options.objectInterfaceMaxObjectsPerRobot > 0 &&
        candidates.size() > options.objectInterfaceMaxObjectsPerRobot) {
      candidates.resize(options.objectInterfaceMaxObjectsPerRobot);
    }

    ++summary.triggerCandidateRobots;
    double innovationScore = 0.0;
    for (const ObjectInterfaceCandidate &candidate : candidates) {
      if (std::isfinite(candidate.score) && candidate.score > 0.0) {
        innovationScore += candidate.score;
      }
    }
    summary.interfaceInnovationScore += innovationScore;
    if (robotId < innovationAccumulators.size() &&
        options.objectInterfaceResponseTrigger ==
            ManualObjectDranObjectInterfaceResponseTrigger::
                AccumulatedInnovation) {
      innovationAccumulators[robotId] += innovationScore;
      summary.accumulatedInnovationScore += innovationAccumulators[robotId];
      const double minInnovationScore =
          std::isfinite(options.objectInterfaceMinInnovationScore)
              ? std::max(0.0, options.objectInterfaceMinInnovationScore)
              : std::numeric_limits<double>::infinity();
      if (innovationAccumulators[robotId] < minInnovationScore) {
        ++summary.skippedTriggerRobots;
        continue;
      }
    }
    if (options.objectInterfaceResponseTrigger ==
        ManualObjectDranObjectInterfaceResponseTrigger::LocalInnovation) {
      const double minInnovationScore =
          std::isfinite(options.objectInterfaceMinInnovationScore)
              ? std::max(0.0, options.objectInterfaceMinInnovationScore)
              : std::numeric_limits<double>::infinity();
      if (innovationScore < minInnovationScore) {
        ++summary.skippedTriggerRobots;
        continue;
      }
    }
    if (options.objectInterfaceResponseTrigger ==
        ManualObjectDranObjectInterfaceResponseTrigger::PredictedDecrease) {
      const double predictedDecrease = objectInterfacePredictedDecrease(
          robot, candidates, Q, initialGradient, options, blockWidth);
      summary.predictedModelDecrease += predictedDecrease;
      const double minPredictedDecrease =
          std::isfinite(options.objectInterfaceMinPredictedDecrease)
              ? std::max(0.0, options.objectInterfaceMinPredictedDecrease)
              : std::numeric_limits<double>::infinity();
      if (predictedDecrease < minPredictedDecrease) {
        ++summary.skippedTriggerRobots;
        continue;
      }
    }

    ++summary.triggeredRobots;
    ++summary.activeRobots;
    summary.modelCostBefore += problem.f(current);
    const std::size_t acceptedBefore = summary.acceptedObjectBlocks;
    if (options.objectInterfaceResponseStrategy ==
        ManualObjectDranObjectInterfaceResponseStrategy::Batch) {
      applyBatchObjectInterfaceSchurResponseForRobot(
          robot, candidates, Q, problem, initialGradient, options, d,
          blockWidth, current, summary);
    } else {
      applySerialObjectInterfaceSchurResponseForRobot(
          robot, candidates, Q, problem, options, d, blockWidth, current,
          summary);
    }
    if (robotId < innovationAccumulators.size() &&
        options.objectInterfaceResponseTrigger ==
            ManualObjectDranObjectInterfaceResponseTrigger::
                AccumulatedInnovation &&
        summary.acceptedObjectBlocks > acceptedBefore) {
      innovationAccumulators[robotId] = 0.0;
    }
    summary.modelCostAfter += problem.f(current);
    states[robotId] = current;
  }
  return summary;
}

}  // namespace

ManualObjectDranRunResult
runManualObjectDran(const std::string &directory,
                    const ManualObjectDranOptions &options) {
  ObjectPGODirectoryData dataset =
      loadObjectAwareG2ODirectory(directory, options.numRobots,
                                  options.numObjects);
  if (dataset.numRobots == 0 || dataset.numObjects == 0 ||
      dataset.dimension == 0) {
    throw std::runtime_error("ManualObjectDran requires object-aware data");
  }
  const unsigned numRobots = static_cast<unsigned>(dataset.numRobots);
  const unsigned d = static_cast<unsigned>(dataset.dimension);
  const unsigned r = d;
  const unsigned blockWidth = d + 1;
  const std::size_t blockBytes =
      static_cast<std::size_t>(d) * blockWidth * sizeof(double);
  const std::vector<NeighborRound> topologySchedule =
      makeTopologySchedule(numRobots, options);
  const NeighborRound cacheNeighbors =
      unionNeighbors(topologySchedule, numRobots);

  std::vector<std::vector<RelativeSEMeasurement>> localMeasurements;
  std::vector<SparseMatrix> baseQs;
  ObjectInitializationResult initialization =
      initializeObjectStates(dataset, d, cacheNeighbors, options);
  std::vector<Matrix> states = initialization.states;
  NeighborStateCache neighborStateCache =
      initializeNeighborStateCache(cacheNeighbors, states);
  NeighborObjectAges neighborObjectAges =
      initializeNeighborObjectAges(dataset.numRobots, dataset.numObjects);
  std::vector<double> objectInterfaceInnovationAccumulators(dataset.numRobots,
                                                            0.0);
  localMeasurements.reserve(dataset.numRobots);
  baseQs.reserve(dataset.numRobots);
  for (const ObjectPGORobotData &robot : dataset.robots) {
    std::vector<RelativeSEMeasurement> measurements =
        objectAwareLocalMeasurements(robot);
    const std::size_t localNumPoses =
        robot.trajectoryVertexIds.size() + dataset.numObjects;
    baseQs.push_back(fullConnectionLaplacian(measurements, localNumPoses, d));
    localMeasurements.push_back(std::move(measurements));
  }

  ManualObjectDranRunResult runResult;
  runResult.numRobots = dataset.numRobots;
  runResult.numObjects = dataset.numObjects;
  runResult.knownObjectCopies = dataset.numRobots * dataset.numObjects;
  runResult.observedObjectCopies = initialization.observedObjectCopies;
  runResult.relayOnlyObjectCopies = initialization.relayOnlyObjectCopies;
  runResult.relayObjectCopies = runResult.relayOnlyObjectCopies;
  runResult.initializedFromNeighborObjectCopies =
      initialization.initializedFromNeighborObjectCopies;
  runResult.staleObjectCopies = initialization.staleObjectCopies;

  std::size_t cumulativeCommPoseCount = 0;
  double cumulativeCommMb = 0.0;
  const auto runStart = std::chrono::high_resolution_clock::now();

  auto elapsedSeconds = [&]() {
    return std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - runStart)
        .count();
  };

  auto summarize = [&](unsigned iter, const NeighborRound &summaryNeighbors,
                       const std::vector<Matrix> &summaryStates,
                       const NeighborStateCache &summaryCache,
                       const NeighborObjectAges &summaryAges,
                       const std::vector<std::vector<bool>> &selectedForSend,
                       std::size_t iterCommPoseCount,
                       std::size_t staleObjectCommTriggers) {
    ManualObjectDranIterationSummary summary;
    summary.iter = iter;
    summary.time = elapsedSeconds();
    summary.measurementCost =
        totalMeasurementCost(localMeasurements, summaryStates, d);
    summary.gradient = 0.0;
    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      SparseMatrix Q;
      SparseMatrix G;
      std::size_t activePairs = 0;
      double consensusCost = 0.0;
      addConsensusTerms(dataset, summaryNeighbors, summaryStates[robotId],
                        summaryCache[robotId], baseQs, robotId, options.beta,
                        d, Q, G, activePairs, consensusCost);
      summary.activeConsensusPairs += activePairs;
    summary.consensusCost += consensusCost;
      QuadraticProblem problem(
          dataset.robots[robotId].trajectoryVertexIds.size() +
              dataset.numObjects,
          d, r);
      problem.setQ(Q);
      problem.setG(G);
      const Matrix localRieGrad = problem.RieGrad(summaryStates[robotId]);
      summary.gradient += localRieGrad.norm();
      const ObjectPGORobotData &robot = dataset.robots[robotId];
      const std::vector<bool> observed =
          observedObjects(robot, dataset.numObjects);
      const std::size_t objectOffset = robot.trajectoryVertexIds.size();
      for (std::size_t objectId = 0; objectId < dataset.numObjects;
           ++objectId) {
        const std::size_t objectIndex = objectOffset + objectId;
        ManualObjectDranObjectInterfaceDiagnostic diagnostic;
        diagnostic.iter = iter;
        diagnostic.robotId = robotId;
        diagnostic.objectId = objectId;
        diagnostic.observed = observed[objectId];
        diagnostic.relayOnly = !diagnostic.observed;
        diagnostic.age =
            objectInterfaceAge(summaryNeighbors, summaryAges, robotId,
                               objectId);
        diagnostic.residualNorm = objectMeasurementResidualNorm(
            robot, summaryStates[robotId], objectId, d);
        diagnostic.gradientNorm =
            blockPose(localRieGrad, objectIndex, d).norm();
        diagnostic.stiffness =
            objectBlockDiagonalStiffness(Q, objectIndex, d);
        diagnostic.neighborDisagreement = objectNeighborDisagreement(
            dataset, summaryNeighbors, summaryStates[robotId],
            summaryCache[robotId], robotId, objectId, d);
        diagnostic.selectedForSend =
            robotId < selectedForSend.size() &&
            objectId < selectedForSend[robotId].size() &&
            selectedForSend[robotId][objectId];
        runResult.objectInterfaceDiagnostics.push_back(diagnostic);
      }
    }
    summary.knownObjectCopies = runResult.knownObjectCopies;
    summary.relayObjectCopies = runResult.relayObjectCopies;
    summary.observedObjectCopies = runResult.observedObjectCopies;
    summary.relayOnlyObjectCopies = runResult.relayOnlyObjectCopies;
    summary.initializedFromNeighborObjectCopies =
        runResult.initializedFromNeighborObjectCopies;
    summary.staleObjectCopies = runResult.staleObjectCopies;
    summary.staleObjectCommTriggers = staleObjectCommTriggers;
    summary.commPoseCount = iterCommPoseCount;
    summary.iterCommMb =
        static_cast<double>(summary.commPoseCount * blockBytes) /
        (1024.0 * 1024.0);
    cumulativeCommPoseCount += summary.commPoseCount;
    cumulativeCommMb += summary.iterCommMb;
    summary.cumulativeCommPoseCount = cumulativeCommPoseCount;
    summary.cumulativeCommMb = cumulativeCommMb;
    runResult.iterations.push_back(summary);
  };

  summarize(0, topologySchedule.front(), states, neighborStateCache,
            neighborObjectAges,
            initialSelectedForSend(dataset, topologySchedule.front()),
            countDirectedObjectPairs(dataset, topologySchedule.front()), 0);
  for (unsigned iter = 1; iter <= options.maxIterations; ++iter) {
    const NeighborRound &neighbors =
        topologySchedule[(iter - 1) % topologySchedule.size()];
    const std::vector<Matrix> previousStates = states;
    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      SparseMatrix Q;
      SparseMatrix G;
      std::size_t activePairs = 0;
      double consensusCost = 0.0;
      addConsensusTerms(dataset, neighbors, previousStates[robotId],
                        neighborStateCache[robotId], baseQs, robotId,
                        options.beta, d, Q, G, activePairs, consensusCost);
      addTranslationProxTerms(previousStates[robotId],
                              options.translationProxWeight, d, Q, G);
      QuadraticProblem problem(
          dataset.robots[robotId].trajectoryVertexIds.size() +
              dataset.numObjects,
          d, r);
      problem.setQ(Q);
      problem.setG(G);
      ROPTResult localResult;
      states[robotId] =
          optimizeLocal(problem, previousStates[robotId], options, iter,
                        localResult);
      if (options.projectToSEAfterLocalSolve) {
        states[robotId] = projectStateToSE(states[robotId], d);
      }
    }
    const ObjectExchangeStats exchangeStats =
        exchangeObjectUpdates(dataset, neighbors, options, states,
                              neighborStateCache, neighborObjectAges, d,
                              false);
    const ManualObjectDranObjectInterfaceResponseSummary responseSummary =
        applyObjectInterfaceSchurResponse(
            dataset, neighbors, options, baseQs, neighborStateCache,
            neighborObjectAges, exchangeStats.selectedForReceiver, states,
            objectInterfaceInnovationAccumulators, iter, d, r);
    if (options.objectInterfaceMode !=
        ManualObjectDranObjectInterfaceMode::CopyBaseline) {
      runResult.objectInterfaceResponseSummaries.push_back(responseSummary);
    }
    summarize(iter, neighbors, states, neighborStateCache,
              neighborObjectAges,
              exchangeStats.selectedForSend,
              exchangeStats.communicated,
              exchangeStats.staleObjectCommTriggers);
  }

  runResult.finalEstimates = states;
  runResult.elapsedSeconds = elapsedSeconds();
  return runResult;
}

}  // namespace DPGO

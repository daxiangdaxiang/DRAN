/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_utils.h>
#include <DPGO/GeodesicObjectProblem.h>
#include <DPGO/GeodesicSE3.h>
#include <DPGO/ObjectConsensus.h>
#include <DPGO/ObjectGaugeCoupling.h>
#include <DPGO/ObjectPGOData.h>
#include <DPGO/QuadraticOptimizer.h>
#include <DPGO/QuadraticProblem.h>
#include <DPGO/RoptOptimizer.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace std;
using namespace DPGO;

namespace {

double readEnvDouble(const char *name, double defaultValue);
size_t readEnvSizeT(const char *name, size_t defaultValue);
bool readEnvBool(const char *name, bool defaultValue);
string readEnvString(const char *name, const string &defaultValue);
double computeOriginalMeasurementCost(const ObjectPGORobotData &robot,
                                      const Matrix &state, unsigned d);

Matrix defaultPose(unsigned d) {
  Matrix pose = Matrix::Zero(d, d + 1);
  pose.block(0, 0, d, d) = Matrix::Identity(d, d);
  return pose;
}

Matrix assembleInitialState(const vector<Matrix> &trajectoryPoses,
                            const vector<Matrix> &objectPoses, unsigned d) {
  const size_t totalPoses = trajectoryPoses.size() + objectPoses.size();
  Matrix state = Matrix::Zero(d, totalPoses * (d + 1));
  for (size_t i = 0; i < totalPoses; ++i) {
    Matrix pose = i < trajectoryPoses.size() ? trajectoryPoses[i]
                                             : objectPoses[i - trajectoryPoses.size()];
    if (pose.rows() != static_cast<int>(d) ||
        pose.cols() != static_cast<int>(d + 1)) {
      pose = defaultPose(d);
    }
    state.block(0, i * (d + 1), d, d + 1) = pose;
  }
  return state;
}

Matrix assembleObjectAwareChordalState(
    const ObjectPGODirectoryData &dataset, size_t robotId,
    const Matrix &globalT, const vector<size_t> &trajectoryOffsets,
    size_t objectOffset, unsigned d) {
  const ObjectPGORobotData &robot = dataset.robots[robotId];
  const size_t totalPoses = robot.trajectoryVertexIds.size() + dataset.numObjects;
  Matrix state = Matrix::Zero(d, totalPoses * (d + 1));
  for (size_t localIdx = 0; localIdx < robot.trajectoryVertexIds.size();
       ++localIdx) {
    const size_t globalIdx = trajectoryOffsets[robotId] + localIdx;
    state.block(0, localIdx * (d + 1), d, d + 1) =
        globalT.block(0, globalIdx * (d + 1), d, d + 1);
  }
  for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
    const size_t localIdx = robot.trajectoryVertexIds.size() + objectId;
    const size_t globalIdx = objectOffset + objectId;
    state.block(0, localIdx * (d + 1), d, d + 1) =
        globalT.block(0, globalIdx * (d + 1), d, d + 1);
  }
  return state;
}

vector<RelativeSEMeasurement> objectAwareLocalMeasurements(
    const ObjectPGORobotData &robot) {
  vector<RelativeSEMeasurement> localMeasurements = robot.trajectoryMeasurements;
  localMeasurements.reserve(localMeasurements.size() +
                            robot.objectObservationMeasurements.size());
  for (const ObjectObservationMeasurement &obs :
       robot.objectObservationMeasurements) {
    RelativeSEMeasurement m = obs.measurement;
    m.p1 = obs.trajectoryLocalIndex;
    m.p2 = robot.trajectoryVertexIds.size() + obs.objectLocalIndex;
    m.r1 = robot.robotId;
    m.r2 = robot.robotId;
    localMeasurements.push_back(std::move(m));
  }
  return localMeasurements;
}

vector<Matrix> chordalObjectAwareInitialStates(
    const ObjectPGODirectoryData &dataset, unsigned d) {
  vector<size_t> trajectoryOffsets(dataset.robots.size(), 0);
  size_t trajectoryCount = 0;
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    trajectoryOffsets[robotId] = trajectoryCount;
    trajectoryCount += dataset.robots[robotId].trajectoryVertexIds.size();
  }
  const size_t objectOffset = trajectoryCount;
  const size_t totalPoses = trajectoryCount + dataset.numObjects;

  vector<RelativeSEMeasurement> globalMeasurements;
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
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

  Matrix globalT = chordalInitialization(d, totalPoses, globalMeasurements);
  vector<Matrix> states;
  states.reserve(dataset.robots.size());
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    states.push_back(assembleObjectAwareChordalState(
        dataset, robotId, globalT, trajectoryOffsets, objectOffset, d));
  }
  return states;
}

Matrix localObjectAwareChordalState(const ObjectPGORobotData &robot,
                                    size_t numObjects, unsigned d) {
  const size_t localNumPoses = robot.trajectoryVertexIds.size() + numObjects;
  Matrix state = assembleInitialState(robot.trajectoryInitialPoses,
                                      robot.objectInitialPoses, d);

  vector<RelativeSEMeasurement> localMeasurements =
      objectAwareLocalMeasurements(robot);

  vector<int> compactIndex(localNumPoses, -1);
  vector<bool> incident(localNumPoses, false);
  for (const RelativeSEMeasurement &m : localMeasurements) {
    if (m.p1 < localNumPoses) {
      incident[m.p1] = true;
    }
    if (m.p2 < localNumPoses) {
      incident[m.p2] = true;
    }
  }
  size_t compactNumPoses = 0;
  auto markPose = [&](size_t poseIndex) {
    if (poseIndex < localNumPoses && compactIndex[poseIndex] < 0) {
      compactIndex[poseIndex] = static_cast<int>(compactNumPoses++);
    }
  };
  if (!incident.empty() && incident[0]) {
    markPose(0);
  }
  for (const RelativeSEMeasurement &m : localMeasurements) {
    markPose(m.p1);
    markPose(m.p2);
  }
  if (compactNumPoses < 2 || localMeasurements.empty()) {
    return state;
  }

  vector<RelativeSEMeasurement> compactMeasurements;
  compactMeasurements.reserve(localMeasurements.size());
  for (RelativeSEMeasurement m : localMeasurements) {
    if (m.p1 >= localNumPoses || m.p2 >= localNumPoses ||
        compactIndex[m.p1] < 0 || compactIndex[m.p2] < 0) {
      continue;
    }
    m.p1 = static_cast<size_t>(compactIndex[m.p1]);
    m.p2 = static_cast<size_t>(compactIndex[m.p2]);
    m.r1 = 0;
    m.r2 = 0;
    compactMeasurements.push_back(std::move(m));
  }
  if (compactMeasurements.empty()) {
    return state;
  }

  Matrix compactState =
      chordalInitialization(d, compactNumPoses, compactMeasurements);
  for (size_t localIdx = 0; localIdx < localNumPoses; ++localIdx) {
    if (compactIndex[localIdx] >= 0) {
      state.block(0, localIdx * (d + 1), d, d + 1) =
          compactState.block(0,
                             static_cast<size_t>(compactIndex[localIdx]) *
                                 (d + 1),
                             d, d + 1);
    }
  }
  return state;
}

vector<Matrix> localChordalObjectAwareInitialStates(
    const ObjectPGODirectoryData &dataset, unsigned d) {
  vector<Matrix> states;
  states.reserve(dataset.robots.size());
  for (const ObjectPGORobotData &robot : dataset.robots) {
    try {
      states.push_back(localObjectAwareChordalState(robot, dataset.numObjects, d));
    } catch (const std::exception &e) {
      cerr << "Local chordal initialization failed for robot "
           << robot.robotId << ": " << e.what()
           << ". Falling back to g2o vertex initialization for this robot."
           << endl;
      states.push_back(assembleInitialState(robot.trajectoryInitialPoses,
                                            robot.objectInitialPoses, d));
    }
  }
  return states;
}

vector<unsigned> ringNeighbors(unsigned robotId, unsigned numRobots,
                               unsigned hops) {
  vector<unsigned> neighbors;
  if (numRobots < 2) {
    return neighbors;
  }
  hops = std::max(1u, std::min(hops, numRobots - 1));
  for (unsigned offset = 1; offset <= hops; ++offset) {
    const unsigned prev = (robotId + numRobots - offset) % numRobots;
    const unsigned next = (robotId + offset) % numRobots;
    if (prev != robotId &&
        std::find(neighbors.begin(), neighbors.end(), prev) == neighbors.end()) {
      neighbors.push_back(prev);
    }
    if (next != robotId &&
        std::find(neighbors.begin(), neighbors.end(), next) == neighbors.end()) {
      neighbors.push_back(next);
    }
  }
  return neighbors;
}

vector<unsigned> completeNeighbors(unsigned robotId, unsigned numRobots) {
  vector<unsigned> neighbors;
  neighbors.reserve(numRobots > 0 ? numRobots - 1 : 0);
  for (unsigned other = 0; other < numRobots; ++other) {
    if (other != robotId) {
      neighbors.push_back(other);
    }
  }
  return neighbors;
}

vector<unsigned> makeNeighbors(unsigned robotId, unsigned numRobots,
                               const string &topology, unsigned ringHops) {
  if (topology == "complete") {
    return completeNeighbors(robotId, numRobots);
  }
  return ringNeighbors(robotId, numRobots, ringHops);
}

Matrix blockPose(const Matrix &state, size_t poseIndex, unsigned d) {
  return state.block(0, poseIndex * (d + 1), d, d + 1);
}

struct WeightedNeighbor {
  unsigned id{0};
  double weight{1.0};
};

using WeightedNeighborRound = vector<vector<WeightedNeighbor>>;

enum class ObjectConsensusMode {
  DirectAdmm,
  EdgeAdmm,
  Penalty,
  ProxMixing,
  GradientTracking,
  ExactDiffusion
};

enum class ObjectLocalObjective {
  Chordal,
  Geodesic
};

enum class GeodesicSolverMode {
  Gradient,
  RtrGn
};

enum class GeodesicAuxiliaryMode {
  None,
  TangentResidualFeedback
};

struct EdgeKey {
  unsigned a{0};
  unsigned b{0};

  bool operator<(const EdgeKey &other) const {
    return a < other.a || (a == other.a && b < other.b);
  }
};

using EdgeObjectTargets = map<EdgeKey, vector<Matrix>>;

EdgeKey makeEdgeKey(unsigned robotA, unsigned robotB) {
  if (robotA < robotB) {
    return EdgeKey{robotA, robotB};
  }
  return EdgeKey{robotB, robotA};
}

ObjectConsensusMode parseObjectConsensusMode(const string &text) {
  if (text.empty() || text == "direct_admm" || text == "cadmm" ||
      text == "current") {
    return ObjectConsensusMode::DirectAdmm;
  }
  if (text == "edge_admm" || text == "edge-admm") {
    return ObjectConsensusMode::EdgeAdmm;
  }
  if (text == "penalty" || text == "prox_penalty") {
    return ObjectConsensusMode::Penalty;
  }
  if (text == "prox_mixing" || text == "proximal_mixing" ||
      text == "mixing") {
    return ObjectConsensusMode::ProxMixing;
  }
  if (text == "grad_tracking" || text == "gradient_tracking" ||
      text == "riemannian_gradient_tracking") {
    return ObjectConsensusMode::GradientTracking;
  }
  if (text == "exact_diffusion" || text == "riemannian_exact_diffusion" ||
      text == "riemannian_exact") {
    return ObjectConsensusMode::ExactDiffusion;
  }
  throw runtime_error("Unsupported object consensus mode: " + text);
}

ObjectLocalObjective parseObjectLocalObjective(const string &text) {
  if (text.empty() || text == "chordal" || text == "dpgo") {
    return ObjectLocalObjective::Chordal;
  }
  if (text == "geodesic" || text == "riemannian_geodesic") {
    return ObjectLocalObjective::Geodesic;
  }
  throw runtime_error("Unsupported object local objective: " + text);
}

string objectLocalObjectiveName(ObjectLocalObjective objective) {
  switch (objective) {
    case ObjectLocalObjective::Chordal:
      return "chordal";
    case ObjectLocalObjective::Geodesic:
      return "geodesic";
  }
  return "unknown";
}

GeodesicSolverMode parseGeodesicSolverMode(const string &text) {
  if (text.empty() || text == "rtr_gn" || text == "rtr-gn" ||
      text == "rtr" || text == "rtr_newton" || text == "newton" ||
      text == "approx_newton") {
    return GeodesicSolverMode::RtrGn;
  }
  if (text == "gradient" || text == "gd" || text == "legacy") {
    return GeodesicSolverMode::Gradient;
  }
  throw runtime_error("Unsupported OBJECT_GEODESIC_SOLVER: " + text);
}

string geodesicSolverModeName(GeodesicSolverMode mode) {
  switch (mode) {
    case GeodesicSolverMode::Gradient:
      return "gradient";
    case GeodesicSolverMode::RtrGn:
      return "rtr_gn";
  }
  return "unknown";
}

GeodesicAuxiliaryMode parseGeodesicAuxiliaryMode(const string &text) {
  if (text.empty() || text == "none" || text == "off" ||
      text == "false" || text == "0") {
    return GeodesicAuxiliaryMode::None;
  }
  if (text == "tangent_residual_feedback" ||
      text == "residual_feedback" ||
      text == "tangent_feedback") {
    return GeodesicAuxiliaryMode::TangentResidualFeedback;
  }
  if (text == "tangent_grad_tracking" ||
      text == "tangent_gradient_tracking" ||
      text == "geodesic_grad_tracking") {
    throw runtime_error(
        "OBJECT_GEODESIC_AUX_MODE=tangent_grad_tracking is reserved for a "
        "future true tangent-space tracker; use tangent_residual_feedback");
  }
  throw runtime_error("Unsupported OBJECT_GEODESIC_AUX_MODE: " + text);
}

string geodesicAuxiliaryModeName(GeodesicAuxiliaryMode mode) {
  switch (mode) {
    case GeodesicAuxiliaryMode::None:
      return "none";
    case GeodesicAuxiliaryMode::TangentResidualFeedback:
      return "tangent_residual_feedback";
  }
  return "unknown";
}

string objectConsensusModeName(ObjectConsensusMode mode) {
  switch (mode) {
    case ObjectConsensusMode::DirectAdmm:
      return "direct_admm";
    case ObjectConsensusMode::EdgeAdmm:
      return "edge_admm";
    case ObjectConsensusMode::Penalty:
      return "penalty";
    case ObjectConsensusMode::ProxMixing:
      return "prox_mixing";
    case ObjectConsensusMode::GradientTracking:
      return "grad_tracking";
    case ObjectConsensusMode::ExactDiffusion:
      return "exact_diffusion";
  }
  return "unknown";
}

bool usesQuadraticConsensus(ObjectConsensusMode mode) {
  return mode == ObjectConsensusMode::DirectAdmm ||
         mode == ObjectConsensusMode::EdgeAdmm ||
         mode == ObjectConsensusMode::Penalty ||
         mode == ObjectConsensusMode::ProxMixing ||
         mode == ObjectConsensusMode::GradientTracking ||
         mode == ObjectConsensusMode::ExactDiffusion;
}

bool usesDistributedObjectCorrection(ObjectConsensusMode mode) {
  return mode == ObjectConsensusMode::GradientTracking ||
         mode == ObjectConsensusMode::ExactDiffusion;
}

vector<string> splitCsvLine(const string &line) {
  vector<string> tokens;
  string token;
  stringstream ss(line);
  while (getline(ss, token, ',')) {
    tokens.push_back(token);
  }
  return tokens;
}

void addWeightedNeighbor(vector<WeightedNeighbor> &neighbors, unsigned id,
                         double weight) {
  for (WeightedNeighbor &neighbor : neighbors) {
    if (neighbor.id == id) {
      neighbor.weight = weight;
      return;
    }
  }
  neighbors.push_back(WeightedNeighbor{id, weight});
}

WeightedNeighborRound makeDefaultWeightedNeighbors(unsigned numRobots,
                                                   const string &topology,
                                                   unsigned ringHops) {
  WeightedNeighborRound round(numRobots);
  for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
    for (unsigned neighborId :
         makeNeighbors(robotId, numRobots, topology, ringHops)) {
      round[robotId].push_back(WeightedNeighbor{neighborId, 1.0});
    }
  }
  return round;
}

vector<WeightedNeighborRound> readTopologySchedule(const string &path,
                                                   unsigned numRobots,
                                                   const string &weightMode) {
  vector<WeightedNeighborRound> schedule;
  if (path.empty()) {
    return schedule;
  }

  ifstream input(path);
  if (!input.is_open()) {
    throw runtime_error("Could not open communication topology file: " + path);
  }

  string line;
  while (getline(input, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    vector<string> tokens = splitCsvLine(line);
    if (tokens.size() < 4 || tokens[0] == "round") {
      continue;
    }
    const unsigned roundIdx = static_cast<unsigned>(stoul(tokens[0]));
    const unsigned src = static_cast<unsigned>(stoul(tokens[1]));
    const unsigned dst = static_cast<unsigned>(stoul(tokens[2]));
    double weight = stod(tokens[3]);
    if (weightMode == "unit") {
      weight = 1.0;
    } else if (weightMode != "matrix") {
      throw runtime_error("Unsupported communication weight mode: " + weightMode);
    }
    if (src >= numRobots || dst >= numRobots || src == dst) {
      throw runtime_error("Invalid communication edge in " + path);
    }
    if (schedule.size() <= roundIdx) {
      schedule.resize(roundIdx + 1, WeightedNeighborRound(numRobots));
    }
    addWeightedNeighbor(schedule[roundIdx][src], dst, weight);
    addWeightedNeighbor(schedule[roundIdx][dst], src, weight);
  }
  return schedule;
}

WeightedNeighborRound unionWeightedNeighbors(
    const vector<WeightedNeighborRound> &schedule, unsigned numRobots) {
  WeightedNeighborRound result(numRobots);
  for (const WeightedNeighborRound &round : schedule) {
    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      for (const WeightedNeighbor &neighbor : round[robotId]) {
        addWeightedNeighbor(result[robotId], neighbor.id, neighbor.weight);
      }
    }
  }
  return result;
}

vector<bool> observedObjects(const ObjectPGORobotData &robot,
                             size_t numObjects) {
  vector<bool> observed(numObjects, false);
  for (const ObjectObservationMeasurement &obs :
       robot.objectObservationMeasurements) {
    if (obs.objectGlobalVertexId < numObjects) {
      observed[obs.objectGlobalVertexId] = true;
    }
  }
  return observed;
}

Matrix averagePoseBlocks(const vector<Matrix> &poses, unsigned d) {
  Matrix average = defaultPose(d);
  if (poses.empty()) {
    return average;
  }

  Matrix rotationMoment = Matrix::Zero(d, d);
  Matrix translation = Matrix::Zero(d, 1);
  for (const Matrix &pose : poses) {
    if (pose.rows() != static_cast<int>(d) ||
        pose.cols() != static_cast<int>(d + 1)) {
      continue;
    }
    rotationMoment += pose.leftCols(d);
    translation += pose.rightCols(1);
  }
  average.leftCols(d) = projectToRotationGroup(rotationMoment);
  average.rightCols(1) = translation / static_cast<double>(poses.size());
  return average;
}

Matrix weightedAveragePoseBlocks(const vector<pair<Matrix, double>> &poses,
                                 unsigned d) {
  Matrix average = defaultPose(d);
  if (poses.empty()) {
    return average;
  }

  Matrix rotationMoment = Matrix::Zero(d, d);
  Matrix translation = Matrix::Zero(d, 1);
  double totalWeight = 0.0;
  for (const auto &entry : poses) {
    const Matrix &pose = entry.first;
    const double weight = entry.second;
    if (weight <= 0.0 || pose.rows() != static_cast<int>(d) ||
        pose.cols() != static_cast<int>(d + 1)) {
      continue;
    }
    rotationMoment += weight * pose.leftCols(d);
    translation += weight * pose.rightCols(1);
    totalWeight += weight;
  }
  if (totalWeight <= 0.0) {
    return average;
  }

  average.leftCols(d) = projectToRotationGroup(rotationMoment);
  average.rightCols(1) = translation / totalWeight;
  return average;
}

Matrix dampedPoseUpdate(const Matrix &currentPose, const Matrix &targetPose,
                        double alpha, unsigned d) {
  Matrix updated = currentPose;
  if (alpha <= 0.0) {
    return updated;
  }
  const double clampedAlpha = std::max(0.0, std::min(1.0, alpha));
  const Matrix blended =
      (1.0 - clampedAlpha) * currentPose + clampedAlpha * targetPose;
  updated.leftCols(d) = projectToRotationGroup(blended.leftCols(d));
  updated.rightCols(1) = blended.rightCols(1);
  return updated;
}

Matrix symPart(const Matrix &A) {
  return 0.5 * (A + A.transpose());
}

Matrix projectPoseGradient(const Matrix &pose, const Matrix &ambientGradient,
                           unsigned d) {
  Matrix projected = ambientGradient;
  if (pose.rows() != static_cast<int>(d) ||
      pose.cols() != static_cast<int>(d + 1) ||
      ambientGradient.rows() != static_cast<int>(d) ||
      ambientGradient.cols() != static_cast<int>(d + 1)) {
    return projected;
  }
  const Matrix R = pose.leftCols(d);
  const Matrix G = ambientGradient.leftCols(d);
  projected.leftCols(d) = G - R * symPart(R.transpose() * G);
  projected.rightCols(1) = ambientGradient.rightCols(1);
  return projected;
}

Matrix skew3(const Matrix &v) {
  Matrix S = Matrix::Zero(3, 3);
  S(0, 1) = -v(2);
  S(0, 2) = v(1);
  S(1, 0) = v(2);
  S(1, 2) = -v(0);
  S(2, 0) = -v(1);
  S(2, 1) = v(0);
  return S;
}

Matrix so3LogVector(const Matrix &R) {
  Matrix omega = Matrix::Zero(3, 1);
  const double cosTheta =
      std::max(-1.0, std::min(1.0, 0.5 * (R.trace() - 1.0)));
  const double theta = std::acos(cosTheta);
  const Matrix skew = 0.5 * (R - R.transpose());
  omega << skew(2, 1), skew(0, 2), skew(1, 0);
  if (theta < 1e-10) {
    return omega;
  }
  const double sinTheta = std::sin(theta);
  if (std::abs(sinTheta) < 1e-10) {
    return theta * omega / std::max(1e-12, omega.norm());
  }
  return (theta / sinTheta) * omega;
}

Matrix so3ExpVector(const Matrix &omega) {
  return expmap(skew3(omega));
}

Matrix projectPoseBlock(const Matrix &pose, unsigned d) {
  Matrix projected = pose;
  if (pose.rows() == static_cast<int>(d) &&
      pose.cols() == static_cast<int>(d + 1)) {
    projected.leftCols(d) = projectToRotationGroup(pose.leftCols(d));
  }
  return projected;
}

Matrix weightedAverageBlocks(const vector<pair<Matrix, double>> &blocks,
                             const Matrix &fallback) {
  if (blocks.empty()) {
    return fallback;
  }
  Matrix sum = Matrix::Zero(fallback.rows(), fallback.cols());
  double totalWeight = 0.0;
  for (const auto &entry : blocks) {
    const Matrix &block = entry.first;
    const double weight = entry.second;
    if (weight <= 0.0 || block.rows() != fallback.rows() ||
        block.cols() != fallback.cols()) {
      continue;
    }
    sum += weight * block;
    totalWeight += weight;
  }
  if (totalWeight <= 0.0) {
    return fallback;
  }
  return sum / totalWeight;
}

double selfMixingWeight(const vector<WeightedNeighbor> &neighbors,
                        const string &weightMode) {
  if (weightMode == "matrix") {
    double offDiagonalWeight = 0.0;
    for (const WeightedNeighbor &neighbor : neighbors) {
      offDiagonalWeight += std::max(0.0, neighbor.weight);
    }
    return std::max(0.0, 1.0 - offDiagonalWeight);
  }
  return 1.0 / static_cast<double>(neighbors.size() + 1);
}

const WeightedNeighbor *findWeightedNeighbor(
    const vector<WeightedNeighbor> &neighbors, unsigned neighborId) {
  for (const WeightedNeighbor &neighbor : neighbors) {
    if (neighbor.id == neighborId) {
      return &neighbor;
    }
  }
  return nullptr;
}

double robotObjectConfidence(
    size_t robotId, size_t objectId,
    const vector<vector<bool>> &observedObjects,
    const vector<vector<bool>> &knownObjects,
    double observedConfidence, double relayConfidence) {
  if (robotId >= knownObjects.size() ||
      objectId >= knownObjects[robotId].size() ||
      !knownObjects[robotId][objectId]) {
    return 0.0;
  }
  const bool observed =
      robotId < observedObjects.size() &&
      objectId < observedObjects[robotId].size() &&
      observedObjects[robotId][objectId];
  return observed ? observedConfidence : relayConfidence;
}

double effectiveConsensusWeight(
    double topologyWeight, size_t robotId, size_t neighborId,
    size_t objectId, const vector<vector<bool>> &observedObjects,
    const vector<vector<bool>> &knownObjects,
    double observedConfidence, double relayConfidence) {
  const double localConfidence =
      robotObjectConfidence(robotId, objectId, observedObjects, knownObjects,
                            observedConfidence, relayConfidence);
  const double neighborConfidence =
      robotObjectConfidence(neighborId, objectId, observedObjects, knownObjects,
                            observedConfidence, relayConfidence);
  if (localConfidence <= 0.0 || neighborConfidence <= 0.0) {
    return 0.0;
  }
  return topologyWeight * std::sqrt(localConfidence * neighborConfidence);
}

size_t countKnownObjectCopies(const vector<vector<bool>> &knownObjects) {
  size_t count = 0;
  for (const vector<bool> &robotKnownObjects : knownObjects) {
    count += static_cast<size_t>(
        std::count(robotKnownObjects.begin(), robotKnownObjects.end(), true));
  }
  return count;
}

size_t countRelayObjectCopies(const vector<vector<bool>> &knownObjects,
                              const vector<vector<bool>> &observedObjects) {
  size_t count = 0;
  for (size_t robotId = 0; robotId < knownObjects.size(); ++robotId) {
    for (size_t objectId = 0; objectId < knownObjects[robotId].size();
         ++objectId) {
      const bool observed =
          robotId < observedObjects.size() &&
          objectId < observedObjects[robotId].size() &&
          observedObjects[robotId][objectId];
      if (knownObjects[robotId][objectId] && !observed) {
        ++count;
      }
    }
  }
  return count;
}

size_t initializeRelayObjectStates(const ObjectPGODirectoryData &dataset,
                                   const WeightedNeighborRound &neighbors,
                                   unsigned d, vector<Matrix> &states,
                                   vector<vector<bool>> &knownObjects) {
  const size_t numRobots = dataset.robots.size();
  if (numRobots == 0 || dataset.numObjects == 0) {
    return 0;
  }

  size_t initialized = 0;
  for (size_t pass = 0; pass < numRobots; ++pass) {
    const vector<vector<bool>> previousKnown = knownObjects;
    const vector<Matrix> previousStates = states;
    vector<tuple<size_t, size_t, Matrix>> updates;

    for (size_t robotId = 0; robotId < numRobots; ++robotId) {
      if (robotId >= neighbors.size()) {
        continue;
      }
      for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
        if (previousKnown[robotId][objectId]) {
          continue;
        }

        vector<Matrix> neighborPredictions;
        for (const WeightedNeighbor &neighbor : neighbors[robotId]) {
          const size_t neighborId = neighbor.id;
          if (neighborId >= numRobots ||
              !previousKnown[neighborId][objectId]) {
            continue;
          }
          const size_t neighborObjectIndex =
              dataset.robots[neighborId].trajectoryVertexIds.size() + objectId;
          neighborPredictions.push_back(
              blockPose(previousStates[neighborId], neighborObjectIndex, d));
        }
        if (neighborPredictions.empty()) {
          continue;
        }

        updates.emplace_back(robotId, objectId,
                             averagePoseBlocks(neighborPredictions, d));
      }
    }

    if (updates.empty()) {
      break;
    }

    for (const auto &update : updates) {
      const size_t robotId = std::get<0>(update);
      const size_t objectId = std::get<1>(update);
      const Matrix &pose = std::get<2>(update);
      const size_t objectIndex =
          dataset.robots[robotId].trajectoryVertexIds.size() + objectId;
      states[robotId].block(0, objectIndex * (d + 1), d, d + 1) = pose;
      if (!knownObjects[robotId][objectId]) {
        knownObjects[robotId][objectId] = true;
        ++initialized;
      }
    }
  }

  cout << "Object relay initialization known copies = "
       << countKnownObjectCopies(knownObjects) << "/"
       << numRobots * dataset.numObjects
       << " | initialized_from_neighbors = " << initialized << "." << endl;
  return initialized;
}

EdgeObjectTargets initializeEdgeObjectTargets(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &neighbors,
    const vector<vector<bool>> &knownObjects,
    const vector<Matrix> &states, unsigned d) {
  EdgeObjectTargets targets;
  for (unsigned robotId = 0; robotId < neighbors.size(); ++robotId) {
    const size_t objectOffset =
        dataset.robots[robotId].trajectoryVertexIds.size();
    for (const WeightedNeighbor &neighbor : neighbors[robotId]) {
      const unsigned neighborId = neighbor.id;
      if (robotId > neighborId || neighborId >= dataset.robots.size()) {
        continue;
      }
      const size_t neighborObjectOffset =
          dataset.robots[neighborId].trajectoryVertexIds.size();
      const EdgeKey edge = makeEdgeKey(robotId, neighborId);
      vector<Matrix> &edgeTargets = targets[edge];
      edgeTargets.resize(dataset.numObjects);
      for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
        if (!knownObjects[robotId][objectId] ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        vector<Matrix> endpointPoses;
        endpointPoses.push_back(
            blockPose(states[robotId], objectOffset + objectId, d));
        endpointPoses.push_back(
            blockPose(states[neighborId], neighborObjectOffset + objectId, d));
        edgeTargets[objectId] = averagePoseBlocks(endpointPoses, d);
      }
    }
  }
  return targets;
}

size_t updateEdgeObjectTargets(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &observedObjects,
    const vector<vector<bool>> &knownObjects,
    const vector<ObjectConsensus> &consensuses,
    const vector<Matrix> &states, double beta, double observedConfidence,
    double relayConfidence, unsigned d,
    EdgeObjectTargets &targets) {
  size_t updated = 0;
  if (beta <= 0.0) {
    return updated;
  }

  for (unsigned robotId = 0; robotId < activeNeighbors.size(); ++robotId) {
    for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
      const unsigned neighborId = neighbor.id;
      if (robotId > neighborId || neighborId >= dataset.robots.size()) {
        continue;
      }
      const WeightedNeighbor *reverseNeighbor =
          neighborId < activeNeighbors.size()
              ? findWeightedNeighbor(activeNeighbors[neighborId], robotId)
              : nullptr;
      const double weightA = neighbor.weight;
      const double weightB =
          reverseNeighbor == nullptr ? neighbor.weight : reverseNeighbor->weight;
      const size_t objectOffsetA =
          dataset.robots[robotId].trajectoryVertexIds.size();
      const size_t objectOffsetB =
          dataset.robots[neighborId].trajectoryVertexIds.size();
      const EdgeKey edge = makeEdgeKey(robotId, neighborId);
      vector<Matrix> &edgeTargets = targets[edge];
      edgeTargets.resize(dataset.numObjects);

      for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
        if (!knownObjects[robotId][objectId] ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        const double effectiveWeightA = effectiveConsensusWeight(
            weightA, robotId, neighborId, objectId, observedObjects,
            knownObjects, observedConfidence, relayConfidence);
        const double effectiveWeightB = effectiveConsensusWeight(
            weightB, neighborId, robotId, objectId, observedObjects,
            knownObjects, observedConfidence, relayConfidence);
        if (effectiveWeightA <= 0.0 || effectiveWeightB <= 0.0) {
          continue;
        }

        Matrix numerator = Matrix::Zero(d, d + 1);
        double denominator = 0.0;
        auto addEndpoint = [&](unsigned sideId, unsigned otherId,
                               double weight, size_t objectOffset) {
          const Matrix pose = blockPose(states[sideId],
                                        objectOffset + objectId, d);
          const Matrix lambda =
              consensuses[sideId].getDual(otherId, objectId);
          numerator += beta * weight * weight * pose + weight * lambda;
          denominator += beta * weight * weight;
        };

        addEndpoint(robotId, neighborId, effectiveWeightA, objectOffsetA);
        addEndpoint(neighborId, robotId, effectiveWeightB, objectOffsetB);
        if (denominator <= 0.0) {
          continue;
        }

        Matrix target = numerator / denominator;
        target.leftCols(d) = projectToRotationGroup(target.leftCols(d));
        edgeTargets[objectId] = target;
        ++updated;
      }
    }
  }
  return updated;
}

struct FrameAlignment {
  Matrix R;
  Matrix t;
  size_t commonObjects{0};
  double translationRms{0.0};
  double rotationRms{0.0};
  double score{std::numeric_limits<double>::infinity()};
};

struct RobotFrameEdge {
  size_t targetId{0};
  size_t sourceId{0};
  FrameAlignment alignment;
  double weight{1.0};
};

struct FrameGauge {
  Matrix R;
  Matrix t;
};

struct FrameEdgeSelectionStats {
  size_t candidateEdges{0};
  size_t treeEdges{0};
  size_t testedCycleEdges{0};
  size_t acceptedCycleEdges{0};
  size_t rejectedCycleEdges{0};
  size_t disconnectedCycleEdges{0};
  double cycleQuantile{0.0};
  size_t cycleDof{0};
  double cycleChi2Threshold{0.0};
  double translationSigmaFloor{0.0};
  double rotationSigmaFloor{0.0};
  double meanCycleTranslationResidual{0.0};
  double meanCycleRotationResidual{0.0};
  double maxCycleTranslationResidual{0.0};
  double maxCycleRotationResidual{0.0};
  double meanAcceptedZ2{0.0};
  double maxAcceptedZ2{0.0};
  double minRejectedZ2{0.0};
};

struct FrameGaugeSolverStats {
  string solver{"centralized"};
  size_t rotationIters{0};
  size_t translationIters{0};
  bool rotationConverged{true};
  bool translationConverged{true};
  double rotationStepNorm{0.0};
  double translationStepNorm{0.0};
  double objective{0.0};
  double commMb{0.0};
};

struct FrameSyncRunStats {
  string mode{"graph_sync"};
  string edgePolicy{"directed"};
  FrameEdgeSelectionStats edgeSelection;
  FrameGaugeSolverStats solver;
  bool applied{false};
  size_t numRobots{0};
  size_t connectedRobots{0};
  size_t candidateEdges{0};
  size_t usedEdges{0};
  double meanTranslationRms{0.0};
  double meanRotationRms{0.0};
  double meanScore{0.0};
  double maxScore{0.0};
  double registrationCommMb{0.0};
  double solverCommMb{0.0};
  double commMb{0.0};
};

struct ObjectCopyChordalJacobiStats {
  string scope{"objects_only"};
  size_t rotationIters{0};
  size_t translationIters{0};
  bool rotationConverged{false};
  bool translationConverged{false};
  bool adaptiveBudget{false};
  size_t adaptiveMinIters{0};
  double adaptiveStepTol{0.0};
  double adaptiveRelativeStepDropTol{0.0};
  string rotationStopReason{"not_run"};
  string translationStopReason{"not_run"};
  double rotationStepNorm{0.0};
  double translationStepNorm{0.0};
  double localObjective{0.0};
  double objectLocalObjective{0.0};
  double fixedTrajectoryObjective{0.0};
  double consensusObjective{0.0};
  double scopeObjective{0.0};
  double objective{0.0};
  double commMb{0.0};
  size_t activeDirectedObjectPairs{0};
  size_t updatedVariables{0};
  double consensusKappa{1.0};
  double consensusTau{1.0};
};

bool isDistributedFrameSolver(const string &solver) {
  return solver == "distributed_jacobi" || solver == "jacobi" ||
         solver == "distributed";
}

bool isCentralizedFrameSolver(const string &solver) {
  return solver.empty() || solver == "centralized" ||
         solver == "centralized_qr" || solver == "qr";
}

bool isCycleFrameEdgePolicy(const string &edgePolicy) {
  return edgePolicy == "cycle_consistent" || edgePolicy == "cycle" ||
         edgePolicy == "robust_cycle";
}

FrameGauge identityFrameGauge(unsigned d) {
  FrameGauge gauge;
  gauge.R = Matrix::Identity(d, d);
  gauge.t = Matrix::Zero(d, 1);
  return gauge;
}

double squared(double value) { return value * value; }

double frameEdgeTranslationSigma(const RobotFrameEdge &edge) {
  const double count =
      std::sqrt(static_cast<double>(std::max<size_t>(edge.alignment.commonObjects, 1)));
  return edge.alignment.translationRms / count;
}

double frameEdgeRotationSigma(const RobotFrameEdge &edge) {
  const double count =
      std::sqrt(static_cast<double>(std::max<size_t>(edge.alignment.commonObjects, 1)));
  return edge.alignment.rotationRms / count;
}

double medianValue(vector<double> values, double fallback) {
  if (values.empty()) {
    return fallback;
  }
  std::sort(values.begin(), values.end());
  const size_t mid = values.size() / 2;
  if (values.size() % 2 == 1) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1] + values[mid]);
}

bool isBetterFrameTreeEdge(const RobotFrameEdge &candidate,
                           const RobotFrameEdge &best) {
  if (std::abs(candidate.weight - best.weight) > 1e-12) {
    return candidate.weight > best.weight;
  }
  if (std::abs(candidate.alignment.score - best.alignment.score) > 1e-12) {
    return candidate.alignment.score < best.alignment.score;
  }
  if (candidate.alignment.commonObjects != best.alignment.commonObjects) {
    return candidate.alignment.commonObjects > best.alignment.commonObjects;
  }
  if (candidate.targetId != best.targetId) {
    return candidate.targetId < best.targetId;
  }
  return candidate.sourceId < best.sourceId;
}

double treePathVariance(size_t robotA, size_t robotB,
                        const vector<int> &parent,
                        const vector<double> &parentEdgeVariance) {
  map<size_t, double> varianceToAncestor;
  double variance = 0.0;
  size_t current = robotA;
  while (true) {
    varianceToAncestor[current] = variance;
    if (parent[current] < 0) {
      break;
    }
    variance += parentEdgeVariance[current];
    current = static_cast<size_t>(parent[current]);
  }

  variance = 0.0;
  current = robotB;
  while (true) {
    const auto it = varianceToAncestor.find(current);
    if (it != varianceToAncestor.end()) {
      return it->second + variance;
    }
    if (parent[current] < 0) {
      break;
    }
    variance += parentEdgeVariance[current];
    current = static_cast<size_t>(parent[current]);
  }
  return std::numeric_limits<double>::infinity();
}

size_t treePathLength(size_t robotA, size_t robotB,
                      const vector<int> &parent) {
  map<size_t, size_t> distanceToAncestor;
  size_t distance = 0;
  size_t current = robotA;
  while (true) {
    distanceToAncestor[current] = distance;
    if (parent[current] < 0) {
      break;
    }
    ++distance;
    current = static_cast<size_t>(parent[current]);
  }

  distance = 0;
  current = robotB;
  while (true) {
    const auto it = distanceToAncestor.find(current);
    if (it != distanceToAncestor.end()) {
      return it->second + distance;
    }
    if (parent[current] < 0) {
      break;
    }
    ++distance;
    current = static_cast<size_t>(parent[current]);
  }
  return 0;
}

vector<RobotFrameEdge> selectCycleConsistentFrameEdges(
    const vector<RobotFrameEdge> &candidates, size_t numRobots, unsigned d,
    double cycleQuantile, const string &edgeLogPath,
    FrameEdgeSelectionStats &stats) {
  stats = FrameEdgeSelectionStats();
  stats.candidateEdges = candidates.size();
  stats.cycleQuantile = cycleQuantile;
  // z2 is built from two aggregated scalar residual norms: translation and
  // chordal rotation. Use two degrees of freedom for that diagnostic gate.
  stats.cycleDof = 2;
  stats.cycleChi2Threshold =
      chi2inv(std::min(1.0 - 1e-9, std::max(1e-9, cycleQuantile)),
              stats.cycleDof);
  if (numRobots == 0 || candidates.empty()) {
    return {};
  }

  vector<double> translationSigmas;
  vector<double> rotationSigmas;
  translationSigmas.reserve(candidates.size());
  rotationSigmas.reserve(candidates.size());
  for (const RobotFrameEdge &edge : candidates) {
    translationSigmas.push_back(frameEdgeTranslationSigma(edge));
    rotationSigmas.push_back(frameEdgeRotationSigma(edge));
  }
  stats.translationSigmaFloor =
      std::max(1e-9, 0.1 * medianValue(translationSigmas, 1e-6));
  stats.rotationSigmaFloor =
      std::max(1e-9, 0.1 * medianValue(rotationSigmas, 1e-6));

  ofstream edgeLog;
  if (!edgeLogPath.empty()) {
    edgeLog.open(edgeLogPath);
    if (!edgeLog.is_open()) {
      cerr << "Warning: could not open OBJECT_FRAME_SYNC_EDGE_LOG at "
           << edgeLogPath << "; frame edge diagnostics will not be saved."
           << endl;
    } else {
      edgeLog << "target,source,is_tree,accepted,reason,path_len,common_objects,"
                 "score,weight,translation_rms,rotation_rms,cycle_trans,"
                 "cycle_rot,sigma_trans,sigma_rot,z2,chi2_threshold\n";
    }
  }

  vector<RobotFrameEdge> selected;
  selected.reserve(candidates.size());
  vector<bool> connected(numRobots, false);
  vector<bool> treeEdge(candidates.size(), false);
  vector<FrameGauge> gauges(numRobots);
  vector<int> parent(numRobots, -1);
  vector<double> parentTranslationVariance(numRobots, 0.0);
  vector<double> parentRotationVariance(numRobots, 0.0);
  connected[0] = true;
  gauges[0] = identityFrameGauge(d);
  size_t connectedCount = 1;

  while (connectedCount < numRobots) {
    bool found = false;
    size_t bestIndex = 0;
    for (size_t edgeIndex = 0; edgeIndex < candidates.size(); ++edgeIndex) {
      const RobotFrameEdge &edge = candidates[edgeIndex];
      const bool targetConnected = connected[edge.targetId];
      const bool sourceConnected = connected[edge.sourceId];
      if (targetConnected == sourceConnected) {
        continue;
      }
      if (!found ||
          isBetterFrameTreeEdge(edge, candidates[bestIndex])) {
        found = true;
        bestIndex = edgeIndex;
      }
    }
    if (!found) {
      break;
    }

    const RobotFrameEdge &edge = candidates[bestIndex];
    const double translationVariance =
        squared(std::max(frameEdgeTranslationSigma(edge),
                         stats.translationSigmaFloor));
    const double rotationVariance =
        squared(std::max(frameEdgeRotationSigma(edge),
                         stats.rotationSigmaFloor));
    if (connected[edge.targetId] && !connected[edge.sourceId]) {
      const FrameGauge &targetGauge = gauges[edge.targetId];
      gauges[edge.sourceId].R = targetGauge.R * edge.alignment.R;
      gauges[edge.sourceId].t =
          targetGauge.R * edge.alignment.t + targetGauge.t;
      parent[edge.sourceId] = static_cast<int>(edge.targetId);
      parentTranslationVariance[edge.sourceId] = translationVariance;
      parentRotationVariance[edge.sourceId] = rotationVariance;
      connected[edge.sourceId] = true;
    } else if (connected[edge.sourceId] && !connected[edge.targetId]) {
      const FrameGauge &sourceGauge = gauges[edge.sourceId];
      gauges[edge.targetId].R = sourceGauge.R * edge.alignment.R.transpose();
      gauges[edge.targetId].t =
          sourceGauge.t - gauges[edge.targetId].R * edge.alignment.t;
      parent[edge.targetId] = static_cast<int>(edge.sourceId);
      parentTranslationVariance[edge.targetId] = translationVariance;
      parentRotationVariance[edge.targetId] = rotationVariance;
      connected[edge.targetId] = true;
    }
    treeEdge[bestIndex] = true;
    selected.push_back(edge);
    ++stats.treeEdges;
    ++connectedCount;
    if (edgeLog.is_open()) {
      edgeLog << edge.targetId << ',' << edge.sourceId << ",1,1,tree,"
              << 0 << ',' << edge.alignment.commonObjects << ','
              << edge.alignment.score << ',' << edge.weight << ','
              << edge.alignment.translationRms << ','
              << edge.alignment.rotationRms << ",0,0,0,0,0,"
              << stats.cycleChi2Threshold << '\n';
    }
  }

  double translationResidualSum = 0.0;
  double rotationResidualSum = 0.0;
  double acceptedZ2Sum = 0.0;
  double minRejectedZ2 = std::numeric_limits<double>::infinity();
  for (size_t edgeIndex = 0; edgeIndex < candidates.size(); ++edgeIndex) {
    if (treeEdge[edgeIndex]) {
      continue;
    }
    const RobotFrameEdge &edge = candidates[edgeIndex];
    if (!connected[edge.targetId] || !connected[edge.sourceId]) {
      ++stats.disconnectedCycleEdges;
      if (edgeLog.is_open()) {
        edgeLog << edge.targetId << ',' << edge.sourceId
                << ",0,0,disconnected,0," << edge.alignment.commonObjects
                << ',' << edge.alignment.score << ',' << edge.weight << ','
                << edge.alignment.translationRms << ','
                << edge.alignment.rotationRms << ",0,0,0,0,0,"
                << stats.cycleChi2Threshold << '\n';
      }
      continue;
    }

    const FrameGauge &targetGauge = gauges[edge.targetId];
    const FrameGauge &sourceGauge = gauges[edge.sourceId];
    const double cycleTranslation =
        (sourceGauge.t - targetGauge.t -
         targetGauge.R * edge.alignment.t).norm();
    const double cycleRotation =
        (targetGauge.R * edge.alignment.R - sourceGauge.R).norm();
    const double pathTranslationVariance =
        treePathVariance(edge.targetId, edge.sourceId, parent,
                         parentTranslationVariance);
    const double pathRotationVariance =
        treePathVariance(edge.targetId, edge.sourceId, parent,
                         parentRotationVariance);
    const double edgeTranslationVariance =
        squared(std::max(frameEdgeTranslationSigma(edge),
                         stats.translationSigmaFloor));
    const double edgeRotationVariance =
        squared(std::max(frameEdgeRotationSigma(edge),
                         stats.rotationSigmaFloor));
    const double sigmaTranslation =
        std::sqrt(std::max(squared(stats.translationSigmaFloor),
                           edgeTranslationVariance + pathTranslationVariance));
    const double sigmaRotation =
        std::sqrt(std::max(squared(stats.rotationSigmaFloor),
                           edgeRotationVariance + pathRotationVariance));
    const double z2 = squared(cycleTranslation) /
                          std::max(squared(sigmaTranslation),
                                   squared(stats.translationSigmaFloor)) +
                      squared(cycleRotation) /
                          std::max(squared(sigmaRotation),
                                   squared(stats.rotationSigmaFloor));
    const bool accepted = z2 <= stats.cycleChi2Threshold;
    ++stats.testedCycleEdges;
    translationResidualSum += cycleTranslation;
    rotationResidualSum += cycleRotation;
    stats.maxCycleTranslationResidual =
        std::max(stats.maxCycleTranslationResidual, cycleTranslation);
    stats.maxCycleRotationResidual =
        std::max(stats.maxCycleRotationResidual, cycleRotation);
    if (accepted) {
      selected.push_back(edge);
      ++stats.acceptedCycleEdges;
      acceptedZ2Sum += z2;
      stats.maxAcceptedZ2 = std::max(stats.maxAcceptedZ2, z2);
    } else {
      ++stats.rejectedCycleEdges;
      minRejectedZ2 = std::min(minRejectedZ2, z2);
    }
    if (edgeLog.is_open()) {
      const size_t pathLen =
          treePathLength(edge.targetId, edge.sourceId, parent);
      edgeLog << edge.targetId << ',' << edge.sourceId << ",0,"
              << (accepted ? 1 : 0) << ','
              << (accepted ? "cycle_consistent" : "cycle_rejected") << ','
              << pathLen << ',' << edge.alignment.commonObjects << ','
              << edge.alignment.score << ',' << edge.weight << ','
              << edge.alignment.translationRms << ','
              << edge.alignment.rotationRms << ',' << cycleTranslation << ','
              << cycleRotation << ',' << sigmaTranslation << ','
              << sigmaRotation << ',' << z2 << ','
              << stats.cycleChi2Threshold << '\n';
    }
  }

  if (stats.testedCycleEdges > 0) {
    stats.meanCycleTranslationResidual =
        translationResidualSum / static_cast<double>(stats.testedCycleEdges);
    stats.meanCycleRotationResidual =
        rotationResidualSum / static_cast<double>(stats.testedCycleEdges);
  }
  if (stats.acceptedCycleEdges > 0) {
    stats.meanAcceptedZ2 =
        acceptedZ2Sum / static_cast<double>(stats.acceptedCycleEdges);
  }
  if (stats.rejectedCycleEdges > 0) {
    stats.minRejectedZ2 = minRejectedZ2;
  }
  return selected;
}

double evaluateFrameGaugeObjective(
    unsigned d, const Matrix &gauges,
    const vector<RelativeSEMeasurement> &measurements) {
  double objective = 0.0;
  for (const RelativeSEMeasurement &m : measurements) {
    const Matrix Rtarget = gauges.block(0, m.p1 * (d + 1), d, d);
    const Matrix ttarget = gauges.block(0, m.p1 * (d + 1) + d, d, 1);
    const Matrix Rsource = gauges.block(0, m.p2 * (d + 1), d, d);
    const Matrix tsource = gauges.block(0, m.p2 * (d + 1) + d, d, 1);
    objective += m.kappa * (Rtarget * m.R - Rsource).squaredNorm();
    objective += m.tau * (tsource - ttarget - Rtarget * m.t).squaredNorm();
  }
  return objective;
}

Matrix solveFrameGaugeJacobi(
    unsigned d, size_t numPoses,
    const vector<RelativeSEMeasurement> &measurements,
    size_t maxIters, double tolerance, bool projectEachIter,
    FrameGaugeSolverStats &stats) {
  stats = FrameGaugeSolverStats();
  stats.solver = "distributed_jacobi";
  stats.rotationConverged = false;
  stats.translationConverged = false;
  if (numPoses == 0) {
    stats.rotationConverged = true;
    stats.translationConverged = true;
    return Matrix::Zero(d, 0);
  }

  vector<Matrix> rotations(numPoses, Matrix::Identity(d, d));
  vector<Matrix> translations(numPoses, Matrix::Zero(d, 1));
  const Matrix I = Matrix::Identity(d, d);
  const Matrix zero = Matrix::Zero(d, 1);

  for (size_t iter = 0; iter < maxIters; ++iter) {
    vector<Matrix> nextRotations = rotations;
    double stepSq = 0.0;
    for (size_t poseId = 1; poseId < numPoses; ++poseId) {
      Matrix numerator = Matrix::Zero(d, d);
      double denominator = 0.0;
      for (const RelativeSEMeasurement &m : measurements) {
        if (m.p2 == poseId) {
          numerator += m.kappa * rotations[m.p1] * m.R;
          denominator += m.kappa;
        } else if (m.p1 == poseId) {
          numerator += m.kappa * rotations[m.p2] * m.R.transpose();
          denominator += m.kappa;
        }
      }
      if (denominator <= 0.0) {
        continue;
      }
      Matrix updated = numerator / denominator;
      if (projectEachIter) {
        updated = projectToRotationGroup(updated);
      }
      stepSq += (updated - rotations[poseId]).squaredNorm();
      nextRotations[poseId] = updated;
    }
    nextRotations[0] = I;
    rotations = std::move(nextRotations);
    stats.rotationStepNorm = std::sqrt(stepSq);
    stats.rotationIters = iter + 1;
    if (stats.rotationStepNorm <= tolerance) {
      stats.rotationConverged = true;
      break;
    }
  }
  for (size_t poseId = 1; poseId < numPoses; ++poseId) {
    rotations[poseId] = projectToRotationGroup(rotations[poseId]);
  }
  rotations[0] = I;

  for (size_t iter = 0; iter < maxIters; ++iter) {
    vector<Matrix> nextTranslations = translations;
    double stepSq = 0.0;
    for (size_t poseId = 1; poseId < numPoses; ++poseId) {
      Matrix numerator = Matrix::Zero(d, 1);
      double denominator = 0.0;
      for (const RelativeSEMeasurement &m : measurements) {
        if (m.p2 == poseId) {
          numerator += m.tau * (translations[m.p1] + rotations[m.p1] * m.t);
          denominator += m.tau;
        } else if (m.p1 == poseId) {
          numerator += m.tau * (translations[m.p2] - rotations[poseId] * m.t);
          denominator += m.tau;
        }
      }
      if (denominator <= 0.0) {
        continue;
      }
      const Matrix updated = numerator / denominator;
      stepSq += (updated - translations[poseId]).squaredNorm();
      nextTranslations[poseId] = updated;
    }
    nextTranslations[0] = zero;
    translations = std::move(nextTranslations);
    stats.translationStepNorm = std::sqrt(stepSq);
    stats.translationIters = iter + 1;
    if (stats.translationStepNorm <= tolerance) {
      stats.translationConverged = true;
      break;
    }
  }
  translations[0] = zero;

  Matrix gauges = Matrix::Zero(d, numPoses * (d + 1));
  for (size_t poseId = 0; poseId < numPoses; ++poseId) {
    gauges.block(0, poseId * (d + 1), d, d) = rotations[poseId];
    gauges.block(0, poseId * (d + 1) + d, d, 1) = translations[poseId];
  }
  stats.objective = evaluateFrameGaugeObjective(d, gauges, measurements);
  const double communicatedDoubles =
      2.0 * static_cast<double>(measurements.size()) *
      (static_cast<double>(stats.rotationIters) * d * d +
       static_cast<double>(stats.translationIters) * d);
  stats.commMb = communicatedDoubles * sizeof(double) / (1024.0 * 1024.0);
  return gauges;
}

bool estimatePointCloudRotation(const vector<Matrix> &sourcePoints,
                                const vector<Matrix> &targetPoints,
                                unsigned d, Matrix &R) {
  if (sourcePoints.size() < d || sourcePoints.size() != targetPoints.size()) {
    return false;
  }

  Matrix sourceMean = Matrix::Zero(d, 1);
  Matrix targetMean = Matrix::Zero(d, 1);
  for (size_t i = 0; i < sourcePoints.size(); ++i) {
    sourceMean += sourcePoints[i];
    targetMean += targetPoints[i];
  }
  sourceMean /= static_cast<double>(sourcePoints.size());
  targetMean /= static_cast<double>(targetPoints.size());

  Matrix H = Matrix::Zero(d, d);
  double spread = 0.0;
  for (size_t i = 0; i < sourcePoints.size(); ++i) {
    const Matrix sourceCentered = sourcePoints[i] - sourceMean;
    const Matrix targetCentered = targetPoints[i] - targetMean;
    H += sourceCentered * targetCentered.transpose();
    spread += sourceCentered.squaredNorm() + targetCentered.squaredNorm();
  }
  if (spread < 1e-12) {
    return false;
  }

  Eigen::JacobiSVD<Matrix> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  R = svd.matrixV() * svd.matrixU().transpose();
  if (R.determinant() < 0) {
    Matrix V = svd.matrixV();
    V.col(d - 1) *= -1.0;
    R = V * svd.matrixU().transpose();
  }
  return true;
}

bool evaluateObjectFrameAlignment(
    const vector<Matrix> &sourceRotations,
    const vector<Matrix> &targetRotations,
    const vector<Matrix> &sourcePoints,
    const vector<Matrix> &targetPoints,
    const Matrix &R, unsigned d, FrameAlignment &alignment) {
  if (sourcePoints.empty() || sourcePoints.size() != targetPoints.size() ||
      sourceRotations.size() != sourcePoints.size() ||
      targetRotations.size() != sourcePoints.size()) {
    return false;
  }

  Matrix t = Matrix::Zero(d, 1);
  for (size_t i = 0; i < sourcePoints.size(); ++i) {
    t += targetPoints[i] - R * sourcePoints[i];
  }
  t /= static_cast<double>(sourcePoints.size());

  double translationResidualSq = 0.0;
  double rotationResidualSq = 0.0;
  for (size_t i = 0; i < sourcePoints.size(); ++i) {
    translationResidualSq +=
        (R * sourcePoints[i] + t - targetPoints[i]).squaredNorm();
    rotationResidualSq +=
        (R * sourceRotations[i] - targetRotations[i]).squaredNorm();
  }
  const double count = static_cast<double>(sourcePoints.size());
  alignment.R = R;
  alignment.t = t;
  alignment.commonObjects = sourcePoints.size();
  alignment.translationRms = std::sqrt(translationResidualSq / count);
  alignment.rotationRms = std::sqrt(rotationResidualSq / count);
  alignment.score = alignment.translationRms + 0.1 * alignment.rotationRms +
                    1e-4 / count;
  return true;
}

bool estimateObjectFrameAlignment(const Matrix &targetState,
                                  const ObjectPGORobotData &targetRobot,
                                  const vector<bool> &targetObserved,
                                  const Matrix &sourceState,
                                  const ObjectPGORobotData &sourceRobot,
                                  const vector<bool> &sourceObserved,
                                  size_t numObjects, unsigned d,
                                  size_t minCommonObjects,
                                  FrameAlignment &bestAlignment) {
  vector<Matrix> sourcePoints;
  vector<Matrix> targetPoints;
  vector<Matrix> sourceRotations;
  vector<Matrix> targetRotations;
  for (size_t objectId = 0; objectId < numObjects; ++objectId) {
    if (!targetObserved[objectId] || !sourceObserved[objectId]) {
      continue;
    }
    const size_t targetIdx = targetRobot.trajectoryVertexIds.size() + objectId;
    const size_t sourceIdx = sourceRobot.trajectoryVertexIds.size() + objectId;
    sourceRotations.push_back(
        sourceState.block(0, sourceIdx * (d + 1), d, d));
    targetRotations.push_back(
        targetState.block(0, targetIdx * (d + 1), d, d));
    sourcePoints.push_back(
        sourceState.block(0, sourceIdx * (d + 1) + d, d, 1));
    targetPoints.push_back(
        targetState.block(0, targetIdx * (d + 1) + d, d, 1));
  }
  if (sourcePoints.size() < minCommonObjects) {
    return false;
  }

  vector<Matrix> candidateRotations;
  Matrix rotationMoment = Matrix::Zero(d, d);
  for (size_t i = 0; i < sourceRotations.size(); ++i) {
    rotationMoment += targetRotations[i] * sourceRotations[i].transpose();
  }
  candidateRotations.push_back(projectToRotationGroup(rotationMoment));

  Matrix pointCloudRotation;
  if (estimatePointCloudRotation(sourcePoints, targetPoints, d,
                                 pointCloudRotation)) {
    candidateRotations.push_back(pointCloudRotation);
  }

  bool found = false;
  bestAlignment = FrameAlignment();
  for (const Matrix &R : candidateRotations) {
    FrameAlignment alignment;
    if (!evaluateObjectFrameAlignment(sourceRotations, targetRotations,
                                      sourcePoints, targetPoints, R, d,
                                      alignment)) {
      continue;
    }
    if (!found || alignment.score < bestAlignment.score) {
      bestAlignment = alignment;
      found = true;
    }
  }
  return found;
}

void applyFrameTransform(Matrix &state, const Matrix &R, const Matrix &t,
                         unsigned d) {
  const size_t numPoses = static_cast<size_t>(state.cols()) / (d + 1);
  for (size_t poseIdx = 0; poseIdx < numPoses; ++poseIdx) {
    Matrix poseR = state.block(0, poseIdx * (d + 1), d, d);
    Matrix poset = state.block(0, poseIdx * (d + 1) + d, d, 1);
    state.block(0, poseIdx * (d + 1), d, d) = R * poseR;
    state.block(0, poseIdx * (d + 1) + d, d, 1) = R * poset + t;
  }
}

FrameAlignment averageFrameAlignments(const vector<FrameAlignment> &alignments,
                                      unsigned d) {
  FrameAlignment average;
  if (alignments.empty()) {
    return average;
  }

  Matrix rotationMoment = Matrix::Zero(d, d);
  Matrix translation = Matrix::Zero(d, 1);
  double weightSum = 0.0;
  size_t commonObjects = 0;
  double translationRms = 0.0;
  double rotationRms = 0.0;
  double score = 0.0;
  for (const FrameAlignment &alignment : alignments) {
    const double weight = 1.0 / std::max(alignment.score, 1e-6);
    rotationMoment += weight * alignment.R;
    translation += weight * alignment.t;
    weightSum += weight;
    commonObjects += alignment.commonObjects;
    translationRms += alignment.translationRms;
    rotationRms += alignment.rotationRms;
    score += alignment.score;
  }

  average.R = projectToRotationGroup(rotationMoment);
  average.t = translation / weightSum;
  average.commonObjects = commonObjects;
  average.translationRms =
      translationRms / static_cast<double>(alignments.size());
  average.rotationRms = rotationRms / static_cast<double>(alignments.size());
  average.score = score / static_cast<double>(alignments.size());
  return average;
}

Matrix dampRotationCorrection(const Matrix &R, double alpha, unsigned d) {
  if (alpha >= 1.0) {
    return R;
  }
  if (alpha <= 0.0) {
    return Matrix::Identity(d, d);
  }
  return projectToRotationGroup((1.0 - alpha) * Matrix::Identity(d, d) +
                                alpha * R);
}

vector<RobotFrameEdge> buildObjectFrameEdges(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &neighbors, const vector<vector<bool>> &observed,
    unsigned d, const vector<Matrix> &states, size_t minCommonObjects,
    double maxEdgeWeight, const string &edgePolicy) {
  vector<RobotFrameEdge> edges;
  const size_t numRobots = dataset.robots.size();
  if (numRobots == 0 || dataset.numObjects == 0) {
    return edges;
  }
  map<EdgeKey, RobotFrameEdge> bestUndirectedEdges;

  for (size_t targetId = 0; targetId < numRobots; ++targetId) {
    if (targetId >= neighbors.size()) {
      continue;
    }
    for (const WeightedNeighbor &neighbor : neighbors[targetId]) {
      const size_t sourceId = neighbor.id;
      if (sourceId >= numRobots || sourceId == targetId) {
        continue;
      }
      FrameAlignment alignment;
      if (!estimateObjectFrameAlignment(
              states[targetId], dataset.robots[targetId], observed[targetId],
              states[sourceId], dataset.robots[sourceId], observed[sourceId],
              dataset.numObjects, d, minCommonObjects, alignment)) {
        continue;
      }
      const double confidence =
          std::max(0.0, neighbor.weight) *
          static_cast<double>(alignment.commonObjects) /
          std::max(alignment.score, 1e-6);
      RobotFrameEdge edge;
      edge.targetId = targetId;
      edge.sourceId = sourceId;
      edge.alignment = alignment;
      edge.weight = std::max(1e-6, std::min(maxEdgeWeight, confidence));
      if (edgePolicy == "best_undirected" || edgePolicy == "best") {
        const EdgeKey key = makeEdgeKey(targetId, sourceId);
        const auto bestIt = bestUndirectedEdges.find(key);
        if (bestIt == bestUndirectedEdges.end() ||
            edge.alignment.score < bestIt->second.alignment.score) {
          bestUndirectedEdges[key] = edge;
        }
        continue;
      }
      edges.push_back(std::move(edge));
    }
  }
  for (const auto &entry : bestUndirectedEdges) {
    edges.push_back(entry.second);
  }
  return edges;
}

FrameSyncRunStats synchronizeLocalChordalFramesWithGraph(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &neighbors, unsigned d,
    vector<Matrix> &states, const string &syncMode = "graph_sync",
    bool countRegistrationPayload = false) {
  FrameSyncRunStats runStats;
  runStats.mode = syncMode;
  const size_t numRobots = dataset.robots.size();
  runStats.numRobots = numRobots;
  if (numRobots == 0 || dataset.numObjects == 0) {
    return runStats;
  }

  vector<vector<bool>> observed;
  observed.reserve(numRobots);
  for (const ObjectPGORobotData &robot : dataset.robots) {
    observed.push_back(observedObjects(robot, dataset.numObjects));
  }

  const size_t minCommonObjects =
      readEnvSizeT("OBJECT_FRAME_SYNC_MIN_OBJECTS", d >= 3 ? 2 : 1);
  const double maxEdgeWeight =
      std::max(1e-6, readEnvDouble("OBJECT_FRAME_SYNC_MAX_EDGE_WEIGHT", 100.0));
  const string edgePolicy =
      readEnvString("OBJECT_FRAME_SYNC_EDGE_POLICY", "directed");
  vector<RobotFrameEdge> edges = buildObjectFrameEdges(
      dataset, neighbors, observed, d, states, minCommonObjects, maxEdgeWeight,
      edgePolicy);
  runStats.edgePolicy = edgePolicy;
  vector<size_t> candidateCommonObjectCounts;
  candidateCommonObjectCounts.reserve(edges.size());
  for (const RobotFrameEdge &edge : edges) {
    candidateCommonObjectCounts.push_back(edge.alignment.commonObjects);
  }
  if (countRegistrationPayload) {
    runStats.registrationCommMb =
        frameRegistrationPayloadMb(candidateCommonObjectCounts, d);
  }
  FrameEdgeSelectionStats edgeSelectionStats;
  const double cycleQuantile =
      readEnvDouble("OBJECT_FRAME_SYNC_CYCLE_QUANTILE", 0.99);
  const string edgeLogPath =
      readEnvString("OBJECT_FRAME_SYNC_EDGE_LOG", "");
  if (isCycleFrameEdgePolicy(edgePolicy)) {
    edges = selectCycleConsistentFrameEdges(
        edges, numRobots, d, cycleQuantile, edgeLogPath, edgeSelectionStats);
  } else {
    edgeSelectionStats.candidateEdges = edges.size();
  }
  runStats.edgeSelection = edgeSelectionStats;
  runStats.candidateEdges = edgeSelectionStats.candidateEdges;

  vector<vector<size_t>> adjacency(numRobots);
  for (size_t edgeId = 0; edgeId < edges.size(); ++edgeId) {
    const RobotFrameEdge &edge = edges[edgeId];
    adjacency[edge.targetId].push_back(edge.sourceId);
    adjacency[edge.sourceId].push_back(edge.targetId);
  }

  vector<bool> connected(numRobots, false);
  vector<size_t> frontier;
  frontier.push_back(0);
  connected[0] = true;
  for (size_t head = 0; head < frontier.size(); ++head) {
    const size_t robotId = frontier[head];
    for (size_t neighborId : adjacency[robotId]) {
      if (!connected[neighborId]) {
        connected[neighborId] = true;
        frontier.push_back(neighborId);
      }
    }
  }

  vector<size_t> connectedRobots;
  connectedRobots.reserve(numRobots);
  vector<int> compactIndex(numRobots, -1);
  for (size_t robotId = 0; robotId < numRobots; ++robotId) {
    if (!connected[robotId]) {
      continue;
    }
    compactIndex[robotId] = static_cast<int>(connectedRobots.size());
    connectedRobots.push_back(robotId);
  }
  runStats.connectedRobots = connectedRobots.size();

  if (connectedRobots.size() < 2) {
    cout << "Distributed chordal frame graph sync aligned "
         << connectedRobots.size() << "/" << numRobots
         << " robots using 0 object-registration edges"
         << " | min_common_objects = " << minCommonObjects;
    if (isCycleFrameEdgePolicy(edgePolicy)) {
      cout << " | cycle_candidate_edges = "
           << edgeSelectionStats.candidateEdges
           << " | cycle_tree_edges = " << edgeSelectionStats.treeEdges
           << " | cycle_tested_edges = "
           << edgeSelectionStats.testedCycleEdges
           << " | cycle_accepted_edges = "
           << edgeSelectionStats.acceptedCycleEdges
           << " | cycle_rejected_edges = "
           << edgeSelectionStats.rejectedCycleEdges
           << " | cycle_disconnected_edges = "
           << edgeSelectionStats.disconnectedCycleEdges
           << " | cycle_quantile = "
           << edgeSelectionStats.cycleQuantile
           << " | cycle_chi2_dof = "
           << edgeSelectionStats.cycleDof
           << " | cycle_chi2_threshold = "
           << edgeSelectionStats.cycleChi2Threshold;
    }
    cout << "." << endl;
    if (connectedRobots.size() < numRobots) {
      cerr << "Warning: distributed chordal frame graph sync could not align "
           << (numRobots - connectedRobots.size())
           << " robots; their local gauges are unchanged." << endl;
    }
    runStats.commMb = runStats.registrationCommMb;
    return runStats;
  }

  vector<RelativeSEMeasurement> gaugeMeasurements;
  gaugeMeasurements.reserve(edges.size());
  double scoreSum = 0.0;
  double translationRmsSum = 0.0;
  double rotationRmsSum = 0.0;
  double maxScore = 0.0;
  for (const RobotFrameEdge &edge : edges) {
    if (!connected[edge.targetId] || !connected[edge.sourceId]) {
      continue;
    }
    RelativeSEMeasurement m(
        0, 0, static_cast<size_t>(compactIndex[edge.targetId]),
        static_cast<size_t>(compactIndex[edge.sourceId]), edge.alignment.R,
        edge.alignment.t, edge.weight, edge.weight);
    m.weight = 1.0;
    gaugeMeasurements.push_back(std::move(m));
    scoreSum += edge.alignment.score;
    translationRmsSum += edge.alignment.translationRms;
    rotationRmsSum += edge.alignment.rotationRms;
    maxScore = std::max(maxScore, edge.alignment.score);
  }
  runStats.usedEdges = gaugeMeasurements.size();

  if (gaugeMeasurements.empty()) {
    cerr << "Warning: distributed chordal frame graph sync found a connected"
         << " component but no usable gauge measurements." << endl;
    runStats.commMb = runStats.registrationCommMb;
    return runStats;
  }

  Matrix gauges;
  FrameGaugeSolverStats solverStats;
  const string frameSolver =
      readEnvString("OBJECT_FRAME_SYNC_SOLVER", "centralized");
  const size_t jacobiIters =
      readEnvSizeT("OBJECT_FRAME_SYNC_JACOBI_ITERS", 500);
  const double jacobiTol =
      readEnvDouble("OBJECT_FRAME_SYNC_JACOBI_TOL", 1e-10);
  const bool jacobiProjectEachIter =
      readEnvBool("OBJECT_FRAME_SYNC_JACOBI_PROJECT_EACH_ITER", false);
  if (!isCentralizedFrameSolver(frameSolver) &&
      !isDistributedFrameSolver(frameSolver)) {
    throw runtime_error("Unsupported OBJECT_FRAME_SYNC_SOLVER: " + frameSolver);
  }
  try {
    if (isDistributedFrameSolver(frameSolver)) {
      gauges = solveFrameGaugeJacobi(
          d, connectedRobots.size(), gaugeMeasurements, jacobiIters,
          jacobiTol, jacobiProjectEachIter, solverStats);
    } else {
      solverStats.solver = "centralized";
      gauges = chordalInitialization(d, connectedRobots.size(),
                                     gaugeMeasurements);
      solverStats.objective =
          evaluateFrameGaugeObjective(d, gauges, gaugeMeasurements);
    }
  } catch (const std::exception &e) {
    cerr << "Warning: distributed chordal frame graph sync failed: "
         << e.what() << ". Local gauges are unchanged." << endl;
    runStats.solver = solverStats;
    runStats.solverCommMb = solverStats.commMb;
    runStats.commMb = runStats.registrationCommMb + runStats.solverCommMb;
    return runStats;
  }
  if (solverStats.solver == "distributed_jacobi" &&
      (!solverStats.rotationConverged || !solverStats.translationConverged)) {
    cerr << "Warning: distributed Jacobi frame solver reached its iteration "
         << "limit before convergence"
         << " | rotation_converged = "
         << (solverStats.rotationConverged ? "true" : "false")
         << " | translation_converged = "
         << (solverStats.translationConverged ? "true" : "false")
         << " | rotation_step_norm = " << solverStats.rotationStepNorm
         << " | translation_step_norm = "
         << solverStats.translationStepNorm << endl;
  }

  for (size_t compactId = 0; compactId < connectedRobots.size(); ++compactId) {
    const size_t robotId = connectedRobots[compactId];
    const Matrix R = gauges.block(0, compactId * (d + 1), d, d);
    const Matrix t = gauges.block(0, compactId * (d + 1) + d, d, 1);
    applyFrameTransform(states[robotId], R, t, d);
  }
  runStats.applied = true;
  runStats.solver = solverStats;
  runStats.solverCommMb = solverStats.commMb;
  runStats.commMb = runStats.registrationCommMb + runStats.solverCommMb;
  runStats.meanTranslationRms =
      translationRmsSum / static_cast<double>(gaugeMeasurements.size());
  runStats.meanRotationRms =
      rotationRmsSum / static_cast<double>(gaugeMeasurements.size());
  runStats.meanScore =
      scoreSum / static_cast<double>(gaugeMeasurements.size());
  runStats.maxScore = maxScore;

  cout << "Distributed chordal frame graph sync aligned "
       << connectedRobots.size() << "/" << numRobots
       << " robots using " << gaugeMeasurements.size()
       << " object-registration edges"
       << " | min_common_objects = " << minCommonObjects
       << " | mean_translation_rms = "
       << translationRmsSum / static_cast<double>(gaugeMeasurements.size())
       << " | mean_rotation_rms = "
       << rotationRmsSum / static_cast<double>(gaugeMeasurements.size())
       << " | mean_score = "
       << scoreSum / static_cast<double>(gaugeMeasurements.size())
       << " | max_score = " << maxScore
       << " | max_edge_weight = " << maxEdgeWeight
       << " | edge_policy = " << edgePolicy
       << " | frame_solver = " << solverStats.solver
       << " | frame_solver_rotation_iters = "
       << solverStats.rotationIters
       << " | frame_solver_translation_iters = "
       << solverStats.translationIters
       << " | frame_solver_rotation_converged = "
       << (solverStats.rotationConverged ? "true" : "false")
       << " | frame_solver_translation_converged = "
       << (solverStats.translationConverged ? "true" : "false")
       << " | frame_solver_rotation_step_norm = "
       << solverStats.rotationStepNorm
       << " | frame_solver_translation_step_norm = "
       << solverStats.translationStepNorm
       << " | frame_solver_objective = " << solverStats.objective
       << " | frame_sync_registration_comm_mb = "
       << runStats.registrationCommMb
       << " | frame_solver_comm_mb = " << solverStats.commMb
       << " | frame_sync_comm_mb = " << runStats.commMb
       << " | frame_sync_mode = " << runStats.mode;
  if (isCycleFrameEdgePolicy(edgePolicy)) {
    cout << " | cycle_candidate_edges = "
         << edgeSelectionStats.candidateEdges
         << " | cycle_tree_edges = " << edgeSelectionStats.treeEdges
         << " | cycle_tested_edges = "
         << edgeSelectionStats.testedCycleEdges
         << " | cycle_accepted_edges = "
         << edgeSelectionStats.acceptedCycleEdges
         << " | cycle_rejected_edges = "
         << edgeSelectionStats.rejectedCycleEdges
         << " | cycle_disconnected_edges = "
         << edgeSelectionStats.disconnectedCycleEdges
         << " | cycle_quantile = "
         << edgeSelectionStats.cycleQuantile
         << " | cycle_chi2_dof = "
         << edgeSelectionStats.cycleDof
         << " | cycle_chi2_threshold = "
         << edgeSelectionStats.cycleChi2Threshold
         << " | cycle_sigma_t_floor = "
         << edgeSelectionStats.translationSigmaFloor
         << " | cycle_sigma_r_floor = "
         << edgeSelectionStats.rotationSigmaFloor
         << " | cycle_mean_translation_residual = "
         << edgeSelectionStats.meanCycleTranslationResidual
         << " | cycle_mean_rotation_residual = "
         << edgeSelectionStats.meanCycleRotationResidual
         << " | cycle_max_translation_residual = "
         << edgeSelectionStats.maxCycleTranslationResidual
         << " | cycle_max_rotation_residual = "
         << edgeSelectionStats.maxCycleRotationResidual
         << " | cycle_mean_accepted_z2 = "
         << edgeSelectionStats.meanAcceptedZ2
         << " | cycle_max_accepted_z2 = "
         << edgeSelectionStats.maxAcceptedZ2
         << " | cycle_min_rejected_z2 = "
         << edgeSelectionStats.minRejectedZ2;
  }
  cout << "." << endl;
  if (connectedRobots.size() < numRobots) {
    cerr << "Warning: distributed chordal frame graph sync could not align "
         << (numRobots - connectedRobots.size())
         << " robots; their local gauges are unchanged." << endl;
  }
  return runStats;
}

void synchronizeLocalChordalFrames(const ObjectPGODirectoryData &dataset,
                                   const WeightedNeighborRound &neighbors,
                                   unsigned d, vector<Matrix> &states) {
  const size_t numRobots = dataset.robots.size();
  if (numRobots == 0 || dataset.numObjects == 0) {
    return;
  }

  vector<vector<bool>> observed;
  observed.reserve(numRobots);
  for (const ObjectPGORobotData &robot : dataset.robots) {
    observed.push_back(observedObjects(robot, dataset.numObjects));
  }

  const size_t minCommonObjects =
      readEnvSizeT("OBJECT_FRAME_SYNC_MIN_OBJECTS", d >= 3 ? 2 : 1);
  const size_t refineIters =
      readEnvSizeT("OBJECT_FRAME_SYNC_REFINE_ITERS", 0);
  const double refineAlpha =
      readEnvDouble("OBJECT_FRAME_SYNC_REFINE_ALPHA", 0.5);
  vector<bool> aligned(numRobots, false);
  aligned[0] = true;
  size_t syncedEdges = 0;
  double syncTranslationRmsSum = 0.0;
  double syncRotationRmsSum = 0.0;
  double syncMaxScore = 0.0;
  auto alignedCount = [&aligned]() {
    return static_cast<size_t>(
        std::count(aligned.begin(), aligned.end(), true));
  };
  while (alignedCount() < numRobots) {
    bool foundCandidate = false;
    size_t bestSourceId = 0;
    FrameAlignment bestAlignment;

    for (size_t targetId = 0; targetId < numRobots; ++targetId) {
      if (!aligned[targetId] || targetId >= neighbors.size()) {
        continue;
      }
      for (const WeightedNeighbor &neighbor : neighbors[targetId]) {
        const size_t sourceId = neighbor.id;
        if (sourceId >= numRobots || aligned[sourceId]) {
          continue;
        }
        FrameAlignment alignment;
        if (!estimateObjectFrameAlignment(
                states[targetId], dataset.robots[targetId], observed[targetId],
                states[sourceId], dataset.robots[sourceId], observed[sourceId],
                dataset.numObjects, d, minCommonObjects, alignment)) {
          continue;
        }
        if (!foundCandidate || alignment.score < bestAlignment.score ||
            (std::abs(alignment.score - bestAlignment.score) < 1e-12 &&
             alignment.commonObjects > bestAlignment.commonObjects)) {
          foundCandidate = true;
          bestSourceId = sourceId;
          bestAlignment = alignment;
        }
      }
    }

    if (!foundCandidate) {
      break;
    }

    applyFrameTransform(states[bestSourceId], bestAlignment.R, bestAlignment.t,
                        d);
    aligned[bestSourceId] = true;
    ++syncedEdges;
    syncTranslationRmsSum += bestAlignment.translationRms;
    syncRotationRmsSum += bestAlignment.rotationRms;
    syncMaxScore = std::max(syncMaxScore, bestAlignment.score);
  }

  // Try the reverse direction for any remaining robot whose outgoing edge list
  // is not symmetric in the schedule file.
  bool madeProgress = true;
  while (madeProgress && alignedCount() < numRobots) {
    madeProgress = false;
    for (size_t sourceId = 0; sourceId < numRobots; ++sourceId) {
      if (aligned[sourceId] || sourceId >= neighbors.size()) {
        continue;
      }
      bool foundCandidate = false;
      size_t bestTargetId = 0;
      FrameAlignment bestAlignment;
      for (const WeightedNeighbor &neighbor : neighbors[sourceId]) {
        const size_t targetId = neighbor.id;
        if (targetId >= numRobots || !aligned[targetId]) {
          continue;
        }
        FrameAlignment alignment;
        if (!estimateObjectFrameAlignment(
                states[targetId], dataset.robots[targetId], observed[targetId],
                states[sourceId], dataset.robots[sourceId], observed[sourceId],
                dataset.numObjects, d, minCommonObjects, alignment)) {
          continue;
        }
        if (!foundCandidate || alignment.score < bestAlignment.score) {
          foundCandidate = true;
          bestTargetId = targetId;
          bestAlignment = alignment;
        }
      }
      (void)bestTargetId;
      if (!foundCandidate) {
        continue;
      }
      aligned[sourceId] = true;
      applyFrameTransform(states[sourceId], bestAlignment.R, bestAlignment.t,
                          d);
      ++syncedEdges;
      syncTranslationRmsSum += bestAlignment.translationRms;
      syncRotationRmsSum += bestAlignment.rotationRms;
      syncMaxScore = std::max(syncMaxScore, bestAlignment.score);
      madeProgress = true;
    }
  }

  const size_t finalAlignedCount = alignedCount();
  cout << "Decentralized chordal frame sync aligned " << finalAlignedCount << "/"
       << numRobots << " robots using " << syncedEdges
       << " neighbor object-registration edges"
       << " | min_common_objects = " << minCommonObjects;
  if (syncedEdges > 0) {
    cout << " | mean_translation_rms = "
         << syncTranslationRmsSum / static_cast<double>(syncedEdges)
         << " | mean_rotation_rms = "
         << syncRotationRmsSum / static_cast<double>(syncedEdges)
         << " | max_score = " << syncMaxScore;
  }
  cout << "." << endl;
  if (finalAlignedCount < numRobots) {
    cerr << "Warning: decentralized chordal frame sync could not align "
         << (numRobots - finalAlignedCount)
         << " robots; their local gauges are unchanged." << endl;
  }

  if (finalAlignedCount < 2 || refineIters == 0) {
    return;
  }

  for (size_t refineIter = 0; refineIter < refineIters; ++refineIter) {
    const vector<Matrix> previousStates = states;
    vector<FrameAlignment> corrections(numRobots);
    vector<bool> hasCorrection(numRobots, false);
    double meanScore = 0.0;
    size_t correctedRobots = 0;

    for (size_t robotId = 1; robotId < numRobots; ++robotId) {
      if (!aligned[robotId] || robotId >= neighbors.size()) {
        continue;
      }

      vector<FrameAlignment> localCorrections;
      for (const WeightedNeighbor &neighbor : neighbors[robotId]) {
        const size_t neighborId = neighbor.id;
        if (neighborId >= numRobots || !aligned[neighborId]) {
          continue;
        }
        FrameAlignment correction;
        if (!estimateObjectFrameAlignment(
                previousStates[neighborId], dataset.robots[neighborId],
                observed[neighborId], previousStates[robotId],
                dataset.robots[robotId], observed[robotId], dataset.numObjects,
                d, minCommonObjects, correction)) {
          continue;
        }
        localCorrections.push_back(correction);
      }
      if (localCorrections.empty()) {
        continue;
      }

      FrameAlignment correction = averageFrameAlignments(localCorrections, d);
      correction.R = dampRotationCorrection(correction.R, refineAlpha, d);
      correction.t *= refineAlpha;
      corrections[robotId] = correction;
      hasCorrection[robotId] = true;
      meanScore += correction.score;
      ++correctedRobots;
    }

    for (size_t robotId = 1; robotId < numRobots; ++robotId) {
      if (!hasCorrection[robotId]) {
        continue;
      }
      applyFrameTransform(states[robotId], corrections[robotId].R,
                          corrections[robotId].t, d);
    }

    if (correctedRobots == 0) {
      break;
    }
    cout << "Decentralized chordal frame refine " << refineIter
         << " corrected " << correctedRobots
         << " robots | mean_score = "
         << meanScore / static_cast<double>(correctedRobots)
         << " | alpha = " << refineAlpha << "." << endl;
  }
}

struct ObjectSendState {
  vector<Matrix> lastSentPoses;
  vector<Matrix> lastSentAuxiliaries;
  vector<Matrix> lastSentGeodesicAuxiliaries;
  vector<size_t> ages;
  vector<size_t> geodesicAuxAges;
  vector<bool> initialized;
  vector<bool> geodesicAuxInitialized;
};

struct BoundaryBudgetCandidate {
  unsigned senderId{0};
  unsigned receiverId{0};
  size_t objectId{0};
  Matrix localObject;
  Matrix localAuxiliary;
  Matrix localGeodesicAuxiliary;
  bool sendPreCorrectionAuxiliary{false};
  bool sendObject{false};
  bool sendAuxiliary{false};
  bool sendGeodesicAuxiliary{false};
  bool forced{false};
  double score{0.0};
  double receiverDisagreement{0.0};
  double predictedGain{0.0};
  double predictiveStiffness{0.0};
};

struct ConsensusCorrectionStats {
  size_t updated{0};
  size_t rejected{0};
};

bool shouldSendObjectPose(const Matrix &currentPose, const Matrix &lastSentPose,
                          double poseTol, size_t age, size_t maxAge) {
  if (currentPose.rows() != lastSentPose.rows() ||
      currentPose.cols() != lastSentPose.cols()) {
    return true;
  }
  if ((currentPose - lastSentPose).norm() > poseTol) {
    return true;
  }
  return age >= maxAge;
}

Matrix computeLocalObjectGradient(
    const ObjectPGORobotData &robot, const SparseMatrix &baseQ,
    const Matrix &state, const vector<bool> &knownObjects, unsigned d) {
  Matrix gradient = Matrix::Zero(state.rows(), state.cols());
  if (state.cols() == 0 || baseQ.cols() == 0) {
    return gradient;
  }

  const Matrix ambientGradient = state * baseQ;
  const size_t objectOffset = robot.trajectoryVertexIds.size();
  const unsigned blockWidth = d + 1;
  for (size_t objectId = 0; objectId < knownObjects.size(); ++objectId) {
    if (!knownObjects[objectId]) {
      continue;
    }
    const size_t objectIndex = objectOffset + objectId;
    const size_t colOffset = objectIndex * blockWidth;
    if (colOffset + blockWidth > static_cast<size_t>(state.cols())) {
      continue;
    }
    const Matrix pose = blockPose(state, objectIndex, d);
    const Matrix ambientBlock =
        ambientGradient.block(0, colOffset, d, blockWidth);
    gradient.block(0, colOffset, d, blockWidth) =
        projectPoseGradient(pose, ambientBlock, d);
  }
  return gradient;
}

vector<Matrix> computeLocalObjectGradients(
    const ObjectPGODirectoryData &dataset, const vector<SparseMatrix> &baseQs,
    const vector<Matrix> &states, const vector<vector<bool>> &knownObjects,
    unsigned d) {
  vector<Matrix> gradients;
  gradients.reserve(states.size());
  for (size_t robotId = 0; robotId < states.size(); ++robotId) {
    gradients.push_back(computeLocalObjectGradient(
        dataset.robots[robotId], baseQs[robotId], states[robotId],
        knownObjects[robotId], d));
  }
  return gradients;
}

struct DistributedCorrectionStats {
  size_t updated{0};
  size_t rejected{0};
  size_t staleSkipped{0};
  double stepNorm{0.0};
  double auxiliaryNorm{0.0};
  double acceptedStepScaleSum{0.0};
};

struct GeodesicLocalSolveStats {
  double costInit{0.0};
  double costOpt{0.0};
  double gradNormInit{0.0};
  double gradNormOpt{0.0};
  size_t iterations{0};
  size_t acceptedSteps{0};
  size_t rejectedSteps{0};
};

GeodesicMeasurementTerm makeGeodesicMeasurementTerm(
    const RelativeSEMeasurement &m) {
  GeodesicMeasurementTerm term;
  term.firstPose = m.p1;
  term.secondPose = m.p2;
  term.relativeRotation = m.R;
  term.relativeTranslation = m.t;
  term.rotationalPrecision = m.kappa;
  term.translationalPrecision = m.tau;
  return term;
}

vector<GeodesicMeasurementTerm> makeGeodesicMeasurementTerms(
    const vector<RelativeSEMeasurement> &measurements,
    size_t localNumPoses) {
  vector<GeodesicMeasurementTerm> terms;
  terms.reserve(measurements.size());
  for (const RelativeSEMeasurement &m : measurements) {
    if (m.p1 >= localNumPoses || m.p2 >= localNumPoses) {
      continue;
    }
    terms.push_back(makeGeodesicMeasurementTerm(m));
  }
  return terms;
}

vector<GeodesicConsensusTerm> makeGeodesicConsensusTerms(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &knownObjects,
    const vector<vector<bool>> &observedObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborGeodesicAuxiliaries,
    size_t robotId, double beta, double observedConfidence,
    double relayConfidence, GeodesicAuxiliaryMode geodesicAuxMode,
    double geodesicAuxScale) {
  vector<GeodesicConsensusTerm> terms;
  if (robotId >= dataset.robots.size() || robotId >= activeNeighbors.size()) {
    return terms;
  }

  const ObjectPGORobotData &robot = dataset.robots[robotId];
  const size_t objectOffset = robot.trajectoryVertexIds.size();
  for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
    const unsigned neighborId = neighbor.id;
    if (neighborId >= dataset.robots.size() ||
        neighborId >= knownObjects.size()) {
      continue;
    }
    const auto cacheIt = neighborObjects[robotId].find(neighborId);
    if (cacheIt == neighborObjects[robotId].end()) {
      continue;
    }
    const vector<Matrix> &neighborCache = cacheIt->second;
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId] ||
          objectId >= knownObjects[neighborId].size() ||
          !knownObjects[neighborId][objectId] ||
          objectId >= neighborCache.size() ||
          neighborCache[objectId].size() == 0) {
        continue;
      }
      const double edgeWeight = effectiveConsensusWeight(
          neighbor.weight, robotId, neighborId, objectId, observedObjects,
          knownObjects, observedConfidence, relayConfidence);
      if (edgeWeight <= 0.0) {
        continue;
      }
      GeodesicConsensusTerm term;
      term.pose = objectOffset + objectId;
      term.targetPose = neighborCache[objectId];
      if (geodesicAuxMode ==
          GeodesicAuxiliaryMode::TangentResidualFeedback) {
        const auto auxIt = neighborGeodesicAuxiliaries[robotId].find(neighborId);
        if (auxIt != neighborGeodesicAuxiliaries[robotId].end() &&
            objectId < auxIt->second.size() &&
            auxIt->second[objectId].rows() == 6 &&
            auxIt->second[objectId].cols() == 1) {
          Eigen::Matrix<double, 6, 1> tangent;
          tangent = auxIt->second[objectId];
          term.targetPose =
              applySE3RightTangentStep(term.targetPose, tangent,
                                       -geodesicAuxScale);
        }
      }
      term.beta = beta;
      term.weight = edgeWeight;
      terms.push_back(std::move(term));
    }
  }
  return terms;
}

struct MainMeritFilterStats {
  size_t filteredUpdates{0};
  size_t rejectedUpdates{0};
  double acceptedStepScaleSum{0.0};
};

double computeGeodesicMeasurementError(const RelativeSEMeasurement &m,
                                       const Matrix &R1, const Matrix &t1,
                                       const Matrix &R2, const Matrix &t2) {
  const GeodesicSE3Residual residual =
      evaluateRelativeSE3LogResidual(R1, t1, m.R, m.t, R2, t2);
  return m.kappa * residual.rotation.squaredNorm() +
         m.tau * residual.translation.squaredNorm();
}

double computeGeodesicOriginalMeasurementCost(
    const ObjectPGORobotData &robot, const Matrix &state, unsigned d) {
  if (d != 3) {
    throw runtime_error("Geodesic object objective currently requires SE(3)");
  }
  double cost = 0.0;
  for (const RelativeSEMeasurement &m : robot.trajectoryMeasurements) {
    const Matrix pose1 = blockPose(state, m.p1, d);
    const Matrix pose2 = blockPose(state, m.p2, d);
    cost += computeGeodesicMeasurementError(
        m, pose1.leftCols(d), pose1.rightCols(1), pose2.leftCols(d),
        pose2.rightCols(1));
  }

  for (const ObjectObservationMeasurement &obs :
       robot.objectObservationMeasurements) {
    const RelativeSEMeasurement &m = obs.measurement;
    const Matrix pose1 = blockPose(state, m.p1, d);
    const Matrix pose2 = blockPose(state, m.p2, d);
    cost += computeGeodesicMeasurementError(
        m, pose1.leftCols(d), pose1.rightCols(1), pose2.leftCols(d),
        pose2.rightCols(1));
  }
  return cost;
}

void addGeodesicMeasurementTerms(const RelativeSEMeasurement &m,
                                 const Matrix &state, unsigned d,
                                 vector<Matrix> *rotationGradients,
                                 vector<Matrix> *translationGradients,
                                 double &cost) {
  const Matrix pose1 = blockPose(state, m.p1, d);
  const Matrix pose2 = blockPose(state, m.p2, d);
  const Matrix R1 = pose1.leftCols(d);
  const Matrix t1 = pose1.rightCols(1);
  const Matrix R2 = pose2.leftCols(d);
  const Matrix t2 = pose2.rightCols(1);
  const GeodesicSE3Residual residual =
      evaluateRelativeSE3LogResidual(R1, t1, m.R, m.t, R2, t2);
  cost += m.kappa * residual.rotation.squaredNorm() +
          m.tau * residual.translation.squaredNorm();

  if (rotationGradients == nullptr || translationGradients == nullptr) {
    return;
  }
  const GeodesicSE3Jacobians J =
      linearizeRelativeSE3LogResidual(R1, t1, m.R, m.t, R2, t2);
  Eigen::Matrix<double, 6, 1> weightedResidual;
  weightedResidual.head<3>() = m.kappa * residual.rotation;
  weightedResidual.tail<3>() = m.tau * residual.translation;
  (*rotationGradients)[m.p1] +=
      2.0 * J.Jr_i.transpose() * weightedResidual;
  (*translationGradients)[m.p1] +=
      2.0 * J.Jt_i.transpose() * weightedResidual;
  (*rotationGradients)[m.p2] +=
      2.0 * J.Jr_j.transpose() * weightedResidual;
  (*translationGradients)[m.p2] +=
      2.0 * J.Jt_j.transpose() * weightedResidual;
}

double evaluateGeodesicConsensusPairCost(const Matrix &localPose,
                                         const Matrix &targetPose,
                                         double beta, double weight,
                                         double &primalResidualNorm) {
  const GeodesicSE3Residual residual = evaluateRelativeSE3LogResidual(
      localPose.leftCols(3), localPose.rightCols(1),
      Matrix::Identity(3, 3), Matrix::Zero(3, 1),
      targetPose.leftCols(3), targetPose.rightCols(1));
  const double weightedSq =
      weight * weight *
      (residual.rotation.squaredNorm() +
       residual.translation.squaredNorm());
  primalResidualNorm = std::sqrt(weightedSq);
  return beta * weightedSq;
}

double poseBlockDeltaNorm(const Matrix &previousPose, const Matrix &currentPose,
                          ObjectLocalObjective objective) {
  if (objective == ObjectLocalObjective::Geodesic) {
    const Matrix rotationDelta =
        so3LogVector(previousPose.leftCols(3).transpose() *
                     currentPose.leftCols(3));
    const Matrix translationDelta =
        currentPose.rightCols(1) - previousPose.rightCols(1);
    return std::sqrt(rotationDelta.squaredNorm() +
                     translationDelta.squaredNorm());
  }
  return (currentPose - previousPose).norm();
}

double blockDiagonalMeanAbs(const SparseMatrix &matrix, size_t colOffset,
                            unsigned blockWidth) {
  if (matrix.rows() == 0 || matrix.cols() == 0 ||
      colOffset + blockWidth > static_cast<size_t>(matrix.rows()) ||
      colOffset + blockWidth > static_cast<size_t>(matrix.cols())) {
    return 0.0;
  }
  double diagSum = 0.0;
  for (unsigned c = 0; c < blockWidth; ++c) {
    diagSum += std::abs(matrix.coeff(colOffset + c, colOffset + c));
  }
  return diagSum / static_cast<double>(blockWidth);
}

double evaluateGeodesicLocalObjectiveAndGradient(
    const ObjectPGODirectoryData &dataset,
    const vector<RelativeSEMeasurement> &localMeasurements,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &knownObjects,
    const vector<vector<bool>> &observedObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborObjects,
    size_t robotId, double beta, double observedConfidence,
    double relayConfidence, unsigned d, const Matrix &state,
    vector<Matrix> *rotationGradients,
    vector<Matrix> *translationGradients) {
  if (d != 3) {
    throw runtime_error("Geodesic local objective currently requires SE(3)");
  }
  const ObjectPGORobotData &robot = dataset.robots[robotId];
  const size_t localNumPoses =
      robot.trajectoryVertexIds.size() + dataset.numObjects;
  if (rotationGradients != nullptr) {
    rotationGradients->assign(localNumPoses, Matrix::Zero(3, 1));
  }
  if (translationGradients != nullptr) {
    translationGradients->assign(localNumPoses, Matrix::Zero(3, 1));
  }

  double cost = 0.0;
  for (const RelativeSEMeasurement &m : localMeasurements) {
    if (m.p1 >= localNumPoses || m.p2 >= localNumPoses) {
      continue;
    }
    addGeodesicMeasurementTerms(m, state, d, rotationGradients,
                                translationGradients, cost);
  }

  if (robotId >= activeNeighbors.size()) {
    return cost;
  }
  const size_t objectOffset = robot.trajectoryVertexIds.size();
  for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
    const unsigned neighborId = neighbor.id;
    if (neighborId >= dataset.robots.size() ||
        neighborId >= knownObjects.size()) {
      continue;
    }
    const auto cacheIt = neighborObjects[robotId].find(neighborId);
    if (cacheIt == neighborObjects[robotId].end()) {
      continue;
    }
    const vector<Matrix> &neighborCache = cacheIt->second;
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId] ||
          objectId >= knownObjects[neighborId].size() ||
          !knownObjects[neighborId][objectId] ||
          objectId >= neighborCache.size() ||
          neighborCache[objectId].size() == 0) {
        continue;
      }
      const double edgeWeight = effectiveConsensusWeight(
          neighbor.weight, robotId, neighborId, objectId, observedObjects,
          knownObjects, observedConfidence, relayConfidence);
      if (edgeWeight <= 0.0) {
        continue;
      }
      const size_t localObjectIndex = objectOffset + objectId;
      const Matrix localPose = blockPose(state, localObjectIndex, d);
      const Matrix &targetPose = neighborCache[objectId];
      double pairPrimalResidual = 0.0;
      cost += evaluateGeodesicConsensusPairCost(
          localPose, targetPose, beta, edgeWeight, pairPrimalResidual);

      if (rotationGradients == nullptr || translationGradients == nullptr) {
        continue;
      }
      const GeodesicSE3Residual residual = evaluateRelativeSE3LogResidual(
          localPose.leftCols(3), localPose.rightCols(1),
          Matrix::Identity(3, 3), Matrix::Zero(3, 1),
          targetPose.leftCols(3), targetPose.rightCols(1));
      const GeodesicSE3Jacobians J = linearizeRelativeSE3LogResidual(
          localPose.leftCols(3), localPose.rightCols(1),
          Matrix::Identity(3, 3), Matrix::Zero(3, 1),
          targetPose.leftCols(3), targetPose.rightCols(1));
      const double consensusWeight = beta * edgeWeight * edgeWeight;
      Eigen::Matrix<double, 6, 1> weightedResidual;
      weightedResidual.head<3>() = consensusWeight * residual.rotation;
      weightedResidual.tail<3>() = consensusWeight * residual.translation;
      (*rotationGradients)[localObjectIndex] +=
          2.0 * J.Jr_i.transpose() * weightedResidual;
      (*translationGradients)[localObjectIndex] +=
          2.0 * J.Jt_i.transpose() * weightedResidual;
    }
  }
  return cost;
}

vector<Matrix> computeLocalObjectGeodesicTangentGradients(
    const ObjectPGODirectoryData &dataset,
    const vector<vector<RelativeSEMeasurement>> &localMeasurements,
    const vector<Matrix> &states,
    const vector<vector<bool>> &knownObjects,
    const vector<vector<bool>> &observedObjects, unsigned d) {
  vector<Matrix> objectGradients;
  objectGradients.reserve(dataset.robots.size());
  if (d != 3) {
    return objectGradients;
  }

  const WeightedNeighborRound emptyNeighbors;
  const vector<map<unsigned, vector<Matrix>>> emptyNeighborObjects;
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    Matrix tangentGradients = Matrix::Zero(6, dataset.numObjects);
    if (robotId >= states.size() || robotId >= localMeasurements.size() ||
        robotId >= knownObjects.size()) {
      objectGradients.push_back(std::move(tangentGradients));
      continue;
    }

    vector<Matrix> rotationGradients;
    vector<Matrix> translationGradients;
    evaluateGeodesicLocalObjectiveAndGradient(
        dataset, localMeasurements[robotId], emptyNeighbors, knownObjects,
        observedObjects, emptyNeighborObjects, robotId, 0.0, 1.0, 1.0, d,
        states[robotId], &rotationGradients, &translationGradients);

    const size_t objectOffset =
        dataset.robots[robotId].trajectoryVertexIds.size();
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId]) {
        continue;
      }
      const size_t objectIndex = objectOffset + objectId;
      if (objectIndex >= rotationGradients.size() ||
          objectIndex >= translationGradients.size()) {
        continue;
      }
      tangentGradients.block(0, objectId, 3, 1) =
          rotationGradients[objectIndex];
      const Matrix objectPose = blockPose(states[robotId], objectIndex, d);
      tangentGradients.block(3, objectId, 3, 1) =
          objectPose.leftCols(3).transpose() *
          translationGradients[objectIndex];
    }
    objectGradients.push_back(std::move(tangentGradients));
  }
  return objectGradients;
}

Matrix applyGeodesicGradientStep(const Matrix &state,
                                 const vector<Matrix> &rotationGradients,
                                 const vector<Matrix> &translationGradients,
                                 double step, unsigned d) {
  Matrix updated = state;
  const size_t localNumPoses = rotationGradients.size();
  for (size_t localIdx = 0; localIdx < localNumPoses; ++localIdx) {
    const size_t colOffset = localIdx * (d + 1);
    const Matrix R = state.block(0, colOffset, d, d);
    updated.block(0, colOffset, d, d) =
        R * so3ExpVector(-step * rotationGradients[localIdx]);
    updated.block(0, colOffset + d, d, 1) =
        state.block(0, colOffset + d, d, 1) -
        step * translationGradients[localIdx];
  }
  return updated;
}

GeodesicLocalSolveStats optimizeGeodesicLocalProblem(
    const ObjectPGODirectoryData &dataset,
    const vector<RelativeSEMeasurement> &localMeasurements,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &knownObjects,
    const vector<vector<bool>> &observedObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborObjects,
    size_t robotId, double beta, double observedConfidence,
    double relayConfidence, unsigned d, size_t maxIterations,
    double tolerance, Matrix &state) {
  GeodesicLocalSolveStats stats;
  const double initialStep =
      readEnvDouble("OBJECT_GEODESIC_STEP", 0.25);
  const size_t maxBacktracking =
      readEnvSizeT("OBJECT_GEODESIC_BACKTRACKING_STEPS", 8);
  const double meritTolerance =
      std::max(0.0, readEnvDouble("OBJECT_GEODESIC_MERIT_TOL", 1e-12));

  vector<Matrix> rotationGradients;
  vector<Matrix> translationGradients;
  double cost = evaluateGeodesicLocalObjectiveAndGradient(
      dataset, localMeasurements, activeNeighbors, knownObjects,
      observedObjects, neighborObjects, robotId, beta, observedConfidence,
      relayConfidence, d, state, &rotationGradients, &translationGradients);
  stats.costInit = cost;
  auto gradientNorm = [&]() {
    double gradSq = 0.0;
    for (size_t i = 0; i < rotationGradients.size(); ++i) {
      gradSq += rotationGradients[i].squaredNorm() +
                translationGradients[i].squaredNorm();
    }
    return std::sqrt(gradSq);
  };
  stats.gradNormInit = gradientNorm();
  stats.gradNormOpt = stats.gradNormInit;

  for (size_t iter = 0; iter < maxIterations; ++iter) {
    if (stats.gradNormOpt <= tolerance) {
      break;
    }
    double step = initialStep;
    bool accepted = false;
    Matrix acceptedState = state;
    double acceptedCost = cost;
    for (size_t attempt = 0; attempt <= maxBacktracking; ++attempt) {
      Matrix candidate = applyGeodesicGradientStep(
          state, rotationGradients, translationGradients, step, d);
      const double candidateCost = evaluateGeodesicLocalObjectiveAndGradient(
          dataset, localMeasurements, activeNeighbors, knownObjects,
          observedObjects, neighborObjects, robotId, beta, observedConfidence,
          relayConfidence, d, candidate, nullptr, nullptr);
      if (candidateCost <= cost + meritTolerance) {
        accepted = true;
        acceptedState = std::move(candidate);
        acceptedCost = candidateCost;
        break;
      }
      step *= 0.5;
      ++stats.rejectedSteps;
    }
    if (!accepted) {
      break;
    }
    state = std::move(acceptedState);
    cost = acceptedCost;
    ++stats.acceptedSteps;
    stats.iterations = iter + 1;
    cost = evaluateGeodesicLocalObjectiveAndGradient(
        dataset, localMeasurements, activeNeighbors, knownObjects,
        observedObjects, neighborObjects, robotId, beta, observedConfidence,
        relayConfidence, d, state, &rotationGradients,
        &translationGradients);
    stats.gradNormOpt = gradientNorm();
  }
  stats.costOpt = cost;
  return stats;
}

double computeRobotMeasurementCost(const ObjectPGORobotData &robot,
                                   const Matrix &state, unsigned d,
                                   ObjectLocalObjective objective) {
  if (objective == ObjectLocalObjective::Geodesic) {
    return computeGeodesicOriginalMeasurementCost(robot, state, d);
  }
  return computeOriginalMeasurementCost(robot, state, d);
}

Matrix interpolatePoseBlock(const Matrix &previousPose, const Matrix &trialPose,
                            double alpha, unsigned d,
                            ObjectLocalObjective objective) {
  const double clampedAlpha = std::max(0.0, std::min(1.0, alpha));
  if (objective == ObjectLocalObjective::Geodesic && d == 3) {
    Matrix updated = previousPose;
    const Matrix delta =
        so3LogVector(previousPose.leftCols(3).transpose() *
                     trialPose.leftCols(3));
    updated.leftCols(3) =
        previousPose.leftCols(3) * so3ExpVector(clampedAlpha * delta);
    updated.rightCols(1) =
        (1.0 - clampedAlpha) * previousPose.rightCols(1) +
        clampedAlpha * trialPose.rightCols(1);
    return updated;
  }
  return dampedPoseUpdate(previousPose, trialPose, clampedAlpha, d);
}

Matrix interpolateStateBlocks(const Matrix &previousState,
                              const Matrix &trialState, double alpha,
                              unsigned d, ObjectLocalObjective objective) {
  if (previousState.rows() != trialState.rows() ||
      previousState.cols() != trialState.cols()) {
    return trialState;
  }
  Matrix candidate = previousState;
  const size_t blockWidth = d + 1;
  const size_t numBlocks =
      static_cast<size_t>(previousState.cols()) / blockWidth;
  for (size_t blockId = 0; blockId < numBlocks; ++blockId) {
    const size_t colOffset = blockId * blockWidth;
    const Matrix previousPose =
        previousState.block(0, colOffset, d, blockWidth);
    const Matrix trialPose = trialState.block(0, colOffset, d, blockWidth);
    candidate.block(0, colOffset, d, blockWidth) =
        interpolatePoseBlock(previousPose, trialPose, alpha, d, objective);
  }
  return candidate;
}

MainMeritFilterStats applyMainMeasurementMeritFilter(
    const ObjectPGORobotData &robot, const Matrix &previousState,
    Matrix &trialState, ObjectLocalObjective objective, unsigned d,
    size_t backtrackingSteps, double meritTolerance) {
  MainMeritFilterStats stats;
  const double baseCost =
      computeRobotMeasurementCost(robot, previousState, d, objective);
  const double trialCost =
      computeRobotMeasurementCost(robot, trialState, d, objective);
  if (std::isfinite(trialCost) &&
      trialCost <= baseCost + meritTolerance) {
    return stats;
  }

  ++stats.filteredUpdates;
  double alpha = 0.5;
  bool accepted = false;
  Matrix acceptedState = previousState;
  for (size_t attempt = 0; attempt < backtrackingSteps; ++attempt) {
    Matrix candidate =
        interpolateStateBlocks(previousState, trialState, alpha, d, objective);
    const double candidateCost =
        computeRobotMeasurementCost(robot, candidate, d, objective);
    if (std::isfinite(candidateCost) &&
        candidateCost <= baseCost + meritTolerance) {
      accepted = true;
      acceptedState = std::move(candidate);
      stats.acceptedStepScaleSum += alpha;
      break;
    }
    alpha *= 0.5;
  }

  if (accepted) {
    trialState = std::move(acceptedState);
  } else {
    trialState = previousState;
    ++stats.rejectedUpdates;
  }
  return stats;
}

MainMeritFilterStats applyGeodesicLocalModelMeritFilter(
    const Matrix &previousState, Matrix &trialState,
    const GeodesicObjectProblem &problem, unsigned d,
    size_t backtrackingSteps, double meritTolerance) {
  MainMeritFilterStats stats;
  const double baseCost = problem.f(previousState);
  const double trialCost = problem.f(trialState);
  if (std::isfinite(trialCost) && trialCost <= baseCost + meritTolerance) {
    return stats;
  }

  ++stats.filteredUpdates;
  double alpha = 0.5;
  bool accepted = false;
  Matrix acceptedState = previousState;
  for (size_t attempt = 0; attempt < backtrackingSteps; ++attempt) {
    Matrix candidate = interpolateStateBlocks(
        previousState, trialState, alpha, d, ObjectLocalObjective::Geodesic);
    const double candidateCost = problem.f(candidate);
    if (std::isfinite(candidateCost) &&
        candidateCost <= baseCost + meritTolerance) {
      accepted = true;
      acceptedState = std::move(candidate);
      stats.acceptedStepScaleSum += alpha;
      break;
    }
    alpha *= 0.5;
  }

  if (accepted) {
    trialState = std::move(acceptedState);
  } else {
    trialState = previousState;
    ++stats.rejectedUpdates;
  }
  return stats;
}

size_t countActiveKnownObjectPairs(
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &knownObjects,
    const vector<vector<bool>> &observedObjects,
    double observedConfidence, double relayConfidence) {
  size_t count = 0;
  for (size_t robotId = 0; robotId < activeNeighbors.size(); ++robotId) {
    for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
      const unsigned neighborId = neighbor.id;
      if (neighborId >= knownObjects.size()) {
        continue;
      }
      for (size_t objectId = 0; objectId < knownObjects[robotId].size();
           ++objectId) {
        if (!knownObjects[robotId][objectId] ||
            objectId >= knownObjects[neighborId].size() ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        const double edgeWeight = effectiveConsensusWeight(
            neighbor.weight, robotId, neighborId, objectId, observedObjects,
            knownObjects, observedConfidence, relayConfidence);
        if (edgeWeight > 0.0) {
          ++count;
        }
      }
    }
  }
  return count;
}

struct ObjectCopyConsensusLink {
  unsigned neighborId{0};
  double kappa{1.0};
  double tau{1.0};
};

using ObjectCopyConsensusGraph =
    vector<vector<vector<ObjectCopyConsensusLink>>>;

pair<double, double> medianObjectAwareMeasurementWeights(
    const ObjectPGODirectoryData &dataset) {
  vector<double> kappas;
  vector<double> taus;
  for (const ObjectPGORobotData &robot : dataset.robots) {
    for (const RelativeSEMeasurement &m : robot.trajectoryMeasurements) {
      if (m.kappa > 0.0) {
        kappas.push_back(m.kappa);
      }
      if (m.tau > 0.0) {
        taus.push_back(m.tau);
      }
    }
    for (const ObjectObservationMeasurement &obs :
         robot.objectObservationMeasurements) {
      if (obs.measurement.kappa > 0.0) {
        kappas.push_back(obs.measurement.kappa);
      }
      if (obs.measurement.tau > 0.0) {
        taus.push_back(obs.measurement.tau);
      }
    }
  }
  return std::make_pair(medianValue(kappas, 1.0), medianValue(taus, 1.0));
}

ObjectCopyConsensusGraph buildObjectCopyConsensusGraph(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &neighbors,
    const vector<vector<bool>> &knownObjects,
    const vector<vector<bool>> &observedObjects,
    double observedConfidence, double relayConfidence,
    double consensusKappa, double consensusTau,
    size_t &undirectedConsensusEdges) {
  const size_t numRobots = dataset.robots.size();
  ObjectCopyConsensusGraph graph(
      numRobots, vector<vector<ObjectCopyConsensusLink>>(dataset.numObjects));
  undirectedConsensusEdges = 0;
  map<tuple<unsigned, unsigned, size_t>, bool> processedEdges;
  for (unsigned robotId = 0; robotId < neighbors.size(); ++robotId) {
    for (const WeightedNeighbor &neighbor : neighbors[robotId]) {
      const unsigned neighborId = neighbor.id;
      if (neighborId >= numRobots || robotId == neighborId) {
        continue;
      }
      const unsigned robotA = std::min(robotId, neighborId);
      const unsigned robotB = std::max(robotId, neighborId);
      const WeightedNeighbor *reverseNeighbor =
          neighborId < neighbors.size()
              ? findWeightedNeighbor(neighbors[neighborId], robotId)
              : nullptr;
      for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
        const tuple<unsigned, unsigned, size_t> edgeKey(
            robotA, robotB, objectId);
        if (processedEdges[edgeKey]) {
          continue;
        }
        processedEdges[edgeKey] = true;
        if (!knownObjects[robotId][objectId] ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        const double forwardWeight = effectiveConsensusWeight(
            neighbor.weight, robotId, neighborId, objectId, observedObjects,
            knownObjects, observedConfidence, relayConfidence);
        const double reverseWeight =
            reverseNeighbor == nullptr
                ? forwardWeight
                : effectiveConsensusWeight(
                      reverseNeighbor->weight, neighborId, robotId, objectId,
                      observedObjects, knownObjects, observedConfidence,
                      relayConfidence);
        const double edgeWeight =
            0.5 * (std::max(0.0, forwardWeight) +
                   std::max(0.0, reverseWeight));
        if (edgeWeight <= 0.0) {
          continue;
        }
        const double kappa = consensusKappa * edgeWeight;
        const double tau = consensusTau * edgeWeight;
        if (kappa <= 0.0 && tau <= 0.0) {
          continue;
        }
        graph[robotId][objectId].push_back(
            ObjectCopyConsensusLink{neighborId, kappa, tau});
        graph[neighborId][objectId].push_back(
            ObjectCopyConsensusLink{robotId, kappa, tau});
        ++undirectedConsensusEdges;
      }
    }
  }
  return graph;
}

vector<vector<bool>> computeObjectCopyConnectedVariables(
    const ObjectPGODirectoryData &dataset,
    const vector<vector<RelativeSEMeasurement>> &localMeasurements,
    const ObjectCopyConsensusGraph &consensusGraph) {
  const size_t numRobots = dataset.robots.size();
  vector<vector<bool>> connected(numRobots);
  for (size_t robotId = 0; robotId < numRobots; ++robotId) {
    connected[robotId].assign(
        dataset.robots[robotId].trajectoryVertexIds.size() +
            dataset.numObjects,
        false);
  }
  if (numRobots == 0 || connected[0].empty()) {
    return connected;
  }

  vector<pair<size_t, size_t>> frontier;
  connected[0][0] = true;
  frontier.emplace_back(0, 0);
  for (size_t head = 0; head < frontier.size(); ++head) {
    const size_t robotId = frontier[head].first;
    const size_t localIdx = frontier[head].second;

    for (const RelativeSEMeasurement &m : localMeasurements[robotId]) {
      size_t nextIdx = std::numeric_limits<size_t>::max();
      if (m.p1 == localIdx) {
        nextIdx = m.p2;
      } else if (m.p2 == localIdx) {
        nextIdx = m.p1;
      }
      if (nextIdx < connected[robotId].size() &&
          !connected[robotId][nextIdx]) {
        connected[robotId][nextIdx] = true;
        frontier.emplace_back(robotId, nextIdx);
      }
    }

    const size_t objectOffset =
        dataset.robots[robotId].trajectoryVertexIds.size();
    if (localIdx < objectOffset) {
      continue;
    }
    const size_t objectId = localIdx - objectOffset;
    if (objectId >= dataset.numObjects ||
        robotId >= consensusGraph.size()) {
      continue;
    }
    for (const ObjectCopyConsensusLink &link :
         consensusGraph[robotId][objectId]) {
      const size_t neighborId = link.neighborId;
      if (neighborId >= numRobots) {
        continue;
      }
      const size_t neighborIdx =
          dataset.robots[neighborId].trajectoryVertexIds.size() + objectId;
      if (neighborIdx < connected[neighborId].size() &&
          !connected[neighborId][neighborIdx]) {
        connected[neighborId][neighborIdx] = true;
        frontier.emplace_back(neighborId, neighborIdx);
      }
    }
  }
  return connected;
}

double evaluateObjectCopyChordalObjective(
    const ObjectPGODirectoryData &dataset,
    const vector<vector<RelativeSEMeasurement>> &localMeasurements,
    const ObjectCopyConsensusGraph &consensusGraph,
    const vector<vector<Matrix>> &rotations,
    const vector<vector<Matrix>> &translations,
    double &localObjective, double &objectLocalObjective,
    double &fixedTrajectoryObjective, double &consensusObjective) {
  localObjective = 0.0;
  objectLocalObjective = 0.0;
  fixedTrajectoryObjective = 0.0;
  consensusObjective = 0.0;
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const size_t objectOffset =
        dataset.robots[robotId].trajectoryVertexIds.size();
    for (const RelativeSEMeasurement &m : localMeasurements[robotId]) {
      if (m.p1 >= rotations[robotId].size() ||
          m.p2 >= rotations[robotId].size()) {
        continue;
      }
      const double edgeObjective =
          m.kappa *
          (rotations[robotId][m.p1] * m.R -
           rotations[robotId][m.p2])
              .squaredNorm() +
          m.tau *
          (translations[robotId][m.p2] -
           translations[robotId][m.p1] -
           rotations[robotId][m.p1] * m.t)
              .squaredNorm();
      localObjective += edgeObjective;
      if (m.p1 >= objectOffset || m.p2 >= objectOffset) {
        objectLocalObjective += edgeObjective;
      } else {
        fixedTrajectoryObjective += edgeObjective;
      }
    }
  }

  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const size_t objectOffset =
        dataset.robots[robotId].trajectoryVertexIds.size();
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      const size_t localIdx = objectOffset + objectId;
      if (localIdx >= rotations[robotId].size()) {
        continue;
      }
      for (const ObjectCopyConsensusLink &link :
           consensusGraph[robotId][objectId]) {
        const size_t neighborId = link.neighborId;
        if (robotId >= neighborId || neighborId >= dataset.robots.size()) {
          continue;
        }
        const size_t neighborIdx =
            dataset.robots[neighborId].trajectoryVertexIds.size() + objectId;
        if (neighborIdx >= rotations[neighborId].size()) {
          continue;
        }
        consensusObjective +=
            link.kappa *
            (rotations[robotId][localIdx] -
             rotations[neighborId][neighborIdx])
                .squaredNorm();
        consensusObjective +=
            link.tau *
            (translations[neighborId][neighborIdx] -
             translations[robotId][localIdx])
                .squaredNorm();
      }
    }
  }
  return localObjective + consensusObjective;
}

ObjectCopyChordalJacobiStats refineObjectCopiesWithDistributedChordalJacobi(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &neighbors,
    const vector<vector<bool>> &observedObjects,
    const vector<vector<bool>> &knownObjects,
    double observedConfidence, double relayConfidence,
    unsigned d, vector<Matrix> &states) {
  ObjectCopyChordalJacobiStats stats;
  if (dataset.robots.empty() || states.empty()) {
    return stats;
  }
  if (states.size() < dataset.robots.size()) {
    throw runtime_error("Object-copy Jacobi initialization received fewer "
                        "state blocks than robots");
  }

  const size_t maxIters = readEnvSizeT(
      "OBJECT_INIT_OBJECT_JACOBI_ITERS",
      readEnvSizeT("OBJECT_INIT_JACOBI_ITERS", 20));
  const double tolerance = readEnvDouble(
      "OBJECT_INIT_OBJECT_JACOBI_TOL",
      readEnvDouble("OBJECT_INIT_JACOBI_TOL", 1e-8));
  const bool projectEachIter = readEnvBool(
      "OBJECT_INIT_OBJECT_JACOBI_PROJECT_EACH_ITER",
      readEnvBool("OBJECT_INIT_JACOBI_PROJECT_EACH_ITER", false));
  const double relaxation = std::max(
      1e-6,
      std::min(1.0, readEnvDouble("OBJECT_INIT_OBJECT_JACOBI_RELAXATION",
                                  readEnvDouble("OBJECT_INIT_JACOBI_RELAXATION",
                                                1.0))));
  const bool adaptiveBudget =
      readEnvBool("OBJECT_INIT_OBJECT_JACOBI_ADAPTIVE_BUDGET",
                  readEnvBool(
                      "OBJECT_INIT_OBJECT_JACOBI_ADAPTIVE_EARLY_STOP",
                      false));
  const size_t adaptiveMinIters =
      readEnvSizeT("OBJECT_INIT_OBJECT_JACOBI_MIN_ITERS",
                   adaptiveBudget ? 1 : 0);
  const double adaptiveStepTol =
      std::max(0.0, readEnvDouble(
                        "OBJECT_INIT_OBJECT_JACOBI_BUDGET_STEP_TOL",
                        readEnvDouble(
                            "OBJECT_INIT_OBJECT_JACOBI_STEP_TOL", 0.0)));
  const double adaptiveRelativeStepDropTol =
      std::max(0.0, readEnvDouble(
                        "OBJECT_INIT_OBJECT_JACOBI_RELATIVE_STEP_DROP_TOL",
                        readEnvDouble(
                            "OBJECT_INIT_OBJECT_JACOBI_REL_STEP_TOL", 0.0)));
  const double consensusScale = std::max(
      0.0,
      readEnvDouble("OBJECT_INIT_OBJECT_JACOBI_CONSENSUS_SCALE",
                    readEnvDouble("OBJECT_INIT_CONSENSUS_WEIGHT_SCALE", 1.0)));
  stats.adaptiveBudget = adaptiveBudget;
  stats.adaptiveMinIters = adaptiveMinIters;
  stats.adaptiveStepTol = adaptiveStepTol;
  stats.adaptiveRelativeStepDropTol = adaptiveRelativeStepDropTol;
  const string scopeText =
      readEnvString("OBJECT_INIT_OBJECT_JACOBI_SCOPE", "all");
  bool updateTrajectoryVariables = true;
  if (scopeText == "all" || scopeText == "full" ||
      scopeText == "all_variables") {
    stats.scope = "all";
  } else if (scopeText == "objects" || scopeText == "object" ||
             scopeText == "objects_only" || scopeText == "object_only") {
    updateTrajectoryVariables = false;
    stats.scope = "objects_only";
  } else {
    throw runtime_error("Unsupported OBJECT_INIT_OBJECT_JACOBI_SCOPE: " +
                        scopeText);
  }
  const pair<double, double> medianWeights =
      medianObjectAwareMeasurementWeights(dataset);
  stats.consensusKappa = medianWeights.first * consensusScale;
  stats.consensusTau = medianWeights.second * consensusScale;

  vector<vector<RelativeSEMeasurement>> localMeasurements;
  localMeasurements.reserve(dataset.robots.size());
  for (const ObjectPGORobotData &robot : dataset.robots) {
    localMeasurements.push_back(objectAwareLocalMeasurements(robot));
  }

  size_t undirectedConsensusEdges = 0;
  ObjectCopyConsensusGraph consensusGraph = buildObjectCopyConsensusGraph(
      dataset, neighbors, knownObjects, observedObjects, observedConfidence,
      relayConfidence, stats.consensusKappa, stats.consensusTau,
      undirectedConsensusEdges);
  vector<vector<bool>> connected = computeObjectCopyConnectedVariables(
      dataset, localMeasurements, consensusGraph);

  vector<vector<Matrix>> rotations(dataset.robots.size());
  vector<vector<Matrix>> translations(dataset.robots.size());
  size_t connectedVariables = 0;
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const size_t localNumPoses =
        dataset.robots[robotId].trajectoryVertexIds.size() +
        dataset.numObjects;
    rotations[robotId].resize(localNumPoses);
    translations[robotId].resize(localNumPoses);
    for (size_t localIdx = 0; localIdx < localNumPoses; ++localIdx) {
      rotations[robotId][localIdx] =
          projectToRotationGroup(
              states[robotId].block(0, localIdx * (d + 1), d, d));
      translations[robotId][localIdx] =
          states[robotId].block(0, localIdx * (d + 1) + d, d, 1);
      if (connected[robotId][localIdx]) {
        ++connectedVariables;
      }
    }
  }
  const Matrix rootRotation =
      updateTrajectoryVariables ? Matrix::Identity(d, d) : rotations[0][0];
  const Matrix rootTranslation =
      updateTrajectoryVariables ? Matrix::Zero(d, 1) : translations[0][0];
  rotations[0][0] = rootRotation;
  translations[0][0] = rootTranslation;

  auto isRoot = [](size_t robotId, size_t localIdx) {
    return robotId == 0 && localIdx == 0;
  };
  auto isObjectVariable = [&dataset](size_t robotId, size_t localIdx) {
    return robotId < dataset.robots.size() &&
           localIdx >= dataset.robots[robotId].trajectoryVertexIds.size();
  };
  auto isKnownObjectVariable = [&](size_t robotId, size_t localIdx) {
    if (!isObjectVariable(robotId, localIdx) ||
        robotId >= knownObjects.size()) {
      return false;
    }
    const size_t objectId =
        localIdx - dataset.robots[robotId].trajectoryVertexIds.size();
    return objectId < knownObjects[robotId].size() &&
           knownObjects[robotId][objectId];
  };
  auto shouldUpdateVariable = [&](size_t robotId, size_t localIdx) {
    if (robotId >= connected.size() ||
        localIdx >= connected[robotId].size() ||
        !connected[robotId][localIdx] || isRoot(robotId, localIdx)) {
      return false;
    }
    return updateTrajectoryVariables ||
           isKnownObjectVariable(robotId, localIdx);
  };
  auto shouldWriteVariable = [&](size_t robotId, size_t localIdx) {
    if (robotId >= connected.size() ||
        localIdx >= connected[robotId].size() ||
        !connected[robotId][localIdx]) {
      return false;
    }
    return updateTrajectoryVariables ||
           isKnownObjectVariable(robotId, localIdx);
  };
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    for (size_t localIdx = 0; localIdx < rotations[robotId].size();
         ++localIdx) {
      if (shouldUpdateVariable(robotId, localIdx)) {
        ++stats.updatedVariables;
      }
    }
  }

  if (maxIters == 0 || consensusScale == 0.0) {
    stats.rotationConverged = true;
    stats.translationConverged = true;
    stats.rotationStopReason = "disabled";
    stats.translationStopReason = "disabled";
  }

  auto adaptiveStopReason =
      [&](size_t iterCount, double previousStepNorm,
          double currentStepNorm) -> string {
    if (!adaptiveBudget || iterCount < adaptiveMinIters) {
      return "";
    }
    if (adaptiveStepTol > 0.0 && currentStepNorm <= adaptiveStepTol) {
      return "adaptive_step_tol";
    }
    if (adaptiveRelativeStepDropTol > 0.0 &&
        std::isfinite(previousStepNorm) && previousStepNorm > 0.0) {
      const double relativeDrop =
          (previousStepNorm - currentStepNorm) / previousStepNorm;
      if (relativeDrop >= 0.0 &&
          relativeDrop <= adaptiveRelativeStepDropTol) {
        return "adaptive_relative_step_drop";
      }
    }
    return "";
  };

  double previousRotationStepNorm = std::numeric_limits<double>::infinity();
  for (size_t iter = 0; iter < maxIters && consensusScale > 0.0; ++iter) {
    vector<vector<Matrix>> nextRotations = rotations;
    double stepSq = 0.0;
    for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
      const size_t objectOffset =
          dataset.robots[robotId].trajectoryVertexIds.size();
      for (size_t localIdx = 0; localIdx < rotations[robotId].size();
           ++localIdx) {
        if (!shouldUpdateVariable(robotId, localIdx)) {
          continue;
        }
        Matrix numerator = Matrix::Zero(d, d);
        double denominator = 0.0;
        for (const RelativeSEMeasurement &m : localMeasurements[robotId]) {
          if (m.p1 >= connected[robotId].size() ||
              m.p2 >= connected[robotId].size()) {
            continue;
          }
          if (m.p2 == localIdx && connected[robotId][m.p1]) {
            numerator += m.kappa * rotations[robotId][m.p1] * m.R;
            denominator += m.kappa;
          } else if (m.p1 == localIdx && connected[robotId][m.p2]) {
            numerator += m.kappa * rotations[robotId][m.p2] *
                         m.R.transpose();
            denominator += m.kappa;
          }
        }
        if (localIdx >= objectOffset) {
          const size_t objectId = localIdx - objectOffset;
          if (objectId < dataset.numObjects) {
            for (const ObjectCopyConsensusLink &link :
                 consensusGraph[robotId][objectId]) {
              const size_t neighborId = link.neighborId;
              if (neighborId >= dataset.robots.size() ||
                  neighborId >= rotations.size()) {
                continue;
              }
              const size_t neighborIdx =
                  dataset.robots[neighborId].trajectoryVertexIds.size() +
                  objectId;
              if (neighborIdx >= rotations[neighborId].size() ||
                  !connected[neighborId][neighborIdx]) {
                continue;
              }
              numerator += link.kappa * rotations[neighborId][neighborIdx];
              denominator += link.kappa;
            }
          }
        }
        if (denominator <= 0.0) {
          continue;
        }
        Matrix updated =
            (1.0 - relaxation) * rotations[robotId][localIdx] +
            relaxation * (numerator / denominator);
        if (projectEachIter) {
          updated = projectToRotationGroup(updated);
        }
        stepSq += (updated - rotations[robotId][localIdx]).squaredNorm();
        nextRotations[robotId][localIdx] = updated;
      }
    }
    nextRotations[0][0] = rootRotation;
    rotations = std::move(nextRotations);
    stats.rotationStepNorm = std::sqrt(stepSq);
    stats.rotationIters = iter + 1;
    if (stats.rotationStepNorm <= tolerance) {
      stats.rotationConverged = true;
      stats.rotationStopReason = "tolerance";
      break;
    }
    const string adaptiveReason = adaptiveStopReason(
        stats.rotationIters, previousRotationStepNorm,
        stats.rotationStepNorm);
    if (!adaptiveReason.empty()) {
      stats.rotationStopReason = adaptiveReason;
      break;
    }
    previousRotationStepNorm = stats.rotationStepNorm;
  }
  if (stats.rotationStopReason == "not_run" && maxIters > 0 &&
      consensusScale > 0.0) {
    stats.rotationStopReason = "max_iters";
  }
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    for (size_t localIdx = 0; localIdx < rotations[robotId].size();
         ++localIdx) {
      if (shouldWriteVariable(robotId, localIdx)) {
        rotations[robotId][localIdx] =
            projectToRotationGroup(rotations[robotId][localIdx]);
      }
    }
  }
  rotations[0][0] = rootRotation;

  double previousTranslationStepNorm = std::numeric_limits<double>::infinity();
  for (size_t iter = 0; iter < maxIters && consensusScale > 0.0; ++iter) {
    vector<vector<Matrix>> nextTranslations = translations;
    double stepSq = 0.0;
    for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
      const size_t objectOffset =
          dataset.robots[robotId].trajectoryVertexIds.size();
      for (size_t localIdx = 0; localIdx < translations[robotId].size();
           ++localIdx) {
        if (!shouldUpdateVariable(robotId, localIdx)) {
          continue;
        }
        Matrix numerator = Matrix::Zero(d, 1);
        double denominator = 0.0;
        for (const RelativeSEMeasurement &m : localMeasurements[robotId]) {
          if (m.p1 >= connected[robotId].size() ||
              m.p2 >= connected[robotId].size()) {
            continue;
          }
          if (m.p2 == localIdx && connected[robotId][m.p1]) {
            numerator +=
                m.tau *
                (translations[robotId][m.p1] +
                 rotations[robotId][m.p1] * m.t);
            denominator += m.tau;
          } else if (m.p1 == localIdx && connected[robotId][m.p2]) {
            numerator +=
                m.tau *
                (translations[robotId][m.p2] -
                 rotations[robotId][localIdx] * m.t);
            denominator += m.tau;
          }
        }
        if (localIdx >= objectOffset) {
          const size_t objectId = localIdx - objectOffset;
          if (objectId < dataset.numObjects) {
            for (const ObjectCopyConsensusLink &link :
                 consensusGraph[robotId][objectId]) {
              const size_t neighborId = link.neighborId;
              if (neighborId >= dataset.robots.size() ||
                  neighborId >= translations.size()) {
                continue;
              }
              const size_t neighborIdx =
                  dataset.robots[neighborId].trajectoryVertexIds.size() +
                  objectId;
              if (neighborIdx >= translations[neighborId].size() ||
                  !connected[neighborId][neighborIdx]) {
                continue;
              }
              numerator += link.tau * translations[neighborId][neighborIdx];
              denominator += link.tau;
            }
          }
        }
        if (denominator <= 0.0) {
          continue;
        }
        const Matrix updated =
            (1.0 - relaxation) * translations[robotId][localIdx] +
            relaxation * (numerator / denominator);
        stepSq +=
            (updated - translations[robotId][localIdx]).squaredNorm();
        nextTranslations[robotId][localIdx] = updated;
      }
    }
    nextTranslations[0][0] = rootTranslation;
    translations = std::move(nextTranslations);
    stats.translationStepNorm = std::sqrt(stepSq);
    stats.translationIters = iter + 1;
    if (stats.translationStepNorm <= tolerance) {
      stats.translationConverged = true;
      stats.translationStopReason = "tolerance";
      break;
    }
    const string adaptiveReason = adaptiveStopReason(
        stats.translationIters, previousTranslationStepNorm,
        stats.translationStepNorm);
    if (!adaptiveReason.empty()) {
      stats.translationStopReason = adaptiveReason;
      break;
    }
    previousTranslationStepNorm = stats.translationStepNorm;
  }
  if (stats.translationStopReason == "not_run" && maxIters > 0 &&
      consensusScale > 0.0) {
    stats.translationStopReason = "max_iters";
  }
  translations[0][0] = rootTranslation;

  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    for (size_t localIdx = 0; localIdx < rotations[robotId].size();
         ++localIdx) {
      if (!shouldWriteVariable(robotId, localIdx)) {
        continue;
      }
      states[robotId].block(0, localIdx * (d + 1), d, d) =
          rotations[robotId][localIdx];
      states[robotId].block(0, localIdx * (d + 1) + d, d, 1) =
          translations[robotId][localIdx];
    }
  }

  stats.objective = evaluateObjectCopyChordalObjective(
      dataset, localMeasurements, consensusGraph, rotations, translations,
      stats.localObjective, stats.objectLocalObjective,
      stats.fixedTrajectoryObjective, stats.consensusObjective);
  stats.scopeObjective =
      updateTrajectoryVariables
          ? stats.objective
          : (stats.objectLocalObjective + stats.consensusObjective);
  stats.activeDirectedObjectPairs = countActiveKnownObjectPairs(
      neighbors, knownObjects, observedObjects, observedConfidence,
      relayConfidence);
  const double communicatedDoubles =
      static_cast<double>(stats.activeDirectedObjectPairs) *
      (static_cast<double>(stats.rotationIters) * d * d +
       static_cast<double>(stats.translationIters) * d);
  stats.commMb = communicatedDoubles * sizeof(double) / (1024.0 * 1024.0);

  cout << "Distributed object-copy chordal Jacobi initialization"
       << " | object_jacobi_scope = " << stats.scope
       << " | object_jacobi_connected_variables = " << connectedVariables
       << " | object_jacobi_updated_variables = "
       << stats.updatedVariables
       << " | object_jacobi_undirected_consensus_edges = "
       << undirectedConsensusEdges
       << " | object_jacobi_active_directed_pairs = "
       << stats.activeDirectedObjectPairs
       << " | object_jacobi_consensus_kappa = " << stats.consensusKappa
       << " | object_jacobi_consensus_tau = " << stats.consensusTau
       << " | object_jacobi_rotation_iters = " << stats.rotationIters
       << " | object_jacobi_translation_iters = "
       << stats.translationIters
       << " | object_jacobi_rotation_converged = "
       << (stats.rotationConverged ? "true" : "false")
       << " | object_jacobi_translation_converged = "
       << (stats.translationConverged ? "true" : "false")
       << " | object_jacobi_adaptive_budget = "
       << (stats.adaptiveBudget ? "true" : "false")
       << " | object_jacobi_min_iters = " << stats.adaptiveMinIters
       << " | object_jacobi_budget_step_tol = "
       << stats.adaptiveStepTol
       << " | object_jacobi_relative_step_drop_tol = "
       << stats.adaptiveRelativeStepDropTol
       << " | object_jacobi_rotation_stop_reason = "
       << stats.rotationStopReason
       << " | object_jacobi_translation_stop_reason = "
       << stats.translationStopReason
       << " | object_jacobi_rotation_step_norm = "
       << stats.rotationStepNorm
       << " | object_jacobi_translation_step_norm = "
       << stats.translationStepNorm
       << " | object_jacobi_local_objective = "
       << stats.localObjective
       << " | object_jacobi_object_local_objective = "
       << stats.objectLocalObjective
       << " | object_jacobi_fixed_trajectory_objective = "
       << stats.fixedTrajectoryObjective
       << " | object_jacobi_consensus_objective = "
       << stats.consensusObjective
       << " | object_jacobi_scope_objective = "
       << stats.scopeObjective
       << " | object_jacobi_objective = " << stats.objective
       << " | object_jacobi_comm_mb = " << stats.commMb
       << " | relaxation = " << relaxation
       << " | project_each_iter = "
       << (projectEachIter ? "true" : "false")
       << "." << endl;
  const bool stoppedByBudget =
      stats.rotationStopReason.rfind("adaptive_", 0) == 0 ||
      stats.translationStopReason.rfind("adaptive_", 0) == 0;
  if ((!stats.rotationConverged || !stats.translationConverged) &&
      !stoppedByBudget) {
    cerr << "Warning: distributed object-copy chordal Jacobi initialization "
         << "reached its iteration limit before convergence"
         << " | rotation_converged = "
         << (stats.rotationConverged ? "true" : "false")
         << " | translation_converged = "
         << (stats.translationConverged ? "true" : "false")
         << " | rotation_step_norm = " << stats.rotationStepNorm
         << " | translation_step_norm = "
         << stats.translationStepNorm << endl;
  }
  return stats;
}

DistributedCorrectionStats applyRiemannianGradientTracking(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &knownObjects,
    const vector<vector<bool>> &observedObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborAuxiliaries,
    const vector<SparseMatrix> &baseQs, double alpha,
    double observedConfidence, double relayConfidence,
    const string &weightMode, unsigned d, vector<Matrix> &states,
    vector<Matrix> &trackers, vector<Matrix> &previousGradients,
    vector<Matrix> &outgoingAuxiliaries) {
  DistributedCorrectionStats stats;
  if (alpha <= 0.0 || states.empty()) {
    return stats;
  }

  const unsigned blockWidth = d + 1;
  vector<Matrix> correctedStates = states;
  vector<Matrix> mixedTrackers = trackers;

  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    if (robotId >= activeNeighbors.size()) {
      continue;
    }
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const size_t objectOffset = robot.trajectoryVertexIds.size();
    const double selfWeight =
        weightMode == "matrix"
            ? selfMixingWeight(activeNeighbors[robotId], weightMode)
            : 1.0;

    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId]) {
        continue;
      }

      const size_t objectIndex = objectOffset + objectId;
      const Matrix localPose = blockPose(states[robotId], objectIndex, d);
      const Matrix localTracker =
          blockPose(trackers[robotId], objectIndex, d);

      vector<pair<Matrix, double>> poseBlocks;
      vector<pair<Matrix, double>> trackerBlocks;
      poseBlocks.emplace_back(localPose, selfWeight);
      trackerBlocks.emplace_back(localTracker, selfWeight);

      for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
        const unsigned neighborId = neighbor.id;
        if (neighborId >= knownObjects.size() ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        const double mixWeight = effectiveConsensusWeight(
            neighbor.weight, robotId, neighborId, objectId, observedObjects,
            knownObjects, observedConfidence, relayConfidence);
        if (mixWeight <= 0.0) {
          continue;
        }
        const auto objectCacheIt = neighborObjects[robotId].find(neighborId);
        if (objectCacheIt != neighborObjects[robotId].end() &&
            objectId < objectCacheIt->second.size() &&
            objectCacheIt->second[objectId].size() != 0) {
          poseBlocks.emplace_back(objectCacheIt->second[objectId], mixWeight);
        }
        const auto auxCacheIt = neighborAuxiliaries[robotId].find(neighborId);
        if (auxCacheIt != neighborAuxiliaries[robotId].end() &&
            objectId < auxCacheIt->second.size() &&
            auxCacheIt->second[objectId].size() != 0) {
          trackerBlocks.emplace_back(auxCacheIt->second[objectId], mixWeight);
        }
      }

      const Matrix mixedPose = weightedAveragePoseBlocks(poseBlocks, d);
      const Matrix mixedTracker =
          weightedAverageBlocks(trackerBlocks, localTracker);
      const Matrix targetPose =
          projectPoseBlock(mixedPose - alpha * mixedTracker, d);
      const Matrix correctedPose =
          dampedPoseUpdate(localPose, targetPose, alpha, d);
      correctedStates[robotId].block(0, objectIndex * blockWidth, d,
                                     blockWidth) = correctedPose;
      mixedTrackers[robotId].block(0, objectIndex * blockWidth, d,
                                   blockWidth) = mixedTracker;
      stats.stepNorm += (correctedPose - localPose).squaredNorm();
      stats.auxiliaryNorm += mixedTracker.squaredNorm();
      ++stats.updated;
    }
  }

  states = std::move(correctedStates);
  vector<Matrix> newGradients =
      computeLocalObjectGradients(dataset, baseQs, states, knownObjects, d);
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const size_t objectOffset = robot.trajectoryVertexIds.size();
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId]) {
        continue;
      }
      const size_t objectIndex = objectOffset + objectId;
      const size_t colOffset = objectIndex * blockWidth;
      const Matrix updatedTracker =
          mixedTrackers[robotId].block(0, colOffset, d, blockWidth) +
          newGradients[robotId].block(0, colOffset, d, blockWidth) -
          previousGradients[robotId].block(0, colOffset, d, blockWidth);
      trackers[robotId].block(0, colOffset, d, blockWidth) = updatedTracker;
    }
  }
  previousGradients = std::move(newGradients);
  outgoingAuxiliaries = trackers;
  stats.stepNorm = std::sqrt(stats.stepNorm);
  stats.auxiliaryNorm = std::sqrt(stats.auxiliaryNorm);
  return stats;
}

DistributedCorrectionStats applyRiemannianExactDiffusion(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &knownObjects,
    const vector<vector<bool>> &observedObjects,
    const vector<SparseMatrix> &baseQs, double alpha,
    double observedConfidence, double relayConfidence,
    const string &weightMode, unsigned d, vector<Matrix> &states,
    vector<Matrix> &previousPsi, vector<Matrix> &outgoingAuxiliaries) {
  DistributedCorrectionStats stats;
  if (alpha <= 0.0 || states.empty()) {
    return stats;
  }

  const unsigned blockWidth = d + 1;
  const vector<Matrix> gradients =
      computeLocalObjectGradients(dataset, baseQs, states, knownObjects, d);
  vector<Matrix> localPhi = states;
  vector<Matrix> newPsi = previousPsi;

  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const size_t objectOffset = robot.trajectoryVertexIds.size();
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId]) {
        continue;
      }
      const size_t objectIndex = objectOffset + objectId;
      const size_t colOffset = objectIndex * blockWidth;
      const Matrix pose = blockPose(states[robotId], objectIndex, d);
      const Matrix gradient =
          gradients[robotId].block(0, colOffset, d, blockWidth);
      const Matrix psi = projectPoseBlock(pose - alpha * gradient, d);
      const Matrix prevPsi =
          previousPsi[robotId].block(0, colOffset, d, blockWidth);
      const Matrix phi = projectPoseBlock(psi + pose - prevPsi, d);
      newPsi[robotId].block(0, colOffset, d, blockWidth) = psi;
      localPhi[robotId].block(0, colOffset, d, blockWidth) = phi;
      stats.auxiliaryNorm += phi.squaredNorm();
    }
  }

  vector<Matrix> correctedStates = states;
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    if (robotId >= activeNeighbors.size()) {
      continue;
    }
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const size_t objectOffset = robot.trajectoryVertexIds.size();
    const double selfWeight =
        weightMode == "matrix"
            ? selfMixingWeight(activeNeighbors[robotId], weightMode)
            : 1.0;
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId]) {
        continue;
      }
      const size_t objectIndex = objectOffset + objectId;
      const Matrix localPose = blockPose(states[robotId], objectIndex, d);
      const Matrix localPhiBlock = blockPose(localPhi[robotId], objectIndex, d);
      vector<pair<Matrix, double>> phiBlocks;
      phiBlocks.emplace_back(localPhiBlock, selfWeight);
      for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
        const unsigned neighborId = neighbor.id;
        if (neighborId >= knownObjects.size() ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        const double mixWeight = effectiveConsensusWeight(
            neighbor.weight, robotId, neighborId, objectId, observedObjects,
            knownObjects, observedConfidence, relayConfidence);
        if (mixWeight <= 0.0) {
          continue;
        }
        const size_t neighborObjectIndex =
            dataset.robots[neighborId].trajectoryVertexIds.size() + objectId;
        phiBlocks.emplace_back(blockPose(localPhi[neighborId],
                                         neighborObjectIndex, d),
                               mixWeight);
      }
      const Matrix correctedPose =
          dampedPoseUpdate(localPose, weightedAveragePoseBlocks(phiBlocks, d),
                           alpha, d);
      correctedStates[robotId].block(0, objectIndex * blockWidth, d,
                                     blockWidth) = correctedPose;
      stats.stepNorm += (correctedPose - localPose).squaredNorm();
      ++stats.updated;
    }
  }

  states = std::move(correctedStates);
  previousPsi = std::move(newPsi);
  outgoingAuxiliaries = std::move(localPhi);
  stats.stepNorm = std::sqrt(stats.stepNorm);
  stats.auxiliaryNorm = std::sqrt(stats.auxiliaryNorm);
  return stats;
}

size_t applyRelayConsensusAveraging(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &observedObjects,
    const vector<vector<bool>> &knownObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborObjects,
    const vector<ObjectConsensus> &consensuses,
    double beta, unsigned d, vector<Matrix> &states) {
  size_t updated = 0;
  const unsigned blockWidth = d + 1;
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    if (robotId >= activeNeighbors.size()) {
      continue;
    }
    const size_t objectOffset =
        dataset.robots[robotId].trajectoryVertexIds.size();
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId] ||
          observedObjects[robotId][objectId]) {
        continue;
      }

      Matrix numerator = Matrix::Zero(d, blockWidth);
      double denominator = 0.0;
      for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
        const unsigned neighborId = neighbor.id;
        if (neighborId >= knownObjects.size() ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        const auto cacheIt = neighborObjects[robotId].find(neighborId);
        if (cacheIt == neighborObjects[robotId].end()) {
          continue;
        }
        const vector<Matrix> &neighborCache = cacheIt->second;
        if (objectId >= neighborCache.size() ||
            neighborCache[objectId].size() == 0) {
          continue;
        }
        const double weight = neighbor.weight;
        const Matrix lambda =
            consensuses[robotId].getDual(neighborId, objectId);
        numerator += beta * weight * weight * neighborCache[objectId] -
                     weight * lambda;
        denominator += beta * weight * weight;
      }
      if (denominator <= 0.0) {
        continue;
      }

      Matrix averaged = numerator / denominator;
      const size_t localObjectIndex = objectOffset + objectId;
      states[robotId].block(0, localObjectIndex * blockWidth, d, d) =
          projectToRotationGroup(averaged.leftCols(d));
      states[robotId].block(0, localObjectIndex * blockWidth + d, d, 1) =
          averaged.rightCols(1);
      ++updated;
    }
  }
  return updated;
}

ConsensusCorrectionStats applyProximalObjectMixing(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &observedObjects,
    const vector<vector<bool>> &knownObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborObjects,
    double alpha, double observedConfidence, double relayConfidence,
    size_t backtrackingSteps, double meritTolerance,
    unsigned d, vector<Matrix> &states) {
  ConsensusCorrectionStats stats;
  if (alpha <= 0.0) {
    return stats;
  }

  const unsigned blockWidth = d + 1;
  vector<vector<bool>> hasTarget(dataset.robots.size(),
                                 vector<bool>(dataset.numObjects, false));
  vector<vector<Matrix>> targets(dataset.robots.size(),
                                 vector<Matrix>(dataset.numObjects));
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    if (robotId >= activeNeighbors.size()) {
      continue;
    }
    const size_t objectOffset =
        dataset.robots[robotId].trajectoryVertexIds.size();
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!knownObjects[robotId][objectId]) {
        continue;
      }
      const size_t localObjectIndex = objectOffset + objectId;
      const Matrix localPose = blockPose(states[robotId], localObjectIndex, d);
      vector<pair<Matrix, double>> poses;
      poses.emplace_back(
          localPose,
          std::max(0.0, robotObjectConfidence(
                              robotId, objectId, observedObjects, knownObjects,
                              observedConfidence, relayConfidence)));
      for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
        const unsigned neighborId = neighbor.id;
        if (neighborId >= knownObjects.size() ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        const auto cacheIt = neighborObjects[robotId].find(neighborId);
        if (cacheIt == neighborObjects[robotId].end()) {
          continue;
        }
        const vector<Matrix> &neighborCache = cacheIt->second;
        if (objectId >= neighborCache.size() ||
            neighborCache[objectId].size() == 0) {
          continue;
        }
        const double effectiveWeight = effectiveConsensusWeight(
            neighbor.weight, robotId, neighborId, objectId, observedObjects,
            knownObjects, observedConfidence, relayConfidence);
        if (effectiveWeight <= 0.0) {
          continue;
        }
        poses.emplace_back(neighborCache[objectId], effectiveWeight);
      }
      if (poses.size() <= 1) {
        continue;
      }

      targets[robotId][objectId] = weightedAveragePoseBlocks(poses, d);
      hasTarget[robotId][objectId] = true;
    }
  }

  if (backtrackingSteps == 0) {
    for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
      const size_t objectOffset =
          dataset.robots[robotId].trajectoryVertexIds.size();
      for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
        if (!hasTarget[robotId][objectId]) {
          continue;
        }
        const size_t objectIndex = objectOffset + objectId;
        const Matrix localPose = blockPose(states[robotId], objectIndex, d);
        states[robotId].block(0, objectIndex * blockWidth, d, blockWidth) =
            dampedPoseUpdate(localPose, targets[robotId][objectId], alpha, d);
        ++stats.updated;
      }
    }
    return stats;
  }

  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const size_t objectOffset = robot.trajectoryVertexIds.size();
    Matrix relayUpdatedState = states[robotId];
    size_t relayTargetCount = 0;
    size_t observedTargetCount = 0;

    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (!hasTarget[robotId][objectId]) {
        continue;
      }
      const bool observed =
          robotId < observedObjects.size() &&
          objectId < observedObjects[robotId].size() &&
          observedObjects[robotId][objectId];
      if (observed) {
        ++observedTargetCount;
        continue;
      }
      const size_t objectIndex = objectOffset + objectId;
      const Matrix localPose = blockPose(states[robotId], objectIndex, d);
      relayUpdatedState.block(0, objectIndex * blockWidth, d, blockWidth) =
          dampedPoseUpdate(localPose, targets[robotId][objectId], alpha, d);
      ++relayTargetCount;
    }

    if (observedTargetCount == 0) {
      if (relayTargetCount > 0) {
        states[robotId] = std::move(relayUpdatedState);
        stats.updated += relayTargetCount;
      }
      continue;
    }

    const double baseCost =
        computeOriginalMeasurementCost(robot, relayUpdatedState, d);
    double trialAlpha = alpha;
    bool accepted = false;
    Matrix acceptedState = relayUpdatedState;

    for (size_t attempt = 0; attempt <= backtrackingSteps; ++attempt) {
      Matrix candidate = relayUpdatedState;
      for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
        if (!hasTarget[robotId][objectId]) {
          continue;
        }
        const bool observed =
            robotId < observedObjects.size() &&
            objectId < observedObjects[robotId].size() &&
            observedObjects[robotId][objectId];
        if (!observed) {
          continue;
        }
        const size_t objectIndex = objectOffset + objectId;
        const Matrix localPose =
            blockPose(relayUpdatedState, objectIndex, d);
        candidate.block(0, objectIndex * blockWidth, d, blockWidth) =
            dampedPoseUpdate(localPose, targets[robotId][objectId],
                             trialAlpha, d);
      }

      const double candidateCost =
          computeOriginalMeasurementCost(robot, candidate, d);
      if (candidateCost <= baseCost + meritTolerance) {
        accepted = true;
        acceptedState = std::move(candidate);
        break;
      }
      trialAlpha *= 0.5;
    }

    if (accepted) {
      states[robotId] = std::move(acceptedState);
      stats.updated += relayTargetCount + observedTargetCount;
    } else {
      states[robotId] = std::move(relayUpdatedState);
      stats.updated += relayTargetCount;
      stats.rejected += observedTargetCount;
    }
  }
  return stats;
}

double evaluateGaugeObservationCost(
    const Matrix &state,
    const vector<GaugeCouplingObservation> &observations, unsigned d) {
  if (d != 3) {
    return 0.0;
  }
  double cost = 0.0;
  const unsigned blockWidth = d + 1;
  for (const GaugeCouplingObservation &obs : observations) {
    if (obs.weight <= 0.0 ||
        obs.targetPose.rows() != static_cast<int>(d) ||
        obs.targetPose.cols() != static_cast<int>(blockWidth)) {
      continue;
    }
    const size_t colOffset = obs.poseIndex * blockWidth;
    if (colOffset + blockWidth > static_cast<size_t>(state.cols())) {
      continue;
    }
    const Matrix localPose = blockPose(state, obs.poseIndex, d);
    const Matrix rotationResidual =
        so3LogVector(localPose.leftCols(3).transpose() *
                     obs.targetPose.leftCols(3));
    const Matrix translationResidual =
        obs.targetPose.rightCols(1) - localPose.rightCols(1);
    cost += obs.weight * obs.weight *
            (rotationResidual.squaredNorm() +
             translationResidual.squaredNorm());
  }
  return cost;
}

DistributedCorrectionStats applyNeighborObjectGaugeCoupling(
    const ObjectPGODirectoryData &dataset,
    const WeightedNeighborRound &activeNeighbors,
    const vector<vector<bool>> &observedObjects,
    const vector<vector<bool>> &knownObjects,
    const vector<map<unsigned, vector<Matrix>>> &neighborObjects,
    const vector<map<unsigned, vector<size_t>>> &neighborObjectUpdateIters,
    double alpha, double maxStep, size_t minObjects, bool anchorRoot,
    bool observedOnly, double observedConfidence, double relayConfidence,
    size_t currentIter, size_t cacheMaxAge, double cacheFreshnessDecay,
    bool meritFilter, size_t meritBacktrackingSteps, double meritTolerance,
    double meritConsensusWeight, ObjectLocalObjective localObjective,
    unsigned d, vector<Matrix> &states) {
  DistributedCorrectionStats stats;
  if (alpha <= 0.0 || d != 3) {
    return stats;
  }

  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    if (anchorRoot && robotId == 0) {
      continue;
    }
    if (robotId >= activeNeighbors.size() || robotId >= states.size() ||
        robotId >= knownObjects.size()) {
      continue;
    }

    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const size_t objectOffset = robot.trajectoryVertexIds.size();
    vector<GaugeCouplingObservation> observations;
    vector<bool> touchedObjects(dataset.numObjects, false);
    for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
      const unsigned neighborId = neighbor.id;
      if (neighborId >= knownObjects.size()) {
        continue;
      }
      const auto cacheIt = neighborObjects[robotId].find(neighborId);
      if (cacheIt == neighborObjects[robotId].end()) {
        continue;
      }
      const vector<Matrix> &neighborCache = cacheIt->second;
      const auto updateIt = neighborObjectUpdateIters[robotId].find(neighborId);
      const vector<size_t> *neighborUpdateIters =
          updateIt == neighborObjectUpdateIters[robotId].end()
              ? nullptr
              : &updateIt->second;
      for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
        if (!knownObjects[robotId][objectId] ||
            !knownObjects[neighborId][objectId]) {
          continue;
        }
        if (observedOnly) {
          const bool localObserved =
              robotId < observedObjects.size() &&
              objectId < observedObjects[robotId].size() &&
              observedObjects[robotId][objectId];
          const bool neighborObserved =
              neighborId < observedObjects.size() &&
              objectId < observedObjects[neighborId].size() &&
              observedObjects[neighborId][objectId];
          if (!localObserved || !neighborObserved) {
            continue;
          }
        }
        if (objectId >= neighborCache.size() ||
            neighborCache[objectId].size() == 0) {
          continue;
        }
        const double edgeWeight = effectiveConsensusWeight(
            neighbor.weight, robotId, neighborId, objectId, observedObjects,
            knownObjects, observedConfidence, relayConfidence);
        if (edgeWeight <= 0.0) {
          continue;
        }
        double freshnessWeight = 1.0;
        if (neighborUpdateIters != nullptr &&
            objectId < neighborUpdateIters->size()) {
          freshnessWeight = cacheFreshnessWeight(
              currentIter, (*neighborUpdateIters)[objectId], cacheMaxAge,
              cacheFreshnessDecay);
        }
        if (freshnessWeight <= 0.0) {
          ++stats.staleSkipped;
          continue;
        }
        observations.push_back(GaugeCouplingObservation{
            objectOffset + objectId, neighborCache[objectId],
            edgeWeight * freshnessWeight});
        touchedObjects[objectId] = true;
      }
    }

    const size_t uniqueObjects = static_cast<size_t>(
        std::count(touchedObjects.begin(), touchedObjects.end(), true));
    if (uniqueObjects < minObjects) {
      continue;
    }

    Matrix correctionR;
    Matrix correctionT;
    GaugeCouplingCorrectionStats correctionStats;
    if (!estimateSE3GaugeCorrection(states[robotId], observations, d,
                                    minObjects, correctionR, correctionT,
                                    &correctionStats)) {
      continue;
    }
    if (meritFilter) {
      auto meritFunction = [&](const Matrix &candidate) {
        return computeRobotMeasurementCost(robot, candidate, d,
                                           localObjective) +
               meritConsensusWeight *
                   evaluateGaugeObservationCost(candidate, observations, d);
      };
      states[robotId] = applySE3GaugeCorrectionWithMerit(
          states[robotId], correctionR, correctionT, alpha, maxStep, d,
          meritBacktrackingSteps, meritTolerance, meritFunction,
          &correctionStats);
      if (!correctionStats.meritAccepted) {
        ++stats.rejected;
        continue;
      }
    } else {
      states[robotId] = applySE3GaugeCorrection(
          states[robotId], correctionR, correctionT, alpha, maxStep, d,
          &correctionStats);
    }
    if (correctionStats.appliedStepNorm <= 1e-12) {
      continue;
    }
    ++stats.updated;
    stats.stepNorm += correctionStats.appliedStepNorm;
    stats.auxiliaryNorm += correctionStats.score;
    stats.acceptedStepScaleSum += correctionStats.appliedScale;
  }
  return stats;
}

bool writeTrajectoryPoseRows(const string &path,
                             const ObjectPGODirectoryData &dataset,
                             const vector<Matrix> &states,
                             const Matrix &lift,
                             unsigned d) {
  if (path.empty() || d != 3) {
    return false;
  }

  ofstream out(path);
  if (!out.is_open()) {
    cerr << "Could not write pose output: " << path << endl;
    return false;
  }

  out << "id x y z qx qy qz qw\n";
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const Matrix T = lift.transpose() * states[robotId];
    for (size_t localIdx = 0; localIdx < robot.trajectoryVertexIds.size();
         ++localIdx) {
      Eigen::Matrix3d R = projectToRotationGroup(
          T.block(0, localIdx * (d + 1), d, d));
      Eigen::Quaterniond q(R);
      q.normalize();
      Matrix t = T.block(0, localIdx * (d + 1) + d, d, 1);
      out << robot.trajectoryVertexIds[localIdx] << ' ' << t(0) << ' '
          << t(1) << ' ' << t(2) << ' ' << q.x() << ' ' << q.y() << ' '
          << q.z() << ' ' << q.w() << '\n';
    }
  }
  return true;
}

bool writeObjectAwarePoseRows(const string &path,
                              const ObjectPGODirectoryData &dataset,
                              const vector<Matrix> &states,
                              const vector<vector<bool>> &knownObjects,
                              const Matrix &lift,
                              unsigned d) {
  if (path.empty() || d != 3) {
    return false;
  }

  ofstream out(path);
  if (!out.is_open()) {
    cerr << "Could not write object-aware pose output: " << path << endl;
    return false;
  }

  out << "type robot_id local_vertex_id object_id x y z qx qy qz qw\n";
  for (size_t robotId = 0; robotId < dataset.robots.size(); ++robotId) {
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    const Matrix T = lift.transpose() * states[robotId];
    for (size_t localIdx = 0; localIdx < robot.trajectoryVertexIds.size();
         ++localIdx) {
      Eigen::Matrix3d R = projectToRotationGroup(
          T.block(0, localIdx * (d + 1), d, d));
      Eigen::Quaterniond q(R);
      q.normalize();
      Matrix t = T.block(0, localIdx * (d + 1) + d, d, 1);
      out << "trajectory " << robotId << ' '
          << robot.trajectoryVertexIds[localIdx] << " -1 "
          << t(0) << ' ' << t(1) << ' ' << t(2) << ' ' << q.x() << ' '
          << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
    for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      if (robotId >= knownObjects.size() ||
          objectId >= knownObjects[robotId].size() ||
          !knownObjects[robotId][objectId]) {
        continue;
      }
      const size_t poseIndex = robot.trajectoryVertexIds.size() + objectId;
      const size_t localVertexId =
          objectId < robot.objectLocalVertexIds.size()
              ? robot.objectLocalVertexIds[objectId]
              : robot.objectStart + objectId;
      Eigen::Matrix3d R = projectToRotationGroup(
          T.block(0, poseIndex * (d + 1), d, d));
      Eigen::Quaterniond q(R);
      q.normalize();
      Matrix t = T.block(0, poseIndex * (d + 1) + d, d, 1);
      out << "object " << robotId << ' ' << localVertexId << ' '
          << objectId << ' ' << t(0) << ' ' << t(1) << ' ' << t(2) << ' '
          << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w()
          << '\n';
    }
  }
  return true;
}

double readEnvDouble(const char *name, double defaultValue) {
  const char *value = std::getenv(name);
  return value == nullptr ? defaultValue : atof(value);
}

size_t readEnvSizeT(const char *name, size_t defaultValue) {
  const char *value = std::getenv(name);
  return value == nullptr ? defaultValue : static_cast<size_t>(atoi(value));
}

string readEnvString(const char *name, const string &defaultValue) {
  const char *value = std::getenv(name);
  return value == nullptr ? defaultValue : string(value);
}

bool readEnvBool(const char *name, bool defaultValue) {
  const char *value = std::getenv(name);
  if (value == nullptr) {
    return defaultValue;
  }
  const string text(value);
  return text == "1" || text == "true" || text == "TRUE" || text == "on" ||
         text == "ON";
}

string normalizeInitMode(const string &text) {
  if (text == "0" || text == "false" || text == "FALSE" || text == "off" ||
      text == "OFF" || text == "raw") {
    return "raw";
  }
  if (text == "1" || text == "true" || text == "TRUE" || text == "on" ||
      text == "ON" || text == "centralized" || text == "centralized_chordal") {
    return "centralized_chordal";
  }
  if (text == "local" || text == "local_chordal" ||
      text == "decentralized_chordal_raw_gauge") {
    return "local_chordal";
  }
  if (text == "local_chordal_synced" ||
      text == "decentralized_chordal") {
    return "local_chordal_synced";
  }
  if (text == "local_chordal_sync_pgo" ||
      text == "decentralized_chordal_sync_pgo" ||
      text == "decentralized_chordal_graph_sync" ||
      text == "distributed_chordal_sync") {
    return "local_chordal_sync_pgo";
  }
  if (text == "local_chordal_disco_sync" ||
      text == "decentralized_chordal_disco_sync" ||
      text == "disco_frame_sync" ||
      text == "disco_style_frame_sync") {
    return "local_chordal_disco_sync";
  }
  if (text == "distributed_chordal_object_jacobi" ||
      text == "decentralized_chordal_object_jacobi" ||
      text == "distributed_object_copy_chordal_jacobi" ||
      text == "object_copy_chordal_jacobi" ||
      text == "local_chordal_sync_pgo_object_jacobi") {
    return "local_chordal_sync_pgo_object_jacobi";
  }
  throw runtime_error("Unsupported object initialization mode: " + text);
}

double computeOriginalMeasurementCost(const ObjectPGORobotData &robot,
                                     const Matrix &state, unsigned d) {
  double cost = 0.0;
  for (const RelativeSEMeasurement &m : robot.trajectoryMeasurements) {
    const Matrix pose1 = blockPose(state, m.p1, d);
    const Matrix pose2 = blockPose(state, m.p2, d);
    cost += computeMeasurementError(m, pose1.leftCols(d), pose1.rightCols(1),
                                    pose2.leftCols(d), pose2.rightCols(1));
  }

  for (const ObjectObservationMeasurement &obs :
       robot.objectObservationMeasurements) {
    const RelativeSEMeasurement &m = obs.measurement;
    const Matrix pose1 = blockPose(state, m.p1, d);
    const Matrix pose2 = blockPose(state, m.p2, d);
    cost += computeMeasurementError(m, pose1.leftCols(d), pose1.rightCols(1),
                                    pose2.leftCols(d), pose2.rightCols(1));
  }

  return cost;
}

SparseMatrix fullConnectionLaplacian(
    const vector<RelativeSEMeasurement> &measurements, size_t numPoses,
    unsigned d) {
  SparseMatrix full((d + 1) * numPoses, (d + 1) * numPoses);
  if (measurements.empty()) {
    return full;
  }
  const SparseMatrix compact = constructConnectionLaplacianSE(measurements);
  vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(compact.nonZeros());
  for (int outer = 0; outer < compact.outerSize(); ++outer) {
    for (SparseMatrix::InnerIterator it(compact, outer); it; ++it) {
      if (it.row() < full.rows() && it.col() < full.cols()) {
        triplets.emplace_back(it.row(), it.col(), it.value());
      }
    }
  }
  full.setFromTriplets(triplets.begin(), triplets.end());
  return full;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    cout << "Object-based multi-robot pose graph optimization example." << endl;
    cout << "Usage: " << argv[0]
         << " <g2o_dir> [num_robots=0] [num_objects=0] [max_iters=20]"
         << " [beta=1] [eta=1] [csv_path] [object_pose_tol=0.001]"
         << " [object_max_age=5] [pose_output_path] [topology=ring]"
         << " [ring_hops=1] [local_tr_iters=5] [local_tr_tol=1e-2]"
         << " [init_mode=centralized_chordal] [topology_file]" << endl;
    return 1;
  }

  const string g2oDir = argv[1];
  const size_t numRobotsArg = (argc >= 3) ? static_cast<size_t>(atoi(argv[2])) : 0;
  const size_t numObjectsArg = (argc >= 4) ? static_cast<size_t>(atoi(argv[3])) : 0;
  const unsigned maxIters = (argc >= 5) ? static_cast<unsigned>(atoi(argv[4])) : 20;
  const double beta = (argc >= 6) ? atof(argv[5]) : 1.0;
  const double eta = (argc >= 7) ? atof(argv[6]) : 1.0;
  const string csvPath = (argc >= 8) ? argv[7] : string();
  const double objectPoseTol =
      (argc >= 9) ? atof(argv[8]) : readEnvDouble("OBJECT_POSE_TOL", 1e-3);
  const size_t objectMaxAge =
      (argc >= 10) ? static_cast<size_t>(atoi(argv[9]))
                   : readEnvSizeT("OBJECT_MAX_AGE", 5);
  string poseOutputPath;
  if (argc >= 11) {
    poseOutputPath = argv[10];
  } else if (const char *envPath = std::getenv("DPGO_POSE_OUTPUT")) {
    poseOutputPath = envPath;
  }
  const string topology =
      (argc >= 12) ? argv[11] : readEnvString("OBJECT_TOPOLOGY", "ring");
  const unsigned ringHops =
      (argc >= 13) ? static_cast<unsigned>(atoi(argv[12]))
                   : static_cast<unsigned>(readEnvSizeT("OBJECT_RING_HOPS", 1));
  const unsigned localTrustRegionIterations =
      (argc >= 14) ? static_cast<unsigned>(atoi(argv[13]))
                   : static_cast<unsigned>(
                         readEnvSizeT("OBJECT_LOCAL_TR_ITERS", 5));
  const double localTrustRegionTolerance =
      (argc >= 15) ? atof(argv[14])
                   : readEnvDouble("OBJECT_LOCAL_TR_TOL", 1e-2);
  string initMode = (argc >= 16) ? argv[15] : readEnvString("OBJECT_INIT_MODE", "");
  if (initMode.empty()) {
    initMode = readEnvBool("OBJECT_CHORDAL_INIT", true)
                   ? "centralized_chordal"
                   : "raw";
  }
  initMode = normalizeInitMode(initMode);
  const string topologyFile =
      (argc >= 17) ? argv[16] : readEnvString("COMM_TOPOLOGY_FILE", "");
  const string topologyWeightMode =
      readEnvString("COMM_WEIGHT_MODE", "unit");
  const string initTopologyFile =
      readEnvString("OBJECT_INIT_COMM_TOPOLOGY_FILE",
                    readEnvString("COMM_INIT_TOPOLOGY_FILE", ""));
  const string initTopology =
      readEnvString("OBJECT_INIT_TOPOLOGY", "");
  const unsigned initRingHops =
      static_cast<unsigned>(
          readEnvSizeT("OBJECT_INIT_RING_HOPS", ringHops));
  const string initTopologyWeightMode =
      readEnvString("OBJECT_INIT_COMM_WEIGHT_MODE", topologyWeightMode);
  string initTopologySource = "runtime_union";
  string effectiveInitTopology = topology;
  string effectiveInitTopologyFile = topologyFile;
  unsigned effectiveInitRingHops = ringHops;
  if (!initTopologyFile.empty()) {
    initTopologySource = "file";
    effectiveInitTopology = "<file>";
    effectiveInitTopologyFile = initTopologyFile;
  } else if (!initTopology.empty()) {
    initTopologySource = "default";
    effectiveInitTopology = initTopology;
    effectiveInitTopologyFile.clear();
    effectiveInitRingHops = initRingHops;
  }
  const string objectPoseOutputPath =
      readEnvString("DPGO_OBJECT_POSE_OUTPUT", "");
  const bool relayBarycentricUpdates =
      readEnvBool("OBJECT_RELAY_BARYCENTRIC_UPDATES", false);
  const string consensusModeText =
      readEnvString("OBJECT_CONSENSUS_MODE", "penalty");
  const ObjectConsensusMode consensusMode =
      parseObjectConsensusMode(consensusModeText);
  const string localObjectiveText =
      readEnvString("OBJECT_LOCAL_OBJECTIVE",
                    readEnvString("OBJECT_OBJECTIVE", "chordal"));
  const ObjectLocalObjective localObjective =
      parseObjectLocalObjective(localObjectiveText);
  double defaultConsensusAlpha = 0.0;
  if (consensusMode == ObjectConsensusMode::ProxMixing) {
    defaultConsensusAlpha = 0.2;
  } else if (consensusMode == ObjectConsensusMode::GradientTracking) {
    defaultConsensusAlpha = 0.05;
  } else if (consensusMode == ObjectConsensusMode::ExactDiffusion) {
    defaultConsensusAlpha = 0.01;
  }
  const double consensusAlpha =
      readEnvDouble("OBJECT_CONSENSUS_ALPHA", defaultConsensusAlpha);
  const double observedConsensusConfidence =
      std::max(0.0, readEnvDouble("OBJECT_OBSERVED_CONSENSUS_CONFIDENCE", 1.0));
  const double relayConsensusConfidence =
      std::max(0.0, readEnvDouble("OBJECT_RELAY_CONSENSUS_CONFIDENCE", 1.0));
  const size_t consensusBacktrackingSteps =
      readEnvSizeT("OBJECT_CONSENSUS_BACKTRACKING_STEPS", 0);
  const double consensusMeritTolerance =
      std::max(0.0, readEnvDouble("OBJECT_CONSENSUS_MERIT_TOL", 1e-10));
  const size_t geodesicLocalIterations =
      readEnvSizeT("OBJECT_GEODESIC_LOCAL_ITERS",
                   localTrustRegionIterations);
  const double geodesicLocalTolerance =
      readEnvDouble("OBJECT_GEODESIC_LOCAL_TOL",
                    localTrustRegionTolerance);
  const string geodesicSolverText =
      readEnvString("OBJECT_GEODESIC_SOLVER", "rtr_gn");
  const GeodesicSolverMode geodesicSolverMode =
      localObjective == ObjectLocalObjective::Geodesic
          ? parseGeodesicSolverMode(geodesicSolverText)
          : GeodesicSolverMode::RtrGn;
  const double geodesicTrustRegionInitialRadius =
      readEnvDouble("OBJECT_GEODESIC_TR_RADIUS", 10.0);
  const int geodesicTrustRegionMaxInnerIterations =
      static_cast<int>(
          readEnvSizeT("OBJECT_GEODESIC_TR_MAX_INNER_ITERS", 30));
  const bool geodesicRtrGnActive =
      localObjective == ObjectLocalObjective::Geodesic &&
      geodesicSolverMode == GeodesicSolverMode::RtrGn;
  const double geodesicProximalWeight =
      std::max(0.0, readEnvDouble("OBJECT_GEODESIC_PROX_WEIGHT",
                                  geodesicRtrGnActive ? 5.0 : 0.0));
  const double geodesicMaxBlockStep =
      std::max(0.0, readEnvDouble("OBJECT_GEODESIC_MAX_BLOCK_STEP",
                                  geodesicRtrGnActive ? 0.005 : 0.0));
  const bool geodesicChordalPredictor =
      localObjective == ObjectLocalObjective::Geodesic &&
      readEnvBool("OBJECT_GEODESIC_CHORDAL_PREDICTOR", false);
  const GeodesicAuxiliaryMode geodesicAuxMode =
      localObjective == ObjectLocalObjective::Geodesic
          ? parseGeodesicAuxiliaryMode(
                readEnvString("OBJECT_GEODESIC_AUX_MODE", "none"))
          : GeodesicAuxiliaryMode::None;
  const double geodesicAuxScale =
      std::max(0.0, readEnvDouble("OBJECT_GEODESIC_AUX_SCALE", 1e-3));
  const bool geodesicAuxPoseTriggeredOnly =
      readEnvBool("OBJECT_GEODESIC_AUX_POSE_TRIGGERED_ONLY", true);
  const bool geodesicLocalMeritFilter =
      localObjective == ObjectLocalObjective::Geodesic &&
      readEnvBool("OBJECT_GEODESIC_LOCAL_MERIT_FILTER", true);
  const size_t geodesicLocalMeritBacktrackingSteps =
      readEnvSizeT("OBJECT_GEODESIC_LOCAL_MERIT_BACKTRACKING_STEPS",
                   geodesicLocalMeritFilter ? 6 : 0);
  const double geodesicLocalMeritTolerance =
      std::max(0.0,
               readEnvDouble("OBJECT_GEODESIC_LOCAL_MERIT_TOL", 1e-10));
  const bool mainMeritFilter =
      readEnvBool("OBJECT_MAIN_MERIT_FILTER", false);
  const size_t mainMeritBacktrackingSteps =
      readEnvSizeT("OBJECT_MAIN_MERIT_BACKTRACKING_STEPS",
                   mainMeritFilter ? 6 : 0);
  const double mainMeritTolerance =
      std::max(0.0, readEnvDouble("OBJECT_MAIN_MERIT_TOL", 1e-10));
  const bool residualFeedback =
      readEnvBool("OBJECT_RESIDUAL_FEEDBACK", false);
  const double residualFeedbackScale =
      readEnvDouble("OBJECT_RESIDUAL_FEEDBACK_SCALE", 0.1);
  const bool gaugeCoupling =
      readEnvBool("OBJECT_GAUGE_COUPLING", false);
  const double gaugeCouplingAlpha =
      readEnvDouble("OBJECT_GAUGE_COUPLING_ALPHA", 0.2);
  const double gaugeCouplingMaxStep =
      std::max(0.0, readEnvDouble("OBJECT_GAUGE_COUPLING_MAX_STEP", 0.02));
  const size_t gaugeCouplingMinObjects =
      readEnvSizeT("OBJECT_GAUGE_COUPLING_MIN_OBJECTS", 2);
  const bool gaugeCouplingAnchorRoot =
      readEnvBool("OBJECT_GAUGE_COUPLING_ANCHOR_ROOT", true);
  const bool gaugeCouplingObservedOnly =
      readEnvBool("OBJECT_GAUGE_COUPLING_OBSERVED_ONLY", false);
  const size_t cacheFreshnessMaxAge =
      readEnvSizeT("OBJECT_CACHE_MAX_AGE", 0);
  const double cacheFreshnessDecay =
      std::max(0.0, readEnvDouble("OBJECT_CACHE_FRESHNESS_DECAY", 0.0));
  const bool gaugeMeritFilter =
      readEnvBool("OBJECT_GAUGE_MERIT_FILTER", false);
  const size_t gaugeMeritBacktrackingSteps =
      readEnvSizeT("OBJECT_GAUGE_MERIT_BACKTRACKING_STEPS",
                   gaugeMeritFilter ? 4 : 0);
  const double gaugeMeritTolerance =
      std::max(0.0, readEnvDouble("OBJECT_GAUGE_MERIT_TOL", 1e-10));
  const double gaugeMeritConsensusWeight =
      std::max(0.0,
               readEnvDouble("OBJECT_GAUGE_MERIT_CONSENSUS_WEIGHT", 1.0));
  const size_t frameSyncRuntimePeriod =
      readEnvSizeT("OBJECT_FRAME_SYNC_RUNTIME_PERIOD", 0);
  const bool frameSyncRuntimeCountRegistration =
      readEnvBool("OBJECT_FRAME_SYNC_RUNTIME_COUNT_REGISTRATION_COMM", true);
  const string communicationPolicy =
      readEnvString("OBJECT_COMMUNICATION_POLICY", "triggered");
  const bool innovationGatedCommunication =
      communicationPolicy == "innovation_gated" ||
      communicationPolicy == "innovation-gated";
  const bool boundaryBudgetCommunication =
      communicationPolicy == "boundary_budget" ||
      communicationPolicy == "boundary-budget" ||
      communicationPolicy == "receiver_boundary_budget" ||
      communicationPolicy == "receiver-boundary-budget" ||
      communicationPolicy == "boundary_predictive" ||
      communicationPolicy == "boundary-predictive" ||
      communicationPolicy == "predictive_boundary_budget" ||
      communicationPolicy == "predictive-boundary-budget";
  const bool predictiveBoundaryBudgetCommunication =
      communicationPolicy == "boundary_predictive" ||
      communicationPolicy == "boundary-predictive" ||
      communicationPolicy == "predictive_boundary_budget" ||
      communicationPolicy == "predictive-boundary-budget";
  if (communicationPolicy != "triggered" &&
      communicationPolicy != "default" &&
      !innovationGatedCommunication &&
      !boundaryBudgetCommunication) {
    throw runtime_error("Unsupported OBJECT_COMMUNICATION_POLICY: " +
                        communicationPolicy);
  }
  const double innovationThreshold =
      std::max(0.0, readEnvDouble("OBJECT_INNOVATION_THRESHOLD",
                                  objectPoseTol));
  const double boundaryBudgetPoseFraction =
      std::max(0.0, readEnvDouble("OBJECT_BOUNDARY_BUDGET_POSE_FRACTION",
                                  1.0));
  const char *boundaryBudgetMaxPosesEnv =
      std::getenv("OBJECT_BOUNDARY_BUDGET_MAX_POSES");
  const bool hasBoundaryBudgetMaxPoses = boundaryBudgetMaxPosesEnv != nullptr;
  const size_t boundaryBudgetMaxPoses =
      hasBoundaryBudgetMaxPoses
          ? readEnvSizeT("OBJECT_BOUNDARY_BUDGET_MAX_POSES", 0)
          : std::numeric_limits<size_t>::max();
  const char *boundaryPredictiveGainFractionEnv =
      std::getenv("OBJECT_BOUNDARY_PREDICTIVE_GAIN_FRACTION");
  const bool hasBoundaryPredictiveGainFraction =
      boundaryPredictiveGainFractionEnv != nullptr;
  const double boundaryPredictiveGainFraction =
      hasBoundaryPredictiveGainFraction
          ? std::min(1.0,
                     std::max(0.0,
                              readEnvDouble(
                                  "OBJECT_BOUNDARY_PREDICTIVE_GAIN_FRACTION",
                                  1.0)))
          : -1.0;
  const double outerStopConsensusCost =
      readEnvDouble("OBJECT_OUTER_STOP_CONSENSUS_COST", -1.0);
  const size_t outerStopMinIters =
      readEnvSizeT("OBJECT_OUTER_STOP_MIN_ITERS", 0);

  cout << "Object-based multi-robot pose graph optimization example." << endl;
  cout << "Loading object-aware dataset from " << g2oDir << "." << endl;

  ObjectPGODirectoryData dataset =
      loadObjectAwareG2ODirectory(g2oDir, numRobotsArg, numObjectsArg);

  const unsigned numRobots = static_cast<unsigned>(dataset.numRobots);
  const unsigned d = static_cast<unsigned>(dataset.dimension);
  const unsigned r = d;
  const unsigned blockWidth = d + 1;
  const size_t blockBytes = static_cast<size_t>(d) * blockWidth * sizeof(double);
  if (localObjective == ObjectLocalObjective::Geodesic &&
      (d != 3 || r != d)) {
    throw runtime_error("Geodesic DRAN-Object currently requires drone SE(3) "
                        "with r=d=3");
  }
  if (localObjective == ObjectLocalObjective::Geodesic &&
      consensusMode != ObjectConsensusMode::Penalty) {
    throw runtime_error("Geodesic DRAN-Object currently supports "
                        "OBJECT_CONSENSUS_MODE=penalty only");
  }
  if (geodesicAuxMode != GeodesicAuxiliaryMode::None &&
      (localObjective != ObjectLocalObjective::Geodesic ||
       geodesicSolverMode != GeodesicSolverMode::RtrGn ||
       consensusMode != ObjectConsensusMode::Penalty)) {
    throw runtime_error("OBJECT_GEODESIC_AUX_MODE currently requires "
                        "geodesic local objective, rtr_gn solver, and "
                        "OBJECT_CONSENSUS_MODE=penalty");
  }
  if (residualFeedback &&
      localObjective != ObjectLocalObjective::Chordal) {
    throw runtime_error("OBJECT_RESIDUAL_FEEDBACK currently supports the "
                        "chordal local objective only");
  }
  if (residualFeedback && consensusMode != ObjectConsensusMode::Penalty) {
    throw runtime_error("OBJECT_RESIDUAL_FEEDBACK currently supports "
                        "OBJECT_CONSENSUS_MODE=penalty only");
  }
  if (residualFeedback && usesDistributedObjectCorrection(consensusMode)) {
    throw runtime_error("OBJECT_RESIDUAL_FEEDBACK cannot share auxiliary "
                        "channels with gradient_tracking/exact_diffusion");
  }
  if (gaugeCoupling && d != 3) {
    throw runtime_error("OBJECT_GAUGE_COUPLING currently supports SE(3) "
                        "object states only");
  }
  if (frameSyncRuntimePeriod > 0 &&
      consensusMode != ObjectConsensusMode::Penalty) {
    throw runtime_error("OBJECT_FRAME_SYNC_RUNTIME_PERIOD currently supports "
                        "OBJECT_CONSENSUS_MODE=penalty only");
  }
  if (frameSyncRuntimePeriod > 0 && residualFeedback) {
    throw runtime_error("OBJECT_FRAME_SYNC_RUNTIME_PERIOD cannot be combined "
                        "with OBJECT_RESIDUAL_FEEDBACK");
  }
  if (consensusMode == ObjectConsensusMode::EdgeAdmm && r != d) {
    throw runtime_error("edge_admm object consensus currently requires r == d");
  }

  cout << "Loaded " << numRobots << " robots, " << dataset.numObjects
       << " shared objects, dimension " << d << "." << endl;
  cout << "Object consensus topology: " << topology
       << " | ring_hops = " << ringHops
       << " | local_tr_iters = " << localTrustRegionIterations
       << " | local_tr_tol = " << localTrustRegionTolerance
       << " | init_mode = " << initMode
       << " | topology_file = "
       << (topologyFile.empty() ? string("<none>") : topologyFile)
       << " | topology_weight_mode = " << topologyWeightMode
       << " | init_topology_source = " << initTopologySource
       << " | init_topology = " << effectiveInitTopology
       << " | init_ring_hops = " << effectiveInitRingHops
       << " | init_topology_file = "
       << (effectiveInitTopologyFile.empty() ? string("<none>")
                                             : effectiveInitTopologyFile)
       << " | init_topology_weight_mode = " << initTopologyWeightMode
       << " | local_objective = "
       << objectLocalObjectiveName(localObjective)
       << " | geodesic_solver = "
       << geodesicSolverModeName(geodesicSolverMode)
       << " | consensus_mode = " << objectConsensusModeName(consensusMode)
       << " | consensus_alpha = " << consensusAlpha
       << " | geodesic_local_iters = " << geodesicLocalIterations
       << " | geodesic_local_tol = " << geodesicLocalTolerance
       << " | geodesic_tr_radius = "
       << geodesicTrustRegionInitialRadius
       << " | geodesic_tr_max_inner_iters = "
       << geodesicTrustRegionMaxInnerIterations
       << " | geodesic_prox_weight = "
       << geodesicProximalWeight
       << " | geodesic_max_block_step = "
       << geodesicMaxBlockStep
       << " | geodesic_chordal_predictor = "
       << (geodesicChordalPredictor ? "on" : "off")
       << " | geodesic_aux_mode = "
       << geodesicAuxiliaryModeName(geodesicAuxMode)
       << " | geodesic_aux_scale = "
       << geodesicAuxScale
       << " | geodesic_aux_pose_triggered_only = "
       << (geodesicAuxPoseTriggeredOnly ? "on" : "off")
       << " | geodesic_local_merit_filter = "
       << (geodesicLocalMeritFilter ? "on" : "off")
       << " | geodesic_local_merit_backtracking_steps = "
       << geodesicLocalMeritBacktrackingSteps
       << " | geodesic_local_merit_tol = "
       << geodesicLocalMeritTolerance
       << " | consensus_backtracking_steps = "
       << consensusBacktrackingSteps
       << " | consensus_merit_tol = " << consensusMeritTolerance
       << " | main_merit_filter = "
       << (mainMeritFilter ? "on" : "off")
       << " | main_merit_backtracking_steps = "
       << mainMeritBacktrackingSteps
       << " | main_merit_tol = " << mainMeritTolerance
       << " | residual_feedback = "
       << (residualFeedback ? "on" : "off")
       << " | residual_feedback_scale = "
       << residualFeedbackScale
       << " | gauge_coupling = "
       << (gaugeCoupling ? "on" : "off")
       << " | gauge_coupling_alpha = " << gaugeCouplingAlpha
       << " | gauge_coupling_max_step = "
       << gaugeCouplingMaxStep
       << " | gauge_coupling_min_objects = "
       << gaugeCouplingMinObjects
       << " | gauge_coupling_anchor_root = "
       << (gaugeCouplingAnchorRoot ? "on" : "off")
       << " | gauge_coupling_observed_only = "
       << (gaugeCouplingObservedOnly ? "on" : "off")
       << " | cache_freshness_max_age = "
       << cacheFreshnessMaxAge
       << " | cache_freshness_decay = "
       << cacheFreshnessDecay
       << " | gauge_merit_filter = "
       << (gaugeMeritFilter ? "on" : "off")
       << " | gauge_merit_backtracking_steps = "
       << gaugeMeritBacktrackingSteps
       << " | gauge_merit_tol = " << gaugeMeritTolerance
       << " | gauge_merit_consensus_weight = "
       << gaugeMeritConsensusWeight
       << " | frame_sync_runtime_period = "
       << frameSyncRuntimePeriod
       << " | frame_sync_runtime_count_registration_comm = "
       << (frameSyncRuntimeCountRegistration ? "on" : "off")
       << " | communication_policy = "
       << (predictiveBoundaryBudgetCommunication
               ? "boundary_predictive"
               : (boundaryBudgetCommunication
                      ? "boundary_budget"
                      : (innovationGatedCommunication ? "innovation_gated"
                                                      : "triggered")))
       << " | innovation_threshold = " << innovationThreshold
       << " | boundary_budget_pose_fraction = "
       << boundaryBudgetPoseFraction
       << " | boundary_budget_max_poses = "
       << (hasBoundaryBudgetMaxPoses
               ? to_string(boundaryBudgetMaxPoses)
               : string("<none>"))
       << " | boundary_predictive_gain_fraction = "
       << (hasBoundaryPredictiveGainFraction
               ? to_string(boundaryPredictiveGainFraction)
               : string("<none>"))
       << " | outer_stop_consensus_cost = "
       << (outerStopConsensusCost >= 0.0
               ? to_string(outerStopConsensusCost)
               : string("<none>"))
       << " | outer_stop_min_iters = " << outerStopMinIters
       << " | observed_confidence = " << observedConsensusConfidence
       << " | relay_confidence = " << relayConsensusConfidence
       << " | relay_barycentric_updates = "
       << (relayBarycentricUpdates ? "on" : "off")
       << "." << endl;
  if (dataset.numObjects == 0) {
    cout << "No shared objects were found or inferred; object consensus is"
         << " inactive and communication will remain zero." << endl;
  }

  ofstream csv;
  if (!csvPath.empty()) {
    csv.open(csvPath);
    csv << "iter,local_model_cost,measurement_cost,consensus_cost,"
           "chordal_measurement_cost,geodesic_measurement_cost,"
           "primal_residual,dual_residual,total_gradnorm,object_comm_poses,"
           "object_comm_payload_blocks,innovation_gate_skipped,"
           "innovation_gate_forced,receiver_boundary_updates,"
           "receiver_boundary_cold_starts,"
           "receiver_boundary_cache_delta_sum,"
           "receiver_boundary_cache_delta_max,"
           "receiver_boundary_disagreement_sum,"
           "receiver_boundary_disagreement_max,"
           "boundary_budget_candidates,boundary_budget_selected,"
           "boundary_budget_forced,boundary_budget_skipped,"
           "boundary_budget_score_sum,"
           "boundary_budget_selected_score_sum,"
           "boundary_predictive_gain_sum,"
           "boundary_predictive_selected_gain_sum,"
           "boundary_predictive_stiffness_sum,"
           "boundary_predictive_gain_target_sum,"
           "boundary_predictive_selected_gain_fraction,"
           "comm_mb,cumulative_object_comm_poses,"
           "cumulative_comm_mb,"
           "known_object_copies,relay_object_copies,active_consensus_pairs,"
           "relay_closed_form_updates,iter_time_sec,cumulative_time_sec,"
           "consensus_mode,consensus_correction_updates,"
           "consensus_rejected_updates,edge_target_updates,"
           "distributed_step_norm,distributed_aux_norm,"
           "gauge_merit_filtered_updates,gauge_merit_rejected_updates,"
           "gauge_merit_mean_step_scale,stale_cache_skipped,"
           "main_merit_filtered_updates,main_merit_rejected_updates,"
           "main_merit_mean_step_scale,"
           "geodesic_step_limited_updates,geodesic_step_mean_scale,"
           "geodesic_step_max_block_delta,"
           "residual_feedback_payload_blocks,"
           "geodesic_aux_payloads,geodesic_aux_comm_mb,"
           "frame_sync_used_edges,frame_sync_connected_robots,"
           "frame_sync_comm_mb,frame_sync_registration_comm_mb,"
           "frame_sync_solver_comm_mb,frame_sync_objective\n";
  }

  Matrix lift = fixedStiefelVariable(d, r);

  vector<SparseMatrix> baseQs;
  vector<vector<RelativeSEMeasurement>> robotLocalMeasurements;
  vector<unique_ptr<QuadraticProblem>> problems;
  vector<unique_ptr<QuadraticOptimizer>> optimizers;
  vector<unique_ptr<GeodesicObjectProblem>> geodesicProblems;
  vector<unique_ptr<RoptOptimizer>> geodesicOptimizers;
  vector<Matrix> currentEstimates;
  vector<ObjectConsensus> consensuses;
  vector<WeightedNeighborRound> topologySchedule =
      readTopologySchedule(topologyFile, numRobots, topologyWeightMode);
  if (topologySchedule.empty()) {
    topologySchedule.push_back(
        makeDefaultWeightedNeighbors(numRobots, topology, ringHops));
  }
  const WeightedNeighborRound runtimeNeighbors =
      unionWeightedNeighbors(topologySchedule, numRobots);
  WeightedNeighborRound setupNeighbors;
  if (!initTopologyFile.empty()) {
    vector<WeightedNeighborRound> initTopologySchedule =
        readTopologySchedule(initTopologyFile, numRobots,
                             initTopologyWeightMode);
    if (initTopologySchedule.empty()) {
      throw runtime_error("OBJECT_INIT_COMM_TOPOLOGY_FILE did not provide any "
                          "valid initialization topology rounds: " +
                          initTopologyFile);
    }
    setupNeighbors = unionWeightedNeighbors(initTopologySchedule, numRobots);
  } else if (!initTopology.empty()) {
    setupNeighbors =
        makeDefaultWeightedNeighbors(numRobots, initTopology, initRingHops);
  } else {
    setupNeighbors = runtimeNeighbors;
  }
  vector<vector<bool>> robotObservedObjects;
  robotObservedObjects.reserve(numRobots);
  for (const ObjectPGORobotData &robot : dataset.robots) {
    robotObservedObjects.push_back(observedObjects(robot, dataset.numObjects));
  }
  vector<vector<WeightedNeighbor>> neighborLists;
  vector<map<unsigned, vector<Matrix>>> neighborObjects;
  vector<map<unsigned, vector<size_t>>> neighborObjectUpdateIters;
  vector<map<unsigned, vector<Matrix>>> neighborAuxiliaries;
  vector<map<unsigned, vector<Matrix>>> neighborGeodesicAuxiliaries;
  vector<map<unsigned, ObjectSendState>> outgoingObjectStates;
  const bool usesAuxiliaryExchange =
      usesDistributedObjectCorrection(consensusMode) || residualFeedback;
  const bool usesGeodesicAuxiliaryExchange =
      geodesicAuxMode != GeodesicAuxiliaryMode::None;
  baseQs.reserve(numRobots);
  robotLocalMeasurements.reserve(numRobots);
  problems.reserve(numRobots);
  optimizers.reserve(numRobots);
  geodesicProblems.reserve(numRobots);
  geodesicOptimizers.reserve(numRobots);
  currentEstimates.reserve(numRobots);
  consensuses.reserve(numRobots);
  neighborLists.reserve(numRobots);
  neighborObjects.reserve(numRobots);
  neighborObjectUpdateIters.reserve(numRobots);
  neighborAuxiliaries.reserve(numRobots);
  neighborGeodesicAuxiliaries.reserve(numRobots);
  outgoingObjectStates.reserve(numRobots);

  double initializationCommMB = 0.0;
  FrameSyncRunStats initFrameSyncStats;
  ObjectCopyChordalJacobiStats objectJacobiStats;
  const bool runObjectCopyJacobiInit =
      initMode == "local_chordal_sync_pgo_object_jacobi";
  const bool runDiscoStyleInit =
      initMode == "local_chordal_disco_sync";
  vector<Matrix> initialStates;
  if (initMode == "centralized_chordal") {
    initialStates = chordalObjectAwareInitialStates(dataset, d);
  } else if (initMode == "local_chordal" ||
             initMode == "local_chordal_synced" ||
             initMode == "local_chordal_sync_pgo" ||
             initMode == "local_chordal_disco_sync" ||
             initMode == "local_chordal_sync_pgo_object_jacobi") {
    initialStates = localChordalObjectAwareInitialStates(dataset, d);
    if (initMode == "local_chordal_synced") {
      synchronizeLocalChordalFrames(dataset, setupNeighbors, d, initialStates);
    } else if (initMode == "local_chordal_sync_pgo" ||
               initMode == "local_chordal_disco_sync" ||
               initMode == "local_chordal_sync_pgo_object_jacobi") {
      initFrameSyncStats = synchronizeLocalChordalFramesWithGraph(
          dataset, setupNeighbors, d, initialStates,
          runDiscoStyleInit ? "disco_style_init" : "graph_sync_init",
          runDiscoStyleInit);
      initializationCommMB += initFrameSyncStats.commMb;
    }
  } else {
    initialStates.reserve(numRobots);
    for (const ObjectPGORobotData &robot : dataset.robots) {
      initialStates.push_back(assembleInitialState(robot.trajectoryInitialPoses,
                                                  robot.objectInitialPoses, d));
    }
  }

  vector<vector<bool>> robotKnownObjects = robotObservedObjects;
  if (initMode == "centralized_chordal") {
    for (vector<bool> &knownObjects : robotKnownObjects) {
      std::fill(knownObjects.begin(), knownObjects.end(), true);
    }
  }
  initializeRelayObjectStates(dataset, setupNeighbors, d, initialStates,
                              robotKnownObjects);
  if (runObjectCopyJacobiInit) {
    objectJacobiStats = refineObjectCopiesWithDistributedChordalJacobi(
        dataset, setupNeighbors, robotObservedObjects, robotKnownObjects,
        observedConsensusConfidence, relayConsensusConfidence, d,
        initialStates);
    initializationCommMB += objectJacobiStats.commMb;
  }
  const size_t knownObjectCopies =
      countKnownObjectCopies(robotKnownObjects);
  const size_t relayObjectCopies =
      countRelayObjectCopies(robotKnownObjects, robotObservedObjects);
  cout << "Object known copies = " << knownObjectCopies << "/"
       << numRobots * dataset.numObjects
       << " | relay-only copies = " << relayObjectCopies << "." << endl;

  const string initSummaryPath =
      readEnvString("DPGO_OBJECT_INIT_SUMMARY_OUTPUT", "");
  if (!initSummaryPath.empty()) {
    ofstream initSummary(initSummaryPath);
    if (!initSummary.is_open()) {
      cerr << "Could not write initialization summary: "
           << initSummaryPath << endl;
    } else {
      initSummary
          << "init_mode,init_topology_source,init_topology,"
             "init_ring_hops,init_topology_file,"
             "init_topology_weight_mode,"
             "known_object_copies,relay_object_copies,"
             "object_jacobi_scope,"
             "object_jacobi_rotation_iters,object_jacobi_translation_iters,"
             "object_jacobi_rotation_converged,"
             "object_jacobi_translation_converged,"
             "object_jacobi_adaptive_budget,"
             "object_jacobi_min_iters,"
             "object_jacobi_budget_step_tol,"
             "object_jacobi_relative_step_drop_tol,"
             "object_jacobi_rotation_stop_reason,"
             "object_jacobi_translation_stop_reason,"
             "object_jacobi_rotation_step_norm,"
             "object_jacobi_translation_step_norm,"
             "object_jacobi_local_objective,"
             "object_jacobi_object_local_objective,"
             "object_jacobi_fixed_trajectory_objective,"
             "object_jacobi_consensus_objective,"
             "object_jacobi_scope_objective,"
             "object_jacobi_objective,"
             "object_jacobi_active_directed_pairs,"
             "object_jacobi_updated_variables,"
             "object_jacobi_consensus_kappa,"
             "object_jacobi_consensus_tau,"
             "object_jacobi_comm_mb,"
             "frame_sync_mode,frame_sync_solver,"
             "frame_sync_connected_robots,frame_sync_used_edges,"
             "frame_sync_candidate_edges,frame_sync_objective,"
             "frame_sync_registration_comm_mb,frame_sync_solver_comm_mb,"
             "frame_sync_comm_mb,frame_sync_rotation_iters,"
             "frame_sync_translation_iters,frame_sync_rotation_converged,"
             "frame_sync_translation_converged,"
             "frame_sync_mean_translation_rms,frame_sync_mean_rotation_rms,"
             "frame_sync_mean_score,frame_sync_max_score,"
             "frame_sync_edge_policy,frame_sync_cycle_rejected_edges,"
             "initialization_comm_mb\n";
      initSummary << initMode << ',' << initTopologySource << ','
                  << effectiveInitTopology << ','
                  << effectiveInitRingHops << ','
                  << (effectiveInitTopologyFile.empty()
                          ? string("<none>")
                          : effectiveInitTopologyFile)
                  << ',' << initTopologyWeightMode << ','
                  << knownObjectCopies << ','
                  << relayObjectCopies << ','
                  << objectJacobiStats.scope << ','
                  << objectJacobiStats.rotationIters << ','
                  << objectJacobiStats.translationIters << ','
                  << (objectJacobiStats.rotationConverged ? "true" : "false")
                  << ','
                  << (objectJacobiStats.translationConverged ? "true"
                                                             : "false")
                  << ','
                  << (objectJacobiStats.adaptiveBudget ? "true" : "false")
                  << ',' << objectJacobiStats.adaptiveMinIters << ','
                  << objectJacobiStats.adaptiveStepTol << ','
                  << objectJacobiStats.adaptiveRelativeStepDropTol << ','
                  << objectJacobiStats.rotationStopReason << ','
                  << objectJacobiStats.translationStopReason
                  << ',' << objectJacobiStats.rotationStepNorm << ','
                  << objectJacobiStats.translationStepNorm << ','
                  << objectJacobiStats.localObjective << ','
                  << objectJacobiStats.objectLocalObjective << ','
                  << objectJacobiStats.fixedTrajectoryObjective << ','
                  << objectJacobiStats.consensusObjective << ','
                  << objectJacobiStats.scopeObjective << ','
                  << objectJacobiStats.objective << ','
                  << objectJacobiStats.activeDirectedObjectPairs << ','
                  << objectJacobiStats.updatedVariables << ','
                  << objectJacobiStats.consensusKappa << ','
                  << objectJacobiStats.consensusTau << ','
                  << objectJacobiStats.commMb << ','
                  << initFrameSyncStats.mode << ','
                  << initFrameSyncStats.solver.solver << ','
                  << initFrameSyncStats.connectedRobots << ','
                  << initFrameSyncStats.usedEdges << ','
                  << initFrameSyncStats.candidateEdges << ','
                  << initFrameSyncStats.solver.objective << ','
                  << initFrameSyncStats.registrationCommMb << ','
                  << initFrameSyncStats.solverCommMb << ','
                  << initFrameSyncStats.commMb << ','
                  << initFrameSyncStats.solver.rotationIters << ','
                  << initFrameSyncStats.solver.translationIters << ','
                  << (initFrameSyncStats.solver.rotationConverged ? "true"
                                                                  : "false")
                  << ','
                  << (initFrameSyncStats.solver.translationConverged
                          ? "true"
                          : "false")
                  << ',' << initFrameSyncStats.meanTranslationRms << ','
                  << initFrameSyncStats.meanRotationRms << ','
                  << initFrameSyncStats.meanScore << ','
                  << initFrameSyncStats.maxScore << ','
                  << initFrameSyncStats.edgePolicy << ','
                  << initFrameSyncStats.edgeSelection.rejectedCycleEdges
                  << ','
                  << initializationCommMB << '\n';
    }
  }

  EdgeObjectTargets edgeObjectTargets;

  for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
    const ObjectPGORobotData &robot = dataset.robots[robotId];
    vector<RelativeSEMeasurement> localMeasurements =
        objectAwareLocalMeasurements(robot);

    const size_t localNumPoses = robot.trajectoryVertexIds.size() + dataset.numObjects;
    SparseMatrix localQ =
        fullConnectionLaplacian(localMeasurements, localNumPoses, d);
    robotLocalMeasurements.push_back(localMeasurements);

    auto problem = make_unique<QuadraticProblem>(localNumPoses, d, r);
    problem->setQ(localQ);
    problem->setG(SparseMatrix(r, (d + 1) * localNumPoses));

    auto optimizer = make_unique<QuadraticOptimizer>(problem.get());
    optimizer->setAlgorithm(ROPTALG::RTR);
    optimizer->setTrustRegionIterations(localTrustRegionIterations);
    optimizer->setTrustRegionTolerance(localTrustRegionTolerance);
    optimizer->setTrustRegionInitialRadius(10.0);
    optimizer->setTrustRegionMaxInnerIterations(30);
    optimizer->setVerbose(false);

    unique_ptr<GeodesicObjectProblem> geodesicProblem;
    unique_ptr<RoptOptimizer> geodesicOptimizer;
    if (localObjective == ObjectLocalObjective::Geodesic) {
      geodesicProblem = make_unique<GeodesicObjectProblem>(localNumPoses);
      geodesicProblem->setMeasurements(
          makeGeodesicMeasurementTerms(localMeasurements, localNumPoses));
      geodesicOptimizer =
          make_unique<RoptOptimizer>(geodesicProblem.get(), localNumPoses, d, r);
      geodesicOptimizer->setAlgorithm(ROPTALG::RTR);
      geodesicOptimizer->setTrustRegionIterations(
          static_cast<unsigned>(geodesicLocalIterations));
      geodesicOptimizer->setTrustRegionTolerance(geodesicLocalTolerance);
      geodesicOptimizer->setTrustRegionInitialRadius(
          geodesicTrustRegionInitialRadius);
      geodesicOptimizer->setTrustRegionMaxInnerIterations(
          geodesicTrustRegionMaxInnerIterations);
      geodesicOptimizer->setVerbose(false);
    }

    Matrix T0 = initialStates[robotId];
    Matrix Y0 = lift * T0;

    baseQs.push_back(localQ);
    problems.push_back(std::move(problem));
    optimizers.push_back(std::move(optimizer));
    geodesicProblems.push_back(std::move(geodesicProblem));
    geodesicOptimizers.push_back(std::move(geodesicOptimizer));
    currentEstimates.push_back(std::move(Y0));
    consensuses.emplace_back(d, r);
    neighborLists.push_back(runtimeNeighbors[robotId]);
    neighborObjects.emplace_back();
    neighborObjectUpdateIters.emplace_back();
    neighborAuxiliaries.emplace_back();
    neighborGeodesicAuxiliaries.emplace_back();
    outgoingObjectStates.emplace_back();
  }

  vector<Matrix> previousObjectGradients;
  vector<Matrix> distributedTrackers;
  vector<Matrix> previousDiffusionPsi;
  vector<Matrix> outgoingAuxiliaries;
  vector<Matrix> outgoingGeodesicAuxiliaries;
  if (usesDistributedObjectCorrection(consensusMode) || residualFeedback) {
    previousObjectGradients = computeLocalObjectGradients(
        dataset, baseQs, currentEstimates, robotKnownObjects, d);
    if (consensusMode == ObjectConsensusMode::GradientTracking) {
      distributedTrackers = previousObjectGradients;
      outgoingAuxiliaries = distributedTrackers;
    } else if (consensusMode == ObjectConsensusMode::ExactDiffusion) {
      previousDiffusionPsi = currentEstimates;
      outgoingAuxiliaries = currentEstimates;
    } else if (residualFeedback) {
      outgoingAuxiliaries = previousObjectGradients;
    }
  }
  if (usesGeodesicAuxiliaryExchange) {
    outgoingGeodesicAuxiliaries =
        computeLocalObjectGeodesicTangentGradients(
            dataset, robotLocalMeasurements, currentEstimates,
            robotKnownObjects, robotObservedObjects, d);
  }

  if (consensusMode == ObjectConsensusMode::EdgeAdmm) {
    edgeObjectTargets = initializeEdgeObjectTargets(
        dataset, runtimeNeighbors, robotKnownObjects, currentEstimates, d);
    cout << "Initialized edge object targets for "
         << edgeObjectTargets.size() << " communication edges." << endl;
  }

  for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
    for (const WeightedNeighbor &neighbor : neighborLists[robotId]) {
      const unsigned neighborId = neighbor.id;
      const size_t neighborObjectOffset =
          dataset.robots[neighborId].trajectoryVertexIds.size();
      vector<Matrix> cachedObjects(dataset.numObjects);
      vector<size_t> cachedObjectUpdateIters(dataset.numObjects, 0);
      vector<Matrix> cachedAuxiliaries(dataset.numObjects);
      vector<Matrix> cachedGeodesicAuxiliaries(dataset.numObjects);
      for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
        if (!robotKnownObjects[neighborId][objectId]) {
          continue;
        }
        cachedObjects[objectId] = blockPose(currentEstimates[neighborId],
                                            neighborObjectOffset + objectId, d);
        if (usesAuxiliaryExchange) {
          cachedAuxiliaries[objectId] =
              blockPose(outgoingAuxiliaries[neighborId],
                        neighborObjectOffset + objectId, d);
        }
        if (usesGeodesicAuxiliaryExchange &&
            neighborId < outgoingGeodesicAuxiliaries.size() &&
            outgoingGeodesicAuxiliaries[neighborId].rows() == 6 &&
            objectId < static_cast<size_t>(
                           outgoingGeodesicAuxiliaries[neighborId].cols())) {
          cachedGeodesicAuxiliaries[objectId] =
              outgoingGeodesicAuxiliaries[neighborId].block(0, objectId, 6, 1);
        }
      }
      neighborObjects[robotId].emplace(neighborId, std::move(cachedObjects));
      neighborObjectUpdateIters[robotId].emplace(
          neighborId, std::move(cachedObjectUpdateIters));
      if (usesAuxiliaryExchange) {
        neighborAuxiliaries[robotId].emplace(neighborId,
                                             std::move(cachedAuxiliaries));
      }
      if (usesGeodesicAuxiliaryExchange) {
        neighborGeodesicAuxiliaries[robotId].emplace(
            neighborId, std::move(cachedGeodesicAuxiliaries));
      }

      ObjectSendState sendState;
      sendState.lastSentPoses.resize(dataset.numObjects);
      sendState.lastSentAuxiliaries.resize(dataset.numObjects);
      sendState.lastSentGeodesicAuxiliaries.resize(dataset.numObjects);
      sendState.ages.assign(dataset.numObjects, 0);
      sendState.geodesicAuxAges.assign(dataset.numObjects, 0);
      sendState.initialized.assign(dataset.numObjects, false);
      sendState.geodesicAuxInitialized.assign(dataset.numObjects, false);
      outgoingObjectStates[robotId].emplace(neighborId, std::move(sendState));
    }
  }

  cout << "Running " << maxIters << " outer iterations..." << endl;

  size_t cumulativeObjectCommPoses = 0;
  double cumulativeCommMB = 0.0;
  double cumulativeTimeSec = 0.0;
  vector<vector<SparseMatrix>> scheduledQs(
      topologySchedule.size(), vector<SparseMatrix>(numRobots));
  for (size_t roundIdx = 0; roundIdx < topologySchedule.size(); ++roundIdx) {
    const WeightedNeighborRound &round = topologySchedule[roundIdx];
    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      SparseMatrix localQ = baseQs[robotId];
      const size_t objectOffset =
          dataset.robots[robotId].trajectoryVertexIds.size();
      if (usesQuadraticConsensus(consensusMode)) {
        for (const WeightedNeighbor &neighbor : round[robotId]) {
          const unsigned neighborId = neighbor.id;
          for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
          if (!robotKnownObjects[robotId][objectId] ||
              !robotKnownObjects[neighborId][objectId]) {
            continue;
          }
          const double edgeWeight = effectiveConsensusWeight(
              neighbor.weight, robotId, neighborId, objectId,
              robotObservedObjects, robotKnownObjects,
              observedConsensusConfidence, relayConsensusConfidence);
          if (edgeWeight <= 0.0) {
            continue;
          }
          const size_t localObjectIndex = objectOffset + objectId;
          const size_t colOffset = localObjectIndex * blockWidth;
          for (unsigned c = 0; c < blockWidth; ++c) {
            localQ.coeffRef(colOffset + c, colOffset + c) +=
                beta * edgeWeight * edgeWeight;
          }
        }
      }
      }
      localQ.makeCompressed();
      scheduledQs[roundIdx][robotId] = std::move(localQ);
    }
  }
  vector<size_t> lastQRound(numRobots,
                            std::numeric_limits<size_t>::max());

  for (unsigned iter = 0; iter < maxIters; ++iter) {
    const auto iterStart = std::chrono::steady_clock::now();
    const size_t roundIdx = iter % topologySchedule.size();
    const WeightedNeighborRound &activeNeighbors =
        topologySchedule[roundIdx];
    FrameSyncRunStats iterFrameSyncStats;
    if (frameSyncRuntimePeriod > 0 && iter > 0 &&
        iter % frameSyncRuntimePeriod == 0) {
      iterFrameSyncStats = synchronizeLocalChordalFramesWithGraph(
          dataset, activeNeighbors, d, currentEstimates,
          "disco_style_runtime", frameSyncRuntimeCountRegistration);
    }
    const vector<Matrix> previousEstimates = currentEstimates;
    size_t iterObjectCommPoses = 0;
    size_t iterObjectCommPayloadBlocks = 0;
    size_t iterInnovationGateSkipped = 0;
    size_t iterInnovationGateForced = 0;
    size_t iterReceiverBoundaryUpdates = 0;
    size_t iterReceiverBoundaryColdStarts = 0;
    double iterReceiverBoundaryCacheDeltaSum = 0.0;
    double iterReceiverBoundaryCacheDeltaMax = 0.0;
    double iterReceiverBoundaryDisagreementSum = 0.0;
    double iterReceiverBoundaryDisagreementMax = 0.0;
    size_t iterBoundaryBudgetCandidates = 0;
    size_t iterBoundaryBudgetSelected = 0;
    size_t iterBoundaryBudgetForced = 0;
    size_t iterBoundaryBudgetSkipped = 0;
    double iterBoundaryBudgetScoreSum = 0.0;
    double iterBoundaryBudgetSelectedScoreSum = 0.0;
    double iterBoundaryPredictiveGainSum = 0.0;
    double iterBoundaryPredictiveSelectedGainSum = 0.0;
    double iterBoundaryPredictiveStiffnessSum = 0.0;
    double iterBoundaryPredictiveGainTargetSum = 0.0;
    double iterBoundaryPredictiveSelectedGainFraction = 0.0;
    size_t iterActiveConsensusPairs = 0;
    size_t iterRelayClosedFormUpdates = 0;
    size_t iterConsensusCorrectionUpdates = 0;
    size_t iterConsensusRejectedUpdates = 0;
    size_t iterEdgeTargetUpdates = 0;
    double iterCommMB = 0.0;
    double iterLocalModelCost = 0.0;
    double iterMeasurementCost = 0.0;
    double iterChordalMeasurementCost = 0.0;
    double iterGeodesicMeasurementCost = 0.0;
    double iterConsensusCost = 0.0;
    double iterPrimalResidualSq = 0.0;
    double iterDualResidualSq = 0.0;
    double iterTotalGradNorm = 0.0;
    double iterDistributedStepNorm = 0.0;
    double iterDistributedAuxNorm = 0.0;
    size_t iterGaugeMeritFilteredUpdates = 0;
    size_t iterGaugeMeritRejectedUpdates = 0;
    double iterGaugeMeritStepScaleSum = 0.0;
    size_t iterStaleCacheSkipped = 0;
    size_t iterMainMeritFilteredUpdates = 0;
    size_t iterMainMeritRejectedUpdates = 0;
    double iterMainMeritStepScaleSum = 0.0;
    size_t iterGeodesicStepLimitedUpdates = 0;
    double iterGeodesicStepScaleSum = 0.0;
    double iterGeodesicMaxBlockDelta = 0.0;
    size_t iterResidualFeedbackPayloadBlocks = 0;
    size_t iterGeodesicAuxPayloads = 0;
    vector<Matrix> boundaryPredictiveGradients;
    if (predictiveBoundaryBudgetCommunication &&
        localObjective == ObjectLocalObjective::Chordal) {
      boundaryPredictiveGradients = computeLocalObjectGradients(
          dataset, baseQs, previousEstimates, robotKnownObjects, d);
    }
    vector<BoundaryBudgetCandidate> boundaryBudgetCandidates;

    auto deliverObjectPayload =
        [&](const BoundaryBudgetCandidate &candidate) {
          ObjectSendState &sendState =
              outgoingObjectStates[candidate.senderId].at(
                  candidate.receiverId);
          vector<Matrix> &neighborCache =
              neighborObjects[candidate.receiverId].at(candidate.senderId);
          vector<size_t> &neighborUpdateIters =
              neighborObjectUpdateIters[candidate.receiverId].at(
                  candidate.senderId);
          const bool sendPosePayload =
              candidate.sendObject || candidate.sendAuxiliary ||
              candidate.sendGeodesicAuxiliary;
          if (sendPosePayload) {
            const Matrix oldReceiverCache =
                neighborCache[candidate.objectId];
            if (oldReceiverCache.size() == 0) {
              ++iterReceiverBoundaryColdStarts;
            } else {
              const double cacheDelta = poseBlockDeltaNorm(
                  oldReceiverCache, candidate.localObject, localObjective);
              iterReceiverBoundaryCacheDeltaSum += cacheDelta;
              iterReceiverBoundaryCacheDeltaMax =
                  std::max(iterReceiverBoundaryCacheDeltaMax, cacheDelta);
            }
            iterReceiverBoundaryDisagreementSum +=
                candidate.receiverDisagreement;
            iterReceiverBoundaryDisagreementMax = std::max(
                iterReceiverBoundaryDisagreementMax,
                candidate.receiverDisagreement);
            ++iterReceiverBoundaryUpdates;
            neighborCache[candidate.objectId] = candidate.localObject;
            neighborUpdateIters[candidate.objectId] = iter;
            if (candidate.sendPreCorrectionAuxiliary) {
              vector<Matrix> &neighborAuxiliaryCache =
                  neighborAuxiliaries[candidate.receiverId].at(
                      candidate.senderId);
              neighborAuxiliaryCache[candidate.objectId] =
                  candidate.localAuxiliary;
              sendState.lastSentAuxiliaries[candidate.objectId] =
                  candidate.localAuxiliary;
              iterObjectCommPayloadBlocks += 2;
              if (residualFeedback) {
                ++iterResidualFeedbackPayloadBlocks;
              }
            } else {
              ++iterObjectCommPayloadBlocks;
            }
            sendState.lastSentPoses[candidate.objectId] =
                candidate.localObject;
            sendState.ages[candidate.objectId] = 0;
            sendState.initialized[candidate.objectId] = true;
            ++iterObjectCommPoses;
          } else {
            ++sendState.ages[candidate.objectId];
          }
          if (candidate.sendGeodesicAuxiliary) {
            vector<Matrix> &neighborGeodesicAuxiliaryCache =
                neighborGeodesicAuxiliaries[candidate.receiverId].at(
                    candidate.senderId);
            neighborGeodesicAuxiliaryCache[candidate.objectId] =
                candidate.localGeodesicAuxiliary;
            sendState.lastSentGeodesicAuxiliaries[candidate.objectId] =
                candidate.localGeodesicAuxiliary;
            sendState.geodesicAuxAges[candidate.objectId] = 0;
            sendState.geodesicAuxInitialized[candidate.objectId] = true;
            ++iterGeodesicAuxPayloads;
          } else if (usesGeodesicAuxiliaryExchange) {
            ++sendState.geodesicAuxAges[candidate.objectId];
          }
        };

    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      const Matrix &localState = previousEstimates[robotId];
      const size_t objectOffset =
          dataset.robots[robotId].trajectoryVertexIds.size();
      for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
        const unsigned neighborId = neighbor.id;
        ObjectSendState &sendState = outgoingObjectStates[robotId].at(neighborId);
        for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
          if (!robotKnownObjects[robotId][objectId] ||
              !robotKnownObjects[neighborId][objectId]) {
            continue;
          }
          const double edgeWeight = effectiveConsensusWeight(
              neighbor.weight, robotId, neighborId, objectId,
              robotObservedObjects, robotKnownObjects,
              observedConsensusConfidence, relayConsensusConfidence);
          if (edgeWeight <= 0.0) {
            continue;
          }
          const bool sendPreCorrectionAuxiliary =
              consensusMode == ObjectConsensusMode::GradientTracking ||
              residualFeedback;
          const size_t localObjectIndex = objectOffset + objectId;
          const Matrix localObject = blockPose(localState, localObjectIndex, d);
          Matrix localAuxiliary;
          if (sendPreCorrectionAuxiliary) {
            localAuxiliary = blockPose(outgoingAuxiliaries[robotId],
                                       localObjectIndex, d);
          }
          Matrix localGeodesicAuxiliary;
          if (usesGeodesicAuxiliaryExchange &&
              robotId < outgoingGeodesicAuxiliaries.size() &&
              outgoingGeodesicAuxiliaries[robotId].rows() == 6 &&
              objectId < static_cast<size_t>(
                             outgoingGeodesicAuxiliaries[robotId].cols())) {
            localGeodesicAuxiliary =
                outgoingGeodesicAuxiliaries[robotId].block(0, objectId, 6, 1);
          }
          const bool baseSendObject =
              !sendState.initialized[objectId] ||
              shouldSendObjectPose(localObject,
                                   sendState.lastSentPoses[objectId],
                                   objectPoseTol, sendState.ages[objectId],
                                   objectMaxAge);
          bool baseSendAuxiliary = false;
          if (sendPreCorrectionAuxiliary) {
            baseSendAuxiliary =
                !sendState.initialized[objectId] ||
                shouldSendObjectPose(localAuxiliary,
                                     sendState.lastSentAuxiliaries[objectId],
                                     objectPoseTol, sendState.ages[objectId],
                                     objectMaxAge);
          }
          bool baseSendGeodesicAuxiliary = false;
          if (usesGeodesicAuxiliaryExchange &&
              localGeodesicAuxiliary.size() != 0) {
            baseSendGeodesicAuxiliary =
                baseSendObject ||
                (!geodesicAuxPoseTriggeredOnly &&
                 (!sendState.geodesicAuxInitialized[objectId] ||
                  shouldSendObjectPose(
                      localGeodesicAuxiliary,
                      sendState.lastSentGeodesicAuxiliaries[objectId],
                      objectPoseTol, sendState.geodesicAuxAges[objectId],
                      objectMaxAge)));
          }
          bool hasNeighborPrediction = false;
          double innovationNorm =
              std::numeric_limits<double>::infinity();
          const auto senderNeighborCacheIt =
              neighborObjects[robotId].find(neighborId);
          if (senderNeighborCacheIt != neighborObjects[robotId].end() &&
              objectId < senderNeighborCacheIt->second.size() &&
              senderNeighborCacheIt->second[objectId].size() != 0) {
            hasNeighborPrediction = true;
            innovationNorm = poseBlockDeltaNorm(
                localObject, senderNeighborCacheIt->second[objectId],
                localObjective);
          }
          const bool forcedObjectRefresh =
              !sendState.initialized[objectId] ||
              sendState.ages[objectId] >= objectMaxAge ||
              !hasNeighborPrediction;
          const bool posePayloadRequested =
              baseSendObject || baseSendAuxiliary ||
              baseSendGeodesicAuxiliary;
          const bool innovationGateAllows =
              !innovationGatedCommunication || forcedObjectRefresh ||
              innovationNorm >= innovationThreshold;
          if (innovationGatedCommunication && posePayloadRequested) {
            if (forcedObjectRefresh) {
              ++iterInnovationGateForced;
            } else if (!innovationGateAllows) {
              ++iterInnovationGateSkipped;
            }
          }
          const bool sendObject = baseSendObject && innovationGateAllows;
          const bool sendAuxiliary =
              baseSendAuxiliary && innovationGateAllows;
          const bool sendGeodesicAuxiliary =
              baseSendGeodesicAuxiliary && innovationGateAllows;
          if (sendObject || sendAuxiliary || sendGeodesicAuxiliary) {
            const size_t receiverObjectOffset =
                dataset.robots[neighborId].trajectoryVertexIds.size();
            const Matrix receiverLocalObject =
                blockPose(previousEstimates[neighborId],
                          receiverObjectOffset + objectId, d);
            BoundaryBudgetCandidate candidate;
            candidate.senderId = robotId;
            candidate.receiverId = neighborId;
            candidate.objectId = objectId;
            candidate.localObject = localObject;
            candidate.localAuxiliary = localAuxiliary;
            candidate.localGeodesicAuxiliary = localGeodesicAuxiliary;
            candidate.sendPreCorrectionAuxiliary =
                sendPreCorrectionAuxiliary;
            candidate.sendObject = sendObject;
            candidate.sendAuxiliary = sendAuxiliary;
            candidate.sendGeodesicAuxiliary = sendGeodesicAuxiliary;
            candidate.forced =
                !sendState.initialized[objectId] ||
                sendState.ages[objectId] >= objectMaxAge;
            candidate.receiverDisagreement = poseBlockDeltaNorm(
                receiverLocalObject, localObject, localObjective);
            candidate.score = candidate.receiverDisagreement;
            if (predictiveBoundaryBudgetCommunication &&
                localObjective == ObjectLocalObjective::Chordal &&
                neighborId < boundaryPredictiveGradients.size()) {
              Matrix oldReceiverCache = receiverLocalObject;
              const auto oldCacheIt =
                  neighborObjects[neighborId].find(robotId);
              if (oldCacheIt != neighborObjects[neighborId].end() &&
                  objectId < oldCacheIt->second.size() &&
                  oldCacheIt->second[objectId].size() != 0) {
                oldReceiverCache = oldCacheIt->second[objectId];
              }
              double receiverEdgeWeight = neighbor.weight;
              if (neighborId < activeNeighbors.size()) {
                const WeightedNeighbor *reverseNeighbor =
                    findWeightedNeighbor(activeNeighbors[neighborId],
                                         robotId);
                if (reverseNeighbor != nullptr) {
                  receiverEdgeWeight = reverseNeighbor->weight;
                }
              }
              const double receiverConsensusWeight =
                  effectiveConsensusWeight(
                      receiverEdgeWeight, neighborId, robotId, objectId,
                      robotObservedObjects, robotKnownObjects,
                      observedConsensusConfidence,
                      relayConsensusConfidence);
              const double consensusStiffness =
                  beta * receiverConsensusWeight * receiverConsensusWeight;
              const size_t receiverObjectIndex =
                  receiverObjectOffset + objectId;
              const size_t receiverColOffset =
                  receiverObjectIndex * blockWidth;
              const double measurementStiffness =
                  blockDiagonalMeanAbs(baseQs[neighborId],
                                       receiverColOffset, blockWidth);
              const double totalStiffness =
                  measurementStiffness + consensusStiffness;
              const double alpha =
                  totalStiffness > 1e-12
                      ? consensusStiffness / (totalStiffness + 1e-12)
                      : 0.0;
              Matrix gradientBlock =
                  Matrix::Zero(d, blockWidth);
              if (receiverColOffset + blockWidth <=
                  static_cast<size_t>(
                      boundaryPredictiveGradients[neighborId].cols())) {
                gradientBlock =
                    boundaryPredictiveGradients[neighborId].block(
                        0, receiverColOffset, d, blockWidth);
              }
              const Matrix oldStep =
                  alpha * (oldReceiverCache - receiverLocalObject);
              const Matrix newStep =
                  alpha * (localObject - receiverLocalObject);
              const Matrix stepChange = newStep - oldStep;
              const double linearGain =
                  -(gradientBlock.array() * stepChange.array()).sum();
              const double quadraticGain =
                  -0.5 * measurementStiffness *
                  (newStep.squaredNorm() - oldStep.squaredNorm());
              const double freshnessGain =
                  0.5 * consensusStiffness *
                  (localObject - oldReceiverCache).squaredNorm();
              candidate.predictedGain =
                  std::max(0.0,
                           linearGain + quadraticGain + freshnessGain);
              candidate.predictiveStiffness = totalStiffness;
              const size_t payloadBlocks =
                  std::max<size_t>(
                      1,
                      (sendPreCorrectionAuxiliary ? 2 : 1) +
                          (sendGeodesicAuxiliary ? 1 : 0));
              const double payloadBytes =
                  static_cast<double>(payloadBlocks * blockBytes);
              candidate.score =
                  payloadBytes > 0.0
                      ? candidate.predictedGain / payloadBytes
                      : candidate.predictedGain;
            }
            if (boundaryBudgetCommunication) {
              ++iterBoundaryBudgetCandidates;
              iterBoundaryBudgetScoreSum += candidate.score;
              iterBoundaryPredictiveGainSum += candidate.predictedGain;
              iterBoundaryPredictiveStiffnessSum +=
                  candidate.predictiveStiffness;
              boundaryBudgetCandidates.push_back(std::move(candidate));
            } else {
              deliverObjectPayload(candidate);
            }
          } else {
            ++sendState.ages[objectId];
            if (usesGeodesicAuxiliaryExchange) {
              ++sendState.geodesicAuxAges[objectId];
            }
          }
        }
      }
    }

    if (boundaryBudgetCommunication) {
      vector<size_t> budgetedCandidateIndices;
      budgetedCandidateIndices.reserve(boundaryBudgetCandidates.size());
      for (size_t i = 0; i < boundaryBudgetCandidates.size(); ++i) {
        if (boundaryBudgetCandidates[i].forced) {
          deliverObjectPayload(boundaryBudgetCandidates[i]);
          ++iterBoundaryBudgetForced;
        } else {
          budgetedCandidateIndices.push_back(i);
        }
      }

      std::sort(budgetedCandidateIndices.begin(),
                budgetedCandidateIndices.end(),
                [&](size_t lhs, size_t rhs) {
                  const BoundaryBudgetCandidate &a =
                      boundaryBudgetCandidates[lhs];
                  const BoundaryBudgetCandidate &b =
                      boundaryBudgetCandidates[rhs];
                  if (a.score != b.score) {
                    return a.score > b.score;
                  }
                  return std::tie(a.senderId, a.receiverId, a.objectId) <
                         std::tie(b.senderId, b.receiverId, b.objectId);
                });

      size_t budgetLimit = budgetedCandidateIndices.size();
      double budgetablePredictiveGain = 0.0;
      if (predictiveBoundaryBudgetCommunication &&
          hasBoundaryPredictiveGainFraction) {
        for (const size_t index : budgetedCandidateIndices) {
          budgetablePredictiveGain +=
              boundaryBudgetCandidates[index].predictedGain;
        }
        iterBoundaryPredictiveGainTargetSum =
            boundaryPredictiveGainFraction * budgetablePredictiveGain;
        if (budgetablePredictiveGain > 1e-12) {
          budgetLimit = 0;
          double selectedGain = 0.0;
          for (const size_t index : budgetedCandidateIndices) {
            if (selectedGain >= iterBoundaryPredictiveGainTargetSum) {
              break;
            }
            selectedGain += boundaryBudgetCandidates[index].predictedGain;
            ++budgetLimit;
          }
        }
      } else if (boundaryBudgetPoseFraction < 1.0) {
        budgetLimit = static_cast<size_t>(
            std::ceil(boundaryBudgetPoseFraction *
                      static_cast<double>(budgetedCandidateIndices.size())));
      }
      if (hasBoundaryBudgetMaxPoses) {
        budgetLimit = std::min(budgetLimit, boundaryBudgetMaxPoses);
      }
      budgetLimit = std::min(budgetLimit, budgetedCandidateIndices.size());

      for (size_t order = 0; order < budgetedCandidateIndices.size();
           ++order) {
        const BoundaryBudgetCandidate &candidate =
            boundaryBudgetCandidates[budgetedCandidateIndices[order]];
        if (order < budgetLimit) {
          deliverObjectPayload(candidate);
          ++iterBoundaryBudgetSelected;
          iterBoundaryBudgetSelectedScoreSum += candidate.score;
          iterBoundaryPredictiveSelectedGainSum +=
              candidate.predictedGain;
        } else {
          ObjectSendState &sendState =
              outgoingObjectStates[candidate.senderId].at(
                  candidate.receiverId);
          ++sendState.ages[candidate.objectId];
          if (usesGeodesicAuxiliaryExchange) {
            ++sendState.geodesicAuxAges[candidate.objectId];
          }
          ++iterBoundaryBudgetSkipped;
        }
      }
      const double selectedGainDenominator =
          budgetablePredictiveGain > 1e-12
              ? budgetablePredictiveGain
              : iterBoundaryPredictiveGainSum;
      if (selectedGainDenominator > 1e-12) {
        iterBoundaryPredictiveSelectedGainFraction =
            iterBoundaryPredictiveSelectedGainSum /
            selectedGainDenominator;
      }
    }

    auto buildQuadraticConsensusLinearTerm =
        [&](unsigned robotId, size_t localNumPoses,
            size_t objectOffset) {
          Matrix localG = Matrix::Zero(r, (d + 1) * localNumPoses);
          if (!usesQuadraticConsensus(consensusMode)) {
            return localG;
          }
          for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
            const unsigned neighborId = neighbor.id;
            const vector<Matrix> &neighborCache =
                neighborObjects[robotId].at(neighborId);
            for (size_t objectId = 0; objectId < dataset.numObjects;
                 ++objectId) {
              if (!robotKnownObjects[robotId][objectId] ||
                  !robotKnownObjects[neighborId][objectId]) {
                continue;
              }
              const double edgeWeight = effectiveConsensusWeight(
                  neighbor.weight, robotId, neighborId, objectId,
                  robotObservedObjects, robotKnownObjects,
                  observedConsensusConfidence, relayConsensusConfidence);
              if (edgeWeight <= 0.0) {
                continue;
              }
              const size_t localObjectIndex = objectOffset + objectId;
              const size_t colOffset = localObjectIndex * blockWidth;
              const bool useDual =
                  consensusMode == ObjectConsensusMode::DirectAdmm ||
                  consensusMode == ObjectConsensusMode::EdgeAdmm;
              const Matrix lambda =
                  useDual ? consensuses[robotId].getDual(neighborId, objectId)
                          : Matrix::Zero(r, blockWidth);
              Matrix consensusTarget;
              if (consensusMode == ObjectConsensusMode::EdgeAdmm) {
                const EdgeKey edge = makeEdgeKey(robotId, neighborId);
                const auto edgeIt = edgeObjectTargets.find(edge);
                if (edgeIt == edgeObjectTargets.end() ||
                    objectId >= edgeIt->second.size() ||
                    edgeIt->second[objectId].size() == 0) {
                  continue;
                }
                consensusTarget = edgeIt->second[objectId];
              } else {
                if (objectId >= neighborCache.size() ||
                    neighborCache[objectId].size() == 0) {
                  continue;
                }
                consensusTarget = neighborCache[objectId];
              }

              localG.block(0, colOffset, d, blockWidth) +=
                  -beta * edgeWeight * edgeWeight * consensusTarget +
                  edgeWeight * lambda;
              if (residualFeedback) {
                const auto auxIt =
                    neighborAuxiliaries[robotId].find(neighborId);
                if (auxIt != neighborAuxiliaries[robotId].end() &&
                    objectId < auxIt->second.size() &&
                    auxIt->second[objectId].size() != 0) {
                  localG.block(0, colOffset, d, blockWidth) +=
                      residualFeedbackScale * edgeWeight *
                      auxIt->second[objectId];
                }
              }
            }
          }
          return localG;
        };

    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      QuadraticProblem &problem = *problems[robotId];
      const ObjectPGORobotData &robot = dataset.robots[robotId];
      const size_t localNumPoses = robot.trajectoryVertexIds.size() + dataset.numObjects;
      const size_t objectOffset = robot.trajectoryVertexIds.size();

      if (localObjective == ObjectLocalObjective::Geodesic) {
        Matrix Yopt = previousEstimates[robotId];
        double localModelCost = 0.0;
        double localGradNorm = 0.0;
        if (geodesicChordalPredictor) {
          if (lastQRound[robotId] != roundIdx) {
            problem.setQ(scheduledQs[roundIdx][robotId]);
            lastQRound[robotId] = roundIdx;
          }
          Matrix localG = buildQuadraticConsensusLinearTerm(
              robotId, localNumPoses, objectOffset);
          SparseMatrix localGSparse = localG.sparseView();
          problem.setG(localGSparse);
          Yopt = optimizers[robotId]->optimize(Yopt);
        }
        if (geodesicSolverMode == GeodesicSolverMode::Gradient) {
          const GeodesicLocalSolveStats result = optimizeGeodesicLocalProblem(
              dataset, robotLocalMeasurements[robotId], activeNeighbors,
              robotKnownObjects, robotObservedObjects, neighborObjects,
              robotId, beta, observedConsensusConfidence,
              relayConsensusConfidence, d, geodesicLocalIterations,
              geodesicLocalTolerance, Yopt);
          localModelCost = result.costOpt;
          localGradNorm = result.gradNormOpt;
        } else {
          if (!geodesicProblems[robotId] || !geodesicOptimizers[robotId]) {
            throw runtime_error("Missing geodesic ROPTLIB problem for robot " +
                                to_string(robotId));
          }
          vector<GeodesicConsensusTerm> geodesicTerms =
              makeGeodesicConsensusTerms(
                  dataset, activeNeighbors, robotKnownObjects,
                  robotObservedObjects, neighborObjects,
                  neighborGeodesicAuxiliaries, robotId, beta,
                  observedConsensusConfidence, relayConsensusConfidence,
                  geodesicAuxMode, geodesicAuxScale);
          if (geodesicProximalWeight > 0.0) {
            vector<GeodesicConsensusTerm> proximalTerms =
                makeGeodesicProximalTerms(previousEstimates[robotId],
                                          geodesicProximalWeight);
            geodesicTerms.insert(geodesicTerms.end(),
                                 proximalTerms.begin(),
                                 proximalTerms.end());
          }
          geodesicProblems[robotId]->setConsensusTerms(geodesicTerms);
          Yopt = geodesicOptimizers[robotId]->optimize(Yopt);
          const ROPTResult result =
              geodesicOptimizers[robotId]->getOptResult();
          localModelCost = result.fOpt;
          localGradNorm = result.gradNormOpt;
          if (geodesicLocalMeritFilter) {
            const MainMeritFilterStats meritStats =
                applyGeodesicLocalModelMeritFilter(
                    previousEstimates[robotId], Yopt,
                    *geodesicProblems[robotId], d,
                    geodesicLocalMeritBacktrackingSteps,
                    geodesicLocalMeritTolerance);
            iterMainMeritFilteredUpdates += meritStats.filteredUpdates;
            iterMainMeritRejectedUpdates += meritStats.rejectedUpdates;
            iterMainMeritStepScaleSum += meritStats.acceptedStepScaleSum;
            if (meritStats.filteredUpdates > 0) {
              localModelCost = geodesicProblems[robotId]->f(Yopt);
              localGradNorm = geodesicProblems[robotId]->RieGradNorm(Yopt);
            }
          }
        }
        if (geodesicMaxBlockStep > 0.0) {
          double appliedScale = 1.0;
          double maxBlockDelta = 0.0;
          Matrix limitedY = limitGeodesicStateStep(
              previousEstimates[robotId], Yopt, geodesicMaxBlockStep,
              &appliedScale, &maxBlockDelta);
          iterGeodesicMaxBlockDelta =
              std::max(iterGeodesicMaxBlockDelta, maxBlockDelta);
          if (appliedScale < 1.0) {
            Yopt = std::move(limitedY);
            ++iterGeodesicStepLimitedUpdates;
            iterGeodesicStepScaleSum += appliedScale;
            if (geodesicSolverMode == GeodesicSolverMode::RtrGn &&
                geodesicProblems[robotId]) {
              localModelCost = geodesicProblems[robotId]->f(Yopt);
              localGradNorm = geodesicProblems[robotId]->RieGradNorm(Yopt);
            } else {
              localModelCost = evaluateGeodesicLocalObjectiveAndGradient(
                  dataset, robotLocalMeasurements[robotId], activeNeighbors,
                  robotKnownObjects, robotObservedObjects, neighborObjects,
                  robotId, beta, observedConsensusConfidence,
                  relayConsensusConfidence, d, Yopt, nullptr, nullptr);
            }
          }
        }
        if (mainMeritFilter) {
          const MainMeritFilterStats meritStats =
              applyMainMeasurementMeritFilter(
                  robot, previousEstimates[robotId], Yopt, localObjective, d,
                  mainMeritBacktrackingSteps, mainMeritTolerance);
          iterMainMeritFilteredUpdates += meritStats.filteredUpdates;
          iterMainMeritRejectedUpdates += meritStats.rejectedUpdates;
          iterMainMeritStepScaleSum += meritStats.acceptedStepScaleSum;
        }
        currentEstimates[robotId] = std::move(Yopt);
        iterLocalModelCost += localModelCost;
        iterTotalGradNorm += localGradNorm;
        continue;
      }

      Matrix localG = buildQuadraticConsensusLinearTerm(
          robotId, localNumPoses, objectOffset);

      if (lastQRound[robotId] != roundIdx) {
        problem.setQ(scheduledQs[roundIdx][robotId]);
        lastQRound[robotId] = roundIdx;
      }
      SparseMatrix localGSparse = localG.sparseView();
      problem.setG(localGSparse);

      Matrix Yopt = optimizers[robotId]->optimize(previousEstimates[robotId]);
      if (mainMeritFilter) {
        const MainMeritFilterStats meritStats =
            applyMainMeasurementMeritFilter(
                robot, previousEstimates[robotId], Yopt, localObjective, d,
                mainMeritBacktrackingSteps, mainMeritTolerance);
        iterMainMeritFilteredUpdates += meritStats.filteredUpdates;
        iterMainMeritRejectedUpdates += meritStats.rejectedUpdates;
        iterMainMeritStepScaleSum += meritStats.acceptedStepScaleSum;
      }
      currentEstimates[robotId] = Yopt;

      const ROPTResult result = optimizers[robotId]->getOptResult();
      iterLocalModelCost += result.fOpt;
      iterTotalGradNorm += result.gradNormOpt;
    }

    if (gaugeCoupling) {
      const DistributedCorrectionStats correctionStats =
          applyNeighborObjectGaugeCoupling(
              dataset, activeNeighbors, robotObservedObjects, robotKnownObjects,
              neighborObjects, neighborObjectUpdateIters, gaugeCouplingAlpha,
              gaugeCouplingMaxStep, gaugeCouplingMinObjects,
              gaugeCouplingAnchorRoot, gaugeCouplingObservedOnly,
              observedConsensusConfidence, relayConsensusConfidence, iter,
              cacheFreshnessMaxAge, cacheFreshnessDecay, gaugeMeritFilter,
              gaugeMeritBacktrackingSteps, gaugeMeritTolerance,
              gaugeMeritConsensusWeight, localObjective, d,
              currentEstimates);
      iterConsensusCorrectionUpdates += correctionStats.updated;
      iterGaugeMeritFilteredUpdates +=
          gaugeMeritFilter ? correctionStats.updated : 0;
      iterGaugeMeritRejectedUpdates += correctionStats.rejected;
      iterGaugeMeritStepScaleSum += correctionStats.acceptedStepScaleSum;
      iterStaleCacheSkipped += correctionStats.staleSkipped;
      iterDistributedStepNorm += correctionStats.stepNorm;
      iterDistributedAuxNorm += correctionStats.auxiliaryNorm;
    }

    if (consensusMode == ObjectConsensusMode::ProxMixing) {
      const ConsensusCorrectionStats correctionStats =
          applyProximalObjectMixing(
          dataset, activeNeighbors, robotObservedObjects, robotKnownObjects,
          neighborObjects, consensusAlpha, observedConsensusConfidence,
          relayConsensusConfidence, consensusBacktrackingSteps,
          consensusMeritTolerance, d, currentEstimates);
      iterConsensusCorrectionUpdates += correctionStats.updated;
      iterConsensusRejectedUpdates += correctionStats.rejected;
    }

    if (consensusMode == ObjectConsensusMode::GradientTracking) {
      const DistributedCorrectionStats correctionStats =
          applyRiemannianGradientTracking(
              dataset, activeNeighbors, robotKnownObjects, robotObservedObjects,
              neighborObjects, neighborAuxiliaries, baseQs, consensusAlpha,
              observedConsensusConfidence, relayConsensusConfidence,
              topologyWeightMode, d, currentEstimates, distributedTrackers,
              previousObjectGradients, outgoingAuxiliaries);
      iterConsensusCorrectionUpdates += correctionStats.updated;
      iterDistributedStepNorm += correctionStats.stepNorm;
      iterDistributedAuxNorm += correctionStats.auxiliaryNorm;
    }

    if (consensusMode == ObjectConsensusMode::ExactDiffusion) {
      const DistributedCorrectionStats correctionStats =
          applyRiemannianExactDiffusion(
              dataset, activeNeighbors, robotKnownObjects, robotObservedObjects,
              baseQs, consensusAlpha, observedConsensusConfidence,
              relayConsensusConfidence, topologyWeightMode, d,
              currentEstimates, previousDiffusionPsi, outgoingAuxiliaries);
      iterConsensusCorrectionUpdates += correctionStats.updated;
      iterDistributedStepNorm += correctionStats.stepNorm;
      iterDistributedAuxNorm += correctionStats.auxiliaryNorm;
      iterObjectCommPayloadBlocks += countActiveKnownObjectPairs(
          activeNeighbors, robotKnownObjects, robotObservedObjects,
          observedConsensusConfidence, relayConsensusConfidence);
    }

    if (residualFeedback) {
      outgoingAuxiliaries = computeLocalObjectGradients(
          dataset, baseQs, currentEstimates, robotKnownObjects, d);
    }
    if (usesGeodesicAuxiliaryExchange) {
      outgoingGeodesicAuxiliaries =
          computeLocalObjectGeodesicTangentGradients(
              dataset, robotLocalMeasurements, currentEstimates,
              robotKnownObjects, robotObservedObjects, d);
    }

    if (relayBarycentricUpdates) {
      iterRelayClosedFormUpdates = applyRelayConsensusAveraging(
          dataset, activeNeighbors, robotObservedObjects, robotKnownObjects,
          neighborObjects, consensuses, beta, d, currentEstimates);
    }

    if (consensusMode == ObjectConsensusMode::EdgeAdmm) {
      iterEdgeTargetUpdates = updateEdgeObjectTargets(
          dataset, activeNeighbors, robotObservedObjects, robotKnownObjects,
          consensuses, currentEstimates, beta, observedConsensusConfidence,
          relayConsensusConfidence, d, edgeObjectTargets);
    }

    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      const ObjectPGORobotData &robot = dataset.robots[robotId];
      const double chordalCost =
          computeOriginalMeasurementCost(robot, currentEstimates[robotId], d);
      iterChordalMeasurementCost += chordalCost;
      double geodesicCost = std::numeric_limits<double>::quiet_NaN();
      if (d == 3) {
        geodesicCost = computeGeodesicOriginalMeasurementCost(
            robot, currentEstimates[robotId], d);
        iterGeodesicMeasurementCost += geodesicCost;
      }
      if (localObjective == ObjectLocalObjective::Geodesic) {
        iterMeasurementCost += geodesicCost;
      } else {
        iterMeasurementCost += chordalCost;
      }
    }

    for (unsigned robotId = 0; robotId < numRobots; ++robotId) {
      const ObjectPGORobotData &robot = dataset.robots[robotId];
      const Matrix &localState = currentEstimates[robotId];
      for (const WeightedNeighbor &neighbor : activeNeighbors[robotId]) {
        const unsigned neighborId = neighbor.id;
        const auto neighborCacheIt = neighborObjects[robotId].find(neighborId);
        const vector<Matrix> *neighborCache =
            neighborCacheIt == neighborObjects[robotId].end()
                ? nullptr
                : &neighborCacheIt->second;
        for (size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
          if (!robotKnownObjects[robotId][objectId] ||
              !robotKnownObjects[neighborId][objectId]) {
            continue;
          }
          const double edgeWeight = effectiveConsensusWeight(
              neighbor.weight, robotId, neighborId, objectId,
              robotObservedObjects, robotKnownObjects,
              observedConsensusConfidence, relayConsensusConfidence);
          if (edgeWeight <= 0.0) {
            continue;
          }
          const size_t localObjectIndex = robot.trajectoryVertexIds.size() + objectId;
          const Matrix localObject = blockPose(localState, localObjectIndex, d);
          Matrix consensusTarget;
          if (consensusMode == ObjectConsensusMode::EdgeAdmm) {
            const EdgeKey edge = makeEdgeKey(robotId, neighborId);
            const auto edgeIt = edgeObjectTargets.find(edge);
            if (edgeIt == edgeObjectTargets.end() ||
                objectId >= edgeIt->second.size() ||
                edgeIt->second[objectId].size() == 0) {
              continue;
            }
            consensusTarget = edgeIt->second[objectId];
          } else {
            if (neighborCache == nullptr || objectId >= neighborCache->size() ||
                (*neighborCache)[objectId].size() == 0) {
              continue;
            }
            consensusTarget = (*neighborCache)[objectId];
          }
          ++iterActiveConsensusPairs;
          double pairPrimalResidual = 0.0;
          if (localObjective == ObjectLocalObjective::Geodesic) {
            iterConsensusCost += evaluateGeodesicConsensusPairCost(
                localObject, consensusTarget, beta, edgeWeight,
                pairPrimalResidual);
          } else {
            iterConsensusCost +=
                consensuses[robotId].evaluateConsensusCost(
                    neighborId, objectId, localObject, consensusTarget, beta,
                    edgeWeight,
                    pairPrimalResidual);
          }
          iterPrimalResidualSq += pairPrimalResidual * pairPrimalResidual;
          const Matrix previousObject = blockPose(previousEstimates[robotId],
                                                  localObjectIndex, d);
          const double pairDualResidual =
              poseBlockDeltaNorm(previousObject, localObject, localObjective);
          iterDualResidualSq += pairDualResidual * pairDualResidual;
          if (consensusMode == ObjectConsensusMode::DirectAdmm ||
              consensusMode == ObjectConsensusMode::EdgeAdmm) {
            consensuses[robotId].updateDual(neighborId, objectId, localObject,
                                            consensusTarget, beta, eta,
                                            edgeWeight);
          }
        }
      }
    }

    const double iterObjectCommMB =
        static_cast<double>(iterObjectCommPayloadBlocks * blockBytes) /
        (1024.0 * 1024.0);
    const double iterGeodesicAuxCommMB =
        static_cast<double>(iterGeodesicAuxPayloads * 6 * sizeof(double)) /
        (1024.0 * 1024.0);
    iterCommMB =
        iterObjectCommMB + iterGeodesicAuxCommMB + iterFrameSyncStats.commMb;
    cumulativeObjectCommPoses += iterObjectCommPoses;
    cumulativeCommMB += iterCommMB;
    const auto iterEnd = std::chrono::steady_clock::now();
    const double iterTimeSec =
        std::chrono::duration<double>(iterEnd - iterStart).count();
    cumulativeTimeSec += iterTimeSec;
    const double iterPrimalResidual = std::sqrt(iterPrimalResidualSq);
    const double iterDualResidual = std::sqrt(iterDualResidualSq);
    const double iterMainMeritMeanStepScale =
        iterMainMeritFilteredUpdates == 0
            ? 1.0
            : iterMainMeritStepScaleSum /
                  static_cast<double>(iterMainMeritFilteredUpdates);
    const double iterGeodesicStepMeanScale =
        iterGeodesicStepLimitedUpdates == 0
            ? 1.0
            : iterGeodesicStepScaleSum /
                  static_cast<double>(iterGeodesicStepLimitedUpdates);
    const double iterGaugeMeritMeanStepScale =
        iterGaugeMeritFilteredUpdates == 0
            ? 1.0
            : iterGaugeMeritStepScaleSum /
                  static_cast<double>(iterGaugeMeritFilteredUpdates);

    cout << "iter " << setw(3) << iter
         << " | local_model_cost = " << setw(12) << iterLocalModelCost
         << " | measurement_cost = " << setw(12) << iterMeasurementCost
         << " | chordal_measurement_cost = " << setw(12)
         << iterChordalMeasurementCost
         << " | geodesic_measurement_cost = " << setw(12)
         << iterGeodesicMeasurementCost
         << " | consensus_cost = " << setw(12) << iterConsensusCost
         << " | primal_residual = " << setw(12) << iterPrimalResidual
         << " | dual_residual = " << setw(12) << iterDualResidual
         << " | total_gradnorm = " << setw(12) << iterTotalGradNorm
         << " | object_comm_poses = " << iterObjectCommPoses
         << " | object_comm_payload_blocks = "
         << iterObjectCommPayloadBlocks
         << " | innovation_gate_skipped = "
         << iterInnovationGateSkipped
         << " | innovation_gate_forced = "
         << iterInnovationGateForced
         << " | receiver_boundary_updates = "
         << iterReceiverBoundaryUpdates
         << " | receiver_boundary_cold_starts = "
         << iterReceiverBoundaryColdStarts
         << " | receiver_boundary_cache_delta_sum = "
         << iterReceiverBoundaryCacheDeltaSum
         << " | receiver_boundary_cache_delta_max = "
         << iterReceiverBoundaryCacheDeltaMax
         << " | receiver_boundary_disagreement_sum = "
         << iterReceiverBoundaryDisagreementSum
         << " | receiver_boundary_disagreement_max = "
         << iterReceiverBoundaryDisagreementMax
         << " | boundary_budget_candidates = "
         << iterBoundaryBudgetCandidates
         << " | boundary_budget_selected = "
         << iterBoundaryBudgetSelected
         << " | boundary_budget_forced = "
         << iterBoundaryBudgetForced
         << " | boundary_budget_skipped = "
         << iterBoundaryBudgetSkipped
         << " | boundary_budget_score_sum = "
         << iterBoundaryBudgetScoreSum
         << " | boundary_budget_selected_score_sum = "
         << iterBoundaryBudgetSelectedScoreSum
         << " | boundary_predictive_gain_sum = "
         << iterBoundaryPredictiveGainSum
         << " | boundary_predictive_selected_gain_sum = "
         << iterBoundaryPredictiveSelectedGainSum
         << " | boundary_predictive_stiffness_sum = "
         << iterBoundaryPredictiveStiffnessSum
         << " | boundary_predictive_gain_target_sum = "
         << iterBoundaryPredictiveGainTargetSum
         << " | boundary_predictive_selected_gain_fraction = "
         << iterBoundaryPredictiveSelectedGainFraction
         << " | comm_mb = " << iterCommMB
         << " | active_consensus_pairs = " << iterActiveConsensusPairs
         << " | relay_closed_form_updates = "
         << iterRelayClosedFormUpdates
         << " | consensus_correction_updates = "
         << iterConsensusCorrectionUpdates
         << " | consensus_rejected_updates = "
         << iterConsensusRejectedUpdates
         << " | edge_target_updates = " << iterEdgeTargetUpdates
         << " | distributed_step_norm = " << iterDistributedStepNorm
         << " | distributed_aux_norm = " << iterDistributedAuxNorm
         << " | gauge_merit_filtered_updates = "
         << iterGaugeMeritFilteredUpdates
         << " | gauge_merit_rejected_updates = "
         << iterGaugeMeritRejectedUpdates
         << " | gauge_merit_mean_step_scale = "
         << iterGaugeMeritMeanStepScale
         << " | stale_cache_skipped = " << iterStaleCacheSkipped
         << " | main_merit_filtered_updates = "
         << iterMainMeritFilteredUpdates
         << " | main_merit_rejected_updates = "
         << iterMainMeritRejectedUpdates
         << " | main_merit_mean_step_scale = "
         << iterMainMeritMeanStepScale
         << " | geodesic_step_limited_updates = "
         << iterGeodesicStepLimitedUpdates
         << " | geodesic_step_mean_scale = "
         << iterGeodesicStepMeanScale
         << " | geodesic_step_max_block_delta = "
         << iterGeodesicMaxBlockDelta
         << " | residual_feedback_payload_blocks = "
         << iterResidualFeedbackPayloadBlocks
         << " | geodesic_aux_payloads = "
         << iterGeodesicAuxPayloads
         << " | geodesic_aux_comm_mb = "
         << iterGeodesicAuxCommMB
         << " | frame_sync_used_edges = "
         << iterFrameSyncStats.usedEdges
         << " | frame_sync_connected_robots = "
         << iterFrameSyncStats.connectedRobots
         << " | frame_sync_comm_mb = "
         << iterFrameSyncStats.commMb
         << " | frame_sync_registration_comm_mb = "
         << iterFrameSyncStats.registrationCommMb
         << " | frame_sync_solver_comm_mb = "
         << iterFrameSyncStats.solverCommMb
         << " | frame_sync_objective = "
         << iterFrameSyncStats.solver.objective
         << " | iter_time_sec = " << iterTimeSec << endl;

    if (!csvPath.empty()) {
      csv << iter << ',' << iterLocalModelCost << ',' << iterMeasurementCost
          << ',' << iterConsensusCost << ',' << iterChordalMeasurementCost
          << ',' << iterGeodesicMeasurementCost << ',' << iterPrimalResidual
          << ','
          << iterDualResidual << ',' << iterTotalGradNorm << ','
          << iterObjectCommPoses << ',' << iterObjectCommPayloadBlocks << ','
          << iterInnovationGateSkipped << ','
          << iterInnovationGateForced << ','
          << iterReceiverBoundaryUpdates << ','
          << iterReceiverBoundaryColdStarts << ','
          << iterReceiverBoundaryCacheDeltaSum << ','
          << iterReceiverBoundaryCacheDeltaMax << ','
          << iterReceiverBoundaryDisagreementSum << ','
          << iterReceiverBoundaryDisagreementMax << ','
          << iterBoundaryBudgetCandidates << ','
          << iterBoundaryBudgetSelected << ','
          << iterBoundaryBudgetForced << ','
          << iterBoundaryBudgetSkipped << ','
          << iterBoundaryBudgetScoreSum << ','
          << iterBoundaryBudgetSelectedScoreSum << ','
          << iterBoundaryPredictiveGainSum << ','
          << iterBoundaryPredictiveSelectedGainSum << ','
          << iterBoundaryPredictiveStiffnessSum << ','
          << iterBoundaryPredictiveGainTargetSum << ','
          << iterBoundaryPredictiveSelectedGainFraction << ','
          << iterCommMB << ','
          << cumulativeObjectCommPoses << ',' << cumulativeCommMB << ','
          << knownObjectCopies << ',' << relayObjectCopies << ','
          << iterActiveConsensusPairs << ',' << iterRelayClosedFormUpdates
          << ',' << iterTimeSec << ',' << cumulativeTimeSec << ','
          << objectConsensusModeName(consensusMode) << ','
          << iterConsensusCorrectionUpdates << ','
          << iterConsensusRejectedUpdates << ','
          << iterEdgeTargetUpdates << ','
          << iterDistributedStepNorm << ','
          << iterDistributedAuxNorm << ','
          << iterGaugeMeritFilteredUpdates << ','
          << iterGaugeMeritRejectedUpdates << ','
          << iterGaugeMeritMeanStepScale << ','
          << iterStaleCacheSkipped << ','
          << iterMainMeritFilteredUpdates << ','
          << iterMainMeritRejectedUpdates << ','
          << iterMainMeritMeanStepScale << ','
          << iterGeodesicStepLimitedUpdates << ','
          << iterGeodesicStepMeanScale << ','
          << iterGeodesicMaxBlockDelta << ','
          << iterResidualFeedbackPayloadBlocks << ','
          << iterGeodesicAuxPayloads << ','
          << iterGeodesicAuxCommMB << ','
          << iterFrameSyncStats.usedEdges << ','
          << iterFrameSyncStats.connectedRobots << ','
          << iterFrameSyncStats.commMb << ','
          << iterFrameSyncStats.registrationCommMb << ','
          << iterFrameSyncStats.solverCommMb << ','
          << iterFrameSyncStats.solver.objective << '\n';
    }
    if (outerStopConsensusCost >= 0.0 &&
        static_cast<size_t>(iter + 1) >= outerStopMinIters &&
        iterConsensusCost <= outerStopConsensusCost) {
      cout << "Stopping outer iterations after " << (iter + 1)
           << " iterations because consensus_cost = " << iterConsensusCost
           << " <= OBJECT_OUTER_STOP_CONSENSUS_COST = "
           << outerStopConsensusCost
           << " and completed iterations >= OBJECT_OUTER_STOP_MIN_ITERS = "
           << outerStopMinIters << "." << endl;
      break;
    }
  }

  cout << "Finished. cumulative_object_comm_poses = "
       << cumulativeObjectCommPoses << " | cumulative_comm_mb = "
       << cumulativeCommMB
       << " | initialization_comm_mb = " << initializationCommMB
       << " | total_comm_mb = " << (initializationCommMB + cumulativeCommMB)
       << " | cumulative_time_sec = "
       << cumulativeTimeSec << endl;
  if (!poseOutputPath.empty()) {
    if (writeTrajectoryPoseRows(poseOutputPath, dataset, currentEstimates,
                                lift, d)) {
      cout << "Saved final trajectory poses to " << poseOutputPath << endl;
    }
  }
  if (!objectPoseOutputPath.empty()) {
    if (writeObjectAwarePoseRows(objectPoseOutputPath, dataset,
                                 currentEstimates, robotKnownObjects, lift,
                                 d)) {
      cout << "Saved final object-aware poses to "
           << objectPoseOutputPath << endl;
    }
  }

  return 0;
}

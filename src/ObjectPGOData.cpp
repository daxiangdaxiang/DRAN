/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/ObjectPGOData.h>

#include <DPGO/DPGO_types.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace DPGO {
namespace {

struct ParsedEdge {
  size_t v1{0};
  size_t v2{0};
  RelativeSEMeasurement measurement;
};

struct ParsedRobotFile {
  std::vector<size_t> vertexIds;
  std::map<size_t, Matrix> vertexInitialPoses;
  std::vector<ParsedEdge> edges;
  size_t dimension{0};
};

Matrix makePoseBlock2(double x, double y, double theta) {
  Matrix pose(2, 3);
  pose.leftCols(2) = Eigen::Rotation2Dd(theta).toRotationMatrix();
  pose.col(2) = Eigen::Vector2d(x, y);
  return pose;
}

Matrix makePoseBlock3(double x, double y, double z, double qx, double qy,
                      double qz, double qw) {
  Matrix pose(3, 4);
  pose.leftCols(3) =
      Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
  pose.col(3) = Eigen::Vector3d(x, y, z);
  return pose;
}

size_t maxVertexId(const std::vector<size_t> &vertexIds,
                   const std::vector<ParsedEdge> &edges) {
  size_t maxId = 0;
  bool found = false;
  for (size_t id : vertexIds) {
    maxId = std::max(maxId, id);
    found = true;
  }
  for (const ParsedEdge &edge : edges) {
    maxId = std::max(maxId, edge.v1);
    maxId = std::max(maxId, edge.v2);
    found = true;
  }
  return found ? maxId : 0;
}

RelativeSEMeasurement invertMeasurement(const RelativeSEMeasurement &m) {
  RelativeSEMeasurement inv = m;
  inv.R = m.R.transpose();
  inv.t = -inv.R * m.t;
  std::swap(inv.r1, inv.r2);
  std::swap(inv.p1, inv.p2);
  return inv;
}

bool parseEdgeSe2Line(std::istringstream &strm, ParsedEdge &edge) {
  size_t i = 0;
  size_t j = 0;
  double dx = 0.0;
  double dy = 0.0;
  double dtheta = 0.0;
  double I11 = 0.0;
  double I12 = 0.0;
  double I13 = 0.0;
  double I22 = 0.0;
  double I23 = 0.0;
  double I33 = 0.0;
  if (!(strm >> i >> j >> dx >> dy >> dtheta >> I11 >> I12 >> I13 >> I22 >>
        I23 >> I33)) {
    return false;
  }

  edge.v1 = i;
  edge.v2 = j;
  edge.measurement.r1 = 0;
  edge.measurement.r2 = 0;
  edge.measurement.p1 = i;
  edge.measurement.p2 = j;
  edge.measurement.t = Eigen::Matrix<double, 2, 1>(dx, dy);
  edge.measurement.R = Eigen::Rotation2Dd(dtheta).toRotationMatrix();

  Eigen::Matrix2d tranCov;
  tranCov << I11, I12, I12, I22;
  edge.measurement.tau = 2 / tranCov.inverse().trace();
  edge.measurement.kappa = I33;
  edge.measurement.weight = 1.0;
  return true;
}

bool parseEdgeSe3QuatLine(std::istringstream &strm, ParsedEdge &edge) {
  size_t i = 0;
  size_t j = 0;
  double dx = 0.0;
  double dy = 0.0;
  double dz = 0.0;
  double dqx = 0.0;
  double dqy = 0.0;
  double dqz = 0.0;
  double dqw = 0.0;
  double I11 = 0.0;
  double I12 = 0.0;
  double I13 = 0.0;
  double I14 = 0.0;
  double I15 = 0.0;
  double I16 = 0.0;
  double I22 = 0.0;
  double I23 = 0.0;
  double I24 = 0.0;
  double I25 = 0.0;
  double I26 = 0.0;
  double I33 = 0.0;
  double I34 = 0.0;
  double I35 = 0.0;
  double I36 = 0.0;
  double I44 = 0.0;
  double I45 = 0.0;
  double I46 = 0.0;
  double I55 = 0.0;
  double I56 = 0.0;
  double I66 = 0.0;
  if (!(strm >> i >> j >> dx >> dy >> dz >> dqx >> dqy >> dqz >> dqw >> I11 >>
        I12 >> I13 >> I14 >> I15 >> I16 >> I22 >> I23 >> I24 >> I25 >> I26 >>
        I33 >> I34 >> I35 >> I36 >> I44 >> I45 >> I46 >> I55 >> I56 >>
        I66)) {
    return false;
  }

  edge.v1 = i;
  edge.v2 = j;
  edge.measurement.r1 = 0;
  edge.measurement.r2 = 0;
  edge.measurement.p1 = i;
  edge.measurement.p2 = j;
  edge.measurement.t = Eigen::Matrix<double, 3, 1>(dx, dy, dz);
  edge.measurement.R =
      Eigen::Quaterniond(dqw, dqx, dqy, dqz).toRotationMatrix();

  Eigen::Matrix3d tranCov;
  tranCov << I11, I12, I13, I12, I22, I23, I13, I23, I33;
  edge.measurement.tau = 3 / tranCov.inverse().trace();

  Eigen::Matrix3d rotCov;
  rotCov << I44, I45, I46, I45, I55, I56, I46, I56, I66;
  edge.measurement.kappa = 3 / (2 * rotCov.inverse().trace());
  edge.measurement.weight = 1.0;
  return true;
}

ParsedRobotFile parseRobotFile(const std::filesystem::path &filename) {
  ParsedRobotFile parsed;
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Could not open g2o file: " + filename.string());
  }

  std::string line;
  std::string token;
  while (std::getline(infile, line)) {
    if (line.empty() || line[0] == '#' || line[0] == '%') {
      continue;
    }

    std::istringstream strm(line);
    strm >> token;
    if (token == "VERTEX_SE2") {
      size_t vertexId = 0;
      double x = 0.0;
      double y = 0.0;
      double theta = 0.0;
      if (strm >> vertexId >> x >> y >> theta) {
        parsed.vertexIds.push_back(vertexId);
        parsed.vertexInitialPoses[vertexId] = makePoseBlock2(x, y, theta);
        parsed.dimension = std::max<size_t>(parsed.dimension, 2);
      }
      continue;
    }

    if (token == "VERTEX_SE3:QUAT") {
      size_t vertexId = 0;
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      double qx = 0.0;
      double qy = 0.0;
      double qz = 0.0;
      double qw = 1.0;
      if (strm >> vertexId >> x >> y >> z >> qx >> qy >> qz >> qw) {
        parsed.vertexIds.push_back(vertexId);
        parsed.vertexInitialPoses[vertexId] =
            makePoseBlock3(x, y, z, qx, qy, qz, qw);
        parsed.dimension = std::max<size_t>(parsed.dimension, 3);
      }
      continue;
    }

    if (token == "EDGE_SE2") {
      ParsedEdge edge;
      if (parseEdgeSe2Line(strm, edge)) {
        parsed.dimension = std::max<size_t>(parsed.dimension, 2);
        parsed.edges.push_back(std::move(edge));
      }
      continue;
    }

    if (token == "EDGE_SE3:QUAT") {
      ParsedEdge edge;
      if (parseEdgeSe3QuatLine(strm, edge)) {
        parsed.dimension = std::max<size_t>(parsed.dimension, 3);
        parsed.edges.push_back(std::move(edge));
      }
      continue;
    }
  }

  for (const ParsedEdge &edge : parsed.edges) {
    parsed.vertexIds.push_back(edge.v1);
    parsed.vertexIds.push_back(edge.v2);
  }

  std::sort(parsed.vertexIds.begin(), parsed.vertexIds.end());
  parsed.vertexIds.erase(std::unique(parsed.vertexIds.begin(),
                                     parsed.vertexIds.end()),
                         parsed.vertexIds.end());
  return parsed;
}

size_t inferObjectStartFromConsecutiveSuffix(const std::vector<size_t> &vertexIds,
                                             const std::vector<ParsedEdge> &edges) {
  if (vertexIds.empty()) {
    return 0;
  }

  std::set<std::pair<size_t, size_t>> consecutiveEdges;
  for (const ParsedEdge &edge : edges) {
    size_t lo = std::min(edge.v1, edge.v2);
    size_t hi = std::max(edge.v1, edge.v2);
    consecutiveEdges.insert(std::make_pair(lo, hi));
  }

  size_t trajectoryEnd = vertexIds.front();
  for (size_t idx = 1; idx < vertexIds.size(); ++idx) {
    const size_t prev = vertexIds[idx - 1];
    const size_t cur = vertexIds[idx];
    if (cur != prev + 1) {
      break;
    }
    if (consecutiveEdges.find(std::make_pair(prev, cur)) == consecutiveEdges.end()) {
      break;
    }
    trajectoryEnd = cur;
  }

  return trajectoryEnd + 1;
}

std::vector<std::filesystem::path> resolveRobotFiles(const std::string &directory,
                                                    size_t numRobots) {
  std::vector<std::filesystem::path> files;
  const std::filesystem::path dirPath(directory);
  if (numRobots > 0) {
    files.reserve(numRobots);
    for (size_t robotId = 0; robotId < numRobots; ++robotId) {
      std::filesystem::path file = dirPath / ("rob_" + std::to_string(robotId) + ".g2o");
      if (!std::filesystem::exists(file)) {
        throw std::runtime_error("Missing robot file: " + file.string());
      }
      files.push_back(file);
    }
    return files;
  }

  for (size_t robotId = 0;; ++robotId) {
    std::filesystem::path file = dirPath / ("rob_" + std::to_string(robotId) + ".g2o");
    if (!std::filesystem::exists(file)) {
      break;
    }
    files.push_back(file);
  }

  if (files.empty()) {
    throw std::runtime_error("Could not infer any rob_*.g2o files in: " + directory);
  }
  return files;
}

}  // namespace

ObjectPGODirectoryData loadObjectAwareG2ODirectory(const std::string &directory,
                                                   size_t numRobots,
                                                   size_t numObjects) {
  ObjectPGODirectoryData dataset;
  const std::vector<std::filesystem::path> files = resolveRobotFiles(directory, numRobots);
  dataset.numRobots = files.size();
  dataset.robots.resize(dataset.numRobots);
  dataset.numObjects = numObjects;
  dataset.numObjectsInferred = (numObjects == 0);

  std::vector<ParsedRobotFile> parsedFiles;
  parsedFiles.reserve(dataset.numRobots);
  for (const std::filesystem::path &file : files) {
    parsedFiles.push_back(parseRobotFile(file));
    dataset.dimension = std::max(dataset.dimension, parsedFiles.back().dimension);
  }

  if (dataset.dimension == 0) {
    throw std::runtime_error("No EDGE_SE2 or EDGE_SE3:QUAT measurements found in " +
                             directory);
  }

  std::vector<size_t> inferredObjectCounts;
  inferredObjectCounts.reserve(dataset.numRobots);
  if (dataset.numObjects == 0) {
    for (const ParsedRobotFile &parsed : parsedFiles) {
      const size_t objectStart = inferObjectStartFromConsecutiveSuffix(parsed.vertexIds,
                                                                       parsed.edges);
      const size_t maxId = maxVertexId(parsed.vertexIds, parsed.edges);
      if (maxId < objectStart) {
        inferredObjectCounts.push_back(0);
      } else {
        inferredObjectCounts.push_back(maxId - objectStart + 1);
      }
    }

    dataset.numObjects = *std::min_element(inferredObjectCounts.begin(),
                                           inferredObjectCounts.end());
  }

  for (size_t robotId = 0; robotId < dataset.numRobots; ++robotId) {
    const ParsedRobotFile &parsed = parsedFiles[robotId];
    ObjectPGORobotData &robot = dataset.robots[robotId];
    robot.robotId = robotId;
    robot.filename = files[robotId].string();

    const size_t maxId = maxVertexId(parsed.vertexIds, parsed.edges);
    if (maxId + 1 < dataset.numObjects) {
      throw std::runtime_error("numObjects exceeds the vertex id range in " +
                               robot.filename);
    }

    size_t objectStart = 0;
    if (numObjects > 0) {
      objectStart = maxId + 1 - dataset.numObjects;
    } else {
      objectStart = inferObjectStartFromConsecutiveSuffix(parsed.vertexIds, parsed.edges);
      const size_t inferredCount = (maxId >= objectStart) ? (maxId - objectStart + 1) : 0;
      if (dataset.numObjects > inferredCount) {
        dataset.numObjects = inferredCount;
      }
    }

    robot.objectStart = objectStart;

    for (size_t vertexId : parsed.vertexIds) {
      if (vertexId < objectStart) {
        robot.trajectoryVertexToLocal.emplace(
            vertexId, robot.trajectoryVertexIds.size());
        robot.trajectoryVertexIds.push_back(vertexId);
        const auto poseIt = parsed.vertexInitialPoses.find(vertexId);
        if (poseIt != parsed.vertexInitialPoses.end()) {
          robot.trajectoryInitialPoses.push_back(poseIt->second);
        } else {
          robot.trajectoryInitialPoses.push_back(Matrix());
        }
        continue;
      }
      if (vertexId >= objectStart + dataset.numObjects) {
        continue;
      }
      const size_t globalObjectId = vertexId - objectStart;
      robot.objectLocalVertexIds.push_back(vertexId);
      robot.objectLocalToGlobal.emplace(vertexId, globalObjectId);
      robot.objectGlobalToLocalVertex.emplace(globalObjectId, vertexId);
      const auto poseIt = parsed.vertexInitialPoses.find(vertexId);
      if (poseIt != parsed.vertexInitialPoses.end()) {
        robot.objectInitialPoses.push_back(poseIt->second);
      } else {
        robot.objectInitialPoses.push_back(Matrix());
      }
    }

    for (const ParsedEdge &edge : parsed.edges) {
      const bool firstIsObject = edge.v1 >= objectStart &&
                                 edge.v1 < objectStart + dataset.numObjects;
      const bool secondIsObject = edge.v2 >= objectStart &&
                                  edge.v2 < objectStart + dataset.numObjects;
      const bool firstIsTrajectory = edge.v1 < objectStart;
      const bool secondIsTrajectory = edge.v2 < objectStart;

      if (firstIsTrajectory && secondIsTrajectory) {
        RelativeSEMeasurement measurement = edge.measurement;
        measurement.r1 = robotId;
        measurement.r2 = robotId;
        measurement.p1 = robot.trajectoryVertexToLocal.at(edge.v1);
        measurement.p2 = robot.trajectoryVertexToLocal.at(edge.v2);
        robot.trajectoryMeasurements.push_back(std::move(measurement));
        continue;
      }

      if (firstIsTrajectory && secondIsObject) {
        ObjectObservationMeasurement obs;
        obs.robotId = robotId;
        obs.trajectoryVertexId = edge.v1;
        obs.trajectoryLocalIndex = robot.trajectoryVertexToLocal.at(edge.v1);
        obs.objectLocalVertexId = edge.v2;
        obs.objectGlobalVertexId = edge.v2 - objectStart;
        obs.objectLocalIndex = obs.objectGlobalVertexId;
        obs.measurement = edge.measurement;
        obs.measurement.r1 = robotId;
        obs.measurement.r2 = robotId;
        obs.measurement.p1 = obs.trajectoryLocalIndex;
        obs.measurement.p2 =
            robot.trajectoryVertexIds.size() + obs.objectLocalIndex;
        robot.objectObservationMeasurements.push_back(std::move(obs));
        continue;
      }

      if (secondIsTrajectory && firstIsObject) {
        ObjectObservationMeasurement obs;
        obs.robotId = robotId;
        obs.trajectoryVertexId = edge.v2;
        obs.trajectoryLocalIndex = robot.trajectoryVertexToLocal.at(edge.v2);
        obs.objectLocalVertexId = edge.v1;
        obs.objectGlobalVertexId = edge.v1 - objectStart;
        obs.objectLocalIndex = obs.objectGlobalVertexId;
        obs.measurement = invertMeasurement(edge.measurement);
        obs.measurement.r1 = robotId;
        obs.measurement.r2 = robotId;
        obs.measurement.p1 = obs.trajectoryLocalIndex;
        obs.measurement.p2 =
            robot.trajectoryVertexIds.size() + obs.objectLocalIndex;
        robot.objectObservationMeasurements.push_back(std::move(obs));
        continue;
      }

      if (firstIsObject && secondIsObject) {
        continue;
      }
    }
  }

  return dataset;
}

}  // namespace DPGO

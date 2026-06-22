/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGO_OBJECTPGODATA_H
#define DPGO_OBJECTPGODATA_H

#include <DPGO/RelativeSEMeasurement.h>

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace DPGO {

/** Object observation attached to a single robot file.
 *
 * The measurement is stored in the same SE(d) form used elsewhere in DPGO,
 * but the second endpoint is the object vertex instead of another pose.
 */
struct ObjectObservationMeasurement {
  size_t robotId{0};
  size_t trajectoryVertexId{0};
  size_t trajectoryLocalIndex{0};
  size_t objectLocalVertexId{0};
  size_t objectGlobalVertexId{0};
  size_t objectLocalIndex{0};
  RelativeSEMeasurement measurement;
};

/** Parsed data for one robot-specific g2o file. */
struct ObjectPGORobotData {
  size_t robotId{0};
  std::string filename;
  size_t objectStart{0};
  std::vector<size_t> trajectoryVertexIds;
  std::vector<size_t> objectLocalVertexIds;
  std::map<size_t, size_t> trajectoryVertexToLocal;
  std::map<size_t, size_t> objectLocalToGlobal;
  std::map<size_t, size_t> objectGlobalToLocalVertex;
  std::vector<Matrix> trajectoryInitialPoses;
  std::vector<Matrix> objectInitialPoses;
  std::vector<RelativeSEMeasurement> trajectoryMeasurements;
  std::vector<ObjectObservationMeasurement> objectObservationMeasurements;
};

/** Parsed data for a directory of robot-specific g2o files. */
struct ObjectPGODirectoryData {
  size_t dimension{0};
  size_t numRobots{0};
  size_t numObjects{0};
  bool numObjectsInferred{false};
  std::vector<ObjectPGORobotData> robots;
};

/** Load an object-aware g2o directory containing rob_0.g2o ... rob_N.g2o.
 *
 * @param directory Directory containing robot files.
 * @param numRobots Number of robot files to load. Use 0 to infer by probing
 *                  rob_0.g2o, rob_1.g2o, ...
 * @param numObjects Number of shared objects. Use 0 to infer the common
 *                   object suffix from the files.
 */
ObjectPGODirectoryData loadObjectAwareG2ODirectory(const std::string &directory,
                                                   size_t numRobots = 0,
                                                   size_t numObjects = 0);

}  // namespace DPGO

#endif

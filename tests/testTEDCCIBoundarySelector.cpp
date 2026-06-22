#include <DPGO/TEDCCI.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

RelativeSEMeasurement edge(unsigned r1, unsigned p1, unsigned r2,
                           unsigned p2) {
  return RelativeSEMeasurement(r1, r2, p1, p2, Matrix::Identity(3, 3),
                               Vector::Zero(3), 1.0, 1.0);
}

std::vector<PoseKey> sorted(std::vector<PoseKey> keys) {
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  return keys;
}

void expectKeysEqual(std::vector<PoseKey> actual,
                     std::vector<PoseKey> expected) {
  actual = sorted(std::move(actual));
  expected = sorted(std::move(expected));
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(actual[i], expected[i]);
  }
}

}  // namespace

TEST(testDPGO, TEDCCIBoundarySelectorMarksAnchorAndSharedChainEndpoint) {
  TEDCCIParams params;
  params.anchor_robot_id = 0;
  params.anchor_pose_id = 0;
  const std::vector<PoseKey> localPoses = {PoseKey{0, 0}, PoseKey{0, 1}};
  const std::vector<RelativeSEMeasurement> localMeasurements = {
      edge(0, 0, 0, 1)};
  const std::vector<RelativeSEMeasurement> sharedMeasurements = {
      edge(0, 1, 1, 2)};

  const TEDCCIPartition partition = BoundarySelector::SelectBoundaryVariables(
      0, localPoses, localMeasurements, sharedMeasurements, params);

  EXPECT_EQ(partition.local_robot_id, 0);
  expectKeysEqual(partition.local_poses, localPoses);
  expectKeysEqual(partition.boundary_poses, {PoseKey{0, 0}, PoseKey{0, 1}});
  EXPECT_TRUE(partition.interior_poses.empty());
  ASSERT_EQ(partition.neighbor_boundary_poses.count(1), 1u);
  expectKeysEqual(partition.neighbor_boundary_poses.at(1), {PoseKey{0, 1}});
}

TEST(testDPGO, TEDCCIBoundarySelectorHandlesSparseInterRobotLoopClosures) {
  TEDCCIParams params;
  const std::vector<PoseKey> localPoses = {PoseKey{1, 2}, PoseKey{1, 3},
                                           PoseKey{1, 4}};
  const std::vector<RelativeSEMeasurement> sharedMeasurements = {
      edge(0, 1, 1, 2), edge(1, 4, 2, 0)};

  const TEDCCIPartition partition = BoundarySelector::SelectBoundaryVariables(
      1, localPoses, {}, sharedMeasurements, params);

  expectKeysEqual(partition.boundary_poses, {PoseKey{1, 2}, PoseKey{1, 4}});
  expectKeysEqual(partition.interior_poses, {PoseKey{1, 3}});
  ASSERT_EQ(partition.neighbor_boundary_poses.count(0), 1u);
  ASSERT_EQ(partition.neighbor_boundary_poses.count(2), 1u);
  expectKeysEqual(partition.neighbor_boundary_poses.at(0), {PoseKey{1, 2}});
  expectKeysEqual(partition.neighbor_boundary_poses.at(2), {PoseKey{1, 4}});
}

TEST(testDPGO, TEDCCIBoundarySelectorChoosesDeterministicFallbackBoundary) {
  TEDCCIParams params;
  params.enable_component_fallback = true;
  const std::vector<PoseKey> localPoses = {PoseKey{2, 4}, PoseKey{2, 3}};

  const TEDCCIPartition partition = BoundarySelector::SelectBoundaryVariables(
      2, localPoses, {}, {}, params);

  expectKeysEqual(partition.boundary_poses, {PoseKey{2, 3}});
  expectKeysEqual(partition.interior_poses, {PoseKey{2, 4}});
  EXPECT_TRUE(partition.neighbor_boundary_poses.empty());
}

TEST(testDPGO, TEDCCIBoundarySelectorDoesNotAddAnchorOwnedByOtherRobot) {
  TEDCCIParams params;
  params.anchor_robot_id = 0;
  params.anchor_pose_id = 0;
  const std::vector<PoseKey> localPoses = {PoseKey{1, 0}, PoseKey{1, 1}};
  const std::vector<RelativeSEMeasurement> sharedMeasurements = {
      edge(1, 1, 2, 0)};

  const TEDCCIPartition partition = BoundarySelector::SelectBoundaryVariables(
      1, localPoses, {}, sharedMeasurements, params);

  expectKeysEqual(partition.boundary_poses, {PoseKey{1, 1}});
  expectKeysEqual(partition.interior_poses, {PoseKey{1, 0}});
}

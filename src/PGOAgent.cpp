/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_utils.h>
#include <DPGO/PGOAgent.h>
#include <DPGO/QuadraticOptimizer.h>

#include <Eigen/CholmodSupport>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <ostream>
#include <random>
#include <utility>
#include <vector>

#include "DPGO/DPGO_types.h"
#include "DPGO/QuadraticProblem.h"
#include "DPGO/RelativeSEMeasurement.h"
#include "StieVariable.h"

using std::lock_guard;
using std::thread;
using std::unique_lock;
using std::vector;

namespace DPGO {

PGOAgent::PGOAgent(unsigned ID, const PGOAgentParameters &params)
    : mID(ID),
      d(params.d),
      r(params.r),
      n(1),
      mParams(params),
      mState(PGOAgentState::WAIT_FOR_DATA),
      mStatus(ID, mState, 0, 0, false, 0),
      mRobustCost(params.robustCostType, params.robustCostParams),
      mProblemPtr(nullptr),
      mInstanceNumber(0),
      mIterationNumber(0),
      mNumPosesReceived(0),
      mLogger(params.logDirectory) {
  if (mParams.verbose) {
    std::cout << "Initializing PGO agent..." << std::endl;
    std::cout << params << std::endl;
  }

  // Initialize X
  X = Matrix::Zero(r, d + 1);
  X.block(0, 0, d, d) = Matrix::Identity(d, d);
  if (mID == 0) setLiftingMatrix(fixedStiefelVariable(d, r));
  resetTeamStatus();
}

PGOAgent::~PGOAgent() {
  // Make sure that optimization thread is not running, before exiting
  endOptimizationLoop();
}

void PGOAgent::setX(const Matrix &Xin) {
  lock_guard<mutex> lock(mPosesMutex);
  assert(mState != PGOAgentState::WAIT_FOR_DATA);
  assert(Xin.rows() == relaxation_rank());
  assert(Xin.cols() ==
         (dimension() + 1) * (num_poses() + neighborSharedPoseIDs.size()));
  mState = PGOAgentState::INITIALIZED;
  X = Xin;
  Y = Xin;
  if (mParams.acceleration) {
    initializeAcceleration();
  }
  if (mParams.verbose) {
    printf("Robot %u resets trajectory estimates. New trajectory length = %u\n",
           getID(), num_poses());
  }
  setX_private();
  setY_shared();
  construct_shared_GMatrix();
  construct_private_GMatrix();
  H_local = shared_mProblemPtr->RieGrad(Y_shared);
}
void PGOAgent::setX_private() {
  X_private = Matrix::Zero(r, localprivatePoseIDs.size() * (d + 1));
  size_t i = 0;
  for (const auto &poseid : localprivatePoseIDs) {
    unsigned int index = poseid.second;
    X_private.block(0, i * (d + 1), r, d + 1) =
        X.block(0, index * (d + 1), r, d + 1);
    i++;
  }
}

void PGOAgent::setY_shared() {
  size_t i = 0;
  Y_shared = Matrix::Zero(
      r, (localSharedPoseIDs.size() + neighborSharedPoseIDs.size()) * (d + 1));
  for (auto &it : localSharedPoseIDs) {
    assert(it.first == mID);
    unsigned int index = it.second;
    Y_shared.block(0, i * (d + 1), r, d + 1) =
        X.block(0, index * (d + 1), r, d + 1);

    i++;
  }
  i = 0;
  for (auto &it : neighborSharedPoseIDs) {
    Y_shared.block(0, (i + localSharedPoseIDs.size()) * (d + 1), r, d + 1) =
        X.block(0, (i + num_poses()) * (d + 1), r, d + 1);
    i++;
  }
}
void PGOAgent::set_whole_X() {
  for (unsigned int i = 0; i < num_poses(); i++) {
    auto it = localprivatePoseIDs.find(std::make_pair(mID, i));

    if (it != localprivatePoseIDs.end()) {
      size_t index = std::distance(localprivatePoseIDs.begin(), it);

      X.block(0, i * (d + 1), r, d + 1) =
          X_private.block(0, index * (d + 1), r, d + 1);
    } else {
      auto it2 = localSharedPoseIDs.find(std::make_pair(mID, i));
      if (it2 != localSharedPoseIDs.end()) {
        size_t index = std::distance(localSharedPoseIDs.begin(), it2);

        X.block(0, i * (d + 1), r, d + 1) =
            Y_shared.block(0, index * (d + 1), r, d + 1);
      }
    }
  }
  for (size_t i = 0; i < neighborSharedPoseIDs.size(); i++) {
    X.block(0, (i + num_poses()) * (d + 1), r, d + 1) =
        Y_shared.block(0, (i + localSharedPoseIDs.size()) * (d + 1), r, d + 1);
  }
}
bool PGOAgent::getX(Matrix &Mout) {
  lock_guard<mutex> lock(mPosesMutex);
  // std::cout<<mID<<"martix"<<std::endl;
  // std::cout<<X<<std::endl;
  Mout = Matrix::Zero(r, num_poses() * (d + 1));
  for (unsigned int i = 0; i < num_poses(); i++) {
    auto it = localprivatePoseIDs.find(std::make_pair(mID, i));

    if (it != localprivatePoseIDs.end()) {
      size_t index = std::distance(localprivatePoseIDs.begin(), it);
      Mout.block(0, i * (d + 1), r, d + 1) =
          X_private.block(0, index * (d + 1), r, d + 1);

    } else {
      auto it2 = localSharedPoseIDs.find(std::make_pair(mID, i));
      if (it2 != localSharedPoseIDs.end()) {
        size_t index = std::distance(localSharedPoseIDs.begin(), it2);
        Mout.block(0, i * (d + 1), r, d + 1) =
            Y_shared.block(0, index * (d + 1), r, d + 1);
      }
    }
  }

  return true;
}

bool PGOAgent::getSharedPose(unsigned int index, Matrix &Mout) {
  if (mState != PGOAgentState::INITIALIZED) return false;
  lock_guard<mutex> lock(mPosesMutex);
  if (index >= num_poses() + neighborSharedPoseIDs.size()) return false;
  Mout = X.block(0, index * (d + 1), r, d + 1);
  return true;
}

bool PGOAgent::getAuxSharedPose(unsigned int index, Matrix &Mout) {
  assert(mParams.acceleration);
  if (mState != PGOAgentState::INITIALIZED) return false;
  lock_guard<mutex> lock(mPosesMutex);
  if (index >= num_poses()) return false;
  Mout = Y.block(0, index * (d + 1), r, d + 1);
  return true;
}

bool PGOAgent::getSharedPoseDict(PoseDict &map) {
  if (mState != PGOAgentState::INITIALIZED) return false;
  map.clear();
  lock_guard<mutex> lock(mPosesMutex);
  size_t index = 0;
  for (const auto &seperator : localSharedPoseIDs) {
    map[seperator] = Y_shared.block(0, index * (d + 1), r, d + 1);
    index++;
  }
  index = localSharedPoseIDs.size();
  for (auto &neighbor : neighborSharedPoseIDs) {
    map[neighbor] = Y_shared.block(0, index * (d + 1), r, d + 1);
    index++;
  }
  return true;
}
bool PGOAgent::getShared_H(std::map<PoseID, Matrix> &map) {
  if (mState != PGOAgentState::INITIALIZED) return false;
  map.clear();
  lock_guard<mutex> lock(mPosesMutex);
  size_t cnt = 0;
  for (auto &it : localSharedPoseIDs) {
    map[it] = H_local.block(0, cnt * (d + 1), r, d + 1);
    cnt++;
  }
  for (auto &it : neighborSharedPoseIDs) {
    map[it] = H_local.block(0, cnt * (d + 1), r, d + 1);
    cnt++;
  }
  return true;
}
bool PGOAgent::getAuxSharedPoseDict(PoseDict &map) {
  assert(mParams.acceleration);
  if (mState != PGOAgentState::INITIALIZED) return false;
  map.clear();
  lock_guard<mutex> lock(mPosesMutex);
  for (const auto &mSharedPose : localSharedPoseIDs) {
    unsigned idx = std::get<1>(mSharedPose);
    map[mSharedPose] = Y.block(0, idx * (d + 1), r, d + 1);
  }
  return true;
}

void PGOAgent::setLiftingMatrix(const Matrix &M) {
  assert(M.rows() == r);
  assert(M.cols() == d);
  YLift.emplace(M);
}

void PGOAgent::setPoseGraph(
    const std::vector<RelativeSEMeasurement> &inputOdometry,
    const std::vector<RelativeSEMeasurement> &inputPrivateLoopClosures,
    const std::vector<RelativeSEMeasurement> &inputSharedLoopClosures,
    const Matrix &TInit) {
  assert(!isOptimizationRunning());
  assert(mState == PGOAgentState::WAIT_FOR_DATA);
  assert(n == 1);

  if (inputOdometry.empty()) return;

  for (const auto &edge : inputOdometry) {
    addOdometry(edge);
  }
  for (const auto &edge : inputPrivateLoopClosures) {
    addPrivateLoopClosure(edge);
  }
  for (const auto &edge : inputSharedLoopClosures) {
    addSharedLoopClosure(edge);
  }

  vector<RelativeSEMeasurement> localMeasurements = odometry;

  localMeasurements.insert(localMeasurements.end(), privateLoopClosures.begin(),
                           privateLoopClosures.end());
  

  // for (auto &m : localMeasurements) {
  //   auto it_i = localprivatePoseIDs.find(std::make_pair(m.r1, m.p1));
  //   auto it_j = localprivatePoseIDs.find(std::make_pair(m.r2, m.p2));

  //   if (it_i != localprivatePoseIDs.end() && it_j != localprivatePoseIDs.end())
  //     private_measurements.push_back(m);
  //   else if (it_i == localprivatePoseIDs.end() &&
  //            it_j == localprivatePoseIDs.end())
  //     shared_shared_measurements.push_back(m);
  //   else
  //     private_shared_measurements.push_back(m);
  // }
  // for(auto &m:sharedLoopClosures)
  //   shared_shared_measurements.push_back(m);
  // std::cout << "private ids: " << std::endl;
  // for (auto i : localprivatePoseIDs) {
  //   std::cout << i.first << ":" << i.second << ", ";
  // }
  // std::cout << std::endl;
  // std::cout << "shared ids: " << std::endl;
  // for (auto i : localSharedPoseIDs) {
  //   std::cout << i.first << ":" << i.second << ", ";
  // }
  // std::cout << std::endl;
  // std::cout << "neighbor ids: " << std::endl;
  // for (auto i : neighborSharedPoseIDs) {
  //   std::cout << i.first << ":" << i.second << ", ";
  // }
  // std::cout << std::endl;
  std::cout << "num of pose:" << num_poses()
            << ",private id:" << localprivatePoseIDs.size()
            << ",shared id:" << localSharedPoseIDs.size()
            << ",neighbor id:" << neighborSharedPoseIDs.size() << std::endl;
  // Check validity of initial trajectory estimate, if provided
  bool local_init = true;
  unsigned expected_rows = dimension();
  unsigned expected_cols = (dimension() + 1) * num_poses();
  if (TInit.rows() > 0 && TInit.cols() > 0) {
    if (TInit.rows() == expected_rows && TInit.cols() == expected_cols) {
      local_init = false;
    } else {
      local_init = true;
      printf(
          "Error: provided initial trajectory has wrong dimension! "
          "Expect (%u,%u), received (%ld, %ld). Using local initialization. \n",
          expected_rows, expected_cols, TInit.rows(), TInit.cols());
    }
  }

  // Create new optimization problem
  mProblemPtr = new QuadraticProblem(num_poses() + neighborSharedPoseIDs.size(),
                                     dimension(), relaxation_rank());
  private_mProblemPtr = new QuadraticProblem(
      num_poses(), dimension(), relaxation_rank());
  shared_mProblemPtr = new QuadraticProblem(
       neighborSharedPoseIDs.size(), dimension(),
      relaxation_rank());
  // Robot can construct the quadratic cost matrix now, as it does not depend on
  // neighbor values constructQMatrix();
  construct_whole_QMatrix();
  construct_private_QMatrix();
  construct_shared_QMatrix();

  // Initialize trajectory estimate in an arbitrary frame
  if (!local_init) {
    if (mParams.verbose) printf("Using provided trajectory initialization.\n");
    TLocalInit.emplace(TInit.block(0, 0, expected_rows, expected_cols));
  } else {
    if (mParams.verbose) printf("Using internal trajectory initialization.\n");
    localInitialization();
  }

  // Waiting for initialization in the GLOBAL frame
  mState = PGOAgentState::WAIT_FOR_INITIALIZATION;

  // If I am the first robot or if cross-robot initialization if off,
  // I will consider myself as initialized in the global frame
  // if (mID == 0 || !mParams.multirobot_initialization) {
  //   X = YLift.value() * TLocalInit.value();  // Lift to correct relaxation
  //   rank XInit.emplace(X); mState = PGOAgentState::INITIALIZED; if
  //   (mParams.acceleration) {
  //     initializeAcceleration();
  //   }

  //   // Save initial trajectory
  //   if (mParams.logData) {
  //     mLogger.logTrajectory(dimension(), num_poses(), TLocalInit.value(),
  //     "trajectory_initial.csv");
  //   }
  // }
}

void PGOAgent::addOdometry(const RelativeSEMeasurement &factor) {
  assert(mState != PGOAgentState::INITIALIZED);
  // check that this is a odometry measurement
  assert(factor.r1 == mID);
  assert(factor.r2 == mID);
  assert(factor.p1 + 1 == factor.p2);
  assert(factor.R.rows() == d && factor.R.cols() == d);
  assert(factor.t.rows() == d && factor.t.cols() == 1);

  // update number of poses
  n = std::max(n, (unsigned)factor.p2 + 1);

  lock_guard<mutex> mLock(mMeasurementsMutex);
  odometry.push_back(factor);
}

void PGOAgent::addPrivateLoopClosure(const RelativeSEMeasurement &factor) {
  assert(mState != PGOAgentState::INITIALIZED);
  assert(factor.r1 == mID);
  assert(factor.r2 == mID);
  assert(factor.R.rows() == d && factor.R.cols() == d);
  assert(factor.t.rows() == d && factor.t.cols() == 1);

  // update number of poses
  n = std::max(n, (unsigned)std::max(factor.p1 + 1, factor.p2 + 1));

  lock_guard<mutex> lock(mMeasurementsMutex);
  privateLoopClosures.push_back(factor);
}

void PGOAgent::addSharedLoopClosure(const RelativeSEMeasurement &factor) {
  assert(mState != PGOAgentState::INITIALIZED);
  assert(factor.R.rows() == d && factor.R.cols() == d);
  assert(factor.t.rows() == d && factor.t.cols() == 1);
  PoseID bot1 = std::make_pair(factor.r1, factor.p1);
  PoseID bot2 = std::make_pair(factor.r2, factor.p2);

  if (factor.r1 == mID) {
    assert(factor.r2 != mID);
    n = std::max(n, (unsigned)factor.p1 + 1);
    localSharedPoseIDs.insert(bot1);
    neighborSharedPoseIDs.insert(bot2);
    neighborRobotIDs.insert(factor.r2);

    // insert neighbor by order
    // auto it = std::find(shared_neighbor.begin(), shared_neighbor.end(),
    // bot2); if (it == shared_neighbor.end())
    //   shared_neighbor.push_back(bot2);
  } else {
    assert(factor.r2 == mID);
    n = std::max(n, (unsigned)factor.p2 + 1);
    localSharedPoseIDs.insert(bot2);
    neighborSharedPoseIDs.insert(bot1);
    neighborRobotIDs.insert(factor.r1);

    // auto it = std::find(shared_neighbor.begin(), shared_neighbor.end(),
    // bot1); if (it == shared_neighbor.end())
    //   shared_neighbor.push_back(bot1);
  }
  lock_guard<mutex> lock(mMeasurementsMutex);
  sharedLoopClosures.push_back(factor);
}

Matrix PGOAgent::computeNeighborTransform(const PoseID &nID,
                                          const Matrix &var) {
  assert(YLift);
  assert(var.rows() == r);
  assert(var.cols() == d + 1);

  // Find the corresponding inter-robot loop closure
  RelativeSEMeasurement &m = findSharedLoopClosureWithNeighbor(nID);

  // Notations:
  // world1: world frame before alignment
  // world2: world frame after alignment
  // frame1 : frame associated to my public pose
  // frame2 : frame associated to neighbor's public pose
  Matrix dT = Matrix::Identity(d + 1, d + 1);
  dT.block(0, 0, d, d) = m.R;
  dT.block(0, d, d, 1) = m.t;
  Matrix T_world2_frame2 = Matrix::Identity(d + 1, d + 1);
  T_world2_frame2.block(0, 0, d, d + 1) =
      YLift.value().transpose() *
      var;  // Round the received neighbor pose value back to SE(d)
  Matrix T = TLocalInit.value();
  Matrix T_frame1_frame2 = Matrix::Identity(d + 1, d + 1);
  Matrix T_world1_frame1 = Matrix::Identity(d + 1, d + 1);
  if (m.r1 == nID.first) {
    // Incoming edge
    T_frame1_frame2 = dT.inverse();
    T_world1_frame1.block(0, 0, d, d + 1) =
        T.block(0, m.p2 * (d + 1), d, d + 1);
  } else {
    // Outgoing edge
    T_frame1_frame2 = dT;
    T_world1_frame1.block(0, 0, d, d + 1) =
        T.block(0, m.p1 * (d + 1), d, d + 1);
  }
  Matrix T_world2_frame1 = T_world2_frame2 * T_frame1_frame2.inverse();
  Matrix T_world2_world1 = T_world2_frame1 * T_world1_frame1.inverse();
  checkRotationMatrix(T_world2_world1.block(0, 0, d, d));
  return T_world2_world1;
}

Matrix PGOAgent::computeRobustNeighborTransformTwoStage(
    unsigned int neighborID, const PoseDict &poseDict) {
  std::vector<Matrix> RVec;
  std::vector<Vector> tVec;
  for (const auto &it : poseDict) {
    const PoseID nID = it.first;
    const auto var = it.second;
    if (neighborSharedPoseIDs.find(nID) != neighborSharedPoseIDs.end()) {
      const auto T = computeNeighborTransform(nID, var);
      RVec.emplace_back(T.block(0, 0, d, d));
      tVec.emplace_back(T.block(0, d, d, 1));
    }
  }
  int m = (int)RVec.size();
  const Vector kappa = Vector::Ones(m);
  const Vector tau = Vector::Ones(m);
  Matrix ROpt;
  Vector tOpt;
  std::vector<size_t> inlierIndices;
  // Perform robust single rotation averaging
  double maxRotationError = angular2ChordalSO3(0.5);  // approximately 30 deg
  robustSingleRotationAveraging(ROpt, inlierIndices, RVec, kappa,
                                maxRotationError);
  int inlierSize = (int)inlierIndices.size();
  printf(
      "[RobustRelativeTransform] This robot %u, neighbor %u: finds %i "
      "inliers out of %i measurements.\n",
      getID(), neighborID, inlierSize, m);
  if (inlierSize == 0) {
    throw std::runtime_error(
        "Robust single rotation averaging returns empty inlier set!");
  }
  // Perform single translation averaging on the inlier set
  std::vector<Vector> tVecInliers;
  for (const auto index : inlierIndices) {
    tVecInliers.emplace_back(tVec[index]);
  }
  singleTranslationAveraging(tOpt, tVecInliers);
  // Return transformation as matrix
  Matrix TOpt = Matrix::Identity(dimension() + 1, dimension() + 1);
  TOpt.block(0, 0, d, d) = ROpt;
  TOpt.block(0, d, d, 1) = tOpt;
  return TOpt;
}

Matrix PGOAgent::computeRobustNeighborTransform(unsigned int neighborID,
                                                const PoseDict &poseDict) {
  std::vector<Matrix> RVec;
  std::vector<Vector> tVec;
  for (const auto &it : poseDict) {
    const PoseID nID = it.first;
    const auto var = it.second;
    if (neighborSharedPoseIDs.find(nID) != neighborSharedPoseIDs.end()) {
      const auto T = computeNeighborTransform(nID, var);
      RVec.emplace_back(T.block(0, 0, d, d));
      tVec.emplace_back(T.block(0, d, d, 1));
    }
  }
  int m = (int)RVec.size();
  const Vector kappa =
      1.82 * Vector::Ones(m);  // rotation stddev approximately 30 degree
  const Vector tau = 0.01 * Vector::Ones(m);  // translation stddev 10 m
  const double cbar = RobustCost::computeErrorThresholdAtQuantile(0.9, 3);
  Matrix ROpt;
  Vector tOpt;
  std::vector<size_t> inlierIndices;
  robustSinglePoseAveraging(ROpt, tOpt, inlierIndices, RVec, tVec, kappa, tau,
                            cbar);
  int inlierSize = (int)inlierIndices.size();
  printf(
      "[RobustRelativeTransform] This robot %u, neighbor %u: finds %i "
      "inliers out of %i measurements.\n",
      getID(), neighborID, inlierSize, m);
  if (inlierSize == 0) {
    throw std::runtime_error(
        "Robust single pose averaging returns empty inlier set!");
  }
  // Return transformation as matrix
  Matrix TOpt = Matrix::Identity(dimension() + 1, dimension() + 1);
  TOpt.block(0, 0, d, d) = ROpt;
  TOpt.block(0, d, d, 1) = tOpt;
  return TOpt;
}

void PGOAgent::initializeInGlobalFrame(unsigned neighborID,
                                       const PoseDict &poseDict) {
  // Require the lifting matrix to initialize
  assert(YLift);

  // Halt optimization
  bool optimizationHalted = false;
  if (isOptimizationRunning()) {
    if (mParams.verbose)
      printf("Robot %u halting optimization thread...\n", getID());
    optimizationHalted = true;
    endOptimizationLoop();
  }

  // Halt insertion of new poses
  lock_guard<mutex> tLock(mPosesMutex);

  // Halt insertion of new measurements
  lock_guard<mutex> mLock(mMeasurementsMutex);

  // Clear cache
  lock_guard<mutex> nLock(mNeighborPosesMutex);
  neighborPoseDict.clear();
  neighborAuxPoseDict.clear();

  // Compute relative transform to neighbor's frame of reference
  Matrix T_world2_world1;
  try {
    T_world2_world1 =
        computeRobustNeighborTransformTwoStage(neighborID, poseDict);
  } catch (const std::runtime_error &e) {
    printf(
        "Robust initialization is not successful! Abort and wait to try "
        "again...\n");
    return;
  }

  // Apply global transformation to local trajectory estimate
  Matrix T = TLocalInit.value();
  Matrix T_world1_frame = Matrix::Identity(d + 1, d + 1);
  Matrix T_world2_frame = Matrix::Identity(d + 1, d + 1);
  for (size_t i = 0; i < num_poses(); ++i) {
    T_world1_frame.block(0, 0, d, d + 1) = T.block(0, i * (d + 1), d, d + 1);
    T_world2_frame = T_world2_world1 * T_world1_frame;
    T.block(0, i * (d + 1), d, d + 1) = T_world2_frame.block(0, 0, d, d + 1);
  }

  // Lift back to correct relaxation rank
  X = YLift.value() * T;
  XInit.emplace(X);

  // Mark this agent as initialized
  mState = PGOAgentState::INITIALIZED;

  // Initialize auxiliary variables
  if (mParams.acceleration) {
    initializeAcceleration();
  }

  // Log initial trajectory
  if (mParams.logData) {
    mLogger.logTrajectory(dimension(), num_poses(), T,
                          "trajectory_initial.csv");
  }

  if (optimizationHalted) startOptimizationLoop(mRate);
}

void PGOAgent::update_sharedX(unsigned sharedID, const PoseDict &poseDict) {
  assert(sharedID != mID);
  for (const auto &it : poseDict) {
    const auto nID = it.first;
    const auto var = it.second;
    if (neighborSharedPoseIDs.find(nID) != neighborSharedPoseIDs.end()) {
      X_neighbor[nID] = var;
    } else {
      if (localSharedPoseIDs.find(nID) != localSharedPoseIDs.end()) {
        // std::cout<<mID<<" find  seperator "<<nID.first<<" "<<nID.second<<"
        // from robot"<<sharedID<<std::endl; std::cout<<"updating
        // seperator"<<std::endl; std::cout<<var<<std::endl;
        // std::cout<<X.block(0,nID.second*(d+1),r,d+1)<<std::endl;
        X_seperator[nID][sharedID] = var;
      }
    }
  }
  return;
}
void PGOAgent::update_sharedH(unsigned sharedID,
                              const std::map<PoseID, Matrix> &shared_H) {
  assert(sharedID != mID);
  for (const auto &it : shared_H) {
    const auto nID = it.first;
    const auto var = it.second;
    if (neighborSharedPoseIDs.find(nID) != neighborSharedPoseIDs.end()) {
      H_neighbor[nID] = var;
      // std::cout<<"neighbor:
      // "<<nID.first<<","<<nID.second<<std::endl<<var<<std::endl;
    } else {
      if (localSharedPoseIDs.find(nID) != localSharedPoseIDs.end()) {
        // std::cout<<mID<<" find  seperator "<<nID.first<<" "<<nID.second<<"
        // from robot"<<sharedID<<std::endl;
        H_seperator[nID][sharedID] = var;
        // std::cout<<"seperator: "<<nID.first<<","<<nID.second<<"from:
        // "<<sharedID<<std::endl<<var<<std::endl;
      }
    }
  }
  // std::cout<<"update shared_H done"<<std::endl;
  return;
}
void PGOAgent::updateNeighborPoses(unsigned neighborID,
                                   const PoseDict &poseDict) {
  assert(neighborID != mID);
  // Initialize this robot in the global frame, if not initialized
  const auto neighborState = getNeighborStatus(neighborID).state;
  if (mState == PGOAgentState::WAIT_FOR_INITIALIZATION &&
      neighborState == PGOAgentState::INITIALIZED) {
    initializeInGlobalFrame(neighborID, poseDict);
  }
  // Save neighbor public poses in local cache
  for (const auto &it : poseDict) {
    const auto nID = it.first;
    const auto var = it.second;
    assert(nID.first == neighborID);
    assert(var.rows() == r);
    assert(var.cols() == d + 1);
    mNumPosesReceived++;
    if (neighborSharedPoseIDs.find(nID) == neighborSharedPoseIDs.end())
      continue;
    // Only save poses from neighbors if this agent is initialized
    // and if the sending agent is also initialized
    if (mState == PGOAgentState::INITIALIZED &&
        neighborState == PGOAgentState::INITIALIZED) {
      lock_guard<mutex> lock(mNeighborPosesMutex);
      neighborPoseDict[nID] = var;
    }
  }
}

void PGOAgent::updateAuxNeighborPoses(unsigned neighborID,
                                      const PoseDict &poseDict) {
  assert(mParams.acceleration);
  assert(neighborID != mID);
  for (const auto &it : poseDict) {
    const auto nID = it.first;
    const auto var = it.second;
    assert(nID.first == neighborID);
    assert(var.rows() == r);
    assert(var.cols() == d + 1);
    mNumPosesReceived++;
    if (neighborSharedPoseIDs.find(nID) == neighborSharedPoseIDs.end())
      continue;
    // Only save poses from neighbors if this agent is initialized
    // and if the sending agent is also initialized
    if (mState == PGOAgentState::INITIALIZED &&
        getNeighborStatus(neighborID).state == PGOAgentState::INITIALIZED) {
      lock_guard<mutex> lock(mNeighborPosesMutex);
      neighborAuxPoseDict[nID] = var;
    }
  }
}

bool PGOAgent::getTrajectoryInLocalFrame(Matrix &Trajectory) {
  if (mState != PGOAgentState::INITIALIZED) {
    return false;
  }
  lock_guard<mutex> lock(mPosesMutex);

  Matrix T = X.block(0, 0, r, d).transpose() * X;
  Matrix t0 = T.block(0, d, d, 1);

  for (unsigned i = 0; i < n; ++i) {
    T.block(0, i * (d + 1), d, d) =
        projectToRotationGroup(T.block(0, i * (d + 1), d, d));
    T.block(0, i * (d + 1) + d, d, 1) = T.block(0, i * (d + 1) + d, d, 1) - t0;
  }

  Trajectory = T;
  return true;
}

bool PGOAgent::getTrajectoryInGlobalFrame(Matrix &Trajectory) {
  if (!globalAnchor) return false;
  assert(globalAnchor.value().rows() == relaxation_rank());
  assert(globalAnchor.value().cols() == dimension() + 1);
  if (mState != PGOAgentState::INITIALIZED) return false;
  lock_guard<mutex> lock(mPosesMutex);

  Matrix T = globalAnchor.value().block(0, 0, r, d).transpose() * X;
  Matrix t0 = globalAnchor.value().block(0, 0, r, d).transpose() *
              globalAnchor.value().block(0, d, r, 1);

  for (unsigned i = 0; i < n; ++i) {
    T.block(0, i * (d + 1), d, d) =
        projectToRotationGroup(T.block(0, i * (d + 1), d, d));
    T.block(0, i * (d + 1) + d, d, 1) = T.block(0, i * (d + 1) + d, d, 1) - t0;
  }

  Trajectory = T;
  return true;
}

bool PGOAgent::getPoseInGlobalFrame(unsigned int poseID, Matrix &T) {
  if (!globalAnchor) return false;
  assert(globalAnchor.value().rows() == relaxation_rank());
  assert(globalAnchor.value().cols() == dimension() + 1);
  if (mState != PGOAgentState::INITIALIZED) return false;
  lock_guard<mutex> lock(mPosesMutex);
  if (poseID < 0 || poseID >= num_poses()) return false;
  Matrix Ya = globalAnchor.value().block(0, 0, r, d);
  Matrix pa = globalAnchor.value().block(0, d, r, 1);
  Matrix t0 = Ya.transpose() * pa;
  Matrix Xi = X.block(0, poseID * (d + 1), r, d + 1);
  Matrix Ti = Ya.transpose() * Xi;
  Ti.block(0, d, d, 1) -= t0;
  assert(Ti.rows() == d);
  assert(Ti.cols() == d + 1);
  T = Ti;
  return true;
}

bool PGOAgent::getNeighborPoseInGlobalFrame(unsigned int neighborID,
                                            unsigned int poseID, Matrix &T) {
  if (!globalAnchor) return false;
  assert(globalAnchor.value().rows() == relaxation_rank());
  assert(globalAnchor.value().cols() == dimension() + 1);
  if (mState != PGOAgentState::INITIALIZED) return false;
  lock_guard<mutex> lock(mNeighborPosesMutex);
  PoseID nID = std::make_pair(neighborID, poseID);
  if (neighborPoseDict.find(nID) != neighborPoseDict.end()) {
    Matrix Ya = globalAnchor.value().block(0, 0, r, d);
    Matrix pa = globalAnchor.value().block(0, d, r, 1);
    Matrix t0 = Ya.transpose() * pa;
    Matrix Xi = neighborPoseDict.at(nID);
    assert(Xi.rows() == r);
    assert(Xi.cols() == d + 1);
    Matrix Ti = Ya.transpose() * Xi;
    Ti.block(0, d, d, 1) -= t0;
    assert(Ti.rows() == d);
    assert(Ti.cols() == d + 1);
    T = Ti;
    return true;
  }
  return false;
}

std::vector<unsigned> PGOAgent::getNeighborPublicPoses(
    const unsigned &neighborID) const {
  // Check that neighborID is indeed a neighbor of this agent
  assert(neighborRobotIDs.find(neighborID) != neighborRobotIDs.end());
  std::vector<unsigned> poseIndices;
  for (PoseID pair : neighborSharedPoseIDs) {
    if (pair.first == neighborID) {
      poseIndices.push_back(pair.second);
    }
  }
  return poseIndices;
}

std::vector<unsigned> PGOAgent::getNeighbors() const {
  std::vector<unsigned> v(neighborRobotIDs.size());
  std::copy(neighborRobotIDs.begin(), neighborRobotIDs.end(), v.begin());
  return v;
}

void PGOAgent::reset() {
  // Terminate optimization thread if running
  endOptimizationLoop();

  if (mParams.logData) {
    // Save measurements (including final weights)
    std::vector<RelativeSEMeasurement> measurements = odometry;
    measurements.insert(measurements.end(), privateLoopClosures.begin(),
                        privateLoopClosures.end());
    measurements.insert(measurements.end(), sharedLoopClosures.begin(),
                        sharedLoopClosures.end());
    mLogger.logMeasurements(measurements, "measurements.csv");

    // Save trajectory estimates after rounding
    Matrix T;
    if (getTrajectoryInGlobalFrame(T)) {
      mLogger.logTrajectory(dimension(), num_poses(), T,
                            "trajectory_optimized.csv");
      std::cout << "Saved optimized trajectory to " << mParams.logDirectory
                << std::endl;
    }

    // Save solution before rounding
    writeMatrixToFile(X, mParams.logDirectory + "X.txt");
  }

  mInstanceNumber++;
  mIterationNumber = 0;
  mNumPosesReceived = 0;

  // Assume that the old lifting matrix can still be used
  mState = PGOAgentState::WAIT_FOR_DATA;
  mStatus = PGOAgentStatus(getID(), mState, mInstanceNumber, mIterationNumber,
                           false, 0);

  odometry.clear();
  privateLoopClosures.clear();
  sharedLoopClosures.clear();

  neighborPoseDict.clear();
  neighborAuxPoseDict.clear();
  localSharedPoseIDs.clear();
  neighborSharedPoseIDs.clear();
  neighborRobotIDs.clear();
  resetTeamStatus();

  if (mProblemPtr) {
    delete mProblemPtr;
    mProblemPtr = nullptr;
  }
  mRobustCost.reset();
  globalAnchor.reset();
  TLocalInit.reset();
  XInit.reset();

  mOptimizationRequested = false;
  mPublishPublicPosesRequested = false;
  mPublishWeightsRequested = false;

  n = 1;
  X = Matrix::Zero(r, d + 1);
  X.block(0, 0, d, d) = Matrix::Identity(d, d);
}

void PGOAgent::iterate(bool doOptimization) {
  mIterationNumber++;
  // Perform iteration
  if (mState == PGOAgentState::INITIALIZED) {
    // lock pose update
    unique_lock<mutex> tLock(mPosesMutex);

    // lock measurements
    unique_lock<mutex> mLock(mMeasurementsMutex);

    // lock neighbor pose update
    unique_lock<mutex> lock(mNeighborPosesMutex);

    if (!doOptimization) {
      std::cout << "robot: " << getID() << " perform updating" << std::endl;

      step1();
    } else {
      std::cout << "robot: " << getID() << " perform consensus" << std::endl;

      step2();
    }
    // bool success;
    // success = updateX_new(doOptimization);
  }
}
void PGOAgent::step1() {
  RGrad_Y_shared_prev = shared_mProblemPtr->RieGrad(Y_shared);
  // update private variables
  X_private_Prev = X_private;
  double private_gd_stepsize = 1e-3;
  construct_private_GMatrix();
  double f_private_Init = private_mProblemPtr->f(X_private);
  double private_gradNormInit = private_mProblemPtr->RieGradNorm(X_private);
  QuadraticOptimizer private_optimizer(private_mProblemPtr);
  private_optimizer.setVerbose(false);
  private_optimizer.setAlgorithm(ROPTALG::RGD);
  private_optimizer.setGradientDescentStepsize(private_gd_stepsize);
  X_private = private_optimizer.optimize(X_private_Prev);
  double f_private_Opt = private_mProblemPtr->f(X_private);
  double private_gradNormOpt = private_mProblemPtr->RieGradNorm(X_private);
  std::cout << "private delta: " << f_private_Init - f_private_Opt << std::endl;

  // update shared variables
  construct_shared_GMatrix();
  double shared_gradNormInit = shared_mProblemPtr->RieGradNorm(Y_shared);

  double f_shared_Init = shared_mProblemPtr->f(Y_shared);

  Y_shared_Prev = Y_shared;
  double share_gd_stepsize = 1e-3;
  QuadraticOptimizer shared_optimizer(shared_mProblemPtr);
  shared_optimizer.setGradientDescentStepsize(share_gd_stepsize);
  Y_shared = shared_optimizer.gradientDescent_H(Y_shared_Prev, H_local);
  double f_shared_Opt = shared_mProblemPtr->f(Y_shared);
  double shared_gradNormOpt = shared_mProblemPtr->RieGradNorm(Y_shared);

  // for (size_t i = 0;
  //      i < localSharedPoseIDs.size() + neighborSharedPoseIDs.size(); i++) {
  //   Matrix R = Y_shared.block(0, i * (d + 1), d, d);
  //   Vector t = Y_shared.block(0, i * (d + 1) + d, d, 1);
  //   Matrix gradr = H_local.block(0, i * (d + 1), d, d);
  //   Vector gradt = H_local.block(0, i * (d + 1) + d, d, 1);
  // }
  std::cout << "shared delta: " << f_shared_Init - f_shared_Opt << std::endl;
  std::cout << "init share_RGrad: " << shared_gradNormInit
            << " after share_RGrad: " << shared_gradNormOpt << std::endl
            << std::endl;
}
void PGOAgent::step2() {
  // perform consensus
  double stepsize = 1e-1;
  double num_iter = 3;
  consensus_step(stepsize, num_iter);

  QuadraticOptimizer shared_optimizer(shared_mProblemPtr);

  // update gradient tracking term
  size_t cnt = 0;

  for (auto &poseid : localSharedPoseIDs) {
    Matrix H_sum = Matrix::Zero(d, d + 1);
    double num_of_neighbor = H_seperator[poseid].size() + 1;
    Matrix current_Y = Y_shared.block(0, cnt * (d + 1), d, d + 1);

    for (auto &it : H_seperator[poseid]) {
      H_sum += shared_optimizer.vector_transport(current_Y, it.second) /
               num_of_neighbor;
    }
    H_sum += shared_optimizer.vector_transport(
                 current_Y, H_local.block(0, cnt * (d + 1), d, d + 1)) /
             num_of_neighbor;
    H_sum -= shared_optimizer.vector_transport(
        current_Y, RGrad_Y_shared_prev.block(0, cnt * (d + 1), d, d + 1));
    H_local.block(0, cnt * (d + 1), d, d + 1) = H_sum;
    cnt++;
  }
  for (auto &poseid : neighborSharedPoseIDs) {
    Matrix H_sum = Matrix::Zero(d, d + 1);
    double num_of_neighbor = 2;
    Matrix current_Y = Y_shared.block(0, cnt * (d + 1), d, d + 1);
    H_sum += shared_optimizer.vector_transport(current_Y, H_neighbor[poseid]) /
             num_of_neighbor;
    H_sum += shared_optimizer.vector_transport(
                 current_Y, H_local.block(0, cnt * (d + 1), d, d + 1)) /
             num_of_neighbor;
    H_sum -= shared_optimizer.vector_transport(
        current_Y, RGrad_Y_shared_prev.block(0, cnt * (d + 1), d, d + 1));
    H_local.block(0, cnt * (d + 1), d, d + 1) = H_sum;
    cnt++;
  }
  H_local += shared_mProblemPtr->RieGrad(Y_shared);
  // std::cout << Y_shared.block(0, 0, 3, 3).transpose() *
  //                  H_local.block(0, 0, 3, 3)
  //           << std::endl;
  std::cout << "update gt term done" << std::endl << std::endl;
}
void PGOAgent::constructQMatrix() {
  vector<RelativeSEMeasurement> privateMeasurements = odometry;
  privateMeasurements.insert(privateMeasurements.end(),
                             privateLoopClosures.begin(),
                             privateLoopClosures.end());

  // Initialize Q with private measurements
  SparseMatrix Q = constructConnectionLaplacianSE(privateMeasurements);

  // Initialize relative SE matrix in homogeneous form
  Matrix T = Matrix::Zero(d + 1, d + 1);

  // Initialize aggregate weight matrix
  Matrix Omega = Matrix::Zero(d + 1, d + 1);

  // Go through shared loop closures
  for (const auto &m : sharedLoopClosures) {
    // Set relative SE matrix (homogeneous form)
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;

    // Set aggregate weight matrix
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;

    if (m.r1 == mID) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      assert(m.r2 != mID);

      // Modify quadratic cost
      size_t idx = m.p1;

      Matrix W = T * Omega * T.transpose();

      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < d + 1; ++row) {
          Q.coeffRef(idx * (d + 1) + row, idx * (d + 1) + col) += W(row, col);
        }
      }

    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      assert(m.r2 == mID);

      // Modify quadratic cost
      size_t idx = m.p2;

      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < d + 1; ++row) {
          Q.coeffRef(idx * (d + 1) + row, idx * (d + 1) + col) +=
              Omega(row, col);
        }
      }
    }
  }

  assert(mProblemPtr);
  mProblemPtr->setQ(Q);
}
// void PGOAgent::testQ() {
//   construct_whole_QMatrix();
//   SparseMatrix q = mProblemPtr->getQ();
//   // std::cout<<q.block(54*4,q.cols()-4,4,4)<<std::endl;
//   // std::cout<<q.block(num_poses()*4,0,4,-1)<<std::endl;
//   // std::cout<<q.rightCols(4)<<std::endl;
//   // std::cout<<q.rows()<<","<<q.cols()<<std::endl;
// }
void PGOAgent::construct_whole_QMatrix() {
  vector<RelativeSEMeasurement> privateMeasurements = odometry;

  privateMeasurements.insert(privateMeasurements.end(),
                             privateLoopClosures.begin(),
                             privateLoopClosures.end());
  // privateMeasurements.insert(privateMeasurements.end(),
  // sharedLoopClosures.begin(), sharedLoopClosures.end());

  // Initialize Q with private measurements
  // SparseMatrix Q = constructConnectionLaplacianSE(privateMeasurements);
  SparseMatrix Q = construct_whole_ConnectionLaplacianSE(
      privateMeasurements, sharedLoopClosures, neighborSharedPoseIDs);

  assert(mProblemPtr);
  mProblemPtr->setQ(Q);
}
void PGOAgent::construct_private_QMatrix() {
  vector<RelativeSEMeasurement> privateMeasurements = odometry;
  privateMeasurements.insert(privateMeasurements.end(),
                             privateLoopClosures.begin(),
                             privateLoopClosures.end());
  SparseMatrix Q = constructConnectionLaplacianSE(privateMeasurements);

  // Initialize relative SE matrix in homogeneous form
  Matrix T = Matrix::Zero(d + 1, d + 1);

  // Initialize aggregate weight matrix
  Matrix Omega = Matrix::Zero(d + 1, d + 1);
  for (const auto &m : sharedLoopClosures) {
    // Set relative SE matrix (homogeneous form)
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;
    // Set aggregate weight matrix
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }


    if (m.r1 == mID) {

      size_t index_i = m.p1;

      Matrix W = T * Omega * T.transpose();
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < d + 1; ++row) {
          Q.coeffRef(index_i * (d + 1) + row, index_i * (d + 1) + col) +=
              W(row, col);
        }
      }
    } else  {

      size_t index_j = m.p2;
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < d + 1; ++row) {
          Q.coeffRef(index_j * (d + 1) + row, index_j * (d + 1) + col) +=
              Omega(row, col);
        }
      }
    }
  }

  assert(private_mProblemPtr);
  private_mProblemPtr->setQ(Q);
}
void PGOAgent::construct_shared_QMatrix() {
  SparseMatrix Q = construct_shared_ConnectionLaplacianSE(
      shared_shared_measurements, sharedLoopClosures, localSharedPoseIDs,
      neighborSharedPoseIDs);

  Matrix T = Matrix::Zero(d + 1, d + 1);

  // Initialize aggregate weight matrix
  Matrix Omega = Matrix::Zero(d + 1, d + 1);

  // Go through shared loop closures
  for (const auto &m : private_shared_measurements) {
    // Set relative SE matrix (homogeneous form)
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;
    // Set aggregate weight matrix
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;
    auto it_i = localSharedPoseIDs.find(std::make_pair(m.r1, m.p1));
    auto it_j = localSharedPoseIDs.find(std::make_pair(m.r2, m.p2));

    if (it_i != localSharedPoseIDs.end()) {
      if (it_j != localSharedPoseIDs.end()) {
        std::cout << "share error" << std::endl;
      }
      size_t index_i = std::distance(localSharedPoseIDs.begin(), it_i);
      Matrix W = T * Omega * T.transpose();

      for (size_t col = 0; col < d + 1; ++col)
        for (size_t row = 0; row < d + 1; ++row)
          Q.coeffRef(index_i * (d + 1) + row, index_i * (d + 1) + col) +=
              W(row, col);

    } else if (it_j != localSharedPoseIDs.end()) {
      if (it_i != localSharedPoseIDs.end()) {
        std::cout << "share2 error" << std::endl;
      }
      size_t index_j = std::distance(localSharedPoseIDs.begin(), it_j);
      for (size_t col = 0; col < d + 1; ++col)
        for (size_t row = 0; row < d + 1; ++row)
          Q.coeffRef(index_j * (d + 1) + row, index_j * (d + 1) + col) +=
              Omega(row, col);
    }
  }
  assert(shared_mProblemPtr);
  shared_mProblemPtr->setQ(Q);
}
bool PGOAgent::constructGMatrix(const PoseDict &poseDict) {
  SparseMatrix G(relaxation_rank(), (dimension() + 1) * num_poses());

  for (const auto &m : sharedLoopClosures) {
    // Construct relative SE matrix in homogeneous form
    Matrix T = Matrix::Zero(d + 1, d + 1);
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;

    // Construct aggregate weight matrix
    Matrix Omega = Matrix::Zero(d + 1, d + 1);
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;

    if (m.r1 == mID) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      assert(m.r2 != mID);

      // Read neighbor's pose
      const PoseID nID = std::make_pair(m.r2, m.p2);
      auto KVpair = poseDict.find(nID);
      if (KVpair == poseDict.end()) {
        if (mParams.verbose) {
          printf(
              "constructGMatrix: robot %u cannot find neighbor pose (%u, %u)\n",
              getID(), nID.first, nID.second);
        }
        return false;
      }
      Matrix Xj = KVpair->second;

      size_t idx = m.p1;

      // Modify linear cost
      Matrix L = -Xj * Omega * T.transpose();
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < r; ++row) {
          G.coeffRef(row, idx * (d + 1) + col) += L(row, col);
        }
      }

    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      assert(m.r2 == mID);

      // Read neighbor's pose
      const PoseID nID = std::make_pair(m.r1, m.p1);
      auto KVpair = poseDict.find(nID);
      if (KVpair == poseDict.end()) {
        if (mParams.verbose) {
          printf(
              "constructGMatrix: robot %u cannot find neighbor pose (%u, %u)\n",
              getID(), nID.first, nID.second);
        }
        return false;
      }
      Matrix Xi = KVpair->second;

      size_t idx = m.p2;

      // Modify linear cost
      Matrix L = -Xi * T * Omega;
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < r; ++row) {
          G.coeffRef(row, idx * (d + 1) + col) += L(row, col);
        }
      }
    }
  }

  assert(mProblemPtr);
  mProblemPtr->setG(G);
  return true;
}

bool PGOAgent::construct_shared_GMatrix() {
  SparseMatrix G(relaxation_rank(),
                 (dimension() + 1) * (localSharedPoseIDs.size() +
                                      neighborSharedPoseIDs.size()));
  for (const auto &m : private_shared_measurements) {
    Matrix T = Matrix::Zero(d + 1, d + 1);
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;
    Matrix Omega = Matrix::Zero(d + 1, d + 1);
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;
    auto it_i = localSharedPoseIDs.find(std::make_pair(m.r1, m.p1));
    if (it_i != localSharedPoseIDs.end()) {
      size_t index_i = std::distance(localSharedPoseIDs.begin(), it_i);
      auto it_j = localprivatePoseIDs.find(std::make_pair(m.r2, m.p2));

      if (it_j == localprivatePoseIDs.end())
        std::cout << "local pose id " << m.r2 << ":" << m.p2 << "  error"
                  << std::endl;
      size_t index_j = std::distance(localprivatePoseIDs.begin(), it_j);
      Matrix Xj = X_private.block(0, index_j * (d + 1), r, d + 1);
      Matrix L = -Xj * Omega * T.transpose();
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < r; ++row) {
          G.coeffRef(row, index_i * (d + 1) + col) += L(row, col);
        }
      }
    } else {
      auto it = localprivatePoseIDs.find(std::make_pair(m.r1, m.p1));

      if (it == localprivatePoseIDs.end())
        std::cout << "neighbor assert error" << std::endl;
      size_t index_i = std::distance(localprivatePoseIDs.begin(), it);
      auto it_j = localSharedPoseIDs.find(std::make_pair(m.r2, m.p2));
      size_t index_j = std::distance(localSharedPoseIDs.begin(), it_j);
      Matrix Xi = X_private.block(0, index_i * (d + 1), r, d + 1);
      Matrix L = -Xi * T * Omega;
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < r; ++row) {
          G.coeffRef(row, index_j * (d + 1) + col) += L(row, col);
        }
      }
    }
  }
  assert(shared_mProblemPtr);
  shared_mProblemPtr->setG(G);
  return true;
}
bool PGOAgent::construct_private_GMatrix() {
  SparseMatrix G(relaxation_rank(),
                 (dimension() + 1) * localprivatePoseIDs.size());
  for (const auto &m : private_shared_measurements) {
    Matrix T = Matrix::Zero(d + 1, d + 1);
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;
    Matrix Omega = Matrix::Zero(d + 1, d + 1);
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;
    auto it_i = localprivatePoseIDs.find(std::make_pair(m.r1, m.p1));
    if (it_i != localprivatePoseIDs.end()) {
      size_t index_i = std::distance(localprivatePoseIDs.begin(), it_i);
      auto it_j = localSharedPoseIDs.find(std::make_pair(m.r2, m.p2));
      if (it_j == localSharedPoseIDs.end())
        std::cout << "assert error" << std::endl;
      size_t index_j = std::distance(localSharedPoseIDs.begin(), it_j);
      Matrix Xj = Y_shared.block(0, index_j * (d + 1), r, d + 1);

      Matrix L = -Xj * Omega * T.transpose();
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < r; ++row) {
          G.coeffRef(row, index_i * (d + 1) + col) += L(row, col);
        }
      }
    } else {
      auto it = localSharedPoseIDs.find(std::make_pair(m.r1, m.p1));
      if (it == localSharedPoseIDs.end())
        std::cout << "neighbor assert error" << std::endl;
      size_t index_i = std::distance(localSharedPoseIDs.begin(), it);
      auto it_j = localprivatePoseIDs.find(std::make_pair(m.r2, m.p2));
      size_t index_j = std::distance(localprivatePoseIDs.begin(), it_j);
      Matrix Xi = Y_shared.block(0, index_i * (d + 1), r, d + 1);
      Matrix L = -Xi * T * Omega;
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < r; ++row) {
          G.coeffRef(row, index_j * (d + 1) + col) += L(row, col);
        }
      }
    }
  }
  assert(private_mProblemPtr);
  private_mProblemPtr->setG(G);
  return true;
}
void PGOAgent::startOptimizationLoop(double freq) {
  // Asynchronous updates currently restricted to non-accelerated updates
  assert(!mParams.acceleration);

  if (isOptimizationRunning()) {
    if (mParams.verbose)
      printf("startOptimizationLoop: optimization thread already running! \n");
    return;
  }

  mRate = freq;

  mOptimizationThread = new thread(&PGOAgent::runOptimizationLoop, this);
}

void PGOAgent::runOptimizationLoop() {
  if (mParams.verbose)
    printf("Robot %u optimization thread running at %f Hz.\n", getID(), mRate);

  // Create exponential distribution with the desired rate
  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 rng(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::exponential_distribution<double> ExponentialDistribution(mRate);

  while (true) {
    double sleepUs =
        1e6 * ExponentialDistribution(rng);  // sleeping time in microsecond

    usleep(sleepUs);

    iterate(true);

    // Check if finish requested
    if (mEndLoopRequested) {
      break;
    }
  }
}

void PGOAgent::endOptimizationLoop() {
  if (!isOptimizationRunning()) return;

  mEndLoopRequested = true;

  // wait for thread to finish
  mOptimizationThread->join();

  delete mOptimizationThread;

  mOptimizationThread = nullptr;

  mEndLoopRequested = false;  // reset request flag

  if (mParams.verbose)
    printf("Robot %u optimization thread exited. \n", getID());
}

bool PGOAgent::isOptimizationRunning() {
  return mOptimizationThread != nullptr;
}

RelativeSEMeasurement &PGOAgent::findSharedLoopClosureWithNeighbor(
    const PoseID &nID) {
  const unsigned neighborID = nID.first;
  const unsigned neighborPose = nID.second;
  for (auto &m : sharedLoopClosures) {
    if ((m.r1 == neighborID && m.p1 == neighborPose) ||
        (m.r2 == neighborID && m.p2 == neighborPose)) {
      return m;
    }
  }

  // The desired measurement is not found. Throw a runtime error.
  throw std::runtime_error("Cannot find shared loop closure with neighbor.");
}

RelativeSEMeasurement &PGOAgent::findSharedLoopClosure(const PoseID &srcID,
                                                       const PoseID &dstID) {
  for (auto &m : sharedLoopClosures) {
    if (m.r1 == srcID.first && m.p1 == srcID.second && dstID.first == m.r2 &&
        dstID.second == m.p2) {
      return m;
    }
  }

  // The desired measurement is not found. Throw a runtime error.
  throw std::runtime_error("Cannot find specified shared loop closure.");
}

void PGOAgent::localInitialization() {
  std::vector<RelativeSEMeasurement> measurements = odometry;
  measurements.insert(measurements.end(), privateLoopClosures.begin(),
                      privateLoopClosures.end());

  Matrix T0;
  if (mParams.robustCostType == RobustCostType::L2) {
    T0 = chordalInitialization(dimension(), num_poses(), measurements);
  } else {
    // In robust mode, we do not trust the loop closures and hence initialize
    // from odometry
    T0 = odometryInitialization(dimension(), num_poses(), odometry);
  }

  assert(T0.rows() == d);
  assert(T0.cols() == (d + 1) * n);
  TLocalInit.emplace(T0);
}

Matrix PGOAgent::localPoseGraphOptimization() {
  // Compute initialization if necessary
  if (!TLocalInit) localInitialization();

  // Compute connection laplacian
  std::vector<RelativeSEMeasurement> measurements = odometry;
  measurements.insert(measurements.end(), privateLoopClosures.begin(),
                      privateLoopClosures.end());
  SparseMatrix Q = constructConnectionLaplacianSE(measurements);

  // Form optimization problem
  QuadraticProblem problem(n, d, d);
  problem.setQ(Q);

  // Initialize optimizer object
  QuadraticOptimizer optimizer(&problem);
  optimizer.setVerbose(mParams.verbose);
  optimizer.setTrustRegionInitialRadius(10);
  optimizer.setTrustRegionIterations(10);
  optimizer.setTrustRegionTolerance(1e-1);
  optimizer.setTrustRegionMaxInnerIterations(50);

  // Optimize
  Matrix Topt = optimizer.optimize(TLocalInit.value());
  if (mParams.verbose)
    printf("Optimization time: %f sec.\n",
           optimizer.getOptResult().elapsedMs / 1e3);
  return Topt;
}

bool PGOAgent::getLiftingMatrix(Matrix &M) const {
  assert(mID == 0);
  if (YLift.has_value()) {
    M = YLift.value();
    return true;
  }
  return false;
}

void PGOAgent::setGlobalAnchor(const Matrix &M) {
  assert(M.rows() == relaxation_rank());
  assert(M.cols() == dimension() + 1);
  globalAnchor.emplace(M);
}

bool PGOAgent::shouldTerminate() {
  // terminate if reached maximum iterations
  if (iteration_number() > mParams.maxNumIters) {
    printf("Reached maximum iterations.\n");
    return true;
  }

  for (size_t robot = 0; robot < mParams.numRobots; ++robot) {
    PGOAgentStatus robotStatus = mTeamStatus[robot];
    assert(robotStatus.agentID == robot);
    if (robotStatus.state != PGOAgentState::INITIALIZED) {
      return false;
    }
  }

  // Check if all agents are ready to terminate optimization
  for (size_t robot = 0; robot < mParams.numRobots; ++robot) {
    PGOAgentStatus robotStatus = mTeamStatus[robot];
    if (!robotStatus.readyToTerminate) {
      return false;
    }
  }

  return true;
}

bool PGOAgent::shouldRestart() const {
  if (mParams.acceleration) {
    return ((mIterationNumber + 1) % mParams.restartInterval == 0);
  }
  return false;
}

void PGOAgent::restartNesterovAcceleration(bool doOptimization) {
  if (mParams.acceleration && mState == PGOAgentState::INITIALIZED) {
    if (mParams.verbose) {
      printf("Robot %u restarts Nesteorv acceleration.\n", getID());
    }
    X = XPrev;
    updateX(doOptimization, false);
    V = X;
    Y = X;
    gamma = 0;
    alpha = 0;
  }
}

void PGOAgent::initializeAcceleration() {
  assert(mParams.acceleration);
  if (mState == PGOAgentState::INITIALIZED) {
    XPrev = X;
    gamma = 0;
    alpha = 0;
    V = X;
    Y = X;
  }
}

void PGOAgent::updateGamma() {
  assert(mParams.acceleration);
  assert(mState == PGOAgentState::INITIALIZED);
  gamma = (1 + sqrt(1 + 4 * pow(mParams.numRobots, 2) * pow(gamma, 2))) /
          (2 * mParams.numRobots);
}

void PGOAgent::updateAlpha() {
  assert(mParams.acceleration);
  assert(mState == PGOAgentState::INITIALIZED);
  alpha = 1 / (gamma * mParams.numRobots);
}

void PGOAgent::updateY() {
  assert(mParams.acceleration);
  assert(mState == PGOAgentState::INITIALIZED);
  LiftedSEManifold manifold(relaxation_rank(), dimension(), num_poses());
  Matrix M = (1 - alpha) * X + alpha * V;
  Y = manifold.project(M);
}

void PGOAgent::updateV() {
  assert(mParams.acceleration);
  assert(mState == PGOAgentState::INITIALIZED);
  LiftedSEManifold manifold(relaxation_rank(), dimension(), num_poses());
  Matrix M = V + gamma * (X - Y);
  V = manifold.project(M);
}

bool PGOAgent::updateX(bool doOptimization, bool acceleration) {
  if (!doOptimization) {
    if (acceleration) {
      X = Y;
    }
    return true;
  }

  if (mParams.verbose)
    printf("Robot %u optimize at iteration %u... \n", getID(),
           iteration_number());

  if (acceleration) assert(mParams.acceleration);

  assert(mState == PGOAgentState::INITIALIZED);

  // Update quadratic cost matrix (unless using L2 cost function since it does
  // not change measurement weights)
  if (mParams.robustCostType != RobustCostType::L2) {
    constructQMatrix();
  }

  // Construct linear cost matrix (depending on neighbor robots' poses)
  bool hasG;
  if (acceleration) {
    hasG = constructGMatrix(neighborAuxPoseDict);
  } else {
    hasG = constructGMatrix(neighborPoseDict);
  }

  // Skip update if G matrix is not constructed successfully
  if (!hasG) {
    if (mParams.verbose) {
      printf("Robot %u could not construct G matrix. Skip update...\n",
             getID());
    }
    return false;
  }

  // Initialize optimizer
  QuadraticOptimizer optimizer(mProblemPtr);
  optimizer.setVerbose(mParams.verbose);
  optimizer.setAlgorithm(mParams.algorithm);
  optimizer.setTrustRegionTolerance(1e-2);  // Force optimizer to make progress
  optimizer.setTrustRegionIterations(1);
  optimizer.setTrustRegionMaxInnerIterations(10);
  optimizer.setTrustRegionInitialRadius(100);

  // Starting solution
  Matrix XInit;
  if (acceleration) {
    XInit = Y;
  } else {
    XInit = X;
  }
  assert(XInit.rows() == relaxation_rank());
  assert(XInit.cols() == (dimension() + 1) * num_poses());

  // Optimize!
  X = optimizer.optimize(XInit);
  assert(X.rows() == relaxation_rank());
  assert(X.cols() == (dimension() + 1) * num_poses());

  // Print optimization statistics
  const auto &result = optimizer.getOptResult();
  if (mParams.verbose) {
    printf("df: %f, gn0: %f, gn1: %f, df/gn0: %f\n", result.fInit - result.fOpt,
           result.gradNormInit, result.gradNormOpt,
           (result.fInit - result.fOpt) / result.gradNormInit);
  }

  return true;
}
bool PGOAgent::updateX_new(bool doOptimization) {
  if (!doOptimization) {
    return true;
  }

  if (mParams.verbose)
    printf("consensus Robot %u optimize at iteration %u... \n", getID(),
           iteration_number());

  assert(mState == PGOAgentState::INITIALIZED);

  // Starting solution
  XPrev = X;
  YPrev = Y;
  double stepsize = 1e-1;
  // double stepsize2=1e4;
  // double stepsize3=1e0;
  double num_iter = 3;
  // double num_iter2=10;
  // double num_iter3=3;

  // if(mIterationNumber%15==0){
  //   stepsize/=2;
  //   stepsize2/=10;
  //   if(stepsize2<5e-2){
  //     stepsize2=1e-2;
  //     num_iter2=4;
  //     }
  //   stepsize3/=10;
  //     if(stepsize2<5e-2){
  //     stepsize2=1e-2;
  //     num_iter2=4;
  //     }

  // }
  // if(mIterationNumber%50==0){
  //   num_iter--;
  //   num_iter2--;
  //   num_iter3--;

  // }
  double gd_stepsize = 1e-3;
  double f_shared_Init = shared_mProblemPtr->f(Y_shared);

  // gradient_tracking_step(stepsize2,num_iter2);
  consensus_step(stepsize, num_iter);

  double f_shared_Opt = shared_mProblemPtr->f(Y_shared);
  double shared_gradNormOpt = shared_mProblemPtr->RieGradNorm(Y_shared);

  // update_private_variable_step(stepsize3,num_iter3);
  QuadraticOptimizer optimizer(mProblemPtr);
  optimizer.setVerbose(false);
  optimizer.setAlgorithm(ROPTALG::RGD);
  optimizer.setGradientDescentStepsize(gd_stepsize);

  // std::cout<<"gradient descent"<<std::endl;
  double fOpt_init = mProblemPtr->f(XPrev);
  double fwhole_opt_init = mProblemPtr->f(YPrev);
  Y = optimizer.optimize(YPrev);
  set_whole_X();
  double fOpt = mProblemPtr->f(X);
  double fwhole_opt = mProblemPtr->f(Y);
  // std::cout << "gradient descent cost " << 2 * fOpt << std::endl;
  // double gradNormOpt = mProblemPtr->RieGradNorm(X);
  // std::cout << "private init : " << f_private_Init << std::endl;

  // std::cout << "shared init : " << f_shared_Init << std::endl;
  std::cout << "shared delta: " << f_shared_Init - f_shared_Opt << std::endl;
  std::cout << "init whole_fopt: " << fOpt_init << " after whole_fopt: " << fOpt
            << std::endl;
  std::cout << "origin whole : " << fwhole_opt_init
            << " after origin whole: " << fwhole_opt << std::endl;
  // std::cout << "caculate whole:" << f_shared_Init + f_private_Init -
  // fx_shared
  //           << " ," << std::endl;
  // std::cout << "shared item: " << fy_shared << " second: " << fx_shared
  //           << " delta: " << fy_shared - fx_shared << std::endl;
  // printf("for robot:%u df: %f, gn0: %f, gn1: %f, df/gn0: %f\n", mID,
  //        fInit - fOpt, gradNormInit, gradNormOpt,
  //        (fInit - fOpt) / gradNormInit);

  return true;
}
void PGOAgent::get_distance_r(const Matrix &R,
                              const std::map<unsigned, Matrix> &sep_neighbors,
                              double &dis_r) {
  dis_r = 0;
  for (const auto &neighbor_var : sep_neighbors) {
    dis_r -=
        0.5 * (logmap(R.transpose() * neighbor_var.second.block(0, 0, 3, 3)) *
               logmap(R.transpose() * neighbor_var.second.block(0, 0, 3, 3)))
                  .trace();
  }
  return;
}
void PGOAgent::get_distance_t(const Vector &t,
                              const std::map<unsigned, Matrix> &sep_neighbors,
                              double &dis_t) {
  dis_t = 0;
  for (const auto &neighbor_var : sep_neighbors) {
    dis_t += 0.5 * (t - neighbor_var.second.col(3)).norm();
  }
  return;
}
// void PGOAgent::consensus_adaptive_stepsize(Matrix grad,Matrix)

void PGOAgent::consensus_step(double stepsize, int num_iter) {
  // std::cout << "cost before: " << 2 * shared_mProblemPtr->f(Y_shared)
  //           << std::endl;
  // go through seperator
  double finit = shared_mProblemPtr->f(Y_shared);
  size_t cnt = 0;
  for (const auto &seperator : localSharedPoseIDs) {
    Matrix gradR = Matrix::Zero(3, 3);
    Vector gradt = Vector::Zero(3);
    Matrix var;
    var = Y_shared.block(0, cnt * (d + 1), r, d + 1);
    // std::cout<<var<<std::endl<<"compare"<<std::endl;

    Matrix R = var.block(0, 0, 3, 3);
    Vector t = var.col(3);
    // std::cout<<R<<std::endl;
    //
    // std::cout<<R.transpose()*neighbor_var.second.block(0,0,3,3)<<std::endl;
    for (const auto &neighbor_var : X_seperator[seperator]) {
      gradR += logmap(R.transpose() * neighbor_var.second.block(0, 0, 3, 3));
      gradt += (neighbor_var.second.col(3) - t);
    }
    Y_shared.block(0, cnt * (d + 1), r, 3) =
        (R * expmap(stepsize * gradR)).eval();
    Y_shared.col(cnt * (d + 1) + 3) = (t + stepsize * gradt).eval();
    cnt++;
  }
  for (const auto &neighbor : neighborSharedPoseIDs) {
    Matrix gradR = Matrix::Zero(3, 3);
    Vector gradt = Vector::Zero(3);
    Matrix var;
    var = Y_shared.block(0, cnt * (d + 1), r, d + 1);
    // std::cout<<var<<std::endl<<"compare"<<std::endl;

    Matrix R = var.block(0, 0, 3, 3);
    Vector t = var.col(3);
    Matrix neighbor_var = X_neighbor[neighbor];
    gradR = logmap(R.transpose() * neighbor_var.block(0, 0, 3, 3));
    gradt = (neighbor_var.col(3) - t);
    Y_shared.block(0, cnt * (d + 1), r, 3) =
        (R * expmap(stepsize * gradR)).eval();
    Y_shared.col(cnt * (d + 1) + 3) = (t + stepsize * gradt).eval();
    cnt++;
  }
  std::cout << "consensus delta: " << shared_mProblemPtr->f(Y_shared) - finit
            << std::endl;
}

// void PGOAgent::gradient_tracking_step(double stepsize, int num_iter) {
//   std::cout << "cost before: " << 2 * mProblemPtr->f(X) << std::endl;
//   grad_prev = mProblemPtr->RieGrad(XPrev);
//   grad = mProblemPtr->RieGrad(X);
//   Matrix dgrad = grad - grad_prev;
//   // Matrix dgrad=grad-grad_prev;

//   // int num_iter=8;
//   double tau = 0.1;
//   // go through seperator
//   for (auto seperator : localSharedPoseIDs) {
//     unsigned index = seperator.second;
//     Matrix var = X.block(0, index * (d + 1), r, d + 1);
//     Matrix R = var.block(0, 0, 3, 3);
//     Vector t = var.col(3);
//     // grad
//     Matrix grad_delta = Matrix::Zero(3, 4);
//     Matrix R_local = XPrev.block(0, index * (d + 1), d, d);
//     grad_delta.block(0, 0, 3, 3) =
//         grad.block(0, index * (d + 1), d, d) -
//         R * R_local.transpose() * grad_prev.block(0, index * (d + 1), d,
//         d);
//     grad_delta.col(3) =
//         grad.col(index * (d + 1) + d) - grad_prev.col(index * (d + 1) + d);
//     // grad_delta = dgrad.block(0, index * (d + 1), r, d + 1);
//     // Matrix H_sum=H_local[seperator];

//     // does not include myself
//     Matrix H_sum = Matrix::Zero(3, 4);

//     for (auto &neighbor_var : H_seperator[seperator]) {
//       // H_sum+=neighbor_var.second;
//       Matrix neighbor_R =
//           X_seperator[seperator][neighbor_var.first].block(0, 0, 3, 3);
//       //
//       std::cout<<"H_grad"<<neighbor_R.transpose()*neighbor_var.second.block(0,0,3,3)<<std::endl;
//       H_sum.block(0, 0, 3, 3) +=
//           R * neighbor_R.transpose() * neighbor_var.second.block(0, 0, 3,
//           3);
//       H_sum.col(3) += neighbor_var.second.col(3);
//     }
//     // H_sum/=H_seperator[seperator].size()+1;
//     H_sum /= H_seperator[seperator].size();

//     // H_local[seperator]=H_sum+grad_delta;
//     H_local[seperator] = H_sum + grad_delta;
//     // std::cout<<"grad"<< H_local[seperator]<<std::endl;

//     Matrix gradR = H_local[seperator].block(0, 0, 3, 3);
//     Vector gradt = H_local[seperator].col(3);

//     // select stepsize

//     double stepsize_r, stepsize_t;
//     stepsize_r = stepsize_t = stepsize;
//     bool flag_r, flag_t, flag_update_r, flag_update_t;
//     flag_r = flag_t = flag_update_r = flag_update_t = true;
//     Matrix Xtemp = X;
//     double fx = mProblemPtr->f(X);
//     for (int j = 0; j < num_iter; j++) {
//       if (flag_r) {
//         // std::cout<<"gradR"<<R.transpose()*gradR<<std::endl;

//         Xtemp.block(0, index * (d + 1), r, d) =
//             R * expmap(-stepsize_r * R.transpose() * gradR);
//         double f_newx = mProblemPtr->f(Xtemp);
//         // std::cout<<"delta_f for R: "<<fx-f_newx<<std::endl;

//         // if(fx-f_newx<stepsize_r*rau*norm_r*norm_r)
//         if (fx - f_newx < 0)
//           stepsize_r *= tau;
//         else {
//           flag_r = false;
//           // std::cout<<"stepsize_r:"<<stepsize_r<<std::endl;
//         }
//       }
//       if (!flag_r)
//         break;
//       if (j == num_iter - 1) {
//         // std::cout<<"fail to update stepsize_r:"<<index<<std::endl;
//         flag_update_r = false;
//       }
//     }
//     for (int j = 0; j < num_iter; j++) {
//       if (flag_t) {
//         Xtemp.col(index * (d + 1) + d) = (t - stepsize_t * gradt);
//         double f_newx = mProblemPtr->f(Xtemp);
//         // std::cout<<"delta_f for t: "<<fx-f_newx<<std::endl;
//         // if(fx-f_newx<stepsize_t*rau*norm_t*norm_t)
//         if (fx - f_newx < 0)
//           stepsize_t *= tau;
//         else {
//           flag_t = false;

//           // std::cout<<"stepsize_t:"<<stepsize_t<<std::endl;
//         }
//       }
//       if (!flag_t)
//         break;
//       if (j == num_iter - 1) {
//         // std::cout<<"fail to update stepsize_t"<<index<<std::endl;
//         flag_update_r = false;
//       }
//     }
//     if (flag_update_r)
//       X.block(0, index * (d + 1), r, d) =
//           R * expmap(-stepsize_r * R.transpose() * gradR);
//     if (flag_update_t)
//       X.col(index * (d + 1) + d) = (t - stepsize_t * gradt).eval();
//     // std::cout<<" shared seperator stepsize r: "<<stepsize_r<<" stepsize
//     t:
//     // "<<stepsize_t<<std::endl;
//   }
//   // go through shared_neighbor
//   for (unsigned i = 0; i < shared_neighbor.size(); i++) {
//     unsigned index = num_poses() + i;
//     Matrix grad_delta = Matrix::Zero(3, 4);
//     Matrix var = X.block(0, index * (d + 1), r, d + 1);
//     Matrix R = var.block(0, 0, r, d);
//     Vector t = var.col(d);
//     grad_delta = grad.block(0, index * (d + 1), r, d + 1),
//     -grad_prev.block(0, index * (d + 1), r, d + 1);
//     // Matrix neighbor_H=H_neighbor[shared_neighbor[i]];
//     //
//     H_local[shared_neighbor[i]]=(neighbor_H+H_local[shared_neighbor[i]])/2+grad_delta;
//     // does not include myself
//     Matrix R_local = XPrev.block(0, index * (d + 1), d, d);
//     grad_delta.block(0, 0, 3, 3) =
//         grad.block(0, index * (d + 1), d, d) -
//         R * R_local.transpose() * grad_prev.block(0, index * (d + 1), d,
//         d);
//     // std::cout<<"delta_R "<<-R_local.transpose()*grad_prev.block(0, index
//     * (d
//     // + 1), d, d )<<std::endl;
//     grad_delta.col(3) =
//         grad.col(index * (d + 1) + d) - grad_prev.col(index * (d + 1) + d);
//     Matrix neighbor_R = X_neighbor[shared_neighbor[i]].block(0, 0, 3, 3);
//     Matrix neighbor_H = H_neighbor[shared_neighbor[i]];
//     H_local[shared_neighbor[i]].block(0, 0, 3, 3) =
//         R * neighbor_R.transpose() * neighbor_H.block(0, 0, 3, 3);
//     H_local[shared_neighbor[i]].col(3) = neighbor_H.col(3);
//     H_local[shared_neighbor[i]] += grad_delta;
//     // H_local[shared_neighbor[i]]=neighbor_H+grad_delta;

//     Matrix gradR = H_local[shared_neighbor[i]].block(0, 0, 3, 3);
//     Vector gradt = H_local[shared_neighbor[i]].col(3);

//     // select stepsize

//     double stepsize_r, stepsize_t;
//     stepsize_r = stepsize_t = stepsize;
//     bool flag_r, flag_t, flag_update_r, flag_update_t;
//     flag_r = flag_t = flag_update_r = flag_update_t = true;
//     Matrix Xtemp = X;
//     double fx = mProblemPtr->f(X);
//     for (int j = 0; j < num_iter; j++) {
//       if (flag_r) {
//         Xtemp.block(0, index * (d + 1), r, d) =
//             R * expmap(-stepsize_r * R.transpose() * gradR);
//         double f_newx = mProblemPtr->f(Xtemp);
//         // std::cout<<"delta_R "<<R.transpose()*gradR<<std::endl;

//         // if(fx-f_newx<stepsize_r*rau*norm_r*norm_r)
//         if (fx - f_newx < 0)
//           stepsize_r *= tau;
//         else
//           flag_r = false;
//       }
//       if (!flag_r)
//         break;
//       if (j == num_iter - 1) {
//         // std::cout<<"fail to update stepsize_r"<<std::endl;
//         flag_update_r = false;
//       }
//     }
//     for (int j = 0; j < num_iter; j++) {
//       if (flag_t) {
//         Xtemp.col(index * (d + 1) + d) = (t - stepsize_t * gradt);
//         double f_newx = mProblemPtr->f(Xtemp);
//         // std::cout<<"delta_f for t: "<<fx-f_newx<<std::endl;
//         // if(fx-f_newx<stepsize_t*rau*norm_t*norm_t)
//         if (fx - f_newx < 0)
//           stepsize_t *= tau;
//         else
//           flag_t = false;
//       }
//       if (!flag_t)
//         break;
//       if (j == num_iter - 1) {
//         // std::cout<<"fail to update stepsize_r"<<std::endl;
//         flag_update_r = false;
//       }
//     }

//     if (flag_update_r)
//       X.block(0, index * (d + 1), r, d) =
//           R * expmap(-stepsize_r * R.transpose() * gradR);
//     if (flag_update_t)
//       X.col(index * (d + 1) + d) = (t - stepsize_t * gradt).eval();
//     // std::cout<<" shared neighbor stepsize r: "<<stepsize_r<<" stepsize
//     t:
//     // "<<stepsize_t<<std::endl;
//   }
//   std::cout << "cost after: " << 2 * mProblemPtr->f(X) << std::endl;

//   std::cout << "updating gradient tracking done" << std::endl;
// }

void PGOAgent::resetTeamStatus() {
  mTeamStatus.clear();
  for (unsigned robot = 0; robot < mParams.numRobots; ++robot) {
    mTeamStatus.emplace_back(robot);
  }
}

bool PGOAgent::shouldUpdateLoopClosureWeights() const {
  // No need to update weight if using L2 cost
  if (mParams.robustCostType == RobustCostType::L2) return false;

  return ((mIterationNumber + 1) % mParams.robustOptInnerIters == 0);
}

void PGOAgent::updateLoopClosuresWeights() {
  assert(mState == PGOAgentState::INITIALIZED);

  // Update private loop closures
  for (auto &m : privateLoopClosures) {
    if (m.isKnownInlier) continue;
    Matrix Y1 = X.block(0, m.p1 * (d + 1), r, d);
    Matrix p1 = X.block(0, m.p1 * (d + 1) + d, r, 1);
    Matrix Y2 = X.block(0, m.p2 * (d + 1), r, d);
    Matrix p2 = X.block(0, m.p2 * (d + 1) + d, r, 1);
    double residual = std::sqrt(computeMeasurementError(m, Y1, p1, Y2, p2));
    double weight = mRobustCost.weight(residual);
    m.weight = weight;
    if (mParams.verbose) {
      printf(
          "Agent %u update edge: (%u, %u) -> (%u, %u), residual = %f, "
          "weight = %f \n",
          getID(), m.r1, m.p1, m.r2, m.p2, residual, weight);
    }
  }

  // Update shared loop closures
  // Agent i is only responsible for updating edge weights with agent j,where j>
  // i
  for (auto &m : sharedLoopClosures) {
    if (m.isKnownInlier) continue;
    Matrix Y1, Y2, p1, p2;
    if (m.r1 == getID()) {
      if (m.r2 < getID()) continue;

      Y1 = X.block(0, m.p1 * (d + 1), r, d);
      p1 = X.block(0, m.p1 * (d + 1) + d, r, 1);
      const PoseID nbrPoseID = std::make_pair(m.r2, m.p2);
      auto KVpair = neighborPoseDict.find(nbrPoseID);
      if (KVpair == neighborPoseDict.end()) {
        printf("Agent %u cannot update edge: (%u, %u) -> (%u, %u). \n", getID(),
               m.r1, m.p1, m.r2, m.p2);
        continue;
      }
      Matrix X2 = KVpair->second;
      Y2 = X2.block(0, 0, r, d);
      p2 = X2.block(0, d, r, 1);
    } else {
      if (m.r1 < getID()) continue;

      Y2 = X.block(0, m.p2 * (d + 1), r, d);
      p2 = X.block(0, m.p2 * (d + 1) + d, r, 1);
      const PoseID nbrPoseID = std::make_pair(m.r1, m.p1);
      auto KVpair = neighborPoseDict.find(nbrPoseID);
      if (KVpair == neighborPoseDict.end()) {
        printf("Agent %u cannot update edge: (%u, %u) -> (%u, %u). \n", getID(),
               m.r1, m.p1, m.r2, m.p2);
        continue;
      }
      Matrix X1 = KVpair->second;
      Y1 = X1.block(0, 0, r, d);
      p1 = X1.block(0, d, r, 1);
    }
    double residual = std::sqrt(computeMeasurementError(m, Y1, p1, Y2, p2));
    double weight = mRobustCost.weight(residual);
    m.weight = weight;
    if (mParams.verbose) {
      printf(
          "Agent %u update edge: (%u, %u) -> (%u, %u), residual = %f, "
          "weight = %f \n",
          getID(), m.r1, m.p1, m.r2, m.p2, residual, weight);
    }
  }
  mPublishWeightsRequested = true;
}

double PGOAgent::computeConvergedLoopClosureRatio() {
  // Currently, this function is only meaningful for GNC_TLS
  if (mParams.robustCostType != RobustCostType::GNC_TLS) {
    return 1.0;
  }

  double totalCount = 0;
  double acceptCount = 0;
  double rejectCount = 0;
  for (const auto &m : privateLoopClosures) {
    if (m.isKnownInlier) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }
  for (const auto &m : sharedLoopClosures) {
    if (m.isKnownInlier) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }
  double convergedCount = acceptCount + rejectCount;

  if (mParams.verbose) {
    printf(
        "Robot %u :\n "
        "accepted loop closures: %i\n "
        "rejected loop closures: %i\n "
        "undecided loop closures: %i\n",
        getID(), (int)acceptCount, (int)rejectCount,
        (int)(totalCount - convergedCount));
  }

  return convergedCount / totalCount;
}

bool PGOAgent::isDuplicateMeasurement(
    const RelativeSEMeasurement &m,
    const vector<RelativeSEMeasurement> &measurements) {
  for (const RelativeSEMeasurement &m2 : measurements) {
    if (m.r1 == m2.r1 && m.r2 == m2.r2 && m.p1 == m2.p1 && m.p2 == m2.p2) {
      return true;
    }
  }
  return false;
}

}  // namespace DPGO

/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/PGOAgent.h>
#include <DPGO/QuadraticProblem.h>

#include <Eigen/Geometry>

#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace DPGO;

namespace {

std::string lowercase(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

Matrix orthonormalComplement(const Matrix &Y) {
  const unsigned rows = static_cast<unsigned>(Y.rows());
  const unsigned cols = static_cast<unsigned>(Y.cols());
  if (rows <= cols) {
    return Matrix::Zero(rows, 0);
  }

  Matrix Q = Matrix::Zero(rows, rows - cols);
  std::vector<Vector> basis;
  basis.reserve(rows);
  for (unsigned col = 0; col < cols; ++col) {
    Vector v = Y.col(col);
    const double norm = v.norm();
    if (norm > 1e-12) {
      basis.push_back(v / norm);
    }
  }

  unsigned outCol = 0;
  for (unsigned row = 0; row < rows && outCol < rows - cols; ++row) {
    Vector v = Vector::Zero(rows);
    v(row) = 1.0;
    for (const auto &b : basis) {
      v -= b * b.dot(v);
    }
    const double norm = v.norm();
    if (norm > 1e-10) {
      v /= norm;
      Q.col(outCol++) = v;
      basis.push_back(v);
    }
  }
  return Q.leftCols(outCol);
}

size_t liftedTangentDeltaDoubles(unsigned r, unsigned d) {
  const size_t skewDoubles = static_cast<size_t>(d) * (d - 1) / 2;
  const size_t normalDoubles =
      r > d ? static_cast<size_t>(r - d) * d : 0;
  const size_t translationDoubles = r;
  return skewDoubles + normalDoubles + translationDoubles;
}

Matrix reconstructLiftedTangentDelta(const Matrix &previousPose,
                                     const Matrix &sharedPose, unsigned d) {
  const unsigned r = static_cast<unsigned>(previousPose.rows());
  Matrix reconstructed = previousPose;
  const Matrix Y = previousPose.leftCols(d);
  const Matrix deltaY = sharedPose.leftCols(d) - Y;
  const Matrix ytDelta = Y.transpose() * deltaY;
  const Matrix skew = 0.5 * (ytDelta - ytDelta.transpose());
  const Matrix Q = orthonormalComplement(Y);

  Matrix tangent = Y * skew;
  if (Q.cols() > 0) {
    tangent += Q * (Q.transpose() * deltaY);
  }

  reconstructed.leftCols(d) = projectToStiefelManifold(Y + tangent);
  reconstructed.col(d) = previousPose.col(d) +
                         (sharedPose.col(d) - previousPose.col(d));
  LiftedSEManifold poseManifold(r, d, 1);
  return poseManifold.project(reconstructed);
}

bool writePoseRows(const string &path, const vector<PGOAgent *> &agents,
                   unsigned numRobots, unsigned numPoses, unsigned posesPerRobot,
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
  for (unsigned robot = 0; robot < numRobots; ++robot) {
    Matrix T;
    if (!agents[robot]->getTrajectoryInGlobalFrame(T)) {
      return false;
    }
    const unsigned startIdx = robot * posesPerRobot;
    unsigned endIdx = (robot + 1) * posesPerRobot;
    if (robot == numRobots - 1) {
      endIdx = numPoses;
    }
    for (unsigned idx = startIdx; idx < endIdx; ++idx) {
      const unsigned localIdx = idx - startIdx;
      Eigen::Matrix3d R = T.block(0, localIdx * (d + 1), d, d);
      Eigen::Quaterniond q(R);
      q.normalize();
      Matrix t = T.block(0, localIdx * (d + 1) + d, d, 1);
      out << idx << ' ' << t(0) << ' ' << t(1) << ' ' << t(2) << ' '
          << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w()
          << '\n';
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc < 3) {
    cout << "Multi-robot pose graph optimization example. " << endl;
    cout << "Usage: " << argv[0]
         << " [# robots] [input .g2o file] [max iters] [grad tol]"
         << " [relative loss tol] [csv path] [RTR iterations]"
         << " [RTR max inner iters] [RTR tol] [RTR initial radius]"
         << " [stable loss rounds]"
         << " [baseline_full|selected|decentralized|decentralized_event|decentralized_event_colored|decentralized_residual|decentralized_residual_colored|decentralized_residual_graph_colored|decentralized_residual_colored_sweep|decentralized_residual_colored_predictive|decentralized_residual_colored_conflict|decentralized_residual_colored_conflict_refine|decentralized_residual_colored_conflict_sweep|decentralized_residual_colored_winner_refine|decentralized_residual_colored_feedback_refine|decentralized_residual_colored_curvature_feedback|decentralized_residual_colored_schur_feedback|decentralized_residual_colored_schur_late_feedback|decentralized_residual_async_schur_late_feedback|decentralized_hybrid_warmup_async_schur_late_feedback|decentralized_adaptive_budget_async_schur_late_feedback|decentralized_majorized_boundary_async_schur_late_feedback|decentralized_boundary_jacobi_correction|decentralized_boundary_model_async_schur_late_feedback|decentralized_boundary_model_packet_async_schur_late_feedback|decentralized_boundary_precond_packet_async_schur_late_feedback|decentralized_boundary_response_async_schur_late_feedback|decentralized_boundary_batch_model_async_schur_late_feedback|decentralized_budgeted_boundary_batch_model_async_schur_late_feedback|decentralized_boundary_batch_response_async_schur_late_feedback|decentralized_boundary_schur_response_async_schur_late_feedback|decentralized_budgeted_boundary_schur_response_async_schur_late_feedback|decentralized_boundary_packet_schur_response_async_schur_late_feedback|decentralized_amm_boundary_async_schur_late_feedback|decentralized_local_amm_async_schur_late_feedback|decentralized_local_amm_interface_delta_async_schur_late_feedback|decentralized_interface_state_async_schur_late_feedback|decentralized_interface_delta_async_schur_late_feedback]"
         << " [decentralized refresh period] [event pose tol]"
         << " [event max age] [event local grad tol]"
         << " [event budget fraction] [event max poses per neighbor]"
         << " [RTR|RGD] [event residual tol] [acceleration 0|1]"
         << " [neighbor direction gain] [pose output path]"
         << " [hybrid warmup rounds]" << endl;
    exit(1);
  }

  cout << "Multi-robot pose graph optimization example. " << endl;

  int num_robots = atoi(argv[1]);
  if (num_robots <= 0) {
    cout << "Number of robots must be positive!" << endl;
    exit(1);
  }
  cout << "Simulating " << num_robots << " robots." << endl;

  size_t num_poses;
  vector<RelativeSEMeasurement> dataset = read_g2o_file(argv[2], num_poses);
  cout << "Loaded dataset from file " << argv[2] << "." << endl;

  /**
  ###########################################
  Options
  ###########################################
  */
  unsigned int n, d, r;
  d = (!dataset.empty() ? dataset[0].t.size() : 0);
  n = num_poses;
  r = 5;
  if (const char *rankEnv = std::getenv("DRAN_RELAXATION_RANK")) {
    const int requestedRank = std::atoi(rankEnv);
    if (requestedRank >= static_cast<int>(d)) {
      r = static_cast<unsigned>(requestedRank);
    } else {
      cerr << "Ignoring DRAN_RELAXATION_RANK=" << rankEnv
           << " because it is smaller than problem dimension d=" << d
           << "." << endl;
    }
  }
  cout << "Using relaxation rank r=" << r << "." << endl;
  bool acceleration = false;
  bool verbose = false;
  unsigned numIters = 300;
  if (argc >= 4) {
    numIters = static_cast<unsigned>(atoi(argv[3]));
  }
  double gradTol = 1e-2;
  if (argc >= 5) {
    gradTol = atof(argv[4]);
  }
  double relCostTol = 1e-6;
  if (argc >= 6) {
    relCostTol = atof(argv[5]);
  }
  string csvPath;
  if (argc >= 7) {
    csvPath = argv[6];
  }
  unsigned trustRegionIterations = 5;
  if (argc >= 8) {
    trustRegionIterations = static_cast<unsigned>(atoi(argv[7]));
  }
  int trustRegionMaxInnerIterations = 50;
  if (argc >= 9) {
    trustRegionMaxInnerIterations = atoi(argv[8]);
  }
  double trustRegionTolerance = 1e-4;
  if (argc >= 10) {
    trustRegionTolerance = atof(argv[9]);
  }
  double trustRegionInitialRadius = 100;
  if (argc >= 11) {
    trustRegionInitialRadius = atof(argv[10]);
  }
  unsigned stableLossRequired = 5;
  if (argc >= 12) {
    stableLossRequired = static_cast<unsigned>(atoi(argv[11]));
  }
  string updateMode = "selected";
  if (argc >= 13) {
    updateMode = argv[12];
  }
  unsigned decentralizedRefreshPeriod = 1;
  if (argc >= 14) {
    decentralizedRefreshPeriod = static_cast<unsigned>(atoi(argv[13]));
    if (decentralizedRefreshPeriod == 0) {
      decentralizedRefreshPeriod = 1;
    }
  }
  double eventPoseTol = 1e-3;
  if (argc >= 15) {
    eventPoseTol = atof(argv[14]);
  }
  unsigned eventMaxAge = 5;
  if (argc >= 16) {
    eventMaxAge = static_cast<unsigned>(atoi(argv[15]));
    if (eventMaxAge == 0) {
      eventMaxAge = 1;
    }
  }
  double eventLocalGradTol = 0.0;
  if (argc >= 17) {
    eventLocalGradTol = atof(argv[16]);
  }
  double eventBudgetFraction = 1.0;
  if (argc >= 18) {
    eventBudgetFraction = atof(argv[17]);
    if (eventBudgetFraction <= 0.0) {
      eventBudgetFraction = 1.0;
    }
  }
  unsigned eventMaxPosesPerNeighbor = 0;
  if (argc >= 19) {
    eventMaxPosesPerNeighbor = static_cast<unsigned>(atoi(argv[18]));
  }
  ROPTALG localAlgorithm = ROPTALG::RTR;
  if (argc >= 20) {
    string algorithmName = argv[19];
    if (algorithmName == "RGD" || algorithmName == "rgd") {
      localAlgorithm = ROPTALG::RGD;
    } else {
      localAlgorithm = ROPTALG::RTR;
    }
  }
  double eventResidualTol = 1e-4;
  if (argc >= 21) {
    eventResidualTol = atof(argv[20]);
  }
  if (argc >= 22) {
    acceleration = atoi(argv[21]) != 0;
  }
  double neighborDirectionGain = 0.5;
  if (argc >= 23) {
    neighborDirectionGain = atof(argv[22]);
    if (neighborDirectionGain < 0.0) {
      neighborDirectionGain = 0.0;
    }
  }
  string poseOutputPath;
  if (argc >= 24) {
    poseOutputPath = argv[23];
  } else if (const char *envPath = std::getenv("DPGO_POSE_OUTPUT")) {
    poseOutputPath = envPath;
  }
  auto parseNonnegativeUnsigned = [](const char *value) {
    const int parsed = atoi(value);
    return parsed > 0 ? static_cast<unsigned>(parsed) : 0u;
  };
  unsigned hybridWarmupRounds = 0;
  if (const char *envWarmup = std::getenv("DRAN_HYBRID_WARMUP_ROUNDS")) {
    hybridWarmupRounds = parseNonnegativeUnsigned(envWarmup);
  }
  if (argc >= 25) {
    hybridWarmupRounds = parseNonnegativeUnsigned(argv[24]);
  }
  const string configuredUpdateMode = updateMode;
  const bool hybridWarmupMode =
      configuredUpdateMode ==
      "decentralized_hybrid_warmup_async_schur_late_feedback";
  const bool adaptiveBudgetMode =
      configuredUpdateMode ==
      "decentralized_adaptive_budget_async_schur_late_feedback";
  const bool majorizedBoundaryMode =
      configuredUpdateMode ==
      "decentralized_majorized_boundary_async_schur_late_feedback";
  const bool boundaryJacobiMode =
      configuredUpdateMode == "decentralized_boundary_jacobi_correction";
  const bool boundaryModelMode =
      configuredUpdateMode ==
      "decentralized_boundary_model_async_schur_late_feedback";
  const bool boundaryPrecondPacketMode =
      configuredUpdateMode ==
      "decentralized_boundary_precond_packet_async_schur_late_feedback";
  const bool boundaryModelPacketMode =
      configuredUpdateMode ==
          "decentralized_boundary_model_packet_async_schur_late_feedback" ||
      boundaryPrecondPacketMode;
  const bool boundaryResponseMode =
      configuredUpdateMode ==
      "decentralized_boundary_response_async_schur_late_feedback";
  const bool budgetedBoundarySchurResponseMode =
      configuredUpdateMode ==
      "decentralized_budgeted_boundary_schur_response_async_schur_late_feedback";
  const bool budgetedBoundaryBatchModelMode =
      configuredUpdateMode ==
          "decentralized_budgeted_boundary_batch_model_async_schur_late_feedback" ||
      budgetedBoundarySchurResponseMode;
  const bool boundaryBatchResponseMode =
      configuredUpdateMode ==
      "decentralized_boundary_batch_response_async_schur_late_feedback";
  const bool boundaryPacketSchurResponseMode =
      configuredUpdateMode ==
      "decentralized_boundary_packet_schur_response_async_schur_late_feedback";
  const bool boundarySchurResponseMode =
      configuredUpdateMode ==
          "decentralized_boundary_schur_response_async_schur_late_feedback" ||
      boundaryPacketSchurResponseMode || budgetedBoundarySchurResponseMode;
  const bool ammBoundaryMode =
      configuredUpdateMode ==
      "decentralized_amm_boundary_async_schur_late_feedback";
  const bool localAmmMode =
      configuredUpdateMode ==
          "decentralized_local_amm_async_schur_late_feedback" ||
      configuredUpdateMode ==
          "decentralized_local_amm_interface_delta_async_schur_late_feedback";
  const bool boundaryBatchResponseLikeMode =
      boundaryBatchResponseMode || boundarySchurResponseMode;
  const bool boundaryBatchModelMode =
      configuredUpdateMode ==
          "decentralized_boundary_batch_model_async_schur_late_feedback" ||
      budgetedBoundaryBatchModelMode || boundaryBatchResponseLikeMode;
  const bool interfaceStateMode =
      configuredUpdateMode ==
      "decentralized_interface_state_async_schur_late_feedback";
  const bool interfaceDeltaMode =
      configuredUpdateMode ==
          "decentralized_interface_delta_async_schur_late_feedback" ||
      configuredUpdateMode ==
          "decentralized_local_amm_interface_delta_async_schur_late_feedback";
  const bool configuredAsyncSchurMode =
      configuredUpdateMode ==
          "decentralized_residual_async_schur_late_feedback" ||
      hybridWarmupMode || adaptiveBudgetMode || majorizedBoundaryMode ||
      boundaryModelMode || boundaryModelPacketMode || boundaryResponseMode ||
      boundaryBatchModelMode || ammBoundaryMode || localAmmMode ||
      interfaceStateMode || interfaceDeltaMode;
  auto parseEnvUnsigned = [](const char *name, unsigned fallback) {
    if (const char *value = std::getenv(name)) {
      const int parsed = atoi(value);
      if (parsed >= 0) {
        return static_cast<unsigned>(parsed);
      }
    }
    return fallback;
  };
  auto parseEnvDouble = [](const char *name, double fallback) {
    if (const char *value = std::getenv(name)) {
      char *end = nullptr;
      const double parsed = std::strtod(value, &end);
      if (end != value && std::isfinite(parsed)) {
        return parsed;
      }
    }
    return fallback;
  };
  auto parseEnvBool = [](const char *name, bool fallback) {
    if (const char *value = std::getenv(name)) {
      const std::string parsed(value);
      if (parsed == "1" || parsed == "true" || parsed == "TRUE" ||
          parsed == "on" || parsed == "ON") {
        return true;
      }
      if (parsed == "0" || parsed == "false" || parsed == "FALSE" ||
          parsed == "off" || parsed == "OFF") {
        return false;
      }
    }
    return fallback;
  };
  const unsigned adaptiveBudgetExtraRTR =
      parseEnvUnsigned("DRAN_ADAPTIVE_LOCAL_BUDGET_EXTRA_RTR", 1);
  const double adaptiveBudgetResidualTol =
      parseEnvDouble("DRAN_ADAPTIVE_LOCAL_BUDGET_RESIDUAL_TOL",
                     std::max(0.0, eventResidualTol));
  const unsigned adaptiveBudgetMinSeparatorPoses =
      parseEnvUnsigned("DRAN_ADAPTIVE_LOCAL_BUDGET_MIN_SEPARATOR_POSES", 1);
  const unsigned adaptiveBudgetMaxRobots =
      parseEnvUnsigned("DRAN_ADAPTIVE_LOCAL_BUDGET_MAX_ROBOTS", 0);
  const double boundaryJacobiStep =
      parseEnvDouble("DRAN_BOUNDARY_JACOBI_STEP", 0.2);
  const double boundaryJacobiMaxBlockNorm =
      parseEnvDouble("DRAN_BOUNDARY_JACOBI_MAX_BLOCK_NORM", 0.02);
  const bool boundaryJacobiRequireDecrease =
      parseEnvBool("DRAN_BOUNDARY_JACOBI_REQUIRE_DECREASE", true);
  const unsigned boundaryJacobiBacktrackingSteps =
      parseEnvUnsigned("DRAN_BOUNDARY_JACOBI_BACKTRACKING_STEPS", 5);
  const double boundaryModelStep =
      parseEnvDouble("DRAN_BOUNDARY_MODEL_STEP", 0.2);
  const double boundaryModelMaxBlockNorm =
      parseEnvDouble("DRAN_BOUNDARY_MODEL_MAX_BLOCK_NORM", 0.02);
  const bool boundaryModelRequireDecrease =
      parseEnvBool("DRAN_BOUNDARY_MODEL_REQUIRE_DECREASE", true);
  const unsigned boundaryModelBacktrackingSteps =
      parseEnvUnsigned("DRAN_BOUNDARY_MODEL_BACKTRACKING_STEPS", 5);
  const double boundaryModelGain =
      parseEnvDouble("DRAN_BOUNDARY_MODEL_GAIN", 1.0);
  std::string boundaryPacketStepMode =
      boundaryPrecondPacketMode ? "preconditioned" : "gradient";
  if (const char *modeEnv = std::getenv("DRAN_BOUNDARY_PACKET_STEP_MODE")) {
    boundaryPacketStepMode = lowercase(modeEnv);
  }
  if (boundaryPacketStepMode == "precond" ||
      boundaryPacketStepMode == "schur") {
    boundaryPacketStepMode = "preconditioned";
  }
  if (boundaryPacketStepMode != "gradient" &&
      boundaryPacketStepMode != "preconditioned") {
    std::cerr << "Unknown DRAN_BOUNDARY_PACKET_STEP_MODE='"
              << boundaryPacketStepMode << "', using gradient." << std::endl;
    boundaryPacketStepMode = "gradient";
  }
  const bool boundaryPacketUsePreconditionedStep =
      boundaryPacketStepMode == "preconditioned";
  const double boundaryResponseStep =
      parseEnvDouble("DRAN_BOUNDARY_RESPONSE_STEP", boundaryModelStep);
  const double boundaryResponseGain =
      parseEnvDouble("DRAN_BOUNDARY_RESPONSE_GAIN", 1.0);
  const double boundaryResponseMaxBlockNorm =
      parseEnvDouble("DRAN_BOUNDARY_RESPONSE_MAX_BLOCK_NORM",
                     boundaryModelMaxBlockNorm);
  const bool boundaryResponseRequireDecrease =
      parseEnvBool("DRAN_BOUNDARY_RESPONSE_REQUIRE_DECREASE", true);
  const unsigned boundaryResponseBacktrackingSteps =
      parseEnvUnsigned("DRAN_BOUNDARY_RESPONSE_BACKTRACKING_STEPS",
                       boundaryModelBacktrackingSteps);
  const bool boundaryBatchModelRequireMerit =
      parseEnvBool("DRAN_BOUNDARY_BATCH_MODEL_REQUIRE_MERIT", true);
  const double boundaryBatchResponseStep =
      parseEnvDouble("DRAN_BOUNDARY_BATCH_RESPONSE_STEP",
                     boundaryResponseStep);
  const double boundaryBatchResponseGain =
      parseEnvDouble("DRAN_BOUNDARY_BATCH_RESPONSE_GAIN",
                     boundaryResponseGain);
  const double boundaryBatchResponseMaxBlockNorm =
      parseEnvDouble("DRAN_BOUNDARY_BATCH_RESPONSE_MAX_BLOCK_NORM",
                     boundaryResponseMaxBlockNorm);
  const bool boundaryBatchResponseRequireDecrease =
      parseEnvBool("DRAN_BOUNDARY_BATCH_RESPONSE_REQUIRE_DECREASE",
                   boundaryResponseRequireDecrease);
  const unsigned boundaryBatchResponseBacktrackingSteps =
      parseEnvUnsigned("DRAN_BOUNDARY_BATCH_RESPONSE_BACKTRACKING_STEPS",
                       boundaryResponseBacktrackingSteps);
  const bool boundaryBatchResponsePostRefine =
      parseEnvBool("DRAN_BOUNDARY_BATCH_RESPONSE_POST_REFINE", false);
  const double boundarySchurResponseStep =
      parseEnvDouble("DRAN_BOUNDARY_SCHUR_RESPONSE_STEP",
                     boundaryBatchResponseStep);
  const double boundarySchurResponseGain =
      parseEnvDouble("DRAN_BOUNDARY_SCHUR_RESPONSE_GAIN",
                     boundaryBatchResponseGain);
  const double boundarySchurResponseMaxBlockNorm =
      parseEnvDouble("DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCK_NORM",
                     boundaryBatchResponseMaxBlockNorm);
  const double boundarySchurResponseDamping =
      parseEnvDouble("DRAN_BOUNDARY_SCHUR_RESPONSE_DAMPING", 1e-2);
  const unsigned boundarySchurResponseMaxBlocks =
      parseEnvUnsigned("DRAN_BOUNDARY_SCHUR_RESPONSE_MAX_BLOCKS", 80);
  const bool boundarySchurResponseRequireDecrease =
      parseEnvBool("DRAN_BOUNDARY_SCHUR_RESPONSE_REQUIRE_DECREASE",
                   boundaryBatchResponseRequireDecrease);
  const unsigned boundarySchurResponseBacktrackingSteps =
      parseEnvUnsigned("DRAN_BOUNDARY_SCHUR_RESPONSE_BACKTRACKING_STEPS",
                       boundaryBatchResponseBacktrackingSteps);
  const bool boundarySchurResponsePostRefine =
      parseEnvBool("DRAN_BOUNDARY_SCHUR_RESPONSE_POST_REFINE",
                   boundaryBatchResponsePostRefine);
  const bool boundarySchurResponseSkipBaselineRefine =
      parseEnvBool("DRAN_BOUNDARY_SCHUR_RESPONSE_SKIP_BASELINE_REFINE",
                   false);
  const unsigned boundarySchurResponseBaselineRefinePeriod =
      parseEnvUnsigned("DRAN_BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_PERIOD",
                       0);
  const unsigned boundarySchurResponseBaselineRefineOffset =
      parseEnvUnsigned("DRAN_BOUNDARY_SCHUR_RESPONSE_BASELINE_REFINE_OFFSET",
                       0);
  const double boundaryPacketSchurDampingGain =
      parseEnvDouble("DRAN_BOUNDARY_PACKET_SCHUR_DAMPING_GAIN",
                     boundaryPacketSchurResponseMode ? 0.01 : 0.0);
  const double boundaryPacketSchurDampingMax =
      parseEnvDouble("DRAN_BOUNDARY_PACKET_SCHUR_DAMPING_MAX", 1.0);
  const double ammBoundaryBeta =
      std::max(0.0, parseEnvDouble("DRAN_AMM_BOUNDARY_BETA", 0.6));
  const double ammBoundaryMaxStepScale =
      std::max(0.0, parseEnvDouble("DRAN_AMM_BOUNDARY_MAX_STEP_SCALE", 2.0));
  const bool localAmmRequireDecrease =
      parseEnvBool("DRAN_LOCAL_AMM_REQUIRE_DECREASE", true);
  const bool localAmmRequireGradNonIncrease =
      parseEnvBool("DRAN_LOCAL_AMM_REQUIRE_GRAD_NONINCREASE", false);
  std::string localSolverName = "roptlib";
  if (const char *solverEnv = std::getenv("DRAN_LOCAL_SOLVER")) {
    localSolverName = lowercase(solverEnv);
  }
  const bool useManualLocalSolver =
      localSolverName == "manual_newton" || localSolverName == "manual_tr" ||
      localSolverName == "handwritten_newton";
  const bool useReducedRotationLocalSolver =
      localSolverName == "reduced_rotation" ||
      localSolverName == "reduced_rotation_newton" ||
      localSolverName == "translation_eliminated";
  if (!useManualLocalSolver && !useReducedRotationLocalSolver &&
      localSolverName != "roptlib" &&
      localSolverName != "roptlib_rtr") {
    std::cerr << "Unknown DRAN_LOCAL_SOLVER='" << localSolverName
              << "', using roptlib." << std::endl;
    localSolverName = "roptlib";
  }
  const bool localAmmCompareBaseline =
      parseEnvBool("DRAN_LOCAL_AMM_COMPARE_BASELINE", false);
  const bool localAmmRequireBeatBaseline =
      parseEnvBool("DRAN_LOCAL_AMM_REQUIRE_BEAT_BASELINE", true);
  const double boundaryBatchBudgetFraction = parseEnvDouble(
      "DRAN_BOUNDARY_BATCH_BUDGET_FRACTION",
      parseEnvDouble("DRAN_BOUNDARY_BATCH_MODEL_BUDGET_FRACTION", 1.0));
  const unsigned boundaryBatchMaxPosesPerReceiver = parseEnvUnsigned(
      "DRAN_BOUNDARY_BATCH_MAX_POSES_PER_RECEIVER",
      parseEnvUnsigned("DRAN_BOUNDARY_BATCH_MODEL_MAX_POSES_PER_RECEIVER", 0));
  const unsigned interfaceDeltaTopK =
      parseEnvUnsigned("DRAN_INTERFACE_DELTA_TOPK", 4);
  std::string interfaceDeltaCodec = "hybrid";
  if (const char *codecEnv = std::getenv("DRAN_INTERFACE_DELTA_CODEC")) {
    interfaceDeltaCodec = lowercase(codecEnv);
  } else if (const char *codecEnv =
                 std::getenv("DRAN_INTERFACE_DELTA_PAYLOAD")) {
    interfaceDeltaCodec = lowercase(codecEnv);
  }
  if (interfaceDeltaCodec == "sparse_matrix") {
    interfaceDeltaCodec = "sparse";
  } else if (interfaceDeltaCodec == "lifted_tangent" ||
             interfaceDeltaCodec == "stiefel_tangent") {
    interfaceDeltaCodec = "tangent";
  } else if (interfaceDeltaCodec != "sparse" &&
             interfaceDeltaCodec != "hybrid" &&
             interfaceDeltaCodec != "tangent") {
    std::cerr << "Unknown DRAN_INTERFACE_DELTA_CODEC='"
              << interfaceDeltaCodec << "', using sparse." << std::endl;
    interfaceDeltaCodec = "sparse";
  }
  const double interfaceDeltaMaxReconError =
      parseEnvDouble("DRAN_INTERFACE_DELTA_MAX_RECON_ERROR",
                     0.5 * eventPoseTol);

  ofstream csv;
  if (!csvPath.empty()) {
    csv.open(csvPath);
    csv << "iter,selected_robot,cost,gradnorm,comm_poses,comm_mb,"
           "cumulative_comm_poses,cumulative_comm_mb,rel_cost_change,"
           "loss_tol_reached,grad_tol_reached,both_tol_reached,"
           "compute_seq_ms,compute_parallel_ms,cumulative_parallel_ms,"
           "adaptive_refine_robots,adaptive_refine_calls,"
           "adaptive_refine_ms,cumulative_adaptive_refine_ms,"
           "local_amm_attempted,local_amm_accepted,local_amm_rejected,"
           "local_amm_cost_decrease,local_amm_grad_decrease,"
           "local_amm_step_norm,local_amm_ms,"
           "local_amm_baseline_compared,local_amm_selected,"
           "local_amm_baseline_selected,local_amm_vs_baseline_cost_delta,"
           "local_amm_vs_baseline_grad_delta,"
           "cumulative_local_amm_attempted,"
           "cumulative_local_amm_accepted,"
           "cumulative_local_amm_rejected,"
           "cumulative_local_amm_cost_decrease,"
           "cumulative_local_amm_grad_decrease,"
           "cumulative_local_amm_step_norm,cumulative_local_amm_ms,"
           "cumulative_local_amm_baseline_compared,"
           "cumulative_local_amm_selected,"
           "cumulative_local_amm_baseline_selected,"
           "cumulative_local_amm_vs_baseline_cost_delta,"
           "cumulative_local_amm_vs_baseline_grad_delta,"
           "majorized_boundary_candidates,majorized_boundary_selected,"
           "majorized_boundary_score_sum,"
           "majorized_boundary_selected_score_sum,"
           "boundary_jacobi_attempted,boundary_jacobi_accepted,"
           "boundary_jacobi_rejected,boundary_jacobi_blocks,"
           "boundary_jacobi_step_norm,boundary_jacobi_cost_decrease,"
           "boundary_jacobi_ms,cumulative_boundary_jacobi_accepted,"
           "cumulative_boundary_jacobi_ms,"
           "boundary_model_predictions,boundary_model_computed_step_norm,"
           "boundary_model_computed_cost_decrease,boundary_model_compute_ms,"
           "boundary_model_merit_evaluations,"
           "boundary_model_merit_accepted,boundary_model_merit_rejected,"
           "boundary_model_merit_decrease,"
           "boundary_model_merit_grad_decrease,boundary_model_merit_ms,"
           "cumulative_boundary_model_predictions,"
           "cumulative_boundary_model_compute_ms,"
           "cumulative_boundary_model_merit_evaluations,"
           "cumulative_boundary_model_merit_accepted,"
           "cumulative_boundary_model_merit_rejected,"
           "cumulative_boundary_model_merit_decrease,"
           "cumulative_boundary_model_merit_grad_decrease,"
           "cumulative_boundary_model_merit_ms,"
           "boundary_model_packet_payload_blocks,"
           "boundary_model_packet_model_pose_blocks,"
           "boundary_model_packet_grad_norm,"
           "boundary_model_packet_stiffness_sum,"
           "boundary_model_packet_freshness_sum,"
           "boundary_model_packet_preconditioned_step_norm,"
           "boundary_model_packet_schur_sensitivity_sum,"
           "boundary_model_packet_reduced_preconditioner_sum,"
           "boundary_model_packet_comm_mb,"
           "cumulative_boundary_model_packet_payload_blocks,"
           "cumulative_boundary_model_packet_model_pose_blocks,"
           "cumulative_boundary_model_packet_grad_norm,"
           "cumulative_boundary_model_packet_stiffness_sum,"
           "cumulative_boundary_model_packet_freshness_sum,"
           "cumulative_boundary_model_packet_preconditioned_step_norm,"
           "cumulative_boundary_model_packet_schur_sensitivity_sum,"
           "cumulative_boundary_model_packet_reduced_preconditioner_sum,"
           "cumulative_boundary_model_packet_comm_mb,"
           "boundary_response_attempted,boundary_response_accepted,"
           "boundary_response_rejected,boundary_response_blocks,"
           "boundary_response_step_norm,"
           "boundary_response_grad_delta_norm,"
           "boundary_response_cost_decrease,boundary_response_ms,"
           "cumulative_boundary_response_accepted,"
           "cumulative_boundary_response_blocks,"
           "cumulative_boundary_response_step_norm,"
           "cumulative_boundary_response_grad_delta_norm,"
           "cumulative_boundary_response_cost_decrease,"
           "cumulative_boundary_response_ms,"
           "boundary_batch_response_attempted,"
           "boundary_batch_response_accepted,"
           "boundary_batch_response_rejected,"
           "boundary_batch_response_blocks,"
           "boundary_batch_response_step_norm,"
           "boundary_batch_response_grad_delta_norm,"
           "boundary_batch_response_cost_decrease,"
           "boundary_batch_response_ms,"
           "cumulative_boundary_batch_response_accepted,"
           "cumulative_boundary_batch_response_blocks,"
           "cumulative_boundary_batch_response_step_norm,"
           "cumulative_boundary_batch_response_grad_delta_norm,"
           "cumulative_boundary_batch_response_cost_decrease,"
           "cumulative_boundary_batch_response_ms,"
           "boundary_batch_model_receivers,"
           "boundary_batch_model_candidate_poses,"
           "boundary_batch_model_merit_evaluations,"
           "boundary_batch_model_merit_accepted,"
           "boundary_batch_model_merit_rejected,"
           "boundary_batch_model_merit_decrease,"
           "boundary_batch_model_merit_grad_decrease,"
           "boundary_batch_model_refine_ms,"
           "boundary_batch_model_budget_candidates,"
           "boundary_batch_model_budget_selected,"
           "boundary_batch_model_budget_dropped,"
           "cumulative_boundary_batch_model_receivers,"
           "cumulative_boundary_batch_model_candidate_poses,"
           "cumulative_boundary_batch_model_merit_evaluations,"
           "cumulative_boundary_batch_model_merit_accepted,"
           "cumulative_boundary_batch_model_merit_rejected,"
           "cumulative_boundary_batch_model_merit_decrease,"
           "cumulative_boundary_batch_model_merit_grad_decrease,"
           "cumulative_boundary_batch_model_refine_ms,"
           "cumulative_boundary_batch_model_budget_candidates,"
           "cumulative_boundary_batch_model_budget_selected,"
           "cumulative_boundary_batch_model_budget_dropped,"
           "interface_state_payload_blocks,interface_state_grad_norm,"
           "interface_state_stiffness_sum,interface_state_freshness_sum,"
           "interface_state_comm_mb,interface_state_compute_ms,"
           "cumulative_interface_state_payload_blocks,"
           "cumulative_interface_state_comm_mb,"
           "cumulative_interface_state_compute_ms,"
           "interface_delta_candidate_pose_blocks,"
           "interface_delta_cache_miss_pose_blocks,"
           "interface_delta_rejected_pose_blocks,"
           "interface_delta_sparse_pose_blocks,"
           "interface_delta_tangent_pose_blocks,"
           "interface_delta_full_pose_blocks,"
           "interface_delta_delta_pose_blocks,"
           "interface_delta_delta_entries,"
           "interface_delta_comm_mb,"
           "interface_delta_reconstruction_error_sum,"
           "interface_delta_reconstruction_error_max,"
           "cumulative_interface_delta_candidate_pose_blocks,"
           "cumulative_interface_delta_cache_miss_pose_blocks,"
           "cumulative_interface_delta_rejected_pose_blocks,"
           "cumulative_interface_delta_sparse_pose_blocks,"
           "cumulative_interface_delta_tangent_pose_blocks,"
           "cumulative_interface_delta_full_pose_blocks,"
           "cumulative_interface_delta_delta_pose_blocks,"
           "cumulative_interface_delta_delta_entries,"
           "cumulative_interface_delta_comm_mb\n";
  }

  // Construct the centralized problem (used for evaluation)
  SparseMatrix QCentral = constructConnectionLaplacianSE(dataset);
  QuadraticProblem problemCentral(n, d, r);
  problemCentral.setQ(QCentral);


  /**
  ###########################################
  Partition dataset into robots
  ###########################################
  */
  unsigned int num_poses_per_robot = num_poses / num_robots;
  if (num_poses_per_robot <= 0) {
    cout << "More robots than total number of poses! Decrease the number of robots" << endl;
    exit(1);
  }

  // create mapping from global pose index to local pose index
  map<unsigned, PoseID> PoseMap;
  for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
    unsigned startIdx = robot * num_poses_per_robot;
    unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
    if (robot == (unsigned) num_robots - 1) endIdx = n;
    for (unsigned idx = startIdx; idx < endIdx; ++idx) {
      unsigned localIdx = idx - startIdx;  // this is the local ID of this pose
      PoseID pose = make_pair(robot, localIdx);
      PoseMap[idx] = pose;
    }
  }

  vector<vector<RelativeSEMeasurement>> odometry(num_robots);
  vector<vector<RelativeSEMeasurement>> private_loop_closures(num_robots);
  vector<vector<RelativeSEMeasurement>> shared_loop_closure(num_robots);
  for (auto mIn : dataset) {
    PoseID src = PoseMap[mIn.p1];
    PoseID dst = PoseMap[mIn.p2];

    unsigned srcRobot = src.first;
    unsigned srcIdx = src.second;
    unsigned dstRobot = dst.first;
    unsigned dstIdx = dst.second;

    RelativeSEMeasurement m(srcRobot, dstRobot, srcIdx, dstIdx, mIn.R, mIn.t,
                            mIn.kappa, mIn.tau);

    if (srcRobot == dstRobot) {
      // private measurement
      if (srcIdx + 1 == dstIdx) {
        // Odometry
        odometry[srcRobot].push_back(m);
      } else {
        // private loop closure
        private_loop_closures[srcRobot].push_back(m);
      }
    } else {
      // shared measurement
      shared_loop_closure[srcRobot].push_back(m);
      shared_loop_closure[dstRobot].push_back(m);
    }
  }

  /**
  ###########################################
  Initialization
  ###########################################
  */
  vector<PGOAgent *> agents;
  for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
    PGOAgentParameters options(d, r, num_robots);
    options.acceleration = acceleration;
    options.algorithm = localAlgorithm;
    options.useConsensusCopies = false;
    options.trustRegionIterations = trustRegionIterations;
    options.trustRegionMaxInnerIterations = trustRegionMaxInnerIterations;
    options.trustRegionTolerance = trustRegionTolerance;
    options.trustRegionInitialRadius = trustRegionInitialRadius;
    options.useManualLocalSolver = useManualLocalSolver;
    options.useReducedRotationLocalSolver = useReducedRotationLocalSolver;
    options.verbose = verbose;

    auto *agent = new PGOAgent(robot, options);
    if (configuredAsyncSchurMode) {
      agent->enableAdaptiveTrustRegionRadius(true);
    }

    // All agents share a special, common matrix called the 'lifting matrix' which the first agent will generate
    if (robot > 0) {
      Matrix M;
      agents[0]->getLiftingMatrix(M);
      agent->setLiftingMatrix(M);
    }

    agent->setPoseGraph(odometry[robot], private_loop_closures[robot],
                        shared_loop_closure[robot]);
    agents.push_back(agent);
  }

  /**
  ##########################################################################################
  For this demo, we initialize each robot's estimate from the centralized chordal relaxation
  ##########################################################################################
  */
  Matrix TChordal = chordalInitialization(d, n, dataset);
  Matrix XChordal = fixedStiefelVariable(d, r) * TChordal; // Lift estimate to the correct relaxation rank
  for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
    unsigned startIdx = robot * num_poses_per_robot;
    unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
    if (robot == (unsigned) num_robots - 1) endIdx = n;
    agents[robot]->setX(XChordal.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)));
  }

  std::vector<unsigned> robotColors(num_robots, 0);
  unsigned numRobotColors = 1;
  for (unsigned robot = 0; robot < static_cast<unsigned>(num_robots); ++robot) {
    std::set<unsigned> usedColors;
    for (unsigned neighborID : agents[robot]->getNeighbors()) {
      if (neighborID < robot) {
        usedColors.insert(robotColors[neighborID]);
      }
    }
    unsigned color = 0;
    while (usedColors.find(color) != usedColors.end()) {
      ++color;
    }
    robotColors[robot] = color;
    numRobotColors = std::max(numRobotColors, color + 1);
  }

  /**
  ###########################################
  Optimization loop
  ###########################################
  */
  Matrix Xopt(r, n * (d + 1));
  unsigned selectedRobot = 0;
  const size_t posePayloadBytes = r * (d + 1) * sizeof(double);
  const double posePayloadMB =
      static_cast<double>(posePayloadBytes) / (1024.0 * 1024.0);
  const size_t interfaceStatePayloadBytes =
      (2 * r * (d + 1) + 2) * sizeof(double);
  const double interfaceStatePayloadMB =
      static_cast<double>(interfaceStatePayloadBytes) / (1024.0 * 1024.0);
  const bool boundaryPacketSendsSchurScalars =
      boundaryPacketUsePreconditionedStep || boundaryPacketSchurResponseMode;
  const size_t boundaryModelPacketPayloadBytes =
      interfaceStatePayloadBytes +
      (boundaryPacketUsePreconditionedStep ? r * (d + 1) * sizeof(double)
                                           : 0) +
      (boundaryPacketSendsSchurScalars ? 3 * sizeof(double) : 0);
  const double boundaryModelPacketPayloadMB =
      static_cast<double>(boundaryModelPacketPayloadBytes) / (1024.0 * 1024.0);
  size_t cumulativeCommPoses = 0;
  double cumulativeCommMB = 0.0;
  size_t convergenceCommPoses = 0;
  double convergenceCommMB = 0.0;
  int convergenceIter = -1;
  size_t lossTolCommPoses = 0;
  double lossTolCommMB = 0.0;
  int lossTolIter = -1;
  size_t gradTolCommPoses = 0;
  double gradTolCommMB = 0.0;
  int gradTolIter = -1;
  unsigned stableLossIters = 0;
  double prevCost = std::numeric_limits<double>::quiet_NaN();
  double finalCost = std::numeric_limits<double>::quiet_NaN();
  double finalGradNorm = std::numeric_limits<double>::quiet_NaN();
  unsigned finalIter = 0;
  double cumulativeParallelComputeMs = 0.0;
  size_t cumulativeAdaptiveRefineRobots = 0;
  size_t cumulativeAdaptiveRefineCalls = 0;
  double cumulativeAdaptiveRefineMs = 0.0;
  size_t cumulativeLocalAmmAttempted = 0;
  size_t cumulativeLocalAmmAccepted = 0;
  size_t cumulativeLocalAmmRejected = 0;
  double cumulativeLocalAmmCostDecrease = 0.0;
  double cumulativeLocalAmmGradDecrease = 0.0;
  double cumulativeLocalAmmStepNorm = 0.0;
  double cumulativeLocalAmmMs = 0.0;
  size_t cumulativeLocalAmmBaselineCompared = 0;
  size_t cumulativeLocalAmmSelected = 0;
  size_t cumulativeLocalAmmBaselineSelected = 0;
  double cumulativeLocalAmmVsBaselineCostDelta = 0.0;
  double cumulativeLocalAmmVsBaselineGradDelta = 0.0;
  size_t cumulativeMajorizedBoundaryCandidates = 0;
  size_t cumulativeMajorizedBoundarySelected = 0;
  double cumulativeMajorizedBoundaryScore = 0.0;
  double cumulativeMajorizedBoundarySelectedScore = 0.0;
  size_t cumulativeBoundaryJacobiAttempted = 0;
  size_t cumulativeBoundaryJacobiAccepted = 0;
  size_t cumulativeBoundaryJacobiRejected = 0;
  size_t cumulativeBoundaryJacobiBlocks = 0;
  double cumulativeBoundaryJacobiStepNorm = 0.0;
  double cumulativeBoundaryJacobiCostDecrease = 0.0;
  double cumulativeBoundaryJacobiMs = 0.0;
  size_t cumulativeBoundaryModelPredictions = 0;
  double cumulativeBoundaryModelStepNorm = 0.0;
  double cumulativeBoundaryModelCostDecrease = 0.0;
  double cumulativeBoundaryModelMs = 0.0;
  size_t cumulativeBoundaryModelMeritEvaluations = 0;
  size_t cumulativeBoundaryModelMeritAccepted = 0;
  size_t cumulativeBoundaryModelMeritRejected = 0;
  double cumulativeBoundaryModelMeritDecrease = 0.0;
  double cumulativeBoundaryModelMeritGradDecrease = 0.0;
  double cumulativeBoundaryModelMeritMs = 0.0;
  size_t cumulativeBoundaryModelPacketPayloadBlocks = 0;
  size_t cumulativeBoundaryModelPacketModelPoseBlocks = 0;
  double cumulativeBoundaryModelPacketGradNorm = 0.0;
  double cumulativeBoundaryModelPacketStiffnessSum = 0.0;
  double cumulativeBoundaryModelPacketFreshnessSum = 0.0;
  double cumulativeBoundaryModelPacketPreconditionedStepNorm = 0.0;
  double cumulativeBoundaryModelPacketSchurSensitivitySum = 0.0;
  double cumulativeBoundaryModelPacketReducedPreconditionerSum = 0.0;
  double cumulativeBoundaryModelPacketCommMB = 0.0;
  size_t cumulativeBoundaryResponseAttempted = 0;
  size_t cumulativeBoundaryResponseAccepted = 0;
  size_t cumulativeBoundaryResponseRejected = 0;
  size_t cumulativeBoundaryResponseBlocks = 0;
  double cumulativeBoundaryResponseStepNorm = 0.0;
  double cumulativeBoundaryResponseGradDeltaNorm = 0.0;
  double cumulativeBoundaryResponseCostDecrease = 0.0;
  double cumulativeBoundaryResponseMs = 0.0;
  size_t cumulativeBoundaryBatchResponseAttempted = 0;
  size_t cumulativeBoundaryBatchResponseAccepted = 0;
  size_t cumulativeBoundaryBatchResponseRejected = 0;
  size_t cumulativeBoundaryBatchResponseBlocks = 0;
  double cumulativeBoundaryBatchResponseStepNorm = 0.0;
  double cumulativeBoundaryBatchResponseGradDeltaNorm = 0.0;
  double cumulativeBoundaryBatchResponseCostDecrease = 0.0;
  double cumulativeBoundaryBatchResponseMs = 0.0;
  size_t cumulativeBoundaryBatchModelReceivers = 0;
  size_t cumulativeBoundaryBatchModelCandidatePoses = 0;
  size_t cumulativeBoundaryBatchModelMeritEvaluations = 0;
  size_t cumulativeBoundaryBatchModelMeritAccepted = 0;
  size_t cumulativeBoundaryBatchModelMeritRejected = 0;
  double cumulativeBoundaryBatchModelMeritDecrease = 0.0;
  double cumulativeBoundaryBatchModelMeritGradDecrease = 0.0;
  double cumulativeBoundaryBatchModelRefineMs = 0.0;
  size_t cumulativeBoundaryBatchModelBudgetCandidates = 0;
  size_t cumulativeBoundaryBatchModelBudgetSelected = 0;
  size_t cumulativeBoundaryBatchModelBudgetDropped = 0;
  size_t cumulativeInterfaceStatePayloadBlocks = 0;
  double cumulativeInterfaceStateCommMB = 0.0;
  double cumulativeInterfaceStateComputeMs = 0.0;
  size_t cumulativeInterfaceDeltaCandidatePoseBlocks = 0;
  size_t cumulativeInterfaceDeltaCacheMissPoseBlocks = 0;
  size_t cumulativeInterfaceDeltaRejectedPoseBlocks = 0;
  size_t cumulativeInterfaceDeltaSparsePoseBlocks = 0;
  size_t cumulativeInterfaceDeltaTangentPoseBlocks = 0;
  size_t cumulativeInterfaceDeltaFullPoseBlocks = 0;
  size_t cumulativeInterfaceDeltaDeltaPoseBlocks = 0;
  size_t cumulativeInterfaceDeltaDeltaEntries = 0;
  double cumulativeInterfaceDeltaCommMB = 0.0;
  std::vector<std::map<unsigned, std::map<unsigned, Matrix>>> lastSentPose(
      num_robots);
  std::vector<std::map<unsigned, std::map<unsigned, Matrix>>> lastSentDelta(
      num_robots);
  std::vector<std::map<unsigned, std::map<unsigned, unsigned>>> lastSentIter(
      num_robots);
  std::vector<double> latestLocalGradNorm(num_robots, 0.0);
  double ammBoundaryS = 1.0;
  double ammBoundaryGamma = 0.0;
  std::vector<double> localAmmS(num_robots, 1.0);
  std::vector<double> localAmmGamma(num_robots, 0.0);
  auto elapsedMs = [](const std::chrono::high_resolution_clock::time_point &t0) {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0)
            .count()) /
           1000.0;
  };
  cout << "Running " << numIters << " iterations..." << endl;
  cout << "local_solver = " << localSolverName
       << " | boundary_packet_step_mode = " << boundaryPacketStepMode
       << " | boundary_packet_schur_damping_gain = "
       << boundaryPacketSchurDampingGain
       << " | boundary_packet_schur_damping_max = "
       << boundaryPacketSchurDampingMax
       << " | local_amm_compare_baseline = " << localAmmCompareBaseline
       << " | local_amm_require_beat_baseline = "
       << localAmmRequireBeatBaseline << endl;
  for (unsigned iter = 0; iter < numIters; ++iter) {
    if (ammBoundaryMode) {
      const double nextS =
          0.5 + 0.5 * std::sqrt(4.0 * ammBoundaryS * ammBoundaryS + 1.0);
      ammBoundaryGamma = (ammBoundaryS - 1.0) / nextS;
      ammBoundaryS = nextS;
    }
    updateMode = configuredUpdateMode;
    if (hybridWarmupMode) {
      updateMode =
          iter < hybridWarmupRounds
              ? "decentralized"
              : "decentralized_residual_async_schur_late_feedback";
    }
    if (adaptiveBudgetMode || majorizedBoundaryMode || boundaryModelMode ||
        boundaryModelPacketMode || boundaryResponseMode ||
        boundaryBatchModelMode || ammBoundaryMode || localAmmMode ||
        interfaceStateMode || interfaceDeltaMode) {
      updateMode = "decentralized_residual_async_schur_late_feedback";
    }
    const bool asyncSchurMode =
        updateMode == "decentralized_residual_async_schur_late_feedback";
    const unsigned effectiveAsyncIter =
        hybridWarmupMode && iter >= hybridWarmupRounds
            ? iter - hybridWarmupRounds
            : iter;
    size_t iterCommPoses = 0;
    int selectedRobotForLog = -1;
    double iterSequentialComputeMs = 0.0;
    double iterParallelComputeMs = 0.0;
    size_t adaptiveRefineRobots = 0;
    size_t adaptiveRefineCalls = 0;
    double adaptiveRefineMs = 0.0;
    size_t localAmmAttempted = 0;
    size_t localAmmAccepted = 0;
    size_t localAmmRejected = 0;
    double localAmmCostDecrease = 0.0;
    double localAmmGradDecrease = 0.0;
    double localAmmStepNorm = 0.0;
    double localAmmMs = 0.0;
    size_t localAmmBaselineCompared = 0;
    size_t localAmmSelected = 0;
    size_t localAmmBaselineSelected = 0;
    double localAmmVsBaselineCostDelta = 0.0;
    double localAmmVsBaselineGradDelta = 0.0;
    size_t majorizedBoundaryCandidates = 0;
    size_t majorizedBoundarySelected = 0;
    double majorizedBoundaryScoreSum = 0.0;
    double majorizedBoundarySelectedScoreSum = 0.0;
    size_t boundaryJacobiAttempted = 0;
    size_t boundaryJacobiAccepted = 0;
    size_t boundaryJacobiRejected = 0;
    size_t boundaryJacobiBlocks = 0;
    double boundaryJacobiStepNorm = 0.0;
    double boundaryJacobiCostDecrease = 0.0;
    double boundaryJacobiMs = 0.0;
    size_t boundaryModelPredictions = 0;
    double boundaryModelStepNorm = 0.0;
    double boundaryModelCostDecrease = 0.0;
    double boundaryModelMs = 0.0;
    size_t boundaryModelMeritEvaluations = 0;
    size_t boundaryModelMeritAccepted = 0;
    size_t boundaryModelMeritRejected = 0;
    double boundaryModelMeritDecrease = 0.0;
    double boundaryModelMeritGradDecrease = 0.0;
    double boundaryModelMeritMs = 0.0;
    size_t boundaryModelPacketPayloadBlocks = 0;
    size_t boundaryModelPacketModelPoseBlocks = 0;
    double boundaryModelPacketGradNorm = 0.0;
    double boundaryModelPacketStiffnessSum = 0.0;
    double boundaryModelPacketFreshnessSum = 0.0;
    double boundaryModelPacketPreconditionedStepNorm = 0.0;
    double boundaryModelPacketSchurSensitivitySum = 0.0;
    double boundaryModelPacketReducedPreconditionerSum = 0.0;
    size_t boundaryModelPacketExtraPayloadBytes = 0;
    size_t boundaryResponseAttempted = 0;
    size_t boundaryResponseAccepted = 0;
    size_t boundaryResponseRejected = 0;
    size_t boundaryResponseBlocks = 0;
    double boundaryResponseStepNorm = 0.0;
    double boundaryResponseGradDeltaNorm = 0.0;
    double boundaryResponseCostDecrease = 0.0;
    double boundaryResponseMs = 0.0;
    size_t boundaryBatchResponseAttempted = 0;
    size_t boundaryBatchResponseAccepted = 0;
    size_t boundaryBatchResponseRejected = 0;
    size_t boundaryBatchResponseBlocks = 0;
    double boundaryBatchResponseStepNorm = 0.0;
    double boundaryBatchResponseGradDeltaNorm = 0.0;
    double boundaryBatchResponseCostDecrease = 0.0;
    double boundaryBatchResponseMs = 0.0;
    size_t boundaryBatchModelReceivers = 0;
    size_t boundaryBatchModelCandidatePoses = 0;
    size_t boundaryBatchModelMeritEvaluations = 0;
    size_t boundaryBatchModelMeritAccepted = 0;
    size_t boundaryBatchModelMeritRejected = 0;
    double boundaryBatchModelMeritDecrease = 0.0;
    double boundaryBatchModelMeritGradDecrease = 0.0;
    double boundaryBatchModelRefineMs = 0.0;
    size_t boundaryBatchModelBudgetCandidates = 0;
    size_t boundaryBatchModelBudgetSelected = 0;
    size_t boundaryBatchModelBudgetDropped = 0;
    size_t interfaceStatePayloadBlocks = 0;
    double interfaceStateGradNorm = 0.0;
    double interfaceStateStiffnessSum = 0.0;
    double interfaceStateFreshnessSum = 0.0;
    double interfaceStateComputeMs = 0.0;
    size_t interfaceDeltaCandidatePoseBlocks = 0;
    size_t interfaceDeltaCacheMissPoseBlocks = 0;
    size_t interfaceDeltaRejectedPoseBlocks = 0;
    size_t interfaceDeltaSparsePoseBlocks = 0;
    size_t interfaceDeltaTangentPoseBlocks = 0;
    size_t interfaceDeltaFullPoseBlocks = 0;
    size_t interfaceDeltaDeltaPoseBlocks = 0;
    size_t interfaceDeltaDeltaEntries = 0;
    double interfaceDeltaReconstructionErrorSum = 0.0;
    double interfaceDeltaReconstructionErrorMax = 0.0;
    size_t iterCommPayloadBytes = 0;
    std::vector<bool> adaptiveRefinedThisIter(num_robots, false);
    std::vector<double> adaptiveResidualMax(num_robots, 0.0);
    std::vector<unsigned> adaptiveSeparatorPoses(num_robots, 0);
    const bool asyncFeedbackReady =
        asyncSchurMode &&
        (lossTolIter >= 0 ||
         effectiveAsyncIter >= std::max(5u, 3u * stableLossRequired));
    const bool adaptiveBudgetReady =
        (adaptiveBudgetMode || majorizedBoundaryMode) &&
        adaptiveBudgetExtraRTR > 0 && iter > 0;

    const bool eventMode = updateMode == "decentralized_event" ||
                           updateMode == "decentralized_event_colored" ||
                           updateMode == "decentralized_residual" ||
                           updateMode == "decentralized_residual_colored" ||
                           updateMode == "decentralized_residual_graph_colored" ||
                           updateMode == "decentralized_residual_colored_sweep" ||
                           updateMode == "decentralized_residual_colored_predictive" ||
                           updateMode == "decentralized_residual_colored_conflict" ||
                           updateMode == "decentralized_residual_colored_conflict_refine" ||
                           updateMode == "decentralized_residual_colored_conflict_sweep" ||
                           updateMode == "decentralized_residual_colored_winner_refine" ||
                           updateMode == "decentralized_residual_colored_feedback_refine" ||
                           updateMode == "decentralized_residual_colored_curvature_feedback" ||
                           updateMode == "decentralized_residual_colored_schur_feedback" ||
                           updateMode == "decentralized_residual_colored_schur_late_feedback" ||
                           boundaryJacobiMode ||
                           asyncSchurMode;
    const bool coloredEventMode =
        updateMode == "decentralized_event_colored" ||
        updateMode == "decentralized_residual_colored" ||
        updateMode == "decentralized_residual_graph_colored" ||
        updateMode == "decentralized_residual_colored_sweep" ||
        updateMode == "decentralized_residual_colored_predictive" ||
        updateMode == "decentralized_residual_colored_conflict" ||
        updateMode == "decentralized_residual_colored_conflict_refine" ||
        updateMode == "decentralized_residual_colored_conflict_sweep" ||
        updateMode == "decentralized_residual_colored_winner_refine" ||
        updateMode == "decentralized_residual_colored_feedback_refine" ||
        updateMode == "decentralized_residual_colored_curvature_feedback" ||
        updateMode == "decentralized_residual_colored_schur_feedback" ||
        updateMode == "decentralized_residual_colored_schur_late_feedback";
    const bool residualEventMode =
        updateMode == "decentralized_residual" ||
        updateMode == "decentralized_residual_colored" ||
        updateMode == "decentralized_residual_graph_colored" ||
        updateMode == "decentralized_residual_colored_sweep" ||
        updateMode == "decentralized_residual_colored_predictive" ||
        updateMode == "decentralized_residual_colored_conflict" ||
        updateMode == "decentralized_residual_colored_conflict_refine" ||
        updateMode == "decentralized_residual_colored_conflict_sweep" ||
        updateMode == "decentralized_residual_colored_winner_refine" ||
        updateMode == "decentralized_residual_colored_feedback_refine" ||
        updateMode == "decentralized_residual_colored_curvature_feedback" ||
        updateMode == "decentralized_residual_colored_schur_feedback" ||
        updateMode == "decentralized_residual_colored_schur_late_feedback" ||
        boundaryJacobiMode ||
        asyncSchurMode;
    const bool graphColoredEventMode =
        updateMode == "decentralized_residual_graph_colored";
    const bool sweepCorrectionMode =
        updateMode == "decentralized_residual_colored_sweep" ||
        updateMode == "decentralized_residual_colored_conflict_sweep";
    const bool predictiveDirectionMode =
        updateMode == "decentralized_residual_colored_predictive";
    const bool conflictAwareMode =
        updateMode == "decentralized_residual_colored_conflict" ||
        updateMode == "decentralized_residual_colored_winner_refine";
    const bool conflictRefineMode =
        updateMode == "decentralized_residual_colored_conflict_refine" ||
        updateMode == "decentralized_residual_colored_conflict_sweep" ||
        updateMode == "decentralized_residual_colored_winner_refine" ||
        updateMode == "decentralized_residual_colored_feedback_refine" ||
        updateMode == "decentralized_residual_colored_curvature_feedback" ||
        updateMode == "decentralized_residual_colored_schur_feedback" ||
        updateMode == "decentralized_residual_colored_schur_late_feedback" ||
        asyncFeedbackReady || adaptiveBudgetReady;
    const bool winnerRefineMode =
        updateMode == "decentralized_residual_colored_winner_refine";
    const bool feedbackRefineMode =
        updateMode == "decentralized_residual_colored_feedback_refine" ||
        updateMode == "decentralized_residual_colored_curvature_feedback" ||
        updateMode == "decentralized_residual_colored_schur_feedback" ||
        (updateMode == "decentralized_residual_colored_schur_late_feedback" &&
         lossTolIter >= 0) ||
        asyncFeedbackReady || adaptiveBudgetReady;
    const bool curvatureFeedbackMode =
        updateMode == "decentralized_residual_colored_curvature_feedback";
    const bool schurFeedbackMode =
        updateMode == "decentralized_residual_colored_schur_feedback" ||
        (updateMode == "decentralized_residual_colored_schur_late_feedback" &&
         lossTolIter >= 0) ||
        asyncFeedbackReady || adaptiveBudgetReady;
    const double communicationPoseTol =
        residualEventMode && lossTolIter >= 0 ? 0.2 * eventPoseTol
                                             : eventPoseTol;
    const unsigned colorPeriod = graphColoredEventMode ? numRobotColors : 2;
    const unsigned activeColor = iter % colorPeriod;
    const unsigned nextActiveColor = (iter + 1) % colorPeriod;
    auto eventColor = [&](unsigned robotID) {
      return graphColoredEventMode ? robotColors[robotID] : robotID % 2;
    };
    auto sparseDeltaPayloadBytes = [&](size_t entries) {
      return 2 * sizeof(unsigned) + entries * (sizeof(unsigned) + sizeof(double));
    };
    auto tangentDeltaPayloadBytes = [&]() {
      return 2 * sizeof(unsigned) +
             liftedTangentDeltaDoubles(r, d) * sizeof(double);
    };
    auto recordAuxPosePayloads = [&](size_t poseBlocks) {
      if (interfaceDeltaMode) {
        interfaceDeltaFullPoseBlocks += poseBlocks;
        iterCommPayloadBytes += poseBlocks * posePayloadBytes;
      }
    };
    auto updateSentPoseWithOptionalDelta =
        [&](unsigned senderID, unsigned receiverID, unsigned poseIndex,
            const Matrix &sharedPose, bool forceFull) {
          auto &sentToReceiver = lastSentPose[senderID][receiverID];
          auto &sentIterToReceiver = lastSentIter[senderID][receiverID];
          const auto previousPoseIt = sentToReceiver.find(poseIndex);
          const bool sparseCodecEnabled =
              (interfaceDeltaCodec == "sparse" ||
               interfaceDeltaCodec == "hybrid") &&
              interfaceDeltaTopK > 0;
          const bool tangentCodecEnabled =
              interfaceDeltaCodec == "tangent" ||
              interfaceDeltaCodec == "hybrid";
          const bool deltaCodecEnabled =
              sparseCodecEnabled || tangentCodecEnabled;
          if (interfaceDeltaMode) {
            if (previousPoseIt == sentToReceiver.end()) {
              ++interfaceDeltaCacheMissPoseBlocks;
            } else if (!forceFull && deltaCodecEnabled) {
              ++interfaceDeltaCandidatePoseBlocks;
            }
          }
          const bool canTryDelta =
              interfaceDeltaMode && !forceFull && deltaCodecEnabled &&
              previousPoseIt != sentToReceiver.end();
          Matrix poseForReceiver = sharedPose;
          bool usedDelta = false;
          bool usedTangentDelta = false;
          size_t sentEntries = 0;
          double reconstructionError = 0.0;
          if (canTryDelta) {
            size_t selectedPayloadBytes = posePayloadBytes;
            auto considerDeltaCandidate =
                [&](const Matrix &reconstructed, size_t entries,
                    size_t payloadBytes, bool tangentCodec) {
                  const double candidateError =
                      (sharedPose - reconstructed).norm();
                  if (!std::isfinite(candidateError) ||
                      candidateError > interfaceDeltaMaxReconError ||
                      payloadBytes >= posePayloadBytes) {
                    return;
                  }
                  if (!usedDelta || payloadBytes < selectedPayloadBytes) {
                    poseForReceiver = reconstructed;
                    usedDelta = true;
                    usedTangentDelta = tangentCodec;
                    sentEntries = entries;
                    reconstructionError = candidateError;
                    selectedPayloadBytes = payloadBytes;
                  }
                };

            if (sparseCodecEnabled) {
              const Matrix delta = sharedPose - previousPoseIt->second;
              struct DeltaEntry {
                double magnitude;
                unsigned row;
                unsigned col;
                double value;
              };
              std::vector<DeltaEntry> entries;
              entries.reserve(static_cast<size_t>(delta.rows() * delta.cols()));
              for (unsigned row = 0; row < static_cast<unsigned>(delta.rows());
                   ++row) {
                for (unsigned col = 0;
                     col < static_cast<unsigned>(delta.cols()); ++col) {
                  entries.push_back(
                      {std::abs(delta(row, col)), row, col, delta(row, col)});
                }
              }
              std::sort(entries.begin(), entries.end(),
                        [](const DeltaEntry &a, const DeltaEntry &b) {
                          return a.magnitude > b.magnitude;
                        });
              const size_t sparseEntries =
                  std::min<size_t>(interfaceDeltaTopK, entries.size());
              Matrix sparseDelta = Matrix::Zero(delta.rows(), delta.cols());
              for (size_t index = 0; index < sparseEntries; ++index) {
                sparseDelta(entries[index].row, entries[index].col) =
                    entries[index].value;
              }
              LiftedSEManifold poseManifold(r, d, 1);
              considerDeltaCandidate(
                  poseManifold.project(previousPoseIt->second + sparseDelta),
                  sparseEntries, sparseDeltaPayloadBytes(sparseEntries),
                  false);
            }

            if (tangentCodecEnabled) {
              const Matrix reconstructed = reconstructLiftedTangentDelta(
                  previousPoseIt->second, sharedPose, d);
              considerDeltaCandidate(reconstructed,
                                     liftedTangentDeltaDoubles(r, d),
                                     tangentDeltaPayloadBytes(), true);
            }
          }

          if (interfaceDeltaMode) {
            if (usedDelta) {
              ++interfaceDeltaDeltaPoseBlocks;
              interfaceDeltaDeltaEntries += sentEntries;
              if (usedTangentDelta) {
                ++interfaceDeltaTangentPoseBlocks;
                iterCommPayloadBytes += tangentDeltaPayloadBytes();
              } else {
                ++interfaceDeltaSparsePoseBlocks;
                iterCommPayloadBytes += sparseDeltaPayloadBytes(sentEntries);
              }
              interfaceDeltaReconstructionErrorSum += reconstructionError;
              interfaceDeltaReconstructionErrorMax =
                  std::max(interfaceDeltaReconstructionErrorMax,
                           reconstructionError);
            } else {
              if (canTryDelta) {
                ++interfaceDeltaRejectedPoseBlocks;
              }
              ++interfaceDeltaFullPoseBlocks;
              iterCommPayloadBytes += posePayloadBytes;
            }
          }

          if (previousPoseIt != sentToReceiver.end()) {
            lastSentDelta[senderID][receiverID][poseIndex] =
                sharedPose - previousPoseIt->second;
          }
          sentToReceiver[poseIndex] = poseForReceiver;
          sentIterToReceiver[poseIndex] = iter;
          return poseForReceiver;
        };
    if (updateMode == "decentralized" || eventMode) {
      const bool refreshThisRound =
          updateMode == "decentralized" &&
          (iter == 0 || (iter % decentralizedRefreshPeriod == 0));
      const bool eventInitialRound =
          eventMode && iter == 0;
      const bool consistencySweepRound =
          residualEventMode && iter > 0 && (iter % eventMaxAge == 0);
      std::vector<bool> shouldOptimize(num_robots,
                                       updateMode == "decentralized" &&
                                           refreshThisRound);
      std::vector<bool> needsConsistencyRefresh(num_robots, false);
      std::vector<bool> activeReceiver(num_robots, true);
      std::vector<bool> needsBoundaryJacobiCorrection(num_robots, false);
      auto computeBoundaryModelPredictions =
          [&](std::vector<PoseDict> &predictionsByRobot) {
            predictionsByRobot.assign(num_robots, PoseDict{});
            if (!boundaryModelMode) {
              return;
            }
            double predictionStageMaxMs = 0.0;
            for (auto *robotPtr : agents) {
              PoseDict predictedPoses;
              double costBeforePrediction = 0.0;
              double costAfterPrediction = 0.0;
              double predictionStepNorm = 0.0;
              unsigned predictedBlocks = 0;
              const auto t0 = std::chrono::high_resolution_clock::now();
              const bool ok = robotPtr->computeBoundaryJacobiPredictions(
                  boundaryModelStep, boundaryModelMaxBlockNorm,
                  boundaryModelRequireDecrease,
                  boundaryModelBacktrackingSteps, predictedPoses,
                  costBeforePrediction, costAfterPrediction,
                  predictionStepNorm, predictedBlocks);
              const double dtMs = elapsedMs(t0);
              iterSequentialComputeMs += dtMs;
              predictionStageMaxMs = std::max(predictionStageMaxMs, dtMs);
              boundaryModelMs += dtMs;
              if (!ok || predictedPoses.empty()) {
                continue;
              }
              predictionsByRobot[robotPtr->getID()] = predictedPoses;
              boundaryModelStepNorm += predictionStepNorm;
              boundaryModelCostDecrease +=
                  std::max(0.0,
                           costBeforePrediction - costAfterPrediction);
            }
            iterParallelComputeMs += predictionStageMaxMs;
          };
      auto computeBoundaryModelPackets =
          [&](std::vector<std::map<unsigned, BoundaryInterfaceState>>
                  &packetsByRobot) {
            packetsByRobot.assign(num_robots,
                                  std::map<unsigned, BoundaryInterfaceState>{});
            if (!boundaryModelPacketMode && !boundaryResponseMode &&
                !boundaryPacketSchurResponseMode) {
              return;
            }
            double stageMaxMs = 0.0;
            for (auto *senderPtr : agents) {
              std::vector<BoundaryInterfaceState> states;
              double localGradNorm = 0.0;
              double localStiffnessSum = 0.0;
              const auto t0 = std::chrono::high_resolution_clock::now();
              const bool ok = senderPtr->computeBoundaryInterfaceState(
                  states, localGradNorm, localStiffnessSum);
              const double dtMs = elapsedMs(t0);
              iterSequentialComputeMs += dtMs;
              stageMaxMs = std::max(stageMaxMs, dtMs);
              if (!ok || states.empty()) {
                continue;
              }
              auto &senderPackets = packetsByRobot[senderPtr->getID()];
              for (const BoundaryInterfaceState &state : states) {
                if (state.poseID.first == senderPtr->getID()) {
                  senderPackets[state.poseID.second] = state;
                }
              }
            }
            iterParallelComputeMs += stageMaxMs;
          };
      auto computeInterfaceStateDiagnostics = [&]() {
        if (!interfaceStateMode) {
          return;
        }
        double stageMaxMs = 0.0;
        for (auto *senderPtr : agents) {
          std::vector<BoundaryInterfaceState> states;
          double localGradNorm = 0.0;
          double localStiffnessSum = 0.0;
          const auto t0 = std::chrono::high_resolution_clock::now();
          const bool ok = senderPtr->computeBoundaryInterfaceState(
              states, localGradNorm, localStiffnessSum);
          const double dtMs = elapsedMs(t0);
          iterSequentialComputeMs += dtMs;
          stageMaxMs = std::max(stageMaxMs, dtMs);
          interfaceStateComputeMs += dtMs;
          if (!ok || states.empty()) {
            continue;
          }
          const unsigned senderID = senderPtr->getID();
          for (unsigned receiverID : senderPtr->getNeighbors()) {
            std::set<unsigned> neededPoses;
            for (unsigned poseIndex :
                 agents[receiverID]->getNeighborPublicPoses(senderID)) {
              neededPoses.insert(poseIndex);
            }
            if (neededPoses.empty()) {
              continue;
            }
            for (const BoundaryInterfaceState &state : states) {
              const unsigned poseIndex = state.poseID.second;
              if (neededPoses.find(poseIndex) == neededPoses.end()) {
                continue;
              }
              const auto sentIterIt =
                  lastSentIter[senderID][receiverID].find(poseIndex);
              const unsigned age =
                  sentIterIt == lastSentIter[senderID][receiverID].end()
                      ? eventMaxAge
                      : iter - sentIterIt->second;
              const double freshness =
                  1.0 / (1.0 + static_cast<double>(age));
              ++interfaceStatePayloadBlocks;
              interfaceStateGradNorm += state.gradient.norm();
              interfaceStateStiffnessSum += state.stiffness;
              interfaceStateFreshnessSum += freshness;
            }
          }
        }
        iterParallelComputeMs += stageMaxMs;
      };
      if (residualEventMode && eventLocalGradTol > 0.0) {
        for (auto *robotPtr : agents) {
          const unsigned receiverID = robotPtr->getID();
          if (coloredEventMode && eventColor(receiverID) != activeColor) {
            continue;
          }
          double localCost = 0.0;
          double localGradNorm = 0.0;
          if (robotPtr->evaluateLocalModel(localCost, localGradNorm)) {
            latestLocalGradNorm[receiverID] = localGradNorm;
            if (localGradNorm >= eventLocalGradTol) {
              needsConsistencyRefresh[receiverID] = true;
              shouldOptimize[receiverID] = true;
            }
          }
        }
      }
      if (conflictAwareMode && !consistencySweepRound) {
        std::fill(activeReceiver.begin(), activeReceiver.end(), false);
        for (auto *robotPtr : agents) {
          const unsigned robotID = robotPtr->getID();
          if (eventColor(robotID) != activeColor) {
            continue;
          }

          bool winsConflict = true;
          for (unsigned neighborID : robotPtr->getNeighbors()) {
            if (eventColor(neighborID) != activeColor) {
              continue;
            }
            const double neighborGrad = latestLocalGradNorm[neighborID];
            const double robotGrad = latestLocalGradNorm[robotID];
            if (neighborGrad > robotGrad ||
                (neighborGrad == robotGrad && neighborID < robotID)) {
              winsConflict = false;
              break;
            }
          }
          activeReceiver[robotID] = winsConflict;
          if (!winsConflict) {
            shouldOptimize[robotID] = false;
            needsConsistencyRefresh[robotID] = false;
          }
        }
      }

      // Decentralized refresh: each receiver asks only for the neighbor
      // separator poses needed by its own factors. The event mode additionally
      // lets each sender suppress unchanged poses until a local TTL expires.
      if (refreshThisRound || eventMode) {
        for (auto *agentPtr : agents) {
          const unsigned receiverID = agentPtr->getID();
          if (conflictAwareMode && !consistencySweepRound &&
              !activeReceiver[receiverID]) {
            continue;
          }
          if (coloredEventMode && !consistencySweepRound &&
              eventColor(receiverID) != activeColor) {
            continue;
          }
          for (unsigned neighborID : agentPtr->getNeighbors()) {
            PGOAgent *robotPtr = agents[neighborID];
            PoseDict sharedPoses;
            struct CandidatePose {
              unsigned poseIndex;
              Matrix pose;
              double score;
              bool neverSent;
            };
            std::vector<CandidatePose> candidatePoses;
            bool hasNeverSentCandidate = false;
            const std::vector<unsigned> neededPoses =
                agentPtr->getNeighborPublicPoses(neighborID);
            std::map<unsigned, double> residualScores;
            if (eventMode) {
              agentPtr->getNeighborResidualScores(neighborID, residualScores);
            }
            if (adaptiveBudgetMode) {
              for (const auto &residualScore : residualScores) {
                adaptiveResidualMax[receiverID] =
                    std::max(adaptiveResidualMax[receiverID],
                             residualScore.second);
              }
            }
            if (residualEventMode) {
              for (const auto &residualScore : residualScores) {
                if (residualScore.second >= eventResidualTol) {
                  shouldOptimize[receiverID] = true;
                  break;
                }
              }
            }
            for (unsigned poseIndex : neededPoses) {
              Matrix sharedPose;
              if (robotPtr->getSharedPose(poseIndex, sharedPose)) {
                bool shouldSend = refreshThisRound || eventInitialRound;
                if (consistencySweepRound) {
                  shouldSend = true;
                }
                double score = std::numeric_limits<double>::infinity();
                if (eventMode) {
                  auto &sentToReceiver = lastSentPose[neighborID][receiverID];
                  auto &sentIterToReceiver =
                      lastSentIter[neighborID][receiverID];
                  const auto poseIt = sentToReceiver.find(poseIndex);
                  const auto iterIt = sentIterToReceiver.find(poseIndex);
                  const bool neverSent = poseIt == sentToReceiver.end() ||
                                         iterIt == sentIterToReceiver.end();
                  const unsigned age =
                      neverSent ? eventMaxAge : iter - iterIt->second;
                  const double poseDelta =
                      neverSent ? std::numeric_limits<double>::infinity()
                                : (sharedPose - poseIt->second).norm();
                  double residualScore = 0.0;
                  const auto residualIt = residualScores.find(poseIndex);
                  if (residualIt != residualScores.end()) {
                    residualScore = residualIt->second;
                  }
                  unsigned effectiveMaxAge = eventMaxAge;
                  if (residualEventMode) {
                    if (residualScore >= eventResidualTol) {
                      effectiveMaxAge = std::max(1u, eventMaxAge / 2);
                    } else {
                      effectiveMaxAge = std::max(1u, eventMaxAge * 2);
                    }
                    if (needsConsistencyRefresh[receiverID]) {
                      effectiveMaxAge =
                          std::min(effectiveMaxAge,
                                   std::max(1u, eventMaxAge / 2));
                    }
                  }
                  shouldSend = neverSent || age >= effectiveMaxAge ||
                               poseDelta >= communicationPoseTol;
                  if (shouldSend && neverSent) {
                    hasNeverSentCandidate = true;
                  }
                  score = neverSent
                              ? std::numeric_limits<double>::infinity()
                              : poseDelta +
                                    communicationPoseTol *
                                        static_cast<double>(age) /
                                        static_cast<double>(effectiveMaxAge);
                  if (residualScore > 0.0) {
                    const double freshnessScore =
                        std::isfinite(score) ? score : communicationPoseTol;
                    score = residualScore + freshnessScore;
                  }
                }
                if (shouldSend) {
                  candidatePoses.push_back(
                      {poseIndex, sharedPose, score,
                       std::isinf(score)});
                }
              }
            }
            if (eventMode &&
                (eventBudgetFraction < 1.0 || eventMaxPosesPerNeighbor > 0) &&
                !eventInitialRound &&
                !consistencySweepRound &&
                !hasNeverSentCandidate &&
                !(residualEventMode && needsConsistencyRefresh[receiverID]) &&
                candidatePoses.size() > 1) {
              size_t cap = std::max<size_t>(
                  1, static_cast<size_t>(
                         std::ceil(eventBudgetFraction * neededPoses.size())));
              if (eventMaxPosesPerNeighbor > 0) {
                cap = std::min<size_t>(cap, eventMaxPosesPerNeighbor);
              }
              if (candidatePoses.size() > cap) {
                std::sort(candidatePoses.begin(), candidatePoses.end(),
                          [](const CandidatePose &a, const CandidatePose &b) {
                            return a.score > b.score;
                          });
                candidatePoses.resize(cap);
              }
            }
            if (adaptiveBudgetMode) {
              adaptiveSeparatorPoses[receiverID] +=
                  static_cast<unsigned>(candidatePoses.size());
            }
            for (const CandidatePose &candidate : candidatePoses) {
              Matrix poseForLocalModel = candidate.pose;
              if (eventMode) {
                if (predictiveDirectionMode && neighborDirectionGain > 0.0 &&
                    coloredEventMode && eventColor(neighborID) == activeColor) {
                  auto &sentToReceiver = lastSentPose[neighborID][receiverID];
                  const auto previousPoseIt =
                      sentToReceiver.find(candidate.poseIndex);
                  if (previousPoseIt != sentToReceiver.end()) {
                    LiftedSEManifold poseManifold(r, d, 1);
                    poseForLocalModel =
                        candidate.pose +
                        neighborDirectionGain *
                            (candidate.pose - previousPoseIt->second);
                    poseForLocalModel = poseManifold.project(poseForLocalModel);
                    sharedPoses[std::make_pair(neighborID,
                                                candidate.poseIndex)] =
                        poseForLocalModel;
                  }
                }
                if (curvatureFeedbackMode && lossTolIter >= 0) {
                  auto &sentToReceiver = lastSentPose[neighborID][receiverID];
                  auto &deltaToReceiver =
                      lastSentDelta[neighborID][receiverID];
                  const auto previousPoseIt =
                      sentToReceiver.find(candidate.poseIndex);
                  const auto previousDeltaIt =
                      deltaToReceiver.find(candidate.poseIndex);
                  if (previousPoseIt != sentToReceiver.end() &&
                      previousDeltaIt != deltaToReceiver.end()) {
                    const Matrix currentDelta =
                        candidate.pose - previousPoseIt->second;
                    const Matrix previousDelta = previousDeltaIt->second;
                    const double currentNorm = currentDelta.norm();
                    const double previousNorm = previousDelta.norm();
                    if (currentNorm > 1e-12 && previousNorm > 1e-12) {
                      const double directionCos =
                          (currentDelta.array() * previousDelta.array()).sum() /
                          (currentNorm * previousNorm);
                      const Matrix deltaCurvature =
                          currentDelta - previousDelta;
                      const double curvatureNormSq =
                          deltaCurvature.squaredNorm();
                      if (directionCos > 0.25 && curvatureNormSq > 1e-18) {
                        double alpha =
                            -(currentDelta.array() *
                              deltaCurvature.array()).sum() /
                            curvatureNormSq;
                        alpha = std::max(0.0, std::min(2.0, alpha));
                        const double predictionNorm =
                            alpha * currentNorm;
                        const double maxPredictionNorm =
                            std::max(2.0 * communicationPoseTol,
                                     1.5 * currentNorm);
                        if (alpha > 0.0 &&
                            predictionNorm <= maxPredictionNorm) {
                          LiftedSEManifold poseManifold(r, d, 1);
                          poseForLocalModel =
                              poseManifold.project(candidate.pose +
                                                   alpha * currentDelta);
                        }
                      }
                    }
                  }
                }
                const bool forceFullDelta =
                    candidate.neverSent || eventInitialRound ||
                    consistencySweepRound;
                const Matrix transmittedPose =
                    updateSentPoseWithOptionalDelta(
                        neighborID, receiverID, candidate.poseIndex,
                        candidate.pose, forceFullDelta);
                if (interfaceDeltaMode) {
                  poseForLocalModel = transmittedPose;
                }
              }
              sharedPoses[std::make_pair(neighborID, candidate.poseIndex)] =
                  poseForLocalModel;
            }
            iterCommPoses += sharedPoses.size();
            if (!sharedPoses.empty()) {
              shouldOptimize[receiverID] = true;
              if (boundaryJacobiMode) {
                needsBoundaryJacobiCorrection[receiverID] = true;
              }
              agentPtr->setNeighborStatus(robotPtr->getStatus());
              agentPtr->updateNeighborPoses(robotPtr->getID(), sharedPoses);
            }
            if (acceleration && !candidatePoses.empty()) {
              PoseDict auxSharedPoses;
              for (const CandidatePose &candidate : candidatePoses) {
                Matrix auxSharedPose;
                if (robotPtr->getAuxSharedPose(candidate.poseIndex,
                                               auxSharedPose)) {
                  auxSharedPoses[std::make_pair(neighborID,
                                                candidate.poseIndex)] =
                      auxSharedPose;
                }
              }
              iterCommPoses += auxSharedPoses.size();
              recordAuxPosePayloads(auxSharedPoses.size());
              if (!auxSharedPoses.empty()) {
                agentPtr->setNeighborStatus(robotPtr->getStatus());
                agentPtr->updateAuxNeighborPoses(robotPtr->getID(),
                                                 auxSharedPoses);
              }
            }
          }
        }
      }
      if (eventMode && !residualEventMode && eventLocalGradTol > 0.0) {
        for (auto *robotPtr : agents) {
          if (coloredEventMode && !consistencySweepRound &&
              eventColor(robotPtr->getID()) != activeColor) {
            continue;
          }
          double localCost = 0.0;
          double localGradNorm = 0.0;
          if (robotPtr->evaluateLocalModel(localCost, localGradNorm) &&
              localGradNorm >= eventLocalGradTol) {
            shouldOptimize[robotPtr->getID()] = true;
          }
        }
      }

      // All robots make an independent local trust-region step only after a
      // local refresh event. On stale-cache rounds, they only advance the
      // iteration counter; this avoids spending RTR work on an unchanged model.
      double mainStageMaxMs = 0.0;
      for (auto *robotPtr : agents) {
        const unsigned robotID = robotPtr->getID();
        assert(robotPtr->instance_number() == 0);
        assert(robotPtr->iteration_number() == iter);
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (localAmmMode) {
          if (shouldOptimize[robotID]) {
            const double nextS =
                0.5 + 0.5 * std::sqrt(4.0 * localAmmS[robotID] *
                                           localAmmS[robotID] +
                                       1.0);
            localAmmGamma[robotID] =
                (localAmmS[robotID] - 1.0) / nextS;
            localAmmS[robotID] = nextS;
          } else {
            localAmmGamma[robotID] = 0.0;
          }
          LocalAmmIterationStats ammStats;
          robotPtr->iterateGuardedLocalAmm(
              shouldOptimize[robotID], localAmmGamma[robotID],
              localAmmRequireDecrease, localAmmRequireGradNonIncrease,
              localAmmCompareBaseline, localAmmRequireBeatBaseline,
              trustRegionIterations, ammStats);
          if (ammStats.attempted) {
            ++localAmmAttempted;
            if (ammStats.baselineCompared) {
              ++localAmmBaselineCompared;
              localAmmVsBaselineCostDelta += ammStats.vsBaselineCostDelta;
              localAmmVsBaselineGradDelta += ammStats.vsBaselineGradDelta;
              if (ammStats.ammSelected) {
                ++localAmmSelected;
              } else if (ammStats.baselineSelected) {
                ++localAmmBaselineSelected;
              }
            }
            if (ammStats.accepted) {
              ++localAmmAccepted;
              localAmmCostDecrease +=
                  std::max(0.0, ammStats.costBefore - ammStats.costAfter);
              localAmmGradDecrease +=
                  std::max(0.0, ammStats.gradBefore - ammStats.gradAfter);
              localAmmStepNorm += ammStats.stepNorm;
            } else {
              ++localAmmRejected;
              localAmmS[robotID] = 1.0;
              localAmmGamma[robotID] = 0.0;
            }
          }
        } else {
          robotPtr->iterate(shouldOptimize[robotID]);
        }
        const double dtMs = elapsedMs(t0);
        iterSequentialComputeMs += dtMs;
        mainStageMaxMs = std::max(mainStageMaxMs, dtMs);
        if (localAmmMode) {
          localAmmMs += dtMs;
        }
      }
      iterParallelComputeMs += mainStageMaxMs;
      computeInterfaceStateDiagnostics();

      if (residualEventMode && iter > 0) {
        const double postStepPoseTol = 0.5 * communicationPoseTol;
        std::vector<bool> needsConflictRefine(num_robots, false);
        std::vector<bool> adaptiveSeparatorSensitive(num_robots, false);
        for (auto *senderPtr : agents) {
          const unsigned senderID = senderPtr->getID();
          if (!shouldOptimize[senderID]) {
            continue;
          }

          for (unsigned receiverID : senderPtr->getNeighbors()) {
            const bool sameColorConflict =
                conflictRefineMode && eventColor(receiverID) == activeColor &&
                eventColor(senderID) == activeColor;
            const bool asyncConflict =
                asyncSchurMode && conflictRefineMode &&
                (asyncFeedbackReady || adaptiveBudgetReady);
            if (coloredEventMode && !consistencySweepRound &&
                eventColor(receiverID) != nextActiveColor &&
                !sameColorConflict) {
              continue;
            }
            PGOAgent *receiverPtr = agents[receiverID];
            PoseDict postStepPoses;
            double maxPostStepPoseDelta = 0.0;
            const std::vector<unsigned> neededPoses =
                receiverPtr->getNeighborPublicPoses(senderID);

            for (unsigned poseIndex : neededPoses) {
              Matrix sharedPose;
              if (!senderPtr->getSharedPose(poseIndex, sharedPose)) {
                continue;
              }

              auto &sentToReceiver = lastSentPose[senderID][receiverID];
              const auto poseIt = sentToReceiver.find(poseIndex);
              const bool neverSent = poseIt == sentToReceiver.end();
              const double poseDelta =
                  neverSent ? std::numeric_limits<double>::infinity()
                            : (sharedPose - poseIt->second).norm();
              const bool highConsistencyNeed =
                  needsConsistencyRefresh[senderID] || consistencySweepRound;
              const double effectivePostTol =
                  highConsistencyNeed ? 0.25 * communicationPoseTol
                                      : postStepPoseTol;

              if (neverSent || poseDelta >= effectivePostTol) {
                const Matrix transmittedPose =
                    updateSentPoseWithOptionalDelta(
                        senderID, receiverID, poseIndex, sharedPose,
                        neverSent || highConsistencyNeed);
                postStepPoses[std::make_pair(senderID, poseIndex)] =
                    transmittedPose;
                maxPostStepPoseDelta =
                    std::max(maxPostStepPoseDelta, poseDelta);
              }
            }

            if (!postStepPoses.empty()) {
              iterCommPoses += postStepPoses.size();
              receiverPtr->setNeighborStatus(senderPtr->getStatus());
              receiverPtr->updateNeighborPoses(senderID, postStepPoses);
              if (boundaryJacobiMode) {
                needsBoundaryJacobiCorrection[receiverID] = true;
              }
              const bool highYieldAsyncConflict =
                  asyncConflict &&
                  (maxPostStepPoseDelta >= communicationPoseTol ||
                   postStepPoses.size() >=
                       std::max<size_t>(1, neededPoses.size() / 4));
              if (sameColorConflict || highYieldAsyncConflict) {
                needsConflictRefine[receiverID] = true;
                adaptiveSeparatorSensitive[receiverID] = true;
                adaptiveSeparatorPoses[receiverID] +=
                    static_cast<unsigned>(postStepPoses.size());
              }
              if (acceleration) {
                PoseDict auxPostStepPoses;
                for (const auto &postStepPose : postStepPoses) {
                  Matrix auxSharedPose;
                  if (senderPtr->getAuxSharedPose(postStepPose.first.second,
                                                  auxSharedPose)) {
                    auxPostStepPoses[postStepPose.first] = auxSharedPose;
                  }
                }
                iterCommPoses += auxPostStepPoses.size();
                recordAuxPosePayloads(auxPostStepPoses.size());
                if (!auxPostStepPoses.empty()) {
                  receiverPtr->setNeighborStatus(senderPtr->getStatus());
                  receiverPtr->updateAuxNeighborPoses(senderID,
                                                      auxPostStepPoses);
                }
              }
            }
          }
        }
        if (conflictRefineMode) {
          std::vector<bool> refinedRobots(num_robots, false);
          const unsigned refineTrustRegionIterations =
              schurFeedbackMode ? 1u : 0u;
          std::vector<unsigned> adaptiveTrustRegionIterations(
              num_robots, refineTrustRegionIterations);
          if (adaptiveBudgetReady && schurFeedbackMode) {
            struct AdaptiveBudgetCandidate {
              unsigned robotID;
              double score;
              double majorizedScore;
            };
            std::vector<AdaptiveBudgetCandidate> adaptiveCandidates;
            for (auto *robotPtr : agents) {
              const unsigned robotID = robotPtr->getID();
              if (!needsConflictRefine[robotID] ||
                  !(shouldOptimize[robotID] || winnerRefineMode ||
                    asyncSchurMode)) {
                continue;
              }
              if (adaptiveSeparatorPoses[robotID] <
                  adaptiveBudgetMinSeparatorPoses) {
                continue;
              }
              const ROPTResult opt = robotPtr->getLastOptimizationResult();
              const double gradRatio =
                  opt.gradNormInit > 1e-12
                      ? opt.gradNormOpt / opt.gradNormInit
                      : 0.0;
              const bool optimizerStruggled =
                  !opt.success || opt.rtrRejectedSteps > 0 ||
                  (opt.gradNormInit > eventLocalGradTol && gradRatio > 0.9);
              const bool highResidual =
                  adaptiveResidualMax[robotID] >= adaptiveBudgetResidualTol;
              if (!highResidual && !adaptiveSeparatorSensitive[robotID] &&
                  !optimizerStruggled) {
                continue;
              }
              const double residualScore =
                  adaptiveResidualMax[robotID] *
                  std::sqrt(static_cast<double>(
                      std::max(1u, adaptiveSeparatorPoses[robotID])));
              const double sensitivityScore =
                  adaptiveSeparatorSensitive[robotID] ? 1.0 : 0.0;
              const double struggleScore = optimizerStruggled ? 0.5 : 0.0;
              double objectiveDecrease = 0.0;
              if (std::isfinite(opt.fInit) && std::isfinite(opt.fOpt)) {
                objectiveDecrease = std::max(0.0, opt.fInit - opt.fOpt);
              }
              double normalizedGradDecrease = 0.0;
              if (std::isfinite(opt.gradNormInit) &&
                  std::isfinite(opt.gradNormOpt)) {
                normalizedGradDecrease =
                    std::max(0.0, opt.gradNormInit - opt.gradNormOpt) /
                    (1.0 + std::max(0.0, opt.gradNormInit));
              }
              const double separatorScale = std::log1p(static_cast<double>(
                  std::max(1u, adaptiveSeparatorPoses[robotID])));
              const double majorizedScore =
                  residualScore +
                  separatorScale *
                      (objectiveDecrease + normalizedGradDecrease) +
                  sensitivityScore + struggleScore;
              const double score =
                  majorizedBoundaryMode
                      ? majorizedScore
                      : residualScore + sensitivityScore + struggleScore;
              adaptiveCandidates.push_back(
                  {robotID, score, majorizedScore});
            }
            if (majorizedBoundaryMode) {
              majorizedBoundaryCandidates = adaptiveCandidates.size();
              for (const AdaptiveBudgetCandidate &candidate :
                   adaptiveCandidates) {
                majorizedBoundaryScoreSum += candidate.majorizedScore;
              }
            }
            std::sort(adaptiveCandidates.begin(), adaptiveCandidates.end(),
                      [](const AdaptiveBudgetCandidate &a,
                         const AdaptiveBudgetCandidate &b) {
                        if (a.score == b.score) {
                          return a.robotID < b.robotID;
                        }
                        return a.score > b.score;
                      });
            if (adaptiveBudgetMaxRobots > 0 &&
                adaptiveCandidates.size() > adaptiveBudgetMaxRobots) {
              adaptiveCandidates.resize(adaptiveBudgetMaxRobots);
            }
            for (const AdaptiveBudgetCandidate &candidate :
                 adaptiveCandidates) {
              adaptiveTrustRegionIterations[candidate.robotID] =
                  refineTrustRegionIterations + adaptiveBudgetExtraRTR;
              if (majorizedBoundaryMode) {
                ++majorizedBoundarySelected;
                majorizedBoundarySelectedScoreSum += candidate.majorizedScore;
              }
            }
          }
          double conflictStageMaxMs = 0.0;
          for (auto *robotPtr : agents) {
            const unsigned robotID = robotPtr->getID();
            if (needsConflictRefine[robotID] &&
                (shouldOptimize[robotID] || winnerRefineMode ||
                 asyncSchurMode)) {
              const unsigned localTrustRegionIterations =
                  adaptiveTrustRegionIterations[robotID];
              if (adaptiveBudgetMode &&
                  localTrustRegionIterations <= refineTrustRegionIterations) {
                continue;
              }
              const auto t0 = std::chrono::high_resolution_clock::now();
              robotPtr->refineLocalOptimization(
                  1, false, localTrustRegionIterations);
              const double dtMs = elapsedMs(t0);
              iterSequentialComputeMs += dtMs;
              conflictStageMaxMs = std::max(conflictStageMaxMs, dtMs);
              refinedRobots[robotID] = true;
              if (localTrustRegionIterations > refineTrustRegionIterations) {
                adaptiveRefinedThisIter[robotID] = true;
                adaptiveRefineCalls++;
                adaptiveRefineMs += dtMs;
              }
            }
          }
          iterParallelComputeMs += conflictStageMaxMs;
          if (feedbackRefineMode) {
            std::vector<bool> feedbackRefineRobots(num_robots, false);
            std::vector<bool> schurFeedbackApplied(num_robots, false);
            const double feedbackPoseTol =
                asyncSchurMode ? communicationPoseTol
                               : 0.5 * communicationPoseTol;
            std::vector<double> feedbackRefineMs(num_robots, 0.0);
            std::vector<PoseDict> batchBaselineModelPosesByReceiver(
                num_robots);
            std::vector<PoseDict> batchModelPosesByReceiver(num_robots);
            std::vector<size_t> batchCandidatePosesByReceiver(num_robots, 0);
            std::vector<PoseDict> feedbackBoundaryModelPredictionsByRobot(
                num_robots);
            if (boundaryModelMode) {
              computeBoundaryModelPredictions(
                  feedbackBoundaryModelPredictionsByRobot);
            }
            std::vector<std::map<unsigned, BoundaryInterfaceState>>
                feedbackBoundaryModelPacketsByRobot(num_robots);
            if (boundaryModelPacketMode || boundaryResponseMode ||
                boundaryPacketSchurResponseMode) {
              computeBoundaryModelPackets(feedbackBoundaryModelPacketsByRobot);
            }
            std::vector<std::map<unsigned, std::set<unsigned>>>
                budgetedFeedbackPoseWhitelistBySender(num_robots);
            if (budgetedBoundaryBatchModelMode) {
              struct BudgetCandidatePose {
                unsigned senderID;
                unsigned receiverID;
                unsigned poseIndex;
                double score;
                bool neverSent;
              };
              for (auto *receiverPtr : agents) {
                const unsigned receiverID = receiverPtr->getID();
                std::vector<BudgetCandidatePose> budgetCandidates;
                for (unsigned senderID : receiverPtr->getNeighbors()) {
                  if (senderID >= agents.size() ||
                      !refinedRobots[senderID]) {
                    continue;
                  }
                  if (adaptiveBudgetMode &&
                      adaptiveTrustRegionIterations[receiverID] <=
                          refineTrustRegionIterations) {
                    continue;
                  }
                  PGOAgent *senderPtr = agents[senderID];
                  std::map<unsigned, double> residualScores;
                  receiverPtr->getNeighborResidualScores(senderID,
                                                         residualScores);
                  const std::vector<unsigned> neededPoses =
                      receiverPtr->getNeighborPublicPoses(senderID);
                  for (unsigned poseIndex : neededPoses) {
                    Matrix sharedPose;
                    if (!senderPtr->getSharedPose(poseIndex, sharedPose)) {
                      continue;
                    }
                    auto &sentToReceiver =
                        lastSentPose[senderID][receiverID];
                    auto &sentIterToReceiver =
                        lastSentIter[senderID][receiverID];
                    const auto poseIt = sentToReceiver.find(poseIndex);
                    const bool neverSent =
                        poseIt == sentToReceiver.end();
                    const auto iterIt = sentIterToReceiver.find(poseIndex);
                    const unsigned age =
                        iterIt == sentIterToReceiver.end()
                            ? eventMaxAge
                            : iter - iterIt->second;
                    const double poseDelta =
                        neverSent ? std::numeric_limits<double>::infinity()
                                  : (sharedPose - poseIt->second).norm();
                    if (!neverSent && poseDelta < feedbackPoseTol) {
                      continue;
                    }
                    double residualScore = 0.0;
                    const auto residualIt = residualScores.find(poseIndex);
                    if (residualIt != residualScores.end()) {
                      residualScore = residualIt->second;
                    }
                    double score = std::numeric_limits<double>::infinity();
                    if (!neverSent) {
                      const double normalizedDelta =
                          poseDelta / std::max(feedbackPoseTol, 1e-12);
                      const double normalizedAge =
                          static_cast<double>(age) /
                          std::max(1.0, static_cast<double>(eventMaxAge));
                      const double normalizedResidual =
                          residualScore /
                          std::max(eventResidualTol, 1e-12);
                      score = std::min(normalizedDelta, 10.0) +
                              0.25 * std::min(normalizedAge, 2.0) +
                              std::log1p(std::max(0.0, normalizedResidual));
                      if (residualScore >= eventResidualTol &&
                          eventResidualTol > 0.0) {
                        score += 1.0;
                      }
                    }
                    budgetCandidates.push_back(
                        {senderID, receiverID, poseIndex, score, neverSent});
                  }
                }
                boundaryBatchModelBudgetCandidates +=
                    budgetCandidates.size();
                if (budgetCandidates.empty()) {
                  continue;
                }

                std::vector<BudgetCandidatePose> rankedCandidates;
                rankedCandidates.reserve(budgetCandidates.size());
                for (const auto &candidate : budgetCandidates) {
                  if (candidate.neverSent) {
                    budgetedFeedbackPoseWhitelistBySender[candidate.senderID]
                        [candidate.receiverID]
                            .insert(candidate.poseIndex);
                  } else {
                    rankedCandidates.push_back(candidate);
                  }
                }
                if (!rankedCandidates.empty()) {
                  double fraction = boundaryBatchBudgetFraction;
                  if (!std::isfinite(fraction) || fraction <= 0.0) {
                    fraction = 1.0;
                  }
                  size_t cap = rankedCandidates.size();
                  if (fraction < 1.0 ||
                      boundaryBatchMaxPosesPerReceiver > 0) {
                    cap = std::max<size_t>(
                        1, static_cast<size_t>(
                               std::ceil(fraction *
                                         rankedCandidates.size())));
                    if (boundaryBatchMaxPosesPerReceiver > 0) {
                      cap = std::min<size_t>(
                          cap, boundaryBatchMaxPosesPerReceiver);
                    }
                    cap = std::min<size_t>(cap, rankedCandidates.size());
                  }
                  std::sort(
                      rankedCandidates.begin(), rankedCandidates.end(),
                      [](const BudgetCandidatePose &a,
                         const BudgetCandidatePose &b) {
                        if (std::abs(a.score - b.score) > 1e-12) {
                          return a.score > b.score;
                        }
                        if (a.senderID != b.senderID) {
                          return a.senderID < b.senderID;
                        }
                        return a.poseIndex < b.poseIndex;
                      });
                  for (size_t idx = 0; idx < cap; ++idx) {
                    const auto &candidate = rankedCandidates[idx];
                    budgetedFeedbackPoseWhitelistBySender[candidate.senderID]
                        [candidate.receiverID]
                            .insert(candidate.poseIndex);
                  }
                }

                size_t selectedForReceiver = 0;
                for (const auto &senderEntries :
                     budgetedFeedbackPoseWhitelistBySender) {
                  const auto receiverIt = senderEntries.find(receiverID);
                  if (receiverIt != senderEntries.end()) {
                    selectedForReceiver += receiverIt->second.size();
                  }
                }
                boundaryBatchModelBudgetSelected += selectedForReceiver;
                boundaryBatchModelBudgetDropped +=
                    budgetCandidates.size() - selectedForReceiver;
              }
            }
            for (auto *senderPtr : agents) {
              const unsigned senderID = senderPtr->getID();
              if (!refinedRobots[senderID]) {
                continue;
              }
              for (unsigned receiverID : senderPtr->getNeighbors()) {
                if (adaptiveBudgetMode &&
                    adaptiveTrustRegionIterations[receiverID] <=
                        refineTrustRegionIterations) {
                  continue;
                }
                PGOAgent *receiverPtr = agents[receiverID];
                PoseDict feedbackPoses;
                PoseDict feedbackBaselineModelPoses;
                PoseDict feedbackModelPoses;
                unsigned boundaryModelCandidatePoses = 0;
                const std::vector<unsigned> neededPoses =
                    receiverPtr->getNeighborPublicPoses(senderID);
                for (unsigned poseIndex : neededPoses) {
                  Matrix sharedPose;
                  if (!senderPtr->getSharedPose(poseIndex, sharedPose)) {
                    continue;
                  }
                  auto &sentToReceiver = lastSentPose[senderID][receiverID];
                  auto &sentIterToReceiver = lastSentIter[senderID][receiverID];
                  const auto poseIt = sentToReceiver.find(poseIndex);
                  const bool neverSent = poseIt == sentToReceiver.end();
                  const auto iterIt = sentIterToReceiver.find(poseIndex);
                  const unsigned age =
                      iterIt == sentIterToReceiver.end()
                          ? eventMaxAge
                          : iter - iterIt->second;
                  const double poseDelta =
                      neverSent ? std::numeric_limits<double>::infinity()
                                : (sharedPose - poseIt->second).norm();
                  if (neverSent || poseDelta >= feedbackPoseTol) {
                    if (budgetedBoundaryBatchModelMode) {
                      const auto receiverWhitelistIt =
                          budgetedFeedbackPoseWhitelistBySender[senderID]
                              .find(receiverID);
                      if (receiverWhitelistIt ==
                              budgetedFeedbackPoseWhitelistBySender[senderID]
                                  .end() ||
                          receiverWhitelistIt->second.find(poseIndex) ==
                              receiverWhitelistIt->second.end()) {
                        continue;
                      }
                    }
                    Matrix baselineModelPose = sharedPose;
                    if (schurFeedbackMode && !neverSent &&
                        neighborDirectionGain > 0.0) {
                      const Matrix senderStep = sharedPose - poseIt->second;
                      const double stepNorm = senderStep.norm();
                      const ROPTResult senderOpt =
                          senderPtr->getLastOptimizationResult();
                      const double modelGain =
                          asyncSchurMode
                              ? neighborDirectionGain /
                                    (1.0 + static_cast<double>(age) +
                                     0.5 * senderOpt.rtrRejectedSteps)
                              : neighborDirectionGain;
                      const double maxCorrectionNorm =
                          std::max(2.0 * communicationPoseTol,
                                   1.5 * stepNorm);
                      const double correctionNorm =
                          modelGain * stepNorm;
                      if (stepNorm > 1e-12 &&
                          correctionNorm <= maxCorrectionNorm) {
                        LiftedSEManifold poseManifold(r, d, 1);
                        baselineModelPose =
                            poseManifold.project(sharedPose +
                                                 modelGain * senderStep);
                      }
                    }
                    Matrix modelPose = baselineModelPose;
                    if (boundaryModelMode && boundaryModelGain > 0.0) {
                      const PoseID boundaryKey =
                          std::make_pair(senderID, poseIndex);
                      const auto predictionIt =
                          feedbackBoundaryModelPredictionsByRobot[senderID]
                              .find(boundaryKey);
                      if (predictionIt !=
                          feedbackBoundaryModelPredictionsByRobot[senderID]
                              .end()) {
                        LiftedSEManifold poseManifold(r, d, 1);
                        modelPose = poseManifold.project(
                            modelPose +
                            boundaryModelGain *
                                (predictionIt->second - sharedPose));
                        ++boundaryModelPredictions;
                        ++boundaryModelCandidatePoses;
                      }
                    }
                    if (ammBoundaryMode && !neverSent &&
                        ammBoundaryBeta > 0.0 && ammBoundaryGamma > 0.0) {
                      const Matrix senderStep = sharedPose - poseIt->second;
                      const double stepNorm = senderStep.norm();
                      bool directionSafe = stepNorm > 1e-12;
                      const auto previousDeltaIt =
                          lastSentDelta[senderID][receiverID].find(poseIndex);
                      if (directionSafe &&
                          previousDeltaIt !=
                              lastSentDelta[senderID][receiverID].end()) {
                        const double previousNorm =
                            previousDeltaIt->second.norm();
                        if (previousNorm > 1e-12) {
                          const double directionCos =
                              (senderStep.array() *
                               previousDeltaIt->second.array())
                                  .sum() /
                              (stepNorm * previousNorm);
                          directionSafe = directionCos >= 0.0;
                        }
                      }
                      if (directionSafe) {
                        const ROPTResult senderOpt =
                            senderPtr->getLastOptimizationResult();
                        const double freshness =
                            1.0 /
                            (1.0 + static_cast<double>(age) +
                             0.5 * senderOpt.rtrRejectedSteps);
                        const double ammGain =
                            ammBoundaryBeta * ammBoundaryGamma * freshness;
                        Matrix correction =
                            (modelPose - sharedPose) + ammGain * senderStep;
                        const double correctionNorm = correction.norm();
                        const double maxCorrectionNorm = std::max(
                            2.0 * communicationPoseTol,
                            ammBoundaryMaxStepScale * stepNorm);
                        if (maxCorrectionNorm > 0.0 &&
                            correctionNorm > maxCorrectionNorm) {
                          correction *= maxCorrectionNorm / correctionNorm;
                        }
                        LiftedSEManifold poseManifold(r, d, 1);
                        modelPose =
                            poseManifold.project(sharedPose + correction);
                        ++boundaryModelPredictions;
                        ++boundaryModelCandidatePoses;
                      }
                    }
                    if ((boundaryModelPacketMode || boundaryResponseMode) &&
                        (boundaryResponseMode ? boundaryResponseStep
                                              : boundaryModelStep) > 0.0 &&
                        (boundaryResponseMode ? boundaryResponseGain
                                              : boundaryModelGain) > 0.0) {
                      const auto packetIt =
                          feedbackBoundaryModelPacketsByRobot[senderID]
                              .find(poseIndex);
                      if (packetIt !=
                          feedbackBoundaryModelPacketsByRobot[senderID]
                              .end()) {
                        const BoundaryInterfaceState &packet =
                            packetIt->second;
                        const double stiffness =
                            std::isfinite(packet.stiffness) &&
                                    packet.stiffness > 1e-8
                                ? packet.stiffness
                                : 1.0;
                        const bool usePacketPreconditionedStep =
                            boundaryPacketUsePreconditionedStep &&
                            packet.preconditionedStep.rows() ==
                                packet.gradient.rows() &&
                            packet.preconditionedStep.cols() ==
                                packet.gradient.cols() &&
                            packet.preconditionedStep.allFinite();
                        Matrix correction =
                            usePacketPreconditionedStep
                                ? packet.preconditionedStep
                                : -packet.gradient / stiffness;
                        const double correctionNorm = correction.norm();
                        const double packetMaxBlockNorm =
                            boundaryResponseMode
                                ? boundaryResponseMaxBlockNorm
                                : boundaryModelMaxBlockNorm;
                        if (packetMaxBlockNorm > 0.0 &&
                            correctionNorm > packetMaxBlockNorm) {
                          correction *= packetMaxBlockNorm / correctionNorm;
                        }
                        const unsigned freshnessAge = neverSent ? eventMaxAge
                                                                : age;
                        const double freshness =
                            1.0 /
                            (1.0 + static_cast<double>(freshnessAge));
                        LiftedSEManifold poseManifold(r, d, 1);
                        const double packetStep =
                            boundaryResponseMode ? boundaryResponseStep
                                                 : boundaryModelStep;
                        const double packetGain =
                            boundaryResponseMode ? boundaryResponseGain
                                                 : boundaryModelGain;
                        const Matrix packetModelPose = poseManifold.project(
                            packet.pose +
                            packetStep * freshness * correction);
                        modelPose = poseManifold.project(
                            modelPose + packetGain *
                                            (packetModelPose - packet.pose));
                        ++boundaryModelPacketPayloadBlocks;
                        ++boundaryModelPacketModelPoseBlocks;
                        boundaryModelPacketGradNorm +=
                            packet.gradient.norm();
                        boundaryModelPacketStiffnessSum += stiffness;
                        boundaryModelPacketFreshnessSum += freshness;
                        if (usePacketPreconditionedStep &&
                            packet.preconditionedStep.size() > 0 &&
                            packet.preconditionedStep.allFinite()) {
                          boundaryModelPacketPreconditionedStepNorm +=
                              packet.preconditionedStep.norm();
                        }
                        if (usePacketPreconditionedStep &&
                            std::isfinite(packet.schurSensitivity)) {
                          boundaryModelPacketSchurSensitivitySum +=
                              packet.schurSensitivity;
                        }
                        if (usePacketPreconditionedStep &&
                            std::isfinite(packet.reducedPreconditioner)) {
                          boundaryModelPacketReducedPreconditionerSum +=
                              packet.reducedPreconditioner;
                        }
                        boundaryModelPacketExtraPayloadBytes +=
                            boundaryModelPacketPayloadBytes > posePayloadBytes
                                ? boundaryModelPacketPayloadBytes -
                                      posePayloadBytes
                                : 0;
                        ++boundaryModelCandidatePoses;
                      }
                    }
                    const PoseID feedbackKey =
                        std::make_pair(senderID, poseIndex);
                    const Matrix transmittedPose =
                        updateSentPoseWithOptionalDelta(
                            senderID, receiverID, poseIndex, sharedPose,
                            neverSent);
                    feedbackPoses[feedbackKey] = transmittedPose;
                    if (interfaceDeltaMode) {
                      feedbackBaselineModelPoses[feedbackKey] =
                          transmittedPose;
                      feedbackModelPoses[feedbackKey] = transmittedPose;
                    } else {
                      feedbackBaselineModelPoses[feedbackKey] =
                          baselineModelPose;
                      feedbackModelPoses[feedbackKey] = modelPose;
                    }
                  }
                }
                if (!feedbackPoses.empty()) {
                  iterCommPoses += feedbackPoses.size();
                  receiverPtr->setNeighborStatus(senderPtr->getStatus());
                  receiverPtr->updateNeighborPoses(senderID, feedbackPoses);
                  if (boundaryBatchModelMode && schurFeedbackMode) {
                    auto &batchBaseline =
                        batchBaselineModelPosesByReceiver[receiverID];
                    auto &batchModel =
                        batchModelPosesByReceiver[receiverID];
                    for (const auto &it : feedbackPoses) {
                      batchBaseline[it.first] = it.second;
                    }
                    for (const auto &it : feedbackModelPoses) {
                      batchModel[it.first] = it.second;
                    }
                    batchCandidatePosesByReceiver[receiverID] +=
                        feedbackModelPoses.size();
                  } else if (schurFeedbackMode) {
                    if (!asyncSchurMode ||
                        !schurFeedbackApplied[receiverID]) {
                      const auto t0 =
                          std::chrono::high_resolution_clock::now();
                      const unsigned localTrustRegionIterations =
                          adaptiveTrustRegionIterations[receiverID];
                      if (boundaryResponseMode &&
                          boundaryModelCandidatePoses > 0) {
                        ++boundaryResponseAttempted;
                        double responseCostBefore = 0.0;
                        double responseCostAfter = 0.0;
                        double responseStepNorm = 0.0;
                        double responseGradDeltaNorm = 0.0;
                        unsigned responseBlocks = 0;
                        const auto responseT0 =
                            std::chrono::high_resolution_clock::now();
                        const bool responseAccepted =
                            receiverPtr->applyBoundaryResponseCorrection(
                                senderID, feedbackPoses, feedbackModelPoses,
                                boundaryResponseGain, boundaryResponseStep,
                                boundaryResponseMaxBlockNorm,
                                boundaryResponseRequireDecrease,
                                boundaryResponseBacktrackingSteps,
                                responseCostBefore, responseCostAfter,
                                responseStepNorm, responseGradDeltaNorm,
                                responseBlocks);
                        const double responseDtMs = elapsedMs(responseT0);
                        boundaryResponseMs += responseDtMs;
                        boundaryResponseBlocks += responseBlocks;
                        boundaryResponseGradDeltaNorm +=
                            responseGradDeltaNorm;
                        if (responseAccepted) {
                          ++boundaryResponseAccepted;
                          boundaryResponseStepNorm += responseStepNorm;
                          boundaryResponseCostDecrease +=
                              std::max(0.0, responseCostBefore -
                                                responseCostAfter);
                        } else {
                          ++boundaryResponseRejected;
                        }
                        receiverPtr->refineLocalOptimizationWithNeighborModel(
                            senderID, feedbackBaselineModelPoses, 1,
                            localTrustRegionIterations);
                      } else {
                        const PoseDict *selectedModelPoses =
                            &feedbackModelPoses;
                        bool boundaryModelCandidateSelected = false;
                        double boundaryModelCandidateMeritDecrease = 0.0;
                        double boundaryModelCandidateGradDecrease = 0.0;
                        if ((boundaryModelMode || boundaryModelPacketMode ||
                             ammBoundaryMode) &&
                            boundaryModelCandidatePoses > 0) {
                          double baselineCost = 0.0;
                          double baselineGradNorm = 0.0;
                          double candidateCost = 0.0;
                          double candidateGradNorm = 0.0;
                          const auto meritT0 =
                              std::chrono::high_resolution_clock::now();
                          const bool baselineOk =
                              receiverPtr->evaluateLocalModelWithNeighborModel(
                                  senderID, feedbackBaselineModelPoses,
                                  baselineCost, baselineGradNorm);
                          const bool candidateOk =
                              receiverPtr->evaluateLocalModelWithNeighborModel(
                                  senderID, feedbackModelPoses, candidateCost,
                                  candidateGradNorm);
                          const double meritDtMs = elapsedMs(meritT0);
                          boundaryModelMeritMs += meritDtMs;
                          ++boundaryModelMeritEvaluations;
                          if (baselineOk && candidateOk &&
                              candidateCost <= baselineCost &&
                              candidateGradNorm <= baselineGradNorm) {
                            boundaryModelCandidateSelected = true;
                            boundaryModelCandidateMeritDecrease =
                                std::max(0.0,
                                         baselineCost - candidateCost);
                            boundaryModelCandidateGradDecrease =
                                std::max(0.0,
                                         baselineGradNorm -
                                             candidateGradNorm);
                          } else {
                            ++boundaryModelMeritRejected;
                            selectedModelPoses = &feedbackBaselineModelPoses;
                          }
                        }
                        Matrix receiverXBefore;
                        double trueCostBefore = 0.0;
                        double trueGradBefore = 0.0;
                        if (boundaryModelCandidateSelected &&
                            (!receiverPtr->getX(receiverXBefore) ||
                             !receiverPtr->evaluateLocalModel(
                                 trueCostBefore, trueGradBefore))) {
                          boundaryModelCandidateSelected = false;
                          selectedModelPoses = &feedbackBaselineModelPoses;
                          ++boundaryModelMeritRejected;
                        }
                        if (boundaryModelCandidateSelected) {
                          receiverPtr->refineLocalOptimizationWithNeighborModel(
                              senderID, feedbackModelPoses, 1,
                              localTrustRegionIterations);
                          double trueCostAfter = 0.0;
                          double trueGradAfter = 0.0;
                          const bool postOk =
                              receiverPtr->evaluateLocalModel(trueCostAfter,
                                                              trueGradAfter);
                          if (postOk && trueCostAfter <= trueCostBefore) {
                            ++boundaryModelMeritAccepted;
                            boundaryModelMeritDecrease +=
                                boundaryModelCandidateMeritDecrease;
                            boundaryModelMeritGradDecrease +=
                                boundaryModelCandidateGradDecrease;
                          } else {
                            receiverPtr->setX(receiverXBefore);
                            receiverPtr
                                ->refineLocalOptimizationWithNeighborModel(
                                    senderID, feedbackBaselineModelPoses, 1,
                                    localTrustRegionIterations);
                            ++boundaryModelMeritRejected;
                          }
                        } else {
                          receiverPtr->refineLocalOptimizationWithNeighborModel(
                              senderID, *selectedModelPoses, 1,
                              localTrustRegionIterations);
                        }
                      }
                      const double dtMs = elapsedMs(t0);
                      iterSequentialComputeMs += dtMs;
                      feedbackRefineMs[receiverID] += dtMs;
                      refinedRobots[receiverID] = true;
                      schurFeedbackApplied[receiverID] = true;
                      if (localTrustRegionIterations >
                          refineTrustRegionIterations) {
                        adaptiveRefinedThisIter[receiverID] = true;
                        adaptiveRefineCalls++;
                        adaptiveRefineMs += dtMs;
                      }
                    }
                  } else {
                    feedbackRefineRobots[receiverID] = true;
                  }
                }
              }
            }
            if (boundaryBatchModelMode) {
              for (auto *receiverPtr : agents) {
                const unsigned receiverID = receiverPtr->getID();
                PoseDict &candidatePoses =
                    batchModelPosesByReceiver[receiverID];
                if (candidatePoses.empty()) {
                  continue;
                }
                const auto t0 =
                    std::chrono::high_resolution_clock::now();
                ++boundaryBatchModelReceivers;
                boundaryBatchModelCandidatePoses +=
                    batchCandidatePosesByReceiver[receiverID];
                const PoseDict &baselinePoses =
                    batchBaselineModelPosesByReceiver[receiverID];
                if (boundaryBatchResponseLikeMode) {
                  std::map<PoseID, BoundaryInterfaceState>
                      schurResponsePackets;
                  if (boundaryPacketSchurResponseMode) {
                    for (const auto &poseIt : candidatePoses) {
                      const PoseID boundaryKey = poseIt.first;
                      const unsigned senderID = boundaryKey.first;
                      const unsigned poseIndex = boundaryKey.second;
                      if (senderID >=
                          feedbackBoundaryModelPacketsByRobot.size()) {
                        continue;
                      }
                      const auto packetIt =
                          feedbackBoundaryModelPacketsByRobot[senderID]
                              .find(poseIndex);
                      if (packetIt ==
                          feedbackBoundaryModelPacketsByRobot[senderID]
                              .end()) {
                        continue;
                      }
                      const BoundaryInterfaceState &packet =
                          packetIt->second;
                      schurResponsePackets[boundaryKey] = packet;
                      ++boundaryModelPacketPayloadBlocks;
                      boundaryModelPacketGradNorm += packet.gradient.norm();
                      boundaryModelPacketStiffnessSum += packet.stiffness;
                      boundaryModelPacketFreshnessSum += 1.0;
                      if (boundaryPacketUsePreconditionedStep &&
                          packet.preconditionedStep.size() > 0 &&
                          packet.preconditionedStep.allFinite()) {
                        boundaryModelPacketPreconditionedStepNorm +=
                            packet.preconditionedStep.norm();
                      }
                      if (boundaryPacketSendsSchurScalars &&
                          std::isfinite(packet.schurSensitivity)) {
                        boundaryModelPacketSchurSensitivitySum +=
                            packet.schurSensitivity;
                      }
                      if (boundaryPacketSendsSchurScalars &&
                          std::isfinite(packet.reducedPreconditioner)) {
                        boundaryModelPacketReducedPreconditionerSum +=
                            packet.reducedPreconditioner;
                      }
                      boundaryModelPacketExtraPayloadBytes +=
                          boundaryModelPacketPayloadBytes > posePayloadBytes
                              ? boundaryModelPacketPayloadBytes -
                                    posePayloadBytes
                              : 0;
                    }
                  }
                  const auto applyBatchResponse = [&]() {
                    ++boundaryBatchResponseAttempted;
                    double responseCostBefore = 0.0;
                    double responseCostAfter = 0.0;
                    double responseStepNorm = 0.0;
                    double responseGradDeltaNorm = 0.0;
                    unsigned responseBlocks = 0;
                    const auto responseT0 =
                        std::chrono::high_resolution_clock::now();
                    const bool responseAccepted =
                        boundarySchurResponseMode
                            ? receiverPtr
                                  ->applyBoundarySchurResponseCorrectionWithNeighborModels(
                                      candidatePoses,
                                      boundarySchurResponseGain,
                                      boundarySchurResponseStep,
                                      boundarySchurResponseMaxBlockNorm,
                                      boundarySchurResponseDamping,
                                      boundarySchurResponseMaxBlocks,
                                      boundarySchurResponseRequireDecrease,
                                      boundarySchurResponseBacktrackingSteps,
                                      responseCostBefore, responseCostAfter,
                                      responseStepNorm,
                                      responseGradDeltaNorm,
                                      responseBlocks,
                                      boundaryPacketSchurResponseMode &&
                                              !schurResponsePackets.empty()
                                          ? &schurResponsePackets
                                          : nullptr,
                                      boundaryPacketSchurDampingGain,
                                      boundaryPacketSchurDampingMax)
                            : receiverPtr
                                  ->applyBoundaryResponseCorrectionWithNeighborModels(
                                      candidatePoses,
                                      boundaryBatchResponseGain,
                                      boundaryBatchResponseStep,
                                      boundaryBatchResponseMaxBlockNorm,
                                      boundaryBatchResponseRequireDecrease,
                                      boundaryBatchResponseBacktrackingSteps,
                                      responseCostBefore, responseCostAfter,
                                      responseStepNorm,
                                      responseGradDeltaNorm,
                                      responseBlocks);
                    boundaryBatchResponseMs += elapsedMs(responseT0);
                    boundaryBatchResponseBlocks += responseBlocks;
                    boundaryBatchResponseGradDeltaNorm +=
                        responseGradDeltaNorm;
                    if (responseAccepted) {
                      ++boundaryBatchResponseAccepted;
                      boundaryBatchResponseStepNorm += responseStepNorm;
                      boundaryBatchResponseCostDecrease +=
                          std::max(0.0, responseCostBefore -
                                            responseCostAfter);
                    } else {
                      ++boundaryBatchResponseRejected;
                    }
                    return responseAccepted;
                  };
                  const bool postRefineResponse =
                      boundarySchurResponseMode
                          ? boundarySchurResponsePostRefine
                          : boundaryBatchResponsePostRefine;
                  if (postRefineResponse) {
                    receiverPtr->refineLocalOptimizationWithNeighborModels(
                        baselinePoses, 1,
                        adaptiveTrustRegionIterations[receiverID]);
                    applyBatchResponse();
                  } else {
                    const bool responseAccepted = applyBatchResponse();
                    const bool skipBaselineRefine =
                        boundarySchurResponseMode &&
                        boundarySchurResponseSkipBaselineRefine &&
                        responseAccepted &&
                        (boundarySchurResponseBaselineRefinePeriod == 0 ||
                         iter % boundarySchurResponseBaselineRefinePeriod !=
                             boundarySchurResponseBaselineRefineOffset %
                                 boundarySchurResponseBaselineRefinePeriod);
                    if (!skipBaselineRefine) {
                      receiverPtr->refineLocalOptimizationWithNeighborModels(
                          baselinePoses, 1,
                          adaptiveTrustRegionIterations[receiverID]);
                    }
                  }
                } else {
                  ++boundaryBatchModelMeritEvaluations;
                  double baselineCost = 0.0;
                  double baselineGradNorm = 0.0;
                  double candidateCost = 0.0;
                  double candidateGradNorm = 0.0;
                  const bool baselineOk =
                      receiverPtr->evaluateLocalModelWithNeighborModels(
                          baselinePoses, baselineCost, baselineGradNorm);
                  const bool candidateOk =
                      receiverPtr->evaluateLocalModelWithNeighborModels(
                          candidatePoses, candidateCost, candidateGradNorm);
                  const bool candidateAccepted =
                      !boundaryBatchModelRequireMerit ||
                      (baselineOk && candidateOk &&
                       candidateCost <= baselineCost &&
                       candidateGradNorm <= baselineGradNorm);
                  if (candidateAccepted) {
                    ++boundaryBatchModelMeritAccepted;
                    boundaryBatchModelMeritDecrease +=
                        std::max(0.0, baselineCost - candidateCost);
                    boundaryBatchModelMeritGradDecrease +=
                        std::max(0.0, baselineGradNorm - candidateGradNorm);
                    receiverPtr->refineLocalOptimizationWithNeighborModels(
                        candidatePoses, 1,
                        adaptiveTrustRegionIterations[receiverID]);
                  } else {
                    ++boundaryBatchModelMeritRejected;
                    receiverPtr->refineLocalOptimizationWithNeighborModels(
                        baselinePoses, 1,
                        adaptiveTrustRegionIterations[receiverID]);
                  }
                }
                const double dtMs = elapsedMs(t0);
                iterSequentialComputeMs += dtMs;
                feedbackRefineMs[receiverID] += dtMs;
                boundaryBatchModelRefineMs += dtMs;
                refinedRobots[receiverID] = true;
                if (adaptiveTrustRegionIterations[receiverID] >
                    refineTrustRegionIterations) {
                  adaptiveRefinedThisIter[receiverID] = true;
                  adaptiveRefineCalls++;
                  adaptiveRefineMs += dtMs;
                }
              }
            }
            for (auto *robotPtr : agents) {
              const unsigned robotID = robotPtr->getID();
              if (feedbackRefineRobots[robotID] && !refinedRobots[robotID]) {
                const unsigned localTrustRegionIterations =
                    adaptiveTrustRegionIterations[robotID];
                const auto t0 = std::chrono::high_resolution_clock::now();
                robotPtr->refineLocalOptimization(
                    1, false, localTrustRegionIterations);
                const double dtMs = elapsedMs(t0);
                iterSequentialComputeMs += dtMs;
                feedbackRefineMs[robotID] += dtMs;
                if (localTrustRegionIterations > refineTrustRegionIterations) {
                  adaptiveRefinedThisIter[robotID] = true;
                  adaptiveRefineCalls++;
                  adaptiveRefineMs += dtMs;
                }
              }
            }
            iterParallelComputeMs +=
                *std::max_element(feedbackRefineMs.begin(),
                                  feedbackRefineMs.end());
          }
        }
      }
      if (boundaryJacobiMode) {
        double boundaryJacobiStageMaxMs = 0.0;
        for (auto *robotPtr : agents) {
          const unsigned robotID = robotPtr->getID();
          if (!needsBoundaryJacobiCorrection[robotID]) {
            continue;
          }
          ++boundaryJacobiAttempted;
          double costBeforeCorrection = 0.0;
          double costAfterCorrection = 0.0;
          double correctionStepNorm = 0.0;
          unsigned correctedBlocks = 0;
          const auto t0 = std::chrono::high_resolution_clock::now();
          const bool accepted = robotPtr->applyBoundaryJacobiCorrection(
              boundaryJacobiStep, boundaryJacobiMaxBlockNorm,
              boundaryJacobiRequireDecrease, boundaryJacobiBacktrackingSteps,
              costBeforeCorrection, costAfterCorrection, correctionStepNorm,
              correctedBlocks);
          const double dtMs = elapsedMs(t0);
          iterSequentialComputeMs += dtMs;
          boundaryJacobiStageMaxMs =
              std::max(boundaryJacobiStageMaxMs, dtMs);
          boundaryJacobiMs += dtMs;
          boundaryJacobiBlocks += correctedBlocks;
          if (accepted) {
            ++boundaryJacobiAccepted;
            boundaryJacobiStepNorm += correctionStepNorm;
            boundaryJacobiCostDecrease +=
                std::max(0.0, costBeforeCorrection - costAfterCorrection);
          } else {
            ++boundaryJacobiRejected;
          }
        }
        iterParallelComputeMs += boundaryJacobiStageMaxMs;
      }
      if (sweepCorrectionMode && iter > 0 && !consistencySweepRound) {
        double sweepStageMaxMs = 0.0;
        for (auto *robotPtr : agents) {
          const unsigned robotID = robotPtr->getID();
          if (eventColor(robotID) == nextActiveColor) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            robotPtr->refineLocalOptimization(1, false);
            const double dtMs = elapsedMs(t0);
            iterSequentialComputeMs += dtMs;
            sweepStageMaxMs = std::max(sweepStageMaxMs, dtMs);
          }
        }
        iterParallelComputeMs += sweepStageMaxMs;
      }
    } else if (updateMode == "baseline_full") {
      PGOAgent *selectedRobotPtr = agents[selectedRobot];
      selectedRobotForLog = static_cast<int>(selectedRobotPtr->getID());

      // Legacy baseline: selected robot communicates with every other robot and
      // receives each sender's whole public separator dictionary.
      double idleStageMaxMs = 0.0;
      for (auto *robotPtr : agents) {
        assert(robotPtr->instance_number() == 0);
        assert(robotPtr->iteration_number() == iter);
        if (robotPtr->getID() != selectedRobot) {
          const auto t0 = std::chrono::high_resolution_clock::now();
          robotPtr->iterate(false);
          const double dtMs = elapsedMs(t0);
          iterSequentialComputeMs += dtMs;
          idleStageMaxMs = std::max(idleStageMaxMs, dtMs);
        }
      }
      iterParallelComputeMs += idleStageMaxMs;

      for (auto *robotPtr : agents) {
        if (robotPtr->getID() == selectedRobot) continue;
        PoseDict sharedPoses;
        if (!robotPtr->getSharedPoseDict(sharedPoses)) {
          continue;
        }
        iterCommPoses += sharedPoses.size();
        selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
        selectedRobotPtr->updateNeighborPoses(robotPtr->getID(), sharedPoses);
      }

      if (acceleration) {
        for (auto *robotPtr : agents) {
          if (robotPtr->getID() == selectedRobot) continue;
          PoseDict auxSharedPoses;
          if (!robotPtr->getAuxSharedPoseDict(auxSharedPoses)) {
            continue;
          }
          iterCommPoses += auxSharedPoses.size();
          selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
          selectedRobotPtr->updateAuxNeighborPoses(robotPtr->getID(),
                                                   auxSharedPoses);
        }
      }

      {
        const auto t0 = std::chrono::high_resolution_clock::now();
        selectedRobotPtr->iterate(true);
        const double dtMs = elapsedMs(t0);
        iterSequentialComputeMs += dtMs;
        iterParallelComputeMs += dtMs;
      }
    } else {
      PGOAgent *selectedRobotPtr = agents[selectedRobot];
      selectedRobotForLog = static_cast<int>(selectedRobotPtr->getID());

      // Non-selected robots perform an iteration
      double idleStageMaxMs = 0.0;
      for (auto *robotPtr : agents) {
        assert(robotPtr->instance_number() == 0);
        assert(robotPtr->iteration_number() == iter);
        if (robotPtr->getID() != selectedRobot) {
          const auto t0 = std::chrono::high_resolution_clock::now();
          robotPtr->iterate(false);
          const double dtMs = elapsedMs(t0);
          iterSequentialComputeMs += dtMs;
          idleStageMaxMs = std::max(idleStageMaxMs, dtMs);
        }
      }
      iterParallelComputeMs += idleStageMaxMs;

      // Selected robot requests only neighbor poses used by its local factors.
      for (unsigned neighborID : selectedRobotPtr->getNeighbors()) {
        PGOAgent *robotPtr = agents[neighborID];
        PoseDict sharedPoses;
        for (unsigned poseIndex :
             selectedRobotPtr->getNeighborPublicPoses(neighborID)) {
          Matrix sharedPose;
          if (robotPtr->getSharedPose(poseIndex, sharedPose)) {
            sharedPoses[std::make_pair(neighborID, poseIndex)] = sharedPose;
          }
        }
        iterCommPoses += sharedPoses.size();
        selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
        selectedRobotPtr->updateNeighborPoses(robotPtr->getID(), sharedPoses);
      }

      // When using acceleration, selected robot also requests auxiliary poses
      if (acceleration) {
        for (unsigned neighborID : selectedRobotPtr->getNeighbors()) {
          PGOAgent *robotPtr = agents[neighborID];
          PoseDict auxSharedPoses;
          for (unsigned poseIndex :
               selectedRobotPtr->getNeighborPublicPoses(neighborID)) {
            Matrix auxSharedPose;
            if (robotPtr->getAuxSharedPose(poseIndex, auxSharedPose)) {
              auxSharedPoses[std::make_pair(neighborID, poseIndex)] =
                  auxSharedPose;
            }
          }
          iterCommPoses += auxSharedPoses.size();
          selectedRobotPtr->setNeighborStatus(robotPtr->getStatus());
          selectedRobotPtr->updateAuxNeighborPoses(robotPtr->getID(),
                                                   auxSharedPoses);
        }
      }

      // Selected robot update
      {
        const auto t0 = std::chrono::high_resolution_clock::now();
        selectedRobotPtr->iterate(true);
        const double dtMs = elapsedMs(t0);
        iterSequentialComputeMs += dtMs;
        iterParallelComputeMs += dtMs;
      }
    }

    // Form centralized solution
    for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
      unsigned startIdx = robot * num_poses_per_robot;
      unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
      if (robot == (unsigned) num_robots - 1) endIdx = n;

      Matrix XRobot;
      if (agents[robot]->getX(XRobot)) {
        Xopt.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)) = XRobot;
      }
    }
    Matrix RGrad = problemCentral.RieGrad(Xopt);
    double RGradNorm  = RGrad.norm();
    double cost = 2 * problemCentral.f(Xopt);
    double relCostChange = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(prevCost)) {
      relCostChange = std::abs(prevCost - cost) /
                      std::max(1.0, std::abs(prevCost));
      if (relCostChange < relCostTol) {
        stableLossIters++;
      } else {
        stableLossIters = 0;
      }
    }
    const double interfaceDeltaCommMB =
        interfaceDeltaMode
            ? static_cast<double>(iterCommPayloadBytes) / (1024.0 * 1024.0)
            : 0.0;
    const double boundaryModelPacketCommMB =
        static_cast<double>(boundaryModelPacketPayloadBlocks) *
        boundaryModelPacketPayloadMB;
    const double boundaryModelPacketExtraCommMB =
        static_cast<double>(boundaryModelPacketExtraPayloadBytes) /
        (1024.0 * 1024.0);
    const double iterCommMB =
        interfaceDeltaMode
            ? interfaceDeltaCommMB
            : static_cast<double>(iterCommPoses) * posePayloadMB +
                  boundaryModelPacketExtraCommMB;
    const double interfaceStateCommMB =
        static_cast<double>(interfaceStatePayloadBlocks) *
        interfaceStatePayloadMB;
    cumulativeCommPoses += iterCommPoses;
    cumulativeCommMB += iterCommMB;
    cumulativeParallelComputeMs += iterParallelComputeMs;
    adaptiveRefineRobots =
        static_cast<size_t>(std::count(adaptiveRefinedThisIter.begin(),
                                       adaptiveRefinedThisIter.end(), true));
    cumulativeAdaptiveRefineRobots += adaptiveRefineRobots;
    cumulativeAdaptiveRefineCalls += adaptiveRefineCalls;
    cumulativeAdaptiveRefineMs += adaptiveRefineMs;
    cumulativeLocalAmmAttempted += localAmmAttempted;
    cumulativeLocalAmmAccepted += localAmmAccepted;
    cumulativeLocalAmmRejected += localAmmRejected;
    cumulativeLocalAmmCostDecrease += localAmmCostDecrease;
    cumulativeLocalAmmGradDecrease += localAmmGradDecrease;
    cumulativeLocalAmmStepNorm += localAmmStepNorm;
    cumulativeLocalAmmMs += localAmmMs;
    cumulativeLocalAmmBaselineCompared += localAmmBaselineCompared;
    cumulativeLocalAmmSelected += localAmmSelected;
    cumulativeLocalAmmBaselineSelected += localAmmBaselineSelected;
    cumulativeLocalAmmVsBaselineCostDelta += localAmmVsBaselineCostDelta;
    cumulativeLocalAmmVsBaselineGradDelta += localAmmVsBaselineGradDelta;
    cumulativeMajorizedBoundaryCandidates += majorizedBoundaryCandidates;
    cumulativeMajorizedBoundarySelected += majorizedBoundarySelected;
    cumulativeMajorizedBoundaryScore += majorizedBoundaryScoreSum;
    cumulativeMajorizedBoundarySelectedScore +=
        majorizedBoundarySelectedScoreSum;
    cumulativeBoundaryJacobiAttempted += boundaryJacobiAttempted;
    cumulativeBoundaryJacobiAccepted += boundaryJacobiAccepted;
    cumulativeBoundaryJacobiRejected += boundaryJacobiRejected;
    cumulativeBoundaryJacobiBlocks += boundaryJacobiBlocks;
    cumulativeBoundaryJacobiStepNorm += boundaryJacobiStepNorm;
    cumulativeBoundaryJacobiCostDecrease += boundaryJacobiCostDecrease;
    cumulativeBoundaryJacobiMs += boundaryJacobiMs;
    cumulativeBoundaryModelPredictions += boundaryModelPredictions;
    cumulativeBoundaryModelStepNorm += boundaryModelStepNorm;
    cumulativeBoundaryModelCostDecrease += boundaryModelCostDecrease;
    cumulativeBoundaryModelMs += boundaryModelMs;
    cumulativeBoundaryModelMeritEvaluations +=
        boundaryModelMeritEvaluations;
    cumulativeBoundaryModelMeritAccepted += boundaryModelMeritAccepted;
    cumulativeBoundaryModelMeritRejected += boundaryModelMeritRejected;
    cumulativeBoundaryModelMeritDecrease += boundaryModelMeritDecrease;
    cumulativeBoundaryModelMeritGradDecrease +=
        boundaryModelMeritGradDecrease;
    cumulativeBoundaryModelMeritMs += boundaryModelMeritMs;
    cumulativeBoundaryModelPacketPayloadBlocks +=
        boundaryModelPacketPayloadBlocks;
    cumulativeBoundaryModelPacketModelPoseBlocks +=
        boundaryModelPacketModelPoseBlocks;
    cumulativeBoundaryModelPacketGradNorm += boundaryModelPacketGradNorm;
    cumulativeBoundaryModelPacketStiffnessSum +=
        boundaryModelPacketStiffnessSum;
    cumulativeBoundaryModelPacketFreshnessSum +=
        boundaryModelPacketFreshnessSum;
    cumulativeBoundaryModelPacketPreconditionedStepNorm +=
        boundaryModelPacketPreconditionedStepNorm;
    cumulativeBoundaryModelPacketSchurSensitivitySum +=
        boundaryModelPacketSchurSensitivitySum;
    cumulativeBoundaryModelPacketReducedPreconditionerSum +=
        boundaryModelPacketReducedPreconditionerSum;
    cumulativeBoundaryModelPacketCommMB += boundaryModelPacketCommMB;
    cumulativeBoundaryResponseAttempted += boundaryResponseAttempted;
    cumulativeBoundaryResponseAccepted += boundaryResponseAccepted;
    cumulativeBoundaryResponseRejected += boundaryResponseRejected;
    cumulativeBoundaryResponseBlocks += boundaryResponseBlocks;
    cumulativeBoundaryResponseStepNorm += boundaryResponseStepNorm;
    cumulativeBoundaryResponseGradDeltaNorm += boundaryResponseGradDeltaNorm;
    cumulativeBoundaryResponseCostDecrease += boundaryResponseCostDecrease;
    cumulativeBoundaryResponseMs += boundaryResponseMs;
    cumulativeBoundaryBatchResponseAttempted +=
        boundaryBatchResponseAttempted;
    cumulativeBoundaryBatchResponseAccepted +=
        boundaryBatchResponseAccepted;
    cumulativeBoundaryBatchResponseRejected +=
        boundaryBatchResponseRejected;
    cumulativeBoundaryBatchResponseBlocks += boundaryBatchResponseBlocks;
    cumulativeBoundaryBatchResponseStepNorm +=
        boundaryBatchResponseStepNorm;
    cumulativeBoundaryBatchResponseGradDeltaNorm +=
        boundaryBatchResponseGradDeltaNorm;
    cumulativeBoundaryBatchResponseCostDecrease +=
        boundaryBatchResponseCostDecrease;
    cumulativeBoundaryBatchResponseMs += boundaryBatchResponseMs;
    cumulativeBoundaryBatchModelReceivers += boundaryBatchModelReceivers;
    cumulativeBoundaryBatchModelCandidatePoses +=
        boundaryBatchModelCandidatePoses;
    cumulativeBoundaryBatchModelMeritEvaluations +=
        boundaryBatchModelMeritEvaluations;
    cumulativeBoundaryBatchModelMeritAccepted +=
        boundaryBatchModelMeritAccepted;
    cumulativeBoundaryBatchModelMeritRejected +=
        boundaryBatchModelMeritRejected;
    cumulativeBoundaryBatchModelMeritDecrease +=
        boundaryBatchModelMeritDecrease;
    cumulativeBoundaryBatchModelMeritGradDecrease +=
        boundaryBatchModelMeritGradDecrease;
    cumulativeBoundaryBatchModelRefineMs += boundaryBatchModelRefineMs;
    cumulativeBoundaryBatchModelBudgetCandidates +=
        boundaryBatchModelBudgetCandidates;
    cumulativeBoundaryBatchModelBudgetSelected +=
        boundaryBatchModelBudgetSelected;
    cumulativeBoundaryBatchModelBudgetDropped +=
        boundaryBatchModelBudgetDropped;
    cumulativeInterfaceStatePayloadBlocks += interfaceStatePayloadBlocks;
    cumulativeInterfaceStateCommMB += interfaceStateCommMB;
    cumulativeInterfaceStateComputeMs += interfaceStateComputeMs;
    cumulativeInterfaceDeltaCandidatePoseBlocks +=
        interfaceDeltaCandidatePoseBlocks;
    cumulativeInterfaceDeltaCacheMissPoseBlocks +=
        interfaceDeltaCacheMissPoseBlocks;
    cumulativeInterfaceDeltaRejectedPoseBlocks +=
        interfaceDeltaRejectedPoseBlocks;
    cumulativeInterfaceDeltaSparsePoseBlocks += interfaceDeltaSparsePoseBlocks;
    cumulativeInterfaceDeltaTangentPoseBlocks +=
        interfaceDeltaTangentPoseBlocks;
    cumulativeInterfaceDeltaFullPoseBlocks += interfaceDeltaFullPoseBlocks;
    cumulativeInterfaceDeltaDeltaPoseBlocks += interfaceDeltaDeltaPoseBlocks;
    cumulativeInterfaceDeltaDeltaEntries += interfaceDeltaDeltaEntries;
    cumulativeInterfaceDeltaCommMB += interfaceDeltaCommMB;
    finalCost = cost;
    finalGradNorm = RGradNorm;
    finalIter = iter;
    const bool lossTolReached = stableLossIters >= stableLossRequired;
    const bool gradTolReached = RGradNorm < gradTol;
    const bool bothTolReached = lossTolReached && gradTolReached;

    if (lossTolIter < 0 && lossTolReached) {
      lossTolIter = static_cast<int>(iter);
      lossTolCommPoses = cumulativeCommPoses;
      lossTolCommMB = cumulativeCommMB;
      std::cout << "Loss tolerance reached at iter " << lossTolIter
                << " | comm_poses = " << lossTolCommPoses
                << " | comm_mb = " << lossTolCommMB << std::endl;
    }
    if (gradTolIter < 0 && gradTolReached) {
      gradTolIter = static_cast<int>(iter);
      gradTolCommPoses = cumulativeCommPoses;
      gradTolCommMB = cumulativeCommMB;
      std::cout << "Grad tolerance reached at iter " << gradTolIter
                << " | comm_poses = " << gradTolCommPoses
                << " | comm_mb = " << gradTolCommMB << std::endl;
    }

    std::cout << std::setprecision(6)
              << "Iter = " << iter << " | "
              << "configured_mode = " << configuredUpdateMode << " | "
              << "effective_mode = " << updateMode << " | "
              << "cost = " << cost << " | "
              << "gradnorm = " << RGradNorm << " | "
              << "comm_poses = " << iterCommPoses << " | "
              << "comm_mb = " << iterCommMB << " | "
              << "cum_comm_mb = " << cumulativeCommMB << " | "
              << "seq_compute_ms = " << iterSequentialComputeMs << " | "
              << "parallel_compute_ms = " << iterParallelComputeMs << " | "
              << "adaptive_refine_robots = " << adaptiveRefineRobots << " | "
              << "adaptive_refine_calls = " << adaptiveRefineCalls << " | "
              << "adaptive_refine_ms = " << adaptiveRefineMs << " | "
              << "local_amm_attempted = " << localAmmAttempted << " | "
              << "local_amm_accepted = " << localAmmAccepted << " | "
              << "local_amm_rejected = " << localAmmRejected << " | "
              << "local_amm_cost_decrease = "
              << localAmmCostDecrease << " | "
              << "local_amm_grad_decrease = "
              << localAmmGradDecrease << " | "
              << "local_amm_step_norm = " << localAmmStepNorm << " | "
              << "local_amm_ms = " << localAmmMs << " | "
              << "local_amm_baseline_compared = "
              << localAmmBaselineCompared << " | "
              << "local_amm_selected = " << localAmmSelected << " | "
              << "local_amm_baseline_selected = "
              << localAmmBaselineSelected << " | "
              << "local_amm_vs_baseline_cost_delta = "
              << localAmmVsBaselineCostDelta << " | "
              << "local_amm_vs_baseline_grad_delta = "
              << localAmmVsBaselineGradDelta << " | "
              << "majorized_boundary_candidates = "
              << majorizedBoundaryCandidates << " | "
              << "majorized_boundary_selected = "
              << majorizedBoundarySelected << " | "
              << "majorized_boundary_score_sum = "
              << majorizedBoundaryScoreSum << " | "
              << "majorized_boundary_selected_score_sum = "
              << majorizedBoundarySelectedScoreSum << " | "
              << "boundary_jacobi_attempted = "
              << boundaryJacobiAttempted << " | "
              << "boundary_jacobi_accepted = "
              << boundaryJacobiAccepted << " | "
              << "boundary_jacobi_rejected = "
              << boundaryJacobiRejected << " | "
              << "boundary_jacobi_blocks = " << boundaryJacobiBlocks
              << " | "
              << "boundary_jacobi_step_norm = "
              << boundaryJacobiStepNorm << " | "
              << "boundary_jacobi_cost_decrease = "
              << boundaryJacobiCostDecrease << " | "
              << "boundary_jacobi_ms = " << boundaryJacobiMs
              << " | "
              << "boundary_model_predictions = "
              << boundaryModelPredictions << " | "
              << "boundary_model_computed_step_norm = "
              << boundaryModelStepNorm << " | "
              << "boundary_model_computed_cost_decrease = "
              << boundaryModelCostDecrease << " | "
              << "boundary_model_compute_ms = " << boundaryModelMs
              << " | "
              << "boundary_model_merit_evaluations = "
              << boundaryModelMeritEvaluations << " | "
              << "boundary_model_merit_accepted = "
              << boundaryModelMeritAccepted << " | "
              << "boundary_model_merit_rejected = "
              << boundaryModelMeritRejected << " | "
              << "boundary_model_merit_decrease = "
              << boundaryModelMeritDecrease << " | "
              << "boundary_model_merit_grad_decrease = "
              << boundaryModelMeritGradDecrease << " | "
              << "boundary_model_merit_ms = "
              << boundaryModelMeritMs << " | "
              << "boundary_model_packet_payload_blocks = "
              << boundaryModelPacketPayloadBlocks << " | "
              << "boundary_model_packet_model_pose_blocks = "
              << boundaryModelPacketModelPoseBlocks << " | "
              << "boundary_model_packet_grad_norm = "
              << boundaryModelPacketGradNorm << " | "
              << "boundary_model_packet_stiffness_sum = "
              << boundaryModelPacketStiffnessSum << " | "
              << "boundary_model_packet_freshness_sum = "
              << boundaryModelPacketFreshnessSum << " | "
              << "boundary_model_packet_preconditioned_step_norm = "
              << boundaryModelPacketPreconditionedStepNorm << " | "
              << "boundary_model_packet_schur_sensitivity_sum = "
              << boundaryModelPacketSchurSensitivitySum << " | "
              << "boundary_model_packet_reduced_preconditioner_sum = "
              << boundaryModelPacketReducedPreconditionerSum << " | "
              << "boundary_model_packet_comm_mb = "
              << boundaryModelPacketCommMB << " | "
              << "boundary_response_attempted = "
              << boundaryResponseAttempted << " | "
              << "boundary_response_accepted = "
              << boundaryResponseAccepted << " | "
              << "boundary_response_rejected = "
              << boundaryResponseRejected << " | "
              << "boundary_response_blocks = "
              << boundaryResponseBlocks << " | "
              << "boundary_response_step_norm = "
              << boundaryResponseStepNorm << " | "
              << "boundary_response_grad_delta_norm = "
              << boundaryResponseGradDeltaNorm << " | "
              << "boundary_response_cost_decrease = "
              << boundaryResponseCostDecrease << " | "
              << "boundary_response_ms = "
              << boundaryResponseMs << " | "
              << "boundary_batch_response_attempted = "
              << boundaryBatchResponseAttempted << " | "
              << "boundary_batch_response_accepted = "
              << boundaryBatchResponseAccepted << " | "
              << "boundary_batch_response_rejected = "
              << boundaryBatchResponseRejected << " | "
              << "boundary_batch_response_blocks = "
              << boundaryBatchResponseBlocks << " | "
              << "boundary_batch_response_step_norm = "
              << boundaryBatchResponseStepNorm << " | "
              << "boundary_batch_response_grad_delta_norm = "
              << boundaryBatchResponseGradDeltaNorm << " | "
              << "boundary_batch_response_cost_decrease = "
              << boundaryBatchResponseCostDecrease << " | "
              << "boundary_batch_response_ms = "
              << boundaryBatchResponseMs << " | "
              << "boundary_batch_model_receivers = "
              << boundaryBatchModelReceivers << " | "
              << "boundary_batch_model_candidate_poses = "
              << boundaryBatchModelCandidatePoses << " | "
              << "boundary_batch_model_merit_evaluations = "
              << boundaryBatchModelMeritEvaluations << " | "
              << "boundary_batch_model_merit_accepted = "
              << boundaryBatchModelMeritAccepted << " | "
              << "boundary_batch_model_merit_rejected = "
              << boundaryBatchModelMeritRejected << " | "
              << "boundary_batch_model_merit_decrease = "
              << boundaryBatchModelMeritDecrease << " | "
              << "boundary_batch_model_merit_grad_decrease = "
              << boundaryBatchModelMeritGradDecrease << " | "
              << "boundary_batch_model_refine_ms = "
              << boundaryBatchModelRefineMs << " | "
              << "boundary_batch_model_budget_candidates = "
              << boundaryBatchModelBudgetCandidates << " | "
              << "boundary_batch_model_budget_selected = "
              << boundaryBatchModelBudgetSelected << " | "
              << "boundary_batch_model_budget_dropped = "
              << boundaryBatchModelBudgetDropped << " | "
              << "interface_state_payload_blocks = "
              << interfaceStatePayloadBlocks << " | "
              << "interface_state_grad_norm = "
              << interfaceStateGradNorm << " | "
              << "interface_state_stiffness_sum = "
              << interfaceStateStiffnessSum << " | "
              << "interface_state_freshness_sum = "
              << interfaceStateFreshnessSum << " | "
              << "interface_state_comm_mb = "
              << interfaceStateCommMB << " | "
              << "interface_state_compute_ms = "
              << interfaceStateComputeMs << " | "
              << "interface_delta_candidate_pose_blocks = "
              << interfaceDeltaCandidatePoseBlocks << " | "
              << "interface_delta_cache_miss_pose_blocks = "
              << interfaceDeltaCacheMissPoseBlocks << " | "
              << "interface_delta_rejected_pose_blocks = "
              << interfaceDeltaRejectedPoseBlocks << " | "
              << "interface_delta_codec = " << interfaceDeltaCodec << " | "
              << "interface_delta_sparse_pose_blocks = "
              << interfaceDeltaSparsePoseBlocks << " | "
              << "interface_delta_tangent_pose_blocks = "
              << interfaceDeltaTangentPoseBlocks << " | "
              << "interface_delta_full_pose_blocks = "
              << interfaceDeltaFullPoseBlocks << " | "
              << "interface_delta_delta_pose_blocks = "
              << interfaceDeltaDeltaPoseBlocks << " | "
              << "interface_delta_delta_entries = "
              << interfaceDeltaDeltaEntries << " | "
              << "interface_delta_comm_mb = "
              << interfaceDeltaCommMB << " | "
              << "interface_delta_reconstruction_error_sum = "
              << interfaceDeltaReconstructionErrorSum << " | "
              << "interface_delta_reconstruction_error_max = "
              << interfaceDeltaReconstructionErrorMax
              << std::endl;

    if (csv.is_open()) {
      csv << iter << "," << selectedRobotForLog << "," << cost << ","
          << RGradNorm << "," << iterCommPoses << "," << iterCommMB << ","
          << cumulativeCommPoses << "," << cumulativeCommMB << ","
          << relCostChange << "," << lossTolReached << "," << gradTolReached
          << "," << bothTolReached << "," << iterSequentialComputeMs << ","
          << iterParallelComputeMs << "," << cumulativeParallelComputeMs
          << "," << adaptiveRefineRobots << "," << adaptiveRefineCalls
          << "," << adaptiveRefineMs << "," << cumulativeAdaptiveRefineMs
          << "," << localAmmAttempted << "," << localAmmAccepted
          << "," << localAmmRejected << "," << localAmmCostDecrease
          << "," << localAmmGradDecrease << "," << localAmmStepNorm
          << "," << localAmmMs << "," << localAmmBaselineCompared
          << "," << localAmmSelected << "," << localAmmBaselineSelected
          << "," << localAmmVsBaselineCostDelta << ","
          << localAmmVsBaselineGradDelta << ","
          << cumulativeLocalAmmAttempted << ","
          << cumulativeLocalAmmAccepted << "," << cumulativeLocalAmmRejected
          << ","
          << cumulativeLocalAmmCostDecrease << ","
          << cumulativeLocalAmmGradDecrease << ","
          << cumulativeLocalAmmStepNorm << "," << cumulativeLocalAmmMs
          << "," << cumulativeLocalAmmBaselineCompared << ","
          << cumulativeLocalAmmSelected << ","
          << cumulativeLocalAmmBaselineSelected << ","
          << cumulativeLocalAmmVsBaselineCostDelta << ","
          << cumulativeLocalAmmVsBaselineGradDelta
          << "," << majorizedBoundaryCandidates << ","
          << majorizedBoundarySelected << "," << majorizedBoundaryScoreSum
          << "," << majorizedBoundarySelectedScoreSum
          << "," << boundaryJacobiAttempted << ","
          << boundaryJacobiAccepted << "," << boundaryJacobiRejected << ","
          << boundaryJacobiBlocks << "," << boundaryJacobiStepNorm << ","
          << boundaryJacobiCostDecrease << "," << boundaryJacobiMs << ","
          << cumulativeBoundaryJacobiAccepted << ","
          << cumulativeBoundaryJacobiMs << ","
          << boundaryModelPredictions << "," << boundaryModelStepNorm << ","
          << boundaryModelCostDecrease << "," << boundaryModelMs << ","
          << boundaryModelMeritEvaluations << ","
          << boundaryModelMeritAccepted << ","
          << boundaryModelMeritRejected << ","
          << boundaryModelMeritDecrease << ","
          << boundaryModelMeritGradDecrease << ","
          << boundaryModelMeritMs << ","
          << cumulativeBoundaryModelPredictions << ","
          << cumulativeBoundaryModelMs << ","
          << cumulativeBoundaryModelMeritEvaluations << ","
          << cumulativeBoundaryModelMeritAccepted << ","
          << cumulativeBoundaryModelMeritRejected << ","
          << cumulativeBoundaryModelMeritDecrease << ","
          << cumulativeBoundaryModelMeritGradDecrease << ","
          << cumulativeBoundaryModelMeritMs << ","
          << boundaryModelPacketPayloadBlocks << ","
          << boundaryModelPacketModelPoseBlocks << ","
          << boundaryModelPacketGradNorm << ","
          << boundaryModelPacketStiffnessSum << ","
          << boundaryModelPacketFreshnessSum << ","
          << boundaryModelPacketPreconditionedStepNorm << ","
          << boundaryModelPacketSchurSensitivitySum << ","
          << boundaryModelPacketReducedPreconditionerSum << ","
          << boundaryModelPacketCommMB << ","
          << cumulativeBoundaryModelPacketPayloadBlocks << ","
          << cumulativeBoundaryModelPacketModelPoseBlocks << ","
          << cumulativeBoundaryModelPacketGradNorm << ","
          << cumulativeBoundaryModelPacketStiffnessSum << ","
          << cumulativeBoundaryModelPacketFreshnessSum << ","
          << cumulativeBoundaryModelPacketPreconditionedStepNorm << ","
          << cumulativeBoundaryModelPacketSchurSensitivitySum << ","
          << cumulativeBoundaryModelPacketReducedPreconditionerSum << ","
          << cumulativeBoundaryModelPacketCommMB << ","
          << boundaryResponseAttempted << ","
          << boundaryResponseAccepted << ","
          << boundaryResponseRejected << ","
          << boundaryResponseBlocks << ","
          << boundaryResponseStepNorm << ","
          << boundaryResponseGradDeltaNorm << ","
          << boundaryResponseCostDecrease << ","
          << boundaryResponseMs << ","
          << cumulativeBoundaryResponseAccepted << ","
          << cumulativeBoundaryResponseBlocks << ","
          << cumulativeBoundaryResponseStepNorm << ","
          << cumulativeBoundaryResponseGradDeltaNorm << ","
          << cumulativeBoundaryResponseCostDecrease << ","
          << cumulativeBoundaryResponseMs << ","
          << boundaryBatchResponseAttempted << ","
          << boundaryBatchResponseAccepted << ","
          << boundaryBatchResponseRejected << ","
          << boundaryBatchResponseBlocks << ","
          << boundaryBatchResponseStepNorm << ","
          << boundaryBatchResponseGradDeltaNorm << ","
          << boundaryBatchResponseCostDecrease << ","
          << boundaryBatchResponseMs << ","
          << cumulativeBoundaryBatchResponseAccepted << ","
          << cumulativeBoundaryBatchResponseBlocks << ","
          << cumulativeBoundaryBatchResponseStepNorm << ","
          << cumulativeBoundaryBatchResponseGradDeltaNorm << ","
          << cumulativeBoundaryBatchResponseCostDecrease << ","
          << cumulativeBoundaryBatchResponseMs << ","
          << boundaryBatchModelReceivers << ","
          << boundaryBatchModelCandidatePoses << ","
          << boundaryBatchModelMeritEvaluations << ","
          << boundaryBatchModelMeritAccepted << ","
          << boundaryBatchModelMeritRejected << ","
          << boundaryBatchModelMeritDecrease << ","
          << boundaryBatchModelMeritGradDecrease << ","
          << boundaryBatchModelRefineMs << ","
          << boundaryBatchModelBudgetCandidates << ","
          << boundaryBatchModelBudgetSelected << ","
          << boundaryBatchModelBudgetDropped << ","
          << cumulativeBoundaryBatchModelReceivers << ","
          << cumulativeBoundaryBatchModelCandidatePoses << ","
          << cumulativeBoundaryBatchModelMeritEvaluations << ","
          << cumulativeBoundaryBatchModelMeritAccepted << ","
          << cumulativeBoundaryBatchModelMeritRejected << ","
          << cumulativeBoundaryBatchModelMeritDecrease << ","
          << cumulativeBoundaryBatchModelMeritGradDecrease << ","
          << cumulativeBoundaryBatchModelRefineMs << ","
          << cumulativeBoundaryBatchModelBudgetCandidates << ","
          << cumulativeBoundaryBatchModelBudgetSelected << ","
          << cumulativeBoundaryBatchModelBudgetDropped << ","
          << interfaceStatePayloadBlocks << ","
          << interfaceStateGradNorm << ","
          << interfaceStateStiffnessSum << ","
          << interfaceStateFreshnessSum << ","
          << interfaceStateCommMB << ","
          << interfaceStateComputeMs << ","
          << cumulativeInterfaceStatePayloadBlocks << ","
          << cumulativeInterfaceStateCommMB << ","
          << cumulativeInterfaceStateComputeMs << ","
          << interfaceDeltaCandidatePoseBlocks << ","
          << interfaceDeltaCacheMissPoseBlocks << ","
          << interfaceDeltaRejectedPoseBlocks << ","
          << interfaceDeltaSparsePoseBlocks << ","
          << interfaceDeltaTangentPoseBlocks << ","
          << interfaceDeltaFullPoseBlocks << ","
          << interfaceDeltaDeltaPoseBlocks << ","
          << interfaceDeltaDeltaEntries << ","
          << interfaceDeltaCommMB << ","
          << interfaceDeltaReconstructionErrorSum << ","
          << interfaceDeltaReconstructionErrorMax << ","
          << cumulativeInterfaceDeltaCandidatePoseBlocks << ","
          << cumulativeInterfaceDeltaCacheMissPoseBlocks << ","
          << cumulativeInterfaceDeltaRejectedPoseBlocks << ","
          << cumulativeInterfaceDeltaSparsePoseBlocks << ","
          << cumulativeInterfaceDeltaTangentPoseBlocks << ","
          << cumulativeInterfaceDeltaFullPoseBlocks << ","
          << cumulativeInterfaceDeltaDeltaPoseBlocks << ","
          << cumulativeInterfaceDeltaDeltaEntries << ","
          << cumulativeInterfaceDeltaCommMB
          << "\n";
    }

    if (convergenceIter < 0 && bothTolReached) {
      convergenceIter = static_cast<int>(iter);
      convergenceCommPoses = cumulativeCommPoses;
      convergenceCommMB = cumulativeCommMB;
      std::cout << "Convergence marker reached at iter " << convergenceIter
                << " | comm_poses = " << convergenceCommPoses
                << " | comm_mb = " << convergenceCommMB << std::endl;
    }

    // Exit only when both quality criteria are met.
    if (bothTolReached) {
      break;
    }
    prevCost = cost;

    if (updateMode != "decentralized" &&
        updateMode != "decentralized_event" &&
        updateMode != "decentralized_event_colored" &&
        updateMode != "decentralized_residual" &&
        updateMode != "decentralized_residual_graph_colored" &&
        updateMode != "decentralized_residual_colored_sweep" &&
        updateMode != "decentralized_residual_colored_predictive" &&
        updateMode != "decentralized_residual_colored_conflict" &&
        updateMode != "decentralized_residual_colored_conflict_refine" &&
        updateMode != "decentralized_residual_colored_conflict_sweep" &&
        updateMode != "decentralized_residual_colored_winner_refine" &&
        updateMode != "decentralized_residual_colored_feedback_refine" &&
        updateMode != "decentralized_residual_colored_curvature_feedback" &&
        updateMode != "decentralized_residual_colored_schur_feedback" &&
        updateMode != "decentralized_residual_colored_schur_late_feedback" &&
        updateMode != "decentralized_boundary_jacobi_correction" &&
        !asyncSchurMode &&
        updateMode != "decentralized_residual_colored") {
      PGOAgent *selectedRobotPtr = agents[selectedRobot];
      // Select next robot with largest gradient norm. This branch is only for
      // the legacy selected-robot baseline; decentralized mode never uses this.
      std::vector<unsigned> neighbors = selectedRobotPtr->getNeighbors();
      if (neighbors.empty()) {
        selectedRobot = selectedRobotPtr->getID();
      } else {
        std::vector<double> gradNorms;
        for (size_t robot = 0; robot < (unsigned) num_robots; ++robot) {
          unsigned startIdx = robot * num_poses_per_robot;
          unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
          if (robot == (unsigned) num_robots - 1) endIdx = n;
          Matrix RGradRobot = RGrad.block(
              0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1));
          gradNorms.push_back(RGradRobot.norm());
        }
        selectedRobot =
            std::max_element(gradNorms.begin(), gradNorms.end()) -
            gradNorms.begin();
      }
    }

    // Share global anchor for rounding
    Matrix M;
    agents[0]->getSharedPose(0, M);
    for (auto agentPtr : agents) {
      agentPtr->setGlobalAnchor(M);
    }
  }

  if (!poseOutputPath.empty()) {
    if (writePoseRows(poseOutputPath, agents, num_robots, n,
                      num_poses_per_robot, d)) {
      cout << "Saved final poses to " << poseOutputPath << endl;
    }
  }

  std::cout << "SUMMARY"
            << " configured_update_mode=" << configuredUpdateMode
            << " final_effective_update_mode=" << updateMode
            << " hybrid_warmup_rounds=" << hybridWarmupRounds
            << " final_iter=" << finalIter
            << " final_cost=" << finalCost
            << " final_gradnorm=" << finalGradNorm
            << " total_comm_poses=" << cumulativeCommPoses
            << " total_comm_mb=" << cumulativeCommMB
            << " loss_tol_iter=" << lossTolIter
            << " loss_tol_comm_poses=" << lossTolCommPoses
            << " loss_tol_comm_mb=" << lossTolCommMB
            << " grad_tol_iter=" << gradTolIter
            << " grad_tol_comm_poses=" << gradTolCommPoses
            << " grad_tol_comm_mb=" << gradTolCommMB
            << " convergence_iter=" << convergenceIter
            << " convergence_comm_poses=" << convergenceCommPoses
            << " convergence_comm_mb=" << convergenceCommMB
            << " cumulative_parallel_compute_ms="
            << cumulativeParallelComputeMs
            << " adaptive_total_refine_robots="
            << cumulativeAdaptiveRefineRobots
            << " adaptive_total_refine_calls="
            << cumulativeAdaptiveRefineCalls
            << " adaptive_total_refine_ms="
            << cumulativeAdaptiveRefineMs
            << " local_amm_total_attempted="
            << cumulativeLocalAmmAttempted
            << " local_amm_total_accepted="
            << cumulativeLocalAmmAccepted
            << " local_amm_total_rejected="
            << cumulativeLocalAmmRejected
            << " local_amm_total_cost_decrease="
            << cumulativeLocalAmmCostDecrease
            << " local_amm_total_grad_decrease="
            << cumulativeLocalAmmGradDecrease
            << " local_amm_total_step_norm="
            << cumulativeLocalAmmStepNorm
            << " local_amm_total_ms="
            << cumulativeLocalAmmMs
            << " local_amm_total_baseline_compared="
            << cumulativeLocalAmmBaselineCompared
            << " local_amm_total_selected="
            << cumulativeLocalAmmSelected
            << " local_amm_total_baseline_selected="
            << cumulativeLocalAmmBaselineSelected
            << " local_amm_total_vs_baseline_cost_delta="
            << cumulativeLocalAmmVsBaselineCostDelta
            << " local_amm_total_vs_baseline_grad_delta="
            << cumulativeLocalAmmVsBaselineGradDelta
            << " majorized_boundary_total_candidates="
            << cumulativeMajorizedBoundaryCandidates
            << " majorized_boundary_total_selected="
            << cumulativeMajorizedBoundarySelected
            << " majorized_boundary_total_score="
            << cumulativeMajorizedBoundaryScore
            << " majorized_boundary_total_selected_score="
            << cumulativeMajorizedBoundarySelectedScore
            << " boundary_jacobi_total_attempted="
            << cumulativeBoundaryJacobiAttempted
            << " boundary_jacobi_total_accepted="
            << cumulativeBoundaryJacobiAccepted
            << " boundary_jacobi_total_rejected="
            << cumulativeBoundaryJacobiRejected
            << " boundary_jacobi_total_blocks="
            << cumulativeBoundaryJacobiBlocks
            << " boundary_jacobi_total_step_norm="
            << cumulativeBoundaryJacobiStepNorm
            << " boundary_jacobi_total_cost_decrease="
            << cumulativeBoundaryJacobiCostDecrease
            << " boundary_jacobi_total_ms="
            << cumulativeBoundaryJacobiMs
            << " boundary_model_total_predictions="
            << cumulativeBoundaryModelPredictions
            << " boundary_model_total_computed_step_norm="
            << cumulativeBoundaryModelStepNorm
            << " boundary_model_total_computed_cost_decrease="
            << cumulativeBoundaryModelCostDecrease
            << " boundary_model_total_compute_ms="
            << cumulativeBoundaryModelMs
            << " boundary_model_total_merit_evaluations="
            << cumulativeBoundaryModelMeritEvaluations
            << " boundary_model_total_merit_accepted="
            << cumulativeBoundaryModelMeritAccepted
            << " boundary_model_total_merit_rejected="
            << cumulativeBoundaryModelMeritRejected
            << " boundary_model_total_merit_decrease="
            << cumulativeBoundaryModelMeritDecrease
            << " boundary_model_total_merit_grad_decrease="
            << cumulativeBoundaryModelMeritGradDecrease
            << " boundary_model_total_merit_ms="
            << cumulativeBoundaryModelMeritMs
            << " boundary_model_packet_total_payload_blocks="
            << cumulativeBoundaryModelPacketPayloadBlocks
            << " boundary_model_packet_total_model_pose_blocks="
            << cumulativeBoundaryModelPacketModelPoseBlocks
            << " boundary_model_packet_total_grad_norm="
            << cumulativeBoundaryModelPacketGradNorm
            << " boundary_model_packet_total_stiffness_sum="
            << cumulativeBoundaryModelPacketStiffnessSum
            << " boundary_model_packet_total_freshness_sum="
            << cumulativeBoundaryModelPacketFreshnessSum
            << " boundary_model_packet_total_preconditioned_step_norm="
            << cumulativeBoundaryModelPacketPreconditionedStepNorm
            << " boundary_model_packet_total_schur_sensitivity_sum="
            << cumulativeBoundaryModelPacketSchurSensitivitySum
            << " boundary_model_packet_total_reduced_preconditioner_sum="
            << cumulativeBoundaryModelPacketReducedPreconditionerSum
            << " boundary_model_packet_total_comm_mb="
            << cumulativeBoundaryModelPacketCommMB
            << " boundary_model_packet_payload_bytes_per_block="
            << boundaryModelPacketPayloadBytes
            << " boundary_response_total_attempted="
            << cumulativeBoundaryResponseAttempted
            << " boundary_response_total_accepted="
            << cumulativeBoundaryResponseAccepted
            << " boundary_response_total_rejected="
            << cumulativeBoundaryResponseRejected
            << " boundary_response_total_blocks="
            << cumulativeBoundaryResponseBlocks
            << " boundary_response_total_step_norm="
            << cumulativeBoundaryResponseStepNorm
            << " boundary_response_total_grad_delta_norm="
            << cumulativeBoundaryResponseGradDeltaNorm
            << " boundary_response_total_cost_decrease="
            << cumulativeBoundaryResponseCostDecrease
            << " boundary_response_total_ms="
            << cumulativeBoundaryResponseMs
            << " boundary_batch_response_total_attempted="
            << cumulativeBoundaryBatchResponseAttempted
            << " boundary_batch_response_total_accepted="
            << cumulativeBoundaryBatchResponseAccepted
            << " boundary_batch_response_total_rejected="
            << cumulativeBoundaryBatchResponseRejected
            << " boundary_batch_response_total_blocks="
            << cumulativeBoundaryBatchResponseBlocks
            << " boundary_batch_response_total_step_norm="
            << cumulativeBoundaryBatchResponseStepNorm
            << " boundary_batch_response_total_grad_delta_norm="
            << cumulativeBoundaryBatchResponseGradDeltaNorm
            << " boundary_batch_response_total_cost_decrease="
            << cumulativeBoundaryBatchResponseCostDecrease
            << " boundary_batch_response_total_ms="
            << cumulativeBoundaryBatchResponseMs
            << " boundary_batch_model_total_receivers="
            << cumulativeBoundaryBatchModelReceivers
            << " boundary_batch_model_total_candidate_poses="
            << cumulativeBoundaryBatchModelCandidatePoses
            << " boundary_batch_model_total_merit_evaluations="
            << cumulativeBoundaryBatchModelMeritEvaluations
            << " boundary_batch_model_total_merit_accepted="
            << cumulativeBoundaryBatchModelMeritAccepted
            << " boundary_batch_model_total_merit_rejected="
            << cumulativeBoundaryBatchModelMeritRejected
            << " boundary_batch_model_total_merit_decrease="
            << cumulativeBoundaryBatchModelMeritDecrease
            << " boundary_batch_model_total_merit_grad_decrease="
            << cumulativeBoundaryBatchModelMeritGradDecrease
            << " boundary_batch_model_total_refine_ms="
            << cumulativeBoundaryBatchModelRefineMs
            << " boundary_batch_model_total_budget_candidates="
            << cumulativeBoundaryBatchModelBudgetCandidates
            << " boundary_batch_model_total_budget_selected="
            << cumulativeBoundaryBatchModelBudgetSelected
            << " boundary_batch_model_total_budget_dropped="
            << cumulativeBoundaryBatchModelBudgetDropped
            << " interface_state_total_payload_blocks="
            << cumulativeInterfaceStatePayloadBlocks
            << " interface_state_payload_bytes_per_block="
            << interfaceStatePayloadBytes
            << " interface_state_total_comm_mb="
            << cumulativeInterfaceStateCommMB
            << " interface_state_total_compute_ms="
            << cumulativeInterfaceStateComputeMs
            << " interface_delta_total_candidate_pose_blocks="
            << cumulativeInterfaceDeltaCandidatePoseBlocks
            << " interface_delta_total_cache_miss_pose_blocks="
            << cumulativeInterfaceDeltaCacheMissPoseBlocks
            << " interface_delta_total_rejected_pose_blocks="
            << cumulativeInterfaceDeltaRejectedPoseBlocks
            << " interface_delta_total_sparse_pose_blocks="
            << cumulativeInterfaceDeltaSparsePoseBlocks
            << " interface_delta_total_tangent_pose_blocks="
            << cumulativeInterfaceDeltaTangentPoseBlocks
            << " interface_delta_total_full_pose_blocks="
            << cumulativeInterfaceDeltaFullPoseBlocks
            << " interface_delta_total_delta_pose_blocks="
            << cumulativeInterfaceDeltaDeltaPoseBlocks
            << " interface_delta_total_delta_entries="
            << cumulativeInterfaceDeltaDeltaEntries
            << " interface_delta_total_comm_mb="
            << cumulativeInterfaceDeltaCommMB
            << std::endl;

  for (auto agentPtr : agents) {
    agentPtr->reset();
  }

  return 0;
}

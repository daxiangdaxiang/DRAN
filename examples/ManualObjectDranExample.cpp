#include <DPGO/ManualObjectDran.h>

#include <DPGO/DPGO_utils.h>
#include <DPGO/ObjectPGOData.h>

#include <Eigen/Geometry>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool parseBool(const std::string &value) {
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "on";
}

DPGO::ManualObjectDranTopology parseTopology(const std::string &value) {
  if (value == "ring") {
    return DPGO::ManualObjectDranTopology::Ring;
  }
  if (value == "complete") {
    return DPGO::ManualObjectDranTopology::Complete;
  }
  throw std::invalid_argument("Unknown topology: " + value);
}

DPGO::ManualObjectDranCommunicationPolicy parseCommunicationPolicy(
    const std::string &value) {
  if (value == "all" || value == "full") {
    return DPGO::ManualObjectDranCommunicationPolicy::All;
  }
  if (value == "triggered" || value == "event_triggered") {
    return DPGO::ManualObjectDranCommunicationPolicy::Triggered;
  }
  throw std::invalid_argument("Unknown communication policy: " + value);
}

DPGO::ManualObjectDranObjectInitialization parseObjectInitialization(
    const std::string &value) {
  if (value == "centralized_chordal" || value == "centralized") {
    return DPGO::ManualObjectDranObjectInitialization::CentralizedChordal;
  }
  if (value == "neighbor_average" || value == "neighbor") {
    return DPGO::ManualObjectDranObjectInitialization::NeighborAverage;
  }
  throw std::invalid_argument("Unknown object initialization: " + value);
}

DPGO::ManualObjectDranObjectInitializationAnchorMode
parseObjectInitializationAnchorMode(const std::string &value) {
  if (value == "constant" || value == "uniform") {
    return DPGO::ManualObjectDranObjectInitializationAnchorMode::Constant;
  }
  if (value == "information" || value == "info" ||
      value == "measurement_information") {
    return DPGO::ManualObjectDranObjectInitializationAnchorMode::Information;
  }
  throw std::invalid_argument("Unknown object initialization anchor mode: " +
                              value);
}

DPGO::ManualObjectDranObjectInterfaceMode parseObjectInterfaceMode(
    const std::string &value) {
  if (value == "copy_baseline" || value == "baseline" || value == "copy") {
    return DPGO::ManualObjectDranObjectInterfaceMode::CopyBaseline;
  }
  if (value == "reduced_schur_response" || value == "schur_response" ||
      value == "object_schur_response") {
    return DPGO::ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  }
  throw std::invalid_argument("Unknown object interface mode: " + value);
}

DPGO::ManualObjectDranObjectInterfaceResponseStrategy
parseObjectInterfaceResponseStrategy(const std::string &value) {
  if (value == "serial") {
    return DPGO::ManualObjectDranObjectInterfaceResponseStrategy::Serial;
  }
  if (value == "batch" || value == "cached_batch") {
    return DPGO::ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  }
  throw std::invalid_argument("Unknown object interface response strategy: " +
                              value);
}

DPGO::ManualObjectDranObjectInterfaceResponseTrigger
parseObjectInterfaceResponseTrigger(const std::string &value) {
  if (value == "periodic") {
    return DPGO::ManualObjectDranObjectInterfaceResponseTrigger::Periodic;
  }
  if (value == "predicted_decrease" || value == "adaptive" ||
      value == "model_decrease") {
    return DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        PredictedDecrease;
  }
  if (value == "local_innovation" || value == "innovation") {
    return DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        LocalInnovation;
  }
  if (value == "accumulated_innovation" || value == "innovation_bucket" ||
      value == "accumulated") {
    return DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        AccumulatedInnovation;
  }
  throw std::invalid_argument("Unknown object interface response trigger: " +
                              value);
}

std::string topologyName(DPGO::ManualObjectDranTopology topology) {
  switch (topology) {
    case DPGO::ManualObjectDranTopology::Ring:
      return "ring";
    case DPGO::ManualObjectDranTopology::Complete:
      return "complete";
  }
  return "ring";
}

std::string communicationPolicyName(
    DPGO::ManualObjectDranCommunicationPolicy policy) {
  switch (policy) {
    case DPGO::ManualObjectDranCommunicationPolicy::All:
      return "all";
    case DPGO::ManualObjectDranCommunicationPolicy::Triggered:
      return "triggered";
  }
  return "triggered";
}

std::string objectInitializationName(
    DPGO::ManualObjectDranObjectInitialization initialization) {
  switch (initialization) {
    case DPGO::ManualObjectDranObjectInitialization::CentralizedChordal:
      return "centralized_chordal";
    case DPGO::ManualObjectDranObjectInitialization::NeighborAverage:
      return "neighbor_average";
  }
  return "centralized_chordal";
}

std::string objectInitializationAnchorModeName(
    DPGO::ManualObjectDranObjectInitializationAnchorMode mode) {
  switch (mode) {
    case DPGO::ManualObjectDranObjectInitializationAnchorMode::Constant:
      return "constant";
    case DPGO::ManualObjectDranObjectInitializationAnchorMode::Information:
      return "information";
  }
  return "constant";
}

std::string objectInterfaceModeName(
    DPGO::ManualObjectDranObjectInterfaceMode mode) {
  switch (mode) {
    case DPGO::ManualObjectDranObjectInterfaceMode::CopyBaseline:
      return "copy_baseline";
    case DPGO::ManualObjectDranObjectInterfaceMode::ReducedSchurResponse:
      return "reduced_schur_response";
  }
  return "copy_baseline";
}

std::string objectInterfaceResponseStrategyName(
    DPGO::ManualObjectDranObjectInterfaceResponseStrategy strategy) {
  switch (strategy) {
    case DPGO::ManualObjectDranObjectInterfaceResponseStrategy::Serial:
      return "serial";
    case DPGO::ManualObjectDranObjectInterfaceResponseStrategy::Batch:
      return "batch";
  }
  return "serial";
}

std::string objectInterfaceResponseTriggerName(
    DPGO::ManualObjectDranObjectInterfaceResponseTrigger trigger) {
  switch (trigger) {
    case DPGO::ManualObjectDranObjectInterfaceResponseTrigger::Periodic:
      return "periodic";
    case DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        PredictedDecrease:
      return "predicted_decrease";
    case DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        LocalInnovation:
      return "local_innovation";
    case DPGO::ManualObjectDranObjectInterfaceResponseTrigger::
        AccumulatedInnovation:
      return "accumulated_innovation";
  }
  return "periodic";
}

void printUsage(const char *program) {
  std::cout << "Usage: " << program
            << " --data_dir path --num_robots N [options]\n"
            << "Options:\n"
            << "  --num_objects N\n"
            << "  --iters N\n"
            << "  --beta value\n"
            << "  --topology ring|complete\n"
            << "  --ring_hops N\n"
            << "  --topology_file path\n"
            << "  --topology_weight_mode unit|matrix\n"
            << "  --communication_policy all|triggered\n"
            << "  --object_initialization centralized_chordal|neighbor_average\n"
            << "  --object_initialization_anchor_mode constant|information\n"
            << "  --object_initialization_consensus_rounds N\n"
            << "  --object_initialization_observed_anchor_weight value\n"
            << "  --object_initialization_relay_anchor_weight value\n"
            << "  --object_interface_mode copy_baseline|reduced_schur_response\n"
            << "  --object_interface_response_strategy serial|batch\n"
            << "  --object_interface_response_trigger periodic|predicted_decrease|local_innovation|accumulated_innovation\n"
            << "  --object_interface_response_period N\n"
            << "  --object_interface_min_predicted_decrease value\n"
            << "  --object_interface_min_innovation_score value\n"
            << "  --object_interface_schur_damping value\n"
            << "  --object_interface_step_gain value\n"
            << "  --object_interface_max_objects_per_robot N\n"
            << "  --object_interface_max_private_cols N\n"
            << "  --object_interface_max_block_step_norm value\n"
            << "  --object_interface_require_decrease true|false\n"
            << "  --object_pose_tol value\n"
            << "  --object_max_age N\n"
            << "  --translation_prox_weight value\n"
            << "  --project_to_se_after_local_solve true|false\n"
            << "  --coupled_translation_trust_region true|false\n"
            << "  --translation_elimination_prox_weight value\n"
            << "  --local_solver reduced_rotation|manual_full|full_equiv_hybrid\n"
            << "  --full_solver_outer_iterations N\n"
            << "  --local_max_iterations N\n"
            << "  --local_max_tcg_iterations N\n"
            << "  --local_grad_norm_tol value\n"
            << "  --trust_region_initial_radius value\n"
            << "  --reduced_rotation_preconditioner none|jacobi|schur_jacobi|cholesky|portfolio\n"
            << "    portfolio currently evaluates none|jacobi only; use schur_jacobi explicitly\n"
            << "  --output_dir path\n"
            << "  --save true|false\n";
}

void writeIterationCsv(const std::filesystem::path &path,
                       const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open CSV output: " + path.string());
  }
  out << "iter,time,global_cost,measurement_cost,consensus_cost,"
         "augmented_cost,gradient,active_consensus_pairs,comm_pose_count,"
         "iter_comm_mb,cumulative_comm_pose_count,cumulative_comm_mb,"
         "known_object_copies,relay_object_copies,observed_object_copies,"
         "relay_only_object_copies,initialized_from_neighbor_object_copies,"
         "stale_object_copies,stale_object_comm_triggers\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranIterationSummary &row :
       result.iterations) {
    out << row.iter << "," << row.time << "," << row.measurementCost << ","
        << row.measurementCost << "," << row.consensusCost << ","
        << row.measurementCost + row.consensusCost << "," << row.gradient
        << "," << row.activeConsensusPairs << "," << row.commPoseCount << ","
        << row.iterCommMb << "," << row.cumulativeCommPoseCount << ","
        << row.cumulativeCommMb << "," << row.knownObjectCopies << ","
        << row.relayObjectCopies << "," << row.observedObjectCopies << ","
        << row.relayOnlyObjectCopies << ","
        << row.initializedFromNeighborObjectCopies << ","
        << row.staleObjectCopies << ","
        << row.staleObjectCommTriggers << "\n";
  }
}

void writeObjectInterfaceResponseCsv(
    const std::filesystem::path &path,
    const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "Could not open object interface response CSV output: " +
        path.string());
  }
  out << "iter,scheduled,active_robots,candidate_object_blocks,"
         "accepted_object_blocks,rejected_object_blocks,"
         "batch_response_robots,accepted_robot_batches,"
         "rejected_robot_batches,trigger_candidate_robots,"
         "triggered_robots,skipped_trigger_robots,model_cost_before,"
         "model_cost_after,interface_innovation_score,"
         "accumulated_innovation_score,predicted_model_decrease,"
         "accepted_model_decrease,step_norm\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranObjectInterfaceResponseSummary &row :
       result.objectInterfaceResponseSummaries) {
    out << row.iter << "," << (row.scheduled ? 1 : 0) << ","
        << row.activeRobots << ","
        << row.candidateObjectBlocks << "," << row.acceptedObjectBlocks
        << "," << row.rejectedObjectBlocks << ","
        << row.batchResponseRobots << "," << row.acceptedRobotBatches
        << "," << row.rejectedRobotBatches << ","
        << row.triggerCandidateRobots << "," << row.triggeredRobots
        << "," << row.skippedTriggerRobots << "," << row.modelCostBefore
        << "," << row.modelCostAfter << "," << row.interfaceInnovationScore
        << "," << row.accumulatedInnovationScore
        << "," << row.predictedModelDecrease
        << "," << row.acceptedModelDecrease
        << "," << row.stepNorm << "\n";
  }
}

void writeObjectInterfaceCsv(const std::filesystem::path &path,
                             const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open object interface CSV output: " +
                             path.string());
  }
  out << "iter,robot,object_id,observed,relay_only,age,residual_norm,"
         "gradient_norm,stiffness,neighbor_disagreement,selected_for_send\n";
  out << std::setprecision(20);
  for (const DPGO::ManualObjectDranObjectInterfaceDiagnostic &row :
       result.objectInterfaceDiagnostics) {
    out << row.iter << "," << row.robotId << "," << row.objectId << ","
        << (row.observed ? 1 : 0) << "," << (row.relayOnly ? 1 : 0)
        << "," << row.age << "," << row.residualNorm << ","
        << row.gradientNorm << "," << row.stiffness << ","
        << row.neighborDisagreement << ","
        << (row.selectedForSend ? 1 : 0) << "\n";
  }
}

void writeObjectAwareEstimates(const std::filesystem::path &path,
                               const DPGO::ObjectPGODirectoryData &dataset,
                               const DPGO::ManualObjectDranRunResult &result) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Could not open estimate output: " +
                             path.string());
  }
  if (dataset.dimension != 3) {
    throw std::runtime_error(
        "manual_object_dran object_poses.txt output currently expects SE(3)");
  }
  out << std::setprecision(20);
  out << "type robot_id local_vertex_id object_id x y z qx qy qz qw\n";
  for (std::size_t robotId = 0; robotId < result.finalEstimates.size();
       ++robotId) {
    const DPGO::Matrix &state = result.finalEstimates[robotId];
    const DPGO::ObjectPGORobotData &robot = dataset.robots[robotId];
    const unsigned d = static_cast<unsigned>(state.rows());
    const unsigned blockWidth = d + 1;
    for (std::size_t localId = 0; localId < robot.trajectoryVertexIds.size();
         ++localId) {
      const Eigen::Matrix3d R = DPGO::projectToRotationGroup(
          state.block(0, localId * blockWidth, d, d));
      Eigen::Quaterniond q(R);
      q.normalize();
      const DPGO::Matrix t =
          state.block(0, localId * blockWidth + d, d, 1);
      out << "trajectory " << robotId << ' '
          << robot.trajectoryVertexIds[localId] << " -1 " << t(0) << ' '
          << t(1) << ' ' << t(2) << ' ' << q.x() << ' ' << q.y() << ' '
          << q.z() << ' ' << q.w() << '\n';
    }
    for (std::size_t objectId = 0; objectId < dataset.numObjects; ++objectId) {
      const std::size_t poseIndex = robot.trajectoryVertexIds.size() + objectId;
      const std::size_t localVertexId =
          objectId < robot.objectLocalVertexIds.size()
              ? robot.objectLocalVertexIds[objectId]
              : robot.objectStart + objectId;
      const Eigen::Matrix3d R = DPGO::projectToRotationGroup(
          state.block(0, poseIndex * blockWidth, d, d));
      Eigen::Quaterniond q(R);
      q.normalize();
      const DPGO::Matrix t =
          state.block(0, poseIndex * blockWidth + d, d, 1);
      out << "object " << robotId << ' ' << localVertexId << ' ' << objectId
          << ' ' << t(0) << ' ' << t(1) << ' ' << t(2) << ' ' << q.x()
          << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string dataDir;
  std::string outputDir = ".";
  bool save = true;
  DPGO::ManualObjectDranOptions options;

  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto requireValue = [&](const std::string &name) -> std::string {
        if (i + 1 >= argc) {
          throw std::invalid_argument("Missing value for " + name);
        }
        return argv[++i];
      };

      if (arg == "--help" || arg == "-h") {
        printUsage(argv[0]);
        return 0;
      } else if (arg == "--data_dir" || arg == "--dataset") {
        dataDir = requireValue(arg);
      } else if (arg == "--num_robots" || arg == "--num_nodes") {
        options.numRobots =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--num_objects") {
        options.numObjects =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--iters") {
        options.maxIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--beta") {
        options.beta = std::stod(requireValue(arg));
      } else if (arg == "--topology") {
        options.topology = parseTopology(requireValue(arg));
      } else if (arg == "--ring_hops") {
        options.ringHops =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--topology_file") {
        options.topologyFile = requireValue(arg);
      } else if (arg == "--topology_weight_mode") {
        options.topologyWeightMode = requireValue(arg);
      } else if (arg == "--communication_policy") {
        options.communicationPolicy =
            parseCommunicationPolicy(requireValue(arg));
      } else if (arg == "--object_initialization") {
        options.objectInitialization =
            parseObjectInitialization(requireValue(arg));
      } else if (arg == "--object_initialization_anchor_mode") {
        options.objectInitializationAnchorMode =
            parseObjectInitializationAnchorMode(requireValue(arg));
      } else if (arg == "--object_initialization_consensus_rounds") {
        options.objectInitializationConsensusRounds =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_initialization_observed_anchor_weight") {
        options.objectInitializationObservedAnchorWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--object_initialization_relay_anchor_weight") {
        options.objectInitializationRelayAnchorWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_mode") {
        options.objectInterfaceMode =
            parseObjectInterfaceMode(requireValue(arg));
      } else if (arg == "--object_interface_response_strategy") {
        options.objectInterfaceResponseStrategy =
            parseObjectInterfaceResponseStrategy(requireValue(arg));
      } else if (arg == "--object_interface_response_trigger") {
        options.objectInterfaceResponseTrigger =
            parseObjectInterfaceResponseTrigger(requireValue(arg));
      } else if (arg == "--object_interface_response_period") {
        options.objectInterfaceResponsePeriod =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_interface_min_predicted_decrease") {
        options.objectInterfaceMinPredictedDecrease =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_min_innovation_score") {
        options.objectInterfaceMinInnovationScore =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_schur_damping") {
        options.objectInterfaceSchurDamping = std::stod(requireValue(arg));
      } else if (arg == "--object_interface_step_gain") {
        options.objectInterfaceStepGain = std::stod(requireValue(arg));
      } else if (arg == "--object_interface_max_objects_per_robot") {
        options.objectInterfaceMaxObjectsPerRobot =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_interface_max_private_cols") {
        options.objectInterfaceMaxPrivateCols =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--object_interface_max_block_step_norm") {
        options.objectInterfaceMaxBlockStepNorm =
            std::stod(requireValue(arg));
      } else if (arg == "--object_interface_require_decrease") {
        options.objectInterfaceRequireDecrease = parseBool(requireValue(arg));
      } else if (arg == "--object_pose_tol") {
        options.objectPoseTolerance = std::stod(requireValue(arg));
      } else if (arg == "--object_max_age") {
        options.objectMaxAge =
            static_cast<std::size_t>(std::stoul(requireValue(arg)));
      } else if (arg == "--translation_prox_weight") {
        options.translationProxWeight = std::stod(requireValue(arg));
      } else if (arg == "--project_to_se_after_local_solve") {
        options.projectToSEAfterLocalSolve = parseBool(requireValue(arg));
      } else if (arg == "--coupled_translation_trust_region") {
        options.coupledTranslationTrustRegion =
            parseBool(requireValue(arg));
      } else if (arg == "--translation_elimination_prox_weight") {
        options.translationEliminationProxWeight =
            std::stod(requireValue(arg));
      } else if (arg == "--local_solver") {
        options.localSolver =
            DPGO::parseManualDpgoMmLocalSolver(requireValue(arg));
      } else if (arg == "--full_solver_outer_iterations") {
        options.fullSolverOuterIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_max_iterations") {
        options.localTrustRegionIterations =
            static_cast<unsigned>(std::stoul(requireValue(arg)));
      } else if (arg == "--local_max_tcg_iterations") {
        options.localTrustRegionMaxInnerIterations =
            std::stoi(requireValue(arg));
      } else if (arg == "--local_grad_norm_tol") {
        options.localTrustRegionTolerance = std::stod(requireValue(arg));
      } else if (arg == "--trust_region_initial_radius") {
        options.localTrustRegionInitialRadius = std::stod(requireValue(arg));
      } else if (arg == "--reduced_rotation_preconditioner") {
        options.reducedRotationPreconditioner =
            DPGO::parseManualDpgoMmReducedRotationPreconditioner(
                requireValue(arg));
      } else if (arg == "--output_dir") {
        outputDir = requireValue(arg);
      } else if (arg == "--save") {
        save = parseBool(requireValue(arg));
      } else {
        throw std::invalid_argument("Unknown option: " + arg);
      }
    }

    if (dataDir.empty()) {
      printUsage(argv[0]);
      return 1;
    }

    std::cout << "ABLATION_CONFIG method=manual_object_dran"
              << " base=object_copy_riemannian_local_model"
              << " shared_variable=object_copies"
              << " consensus=penalty"
              << " init=" << objectInitializationName(options.objectInitialization)
              << " topology=" << topologyName(options.topology)
              << " ring_hops=" << options.ringHops
              << " topology_file=" << options.topologyFile
              << " topology_weight_mode=" << options.topologyWeightMode
              << " communication_policy="
              << communicationPolicyName(options.communicationPolicy)
              << " object_initialization="
              << objectInitializationName(options.objectInitialization)
              << " object_initialization_anchor_mode="
              << objectInitializationAnchorModeName(
                     options.objectInitializationAnchorMode)
              << " object_initialization_consensus_rounds="
              << options.objectInitializationConsensusRounds
              << " object_initialization_observed_anchor_weight="
              << options.objectInitializationObservedAnchorWeight
              << " object_initialization_relay_anchor_weight="
              << options.objectInitializationRelayAnchorWeight
              << " object_interface_mode="
              << objectInterfaceModeName(options.objectInterfaceMode)
              << " object_interface_response_strategy="
              << objectInterfaceResponseStrategyName(
                     options.objectInterfaceResponseStrategy)
              << " object_interface_response_trigger="
              << objectInterfaceResponseTriggerName(
                     options.objectInterfaceResponseTrigger)
              << " object_interface_response_period="
              << options.objectInterfaceResponsePeriod
              << " object_interface_min_predicted_decrease="
              << options.objectInterfaceMinPredictedDecrease
              << " object_interface_min_innovation_score="
              << options.objectInterfaceMinInnovationScore
              << " object_interface_schur_damping="
              << options.objectInterfaceSchurDamping
              << " object_interface_step_gain="
              << options.objectInterfaceStepGain
              << " object_interface_max_objects_per_robot="
              << options.objectInterfaceMaxObjectsPerRobot
              << " object_interface_max_private_cols="
              << options.objectInterfaceMaxPrivateCols
              << " object_interface_max_block_step_norm="
              << options.objectInterfaceMaxBlockStepNorm
              << " object_interface_require_decrease="
              << (options.objectInterfaceRequireDecrease ? "true" : "false")
              << " object_pose_tol=" << options.objectPoseTolerance
              << " object_max_age=" << options.objectMaxAge
              << " translation_prox_weight="
              << options.translationProxWeight
              << " project_to_se_after_local_solve="
              << (options.projectToSEAfterLocalSolve ? "true" : "false")
              << " coupled_translation_trust_region="
              << (options.coupledTranslationTrustRegion ? "true" : "false")
              << " translation_elimination_prox_weight="
              << options.translationEliminationProxWeight
              << " beta=" << options.beta
              << " local_solver="
              << DPGO::manualDpgoMmLocalSolverName(options.localSolver)
              << " full_solver_outer_iterations="
              << options.fullSolverOuterIterations
              << " reduced_rotation_preconditioner="
              << DPGO::manualDpgoMmReducedRotationPreconditionerName(
                     options.reducedRotationPreconditioner)
              << " local_max_iterations="
              << options.localTrustRegionIterations
              << " local_max_tcg_iterations="
              << options.localTrustRegionMaxInnerIterations
              << std::endl;

    const DPGO::ManualObjectDranRunResult result =
        DPGO::runManualObjectDran(dataDir, options);

    std::cout << std::setprecision(20);
    for (const DPGO::ManualObjectDranIterationSummary &row :
         result.iterations) {
      std::cout << "ITER_SUMMARY iter=" << row.iter
                << " time=" << row.time
                << " global_cost=" << row.measurementCost
                << " measurement_cost=" << row.measurementCost
                << " consensus_cost=" << row.consensusCost
                << " augmented_cost="
                << row.measurementCost + row.consensusCost
                << " gradient=" << row.gradient
                << " known_object_copies=" << row.knownObjectCopies
                << " relay_object_copies=" << row.relayObjectCopies
                << " observed_object_copies=" << row.observedObjectCopies
                << " relay_only_object_copies="
                << row.relayOnlyObjectCopies
                << " initialized_from_neighbor_object_copies="
                << row.initializedFromNeighborObjectCopies
                << " stale_object_copies=" << row.staleObjectCopies
                << " stale_object_comm_triggers="
                << row.staleObjectCommTriggers
                << " active_consensus_pairs=" << row.activeConsensusPairs
                << " comm_pose_count=" << row.commPoseCount
                << " iter_comm_mb=" << row.iterCommMb
                << " cumulative_comm_pose_count="
                << row.cumulativeCommPoseCount
                << " cumulative_comm_mb=" << row.cumulativeCommMb
                << std::endl;
    }
    for (const DPGO::ManualObjectDranObjectInterfaceResponseSummary &row :
         result.objectInterfaceResponseSummaries) {
      std::cout << "OBJECT_INTERFACE_RESPONSE iter=" << row.iter
                << " scheduled=" << (row.scheduled ? 1 : 0)
                << " active_robots=" << row.activeRobots
                << " candidate_object_blocks="
                << row.candidateObjectBlocks
                << " accepted_object_blocks=" << row.acceptedObjectBlocks
                << " rejected_object_blocks=" << row.rejectedObjectBlocks
                << " batch_response_robots=" << row.batchResponseRobots
                << " accepted_robot_batches=" << row.acceptedRobotBatches
                << " rejected_robot_batches=" << row.rejectedRobotBatches
                << " trigger_candidate_robots="
                << row.triggerCandidateRobots
                << " triggered_robots=" << row.triggeredRobots
                << " skipped_trigger_robots="
                << row.skippedTriggerRobots
                << " model_cost_before=" << row.modelCostBefore
                << " model_cost_after=" << row.modelCostAfter
                << " interface_innovation_score="
                << row.interfaceInnovationScore
                << " accumulated_innovation_score="
                << row.accumulatedInnovationScore
                << " predicted_model_decrease="
                << row.predictedModelDecrease
                << " accepted_model_decrease="
                << row.acceptedModelDecrease
                << " step_norm=" << row.stepNorm << std::endl;
    }

    if (save) {
      const std::filesystem::path outputPath(outputDir);
      std::filesystem::create_directories(outputPath);
      writeIterationCsv(outputPath / "iteration_summary.csv", result);
      writeObjectInterfaceCsv(outputPath / "object_interface_summary.csv",
                              result);
      writeObjectInterfaceResponseCsv(
          outputPath / "object_interface_response_summary.csv", result);
      const DPGO::ObjectPGODirectoryData dataset =
          DPGO::loadObjectAwareG2ODirectory(dataDir, options.numRobots,
                                            options.numObjects);
      writeObjectAwareEstimates(outputPath / "object_poses.txt", dataset,
                                result);
    }

    if (!result.iterations.empty()) {
      const auto &last = result.iterations.back();
      std::cout << "final measurement_cost: " << last.measurementCost
                << std::endl;
      std::cout << "final consensus_cost: " << last.consensusCost
                << std::endl;
      std::cout << "final augmented_cost: "
                << last.measurementCost + last.consensusCost << std::endl;
      std::cout << "final gradient: " << last.gradient << std::endl;
      std::cout << "time: " << result.elapsedSeconds << " s" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "manual_object_dran failed: " << e.what() << std::endl;
    return 2;
  }

  return 0;
}

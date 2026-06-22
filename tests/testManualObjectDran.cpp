#include <DPGO/ManualObjectDran.h>

#include "gtest/gtest.h"

#include <cmath>
#include <filesystem>
#include <fstream>

using namespace DPGO;

TEST(testDPGO, ManualObjectDranDefaultsCoupledTranslationTrustRegion) {
  ManualObjectDranOptions options;
  EXPECT_TRUE(options.coupledTranslationTrustRegion);
  EXPECT_EQ(options.objectInitialization,
            ManualObjectDranObjectInitialization::CentralizedChordal);
  EXPECT_EQ(options.objectInterfaceMode,
            ManualObjectDranObjectInterfaceMode::CopyBaseline);
  EXPECT_EQ(options.objectInterfaceResponseStrategy,
            ManualObjectDranObjectInterfaceResponseStrategy::Serial);
}

TEST(testDPGO, ManualObjectDranUsesTopologyFileSchedule) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  const std::filesystem::path topologyPath =
      std::filesystem::temp_directory_path() /
      "manual_object_dran_topology_file_test.csv";
  {
    std::ofstream out(topologyPath);
    out << "round,src,dst,weight\n";
    out << "0,0,1,0.25\n";
    out << "1,1,2,0.25\n";
    out << "1,2,3,0.25\n";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 2;
  options.topologyFile = topologyPath.string();
  options.topologyWeightMode = "unit";
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);
  std::filesystem::remove(topologyPath);

  ASSERT_EQ(result.iterations.size(), 3u);
  ASSERT_GT(result.numObjects, 0u);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs,
            2u * result.numObjects);
  EXPECT_EQ(result.iterations[1].activeConsensusPairs,
            2u * result.numObjects);
  EXPECT_EQ(result.iterations.back().activeConsensusPairs,
            4u * result.numObjects);
}

TEST(testDPGO, ManualObjectDranRejectsMalformedTopologyFiles) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  auto expectTopologyThrows = [&](const std::string &name,
                                  const std::string &contents) {
    const std::filesystem::path topologyPath =
        std::filesystem::temp_directory_path() / name;
    {
      std::ofstream out(topologyPath);
      out << contents;
    }
    ManualObjectDranOptions options;
    options.numRobots = 21;
    options.numObjects = 0;
    options.maxIterations = 0;
    options.topologyFile = topologyPath.string();
    options.topologyWeightMode = "matrix";
    EXPECT_THROW(runManualObjectDran(dataDir, options), std::runtime_error);
    std::filesystem::remove(topologyPath);
  };

  expectTopologyThrows("manual_object_dran_topology_missing_round.csv",
                       "round,src,dst,weight\n1,0,1,0.25\n");
  expectTopologyThrows("manual_object_dran_topology_bad_weight.csv",
                       "round,src,dst,weight\n0,0,1,nan\n");
}

TEST(testDPGO, ManualObjectDranRunsDroneObjectCopiesWithReducedSolver) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.beta = 1.0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 5;
  options.reducedRotationPreconditioner =
      ManualDpgoMmReducedRotationPreconditioner::Portfolio;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.numRobots, 21u);
  ASSERT_GT(result.numObjects, 0u);
  EXPECT_EQ(result.knownObjectCopies, result.numRobots * result.numObjects);
  EXPECT_GT(result.observedObjectCopies, 0u);
  EXPECT_GT(result.relayObjectCopies, 0u);
  EXPECT_EQ(result.relayOnlyObjectCopies, result.relayObjectCopies);
  EXPECT_LT(result.relayObjectCopies,
            result.numRobots * result.numObjects);
  EXPECT_EQ(result.initializedFromNeighborObjectCopies, 0u);
  EXPECT_EQ(result.staleObjectCopies, 0u);
  ASSERT_EQ(result.iterations.size(), 2u);
  ASSERT_EQ(result.objectInterfaceDiagnostics.size(),
            result.iterations.size() * result.numRobots * result.numObjects);
  bool sawObservedObject = false;
  bool sawRelayObject = false;
  bool sawSelectedForSend = false;
  for (const ManualObjectDranObjectInterfaceDiagnostic &row :
       result.objectInterfaceDiagnostics) {
    EXPECT_LT(row.robotId, result.numRobots);
    EXPECT_LT(row.objectId, result.numObjects);
    EXPECT_EQ(row.relayOnly, !row.observed);
    EXPECT_TRUE(std::isfinite(row.residualNorm));
    EXPECT_TRUE(std::isfinite(row.gradientNorm));
    EXPECT_TRUE(std::isfinite(row.stiffness));
    EXPECT_TRUE(std::isfinite(row.neighborDisagreement));
    sawObservedObject = sawObservedObject || row.observed;
    sawRelayObject = sawRelayObject || row.relayOnly;
    sawSelectedForSend = sawSelectedForSend || row.selectedForSend;
  }
  EXPECT_TRUE(sawObservedObject);
  EXPECT_TRUE(sawRelayObject);
  EXPECT_TRUE(sawSelectedForSend);
  EXPECT_EQ(result.iterations.front().activeConsensusPairs, 2520u);
  EXPECT_EQ(result.iterations.front().knownObjectCopies,
            result.knownObjectCopies);
  EXPECT_EQ(result.iterations.front().relayObjectCopies,
            result.relayObjectCopies);
  EXPECT_EQ(result.iterations.front().observedObjectCopies,
            result.observedObjectCopies);
  EXPECT_EQ(result.iterations.front().relayOnlyObjectCopies,
            result.relayOnlyObjectCopies);
  EXPECT_EQ(result.iterations.front().initializedFromNeighborObjectCopies,
            result.initializedFromNeighborObjectCopies);
  EXPECT_EQ(result.iterations.front().staleObjectCopies,
            result.staleObjectCopies);
  EXPECT_EQ(result.iterations.front().staleObjectCommTriggers, 0u);
  EXPECT_GT(result.iterations.front().commPoseCount, 0u);
  EXPECT_TRUE(std::isfinite(result.iterations.back().measurementCost));
  EXPECT_TRUE(std::isfinite(result.iterations.back().consensusCost));
  EXPECT_TRUE(std::isfinite(result.iterations.back().gradient));
}

TEST(testDPGO, ManualObjectDranNeighborAverageInitializationReportsRelays) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 0;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 1u);
  ASSERT_EQ(result.objectInterfaceDiagnostics.size(),
            result.numRobots * result.numObjects);
  EXPECT_EQ(result.knownObjectCopies, result.numRobots * result.numObjects);
  EXPECT_GT(result.relayOnlyObjectCopies, 0u);
  EXPECT_EQ(result.relayObjectCopies, result.relayOnlyObjectCopies);
  EXPECT_GT(result.initializedFromNeighborObjectCopies, 0u);
  EXPECT_LE(result.initializedFromNeighborObjectCopies,
            result.relayOnlyObjectCopies);
  EXPECT_LE(result.staleObjectCopies, result.relayOnlyObjectCopies);
  EXPECT_EQ(result.iterations.front().initializedFromNeighborObjectCopies,
            result.initializedFromNeighborObjectCopies);
  EXPECT_EQ(result.iterations.front().staleObjectCopies,
            result.staleObjectCopies);
  EXPECT_LT(result.iterations.front().measurementCost, 10.0);
  EXPECT_LT(result.iterations.front().consensusCost, 1000.0);
}

TEST(testDPGO, ManualObjectDranInitializationConsensusDoesNotAffectCentralized) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions baseOptions;
  baseOptions.numRobots = 21;
  baseOptions.numObjects = 0;
  baseOptions.maxIterations = 0;
  baseOptions.objectInitialization =
      ManualObjectDranObjectInitialization::CentralizedChordal;
  baseOptions.localTrustRegionIterations = 1;
  baseOptions.localTrustRegionMaxInnerIterations = 1;

  ManualObjectDranOptions consensusOptions = baseOptions;
  consensusOptions.objectInitializationConsensusRounds = 5;
  consensusOptions.objectInitializationObservedAnchorWeight = 2.0;
  consensusOptions.objectInitializationRelayAnchorWeight = 1.0;

  const ManualObjectDranRunResult base =
      runManualObjectDran(dataDir, baseOptions);
  const ManualObjectDranRunResult consensus =
      runManualObjectDran(dataDir, consensusOptions);

  ASSERT_EQ(base.iterations.size(), 1u);
  ASSERT_EQ(consensus.iterations.size(), 1u);
  EXPECT_DOUBLE_EQ(consensus.iterations.front().measurementCost,
                   base.iterations.front().measurementCost);
  EXPECT_DOUBLE_EQ(consensus.iterations.front().consensusCost,
                   base.iterations.front().consensusCost);
  EXPECT_EQ(consensus.initializedFromNeighborObjectCopies,
            base.initializedFromNeighborObjectCopies);
  EXPECT_EQ(consensus.staleObjectCopies, base.staleObjectCopies);
}

TEST(testDPGO, ManualObjectDranInitializationConsensusReducesDisagreement) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions baseOptions;
  baseOptions.numRobots = 21;
  baseOptions.numObjects = 0;
  baseOptions.maxIterations = 0;
  baseOptions.topology = ManualObjectDranTopology::Ring;
  baseOptions.ringHops = 1;
  baseOptions.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  baseOptions.localTrustRegionIterations = 1;
  baseOptions.localTrustRegionMaxInnerIterations = 1;

  ManualObjectDranOptions consensusOptions = baseOptions;
  consensusOptions.objectInitializationConsensusRounds = 5;
  consensusOptions.objectInitializationObservedAnchorWeight = 1.0;

  const ManualObjectDranRunResult base =
      runManualObjectDran(dataDir, baseOptions);
  const ManualObjectDranRunResult consensus =
      runManualObjectDran(dataDir, consensusOptions);

  ASSERT_EQ(base.iterations.size(), 1u);
  ASSERT_EQ(consensus.iterations.size(), 1u);
  EXPECT_LT(consensus.iterations.front().consensusCost,
            base.iterations.front().consensusCost);
  EXPECT_LT(consensus.iterations.front().measurementCost, 10.0);
}

TEST(testDPGO, ManualObjectDranInformationAnchorChangesInitializationConsensus) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions constantOptions;
  constantOptions.numRobots = 21;
  constantOptions.numObjects = 0;
  constantOptions.maxIterations = 0;
  constantOptions.topology = ManualObjectDranTopology::Ring;
  constantOptions.ringHops = 1;
  constantOptions.objectInitialization =
      ManualObjectDranObjectInitialization::NeighborAverage;
  constantOptions.objectInitializationConsensusRounds = 5;
  constantOptions.objectInitializationObservedAnchorWeight = 1.0;
  constantOptions.localTrustRegionIterations = 1;
  constantOptions.localTrustRegionMaxInnerIterations = 1;

  ManualObjectDranOptions informationOptions = constantOptions;
  informationOptions.objectInitializationAnchorMode =
      ManualObjectDranObjectInitializationAnchorMode::Information;

  const ManualObjectDranRunResult constant =
      runManualObjectDran(dataDir, constantOptions);
  const ManualObjectDranRunResult information =
      runManualObjectDran(dataDir, informationOptions);

  ASSERT_EQ(constant.iterations.size(), 1u);
  ASSERT_EQ(information.iterations.size(), 1u);
  const double costDelta =
      std::abs(information.iterations.front().measurementCost -
               constant.iterations.front().measurementCost) +
      std::abs(information.iterations.front().consensusCost -
               constant.iterations.front().consensusCost);
  EXPECT_GT(costDelta, 1e-9);
  EXPECT_LT(information.iterations.front().measurementCost, 10.0);
  EXPECT_TRUE(std::isfinite(information.iterations.front().consensusCost));
}

TEST(testDPGO, ManualObjectDranRunsFullThenReducedSchedule) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 2;
  options.localSolver = ManualDpgoMmLocalSolver::ManualFull;
  options.fullSolverOuterIterations = 1;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 1;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 3u);
  EXPECT_TRUE(std::isfinite(result.iterations.back().measurementCost));
}

TEST(testDPGO, ManualObjectDranReducedSchurResponseModeReportsWork) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceSchurDamping = 0.01;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 3;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.iterations.size(), 2u);
  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 1u);
  const ManualObjectDranObjectInterfaceResponseSummary &summary =
      result.objectInterfaceResponseSummaries.front();
  EXPECT_EQ(summary.iter, 1u);
  EXPECT_GT(summary.activeRobots, 0u);
  EXPECT_GT(summary.candidateObjectBlocks, 0u);
  EXPECT_EQ(summary.candidateObjectBlocks,
            summary.acceptedObjectBlocks + summary.rejectedObjectBlocks);
  EXPECT_TRUE(std::isfinite(summary.modelCostBefore));
  EXPECT_TRUE(std::isfinite(summary.modelCostAfter));
  EXPECT_TRUE(std::isfinite(summary.acceptedModelDecrease));
  EXPECT_TRUE(std::isfinite(summary.stepNorm));
}

TEST(testDPGO, ManualObjectDranReducedSchurBatchResponseReportsBatches) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 3;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 1u);
  const ManualObjectDranObjectInterfaceResponseSummary &summary =
      result.objectInterfaceResponseSummaries.front();
  EXPECT_GT(summary.activeRobots, 0u);
  EXPECT_EQ(summary.batchResponseRobots, summary.activeRobots);
  EXPECT_EQ(summary.batchResponseRobots,
            summary.acceptedRobotBatches + summary.rejectedRobotBatches);
  EXPECT_GT(summary.candidateObjectBlocks, 0u);
  EXPECT_EQ(summary.candidateObjectBlocks,
            summary.acceptedObjectBlocks + summary.rejectedObjectBlocks);
}

TEST(testDPGO, ManualObjectDranReducedSchurResponseHonorsPeriod) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 3;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceResponsePeriod = 2;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 3u);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[0].iter, 1u);
  EXPECT_FALSE(result.objectInterfaceResponseSummaries[0].scheduled);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[0].activeRobots, 0u);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[1].iter, 2u);
  EXPECT_TRUE(result.objectInterfaceResponseSummaries[1].scheduled);
  EXPECT_GT(result.objectInterfaceResponseSummaries[1].activeRobots, 0u);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[2].iter, 3u);
  EXPECT_FALSE(result.objectInterfaceResponseSummaries[2].scheduled);
  EXPECT_EQ(result.objectInterfaceResponseSummaries[2].activeRobots, 0u);
}

TEST(testDPGO, ManualObjectDranAdaptiveResponseSkipsLowPredictedDecrease) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceResponseTrigger =
      ManualObjectDranObjectInterfaceResponseTrigger::PredictedDecrease;
  options.objectInterfaceMinPredictedDecrease = 1e100;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 1u);
  const ManualObjectDranObjectInterfaceResponseSummary &summary =
      result.objectInterfaceResponseSummaries.front();
  EXPECT_TRUE(summary.scheduled);
  EXPECT_EQ(summary.activeRobots, 0u);
  EXPECT_GT(summary.triggerCandidateRobots, 0u);
  EXPECT_EQ(summary.triggeredRobots, 0u);
  EXPECT_EQ(summary.triggerCandidateRobots, summary.skippedTriggerRobots);
  EXPECT_GT(summary.predictedModelDecrease, 0.0);
  EXPECT_TRUE(std::isfinite(summary.predictedModelDecrease));
  EXPECT_EQ(summary.candidateObjectBlocks, 0u);
}

TEST(testDPGO, ManualObjectDranLocalInnovationTriggerSkipsBeforeSchurModel) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 1;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceResponseTrigger =
      ManualObjectDranObjectInterfaceResponseTrigger::LocalInnovation;
  options.objectInterfaceMinInnovationScore = 1e100;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 1u);
  const ManualObjectDranObjectInterfaceResponseSummary &summary =
      result.objectInterfaceResponseSummaries.front();
  EXPECT_TRUE(summary.scheduled);
  EXPECT_EQ(summary.activeRobots, 0u);
  EXPECT_GT(summary.triggerCandidateRobots, 0u);
  EXPECT_EQ(summary.triggeredRobots, 0u);
  EXPECT_EQ(summary.triggerCandidateRobots, summary.skippedTriggerRobots);
  EXPECT_GT(summary.interfaceInnovationScore, 0.0);
  EXPECT_EQ(summary.predictedModelDecrease, 0.0);
  EXPECT_EQ(summary.candidateObjectBlocks, 0u);
}

TEST(testDPGO, ManualObjectDranAccumulatedInnovationCarriesSkippedSignal) {
  const std::string dataDir =
      "data/chordal_dataset/drone_g2o_file/range_10_tau_0.01_ang_error_1";
  if (!std::filesystem::exists(dataDir)) {
    GTEST_SKIP() << "missing drone object-aware data";
  }

  ManualObjectDranOptions options;
  options.numRobots = 21;
  options.numObjects = 0;
  options.maxIterations = 2;
  options.topology = ManualObjectDranTopology::Ring;
  options.ringHops = 1;
  options.objectInterfaceMode =
      ManualObjectDranObjectInterfaceMode::ReducedSchurResponse;
  options.objectInterfaceResponseStrategy =
      ManualObjectDranObjectInterfaceResponseStrategy::Batch;
  options.objectInterfaceResponseTrigger =
      ManualObjectDranObjectInterfaceResponseTrigger::AccumulatedInnovation;
  options.objectInterfaceMinInnovationScore = 1e100;
  options.objectInterfaceMaxObjectsPerRobot = 6;
  options.localSolver = ManualDpgoMmLocalSolver::ReducedRotation;
  options.localTrustRegionIterations = 1;
  options.localTrustRegionMaxInnerIterations = 2;

  const ManualObjectDranRunResult result =
      runManualObjectDran(dataDir, options);

  ASSERT_EQ(result.objectInterfaceResponseSummaries.size(), 2u);
  const ManualObjectDranObjectInterfaceResponseSummary &first =
      result.objectInterfaceResponseSummaries[0];
  const ManualObjectDranObjectInterfaceResponseSummary &second =
      result.objectInterfaceResponseSummaries[1];
  EXPECT_TRUE(first.scheduled);
  EXPECT_TRUE(second.scheduled);
  EXPECT_EQ(first.activeRobots, 0u);
  EXPECT_EQ(second.activeRobots, 0u);
  EXPECT_EQ(first.triggerCandidateRobots, first.skippedTriggerRobots);
  EXPECT_EQ(second.triggerCandidateRobots, second.skippedTriggerRobots);
  EXPECT_GT(first.accumulatedInnovationScore, 0.0);
  EXPECT_GT(second.accumulatedInnovationScore,
            first.accumulatedInnovationScore);
  EXPECT_EQ(first.predictedModelDecrease, 0.0);
  EXPECT_EQ(second.predictedModelDecrease, 0.0);
}

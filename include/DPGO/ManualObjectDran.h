/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#ifndef MANUAL_OBJECT_DRAN_H
#define MANUAL_OBJECT_DRAN_H

#include <DPGO/DPGO_types.h>
#include <DPGO/ManualDpgoMm.h>

#include <cstddef>
#include <string>
#include <vector>

namespace DPGO {

enum class ManualObjectDranTopology {
  Ring,
  Complete,
};

enum class ManualObjectDranCommunicationPolicy {
  All,
  Triggered,
};

enum class ManualObjectDranObjectInitialization {
  CentralizedChordal,
  NeighborAverage,
};

enum class ManualObjectDranObjectInitializationAnchorMode {
  Constant,
  Information,
};

enum class ManualObjectDranObjectInterfaceMode {
  CopyBaseline,
  ReducedSchurResponse,
};

enum class ManualObjectDranObjectInterfaceResponseStrategy {
  Serial,
  Batch,
};

enum class ManualObjectDranObjectInterfaceResponseTrigger {
  Periodic,
  PredictedDecrease,
  LocalInnovation,
  AccumulatedInnovation,
};

struct ManualObjectDranOptions {
  std::size_t numRobots{0};
  std::size_t numObjects{0};
  unsigned maxIterations{20};
  double beta{1.0};
  ManualObjectDranTopology topology{ManualObjectDranTopology::Ring};
  unsigned ringHops{1};
  std::string topologyFile;
  std::string topologyWeightMode{"unit"};
  ManualObjectDranCommunicationPolicy communicationPolicy{
      ManualObjectDranCommunicationPolicy::Triggered};
  ManualObjectDranObjectInitialization objectInitialization{
      ManualObjectDranObjectInitialization::CentralizedChordal};
  ManualObjectDranObjectInitializationAnchorMode
      objectInitializationAnchorMode{
          ManualObjectDranObjectInitializationAnchorMode::Constant};
  std::size_t objectInitializationConsensusRounds{0};
  double objectInitializationObservedAnchorWeight{1.0};
  double objectInitializationRelayAnchorWeight{0.0};
  ManualObjectDranObjectInterfaceMode objectInterfaceMode{
      ManualObjectDranObjectInterfaceMode::CopyBaseline};
  ManualObjectDranObjectInterfaceResponseStrategy
      objectInterfaceResponseStrategy{
          ManualObjectDranObjectInterfaceResponseStrategy::Serial};
  ManualObjectDranObjectInterfaceResponseTrigger objectInterfaceResponseTrigger{
      ManualObjectDranObjectInterfaceResponseTrigger::Periodic};
  std::size_t objectInterfaceResponsePeriod{1};
  double objectInterfaceMinPredictedDecrease{0.0};
  double objectInterfaceMinInnovationScore{0.0};
  double objectInterfaceSchurDamping{1e-2};
  double objectInterfaceStepGain{1.0};
  std::size_t objectInterfaceMaxObjectsPerRobot{0};
  unsigned objectInterfaceMaxPrivateCols{64};
  double objectInterfaceMaxBlockStepNorm{0.25};
  bool objectInterfaceRequireDecrease{true};
  double objectPoseTolerance{1e-3};
  std::size_t objectMaxAge{5};
  double translationProxWeight{0.0};
  bool projectToSEAfterLocalSolve{true};
  bool coupledTranslationTrustRegion{true};
  double translationEliminationProxWeight{0.0};
  ManualDpgoMmLocalSolver localSolver{
      ManualDpgoMmLocalSolver::ReducedRotation};
  unsigned fullSolverOuterIterations{0};
  unsigned localTrustRegionIterations{1};
  int localTrustRegionMaxInnerIterations{10};
  double localTrustRegionTolerance{1e-3};
  double localTrustRegionInitialRadius{10.0};
  ManualDpgoMmReducedRotationPreconditioner reducedRotationPreconditioner{
      ManualDpgoMmReducedRotationPreconditioner::None};
};

struct ManualObjectDranIterationSummary {
  unsigned iter{0};
  double time{0.0};
  double measurementCost{0.0};
  double consensusCost{0.0};
  double gradient{0.0};
  std::size_t knownObjectCopies{0};
  std::size_t relayObjectCopies{0};
  std::size_t observedObjectCopies{0};
  std::size_t relayOnlyObjectCopies{0};
  std::size_t initializedFromNeighborObjectCopies{0};
  std::size_t staleObjectCopies{0};
  std::size_t staleObjectCommTriggers{0};
  std::size_t activeConsensusPairs{0};
  std::size_t commPoseCount{0};
  double iterCommMb{0.0};
  std::size_t cumulativeCommPoseCount{0};
  double cumulativeCommMb{0.0};
};

struct ManualObjectDranObjectInterfaceDiagnostic {
  unsigned iter{0};
  std::size_t robotId{0};
  std::size_t objectId{0};
  bool observed{false};
  bool relayOnly{false};
  std::size_t age{0};
  double residualNorm{0.0};
  double gradientNorm{0.0};
  double stiffness{0.0};
  double neighborDisagreement{0.0};
  bool selectedForSend{false};
};

struct ManualObjectDranObjectInterfaceResponseSummary {
  unsigned iter{0};
  bool scheduled{false};
  std::size_t activeRobots{0};
  std::size_t candidateObjectBlocks{0};
  std::size_t acceptedObjectBlocks{0};
  std::size_t rejectedObjectBlocks{0};
  std::size_t batchResponseRobots{0};
  std::size_t acceptedRobotBatches{0};
  std::size_t rejectedRobotBatches{0};
  std::size_t triggerCandidateRobots{0};
  std::size_t triggeredRobots{0};
  std::size_t skippedTriggerRobots{0};
  double modelCostBefore{0.0};
  double modelCostAfter{0.0};
  double interfaceInnovationScore{0.0};
  double accumulatedInnovationScore{0.0};
  double predictedModelDecrease{0.0};
  double acceptedModelDecrease{0.0};
  double stepNorm{0.0};
};

struct ManualObjectDranRunResult {
  std::size_t numRobots{0};
  std::size_t numObjects{0};
  std::size_t knownObjectCopies{0};
  std::size_t relayObjectCopies{0};
  std::size_t observedObjectCopies{0};
  std::size_t relayOnlyObjectCopies{0};
  std::size_t initializedFromNeighborObjectCopies{0};
  std::size_t staleObjectCopies{0};
  std::vector<ManualObjectDranIterationSummary> iterations;
  std::vector<ManualObjectDranObjectInterfaceDiagnostic>
      objectInterfaceDiagnostics;
  std::vector<ManualObjectDranObjectInterfaceResponseSummary>
      objectInterfaceResponseSummaries;
  std::vector<Matrix> finalEstimates;
  double elapsedSeconds{0.0};
};

ManualObjectDranRunResult
runManualObjectDran(const std::string &directory,
                    const ManualObjectDranOptions &options);

}  // namespace DPGO

#endif

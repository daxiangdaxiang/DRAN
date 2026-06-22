/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology
 * -------------------------------------------------------------------------- */

#ifndef MANUAL_DPGO_MM_H
#define MANUAL_DPGO_MM_H

#include <DPGO/DPGO_types.h>

#include <cstddef>
#include <string>
#include <vector>

namespace DPGO {

class QuadraticProblem;

enum class ManualDpgoMmScheme {
  MM,
  AMM,
};

enum class ManualDpgoMmLocalSolver {
  ManualFull,
  FullEquivHybrid,
  ReducedRotation,
};

enum class ManualDpgoMmFullEquivHybridBackend {
  ManualFullRtr,
  SparseDirectSchur,
  PcgSchur,
  PcgFull,
};

enum class ManualDpgoMmReducedRotationPreconditioner {
  None,
  Jacobi,
  SchurJacobi,
  Cholesky,
  Portfolio,
  AdaptivePortfolio,
};

enum class ManualDpgoMmReducedSurrogateMode {
  TrueLocal,
  DpgoSimple,
  EdgeTightQuadratic,
};

enum class ManualDpgoMmSurrogateMode {
  Legacy,
  WeightedEdgeSplit,
  AdaptiveSpectral,
  VariableProjectedSchur,
};

enum class ManualDpgoMmEdgeSplitThetaMode {
  Constant,
  Degree,
  Curvature,
  AdaptiveConditioned,
};

enum class ManualDpgoMmMmAcceleratorMode {
  None,
  NesterovLegacy,
  Anderson,
  Squarem,
};

enum class ManualDpgoMmMmSafeguard {
  LocalSurrogate,
  LocalSurrogatePlusBoundary,
  DebugGlobal,
};

enum class ManualDpgoMmFullEquivHybridSchwarzBlockMode {
  PerPose,
  EdgePair,
};

enum class ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode {
  Gradient,
  ModelDecrease,
};

enum class ManualDpgoMmPostExchangeDeltaMode {
  Absolute,
  Weighted,
};

enum class ManualDpgoMmCommunicationDeltaMode {
  Sparse,
  Tangent,
  Hybrid,
};

struct ManualDpgoMmOptions {
  unsigned numRobots{5};
  unsigned maxIterations{20};
  unsigned trustRegionIterations{1};
  unsigned trustRegionAcceptedIterations{1};
  int trustRegionMaxInnerIterations{50};
  double trustRegionTolerance{1e-3};
  double trustRegionInitialRadius{10.0};
  unsigned relaxationRank{0};
  ManualDpgoMmScheme scheme{ManualDpgoMmScheme::MM};
  ManualDpgoMmLocalSolver localSolver{ManualDpgoMmLocalSolver::ReducedRotation};
  ManualDpgoMmFullEquivHybridBackend fullEquivHybridBackend{
      ManualDpgoMmFullEquivHybridBackend::ManualFullRtr};
  ManualDpgoMmReducedRotationPreconditioner reducedRotationPreconditioner{
      ManualDpgoMmReducedRotationPreconditioner::None};
  ManualDpgoMmReducedSurrogateMode reducedSurrogateMode{
      ManualDpgoMmReducedSurrogateMode::TrueLocal};
  ManualDpgoMmSurrogateMode surrogateMode{ManualDpgoMmSurrogateMode::Legacy};
  ManualDpgoMmEdgeSplitThetaMode edgeSplitThetaMode{
      ManualDpgoMmEdgeSplitThetaMode::Constant};
  double edgeSplitThetaDefault{0.5};
  double edgeSplitThetaMin{0.15};
  double edgeSplitThetaMax{0.85};
  std::vector<double> edgeSplitThetaCandidates{0.2, 0.35, 0.5, 0.65, 0.8};
  ManualDpgoMmMmAcceleratorMode mmAcceleratorMode{
      ManualDpgoMmMmAcceleratorMode::NesterovLegacy};
  ManualDpgoMmMmSafeguard mmSafeguard{
      ManualDpgoMmMmSafeguard::LocalSurrogate};
  bool debugSurrogateBoundCheck{false};
  unsigned debugSurrogateBoundSamples{8};
  bool fullEquivHybridSchurWarmStart{false};
  bool fullEquivHybridFinalPolishRtr{false};
  double fullEquivHybridSchurDamping{1e-6};
  double fullEquivHybridLinearRelativeTolerance{1e-6};
  double fullEquivHybridLinearAbsoluteTolerance{1e-10};
  unsigned fullEquivHybridLinearMaxIterations{200};
  bool fullEquivHybridLinearBlockJacobiPreconditioner{true};
  bool fullEquivHybridSparseMatrixVectorProduct{true};
  bool fullEquivHybridLinearTranslationSchurPreconditioner{false};
  bool fullEquivHybridLocalChainPreconditioner{false};
  bool fullEquivHybridTranslationBlockPreconditioner{false};
  bool fullEquivHybridTranslationSparseSchurPreconditioner{false};
  bool fullEquivHybridTranslationLocalSchurPreconditioner{false};
  unsigned fullEquivHybridTranslationLocalSchurMaxActivePoses{32};
  bool fullEquivHybridLaplacianDeflationPreconditioner{false};
  unsigned fullEquivHybridLaplacianDeflationBasisSize{4};
  unsigned fullEquivHybridLaplacianDeflationMaxEigenPoses{512};
  bool fullEquivHybridReducedRotationPreconditioner{false};
  bool fullEquivHybridReducedRotationInitialGuess{false};
  bool fullEquivHybridTranslationRecoveryInitialGuess{false};
  bool fullEquivHybridRqnWarmStart{false};
  bool fullEquivHybridRqnMemoryPreconditioner{false};
  unsigned fullEquivHybridRqnMemorySize{5};
  double fullEquivHybridRqnMinCurvatureRatio{1e-8};
  double fullEquivHybridWarmStartMaxNorm{1.0};
  unsigned fullEquivHybridWarmStartBacktrackingSteps{4};
  bool fullEquivHybridSelectBestBacktrackingTrial{false};
  bool fullEquivHybridStepParetoSelector{false};
  double fullEquivHybridStepParetoMinDecreaseRatio{0.9};
  double fullEquivHybridBacktrackingSharedPoseProxWeight{0.0};
  bool fullEquivHybridProjectedModelRhoGuard{false};
  double fullEquivHybridProjectedModelRhoEta{1e-4};
  bool fullEquivHybridActiveSeparatorCorrection{false};
  double fullEquivHybridActiveSeparatorStepCap{5e-2};
  bool fullEquivHybridActiveSeparatorBlockJacobi{false};
  bool fullEquivHybridActiveSeparatorCompactSchur{false};
  bool fullEquivHybridActiveSeparatorLmSchur{false};
  unsigned fullEquivHybridActiveSeparatorLmSchurMaxBoundaryPoses{32};
  unsigned fullEquivHybridActiveSeparatorLmSchurMaxPrivateCols{64};
  double fullEquivHybridActiveSeparatorLmSchurDamping{1e-6};
  unsigned fullEquivHybridActiveSeparatorLmSchurBacktrackingSteps{4};
  double fullEquivHybridActiveSeparatorLmSchurMinCostImprovement{0.0};
  unsigned fullEquivHybridActiveSeparatorLmSchurMaxRounds{0};
  ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode
      fullEquivHybridActiveSeparatorLmSchurScoreMode{
          ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode::
              Gradient};
  bool fullEquivHybridActiveSeparatorLmSchurGradientGuard{false};
  double fullEquivHybridActiveSeparatorLmSchurMaxGradientIncreaseRatio{0.0};
  bool fullEquivHybridTranslationRecoveryPolish{false};
  bool fullEquivHybridTranslationRecoveryPolishGradientGuard{false};
  double fullEquivHybridTranslationRecoveryPolishMaxGradientIncreaseRatio{0.0};
  bool fullEquivHybridTranslationRecoveryPolishBacktracking{false};
  unsigned fullEquivHybridTranslationRecoveryPolishBacktrackingSteps{4};
  bool fullEquivHybridTranslationRecoveryPolishMeritSelector{false};
  double fullEquivHybridTranslationRecoveryPolishMeritMinCostRecoveryRatio{
      0.9};
  bool fullEquivHybridTranslationRecoveryPolishSelectedOnly{false};
  bool fullEquivHybridTranslationRecoveryStepTrials{false};
  bool fullEquivHybridLocalPortfolio{false};
  bool manualFullPortfolio{false};
  bool fullEquivHybridSchwarzPreSmoothing{false};
  unsigned fullEquivHybridSchwarzSweeps{1};
  double fullEquivHybridSchwarzMaxBlockNorm{0.25};
  ManualDpgoMmFullEquivHybridSchwarzBlockMode
      fullEquivHybridSchwarzBlockMode{
          ManualDpgoMmFullEquivHybridSchwarzBlockMode::PerPose};
  bool centralizedChordalInit{true};
  unsigned distributedInitializationRefinementRounds{4};
  bool distributedInitializationFixedNeighborChordalRefit{false};
  std::string externalInitialEstimatePath;
  bool recordLocalModelDiagnostics{false};
  bool parallelLocalSolves{true};
  bool adaptiveReducedTcg{false};
  int adaptiveReducedTcgMaxIterations{50};
  double adaptiveReducedTcgGradientRatio{0.15};
  double reducedRotationTcgRelativeTolerance{0.1};
  bool localStateExtrapolation{false};
  double localStateExtrapolationGamma{0.5};
  std::vector<double> localStateExtrapolationGammas;
  bool localStateExtrapolationUseActiveSurrogate{false};
  bool localBoundaryProximalCandidate{false};
  double localBoundaryProximalWeight{0.1};
  std::vector<double> localBoundaryProximalWeights;
  bool localModelGExtrapolation{false};
  double localModelGExtrapolationGamma{0.5};
  std::vector<double> localModelGExtrapolationGammas;
  bool coupledStateGExtrapolation{false};
  double coupledStateGExtrapolationGamma{0.5};
  std::vector<double> coupledStateGExtrapolationGammas;
  bool localAndersonAcceleration{false};
  double localAndersonMaxAlpha{1.0};
  double localSquaremMaxAlpha{10.0};
  double localCandidateCostTieTolerance{0.0};
  bool globalStateExtrapolation{false};
  double globalStateExtrapolationGamma{0.5};
  std::vector<double> globalStateExtrapolationGammas;
  bool globalAndersonAcceleration{false};
  double globalAndersonMaxAlpha{1.0};
  bool localGradientCorrection{false};
  double localGradientCorrectionStep{1e-3};
  std::vector<double> localGradientCorrectionSteps;
  bool localGradientCorrectionSharedStep{false};
  bool localGradientCorrectionNeighborhoodStep{false};
  bool localGradientCorrectionCurvatureStep{false};
  bool localGradientCorrectionCoupledDirection{false};
  double localGradientCorrectionCoupledDirectionBudgetFraction{1.0};
  unsigned localGradientCorrectionCoupledDirectionMaxPacketsPerReceiver{0};
  unsigned localGradientCorrectionCoupledDirectionTopKEntries{0};
  double localGradientCorrectionCoupledDirectionMinScore{0.0};
  double localGradientCorrectionCoupledDirectionMinScoreRatio{0.0};
  double localGradientCorrectionCoupledDirectionByteBudgetMb{0.0};
  bool localGradientCorrectionBoundaryOnly{false};
  bool localGradientCorrectionBlockJacobi{false};
  bool localGradientCorrectionCompactSchur{false};
  unsigned localGradientCorrectionCompactSchurMaxBoundaryPoses{32};
  unsigned localGradientCorrectionCompactSchurMaxPrivateCols{64};
  double localGradientCorrectionCompactSchurDamping{1e-6};
  double localGradientCorrectionCompactSchurMinCostDecrease{0.0};
  bool localGradientCorrectionCompactSchurGradientGuard{false};
  double localGradientCorrectionCompactSchurMaxGradientIncreaseRatio{0.0};
  bool localGradientCorrectionFreshNeighborExchange{false};
  double localGradientCorrectionFreshMinPoseDelta{0.0};
  ManualDpgoMmPostExchangeDeltaMode localGradientCorrectionFreshDeltaMode{
      ManualDpgoMmPostExchangeDeltaMode::Absolute};
  double localGradientCorrectionFreshBudgetFraction{1.0};
  unsigned localGradientCorrectionFreshMaxPosesPerReceiver{0};
  unsigned localGradientCorrectionInnerRounds{1};
  unsigned localGradientCorrectionMaxRounds{0};
  unsigned postExchangePeriod{1};
  double postExchangeMinPoseDelta{0.0};
  ManualDpgoMmPostExchangeDeltaMode postExchangeDeltaMode{
      ManualDpgoMmPostExchangeDeltaMode::Absolute};
  double postExchangeBudgetFraction{1.0};
  unsigned postExchangeMaxPosesPerReceiver{0};
  std::string communicationTopologyFile;
  bool communicationTopologyInitialFull{true};
  unsigned communicationTopologyMaxRelayHops{0};
  double communicationTopologyHopBudgetFraction{1.0};
  bool communicationTopologyStaleAwareRelayScore{false};
  double communicationTopologyStaleAwareRelayAgeGain{1.0};
  bool communicationTopologyDeltaCompression{false};
  ManualDpgoMmCommunicationDeltaMode communicationTopologyDeltaMode{
      ManualDpgoMmCommunicationDeltaMode::Sparse};
  unsigned communicationTopologyDeltaTopK{4};
  double communicationTopologyDeltaMaxReconstructionError{5e-4};
  bool communicationTopologyLiftedDeltaCompression{false};
  double communicationTopologyLiftedDeltaMaxReconstructionError{5e-4};
  unsigned communicationTopologyLiftedDeltaRank{2};
  bool communicationTopologyValueScheduler{false};
  std::string communicationTopologyValueSchedulerMode{
      "static_sensitivity_staleness"};
  double communicationTopologyValueByteBudgetMb{0.0};
  double communicationTopologyValueMinScoreRatio{0.0};
  bool communicationTopologyValueBudgetPacing{true};
  bool communicationTopologyInterfaceModel{false};
  std::string communicationTopologyInterfaceModelPayload{"direction_block"};
  bool communicationTopologyInterfaceModelLocalMerit{true};
  bool communicationTopologyBoundaryCandidate{false};
  bool communicationTopologyBoundarySurrogate{false};
  double communicationTopologyBoundarySurrogateStaleGain{1.0};
  bool communicationTopologyReducedInterfaceModel{false};
  double communicationTopologyReducedInterfaceWeight{0.05};
  unsigned communicationTopologyReducedInterfaceMaxLocalIterations{0};
  bool communicationTopologyReducedInterfaceCandidate{false};
  bool staleBoundaryProximal{false};
  double staleBoundaryProximalWeight{0.0};
  bool staleBoundaryPrediction{false};
  double staleBoundaryPredictionGain{1.0};
  bool syncAmmReferenceAfterLocalGradientCorrection{true};
  bool globalGradientCorrection{false};
  double globalGradientCorrectionStep{1e-3};
  std::vector<double> globalGradientCorrectionSteps;
  bool resetLocalHistoryAfterGlobalCorrection{false};
  bool syncAmmReferenceAfterGlobalCorrection{false};
  double ammEta0{5e-4};
  double ammEta1{2.5e-2};
  double ammPsi{1e-10};
  double ammPhi{1e-6};
  double ammGammaScale{1.0};
  std::vector<double> ammGammaScales;
  double ammAcceptedDelta{5e-4};
  int ammOscillationCountPeriod{15};
  int ammMaxOscillations{12};
  bool ammProximalStart{false};
  bool ammDpgoRestartFallback{false};
  bool ammDpgoSurrogateParity{false};
  bool ammBaselineSurrogateState{false};
  bool ammDpgoRecursiveSimpleState{false};
  unsigned ammRecursiveSimpleReanchorPeriod{0};
  bool ammDpgoStrictRefinedGate{false};
  bool ammDpgoRecoverTranslationsAfterProximal{false};
  bool ammDpgoRefinedStartsAtRecoveredProximal{false};
  bool ammDpgoMixedSurrogatePortfolio{false};
  int ammMixedSurrogateSkipSimpleAfterTrueLocalStreak{0};
  int ammMixedSurrogateForceSimpleEverySkippedRounds{0};
  int ammCooldownAfterRejected{0};
  bool ammProxResetSkipRefinedSolve{false};
  bool ammDpgoProximalFallbackOnly{false};
  bool ammLazyPlainAfterCertificate{false};
  bool ammSurrogateFirstExactEvaluation{false};
  int ammMaxSoftRestartHits0{10};
  int ammMaxSoftRestartHits1{25};
  bool ammLocalMeritFilter{true};
  double ammLocalMeritCostTieTolerance{1e-4};
  bool traceAmm{false};
  bool profileOptimizer{false};
  bool candidateEvaluationCache{false};
  bool candidateObjectReuse{false};
  bool reducedAdaptivePortfolioCertifiedFastPath{false};
  bool reducedRotationDirectObjective{false};
  bool fusedCandidateEvaluation{false};
  bool lazyCandidateGradientEvaluation{false};
  bool lazySolverStartGradientEvaluation{false};
  bool lazySurrogateCandidateGradientEvaluation{false};
  bool reducedRotationCurvatureCauchyCandidate{false};
  bool reducedRotationCurvatureFallbackCandidate{false};
  bool reducedRotationGradientBoundaryCandidate{true};
  bool reducedRotationSurrogateTcgAccept{false};
  bool reducedRotationSkipRedundantCandidateProjection{false};
  bool printIterationSummary{true};
  bool save{true};
  bool saveIterationEstimates{false};
  bool verbose{false};
  std::string outputDirectory{"."};
  std::string lossName{"trivial"};
};

struct ManualDpgoMmIterationSummary {
  unsigned iter{0};
  double time{0.0};
  double globalCost{0.0};
  double gradient{0.0};
  std::size_t commPoseCount{0};
  double iterCommMb{0.0};
  std::size_t cumulativeCommPoseCount{0};
  double cumulativeCommMb{0.0};
  std::size_t initializationCommPoseCount{0};
  double initializationCommMb{0.0};
  std::size_t outerCommPoseCount{0};
  double outerCommMb{0.0};
  double poseIterCommMb{0.0};
  double poseCumulativeCommMb{0.0};
  std::size_t localGradientScalarCount{0};
  double localGradientIterScalarCommMb{0.0};
  double localGradientCumulativeScalarCommMb{0.0};
  double localGradientSelectedStep{0.0};
  std::size_t localGradientFreshPoseCount{0};
  double localGradientFreshPoseCommMb{0.0};
  std::size_t localGradientFreshSkippedPoseCount{0};
  std::size_t localGradientInnerRoundCount{0};
  std::size_t postExchangePoseCount{0};
  double postExchangePoseCommMb{0.0};
  std::size_t postExchangeSkippedPoseCount{0};
  std::size_t localModelFailures{0};
  std::size_t localOptimizationFailures{0};
  std::size_t localAcceptedIterationCount{0};
  std::size_t fullEquivHybridWarmStartCandidateCount{0};
  std::size_t fullEquivHybridWarmStartAcceptedCount{0};
  std::size_t fullEquivHybridWarmStartGuardRejectedCount{0};
  std::size_t fullEquivHybridSchurStepCandidateCount{0};
  std::size_t fullEquivHybridSchurStepAcceptedCount{0};
  std::size_t fullEquivHybridSchurStepGuardRejectedCount{0};
  std::size_t fullEquivHybridLocalPortfolioCandidateCount{0};
  std::size_t fullEquivHybridLocalPortfolioSelectedUnsmoothedCount{0};
  std::size_t fullEquivHybridLocalPortfolioSelectedSchwarzOnlyCount{0};
  std::size_t fullEquivHybridLocalPortfolioSelectedSchwarzFehCount{0};
  std::size_t fullEquivHybridSchwarzSmoothingSweepCount{0};
  std::size_t fullEquivHybridSchwarzSmoothingCandidateCount{0};
  std::size_t fullEquivHybridSchwarzSmoothingAcceptedCount{0};
  std::size_t fullEquivHybridSchwarzSmoothingRejectedCount{0};
  double fullEquivHybridSchwarzSmoothingCostDecrease{0.0};
  double fullEquivHybridSchwarzSmoothingTimeSec{0.0};
  std::size_t fullEquivHybridLinearPcgIterationCount{0};
  double fullEquivHybridLinearInitialResidual{0.0};
  double fullEquivHybridLinearFinalResidual{0.0};
  double fullEquivHybridLinearFullResidual{0.0};
  double fullEquivHybridLinearSolveTimeSec{0.0};
  std::size_t fullEquivHybridStepTrialCount{0};
  std::size_t fullEquivHybridStepTrialAcceptedCount{0};
  std::size_t fullEquivHybridStepTrialRejectedCount{0};
  double fullEquivHybridStepPredictedDecreaseSum{0.0};
  double fullEquivHybridStepActualDecreaseSum{0.0};
  double fullEquivHybridStepRhoSum{0.0};
  std::size_t fullEquivHybridStepRhoCount{0};
  double fullEquivHybridStepAcceptedScaleSum{0.0};
  double fullEquivHybridStepAcceptedPredictedDecreaseSum{0.0};
  double fullEquivHybridStepAcceptedActualDecreaseSum{0.0};
  double fullEquivHybridStepAcceptedRhoSum{0.0};
  std::size_t fullEquivHybridStepAcceptedRhoCount{0};
  std::size_t fullEquivHybridStepParetoCandidateCount{0};
  std::size_t fullEquivHybridStepParetoSelectedCount{0};
  double fullEquivHybridStepParetoSelectedScaleSum{0.0};
  double fullEquivHybridStepParetoBestDecreaseSum{0.0};
  double fullEquivHybridStepParetoSelectedDecreaseSum{0.0};
  double fullEquivHybridStepParetoSelectedGradientSum{0.0};
  std::size_t fullEquivHybridStepParetoGradientEvalCount{0};
  double fullEquivHybridStepParetoGradientEvalTimeSec{0.0};
  std::size_t fullEquivHybridTranslationRecoveryStepTrialCandidateCount{0};
  std::size_t fullEquivHybridTranslationRecoveryStepTrialHardAcceptedCount{0};
  std::size_t fullEquivHybridTranslationRecoveryStepTrialSelectedCount{0};
  std::size_t fullEquivHybridActiveSeparatorCandidateCount{0};
  std::size_t fullEquivHybridActiveSeparatorAcceptedCount{0};
  std::size_t fullEquivHybridActiveSeparatorRejectedCount{0};
  double fullEquivHybridActiveSeparatorStepSum{0.0};
  double fullEquivHybridActiveSeparatorCostDecreaseSum{0.0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurCandidateCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurAcceptedCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurGuardRejectedCount{0};
  std::size_t
      fullEquivHybridActiveSeparatorLmSchurGradientGuardRejectedCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurSolveFailureCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurFallbackCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurBoundaryColCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurPrivateColCount{0};
  std::size_t fullEquivHybridActiveSeparatorLmSchurBacktrackingTrialCount{0};
  double fullEquivHybridActiveSeparatorLmSchurAlphaSum{0.0};
  double fullEquivHybridActiveSeparatorLmSchurCostDecreaseSum{0.0};
  double fullEquivHybridActiveSeparatorLmSchurGradientChangeSum{0.0};
  std::size_t fullEquivHybridTranslationRecoveryPolishAttemptCount{0};
  std::size_t fullEquivHybridTranslationRecoveryPolishAcceptedCount{0};
  std::size_t fullEquivHybridTranslationRecoveryPolishRejectedCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryPolishGradientGuardRejectedCount{0};
  std::size_t fullEquivHybridTranslationRecoveryPolishBacktrackingTriggerCount{
      0};
  std::size_t fullEquivHybridTranslationRecoveryPolishBacktrackingTrialCount{
      0};
  std::size_t
      fullEquivHybridTranslationRecoveryPolishBacktrackingAcceptedCount{0};
  double fullEquivHybridTranslationRecoveryPolishBacktrackingAlphaSum{0.0};
  std::size_t fullEquivHybridTranslationRecoveryPolishMeritCandidateCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryPolishMeritSelectedFullCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryPolishMeritSelectedPartialCount{0};
  double fullEquivHybridTranslationRecoveryPolishMeritSelectedAlphaSum{0.0};
  double fullEquivHybridTranslationRecoveryPolishMeritBestDecreaseSum{0.0};
  double fullEquivHybridTranslationRecoveryPolishMeritSelectedDecreaseSum{
      0.0};
  double fullEquivHybridTranslationRecoveryPolishCostDecreaseSum{0.0};
  double fullEquivHybridTranslationRecoveryPolishGradientChangeSum{0.0};
  std::size_t fullEquivHybridSparseMatrixVectorProductCount{0};
  std::size_t fullEquivHybridReducedRotationInitialGuessCandidateCount{0};
  std::size_t fullEquivHybridReducedRotationInitialGuessUsedCount{0};
  std::size_t fullEquivHybridReducedRotationInitialGuessRejectedCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryInitialGuessCandidateCount{0};
  std::size_t fullEquivHybridTranslationRecoveryInitialGuessUsedCount{0};
  std::size_t
      fullEquivHybridTranslationRecoveryInitialGuessRejectedCount{0};
  std::size_t fullEquivHybridTranslationSchurPreconditionerApplicationCount{0};
  std::size_t fullEquivHybridTranslationSchurPreconditionerFactorizationCount{
      0};
  std::size_t fullEquivHybridTranslationSchurPreconditionerFallbackCount{0};
  std::size_t fullEquivHybridTranslationBlockPreconditionerApplicationCount{0};
  std::size_t fullEquivHybridTranslationBlockPreconditionerFactorizationCount{
      0};
  std::size_t fullEquivHybridTranslationBlockPreconditionerFallbackCount{0};
  std::size_t
      fullEquivHybridTranslationSparseSchurPreconditionerApplicationCount{0};
  std::size_t
      fullEquivHybridTranslationSparseSchurPreconditionerFactorizationCount{0};
  std::size_t fullEquivHybridTranslationSparseSchurPreconditionerFallbackCount{
      0};
  std::size_t
      fullEquivHybridTranslationLocalSchurPreconditionerApplicationCount{0};
  std::size_t
      fullEquivHybridTranslationLocalSchurPreconditionerFactorizationCount{0};
  std::size_t fullEquivHybridTranslationLocalSchurPreconditionerFallbackCount{
      0};
  std::size_t fullEquivHybridTranslationLocalSchurPreconditionerActivePoseCount{
      0};
  std::size_t
      fullEquivHybridTranslationLocalSchurPreconditionerActiveColumnCount{0};
  std::size_t fullEquivHybridLaplacianDeflationPreconditionerApplicationCount{
      0};
  std::size_t fullEquivHybridLaplacianDeflationPreconditionerFactorizationCount{
      0};
  std::size_t fullEquivHybridLaplacianDeflationPreconditionerFallbackCount{0};
  std::size_t fullEquivHybridLaplacianDeflationPreconditionerBasisDimension{0};
  std::size_t
      fullEquivHybridReducedRotationPreconditionerApplicationCount{0};
  std::size_t fullEquivHybridReducedRotationPreconditionerFactorizationCount{
      0};
  std::size_t fullEquivHybridRqnUsedCount{0};
  std::size_t fullEquivHybridRqnAcceptedPairCount{0};
  std::size_t fullEquivHybridRqnRejectedPairCount{0};
  std::size_t fullEquivHybridRqnMemorySize{0};
  std::size_t fullEquivHybridRqnPreconditionerApplicationCount{0};
  std::size_t fullEquivHybridLocalChainPreconditionerApplicationCount{0};
  std::size_t fullEquivHybridLocalChainPreconditionerFactorizationCount{0};
  std::size_t fullEquivHybridLocalChainPreconditionerFallbackCount{0};
  std::size_t adaptiveRefinementCount{0};
  std::size_t extrapolationAcceptedCount{0};
  std::size_t extrapolationRejectedCount{0};
  std::size_t gExtrapolationAcceptedCount{0};
  std::size_t gExtrapolationRejectedCount{0};
  std::size_t coupledExtrapolationAcceptedCount{0};
  std::size_t coupledExtrapolationRejectedCount{0};
  std::size_t andersonAcceptedCount{0};
  std::size_t andersonRejectedCount{0};
  std::size_t squaremAcceptedCount{0};
  std::size_t squaremRejectedCount{0};
  std::size_t globalExtrapolationAcceptedCount{0};
  std::size_t globalExtrapolationRejectedCount{0};
  std::size_t globalAndersonAcceptedCount{0};
  std::size_t globalAndersonRejectedCount{0};
  std::size_t localGradientAcceptedCount{0};
  std::size_t localGradientRejectedCount{0};
  std::size_t localGradientCoupledDirectionCandidateCount{0};
  std::size_t localGradientCoupledDirectionAcceptedCount{0};
  std::size_t localGradientCoupledDirectionRejectedCount{0};
  std::size_t localGradientCoupledDirectionPacketCount{0};
  std::size_t localGradientCoupledDirectionSkippedPacketCount{0};
  std::size_t localGradientCoupledDirectionScoreSkippedPacketCount{0};
  std::size_t localGradientCoupledDirectionByteBudgetSkippedPacketCount{0};
  std::size_t localGradientCoupledDirectionScalarCount{0};
  double localGradientCoupledDirectionScalarCommMb{0.0};
  std::size_t localGradientCompactSchurCandidateCount{0};
  std::size_t localGradientCompactSchurAcceptedCount{0};
  std::size_t localGradientCompactSchurGuardRejectedCount{0};
  std::size_t localGradientCompactSchurGradientGuardRejectedCount{0};
  std::size_t localGradientCompactSchurSolveFailureCount{0};
  std::size_t localGradientCompactSchurBoundaryColCount{0};
  std::size_t localGradientCompactSchurPrivateColCount{0};
  double localGradientCompactSchurGradientChangeSum{0.0};
  std::size_t globalGradientAcceptedCount{0};
  std::size_t globalGradientRejectedCount{0};
  std::size_t localHistorySyncCount{0};
  double localModelCostBefore{0.0};
  double localModelCostAfter{0.0};
  double localModelGradientBefore{0.0};
  double localModelGradientAfter{0.0};
  double boundaryEdgeCostBefore{0.0};
  double boundaryEdgeCostAfter{0.0};
  double separatorDeltaNorm{0.0};
  std::size_t boundaryProximalCandidateCount{0};
  std::size_t boundaryProximalAcceptedCount{0};
  std::size_t edgeTightQuadraticEvalCount{0};
  double edgeTightQuadraticSurrogateCostSum{0.0};
  double edgeTightQuadraticTrueCostSum{0.0};
  double edgeTightQuadraticMajorizationGapMin{0.0};
  std::size_t ammAcceleratedAcceptedCount{0};
  std::size_t ammRestartCount{0};
  std::size_t ammHardRestartCount{0};
  std::size_t ammSoftRestartCount{0};
  std::size_t ammPhiFallbackCount{0};
  std::size_t ammLocalMeritRejectedCount{0};
  std::size_t ammProximalStartCount{0};
  std::size_t ammSkippedCount{0};
  std::size_t ammTraceCount{0};
  std::size_t ammRefinedCount{0};
  std::size_t ammProxResetCount{0};
  std::size_t ammRestartUsedXakhCount{0};
  std::size_t ammRestartCertificateCount{0};
  std::size_t ammRestartCertificatePassedCount{0};
  std::size_t ammRestartCertificateFailedCount{0};
  double ammRestartCertificateMinMargin{0.0};
  std::size_t ammTranslationRecoveryAttemptCount{0};
  std::size_t ammTranslationRecoveryAcceptedCount{0};
  std::size_t ammTranslationRecoveryRejectedCount{0};
  std::size_t ammMixedSurrogateCandidateCount{0};
  std::size_t ammMixedSurrogateTrueLocalAcceptedCount{0};
  std::size_t ammMixedSurrogateSimpleSelectedCount{0};
  std::size_t ammMixedSurrogateTrueLocalSelectedCount{0};
  std::size_t ammMixedSurrogateExtrapolatedSelectedCount{0};
  std::size_t ammMixedSurrogateOtherSelectedCount{0};
  std::size_t ammMixedSurrogateSimpleSkippedCount{0};
  std::size_t ammMixedSurrogateSimpleForcedRefreshCount{0};
  double ammMeanGamma{0.0};
  double ammMeanGkhInitial{0.0};
  double ammMeanMinG{0.0};
  double ammMeanGkAfterAccelerated{0.0};
  double ammMeanGkhAfterRestartCheck{0.0};
  double ammMeanFinalGk{0.0};
  double ammMeanPhiLhs{0.0};
  double ammMeanPhiRhs{0.0};
};

struct ManualDpgoMmRunResult {
  std::vector<ManualDpgoMmIterationSummary> iterations;
  std::vector<Matrix> initializationEstimates;
  std::vector<Matrix> iterationEstimates;
  Matrix finalEstimate;
  double finalObjective{0.0};
  double finalGradient{0.0};
  double elapsedSeconds{0.0};
  std::size_t surrogateBoundCheckCount{0};
  std::size_t surrogateBoundViolationCount{0};
  double surrogateBoundMinMargin{0.0};
  std::size_t variableProjectedSchurCandidateCount{0};
  std::size_t variableProjectedSchurAcceptedCount{0};
  std::size_t boundaryProximalCandidateCount{0};
  std::size_t boundaryProximalAcceptedCount{0};
  std::size_t optimizerReducedSolveCount{0};
};

struct ManualDpgoMmLocalModelSummary {
  unsigned robot{0};
  bool ok{false};
  double cost{0.0};
  double gradient{0.0};
};

struct ManualDpgoMmLocalModelSnapshot {
  unsigned robot{0};
  bool ok{false};
  SparseMatrix q;
  SparseMatrix g;
};

struct ManualDpgoMmQuadraticStats {
  double globalCost{0.0};
  double gradient{0.0};
};

struct ManualDpgoMmWeightedEdgeSplitStats {
  double objective{0.0};
  double surrogate{0.0};
  double gap{0.0};
  double identityGap{0.0};
};

ManualDpgoMmScheme parseManualDpgoMmScheme(const std::string &value,
                                           bool accelerated);

ManualDpgoMmLocalSolver
parseManualDpgoMmLocalSolver(const std::string &value);

ManualDpgoMmFullEquivHybridBackend
parseManualDpgoMmFullEquivHybridBackend(const std::string &value);

ManualDpgoMmReducedRotationPreconditioner
parseManualDpgoMmReducedRotationPreconditioner(const std::string &value);

ManualDpgoMmReducedSurrogateMode
parseManualDpgoMmReducedSurrogateMode(const std::string &value);

ManualDpgoMmSurrogateMode
parseManualDpgoMmSurrogateMode(const std::string &value);

ManualDpgoMmEdgeSplitThetaMode
parseManualDpgoMmEdgeSplitThetaMode(const std::string &value);

ManualDpgoMmMmAcceleratorMode
parseManualDpgoMmMmAcceleratorMode(const std::string &value);

ManualDpgoMmMmSafeguard
parseManualDpgoMmMmSafeguard(const std::string &value);

ManualDpgoMmFullEquivHybridSchwarzBlockMode
parseManualDpgoMmFullEquivHybridSchwarzBlockMode(const std::string &value);

ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode
parseManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode(
    const std::string &value);

std::string manualDpgoMmSchemeName(ManualDpgoMmScheme scheme);

std::string manualDpgoMmLocalSolverName(ManualDpgoMmLocalSolver solver);

std::string manualDpgoMmFullEquivHybridBackendName(
    ManualDpgoMmFullEquivHybridBackend backend);

std::string manualDpgoMmReducedRotationPreconditionerName(
    ManualDpgoMmReducedRotationPreconditioner preconditioner);

std::string manualDpgoMmReducedSurrogateModeName(
    ManualDpgoMmReducedSurrogateMode mode);

std::string manualDpgoMmSurrogateModeName(ManualDpgoMmSurrogateMode mode);

std::string manualDpgoMmEdgeSplitThetaModeName(
    ManualDpgoMmEdgeSplitThetaMode mode);

std::string manualDpgoMmMmAcceleratorModeName(
    ManualDpgoMmMmAcceleratorMode mode);

std::string manualDpgoMmMmSafeguardName(ManualDpgoMmMmSafeguard mode);

std::string manualDpgoMmFullEquivHybridSchwarzBlockModeName(
    ManualDpgoMmFullEquivHybridSchwarzBlockMode mode);

std::string manualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreModeName(
    ManualDpgoMmFullEquivHybridActiveSeparatorLmSchurScoreMode mode);

std::size_t manualDpgoMmPosePayloadBytes(unsigned dimension);

ManualDpgoMmQuadraticStats
evaluateManualDpgoMmQuadraticStats(const QuadraticProblem &problem,
                                   const Matrix &X);

ManualDpgoMmWeightedEdgeSplitStats evaluateManualDpgoMmWeightedEdgeSplit(
    const Matrix &a, const Matrix &b, const Matrix &ak, const Matrix &bk,
    double theta);

double evaluateManualDpgoMmDegreeEdgeSplitTheta(double degreeAlpha,
                                                double degreeBeta,
                                                double thetaMin,
                                                double thetaMax);

double evaluateManualDpgoMmCurvatureEdgeSplitTheta(double curvatureAlpha,
                                                   double curvatureBeta,
                                                   double thetaMin,
                                                   double thetaMax);

double evaluateManualDpgoMmAdaptiveConditionedEdgeSplitTheta(
    double degreeAlpha, double degreeBeta, double curvatureAlpha,
    double curvatureBeta, const std::vector<double> &thetaCandidates,
    double thetaMin, double thetaMax);

double evaluateManualDpgoMmScalarYoungMajorizerGap(
    const Matrix &deltaAlpha, const Matrix &deltaBeta,
    const Matrix &crossBlock, double eta);

double evaluateManualDpgoMmBlockGershgorinMajorizerGap(
    const Matrix &delta, const Matrix &schur, unsigned blockDim);

ManualDpgoMmRunResult runManualDpgoMm(const std::string &datasetPath,
                                      const ManualDpgoMmOptions &options);

std::vector<ManualDpgoMmLocalModelSummary>
inspectManualDpgoMmInitialLocalModels(const std::string &datasetPath,
                                      const ManualDpgoMmOptions &options);

std::vector<ManualDpgoMmLocalModelSnapshot>
inspectManualDpgoMmInitialLocalModelSnapshots(
    const std::string &datasetPath, const ManualDpgoMmOptions &options);

}  // namespace DPGO

#endif

/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef REDUCEDROTATIONQUADRATICOPTIMIZER_H
#define REDUCEDROTATIONQUADRATICOPTIMIZER_H

#include <DPGO/DPGO_types.h>
#include <DPGO/QuadraticProblem.h>

#include <algorithm>
#include <cstddef>
#include <memory>

namespace DPGO {

/**
 * Local quadratic optimizer that treats translation columns as conditionally
 * eliminated Euclidean variables for fixed rotation/Stiefel columns.
 */
class ReducedRotationQuadraticOptimizer {
 public:
  explicit ReducedRotationQuadraticOptimizer(QuadraticProblem *p);
  ~ReducedRotationQuadraticOptimizer();

  Matrix optimize(const Matrix &Y);

  Matrix recoverTranslationsForRotations(const Matrix &Y);
  double evaluateReducedObjective(const Matrix &Y);
  double evaluateQuadraticObjectiveDirect(const Matrix &Y) const;
  Matrix buildReducedGradient(const Matrix &Y);
  Matrix applyReducedSchurHv(const Matrix &Y, const Matrix &rotationEta);

  ROPTResult getOptResult() const { return result; }
  std::size_t getTranslationFactorizationCount() const;
  std::size_t getTranslationRecoveryCostValidationCount() const;
  double getProfileTranslationRecoverySeconds() const {
    return profileTranslationRecoverySeconds;
  }
  double getProfileTranslationFactorizationSeconds() const {
    return profileTranslationFactorizationSeconds;
  }
  double getProfileTranslationCacheLookupSeconds() const {
    return profileTranslationCacheLookupSeconds;
  }
  std::size_t getProfileTranslationRecoveryCount() const {
    return profileTranslationRecoveryCount;
  }
  std::size_t getProfileTranslationFactorizationCount() const {
    return profileTranslationFactorizationCount;
  }
  std::size_t getProfileTranslationCacheLookupCount() const {
    return profileTranslationCacheLookupCount;
  }
  double getProfileReducedHessianProductSeconds() const {
    return profileReducedHessianProductSeconds;
  }
  std::size_t getProfileReducedHessianProductCount() const {
    return profileReducedHessianProductCount;
  }
  double getProfileReducedObjectiveSeconds() const {
    return profileReducedObjectiveSeconds;
  }
  std::size_t getProfileReducedObjectiveCount() const {
    return profileReducedObjectiveCount;
  }
  double getProfileReducedGradientBuildSeconds() const {
    return profileReducedGradientBuildSeconds;
  }
  std::size_t getProfileReducedGradientBuildCount() const {
    return profileReducedGradientBuildCount;
  }
  double getProfileReducedPreconditionerSeconds() const {
    return profileReducedPreconditionerSeconds;
  }
  std::size_t getProfileReducedPreconditionerCount() const {
    return profileReducedPreconditionerCount;
  }
  double getProfileReducedProjectionSeconds() const {
    return profileReducedProjectionSeconds;
  }
  std::size_t getProfileReducedProjectionCount() const {
    return profileReducedProjectionCount;
  }
  double getProfileReducedRetractionSeconds() const {
    return profileReducedRetractionSeconds;
  }
  std::size_t getProfileReducedRetractionCount() const {
    return profileReducedRetractionCount;
  }
  double getProfileReducedStepNormSeconds() const {
    return profileReducedStepNormSeconds;
  }
  std::size_t getProfileReducedStepNormCount() const {
    return profileReducedStepNormCount;
  }
  double getProfileReducedTrustRegionScalingSeconds() const {
    return profileReducedTrustRegionScalingSeconds;
  }
  std::size_t getProfileReducedTrustRegionScalingCount() const {
    return profileReducedTrustRegionScalingCount;
  }
  double getProfileReducedTranslationResponseSeconds() const {
    return profileReducedTranslationResponseSeconds;
  }
  std::size_t getProfileReducedTranslationResponseCount() const {
    return profileReducedTranslationResponseCount;
  }
  double getProfileReducedRotationProductSeconds() const {
    return profileReducedRotationProductSeconds;
  }
  std::size_t getProfileReducedRotationProductCount() const {
    return profileReducedRotationProductCount;
  }
  std::size_t getProfileCurvatureCauchyCandidateCount() const {
    return profileCurvatureCauchyCandidateCount;
  }
  std::size_t getProfileCurvatureCauchyAcceptedCount() const {
    return profileCurvatureCauchyAcceptedCount;
  }
  std::size_t getProfileCurvatureCauchyFallbackCandidateCount() const {
    return profileCurvatureCauchyFallbackCandidateCount;
  }
  std::size_t getProfileCurvatureCauchyFallbackAcceptedCount() const {
    return profileCurvatureCauchyFallbackAcceptedCount;
  }
  std::size_t getProfileReducedCandidateProjectionSkipCount() const {
    return profileReducedCandidateProjectionSkipCount;
  }

  void setVerbose(bool flag) { verbose = flag; }
  void setTrustRegionIterations(unsigned iter) { maxIterations = iter; }
  void setTrustRegionAcceptedIterations(unsigned iter) {
    maxAcceptedIterations = iter;
  }
  void setTrustRegionTolerance(double tol) { gradientTolerance = tol; }
  void setTrustRegionInitialRadius(double radius) { initialRadius = radius; }
  void setTrustRegionMaxInnerIterations(int iter) { maxCgIterations = iter; }
  void setTruncatedCgRelativeTolerance(double tolerance) {
    truncatedCgRelativeTolerance = std::max(0.0, tolerance);
  }
  void setRecordResultStats(bool flag) { recordResultStats = flag; }
  void setProfileRuntime(bool flag) { profileRuntime = flag; }
  void setValidateTranslationRecoveryCost(bool flag) {
    validateTranslationRecoveryCost = flag;
  }
  void setUseJacobiPreconditioner(bool flag) {
    useJacobiPreconditioner = flag;
  }
  void setUseCholeskyPreconditioner(bool flag) {
    useCholeskyPreconditioner = flag;
  }
  void setUseSchurJacobiPreconditioner(bool flag) {
    useSchurJacobiPreconditioner = flag;
  }
  void setUseCoupledTrustRegionNorm(bool flag) {
    useCoupledTrustRegionNorm = flag;
  }
  void setUseDirectObjectiveEvaluation(bool flag) {
    useDirectObjectiveEvaluation = flag;
  }
  void setUseCurvatureCauchyCandidate(bool flag) {
    useCurvatureCauchyCandidate = flag;
  }
  void setUseCurvatureCauchyFallbackCandidate(bool flag) {
    useCurvatureCauchyFallbackCandidate = flag;
  }
  void setUseGradientBoundaryCandidate(bool flag) {
    useGradientBoundaryCandidate = flag;
  }
  void setUseSurrogateTcgAccept(bool flag) {
    useSurrogateTcgAccept = flag;
  }
  void setSkipRedundantCandidateProjection(bool flag) {
    skipRedundantCandidateProjection = flag;
  }
  void setTranslationEliminationProxWeight(double weight) {
    translationEliminationProxWeight = std::max(0.0, weight);
  }

 private:
  QuadraticProblem *problem;
  bool verbose;
  bool recordResultStats;
  bool profileRuntime;
  bool validateTranslationRecoveryCost;
  bool useJacobiPreconditioner;
  bool useCholeskyPreconditioner;
  bool useSchurJacobiPreconditioner;
  bool useCoupledTrustRegionNorm;
  bool useDirectObjectiveEvaluation;
  bool useCurvatureCauchyCandidate;
  bool useCurvatureCauchyFallbackCandidate;
  bool useGradientBoundaryCandidate;
  bool useSurrogateTcgAccept;
  bool skipRedundantCandidateProjection;
  unsigned maxIterations;
  unsigned maxAcceptedIterations;
  int maxCgIterations;
  double gradientTolerance;
  double initialRadius;
  double initialDamping;
  double truncatedCgRelativeTolerance;
  double translationEliminationProxWeight;
  ROPTResult result;
  double profileTranslationRecoverySeconds;
  double profileTranslationFactorizationSeconds;
  double profileTranslationCacheLookupSeconds;
  double profileReducedHessianProductSeconds;
  double profileReducedObjectiveSeconds;
  double profileReducedGradientBuildSeconds;
  double profileReducedPreconditionerSeconds;
  double profileReducedProjectionSeconds;
  double profileReducedRetractionSeconds;
  double profileReducedStepNormSeconds;
  double profileReducedTrustRegionScalingSeconds;
  double profileReducedTranslationResponseSeconds;
  double profileReducedRotationProductSeconds;
  std::size_t profileTranslationRecoveryCount;
  std::size_t profileTranslationFactorizationCount;
  std::size_t profileTranslationCacheLookupCount;
  std::size_t profileReducedHessianProductCount;
  std::size_t profileReducedObjectiveCount;
  std::size_t profileReducedGradientBuildCount;
  std::size_t profileReducedPreconditionerCount;
  std::size_t profileReducedProjectionCount;
  std::size_t profileReducedRetractionCount;
  std::size_t profileReducedStepNormCount;
  std::size_t profileReducedTrustRegionScalingCount;
  std::size_t profileReducedTranslationResponseCount;
  std::size_t profileReducedRotationProductCount;
  std::size_t profileCurvatureCauchyCandidateCount;
  std::size_t profileCurvatureCauchyAcceptedCount;
  std::size_t profileCurvatureCauchyFallbackCandidateCount;
  std::size_t profileCurvatureCauchyFallbackAcceptedCount;
  std::size_t profileReducedCandidateProjectionSkipCount;
  struct TranslationCache;
  std::unique_ptr<TranslationCache> translationCache;

  Matrix projectRotationTangent(const Matrix &Y, const Matrix &Z) const;
  Matrix retractRotations(const Matrix &Y, const Matrix &eta) const;
  Matrix recoverTranslations(const Matrix &Y, const SparseMatrix &Q) const;
  Matrix translationResponse(const Matrix &rotationEta,
                             const SparseMatrix &Q) const;
  Matrix reducedHessianEta(const Matrix &Y, const Matrix &rotationEta,
                           const SparseMatrix &Q) const;
  double reducedStepNorm(const Matrix &rotationEta,
                         const SparseMatrix &Q) const;
  Matrix scaleToTrustRegion(const Matrix &rotationEta, double radius,
                            const SparseMatrix &Q) const;
  Matrix applyJacobiPreconditioner(const Matrix &Y, const Matrix &residual,
                                   const SparseMatrix &Q) const;
  static double rotationInnerProduct(const Matrix &A, const Matrix &B,
                                     unsigned d);
};

}  // namespace DPGO

#endif

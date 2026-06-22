#ifndef DPGO_OBJECT_GAUGE_COUPLING_H
#define DPGO_OBJECT_GAUGE_COUPLING_H

#include <DPGO/DPGO_types.h>

#include <cstddef>
#include <functional>
#include <vector>

namespace DPGO {

struct GaugeCouplingObservation {
  size_t poseIndex{0};
  Matrix targetPose;
  double weight{1.0};
};

struct GaugeCouplingCorrectionStats {
  bool valid{false};
  size_t usedObjects{0};
  double translationRms{0.0};
  double rotationRms{0.0};
  double score{0.0};
  double rawStepNorm{0.0};
  double appliedScale{0.0};
  double appliedStepNorm{0.0};
  bool meritAccepted{true};
  size_t meritRejectedSteps{0};
  double meritInitial{0.0};
  double meritFinal{0.0};
};

double cacheFreshnessWeight(size_t currentIter, size_t lastUpdateIter,
                            size_t maxAge, double decayRate);

double frameRegistrationPayloadMb(
    const std::vector<size_t> &commonObjectCounts, unsigned d);

bool estimateSE3GaugeCorrection(
    const Matrix &state,
    const std::vector<GaugeCouplingObservation> &observations,
    unsigned d, size_t minObjects, Matrix &R, Matrix &t,
    GaugeCouplingCorrectionStats *stats = nullptr);

Matrix applySE3GaugeCorrection(
    const Matrix &state, const Matrix &R, const Matrix &t, double alpha,
    double maxStep, unsigned d,
    GaugeCouplingCorrectionStats *stats = nullptr);

Matrix applySE3GaugeCorrectionWithMerit(
    const Matrix &state, const Matrix &R, const Matrix &t, double alpha,
    double maxStep, unsigned d, size_t maxBacktracking,
    double meritTolerance,
    const std::function<double(const Matrix &)> &meritFunction,
    GaugeCouplingCorrectionStats *stats = nullptr);

}  // namespace DPGO

#endif

/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DCCI_CHORDAL_INITIALIZATION_H
#define DCCI_CHORDAL_INITIALIZATION_H

#include <DPGO/DCCI_communicator.h>
#include <DPGO/DPGO_types.h>
#include <DPGO/PCGSolver.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <vector>

namespace DPGO {

struct DCCIParams {
  PCGParams rotation_pcg;
  PCGParams translation_pcg;
  bool use_measurement_weight = false;
};

struct DCCIStats {
  PCGResult rotation_pcg;
  PCGResult translation_pcg;
  DCCICommunicationStats communication;
};

enum class DCCITraceStage {
  TranslationPCG,
};

struct DCCITraceFrame {
  DCCITraceStage stage = DCCITraceStage::TranslationPCG;
  unsigned pcg_iteration = 0;
  double residual_norm = 0.0;
  Matrix poses;
};

struct DCCITraceOptions {
  std::vector<DCCITraceFrame> *frames = nullptr;
  unsigned frame_stride = 0;
  unsigned max_frames_per_stage = 16;
};

Matrix distributedChordalInitialization(
    size_t dimension, size_t num_poses,
    const std::vector<RelativeSEMeasurement> &measurements,
    const DistributedPartition &partition, DPGOCommunicator &comm,
    const DCCIParams &params, DCCIStats *stats = nullptr,
    const DCCITraceOptions *trace = nullptr);

}  // namespace DPGO

#endif

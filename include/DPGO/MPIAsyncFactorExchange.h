/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef MPI_ASYNC_FACTOR_EXCHANGE_H
#define MPI_ASYNC_FACTOR_EXCHANGE_H

#include <DPGO/TEDCCI.h>

#include <mpi.h>

#include <set>
#include <vector>

namespace DPGO {

class MPIAsyncFactorExchange : public AsyncFactorExchange {
 public:
  explicit MPIAsyncFactorExchange(MPI_Comm comm = MPI_COMM_WORLD);
  ~MPIAsyncFactorExchange() override;

  void SendFactor(const FactorMessage &msg) override;
  void SendSolution(const SolutionMessage &msg) override;

  std::vector<FactorMessage> PollFactorMessages(
      int receiver_robot_id) override;
  std::vector<SolutionMessage> PollSolutionMessages(
      int receiver_robot_id) override;

  void MarkLinkDown(int robot_a, int robot_b) override;
  void MarkLinkUp(int robot_a, int robot_b) override;

 private:
  struct PendingSend {
    std::vector<unsigned char> payload;
    MPI_Request request = MPI_REQUEST_NULL;
  };

  void SendPayload(std::vector<unsigned char> payload, int receiver, int tag);
  void DrainCompletedSends();
  bool IsLinkDown(int sender, int receiver) const;

  MPI_Comm comm_ = MPI_COMM_NULL;
  int rank_ = 0;
  int size_ = 1;
  std::vector<PendingSend> pending_sends_;
  std::set<std::pair<int, int>> down_links_;
};

}  // namespace DPGO

#endif

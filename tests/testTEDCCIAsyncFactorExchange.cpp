#include <DPGO/TEDCCI.h>

#include "gtest/gtest.h"

using namespace DPGO;

namespace {

FactorMessage makeFactorMessage(int sender, int receiver, uint64_t epoch) {
  FactorMessage msg;
  msg.sender_robot_id = sender;
  msg.receiver_robot_id = receiver;
  msg.epoch = epoch;
  msg.factor_type = FactorType::ROTATION;
  msg.keys = {PoseKey{sender, 1}, PoseKey{receiver, 2}};
  msg.A = Matrix::Identity(2, 4);
  msg.b = Vector::Ones(2) * static_cast<double>(epoch);
  msg.checksum = ComputeFactorMessageChecksum(msg);
  return msg;
}

SolutionMessage makeSolutionMessage(int sender, int receiver, uint64_t epoch) {
  SolutionMessage msg;
  msg.sender_robot_id = sender;
  msg.receiver_robot_id = receiver;
  msg.epoch = epoch;
  msg.factor_type = FactorType::TRANSLATION;
  msg.keys = {PoseKey{receiver, 3}};
  msg.x = Vector::Constant(2, static_cast<double>(epoch));
  msg.checksum = ComputeSolutionMessageChecksum(msg);
  return msg;
}

}  // namespace

TEST(testDPGO, TEDCCIAsyncFactorExchangeDropsDuplicateMessages) {
  InProcessAsyncFactorExchange exchange;
  const FactorMessage msg = makeFactorMessage(0, 1, 5);

  exchange.SendFactor(msg);
  exchange.SendFactor(msg);
  const auto received = exchange.PollFactorMessages(1);

  ASSERT_EQ(received.size(), 1u);
  EXPECT_EQ(received.front().epoch, 5u);
  EXPECT_TRUE(exchange.PollFactorMessages(1).empty());
}

TEST(testDPGO, TEDCCIAsyncFactorExchangeNewestEpochWinsOutOfOrder) {
  InProcessAsyncFactorExchange exchange;
  const FactorMessage newer = makeFactorMessage(0, 1, 7);
  const FactorMessage older = makeFactorMessage(0, 1, 3);

  exchange.SendFactor(newer);
  exchange.SendFactor(older);
  const auto received = exchange.PollFactorMessages(1);

  ASSERT_EQ(received.size(), 1u);
  EXPECT_EQ(received.front().epoch, 7u);
  EXPECT_EQ(received.front().b(0), 7.0);
}

TEST(testDPGO, TEDCCIAsyncFactorExchangeQueuesMessagesAcrossLinkDownUp) {
  InProcessAsyncFactorExchange exchange(/*queue_when_link_down=*/true);
  exchange.MarkLinkDown(0, 1);

  exchange.SendFactor(makeFactorMessage(0, 1, 2));
  EXPECT_TRUE(exchange.PollFactorMessages(1).empty());

  exchange.MarkLinkUp(0, 1);
  const auto received = exchange.PollFactorMessages(1);
  ASSERT_EQ(received.size(), 1u);
  EXPECT_EQ(received.front().epoch, 2u);
}

TEST(testDPGO, TEDCCIAsyncFactorExchangeRejectsChecksumMismatch) {
  InProcessAsyncFactorExchange exchange;
  FactorMessage msg = makeFactorMessage(0, 1, 4);
  msg.checksum += 1;

  exchange.SendFactor(msg);

  EXPECT_TRUE(exchange.PollFactorMessages(1).empty());
}

TEST(testDPGO, TEDCCIAsyncSolutionExchangeIsIdempotentAndNewestWins) {
  InProcessAsyncFactorExchange exchange;
  const SolutionMessage older = makeSolutionMessage(0, 1, 1);
  const SolutionMessage newer = makeSolutionMessage(0, 1, 3);

  exchange.SendSolution(older);
  exchange.SendSolution(newer);
  exchange.SendSolution(older);
  exchange.SendSolution(newer);
  const auto received = exchange.PollSolutionMessages(1);

  ASSERT_EQ(received.size(), 1u);
  EXPECT_EQ(received.front().epoch, 3u);
  EXPECT_EQ(received.front().x(0), 3.0);
}

TEST(testDPGO, TEDCCIAsyncFactorExchangeReplaysLatestFactorToNewParentOwner) {
  InProcessAsyncFactorExchange exchange;
  FactorMessage original = makeFactorMessage(0, 1, 9);
  original.keys = {PoseKey{0, 20}, PoseKey{1, 20}};
  original.A = Matrix::Constant(2, 4, 0.5);
  original.b = Vector::Constant(2, 9.0);
  original.checksum = ComputeFactorMessageChecksum(original);

  exchange.SendFactor(original);
  const auto oldParent = exchange.PollFactorMessages(1);
  ASSERT_EQ(oldParent.size(), 1u);

  FactorMessage replay = original;
  replay.receiver_robot_id = 2;
  replay.checksum = ComputeFactorMessageChecksum(replay);
  exchange.SendFactor(replay);

  EXPECT_TRUE(exchange.PollFactorMessages(1).empty());
  const auto newParent = exchange.PollFactorMessages(2);
  ASSERT_EQ(newParent.size(), 1u);
  EXPECT_EQ(newParent.front().sender_robot_id, 0);
  EXPECT_EQ(newParent.front().receiver_robot_id, 2);
  EXPECT_EQ(newParent.front().epoch, 9u);
  EXPECT_EQ(newParent.front().keys, original.keys);
  EXPECT_EQ(newParent.front().A, original.A);
  EXPECT_EQ(newParent.front().b, original.b);
  EXPECT_EQ(newParent.front().checksum,
            ComputeFactorMessageChecksum(newParent.front()));
}

TEST(testDPGO, TEDCCIAsyncFactorExchangeKeepsTopologyEpochViewsSeparate) {
  InProcessAsyncFactorExchange exchange;
  FactorMessage first = makeFactorMessage(0, 1, 12);
  first.topology_epoch = 1;
  first.checksum = ComputeFactorMessageChecksum(first);
  FactorMessage second = first;
  second.topology_epoch = 2;
  second.b = Vector::Constant(2, 20.0);
  second.checksum = ComputeFactorMessageChecksum(second);

  exchange.SendFactor(first);
  exchange.SendFactor(second);
  const auto factorMessages = exchange.PollFactorMessages(1);

  ASSERT_EQ(factorMessages.size(), 2u);
  EXPECT_EQ(factorMessages[0].topology_epoch, 1u);
  EXPECT_EQ(factorMessages[1].topology_epoch, 2u);
  EXPECT_EQ(factorMessages[0].checksum,
            ComputeFactorMessageChecksum(factorMessages[0]));
  EXPECT_EQ(factorMessages[1].checksum,
            ComputeFactorMessageChecksum(factorMessages[1]));

  SolutionMessage solution1 = makeSolutionMessage(0, 1, 12);
  solution1.topology_epoch = 1;
  solution1.checksum = ComputeSolutionMessageChecksum(solution1);
  SolutionMessage solution2 = solution1;
  solution2.topology_epoch = 2;
  solution2.x = Vector::Constant(2, 22.0);
  solution2.checksum = ComputeSolutionMessageChecksum(solution2);

  exchange.SendSolution(solution1);
  exchange.SendSolution(solution2);
  const auto solutionMessages = exchange.PollSolutionMessages(1);

  ASSERT_EQ(solutionMessages.size(), 2u);
  EXPECT_EQ(solutionMessages[0].topology_epoch, 1u);
  EXPECT_EQ(solutionMessages[1].topology_epoch, 2u);
  EXPECT_EQ(solutionMessages[0].checksum,
            ComputeSolutionMessageChecksum(solutionMessages[0]));
  EXPECT_EQ(solutionMessages[1].checksum,
            ComputeSolutionMessageChecksum(solutionMessages[1]));
}


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

#include <cstdlib>
#include <cassert>
#include <iostream>

using namespace std;
using namespace DPGO;

int main(int argc, char **argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc < 3) {
    cout << "Multi-robot pose graph optimization example. " << endl;
    cout << "Usage: " << argv[0] << " [# robots] [input .g2o file]" << endl;
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
  r = 3;
  bool acceleration = false;
  bool verbose = true;
  // unsigned numIters = 1000;
  unsigned numIters = 100;


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
    options.verbose = verbose;

    auto *agent = new PGOAgent(robot, options);

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
  Matrix XChordal =  TChordal; // Lift estimate to the correct relaxation rank
  for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
    unsigned startIdx = robot * num_poses_per_robot;
    unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
    if (robot == (unsigned) num_robots - 1) endIdx = n;
    vector<PoseID> neighbor_poseid=agents[robot]->get_neighborid();
    Matrix initX(r,(endIdx - startIdx+neighbor_poseid.size()) * (d + 1));
    initX.setZero();
    initX.block(0, 0, r, (endIdx - startIdx) * (d + 1))=XChordal.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1));
    for(size_t k=0;k<neighbor_poseid.size();k++){
      unsigned id=neighbor_poseid[k].first;
      unsigned po=neighbor_poseid[k].second;
      initX.block(0,(endIdx - startIdx+k) * (d + 1),r,d+1)=XChordal.block(0,(id*num_poses_per_robot+po)*(d+1),r,d+1);
    }
    agents[robot]->setX(initX);
  }
  
  // // check Q
  //   vector<PoseID> neighbor=agents[0]->get_neighborid();
  //   PoseID last=neighbor.back();
  //   // for(auto i :neighbor)
  //   //   cout<<i.first<<","<<i.second<<endl;
  //   // cout<<last.first<<","<<last.second<<endl;
  //   vector<unsigned> seperator=agents[0]->get_neighbor_seperators(last);
  //   // for(auto i: seperator)
  //   //   cout<<i<<" "<<agents[0]->num_poses()<<" "<<neighbor.size()<<endl;
  //   agents[0]->testQ();


  // check intialization
  // for(int j=0;j<num_robots;j++){
  // vector<PoseID> neighbor=agents[j]->get_neighborid();
  //   for(int i=0;i<neighbor.size();i++){
  //     cout<<neighbor[i].first<<" "<<neighbor[i].second<<endl;
  //     Matrix neighbor_matrix;
  //     agents[neighbor[i].first]->getSharedPose(neighbor[i].second,neighbor_matrix);
  //     Matrix local_matrix;
  //     // cout<<neighbor_matrix<<endl;
  //     unsigned local_nfpose=agents[j]->num_poses();
  //     agents[j]->getSharedPose(local_nfpose+i,local_matrix);
  //     if(local_matrix==neighbor_matrix)
  //       cout<<"true"<<endl;
  //     else
  //       cout<<"false"<<endl;
  //   }
  // }

  //communicate 

 
/** check communicate**/
  // std::map<PoseID,Matrix> test_neighbor=agents[1]->get_X_neighbor();
  // std::map<PoseID,std::map<unsigned,Matrix>> test_seperator=agents[1]->get_X_seperator();
// //check neighbor
  // // for(auto &i: test_neighbor)
  // //   {cout<<"get variable of neighbor "<<i.first.first<<","<<i.first.second<<endl;
  // //   // cout<<i.second<<endl;
  // //   // Matrix mat;
  // //   // agents[i.first.first]->getSharedPose(i.first.second,mat);
  // //   // cout<<mat<<endl;
  // //   }
// // check seperator
  // cout<<test_seperator.size()<<endl;
  // for(auto& kv: test_seperator){
  //   cout<<"get variable of seperator "<<kv.first.first<<","<<kv.first.second<<endl;
  //   // cout<<"in total "<<agents[1]->get_seperator_neighbors(kv.first.second).size()<<endl;
  //   // cout<<kv.second.size();
  //   // for(auto i : agents[0]->get_seperator_neighbors(kv.first.second))
  //   //   cout<<i.first<<","<<i.second<<";";
  //   cout<<"shared"<<endl;
  //   for(auto mat:kv.second){
  //       cout<<mat.second<<endl;
  //   }
  //   // cout<<endl;
  //   Matrix mat;
  //   agents[kv.first.first]->getSharedPose(kv.first.second,mat);
  //   cout<<"local"<<endl<<mat<<endl;
  // }

  std::cout<<std::endl<<"initial cost: "<< 2 * problemCentral.f(XChordal)<<std::endl;
  
  /**
  ###########################################
  Optimization loop
  ###########################################
  */  

  Matrix Xopt(r, n * (d + 1));
  unsigned selectedRobot = 0;
  cout << "Running " << numIters << " iterations..." << endl;
  for (unsigned iter = 0; iter < numIters; ++iter) {
    PGOAgent *selectedRobotPtr = agents[selectedRobot];

    // Selected robot requests public poses from others
    for(auto *robotPtr : agents) {
      if(selectedRobot==robotPtr->getID())
        continue;
      PoseDict sharedPoses;
      if (robotPtr->getSharedPoseDict(sharedPoses)) 
        selectedRobotPtr->update_sharedX(robotPtr->getID(), sharedPoses); 

      std::map<PoseID,Matrix> shared_H;
      if (robotPtr->getShared_H(shared_H)) 
        selectedRobotPtr->update_sharedH(robotPtr->getID(), shared_H); 

      }

    // Non-selected robots perform an iteration
    for (auto *robotPtr : agents) {
      assert(robotPtr->instance_number() == 0);
      assert(robotPtr->iteration_number() == iter);
      if (robotPtr->getID() != selectedRobot) {
        robotPtr->iterate(false);
      }
    }


    // // Selected robot update
    selectedRobotPtr->iterate(true);

    // cout<<"perform once"<<endl;


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
    std::cout << std::setprecision(6)
              << "Iter = " << iter << " | "
              << "robot = " << selectedRobotPtr->getID() << " | "
              << "cost = " << 2 * problemCentral.f(Xopt) << " | "
              << "gradnorm = " << RGradNorm << std::endl<< std::endl;

    // Exit if gradient norm is sufficiently small
    if (RGradNorm < 0.1) {
      break;
    }
    selectedRobot+=1;
    if(selectedRobot==num_robots)
      selectedRobot=0;

    // Share global anchor for rounding
    Matrix M;
    agents[0]->getSharedPose(0, M);
    for (auto agentPtr : agents) {
      agentPtr->setGlobalAnchor(M);
    }
  }

  for (auto agentPtr : agents) {
    agentPtr->reset();
  }

  exit(0);
}

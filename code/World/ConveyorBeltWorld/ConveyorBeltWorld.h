//  MABE is a product of The Hintze Lab @ MSU
//     for general research information:
//         hintzelab.msu.edu
//     for MABE documentation:
//         github.com/Hintzelab/MABE/wiki
//
//  Copyright (c) 2015 Michigan State University. All rights reserved.
//     to view the full license, visit:
//         github.com/Hintzelab/MABE/wiki/License

#pragma once

#include <World/AbstractWorld.h>
#include <World/ConveyorBeltWorld/ConveyorBelt.h>

#include <cstdlib>
#include <thread>
#include <vector>
#include <valarray>
#include <queue>
#include <algorithm>

class ConveyorBeltWorld : public AbstractWorld {

public:
  static std::shared_ptr<ParameterLink<int>> modePL;
  static std::shared_ptr<ParameterLink<int>> numThreadsPL;
  static std::shared_ptr<ParameterLink<int>> trialsPL;
  static std::shared_ptr<ParameterLink<int>> trialLengthPL;

  // int mode;
  // int numberOfOutputs;
  // int evaluationsPerGeneration;

  static std::shared_ptr<ParameterLink<std::string>> groupNamePL;
  static std::shared_ptr<ParameterLink<std::string>> brainNamePL;
  static std::shared_ptr<ParameterLink<std::string>> dimensionsPL;
  static std::shared_ptr<ParameterLink<std::string>> goalRewardsPL;
  static std::shared_ptr<ParameterLink<bool>> randomizePL;
  static int num_trials; // trialsPL
  static int trial_length; // trialLengthPL
  static std::valarray<int> dimensions; // dimensionsPL
  static ConveyorBelt::ConveyorBeltParams* pParameters;
  static std::vector<Puzzle> puzzles;

  ConveyorBeltWorld(std::shared_ptr<ParametersTable> PT_ = nullptr);
  virtual ~ConveyorBeltWorld() = default;

  std::vector<std::shared_ptr<Organism>>* population_ptr;
  std::queue<int> org_ids_to_evaluate; // indexes into population, remove id when evaluated
  std::mutex org_ids_guard; // locks org_ids_to_evaluate
  std::mutex data_guard;
  std::vector<std::thread> threads;

  auto evaluate_single_thread(int analyze, int visualize, int debug) -> void;

  void evaluate(std::map<std::string, std::shared_ptr<Group>> &groups, int analyze, int visualize, int debug);

  virtual std::unordered_map<std::string, std::unordered_set<std::string>> requiredGroups() override;

  // to facilitate multithreading,
  // all world properties and tracking metrics
  // are allocated in separate memory for each agent, called an 'Experience'
  class Experience {
    public:
      ConveyorBelt env;
      std::valarray<int> numsteps;
      std::valarray<int> numsolved;
      Experience();
      Experience(const ConveyorBelt::ConveyorBeltParams& params) : env(params) {
      };
      void reset() {
        numsteps = 0;
        numsolved = 0;
      }
  };
  std::vector<Experience> experiences;

  // capitalize strings
  auto capitalize(const std::string& str) -> std::string {
    std::string result = str;
    std::transform(std::begin(str), std::end(str), std::begin(result), ::toupper);
    return result;
  }
};


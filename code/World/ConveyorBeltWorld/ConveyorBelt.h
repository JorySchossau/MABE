#pragma once
#include <stdint.h>
#include <valarray>
#include <vector>
#include <tuple> // tuple, tie, make_tuple
#include <algorithm> // clamp
#include <cassert>

using namespace std;
typedef tuple<valarray<int> /*state*/, double /*reward*/, bool /*goal-found*/> step_t;
typedef valarray<int> state_t;

class Puzzle {
  public:
    vector<valarray<int>> features;
    vector<double>
      rewards,
      originalRewards; // original copy of rewards (auto set)
    bool
      solvable, // if a one-time reward is attainable
      solved; // status (useful to know when all puzzles are solved)
    int
      state,
      startState;
    vector<vector<int>> transitions; // state-to-state transition matrix
};

class ConveyorBelt {
  public:
    vector<Puzzle> puzzles;
    vector<double> goalRewards; // goals (good & bad)
    bool randomize; // whether to present items in linear or random order
    Puzzle* currentPuzzle; // point to current puzzle
    int currentPuzzleIndex;
    int numPuzzlesToSolve;
    struct ConveyorBeltParams {
      vector<double> goalRewards;
      bool randomize;
    };
    ConveyorBelt(const ConveyorBeltParams& /*params*/);
    auto reset(void) -> state_t;
    auto step(const int& /*action*/) -> step_t;
    void pass(void);
    struct PuzzleParams {
      vector<valarray<int>> features;
      vector<double> rewards;
      bool solvable;
      int startState;
      vector<vector<int>> transitions;
    };
    void push_back(Puzzle& /*newPuzzle*/);
    void push_back(const PuzzleParams& /*params*/);
    
    template <class Iterable, typename T = typename Iterable::value_type>
    auto contains(const Iterable& iterable, const T& value) -> decltype(std::begin(iterable), std::end(iterable), bool{} ) {
      for (auto& element : iterable) {
        if (value == element)
          return true;
      }
      return false;
    }
};


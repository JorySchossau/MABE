#pragma once
#include <stdint.h>
#include <valarray>
#include <vector>
#include <tuple> // tuple, tie, make_tuple
#include <algorithm> // clamp
#include <cassert>
#include "TDUtils.h"

using namespace std;
using namespace TD;
typedef tuple<valarray<int> /*state*/, double /*reward*/, bool /*goal-found*/> step_t;

class Puzzle {
  public:
    vector<vector<int>> transitions; // state-to-state transition matrix
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
};

class ConvBelt {
  public:
    vector<Puzzle> puzzles;
    vector<double> goalRewards; // goals (good & bad)
    bool randomize; // whether to present items in linear or random order
    Puzzle* currentPuzzle; // point to current puzzle
    int currentPuzzleIndex;
    int numPuzzlesToSolve;
    struct ConvBeltParams {
      const vector<double>& goalRewards;
      const bool& randomize;
    };
    ConvBelt(const ConvBeltParams& /*params*/);
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
};

ConvBelt::ConvBelt(const ConvBeltParams& params)
: goalRewards(params.goalRewards), randomize(params.randomize), currentPuzzleIndex(0), currentPuzzle(nullptr), numPuzzlesToSolve(0) {
//: ConvBelt::ConvBelt(params.goalRewards, params.randomize) {
}

/* reset all puzzles to original reward states
 * and reset number of puzzles to solve */
auto ConvBelt::reset() -> state_t {
  numPuzzlesToSolve = 0;
  for (Puzzle& puzzle : puzzles) {
    puzzle.rewards = puzzle.originalRewards;
    puzzle.state = puzzle.startState;
    puzzle.solved = false;
    numPuzzlesToSolve += puzzle.solvable;
  }
  currentPuzzleIndex = randomize ? Random::getIndex(puzzles.size()) : 0;
  currentPuzzle = &puzzles[currentPuzzleIndex];
  return currentPuzzle->features[currentPuzzle->state];
}

auto ConvBelt::step(const int& action) -> step_t {
  double reward;
  if (action == 1)
    pass();
  else
    currentPuzzle->state = currentPuzzle->transitions[currentPuzzle->state][action];
  if (contains(goalRewards, currentPuzzle->rewards[currentPuzzle->state])) {
    reward = currentPuzzle->rewards[currentPuzzle->state];
    currentPuzzle->rewards[currentPuzzle->state] = -1; // remove food
    --numPuzzlesToSolve;
  }
  else
    reward = currentPuzzle->rewards[currentPuzzle->state];
 
  return make_tuple(/* new world state*/ currentPuzzle->features[currentPuzzle->state],
                    /* reward         */ reward,
                    /* goal reached   */ bool(numPuzzlesToSolve == 0));
}

void ConvBelt::pass() {
  ++currentPuzzleIndex;
  currentPuzzleIndex %= puzzles.size();
  currentPuzzle = &puzzles[currentPuzzleIndex];
}

void ConvBelt::push_back(Puzzle& newPuzzle) {
  puzzles.push_back(newPuzzle);
  end(puzzles)[-1].originalRewards = end(puzzles)[-1].rewards;
  end(puzzles)[-1].state = end(puzzles)[-1].startState;
  if (newPuzzle.solvable)
    ++currentPuzzleIndex;
}

void ConvBelt::push_back(const ConvBelt::PuzzleParams& params) {
  Puzzle puzzle;
  puzzle.features = params.features;
  puzzle.rewards = params.rewards;
  puzzle.solvable = params.solvable;
  puzzle.startState = params.startState;
  puzzle.transitions = params.transitions;
  puzzle.originalRewards = params.rewards;
  puzzle.state = puzzle.startState;
  puzzles.push_back(puzzle);
  if (puzzle.solvable)
    ++currentPuzzleIndex;
}

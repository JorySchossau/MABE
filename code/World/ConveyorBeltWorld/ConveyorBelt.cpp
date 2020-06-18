#include "ConveyorBelt.h"
#include <Utilities/Random.h>
#include <iostream>

ConveyorBelt::ConveyorBelt(const ConveyorBeltParams& params)
: goalRewards(params.goalRewards), randomize(params.randomize), currentPuzzleIndex(0), currentPuzzle(nullptr), numPuzzlesToSolve(0) {
//: ConveyorBelt::ConveyorBelt(params.goalRewards, params.randomize) {
}

/* reset all puzzles to original reward states
 * and reset number of puzzles to solve */
auto ConveyorBelt::reset() -> state_t {
  numPuzzlesToSolve = 0;
  for (Puzzle& puzzle : puzzles) {
    puzzle.rewards = puzzle.originalRewards;
    puzzle.state = puzzle.startState;
    puzzle.solved = false;
    numPuzzlesToSolve += puzzle.solvable;
  }
  if (puzzles.size() == 0) {std::cout << "Error (ConveyorBeltWorld): no puzzles in the conveyor belt." << std::endl; exit(1);}
  currentPuzzleIndex = randomize ? Random::getIndex(puzzles.size()) : 0;
  currentPuzzle = &puzzles[currentPuzzleIndex];
  return currentPuzzle->features[currentPuzzle->state];
}

auto ConveyorBelt::step(const int& action) -> step_t {
  double reward;
  if (action == 1)
    pass();
  else
    currentPuzzle->state = currentPuzzle->transitions[currentPuzzle->state][action];
  if (contains(goalRewards, currentPuzzle->rewards[currentPuzzle->state]) and currentPuzzle->solvable) {
    reward = currentPuzzle->rewards[currentPuzzle->state];
    currentPuzzle->rewards[currentPuzzle->state] = -1; // remove food
    currentPuzzle->solved = true;
    --numPuzzlesToSolve;
  }
  else
    reward = currentPuzzle->rewards[currentPuzzle->state];
 
  return make_tuple(/* new world state*/ currentPuzzle->features[currentPuzzle->state],
                    /* reward         */ reward,
                    /* goal reached   */ bool(numPuzzlesToSolve == 0));
}

void ConveyorBelt::pass() {
  ++currentPuzzleIndex;
  currentPuzzleIndex %= puzzles.size();
  currentPuzzle = &puzzles[currentPuzzleIndex];
}

void ConveyorBelt::push_back(Puzzle& newPuzzle) {
  puzzles.push_back(newPuzzle);
  end(puzzles)[-1].originalRewards = end(puzzles)[-1].rewards;
  end(puzzles)[-1].state = end(puzzles)[-1].startState;
  if (newPuzzle.solvable)
    ++currentPuzzleIndex;
}

void ConveyorBelt::push_back(const ConveyorBelt::PuzzleParams& params) {
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

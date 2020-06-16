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

class Grid {
  public:
    int
      height,
      width;
    state_t
      state,
      startState;
    vector<state_t>
      &holes,
      &goals;
    Grid(state_t dims, vector<state_t>& goals, vector<state_t>& holes, state_t startState);
    auto reset() -> state_t;
    auto step(const int& action) -> step_t;
};

Grid::Grid(state_t dims, vector<state_t>& goals, vector<state_t>& holes, state_t startState={0,0})
: height(dims[0]), width(dims[1]), startState(startState), state(startState), holes(holes), goals(goals) {
}

auto Grid::reset() -> state_t {
  state = startState;
  return state;
}

auto Grid::step(const int& action) -> step_t {
  double reward(-1);
  bool goalFound(false);
  state_t delta{0,0};
  switch (action) {
    case 0: delta[0] = -1; break; // N
    case 2: delta[0] =  1; break; // S
    case 1: delta[1] =  1; break; // E
    case 3: delta[1] = -1; break; // W
    default: assert(action < 4);
  };
  state_t newState = state + delta;
  newState[0] = clamp(newState[0], 0, height-1);
  newState[1] = clamp(newState[1], 0, width-1);
  state = newState;
  // check for goal
  if (containsarray(goals, state)) {
    goalFound = true;
    reward = 0.0;
  } else if (containsarray(holes, state)) {
    reward = -100.0;
  }
  return make_tuple(state, reward, goalFound);
}

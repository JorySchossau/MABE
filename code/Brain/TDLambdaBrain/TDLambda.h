#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <valarray>
#include <algorithm>
#include <functional>

#include "Binner.h" // dimensionality remapping high-to-low (also called 'tiling')
#include "TDUtils.h"

namespace TD {
  using namespace std;

  struct LambdaParams {
    valarray<int> dims;
    int n_features, n_actions;
    double alpha, gamma, epsilon, lmbdaG, lmbdaB, lmbdaN;
  };
  auto str(const LambdaParams& params) -> string {
    stringstream ss;
    ss << "n_features: " << params.n_features << endl;
    ss << "n_actions: " << params.n_actions << endl;
    ss << "alpha: " << params.alpha << endl;
    ss << "gamma: " << params.gamma << endl;
    ss << "epsilon: " << params.epsilon << endl;
    ss << "lmbdaG: " << params.lmbdaG << endl;
    ss << "lmbdaB: " << params.lmbdaB << endl;
    ss << "lmbdaN: " << params.lmbdaN << endl;
    return ss.str();
  }
  
  class Lambda {
    public:
      LambdaParams params;
      valarray<int> allActions,
                    sensoryState,
                    previousSensoryState;
      valarray<double> weights,
                       tracesG, tracesB, tracesN;
      int action, episodeN, confidence /*time since big surprise*/;
      double reward;
      Binner bin;

           Lambda(const LambdaParams& /*params*/);
      void reset();
      auto featurizeState(valarray<int> /*state*/) -> int;
      auto predictPayoffsForAllActions(valarray<int> /*sensoryState*/) -> valarray<double>;
      void plasticUpdate(); // compute action, with learning
      void staticUpdate();  // compute action, no learning
  };

  Lambda::Lambda(const LambdaParams& params)
    : bin(Binner(params.dims)),
      params(params) {
    allActions.resize(params.n_actions);
    iota(begin(allActions), end(allActions), 0);
    const int statespaceSize = prod(params.dims);
    // full state is sensory state + action state, // so sensoryState is all but the final (action) state
    sensoryState.resize(params.dims.size()-1); 
    previousSensoryState.resize(params.dims.size()-1);
    weights.resize(statespaceSize);
    tracesG.resize(statespaceSize);
    tracesB.resize(statespaceSize);
    tracesN.resize(statespaceSize);
    reset();
  }

  void Lambda::reset() {
    action = 0;
    reward = -1.0;
    confidence = 0;
    // zero-out the arrays
    sensoryState = 0;
    previousSensoryState = 0;
    tracesG = 0;
    tracesB = 0;
    tracesN = 0;
    end(weights)[-1] = 1.0;
  }

  auto Lambda::featurizeState(valarray<int> state) -> int { return bin[state]; }

  auto Lambda::predictPayoffsForAllActions(valarray<int> sensoryState) -> valarray<double> {
    valarray<double> result(params.n_actions);
    vector<valarray<bool>> featuresForEachAction(params.n_actions,valarray<bool>(params.n_features));
    valarray<int> fullState(params.dims.size());
    int end = params.dims.size()-1;
    fullState[slice(0,end,1)] = sensoryState; // slice is exclusive upper-bound
    for (int& action : allActions) { // actions are always linear increasing 0,1,2...
      fullState[end] = action;
      // TODO set multiple features to true if featurizeState returns multiple internal features (future work) instead of a single one
      // for now, we assume only single internal features
      featuresForEachAction[action][featurizeState(fullState)] = true;
      //cout << str(weights) << endl;
      //cout << str(featuresForEachAction[action]) << endl;
      //cout << endl;
      result[action] = valarray<double>(weights[featuresForEachAction[action]]).sum();
    }
    return result;
  }

  /* perform a computation update, and learn from previous feedback */
  void Lambda::plasticUpdate() {
    double
      nextActionPredictedPayoff(0.0),
      previousActionCorrectedPayoff(0.0),
      previousActionPredictedPayoff(0.0),
      surprise(0.0);
    int nextAction(0);

    // perform action-selection and predict payoff
    // if positive reward, limit predicted payoff to 0
    if (reward >= 0.0) {
      nextActionPredictedPayoff = 0;
    }
    // if negative reward, then perform epsilon-greedy action-selection
    else {
      // if within epsilon, select random
      // anneal probability over time since last surprising experience
      if ( Random::getDouble(1.0) < (1.0/(200+confidence)) ) {
        nextAction = choice(allActions);
      }
      // if outside epsilon, select best action
      else {
        valarray<double> q_vals = predictPayoffsForAllActions(sensoryState);
        nextAction = argmax(q_vals);
        nextActionPredictedPayoff = q_vals[nextAction];
      }
    }

    // determine the corrected payoff version given the reward actually received
    previousActionCorrectedPayoff = reward + (nextActionPredictedPayoff * params.gamma);

    // prepare full internal state (sensory + action)
    valarray<bool> features(params.n_features);
    valarray<int> fullState(params.dims.size());
    fullState[slice(0,params.dims.size()-1,1)] = previousSensoryState;
    end(fullState)[-1] = action;
    features[featurizeState(fullState)] = true; // TODO could expand this to more than 1 internal feature representation
    previousActionPredictedPayoff = valarray<double>(weights[features]).sum();
    surprise = previousActionCorrectedPayoff - previousActionPredictedPayoff;

    // reset confidence to 0 if sufficiently surprised
    // TODO this is an arbitrary threshold, it would be great to reset annealing scaled by surprise
    if (abs(surprise) > 4) confidence = 0;
    else                   ++confidence;

    // reset eligibility traces for contemporaneous features
    tracesG[features] = 1.0;
    tracesB[features] = 1.0;
    tracesN[features] = 1.0;

    // do weight updates depending on reward value location in spans defined by boundaries [-inf,-1,0,inf] = {Bad,Neutral,Good} spans
    if      (reward >   0) weights += params.alpha * surprise * tracesG;
    else if (reward >= -1) weights += params.alpha * surprise * tracesN;
    else                   weights += params.alpha * surprise * tracesB;

    // do trace updates
    tracesG *= params.lmbdaG;
    tracesB *= params.lmbdaB;
    tracesN *= params.lmbdaN;

    // step the storage of state and action in memory
    previousSensoryState = sensoryState;
    action = nextAction;
  }

  void Lambda::staticUpdate() {
    // same as plastic update, but without learning (a.k.a. 'deployment')

    double nextActionPredictedPayoff(0.0);
    int nextAction(0);

    // greedy action-selection
    valarray<double> q_vals = predictPayoffsForAllActions(sensoryState);
    nextAction = argmax(q_vals);

    // step the storage of state and action in memory
    previousSensoryState = sensoryState;
    action = nextAction;
  }

}

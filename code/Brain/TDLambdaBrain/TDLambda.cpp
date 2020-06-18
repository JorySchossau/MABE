#include <Brain/TDLambdaBrain/TDLambda.h>
#include <vector>

namespace TD {
  using namespace std;
  Binner::Binner() { }
  template <class T>
  Binner::Binner(const T& dims) {
    hash_vec = dims;
    for (int i=0; i<dims.size(); i++) { // note: i<size(), NOT: i<size()-1, because of how slice() works
      valarray<int> subarray = dims[slice(0,i,1)];
      hash_vec[i] = prod(subarray);
    }
  }
  template <class T>
  auto Binner::operator[] (const T& multidim_point) -> int {
    return (multidim_point*hash_vec).sum();
  }
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
  Lambda::Lambda() {};
  Lambda::Lambda(const LambdaParams& params)
    : params(params) {
    bin = Binner(params.dims);
    allActions.resize(params.n_actions);
    iota(begin(allActions), end(allActions), 0);
    const int statespaceSize = prod(params.dims);
    // full state is sensory state + action state, // so sensoryState is all but the final (action) state
    sensoryState.resize(params.dims.size()-1); 
    sensoryState = 0;
    previousSensoryState.resize(params.dims.size()-1);
    previousSensoryState = 0;
    weights.resize(statespaceSize);
    weights = 0;
    originalWeights.resize(statespaceSize);
    originalWeights = 0;
    tracesG.resize(statespaceSize);
    tracesB.resize(statespaceSize);
    tracesN.resize(statespaceSize);
    tracesG = 0;
    tracesB = 0;
    tracesN = 0;
    action = 0;
    reward = -1.0;
    confidence = 0;
  }

  void Lambda::reset() {
    tracesG = 0;
    tracesB = 0;
    tracesN = 0;
    sensoryState = 0;
    previousSensoryState = 0;
    action = 0;
    reward = -1.0;
    // note: do NOT reset confidence
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
      featuresForEachAction[action] = false;
      featuresForEachAction[action][featurizeState(fullState)] = true;
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
        //if ( Random::getDouble(1.0) < (1.0/double(200+confidence)) ) {
      if (use_confidence) {
        if ( Random::P(1.0/double(200+confidence)) ) {
          nextAction = choice(allActions);
        }
        else { // this is redundant with the below inner-else
          valarray<double> q_vals = predictPayoffsForAllActions(sensoryState);
          nextAction = argmax(q_vals);
          nextActionPredictedPayoff = q_vals[nextAction];
        }
      }
      else {
        //if ( Random::getDouble(1.0) < no_confidence_random_action ) {
        if ( Random::P(no_confidence_random_action) ) {
          nextAction = choice(allActions);
        }
        else { // this is redundant with the above inner-else
          valarray<double> q_vals = predictPayoffsForAllActions(sensoryState);
          nextAction = argmax(q_vals);
          nextActionPredictedPayoff = q_vals[nextAction];
        }
      }
      // if outside epsilon, select best action
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
    if (abs(surprise) > 4)  confidence = 0;
    else                    ++confidence;

    // reset eligibility traces for contemporaneous features
    tracesG[features] = 1.0;
    tracesB[features] = 1.0;
    tracesN[features] = 1.0;

    // do weight updates depending on reward value location in spans defined by boundaries [-inf,-1,0,inf] = {Bad,Neutral,Good} spans
    double before = weights.sum();
    if      (reward >=  0) weights += (params.alpha * surprise * tracesG);
    else if (reward >= -1) weights += (params.alpha * surprise * tracesN);
    else                   weights += (params.alpha * surprise * tracesB);
    double after = weights.sum();

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

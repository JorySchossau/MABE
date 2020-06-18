#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <valarray>
#include <algorithm>
#include <functional>

#include "TDUtils.h"

namespace TD {
  using namespace std;

  class Binner {
    public:
      Binner();
      template <class T>
      Binner(const T& /*dims*/);
      //template <class T>
      //auto operator[] (const T& /*multidim_point*/) -> int;
      template <class T>
      int operator[] (const T& /*multidim_point*/);

      // properties
      valarray<int> hash_vec;
  };

  struct LambdaParams {
    valarray<int> dims;
    int n_features, n_actions;
    double alpha, gamma, epsilon, lmbdaG, lmbdaB, lmbdaN;
  };

  auto str(const LambdaParams& params) -> string;
  
  class Lambda {
    public:
      LambdaParams params;
      valarray<int>
        allActions,
        sensoryState,
        previousSensoryState;
      valarray<double>
        weights, originalWeights,
        tracesG, tracesB, tracesN;
      int action, episodeN, confidence /*time since big surprise*/;
      bool use_confidence;
      double no_confidence_random_action;
      double reward;
      Binner bin;
           Lambda();
           Lambda(const LambdaParams& /*params*/);
      void reset();
      auto featurizeState(valarray<int> /*state*/) -> int;
      auto predictPayoffsForAllActions(valarray<int> /*sensoryState*/) -> valarray<double>;
      void plasticUpdate(); // compute action, with learning
      void staticUpdate();  // compute action, no learning
  };


}

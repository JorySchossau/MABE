#pragma once

#include <valarray>
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
}

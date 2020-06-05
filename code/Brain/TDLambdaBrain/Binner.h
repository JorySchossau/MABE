#include <valarray>
#include "TDUtils.h"

namespace TD {
  using namespace std;
  class Binner {
    public:
      Binner() = delete; // this does not work :( we tried

      template <class T>
      Binner(T&& /*dims*/);
      template <class T>
      auto operator[] (T&& /*multidim_point*/) -> int;

      // properties
      valarray<int> hash_vec;
  };

  template <class T>
  Binner::Binner(T&& dims) {
    hash_vec = dims;
    for (int i=0; i<dims.size(); i++) { // note: i<size(), NOT: i<size()-1, because of how slice() works
      valarray<int> subarray = dims[slice(0,i,1)];
      hash_vec[i] = prod(subarray);
    }
  }

  template <class T>
  auto Binner::operator[] (T&& multidim_point) -> int {
    return (multidim_point*hash_vec).sum();
  }
}

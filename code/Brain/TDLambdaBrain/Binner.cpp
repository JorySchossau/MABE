#include <Brain/TDLambdaBrain/Binner.h>

namespace TD {
  Binner::Binner() { }
  template <class T>
  Binner::Binner(const T& dims) {
    hash_vec = dims;
    for (int i=0; i<dims.size(); i++) { // note: i<size(), NOT: i<size()-1, because of how slice() works
      valarray<int> subarray = dims[slice(0,i,1)];
      hash_vec[i] = prod(subarray);
    }
  }

  //template <class T>
  //auto Binner::operator[] (const T& multidim_point) -> int {
  //  return (multidim_point*hash_vec).sum();
  //}
  template <class T>
  int Binner::operator[] (const T& multidim_point) {
    return (multidim_point*hash_vec).sum();
  }
}

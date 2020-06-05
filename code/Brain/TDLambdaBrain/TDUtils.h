#pragma once
#include <sstream>
#include <functional>
#include <algorithm>
#include <Utilities/Random.h>
namespace TD {
  using namespace std;
  template <typename Iterable>
  auto str(Iterable& iterable) -> decltype(begin(iterable), end(iterable), std::string{}) {
    stringstream ss;
    ss << "[";
    for (int i=0; i<iterable.size()-1; i++) {
      ss << iterable[i] << ",";
    }
    ss << iterable[iterable.size()-1] << "]";
    return ss.str();
  }

  // product of all elements
  template <class Iterable, typename T = typename Iterable::value_type>
  auto prod(const Iterable iterable) -> decltype(begin(iterable), end(iterable), T{} ) {
    return accumulate( begin(iterable), end(iterable), (T)1, multiplies<T>() );
  }

  // like py random.choice
  template <class Iterable, typename T = typename Iterable::value_type>
  auto choice(const Iterable iterable) -> decltype(begin(iterable), end(iterable), T{} ) {
    return iterable[ Random::getIndex(iterable.size()) ];
  }

  // like py argmax
  template <class Iterable, typename T = typename Iterable::value_type>
  auto argmax(const Iterable iterable) -> decltype(begin(iterable), end(iterable), int{} ) {
    return static_cast<int>(
      distance(
        begin(iterable),
        max_element(
          begin(iterable),
          end(iterable)
          )
        )
      );
  }
}

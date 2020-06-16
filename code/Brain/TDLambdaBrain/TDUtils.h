#pragma once
#include <sstream>
#include <functional>
#include <algorithm>
#include <Utilities/Random.h>
#include <fstream>
namespace TD {
  using namespace std;
  typedef valarray<int> state_t;

  // turn iterable into a string
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

  // ostream override for iterable
  template <class Iterable, typename T = typename Iterable::value_type>
  ostream& operator<<(ostream& os, Iterable& container) {
    for (auto& element : container)
      os << element << ",";
    return os;
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

  // is container an element of another container
  template <class Iterable, typename T = typename Iterable::value_type>
  auto containsarray(const Iterable& iterable, const T& value) -> decltype(begin(iterable), end(iterable), bool{} ) {
    bool found;
    for (auto& element : iterable) {
      found = true;
      for (int i(element.size()-1); i>=0; --i)
        found &= (element[i] == value[i]);
      if (found)
        return true;
    }
    return false;
  }

  template <class Iterable, typename T = typename Iterable::value_type>
  auto contains(const Iterable& iterable, const T& value) -> decltype(begin(iterable), end(iterable), bool{} ) {
    for (auto& element : iterable) {
      if (value == element)
        return true;
    }
    return false;
  }

  // write iterable to file
  template <class Iterable, typename T = typename Iterable::value_type>
  void write(const char filename[], const Iterable& container) {
    ofstream out(filename,ofstream::out);
    for (auto& item : container)
      out << item << endl;
    out.close();
  }
}

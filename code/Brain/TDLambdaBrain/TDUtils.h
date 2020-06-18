#pragma once
#include <sstream>
#include <functional>
#include <algorithm>
#include <Utilities/Random.h>
#include <fstream>
#include <ctype.h>
#include <vector>
#include <array>
#include <valarray>
namespace TD {
  typedef std::valarray<int> state_t;

  // turn iterable into a string
  template <typename Iterable>
  auto str(Iterable& iterable) -> decltype(std::begin(iterable), std::end(iterable), std::string{}) {
    std::stringstream ss;
    ss << "[";
    for (int i=0; i<iterable.size()-1; i++) {
      ss << iterable[i] << ",";
    }
    ss << iterable[iterable.size()-1] << "]";
    return ss.str();
  }

  // std::ostream override for iterable
  //template <class Iterable, typename T = typename Iterable::value_type>
  //std::ostream& operator<<(std::ostream& os, Iterable& container) {
  //  os << "[";
  //  for (int i(0); i<container.size()-1; i++)
  //    os << container[i] << ",";
  //  os << end(container)[-1] << "]";
  //  return os;
  //}
  //template <class T>
  //std::ostream& operator<<(std::ostream& os, std::vector<T>& container) {
  //  os << "[";
  //  for (int i(0); i<container.size()-1; i++)
  //    os << container[i] << ",";
  //  os << end(container)[-1] << "]";
  //  return os;
  //}
  //template <class T>
  //std::ostream& operator<<(std::ostream& os, std::array<T>& container) {
  //  os << "[";
  //  for (int i(0); i<container.size()-1; i++)
  //    os << container[i] << ",";
  //  os << end(container)[-1] << "]";
  //  return os;
  //}
  //template <>
  //std::ostream& operator<<(std::ostream& os, std::array<char>& container) {
  //  os << container;
  //  return os;
  //}
  //template <class T>
  //std::ostream& operator<<(std::ostream& os, std::valarray<T>& container) {
  //  os << "[";
  //  for (int i(0); i<container.size()-1; i++)
  //    os << container[i] << ",";
  //  os << end(container)[-1] << "]";
  //  return os;
  //}

  // product of all elements
  template <class Iterable, typename T = typename Iterable::value_type>
  auto prod(const Iterable iterable) -> decltype(std::begin(iterable), std::end(iterable), T{} ) {
    return std::accumulate( std::begin(iterable), std::end(iterable), (T)1, std::multiplies<T>() );
  }

  // like py random.choice
  template <class Iterable, typename T = typename Iterable::value_type>
  auto choice(const Iterable iterable) -> decltype(std::begin(iterable), std::end(iterable), T{} ) {
    return iterable[ Random::getIndex(iterable.size()) ];
  }

  // like py argmax
  template <class Iterable, typename T = typename Iterable::value_type>
  auto argmax(const Iterable iterable) -> decltype(std::begin(iterable), std::end(iterable), int{} ) {
    return static_cast<int>(
      std::distance(
        std::begin(iterable),
        std::max_element(
          std::begin(iterable),
          std::end(iterable)
          )
        )
      );
  }

  // is container an element of another container
  template <class Iterable, typename T = typename Iterable::value_type>
  auto containsarray(const Iterable& iterable, const T& value) -> decltype(std::begin(iterable), std::end(iterable), bool{} ) {
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
  auto contains(const Iterable& iterable, const T& value) -> decltype(std::begin(iterable), std::end(iterable), bool{} ) {
    for (auto& element : iterable) {
      if (value == element)
        return true;
    }
    return false;
  }

  // write iterable to file
  template <class Iterable, typename T = typename Iterable::value_type>
  void write(const char filename[], const Iterable& container) {
    std::ofstream out(filename,std::ofstream::out);
    for (auto& item : container)
      out << item << std::endl;
    out.close();
  }

  // capitalize strings
  inline
  auto capitalize(const std::string& str) -> std::string {
    std::string result = str;
    transform(std::begin(str), std::end(str), std::begin(result), toupper);
    return result;
  }
}

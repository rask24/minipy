#ifndef ARRAY_OPS_HPP
#define ARRAY_OPS_HPP

#include <vector>

namespace minipy {
class ArrayOps {
 public:
  static std::vector<double> add(const std::vector<double>& a,
                                 const std::vector<double>& b);
};
}  // namespace minipy

#endif  // ARRAY_OPS_HPP

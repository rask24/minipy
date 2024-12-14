#include "array_ops.hpp"

#include <stdexcept>

namespace minipy {
std::vector<double> ArrayOps::add(const std::vector<double>& a,
                                  const std::vector<double>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Arrays must have the same size");
  }

  std::vector<double> result(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}
}  // namespace minipy

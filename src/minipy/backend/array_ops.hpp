#ifndef ARRAY_OPS_HPP
#define ARRAY_OPS_HPP

#include <algorithm>
#include <thread>
#include <vector>

namespace minipy {
class ArrayOps {
public:
  static std::vector<double> add_cpu(const std::vector<double> &a,
                                     const std::vector<double> &b);
  static std::vector<double> add_gpu(const std::vector<double> &a,
                                     const std::vector<double> &b);

private:
  static void add_range(const std::vector<double> &a,
                        const std::vector<double> &b,
                        std::vector<double> &result, size_t start, size_t end);

  static size_t determine_thread_count(size_t data_size);
};
} // namespace minipy

#endif // ARRAY_OPS_HPP

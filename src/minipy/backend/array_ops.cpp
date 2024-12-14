#include "array_ops.hpp"

#include <algorithm>
#include <stdexcept>

namespace minipy {
void ArrayOps::add_range(const std::vector<double>& a,
                         const std::vector<double>& b,
                         std::vector<double>& result, size_t start,
                         size_t end) {
  constexpr size_t block_size = 64 / sizeof(double);
  for (size_t i = start; i < end; i += block_size) {
    size_t block_end = std::min(i + block_size, end);
    for (size_t j = i; j < block_end; ++j) {
      result[j] = a[j] + b[j];
    }
  }
}

size_t ArrayOps::determine_thread_count(size_t data_size) {
  size_t hw_threads = std::thread::hardware_concurrency();
  if (data_size < 1000) {
    return 1;
  }
  size_t optimal_threads = std::min(hw_threads, data_size / 1000);
  return std::max(size_t(1), optimal_threads);
}

std::vector<double> ArrayOps::add(const std::vector<double>& a,
                                  const std::vector<double>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Arrays must have the same size");
  }

  size_t num_threads = determine_thread_count(a.size());
  std::vector<double> result(a.size());
  std::vector<std::thread> threads;
  size_t chunk_size = a.size() / num_threads;

  threads.reserve(num_threads);
  for (size_t i = 0; i < num_threads - 1; ++i) {
    size_t thread_start = i * chunk_size;
    size_t thread_end = (i + 1) * chunk_size;
    threads.emplace_back(add_range, std::ref(a), std::ref(b), std::ref(result),
                         thread_start, thread_end);
  }

  add_range(a, b, result, (num_threads - 1) * chunk_size, a.size());

  for (auto& thread : threads) {
    thread.join();
  }

  return result;
}
}  // namespace minipy

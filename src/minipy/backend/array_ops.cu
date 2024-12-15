#include <cuda_runtime.h>

#include <stdexcept>

#include "array_ops.hpp"

namespace minipy {
__global__ void add_kernel(const double *a, const double *b, double *result,
                           size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] + b[idx];
  }
}

std::vector<double> ArrayOps::add_gpu(const std::vector<double> &a,
                                      const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Arrays must have the same size");
  }

  size_t size = a.size();
  std::vector<double> result(size);

  double *d_a, *d_b, *d_result;
  cudaMalloc(&d_a, size * sizeof(double));
  cudaMalloc(&d_b, size * sizeof(double));
  cudaMalloc(&d_result, size * sizeof(double));

  cudaMemcpy(d_a, a.data(), size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size * sizeof(double), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);

  cudaMemcpy(result.data(), d_result, size * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);

  return result;
}
}  // namespace minipy

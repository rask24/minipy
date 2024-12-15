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

__global__ void dot_kernel(const double *a, const double *b, double *result,
                           size_t size) {
  __shared__ double shared_mem[256];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  double sum = 0.0;
  if (idx < size) {
    sum = a[idx] * b[idx];
  }

  shared_mem[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_mem[tid] += shared_mem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, shared_mem[0]);
  }
}

std::vector<double> ArrayOps::dot_gpu(const std::vector<double> &a,
                                      const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Arrays must have the same size");
  }

  size_t size = a.size();
  double *d_a, *d_b, *d_result;
  cudaMalloc(&d_a, size * sizeof(double));
  cudaMalloc(&d_b, size * sizeof(double));
  cudaMalloc(&d_result, sizeof(double));

  cudaMemcpy(d_a, a.data(), size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, sizeof(double));

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);

  double host_result;
  cudaMemcpy(&host_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);

  return std::vector<double>{host_result};
}
}  // namespace minipy

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../cuda_check.cuh"
#include "../utils.hpp"

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_M>
__global__ void shared_mem_1D_block_tiling_1D_TB(int M, int N, int K, float alpha,
                                                 const float *A, const float *B, float beta,
                                                 float *C)
{
  const uint tile_row = blockIdx.y;
  const uint tile_col = blockIdx.x;

  const int local_col = threadIdx.x % BLOCK_SIZE_N; // mapping to the output tile matrix
  const int local_row = threadIdx.x / BLOCK_SIZE_N;
    // these are mapping to the output tile matrix which will be (BLOCK_SIZE_M x BLOCK_SIZE_N)

  __shared__ float A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K];
  __shared__ float B_shared[BLOCK_SIZE_K * BLOCK_SIZE_N];

  A += tile_row * BLOCK_SIZE_M * K;
  B += tile_col * BLOCK_SIZE_N;
  C += tile_row * BLOCK_SIZE_M * N + tile_col * BLOCK_SIZE_N;

  const uint a_local_col = threadIdx.x % BLOCK_SIZE_K;
  const uint a_local_row = threadIdx.x / BLOCK_SIZE_K;
  const uint b_local_col = threadIdx.x % BLOCK_SIZE_N;
  const uint b_local_row = threadIdx.x / BLOCK_SIZE_N;

  float sum[THREAD_M] = {0.0f};

  for (uint tile_k = 0; tile_k < K; tile_k += BLOCK_SIZE_K)
  {
    A_shared[a_local_row * BLOCK_SIZE_K + a_local_col] = A[a_local_row * K + a_local_col];
    B_shared[b_local_row * BLOCK_SIZE_N + b_local_col] = B[b_local_row * N + b_local_col];
    __syncthreads();

    // Advance blocktile
    A += BLOCK_SIZE_K;
    B += BLOCK_SIZE_K * N;

    for (uint k = 0; k < BLOCK_SIZE_K; ++k)
    {
      float b_cache = B_shared[k * BLOCK_SIZE_N + local_col];

      for (uint m = 0; m < THREAD_M; ++m)
      {
        sum[m] += A_shared[(local_row * THREAD_M + m) * BLOCK_SIZE_K + k] * b_cache;
      }
    }
    __syncthreads();
  }

  for (uint m = 0; m < THREAD_M; ++m)
  {
    C[(local_row * THREAD_M + m) * N + local_col] =
        alpha * sum[m] + beta * C[(local_row * THREAD_M + m) * N + local_col];
  }
}

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_M>
__global__ void shared_mem_1D_block_tiling_2D_TB(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
  const int tile_row = blockIdx.y;
  const int tile_col = blockIdx.x;

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x; // mapping to the output tile matrix
  /*  
  these are mapping to the output tile matrix which will be (BLOCK_SIZE_M x BLOCK_SIZE_N) HOW?

  local_row => threadIdx.y => [0 - 7] * (THREAD_M) => [0 - 63]
  local_col => threadIdx.x => [0 - 63]

  local_row going 0-7 because each thread handles 8 rows of the output tile matrix
  */


  __shared__ float A_shared[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float B_shared[BLOCK_SIZE_K][BLOCK_SIZE_N];

  // Move to beginning of the appropriate tile
  A += tile_row * BLOCK_SIZE_M * K;
  B += tile_col * BLOCK_SIZE_N;
  C += tile_row * BLOCK_SIZE_M * N + tile_col * BLOCK_SIZE_N;

  const int a_local_col = threadIdx.x % BLOCK_SIZE_K;
  /*
  A_shared arranged as: 
  0 - 7 => row-1
  8 - 15 => row-2
  16 - 23 => row-3

  but as per launched config: 
  threads.x => [0 - 63] => threadIdx.y (0)
  threads.x => [64 - 127] => threadIdx.y (1)
  threads.x => [128 - 191] => threadIdx.y (2)

  so we need to map rows as:
  threadIdx.y * (blockDim.x / BLOCK_SIZE_K) + threadIdx.x / BLOCK_SIZE_K;
  */
  const int a_local_row = threadIdx.y * (blockDim.x / BLOCK_SIZE_K) + threadIdx.x / BLOCK_SIZE_K;

  // although one might seem that following seem very direct mapping
  // const uint a_local_col = threadIdx.y;
  // const uint a_local_row = threadIdx.x;
  // this works, but it uncoalesced access to the A matrix loads (slows down by a factor of ~1.6x)

  const int b_local_col = threadIdx.x;
  const int b_local_row = threadIdx.y;

  // Register file cache for thread results
  float sum[THREAD_M] = {0.0f};

  // Tiling loop over K dimension
  for (int tile_k = 0; tile_k < K; tile_k += BLOCK_SIZE_K)
  {
    A_shared[a_local_row][a_local_col] = A[a_local_row * K + a_local_col];
    B_shared[b_local_row][b_local_col] = B[b_local_row * N + b_local_col];

    __syncthreads();

    // Advance to next tile
    A += BLOCK_SIZE_K;
    B += BLOCK_SIZE_K * N;

    // Matrix multiplication within the tile
    for (int k = 0; k < BLOCK_SIZE_K; ++k)
    {
      // This will be reused to cache it here
      float b_cache = B_shared[k][local_col];

      for (int m = 0; m < THREAD_M; ++m)
      {
        sum[m] += A_shared[local_row * THREAD_M + m][k] * b_cache;
      }
    }

    __syncthreads();
  }

  for (int m = 0; m < THREAD_M; ++m)
  {
    int result_row = local_row * THREAD_M + m;
    C[result_row * N + local_col] = alpha * sum[m] + beta * C[result_row * N + local_col];
  }
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C)
{
  const uint BLOCK_SIZE_M = 64;
  const uint BLOCK_SIZE_N = 64;
  const uint BLOCK_SIZE_K = 8;
  const uint THREAD_M = 8;
  
  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE_N), CEIL_DIV(M, BLOCK_SIZE_M));
  // dim3 blockDim(BLOCK_SIZE_N, BLOCK_SIZE_M / THREAD_M); // (64, 8)
  // shared_mem_1D_block_tiling_2D_TB<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_M>
  //     <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

  dim3 blockDim_linear(BLOCK_SIZE_M * BLOCK_SIZE_N / THREAD_M); // (64*8)
  assert(BLOCK_SIZE_M * BLOCK_SIZE_K == blockDim_linear.x);
  assert(BLOCK_SIZE_N * BLOCK_SIZE_K == blockDim_linear.x);
  shared_mem_1D_block_tiling_1D_TB<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_M>
      <<<gridDim, blockDim_linear>>>(M, N, K, alpha, A, B, beta, C);
}

double inline run_1D_tiling(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

  runSgemm1DBlocktiling(M, N, K, alpha, d_A, d_B, beta, d_C);

  CHECK_CUDA_KERNEL_LAUNCH();
  CHECK_CUDA(cudaDeviceSynchronize());

  // Measure execution time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_runs; i++)
  {
    runSgemm1DBlocktiling(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA_KERNEL_LAUNCH();
  }

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

  CHECK_CUDA(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));

  return milliseconds / num_runs;
}

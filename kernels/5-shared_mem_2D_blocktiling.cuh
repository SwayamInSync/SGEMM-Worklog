#pragma once

#include "../cuda_check.cuh"
#include "../utils.hpp"

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_M, const int THREAD_N>
__global__ void shared_mem_2D_block_tiling_1D_TB(int M, int N, int K, float alpha,
                                                 const float *A, const float *B, float beta,
                                                 float *C)
{
  const uint tile_row = blockIdx.y;
  const uint tile_col = blockIdx.x;

  const uint OUTPUT_BLOCK_SIZE = BLOCK_SIZE_M * BLOCK_SIZE_N;

  // A thread is responsible for THREAD_M x THREAD_N elements in the output tile matrix
  const uint req_threads = OUTPUT_BLOCK_SIZE / (THREAD_M * THREAD_N);

  assert(req_threads == blockDim.x);

  const int thread_col = threadIdx.x % (BLOCK_SIZE_N / THREAD_N); // since now THREAD_N elements done by a thread in column direction
  const int thread_row = threadIdx.x / (BLOCK_SIZE_N / THREAD_N);

  __shared__ float A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K];
  __shared__ float B_shared[BLOCK_SIZE_K * BLOCK_SIZE_N];

  // Move to beginning of the appropriate tile
  A += tile_row * BLOCK_SIZE_M * K;
  B += tile_col * BLOCK_SIZE_N;
  C += tile_row * BLOCK_SIZE_M * N + tile_col * BLOCK_SIZE_N;

  const uint a_local_col = threadIdx.x % BLOCK_SIZE_K;
  const uint a_local_row = threadIdx.x / BLOCK_SIZE_K;

  const uint b_local_col = threadIdx.x % BLOCK_SIZE_N;
  const uint b_local_row = threadIdx.x / BLOCK_SIZE_N;

  // Num of rows gets filled in A_shared at a single step
  // (OUTPUT_BLOCK_SIZE) / (THREAD_M * THREAD_N) => total elements filled in A_shared
  // Rows filled = ((OUTPUT_BLOCK_SIZE) / (THREAD_M * THREAD_N)) / BLOCK_SIZE_K (i.e. elements filled / elements in a row)
  // NOTE that, req_threds = OUTPUT_BLOCK_SIZE / (THREAD_M * THREAD_N) (just use this for simplification)
  const uint stride_A = req_threads / BLOCK_SIZE_K;
  const uint stride_B = req_threads / BLOCK_SIZE_N;

  float sum[THREAD_M][THREAD_N] = {0.0f};

  // register caches for A_shared and B_shared
  float reg_A[THREAD_M] = {0.0f};
  float reg_B[THREAD_N] = {0.0f};

  for (uint tile_k = 0; tile_k < K; tile_k += BLOCK_SIZE_K)
  {
    // fillig shared cache
    for (uint offset = 0; offset < BLOCK_SIZE_M; offset += stride_A)
    {
      A_shared[(a_local_row + offset) * BLOCK_SIZE_K + a_local_col] = A[(a_local_row + offset) * K + a_local_col];
    }

    for (uint offset = 0; offset < BLOCK_SIZE_K; offset += stride_B)
    {
      B_shared[(b_local_row + offset) * BLOCK_SIZE_N + b_local_col] = B[(b_local_row + offset) * N + b_local_col];
    }

    __syncthreads();

    // Advance blocktile
    A += BLOCK_SIZE_K;
    B += BLOCK_SIZE_K * N;

    // per-thread result
    for (uint k = 0; k < BLOCK_SIZE_K; k++)
    {
      // load to register cache
      for (uint m = 0; m < THREAD_M; m++)
        reg_A[m] = A_shared[(thread_row * THREAD_M + m) * BLOCK_SIZE_K + k];

      for (uint n = 0; n < THREAD_N; n++)
        reg_B[n] = B_shared[k * BLOCK_SIZE_N + (thread_col * THREAD_N + n)];

      for (uint res_m = 0; res_m < THREAD_M; res_m++)
      {
        for (uint res_n = 0; res_n < THREAD_N; res_n++)
        {
          sum[res_m][res_n] += reg_A[res_m] * reg_B[res_n];
        }
      }
    }
    __syncthreads();
  }

  for (uint m = 0; m < THREAD_M; ++m)
  {
    for (uint n = 0; n < THREAD_N; ++n)
    {
      C[(thread_row * THREAD_M + m) * N + (thread_col * THREAD_N + n)] =
          alpha * sum[m][n] + beta * C[(thread_row * THREAD_M + m) * N + (thread_col * THREAD_N + n)];
    }
  }
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
  const uint BLOCK_SIZE_M = 128;
  const uint BLOCK_SIZE_N = 128;
  const uint BLOCK_SIZE_K = 8;
  const uint THREAD_M = 8;
  const uint THREAD_N = 8;

  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE_N), CEIL_DIV(M, BLOCK_SIZE_M));
  dim3 blockDim_linear((BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_M * THREAD_N)); // (16*16)
  shared_mem_2D_block_tiling_1D_TB<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_M, THREAD_N>
      <<<gridDim, blockDim_linear>>>(M, N, K, alpha, A, B, beta, C);
}

double inline run_2D_tiling(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

  runSgemm2DBlocktiling(M, N, K, alpha, d_A, d_B, beta, d_C);

  CHECK_CUDA_KERNEL_LAUNCH();
  CHECK_CUDA(cudaDeviceSynchronize());

  // Measure execution time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_runs; i++)
  {
    runSgemm2DBlocktiling(M, N, K, alpha, d_A, d_B, beta, d_C);
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
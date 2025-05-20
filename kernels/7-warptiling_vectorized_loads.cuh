#pragma once

#include "../cuda_check.cuh"
#include "../utils.hpp"

template <int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int THREAD_M, int THREAD_N, int MAX_THREADS>
__global__ __launch_bounds__(MAX_THREADS) void vectorized_loads_kernel(int M, int N, int K, float alpha,
                                                                       float *A, float *B, float beta,
                                                                       float *C)
{

  __shared__ float A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K];
  __shared__ float B_shared[BLOCK_SIZE_K * BLOCK_SIZE_N];

  // concept remains same, one block tile would be treated by one thread block
  const uint block_row = blockIdx.y;
  const uint block_col = blockIdx.x;

  // moving block tile to beginning of A's rows and B's columns
  A += block_row * BLOCK_SIZE_M * K;
  B += block_col * BLOCK_SIZE_N;
  C += block_row * BLOCK_SIZE_M * N + block_col * BLOCK_SIZE_N;

  // splitting block_tiles to further into warp tiles
  const int WARP_SIZE_M = THREAD_M * 16;
  const int WARP_SIZE_N = THREAD_N * 16;

  // total required warp iterations
  const int WM_Iter = CEIL_DIV(BLOCK_SIZE_M, WARP_SIZE_M);
  const int WN_Iter = CEIL_DIV(BLOCK_SIZE_N, WARP_SIZE_N);

  // placement of thread  in a warptile
  const int thread_row = threadIdx.x / (WARP_SIZE_N / THREAD_N);
  const int thread_col = threadIdx.x % (WARP_SIZE_N / THREAD_N);

  // placement of thread in a blocktile (doing a float4 vectorized load)
  const int a_local_row = threadIdx.x / (BLOCK_SIZE_K / 4);
  const int a_local_col = threadIdx.x % (BLOCK_SIZE_K / 4);
  const int b_local_row = threadIdx.x / (BLOCK_SIZE_N / 4);
  const int b_local_col = threadIdx.x % (BLOCK_SIZE_N / 4);

  const int row_stride_A = MAX_THREADS / (BLOCK_SIZE_K / 4); // A block has MAX_THREADS threads and process (BLOCK_SIZE_K / 4) A_shared floats in 1 step
  const int row_stride_B = MAX_THREADS / (BLOCK_SIZE_N / 4);

  // This holds the partial sum of a thread's accumulation
  // each thread does THREAD_M * THREAD_N accumulations per warptile
  // and there are total WM_Iter * WN_Iter warptiles
  // so total accumulations = WM_Iter * WN_Iter * THREAD_M * THREAD_N
  float sum[WM_Iter * WN_Iter * THREAD_M * THREAD_N] = {0.0f};

  float regM[THREAD_M] = {0.0f};
  float regN[THREAD_N] = {0.0f};

  for (int block_idx = 0; block_idx < K; block_idx += BLOCK_SIZE_K)
  {
    for (int offset = 0; offset + row_stride_A <= BLOCK_SIZE_M; offset += row_stride_A)
    {
      float4 temp = reinterpret_cast<float4 *>(&A[(a_local_row + offset) * K + a_local_col * 4])[0];

      // transpose A while storing it
      // As[k_index * BM + m_index] = A_global[m_index_global][k_index_global]
      // k_index comes from (a_local_col * 4 + i)
      // m_index comes from (a_local_row + offset)
      A_shared[(a_local_col * 4 + 0) * BLOCK_SIZE_M + a_local_row + offset] = temp.x;
      A_shared[(a_local_col * 4 + 1) * BLOCK_SIZE_M + a_local_row + offset] = temp.y;
      A_shared[(a_local_col * 4 + 2) * BLOCK_SIZE_M + a_local_row + offset] = temp.z;
      A_shared[(a_local_col * 4 + 3) * BLOCK_SIZE_M + a_local_row + offset] = temp.w;
    }

    for (int offset = 0; offset + row_stride_B <= BLOCK_SIZE_K; offset += row_stride_B)
    {
      reinterpret_cast<float4 *>(&B_shared[(b_local_row + offset) * BLOCK_SIZE_N + b_local_col * 4])[0] =
          reinterpret_cast<float4 *>(&B[(b_local_row + offset) * N + b_local_col * 4])[0];
    }

    __syncthreads();

    for (int wm_idx = 0; wm_idx < WM_Iter; wm_idx++)
    {
      for (int wn_idx = 0; wn_idx < WN_Iter; wn_idx++)
      {
        for (int k = 0; k < BLOCK_SIZE_K; k++)
        {
          // loading to registers
          for (int m = 0; m < THREAD_M; m++)
            regM[m] = A_shared[k * BLOCK_SIZE_M + (wm_idx * WARP_SIZE_M) + thread_row * THREAD_M + m];

          for (int n = 0; n < THREAD_N; n++)
            regN[n] = B_shared[k * BLOCK_SIZE_N + (wn_idx * WARP_SIZE_N) + thread_col * THREAD_N + n];

          // doing the accumulation
          for (int m = 0; m < THREAD_M; m++)
          {
            for (int n = 0; n < THREAD_N; n++)
            {
              sum[(wm_idx * THREAD_M + m) * (WN_Iter * THREAD_N) + wn_idx * THREAD_N + n] += regM[m] * regN[n];
            }
          }
        }
      }
    }
    __syncthreads();
    A += BLOCK_SIZE_K;
    B += BLOCK_SIZE_K * N;
  }

  for (int wm_idx = 0; wm_idx < WM_Iter; wm_idx++)
  {
    for (int wn_idx = 0; wn_idx < WN_Iter; wn_idx++)
    {
      float *C_tile_base = C + (wm_idx * WARP_SIZE_M) * N + (wn_idx * WARP_SIZE_N);
      for (int m_idx_in_thread = 0; m_idx_in_thread < THREAD_M; m_idx_in_thread++) // m_idx_in_thread = 0..7
      {
        // n_base_in_thread will be 0, then 4 (for THREAD_N = 8)
        for (int n_base_in_thread = 0; n_base_in_thread < THREAD_N; n_base_in_thread += 4)
        {
          // Calculate the pointer to the start of the float4 in C
          float *c_float4_ptr = &C_tile_base[(thread_row * THREAD_M + m_idx_in_thread) * N +
                                             (thread_col * THREAD_N + n_base_in_thread)];
          float4 temp_c_values = reinterpret_cast<float4 *>(c_float4_ptr)[0];

          // Calculate the base index for the 1D sum array.
          // This corresponds to the element C[m_idx_in_thread][n_base_in_thread] of this thread's work.
          int sum_array_base_idx = (wm_idx * THREAD_M + m_idx_in_thread) * (WN_Iter * THREAD_N) +
                                   (wn_idx * THREAD_N + n_base_in_thread);

          // Apply alpha * sum + beta * C for each component
          temp_c_values.x = alpha * sum[sum_array_base_idx + 0] + beta * temp_c_values.x;
          temp_c_values.y = alpha * sum[sum_array_base_idx + 1] + beta * temp_c_values.y;
          temp_c_values.z = alpha * sum[sum_array_base_idx + 2] + beta * temp_c_values.z;
          temp_c_values.w = alpha * sum[sum_array_base_idx + 3] + beta * temp_c_values.w;
          // Note: Since THREAD_N=8 (a multiple of 4), sum[sum_array_base_idx + 3] is valid.
          // If THREAD_N could be e.g. 6, needed boundary checks for y,z,w components for sum.

          reinterpret_cast<float4 *>(c_float4_ptr)[0] = temp_c_values;
        }
      }
    }
  }
}

void run_vectorized_loads(int M, int N, int K, float *A, float *B, float *C,
                          float alpha, float beta)
{
  const int MAX_THREADS = 256;
  const uint BLOCK_SIZE_M = 64;
  const uint BLOCK_SIZE_N = 64;
  const uint BLOCK_SIZE_K = 16;
  const uint THREAD_M = 4;
  const uint THREAD_N = 4;

  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE_N), CEIL_DIV(M, BLOCK_SIZE_M));
  dim3 blockDim_linear((BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_M * THREAD_N)); // (16*16)
  vectorized_loads_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_M, THREAD_N, MAX_THREADS>
      <<<gridDim, blockDim_linear>>>(M, N, K, alpha, A, B, beta, C);
}

double inline run_vectorized_smem(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

  run_vectorized_loads(M, N, K, d_A, d_B, d_C, alpha, beta);

  CHECK_CUDA_KERNEL_LAUNCH();
  CHECK_CUDA(cudaDeviceSynchronize());

  // Measure execution time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_runs; i++)
  {
    run_vectorized_loads(M, N, K, d_A, d_B, d_C, alpha, beta);
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
#pragma once

#include "../cuda_check.cuh"
#include "../utils.hpp"

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_M, const int THREAD_N>
__global__ void shared_mem_2D_block_tiling_1D_TB_coalesced_C(int M, int N, int K, float alpha,
                                                             const float *A, const float *B, float beta,
                                                             float *C)
{
  const uint tile_row_idx = blockIdx.y;
  const uint tile_col_idx = blockIdx.x;

  const uint OUTPUT_BLOCK_SIZE = BLOCK_SIZE_M * BLOCK_SIZE_N;

  const uint req_threads = OUTPUT_BLOCK_SIZE / (THREAD_M * THREAD_N);

  const int thread_sub_block_col = threadIdx.x % (BLOCK_SIZE_N / THREAD_N);
  const int thread_sub_block_row = threadIdx.x / (BLOCK_SIZE_N / THREAD_N);

  __shared__ float A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K];
  __shared__ float B_shared[BLOCK_SIZE_K * BLOCK_SIZE_N];

  // Shared memory for the output C tile
  __shared__ float C_shared[BLOCK_SIZE_M * BLOCK_SIZE_N];

  const float *A_tile_ptr = A + tile_row_idx * BLOCK_SIZE_M * K;
  const float *B_tile_ptr = B + tile_col_idx * BLOCK_SIZE_N;
  float *C_tile_ptr = C + tile_row_idx * BLOCK_SIZE_M * N + tile_col_idx * BLOCK_SIZE_N;

  const uint a_load_local_col = threadIdx.x % BLOCK_SIZE_K;
  const uint a_load_local_row = threadIdx.x / BLOCK_SIZE_K;
  const uint b_load_local_col = threadIdx.x % BLOCK_SIZE_N;
  const uint b_load_local_row = threadIdx.x / BLOCK_SIZE_N;

  const uint stride_A_load = blockDim.x / BLOCK_SIZE_K;
  const uint stride_B_load = blockDim.x / BLOCK_SIZE_N;

  float sum[THREAD_M][THREAD_N] = {0.0f};

  float reg_A[THREAD_M];
  float reg_B[THREAD_N];

  for (uint tile_k_loop = 0; tile_k_loop < K; tile_k_loop += BLOCK_SIZE_K)
  {

    for (uint offset = 0; offset < BLOCK_SIZE_M; offset += stride_A_load)
    {
      if ((a_load_local_row + offset) < BLOCK_SIZE_M && (tile_row_idx * BLOCK_SIZE_M + a_load_local_row + offset) < M && (tile_k_loop + a_load_local_col) < K)
      {
        A_shared[(a_load_local_col) * BLOCK_SIZE_M + (a_load_local_row + offset)] =
            A_tile_ptr[(a_load_local_row + offset) * K + a_load_local_col];
      }
      else if ((a_load_local_row + offset) < BLOCK_SIZE_M)
      {
        A_shared[(a_load_local_col) * BLOCK_SIZE_M + (a_load_local_row + offset)] = 0.0f;
      }
    }

    for (uint offset = 0; offset < BLOCK_SIZE_K; offset += stride_B_load)
    {

      if ((b_load_local_row + offset) < BLOCK_SIZE_K && (tile_k_loop + b_load_local_row + offset) < K && (tile_col_idx * BLOCK_SIZE_N + b_load_local_col) < N)
      {
        B_shared[(b_load_local_row + offset) * BLOCK_SIZE_N + b_load_local_col] =
            B_tile_ptr[(b_load_local_row + offset) * N + b_load_local_col];
      }
      else if ((b_load_local_row + offset) < BLOCK_SIZE_K)
      {
        B_shared[(b_load_local_row + offset) * BLOCK_SIZE_N + b_load_local_col] = 0.0f;
      }
    }
    __syncthreads();

    A_tile_ptr += BLOCK_SIZE_K;
    B_tile_ptr += BLOCK_SIZE_K * N;

    for (uint k_inner = 0; k_inner < BLOCK_SIZE_K; k_inner++)
    {
      for (uint m = 0; m < THREAD_M; m++)
      {

        reg_A[m] = A_shared[k_inner * BLOCK_SIZE_M + (thread_sub_block_row * THREAD_M + m)];
      }

      for (uint n = 0; n < THREAD_N; n++)
      {

        reg_B[n] = B_shared[k_inner * BLOCK_SIZE_N + (thread_sub_block_col * THREAD_N + n)];
      }

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

  const uint elements_per_thread_for_C_IO = (BLOCK_SIZE_M * BLOCK_SIZE_N) / blockDim.x; // This should simplify to THREAD_M * THREAD_N

  // Step 1: If beta != 0, load C from global memory to C_shared (coalesced load)
  if (beta != 0.0f)
  {
    for (uint i = 0; i < elements_per_thread_for_C_IO; ++i)
    {
      uint C_element_idx_in_tile = threadIdx.x + i * blockDim.x;
      uint C_row_in_tile = C_element_idx_in_tile / BLOCK_SIZE_N;
      uint C_col_in_tile = C_element_idx_in_tile % BLOCK_SIZE_N;

      if (C_row_in_tile < BLOCK_SIZE_M)
      { // Check row boundary for C_shared
        // Check global boundaries before loading
        if ((tile_row_idx * BLOCK_SIZE_M + C_row_in_tile) < M && (tile_col_idx * BLOCK_SIZE_N + C_col_in_tile) < N)
        {
          C_shared[C_row_in_tile * BLOCK_SIZE_N + C_col_in_tile] =
              C_tile_ptr[C_row_in_tile * N + C_col_in_tile];
        }
        else
        {                                                                // Element is outside global C matrix bounds but within tile logic
          C_shared[C_row_in_tile * BLOCK_SIZE_N + C_col_in_tile] = 0.0f; // Or handle as per problem (e.g. skip if padding)
        }
      }
    }
    __syncthreads();
  }

  // Step 2: Each thread writes its computed 'sum' results into C_shared
  // This is a scattered write to C_shared based on thread's logical sub-block, but shared memory handles this well.
  for (uint m = 0; m < THREAD_M; ++m)
  {
    for (uint n = 0; n < THREAD_N; ++n)
    {
      uint C_s_row = thread_sub_block_row * THREAD_M + m;
      uint C_s_col = thread_sub_block_col * THREAD_N + n;

      // Ensure writes are within the bounds of C_shared and the logical C tile
      if (C_s_row < BLOCK_SIZE_M && C_s_col < BLOCK_SIZE_N)
      {
        // Also check if this part of the C tile corresponds to a valid global C element
        if ((tile_row_idx * BLOCK_SIZE_M + C_s_row) < M && (tile_col_idx * BLOCK_SIZE_N + C_s_col) < N)
        {
          float val = alpha * sum[m][n];
          if (beta != 0.0f)
          {
            val += beta * C_shared[C_s_row * BLOCK_SIZE_N + C_s_col];
          }
          C_shared[C_s_row * BLOCK_SIZE_N + C_s_col] = val;
        }
      }
    }
  }
  __syncthreads();

  // Step 3: Store C_shared to global C (coalesced store)
  for (uint i = 0; i < elements_per_thread_for_C_IO; ++i)
  {
    uint C_element_idx_in_tile = threadIdx.x + i * blockDim.x;
    uint C_row_in_tile = C_element_idx_in_tile / BLOCK_SIZE_N;
    uint C_col_in_tile = C_element_idx_in_tile % BLOCK_SIZE_N;

    if (C_row_in_tile < BLOCK_SIZE_M)
    {
      if ((tile_row_idx * BLOCK_SIZE_M + C_row_in_tile) < M && (tile_col_idx * BLOCK_SIZE_N + C_col_in_tile) < N)
      {
        C_tile_ptr[C_row_in_tile * N + C_col_in_tile] =
            C_shared[C_row_in_tile * BLOCK_SIZE_N + C_col_in_tile];
      }
    }
  }
}


void run_2D_blocktiling_coalesced(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
  const uint BLOCK_SIZE_M = 64;
  const uint BLOCK_SIZE_N = 64;
  const uint BLOCK_SIZE_K = 8;
  const uint THREAD_M = 8;
  const uint THREAD_N = 8;

  dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE_N), CEIL_DIV(M, BLOCK_SIZE_M));
  dim3 blockDim_linear((BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_M * THREAD_N)); // (16*16)
  shared_mem_2D_block_tiling_1D_TB_coalesced_C<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_M, THREAD_N>
      <<<gridDim, blockDim_linear>>>(M, N, K, alpha, A, B, beta, C);
}

double inline run_2D_tiling_coalesced(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

  run_2D_blocktiling_coalesced(M, N, K, alpha, d_A, d_B, beta, d_C);

  CHECK_CUDA_KERNEL_LAUNCH();
  CHECK_CUDA(cudaDeviceSynchronize());

  // Measure execution time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_runs; i++)
  {
    run_2D_blocktiling_coalesced(M, N, K, alpha, d_A, d_B, beta, d_C);
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
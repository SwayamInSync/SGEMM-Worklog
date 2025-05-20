#pragma once
#include "../cuda_check.cuh"
#include "../utils.hpp"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda/barrier>


namespace cg = cooperative_groups;

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, int row_stride_A, int row_stride_B, typename T>
__device__ void load_share_memory_async(float *A_shared, float *B_shared, float *A, float *B, int N, int K, int a_local_row, int a_local_col, int b_local_row, int b_local_col, T &barrier) // generic pass can be used for barrier for any thread scope
{
  for (int offset = 0; offset + row_stride_A <= BLOCK_SIZE_M; offset += row_stride_A)
  {
    // avoiding manual blocking load calls via float4 as this will be async traspose store order
    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
      cuda::memcpy_async(&A_shared[(a_local_col * 4 + i) * BLOCK_SIZE_M + a_local_row + offset],
                         &A[(a_local_row + offset) * K + a_local_col * 4 + i], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), barrier);
    }
  }

  for (int offset = 0; offset + row_stride_B <= BLOCK_SIZE_K; offset += row_stride_B)
  {
    cuda::memcpy_async(&B_shared[(b_local_row + offset) * BLOCK_SIZE_N + b_local_col * 4],
                       &B[(b_local_row + offset) * N + b_local_col * 4], cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), barrier);
  }
}

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int WARP_SIZE_M, const int WARP_SIZE_N, const int WARP_STEPS_M, const int WARP_STEPS_N, const int WARP_SUBTILE_M, const int WARP_SUBTILE_N, const int THREAD_M, const int THREAD_N>
__device__ void partial_dot(float *A_shared, float *B_shared, float *sum, int K, float *regM, float *regN, int warp_row_in_blocktile, int warp_col_in_blocktile, int thread_row_in_warp_subtile, int thread_col_in_warp_subtile)
{
  for (int dot_idx = 0; dot_idx < BLOCK_SIZE_K; dot_idx++)
  {
    // load register with complete warptile
    for (int w_sub_row = 0; w_sub_row < WARP_STEPS_M; w_sub_row++)
    {
      for (int m = 0; m < THREAD_M; m++)
      {
        regM[w_sub_row * THREAD_M + m] = A_shared[dot_idx * BLOCK_SIZE_M + warp_row_in_blocktile * WARP_SIZE_M + w_sub_row * WARP_SUBTILE_M + thread_row_in_warp_subtile * THREAD_M + m];
      }
    }

    for (int w_sub_col = 0; w_sub_col < WARP_STEPS_N; w_sub_col++)
    {
      for (int n = 0; n < THREAD_N; n++)
      {
        regN[w_sub_col * THREAD_N + n] = B_shared[dot_idx * BLOCK_SIZE_N + warp_col_in_blocktile * WARP_SIZE_N + w_sub_col * WARP_SUBTILE_N + thread_col_in_warp_subtile * THREAD_N + n];
      }
    }

    // doing the accumulation
    for (int w_sub_row = 0; w_sub_row < WARP_STEPS_M; w_sub_row++)
    {
      for (int w_sub_col = 0; w_sub_col < WARP_STEPS_N; w_sub_col++)
      {
        for (int m = 0; m < THREAD_M; m++)
        {
          for (int n = 0; n < THREAD_N; n++)
          {
            sum[(w_sub_row * THREAD_M + m) * (WARP_STEPS_N * THREAD_N) + w_sub_col * THREAD_N + n] += regM[w_sub_row * THREAD_M + m] * regN[w_sub_col * THREAD_N + n];
          }
        }
      }
    }
  }
}

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int WARP_SIZE_M, const int WARP_SIZE_N, const int WARP_STEPS_N, const int THREAD_M, const int THREAD_N, const int MAX_THREADS>
__global__ void __launch_bounds__(MAX_THREADS) pingpong_buffering_kernel(int M, int N, int K, float *A, float *B, float *C, const float alpha, const float beta)
{
  auto block = cg::this_thread_block();
  __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barriers[2];

  if (block.thread_rank() == 0)
  {
    init(&barriers[0], block.size());
    init(&barriers[1], block.size());
  }
  __syncthreads(); // can also use block.sync() here, but has more abstraction overhead

  __shared__ float A_shared[2][BLOCK_SIZE_M * BLOCK_SIZE_K];
  __shared__ float B_shared[2][BLOCK_SIZE_K * BLOCK_SIZE_N];
  
  float *smem_A[2] = {A_shared[0], A_shared[1]};
  float *smem_B[2] = {B_shared[0], B_shared[1]};

  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  // const int SWIZZLE = 4;

  // // current swizzle block
  // const int block_idx = blockIdx.x / (SWIZZLE * SWIZZLE);
  // // inside the current swizzle block
  // const int swizzle_idx = blockIdx.x % (SWIZZLE * SWIZZLE);

  // const int block_row = (block_idx / (N / BLOCK_SIZE_N / SWIZZLE)) * SWIZZLE + (swizzle_idx / SWIZZLE);
  // const int block_col = (block_idx % (N / BLOCK_SIZE_N / SWIZZLE)) * SWIZZLE + (swizzle_idx % SWIZZLE);

  const int warpIdx = threadIdx.x / WARP_SIZE;
  const int warp_col_in_blocktile = warpIdx % (BLOCK_SIZE_N / WARP_SIZE_N);
  const int warp_row_in_blocktile = warpIdx / (BLOCK_SIZE_N / WARP_SIZE_N);

  A += block_row * BLOCK_SIZE_M * K;
  B += block_col * BLOCK_SIZE_N;
  C += (block_row * BLOCK_SIZE_M + warp_row_in_blocktile * WARP_SIZE_M) * N + (block_col * BLOCK_SIZE_N + warp_col_in_blocktile * WARP_SIZE_N);

  const int WARP_STEPS_M = (WARP_SIZE_M * WARP_SIZE_N) / (WARP_SIZE * THREAD_M * THREAD_N * WARP_STEPS_N);

  // subtile dimensions
  const int WARP_SUBTILE_M = WARP_SIZE_M / WARP_STEPS_M;
  const int WARP_SUBTILE_N = WARP_SIZE_N / WARP_STEPS_N;

  // thread arrangement in warp subtile
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int thread_row_in_warp_subtile = lane_id / (WARP_SUBTILE_N / THREAD_N);
  const int thread_col_in_warp_subtile = lane_id % (WARP_SUBTILE_N / THREAD_N);

  const int a_local_row = threadIdx.x / (BLOCK_SIZE_K / 4);
  const int a_local_col = threadIdx.x % (BLOCK_SIZE_K / 4);
  const int b_local_row = threadIdx.x / (BLOCK_SIZE_N / 4);
  const int b_local_col = threadIdx.x % (BLOCK_SIZE_N / 4);

  const int row_stride_A = MAX_THREADS / (BLOCK_SIZE_K / 4);
  const int row_stride_B = MAX_THREADS / (BLOCK_SIZE_N / 4);

  float sum[WARP_STEPS_M * THREAD_M * WARP_STEPS_N * THREAD_N] = {0.0f};
  float regM[WARP_STEPS_M * THREAD_M] = {0.0f};
  float regN[WARP_STEPS_N * THREAD_N] = {0.0f};

  // load the first block
  int curr = 0, next = 1;

  load_share_memory_async<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, row_stride_A, row_stride_B>(smem_A[curr], smem_B[curr], A, B, N, K, a_local_row, a_local_col, b_local_row, b_local_col, barriers[curr]);

  for (int block_idx = 0; block_idx < K - BLOCK_SIZE_K; block_idx += BLOCK_SIZE_K)
  {
    A += BLOCK_SIZE_K;
    B += BLOCK_SIZE_K * N;

    load_share_memory_async<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, row_stride_A, row_stride_B>(smem_A[next], smem_B[next], A, B, N, K, a_local_row, a_local_col, b_local_row, b_local_col, barriers[next]);

    barriers[curr].arrive_and_wait();

    partial_dot<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_SIZE_M, WARP_SIZE_N, WARP_STEPS_M, WARP_STEPS_N, WARP_SUBTILE_M, WARP_SUBTILE_N, THREAD_M, THREAD_N>(smem_A[curr], smem_B[curr], sum, K, regM, regN, warp_row_in_blocktile, warp_col_in_blocktile, thread_row_in_warp_subtile, thread_col_in_warp_subtile);

    __syncthreads();
    curr ^= 1;
    next ^= 1;
  }
  barriers[curr].arrive_and_wait();
  partial_dot<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_SIZE_M, WARP_SIZE_N, WARP_STEPS_M, WARP_STEPS_N, WARP_SUBTILE_M, WARP_SUBTILE_N, THREAD_M, THREAD_N>(smem_A[curr], smem_B[curr], sum, K, regM, regN, warp_row_in_blocktile, warp_col_in_blocktile, thread_row_in_warp_subtile, thread_col_in_warp_subtile);

  // write back to global memory
  for (int w_sub_row = 0; w_sub_row < WARP_STEPS_M; w_sub_row++)
  {
    for (int w_sub_col = 0; w_sub_col < WARP_STEPS_N; w_sub_col++)
    {
      // Calculate pointer to the top-left of the current WARP_SUBTILE_M x WARP_SUBTILE_N
      // chunk that this warp is currently writing.
      float *C_subtile_base_ptr = C +
                                  (w_sub_row * WARP_SUBTILE_M) * N + // Offset to the correct row of the sub-tile
                                  (w_sub_col * WARP_SUBTILE_N);      // Offset to the correct col of the sub-tile

      // Each thread now writes its THREAD_M x THREAD_N portion of this C_subtile_base_ptr
      for (int m = 0; m < THREAD_M; m++) // Loop over M-elements this thread handles
      {
        // Loop over N-elements this thread handles, processing 4 elements (a float4) at a time
        for (int n_base = 0; n_base < THREAD_N; n_base += 4)
        {
          // Calculate the precise global memory address for the float4 this thread will read/write in C.
          // (thread_row_in_warp_subtile * THREAD_M + m) gives the thread's row within the current WARP_SUBTILE.
          // (thread_col_in_warp_subtile * THREAD_N + n_base) gives the thread's starting column (float index) within the current WARP_SUBTILE.
          float *C_element_ptr = &C_subtile_base_ptr[(thread_row_in_warp_subtile * THREAD_M + m) * N + // Row offset from C_subtile_base_ptr
                                                     (thread_col_in_warp_subtile * THREAD_N + n_base)  // Column offset for the start of the float4
          ];

          // Load the existing float4 from global C memory (for beta blending)
          float4 c_current_values = reinterpret_cast<float4 *>(C_element_ptr)[0];

          // Calculate the base index into the thread's local 'sum' array.
          // 'sum' array stores results as (WARP_STEPS_M * THREAD_M) rows by (WARP_STEPS_N * THREAD_N) columns.
          // (w_sub_row * THREAD_M + m) is the overall row index in the thread's result grid.
          // (w_sub_col * THREAD_N + n_base) is the overall starting column index for the float4 in the thread's result grid.
          int sum_array_base_idx =
              (w_sub_row * THREAD_M + m) * (WARP_STEPS_N * THREAD_N) +
              (w_sub_col * THREAD_N) + n_base;

          // Perform C = alpha * A*B_results[sum_array] + beta * C_current
          c_current_values.x = alpha * sum[sum_array_base_idx + 0] + beta * c_current_values.x;
          c_current_values.y = alpha * sum[sum_array_base_idx + 1] + beta * c_current_values.y;
          c_current_values.z = alpha * sum[sum_array_base_idx + 2] + beta * c_current_values.z;
          c_current_values.w = alpha * sum[sum_array_base_idx + 3] + beta * c_current_values.w;

          reinterpret_cast<float4 *>(C_element_ptr)[0] = c_current_values;
        }
      }
    }
  }
}


void run_pingpong_kernel(int M, int N, int K, float *A, float *B, float *C, const float alpha, const float beta)
{
  // BLOCK_SIZE_M       (Height of the C tile processed by a thread block)
  // BLOCK_SIZE_N       (Width of the C tile processed by a thread block)
  // BLOCK_SIZE_K       (K-dimension of tiles loaded into shared memory)
  // WARP_SIZE_M        (Height of the C tile processed by a single warp)
  // WARP_SIZE_N        (Width of the C tile processed by a single warp)
  // WARP_STEPS_N       (Number of iterations for a warp along its N-dimension)
  // THREAD_M           (M-dimension of the C sub-tile processed by a single thread per step)
  // THREAD_N           (N-dimension of the C sub-tile processed by a single thread per step)
  // MAX_THREADS        (Total threads in a thread block)

  const int BLOCK_SIZE_M = 64;
  const int BLOCK_SIZE_N = 128;
  const int BLOCK_SIZE_K = 16;
  const int WARP_SIZE_M = 32;
  const int WARP_SIZE_N = 64;
  const int WARP_STEPS_N = 1;
  const int THREAD_M = 4;
  const int THREAD_N = 4;
  const int MAX_THREADS = 128;

  dim3 blockdim(MAX_THREADS);
  const int num_blocks_M = CEIL_DIV(M, BLOCK_SIZE_M);
  const int num_blocks_N = CEIL_DIV(N, BLOCK_SIZE_N);
  // dim3 gridDim(num_blocks_M * num_blocks_N);
  dim3 gridDim(num_blocks_N, num_blocks_M);

  pingpong_buffering_kernel<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, WARP_SIZE_M, WARP_SIZE_N, WARP_STEPS_N, THREAD_M, THREAD_N, MAX_THREADS><<<gridDim, blockdim>>>(M, N, K, A, B, C, alpha, beta);
}

double inline run_pingpong(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

  run_pingpong_kernel(M, N, K, d_A, d_B, d_C, alpha, beta);

  CHECK_CUDA_KERNEL_LAUNCH();
  CHECK_CUDA(cudaDeviceSynchronize());

  // Measure execution time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_runs; i++)
  {
    run_pingpong_kernel(M, N, K, d_A, d_B, d_C, alpha, beta);
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
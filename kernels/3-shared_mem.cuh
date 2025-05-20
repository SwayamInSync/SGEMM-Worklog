#pragma once

#include "../cuda_check.cuh"

template <uint BLOCK_SIZE> // letting this to match the block_size is optimal for performance (common sense)
__global__ void sgemm_shared_mem_kernel(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta)
{
  __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE]; // 1024*4 = 4096 bytes = 4KB
  __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE]; // 1024*4 = 4096 bytes = 4KB

  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;

  int local_row = threadIdx.y;
  int local_col = threadIdx.x;

  // calculate the row and column of the C element to work on
  int row = tile_row * BLOCK_SIZE + local_row;
  int col = tile_col * BLOCK_SIZE + local_col;

  // fill the shared memory
  // tiles will move along the shared axis (K)

  float sum = 0.0f;
  for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++)
  {
    // load A and B to shared mems
    int a_tile_start_row = tile_row * BLOCK_SIZE; // Row in A stays fixed for this thread block
    int a_tile_start_col = tile * BLOCK_SIZE; // columns in A move with tiles

    int b_tile_start_row = tile * BLOCK_SIZE; // rows in B move with tiles
    int b_tile_start_col = tile_col * BLOCK_SIZE; // Column in B stays fixed for this thread block

    if (a_tile_start_row + local_row < M && a_tile_start_col + local_col < K)
    {
      A_shared[local_row][local_col] = A[(a_tile_start_row + local_row)*K + (a_tile_start_col + local_col)];
    }
    else
    {
      A_shared[local_row][local_col] = 0.0f;
    }

    if (b_tile_start_row + local_row < K && b_tile_start_col + local_col < N)
    {
      B_shared[local_row][local_col] = B[(b_tile_start_row + local_row)*N + (b_tile_start_col + local_col)];
    }
    else
    {
      B_shared[local_row][local_col] = 0.0f;
    }

    // sync threads to make sure all threads have loaded the data
    __syncthreads();

    // compute the partial sum
    for (int k = 0; k < BLOCK_SIZE; k++)
    {
      sum += A_shared[local_row][k] * B_shared[k][local_col];
    }

    // sync threads to make sure all threads have finished computing
    __syncthreads();
  }

  // write the result to global memory
  if (row < M && col < N)
  {
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}


template <uint BLOCK_SIZE>
__global__ void sgemm_shared_mem_kernel_1D_TB(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta)
{
  __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE]; // 1024*4 = 4096 bytes = 4KB
  __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE]; // 1024*4 = 4096 bytes = 4KB

  int tile_row = blockIdx.y;
  int tile_col = blockIdx.x;

  int local_row = threadIdx.x / BLOCK_SIZE;
  int local_col = threadIdx.x % BLOCK_SIZE;

  // position the A and B to proper tile locations
  A += tile_row * BLOCK_SIZE * K; 
  B += tile_col * BLOCK_SIZE;
  C += tile_row * BLOCK_SIZE * N + tile_col * BLOCK_SIZE;
  
  float sum = 0.0f;
  for(int tidx = 0; tidx < K; tidx += BLOCK_SIZE)
  {
    A_shared[local_row * BLOCK_SIZE + local_col] = A[local_row*K + local_col];
    B_shared[local_row * BLOCK_SIZE + local_col] = B[local_row*N + local_col];
    __syncthreads();

    // Advance blocktile
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    // compute the partial sum
    for(int k = 0; k < BLOCK_SIZE; k++)
    {
      sum += A_shared[local_row * BLOCK_SIZE + k] * B_shared[k * BLOCK_SIZE + local_col];
    }
    __syncthreads();
  }

  C[local_row * N + local_col] = alpha * sum + beta * C[local_row * N + local_col];
}

void run_shared_kernel(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  const int block_size = 32; // 32x32 threads per block
  dim3 gridDim((N + block_size - 1) / block_size, (M + block_size - 1) / block_size);
  // dim3 blockDim(block_size, block_size);
  // sgemm_shared_mem_kernel<block_size><<<gridDim, blockDim>>>(M, N, K, A, B, C, alpha, beta);
  
  dim3 blockDim(block_size*block_size);
  sgemm_shared_mem_kernel_1D_TB<block_size><<<gridDim, blockDim>>>(M, N, K, A, B, C, alpha, beta);
}

double inline run_shared_mem(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M*N*sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice));

  run_shared_kernel(M, N, K, d_A, d_B, d_C, alpha, beta);
  CHECK_CUDA_KERNEL_LAUNCH();

  CHECK_CUDA(cudaDeviceSynchronize());

  // Measure execution time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for(int i = 0; i < num_runs; i++)
  {
    run_shared_kernel(M, N, K, d_A, d_B, d_C, alpha, beta);
    CHECK_CUDA_KERNEL_LAUNCH();
  }

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

  CHECK_CUDA(cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));

  return milliseconds / num_runs;
}
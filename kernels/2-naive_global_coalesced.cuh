#pragma once
#include "../cuda_check.cuh"

__global__ void sgemm_naive_coalesced_kernel(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta)
{
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N)
  {
    float sum = 0.0f;
    for(int k = 0; k < K; k++)
    {
      sum += A[row * K + k] * B[k*N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}


// OR
// Below is such if you launch kernel with 1D block (i.e only in x direction) with same thread count
// and 2D total grid size

template <uint BLOCK_SIZE>
__global__ void sgemm_naive_coalesced_1D_kernel(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta)
{
  uint row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  uint col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

  if (row < M && col < N)
  {
    float sum = 0.0f;
    for(int k = 0; k < K; k++)
    {
      sum += A[row * K + k] * B[k*N + col];
    }

    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}

double inline run_coalesced(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M*N*sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice));

  dim3 blockDim(32, 32);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

  sgemm_naive_coalesced_kernel<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C, alpha, beta);

  CHECK_CUDA_KERNEL_LAUNCH();
  CHECK_CUDA(cudaDeviceSynchronize());

  // Measure execution time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for(int i = 0; i < num_runs; i++)
  {
    sgemm_naive_coalesced_kernel<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C, alpha, beta);
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


double inline run_coalesced_1D(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M*N*sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice));


  const uint BLOCK_SIZE = 32;

  dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

  sgemm_naive_coalesced_1D_kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C, alpha, beta);

  CHECK_CUDA_KERNEL_LAUNCH();
  CHECK_CUDA(cudaDeviceSynchronize());

  // Measure execution time
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for(int i = 0; i < num_runs; i++)
  {
    sgemm_naive_coalesced_1D_kernel<BLOCK_SIZE><<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C, alpha, beta);
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
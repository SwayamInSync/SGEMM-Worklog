#include <cublas_v2.h>

#include "cuda_check.cuh"

double run_cublas_sgemm(int M, int N, int K, const float *A, const float *B, float *C, const float alpha, const float beta, int num_runs = 10)
{
  float *d_A, *d_B, *d_C;

  CHECK_CUDA(cudaMalloc((void **)&d_A, M*K*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, K*N*sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_C, M*N*sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
  CHECK_CUDA(cudaDeviceSynchronize());

  // timed runs
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < num_runs; i++)
  {
    // Note: cuBLAS uses column-major ordering, so we compute B@A instead of A@B
    // and swap the M and N dimensions to get the same result
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                d_B, N, d_A, K, &beta, d_C, N);
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

  CHECK_CUDA(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  cublasDestroy(handle);
  CHECK_CUDA(cudaFree(d_A));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_C));

  return milliseconds / num_runs;
}
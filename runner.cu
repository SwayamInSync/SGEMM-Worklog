#include <iomanip>

#include "cuda_check.cuh"
#include "kernels/1-naive.cuh"
#include "kernels/2-naive_global_coalesced.cuh"
#include "kernels/3-shared_mem.cuh"
#include "kernels/4-shared_mem_1D_blocktiling.cuh"
#include "kernels/5-shared_mem_2D_blocktiling.cuh"
#include "kernels/6-shared_mem_2D_blocktiling_coalesced.cuh"
#include "kernels/7-warptiling_vectorized_loads.cuh"
#include "kernels/8-hierarchical_warptiling_float4.cuh"
#include "kernels/9-cutlass-inspired-pingpong-buffering.cuh"
#include "kernels/10-multi-gpu-sgemm.cuh"
#include "utils.hpp"
#include "cublas_sgemm.cuh"


int main(int argc, char **argv)
{
  int M = (argc > 1) ? std::stoi(argv[1]) : 1024;
  int N = (argc > 2) ? std::stoi(argv[2]) : 1024;
  int K = (argc > 3) ? std::stoi(argv[3]) : 1024;
  int num_runs = (argc > 4) ? std::stoi(argv[4]) : 10;
  int kernel_index = (argc > 5) ? std::stoi(argv[5]) : -1;
  bool is_profiling = (argc > 6) ? true : false;

  float alpha = 1.0f;
  float beta = 0.0f;

  // Allocate host memory
  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C = (float *)malloc(M * N * sizeof(float));

  float *C_cublas = (float *)malloc(M * N * sizeof(float));

  srand(42);

  initialize_matrices(A, B, C, M, N, K);
  memcpy(C_cublas, C, M * N * sizeof(float));

  double run_time = 0.0;
  std::string kernel_name;

  switch (kernel_index)
  {
  case 1:
      kernel_name = "Naive SGEMM";
      std::cout << "Running naive kernel..." << std::endl;
      run_time = run(M, N, K, A, B, C, alpha, beta, num_runs);
    break;
  case 2:
      kernel_name = "Naive Coalesced SGEMM";
      std::cout << "Running naive coalesced kernel..." << std::endl;
      run_time = run_coalesced(M, N, K, A, B, C, alpha, beta, num_runs);
    break;
  case 3:
      kernel_name = "Shared Memory SGEMM";
      std::cout << "Running shared memory kernel..." << std::endl;
      run_time = run_shared_mem(M, N, K, A, B, C, alpha, beta, num_runs);
    break;
  case 4:
      kernel_name = "Shared Memory 1D Block Tiling SGEMM";
      std::cout << "Running shared memory 1D block tiling kernel..." << std::endl;
      run_time = run_1D_tiling(M, N, K, A, B, C, alpha, beta, num_runs);
    break;
  case 5:
      kernel_name = "Shared Memory 2D Block Tiling SGEMM";
      std::cout << "Running shared memory 2D block tiling kernel..." << std::endl;
      run_time = run_2D_tiling(M, N, K, A, B, C, alpha, beta, num_runs);
      break;
  case 6:
      kernel_name = "Shared Memory 2D Block Tiling Coalesced SGEMM";
      std::cout << "Running shared memory 2D block tiling coalesced kernel..." << std::endl;
      std::cout << "This is slower but coalesced :)" << std::endl;
      run_time = run_2D_tiling_coalesced(M, N, K, A, B, C, alpha, beta, num_runs);
    break;
  case 7:
      kernel_name = "Vectorized Loads SGEMM";
      std::cout << "Running vectorized loads kernel..." << std::endl;
      run_time = run_vectorized_smem(M, N, K, A, B, C, alpha, beta, num_runs);
    break;
  case 8:
      kernel_name = "Hierarchical Warptiling vectorized SGEMM";
      std::cout << "Running hierarchical warptiling kernel..." << std::endl;
      run_time = run_hwarptiling_vec_kernel(M, N, K, A, B, C, alpha, beta, num_runs);
    break;
  case 9:
      kernel_name = "Ping Pong Buffering SGEMM";
      std::cout << "Running ping pong buffering kernel..." << std::endl;
      run_time = run_pingpong(M, N, K, A, B, C, alpha, beta, num_runs);
    break;
  case 10:
      kernel_name = "Multi-GPU SGEMM";
      std::cout << "Running multi-GPU kernel..." << std::endl;
      run_time = run_pingpong_kernel_multi_gpu(M, N, K, A, B, C, alpha, beta, num_runs);
      break;
  default:
      kernel_name = "Cublas SGEMM";
      std::cout << "Running cuBLAS SGEMM..." << std::endl;
      run_time = run_cublas_sgemm(M, N, K, A, B, C_cublas, alpha, beta);
    break;
  }
  double tflops = calculate_tflops(M, N, K, run_time);

  if (!is_profiling)
  {
    std::cout << "Running cuBLAS SGEMM..." << std::endl;
    double cublas_time = run_cublas_sgemm(M, N, K, A, B, C_cublas, alpha, beta);
    double cublas_tflops = calculate_tflops(M, N, K, cublas_time);
    bool is_correct = verify_results(C, C_cublas, M, N);

    print_results(kernel_name, M, N, K, run_time ,tflops, cublas_time, cublas_tflops, is_correct);
  }

  // Clean up
  free(A);
  free(B);
  free(C);
  free(C_cublas);
  
  return 0;
}
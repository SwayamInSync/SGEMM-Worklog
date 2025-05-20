#pragma once

#include <iostream>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

void initialize_matrices(float *A, float *B, float *C, int M, int N, int K)
{
  for(int i = 0; i < M*K; i++)
  {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for(int i = 0; i < K*N; i++)
  {
    B[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for(int i = 0; i < M*N; i++)
  {
    C[i] = static_cast<float>(rand()) / RAND_MAX;
  }
}


bool verify_results(const float *result, const float *cublas_result, int M, int N, 
                            float rtol = 1e-5, float atol = 1e-8) {
    bool all_close = true;
    int mismatches = 0;
    int max_mismatches_to_show = 5; // Limit the number of mismatches to display
    
    for(int i = 0; i < M*N; i++) {
        float abs_diff = std::abs(result[i] - cublas_result[i]);
        // The formula from NumPy's allclose: absolute(a - b) <= (atol + rtol * absolute(b))
        float tolerance = atol + rtol * std::abs(cublas_result[i]);
        
        if(abs_diff > tolerance) {
            if (mismatches < max_mismatches_to_show) {
                std::cout << "Mismatch at index " << i << ": " << result[i] 
                          << " vs " << cublas_result[i] << ", diff: " << abs_diff 
                          << ", tolerance: " << tolerance << std::endl;
            }
            mismatches++;
            all_close = false;
        }
    }
    
    if (!all_close) {
        std::cout << "Total mismatches: " << mismatches << " out of " << (M*N) << " elements "
                  << "(" << (mismatches * 100.0 / (M*N)) << "%)" << std::endl;
    }
    
    return all_close;
}


double calculate_tflops(int M, int N, int K, double time_ms)
{
  // GEMM requires 2*M*N*K floating point operations (asymptotically)
  // For a complete GEMM (C = alpha*A*B + beta*C), we need to add the operations
  // for the scaling and addition: M*N more operations
  double operations = (2.0 * M * N * K) + (M * N);
  
  // Convert milliseconds to seconds
  double time_s = time_ms / 1000.0;
  
  return (operations / time_s) / 1e12;
}


void print_results(const std::string& kernel_name, int M, int N, int K, double run_time, double tflops, double cublas_time, double cublas_tflops, bool is_correct)
{
  std::cout << "\n=== SGEMM Performance Results for " << kernel_name << " ===" << std::endl;
  std::cout << "Matrix dimensions: " << M << "x" << N << "x" << K << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << kernel_name << ":  " << run_time << " ms, " << tflops << " TFLOPS" << std::endl;
  std::cout << "cuBLAS SGEMM: " << cublas_time << " ms, " << cublas_tflops << " TFLOPS" << std::endl;
  std::cout << "Performance ratio (cuBLAS/" << kernel_name << "): " << cublas_tflops / tflops << "x" << std::endl;
  std::cout << "Correctness check: " << (is_correct ? "PASSED" : "FAILED") << std::endl;
}
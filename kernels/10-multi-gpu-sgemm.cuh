#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "./9-cutlass-inspired-pingpong-buffering.cuh"

double run_pingpong_kernel_multi_gpu(int M, int N, int K, float *h_A, float *h_B, float *h_C, const float alpha, const float beta, int num_runs = 1)
{

  const int BLOCK_SIZE_M_CFG = 64;
  const int BLOCK_SIZE_N_CFG = 128;
  const int BLOCK_SIZE_K_CFG = 16;
  const int WARP_SIZE_M_CFG = 32;
  const int WARP_SIZE_N_CFG = 64;
  const int WARP_STEPS_N_CFG = 1;
  const int THREAD_M_CFG = 4;
  const int THREAD_N_CFG = 4;
  const int MAX_THREADS_CFG = 128;

  dim3 blockDim(MAX_THREADS_CFG);

  int device_count;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  int num_gpus = std::min(device_count, 4);

  if (num_gpus == 0)
  {
    std::cerr << "No CUDA-capable GPUs found." << std::endl;
    return 0.0;
  }
  if (num_runs == 1)
  {
    std::cout << "Using " << num_gpus << " GPU(s)." << std::endl;
  }

  if (K % num_gpus != 0)
  {
    std::cerr << "Error: For this split-K implementation, K (" << K << ") must be divisible by the number of GPUs (" << num_gpus << ")." << std::endl;
    return 0.0;
  }
  int K_per_gpu = K / num_gpus;

  if (K_per_gpu < BLOCK_SIZE_K_CFG)
  {
    std::cerr << "Error: K_per_gpu (" << K_per_gpu << ") is smaller than kernel's BLOCK_SIZE_K_CFG (" << BLOCK_SIZE_K_CFG << "). Adjust parameters or K." << std::endl;
    return 0.0;
  }

  cudaEvent_t overall_start_event, overall_stop_event;
  float total_gpu_time_ms_accumulator = 0.0f;
  float max_kernel_time_accumulator = 0.0f;

  CHECK_CUDA(cudaEventCreate(&overall_start_event));
  CHECK_CUDA(cudaEventCreate(&overall_stop_event));

  std::vector<cudaStream_t> streams(num_gpus);
  std::vector<cudaEvent_t> kernel_start_events(num_gpus);
  std::vector<cudaEvent_t> kernel_stop_events(num_gpus);
  std::vector<float> avg_kernel_times_ms(num_gpus, 0.0f);

  std::vector<float *> d_A_subs(num_gpus);
  std::vector<float *> d_B_subs(num_gpus);
  std::vector<float *> d_C_partials(num_gpus);
  std::vector<float *> h_C_partials_temp(num_gpus);

  float *h_C_original_for_beta = nullptr;
  size_t C_total_elements = (size_t)M * N;
  size_t C_total_size_bytes = C_total_elements * sizeof(float);

  if (beta != 0.0f)
  {
    h_C_original_for_beta = (float *)malloc(C_total_size_bytes);
    if (!h_C_original_for_beta)
    {
      std::cerr << "Failed to allocate host memory for C_original_for_beta." << std::endl;
      CHECK_CUDA(cudaEventDestroy(overall_start_event));
      CHECK_CUDA(cudaEventDestroy(overall_stop_event));
      return 0.0;
    }
    memcpy(h_C_original_for_beta, h_C, C_total_size_bytes);
  }

  for (int i = 0; i < num_gpus; ++i)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaEventCreateWithFlags(&kernel_start_events[i], cudaEventDefault));
    CHECK_CUDA(cudaEventCreateWithFlags(&kernel_stop_events[i], cudaEventDefault));

    size_t size_A_sub = (size_t)M * K_per_gpu * sizeof(float);
    size_t size_B_sub = (size_t)K_per_gpu * N * sizeof(float);
    size_t size_C_partial = C_total_size_bytes;

    CHECK_CUDA(cudaMalloc(&d_A_subs[i], size_A_sub));
    CHECK_CUDA(cudaMalloc(&d_B_subs[i], size_B_sub));
    CHECK_CUDA(cudaMalloc(&d_C_partials[i], size_C_partial));

    h_C_partials_temp[i] = (float *)malloc(size_C_partial);
    if (!h_C_partials_temp[i])
    {
      std::cerr << "Failed to allocate host memory for h_C_partials_temp[" << i << "]" << std::endl;

      for (int k = 0; k < i; ++k)
      {
        if (h_C_partials_temp[k])
          free(h_C_partials_temp[k]);
      }

      for (int k = 0; k < i; ++k)
      {
        CHECK_CUDA(cudaSetDevice(k));
        if (d_A_subs[k])
          CHECK_CUDA(cudaFree(d_A_subs[k]));
        if (d_B_subs[k])
          CHECK_CUDA(cudaFree(d_B_subs[k]));
        if (d_C_partials[k])
          CHECK_CUDA(cudaFree(d_C_partials[k]));
        CHECK_CUDA(cudaEventDestroy(kernel_start_events[k]));
        CHECK_CUDA(cudaEventDestroy(kernel_stop_events[k]));
        CHECK_CUDA(cudaStreamDestroy(streams[k]));
      }
      if (h_C_original_for_beta)
        free(h_C_original_for_beta);
      CHECK_CUDA(cudaEventDestroy(overall_start_event));
      CHECK_CUDA(cudaEventDestroy(overall_stop_event));
      return 0.0;
    }
  }

  std::vector<float> current_run_kernel_times_ms(num_gpus);

  for (int run = 0; run < num_runs; ++run)
  {
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(overall_start_event, streams[0]));

    for (int i = 0; i < num_gpus; ++i)
    {
      CHECK_CUDA(cudaSetDevice(i));

      float *src_A_ptr = h_A + (size_t)i * K_per_gpu;
      CHECK_CUDA(cudaMemcpy2DAsync(d_A_subs[i], (size_t)K_per_gpu * sizeof(float),
                                   src_A_ptr, (size_t)K * sizeof(float),
                                   (size_t)K_per_gpu * sizeof(float), M,
                                   cudaMemcpyHostToDevice, streams[i]));

      float *src_B_ptr = h_B + (size_t)i * K_per_gpu * N;
      size_t size_B_sub = (size_t)K_per_gpu * N * sizeof(float);
      CHECK_CUDA(cudaMemcpyAsync(d_B_subs[i], src_B_ptr, size_B_sub, cudaMemcpyHostToDevice, streams[i]));

      dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE_N_CFG), CEIL_DIV(M, BLOCK_SIZE_M_CFG));

      CHECK_CUDA(cudaEventRecord(kernel_start_events[i], streams[i]));
      pingpong_buffering_kernel<BLOCK_SIZE_M_CFG, BLOCK_SIZE_N_CFG, BLOCK_SIZE_K_CFG,
                                WARP_SIZE_M_CFG, WARP_SIZE_N_CFG, WARP_STEPS_N_CFG,
                                THREAD_M_CFG, THREAD_N_CFG, MAX_THREADS_CFG>
          <<<gridDim, blockDim, 0, streams[i]>>>(
              M, N, K_per_gpu, d_A_subs[i], d_B_subs[i], d_C_partials[i], 1.0f, 0.0f);
      CHECK_CUDA(cudaEventRecord(kernel_stop_events[i], streams[i]));
    }

    memset(h_C, 0, C_total_size_bytes);
    float current_run_max_kernel_time = 0.0f;

    for (int i = 0; i < num_gpus; ++i)
    {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));

      CHECK_CUDA(cudaEventElapsedTime(&current_run_kernel_times_ms[i], kernel_start_events[i], kernel_stop_events[i]));
      avg_kernel_times_ms[i] += current_run_kernel_times_ms[i];

      if (current_run_kernel_times_ms[i] > current_run_max_kernel_time)
      {
        current_run_max_kernel_time = current_run_kernel_times_ms[i];
      }

      CHECK_CUDA(cudaMemcpy(h_C_partials_temp[i], d_C_partials[i], C_total_size_bytes, cudaMemcpyDeviceToHost));

      for (size_t j = 0; j < C_total_elements; ++j)
      {
        h_C[j] += h_C_partials_temp[i][j];
      }
    }
    max_kernel_time_accumulator += current_run_max_kernel_time;

    for (int i = 0; i < num_gpus; ++i)
    {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(overall_stop_event, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(overall_stop_event));

    float current_total_gpu_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&current_total_gpu_time_ms, overall_start_event, overall_stop_event));
    total_gpu_time_ms_accumulator += current_total_gpu_time_ms;
  }

  float final_avg_overall_time_ms = (num_runs > 0) ? total_gpu_time_ms_accumulator / num_runs : 0.0f;
  float final_avg_max_kernel_time_ms = (num_runs > 0) ? max_kernel_time_accumulator / num_runs : 0.0f; // avg since they are run in parallel

  for (int i = 0; i < num_gpus; ++i)
  {
    if (num_runs > 0)
      avg_kernel_times_ms[i] /= num_runs;
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Average Overall multi-GPU processing time (including H2D, D2H, Host Sum): " << final_avg_overall_time_ms << " ms" << std::endl;
  std::cout << "Average Parallel Kernel Execution Time (max across GPUs per run): " << final_avg_max_kernel_time_ms << " ms" << std::endl;
  for (int i = 0; i < num_gpus; ++i)
  {
    std::cout << "GPU " << i << " average individual kernel execution time: " << avg_kernel_times_ms[i] << " ms" << std::endl;
  }

  for (size_t j = 0; j < C_total_elements; ++j)
  {
    float s_val = h_C[j];
    float c_initial_val = (beta == 0.0f || !h_C_original_for_beta) ? 0.0f : h_C_original_for_beta[j];
    h_C[j] = alpha * s_val + beta * c_initial_val;
  }

  if (h_C_original_for_beta)
  {
    free(h_C_original_for_beta);
  }
  CHECK_CUDA(cudaEventDestroy(overall_start_event));
  CHECK_CUDA(cudaEventDestroy(overall_stop_event));

  for (int i = 0; i < num_gpus; ++i)
  {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(d_A_subs[i]));
    CHECK_CUDA(cudaFree(d_B_subs[i]));
    CHECK_CUDA(cudaFree(d_C_partials[i]));
    CHECK_CUDA(cudaEventDestroy(kernel_start_events[i]));
    CHECK_CUDA(cudaEventDestroy(kernel_stop_events[i]));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
    if (h_C_partials_temp[i])
    {
      free(h_C_partials_temp[i]);
    }
  }
  if (num_runs == 1)
  {
    std::cout << "Multi-GPU computation complete." << std::endl;
  }

  return (double)final_avg_max_kernel_time_ms;
}
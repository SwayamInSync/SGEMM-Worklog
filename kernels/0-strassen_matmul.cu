#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

const int TILE_DIM = 16;
const int BASE_CASE_THRESHOLD_STRASSEN = 64;

#define CUDA_CHECK(err)                   \
  {                                       \
    gpuAssert((err), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s at %s:%d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

void strassen_recursive_gpu(
    const float *A_dev, const float *B_dev, float *C_dev,
    int N, int lda, int ldb, int ldc,
    int current_depth, int max_depth_cutoff,
    float *workspace_pool, size_t &current_workspace_offset, size_t total_workspace_size);

__global__ void matrix_add_sub_kernel(const float *A, const float *B, float *C,
                                      int N_rows, int N_cols, int op_type,
                                      int lda, int ldb, int ldc)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N_rows && col < N_cols)
  {
    float val_a = A[row * lda + col];
    float val_b = B[row * ldb + col];
    if (op_type == 1)
    {
      C[row * ldc + col] = val_a + val_b;
    }
    else
    {
      C[row * ldc + col] = val_a - val_b;
    }
  }
}

__global__ void strassen_base_kernel(const float *A, const float *B, float *C,
                                     int N, int lda, int ldb, int ldc)
{
  __shared__ float ds_A[TILE_DIM][TILE_DIM];
  __shared__ float ds_B[TILE_DIM][TILE_DIM];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int C_row = by * TILE_DIM + ty;
  int C_col = bx * TILE_DIM + tx;
  float C_value = 0.0f;

  for (int k_tile = 0; k_tile < (N + TILE_DIM - 1) / TILE_DIM; ++k_tile)
  {
    int A_row_shared = ty;
    int A_col_shared = tx;
    int A_row_global = by * TILE_DIM + A_row_shared;
    int A_col_global = k_tile * TILE_DIM + A_col_shared;
    if (A_row_global < N && A_col_global < N)
    {
      ds_A[A_row_shared][A_col_shared] = A[A_row_global * lda + A_col_global];
    }
    else
    {
      ds_A[A_row_shared][A_col_shared] = 0.0f;
    }

    int B_row_shared = ty;
    int B_col_shared = tx;
    int B_row_global = k_tile * TILE_DIM + B_row_shared;
    int B_col_global = bx * TILE_DIM + B_col_shared;
    if (B_row_global < N && B_col_global < N)
    {
      ds_B[B_row_shared][B_col_shared] = B[B_row_global * ldb + B_col_global];
    }
    else
    {
      ds_B[B_row_shared][B_col_shared] = 0.0f;
    }
    __syncthreads();

    if (C_row < N && C_col < N)
    {
      for (int i = 0; i < TILE_DIM; ++i)
      {
        C_value += ds_A[ty][i] * ds_B[i][tx];
      }
    }
    __syncthreads();
  }
  if (C_row < N && C_col < N)
  {
    C[C_row * ldc + C_col] = C_value;
  }
}

__global__ void copy_and_pad_kernel_d2d(const float *M_in_dev, float *M_out_padded_dev,
                                        int N_orig, int M_orig,
                                        int N_padded, int M_padded,
                                        int ld_in_dev, int ld_out_padded_dev)
{
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < N_padded && c < M_padded)
  {
    if (r < N_orig && c < M_orig)
    {
      M_out_padded_dev[r * ld_out_padded_dev + c] = M_in_dev[r * ld_in_dev + c];
    }
    else
    {
      M_out_padded_dev[r * ld_out_padded_dev + c] = 0.0f;
    }
  }
}

__global__ void copy_and_unpad_kernel_d2d(const float *M_in_padded_dev, float *M_out_unpadded_dev,
                                          int N_orig, int M_orig,
                                          int N_padded, int M_padded,
                                          int ld_in_padded_dev, int ld_out_unpadded_dev)
{
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;

  if (r < N_orig && c < M_orig)
  {
    if (r < N_padded && c < M_padded)
    {
      M_out_unpadded_dev[r * ld_out_unpadded_dev + c] = M_in_padded_dev[r * ld_in_padded_dev + c];
    }
  }
}

float strassen_gpu_multiply(const float *h_A, const float *h_B, float *h_C, int N_orig, int &out_padded_N)
{
  if (N_orig == 0)
  {
    out_padded_N = 0;
    return 0.0f;
  }

  int padded_N = N_orig;
  if (N_orig > BASE_CASE_THRESHOLD_STRASSEN)
  {
    padded_N = 1;
    while (padded_N < N_orig)
    {
      padded_N <<= 1;
    }
  }
  out_padded_N = padded_N;

  float *d_A_orig, *d_B_orig;
  size_t N_orig_bytes = (size_t)N_orig * N_orig * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_A_orig, N_orig_bytes));
  CUDA_CHECK(cudaMalloc(&d_B_orig, N_orig_bytes));

  CUDA_CHECK(cudaMemcpy(d_A_orig, h_A, N_orig_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B_orig, h_B, N_orig_bytes, cudaMemcpyHostToDevice));

  float *d_A_padded, *d_B_padded, *d_C_padded;
  size_t padded_N_bytes = (size_t)padded_N * padded_N * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_A_padded, padded_N_bytes));
  CUDA_CHECK(cudaMalloc(&d_B_padded, padded_N_bytes));
  CUDA_CHECK(cudaMalloc(&d_C_padded, padded_N_bytes));

  dim3 copy_threads(16, 16);
  dim3 pad_blocks((padded_N + copy_threads.x - 1) / copy_threads.x,
                  (padded_N + copy_threads.y - 1) / copy_threads.y);

  copy_and_pad_kernel_d2d<<<pad_blocks, copy_threads>>>(d_A_orig, d_A_padded, N_orig, N_orig, padded_N, padded_N, N_orig, padded_N);
  CUDA_CHECK(cudaGetLastError());
  copy_and_pad_kernel_d2d<<<pad_blocks, copy_threads>>>(d_B_orig, d_B_padded, N_orig, N_orig, padded_N, padded_N, N_orig, padded_N);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_A_orig));
  CUDA_CHECK(cudaFree(d_B_orig));

  CUDA_CHECK(cudaMemset(d_C_padded, 0, padded_N_bytes));

  float *d_workspace = nullptr;
  size_t workspace_total_bytes = 0;
  size_t current_workspace_offset_elements = 0;

  bool use_strassen = (padded_N > BASE_CASE_THRESHOLD_STRASSEN) && ((padded_N & (padded_N - 1)) == 0);

  if (use_strassen)
  {
    workspace_total_bytes = (size_t)(3.0 * padded_N * padded_N * sizeof(float));
    if (workspace_total_bytes > 0)
    {
      CUDA_CHECK(cudaMalloc(&d_workspace, workspace_total_bytes));
      if (d_workspace == nullptr && workspace_total_bytes > 0)
      {
        fprintf(stderr, "Warning: Workspace cudaMalloc reported success but returned nullptr. Falling back.\n");
        use_strassen = false;
      }
    }
    else
    {
      use_strassen = false;
    }
  }

  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  CUDA_CHECK(cudaEventRecord(start_event, 0));

  if (use_strassen)
  {
    int max_rec_depth = 0;
    if (padded_N > BASE_CASE_THRESHOLD_STRASSEN)
    {
      max_rec_depth = static_cast<int>(floor(log2(static_cast<double>(padded_N) / BASE_CASE_THRESHOLD_STRASSEN)));
    }
    strassen_recursive_gpu(d_A_padded, d_B_padded, d_C_padded, padded_N,
                           padded_N, padded_N, padded_N,
                           0, max_rec_depth,
                           d_workspace, current_workspace_offset_elements, workspace_total_bytes / sizeof(float));
  }
  else
  {
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((padded_N + TILE_DIM - 1) / TILE_DIM, (padded_N + TILE_DIM - 1) / TILE_DIM);
    strassen_base_kernel<<<numBlocks, threadsPerBlock>>>(d_A_padded, d_B_padded, d_C_padded, padded_N, padded_N, padded_N, padded_N);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(stop_event, 0));
  CUDA_CHECK(cudaEventSynchronize(stop_event));

  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));

  float *d_C_unpadded;
  CUDA_CHECK(cudaMalloc(&d_C_unpadded, N_orig_bytes));

  dim3 unpad_blocks((N_orig + copy_threads.x - 1) / copy_threads.x,
                    (N_orig + copy_threads.y - 1) / copy_threads.y);
  copy_and_unpad_kernel_d2d<<<unpad_blocks, copy_threads>>>(d_C_padded, d_C_unpadded, N_orig, N_orig, padded_N, padded_N, padded_N, N_orig);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_C, d_C_unpadded, N_orig_bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_C_unpadded));
  if (d_workspace)
    CUDA_CHECK(cudaFree(d_workspace));
  CUDA_CHECK(cudaFree(d_A_padded));
  CUDA_CHECK(cudaFree(d_B_padded));
  CUDA_CHECK(cudaFree(d_C_padded));

  return milliseconds;
}

void strassen_recursive_gpu(
    const float *A_dev, const float *B_dev, float *C_dev,
    int N, int lda, int ldb, int ldc,
    int current_depth, int max_depth_cutoff,
    float *workspace_pool, size_t &current_workspace_offset_elements,
    size_t total_workspace_elements)
{

  if (N <= BASE_CASE_THRESHOLD_STRASSEN || current_depth > max_depth_cutoff || (N % 2 != 0 && N > 1))
  {
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    strassen_base_kernel<<<numBlocks, threadsPerBlock>>>(A_dev, B_dev, C_dev, N, lda, ldb, ldc);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  int n_half = N / 2;

  const float *A11 = A_dev;
  const float *A12 = A_dev + n_half;
  const float *A21 = A_dev + (size_t)n_half * lda;
  const float *A22 = A_dev + (size_t)n_half * lda + n_half;

  const float *B11 = B_dev;
  const float *B12 = B_dev + n_half;
  const float *B21 = B_dev + (size_t)n_half * ldb;
  const float *B22 = B_dev + (size_t)n_half * ldb + n_half;

  float *C11 = C_dev;
  float *C12 = C_dev + n_half;
  float *C21 = C_dev + (size_t)n_half * ldc;
  float *C22 = C_dev + (size_t)n_half * ldc + n_half;

  size_t sub_matrix_elems = (size_t)n_half * n_half;

  if (current_workspace_offset_elements + 9 * sub_matrix_elems > total_workspace_elements)
  {
    fprintf(stderr, "Error: Insufficient workspace for Strassen recursion at depth %d, N=%d. Needed %zu elements, have %zu available from current offset %zu (total %zu). Falling back to base kernel.\n",
            current_depth, N, 9 * sub_matrix_elems,
            (total_workspace_elements > current_workspace_offset_elements ? total_workspace_elements - current_workspace_offset_elements : 0),
            current_workspace_offset_elements, total_workspace_elements);
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    strassen_base_kernel<<<numBlocks, threadsPerBlock>>>(A_dev, B_dev, C_dev, N, lda, ldb, ldc);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  size_t p_and_s_start_offset_this_call_elements = current_workspace_offset_elements;

  float *P1 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;
  float *P2 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;
  float *P3 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;
  float *P4 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;
  float *P5 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;
  float *P6 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;
  float *P7 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;
  float *temp_S1 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;
  float *temp_S2 = workspace_pool + current_workspace_offset_elements;
  current_workspace_offset_elements += sub_matrix_elems;

  dim3 add_sub_threads(16, 16);
  dim3 add_sub_blocks((n_half + add_sub_threads.x - 1) / add_sub_threads.x,
                      (n_half + add_sub_threads.y - 1) / add_sub_threads.y);

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(B12, B22, temp_S1, n_half, n_half, -1, ldb, ldb, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  strassen_recursive_gpu(A11, temp_S1, P1, n_half, lda, n_half, n_half, current_depth + 1, max_depth_cutoff, workspace_pool, current_workspace_offset_elements, total_workspace_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(A11, A12, temp_S1, n_half, n_half, 1, lda, lda, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  strassen_recursive_gpu(temp_S1, B22, P2, n_half, n_half, ldb, n_half, current_depth + 1, max_depth_cutoff, workspace_pool, current_workspace_offset_elements, total_workspace_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(A21, A22, temp_S1, n_half, n_half, 1, lda, lda, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  strassen_recursive_gpu(temp_S1, B11, P3, n_half, n_half, ldb, n_half, current_depth + 1, max_depth_cutoff, workspace_pool, current_workspace_offset_elements, total_workspace_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(B21, B11, temp_S1, n_half, n_half, -1, ldb, ldb, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  strassen_recursive_gpu(A22, temp_S1, P4, n_half, lda, n_half, n_half, current_depth + 1, max_depth_cutoff, workspace_pool, current_workspace_offset_elements, total_workspace_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(A11, A22, temp_S1, n_half, n_half, 1, lda, lda, n_half);
  CUDA_CHECK(cudaGetLastError());
  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(B11, B22, temp_S2, n_half, n_half, 1, ldb, ldb, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  strassen_recursive_gpu(temp_S1, temp_S2, P5, n_half, n_half, n_half, n_half, current_depth + 1, max_depth_cutoff, workspace_pool, current_workspace_offset_elements, total_workspace_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(A12, A22, temp_S1, n_half, n_half, -1, lda, lda, n_half);
  CUDA_CHECK(cudaGetLastError());
  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(B21, B22, temp_S2, n_half, n_half, 1, ldb, ldb, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  strassen_recursive_gpu(temp_S1, temp_S2, P6, n_half, n_half, n_half, n_half, current_depth + 1, max_depth_cutoff, workspace_pool, current_workspace_offset_elements, total_workspace_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(A11, A21, temp_S1, n_half, n_half, -1, lda, lda, n_half);
  CUDA_CHECK(cudaGetLastError());
  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(B11, B12, temp_S2, n_half, n_half, 1, ldb, ldb, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  strassen_recursive_gpu(temp_S1, temp_S2, P7, n_half, n_half, n_half, n_half, current_depth + 1, max_depth_cutoff, workspace_pool, current_workspace_offset_elements, total_workspace_elements);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(P5, P4, temp_S1, n_half, n_half, 1, n_half, n_half, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(temp_S1, P2, temp_S2, n_half, n_half, -1, n_half, n_half, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(temp_S2, P6, C11, n_half, n_half, 1, n_half, n_half, ldc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(P1, P2, C12, n_half, n_half, 1, n_half, n_half, ldc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(P3, P4, C21, n_half, n_half, 1, n_half, n_half, ldc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(P5, P1, temp_S1, n_half, n_half, 1, n_half, n_half, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(temp_S1, P3, temp_S2, n_half, n_half, -1, n_half, n_half, n_half);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  matrix_add_sub_kernel<<<add_sub_blocks, add_sub_threads>>>(temp_S2, P7, C22, n_half, n_half, -1, n_half, n_half, ldc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  current_workspace_offset_elements = p_and_s_start_offset_this_call_elements;
}

void print_matrix(const float *M, int rows, int cols, int ld, const std::string &name)
{
  if (rows > 16 || cols > 16)
  {
    return;
  }
  std::cout << name << " (" << rows << "x" << cols << ", ld=" << ld << "):\n";
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      std::cout << M[i * ld + j] << "\t";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

void cpu_matrix_mult(const float *A, const float *B, float *C, int N, int lda, int ldb, int ldc)
{
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      C[i * ldc + j] = 0.0f;
      for (int k = 0; k < N; ++k)
      {
        C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
      }
    }
  }
}

double calculate_tflops(long long M, long long N, long long K, double time_ms)
{

  double operations = (2.0 * M * N * K) + (1.0 * M * N);

  double time_s = time_ms / 1000.0;

  if (time_s == 0)
    return 0.0;

  return (operations / time_s) / 1e12;
}

int main(int argc, char **argv)
{
  int N_orig = 128;
  if (argc > 1)
  {
    N_orig = std::atoi(argv[1]);
    if (N_orig <= 0)
    {
      std::cerr << "Matrix size N_orig must be positive." << std::endl;
      return 1;
    }
  }
  std::cout << "Performing Strassen CUDA matrix multiplication for N_orig = " << N_orig << std::endl;
  std::cout << "BASE_CASE_THRESHOLD_STRASSEN = " << BASE_CASE_THRESHOLD_STRASSEN << std::endl;
  std::cout << "TILE_DIM (for base kernel) = " << TILE_DIM << std::endl;

  std::vector<float> h_A(N_orig * N_orig);
  std::vector<float> h_B(N_orig * N_orig);
  std::vector<float> h_C_gpu(N_orig * N_orig);
  std::vector<float> h_C_cpu(N_orig * N_orig);

  for (int i = 0; i < N_orig * N_orig; ++i)
  {
    h_A[i] = static_cast<float>(rand() % 10) / 2.0f + 0.1f;
    h_B[i] = static_cast<float>(rand() % 10) / 2.0f + 0.1f;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Cleared pre-existing CUDA error: %s\n", cudaGetErrorString(err));
  }

  float elapsed_ms = 0.0f;
  int padded_N_for_tflops = 0;

  try
  {
    elapsed_ms = strassen_gpu_multiply(h_A.data(), h_B.data(), h_C_gpu.data(), N_orig, padded_N_for_tflops);
  }
  catch (const std::runtime_error &e)
  {
    std::cerr << "A runtime error occurred: " << e.what() << std::endl;
    cudaDeviceReset();
    return 1;
  }
  catch (...)
  {
    std::cerr << "An unknown error occurred during GPU computation." << std::endl;
    cudaDeviceReset();
    return 1;
  }

  std::cout << "GPU (Strassen) computation time: " << elapsed_ms << " ms" << std::endl;
  if (padded_N_for_tflops > 0)
  {
    double tflops = calculate_tflops(padded_N_for_tflops, padded_N_for_tflops, padded_N_for_tflops, elapsed_ms);
    std::cout << "Equivalent TFLOPS (based on padded_N=" << padded_N_for_tflops << "): " << tflops << std::endl;
  }

  cpu_matrix_mult(h_A.data(), h_B.data(), h_C_cpu.data(), N_orig, N_orig, N_orig, N_orig);

  bool correct = true;
  float max_error = 0.0f, total_error = 0.0f;
  float tolerance = (N_orig > 128) ? 5e-2f : 1e-2f;

  for (int i = 0; i < N_orig; ++i)
  {
    for (int j = 0; j < N_orig; ++j)
    {
      float gpu_val = h_C_gpu[i * N_orig + j];
      float cpu_val = h_C_cpu[i * N_orig + j];
      float error = std::abs(gpu_val - cpu_val);
      total_error += error;
      if (error > max_error)
        max_error = error;
      if (error > tolerance)
      {
        if (std::abs(cpu_val) > 1e-4 && (error / std::abs(cpu_val)) > tolerance)
        {
          if (correct)
            fprintf(stderr, "Verification FAILED at C[%d][%d]: GPU=%.6f, CPU=%.6f, abs_err=%.6f, rel_err=%.6f\n", i, j, gpu_val, cpu_val, error, error / std::abs(cpu_val));
          correct = false;
        }
        else if (std::abs(cpu_val) <= 1e-4 && error > tolerance)
        {
          if (correct)
            fprintf(stderr, "Verification FAILED at C[%d][%d] (CPU val near zero): GPU=%.6f, CPU=%.6f, abs_err=%.6f\n", i, j, gpu_val, cpu_val, error);
          correct = false;
        }
      }
    }
  }

  print_matrix(h_A.data(), N_orig, N_orig, N_orig, "Host A");
  print_matrix(h_B.data(), N_orig, N_orig, N_orig, "Host B");
  print_matrix(h_C_gpu.data(), N_orig, N_orig, N_orig, "GPU C (Strassen)");
  print_matrix(h_C_cpu.data(), N_orig, N_orig, N_orig, "CPU C (Reference)");

  if (correct)
    std::cout << "Verification PASSED!" << std::endl;
  else
    std::cout << "Verification FAILED!" << std::endl;
  std::cout << "Max absolute error: " << max_error << std::endl;
  std::cout << "Average absolute error: " << ((N_orig > 0) ? (total_error / ((float)N_orig * N_orig)) : 0.0f) << std::endl;

  cudaDeviceReset();
  return 0;
}

# SGEMM-Worklog

SGEMM (Single-precision General Matrix Multiply) computes $C = \alpha AB + \beta C$ where $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$, and $C \in \mathbb{R}^{M \times N}$.

## Compute vs Memory Intensity Analysis

For matrix dimensions $M = N = K = n$, SGEMM $(C = \alpha AB + \beta C)$ requires:
- **Arithmetic operations**: $(2n^3 + n^2)$ FLOPs 
  - $2n^3$ for matrix multiplication $AB$ 
  - $n^2$ for scaling and addition $\alpha(AB) + \beta C$
- **Memory transfers**: $(4n^2)$ elements 
  - Read $A$: $n^2$, Read $B$: $n^2$, Read $C$: $n^2$, Write $C$: $n^2$

The **arithmetic intensity** is:
$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}} = \frac{2n^3 + n^2}{4n^2 \times 4} = \frac{2n^3 + n^2}{16n^2} = \frac{2n + 1}{16} \text{ FLOPs/byte}$



The **A100 performance characteristics**:
- Peak compute: ~312 TFLOPS (FP32)
- Memory bandwidth: ~1.6 TB/s
- **Compute-to-bandwidth ratio**: ~195 FLOPs/byte

For the higer end $N > 2048$ SGEMM's AI significantly exceeds the A100's compute-to-bandwidth ratio, SGEMM is **compute-bound** in this range. Performance is limited by ALU throughput rather than memory bandwidth, making computational optimizations (tiling, vectorization, instruction-level parallelism) more impactful than memory access patterns.
For smaller N, this is more of memory bound, I am thinking to cover this too (but maybe in different worklog)

> All experiments till "Distributed SGEMM" are conducted on NVIDIA A100, Switching to Hopper for the rest


## Table of Contents
- [Naive SGEMM](#naive)
- [Naive Coalesced SGEMM](#naive-coalesced)
- [Shared Memory Tiled Matrix Multiplication](#tiled-matrix-multiplication-using-shared-memory)
- [1D Block Tiling with Shared Memory](#tiled-matrix-multiplication-shared-memroy--1d-blocktiling)
- [2D Block Tiling with Shared Memory](#2d-block-tiling)
- [2D Block Tiling + Coalesced + Padding + Transposed loading of A](#2d-block-tiling-coalesced--padding--transposed-a_shared-loads)
- [Warp tiling + Vectorized Loads](#float4-loads--warptiling)
- [Hierarchical Warp Tiling + Float4 Loads + Thread Group Swizzling](#hierarchical-warptiling--float4-loads--thread-group-swizzling)
- [Ping-Pong Buffering Inspired from CUTLASS](#ping-pong-buffering-inspired-from-cutlass)
- [Distributed SGEMM over multiple GPUs](#distributed-sgemm-over-multiple-gpus)
- [Utilizing Tensor Cores](#) [TBA]

## Naive
- Matrix A: Strided access with stride K → Not coalesced, inefficient.
- Matrix B: BroadcA_sharedt access (same location per warp) → Efficiently coalesced, optimal.
- Matrix C: Strided access with stride N → Not coalesced, inefficient.
```
=== SGEMM Performance Results ===
Matrix dimensions: 4092x4092x4092
Naive SGEMM:  156.46 ms, 0.88 TFLOPS
cuBLA_shared SGEMM: 7.29 ms, 18.81 TFLOPS
Performance ratio (cuBLA_shared/naive): 21.48x
Correctness check: PASSED
```

```
Section: Memory Workload Analysis
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second        42.79
    Mem Busy                               %        49.66
    Max Bandwidth                          %        22.43
    L1/TEX Hit Rate                        %        99.23
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %        68.79
    Mem Pipes Busy                         %        18.04
    --------------------------- ------------ ------------
```

## Naive-Coalesced
- [`kernels/naive_global_coalesced.cuh`](kernels/naive_global_coalesced.cuh)
- Matrix A: Accesses are broadcA_sharedted, not coalesced in the traditional sense, but still efficient due to CUDA’s broadcA_sharedt mechanism.
  - `threadIdx.y` (row) moves slowly than `threadIdx.x` (cols)
  - hence for consecutive threads in a block, they are accessing same row (broadcA_sharedting)
- Matrix B: Accesses are perfectly coalesced, with consecutive threads accessing consecutive memory locations.
  - Depending on `threadIdx.x`
  - for a fixed k, the warp access `B[k*N + col]` to `B[k*N + col+31]` which are 32 consecutive elements in memory
- Matrix C: Both reads and writes are coalesced, leveraging contiguous memory access patterns.
  - `C[row * N + col]`
  - with row fixed across warp, hence warp accesses `C[row*N+col]` to `C[row*N + col + 31]`
  - 32 consecutive read and writes, hence coalesced

```
=== SGEMM Performance Results ===
Matrix dimensions: 4092x4092x4092
Naive SGEMM:  52.06 ms, 2.63 TFLOPS
cuBLA_shared SGEMM: 7.32 ms, 18.72 TFLOPS
Performance ratio (cuBLA_shared/naive): 7.11x
Correctness check: PASSED
```
```
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second       135.52
    Mem Busy                               %        60.91
    Max Bandwidth                          %        60.87
    L1/TEX Hit Rate                        %        95.12
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %        58.55
    Mem Pipes Busy                         %        57.15
    --------------------------- ------------ ------------
```

## Tiled Matrix Multiplication, using shared memory
- [`kernels/3-shared_mem.cuh`](kernels/3-shared_mem.cuh)
```
Algorithm: Tiled Matrix Multiplication with Shared Memory

Input: 
  - Matrices A (M×K), B (K×N)
  - Scalars alpha, beta
Output: 
  - Matrix C = alpha * (A×B) + beta * C

Constants:
  - BLOCK_SIZE: Size of tile (e.g., 32×32)

Kernel Structure:
  - Grid: 2D grid of (⌈N/BLOCK_SIZE⌉, ⌈M/BLOCK_SIZE⌉) blocks
  - Block: 2D block of (BLOCK_SIZE, BLOCK_SIZE) threads

For each block in parallel:
  1. Identify tile coordinates:
     tile_row = blockIdx.y
     tile_col = blockIdx.x

  2. Identify thread's local position within tile:
     local_row = threadIdx.y
     local_col = threadIdx.x

  3. Calculate global position in output matrix C:
     global_row = tile_row * BLOCK_SIZE + local_row
     global_col = tile_col * BLOCK_SIZE + local_col

  4. Allocate shared memory for A and B tiles:
     shared A_tile[BLOCK_SIZE][BLOCK_SIZE]
     shared B_tile[BLOCK_SIZE][BLOCK_SIZE]

  5. Initialize accumulator:
     sum = 0.0

  6. Calculate number of tiles along K dimension:
     num_k_tiles = ⌈K/BLOCK_SIZE⌉

  7. For each k_tile from 0 to num_k_tiles-1:
     a. Calculate tile starting positions:
        a_tile_start_row = tile_row * BLOCK_SIZE
        a_tile_start_col = k_tile * BLOCK_SIZE
        b_tile_start_row = k_tile * BLOCK_SIZE
        b_tile_start_col = tile_col * BLOCK_SIZE

     b. Collaboratively load A tile:
        if (a_tile_start_row + local_row < M AND a_tile_start_col + local_col < K):
            A_tile[local_row][local_col] = A[(a_tile_start_row + local_row) * K + (a_tile_start_col + local_col)]
        else:
            A_tile[local_row][local_col] = 0.0  // Zero padding for boundary cA_sharedes

     c. Collaboratively load B tile:
        if (b_tile_start_row + local_row < K AND b_tile_start_col + local_col < N):
            B_tile[local_row][local_col] = B[(b_tile_start_row + local_row) * N + (b_tile_start_col + local_col)]
        else:
            B_tile[local_row][local_col] = 0.0  // Zero padding for boundary cA_sharedes

     d. Synchronize all threads in block

     e. Compute partial dot product using current tiles:
        for k from 0 to BLOCK_SIZE-1:
            if (k_tile * BLOCK_SIZE + k < K):  // Boundary check for K dimension
                sum += A_tile[local_row][k] * B_tile[k][local_col]

     f. Synchronize all threads in block before next iteration

  8. Write final result to global memory:
     if (global_row < M AND global_col < N):
         C[global_row * N + global_col] = alpha * sum + beta * C[global_row * N + global_col]
```

```
=== SGEMM Performance Results ===
Matrix dimensions: 4096x4096x4096
Naive SGEMM:  27.39 ms, 5.02 TFLOPS
cuBLA_shared SGEMM: 7.28 ms, 18.89 TFLOPS
Performance ratio (cuBLA_shared/naive): 3.76x
Correctness check: PASSED
```

```
Section: Memory Workload Analysis
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second       254.86
    Mem Busy                               %        88.58
    Max Bandwidth                          %        84.63
    L1/TEX Hit Rate                        %         0.42
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %        49.78
    Mem Pipes Busy                         %        74.49
    --------------------------- ------------ ------------

Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           32
    Block Limit Registers                 block            2
    Block Limit Shared Mem                block            3
    Block Limit Warps                     block            2
    Theoretical Active Warps per SM        warp           64
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        99.62
    Achieved Active Warps Per SM           warp        63.76
    ------------------------------- ----------- ------------

Section: Command line profiler metrics
    -------------------------------------------------------------------- -------------- ------------
    Metric Name                                                             Metric Unit Metric Value
    -------------------------------------------------------------------- -------------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio sector/request         4.00
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio sector/request            4
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                  191313
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                 8596809
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                  3221765107
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                   142814537
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                             request    134742016
    l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                             request       524288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                               sector    538968064
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                               sector      2097152
    smsp__sA_shareds_average_data_bytes_per_sector_mem_global_op_ld.pct                     %          100
    smsp__sA_shareds_average_data_bytes_per_sector_mem_global_op_st.pct                     %          100
    smsp__sA_shareds_average_data_bytes_per_wavefront_mem_shared.pct                        %           70
    -------------------------------------------------------------------- -------------- ------------
```

---

**Performance Note: SGEMM Kernel Shared Memory Analysis**

Profiling of the `sgemm_shared_mem_kernel` (with 32x32 `BLOCK_SIZE`, launched A_shared `(128,128,1)` blocks of `(32,32,1)` threads) using NVIDIA Nsight Compute revealed insights into shared memory performance:

* **High Shared Memory Store Contention:** A significant number of "store bank conflicts" for shared memory operations were reported (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` wA_shared approximately 8.6 million). In contrA_sharedt, shared memory load bank conflicts were negligible.
* **Sub-Optimal Shared Memory Access Utilization:** The average data bytes utilized per shared memory access wavefront (`smsp__sA_shareds_average_data_bytes_per_wavefront_mem_shared.pct`) wA_shared around 70%, indicating some inefficiency in data movement during shared memory operations.
* **Code Pattern vs. Profiler Data:** The kernel's shared memory write pattern (e.g., `A_shared[local_row][local_col] = ...`, where `local_row = threadIdx.y` and `local_col = threadIdx.x`) is designed to be bank-conflict-free according to standard CUDA memory access principles (i.e., `local_col` varies 0-31 per warp, mapping to distinct banks).

**Conclusion:**
Despite the shared memory write addressing pattern being theoretically conflict-free, the Nsight Compute metrics indicate that these store operations are a key performance bottleneck. The reported "store bank conflicts" likely point to broader contention or throughput limitations within the GPU's L1/shared memory store pathway for this kernel, rather than purely an address-bank mapping issue with the current code structure. These shared memory store issues are a primary contributor to previously oB_sharederved MIO (Memory Input/Output) pipeline stalls. Further investigation could explore strategies to reduce pressure on this store path, such A_shared using wider data types for shared memory writes if feA_sharedible.

> I see, so the MIO (memory input/output) stalls are happening because warp is waiting for the shared memory access to compelete. In other word need to push less shared memory instructions.

---

## Tiled Matrix Multiplication, Shared Memroy + 1D BlockTiling
- BA_sharedic idea is that now one thread will compute the output for multiple rows (8 in our code)
- [kernels/4-shared_mem_1D_blocktiling.cuh](kernels/4-shared_mem_1D_blocktiling.cuh)

```
=== SGEMM Performance Results ===
Matrix dimensions: 4096x4096x4096
Naive SGEMM:  12.69 ms, 10.83 TFLOPS
cuBLA_shared SGEMM: 7.28 ms, 18.89 TFLOPS
Performance ratio (cuBLA_shared/naive): 1.74x
Correctness check: PASSED
```

```
    Section: Memory Workload Analysis
    --------------------------- ------------ ------------
    Metric Name                  Metric Unit Metric Value
    --------------------------- ------------ ------------
    Memory Throughput           Gbyte/second       159.60
    Mem Busy                               %        82.51
    Max Bandwidth                          %        72.99
    L1/TEX Hit Rate                        %         1.03
    L2 Compression Success Rate            %            0
    L2 Compression Ratio                                0
    L2 Hit Rate                            %        64.82
    Mem Pipes Busy                         %        48.69
    --------------------------- ------------ ------------
    
    Section: Command line profiler metrics
    -------------------------------------------------------------------- -------------- ------------
    Metric Name                                                             Metric Unit Metric Value
    -------------------------------------------------------------------- -------------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio sector/request         4.00
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio sector/request            4
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                   64565
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                 3010783
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                  1342347365
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                    70119647
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                             request     67633152
    l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                             request       524288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                               sector    270532410
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                               sector      2097152
    smsp__sA_shareds_average_data_bytes_per_sector_mem_global_op_ld.pct                     %          100
    smsp__sA_shareds_average_data_bytes_per_sector_mem_global_op_st.pct                     %          100
    smsp__sA_shareds_average_data_bytes_per_wavefront_mem_shared.pct                        %        28.57
    -------------------------------------------------------------------- -------------- ------------
```

**Important Metrics for Shared Memory 1D Block Tiling Kernel**

| Metric                            | Metric Unit   | Metric Value |
| :-------------------------------- | :------------ | :----------- |
| **Duration** | msecond       | **16.80** |
| Achieved FP32 Peak                | %             | **56** |
| DRAM Throughput                   | %             | 8.24         |
| L2 Cache Throughput               | %             | 14.39        |
| SM Busy                           | %             | 56.70        |
| Compute (SM) Throughput           | %             | 56.49        |
| Executed Ipc Active               | inst/cycle    | 1.75         |
| Memory Throughput                 | Gbyte/second  | 159.52       |
| Mem Busy                          | %             | 82.51        |
| L1/TEX Hit Rate                   | %             | 1.03         |
| L2 Hit Rate                       | %             | **64.82** |
| Mem Pipes Busy                    | %             | 48.69        |
| Issued Warp Per Scheduler         |               | 0.44         |
| No Eligible                       | %             | 56.34        |
| Active Warps Per Scheduler        | warp          | 7.90         |
| Eligible Warps Per Scheduler      | warp          | 1.20         |
| **Warp Cycles Per Issued Instruction** | cycle         | **18.10** |
| Achieved Occupancy                | %             | **49.38** |
| Achieved Active Warps Per SM      | warp          | 31.60        |
| Registers Per Thread              | register/thread| 44           |
| Block Limit Registers             | block         | 2            |
| FMA is highest-utilized pipeline  | %             | **56.7** |




---


The profiling data clearly shows that your **1D block tiling kernel ($16.80 \text{ ms}$ duration)** is significantly fA_sharedter than the naive shared memory kernel ($34.48 \text{ ms}$ duration), achieving approximately a **2x speedup**. This improvement comes primarily from better efficiency in compute execution and memory access patterns, even with lower occupancy.

**Major Gains in Tiled Kernel:**

1.  **Reduced Stalls and Improved Execution Efficiency:** The most impactful metric is the **dramatic drop in "Warp Cycles Per Issued Instruction"** from **38.63 cycles** in the naive kernel down to **18.10 cycles**. This signifies that warps are spending *much less time stalled* waiting for operations (especially memory operations) between instruction issues. The naive kernel explicitly reported MIO stalls being a major contributor (49.2% of stall time), which the tiling strategy effectively mitigates.
2.  **Higher Compute Utilization:** The **achieved percentage of peak FP32 performance more than doubled**, from **27% to 56%**. This is directly reflected in the **doubled FMA pipeline utilization** (28.9% to 56.7%). By reducing memory bottlenecks, warps are able to feed the floating-point units more continuously when they are active.
3.  **Improved Memory Locality:** The **L2 Hit Rate increA_shareded significantly** from 49.77% to 64.82%. This indicates that the tiling strategy improved data reuse in the L2 cache, leading to fewer costly accesses to DRAM. Consequently, **DRAM Throughput (%) decreA_shareded** (13.18% to 8.24%) and **Memory Pipes Busy (%) decreA_shareded** (74.51% to 48.69%), reducing pressure on the global memory suB_sharedystem.

**Losses/Trade-offs in Tiled Kernel:**

1.  **Lower Occupancy:** The **Achieved Occupancy dropped significantly** from 99.63% to 49.38%. This is due to increA_shareded register usage per thread (32 to 44), which, combined with the block size, limits the number of blocks and warps that can be resident on an SM simultaneously. The profiler confirms the kernel is **limited by registers per block**. This means fewer warps are available to hide latency.
2.  **Fewer Active/Eligible Warps:** A_shared a direct result of lower occupancy, there are fewer active and eligible warps per scheduler/SM.


The profiling data confirms that the **tiling strategy successfully addressed the primary bottleneck** in the naive kernel, which wA_shared evidently severe memory stalling (MIO stalls). By improving data locality and reducing the stall time between instructions, the tiled kernel allows the computational units (FMA pipelines) to be utilized far more effectively.

While the tiled kernel hA_shared lower occupancy due to increA_shareded register usage and smaller blocks, the significant improvement in **per-warp execution efficiency** (less time stalled) and the corresponding increA_sharede in **compute throughput** (higher FMA utilization and achieved FP32 peak) overwhelmingly contribute to the oB_sharederved 2x speedup. This highlights that achieving high performance often depends more on minimizing stalls and maximizing the efficiency of active warps rather than solely pursuing 100% occupancy if the warps are frequently idle.

The kernel is still memory-bound to some extent ("Mem Busy" 82.51%) and warps are still stalled frequently ("No Eligible" 56.34%, "Warp Cycles Per Issued Instruction" 18.10), suggesting there may be further optimization potential by investigating the remaining stall reA_sharedons or fine-tuning tile sizes and thread block configurations, potentially balancing occupancy and per-warp efficiency.

## 2D Block Tiling
- Idea is similar to previous kernel its just now 1 thread compute an 8x8 output tile.
- [kernels/5-shared_mem_2D_blocktiling.cuh](kernels/5-shared_mem_2D_blocktiling.cuh)

```
=== SGEMM Performance Results for Shared Memory 2D Block Tiling SGEMM ===
Matrix dimensions: 4096x4096x4096
Shared Memory 2D Block Tiling SGEMM:  9.44 ms, 14.56 TFLOPS
cuBLA_shared SGEMM: 7.32 ms, 18.78 TFLOPS
Performance ratio (cuBLA_shared/Shared Memory 2D Block Tiling SGEMM): 1.29x
Correctness check: PASSED


    Section: Command line profiler metrics
    -------------------------------------------------------------------- -------------- ------------
    Metric Name                                                             Metric Unit Metric Value
    -------------------------------------------------------------------- -------------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio sector/request         4.42
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio sector/request        32.00
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                               302118267
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                 1168279
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                   738332752
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                    34722711
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                             request     34078720
    l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                             request       524288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                               sector    150738407
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                               sector     16777216
    smsp__sA_shareds_average_data_bytes_per_sector_mem_global_op_ld.pct                     %        90.28
    smsp__sA_shareds_average_data_bytes_per_sector_mem_global_op_st.pct                     %        12.50
    smsp__sA_shareds_average_data_bytes_per_wavefront_mem_shared.pct                        %        24.18
    -------------------------------------------------------------------- -------------- ------------
```

## 2D Block Tiling Coalesced + PADDING + transposed A_shared loads
- [kernels/6-shared_mem_2D_blocktiling_coalesced.cuh](kernels/6-shared_mem_2D_blocktiling_coalesced.cuh)

- It is the version of the original, modified to ensure coalesced memory access
  - Now it loads the memory in `C_shared` cache and then perform store op
- column padding for resolving bank conflicts
- Transposed loading of global A to shared A and modifying the corresponding accesses.

> But overall this kernel is slower (Shared_memory is fixed constraint on hardware) + There are still many bank conflicts

```
=== SGEMM Performance Results for Shared Memory 2D Block Tiling Coalesced SGEMM ===
Matrix dimensions: 4096x4096x4096
Shared Memory 2D Block Tiling Coalesced SGEMM:  22.31 ms, 6.16 TFLOPS
cuBLA_shared SGEMM: 7.27 ms, 18.91 TFLOPS
Performance ratio (cuBLA_shared/Shared Memory 2D Block Tiling Coalesced SGEMM): 3.07x
Correctness check: PASSED

      Section: Command line profiler metrics
    -------------------------------------------------------------------- -------------- ------------
    Metric Name                                                             Metric Unit Metric Value
    -------------------------------------------------------------------- -------------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio sector/request         3.96
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio sector/request            4
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                               268511579
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                               240306243
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                   671691755
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                   307939705
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                             request     67108864
    l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                             request       524288
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                               sector    265996208
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                               sector      2097152
    smsp__sA_shareds_average_data_bytes_per_sector_mem_global_op_ld.pct                     %          100
    smsp__sA_shareds_average_data_bytes_per_sector_mem_global_op_st.pct                     %          100
    smsp__sA_shareds_average_data_bytes_per_wavefront_mem_shared.pct                        %        17.27
    -------------------------------------------------------------------- -------------- ------------
```
## float4 loads + warptiling
- [kernels/7-warptiling_vectorized_loads.cuh](kernels/7-warptiling_vectorized_loads.cuh)
- In this we will consider few things A_shared follows:
**1. Block Tiling (Work for a Thread Block):**
   - The global output matrix C is divided into large tiles, each of size `BLOCK_SIZE_M x BLOCK_SIZE_N`.
   - Each CUDA thread block is responsible for computing one such  `BLOCK_SIZE_M x BLOCK_SIZE_N` tile of C.
   - Global matrix A is seen A_shared `BLOCK_SIZE_M x BLOCK_SIZE_K` tiles, and B A_shared `BLOCK_SIZE_K x BLOCK_SIZE_N` tiles corresponding to the C tile.

**2. Fixed Number of Threads per Block:**
   - The kernel is launched with a fixed number of threads (e.g., 256). This is specified by `__launch_bounds__`.

**3. Warptiling (Managing Work Within a Block):**
   - A C-tile computed by a block ( `BLOCK_SIZE_M x BLOCK_SIZE_N`) can be larger than what the fixed 256 threads might efficiently process in a single "pA_shareds" or what fits optimally into register/cache hierarchies for each thread's immediate work.
   - To manage this, the  `BLOCK_SIZE_M x BLOCK_SIZE_N` block-tile is further divided into smaller "warptiles" of size `WARP_SIZE_M x WARP_SIZE_N`.
     - `WARP_SIZE_M = THREAD_M * GRID_DIM_THREADS_M` (e.g., `THREAD_M * 16`)
     - `WARP_SIZE_N = THREAD_N * GRID_DIM_THREADS_N` (e.g., `THREAD_N * 16`)
     The `16` here comes from arranging the max threads (256) into a logical e.g., `16x16` grid of threads.
   - The thread block iterates `WM_ITER` times in the M-dimension and `WN_ITER` times in the N-dimension to cover all warptiles within its A_sharedsigned  `BLOCK_SIZE_M x BLOCK_SIZE_N` block-tile.
   > Note: warptile does not mean, one warp would fill a warp tile, it is just a unit to make collective work of multiple warps efficient.

**4. Thread Work (Work per Thread within a Warptile Iteration):**
   - Within each warptile iteration, the `MAX_THREADS` (e.g., 256) are logically arranged into a 2D grid (e.g., `16x16`).
   - Each thread in this logical grid is responsible for computing a small `THREAD_M x THREAD_N` sub-tile of the current warptile (e.g., an `8x8` tile if `THREAD_M=8, THREAD_N=8`).
   - `thread_row` and `thread_col` map `threadIdx.x` to a position in this logical thread grid.

**5. Shared Memory Tiling (K-Dimension Slicing):**
   - For each  `BLOCK_SIZE_M x BLOCK_SIZE_N` block-tile of C, the corresponding `BLOCK_SIZE_M x BLOCK_SIZE_K` tile of A and `BLOCK_SIZE_K x BLOCK_SIZE_N` tile of B are processed.
   - The K-dimension is sliced into smaller `BLOCK_SIZE_K`-sized chunks.
   - The outermost loop iterates through these `BLOCK_SIZE_K`-sized slices of K.
   - In each `BLOCK_SIZE_KIdx` iteration:
     - A `BLOCK_SIZE_M x BLOCK_SIZE_K` portion of A is loaded into shared memory (`A_shared`).
     - A `BLOCK_SIZE_K x BLOCK_SIZE_N` portion of B is loaded into shared memory (`B_shared`).

**6. Vectorized Shared Memory Loading with Strides and Offsets:**
   - Threads cooperatively load data from global memory to shared memory (`A_shared` and `B_shared`) using `float4` vectorized reads.
   - **Matrix A is transposed when loaded into `A_shared`. Matrix B is loaded directly into `B_shared`.**
   - `MAX_THREADS` (256) might not be enough to load the entire `BLOCK_SIZE_M x BLOCK_SIZE_K` (for `A_shared`) or `BLOCK_SIZE_K x BLOCK_SIZE_N` (for `B_shared`) shared memory tile in a single `float4` read per thread
> A thought on why 256 threads per block: Internally the grid of `MAX_THREADS` would be broken into warp tiles + ideally it should also be divisible by 32 (warp), so if I pick a dimension of 16 then warptiles dimensions would be (16x16) i.e. 256. if I instead pick 32 then (32x32) i.e. 1024 some GPU have this max limit per block so not exhausting all resources for one block, 256 seem a sweet spot.  

```
=== SGEMM Performance Results for Vectorized Loads SGEMM ===
Matrix dimensions: 4096x4096x4096
Vectorized Loads SGEMM:  8.68 ms, 15.84 TFLOPS
cuBLAS SGEMM: 7.27 ms, 18.92 TFLOPS
Performance ratio (cuBLAS/Vectorized Loads SGEMM): 1.19x
Correctness check: PASSED

Section: Command line profiler metrics
    -------------------------------------------------------------------- -------------- ------------
    Metric Name                                                             Metric Unit Metric Value
    -------------------------------------------------------------------- -------------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio sector/request        15.71
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio sector/request        16.00
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                   15972
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                               101416504
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                   805365052
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                   168525368
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                             request     16908288
    l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                             request       131072
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                               sector    265710404
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                               sector      2097152
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                     %          100
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                     %          100
    smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct                        %        37.93
    -------------------------------------------------------------------- -------------- ------------

```
### Profiling Summary Table

| Metric                          | Value                                  | Notes                                                                                                |
| :------------------------------ | :------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Launch Configuration** |                                        |                                                                                                      |
| Grid Dimensions                 | (8, 32) -> `CEIL_DIV(N, 128), CEIL_DIV(M, 128)` (assuming N=1024, M=4096 as example) | `gridDim(CEIL_DIV(N, BLOCK_SIZE_N), CEIL_DIV(M, BLOCK_SIZE_M))`                                     |
| Block Dimensions                | 256                                    | `(BLOCK_SIZE_M * BLOCK_SIZE_N) / (THREAD_M * THREAD_N)`  => (128\*128)/(8\*8) = 256                   |
| Registers Per Thread            | 41                                     |                                                                                                      |
| Static Shared Memory Per Block  | 8.19 KB                                | `A_shared[128*16*4] + B_shared[16*128*4]` = 2KB\*4 + 2KB\*4 = 8KB + 8KB -> Error in my calc. Profiler is correct: (128\*16 + 16\*128) \* sizeof(float) = (2048 + 2048) \* 4 bytes = 4096 \* 4 = 16384 bytes = 16KB. The profiler shows 8.19KB. Let's re-check kernel. `A_shared[BLOCK_SIZE_M * BLOCK_SIZE_K]` is 128\*16 = 2048 floats. `B_shared[BLOCK_SIZE_K * BLOCK_SIZE_N]` is 16\*128 = 2048 floats. Total = 4096 floats. 4096 \* 4 bytes/float = 16384 bytes = 16 KB. The profiler output says `Static Shared Memory Per Block Kbyte/block 8.19`. This discrepancy is odd. Assuming profiler is correct for now. Let's use the profiler's value. |
| **Overall Performance** |                                        |                                                                                                      |
| Duration                        | 11.54 ms                               |                                                                                                      |
| SM Frequency                    | 1.06 cycle/ns                          |                                                                                                      |
| DRAM Frequency                  | 1.51 cycle/ns                          |                                                                                                      |
| **Throughput** |                                        |                                                                                                      |
| Compute (SM) Throughput         | 82.21 %                                | High utilization, FMA pipeline is the bottleneck.                                                     |
| Memory Throughput               | 81.54 %                                | High overall, but indicates potential contention or inefficient access.                              |
| L1/TEX Cache Throughput         | 81.99 %                                |                                                                                                      |
| L2 Cache Throughput             | 17.66 %                                |                                                                                                      |
| DRAM Throughput                 | 11.89 %                                | Relatively low, suggesting L1/L2 are serving most requests, or stalls elsewhere limit DRAM demand.        |
| **Occupancy** |                                        |                                                                                                      |
| Theoretical Occupancy           | 62.50 %                                | Limited by registers (41 per thread).                                                                |
| Achieved Occupancy              | 59.60 %                                | Close to theoretical, but improvements could increase warp parallelism.                             |
| **Bottlenecks & Issues** |                                        | **Estimated Speedup (from Nsight)** |
| FMA Pipeline Utilization        | 82.7% (Over-utilized)                  | Kernel is compute-bound by FMA operations.                                                          |
| Shared Memory Bank Conflicts    | 4.0-way average conflict (60.18% of wavefronts) | Significant issue. **Est. Speedup: 49.34%** |
| Scheduler Inefficiency          | Issues instruction every 1.9 cycles (vs. 1) | Low eligible warps (2.92 out of 9.54 active). **Est. Local Speedup: 17.79%** |
| Register Pressure               | Limits theoretical occupancy to 62.5%    | Directly impacts scheduler efficiency. **Est. Speedup (tied to occupancy): 17.79%** |
| Uncoalesced Shared Accesses     | 10% of total wavefronts are excessive    | Contributes to shared memory bank conflicts. **Est. Speedup: 10.29%** |
| **Cache Performance** |                                        |                                                                                                      |
| L1/TEX Hit Rate                 | 0.86 %                                 | Very low, most L1 requests miss and go to L2 or global memory.                                       |
| L2 Hit Rate                     | 64.44 %                                | Decent, but L1 misses put pressure here.                                                            |


* **Compute Bound (FMA Pipeline):** The kernel is primarily limited by the FMA (Fused Multiply-Add) pipeline, which is over-utilized at 82.7%. This suggests the arithmetic intensity is high, and the GPU is spending most of its active time on these floating-point operations.
* **High Memory Throughput but Inefficient Access:**
    * Overall memory throughput is high (81.54%), indicating significant data movement.
    * However, **shared memory bank conflicts are a major bottleneck**, with an average 4.0-way conflict. Optimizing shared memory store patterns could yield a substantial speedup (estimated at 49.34%).
    * **Uncoalesced shared accesses** contribute to this, causing around 10% excessive wavefronts with an estimated speedup of 10.29% if resolved.
* **Scheduler and Occupancy Limitations:**
    * The theoretical occupancy is capped at 62.5% due to high register usage (41 registers per thread).
    * Achieved occupancy (59.60%) is close to theoretical, but the schedulers are not fully utilized, issuing an instruction only every 1.9 cycles on average instead of every cycle. This is attributed to a low number of eligible warps. Improving this could lead to a 17.79% local speedup.
* **Cache Utilization:**
    * The L1/TEX Cache Hit Rate is extremely low (0.86%), meaning almost all requests go further down the memory hierarchy.
    * The L2 Hit Rate is moderate (64.44%). Improving data locality for L1 could reduce L2 pressure and overall memory latency.
* **Vectorized Loads:** While the kernel name suggests vectorized loads, the primary performance limiters are currently post-load, in the shared memory access patterns and compute execution. The impact of global memory vectorization is likely positive but overshadowed.

## Hierarchical WarpTiling + Float4 loads + Thread-Group Swizzling
- [kernels/8-hierarchical_warptiling_float4.cuh](kernels/8-hierarchical_warptiling_float4.cuh)
- Following the ideas from prev kernel but going way too much explicit with warps this time
- We broke the blocktile `BLOCK_SIZE_M x BLOCK_SIZE_N` into the warp tiles `WARP_SIZE_M x WARP_SIZE_N`
- Each warp (32 threads) is responsible of computing `WARP_SIZE_M x WARP_SIZE_N` tile of the output matrix C
  - Inc more arithmetic intensity as: each warp further iterates to compute the sub-warp tiles
  - `warp_subtile_N = WARP_SIZE_N / WARP_STEPS_N` and same for M direction
- Within the computation of this warp-subtile
  - Each thread computes `THREAD_M x THREAD_N` elements in M and N dim for each sub warp tile
  - Threads are mapped spatially, i.e. `num_thread_cols_in_warp_subtile = warp_subtile_N / THREAD_N`
  - Total elements computed by threads in 1 sub-step = `warpSize * THREAD_M * THREAD_N` must equal to `warp_subtile_M * warp_subtile_N`
- **Thread-Group Swizzling** is employed to enhance data locality and reduce L2 cache miss by rearranging thread execution order.

```
=== SGEMM Performance Results for Hierarchical Warptiling vectorized SGEMM ===
Matrix dimensions: 4096x4096x4096
Hierarchical Warptiling vectorized SGEMM:  7.81 ms, 17.59 TFLOPS
cuBLAS SGEMM: 7.28 ms, 18.88 TFLOPS
Performance ratio (cuBLAS/Hierarchical Warptiling vectorized SGEMM): 1.07x
Correctness check: PASSED

    -------------------------------------------------------------------- -------------- ------------
    Metric Name                                                             Metric Unit Metric Value
    -------------------------------------------------------------------- -------------- ------------
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio sector/request        15.88
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio sector/request           16
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum                                   13418
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum                                50963171
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                                   402673299
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                                   101294819
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum                             request     12713984
    l1tex__t_requests_pipe_lsu_mem_global_op_st.sum                             request       131072
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum                               sector    201933010
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum                               sector      2097152
    smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct                     %          100
    smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct                     %          100
    smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct                        %           30
    -------------------------------------------------------------------- -------------- ------------
  ```

  Okay, these Nsight Compute profiling metrics provide a wealth of information about your kernel's performance. Let's distill them into a concise summary table, key observations, and a brief conclusion suitable for a README or report.

---

* **Kernel Name:** `sgemm_hierarchical_warptiling_vectorized_kernel<128, 128, 16, 64, 64, 4, 8, 4, 128>`
    * `(BLOCK_SIZE_M, N, K; WARP_SIZE_M, N; WARP_STEPS_N; THREAD_M, N; MAX_THREADS)`
* **Grid Dimensions:** (32, 32, 1) *[for an example M,N of approx. 4096x4096]*
* **Block Dimensions:** (128, 1, 1) *[128 threads per block]*


| Metric                          | Value               | Unit            | Significance                                   |
| :------------------------------ | :------------------ | :-------------- | :--------------------------------------------- |
| **Overall** |                     |                 |                                                |
| Duration                        | 10.67               | ms              | Total execution time.                          |
| **Compute Performance** |                     |                 |                                                |
| SM Frequency                    | 1.06                | cycle/ns        | Achieved SM clock speed.                       |
| Compute (SM) Throughput         | 88.07               | %               | Utilization of SM's compute units (Very Good). |
| FMA Pipeline Utilization        | 92.9                | %               | Fused Multiply-Add units are heavily used.     |
| SM Busy                         | 92.90               | %               | SMs are active most of the time.               |
| Executed IPC Active             | 2.02                | inst/cycle      | Instructions per cycle when SM is active.      |
| **Memory Performance** |                     |                 |                                                |
| Memory Throughput (Overall)     | 38.60               | %               | Overall memory system utilization.             |
| DRAM Throughput                 | 4.02                | %               | Bandwidth utilization to/from off-chip DRAM (Low). |
| L1/TEX Cache Throughput         | 40.72               | %               | L1 cache bandwidth utilization.                |
| L2 Cache Throughput             | 7.82                | %               | L2 cache bandwidth utilization.                |
| L1/TEX Hit Rate                 | 1.62                | %               | Low L1 hit rate for global/local loads.        |
| L2 Hit Rate                     | 75.28               | %               | Good L2 hit rate, caching global data effectively. |
| **Bottlenecks & Issues** |                     |                 |                                                |
| Shared Store Bank Conflicts     | 4.0-way (avg)       |                 | Significant conflicts (60.06% of wavefronts).  |
| Uncoalesced Shared Accesses     | 12% (excessive WF)  |                 | Inefficient reads from shared memory.          |
| **Scheduler & Occupancy** |                     |                 |                                                |
| Registers Per Thread            | 167                 | regs            | High register usage.                           |
| Static Shared Memory Per Block  | 16.38               | KB              | As configured (`(128*16 + 16*128)*4 B`).        |
| Achieved Occupancy              | 16.86               | %               | Low (Theoretical: 18.75%, limited by registers). |
| Avg. Eligible Warps / Scheduler | 1.18                | warps           | Low, indicating frequent warp stalls.          |
| Cycles with No Eligible Warps   | 49.42               | %               | Scheduler often has no ready warps to issue.   |

---


1. The kernel achieves high SM compute throughput (88.07%) with the FMA pipeline being the most utilized (92.9%). This is desirable for an SGEMM kernel, indicating it's performing a lot of mathematical operations.

2. A high L2 Hit Rate (75.28%) and very low DRAM Throughput (4.02%) suggest that global memory reads for matrices A and B are largely being satisfied by the L2 cache after the initial fetches, which is good for data reuse across blocks.

3.  **Major Bottleneck: Shared Memory Accesses:**
    * **Shared Store Bank Conflicts:** The profiler indicates a significant "4.0-way bank conflict" for shared memory stores (likely in `load_share_memory` when writing to `A_shared` due to the transpose pattern and `BLOCK_SIZE_M=128` being a multiple of bank count). Nsight estimates a potential **24.46% speedup** by resolving this.
        * *Action:* Investigate padding `A_shared` or modifying the write pattern to `A_shared` to mitigate these bank conflicts.
    * **Uncoalesced Shared Accesses:** The report points to "uncoalesced shared accesses resulting in ... excessive wavefronts" (likely reads from `A_shared` or `B_shared` in `perform_partial_dot`). Nsight estimates an **11.38% speedup** here.
        * *Action:* Analyze access patterns to `A_shared` and `B_shared` by threads within a warp in `perform_partial_dot`. Ensure reads are coalesced or data is laid out to promote this.

4.  **Low Occupancy due to High Register Usage:**
    * The kernel uses 167 registers per thread, limiting theoretical occupancy to 18.75% (achieving 16.86%). This low occupancy means fewer warps are resident on the SM to hide memory latencies and keep pipelines full.
    * *Action:* Try to reduce register usage. This might involve:
        * Smaller per-thread tile sizes (`THREAD_M`, `THREAD_N`).
        * Re-computing values instead of storing them if feasible.
        * Careful management of loop variables and intermediate calculations.
        * Using `__launch_bounds__` effectively (already present).

5.  **Scheduler Stalls:**
    * Nearly 50% of the time, schedulers have no eligible warps to issue instructions from. This is often a consequence of low occupancy and memory stalls.
    * *Action:* Improving shared memory access patterns and reducing register pressure to increase occupancy should help provide more eligible warps and better hide latencies.

6.  **Load Imbalance (Minor Indication):**
    * "One or more SMSPs have a much higher number of active cycles than the average". While the estimated speedup is moderate (5%), it could indicate slight imbalances in work distribution or data-dependent behavior. This is lower priority than shared memory or occupancy.

---

The `sgemm_hierarchical_warptiling_vectorized_kernel` demonstrates strong compute utilization, effectively leveraging the FMA pipelines for its core matrix multiplication task. The primary performance limiters are **shared memory inefficiencies** (both store bank conflicts and uncoalesced reads) and **low SM occupancy** due to high register pressure.

Addressing the shared memory access patterns (particularly the transpose store to `A_shared` and subsequent reads) and reducing register usage to improve occupancy are the most promising avenues for significant performance gains. The Nsight Compute profiler provides excellent starting points for these optimizations.

## Ping Pong buffering inspired from CUTLASS
- [kernels/9-cutlass-inspired-pingpong-buffering.cuh](kernels/9-cutlass-inspired-pingpong-buffering.cuh)
- Idea is almost same as the previous kernel but now we will use ping pong buffers (like cutlass)
- This kernel implements a double buffering (often called ping-pong buffering) strategy, similar to techniques used in CUTLASS, to effectively hide memory latency by overlapping data loading from global memory with computation on shared memory.

The core logic involves:
1.  **Dual Shared Memory Buffers:** Two sets of shared memory buffers (`A_shared[2]`, `B_shared[2]`) are allocated for tiles of matrices A and B.
2.  **Synchronized Asynchronous Copies:** Data is loaded into these buffers using `cuda::memcpy_async`.
3.  **Cooperative Group Barriers:** An array of two `cuda::barrier<cuda::thread_scope::thread_scope_block>` objects is used to manage the synchronization of these asynchronous copies. These barriers are initialized by a single thread within the block (identified using `cooperative_groups::this_thread_block()`).
4.  **Pipelined Execution:**
      * While computation (e.g., `partial_dot`) is performed on data in the "current" set of shared memory buffers (e.g., `smem_A[curr]`), the kernel initiates asynchronous loads for the *next* data tile into the "next" set of buffers (e.g., `smem_A[next]`).
      * Before computation starts on the current buffer, threads wait on the corresponding `cuda::barrier` (e.g., `barriers[curr].arrive_and_wait()`) to ensure its data is fully loaded.
      * After computation and initiating the next load, the roles of the "current" and "next" buffers (and their associated barriers) are swapped, enabling continuous overlap of memory transfers and calculations throughout the K-loop.
This keeps the GPU's arithmetic units busy, significantly improving kernel throughput.

```
=== SGEMM Performance Results for Ping Pong Buffering SGEMM ===
Matrix dimensions: 4096x4096x4096
Ping Pong Buffering SGEMM:  11.61 ms, 11.84 TFLOPS
cuBLAS SGEMM: 7.32 ms, 18.77 TFLOPS
Performance ratio (cuBLAS/Ping Pong Buffering SGEMM): 1.59x
Correctness check: PASSED
```

Here's a section for your README based on the profiling details:

---

Kernel launch configuration: `gridDim=(2048,1,1)`, `blockDim=(128,1,1)`.
Template parameters: `BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=16, WARP_SIZE_M=32, WARP_SIZE_N=64, WARP_STEPS_N=1, THREAD_M=4, THREAD_N=4, MAX_THREADS=128`.

**Key Performance Metrics:**

| Metric Category             | Metric Name                       | Value          | Unit           |
| :-------------------------- | :-------------------------------- | :------------- | :------------- |
| **Overall Performance** | Duration                          | 15.42          | ms             |
|                             | SM Frequency                      | 1.06           | GHz            |
| **Throughput (% of Peak)** | Compute (SM) Throughput           | 62.70          | %              |
|                             | Memory Throughput (Overall)       | 79.07          | %              |
|                             | L1/TEX Cache Throughput           | 79.32          | %              |
|                             | L2 Cache Throughput               | 14.07          | %              |
|                             | DRAM Throughput                   | 4.52           | %              |
| **Memory System** | Memory Throughput (Actual)        | 87.42          | Gbyte/second   |
|                             | L1/TEX Hit Rate                   | 50.49          | %              |
|                             | L2 Hit Rate                       | 85.10          | %              |
| **Compute & Scheduling** | Executed Ipc Active               | 2.39           | inst/cycle     |
|                             | SM Busy                           | 62.90          | %              |
|                             | Scheduler No Eligible             | 40.27          | % (Stalled)    |
|                             | Avg. Eligible Warps Per Scheduler | 1.21           | warps          |
| **Resource & Occupancy** | Registers Per Thread              | 123            |                |
|                             | Static Shared Memory Per Block    | 24.59          | Kbyte          |
|                             | Theoretical Occupancy             | 25             | % (Reg Limited)|
|                             | Achieved Occupancy                | 23.46          | %              |

**Analysis & Observations:**

* **Memory Bound Characteristics:** The profiler suggests "Memory is more heavily utilized than Compute." This is supported by high L1/TEX Cache Throughput (79.32%) but very low DRAM Throughput (4.52%), indicating heavy reliance on caches and potentially inefficient DRAM access.
* **Compute Performance:** The kernel achieves ~61-63% of the device's FMA (fp32) peak performance. While respectable, this is likely constrained by memory operations.
* **Low Occupancy:** Theoretical occupancy is only 25%, primarily limited by the high number of registers per thread (123). Low occupancy can hinder latency hiding.
* **Scheduler Stalls:** A significant portion of time (40.27% "No Eligible"), schedulers are unable to issue instructions. The low number of eligible warps (1.21 per scheduler on average) despite having more active warps indicates that warps are frequently stalled, likely waiting on memory operations.

**Key Optimization Opportunities Identified by Nsight Compute:**

1.  **Uncoalesced Global Accesses (Est. Speedup: 64.02%):** This is a major bottleneck. 66% of global memory sectors accessed were excessive, indicating poor coalescing. Addressing this should be a top priority.
2.  **Inefficient L1 Local Stores (Est. Speedup: 76.6%):** Only 1.0 of 32 bytes per L1TEX sector for local stores is utilized. This strongly suggests register spilling (due to 123 registers/thread), leading to inefficient local memory accesses. Reducing register usage is crucial.
3.  **Uncoalesced Shared Accesses & Bank Conflicts (Est. Speedup: 31.02% & 13.01%):** Significant shared memory bank conflicts (1.2-way on average) and excessive wavefronts (31%) indicate non-optimal shared memory access patterns within the `partial_dot` function.
4.  **Improving Instruction Issuing (Est. Local Speedup: 20.93%):** Directly related to the stalls. Resolving memory bottlenecks and improving occupancy should help here.

**Brief Conclusion:**

The ping-pong buffering mechanism is implemented to overlap memory transfers and computation. However, the kernel's current performance is significantly limited by several memory access inefficiencies: uncoalesced global memory access, high register pressure leading to probable spills and inefficient local memory use, and shared memory bank conflicts. The low occupancy, constrained by register usage, further limits the ability to hide latencies.
> Interestingly turing off **Swizzling** improves the performance from ~10TFLOPS to ~12TFLOPS (keeping this default for now)

## Distributed SGEMM over multiple GPUs
- [kernels/10-multi-gpu-sgemm.cuh](kernels/10-multi-gpu-sgemm.cuh)
- This is simple, we took the last kernel and just modified the logic to execute on multiple-GPUs in parallel
```
Running multi-GPU kernel...
Using 4 GPU(s).
Average Overall multi-GPU processing time (including H2D, D2H, Host Sum): 66.087 ms
Average Parallel Kernel Execution Time (max across GPUs per run): 3.338 ms
GPU 0 average individual kernel execution time: 3.338 ms
GPU 1 average individual kernel execution time: 2.996 ms
GPU 2 average individual kernel execution time: 2.951 ms
GPU 3 average individual kernel execution time: 2.977 ms
Multi-GPU computation complete.
Running cuBLAS SGEMM...

=== SGEMM Performance Results for Multi-GPU SGEMM ===
Matrix dimensions: 4096x4096x4096
Multi-GPU SGEMM:  3.34 ms, 41.18 TFLOPS
cuBLAS SGEMM: 7.30 ms, 18.83 TFLOPS
Performance ratio (cuBLAS/Multi-GPU SGEMM): 0.46x
Correctness check: PASSED
```

> There is nothing special to profile this as the kernel is same as our pingpong buffering just running in parallel on multiple-GPUs

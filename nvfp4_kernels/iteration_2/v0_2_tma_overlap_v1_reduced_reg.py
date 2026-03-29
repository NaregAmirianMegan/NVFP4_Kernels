#!POPCORN leaderboard nvfp4_gemv
from task import input_t, output_t
import torch

import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

"""
Now that we have TMA working this will allow us to do a number of things
including overlapping compute with memory ops, more freedom with mapping
threads to k-block widths (reduce k-loop length, reduce reg pressure, etc...),
and we could potentially use TMA built-in swizzling to avoid SMEM bank conflicts.

In this version we implement the overlap of compute/memory ops. 
"""

batched_nvfp4_gemv_cuda_source = """
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;
namespace cde = cuda::device::experimental;

#define RESULTS_PER_WARP 4
#define NUM_WARPS_PER_CTA 4
#define N_DIM_PADDING 128
#define K_BLOCK_SIZE 1024 // 32 threads per warp * 32 NVFP4 elems per thread = 1024 elems per warp (warp handles one k_block)


__global__ void batched_nvfp4_gemv_kernel(const __nv_fp4x2_storage_t* __restrict__ b_ref, const __nv_fp8_storage_t* __restrict__ sfb_ref, 
                                          __half* __restrict__ c_ref, int m, int k, int l, const __grid_constant__ CUtensorMap tensor_map_a,
                                          const __grid_constant__ CUtensorMap tensor_map_sfa) {
    const int l_wtid = threadIdx.x % 32;
    const int l_wid = threadIdx.x / 32;

    float results[RESULTS_PER_WARP] = {0.0f};

    const int k_blocks = k / K_BLOCK_SIZE;

    const int batch = (RESULTS_PER_WARP * NUM_WARPS_PER_CTA * blockIdx.x) / m;
    const int cta_row_start = (RESULTS_PER_WARP * NUM_WARPS_PER_CTA * blockIdx.x) % m;

    const int k_elems = k/2;
    const int k_scalars = k/16;

    const int b_base = batch*k_elems*N_DIM_PADDING;
    const int sfb_base = batch*k_scalars*N_DIM_PADDING;

    b_ref += b_base;
    sfb_ref += sfb_base;

    // Declare SMEM Buffers
    __shared__ alignas(16) uint32_t a_smem[2][NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/8];
    __shared__ alignas(16) uint16_t sfa_smem[2][NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/32];
    __shared__ alignas(16) uint32_t b_smem[2][K_BLOCK_SIZE/8];
    __shared__ alignas(16) uint16_t sfb_smem[2][K_BLOCK_SIZE/32];

    const int total_smem_size_tma = (sizeof(a_smem) + sizeof(sfa_smem))/2;

    // Initialize shared memory barrier with the number of threads participating in the barrier.
    // Make initialized barrier visible in async proxy.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar[2];
    if (threadIdx.x == 0) {
        init(&bar[0], blockDim.x);
        init(&bar[1], blockDim.x);
        //cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    const int warp_result_offset = l_wid*RESULTS_PER_WARP;
    const int elems_per_k_block = K_BLOCK_SIZE/2;

    const int cta_row = blockIdx.x*RESULTS_PER_WARP * NUM_WARPS_PER_CTA;

    barrier::arrival_token token[2];

    int comp_stage = 0;
    int load_stage = 0;

    // Prime load to first SMEM buffer before loop
    if (threadIdx.x == 0) {
        // Load a block
        cde::cp_async_bulk_tensor_2d_global_to_shared(&a_smem[load_stage], &tensor_map_a, 0, cta_row, bar[load_stage]);
        // Load sfa block
        cde::cp_async_bulk_tensor_2d_global_to_shared(&sfa_smem[load_stage], &tensor_map_sfa, 0, cta_row, bar[load_stage]);
        // Load b block
        cuda::memcpy_async(
            b_smem[load_stage], 
            reinterpret_cast<uint16_t const*>(b_ref), 
            cuda::aligned_size_t<16>(sizeof(b_smem[0])),
            bar[load_stage]
        );
        // Load sfb block
        cuda::memcpy_async(
            sfb_smem[load_stage], 
            reinterpret_cast<uint16_t const*>(sfb_ref), 
            cuda::aligned_size_t<16>(sizeof(sfb_smem[0])),
            bar[load_stage]
        );
        b_ref += elems_per_k_block;
        sfb_ref += K_BLOCK_SIZE/16;

        token[load_stage] = cuda::device::barrier_arrive_tx(bar[load_stage], 1, total_smem_size_tma);
    } else {
        token[load_stage] = bar[load_stage].arrive();
    }

    for (int kb = 0; kb < k_blocks; kb++) {
        comp_stage = load_stage;
        load_stage = 1 - load_stage;

        if (kb+1 < k_blocks) {
            // thread 0 for each CTA issues all the async TMA transfers
            if (threadIdx.x == 0) {
                // Load a block
                cde::cp_async_bulk_tensor_2d_global_to_shared(&a_smem[load_stage], &tensor_map_a, (kb+1)*(K_BLOCK_SIZE/8), cta_row, bar[load_stage]);
                // Load sfa block
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sfa_smem[load_stage], &tensor_map_sfa, (kb+1)*(K_BLOCK_SIZE/32), cta_row, bar[load_stage]);
                // Load b block
                cuda::memcpy_async(
                    b_smem[load_stage], 
                    reinterpret_cast<uint16_t const*>(b_ref), 
                    cuda::aligned_size_t<16>(sizeof(b_smem[0])),
                    bar[load_stage]
                );
                // Load sfb block
                cuda::memcpy_async(
                    sfb_smem[load_stage], 
                    reinterpret_cast<uint16_t const*>(sfb_ref), 
                    cuda::aligned_size_t<16>(sizeof(sfb_smem[0])),
                    bar[load_stage]
                );
                b_ref += elems_per_k_block;
                sfb_ref += K_BLOCK_SIZE/16;

                token[load_stage] = cuda::device::barrier_arrive_tx(bar[load_stage], 1, total_smem_size_tma);
            } else {
                token[load_stage] = bar[load_stage].arrive();
            }
        }

        // Wait for the data to have arrived for the comp stage
        bar[comp_stage].wait(std::move(token[comp_stage]));

        uint32_t sfb_fp16x2;
        asm volatile(
            "{\\n"
            "cvt.rn.f16x2.e4m3x2 %0, %1;\\n"
            "}\\n"
                : "=r"(sfb_fp16x2)
                : "h"(sfb_smem[comp_stage][l_wtid])
                : "memory"
            );

        #pragma unroll
        for (int result = 0; result < RESULTS_PER_WARP; result++) {

            uint16_t result_fp16;
             asm volatile(
                "{\\n"
                "// declare registers for A / B tensors\\n"
                ".reg .b8 byte0_0, byte0_1, byte0_2, byte0_3;\\n"
                ".reg .b8 byte0_4, byte0_5, byte0_6, byte0_7;\\n"
                ".reg .b8 byte1_0, byte1_1, byte1_2, byte1_3;\\n"
                ".reg .b8 byte1_4, byte1_5, byte1_6, byte1_7;\\n"

                ".reg .b8 byte2_0, byte2_1, byte2_2, byte2_3;\\n"
                ".reg .b8 byte2_4, byte2_5, byte2_6, byte2_7;\\n"
                ".reg .b8 byte3_0, byte3_1, byte3_2, byte3_3;\\n"
                ".reg .b8 byte3_4, byte3_5, byte3_6, byte3_7;\\n"

                "// declare registers for accumulators\\n"
                ".reg .f16x2 accum_0_0, accum_0_1, accum_0_2, accum_0_3;\\n"
                ".reg .f16x2 accum_1_0, accum_1_1, accum_1_2, accum_1_3;\\n"
                ".reg .f16x2 accum_2_0, accum_2_1, accum_2_2, accum_2_3;\\n"
                ".reg .f16x2 accum_3_0, accum_3_1, accum_3_2, accum_3_3;\\n"

                "// declare registers for scaling factors\\n"
                ".reg .f16x2 sfa_f16x2;\\n"
                ".reg .f16x2 sf_f16x2;\\n"
                
                "// declare registers for conversion\\n"
                ".reg .f16x2 cvt_0_0, cvt_0_1, cvt_0_2, cvt_0_3;\\n"
                ".reg .f16x2 cvt_0_4, cvt_0_5, cvt_0_6, cvt_0_7;\\n"
                ".reg .f16x2 cvt_1_0, cvt_1_1, cvt_1_2, cvt_1_3;\\n"
                ".reg .f16x2 cvt_1_4, cvt_1_5, cvt_1_6, cvt_1_7;\\n"

                ".reg .f16x2 cvt_2_0, cvt_2_1, cvt_2_2, cvt_2_3;\\n"
                ".reg .f16x2 cvt_2_4, cvt_2_5, cvt_2_6, cvt_2_7;\\n"
                ".reg .f16x2 cvt_3_0, cvt_3_1, cvt_3_2, cvt_3_3;\\n"
                ".reg .f16x2 cvt_3_4, cvt_3_5, cvt_3_6, cvt_3_7;\\n"

                ".reg .f16 result_f16, lane0, lane1;\\n"
                ".reg .f16x2 mul_f16x2_0, mul_f16x2_1;\\n"

                "// convert scaling factors from fp8 to f16x2\\n"
                "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %1;\\n"
                
                "// clear accumulators\\n"
                "mov.b32 accum_0_0, 0;\\n"
                "mov.b32 accum_0_1, 0;\\n"
                "mov.b32 accum_0_2, 0;\\n"
                "mov.b32 accum_0_3, 0;\\n"
                "mov.b32 accum_1_0, 0;\\n"
                "mov.b32 accum_1_1, 0;\\n"
                "mov.b32 accum_1_2, 0;\\n"
                "mov.b32 accum_1_3, 0;\\n"
                "mov.b32 accum_2_0, 0;\\n"
                "mov.b32 accum_2_1, 0;\\n"
                "mov.b32 accum_2_2, 0;\\n"
                "mov.b32 accum_2_3, 0;\\n"
                "mov.b32 accum_3_0, 0;\\n"
                "mov.b32 accum_3_1, 0;\\n"
                "mov.b32 accum_3_2, 0;\\n"
                "mov.b32 accum_3_3, 0;\\n"
                
                "// multiply, unpacking and permuting scale factors\\n"
                "mul.rn.f16x2 sf_f16x2, sfa_f16x2, %2;\\n"
                "mov.b32 {lane0, lane1}, sf_f16x2;\\n"
                "mov.b32 mul_f16x2_0, {lane0, lane0};\\n"
                "mov.b32 mul_f16x2_1, {lane1, lane1};\\n"

                "// unpacking A and B tensors\\n"
                "mov.b32 {byte0_0, byte0_1, byte0_2, byte0_3}, %3;\\n"
                "mov.b32 {byte0_4, byte0_5, byte0_6, byte0_7}, %4;\\n"
                "mov.b32 {byte1_0, byte1_1, byte1_2, byte1_3}, %5;\\n"
                "mov.b32 {byte1_4, byte1_5, byte1_6, byte1_7}, %6;\\n"

                "mov.b32 {byte2_0, byte2_1, byte2_2, byte2_3}, %7;\\n"
                "mov.b32 {byte2_4, byte2_5, byte2_6, byte2_7}, %8;\\n"
                "mov.b32 {byte3_0, byte3_1, byte3_2, byte3_3}, %9;\\n"
                "mov.b32 {byte3_4, byte3_5, byte3_6, byte3_7}, %10;\\n"

                "// convert A tensor from fp4 to f16x2\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_0_0, byte0_0;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_0_1, byte0_1;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_0_2, byte0_2;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_0_3, byte0_3;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_0_4, byte0_4;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_0_5, byte0_5;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_0_6, byte0_6;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_0_7, byte0_7;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_1_0, byte1_0;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_1_1, byte1_1;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_1_2, byte1_2;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_1_3, byte1_3;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_1_4, byte1_4;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_1_5, byte1_5;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_1_6, byte1_6;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_1_7, byte1_7;\\n"

                "cvt.rn.f16x2.e2m1x2 cvt_2_0, byte2_0;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_2_1, byte2_1;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_2_2, byte2_2;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_2_3, byte2_3;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_2_4, byte2_4;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_2_5, byte2_5;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_2_6, byte2_6;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_2_7, byte2_7;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_3_0, byte3_0;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_3_1, byte3_1;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_3_2, byte3_2;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_3_3, byte3_3;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_3_4, byte3_4;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_3_5, byte3_5;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_3_6, byte3_6;\\n"
                "cvt.rn.f16x2.e2m1x2 cvt_3_7, byte3_7;\\n"

                "// fma for A[0 - 7] and B[0 - 7]\\n"
                "fma.rn.f16x2 accum_0_0, cvt_0_0, cvt_2_0, accum_0_0;\\n"
                "fma.rn.f16x2 accum_0_1, cvt_0_1, cvt_2_1, accum_0_1;\\n"
                "fma.rn.f16x2 accum_0_2, cvt_0_2, cvt_2_2, accum_0_2;\\n"
                "fma.rn.f16x2 accum_0_3, cvt_0_3, cvt_2_3, accum_0_3;\\n"

                "// fma for A[8 - 15] and B[8 - 15]\\n"
                "fma.rn.f16x2 accum_1_0, cvt_0_4, cvt_2_4, accum_1_0;\\n"
                "fma.rn.f16x2 accum_1_1, cvt_0_5, cvt_2_5, accum_1_1;\\n"
                "fma.rn.f16x2 accum_1_2, cvt_0_6, cvt_2_6, accum_1_2;\\n"
                "fma.rn.f16x2 accum_1_3, cvt_0_7, cvt_2_7, accum_1_3;\\n"

                "// fma for A[16 - 23] and B[16 - 23]\\n"
                "fma.rn.f16x2 accum_2_0, cvt_1_0, cvt_3_0, accum_2_0;\\n"
                "fma.rn.f16x2 accum_2_1, cvt_1_1, cvt_3_1, accum_2_1;\\n"
                "fma.rn.f16x2 accum_2_2, cvt_1_2, cvt_3_2, accum_2_2;\\n"
                "fma.rn.f16x2 accum_2_3, cvt_1_3, cvt_3_3, accum_2_3;\\n"

                "// fma for A[24 - 31] and B[24 - 31]\\n"
                "fma.rn.f16x2 accum_3_0, cvt_1_4, cvt_3_4, accum_3_0;\\n"
                "fma.rn.f16x2 accum_3_1, cvt_1_5, cvt_3_5, accum_3_1;\\n"
                "fma.rn.f16x2 accum_3_2, cvt_1_6, cvt_3_6, accum_3_2;\\n"
                "fma.rn.f16x2 accum_3_3, cvt_1_7, cvt_3_7, accum_3_3;\\n"

                "// tree reduction for accumulators\\n"
                "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_1;\\n"
                "add.rn.f16x2 accum_0_2, accum_0_2, accum_0_3;\\n"
                "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_1;\\n"
                "add.rn.f16x2 accum_1_2, accum_1_2, accum_1_3;\\n"
                "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_1;\\n"
                "add.rn.f16x2 accum_2_2, accum_2_2, accum_2_3;\\n"
                "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_1;\\n"
                "add.rn.f16x2 accum_3_2, accum_3_2, accum_3_3;\\n"

                "add.rn.f16x2 accum_0_0, accum_0_0, accum_0_2;\\n"
                "add.rn.f16x2 accum_1_0, accum_1_0, accum_1_2;\\n"
                "add.rn.f16x2 accum_2_0, accum_2_0, accum_2_2;\\n"
                "add.rn.f16x2 accum_3_0, accum_3_0, accum_3_2;\\n"

                "add.rn.f16x2 accum_0_0, accum_0_0, accum_1_0;\\n"
                "add.rn.f16x2 accum_2_0, accum_2_0, accum_3_0;\\n"

                "// apply scaling factors and final reduction\\n"
                "mul.rn.f16x2 accum_0_0, mul_f16x2_0, accum_0_0;\\n"
                "mul.rn.f16x2 accum_2_0, mul_f16x2_1, accum_2_0;\\n"

                "add.rn.f16x2 accum_0_0, accum_0_0, accum_2_0;\\n"
                
                "mov.b32 {lane0, lane1}, accum_0_0;\\n"
                "add.rn.f16 result_f16, lane0, lane1;\\n"

                "mov.b16 %0, result_f16;\\n"

                "}\\n"
                : "=h"(result_fp16)                                     // 0
                : "h"(sfa_smem[comp_stage][warp_result_offset + result][l_wtid]), "r"(sfb_fp16x2),        // 1, 2
                  "r"(a_smem[comp_stage][warp_result_offset + result][l_wtid*4]), "r"(a_smem[comp_stage][warp_result_offset + result][l_wtid*4 + 1]),   // 3, 4
                  "r"(a_smem[comp_stage][warp_result_offset + result][l_wtid*4 + 2]), "r"(a_smem[comp_stage][warp_result_offset + result][l_wtid*4 + 3]),   // 5, 6
                  "r"(b_smem[comp_stage][l_wtid*4]), "r"(b_smem[comp_stage][l_wtid*4 + 1]),   // 7, 8
                  "r"(b_smem[comp_stage][l_wtid*4 + 2]), "r"(b_smem[comp_stage][l_wtid*4 + 3])    // 9, 10
                : "memory"
            );

            float sum = __half2float(__ushort_as_half(result_fp16));

            // add to result total for this thread and this result
            results[result] += sum;
        }

        __syncthreads(); // Is this sync threads necessary, removing it seems to break something
    }

    // Shuffle reduce sub_total for each thread into thread 0 for warp
    for (int result = 0; result < RESULTS_PER_WARP; result++) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            results[result] += __shfl_down_sync(0xffffffff, results[result], offset);
        }
    }

    // Write all results back to GMEM 
    if (l_wtid == 0) {
        __half results_half[RESULTS_PER_WARP];
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            results_half[result] = __float2half(results[result]);
        }
        reinterpret_cast<uint64_t*>(c_ref)[(batch*m + cta_row_start + RESULTS_PER_WARP*l_wid)/4] = reinterpret_cast<uint64_t*>(results_half)[0];
        //reinterpret_cast<uint64_t*>(c_ref)[(batch*m + cta_row_start + RESULTS_PER_WARP*l_wid)/4 + 1] = reinterpret_cast<uint64_t*>(results_half)[1];
        //reinterpret_cast<float4*>(c_ref)[(batch*m + cta_row_start + RESULTS_PER_WARP*l_wid)/8] = reinterpret_cast<float4*>(results_half)[0];
    }
}

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
  assert(driver_status == cudaDriverEntryPointSuccess);

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

torch::Tensor batched_nvfp4_gemv(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, 
                                 torch::Tensor c_ref, int m, int k, int l) { 
    const int results_per_cta = RESULTS_PER_WARP * NUM_WARPS_PER_CTA;
    const int threads = 32 * NUM_WARPS_PER_CTA;
    const int blocks = (m * l) / results_per_cta;

    // assert(m % results_per_cta == 0)

    // Get a function pointer to the cuTensorMapEncodeTiled driver API.
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    const uint64_t GMEM_WIDTH_A = k/8;
    const uint64_t GMEM_HEIGHT_A = m*l;
    const uint32_t SMEM_WIDTH_A = K_BLOCK_SIZE/8;
    const uint32_t SMEM_HEIGHT_A = results_per_cta;

    CUtensorMap tensor_map_a{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank_a = 2;
    uint64_t size_a[rank_a] = {GMEM_WIDTH_A, GMEM_HEIGHT_A};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride_a[rank_a - 1] = {GMEM_WIDTH_A * sizeof(uint32_t)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size_a[rank_a] = {SMEM_WIDTH_A, SMEM_HEIGHT_A};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride_a[rank_a] = {1, 1};

    // Create the tensor descriptor.
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map_a,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        rank_a,                       // cuuint32_t tensorRank,
        reinterpret_cast<void*>(a_ref.data_ptr()),                 // void *globalAddress,
        size_a,                       // const cuuint64_t *globalDim,
        stride_a,                     // const cuuint64_t *globalStrides,
        box_size_a,                   // const cuuint32_t *boxDim,
        elem_stride_a,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    const uint64_t GMEM_WIDTH_SFA = k/32;
    const uint64_t GMEM_HEIGHT_SFA = m*l;
    const uint32_t SMEM_WIDTH_SFA = K_BLOCK_SIZE/32;
    const uint32_t SMEM_HEIGHT_SFA = results_per_cta;

    CUtensorMap tensor_map_sfa{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank_sfa = 2;
    uint64_t size_sfa[rank_sfa] = {GMEM_WIDTH_SFA, GMEM_HEIGHT_SFA};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride_sfa[rank_sfa - 1] = {GMEM_WIDTH_SFA * sizeof(uint16_t)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size_sfa[rank_sfa] = {SMEM_WIDTH_SFA, SMEM_HEIGHT_SFA};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride_sfa[rank_sfa] = {1, 1};

    // Create the tensor descriptor.
    res = cuTensorMapEncodeTiled(
        &tensor_map_sfa,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
        rank_sfa,                       // cuuint32_t tensorRank,
        reinterpret_cast<void*>(sfa_ref.data_ptr()),                 // void *globalAddress,
        size_sfa,                       // const cuuint64_t *globalDim,
        stride_sfa,                     // const cuuint64_t *globalStrides,
        box_size_sfa,                   // const cuuint32_t *boxDim,
        elem_stride_sfa,                // const cuuint32_t *elementStrides,
        // Interleave patterns can be used to accelerate loading of values that
        // are less than 4 bytes long.
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        // L2 Promotion can be used to widen the effect of a cache-policy to a wider
        // set of L2 cache lines.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        // Any element that is outside of bounds will be set to zero by the TMA transfer.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    
    cudaFuncSetAttribute(
        batched_nvfp4_gemv_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared  // Maximum shared memory
    );

    batched_nvfp4_gemv_kernel<<<blocks, threads>>>(reinterpret_cast<__nv_fp4x2_storage_t*>(b_ref.data_ptr()), reinterpret_cast<__nv_fp8_storage_t*>(sfb_ref.data_ptr()), 
                                                    reinterpret_cast<__half*>(c_ref.data_ptr()), m, k, l, tensor_map_a, tensor_map_sfa);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return c_ref;
}
"""

batched_nvfp4_gemv_cpp_source = """
#include <torch/extension.h>

torch::Tensor batched_nvfp4_gemv(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, torch::Tensor c_ref, int m, int k, int l);
"""

batched_nvfp4_gemv_module = load_inline(
    name='batched_nvfp4_gemv',
    cpp_sources=batched_nvfp4_gemv_cpp_source,
    cuda_sources=batched_nvfp4_gemv_cuda_source,
    functions=['batched_nvfp4_gemv'],
    verbose=True,
    extra_cuda_cflags=[
        '-gencode=arch=compute_100a,code=sm_100a',
        '-Xptxas', '--allow-expensive-optimizations=true',
        '--use_fast_math',
        '--maxrregcount=32',
    ],
)

def kernel(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, m, k, l):
    return batched_nvfp4_gemv_module.batched_nvfp4_gemv(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, m, k, l)


def custom_kernel(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref, sfb_ref, _, _, c_ref = data

    return kernel(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, a_ref.shape[0], a_ref.shape[1]*2, a_ref.shape[2])

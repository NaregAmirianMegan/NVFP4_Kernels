#!POPCORN leaderboard nvfp4_gemv
from task import input_t, output_t
import torch

import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

"""
K_BLOCK_SIZE = 2048
"""

batched_nvfp4_gemv_cuda_source = """
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda/barrier>
#include <cuda/ptx>
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;
namespace cde = cuda::device::experimental;

#define NUM_WARPS_PER_CTA 8 // Must be multiple of 2
#define RESULTS_PER_WG 2
#define RESULTS_PER_CTA (NUM_WARPS_PER_CTA/2)*RESULTS_PER_WG
#define N_DIM_PADDING 128
#define K_BLOCK_SIZE 2048 // Still 32 elements per thread, two warps per k-block


__global__ void batched_nvfp4_gemv_kernel(int m, int k, int l, const __grid_constant__ CUtensorMap tensor_map_a,
                                          const __grid_constant__ CUtensorMap tensor_map_sfa, const __grid_constant__ CUtensorMap tensor_map_b,
                                          const __grid_constant__ CUtensorMap tensor_map_sfb, const __grid_constant__ CUtensorMap tensor_map_c) {
    const int l_wtid = threadIdx.x % 32;
    const int l_wid = threadIdx.x / 32;

    float results[RESULTS_PER_WG] = {0.0f};

    const int k_blocks = k / K_BLOCK_SIZE;

    const int batch = (RESULTS_PER_CTA * blockIdx.x) / m;
    const int cta_row_start = (RESULTS_PER_CTA * blockIdx.x) % m;

    // Declare SMEM Buffers
    // When using TMA we need to be careful because all accesses must be 128B aligned, so all strides
    // in the SMEM buffer must also be multiples of 128
    __shared__ alignas(128) uint32_t a_smem[2][NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/8];
    __shared__ alignas(128) uint16_t sfa_smem[2][NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/32];
    __shared__ alignas(128) uint32_t b_smem[2][K_BLOCK_SIZE/8];
    __shared__ alignas(128) uint16_t sfb_smem[2][ 64 * (((K_BLOCK_SIZE/32) + 64 - 1) / 64) ];
   //__shared__ alignas(128) uint16_t sfb_smem_0[K_BLOCK_SIZE/32];
   //__shared__ alignas(128) uint16_t sfb_smem_1[K_BLOCK_SIZE/32];

   __shared__ alignas(128) uint16_t c_smem[NUM_WARPS_PER_CTA*RESULTS_PER_WARP];

    const int total_smem_size = sizeof(a_smem[0]) + sizeof(sfa_smem[0]) + sizeof(b_smem[0]) + sizeof(uint16_t)*(K_BLOCK_SIZE/32);

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
        cde::cp_async_bulk_tensor_2d_global_to_shared(&b_smem[load_stage], &tensor_map_b, 0, batch, bar[load_stage]);
        // Load sfb block
        //cde::cp_async_bulk_tensor_2d_global_to_shared(&sfb_smem_0, &tensor_map_sfb, 0, batch, bar[load_stage]);
        cde::cp_async_bulk_tensor_2d_global_to_shared(&sfb_smem[load_stage], &tensor_map_sfb, 0, batch, bar[load_stage]);

        token[load_stage] = cuda::device::barrier_arrive_tx(bar[load_stage], 1, total_smem_size);
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
                cde::cp_async_bulk_tensor_2d_global_to_shared(&b_smem[load_stage], &tensor_map_b, (kb+1)*(K_BLOCK_SIZE/8), batch, bar[load_stage]);
                // Load sfb block
                cde::cp_async_bulk_tensor_2d_global_to_shared(&sfb_smem[load_stage], &tensor_map_sfb, (kb+1)*(K_BLOCK_SIZE/32), batch, bar[load_stage]);
                //    if (load_stage == 0) {
                //        cde::cp_async_bulk_tensor_2d_global_to_shared(&sfb_smem_0, &tensor_map_sfb, (kb+1)*(K_BLOCK_SIZE/32), batch, bar[load_stage]);
                //    } else {
                //        cde::cp_async_bulk_tensor_2d_global_to_shared(&sfb_smem_1, &tensor_map_sfb, (kb+1)*(K_BLOCK_SIZE/32), batch, bar[load_stage]);
                //    }

                token[load_stage] = cuda::device::barrier_arrive_tx(bar[load_stage], 1, total_smem_size);
            } else {
                token[load_stage] = bar[load_stage].arrive();
            }
        }

        // Wait for the data to have arrived.
        bar[comp_stage].wait(std::move(token[comp_stage]));

        uint32_t sfb_fp16x2;
        //if (comp_stage == 0) {
        asm volatile(
            "{\\n"
            "cvt.rn.f16x2.e4m3x2 %0, %1;\\n"
            "}\\n"
                : "=r"(sfb_fp16x2)
                : "h"(sfb_smem[comp_stage][l_wtid])
                : "memory"
            );
        //} else {

        //    asm volatile(
        //        "{\\n"
        //        "cvt.rn.f16x2.e4m3x2 %0, %1;\\n"
        //        "}\\n"
        //            : "=r"(sfb_fp16x2)
        //            : "h"(sfb_smem_1[l_wtid])
        //            : "memory"
        //        );
        //}

        //#pragma unroll
        for (int result = 0; result < RESULTS_PER_WARP; result++) {

            uint16_t result_fp16;
            asm volatile(
                "{\\n"

                ".reg .b8 a_byte0, a_byte1, a_byte2, a_byte3;\\n"
                ".reg .b8 b_byte0, b_byte1, b_byte2, b_byte3;\\n"

                ".reg .f16x2 accum_0, accum_1, accum_2, accum_3;\\n"

                ".reg .f16x2 sfa_f16x2;\\n"
                ".reg .f16x2 sf_f16x2;\\n"

                ".reg .f16x2 a_cvt_0, a_cvt_1, a_cvt_2, a_cvt_3;\\n"
                ".reg .f16x2 b_cvt_0, b_cvt_1, b_cvt_2, b_cvt_3;\\n"

                ".reg .f16 result_f16, lane0, lane1;\\n"
                ".reg .f16x2 mul_f16x2_0, mul_f16x2_1;\\n"

                "// convert scaling factors from fp8 to f16x2\\n"
                "cvt.rn.f16x2.e4m3x2 sfa_f16x2, %1;\\n"

                "mov.b32 accum_0, 0;\\n"
                "mov.b32 accum_1, 0;\\n"
                "mov.b32 accum_2, 0;\\n"
                "mov.b32 accum_3, 0;\\n"

                "mul.rn.f16x2 sf_f16x2, sfa_f16x2, %2;\\n"
                "mov.b32 {lane0, lane1}, sf_f16x2;\\n"
                "mov.b32 mul_f16x2_0, {lane0, lane0};\\n"
                "mov.b32 mul_f16x2_1, {lane1, lane1};\\n"

                "mov.b32 {a_byte0, a_byte1, a_byte2, a_byte3}, %3;\\n"
                "mov.b32 {b_byte0, b_byte1, b_byte2, b_byte3}, %7;\\n"

                "cvt.rn.f16x2.e2m1x2 a_cvt_0, a_byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_1, a_byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_2, a_byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_3, a_byte3;\\n"

                "cvt.rn.f16x2.e2m1x2 b_cvt_0, b_byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_1, b_byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_2, b_byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_3, b_byte3;\\n"

                "fma.rn.f16x2 accum_0, a_cvt_0, b_cvt_0, accum_0;\\n"
                "fma.rn.f16x2 accum_1, a_cvt_1, b_cvt_1, accum_1;\\n"
                "fma.rn.f16x2 accum_0, a_cvt_2, b_cvt_2, accum_0;\\n"
                "fma.rn.f16x2 accum_1, a_cvt_3, b_cvt_3, accum_1;\\n"

                "mov.b32 {a_byte0, a_byte1, a_byte2, a_byte3}, %4;\\n"
                "mov.b32 {b_byte0, b_byte1, b_byte2, b_byte3}, %8;\\n"

                "cvt.rn.f16x2.e2m1x2 a_cvt_0, a_byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_1, a_byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_2, a_byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_3, a_byte3;\\n"

                "cvt.rn.f16x2.e2m1x2 b_cvt_0, b_byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_1, b_byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_2, b_byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_3, b_byte3;\\n"

                "fma.rn.f16x2 accum_0, a_cvt_0, b_cvt_0, accum_0;\\n"
                "fma.rn.f16x2 accum_1, a_cvt_1, b_cvt_1, accum_1;\\n"
                "fma.rn.f16x2 accum_0, a_cvt_2, b_cvt_2, accum_0;\\n"
                "fma.rn.f16x2 accum_1, a_cvt_3, b_cvt_3, accum_1;\\n"


                "mov.b32 {a_byte0, a_byte1, a_byte2, a_byte3}, %5;\\n"
                "mov.b32 {b_byte0, b_byte1, b_byte2, b_byte3}, %9;\\n"

                "cvt.rn.f16x2.e2m1x2 a_cvt_0, a_byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_1, a_byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_2, a_byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_3, a_byte3;\\n"

                "cvt.rn.f16x2.e2m1x2 b_cvt_0, b_byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_1, b_byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_2, b_byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_3, b_byte3;\\n"

                "fma.rn.f16x2 accum_2, a_cvt_0, b_cvt_0, accum_2;\\n"
                "fma.rn.f16x2 accum_3, a_cvt_1, b_cvt_1, accum_3;\\n"
                "fma.rn.f16x2 accum_2, a_cvt_2, b_cvt_2, accum_2;\\n"
                "fma.rn.f16x2 accum_3, a_cvt_3, b_cvt_3, accum_3;\\n"

                "mov.b32 {a_byte0, a_byte1, a_byte2, a_byte3}, %6;\\n"
                "mov.b32 {b_byte0, b_byte1, b_byte2, b_byte3}, %10;\\n"

                "cvt.rn.f16x2.e2m1x2 a_cvt_0, a_byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_1, a_byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_2, a_byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 a_cvt_3, a_byte3;\\n"

                "cvt.rn.f16x2.e2m1x2 b_cvt_0, b_byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_1, b_byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_2, b_byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 b_cvt_3, b_byte3;\\n"

                "fma.rn.f16x2 accum_2, a_cvt_0, b_cvt_0, accum_2;\\n"
                "fma.rn.f16x2 accum_3, a_cvt_1, b_cvt_1, accum_3;\\n"
                "fma.rn.f16x2 accum_2, a_cvt_2, b_cvt_2, accum_2;\\n"
                "fma.rn.f16x2 accum_3, a_cvt_3, b_cvt_3, accum_3;\\n"


                "add.rn.f16x2 accum_0, accum_0, accum_1;\\n"
                "add.rn.f16x2 accum_2, accum_2, accum_3;\\n"

                "// apply scaling factors and final reduction\\n"
                "mul.rn.f16x2 accum_0, mul_f16x2_0, accum_0;\\n"
                "mul.rn.f16x2 accum_2, mul_f16x2_1, accum_2;\\n"

                "add.rn.f16x2 accum_0, accum_0, accum_2;\\n"
                
                "mov.b32 {lane0, lane1}, accum_0;\\n"
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

        __syncthreads();
    }

    // Shuffle reduce sub_total for each thread into thread 0 for warp
    for (int result = 0; result < RESULTS_PER_WARP; result++) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            results[result] += __shfl_down_sync(0xffffffff, results[result], offset);
        }
    }

    // Write results to SMEM buffer
    if (l_wtid == 0) {
        __half results_half[RESULTS_PER_WARP];
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            results_half[result] = __float2half(results[result]);
        }
        //c_smem[RESULTS_PER_WARP*l_wid] = reinterpret_cast<uint16_t*>(results_half)[0];
        reinterpret_cast<uint32_t*>(c_smem)[(RESULTS_PER_WARP*l_wid)/2] = reinterpret_cast<uint32_t*>(results_half)[0];
        //reinterpret_cast<uint64_t*>(c_smem)[(RESULTS_PER_WARP*l_wid)/4] = reinterpret_cast<uint64_t*>(results_half)[0];
        //reinterpret_cast<float4*>(c_smem)[(RESULTS_PER_WARP*l_wid)/8] = reinterpret_cast<float4*>(results_half)[0];
    }

    // Ensure SMEM buffer writes are seen by TMA engine
    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    // Send data to GMEM, ensure all data is is read by TMA from SMEM, then destory the barriers used for this CTA
    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_c, cta_row_start, batch, &c_smem);
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        cde::cp_async_bulk_wait_group_read<0>();
    }

    // Destroy barrier. This invalidates the memory region of the barrier. If
    // further computations were to take place in the kernel, this allows the
    //  memory location of the shared memory barrier to be reused.
    // if (threadIdx.x == 0) {
    //     (&bar)->~barrier();
    // }


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

    const uint64_t GMEM_WIDTH_B = k/8;
    const uint64_t GMEM_HEIGHT_B = l;
    const uint32_t SMEM_WIDTH_B = K_BLOCK_SIZE/8;
    const uint32_t SMEM_HEIGHT_B = 1;

    CUtensorMap tensor_map_b{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank_b = 2;
    uint64_t size_b[rank_b] = {GMEM_WIDTH_B, GMEM_HEIGHT_B};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride_b[rank_b - 1] = {GMEM_WIDTH_B * sizeof(uint32_t) * N_DIM_PADDING};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size_b[rank_b] = {SMEM_WIDTH_B, SMEM_HEIGHT_B};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride_b[rank_b] = {1, 1};

    // Create the tensor descriptor.
    res = cuTensorMapEncodeTiled(
        &tensor_map_b,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT32,
        rank_b,                       // cuuint32_t tensorRank,
        reinterpret_cast<void*>(b_ref.data_ptr()),                 // void *globalAddress,
        size_b,                       // const cuuint64_t *globalDim,
        stride_b,                     // const cuuint64_t *globalStrides,
        box_size_b,                   // const cuuint32_t *boxDim,
        elem_stride_b,                // const cuuint32_t *elementStrides,
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

    const uint64_t GMEM_WIDTH_SFB = k/32;
    const uint64_t GMEM_HEIGHT_SFB = l;
    const uint32_t SMEM_WIDTH_SFB = K_BLOCK_SIZE/32;
    const uint32_t SMEM_HEIGHT_SFB = 1;

    CUtensorMap tensor_map_sfb{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank_sfb = 2;
    uint64_t size_sfb[rank_sfb] = {GMEM_WIDTH_SFB, GMEM_HEIGHT_SFB};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride_sfb[rank_sfb - 1] = {GMEM_WIDTH_SFB * sizeof(uint16_t) * N_DIM_PADDING};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size_sfb[rank_sfb] = {SMEM_WIDTH_SFB, SMEM_HEIGHT_SFB};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride_sfb[rank_sfb] = {1, 1};

    // Create the tensor descriptor.
    res = cuTensorMapEncodeTiled(
        &tensor_map_sfb,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
        rank_sfb,                       // cuuint32_t tensorRank,
        reinterpret_cast<void*>(sfb_ref.data_ptr()),                 // void *globalAddress,
        size_sfb,                       // const cuuint64_t *globalDim,
        stride_sfb,                     // const cuuint64_t *globalStrides,
        box_size_sfb,                   // const cuuint32_t *boxDim,
        elem_stride_sfb,                // const cuuint32_t *elementStrides,
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

    const uint64_t GMEM_WIDTH_C = m;
    const uint64_t GMEM_HEIGHT_C = l;
    const uint32_t SMEM_WIDTH_C = NUM_WARPS_PER_CTA*RESULTS_PER_WARP;
    const uint32_t SMEM_HEIGHT_C = 1;

    CUtensorMap tensor_map_c{};
    // rank is the number of dimensions of the array.
    constexpr uint32_t rank_c = 2;
    uint64_t size_c[rank_c] = {GMEM_WIDTH_C, GMEM_HEIGHT_C};
    // The stride is the number of bytes to traverse from the first element of one row to the next.
    // It must be a multiple of 16.
    uint64_t stride_c[rank_c - 1] = {GMEM_WIDTH_C * sizeof(uint16_t)};
    // The box_size is the size of the shared memory buffer that is used as the
    // destination of a TMA transfer.
    uint32_t box_size_c[rank_c] = {SMEM_WIDTH_C, SMEM_HEIGHT_C};
    // The distance between elements in units of sizeof(element). A stride of 2
    // can be used to load only the real component of a complex-valued tensor, for instance.
    uint32_t elem_stride_c[rank_c] = {1, 1};

    // Create the tensor descriptor.
    res = cuTensorMapEncodeTiled(
        &tensor_map_c,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT16,
        rank_c,                       // cuuint32_t tensorRank,
        reinterpret_cast<void*>(c_ref.data_ptr()),                 // void *globalAddress,
        size_c,                       // const cuuint64_t *globalDim,
        stride_c,                     // const cuuint64_t *globalStrides,
        box_size_c,                   // const cuuint32_t *boxDim,
        elem_stride_c,                // const cuuint32_t *elementStrides,
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
    
    batched_nvfp4_gemv_kernel<<<blocks, threads>>>(m, k, l, tensor_map_a, tensor_map_sfa, tensor_map_b, tensor_map_sfb, tensor_map_c);

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

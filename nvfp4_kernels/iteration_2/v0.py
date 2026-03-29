#!POPCORN leaderboard nvfp4_gemv
from task import input_t, output_t
import torch

import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

"""
Trying to use cvt PTX instructions, because it seems like CUDA intrinsics 

"""

batched_nvfp4_gemv_cuda_source = """
#define RESULTS_PER_WARP 4
#define NUM_WARPS_PER_CTA 8
#define N_DIM_PADDING 128
#define K_BLOCK_SIZE 1024 // 32 threads per warp * 32 NVFP4 elems per thread = 1024 elems per warp (warp handles one k_block)


__global__ void batched_nvfp4_gemv_kernel(const __nv_fp4x2_storage_t* __restrict__ a_ref, const __nv_fp4x2_storage_t* __restrict__ b_ref, 
                                          const __nv_fp8_storage_t* __restrict__ sfa_ref, const __nv_fp8_storage_t* __restrict__ sfb_ref, 
                                          __half* __restrict__ c_ref, int m, int k, int l) {
    const int l_wtid = threadIdx.x % 32;
    const int l_wid = threadIdx.x / 32;

    // ISSUE: Only thread 0 of every warp needs this array, compiler should optimize out register usage right?
    float results[RESULTS_PER_WARP] = {0.0f};
    uint32_t b_vals_cvt[16];

    const int k_blocks = k / K_BLOCK_SIZE;

    const int batch = (RESULTS_PER_WARP * NUM_WARPS_PER_CTA * blockIdx.x) / m;
    const int cta_row_start = (RESULTS_PER_WARP * NUM_WARPS_PER_CTA * blockIdx.x) % m;

    const int k_elems = k/2;
    const int k_scalars = k/16;

    const int a_base = (batch*m + cta_row_start)*k_elems;
    const int b_base = batch*k_elems*N_DIM_PADDING;
    const int sfa_base = (batch*m + cta_row_start)*k_scalars;
    const int sfb_base = batch*k_scalars*N_DIM_PADDING;

    a_ref += a_base;
    b_ref += b_base;
    sfa_ref += sfa_base;
    sfb_ref += sfb_base;

    // Declare SMEM Buffers
    __shared__ uint32_t a_smem[NUM_WARPS_PER_CTA*RESULTS_PER_WARP*K_BLOCK_SIZE/8];
    __shared__ uint16_t sfa_smem[NUM_WARPS_PER_CTA*RESULTS_PER_WARP*K_BLOCK_SIZE/32];
    __shared__ uint32_t b_smem[K_BLOCK_SIZE/8];
    __shared__ uint16_t sfb_smem[K_BLOCK_SIZE/32];

    const int warp_result_offset = l_wid*RESULTS_PER_WARP;
    const int elems_per_k_block = K_BLOCK_SIZE/2;

    for (int kb = 0; kb < k_blocks; kb++) {
        // First all threads in the CTA cooperatively load the necessary chunks of data GMEM->SMEM for this k-block
        // Load b_ref and sfb_ref for k-block, only first warp should issue these since all warps in a CTA share this block
        if (l_wid == 0) {
            reinterpret_cast<float4*>(b_smem)[l_wtid] = reinterpret_cast<float4 const*>(b_ref)[l_wtid];
            sfb_smem[l_wtid] = reinterpret_cast<uint16_t const*>(sfb_ref)[l_wtid];
        }
        b_ref += elems_per_k_block;
        sfb_ref += K_BLOCK_SIZE/16;

        // Load block from a_ref (size: NUM_WARPS_PER_CTA*RESULTS_PER_WARP rows each with K_BLOCK_SIZE NVPF4 elements)
        for (int result = warp_result_offset; result < warp_result_offset + RESULTS_PER_WARP; result++) {
            reinterpret_cast<float4*>(a_smem)[result*(K_BLOCK_SIZE/32) + l_wtid] = reinterpret_cast<float4 const*>(a_ref)[result*(k/32) + l_wtid];
            sfa_smem[result*(K_BLOCK_SIZE/32) + l_wtid] = reinterpret_cast<uint16_t const*>(sfa_ref)[result*(k/32) + l_wtid];
        }
        a_ref += elems_per_k_block;
        sfa_ref += K_BLOCK_SIZE/16;

        // Ensure all data for this k-block made it from GMEM to SMEM
        __syncthreads();

        // Load b fragment and sfb scalars from SMEM -> RF and convert to fp16 for this thread

        // Each loop converts 8 NVFP4 values into 8 FP16 values via nvfp4x2 -> fp16x2 data type conversions
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            asm volatile(
                "{\\n"
                ".reg .b8 byte0, byte1, byte2, byte3;\\n"
                "mov.b32 {byte0, byte1, byte2, byte3}, %4;\\n"
                "cvt.rn.f16x2.e2m1x2 %0, byte0;\\n"
                "cvt.rn.f16x2.e2m1x2 %1, byte1;\\n"
                "cvt.rn.f16x2.e2m1x2 %2, byte2;\\n"
                "cvt.rn.f16x2.e2m1x2 %3, byte3;\\n"
                "}\\n"
                    : "=r"(b_vals_cvt[i*4]), "=r"(b_vals_cvt[i*4+1]), "=r"(b_vals_cvt[i*4+2]), "=r"(b_vals_cvt[i*4+3])
                    : "r"(b_smem[l_wtid*4 + i])
                    : "memory"
                );
        }

        uint32_t sfb_fp16x2;
        asm volatile(
            "{\\n"
            "cvt.rn.f16x2.e4m3x2 %0, %1;\\n"
            "}\\n"
                : "=r"(sfb_fp16x2)
                : "h"(sfb_smem[l_wtid])
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

                "// fma for A[0 - 7] and B[0 - 7]\\n"
                "fma.rn.f16x2 accum_0_0, cvt_0_0, %7, accum_0_0;\\n"
                "fma.rn.f16x2 accum_0_1, cvt_0_1, %8, accum_0_1;\\n"
                "fma.rn.f16x2 accum_0_2, cvt_0_2, %9, accum_0_2;\\n"
                "fma.rn.f16x2 accum_0_3, cvt_0_3, %10, accum_0_3;\\n"

                "// fma for A[8 - 15] and B[8 - 15]\\n"
                "fma.rn.f16x2 accum_1_0, cvt_0_4, %11, accum_1_0;\\n"
                "fma.rn.f16x2 accum_1_1, cvt_0_5, %12, accum_1_1;\\n"
                "fma.rn.f16x2 accum_1_2, cvt_0_6, %13, accum_1_2;\\n"
                "fma.rn.f16x2 accum_1_3, cvt_0_7, %14, accum_1_3;\\n"

                "// fma for A[16 - 23] and B[16 - 23]\\n"
                "fma.rn.f16x2 accum_2_0, cvt_1_0, %15, accum_2_0;\\n"
                "fma.rn.f16x2 accum_2_1, cvt_1_1, %16, accum_2_1;\\n"
                "fma.rn.f16x2 accum_2_2, cvt_1_2, %17, accum_2_2;\\n"
                "fma.rn.f16x2 accum_2_3, cvt_1_3, %18, accum_2_3;\\n"

                "// fma for A[24 - 31] and B[24 - 31]\\n"
                "fma.rn.f16x2 accum_3_0, cvt_1_4, %19, accum_3_0;\\n"
                "fma.rn.f16x2 accum_3_1, cvt_1_5, %20, accum_3_1;\\n"
                "fma.rn.f16x2 accum_3_2, cvt_1_6, %21, accum_3_2;\\n"
                "fma.rn.f16x2 accum_3_3, cvt_1_7, %22, accum_3_3;\\n"

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
                : "h"(sfa_smem[(warp_result_offset + result)*(K_BLOCK_SIZE/32) + l_wtid]), "r"(sfb_fp16x2),        // 1, 2
                  "r"(a_smem[(warp_result_offset + result)*(K_BLOCK_SIZE/8) + l_wtid*4]), "r"(a_smem[(warp_result_offset + result)*(K_BLOCK_SIZE/8) + l_wtid*4 + 1]),   // 3, 4
                  "r"(a_smem[(warp_result_offset + result)*(K_BLOCK_SIZE/8) + l_wtid*4 + 2]), "r"(a_smem[(warp_result_offset + result)*(K_BLOCK_SIZE/8) + l_wtid*4 + 3]),   // 5, 6
                  "r"(b_vals_cvt[0]), "r"(b_vals_cvt[1]),   // 7, 8
                  "r"(b_vals_cvt[2]), "r"(b_vals_cvt[3]),    // 9, 10
                  "r"(b_vals_cvt[4]), "r"(b_vals_cvt[5]),   // 11, 12
                  "r"(b_vals_cvt[6]), "r"(b_vals_cvt[7]),    // 13, 14
                  "r"(b_vals_cvt[8]), "r"(b_vals_cvt[9]),   // 15, 16
                  "r"(b_vals_cvt[10]), "r"(b_vals_cvt[11]),    // 17, 18
                  "r"(b_vals_cvt[12]), "r"(b_vals_cvt[13]),   // 19, 20
                  "r"(b_vals_cvt[14]), "r"(b_vals_cvt[15])    // 21, 22
                : "memory"
            );

            float sum = __half2float(__ushort_as_half(result_fp16));

            // Shuffle reduce sub_total for each thread into thread 0 for warp
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            // Thread 0 of warp adds to result total
            if (l_wtid == 0) {
                results[result] += sum;
            }
        }
    }

    // Write all results back to GMEM 
    if (l_wtid == 0) {
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            c_ref[batch*m + cta_row_start + RESULTS_PER_WARP*l_wid + result] = __float2half(results[result]);
        }
    }
}

torch::Tensor batched_nvfp4_gemv(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, 
                                 torch::Tensor c_ref, int m, int k, int l) { 
    const int results_per_cta = RESULTS_PER_WARP * NUM_WARPS_PER_CTA;
    const int threads = 32 * NUM_WARPS_PER_CTA;
    const int blocks = (m * l) / results_per_cta;

    // assert(m % results_per_cta == 0)
    
    batched_nvfp4_gemv_kernel<<<blocks, threads>>>(reinterpret_cast<__nv_fp4x2_storage_t*>(a_ref.data_ptr()), 
                                reinterpret_cast<__nv_fp4x2_storage_t*>(b_ref.data_ptr()), reinterpret_cast<__nv_fp8_storage_t*>(sfa_ref.data_ptr()), 
                                reinterpret_cast<__nv_fp8_storage_t*>(sfb_ref.data_ptr()), reinterpret_cast<__half*>(c_ref.data_ptr()), m, k, l);

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
    ],
)

def kernel(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, m, k, l):
    return batched_nvfp4_gemv_module.batched_nvfp4_gemv(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, m, k, l)


def custom_kernel(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref, sfb_ref, _, _, c_ref = data

    return kernel(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, a_ref.shape[0], a_ref.shape[1]*2, a_ref.shape[2])

#!POPCORN leaderboard nvfp4_gemv
from task import input_t, output_t
import torch

import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

"""
Optimized version that reduces LOP3/PRMT/IMAD instructions by:
1. Pre-converting A matrix to FP16 in shared memory (eliminates conversions in compute loop)
2. Pre-scaling B matrix with sfb (eliminates scale multiplication in compute loop)
3. Pre-computing combined scales as FP16 (reduces FP32 operations)
"""

batched_nvfp4_gemv_cuda_source = """
#define RESULTS_PER_WARP 4
#define NUM_WARPS_PER_CTA 8
#define N_DIM_PADDING 128
#define K_BLOCK_SIZE 512


__global__ void batched_nvfp4_gemv_kernel(const __nv_fp4x2_storage_t* __restrict__ a_ref, const __nv_fp4x2_storage_t* __restrict__ b_ref, 
                                          const __nv_fp8_storage_t* __restrict__ sfa_ref, const __nv_fp8_storage_t* __restrict__ sfb_ref, 
                                          __half* __restrict__ c_ref, int m, int k, int l) {
    const int l_wtid = threadIdx.x % 32;
    const int l_wid = threadIdx.x / 32;

    float results[RESULTS_PER_WARP] = {0.0f};

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

    // Declare SMEM Buffers - now storing pre-converted FP16 values
    __shared__ uint64_t a_smem_raw[NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/16];
    __shared__ __half2 a_smem_fp16[NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/2];
    __shared__ __half sfa_smem[NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/16];
    
    __shared__ uint64_t b_smem_raw[K_BLOCK_SIZE/16];
    __shared__ __half2 b_smem_fp16[K_BLOCK_SIZE/2];
    __shared__ __half sfb_smem[K_BLOCK_SIZE/16];

    const int warp_result_offset = l_wid*RESULTS_PER_WARP;
    const int elems_per_k_block = K_BLOCK_SIZE/2;

    for (int kb = 0; kb < k_blocks; kb++) {
        // Load B and convert to FP16 with pre-scaling
        if (l_wid == 0) {
            // Load raw FP4 data
            b_smem_raw[l_wtid] = reinterpret_cast<uint64_t const*>(b_ref)[l_wtid];
            
            // Load and convert scale factor
            __half_raw sfb_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfb_ref[l_wtid], __NV_E4M3);
            __half sfb_fp16 = *reinterpret_cast<__half*>(&sfb_raw_fp16);
            sfb_smem[l_wtid] = sfb_fp16;
            __half2 sfb_2x = __half2half2(sfb_fp16);
            
            // Convert FP4 to FP16 and pre-scale with sfb
            __nv_fp4x2_storage_t const* b_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&b_smem_raw[l_wtid]);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                __half2 b_fp16 = *reinterpret_cast<__half2*>(&__nv_cvt_fp4x2_to_halfraw2(b_vals[i], __NV_E2M1));
                b_smem_fp16[l_wtid*8 + i] = __hmul2(b_fp16, sfb_2x);
            }
        }
        b_ref += elems_per_k_block;
        sfb_ref += 32;

        // Load A and convert to FP16
        for (int result = warp_result_offset; result < warp_result_offset + RESULTS_PER_WARP; result++) {
            // Load raw FP4 data and scale factor
            a_smem_raw[result][l_wtid] = reinterpret_cast<uint64_t const*>(a_ref)[result*k_scalars + l_wtid];
            __half_raw sfa_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfa_ref[result*k_scalars + l_wtid], __NV_E4M3);
            __half sfa_fp16 = *reinterpret_cast<__half*>(&sfa_raw_fp16);
            sfa_smem[result][l_wtid] = sfa_fp16;
            
            // Convert FP4 to FP16
            __nv_fp4x2_storage_t const* a_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&a_smem_raw[result][l_wtid]);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                a_smem_fp16[result][l_wtid*8 + i] = *reinterpret_cast<__half2*>(&__nv_cvt_fp4x2_to_halfraw2(a_vals[i], __NV_E2M1));
            }
        }
        a_ref += elems_per_k_block;
        sfa_ref += 32;

        __syncthreads();

        // Compute: Now just FP16 FMAs, no conversions!
        #pragma unroll
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            // Load pre-computed scale
            __half sfa_fp16 = sfa_smem[warp_result_offset + result][l_wtid];
            __half2 sfa_2x = __half2half2(sfa_fp16);
            
            // Accumulate dot product in FP16
            __half2 sub_total = __float2half2_rn(0.0f);
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                // Pure FP16 FMA - no conversions needed!
                __half2 a_vals = a_smem_fp16[warp_result_offset + result][l_wtid*8 + i];
                __half2 b_vals = b_smem_fp16[l_wtid*8 + i];
                sub_total = __hfma2(a_vals, b_vals, sub_total);
            }
            
            // Apply scale factor to accumulated result
            sub_total = __hmul2(sub_total, sfa_2x);
            
            // Reduce to scalar
            __half sub_total_lo = __low2half(sub_total);
            __half sub_total_hi = __high2half(sub_total);
            float sum = __half2float(__hadd(sub_total_lo, sub_total_hi));
            
            // Warp reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            if (l_wtid == 0) {
                results[result] += sum;
            }
        }

        __syncthreads();
    }

    // Write results
    if (l_wtid == 0) {
        #pragma unroll
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


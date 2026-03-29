#!POPCORN leaderboard nvfp4_gemv
from task import input_t, output_t
import torch

import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

"""
For this kernel, a warp iterates over the k-dim to compute outputs

Each output will be computed by NUM_CTAS CTAs, each CTA takes a segment of the output (along m-dim)
Within each CTA there will be threads/32 warps, each warp will compute (m/NUM_CTAS)/warps outputs

For each output row assigned to this warp: # putting this loop here allows RF locality of b and sfb b/c all rows in the same k-block use the same b-block/sfb-block
    For each k-block:
        warp loads and computes scaled dot-product

Each thread within each warp computes 16 elements of the dot product and stores the result in RF buffer 

2 NVFP4 elements = 1B 
uint64_t = 8B = 16 NVFP4 elements

Each warp computes 32*16=512 NVFP4 elements per k-block

For each warp
    For each output row
        -> loads same 16 elems or 8B from b across all warps
        -> loads 32 e4m3 float or 32B from sfb across all warps
        For each scaled dot-product computed by a warp it:
            -> loads 16*32 contiguous NVFP4 elements from a (8*32 = 256B of contiguous loads)
            -> loads 32 e4m3 floats from sfa (32B of contiguous loads)

"""

batched_nvfp4_gemv_cuda_source = """
#define RESULTS_PER_WARP 4
#define NUM_WARPS_PER_CTA 8

__global__ void batched_nvfp4_gemv_kernel(const __nv_fp4x2_storage_t* __restrict__ a_ref, const __nv_fp4x2_storage_t* __restrict__ b_ref, 
                                          const __nv_fp8_storage_t* __restrict__ sfa_ref, const __nv_fp8_storage_t* __restrict__ sfb_ref, 
                                          __half* __restrict__ c_ref, int m, int k, int l) {
    int l_tid = threadIdx.x;
    int l_wtid = l_tid % 32;
    int l_wid = l_tid / 32;

    int k_block_size = 512; // 32 threads per warp * 16 NVFP4 elems per thread = 512 elems per warp (warp handles one k_block)
    int k_blocks = k / k_block_size;

    float results[RESULTS_PER_WARP] = {0.0f};

    int batch = (RESULTS_PER_WARP * NUM_WARPS_PER_CTA * blockIdx.x) / m;
    int cta_row_start = (RESULTS_PER_WARP * NUM_WARPS_PER_CTA * blockIdx.x) % m;

    for (int kb = 0; kb < k_blocks; kb++) {
        // Check if this thread's k-range is valid
        // bool valid = (kb * k_block_size + l_wtid * 16) < k;

        // Load 16 values from b into thread RF using 1 uint64_t (8B) load
        uint64_t b_vals_packed = reinterpret_cast<uint64_t const*>(b_ref)[(batch*(k/2)*128 + kb*(k_block_size/2) + l_wtid*8)/8];
        __nv_fp4x2_storage_t const* b_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&b_vals_packed);

        // Load sfb value used for this thread
        __nv_fp8_storage_t sfb = sfb_ref[batch*(k/16)*128 + kb*(32) + l_wtid];
        __half_raw sfb_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
        __half sfb_fp16 = *reinterpret_cast<__half*>(&sfb_raw_fp16);
        float sfb_fp32 = __half2float(sfb_fp16);

        #pragma unroll
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            // Load this threads 16 values from a
            uint64_t a_vals_packed = reinterpret_cast<uint64_t const*>(a_ref)[((batch*m + cta_row_start)*(k/2) + result*(NUM_WARPS_PER_CTA*k/2) + l_wid*(k/2) + kb*(k_block_size/2) + l_wtid*8)/8];
            __nv_fp4x2_storage_t const* a_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&a_vals_packed);

            // Load sfa this value used for this thread
            __nv_fp8_storage_t sfa = sfa_ref[(batch*m + cta_row_start)*(k/16) + result*(NUM_WARPS_PER_CTA*k/16) + l_wid*(k/16) + kb*32 + l_wtid];
            __half_raw sfa_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfa, __NV_E4M3);
            __half sfa_fp16 = *reinterpret_cast<__half*>(&sfa_raw_fp16);
            float sfa_fp32 = __half2float(sfa_fp16);

            // Compute scaling factor
            float block_scale = sfa_fp32*sfb_fp32;

            // Reset thread local dot-product total reg
            __half2 sub_total = __float2half2_rn(0.0f);

            // Compute dot-product in half-precision
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                __half2_raw b_2x_raw_fp16 = __nv_cvt_fp4x2_to_halfraw2(b_vals[i], __NV_E2M1);
                __half2 b_2x_fp16 = *reinterpret_cast<__half2*>(&b_2x_raw_fp16);
                __half2_raw a_2x_raw_fp16 = __nv_cvt_fp4x2_to_halfraw2(a_vals[i], __NV_E2M1);
                __half2 a_2x_fp16 = *reinterpret_cast<__half2*>(&a_2x_raw_fp16);
                sub_total = __hfma2(a_2x_fp16, b_2x_fp16, sub_total);
            }
            // Sum each fp16 float in sub_total
            __half sub_total_lo = __low2half(sub_total);
            __half sub_total_hi = __high2half(sub_total);
            float sub_total_lo_fp32 = __half2float(sub_total_lo);
            float sub_total_hi_fp32 = __half2float(sub_total_hi);
            float sum = (sub_total_lo_fp32 + sub_total_hi_fp32)*block_scale;

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
            c_ref[batch*m + cta_row_start + result*NUM_WARPS_PER_CTA + l_wid] = __float2half(results[result]);
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
)

def kernel(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, m, k, l):
    return batched_nvfp4_gemv_module.batched_nvfp4_gemv(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, m, k, l)


def custom_kernel(data: input_t) -> output_t:
	a_ref, b_ref, sfa_ref, sfb_ref, _, _, c_ref = data

	return kernel(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, a_ref.shape[0], a_ref.shape[1]*2, a_ref.shape[2])

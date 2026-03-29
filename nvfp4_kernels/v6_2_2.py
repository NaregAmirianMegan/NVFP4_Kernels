#!POPCORN leaderboard nvfp4_gemv
from task import input_t, output_t
import torch

import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

"""
Moves sfb conversion to data load section

"""

batched_nvfp4_gemv_cuda_source = """
#define RESULTS_PER_WARP 4
#define NUM_WARPS_PER_CTA 8
#define N_DIM_PADDING 128
#define K_BLOCK_SIZE 512 // 32 threads per warp * 16 NVFP4 elems per thread = 512 elems per warp (warp handles one k_block)


__global__ void batched_nvfp4_gemv_kernel(const __nv_fp4x2_storage_t* __restrict__ a_ref, const __nv_fp4x2_storage_t* __restrict__ b_ref, 
                                          const __nv_fp8_storage_t* __restrict__ sfa_ref, const __nv_fp8_storage_t* __restrict__ sfb_ref, 
                                          __half* __restrict__ c_ref, int m, int k, int l) {
    const int l_wtid = threadIdx.x % 32;
    const int l_wid = threadIdx.x / 32;

    // ISSUE: Only thread 0 of every warp needs this array, compiler should optimize out register usage right?
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

    // Declare SMEM Buffers
    __shared__ uint64_t a_smem[NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/16];
    __shared__ __nv_fp8_storage_t sfa_smem[NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/16];
    __shared__ uint64_t b_smem[K_BLOCK_SIZE/16];
    __shared__ __half2 b_smem_fp16[K_BLOCK_SIZE/2];
    //__shared__ __nv_fp8_storage_t sfb_smem[K_BLOCK_SIZE/16];
    __shared__ float sfb_smem_fp32[K_BLOCK_SIZE/16];

    const int warp_result_offset = l_wid*RESULTS_PER_WARP;
    const int elems_per_k_block = K_BLOCK_SIZE/2;

    for (int kb = 0; kb < k_blocks; kb++) {
        // First all threads in the CTA cooperatively load the necessary chunks of data GMEM->SMEM for this k-block
        // Load b_ref and sfb_ref for k-block, only first warp should issue these since all warps in a CTA share this block
        if (l_wid == 0) {
            b_smem[l_wtid] = reinterpret_cast<uint64_t const*>(b_ref)[l_wtid];
            __half_raw sfb_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfb_ref[l_wtid], __NV_E4M3);
            __nv_fp4x2_storage_t const* b_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&b_smem[l_wtid]);
            __half sfb_fp16 = *reinterpret_cast<__half*>(&sfb_raw_fp16);
            sfb_smem_fp32[l_wtid] = __half2float(sfb_fp16);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                b_smem_fp16[l_wtid*8 + i] = *reinterpret_cast<__half2*>(&__nv_cvt_fp4x2_to_halfraw2(b_vals[i], __NV_E2M1));
            }
        }
        b_ref += elems_per_k_block;
        sfb_ref += 32;

        // Load block from a_ref (size: NUM_WARPS_PER_CTA*RESULTS_PER_WARP rows each with K_BLOCK_SIZE NVPF4 elements)
        for (int result = warp_result_offset; result < warp_result_offset + RESULTS_PER_WARP; result++) {
            a_smem[result][l_wtid] = reinterpret_cast<uint64_t const*>(a_ref)[result*k_scalars + l_wtid];
            sfa_smem[result][l_wtid] = sfa_ref[result*k_scalars + l_wtid];
        }
        a_ref += elems_per_k_block;
        sfa_ref += 32;

        // Ensure all data for this k-block made it from GMEM to SMEM
        __syncthreads();

        // Load sfb value from SMEM -> RF for this thread, convert to float
        //__nv_fp8_storage_t sfb = sfb_smem[l_wtid];
        //__half_raw sfb_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
        //__half sfb_fp16 = *reinterpret_cast<__half*>(&sfb_raw_fp16);
        //float sfb_fp32 = __half2float(sfb_fp16);
        float sfb_fp32 = sfb_smem_fp32[l_wtid];

        #pragma unroll
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            // Load this threads 16 values from a_smem
            __nv_fp4x2_storage_t const* a_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&a_smem[warp_result_offset + result][l_wtid]);

            // Load sfa this value used for this thread
            __nv_fp8_storage_t sfa = sfa_smem[warp_result_offset + result][l_wtid];
            __half_raw sfa_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfa, __NV_E4M3);
            __half sfa_fp16 = *reinterpret_cast<__half*>(&sfa_raw_fp16);
            float sfa_fp32 = __half2float(sfa_fp16);

            // Compute scaling factor
            float block_scale = sfa_fp32*sfb_fp32;

            // Reset thread local dot-product total reg
            __half2 sub_total = __float2half2_rn(0.0f);

            // Compute dot-product in half-precision
            #pragma unroll
            for (int i = 0; i < 8; i+=2) {
                __half2_raw a0_2x_raw_fp16 = __nv_cvt_fp4x2_to_halfraw2(a_vals[i], __NV_E2M1);
                __half2_raw a1_2x_raw_fp16 = __nv_cvt_fp4x2_to_halfraw2(a_vals[i+1], __NV_E2M1);
                __half2 a0_2x_fp16 = *reinterpret_cast<__half2*>(&a0_2x_raw_fp16);
                sub_total = __hfma2(a0_2x_fp16, b_smem_fp16[l_wtid*8 + i], sub_total);
                __half2 a1_2x_fp16 = *reinterpret_cast<__half2*>(&a1_2x_raw_fp16);
                sub_total = __hfma2(a1_2x_fp16, b_smem_fp16[l_wtid*8 + i + 1], sub_total);
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

        __syncthreads();
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

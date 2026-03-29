#!POPCORN leaderboard nvfp4_gemv
from task import input_t, output_t
import torch

import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

batched_nvfp4_gemv_cuda_source = """
__global__ void batched_nvfp4_gemv_kernel(const __nv_fp4x2_storage_t* __restrict__ a_ref, const __nv_fp4x2_storage_t* __restrict__ b_ref, 
                                          const __nv_fp8_storage_t* __restrict__ sfa_ref, const __nv_fp8_storage_t* __restrict__ sfb_ref, 
                                          __half* __restrict__ c_ref, int m, int k, int l, int BK) {
    // store partial results in dynamic shared memory
    extern __shared__ float partial_sums[];

    // since each CTA handles one row of output, k_block = threadIdx.x; 
    int l_tid = threadIdx.x;
    int num_k_blocks = k/BK;

    partial_sums[l_tid] = 0.0f;

    int batch = blockIdx.x / m;
    int row = blockIdx.x % m;
    int c_base = batch*m + row;

    if (l_tid < num_k_blocks) {
        int a_base = batch*m*(k/2) + row*(k/2) + l_tid*(BK/2);
        int b_base = batch*(k/2)*128 + l_tid*(BK/2);
        int sfa_base = batch*m*(k/16) + row*(k/16) + l_tid*(BK/16);
        int sfb_base = batch*(k/16)*128 + l_tid*(BK/16);

        for (int k_idx = 0; k_idx < BK; k_idx += 16) {
            // unpack corresponding sfa, sfb into float32 and multiply them to get block_scale
            __nv_fp8_storage_t sfa = sfa_ref[sfa_base+(k_idx/16)];
            __nv_fp8_storage_t sfb = sfb_ref[sfb_base+(k_idx/16)];
            __half_raw sfa_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfa, __NV_E4M3);
            __half_raw sfb_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
            __half sfa_fp16 = *reinterpret_cast<__half*>(&sfa_raw_fp16);
            __half sfb_fp16 = *reinterpret_cast<__half*>(&sfb_raw_fp16);
            float sfa_fp32 = __half2float(sfa_fp16);
            float sfb_fp32 = __half2float(sfb_fp16);
            float block_scale = sfa_fp32*sfb_fp32;
            
            __half2 sub_total = __float2half2_rn(0.0f);

            // For each element in the 16, unpack a and b into float32s and fma them into sub_total
            for (int j = 0; j < 8; j++) {
    	        __half2_raw a_2x_raw_fp16 = __nv_cvt_fp4x2_to_halfraw2(a_ref[a_base+(k_idx/2)+j], __NV_E2M1);
    	        __half2 a_2x_fp16 = *reinterpret_cast<__half2*>(&a_2x_raw_fp16);
    	        __half2_raw b_2x_raw_fp16 = __nv_cvt_fp4x2_to_halfraw2(b_ref[b_base+(k_idx/2)+j], __NV_E2M1);
    	        __half2 b_2x_fp16 = *reinterpret_cast<__half2*>(&b_2x_raw_fp16);
    	        sub_total = __hfma2(a_2x_fp16, b_2x_fp16, sub_total);
    	    }
    	    // sum each fp16 float in sub_total
    		__half sub_total_lo = __low2half(sub_total);
    		__half sub_total_hi = __high2half(sub_total);
    		float sub_total_lo_fp32 = __half2float(sub_total_lo);
    		float sub_total_hi_fp32 = __half2float(sub_total_hi);
    		float sum = sub_total_lo_fp32 + sub_total_hi_fp32;

    		// multiply that sum by block_scale and add that result to total
            partial_sums[l_tid] += sum*block_scale;
        }
    }

    __syncthreads();

    // Once all partial sums are computed, reduce along the k-dimension
    // Reduction in shared memory (generic for any block size)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (l_tid < stride) {
            partial_sums[l_tid] += partial_sums[l_tid + stride];
        }
        __syncthreads();
    }

    // Once all reductions are complete, convert and store totals back to GMEM
    if (l_tid == 0) {
        __half total_fp16 = __float2half(partial_sums[l_tid]);
        c_ref[c_base] = total_fp16;
    }
}

torch::Tensor batched_nvfp4_gemv(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, 
                                 torch::Tensor c_ref, int m, int k, int l) { 
    // BK is number of elements a single thread FMAs along k-dim
    const int BK = 32;
    const int num_k_blocks = k / BK;
    int threads = 32;
    // threads per CTA = power of 2 greater than num_k_blocks
    while (threads < num_k_blocks) threads <<= 1;

    /* 
    Assertions (enable for debugging):
    assert(num_k_blocks > 1);
    */

    // Launch 1 CTA per row of output
    const int blocks = m*l;
    
    // In v1 each thread computes it's own block-scaled dot-product
    // There are M*L dot-products
    batched_nvfp4_gemv_kernel<<<blocks, threads, threads*sizeof(float)>>>(reinterpret_cast<__nv_fp4x2_storage_t*>(a_ref.data_ptr()), 
                                reinterpret_cast<__nv_fp4x2_storage_t*>(b_ref.data_ptr()), reinterpret_cast<__nv_fp8_storage_t*>(sfa_ref.data_ptr()), 
                                reinterpret_cast<__nv_fp8_storage_t*>(sfb_ref.data_ptr()), reinterpret_cast<__half*>(c_ref.data_ptr()), m, k, l, BK);

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

#!POPCORN leaderboard nvfp4_gemv
from task import input_t, output_t
import torch

from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t

batched_nvfp4_gemv_cuda_source = """
#include <cuda/pipeline>

#define RESULTS_PER_WARP 4
#define NUM_WARPS_PER_CTA 16
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

    // Double-buffered SMEM
    __shared__ uint64_t a_smem[2][NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/16];
    __shared__ __nv_fp8_storage_t sfa_smem[2][NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/16];
    __shared__ uint64_t b_smem[2][K_BLOCK_SIZE/16];
    __shared__ __nv_fp8_storage_t sfb_smem[2][K_BLOCK_SIZE/16];

    // Each thread gets its own pipeline
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Prologue: Launch async load for first k-block
    int write_buf = 0;
    
    if (l_wid == 0) {
        cuda::memcpy_async(&b_smem[write_buf][l_wtid], 
                          &reinterpret_cast<uint64_t const*>(b_ref)[(b_base)/8 + l_wtid], 
                          sizeof(uint64_t), pipe);
        cuda::memcpy_async(&sfb_smem[write_buf][l_wtid], 
                          &sfb_ref[sfb_base + l_wtid], 
                          sizeof(__nv_fp8_storage_t), pipe);
    }
    for (int result = 0; result < RESULTS_PER_WARP; result++) {
        cuda::memcpy_async(&a_smem[write_buf][l_wid*RESULTS_PER_WARP + result][l_wtid],
                          &reinterpret_cast<uint64_t const*>(a_ref)[(a_base + l_wid*RESULTS_PER_WARP*k_elems + result*k_elems)/8 + l_wtid],
                          sizeof(uint64_t), pipe);
        cuda::memcpy_async(&sfa_smem[write_buf][l_wid*RESULTS_PER_WARP + result][l_wtid],
                          &sfa_ref[sfa_base + l_wid*RESULTS_PER_WARP*k_scalars + result*k_scalars + l_wtid],
                          sizeof(__nv_fp8_storage_t), pipe);
    }
    pipe.producer_commit();

    // Main pipelined loop
    for (int kb = 0; kb < k_blocks; kb++) {
        int read_buf = write_buf;
        write_buf = 1 - write_buf;

        // Launch async load for next k-block
        if (kb + 1 < k_blocks) {
            if (l_wid == 0) {
                cuda::memcpy_async(&b_smem[write_buf][l_wtid],
                                  &reinterpret_cast<uint64_t const*>(b_ref)[(b_base + (kb+1)*(K_BLOCK_SIZE/2))/8 + l_wtid],
                                  sizeof(uint64_t), pipe);
                cuda::memcpy_async(&sfb_smem[write_buf][l_wtid],
                                  &sfb_ref[sfb_base + (kb+1)*32 + l_wtid],
                                  sizeof(__nv_fp8_storage_t), pipe);
            }
            for (int result = 0; result < RESULTS_PER_WARP; result++) {
                cuda::memcpy_async(&a_smem[write_buf][l_wid*RESULTS_PER_WARP + result][l_wtid],
                                  &reinterpret_cast<uint64_t const*>(a_ref)[(a_base + l_wid*RESULTS_PER_WARP*k_elems + result*k_elems + (kb+1)*(K_BLOCK_SIZE/2))/8 + l_wtid],
                                  sizeof(uint64_t), pipe);
                cuda::memcpy_async(&sfa_smem[write_buf][l_wid*RESULTS_PER_WARP + result][l_wtid],
                                  &sfa_ref[sfa_base + l_wid*RESULTS_PER_WARP*k_scalars + result*k_scalars + (kb+1)*32 + l_wtid],
                                  sizeof(__nv_fp8_storage_t), pipe);
            }
            pipe.producer_commit();
        }

        // Wait for this thread's async copies to complete, then sync all threads
        pipe.consumer_wait();
        __syncthreads();

        // Compute on current k-block from read_buf
        __nv_fp4x2_storage_t const* b_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&b_smem[read_buf][l_wtid]);

        __nv_fp8_storage_t sfb = sfb_smem[read_buf][l_wtid];
        __half_raw sfb_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
        __half sfb_fp16 = *reinterpret_cast<__half*>(&sfb_raw_fp16);
        float sfb_fp32 = __half2float(sfb_fp16);

        #pragma unroll
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            __nv_fp4x2_storage_t const* a_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&a_smem[read_buf][l_wid*RESULTS_PER_WARP + result][l_wtid]);

            __nv_fp8_storage_t sfa = sfa_smem[read_buf][l_wid*RESULTS_PER_WARP + result][l_wtid];
            __half_raw sfa_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfa, __NV_E4M3);
            __half sfa_fp16 = *reinterpret_cast<__half*>(&sfa_raw_fp16);
            float sfa_fp32 = __half2float(sfa_fp16);

            float block_scale = sfa_fp32*sfb_fp32;

            __half2 sub_total = __float2half2_rn(0.0f);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                __half2_raw b_2x_raw_fp16 = __nv_cvt_fp4x2_to_halfraw2(b_vals[i], __NV_E2M1);
                __half2 b_2x_fp16 = *reinterpret_cast<__half2*>(&b_2x_raw_fp16);
                __half2_raw a_2x_raw_fp16 = __nv_cvt_fp4x2_to_halfraw2(a_vals[i], __NV_E2M1);
                __half2 a_2x_fp16 = *reinterpret_cast<__half2*>(&a_2x_raw_fp16);
                sub_total = __hfma2(a_2x_fp16, b_2x_fp16, sub_total);
            }

            __half sub_total_lo = __low2half(sub_total);
            __half sub_total_hi = __high2half(sub_total);
            float sub_total_lo_fp32 = __half2float(sub_total_lo);
            float sub_total_hi_fp32 = __half2float(sub_total_hi);
            float sum = (sub_total_lo_fp32 + sub_total_hi_fp32)*block_scale;

            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            if (l_wtid == 0) {
                results[result] += sum;
            }
        }

        // Sync before next iteration to ensure all threads done reading before we overwrite
        __syncthreads();
    }

    // Write results back to GMEM 
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

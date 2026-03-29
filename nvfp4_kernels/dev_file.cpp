#define RESULTS_PER_WARP 4
#define NUM_WARPS_PER_CTA 8
#define N_DIM_PADDING 128
#define K_BLOCK_SIZE 512 // 32 threads per warp * 16 NVFP4 elems per thread = 512 elems per warp (warp handles one k_block)
#define NUM_STAGES


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

    // Declare SMEM Buffers
    __shared__ uint64_t a_smem[NUM_STAGES][NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/16];
    __shared__ __nv_fp8_storage_t sfa_smem[NUM_STAGES][NUM_WARPS_PER_CTA*RESULTS_PER_WARP][K_BLOCK_SIZE/16];
    __shared__ uint64_t b_smem[NUM_STAGES][K_BLOCK_SIZE/16];
    __shared__ __nv_fp8_storage_t sfb_smem[NUM_STAGES][K_BLOCK_SIZE/16];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    for (int stage = 0; stage < NUM_STAGES; stage++) {
        pipe.producer_acquire();

        cuda::memcpy_async(&smem[stage][threadIdx.x], &src[idx], sizeof(int), pipe);
        
        pipe.producer_commit();
    }

    for (int kb = 0; kb < k_blocks; kb++) {
        // First all threads in the CTA cooperatively load the necessary chunks of data GMEM->SMEM for this k-block
        // Load b_ref and sfb_ref for k-block, only first warp should issue these since all warps in a CTA share this block
        if (l_wid == 0) {
            b_smem[l_wtid] = reinterpret_cast<uint64_t const*>(b_ref)[(b_base + kb*(K_BLOCK_SIZE/2))/8 + l_wtid];
            sfb_smem[l_wtid] = sfb_ref[sfb_base + kb*(32) + l_wtid];
        }

        // Load block from a_ref (size: NUM_WARPS_PER_CTA*RESULTS_PER_WARP rows each with K_BLOCK_SIZE NVPF4 elements)
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            a_smem[l_wid*RESULTS_PER_WARP + result][l_wtid] = reinterpret_cast<uint64_t const*>(a_ref)[(a_base + l_wid*RESULTS_PER_WARP*k_elems + result*k_elems + kb*(K_BLOCK_SIZE/2))/8 + l_wtid];
            sfa_smem[l_wid*RESULTS_PER_WARP + result][l_wtid] = sfa_ref[sfa_base + l_wid*RESULTS_PER_WARP*k_scalars + result*k_scalars + kb*32 + l_wtid];
        }

        // Ensure all data for this k-block made it from GMEM to SMEM
        __syncthreads();

        // Load b fragment from SMEM -> RF for this thread
        __nv_fp4x2_storage_t const* b_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&b_smem[l_wtid]);

        // Load sfb value from SMEM -> RF for this thread, convert to float
        __nv_fp8_storage_t sfb = sfb_smem[l_wtid];
        __half_raw sfb_raw_fp16 = __nv_cvt_fp8_to_halfraw(sfb, __NV_E4M3);
        __half sfb_fp16 = *reinterpret_cast<__half*>(&sfb_raw_fp16);
        float sfb_fp32 = __half2float(sfb_fp16);

        #pragma unroll
        for (int result = 0; result < RESULTS_PER_WARP; result++) {
            // Load this threads 16 values from a_smem
            __nv_fp4x2_storage_t const* a_vals = reinterpret_cast<__nv_fp4x2_storage_t const*>(&a_smem[l_wid*RESULTS_PER_WARP + result][l_wtid]);

            // Load sfa this value used for this thread
            __nv_fp8_storage_t sfa = sfa_smem[l_wid*RESULTS_PER_WARP + result][l_wtid];
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
            c_ref[batch*m + cta_row_start + RESULTS_PER_WARP*l_wid + result] = __float2half(results[result]);
        }
    }
}




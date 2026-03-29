#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda_fp16.h>

/*
    Implement support for N=64 via TMEM indexing offsets
*/

/*
    Notes:
    - TD stands for Tile Dimension

    Assumptions:
    1) Problem shape is divisible by CTA and SMEM Tile shapes (no tail cases)
    2) We assume TD_SMEM_M/N == TD_MMA_M/N, since TMEM can only store one MMA tile worth of results at a time
*/

enum class CacheHintSm100 : uint64_t {
    EVICT_NORMAL = 0x1000000000000000,
    EVICT_FIRST  = 0x12F0000000000000,
    EVICT_LAST   = 0x14F0000000000000,
};

__device__ void inline tcgen05_commit(const int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
        :
        : "r"(mbar_addr) 
        : "memory"
    );
}

__device__ void mbar_wait(const int mbar_addr, const int phase) {
    uint32_t ticks = 0x989680;  // expiration date for try wait to re-try, from CUTLASS
    asm volatile(
        "{\\n"
        ".reg .pred P1;\\n"
        "LAB_WAIT:\\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\\n" // Acquire semantics assumed here
        "@P1 bra.uni DONE;\\n" // Add .uni here because there won't be warp divergence
        "bra.uni     LAB_WAIT;\\n"
        "DONE:\\n"
        "}"
        :
        : "r"(mbar_addr), "r"(phase), "r"(ticks)
    );
}

__device__ inline void mbar_arrive_expect(const int mbar_addr, const int size) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :
        : "r"(mbar_addr), "r"(size) 
        : "memory"
    );
}

__device__ inline void mbar_init(const int mbar_addr, const int count) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;"
        :
        : "r"(mbar_addr), "r"(count)
    );
}

__device__ uint32_t inline elect_one_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\\n"
    ".reg .pred %%px;\\n"
    "     elect.sync _|%%px, %1;\\n"
    "@%%px mov.s32 %0, 1;\\n"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

__device__ void inline tcgen05_1dtma_g2s_sf(int dst, const void *src, int size, int mbar_addr, CacheHintSm100 cache_policy) {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
        :
        : "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

template<int CTA_GROUP>
__device__ void inline tcgen05_3dtma_g2s_ab(int dst_smem, const void *tmap_ptr, int mn_off, int k_off_coremat, int mbar_addr, CacheHintSm100 cache_policy) {
    asm volatile (
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%0.L2::cache_hint [%1], [%2, {%3, %4, %5}], [%6], %7;"
        :
        : "n"(CTA_GROUP), "r"(dst_smem), "l"(tmap_ptr), "r"(0), "r"(mn_off), "r"(k_off_coremat), "r"(mbar_addr), "l"(cache_policy)
    );
}

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
    assert(driver_status == cudaDriverEntryPointSuccess);
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}

template<int MN_SMEM_TD, int K_SMEM_TD>
void init_tma_map(PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled, CUtensorMap* tmap, void* ptr, uint64_t mn_dim_gmem, uint64_t k_dim_gmem) {
    constexpr uint32_t rank = 3;
    uint64_t dim_gmem[rank] = {32, mn_dim_gmem, k_dim_gmem/32};
    uint64_t stride_gmem[rank - 1] = {k_dim_gmem/2, 16};
    uint32_t dim_smem[rank] = {32, MN_SMEM_TD, K_SMEM_TD/32};
    uint32_t elem_stride[rank] = {1, 1, 1};

    // Create the tensor descriptor.
    auto res = cuTensorMapEncodeTiled(
        tmap,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        rank,                       // cuuint32_t tensorRank,
        ptr,                 // void *globalAddress,
        dim_gmem,                       // const cuuint64_t *globalDim,
        stride_gmem,                     // const cuuint64_t *globalStrides,
        dim_smem,                   // const cuuint32_t *boxDim,
        elem_stride,                // const cuuint32_t *elementStrides,
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
    // ISSUE: Insert error check here on res
}


/*
    For each warp:
        LANES: How many rows of TMEM are loaded
        WIDTH: How many bits in each row
        REPT: WIDTH repeated REPT times

    Number of 32b regs needed is WIDTH * REPT / 32
*/
template<int LANES, int WIDTH, int REPT>
__device__ void inline tcgen05_ld(float* regs, int tmem_addr);
// |
// V
// Specializations
template<>
__device__ void inline tcgen05_ld<32, 32, 16>(float* regs, int tmem_addr) {
    asm volatile (
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 { %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                                                   " %8,  %9,  %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(regs[0]), "=f"(regs[1]), "=f"(regs[2]), "=f"(regs[3]),
          "=f"(regs[4]), "=f"(regs[5]), "=f"(regs[6]), "=f"(regs[7]),
          "=f"(regs[8]), "=f"(regs[9]), "=f"(regs[10]), "=f"(regs[11]),
          "=f"(regs[12]), "=f"(regs[13]), "=f"(regs[14]), "=f"(regs[15])
        : "r"(tmem_addr)
    );
}
template<>
__device__ void inline tcgen05_ld<16, 256, 16>(float* regs, int tmem_addr) {
    asm volatile (
        "tcgen05.ld.sync.aligned.16x256b.x16.b32 {  %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                                                   "  %8,  %9,  %10, %11, %12, %13, %14, %15, "
                                                   "  %16, %17, %18, %19, %20, %21, %22, %23, "
                                                   "  %24, %25, %26, %27, %28, %29, %30, %31, "
                                                   "  %32, %33, %34, %35, %36, %37, %38, %39, "
                                                   "  %40, %41, %42, %43, %44, %45, %46, %47, "
                                                   "  %48, %49, %50, %51, %52, %53, %54, %55, "
                                                   "  %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
        : "=f"(regs[0]),  "=f"(regs[1]),  "=f"(regs[2]),  "=f"(regs[3]),
          "=f"(regs[4]),  "=f"(regs[5]),  "=f"(regs[6]),  "=f"(regs[7]),
          "=f"(regs[8]),  "=f"(regs[9]),  "=f"(regs[10]), "=f"(regs[11]),
          "=f"(regs[12]), "=f"(regs[13]), "=f"(regs[14]), "=f"(regs[15]),
          "=f"(regs[16]), "=f"(regs[17]), "=f"(regs[18]), "=f"(regs[19]),
          "=f"(regs[20]), "=f"(regs[21]), "=f"(regs[22]), "=f"(regs[23]),
          "=f"(regs[24]), "=f"(regs[25]), "=f"(regs[26]), "=f"(regs[27]),
          "=f"(regs[28]), "=f"(regs[29]), "=f"(regs[30]), "=f"(regs[31]),
          "=f"(regs[32]), "=f"(regs[33]), "=f"(regs[34]), "=f"(regs[35]),
          "=f"(regs[36]), "=f"(regs[37]), "=f"(regs[38]), "=f"(regs[39]),
          "=f"(regs[40]), "=f"(regs[41]), "=f"(regs[42]), "=f"(regs[43]),
          "=f"(regs[44]), "=f"(regs[45]), "=f"(regs[46]), "=f"(regs[47]),
          "=f"(regs[48]), "=f"(regs[49]), "=f"(regs[50]), "=f"(regs[51]),
          "=f"(regs[52]), "=f"(regs[53]), "=f"(regs[54]), "=f"(regs[55]),
          "=f"(regs[56]), "=f"(regs[57]), "=f"(regs[58]), "=f"(regs[59]),
          "=f"(regs[60]), "=f"(regs[61]), "=f"(regs[62]), "=f"(regs[63])
        : "r"(tmem_addr)
    );
}

// Copies 32 rows x 128 bits from matrix described in SMEM by desc
// into tmem_ptr
template<int CTA_GROUP>
__device__ void inline tcgen05_cp(int tmem_ptr, uint64_t desc) {
    asm volatile (
        "tcgen05.cp.cta_group::%2.32x128b.warpx4 [%0], %1;"
        :
        : "r"(tmem_ptr), "l"(desc), "n"(CTA_GROUP)
    );
}

__device__ uint64_t inline constexpr encode(uint64_t x) {
    return (x & 0x3FFFF) >> 4;
}

/*
    Un-changing instruction descriptor for tcgen05.mma
    [0-1]   : 0 (reserved)
    [2]     : Sparsity -> Dense = 0
    [3]     : 0 (reserved)
    [4-5]   : Matrix B Scale Factor Data ID -> always 0 for 4 SFs (using all bytes in each TMEM col)
    [6]     : 0 (reserved)
    [7-9]   : atype (Matrix A type) -> E2M1 = 1
    [10-11] : btype (Matrix B type) -> E2M1 = 1
    [12]    : 0 (reserved)
    [13]    : Negate A Matrix -> 0 (no negation)
    [14]    : Negate B Matrix -> 0 (no negation)
    [15]    : Transpose A Matrix -> 0 (transposition not allowed, nor wanted)
    [16]    : Transpose B Matrix -> 0 (^^^^)
    [17-22] : N, Dimension of Matrix B (3 LSBs not included) -> N >> 3
    [23]    : Scale Matrix Type, for both scale_A / scale_B -> UE4M3 = 0
    [24-26] : 0 (reserved)
    [27-28] : M, Dimension of Matrix A (7 LSBs not included) -> M >> 7
    [29-30] : Matrix A Scale Factor Data ID -> always 0 (same as above)
    [31]    : K Dimension -> 0 with Dense from bit 2 makes desired K=64
*/
template<int M, int N>
__device__ uint32_t inline constexpr make_instr_desc() {
    return (1 << 7) | (1 << 10) | ((N >> 3) << 17) | ((M >> 7) << 27);
}

// Complete descriptor with address info
template<int MN_DIM>
__device__ uint64_t inline make_smem_desc(int smem_addr) {
    constexpr int LBO = MN_DIM * 16; // 16B per row in core matrix
    constexpr int SBO = 8 * 16;
    return encode(smem_addr) | (encode(LBO) << 16) | (encode(SBO) << 32) | (0x1ULL << 46);
}

template<int CTA_GROUP>
__device__ void inline tcgen05_dealloc_tmem(int tmem_addr, int n_cols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::%2.sync.aligned.b32 %0, %1;" 
        :
        : "r"(tmem_addr), "r"(n_cols), "n"(CTA_GROUP)
    );
}

// Warp synchronous execution (all threads in a warp execute)
template<int CTA_GROUP>
__device__ void inline tcgen05_alloc_tmem(int *tmem_addr_ptr, const int n_cols) {
    // Performs a cvt.u64.u32, enables proper passing of smem ptr to PTX assembly
    const int tmem_addr_ptr_cvt = static_cast<int>(__cvta_generic_to_shared(tmem_addr_ptr));
    asm volatile (
        "tcgen05.alloc.cta_group::%2.sync.aligned.shared::cta.b32  [%0], %1;"
        :
        : "r"(tmem_addr_ptr_cvt), "r"(n_cols), "n"(CTA_GROUP)
    );
}

// Single thread execution
template<int CTA_GROUP>
__device__ void inline tcgen05_mma_nvfp4(int d_tmem, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc, 
                                        int sfa_tmem, int sfb_tmem, int enable_input_d) {
    asm volatile (
        "{\\n"
        ".reg .pred p;\\n"
        "setp.ne.b32 p, %4, 0;\\n"
        "tcgen05.mma.cta_group::%7.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \\n"
        "}\\n"
        :
        : "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(enable_input_d),
        "r"(sfa_tmem), "r"(sfb_tmem), "n"(CTA_GROUP)
    );
}


constexpr int WARP_SIZE = 32;
constexpr int SF_BLOCK_SIZE = 16;

/*
    Warp 0 will be responsible for all single thread (async) issued instructions, warp 1 or all CTA threads will handle the rest
    We need 1 warp to do the computation and TMA transfers. We need 4 warps in order to read/write all of TMEM (a single warp can only
    access 32 lanes (rows) in TMEM out of the total 128 per SM)

    For now we assume TD_CTA_M/N == TD_SMEM_M/N so each CTA computes just one output tile
    Also we assume TD_SMEM_M/N == TD_MMA_M/N to avoid excess copies from SMEM back to TMEM, although implementing the modifications
    To allow differing sizes shouldn't be too hard
*/
template<int TD_CTA_M, int TD_CTA_N,
         int TD_SMEM_M, int TD_SMEM_N, int TD_SMEM_K, 
         int TD_MMA_M, int TD_MMA_N, int TD_MMA_K>
__global__ void nvfp4_gemm_kernel(__half* __restrict__ c_ref, const int M, const int N, const int K, const __grid_constant__ CUtensorMap tmap_a,
                                          const __grid_constant__ CUtensorMap tmap_b, const uint8_t* sfa_gmem_base, const uint8_t* sfb_gmem_base) {
    /*
        ISSUE: Static assert that TD_SMEM_X are multiples of TD_MMA_X (for all X in {M, N, K})
               Likewise we should assert that TD_CTA_X are multiples of TD_SMEM_X (for all X in {M, N})
        We should also assert that all M, N, K dimensions are multiples of TD_CTA_M/N and TD_SMEM_K
    */

    // Statically computed values
    constexpr int WIDTH_COREMAT = 32; // ISSUE: This will change based on swizzle mode
    constexpr int SF_SMEM_TILESZ = 128 * (TD_SMEM_K / SF_BLOCK_SIZE);
    constexpr int A_SMEM_TILESZ = TD_SMEM_M * (TD_SMEM_K / 2);
    constexpr int B_SMEM_TILESZ = TD_SMEM_N * (TD_SMEM_K / 2);
    constexpr int SMEM_TILE_SZ = A_SMEM_TILESZ + B_SMEM_TILESZ + 2*SF_SMEM_TILESZ;

    // Calculate constants, offsets, etc... for this thread/warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int m_off = blockIdx.y * TD_CTA_M;
    const int n_off = blockIdx.x * TD_CTA_N;

    // Allocate SMEM buffers
    /*
        All Buffers need to be 128B aligned for TMA transfers
        Need TD_SMEM_M * TD_SMEM_K * sizeof(nvfp4) bytes for a_smem
        Need TD_SMEM_M * (TD_SMEM_K/SF_BLOCK_SIZE) * sizeof(fp8) bytes for sfa_smem
        Need TD_SMEM_N * TD_SMEM_K * sizeof(nvfp4) bytes for b_smem
        Need TD_SMEM_N * (TD_SMEM_K/SF_BLOCK_SIZE) * sizeof(fp8) bytes for sfb_smem
    */
    __shared__ alignas(128) char a_smem[A_SMEM_TILESZ];
    __shared__ alignas(128) char sfa_smem[SF_SMEM_TILESZ];
    __shared__ alignas(128) char b_smem[B_SMEM_TILESZ];
    __shared__ alignas(128) char sfb_smem[SF_SMEM_TILESZ];
    // Convert ptrs to properly pass to inline PTX
    const int a_smem_ptr = static_cast<int>(__cvta_generic_to_shared(a_smem));
    const int sfa_smem_ptr = static_cast<int>(__cvta_generic_to_shared(sfa_smem));
    const int b_smem_ptr = static_cast<int>(__cvta_generic_to_shared(b_smem));
    const int sfb_smem_ptr = static_cast<int>(__cvta_generic_to_shared(sfb_smem));

    // Allocate TMEM buffers, single warp execution ISSUE: IMPLEMENT TWO ALLOC TECHNIQUE WITH ONE FOR RESULT ONE FOR SF TILES (THERE ARE TRADEOFFS WITH ALLOCATION SPACE VS NUM ALLOCATIONS, COULD BE A PROB SIZE SPECIFIC THING)
    /*
        We need TD_MMA_N columns for the result (using FP32 accumulation)
        SFA needs TD_MMA_M/32 columns per 64 elements in K (due to (32x4)xcols layout discussed previously) -> (TD_MMA_M/32) * (TD_SMEM_K/64) total columns
        SFB needs TD_MMA_N/32 columns per 64 elements in K (due to (32x4)xcols layout) -> (TD_MMA_N/32) * (TD_SMEM_K/64) total columns
        TMEM Address Structure: 
        [0-15]  : Column Index
        [31-16] : Lane Index
        MMA result buffer
    */
    __shared__ int tmem_addr_ptr[1];
    if (warp_id == 1) { tcgen05_alloc_tmem<1>(tmem_addr_ptr, TD_MMA_N*2 ); }
    __syncthreads(); // ISSUE: Why do we need this here?
    const int tmem_addr_result = tmem_addr_ptr[0];
    const int tmem_addr_sfa = tmem_addr_result + TD_MMA_N;
    const int tmem_addr_sfb = tmem_addr_sfa + (TD_MMA_M/32) * (TD_SMEM_K/64);

    // Setup memory barriers
    __shared__ alignas(8) int64_t mbar[1];
    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbar));

    if (warp_id == 0 && elect_one_sync()) {
        mbar_init(mbar_addr, 1);
        asm volatile("fence.mbarrier_init.release.cluster;"); // ISSUE: Verify we need this here
    }

    int phase = 0;
    
    // ISSUE: No need for M, N tile loops since TD_CTA_X == TD_SMEM_X

    // Loop over K tiles
    for (int k_off = 0; k_off < K; k_off += TD_SMEM_K) {
        // Load GMEM -> SMEM tiles
        if (warp_id == 0 && elect_one_sync()) {
            const int k_off_coremat = k_off / WIDTH_COREMAT;
            tcgen05_3dtma_g2s_ab<1>(a_smem_ptr, &tmap_a, m_off, k_off_coremat, mbar_addr, CacheHintSm100::EVICT_NORMAL);
            tcgen05_3dtma_g2s_ab<1>(b_smem_ptr, &tmap_b, n_off, k_off_coremat, mbar_addr, CacheHintSm100::EVICT_NORMAL);
            /*
                Scale factors are stored in global memory in 4x4x32 chunks, i.e. 512B chunks where each chunk represents a
                128x4 chunk of the SF matrix (in M or N xK)
                So we calculate the offset in each dimension in terms of these 512B chunks:
                k_off / 64 represents the number of 128x4 (512B) chunks along the K dimension which are contiguous (4 * SF_BLOCKS_SIZE = 64)
                m/n_off / 128 represents the number of 512B chunks along the M dimension which are strided by K / 64 512B chunks
            */
            const uint8_t* sfa_g_ptr = sfa_gmem_base + ((k_off / 64) + (m_off / 128) * (K / 64)) * 512; // ISSUE: These could be just simple bit shifts, adjust if compiler doesn't
            const uint8_t* sfb_g_ptr = sfb_gmem_base + ((k_off / 64) + (n_off / 128) * (K / 64)) * 512;
            tcgen05_1dtma_g2s_sf(sfa_smem_ptr, sfa_g_ptr, SF_SMEM_TILESZ, mbar_addr, CacheHintSm100::EVICT_NORMAL);
            tcgen05_1dtma_g2s_sf(sfb_smem_ptr, sfb_g_ptr, SF_SMEM_TILESZ, mbar_addr, CacheHintSm100::EVICT_NORMAL);

            // Signal in mbarrier that we expect SMEM_TILE_SZ bytes to arrive on this mbar object before proceeding to next phase
            mbar_arrive_expect(mbar_addr, SMEM_TILE_SZ);
        }

        // Wait for all bytes to arrive in SMEM by checking the mbarrier object
        mbar_wait(mbar_addr, phase);
        asm volatile("tcgen05.fence::after_thread_sync;"); // ISSUE: Probably don't need this here
        // Toggle for next phase in the k-loop which is the mma
        phase ^= 1;

        // MMA
        if (warp_id == 0 && elect_one_sync()) {
            // Load scale factors SMEM -> TMEM
            for (int sub_k_iter = 0; sub_k_iter < TD_SMEM_K / TD_MMA_K; sub_k_iter++) {
                uint64_t sfa_desc = make_smem_desc<0>(sfa_smem_ptr + (sub_k_iter * 512)); // ISSUE: verify this should input 0 here
                uint64_t sfb_desc = make_smem_desc<0>(sfb_smem_ptr + (sub_k_iter * 512));
                tcgen05_cp<1>(tmem_addr_sfa + 4 * sub_k_iter, sfa_desc);
                tcgen05_cp<1>(tmem_addr_sfb + 4 * sub_k_iter, sfb_desc);
            }

            // Loop over SMEM tile K-dim
            for (int sub_k_iter = 0; sub_k_iter < TD_SMEM_K / TD_MMA_K; sub_k_iter++) {
                uint64_t a_desc = make_smem_desc<TD_MMA_M>(a_smem_ptr + sub_k_iter * TD_MMA_K * TD_MMA_M / 2);
                uint64_t b_desc = make_smem_desc<TD_MMA_N>(b_smem_ptr + sub_k_iter * TD_MMA_K * TD_MMA_N / 2);
                int sfa_tmem = tmem_addr_sfa + 4 * sub_k_iter;
                int sfb_tmem = tmem_addr_sfb + 4 * sub_k_iter + (n_off%128)/32; // ISSUE: Add support for TD_MMA_N = 64 via TMEM col offset here

                tcgen05_mma_nvfp4<1>(tmem_addr_result, a_desc, b_desc, make_instr_desc<TD_MMA_M, TD_MMA_N>(), sfa_tmem, sfb_tmem, k_off + sub_k_iter); // Inputting k_off like this will set enable-input-d so only on the first mma we 0 out the result space in TMEM
            }
            // signal MMA done
            tcgen05_commit(mbar_addr);
        }

        mbar_wait(mbar_addr, phase);
        phase ^= 1;
    }

    // Epilogue
    asm volatile("tcgen05.fence::after_thread_sync;"); // ISSUE: Verify we need this

    // Load MMA into regs (TMEM -> Regs)
    // Each warp loads 16 rows per tcgen05_ld and we have 128 rows, with 4 warps each one is responsible for 32 rows
    float results[TD_MMA_N / 2];
    int rows_per_warp = TD_MMA_M / (blockDim.x / 32);
    for (int sub_m = 0; sub_m <  rows_per_warp / 16; sub_m++) {
        tcgen05_ld<16, 256, 16>(results, tmem_addr_result + (((warp_id * rows_per_warp) + sub_m * 16) << 16));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        // Post process and store from Regs to SMEM (Regs -> SMEM)
        // Transfer result from SMEM -> GMEM (8 comes from 256/32 -> 256b per ld block from above)
        for (int i = 0; i < TD_MMA_N / 8; i++) {
            const int m_offset = m_off + warp_id * rows_per_warp + sub_m * 16 + lane_id / 4;
            const int n_offset = n_off + i * 8 + (lane_id % 4) * 2;
            reinterpret_cast<half2 *>(c_ref + (m_offset)*N + n_offset)[0] = __float22half2_rn({results[i * 4], results[i * 4 + 1]});
            reinterpret_cast<half2 *>(c_ref + (m_offset + 8)*N + n_offset)[0] = __float22half2_rn({results[i * 4 + 2], results[i * 4 + 3]});
        }
    }

    // Free memory
    __syncthreads();
    if (warp_id == 0) { tcgen05_dealloc_tmem<1>(tmem_addr_result, TD_MMA_N * 2); }  
}

torch::Tensor nvfp4_gemm(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, 
                                 torch::Tensor c_ref, int M, int N, int K) { 

    
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    constexpr int M_TILE_SIZE = 128;
    constexpr int N_TILE_SIZE = 128;
    constexpr int K_TILE_SIZE = 512;
    constexpr int K_MMA_SIZE = 64;

    CUtensorMap tmap_a, tmap_b;
    init_tma_map<M_TILE_SIZE, K_TILE_SIZE>(cuTensorMapEncodeTiled, &tmap_a, a_ref.data_ptr(), M, K);
    init_tma_map<N_TILE_SIZE, K_TILE_SIZE>(cuTensorMapEncodeTiled, &tmap_b, b_ref.data_ptr(), N, K);


    auto kernel_inst = nvfp4_gemm_kernel<M_TILE_SIZE, N_TILE_SIZE, M_TILE_SIZE, N_TILE_SIZE, K_TILE_SIZE, M_TILE_SIZE, N_TILE_SIZE, K_MMA_SIZE>;

    cudaFuncSetAttribute(
        kernel_inst,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared  // Maximum shared memory
    );

    constexpr int threads = WARP_SIZE * 4;
    dim3 grid_dim(N/N_TILE_SIZE, M/M_TILE_SIZE);
    kernel_inst<<<grid_dim, threads>>>(reinterpret_cast<__half*>(c_ref.data_ptr()), M, N, K, tmap_a, tmap_b, reinterpret_cast<uint8_t*>(sfa_ref.data_ptr()), reinterpret_cast<uint8_t*>(sfb_ref.data_ptr()));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return c_ref; 
}

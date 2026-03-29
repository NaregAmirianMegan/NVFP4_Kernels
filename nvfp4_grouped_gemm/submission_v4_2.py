#!POPCORN leaderboard nvfp4_group_gemm
import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t
from utils import make_match_reference

"""
Reduced overhead
"""

nvfp4_group_gemm_cuda_source = """

#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda_fp16.h>
#include <chrono>

/*
    Warp specialization and pipelining
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

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

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

template<int MN_SMEM_TD, int K_SMEM_TD, CUtensorMapSwizzle SWIZZLE>
struct tma_3d_map_ab {
    static void init(PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled, CUtensorMap* tmap, void* ptr, uint64_t mn_dim_gmem, uint64_t k_dim_gmem);
};
// For No swizzle canonical layout of 128b segments is 8x1 (or core matrices of 8 rows x 32 element columns (since 32 NVFP4 = 16B = 128b))
template<int MN_SMEM_TD, int K_SMEM_TD>
struct tma_3d_map_ab<MN_SMEM_TD, K_SMEM_TD, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE> {
    static void init(PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled, CUtensorMap* tmap, void* ptr, uint64_t mn_dim_gmem, uint64_t k_dim_gmem) {
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
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, // Interleave patterns can be used to accelerate loading of values that are less than 4 bytes long.
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE, // Swizzling can be used to avoid shared memory bank conflicts.
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE, // L2 Promotion can be used to widen the effect of a cache-policy to a wider set of L2 cache lines.
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // Any element that is outside of bounds will be set to zero by the TMA transfer.
        );
        // ISSUE: Insert error check here on res
    }
};
// For 128B swizzle canonical layout of 128b segments is 8x8 (or core matrices of 8 rows x 256 element columns)
template<int MN_SMEM_TD, int K_SMEM_TD>
struct tma_3d_map_ab<MN_SMEM_TD, K_SMEM_TD, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B> {
    static void init(PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled, CUtensorMap* tmap, void* ptr, uint64_t mn_dim_gmem, uint64_t k_dim_gmem) {
        constexpr uint32_t rank = 3;
        uint64_t dim_gmem[rank] = {256, mn_dim_gmem, k_dim_gmem/256};
        uint64_t stride_gmem[rank - 1] = {k_dim_gmem/2, 128};
        uint32_t dim_smem[rank] = {256, MN_SMEM_TD, K_SMEM_TD/256};
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
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, // Interleave patterns can be used to accelerate loading of values that are less than 4 bytes long.
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B, // Swizzling can be used to avoid shared memory bank conflicts.
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE, // L2 Promotion can be used to widen the effect of a cache-policy to a wider set of L2 cache lines.
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // Any element that is outside of bounds will be set to zero by the TMA transfer.
        );
        // ISSUE: Insert error check here on res
    }
};

template<int K_SMEM_TD>
void tma_3d_map_sf(PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled, CUtensorMap* tmap, void* ptr, uint64_t mn_dim_gmem, uint64_t k_dim_gmem) {
    constexpr uint32_t rank = 3;
    constexpr int K_SMEM_TD_SF = K_SMEM_TD / 16;
    const int k_dim_gmem_sf = k_dim_gmem / 16;
    uint64_t dim_gmem[rank] = {256, 2ULL * (k_dim_gmem_sf / 4), mn_dim_gmem / 128};
    uint64_t stride_gmem[rank - 1] = {256, 512ULL * (k_dim_gmem_sf / 4)};
    uint32_t dim_smem[rank] = {256, 2ULL * (K_SMEM_TD_SF / 4), 1};
    uint32_t elem_stride[rank] = {1, 1, 1};

    // Create the tensor descriptor.
    auto res = cuTensorMapEncodeTiled(
        tmap,                // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
        rank,                       // cuuint32_t tensorRank,
        ptr,                 // void *globalAddress,
        dim_gmem,                       // const cuuint64_t *globalDim,
        stride_gmem,                     // const cuuint64_t *globalStrides,
        dim_smem,                   // const cuuint32_t *boxDim,
        elem_stride,                // const cuuint32_t *elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE, // Interleave patterns can be used to accelerate loading of values that are less than 4 bytes long.
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE, // Swizzling can be used to avoid shared memory bank conflicts.
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE, // L2 Promotion can be used to widen the effect of a cache-policy to a wider set of L2 cache lines.
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // Any element that is outside of bounds will be set to zero by the TMA transfer.
    );

    if (res != CUDA_SUCCESS) {
        printf("TMA encode failed with error %d\\n", res);
    }
}

/*
    For each warp:
        LANES: How many rows of TMEM are loaded
        WIDTH: How many bits in each row
        REPT: WIDTH repeated REPT times
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
__device__ void inline tcgen05_ld<16, 256, 8>(float* regs, int tmem_addr) {
    asm volatile (
        "tcgen05.ld.sync.aligned.16x256b.x8.b32 {    %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                                                   "  %8,  %9,  %10, %11, %12, %13, %14, %15, "
                                                   "  %16, %17, %18, %19, %20, %21, %22, %23, "
                                                   "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(regs[0]),  "=f"(regs[1]),  "=f"(regs[2]),  "=f"(regs[3]),
          "=f"(regs[4]),  "=f"(regs[5]),  "=f"(regs[6]),  "=f"(regs[7]),
          "=f"(regs[8]),  "=f"(regs[9]),  "=f"(regs[10]), "=f"(regs[11]),
          "=f"(regs[12]), "=f"(regs[13]), "=f"(regs[14]), "=f"(regs[15]),
          "=f"(regs[16]), "=f"(regs[17]), "=f"(regs[18]), "=f"(regs[19]),
          "=f"(regs[20]), "=f"(regs[21]), "=f"(regs[22]), "=f"(regs[23]),
          "=f"(regs[24]), "=f"(regs[25]), "=f"(regs[26]), "=f"(regs[27]),
          "=f"(regs[28]), "=f"(regs[29]), "=f"(regs[30]), "=f"(regs[31])
        : "r"(tmem_addr)
    );
}
template<>
__device__ void inline tcgen05_ld<16, 256, 16>(float* regs, int tmem_addr) {
    asm volatile (
        "tcgen05.ld.sync.aligned.16x256b.x16.b32 {    %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
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
template<>
__device__ void inline tcgen05_ld<16, 256, 32>(float* regs, int tmem_addr) {
    asm volatile (
        "tcgen05.ld.sync.aligned.16x256b.x32.b32 {    %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
                                                   "  %8,  %9,  %10, %11, %12, %13, %14, %15, "
                                                   "  %16, %17, %18, %19, %20, %21, %22, %23, "
                                                   "  %24, %25, %26, %27, %28, %29, %30, %31, "
                                                   "  %32, %33, %34, %35, %36, %37, %38, %39, "
                                                   "  %40, %41, %42, %43, %44, %45, %46, %47, "
                                                   "  %48, %49, %50, %51, %52, %53, %54, %55, "
                                                   "  %56, %57, %58, %59, %60, %61, %62, %63, "
                                                   "  %64, %65, %66, %67, %68, %69, %70, %71, "
                                                   "  %72, %73, %74, %75, %76, %77, %78, %79, "
                                                   "  %80, %81, %82, %83, %84, %85, %86, %87, "
                                                   "  %88, %89, %90, %91, %92, %93, %94, %95, "
                                                   "  %96, %97, %98, %99, %100, %101, %102, %103, "
                                                   "  %104, %105, %106, %107, %108, %109, %110, %111, "
                                                   "  %112, %113, %114, %115, %116, %117, %118, %119, "
                                                   "  %120, %121, %122, %123, %124, %125, %126, %127}, [%128];"
        : "=f"(regs[0]),   "=f"(regs[1]),   "=f"(regs[2]),   "=f"(regs[3]),
          "=f"(regs[4]),   "=f"(regs[5]),   "=f"(regs[6]),   "=f"(regs[7]),
          "=f"(regs[8]),   "=f"(regs[9]),   "=f"(regs[10]),  "=f"(regs[11]),
          "=f"(regs[12]),  "=f"(regs[13]),  "=f"(regs[14]),  "=f"(regs[15]),
          "=f"(regs[16]),  "=f"(regs[17]),  "=f"(regs[18]),  "=f"(regs[19]),
          "=f"(regs[20]),  "=f"(regs[21]),  "=f"(regs[22]),  "=f"(regs[23]),
          "=f"(regs[24]),  "=f"(regs[25]),  "=f"(regs[26]),  "=f"(regs[27]),
          "=f"(regs[28]),  "=f"(regs[29]),  "=f"(regs[30]),  "=f"(regs[31]),
          "=f"(regs[32]),  "=f"(regs[33]),  "=f"(regs[34]),  "=f"(regs[35]),
          "=f"(regs[36]),  "=f"(regs[37]),  "=f"(regs[38]),  "=f"(regs[39]),
          "=f"(regs[40]),  "=f"(regs[41]),  "=f"(regs[42]),  "=f"(regs[43]),
          "=f"(regs[44]),  "=f"(regs[45]),  "=f"(regs[46]),  "=f"(regs[47]),
          "=f"(regs[48]),  "=f"(regs[49]),  "=f"(regs[50]),  "=f"(regs[51]),
          "=f"(regs[52]),  "=f"(regs[53]),  "=f"(regs[54]),  "=f"(regs[55]),
          "=f"(regs[56]),  "=f"(regs[57]),  "=f"(regs[58]),  "=f"(regs[59]),
          "=f"(regs[60]),  "=f"(regs[61]),  "=f"(regs[62]),  "=f"(regs[63]),
          "=f"(regs[64]),  "=f"(regs[65]),  "=f"(regs[66]),  "=f"(regs[67]),
          "=f"(regs[68]),  "=f"(regs[69]),  "=f"(regs[70]),  "=f"(regs[71]),
          "=f"(regs[72]),  "=f"(regs[73]),  "=f"(regs[74]),  "=f"(regs[75]),
          "=f"(regs[76]),  "=f"(regs[77]),  "=f"(regs[78]),  "=f"(regs[79]),
          "=f"(regs[80]),  "=f"(regs[81]),  "=f"(regs[82]),  "=f"(regs[83]),
          "=f"(regs[84]),  "=f"(regs[85]),  "=f"(regs[86]),  "=f"(regs[87]),
          "=f"(regs[88]),  "=f"(regs[89]),  "=f"(regs[90]),  "=f"(regs[91]),
          "=f"(regs[92]),  "=f"(regs[93]),  "=f"(regs[94]),  "=f"(regs[95]),
          "=f"(regs[96]),  "=f"(regs[97]),  "=f"(regs[98]),  "=f"(regs[99]),
          "=f"(regs[100]), "=f"(regs[101]), "=f"(regs[102]), "=f"(regs[103]),
          "=f"(regs[104]), "=f"(regs[105]), "=f"(regs[106]), "=f"(regs[107]),
          "=f"(regs[108]), "=f"(regs[109]), "=f"(regs[110]), "=f"(regs[111]),
          "=f"(regs[112]), "=f"(regs[113]), "=f"(regs[114]), "=f"(regs[115]),
          "=f"(regs[116]), "=f"(regs[117]), "=f"(regs[118]), "=f"(regs[119]),
          "=f"(regs[120]), "=f"(regs[121]), "=f"(regs[122]), "=f"(regs[123]),
          "=f"(regs[124]), "=f"(regs[125]), "=f"(regs[126]), "=f"(regs[127])
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
__device__ uint32_t constexpr make_instr_desc() {
    return (1 << 7) | (1 << 10) | ((N >> 3) << 17) | ((M >> 7) << 27);
}

// Complete descriptor with address info
template<int MN_DIM, bool SWIZZLE_128B = false>
__device__ uint64_t inline make_smem_desc(int smem_addr) {
    constexpr uint64_t LBO = SWIZZLE_128B ? 1 : MN_DIM*16;
    constexpr uint64_t SBO = SWIZZLE_128B ? 8 * 128 : 8 * 16;
    constexpr uint64_t SWIZZLE_BITS = SWIZZLE_128B ? 2 : 0;
    return encode(smem_addr) | (encode(LBO) << 16) | (encode(SBO) << 32) | (0x1ULL << 46) | (SWIZZLE_BITS << 61);
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

template<int A, int B>
int constexpr MAX() {
    if (A < B) 
        return B;
    return A;
}

struct GroupDesc {
    void* A_addr;
    void* B_addr;
    __half* C_addr;
    uint8_t* sfa_addr;
    uint8_t* sfb_addr;
    int M;
    int block_start;
};

constexpr int WARP_SIZE = 32;
constexpr int SF_BLOCK_SIZE = 16;

/*
    Warp 0 will be responsible for all single thread (async) issued instructions, warp 1 or all CTA threads will handle the rest
    We need 1 warp to do the computation and TMA transfers. We need 4 warps in order to read/write all of TMEM (a single warp can only
    access 32 lanes (rows) in TMEM out of the total 128 per SM)

    For now we assume TD_CTA_M/N == TD_SMEM_M/N so each CTA computes just one output tile
    Also we assume TD_SMEM_M/N == TD_MMA_M/N to avoid excess copies from SMEM back to TMEM, although implementing the modifications
    To allow differing sizes shouldn't be too hard

    ISSUE: NUM_WARPS should match launch conditions (is there a cleaner way to handle this, maybe launch bounds?)
*/
template<int TD_CTA_M, int TD_CTA_N,
         int TD_SMEM_M, int TD_SMEM_N, int TD_SMEM_K, 
         int TD_MMA_M, int TD_MMA_N, int TD_MMA_K, bool SWIZZLE, int PIPE_STAGES, int NUM_WARPS>
__global__ void nvfp4_group_gemm_kernel(GroupDesc* groups, const __grid_constant__ CUtensorMap tmap_a_temp, 
                                        const __grid_constant__ CUtensorMap tmap_b_temp, const int total_tiles, 
                                        CUtensorMap* d_tmaps, const int N, const int K, const int G) {
    /*
        ISSUE: Static assert that TD_SMEM_X are multiples of TD_MMA_X (for all X in {M, N, K})
               Likewise we should assert that TD_CTA_X are multiples of TD_SMEM_X (for all X in {M, N})
        We should also assert that all M, N, K dimensions are multiples of TD_CTA_M/N and TD_SMEM_K
    */

    // Statically computed values
    constexpr int WIDTH_COREMAT = SWIZZLE ? 256 : 32;
    constexpr int SF_SMEM_TILESZ = 128 * (TD_SMEM_K / SF_BLOCK_SIZE);
    constexpr int A_SMEM_TILESZ = TD_SMEM_M * (TD_SMEM_K / 2);
    constexpr int B_SMEM_TILESZ = TD_SMEM_N * (TD_SMEM_K / 2);
    constexpr int SMEM_TILE_SZ = A_SMEM_TILESZ + B_SMEM_TILESZ + SF_SMEM_TILESZ + (1 + TD_MMA_N/256)*SF_SMEM_TILESZ;

    // Calculate constants, offsets, etc... for this thread/warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Allocate SMEM buffers
    /*
        All Buffers need to be 128B aligned for TMA transfers
        Need TD_SMEM_M * TD_SMEM_K * sizeof(nvfp4) bytes for a_smem
        Need TD_SMEM_M * (TD_SMEM_K/SF_BLOCK_SIZE) * sizeof(fp8) bytes for sfa_smem
        Need TD_SMEM_N * TD_SMEM_K * sizeof(nvfp4) bytes for b_smem
        Need TD_SMEM_N * (TD_SMEM_K/SF_BLOCK_SIZE) * sizeof(fp8) bytes for sfb_smem
    */
    __shared__ alignas(128) char a_smem[PIPE_STAGES * A_SMEM_TILESZ];
    __shared__ alignas(128) char sfa_smem[PIPE_STAGES * SF_SMEM_TILESZ];
    __shared__ alignas(128) char b_smem[PIPE_STAGES * B_SMEM_TILESZ];
    __shared__ alignas(128) char sfb_smem[PIPE_STAGES * ((1 + TD_MMA_N/256) * SF_SMEM_TILESZ)];
    // Convert ptrs to properly pass to inline PTX
    const int a_smem_ptr = static_cast<int>(__cvta_generic_to_shared(a_smem));
    const int sfa_smem_ptr = static_cast<int>(__cvta_generic_to_shared(sfa_smem));
    const int b_smem_ptr = static_cast<int>(__cvta_generic_to_shared(b_smem));
    const int sfb_smem_ptr = static_cast<int>(__cvta_generic_to_shared(sfb_smem));

    // GMEM cache for CTA specific tensor maps
    CUtensorMap* g_A_tmap = d_tmaps + 2*blockIdx.x;
    CUtensorMap* g_B_tmap = g_A_tmap + 1;

    __shared__ CUtensorMap local_A_tmap; // ISSUE: Do these need to be aligned?
    __shared__ CUtensorMap local_B_tmap;

    if (warp_id == 0 && elect_one_sync()) {
        local_A_tmap = tmap_a_temp;
        local_B_tmap = tmap_b_temp;
    }

    const int local_A_tmap_addr = static_cast<int>(__cvta_generic_to_shared(&local_A_tmap));
    const int local_B_tmap_addr = static_cast<int>(__cvta_generic_to_shared(&local_B_tmap));

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

    // Setup memory barriers
    __shared__ alignas(8) int64_t mbars[PIPE_STAGES * 2 + 1];
    const int mbar_addr_tma = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int mbar_addr_mma = mbar_addr_tma + PIPE_STAGES * 8; // 8 because each mbar is 64bits = 8B
    const int mbar_addr_epi = mbar_addr_mma + PIPE_STAGES * 8;

    if (warp_id == 0 && elect_one_sync()) {
        for (int i = 0; i < PIPE_STAGES * 2 + 1; i++) {
            mbar_init(mbar_addr_tma + i * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;"); // ISSUE: Verify we need this here
    } else if (warp_id == 1) { 
        tcgen05_alloc_tmem<1>(tmem_addr_ptr, TD_MMA_N*2); 
    }
    __syncthreads(); // Ensure all threads have correct TMEM ptrs

    const int tmem_addr_result = tmem_addr_ptr[0];
    const int tmem_addr_sfa = tmem_addr_result + TD_MMA_N;
    const int tmem_addr_sfb = tmem_addr_sfa + (TD_MMA_M/32) * (TD_SMEM_K/64);

    constexpr int TMA_WARP = NUM_WARPS - 2;
    constexpr int MMA_WARP = NUM_WARPS - 1;
    
    // ISSUE: No need for M, N tile loops since TD_CTA_X == TD_SMEM_X

    // Work-tile loop
    const int n_tiles = CEIL_DIV(N, TD_CTA_N);
    int group = 0;
    int next_group_start = 0;
    uint8_t* sfa_gmem_base;
    uint8_t* sfb_gmem_base;
    for (int tile_idx = blockIdx.x; tile_idx < total_tiles; tile_idx += gridDim.x) {
        bool update_group = (tile_idx >= next_group_start);
        if (update_group) {
            while (group < G - 1 && tile_idx >= groups[group + 1].block_start) {
                group++;
            }
            next_group_start = (group < G - 1) ? groups[group + 1].block_start : INT_MAX;
        }

        int group_tile = tile_idx - groups[group].block_start;
        int row_idx = group_tile / n_tiles;
        int col_idx = group_tile % n_tiles;
        int m_off = row_idx * TD_CTA_M;
        int n_off = col_idx * TD_CTA_N;

        // Perform MMA
        // TMA thread loops over SMEM tile stages and loads from GMEM->SMEM
        if (warp_id == TMA_WARP) {
            if (update_group) {
                sfa_gmem_base = groups[group].sfa_addr;
                sfb_gmem_base = groups[group].sfb_addr;

                if (elect_one_sync()) {
                    // Adjust M-dim value
                    asm volatile(
                        "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], %1, %2;"
                        :
                        : "r"(local_A_tmap_addr), "n"(1), "r"(groups[group].M)
                    );

                    // Update base addresses
                    asm volatile(
                        "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                        :
                        : "r"(local_A_tmap_addr), "l"(groups[group].A_addr)
                    );

                    asm volatile(
                        "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
                        :
                        : "r"(local_B_tmap_addr), "l"(groups[group].B_addr)
                    ); 
                }

                __syncwarp();

                asm volatile(
                    "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
                    :
                    : "l"(g_A_tmap), "r"(local_A_tmap_addr)
                );
                asm volatile(
                    "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                    :
                    : "l"(g_A_tmap)
                );

                asm volatile(
                    "tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [%0], [%1], 128;"
                    :
                    : "l"(g_B_tmap), "r"(local_B_tmap_addr)
                );
                asm volatile(
                    "fence.proxy.tensormap::generic.acquire.gpu [%0], 128;"
                    :
                    : "l"(g_B_tmap)
                );
            }

            if (elect_one_sync()) {
                auto tma_load_stage = [&](const int k_off, const int stage) {
                    const int k_off_coremat = k_off / WIDTH_COREMAT;
                    const int a_smem_stage_ptr = a_smem_ptr + stage * A_SMEM_TILESZ;
                    const int b_smem_stage_ptr = b_smem_ptr + stage * B_SMEM_TILESZ;
                    const int sfa_smem_stage_ptr = sfa_smem_ptr + stage * SF_SMEM_TILESZ;
                    const int sfb_smem_stage_ptr = sfb_smem_ptr + stage * SF_SMEM_TILESZ  * (TD_MMA_N == 256 ? 2 : 1);
                    const int mbar_addr_tma_stage = mbar_addr_tma + stage * 8;

                    tcgen05_3dtma_g2s_ab<1>(a_smem_stage_ptr, g_A_tmap, m_off, k_off_coremat, mbar_addr_tma_stage, CacheHintSm100::EVICT_NORMAL);
                    tcgen05_3dtma_g2s_ab<1>(b_smem_stage_ptr, g_B_tmap, n_off, k_off_coremat, mbar_addr_tma_stage, CacheHintSm100::EVICT_NORMAL);
                    /*
                        Scale factors are stored in global memory in 4x4x32 chunks, i.e. 512B chunks where each chunk represents a
                        128x4 chunk of the SF matrix (in M or N xK)
                        So we calculate the offset in each dimension in terms of these 512B chunks:
                        k_off / 64 represents the number of 128x4 (512B) chunks along the K dimension which are contiguous (4 * SF_BLOCKS_SIZE = 64)
                        m/n_off / 128 represents the number of 512B chunks along the M dimension which are strided by K / 64 512B chunks
                    */
                    const uint8_t* sfa_g_ptr = sfa_gmem_base + ((k_off / 64) + (m_off / 128) * (K / 64)) * 512; // ISSUE: These could be just simple bit shifts, adjust if compiler doesn't
                    const uint8_t* sfb_g_ptr = sfb_gmem_base + ((k_off / 64) + (n_off / 128) * (K / 64)) * 512;
                    tcgen05_1dtma_g2s_sf(sfa_smem_stage_ptr, sfa_g_ptr, SF_SMEM_TILESZ, mbar_addr_tma_stage, CacheHintSm100::EVICT_NORMAL);
                    tcgen05_1dtma_g2s_sf(sfb_smem_stage_ptr, sfb_g_ptr, SF_SMEM_TILESZ, mbar_addr_tma_stage, CacheHintSm100::EVICT_NORMAL);
                    if constexpr (TD_MMA_N == 256) {
                        tcgen05_1dtma_g2s_sf(sfb_smem_stage_ptr+SF_SMEM_TILESZ, sfb_g_ptr + (K / 64)*512, SF_SMEM_TILESZ, mbar_addr_tma_stage, CacheHintSm100::EVICT_NORMAL);
                    }
                    
                    // Signal in mbarrier that we expect SMEM_TILE_SZ bytes to arrive on this mbar object before proceeding to next phase
                    mbar_arrive_expect(mbar_addr_tma_stage, SMEM_TILE_SZ);
                };

                // Fill the TMA pipe
                for (int stage = 0; stage < PIPE_STAGES; stage++) {
                    tma_load_stage(stage * TD_SMEM_K, stage);
                }

                // Cycle through tile stages, loading tiles once no longer in use by the MMA stage
                int stage = 0;
                for (int k_off = TD_SMEM_K * PIPE_STAGES; k_off < K; k_off += TD_SMEM_K) {
                  mbar_wait(mbar_addr_mma + stage * 8, (((k_off / TD_SMEM_K) / PIPE_STAGES) - 1) % 2);
                  tma_load_stage(k_off, stage);
                  stage = (stage + 1) % PIPE_STAGES;
                }
            }
        }
        // MMA thread loops over SMEM tile stages, loads TMEM and computes MMA ops
        else if (warp_id == MMA_WARP && elect_one_sync()) {
            for (int k_off = 0; k_off < K; k_off += TD_SMEM_K) {
                const int stage = (k_off / TD_SMEM_K) % PIPE_STAGES;
                mbar_wait(mbar_addr_tma + stage * 8, ((k_off / TD_SMEM_K) / PIPE_STAGES) % 2);

                const int a_smem_stage_ptr = a_smem_ptr + stage * A_SMEM_TILESZ;
                const int b_smem_stage_ptr = b_smem_ptr + stage * B_SMEM_TILESZ;
                const int sfa_smem_stage_ptr = sfa_smem_ptr + stage * SF_SMEM_TILESZ;
                const int sfb_smem_stage_ptr = sfb_smem_ptr + stage * SF_SMEM_TILESZ  * (TD_MMA_N == 256 ? 2 : 1);
                const int mbar_addr_mma_stage = mbar_addr_mma + stage * 8;

                // Load scale factors SMEM -> TMEM
                for (int sub_k_iter = 0; sub_k_iter < TD_SMEM_K / TD_MMA_K; sub_k_iter++) {
                    uint64_t sfa_desc = make_smem_desc<0, false>(sfa_smem_stage_ptr + (sub_k_iter * 512)); // ISSUE: verify this should input 0 here
                    uint64_t sfb_desc = make_smem_desc<0, false>(sfb_smem_stage_ptr + (sub_k_iter * 512));
                    tcgen05_cp<1>(tmem_addr_sfa + 4 * sub_k_iter, sfa_desc);
                    tcgen05_cp<1>(tmem_addr_sfb + MAX<TD_MMA_N / 32, 4>() * sub_k_iter, sfb_desc);
                    if constexpr (TD_MMA_N == 256) {
                        uint64_t sfb_desc2 = make_smem_desc<0, false>(sfb_smem_stage_ptr + SF_SMEM_TILESZ + (sub_k_iter * 512));
                        tcgen05_cp<1>(tmem_addr_sfb + 8 * sub_k_iter + 4, sfb_desc2);
                    }
                }

                // Loop over SMEM tile K-dim
                for (int sub_k_iter = 0; sub_k_iter < TD_SMEM_K / TD_MMA_K; sub_k_iter++) {
                    // Stride computed differently depending on swizzle mode because it changes core matrix shape
                    uint64_t a_desc, b_desc;
                    if constexpr (SWIZZLE) {
                        a_desc = make_smem_desc<TD_MMA_M, SWIZZLE>(a_smem_stage_ptr + sub_k_iter * 32);
                        b_desc = make_smem_desc<TD_MMA_N, SWIZZLE>(b_smem_stage_ptr + sub_k_iter * 32);
                    }
                    else {
                        a_desc = make_smem_desc<TD_MMA_M, SWIZZLE>(a_smem_stage_ptr + sub_k_iter * TD_MMA_K * TD_MMA_M / 2);
                        b_desc = make_smem_desc<TD_MMA_N, SWIZZLE>(b_smem_stage_ptr + sub_k_iter * TD_MMA_K * TD_MMA_N / 2);
                    }
                    int sfa_tmem = tmem_addr_sfa + 4 * sub_k_iter;
                    int sfb_tmem;
                    if constexpr (TD_MMA_N == 256) {
                        sfb_tmem = tmem_addr_sfb + 8 * sub_k_iter;
                    } else {
                        sfb_tmem = tmem_addr_sfb + 4 * sub_k_iter + (n_off%128)/32;
                    }

                    tcgen05_mma_nvfp4<1>(tmem_addr_result, a_desc, b_desc, make_instr_desc<TD_MMA_M, TD_MMA_N>(), sfa_tmem, sfb_tmem, k_off + sub_k_iter); // Inputting k_off like this will set enable-input-d so only on the first mma we 0 out the result space in TMEM
                }
                // signal MMA done
                tcgen05_commit(mbar_addr_mma_stage);
            }

            // Signal epilogue to start
            tcgen05_commit(mbar_addr_epi);
        }
        // All warps aside from the two for TMA/MMA are dedicated to the epilogue
        else if (warp_id < NUM_WARPS - 2) {
            mbar_wait(mbar_addr_epi, 0);
            asm volatile("tcgen05.fence::after_thread_sync;"); // ISSUE: Verify we need this

            // Load MMA into regs (TMEM -> Regs)
            // Each warp loads 16 rows per tcgen05_ld and we have 128 rows, with 4 warps each one is responsible for 32 rows
            // Each thread
            float results[TD_MMA_N / 2];
            int rows_per_warp = TD_MMA_M / (NUM_WARPS - 2);
            for (int sub_m = 0; sub_m < rows_per_warp / 16; sub_m++) {
                if (m_off + warp_id * rows_per_warp + sub_m * 16 > groups[group].M) {
                    break;
                }
                if constexpr (TD_MMA_N == 256) {
                    tcgen05_ld<16, 256, 32>(results, tmem_addr_result + (((warp_id * rows_per_warp) + sub_m * 16) << 16));
                }
                else if constexpr (TD_MMA_N == 128) {
                    tcgen05_ld<16, 256, 16>(results, tmem_addr_result + (((warp_id * rows_per_warp) + sub_m * 16) << 16));
                }
                else if constexpr (TD_MMA_N == 64) {
                    tcgen05_ld<16, 256, 8>(results, tmem_addr_result + (((warp_id * rows_per_warp) + sub_m * 16) << 16));
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;");

                // Post process and store from Regs to SMEM (Regs -> SMEM)
                // Transfer result from SMEM -> GMEM (8 comes from 256/32 -> 256b per ld block from above)
                for (int i = 0; i < TD_MMA_N / 8; i++) {
                    const int m_offset = m_off + warp_id * rows_per_warp + sub_m * 16 + lane_id / 4;
                    const int n_offset = n_off + i * 8 + (lane_id % 4) * 2;
                    if (m_offset < groups[group].M) {
                        reinterpret_cast<half2 *>(groups[group].C_addr + (m_offset)*N + n_offset)[0] = __float22half2_rn({results[i * 4], results[i * 4 + 1]});
                    }
                    if (m_offset + 8 < groups[group].M) {
                        reinterpret_cast<half2 *>(groups[group].C_addr + (m_offset + 8)*N + n_offset)[0] = __float22half2_rn({results[i * 4 + 2], results[i * 4 + 3]});
                    }
                }
            }
        } 
        __syncthreads(); // Ensure all previous results have been read from TMEM and written to GMEM before starting computation on the next work tile
        // Reinitialize mbarriers for next tile
        if (warp_id == 0 && elect_one_sync()) {
            for (int i = 0; i < PIPE_STAGES * 2 + 1; i++) {
                mbar_init(mbar_addr_tma + i * 8, 1);
            }
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
        __syncthreads();
    }
    // Free memory
    //asm volatile("bar.sync 1, %0;" :: "r"(WARP_SIZE*(NUM_WARPS - 2)) : "memory");
    __syncthreads();
    if (warp_id == 0) { tcgen05_dealloc_tmem<1>(tmem_addr_result, TD_MMA_N * 2); }
}

static GroupDesc* h_groups;
static GroupDesc* d_groups;
static CUtensorMap* d_tmaps;
static bool allocated = false;
static bool attr_set = false;
static PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled_fn;
static CUtensorMap tmap_a_temp, tmap_b_temp;
static int cache_N = 0, cache_K = 0;

void nvfp4_group_gemm(const std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>& abc_tensors, const std::vector<std::tuple<torch::Tensor, torch::Tensor>>& sf_tensors, const std::vector<std::tuple<int, int, int, int>>& prob_sizes, int N, int K, int G) {

    // Constants
    constexpr int M_TILE_SIZE = 128;
    constexpr int K_MMA_SIZE = 64;
    constexpr int NUM_WARPS = 6;
    constexpr int MAX_G = 8; // Maximum expected group size for a cross-call persistent kernel

    // Configurables
    constexpr int N_TILE_SIZE = 128;
    constexpr int K_TILE_SIZE = 256;
    constexpr bool SWIZZLE = true;
    constexpr int PIPE_STAGES = 5;
    constexpr int NUM_CTAS = 148;

    constexpr CUtensorMapSwizzle SWIZZLE_TYPE = SWIZZLE ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;

    // Query driver for tmap templates. Each CTA needs to fill in tmap_a_temp with the correct pointer / M value, and each tmap_b_temp with the correct pointer
    //CUtensorMap tmap_a_temp, tmap_b_temp;
    //auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    //tma_3d_map_ab<M_TILE_SIZE, K_TILE_SIZE, SWIZZLE_TYPE>::init(cuTensorMapEncodeTiled, &tmap_a_temp, nullptr, M_TILE_SIZE, K);
    //tma_3d_map_ab<N_TILE_SIZE, K_TILE_SIZE, SWIZZLE_TYPE>::init(cuTensorMapEncodeTiled, &tmap_b_temp, nullptr, N, K);
    if (N != cache_N || K != cache_K) {
        cuTensorMapEncodeTiled_fn = get_cuTensorMapEncodeTiled();
        tma_3d_map_ab<M_TILE_SIZE, K_TILE_SIZE, SWIZZLE_TYPE>::init(cuTensorMapEncodeTiled_fn, &tmap_a_temp, nullptr, M_TILE_SIZE, K);
        tma_3d_map_ab<N_TILE_SIZE, K_TILE_SIZE, SWIZZLE_TYPE>::init(cuTensorMapEncodeTiled_fn, &tmap_b_temp, nullptr, N, K);
        cache_N = N;
        cache_K = K;
    }


    if (!allocated) {
        h_groups = (GroupDesc*) malloc(MAX_G * sizeof(GroupDesc));
        cudaMalloc(&d_tmaps, 2 * NUM_CTAS * sizeof(CUtensorMap));
        cudaMalloc(&d_groups, MAX_G * sizeof(GroupDesc));
        allocated = true;
    }

    const int N_blocks = CEIL_DIV(N, N_TILE_SIZE);
    int total_tiles = 0;
    for (int i = 0; i < G; i++) {
        h_groups[i].A_addr     = std::get<0>(abc_tensors[i]).data_ptr();
        h_groups[i].B_addr     = std::get<1>(abc_tensors[i]).data_ptr();
        h_groups[i].C_addr     = reinterpret_cast<__half*>(std::get<2>(abc_tensors[i]).data_ptr());
        h_groups[i].sfa_addr   = reinterpret_cast<uint8_t*>(std::get<0>(sf_tensors[i]).data_ptr());
        h_groups[i].sfb_addr   = reinterpret_cast<uint8_t*>(std::get<1>(sf_tensors[i]).data_ptr());

        int M_size = std::get<0>(prob_sizes[i]);
        h_groups[i].M          = M_size;
        h_groups[i].block_start = total_tiles;
        total_tiles += CEIL_DIV(M_size, M_TILE_SIZE) * N_blocks;
    }

    cudaMemcpy(d_groups, h_groups, G * sizeof(GroupDesc), cudaMemcpyHostToDevice);

    // Ensure writes are visible (WC memory flushes on kernel launch anyway)
    //__asm__ __volatile__("sfence" ::: "memory"); Might be needed, but maybe not

    // === KERNEL LAUNCH ===
    auto kernel_inst = nvfp4_group_gemm_kernel<M_TILE_SIZE, N_TILE_SIZE, M_TILE_SIZE, N_TILE_SIZE, K_TILE_SIZE, M_TILE_SIZE, N_TILE_SIZE, K_MMA_SIZE, SWIZZLE, PIPE_STAGES, NUM_WARPS>;

    if (!attr_set) {
        cudaFuncSetAttribute(
            kernel_inst,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared  // Maximum shared memory
        );
        attr_set = true;
    }

    constexpr int threads = WARP_SIZE * NUM_WARPS;
    kernel_inst<<<NUM_CTAS, threads>>>(d_groups, tmap_a_temp, tmap_b_temp, total_tiles, d_tmaps, N, K, G);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

"""

nvfp4_group_gemm_cpp_source = """

#include <torch/extension.h>

void nvfp4_group_gemm(const std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>& abc_tensors, const std::vector<std::tuple<torch::Tensor, torch::Tensor>>& sf_tensors, const std::vector<std::tuple<int, int, int, int>>& prob_sizes, int N, int K, int G);

"""




nvfp4_group_gemm_module = load_inline(
    name='nvfp4_group_gemm',
    cpp_sources=nvfp4_group_gemm_cpp_source,
    cuda_sources=nvfp4_group_gemm_cuda_source,
    functions=['nvfp4_group_gemm'],
    verbose=True,
    extra_cuda_cflags=[
        '-I/usr/local/lib/python3.12/site-packages/cutlass_library/source/include',
        '-I/usr/local/lib/python3.12/site-packages/cutlass_library/source/tools/util/include',
        '-O3',
        '-gencode=arch=compute_100a,code=sm_100a',
        '-Xptxas', '--allow-expensive-optimizations=true',
        '--use_fast_math',
    ],
)


def custom_kernel(data: input_t) -> output_t:
    import time

    abc_tensors, _, sfasfb_tensors_reordered, problem_sizes = data

    """
    abc_tensors: [(a_ref, b_ref, c_ref), (a_ref, b_ref, c_ref), ...]
    sfasfb_tensors_reordered: [(sfa, sfb), (sfa, sfb), ...]
    problem_sizes (l is always 1): [(m, n, k, l), (m, n, k, l), ...]
    """

    G = len(abc_tensors)

    N = problem_sizes[0][1]
    K = problem_sizes[0][2]
    # nvfp4_group_gemm_module.nvfp4_group_gemm(A_ptrs, B_ptrs, C_ptrs, SFA_ptrs, SFB_ptrs, M_sizes, N, K, G)
    nvfp4_group_gemm_module.nvfp4_group_gemm(abc_tensors, sfasfb_tensors_reordered, problem_sizes, N, K, G)

    return [c for (a, b, c) in abc_tensors]




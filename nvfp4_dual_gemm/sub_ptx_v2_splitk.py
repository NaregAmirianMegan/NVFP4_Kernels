#!POPCORN leaderboard nvfp4_dual_gemm
import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t
from utils import make_match_reference

"""
2SM Split-K with Distributed Shared Memory (DSMEM)
"""

nvfp4_dual_gemm_cuda_source = """

#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda_fp16.h>
#include <math.h>

/*
    Notes:
    - TD stands for Tile Dimension
    - Split-K version divides the K dimension across multiple CTAs in a cluster
    - Uses DSMEM (Distributed Shared Memory) for partial sum reduction

    Assumptions:
    1) Problem shape is divisible by CTA and SMEM Tile shapes (no tail cases)
    2) We assume TD_SMEM_M/N == TD_MMA_M/N, since TMEM can only store one MMA tile worth of results at a time
    3) K dimension is divisible by SPLIT_K * K_TILE_SIZE
*/

enum class CacheHintSm100 : uint64_t {
    EVICT_NORMAL = 0x1000000000000000,
    EVICT_FIRST  = 0x12F0000000000000,
    EVICT_LAST   = 0x14F0000000000000,
};

template<int CTA_GROUP>
__device__ void inline tcgen05_commit(const int mbar_addr) {
    constexpr int16_t cta_mask = (0x1 << CTA_GROUP) - 1;
    asm volatile(
        "tcgen05.commit.cta_group::%1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %2;"
        :
        : "r"(mbar_addr), "n"(CTA_GROUP), "h"(cta_mask)
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
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
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

// Map shared memory address from another CTA in the cluster
__device__ int inline mapa_shared_cluster(int smem_addr, int cta_id) {
    int result;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(result)
        : "r"(smem_addr), "r"(cta_id)
    );
    return result;
}

// Get cluster CTA ID (linear rank within cluster)
__device__ int inline get_cluster_ctaid() {
    int cta_rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    return cta_rank;
}

// Get cluster dimensions
__device__ void inline get_cluster_dims(int &dim_x, int &dim_y) {
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(dim_x));
    asm volatile("mov.u32 %0, %%cluster_nctaid.y;" : "=r"(dim_y));
}

// Get CTA position within cluster
__device__ void inline get_cluster_ctapos(int &pos_x, int &pos_y) {
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(pos_x));
    asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(pos_y));
}

// Cluster barrier arrive
__device__ void inline cluster_arrive() {
    asm volatile("barrier.cluster.arrive.release.aligned;");
}

// Cluster barrier wait
__device__ void inline cluster_wait() {
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

// Combined cluster barrier
__device__ void inline cluster_sync() {
    cluster_arrive();
    cluster_wait();
}

template<int CTA_GROUP>
__device__ void inline tcgen05_2dtma_g2s_sf(int dst_smem, const void *tmap_ptr, int k_off, int m_block_off, int mbar_addr, CacheHintSm100 cache_policy) {
    asm volatile (
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%0.L2::cache_hint [%1], [%2, {%3, %4}], [%5], %6;"
        :
        : "n"(CTA_GROUP), "r"(dst_smem), "l"(tmap_ptr), "r"(k_off), "r"(m_block_off), "r"(mbar_addr), "l"(cache_policy)
    );
}

template<int CTA_GROUP>
__device__ void inline tcgen05_3dtma_g2s_sf(int dst_smem, const void *tmap_ptr, int k_off, int m_block_off, int mbar_addr, CacheHintSm100 cache_policy) {
    const int k_off_blocks = k_off / 256;
    asm volatile (
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::%0.L2::cache_hint [%1], [%2, {%3, %4, %5}], [%6], %7;"
        :
        : "n"(CTA_GROUP), "r"(dst_smem), "l"(tmap_ptr), "r"(0), "r"(k_off_blocks), "r"(m_block_off), "r"(mbar_addr), "l"(cache_policy)
    );
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
void tma_2d_map_sf(PFN_cuTensorMapEncodeTiled_v12000 cuTensorMapEncodeTiled, CUtensorMap* tmap, void* ptr, uint64_t mn_dim_gmem, uint64_t k_dim_gmem) {
    constexpr uint32_t rank = 2;
    constexpr int K_SMEM_TD_SF = K_SMEM_TD / 16;
    const int k_dim_gmem_sf = k_dim_gmem / 16;
    uint64_t dim_gmem[rank] = {512ULL * (k_dim_gmem_sf / 4), mn_dim_gmem / 128};
    uint64_t stride_gmem[rank - 1] = {512ULL * (k_dim_gmem_sf / 4)};
    uint32_t dim_smem[rank] = {512ULL * (K_SMEM_TD_SF / 4), 1};
    uint32_t elem_stride[rank] = {1, 1};

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
        "setp.ne.b32 p, %4, 0;\\n" // ISSUE: Is this inefficient, executing a comparison instr before every mma?
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

__device__ float inline silu(float x) {
    return x / (1.0f + expf(-x));
}

constexpr int WARP_SIZE = 32;
constexpr int SF_BLOCK_SIZE = 16;

/*
    Split-K version with DSMEM:
    - Cluster shape is (CTA_GROUP, SPLIT_K, 1)
    - CTA_GROUP CTAs cooperate for the dual-CTA MMA pattern (X dimension)
    - SPLIT_K CTAs partition the K dimension (Y dimension)
    - Each split-k partition computes a partial sum
    - Partial sums are stored to SMEM, then reduced via DSMEM

    Warp 0 will be responsible for all single thread (async) issued instructions, warp 1 or all CTA threads will handle the rest
    We need 1 warp to do the computation and TMA transfers. We need 4 warps in order to read/write all of TMEM (a single warp can only
    access 32 lanes (rows) in TMEM out of the total 128 per SM)
*/
template<int TD_CTA_M, int TD_CTA_N,
         int TD_SMEM_M, int TD_SMEM_N, int TD_SMEM_K,
         int TD_MMA_M, int TD_MMA_N, int TD_MMA_K, bool SWIZZLE, int PIPE_STAGES, int NUM_WARPS, int CTA_GROUP, int SPLIT_K>
__global__ void __cluster_dims__(CTA_GROUP, SPLIT_K, 1) nvfp4_dual_gemm_splitk_kernel(__half* __restrict__ c_ref, const int M, const int N, const int K, const __grid_constant__ CUtensorMap tmap_a,
                                  const __grid_constant__ CUtensorMap tmap_b1, const __grid_constant__ CUtensorMap tmap_b2, const __grid_constant__ CUtensorMap tmap_sfa,
                                  const __grid_constant__ CUtensorMap tmap_sfb1, const __grid_constant__ CUtensorMap tmap_sfb2) {

    // Statically computed values
    constexpr int WIDTH_COREMAT = SWIZZLE ? 256 : 32;
    constexpr int SFA_SMEM_TILESZ = 128 * (TD_SMEM_K / SF_BLOCK_SIZE); // 2KB
    constexpr int SFB1_SMEM_TILESZ = 128 * (TD_SMEM_K / SF_BLOCK_SIZE); // 2KB
    constexpr int SFB2_SMEM_TILESZ = 128 * (TD_SMEM_K / SF_BLOCK_SIZE); // 2KB
    constexpr int A_SMEM_TILESZ = TD_SMEM_M * (TD_SMEM_K / 2); // 16KB
    constexpr int B1_SMEM_TILESZ = (TD_SMEM_N / CTA_GROUP) * (TD_SMEM_K / 2); // 4KB
    constexpr int B2_SMEM_TILESZ = (TD_SMEM_N / CTA_GROUP) * (TD_SMEM_K / 2); // 4KB
    constexpr int SMEM_TILE_SZ = A_SMEM_TILESZ + B1_SMEM_TILESZ + B2_SMEM_TILESZ + SFA_SMEM_TILESZ + SFB1_SMEM_TILESZ + SFB2_SMEM_TILESZ;

    // Partial sum buffer size: TD_MMA_M * TD_MMA_N floats per result, 2 results (for dual gemm)
    constexpr int PARTIAL_SUM_ELEMS = TD_MMA_M * TD_MMA_N;
    constexpr int PARTIAL_SUM_SIZE = PARTIAL_SUM_ELEMS * sizeof(float);

    // Calculate constants, offsets, etc... for this thread/warp
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int m_off = blockIdx.x * TD_CTA_M;
    const int n_off = blockIdx.y * TD_CTA_N;

    // Get cluster position
    int cta_x, cta_y;  // cta_x is for CTA_GROUP, cta_y is for SPLIT_K
    get_cluster_ctapos(cta_x, cta_y);
    int cta_rank = get_cluster_ctaid();

    // Calculate K range for this split-k partition
    const int k_per_split = K / SPLIT_K;
    const int k_start = cta_y * k_per_split;
    const int k_end = k_start + k_per_split;

    // Allocate SMEM buffers
    __shared__ alignas(128) char a_smem[PIPE_STAGES * A_SMEM_TILESZ];
    __shared__ alignas(128) char sfa_smem[PIPE_STAGES * SFA_SMEM_TILESZ];
    __shared__ alignas(128) char b1_smem[PIPE_STAGES * B1_SMEM_TILESZ];
    __shared__ alignas(128) char sfb1_smem[PIPE_STAGES * SFB1_SMEM_TILESZ];
    __shared__ alignas(128) char b2_smem[PIPE_STAGES * B2_SMEM_TILESZ];
    __shared__ alignas(128) char sfb2_smem[PIPE_STAGES * SFB2_SMEM_TILESZ];

    // Partial sum buffers in SMEM for DSMEM reduction
    __shared__ alignas(128) float partial_sum_1[PARTIAL_SUM_ELEMS];
    __shared__ alignas(128) float partial_sum_2[PARTIAL_SUM_ELEMS];

    // Convert ptrs to properly pass to inline PTX
    const int a_smem_ptr = static_cast<int>(__cvta_generic_to_shared(a_smem));
    const int sfa_smem_ptr = static_cast<int>(__cvta_generic_to_shared(sfa_smem));
    const int b1_smem_ptr = static_cast<int>(__cvta_generic_to_shared(b1_smem));
    const int sfb1_smem_ptr = static_cast<int>(__cvta_generic_to_shared(sfb1_smem));
    const int b2_smem_ptr = static_cast<int>(__cvta_generic_to_shared(b2_smem));
    const int sfb2_smem_ptr = static_cast<int>(__cvta_generic_to_shared(sfb2_smem));
    const int partial_sum_1_ptr = static_cast<int>(__cvta_generic_to_shared(partial_sum_1));
    const int partial_sum_2_ptr = static_cast<int>(__cvta_generic_to_shared(partial_sum_2));

    // Allocate TMEM buffers
    __shared__ int tmem_addr_ptr[1];

    // Setup memory barriers
    __shared__ alignas(8) int64_t mbars[PIPE_STAGES * 2 + 1];
    const int mbar_addr_tma = static_cast<int>(__cvta_generic_to_shared(mbars));
    const int mbar_addr_mma = mbar_addr_tma + PIPE_STAGES * 8;
    const int mbar_addr_epi = mbar_addr_mma + PIPE_STAGES * 8;

    if (warp_id == 0 && elect_one_sync()) {
        for (int i = 0; i < PIPE_STAGES; i++) {
            mbar_init(mbar_addr_tma + i * 8, CTA_GROUP);
            mbar_init(mbar_addr_mma + i * 8, 1);
        }
        mbar_init(mbar_addr_epi, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp_id == 1) {
        tcgen05_alloc_tmem<CTA_GROUP>(&tmem_addr_ptr[0], TD_MMA_N*4);
    }

    // Sync all threads in cluster
    cluster_sync();

    const int tmem_addr_result_1 = tmem_addr_ptr[0];
    const int tmem_addr_result_2 = tmem_addr_result_1 + TD_MMA_N;
    const int tmem_addr_sfa = tmem_addr_result_2 + TD_MMA_N;
    const int tmem_addr_sfb1 = tmem_addr_sfa + 4 * (TD_SMEM_K/64);
    const int tmem_addr_sfb2 = tmem_addr_sfb1 + 4 * (TD_SMEM_K/64);

    constexpr int TMA_WARP = NUM_WARPS - 2;
    constexpr int MMA_WARP = NUM_WARPS - 1;

    int phase = 0;

    // TMA thread loops over SMEM tile stages and loads from GMEM->SMEM
    if (warp_id == TMA_WARP && elect_one_sync()) {
        auto tma_load_stage = [&](const int k_off) {
            const int stage = ((k_off - k_start) / TD_SMEM_K) % PIPE_STAGES;
            mbar_wait(mbar_addr_mma + stage * 8, phase ^ 1);

            if (stage == PIPE_STAGES - 1) {
                phase ^= 1;
            }

            const int k_off_coremat = k_off / WIDTH_COREMAT;
            const int a_smem_stage_ptr = a_smem_ptr + stage * A_SMEM_TILESZ;
            const int b1_smem_stage_ptr = b1_smem_ptr + stage * B1_SMEM_TILESZ;
            const int b2_smem_stage_ptr = b2_smem_ptr + stage * B2_SMEM_TILESZ;
            const int sfa_smem_stage_ptr = sfa_smem_ptr + stage * SFA_SMEM_TILESZ;
            const int sfb1_smem_stage_ptr = sfb1_smem_ptr + stage * SFB1_SMEM_TILESZ;
            const int sfb2_smem_stage_ptr = sfb2_smem_ptr + stage * SFB2_SMEM_TILESZ;
            const int mbar_addr_tma_stage = (mbar_addr_tma + stage * 8) & 0xFEFFFFFF;

            tcgen05_3dtma_g2s_ab<CTA_GROUP>(a_smem_stage_ptr, &tmap_a, m_off, k_off_coremat, mbar_addr_tma_stage, CacheHintSm100::EVICT_LAST);
            tcgen05_3dtma_g2s_ab<CTA_GROUP>(b1_smem_stage_ptr, &tmap_b1, n_off + cta_x * (TD_MMA_N / 2), k_off_coremat, mbar_addr_tma_stage, CacheHintSm100::EVICT_FIRST);
            tcgen05_3dtma_g2s_ab<CTA_GROUP>(b2_smem_stage_ptr, &tmap_b2, n_off + cta_x * (TD_MMA_N / 2), k_off_coremat, mbar_addr_tma_stage, CacheHintSm100::EVICT_FIRST);

            const int k_off_sf = 512 * ((k_off / 16) / 4);
            const int m_block_off = m_off / 128;
            const int n_block_off = n_off / 128;
            tcgen05_3dtma_g2s_sf<CTA_GROUP>(sfa_smem_stage_ptr, &tmap_sfa, k_off_sf, m_block_off, mbar_addr_tma_stage, CacheHintSm100::EVICT_LAST);
            tcgen05_3dtma_g2s_sf<CTA_GROUP>(sfb1_smem_stage_ptr, &tmap_sfb1, k_off_sf, n_block_off, mbar_addr_tma_stage, CacheHintSm100::EVICT_FIRST);
            tcgen05_3dtma_g2s_sf<CTA_GROUP>(sfb2_smem_stage_ptr, &tmap_sfb2, k_off_sf, n_block_off, mbar_addr_tma_stage, CacheHintSm100::EVICT_FIRST);

            mbar_arrive_expect(mbar_addr_tma_stage, SMEM_TILE_SZ);
        };

        // Each split-k partition processes its portion of K
        for (int k_off = k_start; k_off < k_end; k_off += TD_SMEM_K) {
            tma_load_stage(k_off);
        }
    }
    // MMA thread loops over SMEM tile stages, loads TMEM and computes MMA ops
    else if (cta_x == 0 && warp_id == MMA_WARP && elect_one_sync()) {
        for (int k_off = k_start; k_off < k_end; k_off += TD_SMEM_K) {
            const int stage = ((k_off - k_start) / TD_SMEM_K) % PIPE_STAGES;
            mbar_wait(mbar_addr_tma + stage * 8, phase);
            asm volatile("tcgen05.fence::after_thread_sync;");

            if (stage == PIPE_STAGES - 1) {
                phase ^= 1;
            }

            const int a_smem_stage_ptr = a_smem_ptr + stage * A_SMEM_TILESZ;
            const int b1_smem_stage_ptr = b1_smem_ptr + stage * B1_SMEM_TILESZ;
            const int b2_smem_stage_ptr = b2_smem_ptr + stage * B2_SMEM_TILESZ;
            const int sfa_smem_stage_ptr = sfa_smem_ptr + stage * SFA_SMEM_TILESZ;
            const int sfb1_smem_stage_ptr = sfb1_smem_ptr + stage * SFB1_SMEM_TILESZ;
            const int sfb2_smem_stage_ptr = sfb2_smem_ptr + stage * SFB2_SMEM_TILESZ;
            const int mbar_addr_mma_stage = mbar_addr_mma + stage * 8;

            // Load scale factors SMEM -> TMEM
            for (int sub_k_iter = 0; sub_k_iter < TD_SMEM_K / TD_MMA_K; sub_k_iter++) {
                uint64_t sfa_desc = make_smem_desc<0, false>(sfa_smem_stage_ptr + (sub_k_iter * 512));
                uint64_t sfb1_desc = make_smem_desc<0, false>(sfb1_smem_stage_ptr + (sub_k_iter * 512));
                uint64_t sfb2_desc = make_smem_desc<0, false>(sfb2_smem_stage_ptr + (sub_k_iter * 512));
                tcgen05_cp<CTA_GROUP>(tmem_addr_sfa + 4 * sub_k_iter, sfa_desc);
                tcgen05_cp<CTA_GROUP>(tmem_addr_sfb1 + 4 * sub_k_iter, sfb1_desc);
                tcgen05_cp<CTA_GROUP>(tmem_addr_sfb2 + 4 * sub_k_iter, sfb2_desc);
            }

            // Loop over SMEM tile K-dim
            for (int sub_k_iter = 0; sub_k_iter < TD_SMEM_K / TD_MMA_K; sub_k_iter++) {
                uint64_t a_desc, b1_desc, b2_desc;
                if constexpr (SWIZZLE) {
                    a_desc = make_smem_desc<TD_MMA_M, SWIZZLE>(a_smem_stage_ptr + sub_k_iter * 32);
                    b1_desc = make_smem_desc<TD_MMA_N, SWIZZLE>(b1_smem_stage_ptr + sub_k_iter * 32);
                    b2_desc = make_smem_desc<TD_MMA_N, SWIZZLE>(b2_smem_stage_ptr + sub_k_iter * 32);
                }
                else {
                    a_desc = make_smem_desc<TD_MMA_M, SWIZZLE>(a_smem_stage_ptr + sub_k_iter * TD_MMA_K * TD_MMA_M / 2);
                    b1_desc = make_smem_desc<TD_MMA_N, SWIZZLE>(b1_smem_stage_ptr + sub_k_iter * TD_MMA_K * TD_MMA_N / 2);
                    b2_desc = make_smem_desc<TD_MMA_N, SWIZZLE>(b2_smem_stage_ptr + sub_k_iter * TD_MMA_K * TD_MMA_N / 2);
                }

                int sfa_tmem = tmem_addr_sfa + 4 * sub_k_iter;
                int sfb1_tmem = tmem_addr_sfb1 + 4 * sub_k_iter + (n_off%128)/32;
                int sfb2_tmem = tmem_addr_sfb2 + 4 * sub_k_iter + (n_off%128)/32;
                // enable_input_d: use accumulated result if not the first iteration of this split
                int enable_d = (k_off - k_start) + sub_k_iter;
                tcgen05_mma_nvfp4<CTA_GROUP>(tmem_addr_result_1, a_desc, b1_desc, make_instr_desc<TD_MMA_M*CTA_GROUP, TD_MMA_N>(), sfa_tmem, sfb1_tmem, enable_d);
                tcgen05_mma_nvfp4<CTA_GROUP>(tmem_addr_result_2, a_desc, b2_desc, make_instr_desc<TD_MMA_M*CTA_GROUP, TD_MMA_N>(), sfa_tmem, sfb2_tmem, enable_d);
            }
            // signal MMA done
            tcgen05_commit<CTA_GROUP>(mbar_addr_mma_stage);
        }

        // Signal epilogue to start
        tcgen05_commit<CTA_GROUP>(mbar_addr_epi);
    }
    // =========================================================================
    // EPILOGUE PHASE - All threads must participate in barriers
    // =========================================================================

    // Epilogue warps (0-3) wait for MMA completion and load partial results to SMEM
    if (warp_id < NUM_WARPS - 2) {
        mbar_wait(mbar_addr_epi, 0);
        asm volatile("tcgen05.fence::after_thread_sync;");

        // Load MMA results from TMEM to registers
        float results_1[TD_MMA_N/2];
        float results_2[TD_MMA_N/2];
        int rows_per_warp = TD_MMA_M / (NUM_WARPS - 2);

        for (int sub_m = 0; sub_m < rows_per_warp / 16; sub_m++) {
            if constexpr (TD_MMA_N == 128) {
                tcgen05_ld<16, 256, 16>(results_1, tmem_addr_result_1 + (((cta_x * TD_MMA_M + warp_id * rows_per_warp) + sub_m * 16) << 16));
                tcgen05_ld<16, 256, 16>(results_2, tmem_addr_result_2 + (((cta_x * TD_MMA_M + warp_id * rows_per_warp) + sub_m * 16) << 16));
            }
            else if constexpr (TD_MMA_N == 64) {
                tcgen05_ld<16, 256, 8>(results_1, tmem_addr_result_1 + (((cta_x * TD_MMA_M + warp_id * rows_per_warp) + sub_m * 16) << 16));
                tcgen05_ld<16, 256, 8>(results_2, tmem_addr_result_2 + (((cta_x * TD_MMA_M + warp_id * rows_per_warp) + sub_m * 16) << 16));
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;");

            // Store partial sums to shared memory
            for (int i = 0; i < TD_MMA_N / 8; i++) {
                int m_local = (warp_id * rows_per_warp + sub_m * 16 + lane_id / 4);
                int n_local = i * 8 + (lane_id % 4) * 2;
                int idx = m_local * TD_MMA_N + n_local;

                partial_sum_1[idx] = results_1[i * 4];
                partial_sum_1[idx + 1] = results_1[i * 4 + 1];
                partial_sum_1[idx + TD_MMA_N * 8] = results_1[i * 4 + 2];
                partial_sum_1[idx + TD_MMA_N * 8 + 1] = results_1[i * 4 + 3];

                partial_sum_2[idx] = results_2[i * 4];
                partial_sum_2[idx + 1] = results_2[i * 4 + 1];
                partial_sum_2[idx + TD_MMA_N * 8] = results_2[i * 4 + 2];
                partial_sum_2[idx + TD_MMA_N * 8 + 1] = results_2[i * 4 + 3];
            }
        }
    }

    // ALL threads in CTA must hit this barrier (ensures partial sums written to SMEM)
    __syncthreads();

    // ALL CTAs in cluster must hit this barrier (ensures all split-k partitions ready)
    cluster_sync();

    // Reduce partial sums across split-k partitions using DSMEM
    // Only cta_y == 0 and epilogue warps do the reduction work
    if (cta_y == 0 && warp_id < NUM_WARPS - 2) {
        int rows_per_warp = TD_MMA_M / (NUM_WARPS - 2);

        // Each thread reduces its portion of the output
        int thread_elements = PARTIAL_SUM_ELEMS / (NUM_WARPS - 2) / WARP_SIZE;
        int thread_start = (warp_id * WARP_SIZE + lane_id) * thread_elements;

        for (int elem = 0; elem < thread_elements; elem++) {
            int idx = thread_start + elem;
            if (idx < PARTIAL_SUM_ELEMS) {
                float sum_1 = partial_sum_1[idx];
                float sum_2 = partial_sum_2[idx];

                // Read from other split-k partitions via DSMEM
                for (int sk = 1; sk < SPLIT_K; sk++) {
                    // Cluster layout is (CTA_GROUP, SPLIT_K, 1), so CTA at (cta_x, sk) has ID = sk * CTA_GROUP + cta_x
                    int peer_cta = sk * CTA_GROUP + cta_x;

                    // Map peer's shared memory addresses
                    int peer_partial_1_ptr = mapa_shared_cluster(partial_sum_1_ptr, peer_cta);
                    int peer_partial_2_ptr = mapa_shared_cluster(partial_sum_2_ptr, peer_cta);

                    // Load from peer's SMEM
                    float peer_val_1, peer_val_2;
                    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(peer_val_1) : "r"(peer_partial_1_ptr + idx * 4));
                    asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(peer_val_2) : "r"(peer_partial_2_ptr + idx * 4));

                    sum_1 += peer_val_1;
                    sum_2 += peer_val_2;
                }

                // Store reduced results back
                partial_sum_1[idx] = sum_1;
                partial_sum_2[idx] = sum_2;
            }
        }
    }

    // ALL threads sync before reading reduced values (needed for SMEM consistency)
    __syncthreads();

    // Apply SiLU activation and write to global memory (only cta_y == 0 epilogue warps)
    if (cta_y == 0 && warp_id < NUM_WARPS - 2) {
        int rows_per_warp = TD_MMA_M / (NUM_WARPS - 2);

        for (int sub_m = 0; sub_m < rows_per_warp / 16; sub_m++) {
            for (int i = 0; i < TD_MMA_N / 8; i++) {
                const int m_offset = m_off + warp_id * rows_per_warp + sub_m * 16 + lane_id / 4;
                const int n_offset = n_off + i * 8 + (lane_id % 4) * 2;

                int m_local = warp_id * rows_per_warp + sub_m * 16 + lane_id / 4;
                int n_local = i * 8 + (lane_id % 4) * 2;
                int idx = m_local * TD_MMA_N + n_local;

                float r1_0 = partial_sum_1[idx];
                float r1_1 = partial_sum_1[idx + 1];
                float r1_2 = partial_sum_1[idx + TD_MMA_N * 8];
                float r1_3 = partial_sum_1[idx + TD_MMA_N * 8 + 1];

                float r2_0 = partial_sum_2[idx];
                float r2_1 = partial_sum_2[idx + 1];
                float r2_2 = partial_sum_2[idx + TD_MMA_N * 8];
                float r2_3 = partial_sum_2[idx + TD_MMA_N * 8 + 1];

                float result[4];
                result[0] = silu(r1_0) * r2_0;
                result[1] = silu(r1_1) * r2_1;
                result[2] = silu(r1_2) * r2_2;
                result[3] = silu(r1_3) * r2_3;

                reinterpret_cast<half2 *>(c_ref + (m_offset)*N + n_offset)[0] = __float22half2_rn({result[0], result[1]});
                reinterpret_cast<half2 *>(c_ref + (m_offset + 8)*N + n_offset)[0] = __float22half2_rn({result[2], result[3]});
            }
        }
    }

    // Free TMEM - only warp 0 deallocates
    if (warp_id == 0) {
        tcgen05_dealloc_tmem<CTA_GROUP>(tmem_addr_result_1, TD_MMA_N * 4);
    }
}

torch::Tensor nvfp4_dual_gemm(torch::Tensor a_ref, torch::Tensor b1_ref, torch::Tensor b2_ref, torch::Tensor sfa_ref, torch::Tensor sfb1_ref, torch::Tensor sfb2_ref, torch::Tensor c_ref, int M, int N, int K) {

    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

    constexpr int M_TILE_SIZE = 128;
    constexpr int N_TILE_SIZE = 64;
    constexpr int K_TILE_SIZE = 256;
    constexpr int K_MMA_SIZE = 64;
    constexpr bool SWIZZLE = true;
    constexpr int PIPE_STAGES = 3;
    constexpr int NUM_WARPS = 6;
    constexpr int CTA_GROUP = 2;
    constexpr int SPLIT_K = 2;  // Split K dimension across 2 CTAs

    CUtensorMap tmap_a, tmap_b1, tmap_b2, tmap_sfa, tmap_sfb1, tmap_sfb2;
    constexpr CUtensorMapSwizzle SWIZZLE_TYPE = SWIZZLE ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B : CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    tma_3d_map_ab<M_TILE_SIZE, K_TILE_SIZE, SWIZZLE_TYPE>::init(cuTensorMapEncodeTiled, &tmap_a, a_ref.data_ptr(), M, K);
    tma_3d_map_ab<N_TILE_SIZE/CTA_GROUP, K_TILE_SIZE, SWIZZLE_TYPE>::init(cuTensorMapEncodeTiled, &tmap_b1, b1_ref.data_ptr(), N, K);
    tma_3d_map_ab<N_TILE_SIZE/CTA_GROUP, K_TILE_SIZE, SWIZZLE_TYPE>::init(cuTensorMapEncodeTiled, &tmap_b2, b2_ref.data_ptr(), N, K);

    tma_3d_map_sf<K_TILE_SIZE>(cuTensorMapEncodeTiled, &tmap_sfa, sfa_ref.data_ptr(), M, K);
    tma_3d_map_sf<K_TILE_SIZE>(cuTensorMapEncodeTiled, &tmap_sfb1, sfb1_ref.data_ptr(), N, K);
    tma_3d_map_sf<K_TILE_SIZE>(cuTensorMapEncodeTiled, &tmap_sfb2, sfb2_ref.data_ptr(), N, K);


    auto kernel_inst = nvfp4_dual_gemm_splitk_kernel<M_TILE_SIZE, N_TILE_SIZE, M_TILE_SIZE, N_TILE_SIZE, K_TILE_SIZE, M_TILE_SIZE, N_TILE_SIZE, K_MMA_SIZE, SWIZZLE, PIPE_STAGES, NUM_WARPS, CTA_GROUP, SPLIT_K>;

    cudaFuncSetAttribute(
        kernel_inst,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared
    );

    constexpr int threads = WARP_SIZE * NUM_WARPS;
    // Grid: each (x,y) position maps to a cluster of (CTA_GROUP x SPLIT_K) CTAs
    dim3 grid_dim(M/M_TILE_SIZE, N/N_TILE_SIZE);
    kernel_inst<<<grid_dim, threads>>>(reinterpret_cast<__half*>(c_ref.data_ptr()), M, N, K, tmap_a, tmap_b1, tmap_b2, tmap_sfa, tmap_sfb1, tmap_sfb2);


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return c_ref;
}

"""

nvfp4_dual_gemm_cpp_source = """

#include <torch/extension.h>

torch::Tensor nvfp4_dual_gemm(torch::Tensor a_ref, torch::Tensor b1_ref, torch::Tensor b2_ref, torch::Tensor sfa_ref, torch::Tensor sfb1_ref, torch::Tensor sfb2_ref, torch::Tensor c_ref, int M, int N, int K);

"""




nvfp4_dual_gemm_module = load_inline(
    name='nvfp4_dual_gemm_splitk',
    cpp_sources=nvfp4_dual_gemm_cpp_source,
    cuda_sources=nvfp4_dual_gemm_cuda_source,
    functions=['nvfp4_dual_gemm'],
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

def kernel(a_ref, b1_ref, b2_ref, sfa_ref, sfb1_ref, sfb2_ref, c_ref, M, N, K):
    return nvfp4_dual_gemm_module.nvfp4_dual_gemm(a_ref, b1_ref, b2_ref, sfa_ref, sfb1_ref, sfb2_ref, c_ref, M, N, K)


def custom_kernel(data: input_t) -> output_t:
    a_ref, b1_ref, b2_ref, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu, sfa_ref_permuted, sfb1_ref_permuted, sfb2_ref_permuted, c_ref = data

    M = a_ref.shape[0]
    N = b1_ref.shape[0]
    K = a_ref.shape[1]*2

    return kernel(a_ref, b1_ref, b2_ref, sfa_ref_permuted, sfb1_ref_permuted, sfb2_ref_permuted, c_ref, M, N, K)

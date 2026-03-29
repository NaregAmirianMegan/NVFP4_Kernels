#!POPCORN leaderboard nvfp4_gemm
import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t
from utils import make_match_reference

"""
Source: phase0/v0.py


--- Change Log ---
Implement basic GEMM using block-scaled mma operations in order to give us more
granular control over the algorithm and performance.


--- Implementation Details ---
Central to our kernel is the tcgen05 mma instruction. In particular the block-scaled
floating point version using nvfp4 values with e4m3 scalars. There are two variants
we can use:

tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
                                        [d-tmem],  a-desc,  b-desc, idesc,
                                        [scale-A-tmem], [scale-B-tmem], enable-input-d;

tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
                                        [d-tmem], [a-tmem], b-desc, idesc,
                                        [scale-A-tmem], [scale-B-tmem], enable-input-d;

The only difference is the A-block is stored in TMEM rather than SMEM. For this implementation
we choose the first variant. Let each mma instruction compute an mBxnB segment of C with a depth
of kB (so each instruction computes the block-scaled matmul of an mBxkB segment of A with a kBxnB
segment of B). The execution of a single one of these mma ops requires us to:

-> Load mBxkB block of A and kBxnB block of B into SMEM
-> Load mBx(kB/16) block of SFA and (kB/16)xnB block of SFB into TMEM
-> Execute mma and accumulate into resultant mBxnB region in TMEM

For this implementation we will have a single CTA compute each mBxnB block of output (so each CTA
computes along all of k). So for a single CTA this is the algorithm flow:

-> Allocate TMEM (need room for SFA-block, SFB-block, C-block)
-> For each k-block:
    -> Load SFA-block, SFB-block into SMEM
    -> Load A-block, B-block into SMEM
    -> Copy SFA-block, SFB-block into TMEM
    -> Execute mma
-> Move C-block into CTA regs
-> Move C-block from CTA regs to SMEM
-> Store C-block SMEM->GMEM
-> Deallocate TMEM

Below we discuss the specifics of data movement and formatting:

* SMEM data format:


* TMEM data format:



Now that we understand how data should be formatted, stored, and moved around we can write a more
detailed pseudo-code of our algorithm:

-> 


--- Analysis ---

 
"""

nvfp4_gemm_cuda_source = """

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include "cuda_runtime.h"

using namespace cute;

#define CUTLASS_CHECK(status) \
{ \
  cutlass::Status error = status; \
  if (error != cutlass::Status::kSuccess) { \
    std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
  } \
}

// A matrix configuration
using         ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 32;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::half_t;                                // Element type for D matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of D matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using MmaTileShape        = Shape<_128,_128,_64>;                          // MMA's tile size
using ClusterShape        = Shape<_1,_2,_1>;                                // Shape of the threadblocks in a cluster

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCount<21>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100                             // Kernel schedule policy. Auto or using targeted scheduling policy
  >::CollectiveOp;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementD, LayoutDTag, AlignmentD,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::TmaWarpSpecialized1SmNvf4                      // Epilogue schedule policy
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

torch::Tensor nvfp4_gemm(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, 
                                 torch::Tensor c_ref, int m, int n, int k) { 

    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using StrideD   = typename Gemm::GemmKernel::StrideD;

    using ArrayElementA = typename CollectiveMainloop::ArrayElementA;
    ArrayElementA* a_cutlass_ptr = reinterpret_cast<ArrayElementA*>(a_ref.data_ptr());

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

    using ArrayElementB = typename CollectiveMainloop::ArrayElementB;
    ArrayElementB* b_cutlass_ptr = reinterpret_cast<ArrayElementB*>(b_ref.data_ptr());

    // For SFA and SFB tensors layouts
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
    using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    LayoutSFA layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    LayoutSFB layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    using ElementSF = typename CollectiveMainloop::ElementSF;
    ElementSF* sfa_cutlass_ptr = reinterpret_cast<ElementSF*>(sfa_ref.data_ptr());
    ElementSF* sfb_cutlass_ptr = reinterpret_cast<ElementSF*>(sfb_ref.data_ptr());

    using ElementD = typename CollectiveEpilogue::ElementD;
    ElementD* d_cutlass_ptr = reinterpret_cast<ElementD*>(c_ref.data_ptr());

    Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        { // Problem Shape
            m, 
            n, 
            k, 
            1
        },
        { // Mainloop arguments
            a_cutlass_ptr, stride_a,
            b_cutlass_ptr, stride_b,
            sfa_cutlass_ptr, layout_sfa,
            sfb_cutlass_ptr, layout_sfb
        },
        { // Epilogue arguments 
            {
                1.0f, // alpha
                0.0f  // beta
            },
            nullptr, stride_d,
            d_cutlass_ptr, stride_d
        }
    };

    //arguments.scheduler.max_swizzle_size = options.swizzle;

    Gemm gemm;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run());

    cudaDeviceSynchronize();


    return c_ref; 
}

"""

nvfp4_gemm_cpp_source = """

#include <torch/extension.h>

torch::Tensor nvfp4_gemm(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, torch::Tensor c_ref, int m, int n, int k);

"""
















nvfp4_gemm_module = load_inline(
    name='nvfp4_gemm',
    cpp_sources=nvfp4_gemm_cpp_source,
    cuda_sources=nvfp4_gemm_cuda_source,
    functions=['nvfp4_gemm'],
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

def kernel(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, m, n, k):
    return nvfp4_gemm_module.nvfp4_gemm(a_ref, b_ref, sfa_ref, sfb_ref, c_ref, m, n, k)


def custom_kernel(data: input_t) -> output_t:
    a_ref, b_ref, sfa_ref, sfb_ref, sfa_ref_perm, sfb_ref_perm, c_ref = data

    """
    Input Info:
    
    a_ref:
    -> torch.float4_e2m1fn_x2
    -> (m, k//2, 1), stride: (k//2, 1, NA)

    sfa_ref:
    -> torch.float8_e4m3fn
    -> (m, k//16, 1), stride: (k//16, 1, NA)

    b_ref:
    -> torch.float4_e2m1fn_x2
    -> (n, k//2, 1), stride: (k//2, 1, NA)

    sfb_ref:
    -> torch.float8_e4m3fn
    -> (n, k//16, 1), stride: (k//16, 1, NA)

    c_ref:
    -> torch.float16
    -> (m, n, 1), stride: (n, 1, NA)
    
    """

    m = a_ref.shape[0]
    n = b_ref.shape[0]
    k = a_ref.shape[1]*2

    return kernel(a_ref, b_ref, sfa_ref_perm, sfb_ref_perm, c_ref, m, n, k)




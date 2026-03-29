#!POPCORN leaderboard nvfp4_gemm
import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t
from utils import make_match_reference

"""
Source: N/A


--- Change Log ---
N/A


--- Analysis ---

Benchmark Results:
k: 16384; l: 1; m: 128; n: 7168; seed: 1111
 ⏱ 74.8 ± 0.03 µs
 ⚡ 74.5 µs 🐌 75.0 µs

k: 7168; l: 1; m: 128; n: 4096; seed: 1111
 ⏱ 39.2 ± 0.01 µs
 ⚡ 39.1 µs 🐌 39.3 µs

k: 2048; l: 1; m: 128; n: 7168; seed: 1111
 ⏱ 30.8 ± 0.01 µs
 ⚡ 30.6 µs 🐌 30.9 µs

GEOM AVG: 44.866us

This routine just calls torch._scaled_mm on padded scale matrices and transposed b matrix.
Examining the run we see the core operation is done in the kernel:

cutlass3x_sm100_bstensorop_s256x128x64gemm_block_scaled_ue4m3xf4_ue4m3xf4_f32_f16_f16_256x128x256_0_tnn_align32_o_vs16_2sm_bias_f16_relu

which appears to be executing patterns that exploit Blackwell hardware (things like tensor memory, async transfers, tensor core mma),
but this kernel has a number of issues that are hard to address without implementing a version of the kernel in a language with more
granular control. The issues this kernel has include: poor hardware utilization (some SMs unused, low mem/compute throughput), lots of
long scoreboard stalls, syncs/branches/sleeps make up the majority of executed instructions (this doesn't make sense and is really bad 
for performance for a plethora of reasons), SMEM bank conflicts are non-trivially high, and overall this algorithm decomposes into multiple
kernel launches, which further decompose into multiple CTA launches. All of this is likely serialized, causing significant overhead induced
loss of performance. 



A more detailed walkthrough of what happens in this pytorch solution:

(Batch size is always 1 for this problem so the for loop is inconsequential)

1) Scale matrices processed with "to_blocked"
    1.a) 

2) "torch._scaled_mm" is called

  2.a) _scaled_mm_cuda (aten/src/ATen/native/cuda/ScaledBlas.cpp:649)
    - Entry point for CUDA dispatch
    - Creates empty output tensor (** OPT OP)
    - Calls _scaled_mm_out_cuda
  2.b) _scaled_mm_out_cuda (aten/src/ATen/native/cuda/ScaledBlas.cpp:463)
    - Validates device capabilities (requires SM 9.0+ or SM 8.9)
    - Checks matrix dimensions and dtypes
    - Determines scaling type via get_joint_scaling (line 488):
        - Checks candidate scaling pairs in priority order
      - For FP4 inputs with flattened FP8 scales, matches BlockWise1x16
      - Uses is_blockwise_1x16_scaling (ScaledBlas.cpp:150) which verifies:
      scale.numel() == round_up(m, 128) * round_up(ceil_div(k * 2, 16), 4)
    - Resizes output tensor (** OPT OP)
    - Calls _scaled_gemm
  2.c) _scaled_gemm (aten/src/ATen/native/cuda/ScaledBlas.cpp:365)
    - Sets up cublasCommonArgs with matrix pointers and scaling choices
    - Checks for ROCm tunable ops (disabled on CUDA)
    - Calls at::cuda::blas::scaled_gemm
  2.d) at::cuda::blas::scaled_gemm (aten/src/ATen/cuda/CUDABlas.cpp:1963)
    - Sets up cuBLASLt descriptors:
        - Compute type: CUBLAS_COMPUTE_32F
      - Scale type: CUDA_R_32F
    - Configures scale mode via get_scale_mode (CUDABlas.cpp:1896):
        - For BlockWise1x16 with Float8_e4m3fn scales:
      return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;  // Requires CUDA 12.8+
    - Sets matrix layouts (Adesc, Bdesc, Cdesc, Ddesc)
    - Sets scale pointers:
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER → scale_a
  CUBLASLT_MATMUL_DESC_B_SCALE_POINTER → scale_b
  CUBLASLT_MATMUL_DESC_A_SCALE_MODE → CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
  CUBLASLT_MATMUL_DESC_B_SCALE_MODE → CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
  2.e) cublasLtMatmulAlgoGetHeuristic (CUDABlas.cpp:2115)
    - Queries cuBLASLt for best algorithm given the problem configuration
    - Returns algorithm like: cutlass3x_sm100_bstensorop_s256x128x64gemm_block_scaled_ue4m3xf4_ue4m3xf4_f32_f16_f16_256x128x256_0_tnn_align32_o_vs16_2sm_bias_f16_relu
    - This is the kernel mentioned in your analysis comments!
  2.f) cublasLtMatmul (CUDABlas.cpp:2174)
    - Executes the actual GEMM on the GPU
    - Uses the heuristic algorithm selected above
    - Launches CUTLASS kernel with:
        - M=128, N=7168, K=16384 (in fp4 elements, 8192 bytes for K)
      - Blockwise 1x16 scaling (16 fp4 elements per scale factor)
      - FP8 e4m3 scale factors in blocked/swizzled layout

Two big takeaways here:
1) We are creating a temporary output storage tensor, then copying from that temp tensor into the actual c_ref output. This
   is just wasted time, we should be computing directly into c_ref.
2) What configurations is CuBLAS running with (i.e. what is the returned heuristic from cublasLtMatmulAlgoGetHeuristic and
   what algorithm does that result in? This will allow us to understand how to replicate and improve performance). A side note
   here is that since we know the benchmark shapes ahead of time we can further optimize by "pre-computing" the heuristic and
   just exectuting shape specific kernels, which should eliminate some runtime overhead, although this isn't very "real world"
   applicable.

To solve the first issue is fairly easy, we just strip out the pytorch middle man and use CuBLAS/CUTLASS directly, and pass in
c_ref as the output pointer.


Now for the second issue of what algorithm CuBLAS is actually running when cublasLtMatmul is called: 

Unfortunately, CuBLAS isn't open source, so the exact way in which CUTLASS is used is hidden. This forces an engineering
decision where we approach from the top-down or from the bottom up. In other words, we try to analyze what the CuBLAS auto
generated kernel is doing and try to emulate that, or approach the problem and build our own solution from first principles
using CUTLASS.

At the end of the day it's likely we will need to implement our own solution to improve upon CuBLAS, so we may as well just
immediately start from the ground up. This way we can also see the improvements each optimzation makes in real time as we 
develop. Now a second decision: CUTLASS C++ or CUTE DSL? CUTE DSL should offer the same level of granularity in development
without all of the syntactical mess of C++, so this should lead to faster development, we choose CUTE DSL.

"""

# Helper function for ceiling division
def ceil_div(a, b):
    return (a + b - 1) // b

# Helper function to convert scale factor tensor to blocked format
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape

    # Please ensure rows and cols are multiples of 128 and 4 respectively
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()

def custom_kernel(data: input_t) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMM.
    """
    a_ref, b_ref, sfa_ref_cpu, sfb_ref_cpu, _, _, c_ref = data
    
    # Get dimensions from MxNxL layout
    _, _, l = c_ref.shape

    # Call torch._scaled_mm to compute the GEMM result
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b = to_blocked(sfb_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
        res = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b.cuda(),
            bias=None,
            out_dtype=torch.float16,
        )
        c_ref[:, :, l_idx] = res
    return c_ref

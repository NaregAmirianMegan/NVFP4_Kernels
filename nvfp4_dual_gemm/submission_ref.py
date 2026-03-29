#!POPCORN leaderboard nvfp4_dual_gemm

# This is a submission template for popcorn leaderboard 'nvfp4_dual_gemm'.
# Your task is as follows:
# > 
# > You will implement a block scaled dual matrix-matrix multiplication kernel with silu activation optimized for NVIDIA B200.
# > To be explicit, you will be given a tuple of tensors:
# > ```
# > (a, b1, b2, sfa, sfb1, sfb2, c)
# > ```
# > where:
# > * `a` is M x K x L in K-major order in nvfp4(e2m1)
# > * `b1` is N x K x L in K-major order in nvfp4(e2m1)
# > * `b2` is N x K x L in K-major order in nvfp4(e2m1)
# > * `sfa` is M x (K // 16) x L in K-major order in fp8(e4m3fnuz)
# > * `sfb1` is N x (K // 16) x L in K-major order in fp8(e4m3fnuz)
# > * `sfb2` is N x (K // 16) x L in K-major order in fp8(e4m3fnuz)
# > * `c` is M x N x L in fp16
# > 
# > Matrix sizes `M` is divisible by mma_tiler_mn[0], `N` is divisible by mma_tiler_mn[1], `K` is divisible by 256.
# > The ranking criteria is the geometric mean of the benchmark results.
# > For the grand price, your kernel will be evaluated against the speed of light analysis
# > and the solution closest to the speed of light will be awarded the grand price.
# > ```
# > The speed of light analysis based on the max(FP4 Tensor Core math throughput, DRAM memory throughput) of B200 and tested under 1.5Ghz clock:
# >   M   N   K   L time[us] 
# > 256 4096 7168 1 4.708
# > 512 4096 7168 1 8.714
# > 256 3072 4096 1 2.125
# > 512 3072 7168 1 6.535
# > ```
# The deadline for this leaderboard is 2026-01-17 07:59:00+00:00

# You can automatically route this file to specific GPUs by adding a line
# `#!POPCORN gpus <GPUs>` to the header of this file.
# Happy hacking!

import torch
from task import input_t, output_t

# Scaling factor vector size
sf_vec_size = 16

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
    PyTorch reference implementation of NVFP4 block-scaled dual GEMM with silu activation,
    C = silu(A @ B1) * (A @ B2).
    """
    a_ref, b1_ref, b2_ref, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu, _, _, _, c_ref = data
    
    # Get dimensions from MxNxL layout
    m, n, l = c_ref.shape

    # Call torch._scaled_mm to compute the GEMV result
    ref1 = torch.empty(
        (l, m, n),
        dtype=torch.float32,
        device="cuda",
    ).permute(1, 2, 0)
    ref2 = torch.empty(
        (l, m, n),
        dtype=torch.float32,
        device="cuda",
    ).permute(1, 2, 0)
    for l_idx in range(l):
        # Convert the scale factor tensor to blocked format
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b1 = to_blocked(sfb1_ref_cpu[:, :, l_idx])
        scale_b2 = to_blocked(sfb2_ref_cpu[:, :, l_idx])
        # (m, k) @ (n, k).T -> (m, n)
        res1 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b1_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b1.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref1[:, :, l_idx] = res1

        res2 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b2_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b2.cuda(),
            bias=None,
            out_dtype=torch.float32,
        )
        ref2[:, :, l_idx] = res2
    # Do silu on the first GEMM result and multiply with the second GEMM result
    c_ref = (torch.nn.functional.silu(ref1) * ref2).to(torch.float16)
    return c_ref

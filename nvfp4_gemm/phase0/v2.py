#!POPCORN leaderboard nvfp4_gemm
import torch
from torch.utils.cpp_extension import load_inline
from typing import Type, Tuple, Union, List
from task import input_t, output_t
from utils import make_match_reference

import argparse

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack

"""
Source: N/A


--- Change Log ---
Implement basic GEMM using block-scaled mma operations in order to give us more
granular control over the algorithm and performance.


--- Implementation Details ---
Central to our kernel is the tcgen05 mma instruction. In particular the block-scaled
floating point version using nvfp4 values with e4m3 scalars. There are two variants
we can use, the only difference between the two is where the A-block is stored (TMEM or SMEM). 
For this implementation we choose the SMEM variant:

tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
                                        [d-tmem],  a-desc,  b-desc, idesc,
                                        [scale-A-tmem], [scale-B-tmem], enable-input-d;

.kind = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
.cta_group      = { .cta_group::1,   .cta_group::2 }
.scale_vectorsize = { .scale_vec::1X, .scale_vec::2X, .scale_vec::4X, .block16, .block32 }

Let each mma instruction compute an mBxnB segment of C with a depth
of kB (so each instruction computes the block-scaled matmul of an mBxkB segment of A with a kBxnB
segment of B). The execution of a single one of these mma ops requires us to:

-> Load mBxkB block of A and kBxnB block of B into SMEM
-> Load mBx(kB/16) block of SFA and (kB/16)xnB block of SFB into TMEM
-> Execute mma and accumulate into resultant mBxnB region in TMEM

There is the option of having 2 CTAs cooperate on a single mma instruction to widen mB from 128->256,
first we will focus on the single CTA version.

We will implement split-k and pipelining, but for our first implementation each CTA computes along all 
of k without pipelining. For a single CTA this is the algorithm flow:

-> Allocate SMEM (need room for A-block, B-block, SFA-block, SFB-block, C-block)
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

The size of the blocks loaded from GMEM->SMEM can be multiples of the mma blocks.

Data formatting:

The tcgen05 mma instructions require data be stored in SMEM/TMEM in particular formats. We look at this
now.

First let's see how to format the A and B blocks for a single MMA instruction:

We can see from the mma instruction that the location and format of the A/B blocks is determined by the
a-desc/b-desc arguments to the mma instruction. These are "matrix descriptors", which have the following format
and describe how the tile is layed out in SMEM:

[0-13]  : encode( matrix start address )
[16-29] : encode( LBO ) (Leading Byte Offset)
[32-45] : encode( SBO ) (Stride Byte Offset)
[46-48] : 0b001 (fixed constant)
[49-51] : Matrix base offset
[52]    : 0: byte offset relative ( other option is - 1: byte address absolute, not discussed yet here)
[53-60] : 0xb00000000 (fixed constant)
[61-63] : Swizzle (0. No swizzling 1. 128-Byte with 32B atomic swizzling 2. 128-Byte swizzling 4. 64-Byte swizzling 6. 32-Byte swizzling)

* encode(x) = (x & 0x3FFFF) >> 4 (keep the first 14 bits, then divide by 16)

In depth explanation for "LBO" and "SBO":
Before diving into what each field means we have to discuss the concept of a "core matrix". This isn't discussed anywhere in 
official NVIDIA docs as far as I can tell at the timing of writing, and (likely for that reason) LLMs hallucinate incorrect explanations 
for this concept. For some reason it's something that one needs to infer based on how WGMMA instructions worked on Hopper and the 
requirements of tcgen05 MMA instructions. 

This blog by Modular has an explanation that is pretty good, and where I sourced a large part of my understanding from:
https://www.modular.com/blog/matrix-multiplication-on-nvidias-blackwell-part-2-using-hardware-features-to-optimize-matmul

The idea is input tiles to the MMA instruction need to be formatted such that 8 row x 16B col chunks (called a core matrix)
are arranged in an acceptable "canonical layout" for the instruction (swizzle mode, K-major or MN-major). For example, a K-major
matrix is expected to store contiguous columns of core matrices within whole the matrix tile. So if our MMA instruction ingests a 
128x64 tile of NVFP4 (which can be broken down into 16 rows x 2 cols of core matrices), in shared memory we would store the 16 core
matrices in the first column of core matrices contgiously followed by the 16 in the second column (with the 16B rows within each
core matrix also being contiguous). 

Now LBO and SBO are much easier to understand: LBO is the number of bytes needed to reach the next column of core matrices (so, in our 
example above that would be 128*16B), hence the name "Leading" which is typically used to refer to the amount to stride to reach the
start of another dimension. SBO refers to the number of bytes needed to reach the next core matrix within a column (which as far as I
can tell should always be 8*16 = 128B).

"matrix start address" - This is the address of the start of the tile we are using for the MMA
"LBO" - Explained above
"SBO" - Explained above
"Matrix base offset" - I'm unclear on what this means, it has something to do with swizzle addressing
"Swizzle" - Which swizzle mode to use


Now let's look at how to store the tiles of SFA and SFB used to scale the tiles of A and B, stored in TMEM. TMEM is structured as 
128 rows (called lanes) and 512 columns where each row/column element is a 32b (4B) unit. The exact layout the MMA instruction 
expects depends on the dimensions of the MMA operation, but generally for 1B scale factors the m/n dimension is split up into chunks
of size 32 and stacked next to each other. For example, if the SFA tile is 128x4 (corresponding to 128x64 NVFP4 A tile) then there
would be 4 32x4 chunks layed out next each other in TMEM (occupying 32 lanes and 4 columns (each column holds 4 scale factors)). The
diagrams at https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-block-scaling demonstrate what I describe here,
specifically Figure 233.


Next we look at how the result of the MMA op is stored in TMEM. This is relatively straightforward (see diagram at 
https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d) in that each lane in TMEM holds a row of the output,
and each column in TMEM holds a column of the output put (unpacked, so if the output data type of the MMA is < 32b the upper unused bits
in each column just get ignored). So for row major MxN output tile it is stored in M rows (lanes) across N columns in TMEM. The caveat
which is demonstrated in the linked diagram is that a single warp can only access 32 lanes of TMEM, which means you need 4 warps to access
a M=128 row-major output.


Lastly we look at what i-desc contains (https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instruction-descriptor).

We will look at the specific format for NVFP4 (all type formats can be seen at that link). For NVFP4:

The important fields are...

[2]     : Sparsity (Dense = 0, Sparse = 1)
[4-5]   : Matrix B Scale Factor Data ID
[13]    : Negate A Matrix
[14]    : Negate B Matrix
[15]    : Transpose A Matrix
[16]    : Transpose B Matrix
[17-22] : N, Dimension of Matrix B (3 LSBs not included)
[23]    : Scale Matrix Type, for both scale_A / scale_B (UE4M3 = 0)
[27-28] : M, Dimension of Matrix A (7 LSBs not included)
[29-30] : Matrix A Scale Factor Data ID
[31]    : K Dimension (Dense K=64 / Sparse K=128) = 0)

"Sparsity" - MMAs can operate on sparse or dense matrix formats, this is the selector (0 = Dense)
"Matrix B Scale Factor Data ID" - Specifies the byte offset within each TMEM column where the scale factors are located to form the SF matrix.
                                  So for example if you are working with a format that uses 1 SF per row, the 32 row chunks are separated across
                                  separate TMEM columns (spaced apart by 4 "byte sized" columns). 
"Negate A Matrix" - Make A matrix elements negative (only supported for certain configs)
"Negate B Matrix" - Make B matrix elements negative (only supported for certain configs)
"Transpose A Matrix" - Transpose A then multiply (only supported for certain configs)
"Transpose B Matrix" - Transpose B then multiply (only supported for certain configs)
"N Dimension" - N in the KxN tile used for the MMA (only accepts certain values depending on data type and MMA format)
"Scale Matrix Type" - UE8M0 = 1, UE4M3 = 0
"M Dimension" - M in the MxK tile used for the MMA (only accepts certain values depending on data type and MMA format)
"Matrix A Scale Factor Data ID" - Same as for B, but for the A scale factor matrix
"K Dimension" - K in MxK and KxN tiles for the MMA (the exact value is determined by the other configuration inputs to the MMA)


Now that we understand how data should be formatted and stored, as well as how the MMA instruction works with that data, we can discuss 
how data will be moved. Specifically, how input data will be moved from GMEM->SMEM, SMEM->TMEM (for scale factors), and how output data 
will be moved from TMEM->Regs->SMEM->GMEM.

GMEM->SMEM:

For this we use TMA (Tensor Memory Accelerator), which is really a name for the family of cp.async.bulk instructions in PTX.
For out implementation we would like to copy a multi-dimensional tile so we will use the cp.async.bulk.tensor instructions. Specifically,
this version below allows us to copy from GMEM to DSMEM (Distributed Shared Memory of a CTA cluster). Copying to DSMEM allows us to use
what is referred to as "TMA Multi-Cast" which allows us to copy one tensor segment from GMEM to the SMEM of multiple CTAs within a CTA cluster:

// global -> shared::cluster
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}
                                   {, ctaMask} {, cache-policy}

.dst =                  { .shared::cluster }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }
.multicast =            { .multicast::cluster  }

We also need the variant which loads to a single CTA:

// global -> shared::cta
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache-policy}

.dst =                  { .shared::cta }
.src =                  { .global }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .mbarrier::complete_tx::bytes }
.cta_group =            { .cta_group::1, .cta_group::2 }
.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }
.level::cache_hint =    { .L2::cache_hint }


SMEM->TMEM:

For this we use the tcgen05.cp instruction

tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;

.cta_group = { .cta_group::1, .cta_group::2 }
.src_fmt   = { .b6x16_p32 , .b4x16_p64 }
.dst_fmt   = { .b8x16 }
.shape     = { .128x256b, .4x256b, .128x128b, .64x128b**, .32x128b*** }
.multicast = { .warpx2::02_13** , .warpx2::01_23**, .warpx4*** }


TMEM->Regs:

For this we use the tcgen05.ld instruction

tcgen05.ld.sync.aligned.shape1.num{.pack}.b32    r, [taddr];

.shape1 = { .16x64b, .16x128b, .16x256b, .32x32b }
.shape2 = { .16x32bx2 }
.num    = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 }
.pack   = { .pack::16b }


Regs->SMEM:

This can't be done asynchronously, so we just want to maximize coalescence and minimize bank conflicts
(this is where we could apply a function to the outputs like silu)


SMEM->GMEM:

TMA again, but this time we use the Shared -> GMEM variant:

// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
                                   [tensorMap, tensorCoords], [srcMem] {, cache-policy}

.dst =                  { .global }
.src =                  { .shared::cta }
.dim =                  { .1d, .2d, .3d, .4d, .5d }
.completion_mechanism = { .bulk_group }
.load_mode =            { .tile, .tile::scatter4, .im2col_no_offs }
.level::cache_hint =    { .L2::cache_hint }



Now that we understand how data will be moved and processed we need to ensure correctness (i.e. the data we
plan to use in one stage is available from the previous stage):

This is the overall data flow:

Stage:         0       1       2        3       4       5
Pipeline: GMEM -> SMEM -> TMEM, Compute -> Regs -> SMEM -> GMEM

To ensure the TMA has completed stage 0 before starting stage 1 we need to use an mbarrier object to ensure all of the
bytes have arrived in SMEM before we begin using them.

MMA instructions are guaranteed to be ordered after tcgen05.cp instructions if that's the order in which they were
written, so stage 1 should be guaranteed to finish before stage 2 starts. (We might need to be careful of loops here,
since I'm not sure if tcgen05.mma -> tcgen05.cp is guaranteed to be pipelined)

In order to ensure the last MMA has completed before we copy the result from TMEM to Regs we need to insert a memory fence
because tcgen05.mma completetion prior to tcgen05.ld isn't guaranteed (between stages 2 and 3).

(Is there anything we need to do to ensure stage 3 completes before stage 4 begins? Stage 4 should be guaranteed to complete
before stage 5 begins due to memory ordering of shared memory stores, but we might want to double check this as well.)







Finally we have all the components we need to assemble a detailed PTX level description of our algorithm:


Supported M_TILE_SIZE values: 128
Supported K_TILE_SIZE values: 64, 128, 256, 512, ...
Supported N_TILE_SIZE values: 64, 128

Improvements:
- Pipelining / warp specialization
- Consider making stages contiguous in SMEM
- CTA Cluster Implements:
    - Split-K
    - 2SM (M = 256)
    - Multi-cast

- Different epilogue (tcgen05.ld schemes)
- TMA for store
- Clean up Code
- Pipelining / warp specialization
- Dimension swap/transpose

- Look through and clean up issue comments
- Two alloc TMEM technique
- Leaner SF Loading?


--- Analysis ---


"""

@cute.kernel
def kernel(
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        tma_atom_sfa: cute.CopyAtom,
        tma_atom_sfb: cute.CopyAtom,
    ):

    """
    GMEM -> SMEM via TMA

    cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar_ptr, mcast_mask=mask)
    """

@cute.jit
def my_kernel(
        a_tensor: cute.Tensor,
        b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor,
        sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
    ):

def custom_kernel(data: input_t) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMM.
    """
    a_ref, b_ref, sfa_ref, sfb_ref, sfa_permuted, sfb_permuted, c_ref = data

    


    return c_ref

#!POPCORN leaderboard nvfp4_gemm
import torch
from torch.utils.cpp_extension import load_inline
from typing import List
from task import input_t, output_t
from utils import make_match_reference

from reference import generate_input

"""
Figure out input shapes 

M:  128
N:  1024
K:  512
-------
a_ref.dtype: torch.float4_e2m1fn_x2
a_ref.shape: torch.Size([128, 256, 1])
a_ref.stride(): (256, 1, 32768)


sfa_ref.dtype: torch.float8_e4m3fn
sfa_ref.shape: torch.Size([128, 32, 1])
sfa_ref.stride(): (32, 1, 4096)


b_ref.dtype: torch.float4_e2m1fn_x2
b_ref.shape: torch.Size([1024, 256, 1])
b_ref.stride(): (256, 1, 262144)


sfb_ref.dtype: torch.float8_e4m3fn
sfb_ref.shape: torch.Size([1024, 32, 1])
sfb_ref.stride(): (32, 1, 32768)


c_ref.dtype: torch.float16
c_ref.shape: torch.Size([128, 1024, 1])
c_ref.stride(): (1024, 1, 131072)

sfa_ref_perm.dtype: torch.float8_e4m3fn
sfa_ref_perm.shape: torch.Size([32, 4, 1, 4, 8, 1])
sfa_ref_perm.stride(): (16, 4, 4096, 1, 512, 4096)

(32, 4, rest_m, 4, rest_k, l) -> in order of increasing stide length -> (4, 4, 32, rest_k, rest_m, l)
mm32, mm4, mm, kk4, kk, b_grid -> (kk4, mm4, mm32, kk, mm, b_grid)
(0, 4) -> (0, 0, 0, 1, 0, 0) -> 512B offset
(127, 3) -> (3, 3, 31, 0, 0, 0) -> 511B offset 

sfb_ref_perm.dtype: torch.float8_e4m3fn
sfb_ref_perm.shape: torch.Size([32, 4, 8, 4, 8, 1])
sfb_ref_perm.stride(): (16, 4, 4096, 1, 512, 32768)


"""

def main(m, n, k):
    """
    PyTorch reference implementation of NVFP4 block-scaled GEMM.
    """
    a_ref, b_ref, sfa_ref, sfb_ref, sfa_ref_perm, sfb_ref_perm, c_ref = generate_input(m, n, k, 1, 1111, "cpu")

    print("M: ", m)
    print("N: ", n)
    print("K: ", k)
    print("-------")

    print("a_ref.dtype:", a_ref.dtype)
    print("a_ref.shape:", a_ref.shape)
    print("a_ref.stride():", a_ref.stride())
    print("\n")
    print("sfa_ref.dtype:", sfa_ref.dtype)
    print("sfa_ref.shape:", sfa_ref.shape)
    print("sfa_ref.stride():", sfa_ref.stride())
    print("\n")
    print("b_ref.dtype:", b_ref.dtype)
    print("b_ref.shape:", b_ref.shape)
    print("b_ref.stride():", b_ref.stride())
    print("\n")
    print("sfb_ref.dtype:", sfb_ref.dtype)
    print("sfb_ref.shape:", sfb_ref.shape)
    print("sfb_ref.stride():", sfb_ref.stride())
    print("\n")
    print("c_ref.dtype:", c_ref.dtype)
    print("c_ref.shape:", c_ref.shape)
    print("c_ref.stride():", c_ref.stride())

    print("sfa_ref_perm.dtype:", sfa_ref_perm.dtype)
    print("sfa_ref_perm.shape:", sfa_ref_perm.shape)
    print("sfa_ref_perm.stride():", sfa_ref_perm.stride())

    print("sfb_ref_perm.dtype:", sfb_ref_perm.dtype)
    print("sfb_ref_perm.shape:", sfb_ref_perm.shape)
    print("sfb_ref_perm.stride():", sfb_ref_perm.stride())

if __name__ == "__main__":
    main(128, 1024, 512)

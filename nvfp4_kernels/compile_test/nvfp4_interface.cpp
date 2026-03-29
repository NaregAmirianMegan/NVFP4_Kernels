#include <torch/extension.h>
//#include <ATen/ATen.h>
//#include <ATen/Float4_e2m1fn_x2.h>   
//#include <ATen/Float8_e4m3fn.h>

torch::Tensor batched_nvfp4_gemv(torch::Tensor a_ref, torch::Tensor b_ref, torch::Tensor sfa_ref, torch::Tensor sfb_ref, torch::Tensor c_ref, int m, int k, int l);
